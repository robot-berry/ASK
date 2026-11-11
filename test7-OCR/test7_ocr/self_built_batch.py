from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable

from .self_built_intake import SelfBuiltIntakeStore
from .storage import write_json_atomic


BATCH_HEADERS = [
    "row_id",
    "image_path",
    "role",
    "subject",
    "scope",
    "document_group",
    "page_index",
    "clean_row_id",
    "clean_asset_id",
    "device_code",
    "conditions",
    "page_layout",
    "provenance",
    "license_reference",
    "privacy_checked",
    "contains_personal_data",
    "distribution_allowed",
]
ROW_ID = re.compile(r"^[0-9A-Za-z._-]{1,80}$")


@dataclass(frozen=True)
class BatchRowPlan:
    row_number: int
    row_id: str
    image_path: Path
    metadata: dict[str, Any]
    asset_id: str
    image_sha256: str
    already_registered: bool


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: object, field: str) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be exactly true or false")


def _data_url(path: Path) -> str:
    raw = path.read_bytes()
    return "data:application/octet-stream;base64," + base64.b64encode(raw).decode("ascii")


def _comparable(record: dict[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value.pop("registered_at_utc", None)
    return value


class SelfBuiltBatchImporter:
    """Two-pass CSV intake: validate the complete batch before any real registration."""

    def __init__(
        self,
        store: SelfBuiltIntakeStore,
        report_dir: Path,
        synchronizers: list[Callable[[], object]] | None = None,
    ) -> None:
        self.store = store
        self.report_dir = report_dir.resolve()
        self.synchronizers = list(synchronizers or [])

    @staticmethod
    def _manifest_rows(manifest_path: Path) -> tuple[list[dict[str, str]], list[str]]:
        errors: list[str] = []
        try:
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                headers = list(reader.fieldnames or [])
                if headers != BATCH_HEADERS:
                    return [], [
                        "CSV headers must exactly match the documented template: "
                        + ",".join(BATCH_HEADERS)
                    ]
                rows = [dict(row) for row in reader]
        except (OSError, UnicodeError, csv.Error) as exc:
            return [], [f"CSV cannot be read: {exc}"]
        if not rows:
            errors.append("CSV contains no intake rows")
        if len(rows) > 1000:
            errors.append("A single batch cannot exceed 1000 rows")
        return rows, errors

    @staticmethod
    def _image_path(batch_root: Path, value: object) -> Path:
        relative = str(value or "").strip().replace("\\", "/")
        candidate = Path(relative)
        if not relative or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("image_path must be a safe path relative to the CSV directory")
        path = (batch_root / candidate).resolve()
        if batch_root not in path.parents or not path.is_file():
            raise ValueError("image_path is missing or escapes the batch directory")
        if path.stat().st_size <= 0 or path.stat().st_size > 30_000_000:
            raise ValueError("image file is empty or exceeds 30 MB")
        return path

    @staticmethod
    def _metadata(row: dict[str, str]) -> dict[str, Any]:
        try:
            page_index = int(str(row.get("page_index") or "").strip())
        except ValueError as exc:
            raise ValueError("page_index must be a positive integer") from exc
        conditions = [
            item.strip()
            for item in str(row.get("conditions") or "").replace(";", "|").split("|")
            if item.strip()
        ]
        return {
            "role": str(row.get("role") or "").strip(),
            "subject": str(row.get("subject") or "").strip(),
            "scope": str(row.get("scope") or "").strip(),
            "document_group": str(row.get("document_group") or "").strip(),
            "page_index": page_index,
            "clean_asset_id": str(row.get("clean_asset_id") or "").strip() or None,
            "device_code": str(row.get("device_code") or "").strip() or None,
            "conditions": conditions,
            "page_layout": str(row.get("page_layout") or "not_applicable").strip(),
            "provenance": str(row.get("provenance") or "").strip(),
            "license_reference": str(row.get("license_reference") or "").strip() or None,
            "privacy_checked": _parse_bool(row.get("privacy_checked"), "privacy_checked"),
            "contains_personal_data": _parse_bool(
                row.get("contains_personal_data"), "contains_personal_data"
            ),
            "distribution_allowed": _parse_bool(
                row.get("distribution_allowed"), "distribution_allowed"
            ),
        }

    def plan(
        self, manifest_path: Path
    ) -> tuple[list[BatchRowPlan], list[dict[str, Any]], int]:
        manifest_path = manifest_path.resolve()
        rows, manifest_errors = self._manifest_rows(manifest_path)
        failures: list[dict[str, Any]] = [
            {"row_number": 1, "row_id": None, "error": error} for error in manifest_errors
        ]
        if manifest_errors:
            return [], failures, len(rows)
        batch_root = manifest_path.parent.resolve()
        parsed: list[dict[str, Any]] = []
        seen_row_ids: set[str] = set()
        for index, row in enumerate(rows, start=2):
            row_id = str(row.get("row_id") or "").strip()
            try:
                if not ROW_ID.fullmatch(row_id) or row_id in seen_row_ids:
                    raise ValueError("row_id is missing, invalid, or duplicated")
                seen_row_ids.add(row_id)
                image_path = self._image_path(batch_root, row.get("image_path"))
                metadata = self._metadata(row)
                clean_row_id = str(row.get("clean_row_id") or "").strip() or None
                if metadata["role"] in {"photo_capture", "handwriting_capture"}:
                    if bool(clean_row_id) == bool(metadata["clean_asset_id"]):
                        raise ValueError(
                            "capture rows require exactly one of clean_row_id or clean_asset_id"
                        )
                elif clean_row_id or metadata["clean_asset_id"]:
                    raise ValueError("non-capture rows cannot declare a clean-page reference")
                parsed.append(
                    {
                        "row_number": index,
                        "row_id": row_id,
                        "image_path": image_path,
                        "metadata": metadata,
                        "clean_row_id": clean_row_id,
                    }
                )
            except (OSError, ValueError) as exc:
                failures.append({"row_number": index, "row_id": row_id or None, "error": str(exc)})
        if failures:
            return [], failures, len(rows)

        by_row_id = {item["row_id"]: item for item in parsed}
        planned_records: dict[str, dict[str, Any]] = {}
        planned_assets: set[str] = set()
        planned_hashes: set[str] = set()
        plans: list[BatchRowPlan] = []

        def resolve_clean(asset_id: str) -> dict[str, Any]:
            if asset_id in planned_records:
                return planned_records[asset_id]
            return self.store.load(asset_id)

        ordered = sorted(
            parsed,
            key=lambda item: (
                0 if item["metadata"]["role"] == "clean_page" else 1,
                item["row_number"],
            ),
        )
        for item in ordered:
            row_id = item["row_id"]
            try:
                metadata = dict(item["metadata"])
                clean_row_id = item["clean_row_id"]
                if clean_row_id:
                    clean_item = by_row_id.get(clean_row_id)
                    if clean_item is None or clean_item["metadata"]["role"] != "clean_page":
                        raise ValueError("clean_row_id must reference a clean_page row in this CSV")
                    clean_matches = [plan for plan in plans if plan.row_id == clean_row_id]
                    if not clean_matches:
                        raise ValueError("referenced clean row failed or was not planned")
                    metadata["clean_asset_id"] = clean_matches[0].asset_id
                record, _raw = self.store.prepare_registration(
                    metadata, _data_url(item["image_path"]), resolve_clean
                )
                asset_id = record["asset_id"]
                image_sha256 = record["source"]["sha256"]
                if asset_id in planned_assets or image_sha256 in planned_hashes:
                    raise ValueError("multiple CSV rows use identical source image bytes")
                planned_assets.add(asset_id)
                planned_hashes.add(image_sha256)
                planned_records[asset_id] = record
                already_registered = False
                try:
                    existing = self.store.load(asset_id)
                except FileNotFoundError:
                    pass
                else:
                    already_registered = True
                    if _comparable(existing) != _comparable(record):
                        raise ValueError("asset is already registered with different metadata")
                plans.append(
                    BatchRowPlan(
                        row_number=item["row_number"],
                        row_id=row_id,
                        image_path=item["image_path"],
                        metadata=metadata,
                        asset_id=asset_id,
                        image_sha256=image_sha256,
                        already_registered=already_registered,
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(
                    {"row_number": item["row_number"], "row_id": row_id, "error": str(exc)}
                )
        if failures:
            return [], sorted(failures, key=lambda item: item["row_number"]), len(rows)
        plans.sort(key=lambda item: item.row_number)
        return plans, [], len(rows)

    def run(self, manifest_path: Path, *, apply: bool) -> dict[str, Any]:
        manifest_path = manifest_path.resolve()
        manifest_sha256 = _file_sha256(manifest_path) if manifest_path.is_file() else None
        plans, failures, row_count = self.plan(manifest_path)
        now = datetime.now(timezone.utc)
        batch_id = (
            f"BATCH-{now.strftime('%Y%m%d-%H%M%S-%f')}-"
            f"{(manifest_sha256 or 'missing')[:10]}"
        )
        row_results: list[dict[str, Any]] = []
        runtime_failures: list[dict[str, Any]] = []
        status = "rejected_preflight" if failures else "ready"
        if not failures:
            for plan in plans:
                row_results.append(
                    {
                        "row_number": plan.row_number,
                        "row_id": plan.row_id,
                        "role": plan.metadata["role"],
                        "asset_id": plan.asset_id,
                        "image_sha256": plan.image_sha256,
                        "status": "already_registered" if plan.already_registered else "ready",
                    }
                )
        if apply and not failures:
            changed: list[BatchRowPlan] = []
            for plan in plans:
                try:
                    unchanged = _file_sha256(plan.image_path) == plan.image_sha256
                except OSError:
                    unchanged = False
                if not unchanged:
                    changed.append(plan)
            if changed:
                status = "rejected_changed_files"
                runtime_failures = [
                    {
                        "row_number": plan.row_number,
                        "row_id": plan.row_id,
                        "error": "image changed after preflight; no rows were registered",
                    }
                    for plan in changed
                ]
            else:
                status = "applied"
                row_results = []
            ordered_for_registration = sorted(
                plans,
                key=lambda plan: (
                    0 if plan.metadata["role"] == "clean_page" else 1,
                    plan.row_number,
                ),
            )
            for plan in ordered_for_registration if status == "applied" else []:
                try:
                    record = self.store.register(plan.metadata, _data_url(plan.image_path))
                    if record["asset_id"] != plan.asset_id:
                        raise ValueError("content-addressed asset id changed after preflight")
                    row_results.append(
                        {
                            "row_number": plan.row_number,
                            "row_id": plan.row_id,
                            "role": plan.metadata["role"],
                            "asset_id": plan.asset_id,
                            "image_sha256": plan.image_sha256,
                            "status": "already_registered" if plan.already_registered else "registered",
                        }
                    )
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    status = "partial_failure"
                    row_results.append(
                        {
                            "row_number": plan.row_number,
                            "row_id": plan.row_id,
                            "role": plan.metadata["role"],
                            "asset_id": plan.asset_id,
                            "image_sha256": plan.image_sha256,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    break
            if row_results:
                for synchronize in self.synchronizers:
                    try:
                        synchronize()
                    except Exception as exc:  # synchronization is reported, never hidden
                        status = "partial_failure"
                        runtime_failures.append(
                            {
                                "row_number": None,
                                "row_id": None,
                                "error": f"post-intake synchronization failed: {exc}",
                            }
                        )
                        break
            row_results.sort(key=lambda item: item["row_number"])

        report = {
            "schema_version": "self_built_batch_report_v1",
            "batch_id": batch_id,
            "mode": "apply" if apply else "dry_run",
            "status": status,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "rows_in_manifest": row_count,
            "ready_rows": len(plans),
            "failure_count": len(failures)
            + len(runtime_failures)
            + sum(item.get("status") == "failed" for item in row_results),
            "failures": failures + runtime_failures,
            "rows": row_results,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if apply:
            self.report_dir.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.report_dir / f"{batch_id}.json", report)
        return report
