from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from typing import Any

from PIL import Image, UnidentifiedImageError

from .collection_plan import (
    COLLECTION_TARGETS,
    build_collection_plan_rows,
    collection_plan_csv,
)
from .self_built_batch import BATCH_HEADERS
from .self_built_intake import FORMAT_EXTENSIONS, SelfBuiltIntakeStore
from .storage import write_json_atomic


COLLECTION_WORKSPACE_VERSION = "self_built_collection_workspace_v1"
PLAN_FILENAME = "self_built_collection_plan.csv"
READINESS_FILENAME = "collection_readiness.json"
MISSING_FILENAME = "collection_missing_files.csv"
READY_BATCH_FILENAME = "collection_ready_batch.csv"
REGISTRATION_MAP_FILENAME = "collection_registration_map.json"
MAX_IMAGE_BYTES = 30_000_000


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_image_path(workspace_root: Path, value: object) -> Path:
    relative = str(value or "").strip().replace("\\", "/")
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("image_path must be a safe relative path")
    resolved = (workspace_root / candidate).resolve()
    if workspace_root not in resolved.parents:
        raise ValueError("image_path escapes the private collection workspace")
    return resolved


def _read_working_plan(plan_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    if not plan_path.is_file():
        return [], [f"working collection plan is missing: {plan_path}"]
    try:
        with plan_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = list(reader.fieldnames or [])
            if headers != BATCH_HEADERS:
                return [], ["working collection plan headers do not match the batch importer"]
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], [f"working collection plan cannot be read: {exc}"]
    if len(rows) != sum(COLLECTION_TARGETS.values()):
        errors.append("working collection plan must retain exactly 410 rows")
    row_ids = [str(row.get("row_id") or "").strip() for row in rows]
    if not all(row_ids) or len(set(row_ids)) != len(row_ids):
        errors.append("working collection plan row_id values must be non-empty and unique")
    role_counts = Counter(str(row.get("role") or "").strip() for row in rows)
    if dict(role_counts) != COLLECTION_TARGETS:
        errors.append("working collection plan no longer satisfies the four role quotas")
    clean_ids = {
        str(row.get("row_id") or "").strip()
        for row in rows
        if str(row.get("role") or "").strip() == "clean_page"
    }
    for row in rows:
        role = str(row.get("role") or "").strip()
        clean_row_id = str(row.get("clean_row_id") or "").strip()
        clean_asset_id = str(row.get("clean_asset_id") or "").strip()
        if role in {"photo_capture", "handwriting_capture"}:
            if bool(clean_row_id) == bool(clean_asset_id):
                errors.append(
                    f"{row.get('row_id') or '<unknown>'}: capture requires exactly one clean reference"
                )
            elif clean_row_id and clean_row_id not in clean_ids:
                errors.append(
                    f"{row.get('row_id') or '<unknown>'}: clean_row_id is not a clean_page row"
                )
        elif clean_row_id or clean_asset_id:
            errors.append(
                f"{row.get('row_id') or '<unknown>'}: non-capture row cannot declare a clean reference"
            )
    return rows, errors


def _registration_map(workspace_root: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    path = workspace_root / REGISTRATION_MAP_FILENAME
    if not path.is_file():
        return {}, []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {}, [f"guided registration map cannot be read: {exc}"]
    if not isinstance(payload, dict) or payload.get("schema_version") != "collection_registration_map_v1":
        return {}, ["guided registration map has an unsupported schema"]
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, dict):
        return {}, ["guided registration map rows must be an object"]
    rows: dict[str, dict[str, str]] = {}
    for row_id, value in raw_rows.items():
        if not isinstance(row_id, str) or not isinstance(value, dict):
            return {}, ["guided registration map contains an invalid row"]
        asset_id = str(value.get("asset_id") or "")
        digest = str(value.get("source_sha256") or "")
        storage_mode = str(value.get("storage_mode") or "")
        if (
            not asset_id.startswith("SBI-")
            or len(digest) != 64
            or storage_mode not in {"hardlink", "copy", "existing"}
        ):
            return {}, [f"guided registration map contains invalid evidence for {row_id}"]
        rows[row_id] = {
            "asset_id": asset_id,
            "source_sha256": digest,
            "storage_mode": storage_mode,
        }
    return rows, []


def _write_working_plan(path: Path, rows: list[dict[str, str]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=BATCH_HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\ufeff" + stream.getvalue(), encoding="utf-8")
    os.replace(temporary, path)


def _guided_row_payload(row: dict[str, str]) -> dict[str, Any]:
    conditions = [
        item.strip()
        for item in str(row.get("conditions") or "").replace(";", "|").split("|")
        if item.strip()
    ]
    return {
        "row_id": str(row.get("row_id") or ""),
        "image_path": str(row.get("image_path") or ""),
        "role": str(row.get("role") or ""),
        "subject": str(row.get("subject") or ""),
        "scope": str(row.get("scope") or ""),
        "document_group": str(row.get("document_group") or ""),
        "page_index": int(str(row.get("page_index") or "1")),
        "clean_row_id": str(row.get("clean_row_id") or "") or None,
        "device_code": str(row.get("device_code") or "") or None,
        "conditions": conditions,
        "page_layout": str(row.get("page_layout") or "not_applicable"),
        "provenance": str(row.get("provenance") or "self_created"),
    }


def inspect_collection_workspace(workspace_root: Path) -> dict[str, Any]:
    root = workspace_root.resolve()
    plan_path = root / PLAN_FILENAME
    rows, structural_errors = _read_working_plan(plan_path)
    registration_rows, registration_errors = _registration_map(root)
    structural_errors.extend(registration_errors)
    expected = collection_plan_csv()
    try:
        working_bytes = plan_path.read_bytes()
    except OSError:
        working_bytes = b""

    present_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, str]] = []
    invalid_rows: list[dict[str, str]] = []
    privacy_complete = 0
    personal_data_clear = 0
    distribution_declared = 0
    valid_images = 0
    valid_row_ids: set[str] = set()
    declaration_ready_ids: set[str] = set()
    role_present: Counter[str] = Counter()

    for row in rows:
        row_id = str(row.get("row_id") or "").strip()
        role = str(row.get("role") or "").strip()
        relative = str(row.get("image_path") or "").strip()
        privacy_value = str(row.get("privacy_checked") or "").strip().lower()
        personal_value = str(row.get("contains_personal_data") or "").strip().lower()
        distribution_value = str(row.get("distribution_allowed") or "").strip().lower()
        privacy_complete += privacy_value == "true"
        personal_data_clear += personal_value == "false"
        distribution_declared += distribution_value in {"true", "false"}
        if (
            privacy_value == "true"
            and personal_value == "false"
            and distribution_value in {"true", "false"}
        ):
            declaration_ready_ids.add(row_id)
        descriptor = {
            "row_id": row_id,
            "role": role,
            "subject": str(row.get("subject") or "").strip(),
            "image_path": relative,
        }
        try:
            path = _safe_image_path(root, relative)
        except ValueError as exc:
            invalid_rows.append({**descriptor, "reason": str(exc)})
            continue
        if not path.is_file():
            missing_rows.append({**descriptor, "reason": "missing"})
            continue
        try:
            byte_size = path.stat().st_size
            if byte_size < 1 or byte_size > MAX_IMAGE_BYTES:
                raise ValueError("image is empty or exceeds the 30 MB intake limit")
            with Image.open(path) as image:
                image_format = str(image.format or "").upper()
                if image_format not in FORMAT_EXTENSIONS:
                    raise ValueError("image must be PNG, JPEG, WebP, or TIFF")
                width, height = image.size
                image.verify()
            if width < 1 or height < 1:
                raise ValueError("image has invalid dimensions")
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            invalid_rows.append({**descriptor, "reason": str(exc)})
            continue
        present_rows.append(descriptor)
        valid_images += 1
        valid_row_ids.add(row_id)
        role_present[role] += 1

    locally_ready_ids = valid_row_ids & declaration_ready_ids
    incremental_ready_rows: list[dict[str, str]] = []
    blocked_by_clean_dependency: list[dict[str, str]] = []
    for row in rows:
        row_id = str(row.get("row_id") or "").strip()
        if row_id not in locally_ready_ids:
            continue
        role = str(row.get("role") or "").strip()
        clean_row_id = str(row.get("clean_row_id") or "").strip()
        if (
            role in {"photo_capture", "handwriting_capture"}
            and clean_row_id
            and clean_row_id not in locally_ready_ids
        ):
            blocked_by_clean_dependency.append(
                {
                    "row_id": row_id,
                    "role": role,
                    "image_path": str(row.get("image_path") or "").strip(),
                    "required_clean_row_id": clean_row_id,
                }
            )
            continue
        incremental_ready_rows.append(row)
    ready_role_counts = Counter(row["role"] for row in incremental_ready_rows)
    available_guided_rows: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("row_id") or "").strip()
        if row_id in valid_row_ids:
            continue
        role = str(row.get("role") or "").strip()
        clean_row_id = str(row.get("clean_row_id") or "").strip()
        if role in {"photo_capture", "handwriting_capture"} and clean_row_id:
            if clean_row_id not in registration_rows:
                continue
        available_guided_rows.append(_guided_row_payload(row))
    next_guided_row = available_guided_rows[0] if available_guided_rows else None

    planned_rows = len(rows)
    ready_for_import_preflight = bool(
        planned_rows == sum(COLLECTION_TARGETS.values())
        and not structural_errors
        and not missing_rows
        and not invalid_rows
        and valid_images == planned_rows
        and privacy_complete == planned_rows
        and personal_data_clear == planned_rows
        and distribution_declared == planned_rows
        and len(incremental_ready_rows) == planned_rows
    )
    next_actions: list[str] = []
    if structural_errors:
        next_actions.append("修复工作清单结构；系统不会覆盖已经人工编辑的 CSV。")
    if missing_rows:
        next_actions.append(f"按清单路径放入 {len(missing_rows)} 张尚缺的真实图片。")
    if invalid_rows:
        next_actions.append(f"修复 {len(invalid_rows)} 张路径、格式、尺寸或大小异常的图片。")
    if privacy_complete < planned_rows:
        next_actions.append(
            f"逐图完成隐私检查并把 privacy_checked=true（还缺 {planned_rows - privacy_complete} 行）。"
        )
    if personal_data_clear < planned_rows:
        next_actions.append(
            "逐图确认不含姓名、手机号、人脸等个人信息，并把 contains_personal_data=false。"
        )
    if blocked_by_clean_dependency:
        next_actions.append(
            f"先补齐 {len(blocked_by_clean_dependency)} 行照片所依赖的干净原稿。"
        )
    if incremental_ready_rows and not ready_for_import_preflight:
        next_actions.append(
            f"已有 {len(incremental_ready_rows)} 行可用 collection_ready_batch.csv 做增量只读预检。"
        )
    if ready_for_import_preflight:
        next_actions.append("运行批量登记器的只读预检；预检通过后再显式执行 --apply。")

    return {
        "schema_version": COLLECTION_WORKSPACE_VERSION,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(root),
        "working_plan": str(plan_path),
        "readiness_report": str(root / READINESS_FILENAME),
        "missing_files_report": str(root / MISSING_FILENAME),
        "ready_batch_path": str(root / READY_BATCH_FILENAME),
        "working_plan_sha256": _sha256_bytes(working_bytes) if working_bytes else None,
        "template_sha256": _sha256_bytes(expected),
        "template_unchanged": working_bytes == expected,
        "working_copy_preserved": True,
        "planned_rows": planned_rows,
        "files_present": len(present_rows),
        "files_missing": len(missing_rows),
        "files_invalid": len(invalid_rows),
        "valid_images": valid_images,
        "privacy_checked": privacy_complete,
        "personal_data_clear": personal_data_clear,
        "distribution_declared": distribution_declared,
        "ready_batch_rows": len(incremental_ready_rows),
        "guided_registration_count": len(registration_rows),
        "next_guided_row": next_guided_row,
        "available_guided_rows": available_guided_rows,
        "ready_batch_role_counts": {
            role: ready_role_counts[role] for role in COLLECTION_TARGETS
        },
        "ready_for_incremental_preflight": bool(
            incremental_ready_rows and not structural_errors
        ),
        "role_progress": {
            role: {"present": role_present[role], "target": target}
            for role, target in COLLECTION_TARGETS.items()
        },
        "structural_errors": structural_errors,
        "missing_rows": missing_rows,
        "invalid_rows": invalid_rows,
        "blocked_by_clean_dependency": blocked_by_clean_dependency,
        "incremental_ready_rows": incremental_ready_rows,
        "ready_for_import_preflight": ready_for_import_preflight,
        "training_allowed": False,
        "evaluation_allowed": False,
        "source_images_enter_git": False,
        "next_actions": next_actions,
    }


def _write_missing_csv(path: Path, rows: list[dict[str, str]]) -> None:
    stream = io.StringIO(newline="")
    fields = ["row_id", "role", "subject", "image_path", "reason"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\ufeff" + stream.getvalue(), encoding="utf-8")
    os.replace(temporary, path)


def _write_ready_batch_csv(path: Path, rows: list[dict[str, str]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=BATCH_HEADERS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\ufeff" + stream.getvalue(), encoding="utf-8")
    os.replace(temporary, path)


def register_guided_collection_image(
    workspace_root: Path,
    intake_store: SelfBuiltIntakeStore,
    *,
    row_id: str,
    encoded_image: object,
    privacy_checked: object,
    contains_personal_data: object,
    distribution_allowed: object,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Register one real image against the fixed plan without duplicating bytes when possible."""

    root = workspace_root.resolve()
    prepare_collection_workspace(root)
    plan_path = root / PLAN_FILENAME
    rows, errors = _read_working_plan(plan_path)
    registration_rows, registration_errors = _registration_map(root)
    errors.extend(registration_errors)
    if errors:
        raise ValueError("; ".join(errors))
    matches = [row for row in rows if str(row.get("row_id") or "") == row_id]
    if len(matches) != 1:
        raise ValueError("The guided collection row does not exist or is duplicated")
    row = matches[0]
    if privacy_checked is not True:
        raise ValueError("Privacy review must be explicitly completed")
    if contains_personal_data is not False:
        raise ValueError("Images containing personal data cannot enter the collection")
    if not isinstance(distribution_allowed, bool):
        raise ValueError("Distribution permission must be explicitly true or false")

    role = str(row.get("role") or "")
    clean_asset_id = None
    clean_row_id = str(row.get("clean_row_id") or "").strip()
    if role in {"photo_capture", "handwriting_capture"}:
        clean_evidence = registration_rows.get(clean_row_id)
        if clean_evidence is None:
            raise ValueError(f"Upload the paired clean row first: {clean_row_id}")
        clean_asset_id = clean_evidence["asset_id"]
    guided = _guided_row_payload(row)
    metadata = {
        "role": role,
        "subject": guided["subject"],
        "scope": guided["scope"],
        "document_group": guided["document_group"],
        "page_index": guided["page_index"],
        "clean_asset_id": clean_asset_id,
        "device_code": guided["device_code"],
        "conditions": guided["conditions"],
        "page_layout": guided["page_layout"],
        "provenance": guided["provenance"],
        "license_reference": str(row.get("license_reference") or "").strip() or None,
        "privacy_checked": True,
        "contains_personal_data": False,
        "distribution_allowed": distribution_allowed,
    }
    prepared, _raw = intake_store.prepare_registration(metadata, encoded_image)
    existing_mapping = registration_rows.get(row_id)
    if existing_mapping and existing_mapping["asset_id"] != prepared["asset_id"]:
        raise ValueError("This plan row is already registered with different image bytes")
    record = intake_store.register(metadata, encoded_image)
    source_path = intake_store.source_path(record["asset_id"])
    planned_path = _safe_image_path(root, row.get("image_path"))
    expected_suffix = "." + FORMAT_EXTENSIONS[str(record["source"]["format"])]
    if planned_path.suffix.lower() != expected_suffix:
        planned_path = planned_path.with_suffix(expected_suffix).resolve()
        if root not in planned_path.parents:
            raise ValueError("Normalized collection image path escaped the workspace")

    storage_mode = "existing"
    if planned_path.is_file():
        if _sha256_bytes(planned_path.read_bytes()) != record["source"]["sha256"]:
            raise ValueError("The planned image path already contains different bytes")
    else:
        planned_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source_path, planned_path)
            storage_mode = "hardlink"
        except OSError:
            shutil.copy2(source_path, planned_path)
            storage_mode = "copy"

    row["image_path"] = planned_path.relative_to(root).as_posix()
    row["privacy_checked"] = "true"
    row["contains_personal_data"] = "false"
    row["distribution_allowed"] = "true" if distribution_allowed else "false"
    _write_working_plan(plan_path, rows)
    registration_rows[row_id] = {
        "asset_id": record["asset_id"],
        "source_sha256": record["source"]["sha256"],
        "storage_mode": storage_mode,
    }
    write_json_atomic(
        root / REGISTRATION_MAP_FILENAME,
        {
            "schema_version": "collection_registration_map_v1",
            "rows": dict(sorted(registration_rows.items())),
        },
    )
    report = prepare_collection_workspace(root)
    return record, report, {
        **guided,
        "image_path": row["image_path"],
        "asset_id": record["asset_id"],
        "storage_mode": storage_mode,
    }


def prepare_collection_workspace(workspace_root: Path) -> dict[str, Any]:
    """Create the private collection area without overwriting a collector's edits."""
    root = workspace_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / PLAN_FILENAME
    plan_created = False
    if not plan_path.exists():
        temporary = plan_path.with_suffix(plan_path.suffix + ".tmp")
        temporary.write_bytes(collection_plan_csv())
        os.replace(temporary, plan_path)
        plan_created = True

    for row in build_collection_plan_rows():
        _safe_image_path(root, row["image_path"]).parent.mkdir(parents=True, exist_ok=True)

    report = inspect_collection_workspace(root)
    report["working_plan_created_this_run"] = plan_created
    write_json_atomic(root / READINESS_FILENAME, report)
    _write_missing_csv(
        root / MISSING_FILENAME,
        [*report["missing_rows"], *report["invalid_rows"]],
    )
    _write_ready_batch_csv(root / READY_BATCH_FILENAME, report["incremental_ready_rows"])
    return report
