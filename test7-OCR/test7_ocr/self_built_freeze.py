from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from .capture_pair_review import CapturePairStore
from .handwriting_review import HandwritingMaskStore, SELF_BUILT_MASK_SCHEMA, file_sha256
from .question_boundary_review import QuestionBoundaryStore
from .self_built_intake import SelfBuiltIntakeStore


FREEZE_VERSION = re.compile(r"^test7-self-built-v[0-9]{4}\.[0-9]{2}\.[0-9]{2}(?:-[0-9]{2})?$")
DEFAULT_TARGETS = {
    "ocr_gold": 300,
    "clean_pages": 50,
    "photo_pairs": 200,
    "handwriting_pairs": 100,
    "handwriting_masks": 100,
    "boundary_pages": 60,
    "boundary_questions": 300,
    "single_column_pages": 30,
    "double_column_pages": 30,
    "devices": 3,
}
REQUIRED_CAPTURE_CONDITIONS = {"perspective", "shadow", "fold_or_curve"}
BLUR_OR_COMPRESSION = "blur_or_compression"


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json(row) + b"\n" for row in rows)


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    if not path.is_file():
        return rows, [f"missing JSONL file: {path}"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{path.name}:{line_number}: record must be an object")
            continue
        rows.append(row)
    return rows, errors


class SelfBuiltFreezeBuilder:
    """Audit and freeze reviewed evaluation data without copying source photographs."""

    def __init__(
        self,
        *,
        primary_dataset_root: Path,
        primary_dataset_name: str | None = None,
        ocr_gold_path: Path,
        intake: SelfBuiltIntakeStore,
        masks: HandwritingMaskStore,
        boundaries: QuestionBoundaryStore,
        capture_pairs: CapturePairStore,
        freeze_root: Path,
        targets: dict[str, int] | None = None,
        required_capture_conditions: set[str] | None = None,
    ) -> None:
        self.primary_dataset_root = primary_dataset_root.resolve()
        self.primary_dataset_name = str(primary_dataset_name or self.primary_dataset_root.name)
        self.ocr_gold_path = ocr_gold_path.resolve()
        self.intake = intake
        self.masks = masks
        self.boundaries = boundaries
        self.capture_pairs = capture_pairs
        self.freeze_root = freeze_root.resolve()
        self.targets = {**DEFAULT_TARGETS, **(targets or {})}
        self.required_capture_conditions = (
            set(required_capture_conditions)
            if required_capture_conditions is not None
            else REQUIRED_CAPTURE_CONDITIONS | {BLUR_OR_COMPRESSION}
        )

    @staticmethod
    def _version(value: str) -> str:
        if not FREEZE_VERSION.fullmatch(str(value or "")):
            raise ValueError(
                "Freeze version must look like test7-self-built-vYYYY.MM.DD or end in -NN"
            )
        return value

    def _primary_source(self, row: dict[str, Any]) -> Path:
        relative = str((row.get("source") or {}).get("image_path") or "")
        path = (self.primary_dataset_root / relative).resolve()
        if self.primary_dataset_root not in path.parents or not path.is_file():
            raise ValueError(f"{row.get('sample_id')}: OCR Gold source is missing or escaped its root")
        return path

    def _collect_ocr_gold(self, errors: list[str]) -> list[dict[str, Any]]:
        rows, read_errors = _read_jsonl(self.ocr_gold_path)
        errors.extend(read_errors)
        seen: set[str] = set()
        for row in rows:
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in seen:
                errors.append(f"OCR Gold has a missing or duplicate sample_id: {sample_id or '<empty>'}")
                continue
            seen.add(sample_id)
            source = row.get("source") or {}
            review = row.get("review") or {}
            if row.get("schema_version") != "ocr_gold_v1":
                errors.append(f"{sample_id}: invalid OCR Gold schema")
            if row.get("partition") != "test":
                errors.append(f"{sample_id}: OCR Gold must remain in the test split")
            if source.get("dataset") != self.primary_dataset_name:
                errors.append(f"{sample_id}: OCR Gold references the wrong source dataset")
            if source.get("training_allowed") is not False or source.get("evaluation_only") is not True:
                errors.append(f"{sample_id}: OCR Gold usage policy changed")
            human_verified = (
                review.get("status") == "verified"
                and int(review.get("independent_review_count") or 0) >= 2
                and int(review.get("required_independent_review_count") or 2) == 2
                and review.get("source_alignment") == "human_verified"
            )
            source_generated_verified = (
                review.get("status") == "source_generated_verified"
                and int(review.get("independent_review_count", -1)) == 0
                and int(review.get("required_independent_review_count", -1)) == 0
                and review.get("source_alignment") == "generator_fact_source"
                and (row.get("handwriting") or {}).get("gold_status")
                == "synthetic_clean_source"
            )
            if not human_verified and not source_generated_verified:
                errors.append(f"{sample_id}: OCR Gold has no accepted verification basis")
            try:
                source_path = self._primary_source(row)
                if file_sha256(source_path) != source.get("image_sha256"):
                    errors.append(f"{sample_id}: OCR Gold source SHA-256 mismatch")
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        return sorted(rows, key=lambda row: str(row.get("sample_id") or ""))

    def audit(self, *, require_targets: bool) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
        errors: list[str] = []
        intake_summary, intake_errors = self.intake.check()
        pair_summary, pair_errors = self.capture_pairs.check()
        boundary_summary, boundary_errors = self.boundaries.check()
        errors.extend(f"intake: {error}" for error in intake_errors)
        errors.extend(f"capture pair: {error}" for error in pair_errors)
        errors.extend(f"question boundary: {error}" for error in boundary_errors)

        ocr_gold = self._collect_ocr_gold(errors)
        pair_tasks = [
            self.capture_pairs.load(item["capture_asset_id"])
            for item in self.capture_pairs.list()
            if item["status"] == "verified"
        ]
        boundary_tasks = [
            self.boundaries.load(item["asset_id"])
            for item in self.boundaries.list()
            if item["status"] == "verified"
        ]
        mask_tasks: list[dict[str, Any]] = []
        for item in self.masks.list():
            try:
                task = self.masks.load(item["sample_id"])
                if task["annotation"]["status"] != "verified":
                    continue
                self.masks.source_path(item["sample_id"])
                self.masks.mask_path(item["sample_id"])
                mask_tasks.append(task)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"handwriting mask {item['sample_id']}: {exc}")

        selected_asset_ids: set[str] = set()
        verified_capture_ids = {task["capture_asset_id"] for task in pair_tasks}
        verified_handwriting_capture_ids = {
            task["capture_asset_id"]
            for task in pair_tasks
            if task["capture"]["role"] == "handwriting_capture"
        }
        for task in pair_tasks:
            selected_asset_ids.update({task["clean"]["asset_id"], task["capture_asset_id"]})
            if task["clean"]["sha256"] == task["capture"]["sha256"]:
                errors.append(
                    f"capture pair {task['capture_asset_id']}: clean and photographed bytes are identical"
                )
        selected_asset_ids.update(task["asset_id"] for task in boundary_tasks)
        for task in mask_tasks:
            if task.get("schema_version") != SELF_BUILT_MASK_SCHEMA:
                continue
            sample_id = task["sample_id"]
            selected_asset_ids.update({sample_id, task["source"]["clean_asset_id"]})
            if sample_id not in verified_handwriting_capture_ids:
                errors.append(
                    f"handwriting mask {sample_id}: source capture has no verified clean-page pair"
                )

        intake_records: list[dict[str, Any]] = []
        for asset_id in sorted(selected_asset_ids):
            try:
                record = self.intake.load(asset_id)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"selected intake {asset_id}: {exc}")
                continue
            rights = record.get("rights_and_privacy") or {}
            if rights.get("privacy_checked") is not True or rights.get("contains_personal_data") is not False:
                errors.append(f"{asset_id}: privacy audit is incomplete")
            if rights.get("provenance") not in {"self_created", "licensed"}:
                errors.append(f"{asset_id}: source provenance is invalid")
            if rights.get("provenance") == "licensed" and not rights.get("license_reference"):
                errors.append(f"{asset_id}: licensed source has no license reference")
            if not isinstance(rights.get("distribution_allowed"), bool):
                errors.append(f"{asset_id}: distribution permission was not explicitly recorded")
            intake_records.append(record)

        primary_hashes = {
            str((row.get("source") or {}).get("image_sha256") or "") for row in ocr_gold
        } - {""}
        self_built_hashes = {record["source"]["sha256"] for record in intake_records}
        overlap = sorted(primary_hashes & self_built_hashes)
        if overlap:
            errors.append(
                f"cross-dataset leakage: {len(overlap)} source hashes occur in both OCR Gold and self-built intake"
            )

        groups_by_hash: dict[str, set[str]] = {}
        for record in intake_records:
            groups_by_hash.setdefault(record["source"]["sha256"], set()).add(
                record["metadata"]["document_group"]
            )
        leaking_hashes = [digest for digest, groups in groups_by_hash.items() if len(groups) > 1]
        if leaking_hashes:
            errors.append(
                f"document-group leakage: {len(leaking_hashes)} source hashes belong to multiple groups"
            )

        role_counts = Counter(task["capture"]["role"] for task in pair_tasks)
        clean_pages = {task["clean"]["asset_id"] for task in pair_tasks}
        devices = {task["capture"]["device_code"] for task in pair_tasks}
        conditions = {
            condition for task in pair_tasks for condition in task["capture"].get("conditions") or []
        }
        layouts = Counter(task["source"]["page_layout"] for task in boundary_tasks)
        boundary_questions = sum(len(task["annotation"]["questions"]) for task in boundary_tasks)
        verified_self_built_masks = sum(
            task.get("schema_version") == SELF_BUILT_MASK_SCHEMA for task in mask_tasks
        )
        counts = {
            "ocr_gold": len(ocr_gold),
            "clean_pages": len(clean_pages),
            "photo_pairs": role_counts["photo_capture"],
            "handwriting_pairs": role_counts["handwriting_capture"],
            "handwriting_masks": len(mask_tasks),
            "self_built_handwriting_masks": verified_self_built_masks,
            "boundary_pages": len(boundary_tasks),
            "boundary_questions": boundary_questions,
            "single_column_pages": layouts["single_column"],
            "double_column_pages": layouts["double_column"],
            "devices": len(devices),
        }
        if require_targets:
            for key, minimum in self.targets.items():
                if counts.get(key, 0) < minimum:
                    errors.append(f"freeze target {key}: {counts.get(key, 0)}/{minimum}")
            ordinary_required = self.required_capture_conditions - {BLUR_OR_COMPRESSION}
            missing_conditions = sorted(ordinary_required - conditions)
            if missing_conditions:
                errors.append(
                    "freeze target capture conditions missing: " + ", ".join(missing_conditions)
                )
            if BLUR_OR_COMPRESSION in self.required_capture_conditions and not (
                {"blur", "compression"} & conditions
            ):
                errors.append("freeze target capture conditions missing: blur_or_compression")

        payload = {
            "ocr_gold": ocr_gold,
            "intake_sources": sorted(intake_records, key=lambda row: row["asset_id"]),
            "capture_pairs": sorted(pair_tasks, key=lambda row: row["capture_asset_id"]),
            "question_boundaries": sorted(boundary_tasks, key=lambda row: row["asset_id"]),
            "handwriting_masks": sorted(mask_tasks, key=lambda row: row["sample_id"]),
        }
        summary = {
            "schema_version": "self_built_freeze_readiness_v1",
            "ready_to_freeze": not errors and require_targets,
            "require_targets": require_targets,
            "partition_after_freeze": "frozen_test",
            "training_allowed": False,
            "evaluation_allowed_after_freeze": True,
            "source_images_copied": False,
            "counts": counts,
            "targets": dict(self.targets),
            "verified_capture_ids": len(verified_capture_ids),
            "selected_intake_assets": len(intake_records),
            "intake_summary": intake_summary,
            "capture_pair_summary": pair_summary,
            "question_boundary_summary": boundary_summary,
            "error_count": len(errors),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return summary, errors, payload

    def freeze(self, version: str) -> dict[str, Any]:
        version = self._version(version)
        summary, errors, payload = self.audit(require_targets=True)
        if errors:
            raise ValueError("Dataset is not ready to freeze:\n- " + "\n- ".join(errors))
        destination = (self.freeze_root / version).resolve()
        if destination.parent != self.freeze_root:
            raise ValueError("Freeze destination escaped its root")
        if destination.exists():
            raise FileExistsError(f"Immutable freeze version already exists: {destination}")
        self.freeze_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{version}-", dir=self.freeze_root) as temporary:
            staging = Path(temporary)
            payload_files: dict[str, bytes] = {
                "ocr_gold.jsonl": _jsonl_bytes(payload["ocr_gold"]),
                "intake_sources.jsonl": _jsonl_bytes(payload["intake_sources"]),
                "capture_pairs.jsonl": _jsonl_bytes(payload["capture_pairs"]),
                "question_boundaries.jsonl": _jsonl_bytes(payload["question_boundaries"]),
                "handwriting_masks.jsonl": _jsonl_bytes(payload["handwriting_masks"]),
            }
            for relative, content in payload_files.items():
                _write_bytes_atomic(staging / relative, content)

            copied_masks: set[str] = set()
            for task in payload["handwriting_masks"]:
                source_mask = self.masks.mask_path(task["sample_id"])
                digest = task["annotation"]["mask_sha256"]
                relative = f"masks/{digest}.png"
                if relative in copied_masks:
                    continue
                copied_masks.add(relative)
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_mask, target)
                if file_sha256(target) != digest:
                    raise ValueError(f"Copied mask hash mismatch: {task['sample_id']}")

            files = {}
            for path in sorted(item for item in staging.rglob("*") if item.is_file()):
                relative = path.relative_to(staging).as_posix()
                files[relative] = {"sha256": file_sha256(path), "byte_size": path.stat().st_size}
            content_fingerprint = hashlib.sha256(_canonical_json(files)).hexdigest()
            manifest = {
                "schema_version": "self_built_freeze_manifest_v1",
                "dataset_version": version,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "partition": "frozen_test",
                "training_allowed": False,
                "evaluation_allowed": True,
                "source_images_copied": False,
                "source_reference_policy": "dataset_name_relative_path_and_sha256",
                "content_fingerprint": content_fingerprint,
                "summary": summary,
                "files": files,
            }
            _write_bytes_atomic(
                staging / "freeze_manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
            )
            os.replace(staging, destination)
        verified, verify_errors = self.verify_freeze(destination)
        if verify_errors:
            raise ValueError("New freeze failed self-verification: " + "; ".join(verify_errors))
        return verified

    @staticmethod
    def verify_freeze(directory: Path) -> tuple[dict[str, Any], list[str]]:
        directory = directory.resolve()
        manifest_path = directory / "freeze_manifest.json"
        if not manifest_path.is_file():
            return {}, ["freeze_manifest.json is missing"]
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {}, [f"freeze manifest cannot be read: {exc}"]
        errors: list[str] = []
        if manifest.get("schema_version") != "self_built_freeze_manifest_v1":
            errors.append("invalid freeze manifest schema")
        if manifest.get("source_images_copied") is not False:
            errors.append("freeze must not contain copied source photographs")
        files = manifest.get("files") or {}
        if not isinstance(files, dict):
            return manifest, ["freeze file index is invalid"]
        for relative, expected in files.items():
            path = (directory / str(relative)).resolve()
            if directory not in path.parents or not path.is_file():
                errors.append(f"missing frozen payload: {relative}")
                continue
            if file_sha256(path) != (expected or {}).get("sha256"):
                errors.append(f"frozen payload hash mismatch: {relative}")
            if path.stat().st_size != (expected or {}).get("byte_size"):
                errors.append(f"frozen payload size mismatch: {relative}")
        actual_payloads = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file() and path.name != "freeze_manifest.json"
        }
        if actual_payloads != set(files):
            errors.append("freeze contains unindexed or stale payload files")
        fingerprint = hashlib.sha256(_canonical_json(files)).hexdigest()
        if fingerprint != manifest.get("content_fingerprint"):
            errors.append("freeze content fingerprint mismatch")
        return manifest, errors
