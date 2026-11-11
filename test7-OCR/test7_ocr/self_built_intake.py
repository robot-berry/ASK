from __future__ import annotations

import base64
import binascii
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .storage import write_json_atomic


ROLES = {"clean_page", "photo_capture", "boundary_page", "handwriting_capture"}
ROLE_PREFIX = {
    "clean_page": "CLEAN",
    "photo_capture": "PHOTO",
    "boundary_page": "BOUNDARY",
    "handwriting_capture": "HAND",
}
SUBJECTS = {"mathematics", "physics"}
SCOPES = {"middle_school", "high_school"}
CONDITIONS = {
    "frontal",
    "perspective",
    "shadow",
    "fold_or_curve",
    "blur",
    "compression",
    "uneven_light",
    "screen_moire",
}
PAGE_LAYOUTS = {"not_applicable", "single_column", "double_column", "mixed"}
FORMAT_EXTENSIONS = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp", "TIFF": "tif"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SelfBuiltIntakeStore:
    """Quarantine-only, content-addressed intake for human-created evaluation data."""

    def __init__(self, dataset_root: Path, record_dir: Path, summary_path: Path) -> None:
        self.dataset_root = dataset_root.resolve()
        self.source_dir = (self.dataset_root / "sources").resolve()
        self.record_dir = record_dir.resolve()
        self.summary_path = summary_path.resolve()

    @staticmethod
    def _decode_image(encoded: object) -> tuple[bytes, str, int, int]:
        value = str(encoded or "")
        if value.startswith("data:"):
            marker = ";base64,"
            if marker not in value:
                raise ValueError("Uploaded image must use base64 encoding")
            value = value.split(marker, 1)[1]
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Uploaded image is not valid base64 data") from exc
        if not raw or len(raw) > 30_000_000:
            raise ValueError("Uploaded image is empty or exceeds the 30 MB intake limit")
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image_format = str(image.format or "").upper()
                if image_format not in FORMAT_EXTENSIONS:
                    raise ValueError("Only PNG, JPEG, WebP, and TIFF source images are accepted")
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Uploaded image cannot be decoded safely") from exc
        if width < 320 or height < 240:
            raise ValueError("Source image is too small for OCR evaluation")
        if width * height > 80_000_000:
            raise ValueError("Source image exceeds the 80-megapixel safety limit")
        return raw, image_format, int(width), int(height)

    @staticmethod
    def _clean_text(value: object, label: str, minimum: int = 2, maximum: int = 120) -> str:
        text = str(value or "").strip()
        if len(text) < minimum or len(text) > maximum or any(character in "\r\n\t" for character in text):
            raise ValueError(f"{label} must be {minimum}-{maximum} printable characters")
        return text

    def _record_path(self, asset_id: str) -> Path:
        if not asset_id.startswith("SBI-") or any(
            character not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-"
            for character in asset_id
        ):
            raise ValueError("Invalid self-built asset id")
        path = (self.record_dir / f"{asset_id}.json").resolve()
        if path.parent != self.record_dir:
            raise ValueError("Self-built intake record escaped its directory")
        return path

    def _source_path(self, relative_path: str) -> Path:
        path = (self.dataset_root / relative_path).resolve()
        if self.dataset_root not in path.parents or path.parent != self.source_dir:
            raise ValueError("Self-built source path escaped its content-addressed directory")
        return path

    def load(self, asset_id: str) -> dict[str, Any]:
        path = self._record_path(asset_id)
        if not path.is_file():
            raise FileNotFoundError(asset_id)
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("Self-built intake record must be a JSON object")
        self.validate(record, verify_file=True)
        return record

    def list(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.record_dir.is_dir():
            return rows
        for path in sorted(self.record_dir.glob("*.json")):
            record = self.load(path.stem)
            rows.append(
                {
                    "asset_id": record["asset_id"],
                    "role": record["role"],
                    "subject": record["metadata"]["subject"],
                    "scope": record["metadata"]["scope"],
                    "document_group": record["metadata"]["document_group"],
                    "page_index": record["metadata"]["page_index"],
                    "clean_asset_id": record["metadata"]["clean_asset_id"],
                    "device_code": record["metadata"]["device_code"],
                    "conditions": record["metadata"]["conditions"],
                    "page_layout": record["metadata"]["page_layout"],
                    "distribution_allowed": record["rights_and_privacy"]["distribution_allowed"],
                    "width": record["source"]["width"],
                    "height": record["source"]["height"],
                    "byte_size": record["source"]["byte_size"],
                    "sha256": record["source"]["sha256"],
                    "registered_at_utc": record["registered_at_utc"],
                }
            )
        rows.sort(key=lambda row: (row["registered_at_utc"], row["asset_id"]), reverse=True)
        return rows

    def source_path(self, asset_id: str) -> Path:
        record = self.load(asset_id)
        return self._source_path(record["source"]["relative_path"])

    def _clean_record(self, clean_asset_id: str) -> dict[str, Any]:
        record = self.load(clean_asset_id)
        if record["role"] != "clean_page":
            raise ValueError("Paired captures must reference a clean_page asset")
        return record

    def prepare_registration(
        self,
        metadata: dict[str, Any],
        encoded_image: object,
        clean_resolver: Callable[[str], dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], bytes]:
        """Validate an intake candidate without mutating the real registry."""
        if not isinstance(metadata, dict):
            raise ValueError("Intake metadata must be a JSON object")
        role = str(metadata.get("role") or "")
        subject = str(metadata.get("subject") or "")
        scope = str(metadata.get("scope") or "")
        if role not in ROLES:
            raise ValueError("Invalid self-built intake role")
        if subject not in SUBJECTS or scope not in SCOPES:
            raise ValueError("Subject and school scope are required")
        document_group = self._clean_text(metadata.get("document_group"), "document_group")
        try:
            page_index = int(metadata.get("page_index") or 1)
        except (TypeError, ValueError) as exc:
            raise ValueError("page_index must be a positive integer") from exc
        if page_index < 1:
            raise ValueError("page_index must be a positive integer")
        conditions = sorted({str(value) for value in metadata.get("conditions") or []})
        if not set(conditions) <= CONDITIONS:
            raise ValueError("Capture conditions contain an unsupported value")
        page_layout = str(metadata.get("page_layout") or "not_applicable")
        if page_layout not in PAGE_LAYOUTS:
            raise ValueError("Invalid page layout")
        clean_asset_id = str(metadata.get("clean_asset_id") or "").strip() or None
        device_code = str(metadata.get("device_code") or "").strip() or None

        if role in {"photo_capture", "handwriting_capture"}:
            if not clean_asset_id:
                raise ValueError("Photo and handwriting captures require a paired clean asset")
            clean = (
                clean_resolver(clean_asset_id)
                if clean_resolver is not None
                else self._clean_record(clean_asset_id)
            )
            if clean.get("role") != "clean_page":
                raise ValueError("Paired captures must reference a clean_page asset")
            if clean["metadata"]["document_group"] != document_group:
                raise ValueError("Paired clean page must belong to the same document_group")
            if clean["metadata"]["subject"] != subject or clean["metadata"]["scope"] != scope:
                raise ValueError("Paired clean page subject/scope does not match")
            page_index = int(clean["metadata"]["page_index"])
            device_code = self._clean_text(device_code, "device_code", 2, 80)
            if not conditions:
                raise ValueError("Photo and handwriting captures require at least one condition")
            page_layout = "not_applicable"
        else:
            if clean_asset_id is not None or device_code is not None or conditions:
                raise ValueError("Clean and boundary pages cannot declare capture pairing metadata")
        if role == "boundary_page" and page_layout not in {"single_column", "double_column", "mixed"}:
            raise ValueError("Boundary pages require an explicit page layout")
        if role != "boundary_page":
            page_layout = "not_applicable"

        provenance = str(metadata.get("provenance") or "")
        if provenance not in {"self_created", "licensed"}:
            raise ValueError("Source provenance must be self_created or licensed")
        license_reference = str(metadata.get("license_reference") or "").strip() or None
        if provenance == "licensed" and not license_reference:
            raise ValueError("Licensed sources require a license reference")
        if metadata.get("privacy_checked") is not True:
            raise ValueError("Privacy review must be explicitly completed before intake")
        if metadata.get("contains_personal_data") is not False:
            raise ValueError("Images containing personal data cannot enter this intake")
        distribution_allowed = metadata.get("distribution_allowed")
        if not isinstance(distribution_allowed, bool):
            raise ValueError("Distribution permission must be explicitly true or false")

        raw, image_format, width, height = self._decode_image(encoded_image)
        digest = hashlib.sha256(raw).hexdigest()
        extension = FORMAT_EXTENSIONS[image_format]
        relative_path = f"sources/{digest}.{extension}"
        asset_id = f"SBI-{ROLE_PREFIX[role]}-{digest[:16]}"
        record = {
            "schema_version": "self_built_intake_record_v1",
            "asset_id": asset_id,
            "role": role,
            "partition": "quarantine",
            "training_allowed": False,
            "evaluation_allowed": False,
            "source": {
                "dataset": "test7_ocr_self_built_v1",
                "relative_path": relative_path,
                "sha256": digest,
                "format": image_format,
                "width": width,
                "height": height,
                "byte_size": len(raw),
            },
            "metadata": {
                "subject": subject,
                "scope": scope,
                "document_group": document_group,
                "page_index": page_index,
                "clean_asset_id": clean_asset_id,
                "device_code": device_code,
                "conditions": conditions,
                "page_layout": page_layout,
            },
            "rights_and_privacy": {
                "provenance": provenance,
                "license_reference": license_reference,
                "privacy_checked": True,
                "contains_personal_data": False,
                "distribution_allowed": distribution_allowed,
            },
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.validate(record, verify_file=False)
        return record, raw

    def register(self, metadata: dict[str, Any], encoded_image: object) -> dict[str, Any]:
        record, raw = self.prepare_registration(metadata, encoded_image)
        asset_id = record["asset_id"]
        relative_path = record["source"]["relative_path"]
        record_path = self._record_path(asset_id)
        if record_path.is_file():
            existing = self.load(asset_id)
            comparable_existing = dict(existing)
            comparable_record = dict(record)
            comparable_existing.pop("registered_at_utc", None)
            comparable_record.pop("registered_at_utc", None)
            if comparable_existing != comparable_record:
                raise ValueError("This image is already registered with different metadata")
            return existing

        source_path = self._source_path(relative_path)
        self.source_dir.mkdir(parents=True, exist_ok=True)
        if not source_path.is_file():
            temporary = source_path.with_suffix(source_path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, source_path)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(record_path, record)
        self.write_summary()
        return record

    def validate(self, record: dict[str, Any], verify_file: bool = True) -> None:
        if record.get("schema_version") != "self_built_intake_record_v1":
            raise ValueError("Invalid self-built intake schema")
        role = record.get("role")
        if role not in ROLES:
            raise ValueError("Invalid self-built intake role")
        if record.get("partition") != "quarantine":
            raise ValueError("Unreviewed intake assets must remain quarantined")
        if record.get("training_allowed") is not False or record.get("evaluation_allowed") is not False:
            raise ValueError("Quarantined intake assets cannot be used by models or evaluation")
        source = record.get("source") or {}
        if source.get("dataset") != "test7_ocr_self_built_v1":
            raise ValueError("Intake asset references the wrong dataset")
        digest = str(source.get("sha256") or "")
        expected_id = f"SBI-{ROLE_PREFIX[role]}-{digest[:16]}"
        if record.get("asset_id") != expected_id:
            raise ValueError("Intake asset id does not match its role and source hash")
        expected_relative = f"sources/{digest}.{FORMAT_EXTENSIONS.get(source.get('format'), '')}"
        if source.get("relative_path") != expected_relative or len(digest) != 64:
            raise ValueError("Intake source path is not content-addressed")
        if (
            source.get("format") not in FORMAT_EXTENSIONS
            or not isinstance(source.get("width"), int)
            or not isinstance(source.get("height"), int)
            or not isinstance(source.get("byte_size"), int)
            or source["width"] < 1
            or source["height"] < 1
            or source["byte_size"] < 1
        ):
            raise ValueError("Invalid intake source format, dimensions, or size")
        metadata = record.get("metadata") or {}
        if metadata.get("subject") not in SUBJECTS or metadata.get("scope") not in SCOPES:
            raise ValueError("Invalid intake subject or scope")
        if metadata.get("page_layout") not in PAGE_LAYOUTS:
            raise ValueError("Invalid intake page layout")
        if not set(metadata.get("conditions") or []) <= CONDITIONS:
            raise ValueError("Invalid intake capture conditions")
        document_group = str(metadata.get("document_group") or "")
        if len(document_group) < 2 or len(document_group) > 120:
            raise ValueError("Invalid intake document_group")
        if not isinstance(metadata.get("page_index"), int) or metadata["page_index"] < 1:
            raise ValueError("Invalid intake page_index")
        if role in {"photo_capture", "handwriting_capture"}:
            if not metadata.get("clean_asset_id") or not metadata.get("device_code") or not metadata.get("conditions"):
                raise ValueError("Paired captures require clean asset, device, and conditions")
            if metadata.get("page_layout") != "not_applicable":
                raise ValueError("Paired captures cannot define a boundary layout")
        else:
            if metadata.get("clean_asset_id") is not None or metadata.get("device_code") is not None or metadata.get("conditions"):
                raise ValueError("Non-capture intake records contain invalid pairing metadata")
        if role == "boundary_page":
            if metadata.get("page_layout") not in {"single_column", "double_column", "mixed"}:
                raise ValueError("Boundary intake record lacks a valid layout")
        elif metadata.get("page_layout") != "not_applicable":
            raise ValueError("Only boundary pages may define a page layout")
        rights = record.get("rights_and_privacy") or {}
        if rights.get("privacy_checked") is not True or rights.get("contains_personal_data") is not False:
            raise ValueError("Intake privacy gate is not satisfied")
        if rights.get("provenance") not in {"self_created", "licensed"}:
            raise ValueError("Invalid intake provenance")
        if rights.get("provenance") == "licensed" and not rights.get("license_reference"):
            raise ValueError("Licensed intake source lacks a license reference")
        if not isinstance(rights.get("distribution_allowed"), bool):
            raise ValueError("Intake distribution permission is missing")
        if verify_file:
            source_path = self._source_path(str(source.get("relative_path") or ""))
            if not source_path.is_file() or source_path.stat().st_size != source.get("byte_size"):
                raise ValueError("Intake source file is missing or has changed size")
            if file_sha256(source_path) != digest:
                raise ValueError("Intake source SHA-256 mismatch")
            with Image.open(source_path) as image:
                if image.size != (source.get("width"), source.get("height")):
                    raise ValueError("Intake source dimensions changed")
                if str(image.format or "").upper() != source.get("format"):
                    raise ValueError("Intake source format changed")

    def summary(self) -> dict[str, Any]:
        items = self.list()
        roles = Counter(item["role"] for item in items)
        devices = {item["device_code"] for item in items if item.get("device_code")}
        condition_counts = Counter(
            condition for item in items for condition in item.get("conditions") or []
        )
        boundary_layouts = Counter(
            item["page_layout"] for item in items if item["role"] == "boundary_page"
        )
        clean_groups = {
            item["document_group"] for item in items if item["role"] == "clean_page"
        }
        return {
            "schema_version": "self_built_intake_summary_v1",
            "dataset": "test7_ocr_self_built_v1",
            "partition": "quarantine",
            "training_allowed": False,
            "evaluation_allowed": False,
            "counts": dict(sorted(roles.items())),
            "unique_clean_document_groups": len(clean_groups),
            "unique_devices": len(devices),
            "condition_counts": dict(sorted(condition_counts.items())),
            "boundary_layout_counts": dict(sorted(boundary_layouts.items())),
            "targets": {
                "clean_pages": {"current": roles["clean_page"], "minimum": 50},
                "photo_captures": {"current": roles["photo_capture"], "minimum": 200},
                "boundary_pages": {"current": roles["boundary_page"], "minimum": 60},
                "single_column_boundary_pages": {"current": boundary_layouts["single_column"], "minimum": 30},
                "double_column_boundary_pages": {"current": boundary_layouts["double_column"], "minimum": 30},
                "handwriting_captures": {"current": roles["handwriting_capture"], "minimum": 100},
                "devices": {"current": len(devices), "minimum": 3},
            },
            "note": "All newly registered assets remain quarantined and unavailable to training or evaluation until human annotation, rights review, split audit, and dataset freeze are complete.",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def write_summary(self) -> dict[str, Any]:
        summary = self.summary()
        write_json_atomic(self.summary_path, summary)
        return summary

    def check(self) -> tuple[dict[str, Any], list[str]]:
        errors: list[str] = []
        records: dict[str, dict[str, Any]] = {}
        if self.record_dir.is_dir():
            for path in sorted(self.record_dir.glob("*.json")):
                try:
                    record = self.load(path.stem)
                    records[record["asset_id"]] = record
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{path.name}: {exc}")
        for asset_id, record in records.items():
            clean_id = record["metadata"].get("clean_asset_id")
            if not clean_id:
                continue
            clean = records.get(clean_id)
            if clean is None:
                errors.append(f"{asset_id}: paired clean asset is missing")
            elif clean["role"] != "clean_page":
                errors.append(f"{asset_id}: paired asset is not a clean page")
            elif clean["metadata"]["document_group"] != record["metadata"]["document_group"]:
                errors.append(f"{asset_id}: paired document_group mismatch")
        return self.summary(), errors
