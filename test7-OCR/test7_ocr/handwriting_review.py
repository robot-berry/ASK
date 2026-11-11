from __future__ import annotations

import base64
import binascii
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from .self_built_intake import SelfBuiltIntakeStore
from .storage import write_json_atomic


MASK_CLASS_VALUES = {
    "background": 0,
    "printed_content": 1,
    "student_handwriting": 2,
    "teacher_mark": 3,
    "overlap_unrecoverable": 4,
    "uncertain": 255,
}
MASK_CHECK_KEYS = {
    "source_hash_verified",
    "printed_content_preserved",
    "student_handwriting_complete",
    "teacher_marks_classified",
    "overlap_and_uncertain_checked",
}
ALLOWED_MASK_VALUES = set(MASK_CLASS_VALUES.values())
PRIMARY_MASK_DATASET = "testzip_math_physics_300_v1"
SELF_BUILT_MASK_DATASET = "test7_ocr_self_built_v1"
SELF_BUILT_MASK_SCHEMA = "handwriting_mask_review_v2"


def payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HandwritingMaskStore:
    """Hash-bound storage and two-person sign-off for frozen pixel-mask Gold."""

    def __init__(
        self,
        task_dir: Path,
        mask_dir: Path,
        dataset_root: Path,
        intake: SelfBuiltIntakeStore | None = None,
    ) -> None:
        self.task_dir = task_dir.resolve()
        self.mask_dir = mask_dir.resolve()
        self.annotation_root = self.mask_dir.parent
        self.dataset_root = dataset_root.resolve()
        self.intake = intake
        self.dataset_roots = {PRIMARY_MASK_DATASET: self.dataset_root}
        if intake is not None:
            self.dataset_roots[SELF_BUILT_MASK_DATASET] = intake.dataset_root

    @staticmethod
    def _self_built_task(record: dict[str, Any]) -> dict[str, Any]:
        source = record["source"]
        metadata = record["metadata"]
        return {
            "schema_version": SELF_BUILT_MASK_SCHEMA,
            "sample_id": record["asset_id"],
            "partition": "quarantine",
            "training_allowed": False,
            "evaluation_allowed": False,
            "source": {
                "dataset": source["dataset"],
                "image_path": source["relative_path"],
                "image_sha256": source["sha256"],
                "width": source["width"],
                "height": source["height"],
                "subject": metadata["subject"],
                "scope": metadata["scope"],
                "document_group": metadata["document_group"],
                "page_index": metadata["page_index"],
                "asset_id": record["asset_id"],
                "clean_asset_id": metadata["clean_asset_id"],
            },
            "selection_evidence": [
                {
                    "source": "self_built_intake",
                    "kind": "handwriting_capture",
                    "note": "已通过隐私和来源登记的真实带手写照片；仍需人工像素标注和独立复核。",
                }
            ],
            "annotation": {
                "status": "pending",
                "mask_path": None,
                "mask_sha256": None,
                "legend_version": "handwriting_mask_v1",
                "class_values": dict(MASK_CLASS_VALUES),
                "pixel_counts": {key: 0 for key in MASK_CLASS_VALUES},
            },
            "review": {
                "annotator_id": None,
                "reviewer_id": None,
                "checks": {key: None for key in sorted(MASK_CHECK_KEYS)},
                "notes": [],
                "signoffs": [],
                "history": [],
                "updated_at_utc": None,
            },
        }

    def sync_from_intake(self) -> int:
        """Create quarantined mask tasks for every registered handwriting capture."""
        if self.intake is None:
            return 0
        created = 0
        self.task_dir.mkdir(parents=True, exist_ok=True)
        for item in self.intake.list():
            if item["role"] != "handwriting_capture":
                continue
            record = self.intake.load(item["asset_id"])
            expected = self._self_built_task(record)
            path = self._task_path(item["asset_id"])
            if path.is_file():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if self._immutable_projection(existing) != self._immutable_projection(expected):
                    raise ValueError(f"{item['asset_id']}: frozen self-built mask source changed")
                if existing.get("selection_evidence") != expected["selection_evidence"]:
                    raise ValueError(f"{item['asset_id']}: self-built mask evidence changed")
                continue
            write_json_atomic(path, expected)
            created += 1
        return created

    @staticmethod
    def _validate_id(sample_id: str) -> str:
        if not sample_id or any(
            character not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz._-"
            for character in sample_id
        ):
            raise ValueError("Invalid handwriting-mask sample id")
        return sample_id

    def _task_path(self, sample_id: str) -> Path:
        self._validate_id(sample_id)
        path = (self.task_dir / f"{sample_id}.json").resolve()
        if path.parent != self.task_dir:
            raise ValueError("Handwriting-mask task path escaped its directory")
        return path

    def load(self, sample_id: str) -> dict[str, Any]:
        path = self._task_path(sample_id)
        if not path.is_file():
            raise FileNotFoundError(sample_id)
        task = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(task, dict):
            raise ValueError("Handwriting-mask task must be a JSON object")
        self.validate(task)
        if task.get("schema_version") == SELF_BUILT_MASK_SCHEMA:
            if self.intake is None:
                raise ValueError("Self-built handwriting-mask task has no intake registry")
            record = self.intake.load(sample_id)
            expected = self._self_built_task(record)
            if self._immutable_projection(task) != self._immutable_projection(expected):
                raise ValueError("Self-built handwriting-mask source no longer matches intake")
            if task.get("selection_evidence") != expected["selection_evidence"]:
                raise ValueError("Self-built handwriting-mask evidence no longer matches intake")
        return task

    def list(self) -> list[dict[str, Any]]:
        self.sync_from_intake()
        items: list[dict[str, Any]] = []
        if not self.task_dir.is_dir():
            return items
        for path in sorted(self.task_dir.glob("*.json")):
            task = self.load(path.stem)
            annotation = task["annotation"]
            review = task["review"]
            items.append(
                {
                    "sample_id": task["sample_id"],
                    "dataset": task["source"]["dataset"],
                    "partition": task["partition"],
                    "subject": task["source"]["subject"],
                    "status": annotation["status"],
                    "has_mask": annotation.get("mask_path") is not None,
                    "mask_sha256": annotation.get("mask_sha256"),
                    "annotator_id": review.get("annotator_id"),
                    "reviewer_id": review.get("reviewer_id"),
                    "evidence_count": len(task.get("selection_evidence") or []),
                }
            )
        return items

    def source_path(self, sample_id: str) -> Path:
        task = self.load(sample_id)
        source = task["source"]
        dataset_root = self.dataset_roots.get(str(source.get("dataset") or ""))
        if dataset_root is None:
            raise ValueError("Handwriting-mask source dataset is not registered")
        path = (dataset_root / source["image_path"]).resolve()
        if dataset_root not in path.parents or not path.is_file():
            raise FileNotFoundError("Handwriting-mask source image is missing")
        if file_sha256(path) != source["image_sha256"]:
            raise ValueError("Handwriting-mask source image hash mismatch")
        with Image.open(path) as image:
            if image.size != (source["width"], source["height"]):
                raise ValueError("Handwriting-mask source dimensions changed")
        return path

    def mask_path(self, sample_id: str) -> Path:
        task = self.load(sample_id)
        relative = task["annotation"].get("mask_path")
        if not relative:
            raise FileNotFoundError("No handwriting mask has been saved")
        path = (self.annotation_root / relative).resolve()
        if self.annotation_root not in path.parents or not path.is_file():
            raise FileNotFoundError("Handwriting mask is missing")
        if file_sha256(path) != task["annotation"].get("mask_sha256"):
            raise ValueError("Handwriting mask hash mismatch")
        return path

    @staticmethod
    def _pixel_counts(image: Image.Image) -> dict[str, int]:
        histogram = image.histogram()
        return {key: int(histogram[value]) for key, value in MASK_CLASS_VALUES.items()}

    def _decode_mask(self, encoded: object, source: dict[str, Any]) -> tuple[Image.Image, bytes]:
        value = str(encoded or "")
        prefix = "data:image/png;base64,"
        if value.startswith(prefix):
            value = value[len(prefix) :]
        try:
            raw = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Mask is not valid base64 PNG data") from exc
        if not raw or len(raw) > 12_000_000:
            raise ValueError("Mask PNG is empty or too large")
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.format != "PNG":
                    raise ValueError("Mask must be encoded as PNG")
                if opened.size != (int(source["width"]), int(source["height"])):
                    raise ValueError("Mask dimensions must exactly match the source image")
                rgba = opened.convert("RGBA")
                pixels = list(rgba.getdata())
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Mask PNG cannot be decoded safely") from exc
        values: list[int] = []
        for red, green, blue, alpha in pixels:
            if alpha != 255 or red != green or red != blue:
                raise ValueError("Mask PNG must contain opaque grayscale class values")
            if red not in ALLOWED_MASK_VALUES:
                raise ValueError(f"Mask contains invalid class value {red}")
            values.append(red)
        image = Image.new("L", rgba.size)
        image.putdata(values)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
        return image, output.getvalue()

    def _write_content_addressed_mask(self, sample_id: str, png: bytes) -> tuple[str, str]:
        digest = hashlib.sha256(png).hexdigest()
        filename = f"{sample_id}.{digest}.png"
        path = (self.mask_dir / filename).resolve()
        if path.parent != self.mask_dir:
            raise ValueError("Handwriting-mask output path escaped its directory")
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        if not path.is_file():
            temporary = path.with_suffix(".png.tmp")
            with temporary.open("wb") as handle:
                handle.write(png)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        return path.relative_to(self.annotation_root).as_posix(), digest

    @staticmethod
    def _valid_actor(actor_id: object) -> str:
        actor = str(actor_id or "").strip()
        if len(actor) < 2 or len(actor) > 80 or any(character.isspace() for character in actor):
            raise ValueError("Operator id must be 2-80 non-whitespace characters")
        return actor

    @staticmethod
    def _matching_signoff(task: dict[str, Any], role: str, action: str) -> bool:
        mask_sha = task["annotation"].get("mask_sha256")
        return any(
            isinstance(item, dict)
            and item.get("role") == role
            and item.get("action") == action
            and item.get("mask_sha256") == mask_sha
            for item in task["review"].get("signoffs") or []
        )

    @staticmethod
    def _immutable_projection(task: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": task.get("schema_version"),
            "sample_id": task.get("sample_id"),
            "partition": task.get("partition"),
            "training_allowed": task.get("training_allowed"),
            "evaluation_allowed": task.get("evaluation_allowed"),
            "source": task.get("source"),
            "selection_evidence": task.get("selection_evidence"),
            "legend_version": (task.get("annotation") or {}).get("legend_version"),
            "class_values": (task.get("annotation") or {}).get("class_values"),
        }

    def save(
        self,
        sample_id: str,
        updated: dict[str, Any],
        mask_png_base64: object,
        expected_hash: str | None,
    ) -> dict[str, Any]:
        existing = self.load(sample_id)
        if expected_hash and expected_hash != payload_hash(existing):
            raise RuntimeError("Mask task changed since it was opened; reload before saving")
        if not isinstance(updated, dict):
            raise ValueError("Mask task must be a JSON object")
        if self._immutable_projection(updated) != self._immutable_projection(existing):
            raise ValueError("Frozen mask source, evidence, and class definitions are immutable")

        existing_annotation = existing["annotation"]
        updated_annotation = updated.get("annotation") or {}
        existing_review = existing["review"]
        updated_review = updated.get("review") or {}
        for field in ("status", "mask_path", "mask_sha256", "pixel_counts"):
            if updated_annotation.get(field) != existing_annotation.get(field):
                raise ValueError(f"annotation.{field} is server-owned")
        for field in (
            "annotator_id",
            "reviewer_id",
            "signoffs",
            "history",
            "updated_at_utc",
        ):
            if updated_review.get(field) != existing_review.get(field):
                raise ValueError(f"review.{field} is server-owned")

        updated_review["notes"] = [
            str(note).strip() for note in updated_review.get("notes") or [] if str(note).strip()
        ]
        updated_review["signoffs"] = list(existing_review.get("signoffs") or [])
        updated_review["history"] = list(existing_review.get("history") or [])
        updated_review["annotator_id"] = existing_review.get("annotator_id")
        updated_review["reviewer_id"] = existing_review.get("reviewer_id")

        mask_changed = False
        if mask_png_base64 is not None:
            image, png = self._decode_mask(mask_png_base64, existing["source"])
            relative, digest = self._write_content_addressed_mask(sample_id, png)
            mask_changed = digest != existing_annotation.get("mask_sha256")
            updated_annotation["mask_path"] = relative
            updated_annotation["mask_sha256"] = digest
            updated_annotation["pixel_counts"] = self._pixel_counts(image)
        else:
            updated_annotation["mask_path"] = existing_annotation.get("mask_path")
            updated_annotation["mask_sha256"] = existing_annotation.get("mask_sha256")
            updated_annotation["pixel_counts"] = dict(existing_annotation.get("pixel_counts") or {})

        previous_status = existing_annotation.get("status")
        if mask_changed:
            updated_review["annotator_id"] = None
            updated_review["reviewer_id"] = None
            updated_annotation["status"] = "draft"
            updated_review["history"].append(
                {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "from_status": previous_status,
                    "to_status": "draft",
                    "actor_id": None,
                    "action": "mask_changed_signoffs_invalidated",
                }
            )
        else:
            updated_annotation["status"] = previous_status
        if updated_annotation.get("mask_path") and updated_annotation["status"] == "pending":
            updated_annotation["status"] = "draft"
        updated_review["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        updated["annotation"] = updated_annotation
        updated["review"] = updated_review
        self.validate(updated)
        write_json_atomic(self._task_path(sample_id), updated)
        return updated

    def signoff(
        self,
        sample_id: str,
        action: str,
        actor_id: object,
        expected_hash: str | None,
    ) -> dict[str, Any]:
        task = self.load(sample_id)
        if expected_hash and expected_hash != payload_hash(task):
            raise RuntimeError("Mask task changed since it was opened; reload before signing")
        actor = self._valid_actor(actor_id)
        annotation = task["annotation"]
        review = task["review"]
        previous_status = annotation.get("status")
        role: str
        if action == "submit_annotation":
            if previous_status not in {"pending", "draft"}:
                raise ValueError("Only a pending or draft mask can be submitted")
            if not annotation.get("mask_sha256"):
                raise ValueError("Save a pixel mask before annotator submission")
            foreground = sum(
                int(annotation["pixel_counts"].get(key, 0))
                for key in ("student_handwriting", "teacher_mark", "overlap_unrecoverable")
            )
            if foreground <= 0:
                raise ValueError("The selected sample requires at least one handwriting or mark pixel")
            role = "annotator"
            review["annotator_id"] = actor
            review["reviewer_id"] = None
            annotation["status"] = "annotated"
        elif action == "approve_review":
            if previous_status != "annotated" or not review.get("annotator_id"):
                raise ValueError("A reviewer can sign only after annotator submission")
            if actor == review.get("annotator_id"):
                raise ValueError("Reviewer must be distinct from annotator")
            if not self._matching_signoff(task, "annotator", "submit_annotation"):
                raise ValueError("Current mask has no matching annotator sign-off")
            checks = review.get("checks") or {}
            if set(checks) != MASK_CHECK_KEYS or any(value is not True for value in checks.values()):
                raise ValueError("Every mask review check must be true before approval")
            role = "reviewer"
            review["reviewer_id"] = actor
            annotation["status"] = "verified"
        elif action == "request_changes":
            if previous_status not in {"annotated", "verified"}:
                raise ValueError("Changes can be requested only for a submitted mask")
            if actor == review.get("annotator_id"):
                raise ValueError("Reviewer requesting changes must be distinct from annotator")
            if not review.get("notes"):
                raise ValueError("Add a problem note before requesting changes")
            role = "reviewer"
            review["annotator_id"] = None
            review["reviewer_id"] = None
            annotation["status"] = "draft"
        else:
            raise ValueError("Unknown handwriting-mask sign-off action")

        now = datetime.now(timezone.utc).isoformat()
        review.setdefault("signoffs", []).append(
            {
                "at_utc": now,
                "action": action,
                "role": role,
                "actor_id": actor,
                "mask_sha256": annotation.get("mask_sha256"),
                "checks": dict(review.get("checks") or {}),
            }
        )
        review.setdefault("history", []).append(
            {
                "at_utc": now,
                "from_status": previous_status,
                "to_status": annotation["status"],
                "actor_id": actor,
                "action": action,
            }
        )
        review["updated_at_utc"] = now
        self.validate(task)
        write_json_atomic(self._task_path(sample_id), task)
        return task

    def progress_csv(self) -> bytes:
        stream = io.StringIO(newline="")
        fields = [
            "sample_id",
            "subject",
            "status",
            "mask_sha256",
            "annotator_id",
            "reviewer_id",
            "student_handwriting_pixels",
            "teacher_mark_pixels",
            "overlap_unrecoverable_pixels",
            "updated_at_utc",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in self.list():
            task = self.load(item["sample_id"])
            pixels = task["annotation"]["pixel_counts"]
            writer.writerow(
                {
                    **{key: item.get(key) for key in fields if key in item},
                    "student_handwriting_pixels": pixels["student_handwriting"],
                    "teacher_mark_pixels": pixels["teacher_mark"],
                    "overlap_unrecoverable_pixels": pixels["overlap_unrecoverable"],
                    "updated_at_utc": task["review"].get("updated_at_utc"),
                }
            )
        return ("\ufeff" + stream.getvalue()).encode("utf-8")

    def validate(self, task: dict[str, Any]) -> None:
        schema_version = task.get("schema_version")
        if schema_version not in {"handwriting_mask_review_v1", SELF_BUILT_MASK_SCHEMA}:
            raise ValueError("Invalid handwriting-mask schema version")
        if task.get("training_allowed") is not False:
            raise ValueError("Handwriting-mask tasks cannot be used for training")
        source = task.get("source") or {}
        if schema_version == "handwriting_mask_review_v1":
            if task.get("partition") != "frozen_test" or source.get("dataset") != PRIMARY_MASK_DATASET:
                raise ValueError("Legacy handwriting-mask tasks must remain frozen evaluation-only data")
            if "evaluation_allowed" in task:
                raise ValueError("Legacy handwriting-mask task contains an unexpected usage field")
        else:
            if task.get("partition") != "quarantine" or task.get("evaluation_allowed") is not False:
                raise ValueError("Unfrozen self-built masks must remain quarantined")
            if source.get("dataset") != SELF_BUILT_MASK_DATASET:
                raise ValueError("Self-built handwriting-mask task references the wrong dataset")
            required_source_fields = {
                "dataset",
                "image_path",
                "image_sha256",
                "width",
                "height",
                "subject",
                "scope",
                "document_group",
                "page_index",
                "asset_id",
                "clean_asset_id",
            }
            if set(source) != required_source_fields or source.get("asset_id") != task.get("sample_id"):
                raise ValueError("Self-built handwriting-mask source fields are invalid")
        if not isinstance(source.get("width"), int) or not isinstance(source.get("height"), int):
            raise ValueError("Handwriting-mask source dimensions are invalid")
        if int(source["width"]) < 1 or int(source["height"]) < 1:
            raise ValueError("Handwriting-mask source dimensions are invalid")
        annotation = task.get("annotation") or {}
        if annotation.get("legend_version") != "handwriting_mask_v1":
            raise ValueError("Invalid handwriting-mask legend version")
        if annotation.get("class_values") != MASK_CLASS_VALUES:
            raise ValueError("Handwriting-mask class values are immutable")
        if annotation.get("status") not in {
            "pending",
            "draft",
            "annotated",
            "verified",
            "adjudication_required",
            "rejected",
        }:
            raise ValueError("Invalid handwriting-mask status")
        counts = annotation.get("pixel_counts") or {}
        if set(counts) != set(MASK_CLASS_VALUES) or any(
            not isinstance(value, int) or value < 0 for value in counts.values()
        ):
            raise ValueError("Invalid handwriting-mask pixel counts")
        mask_relative = annotation.get("mask_path")
        if mask_relative is None:
            if annotation.get("status") not in {"pending", "rejected"}:
                raise ValueError("Submitted handwriting-mask task has no mask")
        else:
            path = (self.annotation_root / str(mask_relative)).resolve()
            if self.annotation_root not in path.parents or not path.is_file():
                raise ValueError("Handwriting mask file is missing or escaped its root")
            if file_sha256(path) != annotation.get("mask_sha256"):
                raise ValueError("Handwriting mask SHA-256 mismatch")
            with Image.open(path) as image:
                if image.mode != "L" or image.size != (source.get("width"), source.get("height")):
                    raise ValueError("Handwriting mask format or dimensions are invalid")
                colors = image.getcolors(maxcolors=max(1, image.width * image.height)) or []
                if not {int(value) for _count, value in colors} <= ALLOWED_MASK_VALUES:
                    raise ValueError("Handwriting mask contains invalid class values")
                if self._pixel_counts(image) != counts:
                    raise ValueError("Handwriting mask pixel counts do not match the file")
        review = task.get("review") or {}
        checks = review.get("checks") or {}
        if set(checks) != MASK_CHECK_KEYS or any(value not in {True, False, None} for value in checks.values()):
            raise ValueError("Invalid handwriting-mask review checks")
        if annotation.get("status") == "verified":
            annotator = str(review.get("annotator_id") or "").strip()
            reviewer = str(review.get("reviewer_id") or "").strip()
            if not annotator or not reviewer or annotator == reviewer:
                raise ValueError("Verified mask requires two distinct identities")
            if any(value is not True for value in checks.values()):
                raise ValueError("Verified mask requires every review check")
            if not self._matching_signoff(task, "annotator", "submit_annotation"):
                raise ValueError("Verified mask is missing its annotator sign-off")
            if not self._matching_signoff(task, "reviewer", "approve_review"):
                raise ValueError("Verified mask is missing its reviewer sign-off")
