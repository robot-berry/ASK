from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from .self_built_intake import CONDITIONS, SelfBuiltIntakeStore
from .storage import write_json_atomic


PAIR_STATUSES = {"pending", "draft", "annotated", "verified", "rejected"}
CAPTURE_ROLES = {"photo_capture", "handwriting_capture"}
ASSESSMENT_KEYS = {
    "same_page",
    "page_content_complete",
    "capture_usable",
    "conditions_confirmed",
    "privacy_clear",
    "handwriting_state",
}
BOOLEAN_ASSESSMENTS = ASSESSMENT_KEYS - {"handwriting_state"}
HANDWRITING_STATES = {None, "present", "absent", "uncertain"}
PAIR_CHECK_KEYS = {
    "clean_capture_match",
    "content_and_edges_complete",
    "quality_usable",
    "conditions_and_handwriting_correct",
    "privacy_and_rights_rechecked",
}


def payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_pair_target_errors(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    targets = summary.get("targets") or {}
    if not isinstance(targets, dict):
        return ["capture-pair summary has no target definitions"]
    for name in (
        "clean_pages",
        "photo_pairs",
        "handwriting_pairs",
        "devices",
    ):
        target = targets.get(name) or {}
        if not isinstance(target, dict):
            errors.append(f"capture-pair target is missing: {name}")
            continue
        current = int(target.get("current") or 0)
        minimum = int(target.get("minimum") or 0)
        if current < minimum:
            errors.append(f"capture-pair target incomplete: {name}={current}/{minimum}")
    conditions = summary.get("verified_condition_counts") or {}
    if not isinstance(conditions, dict):
        conditions = {}
    for condition in ("perspective", "shadow", "fold_or_curve"):
        if int(conditions.get(condition, 0)) < 1:
            errors.append(f"verified capture condition is missing: {condition}")
    if int(conditions.get("blur", 0)) < 1 and int(conditions.get("compression", 0)) < 1:
        errors.append("verified capture condition is missing: blur_or_compression")
    return errors


class CapturePairStore:
    """Two-person verification for clean-page and photographed capture pairs."""

    def __init__(
        self,
        task_dir: Path,
        manifest_path: Path,
        summary_path: Path,
        intake: SelfBuiltIntakeStore,
    ) -> None:
        self.task_dir = task_dir.resolve()
        self.manifest_path = manifest_path.resolve()
        self.summary_path = summary_path.resolve()
        self.intake = intake

    @staticmethod
    def _validate_id(capture_asset_id: str) -> str:
        if not (
            capture_asset_id.startswith("SBI-PHOTO-")
            or capture_asset_id.startswith("SBI-HAND-")
        ) or any(
            character
            not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-"
            for character in capture_asset_id
        ):
            raise ValueError("Invalid capture-pair asset id")
        return capture_asset_id

    def _task_path(self, capture_asset_id: str) -> Path:
        self._validate_id(capture_asset_id)
        path = (self.task_dir / f"{capture_asset_id}.json").resolve()
        if path.parent != self.task_dir:
            raise ValueError("Capture-pair task escaped its directory")
        return path

    @staticmethod
    def _source_projection(record: dict[str, Any]) -> dict[str, Any]:
        source = record["source"]
        metadata = record["metadata"]
        rights = record["rights_and_privacy"]
        return {
            "asset_id": record["asset_id"],
            "dataset": source["dataset"],
            "relative_path": source["relative_path"],
            "sha256": source["sha256"],
            "width": source["width"],
            "height": source["height"],
            "subject": metadata["subject"],
            "scope": metadata["scope"],
            "document_group": metadata["document_group"],
            "page_index": metadata["page_index"],
            "distribution_allowed": rights["distribution_allowed"],
        }

    @classmethod
    def _default_task(
        cls, capture: dict[str, Any], clean: dict[str, Any]
    ) -> dict[str, Any]:
        capture_source = cls._source_projection(capture)
        capture_source.update(
            {
                "role": capture["role"],
                "device_code": capture["metadata"]["device_code"],
                "conditions": list(capture["metadata"]["conditions"]),
            }
        )
        return {
            "schema_version": "capture_pair_review_v1",
            "capture_asset_id": capture["asset_id"],
            "partition": "quarantine",
            "training_allowed": False,
            "evaluation_allowed": False,
            "clean": cls._source_projection(clean),
            "capture": capture_source,
            "annotation": {
                "status": "pending",
                "assessment": {
                    "same_page": None,
                    "page_content_complete": None,
                    "capture_usable": None,
                    "conditions_confirmed": None,
                    "privacy_clear": None,
                    "handwriting_state": None,
                },
            },
            "review": {
                "annotator_id": None,
                "reviewer_id": None,
                "checks": {key: None for key in sorted(PAIR_CHECK_KEYS)},
                "notes": [],
                "signoffs": [],
                "history": [],
                "updated_at_utc": None,
            },
        }

    def _expected_task(self, capture_asset_id: str) -> dict[str, Any]:
        capture = self.intake.load(capture_asset_id)
        if capture["role"] not in CAPTURE_ROLES:
            raise ValueError("Capture-pair task must reference a photo or handwriting capture")
        clean_id = str(capture["metadata"].get("clean_asset_id") or "")
        clean = self.intake.load(clean_id)
        if clean["role"] != "clean_page":
            raise ValueError("Capture-pair clean source is not a clean page")
        return self._default_task(capture, clean)

    def sync_from_intake(self) -> int:
        created = 0
        self.task_dir.mkdir(parents=True, exist_ok=True)
        for item in self.intake.list():
            if item["role"] not in CAPTURE_ROLES:
                continue
            path = self._task_path(item["asset_id"])
            expected = self._expected_task(item["asset_id"])
            if path.is_file():
                existing = json.loads(path.read_text(encoding="utf-8"))
                for field in ("clean", "capture"):
                    if existing.get(field) != expected[field]:
                        raise ValueError(f"{item['asset_id']}: frozen capture-pair source changed")
                continue
            write_json_atomic(path, expected)
            created += 1
        if created:
            self.write_indexes()
        return created

    def load(self, capture_asset_id: str) -> dict[str, Any]:
        path = self._task_path(capture_asset_id)
        if not path.is_file():
            raise FileNotFoundError(capture_asset_id)
        task = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(task, dict):
            raise ValueError("Capture-pair task must be a JSON object")
        self.validate(task)
        expected = self._expected_task(capture_asset_id)
        if task["clean"] != expected["clean"] or task["capture"] != expected["capture"]:
            raise ValueError("Capture-pair task no longer matches intake")
        return task

    def list(self) -> list[dict[str, Any]]:
        self.sync_from_intake()
        items: list[dict[str, Any]] = []
        if not self.task_dir.is_dir():
            return items
        for path in sorted(self.task_dir.glob("*.json")):
            task = self.load(path.stem)
            items.append(
                {
                    "capture_asset_id": task["capture_asset_id"],
                    "clean_asset_id": task["clean"]["asset_id"],
                    "role": task["capture"]["role"],
                    "subject": task["capture"]["subject"],
                    "scope": task["capture"]["scope"],
                    "document_group": task["capture"]["document_group"],
                    "page_index": task["capture"]["page_index"],
                    "device_code": task["capture"]["device_code"],
                    "conditions": task["capture"]["conditions"],
                    "status": task["annotation"]["status"],
                    "annotator_id": task["review"].get("annotator_id"),
                    "reviewer_id": task["review"].get("reviewer_id"),
                }
            )
        items.sort(
            key=lambda item: (
                item["document_group"],
                item["page_index"],
                item["role"],
                item["capture_asset_id"],
            )
        )
        return items

    @staticmethod
    def _assessment_hash(annotation: dict[str, Any]) -> str:
        return payload_hash(annotation.get("assessment") or {})

    @staticmethod
    def _valid_actor(actor_id: object) -> str:
        actor = str(actor_id or "").strip()
        if len(actor) < 2 or len(actor) > 80 or any(character.isspace() for character in actor):
            raise ValueError("Operator id must be 2-80 non-whitespace characters")
        return actor

    @staticmethod
    def _matching_signoff(task: dict[str, Any], role: str, action: str) -> bool:
        assessment_sha = CapturePairStore._assessment_hash(task["annotation"])
        return any(
            isinstance(item, dict)
            and item.get("role") == role
            and item.get("action") == action
            and item.get("assessment_sha256") == assessment_sha
            for item in task["review"].get("signoffs") or []
        )

    @staticmethod
    def _assessment_ready(task: dict[str, Any]) -> bool:
        assessment = task["annotation"]["assessment"]
        if any(assessment[key] is not True for key in BOOLEAN_ASSESSMENTS):
            return False
        expected = "present" if task["capture"]["role"] == "handwriting_capture" else "absent"
        return assessment["handwriting_state"] == expected

    def save(
        self,
        capture_asset_id: str,
        updated: dict[str, Any],
        expected_hash: str | None,
    ) -> dict[str, Any]:
        existing = self.load(capture_asset_id)
        if expected_hash and expected_hash != payload_hash(existing):
            raise RuntimeError("Capture-pair task changed since it was opened; reload before saving")
        if not isinstance(updated, dict):
            raise ValueError("Capture-pair task must be a JSON object")
        for field in (
            "schema_version",
            "capture_asset_id",
            "partition",
            "training_allowed",
            "evaluation_allowed",
            "clean",
            "capture",
        ):
            if updated.get(field) != existing.get(field):
                raise ValueError(f"Capture-pair task {field} is immutable")
        annotation = updated.get("annotation") or {}
        if annotation.get("status") != existing["annotation"]["status"]:
            raise ValueError("Capture-pair annotation status is server-owned")
        review = updated.get("review") or {}
        existing_review = existing["review"]
        for field in ("annotator_id", "reviewer_id", "signoffs", "history", "updated_at_utc"):
            if review.get(field) != existing_review.get(field):
                raise ValueError(f"Capture-pair review {field} is server-owned")
        review["notes"] = [
            str(note).strip() for note in review.get("notes") or [] if str(note).strip()
        ]
        for field in ("annotator_id", "reviewer_id", "signoffs", "history"):
            review[field] = (
                list(existing_review.get(field) or [])
                if field in {"signoffs", "history"}
                else existing_review.get(field)
            )
        changed = self._assessment_hash(annotation) != self._assessment_hash(existing["annotation"])
        if changed:
            previous_status = existing["annotation"]["status"]
            annotation["status"] = "draft"
            review["annotator_id"] = None
            review["reviewer_id"] = None
            review["history"].append(
                {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "from_status": previous_status,
                    "to_status": "draft",
                    "actor_id": None,
                    "action": "assessment_changed_signoffs_invalidated",
                }
            )
        review["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        updated["annotation"] = annotation
        updated["review"] = review
        self.validate(updated)
        write_json_atomic(self._task_path(capture_asset_id), updated)
        self.write_indexes()
        return updated

    def signoff(
        self,
        capture_asset_id: str,
        action: str,
        actor_id: object,
        expected_hash: str | None,
    ) -> dict[str, Any]:
        task = self.load(capture_asset_id)
        if expected_hash and expected_hash != payload_hash(task):
            raise RuntimeError("Capture-pair task changed since it was opened; reload before signing")
        actor = self._valid_actor(actor_id)
        annotation = task["annotation"]
        review = task["review"]
        previous_status = annotation["status"]
        if action == "submit_annotation":
            if previous_status not in {"pending", "draft"}:
                raise ValueError("Only pending or draft capture pairs can be submitted")
            if not self._assessment_ready(task):
                raise ValueError("Capture-pair assessment is incomplete or inconsistent with its role")
            role = "annotator"
            review["annotator_id"] = actor
            review["reviewer_id"] = None
            annotation["status"] = "annotated"
        elif action == "approve_review":
            if previous_status != "annotated" or not review.get("annotator_id"):
                raise ValueError("Reviewer can sign only after annotator submission")
            if actor == review.get("annotator_id"):
                raise ValueError("Capture-pair reviewer must be distinct from annotator")
            if not self._matching_signoff(task, "annotator", "submit_annotation"):
                raise ValueError("Current assessment has no matching annotator sign-off")
            checks = review.get("checks") or {}
            if set(checks) != PAIR_CHECK_KEYS or any(value is not True for value in checks.values()):
                raise ValueError("Every capture-pair review check must be true before approval")
            if not self._assessment_ready(task):
                raise ValueError("Capture-pair assessment is no longer eligible")
            role = "reviewer"
            review["reviewer_id"] = actor
            annotation["status"] = "verified"
        elif action == "request_changes":
            if previous_status not in {"annotated", "verified"}:
                raise ValueError("Changes can be requested only for submitted capture pairs")
            if actor == review.get("annotator_id"):
                raise ValueError("Reviewer requesting changes must be distinct from annotator")
            if not review.get("notes"):
                raise ValueError("Add a problem note before requesting capture-pair changes")
            role = "reviewer"
            review["annotator_id"] = None
            review["reviewer_id"] = None
            annotation["status"] = "draft"
        elif action == "reject_pair":
            if previous_status not in {"pending", "draft", "annotated"}:
                raise ValueError("Only unverified capture pairs can be rejected")
            if not review.get("notes"):
                raise ValueError("Add a reason before rejecting a capture pair")
            if previous_status == "annotated" and actor == review.get("annotator_id"):
                raise ValueError("A submitted capture pair must be rejected by a distinct reviewer")
            role = "reviewer" if previous_status == "annotated" else "annotator"
            annotation["status"] = "rejected"
        else:
            raise ValueError("Unknown capture-pair sign-off action")
        now = datetime.now(timezone.utc).isoformat()
        review.setdefault("signoffs", []).append(
            {
                "at_utc": now,
                "action": action,
                "role": role,
                "actor_id": actor,
                "assessment_sha256": self._assessment_hash(annotation),
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
        write_json_atomic(self._task_path(capture_asset_id), task)
        self.write_indexes()
        return task

    def validate(self, task: dict[str, Any]) -> None:
        if task.get("schema_version") != "capture_pair_review_v1":
            raise ValueError("Invalid capture-pair schema")
        if task.get("partition") != "quarantine":
            raise ValueError("Unfrozen capture pairs must remain quarantined")
        if task.get("training_allowed") is not False or task.get("evaluation_allowed") is not False:
            raise ValueError("Unfrozen capture pairs cannot be used for training or evaluation")
        self._validate_id(str(task.get("capture_asset_id") or ""))
        clean = task.get("clean") or {}
        capture = task.get("capture") or {}
        if not str(clean.get("asset_id") or "").startswith("SBI-CLEAN-"):
            raise ValueError("Capture pair has an invalid clean source")
        if capture.get("asset_id") != task.get("capture_asset_id") or capture.get("role") not in CAPTURE_ROLES:
            raise ValueError("Capture pair has an invalid captured source")
        for source in (clean, capture):
            if source.get("dataset") != "test7_ocr_self_built_v1":
                raise ValueError("Capture-pair source references the wrong dataset")
            if not isinstance(source.get("width"), int) or not isinstance(source.get("height"), int):
                raise ValueError("Capture-pair source dimensions are invalid")
            if len(str(source.get("sha256") or "")) != 64:
                raise ValueError("Capture-pair source SHA-256 is invalid")
        for field in ("subject", "scope", "document_group", "page_index"):
            if clean.get(field) != capture.get(field):
                raise ValueError(f"Clean and capture source {field} do not match")
        if not str(capture.get("device_code") or "").strip():
            raise ValueError("Capture-pair device code is missing")
        conditions = capture.get("conditions")
        if not isinstance(conditions, list) or not conditions or not set(conditions) <= CONDITIONS:
            raise ValueError("Capture-pair conditions are invalid")
        annotation = task.get("annotation") or {}
        if annotation.get("status") not in PAIR_STATUSES:
            raise ValueError("Invalid capture-pair annotation status")
        assessment = annotation.get("assessment") or {}
        if set(assessment) != ASSESSMENT_KEYS:
            raise ValueError("Capture-pair assessment fields do not match the schema")
        if any(assessment[key] not in {True, False, None} for key in BOOLEAN_ASSESSMENTS):
            raise ValueError("Capture-pair assessment contains an invalid boolean")
        if assessment.get("handwriting_state") not in HANDWRITING_STATES:
            raise ValueError("Capture-pair handwriting state is invalid")
        review = task.get("review") or {}
        checks = review.get("checks") or {}
        if set(checks) != PAIR_CHECK_KEYS or any(value not in {True, False, None} for value in checks.values()):
            raise ValueError("Invalid capture-pair review checks")
        if annotation.get("status") == "verified":
            annotator = str(review.get("annotator_id") or "").strip()
            reviewer = str(review.get("reviewer_id") or "").strip()
            if not annotator or not reviewer or annotator == reviewer:
                raise ValueError("Verified capture pairs require two distinct identities")
            if not self._assessment_ready(task) or any(value is not True for value in checks.values()):
                raise ValueError("Verified capture pair lacks a complete positive assessment")
            if not self._matching_signoff(task, "annotator", "submit_annotation"):
                raise ValueError("Verified capture pair lacks an annotator sign-off")
            if not self._matching_signoff(task, "reviewer", "approve_review"):
                raise ValueError("Verified capture pair lacks a reviewer sign-off")

    def summary(self) -> dict[str, Any]:
        items = self.list()
        statuses = Counter(item["status"] for item in items)
        verified = [item for item in items if item["status"] == "verified"]
        roles = Counter(item["role"] for item in verified)
        clean_pages = {item["clean_asset_id"] for item in verified}
        devices = {item["device_code"] for item in verified}
        conditions = Counter(
            condition for item in verified for condition in item.get("conditions") or []
        )
        return {
            "schema_version": "capture_pair_summary_v1",
            "partition": "quarantine",
            "training_allowed": False,
            "evaluation_allowed": False,
            "task_count": len(items),
            "status_counts": dict(sorted(statuses.items())),
            "verified_pairs": len(verified),
            "verified_clean_pages": len(clean_pages),
            "verified_role_counts": dict(sorted(roles.items())),
            "verified_devices": len(devices),
            "verified_condition_counts": dict(sorted(conditions.items())),
            "targets": {
                "clean_pages": {"current": len(clean_pages), "minimum": 50},
                "photo_pairs": {"current": roles["photo_capture"], "minimum": 200},
                "handwriting_pairs": {"current": roles["handwriting_capture"], "minimum": 100},
                "devices": {"current": len(devices), "minimum": 3},
            },
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def write_indexes(self) -> None:
        items = self.list()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
            for item in items
        )
        temporary = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temporary.write_text(manifest, encoding="utf-8", newline="\n")
        temporary.replace(self.manifest_path)
        write_json_atomic(self.summary_path, self.summary())

    def progress_csv(self) -> bytes:
        stream = io.StringIO(newline="")
        fields = [
            "capture_asset_id",
            "clean_asset_id",
            "role",
            "document_group",
            "page_index",
            "device_code",
            "conditions",
            "status",
            "annotator_id",
            "reviewer_id",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in self.list():
            row = dict(item)
            row["conditions"] = "|".join(item.get("conditions") or [])
            writer.writerow({key: row.get(key) for key in fields})
        return ("\ufeff" + stream.getvalue()).encode("utf-8")

    def check(self) -> tuple[dict[str, Any], list[str]]:
        errors: list[str] = []
        try:
            self.sync_from_intake()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(str(exc))
        if self.task_dir.is_dir():
            for path in sorted(self.task_dir.glob("*.json")):
                try:
                    self.load(path.stem)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{path.name}: {exc}")
        return self.summary(), errors
