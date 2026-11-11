from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from typing import Any

from .self_built_intake import SelfBuiltIntakeStore
from .storage import write_json_atomic


BOUNDARY_CHECK_KEYS = {
    "all_questions_present",
    "boundaries_complete",
    "options_inside_question",
    "reading_order_correct",
    "columns_and_continuations_correct",
}
BOUNDARY_STATUSES = {"pending", "draft", "annotated", "verified", "rejected"}
COLUMNS = {"left", "right", "full_width", "mixed"}


def boundary_target_errors(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    targets = summary.get("targets") or {}
    if not isinstance(targets, dict):
        return ["question-boundary summary has no target definitions"]
    for name in ("pages", "questions", "single_column", "double_column"):
        target = targets.get(name) or {}
        if not isinstance(target, dict):
            errors.append(f"question-boundary target is missing: {name}")
            continue
        current = int(target.get("current") or 0)
        minimum = int(target.get("minimum") or 0)
        if current < minimum:
            errors.append(f"question-boundary target incomplete: {name}={current}/{minimum}")
    return errors


def payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class QuestionBoundaryStore:
    """Human question-boundary annotation over quarantined full-page intake assets."""

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
    def _validate_id(asset_id: str) -> str:
        if not asset_id.startswith("SBI-BOUNDARY-") or any(
            character not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-"
            for character in asset_id
        ):
            raise ValueError("Invalid boundary asset id")
        return asset_id

    def _task_path(self, asset_id: str) -> Path:
        self._validate_id(asset_id)
        path = (self.task_dir / f"{asset_id}.json").resolve()
        if path.parent != self.task_dir:
            raise ValueError("Boundary task escaped its directory")
        return path

    @staticmethod
    def _default_task(record: dict[str, Any]) -> dict[str, Any]:
        source = record["source"]
        metadata = record["metadata"]
        return {
            "schema_version": "question_boundary_review_v1",
            "asset_id": record["asset_id"],
            "partition": "quarantine",
            "training_allowed": False,
            "evaluation_allowed": False,
            "source": {
                "dataset": source["dataset"],
                "relative_path": source["relative_path"],
                "sha256": source["sha256"],
                "width": source["width"],
                "height": source["height"],
                "subject": metadata["subject"],
                "scope": metadata["scope"],
                "document_group": metadata["document_group"],
                "page_index": metadata["page_index"],
                "page_layout": metadata["page_layout"],
            },
            "annotation": {"status": "pending", "questions": []},
            "review": {
                "annotator_id": None,
                "reviewer_id": None,
                "checks": {key: None for key in sorted(BOUNDARY_CHECK_KEYS)},
                "notes": [],
                "signoffs": [],
                "history": [],
                "updated_at_utc": None,
            },
        }

    def sync_from_intake(self) -> int:
        created = 0
        self.task_dir.mkdir(parents=True, exist_ok=True)
        for item in self.intake.list():
            if item["role"] != "boundary_page":
                continue
            path = self._task_path(item["asset_id"])
            record = self.intake.load(item["asset_id"])
            expected = self._default_task(record)
            if path.is_file():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("source") != expected["source"]:
                    raise ValueError(f"{item['asset_id']}: frozen boundary source changed")
                continue
            write_json_atomic(path, expected)
            created += 1
        if created:
            self.write_indexes()
        return created

    def load(self, asset_id: str) -> dict[str, Any]:
        path = self._task_path(asset_id)
        if not path.is_file():
            raise FileNotFoundError(asset_id)
        task = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(task, dict):
            raise ValueError("Boundary task must be a JSON object")
        self.validate(task)
        record = self.intake.load(asset_id)
        if record["role"] != "boundary_page" or task["source"] != self._default_task(record)["source"]:
            raise ValueError("Boundary task source no longer matches intake")
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
                    "asset_id": task["asset_id"],
                    "subject": task["source"]["subject"],
                    "document_group": task["source"]["document_group"],
                    "page_index": task["source"]["page_index"],
                    "page_layout": task["source"]["page_layout"],
                    "status": task["annotation"]["status"],
                    "question_count": len(task["annotation"]["questions"]),
                    "annotator_id": task["review"].get("annotator_id"),
                    "reviewer_id": task["review"].get("reviewer_id"),
                }
            )
        items.sort(key=lambda item: (item["document_group"], item["page_index"], item["asset_id"]))
        return items

    @staticmethod
    def _annotation_hash(annotation: dict[str, Any]) -> str:
        payload = dict(annotation)
        payload.pop("status", None)
        return payload_hash(payload)

    @staticmethod
    def _valid_actor(actor_id: object) -> str:
        actor = str(actor_id or "").strip()
        if len(actor) < 2 or len(actor) > 80 or any(character.isspace() for character in actor):
            raise ValueError("Operator id must be 2-80 non-whitespace characters")
        return actor

    @staticmethod
    def _matching_signoff(task: dict[str, Any], role: str, action: str) -> bool:
        annotation_sha = QuestionBoundaryStore._annotation_hash(task["annotation"])
        return any(
            isinstance(item, dict)
            and item.get("role") == role
            and item.get("action") == action
            and item.get("annotation_sha256") == annotation_sha
            for item in task["review"].get("signoffs") or []
        )

    def save(
        self,
        asset_id: str,
        updated: dict[str, Any],
        expected_hash: str | None,
    ) -> dict[str, Any]:
        existing = self.load(asset_id)
        if expected_hash and expected_hash != payload_hash(existing):
            raise RuntimeError("Boundary task changed since it was opened; reload before saving")
        if not isinstance(updated, dict):
            raise ValueError("Boundary task must be a JSON object")
        for field in (
            "schema_version",
            "asset_id",
            "partition",
            "training_allowed",
            "evaluation_allowed",
            "source",
        ):
            if updated.get(field) != existing.get(field):
                raise ValueError(f"Boundary task {field} is immutable")
        updated_annotation = updated.get("annotation") or {}
        existing_annotation = existing["annotation"]
        if updated_annotation.get("status") != existing_annotation.get("status"):
            raise ValueError("Boundary annotation status is server-owned")
        updated_review = updated.get("review") or {}
        existing_review = existing["review"]
        for field in ("annotator_id", "reviewer_id", "signoffs", "history", "updated_at_utc"):
            if updated_review.get(field) != existing_review.get(field):
                raise ValueError(f"Boundary review {field} is server-owned")
        updated_review["notes"] = [
            str(note).strip() for note in updated_review.get("notes") or [] if str(note).strip()
        ]
        updated_review["annotator_id"] = existing_review.get("annotator_id")
        updated_review["reviewer_id"] = existing_review.get("reviewer_id")
        updated_review["signoffs"] = list(existing_review.get("signoffs") or [])
        updated_review["history"] = list(existing_review.get("history") or [])
        changed = self._annotation_hash(updated_annotation) != self._annotation_hash(existing_annotation)
        previous_status = existing_annotation["status"]
        if changed:
            updated_review["annotator_id"] = None
            updated_review["reviewer_id"] = None
            updated_annotation["status"] = "draft"
            updated_review["history"].append(
                {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "from_status": previous_status,
                    "to_status": "draft",
                    "actor_id": None,
                    "action": "boundary_changed_signoffs_invalidated",
                }
            )
        updated_review["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        updated["annotation"] = updated_annotation
        updated["review"] = updated_review
        self.validate(updated)
        write_json_atomic(self._task_path(asset_id), updated)
        self.write_indexes()
        return updated

    def signoff(
        self,
        asset_id: str,
        action: str,
        actor_id: object,
        expected_hash: str | None,
    ) -> dict[str, Any]:
        task = self.load(asset_id)
        if expected_hash and expected_hash != payload_hash(task):
            raise RuntimeError("Boundary task changed since it was opened; reload before signing")
        actor = self._valid_actor(actor_id)
        annotation = task["annotation"]
        review = task["review"]
        previous_status = annotation["status"]
        role: str
        if action == "submit_annotation":
            if previous_status not in {"pending", "draft"}:
                raise ValueError("Only pending or draft boundary tasks can be submitted")
            if len(annotation["questions"]) < 2:
                raise ValueError("A multi-question boundary page requires at least two question boxes")
            role = "annotator"
            review["annotator_id"] = actor
            review["reviewer_id"] = None
            annotation["status"] = "annotated"
        elif action == "approve_review":
            if previous_status != "annotated" or not review.get("annotator_id"):
                raise ValueError("Reviewer can sign only after annotator submission")
            if actor == review.get("annotator_id"):
                raise ValueError("Boundary reviewer must be distinct from annotator")
            if not self._matching_signoff(task, "annotator", "submit_annotation"):
                raise ValueError("Current boundaries have no matching annotator sign-off")
            checks = review.get("checks") or {}
            if set(checks) != BOUNDARY_CHECK_KEYS or any(value is not True for value in checks.values()):
                raise ValueError("Every boundary review check must be true before approval")
            role = "reviewer"
            review["reviewer_id"] = actor
            annotation["status"] = "verified"
        elif action == "request_changes":
            if previous_status not in {"annotated", "verified"}:
                raise ValueError("Changes can be requested only for submitted boundaries")
            if actor == review.get("annotator_id"):
                raise ValueError("Reviewer requesting changes must be distinct from annotator")
            if not review.get("notes"):
                raise ValueError("Add a problem note before requesting boundary changes")
            role = "reviewer"
            review["annotator_id"] = None
            review["reviewer_id"] = None
            annotation["status"] = "draft"
        else:
            raise ValueError("Unknown boundary sign-off action")
        now = datetime.now(timezone.utc).isoformat()
        annotation_sha = self._annotation_hash(annotation)
        review.setdefault("signoffs", []).append(
            {
                "at_utc": now,
                "action": action,
                "role": role,
                "actor_id": actor,
                "annotation_sha256": annotation_sha,
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
        write_json_atomic(self._task_path(asset_id), task)
        self.write_indexes()
        return task

    def validate(self, task: dict[str, Any]) -> None:
        if task.get("schema_version") != "question_boundary_review_v1":
            raise ValueError("Invalid question-boundary schema")
        if task.get("partition") != "quarantine":
            raise ValueError("Unfrozen boundary tasks must remain quarantined")
        if task.get("training_allowed") is not False or task.get("evaluation_allowed") is not False:
            raise ValueError("Unfrozen boundaries cannot be used for training or evaluation")
        self._validate_id(str(task.get("asset_id") or ""))
        source = task.get("source") or {}
        width = source.get("width")
        height = source.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
            raise ValueError("Boundary source dimensions are invalid")
        annotation = task.get("annotation") or {}
        if annotation.get("status") not in BOUNDARY_STATUSES:
            raise ValueError("Invalid boundary annotation status")
        questions = annotation.get("questions")
        if not isinstance(questions, list):
            raise ValueError("Boundary questions must be a list")
        ids: list[str] = []
        orders: list[int] = []
        for question in questions:
            if not isinstance(question, dict):
                raise ValueError("Every boundary must be a structured question object")
            if set(question) != {
                "question_id",
                "bbox",
                "reading_order",
                "parent_question_id",
                "column",
                "continues_on_next_page",
                "notes",
            }:
                raise ValueError("Boundary question fields do not match the schema")
            question_id = str(question.get("question_id") or "").strip()
            if not question_id or len(question_id) > 80:
                raise ValueError("Each boundary requires a question id")
            bbox = question.get("bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(not isinstance(value, int) for value in bbox)
            ):
                raise ValueError("Boundary bbox must contain four integers")
            x, y, box_width, box_height = bbox
            if x < 0 or y < 0 or box_width < 1 or box_height < 1 or x + box_width > width or y + box_height > height:
                raise ValueError("Question boundary escapes the source page")
            order = question.get("reading_order")
            if not isinstance(order, int) or order < 0:
                raise ValueError("Boundary reading_order must be a non-negative integer")
            if question.get("column") not in COLUMNS:
                raise ValueError("Boundary column is invalid")
            if not isinstance(question.get("continues_on_next_page"), bool):
                raise ValueError("Boundary continuation flag must be boolean")
            if not isinstance(question.get("notes"), str):
                raise ValueError("Boundary notes must be text")
            ids.append(question_id)
            orders.append(order)
        if len(ids) != len(set(ids)):
            raise ValueError("Question ids must be unique on a page")
        if sorted(orders) != list(range(len(questions))):
            raise ValueError("Reading order must be unique and contiguous from zero")
        id_set = set(ids)
        for question in questions:
            parent = question.get("parent_question_id")
            if parent is not None and (parent not in id_set or parent == question["question_id"]):
                raise ValueError("Boundary parent_question_id must reference another question on this page")
        review = task.get("review") or {}
        checks = review.get("checks") or {}
        if set(checks) != BOUNDARY_CHECK_KEYS or any(value not in {True, False, None} for value in checks.values()):
            raise ValueError("Invalid boundary review checks")
        if annotation.get("status") == "verified":
            annotator = str(review.get("annotator_id") or "").strip()
            reviewer = str(review.get("reviewer_id") or "").strip()
            if not annotator or not reviewer or annotator == reviewer:
                raise ValueError("Verified boundaries require two distinct identities")
            if len(questions) < 2 or any(value is not True for value in checks.values()):
                raise ValueError("Verified boundary page requires questions and every review check")
            if not self._matching_signoff(task, "annotator", "submit_annotation"):
                raise ValueError("Verified boundaries lack an annotator sign-off")
            if not self._matching_signoff(task, "reviewer", "approve_review"):
                raise ValueError("Verified boundaries lack a reviewer sign-off")

    def summary(self) -> dict[str, Any]:
        items = self.list()
        statuses = Counter(item["status"] for item in items)
        verified = [item for item in items if item["status"] == "verified"]
        verified_layouts = Counter(item["page_layout"] for item in verified)
        return {
            "schema_version": "question_boundary_summary_v1",
            "partition": "quarantine",
            "training_allowed": False,
            "evaluation_allowed": False,
            "task_count": len(items),
            "status_counts": dict(sorted(statuses.items())),
            "verified_pages": len(verified),
            "verified_questions": sum(item["question_count"] for item in verified),
            "verified_layout_counts": dict(sorted(verified_layouts.items())),
            "targets": {
                "pages": {"current": len(verified), "minimum": 60},
                "questions": {"current": sum(item["question_count"] for item in verified), "minimum": 300},
                "single_column": {"current": verified_layouts["single_column"], "minimum": 30},
                "double_column": {"current": verified_layouts["double_column"], "minimum": 30},
            },
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def write_indexes(self) -> None:
        items = self.list()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = "".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in items
        )
        temporary = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temporary.write_text(manifest, encoding="utf-8", newline="\n")
        temporary.replace(self.manifest_path)
        write_json_atomic(self.summary_path, self.summary())

    def progress_csv(self) -> bytes:
        stream = io.StringIO(newline="")
        fields = [
            "asset_id",
            "document_group",
            "page_index",
            "page_layout",
            "status",
            "question_count",
            "annotator_id",
            "reviewer_id",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in self.list():
            writer.writerow({key: item.get(key) for key in fields})
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
