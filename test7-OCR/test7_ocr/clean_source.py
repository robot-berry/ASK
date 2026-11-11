from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .capture_pair_review import PAIR_CHECK_KEYS, CapturePairStore, payload_hash
from .image_ops import sha256_file


VERIFIED_CLEAN_SOURCE_METHOD = "verified_capture_pair_clean_source_direct_pixels"


@dataclass(frozen=True)
class SourceResolution:
    input_image: str
    input_sha256: str
    effective_image: str
    effective_sha256: str
    substituted: bool
    method: str
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def direct_source_resolution(image_path: Path, image_sha256: str) -> SourceResolution:
    resolved = image_path.resolve()
    return SourceResolution(
        input_image=str(resolved),
        input_sha256=image_sha256,
        effective_image=str(resolved),
        effective_sha256=image_sha256,
        substituted=False,
        method="direct_input",
        evidence=None,
    )


class VerifiedCleanSourceResolver:
    """Resolve exact registered captures to their human-verified clean source."""

    def __init__(
        self,
        pair_store: CapturePairStore,
        *,
        allowed_roles: set[str] | None = None,
        fail_closed_on_invalid_evidence: bool = True,
    ) -> None:
        self.pair_store = pair_store
        self.allowed_roles = set(allowed_roles or {"handwriting_capture"})
        self.fail_closed_on_invalid_evidence = fail_closed_on_invalid_evidence
        self._by_capture_sha256: dict[str, dict[str, Any]] = {}
        self.refresh()

    def refresh(self) -> int:
        candidates: dict[str, dict[str, Any]] = {}
        if not self.pair_store.task_dir.is_dir():
            self._by_capture_sha256 = {}
            return 0
        for path in sorted(self.pair_store.task_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("Capture-pair task must be a JSON object")
                if (raw.get("annotation") or {}).get("status") != "verified":
                    continue
                task = self.pair_store.load(path.stem)
                role = str((task.get("capture") or {}).get("role") or "")
                if role not in self.allowed_roles:
                    continue
                capture_sha256 = str(task["capture"]["sha256"])
                clean_sha256 = str(task["clean"]["sha256"])
                clean_path = self.pair_store.intake.source_path(task["clean"]["asset_id"])
                capture_path = self.pair_store.intake.source_path(task["capture_asset_id"])
                if sha256_file(clean_path) != clean_sha256:
                    raise ValueError("Verified clean source SHA-256 changed")
                if sha256_file(capture_path) != capture_sha256:
                    raise ValueError("Verified capture source SHA-256 changed")
                evidence = {
                    "pair_task_path": str(path.resolve()),
                    "pair_task_sha256": sha256_file(path),
                    "pair_status": "verified",
                    "pair_partition": task["partition"],
                    "capture_asset_id": task["capture_asset_id"],
                    "capture_role": role,
                    "capture_registered_image": str(capture_path),
                    "capture_sha256": capture_sha256,
                    "clean_asset_id": task["clean"]["asset_id"],
                    "clean_registered_image": str(clean_path),
                    "clean_sha256": clean_sha256,
                    "document_group": task["clean"]["document_group"],
                    "page_index": task["clean"]["page_index"],
                    "annotator_id": task["review"]["annotator_id"],
                    "reviewer_id": task["review"]["reviewer_id"],
                }
                previous = candidates.get(capture_sha256)
                if previous and previous["clean_sha256"] != clean_sha256:
                    raise ValueError(
                        "One verified capture hash maps to multiple clean sources"
                    )
                candidates[capture_sha256] = evidence
            except (OSError, ValueError, json.JSONDecodeError):
                if self.fail_closed_on_invalid_evidence:
                    raise
        self._by_capture_sha256 = candidates
        return len(candidates)

    def resolve(self, image_path: Path, image_sha256: str) -> SourceResolution:
        input_path = image_path.resolve()
        evidence = self._by_capture_sha256.get(image_sha256)
        if evidence is None:
            return direct_source_resolution(input_path, image_sha256)
        clean_path = Path(evidence["clean_registered_image"]).resolve()
        clean_sha256 = str(evidence["clean_sha256"])
        if not clean_path.is_file() or sha256_file(clean_path) != clean_sha256:
            raise ValueError("Verified clean source is missing or no longer hash-matched")
        task_path = Path(evidence["pair_task_path"]).resolve()
        if not task_path.is_file() or sha256_file(task_path) != evidence["pair_task_sha256"]:
            raise ValueError("Verified capture-pair evidence changed; refresh before processing")
        return SourceResolution(
            input_image=str(input_path),
            input_sha256=image_sha256,
            effective_image=str(clean_path),
            effective_sha256=clean_sha256,
            substituted=True,
            method=VERIFIED_CLEAN_SOURCE_METHOD,
            evidence=dict(evidence),
        )


def validate_source_resolution(value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise ValueError("Source resolution must be an object")
    substituted = value.get("substituted")
    method = value.get("method")
    for field in ("input_image", "input_sha256", "effective_image", "effective_sha256"):
        if not str(value.get(field) or ""):
            raise ValueError(f"Source resolution is missing {field}")
    input_path = Path(str(value["input_image"])).resolve()
    effective_path = Path(str(value["effective_image"])).resolve()
    if not input_path.is_file() or sha256_file(input_path) != value["input_sha256"]:
        raise ValueError("Input source no longer matches source-resolution SHA-256")
    if not effective_path.is_file() or sha256_file(effective_path) != value["effective_sha256"]:
        raise ValueError("Effective source no longer matches source-resolution SHA-256")
    if substituted is False:
        if method != "direct_input" or value.get("evidence") is not None:
            raise ValueError("Direct input source resolution has invalid evidence")
        if value["input_sha256"] != value["effective_sha256"]:
            raise ValueError("Direct input source hashes disagree")
        return
    if substituted is not True or method != VERIFIED_CLEAN_SOURCE_METHOD:
        raise ValueError("Unsupported source-resolution method")
    evidence = value.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("pair_status") != "verified":
        raise ValueError("Clean-source substitution lacks verified pair evidence")
    if evidence.get("capture_sha256") != value["input_sha256"]:
        raise ValueError("Capture-pair input hash does not match source resolution")
    if evidence.get("clean_sha256") != value["effective_sha256"]:
        raise ValueError("Capture-pair clean hash does not match source resolution")
    if evidence.get("capture_role") != "handwriting_capture":
        raise ValueError("Only verified handwriting captures may use clean-source substitution")
    if not evidence.get("annotator_id") or not evidence.get("reviewer_id"):
        raise ValueError("Clean-source substitution lacks two-person identities")
    if evidence["annotator_id"] == evidence["reviewer_id"]:
        raise ValueError("Clean-source substitution reviewers are not independent")
    registered_capture = Path(str(evidence.get("capture_registered_image") or "")).resolve()
    registered_clean = Path(str(evidence.get("clean_registered_image") or "")).resolve()
    if not registered_capture.is_file() or sha256_file(registered_capture) != value["input_sha256"]:
        raise ValueError("Registered capture no longer matches source-resolution SHA-256")
    if registered_clean != effective_path:
        raise ValueError("Effective source is not the registered clean source")
    pair_path = Path(str(evidence.get("pair_task_path") or "")).resolve()
    if not pair_path.is_file() or sha256_file(pair_path) != evidence.get("pair_task_sha256"):
        raise ValueError("Capture-pair task no longer matches recorded evidence hash")
    try:
        pair_task = json.loads(pair_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Capture-pair task is unreadable") from exc
    if not isinstance(pair_task, dict):
        raise ValueError("Capture-pair task is not an object")
    if (
        pair_task.get("schema_version") != "capture_pair_review_v1"
        or pair_task.get("partition") != "quarantine"
        or pair_task.get("training_allowed") is not False
        or pair_task.get("evaluation_allowed") is not False
        or (pair_task.get("annotation") or {}).get("status") != "verified"
    ):
        raise ValueError("Capture-pair task is not a verified quarantine record")
    capture = pair_task.get("capture") or {}
    clean = pair_task.get("clean") or {}
    expected_pairs = (
        (pair_task.get("capture_asset_id"), evidence.get("capture_asset_id")),
        (capture.get("asset_id"), evidence.get("capture_asset_id")),
        (capture.get("role"), "handwriting_capture"),
        (capture.get("sha256"), value["input_sha256"]),
        (clean.get("asset_id"), evidence.get("clean_asset_id")),
        (clean.get("sha256"), value["effective_sha256"]),
        (clean.get("document_group"), evidence.get("document_group")),
        (clean.get("page_index"), evidence.get("page_index")),
    )
    if any(actual != expected for actual, expected in expected_pairs):
        raise ValueError("Capture-pair task content does not match recorded evidence")
    annotation = pair_task.get("annotation") or {}
    assessment = annotation.get("assessment") or {}
    positive_fields = (
        "same_page",
        "page_content_complete",
        "capture_usable",
        "conditions_confirmed",
        "privacy_clear",
    )
    if (
        any(assessment.get(field) is not True for field in positive_fields)
        or assessment.get("handwriting_state") != "present"
    ):
        raise ValueError("Capture-pair task lacks a complete positive assessment")
    review = pair_task.get("review") or {}
    if (
        review.get("annotator_id") != evidence["annotator_id"]
        or review.get("reviewer_id") != evidence["reviewer_id"]
        or set(review.get("checks") or {}) != PAIR_CHECK_KEYS
        or any(value is not True for value in (review.get("checks") or {}).values())
    ):
        raise ValueError("Capture-pair task lacks matching two-person approval")
    assessment_sha256 = payload_hash(assessment)
    signoffs = review.get("signoffs") or []
    expected_signoffs = (
        ("annotator", "submit_annotation", evidence["annotator_id"]),
        ("reviewer", "approve_review", evidence["reviewer_id"]),
    )
    for role, action, actor_id in expected_signoffs:
        if not any(
            isinstance(item, dict)
            and item.get("role") == role
            and item.get("action") == action
            and item.get("actor_id") == actor_id
            and item.get("assessment_sha256") == assessment_sha256
            for item in signoffs
        ):
            raise ValueError("Capture-pair task lacks hash-bound two-person sign-offs")
