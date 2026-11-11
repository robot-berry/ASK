from __future__ import annotations

import re
from typing import Any


OUTPUT_ISSUE_KEYS = (
    "semantic_content_incomplete",
    "formula_render_error",
    "latex_source_visible",
    "handwriting_visible",
    "visual_crop_incomplete",
    "visual_not_source_crop",
    "adjacent_question_or_noise",
    "layout_clipped_or_overflow",
    "blank_unit_or_punctuation_missing",
    "output_blurry_or_unreadable",
)
QUALITY_ID = re.compile(r"^[0-9A-Za-z._-]+$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def validate_output_quality_review(review: dict[str, Any]) -> None:
    if review.get("schema_version") != "output_quality_review_v1":
        raise ValueError("Invalid output quality review schema version")
    for field in ("run_id", "sample_id"):
        if not QUALITY_ID.fullmatch(str(review.get(field) or "")):
            raise ValueError(f"Invalid output quality review {field}")
    for field in ("source_sha256", "output_sha256"):
        if not SHA256.fullmatch(str(review.get(field) or "")):
            raise ValueError(f"Invalid output quality review {field}")
    verdict = review.get("verdict")
    if verdict not in {"pending", "pass", "fail"}:
        raise ValueError("Output verdict must be pending, pass, or fail")
    issues = review.get("issues")
    if not isinstance(issues, dict) or set(issues) != set(OUTPUT_ISSUE_KEYS):
        raise ValueError("Output review issue checklist is incomplete")
    if any(value not in {True, False, None} for value in issues.values()):
        raise ValueError("Output review issue values must be true, false, or null")
    notes = review.get("notes")
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ValueError("Output review notes must be a list of strings")
    if verdict in {"pass", "fail"} and not str(review.get("reviewer_id") or "").strip():
        raise ValueError("A reviewer identity is required for a final output verdict")
    if verdict == "pass" and any(value is not False for value in issues.values()):
        raise ValueError("Every output issue must be explicitly marked absent before PASS")
    if verdict == "fail" and not any(value is True for value in issues.values()):
        raise ValueError("At least one output issue must be marked present before FAIL")
