#!/usr/bin/env python3
"""Prepare testzip_math_physics_300_v1 as the only primary OCR gold queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "testzip_math_physics_300_v1"
DEFAULT_SEMANTIC_REVIEW = (
    WORKSPACE_ROOT
    / "picture"
    / "test6"
    / "benchmark"
    / "ocr_accuracy_300_semantic_complete_final_20260903"
    / "STRUCTURED_OCR_REVIEW_300.json"
)
DEFAULT_BALANCED_REVIEW = (
    WORKSPACE_ROOT
    / "picture"
    / "test6"
    / "benchmark"
    / "ocr_accuracy_300_balanced_final_20260903"
    / "STRUCTURED_OCR_REVIEW_300.json"
)
DEFAULT_PRIOR_AUDIT = (
    WORKSPACE_ROOT
    / "picture"
    / "test6"
    / "benchmark"
    / "ocr_accuracy_300_20260902"
    / "full_audit_queue"
    / "full_manual_audit.csv"
)
DEFAULT_MANIFEST_OUT = PROJECT_ROOT / "data" / "splits" / "testzip_math_physics_ocr_gold300_manifest.jsonl"
DEFAULT_SUMMARY_OUT = PROJECT_ROOT / "data" / "splits" / "testzip_math_physics_ocr_gold300_summary.json"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "data" / "annotations" / "testzip_math_physics_ocr_gold" / "drafts"
DEFAULT_GOLD_STATUS = PROJECT_ROOT / "data" / "gold" / "testzip_math_physics_300" / "STATUS.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--semantic-review", type=Path, default=DEFAULT_SEMANTIC_REVIEW)
    parser.add_argument("--balanced-review", type=Path, default=DEFAULT_BALANCED_REVIEW)
    parser.add_argument("--prior-audit", type=Path, default=DEFAULT_PRIOR_AUDIT)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--gold-status", type=Path, default=DEFAULT_GOLD_STATUS)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_workspace(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalize_display_text(value: str) -> str:
    text = re.sub(r"\s+", "", str(value or ""))
    return text.translate(str.maketrans({"，": ",", "。": ".", "（": "(", "）": ")"}))


def optional_bool(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    return None


def load_structured_review(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    return payload.get("metrics") or {}, {row["question_id"]: row for row in rows}


def load_prior_audit(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["question_id"]: row for row in csv.DictReader(handle)}


def candidate_entry(source: str, row: dict[str, Any], automatic_pass: bool) -> dict[str, Any]:
    return {
        "source": source,
        "transcript": str(row.get("latest_transcript") or row.get("transcript") or ""),
        "display_transcript": str(
            row.get("latest_display_transcript") or row.get("display_transcript") or ""
        ),
        "automatic_text_pass": bool(automatic_pass),
        "latency_seconds": (
            float(row["latency_seconds"])
            if row.get("latency_seconds") not in (None, "")
            else None
        ),
    }


def is_blank_draft(task: dict[str, Any]) -> bool:
    annotation = task.get("gold_annotation") or {}
    review = task.get("review") or {}
    checks = review.get("checks") or {}
    return bool(
        annotation.get("status") in {"draft", "excluded_duplicate"}
        and not annotation.get("question_type")
        and annotation.get("question_number") is None
        and not annotation.get("stem_text")
        and not annotation.get("options")
        and not annotation.get("canonical_transcript")
        and annotation.get("formula_status") == "pending"
        and not annotation.get("text_blocks")
        and not annotation.get("visual_assets")
        and not annotation.get("ignored_regions")
        and not annotation.get("handwriting_regions")
        and review.get("status") in {"pending", "excluded_duplicate"}
        and not review.get("annotator_id")
        and not review.get("reviewer_1_id")
        and not review.get("reviewer_2_id")
        and not review.get("adjudicator_id")
        and all(value is None for value in checks.values())
        and not review.get("notes")
        and not review.get("history")
    )


def review_priority(
    include_in_benchmark: bool,
    agreement: bool,
    prior_semantic_complete: bool | None,
    error_types: list[str],
) -> str:
    if not include_in_benchmark:
        return "excluded"
    combined = " ".join(error_types).lower()
    if "handwriting" in combined or "watermark" in combined:
        return "p0_handwriting_or_noise"
    if not agreement or prior_semantic_complete is not True:
        return "p1_disagreement_or_known_error"
    return "p2_agreed_prelabel"


def build_task(
    manifest: dict[str, Any],
    semantic: dict[str, Any],
    balanced: dict[str, Any],
    prior: dict[str, Any],
) -> dict[str, Any]:
    semantic_entry = candidate_entry(
        "test6_semantic_complete_final_20260903",
        semantic,
        bool(semantic.get("automatic_structure_pass", semantic.get("automatic_package_pass"))),
    )
    balanced_entry = candidate_entry(
        "test6_balanced_final_20260903",
        balanced,
        bool(balanced.get("automatic_structure_pass", balanced.get("automatic_package_pass"))),
    )
    agreement = normalize_display_text(semantic_entry["display_transcript"]) == normalize_display_text(
        balanced_entry["display_transcript"]
    )
    error_types = [
        value.strip()
        for value in str(prior.get("manual_error_types") or "").split(";")
        if value.strip()
    ]
    prior_semantic = optional_bool(prior.get("manual_semantic_complete"))
    include = bool(manifest["include_in_benchmark"])
    priority = review_priority(include, agreement, prior_semantic, error_types)
    excluded = not include
    return {
        "schema_version": "ocr_gold_review_v1",
        "sample_id": manifest["question_id"],
        "source": {
            "dataset": "testzip_math_physics_300_v1",
            "image_path": manifest["image_path"],
            "image_sha256": manifest["image_sha256"],
            "width": int(manifest["width"]),
            "height": int(manifest["height"]),
            "subject": manifest["subject"],
            "source_group": manifest["source_group"],
            "source_index": int(manifest["source_index"]),
            "document_group": f"{manifest['subject']}:{manifest['source_group']}",
            "include_in_benchmark": include,
            "duplicate_of": manifest.get("duplicate_of"),
            "copyright_status": "unknown_user_provided",
        },
        "candidate_prelabels": {
            "suggested_transcript": semantic_entry["transcript"],
            "normalized_agreement": agreement,
            "candidates": [semantic_entry, balanced_entry],
            "warning": "PRELABEL_IS_NOT_GOLD",
        },
        "prior_manual_evidence": {
            "semantic_complete": prior_semantic,
            "exact_normalized": optional_bool(prior.get("manual_exact_normalized")),
            "error_types": error_types,
            "notes": str(prior.get("manual_notes") or ""),
        },
        "gold_annotation": {
            "status": "excluded_duplicate" if excluded else "draft",
            "question_type": "",
            "question_number": None,
            "stem_text": "",
            "options": [],
            "canonical_transcript": "",
            "formula_status": "pending",
            "text_blocks": [],
            "visual_assets": [],
            "ignored_regions": [],
            "handwriting_regions": [],
        },
        "review": {
            "priority": priority,
            "status": "excluded_duplicate" if excluded else "pending",
            "annotator_id": None,
            "reviewer_1_id": None,
            "reviewer_2_id": None,
            "adjudicator_id": None,
            "checks": {
                "text_exact": None,
                "formula_exact": None,
                "options_complete": None,
                "visual_assets_complete": None,
                "handwriting_excluded": None,
                "reading_order_correct": None,
            },
            "notes": [],
            "history": [],
        },
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    source_manifest_path = dataset_root / "manifest.jsonl"
    source_summary_path = dataset_root / "dataset_summary.json"
    manifest_rows = read_jsonl(source_manifest_path)
    semantic_metrics, semantic_rows = load_structured_review(args.semantic_review)
    balanced_metrics, balanced_rows = load_structured_review(args.balanced_review)
    prior_rows = load_prior_audit(args.prior_audit)

    ids = {row["question_id"] for row in manifest_rows}
    for label, values in (
        ("semantic", semantic_rows),
        ("balanced", balanced_rows),
        ("prior audit", prior_rows),
    ):
        if set(values) != ids:
            raise RuntimeError(f"{label} IDs do not match the 300-row source manifest")

    if len(manifest_rows) != 300:
        raise RuntimeError(f"Expected 300 source records, found {len(manifest_rows)}")

    args.review_dir.mkdir(parents=True, exist_ok=True)
    queue_rows: list[dict[str, Any]] = []
    tasks_created = 0
    blank_tasks_refreshed = 0
    for manifest in sorted(manifest_rows, key=lambda row: row["question_id"]):
        image_path = dataset_root / Path(manifest["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing source image: {image_path}")
        if sha256_file(image_path) != manifest["image_sha256"]:
            raise RuntimeError(f"Image hash mismatch: {manifest['question_id']}")
        with Image.open(image_path) as image:
            if image.size != (1280, 720) or image.format != "PNG":
                raise RuntimeError(
                    f"Unexpected image contract for {manifest['question_id']}: "
                    f"{image.format} {image.size}"
                )

        task = build_task(
            manifest,
            semantic_rows[manifest["question_id"]],
            balanced_rows[manifest["question_id"]],
            prior_rows[manifest["question_id"]],
        )
        task_path = args.review_dir / f"{manifest['question_id']}.json"
        if task_path.exists():
            existing = json.loads(task_path.read_text(encoding="utf-8"))
            if is_blank_draft(existing) and existing != task:
                task_path.write_text(
                    json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                blank_tasks_refreshed += 1
        else:
            task_path.write_text(
                json.dumps(task, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            tasks_created += 1

        queue_rows.append(
            {
                "schema_version": "ocr_gold_manifest_v1",
                "sample_id": task["sample_id"],
                "split": "test",
                "subject": task["source"]["subject"],
                "source_group": task["source"]["source_group"],
                "source_index": task["source"]["source_index"],
                "document_group": task["source"]["document_group"],
                "image_path": task["source"]["image_path"],
                "image_sha256": task["source"]["image_sha256"],
                "include_in_benchmark": task["source"]["include_in_benchmark"],
                "duplicate_of": task["source"]["duplicate_of"],
                "review_priority": task["review"]["priority"],
                "review_status": task["review"]["status"],
                "prelabel_normalized_agreement": task["candidate_prelabels"]["normalized_agreement"],
                "gold_text_available": False,
                "training_allowed": False,
            }
        )

    write_jsonl(args.manifest_out, queue_rows)
    priority_counts = Counter(row["review_priority"] for row in queue_rows)
    subject_counts = Counter(row["subject"] for row in queue_rows)
    summary = {
        "schema_version": "testzip_ocr_gold_summary_v1",
        "dataset": "testzip_math_physics_300_v1",
        "dataset_root": relative_to_workspace(dataset_root),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_dataset_summary_sha256": sha256_file(source_summary_path),
        "source_records": len(queue_rows),
        "benchmark_eligible": sum(row["include_in_benchmark"] for row in queue_rows),
        "excluded_duplicates": sum(not row["include_in_benchmark"] for row in queue_rows),
        "subject_counts": dict(sorted(subject_counts.items())),
        "prelabel_normalized_agreement": sum(row["prelabel_normalized_agreement"] for row in queue_rows),
        "prelabel_disagreement": sum(not row["prelabel_normalized_agreement"] for row in queue_rows),
        "review_priority_counts": dict(sorted(priority_counts.items())),
        "verified_ocr_gold_count": 0,
        "formal_ocr_scoring_enabled": False,
        "answer_pdf_used_as_ocr_gold": False,
        "role": "retired_historical_layout_and_handwriting_regression",
        "primary_ocr_gold": False,
        "current_primary_ocr_gold": "gaokao_physics_300_v1",
        "visual_asset_policy": "crop_original_pixels_only_no_redraw",
        "semantic_prelabel_metrics": semantic_metrics,
        "balanced_prelabel_metrics": balanced_metrics,
        "tasks_created_this_run": tasks_created,
        "blank_tasks_refreshed_this_run": blank_tasks_refreshed,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.gold_status.parent.mkdir(parents=True, exist_ok=True)
    args.gold_status.write_text(
        json.dumps(
            {
                "schema_version": "ocr_gold_status_v1",
                "dataset": "testzip_math_physics_300_v1",
                "status": "retired_historical_regression_only",
                "role": "historical_layout_and_handwriting_regression",
                "primary_ocr_gold": False,
                "current_primary_ocr_gold": "gaokao_physics_300_v1",
                "verified_records": 0,
                "historical_transcript_target": 298,
                "required_records_for_formal_scoring": 0,
                "formal_scoring_enabled": False,
                "reason": "该数据集已从正式 Gold 角色退役；人工转写只可用于历史审计和安全回归。",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
