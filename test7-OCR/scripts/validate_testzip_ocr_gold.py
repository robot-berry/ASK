#!/usr/bin/env python3
"""Validate the primary testzip OCR gold queue and any promoted gold records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "testzip_math_physics_300_v1"
DEFAULT_QUEUE = PROJECT_ROOT / "data" / "splits" / "testzip_math_physics_ocr_gold300_manifest.jsonl"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "data" / "annotations" / "testzip_math_physics_ocr_gold" / "drafts"
DEFAULT_GOLD = PROJECT_ROOT / "data" / "gold" / "testzip_math_physics_300" / "ocr_gold.jsonl"
DEFAULT_STATUS = PROJECT_ROOT / "data" / "gold" / "testzip_math_physics_300" / "STATUS.json"
DEFAULT_VISUAL_CANDIDATES = PROJECT_ROOT / "data" / "splits" / "testzip_visual_bbox_candidates.jsonl"
BANNED_GOLD_KEYS = {
    "answer",
    "standard_answer",
    "canonical_answer",
    "accepted_answers",
    "solution",
    "solution_steps",
    "scoring_points",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--visual-candidates", type=Path, default=DEFAULT_VISUAL_CANDIDATES)
    parser.add_argument("--require-verified", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bbox(
    bbox: Any, width: int, height: int, label: str, errors: list[str]
) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        errors.append(f"{label}: bbox must be [x,y,width,height]")
        return
    if not all(isinstance(value, int) for value in bbox):
        errors.append(f"{label}: bbox values must be integers")
        return
    x, y, box_width, box_height = bbox
    if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
        errors.append(f"{label}: invalid bbox {bbox}")
    elif x + box_width > width or y + box_height > height:
        errors.append(f"{label}: bbox {bbox} exceeds {width}x{height}")


def find_banned_keys(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in BANNED_GOLD_KEYS:
                found.append(child_path)
            found.extend(find_banned_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_banned_keys(child, f"{path}[{index}]"))
    return found


def validate_review_task(
    queue_row: dict[str, Any], task: dict[str, Any], errors: list[str]
) -> bool:
    sample_id = queue_row["sample_id"]
    source = task.get("source") or {}
    if task.get("schema_version") != "ocr_gold_review_v1":
        errors.append(f"{sample_id}: wrong review schema version")
    if task.get("sample_id") != sample_id:
        errors.append(f"{sample_id}: task sample_id mismatch")
    expected_pairs = {
        "dataset": "testzip_math_physics_300_v1",
        "image_path": queue_row["image_path"],
        "image_sha256": queue_row["image_sha256"],
        "subject": queue_row["subject"],
        "source_group": queue_row["source_group"],
        "source_index": queue_row["source_index"],
        "document_group": queue_row["document_group"],
        "include_in_benchmark": queue_row["include_in_benchmark"],
        "duplicate_of": queue_row["duplicate_of"],
    }
    for field, expected in expected_pairs.items():
        if source.get(field) != expected:
            errors.append(f"{sample_id}: source.{field} does not match queue")
    if source.get("width") != 1280 or source.get("height") != 720:
        errors.append(f"{sample_id}: source dimensions are not 1280x720")

    candidates = task.get("candidate_prelabels") or {}
    if candidates.get("warning") != "PRELABEL_IS_NOT_GOLD":
        errors.append(f"{sample_id}: prelabel warning is missing")
    if len(candidates.get("candidates") or []) < 2:
        errors.append(f"{sample_id}: two candidate prelabels are required")

    annotation = task.get("gold_annotation") or {}
    review = task.get("review") or {}
    included = bool(source.get("include_in_benchmark"))
    if not included:
        if annotation.get("status") != "excluded_duplicate" or review.get("status") != "excluded_duplicate":
            errors.append(f"{sample_id}: duplicate exclusion status mismatch")
        return False

    width, height = source["width"], source["height"]
    for index, block in enumerate(annotation.get("text_blocks") or []):
        validate_bbox(block.get("bbox"), width, height, f"{sample_id}.text_blocks[{index}]", errors)
    for group_name in ("ignored_regions", "handwriting_regions"):
        for index, region in enumerate(annotation.get(group_name) or []):
            validate_bbox(region.get("bbox"), width, height, f"{sample_id}.{group_name}[{index}]", errors)
    for index, asset in enumerate(annotation.get("visual_assets") or []):
        validate_bbox(asset.get("source_bbox"), width, height, f"{sample_id}.visual_assets[{index}]", errors)
        if asset.get("preserve_source_pixels") is not True:
            errors.append(f"{sample_id}: visual asset {index} does not preserve source pixels")
        if asset.get("redraw_allowed") is not False:
            errors.append(f"{sample_id}: visual asset {index} permits forbidden redraw")

    annotation_verified = annotation.get("status") == "verified"
    review_verified = review.get("status") == "verified"
    if annotation_verified != review_verified:
        errors.append(f"{sample_id}: annotation/review verified status mismatch")
    if not annotation_verified:
        return False

    people = [
        review.get("annotator_id"),
        review.get("reviewer_1_id"),
        review.get("reviewer_2_id"),
    ]
    if any(not person for person in people) or len(set(people)) != 3:
        errors.append(f"{sample_id}: three distinct human IDs are required")
    checks = review.get("checks") or {}
    if not checks or any(value is not True for value in checks.values()):
        errors.append(f"{sample_id}: every verification check must be true")
    if not str(annotation.get("canonical_transcript") or "").strip():
        errors.append(f"{sample_id}: canonical transcript is empty")
    if not str(annotation.get("stem_text") or "").strip():
        errors.append(f"{sample_id}: stem text is empty")
    if not annotation.get("text_blocks"):
        errors.append(f"{sample_id}: verified gold needs text blocks")
    if annotation.get("formula_status") not in {"human_transcribed", "latex_verified"}:
        errors.append(f"{sample_id}: formula status is not human verified")
    if any(asset.get("edge_complete") is not True for asset in annotation.get("visual_assets") or []):
        errors.append(f"{sample_id}: verified asset has an incomplete crop")
    return True


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    queue = read_jsonl(args.queue)
    if len(queue) != 300:
        errors.append(f"queue: expected 300 rows, found {len(queue)}")
    ids = [row.get("sample_id") for row in queue]
    queue_by_id = {row["sample_id"]: row for row in queue}
    if len(ids) != len(set(ids)):
        errors.append("queue: duplicate sample_id")
    if sum(bool(row.get("include_in_benchmark")) for row in queue) != 298:
        errors.append("queue: benchmark eligible count must be 298")
    subjects = Counter(row.get("subject") for row in queue)
    if subjects != Counter({"mathematics": 150, "physics": 150}):
        errors.append(f"queue: subject counts are wrong: {dict(subjects)}")
    if any(row.get("split") != "test" for row in queue):
        errors.append("queue: every record must remain in the frozen test split")
    if any(row.get("training_allowed") is not False for row in queue):
        errors.append("queue: training must be forbidden")
    banned_queue = find_banned_keys(queue)
    if banned_queue:
        errors.append(f"queue: forbidden answer fields found: {banned_queue[:5]}")

    visual_candidates = read_jsonl(args.visual_candidates)
    if len(visual_candidates) != 300:
        errors.append(f"visual candidates: expected 300 rows, found {len(visual_candidates)}")
    visual_by_id = {row.get("sample_id"): row for row in visual_candidates}
    if set(visual_by_id) != set(ids):
        errors.append("visual candidates: sample IDs do not match queue")
    accepted_visual_assets = 0
    for sample_id, row in visual_by_id.items():
        queue_row = queue_by_id.get(sample_id)
        if queue_row is None:
            continue
        if row.get("image_sha256") != queue_row.get("image_sha256"):
            errors.append(f"{sample_id}: visual candidate source hash mismatch")
        for index, asset in enumerate(row.get("assets") or []):
            if asset.get("accepted_as_candidate") is True:
                accepted_visual_assets += 1
                validate_bbox(asset.get("source_bbox"), 1280, 720, f"{sample_id}.candidate_assets[{index}]", errors)
                if float(asset.get("template_match_score") or 0.0) < 0.94:
                    errors.append(f"{sample_id}: accepted visual candidate score is below 0.94")
            if asset.get("bbox_verified") is not False:
                errors.append(f"{sample_id}: candidate bbox must remain unverified")
            if asset.get("legacy_crop_pixels_reused") is not False:
                errors.append(f"{sample_id}: legacy crop pixels must not be reused")

    task_paths = sorted(args.review_dir.glob("*.json"))
    if len(task_paths) != 300:
        errors.append(f"review tasks: expected 300 files, found {len(task_paths)}")
    task_ids = {path.stem for path in task_paths}
    if task_ids != set(queue_by_id):
        errors.append("review tasks: filenames do not match queue IDs")

    verified = 0
    for task_path in task_paths:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        queue_row = queue_by_id.get(task_path.stem)
        if queue_row is None:
            continue
        banned = find_banned_keys(task)
        if banned:
            errors.append(f"{task_path.stem}: forbidden answer fields found: {banned[:3]}")
        if validate_review_task(queue_row, task, errors):
            verified += 1

        image_path = args.dataset_root.resolve() / Path(queue_row["image_path"])
        if not image_path.is_file():
            errors.append(f"{task_path.stem}: source image missing")
        elif sha256_file(image_path) != queue_row["image_sha256"]:
            errors.append(f"{task_path.stem}: source image hash mismatch")
        else:
            with Image.open(image_path) as image:
                if image.size != (1280, 720) or image.format != "PNG":
                    errors.append(f"{task_path.stem}: image contract mismatch")

    gold_rows = read_jsonl(args.gold)
    if len(gold_rows) != verified:
        errors.append(
            f"gold: promoted row count {len(gold_rows)} does not match verified tasks {verified}"
        )
    if find_banned_keys(gold_rows):
        errors.append("gold: answer or solution fields are forbidden")
    if any(row.get("source", {}).get("dataset") != "testzip_math_physics_300_v1" for row in gold_rows):
        errors.append("historical transcript: a record uses a different dataset")
    for row in gold_rows:
        for asset in row.get("visual_assets") or []:
            if asset.get("preserve_source_pixels") is not True or asset.get("redraw_allowed") is not False:
                errors.append(f"{row.get('sample_id')}: promoted visual asset policy violation")

    status = json.loads(args.status.read_text(encoding="utf-8")) if args.status.exists() else {}
    if status.get("verified_records") != verified:
        errors.append("status: verified count mismatch")
    if bool(status.get("formal_scoring_enabled")):
        errors.append("status: retired Testzip data must never enable formal OCR scoring")
    if status.get("primary_ocr_gold") is not False:
        errors.append("status: retired Testzip data must be marked primary_ocr_gold=false")
    if status.get("current_primary_ocr_gold") != "gaokao_physics_300_v1":
        errors.append("status: current primary OCR Gold must remain gaokao_physics_300_v1")
    if args.require_verified and verified != 298:
        errors.append(f"freeze gate: only {verified}/298 records are verified")

    config_text = (PROJECT_ROOT / "configs" / "self_built_data.yaml").read_text(
        encoding="utf-8"
    )
    if "primary_ocr_gold: gaokao_physics_300_v1" not in config_text:
        errors.append("config: primary OCR Gold is not gaokao_physics_300_v1")
    if "historical_testzip_math_physics_300_v1:" not in config_text:
        errors.append("config: retired Testzip regression role is missing")

    report = {
        "source_records": len(queue),
        "benchmark_eligible": sum(bool(row.get("include_in_benchmark")) for row in queue),
        "review_tasks": len(task_paths),
        "verified_gold": verified,
        "promoted_gold": len(gold_rows),
        "formal_scoring_enabled": False,
        "dataset_role": "retired_historical_layout_and_handwriting_regression",
        "primary_gold_dataset": "gaokao_physics_300_v1",
        "testzip_primary_gold": False,
        "visual_asset_policy": "crop_original_pixels_only_no_redraw",
        "accepted_unverified_visual_bbox_candidates": accepted_visual_assets,
        "error_count": len(errors),
        "passed": not errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    for error in errors[:100]:
        print(f"ERROR: {error}", file=sys.stderr)
    if len(errors) > 100:
        print(f"ERROR: ... and {len(errors) - 100} more", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
