#!/usr/bin/env python3
"""Promote only double-reviewed testzip tasks into formal OCR gold records."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "data" / "annotations" / "testzip_math_physics_ocr_gold" / "drafts"
DEFAULT_GOLD_OUT = PROJECT_ROOT / "data" / "gold" / "testzip_math_physics_300" / "ocr_gold.jsonl"
DEFAULT_STATUS_OUT = PROJECT_ROOT / "data" / "gold" / "testzip_math_physics_300" / "STATUS.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--gold-out", type=Path, default=DEFAULT_GOLD_OUT)
    parser.add_argument("--status-out", type=Path, default=DEFAULT_STATUS_OUT)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


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


def bbox_union(boxes: list[list[int]]) -> list[int]:
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[0] + box[2] for box in boxes)
    bottom = max(box[1] + box[3] for box in boxes)
    return [left, top, right - left, bottom - top]


def verified(task: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    sample_id = task.get("sample_id", "<unknown>")
    source = task.get("source") or {}
    if not source.get("include_in_benchmark"):
        return False, []
    annotation = task.get("gold_annotation") or {}
    review = task.get("review") or {}
    if annotation.get("status") != "verified" or review.get("status") != "verified":
        return False, []
    people = [
        review.get("annotator_id"),
        review.get("reviewer_1_id"),
        review.get("reviewer_2_id"),
    ]
    if any(not person for person in people) or len(set(people)) != 3:
        errors.append(f"{sample_id}: verified record needs three distinct people")
    adjudication_recorded = any(
        isinstance(event, dict) and event.get("to_status") == "adjudication_required"
        for event in review.get("history", [])
    )
    if adjudication_recorded:
        adjudicator = review.get("adjudicator_id")
        if not adjudicator or adjudicator in people:
            errors.append(f"{sample_id}: disputed record needs a distinct fourth adjudicator")
    if not annotation.get("canonical_transcript").strip():
        errors.append(f"{sample_id}: canonical_transcript is empty")
    if not annotation.get("stem_text").strip():
        errors.append(f"{sample_id}: stem_text is empty")
    if not annotation.get("text_blocks"):
        errors.append(f"{sample_id}: verified record needs source text blocks")
    if annotation.get("formula_status") not in {"human_transcribed", "latex_verified"}:
        errors.append(f"{sample_id}: formula status is not human verified")
    checks = review.get("checks") or {}
    if not checks or any(value is not True for value in checks.values()):
        errors.append(f"{sample_id}: every review check must be true")
    for asset in annotation.get("visual_assets") or []:
        if asset.get("preserve_source_pixels") is not True:
            errors.append(f"{sample_id}: visual asset must preserve source pixels")
        if asset.get("redraw_allowed") is not False:
            errors.append(f"{sample_id}: visual asset redraw is forbidden")
        if asset.get("edge_complete") is not True:
            errors.append(f"{sample_id}: visual asset crop is not edge-complete")
    return not errors, errors


def region_for_final(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "region_id": region["region_id"],
        "kind": region["category"],
        "bbox": region["bbox"],
        "reason": region["reason"],
    }


def build_gold(task: dict[str, Any]) -> dict[str, Any]:
    source = task["source"]
    annotation = task["gold_annotation"]
    review = task["review"]
    blocks = sorted(annotation["text_blocks"], key=lambda block: block["reading_order"])
    assets = sorted(annotation["visual_assets"], key=lambda asset: asset["reading_order"])
    reading_order = [block["block_id"] for block in blocks] + [
        asset["asset_id"] for asset in assets
    ]
    reading_order.sort(
        key=lambda item_id: next(
            item["reading_order"]
            for item in [*blocks, *assets]
            if item.get("block_id", item.get("asset_id")) == item_id
        )
    )
    handwriting_regions = annotation.get("handwriting_regions") or []
    ignore_regions = [
        region_for_final(region)
        for region in [
            *(annotation.get("ignored_regions") or []),
            *handwriting_regions,
        ]
    ]
    final_assets = [
        {
            "asset_id": asset["asset_id"],
            "kind": asset["kind"],
            "diagram_type": asset["kind"],
            "source_bbox": asset["source_bbox"],
            "placement_hint": "after_text",
            "preserve_source_pixels": True,
            "redraw_allowed": False,
            "required_facts": [],
        }
        for asset in assets
    ]
    return {
        "schema_version": "ocr_gold_v1",
        "gold_id": f"OCRG-{task['sample_id']}",
        "sample_id": task["sample_id"],
        "partition": "test",
        "subject": source["subject"],
        "scope": "unknown",
        "locale": "zh-CN",
        "source": {
            "dataset": "testzip_math_physics_300_v1",
            "dataset_version": "testzip_math_physics_300_v1",
            "image_path": source["image_path"],
            "image_sha256": source["image_sha256"],
            "width": source["width"],
            "height": source["height"],
            "color_mode": "RGB",
            "document_group": source["document_group"],
            "evaluation_only": True,
            "training_allowed": False,
        },
        "content": {
            "question_type": annotation["question_type"] or "unknown",
            "question_type_label": "",
            "stem_text": annotation["stem_text"],
            "options": annotation["options"],
            "constants": [],
            "semantic_text": annotation["canonical_transcript"],
            "formula_gold_status": annotation["formula_status"],
        },
        "layout": {
            "source_text_region": bbox_union([block["bbox"] for block in blocks]),
            "source_metadata_regions": [],
            "source_ignore_regions": ignore_regions,
            "reading_order": reading_order,
            "body_font_size": None,
        },
        "visual_assets": final_assets,
        "handwriting": {
            "expected": bool(handwriting_regions),
            "gold_status": (
                "human_regions_annotated" if handwriting_regions else "human_verified_absent"
            ),
        },
        "review": {
            "status": "verified",
            "independent_review_count": 2,
            "required_independent_review_count": 2,
            "source_alignment": "human_verified",
        },
    }


def main() -> int:
    args = parse_args()
    task_paths = sorted(args.review_dir.glob("*.json"))
    if len(task_paths) != 300:
        raise RuntimeError(f"Expected 300 review tasks, found {len(task_paths)}")
    gold_records: list[dict[str, Any]] = []
    errors: list[str] = []
    for task_path in task_paths:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        is_verified, record_errors = verified(task)
        errors.extend(record_errors)
        if is_verified:
            gold_records.append(build_gold(task))
    if errors:
        raise RuntimeError("; ".join(errors[:20]))
    gold_records.sort(key=lambda row: row["sample_id"])
    write_jsonl(args.gold_out, gold_records)
    complete = len(gold_records) == 298
    status = {
        "schema_version": "ocr_gold_status_v1",
        "dataset": "testzip_math_physics_300_v1",
        "status": "retired_transcript_complete" if complete else "retired_transcript_partial",
        "role": "historical_layout_and_handwriting_regression",
        "primary_ocr_gold": False,
        "current_primary_ocr_gold": "gaokao_physics_300_v1",
        "verified_records": len(gold_records),
        "historical_transcript_target": 298,
        "required_records_for_formal_scoring": 0,
        "formal_scoring_enabled": False,
        "gold_file": "data/gold/testzip_math_physics_300/ocr_gold.jsonl",
        "gold_file_sha256": sha256_file(args.gold_out),
        "visual_asset_policy": "crop_original_pixels_only_no_redraw",
    }
    args.status_out.write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if args.require_complete and not complete:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
