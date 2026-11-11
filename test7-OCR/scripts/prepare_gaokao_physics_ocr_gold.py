#!/usr/bin/env python3
"""Build and validate source-derived OCR Gold for gaokao_physics_300_v1.

The dataset images are deterministically rendered from the isolated generator fact
source.  This script promotes only the printed question content into OCR Gold; it
never copies answer, solution, scoring, or classification truth into the OCR
surface.  Answer/solution Gold keeps the dataset's separate human-review gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent
DATASET_NAME = "gaokao_physics_300_v1"
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "datasets" / "gaokao_physics_300_v1_0_1"
DEFAULT_QUEUE = PROJECT_ROOT / "data" / "splits" / "gaokao_physics_ocr_gold300_manifest.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "splits" / "gaokao_physics_ocr_gold300_summary.json"
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "data" / "annotations" / "gaokao_physics_ocr_gold" / "drafts"
DEFAULT_GOLD = PROJECT_ROOT / "data" / "gold" / "gaokao_physics_300" / "ocr_gold.jsonl"
DEFAULT_STATUS = PROJECT_ROOT / "data" / "gold" / "gaokao_physics_300" / "STATUS.json"
DEFAULT_VISUALS = PROJECT_ROOT / "data" / "splits" / "gaokao_physics_visual_bbox_candidates.jsonl"
EXPECTED_COUNT = 300
ANSWER_KEYS = {
    "answer",
    "accepted_answers",
    "analysis_gold",
    "canonical_answer",
    "parts",
    "scoring_points",
    "solution",
    "solution_steps",
    "standard_answer",
}
TYPE_LABELS = {
    "single_choice": "单项选择题",
    "multiple_choice": "多项选择题",
    "numeric_fill": "数值填空题",
    "calculation": "计算题",
    "experiment_fill": "实验填空题",
    "experiment_calculation": "实验计算题",
}
CIRCUIT_TYPES = {"parallel_circuit", "battery_circuit", "transformer"}
FORCE_TYPES = {"incline", "pulley", "projectile", "braking", "pursuit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--visuals", type=Path, default=DEFAULT_VISUALS)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing generated outputs without modifying them",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
    )
    write_text_atomic(path, payload)


def find_answer_keys(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in ANSWER_KEYS:
                found.append(child_path)
            found.extend(find_answer_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_answer_keys(child, f"{path}[{index}]"))
    return found


def semantic_text(stem: str, options: list[dict[str, str]]) -> str:
    lines = [stem.strip()]
    lines.extend(f"{option['label']}. {option['text']}" for option in options)
    if not options:
        lines.append("答题要求：写出必要关系式、计算过程和单位。")
    return "\n".join(line for line in lines if line)


def diagram_layout(question_type: str, diagram: dict[str, Any] | None) -> tuple[list[int] | None, list[int]]:
    if not diagram:
        return None, [70, 140, 1140, 510]
    width = 510 if question_type.startswith("experiment") else 360
    x = 1280 - 64 - width
    return [x, 102, width, 510], [70, 140, x - 108, 510]


def visual_kind(diagram_type: str) -> str:
    if diagram_type == "data_table":
        return "table"
    if diagram_type in CIRCUIT_TYPES:
        return "circuit_diagram"
    if diagram_type in FORCE_TYPES:
        return "force_diagram"
    return "physics_diagram"


def build_records(
    dataset_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    dataset_card = json.loads((dataset_root / "dataset_card.json").read_text(encoding="utf-8"))
    validation = json.loads(
        (dataset_root / "qa" / "validation_report.json").read_text(encoding="utf-8")
    )
    manifests = read_jsonl(dataset_root / "manifest.jsonl")
    source_gold = read_jsonl(dataset_root / "gold" / "gold.jsonl")
    manifest_by_id = {str(row["question_id"]): row for row in manifests}
    source_gold_by_id = {str(row["question_id"]): row for row in source_gold}
    errors: list[str] = []
    if len(manifests) != EXPECTED_COUNT or len(manifest_by_id) != EXPECTED_COUNT:
        errors.append(f"source manifest must contain {EXPECTED_COUNT} unique records")
    if len(source_gold) != EXPECTED_COUNT or set(source_gold_by_id) != set(manifest_by_id):
        errors.append("isolated generator Gold IDs do not match the source manifest")
    if dataset_card.get("status") != "candidate_machine_validated_human_review_pending":
        errors.append("unexpected dataset release status")
    if validation.get("machine_validation_status") != "passed":
        errors.append("upstream machine validation has not passed")
    if validation.get("failures"):
        errors.append("upstream validation report contains failures")
    if errors:
        raise ValueError("; ".join(errors))

    queue: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    tasks: dict[str, dict[str, Any]] = {}
    for source_id in sorted(manifest_by_id):
        manifest = manifest_by_id[source_id]
        fact = source_gold_by_id[source_id]
        item_path = dataset_root / "gold" / "items" / f"{source_id}.json"
        item = json.loads(item_path.read_text(encoding="utf-8"))
        if fact.get("canonical_stem") != item.get("stem"):
            raise ValueError(f"{source_id}: canonical stem does not match generator item")
        if fact.get("options") != item.get("options"):
            raise ValueError(f"{source_id}: options do not match generator item")
        if fact.get("classification", {}).get("question_type") != item.get("question_type"):
            raise ValueError(f"{source_id}: question type does not match generator item")
        if fact.get("stimulus_gold", {}).get("diagram_spec") != item.get("diagram"):
            raise ValueError(f"{source_id}: diagram spec does not match generator item")
        required_generator_checks = {
            "symbolic_check",
            "numeric_check",
            "dimension_check",
            "generator_constraints_check",
        }
        verification = fact.get("verification") or {}
        if any(verification.get(key) != "passed" for key in required_generator_checks):
            raise ValueError(f"{source_id}: generator fact checks are incomplete")

        image_info = manifest.get("image") or {}
        image_relative = str(image_info.get("path") or "")
        image_path = (dataset_root / image_relative).resolve()
        if dataset_root.resolve() not in image_path.parents or not image_path.is_file():
            raise ValueError(f"{source_id}: source image is missing or escaped the dataset root")
        expected_hash = str(image_info.get("file_sha256") or "")
        if sha256_file(image_path) != expected_hash:
            raise ValueError(f"{source_id}: source image SHA-256 mismatch")
        with Image.open(image_path) as opened:
            if opened.format != "PNG" or opened.size != (1280, 720):
                raise ValueError(f"{source_id}: expected a 1280x720 PNG")

        question_type = str(item["question_type"])
        options = [
            {"label": str(option["label"]), "text": str(option["text"])}
            for option in item.get("options") or []
        ]
        printed_stem = str(item["stem"]).strip()
        if not options:
            printed_stem += "\n答题要求：写出必要关系式、计算过程和单位。"
        transcript = semantic_text(str(item["stem"]), options)
        diagram = item.get("diagram")
        diagram_bbox, text_bbox = diagram_layout(question_type, diagram)
        family_id = str(item["family_id"])
        source_asset: list[dict[str, Any]] = []
        task_assets: list[dict[str, Any]] = []
        visual_row_assets: list[dict[str, Any]] = []
        if diagram_bbox is not None:
            diagram_type = str(diagram["type"])
            kind = visual_kind(diagram_type)
            source_asset = [
                {
                    "asset_id": "asset_01",
                    "kind": kind,
                    "diagram_type": diagram_type,
                    "source_bbox": diagram_bbox,
                    "placement_hint": "right_of_text",
                    "preserve_source_pixels": True,
                    "redraw_allowed": False,
                    "required_facts": [
                        str(value)
                        for value in fact.get("analysis_gold", {}).get(
                            "required_diagram_facts", []
                        )
                    ],
                }
            ]
            task_assets = [
                {
                    "asset_id": "asset_01",
                    "kind": kind,
                    "source_bbox": diagram_bbox,
                    "reading_order": len(options) + 1,
                    "preserve_source_pixels": True,
                    "redraw_allowed": False,
                    "edge_complete": True,
                    "contains_handwriting": False,
                }
            ]
            visual_row_assets = [
                {
                    "asset_id": "asset_01",
                    "source_bbox": diagram_bbox,
                    "diagram_type": diagram_type,
                    "accepted_as_candidate": True,
                    "bbox_verified": True,
                    "verification_basis": "generator_layout_contract",
                    "crop_policy": "crop_again_from_current_original",
                    "legacy_crop_pixels_reused": False,
                }
            ]

        document_group = f"physics:{family_id}"
        gold_row = {
            "schema_version": "ocr_gold_v1",
            "gold_id": f"OCRG-{source_id}",
            "sample_id": source_id,
            "partition": "test",
            "subject": "physics",
            "scope": "high_school",
            "locale": "zh-CN",
            "source": {
                "dataset": DATASET_NAME,
                "dataset_version": str(manifest["dataset_version"]),
                "image_path": image_relative,
                "image_sha256": expected_hash,
                "width": 1280,
                "height": 720,
                "color_mode": str(image_info.get("color_mode") or "MONO_1BIT"),
                "document_group": document_group,
                "evaluation_only": True,
                "training_allowed": False,
            },
            "content": {
                "question_type": question_type,
                "question_type_label": TYPE_LABELS[question_type],
                "stem_text": printed_stem,
                "options": options,
                "constants": [str(value) for value in item.get("constants") or []],
                "semantic_text": transcript,
                "formula_gold_status": "generator_source_verified",
            },
            "layout": {
                "source_text_region": text_bbox,
                "source_metadata_regions": [
                    {
                        "region_id": "metadata_header",
                        "kind": "header",
                        "bbox": [64, 18, 1152, 62],
                        "reason": "Synthetic dataset title and source question ID are not question content.",
                    },
                    {
                        "region_id": "metadata_type_tag",
                        "kind": "question_type_tag",
                        "bbox": [64, 96, 260, 38],
                        "reason": "The type badge is renderer metadata; question type is stored structurally.",
                    },
                    {
                        "region_id": "metadata_footer",
                        "kind": "footer",
                        "bbox": [64, 666, 1152, 45],
                        "reason": "Dataset constants/status footer is excluded from the question transcript.",
                    },
                ],
                "source_ignore_regions": [],
                "reading_order": ["body", *[f"option_{item['label']}" for item in options], *(["asset_01"] if source_asset else [])],
                "body_font_size": int(manifest.get("render", {}).get("body_font_size") or 36),
            },
            "visual_assets": source_asset,
            "handwriting": {"expected": False, "gold_status": "synthetic_clean_source"},
            "review": {
                "status": "source_generated_verified",
                "independent_review_count": 0,
                "required_independent_review_count": 0,
                "source_alignment": "generator_fact_source",
            },
        }
        answer_leaks = find_answer_keys(gold_row)
        if answer_leaks:
            raise ValueError(f"{source_id}: answer/solution fields leaked into OCR Gold: {answer_leaks}")
        gold.append(gold_row)

        queue.append(
            {
                "schema_version": "ocr_gold_manifest_v1",
                "sample_id": source_id,
                "split": "test",
                "subject": "physics",
                "source_group": family_id,
                "source_index": int(item["variant_index"]),
                "document_group": document_group,
                "image_path": image_relative,
                "image_sha256": expected_hash,
                "include_in_benchmark": True,
                "duplicate_of": None,
                "review_priority": "p2_agreed_prelabel",
                "review_status": "source_generated_verified",
                "prelabel_normalized_agreement": True,
                "gold_text_available": True,
                "gold_verification_method": "generator_fact_source",
                "expected_visual_asset_count": len(source_asset),
                "training_allowed": False,
            }
        )
        visuals.append(
            {
                "schema_version": "visual_bbox_candidate_v1",
                "sample_id": source_id,
                "image_path": image_relative,
                "image_sha256": expected_hash,
                "assets": visual_row_assets,
                "warning": "SOURCE_GENERATED_BBOX_BOUND_TO_RENDER_CONTRACT",
            }
        )
        task = {
            "schema_version": "ocr_gold_review_v1",
            "sample_id": source_id,
            "source": {
                "dataset": DATASET_NAME,
                "image_path": image_relative,
                "image_sha256": expected_hash,
                "width": 1280,
                "height": 720,
                "subject": "physics",
                "source_group": family_id,
                "source_index": int(item["variant_index"]),
                "document_group": document_group,
                "include_in_benchmark": True,
                "duplicate_of": None,
                "copyright_status": "project_owned_synthetic_evaluation",
            },
            "candidate_prelabels": {
                "suggested_transcript": transcript,
                "normalized_agreement": True,
                "candidates": [
                    {
                        "source": "generator_fact_source",
                        "transcript": transcript,
                        "display_transcript": transcript,
                        "automatic_text_pass": True,
                        "latency_seconds": None,
                    }
                ],
                "warning": "PRELABEL_IS_NOT_GOLD",
            },
            "prior_manual_evidence": {
                "semantic_complete": True,
                "exact_normalized": True,
                "error_types": [],
                "notes": "Derived from the same structured fact source that rendered the immutable PNG.",
            },
            "gold_annotation": {
                "status": "draft",
                "question_type": question_type,
                "question_number": source_id.removeprefix("GKP-PHY-"),
                "stem_text": printed_stem,
                "options": options,
                "canonical_transcript": transcript,
                "formula_status": "pending",
                "text_blocks": [],
                "visual_assets": task_assets,
                "ignored_regions": [],
                "handwriting_regions": [],
            },
            "review": {
                "priority": "p2_agreed_prelabel",
                "status": "pending",
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
        tasks[source_id] = task
    return queue, gold, visuals, tasks, {"dataset_card": dataset_card, "validation": validation}


def validate_outputs(
    dataset_root: Path,
    queue_path: Path,
    gold_path: Path,
    visuals_path: Path,
    status_path: Path,
) -> dict[str, Any]:
    queue = read_jsonl(queue_path)
    gold = read_jsonl(gold_path)
    visuals = read_jsonl(visuals_path)
    errors: list[str] = []
    if len(queue) != EXPECTED_COUNT or len({row.get("sample_id") for row in queue}) != EXPECTED_COUNT:
        errors.append(f"queue must contain {EXPECTED_COUNT} unique records")
    if len(gold) != EXPECTED_COUNT or len({row.get("sample_id") for row in gold}) != EXPECTED_COUNT:
        errors.append(f"Gold must contain {EXPECTED_COUNT} unique records")
    queue_ids = {row.get("sample_id") for row in queue}
    if {row.get("sample_id") for row in gold} != queue_ids:
        errors.append("Gold IDs do not match queue IDs")
    if {row.get("sample_id") for row in visuals} != queue_ids:
        errors.append("visual bbox IDs do not match queue IDs")
    if find_answer_keys(gold):
        errors.append("answer or solution keys are present in OCR Gold")
    for row in gold:
        sample_id = str(row.get("sample_id") or "<unknown>")
        source = row.get("source") or {}
        review = row.get("review") or {}
        if source.get("dataset") != DATASET_NAME:
            errors.append(f"{sample_id}: wrong source dataset")
        if source.get("training_allowed") is not False or source.get("evaluation_only") is not True:
            errors.append(f"{sample_id}: invalid evaluation-only policy")
        if review != {
            "status": "source_generated_verified",
            "independent_review_count": 0,
            "required_independent_review_count": 0,
            "source_alignment": "generator_fact_source",
        }:
            errors.append(f"{sample_id}: invalid source-derived verification record")
        image_path = (dataset_root / str(source.get("image_path") or "")).resolve()
        if dataset_root not in image_path.parents or not image_path.is_file():
            errors.append(f"{sample_id}: source image missing")
        elif sha256_file(image_path) != source.get("image_sha256"):
            errors.append(f"{sample_id}: source image hash mismatch")
        for asset in row.get("visual_assets") or []:
            if asset.get("preserve_source_pixels") is not True or asset.get("redraw_allowed") is not False:
                errors.append(f"{sample_id}: direct source-crop policy changed")
            bbox = asset.get("source_bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                errors.append(f"{sample_id}: malformed visual bbox")
            elif bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > 1280 or bbox[1] + bbox[3] > 720:
                errors.append(f"{sample_id}: visual bbox exceeds source image")
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
    if status.get("verified_records") != EXPECTED_COUNT:
        errors.append("status verified count mismatch")
    if status.get("formal_ocr_scoring_enabled") is not True:
        errors.append("source-derived OCR scoring gate is disabled")
    return {
        "schema_version": "gaokao_physics_ocr_gold_validation_v1",
        "dataset": DATASET_NAME,
        "source_records": len(queue),
        "source_generated_verified_gold": len(gold),
        "visual_asset_records": sum(bool(row.get("assets")) for row in visuals),
        "visual_assets": sum(len(row.get("assets") or []) for row in visuals),
        "formal_ocr_scoring_enabled": not errors,
        "benchmark_scope": "synthetic_source_derived_high_school_physics",
        "real_world_generalization_claim_allowed": False,
        "answer_solution_human_review_complete": False,
        "error_count": len(errors),
        "errors": errors,
        "passed": not errors,
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if args.check:
        report = validate_outputs(dataset_root, args.queue, args.gold, args.visuals, args.status)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1

    queue, gold, visuals, tasks, upstream = build_records(dataset_root)
    write_jsonl(args.queue, queue)
    write_jsonl(args.gold, gold)
    write_jsonl(args.visuals, visuals)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    preserved = 0
    refreshed_pending_sources = 0
    for sample_id, task in tasks.items():
        path = args.review_dir / f"{sample_id}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            review = existing.get("review") or {}
            unsigned_pending = review.get("status") == "pending" and not any(
                review.get(field)
                for field in (
                    "annotator_id",
                    "reviewer_1_id",
                    "reviewer_2_id",
                    "adjudicator_id",
                )
            )
            source_changed = existing.get("source") != task.get("source")
            if unsigned_pending and source_changed:
                # A corrected deterministic render changes the source hash but
                # not a user's draft transcription.  Refresh only the frozen
                # source identity; never overwrite annotations or sign-offs.
                existing["source"] = task["source"]
                write_json(path, existing)
                refreshed_pending_sources += 1
                continue
            preserved += 1
            continue
        write_json(path, task)
        created += 1

    status = {
        "schema_version": "ocr_gold_status_v2",
        "dataset": DATASET_NAME,
        "dataset_version": upstream["dataset_card"]["dataset_version"],
        "status": "source_generated_ocr_gold_ready_answer_review_pending",
        "verification_basis": "generator_fact_source_and_hash_bound_deterministic_render",
        "verified_records": EXPECTED_COUNT,
        "required_records": EXPECTED_COUNT,
        "formal_ocr_scoring_enabled": True,
        "benchmark_scope": "synthetic_source_derived_high_school_physics",
        "real_world_generalization_claim_allowed": False,
        "answer_solution_human_review_complete": False,
        "answer_solution_review_note": "The upstream dataset still requires two independent physics reviewers before its answer/solution release can be frozen.",
        "gold_file": args.gold.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "gold_file_sha256": sha256_file(args.gold),
        "visual_asset_policy": "crop_original_pixels_only_no_redraw",
    }
    write_json(args.status, status)
    summary = {
        "schema_version": "gaokao_physics_ocr_gold_summary_v1",
        "dataset": DATASET_NAME,
        "dataset_root": str(dataset_root),
        "source_manifest_sha256": sha256_file(dataset_root / "manifest.jsonl"),
        "source_generator_gold_sha256": sha256_file(dataset_root / "gold" / "gold.jsonl"),
        "source_records": len(queue),
        "benchmark_eligible": len(queue),
        "subject_counts": dict(Counter(row["subject"] for row in queue)),
        "question_type_counts": dict(Counter(row["content"]["question_type"] for row in gold)),
        "source_generated_verified_ocr_gold_count": len(gold),
        "formal_ocr_scoring_enabled": True,
        "benchmark_scope": status["benchmark_scope"],
        "real_world_generalization_claim_allowed": False,
        "answer_pdf_used_as_ocr_gold": False,
        "answer_solution_fields_copied_to_ocr_gold": False,
        "answer_solution_human_review_complete": False,
        "visual_records": sum(bool(row["assets"]) for row in visuals),
        "visual_assets": sum(len(row["assets"]) for row in visuals),
        "visual_asset_policy": status["visual_asset_policy"],
        "optional_review_tasks_created_this_run": created,
        "optional_review_tasks_preserved_this_run": preserved,
        "optional_review_tasks_source_refreshed_this_run": refreshed_pending_sources,
    }
    write_json(args.summary, summary)
    report = validate_outputs(dataset_root, args.queue, args.gold, args.visuals, args.status)
    report["summary"] = summary
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
