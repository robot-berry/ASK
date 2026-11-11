from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from statistics import fmean
from typing import Any

from .storage import write_json_atomic
from .text import contains_latex_source, latex_to_readable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", latex_to_readable(value or ""))
    value = value.replace("，", ",").replace("。", ".").replace("：", ":")
    return re.sub(r"\s+", "", value).lower()


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(prediction: str, gold: str) -> float:
    expected = normalize_text(gold)
    actual = normalize_text(prediction)
    return edit_distance(actual, expected) / max(1, len(expected))


def bbox_iou(left: list[int], right: list[int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection_left = max(lx, rx)
    intersection_top = max(ly, ry)
    intersection_right = min(lx + lw, rx + rw)
    intersection_bottom = min(ly + lh, ry + rh)
    intersection = max(0, intersection_right - intersection_left) * max(
        0, intersection_bottom - intersection_top
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


def _prediction_bbox(asset: dict[str, Any]) -> list[int]:
    value = asset["source_bbox"]
    if isinstance(value, dict):
        return [int(value[key]) for key in ("x", "y", "width", "height")]
    return [int(part) for part in value]


def _visual_recall(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> float:
    if not gold:
        return 1.0
    predicted_boxes = [_prediction_bbox(asset) for asset in predicted]
    hits = 0
    for expected in gold:
        expected_box = [int(part) for part in expected["source_bbox"]]
        if any(bbox_iou(candidate, expected_box) >= 0.80 for candidate in predicted_boxes):
            hits += 1
    return hits / len(gold)


def _visual_precision(predicted: list[dict[str, Any]], gold: list[dict[str, Any]]) -> float:
    if not predicted:
        return 1.0 if not gold else 0.0
    expected_boxes = [[int(part) for part in item["source_bbox"]] for item in gold]
    hits = 0
    for asset in predicted:
        candidate = _prediction_bbox(asset)
        if any(bbox_iou(candidate, expected) >= 0.80 for expected in expected_boxes):
            hits += 1
    return hits / len(predicted)


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _source_provenance_pass(document: dict[str, Any]) -> bool:
    blocks = document.get("source_blocks")
    if not isinstance(blocks, list) or not blocks:
        return False
    for block in blocks:
        if not isinstance(block, dict):
            return False
        disposition = block.get("disposition")
        if disposition not in {"emitted", "ignored"}:
            return False
        if disposition == "ignored" and not str(block.get("reason") or "").strip():
            return False
        bbox = block.get("source_bbox")
        if not isinstance(bbox, dict):
            return False
        try:
            if int(bbox["x"]) < 0 or int(bbox["y"]) < 0:
                return False
            if int(bbox["width"]) <= 0 or int(bbox["height"]) <= 0:
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _report_index_path(run_directory: Path) -> Path:
    finalized = run_directory / "index.finalized.jsonl"
    return finalized if finalized.is_file() else run_directory / "index.jsonl"


def _has_answer_slot(value: str) -> bool:
    return bool(
        re.search(
            r"(?:_{2,}|—{2,}|…{2,}|□|\[\s*填?空?\s*\]|[（(]\s*[）)])",
            value or "",
        )
    )


def _gold_verification_method(row: dict[str, Any]) -> str | None:
    """Return the accepted verification basis without inventing human sign-offs."""
    review = row.get("review") or {}
    if (
        review.get("status") == "verified"
        and int(review.get("independent_review_count", 0)) >= 2
        and int(review.get("required_independent_review_count", 2)) == 2
        and review.get("source_alignment") == "human_verified"
    ):
        return "human_double_review"
    if (
        review.get("status") == "source_generated_verified"
        and int(review.get("independent_review_count", -1)) == 0
        and int(review.get("required_independent_review_count", -1)) == 0
        and review.get("source_alignment") == "generator_fact_source"
        and (row.get("handwriting") or {}).get("gold_status") == "synthetic_clean_source"
    ):
        return "generator_fact_source"
    return None


def evaluate_run(
    run_directory: Path,
    gold_path: Path,
    *,
    required_gold_count: int = 300,
    allow_partial_gold: bool = False,
) -> dict[str, Any]:
    gold_rows = read_jsonl(gold_path)
    unique_gold = {str(row["sample_id"]): row for row in gold_rows}
    verification_methods = {
        sample_id: _gold_verification_method(row)
        for sample_id, row in unique_gold.items()
    }
    verified = [
        row
        for sample_id, row in unique_gold.items()
        if verification_methods[sample_id] is not None
    ]
    formal_ready = len(unique_gold) == required_gold_count and len(verified) == required_gold_count
    if not formal_ready and not allow_partial_gold:
        raise RuntimeError(
            f"Formal scoring locked: {len(verified)}/{required_gold_count} verified Gold records."
        )

    report_rows = read_jsonl(_report_index_path(run_directory))
    latest_reports = {str(row["sample_id"]): row for row in report_rows}
    details: list[dict[str, Any]] = []
    for sample_id, gold in unique_gold.items():
        report = latest_reports.get(sample_id)
        if report is None:
            details.append({"sample_id": sample_id, "evaluated": False, "reason": "prediction_missing"})
            continue
        document_path = Path(report["question_document"])
        if not document_path.exists():
            details.append({"sample_id": sample_id, "evaluated": False, "reason": "document_missing"})
            continue
        document = json.loads(document_path.read_text(encoding="utf-8"))
        gold_text = str(gold["content"]["semantic_text"])
        predicted_text = str(document.get("display_text") or "")
        expected_options = gold["content"].get("options", [])
        predicted_options = document.get("options", [])
        option_exact = [
            (option.get("label"), normalize_text(str(option.get("text", ""))))
            for option in predicted_options
        ] == [
            (option.get("label"), normalize_text(str(option.get("text", ""))))
            for option in expected_options
        ]
        predicted_assets = document.get("visual_assets", [])
        gold_assets = gold.get("visual_assets", [])
        visual_recall = _visual_recall(predicted_assets, gold_assets)
        visual_precision = _visual_precision(predicted_assets, gold_assets)
        visual_policy_pass = all(
            asset.get("extraction") == "direct_source_crop"
            and asset.get("redraw_used") is False
            and asset.get("preserve_source_pixels") is True
            and asset.get("edge_complete") is True
            for asset in predicted_assets
        )
        normalized_exact = normalize_text(predicted_text) == normalize_text(gold_text)
        answer_slot_preserved = not _has_answer_slot(gold_text) or _has_answer_slot(predicted_text)
        stem_exact = normalize_text(str(document.get("stem") or "")) == normalize_text(
            str(gold["content"].get("stem_text") or "")
        )
        readback = report.get("render_readback") or {}
        render_readback_pass = readback.get("enabled") is True and readback.get("passed") is True
        source_provenance_pass = _source_provenance_pass(document)
        latex_leakage = contains_latex_source(predicted_text)
        automatic_pass = report.get("status") == "PASS"
        formula_expected = str(gold["content"].get("formula_gold_status") or "") in {
            "generator_source_verified",
            "human_transcribed",
            "latex_verified",
        }
        critical_checks = [
            normalized_exact,
            answer_slot_preserved,
            stem_exact,
            option_exact,
            visual_recall == 1.0,
            visual_precision == 1.0,
            visual_policy_pass,
            render_readback_pass,
            source_provenance_pass,
            not latex_leakage,
            int(report.get("output_width", 0)) == 800,
        ]
        details.append(
            {
                "sample_id": sample_id,
                "evaluated": True,
                "status": report.get("status"),
                "automatic_pass": automatic_pass,
                "normalized_exact": normalized_exact,
                "answer_slot_preserved": answer_slot_preserved,
                "stem_exact": stem_exact,
                "character_error_rate": round(character_error_rate(predicted_text, gold_text), 6),
                "options_exact": option_exact,
                "formula_expected": formula_expected,
                "formula_semantic_exact": normalized_exact if formula_expected else True,
                "visual_recall_iou_0_80": round(visual_recall, 6),
                "visual_precision_iou_0_80": round(visual_precision, 6),
                "visual_policy_pass": visual_policy_pass,
                "render_readback_pass": render_readback_pass,
                "source_provenance_pass": source_provenance_pass,
                "latex_leakage": latex_leakage,
                "handwriting_expected": bool(gold.get("handwriting", {}).get("expected")),
                "handwriting_guarded": bool(document.get("handwriting_detected"))
                or report.get("status") != "PASS",
                "silent_critical_error": automatic_pass and not all(critical_checks),
                "latency_seconds": float(report.get("elapsed_seconds", 0.0)),
                "width_800": int(report.get("output_width", 0)) == 800,
            }
        )

    evaluated = [row for row in details if row.get("evaluated")]
    missing = len(details) - len(evaluated)
    latency_values = [row["latency_seconds"] for row in evaluated]
    formula_rows = [row for row in evaluated if row["formula_expected"]]
    metrics = {
        "normalized_exact_rate": round(fmean(row["normalized_exact"] for row in evaluated), 6) if evaluated else None,
        "answer_slot_preservation_rate": round(fmean(row["answer_slot_preserved"] for row in evaluated), 6) if evaluated else None,
        "stem_exact_rate": round(fmean(row["stem_exact"] for row in evaluated), 6) if evaluated else None,
        "mean_character_error_rate": round(fmean(row["character_error_rate"] for row in evaluated), 6) if evaluated else None,
        "options_exact_rate": round(fmean(row["options_exact"] for row in evaluated), 6) if evaluated else None,
        "formula_semantic_exact_rate": round(
            fmean(row["formula_semantic_exact"] for row in formula_rows), 6
        ) if formula_rows else None,
        "mean_visual_recall_iou_0_80": round(fmean(row["visual_recall_iou_0_80"] for row in evaluated), 6) if evaluated else None,
        "mean_visual_precision_iou_0_80": round(fmean(row["visual_precision_iou_0_80"] for row in evaluated), 6) if evaluated else None,
        "visual_policy_pass_rate": round(fmean(row["visual_policy_pass"] for row in evaluated), 6) if evaluated else None,
        "render_readback_pass_rate": round(fmean(row["render_readback_pass"] for row in evaluated), 6) if evaluated else None,
        "source_provenance_pass_rate": round(fmean(row["source_provenance_pass"] for row in evaluated), 6) if evaluated else None,
        "latex_leakage_rate": round(fmean(row["latex_leakage"] for row in evaluated), 6) if evaluated else None,
        "handwriting_guard_rate": round(
            fmean(row["handwriting_guarded"] for row in evaluated if row["handwriting_expected"]), 6
        ) if any(row["handwriting_expected"] for row in evaluated) else None,
        "automatic_pass_coverage": round(fmean(row["automatic_pass"] for row in evaluated), 6) if evaluated else None,
        "silent_critical_error_rate": round(fmean(row["silent_critical_error"] for row in evaluated), 6) if evaluated else None,
        "width_800_rate": round(fmean(row["width_800"] for row in evaluated), 6) if evaluated else None,
        "latency_under_5s_rate": round(fmean(value <= 5.0 for value in latency_values), 6) if evaluated else None,
        "mean_latency_seconds": round(fmean(latency_values), 6) if evaluated else None,
        "p95_latency_seconds": round(_percentile(latency_values, 0.95), 6) if evaluated else None,
    }
    gates = {
        "gold_complete": formal_ready,
        "predictions_complete": bool(evaluated) and missing == 0,
        "normalized_exact_at_least_99_9pct": metrics["normalized_exact_rate"] is not None and metrics["normalized_exact_rate"] >= 0.999,
        "answer_slot_preservation_100pct": metrics["answer_slot_preservation_rate"] == 1.0,
        "options_exact_100pct": metrics["options_exact_rate"] == 1.0,
        "formula_semantic_exact_at_least_99_9pct": metrics["formula_semantic_exact_rate"] is None or metrics["formula_semantic_exact_rate"] >= 0.999,
        "visual_recall_at_least_99pct": metrics["mean_visual_recall_iou_0_80"] is not None and metrics["mean_visual_recall_iou_0_80"] >= 0.99,
        "visual_precision_at_least_99pct": metrics["mean_visual_precision_iou_0_80"] is not None and metrics["mean_visual_precision_iou_0_80"] >= 0.99,
        "visual_policy_100pct": metrics["visual_policy_pass_rate"] == 1.0,
        "render_readback_100pct": metrics["render_readback_pass_rate"] == 1.0,
        "source_provenance_100pct": metrics["source_provenance_pass_rate"] == 1.0,
        "latex_leakage_zero": metrics["latex_leakage_rate"] == 0.0,
        "handwriting_guard_100pct": metrics["handwriting_guard_rate"] is None or metrics["handwriting_guard_rate"] == 1.0,
        "silent_critical_error_at_most_0_1pct": metrics["silent_critical_error_rate"] is not None and metrics["silent_critical_error_rate"] <= 0.001,
        "automatic_pass_coverage_at_least_80pct": metrics["automatic_pass_coverage"] is not None and metrics["automatic_pass_coverage"] >= 0.80,
        "width_800_100pct": metrics["width_800_rate"] == 1.0,
        "p95_latency_at_most_5s": metrics["p95_latency_seconds"] is not None and metrics["p95_latency_seconds"] <= 5.0,
    }
    summary = {
        "schema_version": "evaluation_summary_v1",
        "formal_scoring": formal_ready and missing == 0,
        "diagnostic_partial_scoring": not (formal_ready and missing == 0),
        "gold_path": str(gold_path.resolve()),
        "verified_gold_count": len(verified),
        "gold_verification_methods": dict(
            sorted(
                {
                    method: sum(value == method for value in verification_methods.values())
                    for method in set(verification_methods.values())
                    if method is not None
                }.items()
            )
        ),
        "benchmark_scope": (
            "synthetic_source_derived"
            if verified and all(
                verification_methods.get(str(row["sample_id"])) == "generator_fact_source"
                for row in verified
            )
            else "human_reviewed_or_mixed"
        ),
        "real_world_generalization_claim_allowed": False
        if verified and all(
            verification_methods.get(str(row["sample_id"])) == "generator_fact_source"
            for row in verified
        )
        else None,
        "required_gold_count": required_gold_count,
        "prediction_count": len(latest_reports),
        "evaluated_count": len(evaluated),
        "missing_prediction_count": missing,
        "metrics": metrics,
        "release_gates": gates,
        "release_gate_passed": all(gates.values()),
        "details": details,
    }
    write_json_atomic(run_directory / "evaluation.json", summary)
    return summary
