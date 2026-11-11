from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from .evaluation import character_error_rate, normalize_text, read_jsonl
from .external_datasets import resolve_project_path
from .storage import write_json_atomic
from .text import contains_latex_source, latex_to_readable, formulas_agree


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _token_set(value: str) -> set[str]:
    readable = normalize_text(latex_to_readable(value or ""))
    return {token for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", readable) if token}


def _coverage(predicted: str, gold: str) -> float:
    expected = _token_set(gold)
    if not expected:
        return 1.0
    actual = _token_set(predicted)
    return len(expected & actual) / len(expected)


def _formula_match(predicted_formulas: list[str], gold_formulas: list[str]) -> float:
    if not gold_formulas:
        return 1.0
    hits = 0
    for gold in gold_formulas:
        if any(formulas_agree(gold, item) for item in predicted_formulas):
            hits += 1
    return hits / len(gold_formulas)


def _document_formulas(document: dict[str, Any]) -> list[str]:
    formulas: list[str] = []
    for block in document.get("source_blocks") or []:
        if not isinstance(block, dict):
            continue
        if str(block.get("kind") or "") in {"formula", "math"}:
            formulas.append(str(block.get("text") or ""))
    return formulas


def evaluate_external_run(
    run_directory: Path,
    gold_path: Path,
    *,
    dataset_id: str,
    scoring_mode: str | None = None,
) -> dict[str, Any]:
    gold_rows = {str(row["sample_id"]): row for row in read_jsonl(gold_path)}
    index_path = run_directory / "index.finalized.jsonl"
    if not index_path.is_file():
        index_path = run_directory / "index.jsonl"
    reports = {str(row["sample_id"]): row for row in read_jsonl(index_path)} if index_path.is_file() else {}
    details: list[dict[str, Any]] = []
    for sample_id, gold in gold_rows.items():
        mode = scoring_mode or str(gold.get("scoring_mode") or "page_text_and_formulas")
        report = reports.get(sample_id)
        if report is None:
            report_path = run_directory / sample_id / "report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
        if report is None:
            details.append({"sample_id": sample_id, "evaluated": False, "reason": "prediction_missing"})
            continue
        document_path = Path(report.get("question_document") or (run_directory / sample_id / "question_document.json"))
        if not document_path.is_file():
            details.append({"sample_id": sample_id, "evaluated": False, "reason": "document_missing"})
            continue
        document = json.loads(document_path.read_text(encoding="utf-8"))
        predicted_text = str(document.get("display_text") or "")
        gold_text = str(gold.get("gold_text") or gold.get("gold_question") or "")
        gold_formulas = [str(item) for item in (gold.get("gold_formulas") or [])]
        options = gold.get("gold_options") or []
        if options:
            option_gold = "\n".join(str(item) for item in options)
            option_coverage = _coverage(predicted_text, option_gold)
        else:
            option_coverage = 1.0
        text_applicable = bool(gold.get("text_cer_applicable", mode != "stability_only"))
        detail = {
            "sample_id": sample_id,
            "evaluated": True,
            "scoring_mode": mode,
            "status": report.get("status"),
            "runtime_failed": bool(report.get("runtime_failed") or report.get("error")),
            "character_error_rate": round(character_error_rate(predicted_text, gold_text), 6)
            if text_applicable and gold_text
            else None,
            "token_coverage": round(_coverage(predicted_text, gold_text), 6) if gold_text else None,
            "question_coverage": round(
                _coverage(predicted_text, str(gold.get("gold_question") or gold_text)), 6
            ),
            "option_coverage": round(option_coverage, 6),
            "formula_match_rate": round(_formula_match(_document_formulas(document), gold_formulas), 6),
            "latex_leakage": contains_latex_source(predicted_text),
            "width_800": int(report.get("output_width") or 0) == 800,
            "latency_seconds": float(report.get("elapsed_seconds") or 0.0),
            "ignore_answer": bool(gold.get("ignore_answer", True)),
        }
        details.append(detail)

    evaluated = [row for row in details if row.get("evaluated")]
    cer_rows = [row["character_error_rate"] for row in evaluated if row.get("character_error_rate") is not None]
    latency = [float(row["latency_seconds"]) for row in evaluated]
    statuses = Counter(str(row.get("status") or "UNKNOWN") for row in evaluated)
    metrics = {
        "mean_character_error_rate": round(fmean(cer_rows), 6) if cer_rows else None,
        "mean_token_coverage": round(
            fmean(row["token_coverage"] for row in evaluated if row.get("token_coverage") is not None), 6
        )
        if any(row.get("token_coverage") is not None for row in evaluated)
        else None,
        "mean_question_coverage": round(fmean(row["question_coverage"] for row in evaluated), 6)
        if evaluated
        else None,
        "mean_option_coverage": round(fmean(row["option_coverage"] for row in evaluated), 6) if evaluated else None,
        "mean_formula_match_rate": round(fmean(row["formula_match_rate"] for row in evaluated), 6)
        if evaluated
        else None,
        "latex_leakage_rate": round(fmean(row["latex_leakage"] for row in evaluated), 6) if evaluated else None,
        "width_800_rate": round(fmean(row["width_800"] for row in evaluated), 6) if evaluated else None,
        "runtime_success_rate": round(fmean(not row["runtime_failed"] for row in evaluated), 6) if evaluated else None,
        "mean_latency_seconds": round(fmean(latency), 6) if latency else None,
        "p95_latency_seconds": round(_percentile(latency, 0.95), 6) if latency else None,
        "status_counts": dict(sorted(statuses.items())),
    }
    text_cer_not_applicable = scoring_mode == "stability_only" or dataset_id in {"warpdoc", "realdae"}
    if text_cer_not_applicable:
        metrics["mean_character_error_rate"] = None
        metrics["text_cer_not_applicable"] = True
    summary = {
        "schema_version": "external_eval_summary_v1",
        "formal_gold_scoring": False,
        "real_world_generalization_claim_allowed": False,
        "dataset_id": dataset_id,
        "run_directory": str(run_directory.resolve()),
        "gold_path": str(resolve_project_path(gold_path)),
        "evaluated_count": len(evaluated),
        "missing_count": len(details) - len(evaluated),
        "metrics": metrics,
        "details": details,
        "warning": (
            "External regression only. Do not merge into gaokao_physics_300 formal "
            "OCR accuracy or calibrate PASS/REVIEW thresholds from these numbers."
        ),
    }
    return summary


def write_external_eval_report(summary: dict[str, Any], output_path: Path) -> Path:
    write_json_atomic(output_path, summary)
    return output_path


def assemble_external_baseline(report_root: Path | None = None) -> dict[str, Any]:
    from datetime import datetime, timezone

    from .external_datasets import PROJECT_ROOT, load_dataset_registry, load_external_config

    report_root = report_root or (PROJECT_ROOT / "artifacts" / "external")
    datasets = {}
    if report_root.is_dir():
        for dataset_dir in sorted(path for path in report_root.iterdir() if path.is_dir()):
            frozen_reports = sorted(dataset_dir.glob("frozen*/report.json"))
            other_reports = [
                path
                for path in sorted(dataset_dir.glob("*/report.json")) + sorted(dataset_dir.glob("report.json"))
                if "harness_smoke" not in path.as_posix()
            ]
            reports = frozen_reports or other_reports or sorted(dataset_dir.glob("harness_smoke/report.json"))
            latest = None
            for path in reports:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                latest = {
                    "path": str(path),
                    **{key: payload[key] for key in payload if key != "details"},
                }
                if "frozen" in path.as_posix().replace("\\", "/"):
                    latest["upstream_frozen_ocr_executed"] = True
                    latest["harness"] = payload.get("harness") or "upstream_frozen_ocr"
            datasets[dataset_dir.name] = latest
    registry = load_dataset_registry()
    external = load_external_config()
    blocked = {
        dataset_id: entry.get("license_status")
        for dataset_id, entry in (external.get("datasets") or {}).items()
        if entry.get("license_status") not in {"available", "available_research_only", "approved"}
    }
    summary = {
        "schema_version": "external_baseline_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_gold_scoring": False,
        "primary_ocr_gold": registry.get("primary_ocr_gold"),
        "real_world_generalization_claim_allowed": False,
        "license_blocked_downloads": blocked,
        "datasets": datasets,
        "warning": (
            "These numbers are isolated external regressions. They do not replace "
            "gaokao_physics_300 formal scoring and must not be used to retune gates."
        ),
    }
    output = report_root / "BASELINE.json"
    write_json_atomic(output, summary)
    summary["report_path"] = str(output)
    return summary
