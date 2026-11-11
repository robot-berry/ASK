from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

from .external_datasets import PROJECT_ROOT
from .storage import write_json_atomic


FORBIDDEN = (
    "Do not copy these rates into pipeline.json thresholds, prompts, or "
    "PASS/REVIEW rules. External frozen buckets are evaluation_only."
)

DEFAULT_RUNS = (
    {
        "dataset_id": "omnidocbench",
        "label": "OmniDocBench exam_zh",
        "run_directory": PROJECT_ROOT / "artifacts" / "runs" / "omnidocbench_exam_zh_v1_external",
        "eval_report": PROJECT_ROOT
        / "artifacts"
        / "external"
        / "omnidocbench"
        / "frozen_v1"
        / "report.json",
        "path_mismatch": False,
    },
    {
        "dataset_id": "unimer_test",
        "label": "UniMER-Test stratified400",
        "run_directory": PROJECT_ROOT / "artifacts" / "runs" / "unimer_test_stratified400_v1_external",
        "eval_report": PROJECT_ROOT / "artifacts" / "external" / "unimer_test" / "frozen_v1" / "report.json",
        "path_mismatch": True,
        "path_mismatch_note": (
            "Gold is formula-only LaTeX; the product pipeline treats each crop as a "
            "whole question. REJECT rate and formula_match ≈ 0.10 measure path mismatch, "
            "not formula-engine accuracy."
        ),
    },
    {
        "dataset_id": "mathvision_wild",
        "label": "MathVision-Wild frozen_test_photo",
        "run_directory": PROJECT_ROOT / "artifacts" / "runs" / "mathvision_wild_frozen_v1_external",
        "eval_report": PROJECT_ROOT
        / "artifacts"
        / "external"
        / "mathvision_wild"
        / "frozen_v1"
        / "report.json",
        "path_mismatch": False,
    },
)


def classify_failure(error: str) -> str:
    text = str(error or "")
    if "MemoryError" in text:
        return "huge_image_oom"
    if "Empty crop after clamping" in text:
        return "empty_crop_bbox"
    if "truncated" in text.lower():
        return "truncated_image"
    return "other"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_sample_reports(run_directory: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if not run_directory.is_dir():
        return reports
    for report_path in sorted(run_directory.glob("*/report.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not payload.get("sample_id"):
            continue
        reports.append(payload)
    return reports


def _primary_reject_code(issues: list[dict[str, Any]]) -> str | None:
    reject_codes = [
        str(issue.get("code") or "")
        for issue in issues
        if str(issue.get("severity") or "").lower() == "reject" and issue.get("code")
    ]
    return reject_codes[0] if reject_codes else None


def _quartile_edges(values: list[float]) -> tuple[float, float, float] | None:
    if len(values) < 4:
        return None
    ordered = sorted(values)
    def at(ratio: float) -> float:
        position = (len(ordered) - 1) * ratio
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    return at(0.25), at(0.5), at(0.75)


def diagnose_run(
    *,
    dataset_id: str,
    label: str,
    run_directory: Path,
    eval_report: Path,
    path_mismatch: bool = False,
    path_mismatch_note: str | None = None,
) -> dict[str, Any]:
    summary_path = run_directory / "batch_summary.json"
    batch = _load_json(summary_path) if summary_path.is_file() else {}
    failures = list(batch.get("failures") or [])
    crash_groups: dict[str, list[str]] = defaultdict(list)
    for row in failures:
        crash_groups[classify_failure(str(row.get("error") or ""))].append(str(row.get("sample_id") or ""))
    sample_reports = _iter_sample_reports(run_directory)
    status_counts = Counter(str(row.get("status") or "UNKNOWN") for row in sample_reports)
    code_occurrences: Counter[str] = Counter()
    code_samples: dict[str, set[str]] = defaultdict(set)
    code_severity: dict[str, Counter[str]] = defaultdict(Counter)
    reject_driver_samples: dict[str, set[str]] = defaultdict(set)
    primary_reject: Counter[str] = Counter()
    for row in sample_reports:
        sample_id = str(row.get("sample_id") or "")
        issues = [issue for issue in (row.get("issues") or []) if isinstance(issue, dict)]
        seen: set[str] = set()
        for issue in issues:
            code = str(issue.get("code") or "")
            if not code:
                continue
            code_occurrences[code] += 1
            code_samples[code].add(sample_id)
            severity = str(issue.get("severity") or "unknown")
            code_severity[code][severity] += 1
            if severity == "reject":
                reject_driver_samples[code].add(sample_id)
            seen.add(code)
        if str(row.get("status") or "") == "REJECT":
            primary = _primary_reject_code(issues)
            if primary:
                primary_reject[primary] += 1
            else:
                primary_reject["NO_REJECT_CODE"] += 1

    issue_table = []
    for code, occurrences in code_occurrences.most_common():
        issue_table.append(
            {
                "code": code,
                "occurrences": occurrences,
                "samples": len(code_samples[code]),
                "severity_counts": dict(code_severity[code]),
                "reject_driver_samples": len(reject_driver_samples.get(code, ())),
            }
        )

    eval_payload = _load_json(eval_report) if eval_report.is_file() else {}
    details = [row for row in (eval_payload.get("details") or []) if isinstance(row, dict) and row.get("evaluated")]
    cer_values = [float(row["character_error_rate"]) for row in details if row.get("character_error_rate") is not None]
    quartiles = None
    edges = _quartile_edges(cer_values)
    if edges:
        q1, q2, q3 = edges
        buckets = {"q1_low_cer": [], "q2": [], "q3": [], "q4_high_cer": []}
        for row in details:
            cer = row.get("character_error_rate")
            if cer is None:
                continue
            value = float(cer)
            if value <= q1:
                buckets["q1_low_cer"].append(row)
            elif value <= q2:
                buckets["q2"].append(row)
            elif value <= q3:
                buckets["q3"].append(row)
            else:
                buckets["q4_high_cer"].append(row)
        quartiles = {
            "cer_edges": {"q1": round(q1, 6), "q2": round(q2, 6), "q3": round(q3, 6)},
            "buckets": {
                name: {
                    "samples": len(rows),
                    "mean_cer": round(fmean(float(row["character_error_rate"]) for row in rows), 6) if rows else None,
                    "mean_token_coverage": round(
                        fmean(float(row["token_coverage"]) for row in rows if row.get("token_coverage") is not None),
                        6,
                    )
                    if any(row.get("token_coverage") is not None for row in rows)
                    else None,
                    "mean_formula_match_rate": round(
                        fmean(float(row["formula_match_rate"]) for row in rows if row.get("formula_match_rate") is not None),
                        6,
                    )
                    if any(row.get("formula_match_rate") is not None for row in rows)
                    else None,
                }
                for name, rows in buckets.items()
            },
        }
    high_formula_high_cer = [
        str(row["sample_id"])
        for row in details
        if dataset_id != "mathvision_wild"
        and row.get("formula_match_rate") is not None
        and row.get("character_error_rate") is not None
        and float(row["formula_match_rate"]) >= 0.9
        and float(row["character_error_rate"]) >= 0.5
    ]
    metrics = eval_payload.get("metrics") or {}
    if dataset_id == "mathvision_wild" and metrics:
        metrics = {
            **metrics,
            "mean_formula_match_rate_not_applicable": True,
            "mean_formula_match_rate_note": "Gold has no formulas; 1.0 is the empty-gold default, not formula accuracy.",
        }
    return {
        "dataset_id": dataset_id,
        "label": label,
        "run_directory": str(run_directory),
        "formal_gold_scoring": False,
        "threshold_calibration_allowed": False,
        "path_mismatch": path_mismatch,
        "path_mismatch_note": path_mismatch_note,
        "attempted": batch.get("attempted"),
        "completed_sample_count": batch.get("completed_sample_count") or len(sample_reports),
        "status_counts": dict(sorted((batch.get("statuses") or status_counts).items())),
        "crash_groups": {
            kind: {"count": len(ids), "sample_ids": ids[:20]}
            for kind, ids in sorted(crash_groups.items(), key=lambda item: (-len(item[1]), item[0]))
        },
        "issue_codes": issue_table,
        "primary_reject_codes": [
            {"code": code, "samples": count} for code, count in primary_reject.most_common()
        ],
        "isolated_eval": {
            "evaluated_count": eval_payload.get("evaluated_count"),
            "missing_count": eval_payload.get("missing_count"),
            "metrics": metrics,
        },
        "cer_quartile_crosstab": quartiles,
        "high_formula_match_high_cer": {
            "count": len(high_formula_high_cer),
            "rule": "formula_match_rate >= 0.9 and character_error_rate >= 0.5",
            "example_sample_ids": high_formula_high_cer[:8],
        },
        "warning": FORBIDDEN,
    }


def assemble_diagnosis(runs: tuple[dict[str, Any], ...] | None = None) -> dict[str, Any]:
    datasets = []
    for spec in runs or DEFAULT_RUNS:
        datasets.append(
            diagnose_run(
                dataset_id=str(spec["dataset_id"]),
                label=str(spec["label"]),
                run_directory=Path(spec["run_directory"]),
                eval_report=Path(spec["eval_report"]),
                path_mismatch=bool(spec.get("path_mismatch")),
                path_mismatch_note=spec.get("path_mismatch_note"),
            )
        )
    next_rounds = [
        {
            "id": "engineering_hardening",
            "title": "工程加固",
            "not_implemented_this_round": False,
            "status": "implemented",
            "candidates": ["huge_image_oom", "empty_crop_bbox", "truncated_image"],
            "note": "Fix crashes without changing PASS/REVIEW thresholds.",
        },
        {
            "id": "formula_only_path",
            "title": "公式裁图产品路径",
            "not_implemented_this_round": False,
            "status": "implemented",
            "note": "Accept on gaokao Gold / non-frozen dev data; UniMER remains a frozen retest only.",
        },
        {
            "id": "page_text_quality",
            "title": "试卷正文识别",
            "not_implemented_this_round": True,
            "note": "Tune only on non-frozen development data; OmniDoc exam_zh is retest-only.",
        },
    ]
    payload = {
        "schema_version": "external_diagnosis_v1",
        "formal_gold_scoring": False,
        "real_world_generalization_claim_allowed": False,
        "threshold_calibration_allowed": False,
        "primary_ocr_gold": "gaokao_physics_300_v1",
        "warning": FORBIDDEN,
        "datasets": datasets,
        "next_round_candidates": next_rounds,
    }
    output = PROJECT_ROOT / "artifacts" / "external" / "DIAGNOSIS.json"
    write_json_atomic(output, payload)
    payload["report_path"] = str(output)
    return payload
