from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_ROOT = PROJECT_ROOT / "artifacts" / "runs"


def _run_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path.resolve()
    candidate = DEFAULT_RUN_ROOT / path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Run not found: {value}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_text(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _reports(run: Path) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    found: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for path in run.glob("*/report.json"):
        report = _read_json(path)
        sample_id = str(report.get("sample_id") or path.parent.name)
        document_path = Path(str(report["question_document"]))
        document = _read_json(document_path)
        found[sample_id] = report, document
    return found


def compare_runs(
    baseline_run: Path,
    candidate_run: Path,
    *,
    minimum_similarity: float = 0.90,
    minimum_char_ratio: float = 0.90,
    maximum_char_ratio: float = 1.10,
    latency_target_seconds: float = 5.0,
) -> dict[str, Any]:
    baseline = _reports(baseline_run)
    candidate = _reports(candidate_run)
    sample_ids = sorted(set(baseline) & set(candidate))
    rows: list[dict[str, Any]] = []
    regression_count = 0
    for sample_id in sample_ids:
        baseline_report, baseline_document = baseline[sample_id]
        candidate_report, candidate_document = candidate[sample_id]
        baseline_text = _normalized_text(baseline_document.get("display_text"))
        candidate_text = _normalized_text(candidate_document.get("display_text"))
        similarity = SequenceMatcher(None, baseline_text, candidate_text).ratio()
        char_ratio = len(candidate_text) / max(1, len(baseline_text))
        formula_delta = int(candidate_document.get("formula_count") or 0) - int(
            baseline_document.get("formula_count") or 0
        )
        visual_delta = len(candidate_document.get("visual_assets") or []) - len(
            baseline_document.get("visual_assets") or []
        )
        candidate_status = str(candidate_report.get("status") or "")
        baseline_status = str(baseline_report.get("status") or "")
        latency = float(candidate_report.get("elapsed_seconds") or 0.0)
        latency_sla_applicable = bool(
            (candidate_report.get("measurement") or {}).get(
                "online_latency_sla_applicable", True
            )
        )
        reasons: list[str] = []
        if similarity < minimum_similarity:
            reasons.append("TEXT_SIMILARITY_LOW")
        if not minimum_char_ratio <= char_ratio <= maximum_char_ratio:
            reasons.append("TEXT_LENGTH_DRIFT")
        if candidate_status == "REJECT" and baseline_status != "REJECT":
            reasons.append("NEW_REJECT")
        if baseline_status == "REJECT" and candidate_status != "REJECT":
            reasons.append("SAFETY_GATE_RELAXED")
        if formula_delta:
            reasons.append("FORMULA_COUNT_CHANGED")
        # A visual-count change is evidence to inspect, not automatically a
        # regression: combining a labeled strip into one crop is intentional.
        review_signals = []
        if visual_delta:
            review_signals.append("VISUAL_COUNT_CHANGED")
        if latency_sla_applicable and latency > latency_target_seconds:
            review_signals.append("LATENCY_OVER_TARGET")
        if reasons:
            regression_count += 1
        rows.append(
            {
                "sample_id": sample_id,
                "text_similarity": round(similarity, 6),
                "text_char_ratio": round(char_ratio, 6),
                "formula_count_delta": formula_delta,
                "visual_asset_count_delta": visual_delta,
                "baseline_status": baseline_status,
                "candidate_status": candidate_status,
                "candidate_latency_seconds": (
                    round(latency, 6) if latency_sla_applicable else None
                ),
                "candidate_latency_sla_applicable": latency_sla_applicable,
                "regression_reasons": reasons,
                "review_signals": review_signals,
            }
        )

    similarities = [row["text_similarity"] for row in rows]
    latencies = [
        row["candidate_latency_seconds"]
        for row in rows
        if row["candidate_latency_sla_applicable"]
    ]
    return {
        "schema_version": "behavior_run_comparison_v1",
        "formal_gold_scoring": False,
        "warning": (
            "Behavioral comparison detects changes against another pipeline run; "
            "it does not measure OCR accuracy and cannot replace verified OCR Gold."
        ),
        "baseline_run": str(baseline_run),
        "candidate_run": str(candidate_run),
        "baseline_sample_count": len(baseline),
        "candidate_sample_count": len(candidate),
        "compared_sample_count": len(rows),
        "baseline_only_sample_count": len(set(baseline) - set(candidate)),
        "candidate_missing_from_baseline": sorted(set(candidate) - set(baseline)),
        "thresholds": {
            "minimum_text_similarity": minimum_similarity,
            "minimum_text_char_ratio": minimum_char_ratio,
            "maximum_text_char_ratio": maximum_char_ratio,
            "latency_target_seconds": latency_target_seconds,
        },
        "summary": {
            "regression_count": regression_count,
            "minimum_text_similarity": min(similarities) if similarities else None,
            "mean_text_similarity": (
                round(sum(similarities) / len(similarities), 6) if similarities else None
            ),
            "latency_under_target_count": sum(
                latency <= latency_target_seconds for latency in latencies
            ),
            "latency_sample_count": len(latencies),
            "maximum_latency_seconds": max(latencies) if latencies else None,
        },
        "samples": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare pipeline behavior without claiming formal OCR accuracy"
    )
    parser.add_argument("baseline_run")
    parser.add_argument("candidate_run")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-similarity", type=float, default=0.90)
    parser.add_argument("--minimum-char-ratio", type=float, default=0.90)
    parser.add_argument("--maximum-char-ratio", type=float, default=1.10)
    parser.add_argument("--latency-target-seconds", type=float, default=5.0)
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    result = compare_runs(
        _run_path(args.baseline_run),
        _run_path(args.candidate_run),
        minimum_similarity=args.minimum_similarity,
        minimum_char_ratio=args.minimum_char_ratio,
        maximum_char_ratio=args.maximum_char_ratio,
        latency_target_seconds=args.latency_target_seconds,
    )
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 1 if args.fail_on_regression and result["summary"]["regression_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
