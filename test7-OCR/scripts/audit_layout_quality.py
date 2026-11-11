from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 4)


def audit_layout_quality(
    run_directory: Path,
    *,
    internal_fill_threshold: float = 0.50,
) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(run_directory)
    if not 0 <= internal_fill_threshold <= 1:
        raise ValueError("internal fill threshold must be between 0 and 1")

    candidates: list[dict[str, Any]] = []
    missing_metric_samples: list[str] = []
    fallback_samples: list[str] = []
    pillow_samples: list[str] = []
    measured: list[dict[str, Any]] = []
    reports = sorted(run_directory.glob("*/report.json"))

    for report_path in reports:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        sample_id = str(payload.get("sample_id") or report_path.parent.name)
        render = payload.get("render") or {}
        engine = str(render.get("engine") or "unknown")
        if "fallback" in engine:
            fallback_samples.append(sample_id)
            continue
        if engine == "pillow":
            pillow_samples.append(sample_id)
            continue
        if engine != "edge-katex":
            missing_metric_samples.append(sample_id)
            continue

        required = (
            "stem_line_count",
            "stem_min_line_fill_ratio",
            "stem_min_internal_line_fill_ratio",
            "stem_text_wrap_applied",
        )
        if any(key not in render for key in required):
            missing_metric_samples.append(sample_id)
            continue
        row = {
            "sample_id": sample_id,
            "status": str(payload.get("status") or "UNKNOWN"),
            "stem_line_count": int(render["stem_line_count"]),
            "stem_min_line_fill_ratio": float(render["stem_min_line_fill_ratio"]),
            "stem_min_internal_line_fill_ratio": float(
                render["stem_min_internal_line_fill_ratio"]
            ),
            "stem_text_wrap_applied": str(render["stem_text_wrap_applied"]),
            "output_image": str(payload.get("output_image") or ""),
        }
        measured.append(row)
        if row["stem_min_internal_line_fill_ratio"] < internal_fill_threshold:
            candidates.append(row)

    candidates.sort(
        key=lambda row: (
            row["stem_min_internal_line_fill_ratio"],
            row["sample_id"],
        )
    )
    ratios = [row["stem_min_internal_line_fill_ratio"] for row in measured]
    modes: dict[str, int] = {}
    for row in measured:
        mode = row["stem_text_wrap_applied"]
        modes[mode] = modes.get(mode, 0) + 1

    valid = not missing_metric_samples and not fallback_samples
    return {
        "schema_version": "layout_quality_audit_v1",
        "run_id": run_directory.name,
        "run_directory": str(run_directory),
        "formal_gold_scoring": False,
        "visual_layout_qa_only": True,
        "passed": valid,
        "report_count": len(reports),
        "measured_edge_katex_count": len(measured),
        "pillow_unmeasured_count": len(pillow_samples),
        "fallback_count": len(fallback_samples),
        "missing_metric_count": len(missing_metric_samples),
        "internal_fill_threshold": internal_fill_threshold,
        "internal_fill_statistics": {
            "minimum": round(min(ratios), 4) if ratios else None,
            "p05": _percentile(ratios, 0.05),
            "p50": _percentile(ratios, 0.50),
            "p95": _percentile(ratios, 0.95),
        },
        "wrap_modes": dict(sorted(modes.items())),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "fallback_samples": sorted(fallback_samples),
        "missing_metric_samples": sorted(missing_metric_samples),
        "warning": (
            "Low internal-line fill is a visual review signal, not an OCR error. "
            "Pillow-only samples do not expose browser line boxes and are reported separately."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit actual rendered stem line fill without treating final short lines as failures."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--internal-fill-threshold", type=float, default=0.50)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_layout_quality(
        args.run_directory,
        internal_fill_threshold=args.internal_fill_threshold,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
