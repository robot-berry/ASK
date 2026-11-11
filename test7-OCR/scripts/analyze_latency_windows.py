from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metrics(values: list[float], target_seconds: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mean": round(mean(values), 6) if values else None,
        "p50": round(percentile(values, 0.50), 6) if values else None,
        "p95": round(percentile(values, 0.95), 6) if values else None,
        "max": round(max(values), 6) if values else None,
    }
    if target_seconds is not None:
        under = sum(value <= target_seconds for value in values)
        result.update(
            {
                "target": target_seconds,
                "under_target_count": under,
                "under_target_ratio": round(under / len(values), 6) if values else None,
            }
        )
    return result


def analyze(run_directory: Path, window_size: int, target_seconds: float) -> dict[str, Any]:
    if window_size < 1:
        raise ValueError("window_size must be positive")
    index_path = run_directory / "index.jsonl"
    reports: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                reports.append(json.loads(line))

    windows: list[dict[str, Any]] = []
    for start in range(0, len(reports), window_size):
        group = reports[start : start + window_size]
        elapsed = [float(item["elapsed_seconds"]) for item in group]
        recognition = [
            float((item.get("timings_ms") or {}).get("recognition", 0.0)) / 1000
            for item in group
        ]
        service = [
            float(((item.get("recognition") or {}).get("diagnostics") or {}).get(
                "structured_service_elapsed_ms", 0.0
            ))
            / 1000
            for item in group
            if ((item.get("recognition") or {}).get("diagnostics") or {}).get(
                "structured_service_elapsed_ms"
            )
            is not None
        ]
        windows.append(
            {
                "start_ordinal": start + 1,
                "end_ordinal": start + len(group),
                "first_sample_id": group[0]["sample_id"],
                "last_sample_id": group[-1]["sample_id"],
                "sample_count": len(group),
                "elapsed_seconds": metrics(elapsed, target_seconds),
                "recognition_seconds": metrics(recognition),
                "structured_service_seconds": metrics(service),
            }
        )

    return {
        "schema_version": "latency_window_analysis_v1",
        "run_id": run_directory.name,
        "index": str(index_path.resolve()),
        "sample_count": len(reports),
        "window_size": window_size,
        "target_seconds": target_seconds,
        "windows": windows,
        "interpretation_guardrail": (
            "This report shows temporal correlation only. It does not prove which process "
            "caused contention unless independent process/service evidence is also recorded."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize OCR latency in processing-order windows")
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--window-size", type=int, default=25)
    parser.add_argument("--target-seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = analyze(args.run_directory.resolve(), args.window_size, args.target_seconds)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, output)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
