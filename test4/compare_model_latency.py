#!/usr/bin/env python3
"""Compare end-to-end latency of multiple saved Test4 full-model runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def metrics(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_seconds": round(mean(values), 3) if values else None,
        "median_seconds": round(median(values), 3) if values else None,
        "p90_seconds": round(percentile(values, 0.9), 3) if values else None,
        "min_seconds": round(min(values), 3) if values else None,
        "max_seconds": round(max(values), 3) if values else None,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def locate_job(root: Path, subject: str, case_id: str, item: dict[str, Any]) -> Path:
    job_id = str(item.get("job_id") or "")
    candidates = [
        root / "_jobs" / subject / job_id / "job.json",
        root / subject / case_id / "job.json",
    ]
    result_dir = item.get("result_dir")
    if result_dir:
        candidates.append(Path(str(result_dir)) / "job.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"job.json not found for {case_id} ({job_id})")


def repair_count(job_path: Path, job: dict[str, Any]) -> int:
    audit_dir = job_path.parent / "audit"
    if audit_dir.is_dir():
        files = list(audit_dir.glob("provider_repair_round_*.json"))
        if files:
            return len(files)
    validation = job.get("validation")
    entries = validation.get("repair_audit") if isinstance(validation, dict) else None
    rounds = {
        int(entry["repair_round"])
        for entry in entries or []
        if isinstance(entry, dict) and isinstance(entry.get("repair_round"), int)
    }
    return len(rounds)


def collect(label: str, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in ("math", "physics"):
        progress_path = root / f"progress_{subject}.json"
        if not progress_path.is_file():
            continue
        progress = load_json(progress_path)
        cases = progress.get("cases")
        if not isinstance(cases, dict):
            continue
        for case_id, raw_item in cases.items():
            if not isinstance(raw_item, dict):
                continue
            job_path = locate_job(root, subject, str(case_id), raw_item)
            job = load_json(job_path)
            elapsed = job.get("elapsed_seconds")
            preprocess = job.get("preprocess")
            preprocess_elapsed = (
                preprocess.get("elapsed_seconds") if isinstance(preprocess, dict) else None
            )
            rows.append(
                {
                    "model_label": label,
                    "model_id_requested": job.get("model_id_requested"),
                    "subject": subject,
                    "case_id": str(case_id),
                    "job_id": raw_item.get("job_id"),
                    "status": raw_item.get("status"),
                    "completed": raw_item.get("status") == "completed",
                    "reused": bool(raw_item.get("reused_existing_paid_job")),
                    "repair_requests": repair_count(job_path, job),
                    "elapsed_seconds": float(elapsed) if isinstance(elapsed, (int, float)) else None,
                    "preprocess_seconds": (
                        float(preprocess_elapsed)
                        if isinstance(preprocess_elapsed, (int, float))
                        else None
                    ),
                    "post_preprocess_seconds": (
                        round(float(elapsed) - float(preprocess_elapsed), 3)
                        if isinstance(elapsed, (int, float))
                        and isinstance(preprocess_elapsed, (int, float))
                        else None
                    ),
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="saved run label and root directory; repeat for every model",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    run_roots: dict[str, str] = {}
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"invalid --run value: {spec!r}")
        label, directory = spec.split("=", 1)
        root = Path(directory).resolve()
        run_roots[label] = str(root)
        all_rows.extend(collect(label, root))

    summary: dict[str, Any] = {"runs": run_roots, "models": {}}
    for label in run_roots:
        rows = [row for row in all_rows if row["model_label"] == label]
        elapsed_all = [row["elapsed_seconds"] for row in rows if row["elapsed_seconds"] is not None]
        elapsed_completed = [
            row["elapsed_seconds"]
            for row in rows
            if row["completed"] and row["elapsed_seconds"] is not None
        ]
        elapsed_no_repair = [
            row["elapsed_seconds"]
            for row in rows
            if row["repair_requests"] == 0 and row["elapsed_seconds"] is not None
        ]
        post_preprocess = [
            row["post_preprocess_seconds"]
            for row in rows
            if row["post_preprocess_seconds"] is not None
        ]
        summary["models"][label] = {
            "case_count": len(rows),
            "completed": sum(1 for row in rows if row["completed"]),
            "failed": sum(1 for row in rows if not row["completed"]),
            "reused_cases": sum(1 for row in rows if row["reused"]),
            "provider_requests_estimated": len(rows)
            + sum(int(row["repair_requests"]) for row in rows),
            "end_to_end_all": metrics(elapsed_all),
            "end_to_end_completed": metrics(elapsed_completed),
            "end_to_end_no_repair": metrics(elapsed_no_repair),
            "post_preprocess_approximation": metrics(post_preprocess),
        }

    labels = list(run_roots)
    grouped_cases: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        grouped_cases.setdefault((row["subject"], row["case_id"]), []).append(row)
    common_completed_cases: list[dict[str, Any]] = []
    for (subject, case_id), rows in sorted(grouped_cases.items()):
        by_label = {row["model_label"]: row for row in rows}
        if set(by_label) != set(labels):
            continue
        if not all(by_label[label]["completed"] for label in labels):
            continue
        seconds = {
            label: by_label[label]["elapsed_seconds"]
            for label in labels
            if by_label[label]["elapsed_seconds"] is not None
        }
        if len(seconds) != len(labels):
            continue
        common_completed_cases.append(
            {
                "subject": subject,
                "case_id": case_id,
                "elapsed_seconds": seconds,
                "fastest_model": min(seconds, key=seconds.get),
            }
        )
    summary["common_completed_cases"] = common_completed_cases

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "latency_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "latency_by_case.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]) if all_rows else [])
        if all_rows:
            writer.writeheader()
            writer.writerows(all_rows)

    report_lines = [
        "三模型端到端时延对比",
        "",
        "说明：elapsed_seconds 包含 Python 预处理、模型请求、可能的修复请求、校验和渲染；",
        "它适合比较用户实际等待时间，不等同于百炼服务器纯推理时延。",
        "post_preprocess_approximation 仅扣除已记录的预处理时间，仍包含本地校验和渲染。",
        "",
    ]
    for label, item in summary["models"].items():
        completed = item["end_to_end_completed"]
        no_repair = item["end_to_end_no_repair"]
        report_lines.extend(
            [
                f"[{label}]",
                f"完成：{item['completed']}/{item['case_count']}；估算请求：{item['provider_requests_estimated']}",
                (
                    "完成题端到端："
                    f"平均 {completed['mean_seconds']} s，中位 {completed['median_seconds']} s，"
                    f"P90 {completed['p90_seconds']} s"
                ),
                (
                    "无修复题端到端："
                    f"平均 {no_repair['mean_seconds']} s，中位 {no_repair['median_seconds']} s，"
                    f"样本 {no_repair['count']}"
                ),
                "",
            ]
        )
    report_lines.extend(
        [
            "[三个模型均完成的同题对比]",
            f"共同完成题数：{len(common_completed_cases)}",
        ]
    )
    for item in common_completed_cases:
        times = "；".join(
            f"{label} {seconds} s"
            for label, seconds in item["elapsed_seconds"].items()
        )
        report_lines.append(
            f"{item['case_id']}：{times}；最快 {item['fastest_model']}"
        )
    report_lines.append("")
    (output_dir / "LATENCY_REPORT.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
