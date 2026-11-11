#!/usr/bin/env python3
"""Serially benchmark Test6's local-first path without provider calls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
MATH_SOURCE = PROJECT_ROOT / "math" / "source"
sys.path.insert(0, str(MATH_SOURCE))

from search_machine_pipeline import SearchMachinePipeline  # noqa: E402


def natural_key(path: Path) -> tuple[Any, ...]:
    parts: list[Any] = []
    current = ""
    digit = False
    for character in path.stem:
        if character.isdigit() == digit:
            current += character
            continue
        if current:
            parts.append(int(current) if digit else current.lower())
        current = character
        digit = character.isdigit()
    if current:
        parts.append(int(current) if digit else current.lower())
    return tuple(parts)


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(len(ordered) * proportion + 0.999) - 1))
    return ordered[index]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 根节点必须是对象：{path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--previous-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    previous_results = args.previous_results.resolve()
    output_dir = args.output_dir.resolve()
    runs_root = output_dir / "_jobs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = SearchMachinePipeline(runs_root)
    rows: list[dict[str, Any]] = []

    images = sorted(
        (
            path
            for path in dataset.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ),
        key=natural_key,
    )
    for sequence, image_path in enumerate(images, 1):
        previous_job_path = previous_results / "math" / image_path.stem / "job.json"
        previous = load_json(previous_job_path)
        mock_output = previous.get("output_json")
        if not isinstance(mock_output, dict):
            raise ValueError(f"缺少可复用的本地模型结果：{previous_job_path}")

        job = pipeline.new_job(image_path.read_bytes(), image_path.name)
        completed = pipeline.solve(
            str(job["job_id"]),
            use_model=False,
            mock_output=mock_output,
        )
        streaming = completed.get("streaming") or {}
        preprocess = completed.get("preprocess") or {}
        fast_preview = preprocess.get("fast_preview") or {}
        previous_streaming = previous.get("streaming") or {}
        request_image = completed.get("request_image") or {}
        previous_request_image = previous.get("request_image") or {}
        row = {
            "sequence": sequence,
            "case": image_path.stem,
            "status": completed.get("status"),
            "baseline_first_image_seconds": previous_streaming.get(
                "first_image_seconds"
            ),
            "first_image_seconds": streaming.get("first_image_seconds"),
            "first_image_refined_seconds": streaming.get(
                "first_image_refined_seconds"
            ),
            "preprocess_seconds": preprocess.get("elapsed_seconds"),
            "fast_preview_stage_ms": fast_preview.get("elapsed_ms"),
            "fast_first_render_ms": preprocess.get("fast_first_image_render_ms"),
            "refined_first_render_ms": preprocess.get(
                "local_first_image_render_ms"
            ),
            "total_seconds": completed.get("elapsed_seconds"),
            "request_image_sha256": request_image.get("sha256"),
            "baseline_request_image_sha256": previous_request_image.get("sha256"),
            "request_image_unchanged": bool(request_image.get("sha256"))
            and request_image.get("sha256") == previous_request_image.get("sha256"),
            "error": (completed.get("error") or {}).get("message"),
        }
        row["local_first_status"] = (
            "completed"
            if isinstance(row["first_image_seconds"], (int, float))
            and isinstance(row["first_image_refined_seconds"], (int, float))
            and bool(row["request_image_sha256"])
            else "failed"
        )
        rows.append(row)
        print(
            f"[{sequence:02d}/{len(images):02d}] {image_path.stem}: "
            f"first={row['first_image_seconds']}s, "
            f"refined={row['first_image_refined_seconds']}s, "
            f"status={row['status']}",
            flush=True,
        )

    (output_dir / "latency_by_case.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "latency_by_case.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    first_values = [float(row["first_image_seconds"]) for row in rows]
    baseline_values = [
        float(row["baseline_first_image_seconds"])
        for row in rows
        if isinstance(row["baseline_first_image_seconds"], (int, float))
    ]
    refined_values = [float(row["first_image_refined_seconds"]) for row in rows]
    completed_count = sum(row["status"] == "completed" for row in rows)
    local_first_completed_count = sum(
        row["local_first_status"] == "completed" for row in rows
    )
    unchanged_count = sum(bool(row["request_image_unchanged"]) for row in rows)
    average_first = statistics.mean(first_values)
    average_baseline = statistics.mean(baseline_values)
    reduction = 1.0 - average_first / average_baseline
    report_lines = [
        "# Test6 快速本地首图串行时延报告",
        "",
        f"- 测试集：`{dataset}`",
        f"- 执行方式：{len(rows)} 题逐题串行；不调用模型，复用上次模型结果完成本地渲染",
        f"- 快速首图与完整 Test5 精修完成：{local_first_completed_count}/{len(rows)}",
        f"- 旧模型结果最终复放完成：{completed_count}/{len(rows)}（只用于走完渲染，不影响首图统计）",
        f"- 模型主图哈希未变化：{unchanged_count}/{len(rows)}",
        "",
        "## 汇总",
        "",
        f"- 旧首图平均：{average_baseline:.3f} s",
        f"- 新首图平均：{average_first:.3f} s",
        f"- 平均首图降幅：{reduction * 100:.1f}%",
        f"- 新首图中位数：{statistics.median(first_values):.3f} s",
        f"- 新首图 P90：{percentile(first_values, 0.90):.3f} s",
        f"- 新首图最大：{max(first_values):.3f} s",
        f"- 完整 Test5 本地精修平均：{statistics.mean(refined_values):.3f} s",
        "",
        "## 逐题",
        "",
        "| 题目 | 旧首图(s) | 新首图(s) | 本地精修(s) | 主图不变 | 首图链路 | 旧结果复放 |",
        "|---|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['case']} | {float(row['baseline_first_image_seconds']):.3f} | "
            f"{float(row['first_image_seconds']):.3f} | "
            f"{float(row['first_image_refined_seconds']):.3f} | "
            f"{'是' if row['request_image_unchanged'] else '否'} | "
            f"{row['local_first_status']} | {row['status']} |"
        )
    (output_dir / "LATENCY_REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return (
        0
        if local_first_completed_count == len(rows)
        and unchanged_count == len(rows)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
