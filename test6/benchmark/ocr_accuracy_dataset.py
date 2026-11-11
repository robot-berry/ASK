"""Run the current Test6 local OCR pipeline over an arbitrary frozen dataset.

The runner is intentionally serial and resumable.  It never calls the answer
generation provider.  Structured OCR is still enabled because it is a local
part of the production OCR pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parent
LEGACY_RUNNER = ROOT / "ocr_accuracy_300.py"
DEFAULT_DATASET = Path(
    r"D:\robot\Documents\ASK\datasets\testzip_math_physics_2000_v3_eink"
)
DEFAULT_OUTPUT = ROOT / "ocr_accuracy_2000_v3_eink_20260902"


def load_legacy_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "test6_generic_ocr_pipeline", LEGACY_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load OCR runner: {LEGACY_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_record(row: dict[str, Any], ordinal: int) -> dict[str, Any]:
    subject = str(row.get("subject") or "").strip()
    if subject not in {"mathematics", "physics"}:
        raise RuntimeError(f"Unsupported subject at record {ordinal}: {subject!r}")
    subject_code = "MATH" if subject == "mathematics" else "PHYS"
    subject_index = int(row.get("subject_index") or row.get("index") or ordinal)
    image_path = str(row.get("image_path") or row.get("file") or "").replace("\\", "/")
    dataset_quality = row.get("quality") or {}
    if not isinstance(dataset_quality, dict):
        dataset_quality = {}
    return {
        "question_id": str(row.get("question_id") or f"EINK2K-{subject_code}-{subject_index:04d}"),
        "subject": subject,
        "source_group": str(row.get("level") or row.get("source_group") or "unknown"),
        "source_index": subject_index,
        "image_path": image_path,
        "image_sha256": str(row.get("image_sha256") or ""),
        "scope": str(row.get("scope") or "unknown"),
        "origin": str(row.get("origin") or "unknown"),
        "dataset_quality_status": str(dataset_quality.get("status") or "unknown"),
        "dataset_quality_flags": list(dataset_quality.get("flags") or []),
        "dataset_question_complete": dataset_quality.get("question_complete"),
        "dataset_readable": dataset_quality.get("readable"),
        "dataset_multiple_questions": dataset_quality.get("multiple_questions"),
        "gold_verified": bool(row.get("gold_verified")),
        "ground_truth_transcript": row.get("ground_truth_transcript"),
    }


def load_manifest(dataset: Path) -> list[dict[str, Any]]:
    answer_bank = dataset / "answers" / "answer_bank_v1.jsonl"
    manifest_jsonl = dataset / "manifest.jsonl"
    manifest_csv = dataset / "manifest.csv"
    if answer_bank.exists():
        raw_rows = read_jsonl(answer_bank)
    elif manifest_jsonl.exists():
        raw_rows = read_jsonl(manifest_jsonl)
    elif manifest_csv.exists():
        with manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_rows = list(csv.DictReader(handle))
    else:
        raise RuntimeError("Dataset has no supported manifest")
    rows = [normalize_record(row, index) for index, row in enumerate(raw_rows, 1)]
    ids = [row["question_id"] for row in rows]
    paths = [row["image_path"] for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Dataset contains duplicate question IDs")
    if len(set(paths)) != len(paths):
        raise RuntimeError("Dataset contains duplicate image paths")
    missing = [row["image_path"] for row in rows if not (dataset / row["image_path"]).is_file()]
    if missing:
        raise RuntimeError(f"Dataset is missing {len(missing)} images: {missing[:3]}")
    return rows


def manifest_sha256(dataset: Path) -> str:
    for relative in ("answers/answer_bank_v1.jsonl", "manifest.jsonl", "manifest.csv"):
        path = dataset / relative
        if path.exists():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    return ""


def verify_image_hashes(dataset: Path, rows: list[dict[str, Any]]) -> None:
    mismatches: list[str] = []
    for row in rows:
        expected = str(row.get("image_sha256") or "")
        if not expected:
            continue
        actual = hashlib.sha256((dataset / str(row["image_path"])).read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(str(row["question_id"]))
    if mismatches:
        raise RuntimeError(
            f"Dataset image SHA-256 mismatch for {len(mismatches)} rows: {mismatches[:5]}"
        )


def quantile(core: Any, rows: list[dict[str, Any]], field: str, q: float) -> float | None:
    values = [float(row[field]) for row in rows if row.get("status") == "success" and row.get(field) is not None]
    return core.quantile(values, q)


def aggregate(core: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    base = core.aggregate(rows)
    successful = [row for row in rows if row.get("status") == "success"]
    count = len(rows)
    success_count = len(successful)
    structured_times = [float(row.get("structured_ocr_elapsed_ms") or 0) for row in successful]
    structure_passes = sum(bool(row.get("semantic_structure_pass")) for row in successful)
    selected = sum(bool(row.get("structured_ocr_selected")) for row in successful)
    flags = Counter(flag for row in successful for flag in row.get("quality_flags", []))
    semantic_issues = Counter(
        issue for row in successful for issue in row.get("semantic_issue_types", [])
    )
    base.update(
        {
            "failure_count": count - success_count,
            "semantic_structure_pass_count": structure_passes,
            "semantic_structure_pass_rate": round(structure_passes / count, 6) if count else 0.0,
            "semantic_structure_fail_count": success_count - structure_passes,
            "semantic_issue_type_counts": dict(sorted(semantic_issues.items())),
            "semantic_page_noise_count": sum(
                bool(row.get("semantic_page_noise_detected")) for row in successful
            ),
            "dataset_known_quality_issue_count": sum(
                bool(row.get("dataset_quality_flags")) for row in rows
            ),
            "structured_ocr_selected_count": selected,
            "structured_ocr_selected_rate": round(selected / count, 6) if count else 0.0,
            "structured_ocr_attempted_count": sum(value > 0 for value in structured_times),
            "structured_ocr_ms_mean": round(statistics.fmean(structured_times), 3) if structured_times else None,
            "illustration_detected_count": sum(int(row.get("illustration_count") or 0) > 0 for row in successful),
            "semantic_recovery_count": sum(bool(row.get("semantic_recovery_actions")) for row in successful),
            "total_ms_min": quantile(core, rows, "total_ms", 0.0),
            "total_ms_p99": quantile(core, rows, "total_ms", 0.99),
            "total_ms_max": quantile(core, rows, "total_ms", 1.0),
            "ocr_and_layout_ms_mean": round(
                statistics.fmean(float(row.get("ocr_and_layout_ms") or 0) for row in successful), 3
            ) if successful else None,
            "quality_flag_counts": dict(sorted(flags.items())),
        }
    )
    return base


def select_audit_sample(rows: list[dict[str, Any]], limit: int = 60) -> list[dict[str, Any]]:
    successful = [row for row in rows if row.get("status") == "success"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        grouped[(str(row["subject"]), str(row["source_group"]))].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for key in sorted(grouped):
        members = sorted(
            grouped[key],
            key=lambda row: (
                bool(row.get("semantic_structure_pass")),
                float(row.get("weighted_character_confidence") or 0),
                str(row["question_id"]),
            ),
        )
        for row in members[:5]:
            selected[str(row["question_id"])] = row
    ranked = sorted(
        successful,
        key=lambda row: (
            bool(row.get("semantic_structure_pass")),
            float(row.get("weighted_character_confidence") or 0),
            -len(row.get("quality_flags") or []),
            str(row["question_id"]),
        ),
    )
    for row in ranked:
        if len(selected) >= limit:
            break
        selected.setdefault(str(row["question_id"]), row)
    return list(selected.values())[:limit]


def write_details(output: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "question_id", "subject", "source_group", "source_index", "scope", "origin",
        "dataset_quality_status", "dataset_quality_flags", "dataset_question_complete",
        "dataset_readable", "dataset_multiple_questions",
        "image_path", "image_sha256", "status", "error", "compact_character_count",
        "cjk_count", "alphanumeric_count", "line_count", "question_marker_detected",
        "part_marker_count", "weighted_character_confidence", "mean_item_confidence",
        "p10_item_confidence", "ocr_pass_count", "ocr_selected_pass",
        "structured_ocr_selected", "structured_formula_count", "structured_ocr_elapsed_ms",
        "structured_ocr_reason", "semantic_structure_pass", "semantic_issue_count",
        "semantic_issue_types", "semantic_page_noise_detected", "semantic_option_labels",
        "semantic_subquestion_numbers", "semantic_recovery_actions", "illustration_count", "rotation",
        "perspective_corrected", "deskew_angle", "outer_whitespace_cropped",
        "ocr_and_layout_ms", "total_ms", "quality_flags", "transcript",
    ]
    with (output / "details.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["quality_flags"] = ";".join(row.get("quality_flags", []))
            flat["dataset_quality_flags"] = ";".join(row.get("dataset_quality_flags", []))
            flat["semantic_recovery_actions"] = ";".join(row.get("semantic_recovery_actions", []))
            flat["semantic_issue_types"] = ";".join(row.get("semantic_issue_types", []))
            flat["semantic_option_labels"] = ";".join(row.get("semantic_option_labels", []))
            flat["semantic_subquestion_numbers"] = ";".join(
                str(value) for value in row.get("semantic_subquestion_numbers", [])
            )
            writer.writerow(flat)


def write_semantic_review_queue(
    core: Any,
    dataset: Path,
    output: Path,
    rows: list[dict[str, Any]],
) -> int:
    failures = [
        row for row in rows
        if row.get("status") != "success" or not bool(row.get("semantic_structure_pass"))
    ]
    review_root = output / "semantic_review"
    review_root.mkdir(parents=True, exist_ok=True)
    visual_failures = [row for row in failures if row.get("status") == "success"]
    if visual_failures:
        core.write_audit_pages(dataset, review_root, visual_failures)
    fields = [
        "question_id", "subject", "source_group", "source_index", "image_path",
        "status", "semantic_structure_pass", "semantic_issue_count",
        "semantic_issue_types", "weighted_character_confidence", "quality_flags",
        "manual_semantic_complete", "manual_error_types", "manual_notes",
    ]
    with (review_root / "semantic_review_queue.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for source in failures:
            row = dict(source)
            row["semantic_issue_types"] = ";".join(source.get("semantic_issue_types", []))
            row["quality_flags"] = ";".join(source.get("quality_flags", []))
            row["manual_semantic_complete"] = ""
            row["manual_error_types"] = ""
            row["manual_notes"] = ""
            writer.writerow(row)
    return len(failures)


def write_outputs(
    core: Any,
    dataset: Path,
    output: Path,
    manifest: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    started_at: str,
    wall_elapsed_seconds: float,
    final: bool,
) -> dict[str, Any]:
    all_stats = aggregate(core, rows)
    subjects = sorted({str(row["subject"]) for row in manifest})
    levels = sorted({str(row["source_group"]) for row in manifest})
    summary: dict[str, Any] = {
        "schema_version": "test6-local-ocr-benchmark-v1",
        "dataset": str(dataset),
        "dataset_manifest_sha256": manifest_sha256(dataset),
        "dataset_declared_count": len(manifest),
        "processed_count": len(rows),
        "complete": final and len(rows) == len(manifest),
        "execution_mode": "strictly_serial_single_process",
        "answer_generation_model_called": False,
        "ocr_pipeline": "current_test6_rapidocr_plus_local_structured_ocr",
        "ground_truth_transcript_count": sum(bool(row.get("ground_truth_transcript")) for row in manifest),
        "accuracy_status": "manual_audit_required_no_gold_transcripts",
        "started_at": started_at,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "wall_elapsed_seconds": round(wall_elapsed_seconds, 3),
        "all": all_stats,
        "by_subject": {
            subject: aggregate(core, [row for row in rows if row.get("subject") == subject])
            for subject in subjects
        },
        "by_level": {
            level: aggregate(core, [row for row in rows if row.get("source_group") == level])
            for level in levels
        },
        "by_subject_level": {
            f"{subject}/{level}": aggregate(
                core,
                [row for row in rows if row.get("subject") == subject and row.get("source_group") == level],
            )
            for subject in subjects
            for level in levels
            if any(row.get("subject") == subject and row.get("source_group") == level for row in rows)
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_details(output, rows)
    if final:
        sample = select_audit_sample(rows)
        core.write_audit_pages(dataset, output, sample)
        semantic_review_count = write_semantic_review_queue(
            core, dataset, output, rows
        )
        stats = summary["all"]
        report = [
            f"# {len(manifest)}题当前本地OCR全量串行统计",
            "",
            f"- 数据集：`{dataset}`",
            f"- 完成：{stats['success_count']}/{stats['count']}，失败 {stats['failure_count']}题",
            "- 执行方式：单进程逐题串行；未调用答题大模型。",
            "- OCR组成：当前 Test6 RapidOCR + 本机结构化公式 OCR。",
            f"- 标准题干文本：{summary['ground_truth_transcript_count']}题。",
            "- 说明：无人工标准题干，以下置信度和自动结构通过率不等于真实文字正确率。",
            "",
            "## 总体",
            "",
            f"- OCR运行成功率：{stats['success_rate']:.2%}",
            f"- 非空文字率：{stats['nonempty_rate']:.2%}",
            f"- 自动结构通过率：{stats['semantic_structure_pass_rate']:.2%}",
            f"- 自动判为结构不完整：{stats['semantic_structure_fail_count']}题",
            f"- 数据集已标记图片质量问题：{stats['dataset_known_quality_issue_count']}题（需与OCR缺失分开判断）",
            f"- 平均字符加权置信度：{stats['mean_weighted_character_confidence']:.2%}",
            f"- 结构化OCR采用率：{stats['structured_ocr_selected_rate']:.2%}",
            f"- 单题时延均值：{stats['total_ms_mean']} ms",
            f"- 单题时延 P50/P90/P95/P99：{stats['total_ms_p50']} / {stats['total_ms_p90']} / {stats['total_ms_p95']} / {stats['total_ms_p99']} ms",
            f"- 全量墙钟耗时：{summary['wall_elapsed_seconds']} 秒",
            f"- 人工复核抽样：{len(sample)}题，见 `manual_audit_pages/` 和 `manual_audit.csv`。",
            f"- 自动不完整复核队列：{semantic_review_count}题，见 `semantic_review/manual_audit_pages/`。",
            "",
            "## 分科",
            "",
        ]
        for subject, values in summary["by_subject"].items():
            report.append(
                f"- {subject}：{values['success_count']}/{values['count']}成功，"
                f"结构通过率 {values['semantic_structure_pass_rate']:.2%}，"
                f"平均置信度 {values['mean_weighted_character_confidence']:.2%}，"
                f"时延均值/P95 {values['total_ms_mean']} / {values['total_ms_p95']} ms"
            )
        report.extend(["", "## 分组语义结构通过率", ""])
        for name, values in summary["by_subject_level"].items():
            report.append(
                f"- {name}：{values['semantic_structure_pass_count']}/{values['count']}，"
                f"{values['semantic_structure_pass_rate']:.2%}"
            )
        report.extend(["", "## 自动语义缺陷类型", ""])
        report.extend(
            [
                "- `stem_missing`：检测到选项但题干缺失",
                "- `option_sequence_incomplete`：选择题选项序列不完整",
                "- `choice_options_missing`：存在选择题语义但选项缺失",
                "- `subquestion_gap`：小问编号不连续",
                "- `nested_subquestion_incomplete`：出现第(1)问但第(2)问缺失",
                "- `table_row_incomplete`：表格关键数据行不完整",
                "- `formula_gap`：已知高风险公式位置出现缺口",
                "- `symbol_damage`：出现损坏字符或连续问号",
                "- `too_short`：识别文字过短，无法构成完整题意",
                "",
                "### 本次计数",
                "",
            ]
        )
        for name, count in stats["semantic_issue_type_counts"].items():
            report.append(f"- `{name}`：{count}")
        report.extend(["", "## 质量标记", ""])
        for name, count in stats["quality_flag_counts"].items():
            report.append(f"- `{name}`：{count}")
        (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(dataset)
    if args.limit > 0:
        manifest = manifest[: args.limit]
    print(f"Verifying {len(manifest)} dataset image hashes...", flush=True)
    verify_image_hashes(dataset, manifest)
    core = load_legacy_runner()
    results_path = output / "results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        completed = {
            str(row["question_id"]): row for row in read_jsonl(results_path)
        }
    started_at_path = output / "started_at.txt"
    if started_at_path.exists():
        started_at = started_at_path.read_text(encoding="utf-8").strip()
    else:
        started_at = datetime.now(timezone.utc).isoformat()
        started_at_path.write_text(started_at + "\n", encoding="utf-8")
    run_started = time.perf_counter()
    modules = {subject: core.load_test5(subject) for subject in core.TEST5_PATHS}
    engines = {subject: module.RapidOCR() for subject, module in modules.items()}
    pending = [row for row in manifest if str(row["question_id"]) not in completed]
    print(
        f"Serial local OCR: dataset={len(manifest)}, completed={len(completed)}, pending={len(pending)}",
        flush=True,
    )
    for ordinal, record in enumerate(manifest, 1):
        question_id = str(record["question_id"])
        if question_id in completed:
            continue
        print(f"[{ordinal}/{len(manifest)}] OCR {question_id}", flush=True)
        row = dict(record)
        try:
            row.update(
                core.process_image(
                    modules[str(record["subject"])],
                    engines[str(record["subject"])],
                    dataset / str(record["image_path"]),
                )
            )
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "quality_flags": ["processing_failure"],
                    "transcript": "",
                }
            )
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        completed[question_id] = row
        if len(completed) % max(1, args.checkpoint_every) == 0:
            ordered = [completed[str(item["question_id"])] for item in manifest if str(item["question_id"]) in completed]
            write_outputs(
                core, dataset, output, manifest, ordered, started_at,
                time.perf_counter() - run_started, False,
            )
            print(f"checkpoint={len(completed)}/{len(manifest)}", flush=True)
    rows = [completed[str(record["question_id"])] for record in manifest]
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = write_outputs(
        core, dataset, output, manifest, rows, started_at,
        time.perf_counter() - run_started, True,
    )
    print(
        f"Done: {output} success={summary['all']['success_count']}/{len(manifest)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
