from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .answer_guard import ANSWER_GUARD_VERSION
from .config import Settings
from .prompting import (
    TRANSPORT_CONTRACT_VERSION,
    prompt_protocol_sha256,
    read_prompt,
    sha256_text,
)
from .storage import utc_now, write_json


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def _rate(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0


def _latency_distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": round(max(values), 3) if values else None,
    }


def build_benchmark_summary(
    *,
    batch_dir: Path,
    phase: str,
    stage: str,
    preparation: dict[str, Any],
    results: list[dict[str, Any]],
    settings: Settings,
    project_root: Path,
) -> dict[str, Any]:
    total = len(results)
    completed = [row for row in results if row["status"] == "completed"]
    schema_pass = sum(bool(row.get("schema_pass")) for row in results)
    relevance_pass = sum(bool(row.get("relevance_pass")) for row in results)
    latency_pass = sum(bool(row.get("latency_pass")) for row in results)
    answer_guard_pass = sum(bool(row.get("answer_guard_pass")) for row in results)
    display_ready = sum(bool(row.get("display_ready")) for row in results)
    redaction_count = sum(int(row.get("redaction_count") or 0) for row in results)
    quality_pass = sum(bool(row.get("quality_gate_pass")) for row in results)
    model_values = [float(row["latency"]["model_ms"]) for row in completed]
    workflow_values = [float(row["latency"]["workflow_ms"]) for row in completed]
    generation_pipeline_values = [
        float(
            row["latency"].get(
                "generation_pipeline_ms", row["latency"]["workflow_ms"]
            )
        )
        for row in completed
    ]
    files_ready_values = [
        float(
            row["latency"].get(
                "files_ready_with_png_ms",
                row["latency"].get(
                    "generation_pipeline_ms", row["latency"]["workflow_ms"]
                ),
            )
        )
        for row in completed
    ]
    measured_values = [float(row["latency"]["measured_ms"]) for row in completed]
    source_statuses = [
        row.get("source_quality", {}).get("report_status") for row in completed
    ]
    formula_verified_values = [
        row.get("source_quality", {}).get("formula_verified") for row in completed
    ]
    provider_models = sorted(
        {
            str(row.get("provider_model"))
            for row in completed
            if row.get("provider_model")
        }
    )
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in completed:
        usage = row.get("usage") or {}
        for key in token_usage:
            value = usage.get(key)
            if isinstance(value, int):
                token_usage[key] += value
    rates = {
        "request_success_rate": _rate(len(completed), total),
        "schema_pass_rate": _rate(schema_pass, total),
        "relevance_pass_rate": _rate(relevance_pass, total),
        "latency_pass_rate": _rate(latency_pass, total),
        "answer_guard_pass_rate": _rate(answer_guard_pass, total),
        "display_ready_rate": _rate(display_ready, total),
        "combined_quality_pass_rate": _rate(quality_pass, total),
    }
    checks = {
        "schema_rate": rates["schema_pass_rate"] >= settings.min_schema_pass_rate,
        "relevance_rate": rates["relevance_pass_rate"]
        >= settings.min_relevance_pass_rate,
        "latency_rate": rates["latency_pass_rate"] >= settings.min_latency_pass_rate,
        "answer_guard_all": rates["answer_guard_pass_rate"] == 1.0,
        "combined_rate": rates["combined_quality_pass_rate"]
        >= settings.min_combined_pass_rate,
    }
    phase_gate_pass = all(checks.values())
    summary = {
        "benchmark_version": "test8_ans_benchmark_v2_answer_guard",
        "completed_at": utc_now(),
        "dataset": preparation,
        "phase": phase,
        "stage_override": stage,
        "model": settings.model,
        "provider_reported_models": provider_models,
        "response_format": settings.response_format,
        "raw_first_output_policy": True,
        "answer_guard_version": ANSWER_GUARD_VERSION,
        "repair_attempts": settings.repair_attempts,
        "prompt_sha256": {
            subject: sha256_text(read_prompt(project_root, subject))
            for subject in ("math", "physics")
        },
        "prompt_protocol_sha256": {
            subject: prompt_protocol_sha256(project_root, subject)
            for subject in ("math", "physics")
        },
        "transport_contract_version": TRANSPORT_CONTRACT_VERSION,
        "sample_counts": {
            "total": total,
            "completed": len(completed),
            "request_failed": total - len(completed),
            "schema_passed": schema_pass,
            "relevance_passed": relevance_pass,
            "latency_passed": latency_pass,
            "answer_guard_passed": answer_guard_pass,
            "display_ready": display_ready,
            "redacted_fragments": redaction_count,
            "combined_quality_passed": quality_pass,
            "ocr_source_pass": sum(value == "PASS" for value in source_statuses),
            "ocr_source_review": sum(value == "REVIEW" for value in source_statuses),
            "ocr_formula_verified": sum(
                value is True for value in formula_verified_values
            ),
        },
        "rates": rates,
        "token_usage": token_usage,
        "latency": {
            "gate_scope": settings.latency_scope,
            "budget_ms": settings.latency_budget_ms,
            "model_only": _latency_distribution(model_values),
            "workflow": _latency_distribution(workflow_values),
            "generation_pipeline": _latency_distribution(
                generation_pipeline_values
            ),
            "files_ready_with_png": _latency_distribution(files_ready_values),
            "gate_measurement": _latency_distribution(measured_values),
        },
        "thresholds": {
            "minimum_schema_pass_rate": settings.min_schema_pass_rate,
            "minimum_relevance_pass_rate": settings.min_relevance_pass_rate,
            "minimum_latency_pass_rate": settings.min_latency_pass_rate,
            "minimum_answer_guard_pass_rate": 1.0,
            "minimum_combined_pass_rate": settings.min_combined_pass_rate,
        },
        "gate_checks": checks,
        "phase_gate_pass": phase_gate_pass,
        "promotion_gate_pass": phase == "smoke" and phase_gate_pass,
        "correctness_scored": False,
        "correctness_note": (
            "冻结数据集声明 verified gold 为 0；当前只能评估格式、输入相关性与时延，"
            "不能据此声称答案正确率。"
        ),
        "source_quality_note": (
            "OCR 来源状态与公式独立核验率单独报告，当前不混入模型回答门槛；"
            "模型相关性通过不代表 OCR 与原图一致。"
        ),
        "batch_dir": str(batch_dir.resolve()),
        "results": results,
    }
    write_json(batch_dir / "benchmark_summary.json", summary)
    return summary


def validate_full_promotion(
    *,
    summary_path: Path,
    settings: Settings,
    project_root: Path,
    dataset_id: str,
    stage: str,
) -> dict[str, Any]:
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if summary.get("phase") != "smoke" or summary.get("promotion_gate_pass") is not True:
        raise ValueError("full 评测需要一份已通过晋级门槛的 smoke benchmark_summary.json")
    if summary.get("model") != settings.model:
        raise ValueError("smoke 汇总使用的模型与当前模型不一致")
    if summary.get("response_format") != settings.response_format:
        raise ValueError("smoke 汇总使用的结构化输出方式与当前设置不一致")
    if summary.get("repair_attempts") != 0:
        raise ValueError("smoke 汇总不是原始单次回复评测")
    if summary.get("dataset", {}).get("dataset_id") != dataset_id:
        raise ValueError("smoke 汇总使用的数据集与当前数据集不一致")
    if summary.get("stage_override") != stage:
        raise ValueError("smoke 汇总使用的学段覆盖值与当前值不一致")
    latency = summary.get("latency", {})
    if latency.get("gate_scope") != settings.latency_scope:
        raise ValueError("smoke 汇总使用的时延口径与当前设置不一致")
    if latency.get("budget_ms") != settings.latency_budget_ms:
        raise ValueError("smoke 汇总使用的时延预算与当前设置不一致")
    expected_thresholds = {
        "minimum_schema_pass_rate": settings.min_schema_pass_rate,
        "minimum_relevance_pass_rate": settings.min_relevance_pass_rate,
        "minimum_latency_pass_rate": settings.min_latency_pass_rate,
        "minimum_answer_guard_pass_rate": 1.0,
        "minimum_combined_pass_rate": settings.min_combined_pass_rate,
    }
    if summary.get("thresholds") != expected_thresholds:
        raise ValueError("smoke 汇总使用的晋级阈值与当前设置不一致")
    expected_prompts = {
        subject: sha256_text(read_prompt(project_root, subject))
        for subject in ("math", "physics")
    }
    if summary.get("prompt_sha256") != expected_prompts:
        raise ValueError("smoke 之后提示词已变化，请重新运行 smoke")
    expected_protocols = {
        subject: prompt_protocol_sha256(project_root, subject)
        for subject in ("math", "physics")
    }
    if summary.get("prompt_protocol_sha256") != expected_protocols:
        raise ValueError("smoke 之后请求模板已变化，请重新运行 smoke")
    if summary.get("transport_contract_version") != TRANSPORT_CONTRACT_VERSION:
        raise ValueError("smoke 之后传输契约版本已变化，请重新运行 smoke")
    if summary.get("answer_guard_version") != ANSWER_GUARD_VERSION:
        raise ValueError("smoke 之后答案脱敏规则已变化，请重新运行 smoke")
    return summary
