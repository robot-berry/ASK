from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .answer_guard import ANSWER_GUARD_VERSION, sanitize_display_answer
from .client import QwenClient
from .config import Settings
from .models import Question
from .prompting import (
    build_messages,
    build_repair_messages,
    messages_sha256,
    prompt_path,
    read_prompt,
    read_schema,
    schema_path,
    sha256_text,
    TRANSPORT_CONTRACT_VERSION,
)
from .relevance import validate_relevance
from .render import assess_renderability, render_model_output_screenshot, render_tiers
from .schema import OutputParseError, parse_json_content, validate_guidance
from .storage import append_jsonl, new_run_dir, utc_now, write_json, write_text


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WorkflowError(RuntimeError):
    def __init__(self, message: str, *, run_dir: Path | None = None):
        super().__init__(message)
        self.run_dir = run_dir


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    raw_output: str
    answer: dict[str, Any] | None
    display_answer: dict[str, Any] | None
    validation: dict[str, Any]
    attempts: int

    @property
    def quality_gate_pass(self) -> bool:
        return bool(self.validation["quality_gate_pass"])


class GuidanceWorkflow:
    def __init__(
        self,
        settings: Settings,
        *,
        project_root: Path = PROJECT_ROOT,
        runs_root: Path | None = None,
        client: QwenClient | None = None,
    ):
        self.settings = settings
        self.project_root = project_root
        self.runs_root = runs_root or (project_root / "runs")
        self.client = client or QwenClient(settings)

    def preview(self, item: Question) -> dict[str, Any]:
        messages = build_messages(self.project_root, item)
        schema = read_schema(self.project_root, item.subject)
        payload = self.client.build_payload(messages, schema)
        return {
            "question": item.as_dict(),
            "endpoint": self.settings.chat_completions_url,
            "payload": payload,
            "prompt_file": str(prompt_path(self.project_root, item.subject)),
            "schema_file": str(schema_path(self.project_root, item.subject)),
            "latency_gate": {
                "scope": self.settings.latency_scope,
                "budget_ms": self.settings.latency_budget_ms,
            },
        }

    def _validate_attempt(
        self,
        *,
        item: Question,
        content: str,
        request_id: str | None,
        model_latency_ms: float,
        workflow_started_at: float,
        transport_attempts: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
        validation_started_at = time.perf_counter()
        transport_audit: list[dict[str, Any]] = []
        try:
            answer = parse_json_content(
                content,
                normalize_sentence_transport=True,
                normalization_audit=transport_audit,
            )
            schema_report = validate_guidance(answer, item.subject)
        except OutputParseError as exc:
            answer = None
            schema_report = {
                "valid": False,
                "errors": [str(exc)],
                "warnings": [],
            }
        if answer is None:
            relevance_report = {
                "valid": False,
                "method": "local_lexical_anchor_v1",
                "scope": "screening_only_not_semantic_correctness",
                "errors": ["输出不是可解析的五层 JSON，无法执行相关性检查"],
                "metrics": {},
                "evidence": {},
            }
        else:
            relevance_report = validate_relevance(item.question, answer)

        if answer is not None:
            display_answer, answer_guard_report = sanitize_display_answer(
                answer,
                question=item.question,
                subject=item.subject,
            )
        else:
            display_answer = None
            answer_guard_report = {
                "valid": False,
                "version": ANSWER_GUARD_VERSION,
                "method": "not_run_until_output_is_parseable",
                "raw_answer_preserved": True,
                "display_copy_created": False,
                "redaction_count": 0,
                "actions": [],
                "remaining_suspicions": [],
                "errors": ["模型回复不可解析，无法构造展示副本"],
            }

        render_eligibility = assess_renderability(display_answer)
        if answer_guard_report.get("render_blocking"):
            render_eligibility["valid"] = False
            render_eligibility["errors"] = [
                *render_eligibility.get("errors", []),
                "脱敏后仍有明确的最终答案嫌疑片段",
            ]
        soft_gate_failures: list[str] = []
        if not schema_report["valid"]:
            soft_gate_failures.append("严格五层 JSON 格式未通过")
        if not relevance_report["valid"]:
            soft_gate_failures.append("本地关联性筛查未通过")
        if not answer_guard_report["valid"]:
            soft_gate_failures.append("最终答案脱敏置信度不足")
        render_eligibility["soft_gate_failures"] = soft_gate_failures
        render_eligibility["note"] = (
            "句数、字数、字段细节、关联性和脱敏置信度只用于审核告警；"
            "只有无法解析或缺少整层可展示内容才阻止出图。"
        )

        local_validation_ms = (time.perf_counter() - validation_started_at) * 1000
        workflow_latency_ms = (time.perf_counter() - workflow_started_at) * 1000
        upstream_ms = item.upstream_latency_ms or 0.0
        measured_ms = (
            model_latency_ms
            if self.settings.latency_scope == "model_only"
            else workflow_latency_ms
        )
        latency_report = {
            "valid": measured_ms <= self.settings.latency_budget_ms,
            "scope": self.settings.latency_scope,
            "budget_ms": self.settings.latency_budget_ms,
            "measured_ms": round(measured_ms, 3),
            "model_ms": round(model_latency_ms, 3),
            "generation_pipeline_ms": round(workflow_latency_ms, 3),
            "workflow_ms": round(workflow_latency_ms, 3),
            "local_validation_ms": round(local_validation_ms, 3),
            "excluded_upstream_ms": round(upstream_ms, 3),
            "transport_attempts": transport_attempts,
        }
        report = {
            "quality_gate_pass": bool(
                schema_report["valid"]
                and relevance_report["valid"]
                and latency_report["valid"]
                and answer_guard_report["valid"]
            ),
            "display_ready": bool(
                render_eligibility["valid"]
            ),
            "render_eligibility": render_eligibility,
            "schema": schema_report,
            "relevance": relevance_report,
            "latency": latency_report,
            "answer_guard": answer_guard_report,
            "transport_normalization": {
                "applied": bool(transport_audit),
                "content_changed": False,
                "actions": transport_audit,
                "note": (
                    "只把重复或字符串形式的 s 按原出现顺序装入数组；"
                    "不修改模型句子，原始回复保存在 model_output.txt。"
                ),
            },
            "request_id": request_id,
        }
        return answer, display_answer, report

    def _render_and_finalize_latency(
        self,
        *,
        validation: dict[str, Any],
        display_answer: dict[str, Any] | None,
        run_dir: Path,
        workflow_started_at: float,
    ) -> None:
        content_ready = bool(
            validation.get("render_eligibility", {}).get("valid")
            and display_answer is not None
        )
        render_started_at = time.perf_counter()
        image_paths: list[Path] = []
        render_errors: list[str] = []
        if content_ready:
            try:
                image_paths = render_tiers(
                    display_answer,
                    run_dir / "images_800px",
                    width=800,
                )
            except Exception as exc:
                render_errors.append(f"生成 PNG 失败：{exc}")
        else:
            render_errors.extend(
                validation.get("render_eligibility", {}).get("errors", [])
            )
        png_render_ms = (time.perf_counter() - render_started_at) * 1000
        files_ready_ms = (time.perf_counter() - workflow_started_at) * 1000
        latency = validation["latency"]
        if self.settings.latency_scope == "model_only":
            measured_ms = float(latency["model_ms"])
        elif self.settings.latency_scope == "generation_pipeline":
            measured_ms = float(latency["generation_pipeline_ms"])
        else:
            measured_ms = files_ready_ms
        latency.update(
            {
                "scope": self.settings.latency_scope,
                "measured_ms": round(measured_ms, 3),
                "files_ready_with_png_ms": round(files_ready_ms, 3),
                "png_render_ms": round(png_render_ms, 3),
                "png_included_in_default_gate": True,
                "valid": measured_ms <= self.settings.latency_budget_ms,
            }
        )
        render_valid = content_ready and len(image_paths) == 5
        validation["render"] = {
            "valid": render_valid,
            "policy": "minimum_display_gate_strict_quality_is_warning_only",
            "width_px": 800,
            "image_count": len(image_paths),
            "images": [str(path.resolve()) for path in image_paths],
            "errors": [] if render_valid else (
                render_errors or ["未完整生成五张 PNG"]
            ),
        }
        validation["quality_gate_pass"] = bool(
            validation["schema"]["valid"]
            and validation["relevance"]["valid"]
            and validation["answer_guard"]["valid"]
            and render_valid
            and latency["valid"]
        )

    def _render_model_answer_comparison(
        self,
        *,
        validation: dict[str, Any],
        raw_output: str,
        run_dir: Path,
        workflow_started_at: float,
    ) -> None:
        started_at = time.perf_counter()
        path = render_model_output_screenshot(
            raw_output,
            run_dir / "model_answer_screenshot" / "model_output.png",
            width=800,
        )
        validation["model_answer_screenshot"] = {
            "valid": True,
            "path": str(path.resolve()),
            "width_px": 800,
            "render_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "ready_from_workflow_start_ms": round(
                (time.perf_counter() - workflow_started_at) * 1000, 3
            ),
            "excluded_from_product_latency_gate": True,
        }

    def run(self, item: Question) -> RunResult:
        self.settings.validate(require_api_key=True)
        workflow_started = time.perf_counter()
        run_dir = new_run_dir(self.runs_root, item.id)
        messages = build_messages(self.project_root, item)
        schema = read_schema(self.project_root, item.subject)
        prompt = read_prompt(self.project_root, item.subject)
        started_at = utc_now()
        write_json(run_dir / "input.json", item.as_dict())
        write_json(
            run_dir / "request.json",
            {
                "endpoint": self.settings.chat_completions_url,
                "payload": self.client.build_payload(messages, schema),
            },
        )
        metadata: dict[str, Any] = {
            "workflow_version": "test8_ans_v5_render_quality_gate_separation",
            "state": "running",
            "started_at": started_at,
            "question_id": item.id,
            "subject": item.subject,
            "stage": item.stage,
            "question_source": item.question_source,
            "dataset_id": item.dataset_id,
            "model": self.settings.model,
            "response_format": self.settings.response_format,
            "prompt_file": str(prompt_path(self.project_root, item.subject)),
            "prompt_sha256": sha256_text(prompt),
            "schema_file": str(schema_path(self.project_root, item.subject)),
            "repair_attempts_allowed": self.settings.repair_attempts,
            "raw_first_output_policy": True,
            "lossless_transport_normalization": True,
            "answer_guard_version": ANSWER_GUARD_VERSION,
            "transport_contract_version": TRANSPORT_CONTRACT_VERSION,
            "request_messages_sha256": messages_sha256(messages),
        }
        write_json(run_dir / "metadata.json", metadata)

        active_messages = messages
        total_attempts = self.settings.repair_attempts + 1
        selected: RunResult | None = None
        try:
            for attempt_index in range(1, total_attempts + 1):
                completion = self.client.complete(active_messages, schema)
                prefix = f"attempt_{attempt_index:02d}"
                write_json(run_dir / f"{prefix}_response.json", completion.response)
                write_text(run_dir / f"{prefix}_content.txt", completion.content)
                if attempt_index == 1:
                    write_json(run_dir / "provider_response.json", completion.response)
                    write_text(run_dir / "model_output.txt", completion.content)

                answer, display_answer, validation = self._validate_attempt(
                    item=item,
                    content=completion.content,
                    request_id=completion.request_id,
                    model_latency_ms=completion.latency_ms,
                    workflow_started_at=workflow_started,
                    transport_attempts=completion.transport_attempts,
                )
                validation["provider_model"] = completion.response.get("model")
                validation["usage"] = completion.response.get("usage")
                if answer is not None:
                    write_json(run_dir / f"{prefix}_answer.json", answer)
                if display_answer is not None:
                    write_json(
                        run_dir / f"{prefix}_display_answer.json", display_answer
                    )
                    write_json(
                        run_dir / f"{prefix}_answer_redaction.json",
                        validation["answer_guard"],
                    )
                self._render_and_finalize_latency(
                    validation=validation,
                    display_answer=display_answer,
                    run_dir=run_dir,
                    workflow_started_at=workflow_started,
                )
                self._render_model_answer_comparison(
                    validation=validation,
                    raw_output=completion.content,
                    run_dir=run_dir,
                    workflow_started_at=workflow_started,
                )
                write_json(run_dir / f"{prefix}_validation.json", validation)
                write_json(
                    run_dir / f"{prefix}_transport_normalization.json",
                    validation["transport_normalization"],
                )
                selected = RunResult(
                    run_dir=run_dir,
                    raw_output=completion.content,
                    answer=answer,
                    display_answer=display_answer,
                    validation=validation,
                    attempts=attempt_index,
                )
                if validation["quality_gate_pass"]:
                    break
                repair_errors = [
                    *validation.get("render_eligibility", {}).get("errors", []),
                    *validation.get("render", {}).get("errors", []),
                ]
                if attempt_index < total_attempts and repair_errors:
                    active_messages = build_repair_messages(
                        messages, completion.content, repair_errors
                    )
                else:
                    break

            assert selected is not None
            if selected.answer is not None:
                write_json(run_dir / "answer.json", selected.answer)
            if selected.display_answer is not None:
                write_json(run_dir / "display_answer.json", selected.display_answer)
                write_json(
                    run_dir / "answer_redaction.json",
                    selected.validation["answer_guard"],
                )
            write_text(run_dir / "selected_model_output.txt", selected.raw_output)
            write_json(run_dir / "validation.json", selected.validation)
            write_json(
                run_dir / "transport_normalization.json",
                selected.validation["transport_normalization"],
            )
            metadata.update(
                {
                    "state": (
                        "quality_passed"
                        if selected.quality_gate_pass
                        else "completed_with_quality_failure"
                    ),
                    "completed_at": utc_now(),
                    "attempts": selected.attempts,
                    "selected_attempt": selected.attempts,
                    "quality_gate_pass": selected.quality_gate_pass,
                    "display_ready": selected.validation["display_ready"],
                    "request_id": selected.validation["request_id"],
                    "latency": selected.validation["latency"],
                }
            )
            write_json(run_dir / "metadata.json", metadata)
            return selected
        except Exception as exc:
            metadata.update(
                {
                    "state": "failed_request",
                    "completed_at": utc_now(),
                    "error": str(exc),
                }
            )
            write_json(run_dir / "metadata.json", metadata)
            raise WorkflowError(str(exc), run_dir=run_dir) from exc

    def reprocess_saved_run(self, source_run_dir: Path) -> RunResult:
        """Re-run local normalization and gates without another model call."""

        source_run_dir = source_run_dir.resolve()
        input_path = source_run_dir / "input.json"
        raw_path = source_run_dir / "model_output.txt"
        validation_path = source_run_dir / "validation.json"
        for required in (input_path, raw_path, validation_path):
            if not required.is_file():
                raise ValueError(f"历史运行缺少文件：{required}")
        item = Question.from_mapping(
            json.loads(input_path.read_text(encoding="utf-8-sig"))
        )
        raw_output = raw_path.read_text(encoding="utf-8-sig")
        old_validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
        old_latency = old_validation.get("latency", {})
        model_latency_ms = float(old_latency.get("model_ms") or 0.0)
        historical_pipeline_ms = float(
            old_latency.get("generation_pipeline_ms")
            or old_latency.get("workflow_ms")
            or model_latency_ms
        )
        synthetic_started_at = time.perf_counter() - historical_pipeline_ms / 1000
        answer, display_answer, validation = self._validate_attempt(
            item=item,
            content=raw_output,
            request_id=old_validation.get("request_id"),
            model_latency_ms=model_latency_ms,
            workflow_started_at=synthetic_started_at,
            transport_attempts=int(old_latency.get("transport_attempts") or 1),
        )
        validation["provider_model"] = old_validation.get("provider_model")
        validation["usage"] = old_validation.get("usage")

        run_dir = new_run_dir(self.runs_root, f"{item.id}-local-replay")
        write_json(run_dir / "input.json", item.as_dict())
        write_text(run_dir / "model_output.txt", raw_output)
        write_text(run_dir / "selected_model_output.txt", raw_output)
        for name in ("provider_response.json", "request.json"):
            source = source_run_dir / name
            if source.is_file():
                write_text(run_dir / name, source.read_text(encoding="utf-8-sig"))
        if answer is not None:
            write_json(run_dir / "answer.json", answer)
        if display_answer is not None:
            write_json(run_dir / "display_answer.json", display_answer)
            write_json(run_dir / "answer_redaction.json", validation["answer_guard"])
        self._render_and_finalize_latency(
            validation=validation,
            display_answer=display_answer,
            run_dir=run_dir,
            workflow_started_at=synthetic_started_at,
        )
        self._render_model_answer_comparison(
            validation=validation,
            raw_output=raw_output,
            run_dir=run_dir,
            workflow_started_at=synthetic_started_at,
        )
        write_json(
            run_dir / "transport_normalization.json",
            validation["transport_normalization"],
        )
        write_json(run_dir / "validation.json", validation)
        write_json(
            run_dir / "metadata.json",
            {
                "workflow_version": "test8_ans_v5_render_quality_gate_separation",
                "state": (
                    "quality_passed"
                    if validation["quality_gate_pass"]
                    else "completed_with_quality_failure"
                ),
                "completed_at": utc_now(),
                "question_id": item.id,
                "subject": item.subject,
                "stage": item.stage,
                "model": self.settings.model,
                "source_run_dir": str(source_run_dir),
                "model_calls_in_reprocess": 0,
                "quality_gate_pass": validation["quality_gate_pass"],
                "display_ready": validation["display_ready"],
                "latency": validation["latency"],
            },
        )
        return RunResult(
            run_dir=run_dir,
            raw_output=raw_output,
            answer=answer,
            display_answer=display_answer,
            validation=validation,
            attempts=1,
        )

    def reprocess_saved_batch(
        self, source_results: Iterable[dict[str, Any]]
    ) -> tuple[Path, list[dict[str, Any]]]:
        """Replay a saved benchmark locally and keep its model outputs immutable."""

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        batch_dir = (
            self.runs_root / "batches" / f"{stamp}_{uuid4().hex[:8]}_local_replay"
        )
        batch_dir.mkdir(parents=True, exist_ok=False)
        results_path = batch_dir / "results.jsonl"
        results: list[dict[str, Any]] = []
        for source in source_results:
            try:
                source_run_dir = Path(str(source["run_dir"]))
                result = self.reprocess_saved_run(source_run_dir)
                row = {
                    "id": source.get("id"),
                    "subject": source.get("subject"),
                    "status": "completed",
                    "quality_gate_pass": result.quality_gate_pass,
                    "schema_pass": result.validation["schema"]["valid"],
                    "relevance_pass": result.validation["relevance"]["valid"],
                    "latency_pass": result.validation["latency"]["valid"],
                    "answer_guard_pass": result.validation["answer_guard"]["valid"],
                    "display_ready": result.validation["display_ready"],
                    "redaction_count": result.validation["answer_guard"].get(
                        "redaction_count", 0
                    ),
                    "transport_normalized": result.validation[
                        "transport_normalization"
                    ]["applied"],
                    "render": result.validation.get("render"),
                    "model_answer_screenshot": result.validation.get(
                        "model_answer_screenshot"
                    ),
                    "latency": result.validation["latency"],
                    "provider_model": result.validation.get("provider_model"),
                    "usage": result.validation.get("usage"),
                    "source_quality": source.get("source_quality") or {},
                    "source_run_dir": str(source_run_dir.resolve()),
                    "run_dir": str(result.run_dir),
                    "attempts": result.attempts,
                    "model_calls": 0,
                }
            except Exception as exc:
                row = {
                    "id": source.get("id"),
                    "subject": source.get("subject"),
                    "status": "local_reprocess_failed",
                    "quality_gate_pass": False,
                    "source_run_dir": source.get("run_dir"),
                    "model_calls": 0,
                    "error": str(exc),
                }
            append_jsonl(results_path, row)
            results.append(row)
        write_json(
            batch_dir / "summary.json",
            {
                "completed_at": utc_now(),
                "mode": "local_replay_without_model_calls",
                "total": len(results),
                "completed": sum(row["status"] == "completed" for row in results),
                "failed": sum(row["status"] != "completed" for row in results),
                "quality_passed": sum(row["quality_gate_pass"] for row in results),
                "results": results,
            },
        )
        return batch_dir, results

    def run_batch(self, items: Iterable[Question]) -> tuple[Path, list[dict[str, Any]]]:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        batch_dir = self.runs_root / "batches" / f"{stamp}_{uuid4().hex[:8]}"
        batch_dir.mkdir(parents=True, exist_ok=False)
        results_path = batch_dir / "results.jsonl"
        results: list[dict[str, Any]] = []
        for item in items:
            try:
                result = self.run(item)
                row = {
                    "id": item.id,
                    "subject": item.subject,
                    "status": "completed",
                    "quality_gate_pass": result.quality_gate_pass,
                    "schema_pass": result.validation["schema"]["valid"],
                    "relevance_pass": result.validation["relevance"]["valid"],
                    "latency_pass": result.validation["latency"]["valid"],
                    "answer_guard_pass": result.validation["answer_guard"]["valid"],
                    "display_ready": result.validation["display_ready"],
                    "redaction_count": result.validation["answer_guard"].get(
                        "redaction_count", 0
                    ),
                    "transport_normalized": result.validation[
                        "transport_normalization"
                    ]["applied"],
                    "render": result.validation.get("render"),
                    "model_answer_screenshot": result.validation.get(
                        "model_answer_screenshot"
                    ),
                    "latency": result.validation["latency"],
                    "provider_model": result.validation.get("provider_model"),
                    "usage": result.validation.get("usage"),
                    "source_quality": item.source_quality,
                    "run_dir": str(result.run_dir),
                    "attempts": result.attempts,
                }
            except WorkflowError as exc:
                row = {
                    "id": item.id,
                    "subject": item.subject,
                    "status": "request_failed",
                    "quality_gate_pass": False,
                    "run_dir": str(exc.run_dir) if exc.run_dir else None,
                    "error": str(exc),
                }
            append_jsonl(results_path, row)
            results.append(row)
        write_json(
            batch_dir / "summary.json",
            {
                "completed_at": utc_now(),
                "total": len(results),
                "completed": sum(row["status"] == "completed" for row in results),
                "request_failed": sum(
                    row["status"] == "request_failed" for row in results
                ),
                "quality_passed": sum(row["quality_gate_pass"] for row in results),
                "results": results,
            },
        )
        return batch_dir, results


def read_jsonl(path: Path) -> list[Question]:
    items: list[Question] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("每行必须是 JSON 对象")
            items.append(Question.from_mapping(value, index=len(items) + 1))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path} 第 {line_number} 行无效：{exc}") from exc
    if not items:
        raise ValueError(f"{path} 中没有题目")
    return items
