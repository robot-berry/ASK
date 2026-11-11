from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .answer_guard import ANSWER_GUARD_VERSION
from .artifacts import export_comparison_artifacts
from .benchmark import build_benchmark_summary, validate_full_promotion
from .config import Settings
from .dataset import EXPECTED_DATASET_ID, prepare_testzip_items
from .models import Question
from .prompting import prompt_path, read_prompt, read_schema, schema_path, sha256_text
from .render import render_benchmark, render_run
from .review_server import serve_review
from .schema import parse_json_content, validate_guidance
from .storage import write_json
from .workflow import PROJECT_ROOT, GuidanceWorkflow, WorkflowError, read_jsonl


DEFAULT_DATASET = PROJECT_ROOT.parents[1] / "datasets" / EXPECTED_DATASET_ID
DEFAULT_OCR_RESULTS = (
    PROJECT_ROOT.parent
    / "test7-OCR"
    / "artifacts"
    / "runs"
    / "official_full298_v14_behavioral"
)


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def add_question_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--id", default="question", help="题目编号，用于结果目录命名")
    parser.add_argument("--subject", required=True, help="math/数学 或 physics/物理")
    parser.add_argument("--stage", required=True, help="primary/小学、junior/初中、senior/高中")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--question", help="题面文字")
    source.add_argument("--question-file", type=Path, help="UTF-8 题面文本文件")
    parser.add_argument("--ocr", default="", help="可选的不可信 OCR 辅助文字")
    parser.add_argument("--source-image", default="", help="可选来源图片路径，仅留档，不上传")
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="将 question 标记为未人工确认的 OCR，而不是可信题面",
    )


def question_from_args(args: argparse.Namespace) -> Question:
    question = args.question
    if args.question_file:
        question = args.question_file.read_text(encoding="utf-8-sig")
    return Question.from_mapping(
        {
            "id": args.id,
            "subject": args.subject,
            "stage": args.stage,
            "question": question,
            "ocr": args.ocr,
            "source_image": args.source_image,
            "question_source": "ocr_only" if args.ocr_only else "verified_text",
        }
    )


def add_repair_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repair",
        action="store_true",
        help="质量失败时额外调用一次模型；基准评测始终禁用",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test8-ans",
        description="调用 Qwen-Max 生成数学、物理题五档针对性解析",
    )
    parser.add_argument("--model", help="临时覆盖 TEST8_MODEL")
    parser.add_argument(
        "--response-format",
        choices=("json_object", "json_schema"),
        help="临时覆盖结构化输出模式",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("doctor", help="离线检查配置、提示词和 Schema")

    preview = commands.add_parser("preview", help="离线预览将发送给模型的请求")
    add_question_arguments(preview)
    preview.add_argument("--output", type=Path, help="将预览写入 JSON 文件")

    run = commands.add_parser("run", help="调用模型处理单题")
    add_question_arguments(run)
    add_repair_argument(run)

    batch = commands.add_parser("batch", help="顺序处理 JSONL 题集")
    batch.add_argument("--input", type=Path, required=True, help="JSONL 题集")
    batch.add_argument("--limit", type=int, help="只处理前 N 题")
    add_repair_argument(batch)

    benchmark = commands.add_parser(
        "benchmark", help="按 probe/smoke/full 阶段评测冻结 testzip 数据集"
    )
    benchmark.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    benchmark.add_argument("--ocr-results", type=Path, default=DEFAULT_OCR_RESULTS)
    benchmark.add_argument(
        "--phase", choices=("probe", "smoke", "sample", "full"), default="probe"
    )
    benchmark.add_argument(
        "--stage",
        required=True,
        help="数据集学段未知，必须显式指定 primary/junior/senior",
    )
    benchmark.add_argument("--limit", type=int)
    benchmark.add_argument(
        "--skip-image-hash-check", action="store_true", help="跳过原图 SHA-256 检查"
    )
    benchmark.add_argument(
        "--promotion-summary",
        type=Path,
        help="full 阶段必须提供已通过门槛的 smoke benchmark_summary.json",
    )
    benchmark.add_argument(
        "--dry-run", action="store_true", help="只检查数据准备，不调用模型"
    )

    render = commands.add_parser(
        "render", help="质量门槛通过后，将每题五层解析渲染为 800px PNG"
    )
    render.add_argument("--benchmark-summary", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)

    render_one = commands.add_parser(
        "render-run", help="将一个已完成运行的脱敏五层解析渲染为 800px PNG"
    )
    render_one.add_argument("--run-dir", type=Path, required=True)
    render_one.add_argument("--output-dir", type=Path, required=True)

    reprocess = commands.add_parser(
        "reprocess-run", help="不调用模型，重新处理一个已保存的原始回复"
    )
    reprocess.add_argument("--source-run-dir", type=Path, required=True)

    reprocess_benchmark = commands.add_parser(
        "reprocess-benchmark",
        help="不调用模型，按当前本地规则重放一个已保存的整批评测",
    )
    reprocess_benchmark.add_argument(
        "--benchmark-summary", type=Path, required=True
    )

    review = commands.add_parser(
        "review", help="在本机打开审核页，并把逐题反馈写入批次目录"
    )
    review.add_argument("--batch-dir", type=Path, required=True)
    review.add_argument(
        "--port",
        type=int,
        default=0,
        help="本地端口；默认自动选择可用端口",
    )

    validate = commands.add_parser("validate", help="离线校验已有模型 JSON")
    validate.add_argument("--subject", required=True, help="math/数学 或 physics/物理")
    validate.add_argument("--file", type=Path, required=True, help="待校验 JSON 文件")
    return parser


def doctor(settings: Settings) -> int:
    checks: list[dict[str, Any]] = []
    for subject in ("math", "physics"):
        prompt_file = prompt_path(PROJECT_ROOT, subject)
        schema_file = schema_path(PROJECT_ROOT, subject)
        try:
            prompt = read_prompt(PROJECT_ROOT, subject)
            schema = read_schema(PROJECT_ROOT, subject)
            checks.append(
                {
                    "subject": subject,
                    "prompt": str(prompt_file),
                    "prompt_sha256": sha256_text(prompt),
                    "schema": str(schema_file),
                    "schema_id": schema.get("$id"),
                    "ok": True,
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "subject": subject,
                    "prompt": str(prompt_file),
                    "schema": str(schema_file),
                    "ok": False,
                    "error": str(exc),
                }
            )
    report = {
        "ok": all(check["ok"] for check in checks),
        "api_key_configured": bool(settings.api_key),
        "endpoint": settings.chat_completions_url,
        "model": settings.model,
        "response_format": settings.response_format,
        "repair_attempts": settings.repair_attempts,
        "raw_first_output_policy": True,
        "display_policy": "targeted_final_answer_redaction_before_render",
        "answer_guard_version": ANSWER_GUARD_VERSION,
        "latency_gate": {
            "scope": settings.latency_scope,
            "budget_ms": settings.latency_budget_ms,
            "minimum_pass_rate": settings.min_latency_pass_rate,
        },
        "quality_thresholds": {
            "schema": settings.min_schema_pass_rate,
            "relevance": settings.min_relevance_pass_rate,
            "answer_guard": 1.0,
            "combined": settings.min_combined_pass_rate,
        },
        "default_dataset": str(DEFAULT_DATASET),
        "default_ocr_results": str(DEFAULT_OCR_RESULTS),
        "files": checks,
        "note": "doctor 不会发起网络请求。",
    }
    print_json(report)
    return 0 if report["ok"] else 2


def run_benchmark(args: argparse.Namespace, settings: Settings) -> int:
    benchmark_settings = replace(settings, repair_attempts=0)
    items, preparation = prepare_testzip_items(
        dataset_root=args.dataset,
        ocr_results=args.ocr_results,
        phase=args.phase,
        stage=args.stage,
        limit=args.limit,
        verify_image_hashes=not args.skip_image_hash_check,
    )
    if args.dry_run:
        print_json(
            {
                "status": "dataset_ready",
                "preparation": preparation,
                "first_item": items[0].as_dict(),
                "model_calls": 0,
            }
        )
        return 0
    if args.phase == "full":
        if not args.promotion_summary:
            raise ValueError("full 阶段必须提供 --promotion-summary")
        validate_full_promotion(
            summary_path=args.promotion_summary,
            settings=benchmark_settings,
            project_root=PROJECT_ROOT,
            dataset_id=EXPECTED_DATASET_ID,
            stage=args.stage,
        )
    workflow = GuidanceWorkflow(benchmark_settings)
    batch_dir, results = workflow.run_batch(items)
    summary = build_benchmark_summary(
        batch_dir=batch_dir,
        phase=args.phase,
        stage=args.stage,
        preparation=preparation,
        results=results,
        settings=benchmark_settings,
        project_root=PROJECT_ROOT,
    )
    comparison = export_comparison_artifacts(batch_dir, results)
    summary["comparison_artifacts"] = comparison
    write_json(batch_dir / "benchmark_summary.json", summary)
    print_json(
        {
            "status": "gate_passed" if summary["phase_gate_pass"] else "gate_failed",
            "phase": args.phase,
            "summary": str((batch_dir / "benchmark_summary.json").resolve()),
            "rates": summary["rates"],
            "latency": summary["latency"],
            "promotion_gate_pass": summary["promotion_gate_pass"],
            "correctness_scored": False,
            "comparison_artifacts": comparison,
        }
    )
    return 0 if summary["phase_gate_pass"] else 2


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_environment(PROJECT_ROOT)
        if args.model:
            settings = replace(settings, model=args.model)
        if args.response_format:
            settings = replace(settings, response_format=args.response_format)
        settings.validate(require_api_key=False)

        if args.command == "doctor":
            return doctor(settings)
        if args.command == "review":
            serve_review(args.batch_dir, port=args.port)
            return 0
        if args.command == "validate":
            item = Question.from_mapping(
                {
                    "subject": args.subject,
                    "stage": "junior",
                    "question": "仅用于选择校验规则",
                }
            )
            content = args.file.read_text(encoding="utf-8-sig")
            answer = parse_json_content(content)
            report = validate_guidance(answer, item.subject)
            print_json(report)
            return 0 if report["valid"] else 2
        if args.command == "benchmark":
            return run_benchmark(args, settings)
        if args.command == "render":
            report = render_benchmark(args.benchmark_summary, args.output_dir)
            print_json(
                {
                    "status": "rendered",
                    "output_dir": str(args.output_dir.resolve()),
                    "rendered_count": report["rendered_count"],
                    "width_px": report["width_px"],
                }
            )
            return 0
        if args.command == "render-run":
            report = render_run(args.run_dir, args.output_dir)
            print_json(
                {
                    "status": "rendered",
                    "run_dir": str(args.run_dir.resolve()),
                    "output_dir": str(args.output_dir.resolve()),
                    "images": report["images"],
                }
            )
            return 0
        if args.command == "reprocess-run":
            workflow = GuidanceWorkflow(settings)
            result = workflow.reprocess_saved_run(args.source_run_dir)
            print_json(
                {
                    "status": (
                        "quality_passed"
                        if result.quality_gate_pass
                        else "completed_with_quality_failure"
                    ),
                    "model_calls": 0,
                    "run_dir": str(result.run_dir.resolve()),
                    "validation": result.validation,
                }
            )
            return 0 if result.quality_gate_pass else 2
        if args.command == "reprocess-benchmark":
            source_summary = json.loads(
                args.benchmark_summary.read_text(encoding="utf-8-sig")
            )
            source_results = source_summary.get("results")
            if not isinstance(source_results, list) or not source_results:
                raise ValueError("benchmark_summary.json 中没有可重放的 results")
            replay_settings = replace(
                settings,
                model=str(source_summary.get("model") or settings.model),
                response_format=str(
                    source_summary.get("response_format") or settings.response_format
                ),
                repair_attempts=0,
            )
            workflow = GuidanceWorkflow(replay_settings)
            batch_dir, results = workflow.reprocess_saved_batch(source_results)
            summary = build_benchmark_summary(
                batch_dir=batch_dir,
                phase=str(source_summary.get("phase") or "sample"),
                stage=str(source_summary.get("stage_override") or "senior"),
                preparation=dict(source_summary.get("dataset") or {}),
                results=results,
                settings=replay_settings,
                project_root=PROJECT_ROOT,
            )
            summary["source_benchmark_summary"] = str(
                args.benchmark_summary.resolve()
            )
            summary["model_calls_in_reprocess"] = 0
            comparison = export_comparison_artifacts(batch_dir, results)
            summary["comparison_artifacts"] = comparison
            write_json(batch_dir / "benchmark_summary.json", summary)
            print_json(
                {
                    "status": "local_replay_completed",
                    "model_calls": 0,
                    "summary": str((batch_dir / "benchmark_summary.json").resolve()),
                    "rates": summary["rates"],
                    "latency": summary["latency"],
                    "comparison_artifacts": comparison,
                }
            )
            return 0

        if getattr(args, "repair", False):
            settings = replace(settings, repair_attempts=1)
        else:
            settings = replace(settings, repair_attempts=0)
        workflow = GuidanceWorkflow(settings)
        if args.command == "preview":
            preview = workflow.preview(question_from_args(args))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(preview, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print_json({"status": "previewed", "output": str(args.output.resolve())})
            else:
                print_json(preview)
            return 0
        if args.command == "run":
            result = workflow.run(question_from_args(args))
            print_json(
                {
                    "status": (
                        "quality_passed"
                        if result.quality_gate_pass
                        else "completed_with_quality_failure"
                    ),
                    "run_dir": str(result.run_dir.resolve()),
                    "raw_model_output": result.raw_output,
                    "attempts": result.attempts,
                    "validation": result.validation,
                    "answer": result.answer,
                    "display_answer": result.display_answer,
                }
            )
            return 0 if result.quality_gate_pass else 2
        if args.command == "batch":
            items = read_jsonl(args.input)
            if args.limit is not None:
                if args.limit < 1:
                    raise ValueError("--limit 必须大于 0")
                items = items[: args.limit]
            batch_dir, results = workflow.run_batch(items)
            request_failed = sum(row["status"] == "request_failed" for row in results)
            quality_failed = sum(not row["quality_gate_pass"] for row in results)
            print_json(
                {
                    "status": (
                        "quality_passed"
                        if not request_failed and not quality_failed
                        else "completed_with_quality_failure"
                    ),
                    "batch_dir": str(batch_dir.resolve()),
                    "total": len(results),
                    "request_failed": request_failed,
                    "quality_failed": quality_failed,
                }
            )
            return 0 if not request_failed and not quality_failed else 2
        parser.error("未知命令")
    except (OSError, ValueError, WorkflowError, json.JSONDecodeError) as exc:
        payload: dict[str, Any] = {"status": "failed", "error": str(exc)}
        if isinstance(exc, WorkflowError) and exc.run_dir:
            payload["run_dir"] = str(exc.run_dir.resolve())
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    return 2
