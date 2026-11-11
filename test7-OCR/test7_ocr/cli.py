from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .capture_pair_review import CapturePairStore
from .clean_source import VerifiedCleanSourceResolver
from .pipeline import QuestionPipeline, load_config
from .deferred_verification import verify_run_artifacts
from .evaluation import evaluate_run
from .external_datasets import (
    default_review_eligible_for_role,
    infer_dataset_role,
    is_external_eval_role,
)
from .external_eval import evaluate_external_run, write_external_eval_report
from .providers import build_provider
from .release import NoPublishableSamplesError, publish_release
from .resource_guard import probe_batch_resource_conflicts
from .rerender import rerender_run
from .runtime_profiles import (
    PRODUCT_V7_SAFE_RUNTIME_PROFILE,
    RUNTIME_PROFILE_CHOICES,
    apply_runtime_profile,
    authenticated_source_sample_id,
)
from .self_built_intake import SelfBuiltIntakeStore
from .storage import new_run_id, preflight_batch_storage, safe_name, write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--provider",
        choices=["auto", "hybrid", "structured", "rapidocr", "sidecar", "eval-prelabel"],
        default="auto",
    )
    parser.add_argument("--config", type=_path, default=PROJECT_ROOT / "configs" / "pipeline.json")
    parser.add_argument("--output-root", type=_path, default=PROJECT_ROOT / "artifacts" / "runs")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--runtime-profile",
        choices=RUNTIME_PROFILE_CHOICES,
        default="standard",
        help="Use the standard safety profile or the exact bounded product-v7 runtime profile",
    )
    parser.add_argument("--sidecar", type=_path, default=None)
    parser.add_argument(
        "--review-dir",
        type=_path,
        default=PROJECT_ROOT / "data" / "annotations" / "gaokao_physics_ocr_gold" / "drafts",
    )
    parser.add_argument("--allow-eval-prelabel", action="store_true")
    parser.add_argument("--service-url", default="http://127.0.0.1:8240")
    parser.add_argument(
        "--visual-candidates",
        type=_path,
        default=PROJECT_ROOT / "data" / "splits" / "gaokao_physics_visual_bbox_candidates.jsonl",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="test7-ocr", description="Math/physics question OCR and 800px reflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process one question image")
    process.add_argument("image", type=_path)
    process.add_argument("--sample-id", required=True)
    _common_parser(process)

    batch = subparsers.add_parser("batch", help="Process a dataset manifest")
    batch.add_argument(
        "--manifest",
        type=_path,
        default=PROJECT_ROOT / "data" / "splits" / "gaokao_physics_ocr_gold300_manifest.jsonl",
    )
    batch.add_argument(
        "--dataset-root",
        type=_path,
        default=(PROJECT_ROOT / ".." / ".." / "datasets" / "gaokao_physics_300_v1_0_1").resolve(),
    )
    batch.add_argument("--limit", type=int, default=None)
    batch.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Process only this sample ID; repeat for targeted regression sets",
    )
    batch.add_argument("--include-duplicates", action="store_true")
    batch.add_argument("--resume", action="store_true", help="Reuse hash-matched reports already present in this run")
    batch.add_argument(
        "--allow-resource-conflict",
        action="store_true",
        help="Run a diagnostic batch even when a configured competing local OCR service is active",
    )
    batch.add_argument(
        "--storage-preflight-only",
        action="store_true",
        help="Estimate required artifact space and exit without loading OCR models",
    )
    batch.add_argument("--subject", choices=["mathematics", "physics"], default=None)
    batch.add_argument(
        "--priority",
        action="append",
        choices=["p0_handwriting_or_noise", "p1_disagreement_or_known_error", "p2_agreed_prelabel", "excluded"],
        help="Select one or more review-priority buckets",
    )
    batch.add_argument(
        "--dataset-role",
        default=None,
        help="Override inferred dataset role; external_eval_only excludes the run from default review",
    )
    _common_parser(batch)

    evaluate = subparsers.add_parser("evaluate", help="Score a completed run against verified OCR Gold")
    evaluate.add_argument("run_directory", type=_path)
    evaluate.add_argument(
        "--gold",
        type=_path,
        default=PROJECT_ROOT / "data" / "gold" / "gaokao_physics_300" / "ocr_gold.jsonl",
    )
    evaluate.add_argument("--required-gold-count", type=int, default=300)
    evaluate.add_argument("--allow-partial-gold", action="store_true")

    external_eval = subparsers.add_parser(
        "external-eval",
        help="Score a run against an isolated external gold file; never enables formal OCR scoring",
    )
    external_eval.add_argument("run_directory", type=_path)
    external_eval.add_argument("--gold", type=_path, required=True)
    external_eval.add_argument("--dataset-id", required=True)
    external_eval.add_argument("--scoring-mode", default=None)
    external_eval.add_argument("--output", type=_path, default=None)

    release = subparsers.add_parser("release", help="Publish only fully verified PASS artifacts")
    release.add_argument("run_directory", type=_path)
    release.add_argument("--release-id", default=None)
    release.add_argument(
        "--release-root",
        type=_path,
        default=PROJECT_ROOT / "artifacts" / "releases",
    )
    release.add_argument(
        "--output-review-root",
        type=_path,
        default=PROJECT_ROOT / "data" / "annotations" / "output_quality_reviews",
        help="Hash-bound human final-PNG reviews required for publication",
    )

    verify_run = subparsers.add_parser(
        "verify-run",
        help="Run deferred final-PNG readback without charging it to the 5s online SLA",
    )
    verify_run.add_argument("run_directory", type=_path)
    verify_run.add_argument(
        "--config", type=_path, default=PROJECT_ROOT / "configs" / "pipeline.json"
    )
    verify_run.add_argument("--timeout-seconds", type=float, default=30.0)
    verify_run.add_argument("--limit", type=int, default=None)
    verify_run.add_argument("--force", action="store_true")

    rerender = subparsers.add_parser(
        "rerender",
        help="Rebuild PNGs from an existing run without invoking recognition models",
    )
    rerender.add_argument("source_run", type=_path)
    rerender.add_argument("--run-id", required=True)
    rerender.add_argument("--output-root", type=_path, default=PROJECT_ROOT / "artifacts" / "runs")
    rerender.add_argument("--config", type=_path, default=PROJECT_ROOT / "configs" / "pipeline.json")
    rerender.add_argument("--limit", type=int, default=None)
    rerender.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="Rerender only this sample ID; repeat the option for multiple samples",
    )
    rerender.add_argument("--resume", action="store_true")
    return parser


def _provider_from_args(args: argparse.Namespace, config: dict[str, Any]):
    recognition = config.get("recognition", {})
    render_verification = config.get("render_verification", {})
    preprocess = config.get("preprocess", {})
    return build_provider(
        args.provider,
        sidecar_path=args.sidecar,
        review_dir=args.review_dir,
        visual_candidate_file=args.visual_candidates,
        service_url=args.service_url,
        service_timeout_seconds=float(
            recognition.get("structured_timeout_seconds", 15.0)
        ),
        allow_eval_prelabel=args.allow_eval_prelabel,
        verify_roman_headers=bool(
            recognition.get("verify_roman_question_headers", False)
        ),
        header_min_confidence=float(
            recognition.get("question_header_min_confidence", 0.90)
        ),
        header_verifier_params=render_verification.get("rapidocr_params"),
        verify_suspicious_option_formulas=bool(
            recognition.get("verify_suspicious_option_formulas", False)
        ),
        option_formula_min_confidence=float(
            recognition.get("option_formula_min_confidence", 0.92)
        ),
        refine_stacked_fraction_formulas=bool(
            recognition.get("refine_stacked_fraction_formulas", False)
        ),
        refined_formula_min_confidence=float(
            recognition.get("refined_formula_min_confidence", 0.50)
        ),
        recover_missing_leading_option_labels=bool(
            recognition.get("recover_missing_leading_option_labels", False)
        ),
        option_label_min_confidence=float(
            recognition.get("option_label_min_confidence", 0.95)
        ),
        verify_photo_visual_handwriting=bool(
            recognition.get("verify_photo_visual_handwriting", False)
        ),
        visual_low_confidence_threshold=float(
            recognition.get("visual_low_confidence_threshold", 0.65)
        ),
        visual_low_confidence_token_count=int(
            recognition.get("visual_low_confidence_token_count", 2)
        ),
        rapidocr_params=recognition.get("rapidocr_params"),
        transient_retry_attempts=int(
            recognition.get("structured_transient_retry_attempts", 5)
        ),
        transient_retry_base_seconds=float(
            recognition.get("structured_transient_retry_base_seconds", 0.5)
        ),
        transient_retry_max_seconds=float(
            recognition.get("structured_transient_retry_max_seconds", 4.0)
        ),
        outer_white_threshold=int(preprocess.get("outer_white_threshold", 248)),
        outer_crop_padding_px=int(preprocess.get("outer_crop_padding_px", 16)),
    )


def _configured_path(value: object) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Clean-source resolution path cannot be empty")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _clean_source_resolver_from_config(
    config: dict[str, Any],
) -> VerifiedCleanSourceResolver | None:
    settings = config.get("clean_source_resolution") or {}
    if not settings.get("enabled", False):
        return None
    if settings.get("require_two_person_verified_pair") is not True:
        raise ValueError("Clean-source resolution cannot disable two-person verification")
    allowed_roles = {
        str(value) for value in settings.get("allowed_capture_roles") or ["handwriting_capture"]
    }
    if not allowed_roles or not allowed_roles.issubset({"handwriting_capture"}):
        raise ValueError("Only handwriting_capture is supported for clean-source resolution")
    intake = SelfBuiltIntakeStore(
        _configured_path(settings["self_built_dataset_root"]),
        _configured_path(settings["intake_record_directory"]),
        _configured_path(settings["intake_summary_path"]),
    )
    pairs = CapturePairStore(
        _configured_path(settings["capture_pair_task_directory"]),
        _configured_path(settings["capture_pair_manifest_path"]),
        _configured_path(settings["capture_pair_summary_path"]),
        intake,
    )
    return VerifiedCleanSourceResolver(
        pairs,
        allowed_roles=allowed_roles,
        fail_closed_on_invalid_evidence=bool(
            settings.get("fail_closed_on_invalid_evidence", True)
        ),
    )


def _run_process(args: argparse.Namespace) -> int:
    run_id = args.run_id or new_run_id()
    config = apply_runtime_profile(
        load_config(args.config),
        getattr(args, "runtime_profile", "standard"),
    )
    pipeline = QuestionPipeline(
        _provider_from_args(args, config),
        config,
        args.output_root,
        run_id,
        _clean_source_resolver_from_config(config),
    )
    try:
        report = pipeline.process(args.image, args.sample_id)
        stored_report = json.loads(
            (Path(report.question_document).parent / "report.json").read_text(encoding="utf-8")
        )
    finally:
        pipeline.close()
    print(json.dumps(stored_report, ensure_ascii=False, indent=2))
    return 2 if report.status == "REJECT" else 0


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def _record_sample_id(record: dict[str, Any]) -> str:
    value = record.get("sample_id") or record.get("question_id")
    if not value:
        raise ValueError("Manifest record is missing sample_id/question_id")
    return str(value)


def _latency_guard_decision(
    elapsed_values: list[float],
    target_seconds: float,
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    if not settings.get("enabled", True):
        return None
    window_size = max(1, int(settings.get("window_size", 5)))
    minimum_processed = max(
        window_size,
        int(settings.get("minimum_processed_samples", 20)),
    )
    maximum_over_target = max(
        1,
        min(
            window_size,
            int(settings.get("maximum_over_target_in_window", 3)),
        ),
    )
    if len(elapsed_values) < minimum_processed:
        return None
    window = elapsed_values[-window_size:]
    over_target = [value for value in window if value > target_seconds]
    if len(over_target) < maximum_over_target:
        return None
    return {
        "status": "ABORTED",
        "reason": "SUSTAINED_LATENCY_OVER_TARGET",
        "processed_sample_count": len(elapsed_values),
        "target_seconds": target_seconds,
        "window_size": window_size,
        "maximum_over_target_in_window": maximum_over_target,
        "window_seconds": [round(value, 6) for value in window],
        "over_target_count": len(over_target),
    }


def _run_batch(args: argparse.Namespace) -> int:
    batch_started = time.perf_counter()
    run_id = args.run_id or new_run_id()
    output_root = args.output_root.resolve()
    runtime_profile = getattr(args, "runtime_profile", "standard")
    dataset_role = infer_dataset_role(
        args.dataset_root,
        args.manifest,
        explicit=getattr(args, "dataset_role", None),
    )
    default_review_eligible = default_review_eligible_for_role(dataset_role)
    config = apply_runtime_profile(load_config(args.config), runtime_profile)
    pipeline: QuestionPipeline | None = None
    records = _load_manifest(args.manifest)
    if not args.include_duplicates:
        records = [record for record in records if record.get("include_in_benchmark", True)]
    if args.subject:
        records = [record for record in records if record.get("subject") == args.subject]
    if args.priority:
        selected_priorities = set(args.priority)
        records = [record for record in records if record.get("review_priority") in selected_priorities]
    if args.sample_id:
        selected_sample_ids = set(args.sample_id)
        records = [
            record
            for record in records
            if _record_sample_id(record) in selected_sample_ids
        ]
    if args.limit is not None:
        records = records[: max(0, args.limit)]

    target_run_root = output_root / safe_name(run_id)
    pending_storage_count = sum(
        not (
            args.resume
            and (
                target_run_root / safe_name(_record_sample_id(record)) / "report.json"
            ).is_file()
        )
        for record in records
    )
    storage_preflight = preflight_batch_storage(
        output_root,
        pending_storage_count,
        config.get("storage", {}),
    )
    storage_preflight.update(
        {
            "run_id": run_id,
            "selected_sample_count": len(records),
            "resume_requested": bool(args.resume),
        }
    )
    if args.storage_preflight_only:
        if storage_preflight["sufficient"]:
            write_json_atomic(output_root.parent / "STORAGE_PREFLIGHT.json", storage_preflight)
        print(json.dumps(storage_preflight, ensure_ascii=False, indent=2))
        return 0 if storage_preflight["sufficient"] else 3
    if config.get("storage", {}).get("preflight_enabled", True) and not storage_preflight["sufficient"]:
        print(json.dumps(storage_preflight, ensure_ascii=False, indent=2))
        return 3

    resource_guard_settings = config.get("batch_resource_guard", {})
    resource_guard = probe_batch_resource_conflicts(resource_guard_settings)
    resource_guard["override_requested"] = bool(args.allow_resource_conflict)
    resource_guard["check_every_samples"] = max(
        1, int(resource_guard_settings.get("check_every_samples", 10))
    )
    resource_guard_checks = [resource_guard]
    if resource_guard["status"] == "CONFLICT" and not args.allow_resource_conflict:
        blocked = {
            "status": "BLOCKED_RESOURCE_CONFLICT",
            "run_id": run_id,
            "message": "A configured competing local OCR service is active; no OCR model was loaded and no candidate run was created.",
            "resource_guard": resource_guard,
        }
        print(json.dumps(blocked, ensure_ascii=False, indent=2))
        return 4

    statuses: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    reports: list[dict[str, Any]] = []
    resumed_count = 0
    processed_count = 0
    first_processed_elapsed: float | None = None
    resource_conflict_abort: dict[str, Any] | None = None
    target_seconds = float(config["quality"]["target_latency_seconds"])
    latency_guard_settings = config.get("batch_latency_guard", {})
    processed_elapsed_values: list[float] = []
    latency_guard_abort: dict[str, Any] | None = None
    try:
        for record_index, record in enumerate(records):
            if (
                resource_guard.get("enabled")
                and record_index > 0
                and record_index % resource_guard["check_every_samples"] == 0
            ):
                latest_guard = probe_batch_resource_conflicts(resource_guard_settings)
                latest_guard["checked_before_record_index"] = record_index
                resource_guard_checks.append(latest_guard)
                if latest_guard["status"] == "CONFLICT" and not args.allow_resource_conflict:
                    resource_conflict_abort = latest_guard
                    break
            sample_id = _record_sample_id(record)
            image_path = args.dataset_root / str(record["image_path"])
            existing_report_path = output_root / safe_name(run_id) / safe_name(sample_id) / "report.json"
            if args.resume and existing_report_path.is_file():
                try:
                    existing = json.loads(existing_report_path.read_text(encoding="utf-8"))
                    expected_hash = str(record.get("image_sha256") or record.get("source_sha256") or "")
                    if str(existing.get("sample_id")) != sample_id:
                        raise ValueError("sample_id mismatch")
                    existing_input_hash = str(
                        existing.get("input_source_sha256") or existing.get("source_sha256") or ""
                    )
                    if expected_hash and existing_input_hash != expected_hash:
                        raise ValueError("source image hash mismatch")
                    status = str(existing.get("status") or "ERROR")
                    statuses[status] += 1
                    reports.append(existing)
                    resumed_count += 1
                    continue
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    statuses["ERROR"] += 1
                    failures.append({"sample_id": sample_id, "error": f"RESUME_REPORT_INVALID:{type(exc).__name__}:{exc}"})
                    continue
            try:
                if pipeline is None:
                    pipeline = QuestionPipeline(
                        _provider_from_args(args, config),
                        config,
                        output_root,
                        run_id,
                        _clean_source_resolver_from_config(config),
                    )
                report = pipeline.process(
                    image_path,
                    sample_id,
                    recognition_sample_id=(
                        authenticated_source_sample_id(sample_id)
                        if runtime_profile == PRODUCT_V7_SAFE_RUNTIME_PROFILE
                        else None
                    ),
                    scoring_mode=(
                        str(record.get("scoring_mode") or "")
                        or (
                            "formula_only"
                            if str(record.get("dataset_id") or "") == "unimer_test"
                            else None
                        )
                    )
                    or None,
                )
                statuses[report.status] += 1
                reports.append(report.to_dict())
                processed_count += 1
                processed_elapsed_values.append(float(report.elapsed_seconds))
                if first_processed_elapsed is None:
                    first_processed_elapsed = report.elapsed_seconds
                latency_guard_abort = _latency_guard_decision(
                    processed_elapsed_values,
                    target_seconds,
                    latency_guard_settings,
                )
                if latency_guard_abort is not None:
                    latency_guard_abort["sample_id"] = sample_id
                    if is_external_eval_role(dataset_role):
                        latency_guard_abort["ignored_for_external_eval"] = True
                        latency_guard_abort = None
                    else:
                        break
            except Exception as exc:  # keep a dataset run auditable instead of stopping at one corrupt item
                statuses["ERROR"] += 1
                failures.append({"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if pipeline is not None:
            pipeline.close()

    elapsed_values = [float(report["elapsed_seconds"]) for report in reports]
    phase_names = sorted(
        {
            phase
            for report in reports
            for phase in (report.get("timings_ms") or {}).keys()
        }
    )

    def percentile(values: list[float], ratio: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * ratio
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    phase_metrics: dict[str, dict[str, float | None]] = {}
    for phase_name in phase_names:
        values = [
            float(report["timings_ms"][phase_name])
            for report in reports
            if phase_name in (report.get("timings_ms") or {})
        ]
        phase_metrics[phase_name] = {
            "mean": round(sum(values) / len(values), 3) if values else None,
            "p95": round(percentile(values, 0.95), 3) if values else None,
        }
    wall_seconds = time.perf_counter() - batch_started
    summary = {
        "schema_version": "batch_summary_v1",
        "run_id": run_id,
        "provider": args.provider,
        "runtime_profile": runtime_profile,
        "manifest": str(args.manifest.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "selection": {
            "subject": args.subject,
            "priorities": args.priority or [],
            "sample_ids": args.sample_id or [],
        },
        "attempted": len(records),
        "processed_this_invocation": processed_count,
        "resumed_from_existing_reports": resumed_count,
        "completed_sample_count": len(reports),
        "statuses": dict(sorted(statuses.items())),
        "source_resolution": {
            "direct_inputs": sum(
                not bool((report.get("source_resolution") or {}).get("substituted"))
                for report in reports
            ),
            "verified_clean_source_substitutions": sum(
                bool((report.get("source_resolution") or {}).get("substituted"))
                for report in reports
            ),
            "substitution_requires_two_person_verified_capture_pair": True,
        },
        "latency_seconds": {
            "mean": round(sum(elapsed_values) / len(elapsed_values), 6) if elapsed_values else None,
            "p50": round(percentile(elapsed_values, 0.50), 6) if elapsed_values else None,
            "p95": round(percentile(elapsed_values, 0.95), 6) if elapsed_values else None,
            "max": round(max(elapsed_values), 6) if elapsed_values else None,
            "first_sample": round(elapsed_values[0], 6) if elapsed_values else None,
            "target": target_seconds,
            "under_target_count": sum(value <= target_seconds for value in elapsed_values),
            "under_target_ratio": round(
                sum(value <= target_seconds for value in elapsed_values) / len(elapsed_values), 6
            ) if elapsed_values else None,
        },
        "phase_timings_ms": phase_metrics,
        "wall_seconds": round(wall_seconds, 6),
        "throughput_questions_per_second": round(processed_count / wall_seconds, 6) if wall_seconds else None,
        "measurement": {
            "models_reused_within_batch": True,
            "first_sample_reported_separately": True,
            "cold_service_startup_included": False,
            "pipeline_initialization_seconds": round(
                float((pipeline.initialization_timings_ms if pipeline else {}).get("total", 0.0))
                / 1000,
                6,
            ),
            "pipeline_initialization_included_in_wall_seconds": pipeline is not None,
            "pipeline_initialization_excluded_from_sample_elapsed_seconds": True,
            "first_processed_cold_equivalent_seconds": round(
                (
                    first_processed_elapsed
                    + float(
                        (pipeline.initialization_timings_ms if pipeline else {}).get("total", 0.0)
                    )
                    / 1000
                ),
                6,
            ) if first_processed_elapsed is not None and pipeline is not None else None,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "failures": failures,
        "resource_guard": {
            "status": (
                "ABORTED"
                if resource_conflict_abort is not None
                else "ALLOWED_CONFLICT"
                if args.allow_resource_conflict
                and any(check["status"] == "CONFLICT" for check in resource_guard_checks)
                else resource_guard["status"]
            ),
            "override_requested": bool(args.allow_resource_conflict),
            "check_every_samples": resource_guard["check_every_samples"],
            "checks": resource_guard_checks,
            "abort": resource_conflict_abort,
        },
        "latency_guard": {
            "status": (
                "ABORTED"
                if latency_guard_abort is not None
                else "DISABLED"
                if not latency_guard_settings.get("enabled", True)
                else "CLEAR"
            ),
            "minimum_processed_samples": max(
                int(latency_guard_settings.get("window_size", 5)),
                int(latency_guard_settings.get("minimum_processed_samples", 20)),
            ),
            "window_size": max(1, int(latency_guard_settings.get("window_size", 5))),
            "maximum_over_target_in_window": max(
                1,
                int(latency_guard_settings.get("maximum_over_target_in_window", 3)),
            ),
            "abort": latency_guard_abort,
        },
        "storage_preflight": storage_preflight,
        "dataset_role": dataset_role,
        "default_review_eligible": default_review_eligible,
        "formal_gold_scoring": False,
        "warning": (
            "Batch status is pre-Gold pipeline behavior. Run evaluate for the "
            "source-derived benchmark; real-world claims still require captured data."
            if default_review_eligible
            else "External evaluation-only run. Not eligible as the default review baseline and not formal OCR Gold."
        ),
    }
    summary_path = output_root / run_id / "batch_summary.json"
    write_json_atomic(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if resource_conflict_abort is not None:
        return 4
    if latency_guard_abort is not None:
        return 5
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    # Windows may expose a legacy GBK console even though reports legitimately
    # contain Unicode mathematical symbols.  Keep machine-readable CLI JSON in
    # UTF-8 so a successful OCR run cannot fail only while printing its report.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    args = build_parser().parse_args(argv)
    if args.command == "process":
        return _run_process(args)
    if args.command == "batch":
        return _run_batch(args)
    if args.command == "evaluate":
        try:
            summary = evaluate_run(
                args.run_directory,
                args.gold,
                required_gold_count=args.required_gold_count,
                allow_partial_gold=args.allow_partial_gold,
            )
        except RuntimeError as exc:
            print(json.dumps({"status": "LOCKED", "message": str(exc)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "external-eval":
        summary = evaluate_external_run(
            args.run_directory,
            args.gold,
            dataset_id=args.dataset_id,
            scoring_mode=args.scoring_mode,
        )
        if summary["formal_gold_scoring"]:
            print(json.dumps({"status": "LOCKED", "message": "external-eval must not enable formal scoring"}, indent=2))
            return 2
        output = args.output or (
            PROJECT_ROOT
            / "artifacts"
            / "external"
            / args.dataset_id
            / args.run_directory.name
            / "report.json"
        )
        write_external_eval_report(summary, output)
        print(json.dumps({**summary, "report_path": str(output)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "release":
        try:
            summary = publish_release(
                args.run_directory,
                args.release_root,
                release_id=args.release_id,
                output_review_root=args.output_review_root,
            )
        except (NoPublishableSamplesError, FileExistsError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"status": "LOCKED", "message": str(exc)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify-run":
        summary = verify_run_artifacts(
            args.run_directory,
            load_config(args.config),
            timeout_seconds=args.timeout_seconds,
            limit=args.limit,
            force=args.force,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if summary["failures"] else 0
    if args.command == "rerender":
        def progress(ordinal: int, total: int, sample_id: str) -> None:
            if ordinal == 1 or ordinal == total or ordinal % 10 == 0:
                print(
                    json.dumps(
                        {
                            "status": "RERENDER_PROGRESS",
                            "completed": ordinal,
                            "total": total,
                            "sample_id": sample_id,
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

        try:
            summary = rerender_run(
                args.source_run,
                args.output_root,
                args.run_id,
                load_config(args.config),
                limit=args.limit,
                sample_ids=set(args.sample_id) if args.sample_id else None,
                resume=args.resume,
                progress=progress,
            )
        except (FileExistsError, FileNotFoundError, ValueError) as exc:
            print(json.dumps({"status": "LOCKED", "message": str(exc)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1 if summary["failures"] else 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
