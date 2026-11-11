from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .clean_source import validate_source_resolution
from .image_ops import sha256_file
from .providers import RecognitionProvider, TimedRapidOcrProvider
from .storage import write_json_atomic
from .types import BBox, QuestionDocument, RecognitionResult, SourceBlock, VisualAsset
from .validators import status_from_issues, validate_result
from .verification import verify_render_readback


READBACK_WARNING_PREFIXES = (
    "RENDER_READBACK_EMPTY",
    "RENDER_READBACK_LOW_COVERAGE",
    "RENDER_READBACK_OPTION_LOSS",
    "RENDER_READBACK_LATEX_TOKEN",
    "RENDER_READBACK_FAILED",
    "RENDER_READBACK_PENDING",
)
READBACK_RETRY_ISSUES = {
    "RENDER_READBACK_LOW_COVERAGE",
    "RENDER_READBACK_OPTION_LOSS",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _document_from_dict(payload: dict[str, Any]) -> QuestionDocument:
    assets = [
        VisualAsset(
            asset_id=str(item["asset_id"]),
            source_bbox=BBox.from_value(item["source_bbox"]),
            crop_path=str(item["crop_path"]),
            crop_sha256=str(item["crop_sha256"]),
            kind=str(item.get("kind", "question_visual")),
            extraction=str(item.get("extraction", "direct_source_crop")),
            bbox_source=str(item.get("bbox_source", "unknown")),
            bbox_verified=bool(item.get("bbox_verified", False)),
            preserve_source_pixels=bool(item.get("preserve_source_pixels", True)),
            redraw_used=bool(item.get("redraw_used", False)),
            fallback=bool(item.get("fallback", False)),
            edge_complete=bool(item.get("edge_complete", True)),
        )
        for item in payload.get("visual_assets", [])
    ]
    source_blocks = [
        SourceBlock(
            block_id=str(item["block_id"]),
            kind=str(item.get("kind", "unknown")),  # type: ignore[arg-type]
            text=str(item.get("text") or ""),
            source_bbox=(
                BBox.from_value(item["source_bbox"])
                if item.get("source_bbox") is not None
                else None
            ),
            confidence=(
                float(item["confidence"])
                if item.get("confidence") is not None
                else None
            ),
            disposition=str(item.get("disposition", "emitted")),  # type: ignore[arg-type]
            reason=str(item["reason"]) if item.get("reason") is not None else None,
        )
        for item in payload.get("source_blocks", [])
    ]
    return QuestionDocument(
        schema_version=str(payload["schema_version"]),
        sample_id=str(payload["sample_id"]),
        source_image=str(payload["source_image"]),
        source_sha256=str(payload["source_sha256"]),
        provider=str(payload["provider"]),
        canonical_text=str(payload.get("canonical_text") or ""),
        display_text=str(payload.get("display_text") or ""),
        question_number=(
            str(payload["question_number"])
            if payload.get("question_number") is not None
            else None
        ),
        stem=str(payload.get("stem") or ""),
        options=[dict(item) for item in payload.get("options", [])],
        visual_assets=assets,
        handwriting_detected=bool(payload.get("handwriting_detected", False)),
        formula_verified=bool(payload.get("formula_verified", False)),
        formula_specialist_used=bool(payload.get("formula_specialist_used", False)),
        formula_count=int(payload.get("formula_count") or 0),
        warnings=[str(value) for value in payload.get("warnings", [])],
        source_blocks=source_blocks,
        source_resolution=(
            dict(payload["source_resolution"])
            if isinstance(payload.get("source_resolution"), dict)
            else None
        ),
    )


def _clear_readback_warnings(document: QuestionDocument) -> None:
    document.warnings = [
        warning
        for warning in document.warnings
        if not warning.startswith(READBACK_WARNING_PREFIXES)
    ]


def _recognition_for_revalidation(
    document: QuestionDocument,
    previous_report: dict[str, Any],
) -> RecognitionResult:
    previous_codes = {
        str(issue.get("code") or "")
        for issue in previous_report.get("issues", [])
        if isinstance(issue, dict)
    }
    previous_recognition = previous_report.get("recognition")
    previous_diagnostics = (
        previous_recognition.get("diagnostics")
        if isinstance(previous_recognition, dict)
        else None
    )
    return RecognitionResult(
        provider=document.provider,
        transcript=document.canonical_text,
        visual_bboxes=[asset.source_bbox for asset in document.visual_assets],
        visual_bboxes_verified=all(asset.bbox_verified for asset in document.visual_assets),
        handwriting_suspected=document.handwriting_detected,
        formula_verified=document.formula_verified,
        formula_specialist_used=document.formula_specialist_used,
        formula_count=document.formula_count,
        unverified_prelabel="UNVERIFIED_PRELABEL" in previous_codes,
        warnings=list(document.warnings),
        diagnostics=(
            dict(previous_diagnostics)
            if isinstance(previous_diagnostics, dict)
            else {}
        ),
    )


def _assert_artifact_identity(
    report: dict[str, Any],
    document: QuestionDocument,
    output_path: Path,
) -> None:
    if str(report.get("sample_id")) != document.sample_id:
        raise ValueError("Report/document sample_id mismatch")
    if str(report.get("source_sha256")) != document.source_sha256:
        raise ValueError("Report/document source hash mismatch")
    if document.source_resolution is not None:
        validate_source_resolution(document.source_resolution)
        if report.get("source_resolution") != document.source_resolution:
            raise ValueError("Report/document source-resolution evidence mismatch")
        if report.get("input_source_sha256") != document.source_resolution["input_sha256"]:
            raise ValueError("Report input source hash does not match source resolution")
        if report.get("source_sha256") != document.source_resolution["effective_sha256"]:
            raise ValueError("Report effective source hash does not match source resolution")
    source_path = Path(document.source_image)
    if not source_path.is_file() or sha256_file(source_path) != document.source_sha256:
        raise ValueError("Current source image does not match the recorded source hash")
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    if sha256_file(output_path) != str(report.get("output_sha256")):
        raise ValueError("Output PNG hash differs from the generation report")


def _normalize_terminal_readback_state(report_path: Path) -> bool:
    """Remove an unprocessable pending readback from an independent hard reject."""
    report = _load_json(report_path)
    readback = report.get("render_readback") or {}
    if report.get("status") != "REJECT" or readback.get("pending") is not True:
        return False
    remaining_issues = [
        issue
        for issue in report.get("issues", [])
        if isinstance(issue, dict)
        and not str(issue.get("code") or "").startswith(READBACK_WARNING_PREFIXES)
    ]
    if not any(issue.get("severity") == "reject" for issue in remaining_issues):
        return False

    document_path = Path(str(report["question_document"]))
    output_path = Path(str(report["output_image"]))
    document = _document_from_dict(_load_json(document_path))
    _assert_artifact_identity(report, document, output_path)
    _clear_readback_warnings(document)
    report["issues"] = remaining_issues
    report["render_readback"] = {
        "enabled": False,
        "skipped": True,
        "reason": "HARD_REJECT_BEFORE_READBACK",
        "passed": False,
    }
    write_json_atomic(document_path, document.to_dict())
    write_json_atomic(report_path, report)
    return True


def verify_sample_artifact(
    report_path: Path,
    config: dict[str, Any],
    provider: RecognitionProvider,
) -> dict[str, Any]:
    """Read back one immutable generated PNG and atomically refresh its gate status."""
    report = _load_json(report_path)
    document_path = Path(str(report["question_document"]))
    output_path = Path(str(report["output_image"]))
    document_payload = _load_json(document_path)
    document = _document_from_dict(document_payload)
    _assert_artifact_identity(report, document, output_path)
    _clear_readback_warnings(document)

    started = time.perf_counter()
    try:
        readback = provider.recognize(
            output_path,
            sample_id=f"{document.sample_id}:deferred-render-readback",
        )
        metadata = verify_render_readback(
            document,
            readback,
            float(config["render_verification"]["minimum_text_coverage_ratio"]),
        )
        attempts = [
            {
                "scale": 1.0,
                "variant": "color",
                "passed": metadata.get("passed") is True,
                "issues": list(metadata.get("issues") or []),
                "coverage_ratio": metadata.get("coverage_ratio"),
                "found_option_labels": list(metadata.get("found_option_labels") or []),
            }
        ]
        retry_scale = float(
            config["render_verification"].get("deferred_option_retry_scale", 1.0)
        )
        if (
            READBACK_RETRY_ISSUES.intersection(metadata.get("issues") or [])
            and retry_scale > 1.0
        ):
            try:
                with Image.open(output_path) as opened:
                    enlarged = opened.convert("RGB").resize(
                        (
                            round(opened.width * retry_scale),
                            round(opened.height * retry_scale),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                with tempfile.TemporaryDirectory(prefix="test7_readback_retry_") as temporary:
                    retry_path = Path(temporary) / "output.enlarged.png"
                    enlarged.save(retry_path, format="PNG")
                    retry_readback = provider.recognize(
                        retry_path,
                        sample_id=(
                            f"{document.sample_id}:deferred-render-readback-"
                            f"{retry_scale:g}x"
                        ),
                    )
                retry_metadata = verify_render_readback(
                    document,
                    retry_readback,
                    float(config["render_verification"]["minimum_text_coverage_ratio"]),
                )
                attempts.append(
                    {
                        "scale": retry_scale,
                        "variant": "color",
                        "passed": retry_metadata.get("passed") is True,
                        "issues": list(retry_metadata.get("issues") or []),
                        "coverage_ratio": retry_metadata.get("coverage_ratio"),
                        "found_option_labels": list(
                            retry_metadata.get("found_option_labels") or []
                        ),
                    }
                )
                if retry_metadata.get("passed") is True:
                    metadata = retry_metadata
            except (RuntimeError, OSError, ValueError) as retry_exc:
                attempts.append(
                    {
                        "scale": retry_scale,
                        "variant": "color",
                        "passed": False,
                        "issues": ["RENDER_READBACK_RETRY_FAILED"],
                        "error": f"{type(retry_exc).__name__}: {retry_exc}",
                    }
                )
        # Formula-heavy rows can cause the detector to miss a small label or
        # most of a compact equation.  A grayscale derivative changes only the
        # OCR input representation, not the saved PNG, and gives an independent
        # pixel readback before we reject an otherwise complete final image.
        if (
            READBACK_RETRY_ISSUES.intersection(metadata.get("issues") or [])
            and bool(
                config["render_verification"].get(
                    "deferred_option_grayscale_retry", True
                )
            )
        ):
            try:
                with Image.open(output_path) as opened:
                    grayscale = opened.convert("L")
                with tempfile.TemporaryDirectory(
                    prefix="test7_readback_grayscale_"
                ) as temporary:
                    grayscale_path = Path(temporary) / "output.grayscale.png"
                    grayscale.save(grayscale_path, format="PNG")
                    grayscale_readback = provider.recognize(
                        grayscale_path,
                        sample_id=(
                            f"{document.sample_id}:deferred-render-readback-grayscale"
                        ),
                    )
                grayscale_metadata = verify_render_readback(
                    document,
                    grayscale_readback,
                    float(config["render_verification"]["minimum_text_coverage_ratio"]),
                )
                attempts.append(
                    {
                        "scale": 1.0,
                        "variant": "grayscale",
                        "passed": grayscale_metadata.get("passed") is True,
                        "issues": list(grayscale_metadata.get("issues") or []),
                        "coverage_ratio": grayscale_metadata.get("coverage_ratio"),
                        "found_option_labels": list(
                            grayscale_metadata.get("found_option_labels") or []
                        ),
                    }
                )
                if grayscale_metadata.get("passed") is True:
                    metadata = grayscale_metadata
            except (RuntimeError, OSError, ValueError) as grayscale_exc:
                attempts.append(
                    {
                        "scale": 1.0,
                        "variant": "grayscale",
                        "passed": False,
                        "issues": ["RENDER_READBACK_GRAYSCALE_RETRY_FAILED"],
                        "error": (
                            f"{type(grayscale_exc).__name__}: {grayscale_exc}"
                        ),
                    }
                )
        metadata["attempts"] = attempts
        metadata["retry_used"] = len(attempts) > 1
        metadata["enabled"] = True
        metadata["mode"] = "deferred_release_gate"
        for code in metadata.get("issues", []):
            document.warnings.append(str(code))
    except (RuntimeError, OSError, ValueError) as exc:
        warning = f"RENDER_READBACK_FAILED:{type(exc).__name__}:{exc}"
        document.warnings.append(warning)
        metadata = {
            "enabled": True,
            "mode": "deferred_release_gate",
            "passed": False,
            "issues": ["RENDER_READBACK_FAILED"],
            "error": f"{type(exc).__name__}: {exc}",
        }
    verification_elapsed = time.perf_counter() - started
    metadata["verified_at"] = datetime.now(timezone.utc).isoformat()
    metadata["verification_elapsed_seconds"] = round(verification_elapsed, 6)

    with Image.open(output_path) as opened:
        output_image = opened.convert("RGB")
    recognition = _recognition_for_revalidation(document, report)
    online_latency_sla_applicable = bool(
        (report.get("measurement") or {}).get("online_latency_sla_applicable", True)
    )
    issues = validate_result(
        document,
        recognition,
        output_image,
        float(report.get("elapsed_seconds") or 0.0) if online_latency_sla_applicable else 0.0,
        config,
    )
    report["status"] = status_from_issues(issues)
    report["issues"] = [
        {"code": issue.code, "severity": issue.severity, "message": issue.message}
        for issue in issues
    ]
    report["render_readback"] = metadata
    report["post_verification"] = {
        "mode": "deferred_release_gate",
        "completed_at": metadata["verified_at"],
        "elapsed_seconds": metadata["verification_elapsed_seconds"],
        "excluded_from_online_latency_sla": True,
    }
    write_json_atomic(document_path, document.to_dict())
    write_json_atomic(report_path, report)
    return report


def verify_run_artifacts(
    run_directory: Path,
    config: dict[str, Any],
    *,
    timeout_seconds: float = 30.0,
    limit: int | None = None,
    force: bool = False,
    provider: RecognitionProvider | None = None,
) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(run_directory)
    report_paths = sorted(run_directory.glob("*/report.json"))
    candidates: list[Path] = []
    skipped: Counter[str] = Counter()
    for report_path in report_paths:
        report = _load_json(report_path)
        codes = {
            str(issue.get("code") or "")
            for issue in report.get("issues", [])
            if isinstance(issue, dict)
        }
        reject_codes = {
            str(issue.get("code") or "")
            for issue in report.get("issues", [])
            if isinstance(issue, dict) and issue.get("severity") == "reject"
        }
        readback_only_reject = bool(reject_codes) and reject_codes.issubset(
            set(READBACK_WARNING_PREFIXES)
        )
        if (
            (report.get("status") == "REJECT" and not readback_only_reject)
            or "HANDWRITING_PRESENT" in codes
        ):
            skipped["hard_reject"] += 1
            continue
        if "PHOTO_PAGE_HANDWRITING_RISK" in codes:
            skipped["photo_page_review"] += 1
            continue
        readback = report.get("render_readback") or {}
        if not force and readback.get("enabled") is True and readback.get("passed") is True:
            skipped["already_passed"] += 1
            continue
        candidates.append(report_path)
    if limit is not None:
        candidates = candidates[: max(0, limit)]

    owned_provider = provider is None
    if provider is None and candidates:
        verification_config = config.get("render_verification", {})
        recycle_after_requests = verification_config.get(
            "deferred_readback_worker_recycle_after_requests"
        )
        provider = TimedRapidOcrProvider(
            timeout_seconds=timeout_seconds,
            params=verification_config.get("rapidocr_params"),
            recycle_after_requests=(
                int(recycle_after_requests)
                if recycle_after_requests is not None
                else None
            ),
            recycle_after_timeout=bool(
                verification_config.get(
                    "deferred_readback_worker_recycle_after_timeout", False
                )
            ),
        )
    statuses: Counter[str] = Counter()
    readback_results: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    try:
        for report_path in candidates:
            try:
                assert provider is not None
                report = verify_sample_artifact(report_path, config, provider)
                statuses[str(report["status"])] += 1
                readback = report.get("render_readback") or {}
                readback_results["passed" if readback.get("passed") is True else "failed"] += 1
            except Exception as exc:
                failures.append(
                    {
                        "sample_id": report_path.parent.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        if owned_provider and provider is not None:
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    normalized_terminal_readbacks = 0
    for report_path in report_paths:
        try:
            normalized_terminal_readbacks += int(
                _normalize_terminal_readback_state(report_path)
            )
        except Exception as exc:
            failures.append(
                {
                    "sample_id": report_path.parent.name,
                    "error": (
                        "terminal_readback_normalization:"
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

    final_reports = [_load_json(path) for path in report_paths]
    all_statuses = Counter(str(report.get("status") or "INVALID") for report in final_reports)
    aggregate_readback_states: Counter[str] = Counter()
    for report in final_reports:
        readback = report.get("render_readback") or {}
        if readback.get("passed") is True:
            aggregate_readback_states["passed"] += 1
        elif readback.get("pending") is True:
            aggregate_readback_states["pending"] += 1
        elif readback.get("enabled") is True:
            aggregate_readback_states["failed"] += 1
        elif readback.get("skipped") is True:
            reason = str(readback.get("reason") or "unspecified").lower()
            aggregate_readback_states[f"skipped_{reason}"] += 1
        else:
            aggregate_readback_states["disabled"] += 1
    index_path = run_directory / "index.finalized.jsonl"
    index_path.write_text(
        "".join(
            json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n"
            for report in final_reports
        ),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "deferred_readback_summary_v1",
        "run_directory": str(run_directory),
        "discovered": len(report_paths),
        "selected": len(candidates),
        "processed": sum(statuses.values()),
        "processed_statuses_after_verification": dict(sorted(statuses.items())),
        "all_statuses_after_verification": dict(sorted(all_statuses.items())),
        "aggregate_readback_states": dict(sorted(aggregate_readback_states.items())),
        "readback_results": dict(sorted(readback_results.items())),
        "skipped": dict(sorted(skipped.items())),
        "normalized_terminal_readback_count": normalized_terminal_readbacks,
        "failures": failures,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "online_latency_sla_affected": False,
        "release_gate_requires_readback_pass": True,
    }
    write_json_atomic(run_directory / "deferred_readback_summary.json", summary)
    return summary
