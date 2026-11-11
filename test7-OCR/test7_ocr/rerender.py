from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .deferred_verification import (
    _assert_artifact_identity,
    _clear_readback_warnings,
    _document_from_dict,
    _recognition_for_revalidation,
)
from .html_render import edge_render_enabled, render_question_katex
from .image_ops import open_source_image, require_direct_crop, sha256_file
from .output_normalization import normalize_document_for_output
from .render import render_question
from .storage import (
    image_pixel_sha256,
    prepare_sample_storage,
    save_png_atomic,
    safe_name,
    write_json_atomic,
    write_jsonl_atomic,
)
from .text import parse_question
from .types import ProcessReport, VisualAsset
from .validators import status_from_issues, validate_result


ProgressCallback = Callable[[int, int, str], None]


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _reusable_readback_evidence(
    source_report: dict[str, Any],
    output_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Reuse deferred OCR evidence only for the exact same saved PNG bytes."""

    readback = source_report.get("render_readback")
    post_verification = source_report.get("post_verification")
    if (
        str(source_report.get("output_sha256") or "") != output_sha256
        or not isinstance(readback, dict)
        or readback.get("enabled") is not True
        or readback.get("mode") != "deferred_release_gate"
        or not isinstance(readback.get("passed"), bool)
        or not isinstance(post_verification, dict)
        or post_verification.get("mode") != "deferred_release_gate"
    ):
        return None
    source_run_id = str(source_report.get("run_id") or "")
    reused_readback = deepcopy(readback)
    reused_readback["evidence_reused_from_identical_output"] = {
        "source_run_id": source_run_id,
        "output_sha256": output_sha256,
    }
    reused_post = deepcopy(post_verification)
    reused_post["evidence_reused_from_identical_output"] = {
        "source_run_id": source_run_id,
        "output_sha256": output_sha256,
    }
    return reused_readback, reused_post


def _rerender_one(
    source_report_path: Path,
    output_root: Path,
    run_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    source_report = _json_object(source_report_path)
    source_document_path = Path(str(source_report["question_document"])).resolve()
    source_output_path = Path(str(source_report["output_image"])).resolve()
    source_document_payload = _json_object(source_document_path)
    document = _document_from_dict(source_document_payload)
    _assert_artifact_identity(source_report, document, source_output_path)
    original_assets = {
        asset.asset_id: (asset.source_bbox, asset.crop_sha256)
        for asset in document.visual_assets
    }
    normalization_metadata = normalize_document_for_output(document, config)

    storage = prepare_sample_storage(output_root, run_id, document.sample_id)
    source_report_sha256 = sha256_file(source_report_path)
    source_document_sha256 = sha256_file(source_document_path)
    source_image = open_source_image(Path(document.source_image))
    asset_images: list[Image.Image] = []
    rebuilt_assets: list[VisualAsset] = []
    for index, asset in enumerate(document.visual_assets, start=1):
        bbox = asset.source_bbox.clamp(source_image.width, source_image.height)
        if bbox != asset.source_bbox:
            raise ValueError(f"Source asset bbox no longer fits the source image: {asset.asset_id}")
        crop = require_direct_crop(source_image, bbox)
        crop_sha256 = image_pixel_sha256(crop)
        original = original_assets.get(asset.asset_id)
        if (
            original is not None
            and bbox == original[0]
            and crop_sha256 != original[1]
        ):
            raise ValueError(f"Direct source crop hash changed: {asset.asset_id}")
        asset_name = f"asset_{index:03d}.png"
        asset_path = storage.asset_directory / asset_name
        save_png_atomic(crop, asset_path)
        asset_images.append(crop)
        rebuilt_assets.append(
            VisualAsset(
                asset_id=asset.asset_id,
                source_bbox=bbox,
                crop_path=f"assets/{asset_name}",
                crop_sha256=crop_sha256,
                kind=asset.kind,
                extraction=asset.extraction,
                bbox_source=asset.bbox_source,
                bbox_verified=asset.bbox_verified,
                preserve_source_pixels=asset.preserve_source_pixels,
                redraw_used=asset.redraw_used,
                fallback=asset.fallback,
                edge_complete=asset.edge_complete,
            )
        )
    document.visual_assets = rebuilt_assets
    _clear_readback_warnings(document)
    structure_reparsed = bool(
        config.get("rerender", {}).get("reparse_structure", False)
    )
    if structure_reparsed:
        structure_warnings: list[str] = []
        (
            document.question_number,
            document.stem,
            document.options,
        ) = parse_question(
            document.display_text,
            recovery_warnings=structure_warnings,
        )
        document.warnings.extend(structure_warnings)

    started = time.perf_counter()
    if edge_render_enabled(document, config):
        try:
            output_image, render_metadata = render_question_katex(
                document,
                storage.directory,
                config,
            )
        except (FileNotFoundError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            document.warnings.append(f"KATEX_RENDER_FALLBACK:{type(exc).__name__}:{exc}")
            output_image, render_metadata = render_question(document, asset_images, config)
            render_metadata["engine"] = "pillow-safe-fallback"
    else:
        output_image, render_metadata = render_question(document, asset_images, config)
        render_metadata["engine"] = "pillow"
    save_png_atomic(output_image, storage.output_path)
    output_sha256 = sha256_file(storage.output_path)
    elapsed = time.perf_counter() - started

    reused_readback = _reusable_readback_evidence(source_report, output_sha256)
    readback_metadata: dict[str, Any]
    reused_post_verification: dict[str, Any] | None = None
    if reused_readback is None:
        readback_metadata = {
            "enabled": False,
            "passed": False,
            "pending": True,
            "reason": "RERENDER_REQUIRES_DEFERRED_READBACK",
        }
    else:
        readback_metadata, reused_post_verification = reused_readback
        for code in readback_metadata.get("issues", []):
            document.warnings.append(str(code))

    recognition = _recognition_for_revalidation(document, source_report)
    issues = validate_result(document, recognition, output_image, 0.0, config)
    report = ProcessReport(
        schema_version="process_report_v1",
        run_id=run_id,
        sample_id=document.sample_id,
        status=status_from_issues(issues),
        source_image=document.source_image,
        output_image=str(storage.output_path.resolve()),
        question_document=str(storage.document_path.resolve()),
        provider=document.provider,
        elapsed_seconds=round(elapsed, 6),
        timings_ms={"render_only": round(elapsed * 1000, 3)},
        issues=issues,
        source_sha256=document.source_sha256,
        output_sha256=output_sha256,
        output_width=output_image.width,
        output_height=output_image.height,
        input_source_image=str(source_report.get("input_source_image") or document.source_image),
        input_source_sha256=str(
            source_report.get("input_source_sha256") or document.source_sha256
        ),
        source_resolution=document.source_resolution,
    )
    write_json_atomic(storage.document_path, document.to_dict())
    payload = report.to_dict()
    payload.update(
        {
            "recognition": {
                "reused_from_source_run": True,
                "source_provider": source_report.get("provider"),
                "model_invoked": False,
            },
            "render": render_metadata,
            "render_readback": readback_metadata,
            "source_reference_only": True,
            "input_copy_saved": False,
            "measurement": {
                "mode": "render_only",
                "online_latency_sla_applicable": False,
                "elapsed_seconds_contains_ocr": False,
            },
            "rerender": {
                "source_run_id": str(source_report["run_id"]),
                "source_report": str(source_report_path.resolve()),
                "source_report_sha256": source_report_sha256,
                "source_document": str(source_document_path),
                "source_document_sha256": source_document_sha256,
                "assets_recropped_from_current_source": len(rebuilt_assets),
                "structure_reparsed_from_display_text": structure_reparsed,
                "readback_reused_from_identical_output": reused_readback is not None,
            },
            "output_normalization": normalization_metadata,
            "formal_gold_scoring": False,
        }
    )
    if reused_post_verification is not None:
        payload["post_verification"] = reused_post_verification
    write_json_atomic(storage.report_path, payload)
    return payload


def _resume_report(
    destination_report_path: Path,
    source_report_path: Path,
) -> dict[str, Any] | None:
    if not destination_report_path.is_file():
        return None
    try:
        payload = _json_object(destination_report_path)
        source_report = _json_object(source_report_path)
        source_document_path = Path(str(source_report["question_document"])).resolve()
        rerender = payload.get("rerender") or {}
        if (
            rerender.get("source_report_sha256") != sha256_file(source_report_path)
            or rerender.get("source_document_sha256") != sha256_file(source_document_path)
        ):
            return None
        document = _document_from_dict(_json_object(Path(str(payload["question_document"]))))
        _assert_artifact_identity(payload, document, Path(str(payload["output_image"])))
        return payload
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def rerender_run(
    source_run: Path,
    output_root: Path,
    run_id: str,
    config: dict[str, Any],
    *,
    limit: int | None = None,
    sample_ids: set[str] | None = None,
    resume: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    source_run = source_run.resolve()
    output_root = output_root.resolve()
    if not source_run.is_dir():
        raise FileNotFoundError(source_run)
    destination = output_root / safe_name(run_id)
    if destination.resolve() == source_run:
        raise ValueError("Rerender destination cannot overwrite the source run")
    if destination.exists() and not resume:
        raise FileExistsError(destination)

    source_reports = sorted(source_run.glob("*/report.json"))
    if sample_ids is not None:
        requested = {str(sample_id).strip() for sample_id in sample_ids if str(sample_id).strip()}
        available = {report.parent.name for report in source_reports}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(
                "Requested sample IDs are not present in the source run: "
                + ", ".join(missing)
            )
        source_reports = [
            report for report in source_reports if report.parent.name in requested
        ]
    if limit is not None:
        source_reports = source_reports[: max(0, limit)]
    reports: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    resumed = 0
    started = time.perf_counter()
    total = len(source_reports)
    for ordinal, source_report_path in enumerate(source_reports, start=1):
        sample_id = source_report_path.parent.name
        destination_report = destination / safe_name(sample_id) / "report.json"
        report = _resume_report(destination_report, source_report_path) if resume else None
        if report is not None:
            resumed += 1
        else:
            try:
                report = _rerender_one(source_report_path, output_root, run_id, config)
            except Exception as exc:
                failures.append(
                    {"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"}
                )
                if progress is not None:
                    progress(ordinal, total, sample_id)
                continue
        reports.append(report)
        if progress is not None:
            progress(ordinal, total, sample_id)

    write_jsonl_atomic(destination / "index.jsonl", reports)
    statuses = Counter(str(report.get("status") or "INVALID") for report in reports)
    elapsed_values = [float(report["elapsed_seconds"]) for report in reports]
    summary = {
        "schema_version": "rerender_batch_summary_v1",
        "run_id": run_id,
        "source_run": str(source_run),
        "requested_sample_ids": sorted(sample_ids) if sample_ids is not None else None,
        "mode": "render_only",
        "attempted": total,
        "completed_sample_count": len(reports),
        "processed_this_invocation": len(reports) - resumed,
        "resumed_from_existing_reports": resumed,
        "statuses": dict(sorted(statuses.items())),
        "failures": failures,
        "latency_seconds": {
            "scope": "render_only_not_end_to_end",
            "mean": round(sum(elapsed_values) / len(elapsed_values), 6) if elapsed_values else None,
            "p50": round(_percentile(elapsed_values, 0.50), 6) if elapsed_values else None,
            "p95": round(_percentile(elapsed_values, 0.95), 6) if elapsed_values else None,
            "max": round(max(elapsed_values), 6) if elapsed_values else None,
            "target": None,
            "under_target_count": 0,
        },
        "resource_guard": {"status": "NOT_APPLICABLE_RENDER_ONLY"},
        "wall_seconds": round(time.perf_counter() - started, 6),
        "formal_gold_scoring": False,
        "default_review_eligible": False,
        "readback_reused_from_identical_output": sum(
            1
            for report in reports
            if bool(
                (report.get("rerender") or {}).get(
                    "readback_reused_from_identical_output"
                )
            )
        ),
        "warning": (
            "Recognition was reused from the source run. This validates rendering behavior only "
            "and cannot be promoted as an OCR accuracy or end-to-end latency result."
        ),
    }
    write_json_atomic(destination / "batch_summary.json", summary)
    return summary
