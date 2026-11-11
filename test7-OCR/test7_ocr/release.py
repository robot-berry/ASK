from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from PIL import Image

from .clean_source import validate_source_resolution
from .image_ops import open_source_image, require_direct_crop, sha256_file
from .output_quality import QUALITY_ID, validate_output_quality_review
from .storage import image_pixel_sha256, safe_name, write_json_atomic
from .types import BBox


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class NoPublishableSamplesError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_pass_sample(sample_directory: Path) -> dict[str, Any]:
    report_path = sample_directory / "report.json"
    document_path = sample_directory / "question_document.json"
    output_path = sample_directory / "output.png"
    for required in (report_path, document_path, output_path):
        if not required.is_file():
            raise ValueError(f"Missing required release artifact: {required}")

    report = _read_json(report_path)
    document = _read_json(document_path)
    if report.get("status") != "PASS" or report.get("issues"):
        raise ValueError(f"Non-PASS sample cannot be published: {sample_directory.name}")
    if report.get("source_reference_only") is not True or report.get("input_copy_saved") is not False:
        raise ValueError(f"Source-copy policy is missing: {sample_directory.name}")
    readback = report.get("render_readback") or {}
    if readback.get("enabled") is not True or readback.get("passed") is not True:
        raise ValueError(f"Final PNG readback did not pass: {sample_directory.name}")
    if sha256_file(output_path) != report.get("output_sha256"):
        raise ValueError(f"Output hash mismatch: {sample_directory.name}")
    with Image.open(output_path) as opened:
        width, height = opened.size
    if width != 800 or height <= 0:
        raise ValueError(f"Output dimensions are not releasable: {width}x{height}")
    if document.get("handwriting_detected") is not False:
        raise ValueError(f"Handwriting-marked sample cannot be published: {sample_directory.name}")
    if int(document.get("formula_count") or 0) and document.get("formula_verified") is not True:
        raise ValueError(f"Formula-bearing sample is not independently verified: {sample_directory.name}")

    source_path = Path(str(document.get("source_image") or ""))
    if not source_path.is_file() or sha256_file(source_path) != document.get("source_sha256"):
        raise ValueError(f"Source provenance mismatch: {sample_directory.name}")
    source_image = open_source_image(source_path)
    source_resolution = document.get("source_resolution")
    if source_resolution is not None:
        validate_source_resolution(source_resolution)
        if report.get("source_resolution") != source_resolution:
            raise ValueError(f"Source-resolution evidence mismatch: {sample_directory.name}")
        if report.get("input_source_sha256") != source_resolution["input_sha256"]:
            raise ValueError(f"Input source hash mismatch: {sample_directory.name}")
        if document.get("source_sha256") != source_resolution["effective_sha256"]:
            raise ValueError(f"Effective source hash mismatch: {sample_directory.name}")
    elif (
        report.get("input_source_sha256")
        and report.get("input_source_sha256") != report.get("source_sha256")
    ):
        raise ValueError(f"Unexplained source substitution: {sample_directory.name}")
    source_blocks = document.get("source_blocks")
    if not isinstance(source_blocks, list) or not source_blocks:
        raise ValueError(f"Text/formula source provenance is missing: {sample_directory.name}")
    for block in source_blocks:
        if not isinstance(block, dict):
            raise ValueError(f"Unresolved source block cannot be published: {sample_directory.name}")
        disposition = block.get("disposition")
        if disposition not in {"emitted", "ignored"}:
            raise ValueError(f"Unresolved source block cannot be published: {sample_directory.name}")
        if disposition == "ignored" and not str(block.get("reason") or "").strip():
            raise ValueError(
                f"Ignored source block is missing an explicit reason: {sample_directory.name}"
            )
        bbox = BBox.from_value(block.get("source_bbox"))
        if bbox.width <= 0 or bbox.height <= 0 or bbox.clamp(*source_image.size) != bbox:
            raise ValueError(f"Source block bbox is invalid: {sample_directory.name}")
    assets: list[dict[str, Any]] = []
    for asset in document.get("visual_assets", []):
        if (
            asset.get("extraction") != "direct_source_crop"
            or asset.get("redraw_used") is not False
            or asset.get("preserve_source_pixels") is not True
            or asset.get("bbox_verified") is not True
            or asset.get("edge_complete") is not True
            or asset.get("fallback") is not False
        ):
            raise ValueError(f"Visual asset policy failed: {sample_directory.name}/{asset.get('asset_id')}")
        relative = Path(str(asset.get("crop_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe crop path: {relative}")
        crop_path = sample_directory / relative
        if not crop_path.is_file():
            raise ValueError(f"Missing direct crop: {crop_path}")
        with Image.open(crop_path) as opened:
            crop_image = opened.convert("RGB")
            crop_hash = image_pixel_sha256(crop_image)
        if crop_hash != asset.get("crop_sha256"):
            raise ValueError(f"Crop hash mismatch: {crop_path}")
        expected = require_direct_crop(source_image, BBox.from_value(asset["source_bbox"]))
        if image_pixel_sha256(expected) != crop_hash:
            raise ValueError(f"Crop pixels do not match the source bbox: {crop_path}")
        assets.append(
            {
                "asset_id": asset.get("asset_id"),
                "path": relative.as_posix(),
                "pixel_sha256": crop_hash,
                "source_bbox": asset["source_bbox"],
            }
        )
    return {
        "sample_id": str(report.get("sample_id") or sample_directory.name),
        "source_image": str(source_path.resolve()),
        "source_sha256": document["source_sha256"],
        "input_source_image": report.get("input_source_image") or str(source_path.resolve()),
        "input_source_sha256": report.get("input_source_sha256") or document["source_sha256"],
        "source_resolution": source_resolution,
        "output_sha256": report["output_sha256"],
        "output_width": width,
        "output_height": height,
        "assets": assets,
    }


def _validated_human_output_review(
    output_review_root: Path,
    run_id: str,
    sample_record: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    sample_id = sample_record["sample_id"]
    if not QUALITY_ID.fullmatch(run_id) or not QUALITY_ID.fullmatch(sample_id):
        raise ValueError("Unsafe run or sample id in human output review lookup")
    review_path = (output_review_root / run_id / f"{sample_id}.json").resolve()
    if output_review_root not in review_path.parents:
        raise ValueError("Human output review path escaped its root")
    if not review_path.is_file():
        return None, "pending"
    review = _read_json(review_path)
    validate_output_quality_review(review)
    expected = {
        "run_id": run_id,
        "sample_id": sample_id,
        "source_sha256": sample_record["source_sha256"],
        "output_sha256": sample_record["output_sha256"],
    }
    for field, value in expected.items():
        if review.get(field) != value:
            raise ValueError(f"Human output review {field} mismatch: {sample_id}")
    if review["verdict"] != "pass":
        return None, review["verdict"]
    return {
        "path": review_path,
        "reviewer_id": review["reviewer_id"],
        "updated_at_utc": review.get("updated_at_utc"),
        "review_sha256": sha256_file(review_path),
    }, "pass"


def publish_release(
    run_directory: Path,
    release_root: Path,
    release_id: str | None = None,
    output_review_root: Path | None = None,
) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    if not run_directory.is_dir():
        raise FileNotFoundError(run_directory)
    release_root = release_root.resolve()
    release_root.mkdir(parents=True, exist_ok=True)
    output_review_root = (
        output_review_root
        or PROJECT_ROOT / "data" / "annotations" / "output_quality_reviews"
    ).resolve()
    release_name = safe_name(release_id or run_directory.name)
    destination = release_root / release_name
    if destination.exists():
        raise FileExistsError(f"Release already exists and will not be overwritten: {destination}")

    candidates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    rejected_counts: dict[str, int] = {
        "REVIEW": 0,
        "REJECT": 0,
        "INVALID": 0,
        "HUMAN_REVIEW_PENDING": 0,
        "HUMAN_REVIEW_FAILED": 0,
    }
    for sample_directory in sorted(path for path in run_directory.iterdir() if path.is_dir()):
        report_path = sample_directory / "report.json"
        if not report_path.is_file():
            continue
        report = _read_json(report_path)
        status = str(report.get("status") or "INVALID")
        if status != "PASS":
            rejected_counts[status if status in rejected_counts else "INVALID"] += 1
            continue
        record = _validate_pass_sample(sample_directory)
        report_run_id = str(report.get("run_id") or run_directory.name)
        human_review, human_verdict = _validated_human_output_review(
            output_review_root,
            report_run_id,
            record,
        )
        if human_review is None:
            key = "HUMAN_REVIEW_FAILED" if human_verdict == "fail" else "HUMAN_REVIEW_PENDING"
            rejected_counts[key] += 1
            continue
        candidates.append((sample_directory, record, human_review))
    if not candidates:
        raise NoPublishableSamplesError(
            f"No automatically verified and human-approved PASS samples in {run_directory}; "
            "REVIEW/REJECT or unreviewed artifacts remain diagnostic only"
        )

    stage = Path(tempfile.mkdtemp(prefix=f".{release_name}-", dir=release_root))
    try:
        records: list[dict[str, Any]] = []
        for source_directory, record, human_review in candidates:
            target = stage / safe_name(record["sample_id"])
            target.mkdir(parents=True)
            shutil.copy2(source_directory / "output.png", target / "output.png")
            shutil.copy2(source_directory / "question_document.json", target / "question_document.json")
            shutil.copy2(source_directory / "report.json", target / "report.json")
            shutil.copy2(human_review["path"], target / "human_output_review.json")
            if record["assets"]:
                shutil.copytree(source_directory / "assets", target / "assets")
            records.append(
                {
                    **record,
                    "human_output_review": {
                        "reviewer_id": human_review["reviewer_id"],
                        "updated_at_utc": human_review["updated_at_utc"],
                        "review_sha256": human_review["review_sha256"],
                    },
                }
            )
        manifest = {
            "schema_version": "test7_release_manifest_v1",
            "release_id": release_name,
            "source_run": str(run_directory),
            "published_count": len(records),
            "excluded_status_counts": rejected_counts,
            "policy": {
                "pass_only": True,
                "output_width_px": 800,
                "final_png_readback_required": True,
                "formula_verification_required": True,
                "human_final_png_pass_required": True,
                "handwriting_allowed": False,
                "visual_assets": "verified_direct_source_crop_only_no_redraw",
                "source_images_copied": False,
            },
            "samples": records,
        }
        write_json_atomic(stage / "release_manifest.json", manifest)
        os.replace(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {**manifest, "release_directory": str(destination)}
