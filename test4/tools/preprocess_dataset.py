#!/usr/bin/env python3
"""Batch Test4 preprocessing for a folder, with no provider calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any

from PIL import Image


TEST4_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def case_id(path: Path, input_root: Path, digest: str) -> str:
    relative = path.relative_to(input_root).with_suffix("").as_posix()
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", relative).strip("_.") or "image"
    return f"{safe}_{digest[:8]}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_worker(subject: str, input_root: Path, output_root: Path) -> None:
    source_root = TEST4_ROOT / subject / "source"
    sys.path.insert(0, str(source_root))
    from preprocessor.processor import QuestionImageProcessor  # type: ignore

    with tempfile.TemporaryDirectory(prefix=f"test4_{subject}_preprocess_") as temp:
        processor = QuestionImageProcessor(
            Path(temp), enable_reference_library=False
        )
        for source in image_files(input_root):
            digest = sha256(source)
            identifier = case_id(source, input_root, digest)
            case_dir = output_root / "cases" / identifier
            case_dir.mkdir(parents=True, exist_ok=True)
            original_target = case_dir / ("original" + source.suffix.lower())
            if not original_target.exists():
                shutil.copy2(source, original_target)
            report: dict[str, Any] = {
                "case_id": identifier,
                "subject_processor": subject,
                "source_path": str(source),
                "source_sha256": digest,
                "provider_calls": 0,
                "success": False,
            }
            try:
                result = processor.process(
                    source.read_bytes(),
                    source.name,
                    mode="gray",
                    enable_ocr=False,
                    handwriting_mode="none",
                    rotation="auto",
                    quality="fast",
                    extract_questions=False,
                    deskew_mode="safe",
                    geometry_mode="rotate-only",
                )
                processed_source = processor.output_root / result.png
                processed_target = case_dir / f"{subject}_page_rectified.png"
                shutil.copy2(processed_source, processed_target)
                report.update(result.to_dict())
                report.update({
                    "success": True,
                    "preprocessed_sha256": sha256(processed_target),
                    "rectified_page_output": processed_target.name,
                })
            except Exception as exc:  # one bad image must not abort the set
                report["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            (case_dir / f"{subject}_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def run_model_inputs(input_root: Path, output_root: Path) -> None:
    """Write exactly one provider-ready PNG for each source image."""
    source_root = TEST4_ROOT / "math" / "source"
    sys.path.insert(0, str(source_root))
    from preprocessor.processor import QuestionImageProcessor  # type: ignore

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test4_model_input_") as temp:
        processor = QuestionImageProcessor(
            Path(temp), enable_reference_library=False
        )
        for source in image_files(input_root):
            digest = sha256(source)
            identifier = case_id(source, input_root, digest)
            result = processor.process(
                source.read_bytes(),
                source.name,
                mode="gray",
                enable_ocr=False,
                handwriting_mode="none",
                rotation="auto",
                quality="fast",
                extract_questions=False,
                deskew_mode="safe",
                geometry_mode="rotate-only",
            )
            shutil.copy2(
                processor.output_root / result.png,
                output_root / f"{identifier}.png",
            )


def load_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"success": False, "error": {"message": "missing report"}}
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(input_root: Path, output_root: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for source in image_files(input_root):
        digest = sha256(source)
        identifier = case_id(source, input_root, digest)
        case_dir = output_root / "cases" / identifier
        math_report = load_report(case_dir / "math_report.json")
        physics_report = load_report(case_dir / "physics_report.json")
        same_pixels = bool(
            math_report.get("success")
            and physics_report.get("success")
            and math_report.get("preprocessed_sha256")
            == physics_report.get("preprocessed_sha256")
        )
        cases.append({
            "case_id": identifier,
            "source_path": str(source),
            "source_sha256": digest,
            "math_success": bool(math_report.get("success")),
            "physics_success": bool(physics_report.get("success")),
            "same_preprocessed_pixels": same_pixels,
            "math": math_report,
            "physics": physics_report,
        })
    return {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "image_count": len(cases),
        "provider_calls": 0,
        "math_success_count": sum(item["math_success"] for item in cases),
        "physics_success_count": sum(item["physics_success"] for item in cases),
        "identical_output_count": sum(
            item["same_preprocessed_pixels"] for item in cases
        ),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", choices=("math", "physics"))
    args = parser.parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        parser.error(f"input folder does not exist: {input_root}")
    if output_root == input_root or input_root in output_root.parents:
        parser.error("output must not equal or live inside input")
    output_root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        run_worker(args.worker, input_root, output_root)
        return 0
    run_model_inputs(input_root, output_root)
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
