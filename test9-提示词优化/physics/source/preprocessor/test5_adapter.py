"""Adapter for the OCR reflow preprocessor vendored from picture/test5."""

from __future__ import annotations

import io
import importlib.util
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps


# Math and physics must use the same verified local OCR implementation.  Keep
# one canonical module so fixes to table cells, fractions, reflow and figure
# association cannot silently diverge between the two services.
_SHARED_OCR_PATH = (
    Path(__file__).resolve().parents[3]
    / "math"
    / "source"
    / "test5_preprocess.py"
)
_SHARED_OCR_NAME = "test6_shared_local_ocr_preprocess"
if _SHARED_OCR_NAME in sys.modules:
    test5 = sys.modules[_SHARED_OCR_NAME]
else:
    _spec = importlib.util.spec_from_file_location(_SHARED_OCR_NAME, _SHARED_OCR_PATH)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"无法加载共享本地 OCR：{_SHARED_OCR_PATH}")
    test5 = importlib.util.module_from_spec(_spec)
    sys.modules[_SHARED_OCR_NAME] = test5
    _spec.loader.exec_module(test5)

from .processor import ProcessResult


class Test5QuestionImageProcessor:
    """Expose Test5's local OCR/reflow pipeline through Test6's processor API."""

    def __init__(
        self,
        output_root: Path,
        *,
        enable_reference_library: bool = False,
        write_auxiliary_artifacts: bool = True,
    ) -> None:
        del enable_reference_library
        self.output_root = Path(output_root)
        self.write_auxiliary_artifacts = bool(write_auxiliary_artifacts)
        for name in (
            "png",
            "pdf",
            "markdown",
            "analysis_reference",
            "semantic_reference",
            "illustrations",
            "work",
        ):
            (self.output_root / name).mkdir(parents=True, exist_ok=True)
        # Load the local OCR models once when the service starts. Individual
        # jobs remain fully local and never need a model call for image one.
        self._ocr_engine = test5.create_rapidocr_engine()

    def fast_preview(
        self,
        data: bytes,
        *,
        rotation: str = "auto",
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Build Test5's geometry/light-normalized preview without OCR.

        The complete OCR reflow remains authoritative for provider input.  This
        deliberately stops immediately before ``run_ocr`` so a useful local
        first PNG can be published while the expensive analysis continues.
        """

        if not data:
            raise ValueError("上传图片为空")
        if str(rotation) not in {"auto", "0", "90", "180", "270"}:
            raise ValueError("rotation 必须是 auto、0、90、180 或 270")

        started = time.perf_counter()
        with Image.open(io.BytesIO(data)) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
            image = test5.cv2.cvtColor(
                test5.np.asarray(rgb),
                test5.cv2.COLOR_RGB2BGR,
            )
        decoded_at = time.perf_counter()
        if str(rotation) == "auto":
            rotated, applied_rotation = test5.auto_rotate_fast(image)
        else:
            applied_rotation = int(rotation)
            rotated = test5.rotate_explicitly(image, applied_rotation)
        rotated_at = time.perf_counter()
        config = test5.Config(
            reflow_canvas_width=1200,
            reflow_font_size=48,
            ocr_tiled_fallback_enabled=False,
        )
        quad = test5.find_document_quad(rotated, config)
        detected_at = time.perf_counter()
        rectified, perspective_corrected = test5.perspective_warp(rotated, quad)
        warped_at = time.perf_counter()
        rectified, deskew_angle = test5.deskew(rectified)
        deskewed_at = time.perf_counter()
        enhanced = test5.normalize_illumination(rectified)
        illuminated_at = time.perf_counter()
        enhanced, outer_whitespace_cropped, outer_crop_box = (
            test5.crop_outer_whitespace_for_ocr(enhanced)
        )
        finished_at = time.perf_counter()
        preview = Image.fromarray(
            test5.cv2.cvtColor(enhanced, test5.cv2.COLOR_BGR2RGB)
        )
        milliseconds = lambda left, right: round((right - left) * 1000.0, 3)
        return preview, {
            "generation_method": "test5-pre-ocr-geometry-light",
            "ocr_executed": False,
            "rotation": int(applied_rotation),
            "perspective_corrected": bool(perspective_corrected),
            "deskew_angle": round(float(deskew_angle), 3),
            "width": int(preview.width),
            "height": int(preview.height),
            "decode_ms": milliseconds(started, decoded_at),
            "auto_rotate_ms": milliseconds(decoded_at, rotated_at),
            "document_detection_ms": milliseconds(rotated_at, detected_at),
            "perspective_warp_ms": milliseconds(detected_at, warped_at),
            "deskew_ms": milliseconds(warped_at, deskewed_at),
            "illumination_ms": milliseconds(deskewed_at, illuminated_at),
            "outer_whitespace_crop_ms": milliseconds(illuminated_at, finished_at),
            "outer_whitespace_cropped": bool(outer_whitespace_cropped),
            "outer_crop_box": list(outer_crop_box),
            "elapsed_ms": milliseconds(started, finished_at),
        }

    def process(
        self,
        data: bytes,
        filename: str,
        mode: str = "gray",
        enable_ocr: bool = False,
        handwriting_mode: str = "none",
        rotation: str = "auto",
        quality: str = "fast",
        extract_questions: bool = False,
        deskew_mode: str = "safe",
        geometry_mode: str = "perspective",
        readable_ocr_callback: (
            Callable[[str, list[Path], dict[str, object]], None] | None
        ) = None,
    ) -> ProcessResult:
        del mode, enable_ocr, handwriting_mode, quality, extract_questions
        del deskew_mode, geometry_mode
        if not data:
            raise ValueError("上传图片为空")
        if str(rotation) not in {"auto", "0", "90", "180", "270"}:
            raise ValueError("rotation 必须是 auto、0、90、180 或 270")

        started = time.perf_counter()
        job_id = f"test5_{uuid.uuid4().hex[:16]}"
        work_dir = self.output_root / "work" / job_id
        work_dir.mkdir(parents=True, exist_ok=False)
        suffix = Path(filename).suffix.lower()
        if suffix not in test5.SUPPORTED_SUFFIXES:
            suffix = ".png"
        source = work_dir / f"input{suffix}"
        source.write_bytes(data)
        # Keep Test5's layout logic while matching Test6's 800 px e-ink
        # contract after a 2/3 downscale: 48 px source text becomes 32 px.
        # Keep the existing OCR/layout decisions pixel-compatible. Online
        # latency work must not downscale a public page or remove candidates.
        config = test5.Config(
            processing_max_side=0,
            local_fraction_candidate_limit=40,
            reflow_canvas_width=1200,
            reflow_font_size=48,
            output_quality=1,
            save_debug_stages=False,
        )

        def publish_readable_ocr(
            text: str,
            illustration_images: list[Any],
            audit: dict[str, object],
        ) -> None:
            if readable_ocr_callback is None:
                return
            readable_dir = work_dir / "readable_first"
            readable_paths: list[Path] = []
            for index, image in enumerate(illustration_images, 1):
                path = readable_dir / f"figure_{index:02d}.png"
                test5.cv_write_png(path, image, config.output_quality)
                readable_paths.append(path)
            readable_ocr_callback(text, readable_paths, dict(audit))

        try:
            record = test5.process_one(
                source,
                work_dir,
                config,
                str(rotation),
                self._ocr_engine,
                publish_readable_ocr if readable_ocr_callback is not None else None,
            )
            generated = Path(record.output)
            enhanced = Path(record.enhanced_output)
            semantic_reference = Path(
                getattr(record, "semantic_reference_output", "")
                or record.enhanced_output
            )
            if (
                not generated.is_file()
                or not enhanced.is_file()
                or not semantic_reference.is_file()
            ):
                raise RuntimeError("Test5 未生成重排题图、增强原图或语义校验图")

            png_relative = Path("png") / f"{job_id}.png"
            pdf_relative = Path("pdf") / f"{job_id}.pdf"
            markdown_relative = Path("markdown") / f"{job_id}.md"
            analysis_relative = Path("analysis_reference") / f"{job_id}.png"
            semantic_relative = Path("semantic_reference") / f"{job_id}.png"
            png_path = self.output_root / png_relative
            pdf_path = self.output_root / pdf_relative
            markdown_path = self.output_root / markdown_relative
            analysis_path = self.output_root / analysis_relative
            semantic_path = self.output_root / semantic_relative

            shutil.copy2(generated, png_path)
            shutil.copy2(enhanced, analysis_path)
            shutil.copy2(semantic_reference, semantic_path)
            illustration_relatives: list[str] = []
            work_root = work_dir.resolve()
            for index, raw_path in enumerate(
                record.illustration_outputs or [],
                start=1,
            ):
                illustration_source = Path(raw_path).resolve()
                illustration_source.relative_to(work_root)
                if not illustration_source.is_file():
                    raise RuntimeError("Test5 题图裁片不存在")
                illustration_relative = (
                    Path("illustrations") / f"{job_id}_{index:02d}.png"
                )
                shutil.copy2(
                    illustration_source,
                    self.output_root / illustration_relative,
                )
                illustration_relatives.append(illustration_relative.as_posix())
            layout_mode = str(getattr(record, "layout_mode", "ocr-reflow"))
            if self.write_auxiliary_artifacts:
                with Image.open(generated) as opened:
                    page = opened.convert("RGB")
                    page.save(pdf_path, format="PDF", resolution=150.0)
                safe_name = re.sub(r"[\r\n]+", " ", filename)[:200]
                markdown_path.write_text(
                    "# Test5 本地 OCR 重排题图\n\n"
                    f"- 原文件：{safe_name}\n"
                    f"- 旋转：{record.rotation}°\n"
                    f"- 透视矫正：{record.perspective_corrected}\n"
                    f"- 题目裁切：{record.question_cropped}\n"
                    f"- 插图数量：{record.illustration_count}\n"
                    f"- OCR 字符数：{len(record.ocr_text)}\n"
                    f"- OCR 识别次数：{getattr(record, 'ocr_pass_count', 1)}\n"
                    f"- OCR 采用结果：{getattr(record, 'ocr_selected_pass', 'primary')}\n",
                    encoding="utf-8",
                )

            return ProcessResult(
                job_id=job_id,
                png=png_relative.as_posix(),
                pdf=pdf_relative.as_posix(),
                markdown=markdown_relative.as_posix(),
                width=int(record.width),
                height=int(record.height),
                handwriting_mode=f"local-test5-{layout_mode}",
                rotation=int(record.rotation),
                rotation_confidence=1.0,
                rotation_method="test5-local-ocr",
                quality=f"test5-{layout_mode}",
                elapsed_seconds=round(time.perf_counter() - started, 3),
                ocr_text=str(record.ocr_text or ""),
                question_count=1,
                questions=[],
                deskew_detected_angle=float(record.deskew_angle),
                deskew_applied_angle=float(record.deskew_angle),
                deskew_confidence=1.0,
                deskew_status="test5-local",
                cleanup_method=f"test5-{layout_mode}",
                perspective_status=(
                    "applied" if record.perspective_corrected else "not-needed"
                ),
                perspective_confidence=(
                    1.0 if record.perspective_corrected else 0.0
                ),
                analysis_reference=analysis_relative.as_posix(),
                analysis_reference_audit={
                    "generated": True,
                    "generation_method": "test5-enhanced-original",
                    "authoritative_output_modified": False,
                    "printed_text_count": int(record.printed_text_count),
                    "formula_block_count": int(record.formula_block_count),
                    "illustration_count": int(record.illustration_count),
                    "illustration_outputs": len(illustration_relatives),
                    "ocr_character_count": len(record.ocr_text or ""),
                    "ocr_pass_count": int(getattr(record, "ocr_pass_count", 1)),
                    "ocr_selected_pass": str(
                        getattr(record, "ocr_selected_pass", "primary")
                    ),
                    "ocr_quality_audit": dict(
                        getattr(record, "ocr_quality_audit", None) or {}
                    ),
                    "layout_mode": layout_mode,
                    "question_cropped": bool(record.question_cropped),
                    "included_in_provider_request": False,
                },
                semantic_reference=semantic_relative.as_posix(),
                capture_type=f"test5-local-{layout_mode}",
                capture_confidence=1.0,
                illustration_images=illustration_relatives,
            )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


__all__ = ["Test5QuestionImageProcessor"]
