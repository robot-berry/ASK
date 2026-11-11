"""Loopback-only structured OCR service owned by test7-OCR.

The heavy MinerU models deliberately live outside the Git repository.  Set
``TEST7_MINERU_ROOT`` to the directory containing ``mineru.json`` and the
downloaded model tree before starting this module with that environment's
Python interpreter.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Sequence

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from .formula_crops import formula_mfr_candidates
from .providers import (
    STRUCTURED_FORMULA_SERVICE_ENGINE,
    STRUCTURED_SERVICE_API_VERSION,
    STRUCTURED_SERVICE_ENGINE,
    STRUCTURED_SERVICE_PROJECT,
    StructuredServiceProvider,
)
from .runtime_memory import (
    DEFAULT_VRAM_MINIMUM_FREE_BYTES,
    DEFAULT_VRAM_RESERVED_FRACTION_LIMIT,
    should_release_cached_vram,
)
from .warmup import STRUCTURED_WARMUP_PROFILE_VERSION, synthetic_structured_warmup_pages


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MINERU_ROOT = PROJECT_ROOT.parents[1] / "agentpicture" / "MH" / "test1"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FORMULA_LABELS = frozenset({"inline_formula", "display_formula"})
TEXT_LABELS = frozenset({"ocr_text"})
VISUAL_LABELS = frozenset({"image", "chart", "table"})
WHOLE_IMAGE_FORMULA_CROP = "whole-image"


def _safe_corner_bbox(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (int(round(float(part))) for part in value)
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _likely_handwritten_indexes(items: Sequence[dict[str, Any]]) -> set[int]:
    """Flag only low-confidence math-shaped work; callers still hard-reject it."""
    suppressed: set[int] = set()
    for index, item in enumerate(items):
        label = str(item.get("label") or "")
        if label not in FORMULA_LABELS | TEXT_LABELS or _safe_corner_bbox(item.get("bbox")) is None:
            continue
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        value = str(item.get("latex") if label in FORMULA_LABELS else item.get("text") or "")
        value = re.sub(r"\s+", " ", value).strip()
        cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in value)
        operator_count = len(re.findall(r"[=<>≤≥+\-−×÷*/^√△Δ()]", value))
        if label == "ocr_text" and score < 0.75 and cjk_count <= 1 and operator_count >= 2 and len(value) >= 5:
            suppressed.add(index)
        if label == "display_formula" and score < 0.60 and re.match(r"^\s*=", value):
            suppressed.add(index)
    return suppressed


def _public_items(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in items:
        if str(item.get("label") or "") not in FORMULA_LABELS | TEXT_LABELS:
            continue
        output.append(
            {
                key: item.get(key)
                for key in ("label", "bbox", "score", "text", "latex")
                if key in item
            }
        )
    return output


def _public_visual_regions(
    items: Sequence[dict[str, Any]], image_width: int, image_height: int
) -> list[dict[str, Any]]:
    page_area = max(1, image_width * image_height)
    regions: list[dict[str, Any]] = []
    for item in items:
        label = str(item.get("label") or "").strip().lower()
        bbox = _safe_corner_bbox(item.get("bbox"))
        if label not in VISUAL_LABELS or bbox is None:
            continue
        left, top, right, bottom = bbox
        left = max(0, min(image_width - 1, left))
        top = max(0, min(image_height - 1, top))
        right = max(left + 1, min(image_width, right))
        bottom = max(top + 1, min(image_height, bottom))
        area_ratio = (right - left) * (bottom - top) / page_area
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score < 0.20 or area_ratio < 0.0015:
            continue
        region: dict[str, Any] = {
            "label": label,
            "bbox": [left, top, right, bottom],
            "score": round(score, 6),
            "area_ratio": round(area_ratio, 6),
        }
        if label == "table" and isinstance(item.get("html"), str):
            region["html"] = str(item["html"]).strip()[:100_000]
        regions.append(region)
    regions.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return regions


class StructuredRuntime:
    def __init__(self) -> None:
        mineru_root = Path(os.environ.get("TEST7_MINERU_ROOT", str(DEFAULT_MINERU_ROOT))).resolve()
        config_path = mineru_root / "mineru.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"MinerU config not found: {config_path}")
        os.environ["MINERU_FORMULA_CH_SUPPORT"] = "True"
        os.environ["MINERU_TOOLS_CONFIG_JSON"] = str(config_path)
        os.environ["MINERU_MODEL_SOURCE"] = "local"
        # MinerU's CLI disables the cuDNN v8 API before importing torch.  This
        # standalone service used to bypass that startup policy, causing the
        # first unseen OCR tensor shapes to spend several seconds selecting a
        # convolution plan and occasionally fail with a bad allocation.  Match
        # the supported runtime policy explicitly before any MinerU model import.
        os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from mineru.backend.pipeline import batch_analyze as batch_analyze_module
        from mineru.backend.pipeline.batch_analyze import BatchAnalyze
        from mineru.backend.pipeline.model_init import MineruPipelineModel
        import torch

        torch.backends.cudnn.benchmark = False
        self.runtime_policy = {
            "torch_cudnn_v8_api_disabled": (
                os.environ.get("TORCH_CUDNN_V8_API_DISABLED") == "1"
            ),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }

        cleanup_mode = os.environ.get(
            "TEST7_MINERU_VRAM_CLEANUP", "resident_guarded"
        ).strip().lower()
        if cleanup_mode not in {"resident_guarded", "mineru_default"}:
            raise ValueError(
                "TEST7_MINERU_VRAM_CLEANUP must be resident_guarded or mineru_default"
            )
        self.vram_cleanup_policy: dict[str, Any] = {
            "mode": cleanup_mode,
            "reserved_fraction_limit": DEFAULT_VRAM_RESERVED_FRACTION_LIMIT,
            "minimum_free_bytes": DEFAULT_VRAM_MINIMUM_FREE_BYTES,
            "guard_calls": 0,
            "cleanup_calls": 0,
            "cache_reuse_calls": 0,
            "forced_default_calls": 0,
            "pre_request_cleanup_calls": 0,
            "last_snapshot": None,
        }
        self._active_vram_cleanup_mode = cleanup_mode
        self._original_clean_vram = batch_analyze_module.clean_vram
        if cleanup_mode == "resident_guarded" and str(
            os.environ.get("TEST7_FORMULA_OCR_DEVICE", "cuda")
        ).startswith("cuda"):
            def guarded_clean_vram(device: object, vram_threshold: int = 8) -> None:
                policy = self.vram_cleanup_policy
                policy["guard_calls"] = int(policy["guard_calls"]) + 1
                if self._active_vram_cleanup_mode == "mineru_default":
                    policy["cleanup_calls"] = int(policy["cleanup_calls"]) + 1
                    policy["forced_default_calls"] = int(
                        policy["forced_default_calls"]
                    ) + 1
                    self._original_clean_vram(
                        device, vram_threshold=vram_threshold
                    )
                    return
                try:
                    device_properties = torch.cuda.get_device_properties(device)
                    total_bytes = int(device_properties.total_memory)
                    reserved_bytes = int(torch.cuda.memory_reserved(device))
                    free_bytes, _reported_total = torch.cuda.mem_get_info(device)
                    free_bytes = int(free_bytes)
                    snapshot = {
                        "total_bytes": total_bytes,
                        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                        "reserved_bytes": reserved_bytes,
                        "free_bytes": free_bytes,
                    }
                    policy["last_snapshot"] = snapshot
                    release = should_release_cached_vram(
                        total_bytes=total_bytes,
                        reserved_bytes=reserved_bytes,
                        free_bytes=free_bytes,
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    release = True
                if release:
                    policy["cleanup_calls"] = int(policy["cleanup_calls"]) + 1
                    self._original_clean_vram(
                        device, vram_threshold=vram_threshold
                    )
                else:
                    policy["cache_reuse_calls"] = int(policy["cache_reuse_calls"]) + 1

            batch_analyze_module.clean_vram = guarded_clean_vram
        self.runtime_policy["mineru_vram_cleanup"] = cleanup_mode

        self.device = os.environ.get("TEST7_FORMULA_OCR_DEVICE", "cuda").strip() or "cuda"
        model = MineruPipelineModel(
            formula_config={"enable": True},
            table_config={"enable": False},
            lang="ch",
            device=self.device,
        )

        class ModelManager:
            def get_model(self, **_kwargs: object):
                return model

        self.model = model
        self._shape_trace: dict[str, list[tuple[int, ...]]] | None = None
        self._install_ocr_shape_tracing()
        self.analyzer = BatchAnalyze(
            ModelManager(),
            batch_ratio=1,
            formula_enable=True,
            table_enable=False,
        )
        self.lock = threading.Lock()
        self.mineru_root = mineru_root
        self.warmup = self._warm_models()

    def _install_ocr_shape_tracing(self) -> None:
        """Record tensor shapes only; never retain pixels or recognized content."""
        detector = self.model.ocr_model.text_detector
        detector_preprocess = detector._preprocess_det_image

        def traced_detector_preprocess(image):
            result = detector_preprocess(image)
            if result is not None and self._shape_trace is not None:
                self._shape_trace["detector"].append(tuple(int(v) for v in result[0].shape))
            return result

        detector._preprocess_det_image = traced_detector_preprocess

        recognizer = self.model.ocr_model.text_recognizer
        recognizer_resize = recognizer.resize_norm_img

        def traced_recognizer_resize(image, max_wh_ratio):
            result = recognizer_resize(image, max_wh_ratio)
            if self._shape_trace is not None:
                self._shape_trace["recognizer"].append(tuple(int(v) for v in result.shape))
            return result

        recognizer.resize_norm_img = traced_recognizer_resize

    @staticmethod
    def _summarize_shape_trace(
        trace: dict[str, list[tuple[int, ...]]],
    ) -> dict[str, list[dict[str, Any]]]:
        summary: dict[str, list[dict[str, Any]]] = {}
        for stage, shapes in trace.items():
            counts: dict[tuple[int, ...], int] = {}
            for shape in shapes:
                counts[shape] = counts.get(shape, 0) + 1
            summary[stage] = [
                {"shape": list(shape), "count": count}
                for shape, count in sorted(counts.items())
            ]
        return summary

    def _warm_models(self) -> dict[str, Any]:
        enabled = os.environ.get("TEST7_STRUCTURED_WARMUP", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        if not enabled:
            return {
                "enabled": False,
                "completed": False,
                "profile_version": STRUCTURED_WARMUP_PROFILE_VERSION,
                "pages": 0,
                "elapsed_ms": 0.0,
            }
        pages = synthetic_structured_warmup_pages()
        started = time.perf_counter()
        trace: dict[str, list[tuple[int, ...]]] = {"detector": [], "recognizer": []}
        self._shape_trace = trace
        try:
            self.analyzer([(page, True, "ch") for page in pages])
            # Production calls analyze one image at a time.  A multi-page batch
            # alone does not initialize every single-request CUDA/PyTorch path,
            # so repeat the same synthetic pages individually before readiness.
            for page in pages:
                self.analyzer([(page, True, "ch")])
        except Exception as exc:
            return {
                "enabled": True,
                "completed": False,
                "profile_version": STRUCTURED_WARMUP_PROFILE_VERSION,
                "pages": len(pages),
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_shapes": self._summarize_shape_trace(trace),
            }
        finally:
            self._shape_trace = None
        return {
            "enabled": True,
            "completed": True,
            "profile_version": STRUCTURED_WARMUP_PROFILE_VERSION,
            "pages": len(pages),
            "request_shapes": "batch_then_single_page",
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "source": "synthetic_non_dataset_pages",
            "runtime_shapes": self._summarize_shape_trace(trace),
            "runtime_policy": dict(self.runtime_policy),
        }

    def recognize(
        self,
        image: Image.Image,
        *,
        cleanup_mode: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        request_cleanup_mode = cleanup_mode or str(self.vram_cleanup_policy["mode"])
        with self.lock:
            trace: dict[str, list[tuple[int, ...]]] = {"detector": [], "recognizer": []}
            self._shape_trace = trace
            previous_cleanup_mode = self._active_vram_cleanup_mode
            self._active_vram_cleanup_mode = request_cleanup_mode
            try:
                # A deterministic source-derived request must not inherit CUDA
                # allocator/workspace state left by a previous resident-cache
                # photo request or by warmup.  MinerU's normal cleanup happens
                # only after each stage, so explicitly restore that clean entry
                # state before layout analysis as well.
                if (
                    request_cleanup_mode == "mineru_default"
                    and self.vram_cleanup_policy["mode"] == "resident_guarded"
                ):
                    policy = self.vram_cleanup_policy
                    policy["cleanup_calls"] = int(policy["cleanup_calls"]) + 1
                    policy["forced_default_calls"] = int(
                        policy["forced_default_calls"]
                    ) + 1
                    policy["pre_request_cleanup_calls"] = int(
                        policy["pre_request_cleanup_calls"]
                    ) + 1
                    self._original_clean_vram(self.device)
                layout = self.analyzer([(image.convert("RGB"), True, "ch")])[0]
            finally:
                self._active_vram_cleanup_mode = previous_cleanup_mode
                self._shape_trace = None
            runtime_shapes = self._summarize_shape_trace(trace)
        public_items = _public_items(layout)
        suppressed = _likely_handwritten_indexes(public_items)
        for index in suppressed:
            public_items[index]["filtered_as_handwriting"] = True
        transcript_items = [item for index, item in enumerate(public_items) if index not in suppressed]
        transcript = StructuredServiceProvider._transcript_from_items(transcript_items)
        formulas = [
            re.sub(r"\s+", " ", str(item.get("latex") or "")).strip()
            for item in transcript_items
            if str(item.get("label") or "") in FORMULA_LABELS and str(item.get("latex") or "").strip()
        ]
        regions = _public_visual_regions(layout, image.width, image.height)
        return {
            "project": STRUCTURED_SERVICE_PROJECT,
            "service_api_version": STRUCTURED_SERVICE_API_VERSION,
            "transcript": transcript,
            "formulas": formulas,
            "formula_count": len(formulas),
            "items": public_items,
            "layout_regions": regions,
            "visual_region_count": len(regions),
            "handwriting_filtered_count": len(suppressed),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "engine": STRUCTURED_SERVICE_ENGINE,
            "device": self.device,
            "service_warmup_completed": bool(self.warmup.get("completed")),
            "service_warmup_profile_version": self.warmup.get("profile_version"),
            "runtime_shapes": runtime_shapes,
            "vram_cleanup": dict(self.vram_cleanup_policy),
            "request_vram_cleanup_mode": request_cleanup_mode,
        }

    def recognize_formulas(
        self, image: Image.Image, *, whole_image_formula: bool = False
    ) -> dict[str, Any]:
        """Run resident layout and formula models without the slower text OCR stage."""
        import numpy as np

        started = time.perf_counter()
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb)
        with self.lock:
            layout_started = time.perf_counter()
            layout = self.model.layout_model.predict(
                rgb,
                use_paddlex_filter_boxes=False,
            )
            layout_ms = (time.perf_counter() - layout_started) * 1000.0
            formula_candidates, candidate_source = formula_mfr_candidates(
                layout,
                image.width,
                image.height,
                whole_image_formula=whole_image_formula,
            )
            formula_started = time.perf_counter()
            recognized = self.model.mfr_model.predict(
                formula_candidates,
                pixels,
                batch_size=16,
            )
            formula_ms = (time.perf_counter() - formula_started) * 1000.0

        formula_items: list[dict[str, Any]] = []
        formulas: list[str] = []
        for item in recognized:
            bbox = _safe_corner_bbox(item.get("bbox"))
            latex = re.sub(r"\s+", " ", str(item.get("latex") or "")).strip()
            if bbox is None or not latex:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            formula_items.append(
                {
                    "label": str(item.get("label") or "inline_formula"),
                    "bbox": list(bbox),
                    "score": round(score, 6),
                    "latex": latex,
                }
            )
            formulas.append(latex)

        text_regions: list[dict[str, Any]] = []
        for item in layout:
            if str(item.get("label") or "") != "ocr_text":
                continue
            bbox = _safe_corner_bbox(item.get("bbox"))
            if bbox is None:
                continue
            text_regions.append(
                {
                    "bbox": list(bbox),
                    "score": round(float(item.get("score") or 0.0), 6),
                }
            )

        regions = _public_visual_regions(layout, image.width, image.height)
        return {
            "project": STRUCTURED_SERVICE_PROJECT,
            "service_api_version": STRUCTURED_SERVICE_API_VERSION,
            "engine": STRUCTURED_FORMULA_SERVICE_ENGINE,
            "device": self.device,
            "formulas": formulas,
            "formula_count": len(formulas),
            "formula_items": formula_items,
            "text_regions": text_regions,
            "layout_regions": regions,
            "visual_region_count": len(regions),
            "layout_ms": round(layout_ms, 3),
            "formula_ms": round(formula_ms, 3),
            "formula_candidate_source": candidate_source,
            "whole_image_formula": bool(whole_image_formula),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "service_warmup_completed": bool(self.warmup.get("completed")),
            "service_warmup_profile_version": self.warmup.get("profile_version"),
            "vram_cleanup": dict(self.vram_cleanup_policy),
        }

    def detect_layout(self, image: Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        with self.lock:
            layout = self.model.layout_model.predict(
                image.convert("RGB"), use_paddlex_filter_boxes=False
            )
        regions = _public_visual_regions(layout, image.width, image.height)
        return {
            "project": STRUCTURED_SERVICE_PROJECT,
            "service_api_version": STRUCTURED_SERVICE_API_VERSION,
            "layout_regions": regions,
            "visual_region_count": len(regions),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "engine": "test7-local-pp-doclayoutv2",
            "device": self.device,
        }


runtime: StructuredRuntime | None = None
startup_error = ""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global runtime, startup_error
    try:
        runtime = StructuredRuntime()
        startup_error = ""
    except Exception as exc:
        runtime = None
        startup_error = f"{type(exc).__name__}: {exc}"
        raise
    yield
    runtime = None


app = FastAPI(title="test7-OCR local structured OCR", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if runtime is not None else "starting",
        "ready": runtime is not None,
        "project": STRUCTURED_SERVICE_PROJECT,
        "service_api_version": STRUCTURED_SERVICE_API_VERSION,
        "engine": STRUCTURED_SERVICE_ENGINE,
        "capabilities": ["recognize", "recognize-formulas", "layout"],
        "startup_error": startup_error or None,
        "warmup": runtime.warmup if runtime is not None else None,
        "vram_cleanup": (
            dict(runtime.vram_cleanup_policy) if runtime is not None else None
        ),
    }


def _decode_image(payload: bytes) -> Image.Image:
    if not payload or len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "图片为空或超过20MB")
    try:
        with Image.open(BytesIO(payload)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(400, "无法读取图片") from exc
    if image.width * image.height > 50_000_000:
        raise HTTPException(413, "图片像素过大")
    return image


@app.post("/recognize")
def recognize(
    payload: bytes = Body(..., media_type="image/png"),
    x_test7_vram_cleanup: str | None = Header(default=None),
) -> JSONResponse:
    if runtime is None:
        raise HTTPException(503, "本地结构化 OCR 尚未就绪")
    cleanup_mode = str(x_test7_vram_cleanup or "").strip().lower() or None
    if cleanup_mode not in {None, "resident_guarded", "mineru_default"}:
        raise HTTPException(400, "无效的显存清理策略")
    return JSONResponse(
        runtime.recognize(_decode_image(payload), cleanup_mode=cleanup_mode)
    )


@app.post("/recognize-formulas")
def recognize_formulas(
    payload: bytes = Body(..., media_type="image/png"),
    x_test7_formula_crop: str | None = Header(default=None),
) -> JSONResponse:
    if runtime is None:
        raise HTTPException(503, "本地公式与版面模型尚未就绪")
    mode = str(x_test7_formula_crop or "").strip().lower()
    whole_image_formula = mode in {WHOLE_IMAGE_FORMULA_CROP, "1", "true", "yes"}
    return JSONResponse(
        runtime.recognize_formulas(
            _decode_image(payload),
            whole_image_formula=whole_image_formula,
        )
    )


@app.post("/layout")
def layout(payload: bytes = Body(..., media_type="image/png")) -> JSONResponse:
    if runtime is None:
        raise HTTPException(503, "本地版面检测尚未就绪")
    return JSONResponse(runtime.detect_layout(_decode_image(payload)))
