#!/usr/bin/env python3
"""Loopback-only structured OCR service for formulas and surrounding text."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Sequence

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO


ROOT = Path(__file__).resolve().parent
MINERU_ROOT = ROOT.parents[1] / "agentpicture" / "MH" / "test1"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
FORMULA_LABELS = frozenset({"inline_formula", "display_formula"})
TEXT_LABELS = frozenset({"ocr_text"})
VISUAL_LABELS = frozenset({"image", "chart", "table"})
OPTION_MARKER = re.compile(r"(?:^|\s)([A-D])[.．、:：]")
ANSWER_CUE = re.compile(
    r"(?:取值范围|范围|结果|答案|中位数|平均数|众数|值)(?:是|为)\s*[:：]?$"
)


@dataclass(frozen=True)
class LayoutToken:
    text: str
    bbox: tuple[int, int, int, int]
    kind: str
    score: float

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def height(self) -> int:
        return max(1, self.bbox[3] - self.bbox[1])


def safe_bbox(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (int(round(float(item))) for item in value)
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def public_layout_regions(
    items: Sequence[dict[str, Any]],
    *,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """Return bounded PP-DocLayout visual regions for downstream reflow.

    MinerU already executes PP-DocLayoutV2 before formula/text recognition.
    Exposing the resulting image, chart and table boxes lets Test6 replace its
    fragile contour-only figure cropper without paying for another model pass.
    Only simple JSON values leave this loopback process; intermediate arrays
    and cropped pixels owned by MinerU are deliberately not serialized.
    """

    width = max(1, int(image_width))
    height = max(1, int(image_height))
    page_area = width * height
    regions: list[dict[str, Any]] = []
    for item in items:
        label = str(item.get("label") or "").strip().lower()
        if label not in VISUAL_LABELS:
            continue
        bbox = safe_bbox(item.get("bbox"))
        if bbox is None:
            continue
        left, top, right, bottom = bbox
        left = max(0, min(width - 1, left))
        top = max(0, min(height - 1, top))
        right = max(left + 1, min(width, right))
        bottom = max(top + 1, min(height, bottom))
        area_ratio = (right - left) * (bottom - top) / page_area
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        # Tiny boxes are layout noise.  Almost-full-page boxes are retained at
        # this service boundary and rejected later with OCR prose overlap, where
        # the caller has stronger evidence about the selected question.
        if score < 0.20 or area_ratio < 0.0015:
            continue
        region: dict[str, Any] = {
            "label": label,
            "bbox": [left, top, right, bottom],
            "score": round(score, 6),
            "area_ratio": round(area_ratio, 6),
        }
        html_value = item.get("html")
        if label == "table" and isinstance(html_value, str) and html_value.strip():
            # Bound the audit payload even if a malformed recognizer emits an
            # unexpectedly large table document.
            region["html"] = html_value.strip()[:100_000]
        regions.append(region)
    regions.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))
    return regions


def likely_handwritten_item_indexes(items: Sequence[dict[str, Any]]) -> set[int]:
    """Locate answer workings using confidence plus printed page structure.

    The filter never looks at which answer is correct.  It removes only
    low-confidence, math-shaped OCR text and work positioned between a printed
    answer cue and an observable multiple-choice row.  A low-confidence
    trailing display equality is likewise characteristic of a handwritten
    result rather than a printed part of the question.
    """

    parsed: list[tuple[int, str, tuple[int, int, int, int], float, str]] = []
    option_labels: set[str] = set()
    option_top: int | None = None
    for index, item in enumerate(items):
        label = str(item.get("label") or "")
        bbox = safe_bbox(item.get("bbox"))
        if bbox is None or label not in FORMULA_LABELS | TEXT_LABELS:
            continue
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        value = str(item.get("latex") if label in FORMULA_LABELS else item.get("text") or "")
        value = re.sub(r"\s+", " ", value).strip()
        parsed.append((index, label, bbox, score, value))
        labels = set(OPTION_MARKER.findall(value))
        if labels:
            option_labels.update(labels)
            option_top = bbox[1] if option_top is None else min(option_top, bbox[1])

    suppressed: set[int] = set()
    for index, label, _bbox, score, value in parsed:
        cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in value)
        operator_count = len(re.findall(r"[=<>≤≥+\-−×÷*/^√△Δ()]", value))
        if (
            label == "ocr_text"
            and score < 0.75
            and cjk_count <= 1
            and operator_count >= 2
            and len(value) >= 5
        ):
            suppressed.add(index)
        if (
            label == "display_formula"
            and score < 0.60
            and re.match(r"^\s*=", value)
        ):
            suppressed.add(index)

    if option_top is not None and len(option_labels) >= 2:
        cues = [
            item
            for item in parsed
            if item[1] == "ocr_text"
            and item[2][1] < option_top
            and ANSWER_CUE.search(item[4])
        ]
        for index, _label, bbox, _score, value in parsed:
            if index in suppressed or bbox[1] >= option_top:
                continue
            cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in value)
            operator_count = len(re.findall(r"[=<>≤≥+\-−×÷*/^√△Δ()]", value))
            if cjk_count > 2 or operator_count < 2 or len(value) < 4:
                continue
            for _cue_index, _cue_label, cue_bbox, _cue_score, _cue_text in cues:
                cue_height = max(1, cue_bbox[3] - cue_bbox[1])
                item_height = max(1, bbox[3] - bbox[1])
                cue_center_y = (cue_bbox[1] + cue_bbox[3]) / 2
                item_center_y = (bbox[1] + bbox[3]) / 2
                if (
                    bbox[0] >= cue_bbox[2] + 10
                    and abs(item_center_y - cue_center_y) <= 0.90 * max(cue_height, item_height)
                ):
                    suppressed.add(index)
                    break
    return suppressed


def layout_tokens(items: Sequence[dict[str, Any]]) -> list[LayoutToken]:
    tokens: list[LayoutToken] = []
    suppressed = likely_handwritten_item_indexes(items)
    for index, item in enumerate(items):
        if index in suppressed:
            continue
        label = str(item.get("label") or "")
        bbox = safe_bbox(item.get("bbox"))
        if bbox is None:
            continue
        try:
            score = float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if label in FORMULA_LABELS:
            latex = re.sub(r"\s+", " ", str(item.get("latex") or "")).strip()
            if score < 0.35 or not latex:
                continue
            text = f"${latex}$"
            kind = "formula"
        elif label in TEXT_LABELS:
            text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
            if score < 0.50 or not text:
                continue
            kind = "text"
        else:
            continue
        tokens.append(LayoutToken(text=text, bbox=bbox, kind=kind, score=score))
    return tokens


def normalize_joined_line(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s+([，。；：！？、,.!?;:）)\]])", r"\1", value)
    value = re.sub(r"([（(\[])\s+", r"\1", value)
    value = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", value)
    return value


def structured_transcript(items: Sequence[dict[str, Any]]) -> tuple[str, list[str]]:
    tokens = layout_tokens(items)
    if not tokens:
        return "", []
    median_height = float(np.median([token.height for token in tokens]))
    lines: list[list[LayoutToken]] = []
    for token in sorted(tokens, key=lambda item: (item.center_y, item.center_x)):
        target: list[LayoutToken] | None = None
        for line in reversed(lines[-5:]):
            line_y = float(np.median([item.center_y for item in line]))
            line_height = float(np.median([item.height for item in line]))
            if abs(token.center_y - line_y) <= 0.62 * max(token.height, line_height, median_height):
                target = line
                break
        if target is None:
            lines.append([token])
        else:
            target.append(token)

    rendered_lines: list[str] = []
    formulas: list[str] = []
    for line in lines:
        ordered = sorted(line, key=lambda item: item.center_x)
        rendered = normalize_joined_line(" ".join(item.text for item in ordered))
        if rendered:
            rendered_lines.append(rendered)
        formulas.extend(item.text[1:-1] for item in ordered if item.kind == "formula")
    return "\n".join(rendered_lines), formulas


class FormulaOcrRuntime:
    def __init__(self) -> None:
        os.environ["MINERU_FORMULA_CH_SUPPORT"] = "True"
        os.environ["MINERU_TOOLS_CONFIG_JSON"] = str(MINERU_ROOT / "mineru.json")
        os.environ["MINERU_MODEL_SOURCE"] = "local"
        from mineru.backend.pipeline.batch_analyze import BatchAnalyze
        from mineru.backend.pipeline.model_init import MineruPipelineModel

        device = os.environ.get("TEST6_FORMULA_OCR_DEVICE", "cuda").strip() or "cuda"
        model = MineruPipelineModel(
            formula_config={"enable": True},
            table_config={"enable": False},
            lang="ch",
            device=device,
        )

        class Manager:
            def get_model(self, **_kwargs):
                return model

        self.analyzer = BatchAnalyze(
            Manager(),
            batch_ratio=1,
            formula_enable=True,
            table_enable=False,
        )
        self.model = model
        self.device = device
        self.lock = threading.Lock()

    def recognize(self, image: Image.Image) -> dict[str, Any]:
        started = time.perf_counter()
        with self.lock:
            layout = self.analyzer([(image.convert("RGB"), True, "ch")])[0]
        suppressed = likely_handwritten_item_indexes(layout)
        transcript, formulas = structured_transcript(layout)
        layout_regions = public_layout_regions(
            layout,
            image_width=image.width,
            image_height=image.height,
        )
        public_items = []
        for index, item in enumerate(layout):
            if index in suppressed:
                continue
            if str(item.get("label") or "") not in FORMULA_LABELS | TEXT_LABELS:
                continue
            public_items.append(
                {
                    key: item.get(key)
                    for key in ("label", "bbox", "score", "text", "latex")
                    if key in item
                }
            )
        return {
            "transcript": transcript,
            "formulas": formulas,
            "formula_count": len(formulas),
            "items": public_items,
            "layout_regions": layout_regions,
            "visual_region_count": len(layout_regions),
            "handwriting_filtered_count": len(suppressed),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "engine": "local-pp-doclayoutv2-plus-pp-formulanet",
            "device": self.device,
        }

    def recognize_formulas(self, image: Image.Image) -> dict[str, Any]:
        """Recognize formula regions without repeating full-page text OCR."""

        started = time.perf_counter()
        rgb = image.convert("RGB")
        pixels = np.asarray(rgb)
        with self.lock:
            layout_started = time.perf_counter()
            layout = self.model.layout_model.predict(
                rgb,
                use_paddlex_filter_boxes=False,
            )
            layout_ms = round((time.perf_counter() - layout_started) * 1000.0, 3)
            formula_items = [
                dict(item)
                for item in layout
                if str(item.get("label") or "") in FORMULA_LABELS
            ]
            formula_started = time.perf_counter()
            recognized = self.model.mfr_model.predict(
                formula_items,
                pixels,
                batch_size=16,
            )
            formula_ms = round((time.perf_counter() - formula_started) * 1000.0, 3)
        public_formulas: list[dict[str, Any]] = []
        formula_values: list[str] = []
        for item in recognized:
            bbox = safe_bbox(item.get("bbox"))
            latex = re.sub(r"\s+", " ", str(item.get("latex") or "")).strip()
            if bbox is None or not latex:
                continue
            try:
                score = float(item.get("score") or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            public_formulas.append(
                {
                    "label": str(item.get("label") or "inline_formula"),
                    "bbox": list(bbox),
                    "score": round(score, 6),
                    "latex": latex,
                }
            )
            formula_values.append(latex)
        layout_regions = public_layout_regions(
            layout,
            image_width=image.width,
            image_height=image.height,
        )
        return {
            "formulas": formula_values,
            "formula_count": len(formula_values),
            "formula_items": public_formulas,
            "layout_regions": layout_regions,
            "visual_region_count": len(layout_regions),
            "layout_ms": layout_ms,
            "formula_ms": formula_ms,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "engine": "local-pp-doclayoutv2-plus-pp-formulanet-formula-only",
            "device": self.device,
        }

    def detect_layout(self, image: Image.Image) -> dict[str, Any]:
        """Run only the resident PP-DocLayoutV2 detector.

        This fast endpoint is used when RapidOCR text is already good but a
        referenced diagram/table needs a reliable crop.  It avoids the OCR and
        formula-recognition work performed by ``recognize``.
        """

        started = time.perf_counter()
        with self.lock:
            layout = self.model.layout_model.predict(
                image.convert("RGB"),
                use_paddlex_filter_boxes=False,
            )
        layout_regions = public_layout_regions(
            layout,
            image_width=image.width,
            image_height=image.height,
        )
        return {
            "layout_regions": layout_regions,
            "visual_region_count": len(layout_regions),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "engine": "local-pp-doclayoutv2",
            "device": self.device,
        }


runtime: FormulaOcrRuntime | None = None
startup_error = ""


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global runtime, startup_error
    try:
        runtime = FormulaOcrRuntime()
        startup_error = ""
    except Exception as exc:
        runtime = None
        startup_error = f"{type(exc).__name__}: {exc}"
        raise
    yield
    runtime = None


app = FastAPI(title="Test6 Local Structured OCR", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if runtime is not None else "starting",
        "ready": runtime is not None,
        "engine": "local-pp-doclayoutv2-plus-pp-formulanet",
        "startup_error": startup_error or None,
    }


@app.post("/recognize")
def recognize(payload: bytes = Body(..., media_type="image/png")) -> JSONResponse:
    if runtime is None:
        raise HTTPException(503, "本地结构化OCR尚未就绪")
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
    return JSONResponse(runtime.recognize(image))


@app.post("/layout")
def layout(payload: bytes = Body(..., media_type="image/png")) -> JSONResponse:
    if runtime is None:
        raise HTTPException(503, "本地版面检测尚未就绪")
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
    return JSONResponse(runtime.detect_layout(image))


@app.post("/formulas")
def formulas(payload: bytes = Body(..., media_type="image/png")) -> JSONResponse:
    if runtime is None:
        raise HTTPException(503, "本地公式OCR尚未就绪")
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
    return JSONResponse(runtime.recognize_formulas(image))
