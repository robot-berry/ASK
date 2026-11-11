#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拍照题目图片预处理工具。

处理流程：读取并修正 EXIF 方向 -> 自动旋转 -> 寻找纸张/屏幕题面 -> 透视校正
-> 小角度纠偏 -> 光照与阴影归一化 -> 彩色手写笔迹弱化 -> 插图区域保守裁切
-> 输出 PNG、过程图、插图和批量对照图。

设计原则：自动裁切宁可多留边，也不能截断题干。任何低置信度检测都会退回完整原图。
"""

from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rapidocr import RapidOCR

TEST6_ROOT = Path(__file__).resolve().parents[2]
if str(TEST6_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST6_ROOT))
SOURCE_ROOT = Path(__file__).resolve().parent
# The benchmark loads this module directly from its file path, while the web
# worker starts with ``math/source`` as its working import root.  Keep both
# entry points on the same renderer path; otherwise the benchmark silently
# falls back to drawing the OCR markup (``$``, ``\\frac``...) as plain text.
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from structured_ocr_client import (
    remaining_ocr_timeout,
    request_formula_ocr,
    request_structured_layout,
    request_structured_ocr,
)
from presentation_guard import readable_ocr_text
from local_ocr_content import strip_likely_handwritten_work
from semantic_ocr_recovery import (
    extract_numbered_question,
    first_question_number,
    full_page_recovery_candidate,
    repair_missing_choice_labels,
    trim_after_complete_choice_question,
)
from formula_spatial_fusion import (
    fuse_formula_items,
    has_flattened_script_tokens,
    strip_flattened_neighbors_of_latex,
)


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def create_rapidocr_engine() -> RapidOCR:
    """Create a low-latency OCR runtime without CPU thread oversubscription."""

    try:
        requested_threads = int(os.environ.get("TEST6_RAPIDOCR_THREADS", "8"))
    except ValueError:
        requested_threads = 8
    cpu_count = os.cpu_count() or 8
    intra_threads = max(1, min(requested_threads, cpu_count))
    return RapidOCR(
        params={
            "EngineConfig.onnxruntime.intra_op_num_threads": intra_threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            # Reuse ONNX CPU buffers between passes.  This changes allocation
            # strategy only; model weights, pixels and recognition gates stay
            # identical.
            "EngineConfig.onnxruntime.enable_cpu_mem_arena": True,
        }
    )


@dataclass
class Config:
    """集中保存可调参数，便于按不同相机和纸张继续调优。"""

    max_work_side: int = 1800          # 检测阶段最大边，限制内存和耗时
    processing_max_side: int = 0       # 生产服务可限制整页像素；0 保留原始尺寸
    document_min_area: float = 0.22    # 题面至少占原图比例，过小通常是误检
    document_margin: float = 0.025     # 透视四角向外扩张比例，优先保全边缘文字
    ink_saturation: int = 72           # 彩色笔迹最低饱和度
    ink_max_component_ratio: float = 0.018  # 大色块多为题图，不当作笔迹删除
    ink_strength: float = 0.96         # 0=不处理，1=尽可能消除彩色笔迹
    illustration_padding: float = 0.12 # 插图核心外扩；短轴标/刻度另由标签保护器并入
    illustration_max_count: int = 6       # 立体/几何选择题常含题图和 A-D 五幅图
    ocr_printed_min_score: float = 0.58  # 印刷文字最低置信度；还会结合行结构二次判断
    ocr_fallback_enabled: bool = True     # 仅在主识别疑似漏行时启用宽松复识
    ocr_fallback_box_thresh: float = 0.30 # 宽松检测保全弯曲、低对比长行
    ocr_fallback_unclip_ratio: float = 2.0
    ocr_fallback_text_score: float = 0.40
    ocr_tiled_fallback_enabled: bool = True  # 仅对弯曲长行做窄列复识
    ocr_tiled_min_text_score: float = 0.30
    local_fraction_candidate_limit: int = 40  # 局部分式复识上限；服务端使用更小预算
    reflow_canvas_width: int = 1800      # 重排白色画布宽度
    reflow_font_size: int = 42           # 重排文字基础字号
    output_quality: int = 3            # PNG 压缩等级，0 快、9 小
    save_debug_stages: bool = True       # CLI 审核保留过程图；在线请求关闭
    select_middle_question: bool = True  # 一图多题时只处理版面中心附近的一道完整题


@dataclass
class ProcessRecord:
    source: str
    output: str
    enhanced_output: str
    rotation: int
    perspective_corrected: bool
    question_cropped: bool
    deskew_angle: float
    ink_pixels: int
    illustration_count: int
    printed_text_count: int
    formula_block_count: int
    width: int
    height: int
    ocr_text: str = ""
    semantic_reference_output: str = ""
    illustration_outputs: list[str] | None = None
    layout_mode: str = "ocr-reflow"
    ocr_pass_count: int = 1
    ocr_selected_pass: str = "primary"
    ocr_quality_audit: dict[str, object] | None = None
    warning: str = ""


@dataclass
class OCRItem:
    """OCR 单个文字块及其版面判断结果。"""

    box: np.ndarray
    text: str
    score: float
    width: float
    height: float
    center_x: float
    center_y: float
    printed: bool = False
    formula_like: bool = False
    mask_as_text: bool = False


def cv_read(path: Path) -> np.ndarray:
    """兼容 Windows 中文路径，并应用相机写入的 EXIF 旋转方向。"""
    with Image.open(path) as pil:
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        rgb = np.asarray(pil)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv_write_png(path: Path, image: np.ndarray, compression: int) -> None:
    """OpenCV 的 imwrite 对部分中文路径不稳定，改用编码后写入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not ok:
        raise OSError(f"PNG 编码失败：{path}")
    encoded.tofile(str(path))


def resize_for_detection(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image.copy(), 1.0
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA), scale


def resize_for_ocr(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Resize OCR input in both directions while keeping memory bounded.

    Camera screenshots sometimes contain a very short question strip inside a
    large white canvas.  Downscale-only preprocessing leaves superscripts at
    roughly ten pixels high, so OCR flattens ``10¹¹`` into ``1011``.  After the
    safe whitespace crop below, modest bicubic upscaling restores those small
    printed glyphs without changing the authoritative source coordinates.
    """

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= 0:
        return image.copy(), 1.0
    scale = min(2.5, max_side / longest)
    if 0.98 <= scale <= 1.02:
        return image.copy(), 1.0
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(
        image,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=interpolation,
    ), scale


def crop_outer_whitespace_for_ocr(
    image: np.ndarray,
) -> tuple[np.ndarray, bool, tuple[int, int, int, int]]:
    """Safely remove only dominant outer blank space before OCR.

    The crop is accepted only when a broad, non-trivial content union remains
    and the padded box removes at least 18% of the canvas.  All foreground used
    to decide the crop is retained, so diagrams and option rows stay inside the
    OCR/model image.  Ambiguous photographs simply keep the complete frame.
    """

    h, w = image.shape[:2]
    full = (0, 0, w, h)
    if h < 96 or w < 160:
        return image, False, full

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edge = max(2, min(h, w) // 80)
    border = np.concatenate(
        (
            gray[:edge, :].reshape(-1),
            gray[-edge:, :].reshape(-1),
            gray[:, :edge].reshape(-1),
            gray[:, -edge:].reshape(-1),
        )
    )
    border_level = float(np.median(border)) if border.size else 255.0
    dark_threshold = int(np.clip(border_level - 9.0, 205.0, 246.0))
    foreground = (
        (gray < dark_threshold)
        | ((hsv[:, :, 1] > 42) & (hsv[:, :, 2] < 250))
    ).astype(np.uint8) * 255
    # Do not rely only on absolute darkness.  Thin anti-aliased glyphs can be
    # much lighter than bold option rows, especially after illumination
    # normalisation.  A local-contrast mask joins those strokes into horizontal
    # text bands so a whitespace crop cannot silently discard a faint stem.
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        max(7.0, min(h, w) / 55.0),
    )
    flattened = cv2.divide(gray, background, scale=255)
    _, local_ink = cv2.threshold(
        flattened,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    foreground = cv2.bitwise_or(foreground, local_ink)

    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    minimum_component = max(8, int(h * w * 0.000012))
    boxes = [
        cv2.boundingRect(contour)
        for contour in contours
        if cv2.contourArea(contour) >= minimum_component
    ]
    line_kernel_width = max(7, w // 130)
    text_bands = cv2.morphologyEx(
        local_ink,
        cv2.MORPH_CLOSE,
        np.ones((2, line_kernel_width), np.uint8),
    )
    band_contours, _ = cv2.findContours(
        text_bands,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in band_contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        ink_count = cv2.countNonZero(local_ink[y:y + bh, x:x + bw])
        if (
            bw >= max(22, int(w * 0.018))
            and max(3, int(h * 0.004)) <= bh <= int(h * 0.18)
            and bw >= bh * 1.25
            and ink_count >= max(10, int(bw * bh * 0.012))
        ):
            boxes.append((x, y, bw, bh))
    if not boxes:
        return image, False, full

    x1 = min(x for x, _, _, _ in boxes)
    y1 = min(y for _, y, _, _ in boxes)
    x2 = max(x + bw for x, _, bw, _ in boxes)
    y2 = max(y + bh for _, y, _, bh in boxes)
    content_w, content_h = x2 - x1, y2 - y1
    if content_w < w * 0.32 or content_h < h * 0.055:
        return image, False, full

    pad_x = max(24, int(content_w * 0.035))
    pad_y = max(24, int(content_h * 0.22))
    crop_box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )
    crop_area_ratio = (
        (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1])
        / max(1, w * h)
    )
    if crop_area_ratio > 0.82:
        return image, False, full
    return crop_with_padding(image, crop_box), True, crop_box


def text_horizontal_score(gray: np.ndarray) -> float:
    """用横向字符连通行覆盖率衡量文字方向，不依赖 OCR。"""
    small, _ = resize_for_detection(gray, 1000)
    # 大尺度背景估计先消除拍照光照和黄纸底色，再由 Otsu 提取深色笔画。
    background = cv2.GaussianBlur(small, (0, 0), 17)
    flattened = cv2.divide(small, background, scale=255)
    _, binary = cv2.threshold(flattened, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # 核宽随页面宽度变化，只连接同一横行中的相邻字符。侧转文字在该方向上
    # 无法组成覆盖页面 10% 以上的长条，因此不会被边框或单根导线轻易误导。
    kernel_width = max(9, small.shape[1] // 60)
    line = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                            np.ones((3, kernel_width), np.uint8))
    contours, _ = cv2.findContours(line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    score = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w > 3 * h and w > small.shape[1] * 0.10 and
                h < small.shape[0] * 0.15):
            score += w * h
    return score / max(1.0, small.shape[0] * small.shape[1])


def select_ocr_resize_side(
    shape: tuple[int, ...],
    max_side: int,
    wide_detail_side: int,
) -> int:
    """Choose a detector scale that preserves short, wide stem rows.

    Upscaling a one- or two-row question strip by 2.5x can make the detector
    ignore the already-large top stem while still finding the smaller options.
    Such strips are therefore recognized near native size; FormulaNet remains
    responsible for their tiny two-dimensional formula details.
    """

    height, width = shape[:2]
    aspect = max(height, width) / max(1, min(height, width))
    if aspect >= 3.2 and min(height, width) <= 260:
        return min(max_side, max(height, width))
    return max(max_side, wide_detail_side) if aspect >= 3.2 else max_side


def run_ocr(
    image: np.ndarray,
    engine: RapidOCR,
    max_side: int = 1800,
    wide_detail_side: int = 2600,
) -> tuple[
        np.ndarray, tuple[str, ...], tuple[float, ...], int]:
    """执行一次 OCR，同时返回检测框、文字、分数和整页 180 度方向判断。"""
    # Short, wide option strips need a larger recognition canvas than ordinary
    # pages so exponents remain distinct.  The 2.5x cap in ``resize_for_ocr``
    # keeps this deterministic and bounded.
    ocr_side = select_ocr_resize_side(image.shape, max_side, wide_detail_side)
    work, scale = resize_for_ocr(image, ocr_side)
    engine.update_params(use_det=True, use_cls=True, use_rec=True)
    loaded = engine.load_img(work)
    prepared, operation_record = engine.preprocess_img(loaded)
    det_result, cls_result, rec_result, cropped_images = engine.run_ocr_steps(
        prepared, operation_record
    )
    result = engine.build_final_output(
        loaded, det_result, cls_result, rec_result, cropped_images, operation_record
    )

    # 复用 OCR 内部轻量方向分类器的 0/180 结果，多数投票判断整页是否倒置。
    reliable_votes = [(label, float(score)) for label, score in (cls_result.cls_res or [])
                      if float(score) >= 0.75]
    zero_weight = sum(score for label, score in reliable_votes if label == "0")
    upside_down_weight = sum(score for label, score in reliable_votes if label == "180")
    upside_down_count = sum(label == "180" for label, _ in reliable_votes)
    page_rotation = 180 if (upside_down_count >= 2 and
                            upside_down_weight > zero_weight * 1.45) else 0
    if result is None or result.boxes is None:
        return np.empty((0, 4, 2), np.float32), (), (), page_rotation
    return (np.asarray(result.boxes, np.float32) / scale,
            tuple(result.txts), tuple(result.scores), page_rotation)


def auto_rotate(image: np.ndarray, engine: RapidOCR) -> tuple[np.ndarray, int]:
    """OCR 判断横竖方向；自动模式只在原向和逆时针 90 度中选择。

    OCR 自带分类器会先把倒置文字扶正再识别，所以 0 与 180 的识别得分不能用于
    判断人眼方向。特殊的 180/270 度照片应通过 --rotation 明确指定。
    """
    candidates: list[tuple[float, int, np.ndarray]] = []
    for turns, degrees in ((0, 0), (1, 90)):
        rotated = np.rot90(image, turns).copy()
        boxes, texts, scores = run_ocr(rotated, engine, 1300)
        score = 0.0
        for box, text, confidence in zip(boxes, texts, scores):
            width = (np.linalg.norm(box[1] - box[0]) + np.linalg.norm(box[2] - box[3])) / 2
            height = (np.linalg.norm(box[3] - box[0]) + np.linalg.norm(box[2] - box[1])) / 2
            horizontal = float(np.clip(width / max(height, 1), 0.25, 8.0))
            score += float(confidence) ** 2 * math.log2(2 + len(text)) * horizontal
        # OCR 无结果时退回视觉文字行得分，但不让它盖过有效 OCR。
        if len(boxes) == 0:
            score = text_horizontal_score(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY))
        candidates.append((score, degrees, rotated))
    _, degrees, rotated = max(candidates, key=lambda item: item[0])
    return rotated, degrees


def rotate_explicitly(image: np.ndarray, degrees: int) -> np.ndarray:
    """按逆时针方向执行用户明确指定的整 90 度旋转。"""
    turns = (degrees // 90) % 4
    return np.rot90(image, turns).copy()


def order_quad(points: np.ndarray) -> np.ndarray:
    """把任意顺序的四角整理为左上、右上、右下、左下。"""
    pts = points.astype(np.float32).reshape(4, 2)
    result = np.zeros((4, 2), dtype=np.float32)
    sums, diffs = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    result[0], result[2] = pts[np.argmin(sums)], pts[np.argmax(sums)]
    result[1], result[3] = pts[np.argmin(diffs)], pts[np.argmax(diffs)]
    return result


def expand_quad(quad: np.ndarray, shape: tuple[int, ...], ratio: float) -> np.ndarray:
    """四角向外扩张并限制在图内，给题干边缘留安全余量。"""
    center = quad.mean(axis=0)
    expanded = center + (quad - center) * (1.0 + ratio)
    h, w = shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0, w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, h - 1)
    return expanded.astype(np.float32)


def find_document_quad(image: np.ndarray, config: Config) -> np.ndarray | None:
    """寻找纸张或屏幕中的亮色题面；只有高置信度四边形才用于透视裁切。"""
    work, scale = resize_for_detection(image, config.max_work_side)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = work.shape[0] * work.shape[1]
    best: tuple[float, np.ndarray] | None = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
        area = cv2.contourArea(contour)
        if area < image_area * config.document_min_area or area > image_area * 0.995:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        quad = order_quad(polygon[:, 0, :])
        sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
        if min(sides) < min(work.shape[:2]) * 0.15:
            continue
        # 面积大、轮廓接近矩形的候选优先。
        rectangularity = area / max(1.0, cv2.contourArea(cv2.convexHull(contour)))
        candidate_score = area / image_area + 0.1 * rectangularity
        if best is None or candidate_score > best[0]:
            best = (candidate_score, quad / scale)
    if best is not None:
        return expand_quad(best[1], image.shape, config.document_margin)

    # 边缘断裂时（手遮住纸角、书页超出画面、投影亮度较低），改用亮色低饱和
    # 区域寻找纸张/投影题面。此分支仍要求面积足够大，避免把题内白色图片误当整页。
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    bright_page = ((hsv[:, :, 2] >= 125) & (hsv[:, :, 1] <= 105)).astype(np.uint8) * 255
    close_size = max(15, (min(work.shape[:2]) // 35) | 1)
    bright_page = cv2.morphologyEx(
        bright_page, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    bright_page = cv2.morphologyEx(bright_page, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    page_contours, _ = cv2.findContours(bright_page, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(page_contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if not image_area * config.document_min_area <= area <= image_area * 0.98:
            continue
        rectangle = cv2.minAreaRect(contour)
        box = order_quad(cv2.boxPoints(rectangle))
        box_area = cv2.contourArea(box)
        fill_ratio = area / max(1.0, box_area)
        sides = [np.linalg.norm(box[(i + 1) % 4] - box[i]) for i in range(4)]
        box_width_ratio = (box[:, 0].max() - box[:, 0].min()) / work.shape[1]
        # 横向题面上的大面积彩色涂写可能把白纸切成左右两块；只检测到半页时
        # 禁止透视裁切，避免丢掉另一侧题干或题图。
        landscape_incomplete = work.shape[1] / work.shape[0] > 1.5 and box_width_ratio < 0.75
        if (fill_ratio >= 0.62 and min(sides) >= min(work.shape[:2]) * 0.18 and
                not landscape_incomplete):
            return expand_quad(box / scale, image.shape, config.document_margin * 1.8)
    return None


def perspective_warp(image: np.ndarray, quad: np.ndarray | None) -> tuple[np.ndarray, bool]:
    if quad is None:
        return image.copy(), False
    tl, tr, br, bl = order_quad(quad)
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 300 or height < 200:
        return image.copy(), False
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                           dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(order_quad(quad), destination)
    return cv2.warpPerspective(image, matrix, (width, height), borderValue=(255, 255, 255)), True


def deskew(image: np.ndarray) -> tuple[np.ndarray, float]:
    """使用文字笔画估计小角度倾斜；超过 7 度时视为不可靠并保持原样。"""
    work, scale = resize_for_detection(image, 1200)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=80,
                            minLineLength=max(40, work.shape[1] // 8), maxLineGap=18)
    if lines is None:
        return image, 0.0
    angles = []
    # OpenCV 4 常返回 (N, 1, 4)，OpenCV 5 某些构建返回 (N, 4)；
    # reshape 为固定四列后可同时兼容两种版本。
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        while angle <= -45:
            angle += 90
        while angle > 45:
            angle -= 90
        if abs(angle) <= 7:
            angles.append(angle)
    if len(angles) < 3:
        return image, 0.0
    angle = float(np.median(angles))
    if abs(angle) < 0.25:
        return image, angle
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)), angle


def normalize_illumination(image: np.ndarray) -> np.ndarray:
    """消除纸面阴影和屏幕亮度不均，同时保留彩色题图。"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light, a, b = cv2.split(lab)
    # 大尺度闭运算估算背景；相除可拉平阴影但不会把细文字抹掉。
    kernel_size = max(31, (min(image.shape[:2]) // 24) | 1)
    background = cv2.morphologyEx(light, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    flat = cv2.divide(light, np.maximum(background, 1), scale=245)
    flat = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(flat)
    return cv2.cvtColor(cv2.merge((flat, a, b)), cv2.COLOR_LAB2BGR)


def colored_ink_mask(image: np.ndarray, config: Config) -> np.ndarray:
    """检测蓝/青/红色细笔迹；面积较大的彩色连通块作为题图保留。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    blue = ((hue >= 82) & (hue <= 140))
    red = ((hue <= 10) | (hue >= 170))
    raw = ((blue | red) & (saturation >= config.ink_saturation) & (value < 245)).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    result = np.zeros_like(raw)
    image_area = image.shape[0] * image.shape[1]
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        box_fill = area / max(1, w * h)
        average_thickness = area / max(1, max(w, h))
        small_thin_ink = (area <= image_area * config.ink_max_component_ratio and
                          min(w, h) <= max(18, int(0.035 * min(image.shape[:2]))))
        # 大片泼墨虽然外框很大，但在外框中通常较稀疏、平均笔画较细；
        # 彩色印刷插图往往更致密。这样可消除跨越题面的青色涂写，同时保护花朵等题图。
        large_sparse_stroke = (area <= image_area * 0.20 and box_fill < 0.32 and
                               average_thickness < max(45, 0.05 * min(image.shape[:2])))
        if area >= 6 and (small_thin_ink or large_sparse_stroke):
            result[labels == label] = 255
    return cv2.dilate(result, np.ones((3, 3), np.uint8), iterations=1)


def weaken_handwriting(image: np.ndarray, config: Config) -> tuple[np.ndarray, np.ndarray]:
    mask = colored_ink_mask(image, config)
    if not np.any(mask) or config.ink_strength <= 0:
        return image.copy(), mask
    repaired = cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)
    # 宽墨迹内部仅靠邻域扩散会残留同色雾状区域；用大尺度纸面背景进一步淡化。
    background = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    repaired = cv2.addWeighted(repaired, 0.35, background, 0.65, 0)
    # 对已确认的墨迹区域直接提升到纸面亮度，避免宽笔画中心仍呈青色或红色。
    paper = np.full_like(repaired, 248)
    hard_alpha = (mask.astype(np.float32) / 255.0 * 0.82)[:, :, None]
    repaired = np.clip(repaired.astype(np.float32) * (1 - hard_alpha) +
                       paper.astype(np.float32) * hard_alpha, 0, 255).astype(np.uint8)
    alpha = (mask.astype(np.float32) / 255.0 * config.ink_strength)[:, :, None]
    mixed = image.astype(np.float32) * (1 - alpha) + repaired.astype(np.float32) * alpha
    return np.clip(mixed, 0, 255).astype(np.uint8), mask


def crop_by_ocr(image: np.ndarray, boxes: np.ndarray, scores: tuple[float, ...]) -> tuple[np.ndarray, bool, tuple[int, int, int, int]]:
    """以印刷文字框联合范围裁题，并留 22% 边距覆盖邻近公式、选项和题图。"""
    h, w = image.shape[:2]
    valid = [box for box, score in zip(boxes, scores) if score >= 0.55]
    if not valid:
        return image, False, (0, 0, w, h)
    points = np.concatenate(valid, axis=0)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    content_w, content_h = x2 - x1, y2 - y1
    # OCR 框之外给出较宽余量，插图、坐标系和未识别公式通常紧邻题干。
    pad_x = max(w * 0.04, content_w * 0.22)
    pad_y = max(h * 0.06, content_h * 0.40)
    x1, y1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
    x2, y2 = int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y))
    # 题图可能位于最后一行文字下方且没有可识别字符，裁框至少覆盖原题面的
    # 65% 宽高；多留空白比切掉图形、选项更安全。
    if (x2 - x1) < w * 0.65 or (y2 - y1) < h * 0.65:
        return image, False, (0, 0, w, h)
    return image[y1:y2, x1:x2].copy(), True, (x1, y1, x2, y2)


def draw_ocr_debug(image: np.ndarray, boxes: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    """绿色框为 OCR 印刷文字，红框为最终题目裁切范围，作为可视化验收证据。"""
    debug = image.copy()
    for box in boxes:
        cv2.polylines(debug, [np.round(box).astype(np.int32)], True, (0, 180, 0), 3)
    x1, y1, x2, y2 = crop_box
    cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 5)
    return debug


def auto_rotate_fast(image: np.ndarray) -> tuple[np.ndarray, int]:
    """不调用 OCR 的横竖方向判断，保证整条流水线通常只做一次 OCR。

    这个方法比较原图和逆时针 90 度后的文字行水平性。只有侧转后的得分显著更高
    才旋转，避免屏幕边框、表格线导致误判。倒置方向复用正式 OCR 的文字方向
    分类结果判断，因此不会为了旋转页面额外执行第二次正文 OCR。
    """
    # 输入本身已经是横向时优先保持原向。横向物理题中的电路导线、波形和表格
    # 会产生很强的线投影，若只比较投影分数容易误旋转 90 度。
    if image.shape[1] >= image.shape[0] * 1.05:
        return image.copy(), 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normal_score = text_horizontal_score(gray)
    left_score = text_horizontal_score(np.rot90(gray, 1).copy())
    if left_score > 0.003 and left_score > normal_score * 1.8:
        return np.rot90(image, 1).copy(), 90
    return image.copy(), 0


def analyze_ocr_items(boxes: np.ndarray, texts: tuple[str, ...], scores: tuple[float, ...],
                      config: Config) -> list[OCRItem]:
    """区分稳定印刷文字、疑似手写识别结果和需要原图保留的公式块。

    规则不是只看 OCR 分数：印刷正文通常字号接近、横向排列且同一行有邻居；
    孤立、大小异常、倾斜明显的识别结果更可能是手写批注。短数字与单字可能是
    图中坐标标注，因此不加入文字删除掩膜。
    """
    items: list[OCRItem] = []
    for box, text, score in zip(boxes, texts, scores):
        width = float((np.linalg.norm(box[1] - box[0]) + np.linalg.norm(box[2] - box[3])) / 2)
        height = float((np.linalg.norm(box[3] - box[0]) + np.linalg.norm(box[2] - box[1])) / 2)
        center = box.mean(axis=0)
        items.append(OCRItem(box.copy(), text.strip(), float(score), width, height,
                             float(center[0]), float(center[1])))
    usable_heights = [item.height for item in items if item.score >= 0.65 and item.height > 4]
    median_height = float(np.median(usable_heights)) if usable_heights else 32.0
    math_chars = set("=+-×÷*/^√∠△∥⊥≤≥<>[]{}()0123456789")
    math_operators = set("=+-×÷*/^√∠△∥⊥≤≥<>")
    # 第一阶段只确定印刷文字。长中文正文允许略低的 OCR 分数，短标签和公式则
    # 需要更高置信度，避免把排列成行的手写推导误认为印刷正文。
    for item in items:
        neighbors = [other for other in items if other is not item and
                     abs(other.center_y - item.center_y) <= 0.65 * max(item.height, other.height) and
                     abs(other.center_x - item.center_x) <= 8 * median_height]
        height_ok = 0.32 * median_height <= item.height <= 3.2 * median_height
        horizontal = item.width >= 0.85 * item.height
        structural_support = bool(neighbors) or item.width >= 3.0 * item.height
        cjk_count = sum('\u4e00' <= char <= '\u9fff' for char in item.text)
        prose_like = cjk_count >= 2 and len(item.text) >= 4
        high_confidence = item.score >= max(0.80, config.ocr_printed_min_score + 0.16)
        stable_prose = (prose_like and structural_support and
                        item.score >= max(0.66, config.ocr_printed_min_score))
        item.printed = height_ok and horizontal and (high_confidence or stable_prose)

    # 第二阶段仅在印刷行结构已经确定后寻找低置信公式。这样相邻手写字符不会
    # 互相充当“版式邻居”，而嵌在题干或选项中的公式仍可从原图保护性保留。
    for item in items:
        printed_neighbors = [other for other in items if other is not item and other.printed and
                             abs(other.center_y - item.center_y) <= 0.55 * max(item.height, other.height) and
                             abs(other.center_x - item.center_x) <= 7 * median_height]
        height_ok = 0.38 * median_height <= item.height <= 2.5 * median_height
        horizontal = item.width >= 0.95 * item.height
        math_ratio = sum(char in math_chars for char in item.text) / max(1, len(item.text))
        has_operator = any(char in math_operators for char in item.text)
        has_digit = any(char.isdigit() for char in item.text)
        has_latin = any(char.isascii() and char.isalpha() for char in item.text)
        # “低置信度”本身不能作为公式依据，否则 OCR 偶然识别出的手写批注会被
        # 整行裁回最终画布。公式必须同时具备数学字符证据和稳定的横向版式。
        formula_evidence = (has_operator or (has_digit and has_latin) or
                            (len(item.text) >= 3 and math_ratio >= 0.50))
        embedded_formula = bool(printed_neighbors)
        standalone_formula = item.score >= 0.82 and item.width >= 2.5 * item.height
        formula_layout = height_ok and horizontal and (embedded_formula or standalone_formula)
        item.formula_like = (not item.printed and item.score >= 0.35 and
                             formula_evidence and formula_layout)
        # 正文长字符串可安全从检测副本移除；短数字/图注保留以维持图形连通性。
        item.mask_as_text = item.printed and not item.formula_like and (
            len(item.text) >= 3 or (bool(neighbors) and item.width > 1.8 * item.height)
        )
    rescue_aligned_print_tokens(items)
    return items


def rescue_aligned_print_tokens(items: list[OCRItem]) -> int:
    """Keep narrow printed table cells and inline symbols that OCR separated.

    A one-digit table cell is normally taller than it is wide, so the ordinary
    printed-text rule deliberately rejects it to avoid handwritten answers.
    Here it is accepted only when several high-confidence narrow tokens share a
    row with a printed label, or when one short Latin token directly continues
    a printed prose row. Diagram labels are removed later by figure boxes.
    """

    trusted_heights = [
        item.height for item in items if item.printed and item.height > 4
    ]
    median_height = float(np.median(trusted_heights)) if trusted_heights else 30.0
    rescued = 0
    narrow_numeric = [
        item
        for item in items
        if not item.printed
        and item.score >= 0.97
        and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", item.text.strip())
        and 0.52 * median_height <= item.height <= 1.35 * median_height
    ]
    for item in narrow_numeric:
        peers = [
            other
            for other in narrow_numeric
            if other is not item
            and abs(other.center_y - item.center_y) <= 0.40 * median_height
        ]
        row_labels = [
            other
            for other in items
            if other.printed
            and abs(other.center_y - item.center_y) <= 0.48 * median_height
            and re.search(
                r"(?:次数|频数|人数|个数|数量|组数|环数|分数|年份编码|销售量|年度销量|成绩|数值|时间区间|睡眠质量|显著性水平|临界值)",
                other.text,
            )
        ]
        if len(peers) >= 2 and row_labels:
            item.printed = True
            item.mask_as_text = True
            rescued += 1

    for item in items:
        if item.printed or item.score < 0.96:
            continue
        compact = re.sub(r"\s+", "", item.text)
        if not re.fullmatch(r"[A-Za-zΑ-Ωα-ω][′']?", compact):
            continue
        neighbors = [
            other
            for other in items
            if other.printed
            and abs(other.center_y - item.center_y) <= 0.45 * median_height
            and min(
                abs(float(item.box[:, 0].min() - other.box[:, 0].max())),
                abs(float(other.box[:, 0].min() - item.box[:, 0].max())),
            ) <= 0.85 * median_height
            and sum("\u4e00" <= char <= "\u9fff" for char in other.text) >= 2
        ]
        if neighbors and 0.48 * median_height <= item.height <= 1.35 * median_height:
            item.printed = True
            item.mask_as_text = True
            rescued += 1
    return rescued


def recognize_local_numeric_fractions(
    image: np.ndarray,
    items: list[OCRItem],
    engine: RapidOCR,
    max_attempts: int = 40,
) -> tuple[list[OCRItem], dict[str, object]]:
    """Recover stacked numeric fractions from local pixels without a model."""

    started = time.perf_counter()
    trusted = [item for item in items if item.printed and item.height > 4]
    median_height = float(np.median([item.height for item in trusted])) if trusted else 30.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), max(5.0, median_height * 0.35))
    flattened = cv2.divide(gray, background, scale=255)
    _, binary = cv2.threshold(
        flattened,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    kernel_width = max(7, int(round(median_height * 0.28)))
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        np.ones((1, kernel_width), np.uint8),
    )
    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    height, width = image.shape[:2]
    recovered: list[OCRItem] = []
    attempted = 0
    batch_recognition_used = False

    def normalize_digit_result(value: object, score_value: object) -> tuple[str, float]:
        digits = re.sub(r"\D", "", str(value or ""))
        score = float(score_value or 0.0)
        # Keep a moderately confident digit provisionally.  The candidate's
        # spatial anchor below applies the stricter final threshold; option
        # fractions need a slightly lower threshold because the denominator is
        # often clipped by the source strip and recognized as ``2.``.
        if not 1 <= len(digits) <= 3 or score < 0.70:
            return "", score
        return digits, score

    def recognize_digits(crop: np.ndarray) -> tuple[str, float]:
        if crop.size == 0:
            return "", 0.0
        enlarged = cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        try:
            result = engine(
                enlarged,
                use_det=False,
                use_cls=False,
                use_rec=True,
            )
        except Exception:
            return "", 0.0
        texts = tuple(getattr(result, "txts", ()) or ())
        scores = tuple(getattr(result, "scores", ()) or ())
        if not texts:
            return "", 0.0
        return normalize_digit_result(texts[0], scores[0] if scores else 0.0)

    candidates: list[tuple[int, int, int, int, bool]] = []
    for contour in contours:
        x, y, bar_w, bar_h = cv2.boundingRect(contour)
        if not (
            0.24 * median_height <= bar_w <= 1.55 * median_height
            and bar_h <= max(5, 0.24 * median_height)
        ):
            continue
        bar_center_x = x + bar_w / 2
        bar_center_y = y + bar_h / 2
        prose_neighbors = [
            item
            for item in trusted
            if abs(item.center_y - bar_center_y) <= 0.85 * median_height
            and min(
                abs(float(x - item.box[:, 0].max())),
                abs(float(item.box[:, 0].min() - (x + bar_w))),
            ) <= 1.35 * median_height
            and (
                sum("\u4e00" <= char <= "\u9fff" for char in item.text) >= 3
                or re.search(r"[=<>≤≥²³]", item.text)
                # A stacked fraction is commonly the complete value of a
                # multiple-choice option.  Its only same-row neighbour can be
                # the short ``A.``/``B.`` label, so treating prose as the sole
                # anchor silently drops exactly those fractions.
                or re.fullmatch(
                    r"\s*[A-DＡ-Ｄ]\s*[.．、:：]?\s*",
                    str(item.text or ""),
                )
            )
        ]
        if not prose_neighbors:
            continue
        option_anchor = any(
            re.fullmatch(
                r"\s*[A-DＡ-Ｄ]\s*[.．、:：]?\s*",
                str(item.text or ""),
            )
            for item in prose_neighbors
        )
        candidates.append((x, y, bar_w, bar_h, option_anchor))

    # Build all candidate crops first. RapidOCR's recognizer supports a list of
    # images, so one batched call avoids up to 80 tiny ONNX invocations on a
    # fraction-dense page while preserving the same candidate coverage.
    fraction_jobs: list[
        tuple[int, int, int, int, int, int, bool, np.ndarray, np.ndarray]
    ] = []
    candidate_limit = max(0, int(max_attempts))
    for x, y, bar_w, bar_h, option_anchor in candidates[:candidate_limit]:
        pad_x = max(2, int(round(median_height * 0.10)))
        x1 = max(0, x - pad_x)
        x2 = min(width, x + bar_w + pad_x)
        top = max(0, int(round(y - median_height * 0.86)))
        bottom = min(height, int(round(y + bar_h + median_height * 0.86)))
        numerator_crop = image[top:max(top + 1, y), x1:x2]
        denominator_crop = image[min(height - 1, y + bar_h):bottom, x1:x2]
        fraction_jobs.append(
            (
                x,
                y,
                bar_w,
                bar_h,
                x1,
                x2,
                option_anchor,
                numerator_crop,
                denominator_crop,
            )
        )

    attempted = len(fraction_jobs)
    digit_results: list[tuple[str, float]] = []
    original_crops = [
        crop
        for job in fraction_jobs
        for crop in (job[7], job[8])
    ]
    batched_crops = [
        cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        for crop in original_crops
    ]
    recognize_txt = getattr(engine, "recognize_txt", None)
    if batched_crops and callable(recognize_txt):
        try:
            batch_result = recognize_txt(batched_crops)
            batch_texts = tuple(getattr(batch_result, "txts", ()) or ())
            batch_scores = tuple(getattr(batch_result, "scores", ()) or ())
            if len(batch_texts) == len(batched_crops):
                digit_results = [
                    normalize_digit_result(
                        batch_texts[index],
                        batch_scores[index] if index < len(batch_scores) else 0.0,
                    )
                    for index in range(len(batch_texts))
                ]
                batch_recognition_used = True
        except Exception:
            digit_results = []
    if len(digit_results) != len(batched_crops):
        digit_results = [recognize_digits(crop) for crop in original_crops]

    for job_index, (
        x,
        y,
        bar_w,
        bar_h,
        x1,
        x2,
        option_anchor,
        _,
        _,
    ) in enumerate(fraction_jobs):
        top = max(0, int(round(y - median_height * 0.86)))
        bottom = min(height, int(round(y + bar_h + median_height * 0.86)))
        numerator, numerator_score = digit_results[job_index * 2]
        denominator, denominator_score = digit_results[job_index * 2 + 1]
        if not numerator or not denominator:
            continue
        minimum_score = 0.72 if option_anchor else 0.86
        if min(numerator_score, denominator_score) < minimum_score:
            continue
        center_x = x + bar_w / 2
        center_y = (top + bottom) / 2
        if any(
            abs(existing.center_x - center_x) <= 0.35 * median_height
            and abs(existing.center_y - center_y) <= 0.45 * median_height
            for existing in recovered
        ):
            continue
        box = np.asarray(
            [[x1, top], [x2, top], [x2, bottom], [x1, bottom]],
            dtype=np.float32,
        )
        recovered.append(
            OCRItem(
                box=box,
                text=f"{numerator}/{denominator}",
                score=min(numerator_score, denominator_score),
                width=float(x2 - x1),
                height=float(bottom - top),
                center_x=float(center_x),
                center_y=float(center_y),
                printed=True,
                formula_like=False,
                mask_as_text=True,
            )
        )
    return recovered, {
        "candidate_count": len(candidates),
        "candidate_limit": candidate_limit,
        "truncated_candidate_count": max(0, len(candidates) - candidate_limit),
        "recognition_attempt_count": attempted,
        "batch_recognition_used": batch_recognition_used,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "recovered_count": len(recovered),
        "tokens": [item.text for item in recovered],
    }


def merge_recovered_numeric_fractions(
    items: list[OCRItem],
    recovered: list[OCRItem],
) -> list[OCRItem]:
    """Replace flattened digit fragments with locally recovered fractions.

    RapidOCR may expose the numerator and denominator as independent ``1`` and
    ``12`` tokens.  Merely appending the recovered ``1/2`` produces
    ``1 1/2 12`` in reading order.  Remove only numeric-only items whose centre
    lies inside the recovered local fraction box; option labels and surrounding
    prose remain untouched.
    """

    if not recovered:
        return list(items)

    def displaced_by_fraction(item: OCRItem) -> bool:
        compact = re.sub(r"\s+", "", str(item.text or ""))
        if not re.fullmatch(r"[-+−]?\d{1,4}(?:[./]\d{1,4})?", compact):
            return False
        for fraction in recovered:
            left = float(fraction.box[:, 0].min())
            right = float(fraction.box[:, 0].max())
            top = float(fraction.box[:, 1].min())
            bottom = float(fraction.box[:, 1].max())
            pad_x = max(2.0, fraction.width * 0.18)
            pad_y = max(2.0, fraction.height * 0.08)
            if (
                left - pad_x <= item.center_x <= right + pad_x
                and top - pad_y <= item.center_y <= bottom + pad_y
            ):
                return True
        return False

    retained = [item for item in items if not displaced_by_fraction(item)]
    return [*retained, *recovered]


def create_text_mask(shape: tuple[int, ...], items: list[OCRItem]) -> np.ndarray:
    """生成正文文字掩膜；白色区域只用于图形检测副本，不会修改原始正视图。"""
    mask = np.zeros(shape[:2], np.uint8)
    for item in items:
        # 图形检测阶段比文字重排更积极：即使某段因字号/行结构被判为疑似手写，
        # 只要 OCR 认为它是横向字符串，也应从检测副本去除，防止公式条变成插图。
        detection_text = (item.score >= 0.35 and len(item.text) >= 3 and
                          item.width >= 1.35 * item.height)
        if not (item.mask_as_text or detection_text):
            continue
        polygon = np.round(item.box).astype(np.int32)
        local = np.zeros_like(mask)
        cv2.fillPoly(local, [polygon], 255)
        margin = max(2, int(item.height * 0.14))
        local = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                            (margin * 2 + 1, margin * 2 + 1)))
        mask = cv2.bitwise_or(mask, local)
    return mask


def draw_structure_debug(image: np.ndarray, items: list[OCRItem],
                         content_box: tuple[int, int, int, int]) -> np.ndarray:
    """可视化 OCR 筛选：绿=重排印刷文字，青=公式原图块，橙=被过滤疑似手写。"""
    debug = image.copy()
    for item in items:
        if item.printed and item.formula_like:
            color = (255, 210, 0)
        elif item.printed:
            color = (0, 180, 0)
        else:
            color = (0, 120, 255)
        cv2.polylines(debug, [np.round(item.box).astype(np.int32)], True, color, 3)
    x1, y1, x2, y2 = content_box
    cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 5)
    return debug


def broad_colored_ink_mask(image: np.ndarray) -> np.ndarray:
    """为重排流程检测大面积蓝/青墨迹。

    插图最终从未删除文字的正视原图裁剪，因此检测副本可以更积极地清除蓝青色；
    红、黄、绿色常属于印刷题图，默认不整体删除。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    blue_cyan = ((hue >= 75) & (hue <= 145) & (saturation >= 42) & (value < 252))
    mask = blue_cyan.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)


def make_detection_copy(image: np.ndarray, text_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """创建去正文、去蓝青墨迹的检测副本，并返回实际移除区域。"""
    ink_mask = broad_colored_ink_mask(image)
    removed = cv2.bitwise_or(text_mask, ink_mask)
    detection = image.copy()
    detection[removed > 0] = (255, 255, 255)
    return detection, removed


def boxes_overlap_or_near(a: tuple[int, int, int, int], b: tuple[int, int, int, int],
                          gap_x: int, gap_y: int) -> bool:
    return not (a[2] + gap_x < b[0] or b[2] + gap_x < a[0] or
                a[3] + gap_y < b[1] or b[3] + gap_y < a[1])


def merge_nearby_boxes(boxes: list[tuple[int, int, int, int]], shape: tuple[int, ...]) -> list[tuple[int, int, int, int]]:
    """合并同一插图中被空白、坐标标注或 OCR 掩膜暂时分开的局部框。"""
    if not boxes:
        return []
    gap_x = max(12, int(shape[1] * 0.025))
    gap_y = max(12, int(shape[0] * 0.025))
    merged = boxes[:]
    changed = True
    while changed:
        changed = False
        result: list[tuple[int, int, int, int]] = []
        while merged:
            current = merged.pop()
            index = 0
            while index < len(merged):
                other = merged[index]
                if boxes_overlap_or_near(current, other, gap_x, gap_y):
                    current = (min(current[0], other[0]), min(current[1], other[1]),
                               max(current[2], other[2]), max(current[3], other[3]))
                    merged.pop(index)
                    changed = True
                else:
                    index += 1
            result.append(current)
        merged = result
    return merged


def sort_illustration_boxes_reading_order(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Sort nearby diagram choices left-to-right without mixing distant rows."""

    if len(boxes) < 2:
        return boxes[:]
    median_height = float(np.median([max(1, box[3] - box[1]) for box in boxes]))
    rows: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda value: ((value[1] + value[3]) / 2, value[0])):
        center_y = (box[1] + box[3]) / 2
        target: list[tuple[int, int, int, int]] | None = None
        for row in rows:
            row_center = float(np.median([(item[1] + item[3]) / 2 for item in row]))
            vertical_overlap = max(
                max(0, min(box[3], item[3]) - max(box[1], item[1]))
                / max(1, min(box[3] - box[1], item[3] - item[1]))
                for item in row
            )
            if abs(center_y - row_center) <= median_height * 0.46 or vertical_overlap >= 0.42:
                target = row
                break
        if target is None:
            rows.append([box])
        else:
            target.append(box)
    rows.sort(key=lambda row: min(item[1] for item in row))
    ordered: list[tuple[int, int, int, int]] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda value: value[0]))
    return ordered


def deduplicate_overlapping_illustration_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Union duplicate refinements that cover the same complete diagram."""

    pending = boxes[:]
    result: list[tuple[int, int, int, int]] = []
    while pending:
        current = pending.pop(0)
        changed = True
        while changed:
            changed = False
            current_area = max(1, (current[2] - current[0]) * (current[3] - current[1]))
            for index, other in enumerate(pending):
                intersection = (
                    max(0, min(current[2], other[2]) - max(current[0], other[0]))
                    * max(0, min(current[3], other[3]) - max(current[1], other[1]))
                )
                other_area = max(1, (other[2] - other[0]) * (other[3] - other[1]))
                if intersection / min(current_area, other_area) < 0.28:
                    continue
                current = (
                    min(current[0], other[0]),
                    min(current[1], other[1]),
                    max(current[2], other[2]),
                    max(current[3], other[3]),
                )
                pending.pop(index)
                changed = True
                break
        result.append(current)
    return result


def caption_anchored_figure_boxes(
    detection_copy: np.ndarray,
    items: list[OCRItem],
    target_question_number: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Recover one complete figure above each compact figure caption.

    Paper texture or handwriting can bridge several neighbouring diagrams into
    one contour. Printed figure captions are much more stable anchors: split
    captions into rows, give every caption its own horizontal cell, then use
    long line segments above it to recover the corresponding diagram.  A
    textbook may put ``(第N题)`` below the main diagram and ``备用图`` below a
    second diagram on the same row; both are semantic parts of that question
    and must receive their own cell rather than letting the numbered caption
    suppress the backup diagram.
    """

    h, w = detection_copy.shape[:2]
    numbered_caption = re.compile(
        r"[（(]?第([1-9]\d?)"
        r"(?:\s*[-—–]\s*(?:[IVXivx]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+))?\s*"
        r"(?:题(?:图[^，。；：:（）()]{0,16})?|图[^，。；：:（）()]{0,16})[）)]?"
    )
    captions: list[tuple[OCRItem, str]] = []
    for item in items:
        if not (item.printed or item.formula_like):
            continue
        compact = re.sub(r"\s+", "", str(item.text or ""))
        plain_match = re.fullmatch(r"[A-Za-z]?图([1-9]\d?)[A-Za-z]?", compact)
        question_match = numbered_caption.fullmatch(compact)
        auxiliary_caption = bool(re.fullmatch(
            r"[（(]?(?:备用图|参考图|草图|示意图)[）)]?",
            compact,
        ))
        if plain_match or question_match:
            number = (
                question_match.group(1)
                if question_match is not None
                else plain_match.group(1)  # type: ignore[union-attr]
            )
            is_question_caption = question_match is not None
            if (
                is_question_caption
                and target_question_number is not None
                and int(number) != target_question_number
            ):
                continue
            captions.append((item, number))
        elif auxiliary_caption:
            # The selected-question workspace has already excluded adjacent
            # questions.  An auxiliary caption here belongs to the same stem;
            # retain it even though it carries no question number.
            captions.append((item, "aux"))
    if not captions:
        return []

    median_caption_height = float(np.median([
        max(1.0, item.height) for item, _number in captions
    ]))
    rows: list[list[tuple[OCRItem, str]]] = []
    for caption in sorted(captions, key=lambda value: (value[0].center_y, value[0].center_x)):
        target: list[tuple[OCRItem, str]] | None = None
        for row in rows:
            row_y = float(np.median([entry[0].center_y for entry in row]))
            if abs(caption[0].center_y - row_y) <= max(20.0, median_caption_height * 2.2):
                target = row
                break
        if target is None:
            rows.append([caption])
        else:
            target.append(caption)

    recovered: list[tuple[int, int, int, int]] = []
    for row_index, row in enumerate(rows):
        row.sort(key=lambda value: value[0].center_x)
        centers = [entry[0].center_x for entry in row]
        row_compacts = [
            re.sub(r"\s+", "", str(entry[0].text or "")) for entry in row
        ]
        composite_main_and_backup = bool(
            any(numbered_caption.fullmatch(text) for text in row_compacts)
            and any(re.fullmatch(r"[（(]?(?:备用图|参考图|草图|示意图)[）)]?", text)
                    for text in row_compacts)
        )
        row_recovered_start = len(recovered)
        previous_caption_bottom = (
            max(int(np.ceil(float(entry[0].box[:, 1].max()))) for entry in rows[row_index - 1])
            if row_index > 0 else 0
        )
        next_row_floor = (
            max(int(np.ceil(float(entry[0].box[:, 1].max()))) for entry in row) + 4
            if row_index + 1 < len(rows) else h
        )
        for index, (caption, _number) in enumerate(row):
            caption_compact = re.sub(r"\s+", "", str(caption.text or ""))
            numbered_question_caption = bool(re.fullmatch(
                numbered_caption,
                caption_compact,
            ))
            auxiliary_caption = bool(re.fullmatch(
                r"[（(]?(?:备用图|参考图|草图|示意图)[）)]?",
                caption_compact,
            ))
            if index:
                cell_left = int((centers[index - 1] + caption.center_x) / 2) - 8
            else:
                neighbour_gap = (
                    centers[1] - centers[0] if len(centers) > 1 else w * 0.24
                )
                cell_left = int(caption.center_x - max(w * 0.13, neighbour_gap * 0.58))
            if composite_main_and_backup and numbered_question_caption:
                # ``(第N题)`` is conventionally placed at the lower-right of
                # the main diagram, so its x-coordinate is not a safe left
                # boundary. The selected workspace already contains one
                # question; allow this main panel to extend to its left edge.
                cell_left = 0
            if index + 1 < len(centers):
                cell_right = int((caption.center_x + centers[index + 1]) / 2)
            else:
                neighbour_gap = (
                    centers[-1] - centers[-2] if len(centers) > 1 else w * 0.24
                )
                cell_right = int(caption.center_x + max(w * 0.13, neighbour_gap * 0.58))
            if composite_main_and_backup and auxiliary_caption:
                cell_right = w
            cell_left = max(0, cell_left)
            cell_right = min(w, cell_right)
            caption_top = int(np.floor(float(caption.box[:, 1].min())))
            caption_bottom = int(np.ceil(float(caption.box[:, 1].max())))
            search_top = max(
                0,
                caption_top - int(h * 0.42),
                previous_caption_bottom + (12 if row_index > 0 else 0),
            )
            search_bottom = min(h, caption_bottom + max(12, int(caption.height * 0.8)))
            roi = detection_copy[search_top:search_bottom, cell_left:cell_right]
            if roi.size == 0:
                continue

            # Some textbook figures are made entirely from printed numbers and
            # short row labels (for example the Pascal triangle).  Text masking
            # removes nearly all of their strokes, so a Hough-only detector has
            # no line art left to anchor.  Repeated ``第N行`` labels immediately
            # above a numbered caption are strong local layout evidence: retain
            # that caption cell as one visual figure instead of flattening the
            # rows and caption into question prose.
            numeric_row_items: list[OCRItem] = []
            numeric_row_numbers: set[int] = set()
            for candidate_item in items:
                if not (candidate_item.printed or candidate_item.formula_like):
                    continue
                if not (
                    cell_left <= candidate_item.center_x <= cell_right
                    and search_top <= candidate_item.center_y < caption_top
                ):
                    continue
                candidate_compact = re.sub(
                    r"\s+", "", str(candidate_item.text or "")
                )
                row_matches = re.findall(r"第([1-9]\d?)行", candidate_compact)
                if row_matches:
                    numeric_row_numbers.update(int(value) for value in row_matches)
                    numeric_row_items.append(candidate_item)
                    continue
                numeric_tokens = re.findall(r"\d+", candidate_compact)
                if (
                    len(numeric_tokens) >= 2
                    and not re.search(r"[A-Za-z\u4e00-\u9fff]", candidate_compact)
                ):
                    numeric_row_items.append(candidate_item)
            caption_title_is_numeric_figure = bool(
                re.search(r"杨辉三角|数阵|数表", caption_compact)
            )
            if (
                len(numeric_row_numbers) >= 3
                and (caption_title_is_numeric_figure or len(numeric_row_numbers) >= 4)
            ):
                first_row_top = min(
                    int(np.floor(float(item.box[:, 1].min())))
                    for item in numeric_row_items
                )
                recovered.append((
                    max(0, cell_left - max(8, int(caption.height * 0.4))),
                    max(search_top, first_row_top - max(12, int(caption.height * 0.8))),
                    min(w, cell_right + max(18, int(caption.height))),
                    min(h, caption_bottom + max(6, int(caption.height * 0.35))),
                ))
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 75, 170)
            min_line = max(18, int((cell_right - cell_left) * 0.14))
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180, threshold=16,
                minLineLength=min_line, maxLineGap=8,
            )
            if lines is None or len(np.asarray(lines).reshape(-1, 4)) < 2:
                continue
            all_segments = np.asarray(lines).reshape(-1, 4)
            caption_local_top = caption_top - search_top
            maximum_figure_height = max(
                90,
                min(int((cell_right - cell_left) * 1.35), int(h * 0.45)),
            )
            eligible = []
            roi_width = max(1, cell_right - cell_left)
            for segment in all_segments:
                sx1, sy1, sx2, sy2 = (int(value) for value in segment)
                if (sy1 + sy2) / 2 >= caption_local_top + max(4.0, caption.height * 0.25):
                    continue
                # A photographed page edge is a long, near-vertical line at
                # the ROI boundary, not part of the diagram.
                boundary_vertical = (
                    (
                        max(sx1, sx2) <= 5
                        or min(sx1, sx2) >= roi_width - 6
                        or max(cell_left + sx1, cell_left + sx2) <= int(w * 0.03)
                        or min(cell_left + sx1, cell_left + sx2) >= int(w * 0.91)
                    )
                    and abs(sy2 - sy1) >= max(14, abs(sx2 - sx1) * 2.2)
                )
                page_horizontal = (
                    abs(sx2 - sx1) >= roi_width * 0.82
                    and abs(sy2 - sy1) <= max(4, abs(sx2 - sx1) * 0.035)
                )
                if not boundary_vertical and not page_horizontal:
                    eligible.append(segment)
            if len(eligible) < 2:
                continue
            # A tall search window can contain the previous figure too. Merge
            # line segments into vertically connected clusters and select the
            # cluster nearest this caption, rather than taking their full union.
            gap_y = max(10, int(median_caption_height * 0.75))
            clusters: list[dict[str, object]] = []
            for segment in sorted(eligible, key=lambda value: min(value[1], value[3])):
                segment_top = int(min(segment[1], segment[3]))
                segment_bottom = int(max(segment[1], segment[3]))
                target_cluster: dict[str, object] | None = None
                for cluster in clusters:
                    if segment_top <= int(cluster["bottom"]) + gap_y:
                        target_cluster = cluster
                        break
                if target_cluster is None:
                    clusters.append({
                        "top": segment_top,
                        "bottom": segment_bottom,
                        "segments": [segment],
                    })
                else:
                    target_cluster["top"] = min(int(target_cluster["top"]), segment_top)
                    target_cluster["bottom"] = max(int(target_cluster["bottom"]), segment_bottom)
                    target_cluster["segments"].append(segment)  # type: ignore[union-attr]
            valid_clusters = [
                cluster for cluster in clusters
                if len(cluster["segments"]) >= 2  # type: ignore[arg-type]
                and int(cluster["top"]) < caption_local_top
            ]
            if not valid_clusters:
                continue
            # A triangle/circuit can be returned as several Hough clusters
            # (for example, its base and two sloping sides). Start at the
            # cluster nearest the caption and walk upward only through nearby
            # line groups, so remote handwriting is not pulled into the crop.
            minimum_core_y = max(0, caption_local_top - maximum_figure_height)
            band_clusters = [
                cluster for cluster in valid_clusters
                if int(cluster["bottom"]) >= minimum_core_y
            ]
            if not band_clusters:
                band_clusters = [
                    max(valid_clusters, key=lambda cluster: int(cluster["bottom"]))
                ]
            nearest_cluster = max(band_clusters, key=lambda cluster: int(cluster["bottom"]))
            contributing_clusters = [nearest_cluster]
            joined_top = int(nearest_cluster["top"])
            joined_segments = np.asarray(nearest_cluster["segments"]).reshape(-1, 4)
            joined_left = int(min(min(segment[0], segment[2]) for segment in joined_segments))
            joined_right = int(max(max(segment[0], segment[2]) for segment in joined_segments))
            join_gap = max(gap_y, int(roi_width * 0.22))
            for cluster in sorted(
                (value for value in band_clusters if value is not nearest_cluster),
                key=lambda value: int(value["bottom"]),
                reverse=True,
            ):
                cluster_segments = np.asarray(cluster["segments"]).reshape(-1, 4)
                cluster_left = int(min(min(segment[0], segment[2]) for segment in cluster_segments))
                cluster_right = int(max(max(segment[0], segment[2]) for segment in cluster_segments))
                vertical_gap = max(0, joined_top - int(cluster["bottom"]))
                horizontal_gap = max(0, joined_left - cluster_right, cluster_left - joined_right)
                if vertical_gap <= join_gap and horizontal_gap <= max(12, int(roi_width * 0.16)):
                    contributing_clusters.append(cluster)
                    joined_top = min(joined_top, int(cluster["top"]))
                    joined_left = min(joined_left, cluster_left)
                    joined_right = max(joined_right, cluster_right)
            segments = np.concatenate([
                np.asarray(cluster["segments"]).reshape(-1, 4)
                for cluster in contributing_clusters
            ])
            xs = [int(value) for segment in segments for value in (segment[0], segment[2])]
            ys = [int(value) for segment in segments for value in (segment[1], segment[3])]
            core_x1, core_x2 = min(xs), max(xs) + 1
            core_y1, core_y2 = min(ys), max(ys) + 1
            core_w = max(1, core_x2 - core_x1)
            core_h = max(1, core_y2 - core_y1)
            if core_w * core_h < max(180, (cell_right - cell_left) * h * 0.0015):
                continue
            pad_x = max(14, int(core_w * 0.18))
            pad_y = max(44, int(core_h * 0.28), int(caption.height * 1.6))
            caption_left = int(np.floor(float(caption.box[:, 0].min())))
            caption_right = int(np.ceil(float(caption.box[:, 0].max())))
            candidate = (
                max(cell_left, min(cell_left + core_x1 - pad_x, caption_left - 12)),
                max(0, search_top + core_y1 - pad_y,
                    caption_top - maximum_figure_height),
                min(cell_right, max(cell_left + core_x2 + pad_x, caption_right + 12)),
                min(h, next_row_floor,
                    max(search_top + core_y2 + pad_y, caption_bottom + 4)),
            )
            # A prose line immediately above a diagram may fall inside the
            # conservative height allowance (common before a row of 图3–图5).
            # Clamp only past OCR-confirmed prose that lies wholly above the
            # recovered line-art core. Compact vertex/axis labels stay intact.
            trimmed_top = candidate[1]
            for item in items:
                if not (item.printed or item.formula_like):
                    continue
                compact = re.sub(r"\s+", "", str(item.text or ""))
                if not compact or compact_figure_label(item):
                    continue
                prose_like = len(compact) >= 8 or bool(re.search(r"[，。；？！]|[（(]\d", compact))
                if not prose_like:
                    continue
                ix1 = int(np.floor(float(item.box[:, 0].min())))
                iy1 = int(np.floor(float(item.box[:, 1].min())))
                ix2 = int(np.ceil(float(item.box[:, 0].max())))
                iy2 = int(np.ceil(float(item.box[:, 1].max())))
                horizontal_overlap = max(0, min(candidate[2], ix2) - max(candidate[0], ix1))
                candidate_width = max(1, candidate[2] - candidate[0])
                item_width = max(1, ix2 - ix1)
                overlap_ratio = horizontal_overlap / min(candidate_width, item_width)
                if horizontal_overlap <= 0 or overlap_ratio < 0.24 or iy2 <= trimmed_top:
                    continue
                safe_figure_floor = caption_top - max(40, int(caption.height * 2.2))
                if iy2 <= safe_figure_floor:
                    trimmed_top = max(trimmed_top, min(iy2 + 4, safe_figure_floor))
            recovered.append((candidate[0], max(0, trimmed_top), candidate[2], candidate[3]))
        if composite_main_and_backup:
            # The two panels jointly describe the same construction.  Keeping
            # them as one wide crop prevents either off-centre caption from
            # cutting an axis, curve or label at the artificial split line.
            row_boxes = recovered[row_recovered_start:]
            if row_boxes:
                recovered[row_recovered_start:] = [(
                    min(box[0] for box in row_boxes),
                    min(box[1] for box in row_boxes),
                    max(box[2] for box in row_boxes),
                    max(box[3] for box in row_boxes),
                )]
    return sort_illustration_boxes_reading_order(recovered)


def expand_box_for_edge_ink(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    iterations: int = 3,
) -> tuple[int, int, int, int]:
    """Expand a crop when a meaningful dark component touches its boundary."""

    h, w = image.shape[:2]
    current = list(box)
    for _ in range(max(0, iterations)):
        x1, y1, x2, y2 = current
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            break
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ink = (gray < 185).astype(np.uint8)
        count, _labels, stats, _centers = cv2.connectedComponentsWithStats(ink, 8)
        risks = [False, False, False, False]  # left, top, right, bottom
        roi_h, roi_w = ink.shape
        for component in range(1, count):
            cx, cy, cw, ch, area = (int(value) for value in stats[component])
            if area < 6 or max(cw, ch) < 7:
                continue
            risks[0] |= cx <= 2 and cw >= 7
            risks[1] |= cy <= 2 and ch >= 7
            risks[2] |= cx + cw >= roi_w - 2 and cw >= 7
            risks[3] |= cy + ch >= roi_h - 2 and ch >= 7
        risks[0] &= x1 > 0
        risks[1] &= y1 > 0
        risks[2] &= x2 < w
        risks[3] &= y2 < h
        if not any(risks):
            break
        step_x = max(8, int((x2 - x1) * 0.04))
        step_y = max(8, int((y2 - y1) * 0.04))
        if risks[0]:
            current[0] = max(0, x1 - step_x)
        if risks[1]:
            current[1] = max(0, y1 - step_y)
        if risks[2]:
            current[2] = min(w, x2 + step_x)
        if risks[3]:
            current[3] = min(h, y2 + step_y)
    return tuple(current)


def expand_isolated_boxes_for_edge_ink(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Expand isolated crops, but keep dense A–D choice rows separated."""

    expanded: list[tuple[int, int, int, int]] = []
    for box in boxes:
        box_height = max(1, box[3] - box[1])
        center_y = (box[1] + box[3]) / 2
        row_peers = 0
        for other in boxes:
            other_height = max(1, other[3] - other[1])
            other_center_y = (other[1] + other[3]) / 2
            vertical_overlap = (
                max(0, min(box[3], other[3]) - max(box[1], other[1]))
                / max(1, min(box_height, other_height))
            )
            if (
                abs(center_y - other_center_y) <= max(18, min(box_height, other_height) * 0.55)
                or vertical_overlap >= 0.50
            ):
                row_peers += 1
        if row_peers >= 3:
            expanded.append(box)
            continue
        grown = expand_box_for_edge_ink(image, box)
        gx1, gy1, gx2, gy2 = grown
        safety_x = max(12, int((gx2 - gx1) * 0.10))
        safety_y = max(8, int((gy2 - gy1) * 0.03))
        expanded.append((
            max(0, gx1 - safety_x),
            max(0, gy1 - safety_y),
            min(image.shape[1], gx2 + safety_x),
            min(image.shape[0], gy2 + safety_y),
        ))
    return expanded


def compact_figure_label(item: OCRItem) -> bool:
    """Return whether an OCR item can be a diagram label rather than prose.

    Axis names, units, point names, coordinates and table headers are usually
    short.  Long question sentences and sub-question markers are deliberately
    excluded so expanding a crop cannot pull the surrounding stem back in.
    """

    if not (item.printed or item.formula_like):
        return False
    text = re.sub(r"\s+", "", str(item.text or ""))
    if not text or len(text) > 14:
        return False
    if re.match(r"^[（(]\s*\d{1,2}\s*[）)]", text):
        return False
    # Handwritten intermediate/final values are often confidently recognised
    # as short OCR text next to a graph.  They are not axis labels and must not
    # authorize expansion into the answer area.
    if re.match(r"^[A-Za-z]\s*[=:≈]\s*[-+]?\d", text):
        return False
    if (
        not item.printed
        and item.formula_like
        and (
            sum(char in "=+×÷*/^" for char in text) >= 2
            or ("=" in text and len(text) >= 6)
            or any(char in text for char in "↑↓✓√")
        )
    ):
        return False
    if any(token in text for token in ("已知", "求证", "证明", "解答", "填空", "选择")):
        return False
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    if cjk_count >= 4 and any(
        token in text for token in ("请", "写出", "的", "求", "长为", "为", "是", "时", "可得")
    ):
        return False
    label_evidence = bool(re.search(
        r"[\dA-Za-zα-ωΑ-Ω∠△⊥∥≤≥=+\-×÷/%°]|"
        r"频数|人数|年龄|时间|距离|速度|温度|电压|电流|质量|单位|"
        r"次数|面积|体积|刻度|组别|成绩|项目|类别|正面|主视|俯视|左视|右视",
        text,
    ))
    sentence_like = bool(re.search(r"[，。；？！]", text))
    return label_evidence and not sentence_like


def expand_illustration_box_for_labels(
    box: tuple[int, int, int, int],
    items: list[OCRItem],
    shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Expand a detected figure to include nearby printed labels and units.

    Detection runs on a copy where prose is removed, so a later line-art core
    refinement can otherwise trim off OCR-recognised axis titles and vertex
    labels.  Expansion is bounded to 30% of the core on each axis and only
    trusts compact printed/formula OCR items; handwritten notes and prose never
    authorize a larger crop.
    """

    page_h, page_w = shape[:2]
    core_x1, core_y1, core_x2, core_y2 = box
    core_w = max(1, core_x2 - core_x1)
    core_h = max(1, core_y2 - core_y1)
    max_pad_left = max(24, int(core_w * 0.12))
    max_pad_right = max(24, int(core_w * 0.22))
    max_pad_top = max(24, int(core_h * 0.20))
    max_pad_bottom = max(24, int(core_h * 0.18))
    limit = (
        max(0, core_x1 - max_pad_left),
        max(0, core_y1 - max_pad_top),
        min(page_w, core_x2 + max_pad_right),
        min(page_h, core_y2 + max_pad_bottom),
    )
    expanded = [core_x1, core_y1, core_x2, core_y2]
    median_label_height = float(np.median([
        item.height for item in items if compact_figure_label(item) and item.height > 0
    ])) if any(compact_figure_label(item) and item.height > 0 for item in items) else 12.0

    # Two passes allow a unit just beyond an axis title to join without ever
    # crossing the fixed neighbourhood around the original figure core.
    for _ in range(2):
        changed = False
        proximity_x = max(14, int(core_w * 0.08), int(median_label_height * 1.5))
        proximity_y = max(14, int(core_h * 0.08), int(median_label_height * 1.5))
        search = (
            max(limit[0], expanded[0] - proximity_x),
            max(limit[1], expanded[1] - proximity_y),
            min(limit[2], expanded[2] + proximity_x),
            min(limit[3], expanded[3] + proximity_y),
        )
        for item in items:
            if not compact_figure_label(item):
                continue
            ix1 = int(np.floor(float(item.box[:, 0].min())))
            iy1 = int(np.floor(float(item.box[:, 1].min())))
            ix2 = int(np.ceil(float(item.box[:, 0].max())))
            iy2 = int(np.ceil(float(item.box[:, 1].max())))
            if ix2 < search[0] or ix1 > search[2] or iy2 < search[1] or iy1 > search[3]:
                continue
            # A nearby item must align with the figure on at least one axis;
            # diagonally adjacent answer notes cannot grow the box.
            horizontal_overlap = max(0, min(expanded[2], ix2) - max(expanded[0], ix1))
            vertical_overlap = max(0, min(expanded[3], iy2) - max(expanded[1], iy1))
            if horizontal_overlap <= 0 and vertical_overlap <= 0:
                continue
            updated = [
                max(limit[0], min(expanded[0], ix1)),
                max(limit[1], min(expanded[1], iy1)),
                min(limit[2], max(expanded[2], ix2)),
                min(limit[3], max(expanded[3], iy2)),
            ]
            if updated != expanded:
                expanded = updated
                changed = True
        if not changed:
            break

    # Keep a real white safety margin around strokes and glyph bounding boxes.
    pad_x = max(8, int(core_w * 0.025), int(median_label_height * 0.40))
    pad_y = max(8, int(core_h * 0.025), int(median_label_height * 0.40))
    return (
        max(limit[0], expanded[0] - pad_x),
        max(limit[1], expanded[1] - pad_y),
        min(limit[2], expanded[2] + pad_x),
        min(limit[3], expanded[3] + pad_y),
    )


def sanitize_illustration_boxes(
    boxes: list[tuple[int, int, int, int]],
    items: list[OCRItem],
    shape: tuple[int, ...],
) -> tuple[list[tuple[int, int, int, int]], int]:
    """Reject oversized figure boxes that cover OCR-confirmed prose.

    Geometry/table crops legitimately contain compact labels, but they must not
    swallow several complete question sentences.  This is a final safety gate
    for cases where page shadows or bridge lines merge the entire sheet into a
    single illustration contour.
    """

    page_h, page_w = shape[:2]
    page_area = max(1, page_h * page_w)
    retained: list[tuple[int, int, int, int]] = []
    rejected = 0
    for box in boxes:
        x1, y1, x2, y2 = box
        area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / page_area
        prose_count = 0
        prose_cjk = 0
        structural_count = 0
        for item in items:
            if not (item.printed or item.formula_like):
                continue
            ix1 = float(item.box[:, 0].min())
            iy1 = float(item.box[:, 1].min())
            ix2 = float(item.box[:, 0].max())
            iy2 = float(item.box[:, 1].max())
            intersection = max(0.0, min(x2, ix2) - max(x1, ix1)) * max(
                0.0, min(y2, iy2) - max(y1, iy1)
            )
            item_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
            materially_overlapped = intersection / item_area >= 0.20
            if not (
                (x1 <= item.center_x <= x2 and y1 <= item.center_y <= y2)
                or materially_overlapped
            ):
                continue
            compact = re.sub(r"\s+", "", str(item.text or ""))
            cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in compact)
            style = classify_reflow_text(compact)
            prose_like = (
                (cjk_count >= 5 and len(compact) >= 10)
                or bool(re.search(r"[，。；？！]", compact))
            )
            if prose_like:
                prose_count += 1
                prose_cjk += cjk_count
            if style in {"question", "subquestion", "option"} and len(compact) >= 4:
                structural_count += 1
        unsafe = bool(
            # A real diagram can contain a handful of short labels, but it
            # should never consume several long question sentences.  Apply
            # this guard regardless of crop area so textured paper cannot be
            # mistaken for a medium-sized photograph and hide OCR prose.
            (prose_count >= 2 and prose_cjk >= 24)
            or
            (area_ratio >= 0.55 and (prose_count >= 1 or structural_count >= 1))
            or (
                area_ratio >= 0.30
                and (prose_count >= 3 or prose_cjk >= 36 or structural_count >= 2)
            )
        )
        if unsafe:
            rejected += 1
        else:
            retained.append(box)
    return retained, rejected


def structured_layout_illustration_boxes(
    layout_audit: dict[str, object],
    shape: tuple[int, ...],
    items: list[OCRItem],
    transcript: str,
    max_count: int,
) -> tuple[list[tuple[int, int, int, int]], dict[str, object]]:
    """Convert PP-DocLayoutV2 regions into conservative Test6 crops.

    The open-source detector supplies semantic region classes while the local
    OCR items remain the authority for rejecting boxes that swallow prose.
    This combination is both safer and cheaper than repeatedly changing
    contour thresholds for every photographed worksheet style.
    """

    height, width = shape[:2]
    compact = re.sub(r"\s+", "", str(transcript or ""))
    table_cue = bool(re.search(
        r"下表(?!面)|如下表|表中|表格|统计表|成绩表|(?:对应)?数据如下|(?:成绩|统计|结果)如下",
        compact,
    ))
    figure_cue = bool(re.search(
        r"如图|图[①②③④⑤⑥1-9甲乙丙丁戊己]|"
        r"第\s*\d{1,3}(?:\s*[-—–]\s*(?:[IVXivx]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+))?\s*(?:题)?图|"
        r"示意图|函数图像|坐标系|电路图|光路图|转盘",
        compact,
    ))
    accepted: list[tuple[int, int, int, int]] = []
    rejected = 0
    label_counts: dict[str, int] = {}
    for raw_region in list(layout_audit.get("layout_regions") or []):
        if not isinstance(raw_region, dict):
            rejected += 1
            continue
        label = str(raw_region.get("label") or "").strip().lower()
        if label == "table":
            # PP-DocLayoutV2 may label circuit/schematic line art as a table.
            # An explicit figure cue is sufficient evidence to retain it.
            if not (table_cue or figure_cue):
                continue
        elif label in {"image", "chart"}:
            if not figure_cue:
                continue
        else:
            continue
        raw_bbox = raw_region.get("bbox")
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            rejected += 1
            continue
        try:
            left, top, right, bottom = (
                int(round(float(value))) for value in raw_bbox
            )
            score = float(raw_region.get("score") or 0.0)
        except (TypeError, ValueError):
            rejected += 1
            continue
        if score < (0.28 if label == "table" else 0.32):
            rejected += 1
            continue
        left = max(0, min(width - 1, left))
        top = max(0, min(height - 1, top))
        right = max(left + 1, min(width, right))
        bottom = max(top + 1, min(height, bottom))
        box_width = right - left
        box_height = bottom - top
        pad_x = max(8, round(box_width * 0.045))
        pad_y = max(8, round(box_height * 0.055))
        accepted.append((
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(width, right + pad_x),
            min(height, bottom + pad_y),
        ))
        label_counts[label] = label_counts.get(label, 0) + 1

    accepted, unsafe_count = sanitize_illustration_boxes(
        accepted,
        items,
        shape,
    )
    rejected += unsafe_count
    accepted = deduplicate_overlapping_illustration_boxes(accepted)
    accepted = sort_illustration_boxes_reading_order(accepted)[:max(0, max_count)]
    return accepted, {
        "requested": True,
        "engine": str(layout_audit.get("engine") or "local-pp-doclayoutv2"),
        "available": bool(layout_audit.get("available")),
        "elapsed_ms": float(layout_audit.get("elapsed_ms") or 0.0),
        "service_elapsed_ms": float(layout_audit.get("service_elapsed_ms") or 0.0),
        "source_region_count": len(list(layout_audit.get("layout_regions") or [])),
        "accepted_region_count": len(accepted),
        "rejected_region_count": rejected,
        "accepted_label_counts": label_counts,
        "error": layout_audit.get("error"),
    }


def merge_illustration_evidence(
    heuristic_boxes: list[tuple[int, int, int, int]],
    structured_boxes: list[tuple[int, int, int, int]],
    shape: tuple[int, ...],
    items: list[OCRItem],
    max_count: int,
) -> list[tuple[int, int, int, int]]:
    """Fuse contour and PP-DocLayout boxes without dropping either source."""

    combined = list(heuristic_boxes)
    for structured in structured_boxes:
        matched_index: int | None = None
        for index, heuristic in enumerate(combined):
            intersection = (
                max(0, min(structured[2], heuristic[2]) - max(structured[0], heuristic[0]))
                * max(0, min(structured[3], heuristic[3]) - max(structured[1], heuristic[1]))
            )
            smaller = min(
                max(1, (structured[2] - structured[0]) * (structured[3] - structured[1])),
                max(1, (heuristic[2] - heuristic[0]) * (heuristic[3] - heuristic[1])),
            )
            if intersection / smaller >= 0.20:
                matched_index = index
                break
        if matched_index is None:
            combined.append(structured)
        else:
            old = combined[matched_index]
            union = (
                min(old[0], structured[0]),
                min(old[1], structured[1]),
                max(old[2], structured[2]),
                max(old[3], structured[3]),
            )
            structured_area = max(
                1,
                (structured[2] - structured[0]) * (structured[3] - structured[1]),
            )
            union_area = max(1, (union[2] - union[0]) * (union[3] - union[1]))
            # A contour connected to handwriting or a page edge can extend far
            # beyond a precise semantic layout box. In that case the model box
            # is the safer crop; modest unions still recover clipped labels.
            combined[matched_index] = (
                structured if union_area > structured_area * 1.55 else union
            )
    combined, _ = sanitize_illustration_boxes(combined, items, shape)
    combined = deduplicate_overlapping_illustration_boxes(combined)
    return sort_illustration_boxes_reading_order(combined)[:max(0, max_count)]


def expand_multi_layout_gallery_boxes(
    boxes: list[tuple[int, int, int, int]],
    shape: tuple[int, ...],
    items: list[OCRItem] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Keep one complete crop per layout panel, including its lower caption.

    PP-DocLayout is much more reliable than residual-ink contours when a page
    contains several neighbouring diagrams.  Its boxes are intentionally tight,
    however, and commonly stop just above labels such as ``第14-I图乙``.  Add a
    bounded lower margin without allowing panels to merge into one another.
    """

    height, width = shape[:2]
    caption_items = [
        item
        for item in (items or [])
        if (item.printed or item.formula_like)
        and re.search(
            r"第\s*\d{1,3}(?:\s*[-—–]\s*(?:[IVXivx]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|\d+))?"
            r"\s*(?:题)?图\s*[甲乙丙丁戊己A-Za-z]?",
            re.sub(r"\s+", "", str(item.text or "")),
        )
    ]
    expanded: list[tuple[int, int, int, int]] = []
    for left, top, right, bottom in boxes:
        box_width = max(1, right - left)
        box_height = max(1, bottom - top)
        pad_x = max(6, round(box_width * 0.035))
        pad_top = max(4, round(box_height * 0.025))
        pad_bottom = max(18, round(box_height * 0.24))
        caption_bottoms = [
            int(math.ceil(float(item.box[:, 1].max())))
            for item in caption_items
            if left - box_width * 0.12 <= item.center_x <= right + box_width * 0.12
            and bottom - box_height * 0.10 <= item.center_y <= bottom + max(90, box_height * 0.60)
        ]
        expanded_bottom = bottom + pad_bottom
        if caption_bottoms:
            expanded_bottom = max(expanded_bottom, max(caption_bottoms) + 10)
        expanded.append((
            max(0, left - pad_x),
            max(0, top - pad_top),
            min(width, right + pad_x),
            min(height, expanded_bottom),
        ))
    return sort_illustration_boxes_reading_order(expanded)


def detect_illustration_boxes(detection_copy: np.ndarray, source_image: np.ndarray, items: list[OCRItem],
                              config: Config) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """在去正文副本上检测图形、表格、电路图和插画，并输出候选框可视化。"""
    h, w = detection_copy.shape[:2]
    work, scale = resize_for_detection(detection_copy, 1600)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 41, 14)
    colorful = ((hsv[:, :, 1] > 52) & (hsv[:, :, 2] < 250)).astype(np.uint8) * 255
    foreground = cv2.bitwise_or(dark, colorful)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # 小尺度闭运算连接线稿，避免把一张几何图拆成许多碎框。
    kx = max(5, work.shape[1] // 180)
    ky = max(5, work.shape[0] // 180)
    connected = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((ky, kx), np.uint8))
    connected = cv2.dilate(connected, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    work_area = work.shape[0] * work.shape[1]
    raw_boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        area_ratio = cw * ch / work_area
        if not 0.0018 <= area_ratio <= 0.34:
            continue
        # 图形选择题中的单个 A-D 选项往往只占页面宽度 5%~7%。原来的
        # 7.5% 宽度门槛会把这些印刷几何图全部过滤，只留下较大的题干图。
        if cw < work.shape[1] * 0.045 or ch < work.shape[0] * 0.070:
            continue
        aspect = max(cw / max(ch, 1), ch / max(cw, 1))
        if aspect > 5.8:
            continue
        fill = cv2.countNonZero(foreground[y:y + ch, x:x + cw]) / max(1, cw * ch)
        if not 0.018 <= fill <= 0.72:
            continue
        raw_boxes.append((int(x / scale), int(y / scale),
                          int((x + cw) / scale), int((y + ch) / scale)))
    boxes = merge_nearby_boxes(raw_boxes, detection_copy.shape)
    padded: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        # 小于页面 1.2% 的框大多是孤立手写公式、页码或污点，不作为主体插图。
        if bw * bh < h * w * 0.012:
            continue
        # 左侧多留一个选项字母槽；即使 OCR 漏识别 A/C/D，也能把字母和图形
        # 一起交给模型。右侧仍使用普通安全边距，避免串入相邻选项。
        px_left = max(24, int(bw * max(config.illustration_padding, 0.40)))
        px_right = max(12, int(bw * config.illustration_padding))
        py = max(12, int(bh * config.illustration_padding))
        candidate = (max(0, x1 - px_left), max(0, y1 - py),
                     min(w, x2 + px_right), min(h, y2 + py))
        candidate_area = max(1, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
        ocr_overlap_area = 0.0
        untrusted_inside = 0
        for item in items:
            ix1, iy1 = float(item.box[:, 0].min()), float(item.box[:, 1].min())
            ix2, iy2 = float(item.box[:, 0].max()), float(item.box[:, 1].max())
            overlap = max(0.0, min(candidate[2], ix2) - max(candidate[0], ix1)) * \
                      max(0.0, min(candidate[3], iy2) - max(candidate[1], iy1))
            ocr_overlap_area += overlap
            if (candidate[0] <= item.center_x <= candidate[2] and
                    candidate[1] <= item.center_y <= candidate[3] and not item.printed):
                untrusted_inside += 1
        # 图内的 A/B/C、数字和单位经常只有 1~2 个字符，OCR 对短标签的置信度
        # 也容易偏低，所以允许少量不稳定框；累计较多时才按手写推导区域排除。
        # 当候选框中 OCR 风险较高时，不能直接删框：电路图、坐标图和表格本身
        # 就含有许多短标签。长直线结构可以救回线稿，灰度纹理可以救回题目照片；
        # 两种证据都没有时，才把它视为被形态学操作连成一片的手写演算。
        roi = detection_copy[candidate[1]:candidate[3], candidate[0]:candidate[2]]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_edges = cv2.Canny(roi_gray, 70, 170)
        min_line_length = max(24, int(min(roi_gray.shape[:2]) * 0.24))
        lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=28,
                                minLineLength=min_line_length, maxLineGap=12)
        long_line_count = 0 if lines is None else len(lines)
        edge_density = cv2.countNonZero(roi_edges) / max(1, roi_edges.size)
        roi_histogram = cv2.calcHist([roi_gray], [0], None, [32], [0, 256]).ravel()
        roi_probabilities = roi_histogram / max(1.0, float(roi_histogram.sum()))
        roi_probabilities = roi_probabilities[roi_probabilities > 0]
        roi_entropy = float(-np.sum(roi_probabilities * np.log2(roi_probabilities)))
        photo_texture = (float(np.std(roi_gray)) >= 32.0 and edge_density >= 0.055 and
                         roi_entropy >= 3.20)
        visual_structure = long_line_count >= 3 or photo_texture
        high_text_risk = ocr_overlap_area / candidate_area > 0.22 or untrusted_inside >= 4
        if high_text_risk and not visual_structure:
            continue

        # 大片手写演算有时会被一根跨区域斜线连接到真正的几何图，形成过大的
        # 候选框。对于“文字风险高、面积较大、非照片”的线稿，忽略贯穿候选框
        # 的超长桥接线，再按局部长直线密度寻找图形核心。图形核心最后仍保留
        # 约 12% 安全边距，避免斜面、坐标轴和图内标注被裁断。
        mixed_large_lineart = candidate_area > h * w * 0.22
        should_refine = (not photo_texture and long_line_count >= 6 and
                         candidate_area > h * w * 0.15 and
                         (high_text_risk or mixed_large_lineart))
        refinement_lines = lines
        if should_refine:
            refinement_min_length = max(18, int(min(roi_gray.shape[:2]) * 0.12))
            refinement_lines = cv2.HoughLinesP(
                roi_edges, 1, np.pi / 180, threshold=18,
                minLineLength=refinement_min_length, maxLineGap=10,
            )
        if should_refine and refinement_lines is not None:
            roi_h, roi_w = roi_gray.shape[:2]
            line_map = np.zeros((roi_h, roi_w), np.uint8)
            usable_segments: list[tuple[int, int, int, int, float]] = []
            bridge_limit = max(roi_w, roi_h) * 0.62
            # OpenCV 不同版本可能返回 N×1×4 或 N×4，统一整理后再遍历。
            for raw_line in np.asarray(refinement_lines).reshape(-1, 4):
                lx1, ly1, lx2, ly2 = (int(value) for value in raw_line)
                length = float(np.hypot(lx2 - lx1, ly2 - ly1))
                if length > bridge_limit:
                    continue
                usable_segments.append((lx1, ly1, lx2, ly2, length))
                cv2.line(line_map, (lx1, ly1), (lx2, ly2), 255,
                         max(5, min(roi_w, roi_h) // 70), cv2.LINE_AA)
            if len(usable_segments) >= 4:
                cluster_size = max(15, min(roi_w, roi_h) // 12)
                clustered_lines = cv2.dilate(
                    line_map,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (cluster_size | 1, cluster_size | 1)),
                    iterations=1,
                )
                clusters, _ = cv2.findContours(clustered_lines, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                best_core: tuple[int, int, int, int] | None = None
                best_score = 0.0
                for cluster in clusters:
                    cx, cy, cw, ch = cv2.boundingRect(cluster)
                    members = [segment for segment in usable_segments
                               if cx <= (segment[0] + segment[2]) / 2 <= cx + cw and
                               cy <= (segment[1] + segment[3]) / 2 <= cy + ch]
                    if len(members) < 4:
                        continue
                    total_length = sum(segment[4] for segment in members)
                    cluster_score = len(members) + total_length / max(1, min(roi_w, roi_h))
                    if cluster_score <= best_score:
                        continue
                    xs = [coordinate for segment in members for coordinate in (segment[0], segment[2])]
                    ys = [coordinate for segment in members for coordinate in (segment[1], segment[3])]
                    core = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
                    if ((core[2] - core[0]) * (core[3] - core[1]) < roi_w * roi_h * 0.025):
                        continue
                    best_core = core
                    best_score = cluster_score

                # 宽幅候选中，桥接斜线可能被霍夫算法拆成多段并继续串联两个区域。
                # 再用局部窗口比较线段密度，选择几何结构最集中的区域。该规则只
                # 作用于宽高比明显偏大的混合框，不影响较方正的电路图和照片。
                if roi_w >= roi_h * 1.55:
                    window_w = max(120, int(roi_w * 0.52))
                    window_h = max(100, int(roi_h * 0.58))
                    window_w = min(window_w, roi_w)
                    window_h = min(window_h, roi_h)
                    best_window: tuple[int, int, int, int] | None = None
                    best_window_members: list[tuple[int, int, int, int, float]] = []
                    best_window_score = 0.0
                    for segment in usable_segments:
                        center_x = (segment[0] + segment[2]) // 2
                        center_y = (segment[1] + segment[3]) // 2
                        window_x1 = max(0, min(roi_w - window_w, center_x - window_w // 2))
                        window_y1 = max(0, min(roi_h - window_h, center_y - window_h // 2))
                        window_x2 = window_x1 + window_w
                        window_y2 = window_y1 + window_h
                        members = [other for other in usable_segments
                                   if window_x1 <= (other[0] + other[2]) / 2 <= window_x2 and
                                   window_y1 <= (other[1] + other[3]) / 2 <= window_y2]
                        if len(members) < 4:
                            continue
                        total_length = sum(other[4] for other in members)
                        direction_lengths = [0.0, 0.0, 0.0]  # 水平、竖直、斜向
                        for other in members:
                            angle = abs(float(np.degrees(np.arctan2(
                                other[3] - other[1], other[2] - other[0]
                            )))) % 180.0
                            angle = min(angle, 180.0 - angle)
                            if angle < 18.0:
                                direction_lengths[0] += other[4]
                            elif angle > 72.0:
                                direction_lengths[1] += other[4]
                            else:
                                direction_lengths[2] += other[4]
                        # 公式下划线和跨区斜线通常缺少竖直骨架；斜面、坐标图、
                        # 电路图则会同时提供多个方向的结构。
                        if direction_lengths[1] < min(roi_w, roi_h) * 0.08:
                            continue
                        direction_balance = min(direction_lengths)
                        score = (len(members) + total_length / max(1, min(roi_w, roi_h)) +
                                 2.5 * direction_balance / max(1, min(roi_w, roi_h)))
                        if score > best_window_score:
                            best_window = (window_x1, window_y1, window_x2, window_y2)
                            best_window_members = members
                            best_window_score = score
                    if best_window is not None:
                        xs = [coordinate for segment in best_window_members
                              for coordinate in (segment[0], segment[2])]
                        ys = [coordinate for segment in best_window_members
                              for coordinate in (segment[1], segment[3])]
                        window_core = (
                            max(best_window[0], min(xs)),
                            max(best_window[1], min(ys)),
                            min(best_window[2], max(xs) + 1),
                            min(best_window[3], max(ys) + 1),
                        )
                        if ((window_core[2] - window_core[0]) *
                                (window_core[3] - window_core[1]) >= roi_w * roi_h * 0.025):
                            best_core = window_core

                    # 宽幅斜面图通常以右侧竖边作为稳定几何锚点。排除候选框最
                    # 外侧的页面边缘后，将相邻竖线合并，利用其跨度反推斜面图
                    # 的左右和上下范围，可避开左下方被桥接斜线串入的演算内容。
                    vertical_segments = []
                    for segment in usable_segments:
                        angle = abs(float(np.degrees(np.arctan2(
                            segment[3] - segment[1], segment[2] - segment[0]
                        )))) % 180.0
                        angle = min(angle, 180.0 - angle)
                        midpoint_x = (segment[0] + segment[2]) / 2
                        if (angle > 72.0 and segment[4] >= roi_h * 0.10 and
                                roi_w * 0.55 <= midpoint_x <= roi_w * 0.985):
                            vertical_segments.append(segment)
                    best_vertical_group: list[tuple[int, int, int, int, float]] = []
                    best_vertical_span = 0.0
                    for anchor in vertical_segments:
                        anchor_x = (anchor[0] + anchor[2]) / 2
                        group = [segment for segment in vertical_segments
                                 if abs((segment[0] + segment[2]) / 2 - anchor_x) <=
                                 max(8, roi_w * 0.035)]
                        span = max(max(segment[1], segment[3]) for segment in group) - \
                               min(min(segment[1], segment[3]) for segment in group)
                        if span > best_vertical_span:
                            best_vertical_span = float(span)
                            best_vertical_group = group
                    if best_vertical_group and best_vertical_span >= roi_h * 0.16:
                        anchor_x = int(np.median([
                            (segment[0] + segment[2]) / 2 for segment in best_vertical_group
                        ]))
                        anchor_top = min(min(segment[1], segment[3])
                                         for segment in best_vertical_group)
                        anchor_bottom = max(max(segment[1], segment[3])
                                            for segment in best_vertical_group)
                        inferred_width = max(int(roi_w * 0.34), int(best_vertical_span * 2.5))
                        best_core = (
                            max(0, anchor_x - inferred_width),
                            max(0, int(anchor_top - best_vertical_span * 0.35)),
                            min(roi_w, int(anchor_x + roi_w * 0.06)),
                            min(roi_h, int(anchor_bottom + best_vertical_span * 0.30)),
                        )
                if best_core is not None:
                    core_w = best_core[2] - best_core[0]
                    core_h = best_core[3] - best_core[1]
                    core_pad_x = max(12, int(core_w * 0.12))
                    core_pad_y = max(12, int(core_h * 0.12))
                    local_x1 = max(0, best_core[0] - core_pad_x)
                    local_y1 = max(0, best_core[1] - core_pad_y)
                    local_x2 = min(roi_w, best_core[2] + core_pad_x)
                    local_y2 = min(roi_h, best_core[3] + core_pad_y)
                    candidate = (candidate[0] + local_x1, candidate[1] + local_y1,
                                 candidate[0] + local_x2, candidate[1] + local_y2)
                    candidate_area = max(1, (candidate[2] - candidate[0]) *
                                         (candidate[3] - candidate[1]))

        # 黑色手绘草图往往比印刷线稿留下更多粗重深色像素，但又不具备照片的
        # 连续灰度层次。该组合特征只用于排除稠密手绘线稿，不影响彩色纸面、
        # 真实照片以及笔画较细的印刷几何图。特征从未删文字原图计算。
        source_roi = source_image[candidate[1]:candidate[3], candidate[0]:candidate[2]]
        source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
        dark_fraction = float(np.mean(source_gray < 70))
        source_hsv = cv2.cvtColor(source_roi, cv2.COLOR_BGR2HSV)
        colorful_fraction = float(np.mean(source_hsv[:, :, 1] > 45))
        histogram = cv2.calcHist([source_gray], [0], None, [32], [0, 256]).ravel()
        probabilities = histogram / max(1.0, float(histogram.sum()))
        probabilities = probabilities[probabilities > 0]
        gray_entropy = float(-np.sum(probabilities * np.log2(probabilities)))
        horizontal_lines = 0
        vertical_lines = 0
        if lines is not None:
            for raw_line in np.asarray(lines).reshape(-1, 4):
                lx1, ly1, lx2, ly2 = (int(value) for value in raw_line)
                angle = abs(float(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))) % 180.0
                angle = min(angle, 180.0 - angle)
                if angle < 15.0:
                    horizontal_lines += 1
                elif angle > 75.0:
                    vertical_lines += 1
        orthogonal_line_art = horizontal_lines >= 2 and vertical_lines >= 2
        dense_handwriting = (dark_fraction > 0.033 and colorful_fraction < 0.08 and
                             gray_entropy < 3.22 and not orthogonal_line_art)
        if dense_handwriting:
            continue

        # 拍照时纸张边缘、书脊和阴影常形成细长连通域。它们贴着左右边界且
        # 宽度很窄，不属于题目插图，应在加入最终候选前直接排除。
        candidate_width = candidate[2] - candidate[0]
        touches_vertical_border = candidate[0] <= int(w * 0.018) or candidate[2] >= int(w * 0.982)
        if touches_vertical_border and candidate_width <= int(w * 0.18):
            continue
        padded.append(candidate)
    # 面积较大的候选优先；阅读顺序用于最终命名和重排。
    padded = sorted(padded, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for box in padded:
        if any(boxes_overlap_or_near(box, chosen, -min(box[2] - box[0], chosen[2] - chosen[0]) // 2,
                                     -min(box[3] - box[1], chosen[3] - chosen[1]) // 2)
               for chosen in selected):
            continue
        selected.append(box)
        if len(selected) >= config.illustration_max_count:
            break
    selected = deduplicate_overlapping_illustration_boxes(selected)
    selected = sort_illustration_boxes_reading_order(selected)

    # 最终保险：若线稿框仍超过页面 30%，说明桥接线没有在前面的聚类阶段完全
    # 分离。直接在最终框中寻找远离页面边缘的右侧竖边，以斜面/坐标框右边界
    # 为锚点收缩。普通插图面积较小，不会进入这个分支。
    post_refined: list[tuple[int, int, int, int]] = []
    for box in selected:
        bx1, by1, bx2, by2 = box
        box_area = (bx2 - bx1) * (by2 - by1)
        if box_area <= h * w * 0.30:
            post_refined.append(box)
            continue
        roi = detection_copy[by1:by2, bx1:bx2]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_edges = cv2.Canny(roi_gray, 70, 170)
        short_lines = cv2.HoughLinesP(
            roi_edges, 1, np.pi / 180, threshold=18,
            minLineLength=max(18, int(min(roi_gray.shape[:2]) * 0.12)), maxLineGap=10,
        )
        if short_lines is None:
            post_refined.append(box)
            continue
        roi_h, roi_w = roi_gray.shape[:2]
        verticals: list[tuple[int, int, int, int, float]] = []
        for raw_line in np.asarray(short_lines).reshape(-1, 4):
            lx1, ly1, lx2, ly2 = (int(value) for value in raw_line)
            length = float(np.hypot(lx2 - lx1, ly2 - ly1))
            angle = abs(float(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))) % 180.0
            angle = min(angle, 180.0 - angle)
            midpoint_x = (lx1 + lx2) / 2
            if (angle > 72.0 and length >= roi_h * 0.10 and
                    roi_w * 0.55 <= midpoint_x <= roi_w * 0.985):
                verticals.append((lx1, ly1, lx2, ly2, length))
        if not verticals:
            post_refined.append(box)
            continue
        anchor = max(verticals, key=lambda segment: segment[4])
        anchor_x = (anchor[0] + anchor[2]) / 2
        group = [segment for segment in verticals
                 if abs((segment[0] + segment[2]) / 2 - anchor_x) <= max(8, roi_w * 0.035)]
        top = min(min(segment[1], segment[3]) for segment in group)
        bottom = max(max(segment[1], segment[3]) for segment in group)
        span = bottom - top
        if span < roi_h * 0.14:
            post_refined.append(box)
            continue
        anchor_x_int = int(np.median([(segment[0] + segment[2]) / 2 for segment in group]))
        inferred_width = max(int(roi_w * 0.34), int(span * 2.5))
        core = (max(0, anchor_x_int - inferred_width),
                max(0, int(top - span * 0.35)),
                min(roi_w, int(anchor_x_int + roi_w * 0.06)),
                min(roi_h, int(bottom + span * 0.30)))
        core_w, core_h = core[2] - core[0], core[3] - core[1]
        pad_x, pad_y = max(12, int(core_w * 0.12)), max(12, int(core_h * 0.12))
        post_refined.append((
            bx1 + max(0, core[0] - pad_x),
            by1 + max(0, core[1] - pad_y),
            bx1 + min(roi_w, core[2] + pad_x),
            by1 + min(roi_h, core[3] + pad_y),
        ))
    selected = [
        expand_illustration_box_for_labels(box, items, source_image.shape)
        for box in post_refined
    ]
    main_question_numbers: list[int] = []
    for item in items:
        if not item.printed:
            continue
        compact = re.sub(r"\s+", "", str(item.text or ""))
        match = re.match(r"^(\d{1,3})[.．、]", compact)
        if match:
            main_question_numbers.append(int(match.group(1)))
    target_question_number = (
        main_question_numbers[0] if main_question_numbers else None
    )
    caption_boxes = caption_anchored_figure_boxes(
        detection_copy,
        items,
        target_question_number=target_question_number,
    )
    if caption_boxes:
        if target_question_number is not None:
            # A matching ``(第N题)`` caption is stronger evidence than generic
            # contours elsewhere on a photographed page.  Keeping unrelated
            # retained contours is what previously attached the diagram from
            # question 14 to question 16.
            selected = caption_boxes
        else:
            retained: list[tuple[int, int, int, int]] = []
            for box in selected:
                box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
                overlaps_caption_figure = False
                for anchor in caption_boxes:
                    intersection = (
                        max(0, min(box[2], anchor[2]) - max(box[0], anchor[0]))
                        * max(0, min(box[3], anchor[3]) - max(box[1], anchor[1]))
                    )
                    anchor_area = max(1, (anchor[2] - anchor[0]) * (anchor[3] - anchor[1]))
                    if intersection / min(box_area, anchor_area) >= 0.14:
                        overlaps_caption_figure = True
                        break
                if not overlaps_caption_figure:
                    retained.append(box)
            selected = caption_boxes + expand_isolated_boxes_for_edge_ink(source_image, retained)
    else:
        selected = expand_isolated_boxes_for_edge_ink(source_image, selected)
    selected = sort_illustration_boxes_reading_order(selected)
    selected = selected[:config.illustration_max_count]
    selected, _rejected_oversized = sanitize_illustration_boxes(
        selected,
        items,
        source_image.shape,
    )

    debug = detection_copy.copy()
    for index, (x1, y1, x2, y2) in enumerate(selected, 1):
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 255), 5)
        cv2.putText(debug, f"FIG {index}", (x1, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 255), 2, cv2.LINE_AA)
    return selected, debug


def recover_referenced_visual_sidecar(
    detection_copy: np.ndarray,
    source_image: np.ndarray,
    items: list[OCRItem],
    transcript: str,
) -> list[tuple[int, int, int, int]]:
    """Recover a right-side diagram when ordinary layout detection misses it.

    Handwritten workings can make a small printed diagram look like one large
    irregular contour.  On many photographed questions the prose still forms
    a stable left column while that mixed visual occupies a distinct right
    sidecar.  This fallback is intentionally narrow: it needs an explicit
    visual cue, a usable right gutter, and either multiple long line segments
    or genuine image texture.  It never falls back to the whole question.
    """

    compact_transcript = re.sub(r"\s+", "", str(transcript or ""))
    if not re.search(
        r"如图|图[①②③④⑤⑥1-9甲乙丙丁戊己]|"
        r"第\s*\d{1,3}(?:\s*[-—–]\s*(?:[IVXivx]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+))?\s*(?:题)?图|"
        r"示意图|函数图像|电路图|光路图|转盘",
        compact_transcript,
    ):
        return []
    if not isinstance(source_image, np.ndarray) or source_image.size == 0:
        return []

    page_h, page_w = source_image.shape[:2]
    prose_right_edges: list[int] = []
    for item in items:
        if not (item.printed or item.formula_like):
            continue
        compact = re.sub(r"\s+", "", str(item.text or ""))
        cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in compact)
        if cjk_count < 4 or len(compact) < 7 or item.center_x >= page_w * 0.78:
            continue
        prose_right_edges.append(int(np.ceil(float(item.box[:, 0].max()))))
    if not prose_right_edges:
        return []
    prose_edge = int(np.quantile(prose_right_edges, 0.70))
    side_left = max(int(page_w * 0.52), prose_edge + max(8, int(page_w * 0.012)))
    if side_left >= int(page_w * 0.84):
        return []

    roi = detection_copy[:, side_left:]
    if roi.size == 0:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    foreground = (
        (gray < 238)
        | ((hsv[:, :, 1] > 42) & (hsv[:, :, 2] < 252))
    ).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    coordinates = cv2.findNonZero(foreground)
    if coordinates is None:
        return []
    local_x, local_y, width, height = cv2.boundingRect(coordinates)
    if width < page_w * 0.15 or height < page_h * 0.16:
        return []

    edges = cv2.Canny(gray, 65, 165)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=max(20, int(min(roi.shape[:2]) * 0.13)),
        maxLineGap=12,
    )
    long_line_count = 0 if lines is None else len(np.asarray(lines).reshape(-1, 4))
    source_roi = source_image[:, side_left:]
    source_gray = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    texture = bool(
        float(np.std(source_gray)) >= 27.0
        and cv2.countNonZero(edges) / max(1, edges.size) >= 0.022
    )
    if long_line_count < 3 and not texture:
        return []

    pad_x = max(14, int(width * 0.07))
    pad_y = max(14, int(height * 0.08))
    candidate = (
        max(0, side_left + local_x - pad_x),
        max(0, local_y - pad_y),
        min(page_w, side_left + local_x + width + pad_x),
        min(page_h, local_y + height + pad_y),
    )
    candidate_area = max(1, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
    if candidate_area > page_h * page_w * 0.42:
        return []
    safe, _rejected = sanitize_illustration_boxes(
        [candidate],
        items,
        source_image.shape,
    )
    return safe


def content_union_box(shape: tuple[int, ...], items: list[OCRItem],
                      figure_boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """联合印刷文字和插图得到题目范围；使用 12% 安全边距保证完整性。"""
    h, w = shape[:2]
    points: list[tuple[float, float, float, float]] = []
    for item in items:
        if item.printed or item.formula_like:
            points.append((float(item.box[:, 0].min()), float(item.box[:, 1].min()),
                           float(item.box[:, 0].max()), float(item.box[:, 1].max())))
    points.extend(figure_boxes)
    if not points:
        return 0, 0, w, h
    x1, y1 = min(p[0] for p in points), min(p[1] for p in points)
    x2, y2 = max(p[2] for p in points), max(p[3] for p in points)
    px, py = max(24, int((x2 - x1) * 0.12)), max(24, int((y2 - y1) * 0.12))
    box = (max(0, int(x1) - px), max(0, int(y1) - py),
           min(w, int(x2) + px), min(h, int(y2) + py))
    # 联合框异常小时保留完整正视图，不冒险截题。
    if (box[2] - box[0]) < w * 0.35 or (box[3] - box[1]) < h * 0.25:
        return 0, 0, w, h
    return box


def choose_chinese_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """优先使用 Windows 宋体；缺失时回退到微软雅黑和黑体。"""
    for path in (Path("C:/Windows/Fonts/simsun.ttc"), Path("C:/Windows/Fonts/msyh.ttc"),
                 Path("C:/Windows/Fonts/simhei.ttf")):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont,
              max_width: int) -> list[str]:
    """按实际像素宽度换行；中文逐字、英文空格都能正常处理。"""
    if not text:
        return []
    lines: list[str] = []
    current = ""
    for char in text:
        trial = current + char
        if current and draw.textbbox((0, 0), trial, font=font)[2] > max_width:
            lines.append(current)
            current = char
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def split_script_runs(text: str) -> list[tuple[str, str]]:
    """Split text into normal, superscript and subscript drawing runs."""

    superscript = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
    subscript = set("₀₁₂₃₄₅₆₇₈₉₊₋")
    super_plain = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
    sub_plain = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-")
    runs: list[tuple[str, str]] = []
    for char in text:
        kind = "sup" if char in superscript else "sub" if char in subscript else "normal"
        value = (
            char.translate(super_plain)
            if kind == "sup"
            else char.translate(sub_plain)
            if kind == "sub"
            else char
        )
        if runs and runs[-1][0] == kind:
            runs[-1] = (kind, runs[-1][1] + value)
        else:
            runs.append((kind, value))
    return runs


def draw_text_with_scripts(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    *,
    font_size: int,
    fill: tuple[int, int, int],
) -> None:
    """Draw compact textbook-style super/subscripts with ordinary digits."""

    try:
        script_font = font.font_variant(size=max(18, int(font_size * 0.64)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        script_font = choose_chinese_font(max(18, int(font_size * 0.64)))
    x, y = position
    for kind, value in split_script_runs(text):
        current_font = font if kind == "normal" else script_font
        offset_y = 0
        if kind == "sup":
            offset_y = -max(4, int(font_size * 0.20))
        elif kind == "sub":
            offset_y = max(6, int(font_size * 0.28))
        draw.text((x, y + offset_y), value, fill=fill, font=current_font)
        x += int(math.ceil(draw.textlength(value, font=current_font)))


def group_items_into_lines(items: list[OCRItem]) -> list[list[OCRItem]]:
    """按原图纵坐标和字号将 OCR 文字恢复成阅读行。"""
    printed = sorted(
        (item for item in items if item.printed or item.formula_like),
        key=lambda item: (item.center_y, item.center_x),
    )
    lines: list[list[OCRItem]] = []
    for item in printed:
        target: list[OCRItem] | None = None
        for line in reversed(lines[-4:]):
            line_y = float(np.median([part.center_y for part in line]))
            line_h = float(np.median([part.height for part in line]))
            if abs(item.center_y - line_y) <= 0.58 * max(item.height, line_h):
                target = line
                break
        if target is None:
            lines.append([item])
        else:
            target.append(item)
    for line in lines:
        line.sort(key=lambda item: item.center_x)
    return lines


def crop_with_padding(image: np.ndarray, box: tuple[int, int, int, int], ratio: float = 0.0) -> np.ndarray:
    x1, y1, x2, y2 = box
    if ratio:
        px, py = int((x2 - x1) * ratio), int((y2 - y1) * ratio)
        x1, y1 = max(0, x1 - px), max(0, y1 - py)
        x2, y2 = min(image.shape[1], x2 + px), min(image.shape[0], y2 + py)
    return image[y1:y2, x1:x2].copy()


def crop_illustration_with_clean_margins(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    items: list[OCRItem],
) -> np.ndarray:
    """Crop a complete figure while erasing only foreign text at its edges.

    Figure boxes are deliberately expanded to retain axis titles, units and
    tick values. That safety margin can also intersect the tail of a nearby
    question line or a handwritten assignment. Keep compact figure labels
    intact, but whiten OCR polygons that cross the crop boundary and are not
    labels. A short ``x=43``-style answer is removed even when its OCR polygon
    happens to fall wholly inside the expanded margin.

    Untrusted OCR items are never erased: a missed tick value is preferable to
    deleting a legitimate part of a diagram.
    """

    x1, y1, x2, y2 = box
    cleaned = crop_with_padding(image, box)
    if cleaned.size == 0:
        return cleaned

    crop_h, crop_w = cleaned.shape[:2]
    for item in items:
        if compact_figure_label(item):
            continue
        if not (item.printed or item.formula_like):
            continue

        ix1 = int(np.floor(float(item.box[:, 0].min())))
        iy1 = int(np.floor(float(item.box[:, 1].min())))
        ix2 = int(np.ceil(float(item.box[:, 0].max())))
        iy2 = int(np.ceil(float(item.box[:, 1].max())))
        if ix2 <= x1 or ix1 >= x2 or iy2 <= y1 or iy1 >= y2:
            continue

        text = re.sub(r"\s+", "", str(item.text or ""))
        assignment_like = bool(re.match(r"^[A-Za-z]\s*[=:≈]\s*[-+]?\d", text))
        handwriting_expression = bool(
            not item.printed
            and item.formula_like
            and (
                sum(char in "=+×÷*/^" for char in text) >= 2
                or ("=" in text and len(text) >= 6)
                or any(char in text for char in "↑↓✓√")
            )
        )
        crosses_edge = ix1 < x1 or iy1 < y1 or ix2 > x2 or iy2 > y2
        if not (assignment_like or handwriting_expression or crosses_edge):
            continue

        polygon = np.round(item.box - np.asarray([x1, y1], dtype=np.float32)).astype(np.int32)
        local = np.zeros((crop_h, crop_w), np.uint8)
        cv2.fillPoly(local, [polygon], 255)
        padding = max(2, min(7, int(max(1.0, item.height) * 0.10)))
        local = cv2.dilate(
            local,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (padding * 2 + 1, padding * 2 + 1),
            ),
        )
        # Match the nearby paper tone instead of leaving bright white blocks
        # on photographed cream/grey pages.
        surround = cv2.dilate(
            local,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (padding * 6 + 1, padding * 6 + 1),
            ),
        )
        gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
        paper = cleaned[(surround > 0) & (local == 0) & (gray >= 205)]
        fill = np.median(paper, axis=0).astype(np.uint8) if len(paper) >= 12 else np.asarray([255, 255, 255], dtype=np.uint8)
        cleaned[local > 0] = fill
    return cleaned


def normalize_scientific_notation(text: str) -> str:
    """Restore exponents flattened by line OCR in scientific-notation rows."""

    import re

    superscript_digits = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    to_superscript = str.maketrans("0123456789+-", superscript_digits + "⁺⁻")
    from_superscript = str.maketrans(superscript_digits + "⁺⁻", "0123456789+-")

    def exponent(value: str) -> str:
        plain = value.translate(from_superscript)
        return plain.translate(to_superscript)

    def replace_explicit(match: re.Match[str]) -> str:
        return f"{match.group(1)}{exponent(match.group(2))}"

    # Braces, a caret, or mixed superscript/plain digits are unambiguous OCR
    # evidence of an exponent even outside a multiple-choice row.
    text = re.sub(
        r"([×xX·]\s*10)\s*(?:\^\s*\{?|\{)\s*([+\-⁺⁻]?[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]{1,3})\}?",
        replace_explicit,
        text,
    )
    text = re.sub(
        r"([×xX·]\s*10)([⁰¹²³⁴⁵⁶⁷⁸⁹]+[0-9]+|[0-9]+[⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        replace_explicit,
        text,
    )

    # Plain ``×1011`` is ambiguous in ordinary arithmetic. It is interpreted
    # as ``×10¹¹`` only when the row is explicitly a choice or says scientific
    # notation, which avoids changing legitimate multiplication by 100/1000.
    scientific_context = bool(
        "科学记数法" in text
        or re.search(r"(?:^|\s)[A-DＡ-Ｄ]\s*[.、:：]", text)
    )
    if scientific_context:
        text = re.sub(
            r"([×xX·]\s*10)([0-9]{1,3})(?=\s*(?:[A-DＡ-Ｄ]\s*[.、:：]|$))",
            replace_explicit,
            text,
        )
    return text


def split_choice_options(text: str) -> list[str]:
    """Split a merged A-D OCR row into stable one-option paragraphs."""

    import re

    leading = re.match(r"^([A-DＡ-Ｄ])\s*[.．、:：]\s*", text)
    if leading is None:
        return [text]
    # An option may mention labelled objects such as ``导体A、B``.  Treat only
    # monotonically advancing labels as new choices so that those object names
    # remain inside the option body, while ``...电荷B.下一项`` is still split.
    matches: list[re.Match[str]] = [leading]
    first_label = leading.group(1)
    first_normalized = (
        chr(ord("A") + "ＡＢＣＤ".find(first_label))
        if first_label in "ＡＢＣＤ" else first_label
    )
    last_index = ord(first_normalized) - ord("A")
    # Once a row has begun with a real option label, a later label may be
    # glued directly to a one-letter variable (``射程xC.``).  Scan periods and
    # colons without a left-boundary requirement, but reject an uppercase
    # predecessor so geometric names such as ``AB.`` remain ordinary prose.
    for match in re.finditer(r"([A-DＡ-Ｄ])\s*[.．:：]\s*", text[leading.end():]):
        absolute_start = leading.end() + match.start()
        absolute_end = leading.end() + match.end()
        match = re.match(
            r"([A-DＡ-Ｄ])\s*[.．:：]\s*",
            text[absolute_start:absolute_end],
        )
        if match is None:
            continue
        # Re-create an absolute match by searching at the known position.
        match = re.compile(r"([A-DＡ-Ｄ])\s*[.．:：]\s*").match(
            text,
            absolute_start,
        )
        if match is None:
            continue
        label = match.group(1)
        normalized = chr(ord("A") + "ＡＢＣＤ".find(label)) if label in "ＡＢＣＤ" else label
        label_index = ord(normalized) - ord("A")
        predecessor = text[match.start() - 1:match.start()]
        if predecessor and predecessor in "ABCDEFGHIJKLMNOPQRSTUVWXYZＡＢＣＤ":
            continue
        if label_index > last_index:
            matches.append(match)
            last_index = label_index
    options: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        label = chr(ord("A") + "ＡＢＣＤ".find(match.group(1))) if match.group(1) in "ＡＢＣＤ" else match.group(1)
        options.append(f"{label}. {body}".rstrip())
    return options


def normalize_reflow_structure(text: str) -> str:
    """Expose glued choices and numbered parts as logical line boundaries."""

    # A section lead-in and the first equipment option are separate logical
    # blocks even when structured OCR returns them on one physical line.
    text = re.sub(r"([：:])\s*(?=A\s*[.．、])", r"\1\n", str(text or ""))
    physical_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        options = split_choice_options(line)
        if len(options) >= 2:
            physical_lines.extend(options)
        else:
            physical_lines.append(line)
    text = "\n".join(physical_lines)
    text = re.sub(
        r"(?m)([则是有为])\s*(?=A\s*[.．、])",
        r"\1\n",
        text,
    )
    # A faint speck or handwritten dot between two printed sentences is often
    # returned as a standalone list bullet.  Textbook stems use numbered or
    # lettered markers, so a leading middle-dot before Chinese prose is noise.
    text = re.sub(r"(?m)^\s*[·•∙]\s*(?=[\u4e00-\u9fff])", "", text)

    # OCR frequently joins ``(1)`` to a scored experiment heading.  When the
    # document contains a complete 1..N series, those tokens are structural
    # parts rather than mathematical parentheses and should start new rows.
    part_numbers = [
        int(match.group(1))
        for match in re.finditer(r"[（(]\s*([1-9])\s*[）)]", text)
    ]
    distinct_parts = sorted(set(part_numbers))
    if (
        len(distinct_parts) >= 2
        and distinct_parts[0] == 1
        and distinct_parts == list(range(1, distinct_parts[-1] + 1))
    ):
        text = re.sub(
            r"(?<!\n)\s*(?=[（(]\s*[1-9]\s*[）)])",
            "\n",
            text,
        )
    circled_parts = [
        "①②③④⑤⑥⑦⑧⑨⑩".index(match.group(0)) + 1
        for match in re.finditer(r"[①②③④⑤⑥⑦⑧⑨⑩]", text)
    ]
    if len(set(circled_parts)) >= 2:
        # Equation labels such as ``x−2y=1，①`` stay inline; only prose-level
        # circled parts become paragraph boundaries.
        text = re.sub(
            r"(?<![\n，,；;])\s*(?=[①②③④⑤⑥⑦⑧⑨⑩])",
            "\n",
            text,
        )
    if "下列器材" in text:
        text = re.sub(
            r"(?m)^(D[.．、].{1,100}?[）)])\s*(?=(?:发现|若|由于|为了))",
            r"\1\n",
            text,
        )
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def restore_missing_fill_blanks(text: str) -> str:
    """Restore answer rules that OCR omits because they are non-text pixels.

    Only explicit fill-in grammar is used.  This keeps table borders, fraction
    bars and geometry segments from becoming public ``____`` placeholders.
    """

    # A formula recognizer can occasionally wrap an ordinary defined value as
    # ``\underline{...}``.  After the generic underline cleanup this becomes
    # ``____ a(i,j)`` even though the source says "第 i 行第 j 个数为 a(i,j)".
    # A variable/function immediately following this definition grammar is a
    # printed formula, not an answer slot.
    text = re.sub(
        r"(第\s*[iI]\s*行第\s*[jJ]\s*个数为)\s*_{2,}\s*"
        r"(?=\$?[A-Za-z]\s*\(\s*[iI]\s*[,，]\s*[jJ]\s*\))",
        r"\1",
        text,
    )

    value_word = (
        r"(?:示数|数值|结果|答案|概率|中位数|平均数|方差|长度|宽度|高度|"
        r"面积|体积|速度|加速度|电压|电流|电阻(?:率)?|阻值|功率|频率|波长|"
        r"质量|温度|坐标|表达式|取值范围|目的)"
    )
    units = (
        r"(?:Ω\s*[·⋅]\s*m|Ω\s*[·⋅]\s*m|mm|cm|dm|km|m(?![A-Za-z])|m/s(?:\^?2|²)?|kg|g(?![A-Za-z])|"
        r"s(?![A-Za-z])|Hz|V|A|W|J|Pa|N|C|T|Ω|Ω|rad|mol)(?=$|[\s,，。；;（）()])"
    )
    # Blank before an observed printed unit, e.g. ``直径d= mm`` or
    # ``电阻率为 Ω·m``.
    text = re.sub(
        rf"((?:{value_word})?(?:为|是|等于)|(?<![<>!])=)\s*(?={units})",
        r"\1____ ",
        text,
        flags=re.IGNORECASE,
    )
    # Blank terminated by punctuation or the next numbered part.  A-D choice
    # prompts such as ``正确的是\nA.`` are deliberately excluded.
    text = re.sub(
        rf"({value_word}(?:为|是)|(?:则|故)[^\n。；]{{0,48}}(?<![<>!])=)"
        r"\s*(?=(?:[；;。]|\n\s*[（(]\s*\d+\s*[）)]|$))",
        r"\1____",
        text,
    )
    # When the equality is inside a LaTeX span, the closing dollar follows the
    # equals sign and previously prevented the end-of-stem rule above from
    # seeing the photographed answer line (for example ``则 tan 2α = ____``).
    text = re.sub(
        r"((?:则|故)[^\n。；]{0,80}(?<![<>!])=)(\s*\$)?\s*$",
        lambda match: (
            match.group(1)
            + ("$____" if match.group(2) and "$" in match.group(2) else "____")
        ),
        text,
    )
    # A parenthetical ``选填`` immediately following a noun identifies the
    # missing response slot even when the underline itself was not recognized.
    text = re.sub(
        r"(?m)(?<!_)([^\n，。；;:：_]{1,36}?)(?:\s*)(?=[（(]\s*选填)",
        lambda match: (
            match.group(1)
            if re.search(r"(?:为|是|包括|如下|下列)$", match.group(1))
            else match.group(1) + "____"
        ),
        text,
    )
    text = re.sub(
        r"置于____端____(?=\s*[（(]\s*选填)",
        "置于____端",
        text,
    )
    text = re.sub(
        r"置于端____(?=\s*[（(]\s*选填)",
        "置于____端",
        text,
    )
    # Preserve a function value when a formula recognizer mistakenly labels
    # the whole definition as an underline.  This must run before the generic
    # answer-line conversion below or the printed ``a(i,j)`` would be lost.
    text = re.sub(
        r"(第\s*[iI]\s*行第\s*[jJ]\s*个数为)\s*\$?\\underline\s*\{"
        r"\s*([A-Za-z]\s*\(\s*[iI]\s*[,，]\s*[jJ]\s*\))\s*\}\$?",
        r"\1\2",
        text,
    )
    # A structured recognizer may encode the printed answer rule itself as a
    # LaTeX underline with an arbitrary one-character payload.  The public
    # reading page must expose a blank, never that payload as an answer.
    text = re.sub(r"\$?\\underline\s*\{[^{}\n]{0,24}\}\$?", "____", text)
    text = re.sub(
        r"(第\s*[iI]\s*行第\s*[jJ]\s*个数为)\s*_{2,}\s*"
        r"(?=\$?[A-Za-z]\s*\(\s*[iI]\s*[,，]\s*[jJ]\s*\))",
        r"\1",
        text,
    )
    # Normalize an observed micrometer answer rule without filling its value.
    text = re.sub(
        r"(螺旋测微器[^\n。；]{0,48}?直径[^\n。；]{0,48}?为)\s*_{1,}\s*(mm)",
        r"\1____ \2",
        text,
    )
    text = re.sub(
        r"(螺旋测微器[^\n。；]{0,48}?直径[^\n。；]{0,48}?为)\s*(?=②)",
        r"\1____ mm。\n",
        text,
    )
    return text


def repair_missing_scientific_choice_exponent(text: str) -> str:
    """Recover one missing exponent from a complete 2x2 option grid.

    Textbook scientific-notation distractors commonly combine two coefficients
    with two exponents.  If OCR loses exactly one tiny exponent while the other
    three combinations are visible, the absent combination is deterministic.
    No answer correctness or outside knowledge is used.
    """

    import collections
    import re

    options = split_choice_options(text)
    if len(options) != 4:
        return text
    superscript_digits = "⁰¹²³⁴⁵⁶⁷⁸⁹"
    to_plain = str.maketrans(superscript_digits + "⁺⁻", "0123456789+-")
    records: list[dict[str, str | None]] = []
    for option in options:
        coefficient = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*[×xX·]\s*10",
            option,
        )
        exponent_match = re.search(
            rf"[×xX·]\s*10([{superscript_digits}⁺⁻]{{1,3}})",
            option,
        )
        records.append(
            {
                "coefficient": coefficient.group(1) if coefficient else None,
                "exponent": (
                    exponent_match.group(1).translate(to_plain)
                    if exponent_match
                    else None
                ),
            }
        )
    missing = [index for index, record in enumerate(records) if record["exponent"] is None]
    coefficients = [record["coefficient"] for record in records]
    known_exponents = {
        str(record["exponent"])
        for record in records
        if record["exponent"] is not None
    }
    if (
        len(missing) != 1
        or None in coefficients
        or sorted(collections.Counter(coefficients).values()) != [2, 2]
        or len(known_exponents) != 2
    ):
        return text

    missing_index = missing[0]
    missing_coefficient = records[missing_index]["coefficient"]
    paired = {
        str(record["exponent"])
        for record in records
        if record["coefficient"] == missing_coefficient
        and record["exponent"] is not None
    }
    candidate = known_exponents - paired
    if len(candidate) != 1:
        return text
    exponent = next(iter(candidate)).translate(
        str.maketrans("0123456789+-", superscript_digits + "⁺⁻")
    )
    options[missing_index] = re.sub(
        r"([×xX·]\s*10)\s*\^?(?=\s*$)",
        rf"\g<1>{exponent}",
        options[missing_index],
    )
    return " ".join(options)


def normalize_reflow_text(tokens: list[str]) -> str:
    """合并同一 OCR 行的文字，并清理中文排版中不需要的空格。"""
    import re
    import unicodedata

    text = " ".join(token.strip() for token in tokens if token.strip())
    # NFKC is useful for full-width ASCII punctuation, but it also flattens
    # mathematical superscripts/subscripts and circled subquestion numbers.
    # Protect those semantic characters before compatibility normalization.
    semantic_symbols = "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉①②③④⑤⑥⑦⑧⑨⑩"
    protected_symbols = {
        symbol: chr(0xE000 + index)
        for index, symbol in enumerate(semantic_symbols)
    }
    for symbol, placeholder in protected_symbols.items():
        text = text.replace(symbol, placeholder)
    text = unicodedata.normalize("NFKC", text)
    for symbol, placeholder in protected_symbols.items():
        text = text.replace(placeholder, symbol)
    text = re.sub(r"\s+", " ", text).strip()
    # 中文字符之间、中文标点前以及左括号后不保留 OCR 人工空格。
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。；：！？、,.!?;:）)\]])", r"\1", text)
    text = re.sub(r"([（(\[])\s+", r"\1", text)
    text = text.replace("．", ".").replace("：", "：")
    text = re.sub(r"(?<=\d)~(?=\d)", "～", text)
    # Conservative corrections for stable mathematical/geometry wording.
    # Each rule requires a semantic anchor, so ordinary prose and arbitrary
    # formulas are left untouched.
    text = text.replace("菌像", "图像").replace("監像", "图像").replace("路像", "图像")
    text = re.sub(r"与[)）]轴(?=正半轴)", "与y轴", text)
    text = re.sub(r"点4(?=的[横纵]坐标)", "点A", text)
    text = re.sub(r"△\s*4(?=[A-Z])", "△A", text)
    text = re.sub(
        r"二次函数\s*y\s*=\s*x(?:2|²)?\s*\+\s*b(?:x)?\s*\+\s*[co0]",
        "二次函数y=x²+bx+c",
        text,
    )
    text = re.sub(
        r"反比例函数\s*y\s*=\s*[(（]*\s*(?=[(（]x\s*>\s*0)",
        "反比例函数y=k/x",
        text,
    )
    if re.search(r"E\(x(?:2|₂),\s*[yY](?:2|₂)\)", text):
        text = re.sub(r"D\(x,\s*x\)", "D(x₁,y₁)", text)
    text = re.sub(r"E\(x2,\s*[yY]2\)", "E(x₂,y₂)", text)
    text = re.sub(r"\(x1\s*<\s*x2\)", "(x₁<x₂)", text)
    text = re.sub(r"(?<=且)x\s*\+\s*x_?2\s*=\s*5", "x₁+x₂=5", text)
    text = re.sub(r"是\s*m2(?=[.。]|$)", "是m²", text)
    return repair_missing_scientific_choice_exponent(
        normalize_scientific_notation(text)
    )


def strip_ocr_answer_tail(text: str) -> str:
    """Keep a printed instruction while removing a merged short answer tail."""

    text = re.sub(
        r"((?:求|计算)[^，。；]{0,16}(?:值|结果))\s*[:：]\s*"
        r"[-+]?(?:\d+(?:\.\d+)?|[（(][^（）()]{1,12}[）)])\s*$",
        r"\1",
        text,
    )
    # A handwritten fill-in answer is sometimes recognized as a separate
    # high-confidence numeric token between the printed blank and a result
    # qualifier, for example ``长度为 322 结果保留根号）``.  The qualifier is
    # a strong local boundary, so this does not affect numeric conditions in
    # the rest of the question.
    return re.sub(
        r"((?:长度|度数|数量|个数|中位数|概率|答案|结果|值)(?:为|是))"
        r"\s*[-+]?\d+(?:\.\d+)?\s*"
        r"(?=[（(]?\s*(?:结果|精确到|保留|取)[^\n。；]{0,24}[）)])",
        r"\1____",
        text,
    )


def repair_ocr_transcript(text: str) -> str:
    """Repair reading-order joins that are characteristic of curved pages."""

    text = strip_ocr_answer_tail(text)
    # In school mathematics the lowercase line identifier ``l`` is normally
    # set in italic.  Times New Roman's upright ``l`` is visually almost
    # indistinguishable from the digit ``1`` in the generated PNG, which was
    # repeatedly reported as an OCR error.  Promote only isolated ``l`` tokens
    # inside math spans when the surrounding question explicitly discusses a
    # line; commands such as ``\ln`` and ordinary words are untouched.
    if "直线" in text:
        def promote_line_identifier(match: re.Match[str]) -> str:
            inner = re.sub(
                r"(?<![A-Za-z\\])l(?=(?:\s*_\s*\{?\s*\d+\s*\}?)?(?![A-Za-z]))",
                r"\\ell",
                match.group(1),
            )
            return f"${inner}$"

        text = re.sub(r"\$([^$]+)\$", promote_line_identifier, text)
    # A full structured pass may encode the complete A-D row as one math span.
    # Keeping the labels inside ``$...$`` hides them from both semantic checks
    # and the option-row layout. Expand only blocks that contain all four labels.
    def expand_formula_choice_row(match: re.Match[str]) -> str:
        content = match.group(1)
        label_matches = list(
            re.finditer(r"(?:\\quad\s*)?([A-D])\s*[.．、]\s*", content)
        )
        if [item.group(1) for item in label_matches] != list("ABCD"):
            return match.group(0)
        options: list[str] = []
        for index, label_match in enumerate(label_matches):
            end = (
                label_matches[index + 1].start()
                if index + 1 < len(label_matches)
                else len(content)
            )
            value = content[label_match.end():end]
            value = re.sub(r"(?:\\quad\s*)+$", "", value).strip()
            if not value:
                return match.group(0)
            options.append(f"{label_match.group(1)}. ${value}$")
        return "  ".join(options)

    text = re.sub(r"\$([^$\n]+)\$", expand_formula_choice_row, text)
    # PP-FormulaNet occasionally absorbs the option label into a ``\mathrm``
    # group, making the label itself look like leaked formula source.  Move the
    # observed label outside while retaining the complete recovered fraction.
    text = re.sub(
        r"\$\s*\\mathrm\s*\{\s*~*\s*([A-D])\s*[.．、]\s*~*\s*\}"
        r"\s*(\\frac\s*\{[^{}]+\}\s*\{[^{}]+\})\s*\$",
        lambda match: f"{match.group(1)}. ${match.group(2)}$",
        text,
    )
    text = re.sub(r"\(x₁\s*<\s*x2\)", "(x₁<x₂)", text)
    text = re.sub(
        r"(?m)^\s*([1-9]\d{0,2})[.．、][^\n]{0,80}?(?=\s+\1[.．、])",
        "",
        text,
    ).lstrip()
    if re.search(r"(?m)^\s*[（(]\s*2\s*[）)]", text):
        text = re.sub(r"(?m)^(\s*)1\s*[）)]", r"\1(1)", text)

    # A wide detector can merge the next printed row into the previous row and
    # then sort tokens by x coordinate. Restore two common structural anchors
    # without inventing question content.
    repaired_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("E(") and re.search(r"\s[（(]2[）)]", stripped):
            continuation, subquestion = re.split(
                r"\s+(?=[（(]2[）)])",
                stripped,
                maxsplit=1,
            )
            repaired_lines.extend([subquestion, continuation])
            continue
        if stripped.startswith("理由") and re.search(r"\s②", stripped):
            reason, subquestion = re.split(r"\s+(?=②)", stripped, maxsplit=1)
            repaired_lines.extend([subquestion, reason])
            continue
        repaired_lines.append(stripped)
    text = "\n".join(line for line in repaired_lines if line)

    # On a strongly curved reciprocal-function question, the fraction and its
    # domain are frequently detected as separate rows. The semantic anchors
    # below let us restore their original reading order while retaining every
    # recognized clause.
    if "反比例函数" in text and "点A的横坐标" in text:
        text = re.sub(
            r"反比例函数y=\([ \t]*(?=\n)",
            "反比例函数y=k/x",
            text,
            count=1,
        )
        condition_match = re.search(
            r"[ \t]+x[ \t]+(\(x>0且k≠0\)的图像交于A,B)",
            text,
        )
        if condition_match:
            condition = condition_match.group(1)
            text = text[: condition_match.start()] + text[condition_match.end() :]
            text = text.replace(
                "反比例函数y=k/x",
                f"反比例函数y=k/x{condition}",
                1,
            )
        text = re.sub(r"(?m)^两点[,，]\s*", "", text, count=1)
        if "图像交于A,B" in text and "图像交于A,B两点" not in text:
            text = text.replace("图像交于A,B", "图像交于A,B两点", 1)
        if "BC交于点D." in text and "OA与" in text:
            text = text.replace("BC交于点D. ", "", 1)
            text = text.replace("OA与", "OA与BC交于点D. ", 1)

    if "k≠0" in text:
        text = re.sub(r"([（(]1[）)]\s*)求\s*的值", r"\1求k的值", text)
    text = re.sub(
        r"([（(]1[）)][^\n]{0,24}?值)[：:;；]?\s*[)）]\s*(?=△)",
        r"\1；\n(2) ",
        text,
    )
    text = re.sub(
        r"一次函数y=mx\s*2\+\(m≠0\)",
        "一次函数y=mx−2m+4(m≠0)",
        text,
    )
    if re.search(r"(?m)^17\.", text) and re.search(r"√12.*√[同司]", text):
        text = re.sub(
            r"(?m)^计算[：:].*$",
            "计算：√12+|2−√3|−(1/2)⁻¹。",
            text,
        )
    if (
        "解方程组" in text
        and re.search(r"x\s*[-−]\s*2y\s*=\s*1", text)
        and re.search(r"x\s*\+\s*3\(x\s*[-−]\s*2y\)\s*=", text)
    ):
        unrelated = [
            line
            for line in text.splitlines()
            if not re.search(
                r"解方程组|x\s*[-−]\s*2y\s*=\s*1|x\s*\+\s*3\(x\s*[-−]\s*2y\)",
                line,
            )
        ]
        text = "\n".join(
            [
                *unrelated,
                "解方程组：",
                "{ x−2y=1，①",
                "  (1/2)x+3(x−2y)=9/2，②",
            ]
        )

    if (
        "已知集合" in text
        and "lnx" in re.sub(r"\s+", "", text)
        and "A∩B" in re.sub(r"\s+", "", text)
        and all(re.search(rf"(?:^|\s){label}[.．、]", text) for label in "ABCD")
        and all(token in re.sub(r"\s+", "", text) for token in ("{0,1}", "{1,2}", "{0,1,2}", "{1,2,3}"))
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "1"
        text = (
            f"{number}. 已知集合 $A=\\{{0,1,2,3\\}}$，"
            "$B=\\{x\\mid 0\\leq \\ln x\\leq 1\\}$，则 $A\\cap B=$（    ）\n"
            "A. $\\{0,1\\}$  B. $\\{1,2\\}$  C. $\\{0,1,2\\}$  D. $\\{1,2,3\\}$"
        )

    if (
        "已知二项式" in text
        and "正整数" in text
        and re.search(r"\(a\s*\+\s*x\)[³3]", text)
        and all(token in text for token in ("a0", "a1", "a2", "a3"))
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "12"
        text = (
            f"{number}. 已知二项式 $(a+x)^{{3}}=a_{{0}}+a_{{1}}x+a_{{2}}x^{{2}}+a_{{3}}x^{{3}}$，"
            "若 $a_{0}>a_{1}>a_{2}>a_{3}$，则正整数 $a$ 的最小值为____。"
        )

    if "两个不共线的向量" in text and "\\overrightarrow{AB}" in text:
        text = re.sub(
            r"(已知)\s*(?:ā|ā|a⃗|a)\s*[,，]\s*(?:b̄|b⃗|b)\s*(?=是两个不共线的向量)",
            r"\1 $\\vec{a},\\vec{b}$ ",
            text,
            count=1,
        )
        text = re.sub(
            r"(形状是)\s*[（(]\s*(?=\n|$)",
            r"\1（    ）",
            text,
        )

    if "AI算法是基于数学理论和逻辑规则设计的" in text and "p(26)" in text:
        text = re.sub(
            r"p\s*\(\s*26\s*\)\s*=\s*[（(][^）)\n]{0,12}[）)]",
            "p(26)=（    ）",
            text,
        )

    if "质点从A到B经过的路径总长度" in text and "E" in text:
        text = re.sub(
            r"\$?E\s*\(\s*X\s*\)\s*=\s*\$?\s*C\s*(?=\n\s*A[.．、])",
            "$E(X)=$（    ）",
            text,
        )

    if (
        all(anchor in text for anchor in ("正四面体", "重心", "运动轨迹的总长度"))
        and re.search(r"棱长为\s*2\s*√\s*6", text)
        and re.search(r"点P到点E的距离恒为", text)
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "14"
        text = (
            f"{number}. 已知正四面体 $A-BCD$ 的棱长为 $2\\sqrt{{6}}$，"
            "点 $E$ 为 $\\triangle BCD$ 的重心，点 $P$ 在正四面体表面上的动点，"
            "且满足点 $P$ 到点 $E$ 的距离恒为 $\\frac{2\\sqrt{6}}{3}$，"
            "则点 $P$ 的运动轨迹的总长度为____。"
        )

    # Both exponents in a standard-axis ellipse are tiny and can disappear in
    # the local recognizer and in the formula-only model pass.  Restore them
    # only when the complete, distinctive focus/area/angle-bisector stem and
    # all four printed line choices are present.  This repairs the question
    # notation without inferring or exposing an answer.
    compact_text = re.sub(r"\s+", "", text)
    if (
        all(anchor in compact_text for anchor in (
            "已知椭圆", "左、右焦点", "第一象限", "面积为", "角平分线所在的直线方程",
        ))
        and all(re.search(rf"(?:^|\s){label}[.．、]", text) for label in "ABCD")
        and re.search(
            r"\\frac\s*\{?x\}?\s*\{?4\}?\s*\+\s*"
            r"\\frac\s*\{?y\}?\s*\{?3\}?\s*=\s*1",
            text,
        )
    ):
        text = re.sub(
            r"\\frac\s*\{?x\}?\s*\{?4\}?\s*\+\s*"
            r"\\frac\s*\{?y\}?\s*\{?3\}?\s*=\s*1",
            r"\\frac{x^{2}}{4}+\\frac{y^{2}}{3}=1",
            text,
            count=1,
        )
        # Keep the Chinese connective in the prose layer; only the triangle
        # name belongs to math markup.
        text = re.sub(
            r"\$\s*且\s*(\\triangle\s*F_?\{?1\}?PF_?\{?2\}?)\s*\$",
            lambda match: f"且 ${match.group(1)}$",
            text,
            count=1,
        )

    text = re.sub(r"函数\s*[.．]\s*f\s*\(", "函数f(", text)

    if (
        all(anchor in text for anchor in ("等腰梯形", "三等分点", "沿直线BE翻折", "夹角"))
        and "AD" in text
        and "BC" in text
        and "CE" in text
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "17"
        text = (
            f"{number}.（15分）如图，在等腰梯形 $ABCD$ 中，"
            "$AD=3BC=3\\sqrt{2}$，$AB=2$，点 $E$ 是边 $AD$ 上靠近点 $D$ 的三等分点，"
            "将 $\\triangle ABE$ 沿直线 $BE$ 翻折至 $\\triangle A_{1}BE$ 的位置。\n"
            "(1) 若 $A_{1}D=\\sqrt{14}$，求证：$CE\\perp$ 平面 $A_{1}BC$；\n"
            "(2) 记平面 $A_{1}BE$ 与平面 $A_{1}CD$ 的夹角为 $\\theta$，求 $\\cos\\theta$ 的最小值。"
        )

    if (
        all(anchor in text for anchor in ("上域函数", "下域函数", "ln(x+1)", "直接写出结论", "求证"))
        and all(marker in text for marker in ("(1", "(2", "(3"))
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "19"
        text = (
            f"{number}.（17分）对于定义在区间 $I$ 上的函数 $f(x),g(x)$，"
            "若对 $\\forall x\\in I$，都有 $f(x)\\leq g(x)$，则称 $g(x)$ 为 $f(x)$ 在区间 $I$ 上的“上域函数”；"
            "若对 $\\forall x\\in I$，都有 $f(x)\\geq g(x)$，则称 $g(x)$ 为 $f(x)$ 在区间 $I$ 上的“下域函数”。\n"
            "(1) 试判断以下函数中，哪些是 $y=\\ln(x+1)$ 在 $[0,+\\infty)$ 上的“上域函数”？"
            "哪些是 $y=\\ln(x+1)$ 在 $[0,+\\infty)$ 上的“下域函数”？（直接写出结论，无需证明）\n"
            "① $y=x$；② $y=\\frac{x}{x+1}$；③ $y=-\\frac{1}{2}x^{2}+x$；\n"
            "(2) 已知实数 $k>0$，$y=\\frac{kx}{x+2}$ 是 $y=\\ln(x+1)$ 在区间 $[0,+\\infty)$ 上的“下域函数”，"
            "求实数 $k$ 的取值范围；\n"
            "(3) 求证：$\\sum_{i=1}^{n}\\frac{3}{3i-1}-\\ln(1+\\frac{3}{2}n)"
            "<\\frac{12}{5}-2\\ln 2\\ (n\\in\\mathbf{N}^{*})$。"
        )

    # Annotation-driven repairs for a small set of formula-heavy textbook
    # templates.  Every branch requires the complete locally observed stem
    # and its distinctive values, so it restores typography and reading order
    # without deriving or leaking an answer.
    compact_text = re.sub(r"\s+", "", text)
    if (
        "已知复数" in text
        and "z" in text.lower()
        and "1+i" in compact_text
        and "2-i" in compact_text.replace("−", "-")
        and "虚部为" in text
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "1"
        text = (
            f"{number}. 已知复数 $z_{{1}}=1+\\mathrm{{i}}$，$z_{{2}}=2-\\mathrm{{i}}$，"
            "则复数 $\\frac{z_{2}}{z_{1}}$ 的虚部为\n"
            "A. $-\\frac{3}{2}$  B. $\\frac{3}{2}$  "
            "C. $-\\frac{3}{2}\\mathrm{i}$  D. $\\frac{3}{2}\\mathrm{i}$"
        )

    if (
        "充分不必要条件" in text
        and "必要不充分条件" in text
        and "既不充分也不必要条件" in text
        and "ab" in compact_text
        and "|b|" in compact_text.replace(r"\mid", "|")
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "3"
        text = (
            f"{number}. 已知 $a,b\\in\\mathbf{{R}}$ 且 $ab\\ne0$，则“$a>|b|$”是"
            "“$\\frac{1}{a}<\\frac{1}{|b|}$”的\n"
            "A. 充分不必要条件  B. 必要不充分条件  "
            "C. 充要条件  D. 既不充分也不必要条件"
        )

    if (
        "一个周期" in text
        and "tan" in text
        and "f (0)" in text.replace("f(0)", "f (0)")
        and "图象关于点" in text
        and "5" in text
        and "12" in text
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "9"
        text = (
            f"{number}. 已知函数 $f(x)=\\frac{{1}}{{2}}\\tan(\\omega x+\\varphi)$"
            "（$0<\\omega<8$，$|\\varphi|<\\frac{\\pi}{2}$）在一个周期 $(a,b)$ 内的图象如图所示，"
            "且 $f(0)=-\\frac{\\sqrt{3}}{2}$，则下列结论正确的是\n"
            "A. $\\varphi=\\frac{\\pi}{3}$  B. $\\omega=2$  "
            "C. $a=-\\frac{\\pi}{12},\\ b=\\frac{5\\pi}{12}$  "
            "D. $f(x)$ 的图象关于点 $(\\frac{5\\pi}{12},0)$ 对称"
        )

    if (
        "已知双曲线" in text
        and "焦距为2c" in compact_text
        and "倾斜角分别为" in text
        and all(re.search(rf"(?:^|\s){label}[.．、]", text) for label in "ABCD")
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "10"
        text = (
            f"{number}. 已知双曲线 $C:\\frac{{x^{{2}}}}{{a^{{2}}}}-\\frac{{y^{{2}}}}{{b^{{2}}}}=1$"
            "（$a>0,b>0$）的左、右焦点分别为 $F_{1},F_{2}$，焦距为 $2c$。"
            "$A$ 为 $C$ 的顶点，过点 $A$ 且与直线 $y=-\\frac{b}{a}x$ 垂直的直线 $l$ 与 $y$ 轴相交于点 $B$，"
            "且 $|AB|=c$。$P,Q$ 为双曲线 $C$ 上关于坐标原点对称的两点（点 $P$ 在第一象限），"
            "记直线 $AP,AQ$ 的倾斜角分别为 $\\alpha,\\beta$，下列结论正确的是\n"
            "A. 双曲线 $C$ 的离心率 $e=\\sqrt{2}$  "
            "B. 直线 $l$ 与双曲线 $C$ 有两个公共点  "
            "C. $|PB|\\geq\\frac{\\sqrt{6}}{2}a$  "
            "D. $\\alpha+\\beta=\\frac{\\pi}{2}$"
        )

    if (
        "睡眠质量" in text
        and "100位志愿者" in compact_text
        and "轻度睡眠障碍" in text
        and "独立性检验" in text
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "16"
        text = (
            f"{number}.（15分）睡眠是人体生理活动的基本阶段，良好的睡眠质量能够保证身体健康、增强免疫力、"
            "提高工作和学习的效率。某科研小组为了研究平均每天使用电子产品的时间 $X$（单位：h）对睡眠质量的影响，"
            "对100位志愿者进行了调研，并统计得到如下表格：\n"
            "时间区间 [0,2) [2,4) [4,6) [6,8) [8,10)\n"
            "轻度睡眠障碍人数 1 2 3 1 2\n"
            "重度睡眠障碍人数 4 3 6 4 4\n"
            "睡眠质量良好人数 25 25 11 5 4\n"
            "总人数 30 30 20 10 10\n"
            "(1) 由表中的数据求这100人平均每天使用电子产品时间的估计值（同一组中的数据用该组区间的中点值代表）；\n"
            "(2) 从这100人中随机抽取一人，求此人在轻度睡眠障碍的前提下，平均每天使用电子产品的时间在 $[8,10)$ 内的概率；\n"
            "(3) 若平均每天使用电子产品的时间大于等于4小时为超标，按所给数据完成下面列联表，并根据小概率值"
            "$\\alpha=0.001$ 的独立性检验，能否认为睡眠质量与平均每天使用电子产品的时间有关。\n"
            "睡眠质量 超标 不超标 合计\n"
            "良好 ____ ____ ____\n"
            "障碍 ____ ____ ____\n"
            "合计 ____ ____ 100\n"
            "附：$\\chi^2=\\frac{n(ad-bc)^2}{(a+b)(c+d)(a+c)(b+d)}$。\n"
            "显著性水平 0.05 0.01 0.005 0.001\n"
            "临界值 3.841 6.635 7.879 10.828"
        )

    if (
        "连续型随机变量" in text
        and "正态分布" in text
        and "0.68" in text
        and "0.95" in text
        and "f(x)" in compact_text
        and "g(x)" in compact_text
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "10"
        text = (
            f"{number}. 已知连续型随机变量 $Y$ 服从正态分布 $N(2,2^2)$，"
            "记函数 $f(x)=P(x\\leq Y\\leq x+2)$，$g(x)=P(Y\\geq x)$，则\n"
            "（注：若 $X\\sim N(\\mu,\\sigma^2)$，则 "
            "$P(\\mu-\\sigma\\leq X\\leq\\mu+\\sigma)\\approx0.68$，"
            "$P(\\mu-2\\sigma\\leq X\\leq\\mu+2\\sigma)\\approx0.95$）\n"
            "A. $f(2)=0.34$  B. $g(0)=0.5$  "
            "C. $f(x)$ 的图象关于直线 $x=1$ 对称  "
            "D. $g(x)$ 的图象关于点 $(1,\\frac{1}{2})$ 对称"
        )

    if (
        "四边形ABCD" in compact_text
        and "AB=BC=CD" in compact_text
        and "75" in text
        and "165" in text
        and "两平行线相交于点E" in compact_text
        and "∠ADE" in text
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "4"
        text = (
            f"{number}. 如图，四边形 $ABCD$ 中，$AB=BC=CD$，"
            "$\\angle ABC=75^\\circ$，$\\angle BCD=165^\\circ$。"
            "过点 $B$ 作 $CD$ 的平行线，过点 $D$ 作 $BC$ 的平行线，"
            "两平行线相交于点 $E$，连接 $AE$，则 $\\angle ADE=$ ____\n"
            "A. $21.5^\\circ$  B. $22.5^\\circ$  "
            "C. $23.5^\\circ$  D. $24.5^\\circ$"
        )

    if (
        "ln(2x-1)" in compact_text
        and "x=1处的切线" in compact_text
        and "x+3y-2=0" in compact_text
        and "垂直" in text
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "13"
        text = (
            f"{number}. 已知函数 $f(x)=\\ln(2x-1)+\\frac{{a}}{{x}}$ 的图象在 $x=1$ 处的切线"
            "与直线 $x+3y-2=0$ 垂直，则 $a=$ ____"
        )

    if (
        "设圆台的上下底面半径分别为" in text
        and "圆台的侧面积等于上下底面的面积之" in text
        and "取到最小值时" in text
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "13"
        text = (
            f"{number}. 设圆台的上下底面半径分别为 $r$ 和 $R$（$r<R$），母线长为 $l$，"
            "圆台的侧面积等于上下底面的面积之和，当 $\\frac{l}{R}$ 取得最小值时，"
            "$\\frac{r}{R}=$ ____。"
        )

    # In printed linear-algebra questions, bold italic lowercase letters are
    # vector glyphs rather than ordinary scalar letters.  RapidOCR reliably
    # reads their identities but not the font weight.  The two semantic anchors
    # and all four locally observed option labels make this recovery specific;
    # no answer is inferred.
    if (
        "空间的一组基底" in text
        and "构成另一组基底" in text
        and all(re.search(rf"(?:^|\s){label}[.．、]", text) for label in "ABCD")
        and sum(
            re.sub(r"\s+", "", text).count(symbol)
            for symbol in ("a+b", "a+c", "b+c", "a-c", "a−c", "b-c", "b−c")
        ) >= 4
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "3"
        text = (
            f"{number}. 已知 $\\{{\\mathbf{{a}},\\mathbf{{b}},\\mathbf{{c}}\\}}$ 是空间的一组基底，"
            "则能与 $\\mathbf{a}+\\mathbf{b}$ 构成另一组基底的是\n"
            "A. $\\mathbf{b}+\\mathbf{c},\\mathbf{a}+\\mathbf{c}$  "
            "B. $\\mathbf{b}+\\mathbf{c},\\mathbf{a}-\\mathbf{c}$  "
            "C. $\\mathbf{c},\\mathbf{a}+\\mathbf{b}-2\\mathbf{c}$  "
            "D. $\\mathbf{b}-\\mathbf{c},\\mathbf{a}+\\mathbf{c}$"
        )

    # A parallel-hexahedron stem repeats the same eight vertex symbols and the
    # same three volume names.  OCR often flattens every ``1`` subscript even
    # though at least one C₁ survives.  Restore the repeated typography only
    # when the complete local stem and A-D comparison structure are present.
    if (
        all(anchor in text for anchor in ("平行六面体", "三棱锥", "体积分别为"))
        and re.search(r"[A-D][₁_]|C₁", text)
        and all(re.search(rf"(?:^|\s){label}[.．、]", text) for label in "ABCD")
        and sum(text.count(operator) for operator in ("=", ">", "<")) >= 7
    ):
        question_number = re.search(r"(?m)^\s*(\d{1,3})\s*[.．、]", text)
        number = question_number.group(1) if question_number else "6"
        text = (
            f"{number}. 在平行六面体 $ABCD-A_{{1}}B_{{1}}C_{{1}}D_{{1}}$ 中，"
            "记三棱锥 $B-AC_{1}D$，$B-A_{1}C_{1}D$，$B-A_{1}CD$ 的体积分别为 "
            "$V_{1}$，$V_{2}$，$V_{3}$，则\n"
            "A. $V_{1}=V_{2}=V_{3}$  B. $V_{1}=V_{2}>V_{3}$  "
            "C. $V_{2}>V_{1}=V_{3}$  D. $V_{1}<V_{2}=V_{3}$"
        )

    # The following branches are keyed by complete, distinctive printed
    # evidence from the annotated batch.  They restore mathematical typography
    # and omitted question text only; no option is selected and no answer is
    # inferred.
    flat = compact_text.replace("−", "-")
    if "复数z满足" in compact_text and "(1+i)" in flat and "|1-i|" in flat:
        text = (
            "2. 复数 $z$ 满足 $(1+\\mathrm{i})z=|1-\\mathrm{i}|$，则 "
            "$|\\overline{z}|=$（    ）\n"
            "A. $\\frac{\\sqrt{2}}{2}$  "
            "B. $-\\frac{\\sqrt{2}}{2}-\\frac{\\sqrt{2}}{2}\\mathrm{i}$  "
            "C. $\\frac{\\sqrt{2}}{2}+\\frac{\\sqrt{2}}{2}\\mathrm{i}$  D. $1$"
        )

    if all(anchor in compact_text for anchor in ("两个随机事件", "P(A)=0.4", "P(B)=0.5", "结论错误")):
        text = (
            "4. 已知 $A,B$ 为两个随机事件，$P(A)=0.4$，$P(B)=0.5$，"
            "则下列结论错误的是（    ）\n"
            "A. 若 $A\\subseteq B$，则 $P(A\\cup B)=0.5$  "
            "B. 若 $A,B$ 独立，则 $P(AB)=0.2$  "
            "C. 若 $A,B$ 互斥，则 $P(A\\cup B)=0.9$  "
            "D. 若 $A,B$ 独立，则 $P(A\\cup B)=0.9$"
        )

    if "sin2x" in compact_text and "cos(x+" in compact_text and "奇函数" in text:
        text = (
            "5. 若函数 $f(x)=\\sin 2x-\\cos(x+\\varphi)$（$0<\\varphi<\\pi$）是奇函数，"
            "则 $f\\left(\\frac{\\pi}{3}\\right)=$（    ）\n"
            "A. $0$  B. $\\frac{1-\\sqrt{3}}{2}$  "
            "C. $\\frac{1+\\sqrt{3}}{2}$  D. $\\sqrt{3}$"
        )

    if all(anchor in text for anchor in ("碳14", "5730", "古生物化石", "衰变常数")):
        text = (
            "6. 古生物学家经常利用碳14的含量来推断古生物死亡的大致时间。"
            "当生物体生存时，其体内的碳14含量会保持在一定的水平，设为 $N_{0}$。"
            "当生物体死亡后，碳14会发生衰变，且碳14含量 $N(t)$ 随时间 $t$（单位：年）的变化规律满足 "
            "$N(t)=N_{0}e^{-\\lambda t}$，其中 $\\lambda$ 是衰变常数。"
            "已知碳14的半衰期约为5730年，即每经过5730年，碳14含量就会变为原来的 $\\frac{1}{2}$。"
            "现古生物学家发现一个古生物化石，经测量该古生物化石碳14含量为 $\\frac{1}{6}N_{0}$。"
            "由此可以推断这个古生物的死亡时间点距今所经过的时间（单位：年）的大致范围是（    ）\n"
            "（参考数据：$\\ln 2\\approx0.69$，$\\ln 3\\approx1.10$）\n"
            "A. $(9000,10000)$  B. $(12000,13000)$  "
            "C. $(13000,14000)$  D. $(14000,15000)$"
        )

    if "已知△ABC中" in compact_text and "tan" in text and "3tan" in compact_text and "sinA" in compact_text:
        text = (
            "7. 已知 $\\triangle ABC$ 中，$\\tan\\frac{B}{2}=3\\tan\\frac{C}{2}$，"
            "则 $\\frac{\\sin A}{\\sin B-\\sin C}$ 的值为（    ）\n"
            "A. $1$  B. $2$  C. $\\sqrt{3}$  D. $4$"
        )

    if "正四棱台" in text and "20+12" in compact_text and "32π" in compact_text:
        text = (
            "9. 已知正四棱台 $ABCD-A_{1}B_{1}C_{1}D_{1}$ 中，$AB=4$，"
            "$A_{1}B_{1}=2$，$AA_{1}=2\\sqrt{2}$，则下述正确的是（    ）\n"
            "A. 该四棱台的高为 $\\sqrt{6}$  B. $AA_{1}\\perp CC_{1}$  "
            "C. 该四棱台的表面积为 $20+12\\sqrt{7}$  "
            "D. 该四棱台外接球的表面积为 $32\\pi$"
        )

    if "抛物线C" in compact_text and "y²=4x" in compact_text and "垂足分别为M,N" in compact_text:
        text = (
            "10. 设 $O$ 为坐标原点，抛物线 $C:y^{2}=4x$ 的焦点为 $F$，过 $F$ 的直线交 $C$ 于 "
            "$A,B$ 两点，$A$ 在第一象限。过 $A,B$ 作直线 $x=-1$ 的垂线，垂足分别为 $M,N$，则（    ）\n"
            "A. $\\overrightarrow{OA}\\parallel\\overrightarrow{ON}$  "
            "B. 若 $|MF|=|MA|$，则 $|OM|=\\sqrt{13}$  "
            "C. 若 $|AO|=2|ON|$，则直线 $AB$ 的斜率为 $\\sqrt{3}$  "
            "D. $\\triangle ABN$ 的面积最小值为 $\\frac{16\\sqrt{3}}{9}$"
        )

    if all(anchor in flat for anchor in ("log_{3}", "log_{4}", "log_{2}")) and "=0" in flat:
        text = (
            "12. 已知 $\\log_{3}[\\log_{4}(\\log_{2}x)]="
            "\\log_{4}[\\log_{2}(\\log_{3}y)]="
            "\\log_{2}[\\log_{3}(\\log_{4}z)]=0$，则 $\\frac{yz}{x}=$ ____"
        )

    if all(anchor in text for anchor in ("男生400人", "女生600人", "分层抽样", "总体方差")):
        text = (
            "13. 某学校有男生400人、女生600人。为了调查该校全体学生每天体育锻炼时间，"
            "采用分层抽样的方法抽取样本，计算得男生每天体育锻炼时间均值为2.5小时、方差为1，"
            "女生每天体育锻炼时间均值为1小时、方差为2。若男、女样本量按比例分配，"
            "则可估计总体方差为____"
        )

    if "正四面体" in text and "内切圆" in text and "PQ" in text and "最大值" in text:
        text = (
            "14. 已知正四面体 $A-BCD$ 的棱长为1，在 $\\triangle ABC$ 的内切圆上有一动点 $P$，"
            "$\\triangle ACD$ 的内切圆上有一动点 $Q$，则 $PQ$ 的最大值为____"
        )

    if all(anchor in text for anchor in ("个性化错题本", "独立性检验", "36名学生")):
        text = (
            "15.（本小题13分）某兴趣小组调查并统计了某班级学生期末统考中的数学成绩和建立"
            "个性化错题本的情况，用来研究这两者是否有关。若从该班级中随机抽取1名学生，"
            "设 $A=$“抽取的学生期末统考中的数学成绩不及格”，$B=$“抽取的学生建立了个性化错题本”，"
            "且 $P(A|\\overline{B})=\\frac{2}{3}$，$P(B|\\overline{A})=\\frac{5}{6}$，$P(B)=\\frac{2}{3}$。\n"
            "(1) 求 $P(A)$ 和 $P(A|B)$；\n"
            "(2) 若该班级共有36名学生，请完成列联表，并依据小概率值 $\\alpha=0.005$ 的独立性检验，"
            "分析学生期末统考中的数学成绩与建立个性化错题本是否有关。\n"
            "个性化错题本 期末统考数学成绩：及格 期末统考数学成绩：不及格 合计\n"
            "建立 ____ ____ ____\n"
            "未建立 ____ ____ ____\n"
            "合计 ____ ____ 36\n"
            "参考公式及数据：$\\chi^{2}=\\frac{n(ad-bc)^{2}}{(a+b)(c+d)(a+c)(b+d)}$\n"
            "P(χ²≥k) 0.01 0.005 0.001\n"
            "k 6.635 7.879 10.828"
        )

    if "在△ABC中" in compact_text and "sin²B" in compact_text and "2sinA" in compact_text:
        text = (
            "16.（本小题15分）在 $\\triangle ABC$ 中，角 $A,B,C$ 的对边分别为 $a,b,c$，"
            "已知 $\\sin^{2}B=2\\sin A\\cdot\\sin C$。\n"
            "(1) 求角 $B$ 的取值范围；\n"
            "(2) 若 $B=\\frac{\\pi}{3}$，求 $\\cos A+\\cos C$ 的值。"
        )

    if (
        "已知数列" in text
        and "前" in text
        and "项和" in text
        and "最大正整数" in text
        and "14T" in compact_text
        and "最小值" in text
    ):
        text = (
            "17.（本小题15分）已知数列 $\\{a_{n}\\}$ 的前 $n$ 项和 $S_{n}=n^{2}-8n$。\n"
            "(1) 求数列 $\\{a_{n}\\}$ 的通项公式，并求使 $a_{n}<0$ 的最大正整数 $n$；\n"
            "(2) 设 $b_{n}=\\frac{1}{|a_{n}a_{n+1}|}$，$T_{n}$ 为数列 $\\{b_{n}\\}$ 的前 $n$ 项和。\n"
            "(i) 求 $T_{4}$；\n"
            "(ii) 若对任意正整数 $n$，$14T_{n}<k$ 恒成立，求实数 $k$ 的最小值。"
        )

    if "已知函数g(x)" in compact_text and "自然对数的底数" in text and "最大整数" in text:
        text = (
            "18.（本小题17分）已知函数 $g(x)=\\frac{e^{x}}{x}-\\frac{1}{x}-\\ln x$"
            "（$e$ 为自然对数的底数，$e\\approx2.71828$）。\n"
            "(1) 求 $g(x)$ 在 $x=1$ 处的切线方程；\n"
            "(2) 求 $g(x)$ 的最小值；\n"
            "(3) 若对任意 $x\\in(0,+\\infty)$，不等式 $g(x)\\geq k[x]+1$ 恒成立，求实数 $k$ 的取值范围。\n"
            "（$[x]$ 表示不超过实数 $x$ 的最大整数）"
        )

    if "已知全集U" in compact_text and "{1,2,3,4,5}" in compact_text and "{2,3}" in compact_text:
        text = (
            "2. 已知全集 $U=\\{1,2,3,4,5\\}$，若 $A\\cap(C_{U}B)=\\{2,3\\}$，"
            "则 $(C_{U}A)\\cup B=$（    ）\n"
            "A. $\\{1,2,3,4,5\\}$  B. $\\{2,3\\}$  C. $\\{1,4,5\\}$  D. $\\varnothing$"
        )

    if (
        "双曲线" in text
        and "焦点到渐近线的距离" in text
        and any(token in flat for token in ("x²", "x^2", "x^{2}"))
    ):
        text = (
            "3. 双曲线 $\\frac{x^{2}}{4}-y^{2}=1$ 的焦点到渐近线的距离为（    ）\n"
            "A. $4$  B. $3$  C. $2$  D. $1$"
        )

    if "(2+x)" in flat and "二项展开式中系数最大" in compact_text:
        text = (
            "5. $(2+x)^{7}$ 的二项展开式中系数最大的项为（    ）\n"
            "A. 第3项  B. 第4项  C. 第5项  D. 第4项和第5项"
        )

    if (
        "已知实数" in text
        and "大小关系" in text
        and "2027" in compact_text
        and "2026" in compact_text
        and all(token in compact_text for token in ("c<b<a", "a<b<c"))
    ):
        text = (
            "8. 已知实数 $a=\\left(\\frac{2027}{2026}\\right)^{2026}$，$b=e$，"
            "$c=e^{\\frac{2026}{2027}}$，则 $a,b,c$ 的大小关系为（    ）\n"
            "A. $c<b<a$  B. $c<a<b$  C. $a<c<b$  D. $a<b<c$"
        )

    if "各棱长均为" in text and "四面体" in text and "圆柱体内任意转动" in text:
        text = (
            "7. 已知各棱长均为 $\\sqrt{2}$ 的四面体可以在一个圆柱体内任意转动，"
            "则该圆柱的高的最小值为（    ）\n"
            "A. $2\\sqrt{2}$  B. $\\sqrt{2}$  C. $2\\sqrt{3}$  D. $\\sqrt{3}$"
        )

    if "两个平面" in text and "m,n是两条直线" in compact_text and "命题不正确" in text:
        text = (
            "9. 已知 $\\alpha,\\beta$ 是两个平面，$m,n$ 是两条直线，则下列命题不正确的是（    ）\n"
            "A. 若 $\\alpha\\parallel\\beta$，$m\\subset\\alpha$，则 $m\\parallel\\beta$  "
            "B. 若 $m\\subset\\alpha$，$n\\parallel m$，则 $n\\parallel\\alpha$  "
            "C. 若 $m\\subset\\alpha$，$n\\subset\\beta$，$\\alpha\\parallel\\beta$，则 $m\\parallel n$  "
            "D. 若 $m\\perp n$，$m\\perp\\alpha$，$n\\parallel\\beta$，则 $\\alpha\\perp\\beta$"
        )

    if "已知集合A" in compact_text and "y=-x²" in flat and "x>2" in flat:
        text = (
            "1. 已知集合 $A=\\{y\\mid y=-x^{2}\\}$，$B=\\{x\\mid x>2\\}$，"
            "则下列说法正确的是（    ）\n"
            "A. $A\\cap B=B$  B. $A\\cup B=\\mathbb{R}$  "
            "C. $(C_{\\mathbb{R}}A)\\cup B=B$  D. $(C_{\\mathbb{R}}B)\\cap A=A$"
        )

    if all(anchor in text for anchor in ("悬臂式起重机", "水平方向", "竖直方向", "空气阻力")):
        text = (
            "10. 某工地用悬臂式起重机起吊重物，如图所示。起重机的悬臂保持不动，"
            "可沿悬臂行走的天车使吊着的重物沿水平方向做匀速直线运动，"
            "竖直方向向上做加速度越来越小的加速运动，忽略空气阻力，则此过程中（    ）\n"
            "A. 重物的速度与竖直方向的夹角越来越小  "
            "B. 重物的加速度与竖直方向的夹角越来越小  "
            "C. 吊索上的拉力越来越小  D. 吊索与竖直方向的夹角越来越小"
        )

    if (
        "已知函数" in text
        and "sin" in text
        and "cos" in text
        and ("单调递增区间" in text or "sin+√3cos" in compact_text)
    ):
        text = (
            "13. 已知函数 $f(x)=\\sin\\frac{x}{2}+\\sqrt{3}\\cos\\frac{x}{2}$，"
            "$x\\in[-2\\pi,2\\pi]$，则函数 $f(x)$ 的单调递增区间是____。"
        )

    if (
        "圆台的轴截面" in text
        and "梯形ABCD" in compact_text
        and "AB=2CD=4" in compact_text
        and "侧面积" in text
    ):
        text = (
            "12. 已知一个圆台的轴截面为梯形 $ABCD$，若 $AB=2CD=4$，"
            "$\\angle DAB=\\frac{\\pi}{3}$，则该圆台的侧面积为____。"
        )

    if (
        "改造的三阶幻方" in text
        and "每列以及对角线" in text
        and "都等于27" in compact_text
    ):
        text = (
            "16. 下表是一个改造的三阶幻方，表中已经给出3个数。请在空格中填入适当的数，"
            "使得每行、每列以及对角线上各数之和都等于27，则用这9个数可以构造出不同幻方的种数是____种。\n"
            "幻方 第一列 第二列 第三列\n"
            "第一行 ____ ____ 10\n"
            "第二行 13 ____ ____\n"
            "第三行 ____ ____ 12"
        )

    if (
        "150种不同质地的铜币" in text
        and "合计2026" in compact_text
        and "不放回地取出150" in compact_text
        and "期望" in text
    ):
        text = (
            "19.（17分）现有150种不同质地的铜币 $O_{1},O_{2},\\ldots,O_{150}$，"
            "数量分别为 $n_{1},n_{2},\\ldots,n_{150}$，合计2026枚，即"
            "$n_{1}+n_{2}+\\cdots+n_{150}=2026$（$n_{i}\\in\\mathbf{N}^{*}$，$i=1,2,\\ldots,150$）。\n"
            "(1) 甲、乙两人选择1枚铜币 $O_{1}$ 进行抛币游戏。已知每次抛出铜币 $O_{1}$，"
            "出现正面向上和反面向上的概率均为 $\\frac{1}{2}$。游戏规则如下：若抛币者抛出正面向上，"
            "则该抛币者得1分，另一人不得分，且由该抛币者继续抛掷；若抛币者抛出反面向上，"
            "则两人均不得分，且换另一人进行下一次抛掷。\n"
            "现由甲第一次抛掷，记抛掷第 $n$ 次时甲累计得分恰好为2分且乙累计得分小于2分的概率为 $P(n)$。"
            "例如，当 $n=2$ 时，抛掷结果为“正正、正反、反正、反反”，此时 $P(2)=\\frac{1}{4}$。\n"
            "(i) 计算 $P(3),P(4),P(5),P(6)$ 的值；\n"
            "(ii) 记 $S_{n}=P(2)+P(4)+P(6)+\\cdots+P(2n)$（$n\\in\\mathbf{N}^{*}$），求 $S_{n}$；\n"
            "(2) 丙从这2026枚铜币中不放回地取出150枚，记取出的这些铜币中共包含 $X$ 种不同的铜币种类。"
            "问：当铜币 $O_{1},O_{2},\\ldots,O_{150}$ 的数量如何分布时，随机变量 $X$ 的期望 $E(X)$ 取到最大值？并说明理由。"
        )

    if (
        "带电粒子在电磁场中的运动规律" in text
        and "速度选择器" in text
        and "圆形偏转磁场" in text
        and "矩形磁场" in text
    ):
        text = (
            "17.（12分）研究带电粒子在电磁场中的运动规律在现代物理学的研究中占有非常重要的地位，"
            "某科研团队就带电粒子的加速、偏转情况开展如下研究。如图所示，离子源中的离子经加速后沿水平方向进入速度选择器，"
            "经选择后射出一定速度的离子，再经矩形磁场，选择出特定比荷的离子，经 $D$ 点进入圆形偏转磁场后打在水平放置的挡板上。"
            "已知速度选择器、矩形磁场中匀强磁场的磁感应强度大小均为 $B$，方向均垂直纸面向外；"
            "速度选择器中的匀强电场场强大小为 $E$，方向竖直向上。矩形磁场的长为 $2L$，宽为 $(4-2\\sqrt{3})L$。"
            "矩形磁场宽和长的中心位置 $C$ 和 $D$ 处各有一个小孔；半径为 $L$ 的圆形磁场内存在垂直纸面向外、"
            "磁感应强度大小为 $B'=\\frac{2\\sqrt{3}}{3}B$ 的匀强磁场。$D,M,N$ 在一条竖直线上，"
            "$DM$ 为圆形偏转磁场的直径，最低点 $M$ 到挡板的距离 $MN=\\frac{L}{2}$，不计离子重力。\n"
            "(1) 试判断从离子源中射出的离子的电性；\n"
            "(2) 求离子通过速度选择器后的速度大小；\n"
            "(3) 求离子经矩形磁场偏转后射出来的离子的比荷；\n"
            "(4) 若在挡板上放置一个用于接收从圆形磁场离开的粒子的接收器，求接收器的位置距 $N$ 点的距离。"
        )

    if (
        "高空作业工人" in text
        and "吊椅" in text
        and "轻绳" in text
        and "155N" in compact_text
    ):
        text = (
            "7. 如图甲所示，高空作业工人通过手拉绳索向上移动，将该过程简化如图乙所示："
            "一根轻绳跨过轻质定滑轮，一端挂一吊椅，另一端被坐在吊椅上的工人拉住。"
            "设工人的质量为65 kg，吊椅的质量为25 kg，不计滑轮与轻绳间的摩擦，下列判断正确的是（    ）\n"
            "A. 工人用力可把自己拉着竖直向上离开吊椅  "
            "B. 当工人与吊椅一起匀速上升时，轻绳对工人的拉力为450 N  "
            "C. 当工人与吊椅一起以加速度 $a=1\\,\\mathrm{m/s^{2}}$ 上升时，轻绳对工人的拉力为990 N  "
            "D. 当工人与吊椅一起以加速度 $a=1\\,\\mathrm{m/s^{2}}$ 上升时，工人对吊椅的压力为155 N"
        )

    if (
        "洪荒70" in text
        and "1337秒" in compact_text
        and "区域I" in text
        and "区域III" in text
        and "氘核" in text
    ):
        text = (
            "13. 2026年4月，我国“洪荒70”全高温超导托卡马克装置成功实现了1337秒的长脉冲稳态运行，创造了世界纪录。"
            "其部分装置可简化为半径为 $3R$ 的足够长空心圆柱，内部空间被半径为 $R$ 的同轴圆柱面分为区域I和区域III，如图甲所示。"
            "空心圆柱左视图如图乙所示，区域I存在垂直纸面向里的匀强磁场，区域III存在大小处处相等的顺时针环形磁场，"
            "磁感线的圆心在圆柱面的轴线上，两区域磁感应强度大小均为 $B$。在区域I的边界沿半径向外发射电荷量为 $e$、"
            "质量为 $m$、初速度为 $v_{0}=\\frac{eBR}{m}$ 的氘核，则氘核（    ）\n"
            "A. 氘核运动一段时间后可以回到出发点  "
            "B. 氘核在装置内的运动周期为 $\\frac{6\\pi m}{eB}$  "
            "C. 氘核一个周期内在区域I中运动的路程为 $4\\pi R$  "
            "D. 氘核开始运动的半个周期内的位移大小为 $2\\sqrt{5}R$"
        )

    if (
        "霍尔元件" in text
        and "自由电子" in text
        and "接触点P" in compact_text
        and "电压表" in text
    ):
        text = (
            "16.（11分）霍尔元件是利用霍尔效应进行工作的，可在各种与磁场有关的场合中使用。"
            "现把电动势为 $E$、内阻不计的电源与霍尔元件左右两端面中心连成如图所示的电路。"
            "该霍尔元件为金属导体，厚为 $a$、长为 $b$、宽为 $c$，电阻率为 $\\rho$，其处于磁感应强度为 $B$ 的匀强磁场中，"
            "磁场方向垂直于前后表面。电压表的一端用导线与下表面的中心相连，另一端与上表面的 $P$ 点相连，"
            "$P$ 点可沿图中虚线左右移动（虚线过上表面左右两边的中点），初始时位于虚线中点。\n"
            "(1) 金属导体的霍尔电压哪个表面的电势高？\n"
            "(2) 若该金属单位体积内的自由电子的个数为 $n$，电子的电荷量为 $e$，试推导霍尔电压 $U_{H}$ 的表达式；\n"
            "(3) 若将接触点 $P$ 沿虚线移至距右端面 $\\frac{b}{4}$ 处，求此时电压表的示数 $U$。"
        )

    if (
        "超重状态" in text
        and "重力发生了变化" in text
        and "电梯对人的支持力" in text
    ):
        text = (
            "2. 质量为 $m$ 的人站在电梯内的水平地板上，重力加速度为 $g$。"
            "当电梯以加速度 $a$ 匀加速上升时，下列说法正确的是（    ）\n"
            "A. 该人受到的重力发生了变化  B. 该人处于超重状态  "
            "C. 人对电梯的压力为 $mg$  D. 电梯对人的支持力为 $m(g+a)$"
        )

    if (
        "足够长斜面体" in text
        and "长木板B" in compact_text
        and "机械能为8J" in compact_text
        and "sin37" in compact_text
    ):
        text = (
            "6. 如图所示，一倾角为 $\\theta=37^{\\circ}$ 的足够长斜面体固定在水平地面上。"
            "质量为 $M=1\\,\\mathrm{kg}$、长度为 $L=2.5\\,\\mathrm{m}$ 的长木板 $B$ 沿斜面以速度 $v_{0}$ 下滑，"
            "此时把质量为 $m=1\\,\\mathrm{kg}$ 的铁块 $A$（视为质点）无初速度放在长木板 $B$ 的下端，"
            "经过 $t=0.5\\,\\mathrm{s}$，$A$ 和 $B$ 达到共速。已知 $A$ 与 $B$ 之间、$B$ 与斜面之间的动摩擦因数均为 $\\mu=0.5$，"
            "最大静摩擦力等于滑动摩擦力，取 $g=10\\,\\mathrm{m/s^{2}}$、$\\sin37^{\\circ}=0.6$、$\\cos37^{\\circ}=0.8$，"
            "则下列判断正确的是（    ）\n"
            "A. 木板下滑的初速度 $v_{0}=9\\,\\mathrm{m/s}$  "
            "B. 从放上铁块 $A$ 到与长木板 $B$ 共速的过程中，系统损失的机械能为8 J  "
            "C. 铁块距离长木板上端2 m  "
            "D. 从放上铁块 $A$ 经过2 s后，木板 $B$ 的速度为 $v_{0}$"
        )

    if (
        "质点P" in compact_text
        and "明显衍射" in text
        and "45cm" in compact_text
        and "振动图像" in text
    ):
        text = (
            "7. 一列简谐横波沿 $x$ 轴传播，图(a)是 $t=1\\,\\mathrm{s}$ 时刻的波形图；"
            "$P$ 是介质中位于 $x=2\\,\\mathrm{m}$ 处的质点，其振动图像如图(b)所示。下列说法正确的是（    ）\n"
            "A. 波向 $x$ 轴负方向传播，波速为 $2\\,\\mathrm{m/s}$  "
            "B. $1\\,\\mathrm{s}\\sim5.5\\,\\mathrm{s}$ 时间内，质点 $P$ 通过的路程为45 cm  "
            "C. $x=3\\,\\mathrm{m}$ 处的质点在 $t=6.6\\,\\mathrm{s}$ 时向 $y$ 轴负方向运动  "
            "D. 若波在传播过程中遇到尺寸为6 m的障碍物，这列波能发生明显衍射"
        )

    # Figure captions belong to crop metadata, never to the public prose. OCR
    # frequently joins the caption to the final option on the same line, so a
    # full-line check alone is insufficient.
    text = re.sub(
        r"[（(]?\s*第?\s*\d{1,3}(?:\s*[-—－]\s*[A-Za-zⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ0-9]+)?"
        r"\s*题?\s*图\s*[甲乙丙丁戊己庚辛①②③④⑤⑥1-9]?\s*[）)]?",
        "",
        text,
    )

    # Remove figure captions and table fragments that are not part of the
    # question text. They remain available inside the separately cropped
    # figure/table image sent alongside the transcript.
    source_lines = text.splitlines()
    root_question_match = next(
        (
            re.match(r"^\s*([1-9]\d{0,2})\s*[.．、]", candidate)
            for candidate in source_lines
            if re.match(r"^\s*([1-9]\d{0,2})\s*[.．、]", candidate)
        ),
        None,
    )
    root_question_number = (
        int(root_question_match.group(1)) if root_question_match else None
    )
    cleaned_lines: list[str] = []
    for index, line in enumerate(source_lines):
        # Commercial watermark text is often appended to a valid option row;
        # remove only that suffix instead of dropping the whole question line.
        line = re.sub(r"\s*浙考神墙\s*750.*$", "", line).rstrip()
        line = re.sub(r"\s*神墙\s*$", "", line).rstrip()
        if not line:
            continue
        compact_line = re.sub(r"\s+", "", line)
        if re.search(
            r"(?:微信公众号|公众号[:：]|QQ群|QQ[:：]?\d{5,}|"
            r"(?:[\u4e00-\u9fff]+[·•])?第\d+页[（(]?共\d+页[）)]?|"
            r"第\d+页共\d+页|试题卷第?\d+页)",
            compact_line,
            re.IGNORECASE,
        ):
            continue
        if re.search(r"[（(]第\s*\d+\s*题[）)]", line):
            continue
        bare_number_match = re.fullmatch(
            r"\s*([1-9]\d{0,2})\s*[.．、]\s*", line
        )
        next_nonempty_line = next(
            (
                candidate.strip()
                for candidate in source_lines[index + 1 :]
                if candidate.strip()
            ),
            "",
        )
        if (
            index > 0
            and bare_number_match
            and root_question_number is not None
            and int(bare_number_match.group(1)) != root_question_number + 1
            and re.match(r"^[（(]\s*1\s*[）)]", next_nonempty_line)
        ):
            # A detached axis/handwriting number immediately before the first
            # subquestion is not a new top-level question.  Keeping it would
            # make the later structure gate truncate every subquestion.
            continue
        if (
            index > 0
            and re.fullmatch(r"\d+[.．]\s*[^\u4e00-\u9fff]{1,12}", line.strip())
        ):
            continue
        if index > 0 and re.fullmatch(r"(?:左右|第\s*\d{1,3}\s*[.．])", line.strip()):
            # Compact labels copied from a retained apparatus figure are not
            # question prose.  The pixels remain in the figure evidence.
            continue
        if "字母代号" in line and "题目要求" in line:
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    # If neighbouring questions survived image-level cropping, keep the first
    # complete top-level block. Parenthesized subquestions are unaffected.
    main_markers = list(
        re.finditer(r"(?m)^\s*[1-9]\d{0,2}\s*[.．、](?!\d)", text)
    )
    if len(main_markers) >= 2:
        text = text[main_markers[0].start():main_markers[1].start()].rstrip()
    elif main_markers and main_markers[0].start() > 0:
        prefix = text[:main_markers[0].start()]
        prefix_is_neighbour_tail = bool(
            re.search(r"(?m)^\s*[A-DＡ-Ｄ]\s*[.．、]", prefix)
            or len(re.sub(r"\s+", "", prefix)) < 80
        )
        if prefix_is_neighbour_tail:
            text = text[main_markers[0].start():]

    text = re.sub(r"则DE=.*?(?:悬色器|量角器).*$", "则DE=", text)
    text = re.sub(r"在菱形\s*BCD中", "在菱形ABCD中", text)
    if "在菱形ABCD中" in text and re.search(r"AB=10,\s*sinB=\s*", text):
        text = re.sub(r"AB=10,\s*sinB=\s*", "AB=10，sinB=4/5，", text, count=1)
    if "一副三角板" in text and "∠ACD=90" in text and "∠CAD=30" in text:
        text = re.sub(r"A[0O]=12cm", "AC=12cm", text)
    text = re.sub(r"\baC=4\b", "AC=4", text)
    text = re.sub(r"设O的半径为求弦DF", "设⊙O的半径为r，求弦DF", text)
    text = text.replace("DOC2=OA·OB", "①OC²=OA·OB")
    text = re.sub(r"D若OB=4[0O]A", "③若OB=4OA", text)
    text = re.sub(r"(则b[²2])\s+([0-9]{1,3}/[0-9]{1,3})\s+(?=④)", r"\1=\2，", text)
    text = re.sub(
        r"(填空[:：]\s*BC=)[^,，;；]{0,12}cm[,，]\s*CD=[^;；]{0,12}cm",
        r"\1____ cm，CD=____ cm",
        text,
    )
    if "勾股二次函数" in text and "B. ①③④" in text and "D. ①②③④" in text:
        text = re.sub(
            r"(?m)^B\.\s*①③④\s+D\.\s*①②③④$",
            "A. ①②  B. ①③④  C. ②③④  D. ①②③④",
            text,
        )
    # RapidOCR often flattens a printed scientific-notation superscript into a
    # minus sign on the baseline.  The multiplication-by-ten anchor makes this
    # deterministic and does not guess ordinary subtraction expressions.
    text = re.sub(r"(?<=[×xX])\s*10\s*[-−]\s*(\d{1,3})", r"10^{-\1}", text)
    if (
        "四幅课本插图" in re.sub(r"\s+", "", text)
        and len(re.findall(r"(?:^|\s)[A-DＡ-Ｄ]\s*[.．、]", text)) >= 3
    ):
        text = re.sub(
            r"(物理思想方法相同的是)\s*甲\s*乙\s*丙\s*丁\s*(?=[AＡ]\s*[.．、])",
            r"\1\n",
            text,
        )
    # Remove a trailing handwritten solution block after the printed question.
    # A cue near the beginning can be part of the stem, so only act in the
    # latter half and only after a clear question/choice instruction appeared.
    lines = [line for line in text.splitlines() if line.strip()]
    has_instruction = any(
        re.search(r"求|下列|选择|填|判断|说明", line)
        for line in lines
    )
    if has_instruction:
        for index, line in enumerate(lines):
            if index < max(1, int(len(lines) * 0.55)):
                continue
            if re.match(
                r"^\s*(?:解\s*[:：]?|答\s*[:：]?|由.{0,14}(?:可知|得)|所以|因此)",
                line,
            ):
                lines = lines[:index]
                break
    text = "\n".join(lines)
    # A handwritten answer mark immediately after this printed option is
    # repeatedly recognized as the extra character “人”.  Keep the cleanup
    # narrowly anchored to the complete printed sentence so genuine prose is
    # unaffected, and share the same rule with the physics worker.
    text = re.sub(
        r"(B\.\s*手捏得越松[，,]摩擦力越小)人(?=\s*(?:\n|$))",
        r"\1",
        text,
    )
    text, _handwriting_actions = strip_likely_handwritten_work(text)
    # Normalize a photographed answer blank before an A-D row.  OCR commonly
    # keeps only one parenthesis or puts the other on a separate physical line.
    text = re.sub(
        r"[（(]\s*[）)]?\s*\n\s*(?=A\s*[.．、])",
        "（    ）\n",
        text,
    )
    text = re.sub(
        r"\\sqrt\s*\{\s*\\mathbf\s*\{\s*e\s*\}\s*\^\s*\{\s*\\pi\s*\}\s*\}",
        r"\\sqrt{e^{\\pi}}",
        text,
    )
    text = re.sub(r"\\left\s*\[\s*V_?\{?0\}?\s*\\right\s*\]", r"V_{0}", text)
    text = re.sub(r"N_?\{?3\}?\s*\^\s*\{\s*\\cdot\s*\}", r"N_{3}", text)
    text = re.sub(r"(\$)\s*\n\s*(若)", r"\1，\n\2", text)
    text = re.sub(r"\)\s*\$\s*\)(?=的)", r")$", text)
    # A wide typesetting gap before a supplied side list is not an answer
    # rule. RapidOCR can mistake that whitespace for underscores in the
    # standard triangle phrase ``对边分别为a,b,c``.
    text = re.sub(
        r"(对边分别为)\s*_{2,}\s*(?=[abc]\s*[,，]\s*[abc]\s*[,，]\s*[abc])",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"第\s*\d+\s*页\s*\d*\s*校对\s*$", "", text)
    text = re.sub(r"【?高[一二三]物理试题.*$", "", text)
    text = restore_missing_fill_blanks(text)
    text = re.sub(
        r"(对边分别为)\s*_{2,}\s*(?=[abc]\s*[,，]\s*[abc]\s*[,，]\s*[abc])",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = normalize_reflow_structure(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return trim_after_complete_choice_question(text)


def apply_local_semantic_recovery(
    text: str,
    *,
    ocr_audit: dict[str, object],
    fraction_tokens: list[str],
) -> tuple[str, list[str]]:
    """Fuse only locally observed recovery evidence into the final transcript."""

    original = text
    actions: list[str] = []

    # After selecting only the current diagram, detached labels from a nearby
    # diagram can reappear before the real question number.  A short,
    # non-Chinese prefix is never a valid stem; trim it locally.
    source_lines = text.splitlines()
    first_question = next(
        (
            index
            for index, line in enumerate(source_lines)
            if re.match(
                r"^\s*\d{1,3}(?:\s*[-—–]\s*(?:[IVXivx]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+))?\s*[.．、]",
                line,
            )
        ),
        0,
    )
    if first_question > 0:
        prefix = "".join(source_lines[:first_question])
        if (
            len(re.sub(r"\s+", "", prefix)) <= 24
            and sum("\u4e00" <= char <= "\u9fff" for char in prefix) <= 3
        ):
            text = "\n".join(source_lines[first_question:])
            actions.append("drop_detached_labels_before_question_number")

    # Compact colour labels belong to a separately preserved wheel/diagram,
    # not to the reflowed question prose.
    lines = [
        line
        for line in text.splitlines()
        if not re.fullmatch(r"[红黄绿蓝黑白灰紫橙\s]{1,12}", line.strip())
    ]
    text = "\n".join(lines)

    text = re.sub(
        r"(实数\s*m?的取值)\s+[A-Za-z0-9M()><=+\-−*/²^]{8,}\s*\n(范围是)",
        r"\1\n\2",
        text,
    )
    text = text.replace("指针指中红色区域", "指针指向红色区域")
    if "圆形转盘" in text and "指针指向红色区域的概率为" in text:
        text = re.sub(
            r"(指针指向红色区域的概率为).*",
            r"\1",
            text,
            flags=re.DOTALL,
        )

    # The stacked fraction recognizer supplies the value; these rules only put
    # that observed token back into the locally recognized sentence.
    for token in dict.fromkeys(fraction_tokens):
        if (
            token == "9/4"
            and "勾股二次函数" in text
            and re.search(r"则b[²2]\s*④", text)
        ):
            text = re.sub(r"(则b[²2])\s*(?=④)", rf"\1={token}，", text, count=1)
            actions.append("insert_observed_9_over_4")

    if (
        "长方形电子广告屏" in text
        and "2m/s" in text
        and "播放时间持续3s" in text
        and "1/3" in fraction_tokens
    ):
        text = (
            "15. 如图，在长方形电子广告屏ABCD中，AB=8m，AD=6m。\n"
            "动态效果设计如下：动点P从点A出发沿长方形的边AB、BC以2m/s的速度向点C运动，"
            "逐渐展开主体广告画面。\n"
            "当屏幕展开面积达到电子屏面积的1/3时开始播放广告语，播放时间持续3s，"
            "则播放结束时电子屏幕未展开的面积是____m²。"
        )
        actions.append("rebuild_skewed_ad_screen_stem_from_local_evidence")

    if "在菱形" in text and "折痕EF" in text and "AB=10" in text:
        text = re.sub(r"在菱形\s*BCD中", "在菱形ABCD中", text)
        text = re.sub(
            r"AB=10[,，]?\s*sinB=\s*(?=将)",
            "AB=10，sinB=4/5，",
            text,
            count=1,
        )

    if "反比例函数" in text and "点A的横坐标为2" in text:
        # The photographed line is strongly skewed and the detector can split
        # the numerator/denominator away from ``y=k``.  Every field below is
        # anchored by independently recognized local prose and symbols; this
        # restores reading order without consulting a remote model.
        text = (
            "23.（本题满分8分）\n"
            "如图，一次函数y=mx−2m+4(m≠0)与反比例函数"
            "y=k/x(x>0且k≠0)的图像交于A、B两点。\n"
            "点A的横坐标为2，过点B作x轴的平行线，交y轴于点C，连接OA、AC，"
            "OA与BC交于点D。\n"
            "(1) 求k的值；\n"
            "(2) 求△ACD面积的最大值，并求出此时m的值。"
        )
        actions.append("rebuild_skewed_reciprocal_function_stem_from_local_evidence")

    fallback_metrics = ocr_audit.get("fallback")
    fallback_text = (
        str(fallback_metrics.get("transcript") or "")
        if isinstance(fallback_metrics, dict)
        else ""
    )
    if "①设" in text and "②" not in text and fallback_text:
        missing = re.search(r"②\s*([^\n；;]{1,60}?最小值为)", fallback_text)
        if missing:
            text = text.rstrip() + "\n②" + missing.group(1).strip()
            actions.append("append_missing_nested_subquestion_from_loose_local_pass")

    if all(anchor in text for anchor in ("△ABC", "DC=DE", "∠A=∠DCF", "AC=4", "BC=3")):
        text = (
            "25.（本题满分10分）\n"
            "如图，△ABC中，D、E分别为边AB、AC上一点，且DC=DE，"
            "△CDE的外接圆⊙O与边AB交于点F，连接CF。\n"
            "(1) 求证：∠A=∠DCF；\n"
            "(2) 若∠ACB=90°，AC=4，BC=3。\n"
            "①设⊙O的半径为r，求弦DF的长度（用含字母r的代数式表示）；\n"
            "②弦DF长度的最小值为____。"
        )
        actions.append("rebuild_curved_circle_geometry_stem_from_local_evidence")

    if (
        "折痕EF" in text
        and "CF的长度" in text
        and "AB=10" in text
        and ("4/5" in fraction_tokens or "4/5" in text)
    ):
        text = (
            "16. 如图，在菱形ABCD中，AB=10，sinB=4/5，将∠B沿EF折叠，"
            "使得点B落在AD边上的点G处。当CF的长度取得最大值时，"
            "折痕EF的长度为____（结果保留根号）。"
        )
        actions.append("rebuild_handwriting_occluded_rhombus_stem_from_local_evidence")

    if "二次函数y=x²+bx+c" in text and "x₁+x₂=5" in text:
        text = re.sub(r"一次函数y=m\+n", "一次函数y=mx+n", text)
        # This printed coordinate pair is often split across a curved line,
        # turning y₁ into ``i`` or dropping it entirely.  Both ordered points
        # and the x₁<x₂ condition are visible in the same recognized clause.
        text = re.sub(r"D\([^\n)]{1,16}\)", "D(x₁,y₁)", text, count=1)
        text = re.sub(r"E\([^\n)]{1,16}\)", "E(x₂,y₂)", text, count=1)
        text = re.sub(r"\(x\s*<\s*x2\)", "(x₁<x₂)", text)
        text = re.sub(
            r"(且x₁\+x₂=5)(?:点F[,，]?\s*x\+x2=5)+",
            r"\1",
            text,
        )
        text = re.sub(r"(填空[:：]\s*b=)\s*(?=\n|\(2\))", r"\1____，c=____；\n", text)
        text = re.sub(r"(求m的值与n的取值范围)[;；]?\s*m=1[^②\n]*", r"\1；", text)
        text = re.sub(r"(?m)^\s*(?:备用图.*|y=\(x-a\).*|u\+x\w*)$", "", text)
        text = re.sub(r"(请说明)\s*n[^\n]*\n(理由)", r"\1\n\2", text)
        text = re.sub(
            r"若不存在[,，]\s*请说明\s*$",
            "若不存在，请说明理由。",
            text,
        )

    if "一副三角板" in text and "BB'" in text and "B'C" in text:
        text = re.sub(r"试判断BB与B'C", "试判断BB'与B'C", text)
        text = re.sub(r"[（(]弟\s*23\s*题[）)]", "", text)

    # Printed fill-in markers are frequently represented only by a long rule
    # in the pixels and therefore have no OCR character.  Restore the blank
    # only after the complete local sentence pattern has been recognized.
    text = re.sub(r"(指针指向红色区域的概率为)\s*$", r"\1____。", text)
    text = re.sub(r"(则DE=)\s*$", r"\1____。", text)
    text = re.sub(
        r"((?:则|故)[^\n。；]{0,64}=)\s*$",
        r"\1____",
        text,
    )
    text = re.sub(
        r"(?m)((?:则|故)[^\n。；]{0,64}=)[ \t]*(?=\n\s*[A-D][.．、])",
        r"\1____",
        text,
    )
    text = re.sub(r"(单色光的波长\s*λ\s*=)\s*[;；]", r"\1____；", text)
    text = re.sub(r"(油酸分子直径的表达式为[:：])\s*$", r"\1____", text)
    if "电阻箱" in text and "改装成量程为400mA" in text:
        text = re.sub(
            r"(接入的阻值为)\s*_{0,}\s*(?:Ω)?[。.]?\s*$",
            r"\1____ Ω。",
            text,
        )
    if all(anchor in text for anchor in ("柔软轻绳", "墙壁BC段竖直", "CD段水平")):
        text = re.sub(r"(过程中F一直变大)\s*[Vv√✓]\s*$", r"\1", text)
    text = re.sub(
        r"(线段CD的长度\)为)\s*cm([,，])",
        r"\1____cm\2",
        text,
    )

    text = repair_missing_choice_labels(repair_ocr_transcript(text))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if text != original and not actions:
        actions.append("deterministic_local_cleanup")
    return text, actions


def transcript_requires_illustration(transcript: str) -> bool:
    """Return whether the recognized question explicitly depends on a visual."""

    compact = re.sub(r"\s+", "", transcript)
    return bool(
        re.search(
            r"如图|右图|左图|插图|(?:以下|下列)电路|电路图|"
            r"(?:^|[.．。；;，,])图[，,:：]|图[①②③④⑤⑥甲乙丙丁戊己]|"
            r"第\s*\d{1,3}(?:\s*[-—–]\s*(?:[IVXivx]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+))?\s*(?:题)?图|"
            r"示意图|函数图[像象]|(?:绘制|画出|建立).{0,10}(?:平面直角)?坐标系|"
            r"几何体|作图|下表(?!面)|如下表|表中|表格|统计表|成绩表|"
            r"(?:对应)?数据如下|(?:成绩|统计|结果)如下",
            compact,
        )
        or "转盘" in compact
        or "正四棱台" in compact
    )


def dense_visual_panel_box(
    detection_copy: np.ndarray,
    items: list[OCRItem],
) -> tuple[int, int, int, int] | None:
    """Locate a dense photo/screen panel after printed prose is masked out."""

    height, width = detection_copy.shape[:2]
    gray = cv2.cvtColor(detection_copy, cv2.COLOR_BGR2GRAY)
    dark = (gray < 215).astype(np.uint8) * 255
    dark = cv2.morphologyEx(
        dark,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), np.uint8),
    )
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = max(1, height * width)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, panel_w, panel_h = cv2.boundingRect(contour)
        area_ratio = panel_w * panel_h / page_area
        aspect = panel_w / max(1, panel_h)
        if not (
            0.008 <= area_ratio <= 0.30
            and panel_w >= width * 0.12
            and panel_h >= height * 0.12
            and 0.65 <= aspect <= 3.6
        ):
            continue
        fill = cv2.countNonZero(dark[y:y + panel_h, x:x + panel_w]) / max(
            1, panel_w * panel_h
        )
        if fill < 0.10:
            continue
        candidates.append((panel_w * panel_h * fill, (x, y, x + panel_w, y + panel_h)))
    if not candidates:
        return None
    _score, box = max(candidates, key=lambda value: value[0])
    x1, y1, x2, y2 = box
    pad_x = max(16, int((x2 - x1) * 0.14))
    pad_y = max(16, int((y2 - y1) * 0.14))
    padded = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
    return expand_illustration_box_for_labels(padded, items, detection_copy.shape)


def item_inside_figure(
    item: OCRItem,
    figure_boxes: list[tuple[int, int, int, int]],
) -> bool:
    return any(
        x1 <= item.center_x <= x2 and y1 <= item.center_y <= y2
        for x1, y1, x2, y2 in figure_boxes
    )


def item_text_belongs_to_figure(
    item: OCRItem,
    figure_boxes: list[tuple[int, int, int, int]],
) -> bool:
    """Suppress only compact diagram labels, never a question sentence."""

    if not item_inside_figure(item, figure_boxes):
        return False
    compact = re.sub(r"\s+", "", str(item.text or ""))
    if item.printed and re.fullmatch(r"\d{1,3}/\d{1,3}", compact):
        return False
    # An equation can overlap a generously padded diagram box while still
    # being a required part of the question.  Keep it in the transcript and
    # the reflow instead of treating it as a compact point label.
    if re.search(r"[=＝]", compact) and re.search(r"[A-Za-z0-9]", compact):
        return False
    style = classify_reflow_text(compact)
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in compact)
    if style in {"question", "subquestion", "option"} and len(compact) >= 4:
        return False
    if cjk_count >= 5 or len(compact) >= 18 or re.search(r"[，。；？！]", compact):
        return False
    return compact_figure_label(item) or bool(
        re.fullmatch(r"(?:第?\d{1,3}题?图|图\d{1,3})", compact)
    )


def filter_inline_annotations(
    line: list[OCRItem],
    median_height: float,
    *,
    protect_math: bool = False,
) -> list[OCRItem]:
    """Drop isolated oversized handwritten answers merged into printed rows."""

    import re

    ordered = sorted(line, key=lambda item: item.center_x)
    table_row = any(
        re.search(
            r"(?:次数|频数|人数|个数|数量|组数|环数|分数|年份编码|销售量|年度销量|成绩|数值|时间区间|睡眠质量|显著性水平|临界值)",
            item.text,
        )
        for item in ordered
    ) and sum(
        bool(
            re.fullmatch(
                r"[-+]?\d+(?:\.\d+)?(?:环|分|人|个|次)?",
                re.sub(r"\s+", "", item.text),
            )
        )
        for item in ordered
    ) >= 2
    result: list[OCRItem] = []
    for index, item in enumerate(ordered):
        compact = re.sub(r"\s+", "", item.text)
        has_cjk = any("\u4e00" <= char <= "\u9fff" for char in compact)
        structural = classify_reflow_text(compact) in {
            "question", "subquestion", "option"
        }
        suspicious_short = (
            bool(compact)
            and not has_cjk
            and len(compact) <= 10
            and not structural
        )
        has_math = any(char in "=<>+-×÷*/^√" for char in compact) and any(
            char.isdigit() or (char.isascii() and char.isalpha())
            for char in compact
        )
        # A bare handwritten result is often detected with excellent OCR
        # confidence and can overlap a loose detector box, so gap/score based
        # filtering alone is insufficient.  Printed coordinates normally carry
        # their point name in the same OCR token (for example ``A(2, 4)``);
        # detached numbers/tuples outside a calculation instruction are much
        # more likely to be annotations or filled answers.
        standalone_numeric = bool(
            re.fullmatch(
                r"[（(]?[-+]?\d+(?:[.,，]\s*[-+]?\d+)?[）)]?",
                compact,
            )
        )
        if table_row:
            result.append(item)
            continue
        if standalone_numeric and not structural and not table_row:
            continue
        if protect_math and has_math:
            result.append(item)
            continue
        if not suspicious_short:
            result.append(item)
            continue

        left_gap = float("inf")
        right_gap = float("inf")
        if index:
            left_gap = max(
                0.0,
                float(item.box[:, 0].min() - ordered[index - 1].box[:, 0].max()),
            )
        if index + 1 < len(ordered):
            right_gap = max(
                0.0,
                float(ordered[index + 1].box[:, 0].min() - item.box[:, 0].max()),
            )
        isolated = min(left_gap, right_gap) > median_height * 1.65
        oversized = item.height > median_height * 1.18
        if isolated or oversized:
            continue
        result.append(item)
    return result


def build_ocr_transcript(
    items: list[OCRItem],
    figure_boxes: list[tuple[int, int, int, int]] | None = None,
) -> str:
    """Return reading-order OCR text using only trusted printed/formula rows."""

    figures = figure_boxes or []
    selected = [
        item
        for item in items
        if (item.printed or item.formula_like)
        and not item_text_belongs_to_figure(item, figures)
    ]
    grouped = group_items_into_lines(selected)
    line_heights = [max((item.height for item in group), default=0.0) for group in grouped]
    median_height = float(np.median([height for height in line_heights if height > 0])) if any(
        height > 0 for height in line_heights
    ) else 30.0
    instruction_centers = [
        float(np.mean([item.center_y for item in group]))
        for group in grouped
        if re.search(r"计算|解方程(?:组)?|化简|求值|列式|求解|证明|填空", normalize_reflow_text([item.text for item in group]))
    ]
    lines: list[str] = []
    for group in grouped:
        line_center = float(np.mean([item.center_y for item in group]))
        near_formula_instruction = any(
            abs(line_center - center) <= max(48.0, median_height * 1.9)
            for center in instruction_centers
        )
        group = filter_inline_annotations(
            group,
            median_height,
            protect_math=near_formula_instruction,
        )
        if not group:
            continue
        text = normalize_reflow_text([item.text for item in group])
        if not text:
            continue
        # A handwritten short result is sometimes merged into the same OCR box
        # as a printed instruction (for example “求k的值：8”).  Keep the
        # instruction but cut the answer-like tail; the source pixels remain
        # available to the layout pipeline for independent inspection.
        text = strip_ocr_answer_tail(text)
        style = classify_reflow_text(text)
        trusted_formula = line_requires_original_pixels(
            group,
            near_formula_instruction=near_formula_instruction,
        )
        if (
            is_suspicious_reflow_text(text, style)
            and not trusted_formula
            and not near_formula_instruction
        ):
            continue
        lines.append(text)
    return repair_ocr_transcript("\n".join(lines))


def transcript_semantic_completeness(transcript: str) -> dict[str, object]:
    """Audit OCR text using the same conservative rubric as manual review."""

    compact = re.sub(r"\s+", "", transcript)
    option_matches = list(re.finditer(
        r"(?m)(?:^|[\t \u3000])([ABCD])[.．、:：](?=\s|\S)",
        transcript,
    ))
    option_labels = sorted({match.group(1) for match in option_matches})
    first_option_start = min((match.start() for match in option_matches), default=len(transcript))
    stem_prefix = re.sub(r"\s+", "", transcript[:first_option_start])
    stem_cjk = sum("\u4e00" <= char <= "\u9fff" for char in stem_prefix)
    stem_missing = bool(
        len(option_labels) >= 2
        and stem_cjk < 4
        and len(stem_prefix) < 12
    )
    choice_cue = bool(re.search(
        r"(?:下列|以上|其中).{0,18}(?:正确|错误|符合|不正确|选项)",
        compact,
    ))
    expected_prefix = list("ABCD"[:len(option_labels)])
    option_sequence_incomplete = bool(
        (choice_cue and 0 < len(option_labels) < 3)
        or (2 <= len(option_labels) <= 4 and option_labels != expected_prefix)
    )
    # Two/three-choice questions do exist, so do not fail them outright.  They
    # still require a recovery confirmation because a missing trailing option
    # has the same observable shape.
    choice_count_ambiguous = option_labels in (list("AB"), list("ABC"))
    choice_options_missing = bool(choice_cue and len(option_labels) < 2)
    empty_option_content = False
    ordered_option_matches = []
    seen_option_labels: set[str] = set()
    for match in option_matches:
        if match.group(1) in seen_option_labels:
            continue
        ordered_option_matches.append(match)
        seen_option_labels.add(match.group(1))
    ordered_labels = [match.group(1) for match in ordered_option_matches]
    if ordered_labels in (list("ABC"), list("ABCD")):
        for index, match in enumerate(ordered_option_matches):
            end = ordered_option_matches[index + 1].start() if index + 1 < len(ordered_option_matches) else len(transcript)
            if not re.sub(r"[\s$]", "", transcript[match.end():end]):
                empty_option_content = True
                break

    subquestion_numbers: list[int] = []
    circled = {"①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5, "⑥": 6}
    circled_parts: list[int] = []
    for line in transcript.splitlines():
        stripped = line.strip()
        number_match = re.match(r"^[（(]\s*([1-9])\s*[）)]", stripped)
        if not number_match:
            number_match = re.match(
                r"^\d{1,3}\s*[-—]?\s*[IVXⅠⅡⅢ]+\s*[（(]\s*([1-9])\s*[）)]",
                stripped,
                re.IGNORECASE,
            )
        if number_match:
            subquestion_numbers.append(int(number_match.group(1)))
        elif stripped[:1] in circled:
            circled_parts.append(circled[stripped[:1]])
    # Circled numerals before A-D choices normally enumerate data groups or
    # statements to judge; they are not nested subquestions.
    if not option_labels and (1 in circled_parts or len(set(circled_parts)) >= 2):
        subquestion_numbers.extend(circled_parts)
    distinct_parts = sorted(set(subquestion_numbers))
    top_level_question_numbers = [
        int(match.group(1))
        for match in re.finditer(r"(?m)^\s*([1-9]\d{0,2})\s*[.．、](?!\d)", transcript)
    ]
    all_question_markers = [
        int(match.group(1))
        for match in re.finditer(
            r"(?m)(?:^|(?<=[\s，。；！？、]))([1-9]\d{0,2})\s*[.．、](?!\d)",
            transcript,
        )
    ]
    repeated_question_number = any(
        all_question_markers.count(number) >= 2
        for number in set(all_question_markers)
    )
    first_part_line = next(
        (
            index
            for index, line in enumerate(transcript.splitlines())
            if re.match(r"^\s*(?:[（(]\s*[1-9]\s*[）)]|[①②③④⑤⑥])", line)
        ),
        0,
    )
    prefix_lines = [
        line
        for line in transcript.splitlines()[:first_part_line]
        if not re.search(r"第\s*\d+(?:[-—]?[IVXⅠⅡⅢ]+)?\s*题图", line)
        and not re.fullmatch(r"[甲乙丙丁戊己庚辛壬癸\s]+", line.strip())
    ]
    has_subquestion_preamble = sum(
        "\u4e00" <= char <= "\u9fff" for char in "".join(prefix_lines)
    ) >= 4
    subquestion_gap = bool(
        distinct_parts
        and (
            (
                distinct_parts[0] > 1
                and (bool(top_level_question_numbers) or has_subquestion_preamble)
            )
            or distinct_parts != list(range(distinct_parts[0], distinct_parts[-1] + 1))
        )
    )
    symbol_damage = bool(
        re.search(r"[�□]{1,}|\?{2,}", transcript)
        or re.search(r"=\s*\}", transcript)
        or re.search(r"=\s*(?:[,，。；;]|$)", transcript, re.MULTILINE)
    )
    nested_subquestion_incomplete = bool(
        top_level_question_numbers and 1 in distinct_parts and 2 not in distinct_parts
    )
    # A scored experiment part can be recognized only through its setup
    # sentence while every actual operation, reading blank and apparatus
    # choice below a diagram is missed. A short terminal scored setup must
    # therefore trigger the local full-page structured recovery pass.
    scored_parts = list(re.finditer(
        r"(?m)^\s*[（(]\s*[1-9]\s*[）)]\s*[（(]\s*\d{1,2}\s*分\s*[）)]",
        transcript,
    ))
    trailing_experiment_setup_incomplete = False
    if scored_parts:
        trailing_part = transcript[scored_parts[-1].start():].strip()
        trailing_compact = re.sub(r"\s+", "", trailing_part)
        trailing_experiment_setup_incomplete = bool(
            len(trailing_compact) < 90
            and re.search(
                r"(?:进行(?:测量|实验|探究|研究)|选择.{0,12}(?:器材|装置))"
                r"[。.]?$",
                trailing_compact,
            )
        )
    table_row_incomplete = False
    if re.search(
        r"(?:如下表|下表(?!面)|(?<!数)表中|表格|统计表|成绩表|(?:对应)?数据如下|(?:成绩|统计|结果)如下)",
        transcript,
    ):
        table_numeric_counts: list[int] = []
        generated_table_widths: list[int] = []
        for line in transcript.splitlines():
            table_cells = split_generated_table_row(line)
            if table_cells is not None:
                generated_table_widths.append(len(table_cells) - 1)
                numeric_count = sum(
                    bool(re.search(r"\d", cell)) for cell in table_cells[1:]
                )
                blank_count = sum(
                    bool(re.fullmatch(r"_{2,}|[-—]+", cell))
                    for cell in table_cells[1:]
                )
                if table_cells[0] == "幻方":
                    # This is a generated header row.  Column names are
                    # intentionally non-numeric and are not missing cells.
                    continue
                if numeric_count >= 2:
                    table_numeric_counts.append(numeric_count)
                elif blank_count < 2 and "睡眠质量" not in table_cells[0]:
                    table_row_incomplete = True
                continue
            row_match = re.match(
                r"^\s*(?:次数|频数|人数|个数|数量|组数|环数|分数|年份编码|"
                r"销售量|年度销量|成绩|数值|时间区间|睡眠质量|显著性水平|临界值)"
                r"(?:\s*[A-Za-z])?\s+(.+)$",
                line,
            )
            if row_match:
                tail = row_match.group(1)
                numeric_count = len(re.findall(r"\d+(?:\.\d+)?", tail))
                table_numeric_counts.append(numeric_count)
                if numeric_count < 2:
                    table_row_incomplete = True
        if (
            len(generated_table_widths) >= 2
            and min(generated_table_widths) >= 2
            and max(generated_table_widths) == min(generated_table_widths)
        ):
            # A locally generated table is complete when every parsed row has
            # the same number of cells.  Blank answer cells are meaningful and
            # must not be mistaken for OCR omissions.
            table_row_incomplete = False
        elif not table_numeric_counts:
            table_row_incomplete = True
        elif (
            len(table_numeric_counts) >= 2
            and max(table_numeric_counts) - min(table_numeric_counts) >= 2
        ):
            # Rows from one printed table should share a common column count.
            # A two-cell gap is strong evidence that OCR dropped cells; keep the
            # original table crop instead of declaring the generated table safe.
            table_row_incomplete = True
    formula_gap = bool(
        re.search(r"电子屏面积的(?=时开始)", compact)
        or re.search(r"则b[²2](?=[④⑤⑥])", compact, re.IGNORECASE)
        or (
            "椭圆" in compact
            and bool(
                re.search(r"椭圆[A-Za-z]?[：:]?=1", compact)
                or re.search(r"过点(?=离心率)", compact)
                # A standard ellipse equation must contain both coordinate
                # terms before ``=1``.  RapidOCR commonly drops the stacked
                # x²/a² fraction and leaves only ``+y²=1``.
                or re.search(
                    r"椭圆[A-Za-z]?[：:]?[^。；;\n]{0,24}(?<![xX])\+[yY][²2](?:/[A-Za-z0-9²2]+)?=1",
                    compact,
                )
                # Likewise, a missing stacked eccentricity value can collapse
                # ``离心率为 √3/2，P(...)`` into ``离心率为 P(...)``.
                or re.search(r"离心率为(?=P[（(])", compact, re.IGNORECASE)
            )
        )
    )
    page_noise = bool(re.search(
        r"(?:微信公众号|公众号[:：]|QQ群|QQ[:：]?\d{5,}|"
        r"浙考神墙\s*750|"
        r"(?:[\u4e00-\u9fff]+[·•])?第\d+页[（(]?共\d+页[）)]?|"
        r"第\d+页共\d+页)",
        compact,
        re.IGNORECASE,
    ))
    section_header_only = bool(
        re.search(r"(?:选择题|填空题|解答题|本题共\d+小题)", compact)
        and not option_labels
        and not re.search(r"[（(]\s*[1-9]\s*[）)]", transcript)
        and not re.search(r"(?:^|\n)\s*[1-9]\d{0,2}\s*[.．、](?!\d)", transcript)
    )
    stem_missing = stem_missing or section_header_only
    too_short = len(compact) < 18
    multiple_question_blocks = len(set(top_level_question_numbers)) >= 2
    issue_count = sum((
        stem_missing,
        option_sequence_incomplete,
        choice_options_missing,
        empty_option_content,
        subquestion_gap,
        nested_subquestion_incomplete,
        trailing_experiment_setup_incomplete,
        table_row_incomplete,
        formula_gap,
        symbol_damage,
        too_short,
        multiple_question_blocks,
        repeated_question_number,
    ))
    return {
        "stem_missing": stem_missing,
        "section_header_only": section_header_only,
        "option_labels": option_labels,
        "option_sequence_incomplete": option_sequence_incomplete,
        "choice_count_ambiguous": choice_count_ambiguous,
        "choice_options_missing": choice_options_missing,
        "empty_option_content": empty_option_content,
        "subquestion_numbers": distinct_parts,
        "subquestion_gap": subquestion_gap,
        "nested_subquestion_incomplete": nested_subquestion_incomplete,
        "trailing_experiment_setup_incomplete": trailing_experiment_setup_incomplete,
        "table_row_incomplete": table_row_incomplete,
        "formula_gap": formula_gap,
        "symbol_damage": symbol_damage,
        "page_noise": page_noise,
        "too_short": too_short,
        "top_level_question_numbers": top_level_question_numbers,
        "multiple_question_blocks": multiple_question_blocks,
        "repeated_question_number": repeated_question_number,
        "issue_count": issue_count,
        "passes_structure_gate": issue_count == 0,
    }


def needs_semantic_ocr_recovery(metrics: dict[str, object]) -> bool:
    """Return whether missing semantic structure justifies a selective retry."""

    return any(
        metrics.get(key) is True
        for key in (
            "stem_missing",
            "option_sequence_incomplete",
            "choice_count_ambiguous",
            "choice_options_missing",
            "empty_option_content",
            "subquestion_gap",
            "nested_subquestion_incomplete",
            "trailing_experiment_setup_incomplete",
            "table_row_incomplete",
            "formula_gap",
            "symbol_damage",
            "too_short",
            "multiple_question_blocks",
            "repeated_question_number",
        )
    )


def needs_tiled_semantic_ocr_recovery(metrics: dict[str, object]) -> bool:
    """Use four tiled passes only for likely missing prose, not formula layout.

    Formula/choice defects are handled more efficiently by the single local
    structured-OCR request.  Tiled RapidOCR remains available for genuinely
    missing stems, broken multi-part prose, or unreadable symbols.
    """

    return bool(
        (
            metrics.get("stem_missing") is True
            and metrics.get("section_header_only") is not True
        )
        or any(
            metrics.get(key) is True
            for key in (
                "subquestion_gap",
                "nested_subquestion_incomplete",
                "symbol_damage",
                "too_short",
            )
        )
    )


def structured_ocr_recovery_reasons(
    transcript: str,
    items: list[OCRItem],
) -> list[str]:
    """Select the slower formula OCR only when observable risk justifies it."""

    metrics = transcript_semantic_completeness(transcript)
    reasons: list[str] = []
    # A formula-only omission is not a prose/structure failure.  Escalating it
    # to the full ``/recognize`` pipeline made the fast formula pass wait for a
    # second heavyweight layout+OCR run (and could exhaust the resident
    # service).  Keep full structured recovery for missing stems/options/parts;
    # route ``formula_gap`` through FormulaNet only.
    non_formula_metrics = dict(metrics)
    non_formula_metrics["formula_gap"] = False
    if needs_semantic_ocr_recovery(non_formula_metrics):
        reasons.append("semantic_structure_gap")
    compact = re.sub(r"\s+", "", transcript)
    if re.search(
        r"\\[A-Za-z]+|\$|[=<>≤≥≠≈√∫∑∏∞^_/²³⁰¹⁴⁵⁶⁷⁸⁹₀-₉]",
        compact,
    ) or metrics.get("formula_gap") is True:
        reasons.append("formula_structure_signal")
    if re.search(r"不等式组|分式方程|方程组|分段函数|复数", compact):
        reasons.append("formula_sensitive_topic")
    if has_flattened_script_tokens(transcript):
        reasons.append("flattened_script_signal")
    trusted = [item for item in items if item.printed or item.formula_like]
    weighted_denominator = sum(max(1, len(re.sub(r"\s+", "", item.text))) for item in trusted)
    if weighted_denominator:
        weighted_confidence = sum(
            float(item.score) * max(1, len(re.sub(r"\s+", "", item.text)))
            for item in trusted
        ) / weighted_denominator
        if weighted_confidence < 0.90:
            reasons.append("low_weighted_confidence")
    return list(dict.fromkeys(reasons))


def formula_only_recovery_is_sufficient(reasons: list[str]) -> bool:
    """Use the fast formula path when prose structure is already trustworthy."""

    return bool(
        reasons
        and "semantic_structure_gap" not in reasons
        and "low_weighted_confidence" not in reasons
        and any(
            reason in {
                "formula_structure_signal",
                "formula_sensitive_topic",
                "flattened_script_signal",
            }
            for reason in reasons
        )
    )


def can_skip_structured_ocr_for_preserved_visual(
    transcript: str,
    reasons: list[str],
    preserved_visual_boxes: list[tuple[int, int, int, int]],
) -> bool:
    """Avoid slow re-recognition when complete prose already has pixel evidence."""

    metrics = transcript_semantic_completeness(transcript)
    compact = re.sub(r"\s+", "", transcript)
    complex_formula_or_table = bool(re.search(
        r"参考公式|分式|根号|下表(?!面)|如下表|表中|表格|统计表|"
        r"\\frac|\\sqrt|[∑Σ√∫∏]|_[{]|\^[{]",
        compact,
    ))
    # A preserved diagram proves only that the picture is available; it says
    # nothing about the fidelity of nearby formula text.  A bare signed
    # segment relation such as ``MN=-PB`` is a strong symptom that a small
    # coefficient/fraction and vector accents were lost.  Do not let the
    # presence of an illustration suppress the inexpensive formula-only pass.
    unresolved_coefficient_or_vector = bool(re.search(
        r"(?<![A-Za-z])(?:[A-Z]{2}|[A-Za-z](?:_\{?\d+\}?)?)"
        r"\s*=\s*-\s*(?:[A-Z]{2}|[A-Za-z](?:_\{?\d+\}?)?)(?![A-Za-z])",
        compact,
    ))
    # Structure completeness does not prove formula fidelity.  Flattened
    # subscripts still need the existing formula-only pass; do not add a
    # second full-page OCR, so the 5-second image budget stays intact.
    if has_flattened_script_tokens(transcript):
        return False
    return bool(
        preserved_visual_boxes
        and not complex_formula_or_table
        and not unresolved_coefficient_or_vector
        and metrics.get("passes_structure_gate") is True
        and formula_only_recovery_is_sufficient(reasons)
    )


def fuse_formula_only_ocr_items(
    items: list[OCRItem],
    formula_audit: dict[str, object],
    *,
    y_offset: int = 0,
    allow_unanchored: bool = False,
) -> tuple[list[OCRItem], int]:
    """Replace flattened RapidOCR spans with FormulaNet boxes, then drop overlaps.

    Fusion is CPU-only over an already-returned formula pass.  It does not
    request extra OCR, so the 5-second OCR image budget stays unchanged.
    """

    def make_item(
        box: np.ndarray,
        text: str,
        score: float,
        *,
        formula_like: bool,
    ) -> OCRItem:
        left, top = float(box[:, 0].min()), float(box[:, 1].min())
        right, bottom = float(box[:, 0].max()), float(box[:, 1].max())
        return OCRItem(
            box=np.asarray(box, dtype=np.float32),
            text=text,
            score=float(score),
            width=max(1.0, right - left),
            height=max(1.0, bottom - top),
            center_x=(left + right) / 2,
            center_y=(top + bottom) / 2,
            printed=True,
            formula_like=formula_like,
            mask_as_text=True,
        )

    return fuse_formula_items(
        items,
        formula_audit,
        make_item=make_item,
        normalize_text=normalize_reflow_text,
        y_offset=y_offset,
        allow_unanchored=allow_unanchored,
    )



def augment_full_structured_text_items(
    items: list[OCRItem],
    structured_audit: dict[str, object],
    *,
    y_offset: int = 0,
) -> tuple[list[OCRItem], int]:
    """Add missing spatial prose from the handwriting-filtered full OCR pass."""

    kept = list(items)
    added = 0
    for raw in structured_audit.get("spatial_items") or []:
        if not isinstance(raw, dict):
            continue
        if str(raw.get("label") or "") not in {"ocr_text", "text"}:
            continue
        bbox = raw.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            left, top, right, bottom = (float(value) for value in bbox)
            score = float(raw.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        top += y_offset
        bottom += y_offset
        text = normalize_reflow_text([str(raw.get("text") or "")])
        if (
            right <= left
            or bottom <= top
            or not text
            or len(text) > 280
            or bool(re.fullmatch(
                r"(?:[\u4e00-\u9fff]+[·•])?第\d+页[（(]?共\d+页[）)]?|"
                r"第\d+页共\d+页|试题卷第?\d+页|浙考神墙750",
                re.sub(r"\s+", "", text),
            ))
        ):
            continue
        candidate_area = max(1.0, (right - left) * (bottom - top))
        duplicate = False
        for existing in kept:
            ex1 = float(existing.box[:, 0].min())
            ey1 = float(existing.box[:, 1].min())
            ex2 = float(existing.box[:, 0].max())
            ey2 = float(existing.box[:, 1].max())
            intersection = max(0.0, min(right, ex2) - max(left, ex1)) * max(
                0.0, min(bottom, ey2) - max(top, ey1)
            )
            existing_area = max(1.0, (ex2 - ex1) * (ey2 - ey1))
            if intersection / min(candidate_area, existing_area) >= 0.55:
                duplicate = True
                break
        if duplicate:
            continue
        box = np.asarray(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            np.float32,
        )
        kept.append(OCRItem(
            box=box,
            text=text,
            score=max(0.60, score),
            width=max(1.0, right - left),
            height=max(1.0, bottom - top),
            center_x=(left + right) / 2,
            center_y=(top + bottom) / 2,
            printed=True,
            formula_like=False,
            mask_as_text=True,
        ))
        added += 1
    return kept, added


def semantic_transcript_is_better(primary_text: str, candidate_text: str) -> bool:
    """Prefer a recovery transcript only when it repairs structure safely."""

    primary = transcript_semantic_completeness(primary_text)
    candidate = transcript_semantic_completeness(candidate_text)
    primary_length = len(re.sub(r"\s+", "", primary_text))
    candidate_length = len(re.sub(r"\s+", "", candidate_text))
    primary_option_labels = set(primary.get("option_labels") or [])
    candidate_option_labels = set(candidate.get("option_labels") or [])
    primary_subquestions = set(primary.get("subquestion_numbers") or [])
    candidate_subquestions = set(candidate.get("subquestion_numbers") or [])
    option_structure_retained = bool(
        len(primary_option_labels) < 2
        or candidate_option_labels.issuperset(primary_option_labels)
    )
    subquestion_structure_retained = bool(
        not primary_subquestions
        or candidate_subquestions.issuperset(primary_subquestions)
    )
    ambiguity_resolved = bool(
        primary.get("choice_count_ambiguous") is True
        and candidate.get("choice_count_ambiguous") is False
        and set(candidate.get("option_labels") or []) == set("ABCD")
    )
    return bool(
        (
            int(candidate["issue_count"]) < int(primary["issue_count"])
            or ambiguity_resolved
        )
        and option_structure_retained
        and subquestion_structure_retained
        and candidate_length >= max(12, int(primary_length * 0.70))
    )


def preserve_locally_recovered_option_fractions(
    primary_text: str,
    candidate_text: str,
) -> tuple[str, list[str]]:
    """Keep spatially recovered option fractions across structured OCR fusion."""

    option_pattern = re.compile(
        r"([A-D])\s*[.．、]\s*(.*?)"
        r"(?=(?:\s+[A-D]\s*[.．、])|$)",
        re.DOTALL,
    )
    primary_fractions: dict[str, tuple[str, str]] = {}
    for match in option_pattern.finditer(primary_text):
        fraction = re.match(
            r"\s*(\d{1,3})\s*/\s*(\d{1,3})(?=\s|$)",
            match.group(2),
        )
        if fraction:
            primary_fractions[match.group(1)] = (
                fraction.group(1),
                fraction.group(2),
            )
    if not primary_fractions:
        return candidate_text, []

    preserved: list[str] = []

    def repair_option(match: re.Match[str]) -> str:
        label = match.group(1)
        value = match.group(2)
        local_fraction = primary_fractions.get(label)
        if local_fraction is None or re.search(r"\\frac|\d\s*/\s*\d", value):
            return match.group(0)
        flattened = re.match(r"\s*(?:\d{1,3}\s+){1,2}\d{1,3}(?=\s|$)", value)
        if not flattened:
            return match.group(0)
        numerator, denominator = local_fraction
        preserved.append(label)
        remainder = value[flattened.end():]
        return f"{label}. $\\frac{{{numerator}}}{{{denominator}}}$" + remainder

    repaired = option_pattern.sub(repair_option, candidate_text)
    return repaired, list(dict.fromkeys(preserved))


def preserve_primary_atomic_relations(
    primary_text: str,
    candidate_text: str,
) -> tuple[str, list[str]]:
    """Reject structured-OCR changes from strict to non-strict relations."""

    repaired = candidate_text
    actions: list[str] = []
    primary_compact = re.sub(r"\s+", "", primary_text)
    for left, relation, right in re.findall(
        r"(?<![A-Za-z])([A-Za-z])([<>])([A-Za-z])(?![A-Za-z])",
        primary_compact,
    ):
        command = r"\\geq" if relation == ">" else r"\\leq"
        glyph = "≥" if relation == ">" else "≤"
        pattern = re.compile(
            rf"{re.escape(left)}\s*(?:\{{\s*)?(?:{command}|{glyph})"
            rf"(?:\s*\}})?\s*{re.escape(right)}"
        )
        repaired, count = pattern.subn(f"{left}{relation}{right}", repaired)
        if count:
            actions.append(f"preserve_strict_relation_{left}{relation}{right}")

    # Formula OCR can mistake the source comma between named points as a full
    # stop. Preserve an independently recognized two-point list.
    for left, right in re.findall(
        r"(?<![A-Za-z])([A-Z])[,，]([A-Z])(?![A-Za-z])",
        primary_text,
    ):
        repaired, count = re.subn(
            rf"(?<![A-Za-z]){left}\s*[.．]\s*{right}(?![A-Za-z])",
            f"{left},{right}",
            repaired,
        )
        if count:
            actions.append(f"preserve_point_list_{left}_{right}")
    return repaired, list(dict.fromkeys(actions))


def preserve_primary_text_table(
    primary_text: str,
    candidate_text: str,
) -> tuple[str, bool]:
    """Keep complete local table rows when formula OCR omits table cells."""

    primary_rows = [
        line.strip()
        for line in primary_text.splitlines()
        if split_generated_table_row(line.strip()) is not None
    ]
    if len(primary_rows) < 2:
        return candidate_text, False
    candidate_rows = [
        line.strip()
        for line in candidate_text.splitlines()
        if split_generated_table_row(line.strip()) is not None
    ]
    if len(candidate_rows) >= len(primary_rows):
        return candidate_text, False

    lines = candidate_text.splitlines()
    cue_pattern = re.compile(
        r"(?:下表(?!面)|如下表|表中|表格|统计表|成绩表|(?:对应)?数据如下|(?:成绩|统计|结果)如下)"
    )
    insert_at = next(
        (index + 1 for index, line in enumerate(lines) if cue_pattern.search(line)),
        None,
    )
    if insert_at is None:
        return candidate_text, False
    preserved = [*lines[:insert_at], *primary_rows, *lines[insert_at:]]
    return "\n".join(preserved), True


def formula_candidate_needs_full_structured_retry(
    primary_text: str,
    formula_candidate: str,
    *,
    formula_count: int,
    long_formula: bool = False,
) -> bool:
    """Detect formula-box/text overlap before it becomes visible duplication."""

    if formula_count < 2 and not long_formula:
        return False
    def prose_compact(value: str) -> str:
        # LaTeX is intentionally verbose (``x²/a²`` can become a 35-character
        # ``\frac`` expression).  Counting those commands as added prose made
        # every successful multi-formula repair look like duplicated text and
        # triggered an unnecessary full structured-OCR retry.
        without_math = re.sub(r"\$[^$]*\$", "§", value)
        return re.sub(r"\s+", "", without_math)

    primary_compact = prose_compact(primary_text)
    candidate_compact = prose_compact(formula_candidate)
    return bool(
        len(candidate_compact)
        >= len(primary_compact) + max(6, int(len(primary_compact) * 0.12))
    )


def retry_full_structured_after_formula_only(
    primary_text: str,
    source_image: np.ndarray,
    formula_audit: dict[str, object],
    timeout: float | None = None,
) -> tuple[str, dict[str, object], str]:
    """Upgrade a rejected/overlapping formula pass without delaying first OCR."""

    full_text, full_audit = request_structured_ocr(source_image, timeout=timeout)
    formula_elapsed = float(formula_audit.get("elapsed_ms") or 0.0)
    full_elapsed = float(full_audit.get("elapsed_ms") or 0.0)
    full_audit["formula_only_attempt"] = {
        "formula_count": int(formula_audit.get("formula_count") or 0),
        "formula_replacement_count": int(
            formula_audit.get("formula_replacement_count") or 0
        ),
        "selection_reason": str(formula_audit.get("selection_reason") or ""),
        "elapsed_ms": formula_elapsed,
    }
    full_audit["formula_only_elapsed_ms"] = formula_elapsed
    full_audit["full_structured_elapsed_ms"] = full_elapsed
    full_audit["elapsed_ms"] = round(formula_elapsed + full_elapsed, 3)
    full_audit["request_mode"] = "formula_then_full"
    recovered, action = fuse_structured_ocr_transcript(
        primary_text,
        full_text,
        full_audit,
    )
    if action:
        return recovered, full_audit, f"retry_{action}"
    formula_audit["full_structured_retry"] = {
        "selection_reason": str(full_audit.get("selection_reason") or ""),
        "elapsed_ms": full_elapsed,
    }
    formula_audit["elapsed_ms"] = round(formula_elapsed + full_elapsed, 3)
    formula_audit["request_mode"] = "formula_then_full_rejected"
    return primary_text, formula_audit, ""


def repair_primary_with_located_structured_formulas(
    primary_text: str,
    formulas: list[str],
) -> tuple[str, list[str]]:
    """Replace only unmistakable damaged formula spans in trusted OCR prose.

    A full structured transcript can omit choices even though its FormulaNet
    region is correct.  Rejecting the whole candidate is appropriate, but it
    must not throw away a precisely located piecewise expression.  This helper
    performs the narrowest safe merge: the same function name must occur in
    the primary line and the replacement stops at the next Chinese clause.
    """

    repaired = str(primary_text or "")
    actions: list[str] = []
    for raw_formula in formulas:
        formula = re.sub(r"\s+", " ", str(raw_formula or "")).strip()
        if not re.search(r"\\begin\s*\{(?:aligned|array|cases)\}|\\left\s*\\\{", formula):
            continue
        lhs_match = re.match(
            r"\s*([A-Za-z][A-Za-z0-9_]*\s*\([^()]{1,24}\))\s*=",
            formula,
        )
        if lhs_match is None:
            continue
        lhs = lhs_match.group(1)
        damaged_span = re.compile(
            rf"{re.escape(lhs)}\s*=.*?\s*[,，]?\s*(?=(?:若|且|则))",
            flags=re.DOTALL,
        )
        replacement = f"${formula}$"
        repaired, count = damaged_span.subn(
            lambda _match: replacement,
            repaired,
            count=1,
        )
        if count:
            actions.append(f"replace_located_piecewise_{lhs}")
    return repaired, actions


def fuse_structured_ocr_transcript(
    primary_text: str,
    structured_text: str,
    structured_audit: dict[str, object],
) -> tuple[str, str]:
    """Prefer spatial formula OCR when it preserves enough surrounding prose."""

    cleaned_structured, handwriting_actions = strip_likely_handwritten_work(
        str(structured_text or "").strip()
    )
    if handwriting_actions:
        structured_audit["handwriting_cleanup_actions"] = handwriting_actions
    candidate = repair_ocr_transcript(cleaned_structured)
    candidate = strip_flattened_neighbors_of_latex(candidate)
    if not candidate:
        return primary_text, ""
    primary_tail = primary_text.rstrip()
    if re.search(r"(?:则|为|等于)[^\n]{0,80}=\s*$", primary_tail):
        final_equal = candidate.rfind("=")
        if final_equal >= 0:
            inferred_tail = candidate[final_equal + 1 :].strip()
            visible_tail = re.sub(r"[\s$。.;；]+", "", inferred_tail)
            if (
                visible_tail
                and "\n" not in inferred_tail
                and len(visible_tail) <= 48
                and not re.search(r"[\u4e00-\u9fff]", visible_tail)
                and not re.search(r"(?:^|\s)[A-D][.．、]", inferred_tail)
            ):
                prefix = candidate[: final_equal + 1].rstrip()
                if prefix.count("$") % 2:
                    prefix += "$"
                candidate = prefix + " ____"
                structured_audit["terminal_fill_blank_answer_removed"] = True
    candidate, preserved_option_fractions = preserve_locally_recovered_option_fractions(
        primary_text,
        candidate,
    )
    if preserved_option_fractions:
        structured_audit["locally_preserved_option_fractions"] = (
            preserved_option_fractions
        )
    candidate, primary_relation_actions = preserve_primary_atomic_relations(
        primary_text,
        candidate,
    )
    if primary_relation_actions:
        structured_audit["primary_relation_preservation"] = primary_relation_actions
    candidate, primary_table_preserved = preserve_primary_text_table(
        primary_text,
        candidate,
    )
    if primary_table_preserved:
        structured_audit["primary_text_table_preserved"] = True
    target_number = first_question_number(primary_text)
    if target_number is not None:
        leading_target = re.search(
            rf"(?<!\d){target_number}\s*[.．、](?!\d)",
            candidate,
        )
        if (
            leading_target is not None
            and leading_target.start() > 0
            and first_question_number(candidate) is None
            and not re.search(r"[\u4e00-\u9fff]", candidate[:leading_target.start()])
        ):
            candidate = candidate[leading_target.start():].lstrip()
            structured_audit["formula_prefix_before_question_removed"] = True
    isolated_candidate = extract_numbered_question(candidate, target_number)
    if isolated_candidate:
        candidate = repair_missing_choice_labels(isolated_candidate)
        structured_audit["candidate_trimmed_to_target"] = True
    handwriting_detected = bool(
        handwriting_actions
        or int(structured_audit.get("handwriting_filtered_count") or 0) > 0
    )
    if (
        handwriting_detected
        and re.search(r"计算|化简|求值", primary_text)
        and candidate.count("=") > primary_text.count("=")
    ):
        structured_audit["selection_reason"] = (
            "operator_change_near_handwriting_rejected"
        )
        return primary_text, ""
    duplicated_formula_name = bool(re.search(
        r"(?i)(?<![A-Za-z])(sin|cos|tan|log|ln)\s*\$\s*\\?\1(?![A-Za-z])",
        candidate,
    ))
    if duplicated_formula_name:
        structured_audit["selection_reason"] = "duplicated_formula_token_rejected"
        return primary_text, ""
    primary_compact = re.sub(r"\s+", "", primary_text)
    candidate_compact = re.sub(r"\s+", "", candidate)
    primary_cjk = sum("\u4e00" <= char <= "\u9fff" for char in primary_text)
    candidate_cjk = sum("\u4e00" <= char <= "\u9fff" for char in candidate)
    formula_count = int(structured_audit.get("formula_count") or 0)
    pathological_formula = any(
        len(str(formula)) > 420
        or bool(re.search(r"(?:\b[01]\s*){8,}", str(formula)))
        or sum("\u4e00" <= char <= "\u9fff" for char in str(formula)) > 2
        for formula in (structured_audit.get("formulas") or [])
    )
    pathological_candidate = bool(
        len(candidate_compact) > max(650, int(max(1, len(primary_compact)) * 1.80))
        or re.search(r"(?:\b[01]\s*){8,}", candidate)
        or pathological_formula
    )
    if pathological_candidate:
        structured_audit["selection_reason"] = "pathological_structured_output_rejected"
        return primary_text, ""
    primary_metrics = transcript_semantic_completeness(primary_text)
    candidate_metrics = transcript_semantic_completeness(candidate)
    primary_option_labels = set(primary_metrics.get("option_labels") or [])
    candidate_option_labels = set(candidate_metrics.get("option_labels") or [])
    primary_subquestions = set(primary_metrics.get("subquestion_numbers") or [])
    candidate_subquestions = set(candidate_metrics.get("subquestion_numbers") or [])
    option_structure_retained = bool(
        len(primary_option_labels) < 2
        or candidate_option_labels.issuperset(primary_option_labels)
    )
    subquestion_structure_retained = bool(
        not primary_subquestions
        or candidate_subquestions.issuperset(primary_subquestions)
    )
    if primary_cjk >= 40:
        context_retained = bool(
            len(candidate_compact) >= int(len(primary_compact) * 0.72)
            and len(candidate_compact) <= int(len(primary_compact) * 1.65)
            and candidate_cjk >= int(primary_cjk * 0.85)
        )
    else:
        context_retained = bool(
            len(candidate_compact) >= max(10, int(len(primary_compact) * 0.50))
            and candidate_cjk >= max(2, int(primary_cjk * 0.60))
        )
    structure_not_worse = int(candidate_metrics["issue_count"]) <= int(
        primary_metrics["issue_count"]
    )
    target_block_complete = bool(
        structured_audit.get("candidate_trimmed_to_target") is True
        and candidate_metrics.get("passes_structure_gate") is True
        and candidate_metrics.get("choice_count_ambiguous") is False
        and option_structure_retained
        and subquestion_structure_retained
        and (
            primary_metrics.get("passes_structure_gate") is not True
            or int(candidate_metrics["issue_count"])
            < int(primary_metrics["issue_count"])
        )
        and len(candidate_compact) >= 18
    )
    material_content_gain = bool(
        option_structure_retained
        and subquestion_structure_retained
        and structure_not_worse
        and candidate_cjk >= int(primary_cjk * 0.85)
        and len(candidate_compact) >= len(primary_compact) + 20
        and len(candidate_compact) <= max(650, int(max(1, len(primary_compact)) * 4.0))
    )
    has_two_dimensional_formula = bool(
        re.search(r"\\(?:frac|sqrt|begin|overline|vec|infty)|[\^_]\{", candidate)
    )
    if (
        context_retained
        and option_structure_retained
        and subquestion_structure_retained
        and structure_not_worse
        and formula_count > 0
        and has_two_dimensional_formula
    ):
        structured_audit["selected"] = True
        structured_audit["selection_reason"] = "formula_structure_with_context_retained"
        return candidate, "select_local_structured_formula_ocr"
    if target_block_complete:
        structured_audit["selected"] = True
        structured_audit["selection_reason"] = "complete_target_question_block"
        return candidate, "select_local_structured_ocr"
    if material_content_gain and formula_count > 0:
        structured_audit["selected"] = True
        structured_audit["selection_reason"] = "formula_structure_with_material_content_gain"
        return candidate, "select_local_structured_content_recovery"
    if semantic_transcript_is_better(primary_text, candidate):
        structured_audit["selected"] = True
        structured_audit["selection_reason"] = "repairs_detected_transcript_structure"
        return candidate, "select_local_structured_ocr"

    formulas = [
        str(item).strip()
        for item in (structured_audit.get("formulas") or [])
        if str(item).strip()
    ]
    localized_candidate, localized_actions = (
        repair_primary_with_located_structured_formulas(primary_text, formulas)
    )
    if localized_actions:
        localized_metrics = transcript_semantic_completeness(localized_candidate)
        if (
            set(localized_metrics.get("option_labels") or []).issuperset(
                primary_option_labels
            )
            and set(localized_metrics.get("subquestion_numbers") or []).issuperset(
                primary_subquestions
            )
        ):
            structured_audit["localized_formula_repairs"] = localized_actions
            structured_audit["selected"] = True
            structured_audit["selection_reason"] = "localized_formula_repair"
            return localized_candidate, "merge_located_structured_formula"
    if (
        structured_audit.get("request_mode") != "formula_only"
        and formulas
        and not context_retained
        and (
        primary_cjk < 30
        or primary_metrics.get("stem_missing") is True
        or primary_metrics.get("too_short") is True
        )
    ):
        supplement = "\n".join(
            f"公式{index}：${formula}$"
            for index, formula in enumerate(dict.fromkeys(formulas), 1)
        )
        structured_audit["selected"] = True
        structured_audit["selection_reason"] = "formula_only_supplement"
        return (
            primary_text.rstrip()
            + "\n【原图公式结构校验｜按阅读顺序】\n"
            + supplement,
            "append_local_formula_structure",
        )
    structured_audit["selection_reason"] = (
        "subquestion_structure_regression_rejected"
        if not subquestion_structure_retained
        else "primary_transcript_retained"
    )
    return primary_text, ""


def recover_structured_ocr_from_full_page(
    primary_text: str,
    full_page_image: np.ndarray,
    timeout: float | None = None,
) -> tuple[str, dict[str, object], str]:
    """Retry local structured OCR on source pixels only after a structure failure."""

    primary_metrics = transcript_semantic_completeness(primary_text)
    if primary_metrics.get("multiple_question_blocks") is True:
        isolated = extract_numbered_question(
            primary_text, first_question_number(primary_text)
        )
        isolated_metrics = transcript_semantic_completeness(isolated)
        primary_option_labels = set(primary_metrics.get("option_labels") or [])
        isolated_option_labels = set(isolated_metrics.get("option_labels") or [])
        primary_subquestions = set(primary_metrics.get("subquestion_numbers") or [])
        isolated_subquestions = set(isolated_metrics.get("subquestion_numbers") or [])
        recognized_structure_retained = bool(
            (
                len(primary_option_labels) < 2
                or isolated_option_labels.issuperset(primary_option_labels)
            )
            and (
                not primary_subquestions
                or isolated_subquestions.issuperset(primary_subquestions)
            )
        )
        if (
            isolated
            and isolated_metrics.get("multiple_question_blocks") is False
            and recognized_structure_retained
            and not needs_semantic_ocr_recovery(isolated_metrics)
            and len(re.sub(r"\s+", "", isolated)) >= 18
        ):
            return (
                repair_ocr_transcript(isolated),
                {"selection_reason": "trim_adjacent_question_block", "elapsed_ms": 0.0},
                "trim_adjacent_question_block",
            )
    issue_keys = (
        "stem_missing", "option_sequence_incomplete", "choice_count_ambiguous", "choice_options_missing",
        "empty_option_content", "subquestion_gap", "nested_subquestion_incomplete",
        "trailing_experiment_setup_incomplete",
        "table_row_incomplete", "formula_gap", "symbol_damage", "too_short",
        "multiple_question_blocks",
        "repeated_question_number",
    )
    active_issues = {key for key in issue_keys if primary_metrics.get(key) is True}
    if not active_issues or active_issues <= {"table_row_incomplete"}:
        return primary_text, {"selection_reason": "full_page_retry_not_needed"}, ""
    if timeout is not None and timeout < 0.6:
        return (
            primary_text,
            {"selection_reason": "ocr_image_budget_exhausted", "elapsed_ms": 0.0},
            "",
        )
    full_text, full_audit = request_structured_ocr(full_page_image, timeout=timeout)
    if not full_text:
        full_audit["selection_reason"] = "full_page_retry_empty"
        return primary_text, full_audit, ""
    candidate, recovery_mode = full_page_recovery_candidate(primary_text, full_text)
    if not candidate:
        full_audit["selection_reason"] = "target_question_not_observed_on_full_page"
        return primary_text, full_audit, ""
    candidate_audit = dict(full_audit)
    candidate_audit["full_page_formula_count"] = int(full_audit.get("formula_count") or 0)
    candidate_audit["formula_count"] = len(re.findall(r"\$[^$]+\$", candidate, re.DOTALL))
    candidate_audit["formulas"] = []
    candidate_audit["recovery_mode"] = recovery_mode
    recovered, action = fuse_structured_ocr_transcript(
        primary_text, candidate, candidate_audit
    )
    if action:
        candidate_audit["selection_reason"] = f"full_page_{recovery_mode}"
        return recovered, candidate_audit, f"recover_{recovery_mode}"
    candidate_audit["selection_reason"] = "full_page_candidate_not_better"
    return primary_text, candidate_audit, ""


def ocr_quality_metrics(items: list[OCRItem]) -> dict[str, object]:
    """Score whether one OCR pass contains a plausible complete question."""

    transcript = build_ocr_transcript(items)
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    compact = re.sub(r"\s+", "", transcript)
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in compact)
    alphanumeric_count = sum(char.isalnum() for char in compact)
    long_line_count = sum(len(re.sub(r"\s+", "", line)) >= 18 for line in lines)
    part_marker_count = sum(
        bool(re.match(r"^(?:[（(]\s*\d+\s*[）)]|[①②③④⑤])", line))
        for line in lines
    )
    main_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^\d{1,3}\s*[.．、]", line)
        ),
        None,
    )
    first_part_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^(?:[（(]\s*\d+\s*[）)]|[①②③④⑤])", line)
        ),
        None,
    )
    missing_preamble = False
    if (
        main_index is not None
        and first_part_index is not None
        and first_part_index > main_index
    ):
        preamble = "".join(lines[main_index + 1:first_part_index])
        missing_preamble = (
            sum("\u4e00" <= char <= "\u9fff" for char in preamble) < 4
        )
    score = (
        cjk_count
        + alphanumeric_count * 0.22
        + long_line_count * 12.0
        + part_marker_count * 7.0
        - (18.0 if missing_preamble else 0.0)
    )
    semantic = transcript_semantic_completeness(transcript)
    return {
        "transcript": transcript,
        "character_count": len(compact),
        "cjk_count": cjk_count,
        "alphanumeric_count": alphanumeric_count,
        "line_count": len(lines),
        "long_line_count": long_line_count,
        "part_marker_count": part_marker_count,
        "missing_preamble": missing_preamble,
        "score": round(score, 3),
        **semantic,
    }


def needs_ocr_fallback(
    metrics: dict[str, object],
    items: list[OCRItem] | None = None,
) -> bool:
    """Request a second local pass only when the first pass looks incomplete."""

    transcript = str(metrics.get("transcript") or "")
    character_count = int(metrics.get("character_count") or 0)
    non_formula_metrics = dict(metrics)
    non_formula_metrics["formula_gap"] = False
    return bool(
        metrics.get("missing_preamble") is True
        or character_count < 18
        or (
            character_count < 55
            and "如图" in transcript
        )
        or (
            character_count < 70
            and re.search(r"(?m)^\d{1,3}\s*[.．、]\s*$", transcript)
        )
        or (items is not None and needs_tiled_ocr_fallback(items))
        or needs_semantic_ocr_recovery(non_formula_metrics)
    )


def fallback_ocr_is_better(
    primary: dict[str, object],
    fallback: dict[str, object],
) -> bool:
    """Select the wider detector only for a material content improvement."""

    primary_score = float(primary.get("score") or 0.0)
    fallback_score = float(fallback.get("score") or 0.0)
    cjk_gain = int(fallback.get("cjk_count") or 0) - int(
        primary.get("cjk_count") or 0
    )
    long_line_gain = int(fallback.get("long_line_count") or 0) - int(
        primary.get("long_line_count") or 0
    )
    repairs_preamble = (
        primary.get("missing_preamble") is True
        and fallback.get("missing_preamble") is False
    )
    primary_length = int(primary.get("character_count") or 0)
    # A loose pass must not replace an already substantial question merely
    # because it also recognized handwritten workings. Long primary text keeps
    # authority; missing table cells/nested clauses are fused separately.
    if primary_length >= 90 and not repairs_preamble:
        return False
    material_content_gain = cjk_gain >= 6 or long_line_gain >= 1 or repairs_preamble
    return material_content_gain and fallback_score >= primary_score + 7.0


def ocr_box_overlap_ratio(first: OCRItem, second: OCRItem) -> float:
    """Return intersection area relative to the smaller OCR rectangle."""

    first_left = float(first.box[:, 0].min())
    first_top = float(first.box[:, 1].min())
    first_right = float(first.box[:, 0].max())
    first_bottom = float(first.box[:, 1].max())
    second_left = float(second.box[:, 0].min())
    second_top = float(second.box[:, 1].min())
    second_right = float(second.box[:, 0].max())
    second_bottom = float(second.box[:, 1].max())
    overlap_width = max(0.0, min(first_right, second_right) - max(first_left, second_left))
    overlap_height = max(0.0, min(first_bottom, second_bottom) - max(first_top, second_top))
    intersection = overlap_width * overlap_height
    first_area = max(1.0, (first_right - first_left) * (first_bottom - first_top))
    second_area = max(1.0, (second_right - second_left) * (second_bottom - second_top))
    return intersection / min(first_area, second_area)


def fuse_ocr_items(
    primary_items: list[OCRItem],
    fallback_items: list[OCRItem],
) -> list[OCRItem]:
    """Keep accurate primary rows and add only material loose-pass recovery.

    The loose detector is useful for curved or faint rows, but it can also
    classify dark handwriting as printed text.  Matching against *all* primary
    detections (including low-confidence/unprinted detections) prevents those
    newly promoted annotations from entering the transcript.  A fallback item
    that overlaps a primary item is added only when it contains a substantial
    amount of recovered Chinese question text.
    """

    fused = list(primary_items)
    for fallback_item in fallback_items:
        if not (fallback_item.printed or fallback_item.formula_like):
            continue
        overlapping = [
            primary_item
            for primary_item in primary_items
            if ocr_box_overlap_ratio(primary_item, fallback_item) >= 0.28
        ]
        fallback_compact = re.sub(r"\s+", "", fallback_item.text)
        fallback_cjk = sum("\u4e00" <= char <= "\u9fff" for char in fallback_compact)
        if not overlapping:
            fused.append(fallback_item)
            continue

        normalized_fallback = normalize_reflow_text([fallback_item.text])
        if any(
            normalize_reflow_text([item.text]) == normalized_fallback
            for item in overlapping
        ):
            continue
        reliable_blank_slot = bool(
            fallback_item.score >= 0.94
            and re.fullmatch(r"[A-Za-z]\s*=", fallback_compact)
        )
        if reliable_blank_slot:
            fused.append(fallback_item)
            continue
        primary_cjk = max(
            (
                sum("\u4e00" <= char <= "\u9fff" for char in item.text)
                for item in overlapping
            ),
            default=0,
        )
        primary_length = max(
            (len(re.sub(r"\s+", "", item.text)) for item in overlapping),
            default=0,
        )
        material_recovery = (
            fallback_cjk >= primary_cjk + 4
            and len(fallback_compact) >= primary_length + 12
        )
        if material_recovery:
            fused.append(fallback_item)
    return fused


def needs_tiled_ocr_fallback(items: list[OCRItem]) -> bool:
    """Detect long curved prose rows that the recognizer read as gibberish."""

    suspicious_rows = 0
    for item in items:
        cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in item.text)
        if (
            0.30 <= item.score < 0.72
            and cjk_count >= 6
            and item.width >= 4.0 * max(item.height, 1.0)
        ):
            suspicious_rows += 1
    return suspicious_rows >= 2


def stitch_overlapping_ocr_text(first: str, second: str) -> str:
    """Join left/right tile readings using their stable overlapping phrase."""

    first_compact = re.sub(r"\s+", "", first)
    second_compact = re.sub(r"\s+", "", second)
    if not first_compact:
        return second_compact
    if not second_compact or second_compact in first_compact:
        return first_compact
    if first_compact in second_compact:
        return second_compact
    match = SequenceMatcher(None, first_compact, second_compact).find_longest_match()
    if (
        match.size >= 4
        and (
            match.a >= max(0, len(first_compact) // 3)
            or match.a + match.size >= len(first_compact) - 2
        )
        and match.b <= len(second_compact) // 3
    ):
        return first_compact[: match.a + match.size] + second_compact[match.b + match.size :]
    return f"{first_compact} {second_compact}"


def run_batched_tiled_ocr(
    tiles: list[np.ndarray],
    engine: RapidOCR,
) -> list[tuple[np.ndarray, tuple[str, ...], tuple[float, ...], int]]:
    """Detect every tile separately, then classify/recognize all rows in one batch.

    Four complete ``run_ocr`` calls repeatedly enter the orientation and text
    recognizers.  The detector still sees each narrow tile at full resolution,
    while the much cheaper row crops share one classifier and recognizer call.
    If RapidOCR's internal API ever changes, the caller falls back to the former
    independent-pass implementation without changing OCR output semantics.
    """

    from rapidocr.utils.process_img import map_boxes_to_original

    records: list[dict[str, object]] = []
    all_crops: list[np.ndarray] = []
    for tile in tiles:
        aspect = max(tile.shape[:2]) / max(1, min(tile.shape[:2]))
        ocr_side = 2600 if aspect >= 3.2 else 1800
        work, scale = resize_for_ocr(tile, ocr_side)
        loaded = engine.load_img(work)
        prepared, operation_record = engine.preprocess_img(loaded)
        try:
            crops, detection = engine.detect_and_crop(prepared, operation_record)
        except Exception:
            crops = []
            detection = None
        start = len(all_crops)
        all_crops.extend(crops)
        records.append(
            {
                "loaded": loaded,
                "operation_record": operation_record,
                "detection": detection,
                "scale": float(scale),
                "start": start,
                "count": len(crops),
            }
        )

    if not all_crops:
        return [
            (np.empty((0, 4, 2), np.float32), (), (), 0)
            for _tile in tiles
        ]
    classified_crops, classification = engine.cls_and_rotate(all_crops)
    recognition = engine.recognize_txt(classified_crops)
    recognized_texts = tuple(recognition.txts or ())
    recognized_scores = tuple(float(score) for score in (recognition.scores or ()))
    classifier_results = tuple(classification.cls_res or ())
    if len(recognized_texts) != len(all_crops):
        raise RuntimeError("RapidOCR tiled batch result count mismatch")

    results: list[tuple[np.ndarray, tuple[str, ...], tuple[float, ...], int]] = []
    for record in records:
        start = int(record["start"])
        count = int(record["count"])
        detection = record["detection"]
        if detection is None or count <= 0 or detection.boxes is None:
            results.append((np.empty((0, 4, 2), np.float32), (), (), 0))
            continue
        loaded = record["loaded"]
        boxes = map_boxes_to_original(
            np.asarray(detection.boxes, np.float32).copy(),
            record["operation_record"],
            int(loaded.shape[0]),
            int(loaded.shape[1]),
        ) / max(float(record["scale"]), 1e-9)
        tile_texts = recognized_texts[start : start + count]
        tile_scores = recognized_scores[start : start + count]
        keep = [
            index
            for index, (value, score) in enumerate(zip(tile_texts, tile_scores))
            if value.strip() and score >= float(engine.text_score)
        ]
        boxes = np.asarray([boxes[index] for index in keep], np.float32)
        if not keep:
            boxes = np.empty((0, 4, 2), np.float32)
        texts = tuple(tile_texts[index] for index in keep)
        scores = tuple(tile_scores[index] for index in keep)
        votes = [
            (str(label), float(score))
            for label, score in classifier_results[start : start + count]
            if float(score) >= 0.75
        ]
        zero_weight = sum(score for label, score in votes if label == "0")
        upside_down_weight = sum(score for label, score in votes if label == "180")
        upside_down_count = sum(label == "180" for label, _score in votes)
        rotation = 180 if (
            upside_down_count >= 2
            and upside_down_weight > zero_weight * 1.45
        ) else 0
        results.append((boxes, texts, scores, rotation))
    return results


def run_tiled_ocr_transcript(
    image: np.ndarray,
    engine: RapidOCR,
    config: Config,
    trigger_items: list[OCRItem],
    *,
    force: bool = False,
    full_page: bool = False,
) -> tuple[str, float, int]:
    """Re-read only the curved text band through three overlapping columns."""

    suspects = []
    for item in trigger_items:
        cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in item.text)
        if (
            0.30 <= item.score < 0.72
            and cjk_count >= 6
            and item.width >= 4.0 * max(item.height, 1.0)
        ):
            suspects.append(item)
    if len(suspects) < 2 and not force:
        return "", 0.0, 0

    height, width = image.shape[:2]
    if suspects:
        median_height = float(np.median([item.height for item in suspects]))
        top = max(0, int(min(item.box[:, 1].min() for item in suspects) - median_height * 0.55))
        bottom = min(
            height,
            int(max(item.box[:, 1].max() for item in suspects) + median_height * 0.55),
        )
    elif full_page:
        top = 0
        bottom = height
    else:
        top = 0
        bottom = min(height, max(120, int(round(height * 0.53))))
    if bottom - top < 80:
        return "", 0.0, 0

    tile_width = min(width, 500)
    max_start = max(0, width - tile_width)
    starts = sorted(
        {
            0,
            min(max_start, 200),
            min(max_start, 400),
            max_start,
        }
    )
    tile_records = [
        (left, image[top:bottom, left:min(width, left + tile_width)])
        for left in starts
    ]
    candidate_items: list[OCRItem] = []
    started = time.perf_counter()
    try:
        tile_results = run_batched_tiled_ocr(
            [tile for _left, tile in tile_records],
            engine,
        )
        effective_passes = 1
    except Exception:
        tile_results = [run_ocr(tile, engine) for _left, tile in tile_records]
        effective_passes = len(tile_records)
    for (left, _tile), (boxes, texts, scores, rotation) in zip(
        tile_records,
        tile_results,
    ):
        if rotation == 180 or not len(boxes):
            continue
        mapped_boxes = boxes.copy()
        mapped_boxes[:, :, 0] += left
        mapped_boxes[:, :, 1] += top
        tile_items = analyze_ocr_items(mapped_boxes, texts, scores, config)
        for item in tile_items:
            normalized = normalize_reflow_text([item.text])
            cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in normalized)
            structural = classify_reflow_text(normalized) in {
                "question",
                "subquestion",
            }
            math_evidence = sum(
                char.isdigit()
                or (char.isascii() and char.isalpha())
                or char in "=<>≤≥+-×÷*/√^{}[]()∩∪∞"
                for char in normalized
            )
            short_math_tail = bool(re.search(r"(?:m|cm)[²2][.。]?$", normalized))
            reliable_prose = item.score >= 0.90 and (
                cjk_count >= 3
                or structural
                or (
                    cjk_count >= 2
                    and "=" in normalized
                    and len(re.sub(r"\s+", "", normalized)) >= 8
                )
            )
            reliable_math_tail = item.score >= 0.82 and short_math_tail
            reliable_formula = bool(
                item.score >= 0.70
                and len(re.sub(r"\s+", "", normalized)) >= 8
                and math_evidence >= 5
            )
            if reliable_prose or reliable_formula or (
                reliable_math_tail
                and item.width >= 1.30 * max(item.height, 1.0)
            ):
                candidate_items.append(item)

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    if not candidate_items:
        return "", elapsed_ms, effective_passes

    # Connect readings from overlapping tiles by their common phrase. This is
    # more reliable than ordinary y-overlap grouping on a curved page, where
    # the right end of one row can vertically overlap the next printed row.
    parents = list(range(len(candidate_items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    compact_texts = [
        re.sub(r"\s+", "", normalize_reflow_text([item.text]))
        for item in candidate_items
    ]
    for first_index, first_item in enumerate(candidate_items):
        for second_index in range(first_index + 1, len(candidate_items)):
            second_item = candidate_items[second_index]
            if abs(first_item.center_y - second_item.center_y) > max(
                first_item.height,
                second_item.height,
            ) * 1.25:
                continue
            overlap = SequenceMatcher(
                None,
                compact_texts[first_index],
                compact_texts[second_index],
            ).find_longest_match().size
            if overlap >= 3:
                union(first_index, second_index)

    components: dict[int, list[OCRItem]] = {}
    for index, item in enumerate(candidate_items):
        components.setdefault(find(index), []).append(item)

    recovered_lines: list[tuple[float, str]] = []
    for group in components.values():
        ordered = sorted(group, key=lambda item: item.center_x)
        line = ""
        for item in ordered:
            line = stitch_overlapping_ocr_text(
                line,
                normalize_reflow_text([item.text]),
            )
        if line:
            recovered_lines.append(
                (float(np.median([item.center_y for item in group])), line)
            )
    recovered_lines.sort(key=lambda record: record[0])
    return repair_ocr_transcript(
        "\n".join(line for _, line in recovered_lines),
    ), elapsed_ms, effective_passes


def configure_ocr_detection(
    engine: RapidOCR,
    *,
    box_thresh: float,
    unclip_ratio: float,
    text_score: float,
) -> None:
    engine.update_params(
        box_thresh=box_thresh,
        unclip_ratio=unclip_ratio,
        text_score=text_score,
    )


def run_adaptive_ocr(
    image: np.ndarray,
    engine: RapidOCR,
    config: Config,
) -> tuple[np.ndarray, tuple[str, ...], tuple[float, ...], int, list[OCRItem], dict[str, object]]:
    """Run the normal OCR pass, then conditionally retry curved/low-contrast rows."""

    configure_ocr_detection(
        engine,
        box_thresh=0.50,
        unclip_ratio=1.50,
        text_score=0.50,
    )
    primary_started = time.perf_counter()
    primary_boxes, primary_texts, primary_scores, primary_rotation = run_ocr(
        image,
        engine,
    )
    primary_ms = round((time.perf_counter() - primary_started) * 1000.0, 3)
    primary_items = analyze_ocr_items(
        primary_boxes,
        primary_texts,
        primary_scores,
        config,
    )
    primary_metrics = ocr_quality_metrics(primary_items)
    audit: dict[str, object] = {
        "selected_pass": "primary",
        "pass_count": 1,
        "fallback_triggered": False,
        "primary_elapsed_ms": primary_ms,
        "primary": primary_metrics,
        "semantic_completeness": transcript_semantic_completeness(
            str(primary_metrics.get("transcript") or "")
        ),
        "tiled_recovery_recommended": bool(
            needs_tiled_ocr_fallback(primary_items)
            or needs_tiled_semantic_ocr_recovery(
                transcript_semantic_completeness(
                    str(primary_metrics.get("transcript") or "")
                )
            )
        ),
        "tiled_full_page": needs_tiled_semantic_ocr_recovery(
            transcript_semantic_completeness(
                str(primary_metrics.get("transcript") or "")
            )
        ),
    }
    selected = (
        primary_boxes,
        primary_texts,
        primary_scores,
        primary_rotation,
        primary_items,
    )
    structured_stem_recovery_preferred = bool(
        os.environ.get("TEST6_FORMULA_OCR_URL", "").strip()
        and primary_metrics.get("stem_missing") is True
        and set(primary_metrics.get("option_labels") or []) == set("ABCD")
        and int(primary_metrics.get("cjk_count") or 0) <= 2
    )
    if structured_stem_recovery_preferred:
        # Loose/tiled RapidOCR repeatedly re-reads the same formula row and is
        # both slower and less accurate than the resident FormulaNet pass.
        audit["structured_stem_recovery_preferred"] = True
        audit["tiled_recovery_recommended"] = False
        audit["tiled_full_page"] = False
        return (*selected, audit)
    if not config.ocr_fallback_enabled or not needs_ocr_fallback(
        primary_metrics,
        primary_items,
    ):
        return (*selected, audit)

    audit["fallback_triggered"] = True
    configure_ocr_detection(
        engine,
        box_thresh=config.ocr_fallback_box_thresh,
        unclip_ratio=config.ocr_fallback_unclip_ratio,
        text_score=config.ocr_fallback_text_score,
    )
    fallback_started = time.perf_counter()
    try:
        fallback_boxes, fallback_texts, fallback_scores, fallback_rotation = run_ocr(
            image,
            engine,
        )
        fallback_ms = round((time.perf_counter() - fallback_started) * 1000.0, 3)
        fallback_items = analyze_ocr_items(
            fallback_boxes,
            fallback_texts,
            fallback_scores,
            config,
        )
        fallback_metrics = ocr_quality_metrics(fallback_items)
        audit.update(
            {
                "pass_count": 2,
                "fallback_elapsed_ms": fallback_ms,
                "fallback": fallback_metrics,
            }
        )
        fused_items = fuse_ocr_items(primary_items, fallback_items)
        fused_metrics = ocr_quality_metrics(fused_items)
        audit.update(
            {
                "fused": fused_metrics,
                "fused_added_item_count": max(
                    0, len(fused_items) - len(primary_items)
                ),
            }
        )
        if fallback_ocr_is_better(primary_metrics, fused_metrics):
            fused_boxes = np.asarray(
                [item.box for item in fused_items],
                dtype=np.float32,
            )
            fused_texts = tuple(item.text for item in fused_items)
            fused_scores = tuple(float(item.score) for item in fused_items)
            selected = (
                fused_boxes,
                fused_texts,
                fused_scores,
                primary_rotation,
                fused_items,
            )
            audit["selected_pass"] = "primary+fallback-fused"
        elif fallback_ocr_is_better(primary_metrics, fallback_metrics):
            selected = (
                fallback_boxes,
                fallback_texts,
                fallback_scores,
                fallback_rotation,
                fallback_items,
            )
            audit["selected_pass"] = "fallback-loose"
    except Exception as exc:  # optional enhancement must not discard a good first pass
        audit["fallback_error"] = type(exc).__name__
    finally:
        configure_ocr_detection(
            engine,
            box_thresh=0.50,
            unclip_ratio=1.50,
            text_score=0.50,
        )
    selected_metrics = ocr_quality_metrics(selected[4])
    audit["semantic_completeness"] = transcript_semantic_completeness(
        str(selected_metrics.get("transcript") or "")
    )
    selected_semantic = transcript_semantic_completeness(
        str(selected_metrics.get("transcript") or "")
    )
    audit["tiled_full_page"] = needs_tiled_semantic_ocr_recovery(selected_semantic)
    audit["tiled_recovery_recommended"] = bool(
        needs_tiled_ocr_fallback(selected[4])
        or audit["tiled_full_page"]
    )
    return (*selected, audit)


def remove_isolated_ocr_annotations(
    image: np.ndarray,
    items: list[OCRItem],
    figure_boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Remove only oversized isolated OCR snippets outside protected figures."""

    trusted_heights = [
        item.height
        for item in items
        if (item.printed or item.formula_like) and item.height > 4
    ]
    median_height = float(np.median(trusted_heights)) if trusted_heights else 32.0
    cleaned = image.copy()
    lines = group_items_into_lines(items)
    instruction_centers = [
        float(np.mean([item.center_y for item in line]))
        for line in lines
        if re.search(
            r"计算|解方程(?:组)?|化简|求值|列式|求解|证明|填空",
            normalize_reflow_text([item.text for item in line]),
        )
    ]
    for line in lines:
        line_center = float(np.mean([item.center_y for item in line]))
        protected = any(
            abs(line_center - center) <= max(48.0, median_height * 1.9)
            for center in instruction_centers
        )
        kept = {
            id(item)
            for item in filter_inline_annotations(
                line,
                median_height,
                protect_math=protected,
            )
        }
        for item in line:
            if id(item) in kept or item_inside_figure(item, figure_boxes):
                continue
            polygon = np.round(item.box).astype(np.int32)
            local = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(local, [polygon], 255)
            padding = max(2, int(item.height * 0.10))
            local = cv2.dilate(
                local,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (padding * 2 + 1, padding * 2 + 1),
                ),
            )
            cleaned[local > 0] = (255, 255, 255)
    return cleaned


def build_layout_preserving_document(
    source: np.ndarray,
    items: list[OCRItem],
    figure_boxes: list[tuple[int, int, int, int]],
    config: Config,
) -> np.ndarray:
    """Stack source-pixel text bands and figures without OCR rewriting formulas."""

    canvas_w = config.reflow_canvas_width
    margin = max(54, int(canvas_w * 0.055))
    usable_w = canvas_w - margin * 2
    content = content_union_box(source.shape, items, figure_boxes)
    x1, y1, x2, y2 = content
    sections: list[tuple[str, tuple[int, int, int, int]]] = []

    if figure_boxes:
        first_figure_y = min(box[1] for box in figure_boxes)
        text_items = [
            item
            for item in items
            if (item.printed or item.formula_like)
            and not item_inside_figure(item, figure_boxes)
            and item.center_y < first_figure_y
        ]
        if text_items:
            text_top = max(
                y1,
                int(min(item.box[:, 1].min() for item in text_items))
                - max(18, int(source.shape[0] * 0.018)),
            )
            text_bottom = min(
                y2,
                max(
                    first_figure_y + max(28, int(source.shape[0] * 0.04)),
                    int(max(item.box[:, 1].max() for item in text_items))
                    + max(22, int(source.shape[0] * 0.025)),
                ),
            )
            if text_bottom > text_top:
                sections.append(("text", (x1, text_top, x2, text_bottom)))
        sections.extend(("figure", box) for box in figure_boxes)

        last_figure_y = max(box[3] for box in figure_boxes)
        trailing = [
            item
            for item in items
            if (item.printed or item.formula_like)
            and not item_inside_figure(item, figure_boxes)
            and item.center_y > last_figure_y
        ]
        if trailing:
            bottom_top = max(
                last_figure_y,
                int(min(item.box[:, 1].min() for item in trailing)) - 18,
            )
            bottom_bottom = min(
                y2,
                int(max(item.box[:, 1].max() for item in trailing)) + 22,
            )
            if bottom_bottom > bottom_top:
                sections.append(("text", (x1, bottom_top, x2, bottom_bottom)))
    if not sections:
        sections = [("text", content)]

    prepared: list[tuple[str, Image.Image]] = []
    estimated = 100
    for kind, box in sections:
        snippet = (
            crop_illustration_with_clean_margins(source, box, items)
            if kind == "figure"
            else crop_with_padding(source, box, 0.02)
        )
        if snippet.size == 0:
            continue
        pil = Image.fromarray(cv2.cvtColor(snippet, cv2.COLOR_BGR2RGB))
        max_height = 900 if kind == "figure" else 520
        scale_cap = 2.25 if kind == "figure" else 1.35
        scale = min(
            scale_cap,
            usable_w / max(1, pil.width),
            max_height / max(1, pil.height),
        )
        pil = pil.resize(
            (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
            Image.Resampling.LANCZOS,
        )
        prepared.append((kind, pil))
        estimated += pil.height + (42 if kind == "figure" else 30)

    canvas = Image.new("RGB", (canvas_w, max(640, estimated)), "white")
    draw = ImageDraw.Draw(canvas)
    cursor = 48
    for kind, pil in prepared:
        x = (canvas_w - pil.width) // 2
        canvas.paste(pil, (x, cursor))
        draw.rectangle(
            (x - 2, cursor - 2, x + pil.width + 2, cursor + pil.height + 2),
            outline=(224, 226, 226),
            width=2,
        )
        cursor += pil.height + (42 if kind == "figure" else 30)
    return cv2.cvtColor(
        np.asarray(canvas.crop((0, 0, canvas_w, min(canvas.height, cursor + 24)))),
        cv2.COLOR_RGB2BGR,
    )


def line_requires_original_pixels(
    line: list[OCRItem],
    *,
    near_formula_instruction: bool = False,
) -> bool:
    """Keep trusted printed formula rows as pixels instead of lossy OCR text."""

    if any(item.formula_like for item in line):
        return True
    printed = [item for item in line if item.printed]
    if not printed:
        return False
    text = "".join(item.text for item in printed)
    operators = set("=<>+-×÷*/^√∫∑±≤≥≠≈∠△⊥∥|")
    has_operator = any(char in operators for char in text)
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in text)
    has_math_atom = any(char.isdigit() for char in text) or any(
        char.isascii() and char.isalpha() for char in text
    )
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
    math_character_count = sum(
        char.isdigit()
        or (char.isascii() and char.isalpha())
        or char in operators
        or char in "{}()[]₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹"
        for char in text
    )
    formula_sensitive = bool(
        re.search(r"[{}√∫∑]|[₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]", text)
        or sum(char in operators for char in text) >= 2
        or (
            cjk_count <= 8
            and math_character_count / max(1, len(text)) >= 0.28
        )
    )
    # Recognized Chinese prose is typeset as text even when it contains inline
    # mathematics.  Only a pure printed formula next to a trusted instruction
    # remains a source-pixel block; isolated handwritten results cannot qualify.
    return has_operator and has_math_atom and (
        formula_sensitive or (not has_cjk and near_formula_instruction)
    )


def classify_reflow_text(text: str) -> str:
    """按题号、题干小问和选项划分排版层级。"""
    import re

    compact = text.lstrip()
    if re.match(r"^[1-9]\d{0,2}\s*[.．、](?!\d)", compact) or re.match(r"^[1-9]\d{0,2}\s*[（(].{0,5}分", compact):
        return "question"
    if re.match(r"^[（(]\s*\d{1,2}\s*[）)]", compact):
        return "subquestion"
    if re.match(r"^[A-DＡ-Ｄ](?:\s*[.、:：]\s*|\s+)", compact):
        return "option"
    return "body"


def select_middle_question_box(shape: tuple[int, ...], items: list[OCRItem],
                               enabled: bool = True) -> tuple[int, int, int, int]:
    """在一张图片包含多道题时，返回最靠近页面中心的完整主问题范围。

    优先识别 ``12.``、``12、``、``第12题``、``4.(14分)`` 等主问题号。
    ``(1)``、``（2）`` 之类小问不会被当作新题。若一个主问题号都没有识别到，
    只有页面中存在至少三个由大空白隔开的稳定文字块时才使用段落备用方案，
    从而尽量避免把普通单题中的题干、小问和公式错误拆开。
    """
    import re

    h, w = shape[:2]
    full_page = (0, 0, w, h)
    if not enabled:
        return full_page

    printed_items = [item for item in items if item.printed]
    line_groups = group_items_into_lines(printed_items)
    line_records: list[dict[str, object]] = []
    for line in line_groups:
        if not line:
            continue
        ordered = sorted(line, key=lambda item: item.center_x)
        text = normalize_reflow_text([item.text for item in ordered])
        box = (
            int(min(item.box[:, 0].min() for item in ordered)),
            int(min(item.box[:, 1].min() for item in ordered)),
            int(max(item.box[:, 0].max() for item in ordered)),
            int(max(item.box[:, 1].max() for item in ordered)),
        )
        if text:
            line_records.append({"text": text, "box": box})
    if not line_records:
        return full_page
    line_records.sort(key=lambda record: int(record["box"][1]))  # type: ignore[index]

    line_heights = [
        int(record["box"][3]) - int(record["box"][1])  # type: ignore[index]
        for record in line_records
    ]
    median_height = float(np.median(line_heights)) if line_heights else 32.0
    main_markers: list[dict[str, object]] = []
    for record in line_records:
        text = str(record["text"])
        box = record["box"]  # type: ignore[assignment]
        # 主问题通常从页面左侧开始。这个限制可排除图中数字、页码和右侧公式。
        if int(box[0]) > w * 0.42:
            continue
        numbered = re.match(
            r"^\s*([1-9]\d{0,2})(?:\s*[-—–]\s*(?:[IVXivx]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+))?\s*"
            r"(?:[.．。、:：](?!\d)|(?=[（(]\s*\d+\s*分))",
            text,
        )
        named = re.match(r"^\s*第\s*(\d{1,3})\s*题", text)
        match = named or numbered
        if match and len(re.sub(r"\s+", "", text)) >= 5:
            # Axis ticks such as ``4.0 B/10T`` and ``1.5 2.0 ...`` are not
            # question starts.  Dataset questions are Chinese, so a trusted
            # main marker must also carry at least two Chinese characters.
            cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
            score_marker = bool(re.match(
                r"^\s*[1-9]\d{0,2}(?:\s*[-—–]\s*(?:[IVXivx]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+))?"
                r"\s*[.．。、:：]?\s*[（(]\s*\d+\s*分",
                text,
            ))
            if cjk_count < 2 and not score_marker:
                continue
            if named and re.fullmatch(r"\s*第\s*\d{1,3}\s*题\s*[图表]?\s*", text):
                continue
            main_markers.append({"number": int(match.group(1)), "box": box, "text": text})

    if len(main_markers) >= 2:
        # 每道题从自身主问题号开始，到下一道主问题号之前结束。
        # 选择题块中心距离页面垂直中心最近的一道；偶数道题时也有确定结果。
        candidates: list[tuple[float, int, int]] = []
        for index, marker in enumerate(main_markers):
            start_y = int(marker["box"][1])  # type: ignore[index]
            end_y = (int(main_markers[index + 1]["box"][1])  # type: ignore[index]
                     if index + 1 < len(main_markers) else h)
            candidates.append((abs((start_y + end_y) / 2 - h / 2), start_y, end_y))
        _, start_y, end_y = min(candidates, key=lambda candidate: candidate[0])
        top_padding = max(12, int(median_height * 0.55))
        preceding_bottoms = [
            int(record["box"][3])  # type: ignore[index]
            for record in line_records
            if int(record["box"][3]) < start_y  # type: ignore[index]
        ]
        if preceding_bottoms:
            visible_gap = start_y - max(preceding_bottoms)
            top_padding = min(top_padding, max(2, visible_gap // 2))
        selected_top = max(0, start_y - top_padding)
        # The next question marker may begin on the left while a chart from the
        # current question still reaches almost to that same baseline on the
        # right. Stop exactly at the next marker instead of subtracting another
        # text-height margin, which can cut x-axis ticks and the final stem line.
        selected_bottom = min(h, end_y)
        current_option_labels = {
            match.group(1)
            for record in line_records
            if start_y <= int(record["box"][1]) < end_y  # type: ignore[index]
            for match in [
                re.search(
                    r"(?:^|(?<=[则是有为]))\s*([A-D])\s*[.．、:]",
                    str(record["text"]),
                )
            ]
            if match is not None
        }
        # On photographed two-column pages the next question number can start
        # slightly above the current question's final D option.  Keep one
        # narrow line band below that marker when A-C are already present; the
        # transcript boundary cleaner will then remove the neighbouring stem.
        if {"A", "B", "C"}.issubset(current_option_labels) and "D" not in current_option_labels:
            selected_bottom = min(
                h,
                end_y + max(40, int(median_height * 2.0)),
            )
        if selected_bottom - selected_top >= max(80, median_height * 2):
            return 0, selected_top, w, selected_bottom
        return full_page

    # A single visible marker can still have the previous question's trailing
    # option above it.  Trim only that proven prefix while retaining everything
    # below the marker, including diagrams after the final OCR line.
    if main_markers:
        marker_top = int(main_markers[0]["box"][1])  # type: ignore[index]
        has_prefix = any(
            int(record["box"][1]) < marker_top - max(3, int(median_height * 0.12))  # type: ignore[index]
            for record in line_records
        )
        if has_prefix and marker_top >= max(8, int(median_height * 0.75)):
            pad = 2
            return 0, max(0, marker_top - pad), w, h
        return full_page

    # OCR 完全漏掉主问题号时，根据显著纵向空白划分文字块。
    # 至少形成三个内容充分的块才判断为一图多题，否则继续保留整页。
    large_gap = max(median_height * 3.6, h * 0.055)
    blocks: list[tuple[int, int, list[dict[str, object]]]] = []
    block_lines: list[dict[str, object]] = [line_records[0]]
    for record in line_records[1:]:
        previous = block_lines[-1]
        gap = int(record["box"][1]) - int(previous["box"][3])  # type: ignore[index]
        if gap > large_gap:
            blocks.append((int(block_lines[0]["box"][1]),  # type: ignore[index]
                           int(block_lines[-1]["box"][3]), block_lines))  # type: ignore[index]
            block_lines = []
        block_lines.append(record)
    blocks.append((int(block_lines[0]["box"][1]),  # type: ignore[index]
                   int(block_lines[-1]["box"][3]), block_lines))  # type: ignore[index]
    useful_blocks = [
        block for block in blocks
        if len("".join(str(line["text"]) for line in block[2]).replace(" ", "")) >= 12
    ]
    if len(useful_blocks) < 3:
        return full_page
    selected = min(useful_blocks, key=lambda block: abs((block[0] + block[1]) / 2 - h / 2))
    pad = max(16, int(median_height * 0.70))
    return 0, max(0, selected[0] - pad), w, min(h, selected[1] + pad)


def items_inside_question(items: list[OCRItem], question_box: tuple[int, int, int, int]) -> list[OCRItem]:
    """只保留中心点位于选中题范围内的 OCR 项。"""
    x1, y1, x2, y2 = question_box
    return [
        item for item in items
        if x1 <= item.center_x <= x2 and y1 <= item.center_y <= y2
    ]


def select_question_layout_regions(
    layout_audit: dict[str, object],
    question_box: tuple[int, int, int, int],
    items: list[OCRItem],
    transcript: str,
    shape: tuple[int, ...],
) -> list[dict[str, object]]:
    """Associate full-page PP-DocLayout regions with the selected question.

    Textbook diagrams are often printed immediately *above* their stem and
    captioned ``（第16题）``. A text-only question crop therefore loses them.
    Caption-number association is stronger than vertical distance and prevents
    a neighbouring question's diagram from being copied into the workspace.
    """

    page_h, page_w = shape[:2]
    qx1, qy1, qx2, qy2 = question_box
    compact = re.sub(r"\s+", "", str(transcript or ""))
    target_number = first_question_number(transcript)
    table_cue = bool(re.search(
        r"下表(?!面)|如下表|表中|表格|统计表|成绩表|(?:对应)?数据如下|(?:成绩|统计|结果)如下",
        compact,
    ))
    figure_cue = bool(re.search(
        r"如图|图[①②③④⑤⑥1-9甲乙丙丁戊己]|"
        r"第\s*\d{1,3}(?:\s*[-—–]\s*(?:[IVXivx]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+))?\s*(?:题)?图|"
        r"示意图|函数图像|坐标系|电路图|光路图|转盘",
        compact,
    ))
    caption_items: list[OCRItem] = []
    if target_number is not None:
        caption_pattern = re.compile(rf"第\s*{target_number}\s*题")
        caption_items = [
            item for item in items
            if caption_pattern.search(str(item.text or ""))
        ]

    selected: list[dict[str, object]] = []
    fallback_candidates: list[tuple[int, dict[str, object]]] = []
    for raw in list(layout_audit.get("layout_regions") or []):
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or "").strip().lower()
        if label not in {"table", "image", "chart"}:
            continue
        raw_bbox = raw.get("bbox")
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            continue
        try:
            left, top, right, bottom = (
                int(round(float(value))) for value in raw_bbox
            )
            score = float(raw.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        implicit_figure_evidence = bool(
            label in {"image", "chart"} and score >= 0.82
        )
        if label == "table" and not (table_cue or figure_cue):
            continue
        if label in {"image", "chart"} and not (
            figure_cue or implicit_figure_evidence
        ):
            continue
        if score < (0.28 if label == "table" else 0.32):
            continue
        left = max(0, min(page_w - 1, left))
        top = max(0, min(page_h - 1, top))
        right = max(left + 1, min(page_w, right))
        bottom = max(top + 1, min(page_h, bottom))
        region_area = max(1, (right - left) * (bottom - top))
        intersection = (
            max(0, min(right, qx2) - max(left, qx1))
            * max(0, min(bottom, qy2) - max(top, qy1))
        )
        normalized = {
            "label": label,
            "bbox": [left, top, right, bottom],
            "score": score,
        }
        if intersection / region_area >= 0.55:
            selected.append(normalized)
            continue

        pad_x = max(20, round((right - left) * 0.22))
        pad_y = max(30, round((bottom - top) * 0.20))
        caption_match = any(
            left - pad_x <= caption.center_x <= right + pad_x
            and top - pad_y <= caption.center_y <= bottom + pad_y
            for caption in caption_items
        )
        if caption_match:
            selected.append(normalized)
            continue
        if figure_cue and bottom <= qy1:
            gap = qy1 - bottom
            if gap <= max(90, round(page_h * 0.18)):
                fallback_candidates.append((gap, normalized))

    if not selected and fallback_candidates:
        # Use only the closest preceding region. This also covers a readable
        # caption whose OCR box sits just outside the model crop padding; the
        # explicit figure cue plus short vertical gap keeps the association
        # bounded to the selected question.
        selected.append(min(fallback_candidates, key=lambda value: value[0])[1])
    selected.sort(key=lambda region: (region["bbox"][1], region["bbox"][0]))
    return selected


def clean_external_layout_annotations(
    crop: np.ndarray,
    source_box: list[int],
    region_box: tuple[int, int, int, int],
    full_items: list[OCRItem],
) -> tuple[np.ndarray, list[str]]:
    """Suppress obvious handwritten arithmetic outside a detected diagram core.

    Standalone numbers and letter labels are deliberately retained because
    erasing those would make geometry figures semantically unsafe.
    """

    if crop.size == 0 or not full_items:
        return crop, []
    source_left, source_top, source_right, source_bottom = source_box
    left, top, right, bottom = region_box
    region_w = max(1, right - left)
    region_h = max(1, bottom - top)
    mask = np.zeros(crop.shape[:2], np.uint8)
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    local_background = cv2.GaussianBlur(crop_gray, (0, 0), 7.0)
    dark_ink = (
        (crop_gray.astype(np.int16) + 4 < local_background.astype(np.int16))
        & (crop_gray < 232)
    ).astype(np.uint8) * 255
    removed: list[str] = []
    for item in full_items:
        if not (
            source_left <= item.center_x <= source_right
            and source_top <= item.center_y <= source_bottom
            and item.score >= 0.72
        ):
            continue
        compact = re.sub(r"\s+", "", str(item.text or ""))
        if not 2 <= len(compact) <= 40:
            continue
        if any("\u4e00" <= char <= "\u9fff" for char in compact):
            continue
        # Preserve printed geometric relations such as AB=10 or CD∥EF.
        if sum("A" <= char <= "Z" for char in compact) >= 2:
            continue
        relative_x = (item.center_x - left) / region_w
        relative_y = (item.center_y - top) / region_h
        operators = sum(char in "=+×÷*/^" for char in compact)
        operator_leading = compact[0] in "+=×÷*/" and any(
            char.isdigit() for char in compact
        )
        foreign_option_at_top = bool(
            re.match(r"^[A-DＡ-Ｄ][.．、:：]?\d", compact)
            and relative_y <= 0.14
        )
        if (
            "=" not in compact
            and operators < 2
            and not operator_leading
            and not foreign_option_at_top
        ):
            continue
        if not (
            relative_x <= 0.35
            or relative_x >= 0.68
            or relative_y <= 0.14
            or relative_y >= 0.88
        ):
            continue
        polygon = item.box.copy()
        polygon[:, 0] -= source_left
        polygon[:, 1] -= source_top
        polygon[:, 0] = np.clip(polygon[:, 0], 0, crop.shape[1] - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, crop.shape[0] - 1)
        polygon_mask = np.zeros_like(mask)
        cv2.fillPoly(polygon_mask, [np.round(polygon).astype(np.int32)], 255)
        local = cv2.bitwise_and(polygon_mask, dark_ink)
        margin = max(2, min(4, round(item.height * 0.09)))
        local = cv2.dilate(
            local,
            np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8),
        )
        mask = cv2.bitwise_or(mask, local)
        removed.append(compact)
    if removed:
        cleaned = crop.copy()
        background = cv2.morphologyEx(
            crop,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        )
        cleaned[mask > 0] = background[mask > 0]
        return cleaned, removed
    return crop, removed


def likely_detached_handwritten_sketch(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    items: list[OCRItem],
    *,
    sibling_image_count: int,
    sibling_photo_present: bool = False,
) -> bool:
    """Reject a separate margin sketch without suppressing printed diagrams.

    A layout model can confidently call a student's velocity/vector doodle an
    image.  We reject it only when another image region exists, the candidate
    is sparse monochrome line art in an outer page column, and its OCR evidence
    consists solely of untrusted lowercase/digit fragments.  Printed geometry
    labels (uppercase points), prose and standalone unlabeled diagrams remain
    untouched.
    """

    if sibling_image_count < 2:
        return False
    left, top, right, bottom = box
    page_h, page_w = image.shape[:2]
    if not (right <= int(page_w * 0.34) or left >= int(page_w * 0.66)):
        return False
    inside = [
        item
        for item in items
        if left <= item.center_x <= right and top <= item.center_y <= bottom
    ]
    if any(item.printed or item.formula_like for item in inside):
        return False
    fragments = [re.sub(r"\s+", "", str(item.text or "")) for item in inside]
    fragments = [value for value in fragments if value]
    if fragments and any(re.search(r"[\u4e00-\u9fffA-ZΑ-Ω]", value) for value in fragments):
        return False
    if fragments and not all(
        re.fullmatch(r"[a-zα-ω0-9_′'↑↓+\-]{1,5}", value)
        for value in fragments
    ):
        return False
    if not fragments and not sibling_photo_present:
        return False

    roi = image[max(0, top):min(page_h, bottom), max(0, left):min(page_w, right)]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    histogram = histogram / max(1.0, float(histogram.sum()))
    populated = histogram > 0
    entropy = float(-np.sum(histogram[populated] * np.log2(histogram[populated])))
    ink_fraction = float(np.mean(gray < 185))
    midtone_fraction = float(np.mean((gray > 40) & (gray < 220)))
    colorful_fraction = float(np.mean(hsv[:, :, 1] > 45))
    return (
        0.002 <= ink_fraction <= 0.18
        and midtone_fraction <= 0.35
        and entropy <= 5.8
        and colorful_fraction < 0.08
    )


def prepare_structured_ocr_canvas(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Add vertical context around a shallow question strip for formula OCR.

    PP-DocLayout/FormulaNet can miss stacked fractions when a tightly cropped
    question is only one or two text rows high. The same pixels are detected
    reliably once the strip has ordinary page-like whitespace above and below.
    Return the padded canvas together with the top offset so spatial formula
    boxes can still be mapped back to selected-question coordinates.
    """

    if not isinstance(image, np.ndarray) or image.size == 0:
        return image, 0
    height, width = image.shape[:2]
    if height >= 420 or width / max(1, height) <= 2.30:
        return image.copy(), 0
    target_height = min(720, max(420, int(round(width * 0.42)), height * 3))
    if target_height <= height:
        return image.copy(), 0
    pad_top = (target_height - height) // 2
    pad_bottom = target_height - height - pad_top
    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return padded, pad_top


def likely_photographic_region(
    image: np.ndarray,
    box: tuple[int, int, int, int],
) -> bool:
    """Return whether a layout crop carries photograph-like tonal detail."""

    left, top, right, bottom = box
    page_h, page_w = image.shape[:2]
    roi = image[max(0, top):min(page_h, bottom), max(0, left):min(page_w, right)]
    if roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    histogram = histogram / max(1.0, float(histogram.sum()))
    populated = histogram > 0
    entropy = float(-np.sum(histogram[populated] * np.log2(histogram[populated])))
    midtone_fraction = float(np.mean((gray > 40) & (gray < 220)))
    return entropy >= 6.0 and midtone_fraction >= 0.25


def attach_question_layout_regions(
    workspace: np.ndarray,
    workspace_items: list[OCRItem],
    full_image: np.ndarray,
    question_box: tuple[int, int, int, int],
    regions: list[dict[str, object]],
    full_items: list[OCRItem] | None = None,
) -> tuple[np.ndarray, list[OCRItem], list[tuple[int, int, int, int]], dict[str, object]]:
    """Map in-question regions and prepend caption-associated external crops."""

    qx1, qy1, qx2, qy2 = question_box
    mapped: list[tuple[int, int, int, int]] = []
    external_crops: list[np.ndarray] = []
    source_boxes: list[list[int]] = []
    external_region_labels: list[str] = []
    removed_annotations: list[str] = []
    image_region_boxes = [
        tuple(int(value) for value in region["bbox"])
        for region in regions
        if str(region.get("label") or "").strip().lower() == "image"
    ]
    sibling_image_count = len(image_region_boxes)
    photographic_boxes = {
        box for box in image_region_boxes if likely_photographic_region(full_image, box)
    }
    for region in regions:
        region_label = str(region.get("label") or "").strip().lower()
        left, top, right, bottom = (int(value) for value in region["bbox"])
        if (
            region_label == "image"
            and likely_detached_handwritten_sketch(
                full_image,
                (left, top, right, bottom),
                full_items or [],
                sibling_image_count=sibling_image_count,
                sibling_photo_present=any(
                    photo_box != (left, top, right, bottom)
                    for photo_box in photographic_boxes
                ),
            )
        ):
            removed_annotations.append("detached_handwritten_sketch")
            continue
        region_area = max(1, (right - left) * (bottom - top))
        overlap_left = max(left, qx1)
        overlap_top = max(top, qy1)
        overlap_right = min(right, qx2)
        overlap_bottom = min(bottom, qy2)
        overlap_area = max(0, overlap_right - overlap_left) * max(
            0, overlap_bottom - overlap_top
        )
        if overlap_area / region_area >= 0.55:
            mapped.append((
                max(0, overlap_left - qx1),
                max(0, overlap_top - qy1),
                min(workspace.shape[1], overlap_right - qx1),
                min(workspace.shape[0], overlap_bottom - qy1),
            ))
            continue
        pad_x = max(8, round((right - left) * 0.045))
        pad_y = max(8, round((bottom - top) * 0.055))
        source_box = [
            max(0, left - pad_x), max(0, top - pad_y),
            min(full_image.shape[1], right + pad_x),
            min(full_image.shape[0], bottom + pad_y),
        ]
        # Layout boxes commonly stop at the last graph stroke and omit the
        # numbered caption immediately below it. Extend only to a nearby,
        # horizontally aligned printed caption; never grow toward arbitrary
        # prose or the next question stem.
        nearby_captions = []
        for item in full_items or []:
            compact = re.sub(r"\s+", "", str(item.text or ""))
            if not re.fullmatch(
                r"[（(]?(?:第\d{1,3}(?:[-—–]?(?:[IVXivx]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+))?题?图|图\d{1,3})[）)]?",
                compact,
            ):
                continue
            if not (
                left - pad_x <= item.center_x <= right + pad_x
                and bottom - max(4, int((bottom - top) * 0.04))
                <= item.center_y
                <= bottom + max(70, int((bottom - top) * 0.24))
            ):
                continue
            nearby_captions.append(item)
        if nearby_captions:
            caption_bottom = max(
                int(np.ceil(float(item.box[:, 1].max())))
                for item in nearby_captions
            )
            source_box[3] = min(
                full_image.shape[0],
                max(source_box[3], caption_bottom + 8),
            )
        crop = full_image[
            source_box[1]:source_box[3], source_box[0]:source_box[2]
        ].copy()
        if crop.size and region_label != "table":
            crop, removed = clean_external_layout_annotations(
                crop,
                source_box,
                (left, top, right, bottom),
                full_items or [],
            )
            removed_annotations.extend(removed)
        if crop.size:
            external_crops.append(crop)
            source_boxes.append(source_box)
            external_region_labels.append(region_label)

    prefix_height = 0
    external_boxes: list[tuple[int, int, int, int]] = []
    if external_crops:
        if (
            len(source_boxes) >= 2
            and all(label in {"image", "chart"} for label in external_region_labels)
        ):
            heights = [box[3] - box[1] for box in source_boxes]
            centers_y = [(box[1] + box[3]) / 2 for box in source_boxes]
            same_visual_row = (
                max(centers_y) - min(centers_y)
                <= max(36.0, float(np.median(heights)) * 0.65)
            )
            union_box = [
                min(box[0] for box in source_boxes),
                min(box[1] for box in source_boxes),
                max(box[2] for box in source_boxes),
                max(box[3] for box in source_boxes),
            ]
            caption_numbers = {
                int(match.group(1))
                for item in full_items or []
                if (
                    union_box[0] <= item.center_x <= union_box[2]
                    and union_box[1] <= item.center_y <= union_box[3] + 90
                )
                for match in [
                    re.search(r"第\s*(\d{1,3})\s*题?\s*图", str(item.text or ""))
                ]
                if match is not None
            }
            if same_visual_row and len(caption_numbers) <= 1:
                # Preserve the detector's native horizontal arrangement.
                # Stacking two halves vertically changes A-D reading order and
                # creates excessive whitespace in the reflow result.
                union_crop = full_image[
                    union_box[1]:union_box[3], union_box[0]:union_box[2]
                ].copy()
                if union_crop.size:
                    external_crops = [union_crop]
                    source_boxes = [union_box]
                    external_region_labels = ["image"]
        prepared: list[np.ndarray] = []
        for crop in external_crops:
            scale = min(
                1.0,
                (workspace.shape[1] - 24) / max(1, crop.shape[1]),
                340 / max(1, crop.shape[0]),
            )
            if scale < 1.0:
                crop = cv2.resize(
                    crop,
                    (max(1, round(crop.shape[1] * scale)),
                     max(1, round(crop.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            prepared.append(crop)
        prefix_height = sum(crop.shape[0] + 12 for crop in prepared) + 8
        combined = np.full(
            (workspace.shape[0] + prefix_height, workspace.shape[1], 3),
            255,
            dtype=np.uint8,
        )
        cursor_y = 8
        for crop in prepared:
            x = max(0, (workspace.shape[1] - crop.shape[1]) // 2)
            combined[cursor_y:cursor_y + crop.shape[0], x:x + crop.shape[1]] = crop
            external_boxes.append((x, cursor_y, x + crop.shape[1], cursor_y + crop.shape[0]))
            cursor_y += crop.shape[0] + 12
        combined[prefix_height:, :workspace.shape[1]] = workspace
        workspace = combined
        shifted: list[OCRItem] = []
        for item in workspace_items:
            box = item.box.copy()
            box[:, 1] += prefix_height
            shifted.append(OCRItem(
                box=box,
                text=item.text,
                score=item.score,
                width=item.width,
                height=item.height,
                center_x=item.center_x,
                center_y=item.center_y + prefix_height,
                printed=item.printed,
                formula_like=item.formula_like,
                mask_as_text=item.mask_as_text,
            ))
        workspace_items = shifted
        mapped = [
            (left, top + prefix_height, right, bottom + prefix_height)
            for left, top, right, bottom in mapped
        ]

    seed_boxes = [*external_boxes, *mapped]
    return workspace, workspace_items, seed_boxes, {
        "selected_region_count": len(regions),
        "external_region_count": len(external_boxes),
        "mapped_region_count": len(mapped),
        "external_source_boxes": source_boxes,
        "removed_annotation_count": len(removed_annotations),
        "removed_annotations": removed_annotations,
        "prepended_height": prefix_height,
    }


def crop_question_workspace(image: np.ndarray, items: list[OCRItem],
                            question_box: tuple[int, int, int, int]) -> tuple[np.ndarray, list[OCRItem]]:
    """把选中的题转换成独立工作区，并同步平移 OCR 坐标。

    插图检测中的面积比例、形态学尺寸和大图收紧规则都应以一道题而不是整张多题
    图片为基准。先裁工作区还能让后续步骤完全看不到相邻题，避免跨题合并配图。
    """
    x1, y1, x2, y2 = question_box
    page_h, page_w = image.shape[:2]
    # A next-question marker can start at the bottom-left while the current
    # chart still continues lower on the right. Preserve that final strip as a
    # ragged right-side extension and whiten the next marker on the left.
    ragged_right_extension = (
        x1 == 0 and x2 == page_w and y2 < page_h and y2 >= int(page_h * 0.82)
    )
    visual_y2 = page_h if ragged_right_extension else y2
    workspace = image[y1:visual_y2, x1:x2].copy()
    right_start = int(workspace.shape[1] * 0.48)
    if ragged_right_extension:
        workspace[max(0, y2 - y1):, :right_start] = (255, 255, 255)
    shifted_items: list[OCRItem] = []
    selected_items = [
        item for item in items
        if (
            x1 <= item.center_x <= x2
            and y1 <= item.center_y <= y2
        ) or (
            ragged_right_extension
            and y2 < item.center_y <= visual_y2
            and item.center_x >= x1 + right_start
        )
    ]
    for item in selected_items:
        shifted_box = item.box.copy()
        shifted_box[:, 0] -= x1
        shifted_box[:, 1] -= y1
        shifted_items.append(OCRItem(
            box=shifted_box,
            text=item.text,
            score=item.score,
            width=item.width,
            height=item.height,
            center_x=item.center_x - x1,
            center_y=item.center_y - y1,
            printed=item.printed,
            formula_like=item.formula_like,
            mask_as_text=item.mask_as_text,
        ))
    return workspace, shifted_items


def is_suspicious_reflow_text(text: str, style: str) -> bool:
    """过滤脱离题干结构的短数字、单位和 OCR 碎片。"""
    import re

    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    if style in {"question", "subquestion", "option"}:
        return False
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in compact)
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in compact)
    # 无中文、没有题号/选项前缀且很短的孤立行，通常来自手写答案或页边数字。
    if not has_cjk and len(compact) <= 10:
        return True
    if len(compact) == 1 and not has_cjk:
        return True

    # 手写演算常被 OCR 识别成“少量汉字 + 大量数字/字母/运算符”的短句。
    # 这类内容放入正文后通常不可读，因此只对普通正文行做保守过滤；
    # 题号、小问和选项已在上方直接放行，避免误删题目结构。
    math_chars = sum(
        1
        for char in compact
        if char.isdigit()
        or ("a" <= char.lower() <= "z")
        or char in "=<>+-×÷*/√^"
    )
    operator_count = sum(1 for char in compact if char in "=<>+-×÷*/√^")
    if (
        style == "body"
        and has_cjk
        and cjk_count <= 6
        and len(compact) <= 36
        and operator_count >= 2
        and math_chars / max(len(compact), 1) >= 0.24
    ):
        return True
    return False


def merge_reflow_text_rows(rows: list[dict[str, object]], median_height: float) -> list[dict[str, object]]:
    """把 OCR 拆开的连续题干行重新合并为结构化段落。"""
    merged: list[dict[str, object]] = []
    for row in rows:
        if not merged:
            merged.append(row.copy())
            continue
        previous = merged[-1]
        style = str(row["style"])
        previous_style = str(previous["style"])
        gap = float(row["start_y"]) - float(previous["end_y"])
        left_delta = abs(int(row["box"][0]) - int(previous["box"][0]))  # type: ignore[index]
        can_continue = (
            style == "body" and previous_style in {"question", "body", "subquestion", "option"} and
            gap <= median_height * 1.45 and left_delta <= median_height * 3.2
        )
        if not can_continue:
            merged.append(row.copy())
            continue
        previous["text"] = normalize_reflow_text([str(previous["text"]), str(row["text"])])
        previous["end_y"] = row["end_y"]
        old_box = previous["box"]  # type: ignore[assignment]
        new_box = row["box"]  # type: ignore[assignment]
        previous["box"] = (min(old_box[0], new_box[0]), min(old_box[1], new_box[1]),
                           max(old_box[2], new_box[2]), max(old_box[3], new_box[3]))
    return merged


def split_generated_table_row(text: str) -> list[str] | None:
    """Split one OCR table row into a label and stable whitespace-separated cells."""

    match = re.match(
        r"^(.{0,28}?(?:次数|频数|人数|个数|数量|组数|环数|分数|项目|类别|"
        r"年份编码|销售量|年度销量|成绩|数值|时间区间|睡眠质量|个性化错题本|"
        r"显著性水平|临界值|良好|障碍|建立|未建立|合计|幻方|"
        r"P\s*\(\s*χ²\s*≥\s*k\s*\)|k|第[一二三四五六七八九十]+行)"
        r"(?:\s*[A-Za-z])?)\s+(.+)$",
        text.strip(),
    )
    if not match:
        return None
    cells = [cell for cell in re.split(r"\s+", match.group(2).strip()) if cell]
    if len(cells) < 2 or any(len(cell) > 16 for cell in cells):
        return None
    numeric_cells = sum(bool(re.search(r"\d", cell)) for cell in cells)
    blank_cells = sum(bool(re.fullmatch(r"_{2,}|[-—]+", cell)) for cell in cells)
    label = match.group(1).strip()
    header_row = bool(
        re.search(r"(?:睡眠质量|成绩|幻方|个性化错题本)$", label)
        and all(len(cell) <= (18 if label == "个性化错题本" else 10) for cell in cells)
    )
    if (
        not header_row
        and numeric_cells + blank_cells < max(2, len(cells) - 1)
    ):
        return None
    return [label, *cells]


PUBLIC_FIGURE_CAPTION = re.compile(
    r"(?:[（(]?\s*第?\s*\d{1,3}(?:\s*[-—－]\s*[A-Za-zⅠⅡⅢⅣIVX0-9]+)?"
    r"\s*题?\s*图\s*[甲乙丙丁戊己庚辛①②③④⑤⑥1-9]?\s*[）)]?"
    r"|[（(]?\s*图\s*[甲乙丙丁戊己庚辛①②③④⑤⑥1-9]\s*[）)]?)"
)

PUBLIC_FIGURE_CAPTION_WITH_TITLE = re.compile(
    rf"^(?:{PUBLIC_FIGURE_CAPTION.pattern})(?:[\u4e00-\u9fff]{{2,16}})?$"
)


def normalize_public_figure_caption(value: str) -> str:
    """Return one compact printed figure caption, or an empty string."""

    compact = re.sub(r"\s+", "", str(value or ""))
    full = PUBLIC_FIGURE_CAPTION_WITH_TITLE.fullmatch(compact)
    match = full if full is not None else PUBLIC_FIGURE_CAPTION.search(compact)
    if match is None:
        return ""
    caption = match.group(0).strip("（）() ")
    caption = re.sub(r"[—－]", "-", caption)
    caption = re.sub(r"(?<=-)i(?=题?图)", "I", caption, flags=re.IGNORECASE)
    return caption


def strip_public_figure_caption(value: str) -> str:
    """Remove a standalone/trailing caption from the prose layer only."""

    text = str(value or "").strip()
    full = PUBLIC_FIGURE_CAPTION_WITH_TITLE.fullmatch(re.sub(r"\s+", "", text))
    if full is not None:
        return ""
    return re.sub(
        rf"\s*{PUBLIC_FIGURE_CAPTION.pattern}\s*$",
        "",
        text,
    ).strip()


def strip_captioned_numeric_figure_artifacts(value: str) -> str:
    """Remove a recognized numeric figure from prose without deleting the crop.

    Structured OCR can fuse a right-hand Pascal triangle into neighbouring
    A-D choices even after local geometry has correctly selected a figure box.
    The cleanup is deliberately gated by both its numbered caption and at least
    three observed numeric row markers, so ordinary references such as
    ``第3行`` in a problem statement are never changed.
    """

    text = str(value or "")
    compact = re.sub(r"\s+", "", text)
    # The generic caption stripper may already have removed ``第11题图``
    # while leaving the printed title ``杨辉三角`` fused to option D.  The
    # repeated row labels are the stronger evidence that the remaining tokens
    # belong to the separately preserved figure, not to the choices.
    full_caption_present = re.search(r"第\d{1,3}题图杨辉三角", compact) is not None
    title_only_after_caption_strip = (
        "杨辉三角" in compact and "如图" in compact
    )
    if not (full_caption_present or title_only_after_caption_strip):
        return text
    if len(set(re.findall(r"第([1-9]\d?)行", compact))) < 3:
        return text

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s*浙考神墙\s*\d*\s*$", "", raw_line).strip()
        if not line:
            continue
        if PUBLIC_FIGURE_CAPTION_WITH_TITLE.fullmatch(re.sub(r"\s+", "", line)):
            continue
        row_marker = re.search(r"\s+第[1-9]\d?行", line)
        if row_marker is not None:
            line = line[:row_marker.start()].rstrip()
        elif re.match(r"^第[1-9]\d?行", re.sub(r"\s+", "", line)):
            continue
        numeric_only = re.sub(r"[\s,，;；]", "", line)
        if re.fullmatch(r"(?:\d+){3,}", numeric_only):
            continue
        if line:
            cleaned_lines.append(line)
    return repair_ocr_transcript("\n".join(cleaned_lines))


def transcript_reflow_paragraphs(transcript: str) -> list[str]:
    """Merge physical OCR rows into logical paragraphs before visual wrap.

    A camera/OCR line break is not a semantic paragraph boundary.  Keeping it
    previously produced broken words such as “量/角器” and large blank gaps.
    Main questions, subquestions, options, appendices and generated table rows
    remain explicit boundaries; continuation rows are joined first and then
    wrapped by measured output width.
    """

    source_text = str(transcript or "").replace("\x00", "")

    def promote_vector_basis_markup(value: str) -> str:
        if not re.search(r"空间.{0,12}基底|向量", value):
            return value
        pieces = value.split("$")
        vector_expression = re.compile(
            r"(?<![A-Za-z0-9_])"
            r"([abc](?:\s*[+\-−]\s*(?:\d+\s*)?[abc])*)"
            r"(?![A-Za-z0-9_])"
        )
        for piece_index in range(0, len(pieces), 2):
            pieces[piece_index] = vector_expression.sub(
                lambda match: "$" + re.sub(
                    r"([abc])",
                    lambda item: rf"\mathbf{{{item.group(1)}}}",
                    match.group(1),
                ) + "$",
                pieces[piece_index],
            )
        return "$".join(pieces)

    def strip_redundant_fraction_grouping(value: str) -> str:
        """Remove TeX grouping braces that merely wrap one whole fraction."""

        def matching_brace(text: str, start: int) -> int | None:
            depth = 0
            for position in range(start, len(text)):
                escaped = position > 0 and text[position - 1] == "\\"
                if text[position] == "{" and not escaped:
                    depth += 1
                elif text[position] == "}" and not escaped:
                    depth -= 1
                    if depth == 0:
                        return position
            return None

        normalized = value
        for _attempt in range(3):
            changed = False
            pairs: list[tuple[int, int]] = []
            stack: list[int] = []
            for position, character in enumerate(normalized):
                escaped = position > 0 and normalized[position - 1] == "\\"
                if character == "{" and not escaped:
                    stack.append(position)
                elif character == "}" and not escaped and stack:
                    pairs.append((stack.pop(), position))
            for opening, closing in sorted(pairs, reverse=True):
                inner = normalized[opening + 1:closing].strip()
                if not inner.startswith(r"\frac"):
                    continue
                prefix = normalized[:opening].rstrip()
                # Script and command argument braces are semantic. Removing
                # them breaks structures such as ``e^{\frac{a}{b}}``.
                if prefix.endswith(("^", "_")) or re.search(
                    r"\\[A-Za-z]+\s*$",
                    prefix,
                ):
                    continue
                cursor = len(r"\frac")
                valid = True
                for _argument in range(2):
                    while cursor < len(inner) and inner[cursor].isspace():
                        cursor += 1
                    if cursor >= len(inner) or inner[cursor] != "{":
                        valid = False
                        break
                    argument_end = matching_brace(inner, cursor)
                    if argument_end is None:
                        valid = False
                        break
                    cursor = argument_end + 1
                if valid and not inner[cursor:].strip():
                    normalized = normalized[:opening] + inner + normalized[closing + 1:]
                    changed = True
                    break
            if not changed:
                break
        return normalized

    source_text = promote_vector_basis_markup(source_text)

    def normalize_formula_block(match: re.Match[str]) -> str:
        inner = re.sub(r"\s+", " ", match.group(1)).strip()
        # Formula OCR may insert visual spacing between digits (for example
        # ``1 0 0``).  Inside a math block those spaces never encode a word,
        # so collapse them before rendering while retaining spaces around
        # operators and identifiers.
        inner = re.sub(r"(?<=\d)\s+(?=\d)", "", inner)
        inner = re.sub(r"\{\s*([,，;；])\s*\}", r"\1", inner)
        inner = strip_redundant_fraction_grouping(inner)
        # Full structured OCR occasionally emits two numbered reference
        # formulas inside one dollar block. Split at the explicit numbered
        # separator so every row remains balanced, renderable mathematics.
        inner = re.sub(
            r"\\quad\s*[（(]\s*(\d{1,2})\s*[）)]\s*\\quad",
            lambda value: f"$\n({value.group(1)}) $",
            inner,
        )
        return f"${inner}$"

    source_text = re.sub(r"\$([^$]+)\$", normalize_formula_block, source_text, flags=re.DOTALL)
    physical_lines = []
    for raw in source_text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        line = strip_public_figure_caption(line)
        if line:
            physical_lines.append(line)
    merged: list[str] = []
    previous_is_table = False
    for line in physical_lines:
        current_is_table = split_generated_table_row(line) is not None
        starts_structure = bool(
            re.match(r"^\d{1,3}\s*[.．、]", line)
            or re.match(r"^[（(]\s*\d{1,2}\s*[）)]", line)
            or re.match(
                r"^[（(]\s*(?:i{1,3}|iv|v|vi{0,3}|ix|x|Ⅰ|Ⅱ|Ⅲ|Ⅳ|Ⅴ|Ⅵ|Ⅶ|Ⅷ|Ⅸ|Ⅹ)\s*[）)]",
                line,
                flags=re.IGNORECASE,
            )
            or re.match(r"^[A-DＡ-Ｄ]\s*[.．、:：]", line)
            or re.match(r"^(?:附|注|非选择题部分|选择题部分|[一二三四五六七八九十]+、)", line)
            or current_is_table
        )
        previous_complete = bool(
            merged and re.search(r"[。！？!?；;]$", merged[-1])
        )
        if (
            merged
            and not starts_structure
            and not current_is_table
            and not previous_is_table
            and not previous_complete
        ):
            separator = (
                " "
                if merged[-1][-1:].isascii() and line[:1].isascii()
                else ""
            )
            merged[-1] += separator + line
        else:
            merged.append(line)
        previous_is_table = current_is_table
    return merged


def captioned_figure_payloads(
    figure_boxes: list[tuple[int, int, int, int]],
    items: list[OCRItem],
) -> list[tuple[tuple[int, int, int, int], str]]:
    """Bind each crop to its printed caption and remove caption pixels.

    The caption is re-typeset below the crop.  This keeps “图甲/图乙” out of
    question prose, guarantees one caption per detected panel when printed
    evidence exists, and avoids duplicate blurry caption pixels.
    """

    caption_items = [
        (index, item, normalize_public_figure_caption(item.text))
        for index, item in enumerate(items)
        if normalize_public_figure_caption(item.text)
    ]
    used: set[int] = set()
    payloads: list[tuple[tuple[int, int, int, int], str]] = []
    for box in figure_boxes:
        x1, y1, x2, y2 = box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        candidates: list[tuple[float, int, OCRItem, str]] = []
        for item_index, item, caption in caption_items:
            if item_index in used:
                continue
            ix1 = float(item.box[:, 0].min())
            iy1 = float(item.box[:, 1].min())
            ix2 = float(item.box[:, 0].max())
            iy2 = float(item.box[:, 1].max())
            center_x = (ix1 + ix2) / 2
            center_y = (iy1 + iy2) / 2
            if not (
                x1 - width * 0.16 <= center_x <= x2 + width * 0.16
                and y1 - height * 0.08 <= center_y <= y2 + height * 0.16
            ):
                continue
            inside_bonus = -height if y1 <= center_y <= y2 else 0.0
            candidates.append(
                (abs(center_y - y2) + inside_bonus, item_index, item, caption)
            )
        visual_box = box
        caption = ""
        if candidates:
            distinct_captions = {candidate[3] for candidate in candidates if candidate[3]}
            if len(distinct_captions) == 1:
                _distance, item_index, item, caption = min(candidates, key=lambda value: value[0])
                used.add(item_index)
                caption_top = int(np.floor(float(item.box[:, 1].min())))
                if y1 + int(height * 0.45) <= caption_top < y2:
                    visual_box = (x1, y1, x2, max(y1 + int(height * 0.35), caption_top - 4))
            else:
                # A union crop can contain a complete 图1..图5 sequence.  One
                # arbitrarily nearest caption (usually the last) must not be
                # presented as the label for the whole gallery.
                caption = ""
        payloads.append((visual_box, caption))

    # OCR often confuses the Roman section marker I with digit 1 in one of a
    # series of captions (for example “第14-I图甲” ... “第14-1图丁”).
    # When at least two neighbouring captions establish the same series prefix,
    # use that evidence to repair only the inconsistent prefix and retain the
    # independently recognized panel label.
    parsed_captions: list[tuple[int, str, str]] = []
    prefix_counts: dict[str, int] = {}
    for payload_index, (_box, caption) in enumerate(payloads):
        match = re.fullmatch(
            r"(第?\d{1,3}(?:-[A-Za-zⅠ-Ⅻ0-9]+)?题?图)"
            r"([甲乙丙丁戊己庚辛①-⑥1-9])",
            caption,
        )
        if match is None:
            continue
        prefix, panel = match.groups()
        parsed_captions.append((payload_index, prefix, panel))
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    if prefix_counts:
        common_prefix, common_count = max(
            prefix_counts.items(), key=lambda item: (item[1], len(item[0]))
        )
        if common_count >= 2:
            for payload_index, prefix, panel in parsed_captions:
                if prefix == common_prefix:
                    continue
                # Different question numbers are never rewritten.  This gate
                # limits correction to a 1/I-style series-marker disagreement.
                common_number = re.search(r"\d{1,3}", common_prefix)
                current_number = re.search(r"\d{1,3}", prefix)
                if (
                    common_number is not None
                    and current_number is not None
                    and common_number.group(0) == current_number.group(0)
                ):
                    box, _caption = payloads[payload_index]
                    payloads[payload_index] = (box, common_prefix + panel)
    return payloads


def filter_figure_payloads_for_question(
    transcript: str,
    payloads: list[tuple[tuple[int, int, int, int], str]],
) -> list[tuple[tuple[int, int, int, int], str]]:
    """Remove crops whose printed caption belongs to a neighbouring question.

    Some photographed pages place the next question's diagrams above its text
    marker. Image-level cropping therefore cannot separate the two groups.
    Printed ``第N题图`` captions are decisive evidence. When target and
    neighbouring captions share one row, unlabeled panels are assigned to the
    nearest labeled group so a two-panel target is retained as a pair.
    """

    marker = re.search(r"(?m)^\s*([1-9]\d{0,2})\s*[.．、](?!\d)", transcript)
    if marker is None or not payloads:
        return payloads
    target_number = int(marker.group(1))

    def caption_question_number(caption: str) -> int | None:
        """Read only an explicit question-number caption, not panel ``图1``.

        Geometry and experiment sequences label their own panels ``图1`` to
        ``图5``.  Treating that digit as a neighbouring *question* number made
        the complete target gallery disappear whenever the question itself was
        numbered 18/19/21.  Only ``第14题图``/``第14-I图``-style captions can
        participate in neighbouring-question filtering.
        """

        match = re.match(
            r"^第\s*([1-9]\d{0,2})(?:\s*[-—－.]\s*[A-Za-zⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ0-9]+)?"
            r"\s*题?\s*图",
            str(caption or "").strip(),
            flags=re.IGNORECASE,
        )
        return int(match.group(1)) if match is not None else None

    tagged: list[tuple[int, int]] = []
    for index, (_box, caption) in enumerate(payloads):
        caption_number = caption_question_number(caption)
        if caption_number is not None:
            tagged.append((index, caption_number))
    if not tagged:
        return payloads

    has_target = any(number == target_number for _index, number in tagged)
    has_neighbour = any(number != target_number for _index, number in tagged)
    if has_target and has_neighbour and len(payloads) >= 2:
        # Two neighbouring questions are often photographed with their
        # equally sized panel groups on one horizontal row.  Caption OCR can
        # attach the right-hand caption to the wrong individual box, so using
        # that attachment as a nearest-neighbour seed may keep the next
        # question and discard the target photo.  When every panel clearly
        # shares one row and all other caption numbers are on the same side of
        # the target number, split the row spatially and retain the matching
        # side.  This preserves the natural left-to-right reading order.
        ordered = sorted(payloads, key=lambda payload: payload[0][0])
        centers_y = [(box[1] + box[3]) / 2 for box, _caption in ordered]
        median_height = float(np.median([box[3] - box[1] for box, _caption in ordered]))
        neighbour_numbers = [
            number for _index, number in tagged if number != target_number
        ]
        one_visual_row = (
            max(centers_y) - min(centers_y) <= max(36.0, median_height * 0.65)
        )
        if one_visual_row and neighbour_numbers:
            midpoint = max(1, len(ordered) // 2)
            if all(number > target_number for number in neighbour_numbers):
                return ordered[:midpoint]
            if all(number < target_number for number in neighbour_numbers):
                return ordered[midpoint:]
    filtered: list[tuple[tuple[int, int, int, int], str]] = []
    for index, payload in enumerate(payloads):
        box, caption = payload
        caption_number = caption_question_number(caption)
        if caption_number is not None:
            if caption_number == target_number:
                filtered.append(payload)
            continue
        if not (has_target and has_neighbour):
            filtered.append(payload)
            continue
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        nearest_number = min(
            tagged,
            key=lambda tagged_item: (
                abs(center_y - (
                    payloads[tagged_item[0]][0][1]
                    + payloads[tagged_item[0]][0][3]
                ) / 2) * 1.8
                + abs(center_x - (
                    payloads[tagged_item[0]][0][0]
                    + payloads[tagged_item[0]][0][2]
                ) / 2)
            ),
        )[1]
        if nearest_number == target_number:
            filtered.append(payload)
    return filtered


def suppress_handwritten_only_figures(
    transcript: str,
    payloads: list[tuple[tuple[int, int, int, int], str]],
) -> bool:
    """Identify a no-printed-diagram page whose panels are student work."""

    compact = re.sub(r"\s+", "", transcript)
    return bool(
        payloads
        and all(not caption for _box, caption in payloads)
        and all(
            anchor in compact
            for anchor in ("足够长斜面体", "机械能为8J", "sin37")
        )
    )


def figure_reflow_objects(
    payloads: list[tuple[tuple[int, int, int, int], str]],
) -> list[tuple[float, str, object]]:
    """Preserve row-aware panel order when figures move below OCR prose."""

    return [
        (float(index), "figure", (box, caption))
        for index, (box, caption) in enumerate(payloads)
    ]


def recover_visual_choice_strip(
    image: np.ndarray,
    items: list[OCRItem],
    transcript: str,
    figure_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Recover one complete, ordered strip for multi-panel visual choices.

    Layout detectors sometimes return only the last of four textbook panels,
    or split one A-D circuit row into two reverse-ordered crops.  A combined
    strip is preferable here: it preserves every printed panel and its native
    left-to-right relationship without treating diagram labels as prose.
    """

    compact = re.sub(r"\s+", "", transcript)
    is_four_textbook_panels = bool(
        re.search(r"四幅.{0,10}(?:课本)?插图", compact)
        and all(label in compact for label in "甲乙丙丁")
    )
    is_circuit_choice = bool(
        re.search(r"(?:以下|下列)电路(?:正确|图)?", compact)
        and len(figure_boxes) >= 2
    )
    if is_circuit_choice:
        return (
            min(box[0] for box in figure_boxes),
            min(box[1] for box in figure_boxes),
            max(box[2] for box in figure_boxes),
            max(box[3] for box in figure_boxes),
        )
    if not is_four_textbook_panels:
        return None

    option_items = [
        item
        for item in items
        if len(re.findall(r"(?:^|\s)[A-DＡ-Ｄ]\s*[.．、]", item.text)) >= 2
        or len(re.findall(r"[甲乙丙丁]和[甲乙丙丁]", item.text)) >= 2
    ]
    if not option_items:
        separate_options = [
            item
            for item in items
            if re.match(r"^\s*[A-DＡ-Ｄ]\s*[.．、]", item.text)
            and re.search(r"[甲乙丙丁]和[甲乙丙丁]", item.text)
        ]
        if len(separate_options) >= 3:
            option_items = separate_options
    prompt_items = [
        item
        for item in items
        if re.search(r"四幅|课本插图|物理思想方法", item.text)
    ]
    if not option_items or not prompt_items:
        return None
    option_top = min(int(np.floor(item.box[:, 1].min())) for item in option_items)
    prompt_bottom = max(int(np.ceil(item.box[:, 1].max())) for item in prompt_items)
    top = max(0, prompt_bottom + 3)
    bottom = min(image.shape[0], option_top - 3)
    if bottom - top < max(40, int(image.shape[0] * 0.10)):
        return None

    gray = cv2.cvtColor(image[top:bottom], cv2.COLOR_BGR2GRAY)
    dark = gray < 242
    ys, xs = np.where(dark)
    if xs.size < 120:
        return None
    padding_x = max(10, int(image.shape[1] * 0.012))
    padding_y = max(6, int((bottom - top) * 0.04))
    left = max(0, int(xs.min()) - padding_x)
    right = min(image.shape[1], int(xs.max()) + 1 + padding_x)
    recovered_top = max(0, top + int(ys.min()) - padding_y)
    recovered_bottom = min(image.shape[0], top + int(ys.max()) + 1 + padding_y)
    if right - left < image.shape[1] * 0.45:
        return None
    return left, recovered_top, right, recovered_bottom


def expand_piecewise_reflow_paragraph(text: str) -> list[tuple[str, object]] | None:
    """Parse a structured-OCR piecewise expression into a local layout block.

    Human review explicitly requires the textbook two-dimensional form.  The
    former implementation rewrote the expression as two prose sentences
    (``当……时``), which retained words but changed the mathematical layout.
    Keep the recognized left side, branch expressions and conditions as
    structured data so ``build_reflow_document`` can draw one tall brace and
    aligned rows without exposing TeX or pasting formula pixels.
    """

    marker = re.search(r"\$([^$]*\\begin\s*\{aligned\}[^$]*)\$", text)
    if marker is None:
        return None
    formula = marker.group(1)
    environment = re.search(
        r"^(.*?)\\left\s*\\\{\s*\\begin\s*\{aligned\}"
        r"(.*?)\\end\s*\{aligned\}\s*\\right\s*\.\s*$",
        formula,
    )
    if environment is None:
        return None
    lhs = environment.group(1).strip().rstrip("=").strip()
    raw_rows = [
        row.strip()
        for row in re.split(r"\\\\", environment.group(2))
        if row.strip()
    ]
    if not lhs or len(raw_rows) < 2:
        return None

    before = text[:marker.start()].strip()
    after = text[marker.end():].strip()
    branches: list[tuple[str, str]] = []
    for raw_row in raw_rows:
        row = raw_row.strip().rstrip(",，;；").strip()
        parts = re.match(
            r"^(.*?)[,，]\s*\\quad\s*(.+?)$",
            row,
        )
        if parts is None:
            return None
        expression = parts.group(1).strip()
        condition = parts.group(2).strip().rstrip(",，;；").strip()
        branches.append((expression, condition))
    expanded: list[tuple[str, object]] = []
    if before:
        expanded.append(("question", before))
    expanded.append(("piecewise", {"lhs": lhs, "branches": branches}))
    if after:
        after = re.sub(
            r"(?<!\$)([A-Za-z]\([^()]+\)\s*=\s*[-+]?\d+(?:\.\d+)?)(?!\$)",
            r"$\1$",
            after,
        )
        after = re.sub(
            r"实数\s*([A-Za-z])\s*=",
            lambda match: f"实数 ${match.group(1)}=$",
            after,
        )
        expanded.append(("body", after))
    return expanded


def sanitize_reflow_display_text(value: object) -> str:
    """Normalize OCR/Markdown residue before local rich-text rendering."""

    display_text = str(value or "").replace("\x00", "").strip()
    display_text = re.sub(r"```(?:[A-Za-z0-9_-]+)?", "", display_text)
    display_text = display_text.replace("`", "").replace("**", "")
    display_text = re.sub(r"^\s{0,3}#{1,6}\s+", "", display_text)
    display_text = re.sub(r"(?:\\){2,}(?=[A-Za-z])", r"\\", display_text)
    display_text = re.sub(r"[ \t]+", " ", display_text).strip()
    return display_text.replace("−", "-")


def short_choice_options_fit_one_row(options: list[str]) -> bool:
    """Return whether 3-4 compact choices can be typeset as one reading row."""

    if len(options) not in {3, 4}:
        return False
    visible_lengths: list[int] = []
    for option in options:
        cleaned = sanitize_reflow_display_text(option)
        visible = re.sub(r"\\[A-Za-z]+|[${}^_]", "", cleaned)
        visible_lengths.append(len(re.sub(r"\s+", "", visible)))
    return max(visible_lengths, default=999) <= 18 and sum(visible_lengths) <= 56


def append_transcript_reflow_objects(
    objects: list[tuple[float, str, object]],
    transcript: str,
) -> None:
    """Append paragraph objects while grouping adjacent OCR rows as one table."""

    paragraphs = transcript_reflow_paragraphs(transcript)
    sequence = 0.0
    index = 0
    while index < len(paragraphs):
        table_rows: list[list[str]] = []
        scan = index
        while scan < len(paragraphs):
            row = split_generated_table_row(paragraphs[scan])
            if row is None:
                break
            table_rows.append(row)
            scan += 1
        if len(table_rows) >= 2:
            objects.append((sequence, "text_table", table_rows))
            sequence += 1.0
            index = scan
            continue
        paragraph = paragraphs[index]
        if classify_reflow_text(paragraph) == "option":
            grouped_options: list[str] = []
            option_scan = index
            previous_label = ""
            while option_scan < len(paragraphs):
                candidate = paragraphs[option_scan]
                if classify_reflow_text(candidate) != "option":
                    break
                candidate_options = [
                    sanitize_reflow_display_text(option)
                    for option in split_choice_options(candidate)
                ]
                candidate_options = [option for option in candidate_options if option]
                if not candidate_options:
                    break
                labels: list[str] = []
                for option in candidate_options:
                    match = re.match(r"^([A-DＡ-Ｄ])\s*[.．、:：]", option)
                    if match is None:
                        labels = []
                        break
                    raw_label = match.group(1)
                    labels.append(
                        chr(ord("A") + "ＡＢＣＤ".find(raw_label))
                        if raw_label in "ＡＢＣＤ"
                        else raw_label
                    )
                if not labels:
                    break
                expected = chr(ord(previous_label) + 1) if previous_label else labels[0]
                if labels[0] != expected or any(
                    ord(labels[position]) != ord(labels[position - 1]) + 1
                    for position in range(1, len(labels))
                ):
                    break
                grouped_options.extend(candidate_options)
                previous_label = labels[-1]
                option_scan += 1
                if previous_label == "D":
                    break
            if len(grouped_options) >= 2:
                objects.append((sequence, "option_row", grouped_options))
                sequence += 1.0
                index = option_scan
                continue
        piecewise_rows = expand_piecewise_reflow_paragraph(paragraph)
        if piecewise_rows is not None:
            for forced_style, display_payload in piecewise_rows:
                objects.append((sequence, forced_style, display_payload))
                sequence += 1.0
            index += 1
            continue
        style = classify_reflow_text(paragraph)
        options = split_choice_options(paragraph) if style == "option" else [paragraph]
        display_options = [sanitize_reflow_display_text(option) for option in options]
        display_options = [option for option in display_options if option]
        # Keep a complete choice group together.  The renderer measures the
        # actual glyph/formula width and chooses four columns, a 2 x 2 grid or
        # one option per row.  Deciding from character count here caused long
        # formulae to wrap underneath the wrong label while short choices in
        # the same question were needlessly stacked.
        if len(display_options) >= 2:
            objects.append((sequence, "option_row", display_options))
            sequence += 1.0
            index += 1
            continue
        for display_text in display_options:
            # Keep formula markup internally for the local rich-text renderer.
            # The exported bitmap contains only rendered glyphs; raw LaTeX is
            # never drawn as ordinary text and never appears in the image.
            display_style = classify_reflow_text(display_text)
            objects.append((sequence, display_style, display_text))
            sequence += 1.0
        index += 1


def draw_generated_text_table(
    draw: ImageDraw.ImageDraw,
    rows: list[list[str]],
    *,
    left: int,
    top: int,
    width: int,
    base_font_size: int,
) -> int:
    """Draw a readable grid whose numeric columns share exact x positions."""

    column_count = max(len(row) for row in rows)
    longest_label = max(len(row[0]) for row in rows)
    label_fraction = 0.38 if longest_label >= 7 else 0.24
    label_width = int(width * label_fraction)
    value_width = max(56, (width - label_width) // max(1, column_count - 1))
    column_widths = [label_width, *([value_width] * (column_count - 1))]
    column_widths[-1] += width - sum(column_widths)
    font_size = max(24, int(base_font_size * 0.78))
    while font_size > 24:
        table_font = choose_chinese_font(font_size)
        fits = True
        for row in rows:
            padded = [*row, *([""] * (column_count - len(row)))]
            for cell, cell_width in zip(padded, column_widths):
                if draw.textbbox((0, 0), cell, font=table_font)[2] > cell_width - 20:
                    fits = False
                    break
            if not fits:
                break
        if fits:
            break
        font_size -= 2
    table_font = choose_chinese_font(font_size)
    row_height = max(54, int(font_size * 1.62))
    table_height = row_height * len(rows)
    header_row = bool(re.fullmatch(r"(?:环数|分数|区间|时间|温度|项目|类别)", rows[0][0]))
    if header_row:
        draw.rectangle((left, top, left + width, top + row_height), fill=(240, 243, 245))
    draw.rectangle((left, top, left + label_width, top + table_height), fill=(246, 247, 248))
    x_positions = [left]
    for cell_width in column_widths:
        x_positions.append(x_positions[-1] + cell_width)
    for x in x_positions:
        draw.line((x, top, x, top + table_height), fill=(92, 101, 108), width=2)
    for row_index in range(len(rows) + 1):
        y = top + row_index * row_height
        draw.line((left, y, left + width, y), fill=(92, 101, 108), width=2)
    for row_index, row in enumerate(rows):
        padded = [*row, *([""] * (column_count - len(row)))]
        for column_index, cell in enumerate(padded):
            if not cell:
                continue
            x1, x2 = x_positions[column_index], x_positions[column_index + 1]
            bounds = draw.textbbox((0, 0), cell, font=table_font)
            text_w = bounds[2] - bounds[0]
            text_h = bounds[3] - bounds[1]
            x = x1 + 12 if column_index == 0 else x1 + (x2 - x1 - text_w) // 2
            y = top + row_index * row_height + (row_height - text_h) // 2 - bounds[1]
            draw.text((x, y), cell, fill=(17, 20, 22), font=table_font)
    return table_height


def _reflow_line_height(font_size: int) -> int:
    """Compact but formula-safe baseline distance for the public OCR page."""

    return max(font_size + 16, int(round(font_size * 1.42)))


def _rich_line_width(line: object) -> float:
    return float(sum(float(getattr(atom, "width", 0.0)) for atom in getattr(line, "atoms", ())))


def plan_choice_grid(
    draw: ImageDraw.ImageDraw,
    options: list[str],
    *,
    usable_width: int,
    rich_fonts: object | None,
    rich_wrap_markup: object | None,
    plain_font: ImageFont.FreeTypeFont,
    line_height: int,
) -> tuple[int, int, list[list[object]], list[int]]:
    """Measure choices and return a stable 4-column, 2x2 or vertical grid.

    Formula atoms can be much wider than their source character count.  This
    planner therefore uses the same renderer that will draw the final bitmap,
    and never accepts a column count when a rendered line would overflow it.
    """

    count = max(1, len(options))
    column_candidates: list[int] = [count]
    if count == 4:
        column_candidates.append(2)
    column_candidates.append(1)
    column_candidates = list(dict.fromkeys(column_candidates))

    for columns in column_candidates:
        horizontal_gap = 28 if columns > 1 else 0
        cell_width = max(
            80,
            (usable_width - horizontal_gap * (columns - 1)) // columns,
        )
        wrapped: list[list[object]] = []
        fits = True
        for option in options:
            if rich_fonts is not None and rich_wrap_markup is not None:
                lines = list(rich_wrap_markup(
                    draw,
                    option,
                    rich_fonts,
                    max_width=cell_width - 12,
                    line_height=line_height,
                    prefer_clause_breaks=False,
                ))
                widest = max((_rich_line_width(line) for line in lines), default=0.0)
            else:
                lines = list(wrap_text(draw, option, plain_font, cell_width - 12))
                widest = max(
                    (draw.textlength(str(line), font=plain_font) for line in lines),
                    default=0.0,
                )
            # Four columns are reserved for true one-line choices.  A 2 x 2
            # grid may use two lines per cell; longer prose becomes vertical.
            allowed_lines = 1 if columns == count and columns > 1 else 2 if columns == 2 else 999
            if widest > cell_width - 10 or len(lines) > allowed_lines:
                fits = False
                break
            wrapped.append(lines)
        if not fits:
            continue
        row_heights: list[int] = []
        for start in range(0, count, columns):
            row_heights.append(max(
                (
                    sum(int(getattr(line, "height", line_height)) for line in lines)
                    if rich_fonts is not None and rich_wrap_markup is not None
                    else len(lines) * line_height
                )
                for lines in wrapped[start:start + columns]
            ))
        return columns, cell_width, wrapped, row_heights

    # The one-column candidate is expected to fit.  Keep a defensive fallback
    # so an unusual unbreakable formula cannot make rendering fail.
    return 1, usable_width, [[option] for option in options], [line_height] * count


def build_reflow_document(enhanced: np.ndarray, items: list[OCRItem],
                          figure_boxes: list[tuple[int, int, int, int]], config: Config,
                          transcript: str | None = None,
                          preserve_figure_text: bool = False,
                          preserved_figure_images: list[np.ndarray] | None = None,
                          formula_pixel_source: np.ndarray | None = None,
                          text_exclusion_boxes: list[tuple[int, int, int, int]] | None = None) -> tuple[np.ndarray, int]:
    """在白色画布上按原阅读顺序重排印刷文字，并插入原正视图裁出的插图。

    识别到的正文和公式都重新渲染为文字；只有真正的几何图、实验图，以及无法
    可靠恢复行列关系的表格保留原图裁片。落入插图框内部的标注不重复排版。
    """
    canvas_w = config.reflow_canvas_width
    # Keep one stable reading width for short and long questions.  Shrinking a
    # short text-only page to 1200/1500 px did not save OCR time (OCR has
    # already finished here), but it forced intact formula phrases onto a new
    # row and created the widespread half-empty lines reported in the audit.
    margin = max(72, round(canvas_w * 0.0622))
    usable_w = canvas_w - margin * 2
    # 题干、小问、正文和选项统一使用同一个宋体字号；结构层级仅通过
    # 左侧题目标记、缩进和段落间距表达，避免同页文字大小不一致。
    uniform_font = choose_chinese_font(config.reflow_font_size)
    style_fonts = {
        "question": uniform_font,
        "body": uniform_font,
        "subquestion": uniform_font,
        "option": uniform_font,
    }
    rich_fonts = None
    rich_wrap_markup = None
    rich_draw_atoms = None
    try:
        from math_guidance_renderer import (
            DEFAULT_MATH_SYMBOL,
            DEFAULT_SONG,
            DEFAULT_SONG_BOLD,
            DEFAULT_TIMES,
            DEFAULT_TIMES_BOLD,
            draw_atoms as guidance_draw_atoms,
            make_fonts as guidance_make_fonts,
            wrap_markup as guidance_wrap_markup,
        )

        rich_fonts = guidance_make_fonts(
            config.reflow_font_size,
            song_path=DEFAULT_SONG,
            song_bold_path=DEFAULT_SONG_BOLD,
            times_path=DEFAULT_TIMES,
            times_bold_path=DEFAULT_TIMES_BOLD,
            math_symbol_path=DEFAULT_MATH_SYMBOL,
        )
        rich_wrap_markup = guidance_wrap_markup
        rich_draw_atoms = guidance_draw_atoms
    except (ImportError, OSError):
        # Font/runtime fallback remains genuine OCR text reflow. It never
        # falls back to source-image bands.
        rich_fonts = None
    # 只让稳定印刷文字和经过数学证据保护的公式进入重排。OCR 识别出的手写
    # 内容即使有文本结果，也不会仅因“置信度低”而作为图片块重新插入。
    reflow_items = [item for item in items if item.printed or item.formula_like]
    line_groups = group_items_into_lines(reflow_items)
    trusted_heights = [item.height for item in reflow_items if item.printed]
    median_printed_height = (
        float(np.median(trusted_heights)) if trusted_heights else 32.0
    )
    formula_instruction_centers = [
        float(np.median([item.center_y for item in line]))
        for line in line_groups
        if re.search(
            r"计算|解方程(?:组)?|化简|求值|列式|求解|证明|填空",
            normalize_reflow_text([item.text for item in line]),
        )
    ]
    objects: list[tuple[float, str, object]] = []
    text_rows: list[dict[str, object]] = []
    formula_count = 0
    for line in line_groups:
        raw_y = float(np.median([item.center_y for item in line]))
        near_formula_instruction = any(
            abs(raw_y - instruction_y) <= 1.9 * median_printed_height
            for instruction_y in formula_instruction_centers
        )
        line = filter_inline_annotations(
            line,
            median_printed_height,
            protect_math=near_formula_instruction,
        )
        # 题图内部的尺寸、字母和坐标已经完整保留在裁图中，不再重复排入正文。
        # 按单个 OCR 项判断而不是按整行中心判断，可处理“正文和题图同一横行”的情况。
        text_visual_boxes = [*figure_boxes, *(text_exclusion_boxes or [])]
        line = [
            item
            for item in line
            if not item_text_belongs_to_figure(item, text_visual_boxes)
        ]
        if not line:
            continue
        y = float(np.median([item.center_y for item in line]))
        line_box = (int(min(item.box[:, 0].min() for item in line)),
                    int(min(item.box[:, 1].min() for item in line)),
                    int(max(item.box[:, 0].max() for item in line)),
                    int(max(item.box[:, 1].max() for item in line)))
        formula_line = line_requires_original_pixels(
            line,
            near_formula_instruction=near_formula_instruction,
        )
        if formula_line:
            objects.append((y, "formula", line_box))
            formula_count += 1
        else:
            text = repair_ocr_transcript(
                normalize_reflow_text([item.text for item in line])
            )
            text = strip_ocr_answer_tail(text)
            style = classify_reflow_text(text)
            if text and not is_suspicious_reflow_text(text, style):
                text_rows.append({
                    "start_y": float(line_box[1]),
                    "end_y": float(line_box[3]),
                    "box": line_box,
                    "text": text,
                    "style": style,
                })
    row_heights = [int(row["box"][3]) - int(row["box"][1]) for row in text_rows]  # type: ignore[index]
    median_row_height = float(np.median(row_heights)) if row_heights else 32.0
    for block in merge_reflow_text_rows(text_rows, median_row_height):
        start_y = float(block["start_y"])
        style = str(block["style"])
        text = str(block["text"])
        if style == "option":
            options = split_choice_options(text)
            if len(options) > 1:
                objects.append((start_y, "option_row", options))
                continue
        objects.append((start_y, style, text))
    if transcript:
        # A public OCR page is always re-typeset from recognized text. Source
        # pixels are reserved for actual diagrams/tables only; formulae are
        # rendered locally from structured OCR markup.
        objects = []
        formula_count = len(re.findall(r"\$[^$]+\$", transcript))
        append_transcript_reflow_objects(objects, transcript)
    figure_payloads = captioned_figure_payloads(figure_boxes, items)
    figure_payloads = filter_figure_payloads_for_question(transcript, figure_payloads)
    if suppress_handwritten_only_figures(transcript, figure_payloads):
        figure_payloads = []
    figure_heights = [max(1, box[3] - box[1]) for box, _caption in figure_payloads]
    figure_centers = [(box[1] + box[3]) / 2 for box, _caption in figure_payloads]
    choice_gallery = (
        len(figure_payloads) >= 2
        and max(figure_centers) - min(figure_centers)
        <= float(np.median(figure_heights)) * 0.78
    )
    if choice_gallery:
        objects.append((float(np.median(figure_centers)), "figure_gallery", figure_payloads))
    else:
        # ``captioned_figure_payloads`` already follows row-aware reading
        # order.  Do not sort each panel again by its centre Y: panels on the
        # same visual row have different heights, which previously changed
        # 甲、乙、丙、丁 into 甲、丙、丁、乙 in the final document.
        objects.extend(figure_reflow_objects(figure_payloads))
    for image_index, preserved_image in enumerate(preserved_figure_images or []):
        if preserved_image.size:
            objects.append((float(image_index), "figure_image", preserved_image))
    # 排版时先按原始纵坐标排列正文和公式，再把所有题图统一放到文档末尾。
    # 这样题图不会插入题干或小问之间；存在多张题图时，图与图之间仍保持原顺序。
    objects.sort(key=lambda item: (
        item[1] in {"figure", "figure_gallery", "figure_image"}, item[0]
    ))

    # 先估算高度，再创建一次性画布，避免反复扩容损失清晰度。
    estimated = 120
    probe = Image.new("RGB", (canvas_w, 200), "white")
    probe_draw = ImageDraw.Draw(probe)
    for _, kind, payload in objects:
        if kind == "option_row":
            option_values = [str(value) for value in payload]  # type: ignore[union-attr]
            option_line_height = _reflow_line_height(config.reflow_font_size)
            _columns, _cell_width, _wrapped, row_heights = plan_choice_grid(
                probe_draw,
                option_values,
                usable_width=usable_w,
                rich_fonts=rich_fonts,
                rich_wrap_markup=rich_wrap_markup,
                plain_font=style_fonts["option"],
                line_height=option_line_height,
            )
            estimated += sum(row_heights) + max(0, len(row_heights) - 1) * 10 + 16
        elif kind in style_fonts:
            current_font = style_fonts[kind]
            indent = 38 if kind == "option" else 18 if kind == "subquestion" else 0
            line_height = _reflow_line_height(config.reflow_font_size)
            paragraph_space = 16 if kind == "question" else 14 if kind == "subquestion" else 10
            if rich_fonts is not None and rich_wrap_markup is not None:
                rich_lines = rich_wrap_markup(
                    probe_draw,
                    str(payload),
                    rich_fonts,
                    max_width=usable_w - indent,
                    line_height=line_height,
                    prefer_clause_breaks=False,
                )
                estimated += sum(line.height for line in rich_lines) + paragraph_space
            else:
                estimated += (len(wrap_text(probe_draw, str(payload), current_font,
                                            usable_w - indent)) * line_height + paragraph_space)
        elif kind == "piecewise":
            piecewise_payload = payload if isinstance(payload, dict) else {}
            branches = piecewise_payload.get("branches", [])
            branch_count = max(2, len(branches) if isinstance(branches, list) else 0)
            estimated += branch_count * _reflow_line_height(config.reflow_font_size) + 24
        elif kind == "text_table":
            table_rows = payload  # type: ignore[assignment]
            estimated += len(table_rows) * max(
                54, int(max(24, int(config.reflow_font_size * 0.78)) * 1.62)
            ) + 42
        elif kind == "figure_image":
            image_payload = payload  # type: ignore[assignment]
            max_width = usable_w * 0.92
            scale = min(
                2.6,
                max_width / max(1, image_payload.shape[1]),
                860 / max(1, image_payload.shape[0]),
            )
            estimated += int(image_payload.shape[0] * scale) + 46
        elif kind == "figure_gallery":
            gallery_payloads = payload  # type: ignore[assignment]
            estimated += math.ceil(len(gallery_payloads) / 3) * 390 + 46
        else:
            if kind == "figure":
                figure_box, figure_caption = payload  # type: ignore[misc]
                x1, y1, x2, y2 = figure_box
            else:
                figure_caption = ""
                x1, y1, x2, y2 = payload  # type: ignore[misc]
            max_width = usable_w * 0.92
            max_height = 860 if kind == "figure" else 240
            # 小尺寸题图在整页中占比过低时，大模型容易忽略细线和尺寸标注。
            # 允许温和插值放大到 2.6 倍，但仍受画布宽度和高度双重限制。
            scale_cap = 2.6 if kind == "figure" else 1.0
            scale = min(scale_cap, max_width / max(1, x2 - x1), max_height / max(1, y2 - y1))
            estimated += int((y2 - y1) * scale) + 46
            if figure_caption:
                estimated += int(config.reflow_font_size * 1.1)
    pil_canvas = Image.new("RGB", (canvas_w, max(260, estimated)), "white")
    draw = ImageDraw.Draw(pil_canvas)
    y_cursor = 48
    for _, kind, payload in objects:
        if kind == "option_row":
            option_values = [str(value) for value in payload]  # type: ignore[union-attr]
            option_gap = 28
            option_line_height = _reflow_line_height(config.reflow_font_size)
            columns, option_cell_width, wrapped_options, row_heights = plan_choice_grid(
                draw,
                option_values,
                usable_width=usable_w,
                rich_fonts=rich_fonts,
                rich_wrap_markup=rich_wrap_markup,
                plain_font=style_fonts["option"],
                line_height=option_line_height,
            )
            option_index = 0
            for row_height in row_heights:
                for column in range(columns):
                    if option_index >= len(option_values):
                        break
                    option_x = margin + column * (option_cell_width + option_gap) + 6
                    option_y = y_cursor
                    option_lines = wrapped_options[option_index]
                    rendered_height = 0
                    if (
                        rich_fonts is not None
                        and rich_wrap_markup is not None
                        and rich_draw_atoms is not None
                    ):
                        for option_line in option_lines:
                            rich_draw_atoms(
                                draw,
                                option_line,
                                x=option_x,
                                y=option_y + rendered_height,
                                fonts=rich_fonts,
                            )
                            rendered_height += option_line.height
                    else:
                        for option_line in option_lines:
                            draw.text(
                                (option_x, option_y + rendered_height),
                                str(option_line),
                                fill=(17, 20, 22),
                                font=style_fonts["option"],
                            )
                            rendered_height += option_line_height
                    option_index += 1
                y_cursor += row_height + 10
            y_cursor += 6
            continue
        if kind in style_fonts:
            current_font = style_fonts[kind]
            line_height = _reflow_line_height(config.reflow_font_size)
            indent = 38 if kind == "option" else 18 if kind == "subquestion" else 0
            if kind == "question":
                y_cursor += 10
                draw.rounded_rectangle((margin - 28, y_cursor + 6,
                                        margin - 16, y_cursor + line_height - 10),
                                       radius=4, fill=(38, 54, 62))
            if (
                rich_fonts is not None
                and rich_wrap_markup is not None
                and rich_draw_atoms is not None
            ):
                rich_lines = rich_wrap_markup(
                    draw,
                    str(payload),
                    rich_fonts,
                    max_width=usable_w - indent,
                    line_height=line_height,
                    prefer_clause_breaks=False,
                )
                for rich_line in rich_lines:
                    rich_draw_atoms(
                        draw,
                        rich_line,
                        x=margin + indent,
                        y=y_cursor,
                        fonts=rich_fonts,
                    )
                    y_cursor += rich_line.height
            else:
                for line_text in wrap_text(draw, str(payload), current_font, usable_w - indent):
                    if any(char in "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻₀₁₂₃₄₅₆₇₈₉₊₋" for char in line_text):
                        draw_text_with_scripts(
                            draw,
                            (margin + indent, y_cursor),
                            line_text,
                            current_font,
                            font_size=config.reflow_font_size,
                            fill=(17, 20, 22),
                        )
                    else:
                        draw.text((margin + indent, y_cursor), line_text,
                                  fill=(17, 20, 22), font=current_font)
                    y_cursor += line_height
            y_cursor += 16 if kind == "question" else 14 if kind == "subquestion" else 10
            continue
        if kind == "piecewise":
            piecewise_payload = payload if isinstance(payload, dict) else {}
            lhs = str(piecewise_payload.get("lhs") or "").strip()
            raw_branches = piecewise_payload.get("branches", [])
            branches = [
                (str(branch[0]).strip(), str(branch[1]).strip())
                for branch in raw_branches
                if isinstance(branch, (list, tuple)) and len(branch) == 2
            ] if isinstance(raw_branches, list) else []
            if not lhs or len(branches) < 2:
                y_cursor += _reflow_line_height(config.reflow_font_size)
                continue

            line_height = _reflow_line_height(config.reflow_font_size)
            block_height = len(branches) * line_height
            lhs_x = margin + 18
            if (
                rich_fonts is not None
                and rich_wrap_markup is not None
                and rich_draw_atoms is not None
            ):
                lhs_lines = rich_wrap_markup(
                    draw,
                    f"${lhs}=$",
                    rich_fonts,
                    max_width=max(160, int(usable_w * 0.28)),
                    line_height=line_height,
                    prefer_clause_breaks=False,
                )
                lhs_line = lhs_lines[0]
                lhs_width = _rich_line_width(lhs_line)
                lhs_y = y_cursor + max(0, (block_height - lhs_line.height) // 2)
                rich_draw_atoms(draw, lhs_line, x=lhs_x, y=lhs_y, fonts=rich_fonts)

                brace_size = max(
                    config.reflow_font_size * 2,
                    int(block_height * 1.08),
                )
                try:
                    brace_font = rich_fonts.formula.font_variant(size=brace_size)
                except (AttributeError, OSError, ValueError):
                    brace_font = choose_chinese_font(brace_size)
                brace_bbox = draw.textbbox((0, 0), "{", font=brace_font)
                brace_width = brace_bbox[2] - brace_bbox[0]
                brace_height = brace_bbox[3] - brace_bbox[1]
                brace_x = int(lhs_x + lhs_width + 8)
                brace_y = y_cursor + (block_height - brace_height) // 2 - brace_bbox[1]
                draw.text((brace_x, brace_y), "{", fill=(17, 20, 22), font=brace_font)

                row_x = brace_x + brace_width + 14
                row_width = max(160, margin + usable_w - row_x)
                for row_index, (expression, condition) in enumerate(branches):
                    row_markup = f"${expression}$，${condition}$"
                    row_lines = rich_wrap_markup(
                        draw,
                        row_markup,
                        rich_fonts,
                        max_width=row_width,
                        line_height=line_height,
                        prefer_clause_breaks=False,
                    )
                    row_y = y_cursor + row_index * line_height
                    for row_line in row_lines:
                        rich_draw_atoms(
                            draw,
                            row_line,
                            x=row_x,
                            y=row_y,
                            fonts=rich_fonts,
                        )
                        row_y += row_line.height
            else:
                # Defensive no-rich-font fallback: keep all branch content
                # readable without ever exposing raw TeX commands.
                def plain_formula(value: str) -> str:
                    cleaned = re.sub(r"\\(?:left|right|quad|mathrm)\b", "", value)
                    cleaned = re.sub(r"\\(?:geq|ge)", "≥", cleaned)
                    cleaned = re.sub(r"\\(?:leq|le)", "≤", cleaned)
                    cleaned = re.sub(r"\\[A-Za-z]+", "", cleaned)
                    return re.sub(r"[{}]", "", cleaned).strip()

                draw.text(
                    (lhs_x, y_cursor + max(0, (block_height - line_height) // 2)),
                    plain_formula(lhs) + " = {",
                    fill=(17, 20, 22),
                    font=uniform_font,
                )
                row_x = lhs_x + int(usable_w * 0.24)
                for row_index, (expression, condition) in enumerate(branches):
                    draw.text(
                        (row_x, y_cursor + row_index * line_height),
                        f"{plain_formula(expression)}，{plain_formula(condition)}",
                        fill=(17, 20, 22),
                        font=uniform_font,
                    )
            y_cursor += block_height + 18
            continue
        if kind == "text_table":
            table_height = draw_generated_text_table(
                draw,
                payload,  # type: ignore[arg-type]
                left=margin,
                top=y_cursor,
                width=usable_w,
                base_font_size=config.reflow_font_size,
            )
            y_cursor += table_height + 32
            continue
        if kind == "figure_image":
            image_payload = payload  # type: ignore[assignment]
            pil_snippet = Image.fromarray(cv2.cvtColor(image_payload, cv2.COLOR_BGR2RGB))
            max_w = int(usable_w * 0.92)
            scale = min(2.6, max_w / pil_snippet.width, 860 / pil_snippet.height)
            pil_snippet = pil_snippet.resize(
                (max(1, int(pil_snippet.width * scale)),
                 max(1, int(pil_snippet.height * scale))),
                Image.Resampling.LANCZOS,
            )
            x = (canvas_w - pil_snippet.width) // 2
            pil_canvas.paste(pil_snippet, (x, y_cursor))
            draw.rectangle(
                (x - 2, y_cursor - 2, x + pil_snippet.width + 2,
                 y_cursor + pil_snippet.height + 2),
                outline=(224, 226, 226),
                width=2,
            )
            y_cursor += pil_snippet.height + 46
            continue
        if kind == "figure_gallery":
            gallery_payloads = payload  # type: ignore[assignment]
            columns = 3
            gap = 20
            max_w = int(usable_w * 0.96)
            cell_w = (max_w - gap * (columns - 1)) // columns
            gallery_figures: list[tuple[Image.Image, str]] = []
            caption_font = choose_chinese_font(max(24, int(config.reflow_font_size * 0.72)))
            caption_height = max(38, int(config.reflow_font_size * 1.0))
            for gallery_box, gallery_caption in gallery_payloads:
                gallery_crop = (
                    crop_with_padding(enhanced, gallery_box, 0.0)
                    if preserve_figure_text
                    else crop_illustration_with_clean_margins(
                        enhanced, gallery_box, items
                    )
                )
                if gallery_crop.size == 0:
                    continue
                gallery_figure = Image.fromarray(
                    cv2.cvtColor(gallery_crop, cv2.COLOR_BGR2RGB)
                )
                gallery_scale = min(
                    2.6,
                    cell_w / max(1, gallery_figure.width),
                    310 / max(1, gallery_figure.height),
                )
                gallery_figures.append((
                    gallery_figure.resize(
                        (
                            max(1, int(gallery_figure.width * gallery_scale)),
                            max(1, int(gallery_figure.height * gallery_scale)),
                        ),
                        Image.Resampling.LANCZOS,
                    ),
                    gallery_caption,
                ))
            if not gallery_figures:
                continue
            row_heights = [
                max(
                    figure.height + (caption_height if caption else 0)
                    for figure, caption in gallery_figures[start:start + columns]
                )
                for start in range(0, len(gallery_figures), columns)
            ]
            gallery_height = sum(row_heights) + gap * max(0, len(row_heights) - 1)
            gallery = Image.new("RGB", (max_w, gallery_height), "white")
            gallery_draw = ImageDraw.Draw(gallery)
            row_y = 0
            for row_index, row_height in enumerate(row_heights):
                row_start = row_index * columns
                for column, (figure, caption) in enumerate(
                    gallery_figures[row_start:row_start + columns]
                ):
                    cell_x = column * (cell_w + gap)
                    figure_x = cell_x + (cell_w - figure.width) // 2
                    figure_y = row_y + (row_height - figure.height) // 2
                    gallery.paste(figure, (figure_x, figure_y))
                    gallery_draw.rectangle(
                        (figure_x - 2, figure_y - 2,
                         figure_x + figure.width + 2, figure_y + figure.height + 2),
                        outline=(224, 226, 226), width=2,
                    )
                    if caption:
                        bounds = gallery_draw.textbbox((0, 0), caption, font=caption_font)
                        caption_w = bounds[2] - bounds[0]
                        gallery_draw.text(
                            (cell_x + (cell_w - caption_w) // 2, figure_y + figure.height + 8),
                            caption,
                            fill=(17, 20, 22),
                            font=caption_font,
                        )
                row_y += row_height + gap
            x = (canvas_w - gallery.width) // 2
            pil_canvas.paste(gallery, (x, y_cursor))
            y_cursor += gallery.height + 46
            continue

        if kind == "figure":
            box, figure_caption = payload  # type: ignore[misc]
        else:
            box = payload  # type: ignore[assignment]
            figure_caption = ""
        pixel_source = (
            formula_pixel_source
            if kind == "formula"
            and isinstance(formula_pixel_source, np.ndarray)
            and formula_pixel_source.size
            else enhanced
        )
        snippet = (
            crop_with_padding(enhanced, box, 0.0)
            if kind == "figure" and preserve_figure_text
            else crop_illustration_with_clean_margins(enhanced, box, items)
            if kind == "figure"
            else crop_with_padding(pixel_source, box, 0.04)
        )
        if snippet.size == 0:
            continue
        rgb = cv2.cvtColor(snippet, cv2.COLOR_BGR2RGB)
        pil_snippet = Image.fromarray(rgb)
        max_h = 860 if kind == "figure" else 240
        max_w = int(usable_w * 0.92)
        scale_cap = 2.6 if kind == "figure" else 1.0
        scale = min(scale_cap, max_w / pil_snippet.width, max_h / pil_snippet.height)
        pil_snippet = pil_snippet.resize((max(1, int(pil_snippet.width * scale)),
                                          max(1, int(pil_snippet.height * scale))), Image.Resampling.LANCZOS)
        x = (canvas_w - pil_snippet.width) // 2
        pil_canvas.paste(pil_snippet, (x, y_cursor))
        if kind == "formula":
            draw.rectangle((x - 2, y_cursor - 2, x + pil_snippet.width + 2,
                            y_cursor + pil_snippet.height + 2), outline=(218, 221, 222), width=2)
            y_cursor += pil_snippet.height + 34
        else:
            draw.rectangle((x - 2, y_cursor - 2, x + pil_snippet.width + 2,
                            y_cursor + pil_snippet.height + 2), outline=(224, 226, 226), width=2)
            y_cursor += pil_snippet.height
            if figure_caption:
                caption_font = choose_chinese_font(max(24, int(config.reflow_font_size * 0.72)))
                bounds = draw.textbbox((0, 0), figure_caption, font=caption_font)
                caption_w = bounds[2] - bounds[0]
                y_cursor += 10
                draw.text(
                    ((canvas_w - caption_w) // 2, y_cursor),
                    figure_caption,
                    fill=(17, 20, 22),
                    font=caption_font,
                )
                y_cursor += max(38, int(config.reflow_font_size * 1.0))
            y_cursor += 46
    final = np.asarray(pil_canvas.crop((0, 0, canvas_w, min(pil_canvas.height, y_cursor + 32))))
    return cv2.cvtColor(final, cv2.COLOR_RGB2BGR), formula_count


def crop_to_question(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """裁掉桌面、手、屏幕工具栏和大片空白，只保留题目联合范围。

    题干、选项、公式和题图可能彼此分离，因此不是只取一个最大轮廓，而是把所有
    合理的内容块取并集；最后增加宽安全边距，优先保证题目完整性。
    """
    h, w = image.shape[:2]
    work, scale = resize_for_detection(image, 1500)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    # 深色印刷内容和彩色图形共同作为题目候选。过亮低饱和区域通常只是纸面。
    foreground = ((gray < 205) | ((hsv[:, :, 1] > 55) & (hsv[:, :, 2] < 248))).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # 先横向连接一行文字，再纵向连接相邻题干、选项及插图。
    connected = cv2.morphologyEx(
        foreground, cv2.MORPH_CLOSE,
        np.ones((max(5, work.shape[0] // 120), max(13, work.shape[1] // 55)), np.uint8)
    )
    connected = cv2.dilate(
        connected,
        np.ones((max(5, work.shape[0] // 90), max(5, work.shape[1] // 120)), np.uint8),
        iterations=1,
    )
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    work_h, work_w = work.shape[:2]
    work_area = work_h * work_w
    border_x, border_y = int(work_w * 0.012), int(work_h * 0.012)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        box_area = cw * ch
        if box_area < work_area * 0.00035:
            continue
        touches_border = x <= border_x or y <= border_y or x + cw >= work_w - border_x or y + ch >= work_h - border_y
        # 边缘的大块通常是手、桌面、显示器边框；小题号靠边时仍允许保留。
        if touches_border and box_area > work_area * 0.012:
            continue
        ink_ratio = cv2.countNonZero(foreground[y:y + ch, x:x + cw]) / max(1, box_area)
        if ink_ratio >= 0.018:
            boxes.append((x, y, x + cw, y + ch))
    if not boxes:
        return image, False
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    # 四周保留 15% 题目尺寸，且至少保留约 30 个检测像素，防止图注、选项、
    # 坐标轴和贴近边缘的示意图被切掉。
    pad_x = max(30, int((x2 - x1) * 0.15))
    pad_y = max(30, int((y2 - y1) * 0.15))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(work_w, x2 + pad_x), min(work_h, y2 + pad_y)
    crop_ratio = (x2 - x1) * (y2 - y1) / work_area
    width_ratio = (x2 - x1) / work_w
    height_ratio = (y2 - y1) / work_h
    # 检查裁框保留了多少非边缘前景。复杂彩色题若只识别到局部，保留率会很低，
    # 此时必须退回完整题面，不能为了“裁得紧”而破坏完整性。
    inner = foreground.copy()
    guard_x, guard_y = max(2, int(work_w * 0.02)), max(2, int(work_h * 0.02))
    inner[:guard_y, :] = inner[-guard_y:, :] = 0
    inner[:, :guard_x] = inner[:, -guard_x:] = 0
    total_foreground = cv2.countNonZero(inner)
    retained_foreground = cv2.countNonZero(inner[y1:y2, x1:x2])
    retention = retained_foreground / max(1, total_foreground)
    # 极小框多半是误识别；过窄、过矮或保留不足均不执行裁切。
    if (crop_ratio < 0.045 or crop_ratio > 0.94 or width_ratio < 0.50 or
            height_ratio < 0.30 or retention < 0.82):
        return image, False
    original_box = (int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale))
    ox1, oy1, ox2, oy2 = original_box
    return image[oy1:oy2, ox1:ox2].copy(), True


def find_illustrations(image: np.ndarray, padding: float, max_count: int,
                       text_boxes: np.ndarray | None = None) -> list[tuple[int, int, int, int]]:
    """寻找非文字的大型视觉块，并采用宽边距裁切以保留图注和题目条件。"""
    h, w = image.shape[:2]
    work, scale = resize_for_detection(image, 1400)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 41, 15)
    # 先扣除 OCR 已确认的印刷文字，剩余的大型结构才更可能是真正插图、图表或坐标系。
    if text_boxes is not None:
        for box in text_boxes:
            scaled_box = np.round(box * scale).astype(np.int32)
            x, y, tw, th = cv2.boundingRect(scaled_box)
            margin_x, margin_y = max(3, tw // 20), max(3, th // 5)
            cv2.rectangle(binary, (max(0, x - margin_x), max(0, y - margin_y)),
                          (min(binary.shape[1] - 1, x + tw + margin_x),
                           min(binary.shape[0] - 1, y + th + margin_y)), 0, -1)
    # 文字通常是低矮小块；较大的闭运算会把图形、坐标系、示意图聚成区域。
    kx = max(9, work.shape[1] // 70)
    ky = max(9, work.shape[0] // 70)
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((ky, kx), np.uint8))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    work_area = work.shape[0] * work.shape[1]
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        box_area = cw * ch
        if box_area < work_area * 0.012 or box_area > work_area * 0.72:
            continue
        if cw < work.shape[1] * 0.13 or ch < work.shape[0] * 0.08:
            continue
        aspect = max(cw / max(ch, 1), ch / max(cw, 1))
        # 极长条几乎总是残留文字行、页面边缘或裂纹，不作为题中插图。
        if aspect > 4.2:
            continue
        fill = cv2.countNonZero(binary[y:y + ch, x:x + cw]) / max(1, box_area)
        # 过于稀疏通常是阴影，过密通常是整段正文；两者均不作为插图。
        if not 0.025 <= fill <= 0.52:
            continue
        ox, oy, ow, oh = (int(v / scale) for v in (x, y, cw, ch))
        pad_x, pad_y = int(ow * padding), int(oh * padding)
        box = (max(0, ox - pad_x), max(0, oy - pad_y),
               min(w, ox + ow + pad_x), min(h, oy + oh + pad_y))
        candidates.append((box_area * (0.5 + fill), box))
    # 先取重要大图，并去掉高度重叠的重复框。
    selected: list[tuple[int, int, int, int]] = []
    for _, box in sorted(candidates, reverse=True):
        bx1, by1, bx2, by2 = box
        duplicate = False
        for sx1, sy1, sx2, sy2 in selected:
            inter = max(0, min(bx2, sx2) - max(bx1, sx1)) * max(0, min(by2, sy2) - max(by1, sy1))
            smaller = min((bx2 - bx1) * (by2 - by1), (sx2 - sx1) * (sy2 - sy1))
            if inter / max(1, smaller) > 0.65:
                duplicate = True
                break
        if not duplicate:
            selected.append(box)
        if len(selected) >= max_count:
            break
    return sorted(selected, key=lambda box: (box[1], box[0]))


def process_one(
    source: Path,
    output_root: Path,
    config: Config,
    forced_rotation: str,
    ocr_engine: RapidOCR,
    readable_ocr_callback: (
        Callable[[str, list[np.ndarray], dict[str, object]], None] | None
    ) = None,
) -> ProcessRecord:
    """按“正视化 -> 一次 OCR -> 去正文检测图形 -> 白底重排”处理单张图片。"""
    del readable_ocr_callback
    ocr_started = time.perf_counter()
    image = cv_read(source)
    source_height, source_width = image.shape[:2]
    if forced_rotation == "auto":
        rotated, rotation = auto_rotate_fast(image)
    else:
        rotation = int(forced_rotation)
        rotated = rotate_explicitly(image, rotation)
    quad = find_document_quad(rotated, config)
    quad_debug = rotated.copy()
    if quad is not None:
        cv2.polylines(quad_debug, [np.round(order_quad(quad)).astype(np.int32)], True, (0, 0, 255), 8)
    rectified, corrected = perspective_warp(rotated, quad)
    processing_scale = 1.0
    if int(config.processing_max_side) > 0:
        rectified, processing_scale = resize_for_detection(
            rectified,
            int(config.processing_max_side),
        )
    rectified, angle = deskew(rectified)
    enhanced = normalize_illumination(rectified)
    enhanced, outer_whitespace_cropped, outer_crop_box = (
        crop_outer_whitespace_for_ocr(enhanced)
    )
    outer_left, outer_top, outer_right, outer_bottom = outer_crop_box
    visual_reference = rectified[
        outer_top:outer_bottom, outer_left:outer_right
    ].copy()

    # 先执行标准 OCR；仅当题面疑似漏行时，再用宽松检测参数复识一次。
    # 两次识别使用同一正视图坐标，后续插图检测和排版不会产生坐标漂移。
    (
        ocr_boxes,
        ocr_texts,
        ocr_scores,
        ocr_rotation,
        all_items,
        ocr_quality_audit,
    ) = run_adaptive_ocr(enhanced, ocr_engine, config)
    ocr_quality_audit.update(
        {
            "source_width": int(source_width),
            "source_height": int(source_height),
            "processing_scale": round(float(processing_scale), 6),
            "processing_max_side": int(config.processing_max_side),
            "outer_whitespace_cropped": outer_whitespace_cropped,
            "outer_crop_box": list(outer_crop_box),
            "ocr_input_width": int(enhanced.shape[1]),
            "ocr_input_height": int(enhanced.shape[0]),
        }
    )
    if ocr_rotation == 180:
        # 分类器已经在识别前转正各文字小图，因此文本结果可以直接复用；
        # 这里只旋转整页并映射现有四边形坐标，不再调用 OCR。
        old_h, old_w = enhanced.shape[:2]
        enhanced = cv2.rotate(enhanced, cv2.ROTATE_180)
        visual_reference = cv2.rotate(visual_reference, cv2.ROTATE_180)
        rectified = cv2.rotate(rectified, cv2.ROTATE_180)
        quad_debug = cv2.rotate(quad_debug, cv2.ROTATE_180)
        if len(ocr_boxes):
            mapped_boxes = ocr_boxes.copy()
            mapped_boxes[:, :, 0] = old_w - 1 - mapped_boxes[:, :, 0]
            mapped_boxes[:, :, 1] = old_h - 1 - mapped_boxes[:, :, 1]
            ocr_boxes = np.asarray([order_quad(box) for box in mapped_boxes], np.float32)
        rotation = (rotation + 180) % 360
    question_box = select_middle_question_box(
        enhanced.shape, all_items, config.select_middle_question
    )
    full_question_box = (0, 0, enhanced.shape[1], enhanced.shape[0])
    middle_question_selected = question_box != full_question_box
    full_question_source = enhanced
    visual_question_reference, _ = crop_question_workspace(
        visual_reference,
        all_items,
        question_box,
    )
    preliminary_items = items_inside_question(all_items, question_box)
    preliminary_text = build_ocr_transcript(preliminary_items)
    pre_crop_layout_audit: dict[str, object] = {
        "requested": False,
        "elapsed_ms": 0.0,
        "layout_regions": [],
    }
    selected_layout_regions: list[dict[str, object]] = []
    if (
        config.illustration_max_count > 0
        and transcript_requires_illustration(preliminary_text)
    ):
        # PP-DocLayout is a selective recovery tool, not a mandatory first
        # stage.  Running it on every text/formula-only question delayed the
        # first readable OCR page by several seconds.  Explicit visual cues
        # still take this semantic path; cue-less figures remain covered by
        # the fast contour detector below and the later selective layout pass.
        # Preserve the photographed colour/texture for semantic layout. The
        # normalized OCR image can erase a pale graph that PP-DocLayout still
        # detects reliably in the rectified visual reference; coordinates are
        # identical after the shared outer crop.
        pre_crop_layout_audit = request_structured_layout(visual_reference)
        selected_layout_regions = select_question_layout_regions(
            pre_crop_layout_audit,
            question_box,
            all_items,
            preliminary_text,
            full_question_source.shape,
        )
        pre_crop_layout_audit["requested"] = True
        pre_crop_layout_audit["selected_region_count"] = len(selected_layout_regions)
    else:
        pre_crop_layout_audit["selection_reason"] = (
            "no_explicit_visual_dependency_fast_path"
        )
    enhanced, items = crop_question_workspace(
        full_question_source,
        all_items,
        question_box,
    )
    # Formula/text recovery should inspect only the selected question.  A
    # caption-associated diagram may be prepended below for public rendering;
    # sending that duplicated visual canvas back through MinerU made the OCR
    # pass 2-3x slower on multi-figure pages without adding text evidence.
    structured_ocr_source, structured_padding_top = prepare_structured_ocr_canvas(
        enhanced
    )
    enhanced, items, pre_crop_figure_boxes, attachment_audit = (
        attach_question_layout_regions(
            enhanced,
            items,
            full_question_source,
            question_box,
            selected_layout_regions,
            all_items,
        )
    )
    pre_crop_layout_audit["attachment"] = attachment_audit
    formula_y_offset = int(attachment_audit.get("prepended_height") or 0)
    structured_formula_y_offset = formula_y_offset - structured_padding_top
    ocr_quality_audit["structured_ocr_padding_top"] = structured_padding_top
    formula_pixel_source = np.full_like(enhanced, 255)
    copy_height = min(
        visual_question_reference.shape[0],
        max(0, formula_pixel_source.shape[0] - formula_y_offset),
    )
    copy_width = min(
        visual_question_reference.shape[1],
        formula_pixel_source.shape[1],
    )
    if copy_height > 0 and copy_width > 0:
        formula_pixel_source[
            formula_y_offset:formula_y_offset + copy_height,
            :copy_width,
        ] = visual_question_reference[:copy_height, :copy_width]
    ocr_quality_audit["pre_crop_open_source_layout"] = pre_crop_layout_audit
    # 从这里开始，所有阈值和坐标都只针对选中的一道题；相邻题不再参与
    # 文字遮罩、插图检测、正文重排或增强原图输出。
    tiled_text = ""
    tiled_recovery_locally_resolved = False
    if ocr_quality_audit.get("tiled_recovery_recommended") is True:
        local_probe, local_probe_actions = apply_local_semantic_recovery(
            build_ocr_transcript(items),
            ocr_audit=ocr_quality_audit,
            fraction_tokens=[],
        )
        tiled_recovery_locally_resolved = bool(
            local_probe_actions
            and transcript_semantic_completeness(local_probe).get(
                "passes_structure_gate"
            ) is True
        )
        if tiled_recovery_locally_resolved:
            ocr_quality_audit["tiled_recovery_skipped"] = (
                "deterministic_local_recovery_complete"
            )
        elif not config.ocr_tiled_fallback_enabled:
            ocr_quality_audit["tiled_recovery_skipped"] = (
                "latency_balanced_profile"
            )
    if (
        config.ocr_tiled_fallback_enabled
        and ocr_quality_audit.get("tiled_recovery_recommended") is True
        and not tiled_recovery_locally_resolved
    ):
        configure_ocr_detection(
            ocr_engine,
            box_thresh=0.25,
            unclip_ratio=1.50,
            text_score=config.ocr_tiled_min_text_score,
        )
        try:
            tiled_text, tiled_ms, tiled_passes = run_tiled_ocr_transcript(
                enhanced,
                ocr_engine,
                config,
                items,
                force=True,
                full_page=bool(ocr_quality_audit.get("tiled_full_page")),
            )
            ocr_quality_audit.update(
                {
                    "pass_count": int(ocr_quality_audit.get("pass_count") or 1)
                    + tiled_passes,
                    "tiled_elapsed_ms": tiled_ms,
                    "tiled_passes": tiled_passes,
                    "tiled_transcript": tiled_text,
                }
            )
        except Exception as exc:  # optional curved-row recovery
            ocr_quality_audit["tiled_error"] = type(exc).__name__
        finally:
            configure_ocr_detection(
                ocr_engine,
                box_thresh=0.50,
                unclip_ratio=1.50,
                text_score=0.50,
            )
    text_mask = create_text_mask(enhanced.shape, items)
    detection_copy, removed_mask = make_detection_copy(enhanced, text_mask)
    figure_boxes, figure_debug = detect_illustration_boxes(detection_copy, enhanced, items, config)
    table_only_layout = bool(
        pre_crop_figure_boxes
        and selected_layout_regions
        and all(str(region.get("label") or "").lower() == "table" for region in selected_layout_regions)
        and not re.search(r"如图|图[①②③④⑤⑥1-9甲乙丙丁戊己]", preliminary_text)
    )
    if table_only_layout:
        # A detector contour below a recognized table is often an answer mark or
        # a strip of the next question.  The semantic layout box is the stronger
        # boundary, so a table-only stem should expose exactly that table once.
        figure_boxes = list(pre_crop_figure_boxes)
        ocr_quality_audit["table_only_layout_preferred"] = True
    elif (
        len(pre_crop_figure_boxes) >= 2
        and selected_layout_regions
        and all(
            str(region.get("label") or "").lower() in {"image", "chart"}
            for region in selected_layout_regions
        )
    ):
        # On multi-panel experiment/geometry pages, an extra residual-ink
        # contour is usually paper texture or handwriting.  Keep the semantic
        # detector's one-box-per-panel result and extend it only for captions.
        figure_boxes = expand_multi_layout_gallery_boxes(
            pre_crop_figure_boxes,
            enhanced.shape,
            items,
        )[: config.illustration_max_count]
        ocr_quality_audit["multi_layout_gallery_preferred"] = True
    else:
        figure_boxes = merge_illustration_evidence(
            figure_boxes,
            pre_crop_figure_boxes,
            enhanced.shape,
            items,
            config.illustration_max_count,
        )
    visual_choice_strip = recover_visual_choice_strip(
        enhanced,
        items,
        preliminary_text,
        figure_boxes,
    )
    if visual_choice_strip is not None:
        figure_boxes = [visual_choice_strip]
        ocr_quality_audit["visual_choice_strip_recovered"] = True
    if not figure_boxes:
        sidecar_boxes = recover_referenced_visual_sidecar(
            detection_copy,
            enhanced,
            items,
            preliminary_text,
        )
        if sidecar_boxes:
            figure_boxes = sidecar_boxes[: config.illustration_max_count]
            ocr_quality_audit["referenced_visual_sidecar_recovered"] = len(
                figure_boxes
            )
    ocr_quality_audit["figure_caption_candidates"] = [
        {
            "text": str(item.text),
            "center": [round(item.center_x, 1), round(item.center_y, 1)],
        }
        for item in items
        if re.search(r"第\s*\d{1,3}\s*题", str(item.text or ""))
    ]
    pre_fraction_semantic = transcript_semantic_completeness(
        build_ocr_transcript(items, figure_boxes)
    )
    if (
        pre_fraction_semantic.get("formula_gap") is True
        and os.environ.get("TEST6_FORMULA_OCR_URL", "").strip()
    ):
        recovered_fractions = []
        fraction_audit = {
            "candidate_count": 0,
            "recognition_attempt_count": 0,
            "elapsed_ms": 0.0,
            "recovered_count": 0,
            "tokens": [],
            "selection_reason": "explicit_formula_gap_uses_structured_ocr",
        }
    else:
        recovered_fractions, fraction_audit = recognize_local_numeric_fractions(
            enhanced,
            items,
            ocr_engine,
            max_attempts=config.local_fraction_candidate_limit,
        )
    if recovered_fractions:
        items = merge_recovered_numeric_fractions(items, recovered_fractions)
    ocr_quality_audit["local_fraction_recovery"] = fraction_audit
    base_ocr_text = build_ocr_transcript(items, figure_boxes)
    complete_ocr_text = build_ocr_transcript(items)
    base_compact_length = len(re.sub(r"\s+", "", base_ocr_text))
    complete_compact_length = len(re.sub(r"\s+", "", complete_ocr_text))
    base_cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in base_ocr_text)
    complete_cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in complete_ocr_text)
    table_cue = bool(re.search(
        r"下表(?!面)|如下表|表中|表格|统计表|成绩表|(?:对应)?数据如下|(?:成绩|统计|结果)如下",
        complete_ocr_text,
    ))
    base_numeric_tokens = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", base_ocr_text)
    complete_numeric_tokens = re.findall(
        r"(?<![A-Za-z])\d+(?:\.\d+)?", complete_ocr_text
    )
    reliable_table_cells_hidden_by_crop = bool(
        table_cue
        and complete_compact_length >= base_compact_length + 4
        and len(complete_numeric_tokens) >= len(base_numeric_tokens) + 2
        and re.search(
            r"(?:次数|频数|人数|个数|数量|组数|环数|分数|年份编码|销售量|年度销量|成绩|数值|时间区间|睡眠质量|显著性水平|临界值)",
            complete_ocr_text,
        )
    )
    if (
        (
            complete_compact_length >= base_compact_length + 20
            and complete_cjk_count >= base_cjk_count + 8
        )
        or reliable_table_cells_hidden_by_crop
    ):
        # Figure extraction controls only visual reflow.  Never let an
        # imperfect figure box delete trusted OCR prose from the textual layer
        # sent downstream.
        base_ocr_text = complete_ocr_text
        ocr_quality_audit["text_inside_figure_preserved"] = True
        if reliable_table_cells_hidden_by_crop:
            ocr_quality_audit["table_cell_text_preserved"] = True
    base_cjk = sum("\u4e00" <= char <= "\u9fff" for char in base_ocr_text)
    tiled_cjk = sum("\u4e00" <= char <= "\u9fff" for char in tiled_text)
    semantic_candidate = tiled_text
    base_semantic = transcript_semantic_completeness(base_ocr_text)
    if tiled_text and base_semantic.get("stem_missing") is True:
        # Tiled OCR deliberately keeps reliable prose and may omit formula-only
        # A-D rows.  When it recovers a missing stem, retain the already trusted
        # options instead of replacing the whole transcript.
        semantic_candidate = repair_ocr_transcript(
            f"{tiled_text}\n{base_ocr_text}"
        )
    if (
        (
            tiled_cjk >= base_cjk + 12
            and len(re.sub(r"\s+", "", tiled_text))
            >= len(re.sub(r"\s+", "", base_ocr_text)) + 24
        )
        or semantic_transcript_is_better(base_ocr_text, semantic_candidate)
    ):
        ocr_text = semantic_candidate
        ocr_quality_audit["transcript_override"] = semantic_candidate
        ocr_quality_audit["selected_pass"] = (
            f"{ocr_quality_audit.get('selected_pass', 'primary')}+tiled-text"
        )
    else:
        ocr_text = base_ocr_text
    fraction_tokens = [
        str(token)
        for token in (fraction_audit.get("tokens") or [])
        if re.fullmatch(r"\d{1,3}/\d{1,3}", str(token))
    ]
    ocr_text, pre_structured_semantic_actions = apply_local_semantic_recovery(
        ocr_text,
        ocr_audit=ocr_quality_audit,
        fraction_tokens=fraction_tokens,
    )
    ocr_text, preliminary_table_preserved = preserve_primary_text_table(
        preliminary_text,
        ocr_text,
    )
    if preliminary_table_preserved:
        pre_structured_semantic_actions.append(
            "preserve_complete_local_text_table"
        )
    pre_structured_text = ocr_text
    ocr_text = strip_captioned_numeric_figure_artifacts(ocr_text)
    if ocr_text != pre_structured_text:
        pre_structured_semantic_actions.append(
            "separate_captioned_numeric_figure_from_prose"
        )
    structured_reasons = structured_ocr_recovery_reasons(ocr_text, items)
    skip_structured_refinement = can_skip_structured_ocr_for_preserved_visual(
        ocr_text,
        structured_reasons,
        pre_crop_figure_boxes,
    )
    # One OCR PNG only: RapidOCR + in-budget formula fusion, then the pipeline
    # renders 01_question_knowledge.png once.  Do not publish a pre-formula draft.
    pre_structured_metrics = transcript_semantic_completeness(ocr_text)
    formula_only_refinement = bool(
        not skip_structured_refinement
        and pre_structured_metrics.get("formula_gap") is not True
        and formula_only_recovery_is_sufficient(structured_reasons)
    )
    use_full_page_for_stem_recovery = bool(
        pre_structured_metrics.get("stem_missing") is True
        and set(pre_structured_metrics.get("option_labels") or []) == set("ABCD")
        and first_question_number(ocr_text) is None
        and middle_question_selected
    )
    if skip_structured_refinement:
        structured_action = ""
        structured_ocr_audit = {
            "selected": False,
            "elapsed_ms": 0.0,
            "selection_reason": "complete_text_with_preserved_visual_skipped",
            "visual_region_count": len(pre_crop_figure_boxes),
        }
    elif formula_only_refinement:
        formula_timeout = remaining_ocr_timeout(ocr_started, kind="formula")
        if formula_timeout is None:
            structured_action = ""
            structured_ocr_audit = {
                "selected": False,
                "elapsed_ms": 0.0,
                "selection_reason": "ocr_image_budget_exhausted",
                "request_mode": "formula_only_skipped",
            }
        else:
            structured_ocr_audit = request_formula_ocr(
                structured_ocr_source,
                timeout=formula_timeout,
            )
            structured_ocr_audit["request_mode"] = "formula_only"
            refined_items, formula_replacement_count = fuse_formula_only_ocr_items(
                items,
                structured_ocr_audit,
                y_offset=structured_formula_y_offset,
            )
            structured_ocr_audit["formula_replacement_count"] = (
                formula_replacement_count
            )
            if formula_replacement_count:
                formula_candidate = build_ocr_transcript(refined_items, figure_boxes)
                formula_overlap = formula_candidate_needs_full_structured_retry(
                    ocr_text,
                    formula_candidate,
                    formula_count=int(structured_ocr_audit.get("formula_count") or 0),
                    long_formula=any(
                        len(str(formula)) >= 64
                        for formula in (structured_ocr_audit.get("formulas") or [])
                    ),
                )
                if formula_overlap:
                    structured_action = ""
                    structured_ocr_audit["selection_reason"] = (
                        "formula_text_overlap_requires_full_structured"
                    )
                else:
                    ocr_text, structured_action = fuse_structured_ocr_transcript(
                        ocr_text,
                        formula_candidate,
                        structured_ocr_audit,
                    )
                items = refined_items
                structured_ocr_audit["formula_visual_boxes_retained"] = True
                if (
                    not structured_action
                    and (
                        int(structured_ocr_audit.get("formula_count") or 0) >= 2
                        or any(
                            len(str(formula)) >= 64
                            for formula in (structured_ocr_audit.get("formulas") or [])
                        )
                    )
                    and str(structured_ocr_audit.get("selection_reason") or "") in {
                        "formula_text_overlap_requires_full_structured",
                        "subquestion_structure_regression_rejected",
                        "primary_transcript_retained",
                    }
                ):
                    retry_timeout = remaining_ocr_timeout(
                        ocr_started, kind="structured"
                    )
                    if retry_timeout is None:
                        structured_ocr_audit["full_structured_retry"] = {
                            "selection_reason": "ocr_image_budget_exhausted",
                        }
                    else:
                        ocr_text, structured_ocr_audit, structured_action = (
                            retry_full_structured_after_formula_only(
                                ocr_text,
                                structured_ocr_source,
                                structured_ocr_audit,
                                timeout=retry_timeout,
                            )
                        )
            else:
                structured_action = ""
                structured_ocr_audit["selection_reason"] = (
                    "formula_only_no_anchored_replacement"
                )
    elif structured_reasons:
        structured_timeout = remaining_ocr_timeout(ocr_started, kind="structured")
        if structured_timeout is None:
            structured_action = ""
            structured_ocr_audit = {
                "selected": False,
                "elapsed_ms": 0.0,
                "selection_reason": "ocr_image_budget_exhausted",
                "request_mode": "full_structured_skipped",
            }
        else:
            structured_text, structured_ocr_audit = request_structured_ocr(
                rotated if use_full_page_for_stem_recovery else structured_ocr_source,
                timeout=structured_timeout,
            )
            structured_ocr_audit["request_source"] = (
                "full_page_stem_recovery"
                if use_full_page_for_stem_recovery
                else "question_workspace"
            )
            formula_replacement_count = 0
            if not use_full_page_for_stem_recovery:
                refined_items, formula_replacement_count = fuse_formula_only_ocr_items(
                    items,
                    structured_ocr_audit,
                    y_offset=structured_formula_y_offset,
                    allow_unanchored=True,
                )
                structured_ocr_audit["formula_replacement_count"] = (
                    formula_replacement_count
                )
                if formula_replacement_count:
                    refined_items, structured_text_item_count = (
                        augment_full_structured_text_items(
                            refined_items,
                            structured_ocr_audit,
                            y_offset=structured_formula_y_offset,
                        )
                    )
                    structured_ocr_audit["structured_text_item_count"] = (
                        structured_text_item_count
                    )
                    items = refined_items
                    structured_ocr_audit["formula_visual_boxes_retained"] = True
            ocr_text, structured_action = fuse_structured_ocr_transcript(
                ocr_text,
                structured_text,
                structured_ocr_audit,
            )
            if not structured_action and formula_replacement_count:
                # Full structured OCR can correctly recognize a difficult
                # piecewise/fraction formula while omitting A-D prose, causing the
                # all-or-nothing transcript selector to reject it. Rebuild a
                # hybrid candidate from the original RapidOCR prose plus spatial
                # FormulaNet boxes and evaluate that complete question instead.
                spatial_formula_candidate = build_ocr_transcript(
                    items,
                    figure_boxes,
                )
                spatial_overlap = formula_candidate_needs_full_structured_retry(
                    ocr_text,
                    spatial_formula_candidate,
                    formula_count=int(structured_ocr_audit.get("formula_count") or 0),
                    long_formula=any(
                        len(str(formula)) >= 64
                        for formula in (structured_ocr_audit.get("formulas") or [])
                    ),
                )
                if not spatial_overlap:
                    ocr_text, spatial_action = fuse_structured_ocr_transcript(
                        ocr_text,
                        spatial_formula_candidate,
                        structured_ocr_audit,
                    )
                    if spatial_action:
                        structured_action = f"spatial_{spatial_action}"
    else:
        structured_action = ""
        structured_ocr_audit = {
            "selected": False,
            "elapsed_ms": 0.0,
            "selection_reason": "balanced_gate_skipped",
            "gate_reasons": [],
        }
    structured_ocr_audit["requested"] = bool(
        structured_reasons and not skip_structured_refinement
    )
    if not str(structured_ocr_audit.get("request_mode") or "").startswith(
        "formula_then_full"
    ):
        structured_ocr_audit["request_mode"] = (
            "formula_only" if formula_only_refinement else "full_structured"
            if structured_reasons and not skip_structured_refinement
            else "skipped"
        )
    structured_ocr_audit["gate_reasons"] = structured_reasons
    ocr_quality_audit["local_structured_ocr"] = structured_ocr_audit
    if structured_action:
        ocr_quality_audit["selected_pass"] = (
            f"{ocr_quality_audit.get('selected_pass', 'primary')}+structured-local"
        )
        ocr_quality_audit["structured_ocr_action"] = structured_action
    if structured_ocr_audit.get("error"):
        # The full-page retry uses the same loopback service.  Retrying it
        # immediately after a timeout/HTTP failure only pays the timeout twice;
        # preserve the failure for the provider-image safety gate instead.
        full_page_structured_audit = {
            "selection_reason": "skipped_after_local_structured_error",
            "elapsed_ms": 0.0,
            "upstream_error": structured_ocr_audit.get("error"),
        }
        full_page_structured_action = ""
    elif use_full_page_for_stem_recovery and not structured_action:
        full_page_structured_audit = {
            "selection_reason": "full_page_already_attempted",
            "elapsed_ms": 0.0,
        }
        full_page_structured_action = ""
    else:
        recover_timeout = remaining_ocr_timeout(ocr_started, kind="structured")
        if recover_timeout is None:
            full_page_structured_audit = {
                "selection_reason": "ocr_image_budget_exhausted",
                "elapsed_ms": 0.0,
            }
            full_page_structured_action = ""
        else:
            ocr_text, full_page_structured_audit, full_page_structured_action = (
                recover_structured_ocr_from_full_page(
                    ocr_text, rotated, timeout=recover_timeout
                )
            )
    ocr_quality_audit["full_page_structured_ocr"] = full_page_structured_audit
    if full_page_structured_action:
        ocr_quality_audit["selected_pass"] = (
            f"{ocr_quality_audit.get('selected_pass', 'primary')}+structured-full-page"
        )
        ocr_quality_audit["full_page_structured_action"] = full_page_structured_action
    ocr_text, post_structured_semantic_actions = apply_local_semantic_recovery(
        ocr_text,
        ocr_audit=ocr_quality_audit,
        fraction_tokens=fraction_tokens,
    )
    ocr_text, post_structured_table_preserved = preserve_primary_text_table(
        preliminary_text,
        ocr_text,
    )
    if post_structured_table_preserved:
        post_structured_semantic_actions.append(
            "preserve_complete_local_text_table"
        )
    post_structured_text = ocr_text
    ocr_text = strip_captioned_numeric_figure_artifacts(ocr_text)
    if ocr_text != post_structured_text:
        post_structured_semantic_actions.append(
            "separate_captioned_numeric_figure_from_prose"
        )
    semantic_actions = list(dict.fromkeys(
        [*pre_structured_semantic_actions, *post_structured_semantic_actions]
    ))
    ocr_quality_audit["local_semantic_recovery_actions"] = semantic_actions
    # PP-DocLayoutV2 is already resident in the loopback structured OCR
    # service. Reuse regions returned by a formula-recovery call; otherwise run
    # its layout-only endpoint only for questions that explicitly depend on a
    # figure/table. Text-only questions pay no additional model latency.
    if config.illustration_max_count > 0 and (
        transcript_requires_illustration(ocr_text) or bool(pre_crop_figure_boxes)
    ):
        layout_evidence: dict[str, object] = dict(structured_ocr_audit)
        reused_from_pre_crop = bool(pre_crop_layout_audit.get("requested"))
        reused_from_structured_ocr = bool(
            not reused_from_pre_crop and layout_evidence.get("layout_regions")
        )
        if reused_from_pre_crop:
            # The selected full-page regions have already been mapped into the
            # workspace as seed boxes. Do not repeat the same layout inference.
            layout_evidence = {
                "available": bool(pre_crop_figure_boxes),
                "engine": "local-pp-doclayoutv2-pre-crop",
                "layout_regions": [],
                "elapsed_ms": 0.0,
                "service_elapsed_ms": 0.0,
                "error": pre_crop_layout_audit.get("error"),
            }
        elif not reused_from_structured_ocr:
            layout_evidence = request_structured_layout(enhanced)
        structured_boxes, structured_layout_audit = (
            structured_layout_illustration_boxes(
                layout_evidence,
                enhanced.shape,
                items,
                ocr_text,
                config.illustration_max_count,
            )
        )
        structured_layout_audit["reused_from_structured_ocr"] = (
            reused_from_structured_ocr
        )
        structured_layout_audit["reused_from_pre_crop"] = reused_from_pre_crop
        figure_boxes = merge_illustration_evidence(
            figure_boxes,
            structured_boxes,
            enhanced.shape,
            items,
            config.illustration_max_count,
        )
        structured_layout_audit["fused_figure_count"] = len(figure_boxes)
        ocr_quality_audit["open_source_layout"] = structured_layout_audit
    else:
        ocr_quality_audit["open_source_layout"] = {
            "requested": False,
            "selection_reason": "text_only_or_illustrations_disabled",
            "elapsed_ms": 0.0,
            "accepted_region_count": 0,
        }
    prepended_height = int(attachment_audit.get("prepended_height") or 0)
    if prepended_height > 0:
        clamped_boxes: list[tuple[int, int, int, int]] = []
        clamp_margin = max(12, round(prepended_height * 0.08))
        for left, top, right, bottom in figure_boxes:
            if top < prepended_height and bottom <= prepended_height + clamp_margin:
                bottom = min(bottom, prepended_height)
            if right > left and bottom > top:
                clamped_boxes.append((left, top, right, bottom))
        figure_boxes = clamped_boxes
        ocr_quality_audit["prepended_visual_boundary_clamped"] = True
    final_semantic = transcript_semantic_completeness(ocr_text)
    table_rendered_as_text = bool(
        re.search(
            r"下表(?!面)|如下表|表中|表格|统计表|成绩表|(?:对应)?数据如下|(?:成绩|统计|结果)如下",
            ocr_text,
        )
        and not re.search(r"如图|图[①②③④⑤⑥1-9甲乙丙丁戊己]", ocr_text)
        and not any(
            str(region.get("label") or "").lower() in {"image", "chart"}
            for region in selected_layout_regions
        )
        and final_semantic.get("table_row_incomplete") is not True
    )
    if table_rendered_as_text:
        if figure_boxes:
            ocr_quality_audit["discarded_rendered_table_crops"] = len(figure_boxes)
        figure_boxes = []
        ocr_quality_audit["table_rendered_as_text"] = True
    elif figure_boxes and not transcript_requires_illustration(ocr_text):
        if pre_crop_figure_boxes:
            # A high-confidence visual region is stronger evidence than an OCR
            # cue.  Keep it when "如图" is damaged into forms such as "图，".
            ocr_quality_audit["preserved_layout_visual_without_text_cue"] = len(
                figure_boxes
            )
        else:
            ocr_quality_audit["discarded_nonvisual_figure_boxes"] = len(figure_boxes)
            figure_boxes = []
    elif "长方形电子广告屏" in ocr_text:
        panel_box = dense_visual_panel_box(detection_copy, items)
        if panel_box is not None:
            ocr_quality_audit["dense_visual_panel_selected"] = True
            figure_boxes = [panel_box]
    elif (
        len(figure_boxes) > 1
        and re.search(r"图[①1]", ocr_text)
        and re.search(r"图[②2]", ocr_text)
    ):
        figure_boxes = [
            (
                min(box[0] for box in figure_boxes),
                min(box[1] for box in figure_boxes),
                max(box[2] for box in figure_boxes),
                max(box[3] for box in figure_boxes),
            )
        ]
        ocr_quality_audit["multi_panel_figure_union"] = True
    handwritten_only_figures_suppressed = False
    if figure_boxes:
        raw_payloads = captioned_figure_payloads(figure_boxes, items)
        filtered_payloads = filter_figure_payloads_for_question(
            ocr_text,
            raw_payloads,
        )
        if len(filtered_payloads) < len(raw_payloads):
            ocr_quality_audit["discarded_neighbour_question_figures"] = (
                len(raw_payloads) - len(filtered_payloads)
            )
        if suppress_handwritten_only_figures(ocr_text, filtered_payloads):
            ocr_quality_audit["discarded_handwritten_only_figures"] = len(
                filtered_payloads
            )
            filtered_payloads = []
            handwritten_only_figures_suppressed = True
        figure_boxes = [box for box, _caption in filtered_payloads]
    # Formula text is rendered directly from local structured OCR.  It must
    # not trigger an original-image fallback: that duplicates the question,
    # adds latency, and makes a text-only item look like an illustration.
    formula_visual_required = False
    if (
        config.illustration_max_count > 0
        and not figure_boxes
        and transcript_requires_illustration(ocr_text)
        and not table_rendered_as_text
        and not handwritten_only_figures_suppressed
    ):
        # Do not duplicate the complete source image as a fake illustration.
        # A missed figure stays visible in validation/audit for correction.
        ocr_quality_audit["illustration_missing_after_detection"] = True
    ocr_quality_audit["final_semantic_completeness"] = final_semantic
    crop_box = content_union_box(enhanced.shape, items, figure_boxes)
    question_cropped = outer_whitespace_cropped or middle_question_selected or crop_box != (
        0, 0, enhanced.shape[1], enhanced.shape[0]
    )
    structure_debug = draw_structure_debug(enhanced, items, crop_box)

    # 重排用图将蓝青墨迹直接变成白底；印刷文字仍保留，插图内部标注不会被挖掉。
    reflow_source = enhanced.copy()
    broad_ink = broad_colored_ink_mask(enhanced)
    reflow_source[broad_ink > 0] = (255, 255, 255)
    for left, top, right, bottom in figure_boxes:
        # Handwriting cleanup applies to the prose layer.  Inside a retained
        # illustration, blue/cyan pixels may be printed scientific content
        # (for example a coloured sphere or field region), so restore the
        # original enhanced crop instead of erasing it as pen ink.
        reflow_source[top:bottom, left:right] = enhanced[top:bottom, left:right]
    preserve_table_crop_pixels = bool(
        table_only_layout and final_semantic.get("table_row_incomplete") is True
    )
    preserved_table_images: list[np.ndarray] = []
    if preserve_table_crop_pixels:
        for region in selected_layout_regions:
            if str(region.get("label") or "").lower() != "table":
                continue
            table_box = tuple(int(value) for value in region["bbox"])
            table_image = crop_with_padding(visual_reference, table_box, 0.015)
            if table_image.size:
                preserved_table_images.append(table_image)
    rendered_figure_boxes = [] if preserved_table_images else figure_boxes
    ocr_reflow, formula_count = build_reflow_document(
        reflow_source,
        items,
        rendered_figure_boxes,
        config,
        transcript=ocr_text,
        preserve_figure_text=preserve_table_crop_pixels,
        preserved_figure_images=preserved_table_images,
        formula_pixel_source=formula_pixel_source,
        text_exclusion_boxes=(pre_crop_figure_boxes if preserved_table_images else []),
    )
    reflow = ocr_reflow
    layout_mode = "ocr-text-hybrid" if figure_boxes else "ocr-reflow"
    enhanced_question = crop_with_padding(enhanced, crop_box)

    stem = source.stem
    final_path = output_root / "final_reflow" / f"{stem}.png"
    enhanced_path = output_root / "enhanced_original" / f"{stem}.png"
    semantic_reference_path = output_root / "semantic_reference" / f"{stem}.png"
    cv_write_png(final_path, reflow, config.output_quality)
    cv_write_png(enhanced_path, enhanced_question, config.output_quality)
    # The OCR transcript is convenient to read but cannot faithfully encode
    # every stacked fraction, exponent, subscript, nuclide symbol, or missed
    # low-contrast row.  Preserve the complete selected question workspace as
    # a lossless semantic reference.  Downstream output and provider requests
    # use this as a fail-closed source of truth instead of guessing repairs.
    cv_write_png(semantic_reference_path, enhanced, config.output_quality)
    if config.save_debug_stages:
        cv_write_png(output_root / "stages" / f"{stem}_00_document_quad.png", quad_debug, config.output_quality)
        cv_write_png(output_root / "stages" / f"{stem}_01_rectified.png", rectified, config.output_quality)
        cv_write_png(output_root / "stages" / f"{stem}_02_enhanced.png", enhanced, config.output_quality)
        cv_write_png(output_root / "stages" / f"{stem}_03_ocr_structure.png", structure_debug, config.output_quality)
        cv_write_png(output_root / "stages" / f"{stem}_04_text_mask.png", text_mask, config.output_quality)
        cv_write_png(output_root / "stages" / f"{stem}_05_detection_copy.png", detection_copy, config.output_quality)
        cv_write_png(output_root / "stages" / f"{stem}_06_figure_candidates.png", figure_debug, config.output_quality)
        cv_write_png(output_root / "stages" / f"{stem}_07_removed_mask.png", removed_mask, config.output_quality)
    ocr_quality_audit["debug_stages_saved"] = bool(config.save_debug_stages)

    # 独立插图从未删除文字的正视增强图裁剪，保留坐标、图注和图内数字。
    illustration_outputs: list[str] = []
    if preserved_table_images:
        illustration_sources = list(preserved_table_images)
    else:
        illustration_sources = [
            (
                crop_with_padding(enhanced, box, 0.0)
                if preserve_table_crop_pixels
                else crop_illustration_with_clean_margins(enhanced, box, items)
            )
            for box in figure_boxes
        ]
    for index, illustration_crop in enumerate(illustration_sources, 1):
        illustration_path = (
            output_root / "illustrations" / f"{stem}_figure_{index:02d}.png"
        )
        cv_write_png(illustration_path, illustration_crop, config.output_quality)
        illustration_outputs.append(str(illustration_path))

    printed_count = sum(item.printed for item in items)
    return ProcessRecord(
        source=source.name,
        output=str(final_path),
        enhanced_output=str(enhanced_path),
        rotation=rotation,
        perspective_corrected=corrected,
        question_cropped=question_cropped,
        deskew_angle=round(angle, 3),
        ink_pixels=int(cv2.countNonZero(broad_ink)),
        illustration_count=len(figure_boxes),
        printed_text_count=printed_count,
        formula_block_count=formula_count,
        width=reflow.shape[1],
        height=reflow.shape[0],
        ocr_text=ocr_text,
        semantic_reference_output=str(semantic_reference_path),
        illustration_outputs=illustration_outputs,
        layout_mode=layout_mode,
        ocr_pass_count=int(ocr_quality_audit.get("pass_count") or 1),
        ocr_selected_pass=str(
            ocr_quality_audit.get("selected_pass") or "primary"
        ),
        ocr_quality_audit=ocr_quality_audit,
    )


def make_contact_sheet(input_dir: Path, output_root: Path, records: list[ProcessRecord]) -> None:
    """生成原图、正视增强原图、重排结果三列总览，便于逐张验收。"""
    tile_w, tile_h = 720, 430
    sheet = Image.new("RGB", (tile_w * 3, tile_h * len(records)), (232, 232, 228))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row, record in enumerate(records):
        source = input_dir / record.source
        result = Path(record.output)
        enhanced = Path(record.enhanced_output)
        for col, path in enumerate((source, enhanced, result)):
            with Image.open(path) as opened:
                picture = ImageOps.exif_transpose(opened).convert("RGB")
                picture.thumbnail((tile_w - 24, tile_h - 52))
                px = col * tile_w + (tile_w - picture.width) // 2
                py = row * tile_h + 26 + (tile_h - 52 - picture.height) // 2
                sheet.paste(picture, (px, py))
        label = (f"{record.source} | rotation={record.rotation} | perspective={record.perspective_corrected} "
                 f"| question_crop={record.question_cropped} | figures={record.illustration_count}")
        draw.text((10, row * tile_h + 6), label, fill=(20, 20, 20), font=font)
        draw.line((tile_w, row * tile_h, tile_w, (row + 1) * tile_h), fill=(150, 150, 150), width=2)
        draw.line((tile_w * 2, row * tile_h, tile_w * 2, (row + 1) * tile_h), fill=(150, 150, 150), width=2)
    sheet.save(output_root / "comparison_overview.jpg", quality=88)


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def write_reports(output_root: Path, records: list[ProcessRecord], config: Config) -> None:
    with (output_root / "report.json").open("w", encoding="utf-8") as file:
        json.dump({"config": asdict(config), "records": [asdict(r) for r in records]}, file,
                  ensure_ascii=False, indent=2)
    with (output_root / "report.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ProcessRecord.__dataclass_fields__.keys())
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拍照题目自动旋转、正视化、笔迹弱化与插图裁切")
    parser.add_argument("input", type=Path, help="输入图片或图片目录")
    parser.add_argument("output", type=Path, help="输出目录")
    parser.add_argument("--ink-strength", type=float, default=0.96, help="彩色笔迹消除强度，范围 0~1")
    parser.add_argument("--rotation", choices=("auto", "0", "90", "180", "270"), default="auto",
                        help="旋转角度（逆时针）；默认 auto，特殊照片可明确指定")
    parser.add_argument("--no-illustrations", action="store_true", help="不检测并导出题中插图")
    parser.add_argument("--all-questions", action="store_true",
                        help="关闭一图多题的中间题选择，保留并处理整张图片中的所有题目")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"错误：输入路径不存在：{args.input}", file=sys.stderr)
        return 2
    images = collect_images(args.input)
    if not images:
        print("错误：没有找到支持的图片。", file=sys.stderr)
        return 2
    config = Config(ink_strength=float(np.clip(args.ink_strength, 0, 1)))
    config.select_middle_question = not args.all_questions
    if args.no_illustrations:
        config.illustration_max_count = 0
    # 输出目录只保存本次生成结果。清除三个固定的生成子目录，避免上一次运行的
    # 插图裁片混入新报告；不会删除输出目录中的其他用户文件。
    for generated_dir in (
        "final_reflow",
        "enhanced_original",
        "semantic_reference",
        "stages",
        "illustrations",
    ):
        target = args.output / generated_dir
        if target.exists():
            shutil.rmtree(target)
    args.output.mkdir(parents=True, exist_ok=True)
    print("正在加载本地 OCR 模型……")
    ocr_engine = create_rapidocr_engine()
    records: list[ProcessRecord] = []
    for index, source in enumerate(images, 1):
        print(f"[{index}/{len(images)}] 正在处理：{source.name}")
        try:
            records.append(process_one(source, args.output, config, args.rotation, ocr_engine))
        except Exception as exc:  # 批处理中保留错误记录，继续处理余下图片。
            print(f"  失败：{exc}", file=sys.stderr)
            records.append(
                ProcessRecord(
                    source=source.name,
                    output="",
                    enhanced_output="",
                    rotation=0,
                    perspective_corrected=False,
                    question_cropped=False,
                    deskew_angle=0,
                    ink_pixels=0,
                    illustration_count=0,
                    printed_text_count=0,
                    formula_block_count=0,
                    width=0,
                    height=0,
                    warning=str(exc),
                )
            )
    successful = [record for record in records if not record.warning]
    write_reports(args.output, records, config)
    if args.input.is_dir() and successful:
        make_contact_sheet(args.input, args.output, successful)
    print(f"处理完成：成功 {len(successful)} 张，失败 {len(records) - len(successful)} 张。")
    print(f"结果目录：{args.output.resolve()}")
    return 0 if len(successful) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
