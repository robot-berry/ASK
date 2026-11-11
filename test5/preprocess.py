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
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rapidocr import RapidOCR


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Config:
    """集中保存可调参数，便于按不同相机和纸张继续调优。"""

    max_work_side: int = 1800          # 检测阶段最大边，限制内存和耗时
    document_min_area: float = 0.22    # 题面至少占原图比例，过小通常是误检
    document_margin: float = 0.025     # 透视四角向外扩张比例，优先保全边缘文字
    ink_saturation: int = 72           # 彩色笔迹最低饱和度
    ink_max_component_ratio: float = 0.018  # 大色块多为题图，不当作笔迹删除
    ink_strength: float = 0.96         # 0=不处理，1=尽可能消除彩色笔迹
    illustration_padding: float = 0.10 # 插图裁切外扩，保留图注、坐标和边缘
    illustration_max_count: int = 3
    ocr_printed_min_score: float = 0.58  # 印刷文字最低置信度；还会结合行结构二次判断
    reflow_canvas_width: int = 1800      # 重排白色画布宽度
    reflow_font_size: int = 42           # 重排文字基础字号
    output_quality: int = 3            # PNG 压缩等级，0 快、9 小
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
    illustration_outputs: list[str] | None = None
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


def run_ocr(image: np.ndarray, engine: RapidOCR, max_side: int = 1800) -> tuple[
        np.ndarray, tuple[str, ...], tuple[float, ...], int]:
    """执行一次 OCR，同时返回检测框、文字、分数和整页 180 度方向判断。"""
    work, scale = resize_for_detection(image, max_side)
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
    return items


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
        if cw < work.shape[1] * 0.075 or ch < work.shape[0] * 0.070:
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
        px = max(12, int(bw * config.illustration_padding))
        py = max(12, int(bh * config.illustration_padding))
        candidate = (max(0, x1 - px), max(0, y1 - py),
                     min(w, x2 + px), min(h, y2 + py))
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
        dense_handwriting = (dark_fraction > 0.033 and colorful_fraction < 0.08 and
                             gray_entropy < 3.22)
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
    selected.sort(key=lambda box: (box[1], box[0]))

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
    selected = post_refined

    debug = detection_copy.copy()
    for index, (x1, y1, x2, y2) in enumerate(selected, 1):
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 255), 5)
        cv2.putText(debug, f"FIG {index}", (x1, max(28, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 255), 2, cv2.LINE_AA)
    return selected, debug


def content_union_box(shape: tuple[int, ...], items: list[OCRItem],
                      figure_boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """联合印刷文字和插图得到题目范围；使用 12% 安全边距保证完整性。"""
    h, w = shape[:2]
    points: list[tuple[float, float, float, float]] = []
    for item in items:
        if item.printed:
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


def group_items_into_lines(items: list[OCRItem]) -> list[list[OCRItem]]:
    """按原图纵坐标和字号将 OCR 文字恢复成阅读行。"""
    printed = sorted((item for item in items if item.printed), key=lambda item: (item.center_y, item.center_x))
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


def normalize_reflow_text(tokens: list[str]) -> str:
    """合并同一 OCR 行的文字，并清理中文排版中不需要的空格。"""
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", " ".join(token.strip() for token in tokens if token.strip()))
    text = re.sub(r"\s+", " ", text).strip()
    # 中文字符之间、中文标点前以及左括号后不保留 OCR 人工空格。
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。；：！？、,.!?;:）)\]])", r"\1", text)
    text = re.sub(r"([（(\[])\s+", r"\1", text)
    text = text.replace("．", ".").replace("：", "：")
    return text


def build_ocr_transcript(items: list[OCRItem]) -> str:
    """Return reading-order OCR text using only trusted printed/formula rows."""

    selected = [item for item in items if item.printed or item.formula_like]
    grouped = group_items_into_lines(selected)
    line_heights = [max((item.height for item in group), default=0.0) for group in grouped]
    median_height = float(np.median([height for height in line_heights if height > 0])) if any(
        height > 0 for height in line_heights
    ) else 30.0
    instruction_centers = [
        float(np.mean([item.center_y for item in group]))
        for group in grouped
        if re.search(r"计算|解方程(?:组)?|化简|求值|列式|求解|证明", normalize_reflow_text([item.text for item in group]))
    ]
    lines: list[str] = []
    for group in grouped:
        text = normalize_reflow_text([item.text for item in group])
        if not text:
            continue
        style = classify_reflow_text(text)
        line_center = float(np.mean([item.center_y for item in group]))
        near_formula_instruction = any(
            abs(line_center - center) <= max(48.0, median_height * 1.9)
            for center in instruction_centers
        )
        trusted_formula = line_requires_original_pixels(
            group,
            near_formula_instruction=near_formula_instruction,
        )
        if is_suspicious_reflow_text(text, style) and not trusted_formula:
            continue
        lines.append(text)
    return "\n".join(lines)


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
    # A Chinese instruction plus formula is reliable. A pure printed formula
    # is retained only when it sits next to a trusted instruction such as
    # “计算” or “解方程组”; an isolated handwritten result cannot qualify.
    return has_operator and has_math_atom and (has_cjk or near_formula_instruction)


def classify_reflow_text(text: str) -> str:
    """按题号、题干小问和选项划分排版层级。"""
    import re

    compact = text.lstrip()
    if re.match(r"^\d{1,3}\s*[.、]", compact) or re.match(r"^\d{1,3}\s*[（(].{0,5}分", compact):
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
        numbered = re.match(r"^\s*(\d{1,3})\s*(?:[.．。、:：]|(?=[（(]\s*\d+\s*分))", text)
        named = re.match(r"^\s*第\s*(\d{1,3})\s*题", text)
        match = named or numbered
        if match and len(re.sub(r"\s+", "", text)) >= 5:
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
        bottom_padding = max(8, int(median_height * 0.30)) if end_y < h else 0
        selected_top = max(0, start_y - top_padding)
        selected_bottom = min(h, end_y - bottom_padding)
        if selected_bottom - selected_top >= max(80, median_height * 2):
            return 0, selected_top, w, selected_bottom
        return full_page

    # 已找到一个明确主问题号时按单题处理，不使用段落备用方案，避免拆散其小问。
    if main_markers:
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


def crop_question_workspace(image: np.ndarray, items: list[OCRItem],
                            question_box: tuple[int, int, int, int]) -> tuple[np.ndarray, list[OCRItem]]:
    """把选中的题转换成独立工作区，并同步平移 OCR 坐标。

    插图检测中的面积比例、形态学尺寸和大图收紧规则都应以一道题而不是整张多题
    图片为基准。先裁工作区还能让后续步骤完全看不到相邻题，避免跨题合并配图。
    """
    x1, y1, x2, y2 = question_box
    workspace = image[y1:y2, x1:x2].copy()
    shifted_items: list[OCRItem] = []
    for item in items_inside_question(items, question_box):
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


def build_reflow_document(enhanced: np.ndarray, items: list[OCRItem],
                          figure_boxes: list[tuple[int, int, int, int]], config: Config) -> tuple[np.ndarray, int]:
    """在白色画布上按原阅读顺序重排印刷文字，并插入原正视图裁出的插图。

    高置信度普通正文重新渲染；低置信度或数学符号较多的行以原图块保留，避免
    OCR 错误破坏公式。落入插图框内部的文字不重复排版，因为插图裁片已包含标注。
    """
    canvas_w = config.reflow_canvas_width
    margin = 112
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
            r"计算|解方程(?:组)?|化简|求值|列式|求解|证明",
            normalize_reflow_text([item.text for item in line]),
        )
    ]
    objects: list[tuple[float, str, object]] = []
    text_rows: list[dict[str, object]] = []
    formula_count = 0
    for line in line_groups:
        # 题图内部的尺寸、字母和坐标已经完整保留在裁图中，不再重复排入正文。
        # 按单个 OCR 项判断而不是按整行中心判断，可处理“正文和题图同一横行”的情况。
        line = [
            item
            for item in line
            if not any(
                x1 <= item.center_x <= x2 and y1 <= item.center_y <= y2
                for x1, y1, x2, y2 in figure_boxes
            )
        ]
        if not line:
            continue
        y = float(np.median([item.center_y for item in line]))
        line_box = (int(min(item.box[:, 0].min() for item in line)),
                    int(min(item.box[:, 1].min() for item in line)),
                    int(max(item.box[:, 0].max() for item in line)),
                    int(max(item.box[:, 1].max() for item in line)))
        center = ((line_box[0] + line_box[2]) / 2, (line_box[1] + line_box[3]) / 2)
        if any(x1 <= center[0] <= x2 and y1 <= center[1] <= y2 for x1, y1, x2, y2 in figure_boxes):
            continue
        near_formula_instruction = any(
            abs(y - instruction_y) <= 1.9 * median_printed_height
            for instruction_y in formula_instruction_centers
        )
        formula_line = line_requires_original_pixels(
            line,
            near_formula_instruction=near_formula_instruction,
        )
        if formula_line:
            objects.append((y, "formula", line_box))
            formula_count += 1
        else:
            text = normalize_reflow_text([item.text for item in line])
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
        objects.append((float(block["start_y"]), str(block["style"]), str(block["text"])))
    for box in figure_boxes:
        objects.append(((box[1] + box[3]) / 2, "figure", box))
    # 排版时先按原始纵坐标排列正文和公式，再把所有题图统一放到文档末尾。
    # 这样题图不会插入题干或小问之间；存在多张题图时，图与图之间仍保持原顺序。
    objects.sort(key=lambda item: (item[1] == "figure", item[0]))

    # 先估算高度，再创建一次性画布，避免反复扩容损失清晰度。
    estimated = 220
    probe = Image.new("RGB", (canvas_w, 200), "white")
    probe_draw = ImageDraw.Draw(probe)
    for _, kind, payload in objects:
        if kind in style_fonts:
            current_font = style_fonts[kind]
            indent = 38 if kind == "option" else 18 if kind == "subquestion" else 0
            line_height = int(config.reflow_font_size * 1.58)
            paragraph_space = 36 if kind == "question" else 28 if kind == "subquestion" else 22
            estimated += (len(wrap_text(probe_draw, str(payload), current_font,
                                        usable_w - indent)) * line_height + paragraph_space)
        else:
            x1, y1, x2, y2 = payload  # type: ignore[misc]
            max_width = usable_w * 0.92
            max_height = 860 if kind == "figure" else 240
            # 小尺寸题图在整页中占比过低时，大模型容易忽略细线和尺寸标注。
            # 允许温和插值放大到 2.6 倍，但仍受画布宽度和高度双重限制。
            scale_cap = 2.6 if kind == "figure" else 1.0
            scale = min(scale_cap, max_width / max(1, x2 - x1), max_height / max(1, y2 - y1))
            estimated += int((y2 - y1) * scale) + 46
    pil_canvas = Image.new("RGB", (canvas_w, max(700, estimated)), "white")
    draw = ImageDraw.Draw(pil_canvas)
    y_cursor = 80
    for _, kind, payload in objects:
        if kind in style_fonts:
            current_font = style_fonts[kind]
            line_height = int(config.reflow_font_size * 1.58)
            indent = 38 if kind == "option" else 18 if kind == "subquestion" else 0
            if kind == "question":
                y_cursor += 10
                draw.rounded_rectangle((margin - 28, y_cursor + 6,
                                        margin - 16, y_cursor + line_height - 10),
                                       radius=4, fill=(38, 54, 62))
            for line_text in wrap_text(draw, str(payload), current_font, usable_w - indent):
                draw.text((margin + indent, y_cursor), line_text,
                          fill=(17, 20, 22), font=current_font)
                y_cursor += line_height
            y_cursor += 32 if kind == "question" else 25 if kind == "subquestion" else 20
            continue
        box = payload  # type: ignore[assignment]
        snippet = crop_with_padding(enhanced, box, 0.04 if kind == "formula" else 0.0)
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
            y_cursor += pil_snippet.height + 46
    final = np.asarray(pil_canvas.crop((0, 0, canvas_w, min(pil_canvas.height, y_cursor + 55))))
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


def process_one(source: Path, output_root: Path, config: Config, forced_rotation: str,
                ocr_engine: RapidOCR) -> ProcessRecord:
    """按“正视化 -> 一次 OCR -> 去正文检测图形 -> 白底重排”处理单张图片。"""
    image = cv_read(source)
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
    rectified, angle = deskew(rectified)
    enhanced = normalize_illumination(rectified)

    # 正视化后只执行一次 OCR；后续全部使用同一组坐标，减少耗时和坐标漂移。
    ocr_boxes, ocr_texts, ocr_scores, ocr_rotation = run_ocr(enhanced, ocr_engine)
    if ocr_rotation == 180:
        # 分类器已经在识别前转正各文字小图，因此文本结果可以直接复用；
        # 这里只旋转整页并映射现有四边形坐标，不再调用 OCR。
        old_h, old_w = enhanced.shape[:2]
        enhanced = cv2.rotate(enhanced, cv2.ROTATE_180)
        rectified = cv2.rotate(rectified, cv2.ROTATE_180)
        quad_debug = cv2.rotate(quad_debug, cv2.ROTATE_180)
        if len(ocr_boxes):
            mapped_boxes = ocr_boxes.copy()
            mapped_boxes[:, :, 0] = old_w - 1 - mapped_boxes[:, :, 0]
            mapped_boxes[:, :, 1] = old_h - 1 - mapped_boxes[:, :, 1]
            ocr_boxes = np.asarray([order_quad(box) for box in mapped_boxes], np.float32)
        rotation = (rotation + 180) % 360
    all_items = analyze_ocr_items(ocr_boxes, ocr_texts, ocr_scores, config)
    question_box = select_middle_question_box(
        enhanced.shape, all_items, config.select_middle_question
    )
    full_question_box = (0, 0, enhanced.shape[1], enhanced.shape[0])
    middle_question_selected = question_box != full_question_box
    enhanced, items = crop_question_workspace(enhanced, all_items, question_box)
    ocr_text = build_ocr_transcript(items)

    # 从这里开始，所有阈值和坐标都只针对选中的一道题；相邻题不再参与
    # 文字遮罩、插图检测、正文重排或增强原图输出。
    text_mask = create_text_mask(enhanced.shape, items)
    detection_copy, removed_mask = make_detection_copy(enhanced, text_mask)
    figure_boxes, figure_debug = detect_illustration_boxes(detection_copy, enhanced, items, config)
    crop_box = content_union_box(enhanced.shape, items, figure_boxes)
    question_cropped = middle_question_selected or crop_box != (
        0, 0, enhanced.shape[1], enhanced.shape[0]
    )
    structure_debug = draw_structure_debug(enhanced, items, crop_box)

    # 重排用图将蓝青墨迹直接变成白底；印刷文字仍保留，插图内部标注不会被挖掉。
    reflow_source = enhanced.copy()
    broad_ink = broad_colored_ink_mask(enhanced)
    reflow_source[broad_ink > 0] = (255, 255, 255)
    reflow, formula_count = build_reflow_document(reflow_source, items, figure_boxes, config)
    enhanced_question = crop_with_padding(enhanced, crop_box)

    stem = source.stem
    final_path = output_root / "final_reflow" / f"{stem}.png"
    enhanced_path = output_root / "enhanced_original" / f"{stem}.png"
    cv_write_png(final_path, reflow, config.output_quality)
    cv_write_png(enhanced_path, enhanced_question, config.output_quality)
    cv_write_png(output_root / "stages" / f"{stem}_00_document_quad.png", quad_debug, config.output_quality)
    cv_write_png(output_root / "stages" / f"{stem}_01_rectified.png", rectified, config.output_quality)
    cv_write_png(output_root / "stages" / f"{stem}_02_enhanced.png", enhanced, config.output_quality)
    cv_write_png(output_root / "stages" / f"{stem}_03_ocr_structure.png", structure_debug, config.output_quality)
    cv_write_png(output_root / "stages" / f"{stem}_04_text_mask.png", text_mask, config.output_quality)
    cv_write_png(output_root / "stages" / f"{stem}_05_detection_copy.png", detection_copy, config.output_quality)
    cv_write_png(output_root / "stages" / f"{stem}_06_figure_candidates.png", figure_debug, config.output_quality)
    cv_write_png(output_root / "stages" / f"{stem}_07_removed_mask.png", removed_mask, config.output_quality)

    # 独立插图从未删除文字的正视增强图裁剪，保留坐标、图注和图内数字。
    illustration_outputs: list[str] = []
    for index, box in enumerate(figure_boxes, 1):
        illustration_path = (
            output_root / "illustrations" / f"{stem}_figure_{index:02d}.png"
        )
        cv_write_png(
            illustration_path,
            crop_with_padding(enhanced, box),
            config.output_quality,
        )
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
        illustration_outputs=illustration_outputs,
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
    for generated_dir in ("final_reflow", "enhanced_original", "stages", "illustrations"):
        target = args.output / generated_dir
        if target.exists():
            shutil.rmtree(target)
    args.output.mkdir(parents=True, exist_ok=True)
    print("正在加载本地 OCR 模型……")
    ocr_engine = RapidOCR()
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
