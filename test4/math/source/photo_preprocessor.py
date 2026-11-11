#!/usr/bin/env python3
"""
Photo preprocessing for photographed mathematics problems.

功能：
1. EXIF 方向纠正；
2. 自动检测纸张/页面四边形；
3. 透视矫正，将斜拍页面尽量转换为正视图；
4. 检测图片清晰度、亮度和对比度；
5. 光照 / 阴影归一化；
6. CLAHE 局部对比度增强；
7. 轻度锐化文字、公式和数学图形线条；
8. 支持单张图片测试；
9. 支持文件夹批量测试；
10. 输出 audit JSON 和 summary.csv。

注意：
- 最终图片仍然是灰度图，不做强制二值化；
- 透视检测失败时会保持原图，不强行变换；
- 原始图片不会被覆盖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import argparse
import csv
import json

import cv2
import numpy as np
from PIL import Image, ImageOps


# ============================================================
# 配置参数
# ============================================================

# ------------------------------------------------------------
# 图片质量检测
# ------------------------------------------------------------

# Laplacian 方差低于这个值时提示可能模糊
BLUR_WARNING_THRESHOLD = 80.0

# 平均亮度范围
BRIGHTNESS_DARK_THRESHOLD = 70.0
BRIGHTNESS_BRIGHT_THRESHOLD = 245.0

# 灰度标准差低于该值认为整体对比度偏低
CONTRAST_WARNING_THRESHOLD = 25.0


# ------------------------------------------------------------
# 光照归一化
# ------------------------------------------------------------

# 必须是奇数
BACKGROUND_BLUR_KERNEL = 51


# ------------------------------------------------------------
# CLAHE
# ------------------------------------------------------------

CLAHE_CLIP_LIMIT = 1.2
CLAHE_TILE_GRID_SIZE = (8, 8)


# ------------------------------------------------------------
# 透视检测安全参数
# ------------------------------------------------------------

# 候选四边形至少占整张图片多少面积
PAGE_MIN_AREA_RATIO = 0.20

# 最大候选轮廓数量
PAGE_MAX_CONTOURS = 30

# 四边形拟合参数
PAGE_APPROX_EPSILON = 0.02

# 输出页面最小尺寸
PAGE_MIN_WIDTH = 150
PAGE_MIN_HEIGHT = 150

# 纸张长宽比保护
# 太离谱就认为检测错了
PAGE_MIN_ASPECT_RATIO = 0.30
PAGE_MAX_ASPECT_RATIO = 3.50


# ============================================================
# PIL / OpenCV 转换
# ============================================================

def pil_to_cv(image: Image.Image) -> np.ndarray:
    """
    PIL RGB -> OpenCV BGR
    """

    rgb = np.asarray(image.convert("RGB"))

    return cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2BGR,
    )


def cv_to_pil(image: np.ndarray) -> Image.Image:
    """
    OpenCV BGR / Gray -> PIL
    """

    if image.ndim == 2:
        return Image.fromarray(image)

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    return Image.fromarray(rgb)


# ============================================================
# 图片质量检测
# ============================================================

def calculate_blur_score(
    gray: np.ndarray,
) -> float:
    """
    使用 Laplacian 方差估计清晰度。

    一般：
        数值越高 -> 越清晰
        数值越低 -> 越模糊

    这里只作为测试指标，不建议把阈值理解为绝对标准。
    """

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    return float(laplacian.var())


def calculate_brightness(
    gray: np.ndarray,
) -> float:
    """
    平均亮度。
    """

    return float(np.mean(gray))


def calculate_contrast(
    gray: np.ndarray,
) -> float:
    """
    使用灰度标准差描述全局对比度。
    """

    return float(np.std(gray))


def evaluate_image_quality(
    gray: np.ndarray,
) -> dict[str, Any]:
    """
    图片基础质量检测。
    """

    blur_score = calculate_blur_score(gray)
    brightness = calculate_brightness(gray)
    contrast = calculate_contrast(gray)

    warnings: list[str] = []

    if blur_score < BLUR_WARNING_THRESHOLD:
        warnings.append("image_may_be_blurry")

    if brightness < BRIGHTNESS_DARK_THRESHOLD:
        warnings.append("image_may_be_too_dark")

    if brightness > BRIGHTNESS_BRIGHT_THRESHOLD:
        warnings.append("image_may_be_overexposed")

    if contrast < CONTRAST_WARNING_THRESHOLD:
        warnings.append("image_may_have_low_contrast")

    return {
        "blur_score": round(blur_score, 3),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "warnings": warnings,
        "quality_ok": len(warnings) == 0,
    }


# ============================================================
# 透视矫正
# ============================================================

def order_points(
    points: np.ndarray,
) -> np.ndarray:
    """
    四角排序：

    0：左上
    1：右上
    2：右下
    3：左下
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    coordinate_sum = points.sum(axis=1)

    ordered[0] = points[
        np.argmin(coordinate_sum)
    ]

    ordered[2] = points[
        np.argmax(coordinate_sum)
    ]

    coordinate_diff = np.diff(
        points,
        axis=1,
    ).reshape(-1)

    ordered[1] = points[
        np.argmin(coordinate_diff)
    ]

    ordered[3] = points[
        np.argmax(coordinate_diff)
    ]

    return ordered


def angle_between(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    """
    计算 ABC 的夹角，单位：度。
    """

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator <= 1e-8:
        return 0.0

    cosine = float(
        np.dot(ba, bc)
        / denominator
    )

    cosine = float(
        np.clip(
            cosine,
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def quadrilateral_is_reasonable(
    rect: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[bool, dict[str, Any]]:
    """
    判断检测到的四边形是否像一个正常页面。

    目的是：
    宁可不做透视，也不要错误拉伸数学题。
    """

    tl, tr, br, bl = rect

    width_top = float(
        np.linalg.norm(tr - tl)
    )

    width_bottom = float(
        np.linalg.norm(br - bl)
    )

    height_left = float(
        np.linalg.norm(bl - tl)
    )

    height_right = float(
        np.linalg.norm(br - tr)
    )

    target_width = max(
        width_top,
        width_bottom,
    )

    target_height = max(
        height_left,
        height_right,
    )

    if target_width <= 0 or target_height <= 0:
        return False, {
            "reason": "invalid_width_or_height",
        }

    aspect_ratio = (
        target_width
        / target_height
    )

    angles = [
        angle_between(bl, tl, tr),
        angle_between(tl, tr, br),
        angle_between(tr, br, bl),
        angle_between(br, bl, tl),
    ]

    # 页面四角不要求严格 90°
    # 因为透视拍摄本来就会改变角度。
    # 这里只过滤极端奇怪的轮廓。
    angle_ok = all(
        25.0 <= angle <= 155.0
        for angle in angles
    )

    aspect_ok = (
        PAGE_MIN_ASPECT_RATIO
        <= aspect_ratio
        <= PAGE_MAX_ASPECT_RATIO
    )

    size_ok = (
        target_width >= PAGE_MIN_WIDTH
        and target_height >= PAGE_MIN_HEIGHT
    )

    return (
        angle_ok
        and aspect_ok
        and size_ok
    ), {
        "width_top": round(width_top, 2),
        "width_bottom": round(width_bottom, 2),
        "height_left": round(height_left, 2),
        "height_right": round(height_right, 2),
        "aspect_ratio": round(
            aspect_ratio,
            4,
        ),
        "corner_angles_deg": [
            round(value, 2)
            for value in angles
        ],
        "angle_ok": angle_ok,
        "aspect_ok": aspect_ok,
        "size_ok": size_ok,
        "image_size_px": [
            image_width,
            image_height,
        ],
    }


def correct_perspective(
    image: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    自动检测页面四边形并进行透视矫正。

    如果没有找到足够可靠的页面，
    直接返回原图。
    """

    original = image.copy()

    height, width = image.shape[:2]

    image_area = (
        width * height
    )

    # --------------------------------------------------------
    # 灰度
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # --------------------------------------------------------
    # 轻度模糊
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # --------------------------------------------------------
    # Canny 边缘
    # --------------------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150,
    )

    # --------------------------------------------------------
    # 闭运算
    #
    # 比单纯膨胀稍微稳一点，
    # 用来连接断裂的纸张边缘。
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )

    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    # --------------------------------------------------------
    # 找轮廓
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    rejected_candidates: list[dict[str, Any]] = []

    page_contour: np.ndarray | None = None
    page_rect: np.ndarray | None = None
    selected_area_ratio = 0.0
    selected_geometry: dict[str, Any] = {}

    # --------------------------------------------------------
    # 从较大的轮廓开始寻找页面
    # --------------------------------------------------------

    for contour in contours[
        :PAGE_MAX_CONTOURS
    ]:

        area = cv2.contourArea(
            contour
        )

        area_ratio = (
            area
            / image_area
        )

        if area_ratio < PAGE_MIN_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            PAGE_APPROX_EPSILON
            * perimeter,
            True,
        )

        if len(approx) != 4:
            continue

        candidate = approx.reshape(
            4,
            2,
        )

        rect = order_points(
            candidate
        )

        reasonable, geometry = (
            quadrilateral_is_reasonable(
                rect,
                width,
                height,
            )
        )

        if not reasonable:
            rejected_candidates.append({
                "area_ratio": round(
                    area_ratio,
                    5,
                ),
                **geometry,
            })
            continue

        page_contour = candidate
        page_rect = rect
        selected_area_ratio = area_ratio
        selected_geometry = geometry

        break

    # --------------------------------------------------------
    # 没找到可靠页面
    # --------------------------------------------------------

    if (
        page_contour is None
        or page_rect is None
    ):
        return original, {
            "perspective_corrected": False,
            "reason": (
                "no_reliable_page_quadrilateral"
            ),
            "candidate_contour_count": len(
                contours
            ),
            "rejected_candidates": (
                rejected_candidates[:5]
            ),
        }

    # --------------------------------------------------------
    # 计算目标尺寸
    # --------------------------------------------------------

    tl, tr, br, bl = page_rect

    width_top = np.linalg.norm(
        tr - tl
    )

    width_bottom = np.linalg.norm(
        br - bl
    )

    max_width = int(
        round(
            max(
                width_top,
                width_bottom,
            )
        )
    )

    height_left = np.linalg.norm(
        bl - tl
    )

    height_right = np.linalg.norm(
        br - tr
    )

    max_height = int(
        round(
            max(
                height_left,
                height_right,
            )
        )
    )

    if (
        max_width < PAGE_MIN_WIDTH
        or max_height < PAGE_MIN_HEIGHT
    ):
        return original, {
            "perspective_corrected": False,
            "reason": (
                "detected_page_output_too_small"
            ),
            "output_candidate_size_px": [
                max_width,
                max_height,
            ],
        }

    # --------------------------------------------------------
    # 正视图四角
    # --------------------------------------------------------

    destination = np.array(
        [
            [0, 0],
            [
                max_width - 1,
                0,
            ],
            [
                max_width - 1,
                max_height - 1,
            ],
            [
                0,
                max_height - 1,
            ],
        ],
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # 透视变换
    # --------------------------------------------------------

    matrix = cv2.getPerspectiveTransform(
        page_rect,
        destination,
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            max_width,
            max_height,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(
            255,
            255,
            255,
        ),
    )

    audit = {
        "perspective_corrected": True,
        "reason": (
            "reliable_page_quadrilateral_detected"
        ),
        "source_corners_px": (
            page_rect
            .round(2)
            .tolist()
        ),
        "page_area_ratio": round(
            selected_area_ratio,
            5,
        ),
        "output_size_px": [
            max_width,
            max_height,
        ],
        "geometry": selected_geometry,
    }

    return warped, audit


# ============================================================
# 光照 / 阴影校正
# ============================================================

def normalize_illumination(
    gray: np.ndarray,
) -> np.ndarray:
    """
    校正纸面上的阴影和不均匀照明。

    不进行二值化。
    """

    kernel = int(
        BACKGROUND_BLUR_KERNEL
    )

    if kernel % 2 == 0:
        kernel += 1

    # 对非常小的图片防止 kernel 超尺寸
    max_kernel = min(
        gray.shape[0],
        gray.shape[1],
    )

    if max_kernel % 2 == 0:
        max_kernel -= 1

    kernel = min(
        kernel,
        max_kernel,
    )

    kernel = max(
        kernel,
        3,
    )

    background = cv2.GaussianBlur(
        gray,
        (
            kernel,
            kernel,
        ),
        sigmaX=0,
    )

    background = np.maximum(
        background,
        1,
    )

    normalized = cv2.divide(
        gray,
        background,
        scale=255,
    )

    return normalized


# ============================================================
# 局部对比度增强
# ============================================================

def enhance_contrast(
    gray: np.ndarray,
) -> np.ndarray:
    """
    CLAHE 局部对比度增强。
    """

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=(
            CLAHE_TILE_GRID_SIZE
        ),
    )

    return clahe.apply(
        gray
    )


# ============================================================
# 文字 / 数学线条锐化
# ============================================================

def sharpen_document(
    gray: np.ndarray,
) -> np.ndarray:
    """
    温和 Unsharp Mask。

    主要增强：
    - 中文文字；
    - 数字；
    - 数学公式；
    - 几何线；
    - 坐标轴；
    - 表格线。

    不使用强锐化，避免产生明显黑边和噪声。
    """

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=1.0,
    )

    sharpened = cv2.addWeighted(
        gray,
        1.55,
        blurred,
        -0.55,
        0,
    )

    return np.clip(
        sharpened,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# 主预处理函数
# ============================================================

def preprocess_photo(
    image: str | Path | Image.Image,
    *,
    perspective: bool = True,
    normalize_light: bool = True,
    enhance: bool = True,
) -> tuple[
    Image.Image,
    dict[str, Any],
]:
    """
    处理一张手机拍摄的数学题照片。

    流程：

    原始照片
        ↓
    EXIF 修正
        ↓
    透视矫正
        ↓
    灰度
        ↓
    原始质量检测
        ↓
    去阴影
        ↓
    CLAHE
        ↓
    轻度锐化
        ↓
    处理后质量检测
        ↓
    PIL 灰度图
    """

    # --------------------------------------------------------
    # 1. 读取图片
    # --------------------------------------------------------

    if isinstance(
        image,
        Image.Image,
    ):
        source = image.copy()
        source_name = "<PIL.Image>"

    else:
        image_path = Path(
            image
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"图片不存在：{image_path}"
            )

        with Image.open(
            image_path
        ) as opened:
            opened.load()
            source = opened.copy()

        source_name = str(
            image_path.resolve()
        )

    original_size = source.size

    # --------------------------------------------------------
    # 2. EXIF 方向纠正
    # --------------------------------------------------------

    source = ImageOps.exif_transpose(
        source
    )

    source = source.convert(
        "RGB"
    )

    exif_corrected_size = (
        source.size
    )

    # --------------------------------------------------------
    # 3. PIL -> OpenCV
    # --------------------------------------------------------

    cv_image = pil_to_cv(
        source
    )

    # --------------------------------------------------------
    # 4. 透视矫正
    # --------------------------------------------------------

    perspective_audit: dict[
        str,
        Any,
    ] = {
        "perspective_corrected": False,
        "reason": "disabled",
    }

    if perspective:
        (
            cv_image,
            perspective_audit,
        ) = correct_perspective(
            cv_image
        )

    perspective_size = [
        int(cv_image.shape[1]),
        int(cv_image.shape[0]),
    ]

    # --------------------------------------------------------
    # 5. 转灰度
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        cv_image,
        cv2.COLOR_BGR2GRAY,
    )

    # --------------------------------------------------------
    # 6. 增强前质量检测
    # --------------------------------------------------------

    original_quality = (
        evaluate_image_quality(
            gray
        )
    )

    processed = gray.copy()

    # --------------------------------------------------------
    # 7. 去阴影 / 光照归一化
    # --------------------------------------------------------

    if normalize_light:
        processed = (
            normalize_illumination(
                processed
            )
        )

    # --------------------------------------------------------
    # 8. CLAHE
    # --------------------------------------------------------

    #if enhance:
    #    processed = (
    #        enhance_contrast(
    #            processed
    #        )
    #    )

    # --------------------------------------------------------
    # 9. 轻度锐化
    # --------------------------------------------------------

    if enhance:
        processed = (
            sharpen_document(
                processed
            )
        )

    # --------------------------------------------------------
    # 10. 处理后质量检测
    # --------------------------------------------------------

    processed_quality = (
        evaluate_image_quality(
            processed
        )
    )

    # --------------------------------------------------------
    # 11. 转回 PIL
    # --------------------------------------------------------

    result = Image.fromarray(
        processed
    )

    # --------------------------------------------------------
    # 12. Audit
    # --------------------------------------------------------

    audit = {
        "source": source_name,

        "original_size_px": [
            original_size[0],
            original_size[1],
        ],

        "exif_corrected_size_px": [
            exif_corrected_size[0],
            exif_corrected_size[1],
        ],

        "perspective_output_size_px":
            perspective_size,

        "output_size_px": [
            result.width,
            result.height,
        ],

        "original_quality":
            original_quality,

        "processed_quality":
            processed_quality,

        "perspective":
            perspective_audit,

        "operations": {
            "exif_transpose": True,

            "perspective_correction":
                perspective,

            "grayscale": True,

            "illumination_normalization":
                normalize_light,

            "clahe_contrast_enhancement":
                enhance,

            "sharpening":
                enhance,

            "binarization":
                False,

            "final_crop":
                bool(
                    perspective_audit.get(
                        "perspective_corrected"
                    )
                ),
        },

        "parameters": {
            "background_blur_kernel":
                (
                    BACKGROUND_BLUR_KERNEL
                    if normalize_light
                    else None
                ),

            "clahe_clip_limit":
                (
                    CLAHE_CLIP_LIMIT
                    if enhance
                    else None
                ),

            "clahe_tile_grid_size":
                (
                    list(
                        CLAHE_TILE_GRID_SIZE
                    )
                    if enhance
                    else None
                ),

            "blur_warning_threshold":
                BLUR_WARNING_THRESHOLD,

            "page_min_area_ratio":
                PAGE_MIN_AREA_RATIO,

            "page_approx_epsilon":
                PAGE_APPROX_EPSILON,
        },
    }

    return result, audit


# ============================================================
# 单张图片处理
# ============================================================

def process_one_image(
    image_path: Path,
    *,
    image_output_dir: Path,
    audit_output_dir: Path,
    perspective: bool,
    normalize_light: bool,
    enhance: bool,
) -> dict[str, Any]:
    """
    处理一张图片，并保存：

    xxx_preprocessed.png
    xxx_audit.json
    """

    try:
        (
            processed_image,
            audit,
        ) = preprocess_photo(
            image_path,
            perspective=perspective,
            normalize_light=normalize_light,
            enhance=enhance,
        )

        input_stem = (
            image_path.stem
        )

        image_output_path = (
            image_output_dir
            / (
                f"{input_stem}"
                "_preprocessed.png"
            )
        )

        audit_output_path = (
            audit_output_dir
            / (
                f"{input_stem}"
                "_audit.json"
            )
        )

        processed_image.save(
            image_output_path,
            format="PNG",
        )

        with audit_output_path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                audit,
                handle,
                ensure_ascii=False,
                indent=2,
            )

        original_quality = (
            audit[
                "original_quality"
            ]
        )

        processed_quality = (
            audit[
                "processed_quality"
            ]
        )

        perspective_info = (
            audit[
                "perspective"
            ]
        )

        return {
            "filename":
                image_path.name,

            "source_path":
                str(
                    image_path.resolve()
                ),

            "perspective_corrected":
                perspective_info.get(
                    "perspective_corrected",
                    False,
                ),

            "perspective_reason":
                perspective_info.get(
                    "reason",
                    "",
                ),

            "page_area_ratio":
                perspective_info.get(
                    "page_area_ratio",
                    "",
                ),

            "original_blur_score":
                original_quality[
                    "blur_score"
                ],

            "original_brightness":
                original_quality[
                    "brightness"
                ],

            "original_contrast":
                original_quality[
                    "contrast"
                ],

            "original_quality_ok":
                original_quality[
                    "quality_ok"
                ],

            "original_warnings":
                ";".join(
                    original_quality[
                        "warnings"
                    ]
                ),

            "processed_blur_score":
                processed_quality[
                    "blur_score"
                ],

            "processed_brightness":
                processed_quality[
                    "brightness"
                ],

            "processed_contrast":
                processed_quality[
                    "contrast"
                ],

            "processed_quality_ok":
                processed_quality[
                    "quality_ok"
                ],

            "processed_warnings":
                ";".join(
                    processed_quality[
                        "warnings"
                    ]
                ),

            "status":
                "success",

            "error":
                "",
        }

    except Exception as exc:

        return {
            "filename":
                image_path.name,

            "source_path":
                str(
                    image_path.resolve()
                ),

            "perspective_corrected":
                "",

            "perspective_reason":
                "",

            "page_area_ratio":
                "",

            "original_blur_score":
                "",

            "original_brightness":
                "",

            "original_contrast":
                "",

            "original_quality_ok":
                "",

            "original_warnings":
                "",

            "processed_blur_score":
                "",

            "processed_brightness":
                "",

            "processed_contrast":
                "",

            "processed_quality_ok":
                "",

            "processed_warnings":
                "",

            "status":
                "failed",

            "error":
                str(exc),
        }


# ============================================================
# 命令行
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "数学题手机照片预处理"
        )
    )

    # --------------------------------------------------------
    # 单张图片
    # --------------------------------------------------------

    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="单张输入图片路径",
    )

    # --------------------------------------------------------
    # 批量目录
    # --------------------------------------------------------

    parser.add_argument(
        "--batch",
        type=Path,
        help="批量处理图片文件夹",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归处理子文件夹",
    )

    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__)
            .resolve()
            .parent
            / "test_output"
        ),
        help=(
            "输出目录，"
            "默认保存在脚本旁边的 "
            "test_output"
        ),
    )

    # --------------------------------------------------------
    # 功能开关
    # --------------------------------------------------------

    parser.add_argument(
        "--no-perspective",
        action="store_true",
        help="关闭透视矫正",
    )

    parser.add_argument(
        "--no-light-normalization",
        action="store_true",
        help="关闭光照/阴影归一化",
    )

    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="关闭 CLAHE 和锐化",
    )

    args = parser.parse_args()

    # ========================================================
    # 输出目录
    # ========================================================

    output_root = (
        args.output_dir
    )

    image_output_dir = (
        output_root
        / "preprocessed"
    )

    audit_output_dir = (
        output_root
        / "audit"
    )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    perspective = (
        not args.no_perspective
    )

    normalize_light = (
        not args.no_light_normalization
    )

    enhance = (
        not args.no_enhance
    )

    # ========================================================
    # 批量模式
    # ========================================================

    if args.batch is not None:

        batch_dir = args.batch

        if not batch_dir.is_dir():
            raise NotADirectoryError(
                "批量输入目录不存在："
                f"{batch_dir}"
            )

        if args.recursive:
            candidates = (
                batch_dir.rglob("*")
            )
        else:
            candidates = (
                batch_dir.glob("*")
            )

        image_paths = sorted(
            path
            for path in candidates
            if (
                path.is_file()
                and path.suffix.lower()
                in supported_extensions
            )
        )

        if not image_paths:
            raise ValueError(
                "文件夹中没有找到"
                f"支持的图片：{batch_dir}"
            )

        print()
        print("=" * 65)
        print("数学题照片批量预处理")
        print("=" * 65)

        print(
            f"输入目录："
            f"{batch_dir.resolve()}"
        )

        print(
            f"图片数量："
            f"{len(image_paths)}"
        )

        print(
            f"透视矫正："
            f"{perspective}"
        )

        print(
            f"光照校正："
            f"{normalize_light}"
        )

        print(
            f"图像增强："
            f"{enhance}"
        )

        print(
            f"输出目录："
            f"{output_root.resolve()}"
        )

        print()

        summary_rows: list[
            dict[str, Any]
        ] = []

        for index, image_path in enumerate(
            image_paths,
            start=1,
        ):

            print(
                f"[{index}/"
                f"{len(image_paths)}] "
                f"{image_path.name}",
                end=" ",
            )

            row = process_one_image(
                image_path,
                image_output_dir=(
                    image_output_dir
                ),
                audit_output_dir=(
                    audit_output_dir
                ),
                perspective=perspective,
                normalize_light=(
                    normalize_light
                ),
                enhance=enhance,
            )

            summary_rows.append(
                row
            )

            if row["status"] == "success":

                perspective_text = (
                    "已拉正"
                    if row[
                        "perspective_corrected"
                    ]
                    else "未拉正"
                )

                print(
                    f"✓ {perspective_text}"
                )

            else:
                print(
                    "✗ "
                    f"{row['error']}"
                )

        # ====================================================
        # summary.csv
        # ====================================================

        summary_path = (
            output_root
            / "summary.csv"
        )

        fieldnames = [
            "filename",
            "source_path",

            "perspective_corrected",
            "perspective_reason",
            "page_area_ratio",

            "original_blur_score",
            "original_brightness",
            "original_contrast",
            "original_quality_ok",
            "original_warnings",

            "processed_blur_score",
            "processed_brightness",
            "processed_contrast",
            "processed_quality_ok",
            "processed_warnings",

            "status",
            "error",
        ]

        with summary_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                summary_rows
            )

        success_count = sum(
            row["status"]
            == "success"
            for row
            in summary_rows
        )

        failed_count = (
            len(summary_rows)
            - success_count
        )

        perspective_count = sum(
            row.get(
                "perspective_corrected"
            ) is True
            for row
            in summary_rows
        )

        print()
        print("=" * 65)
        print("批量处理完成")
        print("=" * 65)

        print(
            f"总图片数："
            f"{len(summary_rows)}"
        )

        print(
            f"成功："
            f"{success_count}"
        )

        print(
            f"失败："
            f"{failed_count}"
        )

        print(
            f"成功透视矫正："
            f"{perspective_count}"
        )

        print(
            "处理后图片："
            f"{image_output_dir.resolve()}"
        )

        print(
            "Audit："
            f"{audit_output_dir.resolve()}"
        )

        print(
            "汇总："
            f"{summary_path.resolve()}"
        )

        return

    # ========================================================
    # 单张模式
    # ========================================================

    if args.input is not None:

        image_path = (
            args.input
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"图片不存在："
                f"{image_path}"
            )

        print()
        print(
            "开始处理："
            f"{image_path.resolve()}"
        )

        result = process_one_image(
            image_path,
            image_output_dir=(
                image_output_dir
            ),
            audit_output_dir=(
                audit_output_dir
            ),
            perspective=perspective,
            normalize_light=(
                normalize_light
            ),
            enhance=enhance,
        )

        print()
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

        print()

        print(
            "处理后图片保存在："
        )

        print(
            image_output_dir.resolve()
        )

        return

    # ========================================================
    # 没有输入
    # ========================================================

    parser.error(
        "请输入单张图片路径，"
        "或者使用 --batch 指定图片文件夹"
    )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()