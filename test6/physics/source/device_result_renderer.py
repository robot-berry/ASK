#!/usr/bin/env python3
"""Validate the six-image, five-level physics device contract."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PIL import Image

from physics_guidance_renderer import (
    DEFAULT_SONG,
    DEFAULT_SONG_BOLD,
    DEFAULT_TIMES,
    DEFAULT_TIMES_BOLD,
    DISPLAY_WIDTH_CM,
    PNG_DPI,
    PNG_DPI_QUANTIZATION_TOLERANCE,
    WIDTH,
    sha256_file,
)


# One local question page followed by five progressively more explicit levels.
DEVICE_IMAGE_NAMES = (
    "01_question_knowledge.png",
    "02_guidance_elite.png",
    "03_guidance_upper.png",
    "04_guidance_middle.png",
    "05_guidance_lower.png",
    "06_guidance_novice.png",
)
DEVICE_IMAGE_KINDS = (
    "question_knowledge",
    "elite",
    "upper",
    "middle",
    "lower",
    "novice",
)


def _first_existing(candidates: list[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"缺少{label}字体；已检查：" + "；".join(str(item) for item in candidates)
    )


def resolve_device_fonts() -> tuple[Path, Path, Path, Path]:
    """Use Song on Windows and a Song-style CJK serif face on Linux."""

    configured_cjk = os.environ.get("TEST3_CJK_FONT", "").strip()
    configured_cjk_bold = os.environ.get("TEST3_CJK_BOLD_FONT", "").strip()
    configured_latin = os.environ.get("TEST3_LATIN_FONT", "").strip()
    configured_latin_bold = os.environ.get("TEST3_LATIN_BOLD_FONT", "").strip()
    cjk = _first_existing(
        ([Path(configured_cjk)] if configured_cjk else [])
        + [
            DEFAULT_SONG,
            Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        ],
        "中文",
    )
    cjk_bold = _first_existing(
        ([Path(configured_cjk_bold)] if configured_cjk_bold else [])
        + ([Path(configured_cjk)] if configured_cjk else [])
        + [
            DEFAULT_SONG_BOLD,
            Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        ],
        "中文粗体",
    )
    latin = _first_existing(
        ([Path(configured_latin)] if configured_latin else [])
        + [
            DEFAULT_TIMES,
            Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        ],
        "拉丁",
    )
    latin_bold = _first_existing(
        ([Path(configured_latin_bold)] if configured_latin_bold else [])
        + [
            DEFAULT_TIMES_BOLD,
            Path("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
        ],
        "拉丁粗体",
    )
    return cjk, cjk_bold, latin, latin_bold


def inspect_device_png(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        image_format = image.format
        dpi = image.info.get("dpi")
    errors: list[str] = []
    if width != WIDTH:
        errors.append(f"width={width}, expected={WIDTH}")
    if height <= 0:
        errors.append("height must be positive")
    if mode != "L":
        errors.append(f"mode={mode}, expected=L")
    if image_format != "PNG":
        errors.append(f"format={image_format}, expected=PNG")
    if (
        not isinstance(dpi, tuple)
        or len(dpi) < 2
        or abs(float(dpi[0]) - PNG_DPI) > PNG_DPI_QUANTIZATION_TOLERANCE
        or abs(float(dpi[1]) - PNG_DPI) > PNG_DPI_QUANTIZATION_TOLERANCE
    ):
        errors.append("DPI does not represent the test3 9 cm display width")
    physical_width_cm = (
        width / float(dpi[0]) * 2.54
        if isinstance(dpi, tuple) and len(dpi) >= 2 and float(dpi[0]) > 0
        else None
    )
    if physical_width_cm is not None and abs(physical_width_cm - DISPLAY_WIDTH_CM) > 0.001:
        errors.append(
            f"physical_width_cm={physical_width_cm:.4f}, expected={DISPLAY_WIDTH_CM}"
        )
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
        "dpi": list(dpi) if isinstance(dpi, tuple) else None,
        "display_width_cm": DISPLAY_WIDTH_CM,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "valid": not errors,
        "errors": errors,
    }


def validate_device_image_set(output_dir: Path, *, require_all: bool = True) -> dict[str, Any]:
    expected = list(DEVICE_IMAGE_NAMES)
    observed = sorted(path.name for path in output_dir.glob("*.png") if path.is_file())
    files: list[dict[str, Any]] = []
    errors: list[str] = []
    for sequence, (filename, kind) in enumerate(
        zip(DEVICE_IMAGE_NAMES, DEVICE_IMAGE_KINDS), 1
    ):
        path = output_dir / filename
        if not path.is_file():
            if require_all:
                errors.append(f"missing {filename}")
            continue
        item = inspect_device_png(path)
        item["kind"] = kind
        item["sequence"] = sequence
        files.append(item)
        errors.extend(f"{filename}: {error}" for error in item["errors"])
    if require_all and observed != expected:
        errors.append(f"image order/set mismatch: observed={observed}")
    return {
        "valid": not errors,
        "expected": expected,
        "observed": observed,
        "image_count": len(files),
        "files": files,
        "errors": errors,
        "contract": {
            "width_px": WIDTH,
            "mode": "L",
            "format": "PNG",
            "display_width_cm": DISPLAY_WIDTH_CM,
            "height": "content_auto",
            "image_count": 6,
            "content_order": list(DEVICE_IMAGE_KINDS),
        },
    }


__all__ = [
    "DEVICE_IMAGE_KINDS",
    "DEVICE_IMAGE_NAMES",
    "inspect_device_png",
    "resolve_device_fonts",
    "validate_device_image_set",
]
