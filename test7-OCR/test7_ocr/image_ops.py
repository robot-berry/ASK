from __future__ import annotations

import hashlib
import math
from pathlib import Path

from PIL import Image, ImageFile, ImageOps

from .types import BBox


RISK_PROBE_MAX_SIDE = 1280


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_source_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
            return image
    except OSError:
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(path) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.info["truncated_decoded"] = True
                return image
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous


def content_bbox(image: Image.Image, threshold: int = 248, padding: int = 16) -> BBox:
    grayscale = image.convert("L")
    mask = grayscale.point(lambda value: 255 if value < threshold else 0)
    detected = mask.getbbox()
    if detected is None:
        return BBox(0, 0, image.width, image.height)
    left, top, right, bottom = detected
    return BBox(
        max(0, left - padding),
        max(0, top - padding),
        min(image.width, right + padding) - max(0, left - padding),
        min(image.height, bottom + padding) - max(0, top - padding),
    )


def direct_crop(image: Image.Image, bbox: BBox) -> Image.Image | None:
    safe = bbox.clamp(image.width, image.height)
    if safe.width <= 0 or safe.height <= 0:
        return None
    return image.crop((safe.x, safe.y, safe.right, safe.bottom))


def require_direct_crop(image: Image.Image, bbox: BBox) -> Image.Image:
    crop = direct_crop(image, bbox)
    if crop is None:
        raise ValueError(f"Empty crop after clamping: {bbox}")
    return crop


def photographed_page_risk(image: Image.Image, bbox: BBox) -> tuple[bool, dict[str, float]]:
    """Conservative guard: photo-like paper is treated as handwriting-risk, not as proof."""
    try:
        import numpy as np
    except ImportError:
        return False, {"available": 0.0}
    crop = direct_crop(image, bbox)
    if crop is None:
        return False, {"empty_crop": 1.0, "probe_downscaled": 0.0}
    original_width, original_height = crop.size
    long_edge = max(original_width, original_height)
    probe_downscaled = 0.0
    if long_edge > RISK_PROBE_MAX_SIDE:
        scale = RISK_PROBE_MAX_SIDE / long_edge
        crop = crop.resize(
            (max(1, int(original_width * scale)), max(1, int(original_height * scale))),
            Image.Resampling.BILINEAR,
        )
        probe_downscaled = 1.0
    pixels = np.asarray(crop, dtype=np.int16)
    channel_spread = pixels.max(axis=2) - pixels.min(axis=2)
    gray = pixels.mean(axis=2)
    colorful_ratio = float(np.mean((channel_spread > 18) & (gray < 248)))
    midtone_ratio = float(np.mean((gray >= 105) & (gray < 232)))
    paper_pixels = gray[gray > 105]
    paper_std = float(paper_pixels.std()) if paper_pixels.size else 0.0
    risk = colorful_ratio >= 0.20 or midtone_ratio >= 0.30
    return risk, {
        "colorful_ratio": round(colorful_ratio, 6),
        "midtone_ratio": round(midtone_ratio, 6),
        "paper_std": round(paper_std, 6),
        "probe_downscaled": probe_downscaled,
        "original_width": float(original_width),
        "original_height": float(original_height),
    }


def bbox_from_polygon(polygon: object) -> BBox | None:
    if polygon is None or isinstance(polygon, (str, bytes)):
        return None
    try:
        points = list(polygon)  # type: ignore[arg-type]
    except TypeError:
        return None
    if not points:
        return None
    try:
        xs = [float(point[0]) for point in points]  # type: ignore[index]
        ys = [float(point[1]) for point in points]  # type: ignore[index]
    except (IndexError, TypeError, ValueError):
        return None
    if not xs or not ys:
        return None
    left = math.floor(min(xs))
    top = math.floor(min(ys))
    right = math.ceil(max(xs))
    bottom = math.ceil(max(ys))
    return BBox(left, top, max(0, right - left), max(0, bottom - top))
