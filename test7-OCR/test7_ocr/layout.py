from __future__ import annotations

import re
from typing import Any

from PIL import Image

from .types import BBox, RecognizedBlock


_NUMBERED_FIGURE_CAPTION_RE = re.compile(
    r"第\s*(\d{1,3})(?:\s*[-—－]\s*[A-Za-zⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)?\s*题\s*图"
)


def _merge_boxes(boxes: list[BBox], gap: int, width: int, height: int) -> list[BBox]:
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        output: list[BBox] = []
        while merged:
            current = merged.pop(0)
            joined = False
            for index, other in enumerate(merged):
                horizontally_close = current.x <= other.right + gap and other.x <= current.right + gap
                vertically_close = current.y <= other.bottom + gap and other.y <= current.bottom + gap
                if horizontally_close and vertically_close:
                    left = min(current.x, other.x)
                    top = min(current.y, other.y)
                    right = max(current.right, other.right)
                    bottom = max(current.bottom, other.bottom)
                    merged[index] = BBox(left, top, right - left, bottom - top)
                    changed = True
                    joined = True
                    break
            if not joined:
                output.append(current)
        merged = output + merged
    return [box.clamp(width, height) for box in merged]


def detect_visual_bboxes(
    image: Image.Image,
    text_blocks: list[RecognizedBlock],
    config: dict[str, Any],
    *,
    question_number: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> list[BBox]:
    """Generate conservative, unverified figure candidates; returned pixels are never redrawn."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    image_area = image.width * image.height
    grayscale = np.asarray(image.convert("L"))

    def candidates_from_ink(
        ink: Any,
        *,
        minimum_width: int,
        minimum_height: int,
        maximum_area_ratio: float,
        maximum_aspect_ratio: float,
        maximum_density: float,
        candidate_padding: int,
    ) -> list[BBox]:
        residual = ink.copy()
        for block in text_blocks:
            if block.bbox is None:
                continue
            bbox = block.bbox.clamp(image.width, image.height)
            padding_x, padding_y = 6, 5
            cv2.rectangle(
                residual,
                (max(0, bbox.x - padding_x), max(0, bbox.y - padding_y)),
                (
                    min(image.width - 1, bbox.right + padding_x),
                    min(image.height - 1, bbox.bottom + padding_y),
                ),
                0,
                thickness=-1,
            )
        residual = cv2.morphologyEx(
            residual,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
            iterations=1,
        )
        residual = cv2.dilate(
            residual,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            residual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates: list[BBox] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            box_area = width * height
            if width < minimum_width or height < minimum_height:
                continue
            if box_area < image_area * 0.004 or box_area > image_area * maximum_area_ratio:
                continue
            if (
                width / max(1, height) > maximum_aspect_ratio
                or height / max(1, width) > maximum_aspect_ratio
            ):
                continue
            raw_density = float(
                np.count_nonzero(ink[y : y + height, x : x + width])
            ) / max(1, box_area)
            if not 0.008 <= raw_density <= maximum_density:
                continue
            padding = candidate_padding
            candidates.append(
                BBox(
                    x - padding,
                    y - padding,
                    width + padding * 2,
                    height + padding * 2,
                ).clamp(image.width, image.height)
            )
        merged = _merge_boxes(candidates, gap=24, width=image.width, height=image.height)
        merged = [
            box
            for box in merged
            if box.width * box.height <= image_area * (maximum_area_ratio + 0.04)
            and box.width >= minimum_width
            and box.height >= minimum_height
        ]
        merged.sort(key=lambda box: (box.y, box.x))
        return merged

    fixed_ink = cv2.threshold(grayscale, 225, 255, cv2.THRESH_BINARY_INV)[1]
    fixed = candidates_from_ink(
        fixed_ink,
        minimum_width=65,
        minimum_height=55,
        maximum_area_ratio=0.48,
        maximum_aspect_ratio=7.0,
        maximum_density=0.55,
        candidate_padding=12,
    )
    if fixed:
        if diagnostics is not None:
            diagnostics.update(
                {"method": "fixed_global_threshold", "candidate_count": len(fixed)}
            )
        return fixed[:3]

    visual_config = config.get("visual_assets", {})
    if not visual_config.get("adaptive_caption_fallback_enabled", True):
        return []
    number_match = re.match(r"\s*(\d{1,3})", question_number or "")
    if number_match is None:
        return []
    target_number = int(number_match.group(1))
    captions: list[tuple[int, BBox]] = []
    for block in text_blocks:
        if block.bbox is None or block.kind != "text":
            continue
        match = _NUMBERED_FIGURE_CAPTION_RE.search(block.text or "")
        if match:
            captions.append((int(match.group(1)), block.bbox))
    target_captions = [bbox for number, bbox in captions if number == target_number]
    if not target_captions:
        return []

    block_size = int(visual_config.get("adaptive_caption_block_size", 81))
    block_size = max(15, block_size + (1 - block_size % 2))
    adaptive_ink = cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        float(visual_config.get("adaptive_caption_c", 18)),
    )
    adaptive = candidates_from_ink(
        adaptive_ink,
        minimum_width=90,
        minimum_height=90,
        maximum_area_ratio=0.38,
        maximum_aspect_ratio=4.0,
        maximum_density=0.35,
        candidate_padding=4,
    )
    selected: list[BBox] = []
    for caption in target_captions:
        caption_center_x = caption.x + caption.width / 2
        eligible = [
            box
            for box in adaptive
            if box.y < caption.y
            and caption.y >= box.y + box.height * 0.35
            and max(0, caption.y - box.bottom) <= 120
            and box.x - box.width * 0.25
            <= caption_center_x
            <= box.right + box.width * 0.25
        ]
        if not eligible:
            continue
        best = max(
            eligible,
            key=lambda box: (
                box.width * box.height,
                -abs((box.x + box.width / 2) - caption_center_x),
            ),
        )
        padding = 8
        left = min(best.x, caption.x) - padding
        # The adaptive contour already retains a four-pixel safety margin.  An
        # additional top expansion can pull the preceding question's option row
        # into a figure whose printed stroke starts close to the page top.
        top = best.y
        right = max(best.right, caption.right) + padding
        bottom = max(best.bottom, caption.bottom) + padding
        selected.append(
            BBox(left, top, right - left, bottom - top).clamp(
                image.width, image.height
            )
        )
    selected = _merge_boxes(selected, gap=12, width=image.width, height=image.height)
    selected.sort(key=lambda box: (box.y, box.x))
    if selected and diagnostics is not None:
        diagnostics.update(
            {
                "method": "adaptive_numbered_caption_fallback",
                "target_question_number": target_number,
                "caption_count": len(target_captions),
                "candidate_count": len(adaptive),
                "selected_count": len(selected),
            }
        )
    return selected[:3]
