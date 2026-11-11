from __future__ import annotations

from typing import Any, Sequence


FORMULA_LABELS = frozenset({"inline_formula", "display_formula"})
WHOLE_IMAGE_FORMULA_MIN_AREA_RATIO = 0.40


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


def formula_mfr_candidates(
    layout: Sequence[dict[str, Any]],
    image_width: int,
    image_height: int,
    *,
    whole_image_formula: bool,
) -> tuple[list[dict[str, Any]], str]:
    """Choose MFR boxes. Formula crops may send whole_image_formula=True."""
    layout_candidates = [
        dict(item)
        for item in layout
        if str(item.get("label") or "") in FORMULA_LABELS
        and _safe_corner_bbox(item.get("bbox")) is not None
    ]
    if not whole_image_formula:
        return layout_candidates, "layout"
    page_area = max(1, image_width * image_height)
    best_ratio = 0.0
    for item in layout_candidates:
        bbox = _safe_corner_bbox(item.get("bbox"))
        if bbox is None:
            continue
        left, top, right, bottom = bbox
        best_ratio = max(best_ratio, (right - left) * (bottom - top) / page_area)
    if layout_candidates and best_ratio >= WHOLE_IMAGE_FORMULA_MIN_AREA_RATIO:
        return layout_candidates, "layout"
    return (
        [
            {
                "label": "display_formula",
                "bbox": [0, 0, max(1, image_width), max(1, image_height)],
                "score": 1.0,
            }
        ],
        "whole_image",
    )
