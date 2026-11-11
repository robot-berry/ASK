"""Spatial FormulaNet fusion and overlapping-text suppression.

CPU-only helpers: they never request a second OCR pass.  Flattened RapidOCR
spans are replaced by already-returned FormulaNet boxes, then duplicate
fragments that occupy the same pixels are dropped.  Callers keep the existing
formula-only path so the OCR PNG stays inside the 5-second budget.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any, Callable, Sequence

import numpy as np


MakeItem = Callable[..., Any]

_MATH_CHARS = set(
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "=<>≤≥≠≈+-−–—×÷*/√^_{}[]().,;:·"
)
_CJK_PUNCT = set("，。；：！？、")
_LATEX_SPAN = re.compile(r"\$[^$]+\$")
_LATEX_COMMAND = re.compile(
    r"\\(?:mathrm|mathbf|mathbb|text|left|right|quad|qquad|cdot|"
    r"times|ln|log|sin|cos|tan|infty|geqslant|leqslant|geq|leq)\s*"
)
_FLATTENED_INDEX = re.compile(
    r"(?<![A-Za-z\\$])(?:[A-Z]n|[A-Za-z][0-9])(?![0-9A-Za-z.])"
)


def item_bbox(item: Any) -> tuple[float, float, float, float]:
    box = np.asarray(item.box, dtype=np.float32)
    return (
        float(box[:, 0].min()),
        float(box[:, 1].min()),
        float(box[:, 0].max()),
        float(box[:, 1].max()),
    )


def bbox_from_array(box: np.ndarray) -> tuple[float, float, float, float]:
    array = np.asarray(box, dtype=np.float32)
    return (
        float(array[:, 0].min()),
        float(array[:, 1].min()),
        float(array[:, 0].max()),
        float(array[:, 1].max()),
    )


def _overlap_wh(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float]:
    horizontal = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    vertical = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return horizontal, vertical


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    horizontal, vertical = _overlap_wh(left, right)
    overlap = horizontal * vertical
    area_left = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    area_right = max(1.0, (right[2] - right[0]) * (right[3] - right[1]))
    union = area_left + area_right - overlap
    return overlap / max(1.0, union)


def bbox_overlap_ratio_min(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    horizontal, vertical = _overlap_wh(left, right)
    overlap = horizontal * vertical
    area_left = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    area_right = max(1.0, (right[2] - right[0]) * (right[3] - right[1]))
    return overlap / max(1.0, min(area_left, area_right))


def has_flattened_script_tokens(text: str) -> bool:
    """Detect RapidOCR flattening that structure gates do not see."""

    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False
    return bool(
        _FLATTENED_INDEX.search(compact)
        or re.search(r"均为1(?=[,，。;；]|$)", compact)
        or re.search(r"直线1[,，]l", compact)
        or re.search(r"(?<![A-Za-z\\$])[A-Za-z]。=(?:[A-Za-z]。=)?0", compact)
    )


def latex_alnum(latex: str) -> str:
    value = str(latex or "")
    value = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"\1\2",
        value,
    )
    value = _LATEX_COMMAND.sub("", value)
    value = re.sub(r"\\[A-Za-z]+", "", value)
    value = re.sub(r"[{}^_\\\s]", "", value)
    return value


def looks_like_flattened_formula(span: str, latex: str) -> bool:
    core = re.sub(r"\s+", "", span or "")
    if not core or not latex:
        return False
    letters = re.sub(r"[^A-Za-z0-9]", "", latex_alnum(latex)).lower()
    observed = re.sub(r"[^A-Za-z0-9]", "", core).lower()
    if not letters or not observed:
        return False
    if observed == letters:
        return True
    if r"\frac" in latex.replace(" ", ""):
        digits = re.findall(r"\d+", latex)
        joined = "".join(digits)
        if core in {joined, "/".join(digits), *digits}:
            return True
    min_contained = max(3, int(round(len(letters) * 0.4)))
    if observed in letters and len(observed) >= min_contained:
        return True
    if letters.startswith(observed) and len(observed) >= min_contained:
        return True
    return SequenceMatcher(None, letters, observed).ratio() >= 0.72 and len(observed) >= min_contained


def is_mathy_span(span: str) -> bool:
    compact = re.sub(r"\s+", "", span or "")
    if not compact:
        return False
    math = sum(char in _MATH_CHARS for char in compact)
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in compact)
    return math >= 1 and math >= cjk


def item_is_math_shaped(text: str, formula_like: bool) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in compact)
    math = sum(
        char.isdigit()
        or (char.isascii() and char.isalpha())
        or char in "=<>≤≥+-×÷*/√^{}[]()∩∪∞πμθλρΩ△Δ"
        for char in compact
    )
    return bool(
        formula_like
        or (
            cjk <= 2
            and math >= max(2, int(len(compact) * 0.45))
        )
    )


def formula_overlaps_line(
    item_box: tuple[float, float, float, float],
    formula_box: tuple[float, float, float, float],
) -> bool:
    horizontal, vertical = _overlap_wh(item_box, formula_box)
    item_h = max(1.0, item_box[3] - item_box[1])
    formula_h = max(1.0, formula_box[3] - formula_box[1])
    formula_w = max(1.0, formula_box[2] - formula_box[0])
    formula_area = formula_w * formula_h
    intersection = horizontal * vertical
    vertical_ratio = vertical / min(item_h, formula_h)
    return bool(
        horizontal > 0
        and vertical_ratio >= 0.28
        and (
            intersection / formula_area >= 0.28
            or horizontal / formula_w >= 0.45
        )
    )


def _char_widths(text: str) -> list[float]:
    return [
        2.0 if ("\u4e00" <= char <= "\u9fff" or char in _CJK_PUNCT) else 1.0
        for char in text
    ]


def split_indices_for_x(
    text: str,
    item_left: float,
    item_right: float,
    cut_left: float,
    cut_right: float,
) -> tuple[int, int]:
    if not text:
        return 0, 0
    widths = _char_widths(text)
    total = sum(widths) or 1.0
    item_width = max(1.0, item_right - item_left)
    start = 0
    end = len(text)
    consumed = 0.0
    for index, width in enumerate(widths):
        x0 = item_left + item_width * consumed / total
        consumed += width
        x1 = item_left + item_width * consumed / total
        if x1 <= cut_left + 1.0:
            start = index + 1
        if x0 < cut_right - 1.0:
            end = index + 1
    if start > end:
        start, end = end, start
    return start, end


def _is_math_char(char: str) -> bool:
    return char in _MATH_CHARS and not ("\u4e00" <= char <= "\u9fff")


def expand_math_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and _is_math_char(text[start - 1]) and text[start - 1] not in "，。；：,":
        start -= 1
    while end < len(text) and _is_math_char(text[end]) and text[end] not in "，。；：,":
        end += 1
    return start, end


def math_runs(text: str) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if _is_math_char(text[index]):
            end = index + 1
            while end < len(text):
                if _is_math_char(text[end]):
                    end += 1
                    continue
                if (
                    text[end] == " "
                    and end + 1 < len(text)
                    and _is_math_char(text[end + 1])
                ):
                    end += 1
                    continue
                break
            if any(char.isalnum() or char in "=<>≤≥≠≈+-×÷*/√^_" for char in text[index:end]):
                runs.append((index, end))
            index = end
        else:
            index += 1
    return runs


def locate_formula_span(text: str, start: int, end: int, latex: str) -> tuple[int, int] | None:
    """Map a formula box onto the RapidOCR math run it flattened, not nearby CJK."""

    runs = math_runs(text)
    if not runs:
        return None
    overlapping = [
        run for run in runs
        if run[0] < end and run[1] > start
    ]
    candidates = overlapping or runs
    matching = [
        run
        for run in candidates
        if looks_like_flattened_formula(text[run[0]:run[1]].strip(" ,，;；"), latex)
    ]
    if matching:
        return max(matching, key=lambda run: run[1] - run[0])
    mathy = [run for run in candidates if is_mathy_span(text[run[0]:run[1]])]
    if mathy:
        return max(mathy, key=lambda run: run[1] - run[0])
    trimmed_start, trimmed_end = start, end
    while trimmed_start < trimmed_end and not _is_math_char(text[trimmed_start]):
        trimmed_start += 1
    while trimmed_end > trimmed_start and not _is_math_char(text[trimmed_end - 1]):
        trimmed_end -= 1
    if trimmed_end > trimmed_start:
        return expand_math_span(text, trimmed_start, trimmed_end)
    return None


def _rect_box(left: float, top: float, right: float, bottom: float) -> np.ndarray:
    return np.asarray(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float32,
    )


def _trim_flattened_suffix(before: str, latex: str) -> str:
    match = re.search(r"([^\u4e00-\u9fff]+)$", before)
    if not match:
        return before
    span = match.group(1)
    core = span.strip(" ,，;；.。:：")
    if not core:
        return before
    if looks_like_flattened_formula(core, latex) or (
        is_mathy_span(core) and looks_like_flattened_formula(core, latex)
    ):
        kept_ws = re.match(r"\s*", span)
        prefix = kept_ws.group(0) if kept_ws else ""
        return before[: match.start()] + prefix
    return before


def _trim_flattened_prefix(after: str, latex: str) -> str:
    match = re.match(r"^([^\u4e00-\u9fff]+)", after)
    if not match:
        return after
    span = match.group(1)
    core = span.strip(" ,，;；.。:：")
    if not core:
        return after
    if looks_like_flattened_formula(core, latex) or (
        is_mathy_span(core) and looks_like_flattened_formula(core, latex)
    ):
        trailing = re.search(r"\s*$", span)
        suffix = trailing.group(0) if trailing else ""
        return suffix + after[match.end() :]
    return after


def strip_flattened_neighbors_of_latex(text: str) -> str:
    """Drop RapidOCR leftovers sitting next to the FormulaNet span."""

    if "$" not in (text or ""):
        return text
    lines: list[str] = []
    for line in str(text).splitlines():
        spans = list(_LATEX_SPAN.finditer(line))
        if not spans:
            lines.append(line)
            continue
        pieces: list[str] = []
        cursor = 0
        for match in spans:
            before = _trim_flattened_suffix(line[cursor:match.start()], match.group(0)[1:-1])
            pieces.append(before)
            pieces.append(match.group(0))
            cursor = match.end()
        after = line[cursor:]
        after = _trim_flattened_prefix(after, spans[-1].group(0)[1:-1])
        pieces.append(after)
        cleaned = re.sub(r"[ \t]{2,}", " ", "".join(pieces)).strip()
        cleaned = re.sub(r"\s+([,，;；.。])", r"\1", cleaned)
        lines.append(cleaned)
    return "\n".join(lines)


def accept_formula_boxes(
    items: Sequence[Any],
    formula_audit: dict[str, object],
    *,
    y_offset: float = 0.0,
    allow_unanchored: bool = False,
) -> list[tuple[np.ndarray, str, float]]:
    trusted_anchors = [
        item_bbox(item)
        for item in items
        if getattr(item, "printed", False) or getattr(item, "formula_like", False)
    ]
    accepted: list[tuple[np.ndarray, str, float]] = []
    for raw in formula_audit.get("formula_items") or []:
        if not isinstance(raw, dict):
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
        latex = re.sub(r"\s+", " ", str(raw.get("latex") or "")).strip()
        latex = re.sub(r"(?<=\d)\s+(?=\d)", "", latex)
        if (
            right <= left
            or bottom <= top
            or not latex
            or len(latex) > 420
            or sum("\u4e00" <= char <= "\u9fff" for char in latex) > 2
            or re.search(r"(?:\b[01]\s*){8,}", latex)
        ):
            continue
        formula_box = (left, top, right, bottom)
        formula_area = max(1.0, (right - left) * (bottom - top))
        anchored = False
        for anchor in trusted_anchors:
            horizontal, vertical = _overlap_wh(formula_box, anchor)
            intersection = horizontal * vertical
            anchor_area = max(1.0, (anchor[2] - anchor[0]) * (anchor[3] - anchor[1]))
            if intersection / formula_area >= 0.12 or intersection / anchor_area >= 0.24:
                anchored = True
                break
        if not anchored and not allow_unanchored:
            continue
        duplicate = False
        for existing_index, (existing, existing_latex, _score) in enumerate(accepted):
            existing_box = bbox_from_array(existing)
            if bbox_overlap_ratio_min(formula_box, existing_box) >= 0.85:
                if len(latex) > len(existing_latex):
                    accepted.pop(existing_index)
                else:
                    duplicate = True
                break
        if not duplicate:
            accepted.append((_rect_box(left, top, right, bottom), latex, score))
    return accepted


def _split_piece(
    item: Any,
    formula_box_array: np.ndarray,
    latex: str,
    make_item: MakeItem,
    normalize_text: Callable[[list[str]], str],
) -> list[Any]:
    text = normalize_text([str(getattr(item, "text", "") or "")])
    item_box = item_bbox(item)
    formula_box = bbox_from_array(formula_box_array)
    ix1, iy1, ix2, iy2 = item_box
    fx1, fy1, fx2, fy2 = formula_box
    item_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))
    formula_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
    horizontal, vertical = _overlap_wh(item_box, formula_box)
    intersection = horizontal * vertical
    math_shaped = item_is_math_shaped(text, bool(getattr(item, "formula_like", False)))
    if math_shaped and intersection / min(item_area, formula_area) >= 0.42:
        marker = re.match(r"^\s*((?:\d{1,3}|[A-D])[.．、:：])", text)
        if not marker:
            return []
        marker_text = marker.group(1)
        marker_width = min(
            max(12.0, (ix2 - ix1) * len(marker_text) / max(1, len(text))),
            max(12.0, (ix2 - ix1) * 0.28),
        )
        return [
            make_item(
                _rect_box(ix1, iy1, ix1 + marker_width, iy2),
                marker_text,
                float(getattr(item, "score", 0.9)),
                formula_like=False,
            )
        ]
    if not formula_overlaps_line(item_box, formula_box):
        return [item]
    mapped_start, mapped_end = split_indices_for_x(text, ix1, ix2, fx1, fx2)
    located = locate_formula_span(text, mapped_start, mapped_end, latex)
    if located is None:
        return [item]
    start, end = located
    overlapped = text[start:end]
    if not overlapped.strip():
        return [item]
    replaceable = is_mathy_span(overlapped) or looks_like_flattened_formula(
        overlapped, latex
    )
    if not replaceable:
        return [item]
    pieces: list[Any] = []
    left_text = text[:start].rstrip()
    right_text = text[end:].lstrip()
    cut_left = ix1 + (ix2 - ix1) * (start / max(1, len(text)))
    cut_right = ix1 + (ix2 - ix1) * (end / max(1, len(text)))
    if left_text:
        pieces.append(
            make_item(
                _rect_box(ix1, iy1, max(ix1 + 8.0, min(cut_left, fx1)), iy2),
                left_text,
                float(getattr(item, "score", 0.9)),
                formula_like=False,
            )
        )
    if right_text:
        pieces.append(
            make_item(
                _rect_box(min(ix2 - 8.0, max(cut_right, fx2)), iy1, ix2, iy2),
                right_text,
                float(getattr(item, "score", 0.9)),
                formula_like=False,
            )
        )
    return pieces or []


def split_items_around_formulas(
    items: Sequence[Any],
    formulas: Sequence[tuple[np.ndarray, str, float]],
    make_item: MakeItem,
    normalize_text: Callable[[list[str]], str],
) -> list[Any]:
    ordered = sorted(formulas, key=lambda row: bbox_from_array(row[0])[0])
    kept: list[Any] = []
    for item in items:
        pieces = [item]
        for formula_box, latex, _score in ordered:
            next_pieces: list[Any] = []
            for piece in pieces:
                if getattr(piece, "formula_like", False):
                    next_pieces.append(piece)
                    continue
                next_pieces.extend(
                    _split_piece(piece, formula_box, latex, make_item, normalize_text)
                )
            pieces = next_pieces
        kept.extend(pieces)
    return kept


def suppress_overlapping_duplicate_items(items: Sequence[Any]) -> list[Any]:
    """Port of test7 overlapping-text suppression onto OCR boxes."""

    retained: list[Any] = []
    drop: set[int] = set()

    def compact_text(item: Any) -> str:
        return re.sub(r"\s+", "", str(getattr(item, "text", "") or ""))

    for item in items:
        text = compact_text(item)
        if len(text) < 2:
            retained.append(item)
            continue
        box = item_bbox(item)
        duplicate = False
        for other_index, other in enumerate(retained):
            if compact_text(other) != text:
                continue
            other_box = item_bbox(other)
            if (
                bbox_iou(box, other_box) >= 0.75
                or bbox_overlap_ratio_min(box, other_box) >= 0.9
            ):
                if getattr(item, "formula_like", False) and not getattr(
                    other, "formula_like", False
                ):
                    retained[other_index] = item
                duplicate = True
                break
        if not duplicate:
            retained.append(item)

    for left_index, left in enumerate(retained):
        left_compact = compact_text(left)
        if len(left_compact) < 4:
            continue
        left_box = item_bbox(left)
        for right_index in range(left_index + 1, len(retained)):
            right = retained[right_index]
            right_compact = compact_text(right)
            if len(right_compact) < 4 or left_compact == right_compact:
                continue
            if len(left_compact) < len(right_compact):
                short, short_compact, short_box, short_index = (
                    left, left_compact, left_box, left_index
                )
                long_item = right
            else:
                short, short_compact, short_box, short_index = (
                    right, right_compact, item_bbox(right), right_index
                )
                long_item = left
            if short_compact not in compact_text(long_item):
                continue
            long_box = item_bbox(long_item)
            horizontal, vertical = _overlap_wh(short_box, long_box)
            overlap_area = horizontal * vertical
            short_area = max(1.0, (short_box[2] - short_box[0]) * (short_box[3] - short_box[1]))
            vertical_ratio = vertical / max(
                1.0,
                min(short_box[3] - short_box[1], long_box[3] - long_box[1]),
            )
            if overlap_area / short_area >= 0.85 and vertical_ratio >= 0.75:
                if getattr(short, "formula_like", False) and not getattr(
                    long_item, "formula_like", False
                ):
                    continue
                drop.add(short_index)

    for index, item in enumerate(retained):
        if getattr(item, "formula_like", False):
            continue
        text = compact_text(item)
        if not text:
            continue
        box = item_bbox(item)
        raw_text = str(getattr(item, "text", "") or "")
        for other in retained:
            if other is item or not getattr(other, "formula_like", False):
                continue
            if not formula_overlaps_line(box, item_bbox(other)):
                continue
            latex = str(getattr(other, "text", "") or "").strip("$")
            if looks_like_flattened_formula(text, latex) and (
                item_is_math_shaped(raw_text, False) or len(text) <= 24
            ):
                drop.add(index)
                break

    return [item for index, item in enumerate(retained) if index not in drop]


def fuse_formula_items(
    items: Sequence[Any],
    formula_audit: dict[str, object],
    *,
    make_item: MakeItem,
    normalize_text: Callable[[list[str]], str],
    y_offset: float = 0.0,
    allow_unanchored: bool = False,
) -> tuple[list[Any], int]:
    """Replace flattened RapidOCR spans with FormulaNet boxes, then dedupe."""

    accepted = accept_formula_boxes(
        items,
        formula_audit,
        y_offset=y_offset,
        allow_unanchored=allow_unanchored,
    )
    if not accepted:
        return list(items), 0
    split_items = split_items_around_formulas(
        items,
        accepted,
        make_item,
        normalize_text,
    )
    for box, latex, score in accepted:
        split_items.append(
            make_item(
                box,
                f"${latex}$",
                max(0.60, score),
                formula_like=True,
            )
        )
    return suppress_overlapping_duplicate_items(split_items), len(accepted)
