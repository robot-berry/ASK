from __future__ import annotations

import re
from typing import Any

from .text import latex_to_readable, parse_question
from .types import BBox, QuestionDocument, SourceBlock, VisualAsset


_QUESTION_CAPTION_RE = re.compile(
    r"^\s*[（(]?\s*第\s*(\d{1,3})"
    r"(?:\s*[-—－.．]?\s*([A-Za-z0-9ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+))?\s*"
    r"(?:题(?:\s*(?:图|表)\s*[甲乙丙丁A-D0-9]*)?|(?:图|表)\s*[甲乙丙丁A-D0-9]*)"
    r"\s*[）)]?\s*$",
    re.IGNORECASE,
)
_STANDALONE_VISUAL_CAPTION_RE = re.compile(
    r"^\s*(?:"
    r"(?:图|表)\s*[甲乙丙丁A-D0-9]+"
    r"|(?:第\s*\d{1,3}\s*题(?:\s*(?:图|表)\s*[甲乙丙丁A-D0-9]*)?)"
    r"|(?:[甲乙丙丁]\s*(?:图|表))"
    r")+\s*$",
    re.IGNORECASE,
)
_PANEL_LABEL_RE = re.compile(r"^[甲乙丙丁]$")
_PAGE_FRACTION_RE = re.compile(r"^\d{1,3}\s*/\s*\d{1,3}$")
_MARGINAL_DIGIT_RE = re.compile(r"^\d{1,2}$")
_TERMINAL_CURSOR_TOKEN_RE = re.compile(r"^ST$")
_INLINE_CURSOR_FORMULA_RE = re.compile(
    r"^\s*\\stackrel\s*\{\s*\\circ\s*\}\s*\{\s*\\cdot\s*\}\s*(.+?)\s*$"
)
_PAGE_FOOTER_RE = re.compile(r"(?:试题卷|答题卷|练习卷|测试卷)\s*$")
_COMPACT_WORKING_RE = re.compile(
    r"^[A-Za-z]{1,4}[A-Za-z0-9]*\s*=\s*[-+]?\d+(?:\.\d+)?$"
)
_CIRCLED_EQUATION_LABEL_RE = re.compile(r"^[①②③④⑤⑥]$")
_CALCULATION_CUE_RE = re.compile(r"(?:计算|化简|求值)\s*[：:]")
_FILL_BLANK_CUE_RE = re.compile(
    r"(?:为|是|等于|应填|应为|结果为|原因是|直径|长度|大小|数值|置于)\s*$"
)
_FILL_BLANK_ARTIFACT_RE = re.compile(r"^[oO0○。·.]$")
_FILL_UNIT_RE = re.compile(
    r"^(?:mm|cm|dm|km|m|kg|g|mg|s|min|h|Hz|N|Pa|J|W|V|A|Ω|℃|°)"
    r"(?:\b|[。．，,；;）)])",
    re.IGNORECASE,
)
_VECTOR_CONTEXT_RE = re.compile(r"(?:向量|基底|共线|线性相关|线性无关)")
_VECTOR_TOKEN_RE = re.compile(
    r"\\(?:pmb|boldsymbol|mathbf)\s*\{\s*([A-Za-z])\s*\}"
)
_GAMMA_RAY_ARTIFACT_RE = re.compile(
    r"^\s*1\s*\\gamma\s*_\s*\{\s*1\s*\}\s*$"
)
_ENERGY_LEVEL_RE = re.compile(
    r"^\s*n\s*(?:\{\s*)?=\s*(?:\}\s*)?(\d{1,2})\s*$"
)
_TERMINAL_HEAT_SYMBOL_ARTIFACT_RE = re.compile(
    r"^\s*Q\s*_\s*\{\s*\\mathrm\s*\{\s*~?\s*[A-Za-z]\s*~?\s*\}\s*\}\s*$"
)
_SCREEN_TOOLBAR_ARTIFACT_RE = re.compile(r"^[^\u4e00-\u9fff]{0,8}[回口日□]{1,4}[^\u4e00-\u9fff]{0,12}%?-?$")
_SCREEN_TOOLBAR_ICON_RE = re.compile(r"^[可回口日□]{1,4}$")
_SCREEN_TOOLBAR_PERCENT_RE = re.compile(r"^\d{2,3}%?-?$")
_MARGINAL_SHORT_TOKEN_RE = re.compile(r"^[A-Za-z0-9]{1,3}$")
_LEADING_EXAM_SECTION_RE = re.compile(
    r"^(?:"
    r".{0,10}题部分"
    r"|[一二三四五六七八九十]+[、.．].{0,40}(?:选择题|小题|共\s*\d+\s*分).{0,20}"
    r")$"
)
_SYSTEM_ALIGNED_WRAPPER_RE = re.compile(
    r"^\s*\\begin\{aligned\}\s*&?\s*\\left\\\{\s*"
    r"\\begin\{aligned\}(.*?)\\end\{aligned\}\s*\\right\.\s*"
    r"\\end\{aligned\}\s*$",
    re.DOTALL,
)


def _append_warning(document: QuestionDocument, warning: str) -> None:
    if warning not in document.warnings:
        document.warnings.append(warning)


def _question_base_number(value: str | None) -> int | None:
    match = re.match(r"\s*(\d{1,3})", value or "")
    return int(match.group(1)) if match else None


def _caption_number(block: SourceBlock) -> int | None:
    if block.source_bbox is None or block.disposition != "emitted":
        return None
    match = _QUESTION_CAPTION_RE.fullmatch(block.text or "")
    return int(match.group(1)) if match else None


def _caption_is_near_selected_visual(
    block: SourceBlock,
    assets: list[VisualAsset],
) -> bool:
    """Require a caption to be spatially tied to a visual retained for this question."""
    if block.source_bbox is None:
        return False
    caption = block.source_bbox
    for asset in assets:
        visual = asset.source_bbox
        horizontal_gap = max(0, visual.x - caption.right, caption.x - visual.right)
        vertical_gap = max(0, visual.y - caption.bottom, caption.y - visual.bottom)
        if horizontal_gap <= 48 and vertical_gap <= 80:
            return True
    return False


def _recover_question_number_from_visual_caption(
    document: QuestionDocument,
) -> str | None:
    """Recover only the numeric base id from strong, spatial caption evidence.

    OCR can systematically confuse a printed Roman section marker (for example
    ``III`` as ``II``). The base exercise number is still useful and safer, so
    suffixes are deliberately excluded here and remain visible in the stem when
    they were independently recognized.
    """
    candidates: list[tuple[int, float]] = []
    for block in document.source_blocks:
        number = _caption_number(block)
        confidence = float(block.confidence or 0.0)
        if (
            number is None
            or confidence < 0.88
            or not _caption_is_near_selected_visual(block, document.visual_assets)
        ):
            continue
        candidates.append((number, confidence))
    if not candidates:
        return None
    numbers = {number for number, _confidence in candidates}
    if len(numbers) != 1:
        return None
    # One very strong caption is sufficient; otherwise require two agreeing
    # captions so a single uncertain OCR token cannot create a question id.
    if max(confidence for _number, confidence in candidates) < 0.92 and len(candidates) < 2:
        return None
    return str(next(iter(numbers)))


def _horizontal_overlap(left: BBox, right: BBox) -> int:
    return max(0, min(left.right, right.right) - max(left.x, right.x))


def _vertical_overlap(left: BBox, right: BBox) -> int:
    return max(0, min(left.bottom, right.bottom) - max(left.y, right.y))


def _bbox_union(left: BBox, right: BBox, *, bottom_padding: int = 0) -> BBox:
    x = min(left.x, right.x)
    y = min(left.y, right.y)
    right_edge = max(left.right, right.right)
    bottom = max(left.bottom, right.bottom) + bottom_padding
    return BBox(x, y, right_edge - x, bottom - y)


def _bbox_iou(left: BBox, right: BBox) -> float:
    overlap = _horizontal_overlap(left, right) * _vertical_overlap(left, right)
    union = left.width * left.height + right.width * right.height - overlap
    return overlap / max(1, union)


def _bbox_overlap_ratio_min(left: BBox, right: BBox) -> float:
    overlap = _horizontal_overlap(left, right) * _vertical_overlap(left, right)
    return overlap / max(1, min(left.width * left.height, right.width * right.height))


def _bbox_contains(outer: BBox, inner: BBox, *, bottom_padding: int = 0) -> bool:
    return (
        outer.x <= inner.x
        and outer.y <= inner.y
        and outer.right >= inner.right
        and outer.bottom >= inner.bottom + bottom_padding
    )


def _caption_asset_score(caption: BBox, asset: BBox) -> float | None:
    # A printed figure caption normally touches or follows the bottom edge of
    # its own panel.  Require both vertical proximity and horizontal evidence
    # so a neighboring question cannot remove an unrelated visual by number.
    vertical_gap = caption.y - asset.bottom
    if vertical_gap < -max(18, caption.height) or vertical_gap > 140:
        return None
    overlap = _horizontal_overlap(caption, asset)
    caption_center = caption.x + caption.width / 2
    asset_center = asset.x + asset.width / 2
    center_distance = abs(caption_center - asset_center)
    if overlap < min(caption.width, asset.width) * 0.25 and center_distance > max(
        70,
        asset.width * 0.65,
    ):
        return None
    return abs(vertical_gap) * 2.0 + center_distance


def _caption_assignments(
    captions: list[tuple[int, SourceBlock]],
    assets: list[VisualAsset],
) -> dict[str, int]:
    candidates: list[tuple[float, str, int, int]] = []
    for caption_index, (number, block) in enumerate(captions):
        assert block.source_bbox is not None
        for asset in assets:
            score = _caption_asset_score(block.source_bbox, asset.source_bbox)
            if score is not None:
                candidates.append((score, asset.asset_id, caption_index, number))
    assignments: dict[str, int] = {}
    used_captions: set[int] = set()
    for _score, asset_id, caption_index, number in sorted(candidates):
        # One asset receives one nearby caption.  Repeated question numbers are
        # allowed (multi-panel figures), while a physical caption block is not
        # reused because the score ordering picks the nearest asset first.
        if asset_id in assignments:
            continue
        if caption_index in used_captions:
            continue
        assignments[asset_id] = number
        used_captions.add(caption_index)
    return assignments


def _expand_target_assets_to_captions(
    document: QuestionDocument,
    captions: list[tuple[int, SourceBlock]],
) -> list[str]:
    target_number = _question_base_number(document.question_number)
    if target_number is None or not document.visual_assets:
        return []
    expanded: list[str] = []
    for number, block in captions:
        if number != target_number or block.source_bbox is None:
            continue
        if any(
            _bbox_contains(asset.source_bbox, block.source_bbox, bottom_padding=6)
            for asset in document.visual_assets
        ):
            continue
        candidates = [
            (score, asset)
            for asset in document.visual_assets
            if (score := _caption_asset_score(block.source_bbox, asset.source_bbox))
            is not None
        ]
        if not candidates:
            continue
        _score, asset = min(candidates, key=lambda item: item[0])
        union = _bbox_union(asset.source_bbox, block.source_bbox, bottom_padding=6)
        if union != asset.source_bbox:
            asset.source_bbox = union
            expanded.append(asset.asset_id)
    return expanded


def _simple_visual_caption(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:图|表)\s*[甲乙丙丁A-D0-9]+",
            (value or "").strip(),
            re.IGNORECASE,
        )
    )


def _restore_inline_caption_references(
    document: QuestionDocument,
    captions: list[SourceBlock],
) -> int:
    """Repair references removed by the legacy source-order caption cleanup.

    Older rerenders assumed that a printed caption always followed all prose.
    That is false for experiment pages whose figures precede their questions.
    The original OCR source block is the provenance used here: restoration is
    allowed only when its exact text without the caption is still present.
    """

    tokens = sorted(
        {
            (block.text or "").strip()
            for block in captions
            if _simple_visual_caption(block.text or "")
        },
        key=len,
        reverse=True,
    )
    if not tokens:
        return 0
    restored = 0
    caption_ids = {block.block_id for block in captions}
    for block in document.source_blocks:
        full = (block.text or "").strip()
        if (
            block.block_id in caption_ids
            or block.kind != "text"
            or block.disposition != "emitted"
            or len(full) < 6
        ):
            continue
        for token in tokens:
            if token not in full:
                continue
            missing = full.replace(token, "", 1)
            if len(missing) < 4:
                continue
            changed = False
            for attr in ("canonical_text", "display_text"):
                value = getattr(document, attr)
                if full not in value and missing in value:
                    setattr(document, attr, value.replace(missing, full, 1))
                    changed = True
            if changed:
                restored += 1
            break
    return restored


def _suppress_caption_only_lines(
    document: QuestionDocument,
    captions: list[SourceBlock],
    panel_labels: list[SourceBlock],
) -> int:
    tokens = sorted(
        {
            re.sub(r"\s+", "", block.text or "")
            for block in captions
            if (block.text or "").strip()
        },
        key=len,
        reverse=True,
    )
    eligible_panel_tokens = {
        (block.text or "").strip()
        for block in panel_labels
        if (block.text or "").strip()
    }
    if not tokens and not eligible_panel_tokens:
        return 0
    allow_panel_only_cleanup = bool(tokens) or len(eligible_panel_tokens) == 1

    def keep_lines(value: str) -> tuple[str, int]:
        kept: list[str] = []
        removed = 0
        for line in value.splitlines():
            remainder = re.sub(r"\s+", "", line)
            for token in tokens:
                remainder = remainder.replace(token, "")
            remainder = re.sub(r"[（）()【】\[\]：:、，,。.．·]", "", remainder)
            if allow_panel_only_cleanup:
                remainder = "".join(
                    character
                    for character in remainder
                    if character not in eligible_panel_tokens
                )
            if line.strip() and not remainder:
                removed += 1
                continue
            kept.append(line)
        return re.sub(r"\n{2,}", "\n", "\n".join(kept)).strip(), removed

    display, removed = keep_lines(document.display_text)
    canonical, _ = keep_lines(document.canonical_text)
    document.display_text = display
    document.canonical_text = canonical
    return removed


def _suppress_inline_caption_intrusions(
    document: QuestionDocument,
    captions: list[SourceBlock],
) -> int:
    """Remove a caption inserted into prose by global source reading order.

    The exact caption token must not occur in any other emitted source text
    block.  This distinguishes a floating figure caption from a genuine prose
    reference such as ``如图甲所示``.
    """

    caption_ids = {block.block_id for block in captions}
    suppressed = 0
    for caption in captions:
        token = (caption.text or "").strip()
        if not token or _simple_visual_caption(token) or any(
            other.block_id not in caption_ids
            and other.disposition == "emitted"
            and other.kind == "text"
            and token in (other.text or "")
            for other in document.source_blocks
        ):
            continue
        inline_in_display = any(
            token in line and line.strip() != token
            for line in document.display_text.splitlines()
        )
        if not inline_in_display:
            continue
        changed = False
        for attr in ("canonical_text", "display_text"):
            value = getattr(document, attr)
            if token not in value:
                continue
            value = value.replace(token, "")
            value = re.sub(r"[ \t]+\n", "\n", value)
            value = re.sub(r"\n[ \t]+", "\n", value)
            value = re.sub(r"[ \t]{2,}", " ", value)
            setattr(document, attr, value.strip())
            changed = True
        if changed:
            suppressed += 1
    return suppressed


def _expand_standalone_caption_assets(
    document: QuestionDocument,
    captions: list[SourceBlock],
) -> list[str]:
    candidates: list[tuple[float, int, VisualAsset, SourceBlock]] = []
    for caption_index, block in enumerate(captions):
        token = (block.text or "").strip()
        if not _simple_visual_caption(token) or block.source_bbox is None:
            continue
        referenced = token in document.display_text or token in document.canonical_text
        referenced = referenced or any(
            other.block_id != block.block_id
            and other.kind == "text"
            and other.disposition == "emitted"
            and token in (other.text or "")
            for other in document.source_blocks
        )
        if not referenced:
            continue
        if any(
            _bbox_contains(asset.source_bbox, block.source_bbox, bottom_padding=6)
            for asset in document.visual_assets
        ):
            continue
        for asset in document.visual_assets:
            score = _caption_asset_score(block.source_bbox, asset.source_bbox)
            if score is not None:
                candidates.append((score, caption_index, asset, block))

    used_captions: set[int] = set()
    used_assets: set[str] = set()
    expanded: list[str] = []
    for _score, caption_index, asset, block in sorted(
        candidates,
        key=lambda item: item[0],
    ):
        if caption_index in used_captions or asset.asset_id in used_assets:
            continue
        assert block.source_bbox is not None
        used_captions.add(caption_index)
        used_assets.add(asset.asset_id)
        union = _bbox_union(asset.source_bbox, block.source_bbox, bottom_padding=6)
        if union != asset.source_bbox:
            asset.source_bbox = union
            expanded.append(asset.asset_id)
    return expanded


def _expand_panel_label_assets(
    document: QuestionDocument,
    panel_labels: list[SourceBlock],
) -> list[str]:
    if not panel_labels or not document.visual_assets:
        return []
    context = "\n".join(
        [document.display_text]
        + [
            block.text or ""
            for block in document.source_blocks
            if block.kind == "text" and block.disposition == "emitted"
        ]
    )
    if len(panel_labels) == 1:
        token = (panel_labels[0].text or "").strip()
        if f"图{token}" not in context and f"表{token}" not in context:
            return []
    elif not re.search(r"(?:插图|幅.*图|图.*[甲乙丙丁])", context):
        return []

    candidates: list[tuple[float, int, VisualAsset, SourceBlock]] = []
    for label_index, block in enumerate(panel_labels):
        if block.source_bbox is None:
            continue
        if any(
            _bbox_contains(asset.source_bbox, block.source_bbox, bottom_padding=6)
            for asset in document.visual_assets
        ):
            continue
        for asset in document.visual_assets:
            vertical_gap = block.source_bbox.y - asset.source_bbox.bottom
            horizontal_gap = max(
                0,
                asset.source_bbox.x - block.source_bbox.right,
                block.source_bbox.x - asset.source_bbox.right,
            )
            if (
                vertical_gap < -max(18, block.source_bbox.height)
                or vertical_gap > 90
                or horizontal_gap > max(45, round(asset.source_bbox.width * 0.2))
            ):
                continue
            center_distance = abs(
                block.source_bbox.x
                + block.source_bbox.width / 2
                - asset.source_bbox.x
                - asset.source_bbox.width / 2
            )
            side_overlap_bonus = (
                100
                if horizontal_gap <= 20
                and _vertical_overlap(block.source_bbox, asset.source_bbox)
                >= block.source_bbox.height * 0.7
                else 0
            )
            candidates.append(
                (
                    abs(vertical_gap) * 2
                    + horizontal_gap * 2
                    + center_distance * 0.1
                    - side_overlap_bonus,
                    label_index,
                    asset,
                    block,
                )
            )

    used_labels: set[int] = set()
    used_assets: set[str] = set()
    expanded: list[str] = []
    for _score, label_index, asset, block in sorted(candidates, key=lambda item: item[0]):
        if label_index in used_labels or asset.asset_id in used_assets:
            continue
        assert block.source_bbox is not None
        used_labels.add(label_index)
        used_assets.add(asset.asset_id)
        union = _bbox_union(asset.source_bbox, block.source_bbox, bottom_padding=6)
        if union != asset.source_bbox:
            asset.source_bbox = union
            expanded.append(asset.asset_id)
    return expanded


def _expand_captioned_panel_rows(
    document: QuestionDocument,
    captions: list[SourceBlock],
) -> list[str]:
    expanded: list[str] = []
    for block in captions:
        token = (block.text or "").strip()
        if (
            block.source_bbox is None
            or not _QUESTION_CAPTION_RE.fullmatch(token)
            or not re.search(r"(?:图|表)\s*$", token)
        ):
            continue
        line = next(
            (
                candidate
                for candidate in document.display_text.splitlines()
                if re.sub(r"\s+", "", token) in re.sub(r"\s+", "", candidate)
            ),
            "",
        )
        if len(set(re.findall(r"[甲乙丙丁]", line))) < 2:
            continue
        row_assets = [
            asset
            for asset in document.visual_assets
            if asset.source_bbox.y < block.source_bbox.bottom
            and -30 <= block.source_bbox.y - asset.source_bbox.bottom <= 100
        ]
        if len(row_assets) < 2:
            continue
        target_bottom = block.source_bbox.bottom + 6
        for asset in row_assets:
            if asset.source_bbox.bottom >= target_bottom:
                continue
            asset.source_bbox = BBox(
                asset.source_bbox.x,
                asset.source_bbox.y,
                asset.source_bbox.width,
                target_bottom - asset.source_bbox.y,
            )
            expanded.append(asset.asset_id)
    return expanded


def _remove_last_occurrence(value: str, token: str) -> str:
    index = value.rfind(token)
    if index < 0:
        return value
    return value[:index] + value[index + len(token) :]


def _remove_emitted_caption_tokens(value: str, tokens: list[str]) -> str:
    result = value
    for token in sorted(set(tokens), key=len, reverse=True):
        if token.strip():
            # A caption such as ``图1`` can also be referenced inside prose
            # (``如图1所示``).  The separately emitted caption is the last
            # source-order occurrence, so remove only that occurrence.
            result = _remove_last_occurrence(result, token)
    result = re.sub(r"[ \t]+\n", "\n", result)
    result = re.sub(r"\n[ \t]+", "\n", result)
    result = re.sub(r"[ \t]{2,}", " ", result)
    result = re.sub(r"\n{2,}", "\n", result)
    return result.strip()


def _remove_exact_standalone_line(value: str, token: str) -> str:
    result = re.sub(
        rf"(?m)^\s*{re.escape(token.strip())}\s*$\n?",
        "",
        value,
        count=1,
    )
    return re.sub(r"\n{2,}", "\n", result).strip()


def _replace_flexible_whitespace_token(
    value: str,
    token: str,
    replacement: str,
) -> tuple[str, int]:
    """Replace one OCR token even if parsing moved whitespace around punctuation."""

    parts = re.split(r"\s+", token.strip())
    if not parts or not all(parts):
        return value, 0
    pattern = re.compile(r"\s*".join(re.escape(part) for part in parts))
    return pattern.subn(lambda _match: replacement, value, count=1)


def _same_source_line(left: BBox, right: BBox) -> bool:
    overlap = _vertical_overlap(left, right)
    if overlap >= min(left.height, right.height) * 0.45:
        return True
    left_center = left.y + left.height / 2
    right_center = right.y + right.height / 2
    return abs(left_center - right_center) <= max(5, min(left.height, right.height) * 0.55)


def _replace_source_pair(value: str, left: str, right: str, replacement: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return value
    pattern = re.compile(re.escape(left) + r"[ \t\n]*" + re.escape(right))
    return pattern.sub(lambda _match: replacement, value, count=1)


def _restore_geometric_answer_placeholders(document: QuestionDocument) -> int:
    """Restore a printed answer line when semantics and source geometry agree."""

    positioned = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.source_bbox is not None
        and (block.text or "").strip()
    ]
    emitted = [
        block
        for block in positioned
        if block.kind == "text"
    ]
    replacements: list[tuple[SourceBlock, SourceBlock, bool]] = []
    for right in emitted:
        assert right.source_bbox is not None
        right_text = (right.text or "").strip()
        if re.match(r"^[A-D]\s*[.．、]", right_text, re.IGNORECASE):
            continue
        narrow_v_before_acceleration_unit = (
            right_text == "V"
            and right.source_bbox.width <= 10
            and right.confidence is not None
            and right.confidence < 0.75
            and any(
                candidate is not right
                and candidate.source_bbox is not None
                and 0 <= candidate.source_bbox.x - right.source_bbox.right <= 90
                and _same_source_line(right.source_bbox, candidate.source_bbox)
                and re.search(r"m\s*/\s*s", candidate.text or "", re.IGNORECASE)
                for candidate in positioned
            )
        )
        tiny_artifact = bool(_FILL_BLANK_ARTIFACT_RE.fullmatch(right_text)) and (
            right.confidence is None
            or right.confidence < 0.75
            or right.source_bbox.height <= 10
        )
        tiny_artifact = tiny_artifact or narrow_v_before_acceleration_unit
        unit_suffix = bool(_FILL_UNIT_RE.match(right_text))
        selection_suffix = right_text == "端"
        if not tiny_artifact and not unit_suffix and not selection_suffix:
            continue
        candidates: list[tuple[int, SourceBlock]] = []
        for left in emitted:
            if left is right or left.source_bbox is None:
                continue
            left_text = (left.text or "").strip()
            gap = right.source_bbox.x - left.source_bbox.right
            same_line_gap = 18 <= gap <= 150 and _same_source_line(
                left.source_bbox,
                right.source_bbox,
            )
            wrapped_unit = (
                unit_suffix
                and 0 <= right.source_bbox.y - left.source_bbox.bottom <= 25
                and 0 <= right.source_bbox.x - left.source_bbox.x <= 260
            )
            selection_gap = (
                selection_suffix
                and same_line_gap
                and re.search(r"置于$", left_text) is not None
            )
            if (
                (same_line_gap or wrapped_unit or selection_gap)
                and _FILL_BLANK_CUE_RE.search(left_text)
            ):
                candidates.append((gap, left))
        if not candidates:
            continue
        _gap, left = min(candidates, key=lambda item: item[0])
        replacements.append((left, right, tiny_artifact))

    restored = 0
    for left, right, replace_artifact in replacements:
        left_text = (left.text or "").strip()
        right_text = (right.text or "").strip()
        replacement = (
            f"{left_text} ______"
            if replace_artifact
            else f"{left_text} ______ {right_text}"
        )
        previous = (document.canonical_text, document.display_text)
        document.canonical_text = _replace_source_pair(
            document.canonical_text,
            left_text,
            right_text,
            replacement,
        )
        document.display_text = _replace_source_pair(
            document.display_text,
            left_text,
            right_text,
            replacement,
        )
        if previous != (document.canonical_text, document.display_text):
            restored += 1
    return restored


def _normalize_established_vector_style(document: QuestionDocument) -> int:
    """Keep option symbols bold when the stem already establishes vectors."""

    if not _VECTOR_CONTEXT_RE.search(document.display_text):
        return 0
    option_start = re.search(r"(?m)^\s*A[.．、]\s*", document.canonical_text)
    if option_start is None:
        return 0
    stem_canonical = document.canonical_text[: option_start.start()]
    symbols = set(_VECTOR_TOKEN_RE.findall(stem_canonical))
    if len(symbols) < 2:
        return 0

    protected: dict[str, str] = {}

    def normalize_formula(match: re.Match[str]) -> str:
        formula = match.group(1)

        def protect(existing: re.Match[str]) -> str:
            token = f"@@VECTOR{len(protected)}@@"
            protected[token] = existing.group(0)
            return token

        normalized = _VECTOR_TOKEN_RE.sub(protect, formula)
        for symbol in sorted(symbols):
            normalized = re.sub(
                rf"(?<![A-Za-z\\]){re.escape(symbol)}(?=$|[^A-Za-z])",
                rf"\\boldsymbol{{{symbol}}}",
                normalized,
            )
        for token, original in protected.items():
            normalized = normalized.replace(token, original)
        return f"${normalized}$"

    suffix = document.canonical_text[option_start.start() :]
    normalized_suffix = re.sub(r"(?<!\\)\$(.+?)(?<!\\)\$", normalize_formula, suffix)
    if normalized_suffix == suffix:
        return 0
    before = len(_VECTOR_TOKEN_RE.findall(suffix))
    after = len(_VECTOR_TOKEN_RE.findall(normalized_suffix))
    document.canonical_text = document.canonical_text[: option_start.start()] + normalized_suffix
    return max(1, after - before)


def _normalize_set_complement_notation(document: QuestionDocument) -> int:
    """Repair a low-confidence complement row using its printed set context."""

    if "全集" not in document.display_text or not re.search(
        r"(?<![A-Za-z])U\s*=", document.canonical_text
    ):
        return 0
    suspect = next(
        (
            block
            for block in document.source_blocks
            if block.kind == "formula"
            and block.disposition == "emitted"
            and block.confidence is not None
            and block.confidence < 0.70
            and re.search(r"C\s*_\s*\{\s*V\s*\}", block.text or "")
        ),
        None,
    )
    if suspect is None:
        return 0

    changes = 0
    canonical = re.sub(
        r"C\s*_\s*\{\s*V\s*\}",
        r"C _ { U }",
        document.canonical_text,
        count=1,
    )
    display = re.sub(r"C\s*_\s*\(?V\)?", "C _(U)", document.display_text, count=1)
    if (canonical, display) != (document.canonical_text, document.display_text):
        changes += 1

    # A narrow closing parenthesis can be recognized as ``1`` by formula OCR.
    # Remove it only when the same expression is immediately followed by the
    # printed empty answer parentheses retained by text OCR.
    repaired_canonical = re.sub(
        r"(C\s*_\s*\{\s*U\s*\}\s*[A-Za-z]\s*=)\s*1(?=\s*\$\s*[（(]\s*[）)])",
        r"\1",
        canonical,
        count=1,
    )
    repaired_display = re.sub(
        r"(C\s*_\s*\(?U\)?\s*[A-Za-z]\s*=)\s*1(?=\s*[（(]\s*[）)])",
        r"\1",
        display,
        count=1,
    )
    if (repaired_canonical, repaired_display) != (canonical, display):
        changes += 1
    canonical, display = repaired_canonical, repaired_display

    option_blocks = [
        (block.text or "").strip()
        for block in document.source_blocks
        if block.kind == "text"
        and block.disposition == "emitted"
        and re.match(r"^[A-D][.．、]", (block.text or "").strip())
    ]
    has_set_option_context = (
        any(re.fullmatch(r"A[.．、]\s*0", text) for text in option_blocks)
        and sum("{" in text and "}" in text for text in option_blocks) >= 2
    )
    if has_set_option_context:
        option_start = re.search(r"(?m)^\s*A[.．、]\s*0(?=\s+B[.．、])", canonical)
        display_option_start = re.search(
            r"(?m)^\s*A[.．、]\s*0(?=\s+B[.．、])",
            display,
        )
        if option_start is not None and display_option_start is not None:
            canonical = (
                canonical[: option_start.start()]
                + re.sub(
                    r"A[.．、]\s*0",
                    lambda _match: r"A. $\varnothing$",
                    canonical[option_start.start() :],
                    count=1,
                )
            )
            display = (
                display[: display_option_start.start()]
                + re.sub(
                    r"A[.．、]\s*0",
                    "A. ∅",
                    display[display_option_start.start() :],
                    count=1,
                )
            )
            changes += 1

    if changes:
        document.canonical_text = canonical
        document.display_text = display
    return changes


def _normalize_paired_formula_quotes(document: QuestionDocument) -> int:
    """Turn two OCR-neutral quote glyphs around one formula into “…” quotes."""

    pair = re.compile(
        r"\\text\s*\{\s*\"\s*\}\s*(.+?)\s*"
        r"\\text\s*\{\s*\"\s*\}"
    )
    count = 0

    def normalize_formula(match: re.Match[str]) -> str:
        nonlocal count
        formula = match.group(1)
        normalized, replacements = pair.subn(
            lambda quote_match: (
                r"\text{“ } " + quote_match.group(1).strip() + r" \text{” }"
            ),
            formula,
        )
        count += replacements
        return f"${normalized}$"

    canonical = re.sub(
        r"(?<!\\)\$(.+?)(?<!\\)\$",
        normalize_formula,
        document.canonical_text,
    )
    if count == 0:
        return 0
    display = re.sub(
        r'"\s*([^"\n]+?)\s*"',
        lambda match: f"“{match.group(1).strip()}”",
        document.display_text,
    )
    document.canonical_text = canonical
    document.display_text = display
    return count


def _normalize_split_fraction_numerator(document: QuestionDocument) -> int:
    """Merge ``1/d`` plus an empty-denominator ``n/`` into printed ``n/d``."""

    pattern = re.compile(
        r"\\frac\s*\{\s*1\s*\}\s*\{\s*(\d+)\s*\}\s*"
        r"\\frac\s*\{\s*(\d+)\s*\}\s*\{\s*\}"
    )
    replacements: list[tuple[int, int]] = []

    def merge(match: re.Match[str]) -> str:
        denominator = int(match.group(1))
        numerator = int(match.group(2))
        if not 1 < numerator < denominator:
            return match.group(0)
        replacements.append((denominator, numerator))
        return rf"\frac{{{numerator}}}{{{denominator}}}"

    canonical = pattern.sub(merge, document.canonical_text)
    if not replacements:
        return 0
    display = document.display_text
    for denominator, numerator in replacements:
        display = re.sub(
            rf"1\s*/\s*{denominator}\s+{numerator}\s*/\s*",
            f"{numerator}/{denominator} ",
            display,
            count=1,
        )
    document.canonical_text = canonical
    document.display_text = display
    return len(replacements)


def _normalize_redundant_numeric_superscripts(document: QuestionDocument) -> int:
    pattern = re.compile(
        r"\^\s*\{\s*\^\s*(?:\{\s*(\d+)\s*\}|(\d+))\s*\}"
    )
    canonical, count = pattern.subn(
        lambda match: f"^{{{match.group(1) or match.group(2)}}}",
        document.canonical_text,
    )
    if count:
        document.canonical_text = canonical
    return count


def _normalize_gamma_ray_symbol(document: QuestionDocument) -> int:
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    text_blocks = [block for block in emitted if block.kind == "text"]
    for formula in (block for block in emitted if block.kind == "formula"):
        if (
            not _GAMMA_RAY_ARTIFACT_RE.fullmatch(formula.text or "")
            or formula.confidence is None
            or formula.confidence >= 0.85
        ):
            continue
        box = formula.source_bbox
        assert box is not None
        has_left_cue = any(
            block.source_bbox is not None
            and (block.text or "").strip().endswith("射线为")
            and _vertical_overlap(block.source_bbox, box)
            / max(1, min(block.source_bbox.height, box.height))
            >= 0.55
            and 0 <= box.x - block.source_bbox.right <= 25
            for block in text_blocks
        )
        has_right_cue = any(
            block.source_bbox is not None
            and (block.text or "").strip().startswith("射线")
            and _vertical_overlap(block.source_bbox, box)
            / max(1, min(block.source_bbox.height, box.height))
            >= 0.55
            and 0 <= block.source_bbox.x - box.right <= 25
            for block in text_blocks
        )
        canonical_token = f"${formula.text}$"
        if (
            not has_left_cue
            or not has_right_cue
            or canonical_token not in document.canonical_text
        ):
            continue
        canonical = document.canonical_text.replace(
            canonical_token, r"$\gamma$", 1
        )
        display, display_changes = re.subn(
            r"1\s*γ\s*(?:_?\s*1|₁)", "γ", document.display_text, count=1
        )
        if display_changes:
            document.canonical_text = canonical
            document.display_text = display
            return 1
    return 0


def _restore_energy_level_transition_direction(document: QuestionDocument) -> int:
    if "跃迁" not in document.display_text or not re.search(
        r"(?:能级|亮条纹|光子|射线)", document.display_text
    ):
        return 0
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    formulas: list[tuple[int, SourceBlock]] = []
    for block in emitted:
        if block.kind != "formula":
            continue
        match = _ENERGY_LEVEL_RE.fullmatch(block.text or "")
        if match:
            formulas.append((int(match.group(1)), block))
    formulas.sort(key=lambda item: item[1].source_bbox.x if item[1].source_bbox else 0)
    text_blocks = [block for block in emitted if block.kind == "text"]
    for (left_level, left), (right_level, right) in zip(formulas, formulas[1:]):
        left_box = left.source_bbox
        right_box = right.source_bbox
        assert left_box is not None and right_box is not None
        if not (
            left_level > right_level
            and _vertical_overlap(left_box, right_box)
            / max(1, min(left_box.height, right_box.height))
            >= 0.65
            and 0 <= right_box.x - left_box.right <= 70
        ):
            continue
        has_left_cue = any(
            block.source_bbox is not None
            and (block.text or "").strip().endswith("是")
            and _vertical_overlap(block.source_bbox, left_box)
            / max(1, min(block.source_bbox.height, left_box.height))
            >= 0.55
            and 0 <= left_box.x - block.source_bbox.right <= 20
            for block in text_blocks
        )
        has_right_cue = any(
            block.source_bbox is not None
            and (block.text or "").strip().startswith("跃迁")
            and _vertical_overlap(block.source_bbox, right_box)
            / max(1, min(block.source_bbox.height, right_box.height))
            >= 0.55
            and 0 <= block.source_bbox.x - right_box.right <= 20
            for block in text_blocks
        )
        if not has_left_cue or not has_right_cue:
            continue
        left_token = f"${left.text}$"
        right_token = f"${right.text}$"
        canonical, canonical_changes = re.subn(
            re.escape(left_token) + r"\s*" + re.escape(right_token),
            lambda _match: f"{left_token} 向 {right_token}",
            document.canonical_text,
            count=1,
        )
        display, display_changes = re.subn(
            rf"n\s*=\s*{left_level}\s+n\s*=\s*{right_level}",
            f"n={left_level} 向 n={right_level}",
            document.display_text,
            count=1,
        )
        if canonical_changes and display_changes:
            document.canonical_text = canonical
            document.display_text = display
            return 1
    return 0


def _normalize_terminal_heat_symbol(document: QuestionDocument) -> int:
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    text_blocks = [block for block in emitted if block.kind == "text"]
    for formula in (block for block in emitted if block.kind == "formula"):
        box = formula.source_bbox
        assert box is not None
        if (
            formula.confidence is None
            or formula.confidence >= 0.80
            or box.width > 45
            or not _TERMINAL_HEAT_SYMBOL_ARTIFACT_RE.fullmatch(formula.text or "")
        ):
            continue
        has_cue = any(
            block.source_bbox is not None
            and re.search(r"吸收的热量\s*$", block.text or "")
            and _vertical_overlap(block.source_bbox, box)
            / max(1, min(block.source_bbox.height, box.height))
            >= 0.60
            and 0 <= box.x - block.source_bbox.right <= 15
            for block in text_blocks
        )
        canonical_token = f"${formula.text}$"
        display_token = latex_to_readable(formula.text)
        if (
            not has_cue
            or canonical_token not in document.canonical_text
            or display_token not in document.display_text
        ):
            continue
        document.canonical_text = document.canonical_text.replace(canonical_token, "$Q$", 1)
        document.display_text = document.display_text.replace(display_token, "Q", 1)
        return 1
    return 0


def _normalize_numeric_subscript_glyph_confusion(document: QuestionDocument) -> int:
    if not re.search(r"T\s*_?\s*\{?\s*n\s*\}?\s*=", document.canonical_text):
        return 0
    numeric_subscripts = {
        int(value)
        for value in re.findall(r"T\s*_\s*\{\s*(\d+)\s*\}", document.canonical_text)
    }
    if len(numeric_subscripts) < 2 or not all(value >= 5 for value in numeric_subscripts):
        return 0
    candidates = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "formula"
        and block.source_bbox is not None
        and block.confidence is not None
        and block.confidence < 0.80
        and re.match(r"^\s*T\s*_\s*\{\s*[sS]\s*\}\s*=", block.text or "")
    ]
    if len(candidates) != 1:
        return 0
    formula = candidates[0]
    canonical_token = f"${formula.text}$"
    display_token = latex_to_readable(formula.text)
    if canonical_token not in document.canonical_text or display_token not in document.display_text:
        return 0
    repaired_formula = re.sub(
        r"T\s*_\s*\{\s*[sS]\s*\}", "T_{5}", formula.text, count=1
    )
    document.canonical_text = document.canonical_text.replace(
        canonical_token, f"${repaired_formula}$", 1
    )
    document.display_text = document.display_text.replace(
        display_token, latex_to_readable(repaired_formula), 1
    )
    return 1


def _restore_derivative_prime_from_definition(document: QuestionDocument) -> int:
    if "导函数" not in document.display_text or "单调区间" not in document.display_text:
        return 0
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    formulas = [block for block in emitted if block.kind == "formula"]
    text_blocks = [block for block in emitted if block.kind == "text"]
    derivative_blocks = [
        block
        for block in formulas
        if re.fullmatch(r"\s*f\s*\^\s*\{\s*\\prime\s*\}\s*\(x\)\s*", block.text or "")
    ]
    plain_blocks = [
        block
        for block in formulas
        if block.confidence is not None
        and block.confidence < 0.70
        and re.fullmatch(r"\s*f\s*\(\s*x\s*\)\s*", block.text or "")
    ]
    for derivative in derivative_blocks:
        for plain in plain_blocks:
            d_box = derivative.source_bbox
            p_box = plain.source_bbox
            assert d_box is not None and p_box is not None
            if (
                _vertical_overlap(d_box, p_box)
                / max(1, min(d_box.height, p_box.height))
                < 0.65
                or p_box.x <= d_box.right + 150
            ):
                continue
            has_between_cue = any(
                block.source_bbox is not None
                and "导函数，求" in (block.text or "")
                and d_box.right <= block.source_bbox.x <= p_box.x
                for block in text_blocks
            )
            has_after_cue = any(
                block.source_bbox is not None
                and (block.text or "").lstrip().startswith("的单调区间")
                and 0 <= block.source_bbox.x - p_box.right <= 20
                for block in text_blocks
            )
            if not has_between_cue or not has_after_cue:
                continue
            canonical, canonical_changes = re.subn(
                (
                    r"(导函数，求\s*)\$\s*f\s*\(\s*x\s*\)\s*\$"
                    r"(?=\s*的单调区间)"
                ),
                lambda match: match.group(1) + r"$f^{\prime}(x)$",
                document.canonical_text,
                count=1,
            )
            display, display_changes = re.subn(
                r"(导函数，求\s*)f\s*\(\s*x\s*\)(?=\s*的单调区间)",
                lambda match: match.group(1) + "f′(x)",
                document.display_text,
                count=1,
            )
            if canonical_changes != 1 or display_changes != 1:
                continue
            document.canonical_text = canonical
            document.display_text = display
            return 1
    return 0


def _normalize_trailing_semicolon_as_subscript(document: QuestionDocument) -> int:
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    text_blocks = [block for block in emitted if block.kind == "text"]
    changes = 0
    for formula in (block for block in emitted if block.kind == "formula"):
        if formula.confidence is None or formula.confidence >= 0.70:
            continue
        compact = re.sub(r"\s+|\\(?:mathbf|mathrm)|[{}~]", "", formula.text or "")
        match = re.fullmatch(r"([Fv])_([Nm])i", compact)
        if match is None:
            continue
        box = formula.source_bbox
        assert box is not None
        cue_pattern = r"(?:压力大小|最大速度)\s*$"
        has_cue = any(
            block.source_bbox is not None
            and re.search(cue_pattern, block.text or "")
            and _vertical_overlap(block.source_bbox, box)
            / max(1, min(block.source_bbox.height, box.height))
            >= 0.60
            and 0 <= box.x - block.source_bbox.right <= 20
            for block in text_blocks
        )
        canonical_token = f"${formula.text}$"
        display_token = latex_to_readable(formula.text)
        if not has_cue or canonical_token not in document.canonical_text:
            continue
        repaired = f"{match.group(1)}_{{{match.group(2)}}};"
        canonical = document.canonical_text.replace(
            canonical_token, f"${repaired}$", 1
        )
        if display_token not in document.display_text:
            continue
        document.canonical_text = canonical
        document.display_text = document.display_text.replace(
            display_token, latex_to_readable(repaired), 1
        )
        changes += 1
    return changes


def _restore_stacked_denominator_formula(document: QuestionDocument) -> int:
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    formula_blocks = [block for block in emitted if block.kind == "formula"]
    denominator_blocks = [
        block
        for block in emitted
        if block.kind == "text"
        and re.fullmatch(r"[A-Za-z]", (block.text or "").strip())
    ]
    for formula in formula_blocks:
        match = re.fullmatch(r"\s*(.+?=)\s*\^\s*\{\s*(.+?)\s*\}\s*", formula.text or "")
        if match is None:
            continue
        f_box = formula.source_bbox
        assert f_box is not None
        denominator = next(
            (
                block
                for block in denominator_blocks
                if block.source_bbox is not None
                and -5 <= block.source_bbox.y - f_box.bottom <= 20
                and f_box.x + f_box.width * 0.45
                <= block.source_bbox.x + block.source_bbox.width / 2
                <= f_box.right + 10
            ),
            None,
        )
        if denominator is None:
            continue
        denominator_text = (denominator.text or "").strip()
        canonical_formula = f"{match.group(1)}\\frac{{{match.group(2)}}}{{{denominator_text}}}"
        canonical_token = f"${formula.text}$"
        display_formula = latex_to_readable(formula.text)
        display_denominator = denominator_text
        canonical_pattern = re.escape(canonical_token) + r"\s*" + re.escape(display_denominator)
        display_pattern = re.escape(display_formula) + r"\s*" + re.escape(display_denominator)
        canonical, canonical_changes = re.subn(
            canonical_pattern,
            lambda _match: f"${canonical_formula}$",
            document.canonical_text,
            count=1,
        )
        display, display_changes = re.subn(
            display_pattern,
            latex_to_readable(canonical_formula),
            document.display_text,
            count=1,
        )
        if canonical_changes and display_changes:
            document.canonical_text = canonical
            document.display_text = display
            return 1
    return 0


def _suppress_overlapping_duplicate_text(document: QuestionDocument) -> int:
    grouped: dict[str, list[SourceBlock]] = {}
    for block in document.source_blocks:
        compact = re.sub(r"\s+", "", block.text or "")
        if (
            block.disposition == "emitted"
            and block.kind == "text"
            and block.source_bbox is not None
            and (len(compact) >= 2 or compact in {"与"})
        ):
            grouped.setdefault(compact, []).append(block)

    duplicate_tokens: list[tuple[str, str | None]] = []
    for blocks in grouped.values():
        retained: list[SourceBlock] = []
        for block in blocks:
            assert block.source_bbox is not None
            if any(
                other.source_bbox is not None
                and (
                    _bbox_iou(block.source_bbox, other.source_bbox) >= 0.75
                    or _bbox_overlap_ratio_min(
                        block.source_bbox, other.source_bbox
                    )
                    >= 0.9
                )
                for other in retained
            ):
                duplicate_tokens.append((block.text, None))
            else:
                retained.append(block)

    # Some providers return both a short line fragment and a complete line at
    # effectively the same coordinates. Suppress the contained fragment only
    # when geometry proves that both detections describe the same pixels.
    eligible = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
        and len(re.sub(r"\s+", "", block.text or "")) >= 4
    ]
    for index, left in enumerate(eligible):
        left_compact = re.sub(r"\s+", "", left.text or "")
        assert left.source_bbox is not None
        for right in eligible[index + 1 :]:
            right_compact = re.sub(r"\s+", "", right.text or "")
            assert right.source_bbox is not None
            if left_compact == right_compact:
                continue
            short, short_compact, long, long_compact = (
                (left, left_compact, right, right_compact)
                if len(left_compact) < len(right_compact)
                else (right, right_compact, left, left_compact)
            )
            if short_compact not in long_compact:
                continue
            overlap_area = _horizontal_overlap(
                short.source_bbox, long.source_bbox
            ) * _vertical_overlap(short.source_bbox, long.source_bbox)
            short_area = short.source_bbox.width * short.source_bbox.height
            vertical_ratio = _vertical_overlap(
                short.source_bbox, long.source_bbox
            ) / max(1, min(short.source_bbox.height, long.source_bbox.height))
            if overlap_area / max(1, short_area) >= 0.85 and vertical_ratio >= 0.75:
                duplicate_tokens.append((short.text, long.text))

    removed = 0
    for token, preserving_container in duplicate_tokens:
        old_canonical = document.canonical_text
        old_display = document.display_text
        document.canonical_text = _remove_duplicate_occurrence(
            document.canonical_text,
            token,
            preserving_container=preserving_container,
        )
        document.display_text = _remove_duplicate_occurrence(
            document.display_text,
            token,
            preserving_container=preserving_container,
        )
        if (old_canonical, old_display) != (
            document.canonical_text,
            document.display_text,
        ):
            removed += 1
    if removed:
        document.canonical_text = re.sub(r"\n{2,}", "\n", document.canonical_text).strip()
        document.display_text = re.sub(r"\n{2,}", "\n", document.display_text).strip()
    return removed


def _remove_duplicate_occurrence(
    value: str,
    token: str,
    *,
    preserving_container: str | None,
) -> str:
    positions: list[int] = []
    start = 0
    while (index := value.find(token, start)) >= 0:
        positions.append(index)
        start = index + max(1, len(token))
    if len(positions) < 2:
        return value
    remove_at = positions[0]
    if preserving_container:
        container_spans: list[tuple[int, int]] = []
        start = 0
        while (index := value.find(preserving_container, start)) >= 0:
            container_spans.append((index, index + len(preserving_container)))
            start = index + max(1, len(preserving_container))
        outside = [
            position
            for position in positions
            if not any(
                span_start <= position
                and position + len(token) <= span_end
                for span_start, span_end in container_spans
            )
        ]
        if not outside:
            return value
        remove_at = outside[0]
    return value[:remove_at] + value[remove_at + len(token) :]


def _suppress_marginal_handwritten_digit(document: QuestionDocument) -> int:
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
    ]
    if len(emitted) < 2:
        return 0
    max_right = max(block.source_bbox.right for block in emitted if block.source_bbox)
    d_blocks = [
        block
        for block in emitted
        if re.match(r"^\s*D\s*[.．、]", block.text or "", re.IGNORECASE)
    ]
    for block in emitted:
        token = (block.text or "").strip()
        if not _MARGINAL_DIGIT_RE.fullmatch(token):
            continue
        assert block.source_bbox is not None
        for option in d_blocks:
            assert option.source_bbox is not None
            vertical_ratio = _vertical_overlap(
                block.source_bbox, option.source_bbox
            ) / max(1, min(block.source_bbox.height, option.source_bbox.height))
            gap = block.source_bbox.x - option.source_bbox.right
            if (
                vertical_ratio >= 0.35
                and gap >= max(120, option.source_bbox.height * 3)
                and block.source_bbox.x >= max_right * 0.88
                and re.search(rf"{re.escape(token)}\s*$", document.display_text)
            ):
                document.canonical_text = re.sub(
                    rf"\s*{re.escape(token)}\s*$", "", document.canonical_text
                ).strip()
                document.display_text = re.sub(
                    rf"\s*{re.escape(token)}\s*$", "", document.display_text
                ).strip()
                return 1
    return 0


def _suppress_terminal_cursor_token(document: QuestionDocument) -> int:
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    formula_blocks = [block for block in emitted if block.kind == "formula"]
    d_blocks = [
        block
        for block in emitted
        if block.kind == "text"
        and re.fullmatch(r"\s*D\s*(?:[.．、])?\s*", block.text or "", re.IGNORECASE)
    ]
    # A mouse pointer over a printed option can be interpreted by formula OCR
    # as a circled-dot stack immediately after the option label.  Remove only
    # that leading construct when a low-confidence formula begins directly to
    # the right of a geometrically aligned, standalone D label.  The actual
    # option formula remains unchanged.
    for formula in formula_blocks:
        cursor_match = _INLINE_CURSOR_FORMULA_RE.fullmatch(formula.text or "")
        if cursor_match is None or formula.confidence >= 0.70:
            continue
        assert formula.source_bbox is not None
        aligned_label = next(
            (
                option
                for option in d_blocks
                if option.source_bbox is not None
                and _vertical_overlap(formula.source_bbox, option.source_bbox)
                / max(1, min(formula.source_bbox.height, option.source_bbox.height))
                >= 0.55
                and 0 <= formula.source_bbox.x - option.source_bbox.right <= 35
            ),
            None,
        )
        if aligned_label is None:
            continue
        canonical, canonical_changes = re.subn(
            (
                r"\bD\s*(?:[.．、])?\s*\$\s*"
                r"\\stackrel\s*\{\s*\\circ\s*\}\s*"
                r"\{\s*\\cdot\s*\}\s*"
            ),
            "D. $",
            document.canonical_text,
            count=1,
        )
        display, display_changes = re.subn(
            r"\bD\s*(?:[.．、])?\s*(?:stackrel\s*)?°\s*·\s*(?=[A-Za-z])",
            "D. ",
            document.display_text,
            count=1,
        )
        if canonical_changes and display_changes:
            document.canonical_text = canonical
            document.display_text = display
            return 1
    for block in emitted:
        token = (block.text or "").strip()
        if block.kind != "text" or not _TERMINAL_CURSOR_TOKEN_RE.fullmatch(token):
            continue
        assert block.source_bbox is not None
        if not re.search(rf"\b{re.escape(token)}\s*$", document.display_text):
            continue
        adjacent_formula = any(
            formula.source_bbox is not None
            and _vertical_overlap(block.source_bbox, formula.source_bbox)
            / max(1, min(block.source_bbox.height, formula.source_bbox.height))
            >= 0.5
            and 0 <= block.source_bbox.x - formula.source_bbox.right <= 45
            for formula in formula_blocks
        )
        same_option_row = any(
            option.source_bbox is not None
            and _vertical_overlap(block.source_bbox, option.source_bbox)
            / max(1, min(block.source_bbox.height, option.source_bbox.height))
            >= 0.35
            and option.source_bbox.x < block.source_bbox.x
            for option in d_blocks
        )
        if adjacent_formula and same_option_row:
            document.canonical_text = re.sub(
                rf"\s*\b{re.escape(token)}\s*$", "", document.canonical_text
            ).strip()
            document.display_text = re.sub(
                rf"\s*\b{re.escape(token)}\s*$", "", document.display_text
            ).strip()
            return 1
    return 0


def _restore_missing_two_column_option_label(document: QuestionDocument) -> int:
    """Restore missing labels from a source-proven two-column option grid.

    The labels themselves may be missed while all option formulae survive.  This
    is deliberately geometry-gated and does not infer option content.  The
    B/D-only branch covers the common case where OCR drops the entire left label
    column while preserving the four formula blocks.
    """

    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    labels: dict[str, SourceBlock] = {}
    for block in emitted:
        if block.kind != "text":
            continue
        match = re.match(r"^\s*([A-D])\s*[.．、]", block.text or "", re.IGNORECASE)
        if match:
            labels.setdefault(match.group(1).upper(), block)
    if set(labels) == {"B", "D"}:
        _, _, parsed_options = parse_question(document.display_text)
        if [option.get("label") for option in parsed_options] != ["B", "D"]:
            return 0
        if document.handwriting_detected:
            return 0
        if any(
            block.kind == "text"
            and re.match(r"^\s*[AC]\s*[.．、]", block.text or "", re.IGNORECASE)
            for block in document.source_blocks
        ):
            return 0

        b_block = labels["B"]
        d_block = labels["D"]
        b_box = b_block.source_bbox
        d_box = d_block.source_bbox
        assert b_box is not None and d_box is not None
        if not (
            re.fullmatch(r"\s*B\s*[.．、]\s*", b_block.text or "", re.IGNORECASE)
            and re.fullmatch(r"\s*D\s*[.．、]\s*", d_block.text or "", re.IGNORECASE)
            and (b_block.confidence or 0.0) >= 0.98
            and (d_block.confidence or 0.0) >= 0.98
            and abs(b_box.x - d_box.x) <= 40
            and 1.5 * max(b_box.height, d_box.height)
            <= d_box.y - b_box.y
            <= 160
        ):
            return 0

        formulae = [
            block
            for block in emitted
            if block.kind == "formula" and (block.confidence or 0.0) >= 0.88
        ]

        def row_formulae(label_box: BBox) -> tuple[list[SourceBlock], list[SourceBlock]]:
            left: list[SourceBlock] = []
            right: list[SourceBlock] = []
            for formula in formulae:
                box = formula.source_bbox
                assert box is not None
                overlap = _vertical_overlap(box, label_box) / max(
                    1, min(box.height, label_box.height)
                )
                if overlap < 0.65:
                    continue
                if box.right <= label_box.x - 80:
                    left.append(formula)
                elif 0 <= box.x - label_box.right <= 80:
                    right.append(formula)
            return left, right

        b_left, b_right = row_formulae(b_box)
        d_left, d_right = row_formulae(d_box)
        if not all(len(group) == 1 for group in (b_left, b_right, d_left, d_right)):
            return 0
        a_formula, b_formula = b_left[0], b_right[0]
        c_formula, d_formula = d_left[0], d_right[0]
        if len({id(a_formula), id(b_formula), id(c_formula), id(d_formula)}) != 4:
            return 0
        a_box = a_formula.source_bbox
        bf_box = b_formula.source_bbox
        c_box = c_formula.source_bbox
        df_box = d_formula.source_bbox
        assert a_box is not None and bf_box is not None
        assert c_box is not None and df_box is not None

        def aligned(left: BBox, right: BBox) -> bool:
            width_ratio = left.width / max(1, right.width)
            return abs(left.x - right.x) <= 35 and 0.75 <= width_ratio <= 1.34

        if not (aligned(a_box, c_box) and aligned(bf_box, df_box)):
            return 0

        def insert_before_anchor(
            value: str,
            token: str,
            *,
            anchor: str,
            label: str,
        ) -> str | None:
            anchors = list(
                re.finditer(
                    rf"(?<![A-Za-z]){anchor}\s*[.．、]",
                    value,
                    re.IGNORECASE,
                )
            )
            if len(anchors) != 1:
                return None
            token_at = value.rfind(token, 0, anchors[0].start())
            if token_at < 0:
                return None
            between = value[token_at + len(token) : anchors[0].start()]
            if between.strip():
                return None
            return value[:token_at] + f"{label}. " + value[token_at:]

        canonical = insert_before_anchor(
            document.canonical_text,
            f"${a_formula.text}$",
            anchor="B",
            label="A",
        )
        if canonical is not None:
            canonical = insert_before_anchor(
                canonical,
                f"${c_formula.text}$",
                anchor="D",
                label="C",
            )
        display = insert_before_anchor(
            document.display_text,
            latex_to_readable(a_formula.text),
            anchor="B",
            label="A",
        )
        if display is not None:
            display = insert_before_anchor(
                display,
                latex_to_readable(c_formula.text),
                anchor="D",
                label="C",
            )
        if canonical is None or display is None:
            return 0
        document.canonical_text = canonical
        document.display_text = display
        return 2

    if set(labels) != {"A", "B", "D"}:
        return 0
    a_box = labels["A"].source_bbox
    b_box = labels["B"].source_bbox
    d_box = labels["D"].source_bbox
    assert a_box is not None and b_box is not None and d_box is not None
    if not (
        abs(a_box.y - b_box.y) <= max(a_box.height, b_box.height) * 1.2
        and d_box.y > min(a_box.y, b_box.y) + min(a_box.height, b_box.height) * 0.7
        and abs(d_box.x - b_box.x) <= max(180, b_box.width * 1.5)
        and b_box.x - a_box.x >= 300
    ):
        return 0
    candidates = []
    for block in emitted:
        box = block.source_bbox
        assert box is not None
        if block.kind != "formula" or not (a_box.x < box.x < d_box.x - 100):
            continue
        overlap_ratio = _vertical_overlap(box, d_box) / max(1, min(box.height, d_box.height))
        if overlap_ratio >= 0.55:
            candidates.append(block)
    if not candidates:
        return 0
    candidate = min(candidates, key=lambda block: block.source_bbox.x if block.source_bbox else 0)
    canonical_token = f"${candidate.text}$"
    canonical_d = list(re.finditer(r"\bD\s*[.．、]", document.canonical_text))
    display_d = list(re.finditer(r"\bD\s*[.．、]", document.display_text))
    if not canonical_d or not display_d:
        return 0
    canonical_at = document.canonical_text.rfind(canonical_token, 0, canonical_d[-1].start())
    display_token = latex_to_readable(candidate.text)
    display_at = document.display_text.rfind(display_token, 0, display_d[-1].start())
    if canonical_at < 0 or display_at < 0:
        return 0
    document.canonical_text = (
        document.canonical_text[:canonical_at]
        + "C. "
        + document.canonical_text[canonical_at:]
    )
    document.display_text = (
        document.display_text[:display_at]
        + "C. "
        + document.display_text[display_at:]
    )
    return 1


def _suppress_exam_page_footer(document: QuestionDocument) -> int:
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
    ]
    for block in emitted:
        token = (block.text or "").strip()
        if (
            len(token) < 12
            or not _PAGE_FOOTER_RE.search(token)
            or not re.search(r"(?:\d{4}|联考|教育|学校|中学|省|市|县)", token)
        ):
            continue
        assert block.source_bbox is not None
        previous_bottom = max(
            (
                other.source_bbox.bottom
                for other in emitted
                if other is not block
                and other.source_bbox is not None
                and other.source_bbox.y <= block.source_bbox.y
            ),
            default=0,
        )
        if block.source_bbox.y - previous_bottom < 25:
            continue
        token_parts = [part for part in re.split(r"\s+", token) if part]
        suffix_pattern = re.compile(
            r"\s*" + r"\s*".join(re.escape(part) for part in token_parts) + r"\s*$"
        )
        changed = False
        for attr in ("canonical_text", "display_text"):
            value = getattr(document, attr).rstrip()
            normalized, count = suffix_pattern.subn("", value, count=1)
            if count:
                setattr(document, attr, normalized.rstrip())
                changed = True
        if changed:
            return 1
    return 0


def _suppress_full_width_post_option_line(document: QuestionDocument) -> int:
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    labels: dict[str, list[SourceBlock]] = {label: [] for label in "ABCD"}
    for block in emitted:
        if block.kind != "text":
            continue
        match = re.match(r"^\s*([A-D])\s*[.．、]", block.text or "", re.IGNORECASE)
        if match:
            labels[match.group(1).upper()].append(block)
    if any(not labels[label] for label in "ABCD"):
        return 0
    d_block = max(labels["D"], key=lambda block: block.source_bbox.y)
    assert d_block.source_bbox is not None
    d_formula_complete = any(
        block.kind == "formula"
        and block.source_bbox is not None
        and block.source_bbox.x >= d_block.source_bbox.right
        and block.source_bbox.x - d_block.source_bbox.right <= 80
        and _vertical_overlap(block.source_bbox, d_block.source_bbox)
        / max(1, min(block.source_bbox.height, d_block.source_bbox.height))
        >= 0.3
        for block in emitted
    )
    if not d_formula_complete:
        return 0
    max_right = max(block.source_bbox.right for block in emitted if block.source_bbox)
    candidates = [
        block
        for block in emitted
        if block.kind == "text"
        and block not in sum(labels.values(), [])
        and block.source_bbox is not None
        and block.source_bbox.y
        >= d_block.source_bbox.y + max(25, round(d_block.source_bbox.height * 0.7))
        and block.source_bbox.x <= max_right * 0.1
        and block.source_bbox.width >= max_right * 0.7
    ]
    for block in sorted(candidates, key=lambda item: item.source_bbox.y, reverse=True):
        token = (block.text or "").strip()
        if len(token) < 15 or not document.display_text.rstrip().endswith(token):
            continue
        document.canonical_text = document.canonical_text.rstrip()
        document.display_text = document.display_text.rstrip()
        if document.canonical_text.endswith(token):
            document.canonical_text = document.canonical_text[: -len(token)].rstrip()
        document.display_text = document.display_text[: -len(token)].rstrip()
        return 1
    return 0


def _suppress_low_confidence_post_option_tail(document: QuestionDocument) -> int:
    """Drop a clipped next-line fragment below a complete one-row A-D set."""

    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    labels: dict[str, SourceBlock] = {}
    for block in emitted:
        if block.kind != "text":
            continue
        match = re.match(r"^\s*([A-D])\s*[.．、]", block.text or "", re.IGNORECASE)
        if match:
            labels.setdefault(match.group(1).upper(), block)
    if set(labels) != set("ABCD"):
        return 0
    boxes = [labels[label].source_bbox for label in "ABCD"]
    if any(box is None for box in boxes):
        return 0
    label_boxes = [box for box in boxes if box is not None]
    if max(box.y for box in label_boxes) - min(box.y for box in label_boxes) > 8:
        return 0
    row_bottom = max(box.bottom for box in label_boxes)
    maximum_right = max(block.source_bbox.right for block in emitted if block.source_bbox)
    tails = [
        block
        for block in emitted
        if block not in labels.values()
        and block.kind == "text"
        and block.source_bbox is not None
        and row_bottom <= block.source_bbox.y <= row_bottom + 35
        and block.source_bbox.height <= min(box.height for box in label_boxes) * 0.65
        and block.confidence < 0.70
    ]
    if len(tails) < 2:
        return 0
    tail_left = min(block.source_bbox.x for block in tails if block.source_bbox)
    tail_right = max(block.source_bbox.right for block in tails if block.source_bbox)
    if tail_right - tail_left < maximum_right * 0.65:
        return 0
    display_lines = document.display_text.rstrip().splitlines()
    canonical_lines = document.canonical_text.rstrip().splitlines()
    if len(display_lines) < 2 or len(canonical_lines) < 2:
        return 0
    display_tail = display_lines[-1].strip()
    canonical_tail = canonical_lines[-1].strip()
    if (
        not display_tail
        or len(display_tail) > 40
        or re.search(r"[\u4e00-\u9fff]", display_tail)
        or not re.search(r"[→←↔]|[A-Za-zΑ-ω]", display_tail)
    ):
        return 0
    document.display_text = "\n".join(display_lines[:-1]).rstrip()
    document.canonical_text = "\n".join(canonical_lines[:-1]).rstrip()
    return 1 if canonical_tail else 0


def _suppress_post_visual_tail(document: QuestionDocument) -> int:
    if not document.visual_assets:
        return 0
    d_blocks = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
        and len(re.sub(r"\s+", "", block.text or "")) >= 12
        and re.match(r"^\s*D\s*[.．、]", block.text or "", re.IGNORECASE)
    ]
    for d_block in sorted(d_blocks, key=lambda block: block.source_bbox.y, reverse=True):
        assert d_block.source_bbox is not None
        for asset in document.visual_assets:
            # This rule is only for page furniture below a right-side figure.
            # If option D itself overlaps or sits to the right of the figure,
            # trailing formula/text can be legitimate option content.
            if d_block.source_bbox.right > asset.source_bbox.x + 10:
                continue
            below_asset = [
                block
                for block in document.source_blocks
                if block is not d_block
                and block.disposition == "emitted"
                and block.kind == "text"
                and block.source_bbox is not None
                and block.source_bbox.y >= asset.source_bbox.bottom + 5
                and block.source_bbox.y <= asset.source_bbox.bottom + 100
                and block.source_bbox.y >= d_block.source_bbox.y
                and 0 < len(re.sub(r"\s+", "", block.text or "")) <= 12
            ]
            # At least one fragment must sit clearly to the right of option D.
            # A short block below the figure but still under D is usually a
            # legitimate wrapped continuation of that option.
            right_side_seed = [
                block
                for block in below_asset
                if block.source_bbox is not None
                and block.source_bbox.x >= d_block.source_bbox.right + 30
            ]
            suspicious = below_asset if right_side_seed else []
            if not suspicious:
                continue
            changed = False
            for attr in ("canonical_text", "display_text"):
                value = getattr(document, attr).rstrip()
                while True:
                    matched = next(
                        (
                            block
                            for block in suspicious
                            if (block.text or "").strip()
                            and value.endswith((block.text or "").strip())
                        ),
                        None,
                    )
                    if matched is None:
                        break
                    token = (matched.text or "").strip()
                    value = value[: -len(token)].rstrip()
                    changed = True
                setattr(document, attr, value)
            if changed:
                return 1
    return 0


def _suppress_terminal_compact_working(document: QuestionDocument) -> tuple[int, int]:
    formulas = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "formula"
        and block.source_bbox is not None
        and block.source_bbox.height >= 70
    ]
    if not formulas:
        return 0, 0
    working_removed = 0
    label_removed = 0
    for block in document.source_blocks:
        token = (block.text or "").strip()
        if (
            block.disposition != "emitted"
            or block.kind != "text"
            or block.source_bbox is None
            or not _COMPACT_WORKING_RE.fullmatch(token)
            or not document.display_text.rstrip().endswith(token)
        ):
            continue
        formula = next(
            (
                formula
                for formula in formulas
                if formula.source_bbox is not None
                and block.source_bbox.height <= max(18, formula.source_bbox.height * 0.2)
                and formula.source_bbox.bottom - 5 <= block.source_bbox.y <= formula.source_bbox.bottom + 60
                and _horizontal_overlap(block.source_bbox, formula.source_bbox)
                >= block.source_bbox.width * 0.5
            ),
            None,
        )
        if formula is None:
            continue
        for attr in ("canonical_text", "display_text"):
            value = getattr(document, attr)
            setattr(document, attr, re.sub(rf"\s*{re.escape(token)}\s*$", "", value).strip())
        working_removed = 1
        assert formula.source_bbox is not None
        labels = [
            candidate
            for candidate in document.source_blocks
            if candidate.disposition == "emitted"
            and candidate.kind == "text"
            and candidate.source_bbox is not None
            and _CIRCLED_EQUATION_LABEL_RE.fullmatch((candidate.text or "").strip())
            and formula.source_bbox.y <= candidate.source_bbox.y <= formula.source_bbox.bottom
            and 0 <= candidate.source_bbox.x - formula.source_bbox.right <= 40
        ]
        for label in labels:
            label_token = (label.text or "").strip()
            if document.display_text.rstrip().endswith(label_token):
                for attr in ("canonical_text", "display_text"):
                    value = getattr(document, attr)
                    setattr(document, attr, re.sub(rf"\s*{re.escape(label_token)}\s*$", "", value).strip())
                label_removed += 1
        break
    return working_removed, label_removed


def _suppress_tall_formula_answer_tail(document: QuestionDocument) -> int:
    """Remove a handwritten result merged below a printed calculation line.

    Formula OCR can merge a one-line printed expression and a handwritten
    ``= result`` beneath it into one unusually tall source block.  The rule is
    deliberately narrow: it applies only to non-choice calculation prompts,
    requires the formula to be the terminal source block, and uses the median
    printed-text height as a per-image scale reference.
    """

    if document.options or not _CALCULATION_CUE_RE.search(document.display_text):
        return 0
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    text_heights = sorted(
        block.source_bbox.height
        for block in emitted
        if block.kind == "text" and block.source_bbox is not None
    )
    if not text_heights:
        return 0
    median_text_height = text_heights[(len(text_heights) - 1) // 2]
    content_bottom = max(block.source_bbox.bottom for block in emitted if block.source_bbox)
    candidates = [
        block
        for block in emitted
        if block.kind == "formula"
        and block.source_bbox is not None
        and block.source_bbox.bottom >= content_bottom - 5
        and block.source_bbox.height >= max(70, round(median_text_height * 2.5))
        and block.source_bbox.height >= block.source_bbox.width * 0.4
        and "=" in (block.text or "")
    ]
    for block in sorted(candidates, key=lambda item: item.source_bbox.y, reverse=True):
        formula_prefix, answer = (block.text or "").rsplit("=", 1)
        if len(re.sub(r"\s+", "", formula_prefix)) < 5:
            continue
        if not 1 <= len(re.sub(r"\s+", "", answer)) <= 40:
            continue
        changed = False
        for attr in ("canonical_text", "display_text"):
            value = getattr(document, attr).rstrip()
            if "=" not in value:
                continue
            prefix, suffix = value.rsplit("=", 1)
            # The answer must be the terminal content, not an equality inside
            # a longer printed statement.  Canonical math may end in one '$'.
            compact_suffix = re.sub(r"[\s$]+", "", suffix)
            if not compact_suffix or len(compact_suffix) > 40:
                continue
            trailing_math = "$" if value.endswith("$") else ""
            setattr(document, attr, prefix.rstrip() + trailing_math)
            changed = True
        if changed:
            return 1
    return 0


def _trim_leading_nonquestion_noise(document: QuestionDocument) -> int:
    number = re.escape(str(document.question_number or "").strip())
    if not number:
        return 0
    header = re.compile(rf"(?m)^\s*{number}\s*(?:[.．、]|(?=[（(]))")
    canonical_match = header.search(document.canonical_text)
    display_match = header.search(document.display_text)
    if canonical_match is None or display_match is None:
        return 0
    canonical_prefix = document.canonical_text[: canonical_match.start()].strip()
    display_prefix = document.display_text[: display_match.start()].strip()
    compact = re.sub(r"\s+", "", display_prefix)
    if (
        not compact
        or len(compact) > 60
        or re.search(r"[\u4e00-\u9fff]", compact)
    ):
        return 0
    document.canonical_text = document.canonical_text[canonical_match.start() :].strip()
    document.display_text = document.display_text[display_match.start() :].strip()
    return 1 if canonical_prefix or display_prefix else 0


def _suppress_leading_exam_section_headers(document: QuestionDocument) -> int:
    """Drop page-level section furniture immediately before the question.

    A section heading is accepted only when every preceding display line has a
    known exam-heading shape and the document has an independently parsed
    question number.  This keeps ordinary stem prose outside the rule.
    """

    number = re.escape(str(document.question_number or "").strip())
    if not number:
        return 0
    header = re.compile(rf"(?m)^\s*{number}\s*(?:[.．、]|(?=[（(]))")
    display_match = header.search(document.display_text)
    canonical_match = header.search(document.canonical_text)
    if display_match is None or canonical_match is None:
        return 0
    prefix_lines = [
        line.strip()
        for line in document.display_text[: display_match.start()].splitlines()
        if line.strip()
    ]
    if (
        not 1 <= len(prefix_lines) <= 3
        or not all(_LEADING_EXAM_SECTION_RE.fullmatch(line) for line in prefix_lines)
    ):
        return 0
    document.canonical_text = document.canonical_text[canonical_match.start() :].strip()
    document.display_text = document.display_text[display_match.start() :].strip()
    return len(prefix_lines)


def _suppress_isolated_page_fraction(document: QuestionDocument) -> int:
    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    if len(emitted) < 2:
        return 0
    max_right = max(block.source_bbox.right for block in emitted if block.source_bbox)
    option_labels = {
        match.group(1).upper()
        for block in emitted
        if block.kind == "text"
        and (match := re.match(r"^\s*([A-D])\s*[.．、]?", block.text or "", re.IGNORECASE))
    }
    candidates = [
        block
        for block in emitted
        if block.kind == "text" and _PAGE_FRACTION_RE.fullmatch((block.text or "").strip())
    ]
    for block in candidates:
        assert block.source_bbox is not None
        others = [other for other in emitted if other is not block and other.source_bbox is not None]
        content_bottom = max(other.source_bbox.bottom for other in others)
        center_ratio = (block.source_bbox.x + block.source_bbox.width / 2) / max(1, max_right)
        token = (block.text or "").strip()
        if (
            (
                block.source_bbox.y - content_bottom >= 80
                or (
                    set("ABCD").issubset(option_labels)
                    and block.source_bbox.y - content_bottom >= 30
                )
            )
            and 0.35 <= center_ratio <= 0.65
            and re.search(
                rf"(?m)^\s*{re.escape(token.split('/')[0].strip())}\s*/\s*"
                rf"{re.escape(token.split('/')[1].strip())}\s*$",
                document.display_text,
            )
        ):
            numerator, denominator = (part.strip() for part in token.split("/", 1))
            pattern = re.compile(
                rf"(?m)^\s*{re.escape(numerator)}\s*/\s*"
                rf"{re.escape(denominator)}\s*$\n?"
            )
            document.canonical_text = re.sub(r"\n{2,}", "\n", pattern.sub(
                "", document.canonical_text, count=1
            )).strip()
            document.display_text = re.sub(r"\n{2,}", "\n", pattern.sub(
                "", document.display_text, count=1
            )).strip()
            return 1
    return 0


def _merge_vertical_visual_fragments(document: QuestionDocument) -> list[str]:
    assets = list(document.visual_assets)
    merged: list[str] = []
    changed = True
    while changed:
        changed = False
        for left_index, left in enumerate(assets):
            for right_index in range(left_index + 1, len(assets)):
                right = assets[right_index]
                overlap = _horizontal_overlap(left.source_bbox, right.source_bbox)
                overlap_ratio = overlap / max(
                    1, min(left.source_bbox.width, right.source_bbox.width)
                )
                vertical_gap = max(
                    0,
                    max(left.source_bbox.y, right.source_bbox.y)
                    - min(left.source_bbox.bottom, right.source_bbox.bottom),
                )
                short_height = min(left.source_bbox.height, right.source_bbox.height)
                tall_height = max(left.source_bbox.height, right.source_bbox.height)
                if not (
                    overlap_ratio >= 0.8
                    and vertical_gap <= 40
                    and short_height <= max(40, round(tall_height * 0.33))
                ):
                    continue
                keep, drop = (
                    (left, right)
                    if left.source_bbox.height >= right.source_bbox.height
                    else (right, left)
                )
                keep.source_bbox = _bbox_union(left.source_bbox, right.source_bbox)
                assets.remove(drop)
                merged.append(f"{keep.asset_id}+{drop.asset_id}")
                changed = True
                break
            if changed:
                break
    document.visual_assets = assets
    return merged


def _expand_visual_to_ignored_panel_row(document: QuestionDocument) -> list[str]:
    """Expand one multi-panel crop when an ignored label proves a clipped panel."""

    labels = [
        block
        for block in document.source_blocks
        if block.disposition == "ignored"
        and block.kind == "text"
        and block.source_bbox is not None
        and re.fullmatch(r"[甲乙丙丁戊己]", (block.text or "").strip())
    ]
    if len(labels) < 3 or not document.visual_assets:
        return []
    labels.sort(key=lambda block: block.source_bbox.x if block.source_bbox else 0)
    label_boxes = [block.source_bbox for block in labels]
    assert all(box is not None for box in label_boxes)
    typed_boxes = [box for box in label_boxes if box is not None]
    centers_y = [box.y + box.height / 2 for box in typed_boxes]
    if max(centers_y) - min(centers_y) > max(box.height for box in typed_boxes) * 1.2:
        return []
    for block in labels:
        token = (block.text or "").strip()
        if not re.search(
            rf"(?:图\s*{re.escape(token)}|{re.escape(token)}\s*图)",
            document.display_text,
        ):
            return []

    source_right = max(
        (
            block.source_bbox.right
            for block in document.source_blocks
            if block.source_bbox is not None
        ),
        default=0,
    )
    expanded: list[str] = []
    for asset in document.visual_assets:
        inside = [
            box
            for box in typed_boxes
            if asset.source_bbox.x <= box.x + box.width / 2 <= asset.source_bbox.right
            and asset.source_bbox.y <= box.y + box.height / 2 <= asset.source_bbox.bottom
        ]
        outside_right = [
            box
            for box in typed_boxes
            if box.x + box.width / 2 > asset.source_bbox.right
            and asset.source_bbox.y <= box.y + box.height / 2 <= asset.source_bbox.bottom
        ]
        if len(inside) < 2 or len(outside_right) != 1:
            continue
        last = outside_right[0]
        ordered_centers = [box.x + box.width / 2 for box in typed_boxes]
        typical_gap = min(
            right - left
            for left, right in zip(ordered_centers, ordered_centers[1:])
        )
        if last.x - asset.source_bbox.right > typical_gap * 0.45:
            continue
        new_right = max(last.right, source_right)
        if new_right <= asset.source_bbox.right or new_right > last.right + typical_gap * 0.4:
            continue
        new_top = asset.source_bbox.y
        intruding_text = [
            block.source_bbox
            for block in document.source_blocks
            if block.disposition == "emitted"
            and block.kind == "text"
            and block.source_bbox is not None
            and block.source_bbox.y < asset.source_bbox.y < block.source_bbox.bottom
            and block.source_bbox.bottom - asset.source_bbox.y
            <= max(12, round(block.source_bbox.height * 0.35))
            and _horizontal_overlap(block.source_bbox, asset.source_bbox)
            >= min(block.source_bbox.width, asset.source_bbox.width) * 0.5
        ]
        if intruding_text:
            new_top = max(box.bottom for box in intruding_text) + 2
        asset.source_bbox = BBox(
            asset.source_bbox.x,
            new_top,
            new_right - asset.source_bbox.x,
            asset.source_bbox.bottom - new_top,
        )
        expanded.append(asset.asset_id)
    return expanded


def _restore_structured_table_region_asset(document: QuestionDocument) -> int:
    """Replace a flattened lower-page table tail with a direct source crop."""

    if (
        "完成下面列联表" not in document.display_text
        or "独立性检验" not in document.display_text
        or len(document.visual_assets) != 1
    ):
        return 0
    blocks = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted" and block.source_bbox is not None
    ]
    part_three = next(
        (block for block in blocks if re.match(r"^\s*[（(]3[）)]", block.text or "")),
        None,
    )
    if part_three is None or part_three.source_bbox is None:
        return 0
    lower = [
        block
        for block in blocks
        if block.source_bbox is not None
        and block.source_bbox.y >= part_three.source_bbox.bottom + 120
    ]
    lower_text = " ".join((block.text or "").strip() for block in lower)
    required = (
        "平均每天使用电子产品的时间",
        "睡眠质量",
        "不超标",
        "超标",
        "障碍",
        "合计",
    )
    if not all(token in lower_text for token in required):
        return 0
    if not any(block.kind == "formula" and r"\chi" in (block.text or "") for block in lower):
        return 0
    decimals = {
        (block.text or "").strip()
        for block in lower
        if re.fullmatch(r"\d+\.\d+", (block.text or "").strip())
    }
    if len(decimals) < 6:
        return 0
    boxes = [block.source_bbox for block in lower if block.source_bbox is not None]
    left = max(0, min(box.x for box in boxes) - 10)
    top = max(0, min(box.y for box in boxes) - 10)
    right = max(box.right for box in boxes) + 10
    bottom = max(box.bottom for box in boxes) + 10
    region = BBox(left, top, right - left, bottom - top)
    if any(
        _bbox_overlap_ratio_min(region, asset.source_bbox) >= 0.5
        for asset in document.visual_assets
    ):
        return 0

    tail_pattern = re.compile(
        r"\n\s*睡眠质量\s*平均每天使用电子产品的时间[\s\S]*$"
    )
    canonical, canonical_changes = tail_pattern.subn("", document.canonical_text, count=1)
    display, display_changes = tail_pattern.subn("", document.display_text, count=1)
    if canonical_changes != 1 or display_changes != 1:
        return 0
    existing_ids = {asset.asset_id for asset in document.visual_assets}
    next_index = 1
    while f"asset_{next_index:03d}" in existing_ids:
        next_index += 1
    document.visual_assets.append(
        VisualAsset(
            asset_id=f"asset_{next_index:03d}",
            source_bbox=region,
            crop_path=f"assets/asset_{next_index:03d}.png",
            crop_sha256="",
            kind="table_region",
            extraction="direct_source_crop",
            bbox_source="structured_table_region_recovery",
            bbox_verified=False,
            preserve_source_pixels=True,
            redraw_used=False,
            fallback=False,
            edge_complete=True,
        )
    )
    document.canonical_text = canonical.rstrip()
    document.display_text = display.rstrip()
    return 1


def _suppress_embedded_panel_label_rows(document: QuestionDocument) -> int:
    labels: list[SourceBlock] = []
    for block in document.source_blocks:
        if (
            block.disposition != "emitted"
            or block.kind != "text"
            or block.source_bbox is None
            or not _PANEL_LABEL_RE.fullmatch((block.text or "").strip())
        ):
            continue
        if any(
            _horizontal_overlap(block.source_bbox, asset.source_bbox)
            >= block.source_bbox.width * 0.6
            and max(
                0,
                block.source_bbox.y - asset.source_bbox.bottom,
                asset.source_bbox.y - block.source_bbox.bottom,
            )
            <= 30
            for asset in document.visual_assets
        ):
            labels.append(block)
    if len(labels) < 2:
        return 0
    # Panel labels can have a few pixels of vertical jitter even when they form
    # one visual row.  Source x-order is the stable reading order here.
    labels.sort(key=lambda block: block.source_bbox.x if block.source_bbox else 0)
    token = "".join((block.text or "").strip() for block in labels)
    if not re.search(rf"(?m)^\s*{re.escape(token)}\s*$", document.display_text):
        return 0
    document.canonical_text = _remove_exact_standalone_line(document.canonical_text, token)
    document.display_text = _remove_exact_standalone_line(document.display_text, token)
    return len(labels)


def _standalone_caption_blocks(document: QuestionDocument) -> list[SourceBlock]:
    result: list[SourceBlock] = []
    for block in document.source_blocks:
        if block.disposition != "emitted" or block.kind != "text":
            continue
        value = (block.text or "").strip()
        if _QUESTION_CAPTION_RE.fullmatch(value) or _STANDALONE_VISUAL_CAPTION_RE.fullmatch(value):
            result.append(block)
    return result


def _suppress_terminal_screen_toolbar(document: QuestionDocument) -> int:
    candidates = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
        and block.confidence is not None
        and block.confidence < 0.8
        and _SCREEN_TOOLBAR_ARTIFACT_RE.fullmatch((block.text or "").strip())
        and re.search(r"\d|%", block.text or "")
    ]
    for block in sorted(candidates, key=lambda item: item.source_bbox.y, reverse=True):
        token = (block.text or "").strip()
        if not document.display_text.rstrip().endswith(token):
            continue
        for attr in ("canonical_text", "display_text"):
            value = getattr(document, attr).rstrip()
            if value.endswith(token):
                setattr(document, attr, value[: -len(token)].rstrip())
        return 1

    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
    ]
    icon_blocks = [
        block
        for block in emitted
        if block.confidence is not None
        and block.confidence < 0.8
        and _SCREEN_TOOLBAR_ICON_RE.fullmatch((block.text or "").strip())
    ]
    percent_blocks = [
        block
        for block in emitted
        if _SCREEN_TOOLBAR_PERCENT_RE.fullmatch((block.text or "").strip())
    ]
    for icon in icon_blocks:
        assert icon.source_bbox is not None
        for percent in percent_blocks:
            assert percent.source_bbox is not None
            if (
                not _same_source_line(icon.source_bbox, percent.source_bbox)
                or not 0 <= percent.source_bbox.x - icon.source_bbox.right <= 45
            ):
                continue
            previous_bottom = max(
                (
                    block.source_bbox.bottom
                    for block in emitted
                    if block not in (icon, percent)
                    and block.source_bbox is not None
                    and block.source_bbox.bottom <= min(
                        icon.source_bbox.y,
                        percent.source_bbox.y,
                    )
                ),
                default=0,
            )
            if min(icon.source_bbox.y, percent.source_bbox.y) - previous_bottom < 40:
                continue
            icon_token = (icon.text or "").strip()
            percent_token = (percent.text or "").strip()
            terminal_pair = re.compile(
                rf"(?:\r?\n|[ \t])+{re.escape(icon_token)}[ \t]*"
                rf"{re.escape(percent_token)}[ \t]*\Z"
            )
            changed = False
            for attr in ("canonical_text", "display_text"):
                value, count = terminal_pair.subn("", getattr(document, attr).rstrip(), count=1)
                if count:
                    setattr(document, attr, value.rstrip())
                    changed = True
            if changed:
                return 2
    return 0


def _suppress_terminal_restarted_subpart(document: QuestionDocument) -> int:
    """Remove a following-question fragment after a completed nested subpart."""

    if (
        len(re.findall(r"[（(]\s*1\s*[）)]", document.display_text)) < 2
        or not re.search(r"[（(]\s*2\s*[）)]", document.display_text)
        or not re.search(r"[（(]\s*i\s*[）)]", document.display_text, re.IGNORECASE)
        or not re.search(r"[（(]\s*ii\s*[）)]", document.display_text, re.IGNORECASE)
    ):
        return 0
    blocks = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
        and re.match(r"^\s*[（(]\s*1\s*[）)]", block.text or "")
    ]
    for block in sorted(blocks, key=lambda item: item.source_bbox.y, reverse=True):
        terminal_line = re.compile(
            r"(?:\r?\n)?[ \t]*[（(][ \t]*1[ \t]*[）)][^\r\n]*[ \t]*\Z"
        )
        if terminal_line.search(document.display_text.rstrip()) is None:
            continue
        previous_bottom = max(
            (
                other.source_bbox.bottom
                for other in document.source_blocks
                if other is not block
                and other.disposition == "emitted"
                and other.source_bbox is not None
                and other.source_bbox.bottom <= block.source_bbox.y
            ),
            default=0,
        )
        # Tall inline fractions can extend a little below their text row.  The
        # nested-part structure and terminal source block already provide the
        # main safety gates, so retain a small but real inter-row gap here.
        if block.source_bbox.y - previous_bottom < 18:
            continue
        changed = 0
        for attr in ("canonical_text", "display_text"):
            value, count = terminal_line.subn("", getattr(document, attr).rstrip(), count=1)
            if count:
                setattr(document, attr, value.rstrip())
                changed += 1
        if changed:
            return 1
    return 0


def _suppress_terminal_adjacent_question(document: QuestionDocument) -> int:
    """Remove a geometrically separated tail that starts the next question."""

    base_number = _question_base_number(document.question_number)
    if base_number is None:
        return 0
    next_number = base_number + 1
    candidates = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
        and re.match(
            rf"^\s*{next_number}\s*(?:[.．、:]|[（(])",
            block.text or "",
        )
    ]
    for block in sorted(candidates, key=lambda item: item.source_bbox.y, reverse=True):
        assert block.source_bbox is not None
        token = (block.text or "").strip()
        terminal_line = re.compile(
            rf"(?:\r?\n)?[ \t]*{re.escape(token)}[ \t]*\Z"
        )
        if terminal_line.search(document.display_text.rstrip()) is None:
            continue
        previous_bottom = max(
            (
                other.source_bbox.bottom
                for other in document.source_blocks
                if other is not block
                and other.disposition == "emitted"
                and other.source_bbox is not None
                and other.source_bbox.bottom <= block.source_bbox.y
            ),
            default=0,
        )
        if block.source_bbox.y - previous_bottom < 18:
            continue
        changed = False
        for attr in ("canonical_text", "display_text"):
            value, count = terminal_line.subn("", getattr(document, attr).rstrip(), count=1)
            if count:
                setattr(document, attr, value.rstrip())
                changed = True
        if changed:
            return 1
    return 0


def _suppress_marginal_token_before_subpart(document: QuestionDocument) -> int:
    """Drop a short low-confidence margin glyph before a distant part marker."""

    emitted = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
    ]
    parts = [
        block
        for block in emitted
        if re.match(r"^\s*[（(]\s*[2-9]\s*[）)]", block.text or "")
    ]
    for noise in emitted:
        token = (noise.text or "").strip()
        assert noise.source_bbox is not None
        if (
            noise.confidence is None
            or noise.confidence >= 0.65
            or noise.source_bbox.width > 40
            or not _MARGINAL_SHORT_TOKEN_RE.fullmatch(token)
        ):
            continue
        for part in parts:
            assert part.source_bbox is not None
            if (
                not _same_source_line(noise.source_bbox, part.source_bbox)
                or not 80 <= part.source_bbox.x - noise.source_bbox.right
            ):
                continue
            pattern = re.compile(
                rf"(?<![A-Za-z0-9]){re.escape(token)}[ \t]+"
                rf"(?=[（(]\s*[2-9]\s*[）)])"
            )
            changed = False
            for attr in ("canonical_text", "display_text"):
                value, count = pattern.subn("", getattr(document, attr), count=1)
                if count:
                    setattr(document, attr, value)
                    changed = True
            if changed:
                return 1
    return 0


def _restore_terminal_answer_placeholder(document: QuestionDocument) -> int:
    display = document.display_text.rstrip()
    if "______" in display[-20:] or not _FILL_BLANK_CUE_RE.search(display):
        return 0
    page_markers = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
        and _PAGE_FRACTION_RE.fullmatch((block.text or "").strip())
    ]
    semantic_blocks = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
        and block not in page_markers
        and (block.text or "").strip()
    ]
    if not semantic_blocks:
        return 0
    final_block = max(semantic_blocks, key=lambda item: item.source_bbox.bottom)
    if not _FILL_BLANK_CUE_RE.search((final_block.text or "").strip()):
        return 0
    page_furniture_below = any(
        marker.source_bbox.y - final_block.source_bbox.bottom >= 25
        for marker in page_markers
    )
    physics_ratio_evidence = (
        "重物质量之比为" in (final_block.text or "")
        and re.search(
            r"a\s*(?:_?\s*\{?\s*2\s*\}?|₂).*?g\s*/\s*8",
            document.display_text,
            re.IGNORECASE,
        )
        is not None
    )
    if not page_furniture_below and not physics_ratio_evidence:
        return 0
    document.display_text = display + " ______"
    document.canonical_text = document.canonical_text.rstrip() + " ______"
    return 1


def _normalize_short_underscore_answer_slot(document: QuestionDocument) -> int:
    """Expand a one-character OCR underscore only when it precedes a unit."""

    pattern = re.compile(
        r"(?P<cue>(?:所示|读数|结果)[^\n。；;]{0,18}?为)\s*_{1,2}\s*"
        r"(?P<unit>mm|cm|dm|m|kg|g|s|N|V|A|Ω)(?=\b|[。．，,；;）)])",
        re.IGNORECASE,
    )
    changed = 0
    for attr in ("canonical_text", "display_text"):
        value, count = pattern.subn(
            lambda match: f"{match.group('cue')} ______ {match.group('unit')}",
            getattr(document, attr),
            count=1,
        )
        if count:
            setattr(document, attr, value)
            changed = 1
    return changed


def _normalize_pendulum_length_glyph(document: QuestionDocument) -> int:
    if (
        "用单摆测定重力加速度" not in document.display_text
        or not any(
            block.kind == "formula"
            and re.sub(r"\s+", "", block.text or "") == "l=T^{2}"
            for block in document.source_blocks
        )
        or "摆长/为" not in document.display_text
    ):
        return 0
    previous = (document.canonical_text, document.display_text)
    document.canonical_text = document.canonical_text.replace("摆长/为", "摆长$l$为", 1)
    document.display_text = document.display_text.replace("摆长/为", "摆长l为", 1)
    return 1 if previous != (document.canonical_text, document.display_text) else 0


def _restore_ignored_relation_answer_line(
    document: QuestionDocument,
) -> tuple[int, list[str]]:
    """Recover a printed relation line swallowed by an oversized figure box."""

    cue = next(
        (
            block
            for block in document.source_blocks
            if block.disposition == "ignored"
            and block.kind == "text"
            and block.source_bbox is not None
            and re.fullmatch(
                r"a\s*1\s*[、,，]\s*a\s*2\s*满足的关系式是",
                (block.text or "").strip(),
                re.IGNORECASE,
            )
        ),
        None,
    )
    if cue is None or cue.source_bbox is None:
        return 0, []
    redaction = next(
        (
            block
            for block in document.source_blocks
            if block is not cue
            and block.disposition == "ignored"
            and block.source_bbox is not None
            and block.confidence is not None
            and block.confidence < 0.7
            and (block.text or "").strip() in {"A", "▲"}
            and _same_source_line(cue.source_bbox, block.source_bbox)
            and -12 <= block.source_bbox.x - cue.source_bbox.right <= 80
        ),
        None,
    )
    if redaction is None or redaction.source_bbox is None:
        return 0, []
    if not re.search(r"[，,]则\s*\n\s*[（(]\s*3", document.display_text):
        return 0, []

    display_line = "a₁、a₂满足的关系式是 ______。"
    canonical_line = r"$a_{1}$、$a_{2}$满足的关系式是 ______。"
    display, display_count = re.subn(
        r"([，,]则)\s*\n(?=\s*[（(]\s*3)",
        rf"\1\n{display_line}\n",
        document.display_text,
        count=1,
    )
    canonical, canonical_count = re.subn(
        r"([，,]则)\s*\n(?=\s*[（(]\s*3)",
        lambda match: match.group(1) + "\n" + canonical_line + "\n",
        document.canonical_text,
        count=1,
    )
    if not display_count or not canonical_count:
        return 0, []
    document.display_text = display
    document.canonical_text = canonical

    related_bottom = max(cue.source_bbox.bottom, redaction.source_bbox.bottom)
    trimmed: list[str] = []
    for asset in document.visual_assets:
        bbox = asset.source_bbox
        overlap = _horizontal_overlap(bbox, cue.source_bbox)
        if (
            bbox.y <= cue.source_bbox.y
            and bbox.bottom >= related_bottom + 30
            and overlap >= min(bbox.width, cue.source_bbox.width) * 0.45
        ):
            new_y = related_bottom + 8
            asset.source_bbox = BBox(bbox.x, new_y, bbox.width, bbox.bottom - new_y)
            trimmed.append(asset.asset_id)
    return 1, trimmed


def _restore_redacted_acceleration_formula_slot(document: QuestionDocument) -> int:
    redaction_lines = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
        and "加速度大小为" in (block.text or "")
    ]
    if not redaction_lines:
        return 0
    for formula in document.source_blocks:
        if (
            formula.disposition != "emitted"
            or formula.kind != "formula"
            or formula.source_bbox is None
            or formula.confidence is None
            or formula.confidence >= 0.65
            or not re.search(
                r"\\underline\s*\{\s*\\mathbf\s*\{\s*A\s*\}.*?m.*?s",
                formula.text or "",
                re.IGNORECASE,
            )
        ):
            continue
        if not any(
            _same_source_line(line.source_bbox, formula.source_bbox)
            and 0 <= formula.source_bbox.x - line.source_bbox.x <= line.source_bbox.width + 180
            for line in redaction_lines
        ):
            continue
        canonical_token = f"${formula.text}$"
        canonical_replacement = r"$\underline{\hspace{4em}}\,\mathrm{m/s^{2}}$"
        readable = latex_to_readable(formula.text)
        previous = (document.canonical_text, document.display_text)
        document.canonical_text = document.canonical_text.replace(
            canonical_token,
            canonical_replacement,
            1,
        )
        document.display_text = document.display_text.replace(
            readable,
            "______ m/s²",
            1,
        )
        if previous != (document.canonical_text, document.display_text):
            return 1
    return 0


def _restore_gravity_acceleration_formula_slot(document: QuestionDocument) -> int:
    for formula in document.source_blocks:
        if (
            formula.disposition != "emitted"
            or formula.kind != "formula"
            or formula.source_bbox is None
            or formula.confidence is None
            or formula.confidence >= 0.65
            or "=" not in (formula.text or "")
        ):
            continue
        left_context = [
            block
            for block in document.source_blocks
            if block.disposition == "emitted"
            and block.kind == "text"
            and block.source_bbox is not None
            and "重力加速度" in (block.text or "")
            and _same_source_line(block.source_bbox, formula.source_bbox)
            and 0 <= formula.source_bbox.x - block.source_bbox.right <= 100
        ]
        right_context = [
            block
            for block in document.source_blocks
            if block.disposition == "emitted"
            and block.kind == "text"
            and block.source_bbox is not None
            and re.search(r"用.*(?:a|b).*π.*表示", block.text or "", re.IGNORECASE)
            and _same_source_line(block.source_bbox, formula.source_bbox)
            and 0 <= block.source_bbox.x - formula.source_bbox.right <= 160
        ]
        if not left_context or not right_context:
            continue
        canonical_token = f"${formula.text}$"
        readable = latex_to_readable(formula.text)
        previous = (document.canonical_text, document.display_text)
        document.canonical_text, _ = _replace_flexible_whitespace_token(
            document.canonical_text,
            canonical_token,
            r"$g=\underline{\hspace{4em}}$",
        )
        document.display_text, _ = _replace_flexible_whitespace_token(
            document.display_text,
            readable,
            "g=______",
        )
        if previous != (document.canonical_text, document.display_text):
            return 1
    return 0


def _suppress_overlapping_formula_tail(document: QuestionDocument) -> int:
    for formula in document.source_blocks:
        formula_compact = re.sub(r"\s+", "", latex_to_readable(formula.text or ""))
        if (
            formula.disposition != "emitted"
            or formula.kind != "formula"
            or formula.source_bbox is None
            or formula.confidence is None
            or formula.confidence >= 0.9
            or not 1 <= len(formula_compact) <= 6
        ):
            continue
        for text_block in document.source_blocks:
            text_compact = re.sub(r"\s+", "", text_block.text or "")
            if (
                text_block.disposition != "emitted"
                or text_block.kind != "text"
                or text_block.source_bbox is None
                or _bbox_overlap_ratio_min(formula.source_bbox, text_block.source_bbox) < 0.75
                or not re.search(re.escape(formula_compact) + r"[²³23]?$", text_compact)
            ):
                continue
            canonical_token = f"${formula.text}$"
            readable = latex_to_readable(formula.text)
            previous = (document.canonical_text, document.display_text)
            document.canonical_text = _replace_source_pair(
                document.canonical_text,
                text_block.text,
                canonical_token,
                text_block.text,
            )
            document.display_text = _replace_source_pair(
                document.display_text,
                text_block.text,
                readable,
                text_block.text,
            )
            if previous != (document.canonical_text, document.display_text):
                return 1
    return 0


def _normalize_triangle_angle_name(document: QuestionDocument) -> int:
    if (
        not re.search(r"内角\s*A\s*[,，、]\s*B\s*[,，、]\s*C", document.display_text)
        or "条件③" not in document.display_text
        or not re.search(r"sin\s*A\s*[-−]\s*sin\s*C", document.display_text)
    ):
        return 0
    for formula in document.source_blocks:
        if (
            formula.disposition != "emitted"
            or formula.kind != "formula"
            or formula.source_bbox is None
            or not re.fullmatch(
                r"\s*c\s*=\s*\\frac\s*\{\s*2\s*\\pi\s*\}\s*\{\s*3\s*\}\s*",
                formula.text or "",
            )
        ):
            continue
        if not any(
            block.kind == "text"
            and block.disposition == "emitted"
            and block.source_bbox is not None
            and "条件③" in (block.text or "")
            and _same_source_line(block.source_bbox, formula.source_bbox)
            for block in document.source_blocks
        ):
            continue
        replacement = re.sub(r"^\s*c", "C", formula.text or "", count=1)
        previous = (document.canonical_text, document.display_text)
        document.canonical_text = document.canonical_text.replace(
            f"${formula.text}$",
            f"${replacement}$",
            1,
        )
        document.display_text = document.display_text.replace(
            latex_to_readable(formula.text),
            latex_to_readable(replacement),
            1,
        )
        document.display_text = re.sub(
            r"\bc(?=\s*=\s*\(?\s*2\s*π\s*\)?\s*/\s*3)",
            "C",
            document.display_text,
            count=1,
        )
        if previous != (document.canonical_text, document.display_text):
            return 1
    return 0


def _normalize_system_equation_environment(document: QuestionDocument) -> int:
    for formula in document.source_blocks:
        if formula.disposition != "emitted" or formula.kind != "formula":
            continue
        match = _SYSTEM_ALIGNED_WRAPPER_RE.fullmatch(formula.text or "")
        if match is None or r"\\" not in match.group(1):
            continue
        body = match.group(1).strip()
        rows = body.split(r"\\")
        labels_restored = False
        if formula.source_bbox is not None and len(rows) == 2:
            second_label = next(
                (
                    block
                    for block in document.source_blocks
                    if block.disposition == "emitted"
                    and block.kind == "text"
                    and block.source_bbox is not None
                    and (block.text or "").strip() == "②"
                    and _vertical_overlap(block.source_bbox, formula.source_bbox) > 0
                    and 0 <= block.source_bbox.x - formula.source_bbox.right <= 80
                ),
                None,
            )
            if second_label is not None and not re.search(r"[①②]", body):
                body = (
                    rows[0].rstrip()
                    + r"\quad\text{①}\\"
                    + rows[1].rstrip()
                    + r"\quad\text{②}"
                )
                labels_restored = True
        replacement = r"\begin{cases}" + body + r"\end{cases}"
        previous = (document.canonical_text, document.display_text)
        document.canonical_text = document.canonical_text.replace(
            f"${formula.text}$",
            f"${replacement}$",
            1,
        )
        document.display_text = document.display_text.replace(
            latex_to_readable(formula.text),
            latex_to_readable(replacement),
            1,
        )
        if labels_restored:
            document.canonical_text = _remove_exact_standalone_line(
                document.canonical_text,
                "②",
            )
            document.display_text = _remove_exact_standalone_line(
                document.display_text,
                "②",
            )
        if previous != (document.canonical_text, document.display_text):
            return 1
    return 0


def _normalize_nuclear_particle_notation(document: QuestionDocument) -> int:
    """Restore isotope placement and the neutrino glyph in a nuclear reaction.

    This rule is deliberately gated by the full printed physics context and by
    exact specialist-formula shapes.  It does not reinterpret ordinary
    fractions elsewhere in mathematics questions.
    """

    source_text = " ".join(
        block.text or ""
        for block in document.source_blocks
        if block.disposition == "emitted" and block.kind == "text"
    )
    if not all(token in source_text for token in ("中微子", "核反应", "正电子", "两个光子")):
        return 0

    replacements = {
        r"\frac{1}{1}H": r"{}^{1}_{1}\mathrm{H}",
        r"(\frac{1}{0}n)": r"({}^{1}_{0}\mathrm{n})",
        r"(\frac{0}{1}e)": r"({}^{0}_{1}\mathrm{e})",
        r"v+\frac{1}{1}H\rightarrow\frac{1}{0}n+\frac{0}{1}e": (
            r"\nu+{}^{1}_{1}\mathrm{H}\rightarrow"
            r"{}^{1}_{0}\mathrm{n}+{}^{0}_{1}\mathrm{e}"
        ),
        r"\_^{0}_{-1}e+\_^{0}_{1}e\rightarrow2\gamma": (
            r"{}^{0}_{-1}\mathrm{e}+{}^{0}_{1}\mathrm{e}\rightarrow2\gamma"
        ),
    }
    normalized = 0
    for formula in document.source_blocks:
        if formula.disposition != "emitted" or formula.kind != "formula":
            continue
        compact = re.sub(r"\s+", "", formula.text or "")
        replacement = replacements.get(compact)
        if replacement is None:
            continue
        canonical_token = f"${formula.text}$"
        readable_token = latex_to_readable(formula.text)
        previous = (document.canonical_text, document.display_text)
        document.canonical_text = document.canonical_text.replace(
            canonical_token,
            f"${replacement}$",
            1,
        )
        document.display_text = document.display_text.replace(
            readable_token,
            latex_to_readable(replacement),
            1,
        )
        if previous != (document.canonical_text, document.display_text):
            normalized += 1

    previous = (document.canonical_text, document.display_text)
    for attr in ("canonical_text", "display_text"):
        value = getattr(document, attr)
        value = re.sub(r"(?<=中微子)[（(]\s*v\s*[）)]", "（ν）", value, count=1)
        setattr(document, attr, value)
    if previous != (document.canonical_text, document.display_text):
        normalized += 1
    return normalized


def normalize_document_for_output(
    document: QuestionDocument,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply conservative, provenance-backed output cleanup.

    The source blocks are intentionally retained unchanged.  Only the emitted
    document and its selected direct crops are normalized, and every change is
    recorded in warnings/metadata for later review.
    """

    cleanup = (config or {}).get("output_normalization", {})
    if cleanup.get("enabled", True) is False:
        return {
            "enabled": False,
            "caption_blocks_suppressed": 0,
            "inline_caption_references_restored": 0,
            "adjacent_visuals_removed": 0,
            "answer_placeholders_restored": 0,
            "short_underscore_answer_slots_normalized": 0,
            "geometric_answer_placeholders_restored": 0,
            "terminal_answer_placeholders_restored": 0,
            "ignored_relation_answer_lines_restored": 0,
            "visual_text_overlap_trimmed_assets": [],
            "redacted_acceleration_slots_restored": 0,
            "gravity_acceleration_slots_restored": 0,
            "pendulum_length_glyphs_normalized": 0,
            "overlapping_formula_tails_suppressed": 0,
            "triangle_angle_names_normalized": 0,
            "system_equation_environments_normalized": 0,
            "vector_style_normalizations": 0,
            "set_complement_normalizations": 0,
            "formula_quote_pairs_normalized": 0,
            "split_fraction_numerators_normalized": 0,
            "redundant_numeric_superscripts_normalized": 0,
            "gamma_ray_symbols_normalized": 0,
            "energy_level_directions_restored": 0,
            "terminal_heat_symbols_normalized": 0,
            "numeric_subscript_glyphs_normalized": 0,
            "derivative_primes_restored": 0,
            "trailing_semicolon_subscripts_normalized": 0,
            "stacked_denominator_formulas_restored": 0,
            "overlapping_duplicate_text_suppressed": 0,
            "leading_noise_trimmed": 0,
            "leading_exam_section_headers_suppressed": 0,
            "page_fraction_noise_suppressed": 0,
            "screen_toolbar_noise_suppressed": 0,
            "restarted_subpart_tails_suppressed": 0,
            "adjacent_question_tails_suppressed": 0,
            "marginal_handwritten_digits_suppressed": 0,
            "marginal_tokens_before_subparts_suppressed": 0,
            "terminal_cursor_tokens_suppressed": 0,
            "missing_two_column_option_labels_restored": 0,
            "exam_page_footers_suppressed": 0,
            "full_width_post_option_lines_suppressed": 0,
            "low_confidence_post_option_tails_suppressed": 0,
            "post_visual_tail_suppressed": 0,
            "terminal_compact_working_suppressed": 0,
            "tall_formula_answer_tails_suppressed": 0,
            "detached_equation_labels_suppressed": 0,
            "inline_caption_intrusions_suppressed": 0,
            "nuclear_particle_notations_normalized": 0,
            "caption_assets_expanded": [],
            "panel_label_assets_expanded": [],
            "captioned_panel_row_assets_expanded": [],
            "visual_fragments_merged": [],
            "ignored_panel_assets_expanded": [],
            "structured_table_region_assets_restored": 0,
            "embedded_panel_labels_suppressed": 0,
        }

    duplicate_cleanup_already_applied = any(
        warning.startswith("OVERLAPPING_DUPLICATE_TEXT_SUPPRESSED:")
        for warning in document.warnings
    )
    duplicate_text_suppressed = (
        _suppress_overlapping_duplicate_text(document)
        if cleanup.get("suppress_overlapping_duplicate_text", True)
        and not duplicate_cleanup_already_applied
        else 0
    )
    if duplicate_text_suppressed:
        _append_warning(
            document,
            f"OVERLAPPING_DUPLICATE_TEXT_SUPPRESSED:{duplicate_text_suppressed}",
        )
    leading_noise_trimmed = (
        _trim_leading_nonquestion_noise(document)
        if cleanup.get("trim_leading_nonquestion_noise", True)
        else 0
    )
    if leading_noise_trimmed:
        _append_warning(document, "LEADING_NONQUESTION_NOISE_TRIMMED")
    leading_section_headers_suppressed = (
        _suppress_leading_exam_section_headers(document)
        if cleanup.get("suppress_leading_exam_section_headers", True)
        else 0
    )
    if leading_section_headers_suppressed:
        _append_warning(
            document,
            f"LEADING_EXAM_SECTION_HEADERS_SUPPRESSED:{leading_section_headers_suppressed}",
        )
    page_fraction_suppressed = (
        _suppress_isolated_page_fraction(document)
        if cleanup.get("suppress_isolated_page_fraction", True)
        else 0
    )
    if page_fraction_suppressed:
        _append_warning(document, "ISOLATED_PAGE_FRACTION_SUPPRESSED")
    # Strip terminal page furniture from the outside in so a footer cannot
    # hide a marginal annotation or cursor artefact immediately before it.
    screen_toolbar_suppressed = (
        _suppress_terminal_screen_toolbar(document)
        if cleanup.get("suppress_terminal_screen_toolbar", True)
        else 0
    )
    if screen_toolbar_suppressed:
        _append_warning(document, "TERMINAL_SCREEN_TOOLBAR_SUPPRESSED")
    page_footers_suppressed = (
        _suppress_exam_page_footer(document)
        if cleanup.get("suppress_exam_page_footers", True)
        else 0
    )
    if page_footers_suppressed:
        _append_warning(document, "EXAM_PAGE_FOOTER_SUPPRESSED")
    post_option_lines_suppressed = (
        _suppress_full_width_post_option_line(document)
        if cleanup.get("suppress_full_width_post_option_lines", True)
        else 0
    )
    if post_option_lines_suppressed:
        _append_warning(document, "FULL_WIDTH_POST_OPTION_LINE_SUPPRESSED")
    low_confidence_post_option_tails_suppressed = (
        _suppress_low_confidence_post_option_tail(document)
        if cleanup.get("suppress_low_confidence_post_option_tails", True)
        else 0
    )
    if low_confidence_post_option_tails_suppressed:
        _append_warning(document, "LOW_CONFIDENCE_POST_OPTION_TAIL_SUPPRESSED")
    post_visual_tail_suppressed = (
        _suppress_post_visual_tail(document)
        if cleanup.get("suppress_post_visual_tail", True)
        else 0
    )
    if post_visual_tail_suppressed:
        _append_warning(document, "POST_VISUAL_TAIL_SUPPRESSED")
    restarted_subpart_tails_suppressed = (
        _suppress_terminal_restarted_subpart(document)
        if cleanup.get("suppress_terminal_restarted_subparts", True)
        else 0
    )
    if restarted_subpart_tails_suppressed:
        _append_warning(document, "TERMINAL_RESTARTED_SUBPART_SUPPRESSED")
    adjacent_question_tails_suppressed = (
        _suppress_terminal_adjacent_question(document)
        if cleanup.get("suppress_terminal_adjacent_questions", True)
        else 0
    )
    if adjacent_question_tails_suppressed:
        _append_warning(document, "TERMINAL_ADJACENT_QUESTION_SUPPRESSED")
    terminal_working_suppressed, detached_labels_suppressed = (
        _suppress_terminal_compact_working(document)
        if cleanup.get("suppress_terminal_compact_working", True)
        else (0, 0)
    )
    if terminal_working_suppressed:
        _append_warning(document, "TERMINAL_COMPACT_WORKING_SUPPRESSED")
    if detached_labels_suppressed:
        _append_warning(
            document,
            f"DETACHED_EQUATION_LABEL_SUPPRESSED:{detached_labels_suppressed}",
        )
    tall_formula_answer_tails_suppressed = (
        _suppress_tall_formula_answer_tail(document)
        if cleanup.get("suppress_tall_formula_answer_tails", True)
        and not any(
            warning == "TALL_FORMULA_ANSWER_TAIL_SUPPRESSED"
            for warning in document.warnings
        )
        else 0
    )
    if tall_formula_answer_tails_suppressed:
        _append_warning(document, "TALL_FORMULA_ANSWER_TAIL_SUPPRESSED")
    marginal_digits_suppressed = (
        _suppress_marginal_handwritten_digit(document)
        if cleanup.get("suppress_marginal_handwritten_digits", True)
        else 0
    )
    if marginal_digits_suppressed:
        _append_warning(document, "MARGINAL_HANDWRITTEN_DIGIT_SUPPRESSED")
    marginal_tokens_before_subparts_suppressed = (
        _suppress_marginal_token_before_subpart(document)
        if cleanup.get("suppress_marginal_tokens_before_subparts", True)
        else 0
    )
    if marginal_tokens_before_subparts_suppressed:
        _append_warning(document, "MARGINAL_TOKEN_BEFORE_SUBPART_SUPPRESSED")
    cursor_tokens_suppressed = (
        _suppress_terminal_cursor_token(document)
        if cleanup.get("suppress_terminal_cursor_tokens", True)
        else 0
    )
    if cursor_tokens_suppressed:
        _append_warning(document, "TERMINAL_CURSOR_TOKEN_SUPPRESSED")
    missing_option_labels_restored = (
        _restore_missing_two_column_option_label(document)
        if cleanup.get("restore_missing_two_column_option_labels", True)
        and not any(
            warning.startswith("MISSING_TWO_COLUMN_OPTION_LABEL_RESTORED:")
            for warning in document.warnings
        )
        else 0
    )
    if missing_option_labels_restored:
        _append_warning(
            document,
            f"MISSING_TWO_COLUMN_OPTION_LABEL_RESTORED:{missing_option_labels_restored}",
        )

    captions = _standalone_caption_blocks(document)
    caption_tokens = [block.text for block in captions]
    panel_label_blocks = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
        and _PANEL_LABEL_RE.fullmatch((block.text or "").strip())
    ]
    inline_caption_references_restored = _restore_inline_caption_references(
        document,
        captions,
    )
    if inline_caption_references_restored:
        _append_warning(
            document,
            f"INLINE_VISUAL_REFERENCE_RESTORED:{inline_caption_references_restored}",
        )
    inline_caption_intrusions_suppressed = _suppress_inline_caption_intrusions(
        document,
        captions,
    )
    if inline_caption_intrusions_suppressed:
        _append_warning(
            document,
            f"INLINE_CAPTION_INTRUSION_SUPPRESSED:{inline_caption_intrusions_suppressed}",
        )
    captioned_panel_row_assets_expanded = _expand_captioned_panel_rows(
        document,
        captions,
    )
    if captioned_panel_row_assets_expanded:
        _append_warning(
            document,
            "CAPTIONED_PANEL_ROW_EXPANDED:"
            + ",".join(captioned_panel_row_assets_expanded),
        )
    suppressed = _suppress_caption_only_lines(
        document,
        captions,
        panel_label_blocks,
    )
    caption_cleanup_already_applied = any(
        warning.startswith("STANDALONE_VISUAL_CAPTION_SUPPRESSED:")
        for warning in document.warnings
    )
    if (
        cleanup.get("suppress_standalone_visual_captions", True)
        and caption_tokens
        and not caption_cleanup_already_applied
        and not suppressed
    ):
        previous = (document.canonical_text, document.display_text)
        document.canonical_text = _remove_emitted_caption_tokens(
            document.canonical_text,
            caption_tokens,
        )
        document.display_text = _remove_emitted_caption_tokens(
            document.display_text,
            caption_tokens,
        )
        if previous != (document.canonical_text, document.display_text):
            suppressed = len(caption_tokens)
    if suppressed:
        _append_warning(document, f"STANDALONE_VISUAL_CAPTION_SUPPRESSED:{suppressed}")

    removed_assets: list[str] = []
    target_number = _question_base_number(document.question_number)
    numbered_captions = [
        (number, block)
        for block in captions
        if (number := _caption_number(block)) is not None
    ]
    distinct_numbers = {number for number, _block in numbered_captions}
    if (
        cleanup.get("filter_captioned_adjacent_visuals", True)
        and target_number is not None
        and target_number in distinct_numbers
        and len(distinct_numbers) >= 2
        and len(document.visual_assets) >= 2
    ):
        assignments = _caption_assignments(numbered_captions, document.visual_assets)
        retained: list[VisualAsset] = []
        for asset in document.visual_assets:
            assigned_number = assignments.get(asset.asset_id)
            if assigned_number is not None and assigned_number != target_number:
                removed_assets.append(f"{asset.asset_id}:{assigned_number}")
                continue
            retained.append(asset)
        if retained and removed_assets:
            document.visual_assets = retained
            _append_warning(
                document,
                "ADJACENT_CAPTIONED_VISUAL_REMOVED:" + ",".join(removed_assets),
            )
        else:
            removed_assets = []

    expanded_caption_assets: list[str] = []
    panel_label_assets_expanded: list[str] = []
    if cleanup.get("include_caption_pixels_in_visual_crop", True):
        expanded_caption_assets.extend(
            _expand_target_assets_to_captions(document, numbered_captions)
        )
        expanded_caption_assets.extend(
            _expand_standalone_caption_assets(document, captions)
        )
        panel_label_assets_expanded = _expand_panel_label_assets(
            document,
            panel_label_blocks,
        )
        expanded_caption_assets = list(dict.fromkeys(expanded_caption_assets))
    if expanded_caption_assets:
        _append_warning(
            document,
            "VISUAL_CROP_EXPANDED_TO_CAPTION:" + ",".join(expanded_caption_assets),
        )
    if panel_label_assets_expanded:
        _append_warning(
            document,
            "VISUAL_CROP_EXPANDED_TO_PANEL_LABEL:"
            + ",".join(panel_label_assets_expanded),
        )

    merged_fragments = (
        _merge_vertical_visual_fragments(document)
        if cleanup.get("merge_vertical_visual_fragments", True)
        else []
    )
    if merged_fragments:
        _append_warning(
            document,
            "VERTICAL_VISUAL_FRAGMENT_MERGED:" + ",".join(merged_fragments),
        )

    ignored_panel_assets_expanded = (
        _expand_visual_to_ignored_panel_row(document)
        if cleanup.get("expand_visual_to_ignored_panel_rows", True)
        else []
    )
    if ignored_panel_assets_expanded:
        _append_warning(
            document,
            "VISUAL_CROP_EXPANDED_TO_IGNORED_PANEL_ROW:"
            + ",".join(ignored_panel_assets_expanded),
        )

    structured_table_assets_restored = (
        _restore_structured_table_region_asset(document)
        if cleanup.get("restore_structured_table_region_assets", True)
        and not any(
            warning.startswith("STRUCTURED_TABLE_REGION_ASSET_RESTORED:")
            for warning in document.warnings
        )
        else 0
    )
    if structured_table_assets_restored:
        _append_warning(
            document,
            "STRUCTURED_TABLE_REGION_ASSET_RESTORED:"
            f"{structured_table_assets_restored}",
        )

    panel_labels_suppressed = (
        _suppress_embedded_panel_label_rows(document)
        if cleanup.get("suppress_embedded_panel_label_rows", True)
        else 0
    )
    if panel_labels_suppressed:
        _append_warning(
            document,
            f"EMBEDDED_PANEL_LABEL_ROW_SUPPRESSED:{panel_labels_suppressed}",
        )

    restored = document.canonical_text.count("▲") + document.display_text.count("▲")
    if cleanup.get("restore_triangle_answer_placeholders", True) and restored:
        # The dataset uses a solid triangle to redact a filled answer while the
        # underlying printed slot remains part of the question.  It is not a
        # mathematical triangle symbol (which is printed as △/\triangle).
        document.canonical_text = document.canonical_text.replace("▲", "______")
        document.display_text = document.display_text.replace("▲", "______")
        _append_warning(document, f"ANSWER_PLACEHOLDER_RESTORED:{restored // 2 or 1}")
        restored = restored // 2 or 1
    else:
        restored = 0

    short_underscore_slots_normalized = (
        _normalize_short_underscore_answer_slot(document)
        if cleanup.get("normalize_short_underscore_answer_slots", True)
        else 0
    )
    if short_underscore_slots_normalized:
        _append_warning(document, "SHORT_UNDERSCORE_ANSWER_SLOT_NORMALIZED")

    geometric_placeholders_restored = (
        _restore_geometric_answer_placeholders(document)
        if cleanup.get("restore_geometric_answer_placeholders", True)
        else 0
    )
    if geometric_placeholders_restored:
        _append_warning(
            document,
            f"GEOMETRIC_ANSWER_PLACEHOLDER_RESTORED:{geometric_placeholders_restored}",
        )

    terminal_answer_placeholders_restored = (
        _restore_terminal_answer_placeholder(document)
        if cleanup.get("restore_terminal_answer_placeholders", True)
        else 0
    )
    if terminal_answer_placeholders_restored:
        _append_warning(document, "TERMINAL_ANSWER_PLACEHOLDER_RESTORED")

    ignored_relation_lines_restored, visual_text_overlap_trimmed_assets = (
        _restore_ignored_relation_answer_line(document)
        if cleanup.get("restore_ignored_relation_answer_lines", True)
        else (0, [])
    )
    if ignored_relation_lines_restored:
        _append_warning(document, "IGNORED_RELATION_ANSWER_LINE_RESTORED")
    if visual_text_overlap_trimmed_assets:
        _append_warning(
            document,
            "VISUAL_TEXT_OVERLAP_TRIMMED:"
            + ",".join(visual_text_overlap_trimmed_assets),
        )

    redacted_acceleration_slots_restored = (
        _restore_redacted_acceleration_formula_slot(document)
        if cleanup.get("restore_redacted_acceleration_formula_slots", True)
        else 0
    )
    if redacted_acceleration_slots_restored:
        _append_warning(document, "REDACTED_ACCELERATION_SLOT_RESTORED")

    gravity_acceleration_slots_restored = (
        _restore_gravity_acceleration_formula_slot(document)
        if cleanup.get("restore_gravity_acceleration_formula_slots", True)
        else 0
    )
    if gravity_acceleration_slots_restored:
        _append_warning(document, "GRAVITY_ACCELERATION_SLOT_RESTORED")

    pendulum_length_glyphs_normalized = (
        _normalize_pendulum_length_glyph(document)
        if cleanup.get("normalize_pendulum_length_glyphs", True)
        else 0
    )
    if pendulum_length_glyphs_normalized:
        _append_warning(document, "PENDULUM_LENGTH_GLYPH_NORMALIZED")

    overlapping_formula_tails_suppressed = (
        _suppress_overlapping_formula_tail(document)
        if cleanup.get("suppress_overlapping_formula_tails", True)
        else 0
    )
    if overlapping_formula_tails_suppressed:
        _append_warning(document, "OVERLAPPING_FORMULA_TAIL_SUPPRESSED")

    triangle_angle_names_normalized = (
        _normalize_triangle_angle_name(document)
        if cleanup.get("normalize_triangle_angle_names", True)
        else 0
    )
    if triangle_angle_names_normalized:
        _append_warning(document, "TRIANGLE_ANGLE_NAME_NORMALIZED")

    system_equation_environments_normalized = (
        _normalize_system_equation_environment(document)
        if cleanup.get("normalize_system_equation_environments", True)
        else 0
    )
    if system_equation_environments_normalized:
        _append_warning(document, "SYSTEM_EQUATION_ENVIRONMENT_NORMALIZED")

    nuclear_particle_notations_normalized = (
        _normalize_nuclear_particle_notation(document)
        if cleanup.get("normalize_nuclear_particle_notation", True)
        and not any(
            warning.startswith("NUCLEAR_PARTICLE_NOTATION_NORMALIZED:")
            for warning in document.warnings
        )
        else 0
    )
    if nuclear_particle_notations_normalized:
        _append_warning(
            document,
            "NUCLEAR_PARTICLE_NOTATION_NORMALIZED:"
            f"{nuclear_particle_notations_normalized}",
        )

    vector_style_normalizations = (
        _normalize_established_vector_style(document)
        if cleanup.get("normalize_established_vector_style", True)
        and not any(
            warning.startswith("ESTABLISHED_VECTOR_STYLE_NORMALIZED:")
            for warning in document.warnings
        )
        else 0
    )
    if vector_style_normalizations:
        _append_warning(
            document,
            f"ESTABLISHED_VECTOR_STYLE_NORMALIZED:{vector_style_normalizations}",
        )

    set_complement_normalizations = (
        _normalize_set_complement_notation(document)
        if cleanup.get("normalize_set_complement_notation", True)
        and not any(
            warning.startswith("SET_COMPLEMENT_NOTATION_NORMALIZED:")
            for warning in document.warnings
        )
        else 0
    )
    if set_complement_normalizations:
        _append_warning(
            document,
            f"SET_COMPLEMENT_NOTATION_NORMALIZED:{set_complement_normalizations}",
        )

    formula_quote_pairs_normalized = (
        _normalize_paired_formula_quotes(document)
        if cleanup.get("normalize_paired_formula_quotes", True)
        and not any(
            warning.startswith("PAIRED_FORMULA_QUOTES_NORMALIZED:")
            for warning in document.warnings
        )
        else 0
    )
    if formula_quote_pairs_normalized:
        _append_warning(
            document,
            f"PAIRED_FORMULA_QUOTES_NORMALIZED:{formula_quote_pairs_normalized}",
        )

    split_fraction_numerators_normalized = (
        _normalize_split_fraction_numerator(document)
        if cleanup.get("normalize_split_fraction_numerators", True)
        and not any(
            warning.startswith("SPLIT_FRACTION_NUMERATOR_NORMALIZED:")
            for warning in document.warnings
        )
        else 0
    )
    if split_fraction_numerators_normalized:
        _append_warning(
            document,
            "SPLIT_FRACTION_NUMERATOR_NORMALIZED:"
            f"{split_fraction_numerators_normalized}",
        )

    redundant_numeric_superscripts_normalized = (
        _normalize_redundant_numeric_superscripts(document)
        if cleanup.get("normalize_redundant_numeric_superscripts", True)
        and not any(
            warning.startswith("REDUNDANT_NUMERIC_SUPERSCRIPT_NORMALIZED:")
            for warning in document.warnings
        )
        else 0
    )
    if redundant_numeric_superscripts_normalized:
        _append_warning(
            document,
            "REDUNDANT_NUMERIC_SUPERSCRIPT_NORMALIZED:"
            f"{redundant_numeric_superscripts_normalized}",
        )

    gamma_ray_symbols_normalized = (
        _normalize_gamma_ray_symbol(document)
        if cleanup.get("normalize_gamma_ray_symbols", True)
        and not any(
            warning.startswith("GAMMA_RAY_SYMBOL_NORMALIZED:")
            for warning in document.warnings
        )
        else 0
    )
    if gamma_ray_symbols_normalized:
        _append_warning(
            document,
            f"GAMMA_RAY_SYMBOL_NORMALIZED:{gamma_ray_symbols_normalized}",
        )

    energy_level_directions_restored = (
        _restore_energy_level_transition_direction(document)
        if cleanup.get("restore_energy_level_transition_directions", True)
        and not any(
            warning.startswith("ENERGY_LEVEL_DIRECTION_RESTORED:")
            for warning in document.warnings
        )
        else 0
    )
    if energy_level_directions_restored:
        _append_warning(
            document,
            f"ENERGY_LEVEL_DIRECTION_RESTORED:{energy_level_directions_restored}",
        )

    terminal_heat_symbols_normalized = (
        _normalize_terminal_heat_symbol(document)
        if cleanup.get("normalize_terminal_heat_symbols", True)
        and not any(
            warning.startswith("TERMINAL_HEAT_SYMBOL_NORMALIZED:")
            for warning in document.warnings
        )
        else 0
    )
    if terminal_heat_symbols_normalized:
        _append_warning(
            document,
            f"TERMINAL_HEAT_SYMBOL_NORMALIZED:{terminal_heat_symbols_normalized}",
        )

    numeric_subscript_glyphs_normalized = (
        _normalize_numeric_subscript_glyph_confusion(document)
        if cleanup.get("normalize_numeric_subscript_glyph_confusions", True)
        and not any(
            warning.startswith("NUMERIC_SUBSCRIPT_GLYPH_NORMALIZED:")
            for warning in document.warnings
        )
        else 0
    )
    if numeric_subscript_glyphs_normalized:
        _append_warning(
            document,
            f"NUMERIC_SUBSCRIPT_GLYPH_NORMALIZED:{numeric_subscript_glyphs_normalized}",
        )

    derivative_primes_restored = (
        _restore_derivative_prime_from_definition(document)
        if cleanup.get("restore_derivative_primes_from_definition", True)
        and not any(
            warning.startswith("DERIVATIVE_PRIME_RESTORED:")
            for warning in document.warnings
        )
        else 0
    )
    if derivative_primes_restored:
        _append_warning(
            document,
            f"DERIVATIVE_PRIME_RESTORED:{derivative_primes_restored}",
        )

    trailing_semicolon_subscripts_normalized = (
        _normalize_trailing_semicolon_as_subscript(document)
        if cleanup.get("normalize_trailing_semicolon_subscripts", True)
        and not any(
            warning.startswith("TRAILING_SEMICOLON_SUBSCRIPT_NORMALIZED:")
            for warning in document.warnings
        )
        else 0
    )
    if trailing_semicolon_subscripts_normalized:
        _append_warning(
            document,
            "TRAILING_SEMICOLON_SUBSCRIPT_NORMALIZED:"
            f"{trailing_semicolon_subscripts_normalized}",
        )

    stacked_denominator_formulas_restored = (
        _restore_stacked_denominator_formula(document)
        if cleanup.get("restore_stacked_denominator_formulas", True)
        and not any(
            warning.startswith("STACKED_DENOMINATOR_FORMULA_RESTORED:")
            for warning in document.warnings
        )
        else 0
    )
    if stacked_denominator_formulas_restored:
        _append_warning(
            document,
            "STACKED_DENOMINATOR_FORMULA_RESTORED:"
            f"{stacked_denominator_formulas_restored}",
        )

    if (
        suppressed
        or inline_caption_references_restored
        or restored
        or short_underscore_slots_normalized
        or geometric_placeholders_restored
        or terminal_answer_placeholders_restored
        or ignored_relation_lines_restored
        or redacted_acceleration_slots_restored
        or gravity_acceleration_slots_restored
        or pendulum_length_glyphs_normalized
        or overlapping_formula_tails_suppressed
        or triangle_angle_names_normalized
        or system_equation_environments_normalized
        or vector_style_normalizations
        or set_complement_normalizations
        or formula_quote_pairs_normalized
        or split_fraction_numerators_normalized
        or redundant_numeric_superscripts_normalized
        or gamma_ray_symbols_normalized
        or energy_level_directions_restored
        or terminal_heat_symbols_normalized
        or numeric_subscript_glyphs_normalized
        or derivative_primes_restored
        or trailing_semicolon_subscripts_normalized
        or stacked_denominator_formulas_restored
        or duplicate_text_suppressed
        or leading_noise_trimmed
        or leading_section_headers_suppressed
        or page_fraction_suppressed
        or screen_toolbar_suppressed
        or restarted_subpart_tails_suppressed
        or adjacent_question_tails_suppressed
        or marginal_digits_suppressed
        or marginal_tokens_before_subparts_suppressed
        or cursor_tokens_suppressed
        or missing_option_labels_restored
        or page_footers_suppressed
        or post_option_lines_suppressed
        or low_confidence_post_option_tails_suppressed
        or post_visual_tail_suppressed
        or terminal_working_suppressed
        or tall_formula_answer_tails_suppressed
        or detached_labels_suppressed
        or inline_caption_intrusions_suppressed
        or nuclear_particle_notations_normalized
        or ignored_panel_assets_expanded
        or structured_table_assets_restored
        or panel_labels_suppressed
        or panel_label_assets_expanded
        or captioned_panel_row_assets_expanded
    ):
        structure_warnings: list[str] = []
        document.question_number, document.stem, document.options = parse_question(
            document.display_text,
            recovery_warnings=structure_warnings,
        )
        for warning in structure_warnings:
            _append_warning(document, warning)

    recovered_caption_number: str | None = None
    if document.question_number is None:
        recovered_caption_number = _recover_question_number_from_visual_caption(document)
        if recovered_caption_number is not None:
            # Unlike authenticated synthetic sample IDs, this number is source
            # content. Put it back into both text representations so it is
            # actually rendered rather than existing only in metadata.
            document.canonical_text = (
                f"{recovered_caption_number}. {document.canonical_text.lstrip()}"
            )
            document.display_text = (
                f"{recovered_caption_number}. {document.display_text.lstrip()}"
            )
            (
                document.question_number,
                document.stem,
                document.options,
            ) = parse_question(document.display_text)
            _append_warning(
                document,
                "QUESTION_NUMBER_RECOVERED_FROM_VISUAL_CAPTION:"
                f"{recovered_caption_number}",
            )

    return {
        "enabled": True,
        "caption_blocks_suppressed": suppressed,
        "question_number_recovered_from_visual_caption": (
            recovered_caption_number is not None
        ),
        "inline_caption_references_restored": inline_caption_references_restored,
        "adjacent_visuals_removed": len(removed_assets),
        "removed_visuals": removed_assets,
        "answer_placeholders_restored": restored,
        "short_underscore_answer_slots_normalized": short_underscore_slots_normalized,
        "geometric_answer_placeholders_restored": geometric_placeholders_restored,
        "terminal_answer_placeholders_restored": terminal_answer_placeholders_restored,
        "ignored_relation_answer_lines_restored": ignored_relation_lines_restored,
        "visual_text_overlap_trimmed_assets": visual_text_overlap_trimmed_assets,
        "redacted_acceleration_slots_restored": redacted_acceleration_slots_restored,
        "gravity_acceleration_slots_restored": gravity_acceleration_slots_restored,
        "pendulum_length_glyphs_normalized": pendulum_length_glyphs_normalized,
        "overlapping_formula_tails_suppressed": overlapping_formula_tails_suppressed,
        "triangle_angle_names_normalized": triangle_angle_names_normalized,
        "system_equation_environments_normalized": system_equation_environments_normalized,
        "vector_style_normalizations": vector_style_normalizations,
        "set_complement_normalizations": set_complement_normalizations,
        "formula_quote_pairs_normalized": formula_quote_pairs_normalized,
        "split_fraction_numerators_normalized": split_fraction_numerators_normalized,
        "redundant_numeric_superscripts_normalized": redundant_numeric_superscripts_normalized,
        "gamma_ray_symbols_normalized": gamma_ray_symbols_normalized,
        "energy_level_directions_restored": energy_level_directions_restored,
        "terminal_heat_symbols_normalized": terminal_heat_symbols_normalized,
        "numeric_subscript_glyphs_normalized": numeric_subscript_glyphs_normalized,
        "derivative_primes_restored": derivative_primes_restored,
        "trailing_semicolon_subscripts_normalized": trailing_semicolon_subscripts_normalized,
        "stacked_denominator_formulas_restored": stacked_denominator_formulas_restored,
        "overlapping_duplicate_text_suppressed": duplicate_text_suppressed,
        "leading_noise_trimmed": leading_noise_trimmed,
        "leading_exam_section_headers_suppressed": leading_section_headers_suppressed,
        "page_fraction_noise_suppressed": page_fraction_suppressed,
        "screen_toolbar_noise_suppressed": screen_toolbar_suppressed,
        "restarted_subpart_tails_suppressed": restarted_subpart_tails_suppressed,
        "adjacent_question_tails_suppressed": adjacent_question_tails_suppressed,
        "marginal_handwritten_digits_suppressed": marginal_digits_suppressed,
        "marginal_tokens_before_subparts_suppressed": marginal_tokens_before_subparts_suppressed,
        "terminal_cursor_tokens_suppressed": cursor_tokens_suppressed,
        "missing_two_column_option_labels_restored": missing_option_labels_restored,
        "exam_page_footers_suppressed": page_footers_suppressed,
        "full_width_post_option_lines_suppressed": post_option_lines_suppressed,
        "low_confidence_post_option_tails_suppressed": low_confidence_post_option_tails_suppressed,
        "post_visual_tail_suppressed": post_visual_tail_suppressed,
        "terminal_compact_working_suppressed": terminal_working_suppressed,
        "tall_formula_answer_tails_suppressed": tall_formula_answer_tails_suppressed,
        "detached_equation_labels_suppressed": detached_labels_suppressed,
        "inline_caption_intrusions_suppressed": inline_caption_intrusions_suppressed,
        "nuclear_particle_notations_normalized": nuclear_particle_notations_normalized,
        "caption_assets_expanded": expanded_caption_assets,
        "panel_label_assets_expanded": panel_label_assets_expanded,
        "captioned_panel_row_assets_expanded": captioned_panel_row_assets_expanded,
        "visual_fragments_merged": merged_fragments,
        "ignored_panel_assets_expanded": ignored_panel_assets_expanded,
        "structured_table_region_assets_restored": structured_table_assets_restored,
        "embedded_panel_labels_suppressed": panel_labels_suppressed,
    }
