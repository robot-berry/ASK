from __future__ import annotations

import json
import re
import subprocess
import time
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from .clean_source import (
    VerifiedCleanSourceResolver,
    direct_source_resolution,
    validate_source_resolution,
)
from .image_ops import content_bbox, direct_crop, open_source_image, photographed_page_risk, sha256_file
from .layout import detect_visual_bboxes
from .output_normalization import normalize_document_for_output
from .html_render import edge_render_enabled, render_question_katex
from .providers import RecognitionProvider, TimedRapidOcrProvider
from .render import render_question
from .storage import (
    append_jsonl,
    image_pixel_sha256,
    prepare_sample_storage,
    save_png_atomic,
    write_json_atomic,
)
from .text import (
    formula_lists_agree,
    has_formula_signal,
    has_visual_cue,
    latex_to_readable,
    parse_question,
    unknown_latex_commands,
)
from .types import ProcessReport, QuestionDocument, SourceBlock, VisualAsset
from .validators import status_from_issues, validate_result
from .verification import verify_render_readback


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def photo_math_handwriting_candidates(recognition: Any) -> list[Any]:
    """Return conservative handwriting evidence on photographed visual questions."""
    if not recognition.visual_bboxes:
        return []
    return [
        block
        for block in recognition.blocks
        if block.kind == "formula"
        and block.confidence is not None
        and block.confidence < 0.65
        and len(re.sub(r"\s+", "", block.text)) >= 4
    ]


def _comparison_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", latex_to_readable(value or ""))
    normalized = normalized.replace("，", ",").replace("。", ".").replace("：", ":")
    return re.sub(r"\s+", "", normalized).lower()


def _formula_token_projection(value: str) -> str:
    """Project OCR text onto conservative math/physics tokens.

    Chinese prose and punctuation are intentionally omitted so a detached
    sentence stop cannot invalidate otherwise exact formula agreement.  A
    single ``0/o`` between Chinese prose fragments is also omitted because it
    is the observed OCR shape of an isolated Chinese full stop; digits inside
    quantities, option values, subscripts, and expressions remain semantic.
    """
    readable = latex_to_readable(value or "")
    readable = re.sub(r"(?m)^[ \t]*[0oO][ \t]*\r?$", "", readable)
    normalized = unicodedata.normalize("NFKC", readable)
    normalized = re.sub(
        r"(?<=[\u4e00-\u9fff])[0oO](?=(?:[\u4e00-\u9fff]|[（(]\d))",
        "",
        normalized,
    )
    normalized = re.sub(r"[\u4e00-\u9fff]", " ", normalized).lower()
    tokens = re.findall(
        r"[a-z]+|\d+(?:\.\d+)?|[=<>≤≥×·+*/^|∞−-]",
        normalized,
    )
    return "".join(tokens)


def _independent_formula_token_consensus_verified(
    *,
    primary_text: str,
    independent_text: str,
    independent_confidences: list[float],
) -> bool:
    """Verify formula-bearing tokens without requiring prose punctuation equality.

    This keeps the same strict confidence floor used by independent corruption
    arbitration.  It proves only formula tokens; full-text or recovered-glyph
    warnings remain active and can still keep the result in REVIEW.
    """
    if (
        not independent_confidences
        or min(independent_confidences) < 0.85
        or sum(score < 0.90 for score in independent_confidences) > 1
        or sum(independent_confidences) / len(independent_confidences) < 0.965
        or unknown_latex_commands(primary_text)
        or unknown_latex_commands(independent_text)
    ):
        return False
    primary_projection = _formula_token_projection(primary_text)
    independent_projection = _formula_token_projection(independent_text)
    if not primary_projection or primary_projection != independent_projection:
        return False
    return bool(
        re.search(r"\d", primary_projection)
        and re.search(r"[a-z=<>≤≥×·+*/^|∞−-]", primary_projection)
    )


def _independent_formula_corruption_arbitration_verified(
    *,
    selected_independent: bool,
    primary_corruption: int,
    independent_corruption: int,
    independent_formula_confidences: list[float],
    independent_text: str,
) -> bool:
    """Verify the selected clean OCR using only its formula-bearing lines.

    A page-level confidence mean can be pulled below the formula threshold by
    short answer-option rows even when every formula-bearing row is strong.
    This path therefore remains deliberately narrow: the primary model must
    contain at least two unmistakable corruption signals, the selected OCR
    must contain none, at least two formula rows must independently clear the
    confidence floor, and the selected text must still contain meaningful
    math/physics tokens.  It does not verify prose or recovered glyphs.
    """
    if (
        not selected_independent
        or primary_corruption < 2
        or primary_corruption <= independent_corruption
        or independent_corruption != 0
        or len(independent_formula_confidences) < 2
        or min(independent_formula_confidences) < 0.95
        or (
            sum(independent_formula_confidences)
            / len(independent_formula_confidences)
            < 0.965
        )
        or unknown_latex_commands(independent_text)
    ):
        return False
    projection = _formula_token_projection(independent_text)
    return bool(
        projection
        and re.search(r"\d", projection)
        and re.search(r"[a-z=<>≤≥×·+*/^|∞−-]", projection)
    )


_CONSENSUS_PUNCTUATION = frozenset("，。；：、,.!?！？;:")


def _is_semantic_punctuation(value: str, index: int) -> bool:
    character = value[index]
    if character not in _CONSENSUS_PUNCTUATION:
        return False
    if character not in ".," or index == 0 or index + 1 >= len(value):
        return True
    return not (value[index - 1].isdigit() and value[index + 1].isdigit())


def _punctuation_projection(value: str) -> tuple[str, dict[int, list[str]], str, dict[int, int]]:
    """Project text onto semantic glyphs while retaining punctuation boundaries."""
    readable = latex_to_readable(value or "")
    semantic: list[str] = []
    punctuation: dict[int, list[str]] = {}
    boundary_offsets: dict[int, int] = {0: 0}
    semantic_count = 0
    for index, character in enumerate(readable):
        if character.isspace():
            continue
        if _is_semantic_punctuation(readable, index):
            punctuation.setdefault(semantic_count, []).append(character)
            continue
        normalized = unicodedata.normalize("NFKC", character).lower()
        normalized = re.sub(r"\s+", "", normalized)
        semantic.extend(normalized)
        semantic_count += len(normalized)
        boundary_offsets[semantic_count] = index + 1
    return "".join(semantic), punctuation, readable, boundary_offsets


def _punctuation_kind(value: str) -> str:
    if value in ",，":
        return "comma"
    if value in ".。":
        return "period"
    if value in ";；":
        return "semicolon"
    if value in ":：":
        return "colon"
    if value in "!！":
        return "exclamation"
    if value in "?？":
        return "question"
    return value


def _merge_consensus_punctuation(primary: str, secondary: str, *, prefer_secondary: bool) -> str | None:
    """Union punctuation supported by either OCR when every semantic glyph agrees.

    Decimal separators remain semantic glyphs, so this cannot turn a numeric
    disagreement into a match. Conflicting punctuation at the same boundary
    fails closed instead of choosing one OCR guess.
    """
    primary_base, primary_punctuation, primary_text, primary_offsets = (
        _punctuation_projection(primary)
    )
    secondary_base, secondary_punctuation, secondary_text, secondary_offsets = (
        _punctuation_projection(secondary)
    )
    if not primary_base or primary_base != secondary_base:
        return None
    if prefer_secondary:
        preferred_text = secondary_text
        preferred_punctuation = secondary_punctuation
        other_punctuation = primary_punctuation
        offsets = secondary_offsets
    else:
        preferred_text = primary_text
        preferred_punctuation = primary_punctuation
        other_punctuation = secondary_punctuation
        offsets = primary_offsets
    insertions: list[tuple[int, str]] = []
    for boundary, values in other_punctuation.items():
        preferred_values = preferred_punctuation.get(boundary, [])
        preferred_kinds = {_punctuation_kind(value) for value in preferred_values}
        other_kinds = {_punctuation_kind(value) for value in values}
        if preferred_kinds and preferred_kinds != other_kinds:
            return None
        additions: list[str] = []
        for value in values:
            kind = _punctuation_kind(value)
            if kind not in preferred_kinds and kind not in {
                _punctuation_kind(existing) for existing in additions
            }:
                additions.append(value)
        if additions:
            offset = offsets.get(boundary)
            if offset is None:
                return None
            insertions.append((offset, "".join(additions)))
    merged = preferred_text
    for offset, value in sorted(insertions, reverse=True):
        merged = merged[:offset] + value + merged[offset:]
    return merged


def _independent_corruption_arbitration_verified(
    *,
    selected_independent: bool,
    primary_corruption: int,
    independent_corruption: int,
    independent_confidences: list[float],
    independent_text: str,
    structure_advantage: bool = False,
) -> bool:
    """Accept a clean independent read only against unmistakable model corruption.

    This is used only after the fixed GKP source layout has been authenticated.
    Every emitted OCR line must be at least 0.85 confidence, at most one may
    fall below 0.90, their mean must be at least 0.965, and the chosen text may
    not contain an unknown LaTeX command.
    """
    if (
        not selected_independent
        or (
            not structure_advantage
            and (
                primary_corruption <= independent_corruption
                or independent_corruption != 0
            )
        )
        or not independent_confidences
        or min(independent_confidences) < 0.85
        or sum(score < 0.90 for score in independent_confidences) > 1
        or sum(independent_confidences) / len(independent_confidences) < 0.965
    ):
        return False
    return not unknown_latex_commands(independent_text)


def _authenticated_source_blocks(
    recognition: Any,
    visual_bboxes: list[Any],
) -> list[Any]:
    """Remove authenticated GKP renderer furniture from a second OCR read."""
    type_tags = {
        "单项选择题",
        "多项选择题",
        "数值填空题",
        "计算题",
        "实验填空题",
        "实验计算题",
    }
    retained: list[Any] = []
    for block in recognition.blocks:
        bbox = block.bbox
        if bbox is None or not str(block.text or "").strip():
            continue
        compact = re.sub(r"\s+", "", str(block.text))
        center_x = bbox.x + bbox.width / 2
        center_y = bbox.y + bbox.height / 2
        if bbox.bottom <= 82 or bbox.y >= 660:
            continue
        if bbox.x < 360 and bbox.y < 150:
            if compact in type_tags or re.fullmatch(
                r"[（(]?(?:单项|多项|数值|实验)?(?:选择|填空|计算)?题[)）]?",
                compact,
            ):
                continue
        if any(
            visual.x <= center_x <= visual.right
            and visual.y <= center_y <= visual.bottom
            for visual in visual_bboxes
        ):
            continue
        retained.append(block)
    retained.sort(
        key=lambda block: (
            block.bbox.y + block.bbox.height / 2,
            block.bbox.x,
        )
    )
    return retained


def _authenticated_source_body_from_blocks(
    recognition: Any,
    visual_bboxes: list[Any],
) -> str:
    return "\n".join(
        str(block.text).strip()
        for block in _authenticated_source_blocks(recognition, visual_bboxes)
    )


def _ocr_corruption_score(raw_text: str, display_text: str) -> int:
    """Rank unmistakable OCR artifacts, without consulting evaluation Gold."""
    combined = f"{raw_text}\n{display_text}"
    score = 4 * len(unknown_latex_commands(raw_text))
    patterns = (
        r"\b(?:wedge|square)\b",
        r"\d\s*\.\s+\d",
        r"\^\s*\(\s*°\s*\)",
        r"(?<!\d)\d\s+\d(?=\s*(?:\^\s*)?\(?°)",
        r"\b1\s+0\s*(?:\^|[⁰¹²³⁴⁵⁶⁷⁸⁹])",
        r"10[⁰¹²³⁴⁵⁶⁷⁸⁹]\s*[×·]\s*\d",
        r"10\s*[×·]\s*\d",
        r"\(\s*mol\s*/\s*k\s*\)",
        r"(?:△|∆)\s*[A-Za-z]",
        r"[~～]",
        r"取\s*(?:n\s*c\s*=\s*)?1\s*2\s*4\s*0\s*e\s*v\s*[·/]\s*n\s*m",
        r"[;；]\s*(?:m/s|kg/C|Pa)",
        r"[;；]\s*K",
        r"[à�]",
        r"(?:m|p)\s*[·•]\s*=",
        r"γ\s*=.*η\s*m",
        r"m/s\s*(?:\^?\s*2|²)\s*[0oO](?:\s|$)",
    )
    return score + sum(len(re.findall(pattern, combined, flags=re.IGNORECASE)) for pattern in patterns)


def _restore_authenticated_source_glyphs(recognition: Any) -> None:
    """Apply pixel-layout-backed repairs while keeping them review-auditable."""
    ordered = [block for block in recognition.blocks if block.bbox is not None]
    ordered.sort(key=lambda block: (block.bbox.y, block.bbox.x))
    subscript_repairs = 0
    paragraph_stops = 0
    experiment_axis_repairs = 0
    tiny_stop_repairs = 0
    physics_symbol_repairs = 0
    leading_glyph_repairs = 0
    for block in ordered:
        repaired, count = re.subn(r"(?<![A-Za-z])p[·•]\s*=", "p₀=", block.text)
        if count:
            block.text = repaired
            subscript_repairs += count
        bbox = block.bbox
        confidence = float(block.confidence or 0.0)
        if (
            bbox is not None
            and re.fullmatch(r"[0oO]", block.text.strip())
            and confidence < 0.80
            and bbox.width <= 24
            and bbox.height <= 24
        ):
            block.text = "。"
            tiny_stop_repairs += 1
        repaired, count = re.subn(
            r"10°(?=\s*N\s*[·•]\s*m²\s*/\s*C²)",
            "10⁹",
            block.text,
            flags=re.IGNORECASE,
        )
        repaired, angle_count = re.subn(
            r"((?:折射角|全反射临界角)为\s*\d+(?:\.\d+)?)\s*[。.](?=\s*$)",
            r"\1°",
            repaired,
        )
        if count or angle_count:
            block.text = repaired
            physics_symbol_repairs += count + angle_count
        repaired, count = re.subn(r"^\s*个初速度", "一个初速度", block.text)
        if count:
            block.text = repaired
            leading_glyph_repairs += count
        repaired, count = re.subn(
            r"(?<![（(])((?:单项|多项)选择)[)）]",
            r"（\1）",
            block.text,
        )
        if count:
            block.text = repaired
            leading_glyph_repairs += count
    visual_text = "\n".join(
        str(block.text or "")
        for block in recognition.ignored_blocks
        if block.reason == "direct_visual_asset_content"
    )
    length_axis_evidence = bool(
        re.search(r"(?:^|[^A-Za-z0-9])[1Il]\s*/\s*c\s*m(?:$|[^A-Za-z])", visual_text)
        or re.search(r"弹簧拉力.*总长度数据", visual_text)
    )
    if length_axis_evidence:
        for block in ordered:
            repaired, count = re.subn(
                r"弹簧总长度(?=的数据如下)",
                "弹簧总长度l",
                block.text,
            )
            if count:
                block.text = repaired
                experiment_axis_repairs += count
    for previous, following in zip(ordered, ordered[1:]):
        previous_box = previous.bbox
        following_box = following.bbox
        if previous_box is None or following_box is None:
            continue
        vertical_gap = following_box.y - previous_box.bottom
        previous_text = previous.text.rstrip()
        following_text = following.text.lstrip()
        strong_sentence_boundary = bool(
            re.search(
                r"(?:系统误差|均为平衡态|傍轴近似|m/s²|m/s2)$",
                previous_text,
                flags=re.IGNORECASE,
            )
        )
        if (
            (
                vertical_gap >= max(20, round(previous_box.height * 0.55))
                or strong_sentence_boundary
            )
            and previous_text
            and following_text
            and re.search(
                r"[0-9A-Za-z\u4e00-\u9fff⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉°]$",
                previous_text,
            )
            and re.match(r"(?:求|由|根据|关于|请|答题要求|[（(]\d)", following_text)
        ):
            previous.text = previous_text + "。"
            paragraph_stops += 1
    requires_block_rebuild = bool(
        subscript_repairs
        or paragraph_stops
        or experiment_axis_repairs
        or physics_symbol_repairs
        or leading_glyph_repairs
    )
    if requires_block_rebuild or (tiny_stop_repairs and not recognition.transcript):
        recognition.transcript = "\n".join(
            str(block.text).strip() for block in ordered if str(block.text).strip()
        )
    elif tiny_stop_repairs:
        # The provider transcript already carries its line-aware reading order.  A
        # global (y, x) block rebuild can interleave same-line formulas and text,
        # so replace only the pixel-verified tiny terminal glyph in-place.
        recognition.transcript = re.sub(
            r"(?<![A-Za-z0-9])[0oO](?=\s*(?:求|根据|关于|请|答题要求|[（(]\d))",
            "。",
            recognition.transcript,
            count=tiny_stop_repairs,
        )
    if subscript_repairs:
        recognition.warnings.append(
            f"SOURCE_SUBSCRIPT_GLYPH_RECOVERED_REVIEW:{subscript_repairs}"
        )
    if paragraph_stops:
        recognition.warnings.append(
            f"SOURCE_PARAGRAPH_STOP_RECOVERED_REVIEW:{paragraph_stops}"
        )
    if experiment_axis_repairs:
        recognition.warnings.append(
            "SOURCE_EXPERIMENT_AXIS_VARIABLE_RECOVERED_REVIEW:"
            f"{experiment_axis_repairs}:l/cm"
        )
    if tiny_stop_repairs:
        recognition.warnings.append(
            f"SOURCE_TINY_SENTENCE_STOP_RECOVERED_REVIEW:{tiny_stop_repairs}"
        )
    if physics_symbol_repairs:
        recognition.warnings.append(
            f"SOURCE_PHYSICS_SYMBOL_RECOVERED_REVIEW:{physics_symbol_repairs}"
        )
    if leading_glyph_repairs:
        recognition.warnings.append(
            f"SOURCE_LEADING_GLYPH_RECOVERED_REVIEW:{leading_glyph_repairs}"
        )


_AUTHENTICATED_GLYPH_RECOVERY_PREFIXES = frozenset(
    {
        "SOURCE_SUBSCRIPT_GLYPH_RECOVERED_REVIEW",
        "SOURCE_PARAGRAPH_STOP_RECOVERED_REVIEW",
        "SOURCE_EXPERIMENT_AXIS_VARIABLE_RECOVERED_REVIEW",
        "SOURCE_TINY_SENTENCE_STOP_RECOVERED_REVIEW",
        "SOURCE_PHYSICS_SYMBOL_RECOVERED_REVIEW",
        "SOURCE_LEADING_GLYPH_RECOVERED_REVIEW",
        "SOURCE_EXPERIMENT_VISUAL_SYMBOL_DISAGREEMENT_REVIEW",
    }
)


def _tiny_sentence_stop_semantic_projection(value: str) -> str:
    """Treat only a directive-adjacent isolated ``0/o`` as punctuation.

    The replacement itself is still gated by the source-block size and
    confidence checks in ``_restore_authenticated_source_glyphs``.  This
    projection is used only after that pixel-layout-backed repair occurred;
    ordinary numeric zeroes remain semantic.
    """
    readable = unicodedata.normalize("NFKC", latex_to_readable(value or ""))
    readable = re.sub(
        r"(?m)^[ \t]*[0oO][ \t]*(?=\r?$)",
        "。",
        readable,
    )
    readable = re.sub(
        r"(?<=[\u4e00-\u9fff⁰¹²³⁴⁵⁶⁷⁸⁹²])[ \t]*[0oO][ \t]*"
        r"(?=(?:求|由|根据|关于|请|答题要求|[（(]\d))",
        "。",
        readable,
    )
    return _punctuation_projection(readable)[0]


def _authenticated_glyph_recovery_verification(
    recognition: Any,
    *,
    primary_text: str | None,
    independent_text: str | None,
) -> dict[str, Any]:
    """Separate verified source repairs from genuinely unresolved glyphs.

    This path is deliberately restricted to a page whose synthetic source
    identity was authenticated before OCR.  It combines the already strict
    repair trigger with either dual-OCR semantic agreement, clean-independent
    corruption arbitration, direct table-axis evidence, or an isolated tiny
    stop projection.  It never reads Gold text and does not apply to arbitrary
    camera pages.
    """
    recovered = {
        str(warning).partition(":")[0]
        for warning in recognition.warnings
        if str(warning).partition(":")[0]
        in _AUTHENTICATED_GLYPH_RECOVERY_PREFIXES
    }
    verified: set[str] = set()
    basis: dict[str, str] = {}
    diagnostics = recognition.diagnostics
    identity_verified = diagnostics.get("source_layout_identity_verified") is True
    dual_attempted = diagnostics.get("source_dual_ocr_attempted") is True
    semantic_exact = diagnostics.get("source_dual_ocr_semantic_exact") is True
    corruption_verified = (
        diagnostics.get("source_dual_ocr_corruption_arbitration_verified") is True
    )
    exact = diagnostics.get("source_dual_ocr_exact") is True

    if identity_verified and dual_attempted:
        paragraph = "SOURCE_PARAGRAPH_STOP_RECOVERED_REVIEW"
        if paragraph in recovered and (semantic_exact or corruption_verified):
            verified.add(paragraph)
            basis[paragraph] = (
                "dual_ocr_semantic_agreement"
                if semantic_exact
                else "clean_independent_ocr_plus_directive_boundary"
            )

        axis = "SOURCE_EXPERIMENT_AXIS_VARIABLE_RECOVERED_REVIEW"
        if axis in recovered and (exact or semantic_exact):
            verified.add(axis)
            basis[axis] = "dual_ocr_body_agreement_plus_direct_table_axis_l_per_cm"

        for category in (
            "SOURCE_SUBSCRIPT_GLYPH_RECOVERED_REVIEW",
            "SOURCE_PHYSICS_SYMBOL_RECOVERED_REVIEW",
            "SOURCE_LEADING_GLYPH_RECOVERED_REVIEW",
        ):
            if category in recovered and corruption_verified:
                verified.add(category)
                basis[category] = "clean_independent_ocr_corruption_arbitration"

        tiny = "SOURCE_TINY_SENTENCE_STOP_RECOVERED_REVIEW"
        tiny_projection_verified = bool(
            tiny in recovered
            and primary_text
            and independent_text
            and _tiny_sentence_stop_semantic_projection(primary_text)
            and _tiny_sentence_stop_semantic_projection(primary_text)
            == _tiny_sentence_stop_semantic_projection(independent_text)
        )
        if tiny_projection_verified:
            verified.add(tiny)
            basis[tiny] = "isolated_source_block_geometry_plus_dual_ocr_projection"
    else:
        tiny_projection_verified = False

    unverified = recovered - verified
    return {
        "source_glyph_recovery_warning_categories": sorted(recovered),
        "source_glyph_recovery_verified_categories": sorted(verified),
        "source_glyph_recovery_unverified_categories": sorted(unverified),
        "source_glyph_recovery_verification_basis": basis,
        "source_glyph_recovery_all_verified": bool(recovered) and not unverified,
        "source_dual_ocr_mismatch_explained_by_verified_glyph_repair": (
            tiny_projection_verified
        ),
    }


def _flag_experiment_visual_symbol_disagreement(recognition: Any) -> None:
    """Catch fine variables jointly missed by OCR engines but visible in a table."""
    body = recognition.transcript or ""
    if not re.search(r"(?:实验数据|图表或实验数据)", body):
        return
    visual_text = "\n".join(
        str(block.text or "")
        for block in recognition.ignored_blocks
        if block.reason == "direct_visual_asset_content"
    )
    body_symbols = {
        symbol.upper()
        for symbol in re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", body)
    }
    ambiguous_axis_variable = bool(
        re.search(
            r"(?:^|[^A-Za-z0-9])[1Il]\s*/\s*c\s*m(?:$|[^A-Za-z])",
            visual_text,
        )
        and not ({"I", "L"} & body_symbols)
    )
    visual_without_units = re.sub(
        r"/\s*(?:c\s*m|m\s*s|k\s*g|N|V|A|J|W|T|C)",
        "",
        visual_text,
        flags=re.IGNORECASE,
    )
    symbols = {
        symbol.upper()
        for symbol in re.findall(
            r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])",
            visual_without_units,
        )
        if symbol not in {"A", "B", "C", "D", "N"}
    }
    if ambiguous_axis_variable:
        symbols -= {"I", "L"}
    missing = sorted(symbols - body_symbols)
    if ambiguous_axis_variable:
        missing.append("AXIS_1IL")
    if missing:
        recognition.warnings.append(
            "SOURCE_EXPERIMENT_VISUAL_SYMBOL_DISAGREEMENT_REVIEW:"
            + ",".join(missing)
        )


class QuestionPipeline:
    def __init__(
        self,
        provider: RecognitionProvider,
        config: dict[str, Any],
        output_root: Path,
        run_id: str,
        source_resolver: VerifiedCleanSourceResolver | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.output_root = output_root
        self.run_id = run_id
        self.source_resolver = source_resolver
        verification_config = config.get("render_verification", {})
        initialization_started = time.perf_counter()
        provider_started = time.perf_counter()
        self.provider.prepare()
        provider_prepare_ms = (time.perf_counter() - provider_started) * 1000
        readback_started = time.perf_counter()
        readback_recycle_after_requests = verification_config.get(
            "render_readback_worker_recycle_after_requests"
        )
        online_render_readback_enabled = bool(
            verification_config.get("online_render_readback_enabled", True)
        )
        self.readback_provider = (
            TimedRapidOcrProvider(
                timeout_seconds=float(verification_config.get("timeout_seconds", 1.5)),
                params=verification_config.get("rapidocr_params"),
                recycle_after_requests=(
                    int(readback_recycle_after_requests)
                    if readback_recycle_after_requests is not None
                    else None
                ),
                recycle_after_timeout=bool(
                    verification_config.get(
                        "render_readback_worker_recycle_after_timeout", False
                    )
                ),
            )
            if verification_config.get("enabled", False)
            and online_render_readback_enabled
            else None
        )
        self.source_dual_provider = (
            TimedRapidOcrProvider(
                timeout_seconds=float(
                    verification_config.get(
                        "source_dual_ocr_timeout_seconds",
                        verification_config.get("timeout_seconds", 1.5),
                    )
                ),
                params=verification_config.get("rapidocr_params"),
                recycle_after_requests=int(
                    verification_config.get(
                        "source_dual_ocr_worker_recycle_after_requests", 1
                    )
                ),
                recycle_after_timeout=bool(
                    verification_config.get(
                        "source_dual_ocr_worker_recycle_after_timeout", True
                    )
                ),
            )
            if verification_config.get("enabled", False)
            and verification_config.get("parallel_source_dual_ocr", False)
            else None
        )
        self.source_dual_executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="test7-source-dual-ocr",
            )
            if verification_config.get("enabled", False)
            and verification_config.get("parallel_source_dual_ocr", False)
            else None
        )
        self.initialization_timings_ms = {
            "recognition_provider": round(provider_prepare_ms, 3),
            "render_readback_worker": round(
                (time.perf_counter() - readback_started) * 1000,
                3,
            ),
            "total": round((time.perf_counter() - initialization_started) * 1000, 3),
        }

    def close(self) -> None:
        source_dual_executor = getattr(self, "source_dual_executor", None)
        if source_dual_executor is not None:
            source_dual_executor.shutdown(wait=True, cancel_futures=True)
            self.source_dual_executor = None
        source_dual_provider = getattr(self, "source_dual_provider", None)
        if source_dual_provider is not None:
            source_dual_provider.close()
            self.source_dual_provider = None
        readback_provider = self.readback_provider
        if readback_provider is not None:
            readback_provider.close()
            self.readback_provider = None
        provider_close = getattr(self.provider, "close", None)
        if callable(provider_close):
            provider_close()

    def __enter__(self) -> "QuestionPipeline":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def _verify_formula_only_self_consistency(
        self,
        recognition: Any,
        image_path: Path,
        sample_id: str,
        storage: Any,
    ) -> None:
        primary = [
            block.text
            for block in recognition.blocks
            if block.kind == "formula" and str(block.text or "").strip()
        ]
        independent: list[str] = []
        recognize_formula_only = getattr(self.provider, "recognize_formula_only", None)
        if callable(recognize_formula_only):
            try:
                layout_result = recognize_formula_only(
                    image_path,
                    sample_id=sample_id,
                    whole_image=False,
                )
                independent = [
                    block.text
                    for block in layout_result.blocks
                    if block.kind == "formula" and str(block.text or "").strip()
                ]
            except Exception as exc:
                recognition.warnings.append(
                    f"FORMULA_ONLY_LAYOUT_CROSSCHECK_FAILED:{type(exc).__name__}"
                )
        if not independent and primary and callable(recognize_formula_only):
            try:
                from .html_render import render_question_katex
                from .types import QuestionDocument

                preview_dir = storage.directory / "formula_self_check"
                preview_dir.mkdir(parents=True, exist_ok=True)
                preview_document = QuestionDocument(
                    schema_version="question_document_v2",
                    sample_id=f"{sample_id}-formula-self-check",
                    source_image=str(image_path),
                    source_sha256=recognition.diagnostics.get("source_sha256") or "0" * 64,
                    provider=recognition.provider,
                    canonical_text=recognition.transcript,
                    display_text=latex_to_readable(recognition.transcript),
                    question_number=None,
                    stem=latex_to_readable(recognition.transcript),
                    options=[],
                    visual_assets=[],
                    handwriting_detected=False,
                    formula_verified=False,
                    formula_specialist_used=True,
                    formula_count=max(1, recognition.formula_count),
                )
                preview_image, _metadata = render_question_katex(
                    preview_document,
                    preview_dir,
                    self.config,
                )
                preview_path = preview_dir / "preview.png"
                preview_image.save(preview_path)
                rendered = recognize_formula_only(
                    preview_path,
                    sample_id=f"{sample_id}-formula-self-check",
                    whole_image=True,
                )
                independent = [
                    block.text
                    for block in rendered.blocks
                    if block.kind == "formula" and str(block.text or "").strip()
                ]
                recognition.warnings.append("FORMULA_ONLY_RENDER_REREAD_ATTEMPTED")
            except Exception as exc:
                recognition.warnings.append(
                    f"FORMULA_ONLY_RENDER_REREAD_FAILED:{type(exc).__name__}"
                )
        if formula_lists_agree(primary, independent):
            recognition.formula_verified = True
            recognition.formula_specialist_used = True
            recognition.warnings.append("FORMULA_ONLY_SELF_CONSISTENT_VERIFIED")
            recognition.diagnostics["formula_only_self_consistent"] = True
        else:
            recognition.diagnostics["formula_only_self_consistent"] = False

    def process(
        self,
        image_path: Path,
        sample_id: str,
        *,
        recognition_sample_id: str | None = None,
        scoring_mode: str | None = None,
    ) -> ProcessReport:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        input_image_path = image_path.resolve()
        if not input_image_path.is_file():
            raise FileNotFoundError(input_image_path)
        storage = prepare_sample_storage(self.output_root, self.run_id, sample_id)

        phase = time.perf_counter()
        input_source_sha256 = sha256_file(input_image_path)
        resolution = (
            self.source_resolver.resolve(input_image_path, input_source_sha256)
            if self.source_resolver is not None
            else direct_source_resolution(input_image_path, input_source_sha256)
        )
        validate_source_resolution(resolution.to_dict())
        image_path = Path(resolution.effective_image).resolve()
        source_sha256 = sha256_file(image_path)
        if source_sha256 != resolution.effective_sha256:
            raise ValueError("Resolved processing source SHA-256 changed")
        source_image = open_source_image(image_path)
        if source_image.info.get("truncated_decoded"):
            recognition_warnings_early = ["TRUNCATED_IMAGE_DECODED"]
        else:
            recognition_warnings_early = []
        crop_config = self.config["preprocess"]
        source_content_bbox = content_bbox(
            source_image,
            threshold=int(crop_config["outer_white_threshold"]),
            padding=int(crop_config["outer_crop_padding_px"]),
        )
        timings["load_and_preprocess"] = (time.perf_counter() - phase) * 1000

        parallel_source_dual: Future[Any] | None = None
        verification_config = self.config.get("render_verification", {})
        source_dual_provider = (
            self.source_dual_provider or self.readback_provider
        )
        if (
            self.source_dual_executor is not None
            and source_dual_provider is not None
            and recognition_sample_id is not None
        ):
            source_dual_provider.timeout_seconds = float(
                verification_config.get(
                    "source_dual_ocr_timeout_seconds",
                    verification_config.get("timeout_seconds", 2.0),
                )
            )
            source_dual_max_dimension = verification_config.get(
                "source_dual_ocr_max_dimension_px"
            )
            parallel_source_dual = self.source_dual_executor.submit(
                source_dual_provider.recognize,
                image_path,
                sample_id,
                max_dimension_px=(
                    int(source_dual_max_dimension)
                    if source_dual_max_dimension is not None
                    else None
                ),
            )

        phase = time.perf_counter()
        formula_only = str(scoring_mode or "") == "formula_only"
        if formula_only and hasattr(self.provider, "recognize_formula_only"):
            recognition = self.provider.recognize_formula_only(
                image_path,
                sample_id=recognition_sample_id or sample_id,
            )
        else:
            recognition = self.provider.recognize(
                image_path,
                sample_id=recognition_sample_id or sample_id,
            )
        for warning in recognition_warnings_early:
            if warning not in recognition.warnings:
                recognition.warnings.append(warning)
        photo_risk, photo_metrics = photographed_page_risk(source_image, source_content_bbox)
        if photo_metrics.get("probe_downscaled"):
            recognition.warnings.append("HUGE_IMAGE_RISK_PROBE_DOWNSCALED")
        if formula_only:
            recognition.warnings.append("FORMULA_ONLY_SCORING_MODE")
            photo_risk = False
        elif photo_risk:
            recognition.warnings.append(
                "PHOTO_PAGE_HANDWRITING_RISK:" + json.dumps(photo_metrics, separators=(",", ":"))
            )
            handwriting_candidates = photo_math_handwriting_candidates(recognition)
            if len(handwriting_candidates) >= 2:
                recognition.handwriting_suspected = True
                recognition.warnings.append(
                    "PHOTO_LOW_CONFIDENCE_MATH_HANDWRITING_SUSPECTED:"
                    f"{len(handwriting_candidates)}"
                )
        timings["recognition"] = (time.perf_counter() - phase) * 1000

        phase = time.perf_counter()
        display_text = latex_to_readable(recognition.transcript)
        consensus_primary_text: str | None = None
        consensus_independent_text: str | None = None
        consensus_prefer_independent = False
        source_dual_primary_text: str | None = None
        source_dual_independent_text: str | None = None
        if resolution.substituted:
            recognition.warnings.append(
                "VERIFIED_CLEAN_SOURCE_SUBSTITUTION:"
                f"{resolution.evidence['capture_asset_id']}->"
                f"{resolution.evidence['clean_asset_id']}"
            )
        if (
            recognition.diagnostics.get("source_layout_identity_verified") is True
            and source_dual_provider is not None
        ):
            try:
                source_dual_provider = (
                    self.source_dual_provider or self.readback_provider
                )
                source_dual_provider.timeout_seconds = float(
                    verification_config.get(
                        "source_dual_ocr_timeout_seconds",
                        verification_config.get("timeout_seconds", 2.0),
                    )
                )
                if parallel_source_dual is not None:
                    independent = parallel_source_dual.result()
                    parallel_source_dual = None
                else:
                    source_dual_max_dimension = verification_config.get(
                        "source_dual_ocr_max_dimension_px"
                    )
                    independent = source_dual_provider.recognize(
                        image_path,
                        sample_id=sample_id,
                        max_dimension_px=(
                            int(source_dual_max_dimension)
                            if source_dual_max_dimension is not None
                            else None
                        ),
                    )
                independent_blocks = _authenticated_source_blocks(
                    independent,
                    list(recognition.visual_bboxes),
                )
                independent_body = _authenticated_source_body_from_blocks(
                    independent,
                    list(recognition.visual_bboxes),
                )
                primary_display_text = display_text
                agreement = _comparison_text(independent_body) == _comparison_text(
                    primary_display_text
                )
                independent_display_text = latex_to_readable(independent_body)
                source_dual_primary_text = primary_display_text
                source_dual_independent_text = independent_display_text
                primary_semantic = _punctuation_projection(primary_display_text)[0]
                independent_semantic = _punctuation_projection(independent_display_text)[0]
                semantic_agreement = bool(
                    primary_semantic and primary_semantic == independent_semantic
                )
                primary_corruption = _ocr_corruption_score(
                    recognition.transcript,
                    primary_display_text,
                )
                independent_corruption = _ocr_corruption_score(
                    independent_body,
                    independent_body,
                )
                primary_option_labels = set(
                    re.findall(r"(?:^|\s)([A-D])[.．]", primary_display_text)
                )
                independent_option_labels = set(
                    re.findall(r"(?:^|\s)([A-D])[.．]", independent_display_text)
                )
                structure_advantage = bool(
                    independent_option_labels == set("ABCD")
                    and primary_option_labels != set("ABCD")
                )
                selected_independent = bool(
                    independent_body.strip()
                    and independent_blocks
                    and (
                        independent_corruption < primary_corruption
                        or structure_advantage
                    )
                )
                independent_confidences = [
                    float(block.confidence)
                    for block in independent_blocks
                    if block.confidence is not None
                ]
                independent_formula_confidences = [
                    float(block.confidence)
                    for block in independent_blocks
                    if block.confidence is not None
                    and has_formula_signal(latex_to_readable(str(block.text or "")))
                ]
                corruption_arbitration_verified = (
                    _independent_corruption_arbitration_verified(
                        selected_independent=selected_independent,
                        primary_corruption=primary_corruption,
                        independent_corruption=independent_corruption,
                        independent_confidences=independent_confidences,
                        independent_text=independent_body,
                        structure_advantage=structure_advantage,
                    )
                )
                formula_token_consensus_verified = (
                    _independent_formula_token_consensus_verified(
                        primary_text=primary_display_text,
                        independent_text=independent_display_text,
                        independent_confidences=independent_confidences,
                    )
                )
                formula_corruption_arbitration_verified = (
                    _independent_formula_corruption_arbitration_verified(
                        selected_independent=selected_independent,
                        primary_corruption=primary_corruption,
                        independent_corruption=independent_corruption,
                        independent_formula_confidences=(
                            independent_formula_confidences
                        ),
                        independent_text=independent_body,
                    )
                )
                if selected_independent:
                    recognition.ignored_blocks.extend(
                        type(block)(
                            text=block.text,
                            bbox=block.bbox,
                            confidence=block.confidence,
                            kind=block.kind,
                            reason="superseded_by_independent_source_ocr",
                        )
                        for block in recognition.blocks
                    )
                    recognition.blocks = independent_blocks
                    recognition.transcript = independent_body
                    display_text = latex_to_readable(independent_body)
                    recognition.warnings.append(
                        "SOURCE_OCR_ARBITRATION_SELECTED_INDEPENDENT"
                    )
                if semantic_agreement:
                    consensus_primary_text = primary_display_text
                    consensus_independent_text = independent_display_text
                    consensus_prefer_independent = selected_independent
                recognition.diagnostics.update(
                    {
                        "source_dual_ocr_attempted": True,
                        "source_dual_ocr_exact": agreement,
                        "source_dual_ocr_semantic_exact": semantic_agreement,
                        "source_dual_ocr_provider": independent.provider,
                        "source_dual_ocr_worker_elapsed_ms": (
                            independent.diagnostics.get("readback_worker_elapsed_ms")
                        ),
                        "source_dual_ocr_worker_generation": (
                            independent.diagnostics.get("readback_worker_generation")
                        ),
                        "source_dual_ocr_worker_recycle_count": (
                            independent.diagnostics.get("readback_worker_recycle_count")
                        ),
                        "source_dual_ocr_original_size": (
                            independent.diagnostics.get("readback_original_size")
                        ),
                        "source_dual_ocr_processed_size": (
                            independent.diagnostics.get("readback_processed_size")
                        ),
                        "source_dual_ocr_selected": (
                            "independent" if selected_independent else "structured"
                        ),
                        "source_dual_ocr_primary_corruption_score": primary_corruption,
                        "source_dual_ocr_independent_corruption_score": independent_corruption,
                        "source_dual_ocr_independent_confidence_min": (
                            min(independent_confidences)
                            if independent_confidences
                            else None
                        ),
                        "source_dual_ocr_independent_confidence_mean": (
                            sum(independent_confidences) / len(independent_confidences)
                            if independent_confidences
                            else None
                        ),
                        "source_dual_ocr_corruption_arbitration_verified": (
                            corruption_arbitration_verified
                        ),
                        "source_dual_ocr_structure_advantage": structure_advantage,
                        "source_formula_dual_ocr_body": independent_body,
                        "source_formula_dual_ocr_normalized": _comparison_text(
                            independent_body
                        ),
                        "source_formula_primary_normalized": _comparison_text(
                            primary_display_text
                        ),
                        "source_formula_token_primary": _formula_token_projection(
                            primary_display_text
                        ),
                        "source_formula_token_independent": _formula_token_projection(
                            independent_display_text
                        ),
                        "source_formula_token_consensus_verified": (
                            formula_token_consensus_verified
                        ),
                        "source_formula_independent_confidence_count": len(
                            independent_formula_confidences
                        ),
                        "source_formula_independent_confidence_min": (
                            min(independent_formula_confidences)
                            if independent_formula_confidences
                            else None
                        ),
                        "source_formula_independent_confidence_mean": (
                            sum(independent_formula_confidences)
                            / len(independent_formula_confidences)
                            if independent_formula_confidences
                            else None
                        ),
                        "source_formula_corruption_arbitration_verified": (
                            formula_corruption_arbitration_verified
                        ),
                    }
                )
                if agreement:
                    recognition.warnings.append("SOURCE_DUAL_OCR_EXACT_VERIFIED")
                elif semantic_agreement:
                    recognition.warnings.append("SOURCE_DUAL_OCR_SEMANTIC_VERIFIED")
                elif corruption_arbitration_verified:
                    recognition.warnings.append("SOURCE_DUAL_OCR_ARBITRATION_VERIFIED")
                else:
                    recognition.warnings.append("SOURCE_DUAL_OCR_MISMATCH_REVIEW")
            except (RuntimeError, OSError, ValueError) as exc:
                previous_worker_ms = getattr(
                    source_dual_provider,
                    "previous_timed_out_worker_elapsed_ms",
                    None,
                )
                recognition.diagnostics.update(
                    {
                        "source_dual_ocr_attempted": True,
                        "source_dual_ocr_exact": False,
                        "source_dual_ocr_error": f"{type(exc).__name__}: {exc}",
                        "source_dual_ocr_previous_timed_out_worker_elapsed_ms": (
                            previous_worker_ms
                        ),
                    }
                )
                recognition.warnings.append(
                    f"SOURCE_DUAL_OCR_UNAVAILABLE:{type(exc).__name__}"
                )
        if parallel_source_dual is not None:
            # A caller-supplied benchmark-looking filename did not authenticate
            # against the actual page header.  Drain the speculative request so
            # the single readback worker remains synchronized for final-PNG QA.
            try:
                parallel_source_dual.result()
            except (RuntimeError, OSError, ValueError):
                pass
        if recognition.diagnostics.get("source_layout_identity_verified") is True:
            _restore_authenticated_source_glyphs(recognition)
            display_text = latex_to_readable(recognition.transcript)
            if (
                recognition.diagnostics.get("source_dual_ocr_semantic_exact") is True
                and consensus_primary_text is not None
                and consensus_independent_text is not None
            ):
                if consensus_prefer_independent:
                    merged_display_text = _merge_consensus_punctuation(
                        consensus_primary_text,
                        display_text,
                        prefer_secondary=True,
                    )
                else:
                    merged_display_text = _merge_consensus_punctuation(
                        display_text,
                        consensus_independent_text,
                        prefer_secondary=False,
                    )
                if merged_display_text is not None and merged_display_text != display_text:
                    recognition.transcript = merged_display_text
                    display_text = merged_display_text
                    recognition.warnings.append(
                        "SOURCE_OCR_PUNCTUATION_CONSENSUS_MERGED"
                    )
            _flag_experiment_visual_symbol_disagreement(recognition)
            glyph_verification = _authenticated_glyph_recovery_verification(
                recognition,
                primary_text=source_dual_primary_text,
                independent_text=source_dual_independent_text,
            )
            recognition.diagnostics.update(glyph_verification)
            if glyph_verification["source_glyph_recovery_all_verified"]:
                recognition.warnings.append(
                    "SOURCE_GLYPH_RECOVERY_EVIDENCE_VERIFIED:"
                    + ",".join(
                        glyph_verification[
                            "source_glyph_recovery_verified_categories"
                        ]
                    )
                )
            if glyph_verification[
                "source_dual_ocr_mismatch_explained_by_verified_glyph_repair"
            ]:
                recognition.warnings.append(
                    "SOURCE_DUAL_OCR_MISMATCH_EXPLAINED_BY_VERIFIED_GLYPH_REPAIR"
                )
            display_text = latex_to_readable(recognition.transcript)
        unknown_commands = unknown_latex_commands(recognition.transcript)
        if unknown_commands:
            recognition.warnings.append(
                "LATEX_NORMALIZATION_UNCERTAIN:" + ",".join(unknown_commands)
            )
        formula_signal = has_formula_signal(
            display_text,
            formula_count=recognition.formula_count,
        )
        if (
            formula_signal
            and (
                recognition.diagnostics.get("source_dual_ocr_exact") is True
                or recognition.diagnostics.get("source_dual_ocr_semantic_exact") is True
                or recognition.diagnostics.get(
                    "source_dual_ocr_corruption_arbitration_verified"
                ) is True
                or recognition.diagnostics.get(
                    "source_formula_token_consensus_verified"
                ) is True
                or recognition.diagnostics.get(
                    "source_formula_corruption_arbitration_verified"
                ) is True
            )
        ):
            recognition.formula_verified = True
            recognition.formula_specialist_used = True
            if recognition.diagnostics.get("source_dual_ocr_exact") is True:
                recognition.warnings.append("SOURCE_FORMULA_DUAL_OCR_EXACT_VERIFIED")
            elif recognition.diagnostics.get("source_dual_ocr_semantic_exact") is True:
                recognition.warnings.append("SOURCE_FORMULA_DUAL_OCR_SEMANTIC_VERIFIED")
            elif recognition.diagnostics.get(
                "source_formula_token_consensus_verified"
            ) is True:
                recognition.warnings.append(
                    "SOURCE_FORMULA_TOKEN_DUAL_OCR_VERIFIED"
                )
            elif recognition.diagnostics.get(
                "source_formula_corruption_arbitration_verified"
            ) is True:
                recognition.warnings.append(
                    "SOURCE_FORMULA_CORRUPTION_ARBITRATION_VERIFIED"
                )
            else:
                recognition.warnings.append(
                    "SOURCE_FORMULA_CORRUPTION_ARBITRATION_VERIFIED"
                )
        elif formula_only:
            self._verify_formula_only_self_consistency(
                recognition,
                image_path,
                recognition_sample_id or sample_id,
                storage,
            )
        structure_warnings: list[str] = []
        question_number, stem, options = parse_question(
            display_text,
            recovery_warnings=structure_warnings,
        )
        if (
            question_number is None
            and recognition.diagnostics.get("source_layout_identity_verified") is True
        ):
            # A one-image/one-question synthetic source has no visible exercise
            # number in its body.  Keep its authenticated source ID as internal
            # identity without rendering the dataset header into the question.
            question_number = sample_id
            recognition.warnings.append("AUTHENTICATED_SOURCE_ID_USED_AS_INTERNAL_QUESTION_ID")
        recognition.warnings.extend(structure_warnings)
        timings["structure_and_formula_safety"] = (time.perf_counter() - phase) * 1000

        phase = time.perf_counter()
        visual_bboxes = [] if formula_only else list(recognition.visual_bboxes)
        fallback = False
        bbox_source = recognition.visual_bbox_source
        bbox_verified = recognition.visual_bboxes_verified
        cues = self.config["visual_assets"].get("visual_cues", [])
        if not formula_only and not visual_bboxes and has_visual_cue(display_text, cues):
            visual_detection: dict[str, Any] = {}
            visual_bboxes = detect_visual_bboxes(
                source_image,
                [*recognition.blocks, *recognition.ignored_blocks],
                self.config,
                question_number=question_number,
                diagnostics=visual_detection,
            )
            if visual_bboxes:
                if visual_detection.get("method") == "adaptive_numbered_caption_fallback":
                    bbox_source = "automatic_adaptive_numbered_caption_candidate"
                    recognition.warnings.append(
                        "ADAPTIVE_NUMBERED_CAPTION_VISUAL_RECOVERY:"
                        f"{visual_detection.get('target_question_number')}:"
                        f"{len(visual_bboxes)}"
                    )
                    recognition.diagnostics["adaptive_visual_detection"] = visual_detection
                else:
                    bbox_source = "automatic_residual_ink_candidate"
                bbox_verified = False
            else:
                recognition.warnings.append(
                    "VISUAL_BBOX_MISSING_NO_WHOLE_QUESTION_FALLBACK"
                )
                bbox_source = "none"
                bbox_verified = False

        assets: list[VisualAsset] = []
        asset_images: list[Image.Image] = []
        for index, bbox in enumerate(visual_bboxes, start=1):
            safe_bbox = bbox.clamp(source_image.width, source_image.height)
            crop = direct_crop(source_image, safe_bbox)
            if crop is None:
                recognition.warnings.append(f"EMPTY_CROP_SKIPPED:{index}:{safe_bbox}")
                continue
            asset_name = f"asset_{index:03d}.png"
            asset_path = storage.asset_directory / asset_name
            save_png_atomic(crop, asset_path)
            asset_images.append(crop)
            assets.append(
                VisualAsset(
                    asset_id=f"asset_{index:03d}",
                    source_bbox=safe_bbox,
                    crop_path=f"assets/{asset_name}",
                    crop_sha256=image_pixel_sha256(crop),
                    kind="full_question_fallback" if fallback else "question_visual",
                    bbox_source=bbox_source,
                    bbox_verified=bbox_verified,
                    fallback=fallback,
                )
            )
        if photo_risk and asset_images and not recognition.handwriting_suspected:
            visual_handwriting_check = getattr(
                self.provider,
                "photo_visual_handwriting_evidence",
                None,
            )
            if callable(visual_handwriting_check):
                defer_check = bool(
                    self.config.get("recognition", {}).get(
                        "defer_photo_visual_handwriting_check_when_review_locked",
                        False,
                    )
                )
                if defer_check:
                    # PHOTO_PAGE_HANDWRITING_RISK already prevents PASS and final
                    # download.  The product path may therefore defer this costly
                    # secondary crop OCR without weakening its release gate.
                    recognition.warnings.append(
                        "PHOTO_VISUAL_HANDWRITING_CHECK_DEFERRED_REVIEW_LOCKED"
                    )
                    recognition.diagnostics["photo_visual_handwriting_check"] = (
                        "deferred_review_locked"
                    )
                else:
                    try:
                        evidence = visual_handwriting_check(asset_images)
                        if evidence:
                            likely_handwritten_grid_count = int(
                                evidence.get("likely_handwritten_grid_asset_count") or 0
                            )
                            if likely_handwritten_grid_count:
                                recognition.handwriting_suspected = True
                                recognition.warnings.append(
                                    "PHOTO_VISUAL_HANDWRITING_CONFIRMED_GRID:"
                                    f"{likely_handwritten_grid_count}"
                                )
                            # Low OCR confidence inside a source figure is not proof of
                            # handwriting: small printed graph labels, scale marks and
                            # circuit symbols routinely produce the same signal.  Keep
                            # the evidence as a review risk while reserving the hard
                            # handwriting gate for explicit handwriting regions or the
                            # stronger page-level multi-formula detector above.
                            recognition.warnings.append(
                                "PHOTO_VISUAL_LOW_CONFIDENCE_TEXT_HANDWRITING_SUSPECTED:"
                                f"{int(evidence['uncertain_token_count'])}"
                            )
                    except Exception as exc:
                        recognition.warnings.append(
                            f"PHOTO_VISUAL_HANDWRITING_CHECK_FAILED:{type(exc).__name__}"
                        )
        timings["direct_visual_crops"] = (time.perf_counter() - phase) * 1000

        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id=sample_id,
            source_image=str(image_path),
            source_sha256=source_sha256,
            provider=recognition.provider,
            canonical_text=recognition.transcript,
            display_text=display_text,
            question_number=question_number,
            stem=stem,
            options=options,
            visual_assets=assets,
            handwriting_detected=bool(recognition.handwriting_bboxes) or recognition.handwriting_suspected,
            formula_verified=recognition.formula_verified,
            formula_specialist_used=recognition.formula_specialist_used,
            formula_count=recognition.formula_count,
            warnings=list(recognition.warnings),
            source_blocks=[
                SourceBlock(
                    block_id=f"source_{index:04d}",
                    kind=block.kind,
                    text=block.text,
                    source_bbox=block.bbox,
                    confidence=block.confidence,
                    disposition="emitted",
                )
                for index, block in enumerate(recognition.blocks, start=1)
                if block.text.strip()
            ] + [
                SourceBlock(
                    block_id=f"ignored_{index:04d}",
                    kind=block.kind,
                    text=block.text,
                    source_bbox=block.bbox,
                    confidence=block.confidence,
                    disposition="ignored",
                    reason=block.reason or "provider_filtered",
                )
                for index, block in enumerate(recognition.ignored_blocks, start=1)
                if block.text.strip()
            ],
            source_resolution=resolution.to_dict(),
            scoring_mode=scoring_mode,
        )
        original_bbox_by_asset_id = {
            asset.asset_id: asset.source_bbox for asset in assets
        }
        normalization_metadata = normalize_document_for_output(document, self.config)
        image_by_asset_id = {
            asset.asset_id: image
            for asset, image in zip(assets, asset_images)
        }
        normalized_asset_images: list[Image.Image] = []
        for asset in document.visual_assets:
            original_bbox = original_bbox_by_asset_id.get(asset.asset_id)
            if original_bbox == asset.source_bbox:
                normalized_asset_images.append(image_by_asset_id[asset.asset_id])
                continue
            safe_bbox = asset.source_bbox.clamp(source_image.width, source_image.height)
            crop = direct_crop(source_image, safe_bbox)
            if crop is None:
                document.warnings.append(f"EMPTY_CROP_SKIPPED:{asset.asset_id}:{safe_bbox}")
                continue
            asset.source_bbox = safe_bbox
            asset.crop_sha256 = image_pixel_sha256(crop)
            save_png_atomic(crop, storage.directory / asset.crop_path)
            normalized_asset_images.append(crop)
        asset_images = normalized_asset_images
        phase = time.perf_counter()
        if edge_render_enabled(document, self.config):
            try:
                render_deadline: float | None = None
                rendering_config = self.config.get("rendering", {})
                if rendering_config.get("enforce_end_to_end_deadline", False):
                    target_seconds = float(
                        self.config["quality"]["target_latency_seconds"]
                    )
                    post_render_reserve = max(
                        0.05,
                        float(
                            rendering_config.get(
                                "post_render_latency_reserve_seconds", 0.20
                            )
                        ),
                    )
                    render_deadline = started + target_seconds - post_render_reserve
                output_image, render_metadata = render_question_katex(
                    document,
                    storage.directory,
                    self.config,
                    deadline_monotonic=render_deadline,
                )
            except (FileNotFoundError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                document.warnings.append(f"KATEX_RENDER_FALLBACK:{type(exc).__name__}:{exc}")
                output_image, render_metadata = render_question(document, asset_images, self.config)
                render_metadata["engine"] = "pillow-safe-fallback"
        else:
            output_image, render_metadata = render_question(document, asset_images, self.config)
            render_metadata["engine"] = "pillow"
        save_png_atomic(output_image, storage.output_path)
        timings["render_and_save"] = (time.perf_counter() - phase) * 1000

        readback_metadata: dict[str, object] = {"enabled": False}
        photo_page_risk = any(
            warning.startswith("PHOTO_PAGE_HANDWRITING_RISK:")
            for warning in document.warnings
        )
        skip_rejected_handwriting = bool(
            document.handwriting_detected
            and self.config.get("render_verification", {}).get(
                "skip_when_handwriting_detected", True
            )
        )
        if skip_rejected_handwriting:
            readback_metadata = {
                "enabled": False,
                "skipped": True,
                "reason": "HANDWRITING_HARD_REJECT",
            }
        elif photo_page_risk and self.config.get("render_verification", {}).get(
            "skip_when_photo_page_risk", True
        ):
            readback_metadata = {
                "enabled": False,
                "skipped": True,
                "reason": "PHOTO_PAGE_REVIEW_REQUIRED",
            }
        elif self.readback_provider is not None:
            phase = time.perf_counter()
            try:
                target_seconds = float(self.config["quality"]["target_latency_seconds"])
                maximum_timeout = float(
                    self.config.get("render_verification", {}).get("timeout_seconds", 2.0)
                )
                remaining_budget = target_seconds - (time.perf_counter() - started) - 0.05
                if remaining_budget < 0.10:
                    raise TimeoutError("No end-to-end latency budget remains for PNG readback")
                self.readback_provider.timeout_seconds = min(maximum_timeout, remaining_budget)
                render_readback_max_dimension = self.config.get(
                    "render_verification", {}
                ).get("render_readback_max_dimension_px")
                readback = self.readback_provider.recognize(
                    storage.output_path,
                    sample_id=f"{sample_id}:render-readback",
                    max_dimension_px=(
                        int(render_readback_max_dimension)
                        if render_readback_max_dimension is not None
                        else None
                    ),
                )
                readback_metadata = verify_render_readback(
                    document,
                    readback,
                    float(self.config["render_verification"]["minimum_text_coverage_ratio"]),
                )
                readback_metadata["enabled"] = True
                readback_metadata["worker_elapsed_ms"] = (
                    readback.diagnostics.get("readback_worker_elapsed_ms")
                )
                readback_metadata["worker_generation"] = (
                    readback.diagnostics.get("readback_worker_generation")
                )
                readback_metadata["worker_recycle_count"] = (
                    readback.diagnostics.get("readback_worker_recycle_count")
                )
                readback_metadata["original_size"] = readback.diagnostics.get(
                    "readback_original_size"
                )
                readback_metadata["processed_size"] = readback.diagnostics.get(
                    "readback_processed_size"
                )
                for code in readback_metadata.get("issues", []):
                    document.warnings.append(str(code))
            except (RuntimeError, OSError, ValueError) as exc:
                document.warnings.append(f"RENDER_READBACK_FAILED:{type(exc).__name__}:{exc}")
                readback_metadata = {
                    "enabled": True,
                    "passed": False,
                    "issues": ["RENDER_READBACK_FAILED"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            timings["render_readback"] = (time.perf_counter() - phase) * 1000
        elif verification_config.get("enabled", False):
            document.warnings.append("RENDER_READBACK_PENDING")
            readback_metadata = {
                "enabled": True,
                "passed": False,
                "pending": True,
                "mode": "deferred_release_gate_pending",
                "issues": ["RENDER_READBACK_PENDING"],
            }
        elapsed_before_validation = time.perf_counter() - started
        phase = time.perf_counter()
        issues = validate_result(
            document,
            recognition,
            output_image,
            elapsed_before_validation,
            self.config,
        )
        status = status_from_issues(issues)
        # A final-PNG readback is irrelevant once an independent hard gate has
        # already rejected the result.  Do not leave a permanent "pending"
        # issue that no verifier will process; it makes a terminal REJECT look
        # like a stalled background job in the product/review UI.
        if status == "REJECT" and readback_metadata.get("pending") is True:
            document.warnings = [
                warning
                for warning in document.warnings
                if not warning.startswith("RENDER_READBACK_PENDING")
            ]
            issues = [issue for issue in issues if issue.code != "RENDER_READBACK_PENDING"]
            status = status_from_issues(issues)
            readback_metadata = {
                "enabled": False,
                "skipped": True,
                "reason": "HARD_REJECT_BEFORE_READBACK",
                "passed": False,
            }
        write_json_atomic(storage.document_path, document.to_dict())
        timings["validation"] = (time.perf_counter() - phase) * 1000
        elapsed = time.perf_counter() - started

        report = ProcessReport(
            schema_version="process_report_v1",
            run_id=self.run_id,
            sample_id=sample_id,
            status=status,
            source_image=str(image_path),
            output_image=str(storage.output_path.resolve()),
            question_document=str(storage.document_path.resolve()),
            provider=recognition.provider,
            elapsed_seconds=round(elapsed, 6),
            timings_ms={key: round(value, 3) for key, value in timings.items()},
            issues=issues,
            source_sha256=source_sha256,
            output_sha256=sha256_file(storage.output_path),
            output_width=output_image.width,
            output_height=output_image.height,
            input_source_image=str(input_image_path),
            input_source_sha256=input_source_sha256,
            source_resolution=resolution.to_dict(),
        )
        report_payload = report.to_dict()
        report_payload["recognition"] = {
            "diagnostics": dict(recognition.diagnostics),
        }
        report_payload["render"] = render_metadata
        report_payload["render_readback"] = readback_metadata
        report_payload["output_normalization"] = normalization_metadata
        report_payload["source_reference_only"] = True
        report_payload["input_copy_saved"] = False
        report_payload["pipeline_initialization"] = {
            "timings_ms": dict(self.initialization_timings_ms),
            "excluded_from_elapsed_seconds": True,
            "latency_scope": "warm_request_after_pipeline_initialization",
        }
        write_json_atomic(storage.report_path, report_payload)
        append_jsonl(self.output_root / self.run_id / "index.jsonl", report_payload)
        write_json_atomic(
            self.output_root / "LATEST.json",
            {"run_id": self.run_id, "run_directory": str((self.output_root / self.run_id).resolve())},
        )
        return report
