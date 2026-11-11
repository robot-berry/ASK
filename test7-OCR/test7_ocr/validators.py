from __future__ import annotations

import re
from typing import Any

from PIL import Image, ImageStat

from .text import contains_latex_source, has_formula_signal, has_visual_cue
from .types import QuestionDocument, RecognitionResult, Status, ValidationIssue


def _issue(code: str, severity: str, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, severity=severity, message=message)  # type: ignore[arg-type]


def _has_unlabeled_formula_option_grid(document: QuestionDocument) -> bool:
    if document.options or not re.search(
        r"(?:下列|正确|距离为|值为|结果为|等于|选择)",
        document.display_text,
    ):
        return False
    text_blocks = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "text"
        and block.source_bbox is not None
        and re.search(r"[\u4e00-\u9fff]", block.text or "")
    ]
    if not text_blocks:
        return False
    prose_bottom = max(block.source_bbox.bottom for block in text_blocks if block.source_bbox)
    formulas = [
        block
        for block in document.source_blocks
        if block.disposition == "emitted"
        and block.kind == "formula"
        and block.source_bbox is not None
        and block.source_bbox.y >= prose_bottom + 10
    ]
    if len(formulas) < 4:
        return False
    formulas.sort(key=lambda block: block.source_bbox.x + block.source_bbox.width / 2)
    centers = [
        block.source_bbox.x + block.source_bbox.width / 2 for block in formulas
    ]
    split_index = max(
        range(1, len(centers)),
        key=lambda index: centers[index] - centers[index - 1],
    )
    if centers[split_index] - centers[split_index - 1] < 120:
        return False
    left = formulas[:split_index]
    right = formulas[split_index:]
    if len(left) < 2 or len(right) < 2:
        return False
    matched_rows = 0
    unused_right = list(right)
    for left_block in sorted(left, key=lambda block: block.source_bbox.y):
        left_center = left_block.source_bbox.y + left_block.source_bbox.height / 2
        candidates = [
            (
                abs(
                    left_center
                    - (right_block.source_bbox.y + right_block.source_bbox.height / 2)
                ),
                right_block,
            )
            for right_block in unused_right
        ]
        if not candidates:
            continue
        distance, right_block = min(candidates, key=lambda item: item[0])
        tolerance = max(
            35,
            round((left_block.source_bbox.height + right_block.source_bbox.height) * 0.5),
        )
        if distance <= tolerance:
            matched_rows += 1
            unused_right.remove(right_block)
    return matched_rows >= 2


def _requires_complete_four_option_choice(document: QuestionDocument) -> bool:
    compact = re.sub(r"\s+", "", document.display_text)
    return bool(
        re.search(
            r"下列.{0,18}(?:说法|判断|选项|结论)?.{0,8}"
            r"(?:正确|错误|不正确|不错误|符合).{0,4}(?:的是|是)?",
            compact,
        )
    )


def _formula_only_scoring(document: QuestionDocument, config: dict[str, Any]) -> bool:
    mode = document.scoring_mode or (config.get("scoring") or {}).get("mode")
    return str(mode or "") == "formula_only"


def validate_result(
    document: QuestionDocument,
    recognition: RecognitionResult,
    output_image: Image.Image,
    elapsed_seconds: float,
    config: dict[str, Any],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    quality = config["quality"]
    expected_width = int(config["output"]["width_px"])
    minimum_chars = int(quality["minimum_text_characters"])
    formula_only = _formula_only_scoring(document, config)

    if output_image.width != expected_width:
        issues.append(_issue("OUTPUT_WIDTH", "reject", f"Expected {expected_width}px, got {output_image.width}px"))
    if not formula_only and len(document.display_text.strip()) < minimum_chars:
        issues.append(_issue("EMPTY_OR_TOO_SHORT_TEXT", "reject", "OCR text is empty or implausibly short"))
    if (
        not formula_only
        and quality.get("question_number_required", True)
        and document.question_number is None
    ):
        issues.append(
            _issue(
                "QUESTION_NUMBER_MISSING",
                "reject",
                "The target question number was not preserved in the reflowed result",
            )
        )
    recovered_caption_numbers = [
        warning.partition(":")[2]
        for warning in document.warnings
        if warning.startswith("QUESTION_NUMBER_RECOVERED_FROM_VISUAL_CAPTION:")
    ]
    if recovered_caption_numbers and not formula_only and not re.match(
        rf"^\s*(?:第\s*)?{re.escape(recovered_caption_numbers[-1])}\s*[.．、)]",
        document.display_text,
    ):
        issues.append(
            _issue(
                "QUESTION_NUMBER_RECOVERY_NOT_RENDERED",
                "reject",
                "A caption-derived question number exists in metadata but is absent from the rendered text",
            )
        )
    if contains_latex_source(document.display_text):
        severity = "reject" if quality.get("latex_leakage_is_reject", True) else "review"
        issues.append(_issue("LATEX_SOURCE_LEAKAGE", severity, "Rendered text still contains LaTeX source syntax"))

    compact_tail = re.sub(r"\s+", "", document.display_text).rstrip("。．.;；，,")
    explicit_answer_slot = bool(
        re.search(r"(?:_{2,}|—{2,}|…{2,}|□|\[\s*填?空?\s*\]|[（(]\s*[）)])", document.display_text)
    )
    if not explicit_answer_slot and re.search(
        r"(?:答案|结果|数值|大小|估计值|总体方差)?(?:为|是|等于|填|填入|选择)[：:]?$",
        compact_tail,
    ):
        issues.append(
            _issue(
                "ANSWER_SLOT_UNVERIFIED",
                "review",
                "Question ends at an answer cue without a preserved blank, formula, or option",
            )
        )

    if recognition.unverified_prelabel and quality.get("unverified_prelabel_is_review", True):
        issues.append(_issue("UNVERIFIED_PRELABEL", "review", "Candidate OCR was used only as a pipeline fixture"))
    if document.handwriting_detected:
        severity = "reject" if quality.get("handwriting_present_is_reject", True) else "review"
        issues.append(
            _issue(
                "HANDWRITING_PRESENT",
                severity,
                "Detected handwriting has not been proven absent from the reflowed question",
            )
        )
    elif any(warning.startswith("PHOTO_PAGE_HANDWRITING_RISK:") for warning in document.warnings):
        issues.append(
            _issue(
                "PHOTO_PAGE_HANDWRITING_RISK",
                "review",
                "Photographed-page appearance requires a handwriting absence check",
            )
        )
    photo_visual_structure_recovery = (
        bool(document.visual_assets)
        and any(
            warning.startswith("PHOTO_PAGE_HANDWRITING_RISK:")
            for warning in document.warnings
        )
        and any(
            warning.startswith((
                "MISSING_TWO_COLUMN_OPTION_LABEL_RESTORED:",
                "OPTION_SEQUENCE_RECOVERED_FROM_COHERENT_LABELS:",
                "QUESTION_NUMBER_PUNCTUATION_RECOVERED:",
                "QUESTION_NUMBER_RECOVERED_FROM_VISUAL_CAPTION:",
            ))
            for warning in document.warnings
        )
        and any(not asset.bbox_verified for asset in document.visual_assets)
    )
    if photo_visual_structure_recovery:
        issues.append(
            _issue(
                "PHOTO_VISUAL_STRUCTURE_RECOVERY_UNVERIFIED",
                "reject",
                "A photographed page combines automatic structure recovery with an "
                "unverified source crop; require clean-source, mask, or bbox evidence",
            )
        )
    if any(
        warning.startswith("PHOTO_VISUAL_LOW_CONFIDENCE_TEXT_HANDWRITING_SUSPECTED:")
        for warning in document.warnings
    ):
        issues.append(
            _issue(
                "PHOTO_VISUAL_HANDWRITING_RISK",
                "review",
                "Low-confidence compact text was found inside a source figure; "
                "this is ambiguous evidence, not confirmed handwriting",
            )
        )
    formula_signal = has_formula_signal(
        document.display_text,
        formula_count=max(document.formula_count, recognition.formula_count),
    )
    if formula_signal and not document.formula_verified:
        issues.append(_issue("FORMULA_UNVERIFIED", "review", "Formula-bearing content has not passed an independent formula verification stage"))
    if any(
        warning.startswith((
            "SOURCE_DUAL_OCR_MISMATCH_REVIEW",
            "SOURCE_DUAL_OCR_UNAVAILABLE:",
        ))
        for warning in document.warnings
    ) and not formula_signal and not recognition.diagnostics.get(
        "source_dual_ocr_mismatch_explained_by_verified_glyph_repair"
    ):
        issues.append(
            _issue(
                "SOURCE_TEXT_DUAL_OCR_UNVERIFIED",
                "review",
                "The independent source OCR did not exactly confirm the selected printed text",
            )
        )
    glyph_recovery_prefixes = {
        "SOURCE_SUBSCRIPT_GLYPH_RECOVERED_REVIEW",
        "SOURCE_PARAGRAPH_STOP_RECOVERED_REVIEW",
        "SOURCE_EXPERIMENT_AXIS_VARIABLE_RECOVERED_REVIEW",
        "SOURCE_TINY_SENTENCE_STOP_RECOVERED_REVIEW",
        "SOURCE_PHYSICS_SYMBOL_RECOVERED_REVIEW",
        "SOURCE_LEADING_GLYPH_RECOVERED_REVIEW",
        "SOURCE_EXPERIMENT_VISUAL_SYMBOL_DISAGREEMENT_REVIEW",
    }
    recovered_glyph_categories = {
        warning.partition(":")[0]
        for warning in document.warnings
        if warning.partition(":")[0] in glyph_recovery_prefixes
    }
    verified_glyph_categories = {
        str(category)
        for category in recognition.diagnostics.get(
            "source_glyph_recovery_verified_categories", []
        )
    }
    if recovered_glyph_categories - verified_glyph_categories:
        issues.append(
            _issue(
                "SOURCE_GLYPH_RECOVERY_UNVERIFIED",
                "review",
                "A small source glyph or body/visual symbol discrepancy requires confirmation",
            )
        )
    if any(warning.startswith("LATEX_NORMALIZATION_UNCERTAIN:") for warning in document.warnings):
        issues.append(_issue("LATEX_NORMALIZATION_UNCERTAIN", "review", "Unknown formula commands were rendered conservatively and require verification"))
    if any(
        warning.startswith("HYBRID_FORMULA_TEXT_SEGMENT_UNRESOLVED:")
        for warning in document.warnings
    ):
        issues.append(
            _issue(
                "HYBRID_FORMULA_TEXT_SEGMENT_UNRESOLVED",
                "reject",
                "Printed text beside a formula could not be recovered from source pixels",
            )
        )
    if any(
        warning.startswith("LOW_CONFIDENCE_LONG_TEXT_REGION:")
        for warning in document.warnings
    ):
        issues.append(
            _issue(
                "OCR_TEXT_CONFIDENCE_TOO_LOW",
                "reject",
                "A long printed-text region has insufficient OCR confidence for semantic acceptance",
            )
        )
    if any(warning.startswith("KATEX_RENDER_FALLBACK:") for warning in document.warnings):
        issues.append(_issue("KATEX_RENDER_FALLBACK", "review", "KaTeX rendering failed; a safe readable fallback was used"))
    if any(warning.startswith("ADJACENT_QUESTION_TRIMMED:") for warning in document.warnings):
        issues.append(
            _issue(
                "ADJACENT_QUESTION_BOUNDARY_UNVERIFIED",
                "review",
                "A following printed question was trimmed at an automatically detected boundary",
            )
        )
    if any(
        warning.startswith((
            "QUESTION_HEADER_CORRECTED_BY_SECONDARY_OCR:",
            "QUESTION_NUMBER_PUNCTUATION_RECOVERED:",
            "QUESTION_NUMBER_RECOVERED_FROM_VISUAL_CAPTION:",
        ))
        for warning in document.warnings
    ):
        issues.append(
            _issue(
                "QUESTION_HEADER_CORRECTED_UNVERIFIED",
                "review",
                "Automatic source evidence recovered or corrected the printed question header; confirm it against the source image",
            )
        )
    if any(
        warning.startswith("OPTION_FORMULA_SUFFIX_REMOVED_BY_SECONDARY_OCR:")
        for warning in document.warnings
    ):
        issues.append(
            _issue(
                "OPTION_FORMULA_CORRECTED_UNVERIFIED",
                "review",
                "A low-confidence option suffix was removed after independent OCR; confirm the printed option",
            )
        )
    if any(warning.startswith("STACKED_TEXT_ROWS_SPLIT:") for warning in document.warnings):
        issues.append(
            _issue(
                "STACKED_TEXT_ROWS_SPLIT_UNVERIFIED",
                "review",
                "A tall OCR box was split across adjacent rows; confirm the reconstructed reading order",
            )
        )
    if any(
        warning.startswith("STACKED_FRACTION_REFINED_FROM_SOURCE_CROP:")
        for warning in document.warnings
    ):
        issues.append(
            _issue(
                "STACKED_FRACTION_REFINED_UNVERIFIED",
                "review",
                "A split fraction was re-recognized from a tight source crop; confirm the complete formula",
            )
        )
    if any(
        warning == "STACKED_FRACTION_REFINEMENT_UNRESOLVED"
        or warning.startswith("STACKED_FRACTION_REFINEMENT_FAILED:")
        for warning in document.warnings
    ):
        issues.append(
            _issue(
                "STACKED_FRACTION_REFINEMENT_FAILED",
                "reject",
                "A geometrically split fraction could not be recovered as one complete formula",
            )
        )
    if any(
        warning.startswith((
            "MISSING_OPTION_A_RECOVERED_BY_SECONDARY_OCR:",
            "MISSING_TWO_COLUMN_OPTION_LABEL_RESTORED:",
            "OPTION_SEQUENCE_RECOVERED_FROM_COHERENT_LABELS:",
        ))
        for warning in document.warnings
    ):
        issues.append(
            _issue(
                "OPTION_LABEL_RECOVERED_UNVERIFIED",
                "review",
                "Missing option labels were recovered from source pixels or coherent source geometry; confirm them visually",
            )
        )
    if any(
        warning.startswith((
            "QUESTION_HEADER_OCR_DISAGREEMENT:",
            "QUESTION_HEADER_SECONDARY_OCR_UNRESOLVED",
            "QUESTION_HEADER_SECONDARY_OCR_FAILED:",
        ))
        for warning in document.warnings
    ):
        issues.append(
            _issue(
                "QUESTION_HEADER_UNVERIFIED",
                "review",
                "The printed question header could not be independently confirmed",
            )
        )
    warning_codes = set(document.warnings)
    if "RENDER_READBACK_EMPTY" in warning_codes:
        issues.append(_issue("RENDER_READBACK_EMPTY", "reject", "The final PNG could not be read back"))
    if "RENDER_READBACK_LOW_COVERAGE" in warning_codes:
        issues.append(_issue("RENDER_READBACK_LOW_COVERAGE", "review", "Final PNG readback coverage is below threshold"))
    if "RENDER_READBACK_OPTION_LOSS" in warning_codes:
        issues.append(_issue("RENDER_READBACK_OPTION_LOSS", "reject", "One or more option labels disappeared in the final PNG"))
    if "RENDER_READBACK_LATEX_TOKEN" in warning_codes:
        issues.append(_issue("RENDER_READBACK_LATEX_TOKEN", "reject", "Final PNG appears to expose a LaTeX command"))
    if any(warning.startswith("RENDER_READBACK_FAILED:") for warning in document.warnings):
        issues.append(_issue("RENDER_READBACK_FAILED", "review", "Final PNG readback validation failed to run"))
    if "RENDER_READBACK_PENDING" in warning_codes:
        issues.append(
            _issue(
                "RENDER_READBACK_PENDING",
                "review",
                "Final PNG is waiting for independent deferred readback",
            )
        )

    if document.display_text.strip() and not document.source_blocks:
        issues.append(
            _issue(
                "SOURCE_PROVENANCE_MISSING",
                "review",
                "Recognized text/formula blocks have no source-coordinate ledger",
            )
        )
    elif any(
        block.source_bbox is None
        or (block.disposition != "emitted" and not (block.reason or "").strip())
        for block in document.source_blocks
    ):
        issues.append(
            _issue(
                "SOURCE_PROVENANCE_INCOMPLETE",
                "review",
                "One or more recognized blocks lack a source coordinate or an exclusion reason",
            )
        )

    cues = config["visual_assets"].get("visual_cues", [])
    if (
        not formula_only
        and has_visual_cue(document.display_text, cues)
        and not document.visual_assets
    ):
        issues.append(_issue("VISUAL_ASSET_MISSING", "reject", "Text refers to a figure/table but no source crop is attached"))
    for asset in document.visual_assets:
        if asset.redraw_used or asset.extraction != "direct_source_crop":
            issues.append(_issue("VISUAL_REDRAW_FORBIDDEN", "reject", f"{asset.asset_id} is not a direct source crop"))
        if not asset.bbox_verified:
            issues.append(_issue("VISUAL_BBOX_UNVERIFIED", "review", f"{asset.asset_id} boundary requires human verification"))
        if not asset.edge_complete:
            issues.append(_issue("VISUAL_CROP_INCOMPLETE", "reject", f"{asset.asset_id} may be truncated"))
        if asset.fallback:
            issues.append(_issue("VISUAL_FULL_QUESTION_FALLBACK", "review", "Whole question crop used because a reliable figure boundary was unavailable"))

    labels = [option.get("label") for option in document.options]
    if not formula_only and (
        quality.get("unlabeled_formula_option_grid_is_reject", True)
        and _has_unlabeled_formula_option_grid(document)
    ):
        issues.append(
            _issue(
                "UNLABELED_FORMULA_OPTION_GRID",
                "reject",
                "A two-column formula option grid is visible in source geometry, "
                "but its A-D labels were not preserved in the reflowed result",
            )
        )
    if not formula_only and labels and labels != list("ABCD")[: len(labels)]:
        issues.append(_issue("OPTION_SEQUENCE", "reject", f"Unexpected option sequence: {labels}"))
    if (
        not formula_only
        and quality.get("partial_option_set_is_reject", True)
        and 0 < len(labels) < 4
    ):
        missing = [label for label in "ABCD" if label not in labels]
        issues.append(
            _issue(
                "PARTIAL_OPTION_SET",
                "reject",
                "A structured option set is incomplete; missing labels: "
                + ", ".join(missing or ["unknown"]),
            )
        )
    if (
        not formula_only
        and quality.get("incomplete_four_option_choice_is_reject", True)
        and _requires_complete_four_option_choice(document)
        and labels != list("ABCD")
    ):
        missing = [label for label in "ABCD" if label not in labels]
        issues.append(
            _issue(
                "CHOICE_OPTIONS_INCOMPLETE",
                "reject",
                "The prompt requires a complete A-D choice set; missing or unparsed labels: "
                + ", ".join(missing or ["unknown"]),
            )
        )
    empty_option_labels = [
        str(option.get("label") or "?")
        for option in document.options
        if not str(option.get("text") or "").strip()
    ]
    if empty_option_labels and not document.visual_assets and not formula_only:
        issues.append(
            _issue(
                "OPTION_CONTENT_MISSING",
                "reject",
                "Text-only options have empty content: " + ", ".join(empty_option_labels),
            )
        )
    if document.question_number and not formula_only:
        embedded_question = re.compile(
            rf"(?:\r?\n)\s*(?:第\s*)?{re.escape(str(document.question_number))}\s*[.．、]"
        )
        if any(embedded_question.search(str(option.get("text") or "")) for option in document.options):
            issues.append(
                _issue(
                    "ADJACENT_QUESTION_EMBEDDED_IN_OPTION",
                    "reject",
                    "The selected question header appears inside a preceding option block",
                )
            )

    minimum_bottom = int(quality["minimum_bottom_padding_px"])
    bottom_strip = output_image.crop((0, max(0, output_image.height - minimum_bottom), output_image.width, output_image.height)).convert("L")
    if ImageStat.Stat(bottom_strip).extrema[0][0] < 245:
        issues.append(_issue("BOTTOM_TRUNCATION_RISK", "reject", "Content reaches the required bottom safety margin"))

    if elapsed_seconds > float(quality["target_latency_seconds"]):
        issues.append(_issue("LATENCY_OVER_TARGET", "review", f"Latency {elapsed_seconds:.3f}s exceeds target"))
    return issues


def status_from_issues(issues: list[ValidationIssue]) -> Status:
    if any(issue.severity == "reject" for issue in issues):
        return "REJECT"
    if issues:
        return "REVIEW"
    return "PASS"
