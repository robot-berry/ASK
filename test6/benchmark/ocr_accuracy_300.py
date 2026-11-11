"""Run the production Test6 OCR pipeline over the frozen 300-image dataset.

The v1 dataset contains no human-verified question transcripts.  This runner
therefore records full-corpus operational/quality metrics and prepares a fixed,
stratified visual-audit sample.  It deliberately does not label OCR confidence
as accuracy; true accuracy is added only by the manual audit file.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import gc
import html
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path(r"D:\robot\Documents\ASK")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from presentation_guard import readable_ocr_text


DEFAULT_DATASET = WORKSPACE / "datasets" / "testzip_math_physics_300_v1"
DEFAULT_OUTPUT = WORKSPACE / "picture" / "test6" / "benchmark" / "ocr_accuracy_300_20260901"
TEST5_PATHS = {
    "mathematics": WORKSPACE / "picture" / "test6" / "math" / "source" / "test5_preprocess.py",
    # Production physics already imports the verified mathematics OCR module
    # through its adapter. The benchmark must exercise that same canonical
    # implementation instead of a stale vendored copy.
    "physics": WORKSPACE / "picture" / "test6" / "math" / "source" / "test5_preprocess.py",
}


def apply_production_cpu_affinity() -> str:
    """Match the local web workers' performance-core affinity on this PC."""

    if os.name != "nt" or (os.cpu_count() or 0) != 24:
        return "unchanged"
    mask = int(os.environ.get("TEST6_OCR_AFFINITY_MASK", "0xFFFF"), 0)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    if not ctypes.windll.kernel32.SetProcessAffinityMask(process, mask):
        # On hybrid Windows systems the direct call can be denied after Python
        # has initialized a processor group. psutil uses the supported process
        # API sequence and matches the web workers' 0-15 affinity.
        try:
            import psutil

            psutil.Process().cpu_affinity(
                [index for index in range(os.cpu_count() or 0) if mask & (1 << index)]
            )
        except Exception:
            return "failed"
    return f"0x{mask:X}"


def load_test5(subject: str) -> Any:
    path = TEST5_PATHS[subject]
    spec = importlib.util.spec_from_file_location(f"test6_ocr_benchmark_{subject}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Test5 module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return round(float(ordered[low]), 3)
    return round(float(ordered[low] * (high - position) + ordered[high] * (position - low)), 3)


def compact_text(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def process_image(
    test5: Any,
    engine: Any,
    image_path: Path,
    artifact_dir: Path | None = None,
    artifact_stem: str = "",
) -> dict[str, Any]:
    started = time.perf_counter()
    image = test5.cv_read(image_path)
    rotated, rotation = test5.auto_rotate_fast(image)
    quad = test5.find_document_quad(rotated, test5.Config())
    rectified, corrected = test5.perspective_warp(rotated, quad)
    rectified, angle = test5.deskew(rectified)
    enhanced = test5.normalize_illumination(rectified)
    enhanced, outer_cropped, outer_box = test5.crop_outer_whitespace_for_ocr(enhanced)
    outer_left, outer_top, outer_right, outer_bottom = outer_box
    visual_reference = rectified[
        outer_top:outer_bottom, outer_left:outer_right
    ].copy()

    # Match the website's balanced first-image profile. Two extra tiled OCR
    # passes are refinement work and must not delay the public OCR image.
    config = test5.Config(ocr_tiled_fallback_enabled=False)
    ocr_started = time.perf_counter()
    (
        ocr_boxes,
        _ocr_texts,
        _ocr_scores,
        ocr_rotation,
        all_items,
        audit,
    ) = test5.run_adaptive_ocr(enhanced, engine, config)
    if ocr_rotation == 180:
        old_h, old_w = enhanced.shape[:2]
        enhanced = cv2.rotate(enhanced, cv2.ROTATE_180)
        visual_reference = cv2.rotate(visual_reference, cv2.ROTATE_180)
        if len(ocr_boxes):
            mapped_boxes = ocr_boxes.copy()
            mapped_boxes[:, :, 0] = old_w - 1 - mapped_boxes[:, :, 0]
            mapped_boxes[:, :, 1] = old_h - 1 - mapped_boxes[:, :, 1]
            ocr_boxes = np.asarray([test5.order_quad(box) for box in mapped_boxes], np.float32)
        rotation = (rotation + 180) % 360

    question_box = test5.select_middle_question_box(
        enhanced.shape, all_items, config.select_middle_question
    )
    full_question_source = enhanced
    visual_question_reference, _ = test5.crop_question_workspace(
        visual_reference,
        all_items,
        question_box,
    )
    preliminary_items = test5.items_inside_question(all_items, question_box)
    preliminary_text = test5.build_ocr_transcript(preliminary_items)
    pre_crop_layout_audit: dict[str, Any] = {
        "requested": False,
        "elapsed_ms": 0.0,
        "layout_regions": [],
    }
    selected_layout_regions: list[dict[str, Any]] = []
    if hasattr(test5, "select_question_layout_regions"):
        pre_crop_layout_audit = test5.request_structured_layout(visual_reference)
        selected_layout_regions = test5.select_question_layout_regions(
            pre_crop_layout_audit,
            question_box,
            all_items,
            preliminary_text,
            full_question_source.shape,
        )
        pre_crop_layout_audit["requested"] = True
    enhanced, items = test5.crop_question_workspace(
        full_question_source, all_items, question_box
    )
    structured_ocr_source, structured_padding_top = (
        test5.prepare_structured_ocr_canvas(enhanced)
        if hasattr(test5, "prepare_structured_ocr_canvas")
        else (enhanced.copy(), 0)
    )
    pre_crop_figure_boxes: list[tuple[int, int, int, int]] = []
    if hasattr(test5, "attach_question_layout_regions"):
        enhanced, items, pre_crop_figure_boxes, attachment_audit = (
            test5.attach_question_layout_regions(
                enhanced,
                items,
                full_question_source,
                question_box,
                selected_layout_regions,
                all_items,
            )
        )
        pre_crop_layout_audit["attachment"] = attachment_audit
    formula_y_offset = int(
        (pre_crop_layout_audit.get("attachment") or {}).get("prepended_height", 0)
        if isinstance(pre_crop_layout_audit.get("attachment"), dict)
        else 0
    )
    structured_formula_y_offset = formula_y_offset - structured_padding_top
    audit["structured_ocr_padding_top"] = structured_padding_top
    formula_pixel_source = np.full_like(enhanced, 255)
    copy_height = min(
        visual_question_reference.shape[0],
        max(0, formula_pixel_source.shape[0] - formula_y_offset),
    )
    copy_width = min(
        visual_question_reference.shape[1],
        formula_pixel_source.shape[1],
    )
    if copy_height > 0 and copy_width > 0:
        formula_pixel_source[
            formula_y_offset:formula_y_offset + copy_height,
            :copy_width,
        ] = visual_question_reference[:copy_height, :copy_width]


    tiled_text = ""
    tiled_recovery_locally_resolved = False
    if audit.get("tiled_recovery_recommended") is True:
        local_probe, local_probe_actions = test5.apply_local_semantic_recovery(
            test5.build_ocr_transcript(items),
            ocr_audit=audit,
            fraction_tokens=[],
        )
        tiled_recovery_locally_resolved = bool(
            local_probe_actions
            and test5.transcript_semantic_completeness(local_probe).get(
                "passes_structure_gate"
            ) is True
        )
        if tiled_recovery_locally_resolved:
            audit["tiled_recovery_skipped"] = (
                "deterministic_local_recovery_complete"
            )
        elif not config.ocr_tiled_fallback_enabled:
            audit["tiled_recovery_skipped"] = "latency_balanced_profile"
    if (
        config.ocr_tiled_fallback_enabled
        and audit.get("tiled_recovery_recommended") is True
        and not tiled_recovery_locally_resolved
    ):
        test5.configure_ocr_detection(
            engine,
            box_thresh=0.25,
            unclip_ratio=1.50,
            text_score=config.ocr_tiled_min_text_score,
        )
        try:
            tiled_text, tiled_ms, tiled_passes = test5.run_tiled_ocr_transcript(
                enhanced, engine, config, items, force=True
            )
            audit.update(
                {
                    "pass_count": int(audit.get("pass_count") or 1) + tiled_passes,
                    "tiled_elapsed_ms": tiled_ms,
                    "tiled_passes": tiled_passes,
                }
            )
        finally:
            test5.configure_ocr_detection(
                engine, box_thresh=0.50, unclip_ratio=1.50, text_score=0.50
            )

    text_mask = test5.create_text_mask(enhanced.shape, items)
    detection_copy, _removed = test5.make_detection_copy(enhanced, text_mask)
    figures, _debug = test5.detect_illustration_boxes(
        detection_copy, enhanced, items, config
    )
    table_only_layout = bool(
        pre_crop_figure_boxes
        and selected_layout_regions
        and all(str(region.get("label") or "").lower() == "table" for region in selected_layout_regions)
        and not re.search(r"如图|图[①②③④⑤⑥1-9]", preliminary_text)
    )
    if table_only_layout:
        figures = list(pre_crop_figure_boxes)
        audit["table_only_layout_preferred"] = True
    elif (
        len(pre_crop_figure_boxes) >= 2
        and selected_layout_regions
        and all(
            str(region.get("label") or "").lower() in {"image", "chart"}
            for region in selected_layout_regions
        )
    ):
        figures = test5.expand_multi_layout_gallery_boxes(
            pre_crop_figure_boxes,
            enhanced.shape,
            items,
        )[: config.illustration_max_count]
        audit["multi_layout_gallery_preferred"] = True
    elif pre_crop_figure_boxes and hasattr(test5, "merge_illustration_evidence"):
        figures = test5.merge_illustration_evidence(
            figures,
            pre_crop_figure_boxes,
            enhanced.shape,
            items,
            config.illustration_max_count,
        )
    if hasattr(test5, "recover_visual_choice_strip"):
        visual_choice_strip = test5.recover_visual_choice_strip(
            enhanced,
            items,
            preliminary_text,
            figures,
        )
        if visual_choice_strip is not None:
            figures = [visual_choice_strip]
            audit["visual_choice_strip_recovered"] = True
    if not figures and hasattr(test5, "recover_referenced_visual_sidecar"):
        sidecar_boxes = test5.recover_referenced_visual_sidecar(
            detection_copy,
            enhanced,
            items,
            preliminary_text,
        )
        if sidecar_boxes:
            figures = sidecar_boxes[: config.illustration_max_count]
            audit["referenced_visual_sidecar_recovered"] = len(figures)
    fraction_audit: dict[str, Any] = {"tokens": []}
    if hasattr(test5, "recognize_local_numeric_fractions"):
        recovered_fractions, fraction_audit = test5.recognize_local_numeric_fractions(
            enhanced, items, engine
        )
        if recovered_fractions:
            items = test5.merge_recovered_numeric_fractions(items, recovered_fractions)
    base_text = test5.build_ocr_transcript(items, figures)
    complete_text = test5.build_ocr_transcript(items)
    table_cue = bool(re.search(
        r"下表(?!面)|如下表|表中|表格|统计表|成绩表",
        complete_text,
    ))
    base_numeric_tokens = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", base_text)
    complete_numeric_tokens = re.findall(
        r"(?<![A-Za-z])\d+(?:\.\d+)?", complete_text
    )
    reliable_table_cells_hidden_by_crop = bool(
        table_cue
        and len(compact_text(complete_text)) >= len(compact_text(base_text)) + 4
        and len(complete_numeric_tokens) >= len(base_numeric_tokens) + 2
        and re.search(
            r"(?:次数|频数|人数|个数|数量|组数|环数|分数)",
            complete_text,
        )
    )
    if (
        (
            len(compact_text(complete_text)) >= len(compact_text(base_text)) + 20
            and sum("\u4e00" <= char <= "\u9fff" for char in complete_text)
            >= sum("\u4e00" <= char <= "\u9fff" for char in base_text) + 8
        )
        or reliable_table_cells_hidden_by_crop
    ):
        base_text = complete_text
        audit["text_inside_figure_preserved"] = True
        if reliable_table_cells_hidden_by_crop:
            audit["table_cell_text_preserved"] = True
    base_cjk = sum("\u4e00" <= char <= "\u9fff" for char in base_text)
    tiled_cjk = sum("\u4e00" <= char <= "\u9fff" for char in tiled_text)
    semantic_candidate = tiled_text
    base_semantic = test5.transcript_semantic_completeness(base_text)
    if tiled_text and base_semantic.get("stem_missing") is True:
        semantic_candidate = test5.repair_ocr_transcript(f"{tiled_text}\n{base_text}")
    if (
        (
            tiled_cjk >= base_cjk + 12
            and len(compact_text(tiled_text)) >= len(compact_text(base_text)) + 24
        )
        or test5.semantic_transcript_is_better(base_text, semantic_candidate)
    ):
        transcript = semantic_candidate
        selected_pass = f"{audit.get('selected_pass', 'primary')}+tiled-text"
    else:
        transcript = base_text
        selected_pass = str(audit.get("selected_pass") or "primary")
    fraction_tokens = [
        str(token)
        for token in (fraction_audit.get("tokens") or [])
        if re.fullmatch(r"\d{1,3}/\d{1,3}", str(token))
    ]
    pre_structured_actions: list[str] = []
    if hasattr(test5, "apply_local_semantic_recovery"):
        transcript, pre_structured_actions = test5.apply_local_semantic_recovery(
            transcript,
            ocr_audit=audit,
            fraction_tokens=fraction_tokens,
        )
    if hasattr(test5, "preserve_primary_text_table"):
        transcript, preliminary_table_preserved = test5.preserve_primary_text_table(
            preliminary_text,
            transcript,
        )
        if preliminary_table_preserved:
            pre_structured_actions.append("preserve_preliminary_text_table")
    if hasattr(test5, "strip_captioned_numeric_figure_artifacts"):
        cleaned_transcript = test5.strip_captioned_numeric_figure_artifacts(transcript)
        if cleaned_transcript != transcript:
            transcript = cleaned_transcript
            pre_structured_actions.append(
                "separate_captioned_numeric_figure_from_prose"
            )
    readable_ocr_ready_ms = (time.perf_counter() - started) * 1000.0
    readable_semantic = test5.transcript_semantic_completeness(transcript)
    readable_requires_visual = bool(
        hasattr(test5, "transcript_requires_illustration")
        and test5.transcript_requires_illustration(transcript)
    )
    readable_package_ready = bool(
        readable_semantic.get("passes_structure_gate") is True
        and (
            not readable_requires_visual
            or bool(figures)
        )
    ) or bool(
        readable_semantic.get("table_row_incomplete") is True
        and bool(figures)
    )
    structured_reasons = (
        test5.structured_ocr_recovery_reasons(transcript, items)
        if hasattr(test5, "structured_ocr_recovery_reasons")
        else ["legacy_always_request"]
    )
    skip_structured_refinement = bool(
        hasattr(test5, "can_skip_structured_ocr_for_preserved_visual")
        and test5.can_skip_structured_ocr_for_preserved_visual(
            transcript,
            structured_reasons,
            pre_crop_figure_boxes,
        )
    )
    formula_only_refinement = bool(
        not skip_structured_refinement
        and hasattr(test5, "formula_only_recovery_is_sufficient")
        and test5.formula_only_recovery_is_sufficient(structured_reasons)
        and hasattr(test5, "fuse_formula_only_ocr_items")
    )
    pre_structured_metrics = test5.transcript_semantic_completeness(transcript)
    use_full_page_for_stem_recovery = bool(
        pre_structured_metrics.get("stem_missing") is True
        and set(pre_structured_metrics.get("option_labels") or []) == set("ABCD")
        and test5.first_question_number(transcript) is None
        and question_box != (0, 0, full_question_source.shape[1], full_question_source.shape[0])
    )
    if skip_structured_refinement:
        structured_action = ""
        structured_audit = {
            "selected": False,
            "elapsed_ms": 0.0,
            "selection_reason": "complete_text_with_preserved_visual_skipped",
            "visual_region_count": len(pre_crop_figure_boxes),
        }
    elif formula_only_refinement:
        structured_audit = test5.request_formula_ocr(structured_ocr_source)
        structured_audit["request_mode"] = "formula_only"
        refined_items, formula_replacement_count = (
            test5.fuse_formula_only_ocr_items(
                items,
                structured_audit,
                y_offset=structured_formula_y_offset,
            )
        )
        structured_audit["formula_replacement_count"] = formula_replacement_count
        if formula_replacement_count:
            formula_candidate = test5.build_ocr_transcript(refined_items, figures)
            formula_overlap = test5.formula_candidate_needs_full_structured_retry(
                transcript,
                formula_candidate,
                formula_count=int(structured_audit.get("formula_count") or 0),
                long_formula=any(
                    len(str(formula)) >= 64
                    for formula in (structured_audit.get("formulas") or [])
                ),
            )
            if formula_overlap:
                structured_action = ""
                structured_audit["selection_reason"] = (
                    "formula_text_overlap_requires_full_structured"
                )
            else:
                transcript, structured_action = test5.fuse_structured_ocr_transcript(
                    transcript,
                    formula_candidate,
                    structured_audit,
                )
            items = refined_items
            structured_audit["formula_visual_boxes_retained"] = True
            if (
                not structured_action
                and (
                    int(structured_audit.get("formula_count") or 0) >= 2
                    or any(
                        len(str(formula)) >= 64
                        for formula in (structured_audit.get("formulas") or [])
                    )
                )
                and str(structured_audit.get("selection_reason") or "") in {
                    "formula_text_overlap_requires_full_structured",
                    "subquestion_structure_regression_rejected",
                    "primary_transcript_retained",
                }
            ):
                transcript, structured_audit, structured_action = (
                    test5.retry_full_structured_after_formula_only(
                        transcript,
                        structured_ocr_source,
                        structured_audit,
                    )
                )
        else:
            structured_action = ""
            structured_audit["selection_reason"] = (
                "formula_only_no_anchored_replacement"
            )
    elif structured_reasons:
        structured_text, structured_audit = test5.request_structured_ocr(
            rotated if use_full_page_for_stem_recovery else structured_ocr_source
        )
        structured_audit["request_source"] = (
            "full_page_stem_recovery"
            if use_full_page_for_stem_recovery
            else "question_workspace"
        )
        formula_replacement_count = 0
        if not use_full_page_for_stem_recovery:
            refined_items, formula_replacement_count = (
                test5.fuse_formula_only_ocr_items(
                    items,
                    structured_audit,
                    y_offset=structured_formula_y_offset,
                    allow_unanchored=True,
                )
            )
            structured_audit["formula_replacement_count"] = (
                formula_replacement_count
            )
            if formula_replacement_count:
                refined_items, structured_text_item_count = (
                    test5.augment_full_structured_text_items(
                        refined_items,
                        structured_audit,
                        y_offset=structured_formula_y_offset,
                    )
                )
                structured_audit["structured_text_item_count"] = (
                    structured_text_item_count
                )
                items = refined_items
                structured_audit["formula_visual_boxes_retained"] = True
        transcript, structured_action = test5.fuse_structured_ocr_transcript(
            transcript, structured_text, structured_audit
        )
        if not structured_action and formula_replacement_count:
            spatial_formula_candidate = test5.build_ocr_transcript(
                items,
                figures,
            )
            spatial_overlap = test5.formula_candidate_needs_full_structured_retry(
                transcript,
                spatial_formula_candidate,
                formula_count=int(structured_audit.get("formula_count") or 0),
                long_formula=any(
                    len(str(formula)) >= 64
                    for formula in (structured_audit.get("formulas") or [])
                ),
            )
            if not spatial_overlap:
                transcript, spatial_action = test5.fuse_structured_ocr_transcript(
                    transcript,
                    spatial_formula_candidate,
                    structured_audit,
                )
                if spatial_action:
                    structured_action = f"spatial_{spatial_action}"
    else:
        structured_action = ""
        structured_audit = {
            "selected": False,
            "elapsed_ms": 0.0,
            "selection_reason": "balanced_gate_skipped",
            "gate_reasons": [],
        }
    structured_audit["requested"] = bool(
        structured_reasons and not skip_structured_refinement
    )
    structured_audit["gate_reasons"] = structured_reasons
    if not str(structured_audit.get("request_mode") or "").startswith(
        "formula_then_full"
    ):
        structured_audit["request_mode"] = (
            "formula_only" if formula_only_refinement else "full_structured"
            if structured_reasons and not skip_structured_refinement
            else "skipped"
        )
    if structured_action:
        selected_pass = f"{selected_pass}+structured-local"
    if use_full_page_for_stem_recovery and not structured_action:
        full_page_structured_audit = {
            "selection_reason": "full_page_already_attempted",
            "elapsed_ms": 0.0,
        }
        full_page_structured_action = ""
    else:
        transcript, full_page_structured_audit, full_page_structured_action = (
            test5.recover_structured_ocr_from_full_page(transcript, rotated)
        )
    if full_page_structured_action:
        selected_pass = f"{selected_pass}+structured-full-page"
    semantic_actions: list[str] = list(pre_structured_actions)
    if hasattr(test5, "apply_local_semantic_recovery"):
        transcript, post_structured_actions = test5.apply_local_semantic_recovery(
            transcript,
            ocr_audit=audit,
            fraction_tokens=fraction_tokens,
        )
        if hasattr(test5, "preserve_primary_text_table"):
            transcript, preliminary_table_preserved = (
                test5.preserve_primary_text_table(
                    preliminary_text,
                    transcript,
                )
            )
            if preliminary_table_preserved:
                post_structured_actions.append(
                    "preserve_preliminary_text_table"
                )
        if hasattr(test5, "strip_captioned_numeric_figure_artifacts"):
            cleaned_transcript = test5.strip_captioned_numeric_figure_artifacts(transcript)
            if cleaned_transcript != transcript:
                transcript = cleaned_transcript
                post_structured_actions.append(
                    "separate_captioned_numeric_figure_from_prose"
                )
        semantic_actions = list(dict.fromkeys(
            [*semantic_actions, *post_structured_actions]
        ))
    layout_audit: dict[str, Any] = {
        "requested": False,
        "elapsed_ms": 0.0,
        "accepted_region_count": 0,
    }
    if (
        hasattr(test5, "transcript_requires_illustration")
        and (
            test5.transcript_requires_illustration(transcript)
            or bool(pre_crop_figure_boxes)
        )
        and hasattr(test5, "structured_layout_illustration_boxes")
        and hasattr(test5, "merge_illustration_evidence")
    ):
        layout_evidence = dict(structured_audit)
        reused_pre_crop = bool(pre_crop_layout_audit.get("requested"))
        reused = bool(not reused_pre_crop and layout_evidence.get("layout_regions"))
        if reused_pre_crop:
            layout_evidence = {
                "available": bool(pre_crop_figure_boxes),
                "engine": "local-pp-doclayoutv2-pre-crop",
                "layout_regions": [],
                "elapsed_ms": 0.0,
                "error": pre_crop_layout_audit.get("error"),
            }
        elif not reused:
            layout_evidence = test5.request_structured_layout(enhanced)
        structured_boxes, layout_audit = test5.structured_layout_illustration_boxes(
            layout_evidence,
            enhanced.shape,
            items,
            transcript,
            config.illustration_max_count,
        )
        layout_audit["reused_from_structured_ocr"] = reused
        layout_audit["reused_from_pre_crop"] = reused_pre_crop
        if not (reused_pre_crop and figures):
            # ``figures`` already contains the one-box-per-panel regions that
            # were selected from the original full page.  Running them back
            # through residual-contour sanitisation can discard lower panels
            # merely because their labels overlap OCR text (e.g. 图3..图5).
            figures = test5.merge_illustration_evidence(
                figures,
                structured_boxes,
                enhanced.shape,
                items,
                config.illustration_max_count,
            )
        layout_audit["fused_figure_count"] = len(figures)
    prepended_height = int(
        (pre_crop_layout_audit.get("attachment") or {}).get("prepended_height", 0)
        if isinstance(pre_crop_layout_audit.get("attachment"), dict)
        else 0
    )
    if prepended_height > 0:
        clamp_margin = max(12, round(prepended_height * 0.08))
        figures = [
            (
                left,
                top,
                min(right, enhanced.shape[1]),
                min(bottom, prepended_height)
                if top < prepended_height and bottom <= prepended_height + clamp_margin
                else bottom,
            )
            for left, top, right, bottom in figures
            if right > left and bottom > top
        ]
    final_semantic = test5.transcript_semantic_completeness(transcript)
    table_rendered_as_text = bool(
        re.search(
            r"下表(?!面)|如下表|表中|表格|统计表|成绩表|(?:对应)?数据如下|(?:成绩|统计|结果)如下",
            transcript,
        )
        and not re.search(r"如图|图[①②③④⑤⑥1-9甲乙丙丁戊己]", transcript)
        and not any(
            str(region.get("label") or "").lower() in {"image", "chart"}
            for region in selected_layout_regions
        )
        and final_semantic.get("table_row_incomplete") is not True
    )
    if table_rendered_as_text:
        if figures:
            audit["discarded_rendered_table_crops"] = len(figures)
        figures = []
        audit["table_rendered_as_text"] = True
    elif (
        figures
        and hasattr(test5, "transcript_requires_illustration")
        and not test5.transcript_requires_illustration(transcript)
    ):
        if pre_crop_figure_boxes:
            audit["preserved_layout_visual_without_text_cue"] = len(figures)
        else:
            figures = []
    elif (
        len(figures) > 1
        and re.search(r"图[①1]", transcript)
        and re.search(r"图[②2]", transcript)
    ):
        figures = [(
            min(box[0] for box in figures),
            min(box[1] for box in figures),
            max(box[2] for box in figures),
            max(box[3] for box in figures),
        )]
    if figures and hasattr(test5, "captioned_figure_payloads"):
        raw_payloads = test5.captioned_figure_payloads(figures, items)
        filtered_payloads = raw_payloads
        if hasattr(test5, "filter_figure_payloads_for_question"):
            filtered_payloads = test5.filter_figure_payloads_for_question(
                transcript,
                raw_payloads,
            )
            if len(filtered_payloads) < len(raw_payloads):
                audit["discarded_neighbour_question_figures"] = (
                    len(raw_payloads) - len(filtered_payloads)
                )
        if (
            hasattr(test5, "suppress_handwritten_only_figures")
            and test5.suppress_handwritten_only_figures(
                transcript,
                filtered_payloads,
            )
        ):
            audit["discarded_handwritten_only_figures"] = len(filtered_payloads)
            filtered_payloads = []
        figures = [box for box, _caption in filtered_payloads]
    formula_visual_required = False
    requires_visual = bool(
        hasattr(test5, "transcript_requires_illustration")
        and test5.transcript_requires_illustration(transcript)
        and not table_rendered_as_text
        and not bool(audit.get("discarded_handwritten_only_figures"))
    )
    full_question_visual_fallback = False
    if requires_visual and not figures:
        audit["illustration_missing_after_detection"] = True
    semantic_issue_keys = (
        "stem_missing",
        "option_sequence_incomplete",
        "choice_options_missing",
        "empty_option_content",
        "subquestion_gap",
        "nested_subquestion_incomplete",
        "table_row_incomplete",
        "formula_gap",
        "symbol_damage",
        "too_short",
        "multiple_question_blocks",
        "repeated_question_number",
    )
    semantic_issue_types = [
        key for key in semantic_issue_keys if final_semantic.get(key) is True
    ]
    semantic_package_pass = bool(
        (
            final_semantic.get("passes_structure_gate")
            and (not requires_visual or bool(figures))
        )
        or (
            set(semantic_issue_types) == {"table_row_incomplete"}
            and bool(figures)
        )
    )
    ocr_ms = (time.perf_counter() - ocr_started) * 1000.0

    trusted = [item for item in items if item.printed or item.formula_like]
    scores = [float(item.score) for item in trusted]
    weighted_denominator = sum(max(1, len(compact_text(item.text))) for item in trusted)
    weighted_confidence = (
        sum(float(item.score) * max(1, len(compact_text(item.text))) for item in trusted)
        / weighted_denominator
        if weighted_denominator
        else 0.0
    )
    compact = compact_text(transcript)
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in compact)
    alnum_count = sum(char.isalnum() for char in compact)
    question_marker = bool(re.search(r"(?:^|\n)\s*[1-9]\d{0,2}\s*(?:[.．、](?!\d)|[（(])", transcript))
    part_markers = len(re.findall(r"[（(]\s*\d{1,2}\s*[）)]", transcript))
    flags: list[str] = []
    if not compact:
        flags.append("empty_transcript")
    elif len(compact) < 12:
        flags.append("very_short_transcript")
    if weighted_confidence < 0.80:
        flags.append("low_weighted_confidence")
    if scores and quantile(scores, 0.10) is not None and float(quantile(scores, 0.10)) < 0.62:
        flags.append("low_tail_confidence")
    if cjk_count == 0 and alnum_count < 8:
        flags.append("little_readable_content")
    if len(compact) >= 30 and not question_marker:
        flags.append("missing_question_marker")
    if int(audit.get("pass_count") or 1) > 1:
        flags.append("fallback_used")
    if audit.get("missing_preamble") is True:
        flags.append("quality_audit_missing_preamble")
    if not bool(final_semantic.get("passes_structure_gate")):
        flags.append("semantic_structure_gate_failed")
    if structured_audit.get("error"):
        flags.append(f"structured_ocr_{structured_audit['error']}")

    core_ms = (time.perf_counter() - started) * 1000.0
    rendered_path = ""
    selected_source_path = ""
    illustration_paths: list[str] = []
    figure_captions = [
        caption
        for _box, caption in test5.captioned_figure_payloads(figures, items)
    ]
    first_image_ms: float | None = None
    render_and_write_ms = 0.0
    if artifact_dir is not None:
        safe_stem = artifact_stem or image_path.stem
        final_dir = artifact_dir / "ocr_rendered"
        selected_dir = artifact_dir / "selected_question_evidence"
        illustration_dir = artifact_dir / "illustrations"
        for directory in (final_dir, selected_dir, illustration_dir):
            directory.mkdir(parents=True, exist_ok=True)
        reflow_source = enhanced.copy()
        broad_ink = test5.broad_colored_ink_mask(enhanced)
        reflow_source[broad_ink > 0] = (255, 255, 255)
        for left, top, right, bottom in figures:
            reflow_source[top:bottom, left:right] = enhanced[
                top:bottom, left:right
            ]
        preserve_table_crop_pixels = bool(
            table_only_layout and final_semantic.get("table_row_incomplete") is True
        )
        preserved_table_images: list[np.ndarray] = []
        if preserve_table_crop_pixels:
            for region in selected_layout_regions:
                if str(region.get("label") or "").lower() != "table":
                    continue
                table_box = tuple(int(value) for value in region["bbox"])
                table_image = test5.crop_with_padding(
                    visual_reference, table_box, 0.015
                )
                if table_image.size:
                    preserved_table_images.append(table_image)
        rendered_figures = [] if preserved_table_images else figures
        rendered, _formula_count = test5.build_reflow_document(
            reflow_source,
            items,
            rendered_figures,
            config,
            transcript=transcript,
            preserve_figure_text=preserve_table_crop_pixels,
            preserved_figure_images=preserved_table_images,
            formula_pixel_source=formula_pixel_source,
            text_exclusion_boxes=(pre_crop_figure_boxes if preserved_table_images else []),
        )
        rendered_file = final_dir / f"{safe_stem}.png"
        test5.cv_write_png(rendered_file, rendered, config.output_quality)
        first_image_ms = (time.perf_counter() - started) * 1000.0
        selected_file = selected_dir / f"{safe_stem}.png"
        test5.cv_write_png(selected_file, enhanced, config.output_quality)
        rendered_path = str(rendered_file)
        selected_source_path = str(selected_file)
        if preserved_table_images:
            illustration_sources = list(preserved_table_images)
        else:
            illustration_sources = [
                (
                    test5.crop_with_padding(enhanced, figure_box, 0.0)
                    if preserve_table_crop_pixels
                    else test5.crop_illustration_with_clean_margins(
                        enhanced, figure_box, items
                    )
                )
                for figure_box in figures
            ]
        for figure_index, illustration_crop in enumerate(illustration_sources, 1):
            figure_file = illustration_dir / f"{safe_stem}_figure_{figure_index:02d}.png"
            test5.cv_write_png(
                figure_file,
                illustration_crop,
                config.output_quality,
            )
            illustration_paths.append(str(figure_file))
        render_and_write_ms = first_image_ms - core_ms

    return {
        "status": "success",
        "transcript": transcript,
        "compact_character_count": len(compact),
        "cjk_count": cjk_count,
        "alphanumeric_count": alnum_count,
        "line_count": len([line for line in transcript.splitlines() if line.strip()]),
        "part_marker_count": part_markers,
        "question_marker_detected": question_marker,
        "trusted_item_count": len(trusted),
        "all_item_count": len(items),
        "mean_item_confidence": round(statistics.fmean(scores), 6) if scores else 0.0,
        "weighted_character_confidence": round(weighted_confidence, 6),
        "p10_item_confidence": quantile(scores, 0.10),
        "minimum_item_confidence": round(min(scores), 6) if scores else None,
        "ocr_pass_count": int(audit.get("pass_count") or 1),
        "ocr_selected_pass": selected_pass,
        "structured_ocr_selected": bool(structured_audit.get("selected")),
        "structured_ocr_requested": bool(structured_audit.get("requested")),
        "structured_ocr_request_mode": str(
            structured_audit.get("request_mode") or ""
        ),
        "structured_ocr_gate_reasons": list(structured_audit.get("gate_reasons") or []),
        "structured_formula_count": int(structured_audit.get("formula_count") or 0),
        "structured_formula_replacement_count": int(
            structured_audit.get("formula_replacement_count") or 0
        ),
        "structured_text_item_count": int(
            structured_audit.get("structured_text_item_count") or 0
        ),
        "structured_ocr_elapsed_ms": round(
            float(structured_audit.get("elapsed_ms") or 0.0)
            + float(full_page_structured_audit.get("elapsed_ms") or 0.0),
            3,
        ),
        "structured_ocr_reason": str(structured_audit.get("selection_reason") or ""),
        "open_source_layout_requested": bool(layout_audit.get("requested")),
        "open_source_layout_reused": bool(
            layout_audit.get("reused_from_structured_ocr")
            or layout_audit.get("reused_from_pre_crop")
        ),
        "pre_crop_layout_requested": bool(pre_crop_layout_audit.get("requested")),
        "pre_crop_layout_elapsed_ms": round(
            float(pre_crop_layout_audit.get("elapsed_ms") or 0.0), 3
        ),
        "pre_crop_layout_selected_region_count": len(selected_layout_regions),
        "pre_crop_layout_external_region_count": int(
            (pre_crop_layout_audit.get("attachment") or {}).get(
                "external_region_count", 0
            )
            if isinstance(pre_crop_layout_audit.get("attachment"), dict)
            else 0
        ),
        "open_source_layout_region_count": int(
            layout_audit.get("accepted_region_count") or 0
        ),
        "open_source_layout_elapsed_ms": round(
            float(layout_audit.get("elapsed_ms") or 0.0), 3
        ),
        "open_source_layout_error": str(layout_audit.get("error") or ""),
        "pre_crop_layout_selected_region_count": len(selected_layout_regions),
        "pre_crop_layout_selected_regions": selected_layout_regions,
        "pre_crop_layout_attachment": pre_crop_layout_audit.get("attachment") or {},
        "pre_crop_question_box": list(question_box),
        "pre_crop_preliminary_text": preliminary_text,
        "pre_crop_caption_items": [
            {
                "text": str(item.text or ""),
                "center": [round(float(item.center_x), 1), round(float(item.center_y), 1)],
            }
            for item in all_items
            if re.search(r"第\s*\d{1,3}|题图", str(item.text or ""))
        ],
        "pre_crop_layout_source_regions": list(
            pre_crop_layout_audit.get("layout_regions") or []
        ),
        "full_page_structured_ocr_selected": bool(full_page_structured_action),
        "full_page_structured_ocr_elapsed_ms": round(
            float(full_page_structured_audit.get("elapsed_ms") or 0.0), 3
        ),
        "full_page_structured_ocr_reason": str(
            full_page_structured_audit.get("selection_reason") or ""
        ),
        "full_page_structured_formula_count": int(
            full_page_structured_audit.get("full_page_formula_count") or 0
        ),
        "semantic_structure_pass": bool(final_semantic.get("passes_structure_gate")),
        "semantic_issue_count": int(final_semantic.get("issue_count") or 0),
        "semantic_issue_types": semantic_issue_types,
        "semantic_package_pass": semantic_package_pass,
        "semantic_package_evidence": (
            "text_plus_preserved_table_figure"
            if semantic_package_pass and semantic_issue_types
            else "text_structure_gate"
        ),
        "semantic_page_noise_detected": bool(final_semantic.get("page_noise")),
        "semantic_option_labels": list(final_semantic.get("option_labels") or []),
        "semantic_choice_count_ambiguous": bool(
            final_semantic.get("choice_count_ambiguous")
        ),
        "semantic_subquestion_numbers": list(
            final_semantic.get("subquestion_numbers") or []
        ),
        "semantic_recovery_actions": semantic_actions,
        "semantic_requires_visual": requires_visual,
        "formula_visual_evidence_required": formula_visual_required,
        "full_question_visual_fallback": full_question_visual_fallback,
        "table_rendered_as_text": table_rendered_as_text,
        "visual_choice_strip_recovered": bool(
            audit.get("visual_choice_strip_recovered")
        ),
        "discarded_neighbour_question_figures": int(
            audit.get("discarded_neighbour_question_figures") or 0
        ),
        "discarded_handwritten_only_figures": int(
            audit.get("discarded_handwritten_only_figures") or 0
        ),
        "illustration_count": len(figures),
        "rotation": rotation,
        "perspective_corrected": bool(corrected),
        "deskew_angle": round(float(angle), 3),
        "outer_whitespace_cropped": bool(outer_cropped),
        "outer_crop_box": list(outer_box),
        "quality_flags": flags,
        "ocr_and_layout_ms": round(ocr_ms, 3),
        "total_ms": round(core_ms, 3),
        "first_image_ms": round(first_image_ms, 3) if first_image_ms is not None else None,
        "readable_ocr_ready_ms": round(readable_ocr_ready_ms, 3),
        "first_readable_image_ms": round(
            readable_ocr_ready_ms + render_and_write_ms,
            3,
        ),
        # Semantic validation is non-blocking for the public OCR image. Keep
        # the refined/background value separately for latency diagnosis.
        "public_first_image_ms": round(
            readable_ocr_ready_ms + render_and_write_ms,
            3,
        ),
        "background_refined_image_ms": round(
            first_image_ms or core_ms,
            3,
        ),
        "readable_package_ready_before_structured": readable_package_ready,
        "render_and_write_ms": round(render_and_write_ms, 3),
        "rendered_ocr_path": rendered_path,
        "selected_source_path": selected_source_path,
        "illustration_paths": illustration_paths,
        "figure_captions": figure_captions,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "success"]
    timings = [float(row["total_ms"]) for row in successful]
    first_image_timings = [
        float(row["first_image_ms"])
        for row in successful
        if row.get("first_image_ms") is not None
    ]
    first_readable_image_timings = [
        float(row["first_readable_image_ms"])
        for row in successful
        if row.get("first_readable_image_ms") is not None
    ]
    public_first_image_timings = [
        float(row["public_first_image_ms"])
        for row in successful
        if row.get("public_first_image_ms") is not None
    ]
    confidences = [float(row["weighted_character_confidence"]) for row in successful]
    flag_counts = Counter(flag for row in successful for flag in row.get("quality_flags", []))
    semantic_issue_counts = Counter(
        issue for row in successful for issue in row.get("semantic_issue_types", [])
    )
    text_pass_count = sum(bool(row.get("semantic_structure_pass")) for row in successful)
    package_pass_count = sum(bool(row.get("semantic_package_pass")) for row in successful)
    return {
        "count": len(rows),
        "success_count": len(successful),
        "success_rate": round(len(successful) / len(rows), 6) if rows else 0.0,
        "nonempty_count": sum(bool(compact_text(str(row.get("transcript") or ""))) for row in successful),
        "nonempty_rate": round(
            sum(bool(compact_text(str(row.get("transcript") or ""))) for row in successful)
            / len(rows), 6
        ) if rows else 0.0,
        "mean_weighted_character_confidence": round(statistics.fmean(confidences), 6) if confidences else 0.0,
        "median_weighted_character_confidence": quantile(confidences, 0.50),
        "fallback_count": sum(int(row.get("ocr_pass_count") or 1) > 1 for row in successful),
        "mean_ocr_pass_count": round(
            statistics.fmean(float(row.get("ocr_pass_count") or 1) for row in successful),
            3,
        ) if successful else 0.0,
        "structured_ocr_requested_count": sum(
            bool(row.get("structured_ocr_requested")) for row in successful
        ),
        "structured_ocr_selected_count": sum(
            bool(row.get("structured_ocr_selected")) for row in successful
        ),
        "open_source_layout_requested_count": sum(
            bool(row.get("open_source_layout_requested")) for row in successful
        ),
        "open_source_layout_reused_count": sum(
            bool(row.get("open_source_layout_reused")) for row in successful
        ),
        "open_source_layout_added_region_count": sum(
            int(row.get("open_source_layout_region_count") or 0)
            for row in successful
        ),
        "full_page_structured_ocr_selected_count": sum(
            bool(row.get("full_page_structured_ocr_selected")) for row in successful
        ),
        "question_marker_detected_count": sum(bool(row.get("question_marker_detected")) for row in successful),
        "mean_compact_character_count": round(
            statistics.fmean(float(row.get("compact_character_count") or 0) for row in successful), 3
        ) if successful else 0.0,
        "semantic_text_pass_count": text_pass_count,
        "semantic_text_pass_rate": round(text_pass_count / len(rows), 6) if rows else 0.0,
        "semantic_package_pass_count": package_pass_count,
        "semantic_package_pass_rate": round(package_pass_count / len(rows), 6) if rows else 0.0,
        "semantic_issue_type_counts": dict(sorted(semantic_issue_counts.items())),
        "total_ms_p50": quantile(timings, 0.50),
        "total_ms_p90": quantile(timings, 0.90),
        "total_ms_p95": quantile(timings, 0.95),
        "total_ms_mean": round(statistics.fmean(timings), 3) if timings else None,
        "first_image_ms_p50": quantile(first_image_timings, 0.50),
        "first_image_ms_p95": quantile(first_image_timings, 0.95),
        "first_image_ms_mean": (
            round(statistics.fmean(first_image_timings), 3)
            if first_image_timings else None
        ),
        "first_readable_image_ms_p50": quantile(
            first_readable_image_timings, 0.50
        ),
        "first_readable_image_ms_p95": quantile(
            first_readable_image_timings, 0.95
        ),
        "first_readable_image_ms_mean": (
            round(statistics.fmean(first_readable_image_timings), 3)
            if first_readable_image_timings else None
        ),
        "public_first_image_ms_p50": quantile(public_first_image_timings, 0.50),
        "public_first_image_ms_p95": quantile(public_first_image_timings, 0.95),
        "public_first_image_ms_mean": (
            round(statistics.fmean(public_first_image_timings), 3)
            if public_first_image_timings else None
        ),
        "quality_flag_counts": dict(sorted(flag_counts.items())),
    }


def choose_audit_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose 30 deterministic samples: 1/group plus a rotating second sample."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("include_in_benchmark") and row.get("status") == "success":
            grouped[(str(row["subject"]), str(row["source_group"]))].append(row)
    selected: list[dict[str, Any]] = []
    for group_index, key in enumerate(sorted(grouped)):
        members = sorted(grouped[key], key=lambda row: int(row["source_index"]))
        selected.append(members[group_index % len(members)])
        if group_index % 2 == 0 and len(members) > 1:
            selected.append(members[(group_index * 3 + 7) % len(members)])
    unique: dict[str, dict[str, Any]] = {}
    for row in selected:
        unique[str(row["question_id"])] = row
    return list(unique.values())[:30]


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def wrap_chars(text: str, limit: int = 58) -> list[str]:
    lines: list[str] = []
    for source_line in text.splitlines() or [""]:
        value = source_line.strip()
        if not value:
            continue
        lines.extend(value[index:index + limit] for index in range(0, len(value), limit))
    return lines or ["（OCR为空）"]


def write_audit_pages(
    dataset: Path,
    output: Path,
    sample: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]] | None = None,
) -> None:
    pages = output / "manual_audit_pages"
    pages.mkdir(parents=True, exist_ok=True)
    title_font = font(30)
    body_font = font(27)
    for row in sample:
        with Image.open(dataset / str(row["image_path"])) as opened:
            source = opened.convert("RGB")
        source.thumbnail((1280, 720), Image.Resampling.LANCZOS)
        transcript_lines = wrap_chars(readable_ocr_text(row.get("transcript") or ""))
        canvas_h = 80 + source.height + 50 + len(transcript_lines) * 40 + 40
        canvas = Image.new("RGB", (1400, canvas_h), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (40, 24),
            f"{row['question_id']} | {row['subject']} | 字符置信度 "
            f"{float(row.get('weighted_character_confidence') or 0.0):.3f}",
            fill=(15, 18, 22),
            font=title_font,
        )
        canvas.paste(source, ((1400 - source.width) // 2, 75))
        y = 95 + source.height
        draw.line((35, y, 1365, y), fill=(190, 195, 200), width=2)
        y += 20
        draw.text((40, y), "当前工程 OCR 输出：", fill=(15, 18, 22), font=title_font)
        y += 45
        for line in transcript_lines:
            draw.text((55, y), line, fill=(25, 28, 32), font=body_font)
            y += 40
        canvas.save(pages / f"{row['question_id']}.png", compress_level=3)
    template_rows: list[dict[str, Any]] = []
    for row in audit_rows if audit_rows is not None else sample:
        template_rows.append(
            {
                "question_id": row["question_id"],
                "subject": row["subject"],
                "source_group": row["source_group"],
                "source_index": row["source_index"],
                "image_path": row["image_path"],
                "weighted_character_confidence": row.get("weighted_character_confidence", 0.0),
                "manual_exact_normalized": "",
                "manual_semantic_complete": "",
                "manual_error_count": "",
                "manual_error_types": "",
                "manual_notes": "",
            }
        )
    with (output / "manual_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(template_rows[0].keys()))
        writer.writeheader()
        writer.writerows(template_rows)


def write_audit_index(dataset: Path, output: Path, rows: list[dict[str, Any]]) -> None:
    """Write a self-contained visual index for reviewing every generated OCR image."""

    def image_uri(value: str | Path | None) -> str:
        if not value:
            return ""
        path = Path(value)
        return path.resolve().as_uri() if path.exists() else ""

    cards: list[str] = []
    ordered = sorted(
        rows,
        key=lambda row: (
            bool(row.get("semantic_package_pass")),
            str(row.get("subject")),
            str(row.get("question_id")),
        ),
    )
    for row in ordered:
        question_id = str(row.get("question_id") or "")
        subject = str(row.get("subject") or "")
        passed = bool(row.get("semantic_package_pass"))
        issues = ", ".join(str(item) for item in row.get("semantic_issue_types", []))
        source_uri = image_uri(dataset / str(row.get("image_path") or ""))
        rendered_uri = image_uri(row.get("rendered_ocr_path"))
        evidence_uri = image_uri(row.get("selected_source_path"))
        illustration_uris = [
            uri for uri in (image_uri(item) for item in row.get("illustration_paths", [])) if uri
        ]
        image_blocks = []
        for label, uri in (
            ("数据集原图", source_uri),
            ("最终可读 OCR 图", rendered_uri),
            ("后台题目证据", evidence_uri),
        ):
            if uri:
                image_blocks.append(
                    f'<figure><figcaption>{html.escape(label)}</figcaption>'
                    f'<a href="{html.escape(uri)}"><img loading="lazy" src="{html.escape(uri)}"></a></figure>'
                )
        figure_captions = [str(item or "") for item in row.get("figure_captions", [])]
        for index, uri in enumerate(illustration_uris, 1):
            printed_caption = (
                figure_captions[index - 1]
                if index - 1 < len(figure_captions)
                else ""
            )
            label = f"保留插图 {index}"
            if printed_caption:
                label += f"·{printed_caption}"
            image_blocks.append(
                f'<figure><figcaption>{html.escape(label)}</figcaption>'
                f'<a href="{html.escape(uri)}"><img loading="lazy" src="{html.escape(uri)}"></a></figure>'
            )
        first_ms = row.get("public_first_image_ms")
        latency = f"{float(first_ms) / 1000.0:.2f}s" if first_ms is not None else "未记录"
        state = "通过" if passed else "需人工核验"
        error_checks = "".join(
            f'<label><input type="checkbox" value="{html.escape(value)}">{html.escape(label)}</label>'
            for value, label in (
                ("text_missing", "文字缺失"),
                ("text_error", "文字错识别"),
                ("formula_error", "公式错误"),
                ("layout_error", "排版问题"),
                ("figure_missing", "插图缺失"),
                ("figure_incomplete", "插图裁剪不完整"),
                ("handwriting_leak", "手写误识别"),
                ("table_error", "表格错误"),
                ("other", "其他"),
            )
        )
        cards.append(
            f'<article class="card" data-id="{html.escape(question_id)}" '
            f'data-subject="{html.escape(subject)}" '
            f'data-state="{"pass" if passed else "review"}" '
            f'data-group="{html.escape(str(row.get("source_group") or ""))}" '
            f'data-index="{html.escape(str(row.get("source_index") or ""))}" '
            f'data-image="{html.escape(str(row.get("image_path") or ""))}" '
            f'data-confidence="{float(row.get("weighted_character_confidence") or 0.0):.6f}" '
            f'data-human="" '
            f'data-search="{html.escape((question_id + " " + issues).lower())}">'
            f'<header><h2>{html.escape(question_id)}</h2><div class="badges">'
            f'<span>{"数学" if subject == "mathematics" else "物理"}</span>'
            f'<span class="{"ok" if passed else "warn"}">{state}</span>'
            f'<span>首图 {latency}</span><span>配图 {int(row.get("illustration_count") or 0)}</span>'
            f'</div><p class="issues">{html.escape(issues or "未发现自动结构告警")}</p></header>'
            f'<div class="images">{"".join(image_blocks)}</div>'
            f'<details><summary>查看 OCR 文字</summary><pre>{html.escape(readable_ocr_text(row.get("transcript") or ""))}</pre></details>'
            f'<section class="manual"><div class="manual-head"><strong>人工标注</strong>'
            f'<span class="manual-state">未标注</span></div>'
            f'<div class="decision"><button type="button" data-decision="pass">通过</button>'
            f'<button type="button" data-decision="fail">不通过</button>'
            f'<button type="button" data-decision="uncertain">待确认</button>'
            f'<button type="button" data-decision="">清除</button></div>'
            f'<div class="error-types">{error_checks}</div>'
            f'<textarea rows="3" placeholder="补充问题位置、漏字内容或修改建议"></textarea></section>'
            f'</article>'
        )
    review_count = sum(not bool(row.get("semantic_package_pass")) for row in rows)
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Test6 OCR 全量审核</title>
<style>
:root{{--ink:#19212b;--muted:#68717d;--line:#d8dee6;--paper:#fff;--bg:#f3f5f7;--ok:#137333;--warn:#a33d00}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 "Microsoft YaHei",sans-serif}}
.top{{position:sticky;top:0;z-index:2;padding:18px 24px;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line)}}
h1{{font-size:22px;margin:0 0 10px}}.summary{{color:var(--muted);margin:0 0 12px}}.filters{{display:flex;gap:10px;flex-wrap:wrap}}
input,select,textarea,button{{font:inherit}}.filters input,.filters select{{border:1px solid #b8c0ca;border-radius:8px;background:white;padding:9px 12px}}.filters input{{min-width:260px;flex:1}}
main{{max-width:1500px;margin:20px auto;padding:0 18px}}.card{{background:var(--paper);border:1px solid var(--line);border-radius:12px;margin:0 0 18px;padding:18px;box-shadow:0 2px 10px rgba(24,33,43,.05)}}
.card h2{{font-size:18px;margin:0}}.badges{{display:flex;gap:7px;flex-wrap:wrap;margin-top:8px}}.badges span{{background:#eef2f6;border-radius:999px;padding:3px 9px}}.badges .ok{{color:var(--ok);background:#e8f5ec}}.badges .warn{{color:var(--warn);background:#fff1e8}}.issues{{color:var(--muted);margin:8px 0 0}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-top:15px;align-items:start}}figure{{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#fafafa;align-self:start}}figcaption{{padding:7px 10px;background:#edf1f5;font-weight:600}}img{{display:block;width:100%;height:auto;max-height:520px;object-fit:contain;object-position:top center;background:white}}details{{margin-top:14px}}pre{{white-space:pre-wrap;word-break:break-word;background:#f7f8fa;border-radius:8px;padding:14px;font:15px/1.65 "Microsoft YaHei",sans-serif}}
.manual{{margin-top:14px;padding:14px;border:1px solid #cbd5e1;border-radius:10px;background:#f8fafc}}.manual-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}.manual-state{{color:var(--muted)}}
.decision{{display:flex;gap:8px;flex-wrap:wrap}}.decision button,.export-button{{border:1px solid #aeb8c5;border-radius:7px;background:white;padding:7px 13px;cursor:pointer}}.decision button.active[data-decision="pass"]{{background:#daf0df;color:#0b6728;border-color:#75ba87}}.decision button.active[data-decision="fail"]{{background:#ffe4d8;color:#983000;border-color:#e59a77}}.decision button.active[data-decision="uncertain"]{{background:#fff4cf;color:#775600;border-color:#d7bd61}}
.error-types{{display:flex;gap:8px 16px;flex-wrap:wrap;margin:12px 0}}.error-types label{{white-space:nowrap}}.error-types input{{margin-right:5px}}textarea{{width:100%;border:1px solid #b8c0ca;border-radius:8px;padding:9px;resize:vertical;background:white}}.export-button{{background:#1e63b6;color:white;border-color:#1e63b6}}.audit-count{{font-weight:600;color:#1e63b6}}.save-status{{margin:9px 0 0;color:#137333}}.save-status.error{{color:#a33d00}}.save-status a{{color:#1e63b6;font-weight:600}}
.hidden{{display:none}}@media(max-width:650px){{.top{{position:static}}img{{height:auto;max-height:420px}}}}
</style></head><body>
<section class="top"><h1>Test6 OCR 全量审核</h1><p class="summary">共 {len(rows)} 题；自动语义包通过 {len(rows)-review_count} 题；需人工核验 {review_count} 题；<span id="auditCount" class="audit-count">人工已标注 0/{len(rows)}</span>。</p>
<div class="filters"><input id="search" placeholder="搜索题号或告警"><select id="subject"><option value="">全部学科</option><option value="mathematics">数学</option><option value="physics">物理</option></select><select id="state"><option value="">全部自动状态</option><option value="review">只看需核验</option><option value="pass">只看自动通过</option></select><select id="human"><option value="">全部人工状态</option><option value="unmarked">未标注</option><option value="pass">人工通过</option><option value="fail">人工不通过</option><option value="uncertain">待确认</option></select><button id="saveNow" class="export-button" type="button">立即保存</button><button id="exportCsv" class="export-button" type="button">导出标注 CSV</button><button id="exportJson" class="export-button" type="button">导出标注 JSON</button><button id="importJson" type="button">导入标注 JSON</button><input id="importFile" type="file" accept="application/json,.json" hidden></div><p id="saveStatus" class="save-status">正在检查保存状态…</p><p id="fileModeHelp" class="save-status error" hidden>当前是本地文件模式，浏览器可能禁止持久保存。<a href="http://127.0.0.1:8230/ocr-audit/iteration">打开可保存的审核地址</a>。切换前可先导出 JSON，再在新页面导入。</p></section>
<main>{''.join(cards)}</main>
<script>
const q=document.querySelector('#search'),s=document.querySelector('#subject'),t=document.querySelector('#state'),h=document.querySelector('#human');
const cards=[...document.querySelectorAll('.card')];
const storageKey='test6-ocr-manual-audit-v3-full300-20260906';
const annotationsApi='/api/v1/ocr-audit/iteration/annotations';
const saveStatus=document.querySelector('#saveStatus'),fileModeHelp=document.querySelector('#fileModeHelp');
let reviews={{}};
let storageAvailable=true,saveTimer=0,saveInFlight=false,saveAgain=false;
try{{reviews=JSON.parse(localStorage.getItem(storageKey)||'{{}}')||{{}}}}catch(_error){{reviews={{}};storageAvailable=false}}
const stateNames={{pass:'人工通过',fail:'人工不通过',uncertain:'待确认'}};
function collect(c){{return{{question_id:c.dataset.id,subject:c.dataset.subject,source_group:c.dataset.group,source_index:c.dataset.index,image_path:c.dataset.image,weighted_character_confidence:c.dataset.confidence,manual_exact_normalized:'',manual_semantic_complete:c.dataset.human==='pass'?'true':c.dataset.human==='fail'?'false':'',manual_error_count:String([...c.querySelectorAll('.error-types input:checked')].length),manual_error_types:[...c.querySelectorAll('.error-types input:checked')].map(x=>x.value).join(';'),manual_notes:c.querySelector('textarea').value.trim(),decision:c.dataset.human}}}}
function paint(c){{const value=c.dataset.human||'';c.querySelectorAll('[data-decision]').forEach(b=>b.classList.toggle('active',b.dataset.decision===value&&value));c.querySelector('.manual-state').textContent=stateNames[value]||'未标注'}}
function showSaveStatus(text,error=false){{saveStatus.textContent=text;saveStatus.classList.toggle('error',error)}}
function storeBrowser(){{try{{localStorage.setItem(storageKey,JSON.stringify(reviews));storageAvailable=true;return true}}catch(_error){{storageAvailable=false;return false}}}}
function applyReview(c){{const data=reviews[c.dataset.id]||{{}};c.dataset.human=data.decision||'';const types=new Set(String(data.manual_error_types||'').split(';').filter(Boolean));c.querySelectorAll('.error-types input').forEach(x=>x.checked=types.has(x.value));c.querySelector('textarea').value=data.manual_notes||'';paint(c)}}
async function saveServer(){{
  clearTimeout(saveTimer);
  if(!location.protocol.startsWith('http')){{showSaveStatus('本地文件模式不能可靠写入标注文件，请点击下方可保存地址。',true);fileModeHelp.hidden=false;return false}}
  if(saveInFlight){{saveAgain=true;return false}}
  saveInFlight=true;showSaveStatus('正在保存…');
  try{{const response=await fetch(annotationsApi,{{method:'POST',credentials:'same-origin',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{reviews}})}});if(!response.ok)throw new Error('HTTP '+response.status);const result=await response.json();showSaveStatus('已保存到工程文件 · 共 '+result.saved_count+' 题');return true}}
  catch(error){{showSaveStatus('保存失败：'+error.message+'。请先导出 JSON 备份。',true);return false}}
  finally{{saveInFlight=false;if(saveAgain){{saveAgain=false;void saveServer()}}}}
}}
function scheduleSave(immediate=false){{clearTimeout(saveTimer);if(immediate)void saveServer();else saveTimer=setTimeout(()=>void saveServer(),350)}}
function persist(c,immediate=false){{const data=collect(c);if(!data.decision&&!data.manual_error_types&&!data.manual_notes)delete reviews[c.dataset.id];else reviews[c.dataset.id]=data;const localOk=storeBrowser();paint(c);counts();filter();if(location.protocol.startsWith('http'))scheduleSave(immediate);else{{fileModeHelp.hidden=false;showSaveStatus(localOk?'已暂存，但 file:// 模式刷新后仍可能丢失。':'浏览器禁止本地文件保存，请改用下方可保存地址。',true)}}}}
function hydrate(c){{applyReview(c);c.querySelectorAll('[data-decision]').forEach(b=>b.addEventListener('click',()=>{{c.dataset.human=b.dataset.decision;persist(c,true)}}));c.querySelectorAll('.error-types input').forEach(x=>x.addEventListener('change',()=>persist(c,true)));c.querySelector('textarea').addEventListener('input',()=>persist(c,false))}}
function counts(){{document.querySelector('#auditCount').textContent='人工已标注 '+cards.filter(c=>c.dataset.human).length+'/{len(rows)}'}}
function filter(){{cards.forEach(c=>{{const humanOk=!h.value||(h.value==='unmarked'?!c.dataset.human:c.dataset.human===h.value);const show=(!q.value||c.dataset.search.includes(q.value.toLowerCase()))&&(!s.value||c.dataset.subject===s.value)&&(!t.value||c.dataset.state===t.value)&&humanOk;c.classList.toggle('hidden',!show)}})}}
function rowsForExport(){{return cards.map(c=>collect(c))}}
function download(name,text,type){{const link=document.createElement('a');link.href=URL.createObjectURL(new Blob([text],{{type}}));link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}}
function csvCell(value){{const text=String(value??'');return /[",\\r\\n]/.test(text)?'"'+text.replaceAll('"','""')+'"':text}}
document.querySelector('#exportCsv').addEventListener('click',()=>{{const rows=rowsForExport();const fields=['question_id','subject','source_group','source_index','image_path','weighted_character_confidence','manual_exact_normalized','manual_semantic_complete','manual_error_count','manual_error_types','manual_notes'];download('manual_audit.csv','\ufeff'+[fields.join(','),...rows.map(row=>fields.map(key=>csvCell(row[key])).join(','))].join('\\r\\n'),'text/csv;charset=utf-8')}});
document.querySelector('#exportJson').addEventListener('click',()=>download('manual_audit.json',JSON.stringify(rowsForExport(),null,2),'application/json;charset=utf-8'));
document.querySelector('#saveNow').addEventListener('click',()=>void saveServer());
const importFile=document.querySelector('#importFile');document.querySelector('#importJson').addEventListener('click',()=>importFile.click());
importFile.addEventListener('change',async()=>{{try{{const parsed=JSON.parse(await importFile.files[0].text());const items=Array.isArray(parsed)?parsed:Object.values(parsed.reviews||{{}});for(const item of items){{if(!item||!item.question_id)continue;if(!item.decision&&!item.manual_error_types&&!item.manual_notes)continue;reviews[item.question_id]=item}}cards.forEach(applyReview);storeBrowser();counts();filter();await saveServer()}}catch(error){{showSaveStatus('导入失败：'+error.message,true)}}finally{{importFile.value=''}}}});
async function loadSaved(){{if(!location.protocol.startsWith('http')){{fileModeHelp.hidden=false;showSaveStatus(storageAvailable?'当前仅暂存在浏览器，建议改用可保存地址。':'当前浏览器禁止本地文件保存。',true);return}}try{{const response=await fetch(annotationsApi,{{credentials:'same-origin'}});if(!response.ok)throw new Error('HTTP '+response.status);const result=await response.json();reviews={{...(result.reviews||{{}}),...reviews}};cards.forEach(applyReview);storeBrowser();counts();filter();if(Object.keys(reviews).length>result.saved_count)await saveServer();else showSaveStatus('已连接工程标注文件 · 已保存 '+result.saved_count+' 题')}}catch(error){{showSaveStatus('无法读取工程标注：'+error.message,true)}}}}
cards.forEach(hydrate);[q,s,t,h].forEach(x=>x.addEventListener('input',filter));counts();filter();void loadSaved();
</script></body></html>"""
    (output / "AUDIT_INDEX.html").write_text(page, encoding="utf-8")


def write_outputs(dataset: Path, output: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: str(row.get("question_id")))
    summary = {
        "dataset": str(dataset),
        "dataset_declared_count": 300,
        "unique_benchmark_count": sum(bool(row.get("include_in_benchmark")) for row in rows),
        "ground_truth_transcript_count": 0,
        "accuracy_status": "manual_audit_required_no_gold_transcripts",
        "all_300": aggregate(rows),
        "unique_298": aggregate([row for row in rows if row.get("include_in_benchmark")]),
        "by_subject": {
            subject: aggregate([row for row in rows if row.get("subject") == subject])
            for subject in ("mathematics", "physics")
        },
        "by_subject_group": {
            f"{subject}/{group}": aggregate([
                row for row in rows
                if row.get("subject") == subject and row.get("source_group") == group
            ])
            for subject in ("mathematics", "physics")
            for group in (f"Test{index}" for index in range(1, 11))
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_fields = [
        "question_id", "subject", "source_group", "source_index", "image_path",
        "include_in_benchmark", "duplicate_of", "status", "compact_character_count",
        "cjk_count", "line_count", "question_marker_detected", "weighted_character_confidence",
        "mean_item_confidence", "p10_item_confidence", "ocr_pass_count", "ocr_selected_pass",
        "illustration_count", "structured_ocr_requested", "structured_ocr_request_mode",
        "structured_ocr_gate_reasons",
        "structured_ocr_selected", "structured_formula_count",
        "structured_ocr_elapsed_ms", "structured_ocr_reason",
        "open_source_layout_requested", "open_source_layout_reused",
        "open_source_layout_region_count", "open_source_layout_elapsed_ms",
        "open_source_layout_error",
        "pre_crop_layout_selected_region_count", "pre_crop_layout_selected_regions",
        "pre_crop_layout_attachment",
        "pre_crop_question_box", "pre_crop_preliminary_text",
        "pre_crop_caption_items", "pre_crop_layout_source_regions",
        "pre_crop_layout_requested", "pre_crop_layout_elapsed_ms",
        "pre_crop_layout_external_region_count",
        "full_page_structured_ocr_selected", "full_page_structured_ocr_elapsed_ms",
        "full_page_structured_ocr_reason",
        "full_page_structured_formula_count",
        "semantic_structure_pass", "semantic_package_pass", "semantic_package_evidence",
        "semantic_choice_count_ambiguous",
        "semantic_requires_visual", "formula_visual_evidence_required", "full_question_visual_fallback",
        "semantic_issue_count", "semantic_issue_types", "semantic_recovery_actions", "total_ms", "quality_flags",
        "first_image_ms", "readable_ocr_ready_ms", "first_readable_image_ms",
        "public_first_image_ms", "background_refined_image_ms",
        "readable_package_ready_before_structured",
        "render_and_write_ms", "rendered_ocr_path",
        "selected_source_path", "illustration_paths", "figure_captions",
        "error", "transcript",
    ]
    with (output / "details.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flattened = dict(row)
            flattened["quality_flags"] = ";".join(row.get("quality_flags", []))
            flattened["semantic_recovery_actions"] = ";".join(
                row.get("semantic_recovery_actions", [])
            )
            flattened["semantic_issue_types"] = ";".join(
                row.get("semantic_issue_types", [])
            )
            flattened["illustration_paths"] = ";".join(
                row.get("illustration_paths", [])
            )
            flattened["figure_captions"] = ";".join(
                row.get("figure_captions", [])
            )
            writer.writerow(flattened)

    subset_fields = [
        "question_id", "subject", "source_group", "source_index", "image_path",
        "public_first_image_ms", "background_refined_image_ms", "total_ms",
        "semantic_structure_pass", "semantic_package_pass", "semantic_issue_types",
        "semantic_requires_visual", "illustration_count", "rendered_ocr_path",
        "selected_source_path", "illustration_paths", "figure_captions", "transcript",
    ]

    def write_subset(name: str, selected_rows: list[dict[str, Any]]) -> None:
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=subset_fields, extrasaction="ignore")
            writer.writeheader()
            for selected_row in selected_rows:
                flattened = dict(selected_row)
                flattened["semantic_issue_types"] = ";".join(
                    selected_row.get("semantic_issue_types", [])
                )
                flattened["illustration_paths"] = ";".join(
                    selected_row.get("illustration_paths", [])
                )
                flattened["figure_captions"] = ";".join(
                    selected_row.get("figure_captions", [])
                )
                writer.writerow(flattened)

    write_subset(
        "PUBLIC_FIRST_IMAGE_OVER_5S.csv",
        [row for row in rows if float(row.get("public_first_image_ms") or 0.0) > 5000.0],
    )
    write_subset(
        "BACKGROUND_REFINEMENT_OVER_5S.csv",
        [
            row for row in rows
            if float(row.get("background_refined_image_ms") or 0.0) > 5000.0
        ],
    )
    write_subset(
        "AUTO_VALIDATION_REVIEW.csv",
        [
            row for row in rows
            if not bool(row.get("semantic_structure_pass"))
            or not bool(row.get("semantic_package_pass"))
        ],
    )
    sample = choose_audit_sample(rows)
    write_audit_pages(dataset, output, sample, rows)
    write_audit_index(dataset, output, rows)
    report = [
        "# 300题本地OCR统计（待人工样本核验）",
        "",
        f"- 数据集：`{dataset}`",
        f"- 已运行：{summary['all_300']['success_count']}/{summary['all_300']['count']}",
        f"- 唯一候选：{summary['unique_298']['count']}",
        "- 标准题干文本：0（数据集原始说明明确未提供）",
        "- 因此：置信度和非空率是质量指标，不等同于真实OCR正确率。",
        "- 正确率将在 `manual_audit.csv` 完成人工视觉核验后写入最终报告。",
        "",
        "## 全量运行指标",
        "",
        f"- OCR成功率：{summary['all_300']['success_rate']:.2%}",
        f"- 非空文本率：{summary['all_300']['nonempty_rate']:.2%}",
        f"- 平均字符加权置信度：{summary['all_300']['mean_weighted_character_confidence']:.2%}",
        f"- 文字结构自动通过率：{summary['all_300']['semantic_text_pass_rate']:.2%}",
        f"- 文字+保留配图语义包自动通过率：{summary['all_300']['semantic_package_pass_rate']:.2%}",
        f"- PP-DocLayoutV2 触发/复用：{summary['all_300']['open_source_layout_requested_count']} / {summary['all_300']['open_source_layout_reused_count']}",
        f"- 开源版面区域采用数：{summary['all_300']['open_source_layout_added_region_count']}",
        f"- 总耗时 P50/P95：{summary['all_300']['total_ms_p50']} / {summary['all_300']['total_ms_p95']} ms",
        f"- 用户可见OCR首图均值：{summary['all_300']['public_first_image_ms_mean']} ms",
        f"- 用户可见OCR首图 P50/P95：{summary['all_300']['public_first_image_ms_p50']} / {summary['all_300']['public_first_image_ms_p95']} ms",
        f"- 后台精修首图 P50/P95：{summary['all_300']['first_image_ms_p50']} / {summary['all_300']['first_image_ms_p95']} ms",
        f"- 人工分层核验样本：{len(sample)}题",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--question-id",
        action="append",
        dest="question_ids",
        help="run only the named stable question ID; may be repeated",
    )
    parser.add_argument("--merge-from", type=Path, nargs="*")
    parser.add_argument(
        "--rerun-safety-affected",
        action="store_true",
        help="rerun rows that previously selected structured OCR or contain a solution-tail cue",
    )
    parser.add_argument(
        "--save-rendered",
        action="store_true",
        help="save every readable OCR image, selected evidence image, and figure crop",
    )
    return parser.parse_args()


def localize_existing_merged_artifacts(
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    """Point merged audit rows at copied artifacts inside the destination run."""

    artifact_root = output / "generated_results"
    scalar_fields = {
        "rendered_ocr_path": "ocr_rendered",
        "selected_source_path": "selected_question_evidence",
    }
    for row in rows:
        for field, folder in scalar_fields.items():
            source_value = row.get(field)
            if not source_value:
                continue
            local_path = artifact_root / folder / Path(str(source_value)).name
            if local_path.is_file():
                row[field] = str(local_path)
        localized_illustrations: list[str] = []
        for source_value in row.get("illustration_paths", []) or []:
            local_path = artifact_root / "illustrations" / Path(str(source_value)).name
            localized_illustrations.append(
                str(local_path) if local_path.is_file() else str(source_value)
            )
        row["illustration_paths"] = localized_illustrations


def main() -> int:
    args = parse_args()
    affinity = apply_production_cpu_affinity()
    print(f"[benchmark] cpu_affinity={affinity}", flush=True)
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    full_manifest = read_jsonl(dataset / "manifest.jsonl")
    if args.merge_from:
        merged: dict[str, dict[str, Any]] = {}
        for shard in args.merge_from:
            shard_results = shard / "results.jsonl" if shard.is_dir() else shard
            for row in read_jsonl(shard_results):
                merged[str(row["question_id"])] = row
        missing = [
            str(record["question_id"])
            for record in full_manifest
            if str(record["question_id"]) not in merged
        ]
        if missing:
            raise RuntimeError(f"Merged shards are missing {len(missing)} records: {missing[:5]}")
        rows = [merged[str(record["question_id"])] for record in full_manifest]
        localize_existing_merged_artifacts(rows, output)
        with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        write_outputs(dataset, output, rows)
        print(f"Merged {len(rows)} rows: {output}", flush=True)
        return 0
    if args.question_ids:
        wanted = set(args.question_ids)
        manifest = [
            record for record in full_manifest
            if str(record["question_id"]) in wanted
        ]
        found = {str(record["question_id"]) for record in manifest}
        missing_ids = sorted(wanted - found)
        if missing_ids:
            raise RuntimeError(f"Unknown question IDs: {missing_ids}")
    else:
        manifest = full_manifest[max(0, args.offset):]
        if args.limit > 0:
            manifest = manifest[: args.limit]
    results_path = output / "results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        completed = {str(row["question_id"]): row for row in read_jsonl(results_path)}
    safety_rerun_ids: set[str] = set()
    if args.rerun_safety_affected:
        for question_id, row in completed.items():
            transcript = str(row.get("transcript") or "")
            tail = transcript[len(transcript) // 2:]
            if bool(row.get("structured_ocr_selected")) or re.search(
                r"(?:^|\n)\s*(?:解\s*[:：]?|答\s*[:：]?|由.{0,14}(?:可知|得)|所以|因此)",
                tail,
            ):
                safety_rerun_ids.add(question_id)
    current_subject = ""
    current_module: Any = None
    current_engine: Any = None
    for index, record in enumerate(manifest, 1):
        question_id = str(record["question_id"])
        # A failed row is only a durable diagnostic, never a completed sample.
        # Long OCR runs may exhaust/fragment ONNX buffers; allowing a later
        # invocation to retry the row prevents one poisoned engine instance
        # from permanently turning the remainder of a serial audit into
        # failure placeholders.
        completed_ok = (
            question_id in completed
            and str(completed[question_id].get("status") or "") == "success"
        )
        if completed_ok and question_id not in safety_rerun_ids:
            print(f"[{index}/{len(manifest)}] skip {question_id}", flush=True)
            continue
        row = {
            key: record.get(key)
            for key in (
                "question_id", "subject", "source_group", "source_index", "image_path",
                "image_sha256", "include_in_benchmark", "duplicate_of",
            )
        }
        print(f"[{index}/{len(manifest)}] OCR {question_id}", flush=True)
        try:
            subject = str(record["subject"])
            if subject != current_subject:
                current_module = None
                current_engine = None
                gc.collect()
                current_module = load_test5(subject)
                current_engine = (
                    current_module.create_rapidocr_engine()
                    if hasattr(current_module, "create_rapidocr_engine")
                    else current_module.RapidOCR()
                )
                current_subject = subject
            source_path = dataset / str(record["image_path"])
            try:
                result = process_image(
                    current_module,
                    current_engine,
                    source_path,
                    output / "generated_results" if args.save_rendered else None,
                    question_id,
                )
            except Exception as first_exc:
                if "bad allocation" not in str(first_exc).lower():
                    raise
                # ONNX Runtime can retain fragmented buffers during a long
                # serial benchmark. Recreate only the current local engine and
                # retry this question once; never turn one transient allocation
                # failure into a missing benchmark row.
                current_engine = None
                gc.collect()
                current_engine = (
                    current_module.create_rapidocr_engine()
                    if hasattr(current_module, "create_rapidocr_engine")
                    else current_module.RapidOCR()
                )
                result = process_image(
                    current_module,
                    current_engine,
                    source_path,
                    output / "generated_results" if args.save_rendered else None,
                    question_id,
                )
                result.setdefault("quality_flags", []).append(
                    "retried_after_onnx_bad_allocation"
                )
            row.update(result)
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "quality_flags": ["processing_failure"],
                    "transcript": "",
                }
            )
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        completed[question_id] = row
    rows = [completed[str(record["question_id"])] for record in manifest]
    # Keep exactly one durable row per question after targeted reruns so future
    # resumes and audits cannot accidentally consume stale duplicate records.
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_outputs(dataset, output, rows)
    print(f"Done: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
