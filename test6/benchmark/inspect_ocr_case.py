"""Inspect Test6 OCR detections and transcript filtering for one image."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(r"D:\robot\Documents\ASK\picture\test6")
TEST5_PATHS = {
    "mathematics": ROOT / "math" / "source" / "test5_preprocess.py",
    "physics": ROOT / "physics" / "source" / "test5_preprocess.py",
}


def load_test5(subject: str) -> Any:
    path = TEST5_PATHS[subject]
    spec = importlib.util.spec_from_file_location(f"test6_inspect_test5_{subject}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def item_record(test5: Any, item: Any) -> dict[str, Any]:
    x1 = float(np.min(item.box[:, 0]))
    y1 = float(np.min(item.box[:, 1]))
    x2 = float(np.max(item.box[:, 0]))
    y2 = float(np.max(item.box[:, 1]))
    return {
        "text": item.text,
        "score": round(float(item.score), 4),
        "printed": bool(item.printed),
        "formula_like": bool(item.formula_like),
        "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "class": test5.classify_reflow_text(item.text),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--subject", choices=sorted(TEST5_PATHS), default="mathematics"
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    test5 = load_test5(args.subject)
    config = test5.Config()
    engine = test5.RapidOCR()
    image = test5.cv_read(args.image)
    rotated, rotation = test5.auto_rotate_fast(image)
    quad = test5.find_document_quad(rotated, config)
    enhanced, corrected = test5.perspective_warp(rotated, quad)
    enhanced, angle = test5.deskew(enhanced)
    enhanced = test5.normalize_illumination(enhanced)
    enhanced, outer_cropped, outer_box = test5.crop_outer_whitespace_for_ocr(enhanced)
    boxes, texts, scores, ocr_rotation, items, audit = test5.run_adaptive_ocr(
        enhanced, engine, config
    )
    question_box = test5.select_middle_question_box(
        enhanced.shape, items, config.select_middle_question
    )
    workspace, selected_items = test5.crop_question_workspace(
        enhanced, items, question_box
    )
    tiled_text = ""
    if audit.get("tiled_recovery_recommended") is True:
        test5.configure_ocr_detection(
            engine,
            box_thresh=0.25,
            unclip_ratio=1.50,
            text_score=config.ocr_tiled_min_text_score,
        )
        try:
            tiled_text, _tiled_ms, _tiled_passes = test5.run_tiled_ocr_transcript(
                workspace,
                engine,
                config,
                selected_items,
                force=True,
                full_page=bool(audit.get("tiled_full_page")),
            )
        finally:
            test5.configure_ocr_detection(
                engine,
                box_thresh=0.50,
                unclip_ratio=1.50,
                text_score=0.50,
            )
    mask = test5.create_text_mask(workspace.shape, selected_items)
    detection_copy, _ = test5.make_detection_copy(workspace, mask)
    figures, _ = test5.detect_illustration_boxes(
        detection_copy, workspace, selected_items, config
    )
    trusted_boxes = np.asarray(
        [item.box for item in selected_items if item.printed or item.formula_like],
        dtype=np.float32,
    )
    legacy_figures = test5.find_illustrations(
        workspace,
        config.illustration_padding,
        config.illustration_max_count,
        trusted_boxes if len(trusted_boxes) else None,
    )
    result = {
        "image": str(args.image),
        "shape": list(workspace.shape),
        "rotation": rotation,
        "ocr_rotation": ocr_rotation,
        "perspective_corrected": corrected,
        "deskew_angle": angle,
        "outer_cropped": outer_cropped,
        "outer_box": list(outer_box),
        "question_box": list(question_box),
        "audit": audit,
        "figures": [list(box) for box in figures],
        "legacy_figures": [list(box) for box in legacy_figures],
        "caption_figures": [
            list(box)
            for box in test5.caption_anchored_figure_boxes(
                detection_copy, selected_items
            )
        ],
        "transcript": test5.build_ocr_transcript(selected_items, figures),
        "tiled_transcript": tiled_text,
        "items": [
            item_record(test5, item)
            for item in sorted(selected_items, key=lambda value: (value.center_y, value.center_x))
        ],
    }
    if args.summary:
        result["item_count"] = len(result.pop("items"))
        result["transcript_length"] = len(result["transcript"])
        result["tiled_transcript_length"] = len(result["tiled_transcript"])
        result.pop("transcript")
        result.pop("tiled_transcript")
        result.pop("audit")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
