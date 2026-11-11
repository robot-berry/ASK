#!/usr/bin/env python3
"""Measure local layout + formula OCR on one photographed question."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(r"D:\robot\Documents\ASK\agentpicture\MH\test1")
MODEL_ROOT = (
    ROOT
    / "models"
    / "modelscope"
    / "models"
    / "OpenDataLab--PDF-Extract-Kit-1.0"
    / "snapshots"
    / "master"
    / "models"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full-layout-ocr", action="store_true")
    args = parser.parse_args()

    os.environ["MINERU_FORMULA_CH_SUPPORT"] = "True"
    os.environ["MINERU_TOOLS_CONFIG_JSON"] = str(ROOT / "mineru.json")
    os.environ["MINERU_MODEL_SOURCE"] = "local"
    started = time.perf_counter()

    if args.full_layout_ocr:
        from mineru.backend.pipeline.batch_analyze import BatchAnalyze
        from mineru.backend.pipeline.model_init import MineruPipelineModel

        model = MineruPipelineModel(
            formula_config={"enable": True},
            table_config={"enable": False},
            lang="ch",
            device=args.device,
        )

        class Manager:
            def get_model(self, **_kwargs):
                return model

        analyzer = BatchAnalyze(
            Manager(),
            batch_ratio=1,
            formula_enable=True,
            table_enable=False,
        )
        load_seconds = time.perf_counter() - started
        with Image.open(args.image) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        inference_started = time.perf_counter()
        layout = analyzer([(image, True, "ch")])[0]
        inference_seconds = time.perf_counter() - inference_started
        result = {
            "image": str(args.image.resolve()),
            "device": args.device,
            "mode": "full-layout-ocr",
            "load_seconds": round(load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "items": layout,
        }
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0

    from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel
    from mineru.model.mfr.pp_formulanet_plus_m.predict_formula import FormulaRecognizer

    layout_model = PPDocLayoutV2LayoutModel(
        str(MODEL_ROOT / "Layout" / "PP-DocLayoutV2"),
        args.device,
        conf=0.25,
        use_paddlex_filter_boxes=False,
    )
    formula_model = FormulaRecognizer(
        str(MODEL_ROOT / "MFR" / "pp_formulanet_plus_m"),
        args.device,
    )
    load_seconds = time.perf_counter() - started

    with Image.open(args.image) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    pixels = np.asarray(image)
    inference_started = time.perf_counter()
    layout = layout_model.predict(image, use_paddlex_filter_boxes=False)
    layout_seconds = time.perf_counter() - inference_started
    formula_items = [
        dict(item)
        for item in layout
        if item.get("label") in {"inline_formula", "display_formula"}
    ]
    recognition_started = time.perf_counter()
    formulas = formula_model.predict(formula_items, pixels, batch_size=16)
    recognition_seconds = time.perf_counter() - recognition_started
    result = {
        "image": str(args.image.resolve()),
        "device": args.device,
        "load_seconds": round(load_seconds, 3),
        "layout_seconds": round(layout_seconds, 3),
        "formula_seconds": round(recognition_seconds, 3),
        "formula_count": len(formulas),
        "formulas": formulas,
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
