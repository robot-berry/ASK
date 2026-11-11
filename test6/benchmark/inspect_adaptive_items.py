from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "math" / "source" / "test5_preprocess.py"
spec = importlib.util.spec_from_file_location("test6_test5_preprocess", SOURCE)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sys.stdout.reconfigure(encoding="utf-8")

engine = module.RapidOCR()
for name in ("10-12", "10-15"):
    image_path = (
        ROOT
        / "benchmark"
        / f"ocr_adaptive_{name}_20260827"
        / "enhanced_original"
        / f"{name}.png"
    )
    image = module.cv_read(image_path)
    print(f"\n=== {name} ===")
    for label, box_thresh, unclip_ratio, text_score in (
        ("primary", 0.50, 1.50, 0.50),
        ("fallback", 0.30, 2.00, 0.40),
    ):
        module.configure_ocr_detection(
            engine,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
            text_score=text_score,
        )
        boxes, texts, scores, _ = module.run_ocr(image, engine)
        items = module.analyze_ocr_items(boxes, texts, scores, module.Config())
        print(f"-- {label} --")
        for item in items:
            print(
                repr(item.text),
                f"score={item.score:.3f}",
                f"w={item.width:.1f}",
                f"h={item.height:.1f}",
                f"x={item.center_x:.1f}",
                f"y={item.center_y:.1f}",
                f"printed={item.printed}",
                f"formula={item.formula_like}",
            )
