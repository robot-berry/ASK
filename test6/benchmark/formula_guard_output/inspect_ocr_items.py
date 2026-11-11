import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.stdout.reconfigure(encoding="utf-8")
module_path = ROOT / "test5" / "preprocess.py"
spec = importlib.util.spec_from_file_location("test5_current", module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)

image = module.cv_read(Path(__file__).parent / "enhanced_original" / "10-11.png")
engine = module.RapidOCR()
boxes, texts, scores, _ = module.run_ocr(image, engine)
items = module.analyze_ocr_items(boxes, texts, scores, module.Config())
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
