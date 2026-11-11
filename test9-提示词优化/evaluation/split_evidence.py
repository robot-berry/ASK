import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/current_model_math150"
B = json.loads((OUT / "review_bundle_73.json").read_text(encoding="utf-8"))
d = OUT / "review_evidence"
d.mkdir(exist_ok=True)
for e in B:
    (d / (e["case"] + ".json")).write_text(json.dumps(e, ensure_ascii=False, indent=2), encoding="utf-8")
print(len(B), "files ->", d)
