import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/current_model_math150"
B = json.loads((OUT / "review_bundle_73.json").read_text(encoding="utf-8"))

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="+")
ap.add_argument("--raw", action="store_true", default=True)
ap.add_argument("--norm", action="store_true")
a = ap.parse_args()
by = {e["case"]: e for e in B}
for c in a.cases:
    e = by[c]
    print("=" * 70)
    print(c, "http", e["http_status"])
    print("[OCR]", e["ocr_text"])
    raw = e["raw_output_text"] or ""
    print("[RAW]", raw if raw else "(EMPTY)")
    if a.norm or not raw:
        n = e["normalized"]
        for t in n["tiers"]:
            print(f"[{t['tier_key']}] kp={t['knowledge_points']}")
            print("   formulas=", t["base_formulas"])
            print("   pattern=", t["pattern_summary"])
            for p in t["parts"]:
                for m in p.get("methods", []):
                    print("   part", p.get("part"), "method", m.get("method_name"))
                    for s in m.get("steps", []):
                        print("     -", s)
                    if m.get("pitfall"):
                        print("     ! pitfall:", m["pitfall"])
