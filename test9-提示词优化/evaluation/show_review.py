"""Print selected review records for calibration comparison."""
import json, sys

for c in sys.argv[1:]:
    for half in ("gpt_reviews", "model_reviews"):
        p = f"evaluation/current_model_math150/{half}/{c}.json"
        try:
            r = json.load(open(p, encoding="utf-8"))
        except FileNotFoundError:
            continue
        print("=" * 20, half, c, r["status"])
        print("ref:", r["reference_solution"][:300])
        for i, f in enumerate(r["findings"], 1):
            print(f"  [{i}]", f if isinstance(f, str) else f)
