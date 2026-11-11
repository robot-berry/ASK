"""Check completion status / provider source of all 150 case result JSONs."""
import json, glob, os, collections

files = [f for f in glob.glob("evaluation/current_model_math150/*-*.json") if os.path.basename(f)[0].isdigit()]
fallback = []
for f in files:
    k = os.path.basename(f)[:-5]
    r = json.load(open(f, encoding="utf-8"))
    audit = r.get("analysis_audit") or {}
    if audit.get("source") == "local_fallback":
        fallback.append(k)

fallback.sort(key=lambda k: (int(k.split("-")[0]), int(k.split("-")[1])))
print("local_fallback cases", len(fallback))
print(fallback)
