"""Dump text evidence for the 73 unreviewed cases into one bundle for evidence-based review."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/current_model_math150"
progress = json.loads((OUT / "progress.json").read_text(encoding="utf-8"))
targets = [k for k in progress["cases"] if k.split("-")[0] in ("6", "7", "8", "9", "10")]
targets = [k for k in targets if not (k.startswith("6-") and int(k.split("-")[1]) <= 2)]
targets.sort(key=lambda k: tuple(map(int, k.split("-"))))

bundle = []
for key in targets:
    jid = progress["cases"][key]["job_id"]
    run = ROOT / "math/device_runs" / jid
    job = json.loads((run / "job.json").read_text(encoding="utf-8"))
    case = json.loads((OUT / (key + ".json")).read_text(encoding="utf-8"))
    audit = case.get("analysis_audit") or {}
    try:
        raw = json.loads((run / "audit/provider_response.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raw = {}
    pre = job.get("preprocess") or {}
    bundle.append({
        "case": key,
        "job_id": jid,
        "http_status": raw.get("http_status"),
        "ocr_text": pre.get("ocr_text"),
        "structure_gate": ((pre.get("analysis_reference_audit") or {}).get("ocr_quality_audit") or {}).get("semantic_completeness"),
        "raw_output_text": raw.get("output_text"),
        "output_json": job.get("output_json"),
        "validation": job.get("validation"),
        "normalized": {
            "knowledge_points": audit.get("knowledge_points"),
            "base_formulas": audit.get("base_formulas"),
            "parts": audit.get("parts"),
            "tiers": [
                {"tier_key": t.get("tier_key"), "title": t.get("title"),
                 "knowledge_points": t.get("knowledge_points"),
                 "base_formulas": t.get("base_formulas"),
                 "parts": t.get("parts"),
                 "pattern_summary": t.get("pattern_summary")}
                for t in audit.get("tiers") or []
            ],
        },
    })

(OUT / "review_bundle_73.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
print(len(bundle), (OUT / "review_bundle_73.json").stat().st_size)
