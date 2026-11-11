"""Write per-case review records in gpt_reviews-compatible schema.

Usage: python evaluation/save_review.py <*.jsonl shard> [more shards...]
Each line: {"case","status","reference_solution","findings","review_limit_extra"?,"review_mode"?}
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation/current_model_math150/model_reviews"
OUT.mkdir(parents=True, exist_ok=True)
for arg in sys.argv[1:]:
    for line in Path(arg).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        rec = {
            "case": p["case"],
            "reviewer": "DeepSeek-V4.1-Flash (Claude Code session), text-only review",
            "review_mode": p.get(
                "review_mode",
                "OCR text from pipeline preprocess.ocr_text + complete raw provider response; "
                "normalized five-tier audit checked when raw empty; original PNG not viewable in text-only session",
            ),
            "status": p["status"],
            "reference_solution": p["reference_solution"],
            "findings": p["findings"],
            "review_limit": p.get("review_limit_extra", "")
            + "独立内容检查，未使用官方答案；未查看原图与最终图片排版（纯文本评审）。",
        }
        (OUT / (p["case"] + ".json")).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", p["case"], p["status"])
