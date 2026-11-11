"""Summarize the two review halves (GPT 103 cases + current model 73 cases)."""
import json, glob, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "evaluation/current_model_math150/gpt_reviews"
M = ROOT / "evaluation/current_model_math150/model_reviews"

gpt = {Path(f).stem: json.loads(Path(f).read_text(encoding="utf-8")) for f in glob.glob(str(G / "*.json"))}
mdl = {Path(f).stem: json.loads(Path(f).read_text(encoding="utf-8")) for f in glob.glob(str(M / "*.json"))}

print("gpt", len(gpt), "model", len(mdl), "overlap", sorted(set(gpt) & set(mdl)), "union", len(set(gpt) | set(mdl)))
print("gpt  ", collections.Counter(r["status"] for r in gpt.values()))
print("model", collections.Counter(r["status"] for r in mdl.values()))

cat = collections.Counter()
for r in gpt.values():
    for fnd in r["findings"]:
        cat[fnd.get("category", "?") if isinstance(fnd, dict) else "str"] += 1
print("gpt finding categories", sorted(cat.items()))


def bucket(findings):
    txt = " ".join(f.get("evidence", "") if isinstance(f, dict) else f for f in findings)
    keys = []
    if "OCR" in txt or "识别" in txt or "缺失" in txt or "残缺" in txt:
        keys.append("OCR残缺/缺图")
    if "空" in txt or "429" in txt or "模板" in txt:
        keys.append("空输出/通用模板")
    if "公式卡" in txt:
        keys.append("公式卡错误/错配")
    if "泄露" in txt or "直接写" in txt or "选项" in txt or "一致" in txt:
        keys.append("答案泄露/选项越位")
    if "矛盾" in txt or "错误" in txt:
        keys.append("实质数学错误/自相矛盾")
    if "未" in txt and ("完成" in txt or "给出" in txt or "落地" in txt or "闭合" in txt):
        keys.append("解析未完成")
    return keys or ["其他"]


for name, rows in (("gpt", gpt), ("model", mdl)):
    c = collections.Counter()
    for k in rows:
        if rows[k]["status"] == "pass":
            continue
        for b in bucket(rows[k]["findings"]):
            c[b] += 1
    print(name, "非pass归类（可多重）", c.most_common())

by_sec = collections.defaultdict(lambda: collections.Counter())
for k, r in {**gpt, **mdl}.items():
    by_sec[k.split("-")[0]][r["status"]] += 1
for s in sorted(by_sec, key=int):
    print("section", s, dict(by_sec[s]))
