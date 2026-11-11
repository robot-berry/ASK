"""Build a local, filterable review page for OCR semantic-incomplete cases."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent / "ocr_accuracy_300_20260902"
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from presentation_guard import readable_ocr_text


AUDIT_PATH = ROOT / "manual_audit.csv"
RESULTS_PATH = ROOT / "results.jsonl"
OUTPUT_PATH = ROOT / "SEMANTIC_INCOMPLETE_VIEW.html"


ERROR_LABELS = {
    "missed_stem_and_formula": "题干与公式整段漏失",
    "missing_formula": "公式漏失",
    "missing_value": "关键数值漏失",
    "missing_constant": "常量漏失",
    "missing_fraction_denominators": "分式分母漏失",
    "formula_corruption": "公式损坏",
    "missing_fraction": "分式漏失",
    "missing_minus": "负号漏失",
    "infinity_corruption": "无穷符号损坏",
    "formula_omission": "关键公式未输出",
    "annotation_contamination": "手写内容混入",
    "missing_options": "选项漏失",
    "variable_substitution": "变量误识别",
    "adjacent_figure_contamination": "相邻题图混入",
    "table_omission": "表格内容漏失",
    "exponent_corruption": "指数损坏",
    "nuclear_symbol_corruption": "核反应上下标损坏",
    "extra_question_text": "相邻题文字混入",
    "missing_mass_ratio": "质量比漏失",
    "missing_subquestion_structure": "小问结构漏失",
    "handwriting_contamination": "手写演算混入",
    "symbol_substitution": "符号替换错误",
}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    with AUDIT_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        audit_rows = list(csv.DictReader(handle))
    results = {
        str(row["question_id"]): row
        for row in read_jsonl(RESULTS_PATH)
    }
    failures: list[dict[str, object]] = []
    for audit in audit_rows:
        if audit.get("manual_semantic_complete") != "0":
            continue
        question_id = str(audit["question_id"])
        result = results[question_id]
        errors = [
            item
            for item in str(audit.get("manual_error_types") or "").split(";")
            if item
        ]
        failures.append(
            {
                "question_id": question_id,
                "subject": str(audit["subject"]),
                "subject_label": "数学" if audit["subject"] == "mathematics" else "物理",
                "group": str(audit["source_group"]),
                "source_index": int(audit["source_index"]),
                "notes": str(audit.get("manual_notes") or ""),
                "errors": errors,
                "error_labels": [ERROR_LABELS.get(item, item) for item in errors],
                "confidence": round(float(audit["weighted_character_confidence"]) * 100, 2),
                "latency_seconds": round(float(result.get("total_ms") or 0) / 1000, 3),
                "ocr_text": readable_ocr_text(result.get("transcript") or ""),
            }
        )
    failures.sort(key=lambda row: (row["subject"], row["question_id"]))
    payload = json.dumps(failures, ensure_ascii=False).replace("</", "<\\/")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCR语义不完整题目复核</title>
  <style>
    :root {{ color-scheme: light; --ink:#18212f; --muted:#667085; --line:#dfe4ea; --blue:#3159c9; --red:#b42318; --bg:#f5f7fb; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--bg); font-family:"Microsoft YaHei UI","Microsoft YaHei",sans-serif; }}
    .wrap {{ width:min(1480px,calc(100% - 28px)); margin:0 auto; padding:26px 0 60px; }}
    .hero,.criteria,.card {{ background:#fff; border:1px solid var(--line); border-radius:18px; box-shadow:0 5px 20px rgba(16,24,40,.05); }}
    .hero {{ padding:24px 26px; }}
    h1 {{ margin:0 0 10px; font-size:30px; }}
    .lead {{ margin:0; color:var(--muted); line-height:1.75; }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin-top:20px; }}
    .stat {{ padding:14px 16px; border-radius:14px; background:#f2f5ff; }}
    .stat b {{ display:block; font-size:25px; margin-bottom:4px; }}
    .stat span {{ color:var(--muted); font-size:14px; }}
    .criteria {{ margin-top:16px; padding:20px 24px; }}
    .criteria h2 {{ margin:0 0 12px; font-size:20px; }}
    .criteria ol {{ margin:0; padding-left:22px; line-height:1.8; }}
    .criteria .fail {{ color:var(--red); font-weight:700; }}
    .toolbar {{ position:sticky; top:0; z-index:5; display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin:18px 0; padding:12px; border:1px solid var(--line); border-radius:14px; background:rgba(245,247,251,.94); backdrop-filter:blur(10px); }}
    button,select,input {{ min-height:40px; border:1px solid #cfd6df; border-radius:10px; background:#fff; padding:0 13px; font:inherit; }}
    button.active {{ color:#fff; background:var(--blue); border-color:var(--blue); }}
    input {{ flex:1; min-width:220px; }}
    .count {{ margin-left:auto; color:var(--muted); }}
    .grid {{ display:grid; gap:18px; }}
    .card {{ overflow:hidden; }}
    .card-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:12px; padding:18px 20px; border-bottom:1px solid var(--line); }}
    .title {{ margin:0; font-size:20px; }}
    .meta {{ margin-top:7px; color:var(--muted); font-size:14px; }}
    .subject {{ flex:none; padding:6px 11px; border-radius:999px; color:#2447a8; background:#eaf0ff; font-weight:700; }}
    .card-body {{ display:grid; grid-template-columns:minmax(0,1.6fr) minmax(300px,.7fr); gap:18px; padding:18px; }}
    .audit-image {{ width:100%; border:1px solid var(--line); border-radius:12px; background:white; }}
    .audit-link {{ display:block; }}
    .reason {{ padding:16px; border-radius:12px; background:#fff6f5; border:1px solid #fecdca; }}
    .reason h3,.ocr h3 {{ margin:0 0 10px; font-size:16px; }}
    .reason p {{ margin:0; line-height:1.7; }}
    .tags {{ display:flex; flex-wrap:wrap; gap:7px; margin-top:12px; }}
    .tag {{ padding:5px 8px; border-radius:8px; color:#912018; background:#fee4e2; font-size:13px; }}
    .ocr {{ margin-top:14px; }}
    details {{ border:1px solid var(--line); border-radius:12px; background:#fafbfc; }}
    summary {{ cursor:pointer; padding:13px 14px; font-weight:700; }}
    pre {{ margin:0; padding:0 14px 14px; white-space:pre-wrap; word-break:break-word; font:15px/1.7 "Microsoft YaHei UI",sans-serif; }}
    .empty {{ padding:50px; text-align:center; color:var(--muted); }}
    @media (max-width:900px) {{ .stats {{ grid-template-columns:repeat(2,1fr); }} .card-body {{ grid-template-columns:1fr; }} .count {{ width:100%; margin-left:0; }} }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>OCR语义不完整题目复核</h1>
      <p class="lead">数据集 testzip_math_physics_300_v1 · 当前本地OCR · 2026-09-02。下方只列出固定分层核验中判定为语义不完整的题目；点击对照图可查看原尺寸。</p>
      <div class="stats">
        <div class="stat"><b>18</b><span>语义不完整样本</span></div>
        <div class="stat"><b>10</b><span>数学题</span></div>
        <div class="stat"><b>8</b><span>物理题</span></div>
        <div class="stat"><b>11 / 29</b><span>抽样语义完整</span></div>
      </div>
    </section>
    <section class="criteria">
      <h2>判定标准</h2>
      <ol>
        <li>题干条件、设问、相关选项，以及会影响求解的数字、变量、正负号、上下标、单位和公式必须完整。</li>
        <li>几何图、电路图、仪表等可以保留为图片，但裁图必须完整并随OCR文字一并提供。</li>
        <li>只忽略空白、换行、全半角标点和纯排版差异；不忽略内容或数学/物理符号差异。</li>
        <li class="fail">只要存在一处可能改变题意、答案或求解条件的缺失/误识别，就判为“语义不完整”。</li>
        <li>审阅同时检查原始Unicode文字；审阅字体把有效下标显示成方框时，不直接算OCR错误。</li>
      </ol>
    </section>
    <div class="toolbar">
      <button class="active" data-subject="all">全部 18</button>
      <button data-subject="mathematics">数学 10</button>
      <button data-subject="physics">物理 8</button>
      <select id="errorFilter"><option value="all">全部错误类型</option></select>
      <input id="search" type="search" placeholder="搜索题号或判定原因">
      <span class="count" id="visibleCount"></span>
    </div>
    <section class="grid" id="cards"></section>
  </main>
  <script>
    const records = {payload};
    const cards = document.getElementById('cards');
    const errorFilter = document.getElementById('errorFilter');
    const search = document.getElementById('search');
    const visibleCount = document.getElementById('visibleCount');
    let subject = 'all';
    const errorNames = new Map();
    records.forEach(r => r.errors.forEach((e,i) => errorNames.set(e,r.error_labels[i])));
    [...errorNames.entries()].sort((a,b)=>a[1].localeCompare(b[1],'zh-CN')).forEach(([value,label]) => {{
      const option=document.createElement('option'); option.value=value; option.textContent=label; errorFilter.appendChild(option);
    }});
    function auditSrc(id) {{
      return location.protocol === 'file:' ? `manual_audit_pages/${{id}}.png` : `/ocr-audit/pages/${{encodeURIComponent(id)}}.png`;
    }}
    function render() {{
      const needle=search.value.trim().toLowerCase();
      const error=errorFilter.value;
      const shown=records.filter(r => (subject==='all'||r.subject===subject) && (error==='all'||r.errors.includes(error)) && (!needle||`${{r.question_id}} ${{r.notes}} ${{r.error_labels.join(' ')}}`.toLowerCase().includes(needle)));
      cards.replaceChildren(); visibleCount.textContent=`显示 ${{shown.length}} / ${{records.length}} 题`;
      if(!shown.length) {{ const node=document.createElement('div'); node.className='empty'; node.textContent='没有符合当前筛选条件的题目'; cards.appendChild(node); return; }}
      shown.forEach(r => {{
        const card=document.createElement('article'); card.className='card';
        const head=document.createElement('div'); head.className='card-head';
        const heading=document.createElement('div');
        const title=document.createElement('h2'); title.className='title'; title.textContent=r.question_id;
        const meta=document.createElement('div'); meta.className='meta'; meta.textContent=`${{r.group}} 第${{r.source_index}}题 · 字符置信度 ${{r.confidence}}% · 本地处理 ${{r.latency_seconds}}秒`;
        heading.append(title,meta); const badge=document.createElement('span'); badge.className='subject'; badge.textContent=r.subject_label; head.append(heading,badge);
        const body=document.createElement('div'); body.className='card-body';
        const link=document.createElement('a'); link.className='audit-link'; link.href=auditSrc(r.question_id); link.target='_blank'; link.rel='noopener';
        const img=document.createElement('img'); img.className='audit-image'; img.loading='lazy'; img.alt=`${{r.question_id}} 原图与OCR对照`; img.src=auditSrc(r.question_id); link.appendChild(img);
        const side=document.createElement('div'); const reason=document.createElement('div'); reason.className='reason';
        const reasonTitle=document.createElement('h3'); reasonTitle.textContent='为什么判为不完整'; const reasonText=document.createElement('p'); reasonText.textContent=r.notes;
        const tags=document.createElement('div'); tags.className='tags'; r.error_labels.forEach(label=>{{const tag=document.createElement('span');tag.className='tag';tag.textContent=label;tags.appendChild(tag);}}); reason.append(reasonTitle,reasonText,tags);
        const ocr=document.createElement('div'); ocr.className='ocr'; const details=document.createElement('details'); const summary=document.createElement('summary'); summary.textContent='展开纯文字OCR输出'; const pre=document.createElement('pre'); pre.textContent=r.ocr_text; details.append(summary,pre); ocr.appendChild(details); side.append(reason,ocr); body.append(link,side); card.append(head,body); cards.appendChild(card);
      }});
    }}
    document.querySelectorAll('[data-subject]').forEach(button=>button.addEventListener('click',()=>{{document.querySelectorAll('[data-subject]').forEach(item=>item.classList.remove('active'));button.classList.add('active');subject=button.dataset.subject;render();}}));
    errorFilter.addEventListener('change',render); search.addEventListener('input',render); render();
  </script>
</body>
</html>
"""
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} with {len(failures)} failures")


if __name__ == "__main__":
    main()
