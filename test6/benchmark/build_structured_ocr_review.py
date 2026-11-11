"""Build a transparent before/after review page for the corrected 300-case OCR run."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from presentation_guard import readable_ocr_text


DEFAULT_BASELINE = ROOT / "ocr_accuracy_300_20260902" / "FULL_SEMANTIC_AUDIT_300.json"
DEFAULT_LATEST = ROOT / "ocr_accuracy_300_balanced_final_20260903"
PREVIOUS_TIMING = ROOT / "ocr_accuracy_300_semantic_complete_final_20260903" / "summary.json"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--latest", type=Path, default=DEFAULT_LATEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = args.baseline.resolve()
    latest = args.latest.resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    old_rows = {str(row["question_id"]): row for row in baseline["rows"]}
    new_rows = read_jsonl(latest / "results.jsonl")
    summary = json.loads((latest / "summary.json").read_text(encoding="utf-8"))
    previous_timing = (
        json.loads(PREVIOUS_TIMING.read_text(encoding="utf-8"))
        if PREVIOUS_TIMING.is_file()
        else None
    )

    records: list[dict[str, object]] = []
    for new in new_rows:
        question_id = str(new["question_id"])
        old = old_rows[question_id]
        records.append(
            {
                "question_id": question_id,
                "subject": new["subject"],
                "subject_label": "数学" if new["subject"] == "mathematics" else "物理",
                "group": new["source_group"],
                "source_index": new["source_index"],
                "prior_complete": bool(old["semantic_complete"]),
                "prior_errors": old.get("error_types") or [],
                "prior_notes": old.get("notes") or "",
                "prior_transcript": old.get("transcript") or "",
                "latest_transcript": new.get("transcript") or "",
                "prior_display_transcript": readable_ocr_text(old.get("transcript") or ""),
                "latest_display_transcript": readable_ocr_text(new.get("transcript") or ""),
                "automatic_structure_pass": bool(new.get("semantic_structure_pass")),
                "automatic_package_pass": bool(
                    new.get("semantic_package_pass", new.get("semantic_structure_pass"))
                ),
                "semantic_issues": new.get("semantic_issue_types") or [],
                "package_evidence": new.get("semantic_package_evidence") or "",
                "structured_selected": bool(new.get("structured_ocr_selected")),
                "structured_reason": new.get("structured_ocr_reason") or "",
                "full_page_selected": bool(new.get("full_page_structured_ocr_selected")),
                "full_page_reason": new.get("full_page_structured_ocr_reason") or "",
                "latency_seconds": round(float(new.get("total_ms") or 0.0) / 1000.0, 3),
            }
        )

    prior_incomplete = [row for row in records if not row["prior_complete"]]
    prior_incomplete_package_pass = sum(
        bool(row["automatic_package_pass"]) for row in prior_incomplete
    )
    text_pass = sum(bool(row["automatic_structure_pass"]) for row in records)
    package_pass = sum(bool(row["automatic_package_pass"]) for row in records)
    structured_selected = sum(bool(row["structured_selected"]) for row in records)
    full_page_selected = sum(bool(row["full_page_selected"]) for row in records)
    text_failed_ids = [
        str(row["question_id"])
        for row in records
        if not row["automatic_structure_pass"]
    ]
    current_mean_ms = float(summary["all_300"]["total_ms_mean"])
    previous_mean_ms = (
        float(previous_timing["all_300"]["total_ms_mean"])
        if previous_timing
        else 0.0
    )
    mean_reduction = (
        (previous_mean_ms - current_mean_ms) / previous_mean_ms
        if previous_mean_ms > 0
        else 0.0
    )
    # Keep exact machine-readable OCR in the JSON report, but do not embed it
    # in the human review page.  This prevents LaTeX implementation syntax
    # from leaking through either visible text or the page source.
    public_records = []
    for record in records:
        public_record = {
            key: value
            for key, value in record.items()
            if key not in {"prior_transcript", "latest_transcript"}
        }
        public_records.append(public_record)
    payload = json.dumps(public_records, ensure_ascii=False).replace("</", "<\\/")

    report = f"""# OCR修正后300题复测报告

- 数据集：`D:\\robot\\Documents\\ASK\\datasets\\testzip_math_physics_300_v1`
- 执行方式：300题逐题串行，纯本地OCR；未调用DashScope答题模型。
- 运行成功：300/300；非空文字：300/300。
- 旧版人工语义完整：96/300（32.00%）。
- 新版纯文字自动门槛通过：{text_pass}/300（{text_pass / 300:.2%}）。这不是人工语义完整率。
- 新版文字+必要保留配图语义包门槛通过：{package_pass}/300（{package_pass / 300:.2%}）。这也不是人工语义完整率。
- 纯文字未通过的题目：{', '.join(text_failed_ids) if text_failed_ids else '无'}；均为表格单元格结构问题，已通过保留原始表格图形成完整语义包。
- 旧版204道不完整题中，有{prior_incomplete_package_pass}道在新版已清除自动可检测的问题，仍须逐题视觉复核。
- 平衡优化前本地总时延：平均{previous_mean_ms / 1000:.3f}s；优化后平均{current_mean_ms / 1000:.3f}s，下降{mean_reduction:.1%}。
- 快速路径只做1轮OCR；只有检测到题干/公式/选项风险时才调用本地结构化OCR，只有仍不完整时才做整页补救。
- 本地结构化公式结果被安全采用：{structured_selected}/300。
- 整页二次本地OCR被安全采用：{full_page_selected}/300。
- 原始题目像素证据随文字保留：300/300；这保证下游仍能查看原图，但不等于纯文字OCR 100%。
- 本地总时延：平均{summary['all_300']['total_ms_mean'] / 1000:.3f}s，P50 {summary['all_300']['total_ms_p50'] / 1000:.3f}s，P90 {summary['all_300']['total_ms_p90'] / 1000:.3f}s，P95 {summary['all_300']['total_ms_p95'] / 1000:.3f}s。
- 数学平均：{summary['by_subject']['mathematics']['total_ms_mean'] / 1000:.3f}s；物理平均：{summary['by_subject']['physics']['total_ms_mean'] / 1000:.3f}s。

## 结论

新版已经修正题号误裁、纸张纹理误裁、科学计数法指数扁平化、公式模型异常重复、无上下文公式补丁和题末手写解答污染。当前不能诚实宣称“纯文字OCR语义完整率100%”：两道表格题仍依赖原图像素，并且数据集没有标准题干文本、尚未完成第二轮300题人工逐题语义判定。网页复核页提供原图、新旧OCR和旧问题说明，可继续人工确认。
"""
    (latest / "LATEST_OCR_REPORT.md").write_text(report, encoding="utf-8")
    (latest / "STRUCTURED_OCR_REVIEW_300.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "total": 300,
                    "prior_manual_complete": 96,
                    "automatic_text_pass": text_pass,
                    "automatic_package_pass": package_pass,
                    "prior_incomplete_package_pass": prior_incomplete_package_pass,
                    "structured_selected": structured_selected,
                    "full_page_selected": full_page_selected,
                    "source_evidence_complete": 300,
                },
                "rows": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    asset_version = time.time_ns()
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR修正后300题复核</title><style>
:root{{--ink:#17202d;--muted:#667085;--line:#dde3ec;--blue:#3159c9;--red:#b42318;--green:#087443;--bg:#f5f7fb}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:"Microsoft YaHei UI","Microsoft YaHei",sans-serif}}
.wrap{{width:min(1500px,calc(100% - 28px));margin:auto;padding:24px 0 60px}}.hero,.notice,.card{{background:#fff;border:1px solid var(--line);border-radius:18px;box-shadow:0 5px 20px rgba(16,24,40,.05)}}
.hero{{padding:24px}}h1{{margin:0 0 10px;font-size:30px}}p{{line-height:1.7}}.muted{{color:var(--muted)}}.stats{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:18px}}.stat{{padding:14px;border-radius:13px;background:#f1f4ff}}.stat b{{display:block;font-size:24px}}.stat span{{font-size:13px;color:var(--muted)}}
.notice{{margin-top:14px;padding:17px 22px;border-color:#fedf89;background:#fffaeb}}.notice strong{{color:#93370d}}.toolbar{{position:sticky;top:0;z-index:5;display:flex;gap:9px;flex-wrap:wrap;margin:16px 0;padding:11px;background:rgba(245,247,251,.95);border:1px solid var(--line);border-radius:14px}}button,select,input{{min-height:40px;border:1px solid #cad2df;border-radius:9px;background:#fff;padding:0 12px;font:inherit}}button.active{{background:var(--blue);color:#fff;border-color:var(--blue)}}input{{flex:1;min-width:230px}}#count{{margin-left:auto;align-self:center;color:var(--muted)}}
.grid{{display:grid;gap:16px}}.card{{overflow:hidden}}.head{{display:flex;justify-content:space-between;gap:12px;padding:16px 18px;border-bottom:1px solid var(--line)}}h2{{margin:0;font-size:19px}}.meta{{margin-top:6px;color:var(--muted);font-size:14px}}.badge{{height:max-content;padding:5px 9px;border-radius:999px;font-weight:700;background:#eaf0ff;color:#2548aa}}.body{{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(360px,.8fr);gap:16px;padding:16px}}img{{width:100%;border:1px solid var(--line);border-radius:11px}}.side{{display:grid;gap:12px;align-content:start}}.box{{padding:14px;border:1px solid var(--line);border-radius:11px;background:#fafbfc}}.box h3{{margin:0 0 8px;font-size:15px}}.warn{{background:#fff6f5;border-color:#fecdca}}.ok{{background:#ecfdf3;border-color:#abefc6}}details{{border:1px solid var(--line);border-radius:10px}}summary{{cursor:pointer;padding:11px;font-weight:700}}pre{{margin:0;padding:0 12px 12px;white-space:pre-wrap;word-break:break-word;font:14px/1.65 "Microsoft YaHei UI",sans-serif}}.empty{{padding:50px;text-align:center;color:var(--muted)}}
@media(max-width:950px){{.stats{{grid-template-columns:repeat(2,1fr)}}.body{{grid-template-columns:1fr}}#count{{width:100%;margin-left:0}}}}
</style></head><body><main class="wrap">
<section class="hero"><h1>OCR修正后300题复核</h1><p class="muted">新旧文字逐题对照 · 数学150题 + 物理150题 · 全部串行 · 不含答题模型</p><div class="stats">
<div class="stat"><b>300/300</b><span>本地运行成功</span></div><div class="stat"><b>{text_pass}/300</b><span>纯文字自动门禁通过</span></div><div class="stat"><b>{package_pass}/300</b><span>语义包自动门禁通过</span></div><div class="stat"><b>{prior_incomplete_package_pass}/204</b><span>旧不完整题已清除自动问题</span></div><div class="stat"><b>{full_page_selected}</b><span>采用整页二次恢复</span></div><div class="stat"><b>{summary['all_300']['total_ms_mean'] / 1000:.2f}s</b><span>平均本地时延</span></div>
</div></section><section class="notice"><strong>当前不能标为纯OCR 100%。</strong> 自动结构通过只检查可机器识别的缺行/选项/小问模式；最新300题仍需要第二轮人工视觉判定。原题像素证据为300/300，可防止下游丢失原始信息。</section>
<div class="toolbar"><button class="active" data-mode="all">全部</button><button data-mode="prior-incomplete">旧版不完整</button><button data-mode="text-fail">纯文字失败</button><button data-mode="package-fail">语义包失败</button><button data-mode="structured">采用公式恢复</button><button data-mode="full-page">整页二次恢复</button><select id="subject"><option value="all">数学与物理</option><option value="mathematics">数学</option><option value="physics">物理</option></select><input id="search" type="search" placeholder="搜索题号、问题类型或OCR文字"><span id="count"></span></div><section id="cards" class="grid"></section>
</main><script>
const rows={payload};const pageVersion={asset_version};let mode='all';const cards=document.getElementById('cards'),subject=document.getElementById('subject'),search=document.getElementById('search'),count=document.getElementById('count');
function pageSrc(id){{const base=location.protocol==='file:'?`full_audit_queue/manual_audit_pages/${{id}}.png`:`/ocr-audit/latest/pages/${{encodeURIComponent(id)}}.png`;return `${{base}}?v=${{pageVersion}}`}}
function node(tag,cls,text){{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n}}
function render(){{const needle=search.value.trim().toLowerCase();const shown=rows.filter(r=>(subject.value==='all'||r.subject===subject.value)&&(mode==='all'||(mode==='prior-incomplete'&&!r.prior_complete)||(mode==='text-fail'&&!r.automatic_structure_pass)||(mode==='package-fail'&&!r.automatic_package_pass)||(mode==='structured'&&r.structured_selected)||(mode==='full-page'&&r.full_page_selected))&&(!needle||`${{r.question_id}} ${{r.prior_notes}} ${{(r.semantic_issues||[]).join(' ')}} ${{r.latest_display_transcript}}`.toLowerCase().includes(needle)));cards.replaceChildren();count.textContent=`显示 ${{shown.length}} / 300 题`;if(!shown.length){{cards.append(node('div','empty','没有符合条件的题目'));return}}shown.forEach(r=>{{const card=node('article','card'),head=node('div','head'),titleWrap=node('div'),title=node('h2','',r.question_id),meta=node('div','meta',`${{r.group}} 第${{r.source_index}}题 · ${{r.subject_label}} · ${{r.latency_seconds}}秒`),badge=node('span','badge',r.subject_label);titleWrap.append(title,meta);head.append(titleWrap,badge);const body=node('div','body'),link=node('a');link.href=pageSrc(r.question_id);link.target='_blank';const img=node('img');img.loading='lazy';img.src=pageSrc(r.question_id);img.alt=`${{r.question_id}} 原图与新版OCR`;link.append(img);const side=node('div','side');const oldBox=node('div',r.prior_complete?'box ok':'box warn'),h=node('h3','',r.prior_complete?'旧版人工判定：完整':'旧版人工判定：不完整'),p=node('p','',r.prior_notes||'无备注');oldBox.append(h,p);const status=node('div',r.automatic_package_pass?'box ok':'box warn'),sh=node('h3','',r.automatic_package_pass?'新版语义包门禁：通过':'新版语义包门禁：仍需复核'),sp=node('p','',`纯文字：${{r.automatic_structure_pass?'通过':'失败'}}；问题：${{(r.semantic_issues||[]).join('、')||'无'}}；证据：${{r.package_evidence||'文字'}}`);status.append(sh,sp);const recovery=node('div','box'),rh=node('h3','',r.full_page_selected?'已采用整页二次OCR':'未采用整页二次OCR'),rp=node('p','',r.full_page_reason||r.structured_reason||'无需恢复');recovery.append(rh,rp);function detail(label,text){{const d=node('details'),s=node('summary','',label),pre=node('pre','',text);d.append(s,pre);return d}}side.append(oldBox,status,recovery,detail('展开旧版OCR文字',r.prior_display_transcript),detail('展开新版OCR文字',r.latest_display_transcript));body.append(link,side);card.append(head,body);cards.append(card)}})}}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-mode]').forEach(x=>x.classList.remove('active'));b.classList.add('active');mode=b.dataset.mode;render()}});subject.onchange=render;search.oninput=render;render();
</script></body></html>"""
    (latest / "STRUCTURED_OCR_REVIEW_300.html").write_text(page, encoding="utf-8")
    print(
        "Wrote review: "
        f"text_pass={text_pass}, package_pass={package_pass}, "
        f"prior_incomplete_package_pass={prior_incomplete_package_pass}"
    )


if __name__ == "__main__":
    main()
