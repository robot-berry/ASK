from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent / "ocr_accuracy_300_20260902"
QUEUE_DIR = RUN_DIR / "full_audit_queue"
AUDIT_CSV = QUEUE_DIR / "full_manual_audit.csv"
DETAILS_CSV = RUN_DIR / "details.csv"
HTML_OUT = RUN_DIR / "FULL_SEMANTIC_AUDIT_300.html"
MD_OUT = RUN_DIR / "FULL_SEMANTIC_AUDIT_300.md"
JSON_OUT = RUN_DIR / "FULL_SEMANTIC_AUDIT_300.json"
EVIDENCE_JSON = RUN_DIR / "SEMANTIC_EVIDENCE_COVERAGE_300.json"

SUBJECT_LABELS = {"mathematics": "数学", "physics": "物理"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rate(complete: int, total: int) -> str:
    return f"{complete / total * 100:.2f}%" if total else "—"


def main() -> None:
    audit_rows = load_csv(AUDIT_CSV)
    detail_by_id = {row["question_id"]: row for row in load_csv(DETAILS_CSV)}
    if len(audit_rows) != 300:
        raise ValueError(f"预期300题，实际{len(audit_rows)}题")
    if any(row["manual_semantic_complete"] not in {"0", "1"} for row in audit_rows):
        raise ValueError("存在未审核题目")

    rows: list[dict[str, object]] = []
    for row in audit_rows:
        detail = detail_by_id.get(row["question_id"], {})
        error_types = [item for item in row["manual_error_types"].split(";") if item]
        rows.append(
            {
                "question_id": row["question_id"],
                "subject": row["subject"],
                "subject_label": SUBJECT_LABELS.get(row["subject"], row["subject"]),
                "source_group": row["source_group"],
                "source_index": int(row["source_index"]),
                "image_path": row["image_path"],
                "semantic_complete": row["manual_semantic_complete"] == "1",
                "error_types": error_types,
                "notes": row["manual_notes"],
                "confidence": float(row["weighted_character_confidence"] or 0),
                "total_ms": float(detail.get("total_ms") or 0),
                "transcript": detail.get("transcript", ""),
            }
        )

    totals: dict[str, dict[str, object]] = {}
    for key, selected in [
        ("overall", rows),
        ("mathematics", [row for row in rows if row["subject"] == "mathematics"]),
        ("physics", [row for row in rows if row["subject"] == "physics"]),
    ]:
        complete = sum(bool(row["semantic_complete"]) for row in selected)
        totals[key] = {
            "total": len(selected),
            "complete": complete,
            "incomplete": len(selected) - complete,
            "complete_rate": complete / len(selected) if selected else None,
        }

    error_counts = Counter(
        error_type
        for row in rows
        if not row["semantic_complete"]
        for error_type in row["error_types"]
    )
    by_test: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "complete": 0})
    for row in rows:
        key = f"{row['subject']}|{row['source_group']}"
        by_test[key]["total"] += 1
        by_test[key]["complete"] += int(bool(row["semantic_complete"]))

    payload = {
        "title": "300题全量OCR原题语义完整性视觉审核",
        "scope": "仅审核OCR原题文字与保留配图能否完整恢复题意；未调用答题模型；无标准答案文本，因此不是字符准确率。",
        "totals": totals,
        "error_counts": dict(error_counts.most_common()),
        "by_test": dict(sorted(by_test.items())),
        "rows": rows,
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    evidence = (
        json.loads(EVIDENCE_JSON.read_text(encoding="utf-8"))
        if EVIDENCE_JSON.is_file()
        else None
    )
    make_markdown(rows, totals, error_counts, by_test, evidence)
    make_html(rows, totals, error_counts, evidence)
    print(
        f"reviewed=300 complete={totals['overall']['complete']} "
        f"math={totals['mathematics']['complete']}/150 "
        f"physics={totals['physics']['complete']}/150"
    )


def make_markdown(
    rows: list[dict[str, object]],
    totals: dict[str, dict[str, object]],
    error_counts: Counter[str],
    by_test: dict[str, dict[str, int]],
    evidence: dict[str, object] | None,
) -> None:
    lines = [
        "# 300题全量OCR原题语义完整性视觉审核",
        "",
        "> 范围：仅审核OCR原题文字与保留配图能否完整恢复题意；未调用答题模型；测试集没有标准答案文本，因此本报告不是字符准确率报告。",
        "",
        "## 总体结果",
        "",
        "| 范围 | 完整 | 不完整 | 完整率 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in [("overall", "全部"), ("mathematics", "数学"), ("physics", "物理")]:
        item = totals[key]
        lines.append(
            f"| {label} | {item['complete']} | {item['incomplete']} | "
            f"{rate(int(item['complete']), int(item['total']))} |"
        )

    if evidence:
        coverage = evidence["metric_boundary"]["source_pixel_evidence_coverage"]
        lines.extend(
            [
                "",
                "## 当前修正后的语义兜底",
                "",
                f"- 完整原题像素证据覆盖：**{coverage['covered']}/{coverage['total']}（{coverage['coverage_rate']:.2%}）**。",
                "- OCR文字继续用于可复制、可重排的正文；公式、上下标、漏行或选项错位以完整原题像素校验区为准。",
                "- 这项100%是“不丢原题证据”，不是把纯文字OCR准确率改写成100%。",
            ]
        )

    lines.extend(["", "## 判定标准", ""])
    lines.extend(
        [
            "- 完整：OCR文字与保留配图合起来，题干、已知量、符号、公式、全部选项/分问足以恢复原题语义。",
            "- 不完整：关键题干、数值、单位、指数/上下标、分式/根式、选项或分问有漏失、错换，可能改变解题条件或答案含义。",
            "- 不因少量标点、换行、题号变化判错；手写批注只有在遮蔽或污染原题含义时才判为不完整。",
        ]
    )

    lines.extend(["", "## 高频问题", "", "| 错误类型 | 题数 |", "|---|---:|"])
    for error_type, count in error_counts.most_common():
        lines.append(f"| `{error_type}` | {count} |")

    lines.extend(["", "## 分试卷结果", "", "| 学科 | 试卷 | 完整 | 不完整 | 完整率 |", "|---|---|---:|---:|---:|"])
    for key, item in sorted(by_test.items()):
        subject, group = key.split("|", 1)
        lines.append(
            f"| {SUBJECT_LABELS.get(subject, subject)} | {group} | {item['complete']} | "
            f"{item['total'] - item['complete']} | {rate(item['complete'], item['total'])} |"
        )

    incomplete = [row for row in rows if not row["semantic_complete"]]
    lines.extend(
        [
            "",
            f"## 被判定不完整的题目（{len(incomplete)}题）",
            "",
            "| 题号 | 学科/试卷 | 错误类型 | 判定说明 |",
            "|---|---|---|---|",
        ]
    )
    for row in incomplete:
        errors = ", ".join(f"`{item}`" for item in row["error_types"])
        notes = str(row["notes"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {row['question_id']} | {row['subject_label']}/{row['source_group']} | {errors} | {notes} |"
        )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def make_html(
    rows: list[dict[str, object]],
    totals: dict[str, dict[str, object]],
    error_counts: Counter[str],
    evidence: dict[str, object] | None,
) -> None:
    tests = sorted({str(row["source_group"]) for row in rows}, key=lambda value: int(value.removeprefix("Test")))
    error_types = sorted(error_counts)
    cards = []
    for row in rows:
        qid = str(row["question_id"])
        is_complete = bool(row["semantic_complete"])
        status = "完整" if is_complete else "不完整"
        status_class = "ok" if is_complete else "bad"
        error_text = "、".join(str(item) for item in row["error_types"]) or "无"
        search_text = " ".join(
            [qid, str(row["subject_label"]), str(row["source_group"]), error_text, str(row["notes"])]
        ).lower()
        cards.append(
            f"""
            <details class="case" data-subject="{html.escape(str(row['subject']))}"
              data-status="{'complete' if is_complete else 'incomplete'}"
              data-test="{html.escape(str(row['source_group']))}"
              data-errors="{html.escape(';'.join(str(item) for item in row['error_types']))}"
              data-search="{html.escape(search_text)}">
              <summary>
                <span class="qid">{html.escape(qid)}</span>
                <span>{html.escape(str(row['subject_label']))} · {html.escape(str(row['source_group']))}</span>
                <span class="pill {status_class}">{status}</span>
                <span class="note">{html.escape(str(row['notes']))}</span>
              </summary>
              <div class="meta">
                <span>错误类型：{html.escape(error_text)}</span>
                <span>OCR字符置信度：{float(row['confidence']):.3f}</span>
                <span>OCR用时：{float(row['total_ms']) / 1000:.3f}s</span>
              </div>
              <img class="audit-image" loading="lazy" alt="{html.escape(qid)} 原图与OCR输出对照"
                data-local-src="full_audit_queue/manual_audit_pages/{html.escape(qid)}.png"
                data-web-src="/ocr-audit/pages/{html.escape(qid)}.png">
            </details>"""
        )

    summary_cards = []
    for key, label in [("overall", "全部"), ("mathematics", "数学"), ("physics", "物理")]:
        item = totals[key]
        summary_cards.append(
            f"<div class='stat'><b>{label}</b><strong>{item['complete']}/{item['total']}</strong>"
            f"<span>语义完整率 {rate(int(item['complete']), int(item['total']))}</span></div>"
        )
    top_errors = "".join(
        f"<span class='error-chip'>{html.escape(name)} <b>{count}</b></span>"
        for name, count in error_counts.most_common(12)
    )
    test_options = "".join(f"<option value='{test}'>{test}</option>" for test in tests)
    error_options = "".join(
        f"<option value='{html.escape(error_type)}'>{html.escape(error_type)}（{error_counts[error_type]}）</option>"
        for error_type in error_types
    )
    coverage = (
        evidence["metric_boundary"]["source_pixel_evidence_coverage"]
        if evidence
        else None
    )
    upgrade_banner = (
        f"<section class='upgrade'><div><b>当前修正：完整原题证据覆盖 "
        f"{coverage['covered']}/{coverage['total']}（{coverage['coverage_rate']:.2%}）</b>"
        "<span>OCR文字负责可读排版；公式、上下标、漏行和选项错位由完整原题像素校验区兜底。</span></div>"
        "<em>注意：这是语义证据不丢失，不代表纯文字OCR达到100%。</em></section>"
        if coverage
        else ""
    )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>300题全量OCR语义审核</title>
<style>
:root{{--bg:#f4f7fb;--panel:#fff;--text:#172033;--muted:#667085;--line:#dfe5ee;--blue:#315efb;--bad:#c62828;--ok:#18794e}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif}}
.wrap{{max-width:1500px;margin:auto;padding:28px}} h1{{margin:0 0 6px;font-size:28px}} .scope{{color:var(--muted);margin:0 0 20px}}
.downloads{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0 18px}} .downloads a{{padding:8px 12px;border:1px solid #b8c7dc;border-radius:9px;background:#fff;color:#245b97;text-decoration:none;font-weight:700}}
.stats{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}} .stat{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:15px;display:flex;flex-direction:column}}
.stat strong{{font-size:27px}} .stat span{{color:var(--muted)}} .errors{{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0 18px}} .error-chip{{background:#eef2f8;border-radius:999px;padding:4px 10px;color:#39435a}}
.upgrade{{display:flex;justify-content:space-between;gap:18px;align-items:center;background:#eefbf4;border:1px solid #9bd7b7;border-radius:12px;padding:15px 17px;margin:15px 0}} .upgrade div{{display:flex;flex-direction:column}} .upgrade b{{color:#11643d;font-size:17px}} .upgrade span,.upgrade em{{color:#496357}} .upgrade em{{font-style:normal;max-width:420px}}
.toolbar{{position:sticky;top:0;z-index:3;display:grid;grid-template-columns:2fr repeat(3,1fr);gap:10px;background:rgba(244,247,251,.96);padding:12px 0}}
input,select{{width:100%;border:1px solid #cbd3df;border-radius:9px;padding:10px 11px;background:white;font:inherit}} .count{{margin:8px 0;color:var(--muted)}}
.case{{background:var(--panel);border:1px solid var(--line);border-radius:11px;margin:10px 0;overflow:hidden}} summary{{cursor:pointer;display:grid;grid-template-columns:190px 120px 78px 1fr;align-items:center;gap:12px;padding:13px 15px}}
.qid{{font-weight:700}} .pill{{width:max-content;border-radius:999px;padding:2px 9px;font-weight:700}} .pill.ok{{background:#e8f7ef;color:var(--ok)}} .pill.bad{{background:#ffebee;color:var(--bad)}} .note{{color:#44506a}}
.meta{{display:flex;gap:24px;flex-wrap:wrap;border-top:1px solid var(--line);padding:10px 15px;color:var(--muted)}} .audit-image{{display:block;max-width:100%;height:auto;margin:0 auto;padding:12px}}
.hidden{{display:none}} @media(max-width:800px){{.wrap{{padding:14px}}.stats{{grid-template-columns:1fr}}.toolbar{{grid-template-columns:1fr 1fr}}summary{{grid-template-columns:1fr}}.note{{display:none}}}}
</style>
</head>
<body><main class="wrap">
<h1>300题全量OCR原题语义完整性视觉审核</h1>
<p class="scope">仅审核OCR原题文字与保留配图能否完整恢复题意；未调用答题模型。测试集没有标准答案文本，因此这是人工视觉语义审核，不是字符准确率。</p>
<nav class="downloads"><a href="/">返回搜题测试</a><a href="/ocr-audit/data.csv">下载300题明细CSV</a><a href="/ocr-audit/report.md">下载文字报告</a><a href="/ocr-audit/evidence.md">下载100%兜底核验</a></nav>
{upgrade_banner}
<section class="stats">{''.join(summary_cards)}</section>
<div class="errors">{top_errors}</div>
<section class="toolbar">
  <input id="search" type="search" placeholder="搜索题号、错误类型或说明">
  <select id="subject"><option value="">全部学科</option><option value="mathematics">数学</option><option value="physics">物理</option></select>
  <select id="status"><option value="">完整与不完整</option><option value="incomplete">仅不完整</option><option value="complete">仅完整</option></select>
  <select id="test"><option value="">全部试卷</option>{test_options}</select>
  <select id="error"><option value="">全部错误类型</option>{error_options}</select>
</section>
<div id="count" class="count"></div>
<section id="cases">{''.join(cards)}</section>
</main>
<script>
const controls=["search","subject","status","test","error"].map(id=>document.getElementById(id));
const cases=[...document.querySelectorAll(".case")];
function applyFilters(){{
  const [search,subject,status,test,error]=controls.map(el=>el.value.trim().toLowerCase()); let shown=0;
  for(const item of cases){{
    const ok=(!search||item.dataset.search.includes(search))&&(!subject||item.dataset.subject===subject)&&(!status||item.dataset.status===status)&&(!test||item.dataset.test.toLowerCase()===test)&&(!error||item.dataset.errors.toLowerCase().split(";").includes(error));
    item.classList.toggle("hidden",!ok); if(ok) shown++;
  }}
  document.getElementById("count").textContent=`显示 ${{shown}} / 300 题`;
}}
for(const el of controls) el.addEventListener("input",applyFilters);
for(const item of cases) item.addEventListener("toggle",()=>{{
  if(!item.open) return; const img=item.querySelector("img"); if(!img.src) img.src=location.protocol==="file:"?img.dataset.localSrc:img.dataset.webSrc;
}});
applyFilters();
</script></body></html>"""
    HTML_OUT.write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
