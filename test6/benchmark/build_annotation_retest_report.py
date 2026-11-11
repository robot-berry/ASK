#!/usr/bin/env python3
"""Build a read-only before/after report for a human-annotated OCR subset."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_uri(value: str | Path | None) -> str:
    if not value:
        return ""
    path = Path(value).resolve()
    return path.as_uri() if path.exists() else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--retest-run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    with args.baseline_csv.open(encoding="utf-8-sig", newline="") as handle:
        annotations = list(csv.DictReader(handle))
    old_rows = {
        str(row.get("question_id") or ""): row
        for row in read_jsonl(args.baseline_run / "results.jsonl")
    }
    new_rows = {
        str(row.get("question_id") or ""): row
        for row in read_jsonl(args.retest_run / "results.jsonl")
    }

    comparison_rows: list[dict[str, Any]] = []
    cards: list[str] = []
    for annotation in annotations:
        question_id = str(annotation.get("question_id") or "")
        old = old_rows.get(question_id, {})
        new = new_rows.get(question_id, {})
        old_image = file_uri(old.get("rendered_ocr_path"))
        new_image = file_uri(new.get("rendered_ocr_path"))
        source_image = file_uri(args.dataset / str(new.get("image_path") or annotation.get("image_path") or ""))
        current_pass = bool(new.get("semantic_package_pass"))
        current_issues = ";".join(str(item) for item in new.get("semantic_issue_types", []))
        captions = ";".join(str(item) for item in new.get("figure_captions", []) if str(item))
        comparison_rows.append(
            {
                "question_id": question_id,
                "subject": new.get("subject") or annotation.get("subject") or "",
                "previous_decision": annotation.get("decision") or "",
                "previous_error_types": annotation.get("manual_error_types") or "",
                "previous_notes": annotation.get("manual_notes") or "",
                "current_auto_state": "structure_pass" if current_pass else "needs_human_review",
                "current_auto_issues": current_issues,
                "current_figure_count": int(new.get("illustration_count") or 0),
                "current_figure_captions": captions,
                "current_public_first_image_ms": new.get("public_first_image_ms"),
                "source_image": str(args.dataset / str(new.get("image_path") or annotation.get("image_path") or "")),
                "previous_rendered_image": old.get("rendered_ocr_path") or "",
                "current_rendered_image": new.get("rendered_ocr_path") or "",
                "human_recheck_required": "yes",
            }
        )
        figures = []
        for title, uri in (
            ("数据集原图", source_image),
            ("修改前 OCR 图", old_image),
            ("本轮 OCR 图", new_image),
        ):
            if uri:
                figures.append(
                    f'<figure><figcaption>{html.escape(title)}</figcaption>'
                    f'<a href="{html.escape(uri)}"><img loading="lazy" src="{html.escape(uri)}"></a></figure>'
                )
        state_class = "auto-pass" if current_pass else "auto-review"
        state_text = "自动结构检查通过，仍需人工复核" if current_pass else "自动检查仍需人工复核"
        cards.append(
            f'<article class="card" data-id="{html.escape(question_id)}" '
            f'data-state="{html.escape(state_class)}" '
            f'data-search="{html.escape((question_id + " " + str(annotation.get("manual_notes") or "") + " " + str(annotation.get("manual_error_types") or "")).lower())}">'
            f'<header><h2>{html.escape(question_id)}</h2><span class="{state_class}">{state_text}</span></header>'
            f'<section class="annotation"><strong>原人工批注</strong>'
            f'<p><b>问题类型：</b>{html.escape(str(annotation.get("manual_error_types") or "未填写"))}</p>'
            f'<p><b>批注内容：</b>{html.escape(str(annotation.get("manual_notes") or "未填写"))}</p></section>'
            f'<section class="current"><b>本轮自动告警：</b>{html.escape(current_issues or "无（不等于人工通过）")}<br>'
            f'<b>本轮图注：</b>{html.escape(captions or "未识别到印刷图注")}</section>'
            f'<div class="images">{"".join(figures)}</div>'
            f'<details><summary>查看本轮 OCR 文字</summary><pre>{html.escape(str(new.get("transcript") or ""))}</pre></details>'
            f'</article>'
        )

    csv_path = args.retest_run / "ANNOTATION_RETEST_COMPARISON.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    latencies = [
        float(row["current_public_first_image_ms"])
        for row in comparison_rows
        if row.get("current_public_first_image_ms") is not None
    ]
    pass_count = sum(row["current_auto_state"] == "structure_pass" for row in comparison_rows)
    avg_seconds = statistics.fmean(latencies) / 1000.0 if latencies else 0.0
    annotation_count = len(comparison_rows)
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{annotation_count}道人工批注 OCR 复测对照</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f5f7;color:#18212b;font:15px/1.55 "Microsoft YaHei",sans-serif}}
.top{{position:sticky;top:0;z-index:3;background:rgba(255,255,255,.97);border-bottom:1px solid #d8dee6;padding:16px 24px}}
h1{{font-size:22px;margin:0 0 6px}}.summary{{margin:0 0 10px;color:#586472}}input,select{{padding:8px 11px;border:1px solid #b8c0ca;border-radius:8px}}
input{{width:min(620px,70vw)}}main{{max-width:1700px;margin:18px auto;padding:0 18px}}.card{{background:white;border:1px solid #d8dee6;border-radius:12px;padding:18px;margin-bottom:18px}}
header{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}h2{{font-size:18px;margin:0}}header span{{border-radius:999px;padding:4px 10px}}
.auto-pass{{background:#e8f5ec;color:#137333}}.auto-review{{background:#fff1e8;color:#a33d00}}.annotation{{margin:12px 0;background:#fff8df;border-left:5px solid #d59b00;padding:10px 14px}}
.annotation p{{margin:4px 0}}.current{{margin:10px 0;color:#44505d}}.images{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;align-items:start}}
figure{{margin:0;border:1px solid #d8dee6;border-radius:9px;padding:8px;background:#fafbfc}}figcaption{{font-weight:700;margin-bottom:7px}}img{{display:block;width:100%;max-height:780px;object-fit:contain;background:white}}
details{{margin-top:12px}}pre{{white-space:pre-wrap;word-break:break-word;background:#f6f8fa;padding:12px;border-radius:8px}}.hidden{{display:none}}
@media(max-width:900px){{.images{{grid-template-columns:1fr}}}}
</style></head><body><section class="top"><h1>{annotation_count}道人工批注 OCR 复测对照</h1>
<p class="summary">共 {len(comparison_rows)} 道；本轮自动结构检查 {pass_count}/{len(comparison_rows)}；首图平均 {avg_seconds:.2f} 秒。所有题均保留“需人工复核”，不用自动结果覆盖原批注。</p>
<input id="search" placeholder="搜索题号、原批注或问题类型"><select id="state"><option value="">全部</option><option value="auto-review">仍有自动告警</option><option value="auto-pass">无自动告警</option></select></section>
<main>{''.join(cards)}</main><script>
const q=document.querySelector('#search'),s=document.querySelector('#state');function apply(){{const needle=q.value.trim().toLowerCase();document.querySelectorAll('.card').forEach(c=>c.classList.toggle('hidden',!!((needle&&!c.dataset.search.includes(needle))||(s.value&&c.dataset.state!==s.value))))}}q.addEventListener('input',apply);s.addEventListener('change',apply);
</script></body></html>"""
    (args.retest_run / "ANNOTATION_RETEST_COMPARISON.html").write_text(page, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
