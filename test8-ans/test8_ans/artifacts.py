from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

from .storage import append_jsonl, utc_now, write_json


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "p50_ms": _percentile(values, 0.50),
        "p90_ms": _percentile(values, 0.90),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": round(max(values), 3) if values else None,
    }


def export_comparison_artifacts(
    batch_dir: Path, results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Collect parsed images, raw-answer screenshots and latency comparisons."""

    root = batch_dir / "comparison_artifacts"
    guidance_root = root / "guidance_images"
    screenshot_root = root / "model_answer_screenshots"
    source_image_root = root / "source_question_images"
    guidance_root.mkdir(parents=True, exist_ok=False)
    screenshot_root.mkdir(parents=True, exist_ok=False)
    source_image_root.mkdir(parents=True, exist_ok=False)
    manifest_path = root / "comparison_manifest.jsonl"
    csv_path = root / "latency_comparison.csv"
    rows: list[dict[str, Any]] = []

    for result in results:
        item_id = str(result["id"])
        run_dir_value = result.get("run_dir")
        run_dir = Path(str(run_dir_value)) if run_dir_value else None
        copied_guidance: list[str] = []
        copied_screenshot: str | None = None
        copied_source_image: str | None = None
        validation: dict[str, Any] = {}
        if run_dir and run_dir.is_dir():
            validation_path = run_dir / "validation.json"
            if validation_path.is_file():
                validation = json.loads(
                    validation_path.read_text(encoding="utf-8-sig")
                )
            input_path = run_dir / "input.json"
            if input_path.is_file():
                input_record = json.loads(input_path.read_text(encoding="utf-8-sig"))
                source_image_value = input_record.get("source_image")
                if source_image_value:
                    source_image = Path(str(source_image_value))
                    if source_image.is_file():
                        suffix = source_image.suffix.lower() or ".png"
                        target = source_image_root / f"{item_id}{suffix}"
                        shutil.copy2(source_image, target)
                        copied_source_image = str(target.resolve())
            source_guidance = run_dir / "images_800px"
            if source_guidance.is_dir():
                destination = guidance_root / item_id
                destination.mkdir(parents=True, exist_ok=True)
                for tier in ("l1", "l2", "l3", "l4", "l5"):
                    source = source_guidance / f"{tier}.png"
                    if source.is_file():
                        target = destination / source.name
                        shutil.copy2(source, target)
                        copied_guidance.append(str(target.resolve()))
            source_screenshot = (
                run_dir / "model_answer_screenshot" / "model_output.png"
            )
            if source_screenshot.is_file():
                target = screenshot_root / f"{item_id}.png"
                shutil.copy2(source_screenshot, target)
                copied_screenshot = str(target.resolve())

        latency = result.get("latency") or {}
        model_ms = latency.get("model_ms")
        generation_pipeline_ms = latency.get("generation_pipeline_ms")
        files_ready_ms = latency.get("files_ready_with_png_ms")
        local_validation_ms = latency.get("local_validation_ms")
        png_render_ms = latency.get("png_render_ms")
        excluded_upstream_ms = latency.get("excluded_upstream_ms")
        screenshot_report = result.get("model_answer_screenshot") or validation.get(
            "model_answer_screenshot"
        ) or {}
        screenshot_render_ms = screenshot_report.get("render_ms")
        screenshot_ready_ms = screenshot_report.get("ready_from_workflow_start_ms")
        overhead_ms = None
        if isinstance(model_ms, (int, float)) and isinstance(
            files_ready_ms, (int, float)
        ):
            overhead_ms = round(float(files_ready_ms) - float(model_ms), 3)
        pre_png_local_ms = None
        if isinstance(model_ms, (int, float)) and isinstance(
            generation_pipeline_ms, (int, float)
        ):
            pre_png_local_ms = round(
                float(generation_pipeline_ms) - float(model_ms), 3
            )
        failure_reasons: list[str] = []
        for gate in ("schema", "relevance", "answer_guard"):
            report = validation.get(gate) or {}
            if report.get("valid") is False:
                failure_reasons.extend(str(error) for error in report.get("errors", []))
        if not result.get("latency_pass"):
            failure_reasons.append("含解析 PNG 的总时延超过 20 秒")
        failure_reasons = list(dict.fromkeys(failure_reasons))
        render_report = validation.get("render") or {}
        render_failure_reasons = (
            list(dict.fromkeys(str(error) for error in render_report.get("errors", [])))
            if render_report.get("valid") is False
            else []
        )
        row = {
            "id": item_id,
            "subject": result.get("subject"),
            "status": result.get("status"),
            "quality_gate_pass": bool(result.get("quality_gate_pass")),
            "schema_pass": bool(result.get("schema_pass")),
            "relevance_pass": bool(result.get("relevance_pass")),
            "answer_guard_pass": bool(result.get("answer_guard_pass")),
            "display_ready": bool(result.get("display_ready")),
            "render_pass": bool(render_report.get("valid")),
            "model_ms": model_ms,
            "pre_png_local_ms": pre_png_local_ms,
            "local_validation_ms": local_validation_ms,
            "png_render_ms": png_render_ms,
            "parsed_png_ready_ms": files_ready_ms,
            "local_and_png_overhead_ms": overhead_ms,
            "excluded_upstream_ms": excluded_upstream_ms,
            "model_answer_screenshot_render_ms": screenshot_render_ms,
            "model_answer_screenshot_ready_ms": screenshot_ready_ms,
            "latency_pass": bool(result.get("latency_pass")),
            "guidance_image_count": len(copied_guidance),
            "guidance_images": copied_guidance,
            "model_answer_screenshot": copied_screenshot,
            "source_question_image": copied_source_image,
            "failure_reasons": failure_reasons,
            "render_failure_reasons": render_failure_reasons,
            "run_dir": str(run_dir.resolve()) if run_dir and run_dir.is_dir() else None,
        }
        append_jsonl(manifest_path, row)
        rows.append(row)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "id",
                "subject",
                "status",
                "quality_gate_pass",
                "schema_pass",
                "relevance_pass",
                "answer_guard_pass",
                "display_ready",
                "render_pass",
                "model_ms",
                "pre_png_local_ms",
                "local_validation_ms",
                "png_render_ms",
                "parsed_png_ready_ms",
                "local_and_png_overhead_ms",
                "excluded_upstream_ms",
                "model_answer_screenshot_render_ms",
                "model_answer_screenshot_ready_ms",
                "latency_pass",
                "guidance_image_count",
                "source_question_image",
                "failure_reasons",
                "render_failure_reasons",
                "run_dir",
            ],
        )
        writer.writeheader()
        for row in rows:
            csv_row = {key: row.get(key) for key in writer.fieldnames}
            csv_row["failure_reasons"] = " | ".join(row["failure_reasons"])
            csv_row["render_failure_reasons"] = " | ".join(
                row["render_failure_reasons"]
            )
            writer.writerow(csv_row)

    model_values = [
        float(row["model_ms"])
        for row in rows
        if isinstance(row.get("model_ms"), (int, float))
    ]
    gate_measurement_values = [
        float(row["parsed_png_ready_ms"])
        for row in rows
        if isinstance(row.get("parsed_png_ready_ms"), (int, float))
    ]
    complete_image_rows = [row for row in rows if row["guidance_image_count"] == 5]
    complete_image_model_values = [
        float(row["model_ms"])
        for row in complete_image_rows
        if isinstance(row.get("model_ms"), (int, float))
    ]
    complete_image_ready_values = [
        float(row["parsed_png_ready_ms"])
        for row in complete_image_rows
        if isinstance(row.get("parsed_png_ready_ms"), (int, float))
    ]
    overhead_values = [
        float(row["local_and_png_overhead_ms"])
        for row in complete_image_rows
        if isinstance(row.get("local_and_png_overhead_ms"), (int, float))
    ]
    pre_png_local_values = [
        float(row["pre_png_local_ms"])
        for row in complete_image_rows
        if isinstance(row.get("pre_png_local_ms"), (int, float))
    ]
    local_validation_values = [
        float(row["local_validation_ms"])
        for row in complete_image_rows
        if isinstance(row.get("local_validation_ms"), (int, float))
    ]
    png_render_values = [
        float(row["png_render_ms"])
        for row in complete_image_rows
        if isinstance(row.get("png_render_ms"), (int, float))
    ]
    screenshot_render_values = [
        float(row["model_answer_screenshot_render_ms"])
        for row in rows
        if isinstance(row.get("model_answer_screenshot_render_ms"), (int, float))
    ]
    upstream_values = [
        float(row["excluded_upstream_ms"])
        for row in rows
        if isinstance(row.get("excluded_upstream_ms"), (int, float))
    ]
    over_budget_rows = [
        row
        for row in rows
        if isinstance(row.get("parsed_png_ready_ms"), (int, float))
        and float(row["parsed_png_ready_ms"]) > 20_000
    ]
    model_caused_over_budget = [
        row
        for row in over_budget_rows
        if isinstance(row.get("model_ms"), (int, float))
        and float(row["model_ms"]) > 20_000
    ]
    local_crossed_budget = [
        row for row in over_budget_rows if row not in model_caused_over_budget
    ]
    report = {
        "completed_at": utc_now(),
        "batch_dir": str(batch_dir.resolve()),
        "artifact_root": str(root.resolve()),
        "sample_count": len(rows),
        "guidance_complete_count": sum(
            row["guidance_image_count"] == 5 for row in rows
        ),
        "model_answer_screenshot_count": sum(
            bool(row["model_answer_screenshot"]) for row in rows
        ),
        "source_question_image_count": sum(
            bool(row["source_question_image"]) for row in rows
        ),
        "quality_pass_count": sum(row["quality_gate_pass"] for row in rows),
        "latency_pass_count": sum(row["latency_pass"] for row in rows),
        "complete_image_sets_within_20s_count": sum(
            row["guidance_image_count"] == 5 and row["latency_pass"] for row in rows
        ),
        "render_policy": {
            "hard_gate": "可解析、l1-l5 均有可展示内容、脱敏后无明确残留答案",
            "warning_only": "句数、字数、字段细节、关联性、脱敏置信度、20 秒时延",
        },
        "latency": {
            "all_samples": {
                "sample_count": len(rows),
                "model_answer": _distribution(model_values),
                "gate_measurement": _distribution(gate_measurement_values),
            },
            "complete_image_sets": {
                "sample_count": len(complete_image_rows),
                "model_answer": _distribution(complete_image_model_values),
                "parsed_images_ready": _distribution(complete_image_ready_values),
                "local_validation_redaction_and_png_overhead": _distribution(
                    overhead_values
                ),
                "local_before_png": _distribution(pre_png_local_values),
                "local_validation_only": _distribution(local_validation_values),
                "png_render_only": _distribution(png_render_values),
            },
            "diagnostic_model_answer_screenshot_render": _distribution(
                screenshot_render_values
            ),
            "excluded_upstream_input_preparation": _distribution(upstream_values),
            "over_20s_attribution": {
                "over_20s_count": len(over_budget_rows),
                "model_already_over_20s_count": len(model_caused_over_budget),
                "local_processing_crossed_20s_count": len(local_crossed_budget),
                "model_already_over_20s_ids": [row["id"] for row in model_caused_over_budget],
                "local_processing_crossed_20s_ids": [row["id"] for row in local_crossed_budget],
            },
        },
        "latency_comparison_csv": str(csv_path.resolve()),
        "comparison_manifest": str(manifest_path.resolve()),
    }
    feedback_storage_key = "test8_ans_feedback_v1:" + hashlib.sha256(
        "|".join(row["id"] for row in rows).encode("utf-8")
    ).hexdigest()[:20]
    report["feedback_ui"] = {
        "storage": "review_server_json_csv_with_browser_fallback",
        "storage_key": feedback_storage_key,
        "server_files": [
            "review_feedback.json",
            "review_feedback.csv",
            "review_feedback_summary.json",
        ],
        "statuses": ["unreviewed", "passed", "issue"],
        "issue_types": [
            "内容错误",
            "源码或公式显示",
            "最终答案泄漏",
            "层级或结构",
            "排版或截断",
            "原题或OCR",
            "其他",
        ],
    }
    write_json(root / "comparison_report.json", report)

    sections: list[str] = []
    for row in rows:
        raw_item_id = str(row["id"])
        item_id = html.escape(raw_item_id)
        item_id_attribute = html.escape(raw_item_id, quote=True)
        source_html = "<p>无原题图片</p>"
        if row["source_question_image"]:
            relative = Path(row["source_question_image"]).relative_to(root)
            source_html = (
                f'<a href="{html.escape(relative.as_posix())}" target="_blank">'
                f'<img loading="lazy" src="{html.escape(relative.as_posix())}" '
                f'alt="{item_id} 原题图片"></a>'
            )
        raw_html = "<p>无模型回答截图</p>"
        if row["model_answer_screenshot"]:
            relative = Path(row["model_answer_screenshot"]).relative_to(root)
            raw_html = (
                f'<img loading="lazy" src="{html.escape(relative.as_posix())}" '
                f'alt="{item_id} 模型原始回答">'
            )
        guidance_html = "".join(
            f'<img loading="lazy" src="{html.escape(Path(path).relative_to(root).as_posix())}" '
            f'alt="{item_id} 解析图">'
            for path in row["guidance_images"]
        ) or "<p>无解析图</p>"
        render_text = "已生成五张图" if row["render_pass"] else "未完整出图"
        gate_text = (
            f"生成状态：{render_text} ｜ 严格评测（只告警）："
            f"结构 {'通过' if row['schema_pass'] else '未通过'} ｜ "
            f"相关性 {'通过' if row['relevance_pass'] else '未通过'} ｜ "
            f"脱敏置信度 {'通过' if row['answer_guard_pass'] else '待复核'} ｜ "
            f"20秒 {'通过' if row['latency_pass'] else '超时'}"
        )
        reason_html = ""
        if row["failure_reasons"]:
            reason_html = (
                '<p class="failure"><strong>质量告警（不阻止出图）：</strong>'
                + html.escape("；".join(row["failure_reasons"]))
                + "</p>"
            )
        if row["render_failure_reasons"]:
            reason_html += (
                '<p class="render-failure"><strong>出图失败原因：</strong>'
                + html.escape("；".join(row["render_failure_reasons"]))
                + "</p>"
            )
        issue_options = "".join(
            '<label><input type="checkbox" class="feedback-issue" '
            f'value="{html.escape(issue, quote=True)}">{html.escape(issue)}</label>'
            for issue in report["feedback_ui"]["issue_types"]
        )
        sections.append(
            f'<section data-item-id="{item_id_attribute}" data-review-status="unreviewed">'
            f"<h2>{item_id}</h2>"
            '<div class="latency">'
            f"<span>模型生成<strong>{row['model_ms']} ms</strong></span>"
            f"<span>本地前处理<strong>{row['pre_png_local_ms']} ms</strong></span>"
            f"<span>其中校验<strong>{row['local_validation_ms']} ms</strong></span>"
            f"<span>五层 PNG<strong>{row['png_render_ms']} ms</strong></span>"
            f"<span>产品就绪<strong>{row['parsed_png_ready_ms']} ms</strong></span>"
            "</div>"
            f"<p>{gate_text}</p>{reason_html}"
            '<div class="comparison"><div><h3>原题图片</h3>'
            f"{source_html}</div><div><h3>模型原始回答（内部审核）</h3>"
            f"{raw_html}</div><div><h3>脱敏五层解析图</h3>"
            f'<div class="tiers">{guidance_html}</div></div></div>'
            '<details class="feedback-panel"><summary>审核反馈</summary>'
            '<div class="feedback-fields"><label>审核结论<select class="feedback-status">'
            '<option value="unreviewed">未审核</option><option value="passed">通过</option>'
            '<option value="issue">有问题</option></select></label>'
            '<label>严重程度<select class="feedback-severity">'
            '<option value="">未选择</option><option value="low">轻微</option>'
            '<option value="medium">一般</option><option value="high">严重</option>'
            '</select></label></div>'
            f'<div class="issue-options">{issue_options}</div>'
            '<label class="feedback-note-label">问题说明<textarea class="feedback-note" '
            'placeholder="例如：第三层公式仍显示反斜杠；第二句泄漏最终选项……"></textarea></label>'
            '<div class="feedback-actions"><button type="button" class="save-feedback">保存本题反馈</button>'
            '<span class="feedback-message" aria-live="polite"></span></div>'
            '</details></section>'
        )
    feedback_storage_json = json.dumps(feedback_storage_key, ensure_ascii=False)
    document = (
        "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
        "<title>50题解析对比</title><style>"
        "body{font-family:'Microsoft YaHei',sans-serif;margin:24px;background:#f3f6f8;color:#111}"
        "section{background:white;border-radius:18px;padding:20px;margin:0 0 24px}"
        ".comparison{display:grid;grid-template-columns:minmax(260px,.8fr) minmax(320px,1fr) minmax(320px,1fr);gap:20px;align-items:start}"
        ".comparison img{width:100%;height:auto;border-radius:10px;background:#f3f6f8}"
        ".tiers{display:grid;gap:12px}h2,h3{margin-top:0}"
        ".latency{display:flex;flex-wrap:wrap;gap:8px 18px;margin:8px 0}.latency span{background:#f3f6f8;border-radius:8px;padding:8px 10px}.latency strong{display:block;margin-top:3px}"
        ".failure{padding:10px 12px;border-radius:8px;background:#fff8df;color:#111}.render-failure{padding:10px 12px;border-radius:8px;background:#fff3f3;color:#111}"
        ".review-toolbar{position:sticky;top:0;z-index:5;background:#fff;border-radius:14px;padding:14px 16px;margin:0 0 20px;box-shadow:0 4px 16px #00000012}.review-toolbar-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.review-summary{font-weight:700;margin-right:auto}.review-toolbar button,.import-label,.save-feedback{border:0;border-radius:8px;background:#111;color:#fff;padding:9px 13px;cursor:pointer;font:inherit}.review-toolbar button.secondary,.import-label{background:#e8edf1;color:#111}.import-label input{display:none}"
        ".feedback-panel{margin-top:18px;border-top:1px solid #dfe5e9;padding-top:14px}.feedback-panel summary{font-weight:700;cursor:pointer}.feedback-fields{display:flex;flex-wrap:wrap;gap:14px;margin:14px 0}.feedback-fields label{display:grid;gap:5px}.feedback-fields select{min-width:150px;padding:8px;border:1px solid #bbc5cc;border-radius:7px;background:#fff}.issue-options{display:flex;flex-wrap:wrap;gap:8px 16px;margin:8px 0 14px}.issue-options label{display:flex;gap:5px;align-items:center}.feedback-note-label{display:grid;gap:6px}.feedback-note{box-sizing:border-box;width:100%;min-height:88px;padding:10px;border:1px solid #bbc5cc;border-radius:8px;font:inherit;resize:vertical}.feedback-actions{display:flex;gap:12px;align-items:center;margin-top:10px}.feedback-message{color:#246b35}.review-passed{outline:3px solid #cfe9d5}.review-issue{outline:3px solid #f1b9b9}"
        "@media(max-width:1200px){.comparison{grid-template-columns:1fr}}"
        "</style><body><h1>50题：原题、模型回答与脱敏解析图对比</h1>"
        '<p>出图与严格评测已分开：只有无法解析、缺失整层内容或脱敏后仍有明确答案嫌疑才会阻止出图；其他项只供审核。</p>'
        '<div class="review-toolbar"><div class="review-toolbar-row">'
        '<div id="review-summary" class="review-summary">审核进度：0/0</div>'
        '<button type="button" class="secondary" data-filter="all">全部</button>'
        '<button type="button" class="secondary" data-filter="unreviewed">只看未审核</button>'
        '<button type="button" class="secondary" data-filter="issue">只看有问题</button>'
        '<button type="button" id="export-feedback-json">导出反馈 JSON</button>'
        '<button type="button" id="export-feedback-csv">导出反馈 CSV</button>'
        '<label class="import-label">导入反馈 JSON<input id="import-feedback-json" type="file" accept="application/json,.json"></label>'
        '</div><div id="storage-warning"></div></div>'
        + "".join(sections)
        + "<script>"
        f"const STORAGE_KEY={feedback_storage_json};"
        "const sections=[...document.querySelectorAll('section[data-item-id]')];"
        "const SERVER_MODE=location.protocol==='http:'||location.protocol==='https:';"
        "let feedback={};let storageAvailable=true;"
        "try{feedback=JSON.parse(localStorage.getItem(STORAGE_KEY)||'{}')||{}}catch(error){storageAvailable=false;document.getElementById('storage-warning').textContent='当前浏览器禁止本地持久化，请审核后及时导出 JSON。'}"
        "function readSection(section){const id=section.dataset.itemId;return{id,status:section.querySelector('.feedback-status').value,severity:section.querySelector('.feedback-severity').value,issues:[...section.querySelectorAll('.feedback-issue:checked')].map(x=>x.value),note:section.querySelector('.feedback-note').value.trim(),updated_at:new Date().toISOString()}}"
        "function applySection(section){const item=feedback[section.dataset.itemId]||{};section.querySelector('.feedback-status').value=item.status||'unreviewed';section.querySelector('.feedback-severity').value=item.severity||'';section.querySelector('.feedback-note').value=item.note||'';section.querySelectorAll('.feedback-issue').forEach(x=>x.checked=(item.issues||[]).includes(x.value));section.dataset.reviewStatus=item.status||'unreviewed';section.classList.toggle('review-passed',item.status==='passed');section.classList.toggle('review-issue',item.status==='issue')}"
        "function feedbackDocument(){return{version:'test8_ans_review_feedback_v2',storage_key:STORAGE_KEY,exported_at:new Date().toISOString(),items:allItems()}}"
        "async function persist(){if(storageAvailable){try{localStorage.setItem(STORAGE_KEY,JSON.stringify(feedback))}catch(error){storageAvailable=false}}updateSummary();if(SERVER_MODE){try{const response=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(feedbackDocument())});if(!response.ok)throw new Error(await response.text());document.getElementById('storage-warning').textContent='反馈已同步保存到本批次工程文件。';return true}catch(error){document.getElementById('storage-warning').textContent='工程文件保存失败，请立即导出 JSON：'+error.message;return false}}if(!storageAvailable)document.getElementById('storage-warning').textContent='当前为静态页面，反馈未写入工程；请用 review 命令打开或导出 JSON。';return storageAvailable}"
        "function updateSummary(){const items=sections.map(s=>feedback[s.dataset.itemId]||{status:'unreviewed'});const passed=items.filter(x=>x.status==='passed').length;const issues=items.filter(x=>x.status==='issue').length;const reviewed=passed+issues;document.getElementById('review-summary').textContent=`审核进度：${reviewed}/${items.length}｜通过 ${passed}｜有问题 ${issues}｜未审核 ${items.length-reviewed}`;}"
        "sections.forEach(section=>{applySection(section);section.querySelector('.save-feedback').addEventListener('click',async()=>{const item=readSection(section);feedback[item.id]=item;applySection(section);const saved=await persist();const message=section.querySelector('.feedback-message');message.textContent=saved?(SERVER_MODE?'已保存到工程文件':'已保存到浏览器本地'):'保存失败，请导出 JSON';setTimeout(()=>message.textContent='',2500)})});"
        "document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{const filter=button.dataset.filter;sections.forEach(section=>{const status=(feedback[section.dataset.itemId]||{}).status||'unreviewed';section.hidden=filter!=='all'&&status!==filter})}));"
        "function allItems(){return sections.map(section=>feedback[section.dataset.itemId]||{id:section.dataset.itemId,status:'unreviewed',severity:'',issues:[],note:'',updated_at:null})}"
        "function download(name,type,text){const url=URL.createObjectURL(new Blob([text],{type}));const link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}"
        "document.getElementById('export-feedback-json').addEventListener('click',()=>download('test8-ans-feedback.json','application/json',JSON.stringify(feedbackDocument(),null,2)));"
        "function csvCell(value){const text=Array.isArray(value)?value.join('|'):String(value??'');return '\"'+text.replaceAll('\"','\"\"')+'\"'}"
        "document.getElementById('export-feedback-csv').addEventListener('click',()=>{const keys=['id','status','severity','issues','note','updated_at'];const lines=[keys.map(csvCell).join(','),...allItems().map(item=>keys.map(key=>csvCell(item[key])).join(','))];download('test8-ans-feedback.csv','text/csv;charset=utf-8','\\uFEFF'+lines.join('\\r\\n'))});"
        "document.getElementById('import-feedback-json').addEventListener('change',async event=>{const file=event.target.files[0];if(!file)return;try{const data=JSON.parse(await file.text());if(!Array.isArray(data.items))throw new Error('items missing');feedback={};data.items.forEach(item=>{if(item&&item.id)feedback[item.id]=item});sections.forEach(applySection);await persist();alert('反馈已导入')}catch(error){alert('导入失败：'+error.message)}event.target.value='' });"
        "async function loadServerFeedback(){if(!SERVER_MODE){updateSummary();return}try{const response=await fetch('/api/feedback',{cache:'no-store'});if(!response.ok)throw new Error(await response.text());const data=await response.json();if(Array.isArray(data.items)){feedback={};data.items.forEach(item=>{if(item&&item.id)feedback[item.id]=item});sections.forEach(applySection)}document.getElementById('storage-warning').textContent='当前使用工程文件存储，每次保存都会生成 JSON、CSV 和汇总。';updateSummary()}catch(error){document.getElementById('storage-warning').textContent='读取工程反馈失败：'+error.message;updateSummary()}}"
        "loadServerFeedback();"
        "</script></body></html>"
    )
    (root / "overview.html").write_text(document, encoding="utf-8")
    report["overview_html"] = str((root / "overview.html").resolve())
    write_json(root / "comparison_report.json", report)
    return report
