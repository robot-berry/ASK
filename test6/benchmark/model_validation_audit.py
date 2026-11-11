#!/usr/bin/env python3
"""Serially exercise the public Test6 web API and build a human audit report."""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def rule_family(message: str) -> str:
    """Map path-specific diagnostics to a stable rule for human review."""

    text = str(message)
    if text.startswith("课程边界："):
        return "课程术语边界"
    if "source_figures为空" in text:
        return "题目引用图表但未提取图块"
    if "三个档位均必须覆盖原题全部小问" in text:
        return "三个档位覆盖全部小问"
    if "中文说明性下标必须写成" in text:
        return "中文说明性下标格式"
    if "下标必须写成_{...}" in text:
        return "LaTeX 下标花括号格式"
    if "公式花括号不成对" in text:
        return "LaTeX 花括号平衡"
    if "直接给出了" in text or "最终结果" in text or "答案泄漏" in text:
        return "最终答案泄漏"
    if text.startswith("任务失败：") or text.startswith("测试运行失败："):
        return "任务执行失败"
    if "模型传输未成功" in text:
        return "模型传输失败"
    if "本地兜底" in text or "模型五层数量=" in text or "响应解析异常" in text:
        return "模型结果不完整或兜底"
    return text.split("：", 1)[0]


def rule_recommendation(family: str) -> str:
    recommendations = {
        "课程术语边界": "保留为报告项，不拦截；改成结合题干语境判断，并允许人工维护术语白名单。",
        "题目引用图表但未提取图块": "保留。这是原题语义完整性校验；只提示 OCR/裁图复核，不应否定模型解析。",
        "三个档位覆盖全部小问": "保留为质量告警；先准确识别小问数量，再核对每档覆盖，避免按句式误判。",
        "中文说明性下标格式": "不应拦截。渲染前本地规范化为 _{\\text{...}}，无法规范化时仅报告。",
        "LaTeX 下标花括号格式": "不应拦截。优先本地自动补花括号，保留原文与修正记录供抽检。",
        "LaTeX 花括号平衡": "保留，但必须忽略 \\{、\\} 这类字面花括号；仅真实不可渲染时做局部修正。",
        "最终答案泄漏": "保留为高优先级校验；只删除命中的句子，不丢弃其余五层分析，也不追加模型修复。",
        "任务执行失败": "保留为系统故障项，需要重试并记录具体阶段。",
        "模型传输失败": "保留为系统故障项；允许本地兜底，但必须明确标记未调用成功。",
        "模型结果不完整或兜底": "保留为系统故障项；优先缩短提示词或调大复杂题输出预算，避免把兜底伪装成模型结果。",
    }
    return recommendations.get(
        family,
        "先作为报告项保留，不拦截；结合原题与页面结果人工判断后再决定删除或细化规则。",
    )


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def request_json(request: Request, timeout: float = 120.0) -> dict[str, Any]:
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def submit(portal: str, subject: str, source: Path) -> dict[str, Any]:
    media_type = mimetypes.guess_type(source.name)[0] or "image/png"
    request = Request(
        f"{portal.rstrip('/')}/api/v1/{subject}/jobs/raw",
        data=source.read_bytes(),
        headers={"Content-Type": media_type},
        method="POST",
    )
    return request_json(request)


def poll(portal: str, poll_url: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    absolute = f"{portal.rstrip('/')}{poll_url}"
    while True:
        state = request_json(Request(absolute, method="GET"))
        if state.get("status") in {"completed", "failed"}:
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError(f"任务轮询超过{timeout_seconds:.0f}秒")
        time.sleep(max(0.5, min(3.0, float(state.get("retry_after_seconds") or 1.0))))


def review_suggestion(messages: list[str]) -> str:
    joined = "\n".join(messages)
    suggestions: list[str] = []
    if "课程" in joined or "学段" in joined or "双曲线" in joined or "导数" in joined:
        suggestions.append("人工确认术语是否属于本题语境；若只是教材内表述，应改为告警而非拦截")
    if "所求结论" in joined or "答案" in joined or "泄漏" in joined or "最终结果" in joined:
        suggestions.append("核对页面是否直接给出最终答案；必要时仅删除泄漏句，保留其余模型分析")
    if "fallback" in joined or "兜底" in joined or "解析" in joined:
        suggestions.append("检查模型JSON完整性与超时设置，确认是否真的需要整份兜底")
    if "数量" in joined or "step" in joined or "句数" in joined:
        suggestions.append("检查句数限制是否影响自然表达；可改为提示性告警")
    if not suggestions:
        suggestions.append("对照原题与五层结果，判断该提示是否影响正确性或可读性")
    return "；".join(dict.fromkeys(suggestions))


def audit_record(source: Path, public: dict[str, Any], internal: dict[str, Any]) -> dict[str, Any]:
    provider = dict(internal.get("provider_contract") or {})
    validation = dict(internal.get("validation") or {})
    preprocess = dict(internal.get("preprocess") or {})
    violations = [str(item) for item in (provider.get("curriculum_violations") or [])]
    errors = [str(item) for item in (validation.get("errors") or [])]
    warnings = [str(item) for item in (validation.get("warnings") or [])]
    checks: list[str] = []
    error_payload = internal.get("error")
    if isinstance(error_payload, dict):
        task_error = str(error_payload.get("message") or error_payload.get("type") or "")
    else:
        task_error = str(error_payload or "")
    if task_error:
        checks.append("任务失败：" + task_error)
    if (
        internal.get("http_status") != 200
        and not (
            int(provider.get("model_tier_count") or 0) == 5
            and provider.get("model_content_preserved") is True
        )
    ):
        checks.append("模型传输未成功")
    if provider.get("time_budget_fallback") is True:
        checks.append("使用了整份本地兜底")
    if int(provider.get("model_tier_count") or 0) != 5:
        checks.append(f"模型五层数量={int(provider.get('model_tier_count') or 0)}")
    if provider.get("provider_parse_error"):
        checks.append("模型响应解析异常：" + str(provider["provider_parse_error"]))
    checks.extend(f"课程边界：{item}" for item in violations)
    checks.extend(f"严格校验错误：{item}" for item in errors)
    checks.extend(f"严格校验告警：{item}" for item in warnings)
    model_call_success = bool(
        (
            internal.get("provider_request_id")
            and internal.get("http_status") == 200
            and internal.get("model_id_response")
        )
        or (
            int(provider.get("model_tier_count") or 0) == 5
            and provider.get("model_content_preserved") is True
        )
    )
    model_output_used = bool(
        model_call_success
        and int(provider.get("model_tier_count") or 0) == 5
        and provider.get("time_budget_fallback") is not True
    )
    timing = dict(public.get("timing") or {})
    return {
        "source": source.name,
        "source_path": str(source.resolve()),
        "job_id": str(public.get("job_id") or internal.get("job_id") or ""),
        "status": str(public.get("status") or internal.get("status") or ""),
        "model_requested": str(internal.get("model_id_requested") or ""),
        "model_response": str(internal.get("model_id_response") or ""),
        "provider_request_id": str(internal.get("provider_request_id") or ""),
        "http_status": internal.get("http_status"),
        "total_tokens": (internal.get("usage") or {}).get("total_tokens"),
        "model_call_success": model_call_success,
        "model_output_used": model_output_used,
        "model_tier_count": int(provider.get("model_tier_count") or 0),
        "model_content_preserved": bool(provider.get("model_content_preserved")),
        "time_budget_fallback": bool(provider.get("time_budget_fallback")),
        "curriculum_validation_mode": str(provider.get("curriculum_validation_mode") or ""),
        "curriculum_passed": bool(provider.get("curriculum_guard_passed", True)),
        "curriculum_violations": violations,
        "validation_valid": bool(validation.get("valid")),
        "strict_valid": bool(validation.get("strict_valid")),
        "validation_errors": errors,
        "validation_warnings": warnings,
        "repair_audit": list(validation.get("repair_audit") or []),
        "checks_for_human_review": checks,
        "needs_human_review": bool(checks),
        "suggestion": review_suggestion(checks),
        "ocr_text": str(preprocess.get("ocr_text") or ""),
        "elapsed_seconds": public.get("elapsed_seconds"),
        "first_image_seconds": timing.get("first_image_from_created_seconds"),
    }


def copy_evidence(record: dict[str, Any], runs_root: Path, pages: Path) -> list[dict[str, str]]:
    job_dir = runs_root / record["job_id"]
    copied: list[dict[str, str]] = []
    candidates = [(job_dir / "audit" / "original.upload", "原始题图")]
    for name, label in (
        ("01_question_knowledge.png", "原题与知识点"),
        ("02_guidance_elite.png", "学霸解析"),
        ("03_guidance_upper.png", "中上解析"),
        ("04_guidance_middle.png", "中等解析"),
        ("05_guidance_lower.png", "中下解析"),
        ("06_guidance_novice.png", "学渣解析"),
    ):
        candidates.append((job_dir / name, label))
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", Path(record["source"]).stem)
    for index, (source, label) in enumerate(candidates):
        if not source.is_file():
            continue
        target = pages / f"{safe}_{index:02d}.png"
        shutil.copy2(source, target)
        copied.append({"label": label, "path": f"pages/{target.name}"})
    return copied


def write_reports(output: Path, records: list[dict[str, Any]]) -> None:
    (output / "data.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "source", "job_id", "status", "model_requested", "model_response",
        "provider_request_id", "http_status", "total_tokens", "model_call_success",
        "model_output_used", "model_tier_count", "time_budget_fallback",
        "curriculum_validation_mode", "curriculum_passed", "validation_valid",
        "strict_valid", "needs_human_review", "elapsed_seconds", "first_image_seconds",
        "curriculum_violations", "validation_errors", "validation_warnings", "suggestion",
    ]
    with (output / "data.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            row = {key: item.get(key) for key in fields}
            for key in ("curriculum_violations", "validation_errors", "validation_warnings"):
                row[key] = " | ".join(item.get(key) or [])
            writer.writerow(row)

    total = len(records)
    call_success = sum(bool(item.get("model_call_success")) for item in records)
    model_used = sum(bool(item.get("model_output_used")) for item in records)
    review = sum(bool(item.get("needs_human_review")) for item in records)
    rule_counts: Counter[str] = Counter()
    rule_sources: dict[str, list[str]] = {}
    rule_examples: dict[str, list[str]] = {}
    for item in records:
        seen_families: set[str] = set()
        for message in item.get("checks_for_human_review") or []:
            family = rule_family(message)
            rule_examples.setdefault(family, [])
            if message not in rule_examples[family]:
                rule_examples[family].append(message)
            if family not in seen_families:
                rule_counts[family] += 1
                rule_sources.setdefault(family, []).append(str(item.get("source") or ""))
                seen_families.add(family)
    elapsed_values = [
        float(item["elapsed_seconds"])
        for item in records
        if isinstance(item.get("elapsed_seconds"), (int, float))
    ]
    first_values = [
        float(item["first_image_seconds"])
        for item in records
        if isinstance(item.get("first_image_seconds"), (int, float))
    ]
    average_elapsed = round(sum(elapsed_values) / len(elapsed_values), 3) if elapsed_values else None
    average_first = round(sum(first_values) / len(first_values), 3) if first_values else None
    lines = [
        "# Test6 模型调用与校验人工核验报告",
        "",
        f"- 串行测试题数：{total}",
        f"- DashScope 调用成功：{call_success}/{total}",
        f"- 五层模型结果被实际采用：{model_used}/{total}",
        f"- 需人工核验：{review}/{total}",
        f"- 平均总时延：{average_elapsed} 秒；平均首图时延：{average_first} 秒",
        "- 校验策略：所有内容校验只记录、不拦截，也不触发额外模型修复；最终答案泄漏和残缺草稿仅做局部删除，保留其余模型自然语言。",
        "",
        "## 规则级人工核验清单",
        "",
    ]
    if rule_counts:
        lines.extend([
            "| 校验规则 | 命中题数 | 命中题目 | 当前初步建议 |",
            "|---|---:|---|---|",
        ])
        for family, count in rule_counts.most_common():
            sources = "、".join(rule_sources.get(family) or [])
            recommendation = rule_recommendation(family)
            lines.append(f"| {family} | {count} | {sources} | {recommendation} |")
        lines.extend(["", "### 规则原始命中信息", ""])
        for family, _count in rule_counts.most_common():
            lines.append(f"- **{family}**")
            lines.extend(f"  - {message}" for message in rule_examples.get(family, []))
    else:
        lines.append("- 本批次没有触发人工核验项。")
    lines.extend(["", "## 逐题记录", ""])
    for item in records:
        checks = item.get("checks_for_human_review") or []
        lines.extend([
            f"### {item['source']} — {'需人工核验' if checks else '通过'}",
            "",
            f"- 任务：`{item['job_id']}`",
            f"- 模型：`{item['model_response'] or item['model_requested']}`；HTTP {item['http_status']}；五层采用：{'是' if item['model_output_used'] else '否'}",
            f"- 调用凭证：`{item['provider_request_id'] or '无'}`",
            f"- 时延：{item.get('elapsed_seconds')} 秒；首图：{item.get('first_image_seconds')} 秒",
            f"- 建议：{item['suggestion']}",
            "",
        ])
        if checks:
            lines.extend(f"  - {message}" for message in checks)
            lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    cards: list[str] = []
    for item in records:
        checks = item.get("checks_for_human_review") or []
        evidence = "".join(
            f'<figure><img loading="lazy" src="{html.escape(asset["path"])}"><figcaption>{html.escape(asset["label"])}</figcaption></figure>'
            for asset in item.get("evidence") or []
        )
        check_html = "".join(f"<li>{html.escape(message)}</li>" for message in checks) or "<li>未触发</li>"
        cards.append(
            f'<article class="card {"review" if checks else "pass"}">'
            f'<h2>{html.escape(item["source"])} <span>{"需人工核验" if checks else "通过"}</span></h2>'
            f'<p>模型：{html.escape(item["model_response"] or item["model_requested"])} · HTTP {item["http_status"]} · 五层采用：{"是" if item["model_output_used"] else "否"} · 总时延：{item.get("elapsed_seconds")}秒</p>'
            f'<p class="id">任务 {html.escape(item["job_id"])} · 请求ID {html.escape(item["provider_request_id"] or "无")}</p>'
            f'<h3>触发的校验</h3><ul>{check_html}</ul>'
            f'<p><b>人工核验建议：</b>{html.escape(item["suggestion"])}</p>'
            f'<details><summary>查看OCR文字</summary><pre>{html.escape(item.get("ocr_text") or "")}</pre></details>'
            f'<details><summary>查看原题和六张结果图</summary><div class="gallery">{evidence}</div></details>'
            "</article>"
        )
    counts = "；".join(f"{key}{value}题" for key, value in rule_counts.most_common()) or "无"
    rule_rows = "".join(
        "<tr>"
        f"<td>{html.escape(family)}</td>"
        f"<td>{count}</td>"
        f"<td>{html.escape('、'.join(rule_sources.get(family) or []))}</td>"
        f"<td>{html.escape(rule_recommendation(family))}</td>"
        "</tr>"
        for family, count in rule_counts.most_common()
    ) or '<tr><td colspan="4">本批次没有触发人工核验项</td></tr>'
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Test6 模型校验人工核验</title><style>
body{{font-family:"Microsoft YaHei",sans-serif;background:#f3f6fa;color:#172033;margin:0}}main{{max-width:1280px;margin:auto;padding:24px}}.summary,.rules,.card{{background:white;border:1px solid #dbe3ee;border-radius:14px;padding:18px;margin-bottom:16px}}.summary strong{{color:#075fb4}}.card.review{{border-left:6px solid #d97706}}.card.pass{{border-left:6px solid #16a34a}}h2 span{{font-size:14px;padding:4px 10px;border-radius:999px;background:#eef2f7}}.id{{color:#526071;font-size:13px}}pre{{white-space:pre-wrap;background:#f8fafc;padding:12px;border-radius:8px}}details{{margin-top:12px}}summary{{cursor:pointer;color:#075fb4}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #dbe3ee;padding:10px;text-align:left;vertical-align:top}}th{{background:#f8fafc}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:12px}}figure{{margin:0;border:1px solid #dde4ed;border-radius:10px;padding:8px}}img{{width:100%;height:auto;display:block}}figcaption{{text-align:center;padding-top:6px}}</style></head><body><main>
<section class="summary"><h1>Test6 模型调用与校验人工核验</h1><p>串行测试 <strong>{total}</strong> 题；DashScope成功 <strong>{call_success}</strong> 题；模型五层结果实际采用 <strong>{model_used}</strong> 题；需人工核验 <strong>{review}</strong> 题。</p><p>平均总时延 <strong>{average_elapsed}</strong> 秒；平均首图时延 <strong>{average_first}</strong> 秒。</p><p>规则触发：{html.escape(counts)}</p><p>所有内容校验均“只校验、不拦截”，也不触发额外模型修复；答案泄漏与残缺草稿仅局部删除。点击每题可展开OCR文字及全部结果图。</p></section>
<section class="rules"><h2>规则级人工核验清单</h2><table><thead><tr><th>校验规则</th><th>命中题数</th><th>命中题目</th><th>当前初步建议</th></tr></thead><tbody>{rule_rows}</tbody></table></section>
{''.join(cards)}</main></body></html>"""
    (output / "AUDIT_INDEX.html").write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--portal", default="http://127.0.0.1:8230")
    parser.add_argument("--subject", choices=("math", "physics"), default="math")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--merge-existing", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    existing_data = args.output_dir / "data.json"
    if args.report_only:
        if not existing_data.is_file():
            raise SystemExit("没有可重建报告的 data.json")
        records = json.loads(existing_data.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise SystemExit("data.json 中没有测试记录")
        write_reports(args.output_dir, records)
        print(f"已基于 {len(records)} 条现有记录重建人工核验报告", flush=True)
        return 0
    all_sources = sorted(
        (path for path in args.input_dir.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES),
        key=natural_key,
    )
    sources = list(all_sources)
    if args.only:
        selected = {str(name).lower() for name in args.only}
        sources = [path for path in sources if path.name.lower() in selected]
    if args.limit > 0:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit("没有找到测试图片")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages = args.output_dir / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    record_map: dict[str, dict[str, Any]] = {}
    if args.merge_existing and existing_data.is_file():
        for item in json.loads(existing_data.read_text(encoding="utf-8")):
            if isinstance(item, dict) and item.get("source"):
                record_map[str(item["source"])] = item
    jsonl = args.output_dir / "records.jsonl"
    for index, source in enumerate(sources, 1):
        started = time.monotonic()
        try:
            queued = submit(args.portal, args.subject, source)
            public = poll(args.portal, str(queued["poll_url"]), args.timeout)
            job_id = str(public.get("job_id") or queued["job_id"])
            internal_path = args.runs_root / job_id / "job.json"
            internal = json.loads(internal_path.read_text(encoding="utf-8"))
            record = audit_record(source, public, internal)
            record["evidence"] = copy_evidence(record, args.runs_root, pages)
        except (OSError, ValueError, KeyError, TimeoutError, HTTPError, URLError) as exc:
            record = {
                "source": source.name,
                "source_path": str(source.resolve()),
                "job_id": "",
                "status": "runner_failed",
                "model_requested": "",
                "model_response": "",
                "provider_request_id": "",
                "http_status": None,
                "total_tokens": None,
                "model_call_success": False,
                "model_output_used": False,
                "model_tier_count": 0,
                "model_content_preserved": False,
                "time_budget_fallback": False,
                "curriculum_validation_mode": "",
                "curriculum_passed": False,
                "curriculum_violations": [],
                "validation_valid": False,
                "strict_valid": False,
                "validation_errors": [str(exc)],
                "validation_warnings": [],
                "repair_audit": [],
                "checks_for_human_review": ["测试运行失败：" + str(exc)],
                "needs_human_review": True,
                "suggestion": "检查网页服务、测试图片格式和任务日志",
                "ocr_text": "",
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "first_image_seconds": None,
                "evidence": [],
            }
        record_map[source.name] = record
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        records = [record_map[path.name] for path in all_sources if path.name in record_map]
        write_reports(args.output_dir, records)
        print(
            f"[{index}/{len(sources)}] {source.name}: {record['status']}; "
            f"model={record['model_call_success']}; used={record['model_output_used']}; "
            f"review={record['needs_human_review']}; elapsed={record['elapsed_seconds']}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
