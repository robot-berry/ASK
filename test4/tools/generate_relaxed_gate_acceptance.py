#!/usr/bin/env python3
"""Generate math/physics four-image QA output without provider calls."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from PIL import Image


TEST4_ROOT = Path(__file__).resolve().parents[1]
IMAGE_NAMES = (
    "01_question_knowledge.png",
    "02_guidance_advanced.png",
    "03_guidance_standard.png",
    "04_guidance_foundation.png",
)


def _part(steps: list[str], *, pitfall: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "part": "整题",
        "method_id": "M1",
        "method_name": "关系分析法",
        "steps": steps,
        "alternative_methods": [],
    }
    if pitfall is not None:
        value["pitfall"] = pitfall
    return value


def fixture(subject: str) -> dict[str, Any]:
    if subject == "math":
        question = "已知函数 $f(x)$ 及其导函数 $f'(x)$，证明给定区间上的单调性结论。"
        knowledge = ["函数的单调性", "导数符号与单调性的关系"]
        formula = {"formula": "f'(x)>0", "condition": "先作为待核对的中间条件"}
        advanced = [
            "先确定定义域并写出导函数。",
            "核对导数符号与目标区间，允许在证明末尾写“得证”。",
        ]
        standard = [
            "划分需要讨论的区间。",
            "逐段判断导数符号，不直接给出数值型终答。",
            "若路径较长，可以重新分析区间端点；完成推理后写“得证”。",
        ]
        foundation = [
            "先圈出题目给出的定义域。",
            "求导并保留尚未化简的关系。",
            "再检查每个区间内导数的正负。",
            "把符号判断对应到单调性，证明完成时可以写“得证”。",
        ]
        summary = "先确定范围，再建立导数关系并逐段核对；证明收尾措辞不是答案泄露。"
        pitfall = "中间关系可以保留，但不要直接代出题目要求的最终数值。"
    else:
        question = "一质量为$m$的物体受恒力$F$作用，请证明其运动满足给定动力学关系。"
        knowledge = ["受力分析", "牛顿第二定律"]
        formula = {"formula": "F=ma", "condition": "物体所受合力为F"}
        advanced = [
            "先画受力草图并选定正方向。",
            "列出动力学中间关系，证明完成后可以写“得证”。",
        ]
        standard = [
            "明确研究对象和运动方向。",
            "写出合力与加速度之间的关系。",
            "再检查单位和正负号；确认推理闭合后写“得证”。",
        ]
        foundation = [
            "圈出质量和外力两个已知量。",
            "画出物体的受力方向。",
            "选择正方向并列出未求值的动力学式。",
            "逐项检查量纲和符号，证明末尾允许写“得证”。",
        ]
        summary = "先选对象和方向，再列动力学关系并核对量纲；普通检查语句可以保留。"
        pitfall = "不要把力、质量和加速度的单位混用。"

    return {
        "subject": "mathematics" if subject == "math" else "physics",
        "problem_scope": "primary_to_high_school",
        "recognition_status": "clear",
        "uncertain_items": [],
        "question_blocks": [{"type": "text", "text": question, "figure_id": ""}],
        "source_figures": [],
        "visual_evidence": [],
        "knowledge_points": knowledge,
        "base_formulas": [formula],
        "guidance": {
            "advanced": {"title": "学霸版", "parts": [_part(advanced)]},
            "standard": {
                "title": "普通版",
                "parts": [_part(standard, pitfall=pitfall)],
            },
            "foundation": {
                "title": "基础版",
                "parts": [_part(foundation)],
                "pattern_summary": summary,
            },
        },
    }


def run_worker(subject: str, output_root: Path) -> None:
    source_root = TEST4_ROOT / ("math" if subject == "math" else "physics") / "source"
    sys.path.insert(0, str(source_root))
    from search_machine_pipeline import parse_model_output  # type: ignore
    from device_result_renderer import validate_device_image_set  # type: ignore
    if subject == "math":
        from math_guidance_renderer import render_four_images  # type: ignore
    else:
        from physics_guidance_renderer import render_four_images  # type: ignore

    raw = fixture(subject)
    parsed, errors = parse_model_output(json.dumps(raw, ensure_ascii=False))
    if parsed is None or errors:
        raise RuntimeError(f"{subject} fixture validation failed: {errors}")

    subject_dir = output_root / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    source_image = Image.new("L", (1200, 800), 255)
    render_four_images(
        question_id=f"relaxed_gate_{subject}",
        model_id="offline-no-provider",
        source_image=source_image,
        output=parsed,
        output_dir=subject_dir,
    )
    contract = validate_device_image_set(subject_dir)

    direct = fixture(subject)
    if subject == "math":
        direct["question_blocks"][0]["text"] = "求参数 $k$ 的值。"
        direct_answer = "最终得到 $k=2$。"
    else:
        direct["question_blocks"][0]["text"] = (
            "一质量为$m$的物体受恒力$F$作用，求加速度$a$。"
        )
        direct_answer = "最终得到 $a=F/m$。"
    for tier in ("advanced", "standard", "foundation"):
        direct["guidance"][tier]["parts"][0]["steps"][-1] = direct_answer
    _, direct_errors = parse_model_output(json.dumps(direct, ensure_ascii=False))

    report = {
        "subject": subject,
        "provider_calls": 0,
        "allowed_fixture_validation_errors": errors,
        "direct_answer_negative_control": direct_answer,
        "direct_answer_blocked": bool(direct_errors),
        "direct_answer_errors": direct_errors,
        "image_contract": contract,
    }
    (subject_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_index(output_root: Path, summaries: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for summary in summaries:
        subject = str(summary["subject"])
        title = "数学" if subject == "math" else "物理"
        images = "".join(
            f'<figure><img src="{subject}/{html.escape(name)}" '
            f'alt="{title} {html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>'
            for name in IMAGE_NAMES
        )
        cards.append(
            f"<section><h2>{title}</h2>"
            f"<p>四图合同：{'通过' if summary['image_contract']['valid'] else '失败'}；"
            f"直接终答负例：{'已拦截' if summary['direct_answer_blocked'] else '未拦截'}；"
            "模型调用：0。</p>"
            f'<div class="grid">{images}</div></section>'
        )
    page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>数学物理宽松答案门禁验收</title><style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:24px;background:#f4f5f7;color:#222}
section{background:white;padding:20px;margin:0 0 24px;border-radius:12px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
figure{margin:0}img{width:100%;border:1px solid #bbb;background:white}figcaption{margin-top:6px;color:#555}
@media(max-width:900px){.grid{grid-template-columns:1fr}}</style></head><body>
<h1>数学物理宽松答案门禁离线验收</h1>
<p>本页验证“过程词和得证允许、直接最终答案仍拦截”，全程未调用模型。</p>
""" + "".join(cards) + "</body></html>"
    (output_root / "index.html").write_text(page, encoding="utf-8")
    (output_root / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=("math", "physics"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        run_worker(args.worker, output)
        return 0

    for subject in ("math", "physics"):
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--output", str(output), "--worker", subject],
            check=True,
        )
    summaries = [
        json.loads((output / subject / "report.json").read_text(encoding="utf-8"))
        for subject in ("math", "physics")
    ]
    write_index(output, summaries)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
