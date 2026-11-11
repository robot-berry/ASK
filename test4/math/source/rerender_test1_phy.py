#!/usr/bin/env python3
"""Re-render test1 with the phy-derived Pillow renderer and math-style layout."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil

import math_guidance_renderer as renderer

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "test1"
DATASET = ROOT / "dataset"

CROP_OVERRIDES = {
    ("EINK2K-MATH-0068", "fig_1"): [40, 145, 315, 565],
    ("EINK2K-MATH-0445", "fig_table_1"): [225, 135, 675, 405],
    ("EINK2K-MATH-0445", "fig_table_2"): [225, 785, 435, 945],
    ("EINK2K-MATH-0491", "fig_table_1"): [135, 165, 730, 515],
    ("EINK2K-MATH-0491", "fig_form_1"): [175, 635, 425, 735],
    ("EINK2K-MATH-0491", "fig_crit_1"): [135, 735, 420, 945],
}

GUIDANCE_OVERRIDES = {
    "EINK2K-MATH-0401": {
        "advanced": [
            ("（1）", "三角恒等变换", [
                "先把 $f(x)$ 化为一个正弦型函数，再处理 $x+\\frac{\\pi}{2}$ 的相位变化",
                "将平方后的式子化为含 $\\cos 2x$ 的形式，由角频率判断最小正周期",
            ]),
            ("（2）", "积化和差", [
                "把 $f(x)$ 与 $f(x-\\frac{\\pi}{4})$ 分别化为振幅相位形式",
                "使用积化和差得到单一三角函数，并结合给定区间判断极值位置",
            ]),
        ],
        "standard": [
            ("（1）", "三角恒等变换", [
                "用 $\\sin x+\\cos x=\\sqrt{2}\\sin(x+\\frac{\\pi}{4})$ 改写 $f(x)$",
                "代入 $x+\\frac{\\pi}{2}$，再用平方关系化为标准周期形式",
                "根据整理后函数的角频率判断最小正周期",
            ]),
            ("（2）", "积化和差", [
                "整理两个因式的相位并使用积化和差公式合并乘积",
                "把结果写成关于 $2x$ 的单一三角函数",
                "将给定区间转为相位区间，比较端点与驻点",
            ]),
        ],
        "foundation": [
            ("（1）", "辅助角公式", [
                "先用辅助角公式把正弦与余弦合并",
                "代入新的自变量并使用平方公式整理",
                "从角频率与周期的关系判断最小正周期",
            ]),
            ("（2）", "积化和差", [
                "把两个因式写成统一的振幅相位形式",
                "用积化和差把乘积化成常数项与余弦项",
                "在题目给定范围内判断余弦函数的可能取值",
            ]),
        ],
    },
    "EINK2K-MATH-0446": {
        "advanced": [("整题", "等差数列结构法", [
            "选取条件①②，分别设 $a_n=a_1+(n-1)d$ 与 $\\sqrt{S_n}=u_1+(n-1)r$",
            "由 $S_n=[u_1+(n-1)r]^2$ 作差得到 $a_n=S_n-S_{n-1}$",
            "比较一次项系数建立参数关系，再代入 $a_2=a_1+d$ 整理",
        ])],
        "standard": [("整题", "等差数列结构法", [
            "选择①②作为已知条件，写出两个等差数列的通项",
            "由 $\\sqrt{S_n}=u_1+(n-1)r$ 得到前 $n$ 项和的平方表达式",
            "利用 $a_n=S_n-S_{n-1}$ 并与 $a_n=a_1+(n-1)d$ 比较系数",
            "结合正项条件排除不合题意的符号情形",
        ])],
        "foundation": [("整题", "等差数列结构法", [
            "先选①②，目标是从两个等差结构推出待证关系",
            "写出 $a_n=a_1+(n-1)d$ 和 $\\sqrt{S_n}=\\sqrt{S_1}+(n-1)r$",
            "两边平方后用 $a_n=S_n-S_{n-1}$ 转回数列项",
            "按 $n$ 的次数整理并比较系数，最后检查正项条件",
        ])],
    },
}

TEXT_REPLACEMENTS = {
    "senderperp": r"\perp",
    r"\senderperp": r"\perp",
    "b^{2}=a^{2-c}^{2}": "b^{2}=a^{2}-c^{2}",
    "a^{2}=b^{2+c}^{2}": "a^{2}=b^{2}+c^{2}",
    "ABC-A_{1B}_{1C}_{1}": "ABC-A_1B_1C_1",
    "AA_{1B}_{1B}": "AA_1B_1",
    "A_{1B}_{1}": "A_1B_1",
    "BB_{1C}_{1C}": "BB_1C_1C",
    "Coeff{a_{n}}": r"\{a_{n}\}",
    "Coeff{b_{n}}": r"\{b_{n}\}",
    "Coeff{S_{n} - 3}": r"\{S_{n}-3\}",
    "Coeff{\\sqrt{S_{n}}}": r"\{\sqrt{S_{n}}\}",
}


def latest_records() -> dict[str, dict]:
    latest = {}
    for line in (RUN / "results.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("status") == "success":
                latest[record["question_id"]] = record
    return latest


def normalize_text(value: str) -> str:
    for before, after in TEXT_REPLACEMENTS.items():
        value = value.replace(before, after)
    value = value.replace("//", r"\parallel").replace("⊥", r"\perp")
    value = re.sub(r"(?<!\\)vec\s*\(([^()]*)\)", r"\\vec{\1}", value)
    value = re.sub(r"(?<!\\)vec\s*\{", r"\\vec{", value)
    value = re.sub(
        r"\\vec\{([A-Za-z]+)_\{([0-9]+)\}\}",
        r"\\vec{\1}_{\2}",
        value,
    )
    value = re.sub(r"\\vec\{([A-Za-z]+)_([0-9]+)\}", r"\\vec{\1}_{\2}", value)
    # Repair a recurring provider typo in the coordinate dot-product formula:
    # ``x_{1x}_{2}`` means ``x_1 x_2`` (and likewise for y and z).
    value = re.sub(
        r"\b([xyz])_\{1\1\}_\{2\}",
        lambda match: f"{match.group(1)}_{{1}}{match.group(1)}_{{2}}",
        value,
    )
    value = re.sub(
        r"(?<!\\)(frac|sqrt|theta|alpha|beta|gamma|lambda|rho|pi|"
        r"sin|cos|tan|ln|log|perp|parallel|cdot|times|leq|geq|neq|"
        r"sum|prod|left|right|max|min)(?![A-Za-z])",
        r"\\\1",
        value,
    )
    value = value.replace("B_{1D}", "B_1D")
    value = re.sub(
        r"<\s*([^,<>]+)\s*,\s*([^<>]+)\s*>",
        r"\\langle \1,\2\\rangle",
        value,
    )
    line_plane_angle = (
        r"$\sin\theta=\frac{|\vec{AB}\cdot\vec{n}|}{|\vec{AB}||\vec{n}|}$"
    )
    # Consume the optional command slash as part of the repair.  Otherwise the
    # generic command normalizer can leave a visible ``\`` immediately before
    # the newly inserted formula delimiter.
    value = value.replace(
        r"\sinθ = |AB·n| / (|AB||n|)",
        line_plane_angle,
    )
    value = value.replace(
        "sinθ = |AB·n| / (|AB||n|)",
        line_plane_angle,
    )
    # Commands occurring in Chinese prose should be visible textbook symbols,
    # not raw backslash text. Structural commands remain for the formula parser.
    visible_commands = {
        r"\parallel": "∥", r"\perp": "⊥", r"\cdot": "·",
        r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\in": "∈",
        r"\theta": "θ", r"\alpha": "α", r"\beta": "β", r"\gamma": "γ",
        r"\lambda": "λ", r"\rho": "ρ", r"\pi": "π",
        r"\sin": "sin", r"\cos": "cos", r"\tan": "tan", r"\ln": "ln",
    }
    for command, visible in visible_commands.items():
        value = value.replace(command, visible)
    for token, symbol in {"in": "∈", "le": "≤", "ge": "≥", "ne": "≠"}.items():
        value = re.sub(rf"\b{token}\b", symbol, value)

    # Provider output occasionally prefixes a complete Chinese step with one
    # stray delimiter.  Remove only the unmatched edge marker so the genuine
    # inline formula pairs remain intact for the phy rich-formula renderer.
    while value.count("$") % 2 and value.startswith("$"):
        value = value[1:]
    while value.count("$") % 2 and value.endswith("$"):
        value = value[:-1]
    if value.count("$") % 2:
        value = value.replace("$", "")
    pieces = value.split("$")
    token_replacements = {
        "varphi": "φ", "phi": "φ", "omega": "ω",
    }
    for index in range(1, len(pieces), 2):
        pieces[index] = re.sub(r"\bCoS\b", "cos", pieces[index], flags=re.I)
        for token, symbol in token_replacements.items():
            pieces[index] = re.sub(rf"\b{token}\b", symbol, pieces[index])
    value = "$".join(pieces)
    return value


def wrap_vectors_in_prose(value: str) -> str:
    """Route un-delimited vector notation through the phy formula renderer."""
    pieces = value.split("$")
    for index in range(0, len(pieces), 2):
        pieces[index] = re.sub(
            r"\\vec\{([^{}]+)\}(?:_\{?([0-9]+)\}?)?",
            lambda match: "$" + r"\vec{" + match.group(1) + "}"
            + (("_{" + match.group(2) + "}") if match.group(2) else "") + "$",
            pieces[index],
        )
    return "$".join(pieces)


def normalize_tree(value):
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [normalize_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_tree(item) for key, item in value.items()}
    return value


def apply_guidance_overrides(qid: str, output: dict) -> None:
    for tier_key, parts in GUIDANCE_OVERRIDES.get(qid, {}).items():
        output["guidance"][tier_key] = {
            "title": {"advanced": "学霸版", "standard": "普通版", "foundation": "基础版"}[tier_key],
            "parts": [
                {
                    "part": part,
                    "method_id": "M1",
                    "method_name": method,
                    "steps": steps,
                    "alternative_methods": [],
                    **({"pitfall": "注意定义域、下标范围和恒等变形的适用条件。"}
                       if tier_key == "standard" else {}),
                }
                for part, method, steps in parts
            ],
            **({"pattern_summary": "先把题设写成标准通项或恒等式，再比较结构并检查取值限制。"}
               if tier_key == "foundation" else {}),
        }


def prepare_output(qid: str, source: dict) -> dict:
    output = normalize_tree(deepcopy(source))
    for item in output.get("base_formulas", []):
        if isinstance(item, dict) and isinstance(item.get("condition"), str):
            item["condition"] = wrap_vectors_in_prose(item["condition"])
    for tier in output.get("guidance", {}).values():
        for part in tier.get("parts", []):
            part["steps"] = [wrap_vectors_in_prose(step) for step in part.get("steps", [])]
            if isinstance(part.get("pitfall"), str):
                part["pitfall"] = wrap_vectors_in_prose(part["pitfall"])
            for method in part.get("alternative_methods", []):
                method["steps"] = [
                    wrap_vectors_in_prose(step) for step in method.get("steps", [])
                ]
    for figure in output.get("source_figures", []):
        override = CROP_OVERRIDES.get((qid, str(figure.get("figure_id"))))
        if override:
            figure["bbox_1000"] = override
    apply_guidance_overrides(qid, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", action="append", help="只重渲染指定题号，可重复")
    args = parser.parse_args()
    records = latest_records()
    selected = sorted(args.qid or records)

    backup = RUN / "images_katex_layout_backup"
    if not backup.exists() and (RUN / "images").exists():
        shutil.copytree(RUN / "images", backup)

    manifest = []
    for index, qid in enumerate(selected, 1):
        record = records[qid]
        source_image = DATASET / record["source_image"]["relative_path"]
        result = renderer.render_four_images(
            question_id=qid,
            model_id=record.get("model_id_response") or record.get("model_id_requested") or "unknown",
            source_image=source_image,
            output=prepare_output(qid, record["output_json"]),
            output_dir=RUN / "images" / qid,
            lock_source_figure_bboxes=False,
        )
        manifest.append({"question_id": qid, "renderer": "phy-derived-pillow", "render": result})
        print(f"[{index}/{len(selected)}] {qid}")

    (RUN / "phy_rerender_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
