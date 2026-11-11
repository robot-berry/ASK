from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .storage import append_jsonl, utc_now, write_json


TIER_TITLES = {
    "l1": "学霸解析",
    "l2": "中上解析",
    "l3": "中等解析",
    "l4": "中下解析",
    "l5": "基础解析",
}


def _pillow() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "渲染需要 Pillow；请执行 pip install -e .[render]"
        ) from exc
    return Image, ImageDraw, ImageFont


def _font(ImageFont: Any, size: int, *, bold: bool = False) -> Any:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


_SUPERSCRIPT = str.maketrans(
    "0123456789+-=()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
)
_SUBSCRIPT = str.maketrans(
    "0123456789+-=()", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
)
_LATEX_SYMBOLS = {
    "rightarrow": "→",
    "leftarrow": "←",
    "Rightarrow": "⇒",
    "Leftarrow": "⇐",
    "leftrightarrow": "↔",
    "approx": "≈",
    "neq": "≠",
    "ne": "≠",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "times": "×",
    "cdot": "·",
    "div": "÷",
    "pm": "±",
    "infty": "∞",
    "cap": "∩",
    "cup": "∪",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "subseteq": "⊆",
    "parallel": "∥",
    "perp": "⊥",
    "angle": "∠",
    "degree": "°",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "omega": "ω",
    "Delta": "Δ",
    "Omega": "Ω",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "log": "log",
    "ln": "ln",
}


def _plain_script(value: str, *, superscript: bool) -> str:
    compact = re.sub(r"\s+", "", value)
    table = _SUPERSCRIPT if superscript else _SUBSCRIPT
    translated = compact.translate(table)
    if compact and all(char in "0123456789+-=()" for char in compact):
        return translated
    label = "次方" if superscript else "下标"
    return f"（{compact}{label}）"


def humanize_math_markup(value: str) -> str:
    """Convert model LaTeX to readable text before public bitmap rendering.

    The raw model response remains untouched. This display-only conversion is
    deliberately fail-safe: no dollar delimiter or backslash command survives.
    """

    text = str(value)
    while "\\\\" in text:
        text = text.replace("\\\\", "\\")
    for token in ("\\[", "\\]", "\\(", "\\)", "$$", "$"):
        text = text.replace(token, " ")

    wrappers = (
        "mathrm",
        "textrm",
        "text",
        "operatorname",
        "mathbf",
        "mathit",
        "mathsf",
        "underline",
        "underbrace",
        "overline",
        "vec",
    )
    wrapper_pattern = r"\\(?:" + "|".join(wrappers) + r")\s*\{([^{}]*)\}"
    for _ in range(4):
        updated = re.sub(wrapper_pattern, r"\1", text)
        if updated == text:
            break
        text = updated

    fraction_pattern = re.compile(r"\\(?:d?frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    for _ in range(4):
        updated = fraction_pattern.sub(lambda match: f"({match.group(1)})/({match.group(2)})", text)
        if updated == text:
            break
        text = updated
    text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", text)
    text = re.sub(r"\\(?:left|right|quad|qquad)\b|\\[,;!]", "", text)

    for command in sorted(_LATEX_SYMBOLS, key=len, reverse=True):
        text = re.sub(
            rf"\\{re.escape(command)}(?![A-Za-z])",
            _LATEX_SYMBOLS[command],
            text,
        )

    left_brace = "\uFFF0"
    right_brace = "\uFFF1"
    text = text.replace("\\{", left_brace).replace("\\}", right_brace)
    for _ in range(4):
        updated = re.sub(
            r"\^\s*\{([^{}]*)\}",
            lambda match: _plain_script(match.group(1), superscript=True),
            text,
        )
        updated = re.sub(
            r"_\s*\{([^{}]*)\}",
            lambda match: _plain_script(match.group(1), superscript=False),
            updated,
        )
        if updated == text:
            break
        text = updated
    text = re.sub(
        r"\^\s*([A-Za-z0-9+\-=])",
        lambda match: _plain_script(match.group(1), superscript=True),
        text,
    )
    text = re.sub(
        r"_\s*([A-Za-z0-9+\-=])",
        lambda match: _plain_script(match.group(1), superscript=False),
        text,
    )

    text = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\([A-Z])\b", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = text.replace("\\", " ").replace("$", " ")
    text = text.replace("{", "").replace("}", "")
    text = text.replace(left_brace, "{").replace(right_brace, "}")
    text = re.sub(r"\s+([，。！？；：、,.!?;:）】》〉〕］}])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _wrap(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    closing_punctuation = "，。！？；：、,.!?;:）】》〉〕］}”’％%℃°"
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for char in paragraph:
            candidate = current + char
            width = draw.textbbox((0, 0), candidate, font=font)[2]
            if current and width > max_width:
                # 中文收尾标点允许悬挂到右侧留白，避免出现仅有“。”的一行。
                if char in closing_punctuation:
                    lines.append(candidate)
                    current = ""
                else:
                    lines.append(current)
                    current = char
            else:
                current = candidate
        if current or not lines:
            lines.append(current)
    return lines


def _text_values(value: Any) -> list[str]:
    """Losslessly collect strings from a loose model field for display."""

    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        result: list[str] = []
        for child in value:
            result.extend(_text_values(child))
        return result
    return []


def _tier_lines(answer: dict[str, Any], tier: str) -> list[str]:
    """Return public lines without requiring the strict evaluation schema.

    Exact sentence counts, optional extra fields and string-vs-array transport
    differences are evaluation concerns. They must not make an otherwise
    readable model response disappear from the PNG workflow.
    """

    value = answer.get(tier)
    if tier == "l2":
        return [f"• {humanize_math_markup(item)}" for item in _text_values(value)]

    parts = value if isinstance(value, list) else [value]
    lines: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            sentences = _text_values(part.get("s"))
            pitfalls = _text_values(part.get("pit")) if tier == "l5" else []
        else:
            sentences = _text_values(part)
            pitfalls = []
        # n 是模型 JSON 的结构定位字段，只用于各层对齐，不进入公开解析图。
        lines.extend(
            f"• {humanize_math_markup(sentence)}"
            for sentence in sentences
            if str(sentence).strip()
        )
        for pitfall in pitfalls:
            pit = humanize_math_markup(pitfall)
            if pit.strip():
                lines.append(pit if pit.startswith("易错点：") else f"易错点：{pit}")
    return lines


def assess_renderability(answer: dict[str, Any] | None) -> dict[str, Any]:
    """Apply only the minimum hard requirements needed to make five PNGs."""

    if not isinstance(answer, dict):
        return {
            "valid": False,
            "policy": "minimum_display_requirements_v1",
            "errors": ["模型回复无法解析为可展示对象"],
            "warnings": [],
            "tier_line_counts": {},
        }
    counts = {tier: len(_tier_lines(answer, tier)) for tier in TIER_TITLES}
    missing = [tier for tier, count in counts.items() if count == 0]
    return {
        "valid": not missing,
        "policy": "minimum_display_requirements_v1",
        "errors": (
            ["缺少可展示的层级：" + "、".join(missing)] if missing else []
        ),
        "warnings": [],
        "tier_line_counts": counts,
        "hard_requirements": ["回复可解析", "l1-l5 每层至少有一条可展示文字"],
    }


def render_tiers(
    answer: dict[str, Any], output_dir: Path, *, width: int = 800
) -> list[Path]:
    if width != 800:
        raise ValueError("当前设备工作流固定要求输出宽度为 800px")
    Image, ImageDraw, ImageFont = _pillow()
    output_dir.mkdir(parents=True, exist_ok=True)
    title_font = _font(ImageFont, 33, bold=True)
    body_font = _font(ImageFont, 27)
    margin = 40
    card_left = 36
    card_right = width - 36
    card_top = 108
    card_padding_x = 28
    card_padding_y = 28
    content_width = card_right - card_left - card_padding_x * 2
    line_height = 44
    paths: list[Path] = []
    for tier, title in TIER_TITLES.items():
        scratch = Image.new("RGB", (width, 100), "white")
        scratch_draw = ImageDraw.Draw(scratch)
        wrapped: list[tuple[str, str]] = []
        for source_line in _tier_lines(answer, tier):
            if source_line.startswith("易错点："):
                line_kind = "pitfall"
            elif source_line.startswith("•"):
                line_kind = "body"
            else:
                line_kind = "part"
            for line in _wrap(scratch_draw, source_line, body_font, content_width):
                wrapped.append((line, line_kind))
        card_height = max(116, card_padding_y * 2 + len(wrapped) * line_height)
        height = card_top + card_height + 36
        image = Image.new("RGB", (width, height), "#F3F6F8")
        draw = ImageDraw.Draw(image)
        draw.text((margin, 40), title, fill="#111111", font=title_font)
        draw.rounded_rectangle(
            (card_left + 2, card_top + 5, card_right + 2, card_top + card_height + 5),
            radius=22,
            fill="#E4E9ED",
        )
        draw.rounded_rectangle(
            (card_left, card_top, card_right, card_top + card_height),
            radius=22,
            fill="#FFFFFF",
        )
        y = card_top + card_padding_y
        x = card_left + card_padding_x
        for line, _line_kind in wrapped:
            draw.text((x, y), line, fill="#111111", font=body_font)
            y += line_height
        path = output_dir / f"{tier}.png"
        image.save(path, format="PNG", optimize=True)
        paths.append(path)
    return paths


def render_model_output_screenshot(
    raw_output: str, output_path: Path, *, width: int = 800
) -> Path:
    """Render the untouched model text as a comparison screenshot."""

    if width != 800:
        raise ValueError("模型回答截图固定要求输出宽度为 800px")
    Image, ImageDraw, ImageFont = _pillow()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title_font = _font(ImageFont, 31, bold=True)
    body_font = _font(ImageFont, 20)
    card_left = 28
    card_right = width - 28
    card_top = 100
    card_padding = 24
    content_width = card_right - card_left - card_padding * 2
    line_height = 32
    scratch = Image.new("RGB", (width, 100), "white")
    scratch_draw = ImageDraw.Draw(scratch)
    wrapped: list[str] = []
    for source_line in raw_output.splitlines() or [raw_output]:
        wrapped.extend(_wrap(scratch_draw, source_line, body_font, content_width))
    card_height = max(112, card_padding * 2 + len(wrapped) * line_height)
    height = card_top + card_height + 28
    image = Image.new("RGB", (width, height), "#F3F6F8")
    draw = ImageDraw.Draw(image)
    draw.text((36, 34), "模型原始回答", fill="#111111", font=title_font)
    draw.rounded_rectangle(
        (card_left + 2, card_top + 5, card_right + 2, card_top + card_height + 5),
        radius=20,
        fill="#E4E9ED",
    )
    draw.rounded_rectangle(
        (card_left, card_top, card_right, card_top + card_height),
        radius=20,
        fill="#FFFFFF",
    )
    y = card_top + card_padding
    for line in wrapped:
        draw.text((card_left + card_padding, y), line, fill="#111111", font=body_font)
        y += line_height
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def render_run(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Render one locally validated run for diagnostic visual inspection."""

    validation_path = run_dir / "validation.json"
    answer_path = run_dir / "display_answer.json"
    if not validation_path.is_file() or not answer_path.is_file():
        raise ValueError(f"运行目录缺少脱敏结果或校验报告：{run_dir}")
    validation = json.loads(validation_path.read_text(encoding="utf-8-sig"))
    if validation.get("display_ready") is not True:
        raise ValueError(f"该运行未通过展示安全门禁：{run_dir}")
    if output_dir.exists():
        raise ValueError(f"渲染输出目录已存在，避免覆盖：{output_dir}")
    answer = json.loads(answer_path.read_text(encoding="utf-8-sig"))
    paths = render_tiers(answer, output_dir, width=800)
    report = {
        "render_version": "test8_ans_card_png_v5_loose_display_gate",
        "completed_at": utc_now(),
        "run_dir": str(run_dir.resolve()),
        "quality_gate_pass": validation.get("quality_gate_pass") is True,
        "display_ready": True,
        "width_px": 800,
        "height_policy": "unbounded_content_height",
        "content_policy": "redacted_display_answer_only_without_internal_n_labels",
        "latex_policy": "public_unicode_humanized_without_source_markup",
        "images": [str(path.resolve()) for path in paths],
    }
    write_json(output_dir / "render_report.json", report)
    return report


def render_benchmark(summary_path: Path, output_dir: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if output_dir.exists():
        raise ValueError(f"渲染输出目录已存在，避免覆盖：{output_dir}")
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "render_manifest.jsonl"
    rendered: list[dict[str, Any]] = []
    for row in summary.get("results", []):
        if not row.get("display_ready"):
            continue
        run_dir = Path(str(row["run_dir"]))
        answer_path = run_dir / "display_answer.json"
        if not answer_path.is_file():
            raise ValueError(f"通过门槛的运行缺少 display_answer.json：{run_dir}")
        answer = json.loads(answer_path.read_text(encoding="utf-8-sig"))
        item_dir = output_dir / str(row["id"])
        paths = render_tiers(answer, item_dir, width=800)
        rendered_row = {
            "id": row["id"],
            "run_dir": str(run_dir),
            "width_px": 800,
            "images": [str(path.resolve()) for path in paths],
        }
        append_jsonl(manifest_path, rendered_row)
        rendered.append(rendered_row)
    report = {
        "render_version": "test8_ans_card_png_v5_loose_display_gate",
        "completed_at": utc_now(),
        "benchmark_summary": str(summary_path.resolve()),
        "width_px": 800,
        "height_policy": "unbounded_content_height",
        "content_policy": "redacted_display_answer_only_without_internal_n_labels",
        "latex_policy": "public_unicode_humanized_without_source_markup",
        "gate_policy": "render_display_ready_items_and_report_strict_quality_separately",
        "rendered_count": len(rendered),
        "items": rendered,
    }
    write_json(output_dir / "render_report.json", report)
    return report
