#!/usr/bin/env python3
"""Publish one overwrite-only monochrome result-image library per model.

The durable benchmark JSONL remains the audit source of truth.  This module
creates a compact, human-readable latest-result view:

    <library-root>/<model-key>/
        index.json
        <question-id>__question__001.png
        <question-id>__diagram__001.png
        <question-id>__analysis__001.png
        <question-id>__answer__001.png

Publishing is model-scoped and atomic.  A completed test replaces only the
selected model directories; other model libraries are left untouched.
"""

from __future__ import annotations

import argparse
import datetime as dt
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
except ImportError as exc:  # pragma: no cover - exercised by deployment only
    raise RuntimeError(
        "生成结果图需要 Pillow；请执行 pip install -r "
        "evaluation/requirements-render.txt"
    ) from exc


SCHEMA_VERSION = "model_result_image_library.v1"
WIDTH = 640
HEIGHT = 480
BODY_FONT_SIZE = 20
SCRIPT_FONT_SIZE = 14
# Keep stacked fractions close to the surrounding textbook formula size.
# The former 0.62 scale made otherwise identical formula lines look unrelated.
FRACTION_FONT_SCALE = 0.82
FRACTION_SCRIPT_SCALE = 0.72
FRACTION_RULE_GAP_SCALE = 0.14
FRACTION_RULE_MIN_GAP = 3
SMALL_FONT_SIZE = 14
TITLE_FONT_SIZE = 27
MARGIN_X = 24
BODY_TOP = 66
BODY_BOTTOM = 442
LINE_HEIGHT = 32
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")
ORIGINAL_QUESTION_ASSET_NAME = "original_question"
ORIGINAL_QUESTION_ASSET_SOURCE = "original_question_image"
ORIGINAL_QUESTION_TITLE = "完整原题"
ORIGINAL_QUESTION_MARGIN_X = 12
ORIGINAL_QUESTION_TOP = 12
ORIGINAL_QUESTION_BOTTOM = 420
ORIGINAL_QUESTION_LAYOUT_POLICY = "readable_document_tiles_v3"
ORIGINAL_QUESTION_CROP_PADDING = 16
ORIGINAL_QUESTION_MIN_READABLE_SCALE = 0.90
ORIGINAL_QUESTION_TILE_OVERLAP_X = 32
ORIGINAL_QUESTION_TILE_OVERLAP_Y = 32
ORIGINAL_QUESTION_LIGHT_BACKGROUND_MIN = 240
ORIGINAL_QUESTION_BORDER_TOLERANCE = 6
ORIGINAL_QUESTION_BORDER_UNIFORM_RATIO = 0.90
ORIGINAL_QUESTION_STRONG_INK_THRESHOLD = 245
ORIGINAL_QUESTION_SOFT_INK_THRESHOLD = 252
ORIGINAL_QUESTION_BINARY_THRESHOLD_MIN = 128
ORIGINAL_QUESTION_BINARY_THRESHOLD_MAX = 220
ORIGINAL_QUESTION_BINARIZATION_POLICY = "adaptive_background_contrast_v1"
ORIGINAL_QUESTION_ADAPTIVE_BLUR_RADIUS = 18
ORIGINAL_QUESTION_ADAPTIVE_DELTA = 9
ORIGINAL_QUESTION_ABSOLUTE_INK_THRESHOLD = 96
DEFAULT_PAGE_KINDS = ("question", "diagram", "analysis", "answer")
STRICT_THREE_IMAGE_KINDS = ("question", "analysis", "answer")
FIXED_HEIGHT_POLICY = "fixed_640x480_paginated"
AUTO_HEIGHT_POLICY = "width_640_content_auto_single_image"
AUTO_CONTENT_MIN_HEIGHT = 180
AUTO_CONTENT_FOOTER_SPACE = 38
GEOMETRY_ASSET_NAME = "geometry_diagram"
GEOMETRY_ASSET_SOURCE = ORIGINAL_QUESTION_ASSET_SOURCE
GEOMETRY_DIAGRAM_TITLE = "原题几何图"
# Historical v6 records are tolerated up to 800 characters for display so a
# coherent option-by-option check is not hidden.  New v7 output is held to the
# stricter semantic gate before it can become a successful result.
VERIFICATION_DISPLAY_MAX_CHARS = 800
EXAM_RULE_GUESS_MARKERS = (
    "标准正确答案",
    "标准答案",
    "参考答案",
    "命题人",
    "出题人",
    "命题意图",
    "考试意图",
    "考试规则",
    "通常这种题",
    "通常这类题",
    "通常是单选",
    "通常为单选",
    "如果是多选",
    "若是多选",
    "如果为多选",
    "若为多选",
    "如果是单选",
    "若是单选",
    "如果为单选",
    "若为单选",
    "干扰项",
    "代表性",
    "教材语境",
    "旨在纠正",
    "按考试习惯",
    "按命题习惯",
    "题目来源可能",
    "题源误差",
    "题源可能有误",
    "录入错误",
    "理解偏差",
    "最高频考点",
    "考察初衷",
    "必须二选一",
    "答案权重",
)
SOLUTION_SELF_TALK_REVIEW_MARKERS = (
    "我觉得",
    "答案可能是",
    "可能选",
    "可能为选项",
    "大概答案",
    "大概选",
    "猜一个",
    "先试一下",
    "先试试",
    "再试一下",
    "让我想想",
    "不完全是",
    "哪里错",
    "结论修正",
    "答案修正",
    "让我重新",
    "让我再算",
    "等等，",
    "等等,",
    "重新分析",
    "重新检查",
    "重新展开",
    "重新看",
    "重看",
    "之前的结论",
    "刚才的结论",
    "让我们再看",
    "让我们看一遍",
    "让我们再确认",
    "让我们再次确认",
)
SOLUTION_REVIEW_MARKERS = (
    *EXAM_RULE_GUESS_MARKERS,
    *SOLUTION_SELF_TALK_REVIEW_MARKERS,
    "可能会被认为",
)
VERIFICATION_REVIEW_MARKERS = SOLUTION_REVIEW_MARKERS + (
    "？",
    "?",
    "大概是",
    "不对，",
    "不对。",
    "再确认",
)
FINAL_ANSWER_REVIEW_MARKERS = EXAM_RULE_GUESS_MARKERS + (
    "答案可能是",
    "可能选",
    "可能为选项",
    "大概是",
    "大概选",
    "猜测",
    "猜一个",
    "应该是",
)
SOLUTION_REVIEW_MESSAGE = (
    "解题过程未通过逻辑一致性检查，需重新生成或人工复核"
)
# Compatibility alias for callers that used the verification-only display gate.
VERIFICATION_REVIEW_MESSAGE = SOLUTION_REVIEW_MESSAGE
SUBJECT_DISPLAY_NAMES = {
    "mathematics": "数学",
    "math": "数学",
    "physics": "物理",
    "chemistry": "化学",
    "biology": "生物",
    "chinese": "语文",
    "english": "英语",
}
SUPERSCRIPT_CHARACTERS = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "−": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
    "A": "ᴬ",
    "B": "ᴮ",
    "D": "ᴰ",
    "E": "ᴱ",
    "G": "ᴳ",
    "H": "ᴴ",
    "I": "ᴵ",
    "J": "ᴶ",
    "K": "ᴷ",
    "L": "ᴸ",
    "M": "ᴹ",
    "N": "ᴺ",
    "O": "ᴼ",
    "P": "ᴾ",
    "R": "ᴿ",
    "T": "ᵀ",
    "U": "ᵁ",
    "V": "ⱽ",
    "W": "ᵂ",
}
SUBSCRIPT_CHARACTERS = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "−": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "x": "ₓ",
}
SUPERSCRIPT_PLAIN_CHARACTERS = {
    rendered: plain
    for plain, rendered in SUPERSCRIPT_CHARACTERS.items()
    if plain != "−"
}
SUBSCRIPT_PLAIN_CHARACTERS = {
    rendered: plain
    for plain, rendered in SUBSCRIPT_CHARACTERS.items()
    if plain != "−"
}
COMBINING_OVERLINE = "\N{COMBINING OVERLINE}"
COMBINING_VECTOR_ARROW = "\N{COMBINING RIGHT ARROW ABOVE}"
FRACTION_START = "\ue100"
FRACTION_SEPARATOR = "\ue101"
FRACTION_END = "\ue102"
RADICAL_START = "\ue103"
RADICAL_END = "\ue104"
RICH_TEXT_MARKERS = {
    FRACTION_START,
    FRACTION_SEPARATOR,
    FRACTION_END,
    RADICAL_START,
    RADICAL_END,
}
UNSUPPORTED_FORMULA_MESSAGE = "公式含未支持命令，需人工复核"
# Times New Roman on the target Windows image exposes tofu boxes for a small
# set of textbook relation/geometry glyphs.  Physics guidance supplies SimSun
# as an explicit fallback; keep the list narrow so ordinary Latin/Greek math
# retains the requested Times typography.
# U+226A (≪) stays in Times New Roman: the bundled SimSun face reports an
# empty mask for that character, which previously left a visible blank in
# ``\Delta\omega\ll\omega`` even though the LaTeX parser was correct.
FORMULA_FALLBACK_SYMBOLS = frozenset("∠∝⊥∥∦≌△○⊙∩∪∈≤≥≠≈≡∅ℓ")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_component(value: Any, label: str) -> str:
    candidate = str(value or "").strip()
    if (
        not candidate
        or candidate in {".", ".."}
        or SAFE_COMPONENT.fullmatch(candidate) is None
    ):
        raise ValueError(
            f"{label}只能包含字母、数字、点、下划线和连字符"
        )
    return candidate


def filename_component(value: Any, label: str) -> str:
    """Return a stable filename component without rejecting Unicode IDs."""
    raw = str(value or "").strip()
    if not raw or raw in {".", ".."}:
        raise ValueError(f"{label}不能为空")
    if "\x00" in raw:
        raise ValueError(f"{label}包含空字符")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    if not slug:
        slug = "item"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:80]}--{digest}"


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} 的记录不是对象"
                )
            records.append(value)
    return records


def record_model(record: Mapping[str, Any]) -> tuple[str, str]:
    direct = record.get("model")
    if isinstance(direct, Mapping):
        key = direct.get("key")
        model_id = direct.get("model_id")
        if key:
            return (
                safe_component(key, "model key"),
                str(model_id or key),
            )
    models = record.get("models")
    if isinstance(models, Mapping):
        solver = models.get("solver")
        if isinstance(solver, Mapping) and solver.get("key"):
            key = safe_component(solver.get("key"), "model key")
            return key, str(solver.get("model_id") or key)
    task_key = str(record.get("task_key") or "")
    if "::" in task_key:
        key = safe_component(task_key.split("::", 1)[0], "model key")
        return key, key
    raise ValueError("记录缺少可识别的模型 key")


def latest_records(
    records: Iterable[Mapping[str, Any]],
    selected_model_keys: set[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    latest: dict[str, dict[str, dict[str, Any]]] = {}
    for source in records:
        record = dict(source)
        model_key, _ = record_model(record)
        if selected_model_keys is not None and model_key not in selected_model_keys:
            continue
        question_id = str(record.get("question_id") or "").strip()
        if not question_id:
            raise ValueError("question id不能为空")
        latest.setdefault(model_key, {})[question_id] = record
    return latest


def font_candidates(explicit: str | None = None) -> list[Path]:
    values = [
        explicit,
        os.environ.get("RESULT_IMAGE_FONT"),
        r"C:\Windows\Fonts\NotoSerifSC-VF.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\STSONG.TTF",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    return [Path(item).expanduser() for item in values if item]


def resolve_font_path(explicit: str | None = None) -> Path:
    for candidate in font_candidates(explicit):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "未找到可显示中文的字体；请使用 --font 或 RESULT_IMAGE_FONT 指定"
    )


@lru_cache(maxsize=16)
def load_font(path: str, size: int) -> Any:
    return ImageFont.truetype(path, size)


def vector_notation(match: re.Match[str]) -> str:
    """Mark a simple vector for font-independent arrow drawing."""
    value = match.group(1).strip()
    if len(value) == 1:
        return f"{value}{COMBINING_VECTOR_ARROW}"
    if re.fullmatch(r"[A-Za-z0-9]+", value):
        return "".join(
            character
            + (
                COMBINING_VECTOR_ARROW
                if index == len(value) - 1
                else COMBINING_OVERLINE
            )
            for index, character in enumerate(value)
        )
    return f"→({value})"


def directed_segment_notation(match: re.Match[str]) -> str:
    """Mark a directed segment for one arrow drawn above both endpoints."""
    value = match.group(1).strip()
    if re.fullmatch(r"[A-Za-z]{2}", value):
        return (
            f"{value[0]}{COMBINING_OVERLINE}"
            f"{value[1]}{COMBINING_VECTOR_ARROW}"
        )
    return f"→({value})"


def latex_braced_group(
    text: str,
    position: int,
) -> tuple[str, int] | None:
    while position < len(text) and text[position].isspace():
        position += 1
    if position >= len(text) or text[position] != "{":
        return None
    depth = 1
    start = position + 1
    index = start
    while index < len(text):
        character = text[index]
        if character == "\\":
            index += 2
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start:index], index + 1
        index += 1
    return None


def encode_latex_structures(text: str) -> str:
    """Encode valid fractions and radicals as nested internal rich tokens."""
    output: list[str] = []
    index = 0
    fraction_commands = (r"\frac", r"\dfrac", r"\tfrac")
    while index < len(text):
        fraction_command = next(
            (
                command
                for command in fraction_commands
                if text.startswith(command, index)
            ),
            None,
        )
        if fraction_command is not None:
            numerator = latex_braced_group(
                text,
                index + len(fraction_command),
            )
            if numerator is not None:
                denominator = latex_braced_group(text, numerator[1])
                if denominator is not None:
                    output.extend(
                        (
                            FRACTION_START,
                            encode_latex_structures(numerator[0]),
                            FRACTION_SEPARATOR,
                            encode_latex_structures(denominator[0]),
                            FRACTION_END,
                        )
                    )
                    index = denominator[1]
                    continue
        if text.startswith(r"\sqrt", index):
            radicand = latex_braced_group(text, index + len(r"\sqrt"))
            if radicand is not None:
                output.extend(
                    (
                        RADICAL_START,
                        encode_latex_structures(radicand[0]),
                        RADICAL_END,
                    )
                )
                index = radicand[1]
                continue
        output.append(text[index])
        index += 1
    return "".join(output)


def script_notation(
    value: str,
    characters: Mapping[str, str],
    fallback_marker: str,
) -> str:
    value = value.strip()
    if value and all(character in characters for character in value):
        return "".join(characters[character] for character in value)
    return f"{fallback_marker}({value})"


def replace_simple_scripts(text: str) -> str:
    text = re.sub(
        r"\^\{([^{}]+)\}",
        lambda match: script_notation(
            match.group(1),
            SUPERSCRIPT_CHARACTERS,
            "^",
        ),
        text,
    )
    text = re.sub(
        r"_\{([^{}]+)\}",
        lambda match: script_notation(
            match.group(1),
            SUBSCRIPT_CHARACTERS,
            "_",
        ),
        text,
    )
    text = re.sub(
        r"\^([A-Za-z0-9+\-=])",
        lambda match: script_notation(
            match.group(1),
            SUPERSCRIPT_CHARACTERS,
            "^",
        ),
        text,
    )
    text = re.sub(
        r"_([0-9+\-=])",
        lambda match: script_notation(
            match.group(1),
            SUBSCRIPT_CHARACTERS,
            "_",
        ),
        text,
    )
    return re.sub(
        r"_([A-Za-z])(?=$|[^A-Za-z0-9_])",
        lambda match: script_notation(
            match.group(1),
            SUBSCRIPT_CHARACTERS,
            "_",
        ),
        text,
    )


def combination_notation(match: re.Match[str]) -> str:
    lower = script_notation(
        match.group(1),
        SUBSCRIPT_CHARACTERS,
        "_",
    )
    upper = script_notation(
        match.group(2),
        SUPERSCRIPT_CHARACTERS,
        "^",
    )
    return f"C{lower}{upper}"


def overline_notation(match: re.Match[str]) -> str:
    return "".join(
        f"{character}{COMBINING_OVERLINE}"
        for character in match.group(1).strip()
    )


def hat_notation(match: re.Match[str]) -> str:
    """Keep a hat accent readable without relying on a combining glyph."""
    return f"带帽({match.group(1).strip()})"


def normalize_latex_command_prefixes(text: str) -> str:
    """Repair model output that escaped a LaTeX command more than once."""
    text = re.sub(
        r"(?:\\){2,}(?=[A-Za-z])",
        lambda _match: "\\",
        text,
    )
    for delimiter in ("(", ")", "[", "]"):
        text = text.replace(f"\\\\{delimiter}", delimiter)
        text = text.replace(f"\\{delimiter}", delimiter)
    return text


def readable_latex_environments(text: str) -> str:
    """Turn the common cases/array layouts into compact plain-text rows."""
    has_cases = re.search(r"\\+begin\s*\{cases\}", text) is not None
    has_array = re.search(r"\\+begin\s*\{array\}", text) is not None
    if not (has_cases or has_array):
        return text

    text = re.sub(
        r"\\+begin\s*\{cases\}",
        "分段：\n",
        text,
    )
    text = re.sub(r"\\+end\s*\{cases\}", "\n", text)
    text = re.sub(
        r"\\+begin\s*\{array\}\s*(?:\{[^{}]*\})?",
        "表格：\n",
        text,
    )
    text = re.sub(r"\\+end\s*\{array\}", "\n", text)
    text = re.sub(
        r"\\+hline(?![A-Za-z])",
        "\n---\n",
        text,
    )
    text = text.replace("\\\\", "\n")
    text = text.replace("&", " | " if has_array else "，")
    return text


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "；".join(clean_text(item) for item in value if item is not None)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value)
    text = "".join(
        character
        for character in text
        if character in "\n\t" or ord(character) >= 32
    )
    text = text.replace("\t", "  ").strip()
    text = text.replace("**", "").replace("$", "")
    text = readable_latex_environments(text)
    text = normalize_latex_command_prefixes(text)
    text = re.sub(
        r"\\not\s*\\subseteq(?![A-Za-z])",
        "⊈",
        text,
    )
    text = re.sub(
        r"\\not\s*\\subset(?![A-Za-z])",
        "⊄",
        text,
    )
    text = re.sub(
        r"\\not\s*\\parallel(?![A-Za-z])",
        "∦",
        text,
    )
    text = re.sub(
        r"\\not\s*=(?!=)",
        "≠",
        text,
    )
    text = re.sub(
        r"\\pmod\s*\{([^{}]+)\}",
        r"(mod \1)",
        text,
    )
    text = encode_latex_structures(text)
    text = re.sub(
        r"\\(?:text|mathrm|operatorname|mathbf|underline|mathbb|ce)"
        r"\{\s*\}",
        "",
        text,
    )
    # ``{}^{2}`` is a harmless TeX grouping idiom emitted after a primed
    # symbol (for example ``v'_{1}{}^{2}``).  It has no visible meaning, but
    # leaving it in the font stream exposes literal braces on the device.
    # Escaped visible braces (``\{\}``) are deliberately not removed.
    text = re.sub(r"(?<!\\)\{\s*(?<!\\)\}", "", text)
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\times": "×",
        r"\cdot": "·",
        r"\div": "÷",
        r"\pm": "±",
        r"\mp": "减/加",
        r"\leq": "≤",
        r"\le": "≤",
        r"\leqslant": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
        r"\geqslant": "≥",
        r"\neq": "≠",
        r"\ne": "≠",
        r"\approx": "≈",
        # Microsoft YaHei lacks U+223C on some Windows builds and renders it
        # as a tofu box.  ASCII tilde is the same school-textbook relation
        # mark at this display size and is guaranteed by the device font.
        r"\sim": "~",
        r"\cong": "≌",
        r"\equiv": "≡",
        r"\propto": "∝",
        r"\ll": "远小于",
        r"\infty": "∞",
        r"\sum": "Σ",
        r"\prod": "∏",
        r"\int": "∫",
        r"\lim": "lim",
        r"\sin": "sin",
        r"\cos": "cos",
        r"\tan": "tan",
        r"\cot": "cot",
        r"\log": "log",
        r"\ln": "ln",
        r"\exp": "exp",
        r"\max": "max",
        r"\min": "min",
        r"\det": "det",
        r"\sec": "sec",
        r"\csc": "csc",
        r"\arcsin": "arcsin",
        r"\arccos": "arccos",
        r"\arctan": "arctan",
        r"\cap": "∩",
        r"\cup": "∪",
        r"\in": "∈",
        r"\notin": "不属于",
        r"\subseteq": "⊆",
        r"\subset": "⊂",
        r"\setminus": "差",
        r"\complement": "补集",
        r"\varnothing": "∅",
        r"\emptyset": "∅",
        r"\ell": "ℓ",
        r"\forall": "任意",
        r"\exists": "存在",
        r"\odot": "⊙",
        r"\bigcirc": "○",
        r"\circ": "°",
        r"\triangle": "△",
        r"\blacktriangle": "▲",
        r"\rightarrow": "→",
        r"\longrightarrow": "→",
        r"\leftarrow": "←",
        r"\longleftarrow": "←",
        r"\leftrightarrow": "<->",
        r"\to": "→",
        r"\mapsto": "映射到",
        r"\Rightarrow": "=>",
        r"\implies": "=>",
        r"\nRightarrow": "不能推出",
        r"\Leftrightarrow": "<=>",
        r"\iff": "<=>",
        r"\uparrow": "↑",
        r"\downarrow": "↓",
        r"\mid": "|",
        r"\because": "∵",
        r"\therefore": "∴",
        r"\angle": "∠",
        r"\perp": "⊥",
        r"\parallel": "∥",
        r"\langle": "〈",
        r"\rangle": "〉",
        r"\dots": "…",
        r"\cdots": "…",
        r"\ldots": "…",
        r"\pmod": "mod",
        r"\qquad": " ",
        r"\quad": " ",
        r"\n": "\n",
        r"\pi": "π",
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\delta": "δ",
        r"\epsilon": "ε",
        r"\varepsilon": "ϵ",
        r"\zeta": "ζ",
        r"\eta": "η",
        r"\theta": "θ",
        r"\vartheta": "ϑ",
        r"\iota": "ι",
        r"\kappa": "κ",
        r"\lambda": "λ",
        r"\mu": "μ",
        r"\nu": "ν",
        r"\xi": "ξ",
        r"\rho": "ρ",
        r"\sigma": "σ",
        r"\tau": "τ",
        r"\upsilon": "υ",
        r"\phi": "φ",
        r"\varphi": "ϕ",
        r"\chi": "χ",
        r"\psi": "ψ",
        r"\omega": "ω",
        r"\Gamma": "Γ",
        r"\Delta": "Δ",
        r"\Theta": "Θ",
        r"\Lambda": "Λ",
        r"\Xi": "Ξ",
        r"\Pi": "Π",
        r"\Sigma": "Σ",
        r"\Phi": "Φ",
        r"\Psi": "Ψ",
        r"\Omega": "Ω",
        r"\,": " ",
        r"\;": " ",
        r"\!": "",
        r"\{": "{",
        r"\}": "}",
    }
    # Replace alphabetic commands in one left-to-right pass.  Replacing them
    # one command at a time can accidentally join adjacent commands: for
    # example replacing ``\cos`` first turns ``\alpha\cos`` into
    # ``\alphacos``, which is then mistaken for one unsupported command.
    alphabetic_replacements = {
        source[1:]: target
        for source, target in replacements.items()
        if source.startswith("\\") and source[1:].isalpha()
    }
    text = re.sub(
        r"\\([A-Za-z]+)",
        lambda match: alphabetic_replacements.get(
            match.group(1),
            match.group(0),
        ),
        text,
    )
    for source, target in replacements.items():
        if not (source.startswith("\\") and source[1:].isalpha()):
            text = text.replace(source, target)
    for _ in range(6):
        updated = text
        updated = re.sub(
            r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}",
            r"\1",
            updated,
        )
        updated = re.sub(
            r"\\mathbf\{([^{}]*)\}",
            r"\1",
            updated,
        )
        updated = re.sub(
            r"\\(?:underline|mathbb|ce)\{([^{}]*)\}",
            r"\1",
            updated,
        )
        updated = re.sub(
            r"\\binom\{([^{}]+)\}\{([^{}]+)\}",
            combination_notation,
            updated,
        )
        updated = re.sub(
            r"\\(?:overline|bar)\{([^{}]+)\}",
            overline_notation,
            updated,
        )
        updated = re.sub(
            r"\\hat\{([^{}]+)\}",
            hat_notation,
            updated,
        )
        updated = re.sub(
            r"\\overrightarrow\{([^{}]+)\}",
            directed_segment_notation,
            updated,
        )
        updated = re.sub(
            r"\\(?:boldsymbol|bm|vec)\{([^{}]+)\}",
            vector_notation,
            updated,
        )
        updated = replace_simple_scripts(updated)
        updated = re.sub(r"\^\{?∁\}?", "^(∁)", updated)
        if updated == text:
            break
        text = updated
    text = re.sub(
        r"原题第\s*([^：:\s]+)\s*题\s*[：:]",
        r"题面印刷题号\1：",
        text,
    )
    # The production Chinese font intentionally stays fixed for stable
    # pagination.  Normalize rare raw Unicode math glyphs that this font
    # renders as tofu boxes into equivalent readable notation.
    text = text.replace("\u0304", COMBINING_OVERLINE)
    text = text.replace("\u0233", f"y{COMBINING_OVERLINE}")
    text = re.sub(
        r"([A-Za-zΑ-Ωα-ω])\u0302",
        lambda match: f"带帽({match.group(1)})",
        text,
    )
    font_safe_visible_replacements = {
        "⇒": "=>",
        "⇔": "<=>",
        "⇏": "不能推出",
        "⇨": "->",
        "↔": "<->",
        "↦": "映射到",
        "∁": "补集",
        "∉": "不属于",
        "∀": "任意",
        "∃": "存在",
        "≪": "远小于",
        "≅": "≌",
        "∓": "减/加",
        "∖": "差",
        "∛": "三次根号",
        "⊗": "圈乘",
        "☐": "[ ]",
        "▢": "[ ]",
        "✓": "正确",
        "⌊": "下取整(",
        "⌋": ")",
        "ᵧ": "^(γ)",
    }
    for source, target in font_safe_visible_replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\s*([|∩∪])\s*", r"\1", text)
    text = re.sub(
        r"\\(?:dfrac|tfrac|frac)(?![A-Za-z])",
        "分数",
        text,
    )
    text = re.sub(r"\\sqrt(?![A-Za-z])", "根号", text)
    if re.search(r"\\[A-Za-z]+", text):
        return UNSUPPORTED_FORMULA_MESSAGE
    return text


SUBJECT_DISPLAY_LABELS = {
    "mathematics": "数学",
    "math": "数学",
    "physics": "物理",
}
QUESTION_TYPE_DISPLAY_LABELS = {
    "single_choice": "单项选择题",
    "multiple_choice": "多项选择题",
    "fill_blank": "填空题",
    "open_response": "解答题",
    "structured_response": "多小问综合题",
    "non_choice": "非选择题",
}
VERDICT_DISPLAY_LABELS = {
    "true": "成立",
    "false": "不成立",
    "unknown": "待确认",
}


def display_subject(value: Any) -> str:
    text = clean_text(value)
    return SUBJECT_DISPLAY_LABELS.get(text.lower(), text)


def display_question_type(value: Any, *, has_options: bool = False) -> str:
    text = clean_text(value)
    if text.lower() == "unknown":
        return "选择题（单选/多选未标明）" if has_options else "题型待确认"
    return QUESTION_TYPE_DISPLAY_LABELS.get(text.lower(), text)


def display_verdict(value: Any) -> str:
    text = clean_text(value)
    return VERDICT_DISPLAY_LABELS.get(text.lower(), text)


def add_field(
    output: list[tuple[str, str]],
    label: str,
    value: Any,
) -> None:
    text = clean_text(value)
    if text:
        output.append((label, text))


def source_image_metadata(record: Mapping[str, Any]) -> str:
    subject = clean_text(record.get("subject"))
    subject_label = SUBJECT_DISPLAY_NAMES.get(subject.casefold(), subject)
    source_group = clean_text(record.get("source_group"))
    source_index = clean_text(record.get("source_index"))
    parts = [item for item in (subject_label, source_group) if item]
    if source_index:
        parts.append(f"第{source_index}张图")
    return " / ".join(parts)


def _text_display_issue(
    value: Any,
    *,
    max_characters: int | None = None,
    markers: Sequence[str] = SOLUTION_REVIEW_MARKERS,
) -> str | None:
    """Return the first display-only issue found in one solution text."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if max_characters is not None and len(text) > max_characters:
        return "too_long"
    marker = next(
        (
            candidate
            for candidate in markers
            if candidate in text
        ),
        None,
    )
    return f"marker:{marker}" if marker is not None else None


def verification_display_issue(
    result: Mapping[str, Any],
) -> str | None:
    """Return the verification portion of the solution display check."""
    verification = result.get("verification")
    if isinstance(verification, Mapping):
        return _text_display_issue(
            verification.get("result"),
            max_characters=VERIFICATION_DISPLAY_MAX_CHARS,
            markers=VERIFICATION_REVIEW_MARKERS,
        )
    solution = result.get("solution")
    if not isinstance(solution, Mapping):
        solution = {}
    return _text_display_issue(
        solution.get("verification"),
        max_characters=VERIFICATION_DISPLAY_MAX_CHARS,
        markers=VERIFICATION_REVIEW_MARKERS,
    )


def solution_display_issue(
    result: Mapping[str, Any],
) -> str | None:
    """Return an auditable reason for isolating generated solution content."""
    analysis = result.get("analysis")
    if isinstance(analysis, Mapping):
        for key in ("explanation", "method"):
            issue = _text_display_issue(analysis.get(key))
            if issue is not None:
                return f"analysis.{key}:{issue}"
        analysis_steps = analysis.get("steps")
        if isinstance(analysis_steps, list):
            for index, step in enumerate(analysis_steps):
                issue = _text_display_issue(step)
                if issue is not None:
                    return f"analysis.steps[{index}]:{issue}"

    solution = result.get("solution")
    if not isinstance(solution, Mapping):
        solution = {}
    steps = solution.get("steps")
    if isinstance(steps, list):
        for index, step in enumerate(steps):
            issue = _text_display_issue(step)
            if issue is not None:
                return f"solution.steps[{index}]:{issue}"
    else:
        issue = _text_display_issue(steps)
        if issue is not None:
            return f"solution.steps:{issue}"

    issue = verification_display_issue(result)
    if issue is not None:
        return f"solution.verification:{issue}"

    final_answer = result.get("final_answer")
    if isinstance(final_answer, Mapping):
        issue = _text_display_issue(
            final_answer.get("answer"),
            markers=FINAL_ANSWER_REVIEW_MARKERS,
        )
        if issue is not None:
            return f"final_answer.answer:{issue}"
    final = result.get("final")
    if isinstance(final, Mapping):
        issue = _text_display_issue(
            final.get("answer"),
            markers=FINAL_ANSWER_REVIEW_MARKERS,
        )
        if issue is not None:
            return f"final.answer:{issue}"
    issue = _text_display_issue(
        result.get("answer"),
        markers=FINAL_ANSWER_REVIEW_MARKERS,
    )
    if issue is not None:
        return f"answer:{issue}"
    return None


def analysis_fields(
    result: Mapping[str, Any],
    record: Mapping[str, Any] | None = None,
) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    if record is not None:
        add_field(output, "题图序号", source_image_metadata(record))
    question = result.get("question")
    reasoning = result.get("reasoning")
    final = result.get("final")
    if isinstance(question, Mapping) and isinstance(reasoning, Mapping):
        verdicts = result.get("option_verdicts")
        has_options = isinstance(verdicts, list) and bool(verdicts)
        overview = " / ".join(
            clean_text(value)
            for value in (
                display_subject(question.get("subject")),
                display_question_type(
                    question.get("question_type"), has_options=has_options
                ),
            )
            if clean_text(value)
        )
        add_field(output, "分类", overview)
        add_field(output, "知识点", reasoning.get("knowledge_points"))
        add_field(output, "目标", question.get("target"))
        add_field(output, "图中事实", question.get("visual_facts"))
        add_field(output, "基础公式", reasoning.get("formulas"))
        if isinstance(verdicts, list):
            for verdict in verdicts:
                if not isinstance(verdict, Mapping):
                    continue
                add_field(
                    output,
                    f"选项{clean_text(verdict.get('option'))}",
                    (
                        f"{display_verdict(verdict.get('verdict'))}；"
                        f"{clean_text(verdict.get('evidence'))}"
                    ),
                )
        derivation = reasoning.get("derivation")
        if isinstance(derivation, list):
            for index, step in enumerate(derivation, 1):
                add_field(output, f"步骤{index}", step)
        else:
            add_field(output, "步骤", derivation)
        add_field(output, "验证", reasoning.get("independent_check"))
        if isinstance(final, Mapping) and final.get("needs_review") is True:
            add_field(output, "复核提示", final.get("review_reasons"))
        return output or [("结果", "模型未生成可显示的解析内容")]
    classification = result.get("classification")
    analysis = result.get("analysis")
    verification = result.get("verification")
    if (
        isinstance(final, Mapping)
        and isinstance(classification, Mapping)
        and isinstance(analysis, Mapping)
        and isinstance(verification, Mapping)
    ):
        overview = " / ".join(
            clean_text(value)
            for value in (
                classification.get("subject"),
                classification.get("grade"),
                classification.get("question_type"),
            )
            if clean_text(value)
        )
        add_field(output, "分类", overview)
        add_field(
            output,
            "知识点",
            [
                classification.get("knowledge_level_1"),
                classification.get("knowledge_level_2"),
                classification.get("knowledge_level_3"),
            ],
        )
        add_field(output, "题目", analysis.get("normalized_question"))
        add_field(output, "已知", analysis.get("known_conditions"))
        add_field(output, "目标", analysis.get("target"))
        add_field(output, "基础公式", analysis.get("key_relations"))
        if solution_display_issue(result) is not None:
            add_field(output, "解题状态", SOLUTION_REVIEW_MESSAGE)
            return output
        add_field(output, "方法", analysis.get("method"))
        steps = analysis.get("steps")
        if isinstance(steps, list):
            for index, step in enumerate(steps, 1):
                add_field(output, f"步骤{index}", step)
        else:
            add_field(output, "步骤", steps)
        verification_text = "；".join(
            clean_text(value)
            for value in (
                verification.get("method"),
                verification.get("result"),
            )
            if clean_text(value)
        )
        add_field(output, "验证", verification_text)
        if final.get("needs_review") is True:
            add_field(output, "复核提示", final.get("review_reasons"))
        return output or [("结果", "模型未生成可显示的解析内容")]
    solution = result.get("solution")
    if not isinstance(classification, Mapping):
        classification = {}
    if not isinstance(analysis, Mapping):
        analysis = {}
    if not isinstance(solution, Mapping):
        solution = {}

    if "explanation" in analysis:
        overview = " / ".join(
            clean_text(value)
            for value in (
                analysis.get("subject"),
                analysis.get("question_type"),
            )
            if clean_text(value)
        )
        add_field(output, "题型分析", overview)
        add_field(
            output,
            "知识点",
            analysis.get("knowledge_points"),
        )
        if solution_display_issue(result) is not None:
            add_field(output, "解题状态", SOLUTION_REVIEW_MESSAGE)
            return output
        add_field(output, "解析", analysis.get("explanation"))
        return output or [("结果", "模型未生成可显示的解析内容")]

    overview = " / ".join(
        clean_text(value)
        for value in (
            classification.get("subject"),
            classification.get("grade"),
            classification.get("question_type"),
            classification.get("difficulty"),
        )
        if clean_text(value)
    )
    add_field(output, "分类", overview)
    knowledge = [
        classification.get("knowledge_level_1"),
        classification.get("knowledge_level_2"),
        classification.get("knowledge_level_3"),
    ]
    if isinstance(classification.get("knowledge_points"), list):
        knowledge.extend(classification["knowledge_points"])
    add_field(
        output,
        "知识点",
        [item for item in knowledge if clean_text(item)],
    )
    add_field(
        output,
        "题目",
        analysis.get("normalized_question")
        or analysis.get("normalized_problem"),
    )
    add_field(output, "已知", analysis.get("known_conditions"))
    add_field(output, "目标", analysis.get("target"))
    add_field(output, "基础公式", analysis.get("key_relations"))
    if solution_display_issue(result) is not None:
        add_field(output, "解题状态", SOLUTION_REVIEW_MESSAGE)
        return output
    add_field(output, "方法", analysis.get("method"))
    steps = solution.get("steps")
    if isinstance(steps, list):
        for index, step in enumerate(steps, 1):
            add_field(output, f"步骤{index}", step)
    else:
        add_field(output, "步骤", steps)
    add_field(
        output,
        "验证",
        solution.get("verification") or solution.get("checks"),
    )
    return output or [("结果", "模型未生成可显示的解析内容")]


def answer_fields(result: Mapping[str, Any]) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    question = result.get("question")
    reasoning = result.get("reasoning")
    final = result.get("final")
    if isinstance(question, Mapping) and isinstance(final, Mapping):
        add_field(output, "学科", question.get("subject"))
        if isinstance(reasoning, Mapping):
            add_field(output, "知识点", reasoning.get("knowledge_points"))
        add_field(output, "最终答案", final.get("answer"))
        confidence = final.get("confidence")
        if confidence is not None:
            add_field(output, "置信度", confidence)
        if final.get("needs_review") is True:
            add_field(output, "需要复核", final.get("review_reasons"))
        return output or [("最终答案", "模型未生成可显示的答案")]
    classification = result.get("classification")
    analysis = result.get("analysis")
    if isinstance(final, Mapping) and isinstance(classification, Mapping):
        add_field(output, "学科", classification.get("subject"))
        add_field(
            output,
            "知识点",
            [
                classification.get("knowledge_level_1"),
                classification.get("knowledge_level_2"),
                classification.get("knowledge_level_3"),
            ],
        )
        if solution_display_issue(result) is not None:
            add_field(output, "人工复核", SOLUTION_REVIEW_MESSAGE)
            return output
        add_field(output, "最终答案", final.get("answer"))
        confidence = final.get("confidence")
        if confidence is not None:
            add_field(output, "置信度", confidence)
        if final.get("needs_review") is True:
            add_field(output, "需要复核", final.get("review_reasons"))
        return output or [("最终答案", "模型未生成可显示的答案")]
    final_answer = result.get("final_answer")
    if not isinstance(classification, Mapping):
        classification = {}
    if not isinstance(analysis, Mapping):
        analysis = {}
    if not isinstance(final_answer, Mapping):
        final_answer = {}
    add_field(
        output,
        "学科",
        classification.get("subject") or analysis.get("subject"),
    )
    knowledge = [
        classification.get("knowledge_level_1"),
        classification.get("knowledge_level_2"),
        classification.get("knowledge_level_3"),
    ]
    if isinstance(analysis.get("knowledge_points"), list):
        knowledge.extend(analysis["knowledge_points"])
    add_field(
        output,
        "知识点",
        [item for item in knowledge if clean_text(item)],
    )
    if solution_display_issue(result) is not None:
        add_field(output, "人工复核", SOLUTION_REVIEW_MESSAGE)
        return output
    add_field(
        output,
        "最终答案",
        final_answer.get("answer") or result.get("answer"),
    )
    confidence = final_answer.get("confidence")
    if confidence is not None:
        add_field(output, "置信度", confidence)
    return output or [("最终答案", "模型未生成可显示的答案")]


def rich_token_at(
    text: str,
    position: int,
) -> tuple[str, int, str, str | None] | None:
    start = text[position] if position < len(text) else ""
    if start not in {FRACTION_START, RADICAL_START}:
        return None
    stack = [
        FRACTION_END if start == FRACTION_START else RADICAL_END
    ]
    separator: int | None = None
    index = position + 1
    while index < len(text):
        character = text[index]
        if character == FRACTION_START:
            stack.append(FRACTION_END)
        elif character == RADICAL_START:
            stack.append(RADICAL_END)
        elif stack and character == stack[-1]:
            stack.pop()
            if not stack:
                if start == FRACTION_START:
                    if separator is None:
                        return None
                    return (
                        "fraction",
                        index + 1,
                        text[position + 1:separator],
                        text[separator + 1:index],
                    )
                return (
                    "radical",
                    index + 1,
                    text[position + 1:index],
                    None,
                )
        elif (
            character == FRACTION_SEPARATOR
            and start == FRACTION_START
            and len(stack) == 1
            and separator is None
        ):
            separator = index
        index += 1
    return None


def formula_elements(
    text: str,
) -> list[tuple[str, str, str | None]]:
    elements: list[tuple[str, str, str | None]] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        if buffer:
            elements.append(("text", "".join(buffer), None))
            buffer.clear()

    marker_fallbacks = {
        FRACTION_START: "分数(",
        FRACTION_SEPARATOR: ")/(",
        FRACTION_END: ")",
        RADICAL_START: "根号(",
        RADICAL_END: ")",
    }
    index = 0
    while index < len(text):
        character = text[index]
        if character in {"^", "_"}:
            script = parenthesized_script(text, index)
            if script is not None:
                flush_buffer()
                elements.append(
                    (
                        "superscript" if character == "^" else "subscript",
                        script[0],
                        None,
                    )
                )
                index = script[1]
                continue
        token = rich_token_at(text, index)
        if token is not None:
            flush_buffer()
            kind, end, first, second = token
            elements.append((kind, first, second))
            index = end
            continue
        character = text[index]
        buffer.append(marker_fallbacks.get(character, character))
        index += 1
    flush_buffer()
    return elements


def formula_atoms(text: str) -> list[str]:
    """Return wrapping atoms; rich structures and decorations stay intact."""
    atoms: list[str] = []
    index = 0
    while index < len(text):
        token = rich_token_at(text, index)
        if token is not None:
            atoms.append(text[index:token[1]])
            index = token[1]
            continue
        character = text[index]
        if character in {"^", "_"}:
            fallback = parenthesized_script(text, index)
            if fallback is not None:
                atoms.append(text[index:fallback[1]])
                index = fallback[1]
                continue
        if (
            character in {COMBINING_OVERLINE, COMBINING_VECTOR_ARROW}
            and atoms
        ):
            atoms[-1] += character
        else:
            atoms.append(character)
        index += 1
    return atoms


def parenthesized_script(
    text: str,
    position: int,
) -> tuple[str, int] | None:
    if position + 1 >= len(text) or text[position + 1] != "(":
        return None
    depth = 1
    index = position + 2
    while index < len(text):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[position + 2:index], index + 1
        index += 1
    return None


def font_safe_script_value(value: str) -> str:
    marker_fallbacks = {
        FRACTION_START: "分数(",
        FRACTION_SEPARATOR: ")/(",
        FRACTION_END: ")",
        RADICAL_START: "根号(",
        RADICAL_END: ")",
        COMBINING_OVERLINE: "-",
        COMBINING_VECTOR_ARROW: "→",
    }
    output: list[str] = []
    for character in value:
        if character in {"\\", "^"}:
            continue
        if character in SUPERSCRIPT_PLAIN_CHARACTERS:
            output.append(SUPERSCRIPT_PLAIN_CHARACTERS[character])
        elif character in SUBSCRIPT_PLAIN_CHARACTERS:
            output.append(SUBSCRIPT_PLAIN_CHARACTERS[character])
        else:
            output.append(marker_fallbacks.get(character, character))
    return "".join(output)


def formula_runs(text: str) -> list[tuple[str, str, bool, bool]]:
    """Split internal formula text into drawable, font-safe runs."""
    units: list[list[Any]] = []
    index = 0
    while index < len(text):
        character = text[index]
        if character in {"^", "_"}:
            fallback = parenthesized_script(text, index)
            if fallback is not None:
                style = (
                    "superscript"
                    if character == "^"
                    else "subscript"
                )
                visible = font_safe_script_value(fallback[0])
                units.append([style, visible, False, False])
                index = fallback[1]
                continue
            if index + 1 < len(text):
                style = (
                    "superscript"
                    if character == "^"
                    else "subscript"
                )
                units.append(
                    [
                        style,
                        font_safe_script_value(text[index + 1]),
                        False,
                        False,
                    ]
                )
                index += 2
                continue
            units.append(
                [
                    "base",
                    "上标" if character == "^" else "下标",
                    False,
                    False,
                ]
            )
            index += 1
            continue
        if character == "\\":
            index += 1
            continue
        if character == COMBINING_OVERLINE:
            if units and units[-1][0] == "base":
                units[-1][2] = True
            else:
                units.append(["base", "-", False, False])
            index += 1
            continue
        if character == COMBINING_VECTOR_ARROW:
            if units and units[-1][0] == "base":
                units[-1][3] = True
            else:
                units.append(["base", "→", False, False])
            index += 1
            continue
        if character in SUPERSCRIPT_PLAIN_CHARACTERS:
            units.append(
                [
                    "superscript",
                    SUPERSCRIPT_PLAIN_CHARACTERS[character],
                    False,
                    False,
                ]
            )
            index += 1
            continue
        if character in SUBSCRIPT_PLAIN_CHARACTERS:
            units.append(
                [
                    "subscript",
                    SUBSCRIPT_PLAIN_CHARACTERS[character],
                    False,
                    False,
                ]
            )
            index += 1
            continue
        units.append(["base", character, False, False])
        index += 1

    runs: list[list[Any]] = []
    for style, value, overline, vector in units:
        if (
            runs
            and runs[-1][0] == style
            and runs[-1][2] == overline
            and not runs[-1][3]
            and not vector
        ):
            runs[-1][1] += value
        else:
            runs.append([style, value, overline, vector])
    return [
        (str(style), str(value), bool(overline), bool(vector))
        for style, value, overline, vector in runs
    ]


def formula_character_uses_fallback(character: str) -> bool:
    """Return whether one visible formula glyph should use the CJK fallback."""
    if not character:
        return False
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        # CJK punctuation may legitimately appear inside one formula span,
        # for example ``N=3、4、5、\dots`` or a list of requested symbols.
        # Times New Roman renders U+3001 as its .notdef square on the target
        # image, while the configured CJK fallback contains the real glyph.
        or 0x3000 <= code <= 0x303F
        or 0xFF00 <= code <= 0xFFEF
        or character in FORMULA_FALLBACK_SYMBOLS
    )


def formula_font_runs(
    value: str,
    primary_font: Any,
    fallback_font: Any | None = None,
) -> list[tuple[str, Any]]:
    """Split visible formula text into runs supported by the selected font."""
    if not value:
        return []
    if fallback_font is None:
        return [(value, primary_font)]
    output: list[tuple[str, Any]] = []
    for character in value:
        active_font = (
            fallback_font
            if formula_character_uses_fallback(character)
            else primary_font
        )
        if output and output[-1][1] is active_font:
            output[-1] = (output[-1][0] + character, active_font)
        else:
            output.append((character, active_font))
    return output


def formula_run_width(
    draw: ImageDraw.ImageDraw,
    value: str,
    primary_font: Any,
    fallback_font: Any | None = None,
) -> float:
    return float(
        sum(
            draw.textlength(run.replace("\r", " ").replace("\n", " "), font=font)
            for run, font in formula_font_runs(
                value,
                primary_font,
                fallback_font,
            )
        )
    )


def script_font_for(body_font: Any, script_font: Any | None = None) -> Any:
    if script_font is not None:
        return script_font
    font_variant = getattr(body_font, "font_variant", None)
    if callable(font_variant):
        try:
            return font_variant(size=SCRIPT_FONT_SIZE)
        except (OSError, TypeError, ValueError):
            pass
    return body_font


def scaled_font_for(
    font: Any,
    scale: float,
    minimum_size: int,
) -> Any:
    raw_size = getattr(font, "size", BODY_FONT_SIZE)
    try:
        size = max(minimum_size, int(round(float(raw_size) * scale)))
    except (TypeError, ValueError):
        size = minimum_size
    font_variant = getattr(font, "font_variant", None)
    if callable(font_variant):
        try:
            return font_variant(size=size)
        except (OSError, TypeError, ValueError):
            pass
    return font


def radical_hook_width(font: Any) -> int:
    raw_size = getattr(font, "size", BODY_FONT_SIZE)
    try:
        size = float(raw_size)
    except (TypeError, ValueError):
        size = BODY_FONT_SIZE
    return max(9, int(round(size * 0.55)))


def text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: Any,
    script_font: Any | None = None,
    *,
    fallback_body_font: Any | None = None,
    fallback_script_font: Any | None = None,
) -> float:
    small_font = script_font_for(font, script_font)
    total = 0.0
    for kind, first, second in formula_elements(text):
        if kind == "text":
            total += sum(
                formula_run_width(
                    draw,
                    value,
                    font if style == "base" else small_font,
                    fallback_body_font
                    if style == "base"
                    else fallback_script_font,
                )
                for style, value, _, _ in formula_runs(first)
            )
            continue
        if kind == "fraction" and second is not None:
            fraction_font = scaled_font_for(
                font,
                FRACTION_FONT_SCALE,
                8,
            )
            fraction_script_font = scaled_font_for(
                fraction_font,
                FRACTION_SCRIPT_SCALE,
                7,
            )
            fraction_fallback_font = (
                scaled_font_for(
                    fallback_body_font,
                    FRACTION_FONT_SCALE,
                    8,
                )
                if fallback_body_font is not None
                else None
            )
            fraction_fallback_script_font = (
                scaled_font_for(
                    fraction_fallback_font,
                    FRACTION_SCRIPT_SCALE,
                    7,
                )
                if fraction_fallback_font is not None
                else None
            )
            numerator_width = text_width(
                draw,
                first,
                fraction_font,
                fraction_script_font,
                fallback_body_font=fraction_fallback_font,
                fallback_script_font=fraction_fallback_script_font,
            )
            denominator_width = text_width(
                draw,
                second,
                fraction_font,
                fraction_script_font,
                fallback_body_font=fraction_fallback_font,
                fallback_script_font=fraction_fallback_script_font,
            )
            total += max(numerator_width, denominator_width) + 6
            continue
        if kind == "radical":
            total += (
                radical_hook_width(font)
                + 2
                + text_width(
                    draw,
                    first,
                    font,
                    small_font,
                    fallback_body_font=fallback_body_font,
                    fallback_script_font=fallback_script_font,
                )
            )
            continue
        if kind in {"superscript", "subscript"}:
            nested_script_font = scaled_font_for(
                small_font,
                FRACTION_SCRIPT_SCALE,
                7,
            )
            total += text_width(
                draw,
                first,
                small_font,
                nested_script_font,
                fallback_body_font=fallback_script_font,
                fallback_script_font=(
                    scaled_font_for(
                        fallback_script_font,
                        FRACTION_SCRIPT_SCALE,
                        7,
                    )
                    if fallback_script_font is not None
                    else None
                ),
            )
    return total


def wrap_line(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: Any,
    max_width: int,
    script_font: Any | None = None,
) -> list[str]:
    if not text:
        return [""]
    output: list[str] = []
    for raw_line in text.splitlines() or [""]:
        if not raw_line:
            output.append("")
            continue
        current = ""
        for atom in formula_atoms(raw_line):
            candidate = current + atom
            if (
                current
                and text_width(
                    draw,
                    candidate,
                    font,
                    script_font,
                )
                > max_width
            ):
                output.append(current)
                current = atom
            else:
                current = candidate
        output.append(current)
    return output


def layout_field_blocks(
    fields: Sequence[tuple[str, str]],
    body_font: Any,
    script_font: Any | None = None,
) -> list[list[str]]:
    probe = Image.new("1", (WIDTH, HEIGHT), 1)
    draw = ImageDraw.Draw(probe)
    available_width = WIDTH - 2 * MARGIN_X
    blocks: list[list[str]] = []
    for label, value in fields:
        prefix = f"{label}："
        wrapped = wrap_line(
            draw,
            prefix + clean_text(value),
            body_font,
            available_width,
            script_font,
        )
        blocks.append(wrapped)
    return blocks or [["无可显示内容"]]


def layout_lines(
    fields: Sequence[tuple[str, str]],
    body_font: Any,
    script_font: Any | None = None,
) -> list[str]:
    """Return the historical flat layout with one blank between fields."""
    lines: list[str] = []
    for block in layout_field_blocks(fields, body_font, script_font):
        if lines:
            lines.append("")
        lines.extend(block)
    return lines or ["无可显示内容"]


def paginate(lines: Sequence[str]) -> list[list[str]]:
    per_page = max(1, (BODY_BOTTOM - BODY_TOP) // LINE_HEIGHT)
    return [
        list(lines[index:index + per_page])
        for index in range(0, len(lines), per_page)
    ] or [["无可显示内容"]]


def paginate_field_blocks(
    blocks: Sequence[Sequence[str]],
    *,
    prefer_spacing: bool,
) -> list[list[str]]:
    """Paginate complete fields, splitting only a field taller than one page."""
    per_page = max(1, (BODY_BOTTOM - BODY_TOP) // LINE_HEIGHT)
    pages: list[list[str]] = []
    current: list[str] = []

    for raw_block in blocks:
        block = list(raw_block) or [""]
        if len(block) <= per_page:
            if current:
                spacing = 1 if prefer_spacing else 0
                if len(current) + spacing + len(block) <= per_page:
                    if spacing:
                        current.append("")
                    current.extend(block)
                    continue
                # A separator is cosmetic.  Drop it before adding a page when
                # the complete field itself still fits in the remaining rows.
                if len(current) + len(block) <= per_page:
                    current.extend(block)
                    continue
                pages.append(current)
                current = []
            current.extend(block)
            continue

        # Only an individually overlong field may cross a page boundary.
        if current:
            pages.append(current)
            current = []
        offset = 0
        while len(block) - offset > per_page:
            pages.append(block[offset:offset + per_page])
            offset += per_page
        if offset < len(block):
            current.extend(block[offset:])

    if current:
        pages.append(current)
    return pages or [["无可显示内容"]]


def layout_field_pages(
    fields: Sequence[tuple[str, str]],
    body_font: Any,
    script_font: Any | None = None,
) -> list[list[str]]:
    """Use one compact page when content fits; otherwise expand by field."""
    blocks = layout_field_blocks(fields, body_font, script_font)
    per_page = max(1, (BODY_BOTTOM - BODY_TOP) // LINE_HEIGHT)
    compact_line_count = sum(len(block) for block in blocks)
    return paginate_field_blocks(
        blocks,
        prefer_spacing=compact_line_count > per_page,
    )


def font_ascent(font: Any, fallback: int) -> int:
    getmetrics = getattr(font, "getmetrics", None)
    if callable(getmetrics):
        try:
            ascent, _ = getmetrics()
            if int(ascent) > 0:
                return int(ascent)
        except (OSError, TypeError, ValueError):
            pass
    return fallback


def font_vertical_bounds(
    font: Any,
    text: str,
    *,
    fallback_ascent: int,
) -> tuple[int, int]:
    """Return glyph top/bottom relative to a left-baseline anchor."""
    getbbox = getattr(font, "getbbox", None)
    if callable(getbbox) and text:
        try:
            _, top, _, bottom = getbbox(text, anchor="ls")
            return int(math.floor(top)), int(math.ceil(bottom))
        except (OSError, TypeError, ValueError):
            pass
    getmetrics = getattr(font, "getmetrics", None)
    if callable(getmetrics):
        try:
            ascent, descent = getmetrics()
            return -int(ascent), int(descent)
        except (OSError, TypeError, ValueError):
            pass
    return -fallback_ascent, max(2, round(fallback_ascent * 0.25))


def text_runs_vertical_bounds(
    text: str,
    *,
    body_font: Any,
    script_font: Any,
    fallback_body_font: Any | None = None,
    fallback_script_font: Any | None = None,
) -> tuple[int, int]:
    """Measure rich text runs relative to their common body baseline."""
    body_ascent = font_ascent(body_font, BODY_FONT_SIZE)
    superscript_baseline = -max(4, round(body_ascent * 0.42))
    subscript_baseline = max(3, round(body_ascent * 0.28))
    tops: list[int] = []
    bottoms: list[int] = []
    for style, value, _, _ in formula_runs(text):
        if not value:
            continue
        active_font = body_font if style == "base" else script_font
        active_fallback_font = (
            fallback_body_font if style == "base" else fallback_script_font
        )
        if style == "superscript":
            active_baseline = superscript_baseline
        elif style == "subscript":
            active_baseline = subscript_baseline
        else:
            active_baseline = 0
        for run, run_font in formula_font_runs(
            value,
            active_font,
            active_fallback_font,
        ):
            top, bottom = font_vertical_bounds(
                run_font,
                run,
                fallback_ascent=font_ascent(run_font, body_ascent),
            )
            tops.append(active_baseline + top)
            bottoms.append(active_baseline + bottom)
    if not tops:
        return font_vertical_bounds(
            body_font,
            "Ag",
            fallback_ascent=body_ascent,
        )
    return min(tops), max(bottoms)


def formula_vertical_bounds(
    text: str,
    body_font: Any,
    script_font: Any | None = None,
    *,
    fallback_body_font: Any | None = None,
    fallback_script_font: Any | None = None,
) -> tuple[int, int]:
    """Measure nested formula ink bounds relative to the outer baseline."""
    small_font = script_font_for(body_font, script_font)
    body_ascent = font_ascent(body_font, BODY_FONT_SIZE)
    tops: list[int] = []
    bottoms: list[int] = []
    for kind, first, second in formula_elements(text):
        if kind == "text":
            top, bottom = text_runs_vertical_bounds(
                first,
                body_font=body_font,
                script_font=small_font,
                fallback_body_font=fallback_body_font,
                fallback_script_font=fallback_script_font,
            )
        elif kind == "fraction" and second is not None:
            fraction_font = scaled_font_for(
                body_font,
                FRACTION_FONT_SCALE,
                8,
            )
            fraction_script_font = scaled_font_for(
                fraction_font,
                FRACTION_SCRIPT_SCALE,
                7,
            )
            fraction_fallback_font = (
                scaled_font_for(
                    fallback_body_font,
                    FRACTION_FONT_SCALE,
                    8,
                )
                if fallback_body_font is not None
                else None
            )
            fraction_fallback_script_font = (
                scaled_font_for(
                    fraction_fallback_font,
                    FRACTION_SCRIPT_SCALE,
                    7,
                )
                if fraction_fallback_font is not None
                else None
            )
            numerator_top, numerator_bottom = formula_vertical_bounds(
                first,
                fraction_font,
                fraction_script_font,
                fallback_body_font=fraction_fallback_font,
                fallback_script_font=fraction_fallback_script_font,
            )
            denominator_top, denominator_bottom = formula_vertical_bounds(
                second,
                fraction_font,
                fraction_script_font,
                fallback_body_font=fraction_fallback_font,
                fallback_script_font=fraction_fallback_script_font,
            )
            line_y = -max(5, round(body_ascent * 0.36))
            fraction_ascent = font_ascent(
                fraction_font,
                max(8, round(body_ascent * FRACTION_FONT_SCALE)),
            )
            rule_gap = max(
                FRACTION_RULE_MIN_GAP,
                round(fraction_ascent * FRACTION_RULE_GAP_SCALE),
            )
            numerator_baseline = line_y - rule_gap - numerator_bottom
            denominator_baseline = line_y + rule_gap - denominator_top
            top = min(
                line_y,
                numerator_baseline + numerator_top,
                denominator_baseline + denominator_top,
            )
            bottom = max(
                line_y + 1,
                numerator_baseline + numerator_bottom,
                denominator_baseline + denominator_bottom,
            )
        elif kind == "radical":
            radicand_top, radicand_bottom = formula_vertical_bounds(
                first,
                body_font,
                small_font,
                fallback_body_font=fallback_body_font,
                fallback_script_font=fallback_script_font,
            )
            extra_raise = (
                max(3, round(body_ascent * 0.24))
                if FRACTION_START in first
                else 0
            )
            overbar_y = -body_ascent + 1 - extra_raise
            top = min(overbar_y, radicand_top)
            bottom = max(-2, radicand_bottom)
        elif kind in {"superscript", "subscript"}:
            nested_script_font = scaled_font_for(
                small_font,
                FRACTION_SCRIPT_SCALE,
                7,
            )
            nested_fallback_script_font = (
                scaled_font_for(
                    fallback_script_font,
                    FRACTION_SCRIPT_SCALE,
                    7,
                )
                if fallback_script_font is not None
                else None
            )
            child_top, child_bottom = formula_vertical_bounds(
                first,
                small_font,
                nested_script_font,
                fallback_body_font=fallback_script_font,
                fallback_script_font=nested_fallback_script_font,
            )
            offset = (
                -max(4, round(body_ascent * 0.42))
                if kind == "superscript"
                else max(3, round(body_ascent * 0.28))
            )
            top = offset + child_top
            bottom = offset + child_bottom
        else:
            continue
        tops.append(int(top))
        bottoms.append(int(bottom))
    if not tops:
        return font_vertical_bounds(
            body_font,
            "Ag",
            fallback_ascent=body_ascent,
        )
    return min(tops), max(bottoms)


def draw_text_runs_at_baseline(
    draw: ImageDraw.ImageDraw,
    x: float,
    baseline: int,
    text: str,
    *,
    body_font: Any,
    script_font: Any,
    fallback_body_font: Any | None = None,
    fallback_script_font: Any | None = None,
    fill: int = 0,
) -> float:
    cursor = float(x)
    body_ascent = font_ascent(body_font, BODY_FONT_SIZE)
    superscript_baseline = baseline - max(4, round(body_ascent * 0.42))
    subscript_baseline = baseline + max(3, round(body_ascent * 0.28))

    for style, value, overline, vector in formula_runs(text):
        active_font = body_font if style == "base" else script_font
        active_fallback_font = (
            fallback_body_font if style == "base" else fallback_script_font
        )
        if style == "superscript":
            active_baseline = superscript_baseline
        elif style == "subscript":
            active_baseline = subscript_baseline
        else:
            active_baseline = baseline
        run_start = cursor
        for visible_run, visible_font in formula_font_runs(
            value,
            active_font,
            active_fallback_font,
        ):
            draw.text(
                (cursor, active_baseline),
                visible_run,
                fill=fill,
                font=visible_font,
                anchor="ls",
            )
            cursor += float(
                draw.textlength(
                    visible_run.replace("\r", " ").replace("\n", " "),
                    font=visible_font,
                )
            )
        width = cursor - run_start
        if style == "base" and (overline or vector):
            arrow_y = baseline - body_ascent + 1
            start_x = int(round(run_start))
            end_x = int(
                round(run_start + max(width, 2))
            )
            draw.line(
                (start_x, arrow_y, end_x, arrow_y),
                fill=fill,
                width=1,
            )
            if vector:
                draw.line(
                    (end_x - 3, arrow_y - 2, end_x, arrow_y),
                    fill=fill,
                    width=1,
                )
                draw.line(
                    (end_x - 3, arrow_y + 2, end_x, arrow_y),
                    fill=fill,
                    width=1,
                )
    return cursor


def draw_formula_at_baseline(
    draw: ImageDraw.ImageDraw,
    x: float,
    baseline: int,
    text: str,
    *,
    body_font: Any,
    script_font: Any,
    fallback_body_font: Any | None = None,
    fallback_script_font: Any | None = None,
    fill: int = 0,
) -> float:
    """Draw nested text, fractions, and radicals from one baseline."""
    cursor = float(x)
    body_ascent = font_ascent(body_font, BODY_FONT_SIZE)
    for kind, first, second in formula_elements(text):
        if kind == "text":
            cursor = draw_text_runs_at_baseline(
                draw,
                cursor,
                baseline,
                first,
                body_font=body_font,
                script_font=script_font,
                fallback_body_font=fallback_body_font,
                fallback_script_font=fallback_script_font,
                fill=fill,
            )
            continue
        if kind == "fraction" and second is not None:
            fraction_font = scaled_font_for(
                body_font,
                FRACTION_FONT_SCALE,
                8,
            )
            fraction_script_font = scaled_font_for(
                fraction_font,
                FRACTION_SCRIPT_SCALE,
                7,
            )
            fraction_fallback_font = (
                scaled_font_for(
                    fallback_body_font,
                    FRACTION_FONT_SCALE,
                    8,
                )
                if fallback_body_font is not None
                else None
            )
            fraction_fallback_script_font = (
                scaled_font_for(
                    fraction_fallback_font,
                    FRACTION_SCRIPT_SCALE,
                    7,
                )
                if fraction_fallback_font is not None
                else None
            )
            numerator_width = text_width(
                draw,
                first,
                fraction_font,
                fraction_script_font,
                fallback_body_font=fraction_fallback_font,
                fallback_script_font=fraction_fallback_script_font,
            )
            denominator_width = text_width(
                draw,
                second,
                fraction_font,
                fraction_script_font,
                fallback_body_font=fraction_fallback_font,
                fallback_script_font=fraction_fallback_script_font,
            )
            fraction_width = max(
                numerator_width,
                denominator_width,
            ) + 6
            line_y = baseline - max(
                5,
                round(body_ascent * 0.36),
            )
            fraction_ascent = font_ascent(
                fraction_font,
                max(8, round(body_ascent * FRACTION_FONT_SCALE)),
            )
            numerator_top, numerator_bottom = formula_vertical_bounds(
                first,
                fraction_font,
                fraction_script_font,
                fallback_body_font=fraction_fallback_font,
                fallback_script_font=fraction_fallback_script_font,
            )
            denominator_top, denominator_bottom = formula_vertical_bounds(
                second,
                fraction_font,
                fraction_script_font,
                fallback_body_font=fraction_fallback_font,
                fallback_script_font=fraction_fallback_script_font,
            )
            rule_gap = max(
                FRACTION_RULE_MIN_GAP,
                round(fraction_ascent * FRACTION_RULE_GAP_SCALE),
            )
            numerator_baseline = line_y - rule_gap - numerator_bottom
            denominator_baseline = line_y + rule_gap - denominator_top
            draw_formula_at_baseline(
                draw,
                cursor + (fraction_width - numerator_width) / 2,
                numerator_baseline,
                first,
                body_font=fraction_font,
                script_font=fraction_script_font,
                fallback_body_font=fraction_fallback_font,
                fallback_script_font=fraction_fallback_script_font,
                fill=fill,
            )
            draw.line(
                (
                    int(round(cursor)),
                    line_y,
                    int(round(cursor + fraction_width)),
                    line_y,
                ),
                fill=fill,
                width=1,
            )
            draw_formula_at_baseline(
                draw,
                cursor + (fraction_width - denominator_width) / 2,
                denominator_baseline,
                second,
                body_font=fraction_font,
                script_font=fraction_script_font,
                fallback_body_font=fraction_fallback_font,
                fallback_script_font=fraction_fallback_script_font,
                fill=fill,
            )
            cursor += fraction_width
            continue
        if kind == "radical":
            hook_width = radical_hook_width(body_font)
            radicand_width = text_width(
                draw,
                first,
                body_font,
                script_font,
                fallback_body_font=fallback_body_font,
                fallback_script_font=fallback_script_font,
            )
            extra_raise = (
                max(3, round(body_ascent * 0.24))
                if FRACTION_START in first
                else 0
            )
            overbar_y = baseline - body_ascent + 1 - extra_raise
            hook_x = int(round(cursor))
            shoulder_x = hook_x + hook_width
            draw.line(
                (
                    hook_x,
                    baseline - 4,
                    hook_x + 2,
                    baseline - 2,
                    hook_x + 4,
                    baseline - 7,
                    shoulder_x - 2,
                    overbar_y,
                    shoulder_x,
                    overbar_y,
                ),
                fill=fill,
                width=1,
            )
            draw.line(
                (
                    shoulder_x - 1,
                    overbar_y,
                    int(round(shoulder_x + 2 + radicand_width)),
                    overbar_y,
                ),
                fill=fill,
                width=1,
            )
            draw_formula_at_baseline(
                draw,
                shoulder_x + 2,
                baseline,
                first,
                body_font=body_font,
                script_font=script_font,
                fallback_body_font=fallback_body_font,
                fallback_script_font=fallback_script_font,
                fill=fill,
            )
            cursor += hook_width + 2 + radicand_width
            continue
        if kind in {"superscript", "subscript"}:
            nested_script_font = scaled_font_for(
                script_font,
                FRACTION_SCRIPT_SCALE,
                7,
            )
            nested_fallback_script_font = (
                scaled_font_for(
                    fallback_script_font,
                    FRACTION_SCRIPT_SCALE,
                    7,
                )
                if fallback_script_font is not None
                else None
            )
            script_baseline = baseline + (
                -max(4, round(body_ascent * 0.42))
                if kind == "superscript"
                else max(3, round(body_ascent * 0.28))
            )
            start_x = cursor
            cursor = draw_formula_at_baseline(
                draw,
                cursor,
                script_baseline,
                first,
                body_font=script_font,
                script_font=nested_script_font,
                fallback_body_font=fallback_script_font,
                fallback_script_font=nested_fallback_script_font,
                fill=fill,
            )
            if cursor <= start_x:
                cursor = start_x + 1
    return cursor


def draw_formula_text(
    draw: ImageDraw.ImageDraw,
    position: tuple[float, float],
    text: str,
    *,
    body_font: Any,
    script_font: Any,
    fallback_body_font: Any | None = None,
    fallback_script_font: Any | None = None,
    fill: int = 0,
) -> float:
    """Draw formula rich text without exposing internal tokens to fonts."""
    if re.search(r"\\[A-Za-z]+", text):
        text = clean_text(text)
    x, top = position
    baseline = int(round(top)) + font_ascent(
        body_font,
        BODY_FONT_SIZE,
    )
    return draw_formula_at_baseline(
        draw,
        x,
        baseline,
        text,
        body_font=body_font,
        script_font=script_font,
        fallback_body_font=fallback_body_font,
        fallback_script_font=fallback_script_font,
        fill=fill,
    )


def render_page(
    *,
    title: str,
    question_id: str,
    model_id: str,
    lines: Sequence[str],
    page_number: int,
    page_count: int,
    font_path: Path,
) -> Image.Image:
    image = Image.new("1", (WIDTH, HEIGHT), 1)
    draw = ImageDraw.Draw(image)
    title_font = load_font(str(font_path), TITLE_FONT_SIZE)
    body_font = load_font(str(font_path), BODY_FONT_SIZE)
    script_font = load_font(str(font_path), SCRIPT_FONT_SIZE)
    small_font = load_font(str(font_path), SMALL_FONT_SIZE)

    draw.text((MARGIN_X, 14), title, fill=0, font=title_font)
    model_label = clean_text(model_id)
    if len(model_label) > 42:
        model_label = model_label[:39] + "..."
    draw.text(
        (WIDTH - MARGIN_X, 25),
        model_label,
        fill=0,
        font=small_font,
        anchor="ra",
    )
    draw.line(
        (MARGIN_X, 56, WIDTH - MARGIN_X, 56),
        fill=0,
        width=1,
    )
    y = BODY_TOP
    for line in lines:
        draw_formula_text(
            draw,
            (MARGIN_X, y),
            line,
            body_font=body_font,
            script_font=script_font,
        )
        y += LINE_HEIGHT
    footer = f"{question_id}  ·  {page_number}/{page_count}"
    draw.line(
        (MARGIN_X, 449, WIDTH - MARGIN_X, 449),
        fill=0,
        width=1,
    )
    draw.text(
        (WIDTH // 2, 457),
        footer,
        fill=0,
        font=small_font,
        anchor="ma",
    )
    return image


def render_content_auto_image(
    *,
    title: str,
    question_id: str,
    model_id: str,
    lines: Sequence[str],
    font_path: Path,
) -> Image.Image:
    """Render one 640px-wide monochrome image with content-driven height."""
    line_count = max(1, len(lines))
    height = max(
        AUTO_CONTENT_MIN_HEIGHT,
        BODY_TOP + line_count * LINE_HEIGHT + AUTO_CONTENT_FOOTER_SPACE,
    )
    image = Image.new("1", (WIDTH, height), 1)
    draw = ImageDraw.Draw(image)
    title_font = load_font(str(font_path), TITLE_FONT_SIZE)
    body_font = load_font(str(font_path), BODY_FONT_SIZE)
    script_font = load_font(str(font_path), SCRIPT_FONT_SIZE)
    small_font = load_font(str(font_path), SMALL_FONT_SIZE)
    draw.text((MARGIN_X, 14), title, fill=0, font=title_font)
    model_label = clean_text(model_id)
    if len(model_label) > 42:
        model_label = model_label[:39] + "..."
    draw.text(
        (WIDTH - MARGIN_X, 25),
        model_label,
        fill=0,
        font=small_font,
        anchor="ra",
    )
    draw.line((MARGIN_X, 56, WIDTH - MARGIN_X, 56), fill=0, width=1)
    y = BODY_TOP
    for line in lines:
        draw_formula_text(
            draw,
            (MARGIN_X, y),
            line,
            body_font=body_font,
            script_font=script_font,
        )
        y += LINE_HEIGHT
    footer_line_y = height - 31
    draw.line(
        (MARGIN_X, footer_line_y, WIDTH - MARGIN_X, footer_line_y),
        fill=0,
        width=1,
    )
    draw.text(
        (WIDTH // 2, height - 23),
        question_id,
        fill=0,
        font=small_font,
        anchor="ma",
    )
    return image


def save_png(image: Image.Image, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        image.save(
            temporary,
            format="PNG",
            optimize=True,
            compress_level=9,
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "width": image.width,
        "height": image.height,
        "color_mode": "1-bit_black_white",
    }


class RenderAssetValidationError(ValueError):
    """A stable, user-data validation failure for an optional render asset."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def render_asset_warning(
    code: str,
    message: str,
    asset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "status": "warning",
        "warning": {
            "code": code,
            "message": message,
        },
    }
    if isinstance(asset, Mapping):
        source = asset.get("source")
        if isinstance(source, str):
            metadata["source"] = source
        metadata["verified"] = asset.get("verified") is True
        bbox = asset.get("bbox_xyxy")
        if (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in bbox
            )
        ):
            metadata["bbox_xyxy"] = list(bbox)
    return metadata


def parse_geometry_bbox(value: Any) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 4
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise RenderAssetValidationError(
            "geometry_asset_bbox_invalid",
            "bbox_xyxy must contain four finite numbers",
        )
    left, top, right, bottom = (float(item) for item in value)
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise RenderAssetValidationError(
            "geometry_asset_bbox_invalid",
            "bbox_xyxy must have non-negative origin and positive area",
        )
    return left, top, right, bottom


def resolve_dataset_source_path(
    dataset_root: Path | None,
    relative_path: Any,
    *,
    code_prefix: str,
    purpose: str,
) -> tuple[Path, Path]:
    if dataset_root is None:
        raise RenderAssetValidationError(
            f"{code_prefix}_dataset_root_missing",
            f"run.json paths.dataset_root is required for {purpose}",
        )
    try:
        root = dataset_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RenderAssetValidationError(
            f"{code_prefix}_dataset_root_invalid",
            "run.json paths.dataset_root is unavailable",
        ) from exc
    if not root.is_dir():
        raise RenderAssetValidationError(
            f"{code_prefix}_dataset_root_invalid",
            "run.json paths.dataset_root is not a directory",
        )

    raw = str(relative_path or "").strip()
    if not raw or "\x00" in raw:
        raise RenderAssetValidationError(
            f"{code_prefix}_source_path_missing",
            "record.image.relative_path is required",
        )
    native = Path(raw)
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    normalized_parts = PurePosixPath(raw.replace("\\", "/")).parts
    if (
        native.is_absolute()
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or ".." in normalized_parts
    ):
        raise RenderAssetValidationError(
            f"{code_prefix}_source_path_outside_dataset",
            "record.image.relative_path must stay inside paths.dataset_root",
        )

    try:
        candidate = (root / native).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RenderAssetValidationError(
            f"{code_prefix}_source_file_missing",
            "record.image.relative_path does not identify a readable file",
        ) from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RenderAssetValidationError(
            f"{code_prefix}_source_path_outside_dataset",
            "record.image.relative_path resolves outside paths.dataset_root",
        ) from exc
    if not candidate.is_file():
        raise RenderAssetValidationError(
            f"{code_prefix}_source_file_missing",
            "record.image.relative_path does not identify a regular file",
        )
    return root, candidate


def resolve_geometry_source_path(
    dataset_root: Path | None,
    relative_path: Any,
) -> tuple[Path, Path]:
    return resolve_dataset_source_path(
        dataset_root,
        relative_path,
        code_prefix="geometry",
        purpose="geometry assets",
    )


def resolve_original_question_source_path(
    dataset_root: Path | None,
    relative_path: Any,
) -> tuple[Path, Path]:
    return resolve_dataset_source_path(
        dataset_root,
        relative_path,
        code_prefix="original_question",
        purpose="original question pages",
    )


def grayscale_on_white(image: Image.Image) -> Image.Image:
    if "A" in image.getbands() or "transparency" in image.info:
        foreground = image.convert("RGBA")
        background = Image.new("RGBA", foreground.size, "white")
        background.alpha_composite(foreground)
        return background.convert("L")
    return image.convert("L")


def original_question_border_stats(
    grayscale: Image.Image,
) -> tuple[int, float]:
    """Return the dominant border tone and its tolerant coverage ratio."""
    width, height = grayscale.size
    band = max(1, min(4, width // 2, height // 2))
    pixels = grayscale.load()
    values: list[int] = []
    for y in range(band):
        for x in range(width):
            values.append(int(pixels[x, y]))
            values.append(int(pixels[x, height - 1 - y]))
    for x in range(band):
        for y in range(band, max(band, height - band)):
            values.append(int(pixels[x, y]))
            values.append(int(pixels[width - 1 - x, y]))

    histogram = [0] * 256
    for value in values:
        histogram[value] += 1
    background = max(range(256), key=histogram.__getitem__)
    tolerance = ORIGINAL_QUESTION_BORDER_TOLERANCE
    uniform = sum(
        count
        for value, count in enumerate(histogram)
        if abs(value - background) <= tolerance
    )
    ratio = uniform / len(values) if values else 0.0
    return background, ratio


def original_question_content_layout(
    grayscale: Image.Image,
) -> dict[str, Any]:
    """Plan a conservative document crop or a bounded full-frame fallback."""
    width, height = grayscale.size
    full_box = (0, 0, width, height)
    background, border_uniform_ratio = original_question_border_stats(
        grayscale
    )
    fallback_reason: str | None = None
    content_box = full_box
    strong_mask = grayscale.point(
        [
            255 if value <= ORIGINAL_QUESTION_STRONG_INK_THRESHOLD else 0
            for value in range(256)
        ]
    )
    strong_box = strong_mask.getbbox()
    soft_mask = grayscale.point(
        [
            255 if value <= ORIGINAL_QUESTION_SOFT_INK_THRESHOLD else 0
            for value in range(256)
        ]
    )
    soft_box = soft_mask.getbbox()

    border_is_safe_document = (
        background >= ORIGINAL_QUESTION_LIGHT_BACKGROUND_MIN
        and border_uniform_ratio
        >= ORIGINAL_QUESTION_BORDER_UNIFORM_RATIO
    )

    if not border_is_safe_document:
        fallback_reason = "non_document_border_full_frame"
    elif strong_box is None:
        fallback_reason = (
            "no_foreground"
            if soft_box is None
            else "faint_foreground_full_frame"
        )
    else:
        strong_histogram = strong_mask.histogram()
        strong_ratio = (
            strong_histogram[255] / (width * height)
            if width and height
            else 0.0
        )
        if strong_ratio > 0.65:
            fallback_reason = "photo_like_dense_foreground"
        else:
            detected_box = soft_box or strong_box
            left, top, right, bottom = detected_box
            padding = ORIGINAL_QUESTION_CROP_PADDING
            candidate_box = (
                max(0, left - padding),
                max(0, top - padding),
                min(width, right + padding),
                min(height, bottom + padding),
            )
            candidate_area = (
                (candidate_box[2] - candidate_box[0])
                * (candidate_box[3] - candidate_box[1])
            )
            if candidate_area <= width * height * 0.97:
                content_box = candidate_box

    crop_safe = fallback_reason is None
    if not crop_safe:
        content_box = full_box
    if crop_safe:
        mode = "readable_document_tiles"
    elif fallback_reason == "no_foreground":
        mode = "full_frame_fit"
    else:
        mode = "readable_full_frame_tiles"

    return {
        "mode": mode,
        "fallback_reason": fallback_reason,
        "crop_reason": (
            "foreground_bbox"
            if crop_safe and tuple(content_box) != full_box
            else (
                "foreground_bbox_kept_full_frame"
                if crop_safe
                else fallback_reason
            )
        ),
        "background_luma": background,
        "border_uniform_ratio": round(border_uniform_ratio, 6),
        "content_bbox_xyxy": list(content_box),
        "crop_applied": tuple(content_box) != full_box,
    }


def original_question_axis_scale(
    length: int,
    viewport: int,
    overlap: int,
) -> float:
    if length <= viewport:
        return 1.0
    fit_one = viewport / length
    if fit_one >= ORIGINAL_QUESTION_MIN_READABLE_SCALE:
        return min(1.0, fit_one)
    fit_two = (2 * viewport - overlap) / length
    if fit_two >= ORIGINAL_QUESTION_MIN_READABLE_SCALE:
        return min(1.0, fit_two)
    return ORIGINAL_QUESTION_MIN_READABLE_SCALE


def original_question_tile_offsets(
    length: int,
    viewport: int,
    overlap: int,
) -> list[int]:
    if length <= viewport:
        return [0]
    step = viewport - overlap
    count = 1 + math.ceil((length - viewport) / step)
    span = length - viewport
    return [
        int(round(index * span / (count - 1)))
        for index in range(count)
    ]


def original_question_otsu_threshold(grayscale: Image.Image) -> int:
    """Choose a stable global black/white split for the rendered crop."""
    histogram = grayscale.histogram()
    total = grayscale.width * grayscale.height
    if total <= 0:
        return ORIGINAL_QUESTION_BINARY_THRESHOLD_MIN
    weighted_total = sum(
        value * count
        for value, count in enumerate(histogram)
    )
    background_weight = 0
    background_sum = 0
    best_threshold = 0
    best_variance = -1.0
    for threshold, count in enumerate(histogram):
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (
            weighted_total - background_sum
        ) / foreground_weight
        between_variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = threshold
    return best_threshold


def original_question_binarize(grayscale: Image.Image) -> Image.Image:
    """Convert a scan to clear 1-bit ink without losing faint text.

    A single global threshold performs poorly when a question contains a
    photograph or a grey paper background: the dark picture pulls Otsu's
    threshold down and faint printed strokes disappear.  Comparing each
    pixel with a locally blurred background preserves those strokes, while
    the absolute-ink mask keeps large genuinely black shapes solid.
    """
    if grayscale.mode != "L":
        grayscale = grayscale.convert("L")
    local_background = grayscale.filter(
        ImageFilter.GaussianBlur(
            radius=ORIGINAL_QUESTION_ADAPTIVE_BLUR_RADIUS
        )
    )
    local_contrast = ImageChops.subtract(
        local_background,
        grayscale,
        offset=128,
    )
    local_ink = local_contrast.point(
        [
            0
            if value >= 128 + ORIGINAL_QUESTION_ADAPTIVE_DELTA
            else 255
            for value in range(256)
        ],
        mode="1",
    )
    absolute_ink = grayscale.point(
        [
            0
            if value <= ORIGINAL_QUESTION_ABSOLUTE_INK_THRESHOLD
            else 255
            for value in range(256)
        ],
        mode="1",
    )
    return ImageChops.darker(local_ink, absolute_ink)


def render_original_question_pages(
    source: Image.Image,
    *,
    question_id: str,
    font_path: Path,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """Render readable, overlapping document tiles without losing content."""
    if source.width < 1 or source.height < 1:
        raise RenderAssetValidationError(
            "original_question_source_image_invalid",
            "record.image.relative_path has invalid image dimensions",
        )

    available_width = WIDTH - 2 * ORIGINAL_QUESTION_MARGIN_X
    available_height = ORIGINAL_QUESTION_BOTTOM - ORIGINAL_QUESTION_TOP
    grayscale = grayscale_on_white(source)
    layout = original_question_content_layout(grayscale)
    content_box = tuple(layout["content_bbox_xyxy"])
    cropped = grayscale.crop(content_box)

    if layout["mode"] != "full_frame_fit":
        scale = min(
            original_question_axis_scale(
                cropped.width,
                available_width,
                ORIGINAL_QUESTION_TILE_OVERLAP_X,
            ),
            original_question_axis_scale(
                cropped.height,
                available_height,
                ORIGINAL_QUESTION_TILE_OVERLAP_Y,
            ),
        )
        overlap_x = ORIGINAL_QUESTION_TILE_OVERLAP_X
        overlap_y = ORIGINAL_QUESTION_TILE_OVERLAP_Y
    else:
        scale = min(
            1.0,
            available_width / cropped.width,
            available_height / cropped.height,
        )
        overlap_x = 0
        overlap_y = 0

    target_width = max(1, int(round(cropped.width * scale)))
    target_height = max(1, int(round(cropped.height * scale)))
    resized = cropped.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS,
    )
    otsu_threshold = original_question_otsu_threshold(resized)
    binary_threshold = max(
        ORIGINAL_QUESTION_BINARY_THRESHOLD_MIN,
        min(
            ORIGINAL_QUESTION_BINARY_THRESHOLD_MAX,
            otsu_threshold,
        ),
    )
    monochrome = original_question_binarize(resized)
    x_offsets = original_question_tile_offsets(
        target_width,
        available_width,
        overlap_x,
    )
    y_offsets = original_question_tile_offsets(
        target_height,
        available_height,
        overlap_y,
    )
    page_count = len(x_offsets) * len(y_offsets)
    small_font = load_font(str(font_path), SMALL_FONT_SIZE)
    pages: list[Image.Image] = []
    page_windows: list[dict[str, Any]] = []

    for row, source_top in enumerate(y_offsets):
        for column, source_left in enumerate(x_offsets):
            source_right = min(
                target_width,
                source_left + available_width,
            )
            source_bottom = min(
                target_height,
                source_top + available_height,
            )
            window = (
                source_left,
                source_top,
                source_right,
                source_bottom,
            )
            segment = monochrome.crop(window)
            image = Image.new("1", (WIDTH, HEIGHT), 1)
            x = ORIGINAL_QUESTION_MARGIN_X + (
                available_width - segment.width
            ) // 2
            y = ORIGINAL_QUESTION_TOP + (
                available_height - segment.height
            ) // 2
            image.paste(segment, (x, y))

            page_number = len(pages) + 1
            draw = ImageDraw.Draw(image)
            draw.line(
                (
                    ORIGINAL_QUESTION_MARGIN_X,
                    449,
                    WIDTH - ORIGINAL_QUESTION_MARGIN_X,
                    449,
                ),
                fill=0,
                width=1,
            )
            location = ""
            if len(x_offsets) > 1 or len(y_offsets) > 1:
                location = (
                    f"  ·  行{row + 1}/{len(y_offsets)}"
                    f" 列{column + 1}/{len(x_offsets)}"
                )
            draw.text(
                (WIDTH // 2, 457),
                (
                    f"{question_id}  ·  原题放大  ·  "
                    f"{page_number}/{page_count}{location}"
                ),
                fill=0,
                font=small_font,
                anchor="ma",
            )
            pages.append(image)
            page_windows.append(
                {
                    "page": page_number,
                    "row": row + 1,
                    "column": column + 1,
                    "rendered_bbox_xyxy": list(window),
                    "canvas_bbox_xyxy": [
                        x,
                        y,
                        x + segment.width,
                        y + segment.height,
                    ],
                }
            )

    layout.update(
        {
            "layout_policy": ORIGINAL_QUESTION_LAYOUT_POLICY,
            "layout_version": 3,
            "foreground_threshold": (
                ORIGINAL_QUESTION_STRONG_INK_THRESHOLD
            ),
            "soft_foreground_threshold": (
                ORIGINAL_QUESTION_SOFT_INK_THRESHOLD
            ),
            "crop_padding_px": ORIGINAL_QUESTION_CROP_PADDING,
            "minimum_readable_scale": (
                ORIGINAL_QUESTION_MIN_READABLE_SCALE
            ),
            "scale": round(scale, 6),
            "cropped_width": cropped.width,
            "cropped_height": cropped.height,
            "rendered_width": target_width,
            "rendered_height": target_height,
            "viewport_width": available_width,
            "viewport_height": available_height,
            "tile_width": available_width,
            "tile_height": available_height,
            "tile_overlap_x": overlap_x,
            "tile_overlap_y": overlap_y,
            "columns": len(x_offsets),
            "rows": len(y_offsets),
            "page_count": page_count,
            "page_order": "row_major_top_to_bottom_left_to_right",
            "source_foreground_fully_contained": True,
            "rendered_crop_fully_tiled": True,
            "binary_threshold": binary_threshold,
            "otsu_threshold": otsu_threshold,
            "binarization_policy": (
                ORIGINAL_QUESTION_BINARIZATION_POLICY
            ),
            "adaptive_blur_radius": (
                ORIGINAL_QUESTION_ADAPTIVE_BLUR_RADIUS
            ),
            "adaptive_delta": ORIGINAL_QUESTION_ADAPTIVE_DELTA,
            "absolute_ink_threshold": (
                ORIGINAL_QUESTION_ABSOLUTE_INK_THRESHOLD
            ),
            "dithering": "none",
            "page_windows": page_windows,
        }
    )
    return pages, layout


def original_question_pages(
    record: Mapping[str, Any],
    *,
    question_id: str,
    font_path: Path,
    dataset_root: Path | None,
) -> tuple[list[Image.Image], dict[str, Any] | None]:
    # Older completed runs did not record a dataset root.  They remain
    # publishable with their historical analysis/answer-only output.
    if dataset_root is None:
        return [], None

    image_record = record.get("image")
    relative_path = (
        image_record.get("relative_path")
        if isinstance(image_record, Mapping)
        else None
    )
    try:
        _, source_path = resolve_original_question_source_path(
            dataset_root,
            relative_path,
        )
        try:
            with Image.open(source_path) as source:
                source.load()
                source_width, source_height = source.size
                pages, render_layout = render_original_question_pages(
                    source,
                    question_id=question_id,
                    font_path=font_path,
                )
        except RenderAssetValidationError:
            raise
        except (OSError, ValueError) as exc:
            raise RenderAssetValidationError(
                "original_question_source_image_invalid",
                "record.image.relative_path is not a readable image",
            ) from exc

        return pages, {
            "status": "published",
            "source": ORIGINAL_QUESTION_ASSET_SOURCE,
            "source_relative_path": str(relative_path),
            "source_width": source_width,
            "source_height": source_height,
            "source_sha256": sha256_file(source_path),
            **render_layout,
            "page_count": len(pages),
            "title": ORIGINAL_QUESTION_TITLE,
        }
    except RenderAssetValidationError as exc:
        warning = render_asset_warning(exc.code, str(exc))
        warning["source"] = ORIGINAL_QUESTION_ASSET_SOURCE
        if isinstance(relative_path, (str, os.PathLike)):
            warning["source_relative_path"] = str(relative_path)
        return [], warning


def original_question_auto_image(
    record: Mapping[str, Any],
    *,
    dataset_root: Path | None,
) -> tuple[Image.Image | None, dict[str, Any] | None]:
    """Render all detected question foreground at width 640 in one image.

    Pure blank margins may be removed, but every pixel at or below the soft
    foreground threshold remains inside the padded crop.  This improves e-ink
    readability for scans with large white bands without cutting diagrams,
    formulas, or faint printed text.
    """
    if dataset_root is None:
        return None, None
    image_record = record.get("image")
    relative_path = (
        image_record.get("relative_path")
        if isinstance(image_record, Mapping)
        else None
    )
    try:
        _, source_path = resolve_original_question_source_path(
            dataset_root, relative_path
        )
        try:
            with Image.open(source_path) as source:
                source.load()
                source_width, source_height = source.size
                if source_width < 1 or source_height < 1:
                    raise ValueError("invalid image dimensions")
                grayscale = grayscale_on_white(source)
                layout = original_question_content_layout(grayscale)
                content_box = tuple(layout["content_bbox_xyxy"])
                # Some scanned pages have a dark binding/shadow band touching
                # one border, so the conservative tiled-page planner retains
                # the full frame.  Content-auto output can still safely remove
                # blank bands when the soft foreground bounding box proves
                # that no printable foreground exists in the removed region.
                if not layout["crop_applied"]:
                    soft_mask = grayscale.point(
                        [
                            255
                            if value <= ORIGINAL_QUESTION_SOFT_INK_THRESHOLD
                            else 0
                            for value in range(256)
                        ]
                    )
                    soft_box = soft_mask.getbbox()
                    if soft_box is not None:
                        left, top, right, bottom = soft_box
                        padding = ORIGINAL_QUESTION_CROP_PADDING
                        candidate_box = (
                            max(0, left - padding),
                            max(0, top - padding),
                            min(source_width, right + padding),
                            min(source_height, bottom + padding),
                        )
                        candidate_area = (
                            (candidate_box[2] - candidate_box[0])
                            * (candidate_box[3] - candidate_box[1])
                        )
                        if candidate_area <= source_width * source_height * 0.97:
                            content_box = candidate_box
                            layout = {
                                **layout,
                                "mode": "content_auto_blank_margin_crop",
                                "crop_reason": "soft_foreground_bbox_with_padding",
                                "content_bbox_xyxy": list(candidate_box),
                                "crop_applied": True,
                                "removed_area_foreground_safe": True,
                            }
                cropped = grayscale.crop(content_box)
                target_height = max(
                    1, int(round(cropped.height * WIDTH / cropped.width))
                )
                resized = cropped.resize(
                    (WIDTH, target_height), Image.Resampling.LANCZOS
                )
                monochrome = original_question_binarize(resized)
        except (OSError, ValueError) as exc:
            raise RenderAssetValidationError(
                "original_question_source_image_invalid",
                "record.image.relative_path is not a readable image",
            ) from exc
        return monochrome, {
            "status": "published",
            "source": ORIGINAL_QUESTION_ASSET_SOURCE,
            "source_relative_path": str(relative_path),
            "source_width": source_width,
            "source_height": source_height,
            "source_sha256": sha256_file(source_path),
            "layout_policy": AUTO_HEIGHT_POLICY,
            "source_frame_fully_contained": not layout["crop_applied"],
            "source_foreground_fully_contained": True,
            "crop_applied": layout["crop_applied"],
            "crop_reason": layout["crop_reason"],
            "content_bbox_xyxy": layout["content_bbox_xyxy"],
            "crop_padding_px": ORIGINAL_QUESTION_CROP_PADDING,
            "removed_area_foreground_safe": layout.get(
                "removed_area_foreground_safe", layout["crop_applied"]
            ),
            "rendered_width": WIDTH,
            "rendered_height": target_height,
            "page_count": 1,
            "binarization_policy": ORIGINAL_QUESTION_BINARIZATION_POLICY,
            "title": ORIGINAL_QUESTION_TITLE,
        }
    except RenderAssetValidationError as exc:
        warning = render_asset_warning(exc.code, str(exc))
        warning["source"] = ORIGINAL_QUESTION_ASSET_SOURCE
        if isinstance(relative_path, (str, os.PathLike)):
            warning["source_relative_path"] = str(relative_path)
        return None, warning


def render_geometry_diagram_page(
    crop: Image.Image,
    *,
    question_id: str,
    model_id: str,
    font_path: Path,
) -> Image.Image:
    image = Image.new("1", (WIDTH, HEIGHT), 1)
    draw = ImageDraw.Draw(image)
    title_font = load_font(str(font_path), TITLE_FONT_SIZE)
    small_font = load_font(str(font_path), SMALL_FONT_SIZE)

    draw.text(
        (MARGIN_X, 14),
        GEOMETRY_DIAGRAM_TITLE,
        fill=0,
        font=title_font,
    )
    model_label = clean_text(model_id)
    if len(model_label) > 42:
        model_label = model_label[:39] + "..."
    draw.text(
        (WIDTH - MARGIN_X, 25),
        model_label,
        fill=0,
        font=small_font,
        anchor="ra",
    )
    draw.line(
        (MARGIN_X, 56, WIDTH - MARGIN_X, 56),
        fill=0,
        width=1,
    )

    available_width = WIDTH - 2 * MARGIN_X
    available_height = BODY_BOTTOM - BODY_TOP
    scale = min(
        available_width / crop.width,
        available_height / crop.height,
    )
    target_width = min(
        available_width,
        max(1, int(round(crop.width * scale))),
    )
    target_height = min(
        available_height,
        max(1, int(round(crop.height * scale))),
    )
    resized = grayscale_on_white(crop).resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS,
    )
    monochrome = resized.convert("1", dither=Image.Dither.NONE)
    x = MARGIN_X + (available_width - target_width) // 2
    y = BODY_TOP + (available_height - target_height) // 2
    image.paste(monochrome, (x, y))

    draw.line(
        (MARGIN_X, 449, WIDTH - MARGIN_X, 449),
        fill=0,
        width=1,
    )
    draw.text(
        (WIDTH // 2, 457),
        f"{question_id}  ·  1/1",
        fill=0,
        font=small_font,
        anchor="ma",
    )
    return image


def geometry_diagram_page(
    record: Mapping[str, Any],
    *,
    question_id: str,
    model_id: str,
    font_path: Path,
    dataset_root: Path | None,
) -> tuple[Image.Image | None, dict[str, Any] | None]:
    render_assets = record.get("render_assets")
    if render_assets is None:
        return None, None
    if not isinstance(render_assets, Mapping):
        return None, render_asset_warning(
            "geometry_asset_contract_invalid",
            "record.render_assets must be an object",
        )
    if GEOMETRY_ASSET_NAME not in render_assets:
        return None, None
    raw_asset = render_assets.get(GEOMETRY_ASSET_NAME)
    if not isinstance(raw_asset, Mapping):
        return None, render_asset_warning(
            "geometry_asset_contract_invalid",
            "render_assets.geometry_diagram must be an object",
        )
    asset = dict(raw_asset)

    try:
        if asset.get("source") != GEOMETRY_ASSET_SOURCE:
            raise RenderAssetValidationError(
                "geometry_asset_source_invalid",
                "geometry_diagram.source must be original_question_image",
            )
        if asset.get("verified") is not True:
            raise RenderAssetValidationError(
                "geometry_asset_unverified",
                "geometry_diagram.verified must be true",
            )
        bbox = parse_geometry_bbox(asset.get("bbox_xyxy"))
        image_record = record.get("image")
        if not isinstance(image_record, Mapping):
            raise RenderAssetValidationError(
                "geometry_source_path_missing",
                "record.image.relative_path is required",
            )
        _, source_path = resolve_geometry_source_path(
            dataset_root,
            image_record.get("relative_path"),
        )
        try:
            with Image.open(source_path) as source:
                source.load()
                source_width, source_height = source.size
                left, top, right, bottom = bbox
                if (
                    source_width < 1
                    or source_height < 1
                    or right > source_width
                    or bottom > source_height
                ):
                    raise RenderAssetValidationError(
                        "geometry_asset_bbox_out_of_bounds",
                        "bbox_xyxy exceeds the original question image",
                    )
                pixel_bbox = (
                    math.floor(left),
                    math.floor(top),
                    math.ceil(right),
                    math.ceil(bottom),
                )
                if (
                    pixel_bbox[0] < 0
                    or pixel_bbox[1] < 0
                    or pixel_bbox[2] > source_width
                    or pixel_bbox[3] > source_height
                    or pixel_bbox[2] <= pixel_bbox[0]
                    or pixel_bbox[3] <= pixel_bbox[1]
                ):
                    raise RenderAssetValidationError(
                        "geometry_asset_bbox_out_of_bounds",
                        "bbox_xyxy has no valid pixels in the source image",
                    )
                crop = source.crop(pixel_bbox).copy()
        except RenderAssetValidationError:
            raise
        except (OSError, ValueError) as exc:
            raise RenderAssetValidationError(
                "geometry_source_image_invalid",
                "record.image.relative_path is not a readable image",
            ) from exc

        page = render_geometry_diagram_page(
            crop,
            question_id=question_id,
            model_id=model_id,
            font_path=font_path,
        )
        return page, {
            "status": "published",
            "source": GEOMETRY_ASSET_SOURCE,
            "verified": True,
            "bbox_xyxy": list(asset["bbox_xyxy"]),
            "pixel_bbox_xyxy": list(pixel_bbox),
            "source_relative_path": str(image_record["relative_path"]),
            "source_width": source_width,
            "source_height": source_height,
            "source_sha256": sha256_file(source_path),
            "title": GEOMETRY_DIAGRAM_TITLE,
        }
    except RenderAssetValidationError as exc:
        return None, render_asset_warning(exc.code, str(exc), asset)


def render_record(
    record: Mapping[str, Any],
    model_id: str,
    output_dir: Path,
    font_path: Path,
    dataset_root: Path | None = None,
    page_kinds: Sequence[str] = DEFAULT_PAGE_KINDS,
    height_policy: str = FIXED_HEIGHT_POLICY,
) -> dict[str, Any]:
    question_id = str(record.get("question_id") or "").strip()
    question_file = filename_component(question_id, "question id")
    started = time.perf_counter()
    status = str(record.get("status") or "unknown")
    result = record.get("output_json")
    entry: dict[str, Any] = {
        "question_id": question_id,
        "subject": clean_text(record.get("subject")),
        "source_group": clean_text(record.get("source_group")),
        "source_index": record.get("source_index"),
        "recorded_at": record.get("recorded_at"),
        "status": status,
        "result_sha256": (
            sha256_bytes(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            )
            if isinstance(result, Mapping)
            else None
        ),
        "files": [],
    }
    failed_result = status != "success" or not isinstance(result, Mapping)
    if failed_result:
        error = record.get("error")
        if isinstance(error, Mapping):
            entry["error_type"] = clean_text(error.get("type"))
        # Legacy fixed-height libraries keep their historical behaviour. The
        # formal content-auto contract requires all three display kinds for
        # every denominator item, including failed API or validation tasks.
        # These local placeholders are explicitly labelled and never treated
        # as model answers; the source JSONL remains untouched.
        if height_policy != AUTO_HEIGHT_POLICY:
            entry["render_ms"] = round(
                (time.perf_counter() - started) * 1000.0, 3
            )
            return entry
        error_type = (
            clean_text(error.get("type")) if isinstance(error, Mapping) else ""
        )
        error_message = (
            clean_text(error.get("message")) if isinstance(error, Mapping) else ""
        )
        failure_detail = "；".join(
            value for value in (error_type, error_message) if value
        ) or status
        result = {
            "analysis": {
                "subject": clean_text(record.get("subject")),
                "question_type": "生成链路失败",
                "knowledge_points": [],
                "explanation": f"本题未生成可用解析。失败原因：{failure_detail}",
            },
            "answer": "未生成答案",
        }
        entry["failure_placeholder_rendered"] = True

    body_font = load_font(str(font_path), BODY_FONT_SIZE)
    script_font = load_font(str(font_path), SCRIPT_FONT_SIZE)
    render_assets_metadata: dict[str, Any] = {}
    enabled_page_kinds = set(page_kinds)

    auto_height = height_policy == AUTO_HEIGHT_POLICY
    if height_policy not in {FIXED_HEIGHT_POLICY, AUTO_HEIGHT_POLICY}:
        raise ValueError(f"unsupported result image height policy: {height_policy}")

    if "question" in enabled_page_kinds and auto_height:
        question_image, question_metadata = original_question_auto_image(
            record,
            dataset_root=dataset_root,
        )
        question_pages = [question_image] if question_image is not None else []
    elif "question" in enabled_page_kinds:
        question_pages, question_metadata = original_question_pages(
            record,
            question_id=question_id,
            font_path=font_path,
            dataset_root=dataset_root,
        )
    else:
        question_pages, question_metadata = [], None
    if question_metadata is not None:
        render_assets_metadata[
            ORIGINAL_QUESTION_ASSET_NAME
        ] = question_metadata
    if question_pages:
        question_metadata["files"] = []
        for page_index, image in enumerate(question_pages, 1):
            filename = f"{question_file}__question__{page_index:03d}.png"
            metadata = save_png(image, output_dir / filename)
            metadata.update(
                {
                    "kind": "question",
                    "page": page_index,
                    "title": ORIGINAL_QUESTION_TITLE,
                }
            )
            page_windows = question_metadata.get("page_windows")
            if (
                isinstance(page_windows, list)
                and page_index <= len(page_windows)
                and isinstance(page_windows[page_index - 1], Mapping)
            ):
                page_window = page_windows[page_index - 1]
                for key in (
                    "row",
                    "column",
                    "rendered_bbox_xyxy",
                    "canvas_bbox_xyxy",
                ):
                    if key in page_window:
                        metadata[key] = page_window[key]
            entry["files"].append(metadata)
            question_metadata["files"].append(metadata["path"])

    if "diagram" in enabled_page_kinds:
        diagram, diagram_metadata = geometry_diagram_page(
            record,
            question_id=question_id,
            model_id=model_id,
            font_path=font_path,
            dataset_root=dataset_root,
        )
    else:
        diagram, diagram_metadata = None, None
    if diagram_metadata is not None:
        render_assets_metadata[GEOMETRY_ASSET_NAME] = diagram_metadata
    if diagram is not None:
        filename = f"{question_file}__diagram__001.png"
        metadata = save_png(diagram, output_dir / filename)
        metadata.update(
            {
                "kind": "diagram",
                "page": 1,
                "title": GEOMETRY_DIAGRAM_TITLE,
            }
        )
        entry["files"].append(metadata)
        diagram_metadata["file"] = metadata["path"]

    if render_assets_metadata:
        entry["render_assets"] = render_assets_metadata

    # The answer group is deliberately rendered last.  Device clients can
    # therefore present files in index order without revealing the answer
    # before the question, optional verified diagram, and explanation.
    available_page_groups = {
        "analysis": (
            "题目解析",
            analysis_fields(result, record),
        ),
        "answer": (
            "最终答案",
            answer_fields(result),
        ),
    }
    page_groups = [
        (kind, *available_page_groups[kind])
        for kind in page_kinds
        if kind in available_page_groups
    ]
    for kind, title, fields in page_groups:
        if auto_height:
            pages = [layout_lines(fields, body_font, script_font)]
        else:
            pages = layout_field_pages(fields, body_font, script_font)
        for page_index, lines in enumerate(pages, 1):
            filename = (
                f"{question_file}__{kind}__{page_index:03d}.png"
            )
            if auto_height:
                image = render_content_auto_image(
                    title=title,
                    question_id=question_id,
                    model_id=model_id,
                    lines=lines,
                    font_path=font_path,
                )
            else:
                image = render_page(
                    title=title,
                    question_id=question_id,
                    model_id=model_id,
                    lines=lines,
                    page_number=page_index,
                    page_count=len(pages),
                    font_path=font_path,
                )
            metadata = save_png(image, output_dir / filename)
            metadata.update({"kind": kind, "page": page_index})
            entry["files"].append(metadata)
    entry["render_ms"] = round(
        (time.perf_counter() - started) * 1000.0, 3
    )
    return entry


def ensure_model_destination(root: Path, model_key: str) -> Path:
    safe = safe_component(model_key, "model key")
    root_resolved = root.resolve()
    destination = (root_resolved / safe).resolve()
    if destination.parent != root_resolved:
        raise ValueError("模型图库目录越界")
    return destination


def remove_tree_checked(path: Path, root: Path) -> None:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    if path_resolved.parent != root_resolved:
        raise ValueError(f"拒绝删除图库根目录之外的路径：{path}")
    if path_resolved.exists():
        shutil.rmtree(path_resolved)


def replace_model_directory(
    staged: Path,
    destination: Path,
    root: Path,
) -> None:
    backup = root / (
        f".{destination.name}.backup-{uuid.uuid4().hex}"
    )
    had_previous = destination.exists()
    if had_previous:
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except Exception:
        if had_previous and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        remove_tree_checked(backup, root)


def publish_model(
    *,
    model_key: str,
    records: Mapping[str, Mapping[str, Any]],
    experiment_id: str,
    experiment_dir: Path,
    results_path: Path,
    library_root: Path,
    font_path: Path,
    staging_root: Path,
    dataset_root: Path | None = None,
    page_kinds: Sequence[str] = DEFAULT_PAGE_KINDS,
    height_policy: str = FIXED_HEIGHT_POLICY,
    strict_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    destination = ensure_model_destination(library_root, model_key)
    staged = staging_root / safe_component(model_key, "model key")
    staged.mkdir(parents=True, exist_ok=False)

    model_id = model_key
    entries: dict[str, dict[str, Any]] = {}
    for question_id in sorted(records):
        record = records[question_id]
        _, model_id = record_model(record)
        entries[question_id] = render_record(
            record,
            model_id,
            staged,
            font_path,
            dataset_root=dataset_root,
            page_kinds=page_kinds,
            height_policy=height_policy,
        )

    all_files = [
        file_info
        for entry in entries.values()
        for file_info in entry["files"]
    ]
    strict_validation = (
        validate_strict_model_images(
            entries=entries,
            staged=staged,
            contract=strict_contract,
        )
        if strict_contract is not None
        else None
    )
    index = {
        "schema_version": SCHEMA_VERSION,
        "published_at": utc_now(),
        "experiment_id": experiment_id,
        "source_experiment_dir": str(experiment_dir.resolve()),
        "source_results": str(results_path.resolve()),
        "source_results_sha256": sha256_file(results_path),
        "renderer_script_sha256": sha256_file(Path(__file__).resolve()),
        "model_key": model_key,
        "model_id": model_id,
        "replace_policy": {
            "scope": "one_model_latest_completed_test_only",
            "new_test_replaces_previous_model_library": True,
            "same_question_and_page_overwrites": True,
            "interrupted_test_keeps_previous_library": True,
        },
        "image_spec": {
            "format": "PNG",
            "width": WIDTH,
            "height": (
                "content_auto_unbounded"
                if height_policy == AUTO_HEIGHT_POLICY
                else HEIGHT
            ),
            "colors": 2,
            "mode": "1-bit_black_white",
            "dithering": "none",
            "multiple_pages_when_needed": height_policy != AUTO_HEIGHT_POLICY,
            "height_policy": height_policy,
            "enabled_page_kinds": list(page_kinds),
            "question_pages_enabled": "question" in page_kinds,
            "question_page_policy": ORIGINAL_QUESTION_LAYOUT_POLICY,
            "question_crop_padding_px": ORIGINAL_QUESTION_CROP_PADDING,
            "question_minimum_readable_scale": (
                ORIGINAL_QUESTION_MIN_READABLE_SCALE
            ),
            "question_binarization_policy": (
                ORIGINAL_QUESTION_BINARIZATION_POLICY
            ),
            "question_adaptive_blur_radius": (
                ORIGINAL_QUESTION_ADAPTIVE_BLUR_RADIUS
            ),
            "question_adaptive_delta": (
                ORIGINAL_QUESTION_ADAPTIVE_DELTA
            ),
            "question_tile_overlap_px": {
                "x": ORIGINAL_QUESTION_TILE_OVERLAP_X,
                "y": ORIGINAL_QUESTION_TILE_OVERLAP_Y,
            },
            "page_kinds": list(page_kinds),
            "font": str(font_path),
            "font_sha256": sha256_file(font_path),
            "strict_contract": strict_contract,
        },
        "summary": {
            "question_count": len(entries),
            "success_count": sum(
                entry["status"] == "success"
                for entry in entries.values()
            ),
            "failure_count": sum(
                entry["status"] != "success"
                for entry in entries.values()
            ),
            "render_asset_warning_count": sum(
                any(
                    isinstance(asset, Mapping)
                    and asset.get("status") == "warning"
                    for asset in (
                        entry.get("render_assets", {}).values()
                        if isinstance(entry.get("render_assets"), Mapping)
                        else ()
                    )
                )
                for entry in entries.values()
            ),
            "image_count": len(all_files),
            "total_bytes": sum(
                int(item["bytes"]) for item in all_files
            ),
        },
        "questions": entries,
    }
    if strict_validation is not None:
        index["strict_contract_validation"] = strict_validation
    atomic_write(staged / "index.json", json_bytes(index, pretty=True))
    replace_model_directory(staged, destination, library_root)
    return {
        "model_key": model_key,
        "model_id": model_id,
        "directory": str(destination),
        **index["summary"],
    }


def dataset_root_from_run(
    run: Mapping[str, Any],
    experiment_dir: Path,
) -> Path | None:
    paths = run.get("paths")
    if not isinstance(paths, Mapping):
        return None
    raw = paths.get("dataset_root")
    if not isinstance(raw, (str, os.PathLike)) or not str(raw).strip():
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = experiment_dir / candidate
    return candidate


def page_kinds_from_run(run: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve the configured source-preview and generated-result order."""
    output = run.get("result_image_output")
    if not isinstance(output, Mapping) or "sequence" not in output:
        return DEFAULT_PAGE_KINDS
    raw_sequence = output.get("sequence")
    if not isinstance(raw_sequence, list) or not raw_sequence:
        raise ValueError(
            "result_image_output.sequence must be a non-empty array"
        )
    aliases = {
        "question": "question",
        "diagram": "diagram",
        "diagram_if_verified": "diagram",
        "analysis": "analysis",
        "answer": "answer",
    }
    normalized: list[str] = []
    for raw_kind in raw_sequence:
        if not isinstance(raw_kind, str) or raw_kind not in aliases:
            raise ValueError(
                "unsupported result image page kind: "
                + repr(raw_kind)
            )
        kind = aliases[raw_kind]
        if kind in normalized:
            raise ValueError(
                f"duplicate result image page kind: {kind}"
            )
        normalized.append(kind)

    canonical = list(DEFAULT_PAGE_KINDS)
    positions = [canonical.index(kind) for kind in normalized]
    if positions != sorted(positions):
        raise ValueError(
            "result image pages must follow question, diagram, "
            "analysis, answer order"
        )
    if "answer" in normalized and normalized[-1] != "answer":
        raise ValueError("answer must be the last result image page kind")
    return tuple(normalized)


def height_policy_from_run(run: Mapping[str, Any]) -> str:
    output = run.get("result_image_output")
    if not isinstance(output, Mapping):
        return FIXED_HEIGHT_POLICY
    raw = output.get("height_policy")
    if raw is None:
        return FIXED_HEIGHT_POLICY
    aliases = {
        FIXED_HEIGHT_POLICY: FIXED_HEIGHT_POLICY,
        "fixed_480": FIXED_HEIGHT_POLICY,
        AUTO_HEIGHT_POLICY: AUTO_HEIGHT_POLICY,
        "content_auto": AUTO_HEIGHT_POLICY,
        "unlimited": AUTO_HEIGHT_POLICY,
    }
    if raw not in aliases:
        raise ValueError(f"unsupported result image height policy: {raw!r}")
    return aliases[raw]


def strict_image_contract_from_run(
    run: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the enforced three-image contract for modern benchmark runs.

    Older runs without the explicit ``one_image_per_kind``/
    ``expected_image_count`` markers keep their historical rendering behavior.
    A modern run is fail-closed: a contradictory declaration is rejected before
    any model library is replaced.
    """

    output = run.get("result_image_output")
    if not isinstance(output, Mapping):
        return None
    declares_strict_contract = (
        "one_image_per_kind" in output
        or "expected_image_count" in output
    )
    if not declares_strict_contract:
        return None
    if output.get("one_image_per_kind") is not True:
        raise ValueError(
            "strict result-image output requires one_image_per_kind=true"
        )

    page_kinds = page_kinds_from_run(run)
    if page_kinds != STRICT_THREE_IMAGE_KINDS:
        raise ValueError(
            "strict result-image output requires exactly question, analysis, "
            "answer"
        )
    expected_count = output.get("expected_image_count", len(page_kinds))
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count != len(STRICT_THREE_IMAGE_KINDS)
    ):
        raise ValueError(
            "strict result-image output requires expected_image_count=3"
        )
    width = output.get("width", WIDTH)
    if isinstance(width, bool) or width != WIDTH:
        raise ValueError("strict result-image output requires width=640")
    image_format = str(output.get("format", "PNG")).strip().upper()
    if image_format != "PNG":
        raise ValueError("strict result-image output requires format=PNG")
    colors = output.get("colors", 2)
    if isinstance(colors, bool) or colors != 2:
        raise ValueError("strict result-image output requires colors=2")
    height_policy = height_policy_from_run(run)
    if height_policy != AUTO_HEIGHT_POLICY:
        raise ValueError(
            "strict result-image output requires content-auto unbounded height"
        )
    if output.get("fixed_height") is True:
        raise ValueError(
            "strict result-image output cannot declare fixed_height=true"
        )
    return {
        "page_kinds": list(STRICT_THREE_IMAGE_KINDS),
        "expected_image_count": len(STRICT_THREE_IMAGE_KINDS),
        "width": WIDTH,
        "height": "content_auto_unbounded",
        "format": "PNG",
        "colors": 2,
        "mode": "1-bit_black_white",
        "one_image_per_kind": True,
    }


def validate_strict_run_coverage(
    run: Mapping[str, Any],
    latest: Mapping[str, Mapping[str, Mapping[str, Any]]],
    model_keys: Iterable[str],
) -> None:
    """Require every selected model to cover every planned question exactly."""

    raw_question_ids = run.get("question_ids")
    if not isinstance(raw_question_ids, list) or not raw_question_ids:
        raise ValueError(
            "strict result-image output requires a non-empty run.question_ids"
        )
    question_ids = [str(item).strip() for item in raw_question_ids]
    if not all(question_ids) or len(set(question_ids)) != len(question_ids):
        raise ValueError(
            "strict result-image output requires unique non-empty question_ids"
        )
    declared_count = run.get("sample_count", len(question_ids))
    if (
        isinstance(declared_count, bool)
        or not isinstance(declared_count, int)
        or declared_count != len(question_ids)
    ):
        raise ValueError(
            "run.sample_count does not match strict result-image question_ids"
        )
    expected = set(question_ids)
    for model_key in sorted(set(model_keys)):
        observed = set(latest.get(model_key, {}))
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(
                    f"missing {len(missing)}: " + ", ".join(missing[:10])
                )
            if extra:
                details.append(
                    f"unexpected {len(extra)}: " + ", ".join(extra[:10])
                )
            raise ValueError(
                f"{model_key}: incomplete strict result-image coverage; "
                + "; ".join(details)
            )


def validate_strict_model_images(
    *,
    entries: Mapping[str, Mapping[str, Any]],
    staged: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify actual staged PNGs before the model-scoped atomic switch."""

    expected_kinds = list(contract["page_kinds"])
    expected_count = int(contract["expected_image_count"])
    referenced_paths: set[Path] = set()
    for question_id, entry in entries.items():
        raw_files = entry.get("files")
        if not isinstance(raw_files, list):
            raise ValueError(f"{question_id}: rendered files must be an array")
        if len(raw_files) != expected_count:
            raise ValueError(
                f"{question_id}: expected exactly {expected_count} result PNGs, "
                f"observed {len(raw_files)}"
            )
        kinds = [
            str(item.get("kind") or "")
            if isinstance(item, Mapping) else ""
            for item in raw_files
        ]
        if kinds != expected_kinds:
            raise ValueError(
                f"{question_id}: result image kinds/order must be "
                + ", ".join(expected_kinds)
            )
        for item in raw_files:
            assert isinstance(item, Mapping)
            if item.get("page") != 1:
                raise ValueError(
                    f"{question_id}/{item.get('kind')}: expected exactly one page"
                )
            raw_path = item.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(f"{question_id}: rendered PNG path is missing")
            relative = Path(raw_path)
            if relative.is_absolute() or len(relative.parts) != 1:
                raise ValueError(
                    f"{question_id}: rendered PNG path is not model-local"
                )
            path = staged / relative
            if path.suffix.lower() != ".png" or not path.is_file():
                raise ValueError(f"{question_id}: rendered PNG is missing: {path}")
            with Image.open(path) as rendered:
                actual_format = rendered.format
                actual_mode = rendered.mode
                actual_width, actual_height = rendered.size
                rendered.verify()
            if actual_format != "PNG":
                raise ValueError(f"{question_id}: rendered file is not PNG")
            if actual_mode != "1":
                raise ValueError(
                    f"{question_id}: rendered PNG is not 1-bit black/white"
                )
            if actual_width != WIDTH or actual_height <= 0:
                raise ValueError(
                    f"{question_id}: rendered PNG must be 640px wide with "
                    "positive content-driven height"
                )
            if (
                item.get("width") != actual_width
                or item.get("height") != actual_height
                or item.get("color_mode") != "1-bit_black_white"
            ):
                raise ValueError(
                    f"{question_id}: rendered PNG metadata does not match bytes"
                )
            referenced_paths.add(path.resolve())

    actual_paths = {
        path.resolve() for path in staged.glob("*.png") if path.is_file()
    }
    if actual_paths != referenced_paths:
        raise ValueError(
            "strict model result library contains unindexed or missing PNG files"
        )
    return {
        "status": "passed",
        "question_count": len(entries),
        "image_count": len(referenced_paths),
        "expected_images_per_question": expected_count,
        "page_kinds": expected_kinds,
        "actual_png_verified": True,
    }


def write_library_index(library_root: Path) -> dict[str, Any]:
    models: list[dict[str, Any]] = []
    for child in sorted(library_root.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        index_path = child / "index.json"
        if not index_path.is_file():
            continue
        index = load_json(index_path)
        if not isinstance(index, Mapping):
            continue
        summary = index.get("summary")
        if not isinstance(summary, Mapping):
            summary = {}
        models.append(
            {
                "model_key": index.get("model_key") or child.name,
                "model_id": index.get("model_id") or child.name,
                "experiment_id": index.get("experiment_id"),
                "published_at": index.get("published_at"),
                "directory": child.name,
                "question_count": summary.get("question_count", 0),
                "success_count": summary.get("success_count", 0),
                "failure_count": summary.get("failure_count", 0),
                "image_count": summary.get("image_count", 0),
                "total_bytes": summary.get("total_bytes", 0),
            }
        )
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now(),
        "library_root": str(library_root),
        "model_count": len(models),
        "image_count": sum(int(item["image_count"]) for item in models),
        "total_bytes": sum(int(item["total_bytes"]) for item in models),
        "models": models,
    }
    atomic_write(
        library_root / "library_index.json",
        json_bytes(inventory, pretty=True),
    )
    return inventory


def publish_experiment(
    *,
    experiment_dir: Path,
    library_root: Path,
    selected_model_keys: Sequence[str] | None = None,
    font: str | None = None,
    require_completed: bool = True,
) -> dict[str, Any]:
    experiment_dir = experiment_dir.expanduser().resolve()
    run_path = experiment_dir / "run.json"
    results_path = experiment_dir / "results.jsonl"
    if not run_path.is_file():
        raise FileNotFoundError(f"缺少运行文件：{run_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"缺少逐题结果：{results_path}")
    run = load_json(run_path)
    if not isinstance(run, Mapping):
        raise ValueError("run.json 根值必须是对象")
    state = str(run.get("state") or "")
    if require_completed and state != "completed":
        raise RuntimeError(
            f"只有完整测试可以覆盖结果图库；当前状态为 {state or 'unknown'}"
        )
    experiment_id = safe_component(
        run.get("experiment_id") or experiment_dir.name,
        "experiment id",
    )
    selected = (
        {
            safe_component(item, "model key")
            for item in selected_model_keys
        }
        if selected_model_keys
        else None
    )
    records = load_jsonl(results_path)
    latest = latest_records(records, selected)
    if selected is not None:
        missing = selected - set(latest)
        if missing:
            raise ValueError(
                "结果中缺少指定模型：" + ", ".join(sorted(missing))
            )
    if not latest:
        raise ValueError("没有可发布的模型结果")

    font_path = resolve_font_path(font)
    dataset_root = dataset_root_from_run(run, experiment_dir)
    page_kinds = page_kinds_from_run(run)
    height_policy = height_policy_from_run(run)
    strict_contract = strict_image_contract_from_run(run)
    if strict_contract is not None:
        validate_strict_run_coverage(
            run,
            latest,
            selected if selected is not None else latest.keys(),
        )
    library_root = library_root.expanduser().resolve()
    library_root.mkdir(parents=True, exist_ok=True)
    staging_root = library_root / f".staging-{uuid.uuid4().hex}"
    staging_root.mkdir(parents=False, exist_ok=False)
    model_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for model_key in sorted(latest):
            model_reports.append(
                publish_model(
                    model_key=model_key,
                    records=latest[model_key],
                    experiment_id=experiment_id,
                    experiment_dir=experiment_dir,
                    results_path=results_path,
                    library_root=library_root,
                    font_path=font_path,
                    staging_root=staging_root,
                    dataset_root=dataset_root,
                    page_kinds=page_kinds,
                    height_policy=height_policy,
                    strict_contract=strict_contract,
                )
            )
    finally:
        if staging_root.exists():
            remove_tree_checked(staging_root, library_root)
    report = {
        "schema_version": SCHEMA_VERSION,
        "published_at": utc_now(),
        "experiment_id": experiment_id,
        "library_root": str(library_root),
        "font": str(font_path),
        "model_count": len(model_reports),
        "question_count": sum(
            int(item["question_count"]) for item in model_reports
        ),
        "image_count": sum(
            int(item["image_count"]) for item in model_reports
        ),
        "total_bytes": sum(
            int(item["total_bytes"]) for item in model_reports
        ),
        "publish_ms": round(
            (time.perf_counter() - started) * 1000.0, 3
        ),
        "models": model_reports,
    }
    inventory = write_library_index(library_root)
    report["library_model_count"] = inventory["model_count"]
    report["library_image_count"] = inventory["image_count"]
    report["library_total_bytes"] = inventory["total_bytes"]
    atomic_write(
        library_root / "latest_publish.json",
        json_bytes(report, pretty=True),
    )
    return report


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从已完成评测生成每模型覆盖式黑白PNG结果图库"
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="包含 run.json 和 results.jsonl 的完整测试目录",
    )
    parser.add_argument(
        "--library-root",
        required=True,
        help="固定模型结果图库根目录",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="只发布指定模型 key；默认发布结果中的全部模型",
    )
    parser.add_argument(
        "--font",
        help="支持中文的 TTF/TTC/OTF 字体；也可设置 RESULT_IMAGE_FONT",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="允许调试未完成运行；正式工作流禁止使用",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    report = publish_experiment(
        experiment_dir=Path(args.experiment_dir),
        library_root=Path(args.library_root),
        selected_model_keys=args.models,
        font=args.font,
        require_completed=not args.allow_incomplete,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
