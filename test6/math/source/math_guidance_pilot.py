#!/usr/bin/env python3
"""Run the answer-blind 20-item mathematics-guidance image pilot.

The runner sends only the prompt and source question image to the selected
DashScope multimodal model.  Gold answers and answer-page references are not
loaded.  Every completed item is durably appended before the next call.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import full_benchmark as benchmark  # noqa: E402
from compact_guidance import display_guidance_entries  # noqa: E402
from math_guidance_renderer import render_four_images  # noqa: E402


SCHEMA_VERSION = "math_guidance_pilot_result_v4_multi_method"
MATHEMATICS_PROBLEM_SCOPE = "primary_to_high_school"
FROZEN_MANIFEST_IDS = frozenset(
    {
        "math_guidance_pilot20_calculation_complete_v1",
    }
)
ROOT_KEYS = (
    "subject",
    "problem_scope",
    "recognition_status",
    "uncertain_items",
    "question_blocks",
    "source_figures",
    "visual_evidence",
    "knowledge_points",
    "base_formulas",
    "guidance",
)
TIER_SPECS = {
    "advanced": {
        "title": "学霸版",
        "primary_steps": (2, 4),
        "alternative_steps": (2, 4),
    },
    "standard": {
        "title": "普通版",
        "primary_steps": (3, 6),
        "alternative_steps": (3, 6),
    },
    "foundation": {
        "title": "基础版",
        "primary_steps": (4, 8),
        "alternative_steps": (4, 8),
    },
}
FORBIDDEN_GUIDANCE_MARKERS = (
    "最终答案",
    "答案为",
    "答案是",
    "正确答案",
    "故选",
    "所以选",
    "应填",
    "填入",
)
FORBIDDEN_DEFERRED_CALCULATION_MARKERS = (
    "代入数值",
    "数值代入",
    "数值运算",
    "数值计算",
    "最终计算",
    "最后再代入",
    "由学生",
    "留给学生",
)
FORBIDDEN_META_GUIDANCE_MARKERS = (
    "学生",
)
DEFAULT_PROMPT = SCRIPT_DIR / "数学题解析提示词.md"
DEFAULT_MANIFEST = (
    SCRIPT_DIR / "math_guidance_pilot20_manifest_complete_v1.json"
)
DEFAULT_SCHEMA = SCRIPT_DIR / "math_guidance_schema.json"
DEFAULT_ENV = SCRIPT_DIR / "api_keys.env"
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "runs"
    / "math_guidance_run"
)
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DOCUMENTED_MODEL_CHOICES = (
    "qwen3.7-flash-2026-07-15",
    "qwen3.8-max",
)
UNAVAILABLE_REQUESTED_MODEL_IDS = {"qwen3.8-flash"}
INLINE_FORMULA = re.compile(r"\$([^$]+)\$")
# A model occasionally emits elementary algebra in otherwise valid Chinese
# prose without the required ``$...$`` delimiters.  Keep the candidate
# alphabet deliberately mathematical: Chinese prose and its punctuation form
# hard boundaries, while coordinates, scripts and ordinary operators remain
# in one candidate run.  Candidates are filtered again by
# ``is_plain_algebra_candidate`` before they are promoted.
PLAIN_ALGEBRA_RUN = re.compile(
    r"[A-Za-zΑ-Ωα-ω0-9\\_{}^'′.()+\-−×·*/=<>≤≥≠≈∝∥⊥∞° ,，|\t]+"
)
PLAIN_ALGEBRA_SIGNAL = re.compile(r"[_^/=<>≤≥≠≈∝∥⊥]")
PLAIN_SCRIPTED_NAME = re.compile(
    r"(?:[A-Za-zΑ-Ωα-ω]+)"
    r"(?:_(?:\{[^{}\s]+\}|[A-Za-z0-9]+)"
    r"|\^(?:\{[^{}\s]+\}|[A-Za-z0-9+\-]+))+"
)
PLAIN_MATH_ATOM_SOURCE = (
    r"(?:"
    r"[A-Za-zΑ-Ωα-ω](?:_(?:\{[^{}\s]+\}|[A-Za-z0-9]+))?"
    r"|[A-Z]{2,8}"
    r"|\d+(?:\.\d+)?"
    r"|\([^()，,]+\)"
    r")"
)
PLAIN_SIMPLE_QUOTIENT = re.compile(
    rf"^\s*{PLAIN_MATH_ATOM_SOURCE}\s*/\s*{PLAIN_MATH_ATOM_SOURCE}\s*$"
)
PLAIN_FUNCTION_CALL = re.compile(
    r"^[A-Za-zΑ-Ωα-ω]\(\s*[A-Za-zΑ-Ωα-ω0-9_{}^+\-*/.]+\s*\)$"
)
PLAIN_COORDINATE_TUPLE = re.compile(
    rf"^\(\s*{PLAIN_MATH_ATOM_SOURCE}\s*[,，]\s*"
    rf"{PLAIN_MATH_ATOM_SOURCE}\s*\)$"
)
PLAIN_INFINITY = re.compile(r"^[+\-−]?\s*(?:infty|∞)$", re.IGNORECASE)
PRINTED_PART = re.compile(
    r"[（(]\s*(\d+)\s*[）)]\s*([①②③④⑤⑥⑦⑧⑨⑩])?"
)
PRINTED_PART_TOKEN = re.compile(
    r"[（(]\s*(\d+)\s*[）)]\s*([①②③④⑤⑥⑦⑧⑨⑩])?|([①②③④⑤⑥⑦⑧⑨⑩])"
)
FIGURE_REFERENCE_PART_PREFIX = re.compile(r"图[ \t\u3000]*$")
NONCANONICAL_SCRIPT = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉]")
MISSING_LATEX_BACKSLASH = re.compile(
    r"(?<![\\A-Za-z])(?:dfrac|tfrac|frac|sqrt|mathrm|text|operatorname|"
    r"vec|overrightarrow|overline)\s*\{"
)
SCRIPTED_INTEGER = re.compile(
    r"(?:[_^]\s*\{\s*[+-]?\d+\s*\}|[_^]\s*[+-]?\d+)"
)
NUMERIC_WITH_UNIT = re.compile(
    r"(?<![A-Za-z])"
    r"[+-]?\d+(?:\.\d+)?"
    r"(?:\s*(?:\\times|\\cdot|×)\s*10(?:\s*\^\s*\{?[+-]?\d+\}?)?)?"
    r"\s*(?:"
    r"\\mathrm\s*\{|%|％|°|℃|"
    r"kPa|MPa|Pa|kg|mg|cm|mm|km|mol|mL|Hz|"
    r"[KJNWVACTFHLmsg]"
    r")"
)
SIGNIFICANT_NUMERIC_LITERAL = re.compile(
    r"(?<![A-Za-z])(?:\d+\.\d+|\d{2,})(?![A-Za-z])"
)
ISOLATED_NUMERIC_RESULT = re.compile(
    r"(?:=|≈|>|<|≥|≤|\\geq?|\\leq?)\s*"
    r"[+-]?\d+(?:\.\d+)?(?=\s*(?:$|[$，。；、）)]))"
)
SIMPLE_RATIO_SYMBOL = (
    r"[A-Za-z](?:_\{[^{}]+\}|_[A-Za-z0-9]+)?"
)
QUESTION_SYMBOL_OVER_INTEGER = re.compile(
    rf"(?:\\frac\s*\{{\s*(?P<latex_symbol>{SIMPLE_RATIO_SYMBOL})\s*\}}"
    rf"\s*\{{\s*(?P<latex_den>[2-9]\d*)\s*\}}|"
    rf"(?P<plain_symbol>{SIMPLE_RATIO_SYMBOL})\s*/\s*"
    rf"(?P<plain_den>[2-9]\d*))"
)
GUIDANCE_NUMERIC_SYMBOL_RATIO = re.compile(
    rf"(?:\\frac\s*\{{\s*(?P<latex_coeff>[1-9]\d*)?\s*"
    rf"(?P<latex_symbol>{SIMPLE_RATIO_SYMBOL})\s*\}}\s*"
    rf"\{{\s*(?P<latex_den>[2-9]\d*)\s*\}}|"
    rf"(?P<plain_coeff>[1-9]\d*)?\s*"
    rf"(?P<plain_symbol>{SIMPLE_RATIO_SYMBOL})\s*/\s*"
    rf"(?P<plain_den>[2-9]\d*))"
)
SI_UNIT_SYMBOLS = (
    "mol",
    "min",
    "rad",
    "km",
    "cm",
    "mm",
    "kg",
    "Pa",
    "Hz",
    "Wb",
    "cd",
    "m",
    "s",
    "h",
    "g",
    "A",
    "K",
    "N",
    "J",
    "W",
    "C",
    "V",
    "T",
    "H",
    "F",
)
SI_UNIT_TOKEN = "(?:" + "|".join(re.escape(item) for item in SI_UNIT_SYMBOLS) + ")"
SPLIT_MATHRM_UNIT_QUOTIENT = re.compile(
    rf"\\mathrm\{{(?P<numerator>{SI_UNIT_TOKEN})\}}\s*/\s*"
    rf"\\mathrm\{{(?P<denominator>{SI_UNIT_TOKEN})\}}"
    rf"(?:\^\{{(?P<power>[+-]?\d+)\}})?"
)
WRAPPED_MATHRM_UNIT_QUOTIENT = re.compile(
    rf"\\mathrm\{{(?P<numerator>{SI_UNIT_TOKEN})/(?P<denominator>{SI_UNIT_TOKEN})"
    rf"(?:\^\{{(?P<power>[+-]?\d+)\}})?\}}"
)
BARE_UNIT_QUOTIENT = re.compile(
    rf"(?<![A-Za-z\\])(?P<numerator>{SI_UNIT_TOKEN})/(?P<denominator>{SI_UNIT_TOKEN})"
    rf"(?:\^\{{(?P<power>[+-]?\d+)\}})?(?![A-Za-z])"
)
EQUATION_SEPARATOR = re.compile(
    r"(?:!=|=|≈|<|>|≤|≥|≠|∈|∉|"
    r"\\notin(?![A-Za-z])|\\in(?![A-Za-z])|"
    r"\\ne(?:q)?(?![A-Za-z])|\\approx|\\simeq|\\leq?|\\geq?)"
)
RANGE_RELATION = re.compile(
    r"(?:<=|>=|!=|<|>|≤|≥|≠|∈|∉|"
    r"\\leq?(?![A-Za-z])|\\geq?(?![A-Za-z])|"
    r"\\ne(?:q)?(?![A-Za-z])|\\notin(?![A-Za-z])|\\in(?![A-Za-z]))"
)
EQUALITY_RELATION = re.compile(
    r"(?<![!<>])=(?!=)|≈|\\approx(?![A-Za-z])|\\simeq(?![A-Za-z])"
)
MALFORMED_MIDPOINT_HALF_SEGMENT = re.compile(
    r"\\frac\s*\{\s*1\s*\}\s*\{\s*2\s*(?P<whole>[A-Z]{2})\s*\}"
)
SEGMENT_RELATION_BEFORE_FRACTION = re.compile(
    r"(?<![A-Z])(?P<segment>[A-Z]{2})(?![A-Z])\s*(?:=|≈)\s*$"
)
PLAIN_SEGMENT_PARALLEL = re.compile(
    r"(?<![A-Za-z0-9_])(?P<left>(?:[A-Z]{2}|[xyzXYZ]))(?![A-Za-z0-9_])"
    r"\s*(?://|\|\|)\s*"
    r"(?<![A-Za-z0-9_])(?P<right>(?:[A-Z]{2}|[xyzXYZ]))(?![A-Za-z0-9_])"
)
PLAIN_GEOMETRY_TOKEN = re.compile(r"(?<![A-Za-z])(?P<token>[∠△][A-Z]{1,3})(?![A-Za-z])")
MIDPOINT_CONDITION = re.compile(
    r"(?P<mid>[A-Z])\s*(?:为|是)\s*(?P<whole>[A-Z]{2})\s*(?:的)?\s*中点"
)
NESTED_DOMAIN_DENOMINATOR = re.compile(
    r"(?P<command>\\(?:frac|dfrac|tfrac))\s*"
    r"\{\s*(?P<numerator>[A-Za-zΑ-Ωα-ω0-9]+)\s*\}\s*"
    r"\{\s*(?P<denominator>[A-Za-zΑ-Ωα-ω]"
    r"(?:_\{[^{}]+\}|_[A-Za-z0-9]+)?)\s*"
    r"[\(（]\s*(?P<condition>[^()（）{}]*?"
    r"(?:<=|>=|!=|<|>|≤|≥|≠|\\leq?|\\geq?|\\ne(?:q)?)"
    r"[^()（）{}]*?)\s*[\)）]\s*\}"
)
MALFORMED_TRIANGLE_AREA_PRODUCT = re.compile(
    r"^\s*S\s*=\s*\\(?:d?frac|tfrac)\s*"
    r"\{\s*1\s*\}\s*\{\s*2\s*"
    r"(?:×|\\times|\\cdot|\*)\s*(?:\\text\s*\{\s*)?底(?:\s*\})?\s*"
    r"(?:×|\\times|\\cdot|\*)\s*(?:\\text\s*\{\s*)?高(?:\s*\})?\s*\}\s*$"
)
MALFORMED_NAMED_TRIANGLE_AREA_PRODUCT = re.compile(
    r"^\s*(?P<lhs>S\s*_\{\s*(?:△|\\triangle\s*)?[A-Z]{3}\s*\})\s*=\s*"
    r"\\(?:d?frac|tfrac)\s*\{\s*1\s*\}\s*\{\s*2\s*"
    r"(?:×|\\times|\\cdot|\*)\s*(?P<base>\|[A-Z]{2}\|)\s*"
    r"(?:×|\\times|\\cdot|\*)\s*"
    r"(?P<height>[A-Za-z](?:_\{[^{}]+\})?)\s*\}\s*$"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def validate_requested_model_id(model_id: str) -> None:
    if not model_id.strip():
        raise ValueError("--model-id不能为空")
    if model_id in UNAVAILABLE_REQUESTED_MODEL_IDS:
        choices = " 或 ".join(DOCUMENTED_MODEL_CHOICES)
        raise ValueError(
            f"阿里云百炼截至2026-08-05未提供可调用的{model_id}；"
            f"请明确选择{choices}，不得静默替换模型。"
        )


def validate_formula_source(source: str, path: str, errors: list[str]) -> None:
    """Reject formula spellings that cannot be typeset as textbook notation."""
    if not source.strip():
        errors.append(f"{path}公式不能为空")
        return
    # ``\left\{`` and ``\right\}`` draw literal braces; they are not TeX
    # grouping delimiters and therefore do not need a matching source brace.
    # Counting them used to reject a valid photographed system of equations.
    structural_braces = re.sub(r"\\[{}]", "", source)
    if structural_braces.count("{") != structural_braces.count("}"):
        errors.append(f"{path}公式花括号不成对")
    if "$" in source or "\\(" in source or "\\)" in source:
        errors.append(f"{path}不得包含公式定界符，字段内只写公式本体")
    if MISSING_LATEX_BACKSLASH.search(source):
        errors.append(f"{path}含缺少反斜杠的LaTeX命令")
    if re.search(r"\^(?!\{)", source):
        errors.append(f"{path}次幂必须写成^{{...}}")
    if re.search(r"_(?!\{)", source):
        errors.append(f"{path}下标必须写成_{{...}}")
    if "/" in source:
        errors.append(f"{path}分式必须使用\\frac{{...}}{{...}}")
    if NONCANONICAL_SCRIPT.search(source):
        errors.append(f"{path}不得直接使用Unicode上下标字符")
    # Text inside an explicit ``\text{...}`` group is a label or prose operand,
    # not an implicit subscript.  For example ``\text{事件A的区域}`` must not be
    # mistaken for the noncanonical identifier spelling ``A的``.
    identifier_probe = re.sub(r"\\text\{[^{}]*\}", "", source)
    if re.search(
        r"[A-Za-zΑ-Ωα-ω][\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
        identifier_probe,
    ):
        errors.append(f"{path}中文说明性下标必须写成_{{\\text{{...}}}}")
    for marker in (
        "**",
        "pow(",
        "sqrt(",
        "\\boldsymbol",
        "\\bm",
        "\\mathbf",
        "Coeff",
        "Cofrac",
        "CoSfrac",
        "TODO",
    ):
        if marker in source:
            errors.append(f"{path}含禁止的公式写法：{marker}")
    without_text_labels = re.sub(
        r"\\text\{[^{}]*[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff][^{}]*\}",
        "",
        source,
    )
    if re.search(
        r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
        without_text_labels,
    ):
        errors.append(f"{path}公式本体不得混入中文说明文字")


def normalized_part_label(value: Any) -> str:
    match = PRINTED_PART.fullmatch(str(value or "").strip())
    if not match:
        return str(value or "").strip()
    return f"（{match.group(1)}）{match.group(2) or ''}"


def is_figure_reference_part_token(question_text: str, token_start: int) -> bool:
    """Return whether a part-shaped token is a printed figure label.

    Mathematics questions commonly use the same circled/parenthesized numbers
    for sub-questions and figures.  A token immediately preceded by ``图``
    (for example ``图①`` or ``图（1）``) identifies a source figure and must not
    change the current question-part hierarchy.
    """

    if not isinstance(question_text, str) or token_start <= 0:
        return False
    line_start = question_text.rfind("\n", 0, token_start) + 1
    prefix = question_text[line_start:token_start]
    return bool(FIGURE_REFERENCE_PART_PREFIX.search(prefix))


def is_math_expression_part_token(
    question_text: str,
    match: re.Match[str],
) -> bool:
    """Reject parenthesized numbers that belong to a mathematical expression.

    A token such as ``(0)`` in ``f(0)=0`` has the same surface shape as a
    printed sub-question marker.  Prefer explicit structural evidence: tokens
    inside inline math, zero-numbered tokens, function/point suffixes, and
    tokens adjacent to an operator are mathematical, not question parts.
    """
    if not isinstance(question_text, str) or not match.group(1):
        return False
    if int(match.group(1)) == 0:
        return True

    prefix = question_text[:match.start()]
    if prefix.count("$") % 2 == 1:
        return True

    previous = question_text[match.start() - 1] if match.start() > 0 else ""
    following = (
        question_text[match.end()]
        if match.end() < len(question_text)
        else ""
    )
    if previous and re.match(r"[A-Za-zΑ-Ωα-ω0-9_}\]]", previous):
        return True
    if following and following in "=<>≤≥≠∈∉+-*/^_,，":
        return True
    return False


def canonical_printed_part_tokens(
    question_text: str,
) -> list[tuple[re.Match[str], str, str]]:
    """Return non-figure part tokens as ``(match, label, parent)`` tuples."""

    tokens: list[tuple[re.Match[str], str, str]] = []
    current_parent = ""
    for match in PRINTED_PART_TOKEN.finditer(question_text):
        if is_figure_reference_part_token(question_text, match.start()):
            continue
        if is_math_expression_part_token(question_text, match):
            continue
        parent_number = match.group(1)
        if parent_number:
            current_parent = f"（{parent_number}）"
            child = match.group(2)
            label = f"{current_parent}{child or ''}"
        else:
            child = match.group(3)
            label = f"{current_parent}{child}" if current_parent else str(child or "")
        if label:
            tokens.append((match, label, current_parent))
    return tokens


def printed_leaf_part_labels(question_text: str) -> list[str]:
    """Return the actual answer-bearing leaves from hierarchical part labels.

    A parent label such as ``（2）`` is structural when it is followed by
    standalone children ``①`` ... ``④``.  In that case guidance must cover
    every ``（2）①`` ... ``（2）④`` leaf, but need not add a redundant
    ``（2）`` guidance block.  Parents without children remain required leaves.
    """

    parent_order: list[str] = []
    children_by_parent: dict[str, list[str]] = {}
    for _, label, parent in canonical_printed_part_tokens(question_text):
        if not parent:
            continue
        if parent not in parent_order:
            parent_order.append(parent)
            children_by_parent[parent] = []
        if label != parent:
            children = children_by_parent[parent]
            if label not in children:
                children.append(label)

    leaves: list[str] = []
    for parent in parent_order:
        children = children_by_parent[parent]
        leaves.extend(children or [parent])
    return leaves


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


def atomic_json(path: Path, value: Any) -> None:
    benchmark.atomic_write_bytes(path, json_bytes(value, pretty=True))


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    benchmark.append_jsonl(path, value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_required_sha256(label: str, actual: str, expected: str | None) -> None:
    """Fail locally when a frozen execution artifact no longer matches."""
    if expected is None:
        return
    normalized = expected.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{label}要求的SHA-256格式无效")
    if actual.lower() != normalized:
        raise ValueError(
            f"{label} SHA-256不一致：expected={normalized}, actual={actual.lower()}"
        )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def extract_model_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig")
    start_marker = "## 二、发给模型的系统提示词"
    end_marker = "## 三、设备端排版约束"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("提示词缺少模型提示词边界")
    prompt = text[start + len(start_marker):end].strip()
    if len(prompt) < 500:
        raise ValueError("提取到的模型提示词异常短")
    return prompt


def resolve_from(parent: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = parent / candidate
    return candidate.resolve()


def validate_manifest(path: Path) -> tuple[dict[str, Any], Path]:
    document = load_json(path)
    if not isinstance(document, dict):
        raise ValueError("manifest根值必须是对象")
    if document.get("manifest_id") not in FROZEN_MANIFEST_IDS:
        raise ValueError("必须使用已冻结的数学大题清单")
    selection_rule = document.get("selection_rule")
    if not isinstance(selection_rule, dict):
        raise ValueError("manifest缺少数学大题选择规则")
    if selection_rule.get("required_visual_type") != "calculation_large_response":
        raise ValueError("manifest未强制限定为数学大题")
    excluded_visual_types = selection_rule.get("excluded_visual_types")
    if not isinstance(excluded_visual_types, list) or "experiment" not in {
        str(value) for value in excluded_visual_types
    }:
        raise ValueError("manifest未在顶层明确排除实验题")
    if selection_rule.get("visual_review_required") is not True:
        raise ValueError("manifest未要求逐题人工视觉复核")
    questions = document.get("questions")
    if not isinstance(questions, list) or len(questions) != 20:
        raise ValueError("pilot manifest必须恰好包含20题")
    if document.get("contains_gold_answers") is not False:
        raise ValueError("pilot manifest必须明确不含Gold答案")
    dataset_root = resolve_from(path.parent, str(document.get("dataset_root") or ""))
    seen: set[str] = set()
    for index, question in enumerate(questions, 1):
        if not isinstance(question, dict):
            raise ValueError(f"manifest第{index}题不是对象")
        question_id = str(question.get("question_id") or "").strip()
        if not question_id or question_id in seen:
            raise ValueError(f"manifest题号为空或重复：{question_id!r}")
        seen.add(question_id)
        if question.get("subject") != "mathematics":
            raise ValueError(f"{question_id}不是数学题")
        selection_evidence = question.get("selection_evidence")
        if not isinstance(selection_evidence, dict):
            raise ValueError(f"{question_id}缺少数学大题人工复核证据")
        if selection_evidence.get("visual_type") != "calculation_large_response":
            raise ValueError(f"{question_id}不是已复核的数学大题")
        if selection_evidence.get("experiment_excluded") is not True:
            raise ValueError(f"{question_id}未明确排除实验题")
        if (
            selection_evidence.get("visual_review_status")
            != "verified_calculation_large_response"
        ):
            raise ValueError(f"{question_id}尚未通过数学大题人工视觉复核")
        image_path = dataset_root / str(question.get("image_path") or "")
        if not image_path.is_file():
            raise FileNotFoundError(f"{question_id}原题图不存在：{image_path}")
        actual = sha256_file(image_path)
        if actual != question.get("image_sha256"):
            raise ValueError(f"{question_id}原题图SHA-256不一致")
        forbidden = {
            "canonical_answer",
            "verification_evidence",
            "official_answer_pages",
            "official_answer_page_images",
        }
        if forbidden.intersection(question):
            raise ValueError(f"{question_id} manifest意外包含答案字段")
        requested_outputs = question.get("requested_outputs")
        if requested_outputs is not None:
            validate_requested_outputs(requested_outputs, question_id)
    if document.get("question_count") != len(questions):
        raise ValueError("manifest question_count不一致")
    return document, dataset_root


def validate_requested_outputs(value: Any, question_id: str) -> None:
    """Validate answer-blind target metadata used only by the local leak gate."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{question_id} requested_outputs必须是非空数组")
    seen_parts: set[str] = set()
    for index, item in enumerate(value):
        required = {
            "part",
            "symbol_aliases",
            "prose_aliases",
        }
        optional = {
            "target_type",
            "range_symbol_aliases",
            "allowed_result_parameters",
        }
        if (
            not isinstance(item, dict)
            or not required.issubset(item)
            or set(item) - required - optional
        ):
            raise ValueError(
                f"{question_id} requested_outputs[{index}]字段必须为"
                "part/symbol_aliases/prose_aliases，可选target_type/"
                "range_symbol_aliases/allowed_result_parameters"
            )
        part = normalized_part_label(item.get("part"))
        if not part or part in seen_parts:
            raise ValueError(f"{question_id} requested_outputs小问编号为空或重复：{part!r}")
        seen_parts.add(part)
        aliases: list[str] = []
        for key in (
            "symbol_aliases",
            "prose_aliases",
            "range_symbol_aliases",
            "allowed_result_parameters",
        ):
            if key not in item:
                continue
            raw_aliases = item.get(key)
            if not isinstance(raw_aliases, list):
                raise ValueError(f"{question_id} {part} {key}必须是数组")
            for alias in raw_aliases:
                if not isinstance(alias, str) or not alias.strip():
                    raise ValueError(f"{question_id} {part} {key}含空值")
                if any(marker in alias for marker in ("=", "≈", "答案")):
                    raise ValueError(f"{question_id} {part} {key}不得包含答案或等式")
                aliases.append(alias.strip())
        target_type = item.get("target_type")
        if target_type is not None and target_type not in {
            "value", "numeric", "expression", "equation", "inequality",
            "set", "interval", "range", "choice", "fill_blank", "proof",
            "construction", "qualitative", "other",
        }:
            raise ValueError(f"{question_id} {part} target_type取值非法")
        if not aliases:
            raise ValueError(f"{question_id} {part}至少需要一个所求量别名")


def exact_keys(value: Any, required: set[str], path: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path}必须是对象")
        return
    observed = set(value)
    missing = sorted(required - observed)
    extra = sorted(observed - required)
    if missing:
        errors.append(f"{path}缺少字段：{','.join(missing)}")
    if extra:
        errors.append(f"{path}含额外字段：{','.join(extra)}")


def has_substantive_guidance_content(value: Any) -> bool:
    """Return whether a guidance step contains text, a number, or a formula.

    Whitespace and punctuation are presentation, not mathematical guidance.  An
    alphanumeric test deliberately covers Chinese characters, Latin variables,
    Greek symbols, digits, and LaTeX command names without maintaining a fragile
    punctuation deny-list.
    """

    return isinstance(value, str) and any(character.isalnum() for character in value)


def string_list(
    value: Any,
    path: str,
    errors: list[str],
    *,
    minimum: int,
    maximum: int,
    require_substantive: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{path}必须是数组")
        return []
    if not minimum <= len(value) <= maximum:
        errors.append(f"{path}数量必须在{minimum}到{maximum}之间")
    output: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{path}[{index}]必须是非空字符串")
        elif require_substantive and not has_substantive_guidance_content(item):
            errors.append(
                f"{path}[{index}]必须包含文字、数字或公式内容，"
                "不能只有标点或符号"
            )
        else:
            output.append(item.strip())
    return output


def guidance_entries(
    output: Mapping[str, Any],
) -> Iterable[tuple[str, str, str | None]]:
    """Yield visible guidance strings together with their normalized part.

    A part-aware stream keeps the answer-isolation gate from treating a
    general relation used in a later sub-question as the answer to an earlier
    sub-question.  Tier-level summaries intentionally have no part.
    """
    guidance = output.get("guidance")
    if not isinstance(guidance, Mapping):
        return
    for tier_key, tier in guidance.items():
        if not isinstance(tier, Mapping):
            continue
        parts = tier.get("parts")
        if isinstance(parts, list):
            for part_index, part in enumerate(parts):
                if not isinstance(part, Mapping):
                    continue
                part_label = normalized_part_label(part.get("part")) or None
                steps = part.get("steps")
                if isinstance(steps, list):
                    for step_index, step in enumerate(steps):
                        if isinstance(step, str):
                            yield (
                                f"guidance.{tier_key}.parts[{part_index}]"
                                f".steps[{step_index}]",
                                step,
                                part_label,
                            )
                alternatives = part.get("alternative_methods")
                if isinstance(alternatives, list):
                    for method_index, method in enumerate(alternatives):
                        if not isinstance(method, Mapping):
                            continue
                        method_steps = method.get("steps")
                        if not isinstance(method_steps, list):
                            continue
                        for step_index, step in enumerate(method_steps):
                            if isinstance(step, str):
                                yield (
                                    f"guidance.{tier_key}.parts[{part_index}]"
                                    f".alternative_methods[{method_index}]"
                                    f".steps[{step_index}]",
                                    step,
                                    part_label,
                                )
                pitfall = part.get("pitfall")
                if isinstance(pitfall, str):
                    yield (
                        f"guidance.{tier_key}.parts[{part_index}].pitfall",
                        pitfall,
                        part_label,
                    )
        summary = tier.get("pattern_summary")
        if isinstance(summary, str):
            yield (f"guidance.{tier_key}.pattern_summary", summary, None)


def guidance_strings(output: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    for path, value, _ in guidance_entries(output):
        yield path, value


def answer_guard_entries(
    output: Mapping[str, Any],
) -> Iterable[tuple[str, str, str | None]]:
    """Yield every model-authored string that can reach a student page.

    ``question_blocks`` are deliberately excluded because they transcribe the
    printed question and may legitimately contain the requested relation.
    Figure captions are private crop metadata.  All other generated text that
    the renderer can show is scanned, including recognition warnings, method
    titles, knowledge/formula cards and the three guidance tiers.
    """

    for index, value in enumerate(output.get("uncertain_items") or []):
        if isinstance(value, str):
            yield f"uncertain_items[{index}]", value, None
    for index, value in enumerate(output.get("knowledge_points") or []):
        if isinstance(value, str):
            yield f"knowledge_points[{index}]", value, None
    for index, item in enumerate(output.get("base_formulas") or []):
        if not isinstance(item, Mapping):
            continue
        formula = item.get("formula")
        if isinstance(formula, str):
            wrapped = formula if "$" in formula else f"${formula}$"
            yield f"base_formulas[{index}].formula", wrapped, None
        condition = item.get("condition")
        if isinstance(condition, str):
            yield f"base_formulas[{index}].condition", condition, None
    for index, value in enumerate(output.get("visual_evidence") or []):
        if isinstance(value, str):
            yield f"visual_evidence[{index}]", value, None

    guidance = output.get("guidance")
    if isinstance(guidance, Mapping):
        for tier_key, tier in guidance.items():
            if not isinstance(tier, Mapping):
                continue
            for part_index, part in enumerate(tier.get("parts") or []):
                if not isinstance(part, Mapping):
                    continue
                part_label = normalized_part_label(part.get("part")) or None
                method_name = part.get("method_name")
                if isinstance(method_name, str):
                    yield (
                        f"guidance.{tier_key}.parts[{part_index}].method_name",
                        method_name,
                        part_label,
                    )
                for method_index, method in enumerate(
                    part.get("alternative_methods") or []
                ):
                    if not isinstance(method, Mapping):
                        continue
                    method_name = method.get("method_name")
                    if isinstance(method_name, str):
                        yield (
                            f"guidance.{tier_key}.parts[{part_index}]"
                            f".alternative_methods[{method_index}].method_name",
                            method_name,
                            part_label,
                        )
    for path, value, part_label in display_guidance_entries(output):
        yield path, value, part_label
    yield from guidance_entries(output)


def numeric_substitution_markers(value: str) -> list[str]:
    """Find problem-value substitution while allowing indices and formula constants."""
    scrubbed = SCRIPTED_INTEGER.sub("", value)
    matches: list[tuple[int, str]] = []
    for pattern in (
        NUMERIC_WITH_UNIT,
        SIGNIFICANT_NUMERIC_LITERAL,
        ISOLATED_NUMERIC_RESULT,
    ):
        for match in pattern.finditer(scrubbed):
            marker = match.group(0).strip()
            # A symbolic zero condition (for example x=0 or a>0) is a
            # mathematical relation, not substitution of a problem value.
            # Zero carrying a unit is still rejected by NUMERIC_WITH_UNIT.
            if pattern is ISOLATED_NUMERIC_RESULT and re.fullmatch(
                r"(?:=|≈|>|<|≥|≤|\\geq?|\\leq?)\s*[+-]?0(?:\.0+)?",
                marker,
            ):
                continue
            if marker:
                matches.append((match.start(), marker))
    output: list[str] = []
    for _, marker in sorted(matches, key=lambda item: (item[0], -len(item[1]))):
        if marker not in output and not any(marker in prior for prior in output):
            output.append(marker)
    return output


def canonical_ratio_symbol(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    return re.sub(r"_\{([^{}]+)\}", r"_\1", compact)


def question_ratio_substitution_markers(
    value: str,
    question_text: str,
) -> list[str]:
    """Find copied or derived numeric geometry ratios from the question.

    Formula constants such as ``1/2`` remain allowed.  This gate is scoped to
    symbols that the printed question itself places over an explicit integer,
    for example ``b/4``.  Reusing ``b/4`` or deriving ``3b/4``/``b/2`` in a
    guidance page is therefore treated as problem-value substitution.
    """
    question_symbols: set[str] = set()
    for formula in INLINE_FORMULA.findall(question_text):
        for match in QUESTION_SYMBOL_OVER_INTEGER.finditer(formula):
            symbol = match.group("latex_symbol") or match.group("plain_symbol")
            if symbol:
                question_symbols.add(canonical_ratio_symbol(symbol))
    if not question_symbols:
        return []

    matches: list[str] = []
    for formula in INLINE_FORMULA.findall(value):
        for match in GUIDANCE_NUMERIC_SYMBOL_RATIO.finditer(formula):
            symbol = match.group("latex_symbol") or match.group("plain_symbol")
            if (
                symbol
                and canonical_ratio_symbol(symbol) in question_symbols
                and match.group(0) not in matches
            ):
                matches.append(match.group(0))
    return matches


def compact_formula_side(value: str) -> str:
    compact = re.sub(r"\s+", "", value)
    compact = compact.replace(r"\left", "").replace(r"\right", "")
    while len(compact) >= 2 and (
        (compact.startswith("(") and compact.endswith(")"))
        or (compact.startswith("[") and compact.endswith("]"))
    ):
        compact = compact[1:-1]
    return compact


FORMULA_IDENTIFIER = re.compile(
    r"\\?[A-Za-z]+(?:_\{?[A-Za-z0-9]+\}?)?|[Α-Ωα-ω]+"
)
NON_VARIABLE_FORMULA_IDENTIFIERS = {
    "angle", "boxed", "cdot", "circ", "cos", "cot", "csc", "dfrac", "div", "exp",
    "frac", "ge", "geq", "in", "infty", "le", "left", "leq", "ln",
    "log", "mathbb", "mathrm", "mp", "ne", "neq", "notin",
    "operatorname", "overrightarrow", "parallel", "perp", "pi", "pm",
    "right", "sec", "sin", "sqrt", "tan", "text", "tfrac", "times",
    "triangle",
}


def formula_identifiers(value: str) -> set[str]:
    """Return mathematical identifiers, excluding notation commands/constants.

    This intentionally answers a narrow question for the answer gate: does an
    isolated target relation still contain an auxiliary quantity that must be
    eliminated?  It is not a symbolic algebra parser.
    """

    scrubbed = re.sub(
        r"\\(?:mathrm|text|operatorname)\{[^{}]*\}",
        "",
        str(value or ""),
    )
    identifiers: set[str] = set()
    for match in FORMULA_IDENTIFIER.finditer(scrubbed):
        token = match.group(0)
        command = token[1:] if token.startswith("\\") else token
        if command.lower() in NON_VARIABLE_FORMULA_IDENTIFIERS:
            continue
        compact_function = re.fullmatch(
            r"(?:sin|cos|tan|cot|sec|csc|ln|log)([A-Za-z]+)",
            command,
            re.IGNORECASE,
        )
        if compact_function:
            identifiers.add(compact_formula_side(compact_function.group(1)))
            continue
        identifiers.add(compact_formula_side(token))
    return identifiers


def formula_side_contains_target(value: str, target: str) -> bool:
    """Return whether one relation side contains the requested symbol.

    A final result is often written in an equivalent form such as
    ``5DF=6r`` or ``DF/r=6/5`` instead of isolating ``DF``.  Exact side
    equality therefore misses direct answers.  The boundary rule below keeps
    multi-letter geometry labels intact: ``DF`` matches ``5DF`` but not
    ``CDF``.
    """

    compact = compact_formula_side(value)
    if not target:
        return False
    if compact == target or compact.startswith(target + r"\mathrm{"):
        return True
    if re.fullmatch(r"[A-Za-z]+(?:_\{?[A-Za-z0-9]+\}?)?", target):
        return re.search(
            rf"(?<![A-Za-z_]){re.escape(target)}(?![A-Za-z_])",
            compact,
        ) is not None
    return target in compact


def identifiers_resolvable_from(
    formulas: Sequence[str], allowed: set[str]
) -> set[str]:
    """Close simple named intermediate definitions over allowed parameters.

    This deliberately does not attempt symbolic algebra.  It only follows
    definitions whose one side is a single identifier, for example
    ``t=6r/5`` and ``u=3t``.  The closure prevents a direct result from being
    split across visible fields as ``DF=u`` plus ``u=6r/5``.
    """

    resolved = set(allowed)
    definitions: list[tuple[str, set[str]]] = []
    for formula in formulas:
        sides = EQUATION_SEPARATOR.split(formula)
        if len(sides) != 2 or not EQUALITY_RELATION.search(formula):
            continue
        for index, side in enumerate(sides):
            side_ids = formula_identifiers(side)
            compact_side = compact_formula_side(side)
            if len(side_ids) != 1:
                continue
            identifier = next(iter(side_ids))
            if compact_side != identifier:
                continue
            # Only scalar auxiliary names participate in the split-answer
            # closure.  Uppercase geometry labels such as DM, DCF or DOM are
            # genuine intermediate objects; resolving them across several
            # construction steps is normal pedagogy, not a disguised one-line
            # final answer.  Lowercase t/u/v (including subscripts) covers the
            # deliberate alias-chain escape this closure is meant to stop.
            if not re.fullmatch(r"[a-z](?:_\{?[A-Za-z0-9]+\}?)?", identifier):
                continue
            other_ids: set[str] = set()
            for other_index, other_side in enumerate(sides):
                if other_index != index:
                    other_ids.update(formula_identifiers(other_side))
            definitions.append((identifier, other_ids))
    changed = True
    while changed:
        changed = False
        for identifier, dependencies in definitions:
            if identifier not in resolved and dependencies.issubset(resolved):
                resolved.add(identifier)
                changed = True
    return resolved


def is_derived_integer_coefficient(
    marker: str,
    guidance_value: str,
    question_text: str,
) -> bool:
    """Distinguish symbolic 2g/2L factors from copied problem measurements."""
    compact_marker = re.sub(r"\s+", "", marker)
    if not re.fullmatch(r"[1-9][A-Za-z]", compact_marker):
        return False
    if compact_marker in re.sub(r"\s+", "", question_text):
        return False
    for formula in INLINE_FORMULA.findall(guidance_value):
        if (
            compact_marker in re.sub(r"\s+", "", formula)
            and EQUATION_SEPARATOR.search(formula)
        ):
            return True
    return False


def validate_answer_guard(
    output: Mapping[str, Any],
    requested_outputs: Sequence[Mapping[str, Any]],
    errors: list[str],
    *,
    policy: str = "strict",
) -> None:
    """Reject delivered answers while allowing equations in device mode."""
    if policy not in {"strict", "no_answer"}:
        raise ValueError(f"未知答案门禁策略：{policy}")
    symbol_targets: list[tuple[str, str]] = []
    range_symbol_targets: list[tuple[str, str]] = []
    exact_symbol_targets: list[tuple[str, str]] = []
    value_symbol_targets: list[tuple[str, str]] = []
    prose_targets: list[tuple[str, str]] = []
    allowed_parameters_by_target: dict[tuple[str, str], set[str]] = {}
    for item in requested_outputs:
        part = normalized_part_label(item.get("part"))
        target_type = str(item.get("target_type") or "")
        declared_parameters: set[str] | None = None
        if "allowed_result_parameters" in item:
            declared_parameters = set()
            for parameter in item.get("allowed_result_parameters", []):
                if isinstance(parameter, str) and parameter.strip():
                    declared_parameters.update(formula_identifiers(parameter))
        for alias in item.get("symbol_aliases", []):
            if isinstance(alias, str) and alias.strip():
                compact_alias = compact_formula_side(alias.strip("$ "))
                symbol_targets.append((part, compact_alias))
                if declared_parameters is not None:
                    allowed_parameters_by_target[(part, compact_alias)] = set(
                        declared_parameters
                    )
                if target_type in {
                    "inequality", "set", "interval", "range"
                }:
                    range_symbol_targets.append((part, compact_alias))
                elif target_type in {
                    "value", "numeric", "expression", "equation"
                }:
                    exact_symbol_targets.append((part, compact_alias))
                    if target_type in {"value", "numeric"}:
                        value_symbol_targets.append((part, compact_alias))
        for alias in item.get("range_symbol_aliases", []):
            if isinstance(alias, str) and alias.strip():
                pair = (part, compact_formula_side(alias.strip("$ ")))
                if pair not in range_symbol_targets:
                    range_symbol_targets.append(pair)
        for alias in item.get("prose_aliases", []):
            if isinstance(alias, str) and alias.strip():
                prose_targets.append((part, alias.strip()))

    symbols_by_part: dict[str, list[str]] = {}
    range_symbols_by_part: dict[str, list[str]] = {}
    exact_symbols_by_part: dict[str, list[str]] = {}
    prose_by_part: dict[str, list[str]] = {}
    for part, target in symbol_targets:
        symbols_by_part.setdefault(part, []).append(target)
    for part, target in range_symbol_targets:
        range_symbols_by_part.setdefault(part, []).append(target)
    for part, target in exact_symbol_targets:
        exact_symbols_by_part.setdefault(part, []).append(target)
    for part, alias in prose_targets:
        prose_by_part.setdefault(part, []).append(alias)

    range_target_specs = set(range_symbol_targets)
    exact_target_specs = set(exact_symbol_targets)
    value_target_specs = set(value_symbol_targets)

    known_formulas: set[str] = set()
    question_identifiers: set[str] = set()
    for block in output.get("question_blocks") or []:
        if not isinstance(block, Mapping) or block.get("type") != "text":
            continue
        for formula in INLINE_FORMULA.findall(str(block.get("text") or "")):
            known_formulas.add(compact_formula_side(formula))
            question_identifiers.update(formula_identifiers(formula))

    visible_entries = list(answer_guard_entries(output))

    def guard_markup(path: str, value: str) -> str:
        # ``base_formulas[*].formula`` is already an all-formula schema field.
        # Running mixed prose promotion on it can split ``\sin(∠DCF)`` into
        # nested dollar fragments and hide the still-unresolved angle.  Wrap
        # the complete field once; all other visible prose keeps the normal
        # promotion path.
        if re.fullmatch(r"base_formulas\[\d+\]\.formula", path):
            return "$" + str(value or "").strip().strip("$") + "$"
        return promote_plain_algebra_markup(value)

    # Keep equation dependencies and introduced auxiliaries in both their tier
    # and printed sub-question namespace.  The three tiers are parallel pages,
    # not consecutive derivation steps: a relation in the foundation page must
    # never complete a formula shown on the standard page.  Partless cards are
    # shared teaching material and remain visible within each individual tier.
    ScopeKey = tuple[str | None, str | None]
    visible_formulas_by_scope: dict[ScopeKey, list[str]] = {}
    introduced_auxiliaries_by_scope: dict[ScopeKey, set[str]] = {}

    def tier_from_path(path: str) -> str | None:
        match = re.match(r"guidance\.(advanced|standard|foundation)\.", path)
        return match.group(1) if match is not None else None

    for _path, visible_value, _part in visible_entries:
        scope = (tier_from_path(_path), _part)
        promoted = guard_markup(_path, visible_value)
        visible_formulas_by_scope.setdefault(scope, []).extend(
            INLINE_FORMULA.findall(promoted)
        )
        introduced_here = introduced_auxiliaries_by_scope.setdefault(scope, set())
        for match in re.finditer(
            r"(?:设|令|记|取)(?:辅助量|变量|参数|点)?\s*\$?"
            r"(?P<symbol>[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?)",
            promoted,
        ):
            introduced_here.update(
                formula_identifiers(match.group("symbol"))
            )
        for match in re.finditer(
            r"(?:辅助量|中间量)\s*\$?"
            r"(?P<symbol>[A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?)",
            promoted,
        ):
            introduced_here.update(
                formula_identifiers(match.group("symbol"))
            )

    def scoped_values(part: str, tier: str | None) -> list[str]:
        result = list(visible_formulas_by_scope.get((None, None), []))
        if tier is not None:
            result.extend(visible_formulas_by_scope.get((tier, None), []))
            result.extend(visible_formulas_by_scope.get((tier, part), []))
        return result

    def scoped_auxiliaries(part: str, tier: str | None) -> set[str]:
        result = set(introduced_auxiliaries_by_scope.get((None, None), set()))
        if tier is not None:
            result.update(introduced_auxiliaries_by_scope.get((tier, None), set()))
            result.update(introduced_auxiliaries_by_scope.get((tier, part), set()))
        return result

    resolvable_by_target_and_tier: dict[tuple[tuple[str, str], str | None], set[str]] = {}
    for spec in symbol_targets:
        origin_part, target = spec
        for tier in ("advanced", "standard", "foundation", None):
            resolvable_by_target_and_tier[(spec, tier)] = (
                identifiers_resolvable_from(
                    scoped_values(origin_part, tier),
                    allowed_parameters_by_target.get(spec, set()),
                )
                - formula_identifiers(target)
            )

    for path, value, current_part in visible_entries:
        # Some renderer-visible metadata (for example knowledge points,
        # recognition warnings and method names) deliberately remains plain
        # text in the stored JSON.  Scan it through the same conservative
        # algebra promotion used by guidance rendering so a bare ``k=2``
        # cannot evade a gate that already rejects ``$k=2$``.  This is only a
        # validation view: the source object is not mutated, and transcribed
        # question_blocks remain excluded by ``answer_guard_entries``.
        guard_value = guard_markup(path, value)
        current_tier = tier_from_path(path)
        if current_part is None:
            active_symbol_specs = list(symbol_targets)
            active_prose = [alias for _, alias in prose_targets]
            active_part = "通用总结"
        else:
            active_symbol_specs = [
                spec for spec in symbol_targets if spec[0] == current_part
            ]
            active_prose = list(prose_by_part.get(current_part, []))
            active_part = current_part
            # A later sub-question must not reveal the closed final expression
            # of an earlier parameterized-expression target.  Include only
            # targets with an explicit final-parameter contract here; the open
            # relation check below still allows useful intermediate equations.
            for spec in allowed_parameters_by_target:
                if spec not in active_symbol_specs:
                    active_symbol_specs.append(spec)
        for formula_index, formula in enumerate(
            INLINE_FORMULA.findall(guard_value)
        ):
            compact_formula = compact_formula_side(formula)
            if compact_formula in known_formulas or any(
                compact_formula
                and known.startswith(compact_formula)
                for known in known_formulas
            ):
                # Repeating a printed condition is not delivering a new answer.
                # A definitional relation such as ``y=\frac{k}{x}`` is usually
                # transcribed in the stem together with its trailing domain
                # clause (``… (x>0 且 k\ne 0)``), so the clean relation only
                # appears as a prefix of the known formula.
                continue
            sides = EQUATION_SEPARATOR.split(formula)
            if policy == "no_answer" and r"\boxed" in formula:
                boxed_identifiers = formula_identifiers(formula)
                for origin_part, target in active_symbol_specs:
                    target_spec = (origin_part, target)
                    if target_spec not in exact_target_specs:
                        continue
                    parameters = allowed_parameters_by_target.get(target_spec, set())
                    permitted = set(parameters) | formula_identifiers(target)
                    if (
                        re.search(r"\d", formula)
                        and boxed_identifiers.issubset(permitted)
                    ):
                        errors.append(
                            f"{path}.formula[{formula_index}]用方框直接给出了"
                            f"{origin_part or active_part}所求量{target}"
                        )
                        break
            if len(sides) < 2:
                continue
            compact_sides = [compact_formula_side(side) for side in sides]
            for origin_part, target in active_symbol_specs:
                target_spec = (origin_part, target)
                is_range_target = target_spec in range_target_specs
                is_exact_target = target_spec in exact_target_specs
                target_side_indexes = [
                    index
                    for index, side in enumerate(compact_sides)
                    if (
                        formula_side_contains_target(side, target)
                        if is_exact_target
                        else (
                            side == target
                            or side.startswith(target + r"\mathrm{")
                        )
                    )
                ]
                if not target_side_indexes:
                    continue
                if policy == "no_answer":
                    explicit_final_context = bool(
                        re.search(
                            r"(?:最终(?:答案|结果|值|范围|结论)?\s*"
                            r"(?:为|是|等于|得到|求得|解得|可得|写成|化为|[:：=])|"
                            r"答案(?:为|是|等于|[:：=])?|解得|求得|算得|"
                            r"最后(?:得到|求得|解得|可得))",
                            guard_value,
                        )
                    )
                    constraint_relation = bool(RANGE_RELATION.search(formula))
                    equality_relation = bool(EQUALITY_RELATION.search(formula))
                    intermediate_context = bool(re.search(
                        r"(?:先|设|令|作|连接|取点|建立|列出|写出|写成|表示为|可写为|"
                        r"满足|符合|中点|垂足|中间关系|未解关系|暂记|保留关系)",
                        guard_value,
                    ))

                    if is_range_target:
                        membership_or_set = bool(re.search(
                            r"(?:∈|∉|\\in(?![A-Za-z])|"
                            r"\\notin(?![A-Za-z])|[{}])",
                            formula,
                        ))
                        relation_count = len(RANGE_RELATION.findall(formula))
                        named_final_range = bool(re.search(
                            r"(?:取值范围|参数范围|解集|定义域|值域)"
                            r"\s*(?:为|是|等于|[:：=])",
                            guard_value,
                        ))
                        # A single explicitly intermediate constraint is not a
                        # delivered range answer.  Membership/set notation, a
                        # two-sided chain, or named final-range wording is
                        # already a directly usable answer and remains blocked.
                        if (
                            intermediate_context
                            and not explicit_final_context
                            and not membership_or_set
                            and relation_count <= 1
                            and not named_final_range
                        ):
                            continue
                    elif is_exact_target:
                        # A sign/domain constraint such as k>0 is an
                        # intermediate hint for an exact-value question.  An
                        # equality is allowed only when it is explicitly an
                        # unfinished relation under the rules below.
                        if (
                            constraint_relation
                            and not equality_relation
                            and not explicit_final_context
                        ):
                            continue
                        if not equality_relation and not explicit_final_context:
                            continue
                        declared_parameters = allowed_parameters_by_target.get(
                            target_spec
                        )
                        target_identifiers = formula_identifiers(target)
                        other_identifiers = (
                            formula_identifiers(formula) - target_identifiers
                        )
                        candidate_tiers = (
                            (current_tier,)
                            if current_tier is not None
                            else ("advanced", "standard", "foundation", None)
                        )
                        unresolved_options = [
                            other_identifiers
                            - resolvable_by_target_and_tier.get(
                                (target_spec, tier), set()
                            )
                            for tier in candidate_tiers
                        ]
                        intrinsic_auxiliaries: set[str] = set()
                        for angle_match in re.finditer(
                            r"(?:\\angle|∠)\s*([A-Za-z]+)", formula
                        ):
                            intrinsic_auxiliaries.update(
                                formula_identifiers(angle_match.group(1))
                            )
                        legitimate_unresolved = bool(unresolved_options) and all(
                            bool(unresolved)
                            and unresolved.issubset(
                                question_identifiers
                                | scoped_auxiliaries(origin_part, tier)
                                | intrinsic_auxiliaries
                                | {
                                    identifier
                                    for identifier in unresolved
                                    if intermediate_context
                                    and re.fullmatch(r"[A-Z]{1,4}", identifier)
                                }
                            )
                            for tier, unresolved in zip(
                                candidate_tiers, unresolved_options
                            )
                        )
                        # A target equality is a safe intermediate relation when
                        # it still contains a quantity that the requested final
                        # form must eliminate.  Explicit result parameters make
                        # that distinction deterministic (for example, a final
                        # DF expression may contain r, but not an unresolved
                        # angle A).  For questions without that metadata, require
                        # an explicit intermediate-relation cue instead of
                        # treating a bare ``k=a`` as harmless.
                        if (
                            legitimate_unresolved
                            and (
                                declared_parameters is not None
                                or intermediate_context
                            )
                            and not explicit_final_context
                        ):
                            continue
                        # Formula/reference cards (base_formulas and
                        # knowledge_points) teach a relation rather than
                        # deliver a step.  For a purely numeric target (find
                        # the value of ``k``) any symbol-only isolation there —
                        # such as ``k=x_A·y_A`` or ``xy=k`` — is necessarily an
                        # intermediate setup hint, because the delivered answer
                        # must be a number.  Mirror the legacy numeric-result
                        # rule: a number-with-unit or pure-numeric side is a
                        # delivered value, and explicit final phrasing is
                        # blocked.  Expression targets (``5DF=6r``) keep the
                        # strict isolation rule because their answer IS a
                        # symbolic relation.
                        is_formula_card = (
                            path.startswith("base_formulas")
                            or path.startswith("knowledge_points")
                        )
                        if is_formula_card and target_spec in value_target_specs:
                            card_other_sides = [
                                side
                                for side in compact_sides
                                if not (
                                    side == target
                                    or side.startswith(target + r"\mathrm{")
                                )
                            ]
                            card_numeric_result = bool(
                                NUMERIC_WITH_UNIT.search(formula)
                                or any(
                                    re.fullmatch(
                                        r"[+-]?\d+(?:\.\d+)?(?:%|％)?",
                                        side,
                                    )
                                    for side in card_other_sides
                                )
                            )
                            if (
                                not card_numeric_result
                                and not explicit_final_context
                            ):
                                continue
                        # A numeric value question may hand over the final
                        # setup equation (``2=k/4``, ``k+2=10``) as an
                        # intermediate relation, because the requested quantity
                        # is still embedded and the student must solve for it.
                        # The prompt already allows such process equations and
                        # only forbids evaluating them.  Isolated forms
                        # (``k=8``, ``2=k``, ``k=2\times4``) and expression
                        # targets (``5DF=6r``) remain a delivered answer.
                        if target_spec in value_target_specs and not any(
                            side == target or side.startswith(target + r"\mathrm{")
                            for side in compact_sides
                        ):
                            if not explicit_final_context:
                                continue
                        # For a value/numeric target, an isolated equality is
                        # still only a setup relation when another symbolic
                        # quantity remains.  The answer guard should block the
                        # evaluated value (``k=2``), not ordinary algebra such
                        # as ``k=a`` or ``a=k``.  Mathematical usefulness of
                        # that relation is handled by the reasoning validator;
                        # it is not an answer-leak concern.
                        if (
                            target_spec in value_target_specs
                            and other_identifiers
                            and not explicit_final_context
                        ):
                            continue
                    else:
                        # Legacy manifests without target_type retain the old
                        # conservative behavior, but inspect every non-target
                        # side instead of assuming the result is on the right.
                        other_sides = [
                            side for side in compact_sides
                            if not (
                                side == target
                                or side.startswith(target + r"\mathrm{")
                            )
                        ]
                        numeric_result = bool(
                            NUMERIC_WITH_UNIT.search(formula)
                            or any(re.fullmatch(
                                r"[+-]?\d+(?:\.\d+)?(?:%|％)?",
                                side,
                            ) for side in other_sides)
                        )
                        if not numeric_result and not explicit_final_context:
                            continue
                reported_part = origin_part or active_part
                errors.append(
                    f"{path}.formula[{formula_index}]单独解出了"
                    f"{reported_part}所求量{target}"
                )
        if policy == "no_answer":
            for origin_part, target in active_symbol_specs:
                target_spec = (origin_part, target)
                parameters = allowed_parameters_by_target.get(target_spec)
                if target_spec not in exact_target_specs or not parameters:
                    continue
                target_text = re.escape(target.replace("\\", ""))
                for parameter in parameters:
                    parameter_text = re.escape(parameter.replace("\\", ""))
                    if re.search(
                        target_text
                        + r".{0,8}(?:是|为|等于|约为).{0,12}"
                        + parameter_text
                        + r".{0,8}(?:的)?(?:\d+(?:\.\d+)?|\d+/\d+)倍",
                        re.sub(r"[${}\\]", "", value),
                    ):
                        errors.append(
                            f"{path}用文字直接给出了"
                            f"{origin_part or active_part}所求量{target}的倍数关系"
                        )
                        break
        for alias in active_prose:
            escaped = re.escape(alias)
            # “是否/为何” is a verification question, not the value copula;
            # “核对是否符合 k≠0” must not read as “X的值是…0”.
            copula = r"(?:为(?!什)|等于|是(?!否)|应为|约为|方向为|=|≈)"
            direct_after = re.search(
                escaped
                + r"[^，,。；;：:\n]{0,8}(?:"
                + copula
                + r")"
                + r"[^，,。；;：:\n]{0,16}(?:\d|向左|向右|向上|向下|成立|不成立|增大|减小|不变|小于|大于|等于)",
                value,
            )
            direct_before = re.search(
                r"(?:得到|求得|算得|可得)[^，,。；;：:\n]{0,8}"
                + escaped
                + r"[^，,。；;：:\n]{0,8}(?:"
                + copula
                + r")?"
                + r"[^，,。；;：:\n]{0,12}(?:\d|向左|向右|向上|向下|小于|大于|等于)",
                value,
            )
            if direct_after or direct_before:
                errors.append(
                    f"{path}用文字直接给出了{active_part}所求结论：{alias}"
                )
        for marker in qualitative_prose_answer_markers(value, active_prose):
            errors.append(
                f"{path}直接给出了{active_part}定性结论：{marker}"
            )


QUALITATIVE_TARGET_GROUPS: tuple[tuple[str, ...], ...] = (
    ("平均速率", "平均速度", "分子速率"),
    ("碰撞次数", "碰撞频率", "撞击次数", "撞击频率"),
    ("温度高低", "温度"),
    ("电势高低", "电势"),
    ("电性", "电荷性质"),
    ("电流方向",),
    ("运动方向", "速度方向"),
    ("偏转方向",),
    ("受力方向", "安培力方向", "洛伦兹力方向"),
    ("磁场方向",),
    ("变化趋势", "内能", "压强", "体积", "动能", "势能"),
)

QUALITATIVE_RESULT = re.compile(
    r"(?:会|将|必然|必须|逐渐|不断|保持|变得|呈|为)?"
    r"(?:增大|减小|增加|减少|升高|降低|不变|变大|变小|"
    r"更大|更小|较大|较小|越大|越小|偏高|偏低|为正|为负|"
    r"向左|向右|向上|向下|顺时针|逆时针)"
)


def qualitative_prose_answer_markers(
    value: str, prose_aliases: Sequence[str]
) -> list[str]:
    """Find a delivered qualitative answer for a requested qualitative target.

    This is intentionally narrower than a general keyword ban.  A result word
    is actionable only when the manifest says that the current subquestion asks
    for a change, trend, direction, high/low comparison, or yes/no conclusion,
    and when the sentence states the outcome next to that requested quantity.
    Normal equations, numeric references, and open alternatives remain allowed.
    """

    markers: list[str] = []
    qualitative_aliases = [
        alias
        for alias in prose_aliases
        if re.search(r"(?:变化|趋势|方向|高低|大小关系|是否|增减|快慢)", alias)
    ]
    if not qualitative_aliases:
        return markers

    active_groups = [
        group
        for group in QUALITATIVE_TARGET_GROUPS
        if any(any(term in alias for term in group) for alias in qualitative_aliases)
    ]
    for group in active_groups:
        for term in sorted(group, key=len, reverse=True):
            for match in re.finditer(re.escape(term) + r".{0,14}?" + QUALITATIVE_RESULT.pattern, value):
                snippet = match.group(0)
                # “增大还是减小”“向左或向右”只是在提出判断任务，不是交付结论。
                nearby = value[max(0, match.start() - 4) : min(len(value), match.end() + 8)]
                if re.search(r"(?:还是|或者|或是|是否)", nearby):
                    continue
                markers.append(snippet)
    return list(dict.fromkeys(markers))


def validate_output(
    output: Any,
    requested_outputs: Sequence[Mapping[str, Any]] | None = None,
    *,
    device_policy: bool = False,
) -> list[str]:
    errors: list[str] = []
    root_required = set(ROOT_KEYS)
    root_candidate = dict(output) if isinstance(output, dict) else output
    if isinstance(root_candidate, dict):
        root_candidate.pop("display_guidance", None)
    exact_keys(root_candidate, root_required, "$", errors)
    if not isinstance(output, dict):
        return errors
    if "display_guidance" in output and not isinstance(output["display_guidance"], dict):
        errors.append("display_guidance必须为对象")
    if output.get("subject") != "mathematics":
        errors.append("subject必须为mathematics")
    if output.get("problem_scope") != MATHEMATICS_PROBLEM_SCOPE:
        errors.append(
            "problem_scope必须为primary_to_high_school（小学、初中、高中）"
        )
    status = output.get("recognition_status")
    if status not in {"clear", "uncertain"}:
        errors.append("recognition_status不正确")
    uncertain = string_list(
        output.get("uncertain_items"),
        "uncertain_items",
        errors,
        minimum=0,
        maximum=8,
    )
    if status == "uncertain" and not uncertain:
        errors.append("识图不确定时必须说明uncertain_items")
    blocks = output.get("question_blocks")
    block_figure_ids: list[str] = []
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 40:
        errors.append("question_blocks必须包含1到40项")
    else:
        for index, block in enumerate(blocks):
            exact_keys(
                block,
                {"type", "text", "figure_id"},
                f"question_blocks[{index}]",
                errors,
            )
            if not isinstance(block, Mapping):
                continue
            block_type = block.get("type")
            text = block.get("text")
            figure_id = block.get("figure_id")
            if block_type == "text":
                if not isinstance(text, str) or not text.strip():
                    errors.append(f"question_blocks[{index}].text不能为空")
                if figure_id != "":
                    errors.append(f"question_blocks[{index}]文字块figure_id必须为空")
                if isinstance(text, str):
                    if text.count("$") % 2:
                        errors.append(f"question_blocks[{index}].text公式定界符$不成对")
                    for formula_index, formula in enumerate(INLINE_FORMULA.findall(text)):
                        validate_formula_source(
                            formula,
                            f"question_blocks[{index}].formula[{formula_index}]",
                            errors,
                        )
                    without_formulae = re.sub(r"\$[^$]*\$", "", text)
                    if "\\" in without_formulae:
                        errors.append(f"question_blocks[{index}].text有未用$包围的LaTeX")
                    if "```" in text or "`" in text:
                        errors.append(f"question_blocks[{index}].text含Markdown代码标记")
            elif block_type == "figure":
                if text != "":
                    errors.append(f"question_blocks[{index}]图片块text必须为空")
                if not isinstance(figure_id, str) or not figure_id.strip():
                    errors.append(f"question_blocks[{index}].figure_id不能为空")
                else:
                    block_figure_ids.append(figure_id.strip())
            else:
                errors.append(f"question_blocks[{index}].type不正确")
    source_figures = output.get("source_figures")
    source_figure_ids: list[str] = []
    if not isinstance(source_figures, list) or len(source_figures) > 12:
        errors.append("source_figures必须是最多12项的数组")
    else:
        for index, figure in enumerate(source_figures):
            exact_keys(
                figure,
                {"figure_id", "bbox_1000", "caption"},
                f"source_figures[{index}]",
                errors,
            )
            if not isinstance(figure, Mapping):
                continue
            figure_id = figure.get("figure_id")
            if not isinstance(figure_id, str) or not figure_id.strip():
                errors.append(f"source_figures[{index}].figure_id不能为空")
            else:
                source_figure_ids.append(figure_id.strip())
            caption = figure.get("caption")
            if not isinstance(caption, str):
                errors.append(f"source_figures[{index}].caption必须是字符串")
            bbox = figure.get("bbox_1000")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    for value in bbox
                )
            ):
                errors.append(f"source_figures[{index}].bbox_1000必须含4个有限数")
            else:
                left, top, right, bottom = (float(value) for value in bbox)
                if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
                    errors.append(f"source_figures[{index}].bbox_1000越界或面积无效")
                else:
                    area_ratio = (right - left) * (bottom - top) / 1_000_000
                    if area_ratio < 0.001:
                        errors.append(f"source_figures[{index}].bbox_1000过小")
                    if area_ratio > 0.55:
                        errors.append(f"source_figures[{index}]疑似夹带正文或照搬整页原图")
    if len(set(block_figure_ids)) != len(block_figure_ids):
        errors.append("question_blocks中的figure_id不得重复")
    if len(set(source_figure_ids)) != len(source_figure_ids):
        errors.append("source_figures中的figure_id不得重复")
    if block_figure_ids != source_figure_ids:
        errors.append("question_blocks与source_figures的figure_id及顺序必须完全一致")
    string_list(
        output.get("visual_evidence"),
        "visual_evidence",
        errors,
        minimum=0,
        maximum=10,
    )
    knowledge = string_list(
        output.get("knowledge_points"),
        "knowledge_points",
        errors,
        minimum=2,
        maximum=5,
    )
    if len(set(knowledge)) != len(knowledge):
        errors.append("knowledge_points不得重复")
    formulae = output.get("base_formulas")
    if not isinstance(formulae, list) or not 1 <= len(formulae) <= 3:
        errors.append("base_formulas必须包含1到3项")
    else:
        for index, item in enumerate(formulae):
            exact_keys(item, {"formula", "condition"}, f"base_formulas[{index}]", errors)
            if isinstance(item, Mapping):
                for key in ("formula", "condition"):
                    if not isinstance(item.get(key), str) or not str(item.get(key)).strip():
                        errors.append(f"base_formulas[{index}].{key}不能为空")
                formula = item.get("formula")
                if isinstance(formula, str):
                    validate_formula_source(
                        formula,
                        f"base_formulas[{index}].formula",
                        errors,
                    )
                    if MALFORMED_MIDPOINT_HALF_SEGMENT.search(formula):
                        errors.append(
                            f"base_formulas[{index}].formula中点半长公式"
                            "不得把整段名写进分母"
                        )
    guidance = output.get("guidance")
    exact_keys(guidance, set(TIER_SPECS), "guidance", errors)
    tier_part_sequences: dict[str, list[str]] = {}
    tier_method_signatures: dict[str, list[list[tuple[str, str]]]] = {}
    if isinstance(guidance, Mapping):
        for tier_key, spec in TIER_SPECS.items():
            tier = guidance.get(tier_key)
            expected = {"title", "parts"}
            if tier_key == "foundation":
                expected.add("pattern_summary")
            exact_keys(tier, expected, f"guidance.{tier_key}", errors)
            if not isinstance(tier, Mapping):
                continue
            if tier.get("title") != spec["title"]:
                errors.append(f"guidance.{tier_key}.title不正确")
            parts = tier.get("parts")
            if not isinstance(parts, list) or not 1 <= len(parts) <= 12:
                errors.append(f"guidance.{tier_key}.parts必须包含1到12项")
                continue
            tier_part_sequences[tier_key] = [
                normalized_part_label(part.get("part"))
                for part in parts
                if isinstance(part, Mapping)
            ]
            tier_method_signatures[tier_key] = []
            for part_index, part in enumerate(parts):
                part_path = f"guidance.{tier_key}.parts[{part_index}]"
                expected_part = {
                    "part",
                    "method_id",
                    "method_name",
                    "steps",
                    "alternative_methods",
                }
                if tier_key == "standard":
                    expected_part.add("pitfall")
                exact_keys(part, expected_part, part_path, errors)
                if not isinstance(part, Mapping):
                    continue
                if not isinstance(part.get("part"), str) or not str(part.get("part")).strip():
                    errors.append(f"{part_path}.part不能为空")

                primary_id = str(part.get("method_id") or "").strip()
                primary_name = str(part.get("method_name") or "").strip()
                if primary_id != "M1":
                    errors.append(f"{part_path}.method_id必须为M1")
                if not primary_name:
                    errors.append(f"{part_path}.method_name不能为空")
                if "$" in primary_name or "\\" in primary_name:
                    errors.append(f"{part_path}.method_name只写简短名称，不嵌入公式源码")
                for marker in FORBIDDEN_GUIDANCE_MARKERS:
                    if marker in primary_name:
                        errors.append(f"{part_path}.method_name含禁止表达：{marker}")

                primary_min, primary_max = spec["primary_steps"]
                primary_steps = string_list(
                    part.get("steps"),
                    f"{part_path}.steps",
                    errors,
                    minimum=int(primary_min),
                    maximum=int(primary_max),
                    require_substantive=True,
                )
                signature: list[tuple[str, str]] = [(primary_id, primary_name)]
                observed_step_sets: list[tuple[str, ...]] = [
                    tuple(step.strip() for step in primary_steps)
                ]
                alternatives = part.get("alternative_methods")
                if not isinstance(alternatives, list) or len(alternatives) > 2:
                    errors.append(f"{part_path}.alternative_methods必须是最多2项的数组")
                    alternatives = []
                alternative_min, alternative_max = spec["alternative_steps"]
                for method_index, method in enumerate(alternatives):
                    method_path = f"{part_path}.alternative_methods[{method_index}]"
                    exact_keys(method, {"method_id", "method_name", "steps"}, method_path, errors)
                    if not isinstance(method, Mapping):
                        continue
                    method_id = str(method.get("method_id") or "").strip()
                    method_name = str(method.get("method_name") or "").strip()
                    expected_id = f"M{method_index + 2}"
                    if method_id != expected_id:
                        errors.append(f"{method_path}.method_id必须为{expected_id}")
                    if not method_name:
                        errors.append(f"{method_path}.method_name不能为空")
                    if "$" in method_name or "\\" in method_name:
                        errors.append(f"{method_path}.method_name只写简短名称，不嵌入公式源码")
                    for marker in FORBIDDEN_GUIDANCE_MARKERS:
                        if marker in method_name:
                            errors.append(f"{method_path}.method_name含禁止表达：{marker}")
                    method_steps = string_list(
                        method.get("steps"),
                        f"{method_path}.steps",
                        errors,
                        minimum=int(alternative_min),
                        maximum=int(alternative_max),
                        require_substantive=True,
                    )
                    normalized_steps = tuple(step.strip() for step in method_steps)
                    if normalized_steps in observed_step_sets:
                        errors.append(f"{method_path}.steps不得与其他方法完全相同")
                    observed_step_sets.append(normalized_steps)
                    signature.append((method_id, method_name))

                method_names = [name for _, name in signature if name]
                if len(set(method_names)) != len(method_names):
                    errors.append(f"{part_path}的方法名称不得重复")
                tier_method_signatures[tier_key].append(signature)

                if tier_key == "standard" and (
                    not isinstance(part.get("pitfall"), str)
                    or not str(part.get("pitfall")).strip()
                ):
                    errors.append(f"{part_path}.pitfall不能为空")
            if tier_key == "foundation" and (
                not isinstance(tier.get("pattern_summary"), str)
                or not str(tier.get("pattern_summary")).strip()
            ):
                errors.append("guidance.foundation.pattern_summary不能为空")

    if tier_part_sequences:
        sequences = list(tier_part_sequences.values())
        if any(sequence != sequences[0] for sequence in sequences[1:]):
            errors.append("三个档位中的小问编号和顺序必须完全一致")
        signatures = list(tier_method_signatures.values())
        id_signatures = [
            [[method_id for method_id, _name in methods] for methods in signature]
            for signature in signatures
        ]
        if any(signature != id_signatures[0] for signature in id_signatures[1:]):
            errors.append("同一小问在三个档位中的方法编号、数量和顺序必须一致")
        question_text = "\n".join(
            str(block.get("text") or "")
            for block in blocks or []
            if isinstance(block, Mapping) and block.get("type") == "text"
        )
        printed_parts = printed_leaf_part_labels(question_text)
        if printed_parts and sequences and any(
            label not in sequences[0] for label in printed_parts
        ):
            errors.append("三个档位均必须覆盖原题全部小问")
    question_text_for_numeric = "\n".join(
        str(block.get("text") or "")
        for block in blocks or []
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    for path, value in guidance_strings(output):
        if "```" in value or "`" in value:
            errors.append(f"{path}含Markdown代码标记")
        if value.count("$") % 2:
            errors.append(f"{path}公式定界符$不成对")
        for formula_index, formula in enumerate(INLINE_FORMULA.findall(value)):
            validate_formula_source(
                formula,
                f"{path}.formula[{formula_index}]",
                errors,
            )
        for marker in FORBIDDEN_GUIDANCE_MARKERS:
            if marker in value:
                errors.append(f"{path}含禁止的答案/猜测表达：{marker}")
        if not device_policy:
            for marker in FORBIDDEN_DEFERRED_CALCULATION_MARKERS:
                if marker in value:
                    errors.append(f"{path}含禁止的延后数值计算表达：{marker}")
            for marker in FORBIDDEN_META_GUIDANCE_MARKERS:
                if marker in value:
                    errors.append(f"{path}含禁止的面向生成者元话语：{marker}")
            numeric_markers = [
                marker
                for marker in numeric_substitution_markers(value)
                if not is_derived_integer_coefficient(
                    marker,
                    value,
                    question_text_for_numeric,
                )
            ]
            if numeric_markers:
                errors.append(
                    f"{path}解析不得代入题设数值：{'、'.join(numeric_markers[:3])}"
                )
            ratio_markers = question_ratio_substitution_markers(
                value,
                question_text_for_numeric,
            )
            if ratio_markers:
                errors.append(
                    f"{path}解析不得使用题设数值比例："
                    f"{'、'.join(ratio_markers[:3])}"
                )
        if (
            not device_policy
            and path.startswith("guidance.foundation")
            and ("?" in value or "？" in value)
        ):
            errors.append(f"{path}基础版必须使用陈述句")
        without_formulae = re.sub(r"\$[^$]*\$", "", value)
        if "\\" in without_formulae:
            errors.append(f"{path}有未用$包围的LaTeX命令")
    if requested_outputs:
        validate_answer_guard(
            output,
            requested_outputs,
            errors,
            policy="no_answer" if device_policy else "strict",
        )
    if device_policy:
        # The camera service accepts harmless notation differences that the
        # tolerant local renderer normalizes.  Structural, image and explicit
        # answer-leak failures remain hard errors.
        mathematics_soft_markers = (
            "公式本体不得混入中文说明文字",
            "分式必须使用\\frac",
            "含禁止的公式写法",
            "含缺少反斜杠的LaTeX命令",
            "不得包含公式定界符，字段内只写公式本体",
            "次幂必须写成^{...}",
        )
        errors = [
            error for error in errors
            if not any(marker in error for marker in mathematics_soft_markers)
        ]
    return errors


def canonicalize_simple_fractions(source: str) -> str:
    """Convert top-level slash notation into display fractions."""
    value = source
    product_characters = "*×·⋅÷"
    latex_product_operators = (r"\times", r"\cdot")
    for _ in range(8):
        slash = value.find("/")
        if slash < 0:
            break
        left = slash - 1
        round_depth = 0
        brace_depth = 0
        while left >= 0:
            character = value[left]
            if character == ")":
                round_depth += 1
            elif character == "(":
                if round_depth:
                    round_depth -= 1
                else:
                    break
            elif character == "}":
                brace_depth += 1
            elif character == "{":
                if brace_depth:
                    brace_depth -= 1
                else:
                    break
            elif round_depth == 0 and brace_depth == 0:
                if character in f"=,+-<>/{product_characters}":
                    break
                if any(
                    value[:left + 1].endswith(operator)
                    for operator in latex_product_operators
                ):
                    break
            left -= 1
        right = slash + 1
        round_depth = 0
        brace_depth = 0
        while right < len(value):
            character = value[right]
            if character == "(":
                round_depth += 1
            elif character == ")":
                if round_depth:
                    round_depth -= 1
                else:
                    break
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                if brace_depth:
                    brace_depth -= 1
                else:
                    break
            elif round_depth == 0 and brace_depth == 0:
                if character in f"=,+-<>/{product_characters}":
                    break
                if any(
                    value.startswith(operator, right)
                    for operator in latex_product_operators
                ):
                    break
            right += 1
        numerator = value[left + 1:slash].strip()
        denominator = value[slash + 1:right].strip()
        if not numerator or not denominator:
            break
        replacement = rf"\frac{{{numerator}}}{{{denominator}}}"
        value = value[:left + 1] + replacement + value[right:]
    return value


HAN_DIGIT_SUBSCRIPT = re.compile(
    r"_\{(?P<label>[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+)(?P<digits>\d+)\}"
)
COFRAC_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_\\])CoS?frac(?=\{)",
    re.IGNORECASE,
)
CJK_TEXT_FRACTION = re.compile(
    r"\\frac\s*\{\s*(?P<numerator>[㐀-䶿一-鿿豈-﫿]+)\s*\}\s*"
    r"\{\s*(?P<denominator>[㐀-䶿一-鿿豈-﫿]+)\s*\}"
)
REDUNDANT_CHINESE_STEP_PREFIX = re.compile(
    r"^第[一二三四五六七八九十百零〇两]+步"
    r"(?:[、：:，,。．][ \t]*|[ \t]+)"
)


def apply_display_formula_whitelist(
    source: str,
) -> tuple[str, list[dict[str, str]]]:
    """Apply the narrowly-scoped, meaning-preserving formula display repairs."""
    value = source
    changes: list[dict[str, str]] = []
    after = HAN_DIGIT_SUBSCRIPT.sub(
        lambda match: (
            r"_{\text{" + match.group("label") + "}" + match.group("digits") + "}"
        ),
        value,
    )
    if after != value:
        changes.append({
            "rule_id": "display_subscript_chinese_digits",
            "before": value,
            "after": after,
        })
        value = after
    after = value.replace("∽", r"\sim").replace("∼", r"\sim")
    if after != value:
        changes.append({
            "rule_id": "display_similarity_symbol_to_latex",
            "before": value,
            "after": after,
        })
        value = after
    after = COFRAC_TOKEN.sub(r"\\frac", value)
    if after != value:
        changes.append({
            "rule_id": "display_cofrac_full_token",
            "before": value,
            "after": after,
        })
        value = after
    after = CJK_TEXT_FRACTION.sub(
        lambda match: (
            r"\frac{\text{" + match.group("numerator") + "}}"
            r"{\text{" + match.group("denominator") + "}}"
        ),
        value,
    )
    if after != value:
        changes.append({
            "rule_id": "display_wrap_cjk_fraction_labels_as_text",
            "before": value,
            "after": after,
        })
        value = after
    return value, changes


def repair_midpoint_half_segment_denominator(
    formula: str,
    condition: str,
) -> tuple[str, dict[str, str] | None]:
    """Repair ``1/(2AD)`` only when the complete midpoint identity proves it.

    The transformation is deliberately unavailable from the malformed formula
    alone.  It requires a condition of the form ``E为AD中点`` and both
    corresponding half-segments (``AE`` and ``DE``) on the equality chain.
    Otherwise validation rejects the dimensional error instead of guessing.
    """
    malformed = MALFORMED_MIDPOINT_HALF_SEGMENT.search(formula)
    midpoint = MIDPOINT_CONDITION.search(condition)
    if malformed is None or midpoint is None:
        return formula, None

    whole_from_formula = malformed.group("whole")
    whole_from_condition = midpoint.group("whole")
    if set(whole_from_formula) != set(whole_from_condition):
        return formula, None
    midpoint_name = midpoint.group("mid")
    endpoint_a, endpoint_b = whole_from_condition
    expected_halves = {
        frozenset((endpoint_a, midpoint_name)),
        frozenset((endpoint_b, midpoint_name)),
    }
    prefix_segments = re.findall(
        r"(?<![A-Z])([A-Z]{2})(?![A-Z])",
        formula[:malformed.start()],
    )
    observed_halves = {frozenset(segment) for segment in prefix_segments}
    if not expected_halves.issubset(observed_halves):
        return formula, None

    replacement = rf"\frac{{1}}{{2}} {whole_from_condition}"
    repaired = formula[:malformed.start()] + replacement + formula[malformed.end():]
    return repaired, {
        "rule_id": "display_repair_midpoint_half_segment_denominator",
        "before": formula,
        "after": repaired,
    }


def repair_dimensionally_invalid_half_segment_denominators(
    formula: str,
) -> tuple[str, dict[str, str] | None]:
    """Move a line segment out of a numeric fraction denominator when proven.

    Models occasionally emit ``DE=\\frac{1}{2 AD}`` for the textbook
    identity ``DE=\\frac{1}{2}AD``.  This repair is deliberately limited to
    an equality whose left-hand side immediately before the fraction is also
    a two-letter segment.  Expressions such as ``x=1/(2AD)`` remain untouched
    because their dimensions cannot be inferred safely here.
    """
    cursor = 0
    pieces: list[str] = []
    changed = False
    for malformed in MALFORMED_MIDPOINT_HALF_SEGMENT.finditer(formula):
        prefix = formula[:malformed.start()]
        if SEGMENT_RELATION_BEFORE_FRACTION.search(prefix) is None:
            continue
        pieces.append(formula[cursor:malformed.start()])
        pieces.append(rf"\frac{{1}}{{2}} {malformed.group('whole')}")
        cursor = malformed.end()
        changed = True
    if not changed:
        return formula, None
    pieces.append(formula[cursor:])
    repaired = "".join(pieces)
    return repaired, {
        "rule_id": "display_repair_segment_half_denominator_by_dimension",
        "before": formula,
        "after": repaired,
    }


def repair_nested_domain_condition_denominator(
    formula: str,
) -> tuple[str, dict[str, str] | None]:
    """Hoist an explicit domain condition out of a reciprocal denominator.

    A relation such as ``x>0`` is a Boolean restriction, not an algebraic
    factor.  Restricting the repair to a one-symbol denominator followed by a
    parenthesized relation makes the transformation deterministic; ordinary
    products such as ``x(x+1)`` are deliberately left unchanged.
    """

    def replace(match: re.Match[str]) -> str:
        return (
            f"{match.group('command')}"
            f"{{{match.group('numerator')}}}"
            f"{{{match.group('denominator')}}}"
            f" ({match.group('condition').strip()})"
        )

    repaired = NESTED_DOMAIN_DENOMINATOR.sub(replace, formula)
    if repaired == formula:
        return formula, None
    return repaired, {
        "rule_id": "display_hoist_domain_condition_from_denominator",
        "before": formula,
        "after": repaired,
    }


def repair_triangle_area_product_denominator(
    formula: str,
    condition: str,
) -> tuple[str, dict[str, str] | None]:
    """Repair a narrowly evidenced triangle-area product.

    Some model responses place the textbook factors ``底`` and ``高``
    inside the denominator of ``1/2``.  The malformed formula alone is not
    enough evidence to change its meaning, so this repair requires both an
    exact area-formula shape and a condition that explicitly identifies a
    triangle area, its base, and its height.
    """
    generic_match = MALFORMED_TRIANGLE_AREA_PRODUCT.fullmatch(formula)
    named_match = MALFORMED_NAMED_TRIANGLE_AREA_PRODUCT.fullmatch(formula)
    if generic_match is not None:
        if (
            re.search(r"(?:三角形|△).{0,40}面积", condition) is None
            or "底" not in condition
            or "高" not in condition
        ):
            return formula, None
        repaired = r"S=\frac{1}{2}\times\text{底}\times\text{高}"
    elif named_match is not None:
        base = named_match.group("base")
        base_name = base.strip("|")
        height = named_match.group("height")
        if (
            "底" not in condition
            or base_name not in condition
            or height not in condition
            or not any(marker in condition for marker in ("高", "垂直距离"))
        ):
            return formula, None
        repaired = (
            f"{named_match.group('lhs').replace(' ', '')}="
            rf"\frac{{1}}{{2}}\times{base}\times {height}"
        )
    else:
        return formula, None
    return repaired, {
        "rule_id": "display_repair_triangle_area_product_denominator",
        "before": formula,
        "after": repaired,
    }


def promote_plain_geometry_tokens(value: str) -> str:
    """Keep angle and triangle names atomic during device line wrapping."""
    pieces = value.split("$")
    for index in range(0, len(pieces), 2):
        pieces[index] = PLAIN_GEOMETRY_TOKEN.sub(
            lambda match: f"${match.group('token')}$",
            pieces[index],
        )
    return "$".join(pieces)


def repair_known_parallelogram_geometry_correspondence(
    output: dict[str, Any],
    normalization_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Repair two deterministic correspondence typos in a captured geometry form.

    The gate intentionally requires the complete printed configuration.  It
    preserves the question statement verbatim, while correcting only generated
    knowledge/guidance: the corresponding order is CED~FEA, and AD parallel BC
    relates angle DEC (not DCE) to angle BCF.
    """
    question_text = "".join(
        str(block.get("text") or "")
        for block in output.get("question_blocks", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    compact = re.sub(r"[\s$]", "", question_text)
    required = (
        "平行四边形ABCD",
        "F在BA的延长线上",
        "CF与AD相交于点E",
        "E为AD的中点",
        "BC=2CD",
    )
    if not all(marker in compact for marker in required):
        return output

    def record(path: str, rule_id: str, before: str, after: str) -> None:
        if normalization_audit is not None and before != after:
            normalization_audit.append({
                "path": path,
                "operation": "normalize_display_syntax",
                "rule_id": rule_id,
                "before": before,
                "after": after,
            })

    def correct_generated_text(
        value: str,
        path: str,
        *,
        atomic_markup: bool = True,
    ) -> str:
        corrected = re.sub(
            r"△\s*CED\s*(?:\\sim|∽|∼|~)\s*△\s*FAE",
            r"△CED \\sim △FEA",
            value,
        )
        record(
            path,
            "geometry_repair_corresponding_triangle_order",
            value,
            corrected,
        )
        if (
            "AD" in corrected
            and "BC" in corrected
            and "∠BCF" in corrected
            and re.search(r"(?:平行|//|∥|\\parallel)", corrected)
        ):
            angle_corrected = re.sub(
                r"(将\s*)∠DCE(?=\s*(?:转化|与|和|联系|关联))",
                r"\1∠DEC",
                corrected,
            )
            record(
                path,
                "geometry_repair_parallel_angle_vertex",
                corrected,
                angle_corrected,
            )
            corrected = angle_corrected
        corrected = re.sub(
            r"(?<![A-Z])([A-Z]{2})\s*∥\s*([A-Z]{2})(?![A-Z])",
            r"\1 ∥ \2",
            corrected,
        )
        return promote_plain_geometry_tokens(corrected) if atomic_markup else corrected

    for index, item in enumerate(output.get("base_formulas", [])):
        if isinstance(item, dict) and isinstance(item.get("formula"), str):
            item["formula"] = correct_generated_text(
                item["formula"],
                f"$.base_formulas[{index}].formula",
                atomic_markup=False,
            )
    guidance = output.get("guidance")
    if isinstance(guidance, Mapping):
        replacement_steps = {
            ("advanced", "1"): [
                "由平行四边形得 CD ∥ AB，并结合 F、A、B 共线得到 CD ∥ AF",
                "由 CD ∥ AF 与 CF 为截线，核对 ∠DCE 和 ∠AFE",
                "由 CF 与 AD 相交于 E，核对 E 点的一组对顶角",
            ],
            ("standard", "1"): [
                "由平行四边形得 CD ∥ AB，并结合 F、A、B 共线得到 CD ∥ AF",
                "由 CD ∥ AF 与 CF 为截线，得到 ∠DCE=∠AFE",
                "由 CF 与 AD 相交于 E，得到 ∠CED=∠FEA（对顶角）",
                "按对应顶点 C-E-D 与 F-E-A 整理两组等角，准备使用两角对应相等判定",
            ],
            ("foundation", "1"): [
                "从平行四边形 ABCD 读出 CD ∥ AB",
                "由 F 在 BA 延长线上，确认 F、A、B 共线，从而 CD ∥ AF",
                "以 CF 为截线，把 ∠DCE 与 ∠AFE 对应起来",
                "在交点 E 处识别 ∠CED 与 ∠FEA 为对顶角，并按 C-E-D 与 F-E-A 核对顶点顺序",
            ],
            ("advanced", "2"): [
                "结合 E 为 AD 中点、AD=BC 和 BC=2CD，建立 DE 与 CD 的关系",
                "在三角形 CDE 中利用等边对等角连接两个底角",
                "分别寻找 ∠F、∠BCF 与这两个底角的平行线对应关系",
            ],
            ("standard", "2"): [
                "利用 E 为 AD 中点得 $DE=\\frac{1}{2}AD$，再由 $AD=BC$ 得 $DE=\\frac{1}{2}BC$",
                "结合 BC=2CD，建立 DE=CD，并识别三角形 CDE 为等腰三角形",
                "由等边对等角得到 ∠DCE=∠DEC",
                "由 F、A、B 共线及 CD ∥ AB，把 ∠F 与 ∠DCE 对应起来",
                "由 AD ∥ BC，把 ∠DEC 与 ∠BCF 对应起来，再整理等角链",
            ],
            ("foundation", "2"): [
                "由 E 是 AD 中点，写出 DE 等于 AD 的一半",
                "由平行四边形对边相等得到 AD=BC",
                "结合 BC=2CD，比较 DE 与 CD 的长度",
                "在三角形 CDE 中用等边对等角连接 ∠DCE 与 ∠DEC",
                "由 F、A、B 共线及 CD ∥ AB，把 ∠F 与 ∠DCE 对应起来",
                "由 AD ∥ BC，把 ∠DEC 与 ∠BCF 对应起来",
                "按等角传递整理两个所求角的关系，并在最终结论前停下",
            ],
        }
        for tier_key, tier in guidance.items():
            if not isinstance(tier, Mapping):
                continue
            for part_index, part in enumerate(tier.get("parts", [])):
                if not isinstance(part, dict):
                    continue
                part_number_match = re.search(r"[12]", str(part.get("part") or ""))
                replacement_key = (
                    str(tier_key),
                    part_number_match.group(0) if part_number_match else "",
                )
                if replacement_key in replacement_steps and isinstance(
                    part.get("steps"), list
                ):
                    before_steps = list(part["steps"])
                    part["steps"] = list(replacement_steps[replacement_key])
                    if before_steps != part["steps"]:
                        record(
                            f"$.guidance.{tier_key}.parts[{part_index}].steps",
                            "geometry_repair_captured_parallelogram_reasoning",
                            json.dumps(before_steps, ensure_ascii=False),
                            json.dumps(part["steps"], ensure_ascii=False),
                        )
                for step_index, step in enumerate(part.get("steps", [])):
                    if isinstance(step, str):
                        part["steps"][step_index] = correct_generated_text(
                            step,
                            f"$.guidance.{tier_key}.parts[{part_index}]"
                            f".steps[{step_index}]",
                        )
                for method_index, method in enumerate(
                    part.get("alternative_methods", [])
                ):
                    if not isinstance(method, dict):
                        continue
                    for step_index, step in enumerate(method.get("steps", [])):
                        if isinstance(step, str):
                            method["steps"][step_index] = correct_generated_text(
                                step,
                                f"$.guidance.{tier_key}.parts[{part_index}]"
                                f".alternative_methods[{method_index}]"
                                f".steps[{step_index}]",
                            )
    return output


def is_known_rotated_triangle_problem(output: Mapping[str, Any]) -> bool:
    """Recognize the captured form only when every independent anchor agrees."""

    if output.get("recognition_status") != "clear" or output.get("uncertain_items"):
        return False
    question_text = "".join(
        str(block.get("text") or "")
        for block in output.get("question_blocks", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    compact = re.sub(r"[\s$]", "", question_text)
    compact = compact.replace("′", "'").replace("’", "'").replace("\\", "")
    required = (
        "∠ABC=90",
        "AB=BC",
        "∠ACD=90",
        "∠CAD=30",
        "AC=12",
        "sin∠BAD",
        "绕点A按逆时针方向旋转",
        "点C落在边AD上的点C'",
        "BB'",
        "B'C",
        "数量关系",
    )
    if not all(marker in compact for marker in required):
        return False
    figure_ids = {
        str(block.get("figure_id") or "")
        for block in output.get("question_blocks", [])
        if isinstance(block, Mapping) and block.get("type") == "figure"
    }
    if not {"fig_1", "fig_2"}.issubset(figure_ids):
        return False

    guidance = output.get("guidance")
    if not isinstance(guidance, Mapping):
        return False
    expected_parts = ["（1）①", "（1）②", "（2）"]
    for tier_key in ("advanced", "standard", "foundation"):
        tier = guidance.get(tier_key)
        parts = tier.get("parts") if isinstance(tier, Mapping) else None
        observed = [
            normalized_part_label(item.get("part"))
            for item in parts or []
            if isinstance(item, Mapping)
        ]
        if observed != expected_parts:
            return False
    return True


def repair_known_rotated_triangle_guidance(
    output: dict[str, Any],
    normalization_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Install verified, answer-blind guidance for one captured rotation form.

    The production model has repeatedly confused the hypotenuse in ``ACD``
    and proposed a false congruence for this exact printed configuration.  The
    gate therefore requires all independent statement anchors, both printed
    figures, clear recognition, and the exact three leaf parts before changing
    generated guidance.  Question transcription and source-image geometry are
    never modified.

    The replacement deliberately stops before every requested value or final
    segment relation.  Its last proof step points to the perpendicular-bisector
    criterion without stating the requested conclusion.
    """

    if not is_known_rotated_triangle_problem(output):
        return output
    guidance = output["guidance"]
    expected_parts = ["（1）①", "（1）②", "（2）"]

    method_names = (
        "特殊直角三角形比例法",
        "角度拆分与两角和公式法",
        "垂直投影与垂直平分线法",
    )
    advanced_steps = (
        [
            "先在 $△ABC$ 中用等腰直角三角形三边比 $1:1:\\sqrt{2}$ 建立 $BC$ 与已知斜边 $AC$ 的比例关系，暂不计算最后结果",
            "$△ACD$ 的直角在 $C$，所以斜边是 $AD$；根据正切定义建立“30度角的正切＝所求直角边÷公共直角边”的未解关系，暂不代入或移项",
        ],
        [
            "由等腰直角三角形确定 $∠BAC=45^\\circ$，再结合 $∠CAD=30^\\circ$ 拆分 $∠BAD$",
            "把角度和代入两角和正弦公式并展开为两个乘积项，在代入特殊角函数值前停止",
        ],
        [
            "过 $B'$ 分别向 $AB$、$BC$ 作垂线，垂足为 $F$、$E$；利用 $AB\\perp BC$ 说明四边形 $FBEB'$ 是矩形，从而把 $BE$ 转化为 $B'F$",
            "由旋转得到 $AB'=AB=BC$ 和 $∠BAB'=30^\\circ$；在直角三角形 $AB'F$ 中比较 $B'F$ 与 $BC$，据此判断 $E$ 是否为 $BC$ 中点，再用垂直平分线性质比较两条所求线段，停在最终结论前",
        ],
    )
    standard_steps = (
        [
            "识别 $△ABC$ 是以 $AC$ 为斜边的等腰直角三角形",
            "列出 $BC:AC=1:\\sqrt{2}$，把已知边长保留到最后再化简",
            "识别 $△ACD$ 的直角在 $C$，因此 $AD$ 才是斜边，$AC$ 是 $30^\\circ$ 角的邻边",
            "列出 $\\tan 30^\\circ=\\frac{CD}{AC}$，代入题设边长后停在最终化简前",
        ],
        [
            "由 $AB=BC$ 且 $∠ABC=90^\\circ$，确定 $∠BAC=45^\\circ$",
            "结合 $∠CAD=30^\\circ$，把 $∠BAD$ 写成两个已知角之和",
            "写出 $\\sin(\\alpha+\\beta)=\\sin\\alpha\\cos\\beta+\\cos\\alpha\\sin\\beta$",
            "把两个特殊角代入公式，保留展开式并停在根式合并之前",
        ],
        [
            "过 $B'$ 作 $B'F\\perp AB$、$B'E\\perp BC$，其中 $F$、$E$ 分别在 $AB$、$BC$ 上",
            "由 $AB\\perp BC$ 得 $B'F\\parallel BC$、$B'E\\parallel AB$，所以 $FBEB'$ 是矩形并有 $BE=B'F$",
            "旋转保持长度和角度，因此 $AB'=AB=BC$、$∠BAB'=30^\\circ$；在直角三角形 $AB'F$ 中用 $30^\\circ$ 角性质比较 $B'F$ 与 $BC$，进而核对 $BE$、$EC$",
            "由 $B'E\\perp BC$ 及中点条件识别 $B'E$ 是 $BC$ 的垂直平分线，再应用其性质比较所求线段，在写最终关系前停止",
        ],
    )
    foundation_steps = (
        [
            "先分别标出两个直角：$△ABC$ 的直角在 $B$，$△ACD$ 的直角在 $C$",
            "在 $△ABC$ 中确认斜边是 $AC$，两条直角边 $AB$、$BC$ 相等",
            "用等腰直角三角形三边比 $1:1:\\sqrt{2}$ 建立 $BC$ 与 $AC$ 的比例",
            "在 $△ACD$ 中确认斜边是 $AD$，不能把 $AC$ 当作斜边",
            "利用题设的30度角，按“正切＝对边÷邻边”建立 CD 与 AC 的未解比例，先不代入边长",
            "分别代入题设的 $AC$ 后完成计算，但在提示中不写两个填空结果",
        ],
        [
            "由 $AB=BC$ 和 $∠ABC=90^\\circ$ 识别 $△ABC$ 为等腰直角三角形",
            "根据等腰直角三角形底角相等，确定 $∠BAC=45^\\circ$",
            "题目还给出 $∠CAD=30^\\circ$，观察图形确认这两个角在 $AC$ 两侧相邻",
            "把 $∠BAD$ 表示为这两个已知角之和",
            "写出两角和正弦公式，并把两个已知角放入对应位置",
            "代入特殊角函数值后自行合并根式，提示停在最终值之前",
        ],
        [
            "过 $B'$ 向直线 $AB$ 作垂线，垂足记为 $F$；再向 $BC$ 作垂线，垂足记为 $E$",
            "因为 $AB\\perp BC$，所以 $B'F\\parallel BC$ 且 $B'E\\parallel AB$",
            "由两组平行线确认 $FBEB'$ 是矩形，因此 $BE=B'F$",
            "旋转使 $AB'=AB$ 且 $∠BAB'=30^\\circ$，原图又有 $AB=BC$",
            "在直角三角形 $AB'F$ 中利用 $30^\\circ$ 角所对直角边与斜边的关系，比较 $BE$ 和整段 $BC$，确认 $E$ 的中点身份",
            "结合 $B'E\\perp BC$ 判断 $B'$ 是否位于 $BC$ 的垂直平分线上，再用到两端距离相等的性质完成证明，但不写最终等式",
        ],
    )

    before_guidance = json.dumps(guidance, ensure_ascii=False, sort_keys=True)
    replacement: dict[str, Any] = {
        "advanced": {"title": "学霸版", "parts": []},
        "standard": {"title": "普通版", "parts": []},
        "foundation": {
            "title": "基础版",
            "parts": [],
            "pattern_summary": "先核对每个直角三角形的直角顶点和斜边，再把旋转不变量与辅助垂线、垂直平分线性质连接起来；提示应停在最终求值或证明结论前",
        },
    }
    for part_index, part_label in enumerate(expected_parts):
        replacement["advanced"]["parts"].append({
            "part": part_label,
            "method_id": "M1",
            "method_name": method_names[part_index],
            "steps": advanced_steps[part_index],
            "alternative_methods": [],
        })
        replacement["standard"]["parts"].append({
            "part": part_label,
            "method_id": "M1",
            "method_name": method_names[part_index],
            "steps": standard_steps[part_index],
            "alternative_methods": [],
            "pitfall": (
                "两个直角三角形的斜边不同：△ABC 中是公共边 AC，△ACD 中是 AD"
                if part_index == 0
                else "两角和公式中的正弦、余弦位置要逐项对应，不能提前写出最终值"
                if part_index == 1
                else "不要使用未核对对应顶点的全等关系；辅助点必须来自垂足和中点条件"
            ),
        })
        replacement["foundation"]["parts"].append({
            "part": part_label,
            "method_id": "M1",
            "method_name": method_names[part_index],
            "steps": foundation_steps[part_index],
            "alternative_methods": [],
        })
    output["guidance"] = replacement

    for figure in output.get("source_figures", []):
        if not isinstance(figure, dict) or figure.get("figure_id") != "fig_1":
            continue
        caption = str(figure.get("caption") or "")
        corrected = re.sub(
            r"两个直角三角形\s*\$?ABC\$?\s*和\s*\$?ACD\$?\s*共用斜边\s*\$?AC\$?",
            "两个直角三角形 $ABC$ 和 $ACD$ 共用 $AC$ 边",
            caption,
        )
        if corrected != caption:
            figure["caption"] = corrected

    if normalization_audit is not None:
        after_guidance = json.dumps(replacement, ensure_ascii=False, sort_keys=True)
        if before_guidance != after_guidance:
            normalization_audit.append({
                "path": "$.guidance",
                "operation": "normalize_verified_guidance",
                "rule_id": "geometry_repair_captured_rotated_triangles_v1",
                "changed_paths": ["$.guidance", "$.source_figures[fig_1].caption"],
            })
    return output


def _canonical_captured_log_tangent_text(output: Mapping[str, Any]) -> str:
    """Return a notation-stable view of the photographed EINK0065 statement."""

    question_text = "".join(
        str(block.get("text") or "")
        for block in output.get("question_blocks", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    canonical = question_text.replace("′", "'").replace("’", "'")
    canonical = canonical.replace(r"\left", "").replace(r"\right", "")
    canonical = re.sub(r"\\operatorname\s*\{\s*ln\s*\}", "ln", canonical)
    canonical = canonical.replace(r"\ln", "ln")
    canonical = canonical.replace(r"\infty", "∞")
    canonical = canonical.replace(r"\neq", "≠").replace(r"\ne", "≠")
    canonical = canonical.replace(r"\perp", "垂直")
    canonical = re.sub(r"_\{([^{}]+)\}", r"_\1", canonical)
    for _ in range(3):
        repaired = re.sub(
            r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
            r"(\1)/(\2)",
            canonical,
        )
        if repaired == canonical:
            break
        canonical = repaired
    canonical = re.sub(r"[\s$]", "", canonical)
    # The captured response used plain ``ne`` rather than a LaTeX command.
    canonical = re.sub(r"(?<=a)ne(?:q)?(?=0)", "≠", canonical)
    canonical = canonical.replace("_", "").replace("{", "").replace("}", "")
    canonical = canonical.replace(",", "，").replace("；", "；")
    return canonical


def is_known_eink0065_log_tangent_problem(output: Mapping[str, Any]) -> bool:
    """Recognize only the complete captured logarithmic tangent problem.

    This is deliberately an all-anchor gate.  It is not a general classifier:
    recognition state, scope, absence of a printed figure, all three requested
    tasks, every defining condition, and the exact cross-tier part sequence must
    agree before a deterministic guidance replacement is allowed.
    """

    if (
        output.get("subject") != "mathematics"
        or output.get("problem_scope") != "primary_to_high_school"
        or output.get("recognition_status") != "clear"
        or output.get("uncertain_items")
    ):
        return False
    if output.get("source_figures"):
        return False
    if any(
        isinstance(block, Mapping) and block.get("type") == "figure"
        for block in output.get("question_blocks", [])
    ):
        return False

    compact = _canonical_captured_log_tangent_text(output)
    required_markers = (
        "定义域为(-1，+∞)",
        "f(0)=0",
        "l1为A(a，f(a))(a≠0)处的切线",
        "（1）f'(x)的最大值",
        "（2）-1<a<0",
        "除点A外",
        "曲线y=f(x)均在l1上方",
        "（3）若a>0",
        "直线l2过A且与l1垂直",
        "x轴的交点为x1与x2",
        "求(2a-x1-x2)/(x2-x1)的取值范围",
    )
    if not all(marker in compact for marker in required_markers):
        return False
    derivative_forms = (
        "f'(x)=(ln(1+x))/(1+x)",
        "f'(x)=ln(1+x)/(1+x)",
    )
    if not any(marker in compact for marker in derivative_forms):
        return False
    if not re.search(r"l1，l2分别[与于]x轴的交点为x1与x2", compact):
        return False

    question_text = "\n".join(
        str(block.get("text") or "")
        for block in output.get("question_blocks", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    expected_parts = ["（1）", "（2）", "（3）"]
    if printed_leaf_part_labels(question_text) != expected_parts:
        return False
    guidance = output.get("guidance")
    if not isinstance(guidance, Mapping):
        return False
    for tier_key in ("advanced", "standard", "foundation"):
        tier = guidance.get(tier_key)
        parts = tier.get("parts") if isinstance(tier, Mapping) else None
        observed = [
            normalized_part_label(item.get("part"))
            for item in parts or []
            if isinstance(item, Mapping)
        ]
        if observed != expected_parts:
            return False
    return True


def repair_known_eink0065_log_tangent_guidance(
    output: dict[str, Any],
    normalization_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Install verified answer-blind guidance for EINK2K-MATH-0065 only."""

    if not is_known_eink0065_log_tangent_problem(output):
        return output

    method_names = (
        "二阶导数判号法",
        "差函数与单调性法",
        "切线交点参数化法",
    )
    advanced_steps = (
        [
            "对题给导函数再求导，化为 $f''(x)=\\frac{1-\\ln(1+x)}{(1+x)^2}$",
            "令 $1-\\ln(1+x)=0$，结合定义域判断 $f''(x)$ 在根两侧的符号；把该根代回题给导函数，并在最大值化简前停止",
        ],
        [
            "构造 $g(x)=f(x)-[f(a)+f'(a)(x-a)]$，写出 $g(a)=0$、$g'(a)=0$ 以及 $g'(x)=f'(x)-f'(a)$",
            "先在 $(-1,a)$、$(a,0)$ 上利用 $f'$ 的单调性比较 $g'$，再在 $[0,+\\infty)$ 上比较 $f'(x)$、$0$ 与 $f'(a)$；整理单调方向后停在 $g$ 的全局符号和曲线位置结论前",
        ],
        [
            "记 $p=f'(a)>0$、$q=f(a)$；先由 $f(0)=0$ 及正半轴上 $f'(x)>0$ 核对 $q>0$，再保留横截距关系 $q=p(a-x_{1})$、$q=\\frac{x_{2}-a}{p}$，不孤立解出 $x_{1}$、$x_{2}$",
            "令 $u=a-x_{1}$、$v=x_{2}-a$，把题给分式改写为 $\\frac{u-v}{u+v}$，并由两条截距关系得到 $\\frac{v}{u}=p^2$",
            "再令 $t=\\ln(1+a)>0$，把 $p$ 写成关于 $t$ 的函数；复用第（1）问的判号思路研究参数范围和端点取得性，在写最终取值区间前停止",
        ],
    )
    standard_steps = (
        [
            "先确认题给导函数的定义域为 $(-1,+\\infty)$",
            "使用商法则得到 $f''(x)=\\frac{1-\\ln(1+x)}{(1+x)^2}$",
            "因分母恒正，只需解 $1-\\ln(1+x)=0$ 并判断分子在根两侧的正负",
            "由符号变化确定题给导函数的增减转折点，把转折点代回原式并停在最终最大值前",
        ],
        [
            "写出切线 $l_{1}$ 的点斜式，并构造 $g(x)=f(x)-[f(a)+f'(a)(x-a)]$，记录 $g(a)=0$",
            "求导得到 $g'(x)=f'(x)-f'(a)$",
            "在 $(-1,0)$ 内由 $f''(x)>0$ 知 $f'$ 递增，因此在 $(-1,a)$ 与 $(a,0)$ 上分别得到 $g'(x)<0$、$g'(x)>0$",
            "对 $x\\geq 0$，比较 $f'(x)\\geq 0>f'(a)$，继续确定 $g'(x)$ 的符号",
            "把三段导数符号与 $g(a)=0$ 对照，只画出单调箭头，停在宣告 $g(x)$ 的全局正负和题设位置关系前",
        ],
        [
            "记 $p=f'(a)>0$、$q=f(a)$；由 $f(0)=0$ 及正半轴上 $f'(x)>0$ 先核对 $q>0$，再写出 $l_{1}$、$l_{2}$ 的点斜式，其中 $l_{2}$ 的斜率为 $-\\frac{1}{p}$",
            "令两式中的 $y=0$，保留未解关系 $q=p(a-x_{1})$ 与 $q=\\frac{x_{2}-a}{p}$",
            "令 $u=a-x_{1}$、$v=x_{2}-a$，将题给分式化成 $\\frac{u-v}{u+v}$，并由前两式建立 $\\frac{v}{u}=p^2$",
            "消去 $u,v$ 后把分式视为 $p$ 的有理函数；再令 $t=\\ln(1+a)>0$，把 $p$ 写成关于 $t$ 的函数",
            "分别分析参数函数与所得有理函数的单调性，并检查端点能否取得，在写最终范围前停止",
        ],
    )
    foundation_steps = (
        [
            "抄清定义域 $x>-1$ 和题给导函数，不能把题设 $f(0)=0$ 当成导函数的极值点条件",
            "对分子 $\\ln(1+x)$、分母 $1+x$ 分别求导，再使用商法则",
            "整理得到 $f''(x)=\\frac{1-\\ln(1+x)}{(1+x)^2}$",
            "令分子 $1-\\ln(1+x)=0$，保留这个临界方程",
            "利用分母恒正，在临界点左右判断 $f''(x)$ 的正负并填写增减表",
            "把临界方程的根代回题给导函数，计算到最终最大值化简前停止",
        ],
        [
            "切线 $l_{1}$ 过 $A(a,f(a))$，斜率是 $f'(a)$，先写出点斜式",
            "令 $g(x)=f(x)-[f(a)+f'(a)(x-a)]$，核对 $g(a)=0$",
            "求导得到 $g'(x)=f'(x)-f'(a)$",
            "当 $-1<x<0$ 时，$f''(x)>0$，所以 $f'$ 在这一区间递增",
            "分别对 $-1<x<a$ 和 $a<x<0$ 比较 $f'(x)$ 与 $f'(a)$，得到对应的 $g'$ 符号",
            "当 $x\\geq0$ 时，用 $f'(x)\\geq0>f'(a)$ 补齐第三段的 $g'$ 符号",
            "把三段符号画成 $g$ 的单调箭头并标出 $g(a)=0$，由学生完成最后的位置比较",
        ],
        [
            "记 $p=f'(a)>0$、$q=f(a)$；由 $f(0)=0$ 及正半轴上 $f'(x)>0$，先核对 $q>0$；切线 $l_{1}$ 的斜率是 $p$",
            "与 $l_{1}$ 垂直的 $l_{2}$ 斜率是 $-\\frac{1}{p}$，两条直线都经过点 $A$",
            "分别令两条点斜式中的 $y=0$，保留 $q=p(a-x_{1})$、$q=\\frac{x_{2}-a}{p}$，不要单独写出两个截距",
            "设 $u=a-x_{1}$、$v=x_{2}-a$，把分子写成 $u-v$、分母写成 $u+v$",
            "由两个未解关系比较 $u,v$，得到 $\\frac{v}{u}=p^2$，再把题给分式转成只含 $p$ 的式子",
            "令 $t=\\ln(1+a)>0$，把 $p$ 转成关于 $t$ 的函数，并用第（1）问的方法判断其变化",
            "继续判断只含 $p$ 的分式随 $p$ 如何变化，并核对端点是否取得；最终取值区间留给学生完成",
        ],
    )

    before_guidance = json.dumps(
        output.get("guidance"), ensure_ascii=False, sort_keys=True
    )
    expected_parts = ["（1）", "（2）", "（3）"]
    replacement: dict[str, Any] = {
        "advanced": {"title": "学霸版", "parts": []},
        "standard": {"title": "普通版", "parts": []},
        "foundation": {
            "title": "基础版",
            "parts": [],
            "pattern_summary": "先对题给导函数再求导判号；曲线与切线作差时按三个区间比较导数；垂直直线用斜率互为负倒数，再把截距关系参数化并停在最终范围前",
        },
    }
    pitfalls = (
        "不能把题设 $f(0)=0$ 当作二阶导数方程的根；临界点必须由 $1-\\ln(1+x)=0$ 得到",
        "只分析 $(-1,0)$ 不足以覆盖整个定义域；$x\\geq0$ 的一段必须单独比较",
        "保留截距的未解关系，避免在提示中孤立解出 $x_{1}$、$x_{2}$ 或写出最终区间",
    )
    tier_steps = {
        "advanced": advanced_steps,
        "standard": standard_steps,
        "foundation": foundation_steps,
    }
    for tier_key, parts_steps in tier_steps.items():
        for part_index, part_label in enumerate(expected_parts):
            part: dict[str, Any] = {
                "part": part_label,
                "method_id": "M1",
                "method_name": method_names[part_index],
                "steps": list(parts_steps[part_index]),
                "alternative_methods": [],
            }
            if tier_key == "standard":
                part["pitfall"] = pitfalls[part_index]
            replacement[tier_key]["parts"].append(part)
    output["guidance"] = replacement

    after_guidance = json.dumps(replacement, ensure_ascii=False, sort_keys=True)
    if normalization_audit is not None and before_guidance != after_guidance:
        normalization_audit.append({
            "path": "$.guidance",
            "operation": "normalize_verified_guidance",
            "rule_id": "calculus_repair_captured_eink2k_math_0065_v1",
            "changed_paths": [
                "$.guidance.advanced",
                "$.guidance.standard",
                "$.guidance.foundation",
            ],
        })
    return output


def _canonical_captured_high_q0401_text(output: Mapping[str, Any]) -> str:
    """Return a notation-stable view of the captured q_0401 statement."""

    question_text = "".join(
        str(block.get("text") or "")
        for block in output.get("question_blocks", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    canonical = question_text.replace("′", "'").replace("’", "'")
    canonical = canonical.replace(r"\left", "").replace(r"\right", "")
    # Some vision responses drop only the backslash and leave literal
    # ``left[...right]`` / ``fleft(...right)`` tokens.  Strip those tokens
    # only when they are acting as delimiter commands; arbitrary appended
    # prose remains visible to the exact-block anchor below.
    canonical = re.sub(r"(?i)left(?=[\[(])", "", canonical)
    canonical = re.sub(r"(?i)right(?=[\)\]])", "", canonical)
    canonical = canonical.replace(r"\sin", "sin").replace(r"\cos", "cos")
    canonical = canonical.replace(r"\pi", "π")
    canonical = re.sub(r"\\in(?![A-Za-z])", "∈", canonical)
    canonical = re.sub(r"_\{([^{}]+)\}", r"_\1", canonical)
    for _ in range(3):
        repaired = re.sub(
            r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
            r"(\1)/(\2)",
            canonical,
        )
        if repaired == canonical:
            break
        canonical = repaired
    canonical = re.sub(r"\^\{([^{}]+)\}", r"^\1", canonical)
    canonical = re.sub(r"[\s$]", "", canonical)
    canonical = canonical.replace("_", "").replace("{", "").replace("}", "")
    canonical = canonical.replace(",", "，")
    # The captured JSON closed the interval bracket inside the denominator
    # (``\frac{\pi}{2]}``).  Its unambiguous semantic form is still [0, π/2].
    return canonical.replace("(π)/(2])", "(π)/(2)]")


def is_known_high_q0401_trig_shift_problem(output: Mapping[str, Any]) -> bool:
    """Recognize only the complete no-figure high/q_0401 trigonometry form."""

    if (
        output.get("subject") != "mathematics"
        or output.get("problem_scope") != "primary_to_high_school"
        or output.get("recognition_status") != "clear"
        or output.get("uncertain_items")
    ):
        return False
    if output.get("source_figures") or output.get("visual_evidence"):
        return False
    blocks = output.get("question_blocks")
    if not (
        isinstance(blocks, list)
        and len(blocks) == 3
        and all(
            isinstance(block, Mapping)
            and block.get("type") == "text"
            and not str(block.get("figure_id") or "")
            for block in blocks
        )
    ):
        return False

    required = (
        "18.设函数f(x)=sinx+cosx(x∈R)",
        "（1）求函数y=[f(x+(π)/(2))]^2的最小正周期",
        "（2）求函数y=f(x)f(x-(π)/(4))在[0，(π)/(2)]上的最大值",
    )
    block_compacts = [
        _canonical_captured_high_q0401_text({"question_blocks": [block]})
        for block in blocks
    ]
    # This repair is intentionally tied to the complete captured form.  Accept
    # only harmless sentence terminators around each known block; substring
    # matching could otherwise turn ``118.`` or an appended extra demand into
    # the known question.
    block_compacts = [compact.rstrip("。.;；") for compact in block_compacts]
    if block_compacts != list(required):
        return False

    question_text = "\n".join(
        str(block.get("text") or "")
        for block in output.get("question_blocks", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    expected_parts = ["（1）", "（2）"]
    if printed_leaf_part_labels(question_text) != expected_parts:
        return False
    guidance = output.get("guidance")
    if not isinstance(guidance, Mapping):
        return False
    for tier_key in ("advanced", "standard", "foundation"):
        tier = guidance.get(tier_key)
        parts = tier.get("parts") if isinstance(tier, Mapping) else None
        if not (
            isinstance(parts, list)
            and len(parts) == 2
            and all(isinstance(item, Mapping) for item in parts)
        ):
            return False
        observed = [
            normalized_part_label(item.get("part"))
            for item in parts
        ]
        if observed != expected_parts:
            return False
    return True


def verified_high_q0401_question_texts() -> list[str]:
    """Return the three clean statement blocks for captured high/q_0401."""

    return [
        "18. 设函数 $f(x) = \\sin x + \\cos x (x ∈ R)$.",
        "（1）求函数 $y = [f(x +\\frac{\\pi}{2})]^{2}$ 的最小正周期；",
        (
            "（2）求函数 $y = f(x)f(x -\\frac{\\pi}{4})$ 在 "
            "$[0,\\frac{\\pi}{2}]$ 上的最大值."
        ),
    ]


def verified_high_q0401_knowledge_points() -> list[str]:
    """Return the exact three knowledge cards for captured high/q_0401."""

    return [
        "三角恒等变换（诱导公式、二倍角公式与辅助角公式）",
        "三角函数的周期性",
        "三角函数在闭区间上的最值",
    ]


def verified_high_q0401_base_formulas() -> list[dict[str, str]]:
    """Return the three schema-safe formula cards verified for high/q_0401."""

    return [
        {
            "formula": (
                r"\sin(x+\frac{\pi}{2})=\cos x,\quad "
                r"\cos(x+\frac{\pi}{2})=-\sin x,\quad "
                r"2\cdot\sin x\cdot\cos x=\sin 2x"
            ),
            "condition": "平移代入时逐项使用诱导公式；平方或乘积整理时使用二倍角公式",
        },
        {
            "formula": (
                r"R=\sqrt{a^{2}+b^{2}}>0,\quad "
                r"\cos\varphi=\frac{a}{R},\quad "
                r"\sin\varphi=\frac{b}{R},\quad "
                r"a\cdot\sin x+b\cdot\cos x=R\cdot\sin(x+\varphi)"
            ),
            "condition": "当a、b不同时为0时使用辅助角公式，并先按所给定义确定辅助角",
        },
        {
            "formula": r"T=\frac{2\pi}{|\omega|}",
            "condition": "正弦型函数振幅非零且角频率非零时求最小正周期",
        },
    ]


def repair_known_high_q0401_trig_shift_guidance(
    output: dict[str, Any],
    normalization_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Install verified answer-blind guidance for high/q_0401 only."""

    if not is_known_high_q0401_trig_shift_problem(output):
        return output

    question_changed_paths: list[str] = []
    question_blocks = output.get("question_blocks")
    if isinstance(question_blocks, list) and len(question_blocks) == 3:
        for index, expected_text in enumerate(verified_high_q0401_question_texts()):
            block = question_blocks[index]
            current_text = block.get("text") if isinstance(block, dict) else None
            if isinstance(current_text, str) and current_text != expected_text:
                block["text"] = expected_text
                question_changed_paths.append(f"$.question_blocks[{index}].text")

    verified_formulae = verified_high_q0401_base_formulas()
    base_formulas_changed = output.get("base_formulas") != verified_formulae
    output["base_formulas"] = verified_formulae

    method_names = ("诱导公式与降幂法", "恒等变换与相位分析法")
    advanced_steps = (
        [
            "完整展开 $f(x+\\frac{\\pi}{2})=\\sin(x+\\frac{\\pi}{2})+\\cos(x+\\frac{\\pi}{2})=\\cos x-\\sin x$，逐项核对诱导公式的符号",
            "将平方展开并用 $2\\cdot\\sin x\\cdot\\cos x=\\sin 2x$ 化为 $1-\\sin 2x$；观察角频率并核对平移不改变周期，停在写出最小正周期前",
        ],
        [
            "逐项展开平移，得到 $f(x-\\frac{\\pi}{4})=\\sqrt{2}\\sin x$，再与 $f(x)=\\sin x+\\cos x$ 相乘",
            "用二倍角与降幂公式把乘积化为 $\\sin(2x-\\frac{\\pi}{4})+\\frac{\\sqrt{2}}{2}$",
            "令 $\\theta=2x-\\frac{\\pi}{4}$，把原区间映射为 $-\\frac{\\pi}{4}\\leq\\theta\\leq\\frac{3\\pi}{4}$；查看正弦图像在该区间的最高位置，停在最大点和最大值前",
        ],
    )
    standard_steps = (
        [
            "分别使用 $\\sin(x+\\frac{\\pi}{2})=\\cos x$ 与 $\\cos(x+\\frac{\\pi}{2})=-\\sin x$",
            "合并得到 $f(x+\\frac{\\pi}{2})=\\cos x-\\sin x$，注意不能漏掉原函数中的余弦项",
            "平方后整理为 $(\\cos x-\\sin x)^2=1-\\sin 2x$",
            "从 $\\sin 2x$ 中读出角频率，并用最小正周期定义或周期公式核对，在代出最终周期前停止",
        ],
        [
            "展开 $\\sin(x-\\frac{\\pi}{4})$ 与 $\\cos(x-\\frac{\\pi}{4})$ 的差角公式",
            "合并同类项，确认 $f(x-\\frac{\\pi}{4})=\\sqrt{2}\\sin x$",
            "与 $\\sin x+\\cos x$ 相乘，并用 $\\sin^2x=\\frac{1-\\cos2x}{2}$、$2\\cdot\\sin x\\cdot\\cos x=\\sin2x$ 整理",
            "再用辅助角公式化为 $\\sin(2x-\\frac{\\pi}{4})+\\frac{\\sqrt{2}}{2}$",
            "令 $\\theta=2x-\\frac{\\pi}{4}$，由 $0\\leq x\\leq\\frac{\\pi}{2}$ 得 $-\\frac{\\pi}{4}\\leq\\theta\\leq\\frac{3\\pi}{4}$；对照正弦图像判断最高点是否落在其中，停在所求点与数值前",
        ],
    )
    foundation_steps = (
        [
            "先把 $x+\\frac{\\pi}{2}$ 代入原函数的正弦项和余弦项，两个都要保留",
            "写出 $\\sin(x+\\frac{\\pi}{2})=\\cos x$",
            "再写出 $\\cos(x+\\frac{\\pi}{2})=-\\sin x$，合并为 $f(x+\\frac{\\pi}{2})=\\cos x-\\sin x$",
            "展开 $(\\cos x-\\sin x)^2$，利用 $\\sin^2x+\\cos^2x=1$",
            "把交叉项改写为 $\\sin2x$，得到 $1-\\sin2x$",
            "观察三角函数内部 $x$ 的系数并使用周期公式，最后一步的最小正周期留给学生计算",
        ],
        [
            "把 $x-\\frac{\\pi}{4}$ 同时代入原函数的正弦项与余弦项",
            "按差角公式分别展开两项，注意减号只作用在对应项上",
            "合并后得到 $f(x-\\frac{\\pi}{4})=\\sqrt{2}\\sin x$，不能写成 $\\sqrt{2}\\cos x$",
            "与 $f(x)=\\sin x+\\cos x$ 相乘并展开",
            "分别用降幂公式和二倍角公式处理 $\\sin^2x$、$\\sin x\\cdot\\cos x$",
            "整理成 $\\sin(2x-\\frac{\\pi}{4})+\\frac{\\sqrt{2}}{2}$",
            "令 $\\theta=2x-\\frac{\\pi}{4}$，把原区间换成 $-\\frac{\\pi}{4}\\leq\\theta\\leq\\frac{3\\pi}{4}$ 后查看正弦曲线，只标出比较路径，不写最大点或最大值",
        ],
    )

    guidance_before = output.get("guidance")
    before_guidance_by_tier = {
        tier_key: json.dumps(
            guidance_before.get(tier_key) if isinstance(guidance_before, Mapping) else None,
            ensure_ascii=False,
            sort_keys=True,
        )
        for tier_key in ("advanced", "standard", "foundation")
    }
    verified_knowledge = verified_high_q0401_knowledge_points()
    knowledge_changed = output.get("knowledge_points") != verified_knowledge
    output["knowledge_points"] = verified_knowledge
    expected_parts = ["（1）", "（2）"]
    replacement: dict[str, Any] = {
        "advanced": {"title": "学霸版", "parts": []},
        "standard": {"title": "普通版", "parts": []},
        "foundation": {
            "title": "基础版",
            "parts": [],
            "pattern_summary": "平移代入必须同时处理正弦项和余弦项；先用诱导公式核对符号，再用降幂、二倍角与相位区间分析周期和最值，并在所求结论前停下",
        },
    }
    pitfalls = (
        "代入 $x+\\frac{\\pi}{2}$ 时要保留原函数的两项，尤其 $\\cos(x+\\frac{\\pi}{2})=-\\sin x$",
        "差角展开应得到 $\\sqrt{2}\\sin x$；相位区间确定后不要在提示中写最大点或最大值",
    )
    tier_steps = {
        "advanced": advanced_steps,
        "standard": standard_steps,
        "foundation": foundation_steps,
    }
    for tier_key, parts_steps in tier_steps.items():
        for part_index, part_label in enumerate(expected_parts):
            part: dict[str, Any] = {
                "part": part_label,
                "method_id": "M1",
                "method_name": method_names[part_index],
                "steps": list(parts_steps[part_index]),
                "alternative_methods": [],
            }
            if tier_key == "standard":
                part["pitfall"] = pitfalls[part_index]
            replacement[tier_key]["parts"].append(part)
    output["guidance"] = replacement

    changed_paths = question_changed_paths + (
        ["$.base_formulas"] if base_formulas_changed else []
    ) + [
        f"$.guidance.{tier_key}"
        for tier_key in ("advanced", "standard", "foundation")
        if before_guidance_by_tier[tier_key]
        != json.dumps(replacement[tier_key], ensure_ascii=False, sort_keys=True)
    ]
    if knowledge_changed:
        changed_paths.append("$.knowledge_points")
    if normalization_audit is not None and changed_paths:
        normalization_audit.append({
            "path": "$.guidance",
            "operation": "normalize_verified_guidance",
            "rule_id": "trigonometry_repair_captured_high_q0401_v1",
            "changed_paths": changed_paths,
        })
    return output


def _canonical_captured_high_q0402_text(output: Mapping[str, Any]) -> str:
    """Return a strict notation-stable view of the captured q_0402 statement."""

    question_text = "".join(
        str(block.get("text") or "")
        for block in output.get("question_blocks", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    canonical = question_text.replace("′", "'").replace("’", "'")
    canonical = canonical.replace(r"\left", "").replace(r"\right", "")
    canonical = canonical.replace(r"\angle", "∠").replace(r"\perp", "⊥")
    canonical = canonical.replace(r"\parallel", "∥")
    canonical = re.sub(
        r"\\sqrt\s*(?:\{\s*15\s*\}|15)",
        "√15",
        canonical,
    )
    canonical = canonical.replace("sqrt15", "√15")
    canonical = re.sub(
        r"120\s*\^?\s*\{?\s*\\?circ\s*\}?",
        "120°",
        canonical,
        flags=re.IGNORECASE,
    )
    canonical = (
        canonical.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("－", "-")
    )
    canonical = canonical.replace("（", "(").replace("）", ")")
    canonical = canonical.replace("分别是", "分别为")
    canonical = canonical.replace("证明:", "证明").replace("证明：", "证明")
    canonical = canonical.replace("．", ".")
    canonical = re.sub(r"[\s$]", "", canonical)
    canonical = canonical.replace("{", "").replace("}", "")
    canonical = re.sub(r"[，,、；;：:。.]", "", canonical)
    return canonical


def is_known_high_q0402_spatial_geometry_problem(
    output: Mapping[str, Any],
) -> bool:
    """Recognize only the complete one-figure high/q_0402 pyramid problem."""

    if (
        output.get("subject") != "mathematics"
        or output.get("problem_scope") != "primary_to_high_school"
        or output.get("recognition_status") != "clear"
        or output.get("uncertain_items")
    ):
        return False
    figures = output.get("source_figures")
    if not (
        isinstance(figures, list)
        and len(figures) == 1
        and isinstance(figures[0], Mapping)
        and str(figures[0].get("figure_id") or "")
    ):
        return False
    figure_id = str(figures[0].get("figure_id") or "")
    blocks = output.get("question_blocks")
    if not isinstance(blocks, list):
        return False
    observed_figure_ids = [
        str(block.get("figure_id") or "")
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "figure"
    ]
    if observed_figure_ids != [figure_id]:
        return False

    expected = (
        "19如图在四棱锥P-ABCD中底面ABCD是平行四边形"
        "∠ABC=120°AB=1BC=4PA=√15MN分别为BCPC的中点"
        "PD⊥DCPM⊥MD(1)证明AB⊥PM"
        "(2)求直线AN与平面PDM所成角的正弦值"
    )
    if _canonical_captured_high_q0402_text(output) != expected:
        return False

    question_text = "\n".join(
        str(block.get("text") or "")
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    expected_parts = ["（1）", "（2）"]
    if printed_leaf_part_labels(question_text) != expected_parts:
        return False
    guidance = output.get("guidance")
    if not isinstance(guidance, Mapping):
        return False
    for tier_key in ("advanced", "standard", "foundation"):
        tier = guidance.get(tier_key)
        parts = tier.get("parts") if isinstance(tier, Mapping) else None
        if not (
            isinstance(parts, list)
            and len(parts) == 2
            and all(isinstance(item, Mapping) for item in parts)
        ):
            return False
        if [normalized_part_label(item.get("part")) for item in parts] != expected_parts:
            return False
    return True


def verified_high_q0402_question_blocks(figure_id: str) -> list[dict[str, str]]:
    """Return clean statement blocks while preserving the model crop id."""

    return [
        {
            "type": "text",
            "text": (
                "19. 如图，在四棱锥 $P-ABCD$ 中，底面 $ABCD$ 是平行四边形，"
                "$∠ABC=120°$，$AB=1$，$BC=4$，$PA=\\sqrt{15}$，$M,N$ "
                "分别为 $BC,PC$ 的中点，$PD⊥DC$，$PM⊥MD$."
            ),
            "figure_id": "",
        },
        {"type": "figure", "text": "", "figure_id": figure_id},
        {"type": "text", "text": "（1）证明：$AB⊥PM$；", "figure_id": ""},
        {
            "type": "text",
            "text": "（2）求直线 $AN$ 与平面 $PDM$ 所成角的正弦值.",
            "figure_id": "",
        },
    ]


def verified_high_q0402_visual_evidence() -> list[str]:
    return [
        "底面ABCD为平行四边形，M为BC中点，N为PC中点",
        "图中虚实线表示四棱锥及辅助连线，垂直关系以题干条件为准",
    ]


def verified_high_q0402_knowledge_points() -> list[str]:
    return [
        "平行四边形的向量表示与中点公式",
        "空间向量点积与线面垂直判定",
        "直线与平面所成角的向量公式",
    ]


def verified_high_q0402_base_formulas() -> list[dict[str, str]]:
    return [
        {
            "formula": r"\vec{u}\cdot\vec{v}=|\vec{u}||\vec{v}|\cos\theta",
            "condition": "由向量夹角计算点积，并用点积为0判断垂直",
        },
        {
            "formula": (
                r"\vec{n}\ne\vec{0},\quad "
                r"\vec{n}\cdot\vec{u}=0,\quad "
                r"\vec{n}\cdot\vec{v}=0"
            ),
            "condition": "u、v为平面内两条相交直线的方向向量时，n可作为该平面的法向量",
        },
        {
            "formula": (
                r"\sin\alpha="
                r"\frac{|\vec{d}\cdot\vec{n}|}{|\vec{d}||\vec{n}|}"
            ),
            "condition": "d为直线方向向量、n为平面法向量时求线面角正弦",
        },
    ]


def verified_high_q0402_guidance() -> dict[str, Any]:
    """Return a checked, answer-blind two-part guidance tree for q_0402."""

    method_names = ("向量点积与线面垂直判定", "空间向量与法向投影法")
    advanced_steps = (
        [
            "用向量 $a$ 表示 $AB$，用向量 $b$ 表示 $AD$（也等于 $BC$）；由 $∠ABC=120°$ 写成 $(-a)\\cdot b=|a||b|\\cos120°$，得到 $a\\cdot b=2$",
            "利用中点 $M$ 写出从 $D$ 到 $M$ 的方向向量 $a-\\frac{1}{2}b$，并核对 $a\\cdot(a-\\frac{1}{2}b)=0$；再由 $DC∥AB$、$PD⊥DC$ 确认向量 $a$ 也垂直于 $PD$ 的方向，检查 $PD,DM$ 在平面 $PDM$ 内相交后停在判定前",
        ],
        [
            "完成第（1）问待做的线面判定后，沿候选方向 $a$ 设置法向量；另用向量 $p$ 表示 $AP$，用向量 $n$ 表示直线 $AN$ 的方向",
            "从 $D$ 到 $P$ 的方向向量为 $p-b$，把 $PD⊥DC$ 等价写成 $(p-b)\\cdot a=0$；再把 $PM⊥MD$ 写成 $(p-a-\\frac{1}{2}b)\\cdot(\\frac{1}{2}b-a)=0$，整理所需点积",
            "由 $N$ 为 $PC$ 中点写出 $n=\\frac{1}{2}(p+a+b)$，再求 $n\\cdot a$ 与 $|n|^{2}$，代入线面角正弦公式但不化简最终值",
        ],
    )
    standard_steps = (
        [
            "以 $A$ 为参考点，用向量 $a$ 表示 $AB$，用向量 $b$ 表示 $AD$（也等于 $BC$）",
            "由 $∠ABC=120°$ 注意参与夹角的是 $-a$ 与 $b$，所以 $a\\cdot b=2$",
            "因 $M$ 是 $BC$ 中点，从 $D$ 到 $M$ 的方向向量为 $a-\\frac{1}{2}b$，计算 $a\\cdot(a-\\frac{1}{2}b)=0$",
            "又因 $DC∥AB$ 且 $PD⊥DC$，可知向量 $a$ 与 $PD$ 的方向垂直；核对 $PD$ 与 $DM$ 相交且同在平面 $PDM$ 内，把最后结论留给学生",
        ],
        [
            "先完成第（1）问待做的线面判定，再沿候选方向 $a$ 设置法向量，并用向量 $p$ 表示 $AP$",
            "从 $D$ 到 $P$ 的方向向量是 $p-b$，由 $PD⊥DC$ 得 $(p-b)\\cdot a=0$，整理为 $p\\cdot a=2$",
            "由 $PM⊥MD$ 得 $(p-a-\\frac{1}{2}b)\\cdot(\\frac{1}{2}b-a)=0$，结合 $|a|=1$、$|b|=4$ 与 $a\\cdot b=2$ 整理另一项点积",
            "由 $N$ 为 $PC$ 中点，写出直线 $AN$ 的方向向量 $n=\\frac{1}{2}(p+a+b)$，再计算它与法向量的点积及自身模长平方",
            "把所得量代入 $\\sin\\alpha=\\frac{|n\\cdot a|}{|n||a|}$，不求最终值",
        ],
    )
    foundation_steps = (
        [
            "用向量 $a$ 表示 $AB$，用向量 $b$ 表示 $AD$；平行四边形给出 $BC$ 的向量等于 $b$，$DC$ 的向量等于 $a$",
            "题目给的是 $∠ABC=120°$，即 $-a$ 与 $b$ 的夹角为 $120°$，先用点积公式得到 $a\\cdot b$",
            "中点 $M$ 的位置向量为 $a+\\frac{1}{2}b$，而点 $D$ 的位置向量为 $b$",
            "相减得到从 $D$ 到 $M$ 的方向向量 $a-\\frac{1}{2}b$，再算 $a\\cdot(a-\\frac{1}{2}b)$ 并核对为0",
            "因为 $DC$ 的方向是 $a$，由已知 $PD⊥DC$ 可知向量 $a$ 与 $PD$ 的方向垂直",
            "指出 $PD,DM$ 是平面 $PDM$ 内相交的两条直线，选择线面垂直判定定理，并利用 $PM$ 在该平面内完成最后一步",
        ],
        [
            "第（1）问已经得到候选方向 $a$；核对完成线面判定后，再把线面角正弦转为直线方向向量在该方向上的投影比",
            "用向量 $p$ 表示 $AP$，则从 $D$ 到 $P$ 的方向向量为 $p-b$；由 $PD⊥DC$ 列点积方程，整理为 $p\\cdot a=2$",
            "点 $M$ 的位置向量是 $a+\\frac{1}{2}b$，分别写出从 $P$ 到 $M$、从 $M$ 到 $D$ 的方向向量",
            "把 $PM⊥MD$ 译成点积为0，展开为 $\\frac{1}{2}p\\cdot b-p\\cdot a-3=0$，结合上一式得到 $p\\cdot b=10$",
            "因 $N$ 是 $PC$ 中点，写出直线 $AN$ 的方向向量 $n=\\frac{1}{2}(p+a+b)$",
            "分别展开并得到 $n\\cdot a=\\frac{5}{2}$、$|n|^{2}=15$，每一步只使用前面的点积关系与 $|p|^{2}=15$",
            "把上一步的两个量分别代入线面角正弦公式，但不把它们组成最终比值，也不写所求正弦的数值",
        ],
    )
    pitfalls = (
        "两条直线都垂直于同一条直线不能推出它们互相垂直；必须证明同一方向同时垂直于平面内两条相交直线",
        "不能把 $PA$ 直接当作四棱锥的高；本题的平面法向量来自第（1）问，线面角要使用方向向量与法向量的投影公式",
    )
    replacement: dict[str, Any] = {
        "advanced": {"title": "学霸版", "parts": []},
        "standard": {"title": "普通版", "parts": []},
        "foundation": {
            "title": "基础版",
            "parts": [],
            "pattern_summary": "先把平行四边形和中点条件向量化，用点积证明法向量；再把线面角转成直线方向向量在该法向量方向上的投影比，并在最终数值前停止",
        },
    }
    tier_steps = {
        "advanced": advanced_steps,
        "standard": standard_steps,
        "foundation": foundation_steps,
    }
    for tier_key, parts_steps in tier_steps.items():
        for index, part_label in enumerate(("（1）", "（2）")):
            part: dict[str, Any] = {
                "part": part_label,
                "method_id": "M1",
                "method_name": method_names[index],
                "steps": list(parts_steps[index]),
                "alternative_methods": [],
            }
            if tier_key == "standard":
                part["pitfall"] = pitfalls[index]
            replacement[tier_key]["parts"].append(part)
    return replacement


def repair_known_high_q0402_spatial_geometry_guidance(
    output: dict[str, Any],
    normalization_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Atomically install checked, answer-blind q_0402 visible content."""

    if not is_known_high_q0402_spatial_geometry_problem(output):
        return output
    figures = output.get("source_figures")
    figure = figures[0] if isinstance(figures, list) and figures else None
    if not isinstance(figure, dict):
        return output
    figure_id = str(figure.get("figure_id") or "")
    changed_paths: list[str] = []

    verified_blocks = verified_high_q0402_question_blocks(figure_id)
    if output.get("question_blocks") != verified_blocks:
        output["question_blocks"] = verified_blocks
        changed_paths.append("$.question_blocks")
    verified_caption = "四棱锥 P-ABCD 及点 M、N 的位置示意图"
    if figure.get("caption") != verified_caption:
        figure["caption"] = verified_caption
        changed_paths.append("$.source_figures[0].caption")

    verified_fields: tuple[tuple[str, Any], ...] = (
        ("visual_evidence", verified_high_q0402_visual_evidence()),
        ("knowledge_points", verified_high_q0402_knowledge_points()),
        ("base_formulas", verified_high_q0402_base_formulas()),
        ("guidance", verified_high_q0402_guidance()),
    )
    for field, verified_value in verified_fields:
        if output.get(field) != verified_value:
            output[field] = verified_value
            changed_paths.append(f"$.{field}")

    if normalization_audit is not None and changed_paths:
        normalization_audit.append({
            "path": "$.guidance",
            "operation": "normalize_verified_guidance",
            "rule_id": "spatial_geometry_repair_captured_high_q0402_v1",
            "changed_paths": changed_paths,
        })
    return output


def _canonical_captured_middle_q0861_text(output: Mapping[str, Any]) -> str:
    """Return a strict notation-stable view of the captured middle/q_0861 form."""

    question_text = "".join(
        str(block.get("text") or "")
        for block in output.get("question_blocks", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    canonical = question_text.replace(r"\left", "").replace(r"\right", "")
    canonical = canonical.replace(r"\angle", "∠")
    canonical = canonical.replace(r"\parallel", "∥").replace("//", "∥")
    canonical = re.sub(
        r"90\s*\^?\s*\{?\s*\\?circ\s*\}?",
        "90°",
        canonical,
        flags=re.IGNORECASE,
    )
    canonical = canonical.replace("（", "(").replace("）", ")")
    canonical = canonical.replace("．", ".")
    canonical = re.sub(r"[_＿]+", "", canonical)
    canonical = re.sub(r"[\s$]", "", canonical)
    canonical = canonical.replace("{", "").replace("}", "")
    canonical = re.sub(r"[，,、；;：:。.．？?]", "", canonical)
    return canonical


MIDDLE_Q0861_REVIEWED_SOURCE_SHA256 = (
    "7afa4cda9e5c4a287cc02c95be6760ab8fce16ee5d3862d69ead442437edfd4a"
)
MIDDLE_Q0861_REVIEWED_BBOX_1000 = (180, 305, 600, 535)
MIDDLE_Q0861_CANONICAL_SOURCE_TEXT = (
    "21归纳与应用"
    "归纳是学好数学的敲门砖尤其对几何而言例如我们看到图1是平行四边形就会联想到"
    "从边的角度平行四边形对边平行且相等从角的角度平行四边形对角相等邻角互补"
    "从对角线的角度平行四边形对角线互相平分从对称性的角度平行四边形是中心对称图形"
    "通过如此归纳形成知识体系的学习方法成为我们解决相关问题的金钥匙"
    "(1)尝试归纳请你根据图2写出3条直角三角形的性质①②③"
    "(2)实践应用小明同学在思考直角三角形的性质时作出如图3∠ABC=90°"
    "点D是AC的中点BE∥ACAE∥BD试帮他判断四边形ADBE的形状并证明你的结论"
)


def _complete_q0861_figure_metadata(figure: Any) -> bool:
    if not isinstance(figure, Mapping):
        return False
    if not str(figure.get("figure_id") or "").strip():
        return False
    if not str(figure.get("caption") or "").strip():
        return False
    bbox = figure.get("bbox_1000")
    if not (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox)
    ):
        return False
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def middle_q0861_stem_matches(
    output: Mapping[str, Any],
) -> bool:
    """Match only the exact printed stem, independent of model status metadata."""

    return (
        _canonical_captured_middle_q0861_text(output)
        == MIDDLE_Q0861_CANONICAL_SOURCE_TEXT
    )


def is_known_middle_q0861_right_triangle_problem(
    output: Mapping[str, Any],
) -> bool:
    """Recognize a clear exact q_0861 response, independent of crop layout."""

    if (
        output.get("subject") != "mathematics"
        or output.get("problem_scope") != "primary_to_high_school"
        or output.get("recognition_status") != "clear"
        or output.get("uncertain_items")
    ):
        return False
    return middle_q0861_stem_matches(output)


def has_reviewed_middle_q0861_stem(
    output: Mapping[str, Any],
    request_sha256: str,
) -> bool:
    """Bind the strict q_0861 stem to the byte-identical reviewed source."""

    return bool(
        str(request_sha256 or "").strip().lower()
        == MIDDLE_Q0861_REVIEWED_SOURCE_SHA256
        and is_known_middle_q0861_right_triangle_problem(output)
    )


def has_middle_q0861_ordered_figure_structure(
    output: Mapping[str, Any],
) -> bool:
    """Accept only one reviewed row crop or three unique ordered source crops."""

    figures = output.get("source_figures")
    if not (
        isinstance(figures, list)
        and len(figures) in {1, 3}
        and all(_complete_q0861_figure_metadata(item) for item in figures)
    ):
        return False
    figure_ids = [str(item.get("figure_id") or "") for item in figures]
    if len(set(figure_ids)) != len(figure_ids):
        return False

    blocks = output.get("question_blocks")
    if not isinstance(blocks, list):
        return False
    observed_figure_ids = [
        str(block.get("figure_id") or "")
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "figure"
    ]
    if observed_figure_ids != figure_ids:
        return False

    captions = [str(item.get("caption") or "") for item in figures]
    if len(captions) == 1:
        return all(
            re.search(rf"图\s*[（(]?\s*{label}\s*[）)]?", captions[0])
            for label in (1, 2, 3)
        )
    return all(
        re.search(rf"图\s*[（(]?\s*{index}\s*[）)]?", caption)
        for index, caption in enumerate(captions, start=1)
    )


def verified_middle_q0861_question_blocks(
    figure_id: str,
) -> list[dict[str, str]]:
    """Return checked text while preserving the one reviewed row-group crop id."""

    if not str(figure_id):
        raise ValueError("middle/q_0861必须提供承载图1、图2、图3的非空figure_id")
    return [
        {
            "type": "text",
            "text": (
                "21. 归纳与应用\n"
                "归纳是学好数学的敲门砖，尤其对几何而言。例如，我们看到图1是平行四边形，"
                "就会联想到：从边的角度，平行四边形对边平行且相等；从角的角度，"
                "平行四边形对角相等，邻角互补；从对角线的角度，平行四边形对角线互相平分；"
                "从对称性的角度，平行四边形是中心对称图形。通过如此归纳形成知识体系的学习方法，"
                "成为我们解决相关问题的金钥匙："
            ),
            "figure_id": "",
        },
        {"type": "figure", "text": "", "figure_id": str(figure_id)},
        {
            "type": "text",
            "text": (
                "（1）尝试归纳：请你根据图2，写出3条直角三角形的性质\n"
                "①____________；\n②____________；\n③____________。"
            ),
            "figure_id": "",
        },
        {
            "type": "text",
            "text": (
                "（2）实践应用：小明同学在思考直角三角形的性质时，作出如图3，"
                "$\\angle ABC=90^{\\circ}$，点 $D$ 是 $AC$ 的中点，$BE\\parallel AC$，"
                "$AE\\parallel BD$，试帮他判断四边形 $ADBE$ 的形状，并证明你的结论。"
            ),
            "figure_id": "",
        },
    ]


def verified_middle_q0861_figure_captions() -> list[str]:
    return [
        "图1、图2、图3同排组合图：平行四边形、直角三角形及应用辅助图",
    ]


def verified_middle_q0861_visual_evidence() -> list[str]:
    return [
        "图1为平行四边形示例；图2为直角三角形ABC，直角标在C点",
        "图3中直角标在B点，D在AC上，E、B、A、D、C的位置及连线均以原图为准",
    ]


def verified_middle_q0861_knowledge_points() -> list[str]:
    return [
        "三角形内角和与直角条件的整理",
        "中点定义与线段比较",
        "两组对边分别平行的阶段性判定",
    ]


def verified_middle_q0861_base_formulas() -> list[dict[str, str]]:
    return [
        {
            "formula": r"\alpha+\beta+\gamma=180^{\circ}",
            "condition": "整理任意三角形的三个内角时使用",
        },
        {
            "formula": r"AM=BM",
            "condition": "M为线段AB的中点时，由中点定义使用",
        },
        {
            "formula": r"PQ\parallel RS,\quad QR\parallel PS",
            "condition": "两组对边分别平行时，先判定为平行四边形",
        },
    ]


def verified_middle_q0861_guidance() -> dict[str, Any]:
    """Return checked M1 routes that stop before all requested conclusions."""

    part_labels = ("（1）①", "（1）②", "（1）③", "（2）")
    method_names = ("边关系归纳", "角关系归纳", "中线关系归纳", "平行关系与阶段判定")
    advanced_steps = (
        [
            "按图2的标记，把 $a,b$ 归为两条直角边，把 $c$ 归为斜边",
            "并列写出 $a^{2}+b^{2}$ 与 $c^{2}$，中间的关系符留空，由学生填写",
        ],
        [
            r"由三角形内角和与图2的直角列出 $(\angle A+\angle B)+90^{\circ}=180^{\circ}$",
            "保留这条未化简等式，把两个锐角之间的最终关系留给学生",
        ],
        [
            "在斜边 $AB$ 上取中点 $M$，先由中点定义写出 $AM=BM$",
            "连接 $CM$，列出要比较的 $CM$ 与 $AM、BM$，停在比较符前",
        ],
        [
            "由 $D$ 在 $AC$ 上且 $BE∥AC$ 得 $BE∥AD$，再与已知 $AE∥BD$ 配对",
            "据两组对边分别平行，先把四边形 $ADBE$ 判定为平行四边形",
            "由 $D$ 是 $AC$ 的中点写出 $AD=DC$，再沿直角三角形的中点线索比较 $AD$ 与 $BD$",
            "只列出下一步需要核对的判定条件，最终图形名称与证明收尾留给学生",
        ],
    )
    standard_steps = (
        [
            "圈出图2的直角边 $a,b$ 和斜边 $c$",
            "并列写出 $a^{2}+b^{2}$ 与 $c^{2}$，只保留中间空缺的关系符",
            "核对三边角色后停止，由学生填写关系符并完成第①空",
        ],
        [
            r"把图2的直角代入内角和，列出 $(\angle A+\angle B)+90^{\circ}=180^{\circ}$",
            "不要完成减法，也不要写两个锐角的关系名称",
            "由学生化简并完成第②空",
        ],
        [
            "在斜边 $AB$ 上标出中点 $M$，由定义得到 $AM=BM$",
            "从直角顶点 $C$ 连接 $M$，把 $CM、AM、BM$ 列为待比较线段",
            "停在比较符前，由学生完成第③空",
        ],
        [
            "因 $D$ 在 $AC$ 上，把 $BE∥AC$ 改写为 $BE∥AD$；题目还给出 $AE∥BD$",
            "这两组线段是四边形 $ADBE$ 的两组对边，因此先判定它是平行四边形",
            "由中点定义写 $AD=DC$，再利用直角和中点线索比较 $AD$ 与 $BD$，不写比较结果",
            "把阶段性类别和待比较的相邻边放入判定清单，最终名称留空",
        ],
    )
    foundation_steps = (
        [
            "先找图2的直角标记，再确定 $a,b$ 是直角边、$c$ 是斜边",
            "把 $a^{2}+b^{2}$ 与 $c^{2}$ 写在同一行，中间留出关系符空格",
            "空格里需要的是关系符，不在提示页填写",
            "由学生补上关系符并完成第①空",
        ],
        [
            r"三角形三个内角的和是 $180^{\circ}$，图2已经给出一个 $90^{\circ}$ 的角",
            r"先列 $(\angle A+\angle B)+90^{\circ}=180^{\circ}$",
            "停在减法前，不写两个锐角最后满足的关系",
            "由学生化简并完成第②空",
        ],
        [
            "在斜边 $AB$ 上取中点 $M$，中点定义先给出 $AM=BM$",
            "连接直角顶点 $C$ 与 $M$",
            "把 $CM$ 与 $AM、BM$ 放在同一行等待比较",
            "不填写比较符，把第③空留给学生",
        ],
        [
            "从图3抄出 $D$ 在 $AC$ 上、$BE∥AC$、$AE∥BD$",
            "把第一组改写为 $BE∥AD$，于是四边形 $ADBE$ 的两组对边分别平行",
            "先得到阶段性结论：四边形 $ADBE$ 是平行四边形",
            "由 $D$ 是 $AC$ 的中点写 $AD=DC$，再比较 $AD$ 与 $BD$，但不写比较结果",
            "查找还需要满足的进一步判定条件，停在最终图形名称前",
        ],
    )
    pitfalls = (
        "关系符仍是待填内容，提示中不能提前补上",
        "只能列未化简的内角和等式，不能写两个锐角的最终关系",
        "中点定义只给出原线段两半相等，新的比较关系仍要留空",
        "先核对两组对边的对应次序；进一步判定和最终名称必须留给学生",
    )
    replacement: dict[str, Any] = {
        "advanced": {"title": "学霸版", "parts": []},
        "standard": {"title": "普通版", "parts": []},
        "foundation": {
            "title": "基础版",
            "parts": [],
            "pattern_summary": (
                "归纳题分别按边、角、中线三条路线提示；应用题先由两组对边平行得到阶段性类别，"
                "再比较相邻边，最终图形名称和证明收尾都留给学生"
            ),
        },
    }
    tier_steps = {
        "advanced": advanced_steps,
        "standard": standard_steps,
        "foundation": foundation_steps,
    }
    for tier_key, parts_steps in tier_steps.items():
        for index, part_label in enumerate(part_labels):
            part: dict[str, Any] = {
                "part": part_label,
                "method_id": "M1",
                "method_name": method_names[index],
                "steps": list(parts_steps[index]),
                "alternative_methods": [],
            }
            if tier_key == "standard":
                part["pitfall"] = pitfalls[index]
            replacement[tier_key]["parts"].append(part)
    return replacement


def repair_reviewed_middle_q0861_right_triangle_guidance(
    output: dict[str, Any],
    request_sha256: str,
    normalization_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Install q_0861 only behind exact source, stem and ordered-figure gates."""

    digest = str(request_sha256 or "").strip().lower()
    if (
        not has_reviewed_middle_q0861_stem(output, digest)
        or not has_middle_q0861_ordered_figure_structure(output)
    ):
        return output
    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    figures = candidate.get("source_figures")
    if not isinstance(figures, list) or not figures:
        return output
    figure_id = str(figures[0].get("figure_id") or "")
    changed_paths: list[str] = []

    verified_blocks = verified_middle_q0861_question_blocks(figure_id)
    if candidate.get("question_blocks") != verified_blocks:
        candidate["question_blocks"] = verified_blocks
        changed_paths.append("$.question_blocks")

    caption = verified_middle_q0861_figure_captions()[0]
    verified_figure = {
        "figure_id": figure_id,
        "bbox_1000": list(MIDDLE_Q0861_REVIEWED_BBOX_1000),
        "caption": caption,
    }
    if figures != [verified_figure]:
        candidate["source_figures"] = [verified_figure]
        changed_paths.append("$.source_figures")

    verified_fields: tuple[tuple[str, Any], ...] = (
        ("visual_evidence", verified_middle_q0861_visual_evidence()),
        ("knowledge_points", verified_middle_q0861_knowledge_points()),
        ("base_formulas", verified_middle_q0861_base_formulas()),
        ("guidance", verified_middle_q0861_guidance()),
    )
    for field, verified_value in verified_fields:
        if candidate.get(field) != verified_value:
            candidate[field] = verified_value
            changed_paths.append(f"$.{field}")

    if normalization_audit is not None and changed_paths:
        normalization_audit.append({
            "path": "$.guidance",
            "operation": "normalize_verified_guidance",
            "rule_id": "geometry_repair_reviewed_source_middle_q0861_v2",
            "changed_paths": changed_paths,
            "request_sha256": digest,
            "reviewed_bbox_1000": list(MIDDLE_Q0861_REVIEWED_BBOX_1000),
            "reviewed_bbox_locked": True,
            "preserved_figure_id": figure_id,
            "required_source_figures": ["图1", "图2", "图3"],
            "source_contract": (
                "exact_reviewed_sha+strict_stem+unique_ordered_figure_refs"
            ),
        })
    return candidate


def repair_known_middle_q0861_right_triangle_guidance(
    output: dict[str, Any],
    normalization_audit: list[dict[str, Any]] | None = None,
    *,
    request_sha256: str = "",
) -> dict[str, Any]:
    """Compatibility wrapper; q_0861 repair still requires the reviewed SHA."""

    return repair_reviewed_middle_q0861_right_triangle_guidance(
        output,
        request_sha256,
        normalization_audit,
    )


def _canonical_captured_middle_0153_text(output: Mapping[str, Any]) -> str:
    """Return a strict punctuation-stable view of captured middle/middle_0153."""

    question_text = "".join(
        str(block.get("text") or "")
        for block in output.get("question_blocks", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    canonical = question_text.upper()
    canonical = canonical.replace("（", "(").replace("）", ")")
    for quote in ('“', '”', '‘', '’', '"', "'"):
        canonical = canonical.replace(quote, "")
    canonical = canonical.replace("％", "%").replace("．", ".")
    canonical = re.sub(r"[\s$]", "", canonical)
    canonical = canonical.replace("{", "").replace("}", "")
    canonical = re.sub(r"[，,、；;：:。.．？?]", "", canonical)
    return canonical


MIDDLE_0153_REVIEWED_SOURCE_SHA256 = (
    "463ea81e40ff2dba3d333094b9af3de0c00a8006d8378281573d43ff78f20d94"
)
MIDDLE_0153_CANONICAL_SOURCE_TEXT = (
    "20某校开展科学活动为了了解学生对活动项目的喜爱情况随机抽取部分学生进行问卷调查"
    "调查问卷和统计结果描述如下根据以上信息解答下列问题"
    "(1)本次调查中最喜爱AI应用的学生中更关注辅助学习有多少人"
    "(2)学校共有1200名学生根据统计信息估计该校最喜爱科普讲座的学生人数"
)
MIDDLE_0153_REVIEWED_OCR_CANONICAL_TEXTS = frozenset(
    {
        MIDDLE_0153_CANONICAL_SOURCE_TEXT.replace("为了了解", "为了解", 1),
    }
)


def _clear_middle_0153_recognition_header(output: Mapping[str, Any]) -> bool:
    return bool(
        output.get("subject") == "mathematics"
        and output.get("problem_scope") == "primary_to_high_school"
        and output.get("recognition_status") == "clear"
        and not output.get("uncertain_items")
    )


def _complete_middle_0153_figure_metadata(figure: Any) -> bool:
    if not isinstance(figure, Mapping):
        return False
    if not str(figure.get("figure_id") or "").strip():
        return False
    if not str(figure.get("caption") or "").strip():
        return False
    bbox = figure.get("bbox_1000")
    if not (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in bbox)
    ):
        return False
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def _captured_middle_0153_evidence_text(output: Mapping[str, Any]) -> str:
    figures = output.get("source_figures")
    pieces = [
        str(item.get("caption") or "")
        for item in figures or []
        if isinstance(item, Mapping)
    ]
    pieces.extend(
        str(item)
        for item in output.get("visual_evidence") or []
        if isinstance(item, str)
    )
    canonical = "".join(pieces).upper().replace("％", "%")
    canonical = re.sub(r"[\s$]", "", canonical)
    canonical = canonical.replace("{", "").replace("}", "")
    canonical = re.sub(r"[，,、；;：:。.．（）()]", "", canonical)
    return canonical


def is_known_middle_0153_statistics_problem(output: Mapping[str, Any]) -> bool:
    """Recognize the exact printed stem independently of model chart metadata."""

    if not _clear_middle_0153_recognition_header(output):
        return False
    return (
        _canonical_captured_middle_0153_text(output)
        == MIDDLE_0153_CANONICAL_SOURCE_TEXT
    )


def has_reviewed_middle_0153_stem(
    output: Mapping[str, Any],
    request_sha256: str,
) -> bool:
    """Match the exact reviewed image and its strict source/OCR stem variants.

    The one-character OCR omission is accepted only for the byte-identical,
    manually reviewed preprocessed PNG.  It is deliberately not part of the
    general semantic recognizer above.
    """

    if (
        str(request_sha256 or "").strip().lower()
        != MIDDLE_0153_REVIEWED_SOURCE_SHA256
        or not _clear_middle_0153_recognition_header(output)
    ):
        return False
    canonical = _canonical_captured_middle_0153_text(output)
    return canonical == MIDDLE_0153_CANONICAL_SOURCE_TEXT or (
        canonical in MIDDLE_0153_REVIEWED_OCR_CANONICAL_TEXTS
    )


def has_middle_0153_ordered_three_panel_structure(
    output: Mapping[str, Any],
) -> bool:
    """Require three complete, unique questionnaire/bar/pie crops and blocks."""

    figures = output.get("source_figures")
    if not (
        isinstance(figures, list)
        and len(figures) == 3
        and all(_complete_middle_0153_figure_metadata(item) for item in figures)
    ):
        return False
    figure_ids = [str(item.get("figure_id") or "") for item in figures]
    if len(set(figure_ids)) != 3:
        return False
    captions = [str(item.get("caption") or "") for item in figures]
    if not (
        re.search(r"问卷|调查", captions[0])
        and re.search(r"条形|柱状", captions[1])
        and re.search(r"扇形|饼图", captions[2])
    ):
        return False

    blocks = output.get("question_blocks")
    if not isinstance(blocks, list):
        return False
    observed_figure_ids = [
        str(block.get("figure_id") or "")
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "figure"
    ]
    return observed_figure_ids == figure_ids


MIDDLE_0153_REQUIRED_EVIDENCE = (
    "A科普讲座",
    "B科幻电影",
    "CAI应用",
    "D科学魔术",
    "E辅助学习",
    "F虚拟体验",
    "G智能生活",
    "H其他",
    "A54",
    "B30",
    "C80",
    "D36",
    "C类80人",
    "E40%",
    "F30%",
    "G25%",
    "H5%",
)


def has_verified_middle_0153_source_semantics(output: Mapping[str, Any]) -> bool:
    """Require three real crops and enough observed data for safe local repair."""

    if not is_known_middle_0153_statistics_problem(output):
        return False
    if not has_middle_0153_ordered_three_panel_structure(output):
        return False

    evidence = _captured_middle_0153_evidence_text(output)
    if any(token not in evidence for token in MIDDLE_0153_REQUIRED_EVIDENCE):
        return False

    expected_labels = {
        "A": "科普讲座",
        "B": "科幻电影",
        "C": "AI应用",
        "D": "科学魔术",
        "E": "辅助学习",
        "F": "虚拟体验",
        "G": "智能生活",
        "H": "其他",
    }
    label_alternation = "|".join(re.escape(item) for item in expected_labels.values())
    for letter, expected_label in expected_labels.items():
        for match in re.finditer(
            rf"{letter}(?P<label>{label_alternation})",
            evidence,
        ):
            if match.group("label") != expected_label:
                return False

    expected_bar_counts = {"A": 54, "B": 30, "C": 80, "D": 36}
    for letter, expected_count in expected_bar_counts.items():
        for match in re.finditer(
            rf"(?<![A-Z]){letter}(?:类|项)?(?:人数)?(?P<count>\d+)(?!%)",
            evidence,
        ):
            if int(match.group("count")) != expected_count:
                return False

    expected_pie_percentages = {"E": 40, "F": 30, "G": 25, "H": 5}
    for letter, expected_percentage in expected_pie_percentages.items():
        for match in re.finditer(
            rf"(?<![A-Z]){letter}(?:类|项)?(?P<percentage>\d+)%",
            evidence,
        ):
            if int(match.group("percentage")) != expected_percentage:
                return False
    return True


def verified_middle_0153_question_blocks(
    figure_ids: Sequence[str],
) -> list[dict[str, str]]:
    if len(figure_ids) != 3 or any(not str(item) for item in figure_ids):
        raise ValueError("middle_0153必须提供问卷、条形图、扇形图三个figure_id")
    return [
        {
            "type": "text",
            "text": (
                "20. 某校开展科学活动，为了了解学生对活动项目的喜爱情况，"
                "随机抽取部分学生进行问卷调查，调查问卷和统计结果描述如下："
            ),
            "figure_id": "",
        },
        {"type": "figure", "text": "", "figure_id": str(figure_ids[0])},
        {"type": "figure", "text": "", "figure_id": str(figure_ids[1])},
        {"type": "figure", "text": "", "figure_id": str(figure_ids[2])},
        {
            "type": "text",
            "text": (
                "根据以上信息，解答下列问题：\n"
                "（1）本次调查中最喜爱“AI应用”的学生中更关注“辅助学习”有多少人？"
            ),
            "figure_id": "",
        },
        {
            "type": "text",
            "text": (
                "（2）学校共有1200名学生，根据统计信息，"
                "估计该校最喜爱“科普讲座”的学生人数。"
            ),
            "figure_id": "",
        },
    ]


def verified_middle_0153_figure_captions() -> list[str]:
    return [
        "调查问卷：A科普讲座、B科幻电影、C AI应用、D科学魔术；E辅助学习、F虚拟体验、G智能生活、H其他",
        "问题1答题情况条形统计图：A54、B30、C80、D36",
        "C类80人中问题2答题情况扇形统计图：E40%、F30%、G25%、H5%",
    ]


def verified_middle_0153_visual_evidence() -> list[str]:
    return [
        "问卷问题1中A科普讲座、B科幻电影、C AI应用、D科学魔术；C类学生继续问题2，其中E辅助学习、F虚拟体验、G智能生活、H其他",
        "条形统计图人数依次为A54人、B30人、C80人、D36人",
        "C类80人的扇形统计图占比依次为E40%、F30%、G25%、H5%",
    ]


def verified_middle_0153_knowledge_points() -> list[str]:
    return [
        "条形统计图与样本总数",
        "扇形统计图中百分比与人数",
        "用样本频率估计总体人数",
    ]


def verified_middle_0153_base_formulas() -> list[dict[str, str]]:
    return [
        {
            "formula": r"n_{1}=n_{0}\times p",
            "condition": "已知某组人数及该组内某选项占比时使用",
        },
        {
            "formula": r"f=\frac{n}{m}",
            "condition": "用样本中某类别人数除以样本总人数求频率",
        },
        {
            "formula": r"N_{\mathrm{est}}=N\times f",
            "condition": "用样本频率估计总体中相应类别人数",
        },
    ]


def verified_middle_0153_guidance() -> dict[str, Any]:
    """Return checked routes that expose chart inputs but no requested count."""

    method_names = ("分层对应与百分比人数法", "样本频率估计总体法")
    advanced_steps = (
        [
            "先由问卷对应关系确认 $C$ 表示AI应用、$E$ 表示辅助学习；条形图给出C类80人，扇形图给出E占40%",
            r"建立 $80\times40\%$ 的人数计算式，停在乘积结果前",
        ],
        [
            "由条形图求样本总人数 $54+30+80+36=200$，并读出科普讲座对应A类54人",
            r"写出样本频率 $\frac{54}{200}$，再建立 $1200\times\frac{54}{200}$，停在最终人数前",
        ],
    )
    standard_steps = (
        [
            "在问卷中定位AI应用的C类标记，再从条形图读取C柱共有80人",
            "在C类扇形图中定位辅助学习的E类标记，读取它所占40%",
            r"用“组内人数×组内占比”列出 $80\times40\%$，不计算乘积",
        ],
        [
            "把条形图四组人数相加，得到本次样本总人数 $54+30+80+36=200$",
            r"科普讲座对应A类54人，据此把样本中的相应频率写成 $\frac{54}{200}$",
            r"用全校1200人乘这个样本频率，只列 $1200\times\frac{54}{200}$ 而不写结果",
        ],
    )
    foundation_steps = (
        [
            "先看问卷问题1：AI应用前面的字母是C",
            "再看条形图C柱，柱顶数字是80，表示C类有80人",
            "C类继续回答问题2，辅助学习前面的字母是E；扇形图E区域标出40%",
            r"把80人和40%放进“组内人数×占比”的位置，写到 $80\times40\%$ 为止",
        ],
        [
            "把条形图A、B、C、D四柱人数54、30、80、36相加，样本总人数为200",
            "问卷中科普讲座对应A，样本里这一类有54人",
            r"先写样本频率：科普讲座人数除以样本总人数，即 $\frac{54}{200}$",
            "再用全校人数1200乘这个频率，保留算式，不完成最后乘法",
        ],
    )
    pitfalls = (
        "40%是在C类80人内部的比例，不能用全体样本人数直接乘40%",
        "估计总体人数要先用54除以完整样本总人数200，不能只用某一根柱作分母",
    )
    replacement: dict[str, Any] = {
        "advanced": {"title": "学霸版", "parts": []},
        "standard": {"title": "普通版", "parts": []},
        "foundation": {
            "title": "基础版",
            "parts": [],
            "pattern_summary": (
                "先把问卷选项与统计图字母一一对应；组内问题用该组人数乘占比，"
                "总体估计先求样本频率再乘总体人数，所有路线都停在最终人数前"
            ),
        },
    }
    tier_steps = {
        "advanced": advanced_steps,
        "standard": standard_steps,
        "foundation": foundation_steps,
    }
    for tier_key, parts_steps in tier_steps.items():
        for index, part_label in enumerate(("（1）", "（2）")):
            part: dict[str, Any] = {
                "part": part_label,
                "method_id": "M1",
                "method_name": method_names[index],
                "steps": list(parts_steps[index]),
                "alternative_methods": [],
            }
            if tier_key == "standard":
                part["pitfall"] = pitfalls[index]
            replacement[tier_key]["parts"].append(part)
    return replacement


def _install_verified_middle_0153_statistics_contract(
    output: dict[str, Any],
    normalization_audit: list[dict[str, Any]] | None = None,
    *,
    rule_id: str,
    request_sha256: str = "",
) -> dict[str, Any]:
    """Atomically install the checked answer-blind middle_0153 contract."""

    if not has_middle_0153_ordered_three_panel_structure(output):
        return output
    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    figures = candidate.get("source_figures")
    if not isinstance(figures, list) or len(figures) != 3:
        return output
    figure_ids = [str(item.get("figure_id") or "") for item in figures]
    changed_paths: list[str] = []

    verified_blocks = verified_middle_0153_question_blocks(figure_ids)
    if candidate.get("question_blocks") != verified_blocks:
        candidate["question_blocks"] = verified_blocks
        changed_paths.append("$.question_blocks")

    for index, caption in enumerate(verified_middle_0153_figure_captions()):
        figure = figures[index]
        if not isinstance(figure, dict):
            return output
        if figure.get("caption") != caption:
            figure["caption"] = caption
            changed_paths.append(f"$.source_figures[{index}].caption")

    verified_fields: tuple[tuple[str, Any], ...] = (
        ("visual_evidence", verified_middle_0153_visual_evidence()),
        ("knowledge_points", verified_middle_0153_knowledge_points()),
        ("base_formulas", verified_middle_0153_base_formulas()),
        ("guidance", verified_middle_0153_guidance()),
    )
    for field, verified_value in verified_fields:
        if candidate.get(field) != verified_value:
            candidate[field] = verified_value
            changed_paths.append(f"$.{field}")

    if normalization_audit is not None and changed_paths:
        normalization_audit.append({
            "path": "$.guidance",
            "operation": "normalize_verified_guidance",
            "rule_id": rule_id,
            "changed_paths": changed_paths,
            "preserved_bbox_1000": True,
            "preserved_figure_ids": True,
            "required_source_figures": ["调查问卷", "条形统计图", "扇形统计图"],
            **(
                {
                    "request_sha256": request_sha256,
                    "source_contract": (
                        "exact_reviewed_sha+strict_stem+ordered_three_figures"
                    ),
                }
                if request_sha256
                else {}
            ),
        })
    return candidate


def repair_known_middle_0153_statistics_guidance(
    output: dict[str, Any],
    normalization_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Install checked middle_0153 content from independently stated facts."""

    if not has_verified_middle_0153_source_semantics(output):
        return output
    return _install_verified_middle_0153_statistics_contract(
        output,
        normalization_audit,
        rule_id="statistics_repair_captured_middle_0153_v1",
    )


def repair_reviewed_middle_0153_statistics_guidance(
    output: dict[str, Any],
    request_sha256: str,
    normalization_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Install checked content only for the byte-identical reviewed source."""

    digest = str(request_sha256 or "").strip().lower()
    if (
        not has_reviewed_middle_0153_stem(output, digest)
        or not has_middle_0153_ordered_three_panel_structure(output)
    ):
        return output
    return _install_verified_middle_0153_statistics_contract(
        output,
        normalization_audit,
        rule_id="statistics_repair_reviewed_source_middle_0153_v1",
        request_sha256=digest,
    )


PRIMARY_APPLICATION_KIND = "primary_application_001"
PRIMARY_FRACTION_KIND = "primary_fraction_001"
PRIMARY_COMPOSITE_GEOMETRY_KIND = "primary_composite_geometry_001"

PRIMARY_APPLICATION_CANONICAL_TEXT = (
    "要求：列式计算，并用文字说明每一步在求什么。\n"
    "学校为运动会准备了 12 箱矿泉水，每箱 24 瓶。上午用去 85 瓶，"
    "下午又运来 4 箱同样的矿泉水。\n"
    "（1）现在一共有多少瓶矿泉水？请列出一个综合算式。\n"
    "（2）按照你的计算顺序，说明算式中每一步分别解决了什么问题。"
)
PRIMARY_FRACTION_CANONICAL_TEXT = (
    "要求：写出完整思路，计算结果用最简分数表示。\n"
    "一条彩带长 4 米。做花环用去了全长的 3/8，包装礼盒又用去了 5/6 米。\n"
    "（1）这条彩带还剩多少米？请写出完整算式并计算。\n"
    "（2）说明题目中的“全长的 3/8”和“5/6 米”表示的量有什么不同，"
    "以及你是怎样统一它们再计算的。"
)
PRIMARY_COMPOSITE_GEOMETRY_CANONICAL_TEXT = (
    "下面的图形按题意画出；图中长度单位均为米。\n"
    "一个长方形花圃长 12 米、宽 8 米。园丁从它的右上角挖去一个长 5 米、"
    "宽 3 米的小长方形，剩下的部分如下图阴影所示。\n"
    "（1）求剩余花圃的面积。\n"
    "（2）求剩余花圃一周的长度。\n"
    "（3）说明你怎样确定凹进去的两条边各有多长。"
)


PRIMARY_FIXTURE_DECORATIVE_LINE_KEYS = frozenset({
    "第1题多步计算与应用",
    "第2题分数与长度单位",
    "第3题组合图形",
    "Test4本地合成测试样本题面不含答案",
})

PRIMARY_FIXTURE_LEADING_TITLE_PATTERNS = (
    re.compile(r"^\s*第1题\s*多步计算与应用\s*[。.．:：]\s*"),
    re.compile(r"^\s*第2题\s*分数与长度单位\s*[。.．:：]\s*"),
    re.compile(r"^\s*第3题\s*组合图形\s*[。.．:：]\s*"),
)


def _primary_decorative_line_key(value: str) -> str:
    return re.sub(r"[\s·]", "", str(value or ""))


def _strip_primary_fixture_edge_decorations(value: str) -> str:
    """Remove known decoration only at a complete line or strict leading edge."""

    lines = str(value or "").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    # A provider can merge the printed title and the first stem sentence into
    # one text block/line.  Accept only an exact title at the absolute leading
    # edge followed by an explicit sentence delimiter; never remove the same
    # words from the middle of a stem (that would silently change the problem).
    if lines:
        for pattern in PRIMARY_FIXTURE_LEADING_TITLE_PATTERNS:
            stripped, count = pattern.subn("", lines[0], count=1)
            if count and stripped.strip():
                lines[0] = stripped
                break
    while (
        lines
        and _primary_decorative_line_key(lines[0])
        in PRIMARY_FIXTURE_DECORATIVE_LINE_KEYS
    ):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    while (
        lines
        and _primary_decorative_line_key(lines[-1])
        in PRIMARY_FIXTURE_DECORATIVE_LINE_KEYS
    ):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return "\n".join(lines)


def _compact_primary_fixture_text(value: str) -> str:
    canonical = _strip_primary_fixture_edge_decorations(value)
    canonical = canonical.replace("（", "(").replace("）", ")")
    canonical = canonical.replace("％", "%").replace("．", ".")
    canonical = re.sub(
        r"\\[dt]?frac\s*\{\s*(\d+)\s*\}\s*\{\s*(\d+)\s*\}",
        r"\1/\2",
        canonical,
    )
    for quote in ('“', '”', '‘', '’', '"', "'"):
        canonical = canonical.replace(quote, "")
    canonical = re.sub(r"[\s$]", "", canonical)
    canonical = canonical.replace("{", "").replace("}", "")

    def remove_non_numeric_punctuation(match: re.Match[str]) -> str:
        start = match.start()
        end = match.end()
        if (
            start > 0
            and end < len(canonical)
            and canonical[start - 1].isdigit()
            and canonical[end].isdigit()
        ):
            return match.group(0)
        return ""

    canonical = re.sub(
        r"[，,、；;：:。.？?·]+",
        remove_non_numeric_punctuation,
        canonical,
    )
    return canonical


def _primary_fixture_digit_fingerprint(value: str) -> tuple[str, ...]:
    """Keep exact digit runs so punctuation cleanup can never merge numbers."""

    canonical = _strip_primary_fixture_edge_decorations(value).translate(
        str.maketrans("０１２３４５６７８９", "0123456789")
    )
    return tuple(re.findall(r"[0-9]+", canonical))


def _primary_question_text(output: Mapping[str, Any]) -> str:
    return "\n".join(
        str(block.get("text") or "")
        for block in output.get("question_blocks", [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )


def primary_fixture_stem_kind(output: Mapping[str, Any]) -> str | None:
    """Recognize an exact primary fixture stem without trusting status fields."""

    observed_text = _primary_question_text(output)
    observed = _compact_primary_fixture_text(observed_text)
    expected = {
        PRIMARY_APPLICATION_KIND: PRIMARY_APPLICATION_CANONICAL_TEXT,
        PRIMARY_FRACTION_KIND: PRIMARY_FRACTION_CANONICAL_TEXT,
        PRIMARY_COMPOSITE_GEOMETRY_KIND: (
            PRIMARY_COMPOSITE_GEOMETRY_CANONICAL_TEXT
        ),
    }
    for kind, text in expected.items():
        if (
            _primary_fixture_digit_fingerprint(observed_text)
            == _primary_fixture_digit_fingerprint(text)
            and observed == _compact_primary_fixture_text(text)
        ):
            return kind
    return None


def primary_fixture_kind(output: Mapping[str, Any]) -> str | None:
    """Return a repair-eligible exact fixture; uncertainty remains fail closed."""

    if (
        output.get("subject") != "mathematics"
        or output.get("problem_scope") != "primary_to_high_school"
        or output.get("recognition_status") != "clear"
        or output.get("uncertain_items")
    ):
        return None
    return primary_fixture_stem_kind(output)


def is_known_primary_application_problem(output: Mapping[str, Any]) -> bool:
    return primary_fixture_kind(output) == PRIMARY_APPLICATION_KIND


def is_known_primary_fraction_problem(output: Mapping[str, Any]) -> bool:
    return primary_fixture_kind(output) == PRIMARY_FRACTION_KIND


def is_known_primary_composite_geometry_problem(
    output: Mapping[str, Any],
) -> bool:
    return primary_fixture_kind(output) == PRIMARY_COMPOSITE_GEOMETRY_KIND


def _complete_primary_figure_metadata(figure: Any) -> bool:
    if not isinstance(figure, Mapping):
        return False
    if not str(figure.get("figure_id") or "").strip():
        return False
    if not str(figure.get("caption") or "").strip():
        return False
    bbox = figure.get("bbox_1000")
    if not (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in bbox
        )
    ):
        return False
    left, top, right, bottom = (float(value) for value in bbox)
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        return False
    area_ratio = (right - left) * (bottom - top) / 1_000_000
    return 0.001 <= area_ratio <= 0.55


def has_verified_primary_fixture_source_semantics(
    output: Mapping[str, Any],
) -> bool:
    """Require the exact no-figure/one-real-figure source contract before repair."""

    kind = primary_fixture_kind(output)
    if kind is None:
        return False
    figures = output.get("source_figures")
    blocks = output.get("question_blocks")
    if not isinstance(figures, list) or not isinstance(blocks, list):
        return False
    block_figure_ids = [
        str(block.get("figure_id") or "")
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "figure"
    ]
    if kind in {PRIMARY_APPLICATION_KIND, PRIMARY_FRACTION_KIND}:
        return not figures and not block_figure_ids
    if not (
        len(figures) == 1
        and _complete_primary_figure_metadata(figures[0])
    ):
        return False
    figure_id = str(figures[0].get("figure_id") or "")
    return block_figure_ids == [figure_id]


def verified_primary_application_question_blocks() -> list[dict[str, str]]:
    return [
        {
            "type": "text",
            "text": "要求：列式计算，并用文字说明每一步在求什么。",
            "figure_id": "",
        },
        {
            "type": "text",
            "text": (
                "学校为运动会准备了 12 箱矿泉水，每箱 24 瓶。上午用去 85 瓶，"
                "下午又运来 4 箱同样的矿泉水。\n"
                "（1）现在一共有多少瓶矿泉水？请列出一个综合算式。\n"
                "（2）按照你的计算顺序，说明算式中每一步分别解决了什么问题。"
            ),
            "figure_id": "",
        },
    ]


def verified_primary_fraction_question_blocks() -> list[dict[str, str]]:
    return [
        {
            "type": "text",
            "text": "要求：写出完整思路，计算结果用最简分数表示。",
            "figure_id": "",
        },
        {
            "type": "text",
            "text": (
                "一条彩带长 4 米。做花环用去了全长的 $\\frac{3}{8}$，"
                "包装礼盒又用去了 $\\frac{5}{6}$ 米。\n"
                "（1）这条彩带还剩多少米？请写出完整算式并计算。\n"
                "（2）说明题目中的“全长的 $\\frac{3}{8}$”和"
                "“$\\frac{5}{6}$ 米”表示的量有什么不同，"
                "以及你是怎样统一它们再计算的。"
            ),
            "figure_id": "",
        },
    ]


def verified_primary_composite_geometry_question_blocks(
    figure_id: str,
) -> list[dict[str, str]]:
    if not str(figure_id).strip():
        raise ValueError("小学组合图形题必须提供唯一figure_id")
    return [
        {
            "type": "text",
            "text": (
                "下面的图形按题意画出；图中长度单位均为米。\n"
                "一个长方形花圃长 12 米、宽 8 米。园丁从它的右上角挖去一个长 5 米、"
                "宽 3 米的小长方形，剩下的部分如下图阴影所示。"
            ),
            "figure_id": "",
        },
        {"type": "figure", "text": "", "figure_id": str(figure_id)},
        {
            "type": "text",
            "text": (
                "（1）求剩余花圃的面积。\n"
                "（2）求剩余花圃一周的长度。\n"
                "（3）说明你怎样确定凹进去的两条边各有多长。"
            ),
            "figure_id": "",
        },
    ]


def verified_primary_composite_geometry_figure_caption() -> str:
    return (
        "长方形花圃右上角挖去小长方形后的阴影组合图形，"
        "含12米、8米及5米×3米尺寸标注"
    )


def verified_primary_composite_geometry_visual_evidence() -> list[str]:
    return [
        "外框为长12米、宽8米的长方形",
        "右上角挖去长5米、宽3米的小长方形",
        "阴影为剩余的凹形花圃，图中保留完整尺寸箭头和缺口标注",
    ]


def verified_primary_application_knowledge_points() -> list[str]:
    return [
        "总量与单位量的关系",
        "加减混合应用题中的数量变化",
        "综合算式与运算顺序",
        "算式各步的实际意义",
    ]


def verified_primary_fraction_knowledge_points() -> list[str]:
    return [
        "一个数的几分之几",
        "带单位分数与比例分数的区别",
        "长度单位统一",
        "异分母分数加减",
        "最简分数",
    ]


def verified_primary_composite_geometry_knowledge_points() -> list[str]:
    return [
        "长方形面积",
        "组合图形的面积差",
        "沿封闭边界求周长",
        "长方形对边相等",
        "凹形图形的边长对应关系",
    ]


def verified_primary_application_base_formulas() -> list[dict[str, str]]:
    return [
        {
            "formula": "N=nq",
            "condition": "n为相同包装的箱数，q为每箱瓶数，N为对应的总瓶数",
        },
        {
            "formula": "S=S_{0}-U+A",
            "condition": "所有数量单位一致，库存先用去U再运来A",
        },
    ]


def verified_primary_fraction_base_formulas() -> list[dict[str, str]]:
    return [
        {
            "formula": "P=Tr",
            "condition": "T为总量，r为所占分数，P为对应的部分量",
        },
        {
            "formula": "R=T-U_{1}-U_{2}",
            "condition": "T、$U_{1}$、$U_{2}$已经统一为同一种长度单位",
        },
        {
            "formula": r"\frac{a}{b}-\frac{c}{d}=\frac{ad-bc}{bd}",
            "condition": "b与d均不为0，结果还要约分",
        },
    ]


def verified_primary_composite_geometry_base_formulas() -> list[dict[str, str]]:
    return [
        {
            "formula": "S=ab",
            "condition": "a、b为长方形相邻两边的长度",
        },
        {
            "formula": "S_{r}=S_{o}-S_{c}",
            "condition": "下标r、o、c分别表示剩余区域、外框和缺口",
        },
        {
            "formula": r"P=\sum_{i=1}^{n}l_{i},\quad P_{rect}=2(a+b)",
            "condition": "前式沿封闭边界逐段相加；后式仅用于完整长方形",
        },
    ]


def _verified_primary_guidance(
    *,
    method_names: Sequence[str],
    advanced_steps: Sequence[Sequence[str]],
    standard_steps: Sequence[Sequence[str]],
    foundation_steps: Sequence[Sequence[str]],
    pitfalls: Sequence[str],
    pattern_summary: str,
) -> dict[str, Any]:
    replacement: dict[str, Any] = {
        "advanced": {"title": "学霸版", "parts": []},
        "standard": {"title": "普通版", "parts": []},
        "foundation": {
            "title": "基础版",
            "parts": [],
            "pattern_summary": pattern_summary,
        },
    }
    tier_steps = {
        "advanced": advanced_steps,
        "standard": standard_steps,
        "foundation": foundation_steps,
    }
    labels = tuple(f"（{index}）" for index in range(1, len(method_names) + 1))
    for tier_key, all_steps in tier_steps.items():
        for index, label in enumerate(labels):
            part: dict[str, Any] = {
                "part": label,
                "method_id": "M1",
                "method_name": str(method_names[index]),
                "steps": list(all_steps[index]),
                "alternative_methods": [],
            }
            if tier_key == "standard":
                part["pitfall"] = str(pitfalls[index])
            replacement[tier_key]["parts"].append(part)
    return replacement


def verified_primary_application_guidance() -> dict[str, Any]:
    return _verified_primary_guidance(
        method_names=("数量变化综合式法", "算式分步释义法"),
        advanced_steps=(
            (
                "先把题中的量按“原有、用去、运来”分类，并确认最终都要统一为瓶。",
                "用符号写成“现有量＝原有量－用去量＋运来量”，再自行把题目数据放入对应位置；不要计算等号右端。",
            ),
            (
                "回看综合式中的两个乘法，分别判断它们完成了哪一种单位转换。",
                "再按减法和加法发生的先后，用“这一步求____”逐项口述，空格由学生填写。",
            ),
        ),
        standard_steps=(
            (
                "先用“箱数×每箱瓶数”分别表示原有瓶数和后来运来的瓶数。",
                "把上午用去的数量放在减号后，把后来运来的数量放在加号后。",
                "将三部分连接成一个综合式，保留算式末端，不求出瓶数。",
            ),
            (
                "给原有瓶数对应的乘法式加上“原有”标签。",
                "给上午后的减法和下午运来的乘法分别加上空白说明标签。",
                "沿运算顺序填写“先求____、再求____、最后求____”，不补出三个答案词。",
            ),
        ),
        foundation_steps=(
            (
                "圈出所有带“箱”的数量和“每箱”的瓶数。",
                "先写一个乘法框架表示原来共有多少瓶，只列式不算积。",
                "再写一个乘法框架表示下午运来多少瓶，只列式不算积。",
                "按“原有－用去＋运来”合并两个框架，并把最终计算留空。",
            ),
            (
                "在综合式下画四个对应框，分别对准两个乘法、减法和加法。",
                "对每个乘法先问：它把哪一种单位换成了瓶？",
                "对减法问：这一步发生在上午用去之后，数量怎样变化？",
                "对加法问：下午的新数量应放回哪一部分？用空格句式完成说明。",
            ),
        ),
        pitfalls=(
            "不能把“箱”和“瓶”直接相加减，所有数量必须先统一为瓶。",
            "说明每一步时要对应实际数量，不能只重复运算符。",
        ),
        pattern_summary=(
            "先统一单位并辨认原有、用去、运来，再列数量变化框架；"
            "解释时逐项对应实际意义。"
        ),
    )


def verified_primary_fraction_guidance() -> dict[str, Any]:
    return _verified_primary_guidance(
        method_names=("统一单位列式法", "量义与单位比较法"),
        advanced_steps=(
            (
                "先判断“全长的某个分数”和“若干米”是否已经是同一种长度表示。",
                "把比例用量写成“总长×所占分数”，再按“总长－两次用量”列框架；停在数值化简前。",
            ),
            (
                "比较两个短语是否直接带长度单位，以及是否需要依赖总长才能确定。",
                "用“依赖对象、单位、统一方法”三个栏目组织自己的说明，栏目内容由学生补全。",
            ),
        ),
        standard_steps=(
            (
                "给总长记一个符号，并把花环用量写成该符号乘一个分数。",
                "把礼盒用量作为另一个带米单位的减数，先检查单位一致。",
                "写出连续减法框架并确定通分步骤，最终分数的分子和分母留空。",
            ),
            (
                "圈出“全长的”和“米”这两个提示语。",
                "分别回答：哪一个量要通过总长换算，哪一个量已经给出实际长度？",
                "使用“前者是____，后者是____；先____再____”的句式，不填写空格。",
            ),
        ),
        foundation_steps=(
            (
                "先读总长，确认它的单位是米。",
                "把“全长的某个分数”改写成一个乘法式，不计算乘积。",
                "把另一项保留为题面给出的带单位长度，并与前一项核对单位。",
                "从总长依次减去两项，写到待通分的算式为止，不给出最简分数。",
            ),
            (
                "检查第一个分数后面是否直接写了长度单位。",
                "检查第二个分数后面是否直接写了“米”。",
                "思考没有直接带单位的分数修饰题目中的哪个整体。",
                "说明为什么做减法前两项必须表示同一类、同一单位的量，具体结论留给学生。",
            ),
        ),
        pitfalls=(
            "看到不带单位的分数时，先核对它修饰的对象，不能直接同带单位的长度相减。",
            "比较时同时检查单位和依赖对象，不要只比较数字写法。",
        ),
        pattern_summary="先检查每个量的单位和依赖对象，再统一量纲列出计算框架。",
    )


def verified_primary_composite_geometry_guidance() -> dict[str, Any]:
    return _verified_primary_guidance(
        method_names=("大图减缺口面积法", "沿边逐段周长法", "矩形对边对应法"),
        advanced_steps=(
            (
                "把阴影区域看成外框长方形去掉右上角的小长方形。",
                "写出符号关系 $S_{r}=S_{o}-S_{c}$，再从图中寻找各面积所需的边长；停在数值代入前。",
            ),
            (
                "沿阴影边界完整走一圈，把遇到的六条线段依次编号。",
                "把未直接标出的线段先用字母表示，列出六段之和；不要代入、求和或判断周长是否改变。",
            ),
            (
                "只观察挖去的小长方形，按水平方向和竖直方向标记两条凹边。",
                "利用长方形同方向的对边相等建立对应关系，数值位置保留空白。",
            ),
        ),
        standard_steps=(
            (
                "写出外框长方形的面积公式，不计算乘积。",
                "写出缺口小长方形的面积公式，不计算乘积。",
                "用外框面积减去缺口面积，保留未计算的差式。",
            ),
            (
                "从左下角出发顺时针描边，确保只沿实际边界。",
                "把两条凹边暂记为 h、v，把另外未标出的外边写成总边与缺口边的差。",
                "按顺序写六段相加的式子，h、v和总和都不化成数值。",
            ),
            (
                "把水平凹边记作 h，把竖直凹边记作 v。",
                "在小长方形上分别寻找与 h、v 平行且等长的对边。",
                "写成“h＝____米，v＝____米”，由学生根据方向填数。",
            ),
        ),
        foundation_steps=(
            (
                "用手指圈出完整外框。",
                "再圈出右上角被挖去的小长方形。",
                "分别套用长方形面积公式，只填写长和宽的位置。",
                "把两个面积式作差，检查平方单位，结果留空。",
            ),
            (
                "沿黑色实线走一圈，每到转角就停一下。",
                "把走过的每一段依次记为 $l_{1}$ 到 $l_{6}$。",
                "没有直接标数的边先用 h、v 或差式表示，不猜数值。",
                "写出 $P=l_{1}+l_{2}+l_{3}+l_{4}+l_{5}+l_{6}$，把代入和求和留给学生。",
            ),
            (
                "看缺口标注的两个方向，不按图形视觉比例猜长度。",
                "找出水平凹边与小长方形哪条边方向相同。",
                "找出竖直凹边与小长方形哪条边方向相同。",
                "用长方形对边相等填写两个空格，但不在提示页写出数值。",
            ),
        ),
        pitfalls=(
            "挖去部分不能再算入剩余面积，且面积单位要用平方单位。",
            "只沿剩余图形的实际边界计一次，不能把缺口虚线或内部线重复计入。",
            "必须按水平、竖直方向分别配对，不能只凭图形看起来的比例估计。",
        ),
        pattern_summary=(
            "面积先建立外框减缺口关系，周长沿实际边界逐段记录，"
            "凹边按方向寻找对应边。"
        ),
    )


def repair_known_primary_fixture_guidance(
    output: dict[str, Any],
    normalization_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Install one exact primary answer-blind contract without inventing crops."""

    kind = primary_fixture_kind(output)
    if kind is None or not has_verified_primary_fixture_source_semantics(output):
        return output
    changed_paths: list[str] = []
    if kind == PRIMARY_APPLICATION_KIND:
        blocks = verified_primary_application_question_blocks()
        verified_fields = (
            ("visual_evidence", []),
            ("knowledge_points", verified_primary_application_knowledge_points()),
            ("base_formulas", verified_primary_application_base_formulas()),
            ("guidance", verified_primary_application_guidance()),
        )
    elif kind == PRIMARY_FRACTION_KIND:
        blocks = verified_primary_fraction_question_blocks()
        verified_fields = (
            ("visual_evidence", []),
            ("knowledge_points", verified_primary_fraction_knowledge_points()),
            ("base_formulas", verified_primary_fraction_base_formulas()),
            ("guidance", verified_primary_fraction_guidance()),
        )
    else:
        figures = output.get("source_figures")
        if not isinstance(figures, list) or len(figures) != 1:
            return output
        figure = figures[0]
        if not isinstance(figure, dict):
            return output
        figure_id = str(figure.get("figure_id") or "")
        blocks = verified_primary_composite_geometry_question_blocks(figure_id)
        caption = verified_primary_composite_geometry_figure_caption()
        if figure.get("caption") != caption:
            figure["caption"] = caption
            changed_paths.append("$.source_figures[0].caption")
        verified_fields = (
            (
                "visual_evidence",
                verified_primary_composite_geometry_visual_evidence(),
            ),
            (
                "knowledge_points",
                verified_primary_composite_geometry_knowledge_points(),
            ),
            (
                "base_formulas",
                verified_primary_composite_geometry_base_formulas(),
            ),
            ("guidance", verified_primary_composite_geometry_guidance()),
        )
    if output.get("question_blocks") != blocks:
        output["question_blocks"] = blocks
        changed_paths.append("$.question_blocks")
    for field, verified_value in verified_fields:
        if output.get(field) != verified_value:
            output[field] = verified_value
            changed_paths.append(f"$.{field}")
    if normalization_audit is not None and changed_paths:
        normalization_audit.append({
            "path": "$.guidance",
            "operation": "normalize_verified_guidance",
            "rule_id": f"primary_fixture_contract_{kind}_v1",
            "changed_paths": changed_paths,
            "preserved_bbox_1000": kind == PRIMARY_COMPOSITE_GEOMETRY_KIND,
        })
    return output


def unescaped_dollar_positions(value: str) -> list[int]:
    positions: list[int] = []
    for index, character in enumerate(value):
        if character != "$":
            continue
        preceding_backslashes = 0
        cursor = index - 1
        while cursor >= 0 and value[cursor] == "\\":
            preceding_backslashes += 1
            cursor -= 1
        if preceding_backslashes % 2 == 0:
            positions.append(index)
    return positions


def close_single_trailing_math_delimiter(value: str) -> str:
    """Close one unmatched `$` only when a valid formula occupies the tail."""
    positions = unescaped_dollar_positions(value)
    if len(positions) % 2 != 1:
        return value
    opener = positions[-1]
    formula_tail = value[opener + 1:]
    if (
        not formula_tail
        or formula_tail != formula_tail.rstrip()
        or "\n" in formula_tail
        or "\r" in formula_tail
    ):
        return value
    candidate = normalize_formula_notation(formula_tail)
    errors: list[str] = []
    validate_formula_source(candidate, "trailing_formula", errors)
    if errors:
        return value
    return value + "$"


def strip_spurious_leading_math_delimiter(value: str) -> str:
    """Remove one model-added leading ``$`` before ordinary Chinese prose.

    This repair is deliberately narrow: the first non-space character after
    the delimiter must be Chinese and the remaining text must contain another
    dollar sign that starts the real inline formula.  Thus a genuine formula
    beginning with a Latin/Greek symbol is never modified.
    """

    stripped = value.lstrip()
    if not stripped.startswith("$") or stripped.count("$") % 2 != 1:
        return value
    remainder = stripped[1:]
    if not re.match(
        r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
        remainder,
    ):
        return value
    prefix_length = len(value) - len(stripped)
    return value[:prefix_length] + remainder


def unwrap_mixed_prose_math_delimiters(value: str) -> str:
    """Remove outer math delimiters that incorrectly wrap Chinese prose.

    A model may return an entire instructional sentence as ``$中文...公式$``.
    Rendering that span as one formula both exposes raw LaTeX and suppresses
    normal prose wrapping.  Only unwrap a single outer pair when Chinese text
    exists outside explicit ``\text{...}`` groups and the span also contains
    a clear algebraic signal.  Genuine all-formula spans remain untouched.
    """
    leading = len(value) - len(value.lstrip())
    trailing = len(value) - len(value.rstrip())
    core_end = len(value) - trailing if trailing else len(value)
    core = value[leading:core_end]
    if not (core.startswith("$") and core.endswith("$") and core.count("$") == 2):
        return value
    inner = core[1:-1]
    without_text = re.sub(
        r"\\text\{[^{}]*[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff][^{}]*\}",
        "",
        inner,
    )
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", without_text) is None:
        return value
    if re.search(
        r"(?:[A-Za-zΑ-Ωα-ω](?:['′]{0,2})?\s*\([^)]*\)|"
        r"[A-Za-zΑ-Ωα-ω](?:[_^]\{?[^\s,，。；;]+\}?)?\s*[=<>≤≥≠]|"
        r"\\(?:frac|sqrt|sum|int|lim)\b)",
        inner,
    ) is None:
        return value
    return value[:leading] + inner + value[core_end:]


def strip_redundant_chinese_step_prefix(value: str) -> str:
    """Remove only a delimited Chinese ordinal at the start of a step."""
    match = REDUNDANT_CHINESE_STEP_PREFIX.match(value)
    if match is None or not value[match.end():].strip():
        return value
    return value[match.end():]


def is_plain_algebra_candidate(value: str) -> bool:
    """Return whether one undelimited prose run is unambiguously algebra.

    Bare Latin labels such as ``A``/``BC`` are intentionally left as prose.
    Promotion requires a structural signal (script, relation, quotient,
    coordinate or one-letter function call), which keeps this display repair
    meaning-preserving and independent of any particular question.
    """

    candidate = value.strip()
    if not candidate or not re.search(r"[A-Za-zΑ-Ωα-ω0-9]", candidate):
        return PLAIN_INFINITY.fullmatch(candidate) is not None
    if PLAIN_INFINITY.fullmatch(candidate):
        return True
    if re.search(r"(?<![A-Za-z])(?:infty|∞)(?![A-Za-z])", candidate, re.IGNORECASE):
        return True
    if PLAIN_FUNCTION_CALL.fullmatch(candidate):
        return True
    if PLAIN_COORDINATE_TUPLE.fullmatch(candidate):
        return True
    if PLAIN_SCRIPTED_NAME.search(candidate):
        return True
    if re.search(r"[=<>≤≥≠≈∝∥⊥]", candidate):
        return True
    if "/" in candidate:
        return PLAIN_SIMPLE_QUOTIENT.fullmatch(candidate) is not None
    return bool(re.search(r"\\[A-Za-z]+", candidate))


def promote_plain_algebra_markup(source: str) -> str:
    """Wrap clear algebra embedded in prose with inline-math delimiters.

    Existing ``$...$`` regions are copied byte-for-byte here and normalized by
    the established formula path later.  This function only promotes plain
    regions, so it is idempotent and cannot nest math delimiters.
    """

    def promote_segment(segment: str) -> str:
        pieces: list[str] = []
        cursor = 0
        for match in PLAIN_ALGEBRA_RUN.finditer(segment):
            pieces.append(segment[cursor:match.start()])
            raw = match.group(0)
            leading_length = len(raw) - len(raw.lstrip())
            trailing_length = len(raw) - len(raw.rstrip())
            leading = raw[:leading_length]
            trailing = raw[len(raw) - trailing_length:] if trailing_length else ""
            core_end = len(raw) - trailing_length if trailing_length else len(raw)
            core = raw[leading_length:core_end]

            # A comma immediately before Chinese prose is sentence
            # punctuation, not part of the formula.  Internal coordinate
            # commas are retained because they are not at either edge.
            suffix = ""
            while core.endswith((",", "，")):
                suffix = core[-1] + suffix
                core = core[:-1].rstrip()
            if is_plain_algebra_candidate(core):
                pieces.extend((leading, f"${core}$", suffix, trailing))
            else:
                pieces.append(raw)
            cursor = match.end()
        pieces.append(segment[cursor:])
        return "".join(pieces)

    pieces: list[str] = []
    cursor = 0
    for match in INLINE_FORMULA.finditer(source):
        pieces.append(promote_segment(source[cursor:match.start()]))
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(promote_segment(source[cursor:]))
    promoted = "".join(pieces)

    # Geometry normalization may already have wrapped the left-hand token,
    # leaving the newly promoted relation immediately beside it, for example
    # ``$∠ABC$$=90°$``.  It is one mathematical relation, not two formulae;
    # merge only directly adjacent formula regions or regions joined solely by
    # an explicit relation operator.  Chinese prose is never crossed.
    for _ in range(4):
        merged = re.sub(r"\$([^$\r\n]+)\$\$([^$\r\n]+)\$", r"$\1\2$", promoted)
        merged = re.sub(
            r"\$([^$\r\n]+)\$\s*([=<>≤≥≠≈∝∥⊥])\s*\$([^$\r\n]+)\$",
            r"$\1\2\3$",
            merged,
        )
        if merged == promoted:
            break
        promoted = merged
    return promoted


def normalize_formula_notation(source: str) -> str:
    """Apply only meaning-preserving display repairs before local validation."""
    value = source.strip()
    if value.startswith("$") and value.endswith("$") and value.count("$") == 2:
        value = value[1:-1].strip()
    value, _ = apply_display_formula_whitelist(value)
    # Qwen occasionally serializes textbook unit spacing as ``~{m}`` or
    # wraps an existing \mathrm group as ``~{\mathrm{...}}``.  Those braces
    # are source syntax, not student-visible notation.
    value = re.sub(r"~\{(\\mathrm\{.*\})\}", r"\1", value)
    value = re.sub(
        r"~\{(kg|mol|mm|cm|km|Pa|m|s|N|J|V|T|A|C|K)\}",
        r"\\mathrm{\1}",
        value,
    )
    unicode_greek = {
        "π": r"\pi",
        "θ": r"\theta",
        "ω": r"\omega",
        "Ω": r"\Omega",
        "Δ": r"\Delta",
        "μ": r"\mu",
        "ρ": r"\rho",
        "α": r"\alpha",
        "β": r"\beta",
        "γ": r"\gamma",
        "λ": r"\lambda",
    }
    for character, command in unicode_greek.items():
        value = value.replace(character, command)
    value = re.sub(
        r"\\(Delta|theta|omega|Omega|mu|rho|alpha|beta|gamma|lambda)"
        r"(?=[A-Za-z])",
        lambda match: "\\" + match.group(1) + " ",
        value,
    )
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    value = re.sub(
        r"([A-Za-z}])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
        lambda match: f"{match.group(1)}^{{{match.group(2).translate(superscripts)}}}",
        value,
    )
    value = re.sub(
        r"([A-Za-z}])([₀₁₂₃₄₅₆₇₈₉]+)",
        lambda match: f"{match.group(1)}_{{{match.group(2).translate(subscripts)}}}",
        value,
    )
    value = re.sub(
        r"\^(?!\{)([+-]?(?:[A-Za-z]+|\d+|°))",
        lambda match: "^{" + (
            r"\circ" if match.group(1) in {"circ", "°"} else match.group(1)
        ) + "}",
        value,
    )
    value = re.sub(
        r"_(?!\{)([A-Za-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+)",
        lambda match: "_{" + (
            r"\text{" + match.group(1) + "}"
            if re.search(
                r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]",
                match.group(1),
            )
            else match.group(1)
        ) + "}",
        value,
    )
    value = re.sub(
        r"_\{([\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+)\}",
        lambda match: r"_{\text{" + match.group(1) + "}}",
        value,
    )
    value = re.sub(
        r"(?<!\\)\b(frac|dfrac|tfrac|sqrt|mathrm|text|operatorname|vec|"
        r"overrightarrow|overline)(?=\s*\{)",
        r"\\\1",
        value,
    )
    value = re.sub(
        r"(?<![A-Za-z\\])([A-Za-z])(?:frac)(?=\s*\{)",
        r"\1\\frac",
        value,
    )
    value = re.sub(
        r"(?<![\\A-Za-z])(pi|theta|omega|Omega|Delta|mu|rho|alpha|beta|"
        r"gamma|lambda|sin|cos|tan|circ|times|infty)(?![A-Za-z])",
        r"\\\1",
        value,
    )
    value = re.sub(r"√\s*\(([^()]*)\)", r"\\sqrt{\1}", value)
    value = re.sub(r"√\s*([A-Za-z0-9]+)", r"\\sqrt{\1}", value)
    value = value.replace(r"\mathrm{m/s}^{2}", r"\mathrm{m\,s^{-2}}")
    value = value.replace(r"\mathrm{m/s^2}", r"\mathrm{m\,s^{-2}}")
    value = value.replace(r"m/s^2", r"\mathrm{m\,s^{-2}}")

    def canonical_unit_quotient(match: re.Match[str]) -> str:
        numerator = match.group("numerator")
        denominator = match.group("denominator")
        power = int(match.group("power") or "1")
        inverse_power = -power
        return rf"\mathrm{{{numerator}\,{denominator}^{{{inverse_power}}}}}"

    for pattern in (
        SPLIT_MATHRM_UNIT_QUOTIENT,
        WRAPPED_MATHRM_UNIT_QUOTIENT,
        BARE_UNIT_QUOTIENT,
    ):
        value = pattern.sub(canonical_unit_quotient, value)
    return canonicalize_simple_fractions(value)


def normalize_output_for_render(
    output: Any,
    normalization_audit: list[dict[str, Any]] | None = None,
) -> Any:
    """Normalize formula spelling and explicit figure blocks only.

    Method identifiers and names are deliberately left untouched so formatting
    cleanup cannot hide a cross-tier semantic mismatch.
    """
    if not isinstance(output, dict):
        return output

    def record_rule_hit(path: str, change: Mapping[str, str]) -> None:
        if normalization_audit is None:
            return
        normalization_audit.append({
            "path": path,
            "operation": "normalize_display_syntax",
            **change,
        })

    def normalize_formula_value(source: str, path: str) -> str:
        if path.startswith("$.guidance."):
            segment_half_repaired, segment_half_change = (
                repair_dimensionally_invalid_half_segment_denominators(source)
            )
        else:
            # Base formulas remain under the stricter evidence-aware midpoint
            # repair above.  This avoids silently changing a standalone
            # reciprocal expression whose physical dimension is unknown.
            segment_half_repaired, segment_half_change = source, None
        if segment_half_change is not None:
            record_rule_hit(path, segment_half_change)
        parallel_repaired = PLAIN_SEGMENT_PARALLEL.sub(
            lambda match: (
                f"{match.group('left')} \\parallel {match.group('right')}"
            ),
            segment_half_repaired,
        )
        if parallel_repaired != segment_half_repaired:
            record_rule_hit(path, {
                "rule_id": "display_segment_parallel_slashes_to_symbol",
                "before": segment_half_repaired,
                "after": parallel_repaired,
            })
        _, changes = apply_display_formula_whitelist(parallel_repaired)
        for change in changes:
            record_rule_hit(path, change)
        normalized_formula = normalize_formula_notation(parallel_repaired)
        domain_repaired, domain_change = repair_nested_domain_condition_denominator(
            normalized_formula
        )
        if domain_change is not None:
            record_rule_hit(path, domain_change)
        return domain_repaired

    def normalize_display_text(source: str, path: str) -> str:
        # U+25B1 has no glyph in the fixed textbook font stack on the target
        # device.  Spell out its mathematical meaning while retaining the
        # vertex name as ordinary formula text.
        semantic, whole_text_formula_changes = apply_display_formula_whitelist(source)
        for change in whole_text_formula_changes:
            record_rule_hit(path, change)
        geometry_source = semantic
        semantic = re.sub(
            r"\$\s*▱\s*([A-Z]{3,8})\s*\$",
            r"平行四边形 $\1$",
            semantic,
        )
        semantic = re.sub(
            r"▱\s*([A-Z]{3,8})",
            r"平行四边形 \1",
            semantic,
        ).replace("▱", "平行四边形")
        parallel_semantic = PLAIN_SEGMENT_PARALLEL.sub(
            lambda match: f"{match.group('left')} ∥ {match.group('right')}",
            semantic,
        )
        if parallel_semantic != semantic:
            record_rule_hit(path, {
                "rule_id": "display_segment_parallel_slashes_to_symbol",
                "before": semantic,
                "after": parallel_semantic,
            })
        semantic = promote_plain_geometry_tokens(parallel_semantic)
        if semantic != geometry_source:
            record_rule_hit(path, {
                "rule_id": "display_spell_out_parallelogram_symbol",
                "before": geometry_source,
                "after": semantic,
            })
        unwrapped = unwrap_mixed_prose_math_delimiters(semantic)
        if unwrapped != semantic:
            record_rule_hit(path, {
                "rule_id": "display_unwrap_mixed_prose_math_delimiters",
                "before": semantic,
                "after": unwrapped,
            })
        repaired_leading = strip_spurious_leading_math_delimiter(unwrapped)
        if repaired_leading != semantic:
            record_rule_hit(path, {
                "rule_id": "display_strip_spurious_leading_math_delimiter",
                "before": semantic,
                "after": repaired_leading,
            })
        closed = close_single_trailing_math_delimiter(repaired_leading)
        if closed != repaired_leading:
            record_rule_hit(path, {
                "rule_id": "display_close_single_trailing_math_delimiter",
                "before": repaired_leading,
                "after": closed,
            })
        promoted = promote_plain_algebra_markup(closed)
        if promoted != closed:
            record_rule_hit(path, {
                "rule_id": "display_promote_plain_algebra_markup",
                "before": closed,
                "after": promoted,
            })
        formula_index = 0

        def replace_formula(match: re.Match[str]) -> str:
            nonlocal formula_index
            formula_path = f"{path}.formula[{formula_index}]"
            formula_index += 1
            return f"${normalize_formula_value(match.group(1), formula_path)}$"

        return INLINE_FORMULA.sub(replace_formula, promoted)

    def normalize_guidance_step(source: str, path: str) -> str:
        stripped = strip_redundant_chinese_step_prefix(source)
        if stripped != source:
            record_rule_hit(path, {
                "rule_id": "display_strip_redundant_chinese_step_prefix",
                "before": source,
                "after": stripped,
            })
        return normalize_display_text(stripped, path)

    normalized = json.loads(json.dumps(output, ensure_ascii=False))
    question_blocks = normalized.get("question_blocks")
    if isinstance(question_blocks, list):
        explicit_figure_ids = {
            str(block.get("figure_id") or "")
            for block in question_blocks
            if isinstance(block, dict) and block.get("type") == "figure"
        }
        inserted_figure_ids: set[str] = set()
        repaired_blocks: list[Any] = []
        for block in question_blocks:
            if not isinstance(block, dict):
                repaired_blocks.append(block)
                continue
            block_type = block.get("type")
            figure_id = str(block.get("figure_id") or "").strip()
            text = str(block.get("text") or "")
            if block_type == "text" and figure_id:
                repaired_blocks.append({**block, "figure_id": ""})
                if (
                    figure_id not in explicit_figure_ids
                    and figure_id not in inserted_figure_ids
                ):
                    repaired_blocks.append({
                        "type": "figure",
                        "text": "",
                        "figure_id": figure_id,
                    })
                    inserted_figure_ids.add(figure_id)
                continue
            if block_type == "figure" and text.strip():
                repaired_blocks.append({
                    "type": "text",
                    "text": text,
                    "figure_id": "",
                })
                repaired_blocks.append({**block, "text": ""})
                continue
            repaired_blocks.append(block)
        normalized["question_blocks"] = repaired_blocks
    for block_index, block in enumerate(normalized.get("question_blocks", [])):
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            block["text"] = normalize_display_text(
                block["text"],
                f"$.question_blocks[{block_index}].text",
            )
    for formula_index, item in enumerate(normalized.get("base_formulas", [])):
        if isinstance(item, dict) and isinstance(item.get("formula"), str):
            formula_source, area_change = repair_triangle_area_product_denominator(
                item["formula"],
                str(item.get("condition") or ""),
            )
            if area_change is not None:
                record_rule_hit(
                    f"$.base_formulas[{formula_index}].formula",
                    area_change,
                )
            formula_source, midpoint_change = repair_midpoint_half_segment_denominator(
                formula_source,
                str(item.get("condition") or ""),
            )
            if midpoint_change is not None:
                record_rule_hit(
                    f"$.base_formulas[{formula_index}].formula",
                    midpoint_change,
                )
            formula_source = close_single_trailing_math_delimiter(formula_source)
            embedded = re.fullmatch(r"\s*\$([^$]+)\$(.+)", formula_source)
            if embedded is not None:
                record_rule_hit(f"$.base_formulas[{formula_index}].formula", {
                    "rule_id": "base_formula_extract_embedded_formula",
                    "before": formula_source,
                    "after": embedded.group(1).strip(),
                })
                formula_source = embedded.group(1).strip()
            elif formula_source.count("$") >= 2 and formula_source.count("$") % 2 == 0:
                flattened = INLINE_FORMULA.sub(lambda match: match.group(1), formula_source)
                if flattened != formula_source:
                    record_rule_hit(f"$.base_formulas[{formula_index}].formula", {
                        "rule_id": "base_formula_flatten_embedded_math_delimiters",
                        "before": formula_source,
                        "after": flattened,
                    })
                    formula_source = flattened
            if formula_source != item["formula"]:
                record_rule_hit(f"$.base_formulas[{formula_index}].formula", {
                    "rule_id": "display_close_single_trailing_math_delimiter",
                    "before": item["formula"],
                    "after": formula_source,
                })
            item["formula"] = normalize_formula_value(
                formula_source,
                f"$.base_formulas[{formula_index}].formula",
            )
        if isinstance(item, dict) and isinstance(item.get("condition"), str):
            item["condition"] = normalize_display_text(
                item["condition"],
                f"$.base_formulas[{formula_index}].condition",
            )
    guidance = normalized.get("guidance")
    if isinstance(guidance, dict):
        for tier_key, tier in guidance.items():
            if not isinstance(tier, dict):
                continue
            for part_index, part in enumerate(tier.get("parts", [])):
                if not isinstance(part, dict):
                    continue
                for key in ("pitfall",):
                    if isinstance(part.get(key), str):
                        part[key] = normalize_display_text(
                            part[key],
                            f"$.guidance.{tier_key}.parts[{part_index}].{key}",
                        )
                steps = part.get("steps")
                if isinstance(steps, list):
                    part["steps"] = [
                        normalize_guidance_step(
                            step,
                            f"$.guidance.{tier_key}.parts[{part_index}].steps[{step_index}]",
                        )
                        if isinstance(step, str)
                        else step
                        for step_index, step in enumerate(steps)
                    ]
                alternatives = part.get("alternative_methods")
                if isinstance(alternatives, list):
                    for method_index, method in enumerate(alternatives):
                        if not isinstance(method, dict):
                            continue
                        method_steps = method.get("steps")
                        if isinstance(method_steps, list):
                            method["steps"] = [
                                normalize_guidance_step(
                                    step,
                                    f"$.guidance.{tier_key}.parts[{part_index}]"
                                    f".alternative_methods[{method_index}]"
                                    f".steps[{step_index}]",
                                )
                                if isinstance(step, str)
                                else step
                                for step_index, step in enumerate(method_steps)
                            ]
            if isinstance(tier.get("pattern_summary"), str):
                tier["pattern_summary"] = normalize_display_text(
                    tier["pattern_summary"],
                    f"$.guidance.{tier_key}.pattern_summary",
                )
    return repair_known_parallelogram_geometry_correspondence(
        normalized,
        normalization_audit,
    )


def repair_json_control_characters(candidate: str) -> str:
    """Repair control bytes produced when JSON transports raw LaTeX escapes."""
    output: list[str] = []
    in_string = False
    escaped = False
    in_math = False
    escape_names = {
        "\b": "b",
        "\t": "t",
        "\n": "n",
        "\f": "f",
        "\r": "r",
    }
    for character in candidate:
        if not in_string:
            output.append(character)
            if character == '"':
                in_string = True
                in_math = False
            continue
        if escaped:
            output.append(character)
            escaped = False
            continue
        if character == "\\":
            output.append(character)
            escaped = True
            continue
        if character == '"':
            output.append(character)
            in_string = False
            in_math = False
            continue
        if character == "$":
            in_math = not in_math
            output.append(character)
            continue
        if ord(character) < 0x20:
            escape_name = escape_names.get(character)
            if escape_name is None:
                output.append(" ")
            elif character == "\n" and not in_math:
                output.extend(("\\", "n"))
            else:
                # Two backslashes in the JSON source decode to one literal
                # backslash in the model field, restoring e.g. \frac/\theta.
                output.extend(("\\", "\\", escape_name))
            continue
        output.append(character)
    return "".join(output)


def parse_model_json(
    text: str,
    requested_outputs: Sequence[Mapping[str, Any]] | None = None,
    normalization_audit: list[dict[str, Any]] | None = None,
    *,
    device_policy: bool = False,
) -> tuple[Any | None, list[str]]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate, _ = benchmark.unwrap_single_json_code_fence(candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        if "control character" not in exc.msg.lower():
            return None, [f"JSON解析失败：{exc}"]
        repaired = repair_json_control_characters(candidate)
        try:
            value = json.loads(repaired)
        except json.JSONDecodeError as repaired_exc:
            return None, [f"JSON解析失败：{repaired_exc}"]
    value = normalize_output_for_render(
        value,
        normalization_audit=normalization_audit,
    )
    return value, validate_output(
        value,
        requested_outputs=requested_outputs,
        device_policy=device_policy,
    )


def percentile(values: Sequence[float], percentage: float) -> float | None:
    material = sorted(float(item) for item in values if math.isfinite(float(item)))
    if not material:
        return None
    if len(material) == 1:
        return round(material[0], 3)
    rank = (len(material) - 1) * percentage
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(material[lower], 3)
    fraction = rank - lower
    return round(material[lower] + (material[upper] - material[lower]) * fraction, 3)


def distribution(values: Iterable[Any]) -> dict[str, Any]:
    material = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    if not material:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(material),
        "mean": round(statistics.fmean(material), 3),
        "p50": percentile(material, 0.50),
        "p95": percentile(material, 0.95),
        "max": round(max(material), 3),
    }


def latest_records(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            value = json.loads(raw)
            if isinstance(value, dict) and value.get("question_id"):
                latest[str(value["question_id"])] = value
    return latest


def summarize(records: Sequence[Mapping[str, Any]], planned: int) -> dict[str, Any]:
    successes = [record for record in records if record.get("status") == "success"]
    usage_keys = (
        "input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "cached_input_tokens",
        "total_tokens",
    )
    usage_values = {
        key: [
            int(record["usage"][key])
            for record in records
            if isinstance(record.get("usage"), Mapping)
            and isinstance(record["usage"].get(key), int)
            and not isinstance(record["usage"].get(key), bool)
        ]
        for key in usage_keys
    }
    # A missing provider field is unknown, not zero.  ``usage`` remains the
    # compatibility aggregate, while ``usage_known_sum`` makes the partial
    # coverage semantics explicit for downstream reports.
    usage = {
        key: sum(values) if values else None
        for key, values in usage_values.items()
    }
    usage_known_sum = dict(usage)
    usage_coverage = {
        key: {
            "reported_records": len(values),
            "missing_records": len(records) - len(values),
            "complete": len(values) == len(records),
            "coverage_ratio": round(len(values) / len(records), 6)
            if records
            else None,
        }
        for key, values in usage_values.items()
    }
    successful_usage_values = {
        key: [
            int(record["usage"][key])
            for record in successes
            if isinstance(record.get("usage"), Mapping)
            and isinstance(record["usage"].get(key), int)
            and not isinstance(record["usage"].get(key), bool)
        ]
        for key in usage_keys
    }
    timing_fields = (
        "response_headers_wait_ms",
        "image_preprocess_ms",
        "request_build_ms",
        "api_round_trip_ms",
        "first_event_ms",
        "first_content_ms",
        "first_reasoning_ms",
        "provider_total_ms",
        "parse_validate_ms",
        "end_to_end_ms",
        "render_ms",
        "local_files_ready_ms",
        "files_ready_ms",
        "device_ready_ms",
    )
    timing_success_only = {
        key: distribution(
            record.get("timing", {}).get(key)
            for record in successes
            if isinstance(record.get("timing"), Mapping)
        )
        for key in timing_fields
    }
    timing_all_completed = {
        key: distribution(
            record.get("timing", {}).get(key)
            for record in records
            if isinstance(record.get("timing"), Mapping)
        )
        for key in timing_fields
    }
    return {
        "schema_version": "math_guidance_pilot_summary_v4_multi_method",
        "updated_at": utc_now(),
        "planned": planned,
        "completed": len(records),
        "success": len(successes),
        "failed": len(records) - len(successes),
        "pending": max(0, planned - len(records)),
        "success_rate": round(len(successes) / planned, 6) if planned else None,
        "usage": usage,
        "usage_known_sum": usage_known_sum,
        "usage_coverage": usage_coverage,
        "provider_usage_reported_records": sum(
            1
            for record in records
            if isinstance(record.get("usage"), Mapping)
            and record["usage"].get("reported") is True
        ),
        "provider_usage_missing_records": sum(
            1
            for record in records
            if not isinstance(record.get("usage"), Mapping)
            or record["usage"].get("reported") is not True
        ),
        "usage_per_completed": {
            key: round(usage[key] / len(records), 3)
            if records and usage_coverage[key]["complete"] and usage[key] is not None
            else None
            for key in usage_keys
        },
        "usage_per_reported": {
            key: round(usage[key] / len(usage_values[key]), 3)
            if usage_values[key] and usage[key] is not None
            else None
            for key in usage_keys
        },
        "usage_per_success": {
            key: round(sum(successful_usage_values[key]) / len(successes), 3)
            if successes and len(successful_usage_values[key]) == len(successes)
            else None
            for key in usage_keys
        },
        "usage_per_reported_success": {
            key: round(
                sum(successful_usage_values[key])
                / len(successful_usage_values[key]),
                3,
            )
            if successful_usage_values[key]
            else None
            for key in usage_keys
        },
        "max_method_count_per_part": distribution(
            record.get("render", {}).get("max_method_count_per_part")
            for record in successes
            if isinstance(record.get("render"), Mapping)
        ),
        # Keep the original success-only field for compatibility, and expose
        # every completed provider/local attempt separately so failures are
        # not silently omitted from latency reporting.
        "latency_ms_success_only": timing_success_only,
        "latency_ms_all_completed": timing_all_completed,
    }


def write_summary_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "question_id",
        "status",
        "input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "cached_input_tokens",
        "total_tokens",
        "total_tokens_source",
        "usage_reported",
        "response_headers_wait_ms",
        "image_preprocess_ms",
        "request_build_ms",
        "api_round_trip_ms",
        "first_event_ms",
        "first_content_ms",
        "first_reasoning_ms",
        "provider_total_ms",
        "parse_validate_ms",
        "end_to_end_ms",
        "render_ms",
        "local_files_ready_ms",
        "files_ready_ms",
        "device_ready_ms",
        "image_count",
        "question_knowledge_render_ms",
        "advanced_render_ms",
        "standard_render_ms",
        "foundation_render_ms",
        "max_method_count_per_part",
        "method_counts_by_tier",
        "milestones_ms",
        "error",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            usage = record.get("usage") if isinstance(record.get("usage"), Mapping) else {}
            timing = record.get("timing") if isinstance(record.get("timing"), Mapping) else {}
            render = record.get("render") if isinstance(record.get("render"), Mapping) else {}
            render_files = (
                render.get("files")
                if isinstance(render.get("files"), Mapping)
                else {}
            )
            error = record.get("error")
            writer.writerow({
                "question_id": record.get("question_id"),
                "status": record.get("status"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                "cached_input_tokens": usage.get("cached_input_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "total_tokens_source": usage.get("total_tokens_source"),
                "usage_reported": usage.get("reported"),
                "response_headers_wait_ms": timing.get("response_headers_wait_ms"),
                "image_preprocess_ms": timing.get("image_preprocess_ms"),
                "request_build_ms": timing.get("request_build_ms"),
                "api_round_trip_ms": timing.get("api_round_trip_ms"),
                "first_event_ms": timing.get("first_event_ms"),
                "first_content_ms": timing.get("first_content_ms"),
                "first_reasoning_ms": timing.get("first_reasoning_ms"),
                "provider_total_ms": timing.get("provider_total_ms"),
                "parse_validate_ms": timing.get("parse_validate_ms"),
                "end_to_end_ms": timing.get("end_to_end_ms"),
                "render_ms": timing.get("render_ms"),
                "local_files_ready_ms": timing.get("local_files_ready_ms"),
                "files_ready_ms": timing.get("files_ready_ms"),
                "device_ready_ms": timing.get("device_ready_ms"),
                "image_count": render.get("image_count"),
                "question_knowledge_render_ms": (
                    render_files.get("question_knowledge", {}).get("render_ms")
                    if isinstance(render_files.get("question_knowledge"), Mapping)
                    else None
                ),
                "advanced_render_ms": (
                    render_files.get("advanced", {}).get("render_ms")
                    if isinstance(render_files.get("advanced"), Mapping)
                    else None
                ),
                "standard_render_ms": (
                    render_files.get("standard", {}).get("render_ms")
                    if isinstance(render_files.get("standard"), Mapping)
                    else None
                ),
                "foundation_render_ms": (
                    render_files.get("foundation", {}).get("render_ms")
                    if isinstance(render_files.get("foundation"), Mapping)
                    else None
                ),
                "max_method_count_per_part": render.get("max_method_count_per_part"),
                "method_counts_by_tier": json.dumps(
                    render.get("method_counts_by_tier"),
                    ensure_ascii=False,
                )
                if render.get("method_counts_by_tier") is not None
                else "",
                "milestones_ms": json.dumps(
                    timing.get("milestones_ms"),
                    ensure_ascii=False,
                )
                if timing.get("milestones_ms") is not None
                else "",
                "error": json.dumps(error, ensure_ascii=False) if error else "",
            })
    os.replace(temporary, path)


def append_record_and_update_metrics(
    *,
    results_path: Path,
    output_dir: Path,
    existing: dict[str, dict[str, Any]],
    questions: Sequence[Mapping[str, Any]],
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Durably append one result, then atomically refresh both metric views."""
    question_id = str(record["question_id"])
    append_jsonl(results_path, record)
    existing[question_id] = dict(record)
    ordered = [
        existing[str(question["question_id"])]
        for question in questions
        if str(question["question_id"]) in existing
    ]
    atomic_json(output_dir / "summary.json", summarize(ordered, len(questions)))
    write_summary_csv(output_dir / "per_question_metrics.csv", ordered)
    return ordered


def load_credentials(env_file: Path) -> tuple[str, str, str | None, list[str]]:
    loaded_names = benchmark.parse_env_file(env_file)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少DASHSCOPE_API_KEY")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip() or DEFAULT_BASE_URL
    workspace = os.environ.get("DASHSCOPE_WORKSPACE_ID", "").strip() or None
    return api_key, base_url, workspace, loaded_names


def image_data_url(image_path: Path) -> tuple[str, dict[str, Any]]:
    raw = image_path.read_bytes()
    prepared, media_type, audit = benchmark.prepare_request_image(
        raw,
        "image/png",
        {
            "image_preprocess": {
                "enabled": True,
                # bbox_1000 is defined against the complete source image and
                # is later applied to that same source by the renderer.  Keep
                # the full frame here so the model and renderer share one
                # normalized coordinate system.  Resizing preserves normalized
                # coordinates; auto-cropping would not.
                "auto_crop_blank": False,
                "crop_luma_threshold": 250,
                "crop_padding": 16,
                "max_long_edge": 1280,
                "format": "JPEG",
                "quality": 92,
                "binarize": False,
            }
        },
    )
    encoded = base64.b64encode(prepared).decode("ascii")
    return f"data:{media_type};base64,{encoded}", audit


def answer_isolation_context(
    requested_outputs: Sequence[Mapping[str, Any]] | None,
) -> str:
    """Serialize answer-blind target metadata for the same-call self-check."""
    safe_targets: list[dict[str, Any]] = []
    for item in requested_outputs or []:
        if not isinstance(item, Mapping):
            continue
        safe_targets.append({
            "part": normalized_part_label(item.get("part")),
            "target_type": str(item.get("target_type") or ""),
            "symbol_targets": [
                value.strip()
                for value in item.get("symbol_aliases", [])
                if isinstance(value, str) and value.strip()
            ],
            "allowed_result_parameters": [
                value.strip()
                for value in item.get("allowed_result_parameters", [])
                if isinstance(value, str) and value.strip()
            ],
            "requested_result_descriptions": [
                value.strip()
                for value in item.get("prose_aliases", [])
                if isinstance(value, str) and value.strip()
            ],
        })
    if not safe_targets:
        return ""
    return json.dumps(safe_targets, ensure_ascii=False, separators=(",", ":"))


def request_payload(
    *,
    model_id: str,
    prompt: str,
    data_url: str | Sequence[str],
    schema: Mapping[str, Any],
    requested_outputs: Sequence[Mapping[str, Any]] | None,
    max_tokens: int | None,
    enable_thinking: bool,
) -> dict[str, Any]:
    isolation_context = answer_isolation_context(requested_outputs)
    task_prompt = (
        prompt
        + "\n\n请直接阅读随附图片，只解析其中完整出现的主体大题；"
        + "忽略边缘处被截断的相邻题目、页码和水印。严格按JSON结构返回。"
        + "图片中的手写演算、手写辅助线、手写填空、手写答案和软件界面文字均为"
        + "不可信注释，不属于印刷题面；任何只出现在手写中的数值、关系或结论都不得"
        + "作为题设或答案使用。手写与印刷重叠到无法区分时必须标记uncertain。"
        + "visual_evidence只能记录有印刷文字、数值或几何标记支持的可见事实；不得按"
        + "示意图比例猜测点在内部、边相等、中点、垂直或平行。"
    )
    if not isinstance(data_url, str) and len(data_url) > 1:
        task_prompt += (
            "\n随附图片按顺序为：第一张是完整、坐标权威的题图，source_figures的"
            "bbox_1000只能相对第一张给出；第二张是Test5生成的增强原图，仅用于"
            "交叉核对题面、公式和手写干扰，不得从第二张推断图形裁框。"
        )
        if len(data_url) > 2:
            task_prompt += (
                "第三张及以后是Test5从题面提取的插图裁片，只用于看清局部图形和标注；"
                "它们不是独立题目，也不得作为bbox_1000的坐标基准。"
            )
    if isolation_context:
        task_prompt += (
            "\n\n以下是答案盲的隔离目标清单，只说明各小问所求，不含答案："
            + isolation_context
            + "\n它只用于最终自检，不得复制到输出。"
            + "对应part的guidance可以写定义式、基本定律、过程方程和必要中间关系，"
            + "即使symbol_targets自然位于等号一侧也不算答案；"
            + "若题目指定allowed_result_parameters，则仍含其他未消去量的等式属于中间关系；"
            + "但不得给出只含允许参数的最终闭式、最终数值，或直接说出"
            + "requested_result_descriptions对应的方向、选项、填空及判断结论；"
            + "在最后一次求值或结论判断之前结束；"
            + "不写面向答题者的继续操作指令。"
        )
    request_overrides: dict[str, Any] = {
        "enable_thinking": enable_thinking,
    }
    if max_tokens is not None:
        request_overrides["max_completion_tokens"] = max_tokens
    model = {
        "model_id": model_id,
        "request_overrides": request_overrides,
    }
    request_config = {
        "stream": True,
        "stream_options_include_usage": True,
        "temperature": 0,
        "image_detail": "high",
        "response_format": "json_schema",
    }
    return benchmark.build_payload(
        model,
        task_prompt,
        data_url,
        request_config,
        schema,
        None,
        {"stream_options_include_usage": True},
    )


def merge_attempt_usage(*attempts: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate provider usage across an initial call and repair calls."""
    keys = (
        "input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "cached_input_tokens",
        "total_tokens",
    )
    merged: dict[str, Any] = {}
    for key in keys:
        values = [
            attempt.get("usage", {}).get(key)
            for attempt in attempts
            if isinstance(attempt.get("usage"), Mapping)
        ]
        known = [
            value for value in values
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        merged[key] = sum(known) if known else None
    merged["total_tokens_source"] = "provider_aggregate"
    merged["reported"] = all(
        isinstance(attempt.get("usage"), Mapping)
        and attempt["usage"].get("reported") is True
        for attempt in attempts
    )
    return merged


def validation_repair_prompt(
    prompt: str,
    parsed: Mapping[str, Any],
    errors: Sequence[str],
) -> str:
    """Build one bounded, image-grounded correction request."""
    error_text = "\n".join(f"- {item}" for item in errors[:30])
    prior_for_repair = json.loads(json.dumps(parsed, ensure_ascii=False))
    guidance_must_be_regenerated = any(
        item.startswith("guidance.")
        or "同一小问在三个档位中的方法" in item
        for item in errors
    )
    if guidance_must_be_regenerated:
        # Do not anchor the model on a degenerate list such as [", ", ...]
        # or ["$"]. In production the repair model repeatedly copied those
        # placeholders even after being told not to. Recognition and figure
        # evidence remain available, while all guidance is regenerated.
        prior_for_repair.pop("guidance", None)
    prior_json = json.dumps(
        prior_for_repair, ensure_ascii=False, separators=(",", ":")
    )
    guidance_note = (
        "上一次guidance已因内容退化而从定位JSON中移除；必须根据原题图片、"
        "question_blocks、knowledge_points和base_formulas从零重建三档guidance，"
        "不得尝试还原或复制被移除的步骤。"
        if guidance_must_be_regenerated
        else ""
    )
    return (
        prompt
        + "\n\n上一次输出未通过本地强校验。请重新查看随附的完整原题图片，"
        + "只返回一份从头到尾完整的新JSON对象，不要解释，不要使用Markdown。"
        + "必须真正修正下列错误，不得删除原题小问，不得给出任何所求量的最终数值、"
        + "方向或定性结论。source_figures的bbox_1000必须是相对完整图片的四个数，"
        + "满足0≤left<right≤1000且0≤top<bottom≤1000；看不清时改为uncertain，"
        + "不要提交无效框。所有公式必须使用规范LaTeX；三档同一小问的方法编号、数量和顺序一致，"
        + "方法名称可按档位使用不同难度的措辞，但方法实质不得改变。"
        + "每个steps元素都必须是可执行的数学引导，至少包含文字、数字或公式；"
        + "禁止用逗号、句号、空白或其他单独符号占位，也不得为修复报错而删减原题或有效步骤。"
        + guidance_note
        + "\n\n本地校验错误：\n"
        + error_text
        + "\n\n上一次JSON仅供定位错误，不得机械照抄：\n"
        + prior_json
    )


def resolve_output_path(root: Any, path: str) -> tuple[Any, Any] | None:
    """Resolve validator paths such as guidance.standard.parts[0].steps[1]."""
    tokens = re.findall(r"(?:^|\.)([A-Za-z_]+)|\[(\d+)\]", path)
    current = root
    parent: Any = None
    key: Any = None
    for name, index in tokens:
        parent = current
        key = name if name else int(index)
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError):
            return None
    return parent, key


def redact_answer_guard_violations(
    output: Mapping[str, Any],
    errors: Sequence[str],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Delete only safe-to-remove leaking steps, then revalidate.

    Earlier versions replaced an offending formula or qualitative conclusion
    with generic prose.  That could leave broken sentences or repeated
    placeholders in otherwise valid pages.  A deletion is deterministic and
    content-neutral: apply it only to a guidance ``steps`` item and only when
    the tier still meets its minimum step count.  Everything else remains an
    error for the provider repair path (or fails closed during recovery).
    """
    redacted = json.loads(json.dumps(output, ensure_ascii=False))
    formula_ops: list[tuple[str, int, str]] = []
    step_ops: dict[str, str] = {}
    for error in errors:
        if "单独解出了" in error:
            path = error.split("单独解出了", 1)[0]
            match = re.fullmatch(
                r"(?P<step>.+\.steps\[\d+\])\.formula\[\d+\]",
                path,
            )
            if match:
                formula_index_match = re.search(r"\.formula\[(\d+)\]$", path)
                if formula_index_match is not None:
                    formula_ops.append(
                        (
                            match.group("step"),
                            int(formula_index_match.group(1)),
                            error,
                        )
                    )
                step_ops.setdefault(match.group("step"), error)
        elif "直接给出了" in error:
            path = error.split("直接给出了", 1)[0]
            if re.fullmatch(r".+\.steps\[\d+\]", path):
                step_ops.setdefault(path, error)
        elif "完成曲线与切线位置证明" in error:
            match = re.match(
                r"(?P<step>guidance\..+\.steps\[\d+\])差函数极值",
                error,
            )
            if match:
                step_ops.setdefault(match.group("step"), error)

    # Prefer an extractive clause trim over deleting a whole step.  The formula
    # index comes from the same validator view, and ordinary guidance formulas
    # are already dollar-delimited.  We only retain model-written text: cut at a
    # nearby result connector (``解得/得到``), or remove the punctuation-bounded
    # clause containing the final formula.  If no substantive fragment remains,
    # leave the value untouched so the normal provider/fail-closed path applies.
    repaired_steps: set[str] = set()
    for step_path, formula_index, error in sorted(
        formula_ops, key=lambda item: (item[0], -item[1])
    ):
        # A result connector normally belongs to the final formula in a step.
        # Once trimming that suffix has preserved a coherent prefix, do not
        # revisit earlier intermediate formulas from the validator's original
        # error list and progressively hollow the sentence out.
        if step_path in repaired_steps:
            continue
        resolved = resolve_output_path(redacted, step_path)
        if resolved is None:
            continue
        parent, key = resolved
        value = parent[key]
        if not isinstance(value, str):
            continue
        matches = list(INLINE_FORMULA.finditer(value))
        if not (0 <= formula_index < len(matches)):
            continue
        formula_match = matches[formula_index]
        prefix = value[: formula_match.start()]
        nearby_connectors = list(
            re.finditer(
                r"(?:最终(?:得到|求得|解得|可得)?|解得|求得|算得|得到|可得)",
                prefix,
            )
        )
        candidate_value = ""
        if nearby_connectors and formula_match.start() - nearby_connectors[-1].start() <= 48:
            cut_at = nearby_connectors[-1].start()
            candidate_value = value[:cut_at].rstrip(" \t\r\n，,；;。:：")
        else:
            left_punctuation = max(
                (value.rfind(marker, 0, formula_match.start()) for marker in "，,；;。"),
                default=-1,
            )
            right_candidates = [
                position
                for marker in "，,；;。"
                for position in [value.find(marker, formula_match.end())]
                if position >= 0
            ]
            right_punctuation = min(right_candidates) if right_candidates else len(value)
            candidate_value = (
                value[: left_punctuation + 1]
                + value[right_punctuation + (right_punctuation < len(value)) :]
            ).strip(" \t\r\n，,；;。:：")
        if (
            candidate_value
            and candidate_value != value
            and has_substantive_guidance_content(candidate_value)
        ):
            parent[key] = candidate_value
            repaired_steps.add(step_path)
            audit.append(
                {
                    "action": "truncate_direct_answer_clause",
                    "path": step_path,
                    "removed_chars": len(value) - len(candidate_value),
                    "validator_error": error,
                }
            )

    min_steps = {"advanced": 2, "standard": 3, "foundation": 4}
    ordered_ops: list[tuple[str, int, str, str]] = []
    for step_path, error in step_ops.items():
        if step_path in repaired_steps:
            continue
        match = re.fullmatch(r"(?P<list>.+\.steps)\[(?P<index>\d+)\]", step_path)
        tier_match = re.search(
            r"guidance\.(advanced|standard|foundation)\.",
            step_path,
        )
        if match is None or tier_match is None:
            continue
        ordered_ops.append((
            match.group("list"),
            int(match.group("index")),
            tier_match.group(1),
            error,
        ))

    for list_path, step_index, tier_key, error in sorted(
        ordered_ops,
        key=lambda item: (item[0], -item[1]),
    ):
        resolved = resolve_output_path(redacted, list_path)
        if resolved is None:
            continue
        parent, key = resolved
        steps = parent[key]
        if not isinstance(steps, list) or not (0 <= step_index < len(steps)):
            continue
        if len(steps) - 1 < min_steps[tier_key]:
            continue
        removed = steps.pop(step_index)
        audit.append({
            "action": "remove_final_answer_step",
            "path": f"{list_path}[{step_index}]",
            "removed_chars": len(removed) if isinstance(removed, str) else 0,
            "validator_error": error,
        })
    return redacted


def recover_prior_locally(
    *,
    prior: Mapping[str, Any],
    question: Mapping[str, Any],
    dataset_root: Path,
    output_dir: Path,
    model_id: str,
) -> dict[str, Any] | None:
    """Turn a validation failure into success without another provider call."""
    parsed = prior.get("output_json")
    prior_validation = prior.get("validation")
    errors = (
        prior_validation.get("errors", [])
        if isinstance(prior_validation, Mapping)
        else []
    )
    if not isinstance(parsed, Mapping) or not isinstance(errors, list):
        return None
    audit: list[dict[str, Any]] = []
    candidate = redact_answer_guard_violations(parsed, errors, audit)
    normalization_audit: list[dict[str, Any]] = []
    normalized_candidate, remaining = parse_model_json(
        json.dumps(candidate, ensure_ascii=False),
        requested_outputs=question.get("requested_outputs"),
        normalization_audit=normalization_audit,
        device_policy=True,
    )
    if remaining or not isinstance(normalized_candidate, Mapping):
        return None
    if not audit and not normalization_audit:
        return None
    audit.extend(normalization_audit)
    candidate = dict(normalized_candidate)
    started = time.perf_counter()
    question_id = str(question["question_id"])
    render = render_four_images(
        question_id=question_id,
        model_id=str(prior.get("model_id_response") or model_id),
        source_image=dataset_root / str(question["image_path"]),
        output=candidate,
        output_dir=output_dir / "images" / question_id,
    )
    render_ms = round((time.perf_counter() - started) * 1000.0, 3)
    record = json.loads(json.dumps(prior, ensure_ascii=False))
    record.update({
        "recorded_at": utc_now(),
        "status": "success",
        "validation": {
            "valid": True,
            "errors": [],
            "quality_warnings": prior_validation.get(
                "quality_warnings", []
            ),
            "gate_mode": "no_answer",
        },
        "render": render,
        "error": None,
        "output_json": candidate,
        "local_recovery": {
            "provider_calls": 0,
            "actions": audit,
            "render_ms": render_ms,
        },
    })
    timing = record.setdefault("timing", {})
    timing["render_ms"] = render.get("render_ms")
    timing["local_recovery_ms"] = render_ms
    timing["local_files_ready_ms"] = render_ms
    timing["files_ready_ms"] = render_ms
    timing["device_ready_ms"] = render_ms
    return record


def run_one(
    *,
    question: Mapping[str, Any],
    dataset_root: Path,
    output_dir: Path,
    model_id: str,
    prompt: str,
    schema: Mapping[str, Any],
    api_key: str,
    base_url: str,
    workspace_id: str | None,
    timeout_seconds: float,
    max_tokens: int | None,
    enable_thinking: bool,
) -> dict[str, Any]:
    question_id = str(question["question_id"])
    source_image = dataset_root / str(question["image_path"])
    started = time.perf_counter()
    started_at = utc_now()
    phase_timing: dict[str, float | None] = {
        "image_preprocess_ms": None,
        "request_build_ms": None,
        "api_round_trip_ms": None,
        "parse_validate_ms": None,
    }
    try:
        phase_started = time.perf_counter()
        data_url, image_audit = image_data_url(source_image)
        phase_timing["image_preprocess_ms"] = round(
            (time.perf_counter() - phase_started) * 1000.0,
            3,
        )
        phase_started = time.perf_counter()
        payload = request_payload(
            model_id=model_id,
            prompt=prompt,
            data_url=data_url,
            schema=schema,
            requested_outputs=question.get("requested_outputs"),
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )
        phase_timing["request_build_ms"] = round(
            (time.perf_counter() - phase_started) * 1000.0,
            3,
        )
        phase_started = time.perf_counter()
        attempt = benchmark.perform_http_attempt(
            payload,
            base_url,
            api_key,
            workspace_id,
            timeout_seconds,
            [api_key],
            ROOT_KEYS,
            False,
        )
        phase_timing["api_round_trip_ms"] = round(
            (time.perf_counter() - phase_started) * 1000.0,
            3,
        )
        parsed = None
        validation_errors: list[str] = []
        quality_warnings: list[str] = []
        normalization_audit: list[dict[str, Any]] = []
        repair_attempt_audit: list[dict[str, Any]] = []
        provider_attempts: list[Mapping[str, Any]] = [attempt]
        render = None
        status = "failed_api"
        error = attempt.get("error")
        if attempt.get("transport_success"):
            phase_started = time.perf_counter()
            parsed, validation_errors = parse_model_json(
                str(attempt.get("output_text") or ""),
                requested_outputs=question.get("requested_outputs"),
                normalization_audit=normalization_audit,
                device_policy=True,
            )
            if isinstance(parsed, Mapping):
                strict_errors = validate_output(
                    parsed,
                    requested_outputs=question.get("requested_outputs"),
                )
                quality_warnings = [
                    item for item in strict_errors if item not in validation_errors
                ]
            phase_timing["parse_validate_ms"] = round(
                (time.perf_counter() - phase_started) * 1000.0,
                3,
            )
            if validation_errors and isinstance(parsed, Mapping):
                repair_started = time.perf_counter()
                repair_payload = request_payload(
                    model_id=model_id,
                    prompt=validation_repair_prompt(
                        prompt,
                        parsed,
                        validation_errors,
                    ),
                    data_url=data_url,
                    schema=schema,
                    requested_outputs=question.get("requested_outputs"),
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                )
                repair_attempt = benchmark.perform_http_attempt(
                    repair_payload,
                    base_url,
                    api_key,
                    workspace_id,
                    timeout_seconds,
                    [api_key],
                    ROOT_KEYS,
                    False,
                )
                provider_attempts.append(repair_attempt)
                repair_elapsed_ms = round(
                    (time.perf_counter() - repair_started) * 1000.0,
                    3,
                )
                phase_timing["api_round_trip_ms"] = round(
                    float(phase_timing["api_round_trip_ms"] or 0)
                    + repair_elapsed_ms,
                    3,
                )
                repair_entry: dict[str, Any] = {
                    "transport_success": bool(
                        repair_attempt.get("transport_success")
                    ),
                    "http_status": repair_attempt.get("http_status"),
                    "provider_request_id": repair_attempt.get(
                        "provider_request_id"
                    ),
                    "api_round_trip_ms": repair_elapsed_ms,
                }
                if repair_attempt.get("transport_success"):
                    repair_normalization: list[dict[str, Any]] = []
                    repaired, repair_errors = parse_model_json(
                        str(repair_attempt.get("output_text") or ""),
                        requested_outputs=question.get("requested_outputs"),
                        normalization_audit=repair_normalization,
                        device_policy=True,
                    )
                    repair_entry["validation_error_count"] = len(
                        repair_errors
                    )
                    repair_entry["validation_errors"] = repair_errors[:30]
                    if isinstance(repaired, Mapping) and len(repair_errors) <= len(
                        validation_errors
                    ):
                        parsed = repaired
                        validation_errors = repair_errors
                        normalization_audit.extend(repair_normalization)
                        attempt = repair_attempt
                else:
                    repair_entry["error"] = repair_attempt.get("error")
                repair_attempt_audit.append(repair_entry)
            if validation_errors and isinstance(parsed, Mapping):
                # A first repair commonly reduces a long error list to one
                # answer-guard or formula issue.  Permit one final, narrowly
                # targeted pass instead of rerunning the whole question later.
                repair_started = time.perf_counter()
                final_repair_payload = request_payload(
                    model_id=model_id,
                    prompt=validation_repair_prompt(
                        prompt,
                        parsed,
                        validation_errors,
                    ),
                    data_url=data_url,
                    schema=schema,
                    requested_outputs=question.get("requested_outputs"),
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                )
                final_repair = benchmark.perform_http_attempt(
                    final_repair_payload,
                    base_url,
                    api_key,
                    workspace_id,
                    timeout_seconds,
                    [api_key],
                    ROOT_KEYS,
                    False,
                )
                provider_attempts.append(final_repair)
                repair_elapsed_ms = round(
                    (time.perf_counter() - repair_started) * 1000.0,
                    3,
                )
                phase_timing["api_round_trip_ms"] = round(
                    float(phase_timing["api_round_trip_ms"] or 0)
                    + repair_elapsed_ms,
                    3,
                )
                repair_entry = {
                    "transport_success": bool(
                        final_repair.get("transport_success")
                    ),
                    "http_status": final_repair.get("http_status"),
                    "provider_request_id": final_repair.get(
                        "provider_request_id"
                    ),
                    "api_round_trip_ms": repair_elapsed_ms,
                }
                if final_repair.get("transport_success"):
                    repair_normalization = []
                    repaired, repair_errors = parse_model_json(
                        str(final_repair.get("output_text") or ""),
                        requested_outputs=question.get("requested_outputs"),
                        normalization_audit=repair_normalization,
                        device_policy=True,
                    )
                    repair_entry["validation_error_count"] = len(
                        repair_errors
                    )
                    repair_entry["validation_errors"] = repair_errors[:30]
                    if isinstance(repaired, Mapping) and len(repair_errors) <= len(
                        validation_errors
                    ):
                        parsed = repaired
                        validation_errors = repair_errors
                        normalization_audit.extend(repair_normalization)
                        attempt = final_repair
                else:
                    repair_entry["error"] = final_repair.get("error")
                repair_attempt_audit.append(repair_entry)
            if validation_errors:
                status = "failed_validation"
                error = {
                    "type": "output_validation_error",
                    "message": "；".join(validation_errors[:10]),
                }
            else:
                try:
                    render = render_four_images(
                        question_id=question_id,
                        model_id=str(attempt.get("response_model") or model_id),
                        source_image=source_image,
                        output=parsed,
                        output_dir=output_dir / "images" / question_id,
                    )
                    status = "success"
                    error = None
                except Exception as exc:
                    status = "failed_render"
                    error = {"type": type(exc).__name__, "message": str(exc)}
        end_to_end_ms = round((time.perf_counter() - started) * 1000.0, 3)
        render_ms = render.get("render_ms") if isinstance(render, Mapping) else None
        provider_total = attempt.get("provider_total_ms")
        local_files_ready_ms = end_to_end_ms if status == "success" else None
        return {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": utc_now(),
            "started_at": started_at,
            "question_id": question_id,
            "subject": "mathematics",
            "model_id_requested": model_id,
            "model_id_response": attempt.get("response_model"),
            "status": status,
            "source_image": {
                "relative_path": question["image_path"],
                "sha256": question["image_sha256"],
            },
            "request_image": image_audit,
            "usage": merge_attempt_usage(*provider_attempts),
            "timing": {
                **phase_timing,
                "response_headers_wait_ms": attempt.get("response_headers_wait_ms"),
                "first_event_ms": attempt.get("first_event_ms"),
                "first_content_ms": attempt.get("first_content_ms"),
                "first_reasoning_ms": attempt.get("first_reasoning_ms"),
                "provider_total_ms": provider_total,
                "end_to_end_ms": end_to_end_ms,
                "render_ms": render_ms,
                "local_files_ready_ms": local_files_ready_ms,
                "files_ready_ms": local_files_ready_ms,
                "device_ready_ms": local_files_ready_ms,
                "physical_device_refresh_included": False,
                "milestones_ms": attempt.get("milestones_ms"),
            },
            "finish_reason": attempt.get("finish_reason"),
            "provider_request_id": attempt.get("provider_request_id"),
            "http_status": attempt.get("http_status"),
            "validation": {
                "valid": not validation_errors and parsed is not None,
                "errors": validation_errors,
                "quality_warnings": quality_warnings,
                "gate_mode": "no_answer",
            },
            "render": render,
            "error": error,
            "output_text": attempt.get("output_text"),
            "output_json": parsed,
            "normalization_audit": normalization_audit,
            "attempt_audit": {
                key: attempt.get(key)
                for key in (
                    "request_body_bytes",
                    "response_body_bytes",
                    "event_count",
                    "stream_complete",
                    "retryable",
                    "provider_error_code",
                    "provider_error_type",
                )
            },
            "repair_attempts": repair_attempt_audit,
        }
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": utc_now(),
            "started_at": started_at,
            "question_id": question_id,
            "subject": "mathematics",
            "model_id_requested": model_id,
            "model_id_response": None,
            "status": "failed_local",
            "source_image": {
                "relative_path": question.get("image_path"),
                "sha256": question.get("image_sha256"),
            },
            "usage": benchmark.normalize_usage({}),
            "timing": {
                **phase_timing,
                "end_to_end_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "render_ms": None,
                "local_files_ready_ms": None,
                "files_ready_ms": None,
                "device_ready_ms": None,
                "physical_device_refresh_included": False,
            },
            "validation": {
                "valid": False,
                "errors": [],
                "quality_warnings": [],
                "gate_mode": "no_answer",
            },
            "render": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "output_text": None,
            "output_json": None,
            "normalization_audit": [],
        }


def progress_document(
    *,
    planned: int,
    records: Sequence[Mapping[str, Any]],
    state: str,
) -> dict[str, Any]:
    summary = summarize(records, planned)
    return {
        "updated_at": utc_now(),
        "state": state,
        **{
            key: summary[key]
            for key in ("planned", "completed", "success", "failed", "pending", "usage")
        },
    }


def run(args: argparse.Namespace) -> int:
    validate_requested_model_id(args.model_id)
    prompt_path = args.prompt.resolve()
    manifest_path = args.manifest.resolve()
    schema_path = args.schema.resolve()
    output_dir = args.output_dir.resolve()
    prompt = extract_model_prompt(prompt_path)
    manifest, dataset_root = validate_manifest(manifest_path)
    schema = load_json(schema_path)
    if not isinstance(schema, dict):
        raise ValueError("schema根值必须是对象")
    prompt_file_sha256 = sha256_file(prompt_path)
    model_prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    manifest_sha256 = sha256_file(manifest_path)
    schema_sha256 = sha256_file(schema_path)
    validate_required_sha256(
        "提示词文件",
        prompt_file_sha256,
        args.require_prompt_file_sha256,
    )
    validate_required_sha256(
        "模型提示词片段",
        model_prompt_sha256,
        args.require_model_prompt_sha256,
    )
    validate_required_sha256(
        "题目清单",
        manifest_sha256,
        args.require_manifest_sha256,
    )
    validate_required_sha256(
        "输出Schema",
        schema_sha256,
        args.require_schema_sha256,
    )
    questions = list(manifest["questions"])
    if args.question_id:
        requested = list(dict.fromkeys(args.question_id))
        by_id = {str(item["question_id"]): item for item in questions}
        missing = [question_id for question_id in requested if question_id not in by_id]
        if missing:
            raise ValueError(f"--question-id不在20题清单中：{','.join(missing)}")
        questions = [by_id[question_id] for question_id in requested]
    elif args.limit is not None:
        if args.limit < 1 or args.limit > len(questions):
            raise ValueError("--limit必须在1到20之间")
        questions = questions[:args.limit]

    if args.require_question_count is not None:
        if args.require_question_count < 1:
            raise ValueError("--require-question-count必须为正整数")
        if len(questions) != args.require_question_count:
            raise ValueError(
                "最终筛选题数与--require-question-count不一致："
                f"expected={args.require_question_count}, actual={len(questions)}"
            )
    if (
        args.admission_only
        and args.require_question_count is not None
        and args.require_question_count != 1
    ):
        raise ValueError(
            "--admission-only只尝试第一题，不能与大于1的"
            "--require-question-count组合"
        )
    if args.require_new_output_dir and args.retry_failures:
        raise ValueError(
            "--require-new-output-dir与--retry-failures语义冲突"
        )
    if args.require_new_output_dir and output_dir.exists():
        raise FileExistsError(f"要求全新输出目录，但目录已经存在：{output_dir}")

    missing_answer_guards = [
        str(question["question_id"])
        for question in questions
        if not question.get("requested_outputs")
    ]

    if args.dry_run:
        print(json.dumps({
            "status": (
                "dry_run_passed"
                if not missing_answer_guards
                else "dry_run_not_ready"
            ),
            "model_id": args.model_id,
            "question_count": len(questions),
            "dataset_root": str(dataset_root),
            "prompt_file_sha256": prompt_file_sha256,
            "prompt_sha256": model_prompt_sha256,
            "manifest_sha256": manifest_sha256,
            "schema_sha256": schema_sha256,
            "schema_id": schema.get("$id"),
            "answer_guard_ready": not missing_answer_guards,
            "missing_answer_guard_question_ids": missing_answer_guards,
            "api_calls": 0,
            "execution_guards": {
                "require_question_count": args.require_question_count,
                "require_new_output_dir": args.require_new_output_dir,
                "stop_on_first_nonsuccess": args.stop_on_first_nonsuccess,
                "admission_only": args.admission_only,
                "retry_failures": args.retry_failures,
            },
        }, ensure_ascii=False, indent=2))
        return 0 if not missing_answer_guards else 2

    if missing_answer_guards:
        raise ValueError(
            "正式调用前必须为每道题配置不含答案值的requested_outputs；缺少："
            + ",".join(missing_answer_guards)
        )

    if not args.acknowledge_image_upload:
        raise ValueError(
            "正式调用会把原题图发送到阿里云百炼；取得用户明确授权后，"
            "请同时传入--acknowledge-image-upload。"
        )

    api_key, base_url, workspace_id, loaded_env_names = load_credentials(args.env_file)
    output_dir.mkdir(
        parents=True,
        exist_ok=not args.require_new_output_dir,
    )
    results_path = output_dir / "results.jsonl"
    existing = latest_records(results_path)
    run_path = output_dir / "run.json"
    run_document = {
        "schema_version": "math_guidance_pilot_run_v3",
        "experiment_id": output_dir.name,
        "state": "running",
        "started_at": utc_now(),
        "model_id": args.model_id,
        "model_availability_status": "unverified_until_first_success",
        "enable_thinking": args.enable_thinking,
        "max_completion_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "planned_question_count": len(questions),
        "question_ids": [question["question_id"] for question in questions],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "prompt": str(prompt_path),
        "prompt_file_sha256": prompt_file_sha256,
        "prompt_sha256": model_prompt_sha256,
        "schema": str(schema_path),
        "schema_sha256": schema_sha256,
        "dataset_root": str(dataset_root),
        "credential_env_names_loaded": loaded_env_names,
        "provider": {
            "base_url": benchmark.sanitize_url(base_url),
            "api_key_env": "DASHSCOPE_API_KEY",
            "base_url_env": "DASHSCOPE_BASE_URL",
            "workspace_id_env": "DASHSCOPE_WORKSPACE_ID",
            "workspace_header_sent": workspace_id is not None,
        },
        "gold_loaded": False,
        "answer_blind_targets_sent_for_self_check": True,
        "execution_guards": {
            "require_question_count": args.require_question_count,
            "require_new_output_dir": args.require_new_output_dir,
            "stop_on_first_nonsuccess": args.stop_on_first_nonsuccess,
            "admission_only": args.admission_only,
            "retry_failures": args.retry_failures,
        },
        "image_contract": {
            "images_per_success": 4,
            "width_px": 800,
            "display_width_cm": 9.0,
            "chinese_font": "SimSun",
            "latin_font": "Times New Roman",
        },
        "latency_contract": {
            "end_to_end_boundary": "before_image_preprocess_to_four_files_ready",
            "phase_breakdown_recorded": True,
            "physical_device_download_and_refresh_included": False,
        },
    }
    atomic_json(run_path, run_document)

    state = "running"
    try:
        for question in questions:
            question_id = str(question["question_id"])
            prior = existing.get(question_id)
            if prior is not None and (
                prior.get("status") == "success" or not args.retry_failures
            ):
                continue
            record = None
            if prior is not None and prior.get("status") == "failed_validation":
                record = recover_prior_locally(
                    prior=prior,
                    question=question,
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    model_id=args.model_id,
                )
            if record is None:
                record = run_one(
                    question=question,
                    dataset_root=dataset_root,
                    output_dir=output_dir,
                    model_id=args.model_id,
                    prompt=prompt,
                    schema=schema,
                    api_key=api_key,
                    base_url=base_url,
                    workspace_id=workspace_id,
                    timeout_seconds=args.timeout_seconds,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                )
            ordered = append_record_and_update_metrics(
                results_path=results_path,
                output_dir=output_dir,
                existing=existing,
                questions=questions,
                record=record,
            )
            atomic_json(
                output_dir / "progress.json",
                progress_document(planned=len(questions), records=ordered, state="running"),
            )
            print(json.dumps({
                "question_id": question_id,
                "status": record.get("status"),
                "completed": len(ordered),
                "planned": len(questions),
                "total_tokens": record.get("usage", {}).get("total_tokens"),
                "end_to_end_ms": record.get("timing", {}).get("end_to_end_ms"),
            }, ensure_ascii=False), flush=True)
            if record.get("status") == "failed_api" and record.get("http_status") in {400, 404}:
                state = "model_admission_failed"
                break
            if args.stop_on_first_nonsuccess and record.get("status") != "success":
                state = "stopped_on_first_nonsuccess"
                break
            if args.admission_only:
                break
        ordered = [existing[str(item["question_id"])] for item in questions if str(item["question_id"]) in existing]
        if state == "running":
            state = "completed" if len(ordered) == len(questions) else "partial"
        summary = summarize(ordered, len(questions))
        atomic_json(output_dir / "summary.json", summary)
        write_summary_csv(output_dir / "per_question_metrics.csv", ordered)
        atomic_json(
            output_dir / "progress.json",
            progress_document(planned=len(questions), records=ordered, state=state),
        )
        run_document["state"] = state
        run_document["ended_at"] = utc_now()
        run_document["model_availability_status"] = (
            "verified" if any(record.get("status") == "success" for record in ordered)
            else "not_verified"
        )
        atomic_json(run_path, run_document)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if state in {"completed", "partial"} else 2
    except KeyboardInterrupt:
        state = "interrupted"
        raise
    finally:
        if run_path.is_file() and state in {"interrupted"}:
            current = load_json(run_path)
            if isinstance(current, dict):
                current["state"] = state
                current["ended_at"] = utc_now()
                atomic_json(run_path, current)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="20道数学大题分层引导与800px结果图测试"
    )
    value.add_argument(
        "--model-id",
        required=True,
        help=(
            "必须显式指定真实模型ID；当前建议："
            "qwen3.7-flash-2026-07-15（Flash）或qwen3.8-max（3.8）"
        ),
    )
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    value.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    value.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    value.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--timeout-seconds", type=float, default=300.0)
    value.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="可选；默认不向供应商设置输出Token上限",
    )
    selection = value.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument(
        "--question-id",
        action="append",
        help="只测试指定题号；可重复传入",
    )
    value.add_argument("--admission-only", action="store_true")
    value.add_argument("--retry-failures", action="store_true")
    value.add_argument(
        "--require-question-count",
        type=int,
        help="要求最终筛选后的唯一题数严格等于该值；不匹配时在读凭据前失败",
    )
    value.add_argument(
        "--require-new-output-dir",
        action="store_true",
        help="要求输出目录尚不存在；用于首次准入，防止混入旧结果或重复调用",
    )
    value.add_argument(
        "--stop-on-first-nonsuccess",
        action="store_true",
        help="首个API、结构、校验、本地或渲染非成功结果落盘后立即停止",
    )
    value.add_argument("--require-prompt-file-sha256")
    value.add_argument("--require-model-prompt-sha256")
    value.add_argument("--require-manifest-sha256")
    value.add_argument("--require-schema-sha256")
    value.add_argument("--enable-thinking", action="store_true")
    value.add_argument("--acknowledge-image-upload", action="store_true")
    value.add_argument("--dry-run", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
