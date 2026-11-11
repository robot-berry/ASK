#!/usr/bin/env python3
"""Run the answer-blind 20-item physics-guidance image pilot.

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
from physics_guidance_renderer import render_four_images  # noqa: E402


SCHEMA_VERSION = "physics_guidance_pilot_result_v4_multi_method"
FROZEN_MANIFEST_IDS = frozenset(
    {
        "physics_guidance_pilot20_calculation_v2",
        "physics_guidance_pilot20_calculation_complete_t03q14_v3",
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
        "primary_steps": (2, 3),
        "alternative_steps": (2, 3),
    },
    "standard": {
        "title": "普通版",
        "primary_steps": (3, 4),
        "alternative_steps": (3, 4),
    },
    "foundation": {
        "title": "基础版",
        "primary_steps": (4, 5),
        "alternative_steps": (4, 5),
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
DEFAULT_PROMPT = SCRIPT_DIR / "物理题解析提示词.md"
DEFAULT_MANIFEST = (
    SCRIPT_DIR / "physics_guidance_pilot20_manifest_complete_t03q14_v3.json"
)
DEFAULT_SCHEMA = SCRIPT_DIR / "physics_guidance_schema.json"
DEFAULT_ENV = SCRIPT_DIR / "api_keys.env"
DEFAULT_OUTPUT = (
    SCRIPT_DIR
    / "runs"
    / "physics_guidance_run"
)
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DOCUMENTED_MODEL_CHOICES = (
    "qwen3.7-flash-2026-07-15",
    "qwen3.8-max",
)
UNAVAILABLE_REQUESTED_MODEL_IDS = {"qwen3.8-flash"}
INLINE_FORMULA = re.compile(r"\$([^$]+)\$")
SUBSTANTIVE_GUIDANCE_CONTENT = re.compile(
    r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)
PRINTED_PART = re.compile(
    r"[（(]\s*(\d+)\s*[）)]\s*([①②③④⑤⑥⑦⑧⑨⑩])?"
)
PRINTED_PART_TOKEN = re.compile(
    r"[（(]\s*(\d+)\s*[）)]\s*([①②③④⑤⑥⑦⑧⑨⑩])?|([①②③④⑤⑥⑦⑧⑨⑩])"
)
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
    r"(?:=|≈|<|>|≤|≥|\\approx|\\simeq|\\leq?|\\geq?)"
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
    if source.count("{") != source.count("}"):
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
    # Explicit ``\text{...}`` operands may legitimately contain a Latin symbol
    # followed by Chinese prose; that is not an implicit Chinese subscript.
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


def printed_leaf_part_labels(question_text: str) -> list[str]:
    """Return the actual answer-bearing leaves from hierarchical part labels.

    A parent label such as ``（2）`` is structural when it is followed by
    standalone children ``①`` ... ``④``.  In that case guidance must cover
    every ``（2）①`` ... ``（2）④`` leaf, but need not add a redundant
    ``（2）`` guidance block.  Parents without children remain required leaves.
    """

    parent_order: list[str] = []
    children_by_parent: dict[str, list[str]] = {}
    current_parent = ""
    for match in PRINTED_PART_TOKEN.finditer(question_text):
        parent_number = match.group(1)
        if parent_number:
            current_parent = f"（{parent_number}）"
            if current_parent not in parent_order:
                parent_order.append(current_parent)
                children_by_parent[current_parent] = []
            child = match.group(2)
        else:
            child = match.group(3)
        if child and current_parent:
            nested_label = f"{current_parent}{child}"
            children = children_by_parent[current_parent]
            if nested_label not in children:
                children.append(nested_label)

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
        raise ValueError("必须使用已冻结的物理计算大题清单")
    selection_rule = document.get("selection_rule")
    if not isinstance(selection_rule, dict):
        raise ValueError("manifest缺少计算大题选择规则")
    if selection_rule.get("required_visual_type") != "calculation_large_response":
        raise ValueError("manifest未强制限定为计算大题")
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
        if question.get("subject") != "physics":
            raise ValueError(f"{question_id}不是物理题")
        selection_evidence = question.get("selection_evidence")
        if not isinstance(selection_evidence, dict):
            raise ValueError(f"{question_id}缺少计算大题人工复核证据")
        if selection_evidence.get("visual_type") != "calculation_large_response":
            raise ValueError(f"{question_id}不是已复核的物理计算大题")
        if selection_evidence.get("experiment_excluded") is not True:
            raise ValueError(f"{question_id}未明确排除实验题")
        if (
            selection_evidence.get("visual_review_status")
            != "verified_calculation_large_response"
        ):
            raise ValueError(f"{question_id}尚未通过计算大题人工视觉复核")
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
        if not isinstance(item, dict) or set(item) != {
            "part",
            "symbol_aliases",
            "prose_aliases",
        }:
            raise ValueError(
                f"{question_id} requested_outputs[{index}]字段必须为"
                "part/symbol_aliases/prose_aliases"
            )
        part = normalized_part_label(item.get("part"))
        if not part or part in seen_parts:
            raise ValueError(f"{question_id} requested_outputs小问编号为空或重复：{part!r}")
        seen_parts.add(part)
        aliases: list[str] = []
        for key in ("symbol_aliases", "prose_aliases"):
            raw_aliases = item.get(key)
            if not isinstance(raw_aliases, list):
                raise ValueError(f"{question_id} {part} {key}必须是数组")
            for alias in raw_aliases:
                if not isinstance(alias, str) or not alias.strip():
                    raise ValueError(f"{question_id} {part} {key}含空值")
                if any(marker in alias for marker in ("=", "≈", "答案")):
                    raise ValueError(f"{question_id} {part} {key}不得包含答案或等式")
                aliases.append(alias.strip())
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
        elif require_substantive and not SUBSTANTIVE_GUIDANCE_CONTENT.search(item):
            errors.append(
                f"{path}[{index}]必须包含文字、数字或完整公式，不能只含标点或空格"
            )
        else:
            output.append(item.strip())
    return output


def compact_physics_relation(source: str) -> str:
    """Normalize harmless notation differences for narrow physics guards."""

    value = str(source).replace("$", "")
    for marker in (
        r"\left",
        r"\right",
        r"\,",
        r"\;",
        r"\!",
        r"\quad",
        r"\qquad",
        r"\cdot",
        r"\times",
        "~",
    ):
        value = value.replace(marker, "")
    value = re.sub(r"\s+", "", value)
    value = value.replace("v_{0}", "v0").replace("v_0", "v0")
    value = value.replace("^{2}", "^2")
    return value


def validate_physics_consistency(
    source: str,
    path: str,
    errors: list[str],
) -> None:
    """Reject two dimensionally invalid electrostatic-acceleration relations.

    This deliberately does not attempt general symbolic algebra.  It only
    blocks the two concrete, universally invalid spellings observed in the
    production result while leaving equivalent correct relations untouched.
    """

    compact = compact_physics_relation(source)
    charge_work_over_mass = r"(?:qU/m|Uq/m|\\frac\{(?:qU|Uq)\}\{m\})"
    if re.search(
        rf"(?:v0={charge_work_over_mass}|{charge_work_over_mass}=v0)",
        compact,
    ):
        errors.append(
            f"{path}静电加速关系错误：从静止加速的速度不能写成v_0=qU/m"
        )

    reciprocal_kinetic_energy = (
        r"(?:1/\(2mv0\^2\)|"
        r"\\frac\{1\}\{\(?2mv0\^2\)?\})"
    )
    if re.search(
        rf"(?:(?:qU|Uq)={reciprocal_kinetic_energy}|"
        rf"{reciprocal_kinetic_energy}=(?:qU|Uq))",
        compact,
    ):
        errors.append(
            f"{path}静电加速关系错误：qU不能等于动能表达式的倒数"
        )

    reciprocal_kinematics = (
        r"(?:1/\(2at\^2\)|"
        r"\\frac\{1\}\{\(?2at\^2\)?\})"
    )
    if re.search(
        rf"(?:(?:x|y|s)={reciprocal_kinematics}|"
        rf"{reciprocal_kinematics}=(?:x|y|s))",
        compact,
    ):
        errors.append(
            f"{path}匀加速位移关系错误：位移不能写成1/(2at^2)"
        )


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
    for path, value, part_label in display_guidance_entries(output):
        yield path, value, part_label


def guidance_strings(output: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    for path, value, _ in guidance_entries(output):
        yield path, value


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
    prose_targets: list[tuple[str, str]] = []
    for item in requested_outputs:
        part = normalized_part_label(item.get("part"))
        for alias in item.get("symbol_aliases", []):
            if isinstance(alias, str) and alias.strip():
                symbol_targets.append((part, compact_formula_side(alias.strip("$ "))))
        for alias in item.get("prose_aliases", []):
            if isinstance(alias, str) and alias.strip():
                prose_targets.append((part, alias.strip()))

    symbols_by_part: dict[str, list[str]] = {}
    prose_by_part: dict[str, list[str]] = {}
    for part, target in symbol_targets:
        symbols_by_part.setdefault(part, []).append(target)
    for part, alias in prose_targets:
        prose_by_part.setdefault(part, []).append(alias)

    for path, value, current_part in guidance_entries(output):
        if current_part is None:
            active_symbols = [target for _, target in symbol_targets]
            active_prose = [alias for _, alias in prose_targets]
            active_part = "通用总结"
        else:
            active_symbols = symbols_by_part.get(current_part, [])
            active_prose = prose_by_part.get(current_part, [])
            active_part = current_part
        for formula_index, formula in enumerate(INLINE_FORMULA.findall(value)):
            sides = EQUATION_SEPARATOR.split(formula)
            if len(sides) < 2:
                continue
            compact_sides = [compact_formula_side(side) for side in sides]
            for target in active_symbols:
                isolated = any(
                    side == target
                    or side.startswith(target + r"\mathrm{")
                    for side in compact_sides
                )
                if not isolated:
                    continue
                if policy == "no_answer":
                    numeric_result = bool(
                        NUMERIC_WITH_UNIT.search(formula)
                        or ISOLATED_NUMERIC_RESULT.search(
                            "=" + compact_sides[-1]
                        )
                    )
                    conclusion_context = bool(
                        re.search(
                            r"(?:最终|答案|解得|求得|算得|得到|可得|因此|所以|故)",
                            value,
                        )
                    )
                    if not numeric_result and not conclusion_context:
                        continue
                errors.append(
                    f"{path}.formula[{formula_index}]单独解出了"
                    f"{active_part}所求量{target}"
                )
        for alias in active_prose:
            escaped = re.escape(alias)
            direct_after = re.search(
                escaped
                + r".{0,8}(?:为|等于|是|应为|约为|方向为)"
                + r".{0,16}(?:\d|向左|向右|向上|向下|成立|不成立|增大|减小|不变|小于|大于|等于)",
                value,
            )
            direct_before = re.search(
                r"(?:得到|求得|算得|可得).{0,8}"
                + escaped
                + r".{0,8}(?:为|等于|是|约为)?.{0,12}(?:\d|向左|向右|向上|向下|小于|大于|等于)",
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
    if output.get("subject") != "physics":
        errors.append("subject必须为physics")
    if output.get("problem_scope") != "primary_to_high_school":
        errors.append("problem_scope必须为primary_to_high_school")
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
    if not isinstance(formulae, list) or not 0 <= len(formulae) <= 3:
        errors.append("base_formulas必须包含0到3项")
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
                    validate_physics_consistency(
                        formula,
                        f"base_formulas[{index}].formula",
                        errors,
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
                if tier_key == "standard" and (
                    not device_policy
                    or (isinstance(part, Mapping) and "pitfall" in part)
                ):
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

                primary_min, primary_max = (
                    (1, 12) if device_policy else spec["primary_steps"]
                )
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
                alternative_min, alternative_max = (
                    (1, 12) if device_policy else spec["alternative_steps"]
                )
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

                if not device_policy and tier_key == "standard" and (
                    not isinstance(part.get("pitfall"), str)
                    or not str(part.get("pitfall")).strip()
                ):
                    errors.append(f"{part_path}.pitfall不能为空")
            if not device_policy and tier_key == "foundation" and (
                not isinstance(tier.get("pattern_summary"), str)
                or not str(tier.get("pattern_summary")).strip()
            ):
                errors.append("guidance.foundation.pattern_summary不能为空")

    if tier_part_sequences:
        sequences = list(tier_part_sequences.values())
        if any(sequence != sequences[0] for sequence in sequences[1:]):
            errors.append("三个档位中的小问编号和顺序必须完全一致")
        signatures = list(tier_method_signatures.values())
        if (
            not device_policy
            and any(signature != signatures[0] for signature in signatures[1:])
        ):
            errors.append("同一小问在三个档位中的方法编号、名称和顺序必须完全一致")
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
        validate_physics_consistency(value, path, errors)
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
    return errors


def canonicalize_simple_fractions(source: str) -> str:
    """Convert top-level slash notation into display fractions."""
    value = source
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
            elif round_depth == 0 and brace_depth == 0 and character in "=,+-<>/":
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
            elif round_depth == 0 and brace_depth == 0 and character in "=,+-<>/":
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
COFRAC_TOKEN = re.compile(r"(?<![A-Za-z0-9_\\])Cofrac(?=\{)")
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
    after = COFRAC_TOKEN.sub(r"\\frac", value)
    if after != value:
        changes.append({
            "rule_id": "display_cofrac_full_token",
            "before": value,
            "after": after,
        })
        value = after
    return value, changes


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


def strip_redundant_chinese_step_prefix(value: str) -> str:
    """Remove only a delimited Chinese ordinal at the start of a step."""
    match = REDUNDANT_CHINESE_STEP_PREFIX.match(value)
    if match is None or not value[match.end():].strip():
        return value
    return value[match.end():]


def normalize_formula_notation(source: str) -> str:
    """Apply only meaning-preserving display repairs before local validation."""
    value = source.strip()
    if value.startswith("$") and value.endswith("$") and value.count("$") == 2:
        value = value[1:-1].strip()
    value, _ = apply_display_formula_whitelist(value)
    value = value.replace("_∞", r"_{\infty}")
    value = value.replace("∞", r"\infty")
    value = value.replace("∝", r"\propto")
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
        r"\\(Delta|theta|omega|Omega|mu|rho|alpha|beta|gamma|lambda|propto|infty)"
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
    # Frequent Qwen transcription slips.  These repairs only restore a
    # missing command slash or separator; they do not alter any quantity.
    value = re.sub(r"(?<![A-Za-z\\])Coe?frac(?=\s*\{)", r"\\frac", value)
    value = re.sub(r"(?<![A-Za-z\\])left(?=\s*[([{])", r"\\left", value)
    value = re.sub(r"(?<![A-Za-z\\])right(?=\s*[)\]}])", r"\\right", value)
    value = re.sub(r"(?<=[0-9.])mathrm(?=\s*\{)", r"\\mathrm", value)
    value = re.sub(r"(?<![A-Za-z\\])cpi(?![A-Za-z])", r"c\\pi", value)
    value = re.sub(r"(?<![A-Za-z\\])([0-9]+)pi(?![A-Za-z])", r"\1\\pi", value)
    value = re.sub(r"\\pi(?=[rx](?:\b|[_^]))", r"\\pi ", value)
    value = re.sub(
        r"(?<![\\A-Za-z])(pi|theta|omega|Omega|Delta|mu|rho|alpha|beta|"
        r"gamma|lambda|sin|cos|tan|circ|times)(?![A-Za-z])",
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
        _, changes = apply_display_formula_whitelist(source)
        for change in changes:
            record_rule_hit(path, change)
        return normalize_formula_notation(source)

    def normalize_display_text(source: str, path: str) -> str:
        closed = close_single_trailing_math_delimiter(source)
        if closed != source:
            record_rule_hit(path, {
                "rule_id": "display_close_single_trailing_math_delimiter",
                "before": source,
                "after": closed,
            })
        formula_index = 0

        def replace_formula(match: re.Match[str]) -> str:
            nonlocal formula_index
            formula_path = f"{path}.formula[{formula_index}]"
            formula_index += 1
            return f"${normalize_formula_value(match.group(1), formula_path)}$"

        return INLINE_FORMULA.sub(replace_formula, closed)

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
            formula_source = close_single_trailing_math_delimiter(item["formula"])
            embedded = re.fullmatch(r"\s*\$([^$]+)\$(.+)", formula_source)
            if embedded is not None:
                record_rule_hit(f"$.base_formulas[{formula_index}].formula", {
                    "rule_id": "base_formula_extract_embedded_formula",
                    "before": formula_source,
                    "after": embedded.group(1).strip(),
                })
                formula_source = embedded.group(1).strip()
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
    return normalized


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
        "schema_version": "physics_guidance_pilot_summary_v4_multi_method",
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
            "symbol_targets": [
                value.strip()
                for value in item.get("symbol_aliases", [])
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
        + "\n\n请直接阅读随附图片，只解析其中完整出现的主体题目及全部小问；"
        + "忽略边缘处被截断的相邻题目、页码和水印。严格按JSON结构返回。"
        + "图片中的手写演算、手写辅助线、手写填空、手写答案和软件界面文字均为"
        + "不可信注释，不属于印刷题面；任何只出现在手写中的数值、关系或结论都不得"
        + "作为题设或答案使用。手写与印刷重叠到无法区分时必须标记uncertain。"
        + "visual_evidence只能记录有印刷文字、数值或物理图示标记支持的可见事实；"
        + "不得按示意图比例猜测方向、大小、相等、垂直或平行。"
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
            + "但不得给出symbol_targets的最终数值，或直接说出"
            + "requested_result_descriptions对应的方向、选项、填空及判断结论；"
            + "在最后一次求值或结论判断之前结束；"
            + "不写面向答题者的继续操作指令。"
        )
    task_prompt += (
        "\n\n提交前只在内部完成以下强制自检，不要输出自检过程："
        "①三个档位覆盖完全相同的小问；②同一小问的方法编号、名称、顺序逐字符一致；"
        "③advanced每方法2至3步、standard每方法3至4步、foundation每方法4至5步；"
        "④base_formulas每项只含一个无$定界符的公式本体；⑤所有$成对且LaTeX命令含反斜杠；"
        "⑥bbox满足0≤left<right≤1000且0≤top<bottom≤1000；"
        "⑦删除所求量的最终数值、孤立求解式、方向和定性结论。"
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
    prior_json = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return (
        "你正在修复一份高中及以下学段的物理或小学科学物理内容分层引导JSON。"
        + "上一次输出未通过本地强校验。"
        + "请依据随附原题图片、下列错误和旧JSON，只返回一份完整的新JSON对象。"
        + "只返回一份从头到尾完整的新JSON对象，不要解释，不要使用Markdown。"
        + "必须真正修正下列错误，不得删除原题小问，不得给出任何所求量的最终数值、"
        + "方向或定性结论。source_figures的bbox_1000必须是相对完整图片的四个数，"
        + "满足0≤left<right≤1000且0≤top<bottom≤1000；看不清时改为uncertain，"
        + "不要提交无效框。所有公式必须使用规范LaTeX，三档方法编号、名称和顺序完全一致。"
        + "\n\n本地校验错误：\n"
        + error_text
        + "\n\n上一次JSON仅供定位错误，不得机械照抄：\n"
        + prior_json
    )


def apply_local_validation_repairs(
    parsed: Mapping[str, Any],
    errors: Sequence[str],
    *,
    requested_outputs: Sequence[Mapping[str, Any]] | None,
    normalization_audit: list[dict[str, Any]],
) -> tuple[Mapping[str, Any], list[str]]:
    """Resolve deterministic notation and answer-leak errors before API repair."""
    local_audit: list[dict[str, Any]] = []
    candidate = redact_answer_guard_violations(parsed, errors, local_audit)
    candidate = align_redundant_tier_parts(candidate, errors, local_audit)
    candidate = align_question_figure_blocks(candidate, errors, local_audit)
    normalized, remaining = parse_model_json(
        json.dumps(candidate, ensure_ascii=False),
        requested_outputs=requested_outputs,
        normalization_audit=normalization_audit,
        device_policy=True,
    )
    if isinstance(normalized, Mapping) and len(remaining) <= len(errors):
        normalization_audit.extend(local_audit)
        return normalized, remaining
    return parsed, list(errors)


def align_redundant_tier_parts(
    output: Mapping[str, Any],
    errors: Sequence[str],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Drop only trailing duplicate/summary parts when tier prefixes align."""
    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    if not any("三个档位中的小问编号和顺序必须完全一致" in e for e in errors):
        return candidate
    guidance = candidate.get("guidance")
    if not isinstance(guidance, dict):
        return candidate
    tiers = ("advanced", "standard", "foundation")
    part_lists = []
    for tier in tiers:
        block = guidance.get(tier)
        parts = block.get("parts") if isinstance(block, dict) else None
        if not isinstance(parts, list):
            return candidate
        part_lists.append(parts)
    target = min(len(parts) for parts in part_lists)
    if target == 0 or len({len(parts) for parts in part_lists}) == 1:
        return candidate
    prefixes = [[str(part.get("part", "")) for part in parts[:target]] for parts in part_lists]
    if not all(prefix == prefixes[0] for prefix in prefixes[1:]):
        return candidate
    for tier, parts in zip(tiers, part_lists):
        if len(parts) > target:
            del parts[target:]
            audit.append({"action": "drop_redundant_trailing_tier_parts", "tier": tier})
    return candidate


def align_question_figure_blocks(
    output: Mapping[str, Any],
    errors: Sequence[str],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Insert omitted figure placeholders without changing recognized crops."""
    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    if not any(
        "question_blocks与source_figures的figure_id及顺序必须完全一致" in error
        for error in errors
    ):
        return candidate
    blocks = candidate.get("question_blocks")
    figures = candidate.get("source_figures")
    if not isinstance(blocks, list) or not isinstance(figures, list):
        return candidate
    expected = [
        str(figure.get("figure_id") or "")
        for figure in figures
        if isinstance(figure, Mapping) and str(figure.get("figure_id") or "")
    ]
    observed = [
        str(block.get("figure_id") or "")
        for block in blocks
        if isinstance(block, Mapping) and block.get("type") == "figure"
    ]
    # Only add placeholders when the observed list is an ordered subsequence.
    cursor = 0
    for figure_id in observed:
        try:
            cursor = expected.index(figure_id, cursor) + 1
        except ValueError:
            return candidate
    for index, figure_id in enumerate(expected):
        if figure_id in observed:
            continue
        insert_at = len(blocks)
        for previous_id in reversed(expected[:index]):
            positions = [
                position
                for position, block in enumerate(blocks)
                if isinstance(block, Mapping)
                and block.get("type") == "figure"
                and block.get("figure_id") == previous_id
            ]
            if positions:
                insert_at = positions[-1] + 1
                break
        blocks.insert(insert_at, {
            "type": "figure",
            "text": "",
            "figure_id": figure_id,
        })
        observed.append(figure_id)
        audit.append({"action": "insert_missing_figure_block", "figure_id": figure_id})
    return candidate


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
    """Remove only locally identified final-answer leaks, then revalidate."""
    redacted = json.loads(json.dumps(output, ensure_ascii=False))
    formula_ops: list[tuple[str, int, str]] = []
    prose_ops: list[tuple[str, str]] = []
    for error in errors:
        if "单独解出了" in error:
            path = error.split("单独解出了", 1)[0]
            match = re.fullmatch(r"(.+)\.formula\[(\d+)\]", path)
            if match:
                formula_ops.append((match.group(1), int(match.group(2)), error))
        elif "直接给出了" in error:
            path = error.split("直接给出了", 1)[0]
            prose_ops.append((path, error))
        elif "公式本体不得混入中文说明文字" in error:
            path = error.split("公式本体不得混入中文说明文字", 1)[0]
            match = re.fullmatch(r"(.+)\.formula\[(\d+)\]", path)
            if match:
                formula_ops.append((match.group(1), int(match.group(2)), error))

    for path, formula_index, error in sorted(
        formula_ops,
        key=lambda item: (item[0], -item[1]),
    ):
        resolved = resolve_output_path(redacted, path)
        if resolved is None:
            continue
        parent, key = resolved
        value = parent[key]
        if not isinstance(value, str):
            continue
        matches = list(re.finditer(r"\$[^$]+\$", value))
        if formula_index >= len(matches):
            continue
        target = matches[formula_index]
        replacement = "保留含所求量的未解关系式"
        parent[key] = (
            value[:target.start()] + replacement + value[target.end():]
        ).strip()
        audit.append({
            "action": "redact_final_answer_formula",
            "path": path,
            "validator_error": error,
        })

    for path, error in prose_ops:
        resolved = resolve_output_path(redacted, path)
        if resolved is None:
            continue
        parent, key = resolved
        if not isinstance(parent[key], str):
            continue
        parent[key] = (
            "根据相应定义式、状态关系和约束条件进行比较，"
            "在此处保留判断方向，不写出最终结论。"
        )
        audit.append({
            "action": "redact_final_qualitative_conclusion",
            "path": path,
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
    candidate = align_redundant_tier_parts(candidate, errors, audit)
    candidate = align_question_figure_blocks(candidate, errors, audit)
    normalization_audit: list[dict[str, Any]] = []
    normalized_candidate, remaining = parse_model_json(
        json.dumps(candidate, ensure_ascii=False),
        requested_outputs=question.get("requested_outputs"),
        normalization_audit=normalization_audit,
        device_policy=True,
    )
    if remaining or not isinstance(normalized_candidate, Mapping):
        return None
    # A validator failure can be resolved by formula normalization even when
    # the normalizer does not emit an audit item.  Revalidation above is the
    # authority; do not force another provider call solely for an empty audit.
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
            if validation_errors and isinstance(parsed, Mapping):
                parsed, validation_errors = apply_local_validation_repairs(
                    parsed,
                    validation_errors,
                    requested_outputs=question.get("requested_outputs"),
                    normalization_audit=normalization_audit,
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
                    if repair_errors and isinstance(repaired, Mapping):
                        repaired, repair_errors = apply_local_validation_repairs(
                            repaired,
                            repair_errors,
                            requested_outputs=question.get("requested_outputs"),
                            normalization_audit=repair_normalization,
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
            if False and validation_errors and isinstance(parsed, Mapping):
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
            "subject": "physics",
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
            "subject": "physics",
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
        "schema_version": "physics_guidance_pilot_run_v3",
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
            if prior is not None and prior.get("status") in {
                "failed_validation",
                "failed_render",
            }:
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
        description="20道高中物理大题分层引导与800px结果图测试"
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
