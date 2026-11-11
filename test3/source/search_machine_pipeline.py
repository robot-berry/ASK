#!/usr/bin/env python3
"""Camera photo -> preprocessing -> mathematics guidance images."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping
from urllib.parse import urlparse
import uuid

from PIL import Image

import full_benchmark as benchmark
import math_guidance_pilot as pilot
from device_result_renderer import (
    resolve_device_fonts,
    validate_device_image_set,
)
from math_guidance_renderer import render_four_images
from preprocessor import QuestionImageProcessor


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
RUNS_ROOT = PROJECT_ROOT / "device_runs"
DEFAULT_SCHEMA = SOURCE_DIR / "math_guidance_schema.json"
DEFAULT_PROMPT = SOURCE_DIR / "数学题解析提示词.md"
DEFAULT_ENV = SOURCE_DIR / "api_keys.env"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL_ID = "qwen3.7-flash-2026-07-15"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
ACCEPTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
ROOT_KEYS = pilot.ROOT_KEYS


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON根值必须是对象：{path}")
    return value


def safe_filename(value: str) -> str:
    name = Path(value or "camera.jpg").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120] or "camera.jpg"


def load_credentials(env_file: Path = DEFAULT_ENV) -> tuple[str, str, str | None]:
    if env_file.is_file():
        benchmark.parse_env_file(env_file)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少DASHSCOPE_API_KEY；请在source/api_keys.env中配置轮换后的密钥")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip() or DEFAULT_BASE_URL
    parsed_url = urlparse(base_url)
    approved_host = (
        parsed_url.hostname == "dashscope.aliyuncs.com"
        or bool(
            parsed_url.hostname
            and parsed_url.hostname.endswith(".maas.aliyuncs.com")
        )
    )
    if (
        parsed_url.scheme != "https"
        or not approved_host
        or parsed_url.path.rstrip("/") != "/compatible-mode/v1"
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise RuntimeError("DASHSCOPE_BASE_URL必须使用已批准的阿里云百炼HTTPS地址")
    workspace = os.environ.get("DASHSCOPE_WORKSPACE_ID", "").strip() or None
    return api_key, base_url, workspace


def png_data_url(path: Path) -> tuple[str, dict[str, Any]]:
    """Transmit the exact preprocessed PNG used as the renderer source."""

    raw = path.read_bytes()
    with Image.open(path) as image:
        image.load()
        audit = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format,
            "transmitted_media_type": "image/png",
            "same_bytes_as_render_source": True,
        }
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii"), audit


def build_request_payload(
    *,
    model_id: str,
    prompt: str,
    data_url: str,
    schema: Mapping[str, Any],
    max_tokens: int | None,
) -> dict[str, Any]:
    # Reuse the proven mathematics prompt expansion and answer-blind rules.
    return pilot.request_payload(
        model_id=model_id,
        prompt=prompt,
        data_url=data_url,
        schema=schema,
        requested_outputs=None,
        max_tokens=max_tokens,
        enable_thinking=False,
    )


def merge_attempt_usage(*attempts: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate provider usage across the initial and one repair request."""

    keys = (
        "input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "cached_input_tokens",
        "total_tokens",
    )
    merged: dict[str, Any] = {}
    for key in keys:
        known = [
            attempt.get("usage", {}).get(key)
            for attempt in attempts
            if isinstance(attempt.get("usage"), Mapping)
            and isinstance(attempt.get("usage", {}).get(key), int)
            and not isinstance(attempt.get("usage", {}).get(key), bool)
        ]
        merged[key] = sum(known) if known else None
    merged["total_tokens_source"] = "provider_aggregate"
    merged["reported"] = all(
        isinstance(attempt.get("usage"), Mapping)
        and attempt["usage"].get("reported") is True
        for attempt in attempts
    )
    return merged


FORMULA_FIGURE_CAPTION = re.compile(r"(?:公式|等式|不等式|表达式|计算式)")
TRAILING_INCOMPLETE_FORMULA = re.compile(
    r"\$(?P<formula>[^$]{1,120}=\s*)\$(?P<suffix>\s*[，．。；;:]*)$"
)


def normalize_formula_source_figures(
    output: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Turn a model-misclassified printed formula back into faithful text.

    Some vision responses describe the right-hand side of a displayed formula
    as a source figure and place its bbox on a neighbouring question line.  We
    only repair the unambiguous case: a text block ends with an incomplete
    equation, the next figure explicitly calls itself a formula, and an exact
    equation with the same left-hand side already exists in ``base_formulas``.
    No mathematical expression is generated or algebraically changed.
    """

    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    blocks = candidate.get("question_blocks")
    figures = candidate.get("source_figures")
    formulas = candidate.get("base_formulas")
    if not isinstance(blocks, list) or not isinstance(figures, list):
        return candidate, []
    formula_by_lhs: dict[str, str] = {}
    for item in formulas or []:
        if not isinstance(item, Mapping):
            continue
        formula = str(item.get("formula") or "").strip().strip("$")
        sides = pilot.EQUATION_SEPARATOR.split(formula, maxsplit=1)
        if len(sides) != 2:
            continue
        lhs = pilot.compact_formula_side(sides[0])
        if lhs and sides[1].strip():
            formula_by_lhs.setdefault(lhs, formula)
    figures_by_id = {
        str(item.get("figure_id") or ""): item
        for item in figures
        if isinstance(item, Mapping) and str(item.get("figure_id") or "")
    }
    removed: set[str] = set()
    audit: list[dict[str, Any]] = []
    repaired_blocks: list[Any] = []
    for block in blocks:
        if not (
            isinstance(block, Mapping)
            and block.get("type") == "figure"
            and repaired_blocks
        ):
            repaired_blocks.append(block)
            continue
        figure_id = str(block.get("figure_id") or "")
        figure = figures_by_id.get(figure_id)
        previous = repaired_blocks[-1]
        if not (
            isinstance(figure, Mapping)
            and FORMULA_FIGURE_CAPTION.search(str(figure.get("caption") or ""))
            and isinstance(previous, dict)
            and previous.get("type") == "text"
        ):
            repaired_blocks.append(block)
            continue
        previous_text = str(previous.get("text") or "")
        match = TRAILING_INCOMPLETE_FORMULA.search(previous_text)
        if match is None:
            repaired_blocks.append(block)
            continue
        lhs = pilot.compact_formula_side(
            pilot.EQUATION_SEPARATOR.split(match.group("formula"), maxsplit=1)[0]
        )
        full_formula = formula_by_lhs.get(lhs)
        if not full_formula:
            repaired_blocks.append(block)
            continue
        repaired_text = (
            previous_text[: match.start()]
            + f"${full_formula}$"
            + match.group("suffix")
        )
        previous["text"] = repaired_text
        removed.add(figure_id)
        audit.append({
            "operation": "formula_source_figure_to_text",
            "figure_id": figure_id,
            "before": previous_text,
            "after": repaired_text,
        })
    if removed:
        candidate["question_blocks"] = repaired_blocks
        candidate["source_figures"] = [
            item
            for item in figures
            if not (
                isinstance(item, Mapping)
                and str(item.get("figure_id") or "") in removed
            )
        ]
    return candidate, audit


def parse_model_output(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    normalization_audit: list[dict[str, Any]] = []
    parsed, errors = pilot.parse_model_json(
        text,
        requested_outputs=None,
        normalization_audit=normalization_audit,
        # Match the mathematics hardware renderer policy. Explicit answer
        # phrases remain forbidden by the shared validator and system prompt.
        device_policy=True,
    )
    result = dict(parsed) if isinstance(parsed, Mapping) else None
    if result is not None:
        result, formula_figure_audit = normalize_formula_source_figures(result)
        if formula_figure_audit:
            errors = []
        errors.extend(validate_normalized_output(result))
    return result, list(dict.fromkeys(errors))


TARGET_CLAUSE = re.compile(
    r"(?:求解|解(?:方程组|方程|不等式)|求|计算|确定|比较|判断|证明|证出|求证|说明|写出)"
    r"(?P<body>[^。；;\n？?]{1,100})"
)
QUESTION_TARGET_CLAUSE = re.compile(
    r"(?P<body>[^。；;\n]{1,100}?)(?:分别)?(?:是多少|多少|为何|怎样|如何)\s*[？?]?|"
    r"(?P<decision>(?:能否|是否|有无|有没有)[^。；;\n？?]{1,100})\s*[？?]?"
)
TARGET_NOUN = re.compile(
    r"(?:数学期望|期望|频率|概率|最大值|最小值|取值范围|参数范围|定义域|值域|"
    r"解析式|表达式|解集|单调区间|零点个数|根的个数|个数|面积|体积|长度|"
    r"距离|斜率|坐标|大小关系|二面角|夹角|结论)"
)
QUANTIFIED_TARGET_CLAUSE = re.compile(
    r"(?P<alias>还剩|剩下|一共(?:有)?|总共(?:有)?|共有|现在有|"
    r"需(?:要)?|用去|用了|花费|花了|节省|相差|多出|少了|"
    r"增加(?:了)?|减少(?:了)?|每[一-鿿]{1,5}|平均(?:每[一-鿿]{1,4})?)"
    r"[^，,。；;\n？?\d]{0,12}?多少\s*(?P<unit>[一-鿿A-Za-z]{0,4})"
)
TARGET_FORMULA = re.compile(r"\$([^$]{1,64})\$")
PROOF_CLOSURE = re.compile(
    r"(?:得证|证毕|证明完毕|Q\.E\.D\.?|"
    r"(?:故|所以|因此|从而).{0,18}(?:原命题|命题|结论).{0,6}成立)"
)
CHOICE_OR_FILL_ANSWER = re.compile(
    r"(?:正确选项|应选|选择|选项)(?:为|是|：|:)?\s*[A-D](?![A-Za-z])|"
    r"(?:应填|填入|填空答案)(?:为|是|：|:)?\s*[^，。；\n]{1,24}"
)
EXPLICIT_FINAL_FORMULA = re.compile(
    r"(?:最终(?:得到|求得|解得|可得)|答案(?:为|是)|结果(?:为|是)|"
    r"解得|求得|算得)\s*[:：]?\s*\$[^$]+\$"
)


def _target_formula_alias(value: str) -> str | None:
    compact = pilot.compact_formula_side(value.strip())
    if not compact or len(compact) > 64 or pilot.EQUATION_SEPARATOR.search(compact):
        return None
    # A target should name an object, not repeat a complete condition or a
    # purely numeric constant from the question.
    if not re.search(r"[A-Za-zΑ-Ωα-ω\\]", compact):
        return None
    return compact


def infer_dynamic_requested_outputs(
    output: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Infer answer-blind targets from the transcribed question itself.

    Phone uploads do not have the frozen manifest used by the 20-item pilot.
    This conservative extractor supplies the existing part-aware answer guard
    with explicit ``求/比较/证明`` clauses without inventing any answer data.
    """

    blocks = output.get("question_blocks")
    question_text = "\n".join(
        str(block.get("text") or "")
        for block in blocks or []
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    guidance = output.get("guidance")
    advanced = guidance.get("advanced") if isinstance(guidance, Mapping) else None
    parts = advanced.get("parts") if isinstance(advanced, Mapping) else None
    labels = [
        pilot.normalized_part_label(item.get("part"))
        for item in parts or []
        if isinstance(item, Mapping) and pilot.normalized_part_label(item.get("part"))
    ]
    if not labels:
        labels = ["整题"]

    part_token = re.compile(r"[（(]\s*\d+\s*[）)](?:\s*[①-⑩])?|[①-⑩]")
    token_matches = list(part_token.finditer(question_text))
    part_text: dict[str, str] = {}
    for index, match in enumerate(token_matches):
        label = pilot.normalized_part_label(match.group(0))
        end = token_matches[index + 1].start() if index + 1 < len(token_matches) else len(question_text)
        part_text[label] = question_text[match.end():end]
    if not token_matches:
        part_text["整题"] = question_text

    def aliases_from_text(
        value: str,
    ) -> tuple[list[str], list[str], list[str], list[dict[str, str]]]:
        symbols: list[str] = []
        prose: list[str] = []
        decisions: list[str] = []
        quantified: list[dict[str, str]] = []
        terminal_targets: list[str] = []
        clauses: list[str] = []
        clauses.extend(match.group("body") for match in TARGET_CLAUSE.finditer(value))
        for match in QUESTION_TARGET_CLAUSE.finditer(value):
            decision = match.group("decision")
            body = decision or match.group("body") or ""
            if decision:
                body = re.sub(r"^(?:能否|是否|有无|有没有)", "", body)
                body = re.sub(r"^(?:有\s*\d+(?:\.\d+)?%\s*把握)?认为", "", body)
                normalized_decision = body.strip(" ，,：:？?")
                if (
                    1 <= len(normalized_decision) <= 60
                    and normalized_decision not in decisions
                ):
                    decisions.append(normalized_decision)
            else:
                terminal = re.split(
                    r"[，,：:；;]|(?:则|那么|故|请问)",
                    body,
                )[-1]
                terminal = re.sub(r"^(?:则|那么|故|请问|分别)", "", terminal)
                terminal = terminal.strip(" ，,：:？?")
                if 2 <= len(terminal) <= 32 and terminal not in terminal_targets:
                    terminal_targets.append(terminal)
            clauses.append(body)
        for body in clauses:
            for formula in TARGET_FORMULA.findall(body):
                alias = _target_formula_alias(formula)
                if alias and alias not in symbols:
                    symbols.append(alias)
            description = re.sub(r"\$[^$]+\$", "", body)
            description = re.sub(r"^(?:分别|认为|得到|得出)", "", description)
            description = description.strip(" ，,：:？?")
            if 2 <= len(description) <= 60 and description not in prose:
                prose.append(description)
            # Add a specific grammatical core so an answer such as
            # “一级品的频率为0.75” cannot evade an alias that also names
            # both machines.  Generic one-word targets remain disallowed.
            for core in re.findall(
                r"[一-鿿A-Za-z0-9]{1,18}的(?:频率|概率|期望|最大值|最小值|"
                r"取值范围|定义域|值域|解析式|表达式|解集|面积|"
                r"体积|长度|距离|斜率|坐标|个数|大小关系|二面角|夹角)",
                description,
            ):
                if core not in prose:
                    prose.append(core)
            for noun in TARGET_NOUN.findall(description):
                # Prefer the concrete object nearest the requested noun.  A
                # bare alias such as “频率” would incorrectly reject the
                # general definition “频率等于频数除以总数”.
                if description == noun and noun not in prose:
                    prose.append(noun)
                context_index = description.rfind(noun)
                prefix = description[:context_index]
                qualifier = re.search(r"([一-鿿A-Za-z0-9]{1,18})的\s*$", prefix)
                if qualifier:
                    raw_qualifier = qualifier.group(1)
                    short_qualifier = re.split(
                        r"(?:中|内|里|及|和|与)$", raw_qualifier
                    )[-1]
                    short_qualifier = re.split(
                        r"(?:中|内|里|及|和|与)", short_qualifier
                    )[-1]
                    grade_match = re.search(
                        r"([一二三四五六七八九十特优良次]级品)$",
                        raw_qualifier,
                    )
                    for base in dict.fromkeys((
                        raw_qualifier,
                        short_qualifier,
                        grade_match.group(1) if grade_match else "",
                    )):
                        if not base:
                            continue
                        for core in (f"{base}的{noun}", f"{base}{noun}"):
                            if 3 <= len(core) <= 24 and core not in prose:
                                prose.append(core)
            if "有差异" in description:
                for core in ("产品质量有差异", "两者有差异"):
                    if core not in prose:
                        prose.append(core)
        for target in terminal_targets:
            if target not in prose:
                prose.append(target)
        for match in QUANTIFIED_TARGET_CLAUSE.finditer(value):
            item = {
                "alias": match.group("alias"),
                "unit": match.group("unit") or "",
            }
            if item not in quantified:
                quantified.append(item)
        return symbols, prose, decisions, quantified

    requested: list[dict[str, Any]] = []
    for label in dict.fromkeys(labels):
        symbols, prose, decisions, quantified = aliases_from_text(
            part_text.get(label, "")
        )
        if symbols or prose or decisions or quantified:
            requested.append({
                "part": label,
                "symbol_aliases": symbols,
                "prose_aliases": prose,
                "decision_aliases": decisions,
                "quantified_aliases": quantified,
            })
    return requested


def validate_generic_answer_blind(output: Mapping[str, Any]) -> list[str]:
    """Reject unmistakable answer delivery even when no target can be parsed."""

    errors: list[str] = []
    requested = infer_dynamic_requested_outputs(output)
    prose_by_part = {
        pilot.normalized_part_label(item.get("part")): [
            alias for alias in item.get("prose_aliases", [])
            if isinstance(alias, str) and alias.strip()
        ]
        for item in requested
    }
    decisions_by_part = {
        pilot.normalized_part_label(item.get("part")): [
            alias for alias in item.get("decision_aliases", [])
            if isinstance(alias, str) and alias.strip()
        ]
        for item in requested
    }
    quantified_by_part = {
        pilot.normalized_part_label(item.get("part")): [
            alias
            for alias in item.get("quantified_aliases", [])
            if isinstance(alias, Mapping)
            and isinstance(alias.get("alias"), str)
            and alias.get("alias", "").strip()
        ]
        for item in requested
    }
    evidence_text = "\n".join(
        [
            str(block.get("text") or "")
            for block in output.get("question_blocks") or []
            if isinstance(block, Mapping) and block.get("type") == "text"
        ]
        + [
            str(item.get("fact") if isinstance(item, Mapping) else item)
            for item in output.get("visual_evidence") or []
        ]
    )
    known_numbers = {
        match.group(0).replace("％", "%")
        for match in re.finditer(
            r"(?<![0-9.])[+-]?\d+(?:\.\d+)?(?:%|％)?(?![0-9.])",
            evidence_text,
        )
    }
    numeric_assignment = re.compile(
        r"(?P<label>[一-鿿A-Za-z][一-鿿A-Za-z0-9]{0,20}?)"
        r"(?:的)?(?:值|频率|概率|比例|结果)?\s*"
        r"(?:分别|依次)?(?:为|等于|约为|=)\s*"
        r"(?P<number>[+-]?\d+(?:\.\d+)?(?:%|％)?)"
        r"(?=\s*(?:$|[，,、；;。和及与]))"
    )
    explicit_final_numeric = re.compile(
        r"(?:最终(?:答案|结果|数值)?|答案|计算结果|结果|数值)"
        r"(?:为|是|等于|约为)\s*"
        r"(?P<number>[+-]?\d+(?:\.\d+)?(?:%|％)?)|"
        r"(?:因此|所以|故)(?:可得|得到|得出|是|为)?\s*"
        r"(?P<conclusion_number>[+-]?\d+(?:\.\d+)?(?:%|％)?)"
    )
    standalone_numeric = re.compile(
        r"^(?:约为|为|是)?\s*(?P<number>[+-]?\d+(?:\.\d+)?(?:%|％)?)"
        r"\s*[一-鿿A-Za-z°]{0,4}$"
    )
    numeric_only_formula = re.compile(
        r"\$\s*(?P<number>[+-]?\d+(?:\.\d+)?(?:%|％)?)\s*"
        r"(?:\\,?\s*\\mathrm\{[^{}]{1,8}\})?\s*\$"
    )

    for path, value, current_part in pilot.guidance_entries(output):
        if PROOF_CLOSURE.search(value):
            errors.append(f"{path}完成了证明收尾；分层提示必须在得证前停止")
        if CHOICE_OR_FILL_ANSWER.search(value):
            errors.append(f"{path}直接给出了选择或填空答案")
        if EXPLICIT_FINAL_FORMULA.search(value):
            errors.append(f"{path}用结论性措辞直接给出了最终公式或数值")
        for target in quantified_by_part.get(current_part or "", []):
            alias = re.escape(str(target["alias"]).strip())
            unit = re.escape(str(target.get("unit") or "").strip())
            delivered = re.search(
                alias
                + r"[^，,。；;：:\n\d]{0,8}?"
                + r"(?P<number>[+-]?\d+(?:\.\d+)?(?:%|％)?)"
                + (rf"\s*{unit}" if unit else ""),
                value,
            )
            if delivered:
                number = delivered.group("number").replace("％", "%")
                if number not in known_numbers:
                    errors.append(
                        f"{path}直接给出了{current_part or '未编号问题'}的数量结果"
                        f"{delivered.group(0)}"
                    )
                    break
        numeric_text = numeric_only_formula.sub(
            lambda match: match.group("number"),
            value,
        )
        for match in explicit_final_numeric.finditer(numeric_text):
            number = (
                match.group("number") or match.group("conclusion_number") or ""
            ).replace("％", "%")
            if number and number not in known_numbers:
                errors.append(f"{path}用结论性措辞直接给出了最终数值{number}")
                break
        standalone_match = standalone_numeric.fullmatch(
            numeric_text.strip(" ，,。；;！!")
        )
        if standalone_match:
            number = standalone_match.group("number").replace("％", "%")
            if number not in known_numbers:
                errors.append(f"{path}直接给出了孤立的最终数值{number}")
        # Decision questions often have no symbolic target.  If a hint repeats
        # the exact proposition and closes it with an affirmative/negative
        # verdict, it is the requested answer rather than a solving hint.
        verdict_found = False
        for sentence in re.split(r"[。；;！!\n]+", value):
            normalized_value = re.sub(r"\s+", "", sentence)
            if not normalized_value:
                continue
            conditional_rule = bool(re.search(
                r"(?:若|如果|当|只要|一旦).{0,100}(?:则|就)",
                normalized_value,
            ))
            if conditional_rule and re.search(
                r"(?:实际|事实上|计算得|算得|最终|由此).{0,80}"
                r"(?:所以|因此|说明|可见|结论)",
                normalized_value,
            ):
                conditional_rule = False
            conclusion_marker = bool(re.search(
                r"(?:结论为|判定为|因此|所以|故|可见|说明)",
                normalized_value,
            ))
            open_instruction = bool(re.search(
                r"(?:是否|能否|有无|比较.{0,30}(?:大小|结果)|"
                r"检验(?:流程|方法|统计量)|依据.{0,30}(?:判定|判断)(?:是否|能否|有无))",
                normalized_value,
            )) and not conclusion_marker
            hypothesis_definition = bool(re.search(
                r"(?:明确|设定|提出|建立|令|设)(?:零假设|原假设|备择假设|H[_₀0]?)",
                normalized_value,
            )) and not conclusion_marker
            active_decisions = list(decisions_by_part.get(current_part or "", []))
            active_decisions.extend(
                alias
                for alias in prose_by_part.get(current_part or "", [])
                if re.search(
                    r"(?:差异|相关|独立|成立|存在|能否|是否|有无)",
                    alias,
                )
            )
            for alias in dict.fromkeys(active_decisions):
                normalized_alias = re.sub(r"\s+", "", alias)
                direct_confidence = bool(re.search(
                    r"(?:有|没有|无)\d+(?:\.\d+)?%把握认为",
                    normalized_value,
                ))
                variants = {normalized_alias}
                if normalized_alias.startswith("有"):
                    variants.add("没有" + normalized_alias[1:])
                    variants.add("无" + normalized_alias[1:])
                if normalized_alias.startswith("没有"):
                    variants.add("有" + normalized_alias[2:])
                elif normalized_alias.startswith("无"):
                    variants.add("有" + normalized_alias[1:])
                if normalized_alias.startswith("不能"):
                    variants.add("能" + normalized_alias[2:])
                elif normalized_alias.startswith("不"):
                    variants.add(normalized_alias[1:])
                if "有差异" in normalized_alias:
                    variants.add(normalized_alias.replace("有差异", "没有差异"))
                    variants.add(normalized_alias.replace("有差异", "无差异"))
                if normalized_alias == "正确":
                    variants.update(("不正确", "错误"))
                elif normalized_alias == "错误":
                    variants.update(("正确", "不错误"))
                else:
                    variants.add("不" + normalized_alias)
                    variants.add("能" + normalized_alias)
                    variants.add("不能" + normalized_alias)
                stripped_value = normalized_value.strip("。；;！!")
                bare_verdict = any(
                    stripped_value == variant or stripped_value.endswith(variant)
                    for variant in variants
                    if variant
                )
                conclusion_verdict = any(
                    stripped_value.endswith(variant) for variant in variants
                ) and conclusion_marker
                if (
                    (direct_confidence or bare_verdict or conclusion_verdict)
                    and not conditional_rule
                    and not open_instruction
                    and not hypothesis_definition
                ):
                    errors.append(
                        f"{path}直接给出了{current_part or '未编号问题'}的判定结论"
                    )
                    verdict_found = True
                    break
            if verdict_found:
                break

        # Catch a compact pair of newly derived terminal values such as
        # “甲机床为0.75，乙机床为0.60”.  Values already printed in the
        # question/table remain available for setting up an expression.
        unknown_assignments = [
            (match.group("label"), match.group("number").replace("％", "%"))
            for match in numeric_assignment.finditer(value)
            if match.group("number").replace("％", "%") not in known_numbers
            and not re.fullmatch(r"[A-Za-z]{1,3}", match.group("label"))
        ]
        if len(unknown_assignments) >= 2:
            errors.append(
                f"{path}直接给出了多项由题设计算出的数值结果："
                + "、".join(number for _, number in unknown_assignments[:3])
            )
    return errors


def validate_normalized_output(output: Mapping[str, Any]) -> list[str]:
    """Run every hard gate used immediately before publishing result images."""

    errors = pilot.validate_output(
        output,
        requested_outputs=None,
        device_policy=True,
    )
    requested_outputs = infer_dynamic_requested_outputs(output)
    if requested_outputs:
        pilot.validate_answer_guard(
            output,
            requested_outputs,
            errors,
            policy="no_answer",
        )
    errors.extend(validate_generic_answer_blind(output))
    errors.extend(validate_figure_reference_completeness(output))
    return list(dict.fromkeys(errors))


FIGURE_LABEL = r"(?:[0-9０-９]{1,2}(?![0-9０-９])|[一二三四五六七八九十]{1,3}|[甲乙丙丁戊己庚辛]|[A-Ha-h](?![A-Za-z]))"
NUMBERED_FIGURE_REFERENCE = re.compile(
    rf"图\s*[（(]?\s*(?P<label>{FIGURE_LABEL})\s*[）)]?"
    rf"(?P<continuations>(?:\s*[、,，/或和与及]\s*[（(]?\s*{FIGURE_LABEL}\s*[）)]?)*)"
)
CONTINUED_FIGURE_LABEL = re.compile(
    rf"[、,，/或和与及]\s*[（(]?\s*(?P<label>{FIGURE_LABEL})\s*[）)]?"
)
RANGED_FIGURE_REFERENCE = re.compile(
    rf"图\s*[（(]?\s*(?P<start>{FIGURE_LABEL})\s*[）)]?"
    rf"\s*(?:[～~—-]|至|到)\s*(?:图\s*)?[（(]?\s*(?P<end>{FIGURE_LABEL})\s*[）)]?"
)
UNNUMBERED_FIGURE_REFERENCE = re.compile(
    r"(?:如(?:左|右|上|下|左上|右上|左下|右下)?图(?:所示|中)|"
    r"见(?:左|右|上|下|左上|右上|左下|右下)?图|参见图|"
    r"根据(?:左|右|上|下|左上|右上|左下|右下)?图|"
    r"由(?:左|右|上|下|左上|右上|左下|右下)?图可知|"
    r"(?:左|右|上|下|左上|右上|左下|右下)图(?:所示|中))"
)
TABLE_REFERENCE = re.compile(
    r"(?:如(?:下|上|左|右)?表(?:所示|中)?|(?:下|上|左|右)表(?:所示|中)?|"
    r"根据(?:下|上)?表|由(?:下|上)?表可知|表中|附表|"
    r"统计表|列联表|数据表|临界值表)"
)
FIGURE_ID_LABEL = re.compile(
    rf"^(?:fig|figure)[_-]?0*(?P<label>{FIGURE_LABEL})$",
    re.IGNORECASE,
)


def _figure_label(value: str) -> str | None:
    token = value.strip().translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if token.isdigit():
        number = int(token)
        return str(number) if 0 < number <= 99 else None
    if len(token) == 1 and token.upper() in "ABCDEFGH":
        return token.upper()
    if token in "甲乙丙丁戊己庚辛":
        return token
    digits = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if token in digits:
        return str(digits[token])
    if token.startswith("十") and len(token) == 2 and token[1] in digits:
        return str(10 + digits[token[1]])
    if token.endswith("十") and len(token) == 2 and token[0] in digits:
        return str(digits[token[0]] * 10)
    if len(token) == 3 and token[1] == "十" and token[0] in digits and token[2] in digits:
        return str(digits[token[0]] * 10 + digits[token[2]])
    return None


def _numbered_figure_references(value: str) -> set[str]:
    value = value.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    references: set[str] = set()
    for match in NUMBERED_FIGURE_REFERENCE.finditer(value):
        labels = [match.group("label")]
        labels.extend(
            item.group("label")
            for item in CONTINUED_FIGURE_LABEL.finditer(match.group("continuations") or "")
        )
        for raw_label in labels:
            label = _figure_label(raw_label)
            if label is not None:
                references.add(label)
    for match in RANGED_FIGURE_REFERENCE.finditer(value):
        start = _figure_label(match.group("start"))
        end = _figure_label(match.group("end"))
        if start is None or end is None:
            continue
        if start.isdigit() and end.isdigit():
            left, right = int(start), int(end)
            if 0 <= right - left <= 11:
                references.update(str(item) for item in range(left, right + 1))
        elif start in "甲乙丙丁戊己庚辛" and end in "甲乙丙丁戊己庚辛":
            labels = "甲乙丙丁戊己庚辛"
            left, right = labels.index(start), labels.index(end)
            if left <= right:
                references.update(labels[left : right + 1])
        elif start in "ABCDEFGH" and end in "ABCDEFGH":
            left, right = ord(start), ord(end)
            if left <= right:
                references.update(chr(item) for item in range(left, right + 1))
    return references


def _figure_label_family(label: str) -> str:
    if label.isdigit():
        return "digit"
    if label in "甲乙丙丁戊己庚辛":
        return "chinese_panel"
    return "latin_panel"


def validate_figure_reference_completeness(
    output: Mapping[str, Any],
) -> list[str]:
    """Reject a knowledge card that omits a figure explicitly cited by the stem.

    This intentionally checks metadata, not pixels.  A missing or guessed crop is
    sent back to the vision model for one repair attempt; if it remains missing,
    the job fails instead of publishing an incomplete first result image.
    """

    blocks = output.get("question_blocks")
    question_text = "\n".join(
        str(block.get("text") or "")
        for block in blocks or []
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    required = _numbered_figure_references(question_text)
    unnumbered_required = bool(UNNUMBERED_FIGURE_REFERENCE.search(question_text))
    visual_evidence = output.get("visual_evidence")
    evidence_text = "\n".join(
        str(item) for item in visual_evidence or [] if isinstance(item, str)
    )
    table_required = bool(
        TABLE_REFERENCE.search(question_text) or TABLE_REFERENCE.search(evidence_text)
    )
    figures = output.get("source_figures")
    source_figures = figures if isinstance(figures, list) else []
    if not required and not unnumbered_required and not table_required:
        return []
    if not source_figures:
        if required:
            names = "、".join(f"图{item}" for item in sorted(required))
            return [f"题干明确引用{names}，但source_figures为空，第一张原题知识图不完整"]
        if table_required:
            return ["题干或图表证据明确引用表格，但source_figures为空，第一张原题知识图不完整"]
        return ["题干明确写有‘如图所示’，但source_figures为空，第一张原题知识图不完整"]

    represented: set[str] = set()
    conflicts: list[str] = []
    for figure in source_figures:
        if not isinstance(figure, Mapping):
            continue
        figure_id = str(figure.get("figure_id") or "").strip()
        id_match = FIGURE_ID_LABEL.fullmatch(figure_id)
        id_label = _figure_label(id_match.group("label")) if id_match else None
        caption_labels = _numbered_figure_references(str(figure.get("caption") or ""))
        # Numeric fig_N IDs are often sequential schema identifiers while the
        # printed paper labels panels as 甲/乙 or A/B.  Compare only like label
        # systems; the caption then remains the authoritative printed label.
        comparable_id = (
            id_label is not None
            and caption_labels
            and all(
                _figure_label_family(item) == _figure_label_family(id_label)
                for item in caption_labels
            )
        )
        if comparable_id and id_label not in caption_labels:
            conflicts.append(
                f"{figure_id}与caption图号{'、'.join(sorted(caption_labels))}冲突"
            )
        represented.update(caption_labels or ({id_label} if id_label is not None else set()))
    if conflicts:
        return [
            "source_figures的figure_id/caption图号冲突："
            + "；".join(conflicts)
            + "；必须回看原图修正真实bbox和图号"
        ]
    missing = sorted(required - represented)
    if missing:
        names = "、".join(f"图{item}" for item in missing)
        return [
            f"题干引用的{names}未出现在source_figures/figure块中，第一张原题知识图不完整；"
            "必须根据原图补齐真实bbox，禁止猜测"
        ]
    return []


CROSS_TIER_METHOD_IDENTITY_ERROR = (
    "同一小问在三个档位中的方法编号、名称和顺序必须完全一致"
)
METHOD_NAME_GENERIC_SUFFIX = re.compile(
    r"(?:方法|解法|应用|运用|求解|解题|思路|策略|推导|计算|法)+$"
)
METHOD_NAME_CLEAN = re.compile(r"[^A-Za-z0-9\u3400-\u9fff]+")


def _method_objects(part: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    if not isinstance(part, dict):
        return None
    alternatives = part.get("alternative_methods")
    if not isinstance(alternatives, list) or not all(
        isinstance(item, dict) for item in alternatives
    ):
        return None
    return [part, *alternatives]


def _method_name_core(value: str) -> str:
    core = METHOD_NAME_CLEAN.sub("", value).lower()
    previous = None
    while core and core != previous:
        previous = core
        core = METHOD_NAME_GENERIC_SUFFIX.sub("", core)
    return core


def _method_names_are_near_synonyms(names: list[str]) -> bool:
    """Conservatively recognize three cosmetic names for one method.

    A shared two-character mathematical phrase (for example ``导数`` or
    ``平行``) is required after generic suffixes are removed.  The local
    repair therefore cannot merge unrelated methods merely because all three
    happened to be labelled ``M1``.
    """

    cores = [_method_name_core(name) for name in names]
    if not all(cores):
        return False
    if len(set(cores)) == 1:
        return True
    features = [
        {core[index : index + 2] for index in range(len(core) - 1)}
        if len(core) > 1
        else {core}
        for core in cores
    ]
    shared = set.intersection(*features)
    # A single subject-wide word is too weak ("函数求导" and "函数图像"
    # are different methods), while one method-specific mathematical phrase is
    # sufficient once part order, method ids/counts, and substantive steps have
    # already matched.  This covers real cosmetic labels such as
    # "平行导角法 / 平行线找角法 / 平行线性质应用" without merging unrelated
    # methods merely because all three happened to be called M1.
    generic = {
        "函数",
        "方程",
        "代数",
        "几何",
        "计算",
        "求解",
        "方法",
        "解法",
        "应用",
        "性质",
    }
    return bool(shared - generic)


def normalize_cross_tier_method_names(
    output: Mapping[str, Any],
    errors: list[str],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Unify only structurally identical, substantive, near-synonym methods."""

    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    if errors != [CROSS_TIER_METHOD_IDENTITY_ERROR]:
        return candidate
    guidance = candidate.get("guidance")
    if not isinstance(guidance, dict):
        return candidate
    tier_keys = ("advanced", "standard", "foundation")
    parts_by_tier: dict[str, list[dict[str, Any]]] = {}
    for tier_key in tier_keys:
        tier = guidance.get(tier_key)
        parts = tier.get("parts") if isinstance(tier, dict) else None
        if not isinstance(parts, list) or not all(
            isinstance(part, dict) for part in parts
        ):
            return candidate
        parts_by_tier[tier_key] = parts
    part_counts = {len(parts) for parts in parts_by_tier.values()}
    if len(part_counts) != 1:
        return candidate
    part_sequences = [
        [pilot.normalized_part_label(part.get("part")) for part in parts_by_tier[key]]
        for key in tier_keys
    ]
    if not part_sequences[0] or any(
        sequence != part_sequences[0] for sequence in part_sequences[1:]
    ):
        return candidate

    pending: list[tuple[int, int, str, list[str]]] = []
    for part_index in range(len(parts_by_tier["advanced"])):
        tier_methods = [
            _method_objects(parts_by_tier[tier_key][part_index])
            for tier_key in tier_keys
        ]
        if any(methods is None for methods in tier_methods):
            return candidate
        complete_methods = [methods for methods in tier_methods if methods is not None]
        method_counts = {len(methods) for methods in complete_methods}
        if len(method_counts) != 1:
            return candidate
        method_id_sequences = [
            [str(method.get("method_id") or "").strip() for method in methods]
            for methods in complete_methods
        ]
        if not method_id_sequences[0] or any(
            sequence != method_id_sequences[0]
            for sequence in method_id_sequences[1:]
        ):
            return candidate
        for methods in complete_methods:
            for method in methods:
                steps = method.get("steps")
                if not isinstance(steps, list) or not steps or not all(
                    pilot.has_substantive_guidance_content(step) for step in steps
                ):
                    return candidate
        for method_index in range(len(complete_methods[0])):
            names = [
                str(methods[method_index].get("method_name") or "").strip()
                for methods in complete_methods
            ]
            if not all(names):
                return candidate
            if len(set(names)) == 1:
                continue
            if not _method_names_are_near_synonyms(names):
                return candidate
            pending.append((part_index, method_index, names[0], names))

    if not pending:
        return candidate
    for part_index, method_index, canonical_name, previous_names in pending:
        for tier_key in tier_keys[1:]:
            methods = _method_objects(parts_by_tier[tier_key][part_index])
            assert methods is not None
            methods[method_index]["method_name"] = canonical_name
        advanced_methods = _method_objects(parts_by_tier["advanced"][part_index])
        assert advanced_methods is not None
        audit.append(
            {
                "action": "normalize_cross_tier_method_name",
                "part": part_sequences[0][part_index],
                "method_id": str(advanced_methods[method_index].get("method_id")),
                "from": dict(zip(tier_keys, previous_names)),
                "to": canonical_name,
            }
        )
    return candidate


def _guidance_step_quality(output: Mapping[str, Any]) -> tuple[int, int, int]:
    total = 0
    substantive = 0
    alphanumeric_units = 0
    guidance = output.get("guidance")
    if not isinstance(guidance, Mapping):
        return total, substantive, alphanumeric_units
    for tier_key in ("advanced", "standard", "foundation"):
        tier = guidance.get(tier_key)
        parts = tier.get("parts") if isinstance(tier, Mapping) else None
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            methods: list[Mapping[str, Any]] = [part]
            alternatives = part.get("alternative_methods")
            if isinstance(alternatives, list):
                methods.extend(
                    item for item in alternatives if isinstance(item, Mapping)
                )
            for method in methods:
                steps = method.get("steps")
                if not isinstance(steps, list):
                    continue
                for step in steps:
                    total += 1
                    if pilot.has_substantive_guidance_content(step):
                        substantive += 1
                        alphanumeric_units += sum(
                            character.isalnum() for character in str(step)
                        )
    return total, substantive, alphanumeric_units


def _question_alphanumeric_units(output: Mapping[str, Any]) -> int:
    blocks = output.get("question_blocks")
    return sum(
        character.isalnum()
        for block in blocks or []
        if isinstance(block, Mapping) and block.get("type") == "text"
        for character in str(block.get("text") or "")
    )


def _list_item_count(output: Mapping[str, Any], key: str) -> int:
    value = output.get(key)
    return len(value) if isinstance(value, list) else 0


def _tier_part_counts(output: Mapping[str, Any]) -> dict[str, int]:
    guidance = output.get("guidance")
    counts: dict[str, int] = {}
    for tier_key in ("advanced", "standard", "foundation"):
        tier = guidance.get(tier_key) if isinstance(guidance, Mapping) else None
        parts = tier.get("parts") if isinstance(tier, Mapping) else None
        counts[tier_key] = len(parts) if isinstance(parts, list) else 0
    return counts


def repair_candidate_rejection_reasons(
    before: Mapping[str, Any],
    before_errors: list[str],
    candidate: Mapping[str, Any],
    candidate_errors: list[str],
) -> list[str]:
    """Explain why a model repair is not a strict, content-preserving gain."""

    reasons: list[str] = []
    if len(candidate_errors) >= len(before_errors):
        reasons.append("修复后的强校验错误数没有严格下降")
    _, before_substantive, before_units = _guidance_step_quality(before)
    candidate_total, candidate_substantive, candidate_units = _guidance_step_quality(
        candidate
    )
    if candidate_total == 0 or candidate_substantive != candidate_total:
        reasons.append("修复结果仍含空步骤或只有标点/符号的步骤")
    if candidate_substantive < before_substantive:
        reasons.append("修复结果删减了原有实质步骤")
    if candidate_units < before_units:
        reasons.append("修复结果的实质步骤内容量下降")
    if _question_alphanumeric_units(candidate) < _question_alphanumeric_units(before):
        reasons.append("修复结果缩减了原题文字")
    for key, label in (
        ("source_figures", "原题图形"),
        ("knowledge_points", "知识点"),
        ("base_formulas", "基础公式"),
    ):
        if _list_item_count(candidate, key) < _list_item_count(before, key):
            reasons.append(f"修复结果删减了{label}")
    before_parts = _tier_part_counts(before)
    candidate_parts = _tier_part_counts(candidate)
    for tier_key, count in before_parts.items():
        if candidate_parts[tier_key] < count:
            reasons.append(f"修复结果删减了{tier_key}档小问")
    return reasons


def repair_candidate_strictly_improves(
    before: Mapping[str, Any],
    before_errors: list[str],
    candidate: Mapping[str, Any],
    candidate_errors: list[str],
) -> bool:
    return not repair_candidate_rejection_reasons(
        before,
        before_errors,
        candidate,
        candidate_errors,
    )


def validation_errors_are_guidance_only(errors: list[str]) -> bool:
    """Return whether a provider retry only needs to replace tier guidance."""

    return bool(errors) and all(
        item.startswith("guidance.")
        or "同一小问在三个档位中的方法编号、名称和顺序" in item
        for item in errors
    )


def merge_guidance_only_repair(
    before: Mapping[str, Any],
    before_errors: list[str],
    repaired: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]:
    """Keep trusted recognition fields while taking regenerated tier guidance.

    A repair request may correctly regenerate all steps yet re-transcribe fewer
    question characters or figures.  Rejecting that response made the camera
    workflow fail even though only ``guidance`` was broken.  When every original
    error is guidance-scoped, merge only that subtree and run the complete
    validator again; no model-written recognition field is allowed to replace
    the already validated source evidence.
    """

    guidance = repaired.get("guidance")
    if not validation_errors_are_guidance_only(before_errors) or not isinstance(
        guidance, Mapping
    ):
        return None, [], []
    candidate = json.loads(json.dumps(before, ensure_ascii=False))
    candidate["guidance"] = json.loads(json.dumps(guidance, ensure_ascii=False))
    normalized, errors = parse_model_output(json.dumps(candidate, ensure_ascii=False))
    audit: list[dict[str, Any]] = [
        {
            "action": "merge_guidance_only_provider_repair",
            "preserved_fields": [
                key for key in candidate if key != "guidance"
            ],
        }
    ]
    if normalized is not None and errors:
        normalized, errors, local_audit = apply_dynamic_local_repairs(
            normalized, errors
        )
        audit.extend(local_audit)
    return normalized, errors, audit


def apply_dynamic_local_repairs(
    parsed: Mapping[str, Any],
    errors: list[str],
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """Apply only deterministic mathematics repairs, then revalidate strictly."""

    audit: list[dict[str, Any]] = []
    candidate = trim_adjacent_question_fragments(parsed, audit)
    candidate = normalize_tier_part_labels(candidate, audit)
    candidate = normalize_formula_notation(candidate, audit)
    candidate = pilot.redact_answer_guard_violations(candidate, errors, audit)
    candidate = align_question_figure_blocks(candidate, errors, audit)
    normalized, remaining = parse_model_output(json.dumps(candidate, ensure_ascii=False))
    if normalized is not None:
        method_audit: list[dict[str, Any]] = []
        method_candidate = normalize_cross_tier_method_names(
            normalized,
            remaining,
            method_audit,
        )
        if method_audit:
            reparsed, method_remaining = parse_model_output(
                json.dumps(method_candidate, ensure_ascii=False)
            )
            if reparsed is not None and len(method_remaining) < len(remaining):
                normalized = reparsed
                remaining = method_remaining
                audit.extend(method_audit)
    if normalized is not None and len(remaining) <= len(errors):
        return normalized, remaining, audit
    return dict(parsed), list(errors), []


def align_question_figure_blocks(
    output: Mapping[str, Any],
    errors: list[str],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Insert missing figure placeholders without inventing figure crops."""

    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    if not any(
        "question_blocks与source_figures的figure_id及顺序必须完全一致" in item
        for item in errors
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
        blocks.insert(
            insert_at,
            {"type": "figure", "text": "", "figure_id": figure_id},
        )
        observed.append(figure_id)
        audit.append({"action": "insert_missing_figure_block", "figure_id": figure_id})
    return candidate


def parse_model_output_for_repair(
    text: str,
) -> tuple[dict[str, Any] | None, list[str], list[dict[str, Any]]]:
    """Parse once, retain deterministic repairs, and expose all remaining gates."""

    parsed, errors = parse_model_output(text)
    audit: list[dict[str, Any]] = []
    if parsed is not None and errors:
        parsed, errors, audit = apply_dynamic_local_repairs(parsed, errors)
    return parsed, errors, audit


PLAIN_SQRT = re.compile(r"(?<![A-Za-z\\])sqrt\(([^()]+)\)")


def normalize_formula_notation(
    output: Mapping[str, Any],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert unambiguous plain ``sqrt(x)`` inside math spans to LaTeX."""

    candidate = json.loads(json.dumps(output, ensure_ascii=False))

    def visit(value: Any, path: str) -> Any:
        if isinstance(value, dict):
            return {key: visit(child, f"{path}.{key}") for key, child in value.items()}
        if isinstance(value, list):
            return [visit(child, f"{path}[{index}]") for index, child in enumerate(value)]
        if not isinstance(value, str) or "sqrt(" not in value:
            return value
        chunks = value.split("$")
        changed = False
        for index in range(1, len(chunks), 2):
            normalized, count = PLAIN_SQRT.subn(r"\\sqrt{\1}", chunks[index])
            if count:
                chunks[index] = normalized
                changed = True
        if changed:
            result = "$".join(chunks)
            audit.append(
                {
                    "action": "normalize_plain_sqrt",
                    "path": path,
                }
            )
            return result
        return value

    return visit(candidate, "$")


PART_LABEL_PREFIX = re.compile(
    r"^\s*[（(]\s*(\d+)\s*[）)]\s*([①②③④⑤⑥⑦⑧⑨⑩])?"
)


def normalize_tier_part_labels(
    output: Mapping[str, Any],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reduce descriptive model labels like ``（1）求速度`` to ``（1）``."""

    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    guidance = candidate.get("guidance")
    if not isinstance(guidance, dict):
        return candidate
    for tier_key in ("advanced", "standard", "foundation"):
        tier = guidance.get(tier_key)
        parts = tier.get("parts") if isinstance(tier, dict) else None
        if not isinstance(parts, list):
            continue
        for item in parts:
            if not isinstance(item, dict):
                continue
            raw = str(item.get("part") or "")
            match = PART_LABEL_PREFIX.match(raw)
            if not match:
                continue
            normalized = f"（{match.group(1)}）{match.group(2) or ''}"
            if raw.strip() != normalized:
                item["part"] = normalized
                audit.append(
                    {
                        "action": "normalize_tier_part_label",
                        "tier": tier_key,
                        "from": raw,
                        "to": normalized,
                    }
                )
    return candidate


MAIN_QUESTION_HEADER = re.compile(
    r"(?m)^\s*(?:题\s*)?\d{1,3}\s*[.．、]\s*"
)


def trim_adjacent_question_fragments(
    output: Mapping[str, Any],
    audit: list[dict[str, Any]],
) -> dict[str, Any]:
    """Keep the first complete numbered problem, dropping edge fragments.

    Camera frames often retain the previous problem's final ``（3）`` above
    the actual numbered problem. The model sometimes transcribes that fragment
    despite the prompt. This operates on the model's ordered question blocks,
    never on pixels, and only triggers when a complete main number header is
    visible.
    """

    candidate = json.loads(json.dumps(output, ensure_ascii=False))
    blocks = candidate.get("question_blocks")
    if not isinstance(blocks, list):
        return candidate
    header_locations: list[tuple[int, int]] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = str(block.get("text") or "")
        for match in MAIN_QUESTION_HEADER.finditer(text):
            header_locations.append((index, match.start()))
    if not header_locations:
        return candidate
    start_index, start_offset = header_locations[0]
    end_index: int | None = None
    end_offset: int | None = None
    if len(header_locations) > 1:
        end_index, end_offset = header_locations[1]
    kept = json.loads(json.dumps(blocks[start_index:], ensure_ascii=False))
    if kept and start_offset:
        kept[0]["text"] = str(kept[0].get("text") or "")[start_offset:].lstrip()
    if end_index is not None:
        relative_end = end_index - start_index
        if relative_end < len(kept):
            if end_offset and kept[relative_end].get("type") == "text":
                prefix = str(kept[relative_end].get("text") or "")[:end_offset].rstrip()
                kept = kept[:relative_end] + (
                    [{**kept[relative_end], "text": prefix}] if prefix else []
                )
            else:
                kept = kept[:relative_end]
    if kept != blocks:
        candidate["question_blocks"] = kept
        referenced = {
            str(block.get("figure_id") or "")
            for block in kept
            if isinstance(block, Mapping) and block.get("type") == "figure"
        }
        figures = candidate.get("source_figures")
        if isinstance(figures, list):
            candidate["source_figures"] = [
                item
                for item in figures
                if isinstance(item, Mapping)
                and str(item.get("figure_id") or "") in referenced
            ]
        audit.append(
            {
                "action": "trim_adjacent_question_fragments",
                "removed_leading_blocks": start_index,
                "stopped_at_next_main_header": end_index is not None,
            }
        )
    return candidate


def dynamic_repair_prompt(
    parsed: Mapping[str, Any],
    errors: list[str],
    base_prompt: str,
) -> str:
    """Augment the original repair prompt with exact tier coverage evidence."""

    prompt = pilot.validation_repair_prompt(base_prompt, parsed, errors)
    blocks = parsed.get("question_blocks")
    question_text = "\n".join(
        str(block.get("text") or "")
        for block in blocks or []
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    expected = pilot.printed_leaf_part_labels(question_text)
    guidance = parsed.get("guidance")
    observed: dict[str, list[str]] = {}
    if isinstance(guidance, Mapping):
        for tier_key in ("advanced", "standard", "foundation"):
            tier = guidance.get(tier_key)
            parts = tier.get("parts") if isinstance(tier, Mapping) else None
            observed[tier_key] = [
                str(item.get("part") or "")
                for item in parts or []
                if isinstance(item, Mapping)
            ]
    if expected:
        prompt += (
            "\n\n本题从question_blocks检测到的必需小问编号依次为："
            + json.dumps(expected, ensure_ascii=False)
            + "。advanced、standard、foundation三档parts必须各自逐项使用这一完整编号列表，"
            + "不得漏项、合并、增删或改变顺序。上次三档编号为："
            + json.dumps(observed, ensure_ascii=False)
            + "。"
        )
    if any(
        "第一张原题知识图不完整" in item
        or "figure_id/caption图号冲突" in item
        for item in errors
    ):
        prompt += (
            "\n\n完整性强制修复：question_blocks题干明确引用的每一幅编号图都必须在"
            "source_figures中有从本次原图重新核对得到的真实bbox_1000，并在question_blocks"
            "相应阅读位置有同figure_id的figure块。caption须明确写出印刷图号。"
            "不要复制现有错误框、不要编造不可见图；若确实看不清，仍要完整框住可见图并将"
            "recognition_status设为uncertain。"
        )
    return prompt


class SearchMachinePipeline:
    """Single-device mathematics pipeline using the shared photo preprocessor."""

    def __init__(
        self,
        runs_root: Path = RUNS_ROOT,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        timeout_seconds: float = 300.0,
        max_tokens: int | None = 7500,
        processor: QuestionImageProcessor | None = None,
        api_attempt=benchmark.perform_http_attempt,
    ):
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.processor = processor or QuestionImageProcessor(
            self.runs_root / "_preprocess",
            enable_reference_library=False,
        )
        self.api_attempt = api_attempt
        self._solve_lock = threading.Lock()
        self.schema = load_json(DEFAULT_SCHEMA)
        self.prompt = pilot.extract_model_prompt(DEFAULT_PROMPT)

    def new_job(self, data: bytes, filename: str) -> dict[str, Any]:
        if not data:
            raise ValueError("上传图片为空")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("图片不能超过20 MB")
        # Decode once before persisting so arbitrary HTML/SVG cannot become a job.
        try:
            with Image.open(__import__("io").BytesIO(data)) as image:
                image_format = str(image.format or "").upper()
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise ValueError("图片无法识别，请上传JPEG、PNG或WebP照片") from exc
        if image_format not in ACCEPTED_IMAGE_FORMATS:
            raise ValueError("仅支持JPEG、PNG或WebP照片")
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError("图片像素过大，请控制在5000万像素以内")
        job_id = f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex}"
        job_dir = self.runs_root / job_id
        job_dir.mkdir(parents=False, exist_ok=False)
        input_name = safe_filename(filename)
        audit_dir = job_dir / "audit"
        audit_dir.mkdir()
        original_path = audit_dir / "original.upload"
        original_path.write_bytes(data)
        state = {
            "job_id": job_id,
            "status": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "original_filename": input_name,
            "original_path": str(original_path.relative_to(job_dir)).replace("\\", "/"),
            "images": [],
            "error": None,
        }
        atomic_json(job_dir / "job.json", state)
        return state

    def job_path(self, job_id: str) -> Path:
        # Accept legacy 64-bit IDs for already completed local verification
        # jobs; every newly created job uses a full 128-bit UUID.
        if not re.fullmatch(
            r"[0-9]{8}T[0-9]{6}_(?:[0-9a-f]{16}|[0-9a-f]{32})",
            job_id,
        ):
            raise ValueError("任务ID无效")
        path = self.runs_root / job_id
        if not path.is_dir():
            raise FileNotFoundError(job_id)
        return path

    def read_job(self, job_id: str) -> dict[str, Any]:
        return load_json(self.job_path(job_id) / "job.json")

    def _write_state(self, job_dir: Path, state: dict[str, Any], status: str) -> None:
        state["status"] = status
        state["updated_at"] = utc_now()
        atomic_json(job_dir / "job.json", state)

    def solve(
        self,
        job_id: str,
        *,
        use_model: bool = True,
        mock_output: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._solve_lock:
            job_dir = self.job_path(job_id)
            state = load_json(job_dir / "job.json")
            started = time.perf_counter()
            try:
                self._write_state(job_dir, state, "preprocessing")
                original = job_dir / str(state["original_path"])
                preprocess = self.processor.process(
                    original.read_bytes(),
                    str(state["original_filename"]),
                    mode="gray",
                    enable_ocr=False,
                    handwriting_mode="none",
                    rotation="auto",
                    quality="fast",
                    extract_questions=False,
                    deskew_mode="safe",
                )
                raw_processed = self.processor.output_root / preprocess.png
                # Keep the exact preprocessed PNG as both model input and the
                # mathematics renderer source. It lives in a private audit
                # subdirectory and is not part of the four displayed results.
                source_path = job_dir / "audit" / "preprocessed_source.png"
                source_path.write_bytes(raw_processed.read_bytes())
                data_url, request_image = png_data_url(source_path)
                state["preprocess"] = preprocess.to_dict()
                state["preprocess"]["render_source_path"] = str(
                    source_path.relative_to(job_dir)
                ).replace("\\", "/")
                state["request_image"] = request_image
                for relative in (preprocess.png, preprocess.pdf, preprocess.markdown):
                    temporary = (self.processor.output_root / relative).resolve()
                    try:
                        temporary.relative_to(self.processor.output_root.resolve())
                        temporary.unlink(missing_ok=True)
                    except (ValueError, OSError):
                        pass
                self._write_state(job_dir, state, "solving")

                if mock_output is not None:
                    parsed, validation_errors = parse_model_output(
                        json.dumps(mock_output, ensure_ascii=False)
                    )
                    if parsed is None or validation_errors:
                        raise ValueError(
                            "模拟模型结果未通过数学模块校验：" + "；".join(validation_errors[:12])
                        )
                    attempt: Mapping[str, Any] = {
                        "response_model": "mock",
                        "usage": {},
                        "provider_request_id": None,
                        "http_status": 200,
                    }
                elif use_model:
                    api_key, base_url, workspace = load_credentials()
                    payload = build_request_payload(
                        model_id=self.model_id,
                        prompt=self.prompt,
                        data_url=data_url,
                        schema=self.schema,
                        max_tokens=self.max_tokens,
                    )
                    attempt = self.api_attempt(
                        payload,
                        base_url,
                        api_key,
                        workspace,
                        self.timeout_seconds,
                        [api_key],
                        ROOT_KEYS,
                        False,
                    )
                    if attempt.get("transport_success") is not True:
                        raise RuntimeError(str(attempt.get("error") or "模型请求失败"))
                    provider_attempts: list[Mapping[str, Any]] = [attempt]
                    parsed, validation_errors = parse_model_output(
                        str(attempt.get("output_text") or "")
                    )
                    repair_audit: list[dict[str, Any]] = []
                    if parsed is not None and validation_errors:
                        parsed, validation_errors, local_audit = (
                            apply_dynamic_local_repairs(parsed, validation_errors)
                        )
                        repair_audit.extend(local_audit)
                    if parsed is not None and validation_errors:
                        repair_prompt = dynamic_repair_prompt(
                            parsed,
                            validation_errors,
                            self.prompt,
                        )
                        repair_payload = build_request_payload(
                            model_id=self.model_id,
                            prompt=repair_prompt,
                            data_url=data_url,
                            schema=self.schema,
                            max_tokens=self.max_tokens,
                        )
                        repair_attempt = self.api_attempt(
                            repair_payload,
                            base_url,
                            api_key,
                            workspace,
                            self.timeout_seconds,
                            [api_key],
                            ROOT_KEYS,
                            False,
                        )
                        provider_attempts.append(repair_attempt)
                        repair_entry: dict[str, Any] = {
                            "transport_success": bool(
                                repair_attempt.get("transport_success")
                            ),
                            "http_status": repair_attempt.get("http_status"),
                            "provider_request_id": repair_attempt.get(
                                "provider_request_id"
                            ),
                        }
                        if repair_attempt.get("transport_success") is True:
                            repaired, repair_errors = parse_model_output(
                                str(repair_attempt.get("output_text") or "")
                            )
                            candidate_local_audit: list[dict[str, Any]] = []
                            if repaired is not None and repair_errors:
                                repaired, repair_errors, local_audit = (
                                    apply_dynamic_local_repairs(
                                        repaired,
                                        repair_errors,
                                    )
                                )
                                candidate_local_audit.extend(local_audit)
                            if repaired is not None:
                                merged, merged_errors, merged_audit = (
                                    merge_guidance_only_repair(
                                        parsed,
                                        validation_errors,
                                        repaired,
                                    )
                                )
                                if merged is not None:
                                    repaired = merged
                                    repair_errors = merged_errors
                                    candidate_local_audit.extend(merged_audit)
                            repair_entry["validation_errors"] = repair_errors[:30]
                            if repaired is not None:
                                repair_entry["guidance_step_quality"] = list(
                                    _guidance_step_quality(repaired)
                                )
                            rejection_reasons = (
                                repair_candidate_rejection_reasons(
                                    parsed,
                                    validation_errors,
                                    repaired,
                                    repair_errors,
                                )
                                if repaired is not None
                                else ["修复响应不是有效数学JSON对象"]
                            )
                            repair_entry["accepted"] = not rejection_reasons
                            if rejection_reasons:
                                repair_entry["rejection_reasons"] = rejection_reasons
                            if repaired is not None and not rejection_reasons:
                                repair_audit.extend(candidate_local_audit)
                                parsed = repaired
                                validation_errors = repair_errors
                                attempt = repair_attempt
                        else:
                            repair_entry["error"] = repair_attempt.get("error")
                        repair_audit.append(repair_entry)
                    if parsed is None or validation_errors:
                        if parsed is not None:
                            guidance = parsed.get("guidance")
                            observed_parts: dict[str, list[str]] = {}
                            if isinstance(guidance, Mapping):
                                for tier_key in (
                                    "advanced",
                                    "standard",
                                    "foundation",
                                ):
                                    tier = guidance.get(tier_key)
                                    parts = (
                                        tier.get("parts")
                                        if isinstance(tier, Mapping)
                                        else None
                                    )
                                    observed_parts[tier_key] = [
                                        str(item.get("part") or "")
                                        for item in parts or []
                                        if isinstance(item, Mapping)
                                    ]
                            blocks = parsed.get("question_blocks")
                            question_text = "\n".join(
                                str(block.get("text") or "")
                                for block in blocks or []
                                if isinstance(block, Mapping)
                                and block.get("type") == "text"
                            )
                            atomic_json(
                                job_dir / "audit" / "model_validation_failure.json",
                                {
                                    "validation_errors": validation_errors,
                                    "repair_audit": repair_audit,
                                    "printed_leaf_parts": pilot.printed_leaf_part_labels(
                                        question_text
                                    ),
                                    "observed_tier_parts": observed_parts,
                                    "model_output": parsed,
                                },
                            )
                        raise ValueError(
                            "模型结果未通过数学模块强校验："
                            + "；".join(str(item) for item in validation_errors[:12])
                        )
                else:
                    raise ValueError("未提供模型或模拟输出")

                publish_errors = validate_normalized_output(parsed)
                if publish_errors:
                    raise ValueError(
                        "渲染前数学模块强校验失败："
                        + "；".join(publish_errors)
                    )
                self._write_state(job_dir, state, "rendering")
                song, song_bold, times, times_bold = resolve_device_fonts()
                render = render_four_images(
                    question_id=job_id,
                    model_id=str(attempt.get("response_model") or self.model_id),
                    source_image=source_path,
                    output=parsed,
                    output_dir=job_dir,
                    song_path=song,
                    song_bold_path=song_bold,
                    times_path=times,
                    times_bold_path=times_bold,
                )
                contract = validate_device_image_set(job_dir, require_all=True)
                if not contract["valid"]:
                    raise ValueError(
                        "结果图不符合数学模块图片合同：" + "；".join(contract["errors"])
                    )
                # Model and renderer used the same lossless preprocessed pixels.
                state.update(
                    {
                        "model_id_requested": self.model_id,
                        "model_id_response": attempt.get("response_model"),
                        "provider_request_id": attempt.get("provider_request_id"),
                        "http_status": attempt.get("http_status"),
                        "usage": (
                            merge_attempt_usage(*provider_attempts)
                            if use_model and mock_output is None
                            else attempt.get("usage") or {}
                        ),
                        "output_json": parsed,
                        "validation": {
                            "valid": True,
                            "errors": [],
                            "repair_audit": (
                                repair_audit
                                if use_model and mock_output is None
                                else []
                            ),
                        },
                        "render": render,
                        "image_contract": contract,
                        "images": contract["files"],
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "error": None,
                    }
                )
                self._write_state(job_dir, state, "completed")
                return state
            except Exception as exc:
                state["error"] = {"type": type(exc).__name__, "message": str(exc)}
                state["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                self._write_state(job_dir, state, "failed")
                return state

    def recover_validation_failure(self, job_id: str) -> dict[str, Any]:
        """Finish a failed job from its private, already-received model output.

        No provider call is made. This is useful after deploying a deterministic
        normalizer improvement and preserves the original model response audit.
        """

        with self._solve_lock:
            job_dir = self.job_path(job_id)
            state = load_json(job_dir / "job.json")
            diagnostic_path = job_dir / "audit" / "model_validation_failure.json"
            if not diagnostic_path.is_file():
                raise FileNotFoundError("没有可恢复的模型校验诊断")
            diagnostic = load_json(diagnostic_path)
            parsed = diagnostic.get("model_output")
            errors = diagnostic.get("validation_errors")
            if not isinstance(parsed, Mapping) or not isinstance(errors, list):
                raise ValueError("模型校验诊断格式无效")
            repaired, remaining, repair_audit = apply_dynamic_local_repairs(
                parsed,
                [str(item) for item in errors],
            )
            # Historical diagnostics can predate a newly added hard gate.
            # Always reparse with today's complete validator instead of
            # trusting the old error count returned by a local repair.
            reparsed, latest_errors = parse_model_output(
                json.dumps(repaired, ensure_ascii=False)
            )
            if reparsed is None or latest_errors:
                raise ValueError(
                    "本地恢复后仍未通过数学模块强校验："
                    + "；".join(str(item) for item in latest_errors[:12])
                )
            repaired = reparsed
            publish_errors = validate_normalized_output(repaired)
            if publish_errors:
                raise ValueError(
                    "恢复结果渲染前数学模块强校验失败："
                    + "；".join(publish_errors)
                )
            source_path = job_dir / "audit" / "preprocessed_source.png"
            if not source_path.is_file():
                raise FileNotFoundError("预处理渲染源不存在")
            self._write_state(job_dir, state, "rendering")
            song, song_bold, times, times_bold = resolve_device_fonts()
            render = render_four_images(
                question_id=job_id,
                model_id=str(state.get("model_id_response") or self.model_id),
                source_image=source_path,
                output=repaired,
                output_dir=job_dir,
                song_path=song,
                song_bold_path=song_bold,
                times_path=times,
                times_bold_path=times_bold,
            )
            contract = validate_device_image_set(job_dir, require_all=True)
            if not contract["valid"]:
                raise ValueError(
                    "恢复结果图不符合数学模块图片合同："
                    + "；".join(contract["errors"])
                )
            state.update(
                {
                    "output_json": repaired,
                    "validation": {
                        "valid": True,
                        "errors": [],
                        "repair_audit": repair_audit,
                        "recovered_without_provider_call": True,
                    },
                    "render": render,
                    "image_contract": contract,
                    "images": contract["files"],
                    "error": None,
                }
            )
            self._write_state(job_dir, state, "completed")
            return state


__all__ = [
    "DEFAULT_MODEL_ID",
    "MAX_UPLOAD_BYTES",
    "RUNS_ROOT",
    "SearchMachinePipeline",
    "build_request_payload",
    "infer_dynamic_requested_outputs",
    "normalize_formula_source_figures",
    "png_data_url",
    "validate_generic_answer_blind",
    "validate_figure_reference_completeness",
    "validate_normalized_output",
]
