from __future__ import annotations

import json
import re
from typing import Any


ROOT_FIELDS = ["l1", "l2", "l3", "l4", "l5", "f", "pat", "s", "u"]
TIER_SENTENCE_COUNTS = {"l1": 1, "l3": 1, "l4": 2, "l5": 4}


class OutputParseError(ValueError):
    pass


class _ObjectPairs(list[tuple[str, Any]]):
    """JSON object representation that preserves duplicate member names."""


def _normalize_object_pairs(
    value: Any,
    *,
    path: str,
    audit: list[dict[str, Any]],
) -> Any:
    if isinstance(value, _ObjectPairs):
        grouped: dict[str, list[Any]] = {}
        key_order: list[str] = []
        for key, child in value:
            if key not in grouped:
                grouped[key] = []
                key_order.append(key)
            grouped[key].append(child)
        duplicate_keys = [key for key, values in grouped.items() if len(values) > 1]
        is_tier_part = re.fullmatch(r"\$\.(?:l1|l3|l4|l5)\[\d+\]", path) is not None
        unsafe_duplicates = [
            key for key in duplicate_keys if not (is_tier_part and key == "s")
        ]
        if unsafe_duplicates:
            raise OutputParseError(
                f"{path} 含无法安全恢复的重复键：{','.join(unsafe_duplicates)}"
            )
        result: dict[str, Any] = {}
        for key in key_order:
            children = grouped[key]
            child_path = f"{path}.{key}"
            if is_tier_part and key == "s":
                sentences: list[Any] = []
                source_was_array = len(children) == 1 and type(children[0]) is list
                for child in children:
                    normalized = _normalize_object_pairs(
                        child, path=child_path, audit=audit
                    )
                    if isinstance(normalized, list):
                        sentences.extend(normalized)
                    else:
                        sentences.append(normalized)
                result[key] = sentences
                if len(children) > 1:
                    audit.append(
                        {
                            "action": "collect_duplicate_s_keys",
                            "path": child_path,
                            "source_key_count": len(children),
                            "sentence_count": len(sentences),
                        }
                    )
                elif not source_was_array:
                    audit.append(
                        {
                            "action": "wrap_s_string_as_array",
                            "path": child_path,
                            "sentence_count": len(sentences),
                        }
                    )
            else:
                result[key] = _normalize_object_pairs(
                    children[0], path=child_path, audit=audit
                )
        return result
    if isinstance(value, list):
        return [
            _normalize_object_pairs(child, path=f"{path}[{index}]", audit=audit)
            for index, child in enumerate(value)
        ]
    return value


def parse_json_content(
    content: str,
    *,
    allow_code_fence: bool = False,
    normalize_sentence_transport: bool = False,
    normalization_audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        if not allow_code_fence:
            raise OutputParseError("模型输出包含 Markdown 代码围栏，不是原生 JSON")
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        if normalize_sentence_transport:
            raw_value = json.loads(text, object_pairs_hook=_ObjectPairs)
            audit = normalization_audit if normalization_audit is not None else []
            value = _normalize_object_pairs(raw_value, path="$", audit=audit)
        else:
            value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OutputParseError(
            f"不是有效 JSON：第 {exc.lineno} 行第 {exc.colno} 列，{exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise OutputParseError("JSON 根值必须是对象")
    return value


def _string_error(value: Any, path: str, minimum: int, maximum: int) -> str | None:
    if not isinstance(value, str):
        return f"{path} 必须是字符串"
    if not minimum <= len(value) <= maximum:
        return f"{path} 长度必须在 {minimum} 到 {maximum} 个字符之间"
    return None


def validate_guidance(value: Any, subject: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(value, dict):
        return {"valid": False, "errors": ["根值必须是对象"], "warnings": []}

    keys = list(value.keys())
    if keys != ROOT_FIELDS:
        errors.append(f"根字段及顺序必须为 {','.join(ROOT_FIELDS)}")
    for field in ROOT_FIELDS:
        if field not in value:
            errors.append(f"缺少根字段 {field}")

    tier_numbers: dict[str, list[str]] = {}
    for tier, sentence_count in TIER_SENTENCE_COUNTS.items():
        parts = value.get(tier)
        if not isinstance(parts, list):
            errors.append(f"{tier} 必须是数组")
            continue
        if not 1 <= len(parts) <= 12:
            errors.append(f"{tier} 小问数必须在 1 到 12 之间")
        numbers: list[str] = []
        for part_index, part in enumerate(parts):
            path = f"{tier}[{part_index}]"
            expected_keys = {"n", "s", "pit"} if tier == "l5" else {"n", "s"}
            if not isinstance(part, dict):
                errors.append(f"{path} 必须是对象")
                continue
            if set(part) != expected_keys:
                errors.append(f"{path} 字段必须恰好为 {sorted(expected_keys)}")
            number = part.get("n")
            error = _string_error(number, f"{path}.n", 1, 20)
            if error:
                errors.append(error)
            else:
                numbers.append(number)
            sentences = part.get("s")
            if not isinstance(sentences, list):
                errors.append(f"{path}.s 必须是数组")
            elif len(sentences) != sentence_count:
                errors.append(f"{path}.s 必须恰好有 {sentence_count} 句")
            else:
                for sentence_index, sentence in enumerate(sentences):
                    sentence_path = f"{path}.s[{sentence_index}]"
                    error = _string_error(sentence, sentence_path, 8, 60)
                    if error:
                        errors.append(error)
                    elif not 15 <= len(sentence) <= 35:
                        warnings.append(f"{sentence_path} 未落在建议的 15 到 35 字范围")
                if tier == "l1" and sentences and isinstance(sentences[0], str):
                    if not sentences[0].rstrip().endswith(("？", "?")):
                        errors.append(f"{path}.s[0] 必须是问句")
            if tier == "l5":
                error = _string_error(part.get("pit"), f"{path}.pit", 2, 80)
                if error:
                    errors.append(error)
        tier_numbers[tier] = numbers

    baseline = tier_numbers.get("l1")
    if baseline:
        for tier in ("l3", "l4", "l5"):
            if tier_numbers.get(tier) != baseline:
                errors.append(f"{tier} 的小问题号、顺序和数量必须与 l1 一致")

    l2 = value.get("l2")
    if not isinstance(l2, list):
        errors.append("l2 必须是数组")
    else:
        if not 1 <= len(l2) <= 5:
            errors.append("l2 必须有 1 到 5 个知识点")
        elif len(l2) == 1:
            warnings.append("l2 只有 1 个知识点，低于提示词建议的 2 个")
        for index, item in enumerate(l2):
            error = _string_error(item, f"l2[{index}]", 1, 60)
            if error:
                errors.append(error)

    formulas = value.get("f")
    minimum_formulas = 1 if subject == "math" else 0
    if not isinstance(formulas, list):
        errors.append("f 必须是数组")
    else:
        if not minimum_formulas <= len(formulas) <= 3:
            errors.append(f"f 必须有 {minimum_formulas} 到 3 个基础公式")
        for index, formula in enumerate(formulas):
            path = f"f[{index}]"
            if not isinstance(formula, dict) or set(formula) != {"x", "c"}:
                errors.append(f"{path} 必须是只含 x、c 的对象")
                continue
            for key, maximum in (("x", 120), ("c", 100)):
                error = _string_error(formula.get(key), f"{path}.{key}", 1, maximum)
                if error:
                    errors.append(error)
            if isinstance(formula.get("x"), str) and "$" in formula["x"]:
                errors.append(f"{path}.x 不应包含 $ 定界符")

    error = _string_error(value.get("pat"), "pat", 4, 100)
    if error:
        errors.append(error)
    if value.get("s") not in {"c", "u"}:
        errors.append("s 必须为 c 或 u")
    uncertainties = value.get("u")
    if not isinstance(uncertainties, list):
        errors.append("u 必须是数组")
    else:
        if len(uncertainties) > 8:
            errors.append("u 最多有 8 项")
        for index, item in enumerate(uncertainties):
            error = _string_error(item, f"u[{index}]", 1, 100)
            if error:
                errors.append(error)

    char_count = len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    if char_count > 2600:
        warnings.append(f"紧凑 JSON 共 {char_count} 字符，超过建议的 2600 字符")
    return {"valid": not errors, "errors": errors, "warnings": warnings}
