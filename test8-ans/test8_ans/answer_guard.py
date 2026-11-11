from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable


ANSWER_GUARD_VERSION = "targeted_answer_redaction_v4_render_independent"

_INLINE_FORMULA = re.compile(r"\$([^$]+)\$")
_CLAUSE = re.compile(r"[^，,；;。！？!?]+[，,；;。！？!?]?")
_OPTION_BLOCK = re.compile(
    r"(?:^|[\s\n])A[.、．:：]\s*.+?(?:[\s\n])B[.、．:：]",
    re.IGNORECASE | re.DOTALL,
)
_OPTION_ANSWER = re.compile(
    r"(?:最终)?(?:正确答案|答案|正确选项|选项|应选|故选|选择|确认答案)"
    r".{0,8}?(?:为|是|选)?\s*[A-H](?:项|选项)?",
    re.IGNORECASE,
)
_BARE_OPTION_CONCLUSION = re.compile(
    r"(?:^|[，,；;。！？!?])\s*[A-H](?:项|选项)(?:正确|符合|成立)",
    re.IGNORECASE,
)
_PROOF_CLOSE = re.compile(r"(?:证毕|得证|原命题成立|结论成立|命题得证)")
_STRONG_FINAL_RESULT = re.compile(
    r"(?:最终答案|最后答案|答案|最终结果|最终结论|确认结果)"
    r".{0,16}?(?:为|是|等于|约为|=|≈)"
    r".{0,24}?(?:[-+−]?\d|[\(\[（【]|向左|向右|向上|向下|"
    r"顺时针|逆时针|正确|错误|成立|不成立)",
    re.IGNORECASE,
)
_COMPUTED_FINAL_RESULT = re.compile(
    r"^(?:根据题目.{0,10})?(?:计算|求解|解答).{0,8}"
    r"(?:得出|得到|可得|求得|解得|算出).{0,36}"
    r"(?:[-+−]?\d|[\(\[（【]|向左|向右|向上|向下|"
    r"顺时针|逆时针|正确|错误|成立|不成立)",
    re.IGNORECASE,
)
_EXPLICIT_RESULT = re.compile(
    r"(?:[-+−]?\d|[\(\[（【]|向左|向右|向上|向下|顺时针|逆时针|"
    r"增大|减小|增加|减少|升高|降低|不变|正确|错误|成立|不成立)"
)
_DIRECTION_RESULT = re.compile(
    r"(?:向左|向右|向上|向下|顺时针|逆时针|水平向左|水平向右|竖直向上|竖直向下)"
)
_QUALITATIVE_RESULT = re.compile(
    r"(?:增大|减小|增加|减少|升高|降低|不变|变大|变小|更大|更小|"
    r"为正|为负|正确|错误|成立|不成立)"
)

_QUERY_TERMS = (
    "交集",
    "并集",
    "补集",
    "取值范围",
    "定义域",
    "值域",
    "最大值",
    "最小值",
    "函数值",
    "方程的解",
    "不等式的解",
    "面积",
    "周长",
    "长度",
    "角度",
    "斜率",
    "坐标",
    "概率",
    "速度",
    "速率",
    "加速度",
    "位移",
    "时间",
    "质量",
    "方向",
    "电流",
    "电压",
    "电阻",
    "功率",
    "温度",
    "压强",
    "内能",
    "拉力",
    "支持力",
    "摩擦力",
    "浮力",
    "压力",
    "合力",
)


@dataclass(frozen=True)
class AnswerTargets:
    symbols: tuple[str, ...]
    prose: tuple[str, ...]
    choice_question: bool
    judgement_question: bool
    direction_question: bool
    proof_question: bool

    @property
    def resolved(self) -> bool:
        return bool(
            self.symbols
            or self.prose
            or self.choice_question
            or self.judgement_question
            or self.direction_question
            or self.proof_question
        )


def _compact_math(value: str) -> str:
    compact = str(value or "")
    replacements = {
        r"\bigcap": "∩",
        r"\cap": "∩",
        r"\bigcup": "∪",
        r"\cup": "∪",
        r"\in": "∈",
        r"\approx": "≈",
        r"\leq": "≤",
        r"\le": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
    }
    for source, target in replacements.items():
        compact = compact.replace(source, target)
    compact = re.sub(r"\\(?:left|right|displaystyle|mathrm|text)", "", compact)
    compact = compact.replace("$", "").replace("{", "").replace("}", "")
    compact = compact.replace("（", "(").replace("）", ")")
    compact = re.sub(r"\s+", "", compact)
    return compact


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        cleaned = value.strip(" \t\r\n，,；;。:：?？")
        if cleaned and cleaned not in result:
            result.append(cleaned)
    return tuple(result)


def derive_answer_targets(question: str) -> AnswerTargets:
    """Conservatively infer what the printed question asks the learner to find.

    This is deliberately not an answer solver.  It only locates requested
    symbols or named quantities so the display copy can remove a direct final
    assignment without deleting useful intermediate relations.
    """

    symbols: list[str] = []
    prose: list[str] = []
    formulae = _INLINE_FORMULA.findall(question)

    # A blank relation immediately before choices is a strong requested-output
    # marker, for example ``$A \bigcap B =$``.
    for formula in formulae:
        compact = _compact_math(formula)
        match = re.fullmatch(r"(.+?)(?:=|≈|∈)$", compact)
        if match:
            symbols.append(match.group(1))

    query_segments = re.findall(
        r"(?:求|计算|确定|写出|指出|判断|比较)(?:出)?\s*"
        r"(.{1,48}?)(?=[，,。；;？?\n]|$)",
        question,
    )
    for segment in query_segments:
        segment_formulae = _INLINE_FORMULA.findall(segment)
        for formula in segment_formulae:
            compact = _compact_math(formula)
            if compact:
                symbols.append(compact.rstrip("=≈"))
        plain_segment = _INLINE_FORMULA.sub("", segment)
        for match in re.finditer(r"(?<![A-Za-z\\])([A-Za-z]{1,4})(?![A-Za-z])", plain_segment):
            token = match.group(1)
            if token.lower() not in {"sin", "cos", "tan", "log", "ln"}:
                symbols.append(_compact_math(token))
        for term in _QUERY_TERMS:
            if term in segment:
                prose.append(term)
        for match in re.finditer(
            r"(?<![A-Za-z\\])([A-Za-z](?:_\{?[A-Za-z0-9]+\}?)?)(?:\s*的)?值",
            segment,
        ):
            symbols.append(_compact_math(match.group(1)))

    # Common short question endings, such as “速度方向如何” or “面积是多少”.
    for match in re.finditer(
        r"([^，,。；;？?\n]{1,32}?)(?:是多少|为多少|如何|怎样|是否成立|是否正确)"
        r"[？?]?",
        question,
    ):
        segment = match.group(1)
        for term in _QUERY_TERMS:
            if term in segment:
                prose.append(term)

    if re.search(r"(?:解方程|解不等式|方程的解|不等式的解)", question):
        candidates = re.findall(r"(?<![A-Za-z\\])([A-Za-z])(?![A-Za-z])", question)
        symbols.extend(
            candidate
            for candidate in candidates
            if candidate.lower() not in {"a", "b", "c", "d"}
        )

    normalized_symbols = list(_dedupe(_compact_math(value) for value in symbols))
    if any("∩" in symbol for symbol in normalized_symbols):
        prose.append("交集")
    if any("∪" in symbol for symbol in normalized_symbols):
        prose.append("并集")

    choice_question = bool(
        _OPTION_BLOCK.search(question)
        or re.search(r"下列.{0,24}(?:正确|错误|符合|不符合).{0,8}(?:是|有)", question)
    )
    judgement_question = bool(re.search(r"(?:判断|是否|正确的是|错误的是)", question))
    direction_question = "方向" in question and bool(
        re.search(r"(?:求|判断|确定|指出|如何|怎样|哪|什么)", question)
    )
    proof_question = bool(re.search(r"(?:求证|证明)", question))
    return AnswerTargets(
        symbols=_dedupe(normalized_symbols),
        prose=_dedupe(prose),
        choice_question=choice_question,
        judgement_question=judgement_question,
        direction_question=direction_question,
        proof_question=proof_question,
    )


def _text_entries(
    value: Any, *, path: str, parent: Any, key: Any
) -> Iterable[tuple[str, Any, Any]]:
    if isinstance(value, str):
        yield path, parent, key
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _text_entries(
                child,
                path=f"{path}[{index}]",
                parent=value,
                key=index,
            )


def _student_entries(answer: dict[str, Any]) -> Iterable[tuple[str, Any, Any]]:
    """Yield (path, parent, key) for every model-authored student-facing string."""

    for tier in ("l1", "l3", "l4", "l5"):
        parts = answer.get(tier)
        if not isinstance(parts, list):
            yield from _text_entries(
                parts, path=tier, parent=answer, key=tier
            )
            continue
        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                yield from _text_entries(
                    part,
                    path=f"{tier}[{part_index}]",
                    parent=parts,
                    key=part_index,
                )
                continue
            sentences = part.get("s")
            yield from _text_entries(
                sentences,
                path=f"{tier}[{part_index}].s",
                parent=part,
                key="s",
            )
            pit = part.get("pit")
            yield from _text_entries(
                pit,
                path=f"{tier}[{part_index}].pit",
                parent=part,
                key="pit",
            )

    knowledge = answer.get("l2")
    yield from _text_entries(knowledge, path="l2", parent=answer, key="l2")

    formulae = answer.get("f")
    if isinstance(formulae, list):
        for index, item in enumerate(formulae):
            if not isinstance(item, dict):
                continue
            for key in ("x", "c"):
                if isinstance(item.get(key), str):
                    yield f"f[{index}].{key}", item, key

    if isinstance(answer.get("pat"), str):
        yield "pat", answer, "pat"
    uncertain = answer.get("u")
    if isinstance(uncertain, list):
        for index, value in enumerate(uncertain):
            if isinstance(value, str):
                yield f"u[{index}]", uncertain, index


def _isolated_target_relation(compact_clause: str, target: str) -> bool:
    if not target:
        return False
    escaped = re.escape(target)
    if re.fullmatch(r"[A-Za-z](?:_[A-Za-z0-9]+)?", target):
        left = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
    else:
        left = escaped
    relation = r"(?:=|≈|∈|为|是|等于|约为|取值范围为|方向为)"
    return re.search(left + relation + r".+", compact_clause) is not None


def _clause_rules(
    clause: str, targets: AnswerTargets, *, path: str = ""
) -> list[str]:
    rules: list[str] = []
    compact = _compact_math(clause)
    if _OPTION_ANSWER.search(clause) or _BARE_OPTION_CONCLUSION.search(clause):
        rules.append("final_option")
    if _PROOF_CLOSE.search(clause):
        rules.append("proof_close")
    if _STRONG_FINAL_RESULT.search(clause):
        rules.append("strong_final_result")
    if path.startswith("l5") and _COMPUTED_FINAL_RESULT.search(clause):
        rules.append("computed_final_result")
    for target in targets.symbols:
        if _isolated_target_relation(compact, target):
            rules.append("requested_symbol_result")
            break
    for target in targets.prose:
        start = clause.find(target)
        if start < 0:
            continue
        nearby = clause[start : start + len(target) + 28]
        if re.search(r"(?:为|是|等于|约为|取值范围为|方向为|=|≈)", nearby) and (
            _EXPLICIT_RESULT.search(nearby) or "区间" in nearby
        ):
            rules.append("requested_prose_result")
            break
    if targets.direction_question and _DIRECTION_RESULT.search(clause):
        if re.search(r"(?:方向|故|因此|所以|可知|可得|得到|判断)", clause):
            rules.append("final_direction")
    if targets.judgement_question and _QUALITATIVE_RESULT.search(clause):
        if re.search(r"(?:命题|说法|结论|判断|因此|所以|故)", clause):
            rules.append("final_judgement")
    return list(dict.fromkeys(rules))


def _redact_text(
    value: str, targets: AnswerTargets, *, path: str = ""
) -> tuple[str | None, list[str], int]:
    rules: list[str] = []
    for match in _CLAUSE.finditer(value):
        clause = match.group(0)
        clause_rules = _clause_rules(clause, targets, path=path)
        if clause_rules:
            rules.extend(clause_rules)
    if not rules:
        return value, [], 0
    # 展示侧的最小安全单位是模型提供的一整句/一个字符串字段。
    # 只裁掉句中答案从句会留下残缺文本，甚至留下答案尾部；命中后整项删除。
    return None, list(dict.fromkeys(rules)), len(value)


def sanitize_display_answer(
    answer: dict[str, Any], *, question: str, subject: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a redacted display copy while leaving the parsed model answer intact."""

    display = copy.deepcopy(answer)
    targets = derive_answer_targets(question)
    actions: list[dict[str, Any]] = []
    deletions: list[tuple[list[Any], int]] = []
    for path, parent, key in list(_student_entries(display)):
        original = parent[key]
        redacted, rules, removed_chars = _redact_text(
            original, targets, path=path
        )
        if not rules:
            continue
        if redacted is None and isinstance(parent, list) and isinstance(key, int):
            deletions.append((parent, key))
            action = "delete_full_answer_item"
        elif redacted is None:
            parent[key] = ""
            action = "clear_full_answer_field"
        else:
            parent[key] = redacted
            action = "replace_answer_item"
        actions.append(
            {
                "action": action,
                "path": path,
                "rules": rules,
                "removed_chars": removed_chars,
                "original_sha256": hashlib.sha256(
                    original.encode("utf-8")
                ).hexdigest(),
            }
        )

    for parent, key in sorted(deletions, key=lambda item: item[1], reverse=True):
        if 0 <= key < len(parent):
            parent.pop(key)

    remaining: list[dict[str, Any]] = []
    for path, parent, key in _student_entries(display):
        _, rules, _ = _redact_text(parent[key], targets, path=path)
        if rules:
            remaining.append({"path": path, "rules": rules})

    errors: list[str] = []
    if not targets.resolved:
        errors.append("无法从题面可靠定位所求量，需要人工复核脱敏结果")
    if remaining:
        errors.append("脱敏后仍检测到可能的最终答案片段")
    report = {
        "valid": not errors,
        "version": ANSWER_GUARD_VERSION,
        "method": "deterministic_targeted_redaction_without_answer_key",
        "subject": subject,
        "raw_answer_preserved": True,
        "display_copy_created": True,
        "targets": {
            "symbols": list(targets.symbols),
            "prose": list(targets.prose),
            "choice_question": targets.choice_question,
            "judgement_question": targets.judgement_question,
            "direction_question": targets.direction_question,
            "proof_question": targets.proof_question,
        },
        "redaction_count": len(actions),
        "actions": actions,
        "remaining_suspicions": remaining,
        "errors": errors,
        "render_blocking": bool(remaining),
        "render_policy": (
            "脱敏置信度只作严格评测；已创建展示副本时不因无法推断所求量而拒绝出图"
        ),
        "limitations": (
            "未读取标准答案；门禁只删除可由题面所求量和明确结论措辞定位的片段，"
            "不能替代人工答案泄漏标注。"
        ),
    }
    return display, report
