from __future__ import annotations

import re
from typing import Any, Iterable


GENERIC_PHRASES = (
    "分析题意",
    "整理条件",
    "建立关系",
    "选择公式",
    "执行运算",
    "逐步求解",
    "根据题意",
    "结合所学知识",
)
GENERIC_CJK_BIGRAMS = {
    "本题",
    "题目",
    "条件",
    "关系",
    "结果",
    "分析",
    "已知",
    "可以",
    "根据",
    "利用",
    "得到",
    "进行",
    "最后",
    "检查",
}
SYMBOL_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,15}|\d+(?:\.\d+)?")
CJK_RUN = re.compile(r"[\u3400-\u9fff]+")


CONCEPT_CUES = (
    (re.compile(r"=|方程"), {"方程", "等式", "未知数", "两边"}),
    (re.compile(r"<|>|≤|≥|不等式"), {"不等式", "范围", "边界"}),
    (re.compile(r"函数|f\s*\("), {"函数", "定义域", "图像"}),
    (re.compile(r"三角|∠|角|圆|平行|垂直"), {"几何", "角", "三角形", "圆"}),
    (re.compile(r"概率|随机"), {"概率", "事件", "样本"}),
    (re.compile(r"速度|m/s|km/h"), {"速度", "路程", "时间"}),
    (re.compile(r"质量|kg"), {"质量", "物体"}),
    (re.compile(r"电流|电压|电阻|电路"), {"电流", "电压", "电阻", "电路"}),
    (re.compile(r"力|N\b"), {"受力", "物体", "方向"}),
    (re.compile(r"能量|功率|做功"), {"能量", "功", "功率"}),
)


def _flatten_text(value: Any, *, skip_keys: set[str] | None = None) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(
            _flatten_text(item, skip_keys=skip_keys)
            for key, item in value.items()
            if not skip_keys or key not in skip_keys
        )
    if isinstance(value, list):
        return " ".join(_flatten_text(item, skip_keys=skip_keys) for item in value)
    return ""


def _cjk_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    for run in CJK_RUN.findall(text):
        for size in (2, 3):
            for index in range(max(0, len(run) - size + 1)):
                anchor = run[index : index + size]
                if anchor not in GENERIC_CJK_BIGRAMS:
                    anchors.add(anchor)
    return anchors


def _symbol_anchors(text: str) -> set[str]:
    return {match.group(0).lower() for match in SYMBOL_TOKEN.finditer(text)}


def _concept_anchors(text: str) -> set[str]:
    anchors: set[str] = set()
    for pattern, concepts in CONCEPT_CUES:
        if pattern.search(text):
            anchors.update(concepts)
    return anchors


def _tier_texts(answer: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for tier in ("l1", "l3", "l4", "l5"):
        yield tier, _flatten_text(answer.get(tier, []), skip_keys={"n"})


def validate_relevance(question: str, answer: dict[str, Any]) -> dict[str, Any]:
    answer_text = _flatten_text(answer, skip_keys={"n"})
    question_cjk = _cjk_anchors(question) | _concept_anchors(question)
    answer_cjk = _cjk_anchors(answer_text)
    question_symbols = _symbol_anchors(question)
    answer_symbols = _symbol_anchors(answer_text)
    cjk_overlap = sorted(question_cjk & answer_cjk)
    symbol_overlap = sorted(question_symbols & answer_symbols)

    tier_matches: dict[str, int] = {}
    for tier, tier_text in _tier_texts(answer):
        tier_matches[tier] = len(question_cjk & _cjk_anchors(tier_text)) + len(
            question_symbols & _symbol_anchors(tier_text)
        )
    tiers_with_anchor = sum(value > 0 for value in tier_matches.values())
    generic_hits = [phrase for phrase in GENERIC_PHRASES if phrase in answer_text]

    reasons: list[str] = []
    if len(answer_text) < 120:
        reasons.append("五层解析总文本过短，无法形成针对性分析")
    if len(cjk_overlap) + len(symbol_overlap) < 2:
        reasons.append("解析与题面共享的可识别文字或符号锚点不足")
    if tiers_with_anchor < 2:
        reasons.append("L1/L3/L4/L5 中至少两层应出现题面特有锚点")
    if len(generic_hits) >= 3 and len(cjk_overlap) + len(symbol_overlap) < 4:
        reasons.append("解析以通用操作话术为主，缺少本题事实")

    return {
        "valid": not reasons,
        "method": "local_lexical_anchor_v1",
        "scope": "screening_only_not_semantic_correctness",
        "errors": reasons,
        "metrics": {
            "question_cjk_anchor_count": len(question_cjk),
            "question_symbol_anchor_count": len(question_symbols),
            "cjk_overlap_count": len(cjk_overlap),
            "symbol_overlap_count": len(symbol_overlap),
            "tiers_with_anchor": tiers_with_anchor,
            "tier_match_counts": tier_matches,
            "generic_phrase_hits": generic_hits,
            "answer_text_chars": len(answer_text),
        },
        "evidence": {
            "matched_cjk_anchors": cjk_overlap[:30],
            "matched_symbol_anchors": symbol_overlap[:30],
        },
    }
