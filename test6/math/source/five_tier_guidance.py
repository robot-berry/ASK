#!/usr/bin/env python3
"""Select five student-facing levels, preferring model-authored tier prose.

New jobs contain ``display_guidance`` written by the model for all five levels.
The historical projection functions remain only for rerendering old jobs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


DISPLAY_TIER_KEYS = ("elite", "upper", "middle", "lower", "novice")
DISPLAY_TIER_TITLES = {
    "elite": "学霸解析",
    "upper": "中上解析",
    "middle": "中等解析",
    "lower": "中下解析",
    "novice": "学渣解析",
}
DISPLAY_TIER_SECTION_TITLES = {
    "elite": "",
    "upper": "",
    "middle": "",
    "lower": "",
    "novice": "",
}
DISPLAY_TIER_DETAIL_LEVELS = {
    "elite": "questions_only",
    "upper": "knowledge_domain_only",
    "middle": "key_insight",
    "lower": "solution_skeleton",
    "novice": "fallback_detailed",
}

_ELITE_QUESTIONS = (
    "这一问最终要求你得到怎样的结论？",
    "哪些已知条件还没有被你用到？",
    "你的推导是否用齐条件并满足全部限制？",
)

_LOWER_STUDENT_TASK = "按上面的骨架完成关键计算或论证：________"
_NOVICE_STUDENT_TASK = "现在请你完成最后的代入、运算或结论：________"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _parts(output: Mapping[str, Any], tier_key: str) -> list[Mapping[str, Any]]:
    guidance = _mapping(output.get("guidance"))
    tier = _mapping(guidance.get(tier_key))
    value = tier.get("parts")
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _source_parts(output: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Prefer the detailed part list and retain stable fallbacks."""

    return _parts(output, "foundation") or _parts(output, "standard") or _parts(output, "advanced")


def _matching_part(
    output: Mapping[str, Any], tier_key: str, index: int, part_name: str
) -> Mapping[str, Any]:
    candidates = _parts(output, tier_key)
    for candidate in candidates:
        if str(candidate.get("part") or "").strip() == part_name:
            return candidate
    return candidates[index] if index < len(candidates) else {}


def _steps(part: Mapping[str, Any]) -> list[str]:
    value = part.get("steps")
    if not isinstance(value, list):
        return []
    return [
        str(item).strip(" \t\r\n，,、。．；;：:·")
        for item in value
        if str(item).strip(" \t\r\n，,、。．；;：:·")
    ]


def _first_step(part: Mapping[str, Any]) -> str:
    values = _steps(part)
    return values[0] if values else "先找出题目目标与已知条件之间最直接的联系"


def _copy_alternatives(part: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = part.get("alternative_methods")
    if not isinstance(value, list):
        return []
    return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]


def _part_shell(part_name: str, *, steps: list[str], method_name: str = "") -> dict[str, Any]:
    return {
        "part": part_name or "整题",
        "method_id": "M1",
        "method_name": method_name,
        "steps": steps,
        "alternative_methods": [],
    }


def _project_elite(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _part_shell(str(part.get("part") or "整题").strip(), steps=list(_ELITE_QUESTIONS))
        for part in _source_parts(output)
    ]


def _project_middle(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for index, source in enumerate(_source_parts(output)):
        part_name = str(source.get("part") or "整题").strip()
        standard = _matching_part(output, "standard", index, part_name)
        advanced = _matching_part(output, "advanced", index, part_name)
        clue_source = standard or advanced or source
        projected.append(
            _part_shell(
                part_name,
                steps=[f"关键观察：{_first_step(clue_source)}"],
            )
        )
    return projected


def _project_lower(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for index, source in enumerate(_source_parts(output)):
        part_name = str(source.get("part") or "整题").strip()
        advanced = _matching_part(output, "advanced", index, part_name)
        skeleton = advanced or source
        projected.append(
            _part_shell(
                part_name,
                steps=(
                    (_steps(skeleton) or [
                    "整理已知条件和题目目标",
                    "建立条件与目标之间的关系",
                    "分步推导并核对适用条件",
                    ])
                    + [_LOWER_STUDENT_TASK]
                ),
                method_name=str(skeleton.get("method_name") or "").strip(),
            )
        )
    return projected


def _project_novice(output: Mapping[str, Any]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for index, source in enumerate(_source_parts(output)):
        part_name = str(source.get("part") or "整题").strip()
        standard = _matching_part(output, "standard", index, part_name)
        detail = source or standard
        detail_steps = _steps(detail) or _steps(standard)
        part = _part_shell(
            part_name,
            steps=detail_steps + [_NOVICE_STUDENT_TASK],
            method_name=str(detail.get("method_name") or standard.get("method_name") or "").strip(),
        )
        part["method_id"] = str(detail.get("method_id") or "M1")
        part["alternative_methods"] = _copy_alternatives(detail)
        pitfall = str(detail.get("pitfall") or standard.get("pitfall") or "").strip()
        if pitfall:
            part["pitfall"] = pitfall
        projected.append(part)
    return projected


def project_display_tier(tier_key: str, output: Mapping[str, Any]) -> dict[str, Any]:
    """Return direct model prose for one level, with a legacy fallback."""

    if tier_key not in DISPLAY_TIER_KEYS:
        raise KeyError(f"unknown display tier: {tier_key}")
    direct_guidance = _mapping(output.get("display_guidance"))
    direct_page = direct_guidance.get(tier_key)
    if isinstance(direct_page, Mapping):
        page = deepcopy(dict(direct_page))
        # Titles and renderer metadata are trusted local presentation labels;
        # parts, steps, knowledge and formulas remain the checked model text.
        page["tier_key"] = tier_key
        page["title"] = DISPLAY_TIER_TITLES[tier_key]
        page.setdefault("section_title", "")
        page.setdefault("detail_level", f"model_{DISPLAY_TIER_DETAIL_LEVELS[tier_key]}")
        page.setdefault("knowledge_points", [])
        page.setdefault("base_formulas", [])
        page.setdefault("parts", [])
        page.setdefault("pattern_summary", "")
        return page
    foundation = _mapping(_mapping(output.get("guidance")).get("foundation"))
    page: dict[str, Any] = {
        "tier_key": tier_key,
        "title": DISPLAY_TIER_TITLES[tier_key],
        "section_title": DISPLAY_TIER_SECTION_TITLES[tier_key],
        "detail_level": DISPLAY_TIER_DETAIL_LEVELS[tier_key],
        "knowledge_points": [],
        "base_formulas": [],
        "parts": [],
        "pattern_summary": "",
    }
    if tier_key == "elite":
        page["parts"] = _project_elite(output)
    elif tier_key == "upper":
        page["knowledge_points"] = [
            str(item).strip() for item in output.get("knowledge_points", []) if str(item).strip()
        ] or ["题目条件与目标的关系"]
    elif tier_key == "middle":
        page["parts"] = _project_middle(output)
    elif tier_key == "lower":
        page["parts"] = _project_lower(output)
    else:
        page["knowledge_points"] = deepcopy(list(output.get("knowledge_points", [])))
        page["base_formulas"] = deepcopy(list(output.get("base_formulas", [])))
        page["parts"] = _project_novice(output)
        page["pattern_summary"] = str(foundation.get("pattern_summary") or "").strip()
        if output.get("recognition_status") == "uncertain":
            page["uncertain_items"] = deepcopy(list(output.get("uncertain_items", [])))
    return page


__all__ = [
    "DISPLAY_TIER_DETAIL_LEVELS",
    "DISPLAY_TIER_KEYS",
    "DISPLAY_TIER_SECTION_TITLES",
    "DISPLAY_TIER_TITLES",
    "project_display_tier",
]
