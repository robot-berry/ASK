"""Physics wrapper for the shared compact-guidance implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SHARED = (
    Path(__file__).resolve().parents[2]
    / "math"
    / "source"
    / "compact_guidance.py"
)
_SPEC = importlib.util.spec_from_file_location("test6_shared_compact_guidance", _SHARED)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"无法加载公共紧凑协议：{_SHARED}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

COMPACT_PREFIX_KEYS = _MODULE.COMPACT_PREFIX_KEYS
CompactStreamCollector = _MODULE.CompactStreamCollector
DELIVERY_SYSTEM_PROMPT = _MODULE.DELIVERY_SYSTEM_PROMPT
clean_model_analysis_steps = _MODULE.clean_model_analysis_steps
adaptive_image_policy = _MODULE.adaptive_image_policy
classify_problem_complexity = _MODULE.classify_problem_complexity
resolve_provider_image_policy = _MODULE.resolve_provider_image_policy
simple_request_max_tokens = _MODULE.simple_request_max_tokens
curriculum_scope_instruction = _MODULE.curriculum_scope_instruction
curriculum_violations = _MODULE.curriculum_violations
expand_compact_output = _MODULE.expand_compact_output
infer_curriculum_stage = _MODULE.infer_curriculum_stage
local_fallback_compact = _MODULE.local_fallback_compact
parse_provider_document = _MODULE.parse_provider_document
repair_five_tier_document = _MODULE.repair_five_tier_document
answer_blind_analysis_prose = _MODULE.answer_blind_analysis_prose
display_guidance_entries = _MODULE.display_guidance_entries
scrub_visible_answers = _MODULE.scrub_visible_answers

__all__ = [
    "COMPACT_PREFIX_KEYS",
    "CompactStreamCollector",
    "adaptive_image_policy",
    "classify_problem_complexity",
    "resolve_provider_image_policy",
    "simple_request_max_tokens",
    "curriculum_scope_instruction",
    "curriculum_violations",
    "expand_compact_output",
    "infer_curriculum_stage",
    "local_fallback_compact",
    "parse_provider_document",
    "repair_five_tier_document",
    "answer_blind_analysis_prose",
    "display_guidance_entries",
    "scrub_visible_answers",
]
