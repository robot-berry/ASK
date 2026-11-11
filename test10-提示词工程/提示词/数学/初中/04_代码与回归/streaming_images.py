"""Helpers for publishing the first result while ordered JSON is still streaming."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


KNOWLEDGE_PREFIX_KEYS = (
    "subject",
    "problem_scope",
    "recognition_status",
    "uncertain_items",
    "question_blocks",
    "source_figures",
    "visual_evidence",
    "knowledge_points",
    "base_formulas",
)


def completed_ordered_prefix(
    accumulated: str,
    expected_keys: Sequence[str] = KNOWLEDGE_PREFIX_KEYS,
) -> dict[str, Any] | None:
    """Decode completed leading fields without waiting for the root object to close."""

    if not accumulated:
        return None
    start = accumulated.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    position = start + 1
    completed: dict[str, Any] = {}

    def skip_space(index: int) -> int:
        while index < len(accumulated) and accumulated[index].isspace():
            index += 1
        return index

    for expected_key in expected_keys:
        position = skip_space(position)
        try:
            key, position = decoder.raw_decode(accumulated, position)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if key != expected_key:
            return None
        position = skip_space(position)
        if position >= len(accumulated) or accumulated[position] != ":":
            return None
        position = skip_space(position + 1)
        try:
            value, position = decoder.raw_decode(accumulated, position)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        completed[expected_key] = value
        position = skip_space(position)
        if expected_key != expected_keys[-1]:
            if position >= len(accumulated) or accumulated[position] != ",":
                return None
            position += 1
    return completed


def validation_candidate(
    prefix: Mapping[str, Any],
    *,
    subject: str,
    part_labels: Sequence[str],
) -> dict[str, Any]:
    """Add harmless guidance placeholders so existing strict prefix gates can run."""

    labels = [str(item).strip() for item in part_labels if str(item).strip()]
    if not labels:
        labels = ["整题"]

    def parts(step_count: int, *, standard: bool = False) -> list[dict[str, Any]]:
        templates = (
            "识别题设中的已知条件与所求对象",
            "选择对应知识并建立中间关系",
            "检查公式条件与变量范围是否匹配",
            "保留最终求值步骤供学习者完成",
        )
        result: list[dict[str, Any]] = []
        for label in labels:
            item: dict[str, Any] = {
                "part": label,
                "method_id": "M1",
                "method_name": "条件关系法",
                "steps": list(templates[:step_count]),
                "alternative_methods": [],
            }
            if standard:
                item["pitfall"] = "注意题设范围和公式适用条件"
            result.append(item)
        return result

    guidance: dict[str, Any] = {
        "advanced": {"title": "学霸版", "parts": parts(2)},
        "standard": {"title": "普通版", "parts": parts(3, standard=True)},
        "foundation": {"title": "基础版", "parts": parts(4)},
    }
    if subject == "mathematics":
        guidance["foundation"]["pattern_summary"] = (
            "先提取条件，再选择知识建立关系，最后核对适用范围"
        )
    candidate = dict(prefix)
    candidate["guidance"] = guidance
    return candidate


class EarlyKnowledgePublisher:
    """Publish the knowledge page once its ordered JSON prefix is complete."""

    def __init__(
        self,
        *,
        destination: Path,
        validate: Callable[[Mapping[str, Any]], list[str]],
        render: Callable[[Mapping[str, Any], Path], None],
        publish: Callable[[Mapping[str, Any], float | None], None],
        inspect: Callable[[Path], Mapping[str, Any]],
        reject: Callable[[Sequence[str]], None] | None = None,
    ) -> None:
        self.destination = Path(destination)
        self.validate = validate
        self.render = render
        self.publish = publish
        self.inspect = inspect
        self.reject = reject
        self.attempted = False
        self.published = False
        self.errors: list[str] = []

    def __call__(self, accumulated: str, provider_elapsed_ms: float | None) -> None:
        if self.attempted:
            return
        prefix = completed_ordered_prefix(accumulated)
        if prefix is None:
            return
        self.attempted = True
        errors = list(self.validate(prefix))
        if errors:
            self.errors = errors
            if self.reject is not None:
                self.reject(errors)
            return
        temporary = self.destination.with_name(
            f".{self.destination.stem}.stream-{os.getpid()}.tmp.png"
        )
        try:
            self.render(prefix, temporary)
            metadata = dict(self.inspect(temporary))
            if not metadata.get("valid"):
                raise ValueError("首张流式图片未通过图片合同校验")
            os.replace(temporary, self.destination)
            final_metadata = dict(self.inspect(self.destination))
            self.publish(final_metadata, provider_elapsed_ms)
            self.published = True
        finally:
            temporary.unlink(missing_ok=True)

