"""Publish complete model-authored tiers while the provider stream is open."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from streaming_images import completed_ordered_prefix


TIER_FIELDS = (("l1", "elite"), ("l2", "upper"), ("l3", "middle"),
               ("l4", "lower"), ("l5", "novice"))
ROOT_FIELDS = ("l1", "l2", "l3", "l4", "l5", "f", "pat", "s", "u")
TIER_TEXT_FIELDS = ROOT_FIELDS[:5]


class TierStreamPublisher:
    """Synchronous, bounded work at five field boundaries, never per token.

    prepare applies the same local redaction/validation as final publication.
    Only the requested model tier is rendered; any internal validation defaults
    for fields that have not arrived must never be published.

    Novice publishes as soon as l1-l5 close. Formulas and summary may refresh
    the sixth image once when those trailing fields arrive.

    The post-lower budget never ends the stream before a model-authored L5
    page exists. After that page is published, leftover time only cuts the
    trailing formula/summary wait or an L5 draft loop that already has five
    usable sentences. Local templates must not stand in for L5.
    """

    def __init__(self, *, directory: Path, prepare: Callable, render: Callable,
                 inspect: Callable, publish: Callable, audit: dict[str, Any],
                 post_lower_budget_seconds: float = 0.0):
        self.directory = Path(directory)
        self.prepare = prepare
        self.render = render
        self.inspect = inspect
        self.publish = publish
        self.audit = audit
        self.attempted: set[str] = set()
        self.first_content_ms: float | None = None
        self.post_lower_budget_seconds = max(0.0, float(post_lower_budget_seconds or 0.0))
        self.lower_published_at: float | None = None
        self.novice_published_at: float | None = None
        self.novice_metadata_published = False
        self.post_lower_stop_reason: str | None = None

    def should_stop_post_lower_budget(self) -> bool:
        if self.post_lower_stop_reason:
            return True
        if self.post_lower_budget_seconds <= 0 or self.lower_published_at is None:
            return False
        # L5 must be the model's own closed tier, never a local template page.
        if not (self.audit.get("novice") or {}).get("published"):
            return False
        if time.perf_counter() - self.lower_published_at < self.post_lower_budget_seconds:
            return False
        self.post_lower_stop_reason = "post_lower_budget_exhausted"
        return True

    def __call__(self, text: str, elapsed_ms: float | None) -> None:
        if text and self.first_content_ms is None:
            self.first_content_ms = elapsed_ms
        for index, (field, kind) in enumerate(TIER_FIELDS):
            if kind == "novice":
                self._publish_novice(text, elapsed_ms, index, field)
                continue
            if kind in self.attempted:
                continue
            prefix = completed_ordered_prefix(text, ROOT_FIELDS[:index + 1])
            if prefix is None:
                continue
            self.attempted.add(kind)
            self._emit(index, field, kind, prefix, elapsed_ms, refresh=False)

    def _publish_novice(self, text: str, elapsed_ms: float | None, index: int, field: str) -> None:
        full = completed_ordered_prefix(text, ROOT_FIELDS)
        if "novice" not in self.attempted:
            prefix = full or completed_ordered_prefix(text, TIER_TEXT_FIELDS)
            if prefix is None:
                return
            self.attempted.add("novice")
            self._emit(index, field, "novice", prefix, elapsed_ms, refresh=False)
            if full is not None and self.audit.get("novice", {}).get("published"):
                self.novice_metadata_published = True
            return
        if self.novice_metadata_published or not self.audit.get("novice", {}).get("published"):
            return
        if full is None:
            return
        self._emit(index, field, "novice", full, elapsed_ms, refresh=True)

    def _emit(
        self,
        index: int,
        field: str,
        kind: str,
        prefix: Mapping[str, Any],
        elapsed_ms: float | None,
        *,
        refresh: bool,
    ) -> None:
        if refresh:
            entry = self.audit[kind]
        else:
            entry = {"field": field, "provider_ready_ms": elapsed_ms, "published": False}
            self.audit[kind] = entry
        started = time.perf_counter()
        destination = self.directory / f"{index + 2:02d}_guidance_{kind}.png"
        temporary = destination.with_name(f".{destination.stem}.tier.tmp.png")
        try:
            value = prefix[field]
            if not isinstance(value, list) or not value:
                raise ValueError("模型档位为空或不是数组，留待整份回答处理")
            if field == "l2":
                if not all(isinstance(item, str) and item.strip() for item in value):
                    raise ValueError("模型知识点字段不完整")
            elif not all(
                isinstance(item, dict) and isinstance(item.get("n"), str)
                and isinstance(item.get("s"), list) and item["s"]
                and all(isinstance(step, str) and step.strip() for step in item["s"])
                for item in value
            ):
                raise ValueError("模型小问字段不完整")
            if field != "l2":
                expected = {"l1": 2, "l3": 2, "l4": 3, "l5": 5}[field]
                if any(len(item["s"]) < expected for item in value):
                    raise ValueError("模型档位句子尚不齐全，等待最终本地补齐")
                if [item["n"] for item in value] != [item["n"] for item in prefix["l1"]]:
                    raise ValueError("模型小问尚未对齐，等待最终校验")
            output = self.prepare(prefix, kind)
            self.render(kind, output, temporary)
            metadata = dict(self.inspect(temporary))
            if not metadata.get("valid"):
                raise ValueError("流式解析图片合同校验失败")
            os.replace(temporary, destination)
            entry["published"] = True
            processing_ms = round((time.perf_counter() - started) * 1000, 3)
            if refresh:
                entry["metadata_refresh_ms"] = processing_ms
                entry["metadata_refreshed"] = True
                self.novice_metadata_published = True
            else:
                entry["local_processing_ms"] = processing_ms
                if kind == "lower":
                    self.lower_published_at = time.perf_counter()
                elif kind == "novice":
                    self.novice_published_at = time.perf_counter()
            self.publish(kind, dict(self.inspect(destination)), elapsed_ms)
        except Exception as exc:
            # A malformed/unsafe early tier must not abort the provider call.
            if refresh:
                entry["metadata_refresh_error"] = f"{type(exc).__name__}: {exc}"
            else:
                entry["published"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["deferred_to_final"] = True
        finally:
            temporary.unlink(missing_ok=True)
