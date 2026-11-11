from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


SUBJECT_ALIASES = {
    "math": "math",
    "数学": "math",
    "physics": "physics",
    "物理": "physics",
}
STAGE_ALIASES = {
    "primary": "primary",
    "小学": "primary",
    "junior": "junior",
    "初中": "junior",
    "senior": "senior",
    "高中": "senior",
}
QUESTION_SOURCE_VALUES = {"verified_text", "ocr_only"}


def normalize_subject(value: str) -> str:
    normalized = SUBJECT_ALIASES.get(value.strip().lower())
    if not normalized:
        raise ValueError("subject 只支持 math/数学 或 physics/物理")
    return normalized


def normalize_stage(value: str) -> str:
    normalized = STAGE_ALIASES.get(value.strip().lower())
    if not normalized:
        raise ValueError("stage 只支持 primary/小学、junior/初中、senior/高中")
    return normalized


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "-", value.strip()).strip("-.")
    return cleaned[:80] or "question"


@dataclass(frozen=True)
class Question:
    id: str
    subject: str
    stage: str
    question: str
    ocr: str = ""
    source_image: str = ""
    question_source: str = "verified_text"
    dataset_id: str = ""
    image_sha256: str = ""
    upstream_latency_ms: float | None = None
    source_quality: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, index: int = 1) -> "Question":
        subject = normalize_subject(str(value.get("subject", "")))
        stage = normalize_stage(str(value.get("stage", "")))
        question = str(value.get("question", "")).strip()
        if not question:
            raise ValueError("question 不能为空；请传入已识别或人工整理的题面文字")
        question_source = str(value.get("question_source", "verified_text")).strip()
        if question_source not in QUESTION_SOURCE_VALUES:
            raise ValueError("question_source 只支持 verified_text 或 ocr_only")
        raw_upstream_latency = value.get("upstream_latency_ms")
        upstream_latency_ms = None
        if raw_upstream_latency not in (None, ""):
            upstream_latency_ms = float(raw_upstream_latency)
            if upstream_latency_ms < 0:
                raise ValueError("upstream_latency_ms 不能为负数")
        raw_id = str(value.get("id") or f"question-{index:04d}")
        return cls(
            id=safe_id(raw_id),
            subject=subject,
            stage=stage,
            question=question,
            ocr=str(value.get("ocr", "")).strip(),
            source_image=str(value.get("source_image", "")).strip(),
            question_source=question_source,
            dataset_id=str(value.get("dataset_id", "")).strip(),
            image_sha256=str(value.get("image_sha256", "")).strip().lower(),
            upstream_latency_ms=upstream_latency_ms,
            source_quality=(
                dict(value.get("source_quality", {}))
                if isinstance(value.get("source_quality", {}), Mapping)
                else {}
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "stage": self.stage,
            "question": self.question,
            "ocr": self.ocr,
            "source_image": self.source_image,
            "question_source": self.question_source,
            "dataset_id": self.dataset_id,
            "image_sha256": self.image_sha256,
            "upstream_latency_ms": self.upstream_latency_ms,
            "source_quality": self.source_quality,
        }

    @classmethod
    def from_text_file(
        cls, path: Path, *, question_id: str, subject: str, stage: str
    ) -> "Question":
        return cls.from_mapping(
            {
                "id": question_id,
                "subject": subject,
                "stage": stage,
                "question": path.read_text(encoding="utf-8-sig"),
            }
        )
