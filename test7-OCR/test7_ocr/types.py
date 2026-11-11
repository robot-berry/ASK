from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


Status = Literal["PASS", "REVIEW", "REJECT"]


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def clamp(self, image_width: int, image_height: int) -> "BBox":
        x0 = max(0, min(self.x, image_width))
        y0 = max(0, min(self.y, image_height))
        x1 = max(x0, min(self.right, image_width))
        y1 = max(y0, min(self.bottom, image_height))
        return BBox(x0, y0, x1 - x0, y1 - y0)

    @classmethod
    def from_value(cls, value: Any) -> "BBox":
        if isinstance(value, dict):
            return cls(
                int(value["x"]),
                int(value["y"]),
                int(value["width"]),
                int(value["height"]),
            )
        if isinstance(value, (list, tuple)) and len(value) == 4:
            return cls(*(int(part) for part in value))
        raise ValueError(f"Unsupported bbox value: {value!r}")


@dataclass
class RecognizedBlock:
    text: str
    bbox: BBox | None = None
    confidence: float | None = None
    kind: Literal["text", "formula", "unknown"] = "unknown"
    reason: str | None = None


@dataclass
class RecognitionResult:
    provider: str
    transcript: str
    blocks: list[RecognizedBlock] = field(default_factory=list)
    ignored_blocks: list[RecognizedBlock] = field(default_factory=list)
    visual_bboxes: list[BBox] = field(default_factory=list)
    visual_bboxes_verified: bool = False
    visual_bbox_source: str = "none"
    handwriting_bboxes: list[BBox] = field(default_factory=list)
    handwriting_suspected: bool = False
    formula_verified: bool = False
    formula_specialist_used: bool = False
    formula_count: int = 0
    confidence: float | None = None
    unverified_prelabel: bool = False
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualAsset:
    asset_id: str
    source_bbox: BBox
    crop_path: str
    crop_sha256: str
    kind: str = "question_visual"
    extraction: str = "direct_source_crop"
    bbox_source: str = "unknown"
    bbox_verified: bool = False
    preserve_source_pixels: bool = True
    redraw_used: bool = False
    fallback: bool = False
    edge_complete: bool = True


@dataclass
class SourceBlock:
    block_id: str
    kind: Literal["text", "formula", "unknown"]
    text: str
    source_bbox: BBox | None
    confidence: float | None = None
    disposition: Literal["emitted", "ignored", "review_required"] = "emitted"
    reason: str | None = None


@dataclass
class QuestionDocument:
    schema_version: str
    sample_id: str
    source_image: str
    source_sha256: str
    provider: str
    canonical_text: str
    display_text: str
    question_number: str | None
    stem: str
    options: list[dict[str, str]]
    visual_assets: list[VisualAsset]
    handwriting_detected: bool
    formula_verified: bool
    formula_specialist_used: bool
    formula_count: int
    warnings: list[str] = field(default_factory=list)
    source_blocks: list[SourceBlock] = field(default_factory=list)
    source_resolution: dict[str, Any] | None = None
    scoring_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationIssue:
    code: str
    severity: Literal["review", "reject"]
    message: str


@dataclass
class ProcessReport:
    schema_version: str
    run_id: str
    sample_id: str
    status: Status
    source_image: str
    output_image: str
    question_document: str
    provider: str
    elapsed_seconds: float
    timings_ms: dict[str, float]
    issues: list[ValidationIssue]
    source_sha256: str
    output_sha256: str
    output_width: int
    output_height: int
    input_source_image: str | None = None
    input_source_sha256: str | None = None
    source_resolution: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StoredSample:
    directory: Path
    output_path: Path
    document_path: Path
    report_path: Path
    asset_directory: Path
