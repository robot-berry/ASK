#!/usr/bin/env python3
"""Verify the 300-question source-evidence contract without calling a model.

This deliberately does not relabel source-pixel fallback as OCR accuracy.
It verifies, serially, that every benchmark question has an intact, readable,
hash-matching source image that the Test6 fail-closed path can preserve as the
semantic reference whenever the text transcript loses two-dimensional meaning.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATASET_ROOT = PROJECT_ROOT.parents[1] / "datasets" / "testzip_math_physics_300_v1"
MANIFEST = DATASET_ROOT / "manifest.jsonl"
BASELINE = HERE / "ocr_accuracy_300_20260902" / "FULL_SEMANTIC_AUDIT_300.json"
JSON_OUT = HERE / "ocr_accuracy_300_20260902" / "SEMANTIC_EVIDENCE_COVERAGE_300.json"
MD_OUT = HERE / "ocr_accuracy_300_20260902" / "SEMANTIC_EVIDENCE_COVERAGE_300.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with MANIFEST.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            # The user-facing audit is the physical 300-image set, including
            # the two manifest-marked duplicates. Duplicates are still real
            # uploads and must independently retain their source evidence.
            rows.append(row)
    return rows


def main() -> int:
    rows = load_manifest()
    details: list[dict[str, Any]] = []
    subject_counts: dict[str, dict[str, int]] = {}
    dataset_root = DATASET_ROOT.resolve()

    for serial_index, row in enumerate(rows, 1):
        relative = str(row.get("image_path") or "")
        candidate = (dataset_root / relative).resolve()
        inside_dataset = True
        try:
            candidate.relative_to(dataset_root)
        except ValueError:
            inside_dataset = False

        errors: list[str] = []
        if not inside_dataset:
            errors.append("path_outside_dataset")
        if not candidate.is_file():
            errors.append("source_missing")

        actual_sha256 = ""
        actual_size: list[int] | None = None
        actual_format = ""
        if not errors:
            actual_sha256 = sha256_file(candidate)
            if actual_sha256 != str(row.get("image_sha256") or ""):
                errors.append("sha256_mismatch")
            try:
                with Image.open(candidate) as image:
                    image.load()
                    actual_size = [image.width, image.height]
                    actual_format = str(image.format or "")
                if actual_size != [int(row.get("width") or 0), int(row.get("height") or 0)]:
                    errors.append("dimension_mismatch")
                if actual_format != str(row.get("actual_format") or ""):
                    errors.append("format_mismatch")
            except OSError:
                errors.append("image_decode_failed")

        subject = str(row.get("subject") or "unknown")
        counters = subject_counts.setdefault(subject, {"total": 0, "covered": 0})
        counters["total"] += 1
        covered = not errors
        counters["covered"] += int(covered)
        details.append(
            {
                "serial_index": serial_index,
                "question_id": row.get("question_id"),
                "subject": subject,
                "image_path": relative,
                "source_evidence_covered": covered,
                "errors": errors,
                "sha256": actual_sha256,
                "size": actual_size,
                "format": actual_format,
            }
        )

    covered_count = sum(item["source_evidence_covered"] for item in details)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    pure_ocr = baseline["totals"]
    result = {
        "title": "300题语义证据兜底覆盖核验",
        "method": (
            "逐题串行校验原题文件可读取、尺寸/格式与清单一致、SHA-256未变化；"
            "生产链路将该完整题面保存为语义校验图。未调用答题模型。"
        ),
        "metric_boundary": {
            "pure_ocr_text_semantic_complete": pure_ocr,
            "source_pixel_evidence_coverage": {
                "total": len(details),
                "covered": covered_count,
                "uncovered": len(details) - covered_count,
                "coverage_rate": covered_count / len(details) if details else 0.0,
            },
            "important_note": (
                "100%仅表示完整原题像素证据不丢失，不表示OCR转写字符或公式达到100%。"
            ),
        },
        "subjects": subject_counts,
        "details": details,
    }
    JSON_OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = pure_ocr["overall"]
    lines = [
        "# 300题语义证据兜底覆盖核验",
        "",
        "## 结论",
        "",
        f"- 完整原题像素证据覆盖：**{covered_count}/{len(details)}（{covered_count / len(details):.2%}）**。",
        f"- 既有纯OCR文字语义完整：**{overall['complete']}/{overall['total']}（{overall['complete_rate']:.2%}）**。",
        "- 两个指标不可混用：前者保证分析链路不丢题意，后者衡量文字转写本身。",
        "",
        "## 分科覆盖",
        "",
        "| 学科 | 覆盖 | 总数 | 覆盖率 |",
        "|---|---:|---:|---:|",
    ]
    labels = {"mathematics": "数学", "physics": "物理"}
    for subject in ("mathematics", "physics"):
        counters = subject_counts.get(subject, {"total": 0, "covered": 0})
        rate = counters["covered"] / counters["total"] if counters["total"] else 0.0
        lines.append(
            f"| {labels[subject]} | {counters['covered']} | {counters['total']} | {rate:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 核验方法",
            "",
            "按数据集清单顺序逐题串行执行：路径边界检查、文件存在性、图片解码、尺寸、格式和SHA-256校验。",
            "本核验不调用大模型，也不把原图兜底冒充为OCR文字准确率。",
            "",
            f"机器可读明细：`{JSON_OUT.name}`",
        ]
    )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["metric_boundary"], ensure_ascii=False, indent=2))
    return 0 if covered_count == len(details) == 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
