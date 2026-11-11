from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import Question


EXPECTED_DATASET_ID = "testzip_math_physics_300_v1"
SUBJECT_MAP = {"mathematics": "math", "physics": "physics"}


def balanced_sample_rows(
    manifest: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Select a deterministic subject-balanced, Test-group-spread sample."""

    if count < 1:
        raise ValueError("平衡样本数量必须大于 0")
    targets = {
        "mathematics": (count + 1) // 2,
        "physics": count // 2,
    }
    selected_by_subject: dict[str, list[dict[str, Any]]] = {}
    for subject, target in targets.items():
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in manifest:
            if (
                row.get("include_in_benchmark") is True
                and row.get("subject") == subject
            ):
                groups.setdefault(str(row.get("source_group") or ""), []).append(row)
        for rows in groups.values():
            rows.sort(key=lambda row: int(row.get("source_index") or 0))
        ordered_groups = [groups[key] for key in sorted(groups)]
        subject_rows: list[dict[str, Any]] = []
        depth = 0
        while len(subject_rows) < target:
            added = False
            for rows in ordered_groups:
                if depth < len(rows):
                    subject_rows.append(rows[depth])
                    added = True
                    if len(subject_rows) == target:
                        break
            if not added:
                break
            depth += 1
        if len(subject_rows) != target:
            raise ValueError(f"{subject} 可用题目不足 {target} 道")
        selected_by_subject[subject] = subject_rows

    selected: list[dict[str, Any]] = []
    for index in range(max(targets.values())):
        for subject in ("mathematics", "physics"):
            rows = selected_by_subject[subject]
            if index < len(rows):
                selected.append(rows[index])
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} 第 {line_number} 行必须是 JSON 对象")
        rows.append(value)
    return rows


def _safe_member(dataset_root: Path, relative_path: str) -> Path:
    candidate = (dataset_root / relative_path).resolve()
    root = dataset_root.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"数据集路径越界：{relative_path}")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ocr_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        item_id = str(row.get("question_id") or row.get("sample_id") or row.get("id") or "")
        text = str(row.get("transcript") or row.get("ocr_text") or row.get("text") or "").strip()
        if item_id and text:
            result[item_id] = row
    return result


def _directory_ocr_record(root: Path, question_id: str) -> dict[str, Any] | None:
    if any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in question_id):
        raise ValueError(f"不安全的题目 ID：{question_id}")
    item_dir = root / question_id
    document_path = item_dir / "question_document.json"
    report_path = item_dir / "report.json"
    if not document_path.is_file():
        return None
    document = json.loads(document_path.read_text(encoding="utf-8-sig"))
    if document.get("sample_id") != question_id:
        raise ValueError(f"OCR 文档题号不匹配：{document_path}")
    report: dict[str, Any] = {}
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    timings = report.get("timings_ms", {})
    input_ready_keys = (
        "load_and_preprocess",
        "recognition",
        "structure_and_formula_safety",
        "direct_visual_crops",
    )
    input_ready_ms = sum(float(timings.get(key) or 0.0) for key in input_ready_keys)
    if not input_ready_ms and report.get("elapsed_seconds") is not None:
        input_ready_ms = float(report["elapsed_seconds"]) * 1000
    return {
        "question_id": question_id,
        "transcript": document.get("canonical_text") or document.get("display_text"),
        "total_ms": input_ready_ms,
        "source_sha256": str(document.get("source_sha256") or "").lower(),
        "source_quality": {
            "provider": document.get("provider"),
            "report_status": report.get("status"),
            "formula_verified": document.get("formula_verified"),
            "warnings": document.get("warnings", []),
            "model_input_ready_ms": round(input_ready_ms, 3),
            "full_ocr_render_and_validation_ms": (
                round(float(report["elapsed_seconds"]) * 1000, 3)
                if report.get("elapsed_seconds") is not None
                else None
            ),
            "pipeline_initialization_ms": report.get("pipeline_initialization", {})
            .get("timings_ms", {})
            .get("total"),
            "question_document": str(document_path),
            "report": str(report_path) if report_path.is_file() else None,
        },
    }


def prepare_testzip_items(
    *,
    dataset_root: Path,
    ocr_results: Path,
    phase: str,
    stage: str,
    limit: int | None = None,
    verify_image_hashes: bool = True,
) -> tuple[list[Question], dict[str, Any]]:
    root = dataset_root.resolve()
    summary_path = root / "dataset_summary.json"
    manifest_path = root / "manifest.jsonl"
    smoke_path = root / "smoke_20.jsonl"
    for required in (summary_path, manifest_path, smoke_path):
        if not required.is_file():
            raise ValueError(f"缺少评测输入文件：{required}")
    if not ocr_results.exists():
        raise ValueError(f"缺少 OCR 输入来源：{ocr_results}")
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    if summary.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError(
            f"数据集 ID 必须为 {EXPECTED_DATASET_ID}，实际为 {summary.get('dataset_id')}"
        )
    if summary.get("benchmark_policy", {}).get("training_allowed") is not False:
        raise ValueError("数据集没有明确禁止训练，拒绝作为冻结测试集运行")

    manifest = _read_jsonl(manifest_path)
    manifest_by_id = {str(row["question_id"]): row for row in manifest}
    if phase == "full":
        selected = [row for row in manifest if row.get("include_in_benchmark") is True]
    elif phase == "sample":
        selected = balanced_sample_rows(manifest, limit or 50)
    else:
        smoke_rows = _read_jsonl(smoke_path)
        selected = smoke_rows[:1] if phase == "probe" else smoke_rows
    if phase not in {"probe", "smoke", "sample", "full"}:
        raise ValueError("phase 只支持 probe、smoke、sample 或 full")
    if limit is not None and phase != "sample":
        if limit < 1:
            raise ValueError("limit 必须大于 0")
        selected = selected[:limit]

    resolved_ocr = ocr_results.resolve()
    ocr_by_id = _ocr_index(resolved_ocr) if resolved_ocr.is_file() else {}
    items: list[Question] = []
    missing_ocr: list[str] = []
    missing_images: list[str] = []
    hash_mismatches: list[str] = []
    ocr_image_hash_mismatches: list[str] = []
    for selection in selected:
        question_id = str(selection["question_id"])
        manifest_row = manifest_by_id.get(question_id)
        if not manifest_row:
            raise ValueError(f"smoke 条目不在主 manifest 中：{question_id}")
        ocr_row = (
            _directory_ocr_record(resolved_ocr, question_id)
            if resolved_ocr.is_dir()
            else ocr_by_id.get(question_id)
        )
        if not ocr_row:
            missing_ocr.append(question_id)
            continue
        relative_image = str(manifest_row["image_path"])
        image_path = _safe_member(root, relative_image)
        if not image_path.is_file():
            missing_images.append(question_id)
            continue
        expected_hash = str(manifest_row.get("image_sha256") or "").lower()
        ocr_source_hash = str(ocr_row.get("source_sha256") or "").lower()
        if ocr_source_hash and ocr_source_hash != expected_hash:
            ocr_image_hash_mismatches.append(question_id)
            continue
        if verify_image_hashes and _sha256_file(image_path) != expected_hash:
            hash_mismatches.append(question_id)
            continue
        subject = SUBJECT_MAP.get(str(manifest_row.get("subject")))
        if not subject:
            raise ValueError(f"未知科目：{manifest_row.get('subject')}")
        items.append(
            Question.from_mapping(
                {
                    "id": question_id,
                    "subject": subject,
                    "stage": stage,
                    "question": str(
                        ocr_row.get("transcript")
                        or ocr_row.get("ocr_text")
                        or ocr_row.get("text")
                    ),
                    "question_source": "ocr_only",
                    "source_image": str(image_path),
                    "dataset_id": EXPECTED_DATASET_ID,
                    "image_sha256": expected_hash,
                    "upstream_latency_ms": ocr_row.get("total_ms")
                    or ocr_row.get("latency_ms"),
                    "source_quality": ocr_row.get("source_quality", {}),
                }
            )
        )

    preparation = {
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_root": str(root),
        "dataset_status": summary.get("status"),
        "gold_verified_count": summary.get("verified_gold_count", 0),
        "formal_correctness_scoring_count": summary.get(
            "formal_correctness_scoring_count", 0
        ),
        "phase": phase,
        "stage_override": stage,
        "stage_source": "operator_override_dataset_metadata_unknown",
        "selection_strategy": (
            "balanced_subject_and_test_group"
            if phase == "sample"
            else "frozen_phase_selection"
        ),
        "selected_count": len(selected),
        "prepared_count": len(items),
        "ocr_results": str(resolved_ocr),
        "ocr_source_type": "question_document_directory" if resolved_ocr.is_dir() else "jsonl",
        "verify_image_hashes": verify_image_hashes,
        "missing_ocr": missing_ocr,
        "missing_images": missing_images,
        "hash_mismatches": hash_mismatches,
        "ocr_image_hash_mismatches": ocr_image_hash_mismatches,
        "answers_or_gold_loaded": False,
        "training_allowed": False,
    }
    if len(items) != len(selected):
        raise ValueError(
            "评测准备不完整："
            f"选择 {len(selected)} 题，成功准备 {len(items)} 题；"
            f"缺 OCR {len(missing_ocr)}，缺图片 {len(missing_images)}，"
            f"原图哈希不符 {len(hash_mismatches)}，"
            f"OCR 来源哈希不符 {len(ocr_image_hash_mismatches)}"
        )
    return items, preparation
