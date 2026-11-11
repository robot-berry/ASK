from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

from test7_ocr.handwriting_review import HandwritingMaskStore, SELF_BUILT_MASK_SCHEMA
from test7_ocr.self_built_intake import SelfBuiltIntakeStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DATASET_NAME = "testzip_math_physics_300_v1"
DATASET_ROOT = PROJECT_ROOT.parents[1] / "datasets" / DATASET_NAME
SELF_BUILT_DATASET_NAME = "test7_ocr_self_built_v1"
SELF_BUILT_DATASET_ROOT = WORKSPACE_ROOT / "datasets" / SELF_BUILT_DATASET_NAME
GOLD_REVIEW_DIR = (
    PROJECT_ROOT / "data" / "annotations" / "testzip_math_physics_ocr_gold" / "drafts"
)
OUTPUT_REVIEW_ROOT = PROJECT_ROOT / "data" / "annotations" / "output_quality_reviews"
ANNOTATION_ROOT = PROJECT_ROOT / "data" / "annotations" / "handwriting_masks"
TASK_DIR = ANNOTATION_ROOT / "tasks"
MASK_DIR = ANNOTATION_ROOT / "masks"
MANIFEST_PATH = PROJECT_ROOT / "data" / "splits" / "testzip_handwriting_mask_eval_manifest.jsonl"
SUMMARY_PATH = PROJECT_ROOT / "data" / "splits" / "testzip_handwriting_mask_eval_summary.json"
ALLOWED_MASK_VALUES = {0, 1, 2, 3, 4, 255}
CLASS_VALUES = {
    "background": 0,
    "printed_content": 1,
    "student_handwriting": 2,
    "teacher_mark": 3,
    "overlap_unrecoverable": 4,
    "uncertain": 255,
}
CHECK_KEYS = {
    "source_hash_verified",
    "printed_content_preserved",
    "student_handwriting_complete",
    "teacher_marks_classified",
    "overlap_and_uncertain_checked",
}


def build_mask_store() -> HandwritingMaskStore:
    intake = SelfBuiltIntakeStore(
        SELF_BUILT_DATASET_ROOT,
        PROJECT_ROOT / "data" / "annotations" / "self_built_intake" / "records",
        PROJECT_ROOT / "data" / "splits" / "self_built_intake_summary.json",
    )
    return HandwritingMaskStore(TASK_DIR, MASK_DIR, DATASET_ROOT, intake)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def gold_tasks() -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in sorted(GOLD_REVIEW_DIR.glob("*.json")):
        task = read_json(path)
        sample_id = str(task.get("sample_id") or "")
        if sample_id:
            tasks[sample_id] = task
    return tasks


def selection_evidence(tasks: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    evidence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample_id, task in tasks.items():
        prior = task.get("prior_manual_evidence") or {}
        error_types = {str(value) for value in prior.get("error_types") or []}
        if "handwriting_contamination" not in error_types:
            continue
        note = str(prior.get("notes") or "人工记录确认存在手写污染").strip()
        evidence[sample_id].append(
            {
                "source": "ocr_gold_prior_manual_evidence",
                "kind": "handwriting_contamination",
                "note": note,
            }
        )

    if OUTPUT_REVIEW_ROOT.is_dir():
        for path in sorted(OUTPUT_REVIEW_ROOT.glob("*/*.json")):
            try:
                review = read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if review.get("verdict") != "fail":
                continue
            notes = [str(value).strip() for value in review.get("notes") or [] if str(value).strip()]
            handwriting_issue = (review.get("issues") or {}).get("handwriting_visible") is True
            handwriting_note = any("手写" in value or "批改" in value for value in notes)
            if not handwriting_issue and not handwriting_note:
                continue
            sample_id = str(review.get("sample_id") or "")
            if sample_id not in tasks:
                continue
            note = " | ".join(notes) or "成品人工验收确认存在手写或批改污染"
            candidate = {
                "source": "output_quality_review",
                "kind": "handwriting_contamination",
                "note": f"{review.get('run_id')}: {note}",
            }
            if candidate not in evidence[sample_id]:
                evidence[sample_id].append(candidate)
    return evidence


def default_task(source_task: dict[str, Any], evidence: list[dict[str, str]]) -> dict[str, Any]:
    source = source_task["source"]
    return {
        "schema_version": "handwriting_mask_review_v1",
        "sample_id": source_task["sample_id"],
        "partition": "frozen_test",
        "training_allowed": False,
        "source": {
            "dataset": DATASET_NAME,
            "image_path": source["image_path"],
            "image_sha256": source["image_sha256"],
            "width": int(source["width"]),
            "height": int(source["height"]),
            "subject": source["subject"],
            "document_group": source["document_group"],
        },
        "selection_evidence": evidence,
        "annotation": {
            "status": "pending",
            "mask_path": None,
            "mask_sha256": None,
            "legend_version": "handwriting_mask_v1",
            "class_values": dict(CLASS_VALUES),
            "pixel_counts": {key: 0 for key in CLASS_VALUES},
        },
        "review": {
            "annotator_id": None,
            "reviewer_id": None,
            "checks": {key: None for key in sorted(CHECK_KEYS)},
            "notes": [],
            "signoffs": [],
            "history": [],
            "updated_at_utc": None,
        },
    }


def immutable_projection(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": task.get("schema_version"),
        "sample_id": task.get("sample_id"),
        "partition": task.get("partition"),
        "training_allowed": task.get("training_allowed"),
        "source": task.get("source"),
    }


def mask_pixel_counts(path: Path) -> dict[str, int]:
    with Image.open(path) as image:
        mask = image.convert("L")
        histogram = mask.histogram()
    return {key: int(histogram[value]) for key, value in CLASS_VALUES.items()}


def validate_task(task: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    sample_id = str(expected["sample_id"])
    if immutable_projection(task) != immutable_projection(expected):
        errors.append(f"{sample_id}: immutable source fields changed")
    if task.get("selection_evidence") != expected.get("selection_evidence"):
        errors.append(f"{sample_id}: selection evidence is stale")
    source = task.get("source") or {}
    source_path = (DATASET_ROOT / str(source.get("image_path") or "")).resolve()
    if DATASET_ROOT.resolve() not in source_path.parents or not source_path.is_file():
        errors.append(f"{sample_id}: source image is missing or escaped the dataset root")
    else:
        if sha256_file(source_path) != source.get("image_sha256"):
            errors.append(f"{sample_id}: source SHA-256 mismatch")
        with Image.open(source_path) as image:
            if image.size != (int(source.get("width") or 0), int(source.get("height") or 0)):
                errors.append(f"{sample_id}: source dimensions mismatch")
    annotation = task.get("annotation") or {}
    if annotation.get("legend_version") != "handwriting_mask_v1":
        errors.append(f"{sample_id}: invalid legend version")
    if annotation.get("class_values") != CLASS_VALUES:
        errors.append(f"{sample_id}: class values changed")
    status = annotation.get("status")
    if status not in {"pending", "draft", "annotated", "verified", "adjudication_required", "rejected"}:
        errors.append(f"{sample_id}: invalid annotation status")
    mask_relative = annotation.get("mask_path")
    if mask_relative is None:
        if status not in {"pending", "rejected"}:
            errors.append(f"{sample_id}: {status} task has no mask")
    else:
        mask_path = (ANNOTATION_ROOT / str(mask_relative)).resolve()
        if ANNOTATION_ROOT.resolve() not in mask_path.parents or not mask_path.is_file():
            errors.append(f"{sample_id}: mask is missing or escaped the annotation root")
        else:
            if sha256_file(mask_path) != annotation.get("mask_sha256"):
                errors.append(f"{sample_id}: mask SHA-256 mismatch")
            with Image.open(mask_path) as mask:
                if mask.mode != "L":
                    errors.append(f"{sample_id}: mask must use 8-bit L mode")
                if mask.size != (int(source.get("width") or 0), int(source.get("height") or 0)):
                    errors.append(f"{sample_id}: mask dimensions mismatch")
                colors = mask.getcolors(maxcolors=max(1, mask.width * mask.height)) or []
                observed = {int(value) for _count, value in colors}
                if not observed <= ALLOWED_MASK_VALUES:
                    errors.append(f"{sample_id}: mask contains invalid class values {sorted(observed - ALLOWED_MASK_VALUES)}")
            if annotation.get("pixel_counts") != mask_pixel_counts(mask_path):
                errors.append(f"{sample_id}: stored pixel counts do not match mask")
    review = task.get("review") or {}
    checks = review.get("checks") or {}
    if set(checks) != CHECK_KEYS or any(value not in {True, False, None} for value in checks.values()):
        errors.append(f"{sample_id}: review checks are incomplete")
    if status == "verified":
        annotator = str(review.get("annotator_id") or "").strip()
        reviewer = str(review.get("reviewer_id") or "").strip()
        if not annotator or not reviewer or annotator == reviewer:
            errors.append(f"{sample_id}: verified mask requires two distinct identities")
        if any(value is not True for value in checks.values()):
            errors.append(f"{sample_id}: verified mask requires every check to be true")
    return errors


def manifest_row(task: dict[str, Any]) -> dict[str, Any]:
    annotation = task["annotation"]
    review = task["review"]
    return {
        "sample_id": task["sample_id"],
        "dataset": task["source"]["dataset"],
        "partition": task["partition"],
        "training_allowed": task["training_allowed"],
        "evaluation_allowed": task.get("evaluation_allowed", True),
        "subject": task["source"]["subject"],
        "image_path": task["source"]["image_path"],
        "image_sha256": task["source"]["image_sha256"],
        "selection_evidence_count": len(task["selection_evidence"]),
        "status": annotation["status"],
        "mask_path": annotation["mask_path"],
        "mask_sha256": annotation["mask_sha256"],
        "annotator_id": review["annotator_id"],
        "reviewer_id": review["reviewer_id"],
    }


def prepare() -> tuple[dict[str, Any], list[str]]:
    tasks = gold_tasks()
    evidence = selection_evidence(tasks)
    TASK_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    mask_store = build_mask_store()
    created = mask_store.sync_from_intake()
    prepared: list[dict[str, Any]] = []
    errors: list[str] = []
    for sample_id in sorted(evidence):
        expected = default_task(tasks[sample_id], evidence[sample_id])
        task_path = TASK_DIR / f"{sample_id}.json"
        if task_path.is_file():
            task = read_json(task_path)
            # Output review is an append-only source of selection evidence.
            # Keep an existing human mask and its sign-offs intact, while
            # synchronizing the evidence list so a newly recorded failure does
            # not make an otherwise valid task stale.
            if task.get("selection_evidence") != expected["selection_evidence"]:
                task["selection_evidence"] = expected["selection_evidence"]
                write_json_atomic(task_path, task)
        else:
            task = expected
            write_json_atomic(task_path, task)
            created += 1
        errors.extend(validate_task(task, expected))
        prepared.append(task)

    for path in sorted(TASK_DIR.glob("*.json")):
        if path.stem in evidence:
            continue
        try:
            task = mask_store.load(path.stem)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.stem}: {exc}")
            continue
        if task.get("schema_version") != SELF_BUILT_MASK_SCHEMA:
            errors.append(f"{path.stem}: unexpected task outside the evidence-selected or self-built sets")
            continue
        prepared.append(task)

    manifest = [manifest_row(task) for task in prepared]
    write_text_atomic(
        MANIFEST_PATH,
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in manifest),
    )
    statuses = Counter(row["status"] for row in manifest)
    datasets = Counter(row["dataset"] for row in manifest)
    partitions = Counter(row["partition"] for row in manifest)
    summary = {
        "schema_version": "handwriting_mask_eval_summary_v1",
        "datasets": dict(sorted(datasets.items())),
        "partitions": dict(sorted(partitions.items())),
        "training_allowed": False,
        "target_minimum_images": 100,
        "selected_images": len(prepared),
        "remaining_external_or_self_captured_images": max(0, 100 - len(prepared)),
        "created_tasks_this_run": created,
        "status_counts": dict(sorted(statuses.items())),
        "verified_masks": statuses.get("verified", 0),
        "verified_frozen_masks": sum(
            row["status"] == "verified" and row["partition"] == "frozen_test" for row in manifest
        ),
        "verified_quarantined_masks": sum(
            row["status"] == "verified" and row["partition"] == "quarantine" for row in manifest
        ),
        "gold_complete": statuses.get("verified", 0) >= 100,
        "note": "Legacy evidence-selected tasks remain frozen evaluation data. Registered handwriting captures stay quarantined until two-person mask review and D0-E dataset freeze. Model predictions cannot become Gold without human sign-offs.",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(SUMMARY_PATH, summary)
    return summary, errors


def check() -> tuple[dict[str, Any], list[str]]:
    tasks = gold_tasks()
    evidence = selection_evidence(tasks)
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    mask_store = build_mask_store()
    try:
        mask_store.sync_from_intake()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    for sample_id in sorted(evidence):
        task_path = TASK_DIR / f"{sample_id}.json"
        if not task_path.is_file():
            errors.append(f"{sample_id}: task file is missing")
            continue
        task = read_json(task_path)
        expected = default_task(tasks[sample_id], evidence[sample_id])
        errors.extend(validate_task(task, expected))
        rows.append(manifest_row(task))
    for path in sorted(TASK_DIR.glob("*.json")):
        if path.stem in evidence:
            continue
        try:
            task = mask_store.load(path.stem)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.stem}: {exc}")
            continue
        if task.get("schema_version") != SELF_BUILT_MASK_SCHEMA:
            errors.append(f"{path.stem}: unexpected task outside the evidence-selected or self-built sets")
            continue
        rows.append(manifest_row(task))
    statuses = Counter(row["status"] for row in rows)
    datasets = Counter(row["dataset"] for row in rows)
    partitions = Counter(row["partition"] for row in rows)
    summary = {
        "selected_images": len(rows),
        "task_files": len(rows),
        "verified_masks": statuses.get("verified", 0),
        "status_counts": dict(sorted(statuses.items())),
        "dataset_counts": dict(sorted(datasets.items())),
        "partition_counts": dict(sorted(partitions.items())),
        "errors": len(errors),
    }
    return summary, errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or validate the frozen handwriting pixel-mask evaluation queue"
    )
    parser.add_argument("--check", action="store_true", help="Validate without rewriting tasks")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless at least 100 masks have two-person verified status",
    )
    args = parser.parse_args()
    summary, errors = check() if args.check else prepare()
    selected = int(summary.get("selected_images", summary.get("task_files", 0)))
    verified = int(summary.get("verified_masks", 0))
    if args.require_complete and (selected < 100 or verified < 100):
        errors.append(
            f"handwriting mask Gold incomplete: selected={selected}/100, verified={verified}/100"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
