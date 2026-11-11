from __future__ import annotations

import argparse
import json
from pathlib import Path

from test7_ocr.capture_pair_review import CapturePairStore
from test7_ocr.handwriting_review import HandwritingMaskStore
from test7_ocr.question_boundary_review import QuestionBoundaryStore
from test7_ocr.self_built_batch import SelfBuiltBatchImporter
from test7_ocr.self_built_intake import SelfBuiltIntakeStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]


def build_importer() -> SelfBuiltBatchImporter:
    intake = SelfBuiltIntakeStore(
        WORKSPACE_ROOT / "datasets" / "test7_ocr_self_built_v1",
        PROJECT_ROOT / "data" / "annotations" / "self_built_intake" / "records",
        PROJECT_ROOT / "data" / "splits" / "self_built_intake_summary.json",
    )
    masks = HandwritingMaskStore(
        PROJECT_ROOT / "data" / "annotations" / "handwriting_masks" / "tasks",
        PROJECT_ROOT / "data" / "annotations" / "handwriting_masks" / "masks",
        WORKSPACE_ROOT / "datasets" / "testzip_math_physics_300_v1",
        intake,
    )
    boundaries = QuestionBoundaryStore(
        PROJECT_ROOT / "data" / "annotations" / "question_boundaries" / "tasks",
        PROJECT_ROOT / "data" / "splits" / "question_boundary_manifest.jsonl",
        PROJECT_ROOT / "data" / "splits" / "question_boundary_summary.json",
        intake,
    )
    pairs = CapturePairStore(
        PROJECT_ROOT / "data" / "annotations" / "capture_pairs" / "tasks",
        PROJECT_ROOT / "data" / "splits" / "capture_pair_manifest.jsonl",
        PROJECT_ROOT / "data" / "splits" / "capture_pair_summary.json",
        intake,
    )
    return SelfBuiltBatchImporter(
        intake,
        PROJECT_ROOT / "data" / "annotations" / "self_built_intake" / "batches",
        [masks.sync_from_intake, boundaries.sync_from_intake, pairs.sync_from_intake],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight or explicitly import a CSV batch into self-built quarantine"
    )
    parser.add_argument("manifest", type=Path, help="CSV manifest; image paths are relative to it")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Register the batch after a whole-batch preflight; omission is read-only dry-run",
    )
    args = parser.parse_args()
    try:
        report = build_importer().run(args.manifest, apply=args.apply)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] in {"ready", "applied"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
