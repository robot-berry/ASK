from __future__ import annotations

import argparse
import json
from pathlib import Path

from test7_ocr.capture_pair_review import CapturePairStore
from test7_ocr.handwriting_review import HandwritingMaskStore
from test7_ocr.question_boundary_review import QuestionBoundaryStore
from test7_ocr.self_built_freeze import SelfBuiltFreezeBuilder
from test7_ocr.self_built_intake import SelfBuiltIntakeStore
from test7_ocr.storage import write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
READINESS_PATH = PROJECT_ROOT / "data" / "splits" / "self_built_freeze_readiness.json"
FREEZE_ROOT = PROJECT_ROOT / "data" / "frozen"


def build_freezer() -> SelfBuiltFreezeBuilder:
    primary_dataset_root = WORKSPACE_ROOT / "datasets" / "gaokao_physics_300_v1_0_1"
    mask_dataset_root = WORKSPACE_ROOT / "datasets" / "testzip_math_physics_300_v1"
    intake = SelfBuiltIntakeStore(
        WORKSPACE_ROOT / "datasets" / "test7_ocr_self_built_v1",
        PROJECT_ROOT / "data" / "annotations" / "self_built_intake" / "records",
        PROJECT_ROOT / "data" / "splits" / "self_built_intake_summary.json",
    )
    masks = HandwritingMaskStore(
        PROJECT_ROOT / "data" / "annotations" / "handwriting_masks" / "tasks",
        PROJECT_ROOT / "data" / "annotations" / "handwriting_masks" / "masks",
        mask_dataset_root,
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
    return SelfBuiltFreezeBuilder(
        primary_dataset_root=primary_dataset_root,
        primary_dataset_name="gaokao_physics_300_v1",
        ocr_gold_path=PROJECT_ROOT
        / "data"
        / "gold"
        / "gaokao_physics_300"
        / "ocr_gold.jsonl",
        intake=intake,
        masks=masks,
        boundaries=boundaries,
        capture_pairs=pairs,
        freeze_root=FREEZE_ROOT,
        targets={"ocr_gold": 300},
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit or immutably freeze the reviewed test7-OCR self-built evaluation data"
    )
    parser.add_argument(
        "--version",
        help="Create an immutable version such as test7-self-built-v2026.09.06",
    )
    parser.add_argument(
        "--verify-version",
        help="Verify an existing immutable version without changing it",
    )
    parser.add_argument(
        "--require-targets",
        action="store_true",
        help="Fail the readiness check unless all D0-B/C/D quantities and reviews are complete",
    )
    args = parser.parse_args()
    freezer = build_freezer()
    if args.version and args.verify_version:
        parser.error("--version and --verify-version cannot be used together")
    if args.verify_version:
        version = freezer._version(args.verify_version)
        manifest, errors = freezer.verify_freeze(FREEZE_ROOT / version)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        for error in errors:
            print(f"ERROR: {error}")
        return 1 if errors else 0
    if args.version:
        try:
            manifest = freezer.freeze(args.version)
        except (FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"ERROR: {exc}")
            return 1
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    summary, errors, _payload = freezer.audit(require_targets=args.require_targets)
    write_json_atomic(READINESS_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
