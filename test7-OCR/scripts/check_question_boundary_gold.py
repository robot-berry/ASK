from __future__ import annotations

import argparse
import json
from pathlib import Path

from test7_ocr.question_boundary_review import QuestionBoundaryStore, boundary_target_errors
from test7_ocr.self_built_intake import SelfBuiltIntakeStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]


def build_store() -> QuestionBoundaryStore:
    intake = SelfBuiltIntakeStore(
        WORKSPACE_ROOT / "datasets" / "test7_ocr_self_built_v1",
        PROJECT_ROOT / "data" / "annotations" / "self_built_intake" / "records",
        PROJECT_ROOT / "data" / "splits" / "self_built_intake_summary.json",
    )
    return QuestionBoundaryStore(
        PROJECT_ROOT / "data" / "annotations" / "question_boundaries" / "tasks",
        PROJECT_ROOT / "data" / "splits" / "question_boundary_manifest.jsonl",
        PROJECT_ROOT / "data" / "splits" / "question_boundary_summary.json",
        intake,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate quarantined multi-question page boundary Gold and its two-person sign-offs"
    )
    parser.add_argument(
        "--require-targets",
        action="store_true",
        help="Fail unless 60 pages, 300 questions, 30 single-column pages and 30 double-column pages are verified",
    )
    args = parser.parse_args()
    store = build_store()
    summary, errors = store.check()
    if not errors:
        store.write_indexes()
        summary = store.summary()
    if args.require_targets:
        errors.extend(boundary_target_errors(summary))
    output = {**summary, "errors": len(errors)}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
