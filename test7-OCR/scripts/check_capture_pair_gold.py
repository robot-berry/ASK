from __future__ import annotations

import argparse
import json
from pathlib import Path

from test7_ocr.capture_pair_review import (
    CapturePairStore,
    capture_pair_target_errors,
)
from test7_ocr.self_built_intake import SelfBuiltIntakeStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]


def build_store() -> CapturePairStore:
    intake = SelfBuiltIntakeStore(
        WORKSPACE_ROOT / "datasets" / "test7_ocr_self_built_v1",
        PROJECT_ROOT / "data" / "annotations" / "self_built_intake" / "records",
        PROJECT_ROOT / "data" / "splits" / "self_built_intake_summary.json",
    )
    return CapturePairStore(
        PROJECT_ROOT / "data" / "annotations" / "capture_pairs" / "tasks",
        PROJECT_ROOT / "data" / "splits" / "capture_pair_manifest.jsonl",
        PROJECT_ROOT / "data" / "splits" / "capture_pair_summary.json",
        intake,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate clean-page/photo and clean-page/handwriting pairs with two-person sign-offs"
    )
    parser.add_argument(
        "--require-targets",
        action="store_true",
        help="Fail unless all verified pair, device, and capture-condition targets are complete",
    )
    args = parser.parse_args()
    store = build_store()
    summary, errors = store.check()
    if not errors:
        store.write_indexes()
        summary = store.summary()
    if args.require_targets:
        errors.extend(capture_pair_target_errors(summary))
    output = {**summary, "errors": len(errors)}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
