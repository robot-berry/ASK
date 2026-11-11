from __future__ import annotations

import argparse
import json
from pathlib import Path

from test7_ocr.collection_workspace import (
    inspect_collection_workspace,
    prepare_collection_workspace,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_COLLECTION_ROOT = (
    WORKSPACE_ROOT / "datasets" / "test7_ocr_self_built_v1" / "collection"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or inspect the private 410-image self-built collection workspace"
    )
    parser.add_argument("--workspace-root", type=Path, default=DEFAULT_COLLECTION_ROOT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Inspect without creating directories or refreshing readiness reports",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Return a non-zero status until all real images and privacy declarations are ready",
    )
    args = parser.parse_args()
    report = (
        inspect_collection_workspace(args.workspace_root)
        if args.check
        else prepare_collection_workspace(args.workspace_root)
    )
    console_report = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "missing_rows",
            "invalid_rows",
            "incremental_ready_rows",
            "blocked_by_clean_dependency",
        }
    }
    console_report["missing_rows_preview"] = report["missing_rows"][:10]
    console_report["invalid_rows_preview"] = report["invalid_rows"][:10]
    print(json.dumps(console_report, ensure_ascii=False, indent=2))
    if report["structural_errors"]:
        return 2
    if args.require_ready and not report["ready_for_import_preflight"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
