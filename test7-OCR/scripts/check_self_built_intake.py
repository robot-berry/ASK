from __future__ import annotations

import argparse
import json
from pathlib import Path

from test7_ocr.self_built_intake import SelfBuiltIntakeStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate quarantined self-built OCR source intake without using it for models"
    )
    parser.add_argument(
        "--require-targets",
        action="store_true",
        help="Fail unless all source collection targets and required photo conditions are present",
    )
    args = parser.parse_args()
    store = SelfBuiltIntakeStore(
        WORKSPACE_ROOT / "datasets" / "test7_ocr_self_built_v1",
        PROJECT_ROOT / "data" / "annotations" / "self_built_intake" / "records",
        PROJECT_ROOT / "data" / "splits" / "self_built_intake_summary.json",
    )
    summary, errors = store.check()
    store.write_summary()
    if args.require_targets:
        for name, target in summary["targets"].items():
            if int(target["current"]) < int(target["minimum"]):
                errors.append(
                    f"source intake target incomplete: {name}={target['current']}/{target['minimum']}"
                )
        conditions = summary.get("condition_counts") or {}
        for condition in ("perspective", "shadow", "fold_or_curve"):
            if int(conditions.get(condition, 0)) <= 0:
                errors.append(f"required photo condition has no registered sample: {condition}")
        if int(conditions.get("blur", 0)) <= 0 and int(conditions.get("compression", 0)) <= 0:
            errors.append("required photo condition has no registered sample: blur_or_compression")
    output = {**summary, "errors": len(errors)}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
