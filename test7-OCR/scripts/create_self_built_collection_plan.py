from __future__ import annotations

import argparse
import json
from pathlib import Path

from test7_ocr.collection_plan import collection_plan_csv, collection_plan_summary


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "templates" / "self_built_collection_plan.csv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or verify the deterministic 410-row self-built collection plan"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that an existing plan exactly matches the current generated template",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing template; never use this on a collector's filled working copy",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    expected = collection_plan_csv()
    summary = collection_plan_summary()
    if args.check:
        if not output.is_file():
            raise FileNotFoundError(output)
        if output.read_bytes() != expected:
            raise ValueError("Collection plan differs from the deterministic template")
        print(json.dumps({**summary, "path": str(output), "verified": True}, ensure_ascii=False, indent=2))
        return 0
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite an existing collection plan: {output}; use --overwrite only for the template"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    print(json.dumps({**summary, "path": str(output), "created": True}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
