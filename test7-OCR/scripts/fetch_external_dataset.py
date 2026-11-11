from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from test7_ocr.external_datasets import (
    LicenseBlockedError,
    PROJECT_ROOT,
)
from test7_ocr.external_fetch import fetch_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch pinned external evaluation datasets")
    parser.add_argument("dataset_id")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "external_datasets.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = fetch_dataset(args.dataset_id, force=args.force)
    except LicenseBlockedError as exc:
        print(json.dumps({"status": "LICENSE_BLOCKED", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    except (RuntimeError, KeyError) as exc:
        print(json.dumps({"status": "ERROR", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
