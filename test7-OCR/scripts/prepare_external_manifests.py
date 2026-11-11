from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from test7_ocr.external_datasets import PROJECT_ROOT, resolve_project_path
from test7_ocr.external_manifests import (
    prepare_mathvision_wild_manifests,
    prepare_omnidocbench_manifests,
    prepare_restoration_manifest,
    prepare_unimer_manifests,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert fetched external datasets into eval manifests")
    parser.add_argument("dataset_id", choices=["omnidocbench", "unimer_test", "mathvision_wild", "warpdoc", "realdae"])
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args(argv)
    defaults = {
        "omnidocbench": PROJECT_ROOT / "data" / "external" / "omnidocbench",
        "unimer_test": PROJECT_ROOT / "data" / "external" / "unimer",
        "mathvision_wild": PROJECT_ROOT / "data" / "external" / "mathvision_wild",
        "warpdoc": PROJECT_ROOT / "data" / "external" / "document_restoration" / "warpdoc",
        "realdae": PROJECT_ROOT / "data" / "external" / "document_restoration" / "realdae",
    }
    root = resolve_project_path(args.dataset_root or defaults[args.dataset_id])
    if args.dataset_id == "omnidocbench":
        summary = prepare_omnidocbench_manifests(root)
    elif args.dataset_id == "unimer_test":
        summary = prepare_unimer_manifests(root)
    elif args.dataset_id == "mathvision_wild":
        summary = prepare_mathvision_wild_manifests(root)
    else:
        summary = prepare_restoration_manifest(root, args.dataset_id)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
