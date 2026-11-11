from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS_ROOT = (PROJECT_ROOT / ".." / ".." / "datasets").resolve()
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "datasets.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "local_dataset_inventory.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _manifest_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    rows = 0
    eligible = 0
    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            rows += 1
            payload = json.loads(line)
            if payload.get("include_in_benchmark", True):
                eligible += 1
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                if str(row.get("include_in_benchmark", "true")).strip().lower() not in {
                    "0",
                    "false",
                    "no",
                }:
                    eligible += 1
    else:
        return {"path": str(path), "rows": None, "eligible_rows": None}
    return {"path": str(path), "rows": rows, "eligible_rows": eligible}


def audit_inventory(datasets_root: Path, registry_path: Path) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries_by_root: dict[Path, tuple[str, dict[str, Any]]] = {}
    for dataset_id, entry in registry["datasets"].items():
        root = (PROJECT_ROOT / entry["root"]).resolve()
        entries_by_root[root] = (dataset_id, entry)

    image_hash_owners: dict[str, set[str]] = defaultdict(set)
    datasets: list[dict[str, Any]] = []
    unregistered: list[str] = []
    total_images = 0
    total_bytes = 0

    for root in sorted((p for p in datasets_root.iterdir() if p.is_dir()), key=lambda p: p.name):
        registered = entries_by_root.get(root.resolve())
        if registered is None:
            dataset_id = root.name
            entry: dict[str, Any] = {}
            unregistered.append(str(root))
        else:
            dataset_id, entry = registered

        suffix_counts: Counter[str] = Counter()
        resolution_counts: Counter[str] = Counter()
        byte_hash_counts: Counter[str] = Counter()
        image_bytes = 0
        unreadable: list[str] = []
        image_paths = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in image_paths:
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            byte_hash_counts[digest] += 1
            image_hash_owners[digest].add(dataset_id)
            suffix_counts[path.suffix.lower()] += 1
            image_bytes += len(payload)
            try:
                with Image.open(path) as image:
                    resolution_counts[f"{image.width}x{image.height}"] += 1
            except Exception:
                unreadable.append(str(path))

        manifest_path: Path | None = None
        if entry.get("manifest"):
            manifest_path = (PROJECT_ROOT / entry["manifest"]).resolve()
        elif (root / "manifest.jsonl").exists():
            manifest_path = root / "manifest.jsonl"
        elif (root / "manifest.csv").exists():
            manifest_path = root / "manifest.csv"

        duplicate_files = sum(count - 1 for count in byte_hash_counts.values() if count > 1)
        datasets.append(
            {
                "dataset_id": dataset_id,
                "root": str(root),
                "registered": registered is not None,
                "role": entry.get("role", "UNREGISTERED"),
                "primary_ocr_gold": dataset_id == registry.get("primary_ocr_gold"),
                "formal_scoring_enabled": bool(
                    dataset_id == registry.get("primary_ocr_gold")
                    and entry.get("formal_scoring_requires_verified_records")
                ),
                "training_allowed": bool(entry.get("training_allowed", False)),
                "allowed_workflows": entry.get("allowed_workflows", []),
                "counts_toward_self_built_real_world_targets": bool(
                    entry.get("counts_toward_self_built_real_world_targets", False)
                ),
                "image_count": len(image_paths),
                "unique_image_byte_count": len(byte_hash_counts),
                "internal_duplicate_file_count": duplicate_files,
                "image_bytes": image_bytes,
                "suffix_counts": dict(sorted(suffix_counts.items())),
                "top_resolutions": dict(resolution_counts.most_common(10)),
                "unreadable_image_count": len(unreadable),
                "unreadable_images": unreadable,
                "manifest": _manifest_summary(manifest_path),
            }
        )
        total_images += len(image_paths)
        total_bytes += image_bytes

    overlaps = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    for digest, owners in image_hash_owners.items():
        if len(owners) < 2:
            continue
        ordered = sorted(owners)
        overlaps.append({"sha256": digest, "dataset_ids": ordered})
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                pair_counts[(left, right)] += 1

    self_built_entries = [
        item for item in datasets if item["counts_toward_self_built_real_world_targets"]
    ]
    return {
        "schema_version": "local_dataset_inventory_v1",
        "datasets_root": str(datasets_root),
        "primary_ocr_gold": registry.get("primary_ocr_gold"),
        "dataset_directory_count": len(datasets),
        "registered_dataset_count": sum(int(item["registered"]) for item in datasets),
        "unregistered_dataset_roots": unregistered,
        "total_image_files": total_images,
        "total_image_bytes": total_bytes,
        "datasets": datasets,
        "cross_dataset_exact_byte_overlap": {
            "overlapping_sha256_count": len(overlaps),
            "pair_counts": [
                {"left": pair[0], "right": pair[1], "identical_sha256_count": count}
                for pair, count in sorted(pair_counts.items())
            ],
            "items": overlaps,
        },
        "self_built_real_world_coverage": {
            "eligible_dataset_ids": [item["dataset_id"] for item in self_built_entries],
            "present_image_files": sum(item["image_count"] for item in self_built_entries),
            "other_dataset_images_may_not_fill_collection_slots": True,
            "reason": (
                "Gold, electronic-paper derivatives, and unpaired historical scans do not provide "
                "the required clean-source/camera/handwriting document groups and review metadata."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local OCR dataset roles and exact overlaps.")
    parser.add_argument("--datasets-root", type=Path, default=DEFAULT_DATASETS_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = audit_inventory(args.datasets_root.resolve(), args.registry.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "dataset_directory_count",
        "registered_dataset_count",
        "unregistered_dataset_roots",
        "total_image_files",
        "self_built_real_world_coverage",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
