from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def match_crop(source_path: Path, crop_path: Path) -> tuple[list[int] | None, float]:
    source = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if source is None or crop is None:
        return None, 0.0
    source_height, source_width = source.shape[:2]
    crop_height, crop_width = crop.shape[:2]
    if crop_width > source_width or crop_height > source_height:
        return None, 0.0
    response = cv2.matchTemplate(source, crop, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(response)
    x, y = location
    return [int(x), int(y), int(crop_width), int(crop_height)], float(score)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recover candidate source bboxes from earlier illustration crops without reusing their pixels"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "testzip_math_physics_ocr_gold300_manifest.jsonl",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(PROJECT_ROOT / ".." / ".." / "datasets" / "testzip_math_physics_300_v1").resolve(),
    )
    parser.add_argument(
        "--legacy-results",
        type=Path,
        default=(
            PROJECT_ROOT
            / ".."
            / "test6"
            / "benchmark"
            / "opensource_layout_full300_20260904"
            / "results.jsonl"
        ).resolve(),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "testzip_visual_bbox_candidates.jsonl",
    )
    parser.add_argument("--minimum-score", type=float, default=0.94)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    legacy_by_id = {str(row["question_id"]): row for row in read_jsonl(args.legacy_results)}
    output_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for record in manifest:
        sample_id = str(record["sample_id"])
        source_path = args.dataset_root / str(record["image_path"])
        if sha256_file(source_path) != record["image_sha256"]:
            raise ValueError(f"Source hash mismatch: {sample_id}")
        legacy = legacy_by_id.get(sample_id, {})
        assets: list[dict[str, Any]] = []
        for index, raw_crop_path in enumerate(legacy.get("illustration_paths", []), start=1):
            crop_path = Path(raw_crop_path)
            bbox, score = match_crop(source_path, crop_path)
            accepted = bbox is not None and score >= args.minimum_score
            counts["candidate_assets"] += 1
            counts["accepted_assets" if accepted else "rejected_assets"] += 1
            assets.append(
                {
                    "asset_id": f"candidate_{index:02d}",
                    "source_bbox": bbox,
                    "template_match_score": round(score, 6),
                    "accepted_as_candidate": accepted,
                    "bbox_verified": False,
                    "crop_policy": "crop_again_from_current_original",
                    "legacy_crop_pixels_reused": False,
                }
            )
        if assets:
            counts["records_with_assets"] += 1
        output_rows.append(
            {
                "schema_version": "visual_bbox_candidate_v1",
                "sample_id": sample_id,
                "image_path": record["image_path"],
                "image_sha256": record["image_sha256"],
                "assets": assets,
                "warning": "CANDIDATE_ONLY_REQUIRES_HUMAN_REVIEW",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    summary = {
        "records": len(output_rows),
        **dict(sorted(counts.items())),
        "minimum_score": args.minimum_score,
        "output": str(args.output.resolve()),
        "direct_source_crop_required": True,
        "human_verification_required": True,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

