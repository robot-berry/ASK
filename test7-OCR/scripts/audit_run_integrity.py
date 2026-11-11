from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from test7_ocr.storage import image_pixel_sha256


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(run_directory: Path, expected_width: int) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()
    asset_count = 0
    reports = sorted(run_directory.glob("*/report.json"))
    for report_path in reports:
        sample_id = report_path.parent.name
        try:
            report = _json_object(report_path)
            status_counts[str(report.get("status") or "UNKNOWN")] += 1
            output_path = Path(str(report["output_image"]))
            document_path = Path(str(report["question_document"]))
            document = _json_object(document_path)
            if str(report.get("sample_id")) != sample_id:
                raise ValueError("sample_id does not match sample directory")
            with Image.open(output_path) as opened:
                opened.verify()
            with Image.open(output_path) as opened:
                width, height = opened.size
            if width != expected_width:
                raise ValueError(f"output width {width}, expected {expected_width}")
            if width != int(report.get("output_width") or 0):
                raise ValueError("output width does not match report")
            if height != int(report.get("output_height") or 0):
                raise ValueError("output height does not match report")
            if _sha256(output_path) != str(report.get("output_sha256") or ""):
                raise ValueError("output SHA-256 does not match report")

            for asset in document.get("visual_assets") or []:
                asset_count += 1
                asset_path = document_path.parent / str(asset["crop_path"])
                with Image.open(asset_path) as opened:
                    opened.load()
                    width, height = opened.size
                    pixel_sha256 = image_pixel_sha256(opened)
                bbox = asset["source_bbox"]
                if width != int(bbox["width"]) or height != int(bbox["height"]):
                    raise ValueError(f"asset dimensions do not match bbox: {asset_path}")
                if pixel_sha256 != str(asset.get("crop_sha256") or ""):
                    raise ValueError(f"asset pixel hash does not match: {asset_path}")
                if asset.get("extraction") != "direct_source_crop":
                    raise ValueError(f"asset is not a direct source crop: {asset_path}")
                if asset.get("redraw_used") is True:
                    raise ValueError(f"redrawn asset found: {asset_path}")
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"sample_id": sample_id, "error": str(exc)})

    return {
        "schema_version": "run_integrity_audit_v1",
        "run_directory": str(run_directory.resolve()),
        "expected_width": expected_width,
        "report_count": len(reports),
        "status_counts": dict(sorted(status_counts.items())),
        "visual_asset_count": asset_count,
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify saved PNGs, reports, and direct source-crop hashes for a run."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--expected-width", type=int, default=800)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.run_directory.resolve(), args.expected_width)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
