from __future__ import annotations

import argparse
import json
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from test7_ocr.image_ops import content_bbox, open_source_image
from test7_ocr.providers import StructuredServiceProvider
from test7_ocr.storage import write_json_atomic
from test7_ocr.text import latex_to_readable


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _compact(value: str) -> str:
    return "".join(latex_to_readable(value).split())


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure safe outer-whitespace cropping for structured OCR")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:8240")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "benchmarks" / "structured_crop.json")
    args = parser.parse_args()

    provider = StructuredServiceProvider(service_url=args.service_url, timeout_seconds=30.0)
    rows: list[dict[str, object]] = []
    for sample_id in args.sample_id:
        sample_directory = args.run_directory / sample_id
        document = json.loads((sample_directory / "question_document.json").read_text(encoding="utf-8"))
        source_path = Path(document["source_image"])
        source = open_source_image(source_path)
        bbox = content_bbox(source, threshold=248, padding=16)
        crop = source.crop((bbox.x, bbox.y, bbox.right, bbox.bottom))
        with tempfile.TemporaryDirectory() as temporary:
            crop_path = Path(temporary) / "outer-crop.png"
            crop.save(crop_path, format="PNG", optimize=True)
            started = time.perf_counter()
            recognition = provider.recognize(crop_path, sample_id=f"{sample_id}:outer-crop")
            elapsed = time.perf_counter() - started
        expected = _compact(str(document.get("canonical_text") or ""))
        actual = _compact(recognition.transcript)
        rows.append(
            {
                "sample_id": sample_id,
                "source_size": list(source.size),
                "crop_bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                "crop_area_ratio": round((bbox.width * bbox.height) / (source.width * source.height), 6),
                "baseline_recognition_seconds": round(
                    json.loads((sample_directory / "report.json").read_text(encoding="utf-8"))["timings_ms"]["recognition"] / 1000,
                    6,
                ),
                "cropped_recognition_seconds": round(elapsed, 6),
                "normalized_transcript_similarity": round(
                    SequenceMatcher(None, expected, actual, autojunk=False).ratio(),
                    6,
                ),
                "baseline_formula_count": int(document.get("formula_count") or 0),
                "cropped_formula_count": recognition.formula_count,
                "baseline_visual_count": len(document.get("visual_assets", [])),
                "cropped_visual_count": len(recognition.visual_bboxes),
                "cropped_block_count": len(recognition.blocks),
            }
        )
    summary = {"schema_version": "structured_crop_benchmark_v1", "samples": rows}
    write_json_atomic(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
