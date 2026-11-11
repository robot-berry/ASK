from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from test7_ocr.providers import RapidOcrProvider
from test7_ocr.types import QuestionDocument, VisualAsset
from test7_ocr.verification import verify_render_readback


def load_document(path: Path) -> QuestionDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["visual_assets"] = [VisualAsset(**item) for item in payload.get("visual_assets", [])]
    return QuestionDocument(**payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare final-PNG RapidOCR readback settings")
    parser.add_argument("sample_directories", type=Path, nargs="+")
    parser.add_argument("--minimum-coverage", type=float, default=0.72)
    args = parser.parse_args()
    variants = {
        "default": {"Global.log_level": "warning"},
        "threads4": {
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.intra_op_num_threads": 4,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        },
        "threads2_side640": {
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "Det.limit_side_len": 640,
            "Rec.rec_batch_num": 12,
        },
    }
    output: list[dict[str, object]] = []
    for name, params in variants.items():
        provider = RapidOcrProvider(use_cls=False, params=params)
        samples: list[dict[str, object]] = []
        for directory in args.sample_directories:
            document = load_document(directory / "question_document.json")
            started = time.perf_counter()
            recognition = provider.recognize(directory / "output.png")
            elapsed = time.perf_counter() - started
            verification = verify_render_readback(document, recognition, args.minimum_coverage)
            samples.append(
                {
                    "sample_id": document.sample_id,
                    "seconds": round(elapsed, 6),
                    "coverage_ratio": verification["coverage_ratio"],
                    "passed": verification["passed"],
                }
            )
        output.append(
            {
                "variant": name,
                "params": params,
                "mean_seconds": round(sum(float(item["seconds"]) for item in samples) / len(samples), 6),
                "all_passed": all(bool(item["passed"]) for item in samples),
                "samples": samples,
            }
        )
    print(json.dumps({"variants": output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
