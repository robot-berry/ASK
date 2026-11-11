from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from test7_ocr.providers import StructuredServiceProvider
from test7_ocr.warmup import (
    STRUCTURED_WARMUP_PROFILE_VERSION,
    synthetic_structured_warmup_pages,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Warm structured OCR with synthetic non-dataset pages")
    parser.add_argument("--service-url", default="http://127.0.0.1:8240")
    args = parser.parse_args()
    provider = StructuredServiceProvider(service_url=args.service_url, timeout_seconds=30.0)
    results: list[dict[str, object]] = []
    pages = synthetic_structured_warmup_pages()
    with tempfile.TemporaryDirectory() as temporary:
        for index, page in enumerate(pages, start=1):
            path = Path(temporary) / f"synthetic-warmup-{index}.png"
            page.save(path, format="PNG", optimize=True)
            started = time.perf_counter()
            recognition = provider.recognize(path, sample_id=f"synthetic-warmup-{index}")
            results.append(
                {
                    "page": index,
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                    "block_count": len(recognition.blocks),
                    "formula_count": recognition.formula_count,
                    "visual_count": len(recognition.visual_bboxes),
                }
            )
    payload = {
        "source": "synthetic_non_dataset_pages",
        "dataset_or_gold_used": False,
        "profile_version": STRUCTURED_WARMUP_PROFILE_VERSION,
        "pages": results,
        "completed": len(results) == len(pages),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
