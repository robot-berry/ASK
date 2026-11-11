from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from test7_ocr.providers import TimedRapidOcrProvider


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark the isolated RapidOCR verification worker."
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    provider = TimedRapidOcrProvider(
        timeout_seconds=30.0,
        params={
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.intra_op_num_threads": args.threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        },
    )
    measurements: list[dict[str, float | int]] = []
    try:
        for _ in range(max(1, args.runs)):
            started = time.perf_counter()
            result = provider.recognize(args.image)
            measurements.append(
                {
                    "roundtrip_ms": round(
                        (time.perf_counter() - started) * 1000.0, 3
                    ),
                    "worker_ms": round(
                        float(result.diagnostics["readback_worker_elapsed_ms"]), 3
                    ),
                    "characters": len(result.transcript),
                }
            )
    finally:
        provider.close()
    print(
        json.dumps(
            {
                "image": str(args.image.resolve()),
                "threads": args.threads,
                "worker_warmup_ms": provider.warmup_elapsed_ms,
                "measurements": measurements,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
