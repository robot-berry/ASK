from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.processor import QuestionImageProcessor


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="批量评测题图预处理与完整性保护。")
    result.add_argument("input", type=Path)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--limit", type=int, default=0)
    result.add_argument("--stride", type=int, default=1)
    result.add_argument("--files", nargs="*", default=[])
    return result


def main() -> None:
    args = parser().parse_args()
    if args.files:
        files = [args.input / name for name in args.files]
    else:
        files = sorted(
            path for path in args.input.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )[:: max(1, args.stride)]
    if args.limit:
        files = files[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    artifacts = args.out / "artifacts"
    processor = QuestionImageProcessor(artifacts)
    rows: list[dict] = []
    started = time.perf_counter()
    for index, source in enumerate(files, 1):
        row: dict = {"source": str(source), "name": source.name, "ok": False}
        try:
            decoded = cv2.imread(str(source))
            if decoded is not None:
                row["source_width"] = decoded.shape[1]
                row["source_height"] = decoded.shape[0]
            result = processor.process(
                source.read_bytes(),
                source.name,
                mode="binary",
                enable_ocr=False,
                handwriting_mode="none",
                rotation="auto",
                quality="fast",
                extract_questions=True,
                deskew_mode="safe",
            )
            row.update(
                ok=True,
                job_id=result.job_id,
                elapsed_seconds=result.elapsed_seconds,
                rotation=result.rotation,
                rotation_confidence=result.rotation_confidence,
                rotation_method=result.rotation_method,
                deskew_detected_angle=result.deskew_detected_angle,
                deskew_applied_angle=result.deskew_applied_angle,
                deskew_confidence=result.deskew_confidence,
                deskew_status=result.deskew_status,
                question_count=result.question_count,
                fallback_count=sum(card.fallback for card in result.questions or []),
                output_width=result.width,
                output_height=result.height,
                output_png=result.png,
                questions_pdf=result.questions_pdf,
                cleanup_method=result.cleanup_method,
                reference_match=result.reference_match,
                reference_confidence=result.reference_confidence,
                perspective_status=result.perspective_status,
                perspective_confidence=result.perspective_confidence,
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(f"[{index}/{len(files)}] {source.name}: {'ok' if row['ok'] else row.get('error')}", flush=True)

    fieldnames = sorted({key for row in rows for key in row})
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    successes = [row for row in rows if row["ok"]]
    summary = {
        "input": str(args.input.resolve()),
        "processed": len(rows),
        "successes": len(successes),
        "failures": len(rows) - len(successes),
        "fallbacks": sum(int(row.get("fallback_count", 0)) for row in successes),
        "elapsed_wall_seconds": round(time.perf_counter() - started, 3),
        "mean_server_seconds": round(
            sum(float(row["elapsed_seconds"]) for row in successes) / max(1, len(successes)), 3
        ),
        "deskew_statuses": dict(Counter(row.get("deskew_status", "") for row in successes)),
        "rotation_methods": dict(Counter(row.get("rotation_method", "") for row in successes)),
        "perspective_statuses": dict(
            Counter(row.get("perspective_status", "") for row in successes)
        ),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
