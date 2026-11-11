from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2


PROJECT = Path(r"D:\robot\Documents\ASK\picture\test1")
DATASET = Path(r"D:\robot\Documents\ASK\P_DATA\testmini_photo")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from app.processor import QuestionImageProcessor


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "input_manifest.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    names = [line.strip().lstrip("\ufeff") for line in args.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    end = args.end or len(names)
    names = names[args.start:end]
    args.out.mkdir(parents=True, exist_ok=True)
    processor = QuestionImageProcessor(args.out / "artifacts")
    rows: list[dict] = []
    started = time.perf_counter()
    for offset, name in enumerate(names, args.start + 1):
        source = DATASET / name
        row: dict = {"dataset_index": offset, "name": name, "source": str(source), "ok": False}
        try:
            decoded = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("OpenCV cannot decode source")
            row.update(source_width=decoded.shape[1], source_height=decoded.shape[0])
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
                perspective_status=result.perspective_status,
                perspective_confidence=result.perspective_confidence,
                cleanup_method=result.cleanup_method,
                reference_match=result.reference_match,
                reference_confidence=result.reference_confidence,
                output_width=result.width,
                output_height=result.height,
                output_png=result.png,
                output_pdf=result.pdf,
                output_markdown=result.markdown,
                question_count=result.question_count,
                fallback_count=sum(card.fallback for card in result.questions or []),
                minimum_card_width=min((card.width for card in result.questions or []), default=0),
                minimum_card_height=min((card.height for card in result.questions or []), default=0),
                questions_pdf=result.questions_pdf,
                questions_markdown=result.questions_markdown,
            )
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        print(f"[{offset}/{end}] {name}: {'ok' if row['ok'] else row.get('error')}", flush=True)

    fields = sorted({key for row in rows for key in row})
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    successes = [row for row in rows if row["ok"]]
    summary = {
        "manifest": str(args.manifest),
        "range": [args.start, end],
        "processed": len(rows),
        "successes": len(successes),
        "failures": len(rows) - len(successes),
        "wall_seconds": round(time.perf_counter() - started, 3),
        "mean_server_seconds": round(sum(float(row["elapsed_seconds"]) for row in successes) / max(1, len(successes)), 3),
        "p95_server_seconds": sorted(float(row["elapsed_seconds"]) for row in successes)[max(0, int(len(successes) * 0.95) - 1)] if successes else None,
        "cleanup_methods": dict(Counter(row.get("cleanup_method", "") for row in successes)),
        "deskew_statuses": dict(Counter(row.get("deskew_status", "") for row in successes)),
        "perspective_statuses": dict(Counter(row.get("perspective_status", "") for row in successes)),
        "rotations": dict(Counter(str(row.get("rotation", "")) for row in successes)),
        "fallback_cards": sum(int(row.get("fallback_count", 0)) for row in successes),
        "question_cards": sum(int(row.get("question_count", 0)) for row in successes),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
