"""Serially retest the fixed 29-question manual OCR sample with production code."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WORKSPACE = Path(r"D:\robot\Documents\ASK")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from presentation_guard import readable_ocr_text


BASELINE = (
    WORKSPACE
    / "picture"
    / "test6"
    / "benchmark"
    / "ocr_accuracy_300_20260901"
)
DATASET = WORKSPACE / "datasets" / "testzip_math_physics_300_v1"
DEFAULT_OUTPUT = (
    WORKSPACE
    / "picture"
    / "test6"
    / "benchmark"
    / "semantic_completeness_retest_20260901"
)
MODULE_PATHS = {
    "mathematics": WORKSPACE / "picture" / "test6" / "math" / "source" / "test5_preprocess.py",
    "physics": WORKSPACE / "picture" / "test6" / "physics" / "source" / "test5_preprocess.py",
}


def load_module(subject: str) -> Any:
    path = MODULE_PATHS[subject]
    spec = importlib.util.spec_from_file_location(f"test6_retest_{subject}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # The benchmark needs production text/metrics, not the debug PNG stages.
    module.cv_write_png = lambda *_args, **_kwargs: None
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def compact(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_text(text: str, width: int = 48) -> list[str]:
    result: list[str] = []
    for line in text.splitlines() or [""]:
        stripped = line.strip()
        if not stripped:
            continue
        result.extend(stripped[index:index + width] for index in range(0, len(stripped), width))
    return result or ["（OCR为空）"]


def write_audit_page(output: Path, row: dict[str, Any], source: Path) -> None:
    canvas = Image.new("RGB", (1800, 1100), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(30)
    body_font = font(25)
    with Image.open(source) as opened:
        picture = opened.convert("RGB")
        picture.thumbnail((850, 940))
        canvas.paste(picture, ((880 - picture.width) // 2, 100 + (940 - picture.height) // 2))
    draw.text((30, 25), f"{row['question_id']}  优化后人工复核", fill="black", font=title_font)
    draw.text((930, 90), "优化后 OCR：", fill="black", font=title_font)
    y = 140
    for line in wrap_text(readable_ocr_text(row["after_transcript"])):
        draw.text((930, y), line, fill=(20, 20, 20), font=body_font)
        y += 38
        if y > 1010:
            break
    draw.text(
        (930, 1030),
        f"结构门：{row['after_structure_pass']}  用时：{row['after_total_ms']} ms",
        fill=(120, 0, 0),
        font=body_font,
    )
    canvas.save(output / "audit_pages" / f"{row['question_id']}.png")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit_pages").mkdir(exist_ok=True)

    audit_rows = read_csv(BASELINE / "manual_audit.csv")
    if args.limit is not None:
        audit_rows = audit_rows[: max(0, args.limit)]
    baseline_details = {
        row["question_id"]: row for row in read_csv(BASELINE / "details.csv")
    }
    modules = {subject: load_module(subject) for subject in MODULE_PATHS}
    engines = {subject: module.RapidOCR() for subject, module in modules.items()}
    results: list[dict[str, Any]] = []

    for index, audit in enumerate(audit_rows, 1):
        subject = audit["subject"]
        module = modules[subject]
        source = DATASET / audit["image_path"]
        started = time.perf_counter()
        record = module.process_one(
            source,
            args.output / "scratch" / audit["question_id"],
            module.Config(),
            "auto",
            engines[subject],
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        baseline = baseline_details[audit["question_id"]]
        semantic = (record.ocr_quality_audit or {}).get("final_semantic_completeness") or {}
        row: dict[str, Any] = {
            "question_id": audit["question_id"],
            "subject": subject,
            "image_path": audit["image_path"],
            "baseline_manual_semantic_complete": audit["manual_semantic_complete"],
            "baseline_error_types": audit["manual_error_types"],
            "baseline_notes": audit["manual_notes"],
            "baseline_transcript": baseline["transcript"],
            "baseline_compact_chars": len(compact(baseline["transcript"])),
            "after_transcript": record.ocr_text,
            "after_compact_chars": len(compact(record.ocr_text)),
            "after_structure_pass": bool(semantic.get("passes_structure_gate")),
            "after_structure_issues": int(semantic.get("issue_count") or 0),
            "after_option_labels": "".join(semantic.get("option_labels") or []),
            "after_ocr_pass_count": record.ocr_pass_count,
            "after_selected_pass": record.ocr_selected_pass,
            "after_illustration_count": record.illustration_count,
            "after_total_ms": elapsed_ms,
            "manual_retest_semantic_complete": "",
            "manual_retest_error_types": "",
            "manual_retest_notes": "",
        }
        results.append(row)
        write_rows(args.output / "results.csv", results)
        with (args.output / "results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_audit_page(args.output, row, source)
        print(
            f"[{index:02d}/{len(audit_rows):02d}] {row['question_id']} "
            f"chars {row['baseline_compact_chars']}->{row['after_compact_chars']} "
            f"gate={row['after_structure_pass']} {elapsed_ms:.0f}ms",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
