"""Create a 300-case visual audit queue from an existing local OCR run."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys


BENCHMARK_DIR = Path(__file__).resolve().parent
DATASET = Path(r"D:\robot\Documents\ASK\datasets\testzip_math_physics_300_v1")
DEFAULT_RUN_DIR = BENCHMARK_DIR / "ocr_accuracy_300_balanced_final_20260903"
RUNNER = BENCHMARK_DIR / "ocr_accuracy_300.py"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_runner() -> object:
    spec = importlib.util.spec_from_file_location("test6_ocr_accuracy_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = run_dir / "full_audit_queue"
    rows = read_jsonl(run_dir / "results.jsonl")
    if len(rows) != 300:
        raise RuntimeError(f"Expected 300 OCR rows, found {len(rows)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = load_runner()
    runner.write_audit_pages(DATASET, output_dir, rows)
    source = output_dir / "manual_audit.csv"
    target = output_dir / "full_manual_audit.csv"
    if target.exists():
        with target.open("r", encoding="utf-8-sig", newline="") as handle:
            completed = {
                row["question_id"]: row
                for row in csv.DictReader(handle)
                if row.get("manual_semantic_complete") in {"0", "1"}
            }
    else:
        completed = {}
    sample_audit = run_dir / "manual_audit.csv"
    if sample_audit.exists():
        with sample_audit.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("manual_semantic_complete") in {"0", "1"}:
                    completed.setdefault(row["question_id"], row)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        queue = list(csv.DictReader(handle))
    for row in queue:
        previous = completed.get(row["question_id"])
        if previous:
            for field in (
                "manual_exact_normalized",
                "manual_semantic_complete",
                "manual_error_count",
                "manual_error_types",
                "manual_notes",
            ):
                row[field] = previous.get(field, "")
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(queue[0]))
        writer.writeheader()
        writer.writerows(queue)
    source.unlink()
    print(f"Wrote {len(rows)} audit pages and {target}")


if __name__ == "__main__":
    main()
