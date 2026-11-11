from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from test7_ocr.external_eval import evaluate_external_run, write_external_eval_report
from test7_ocr.external_manifests import (
    prepare_mathvision_wild_manifests,
    prepare_omnidocbench_manifests,
    prepare_unimer_manifests,
)
from test7_ocr.storage import write_json_atomic, write_jsonl_atomic


def build_harness_fixtures(root: Path) -> dict[str, Path]:
    omni = root / "omnidocbench"
    images = omni / "images"
    images.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), "white").save(images / "exam.png")
    (omni / "OmniDocBench.json").write_text(
        json.dumps(
            [
                {
                    "page_info": {"page_no": 0, "image_path": "images/exam.png"},
                    "page_attribute": {
                        "data_source": "exam_paper",
                        "language": "simplified_chinese",
                    },
                    "layout_dets": [
                        {"order": 1, "category_type": "text", "text": "已知 x=1，求 y。"},
                        {"order": 2, "category_type": "equation_isolated", "latex": "y=x^2"},
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    unimer = root / "unimer"
    for bucket in ("SPE", "CPE", "SCE", "HWE"):
        folder = unimer / bucket
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), "white").save(folder / "0.png")
        (folder / "0.txt").write_text("x^2", encoding="utf-8")
    mathvision = root / "mathvision_wild"
    (mathvision / "images").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 200), "white").save(mathvision / "images" / "mini.png")
    Image.new("RGB", (200, 200), "white").save(mathvision / "images" / "full.png")
    (mathvision / "annotations.json").write_text(
        json.dumps(
            [
                {
                    "id": "mini-1",
                    "split": "testmini_photo",
                    "image": "images/mini.png",
                    "question": "What is 1+1?",
                    "options": ["1", "2"],
                    "answer": "IGNORE",
                },
                {
                    "id": "full-9",
                    "split": "test_photo",
                    "image": "images/full.png",
                    "question": "Find the area",
                    "options": ["A", "B"],
                    "answer": "IGNORE",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"omnidocbench": omni, "unimer": unimer, "mathvision_wild": mathvision}


def write_synthetic_run(run_root: Path, sample_id: str, display_text: str, formulas: list[str]) -> None:
    sample = run_root / sample_id
    sample.mkdir(parents=True, exist_ok=True)
    document = {
        "display_text": display_text,
        "source_blocks": [{"kind": "formula", "text": item} for item in formulas],
    }
    report = {
        "sample_id": sample_id,
        "status": "REVIEW",
        "output_width": 800,
        "elapsed_seconds": 3.1,
        "question_document": str(sample / "question_document.json"),
    }
    write_json_atomic(sample / "question_document.json", document)
    write_json_atomic(sample / "report.json", report)
    write_jsonl_atomic(run_root / "index.jsonl", [report])
    write_json_atomic(
        run_root / "batch_summary.json",
        {
            "schema_version": "batch_summary_v1",
            "dataset_role": "external_eval_only",
            "default_review_eligible": False,
            "formal_gold_scoring": False,
            "attempted": 1,
            "failures": [],
            "mode": "end_to_end",
        },
    )
