from __future__ import annotations

import json

from test7_ocr.external_datasets import PROJECT_ROOT
from test7_ocr.external_eval import assemble_external_baseline, evaluate_external_run, write_external_eval_report
from test7_ocr.external_harness import build_harness_fixtures, write_synthetic_run
from test7_ocr.external_manifests import (
    prepare_mathvision_wild_manifests,
    prepare_omnidocbench_manifests,
    prepare_unimer_manifests,
)
from test7_ocr.storage import write_json_atomic, write_jsonl_atomic


def main() -> dict:
    fixture_root = PROJECT_ROOT / "data" / "external" / "harness_fixtures"
    roots = build_harness_fixtures(fixture_root)
    omni = prepare_omnidocbench_manifests(
        roots["omnidocbench"],
        exam_manifest=PROJECT_ROOT / "data" / "splits" / "omnidocbench_exam_zh.jsonl",
        exam_gold=PROJECT_ROOT / "data" / "splits" / "omnidocbench_exam_zh_gold.jsonl",
        smoke_manifest=PROJECT_ROOT / "data" / "splits" / "omnidocbench_exam_zh_smoke20.jsonl",
        smoke_gold=PROJECT_ROOT / "data" / "splits" / "omnidocbench_exam_zh_smoke20_gold.jsonl",
    )
    unimer = prepare_unimer_manifests(
        roots["unimer"],
        per_bucket=1,
        manifest=PROJECT_ROOT / "data" / "splits" / "unimer_test_stratified400.jsonl",
        gold=PROJECT_ROOT / "data" / "splits" / "unimer_test_stratified400_gold.jsonl",
        smoke_manifest=PROJECT_ROOT / "data" / "splits" / "unimer_test_smoke20.jsonl",
        smoke_gold=PROJECT_ROOT / "data" / "splits" / "unimer_test_smoke20_gold.jsonl",
    )
    mathvision = prepare_mathvision_wild_manifests(
        roots["mathvision_wild"],
        frozen_manifest=PROJECT_ROOT / "data" / "splits" / "mathvision_wild_frozen.jsonl",
        frozen_gold=PROJECT_ROOT / "data" / "splits" / "mathvision_wild_frozen_gold.jsonl",
        mini_manifest=PROJECT_ROOT / "data" / "splits" / "mathvision_wild_mini.jsonl",
        mini_gold=PROJECT_ROOT / "data" / "splits" / "mathvision_wild_mini_gold.jsonl",
    )

    gold_jobs = [
        ("omnidocbench", PROJECT_ROOT / "data" / "splits" / "omnidocbench_exam_zh_smoke20_gold.jsonl"),
        ("unimer_test", PROJECT_ROOT / "data" / "splits" / "unimer_test_smoke20_gold.jsonl"),
        ("mathvision_wild", PROJECT_ROOT / "data" / "splits" / "mathvision_wild_mini_gold.jsonl"),
    ]
    for dataset_id, gold_path in gold_jobs:
        run = PROJECT_ROOT / "artifacts" / "external" / dataset_id / "harness_smoke"
        gold_rows = [
            json.loads(line)
            for line in gold_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        reports = []
        for row in gold_rows:
            formulas = [str(item) for item in (row.get("gold_formulas") or [])]
            write_synthetic_run(
                run,
                str(row["sample_id"]),
                str(row.get("gold_text") or ""),
                formulas,
            )
            reports.append(
                json.loads((run / str(row["sample_id"]) / "report.json").read_text(encoding="utf-8"))
            )
        write_jsonl_atomic(run / "index.jsonl", reports)
        summary = evaluate_external_run(run, gold_path, dataset_id=dataset_id)
        summary["harness"] = "synthetic_fixture_smoke"
        summary["upstream_frozen_ocr_executed"] = False
        write_external_eval_report(summary, run / "report.json")

    write_json_atomic(
        PROJECT_ROOT / "artifacts" / "external" / "license_blocks.json",
        {
            "formal_gold_scoring": False,
            "blocked": {
                "warpdoc": "license_review_required",
                "realdae": "license_review_required",
                "scut_ensexam": "application_required",
                "m6doc": "application_required",
            },
            "text_cer_not_applicable": True,
        },
    )
    baseline = assemble_external_baseline()
    baseline["manifest_preparation"] = {
        "omnidocbench": omni,
        "unimer_test": unimer,
        "mathvision_wild": mathvision,
    }
    baseline["upstream_download"] = {
        "omnidocbench": "pending_huggingface_snapshot",
        "unimer_test": "pending_local_unimer_test_tree",
        "mathvision_wild": "pending_huggingface_snapshot",
        "warpdoc": "license_blocked",
        "realdae": "license_blocked",
    }
    write_json_atomic(PROJECT_ROOT / "artifacts" / "external" / "BASELINE.json", baseline)
    return baseline


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
