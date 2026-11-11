#!/usr/bin/env python3
"""Run one saved Test4 mathematics job through the local pipeline without a provider call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parent / "source"
sys.path.insert(0, str(SOURCE_ROOT))

from search_machine_pipeline import SearchMachinePipeline  # noqa: E402
from render_local_dataset import raw_math_issues  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fixture_dir",
        type=Path,
        help="Saved job directory containing job.json and audit/original.upload",
    )
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    fixture_dir = args.fixture_dir.resolve()
    output_root = args.output_root.resolve()
    fixture = json.loads(
        (fixture_dir / "job.json").read_text(encoding="utf-8")
    )
    output_json = fixture.get("output_json")
    if not isinstance(output_json, dict):
        raise SystemExit("fixture job.json does not contain output_json")
    original = fixture_dir / "audit" / "original.upload"
    if not original.is_file():
        raise SystemExit("fixture does not contain audit/original.upload")

    pipeline = SearchMachinePipeline(runs_root=output_root)
    state = pipeline.new_job(original.read_bytes(), "math_fixture.jpg")
    result = pipeline.solve(
        state["job_id"],
        use_model=False,
        mock_output=output_json,
    )
    layout_issues = raw_math_issues(result.get("output_json") or {})
    summary = {
        "provider_called": False,
        "job_id": result.get("job_id"),
        "job_dir": str(output_root / str(result.get("job_id"))),
        "status": result.get("status"),
        "validation_valid": bool((result.get("validation") or {}).get("valid")),
        "validation_errors": (result.get("validation") or {}).get("errors", []),
        "image_count": len(result.get("images") or []),
        "image_contract_valid": bool(
            (result.get("image_contract") or {}).get("valid")
        ),
        "raw_math_issue_count": len(layout_issues),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if (
        summary["status"] == "completed"
        and summary["validation_valid"]
        and not summary["validation_errors"]
        and summary["image_count"] == 4
        and summary["image_contract_valid"]
        and summary["raw_math_issue_count"] == 0
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
