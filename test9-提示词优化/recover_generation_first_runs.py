#!/usr/bin/env python3
"""Recover saved failed jobs into four images without any provider request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

from run_full_model_dataset import IMAGE_NAMES, atomic_json, load_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", choices=("math", "physics"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    test4_root = Path(__file__).resolve().parent
    output_dir = args.output_dir.resolve()
    progress_path = output_dir / f"progress_{args.subject}.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress_cases = progress.get("cases")
    if not isinstance(progress_cases, dict):
        raise ValueError("progress cases must be an object")

    failed = [
        (case_id, item)
        for case_id, item in progress_cases.items()
        if isinstance(item, dict) and item.get("status") != "completed"
    ]
    print(
        json.dumps(
            {
                "subject": args.subject,
                "failed_jobs": len(failed),
                "provider_calls": 0,
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    Pipeline = load_pipeline(test4_root, args.subject)
    jobs_dir = output_dir / "_jobs" / args.subject
    pipeline = Pipeline(jobs_dir)
    recovered = 0
    for index, (case_id, item) in enumerate(failed, start=1):
        job_id = str(item.get("job_id") or "")
        try:
            state = pipeline.recover_validation_failure(job_id)
            if state.get("status") != "completed":
                raise ValueError(f"unexpected recovery status: {state.get('status')}")
            job_dir = pipeline.job_path(job_id)
            case_dir = output_dir / args.subject / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            copied: list[str] = []
            for name in IMAGE_NAMES:
                source = job_dir / name
                if not source.is_file():
                    raise FileNotFoundError(f"recovered job is missing {name}")
                shutil.copy2(source, case_dir / name)
                copied.append(name)
            shutil.copy2(job_dir / "job.json", case_dir / "job.json")
            item.update(
                {
                    "status": "completed",
                    "result_dir": str(case_dir),
                    "images": copied,
                    "error": None,
                    "validation_valid": True,
                    "image_contract_valid": True,
                    "recovered_without_provider_call": True,
                    "generation_only": bool(
                        state.get("validation", {}).get("generation_only")
                    ),
                }
            )
            recovered += 1
            print(f"[{index}/{len(failed)}] {case_id}: recovered", flush=True)
        except Exception as exc:
            item["recovery_error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"[{index}/{len(failed)}] {case_id}: not_recoverable: {exc}",
                flush=True,
            )
        atomic_json(progress_path, progress)

    completed = sum(
        1
        for item in progress_cases.values()
        if isinstance(item, dict) and item.get("status") == "completed"
    )
    summary: dict[str, Any] = {
        "subject": args.subject,
        "requested": len(progress_cases),
        "completed": completed,
        "failed": len(progress_cases) - completed,
        "recovered_this_run": recovered,
        "provider_calls": 0,
        "progress_file": str(progress_path),
    }
    atomic_json(output_dir / f"summary_{args.subject}.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if completed == len(progress_cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
