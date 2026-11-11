#!/usr/bin/env python3
"""Run one subject of a mixed Test4 manifest through the production pipeline.

The script is intentionally serial and resumable.  Provider calls are disabled
unless ``--allow-provider-call`` is present.  It writes ordinary JSON and PNG
files only; no HTML report is generated.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any


IMAGE_NAMES = (
    "01_question_knowledge.png",
    "02_guidance_advanced.png",
    "03_guidance_standard.png",
    "04_guidance_foundation.png",
)

PROVIDER_ENV_KEYS = (
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_BASE_URL",
    "DASHSCOPE_WORKSPACE_ID",
)


def load_workspace_provider_env(test4_root: Path) -> None:
    """Load only the approved DashScope settings without exposing secrets."""

    env_path = test4_root.parents[1] / "api_keys.env"
    if not env_path.is_file():
        raise RuntimeError("workspace api_keys.env was not found")
    allowed = set(PROVIDER_ENV_KEYS)
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed or os.environ.get(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            os.environ[key] = value
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def safe_case_id(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value) is None:
        raise ValueError(f"invalid case id: {value!r}")
    return value


def load_pipeline(test4_root: Path, subject: str):
    source = test4_root / subject / "source"
    if not source.is_dir():
        raise FileNotFoundError(f"subject source directory not found: {source}")
    sys.path.insert(0, str(source))
    module = importlib.import_module("search_machine_pipeline")
    return module.SearchMachinePipeline


def completed_copy_is_valid(case_dir: Path) -> bool:
    return all((case_dir / name).is_file() for name in IMAGE_NAMES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--subject", choices=("math", "physics"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-provider-call", action="store_true")
    parser.add_argument("--model-id", default="")
    parser.add_argument(
        "--provider-repair-rounds", type=int, choices=(0, 1, 2), default=2
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-existing-terminal",
        action="store_true",
        help="do not resubmit cases already recorded as completed, failed, or runner_error",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="with --skip-existing-terminal, resubmit failed/runner_error cases only",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("manifest cases must be a list")

    cases: list[dict[str, Any]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict) or raw_case.get("subject") != args.subject:
            continue
        case_id = safe_case_id(str(raw_case.get("id") or ""))
        image = Path(str(raw_case.get("image") or ""))
        if not image.is_absolute():
            image = (manifest_path.parent / image).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"image not found for {case_id}: {image}")
        reuse_job_dir = raw_case.get("reuse_job_dir")
        if reuse_job_dir:
            reuse_path = Path(str(reuse_job_dir))
            if not reuse_path.is_absolute():
                reuse_path = (manifest_path.parent / reuse_path).resolve()
            if not (reuse_path / "job.json").is_file():
                raise FileNotFoundError(
                    f"reusable job not found for {case_id}: {reuse_path}"
                )
        else:
            reuse_path = None
        cases.append({"id": case_id, "image": image, "reuse_job_dir": reuse_path})

    print(
        json.dumps(
            {
                "subject": args.subject,
                "case_count": len(cases),
                "provider_calls_allowed": bool(args.allow_provider_call),
                "model_id": args.model_id or "subject default",
                "provider_repair_rounds": args.provider_repair_rounds,
                "dry_run": bool(args.dry_run),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0
    if not args.allow_provider_call:
        raise RuntimeError("provider calls require --allow-provider-call")

    test4_root = Path(__file__).resolve().parent
    load_workspace_provider_env(test4_root)
    output_dir = args.output_dir.resolve()
    subject_dir = output_dir / args.subject
    jobs_dir = output_dir / "_jobs" / args.subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / f"progress_{args.subject}.json"
    progress: dict[str, Any]
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {"subject": args.subject, "cases": {}}
    progress_cases = progress.setdefault("cases", {})

    Pipeline = load_pipeline(test4_root, args.subject)
    pipeline_kwargs = {
        "max_provider_repair_rounds": args.provider_repair_rounds,
    }
    if args.model_id:
        pipeline_kwargs["model_id"] = args.model_id
    pipeline = Pipeline(jobs_dir, **pipeline_kwargs)

    for index, case in enumerate(cases, start=1):
        case_id = case["id"]
        image: Path = case["image"]
        case_dir = subject_dir / case_id
        previous = progress_cases.get(case_id)
        if (
            args.skip_existing_terminal
            and isinstance(previous, dict)
            and previous.get("status") in {"completed", "failed", "runner_error"}
            and not (
                args.retry_failures
                and previous.get("status") in {"failed", "runner_error"}
            )
        ):
            print(
                f"[{index}/{len(cases)}] {case_id}: existing {previous.get('status')}",
                flush=True,
            )
            continue
        if (
            isinstance(previous, dict)
            and previous.get("status") == "completed"
            and completed_copy_is_valid(case_dir)
        ):
            print(f"[{index}/{len(cases)}] {case_id}: already completed", flush=True)
            continue

        reuse_job_dir = case.get("reuse_job_dir")
        if isinstance(reuse_job_dir, Path):
            reused_state = json.loads(
                (reuse_job_dir / "job.json").read_text(encoding="utf-8")
            )
            if (
                reused_state.get("status") != "completed"
                or not isinstance(reused_state.get("validation"), dict)
                or not reused_state["validation"].get("valid")
                or not isinstance(reused_state.get("image_contract"), dict)
                or not reused_state["image_contract"].get("valid")
            ):
                raise ValueError(f"reusable job is not valid for {case_id}")
            case_dir.mkdir(parents=True, exist_ok=True)
            for name in IMAGE_NAMES:
                source = reuse_job_dir / name
                if not source.is_file():
                    raise FileNotFoundError(f"reusable job is missing {name}")
                shutil.copy2(source, case_dir / name)
            shutil.copy2(reuse_job_dir / "job.json", case_dir / "job.json")
            progress_cases[case_id] = {
                "status": "completed",
                "job_id": reused_state.get("job_id"),
                "source": str(image),
                "result_dir": str(case_dir),
                "images": list(IMAGE_NAMES),
                "reused_existing_paid_job": True,
                "validation_valid": True,
                "image_contract_valid": True,
                "usage": reused_state.get("usage"),
            }
            atomic_json(progress_path, progress)
            print(
                f"[{index}/{len(cases)}] {case_id}: reused completed job",
                flush=True,
            )
            continue

        print(f"[{index}/{len(cases)}] {case_id}: creating job", flush=True)
        attempt_history: list[dict[str, Any]] = []
        if isinstance(previous, dict):
            prior_history = previous.get("attempt_history")
            if isinstance(prior_history, list):
                attempt_history.extend(
                    item for item in prior_history if isinstance(item, dict)
                )
            prior_job_id = previous.get("job_id")
            if prior_job_id:
                attempt_history.append(
                    {
                        "job_id": prior_job_id,
                        "status": previous.get("status"),
                        "error": previous.get("error"),
                    }
                )
        state = pipeline.new_job(image.read_bytes(), image.name)
        job_id = str(state["job_id"])
        progress_cases[case_id] = {
            "status": "running",
            "job_id": job_id,
            "source": str(image),
            "attempt_history": attempt_history,
        }
        atomic_json(progress_path, progress)
        try:
            final = pipeline.solve(job_id, use_model=True)
            status = str(final.get("status") or "unknown")
            job_dir = pipeline.job_path(job_id)
            case_dir.mkdir(parents=True, exist_ok=True)
            copied: list[str] = []
            if status == "completed":
                for name in IMAGE_NAMES:
                    source = job_dir / name
                    if not source.is_file():
                        raise FileNotFoundError(f"completed job is missing {name}")
                    shutil.copy2(source, case_dir / name)
                    copied.append(name)
                shutil.copy2(job_dir / "job.json", case_dir / "job.json")
            progress_cases[case_id] = {
                "status": status,
                "job_id": job_id,
                "source": str(image),
                "result_dir": str(case_dir),
                "images": copied,
                "error": final.get("error"),
                "validation_valid": bool(
                    isinstance(final.get("validation"), dict)
                    and final["validation"].get("valid")
                ),
                "image_contract_valid": bool(
                    isinstance(final.get("image_contract"), dict)
                    and final["image_contract"].get("valid")
                ),
                "usage": final.get("usage"),
                "attempt_history": attempt_history,
            }
            print(f"[{index}/{len(cases)}] {case_id}: {status}", flush=True)
        except Exception as exc:  # keep the full batch resumable
            progress_cases[case_id] = {
                "status": "runner_error",
                "job_id": job_id,
                "source": str(image),
                "error": f"{type(exc).__name__}: {exc}",
                "attempt_history": attempt_history,
            }
            print(
                f"[{index}/{len(cases)}] {case_id}: runner_error: {exc}",
                flush=True,
            )
        atomic_json(progress_path, progress)

    completed = sum(
        1 for item in progress_cases.values()
        if isinstance(item, dict) and item.get("status") == "completed"
    )
    summary = {
        "subject": args.subject,
        "requested": len(cases),
        "completed": completed,
        "failed": len(cases) - completed,
        "progress_file": str(progress_path),
    }
    atomic_json(output_dir / f"summary_{args.subject}.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if completed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
