from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .external_datasets import (
    assert_download_allowed,
    load_external_config,
    resolve_project_path,
    sha256_file,
)


UNIMER_FORBIDDEN_MARKERS = (
    "unimer-1m",
    "unimer_1m",
    "unimer1m",
    "uni mer-1m",
)
LOGICAL_REVISION_DATASETS = frozenset({"omnidocbench", "unimer_test", "mathvision_wild"})


def huggingface_revision_for(dataset_id: str, entry: dict[str, Any]) -> str | None:
    if "hf_revision" in entry:
        value = entry.get("hf_revision")
        return str(value) if value else None
    pin = entry.get("pin_revision")
    if dataset_id in LOGICAL_REVISION_DATASETS:
        return None
    return str(pin) if pin else None


def allow_patterns_for(dataset_id: str) -> list[str] | None:
    if dataset_id == "omnidocbench":
        return ["OmniDocBench.json", "images/*", "README.md"]
    if dataset_id == "unimer_test":
        return [
            "UniMER-Test/**",
            "UniMER-Test.zip",
            "unimer-test/**",
            "unimer-test.zip",
            "README.md",
        ]
    if dataset_id == "mathvision_wild":
        return [
            "test_photo/**",
            "testmini_photo/**",
            "MathVision_Wild/test_photo/**",
            "MathVision_Wild/testmini_photo/**",
            "*.json",
            "*.jsonl",
            "README.md",
        ]
    return None


def snapshot_huggingface(
    repo_id: str,
    destination: Path,
    *,
    revision: str | None,
    allow_patterns: list[str] | None,
) -> str | None:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to fetch this dataset. "
            "Install it in the project venv, then retry."
        ) from exc
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(destination),
        allow_patterns=allow_patterns,
    )
    try:
        info = HfApi().dataset_info(repo_id, revision=revision)
        return getattr(info, "sha", None)
    except Exception:
        return None


def _assert_unimer_test_only(destination: Path) -> None:
    for path in destination.rglob("*"):
        relative = path.relative_to(destination).as_posix().lower()
        if any(marker in relative for marker in UNIMER_FORBIDDEN_MARKERS):
            shutil.rmtree(destination, ignore_errors=True)
            raise RuntimeError(
                "UniMER fetch matched a training-set path; aborting to avoid "
                f"downloading UniMER-1M: {relative}"
            )


def _extract_unimer_test_archives(destination: Path) -> None:
    for zip_path in list(destination.rglob("*.zip")):
        name = zip_path.name.lower()
        if "unimer-test" not in name and "unimertest" not in name:
            continue
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.namelist()
            if any(any(marker in member.lower() for marker in UNIMER_FORBIDDEN_MARKERS) for member in members):
                raise RuntimeError(f"{zip_path.name} contains UniMER-1M/train paths; refusing to extract")
            archive.extractall(destination)


def fetch_dataset(dataset_id: str, *, force: bool = False) -> dict[str, Any]:
    config = load_external_config()
    entry = assert_download_allowed(dataset_id, config)
    destination = resolve_project_path(entry["local_root"])
    if destination.exists() and any(destination.iterdir()) and not force:
        return {
            "status": "ALREADY_PRESENT",
            "dataset_id": dataset_id,
            "local_root": str(destination),
        }
    destination.mkdir(parents=True, exist_ok=True)
    repo = entry.get("hf_repo")
    if not repo:
        raise RuntimeError(
            f"{dataset_id} has no hf_repo; fetch it manually from {entry.get('homepage')}"
        )
    forbidden_repos = {"wanderkid/UniMER-1M", "wanderkid/unimer-1m"}
    if dataset_id == "unimer_test" and str(repo) in forbidden_repos:
        raise RuntimeError(f"{dataset_id} must not fetch {repo}; use wanderkid/UniMER_Dataset")
    revision = huggingface_revision_for(dataset_id, entry)
    commit_sha = snapshot_huggingface(
        str(repo),
        destination,
        revision=revision,
        allow_patterns=allow_patterns_for(dataset_id),
    )
    if dataset_id == "unimer_test":
        _assert_unimer_test_only(destination)
        _extract_unimer_test_archives(destination)
        _assert_unimer_test_only(destination)
    annotation = destination / str(entry.get("annotation_filename") or "")
    annotation_sha256 = sha256_file(annotation) if annotation.is_file() else None
    receipt = {
        "status": "FETCHED",
        "dataset_id": dataset_id,
        "local_root": str(destination),
        "hf_repo": repo,
        "pin_revision": entry.get("pin_revision"),
        "hf_revision": revision,
        "hf_commit_sha": commit_sha,
        "annotation_sha256": annotation_sha256,
        "evaluation_only": True,
        "training_allowed": False,
        "formal_gold_scoring": False,
    }
    (destination / "FETCH_RECEIPT.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt
