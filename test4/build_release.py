#!/usr/bin/env python3
"""Build a UTF-8/PAX Test4 release archive and its integrity manifests."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import sys
import tarfile
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent
TOP_LEVEL = PROJECT_ROOT.name
EXCLUDED_DIRECTORY_NAMES = {
    ".test-deps",
    ".venv",
    "release",
    "device_runs",
    "__pycache__",
    ".pytest_cache",
    ".state",
    ".local",
    ".diag_orientation",
    "_local_data",
    "_preprocess_test",
    "deployment_acceptance",
    "joint_test_output",
    "layout_preview",
    "live_dataset_manifests",
    "live_dataset_runs",
    "local_preprocess_audit",
    "local_test_data",
    "primary_visual_acceptance",
    "tools",
}
EXCLUDED_FILE_NAMES = {"api_keys.env", "RELEASE_MANIFEST.sha256"}
EXCLUDED_SUFFIXES = {".log", ".pyc"}
LOCAL_ONLY_TOP_LEVEL_FILES = {
    "compare_model_latency.py",
    "recover_generation_first_runs.py",
    "run_full_model_dataset.py",
    "full_model_dataset_20260820.json",
    "full_model_dataset_qwen38max_20260820.json",
    "p_data_math_physics_30_20260820.json",
    "p_data_math_physics_canary_2_20260820.json",
    "qwen38max_canary_20260820.json",
    "test9_math_physics_30_20260820.json",
    "math.zip",
}


def release_files() -> list[Path]:
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if path.name in EXCLUDED_FILE_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if len(relative.parts) == 1 and path.name in LOCAL_ONLY_TOP_LEVEL_FILES:
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(PROJECT_ROOT).as_posix())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(files: list[Path], destination: Path) -> None:
    lines = [
        f"{sha256_file(path)}  {path.relative_to(PROJECT_ROOT).as_posix()}"
        for path in files
    ]
    # Linux sha256sum versions in supported distributions require LF-only input.
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_archive(destination: Path, files: list[Path], manifest: Path) -> None:
    with tarfile.open(
        destination,
        mode="w:gz",
        format=tarfile.PAX_FORMAT,
        encoding="utf-8",
    ) as archive:
        for path in [*files, manifest]:
            relative = PurePosixPath(TOP_LEVEL) / PurePosixPath(
                path.relative_to(PROJECT_ROOT).as_posix()
            )
            archive.add(path, arcname=str(relative), recursive=False)


def main() -> int:
    if sys.version_info < (3, 10):
        raise SystemExit(f"Python 3.10+ is required, found {sys.version.split()[0]}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()

    output_directory = (args.output_directory or PROJECT_ROOT / "release").resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    version = (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = output_directory / f"solver-test4-{version}-{stamp}.tar.gz"
    manifest = PROJECT_ROOT / "RELEASE_MANIFEST.sha256"

    files = release_files()
    write_manifest(files, manifest)
    build_archive(archive, files, manifest)
    archive_hash = sha256_file(archive)
    hash_file = archive.with_name(archive.name + ".sha256")
    hash_file.write_text(
        f"{archive_hash}  {archive.name}\n", encoding="utf-8", newline="\n"
    )
    print(f"Archive={archive}")
    print(f"Sha256={archive_hash}")
    print(f"Bytes={archive.stat().st_size}")
    print(f"ManifestFiles={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
