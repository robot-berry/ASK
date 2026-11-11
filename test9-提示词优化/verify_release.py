#!/usr/bin/env python3
"""Verify a Test6 release archive against its source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile

import build_release


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "RELEASE_MANIFEST.sha256"


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_manifest(raw: bytes) -> dict[str, str]:
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise RuntimeError("release manifest must be UTF-8 with LF-only line endings")
    entries: dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not relative
            or relative in entries
        ):
            raise RuntimeError(f"invalid manifest line: {line!r}")
        entries[relative] = digest
    return entries


def latest_archive() -> Path:
    candidates = sorted(
        [
            *list((ROOT / "release").glob("solver-test6-*.tar.gz")),
            *list((ROOT / "release").glob("solver-test6-*.zip")),
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("no Test6 release archive was found")
    return candidates[0]


def verify(archive: Path) -> dict[str, object]:
    archive = archive.resolve()
    manifest_raw = MANIFEST.read_bytes()
    entries = parse_manifest(manifest_raw)
    current = {
        path.relative_to(ROOT).as_posix() for path in build_release.release_files()
    }
    if set(entries) != current:
        raise RuntimeError("manifest file list does not match the current source tree")
    for relative, expected in entries.items():
        actual = digest_bytes((ROOT / relative).read_bytes())
        if actual != expected:
            raise RuntimeError(f"source hash mismatch: {relative}")

    sidecar = archive.with_name(archive.name + ".sha256")
    sidecar_parts = sidecar.read_text(encoding="utf-8").strip().split()
    archive_digest = build_release.sha256_file(archive)
    if sidecar_parts != [archive_digest, archive.name]:
        raise RuntimeError("archive SHA-256 sidecar does not match")

    expected_files = {f"test6/{relative}" for relative in entries}
    expected_files.add("test6/RELEASE_MANIFEST.sha256")
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive, "r") as bundle:
            corrupt = bundle.testzip()
            if corrupt is not None:
                raise RuntimeError(f"corrupt ZIP member: {corrupt}")
            names = {info.filename for info in bundle.infolist()}
            files = {info.filename for info in bundle.infolist() if not info.is_dir()}
            for name in names:
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError(f"unsafe archive member: {name}")
            if files != expected_files:
                raise RuntimeError("archive file list does not match the manifest")
            if bundle.read("test6/RELEASE_MANIFEST.sha256") != manifest_raw:
                raise RuntimeError("embedded manifest does not match the source manifest")
            for relative, expected in entries.items():
                if digest_bytes(bundle.read(f"test6/{relative}")) != expected:
                    raise RuntimeError(f"archive hash mismatch: {relative}")
    else:
        with tarfile.open(archive, "r:gz", encoding="utf-8") as bundle:
            members = bundle.getmembers()
            names = {member.name for member in members}
            files = {member.name for member in members if member.isfile()}
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise RuntimeError(f"unsafe archive member: {member.name}")
            if files != expected_files:
                raise RuntimeError("archive file list does not match the manifest")
            embedded_manifest = bundle.extractfile("test6/RELEASE_MANIFEST.sha256")
            if embedded_manifest is None or embedded_manifest.read() != manifest_raw:
                raise RuntimeError("embedded manifest does not match the source manifest")
            for relative, expected in entries.items():
                handle = bundle.extractfile(f"test6/{relative}")
                if handle is None or digest_bytes(handle.read()) != expected:
                    raise RuntimeError(f"archive hash mismatch: {relative}")

    if not any(name.endswith("数学题解析提示词.md") for name in names):
        raise RuntimeError("mathematics prompt Unicode filename is missing")
    if not any(name.endswith("物理题解析提示词.md") for name in names):
        raise RuntimeError("physics prompt Unicode filename is missing")
    if any(
        "/device_runs/" in name
        or "/.state/" in name
        or "/release/" in name
        or name.endswith("/api_keys.env")
        for name in names
    ):
        raise RuntimeError("archive contains runtime state or a secret env file")

    return {
        "archive": str(archive),
        "sha256": archive_digest,
        "manifest_files": len(entries),
        "archive_files": len(files),
        "cr_bytes": manifest_raw.count(b"\r"),
        "unicode_names": True,
        "archive_hashes": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", nargs="?", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive or latest_archive()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
