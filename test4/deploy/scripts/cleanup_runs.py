#!/usr/bin/env python3
"""Delete only expired completed/failed Test4 job directories."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shutil


JOB_ID = re.compile(r"^\d{8}T\d{6}_[0-9a-f]{32}$")
TERMINAL = {"completed", "failed"}


def updated_at(job_json: Path) -> tuple[str, dt.datetime] | None:
    try:
        payload = json.loads(job_json.read_text(encoding="utf-8-sig"))
        status = str(payload.get("status") or "")
        raw = str(payload.get("updated_at") or "")
        stamp = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=dt.timezone.utc)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return status, stamp.astimezone(dt.timezone.utc)


def clean_root(root: Path, cutoff: dt.datetime, dry_run: bool) -> tuple[int, int]:
    root = root.expanduser().resolve()
    if root in {Path("/"), Path.home().resolve()} or not root.is_dir():
        raise ValueError(f"拒绝清理不安全或不存在的目录：{root}")
    removed = skipped = 0
    for candidate in root.iterdir():
        if (
            not candidate.is_dir()
            or candidate.is_symlink()
            or not JOB_ID.fullmatch(candidate.name)
            or candidate.resolve().parent != root
        ):
            skipped += 1
            continue
        state = updated_at(candidate / "job.json")
        if state is None or state[0] not in TERMINAL or state[1] >= cutoff:
            skipped += 1
            continue
        print(("would-remove" if dry_run else "remove"), candidate)
        if not dry_run:
            shutil.rmtree(candidate)
        removed += 1
    return removed, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.days <= 365:
        parser.error("--days must be between 1 and 365")
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=args.days)
    total_removed = total_skipped = 0
    for root in args.root:
        removed, skipped = clean_root(root, cutoff, args.dry_run)
        total_removed += removed
        total_skipped += skipped
    print(f"removed={total_removed} skipped={total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
