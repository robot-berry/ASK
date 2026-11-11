#!/usr/bin/env python3
"""Generate per-device Test6 credentials and a server-side hash setting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import secrets


SN_PREFIX_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,100}")


def generate_records(count: int, sn_prefix: str) -> list[dict[str, str]]:
    if not 1 <= count <= 10_000:
        raise ValueError("count must be between 1 and 10000")
    if SN_PREFIX_PATTERN.fullmatch(sn_prefix) is None:
        raise ValueError("sn-prefix must use ASCII letters, numbers, '.', '_', ':' or '-'")
    width = max(4, len(str(count)))
    records: list[dict[str, str]] = []
    for index in range(1, count + 1):
        token = "t6_" + secrets.token_urlsafe(32)
        records.append(
            {
                "device_sn": f"{sn_prefix}-{index:0{width}d}",
                "bearer_token": token,
                "token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
            }
        )
    return records


def write_outputs(
    output_dir: Path,
    records: list[dict[str, str]],
    *,
    force: bool,
) -> tuple[Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    credentials_path = output_dir / "device_credentials.csv"
    hashes_path = output_dir / "device_token_hashes.env"
    existing = [path for path in (credentials_path, hashes_path) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite {names}; use --force explicitly")
    output_dir.mkdir(parents=True, exist_ok=True)

    credentials_temp = credentials_path.with_suffix(".csv.tmp")
    hashes_temp = hashes_path.with_suffix(".env.tmp")
    with credentials_temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("device_sn", "bearer_token", "token_sha256"),
        )
        writer.writeheader()
        writer.writerows(records)
    hashes = ",".join(record["token_sha256"] for record in records)
    hashes_temp.write_text(
        f"TEST6_DEVICE_TOKEN_HASHES={hashes}\n",
        encoding="utf-8",
        newline="\n",
    )
    credentials_temp.replace(credentials_path)
    hashes_temp.replace(hashes_path)
    try:
        os.chmod(credentials_path, 0o600)
        os.chmod(hashes_path, 0o600)
    except OSError:
        pass
    return credentials_path, hashes_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--sn-prefix", default="ASK-CAMERA")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="explicitly replace existing credential files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        records = generate_records(args.count, args.sn_prefix)
        credentials_path, hashes_path = write_outputs(
            args.output_dir,
            records,
            force=args.force,
        )
    except (ValueError, FileExistsError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"GeneratedDevices={len(records)}")
    print(f"PrivateCredentials={credentials_path}")
    print(f"ServerHashSetting={hashes_path}")
    print("Keep the CSV private; copy only the env setting to the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
