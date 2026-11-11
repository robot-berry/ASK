#!/usr/bin/env python3
"""Zero-cost readiness checks run under the same account as each backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import socket
import ssl
import tempfile
from urllib.parse import urlparse


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--minimum-free-mb", type=int, default=1024)
    args = parser.parse_args()

    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key or key.startswith("replace_"):
        raise RuntimeError("DASHSCOPE_API_KEY is missing or still a placeholder")

    base_url = os.environ.get("DASHSCOPE_BASE_URL", "").strip() or DEFAULT_BASE_URL
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not (
        host == "dashscope.aliyuncs.com"
        or host.endswith(".maas.aliyuncs.com")
        or host == "api.siliconflow.cn"
    ):
        raise RuntimeError("DASHSCOPE_BASE_URL is not an approved HTTPS endpoint")

    data_dir = args.data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=data_dir, prefix=".preflight-", delete=True):
        pass
    free = shutil.disk_usage(data_dir).free
    if free < args.minimum_free_mb * 1024 * 1024:
        raise RuntimeError(
            f"insufficient free disk space: {free // (1024 * 1024)} MiB"
        )

    context = ssl.create_default_context()
    with socket.create_connection((host, parsed.port or 443), timeout=8) as raw:
        with context.wrap_socket(raw, server_hostname=host) as tls:
            if not tls.getpeercert():
                raise RuntimeError("DashScope TLS certificate was not verified")
    print(f"preflight ok: data={data_dir} free_mb={free // (1024 * 1024)} host={host}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
