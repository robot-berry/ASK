#!/usr/bin/env python3
"""Launch the single-worker mathematics server detached on Windows."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT / "source"


def load_service_env(target: dict[str, str]) -> None:
    """Load only the model credential keys without printing their values."""

    workspace_env = PROJECT.parents[1] / "api_keys.env"
    local_env = SOURCE / "api_keys.env"
    selected = local_env if local_env.is_file() else workspace_env
    if not selected.is_file():
        return
    allowed = {
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_WORKSPACE_ID",
    }
    for raw_line in selected.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in allowed and key not in target:
            target[key] = value.strip().strip('"').strip("'")
PORT = (
    os.environ.get("MATH_TEST3_PORT", "").strip()
    or os.environ.get("TEST3_MATH_PORT", "").strip()
    or os.environ.get("TEST3_PORT", "").strip()
    or "8020"
)
HOST = (
    os.environ.get("MATH_TEST3_HOST", "").strip()
    or os.environ.get("TEST3_HOST", "").strip()
    or "0.0.0.0"
)
if not PORT.isdigit() or not 1 <= int(PORT) <= 65535:
    raise ValueError("MATH_TEST3_PORT/TEST3_PORT 必须是 1 到 65535 之间的端口")
STDOUT = PROJECT / f"server-{PORT}.stdout.log"
STDERR = PROJECT / f"server-{PORT}.stderr.log"

env = dict(os.environ)
load_service_env(env)
env["PYTHONPATH"] = str(SOURCE)
command = [
    sys.executable,
    "-m",
    "uvicorn",
    "search_machine_app:app",
    "--app-dir",
    str(SOURCE),
    "--host",
    HOST,
    "--port",
    PORT,
    "--workers",
    "1",
]
creationflags = 0
if os.name == "nt":
    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_NO_WINDOW
    )
with STDOUT.open("ab") as stdout, STDERR.open("ab") as stderr:
    process = subprocess.Popen(
        command,
        cwd=PROJECT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        creationflags=creationflags,
        close_fds=True,
    )
print(process.pid)
