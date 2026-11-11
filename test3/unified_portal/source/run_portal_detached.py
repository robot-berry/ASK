#!/usr/bin/env python3
"""Launch the unified portal as a hidden background process on Windows."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT / "source"
PORT = os.environ.get("TEST3_UNIFIED_PORT", "8030").strip() or "8030"
HOST = os.environ.get("TEST3_UNIFIED_HOST", "0.0.0.0").strip() or "0.0.0.0"
if not PORT.isdigit() or not 1 <= int(PORT) <= 65535:
    raise ValueError("TEST3_UNIFIED_PORT 必须是 1 到 65535 之间的端口")
if int(PORT) in {8010, 8020}:
    raise ValueError("统一入口不能占用物理或数学后端端口")

env = dict(os.environ)
env["PYTHONPATH"] = str(SOURCE)
command = [
    sys.executable,
    "-m",
    "uvicorn",
    "unified_portal_app:app",
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
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW

with (PROJECT / f"portal-{PORT}.stdout.log").open("ab") as stdout, (
    PROJECT / f"portal-{PORT}.stderr.log"
).open("ab") as stderr:
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
