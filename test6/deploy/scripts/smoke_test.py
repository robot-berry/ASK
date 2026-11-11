#!/usr/bin/env python3
"""Readiness and optional real-upload smoke test for a deployed Test4 URL."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import secrets
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from PIL import Image


MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
EXPECTED_IMAGES = (
    (1, "question_knowledge", "01_question_knowledge.png"),
    (2, "elite", "02_guidance_elite.png"),
    (3, "upper", "03_guidance_upper.png"),
    (4, "middle", "04_guidance_middle.png"),
    (5, "lower", "05_guidance_lower.png"),
    (6, "novice", "06_guidance_novice.png"),
)


def request_json(
    url: str,
    token: str,
    *,
    data: bytes | None = None,
    content_type: str | None = None,
    idempotency_key: str | None = None,
) -> tuple[dict, dict]:
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    request = Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload, dict(response.headers.items())
    except HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from None
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from None


def download_png(url: str, token: str, expected: dict) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected.get("sha256") or "")):
        raise RuntimeError(f"missing or invalid sha256: {expected.get('filename')}")
    if not isinstance(expected.get("bytes"), int) or expected["bytes"] <= 0:
        raise RuntimeError(f"missing or invalid byte length: {expected.get('filename')}")
    request = Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "image/png"})
    with urlopen(request, timeout=90) as response:
        raw = response.read(20 * 1024 * 1024 + 1)
        content_type = response.headers.get_content_type()
    if content_type != "image/png" or len(raw) > 20 * 1024 * 1024:
        raise RuntimeError(f"invalid result response: {url}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected["sha256"]:
        raise RuntimeError(f"sha256 mismatch: {expected['filename']}")
    if len(raw) != expected["bytes"]:
        raise RuntimeError(f"byte length mismatch: {expected['filename']}")
    with Image.open(BytesIO(raw)) as image:
        image.load()
        if image.format != "PNG" or image.width != 800 or image.mode != "L":
            raise RuntimeError(f"PNG contract mismatch: {expected['filename']}")
    print(f"result ok: {expected['filename']} bytes={len(raw)} sha256={digest}")


def expect_unauthorized(url: str, token: str | None = None) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with urlopen(Request(url, headers=headers), timeout=20):
            pass
    except HTTPError as exc:
        if exc.code == 401:
            print("auth negative check ok")
            return
        raise RuntimeError(f"expected 401, received HTTP {exc.code}") from None
    raise RuntimeError("protected API accepted a missing or invalid token")


def multipart_image(raw: bytes, filename: str, media_type: str) -> tuple[bytes, str]:
    boundary = "----test4-smoke-" + secrets.token_hex(12)
    safe_name = Path(filename).name.replace('"', "_")
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{safe_name}"\r\n'
        f"Content-Type: {media_type}\r\n\r\n"
    ).encode("utf-8") + raw + f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Example: https://solver.example.com/")
    parser.add_argument(
        "--token",
        help="Defaults to TEST4_ACCESS_TOKEN; avoid putting a real token in shell history",
    )
    parser.add_argument("--subject", choices=("physics", "math"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--upload-mode", choices=("raw", "multipart"), default="raw")
    parser.add_argument(
        "--idempotency-key",
        help="8-128 safe ASCII characters; a unique key is generated when omitted",
    )
    parser.add_argument("--skip-auth-negative", action="store_true")
    parser.add_argument("--timeout", type=int, default=720)
    args = parser.parse_args()
    base = args.base.rstrip("/") + "/"
    token = (args.token or os.environ.get("TEST4_ACCESS_TOKEN", "")).strip()
    if not token:
        parser.error("set TEST4_ACCESS_TOKEN or provide --token")

    capabilities_url = urljoin(base, "api/v1/capabilities")
    if not args.skip_auth_negative:
        expect_unauthorized(capabilities_url)
        expect_unauthorized(capabilities_url, "definitely-wrong-test4-token")
    capabilities, _ = request_json(capabilities_url, token)
    idempotency = (capabilities.get("upload") or {}).get("idempotency") or {}
    if (
        capabilities.get("api_version") != "v1"
        or len(capabilities.get("results") or []) != len(EXPECTED_IMAGES)
        or idempotency.get("header") != "Idempotency-Key"
    ):
        raise RuntimeError("capabilities contract failed")
    print("capabilities ok")
    if not args.image:
        return 0
    if not args.subject:
        parser.error("--subject is required with --image")
    media_type = MEDIA.get(args.image.suffix.lower())
    if not media_type:
        parser.error("image must be JPEG, PNG or WebP")
    raw = args.image.read_bytes()
    if args.upload_mode == "raw":
        upload_url = urljoin(base, f"api/v1/{args.subject}/jobs/raw")
        upload_body, upload_type = raw, media_type
    else:
        upload_url = urljoin(base, f"api/v1/{args.subject}/jobs")
        upload_body, upload_type = multipart_image(raw, args.image.name, media_type)
    idempotency_key = args.idempotency_key or (
        f"smoke-{args.subject}-{int(time.time())}-{secrets.token_hex(8)}"
    )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._~-]{7,127}", idempotency_key):
        parser.error("--idempotency-key has an invalid format")
    created, headers = request_json(
        upload_url,
        token,
        data=upload_body,
        content_type=upload_type,
        idempotency_key=idempotency_key,
    )
    job_id = str(created.get("job_id") or "")
    poll_url = urljoin(base, str(created.get("poll_url") or ""))
    if not job_id or not poll_url:
        raise RuntimeError("create response is missing job metadata")
    location = next(
        (value for key, value in headers.items() if key.lower() == "location"), ""
    )
    retry_after = next(
        (value for key, value in headers.items() if key.lower() == "retry-after"), ""
    )
    if location != created.get("poll_url") or not str(retry_after).isdigit():
        raise RuntimeError("create response is missing Location or Retry-After")
    if next(
        (value for key, value in headers.items() if key.lower() == "idempotency-replayed"),
        "",
    ).lower() != "false":
        raise RuntimeError("first keyed create did not report Idempotency-Replayed: false")

    replayed, replay_headers = request_json(
        upload_url,
        token,
        data=upload_body,
        content_type=upload_type,
        idempotency_key=idempotency_key,
    )
    replay_flag = next(
        (
            value
            for key, value in replay_headers.items()
            if key.lower() == "idempotency-replayed"
        ),
        "",
    )
    if (
        replayed.get("job_id") != job_id
        or replayed.get("poll_url") != created.get("poll_url")
        or replay_flag.lower() != "true"
    ):
        raise RuntimeError("Idempotency-Key replay did not return the original job")
    print("idempotency replay ok")
    print(f"job created: subject={args.subject} job_id={job_id}")

    deadline = time.monotonic() + args.timeout
    current = created
    while current.get("status") not in {"completed", "failed"}:
        if time.monotonic() >= deadline:
            raise RuntimeError("job timed out")
        wait = int(current.get("retry_after_seconds") or headers.get("Retry-After") or 2)
        time.sleep(max(1, min(wait, 10)))
        current, headers = request_json(poll_url, token)
        print(f"status: {current.get('status')}")
    if current.get("status") != "completed":
        raise RuntimeError(f"job failed: {current.get('error')}")
    images = current.get("images") or []
    observed = [
        (item.get("sequence"), item.get("kind"), item.get("filename"))
        for item in images
    ]
    if observed != list(EXPECTED_IMAGES):
        raise RuntimeError("completed job does not contain the ordered six-image contract")
    for item in images:
        download_png(urljoin(base, item["url"]), token, item)
    print("Test4 end-to-end smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
