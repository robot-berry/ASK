#!/usr/bin/env python3
"""Run a resumable, serial mathematics dataset check against the local Test4 API.

This runner deliberately separates already-created job recovery from creation of
new provider-backed jobs.  A new POST is impossible unless
``--allow-provider-call`` is present.  Every create intent is persisted before
the request and uses a deterministic Idempotency-Key, so an interrupted create
can be safely replayed on the next invocation.

The input is a small JSON manifest, for example::

    {
      "name": "math-canary",
      "cases": [
        {"id": "high_0401", "image": "../../data/q_0401.png",
         "grade": "high", "tags": ["trigonometry"]}
      ]
    }

Only loopback API URLs are accepted.  Credentials are read at runtime and are
never written to any run artifact.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import html
from io import BytesIO
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import time
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from PIL import Image, UnidentifiedImageError


RUNNER_VERSION = "1.0.0"
SCHEMA_VERSION = 1
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 20 * 1024 * 1024
EXPECTED_IMAGES = (
    (1, "question_knowledge", "01_question_knowledge.png"),
    (2, "elite", "02_guidance_elite.png"),
    (3, "upper", "03_guidance_upper.png"),
    (4, "middle", "04_guidance_middle.png"),
    (5, "lower", "05_guidance_lower.png"),
    (6, "novice", "06_guidance_novice.png"),
)
ALLOWED_REMOTE_STATUSES = {
    "queued",
    "preprocessing",
    "solving",
    "rendering",
    "completed",
    "failed",
}
TERMINAL_FAILURE_PHASES = {"failed", "contract_failed"}
CASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z", re.ASCII)
JOB_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z", re.ASCII)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
MEDIA_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
EXTENSION_BY_FORMAT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
PHYSICAL_WIDTH_TOLERANCE_CM = 0.02


class RunnerError(RuntimeError):
    """A safe, user-facing runner failure."""


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )


def read_json(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RunnerError(f"cannot read JSON file {path}: {exc}") from None
    if len(raw) > MAX_JSON_BYTES:
        raise RunnerError(f"JSON file is larger than {MAX_JSON_BYTES} bytes: {path}")
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"invalid UTF-8 JSON in {path}: {exc}") from None


def require_short_text(value: Any, label: str, maximum: int) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise RunnerError(f"{label} must be a string")
    cleaned = value.strip()
    if len(cleaned) > maximum:
        raise RunnerError(f"{label} is longer than {maximum} characters")
    return cleaned


def inspect_source_image(path: Path) -> tuple[str, int, int]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RunnerError(f"cannot inspect source image {path}: {exc}") from None
    if size <= 0 or size > MAX_UPLOAD_BYTES:
        raise RunnerError(
            f"source image must contain 1..{MAX_UPLOAD_BYTES} bytes: {path}"
        )
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image_format = str(image.format or "").upper()
            width, height = image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise RunnerError(f"source is not a readable JPEG, PNG or WebP: {path}: {exc}") from None
    if image_format not in MEDIA_BY_FORMAT or width <= 0 or height <= 0:
        raise RunnerError(f"source is not a supported JPEG, PNG or WebP: {path}")
    return image_format, width, height


def load_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        raw_bytes = path.read_bytes() if path.is_file() else b""
    except OSError as exc:
        raise RunnerError(f"cannot read manifest {path}: {exc}") from None
    if not raw_bytes:
        raise RunnerError(f"manifest does not exist or is empty: {path}")
    if len(raw_bytes) > MAX_JSON_BYTES:
        raise RunnerError("manifest is too large")
    try:
        document = json.loads(raw_bytes.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"invalid manifest JSON: {exc}") from None

    if isinstance(document, list):
        root: dict[str, Any] = {"cases": document}
    elif isinstance(document, dict):
        root = document
    else:
        raise RunnerError("manifest root must be an object or a list of cases")
    unknown_root = set(root) - {"name", "description", "cases"}
    if unknown_root:
        raise RunnerError(f"unsupported manifest fields: {sorted(unknown_root)}")
    raw_cases = root.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RunnerError("manifest cases must be a non-empty list")

    cases: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases, start=1):
        label = f"cases[{index - 1}]"
        if not isinstance(raw_case, dict):
            raise RunnerError(f"{label} must be an object")
        allowed = {"id", "image", "grade", "group", "tags", "notes", "reference"}
        unknown = set(raw_case) - allowed
        if unknown:
            raise RunnerError(f"unsupported fields in {label}: {sorted(unknown)}")
        case_id = require_short_text(raw_case.get("id"), f"{label}.id", 64)
        if not CASE_ID_RE.fullmatch(case_id):
            raise RunnerError(
                f"{label}.id must use 1-64 ASCII letters, digits, dot, underscore or hyphen"
            )
        if case_id in observed_ids:
            raise RunnerError(f"duplicate case id: {case_id}")
        observed_ids.add(case_id)
        image_value = require_short_text(raw_case.get("image"), f"{label}.image", 4096)
        if not image_value:
            raise RunnerError(f"{label}.image is required")
        candidate = Path(image_value)
        source_path = candidate if candidate.is_absolute() else path.parent / candidate
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise RunnerError(f"source image does not exist: {source_path}")
        image_format, width, height = inspect_source_image(source_path)
        source_bytes = source_path.stat().st_size
        source_digest = sha256_file(source_path)

        raw_tags = raw_case.get("tags", [])
        if not isinstance(raw_tags, list) or len(raw_tags) > 20:
            raise RunnerError(f"{label}.tags must be a list with at most 20 entries")
        tags = [require_short_text(value, f"{label}.tags", 80) for value in raw_tags]
        if any(not value for value in tags):
            raise RunnerError(f"{label}.tags cannot contain empty values")
        cases.append(
            {
                "id": case_id,
                "image": image_value,
                "source_path": str(source_path),
                "source_sha256": source_digest,
                "source_bytes": source_bytes,
                "source_format": image_format,
                "source_width": width,
                "source_height": height,
                "source_media_type": MEDIA_BY_FORMAT[image_format],
                "source_copy_name": "source" + EXTENSION_BY_FORMAT[image_format],
                "grade": require_short_text(raw_case.get("grade"), f"{label}.grade", 80),
                "group": require_short_text(raw_case.get("group"), f"{label}.group", 80),
                "tags": tags,
                "notes": require_short_text(raw_case.get("notes"), f"{label}.notes", 2000),
                "reference": require_short_text(
                    raw_case.get("reference"), f"{label}.reference", 10000
                ),
            }
        )
    return {
        "name": require_short_text(root.get("name"), "manifest.name", 160)
        or path.stem,
        "description": require_short_text(
            root.get("description"), "manifest.description", 2000
        ),
        "path": str(path),
        "sha256": sha256_bytes(raw_bytes),
        "cases": cases,
    }


def canonical_loopback_base(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise RunnerError("--base-url must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise RunnerError("--base-url must not contain credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise RunnerError("--base-url must be an origin only, without path/query/fragment")
    host = (parsed.hostname or "").lower()
    if not host:
        raise RunnerError("--base-url is missing a host")
    try:
        parsed.port
    except ValueError as exc:
        raise RunnerError(f"--base-url has an invalid port: {exc}") from None
    allowed = host == "localhost"
    if not allowed:
        try:
            allowed = ipaddress.ip_address(host).is_loopback
        except ValueError:
            allowed = False
    if not allowed:
        raise RunnerError("--base-url host must be loopback (localhost, 127.0.0.0/8 or ::1)")
    # Resolve localhost defensively; a modified hosts file must not redirect the runner.
    if host == "localhost":
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(host, parsed.port or 80, type=socket.SOCK_STREAM)
            }
        except OSError as exc:
            raise RunnerError(f"cannot resolve localhost safely: {exc}") from None
        if not addresses or any(not ipaddress.ip_address(address).is_loopback for address in addresses):
            raise RunnerError("localhost resolves to a non-loopback address")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, "/", "", ""))


def api_url(base_url: str, path: str) -> str:
    if not path.startswith("/") or path.startswith("//"):
        raise RunnerError(f"unsafe API path: {path}")
    parsed = urlsplit(base_url)
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise RunnerError(f"unexpected HTTP redirect ({code}); refusing redirected request")


class LocalApiClient:
    def __init__(self, base_url: str, credential: str, timeout_seconds: float):
        self.base_url = base_url
        self.credential = credential
        self.timeout_seconds = timeout_seconds
        # Ignore proxy environment variables: even a loopback URL must stay local.
        self.opener = build_opener(ProxyHandler({}), RejectRedirects())

    def _headers(self, accept: str) -> dict[str, str]:
        result = {"Accept": accept}
        if self.credential:
            result["Authorization"] = f"Bearer {self.credential}"
        return result

    def request_json(
        self,
        path: str,
        *,
        data: bytes | None = None,
        content_type: str | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        headers = self._headers("application/json")
        if content_type:
            headers["Content-Type"] = content_type
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        method = "POST" if data is not None else "GET"
        request = Request(api_url(self.base_url, path), data=data, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(MAX_JSON_BYTES + 1)
                response_headers = {key.lower(): value for key, value in response.headers.items()}
        except HTTPError as exc:
            status = exc.code
            exc.close()
            raise RunnerError(f"local API returned HTTP {status} for {path}") from None
        except (URLError, TimeoutError, OSError) as exc:
            raise RunnerError(f"local API request failed for {path}: {exc}") from None
        if len(raw) > MAX_JSON_BYTES:
            raise RunnerError(f"local API returned oversized JSON for {path}")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RunnerError(f"local API returned invalid JSON for {path}") from None
        if not isinstance(payload, dict):
            raise RunnerError(f"local API returned a non-object JSON response for {path}")
        return payload, response_headers

    def download_png(self, path: str, expected_bytes: int) -> tuple[bytes, str]:
        request = Request(
            api_url(self.base_url, path),
            headers=self._headers("image/png"),
            method="GET",
        )
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get_content_type()
                raw = response.read(MAX_RESULT_BYTES + 1)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            raise RunnerError(f"local API returned HTTP {status} while downloading {path}") from None
        except (URLError, TimeoutError, OSError) as exc:
            raise RunnerError(f"result download failed for {path}: {exc}") from None
        if content_type != "image/png":
            raise RunnerError(f"result is not image/png: {path}")
        if len(raw) > MAX_RESULT_BYTES or len(raw) != expected_bytes:
            raise RunnerError(f"result byte length does not match metadata: {path}")
        return raw, content_type


def deterministic_idempotency_key(case: dict[str, Any], attempt_number: int) -> str:
    material = (
        "test4-math-local-dataset-v1\0"
        + str(case["id"])
        + "\0"
        + str(case["source_sha256"])
    ).encode("utf-8")
    return f"t4math-v1-{hashlib.sha256(material).hexdigest()}-a{attempt_number}"


def expected_poll_path(job_id: str) -> str:
    return f"/api/v1/math/jobs/{quote(job_id, safe='')}"


def expected_image_path(job_id: str, filename: str) -> str:
    return (
        f"/api/v1/math/jobs/{quote(job_id, safe='')}/images/"
        f"{quote(filename, safe='')}"
    )


def validate_capabilities(payload: dict[str, Any]) -> None:
    expected = [
        {"sequence": sequence, "kind": kind, "filename": filename}
        for sequence, kind, filename in EXPECTED_IMAGES
    ]
    upload = payload.get("upload")
    idempotency = upload.get("idempotency") if isinstance(upload, dict) else None
    subjects = payload.get("subjects")
    subject_ids = {
        item.get("id") for item in subjects if isinstance(item, dict)
    } if isinstance(subjects, list) else set()
    if (
        payload.get("api_version") != "v1"
        or "math" not in subject_ids
        or payload.get("workflow") != "asynchronous"
        or payload.get("results") != expected
        or not isinstance(upload, dict)
        or upload.get("raw_endpoint_suffix") != "/raw"
        or not isinstance(idempotency, dict)
        or idempotency.get("header") != "Idempotency-Key"
    ):
        raise RunnerError("local URL does not expose the required Test4 unified API contract")


def validate_job_envelope(payload: dict[str, Any], expected_job_id: str | None = None) -> str:
    job_id = payload.get("job_id")
    status = payload.get("status")
    if payload.get("api_version") != "v1" or payload.get("subject") != "math":
        raise RunnerError("job response has the wrong API version or subject")
    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise RunnerError("job response has an invalid job id")
    if expected_job_id is not None and job_id != expected_job_id:
        raise RunnerError("job response id does not match the requested job")
    if status not in ALLOWED_REMOTE_STATUSES:
        raise RunnerError("job response has an invalid status")
    if payload.get("poll_url") != expected_poll_path(job_id):
        raise RunnerError("job response has an unexpected poll URL")
    return job_id


def validate_completed_metadata(payload: dict[str, Any], job_id: str) -> list[dict[str, Any]]:
    contract = payload.get("result_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("name") != "test6-six-png"
        or contract.get("version") != "1"
        or contract.get("complete") is not True
    ):
        raise RunnerError("completed job does not declare a complete result contract")
    raw_images = payload.get("images")
    if not isinstance(raw_images, list) or len(raw_images) != len(EXPECTED_IMAGES):
        raise RunnerError("completed job does not contain exactly six result images")
    validated: list[dict[str, Any]] = []
    for raw, (sequence, kind, filename) in zip(raw_images, EXPECTED_IMAGES):
        if not isinstance(raw, dict):
            raise RunnerError("result image metadata is not an object")
        integer_width = raw.get("width")
        integer_height = raw.get("height")
        byte_count = raw.get("bytes")
        digest = raw.get("sha256")
        display_width = raw.get("display_width_cm")
        if (
            raw.get("sequence") != sequence
            or raw.get("kind") != kind
            or raw.get("filename") != filename
            or raw.get("media_type") != "image/png"
            or raw.get("url") != expected_image_path(job_id, filename)
            or not isinstance(integer_width, int)
            or isinstance(integer_width, bool)
            or integer_width != 800
            or not isinstance(integer_height, int)
            or isinstance(integer_height, bool)
            or integer_height <= 0
            or raw.get("mode") != "L"
            or ("format" in raw and raw.get("format") != "PNG")
            or not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count <= 0
            or byte_count > MAX_RESULT_BYTES
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
            or not isinstance(display_width, (int, float))
            or isinstance(display_width, bool)
            or not math.isfinite(float(display_width))
            or abs(float(display_width) - 9.0) > 0.001
        ):
            raise RunnerError(f"invalid six-image metadata for {filename}")
        validated.append(
            {
                "sequence": sequence,
                "kind": kind,
                "filename": filename,
                "media_type": "image/png",
                "url": expected_image_path(job_id, filename),
                "width": integer_width,
                "height": integer_height,
                "mode": "L",
                "format": "PNG",
                "bytes": byte_count,
                "sha256": digest,
                "display_width_cm": float(display_width),
            }
        )
    return validated


def validate_saved_metadata(metadata: Any, job_id: str) -> list[dict[str, Any]]:
    """Revalidate recovery metadata before any saved filename becomes a path."""

    if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
        raise RunnerError("saved completed metadata has an invalid job id")
    return validate_completed_metadata(
        {
            "result_contract": {
                "name": "test6-six-png",
                "version": "1",
                "complete": True,
            },
            "images": metadata,
        },
        job_id,
    )


def inspect_result_png(raw: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
    digest = sha256_bytes(raw)
    if digest != metadata["sha256"] or len(raw) != metadata["bytes"]:
        raise RunnerError(f"download hash/size mismatch for {metadata['filename']}")
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            image_format = image.format
            width, height = image.size
            mode = image.mode
            dpi = image.info.get("dpi")
    except (UnidentifiedImageError, OSError) as exc:
        raise RunnerError(f"download is not a readable PNG: {metadata['filename']}: {exc}") from None
    if image_format != "PNG" or width != 800 or height != metadata["height"] or mode != "L":
        raise RunnerError(f"download violates PNG/800px/L contract: {metadata['filename']}")
    if (
        not isinstance(dpi, (tuple, list))
        or len(dpi) < 2
        or not isinstance(dpi[0], (int, float))
        or not isinstance(dpi[1], (int, float))
        or not math.isfinite(float(dpi[0]))
        or not math.isfinite(float(dpi[1]))
        or float(dpi[0]) <= 0
        or float(dpi[1]) <= 0
    ):
        raise RunnerError(f"download is missing usable PNG DPI: {metadata['filename']}")
    dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
    physical_width_cm = width / dpi_x * 2.54
    if (
        abs(physical_width_cm - 9.0) > PHYSICAL_WIDTH_TOLERANCE_CM
        or abs(dpi_x - dpi_y) > 0.1
    ):
        raise RunnerError(f"download does not encode a 9 cm display width: {metadata['filename']}")
    return {
        "filename": metadata["filename"],
        "sha256": digest,
        "bytes": len(raw),
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
        "dpi": [round(dpi_x, 4), round(dpi_y, 4)],
        "physical_width_cm": round(physical_width_cm, 4),
        "verified_at": utc_now(),
    }


def initial_case_progress(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": case["id"],
        "phase": "pending",
        "source_sha256": case["source_sha256"],
        "source_bytes": case["source_bytes"],
        "current_attempt": 0,
        "attempts": [],
        "updated_at": utc_now(),
    }


def current_attempt(case_progress: dict[str, Any]) -> dict[str, Any] | None:
    attempts = case_progress.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return None
    attempt_number = case_progress.get("current_attempt")
    for attempt in reversed(attempts):
        if attempt.get("number") == attempt_number:
            return attempt
    raise RunnerError(f"progress has an invalid current attempt for {case_progress.get('id')}")


def case_directory(run_dir: Path, case_id: str) -> Path:
    return run_dir / "cases" / case_id


def build_summary(run: dict[str, Any], progress: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    cases = []
    elapsed_values: list[float] = []
    case_lookup = {case["id"]: case for case in run["cases"]}
    for case_id in progress["case_order"]:
        state = progress["cases"][case_id]
        phase = str(state.get("phase") or "unknown")
        counts[phase] = counts.get(phase, 0) + 1
        attempt = current_attempt(state)
        elapsed = attempt.get("elapsed_seconds") if attempt else None
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            elapsed_values.append(float(elapsed))
        source = case_lookup[case_id]
        cases.append(
            {
                "id": case_id,
                "grade": source.get("grade") or "",
                "group": source.get("group") or "",
                "tags": source.get("tags") or [],
                "phase": phase,
                "attempt": state.get("current_attempt", 0),
                "job_id": attempt.get("job_id") if attempt else None,
                "elapsed_seconds": elapsed,
                "verified_images": len((attempt or {}).get("downloads") or {}),
            }
        )
    elapsed_values.sort()
    performance: dict[str, Any] = {"reported_case_count": len(elapsed_values)}
    if elapsed_values:
        p50_index = max(0, (len(elapsed_values) * 50 + 99) // 100 - 1)
        p95_index = max(0, (len(elapsed_values) * 95 + 99) // 100 - 1)
        performance.update(
            {
                "minimum_seconds": round(elapsed_values[0], 3),
                "average_seconds": round(sum(elapsed_values) / len(elapsed_values), 3),
                "p50_seconds_nearest_rank": round(elapsed_values[p50_index], 3),
                "p95_seconds_nearest_rank": round(elapsed_values[p95_index], 3),
                "maximum_seconds": round(elapsed_values[-1], 3),
                "all_at_most_300_seconds": all(value <= 300 for value in elapsed_values),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": run["name"],
        "manifest_sha256": run["manifest_sha256"],
        "updated_at": progress["updated_at"],
        "total_cases": len(progress["case_order"]),
        "all_verified": counts.get("verified", 0) == len(progress["case_order"]),
        "verification_scope": "machine_six_image_contract_only",
        "manual_content_review_required": True,
        "usable_status": "unassessed",
        "counts": counts,
        "performance": performance,
        "cases": cases,
    }


def render_index(run: dict[str, Any], progress: dict[str, Any]) -> str:
    rows: list[str] = []
    case_lookup = {case["id"]: case for case in run["cases"]}
    for case_id in progress["case_order"]:
        source = case_lookup[case_id]
        state = progress["cases"][case_id]
        attempt = current_attempt(state)
        images = ""
        if attempt and isinstance(attempt.get("metadata"), list):
            links = []
            for item in attempt["metadata"]:
                filename = item["filename"]
                relative = f"cases/{quote(case_id)}/{quote(filename)}"
                links.append(
                    f'<a href="{relative}"><img src="{relative}" alt="{html.escape(filename)}"></a>'
                )
            images = '<div class="images">' + "".join(links) + "</div>"
        source_href = f"cases/{quote(case_id)}/{quote(source['source_copy_name'])}"
        attempt_number = state.get("current_attempt", 0)
        rows.append(
            "<section>"
            f"<h2>{html.escape(case_id)}</h2>"
            f"<p>状态：<strong>{html.escape(str(state.get('phase')))}</strong>；"
            f"尝试：{attempt_number}；<a href=\"{source_href}\">原题图</a></p>"
            f"{images}</section>"
        )
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;line-height:1.5;background:#f6f6f6;color:#222}}
main{{max-width:1200px;margin:auto}}section{{background:white;padding:16px;margin:16px 0;border-radius:8px}}
.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}}
.images img{{width:100%;height:auto;border:1px solid #bbb}}a{{color:#075fa8}}
</style></head><body><main><h1>{title}</h1>
<p>本页仅汇总机器合同校验结果；数学内容仍须按 manual_review.md 人工复核。</p>
{rows}</main></body></html>
""".format(title=html.escape(run["name"]), rows="\n".join(rows))


def render_manual_review(run: dict[str, Any]) -> str:
    lines = [
        f"# {run['name']} 人工复核表",
        "",
        "> 六图、哈希、尺寸、灰度和 9 cm 合同通过，不代表数学内容正确。每题六图均须人工查看。",
        "",
    ]
    for case in run["cases"]:
        case_id = case["id"]
        relative_root = f"cases/{case_id}"
        lines.extend(
            [
                f"## {case_id}",
                "",
                f"- 原题图：[{case['source_copy_name']}]({relative_root}/{case['source_copy_name']})",
                f"- 年级：{case.get('grade') or '未标注'}",
                f"- 标签：{', '.join(case.get('tags') or []) or '未标注'}",
                f"- 备注：{case.get('notes') or '无'}",
                f"- 参考：{case.get('reference') or '未提供'}",
                "",
                "- [ ] 原题识别完整，题号、小问、公式、图表无遗漏",
                "- [ ] 知识点与基础公式符合当前学段",
                "- [ ] 学霸层只提问，不出现知识点、公式、方法或解题方向",
                "- [ ] 中上层只点知识域，不说明如何应用",
                "- [ ] 中等层只点破题眼，不继续展开第二步",
                "- [ ] 中下层给出一致、可执行的解题骨架，并保留学生作答空白",
                "- [ ] 学渣层解释充分，但最终代入、运算或结论仍由学生完成",
                "- [ ] 五层同一小问的方法一致，没有互相矛盾或答案泄漏",
                "- [ ] 六图无原始 LaTeX、乱码、截边、重叠或不可读内容",
                "- [ ] 图形、表格、单位与题意一致",
                "- [ ] 人工结论：可用",
                "- 复核人：",
                "- 复核时间：",
                "- 问题记录：",
                "",
            ]
        )
    return "\n".join(lines)


def persist_run_artifacts(run_dir: Path, run: dict[str, Any], progress: dict[str, Any]) -> None:
    progress["updated_at"] = utc_now()
    for state in progress["cases"].values():
        if "updated_at" not in state:
            state["updated_at"] = progress["updated_at"]
    # progress.json is the canonical recovery record and is always replaced first.
    atomic_write_json(run_dir / "progress.json", progress)
    for case_id in progress["case_order"]:
        atomic_write_json(
            case_directory(run_dir, case_id) / "case.json",
            progress["cases"][case_id],
        )
    atomic_write_json(run_dir / "summary.json", build_summary(run, progress))
    atomic_write_text(run_dir / "index.html", render_index(run, progress))
    manual_path = run_dir / "manual_review.md"
    if not manual_path.exists():
        atomic_write_text(manual_path, render_manual_review(run))


def copy_case_sources(run_dir: Path, run: dict[str, Any]) -> None:
    for case in run["cases"]:
        destination = case_directory(run_dir, case["id"]) / case["source_copy_name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and sha256_file(destination) == case["source_sha256"]:
            continue
        source = Path(case["source_path"])
        if sha256_file(source) != case["source_sha256"]:
            raise RunnerError(f"source image changed after manifest validation: {source}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != case["source_sha256"]:
                raise RunnerError(f"source copy hash mismatch: {source}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def make_run_record(manifest: dict[str, Any], base_url: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "name": manifest["name"],
        "description": manifest["description"],
        "created_at": utc_now(),
        "manifest_path": manifest["path"],
        "manifest_sha256": manifest["sha256"],
        "base_url": base_url,
        "subject": "math",
        "serial_execution": True,
        "new_jobs_require_allow_provider_call": True,
        "credential_persisted": False,
        "expected_images": [
            {"sequence": sequence, "kind": kind, "filename": filename}
            for sequence, kind, filename in EXPECTED_IMAGES
        ],
        "expected_image_contract": {
            "count": 4,
            "width_px": 800,
            "mode": "L",
            "format": "PNG",
            "display_width_cm": 9.0,
        },
        "cases": manifest["cases"],
    }


def initialize_or_load_run(
    run_dir: Path, manifest: dict[str, Any], base_url: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_path = run_dir / "run.json"
    progress_path = run_dir / "progress.json"
    if run_path.exists() or progress_path.exists():
        if not run_path.is_file():
            raise RunnerError("run directory has progress but no run.json")
        run = read_json(run_path)
        if not isinstance(run, dict):
            raise RunnerError("run artifacts have invalid JSON shapes")
        if (
            run.get("schema_version") != SCHEMA_VERSION
            or run.get("manifest_sha256") != manifest["sha256"]
            or run.get("base_url") != base_url
            or run.get("cases") != manifest["cases"]
        ):
            raise RunnerError("existing run does not match this manifest/base URL")
        if not progress_path.is_file():
            # New-job submission is only possible after this initializer returns,
            # so a lone run.json is safely recoverable as an all-pending run.
            progress = {
                "schema_version": SCHEMA_VERSION,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "case_order": [case["id"] for case in manifest["cases"]],
                "cases": {
                    case["id"]: initial_case_progress(case)
                    for case in manifest["cases"]
                },
            }
            atomic_write_json(progress_path, progress)
        else:
            progress = read_json(progress_path)
        if not isinstance(progress, dict) or progress.get("schema_version") != SCHEMA_VERSION:
            raise RunnerError("run progress has an invalid JSON shape or schema")
        expected_order = [case["id"] for case in manifest["cases"]]
        if (
            progress.get("case_order") != expected_order
            or not isinstance(progress.get("cases"), dict)
            or set(progress["cases"]) != set(expected_order)
        ):
            raise RunnerError("existing progress case order does not match the manifest")
        for case in manifest["cases"]:
            state = (progress.get("cases") or {}).get(case["id"])
            if (
                not isinstance(state, dict)
                or state.get("source_sha256") != case["source_sha256"]
                or state.get("source_bytes") != case["source_bytes"]
            ):
                raise RunnerError(f"existing progress source mismatch for {case['id']}")
        copy_case_sources(run_dir, run)
        return run, progress

    run_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [item.name for item in run_dir.iterdir() if item.name != ".runner.lock"]
    if unexpected:
        raise RunnerError(
            "refusing to initialize a new run in a non-empty directory: "
            + ", ".join(sorted(unexpected)[:8])
        )
    run = make_run_record(manifest, base_url)
    progress = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "case_order": [case["id"] for case in manifest["cases"]],
        "cases": {
            case["id"]: initial_case_progress(case) for case in manifest["cases"]
        },
    }
    atomic_write_json(run_path, run)
    # Close the only initialization gap immediately.  Source copies and the
    # derived reports can be rebuilt from these two atomic canonical records.
    atomic_write_json(progress_path, progress)
    copy_case_sources(run_dir, run)
    persist_run_artifacts(run_dir, run, progress)
    return run, progress


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    lock_path = run_dir / ".runner.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+b")
    try:
        stream.seek(0)
        if stream.read(1) == b"":
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            raise RunnerError("another runner process already holds this run directory") from None
        yield
    finally:
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()


class DatasetRunner:
    def __init__(
        self,
        run_dir: Path,
        run: dict[str, Any],
        progress: dict[str, Any],
        client: LocalApiClient,
        *,
        allow_provider_call: bool,
        retry_failures: bool,
        max_new_jobs: int,
        job_timeout_seconds: float,
        poll_interval_seconds: float,
    ):
        self.run_dir = run_dir
        self.run = run
        self.progress = progress
        self.client = client
        self.allow_provider_call = allow_provider_call
        self.retry_failures = retry_failures
        self.max_new_jobs = max_new_jobs
        self.job_timeout_seconds = job_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.new_job_requests = 0
        self.capabilities_checked = False
        self.case_lookup = {case["id"]: case for case in run["cases"]}

    def persist(self, case_state: dict[str, Any] | None = None) -> None:
        if case_state is not None:
            case_state["updated_at"] = utc_now()
        persist_run_artifacts(self.run_dir, self.run, self.progress)

    def ensure_capabilities(self) -> None:
        if self.capabilities_checked:
            return
        payload, _ = self.client.request_json("/api/v1/capabilities")
        validate_capabilities(payload)
        self.capabilities_checked = True

    def new_attempt(self, case: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        number = len(state["attempts"]) + 1
        attempt = {
            "number": number,
            "phase": "create_intent",
            "idempotency_key": deterministic_idempotency_key(case, number),
            "intent_saved_at": utc_now(),
            "job_id": None,
            "remote_status": None,
            "metadata": [],
            "downloads": {},
            "elapsed_seconds": None,
            "last_error": None,
        }
        state["attempts"].append(attempt)
        state["current_attempt"] = number
        state["phase"] = "create_intent"
        self.persist(state)
        return attempt

    def submit_intent(
        self, case: dict[str, Any], state: dict[str, Any], attempt: dict[str, Any]
    ) -> bool:
        if not self.allow_provider_call or self.new_job_requests >= self.max_new_jobs:
            return False
        number = attempt.get("number")
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number <= 0
            or attempt.get("idempotency_key")
            != deterministic_idempotency_key(case, number)
        ):
            raise RunnerError(f"saved idempotency intent is invalid for {case['id']}")
        self.ensure_capabilities()
        source_path = case_directory(self.run_dir, case["id"]) / case["source_copy_name"]
        raw = source_path.read_bytes()
        if len(raw) != case["source_bytes"] or sha256_bytes(raw) != case["source_sha256"]:
            raise RunnerError(f"saved source copy changed for {case['id']}")
        self.new_job_requests += 1
        payload, headers = self.client.request_json(
            "/api/v1/math/jobs/raw",
            data=raw,
            content_type=case["source_media_type"],
            idempotency_key=attempt["idempotency_key"],
        )
        job_id = validate_job_envelope(payload)
        poll_path = expected_poll_path(job_id)
        if headers.get("location") != poll_path:
            raise RunnerError("create response has a missing or unexpected Location header")
        retry_after = headers.get("retry-after", "")
        if not retry_after.isdigit():
            raise RunnerError("create response is missing a numeric Retry-After header")
        replayed = headers.get("idempotency-replayed", "").lower()
        if replayed not in {"true", "false"}:
            raise RunnerError("create response is missing Idempotency-Replayed")
        attempt.update(
            {
                "job_id": job_id,
                "remote_status": payload["status"],
                "phase": payload["status"],
                "create_confirmed_at": utc_now(),
                "idempotency_replayed": replayed == "true",
                "last_error": None,
            }
        )
        state["phase"] = attempt["phase"]
        self.persist(state)
        print(f"[{case['id']}] job accepted; status={payload['status']}", flush=True)
        return True

    def poll_to_terminal(
        self, case: dict[str, Any], state: dict[str, Any], attempt: dict[str, Any]
    ) -> bool:
        job_id = attempt.get("job_id")
        if not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            raise RunnerError(f"cannot poll invalid saved job id for {case['id']}")
        self.ensure_capabilities()
        deadline = time.monotonic() + self.job_timeout_seconds
        while True:
            payload, headers = self.client.request_json(expected_poll_path(job_id))
            validate_job_envelope(payload, job_id)
            status = payload["status"]
            attempt["remote_status"] = status
            attempt["phase"] = status
            attempt["last_polled_at"] = utc_now()
            attempt["last_error"] = None
            elapsed = payload.get("elapsed_seconds")
            attempt["elapsed_seconds"] = (
                float(elapsed)
                if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                else None
            )
            state["phase"] = status
            self.persist(state)
            print(f"[{case['id']}] status={status}", flush=True)
            if status == "failed":
                attempt["finished_at"] = utc_now()
                attempt["last_error"] = "remote job failed"
                state["phase"] = "failed"
                self.persist(state)
                return True
            if status == "completed":
                try:
                    attempt["metadata"] = validate_completed_metadata(payload, job_id)
                except RunnerError as exc:
                    attempt["phase"] = "contract_failed"
                    attempt["last_error"] = str(exc)
                    attempt["finished_at"] = utc_now()
                    state["phase"] = "contract_failed"
                    self.persist(state)
                    return True
                attempt["phase"] = "downloading"
                state["phase"] = "downloading"
                self.persist(state)
                return self.download_results(case, state, attempt)
            if time.monotonic() >= deadline:
                attempt["last_error"] = "poll timeout; rerun this command to continue"
                self.persist(state)
                return False
            suggested = payload.get("retry_after_seconds") or headers.get("retry-after")
            try:
                wait_seconds = float(suggested)
            except (TypeError, ValueError):
                wait_seconds = self.poll_interval_seconds
            if not math.isfinite(wait_seconds):
                wait_seconds = self.poll_interval_seconds
            wait_seconds = max(0.25, min(wait_seconds, 10.0))
            time.sleep(wait_seconds)

    def download_results(
        self, case: dict[str, Any], state: dict[str, Any], attempt: dict[str, Any]
    ) -> bool:
        job_id = str(attempt["job_id"])
        try:
            metadata = validate_saved_metadata(attempt.get("metadata"), job_id)
        except RunnerError as exc:
            # Do not use any saved filename until the already-completed job is
            # polled again and returns a fresh canonical contract.
            attempt["metadata"] = []
            attempt["downloads"] = {}
            attempt["phase"] = "completed"
            attempt["last_error"] = f"saved result metadata rejected: {exc}"
            state["phase"] = "completed"
            self.persist(state)
            return False
        attempt["metadata"] = metadata
        # Rebuild from actual local bytes; never trust saved download keys.
        attempt["downloads"] = {}
        for item in metadata:
            filename = item["filename"]
            target = case_directory(self.run_dir, case["id"]) / filename
            if target.is_file():
                try:
                    local_raw = target.read_bytes()
                    verified = inspect_result_png(local_raw, item)
                    attempt["downloads"][filename] = verified
                    self.persist(state)
                    continue
                except (OSError, RunnerError):
                    pass
            try:
                self.ensure_capabilities()
                raw, _ = self.client.download_png(
                    expected_image_path(job_id, filename), item["bytes"]
                )
                verified = inspect_result_png(raw, item)
                atomic_write_bytes(target, raw)
                # Verify the bytes after the atomic replace as well.
                if sha256_file(target) != item["sha256"]:
                    raise RunnerError(f"saved result hash mismatch for {filename}")
                attempt["downloads"][filename] = verified
                attempt["last_error"] = None
                self.persist(state)
                print(f"[{case['id']}] verified {filename}", flush=True)
            except RunnerError as exc:
                attempt["phase"] = "download_error"
                attempt["last_error"] = str(exc)
                state["phase"] = "download_error"
                self.persist(state)
                return False
        expected_names = {filename for _, _, filename in EXPECTED_IMAGES}
        if set(attempt["downloads"]) != expected_names:
            attempt["phase"] = "download_error"
            attempt["last_error"] = "verified download set is not the canonical six-image set"
            state["phase"] = "download_error"
            self.persist(state)
            return False
        attempt["phase"] = "verified"
        attempt["finished_at"] = utc_now()
        attempt["last_error"] = None
        state["phase"] = "verified"
        self.persist(state)
        print(f"[{case['id']}] six-image contract verified", flush=True)
        return True

    def verify_saved_completed_cases(self) -> None:
        for case_id in self.progress["case_order"]:
            state = self.progress["cases"][case_id]
            if state.get("phase") != "verified":
                continue
            attempt = current_attempt(state)
            if not attempt:
                raise RunnerError(f"verified case has no attempt: {case_id}")
            try:
                metadata = validate_saved_metadata(
                    attempt.get("metadata"), str(attempt.get("job_id") or "")
                )
            except RunnerError as exc:
                state["phase"] = "completed"
                attempt["phase"] = "completed"
                attempt["metadata"] = []
                attempt["downloads"] = {}
                attempt["last_error"] = f"saved result metadata rejected: {exc}"
                self.persist(state)
                continue
            attempt["metadata"] = metadata
            valid = True
            rebuilt: dict[str, Any] = {}
            for item in metadata:
                target = case_directory(self.run_dir, case_id) / item["filename"]
                try:
                    rebuilt[item["filename"]] = inspect_result_png(target.read_bytes(), item)
                except (OSError, RunnerError):
                    valid = False
                    break
            if not valid:
                state["phase"] = "download_error"
                attempt["phase"] = "download_error"
                attempt["last_error"] = "local result missing or corrupt; redownload required"
                self.persist(state)
            else:
                attempt["downloads"] = rebuilt

    def resume_existing_attempts(self) -> bool:
        for case_id in self.progress["case_order"]:
            case = self.case_lookup[case_id]
            state = self.progress["cases"][case_id]
            attempt = current_attempt(state)
            if not attempt:
                continue
            phase = state.get("phase")
            if phase == "create_intent" and not attempt.get("job_id"):
                if not self.submit_intent(case, state, attempt):
                    return False
                if not self.poll_to_terminal(case, state, attempt):
                    return False
            elif phase == "create_intent" and attempt.get("job_id"):
                # Handles interruption after the accepted response was copied
                # into the attempt but before the enclosing case phase changed.
                if not self.poll_to_terminal(case, state, attempt):
                    return False
            elif phase in ALLOWED_REMOTE_STATUSES - {"completed", "failed"}:
                if not self.poll_to_terminal(case, state, attempt):
                    return False
            elif phase == "completed":
                # A process may be interrupted after the terminal poll was
                # persisted but before metadata/download state was saved.
                if isinstance(attempt.get("metadata"), list) and len(attempt["metadata"]) == len(EXPECTED_IMAGES):
                    if not self.download_results(case, state, attempt):
                        return False
                elif not self.poll_to_terminal(case, state, attempt):
                    return False
            elif phase == "downloading":
                if not self.download_results(case, state, attempt):
                    return False
            elif phase == "download_error":
                # A download failure never creates a second model job.
                if not self.download_results(case, state, attempt):
                    return False
        return True

    def run_all(self) -> None:
        self.verify_saved_completed_cases()
        if not self.resume_existing_attempts():
            return
        for case_id in self.progress["case_order"]:
            if self.new_job_requests >= self.max_new_jobs:
                break
            case = self.case_lookup[case_id]
            state = self.progress["cases"][case_id]
            phase = state.get("phase")
            if phase == "verified":
                continue
            if phase in TERMINAL_FAILURE_PHASES and not self.retry_failures:
                continue
            if phase not in {"pending", *TERMINAL_FAILURE_PHASES}:
                # An unresolved state was already handled above; do not create
                # another job if recovery did not finish cleanly.
                continue
            if not self.allow_provider_call:
                continue
            attempt = self.new_attempt(case, state)
            if not self.submit_intent(case, state, attempt):
                break
            if not self.poll_to_terminal(case, state, attempt):
                break


def read_credential(args: argparse.Namespace) -> str:
    if args.token_file:
        path = args.token_file.resolve()
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RunnerError(f"cannot read credential file: {exc}") from None
        if len(raw) > 4096:
            raise RunnerError("credential file is unexpectedly large")
        try:
            return raw.decode("utf-8-sig").strip()
        except UnicodeDecodeError:
            raise RunnerError("credential file is not UTF-8 text") from None
    return os.environ.get(args.token_env, "").strip()


def default_run_dir(manifest: dict[str, Any], output_root: Path) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(manifest["path"]).stem).strip("._-")
    if not safe_stem:
        safe_stem = "dataset"
    return output_root.resolve() / f"{safe_stem}_{manifest['sha256'][:12]}"


def dry_run_report(manifest: dict[str, Any], base_url: str, run_dir: Path) -> dict[str, Any]:
    return {
        "dry_run": True,
        "network_requests": 0,
        "provider_jobs_created": 0,
        "base_url": base_url,
        "run_dir": str(run_dir),
        "manifest_sha256": manifest["sha256"],
        "case_count": len(manifest["cases"]),
        "cases": [
            {
                "id": case["id"],
                "source_sha256": case["source_sha256"],
                "source_bytes": case["source_bytes"],
                "source_format": case["source_format"],
                "idempotency_key_attempt_1": deterministic_idempotency_key(case, 1),
            }
            for case in manifest["cases"]
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serial, resumable real-model dataset runner for the local Test4 mathematics API. "
            "New jobs are disabled unless --allow-provider-call is supplied."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Manifest example:
  {"name":"canary","cases":[
    {"id":"high_0401","image":"data/q_0401.png","grade":"high","tags":["trig"]}
  ]}

Exit codes: 0=all verified (or dry-run valid), 1=terminal case failure,
2=runner/API error, 3=work remains, 130=interrupted.

The case id and image hash form the deterministic create key. Give intentional
repeat trials distinct ids (for example high_0401_r1 and high_0401_r2).
""",
    )
    parser.add_argument("manifest", type=Path, help="UTF-8 JSON case manifest")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8130/",
        help="loopback unified portal origin (default: %(default)s)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="exact run directory; default is stable from manifest filename/hash",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "live_dataset_runs",
        help="parent used when --run-dir is omitted",
    )
    parser.add_argument(
        "--allow-provider-call",
        action="store_true",
        help="explicitly allow POST requests that may create paid model jobs",
    )
    parser.add_argument(
        "--max-new-jobs",
        type=int,
        default=1,
        help="maximum create/replay POSTs this invocation (default: %(default)s)",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="create a new deterministic attempt for failed/contract-failed cases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan; make no network request and write no run files",
    )
    parser.add_argument(
        "--token-env",
        default="TEST4_ACCESS_TOKEN",
        help="environment variable read at runtime; its value is never persisted",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        help="optional UTF-8 credential file; when supplied it overrides --token-env",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=90.0,
        help="seconds per local HTTP request (default: %(default)s)",
    )
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=900.0,
        help="seconds to poll one job in this invocation; state remains resumable",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="fallback polling interval in seconds (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_new_jobs < 0:
        parser.error("--max-new-jobs must be at least 0")
    timing_values = (args.request_timeout, args.job_timeout, args.poll_interval)
    if any(not math.isfinite(value) or value <= 0 for value in timing_values):
        parser.error("timeouts and poll interval must be finite and positive")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", args.token_env):
        parser.error("--token-env is not a valid environment variable name")
    if args.retry_failures and not args.allow_provider_call:
        parser.error("--retry-failures also requires --allow-provider-call")
    try:
        manifest = load_manifest(args.manifest)
        base_url = canonical_loopback_base(args.base_url)
        run_dir = (
            args.run_dir.resolve()
            if args.run_dir
            else default_run_dir(manifest, args.output_root)
        )
        if args.dry_run:
            print(json.dumps(dry_run_report(manifest, base_url, run_dir), ensure_ascii=False, indent=2))
            return 0

        credential = read_credential(args)
        run_dir.mkdir(parents=True, exist_ok=True)
        with run_lock(run_dir):
            run, progress = initialize_or_load_run(run_dir, manifest, base_url)
            runner = DatasetRunner(
                run_dir,
                run,
                progress,
                LocalApiClient(base_url, credential, args.request_timeout),
                allow_provider_call=args.allow_provider_call,
                retry_failures=args.retry_failures,
                max_new_jobs=args.max_new_jobs,
                job_timeout_seconds=args.job_timeout,
                poll_interval_seconds=args.poll_interval,
            )
            try:
                runner.run_all()
            except KeyboardInterrupt:
                runner.persist()
                print(f"interrupted; progress saved in {run_dir}", file=sys.stderr)
                return 130
            runner.persist()
            summary = build_summary(run, progress)
            print(
                json.dumps(
                    {
                        "run_dir": str(run_dir),
                        "new_job_requests": runner.new_job_requests,
                        "all_machine_contracts_verified": summary["all_verified"],
                        "manual_content_review_required": True,
                        "counts": summary["counts"],
                    },
                    ensure_ascii=False,
                )
            )
            if summary["all_verified"]:
                return 0
            if any(summary["counts"].get(value, 0) for value in TERMINAL_FAILURE_PHASES):
                return 1
            return 3
    except RunnerError as exc:
        print(f"runner error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"runner error: local file operation failed: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted before run initialization completed", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
