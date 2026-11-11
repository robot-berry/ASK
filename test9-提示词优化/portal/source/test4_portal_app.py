#!/usr/bin/env python3
"""Authenticated phone and hardware gateway for the Test6 streaming search machine."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import hmac
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request as UrlRequest,
    build_opener,
)

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from presentation_guard import public_display_text, source_markup_paths
from semester_curriculum import catalog as semester_catalog, validate_term, snapshot as semester_snapshot


SOURCE_DIR = Path(__file__).resolve().parent
STATIC_DIR = SOURCE_DIR / "static"
AUDIT_RUN_DIR = (
    SOURCE_DIR.parent.parent
    / "benchmark"
    / "ocr_accuracy_300_20260902"
)
AUDIT_REPORT_HTML = AUDIT_RUN_DIR / "FULL_SEMANTIC_AUDIT_300.html"
AUDIT_REPORT_MD = AUDIT_RUN_DIR / "FULL_SEMANTIC_AUDIT_300.md"
AUDIT_EVIDENCE_MD = AUDIT_RUN_DIR / "SEMANTIC_EVIDENCE_COVERAGE_300.md"
AUDIT_DATA_CSV = AUDIT_RUN_DIR / "full_audit_queue" / "full_manual_audit.csv"
AUDIT_PAGES_DIR = AUDIT_RUN_DIR / "full_audit_queue" / "manual_audit_pages"
LATEST_AUDIT_RUN_DIR = (
    SOURCE_DIR.parent.parent
    / "benchmark"
    / "ocr_accuracy_300_balanced_final_20260903"
)
LATEST_AUDIT_REPORT_HTML = LATEST_AUDIT_RUN_DIR / "STRUCTURED_OCR_REVIEW_300.html"
LATEST_AUDIT_REPORT_MD = LATEST_AUDIT_RUN_DIR / "LATEST_OCR_REPORT.md"
LATEST_AUDIT_DATA_CSV = (
    LATEST_AUDIT_RUN_DIR / "full_audit_queue" / "full_manual_audit.csv"
)
LATEST_AUDIT_PAGES_DIR = (
    LATEST_AUDIT_RUN_DIR / "full_audit_queue" / "manual_audit_pages"
)
ITERATION_AUDIT_RUN_DIR = (
    SOURCE_DIR.parent.parent
    / "benchmark"
    / "ocr_annotation_driven_full300_round16_final_20260907"
)
ITERATION_AUDIT_REPORT_HTML = ITERATION_AUDIT_RUN_DIR / "AUDIT_INDEX.html"
ITERATION_AUDIT_ANNOTATION_STORE_DIR = (
    SOURCE_DIR.parent.parent
    / "benchmark"
    / "ocr_round6_full300_review_20260907"
)
ITERATION_AUDIT_ANNOTATIONS_JSON = (
    ITERATION_AUDIT_ANNOTATION_STORE_DIR / "manual_annotations.json"
)
ITERATION_AUDIT_ANNOTATIONS_CSV = (
    ITERATION_AUDIT_ANNOTATION_STORE_DIR / "manual_audit_saved.csv"
)
ITERATION_AUDIT_GENERATED_DIR = ITERATION_AUDIT_RUN_DIR / "generated_results"
ITERATION_AUDIT_DATASET_DIR = (
    SOURCE_DIR.parents[3]
    / "datasets"
    / "testzip_math_physics_300_v1"
    / "Test"
)

ITERATION_AUDIT_ERROR_TYPES = {
    "text_missing",
    "text_error",
    "formula_error",
    "layout_error",
    "figure_missing",
    "figure_incomplete",
    "handwriting_leak",
    "table_error",
    "other",
}

def _loopback_backend(setting: str, default: str) -> str:
    """Allow deployment-specific ports while rejecting non-loopback targets."""

    raw = os.environ.get(setting, "").strip() or default
    parsed = urlparse(raw)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
        or parsed.port is None
    ):
        raise RuntimeError(f"{setting} 必须是带端口的HTTP回环地址")
    return f"http://127.0.0.1:{parsed.port}"


# No request field is ever accepted as a host or port, so this proxy cannot be
# used as a generic network relay.
BACKENDS = {
    "physics": _loopback_backend(
        "TEST6_PHYSICS_BACKEND_URL",
        os.environ.get("TEST4_PHYSICS_BACKEND_URL", "").strip()
        or "http://127.0.0.1:8010",
    ),
    "math": _loopback_backend(
        "TEST6_MATH_BACKEND_URL",
        os.environ.get("TEST4_MATH_BACKEND_URL", "").strip()
        or "http://127.0.0.1:8020",
    ),
}
SUBJECT_LABELS = {"physics": "物理", "math": "数学"}
EXPECTED_BACKEND_SERVICES = {
    "physics": "test6-physics-search-machine",
    "math": "test6-math-search-machine",
}

API_VERSION = "v1"
SERVICE_VERSION = "9.1.9-implicit-multiplication-siliconflow"
ACCESS_TOKEN = (
    os.environ.get("TEST6_ACCESS_TOKEN", "").strip()
    or os.environ.get("TEST4_ACCESS_TOKEN", "").strip()
)
RAW_DEVICE_TOKEN_HASHES = [
    item.strip().lower()
    for item in (
        os.environ.get("TEST6_DEVICE_TOKEN_HASHES", "").strip()
        or os.environ.get("TEST4_DEVICE_TOKEN_HASHES", "")
    ).split(",")
    if item.strip()
]
if any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in RAW_DEVICE_TOKEN_HASHES):
    raise RuntimeError("TEST6_DEVICE_TOKEN_HASHES 必须是逗号分隔的SHA-256十六进制值")
DEVICE_TOKEN_HASHES = tuple(RAW_DEVICE_TOKEN_HASHES)
ALLOW_INSECURE_NO_AUTH = os.environ.get(
    "TEST6_ALLOW_INSECURE_NO_AUTH",
    os.environ.get("TEST4_ALLOW_INSECURE_NO_AUTH", ""),
).strip().lower() in {"1", "true", "yes"}
COOKIE_SECURE = os.environ.get(
    "TEST6_COOKIE_SECURE",
    os.environ.get("TEST4_COOKIE_SECURE", "1"),
).strip().lower() not in {
    "0",
    "false",
    "no",
}
COOKIE_NAME = "__Host-test6_session" if COOKIE_SECURE else "test6_session"
SESSION_TTL_SECONDS = max(
    300,
    min(
        30 * 24 * 3600,
        int(
            os.environ.get(
                "TEST6_SESSION_TTL_SECONDS",
                os.environ.get("TEST4_SESSION_TTL_SECONDS", "604800"),
            )
        ),
    ),
)
if not ACCESS_TOKEN and not ALLOW_INSECURE_NO_AUTH:
    raise RuntimeError(
        "TEST6_ACCESS_TOKEN 未配置；公网部署必须设置至少24位随机访问令牌"
    )
if ACCESS_TOKEN.startswith("replace_"):
    raise RuntimeError("TEST6_ACCESS_TOKEN 仍是示例值")
if ACCESS_TOKEN and len(ACCESS_TOKEN) < 24:
    raise RuntimeError("TEST6_ACCESS_TOKEN 至少需要24个字符")


def _session_signature(payload: str) -> str:
    return hmac.new(
        ACCESS_TOKEN.encode("utf-8"),
        ("test6-browser-session-v1:" + payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _encode_session(expires_at: int, nonce: str) -> str:
    payload = f"{expires_at}.{nonce}"
    return f"{payload}.{_session_signature(payload)}"


def _new_session_value() -> str:
    return _encode_session(
        int(time.time()) + SESSION_TTL_SECONDS,
        secrets.token_hex(16),
    )


def _valid_session(value: str) -> bool:
    if not value or len(value) > 256:
        return False
    try:
        expires_raw, nonce, signature = value.split(".", 2)
        expires_at = int(expires_raw)
    except (TypeError, ValueError):
        return False
    if expires_at < int(time.time()) or not re.fullmatch(r"[0-9a-f]{32}", nonce):
        return False
    payload = f"{expires_at}.{nonce}"
    return secrets.compare_digest(signature, _session_signature(payload))

DEVICE_IMAGE_NAMES = (
    "01_question_knowledge.png",
    "02_guidance_elite.png",
    "03_guidance_upper.png",
    "04_guidance_middle.png",
    "05_guidance_lower.png",
    "06_guidance_novice.png",
)
DEVICE_IMAGE_KINDS = (
    "question_knowledge",
    "elite",
    "upper",
    "middle",
    "lower",
    "novice",
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 20 * 1024 * 1024
MAX_MULTIPART_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
UPSTREAM_TIMEOUT_SECONDS = 20
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
IMAGE_MEDIA_TYPE_PATTERN = re.compile(r"^image/[a-z0-9][a-z0-9.+-]{0,63}$")
IDEMPOTENCY_KEY_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._~-]{7,127}$",
    re.ASCII,
)
DEVICE_SN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$", re.ASCII)
IDEMPOTENCY_PENDING_LEASE_SECONDS = UPSTREAM_TIMEOUT_SECONDS + 15
IDEMPOTENCY_WAIT_SECONDS = UPSTREAM_TIMEOUT_SECONDS + 20
IDEMPOTENCY_RETENTION_SECONDS = 7 * 24 * 3600
_state_db_raw = (
    os.environ.get("TEST6_PORTAL_STATE_DB", "").strip()
    or os.environ.get("TEST4_PORTAL_STATE_DB", "").strip()
)
PORTAL_STATE_DB = Path(_state_db_raw) if _state_db_raw else (
    SOURCE_DIR / ".state" / "gateway.sqlite3"
)
if not PORTAL_STATE_DB.is_absolute():
    raise RuntimeError("TEST6_PORTAL_STATE_DB must be an absolute path")
ALLOWED_UPLOAD_MEDIA_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)
ALLOWED_STATUSES = {
    "queued",
    "preprocessing",
    "solving",
    "rendering",
    "completed",
    "failed",
}


class UpstreamFailure(Exception):
    """A safe, user-facing representation of an upstream failure."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class UploadBodyTooLarge(Exception):
    """Raised before multipart parsing can spool an oversized request."""


class IdempotencyConflict(Exception):
    """The same principal/subject/key was already used for another image."""


class IdempotencyStoreFailure(Exception):
    """The durable idempotency state could not be read or updated safely."""


def _open_idempotency_db() -> sqlite3.Connection:
    connection = sqlite3.connect(
        str(PORTAL_STATE_DB),
        timeout=5,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def _initialize_idempotency_db() -> None:
    connection: sqlite3.Connection | None = None
    try:
        PORTAL_STATE_DB.parent.mkdir(parents=True, exist_ok=True)
        connection = _open_idempotency_db()
        journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(journal_mode).lower() != "wal":
            raise IdempotencyStoreFailure("SQLite WAL mode is unavailable")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_jobs (
                principal_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                key_sha256 TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('pending', 'ready')),
                owner_nonce TEXT,
                lease_expires REAL,
                response_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (principal_id, subject, key_sha256)
            ) WITHOUT ROWID
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idempotency_jobs_updated_at
            ON idempotency_jobs(updated_at)
            """
        )
    except (IdempotencyStoreFailure, OSError, sqlite3.Error) as exc:
        raise RuntimeError(
            "TEST6_PORTAL_STATE_DB could not be initialized"
        ) from exc
    finally:
        if connection is not None:
            connection.close()


_initialize_idempotency_db()


class UploadBodyLimitMiddleware:
    """Enforce the upload limit while ASGI request chunks are arriving."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = str(scope.get("path") or "")
        limited = (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and re.fullmatch(
                r"/api/v1/(?:physics|math)/jobs(?:/raw)?", path
            )
            is not None
        )
        if not limited:
            await self.app(scope, receive, send)
            return
        headers = {
            key.lower(): value
            for key, value in scope.get("headers") or []
        }
        raw_length = headers.get(b"content-length", b"")
        if raw_length.isdigit() and int(raw_length) > MAX_MULTIPART_BYTES:
            response = JSONResponse(
                status_code=413,
                content={"detail": "图片不能超过 20 MB"},
                headers={"Cache-Control": "no-store", "Connection": "close"},
            )
            await response(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body") or b"")
                if received > MAX_MULTIPART_BYTES:
                    raise UploadBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except UploadBodyTooLarge:
            response = JSONResponse(
                status_code=413,
                content={"detail": "图片不能超过 20 MB"},
                headers={"Cache-Control": "no-store", "Connection": "close"},
            )
            await response(scope, receive, send)


class _NoRedirectHandler(HTTPRedirectHandler):
    """Keep every upstream request on its allow-listed loopback origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


# Disable environment-configured proxies and HTTP redirects.  This is stronger
# than checking only the initial URL: every network hop stays on the two fixed
# loopback services above.
_UPSTREAM_OPENER = build_opener(ProxyHandler({}), _NoRedirectHandler())


def backend_for_subject(subject: str) -> str:
    """Resolve a subject through the fixed backend allow-list."""

    backend = BACKENDS.get(subject)
    if backend is None:
        raise HTTPException(404, "请选择物理或数学")
    return backend


def validate_job_id(job_id: str) -> str:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise HTTPException(400, "任务编号格式不正确")
    return job_id


def validate_result_filename(filename: str) -> str:
    if filename not in DEVICE_IMAGE_NAMES:
        raise HTTPException(404, "结果图不存在")
    return filename


def validate_upload(data: bytes, content_type: str | None) -> str:
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    if (
        not IMAGE_MEDIA_TYPE_PATTERN.fullmatch(media_type)
        or media_type not in ALLOWED_UPLOAD_MEDIA_TYPES
    ):
        raise HTTPException(415, "仅支持 JPEG、PNG 或 WebP 图片")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "图片不能超过 20 MB")
    if not data:
        raise HTTPException(400, "请选择一张题目图片")
    return media_type


def idempotency_key_digest(request: Request) -> str | None:
    """Validate and hash the optional header without persisting its raw value."""

    values = request.headers.getlist("idempotency-key")
    if not values:
        return None
    if len(values) != 1 or IDEMPOTENCY_KEY_PATTERN.fullmatch(values[0]) is None:
        raise HTTPException(
            400,
            "Idempotency-Key 必须是 8-128 位 ASCII 字母、数字或 . _ ~ -",
        )
    return hashlib.sha256(values[0].encode("ascii")).hexdigest()


def _stored_idempotency_response(raw: str | None, subject: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw or "")
    except json.JSONDecodeError as exc:
        raise IdempotencyStoreFailure("stored response is invalid") from exc
    if not isinstance(payload, dict):
        raise IdempotencyStoreFailure("stored response is invalid")
    job_id = payload.get("job_id")
    expected_poll = f"/api/v1/{subject}/jobs/{quote(str(job_id or ''), safe='')}"
    if (
        payload.get("subject") != subject
        or not isinstance(job_id, str)
        or JOB_ID_PATTERN.fullmatch(job_id) is None
        or payload.get("poll_url") != expected_poll
    ):
        raise IdempotencyStoreFailure("stored response is invalid")
    return payload


def _claim_idempotency(
    principal_id: str,
    subject: str,
    key_sha256: str,
    image_sha256: str,
) -> tuple[str, str | None, dict[str, Any] | None]:
    """Atomically claim a key, observe a pending owner, or replay a result."""

    now = time.time()
    owner_nonce = secrets.token_hex(16)
    connection: sqlite3.Connection | None = None
    try:
        connection = _open_idempotency_db()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "DELETE FROM idempotency_jobs WHERE updated_at < ?",
            (now - IDEMPOTENCY_RETENTION_SECONDS,),
        )
        row = connection.execute(
            """
            SELECT image_sha256, state, owner_nonce, lease_expires, response_json
            FROM idempotency_jobs
            WHERE principal_id = ? AND subject = ? AND key_sha256 = ?
            """,
            (principal_id, subject, key_sha256),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO idempotency_jobs (
                    principal_id, subject, key_sha256, image_sha256, state,
                    owner_nonce, lease_expires, response_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'pending', ?, ?, NULL, ?, ?)
                """,
                (
                    principal_id,
                    subject,
                    key_sha256,
                    image_sha256,
                    owner_nonce,
                    now + IDEMPOTENCY_PENDING_LEASE_SECONDS,
                    now,
                    now,
                ),
            )
            connection.commit()
            return "owner", owner_nonce, None

        if not secrets.compare_digest(str(row["image_sha256"]), image_sha256):
            raise IdempotencyConflict
        if row["state"] == "ready":
            payload = _stored_idempotency_response(row["response_json"], subject)
            connection.commit()
            return "ready", None, payload
        lease_expires = float(row["lease_expires"] or 0)
        if lease_expires > now:
            connection.commit()
            return "pending", None, None
        connection.execute(
            """
            UPDATE idempotency_jobs
            SET owner_nonce = ?, lease_expires = ?, updated_at = ?
            WHERE principal_id = ? AND subject = ? AND key_sha256 = ?
              AND state = 'pending'
            """,
            (
                owner_nonce,
                now + IDEMPOTENCY_PENDING_LEASE_SECONDS,
                now,
                principal_id,
                subject,
                key_sha256,
            ),
        )
        connection.commit()
        return "owner", owner_nonce, None
    except IdempotencyConflict:
        if connection is not None and connection.in_transaction:
            connection.rollback()
        raise
    except (IdempotencyStoreFailure, OSError, sqlite3.Error, TypeError, ValueError) as exc:
        if connection is not None and connection.in_transaction:
            connection.rollback()
        if isinstance(exc, IdempotencyStoreFailure):
            raise
        raise IdempotencyStoreFailure("idempotency state is unavailable") from exc
    finally:
        if connection is not None:
            connection.close()


def _complete_idempotency(
    principal_id: str,
    subject: str,
    key_sha256: str,
    image_sha256: str,
    owner_nonce: str,
    response: dict[str, Any],
) -> None:
    now = time.time()
    encoded = json.dumps(
        response,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    connection: sqlite3.Connection | None = None
    try:
        connection = _open_idempotency_db()
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            """
            UPDATE idempotency_jobs
            SET state = 'ready', owner_nonce = NULL, lease_expires = NULL,
                response_json = ?, updated_at = ?
            WHERE principal_id = ? AND subject = ? AND key_sha256 = ?
              AND image_sha256 = ? AND state = 'pending' AND owner_nonce = ?
            """,
            (
                encoded,
                now,
                principal_id,
                subject,
                key_sha256,
                image_sha256,
                owner_nonce,
            ),
        )
        if cursor.rowcount != 1:
            raise IdempotencyStoreFailure("idempotency owner changed unexpectedly")
        connection.commit()
    except (IdempotencyStoreFailure, OSError, sqlite3.Error) as exc:
        if connection is not None and connection.in_transaction:
            connection.rollback()
        if isinstance(exc, IdempotencyStoreFailure):
            raise
        raise IdempotencyStoreFailure("idempotency state is unavailable") from exc
    finally:
        if connection is not None:
            connection.close()


def _release_idempotency(
    principal_id: str,
    subject: str,
    key_sha256: str,
    image_sha256: str,
    owner_nonce: str,
) -> None:
    """Release only this request's pending claim after upstream rejection."""

    connection: sqlite3.Connection | None = None
    try:
        connection = _open_idempotency_db()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            DELETE FROM idempotency_jobs
            WHERE principal_id = ? AND subject = ? AND key_sha256 = ?
              AND image_sha256 = ? AND state = 'pending' AND owner_nonce = ?
            """,
            (principal_id, subject, key_sha256, image_sha256, owner_nonce),
        )
        connection.commit()
    except (OSError, sqlite3.Error) as exc:
        if connection is not None and connection.in_transaction:
            connection.rollback()
        raise IdempotencyStoreFailure("idempotency state is unavailable") from exc
    finally:
        if connection is not None:
            connection.close()


def _friendly_upstream_error(status_code: int, subject: str) -> str:
    label = SUBJECT_LABELS[subject]
    messages = {
        400: "题图无法识别，请换一张清晰、完整的照片",
        404: "任务或结果不存在，可能已被清理",
        413: "图片不能超过 20 MB",
        415: "仅支持图片文件",
        429: f"{label}任务较多，请稍后重试",
        503: f"{label}处理服务暂时不可用，请稍后重试",
    }
    return messages.get(status_code, f"{label}处理服务暂时无法完成请求，请稍后重试")


def _read_limited(stream: Any, limit: int, too_large_message: str) -> bytes:
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise UpstreamFailure(502, too_large_message)
    return data


def _request_json(
    subject: str,
    path: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Make a bounded JSON request to one fixed upstream service."""

    url = backend_for_subject(subject) + path
    request = UrlRequest(url, data=data, headers=headers or {}, method=method)
    try:
        with _UPSTREAM_OPENER.open(
            request, timeout=UPSTREAM_TIMEOUT_SECONDS
        ) as response:
            raw = _read_limited(
                response,
                MAX_JSON_BYTES,
                "处理服务返回的数据异常，请稍后重试",
            )
    except HTTPError as exc:
        try:
            exc.close()
        finally:
            raise UpstreamFailure(
                exc.code,
                _friendly_upstream_error(exc.code, subject),
            ) from None
    except (URLError, TimeoutError, OSError):
        raise UpstreamFailure(
            503,
            f"{SUBJECT_LABELS[subject]}处理服务暂时无法连接，请稍后重试",
        ) from None

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise UpstreamFailure(502, "处理服务返回的数据无法识别，请稍后重试") from None
    if not isinstance(payload, dict):
        raise UpstreamFailure(502, "处理服务返回的数据格式不正确，请稍后重试")
    return payload


def _multipart_upload(data: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = "----test6-" + secrets.token_hex(18)
    extension = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }[content_type.lower()]
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="camera.{extension}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("ascii")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    return prefix + data + suffix, boundary


def _submit_job_sync(
    subject: str,
    data: bytes,
    content_type: str,
    principal_id: str,
    device_sn: str | None,
    curriculum_term: str = "",
) -> dict[str, Any]:
    body, boundary = _multipart_upload(data, content_type)
    return _request_json(
        subject,
        "/api/jobs",
        method="POST",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "X-Test6-Device-Id": principal_id,
            **({"X-Curriculum-Term": curriculum_term} if curriculum_term else {}),
            **({"X-Device-SN": device_sn} if device_sn else {}),
        },
    )


def _fetch_job_sync(subject: str, job_id: str) -> dict[str, Any]:
    safe_job_id = quote(validate_job_id(job_id), safe="")
    return _request_json(subject, f"/api/jobs/{safe_job_id}")


def _validated_result_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Enforce metadata required by hardware before declaring a job complete."""

    width = item.get("width")
    height = item.get("height")
    mode = item.get("mode")
    image_format = item.get("format")
    byte_count = item.get("bytes")
    digest = item.get("sha256")
    display_width = item.get("display_width_cm")
    valid = (
        isinstance(width, int)
        and not isinstance(width, bool)
        and width == 800
        and isinstance(height, int)
        and not isinstance(height, bool)
        and height > 0
        and mode == "L"
        and image_format == "PNG"
        and isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and byte_count > 0
        and isinstance(digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and isinstance(display_width, (int, float))
        and not isinstance(display_width, bool)
        and abs(float(display_width) - 9.0) <= 0.001
    )
    if not valid:
        raise UpstreamFailure(502, "结果图元数据不符合 Test6 六图合同，请重新提交")
    return {
        "width": width,
        "height": height,
        "mode": mode,
        "bytes": byte_count,
        "sha256": digest,
        "display_width_cm": float(display_width),
    }


def _analysis_string(value: Any, *, field: str, limit: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise UpstreamFailure(502, f"模型分析字段 {field} 格式不正确")
    text = value.replace("\x00", "").replace("\r", " ").strip()
    if not text and not allow_empty:
        raise UpstreamFailure(502, f"模型分析字段 {field} 为空")
    if len(text) > limit:
        raise UpstreamFailure(502, f"模型分析字段 {field} 过长")
    public_text = public_display_text(text, limit=limit)
    if not public_text and not allow_empty:
        raise UpstreamFailure(502, f"模型分析字段 {field} 为空")
    return public_text


def _sanitize_audit_formulas(value: Any) -> list[dict[str, str]]:
    raw_formulas = value or []
    if not isinstance(raw_formulas, list) or len(raw_formulas) > 4:
        raise UpstreamFailure(502, "模型分析公式格式不正确")
    formulas: list[dict[str, str]] = []
    for item in raw_formulas:
        if not isinstance(item, dict):
            raise UpstreamFailure(502, "模型分析公式项格式不正确")
        formulas.append({
            "formula": _analysis_string(item.get("formula"), field="formula", limit=180),
            "condition": _analysis_string(
                item.get("condition", ""),
                field="formula_condition",
                limit=180,
                allow_empty=True,
            ),
        })
    return formulas


def _sanitize_audit_parts(value: Any, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    raw_parts = value
    if not isinstance(raw_parts, list) or len(raw_parts) > 16:
        raise UpstreamFailure(502, "模型分析小问格式不正确")
    if not raw_parts and not allow_empty:
        raise UpstreamFailure(502, "模型分析小问格式不正确")
    parts: list[dict[str, Any]] = []
    for raw_part in raw_parts:
        if not isinstance(raw_part, dict):
            raise UpstreamFailure(502, "模型分析小问项格式不正确")
        raw_methods = raw_part.get("methods")
        if not isinstance(raw_methods, list) or not raw_methods or len(raw_methods) > 3:
            raise UpstreamFailure(502, "模型分析方法格式不正确")
        methods: list[dict[str, Any]] = []
        for raw_method in raw_methods:
            if not isinstance(raw_method, dict):
                raise UpstreamFailure(502, "模型分析方法项格式不正确")
            raw_steps = raw_method.get("steps")
            if not isinstance(raw_steps, list) or not raw_steps or len(raw_steps) > 8:
                raise UpstreamFailure(502, "模型分析步骤格式不正确")
            methods.append({
                "method_id": _analysis_string(
                    raw_method.get("method_id", ""),
                    field="method_id",
                    limit=24,
                    allow_empty=True,
                ),
                "method_name": _analysis_string(
                    raw_method.get("method_name", ""),
                    field="method_name",
                    limit=80,
                    allow_empty=True,
                ),
                "steps": [
                    _analysis_string(step, field="analysis_step", limit=320)
                    for step in raw_steps
                ],
                "pitfall": _analysis_string(
                    raw_method.get("pitfall", ""),
                    field="pitfall",
                    limit=180,
                    allow_empty=True,
                ),
            })
        parts.append({
            "part": _analysis_string(raw_part.get("part"), field="part", limit=40),
            "methods": methods,
        })
    return parts


def sanitize_analysis_audit(value: Any) -> dict[str, Any] | None:
    """Validate the browser-only, answer-blind analysis text from a backend."""

    if value is None:
        return None
    if not isinstance(value, dict):
        raise UpstreamFailure(502, "模型分析检查数据格式不正确")
    source = value.get("source")
    source_labels = {
        "complete_model": "完整模型自然语言分析",
        "partial_model": "截止前模型自然语言分析",
        "local_fallback": "本地兜底（不是模型原文）",
        "validated_output": "校验后的公共骨架",
    }
    source_notices = {
        "complete_model": "以下五档自然语言均由模型分别生成，并经过本地答案脱敏与强校验。",
        "partial_model": "模型已在截止前完整返回五档自然语言；明确终值已脱敏，缺失题面元数据由本地补齐。",
        "local_fallback": "模型没有完整返回五档解析，以下内容来自本地兜底，不能作为模型五档质量依据。",
        "validated_output": "以下是通过答案脱敏与强校验后的五档分析。",
    }
    if source not in source_labels:
        raise UpstreamFailure(502, "模型分析来源标记不正确")
    model_id = _analysis_string(
        value.get("model_id", ""), field="model_id", limit=80, allow_empty=True
    )
    raw_knowledge = value.get("knowledge_points") or []
    if not isinstance(raw_knowledge, list) or len(raw_knowledge) > 6:
        raise UpstreamFailure(502, "模型分析知识点格式不正确")
    knowledge_points = [
        _analysis_string(item, field="knowledge_point", limit=100)
        for item in raw_knowledge
    ]
    formulas = _sanitize_audit_formulas(value.get("base_formulas"))
    parts = _sanitize_audit_parts(value.get("parts"))
    result = {
        "source": source,
        "source_label": source_labels[source],
        "notice": source_notices[source],
        "model_id": model_id,
        "knowledge_points": knowledge_points,
        "base_formulas": formulas,
        "parts": parts,
    }
    raw_tiers = value.get("tiers")
    if raw_tiers is not None:
        if not isinstance(raw_tiers, list) or len(raw_tiers) != 5:
            raise UpstreamFailure(502, "模型五档分析格式不正确")
        tier_order = ("elite", "upper", "middle", "lower", "novice")
        tier_titles = {
            "elite": "学霸解析",
            "upper": "中上解析",
            "middle": "中等解析",
            "lower": "中下解析",
            "novice": "学渣解析",
        }
        tiers: list[dict[str, Any]] = []
        for expected_key, raw_tier in zip(tier_order, raw_tiers):
            if not isinstance(raw_tier, dict) or raw_tier.get("tier_key") != expected_key:
                raise UpstreamFailure(502, "模型五档分析顺序不正确")
            raw_tier_knowledge = raw_tier.get("knowledge_points") or []
            if not isinstance(raw_tier_knowledge, list) or len(raw_tier_knowledge) > 6:
                raise UpstreamFailure(502, "模型五档知识点格式不正确")
            tier_parts = _sanitize_audit_parts(
                raw_tier.get("parts"), allow_empty=expected_key == "upper"
            )
            if expected_key != "upper" and not tier_parts:
                raise UpstreamFailure(502, "模型五档小问不能为空")
            tiers.append({
                "tier_key": expected_key,
                "title": tier_titles[expected_key],
                "knowledge_points": [
                    _analysis_string(item, field="tier_knowledge", limit=100)
                    for item in raw_tier_knowledge
                ],
                "base_formulas": _sanitize_audit_formulas(
                    raw_tier.get("base_formulas")
                ),
                "parts": tier_parts,
                "pattern_summary": _analysis_string(
                    raw_tier.get("pattern_summary", ""),
                    field="pattern_summary",
                    limit=220,
                    allow_empty=True,
                ),
            })
        result["tiers"] = tiers
    if len(json.dumps(result, ensure_ascii=False)) > 24000:
        raise UpstreamFailure(502, "模型分析检查数据过长")
    if source_markup_paths(result):
        raise UpstreamFailure(502, "模型分析显示文本包含未处理的内部公式标记")
    return result


def _public_seconds(value: Any, *, milliseconds: bool = False) -> float | None:
    """Return a bounded, finite public timing value in seconds."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = float(value) / (1000.0 if milliseconds else 1.0)
    if not math.isfinite(seconds) or seconds < 0 or seconds > 24 * 60 * 60:
        return None
    return round(seconds, 3)


def sanitize_job(
    subject: str,
    payload: dict[str, Any],
    *,
    expected_job_id: str | None = None,
) -> dict[str, Any]:
    """Return only public state and rewrite every image to this origin."""

    backend_for_subject(subject)
    job_id = str(payload.get("job_id") or "")
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise UpstreamFailure(502, "处理服务返回的任务编号不正确，请重新提交")
    if expected_job_id is not None and job_id != expected_job_id:
        raise UpstreamFailure(502, "处理服务返回了不匹配的任务，请重新提交")

    status = str(payload.get("status") or "")
    if status not in ALLOWED_STATUSES:
        raise UpstreamFailure(502, "处理服务返回的任务状态不正确，请稍后重试")

    elapsed_seconds = _public_seconds(payload.get("elapsed_seconds"))
    public: dict[str, Any] = {
        "api_version": API_VERSION,
        "subject": subject,
        "subject_label": SUBJECT_LABELS[subject],
        "curriculum_selection": {key: (payload.get("curriculum_selection") or {}).get(key) for key in ("id", "label", "revision")},
        "job_id": job_id,
        "status": status,
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "elapsed_seconds": elapsed_seconds,
        "error": None,
        "images": [],
        "retry_after_seconds": 2,
        "poll_url": f"/api/v1/{subject}/jobs/{quote(job_id, safe='')}",
        "result_contract": {
            "name": "test6-six-png",
            "version": "1",
            "complete": status == "completed",
            "available_images": 0,
        },
    }
    guidance = payload.get("prompt_guidance") or (payload.get("preprocess") or {}).get("test10_prompt")
    if isinstance(guidance, dict):
        public["prompt_guidance"] = {
            key: guidance.get(key) for key in ("status", "topic", "term", "revision")
        }
        public["prompt_guidance"]["sources"] = [
            {"path": str(item.get("path", "")), "sha256": str(item.get("sha256", ""))}
            for item in (guidance.get("sources") or []) if isinstance(item, dict)
        ]
    else:
        public["prompt_guidance"] = {"status": "not_recorded", "sources": []}
    raw_streaming = payload.get("streaming")
    if isinstance(raw_streaming, dict):
        first_image_seconds = _public_seconds(
            raw_streaming.get("first_image_seconds")
        )
        first_image_from_created_seconds = _public_seconds(
            raw_streaming.get("first_image_from_created_seconds")
        )
        first_image_refined_seconds = _public_seconds(
            raw_streaming.get("first_image_refined_seconds")
        )
        queue_wait_seconds = None
        if (
            first_image_seconds is not None
            and first_image_from_created_seconds is not None
        ):
            queue_wait_seconds = round(
                max(0.0, first_image_from_created_seconds - first_image_seconds),
                3,
            )
        public["timing"] = {
            "queue_wait_seconds": queue_wait_seconds,
            "first_image_processing_seconds": first_image_seconds,
            "first_image_from_created_seconds": first_image_from_created_seconds,
            "first_guidance_from_created_seconds": _public_seconds(
                raw_streaming.get("first_guidance_from_created_seconds")
            ),
            "model_first_content_seconds": _public_seconds(
                raw_streaming.get("model_first_content_ms"), milliseconds=True
            ),
            "guidance_tiers": {
                kind: {
                    "first_published_from_created_seconds": _public_seconds(
                        entry.get("first_published_from_created_seconds")
                    ),
                }
                for kind in ("elite", "upper", "middle", "lower", "novice")
                if isinstance((entry := (raw_streaming.get("guidance_tiers") or {}).get(kind)), dict)
                and entry.get("published") is True
            } if isinstance(raw_streaming.get("guidance_tiers"), dict) else {},
            "first_image_refined_processing_seconds": first_image_refined_seconds,
            "provider_prefix_seconds": _public_seconds(
                raw_streaming.get("provider_prefix_ms"), milliseconds=True
            ),
            "processing_seconds": elapsed_seconds,
            "first_image_refinement_pending": (
                raw_streaming.get("first_image_refinement_pending") is True
            ),
        }
    raw_concurrency = payload.get("concurrency")
    if isinstance(raw_concurrency, dict):
        timing = public.setdefault(
            "timing",
            {
                "queue_wait_seconds": None,
                "processing_seconds": elapsed_seconds,
            },
        )
        timing.update(
            {
                "ocr_slot_wait_seconds": _public_seconds(
                    raw_concurrency.get("ocr_slot_wait_seconds")
                ),
                "ocr_processing_seconds": _public_seconds(
                    raw_concurrency.get("ocr_slot_processing_seconds")
                ),
                "model_slot_wait_seconds": _public_seconds(
                    raw_concurrency.get("model_slot_wait_seconds")
                ),
                "model_call_seconds": _public_seconds(
                    raw_concurrency.get("model_call_seconds")
                ),
            }
        )
    raw_scheduler = payload.get("scheduler")
    if isinstance(raw_scheduler, dict):
        public["scheduler"] = {
            key: value
            for key in (
                "queue_depth_observed",
                "worker_count",
                "ocr_concurrency",
                "model_concurrency",
            )
            if isinstance((value := raw_scheduler.get(key)), int)
            and not isinstance(value, bool)
            and 0 <= value <= 10_000
        }
    analysis_audit = sanitize_analysis_audit(payload.get("analysis_audit"))
    if analysis_audit is not None:
        public["analysis_audit"] = analysis_audit

    if status == "failed":
        public["error"] = {
            "message": "处理没有完成，请检查题图是否清晰、完整后重新拍照提交"
        }

    if status != "failed":
        upstream_images = payload.get("images") or []
        if not isinstance(upstream_images, list):
            if status != "completed":
                upstream_images = []
            else:
                raise UpstreamFailure(502, "六张结果图尚未完整生成，请重新提交")
        if len(upstream_images) > len(DEVICE_IMAGE_NAMES):
            raise UpstreamFailure(502, "结果图格式不符合六图要求，请重新提交")
        observed: list[str] = []
        for item in upstream_images:
            if not isinstance(item, dict):
                raise UpstreamFailure(502, "结果图的数据不完整，请重新提交")
            filename = str(item.get("filename") or "")
            if filename not in DEVICE_IMAGE_NAMES or filename in observed:
                raise UpstreamFailure(502, "结果图格式不符合六图要求，请重新提交")
            observed.append(filename)
        expected_prefix = list(DEVICE_IMAGE_NAMES[: len(observed)])
        if observed != expected_prefix:
            raise UpstreamFailure(502, "结果图必须按既定顺序逐张生成，请重新提交")
        if status == "completed" and len(observed) != len(DEVICE_IMAGE_NAMES):
            raise UpstreamFailure(502, "六张结果图尚未完整生成，请重新提交")

        safe_job_id = quote(job_id, safe="")
        upstream_by_name = {
            str(item.get("filename") or ""): item for item in upstream_images
        }
        public_images: list[dict[str, Any]] = []
        for index, (filename, kind) in enumerate(
            zip(expected_prefix, DEVICE_IMAGE_KINDS), start=1
        ):
            upstream = upstream_by_name[filename]
            metadata = _validated_result_metadata(upstream)
            current: dict[str, Any] = {
                "sequence": index,
                "kind": kind,
                "filename": filename,
                "media_type": "image/png",
                "url": (
                    f"/api/v1/{subject}/jobs/{safe_job_id}/images/"
                    f"{quote(filename, safe='')}"
                ),
                **metadata,
            }
            public_images.append(current)
        public["images"] = public_images
        public["result_contract"]["available_images"] = len(public_images)
    return public


def _fetch_png_sync(subject: str, job_id: str, filename: str) -> bytes:
    safe_job_id = quote(validate_job_id(job_id), safe="")
    safe_filename = quote(validate_result_filename(filename), safe="")
    url = backend_for_subject(subject) + f"/jobs/{safe_job_id}/{safe_filename}"
    request = UrlRequest(url, method="GET", headers={"Accept": "image/png"})
    try:
        with _UPSTREAM_OPENER.open(
            request, timeout=UPSTREAM_TIMEOUT_SECONDS
        ) as response:
            content_type = response.headers.get_content_type()
            data = _read_limited(
                response,
                MAX_RESULT_BYTES,
                "结果图过大，请重新提交题图",
            )
    except HTTPError as exc:
        try:
            exc.close()
        finally:
            raise UpstreamFailure(
                exc.code,
                _friendly_upstream_error(exc.code, subject),
            ) from None
    except (URLError, TimeoutError, OSError):
        raise UpstreamFailure(
            503,
            f"{SUBJECT_LABELS[subject]}结果图暂时无法读取，请稍后重试",
        ) from None

    if content_type != "image/png":
        raise UpstreamFailure(502, "处理服务返回的结果不是有效 PNG 图片")
    validate_device_png(data)
    return data


def validate_device_png(data: bytes) -> None:
    """Decode the trusted result fully and enforce the Test6 PNG geometry."""

    try:
        with Image.open(BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            mode = image.mode
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise UpstreamFailure(502, "处理服务返回的结果不是有效 PNG 图片") from None
    if image_format != "PNG" or width != 800 or height <= 0 or mode != "L":
        raise UpstreamFailure(502, "结果图不符合 Test6 的 PNG 显示规格")


def _subject_health_sync(subject: str) -> dict[str, Any]:
    try:
        payload = _request_json(subject, "/health")
    except UpstreamFailure:
        return {"available": False, "status": "unavailable"}
    healthy = (
        payload.get("status") == "ok"
        and payload.get("worker_alive") is True
        and payload.get("service") == EXPECTED_BACKEND_SERVICES[subject]
    )
    result = {
        "available": healthy,
        "status": "ok" if healthy else "unavailable",
    }
    workers = payload.get("workers")
    if isinstance(workers, dict):
        result["workers"] = {
            key: value
            for key in (
                "configured_workers",
                "alive_workers",
                "pending_jobs",
                "pending_devices",
                "active_jobs",
                "active_devices",
                "completed_jobs",
                "failed_jobs",
            )
            if isinstance((value := workers.get(key)), int)
            and not isinstance(value, bool)
            and 0 <= value <= 1_000_000
        }
    return result


def _upstream_http_exception(exc: UpstreamFailure) -> HTTPException:
    # Upstream 5xx responses become a gateway error, except a genuine service
    # outage, while expected client/queue errors keep their useful status.
    status_code = exc.status_code
    if status_code >= 500 and status_code != 503:
        status_code = 502
    headers = {"Retry-After": "2"} if status_code == 429 else None
    return HTTPException(status_code, exc.detail, headers=headers)


def _principal_id(kind: str, material: str) -> str:
    """Return a stable opaque identity; credentials are never stored in SQLite."""

    message = f"test6-auth-principal-v1:{kind}:{material}".encode("utf-8")
    return hashlib.sha256(message).hexdigest()


def _request_device_principal(principal_id: str, device_sn: str | None) -> str:
    """Separate scheduling/idempotency when a fleet shares one credential."""

    if not device_sn:
        return principal_id
    return _principal_id("device-sn", f"{principal_id}:{device_sn}")


def _authenticated_principal(request: Request) -> str | None:
    """Authenticate and identify a Bearer credential or browser session."""

    if ALLOW_INSECURE_NO_AUTH and not ACCESS_TOKEN:
        return _principal_id("insecure", "local")
    authorization = request.headers.get("authorization", "")
    if authorization:
        if not authorization.lower().startswith("bearer "):
            return None
        candidate = authorization[7:].strip()
        if not candidate or len(candidate) > 1024:
            return None
        if secrets.compare_digest(candidate, ACCESS_TOKEN):
            return _principal_id("primary", ACCESS_TOKEN)
        candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        for expected in DEVICE_TOKEN_HASHES:
            if secrets.compare_digest(candidate_hash, expected):
                return _principal_id("device", candidate_hash)
        return None
    session = request.cookies.get(COOKIE_NAME, "")
    if _valid_session(session):
        return _principal_id("primary", ACCESS_TOKEN)
    return None


def _authorized(request: Request) -> bool:
    return _authenticated_principal(request) is not None


def require_access(request: Request) -> str:
    principal_id = _authenticated_principal(request)
    if principal_id is None:
        raise HTTPException(
            401,
            "需要有效的设备访问令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal_id


app = FastAPI(
    title="Test6 流式物理数学统一搜题机",
    version=SERVICE_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(UploadBodyLimitMiddleware)

from prompt_admin import install as install_prompt_admin

install_prompt_admin(
    app,
    lambda request: (ALLOW_INSECURE_NO_AUTH and not ACCESS_TOKEN
        and request.client is not None
        and request.client.host in {"127.0.0.1", "::1"}
        and request.url.hostname in {"127.0.0.1", "localhost", "::1"})
    or (bool(ACCESS_TOKEN) and _authenticated_principal(request)
        == _principal_id("primary", ACCESS_TOKEN)),
)


def public_json(content: dict[str, Any], *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


def accepted_job(
    content: dict[str, Any],
    *,
    idempotency_replayed: bool | None = None,
) -> JSONResponse:
    response = public_json(content, status_code=202)
    response.headers["Location"] = str(content["poll_url"])
    response.headers["Retry-After"] = str(content["retry_after_seconds"])
    if idempotency_replayed is not None:
        response.headers["Idempotency-Replayed"] = (
            "true" if idempotency_replayed else "false"
        )
    return response


def _idempotency_unavailable() -> HTTPException:
    return HTTPException(
        503,
        "暂时无法安全确认该请求是否已经创建，请保留相同 Idempotency-Key 后重试",
        headers={"Retry-After": "2"},
    )


async def _create_job_from_bytes(
    subject: str,
    request: Request,
    principal_id: str,
    data: bytes,
    media_type: str,
) -> JSONResponse:
    try:
        curriculum_term = validate_term(request.headers.get("x-curriculum-term", ""))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    device_sn = request.headers.get("x-device-sn", "").strip()
    if device_sn and DEVICE_SN_PATTERN.fullmatch(device_sn) is None:
        raise HTTPException(
            400,
            "X-Device-SN 必须是 1-128 位 ASCII 字母、数字或 . _ : -",
        )
    normalized_device_sn = device_sn or None
    request_principal_id = _request_device_principal(
        principal_id, normalized_device_sn
    )
    key_sha256 = idempotency_key_digest(request)
    if key_sha256 is None:
        try:
            payload = await asyncio.to_thread(
                _submit_job_sync,
                subject,
                data,
                media_type,
                request_principal_id,
                normalized_device_sn,
                *((curriculum_term,) if curriculum_term else ()),
            )
            return accepted_job(sanitize_job(subject, payload))
        except UpstreamFailure as exc:
            raise _upstream_http_exception(exc) from exc

    image_sha256 = hashlib.sha256(data + (b"\x00semester:" + curriculum_term.encode("ascii") if curriculum_term else b"")).hexdigest()
    deadline = time.monotonic() + IDEMPOTENCY_WAIT_SECONDS
    while True:
        try:
            action, owner_nonce, replay = await asyncio.to_thread(
                _claim_idempotency,
                request_principal_id,
                subject,
                key_sha256,
                image_sha256,
            )
        except IdempotencyConflict as exc:
            raise HTTPException(
                409,
                "该 Idempotency-Key 已用于同一学科的另一张图片或另一年级学期，请生成新 key",
            ) from exc
        except IdempotencyStoreFailure as exc:
            raise _idempotency_unavailable() from exc

        if action == "ready":
            assert replay is not None
            return accepted_job(replay, idempotency_replayed=True)
        if action == "pending":
            if time.monotonic() >= deadline:
                raise _idempotency_unavailable()
            await asyncio.sleep(0.05)
            continue

        assert action == "owner" and owner_nonce is not None
        try:
            payload = await asyncio.to_thread(
                _submit_job_sync,
                subject,
                data,
                media_type,
                request_principal_id,
                normalized_device_sn,
                *((curriculum_term,) if curriculum_term else ()),
            )
            public = sanitize_job(subject, payload)
        except UpstreamFailure as exc:
            try:
                await asyncio.to_thread(
                    _release_idempotency,
                    request_principal_id,
                    subject,
                    key_sha256,
                    image_sha256,
                    owner_nonce,
                )
            except IdempotencyStoreFailure as store_exc:
                raise _idempotency_unavailable() from store_exc
            raise _upstream_http_exception(exc) from exc

        # A transient SQLite lock after upstream accepted the job must not make
        # the gateway immediately create a duplicate. Retry the durable commit.
        for attempt in range(3):
            try:
                await asyncio.to_thread(
                    _complete_idempotency,
                    request_principal_id,
                    subject,
                    key_sha256,
                    image_sha256,
                    owner_nonce,
                    public,
                )
                return accepted_job(public, idempotency_replayed=False)
            except IdempotencyStoreFailure as exc:
                if attempt == 2:
                    raise _idempotency_unavailable() from exc
                await asyncio.sleep(0.05 * (attempt + 1))


@app.post("/api/v1/auth/login", include_in_schema=False)
async def create_browser_session(request: Request) -> JSONResponse:
    if ALLOW_INSECURE_NO_AUTH and not ACCESS_TOKEN:
        return public_json({"authenticated": True, "mode": "insecure-local"})
    if (request.headers.get("content-type") or "").split(";", 1)[0].strip() != "application/json":
        raise HTTPException(415, "登录请求必须使用JSON")
    raw = await request.body()
    if len(raw) > 4096:
        raise HTTPException(413, "访问令牌过长")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(400, "登录请求格式不正确") from None
    candidate = str(payload.get("token") or "") if isinstance(payload, dict) else ""
    if not candidate or not secrets.compare_digest(candidate, ACCESS_TOKEN):
        raise HTTPException(401, "访问令牌不正确")
    response = public_json({"authenticated": True})
    response.set_cookie(
        COOKIE_NAME,
        _new_session_value(),
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="strict",
        path="/",
    )
    return response


@app.get("/api/v1/auth/session", include_in_schema=False)
def browser_session(request: Request) -> JSONResponse:
    return public_json({"authenticated": _authorized(request)})


@app.delete("/api/v1/auth/logout", include_in_schema=False)
def delete_browser_session() -> JSONResponse:
    response = public_json({"authenticated": False})
    response.delete_cookie(
        COOKIE_NAME,
        path="/",
        secure=COOKIE_SECURE,
        httponly=True,
        samesite="strict",
    )
    return response


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": (
                "default-src 'self'; img-src 'self' data: blob:; "
                "connect-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; object-src 'none'; "
                "base-uri 'none'; form-action 'self'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
        },
    )


@app.get("/ocr-audit", include_in_schema=False)
@app.get("/ocr-audit/latest", include_in_schema=False)
def ocr_audit(_principal_id: str = Depends(require_access)) -> FileResponse:
    if not LATEST_AUDIT_REPORT_HTML.is_file():
        raise HTTPException(404, "300题OCR审核报告尚未生成")
    return FileResponse(
        LATEST_AUDIT_REPORT_HTML,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": (
                "default-src 'self'; img-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
        },
    )


@app.get("/ocr-audit/iteration", include_in_schema=False)
def iteration_ocr_audit(
    _principal_id: str = Depends(require_access),
) -> Response:
    """Serve the current 300-item review over HTTP so annotations are durable."""

    if not ITERATION_AUDIT_REPORT_HTML.is_file():
        raise HTTPException(404, "本轮300题OCR审核报告尚未生成")
    report = ITERATION_AUDIT_REPORT_HTML.read_text(encoding="utf-8")
    dataset_prefix = ITERATION_AUDIT_DATASET_DIR.resolve().as_uri().rstrip("/") + "/"
    generated_prefix = (
        ITERATION_AUDIT_GENERATED_DIR.resolve().as_uri().rstrip("/") + "/"
    )
    report = report.replace(
        dataset_prefix,
        "/ocr-audit/iteration/source/",
    ).replace(
        generated_prefix,
        "/ocr-audit/iteration/generated/",
    )
    audit_batch = ITERATION_AUDIT_RUN_DIR.name
    batch_banner = (
        '<div style="margin:0 0 10px;padding:9px 12px;border:1px solid #8bb7e8;'
        'border-radius:8px;background:#eef6ff;color:#123b67;font-weight:600">'
        f'当前算法结果：{audit_batch}；人工批注源：76 条已保存批注。'
        '本页由当前服务动态加载，请勿用旧的 file:/// 静态快照判断最新结果。'
        '</div>'
    )
    report = report.replace(
        '<section class="top">',
        '<section class="top">' + batch_banner,
        1,
    )
    return Response(
        content=report,
        media_type="text/html; charset=utf-8",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": (
                "default-src 'self'; img-src 'self'; connect-src 'self'; "
                "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                "object-src 'none'; base-uri 'none'; form-action 'self'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
        },
    )


@app.get(
    "/ocr-audit/iteration/source/{subject}/{group}/{filename}",
    include_in_schema=False,
)
def iteration_ocr_audit_source(
    subject: str,
    group: str,
    filename: str,
    _principal_id: str = Depends(require_access),
) -> FileResponse:
    if subject not in {"mathematics", "physics"}:
        raise HTTPException(404, "审核原图不存在")
    if re.fullmatch(r"Test(?:10|[1-9])", group) is None:
        raise HTTPException(404, "审核原图不存在")
    if re.fullmatch(r"(?:10|[1-9])-(?:1[0-5]|[1-9])\.(?:png|jpe?g|webp)", filename, re.I) is None:
        raise HTTPException(404, "审核原图不存在")
    path = ITERATION_AUDIT_DATASET_DIR / subject / group / filename
    if not path.is_file():
        raise HTTPException(404, "审核原图不存在")
    return FileResponse(
        path,
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )


@app.get(
    "/ocr-audit/iteration/generated/{kind}/{filename}",
    include_in_schema=False,
)
def iteration_ocr_audit_generated(
    kind: str,
    filename: str,
    _principal_id: str = Depends(require_access),
) -> FileResponse:
    if kind not in {"ocr_rendered", "selected_question_evidence", "illustrations"}:
        raise HTTPException(404, "审核结果图不存在")
    if re.fullmatch(
        r"TZIP-(?:MATH|PHYS)-T\d{2}-Q\d{2}(?:_figure_\d{2})?\.png",
        filename,
    ) is None:
        raise HTTPException(404, "审核结果图不存在")
    path = ITERATION_AUDIT_GENERATED_DIR / kind / filename
    if not path.is_file():
        raise HTTPException(404, "审核结果图不存在")
    return FileResponse(
        path,
        media_type="image/png",
        # The stable audit URL can point at a newly generated iteration whose
        # filenames intentionally remain the same.  Never let the browser show
        # an earlier crop-heavy bitmap after the server switches iterations.
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


def _validated_iteration_reviews(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or len(value) > 300:
        raise HTTPException(400, "标注数据格式不正确")
    normalized: dict[str, dict[str, str]] = {}
    for question_id, item in value.items():
        if not isinstance(question_id, str) or re.fullmatch(
            r"TZIP-(?:MATH|PHYS)-T\d{2}-Q\d{2}", question_id
        ) is None:
            raise HTTPException(400, "标注题号格式不正确")
        if not isinstance(item, dict):
            raise HTTPException(400, "标注内容格式不正确")
        decision = item.get("decision", "")
        if decision not in {"pass", "fail", "uncertain", ""}:
            raise HTTPException(400, "人工结论格式不正确")
        raw_types = item.get("manual_error_types", "")
        if not isinstance(raw_types, str):
            raise HTTPException(400, "问题类型格式不正确")
        error_types = [entry for entry in raw_types.split(";") if entry]
        if len(error_types) > len(ITERATION_AUDIT_ERROR_TYPES) or any(
            entry not in ITERATION_AUDIT_ERROR_TYPES for entry in error_types
        ):
            raise HTTPException(400, "问题类型包含未知值")
        notes = item.get("manual_notes", "")
        if not isinstance(notes, str) or len(notes) > 2000:
            raise HTTPException(400, "备注不能超过2000字")
        notes = notes.replace("\x00", "").strip()
        if not decision and not error_types and not notes:
            continue
        normalized[question_id] = {
            "question_id": question_id,
            "subject": "mathematics" if "-MATH-" in question_id else "physics",
            "source_group": str(item.get("source_group", ""))[:16],
            "source_index": str(item.get("source_index", ""))[:8],
            "image_path": str(item.get("image_path", ""))[:300],
            "weighted_character_confidence": str(
                item.get("weighted_character_confidence", "")
            )[:32],
            "manual_exact_normalized": "",
            "manual_semantic_complete": (
                "true" if decision == "pass" else "false" if decision == "fail" else ""
            ),
            "manual_error_count": str(len(error_types)),
            "manual_error_types": ";".join(dict.fromkeys(error_types)),
            "manual_notes": notes,
            "decision": decision,
        }
    return normalized


def _write_iteration_reviews(reviews: dict[str, dict[str, str]]) -> None:
    ITERATION_AUDIT_ANNOTATION_STORE_DIR.mkdir(parents=True, exist_ok=True)
    document = {
        "version": 1,
        "updated_at_epoch": int(time.time()),
        "reviews": reviews,
    }
    json_temp = ITERATION_AUDIT_ANNOTATIONS_JSON.with_suffix(".json.tmp")
    json_temp.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(json_temp, ITERATION_AUDIT_ANNOTATIONS_JSON)

    fields = [
        "question_id",
        "subject",
        "source_group",
        "source_index",
        "image_path",
        "weighted_character_confidence",
        "manual_exact_normalized",
        "manual_semantic_complete",
        "manual_error_count",
        "manual_error_types",
        "manual_notes",
        "decision",
    ]
    csv_temp = ITERATION_AUDIT_ANNOTATIONS_CSV.with_suffix(".csv.tmp")
    with csv_temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reviews[key] for key in sorted(reviews))
    os.replace(csv_temp, ITERATION_AUDIT_ANNOTATIONS_CSV)


@app.get("/api/v1/ocr-audit/iteration/annotations", include_in_schema=False)
def get_iteration_ocr_audit_annotations(
    _principal_id: str = Depends(require_access),
) -> JSONResponse:
    reviews: dict[str, Any] = {}
    if ITERATION_AUDIT_ANNOTATIONS_JSON.is_file():
        try:
            stored = json.loads(
                ITERATION_AUDIT_ANNOTATIONS_JSON.read_text(encoding="utf-8")
            )
            if isinstance(stored, dict) and isinstance(stored.get("reviews"), dict):
                reviews = stored["reviews"]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise HTTPException(500, "已保存的标注文件无法读取") from None
    return public_json({"reviews": reviews, "saved_count": len(reviews)})


@app.post("/api/v1/ocr-audit/iteration/annotations", include_in_schema=False)
async def save_iteration_ocr_audit_annotations(
    request: Request,
    _principal_id: str = Depends(require_access),
) -> JSONResponse:
    raw = await request.body()
    if len(raw) > 2 * 1024 * 1024:
        raise HTTPException(413, "标注数据过大")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise HTTPException(400, "标注数据不是有效JSON") from None
    if not isinstance(payload, dict):
        raise HTTPException(400, "标注数据格式不正确")
    reviews = _validated_iteration_reviews(payload.get("reviews"))
    try:
        _write_iteration_reviews(reviews)
    except OSError:
        raise HTTPException(500, "标注文件写入失败") from None
    return public_json(
        {
            "saved": True,
            "saved_count": len(reviews),
            "saved_csv": ITERATION_AUDIT_ANNOTATIONS_CSV.name,
        }
    )


@app.get("/ocr-audit/baseline", include_in_schema=False)
def ocr_audit_baseline(_principal_id: str = Depends(require_access)) -> FileResponse:
    if not AUDIT_REPORT_HTML.is_file():
        raise HTTPException(404, "旧版300题OCR审核报告尚未生成")
    return FileResponse(
        AUDIT_REPORT_HTML,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": (
                "default-src 'self'; img-src 'self'; style-src 'self' 'unsafe-inline'; "
                "script-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
        },
    )


@app.get("/ocr-audit/pages/{filename}", include_in_schema=False)
def ocr_audit_page(
    filename: str,
    _principal_id: str = Depends(require_access),
) -> FileResponse:
    if not re.fullmatch(r"TZIP-(?:MATH|PHYS)-T\d{2}-Q\d{2}\.png", filename):
        raise HTTPException(404, "审核图片不存在")
    path = AUDIT_PAGES_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "审核图片不存在")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/ocr-audit/latest/pages/{filename}", include_in_schema=False)
def latest_ocr_audit_page(
    filename: str,
    _principal_id: str = Depends(require_access),
) -> FileResponse:
    if not re.fullmatch(r"TZIP-(?:MATH|PHYS)-T\d{2}-Q\d{2}\.png", filename):
        raise HTTPException(404, "审核图片不存在")
    path = LATEST_AUDIT_PAGES_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "审核图片不存在")
    return FileResponse(
        path,
        media_type="image/png",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/ocr-audit/latest/data.csv", include_in_schema=False)
def latest_ocr_audit_csv(_principal_id: str = Depends(require_access)) -> FileResponse:
    if not LATEST_AUDIT_DATA_CSV.is_file():
        raise HTTPException(404, "新版审核数据不存在")
    return FileResponse(
        LATEST_AUDIT_DATA_CSV,
        media_type="text/csv; charset=utf-8",
        filename="OCR修正后语义完整性复核_300题.csv",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/ocr-audit/latest/report.md", include_in_schema=False)
def latest_ocr_audit_markdown(
    _principal_id: str = Depends(require_access),
) -> FileResponse:
    if not LATEST_AUDIT_REPORT_MD.is_file():
        raise HTTPException(404, "新版审核文字报告不存在")
    return FileResponse(
        LATEST_AUDIT_REPORT_MD,
        media_type="text/markdown; charset=utf-8",
        filename="OCR修正后300题复测报告.md",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/ocr-audit/data.csv", include_in_schema=False)
def ocr_audit_csv(_principal_id: str = Depends(require_access)) -> FileResponse:
    if not AUDIT_DATA_CSV.is_file():
        raise HTTPException(404, "审核数据不存在")
    return FileResponse(
        AUDIT_DATA_CSV,
        media_type="text/csv; charset=utf-8",
        filename="OCR语义完整性审核_300题.csv",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/ocr-audit/report.md", include_in_schema=False)
def ocr_audit_markdown(_principal_id: str = Depends(require_access)) -> FileResponse:
    if not AUDIT_REPORT_MD.is_file():
        raise HTTPException(404, "审核文字报告不存在")
    return FileResponse(
        AUDIT_REPORT_MD,
        media_type="text/markdown; charset=utf-8",
        filename="OCR语义完整性审核_300题.md",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/ocr-audit/evidence.md", include_in_schema=False)
def ocr_audit_evidence_markdown(
    _principal_id: str = Depends(require_access),
) -> FileResponse:
    if not AUDIT_EVIDENCE_MD.is_file():
        raise HTTPException(404, "语义证据兜底报告不存在")
    return FileResponse(
        AUDIT_EVIDENCE_MD,
        media_type="text/markdown; charset=utf-8",
        filename="语义证据兜底覆盖核验_300题.md",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/api/v1/capabilities")
def capabilities(_principal_id: str = Depends(require_access)) -> JSONResponse:
    return public_json(
        {
            "api_version": API_VERSION,
            "service_version": SERVICE_VERSION,
            "subjects": [
                {"id": subject, "label": SUBJECT_LABELS[subject]}
                for subject in ("physics", "math")
            ],
            "upload": {
                "multipart_field": "image",
                "raw_endpoint_suffix": "/raw",
                "media_types": ["image/jpeg", "image/png", "image/webp"],
                "max_bytes": MAX_UPLOAD_BYTES,
                "device_sn": {
                    "header": "X-Device-SN",
                    "optional": True,
                    "max_length": 128,
                },
                "idempotency": {
                    "header": "Idempotency-Key",
                    "optional": True,
                    "min_length": 8,
                    "max_length": 128,
                    "retention_seconds": IDEMPOTENCY_RETENTION_SECONDS,
                },
            },
            "workflow": "asynchronous",
            "scheduling": {
                "policy": "fair-per-device",
                "identity": "one unique bearer token per hardware device",
            },
            "poll_after_ms": 1200,
            "results": [
                {"sequence": index, "kind": kind, "filename": filename}
                for index, (kind, filename) in enumerate(
                    zip(DEVICE_IMAGE_KINDS, DEVICE_IMAGE_NAMES), start=1
                )
            ],
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    subjects = tuple(BACKENDS)
    checks = await asyncio.gather(
        *(asyncio.to_thread(_subject_health_sync, subject) for subject in subjects)
    )
    subject_state = dict(zip(subjects, checks))
    available_count = sum(bool(item["available"]) for item in checks)
    if available_count == len(subjects):
        status, status_code = "ok", 200
    elif available_count:
        # Keep the portal available when one subject can still serve users.
        status, status_code = "degraded", 200
    else:
        status, status_code = "unavailable", 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": status,
            "service": "test6-unified-search-portal",
            "version": SERVICE_VERSION,
            "subjects": subject_state,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/v1/curriculum-terms")
async def curriculum_terms(principal_id: str = Depends(require_access)) -> dict:
    terms = semester_catalog()
    for item in terms:
        for subject in ("math", "physics"):
            item["subjects"][subject]["prompt"] = semester_snapshot(subject, item["id"])["prompt"]
    return {"terms": terms}


@app.post("/api/v1/{subject}/jobs", status_code=202)
async def create_job(
    subject: str,
    request: Request,
    image: UploadFile = File(...),
    principal_id: str = Depends(require_access),
) -> JSONResponse:
    backend_for_subject(subject)
    try:
        data = await image.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await image.close()
    media_type = validate_upload(data, image.content_type)
    return await _create_job_from_bytes(
        subject,
        request,
        principal_id,
        data,
        media_type,
    )


@app.post("/api/v1/{subject}/jobs/raw", status_code=202)
async def create_raw_job(
    subject: str,
    request: Request,
    principal_id: str = Depends(require_access),
) -> JSONResponse:
    """Hardware-friendly upload accepting the encoded image as the raw body."""

    backend_for_subject(subject)
    content_type = request.headers.get("content-type")
    data = await request.body()
    media_type = validate_upload(data, content_type)
    return await _create_job_from_bytes(
        subject,
        request,
        principal_id,
        data,
        media_type,
    )


@app.get("/api/v1/{subject}/jobs/{job_id}")
async def get_job(
    subject: str,
    job_id: str,
    _principal_id: str = Depends(require_access),
) -> JSONResponse:
    backend_for_subject(subject)
    validate_job_id(job_id)
    try:
        payload = await asyncio.to_thread(_fetch_job_sync, subject, job_id)
        content = sanitize_job(subject, payload, expected_job_id=job_id)
        response = public_json(content)
        if content["status"] not in {"completed", "failed"}:
            response.headers["Retry-After"] = str(content["retry_after_seconds"])
        return response
    except UpstreamFailure as exc:
        raise _upstream_http_exception(exc) from exc


@app.get(
    "/api/v1/{subject}/jobs/{job_id}/images/{filename}",
    include_in_schema=False,
)
async def get_result_image(
    subject: str,
    job_id: str,
    filename: str,
    _principal_id: str = Depends(require_access),
) -> Response:
    backend_for_subject(subject)
    validate_job_id(job_id)
    validate_result_filename(filename)
    try:
        data = await asyncio.to_thread(_fetch_png_sync, subject, job_id, filename)
    except UpstreamFailure as exc:
        raise _upstream_http_exception(exc) from exc
    return Response(
        content=data,
        media_type="image/png",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'inline; filename="{filename}"',
        },
    )
