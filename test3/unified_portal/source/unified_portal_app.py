#!/usr/bin/env python3
"""Same-origin portal for the existing physics and mathematics test3 services."""

from __future__ import annotations

import asyncio
from io import BytesIO
import json
from pathlib import Path
import re
import secrets
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError


SOURCE_DIR = Path(__file__).resolve().parent
STATIC_DIR = SOURCE_DIR / "static"

# These are deliberately fixed loopback-only destinations.  No user input is
# ever accepted as a host or port, so this proxy cannot be used as a generic
# network relay.
BACKENDS = {
    "physics": "http://127.0.0.1:8010",
    "math": "http://127.0.0.1:8020",
}
SUBJECT_LABELS = {"physics": "物理", "math": "数学"}
EXPECTED_BACKEND_SERVICES = {
    "physics": "test3-search-machine",
    "math": "test3-math-search-machine",
}

DEVICE_IMAGE_NAMES = (
    "01_question_knowledge.png",
    "02_guidance_advanced.png",
    "03_guidance_standard.png",
    "04_guidance_foundation.png",
)
DEVICE_IMAGE_KINDS = (
    "question_knowledge",
    "advanced",
    "standard",
    "foundation",
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 20 * 1024 * 1024
MAX_MULTIPART_BYTES = MAX_UPLOAD_BYTES + 1024 * 1024
UPSTREAM_TIMEOUT_SECONDS = 20
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
IMAGE_MEDIA_TYPE_PATTERN = re.compile(r"^image/[a-z0-9][a-z0-9.+-]{0,63}$")
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


class UploadBodyLimitMiddleware:
    """Enforce the upload limit while ASGI request chunks are arriving."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = str(scope.get("path") or "")
        limited = (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and re.fullmatch(r"/api/(?:physics|math)/jobs", path) is not None
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
    if not IMAGE_MEDIA_TYPE_PATTERN.fullmatch(media_type):
        raise HTTPException(415, "仅支持图片文件")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "图片不能超过 20 MB")
    if not data:
        raise HTTPException(400, "请选择一张题目图片")
    return media_type


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
    request = Request(url, data=data, headers=headers or {}, method=method)
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
    boundary = "----test3-" + secrets.token_hex(18)
    extension = "png" if content_type.lower() == "image/png" else "jpg"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="camera.{extension}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("ascii")
    suffix = f"\r\n--{boundary}--\r\n".encode("ascii")
    return prefix + data + suffix, boundary


def _submit_job_sync(subject: str, data: bytes, content_type: str) -> dict[str, Any]:
    body, boundary = _multipart_upload(data, content_type)
    return _request_json(
        subject,
        "/api/jobs",
        method="POST",
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )


def _fetch_job_sync(subject: str, job_id: str) -> dict[str, Any]:
    safe_job_id = quote(validate_job_id(job_id), safe="")
    return _request_json(subject, f"/api/jobs/{safe_job_id}")


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

    public: dict[str, Any] = {
        "subject": subject,
        "subject_label": SUBJECT_LABELS[subject],
        "job_id": job_id,
        "status": status,
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "error": None,
        "images": [],
    }

    if status == "failed":
        public["error"] = {
            "message": "处理没有完成，请检查题图是否清晰、完整后重新拍照提交"
        }

    if status == "completed":
        upstream_images = payload.get("images")
        if not isinstance(upstream_images, list):
            raise UpstreamFailure(502, "四张结果图尚未完整生成，请重新提交")
        observed: list[str] = []
        for item in upstream_images:
            if not isinstance(item, dict):
                raise UpstreamFailure(502, "四张结果图的数据不完整，请重新提交")
            filename = str(item.get("filename") or "")
            if filename not in DEVICE_IMAGE_NAMES or filename in observed:
                raise UpstreamFailure(502, "结果图格式不符合四图要求，请重新提交")
            observed.append(filename)
        if set(observed) != set(DEVICE_IMAGE_NAMES) or len(observed) != 4:
            raise UpstreamFailure(502, "四张结果图尚未完整生成，请重新提交")

        safe_job_id = quote(job_id, safe="")
        public["images"] = [
            {
                "sequence": index,
                "kind": kind,
                "filename": filename,
                "url": (
                    f"/api/{subject}/jobs/{safe_job_id}/images/"
                    f"{quote(filename, safe='')}"
                ),
            }
            for index, (filename, kind) in enumerate(
                zip(DEVICE_IMAGE_NAMES, DEVICE_IMAGE_KINDS), start=1
            )
        ]
    return public


def _fetch_png_sync(subject: str, job_id: str, filename: str) -> bytes:
    safe_job_id = quote(validate_job_id(job_id), safe="")
    safe_filename = quote(validate_result_filename(filename), safe="")
    url = backend_for_subject(subject) + f"/jobs/{safe_job_id}/{safe_filename}"
    request = Request(url, method="GET", headers={"Accept": "image/png"})
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
    """Decode the trusted result fully and enforce the shared test3 geometry."""

    try:
        with Image.open(BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            mode = image.mode
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise UpstreamFailure(502, "处理服务返回的结果不是有效 PNG 图片") from None
    if image_format != "PNG" or width != 800 or height <= 0 or mode != "L":
        raise UpstreamFailure(502, "结果图不符合 test3 的 PNG 显示规格")


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
    return {
        "available": healthy,
        "status": "ok" if healthy else "unavailable",
    }


def _upstream_http_exception(exc: UpstreamFailure) -> HTTPException:
    # Upstream 5xx responses become a gateway error, except a genuine service
    # outage, while expected client/queue errors keep their useful status.
    status_code = exc.status_code
    if status_code >= 500 and status_code != 503:
        status_code = 502
    return HTTPException(status_code, exc.detail)


app = FastAPI(
    title="test3 物理数学统一搜题机",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)
app.add_middleware(UploadBodyLimitMiddleware)


def public_json(content: dict[str, Any], *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


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
        },
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
            "service": "test3-unified-search-portal",
            "subjects": subject_state,
        },
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/{subject}/jobs", status_code=202)
async def create_job(subject: str, image: UploadFile = File(...)) -> JSONResponse:
    backend_for_subject(subject)
    try:
        data = await image.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await image.close()
    media_type = validate_upload(data, image.content_type)
    try:
        payload = await asyncio.to_thread(
            _submit_job_sync,
            subject,
            data,
            media_type,
        )
        return public_json(sanitize_job(subject, payload), status_code=202)
    except UpstreamFailure as exc:
        raise _upstream_http_exception(exc) from exc


@app.get("/api/{subject}/jobs/{job_id}")
async def get_job(subject: str, job_id: str) -> JSONResponse:
    backend_for_subject(subject)
    validate_job_id(job_id)
    try:
        payload = await asyncio.to_thread(_fetch_job_sync, subject, job_id)
        return public_json(
            sanitize_job(subject, payload, expected_job_id=job_id)
        )
    except UpstreamFailure as exc:
        raise _upstream_http_exception(exc) from exc


@app.get(
    "/api/{subject}/jobs/{job_id}/images/{filename}",
    include_in_schema=False,
)
async def get_result_image(subject: str, job_id: str, filename: str) -> Response:
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
