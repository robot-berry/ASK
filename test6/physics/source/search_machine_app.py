#!/usr/bin/env python3
"""FastAPI entry point for the Test6 streaming physics search machine."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import re
import sys
import threading

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from presentation_guard import public_display_text, source_markup_paths
from multi_device_scheduler import ConcurrentJobWorker

from device_result_renderer import DEVICE_IMAGE_NAMES
from search_machine_pipeline import MAX_UPLOAD_BYTES, RUNS_ROOT, SearchMachinePipeline


SOURCE_DIR = Path(__file__).resolve().parent
STATIC_DIR = SOURCE_DIR / "search_machine_static"


def _optional_positive_int(name: str) -> int | None:
    """Return an explicit positive limit, or None to let the model finish."""

    raw = os.environ.get(name, "").strip()
    if not raw or raw.lower() in {"0", "none", "unlimited"}:
        return None
    value = int(raw)
    return value if value > 0 else None


LOGGER = logging.getLogger(__name__)


def _integer_setting(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)).strip())
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


QUEUE_CAPACITY = _integer_setting(
    "TEST6_PHYSICS_QUEUE_CAPACITY", 200, minimum=0, maximum=10_000
)
JOB_WORKERS = _integer_setting(
    "TEST6_PHYSICS_JOB_WORKERS", 1, minimum=1, maximum=32
)
OCR_CONCURRENCY = _integer_setting(
    "TEST6_PHYSICS_OCR_CONCURRENCY", JOB_WORKERS, minimum=1, maximum=JOB_WORKERS
)
MODEL_CONCURRENCY = _integer_setting(
    "TEST6_PHYSICS_MODEL_CONCURRENCY", JOB_WORKERS, minimum=1, maximum=JOB_WORKERS
)
PER_DEVICE_PENDING = _integer_setting(
    "TEST6_PHYSICS_PER_DEVICE_PENDING", 4, minimum=1, maximum=50
)
PER_DEVICE_ACTIVE = _integer_setting(
    "TEST6_PHYSICS_PER_DEVICE_ACTIVE", 1, minimum=1, maximum=4
)
MODEL_ID = (
    os.environ.get("TEST6_PHYSICS_MODEL_ID", "").strip()
    or os.environ.get("TEST4_PHYSICS_MODEL_ID", "").strip()
    or os.environ.get("TEST3_MODEL_ID", "qwen3.7-flash-2026-07-15").strip()
    or "qwen3.7-flash-2026-07-15"
)
COMPLEX_MODEL_ID = (
    os.environ.get("TEST6_PHYSICS_COMPLEX_MODEL_ID", "qwen3.8-max").strip()
    or "qwen3.8-max"
)
MODEL_TIMEOUT_SECONDS = float(
    os.environ.get("TEST6_PHYSICS_MODEL_BUDGET_SECONDS", "40")
)
MAX_OUTPUT_TOKENS = _optional_positive_int("TEST6_PHYSICS_MAX_OUTPUT_TOKENS")


def _non_negative_float_setting(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if raw.lower() in {"none", "unlimited"}:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.0, value)


POST_LOWER_BUDGET_SECONDS = _non_negative_float_setting(
    "TEST6_PHYSICS_POST_LOWER_BUDGET_SECONDS", 15.0
)
OCR_SEMAPHORE = threading.BoundedSemaphore(OCR_CONCURRENCY)
MODEL_SEMAPHORE = threading.BoundedSemaphore(MODEL_CONCURRENCY)


def _new_pipeline() -> SearchMachinePipeline:
    return SearchMachinePipeline(
        model_id=MODEL_ID,
        complex_model_id=COMPLEX_MODEL_ID,
        timeout_seconds=MODEL_TIMEOUT_SECONDS,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_provider_repair_rounds=0,
        ocr_semaphore=OCR_SEMAPHORE,
        model_semaphore=MODEL_SEMAPHORE,
        post_lower_budget_seconds=POST_LOWER_BUDGET_SECONDS,
    )


worker_pipelines = [_new_pipeline() for _ in range(JOB_WORKERS)]
pipeline = worker_pipelines[0]


def _worker_error(job_id: str, error: Exception) -> None:
    LOGGER.exception(
        "unhandled job worker failure: %s", job_id, exc_info=error
    )
    try:
        state = pipeline.read_job(job_id)
        job_dir = pipeline.job_path(job_id)
        state["error"] = {
            "type": "WorkerUnhandledError",
            "message": "服务器处理任务时出现内部错误，请重新提交",
        }
        pipeline._write_state(job_dir, state, "failed")
    except Exception:
        LOGGER.exception("could not persist failed worker job: %s", job_id)


worker = ConcurrentJobWorker(
    [current.solve for current in worker_pipelines],
    capacity=QUEUE_CAPACITY,
    per_device_pending=PER_DEVICE_PENDING,
    per_device_active=PER_DEVICE_ACTIVE,
    thread_name_prefix="test6-physics-job-worker",
    on_error=_worker_error,
)


def recover_interrupted_jobs() -> None:
    """Fail interrupted work and fairly requeue jobs not yet started."""

    if not RUNS_ROOT.is_dir():
        return
    active = {"preprocessing", "solving", "rendering"}
    for job_dir in sorted(RUNS_ROOT.iterdir()):
        if not job_dir.is_dir() or job_dir.name.startswith("_"):
            continue
        try:
            state = pipeline.read_job(job_dir.name)
        except (ValueError, FileNotFoundError, OSError):
            continue
        status = state.get("status")
        if status in active:
            state["error"] = {
                "type": "ServiceRestarted",
                "message": "服务器处理中断，请重新拍照提交",
            }
            pipeline._write_state(job_dir, state, "failed")
        elif status == "queued":
            request_state = state.get("request")
            device_id = (
                str(request_state.get("device_id") or "direct")
                if isinstance(request_state, dict)
                else "direct"
            )
            submitted = worker.submit(job_dir.name, device_id)
            if not submitted.accepted:
                state["error"] = {
                    "type": "QueueFullAfterRestart",
                    "message": "服务器重启后队列已满，请重新提交",
                }
                pipeline._write_state(job_dir, state, "failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    worker.start()
    recover_interrupted_jobs()
    try:
        yield
    finally:
        worker.stop()


app = FastAPI(title="Test6 流式物理搜题机", version="6.3.6", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _analysis_text(value: object, limit: int = 320) -> str:
    return public_display_text(value, limit=limit)


def _public_audit_parts(raw_parts: object) -> list[dict]:
    parts: list[dict] = []
    for raw_part in (raw_parts if isinstance(raw_parts, list) else [])[:16]:
        if not isinstance(raw_part, dict):
            continue
        raw_methods = raw_part.get("methods")
        if not isinstance(raw_methods, list):
            raw_methods = [raw_part]
        methods: list[dict] = []
        for raw_method in raw_methods[:3]:
            if not isinstance(raw_method, dict):
                continue
            steps = [
                _analysis_text(step)
                for step in (raw_method.get("steps") or [])[:8]
                if _analysis_text(step)
            ]
            if not steps:
                continue
            methods.append({
                "method_id": _analysis_text(raw_method.get("method_id"), 24),
                "method_name": _analysis_text(raw_method.get("method_name"), 80),
                "steps": steps,
                "pitfall": _analysis_text(raw_method.get("pitfall"), 180),
            })
        if methods:
            parts.append({
                "part": _analysis_text(raw_part.get("part"), 40) or "整题",
                "methods": methods,
            })
    return parts


def public_analysis_audit(state: dict) -> dict | None:
    """Expose only answer-blind, validated analysis prose for browser inspection."""

    output = state.get("output_json")
    if not isinstance(output, dict):
        return None
    guidance = output.get("guidance")
    if not isinstance(guidance, dict):
        return None
    tier = next(
        (
            guidance.get(key)
            for key in ("standard", "foundation", "advanced")
            if isinstance(guidance.get(key), dict)
        ),
        None,
    )
    parts = _public_audit_parts(tier.get("parts") if isinstance(tier, dict) else [])
    provider = state.get("provider_contract")
    provider = provider if isinstance(provider, dict) else {}
    if provider.get("time_budget_fallback") is True:
        source = "local_fallback"
    elif provider.get("partial_skeleton_reused") is True:
        source = "partial_model"
    elif state.get("model_id_requested") or state.get("model_id_response"):
        source = "complete_model"
    else:
        source = "validated_output"
    formulas = []
    for item in (output.get("base_formulas") or [])[:4]:
        if isinstance(item, dict):
            formula = _analysis_text(item.get("formula"), 180)
            condition = _analysis_text(item.get("condition"), 180)
            if formula:
                formulas.append({"formula": formula, "condition": condition})
    result = {
        "source": source,
        "model_id": _analysis_text(
            state.get("model_id_response") or state.get("model_id_requested"), 80
        ),
        "knowledge_points": [
            _analysis_text(item, 100)
            for item in (output.get("knowledge_points") or [])[:6]
            if _analysis_text(item, 100)
        ],
        "base_formulas": formulas,
        "parts": parts,
    }
    display = output.get("display_guidance")
    if isinstance(display, dict):
        tier_titles = {
            "elite": "学霸解析",
            "upper": "中上解析",
            "middle": "中等解析",
            "lower": "中下解析",
            "novice": "学渣解析",
        }
        tiers = []
        for key in ("elite", "upper", "middle", "lower", "novice"):
            page = display.get(key)
            if not isinstance(page, dict):
                continue
            tier_formulas = []
            for item in (page.get("base_formulas") or [])[:4]:
                if isinstance(item, dict):
                    formula = _analysis_text(item.get("formula"), 180)
                    condition = _analysis_text(item.get("condition"), 180)
                    if formula:
                        tier_formulas.append({"formula": formula, "condition": condition})
            tiers.append({
                "tier_key": key,
                "title": tier_titles[key],
                "knowledge_points": [
                    _analysis_text(item, 100)
                    for item in (page.get("knowledge_points") or [])[:6]
                    if _analysis_text(item, 100)
                ],
                "base_formulas": tier_formulas,
                "parts": _public_audit_parts(page.get("parts")),
                "pattern_summary": _analysis_text(page.get("pattern_summary"), 220),
            })
        if len(tiers) == 5:
            result["tiers"] = tiers
    if not result["parts"] and not result.get("tiers"):
        return None
    if source_markup_paths(result):
        raise RuntimeError("public analysis output contains internal formula markup")
    return result


def public_job(state: dict) -> dict:
    job_id = str(state["job_id"])
    raw_contract = state.get("image_contract")
    public_contract = None
    if isinstance(raw_contract, dict):
        # The persisted contract contains absolute local paths for server-side
        # auditing.  The phone only needs the contract summary; individual
        # public image metadata is returned below with URL paths.
        public_contract = {
            key: raw_contract.get(key)
            for key in (
                "valid",
                "expected",
                "observed",
                "image_count",
                "errors",
                "contract",
            )
        }
    public_error = None
    raw_error = state.get("error")
    if isinstance(raw_error, dict):
        error_type = str(raw_error.get("type") or "ProcessingError")
        friendly_errors = {
            "QueueFull": "当前任务较多，请稍后重试",
            "DeviceQueueFull": "该设备已有较多任务等待，请等待前序任务完成",
            "QueueFullAfterRestart": "服务器重启后队列已满，请重新提交",
            "ServiceRestarted": "服务器处理中断，请重新拍照提交",
        }
        public_error = {
            "type": error_type,
            "message": friendly_errors.get(
                error_type,
                "处理未完成，请核对题图后重新提交；详细原因已保存在服务器日志中",
            ),
        }
    result = {
        "job_id": job_id,
        "status": state.get("status"),
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at"),
        "elapsed_seconds": state.get("elapsed_seconds"),
        "error": public_error,
        "image_contract": public_contract,
    }
    images = []
    for item in state.get("images") or []:
        current = dict(item)
        # Never expose local absolute paths or private audit metadata.
        current.pop("path", None)
        filename = Path(str(item.get("filename") or item.get("path") or "")).name
        if filename:
            current["filename"] = filename
            current["url"] = f"/jobs/{job_id}/{filename}"
        images.append(current)
    result["images"] = images
    analysis_audit = public_analysis_audit(state)
    if analysis_audit is not None:
        result["analysis_audit"] = analysis_audit
    raw_streaming = state.get("streaming")
    if isinstance(raw_streaming, dict):
        result["streaming"] = {
            key: raw_streaming.get(key)
            for key in (
                "enabled",
                "published_images",
                "first_image_provisional",
                "first_image_seconds",
                "first_image_from_created_seconds",
                "first_image_refinement_pending",
                "first_image_refined_seconds",
                "first_guidance_from_created_seconds",
                "model_first_content_ms",
                "initial_first_image_source",
                "first_image_source",
                "model_required_for_first_image",
                "provider_prefix_ms",
            )
        }
    raw_concurrency = state.get("concurrency")
    if isinstance(raw_streaming, dict) and isinstance(raw_streaming.get("guidance_tiers"), dict):
        result["streaming"]["guidance_tiers"] = {
            kind: {
                "published": entry.get("published") is True,
                "first_published_from_created_seconds": entry.get("first_published_from_created_seconds"),
            }
            for kind in ("elite", "upper", "middle", "lower", "novice")
            if isinstance((entry := raw_streaming["guidance_tiers"].get(kind)), dict)
        }
    if isinstance(raw_concurrency, dict):
        result["concurrency"] = {
            key: raw_concurrency.get(key)
            for key in (
                "ocr_slot_wait_seconds",
                "ocr_slot_processing_seconds",
                "model_slot_wait_seconds",
                "model_call_seconds",
            )
        }
    raw_scheduler = state.get("scheduler")
    if isinstance(raw_scheduler, dict):
        result["scheduler"] = {
            key: raw_scheduler.get(key)
            for key in (
                "queue_depth_observed",
                "worker_count",
                "ocr_concurrency",
                "model_concurrency",
            )
        }
    return result


INTERNAL_DEVICE_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
DEVICE_SN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def _request_device(request: Request) -> tuple[str, str | None]:
    """Read the gateway-authenticated opaque device id and optional SN."""

    raw_device_id = request.headers.get("x-test6-device-id", "").strip().lower()
    if raw_device_id and INTERNAL_DEVICE_ID_PATTERN.fullmatch(raw_device_id) is None:
        raise HTTPException(400, "设备身份格式不正确")
    raw_device_sn = request.headers.get("x-device-sn", "").strip()
    if raw_device_sn and DEVICE_SN_PATTERN.fullmatch(raw_device_sn) is None:
        raise HTTPException(400, "设备SN码格式不正确")
    return raw_device_id or "direct", raw_device_sn or None


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/health")
def health() -> JSONResponse:
    alive = worker.is_alive()
    worker_state = worker.snapshot()
    return JSONResponse(
        status_code=200 if alive else 503,
        content={
        "status": "ok" if alive else "unavailable",
        "service": "test6-physics-search-machine",
        "queue_capacity": QUEUE_CAPACITY,
        "queue_size": worker.pending.qsize(),
        "worker_alive": alive,
        "worker_healthy": worker.is_healthy(),
        "workers": worker_state,
        "ocr_concurrency": OCR_CONCURRENCY,
        "model_concurrency": MODEL_CONCURRENCY,
        },
    )


@app.post("/api/jobs", status_code=202)
async def create_job(request: Request, image: UploadFile = File(...)) -> dict:
    if not worker.is_alive():
        raise HTTPException(503, "处理服务暂不可用，请稍后重试")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(415, "仅支持图片文件")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "图片不能超过20 MB")
    device_id, device_sn = _request_device(request)
    try:
        state = await asyncio.to_thread(
            pipeline.new_job,
            data,
            image.filename or "camera.jpg",
            device_id=device_id,
            device_sn=device_sn,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    queue_snapshot = worker.snapshot()
    state["scheduler"] = {
        "queue_depth_observed": int(queue_snapshot["pending_jobs"]) + 1,
        "worker_count": JOB_WORKERS,
        "ocr_concurrency": OCR_CONCURRENCY,
        "model_concurrency": MODEL_CONCURRENCY,
    }
    pipeline._write_state(pipeline.job_path(str(state["job_id"])), state, "queued")
    submitted = worker.submit(str(state["job_id"]), device_id)
    if not submitted.accepted:
        job_dir = pipeline.job_path(str(state["job_id"]))
        error_type = (
            "DeviceQueueFull"
            if submitted.reason == "device_queue_full"
            else "QueueFull"
        )
        message = (
            "该设备已有较多任务等待，请等待前序任务完成"
            if error_type == "DeviceQueueFull"
            else "当前任务较多，请稍后重试"
        )
        state["error"] = {"type": error_type, "message": message}
        pipeline._write_state(job_dir, state, "failed")
        raise HTTPException(429, message)
    return public_job(state)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return public_job(pipeline.read_job(job_id))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "任务不存在") from exc


@app.get("/jobs/{job_id}/{filename}", include_in_schema=False)
def get_result_image(job_id: str, filename: str) -> FileResponse:
    if filename not in DEVICE_IMAGE_NAMES:
        raise HTTPException(404, "结果图不存在")
    try:
        job_dir = pipeline.job_path(job_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "任务不存在") from exc
    target = job_dir / filename
    if not target.is_file():
        raise HTTPException(404, "结果图尚未生成")
    return FileResponse(
        target,
        media_type="image/png",
        headers={
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
