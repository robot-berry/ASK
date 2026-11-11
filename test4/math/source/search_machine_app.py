#!/usr/bin/env python3
"""FastAPI entry point for the Test4 mathematics search machine."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import queue
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from device_result_renderer import DEVICE_IMAGE_NAMES
from search_machine_pipeline import MAX_UPLOAD_BYTES, RUNS_ROOT, SearchMachinePipeline


SOURCE_DIR = Path(__file__).resolve().parent
STATIC_DIR = SOURCE_DIR / "search_machine_static"


def _setting(primary: str, legacy: str, default: str) -> str:
    return os.environ.get(primary, "").strip() or os.environ.get(legacy, "").strip() or default


QUEUE_CAPACITY = max(
    1,
    min(
        20,
        int(
            os.environ.get("TEST4_MATH_QUEUE_CAPACITY", "").strip()
            or _setting("MATH_TEST3_QUEUE_CAPACITY", "TEST3_QUEUE_CAPACITY", "4")
        ),
    ),
)
pipeline = SearchMachinePipeline(
    model_id=os.environ.get("TEST4_MATH_MODEL_ID", "").strip()
    or _setting(
        "MATH_TEST3_MODEL_ID", "TEST3_MODEL_ID", "qwen3.7-flash-2026-07-15"
    )
)
LOGGER = logging.getLogger(__name__)


class JobWorker:
    """A bounded, single-consumer queue suitable for one hardware device."""

    def __init__(self, capacity: int):
        self.pending: queue.Queue[str | None] = queue.Queue(maxsize=capacity)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._loop, name="math-test3-job-worker", daemon=True
        )
        self.thread.start()
        self.recover_interrupted_jobs()

    def stop(self) -> None:
        self.stop_event.set()
        try:
            self.pending.put_nowait(None)
        except queue.Full:
            pass
        if self.thread:
            self.thread.join(timeout=3)

    def submit(self, job_id: str) -> bool:
        try:
            self.pending.put_nowait(job_id)
            return True
        except queue.Full:
            return False

    def is_alive(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                job_id = self.pending.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if job_id is None:
                    return
                try:
                    pipeline.solve(job_id)
                except Exception:
                    # A corrupt/missing job file can fail before solve() enters
                    # its own guarded processing block.  Keep the sole worker
                    # alive so later phone uploads are not stranded forever.
                    LOGGER.exception("unhandled job worker failure: %s", job_id)
                    try:
                        state = pipeline.read_job(job_id)
                        job_dir = pipeline.job_path(job_id)
                        state["error"] = {
                            "type": "WorkerUnhandledError",
                            "message": "服务器处理任务时出现内部错误，请重新提交",
                        }
                        pipeline._write_state(job_dir, state, "failed")
                    except Exception:
                        LOGGER.exception(
                            "could not persist failed worker job: %s", job_id
                        )
            finally:
                self.pending.task_done()

    def recover_interrupted_jobs(self) -> None:
        """Fail stale in-process jobs and requeue jobs never started."""

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
            elif status == "queued" and not self.submit(job_dir.name):
                state["error"] = {
                    "type": "QueueFullAfterRestart",
                    "message": "服务器重启后队列已满，请重新提交",
                }
                pipeline._write_state(job_dir, state, "failed")


worker = JobWorker(QUEUE_CAPACITY)


@asynccontextmanager
async def lifespan(_: FastAPI):
    worker.start()
    try:
        yield
    finally:
        worker.stop()


app = FastAPI(title="Test4 数学搜题机", version="4.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
            "QueueFullAfterRestart": "服务器重启后队列已满，请重新提交",
            "ServiceRestarted": "服务器处理中断，请重新拍照提交",
            "OutputTruncated": "普通题档输出容量不足，请改选复杂题档重新处理",
            "ComplexOutputTruncated": "复杂题档输出仍被截断，请裁剪图片，只保留当前一道题后重试",
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
        "complexity_tier": state.get("complexity_tier", "normal"),
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
    return result


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/health")
def health() -> JSONResponse:
    alive = worker.is_alive()
    return JSONResponse(
        status_code=200 if alive else 503,
        content={
        "status": "ok" if alive else "unavailable",
        "service": "test4-math-search-machine",
        "queue_capacity": QUEUE_CAPACITY,
        "queue_size": worker.pending.qsize(),
        "worker_alive": alive,
        },
    )


@app.post("/api/jobs", status_code=202)
async def create_job(
    image: UploadFile = File(...),
    complexity_tier: str = Form("normal"),
) -> dict:
    if not worker.is_alive():
        raise HTTPException(503, "处理服务暂不可用，请稍后重试")
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(415, "仅支持图片文件")
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "图片不能超过20 MB")
    complexity_tier = complexity_tier.strip().lower()
    if complexity_tier not in {"normal", "complex"}:
        raise HTTPException(422, "题目档位必须是 normal 或 complex")
    try:
        state = pipeline.new_job(data, image.filename or "camera.jpg")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    state["complexity_tier"] = complexity_tier
    pipeline._write_state(pipeline.job_path(str(state["job_id"])), state, "queued")
    if not worker.submit(str(state["job_id"])):
        job_dir = pipeline.job_path(str(state["job_id"]))
        state["error"] = {"type": "QueueFull", "message": "当前任务较多，请稍后重试"}
        pipeline._write_state(job_dir, state, "failed")
        raise HTTPException(429, "当前任务较多，请稍后重试")
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
