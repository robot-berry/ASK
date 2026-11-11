from __future__ import annotations

import argparse
import base64
import binascii
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import io
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from PIL import Image, UnidentifiedImageError

from .cli import _clean_source_resolver_from_config, _provider_from_args
from .deferred_verification import verify_sample_artifact
from .pipeline import QuestionPipeline, load_config
from .providers import TimedRapidOcrProvider
from .runtime_profiles import (
    PRODUCT_V7_SAFE_RUNTIME_PROFILE,
    apply_runtime_profile,
    authenticated_source_sample_id,
)
from .storage import safe_name, write_json_atomic


PRODUCT_API_VERSION = "product_ocr_v8"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
SUPPORTED_IMAGE_FORMATS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}
SOURCE_TOKEN = re.compile(r"^[0-9a-f]{64}\.(?:jpg|png|webp)$")
SAMPLE_TOKEN = re.compile(r"^[0-9A-Za-z._-]+$")
PRODUCT_FEEDBACK_ISSUES = frozenset(
    {
        "semantic_content_incomplete",
        "text_or_symbol_error",
        "formula_error",
        "latex_source_visible",
        "visual_crop_incomplete",
        "handwriting_visible",
        "adjacent_question_or_noise",
        "layout_or_linebreak",
        "truncated_or_overflow",
        "blur_or_unreadable",
        "other",
    }
)


class ProductOcrService:
    """Run real user images without mixing them into Gold or benchmark runs."""

    def __init__(
        self,
        *,
        input_root: Path | None = None,
        output_root: Path | None = None,
        feedback_root: Path | None = None,
        config_path: Path | None = None,
        service_url: str = "http://127.0.0.1:8240",
        pipeline_factory: Callable[[], QuestionPipeline] | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.input_root = (
            input_root or project_root / "artifacts" / "product_inputs"
        ).resolve()
        self.output_root = (
            output_root or project_root / "artifacts" / "product_runs"
        ).resolve()
        self.feedback_root = (
            feedback_root or project_root / "artifacts" / "product_feedback"
        ).resolve()
        self.config_path = (
            config_path or project_root / "configs" / "pipeline.json"
        ).resolve()
        self.service_url = service_url
        self.run_id = "live"
        self._pipeline_factory = pipeline_factory or self._build_pipeline
        self._pipeline: QuestionPipeline | None = None
        self._lock = threading.Lock()
        self._result_lock = threading.Lock()
        self._verification_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="test7-product-final-png-verifier",
        )
        self._verification_jobs: dict[str, Future[None]] = {}
        self._closed = False
        self._last_error: str | None = None

    def _runtime_config(self) -> dict[str, Any]:
        return apply_runtime_profile(
            load_config(self.config_path),
            PRODUCT_V7_SAFE_RUNTIME_PROFILE,
        )

    def _build_pipeline(self) -> QuestionPipeline:
        project_root = Path(__file__).resolve().parent.parent
        config = self._runtime_config()
        args = argparse.Namespace(
            provider="auto",
            sidecar=None,
            review_dir=(
                project_root
                / "data"
                / "annotations"
                / "gaokao_physics_ocr_gold"
                / "drafts"
            ),
            visual_candidates=(
                project_root
                / "data"
                / "splits"
                / "gaokao_physics_visual_bbox_candidates.jsonl"
            ),
            service_url=self.service_url,
            allow_eval_prelabel=False,
        )
        provider = _provider_from_args(args, config)
        resolver = _clean_source_resolver_from_config(config)
        return QuestionPipeline(
            provider,
            config,
            self.output_root,
            self.run_id,
            source_resolver=resolver,
        )

    def prepare(self) -> None:
        with self._lock:
            self._ensure_pipeline()

    def _ensure_pipeline(self) -> QuestionPipeline:
        if self._pipeline is None:
            try:
                self._pipeline = self._pipeline_factory()
                self._last_error = None
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise
        return self._pipeline

    def health(self) -> dict[str, Any]:
        with self._result_lock:
            pending_verifications = sum(
                1 for future in self._verification_jobs.values() if not future.done()
            )
        return {
            "version": PRODUCT_API_VERSION,
            "ready": self._pipeline is not None,
            "provider_mode": "auto (structured service with local fallback)",
            "service_url": self.service_url,
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "accepted_formats": ["JPEG", "PNG", "WebP"],
            "output_width_px": 800,
            "last_error": self._last_error,
            "final_png_verification": {
                "mode": "deferred_hash_bound",
                "pending_jobs": pending_verifications,
                "download_requires_completion": True,
            },
            "storage": {
                "inputs": str(self.input_root),
                "outputs": str(self.output_root / self.run_id),
                "feedback": str(self.feedback_root),
                "isolated_from_gold_and_benchmarks": True,
            },
        }

    def _result_record_path(self, sample_id: str) -> Path:
        if not SAMPLE_TOKEN.fullmatch(sample_id):
            raise ValueError("无效的结果标识")
        run_root = (self.output_root / self.run_id).resolve()
        path = (run_root / sample_id / "product_result.json").resolve()
        if path.parent.parent != run_root:
            raise ValueError("结果路径越界")
        return path

    def result_details(self, sample_id: str) -> dict[str, Any]:
        path = self._result_record_path(sample_id)
        if not path.is_file():
            raise FileNotFoundError(sample_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("version") != PRODUCT_API_VERSION
            or payload.get("sample_id") != sample_id
        ):
            raise ValueError("结果记录版本或标识不匹配")
        self.output_path(sample_id)
        source_token = str(payload.get("source_url") or "").rsplit("/", 1)[-1]
        self.source_path(source_token)
        payload = dict(payload)
        payload["feedback"] = self.load_feedback(sample_id)
        return payload

    def _feedback_path(self, sample_id: str) -> Path:
        if not SAMPLE_TOKEN.fullmatch(sample_id):
            raise ValueError("无效的结果标识")
        path = (self.feedback_root / f"{sample_id}.json").resolve()
        if path.parent != self.feedback_root:
            raise ValueError("反馈路径越界")
        return path

    def load_feedback(self, sample_id: str) -> dict[str, Any] | None:
        path = self._feedback_path(sample_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("sample_id") != sample_id:
            raise ValueError("反馈记录与结果标识不匹配")
        return payload

    def list_feedback(self) -> list[dict[str, Any]]:
        if not self.feedback_root.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for path in self.feedback_root.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("verdict") == "fail"
                    and SAMPLE_TOKEN.fullmatch(str(payload.get("sample_id") or ""))
                ):
                    items.append(payload)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return sorted(
            items,
            key=lambda item: str(item.get("updated_at") or ""),
            reverse=True,
        )

    def save_feedback(self, sample_id: str, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("问题反馈必须是 JSON 对象")
        requested_issues = payload.get("issues")
        if not isinstance(requested_issues, list):
            raise ValueError("请至少选择一个问题类型")
        issues = sorted({str(value) for value in requested_issues})
        if not issues:
            raise ValueError("请至少选择一个问题类型")
        unknown = sorted(set(issues) - PRODUCT_FEEDBACK_ISSUES)
        if unknown:
            raise ValueError("包含不支持的问题类型：" + ", ".join(unknown))
        notes = str(payload.get("notes") or "").strip()
        if len(notes) > 2000:
            raise ValueError("补充说明不能超过 2000 个字符")

        result = self.result_details(sample_id)
        source_token = str(result["source_url"]).rsplit("/", 1)[-1]
        source_path = self.source_path(source_token)
        output_path = self.output_path(sample_id)
        source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        if source_sha256 != result["input"]["sha256"]:
            raise ValueError("原图哈希已变化，拒绝保存反馈")
        if output_sha256 != result["output"]["sha256"]:
            raise ValueError("成图哈希已变化，拒绝保存反馈")

        now = datetime.now(timezone.utc).isoformat()
        previous = self.load_feedback(sample_id)
        history = list(previous.get("history") or []) if previous else []
        if previous:
            history.append(
                {
                    "updated_at": previous.get("updated_at"),
                    "issues": previous.get("issues") or [],
                    "notes": previous.get("notes") or "",
                }
            )
        record = {
            "schema_version": "product_feedback_v1",
            "sample_id": sample_id,
            "verdict": "fail",
            "issues": issues,
            "notes": notes,
            "source_sha256": source_sha256,
            "output_sha256": output_sha256,
            "pipeline_status": result["status"],
            "created_at": previous.get("created_at") if previous else now,
            "updated_at": now,
            "history": history,
            "gold_modified": False,
            "regression_queue": True,
        }
        write_json_atomic(self._feedback_path(sample_id), record)
        return record

    @staticmethod
    def _decode_upload(payload: object) -> tuple[bytes, str, int, int, str]:
        if not isinstance(payload, dict):
            raise ValueError("上传请求必须是 JSON 对象")
        encoded = payload.get("image_base64")
        if not isinstance(encoded, str) or not encoded.strip():
            raise ValueError("请选择一张原题图片")
        encoded = encoded.strip()
        if encoded.startswith("data:"):
            try:
                encoded = encoded.split(",", 1)[1]
            except IndexError as exc:
                raise ValueError("图片数据格式无效") from exc
        estimated_bytes = len(encoded) * 3 // 4
        if estimated_bytes > MAX_UPLOAD_BYTES + 3:
            raise ValueError("图片超过 20MB 上限")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("图片 Base64 数据无效") from exc
        if not raw or len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("图片为空或超过 20MB 上限")
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                image_format = str(opened.format or "").upper()
                width, height = opened.size
                opened.verify()
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
            raise ValueError("文件不是可读取的 JPEG、PNG 或 WebP 图片") from exc
        if image_format not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError("仅支持 JPEG、PNG 或 WebP 图片")
        if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
            raise ValueError("图片尺寸无效或像素数超过 5000 万")
        return raw, SUPPORTED_IMAGE_FORMATS[image_format], width, height, image_format

    def _store_source(self, raw: bytes, extension: str) -> tuple[Path, str]:
        digest = hashlib.sha256(raw).hexdigest()
        token = f"{digest}{extension}"
        self.input_root.mkdir(parents=True, exist_ok=True)
        path = (self.input_root / token).resolve()
        if path.parent != self.input_root:
            raise ValueError("上传文件路径越界")
        if not path.is_file():
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(raw)
            os.replace(temporary, path)
        return path, token

    @staticmethod
    def _status_message(status: str, *, verification_state: str) -> str:
        if verification_state == "pending":
            return "OCR 与 800px 重排已完成，正在后台独立复检最终 PNG；复检通过后自动解锁下载。"
        if verification_state == "failed":
            return "OCR 与重排已完成，但后台最终 PNG 复检执行失败；结果保持锁定。"
        if status == "PASS":
            return "自动校验和最终 PNG 独立复检均已通过，可以下载。"
        return "自动校验未通过，结果仅供诊断；修正或复核后才能作为成品。"

    def _update_deferred_result(
        self,
        sample_id: str,
        verified_report: dict[str, Any] | None,
        error: Exception | None = None,
    ) -> None:
        result_path = self._result_record_path(sample_id)
        with self._result_lock:
            if not result_path.is_file():
                return
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if error is not None or verified_report is None:
                payload["status"] = "REVIEW"
                payload["safe_to_download"] = False
                payload["download_url"] = None
                payload["status_message"] = self._status_message(
                    "REVIEW", verification_state="failed"
                )
                payload["verification"] = {
                    "state": "failed",
                    "mode": "deferred_hash_bound",
                    "error": (
                        f"{type(error).__name__}: {error}"
                        if error is not None
                        else "Deferred verification returned no report"
                    ),
                }
                write_json_atomic(result_path, payload)
                return

            if (
                str(verified_report.get("sample_id")) != sample_id
                or str(verified_report.get("output_sha256"))
                != str(payload.get("output", {}).get("sha256"))
            ):
                raise ValueError("后台复检结果与产品成图身份不一致")
            readback = verified_report.get("render_readback") or {}
            status = str(verified_report.get("status") or "REVIEW")
            readback_passed = readback.get("passed") is True
            safe_to_download = status == "PASS" and readback_passed
            payload["status"] = status
            payload["safe_to_download"] = safe_to_download
            payload["download_url"] = (
                f"/api/product/result/{quote(sample_id, safe='')}/output.png?download=1"
                if safe_to_download
                else None
            )
            payload["status_message"] = self._status_message(
                status, verification_state="complete"
            )
            payload["issues"] = list(verified_report.get("issues") or [])
            payload["verification"] = {
                "state": "complete",
                "mode": "deferred_hash_bound",
                "passed": readback_passed,
                "verified_at": readback.get("verified_at"),
                "elapsed_seconds": readback.get("verification_elapsed_seconds"),
            }
            write_json_atomic(result_path, payload)

    def _run_deferred_verification(self, sample_id: str, report_path: Path) -> None:
        provider: TimedRapidOcrProvider | None = None
        try:
            config = self._runtime_config()
            provider = TimedRapidOcrProvider(
                timeout_seconds=30.0,
                params=config.get("render_verification", {}).get("rapidocr_params"),
            )
            verified_report = verify_sample_artifact(report_path, config, provider)
            self._update_deferred_result(sample_id, verified_report)
        except Exception as exc:
            self._update_deferred_result(sample_id, None, exc)
        finally:
            if provider is not None:
                provider.close()

    def _schedule_deferred_verification(
        self, sample_id: str, report_path: Path
    ) -> None:
        with self._result_lock:
            if self._closed:
                raise RuntimeError("OCR 产品服务正在关闭")
            future = self._verification_executor.submit(
                self._run_deferred_verification,
                sample_id,
                report_path,
            )
            self._verification_jobs[sample_id] = future

    def process_upload(self, payload: object) -> dict[str, Any]:
        request_started = time.perf_counter()
        raw, extension, input_width, input_height, image_format = self._decode_upload(
            payload
        )
        source_path, source_token = self._store_source(raw, extension)
        sample_id = safe_name(
            "LIVE-"
            + time.strftime("%Y%m%d-%H%M%S")
            + "-"
            + source_token[:10]
            + "-"
            + hashlib.sha256(os.urandom(16)).hexdigest()[:6]
        )
        filename_stem = Path(str(payload.get("filename") or "")).stem
        recognition_sample_id = authenticated_source_sample_id(filename_stem)
        with self._lock:
            try:
                report = self._ensure_pipeline().process(
                    source_path,
                    sample_id,
                    recognition_sample_id=recognition_sample_id,
                )
                self._last_error = None
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise
        report_payload = report.to_dict()
        output_path = Path(report.output_image).resolve()
        if not output_path.is_file():
            raise RuntimeError("OCR 已返回，但最终 PNG 没有写入磁盘")
        if report.output_width != 800:
            raise RuntimeError(f"最终 PNG 宽度不是 800px：{report.output_width}px")
        quoted_sample = quote(sample_id, safe="")
        quoted_source = quote(source_token, safe="")
        issue_payloads = report_payload.get("issues") or []
        report_path = (self.output_root / self.run_id / sample_id / "report.json").resolve()
        persisted_report: dict[str, Any] = {}
        if report_path.is_file():
            persisted_report = json.loads(report_path.read_text(encoding="utf-8"))
        readback = persisted_report.get("render_readback") or {}
        deferred_pending = readback.get("pending") is True
        verification_state = "pending" if deferred_pending else "complete"
        safe_to_download = (
            report.status == "PASS"
            and not deferred_pending
            and readback.get("passed") is True
        )
        response = {
            "version": PRODUCT_API_VERSION,
            "sample_id": sample_id,
            "status": report.status,
            "safe_to_download": safe_to_download,
            "status_message": self._status_message(
                report.status,
                verification_state=verification_state,
            ),
            "source_url": f"/api/product/source/{quoted_source}",
            "output_url": f"/api/product/result/{quoted_sample}/output.png",
            "download_url": (
                f"/api/product/result/{quoted_sample}/output.png?download=1"
                if safe_to_download
                else None
            ),
            "input": {
                "filename": str(payload.get("filename") or "原题图片"),
                "format": image_format,
                "width": input_width,
                "height": input_height,
                "sha256": report.input_source_sha256 or report.source_sha256,
            },
            "output": {
                "width": report.output_width,
                "height": report.output_height,
                "sha256": report.output_sha256,
            },
            "provider": report.provider,
            "pipeline_elapsed_seconds": report.elapsed_seconds,
            "request_elapsed_seconds": round(time.perf_counter() - request_started, 3),
            "issues": issue_payloads,
            "timings_ms": report.timings_ms,
            "verification": {
                "state": verification_state,
                "mode": "deferred_hash_bound",
                "passed": readback.get("passed") is True,
            },
            "storage": {
                "gold_or_benchmark_modified": False,
                "input_deduplicated_by_sha256": True,
            },
        }
        write_json_atomic(self._result_record_path(sample_id), response)
        if deferred_pending:
            self._schedule_deferred_verification(sample_id, report_path)
        return response

    def source_path(self, token: str) -> Path:
        if not SOURCE_TOKEN.fullmatch(token):
            raise ValueError("无效的原图标识")
        path = (self.input_root / token).resolve()
        if path.parent != self.input_root:
            raise ValueError("原图路径越界")
        if not path.is_file():
            raise FileNotFoundError(token)
        return path

    def output_path(self, sample_id: str) -> Path:
        if not SAMPLE_TOKEN.fullmatch(sample_id):
            raise ValueError("无效的结果标识")
        run_root = (self.output_root / self.run_id).resolve()
        path = (run_root / sample_id / "output.png").resolve()
        if path.parent.parent != run_root:
            raise ValueError("结果路径越界")
        if not path.is_file():
            raise FileNotFoundError(sample_id)
        return path

    def verified_download_path(self, sample_id: str) -> Path:
        result = self.result_details(sample_id)
        if (
            result.get("safe_to_download") is not True
            or result.get("verification", {}).get("state") != "complete"
            or result.get("verification", {}).get("passed") is not True
            or result.get("feedback") is not None
        ):
            raise RuntimeError("最终 PNG 尚未通过完整自动门禁，禁止下载")
        return self.output_path(sample_id)

    def close(self) -> None:
        with self._result_lock:
            self._closed = True
        self._verification_executor.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            if self._pipeline is not None:
                self._pipeline.close()
                self._pipeline = None
