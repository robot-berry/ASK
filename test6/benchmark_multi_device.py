#!/usr/bin/env python3
"""Explicit live load test for Test6 hardware-style concurrent requests."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime as dt
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener
import uuid


MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return round(ordered[index], 3)


def metric_summary(records: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [float(item[key]) for item in records if isinstance(item.get(key), (int, float))]
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": round(max(values), 3) if values else None,
    }


def read_json_response(request: Request, timeout: float) -> dict[str, Any]:
    try:
        with build_opener().open(request, timeout=timeout) as response:
            raw = response.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise RuntimeError("response JSON exceeded 2 MiB")
    except HTTPError as exc:
        raw = exc.read(64 * 1024)
        try:
            detail = json.loads(raw.decode("utf-8")).get("detail")
        except Exception:
            detail = None
        raise RuntimeError(f"HTTP {exc.code}: {detail or exc.reason}") from None
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"network error: {exc}") from None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("response was not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("response JSON was not an object")
    return payload


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return round(result, 3) if math.isfinite(result) and result >= 0 else None


def run_device_request(
    *,
    base_url: str,
    token: str,
    subject: str,
    device_sn: str,
    image_data: bytes,
    media_type: str,
    timeout_seconds: float,
    poll_seconds: float,
) -> dict[str, Any]:
    created = time.perf_counter()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": media_type,
        "X-Device-SN": device_sn,
        "Idempotency-Key": f"{device_sn}-{uuid.uuid4().hex}",
        "Accept": "application/json",
    }
    record: dict[str, Any] = {"device_sn": device_sn, "subject": subject, "ok": False}
    try:
        payload = read_json_response(
            Request(
                f"{base_url}/api/v1/{subject}/jobs/raw",
                data=image_data,
                headers=headers,
                method="POST",
            ),
            timeout=min(30.0, timeout_seconds),
        )
        record["submit_seconds"] = round(time.perf_counter() - created, 3)
        record["job_id"] = payload.get("job_id")
        first_observed: float | None = None
        deadline = created + timeout_seconds
        while True:
            if payload.get("images") and first_observed is None:
                first_observed = round(time.perf_counter() - created, 3)
            status = str(payload.get("status") or "")
            if status in {"completed", "failed"}:
                break
            if time.perf_counter() >= deadline:
                raise RuntimeError("job polling timed out")
            time.sleep(poll_seconds)
            poll_url = str(payload.get("poll_url") or "")
            if not poll_url.startswith("/api/v1/"):
                raise RuntimeError("server returned an invalid poll_url")
            payload = read_json_response(
                Request(
                    base_url + poll_url,
                    headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
                    method="GET",
                ),
                timeout=min(30.0, max(1.0, deadline - time.perf_counter())),
            )
        timing = payload.get("timing") if isinstance(payload.get("timing"), dict) else {}
        scheduler = payload.get("scheduler") if isinstance(payload.get("scheduler"), dict) else {}
        record.update(
            {
                "status": status,
                "ok": status == "completed",
                "total_observed_seconds": round(time.perf_counter() - created, 3),
                "first_image_observed_seconds": first_observed,
                "queue_wait_seconds": numeric(timing.get("queue_wait_seconds")),
                "ocr_slot_wait_seconds": numeric(timing.get("ocr_slot_wait_seconds")),
                "ocr_processing_seconds": numeric(timing.get("ocr_processing_seconds")),
                "model_slot_wait_seconds": numeric(timing.get("model_slot_wait_seconds")),
                "model_call_seconds": numeric(timing.get("model_call_seconds")),
                "queue_depth_observed": scheduler.get("queue_depth_observed"),
                "worker_count": scheduler.get("worker_count"),
                "image_count": len(payload.get("images") or []),
            }
        )
        if status == "failed":
            record["error"] = str((payload.get("error") or {}).get("message") or "job failed")
    except Exception as exc:
        record["status"] = "request_failed"
        record["error"] = str(exc)
        record["total_observed_seconds"] = round(time.perf_counter() - created, 3)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--subject", choices=("math", "physics", "mixed"), default="mixed")
    parser.add_argument("--math-image", type=Path)
    parser.add_argument("--physics-image", type=Path)
    parser.add_argument("--devices", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--poll-seconds", type=float, default=0.6)
    parser.add_argument("--device-prefix", default="LOADTEST")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-live-calls", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_live_calls:
        raise SystemExit("Refusing live model calls without --confirm-live-calls")
    if not 1 <= args.devices <= 1000 or not 1 <= args.concurrency <= args.devices:
        raise SystemExit("devices must be 1..1000 and concurrency must be 1..devices")
    parsed = urlparse(args.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username:
        raise SystemExit("base-url must be a credential-free HTTP(S) origin")
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("non-loopback load tests require HTTPS")
    token = os.environ.get("TEST6_BENCHMARK_TOKEN", "").strip()
    if not token:
        raise SystemExit("set TEST6_BENCHMARK_TOKEN without putting it on the command line")
    images: dict[str, tuple[bytes, str]] = {}
    required = {args.subject} if args.subject != "mixed" else {"math", "physics"}
    for subject, path in (("math", args.math_image), ("physics", args.physics_image)):
        if subject not in required:
            continue
        if path is None or not path.is_file() or path.suffix.lower() not in MEDIA_TYPES:
            raise SystemExit(f"--{subject}-image must be a JPEG, PNG or WebP file")
        images[subject] = (path.read_bytes(), MEDIA_TYPES[path.suffix.lower()])
    jobs = []
    for index in range(1, args.devices + 1):
        subject = args.subject if args.subject != "mixed" else ("math" if index % 2 else "physics")
        jobs.append((index, subject))
    base_url = args.base_url.rstrip("/")
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = []
        for index, subject in jobs:
            image_data, media_type = images[subject]
            futures.append(
                executor.submit(
                    run_device_request,
                    base_url=base_url,
                    token=token,
                    subject=subject,
                    device_sn=f"{args.device_prefix}-{index:04d}",
                    image_data=image_data,
                    media_type=media_type,
                    timeout_seconds=args.timeout_seconds,
                    poll_seconds=args.poll_seconds,
                )
            )
        records = [future.result() for future in as_completed(futures)]
    records.sort(key=lambda item: str(item["device_sn"]))
    metrics = (
        "submit_seconds",
        "queue_wait_seconds",
        "ocr_slot_wait_seconds",
        "ocr_processing_seconds",
        "first_image_observed_seconds",
        "model_slot_wait_seconds",
        "model_call_seconds",
        "total_observed_seconds",
    )
    summary = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "base_url": base_url,
        "devices": args.devices,
        "concurrency": args.concurrency,
        "subject": args.subject,
        "completed": sum(item.get("ok") is True for item in records),
        "failed": sum(item.get("ok") is not True for item in records),
        "metrics_seconds": {key: metric_summary(records, key) for key in metrics},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fieldnames = sorted({key for item in records for key in item})
    with (args.output_dir / "requests.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
