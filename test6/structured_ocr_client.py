#!/usr/bin/env python3
"""Safe loopback client for Test6's optional local structured OCR service."""

from __future__ import annotations

import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

import cv2
import numpy as np


OCR_IMAGE_BUDGET_SECONDS = 5.0
OCR_RENDER_RESERVE_SECONDS = 0.45
MIN_OCR_NETWORK_SECONDS = 0.6
FORMULA_TIMEOUT_CAP = 2.8
STRUCTURED_TIMEOUT_CAP = 3.2


def ocr_image_budget_seconds() -> float:
    """Hard budget for publishing the OCR PNG, including a render reserve."""

    try:
        return max(
            2.5,
            min(
                8.0,
                float(os.environ.get("TEST6_OCR_IMAGE_BUDGET_SECONDS", "5")),
            ),
        )
    except ValueError:
        return OCR_IMAGE_BUDGET_SECONDS


def remaining_ocr_timeout(started_at: float, *, kind: str) -> float | None:
    """Seconds left for one formula/structured call, or None if the 5s image would slip."""

    leftover = (
        ocr_image_budget_seconds()
        - (time.perf_counter() - float(started_at))
        - OCR_RENDER_RESERVE_SECONDS
    )
    if leftover < MIN_OCR_NETWORK_SECONDS:
        return None
    cap = FORMULA_TIMEOUT_CAP if kind == "formula" else STRUCTURED_TIMEOUT_CAP
    env_key = (
        "TEST6_FORMULA_ONLY_TIMEOUT"
        if kind == "formula"
        else "TEST6_FORMULA_OCR_TIMEOUT"
    )
    try:
        env_cap = float(os.environ.get(env_key, str(cap)))
    except ValueError:
        env_cap = cap
    return round(max(MIN_OCR_NETWORK_SECONDS, min(cap, env_cap, leftover)), 3)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _clamp_timeout(timeout: float | None, default: float, cap: float) -> float:
    value = default if timeout is None else float(timeout)
    return max(0.4, min(cap, value))


def _service_url() -> str:
    raw = os.environ.get("TEST6_FORMULA_OCR_URL", "").strip().rstrip("/")
    if not raw:
        return ""
    parsed = urlparse(raw)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return ""
    return raw


def request_structured_ocr(
    image: np.ndarray,
    timeout: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return a local structured transcript; failures leave RapidOCR untouched."""

    base_url = _service_url()
    audit: dict[str, Any] = {
        "enabled": bool(base_url),
        "available": False,
        "selected": False,
        "engine": "local-pp-doclayoutv2-plus-pp-formulanet",
        "formula_count": 0,
        "formulas": [],
        "formula_items": [],
        "spatial_items": [],
        "handwriting_filtered_count": 0,
        "layout_regions": [],
        "visual_region_count": 0,
        "elapsed_ms": 0.0,
        "error": None,
    }
    if not base_url:
        audit["error"] = "service_not_configured"
        return "", audit
    if not isinstance(image, np.ndarray) or image.size == 0:
        audit["error"] = "empty_image"
        return "", audit

    started = time.perf_counter()
    ok, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    )
    if not ok:
        audit["error"] = "png_encode_failed"
        return "", audit
    timeout = _clamp_timeout(
        timeout,
        _env_float("TEST6_FORMULA_OCR_TIMEOUT", STRUCTURED_TIMEOUT_CAP),
        STRUCTURED_TIMEOUT_CAP,
    )
    request = Request(
        f"{base_url}/recognize",
        data=encoded.tobytes(),
        method="POST",
        headers={"Content-Type": "image/png", "Accept": "application/json"},
    )
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            if int(getattr(response, "status", 200)) != 200:
                raise OSError(f"unexpected_status_{getattr(response, 'status', 0)}")
            payload = json.loads(response.read().decode("utf-8"))
        transcript = str(payload.get("transcript") or "").strip()
        formulas = [
            str(item).strip()
            for item in (payload.get("formulas") or [])
            if str(item).strip()
        ]
        formula_items = [
            dict(item)
            for item in (payload.get("items") or [])
            if isinstance(item, dict)
            and str(item.get("label") or "") in {"inline_formula", "display_formula"}
            and item.get("bbox") is not None
            and str(item.get("latex") or "").strip()
        ]
        spatial_items = [
            dict(item)
            for item in (payload.get("items") or [])
            if isinstance(item, dict)
            and str(item.get("label") or "")
            in {"inline_formula", "display_formula", "ocr_text", "text"}
            and item.get("bbox") is not None
            and str(item.get("latex") or item.get("text") or "").strip()
        ]
        audit.update(
            {
                "available": bool(transcript),
                "formula_count": len(formulas),
                "formulas": formulas,
                "formula_items": formula_items,
                "spatial_items": spatial_items,
                "service_elapsed_ms": float(payload.get("elapsed_ms") or 0.0),
                "item_count": len(payload.get("items") or []),
                "layout_regions": list(payload.get("layout_regions") or []),
                "visual_region_count": int(
                    payload.get("visual_region_count") or 0
                ),
                "handwriting_filtered_count": int(
                    payload.get("handwriting_filtered_count") or 0
                ),
            }
        )
        return transcript, audit
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        audit["error"] = type(exc).__name__
        return "", audit
    finally:
        audit["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)


def request_structured_layout(image: np.ndarray) -> dict[str, Any]:
    """Return visual regions from the resident local PP-DocLayoutV2 model.

    This endpoint intentionally skips text and formula recognition.  It is a
    semantic-completeness fallback for questions that explicitly reference a
    diagram/table, while keeping ordinary text-only questions on the faster
    RapidOCR path.
    """

    base_url = _service_url()
    audit: dict[str, Any] = {
        "enabled": bool(base_url),
        "available": False,
        "engine": "local-pp-doclayoutv2",
        "layout_regions": [],
        "visual_region_count": 0,
        "elapsed_ms": 0.0,
        "service_elapsed_ms": 0.0,
        "error": None,
    }
    if not base_url:
        audit["error"] = "service_not_configured"
        return audit
    if not isinstance(image, np.ndarray) or image.size == 0:
        audit["error"] = "empty_image"
        return audit

    started = time.perf_counter()
    ok, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    )
    if not ok:
        audit["error"] = "png_encode_failed"
        return audit
    timeout = max(
        1.0,
        min(12.0, float(os.environ.get("TEST6_LAYOUT_OCR_TIMEOUT", "8"))),
    )
    request = Request(
        f"{base_url}/layout",
        data=encoded.tobytes(),
        method="POST",
        headers={"Content-Type": "image/png", "Accept": "application/json"},
    )
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            if int(getattr(response, "status", 200)) != 200:
                raise OSError(f"unexpected_status_{getattr(response, 'status', 0)}")
            payload = json.loads(response.read().decode("utf-8"))
        regions = list(payload.get("layout_regions") or [])
        audit.update(
            {
                "available": bool(regions),
                "layout_regions": regions,
                "visual_region_count": len(regions),
                "service_elapsed_ms": float(payload.get("elapsed_ms") or 0.0),
            }
        )
        return audit
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        audit["error"] = type(exc).__name__
        return audit
    finally:
        audit["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)


def request_formula_ocr(
    image: np.ndarray,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Recognize only formula regions, preserving RapidOCR prose and layout."""

    base_url = _service_url()
    audit: dict[str, Any] = {
        "enabled": bool(base_url),
        "available": False,
        "selected": False,
        "engine": "local-pp-doclayoutv2-plus-pp-formulanet-formula-only",
        "formula_count": 0,
        "formulas": [],
        "formula_items": [],
        "layout_regions": [],
        "visual_region_count": 0,
        "elapsed_ms": 0.0,
        "service_elapsed_ms": 0.0,
        "error": None,
    }
    if not base_url:
        audit["error"] = "service_not_configured"
        return audit
    if not isinstance(image, np.ndarray) or image.size == 0:
        audit["error"] = "empty_image"
        return audit

    started = time.perf_counter()
    ok, encoded = cv2.imencode(
        ".png",
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    )
    if not ok:
        audit["error"] = "png_encode_failed"
        return audit
    timeout = _clamp_timeout(
        timeout,
        _env_float("TEST6_FORMULA_ONLY_TIMEOUT", FORMULA_TIMEOUT_CAP),
        FORMULA_TIMEOUT_CAP,
    )
    try:
        # Test6 deployments can point at either the newer structured_service_v2
        # route or the original service bundled with this repository.  Probe
        # the new route first and fall back only on an explicit 404; a timeout
        # or server error must not double the request latency.
        payload = None
        endpoint_used = ""
        for endpoint in ("recognize-formulas", "formulas"):
            request = Request(
                f"{base_url}/{endpoint}",
                data=encoded.tobytes(),
                method="POST",
                headers={"Content-Type": "image/png", "Accept": "application/json"},
            )
            try:
                with build_opener(ProxyHandler({})).open(
                    request, timeout=timeout
                ) as response:
                    if int(getattr(response, "status", 200)) != 200:
                        raise OSError(
                            f"unexpected_status_{getattr(response, 'status', 0)}"
                        )
                    payload = json.loads(response.read().decode("utf-8"))
                    endpoint_used = endpoint
                    break
            except HTTPError as exc:
                if int(getattr(exc, "code", 0)) == 404 and endpoint == "recognize-formulas":
                    continue
                raise
        if payload is None:
            raise OSError("formula_endpoint_not_found")
        formulas = [
            str(item).strip()
            for item in (payload.get("formulas") or [])
            if str(item).strip()
        ]
        formula_items = [
            dict(item)
            for item in (payload.get("formula_items") or [])
            if isinstance(item, dict)
        ]
        audit.update(
            {
                "available": bool(formula_items),
                "formula_count": len(formulas),
                "formulas": formulas,
                "formula_items": formula_items,
                "layout_regions": list(payload.get("layout_regions") or []),
                "visual_region_count": int(payload.get("visual_region_count") or 0),
                "service_elapsed_ms": float(payload.get("elapsed_ms") or 0.0),
                "layout_ms": float(payload.get("layout_ms") or 0.0),
                "formula_ms": float(payload.get("formula_ms") or 0.0),
                "endpoint": endpoint_used,
            }
        )
        return audit
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        audit["error"] = type(exc).__name__
        return audit
    finally:
        audit["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
