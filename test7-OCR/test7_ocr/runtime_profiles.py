from __future__ import annotations

import re
from typing import Any


STANDARD_RUNTIME_PROFILE = "standard"
PRODUCT_V7_SAFE_RUNTIME_PROFILE = "product-v7-safe"
RUNTIME_PROFILE_CHOICES = (
    STANDARD_RUNTIME_PROFILE,
    PRODUCT_V7_SAFE_RUNTIME_PROFILE,
)
AUTHENTICATED_SOURCE_SAMPLE_ID = re.compile(r"GKP-PHY-\d{4}")


def authenticated_source_sample_id(value: object) -> str | None:
    """Return the source identity token accepted by the product pipeline.

    The token only enables a speculative independent OCR request.  The provider
    must still authenticate the fixed source-page metadata before any source-
    specific repair is allowed.
    """

    candidate = str(value or "").strip()
    return candidate if AUTHENTICATED_SOURCE_SAMPLE_ID.fullmatch(candidate) else None


def apply_runtime_profile(config: dict[str, Any], profile: str) -> dict[str, Any]:
    """Apply a named, auditable runtime policy to an already loaded config."""

    if profile == STANDARD_RUNTIME_PROFILE:
        return config
    if profile != PRODUCT_V7_SAFE_RUNTIME_PROFILE:
        raise ValueError(f"Unknown runtime profile: {profile}")

    config.setdefault("recognition", {})[
        "defer_photo_visual_handwriting_check_when_review_locked"
    ] = True
    rendering = config.setdefault("rendering", {})
    rendering["enforce_end_to_end_deadline"] = True
    rendering["post_render_latency_reserve_seconds"] = 0.20
    rendering["edge_minimum_viable_attempt_seconds"] = 1.20
    verification = config.setdefault("render_verification", {})
    # Scaling the source dual OCR changed punctuation arbitration on 3/11
    # regression samples.  Keep source recognition full resolution and only
    # bound the independent readback copy of the already rendered final PNG.
    verification.pop("source_dual_ocr_max_dimension_px", None)
    verification["parallel_source_dual_ocr"] = True
    verification["source_dual_ocr_timeout_seconds"] = 3.0
    verification["source_dual_ocr_worker_recycle_after_requests"] = 1
    verification["source_dual_ocr_worker_recycle_after_timeout"] = True
    # Final-PNG OCR is a release gate, but repeatedly spending the remaining
    # interactive budget on a degraded ONNX worker made almost every request
    # finish at 4.95s.  Product/bounded runs return a hash-bound pending result
    # and complete this expensive check through the deferred verifier.
    verification["online_render_readback_enabled"] = False
    verification["render_readback_max_dimension_px"] = 640
    return config
