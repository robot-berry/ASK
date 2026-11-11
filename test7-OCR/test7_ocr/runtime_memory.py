from __future__ import annotations


DEFAULT_VRAM_RESERVED_FRACTION_LIMIT = 0.82
DEFAULT_VRAM_MINIMUM_FREE_BYTES = 1024 * 1024 * 1024


def should_release_cached_vram(
    *,
    total_bytes: int,
    reserved_bytes: int,
    free_bytes: int,
    reserved_fraction_limit: float = DEFAULT_VRAM_RESERVED_FRACTION_LIMIT,
    minimum_free_bytes: int = DEFAULT_VRAM_MINIMUM_FREE_BYTES,
) -> bool:
    """Return whether a long-lived OCR service should purge its CUDA cache.

    MinerU normally empties the CUDA allocator and runs a full Python garbage
    collection after several stages on an 8 GB GPU.  That is appropriate for a
    one-shot CLI, but it destroys reusable convolution workspaces in this
    resident single-request service.  Keep the cache only while there is ample
    headroom and fall back to MinerU's original cleanup under real pressure.
    """

    total = max(0, int(total_bytes))
    reserved = max(0, int(reserved_bytes))
    free = max(0, int(free_bytes))
    if total <= 0:
        return True
    limit = min(1.0, max(0.0, float(reserved_fraction_limit)))
    minimum_free = max(0, int(minimum_free_bytes))
    return reserved / total >= limit or free < minimum_free
