from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .types import StoredSample


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return cleaned or "sample"


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def prepare_sample_storage(output_root: Path, run_id: str, sample_id: str) -> StoredSample:
    directory = output_root / safe_name(run_id) / safe_name(sample_id)
    asset_directory = directory / "assets"
    asset_directory.mkdir(parents=True, exist_ok=True)
    return StoredSample(
        directory=directory,
        output_path=directory / "output.png",
        document_path=directory / "question_document.json",
        report_path=directory / "report.json",
        asset_directory=asset_directory,
    )


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def save_png_atomic(image: Image.Image, path: Path, *, optimize: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.png")
    image.save(temporary, format="PNG", optimize=optimize)
    os.replace(temporary, path)


def image_pixel_sha256(image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(image.mode.encode("ascii"))
    digest.update(str(image.size).encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_jsonl_atomic(path: Path, payloads: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            for payload in payloads
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise FileNotFoundError(f"No existing filesystem ancestor for {path}")
    return candidate


def _sample_artifact_sizes(output_root: Path, limit: int) -> list[int]:
    if not output_root.is_dir() or limit <= 0:
        return []
    report_paths = list(output_root.glob("*/*/report.json"))
    report_paths.sort(
        key=lambda path: path.stat().st_mtime_ns if path.exists() else 0,
        reverse=True,
    )
    sizes: list[int] = []
    for report_path in report_paths[:limit]:
        total = 0
        for path in report_path.parent.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
            except OSError:
                continue
        if total > 0:
            sizes.append(total)
    return sizes


def preflight_batch_storage(
    output_root: Path,
    pending_sample_count: int,
    storage_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Estimate artifact growth and reserve enough free space before a batch.

    The estimate uses the nearest-rank P95 of recent complete sample directories
    when available.  A configured fallback is used only on a new installation.
    The result is intentionally conservative and contains no source image copies.
    """
    config = storage_config or {}
    pending = max(0, int(pending_sample_count))
    history_limit = max(1, int(config.get("history_sample_limit", 200)))
    percentile_ratio = min(1.0, max(0.5, float(config.get("estimate_percentile", 0.95))))
    multiplier = max(1.0, float(config.get("estimate_safety_multiplier", 1.5)))
    reserve_bytes = max(0, int(config.get("minimum_free_space_bytes", 2 * 1024**3)))
    fallback_bytes = max(1, int(config.get("fallback_bytes_per_sample", 5 * 1024**2)))

    observed = _sample_artifact_sizes(output_root, history_limit)
    if observed:
        ordered = sorted(observed)
        percentile_index = max(0, math.ceil(len(ordered) * percentile_ratio) - 1)
        baseline_bytes = ordered[percentile_index]
        estimate_method = f"historical_p{int(round(percentile_ratio * 100))}"
    else:
        baseline_bytes = fallback_bytes
        estimate_method = "configured_fallback"
    estimated_per_sample = max(1, math.ceil(baseline_bytes * multiplier))
    estimated_new_bytes = pending * estimated_per_sample

    filesystem_probe = _nearest_existing_path(output_root)
    disk = shutil.disk_usage(filesystem_probe)
    required_free_before_start = reserve_bytes + estimated_new_bytes
    sufficient = disk.free >= required_free_before_start
    return {
        "schema_version": "storage_preflight_v1",
        "status": "READY" if sufficient else "BLOCKED",
        "output_root": str(output_root.resolve()),
        "filesystem_probe": str(filesystem_probe),
        "pending_sample_count": pending,
        "observed_sample_directories": len(observed),
        "history_sample_limit": history_limit,
        "estimate_method": estimate_method,
        "baseline_bytes_per_sample": baseline_bytes,
        "estimate_safety_multiplier": multiplier,
        "estimated_bytes_per_sample": estimated_per_sample,
        "estimated_new_bytes": estimated_new_bytes,
        "minimum_free_space_bytes": reserve_bytes,
        "required_free_before_start": required_free_before_start,
        "available_free_bytes": disk.free,
        "available_total_bytes": disk.total,
        "estimated_free_after_batch": disk.free - estimated_new_bytes,
        "sufficient": sufficient,
    }
