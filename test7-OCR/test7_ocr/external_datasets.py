from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "external_datasets.json"
DATASET_REGISTRY_PATH = PROJECT_ROOT / "configs" / "datasets.json"
DOWNLOAD_ALLOWED_STATUS = frozenset(
    {"available", "available_research_only", "approved"}
)
EXTERNAL_EVAL_ROLES = frozenset(
    {
        "ocr_external_test",
        "formula_external_test",
        "photo_external_test",
        "restoration_external_test",
        "external_eval_only",
    }
)
UNIMER_STRATIFY_SEED = "test7-ocr-unimer-stratified400-v1"
UNIMER_BUCKETS = ("SPE", "CPE", "SCE", "HWE")
UNIMER_PER_BUCKET = 100


class LicenseBlockedError(RuntimeError):
    """Raised when an external dataset may not be downloaded or used for training."""


def load_external_config(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or EXTERNAL_CONFIG_PATH).read_text(encoding="utf-8"))


def load_dataset_registry(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or DATASET_REGISTRY_PATH).read_text(encoding="utf-8"))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def get_external_dataset(dataset_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = config or load_external_config()
    entry = (payload.get("datasets") or {}).get(dataset_id)
    if not isinstance(entry, dict):
        raise KeyError(f"Unknown external dataset: {dataset_id}")
    return entry


def license_allows_download(entry: dict[str, Any]) -> bool:
    return str(entry.get("license_status") or "") in DOWNLOAD_ALLOWED_STATUS


def assert_download_allowed(dataset_id: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = get_external_dataset(dataset_id, config)
    if not license_allows_download(entry):
        raise LicenseBlockedError(
            f"Download blocked for {dataset_id}: license_status="
            f"{entry.get('license_status')!r} is not in {sorted(DOWNLOAD_ALLOWED_STATUS)}"
        )
    if entry.get("training_allowed"):
        raise LicenseBlockedError(f"{dataset_id} must remain training_allowed=false")
    return entry


def is_external_eval_role(role: str | None) -> bool:
    return str(role or "") in EXTERNAL_EVAL_ROLES


def infer_dataset_role(
    dataset_root: Path,
    manifest: Path | None = None,
    *,
    explicit: str | None = None,
) -> str:
    if explicit:
        return explicit
    registry = load_dataset_registry()
    root = dataset_root.resolve()
    for entry in (registry.get("datasets") or {}).values():
        try:
            registered_root = resolve_project_path(entry["root"])
        except KeyError:
            continue
        if registered_root == root:
            role = str(entry.get("role") or "")
            if is_external_eval_role(role):
                return "external_eval_only"
            return role
    text = str((manifest or Path()).as_posix()).lower() + " " + str(root).lower()
    if any(token in text for token in ("omnidoc", "unimer", "mathvision", "warpdoc", "realdae", "external")):
        return "external_eval_only"
    return "internal"


def default_review_eligible_for_role(role: str | None) -> bool:
    return not is_external_eval_role(role) and role != "external_eval_only"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_unimer_sample(ids: list[str], bucket: str, limit: int = UNIMER_PER_BUCKET) -> list[str]:
    ranked = sorted(
        ids,
        key=lambda sample_id: hashlib.sha256(
            f"{UNIMER_STRATIFY_SEED}:{bucket}:{sample_id}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[: max(0, limit)]


def local_registry_roots_under(datasets_root: Path, registry: dict[str, Any] | None = None) -> set[Path]:
    payload = registry or load_dataset_registry()
    datasets_root = datasets_root.resolve()
    roots: set[Path] = set()
    for entry in (payload.get("datasets") or {}).values():
        root = resolve_project_path(entry["root"])
        try:
            root.relative_to(datasets_root)
        except ValueError:
            continue
        roots.add(root)
    return roots
