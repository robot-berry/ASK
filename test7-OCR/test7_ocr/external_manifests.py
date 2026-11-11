from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .external_datasets import (
    PROJECT_ROOT,
    UNIMER_BUCKETS,
    UNIMER_PER_BUCKET,
    resolve_project_path,
    sha256_file,
    stable_unimer_sample,
)
from .storage import write_jsonl_atomic


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
OMNIDOC_EXAM_SOURCES = {"exam_paper", "exam-paper", "exam"}
OMNIDOC_ZH = {"simplified_chinese", "zh", "chinese", "ch", "zh_cn"}


def _write_pair(manifest_path: Path, gold_path: Path | None, rows: list[dict[str, Any]], gold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    write_jsonl_atomic(manifest_path, rows)
    if gold_path is not None:
        write_jsonl_atomic(gold_path, gold_rows)
    return {
        "manifest": str(manifest_path),
        "gold": str(gold_path) if gold_path else None,
        "rows": len(rows),
        "training_allowed": False,
        "formal_gold_scoring": False,
    }


def _page_attribute(page: dict[str, Any]) -> dict[str, Any]:
    info = page.get("page_info") if isinstance(page.get("page_info"), dict) else {}
    for candidate in (
        info.get("page_attribute"),
        page.get("page_attribute"),
        page.get("attribute"),
        info.get("attribute"),
    ):
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _page_image_path(page: dict[str, Any], index: int) -> str:
    info = page.get("page_info") or {}
    for key in ("image_path", "img_path", "file_name", "filename"):
        value = info.get(key) or page.get(key)
        if value:
            return str(value).replace("\\", "/")
    page_no = info.get("page_no", index)
    return f"images/{page_no}.jpg"


def _layout_items(page: dict[str, Any]) -> list[dict[str, Any]]:
    items = page.get("layout_dets") or page.get("layout") or []
    return [item for item in items if isinstance(item, dict)]


def _item_category(item: dict[str, Any]) -> str:
    return str(item.get("category_type") or item.get("category") or item.get("type") or "").lower()


def omnidoc_gold_text(page: dict[str, Any]) -> tuple[str, list[str]]:
    chunks: list[str] = []
    formulas: list[str] = []
    for item in sorted(_layout_items(page), key=lambda row: int(row.get("order") or 10**9)):
        category = _item_category(item)
        latex = str(item.get("latex") or item.get("formula") or "").strip()
        text = str(item.get("text") or item.get("ocr_text") or "").strip()
        if "equation" in category or "formula" in category or latex:
            if latex:
                formulas.append(latex)
                chunks.append(latex)
            elif text:
                chunks.append(text)
            continue
        if text:
            chunks.append(text)
    return "\n".join(chunk for chunk in chunks if chunk), formulas


def prepare_omnidocbench_manifests(
    dataset_root: Path,
    *,
    annotation_name: str = "OmniDocBench.json",
    exam_manifest: Path | None = None,
    exam_gold: Path | None = None,
    smoke_manifest: Path | None = None,
    smoke_gold: Path | None = None,
    smoke_limit: int = 20,
) -> dict[str, Any]:
    annotation = dataset_root / annotation_name
    if not annotation.is_file():
        matches = list(dataset_root.rglob("OmniDocBench.json"))
        if not matches:
            raise FileNotFoundError(f"OmniDocBench.json not found under {dataset_root}")
        annotation = matches[0]
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    pages = payload if isinstance(payload, list) else payload.get("pages") or payload.get("data") or []
    exam_rows: list[dict[str, Any]] = []
    exam_gold_rows: list[dict[str, Any]] = []
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            continue
        attr = _page_attribute(page)
        source = str(attr.get("data_source") or attr.get("source") or "").lower()
        language = str(attr.get("language") or attr.get("lang") or "").lower()
        if source not in OMNIDOC_EXAM_SOURCES:
            continue
        if language and language not in OMNIDOC_ZH and "chinese" not in language:
            continue
        relative = _page_image_path(page, index)
        image_path = dataset_root / relative
        if not image_path.is_file():
            name = Path(relative).name
            found = next((path for path in dataset_root.rglob(name) if path.is_file()), None)
            if found is None:
                continue
            relative = found.relative_to(dataset_root).as_posix()
            image_path = found
        sample_id = f"ODB-EXAM-ZH-{index:04d}"
        gold_text, formulas = omnidoc_gold_text(page)
        exam_rows.append(
            {
                "schema_version": "external_eval_manifest_v1",
                "sample_id": sample_id,
                "dataset_id": "omnidocbench",
                "split": "test",
                "bucket": "exam_paper_simplified_chinese",
                "image_path": relative,
                "image_sha256": sha256_file(image_path),
                "include_in_benchmark": True,
                "training_allowed": False,
                "evaluation_only": True,
            }
        )
        exam_gold_rows.append(
            {
                "sample_id": sample_id,
                "scoring_mode": "page_text_and_formulas",
                "gold_text": gold_text,
                "gold_formulas": formulas,
                "ignore_answer": True,
                "text_cer_applicable": True,
            }
        )
    exam_manifest = exam_manifest or (PROJECT_ROOT / "data" / "splits" / "omnidocbench_exam_zh.jsonl")
    exam_gold = exam_gold or (PROJECT_ROOT / "data" / "splits" / "omnidocbench_exam_zh_gold.jsonl")
    smoke_manifest = smoke_manifest or (PROJECT_ROOT / "data" / "splits" / "omnidocbench_exam_zh_smoke20.jsonl")
    smoke_gold = smoke_gold or (PROJECT_ROOT / "data" / "splits" / "omnidocbench_exam_zh_smoke20_gold.jsonl")
    summary = {
        "annotation": str(annotation),
        "annotation_sha256": sha256_file(annotation),
        "exam_zh": _write_pair(exam_manifest, exam_gold, exam_rows, exam_gold_rows),
        "smoke20": _write_pair(
            smoke_manifest,
            smoke_gold,
            exam_rows[:smoke_limit],
            exam_gold_rows[:smoke_limit],
        ),
    }
    return summary


def _guess_unimer_bucket(path: Path) -> str | None:
    text = path.as_posix().upper()
    for bucket in UNIMER_BUCKETS:
        if re.search(rf"(?:^|/){bucket}(?:/|_)", text):
            return bucket
    lower = path.as_posix().lower()
    if "simple" in lower and "print" in lower:
        return "SPE"
    if "complex" in lower and "print" in lower:
        return "CPE"
    if "screen" in lower:
        return "SCE"
    if "hand" in lower:
        return "HWE"
    return None


def _read_latex_sidecar(image_path: Path) -> str:
    for suffix in (".txt", ".tex", ".latex"):
        sidecar = image_path.with_suffix(suffix)
        if sidecar.is_file():
            return sidecar.read_text(encoding="utf-8").strip()
    return ""


def _load_unimer_list_labels(dataset_root: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {bucket: {} for bucket in UNIMER_BUCKETS}
    for labels_path in dataset_root.rglob("*.txt"):
        bucket = _guess_unimer_bucket(labels_path)
        if bucket is None or labels_path.stem.lower() not in {bucket.lower(), labels_path.stem.lower()}:
            if labels_path.stem.upper() not in UNIMER_BUCKETS:
                continue
            bucket = labels_path.stem.upper()
        folder = labels_path.with_suffix("")
        if not folder.is_dir():
            sibling = labels_path.parent / labels_path.stem
            folder = sibling if sibling.is_dir() else labels_path.parent / bucket.lower()
        if not folder.is_dir():
            continue
        lines = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines()]
        images = sorted(
            path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            continue
        indexed = 0
        for image_path in images:
            try:
                line_index = int(image_path.stem)
            except ValueError:
                continue
            if 0 <= line_index < len(lines) and lines[line_index]:
                labels[bucket][image_path.name] = lines[line_index]
                indexed += 1
        if indexed:
            continue
        usable = min(len(images), len(lines))
        for image_path, latex in zip(images[:usable], lines[:usable]):
            if latex:
                labels[bucket][image_path.name] = latex
    return labels


def prepare_unimer_manifests(
    dataset_root: Path,
    *,
    per_bucket: int = UNIMER_PER_BUCKET,
    manifest: Path | None = None,
    gold: Path | None = None,
    smoke_manifest: Path | None = None,
    smoke_gold: Path | None = None,
) -> dict[str, Any]:
    by_bucket: dict[str, list[tuple[str, Path, str]]] = {bucket: [] for bucket in UNIMER_BUCKETS}
    label_maps: dict[str, str] = {}
    for labels_path in dataset_root.rglob("*"):
        if labels_path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if "unimer" not in labels_path.name.lower() and labels_path.name.lower() not in {
            "labels.json",
            "test.json",
            "test.jsonl",
        }:
            continue
        try:
            raw = labels_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if labels_path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            payload = json.loads(raw)
            rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if not isinstance(row, dict):
                continue
            image = str(row.get("image") or row.get("img") or row.get("file") or "")
            latex = str(row.get("latex") or row.get("label") or row.get("formula") or "")
            if image and latex:
                label_maps[Path(image).name] = latex

    list_labels = _load_unimer_list_labels(dataset_root)
    for image_path in sorted(path for path in dataset_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        bucket = _guess_unimer_bucket(image_path)
        if bucket is None:
            continue
        latex = (
            label_maps.get(image_path.name)
            or list_labels.get(bucket, {}).get(image_path.name)
            or _read_latex_sidecar(image_path)
        )
        sample_id = f"UNIMER-{bucket}-{image_path.stem}"
        by_bucket[bucket].append((sample_id, image_path, latex))

    selected_rows: list[dict[str, Any]] = []
    selected_gold: list[dict[str, Any]] = []
    bucket_counts = {}
    for bucket, items in by_bucket.items():
        chosen_ids = set(stable_unimer_sample([item[0] for item in items], bucket, per_bucket))
        bucket_counts[bucket] = {"available": len(items), "selected": len(chosen_ids)}
        for sample_id, image_path, latex in items:
            if sample_id not in chosen_ids:
                continue
            relative = image_path.relative_to(dataset_root).as_posix()
            selected_rows.append(
                {
                    "schema_version": "external_eval_manifest_v1",
                    "sample_id": sample_id,
                    "dataset_id": "unimer_test",
                    "split": "test",
                    "bucket": bucket,
                    "image_path": relative,
                    "image_sha256": sha256_file(image_path),
                    "include_in_benchmark": True,
                    "training_allowed": False,
                    "evaluation_only": True,
                    "scoring_mode": "formula_only",
                }
            )
            selected_gold.append(
                {
                    "sample_id": sample_id,
                    "scoring_mode": "formula_only",
                    "gold_text": latex,
                    "gold_formulas": [latex] if latex else [],
                    "ignore_answer": True,
                    "text_cer_applicable": True,
                }
            )
    manifest = manifest or (PROJECT_ROOT / "data" / "splits" / "unimer_test_stratified400.jsonl")
    gold = gold or (PROJECT_ROOT / "data" / "splits" / "unimer_test_stratified400_gold.jsonl")
    smoke_manifest = smoke_manifest or (PROJECT_ROOT / "data" / "splits" / "unimer_test_smoke20.jsonl")
    smoke_gold = smoke_gold or (PROJECT_ROOT / "data" / "splits" / "unimer_test_smoke20_gold.jsonl")
    return {
        "buckets": bucket_counts,
        "stratified": _write_pair(manifest, gold, selected_rows, selected_gold),
        "smoke20": _write_pair(
            smoke_manifest,
            smoke_gold,
            selected_rows[:20],
            selected_gold[:20],
        ),
    }


def _save_mathvision_image(image_field: Any, destination: Path) -> Path | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_field, dict):
        raw = image_field.get("bytes") or image_field.get("path")
        if isinstance(raw, (bytes, bytearray)):
            destination.write_bytes(bytes(raw))
            return destination
        if isinstance(raw, str) and Path(raw).is_file():
            shutil.copy2(raw, destination)
            return destination
        nested = image_field.get("image")
        if nested is not None and nested is not image_field:
            return _save_mathvision_image(nested, destination)
    if hasattr(image_field, "save"):
        image_field.save(destination)
        return destination
    if isinstance(image_field, (bytes, bytearray)):
        destination.write_bytes(bytes(image_field))
        return destination
    return None


def _iter_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to convert MathVision-Wild parquet files") from exc
    table = pq.read_table(path)
    return table.to_pylist()


def _mathvision_split_from_path(path: Path) -> str:
    text = path.as_posix().lower()
    if "testmini_photo" in text:
        return "testmini_photo"
    if "test_photo" in text:
        return "test_photo"
    if "testmini" in text:
        return "testmini_photo"
    return "test_photo"


def _collect_mathvision_records(dataset_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    json_files = [
        path
        for path in dataset_root.rglob("*")
        if path.suffix.lower() in {".json", ".jsonl"}
        and path.name.lower() != "fetch_receipt.json"
        and "screenshot" not in path.as_posix().lower()
    ]
    for path in sorted(json_files):
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            payload = json.loads(raw)
            if isinstance(payload, dict) and not (payload.get("id") or payload.get("sample_id")):
                rows = payload.get("data") or payload.get("rows") or []
            else:
                rows = payload if isinstance(payload, list) else [payload]
        default_split = _mathvision_split_from_path(path)
        source_dir = path.parent
        for row in rows:
            if isinstance(row, dict) and (row.get("id") or row.get("sample_id")):
                row = dict(row)
                row.setdefault("split", default_split)
                row["_source_dir"] = str(source_dir)
                records.append(row)
    parquet_files = sorted(path for path in dataset_root.rglob("*.parquet"))
    export_root = dataset_root / "images"
    for path in parquet_files:
        split = _mathvision_split_from_path(path)
        for index, row in enumerate(_iter_parquet_rows(path)):
            if not isinstance(row, dict):
                continue
            sample_id = str(row.get("id") or row.get("sample_id") or f"{split}-{index}")
            image_rel = f"images/{split}/{sample_id}.jpg"
            saved = export_root / split / f"{sample_id}.jpg"
            image_field = row.get("image") or row.get("photo") or row.get("decoded")
            if image_field is not None and not saved.is_file():
                png_path = saved.with_suffix(".png")
                written = _save_mathvision_image(image_field, saved) or _save_mathvision_image(image_field, png_path)
                if written is not None:
                    image_rel = written.relative_to(dataset_root).as_posix()
            records.append(
                {
                    "id": sample_id,
                    "split": str(row.get("split") or row.get("subset") or split),
                    "image": image_rel,
                    "question": row.get("question") or row.get("query") or "",
                    "options": row.get("options") or row.get("choices") or [],
                    "answer": row.get("answer"),
                }
            )
    return records


def _mathvision_question_map(dataset_root: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    search_roots = [dataset_root, dataset_root / "upstream_questions"]
    for root in search_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.parquet"):
            if "mathvision_wild" in path.as_posix().lower() and "upstream" not in path.as_posix().lower():
                continue
            try:
                rows = _iter_parquet_rows(path)
            except Exception:
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sample_id = str(row.get("id") or row.get("sample_id") or "")
                if not sample_id:
                    continue
                options = row.get("options") or row.get("choices") or []
                mapping[sample_id] = {
                    "question": str(row.get("question") or row.get("query") or ""),
                    "options": options,
                }
    return mapping


def prepare_mathvision_wild_manifests(
    dataset_root: Path,
    *,
    frozen_manifest: Path | None = None,
    frozen_gold: Path | None = None,
    mini_manifest: Path | None = None,
    mini_gold: Path | None = None,
) -> dict[str, Any]:
    records = _collect_mathvision_records(dataset_root)
    if not records:
        raise FileNotFoundError(f"No MathVision annotations found under {dataset_root}")

    mini_ids = {
        str(row.get("id") or row.get("sample_id"))
        for row in records
        if str(row.get("split") or row.get("subset") or "").lower() in {
            "testmini",
            "testmini_photo",
            "mini",
            "testmini_screenshot",
        }
        or bool(row.get("mini"))
    }
    frozen_rows: list[dict[str, Any]] = []
    frozen_gold_rows: list[dict[str, Any]] = []
    mini_rows: list[dict[str, Any]] = []
    mini_gold_rows: list[dict[str, Any]] = []
    question_map = _mathvision_question_map(dataset_root)
    for row in records:
        sample_id = str(row.get("id") or row.get("sample_id"))
        split = str(row.get("split") or row.get("subset") or "test_photo")
        image_field = (
            row.get("image")
            or row.get("image_path")
            or row.get("file_name")
            or row.get("photo")
            or f"images/{sample_id}.png"
        )
        if isinstance(image_field, dict):
            image_field = image_field.get("path") or image_field.get("src") or image_field.get("file_name") or ""
        relative = str(image_field).replace("\\", "/")
        image_path = dataset_root / relative
        source_dir = Path(str(row.get("_source_dir") or ""))
        if not image_path.is_file() and source_dir.is_dir():
            nested = source_dir / Path(relative).name
            if nested.is_file():
                image_path = nested
                relative = nested.relative_to(dataset_root).as_posix()
        if not image_path.is_file():
            found = next((path for path in dataset_root.rglob(Path(relative).name) if path.is_file()), None)
            if found is None:
                continue
            relative = found.relative_to(dataset_root).as_posix()
            image_path = found
        mapped = question_map.get(sample_id) or question_map.get(str(row.get("id") or ""))
        question = str(row.get("question") or row.get("query") or (mapped or {}).get("question") or "")
        options = row.get("options") or row.get("choices") or (mapped or {}).get("options") or []
        if isinstance(options, dict):
            options = [str(options[key]) for key in sorted(options)]
        elif isinstance(options, list):
            options = [str(item) for item in options]
        else:
            options = []
        gold_text = "\n".join([question, *options]).strip()
        record = {
            "schema_version": "external_eval_manifest_v1",
            "sample_id": f"MVW-{sample_id}",
            "dataset_id": "mathvision_wild",
            "split": split,
            "bucket": "mini" if sample_id in mini_ids else "frozen_test_photo",
            "image_path": relative,
            "image_sha256": sha256_file(image_path),
            "include_in_benchmark": sample_id not in mini_ids,
            "training_allowed": False,
            "evaluation_only": True,
        }
        gold = {
            "sample_id": f"MVW-{sample_id}",
            "scoring_mode": "question_and_options",
            "gold_text": gold_text,
            "gold_question": question,
            "gold_options": options,
            "ignore_answer": True,
            "text_cer_applicable": True,
        }
        if sample_id in mini_ids or str(split).lower().startswith("testmini"):
            mini_rows.append({**record, "include_in_benchmark": True, "bucket": "mini"})
            mini_gold_rows.append(gold)
        if sample_id not in mini_ids:
            frozen_rows.append(record)
            frozen_gold_rows.append(gold)

    frozen_manifest = frozen_manifest or (PROJECT_ROOT / "data" / "splits" / "mathvision_wild_frozen.jsonl")
    frozen_gold = frozen_gold or (PROJECT_ROOT / "data" / "splits" / "mathvision_wild_frozen_gold.jsonl")
    mini_manifest = mini_manifest or (PROJECT_ROOT / "data" / "splits" / "mathvision_wild_mini.jsonl")
    mini_gold = mini_gold or (PROJECT_ROOT / "data" / "splits" / "mathvision_wild_mini_gold.jsonl")
    return {
        "mini": _write_pair(mini_manifest, mini_gold, mini_rows, mini_gold_rows),
        "frozen": _write_pair(frozen_manifest, frozen_gold, frozen_rows, frozen_gold_rows),
    }


def prepare_restoration_manifest(
    dataset_root: Path,
    dataset_id: str,
    *,
    limit: int = 50,
    manifest: Path | None = None,
) -> dict[str, Any]:
    images = sorted(path for path in dataset_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    rows = []
    for index, image_path in enumerate(images[:limit]):
        rows.append(
            {
                "schema_version": "external_eval_manifest_v1",
                "sample_id": f"{dataset_id.upper()}-{index:03d}-{image_path.stem}",
                "dataset_id": dataset_id,
                "split": "test",
                "bucket": "stability50",
                "image_path": image_path.relative_to(dataset_root).as_posix(),
                "image_sha256": sha256_file(image_path),
                "include_in_benchmark": True,
                "training_allowed": False,
                "evaluation_only": True,
            }
        )
    default_name = f"{dataset_id}_stability50.jsonl"
    manifest = manifest or (PROJECT_ROOT / "data" / "splits" / default_name)
    return _write_pair(manifest, None, rows, [])
