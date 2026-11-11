from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

from .capture_pair_review import CapturePairStore, payload_hash as capture_pair_payload_hash
from .collection_workspace import (
    COLLECTION_WORKSPACE_VERSION,
    READINESS_FILENAME,
    prepare_collection_workspace,
    register_guided_collection_image,
)
from .clean_source import validate_source_resolution
from .collection_plan import collection_plan_csv
from .handwriting_review import HandwritingMaskStore, payload_hash
from .output_quality import OUTPUT_ISSUE_KEYS, validate_output_quality_review
from .product_inference import PRODUCT_API_VERSION, ProductOcrService
from .question_boundary_review import QuestionBoundaryStore, payload_hash as boundary_payload_hash
from .self_built_intake import SelfBuiltIntakeStore
from .storage import write_json_atomic
from .text import latex_to_readable, parse_question


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_ROOT = PROJECT_ROOT / "review_ui"
PRODUCT_UI_ROOT = PROJECT_ROOT / "product_ui"
TASK_ID = re.compile(r"^[0-9A-Za-z._-]+$")
REVIEW_API_VERSION = "review_workstation_v32"
REVIEW_UI_VERSION = "review_ui_v34"
GOLD_CHECK_KEYS = {
    "text_exact",
    "formula_exact",
    "options_complete",
    "visual_assets_complete",
    "handwriting_excluded",
    "reading_order_correct",
}


def _find_review_katex_root() -> Path | None:
    candidates = (
        PROJECT_ROOT / "vendor" / "katex",
        PROJECT_ROOT / "node_modules" / "katex" / "dist",
        PROJECT_ROOT.parents[1] / "01" / "web" / "static" / "katex",
    )
    for candidate in candidates:
        if (candidate / "katex.min.js").is_file() and (candidate / "katex.min.css").is_file():
            return candidate.resolve()
    return None


KATEX_ROOT = _find_review_katex_root()


def json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ReviewStore:
    def __init__(
        self,
        review_dir: Path,
        dataset_root: Path,
        runs_root: Path | None = None,
        output_review_root: Path | None = None,
        mask_task_dir: Path | None = None,
        mask_dir: Path | None = None,
        intake_dataset_root: Path | None = None,
        intake_record_dir: Path | None = None,
        intake_summary_path: Path | None = None,
        boundary_task_dir: Path | None = None,
        boundary_manifest_path: Path | None = None,
        boundary_summary_path: Path | None = None,
        capture_pair_task_dir: Path | None = None,
        capture_pair_manifest_path: Path | None = None,
        capture_pair_summary_path: Path | None = None,
        mask_dataset_root: Path | None = None,
        auxiliary_review_dir: Path | None = None,
        auxiliary_dataset_root: Path | None = None,
    ) -> None:
        self.review_dir = review_dir.resolve()
        self.dataset_root = dataset_root.resolve()
        if (auxiliary_review_dir is None) != (auxiliary_dataset_root is None):
            raise ValueError(
                "Auxiliary review directory and dataset root must be configured together"
            )
        self.auxiliary_review_dir = (
            auxiliary_review_dir.resolve() if auxiliary_review_dir is not None else None
        )
        self.auxiliary_dataset_root = (
            auxiliary_dataset_root.resolve() if auxiliary_dataset_root is not None else None
        )
        self.runs_root = (runs_root or PROJECT_ROOT / "artifacts" / "runs").resolve()
        self.output_review_root = (
            output_review_root or PROJECT_ROOT / "data" / "annotations" / "output_quality_reviews"
        ).resolve()
        self._decoded_output_hash_cache: dict[tuple[str, int, int], str] = {}
        self.self_built_intake = SelfBuiltIntakeStore(
            intake_dataset_root
            or PROJECT_ROOT.parents[1] / "datasets" / "test7_ocr_self_built_v1",
            intake_record_dir
            or PROJECT_ROOT / "data" / "annotations" / "self_built_intake" / "records",
            intake_summary_path
            or PROJECT_ROOT / "data" / "splits" / "self_built_intake_summary.json",
        )
        mask_root = PROJECT_ROOT / "data" / "annotations" / "handwriting_masks"
        self.handwriting_masks = HandwritingMaskStore(
            mask_task_dir or mask_root / "tasks",
            mask_dir or mask_root / "masks",
            mask_dataset_root or self.dataset_root,
            self.self_built_intake,
        )
        self.question_boundaries = QuestionBoundaryStore(
            boundary_task_dir
            or PROJECT_ROOT / "data" / "annotations" / "question_boundaries" / "tasks",
            boundary_manifest_path
            or PROJECT_ROOT / "data" / "splits" / "question_boundary_manifest.jsonl",
            boundary_summary_path
            or PROJECT_ROOT / "data" / "splits" / "question_boundary_summary.json",
            self.self_built_intake,
        )
        self.capture_pairs = CapturePairStore(
            capture_pair_task_dir
            or PROJECT_ROOT / "data" / "annotations" / "capture_pairs" / "tasks",
            capture_pair_manifest_path
            or PROJECT_ROOT / "data" / "splits" / "capture_pair_manifest.jsonl",
            capture_pair_summary_path
            or PROJECT_ROOT / "data" / "splits" / "capture_pair_summary.json",
            self.self_built_intake,
        )

    @staticmethod
    def _public_collection_workspace(report: dict) -> dict:
        public = {
            key: value
            for key, value in report.items()
            if key
            not in {
                "missing_rows",
                "invalid_rows",
                "incremental_ready_rows",
                "blocked_by_clean_dependency",
            }
        }
        public["missing_rows_preview"] = list(report.get("missing_rows") or [])[:10]
        public["invalid_rows_preview"] = list(report.get("invalid_rows") or [])[:10]
        public["blocked_by_clean_dependency_preview"] = list(
            report.get("blocked_by_clean_dependency") or []
        )[:10]
        return public

    def collection_workspace_summary(self, *, refresh: bool = False) -> dict:
        workspace_root = self.self_built_intake.dataset_root / "collection"
        readiness_path = workspace_root / READINESS_FILENAME
        if refresh:
            report = prepare_collection_workspace(workspace_root)
        elif readiness_path.is_file():
            report = json.loads(readiness_path.read_text(encoding="utf-8"))
            guided_cache_fields = {
                "guided_registration_count",
                "next_guided_row",
                "available_guided_rows",
            }
            if not guided_cache_fields.issubset(report):
                report = prepare_collection_workspace(workspace_root)
        else:
            report = {
                "schema_version": COLLECTION_WORKSPACE_VERSION,
                "workspace_root": str(workspace_root.resolve()),
                "working_plan": str((workspace_root / "self_built_collection_plan.csv").resolve()),
                "planned_rows": 410,
                "files_present": 0,
                "files_missing": 410,
                "files_invalid": 0,
                "privacy_checked": 0,
                "personal_data_clear": 0,
                "distribution_declared": 0,
                "role_progress": {},
                "structural_errors": ["private collection workspace has not been prepared"],
                "missing_rows": [],
                "invalid_rows": [],
                "ready_for_import_preflight": False,
                "training_allowed": False,
                "evaluation_allowed": False,
                "source_images_enter_git": False,
                "next_actions": ["初始化私有采集工作区并按清单放入真实图片。"],
            }
        if report.get("schema_version") != COLLECTION_WORKSPACE_VERSION:
            raise ValueError("Unsupported private collection workspace report")
        return self._public_collection_workspace(report)

    @staticmethod
    def _validate_id(value: str, label: str) -> str:
        if not TASK_ID.fullmatch(value):
            raise ValueError(f"Invalid {label}")
        return value

    def _task_path(self, sample_id: str) -> Path:
        self._validate_id(sample_id, "sample id")
        path = (self.review_dir / f"{sample_id}.json").resolve()
        if path.parent != self.review_dir:
            raise ValueError("Task path escaped review directory")
        return path

    def _run_path(self, run_id: str) -> Path:
        self._validate_id(run_id, "run id")
        path = (self.runs_root / run_id).resolve()
        if path.parent != self.runs_root:
            raise ValueError("Run path escaped runs directory")
        if not path.is_dir():
            raise FileNotFoundError(run_id)
        return path

    def _output_review_path(self, run_id: str, sample_id: str) -> Path:
        self._validate_id(run_id, "run id")
        self._validate_id(sample_id, "sample id")
        run_dir = (self.output_review_root / run_id).resolve()
        if run_dir.parent != self.output_review_root:
            raise ValueError("Output review path escaped review directory")
        return run_dir / f"{sample_id}.json"

    def load(self, sample_id: str) -> dict:
        path = self._task_path(sample_id)
        if not path.exists():
            raise FileNotFoundError(sample_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_output_task(self, sample_id: str) -> tuple[dict, Path]:
        """Resolve output-review metadata without changing the primary Gold queue."""
        self._validate_id(sample_id, "sample id")
        candidates = [(self.review_dir, self.dataset_root)]
        if self.auxiliary_review_dir is not None and self.auxiliary_dataset_root is not None:
            candidates.append((self.auxiliary_review_dir, self.auxiliary_dataset_root))
        for review_dir, dataset_root in candidates:
            path = (review_dir / f"{sample_id}.json").resolve()
            if path.parent != review_dir:
                raise ValueError("Task path escaped review directory")
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8")), dataset_root
        raise FileNotFoundError(sample_id)

    def candidate_structure(self, sample_id: str) -> dict:
        task = self.load(sample_id)
        transcript = str(task.get("candidate_prelabels", {}).get("suggested_transcript") or "")
        question_number, stem, options = parse_question(transcript)
        stem = re.sub(r"\n\$\s*$", "", stem).strip()
        cleaned_options: list[dict[str, str]] = []
        for option in options:
            text = str(option.get("text") or "")
            text = re.sub(r"\$\$\s+\$\s*$", "$", text)
            text = re.sub(r"\$\$\s*$", "$", text).strip()
            cleaned_options.append({"label": str(option["label"]), "text": text})
        lines = [stem] if stem else []
        lines.extend(f"{option['label']}. {option['text']}" for option in cleaned_options)
        return {
            "display_transcript": latex_to_readable(transcript),
            "question_number": question_number,
            "question_type": "choice" if len(cleaned_options) >= 2 else "",
            "stem_text": stem,
            "options": cleaned_options,
            "canonical_transcript": "\n".join(lines),
        }

    def list(self) -> list[dict]:
        tasks: list[dict] = []
        for path in sorted(self.review_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            tasks.append(
                {
                    "sample_id": payload["sample_id"],
                    "subject": payload["source"]["subject"],
                    "priority": payload["review"]["priority"],
                    "status": payload["review"]["status"],
                    "include_in_benchmark": payload["source"]["include_in_benchmark"],
                    "question_number": payload["gold_annotation"].get("question_number"),
                }
            )
        return tasks

    def source_path(self, sample_id: str) -> Path:
        task, dataset_root = self._load_output_task(sample_id)
        path = (dataset_root / task["source"]["image_path"]).resolve()
        if dataset_root not in path.parents:
            raise ValueError("Source path escaped dataset root")
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != task["source"]["image_sha256"]:
            raise ValueError("Source image hash mismatch")
        return path

    def _output_context(self, run_id: str, sample_id: str) -> tuple[Path, dict, dict]:
        run_path = self._run_path(run_id)
        self._validate_id(sample_id, "sample id")
        sample_path = (run_path / sample_id).resolve()
        if sample_path.parent != run_path:
            raise ValueError("Output path escaped run directory")
        report_path = sample_path / "report.json"
        output_path = sample_path / "output.png"
        if not report_path.is_file() or not output_path.is_file():
            raise FileNotFoundError(f"{run_id}/{sample_id}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_id") != run_id or report.get("sample_id") != sample_id:
            raise ValueError("Output report identity mismatch")
        task, _ = self._load_output_task(sample_id)
        gold_source_sha256 = task["source"]["image_sha256"]
        source_resolution = report.get("source_resolution")
        if source_resolution is not None:
            validate_source_resolution(source_resolution)
            expected_report_fields = {
                "input_source_image": source_resolution["input_image"],
                "input_source_sha256": source_resolution["input_sha256"],
                "source_image": source_resolution["effective_image"],
                "source_sha256": source_resolution["effective_sha256"],
            }
            for field, expected in expected_report_fields.items():
                if report.get(field) != expected:
                    raise ValueError(f"Output report {field} does not match source resolution")
            if source_resolution["input_sha256"] != gold_source_sha256:
                raise ValueError("Output report input hash does not match the Gold task")
        elif report.get("source_sha256") != gold_source_sha256:
            raise ValueError("Output report source hash does not match the Gold task")
        output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        if report.get("output_sha256") != output_sha256:
            raise ValueError("Output image hash mismatch")
        with Image.open(output_path) as opened:
            output_format = opened.format
            output_width, output_height = opened.size
        if output_format != "PNG":
            raise ValueError("Final output is not a PNG image")
        if (
            report.get("output_width") != output_width
            or report.get("output_height") != output_height
        ):
            raise ValueError("Output image dimensions do not match the report")
        return output_path, report, task

    def output_path(self, run_id: str, sample_id: str) -> Path:
        return self._output_context(run_id, sample_id)[0]

    def list_runs(self) -> list[dict]:
        if not self.runs_root.is_dir():
            return []
        runs: list[dict] = []
        for run_path in self.runs_root.iterdir():
            if not run_path.is_dir() or not TASK_ID.fullmatch(run_path.name):
                continue
            reports = list(run_path.glob("*/report.json"))
            output_count = sum(1 for path in reports if (path.parent / "output.png").is_file())
            if output_count == 0:
                continue
            review_dir = self.output_review_root / run_path.name
            reviews = []
            if review_dir.is_dir():
                for path in review_dir.glob("*.json"):
                    try:
                        reviews.append(json.loads(path.read_text(encoding="utf-8")))
                    except (OSError, json.JSONDecodeError):
                        continue
            modified_epoch = run_path.stat().st_mtime
            batch_summary = {}
            try:
                batch_summary = json.loads(
                    (run_path / "batch_summary.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                pass
            attempted = int(batch_summary.get("attempted") or 0)
            latency_summary = batch_summary.get("latency_seconds") or {}
            under_target = int(latency_summary.get("under_target_count") or 0)
            p95_latency = latency_summary.get("p95")
            latency_target = float(latency_summary.get("target") or 5.0)
            latency_sla_pass = (
                float(p95_latency) <= latency_target
                if p95_latency is not None
                else under_target == attempted
            )
            failures = batch_summary.get("failures")
            resource_guard_status = (batch_summary.get("resource_guard") or {}).get("status")
            latency_guard_status = (batch_summary.get("latency_guard") or {}).get("status")
            measurement_mode = str(batch_summary.get("mode") or "end_to_end")
            online_latency_sla_applicable = measurement_mode != "render_only"
            dataset_role = str(batch_summary.get("dataset_role") or "")
            default_review_flag = batch_summary.get("default_review_eligible")
            default_eligible = bool(
                (run_path / "index.finalized.jsonl").is_file()
                and attempted == output_count
                and attempted > 0
                and latency_sla_pass
                and isinstance(failures, list)
                and not failures
                and resource_guard_status in {None, "CLEAR", "DISABLED"}
                and latency_guard_status in {None, "CLEAR", "DISABLED"}
                and online_latency_sla_applicable
                and dataset_role not in {
                    "external_eval_only",
                    "ocr_external_test",
                    "formula_external_test",
                    "photo_external_test",
                    "restoration_external_test",
                }
                and default_review_flag is not False
            )
            runs.append(
                {
                    "run_id": run_path.name,
                    "output_count": output_count,
                    "finalized": (run_path / "index.finalized.jsonl").is_file(),
                    "default_eligible": default_eligible,
                    "measurement_mode": measurement_mode,
                    "online_latency_sla_applicable": online_latency_sla_applicable,
                    "latency_guard_status": latency_guard_status,
                    "modified_at": datetime.fromtimestamp(modified_epoch, timezone.utc).isoformat(),
                    "_modified_epoch": modified_epoch,
                    "human_pass": sum(1 for item in reviews if item.get("verdict") == "pass"),
                    "human_fail": sum(1 for item in reviews if item.get("verdict") == "fail"),
                }
            )
        ordered = sorted(
            runs,
            key=lambda item: (
                not item["default_eligible"],
                not item["finalized"],
                -item["output_count"],
                -item["_modified_epoch"],
                item["run_id"],
            ),
        )
        for item in ordered:
            item.pop("_modified_epoch", None)
        return ordered

    def list_output_items(self, run_id: str) -> list[dict]:
        run_path = self._run_path(run_id)
        items: list[dict] = []
        for report_path in sorted(run_path.glob("*/report.json")):
            sample_id = report_path.parent.name
            if not TASK_ID.fullmatch(sample_id) or not (report_path.parent / "output.png").is_file():
                continue
            try:
                _, report, task = self._output_context(run_id, sample_id)
            except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
                continue
            review = self._load_output_review_file(run_id, sample_id)
            previous_reviews = self.output_review_history(
                run_id, sample_id, report, limit=1000
            )
            previous_failures = [
                item for item in previous_reviews if item.get("verdict") == "fail"
            ]
            inherited_review = (
                None
                if review is not None
                else self._inherited_output_review(previous_reviews)
            )
            human_verdict = (
                review.get("verdict", "pending")
                if review is not None
                else (inherited_review or {}).get("verdict", "pending")
            )
            measurement = report.get("measurement") or {}
            items.append(
                {
                    "sample_id": sample_id,
                    "subject": task["source"]["subject"],
                    "question_number": task["gold_annotation"].get("question_number"),
                    "pipeline_status": report.get("status"),
                    "elapsed_seconds": report.get("elapsed_seconds"),
                    "measurement_mode": str(measurement.get("mode") or "end_to_end"),
                    "online_latency_sla_applicable": (
                        measurement.get("online_latency_sla_applicable") is not False
                    ),
                    "output_width": report.get("output_width"),
                    "output_height": report.get("output_height"),
                    "output_sha256": report.get("output_sha256"),
                    "human_verdict": human_verdict,
                    "human_review_origin": (
                        "current"
                        if review is not None
                        else "inherited"
                        if inherited_review is not None
                        else "pending"
                    ),
                    "human_review_inherited": inherited_review is not None,
                    "inherited_from_run": (
                        inherited_review.get("run_id") if inherited_review else None
                    ),
                    "previous_review_count": len(previous_reviews),
                    "previous_fail_count": len(previous_failures),
                    "same_output_previous_fail": bool(
                        inherited_review
                        and inherited_review.get("verdict") == "fail"
                        and inherited_review.get("same_output")
                    ),
                    "same_failure_evidence_previous_fail": bool(
                        inherited_review
                        and inherited_review.get("verdict") == "fail"
                        and inherited_review.get("same_failure_evidence")
                    ),
                    "clean_source_substituted": bool(
                        (report.get("source_resolution") or {}).get("substituted")
                    ),
                    "issue_codes": [
                        issue.get("code")
                        for issue in report.get("issues", [])
                        if isinstance(issue, dict) and issue.get("code")
                    ],
                }
            )
        return items

    def _default_output_review(self, run_id: str, sample_id: str, report: dict) -> dict:
        return {
            "schema_version": "output_quality_review_v1",
            "run_id": run_id,
            "sample_id": sample_id,
            "source_sha256": report["source_sha256"],
            "output_sha256": report["output_sha256"],
            "verdict": "pending",
            "reviewer_id": None,
            "issues": {key: None for key in OUTPUT_ISSUE_KEYS},
            "notes": [],
            "history": [],
            "updated_at_utc": None,
        }

    def _load_output_review_file(self, run_id: str, sample_id: str) -> dict | None:
        path = self._output_review_path(run_id, sample_id)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _decoded_output_hash(self, path: Path) -> str:
        """Hash displayed RGB pixels, excluding non-visual PNG encoding metadata."""

        stat = path.stat()
        key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        cached = self._decoded_output_hash_cache.get(key)
        if cached is not None:
            return cached
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            digest = hashlib.sha256()
            digest.update(f"{image.width}x{image.height}:RGB\0".encode("ascii"))
            digest.update(image.tobytes())
        value = digest.hexdigest()
        if len(self._decoded_output_hash_cache) >= 4096:
            self._decoded_output_hash_cache.clear()
        self._decoded_output_hash_cache[key] = value
        return value

    def _output_document(self, run_id: str, sample_id: str) -> dict | None:
        run_path = self._run_path(run_id)
        self._validate_id(sample_id, "sample id")
        sample_path = (run_path / sample_id).resolve()
        if sample_path.parent != run_path:
            raise ValueError("Output document path escaped run directory")
        path = sample_path / "question_document.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("sample_id") != sample_id:
            raise ValueError("Output document identity mismatch")
        return payload

    @staticmethod
    def _failure_evidence_unchanged(
        reviewed: dict | None,
        current: dict | None,
        issue_codes: list[str],
        notes: list[str] | None = None,
    ) -> bool:
        """Carry a FAIL only when every cited defect has stable evidence.

        PASS never uses this path. Pixel-sensitive failures such as clipping,
        blur, formula rendering, and visible LaTeX still require an identical
        rendered image. Content/crop failures can safely remain failed across
        layout-only rerenders when their relevant document evidence is exact.
        """

        if not reviewed or not current or not issue_codes:
            return False

        def compact_text(document: dict) -> str:
            return re.sub(r"\s+", "", str(document.get("display_text") or ""))

        def compact_canonical(document: dict) -> str:
            return re.sub(r"\s+", "", str(document.get("canonical_text") or ""))

        def asset_signature(document: dict) -> tuple[tuple[object, ...], ...]:
            assets = document.get("visual_assets") or []
            if not isinstance(assets, list):
                return ()
            return tuple(
                sorted(
                    (
                        str(asset.get("asset_id") or ""),
                        str(asset.get("crop_sha256") or ""),
                        json.dumps(asset.get("source_bbox"), sort_keys=True),
                        str(asset.get("extraction") or ""),
                        bool(asset.get("redraw_used")),
                    )
                    for asset in assets
                    if isinstance(asset, dict)
                )
            )

        same_text = compact_text(reviewed) == compact_text(current)
        same_canonical = compact_canonical(reviewed) == compact_canonical(current)
        same_assets = asset_signature(reviewed) == asset_signature(current)
        pixel_sensitive_note = bool(
            re.search(
                r"(?:换行|排版|对齐|间隔|字体|字号|大小不一致|显示|重叠|越界|截断|模糊)",
                " ".join(str(note) for note in (notes or [])),
            )
        )
        if pixel_sensitive_note:
            return False
        stable_text_issues = {
            "semantic_content_incomplete",
            "blank_unit_or_punctuation_missing",
            "adjacent_question_or_noise",
        }
        stable_asset_issues = {
            "visual_crop_incomplete",
            "visual_not_source_crop",
        }
        for issue in issue_codes:
            if issue in stable_text_issues and same_text and same_canonical:
                continue
            if issue in stable_asset_issues and same_assets and same_text and same_canonical:
                continue
            if (
                issue == "handwriting_visible"
                and same_text
                and same_canonical
                and same_assets
            ):
                continue
            return False
        return True

    @staticmethod
    def _inherited_output_review(previous_reviews: list[dict]) -> dict | None:
        return next(
            (
                item
                for item in previous_reviews
                if item.get("same_source")
                and (
                    item.get("same_output")
                    or (
                        item.get("verdict") == "fail"
                        and item.get("same_failure_evidence")
                    )
                )
            ),
            None,
        )

    def output_review_history(
        self,
        current_run_id: str,
        sample_id: str,
        current_report: dict,
        limit: int = 8,
    ) -> list[dict]:
        """Return validated, read-only verdicts for the same sample in older runs."""
        self._validate_id(current_run_id, "run id")
        self._validate_id(sample_id, "sample id")
        if not self.output_review_root.is_dir():
            return []
        current_output_path = self._run_path(current_run_id) / sample_id / "output.png"
        try:
            current_document = self._output_document(current_run_id, sample_id)
        except (OSError, ValueError, json.JSONDecodeError):
            current_document = None
        history: list[dict] = []
        for run_dir in self.output_review_root.iterdir():
            run_id = run_dir.name
            if (
                run_id == current_run_id
                or not run_dir.is_dir()
                or not TASK_ID.fullmatch(run_id)
            ):
                continue
            review = self._load_output_review_file(run_id, sample_id)
            if review is None or review.get("verdict") not in {"pass", "fail"}:
                continue
            try:
                reviewed_output_path, report, _ = self._output_context(run_id, sample_id)
                self._validate_output_review(review)
            except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
                continue
            if (
                review.get("run_id") != run_id
                or review.get("sample_id") != sample_id
                or review.get("source_sha256") != report.get("source_sha256")
                or review.get("output_sha256") != report.get("output_sha256")
            ):
                continue
            same_source = review["source_sha256"] == current_report.get("source_sha256")
            same_output_bytes = review["output_sha256"] == current_report.get("output_sha256")
            same_output_pixels = False
            if same_source and not same_output_bytes:
                try:
                    same_output_pixels = self._decoded_output_hash(
                        reviewed_output_path
                    ) == self._decoded_output_hash(current_output_path)
                except (FileNotFoundError, OSError, ValueError):
                    same_output_pixels = False
            issue_codes = [
                key
                for key, value in (review.get("issues") or {}).items()
                if value is True
            ]
            try:
                reviewed_document = self._output_document(run_id, sample_id)
            except (OSError, ValueError, json.JSONDecodeError):
                reviewed_document = None
            same_failure_evidence = bool(
                review["verdict"] == "fail"
                and same_source
                and self._failure_evidence_unchanged(
                    reviewed_document,
                    current_document,
                    issue_codes,
                    list(review.get("notes") or []),
                )
            )
            history.append(
                {
                    "run_id": run_id,
                    "verdict": review["verdict"],
                    "issue_codes": issue_codes,
                    "notes": list(review.get("notes") or []),
                    "reviewer_id": review.get("reviewer_id"),
                    "output_sha256": review["output_sha256"],
                    "same_output": same_output_bytes or same_output_pixels,
                    "same_output_bytes": same_output_bytes,
                    "same_output_pixels": same_output_pixels,
                    "same_failure_evidence": same_failure_evidence,
                    "output_match": (
                        "file_hash"
                        if same_output_bytes
                        else "decoded_pixels"
                        if same_output_pixels
                        else None
                    ),
                    "same_source": same_source,
                    "updated_at_utc": review.get("updated_at_utc"),
                }
            )
        history.sort(
            key=lambda item: (str(item.get("updated_at_utc") or ""), item["run_id"]),
            reverse=True,
        )
        return history[: max(0, limit)]

    def load_output_review(self, run_id: str, sample_id: str) -> dict:
        _, report, _ = self._output_context(run_id, sample_id)
        review = self._load_output_review_file(run_id, sample_id)
        previous_reviews = self.output_review_history(
            run_id,
            sample_id,
            report,
        )
        inherited_review = (
            None
            if review is not None
            else self._inherited_output_review(previous_reviews)
        )
        if review is None:
            review = self._default_output_review(run_id, sample_id, report)
        measurement = report.get("measurement") or {
            "mode": "end_to_end",
            "online_latency_sla_applicable": True,
        }
        return {
            "review": review,
            "previous_reviews": previous_reviews,
            "inherited_review": inherited_review,
            "effective_human_verdict": (
                inherited_review.get("verdict")
                if inherited_review is not None
                else review.get("verdict", "pending")
            ),
            "report": {
                "status": report.get("status"),
                "elapsed_seconds": report.get("elapsed_seconds"),
                "measurement": measurement,
                "output_width": report.get("output_width"),
                "output_height": report.get("output_height"),
                "provider": report.get("provider"),
                "issues": report.get("issues", []),
                "render_readback": report.get("render_readback"),
                "source_resolution": report.get("source_resolution"),
            },
        }

    def save_output_review(
        self,
        run_id: str,
        sample_id: str,
        updated: dict,
        expected_hash: str | None,
    ) -> dict:
        _, report, _ = self._output_context(run_id, sample_id)
        existing = self._load_output_review_file(run_id, sample_id)
        if existing is None:
            existing = self._default_output_review(run_id, sample_id, report)
        if expected_hash and expected_hash != json_hash(existing):
            raise RuntimeError("Output review changed since it was opened; reload before saving")
        for field, expected in (
            ("schema_version", "output_quality_review_v1"),
            ("run_id", run_id),
            ("sample_id", sample_id),
            ("source_sha256", report["source_sha256"]),
            ("output_sha256", report["output_sha256"]),
        ):
            if updated.get(field) != expected:
                raise ValueError(f"{field} cannot change")
        persistent_failure = any(
            item.get("verdict") == "fail"
            and (item.get("same_output") or item.get("same_failure_evidence"))
            for item in self.output_review_history(
                run_id, sample_id, report, limit=1000
            )
        )
        if (
            existing.get("verdict") == "pending"
            and updated.get("verdict") in {"pass", "fail"}
            and persistent_failure
        ):
            raise ValueError(
                "当前结果与历史人工 FAIL 的图像或缺陷证据未变；"
                "请先修复对应内容或裁图，不要重复验收"
            )
        updated["history"] = list(existing.get("history") or [])
        if existing.get("verdict") != updated.get("verdict"):
            updated["history"].append(
                {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "from_verdict": existing.get("verdict"),
                    "to_verdict": updated.get("verdict"),
                    "reviewer_id": updated.get("reviewer_id"),
                }
            )
        updated["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._validate_output_review(updated)
        path = self._output_review_path(run_id, sample_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, updated)
        return updated

    @staticmethod
    def _validate_output_review(review: dict) -> None:
        validate_output_quality_review(review)

    def progress_csv(self) -> bytes:
        stream = io.StringIO(newline="")
        fields = [
            "sample_id", "image_path", "subject", "priority", "status",
            "include_in_benchmark", "prelabel_agreement", "question_number",
            "annotator_id", "reviewer_1_id", "reviewer_2_id", "adjudicator_id",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in sorted(self.review_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            source = payload["source"]
            review = payload["review"]
            writer.writerow(
                {
                    "sample_id": payload["sample_id"],
                    "image_path": source["image_path"],
                    "subject": source["subject"],
                    "priority": review["priority"],
                    "status": review["status"],
                    "include_in_benchmark": source["include_in_benchmark"],
                    "prelabel_agreement": payload["candidate_prelabels"].get("normalized_agreement"),
                    "question_number": payload["gold_annotation"].get("question_number"),
                    "annotator_id": review.get("annotator_id"),
                    "reviewer_1_id": review.get("reviewer_1_id"),
                    "reviewer_2_id": review.get("reviewer_2_id"),
                    "adjudicator_id": review.get("adjudicator_id"),
                }
            )
        return ("\ufeff" + stream.getvalue()).encode("utf-8")

    def output_progress_csv(self, run_id: str) -> bytes:
        stream = io.StringIO(newline="")
        fields = [
            "run_id",
            "sample_id",
            "subject",
            "pipeline_status",
            "human_verdict",
            "reviewer_id",
            "output_width",
            "output_height",
            "elapsed_seconds",
            "automatic_issue_codes",
            "human_issue_codes",
            "source_sha256",
            "output_sha256",
            "notes",
            "updated_at_utc",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in self.list_output_items(run_id):
            _, report, task = self._output_context(run_id, item["sample_id"])
            review = self._load_output_review_file(run_id, item["sample_id"])
            review = review or self._default_output_review(run_id, item["sample_id"], report)
            writer.writerow(
                {
                    "run_id": run_id,
                    "sample_id": item["sample_id"],
                    "subject": task["source"]["subject"],
                    "pipeline_status": report.get("status"),
                    "human_verdict": review.get("verdict"),
                    "reviewer_id": review.get("reviewer_id"),
                    "output_width": report.get("output_width"),
                    "output_height": report.get("output_height"),
                    "elapsed_seconds": report.get("elapsed_seconds"),
                    "automatic_issue_codes": "|".join(item["issue_codes"]),
                    "human_issue_codes": "|".join(
                        key for key, value in (review.get("issues") or {}).items() if value is True
                    ),
                    "source_sha256": report.get("source_sha256"),
                    "output_sha256": report.get("output_sha256"),
                    "notes": " | ".join(review.get("notes") or []),
                    "updated_at_utc": review.get("updated_at_utc"),
                }
            )
        return ("\ufeff" + stream.getvalue()).encode("utf-8")

    def save(self, sample_id: str, updated: dict, expected_hash: str | None) -> dict:
        existing = self.load(sample_id)
        if expected_hash and expected_hash != json_hash(existing):
            raise RuntimeError("Task changed since it was opened; reload before saving")
        if updated.get("sample_id") != sample_id:
            raise ValueError("sample_id cannot change")
        for immutable in ("source", "candidate_prelabels", "prior_manual_evidence"):
            if updated.get(immutable) != existing.get(immutable):
                raise ValueError(f"{immutable} is immutable")
        existing_review = existing.get("review") or {}
        updated_review = updated.get("review") or {}
        protected_review_fields = (
            "status",
            "annotator_id",
            "reviewer_1_id",
            "reviewer_2_id",
            "adjudicator_id",
        )
        for field in protected_review_fields:
            if updated_review.get(field) != existing_review.get(field):
                raise ValueError(f"{field} is server-owned; use a role sign-off action")
        updated_review["history"] = list(existing_review.get("history") or [])
        updated_review["signoffs"] = list(existing_review.get("signoffs") or [])
        existing_annotation = existing.get("gold_annotation") or {}
        updated_annotation = updated.get("gold_annotation") or {}
        if updated_annotation.get("status") != existing_annotation.get("status"):
            raise ValueError("gold_annotation.status is server-owned; use a role sign-off action")
        signed_content_changed = (
            self._annotation_hash(updated_annotation) != self._annotation_hash(existing_annotation)
            or updated_review.get("checks") != existing_review.get("checks")
        )
        if signed_content_changed and any(
            existing_review.get(field)
            for field in ("annotator_id", "reviewer_1_id", "reviewer_2_id", "adjudicator_id")
        ):
            previous_status = existing_review.get("status")
            for field in ("annotator_id", "reviewer_1_id", "reviewer_2_id", "adjudicator_id"):
                updated_review[field] = None
            updated_review["status"] = "pending"
            updated_annotation["status"] = "draft"
            updated_review["history"].append(
                {
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "from_status": previous_status,
                    "to_status": "pending",
                    "actor_id": None,
                    "action": "content_changed_signoffs_invalidated",
                }
            )
        updated["review"] = updated_review
        updated["gold_annotation"] = updated_annotation
        self._validate(updated, existing)
        write_json_atomic(self._task_path(sample_id), updated)
        return updated

    @staticmethod
    def _annotation_hash(annotation: dict) -> str:
        payload = dict(annotation)
        payload.pop("status", None)
        return json_hash(payload)

    @staticmethod
    def _validate_annotation_ready(task: dict) -> None:
        annotation = task.get("gold_annotation") or {}
        review = task.get("review") or {}
        if not annotation.get("canonical_transcript", "").strip():
            raise ValueError("Role sign-off requires a canonical transcript")
        if not annotation.get("question_type", "").strip():
            raise ValueError("Role sign-off requires a question type")
        if not annotation.get("stem_text", "").strip():
            raise ValueError("Role sign-off requires a stem")
        question_type = str(annotation.get("question_type") or "").lower()
        if ("choice" in question_type or "选择" in question_type) and len(annotation.get("options") or []) < 2:
            raise ValueError("A choice question requires at least two complete options before sign-off")
        if not annotation.get("text_blocks"):
            raise ValueError("Role sign-off requires at least one source-aligned text block")
        if annotation.get("formula_status") not in {"human_transcribed", "latex_verified"}:
            raise ValueError("Role sign-off requires human formula verification")
        checks = review.get("checks") or {}
        if set(checks) != GOLD_CHECK_KEYS or any(value is not True for value in checks.values()):
            raise ValueError("Every Gold check must be true before role sign-off")

    @staticmethod
    def _valid_actor_id(actor_id: object) -> str:
        actor = str(actor_id or "").strip()
        if len(actor) < 2 or len(actor) > 80 or any(character.isspace() for character in actor):
            raise ValueError("Actor id must be 2-80 non-whitespace characters")
        return actor

    @staticmethod
    def _has_current_signoff(review: dict, annotation_hash: str, role: str, action: str) -> bool:
        return any(
            isinstance(item, dict)
            and item.get("annotation_sha256") == annotation_hash
            and item.get("role") == role
            and item.get("action") == action
            for item in review.get("signoffs") or []
        )

    def signoff(
        self,
        sample_id: str,
        action: str,
        actor_id: object,
        expected_hash: str | None,
    ) -> dict:
        task = self.load(sample_id)
        if expected_hash and expected_hash != json_hash(task):
            raise RuntimeError("Task changed since it was opened; reload before signing")
        actor = self._valid_actor_id(actor_id)
        review = task.get("review") or {}
        annotation = task.get("gold_annotation") or {}
        review["history"] = list(review.get("history") or [])
        review["signoffs"] = list(review.get("signoffs") or [])
        previous_status = review.get("status")
        annotation_hash = self._annotation_hash(annotation)
        role = ""

        if action == "submit_annotation":
            if previous_status != "pending":
                raise ValueError("Only a pending task can be submitted by an annotator")
            self._validate(task, task)
            self._validate_annotation_ready(task)
            role = "annotator"
            review["annotator_id"] = actor
            review["reviewer_1_id"] = None
            review["reviewer_2_id"] = None
            review["adjudicator_id"] = None
            review["status"] = "annotated"
            annotation["status"] = "annotated"
        elif action == "approve_reviewer_1":
            if previous_status != "annotated" or not review.get("annotator_id"):
                raise ValueError("Reviewer 1 can sign only after annotator submission")
            if actor == review.get("annotator_id"):
                raise ValueError("Reviewer 1 must be distinct from the annotator")
            if not self._has_current_signoff(review, annotation_hash, "annotator", "submit_annotation"):
                raise ValueError("The current annotation has no matching annotator sign-off")
            self._validate(task, task)
            self._validate_annotation_ready(task)
            role = "reviewer_1"
            review["reviewer_1_id"] = actor
            review["status"] = "reviewing"
            annotation["status"] = "annotated"
        elif action == "approve_reviewer_2":
            if previous_status != "reviewing" or not review.get("reviewer_1_id"):
                raise ValueError("Reviewer 2 can sign only after reviewer 1 approval")
            if actor in {review.get("annotator_id"), review.get("reviewer_1_id")}:
                raise ValueError("Reviewer 2 must be distinct from earlier roles")
            if not self._has_current_signoff(review, annotation_hash, "reviewer_1", "approve_reviewer_1"):
                raise ValueError("The current annotation has no matching reviewer 1 sign-off")
            self._validate(task, task)
            self._validate_annotation_ready(task)
            role = "reviewer_2"
            review["reviewer_2_id"] = actor
            review["status"] = "verified"
            annotation["status"] = "verified"
        elif action == "request_changes":
            if previous_status not in {"annotated", "reviewing"}:
                raise ValueError("Changes can be requested only during reviewer stages")
            earlier_identities = {review.get("annotator_id")}
            if review.get("reviewer_1_id"):
                earlier_identities.add(review.get("reviewer_1_id"))
            if actor in earlier_identities:
                raise ValueError("The reviewer requesting changes must be distinct from earlier roles")
            if not review.get("notes"):
                raise ValueError("Add a problem note before requesting changes")
            role = "reviewer_2" if review.get("reviewer_1_id") else "reviewer_1"
            review["status"] = "pending"
            annotation["status"] = "draft"
            for field in ("annotator_id", "reviewer_1_id", "reviewer_2_id", "adjudicator_id"):
                review[field] = None
        elif action == "request_adjudication":
            if previous_status != "reviewing" or not review.get("reviewer_1_id"):
                raise ValueError("Adjudication can be requested only by reviewer 2")
            if actor in {review.get("annotator_id"), review.get("reviewer_1_id")}:
                raise ValueError("The disputing reviewer must be distinct from earlier roles")
            if not review.get("notes"):
                raise ValueError("Add a dispute note before requesting adjudication")
            role = "reviewer_2"
            review["reviewer_2_id"] = actor
            review["status"] = "adjudication_required"
            annotation["status"] = "annotated"
        elif action == "approve_adjudication":
            identities = {
                review.get("annotator_id"),
                review.get("reviewer_1_id"),
                review.get("reviewer_2_id"),
            }
            if previous_status != "adjudication_required" or None in identities or len(identities) != 3:
                raise ValueError("Adjudication requires three completed distinct roles")
            if actor in identities:
                raise ValueError("The adjudicator must be a distinct fourth person")
            self._validate(task, task)
            self._validate_annotation_ready(task)
            role = "adjudicator"
            review["adjudicator_id"] = actor
            review["status"] = "verified"
            annotation["status"] = "verified"
        else:
            raise ValueError("Unknown role sign-off action")

        review["signoffs"].append(
            {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "role": role,
                "actor_id": actor,
                "annotation_sha256": annotation_hash,
                "checks": dict(review.get("checks") or {}),
            }
        )
        review["history"].append(
            {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "from_status": previous_status,
                "to_status": review["status"],
                "actor_id": actor,
                "action": action,
            }
        )
        task["review"] = review
        task["gold_annotation"] = annotation
        self._validate(task, {"review": {"status": previous_status}})
        write_json_atomic(self._task_path(sample_id), task)
        return task

    @staticmethod
    def _validate(task: dict, existing: dict | None = None) -> None:
        annotation = task.get("gold_annotation") or {}
        review = task.get("review") or {}
        options = annotation.get("options") or []
        option_labels: list[str] = []
        for option in options:
            if not isinstance(option, dict):
                raise ValueError("Every option must be a structured label/text object")
            label = str(option.get("label") or "").strip()
            text = str(option.get("text") or "").strip()
            if not label or not text:
                raise ValueError("Every saved option requires both a label and text")
            option_labels.append(label)
        if len(option_labels) != len(set(option_labels)):
            raise ValueError("Option labels must be unique")
        for asset in annotation.get("visual_assets", []):
            if asset.get("preserve_source_pixels") is not True or asset.get("redraw_allowed") is not False:
                raise ValueError("Visual assets must preserve source pixels and forbid redraw")
        if review.get("status") == "verified" or annotation.get("status") == "verified":
            identities = [review.get("annotator_id"), review.get("reviewer_1_id"), review.get("reviewer_2_id")]
            if not all(identities) or len(set(identities)) != 3:
                raise ValueError("Verified Gold requires three distinct annotator/reviewer identities")
            previous_status = (existing or {}).get("review", {}).get("status")
            signoffs = review.get("signoffs") or []
            current_annotation_hash = ReviewStore._annotation_hash(annotation)
            if signoffs:
                adjudication_recorded = previous_status == "adjudication_required" or any(
                    isinstance(event, dict)
                    and event.get("action") == "request_adjudication"
                    and event.get("annotation_sha256") == current_annotation_hash
                    for event in signoffs
                )
            else:
                adjudication_recorded = previous_status == "adjudication_required" or any(
                    event.get("to_status") == "adjudication_required"
                    for event in review.get("history", [])
                    if isinstance(event, dict)
                )
            if adjudication_recorded:
                adjudicator = review.get("adjudicator_id")
                if not adjudicator or adjudicator in identities:
                    raise ValueError("A disputed task requires a distinct fourth adjudicator before verification")
            if not annotation.get("canonical_transcript", "").strip():
                raise ValueError("Verified Gold requires a canonical transcript")
            if not annotation.get("question_type", "").strip():
                raise ValueError("Verified Gold requires a question type")
            if not annotation.get("stem_text", "").strip():
                raise ValueError("Verified Gold requires a stem")
            question_type = str(annotation.get("question_type") or "").lower()
            if ("choice" in question_type or "选择" in question_type) and len(options) < 2:
                raise ValueError("A verified choice question requires at least two options")
            if not annotation.get("text_blocks"):
                raise ValueError("Verified Gold requires at least one source-aligned text block")
            if annotation.get("formula_status") not in {"human_transcribed", "latex_verified"}:
                raise ValueError("Verified Gold requires human formula verification")
            checks = review.get("checks") or {}
            if not checks or any(value is not True for value in checks.values()):
                raise ValueError("Every review check must be true before verification")


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore
    product: ProductOcrService

    def _send_json(self, payload: object, status: int = 200, extra_headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, cache_control: str | None = None) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, filename: str | None = None) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path in {"", "/"}:
                # Keep the product and evaluation surfaces unambiguous.  A bare
                # root is the real OCR product; legacy review bookmarks used a
                # query string on the root, so canonicalize those to /review/
                # without dropping their selected run/sample.
                destination = (
                    f"/review/?{parsed.query}" if parsed.query else "/ocr/"
                )
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", destination)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if parsed.path == "/api/product/health":
                self._send_json(self.product.health())
                return
            if parsed.path == "/api/product/feedback":
                items = self.product.list_feedback()
                self._send_json({"items": items, "count": len(items)})
                return
            if parsed.path.startswith("/api/product/feedback/"):
                sample_id = unquote(parsed.path.removeprefix("/api/product/feedback/"))
                feedback = self.product.load_feedback(sample_id)
                if feedback is None:
                    raise FileNotFoundError(sample_id)
                self._send_json(feedback)
                return
            if parsed.path.startswith("/api/product/source/"):
                token = unquote(parsed.path.removeprefix("/api/product/source/"))
                self._send_file(self.product.source_path(token), "private, no-store")
                return
            if parsed.path.startswith("/api/product/result/") and parsed.path.endswith(
                "/output.png"
            ):
                sample_id = unquote(
                    parsed.path.removeprefix("/api/product/result/").removesuffix(
                        "/output.png"
                    )
                )
                if parse_qs(parsed.query).get("download") == ["1"]:
                    output_path = self.product.verified_download_path(sample_id)
                    self._send_bytes(
                        output_path.read_bytes(),
                        "image/png",
                        f"{sample_id}-800px.png",
                    )
                else:
                    output_path = self.product.output_path(sample_id)
                    self._send_file(output_path, "private, no-store")
                return
            if parsed.path.startswith("/api/product/result/"):
                sample_id = unquote(parsed.path.removeprefix("/api/product/result/"))
                self._send_json(self.product.result_details(sample_id))
                return
            if parsed.path == "/api/tasks":
                self._send_json({"tasks": self.store.list()})
                return
            if parsed.path == "/api/mask-items":
                self._send_json({"items": self.store.handwriting_masks.list()})
                return
            if parsed.path == "/api/intake-items":
                self._send_json(
                    {
                        "items": self.store.self_built_intake.list(),
                        "summary": self.store.self_built_intake.summary(),
                        "collection_workspace": self.store.collection_workspace_summary(),
                    }
                )
                return
            if parsed.path == "/api/boundary-items":
                self._send_json(
                    {
                        "items": self.store.question_boundaries.list(),
                        "summary": self.store.question_boundaries.summary(),
                    }
                )
                return
            if parsed.path == "/api/pair-items":
                self._send_json(
                    {
                        "items": self.store.capture_pairs.list(),
                        "summary": self.store.capture_pairs.summary(),
                    }
                )
                return
            if parsed.path == "/api/self-built-collection-plan.csv":
                self._send_bytes(
                    collection_plan_csv(),
                    "text/csv; charset=utf-8",
                    "test7_ocr_self_built_collection_plan.csv",
                )
                return
            if parsed.path == "/api/version":
                self._send_json(
                    {
                        "version": REVIEW_API_VERSION,
                        "ui_version": REVIEW_UI_VERSION,
                        "product_version": PRODUCT_API_VERSION,
                        "product_url": "/ocr/",
                        "review_url": "/review/",
                    }
                )
                return
            if parsed.path == "/api/runs":
                self._send_json({"runs": self.store.list_runs()})
                return
            if parsed.path.startswith("/api/run/") and parsed.path.endswith("/items"):
                run_id = unquote(parsed.path.removeprefix("/api/run/").removesuffix("/items").strip("/"))
                self._send_json({"items": self.store.list_output_items(run_id)})
                return
            if parsed.path.startswith("/api/candidate-structure/"):
                sample_id = unquote(parsed.path.removeprefix("/api/candidate-structure/"))
                self._send_json(self.store.candidate_structure(sample_id))
                return
            if parsed.path.startswith("/api/output-review/"):
                parts = [unquote(part) for part in parsed.path.removeprefix("/api/output-review/").split("/")]
                if len(parts) != 2:
                    raise ValueError("Invalid output review path")
                payload = self.store.load_output_review(parts[0], parts[1])
                self._send_json(payload, extra_headers={"ETag": json_hash(payload["review"])})
                return
            if parsed.path.startswith("/api/mask-task/"):
                sample_id = unquote(parsed.path.removeprefix("/api/mask-task/"))
                task = self.store.handwriting_masks.load(sample_id)
                self._send_json(task, extra_headers={"ETag": payload_hash(task)})
                return
            if parsed.path.startswith("/api/boundary-task/"):
                asset_id = unquote(parsed.path.removeprefix("/api/boundary-task/"))
                task = self.store.question_boundaries.load(asset_id)
                self._send_json(task, extra_headers={"ETag": boundary_payload_hash(task)})
                return
            if parsed.path.startswith("/api/pair-task/"):
                capture_asset_id = unquote(parsed.path.removeprefix("/api/pair-task/"))
                task = self.store.capture_pairs.load(capture_asset_id)
                self._send_json(
                    task, extra_headers={"ETag": capture_pair_payload_hash(task)}
                )
                return
            if parsed.path == "/api/progress.csv":
                self._send_bytes(
                    self.store.progress_csv(),
                    "text/csv; charset=utf-8",
                    "ocr_gold_review_progress.csv",
                )
                return
            if parsed.path == "/api/mask-progress.csv":
                self._send_bytes(
                    self.store.handwriting_masks.progress_csv(),
                    "text/csv; charset=utf-8",
                    "testzip_handwriting_mask_eval_progress.csv",
                )
                return
            if parsed.path == "/api/boundary-progress.csv":
                self._send_bytes(
                    self.store.question_boundaries.progress_csv(),
                    "text/csv; charset=utf-8",
                    "question_boundary_review_progress.csv",
                )
                return
            if parsed.path == "/api/pair-progress.csv":
                self._send_bytes(
                    self.store.capture_pairs.progress_csv(),
                    "text/csv; charset=utf-8",
                    "capture_pair_review_progress.csv",
                )
                return
            if parsed.path.startswith("/api/output-progress/") and parsed.path.endswith(".csv"):
                run_id = unquote(
                    parsed.path.removeprefix("/api/output-progress/").removesuffix(".csv")
                )
                self._send_bytes(
                    self.store.output_progress_csv(run_id),
                    "text/csv; charset=utf-8",
                    f"{run_id}_final_png_review.csv",
                )
                return
            if parsed.path.startswith("/api/task/"):
                sample_id = unquote(parsed.path.removeprefix("/api/task/"))
                task = self.store.load(sample_id)
                self._send_json(task, extra_headers={"ETag": json_hash(task)})
                return
            if parsed.path.startswith("/source/"):
                sample_id = unquote(parsed.path.removeprefix("/source/"))
                self._send_file(self.store.source_path(sample_id))
                return
            if parsed.path.startswith("/mask-source/"):
                sample_id = unquote(parsed.path.removeprefix("/mask-source/"))
                self._send_file(self.store.handwriting_masks.source_path(sample_id), "no-store")
                return
            if parsed.path.startswith("/handwriting-mask/"):
                sample_id = unquote(parsed.path.removeprefix("/handwriting-mask/"))
                self._send_file(self.store.handwriting_masks.mask_path(sample_id), "no-store")
                return
            if parsed.path.startswith("/intake-source/"):
                asset_id = unquote(parsed.path.removeprefix("/intake-source/"))
                self._send_file(self.store.self_built_intake.source_path(asset_id), "no-store")
                return
            if parsed.path.startswith("/output/"):
                parts = [unquote(part) for part in parsed.path.removeprefix("/output/").split("/")]
                if len(parts) != 2:
                    raise ValueError("Invalid output image path")
                self._send_file(self.store.output_path(parts[0], parts[1]))
                return
            if parsed.path.startswith("/katex/"):
                if KATEX_ROOT is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                relative = Path(unquote(parsed.path.removeprefix("/katex/")))
                katex_path = (KATEX_ROOT / relative).resolve()
                if KATEX_ROOT not in katex_path.parents:
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                self._send_file(katex_path)
                return
            if parsed.path in {"/ocr", "/ocr/"} or parsed.path.startswith("/ocr/"):
                product_relative = (
                    "index.html"
                    if parsed.path in {"/ocr", "/ocr/"}
                    else parsed.path.removeprefix("/ocr/")
                )
                product_path = (PRODUCT_UI_ROOT / product_relative).resolve()
                if (
                    PRODUCT_UI_ROOT not in product_path.parents
                    and product_path != PRODUCT_UI_ROOT
                ):
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                self._send_file(
                    product_path,
                    cache_control=(
                        "no-store" if product_relative == "index.html" else "no-cache"
                    ),
                )
                return
            if parsed.path in {"/review", "/review/"} or parsed.path.startswith(
                "/review/"
            ):
                review_relative = (
                    "index.html"
                    if parsed.path in {"/review", "/review/"}
                    else parsed.path.removeprefix("/review/")
                )
                review_path = (UI_ROOT / review_relative).resolve()
                if UI_ROOT not in review_path.parents and review_path != UI_ROOT:
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                self._send_file(
                    review_path,
                    cache_control=(
                        "no-store" if review_relative == "index.html" else "no-cache"
                    ),
                )
                return
            relative = "index.html" if parsed.path in {"", "/"} else parsed.path.lstrip("/")
            static_path = (UI_ROOT / relative).resolve()
            if UI_ROOT not in static_path.parents and static_path != UI_ROOT:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            self._send_file(
                static_path,
                cache_control="no-store" if relative == "index.html" else "no-cache",
            )
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
        except (RuntimeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=400)

    def do_PUT(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/boundary-task/"):
                asset_id = unquote(parsed.path.removeprefix("/api/boundary-task/"))
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2_000_000:
                    raise ValueError("Invalid boundary task request size")
                updated = json.loads(self.rfile.read(length).decode("utf-8"))
                saved = self.store.question_boundaries.save(
                    asset_id, updated, self.headers.get("If-Match")
                )
                self._send_json(saved, extra_headers={"ETag": boundary_payload_hash(saved)})
                return
            if parsed.path.startswith("/api/pair-task/"):
                capture_asset_id = unquote(parsed.path.removeprefix("/api/pair-task/"))
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 200_000:
                    raise ValueError("Invalid capture-pair task request size")
                updated = json.loads(self.rfile.read(length).decode("utf-8"))
                saved = self.store.capture_pairs.save(
                    capture_asset_id, updated, self.headers.get("If-Match")
                )
                self._send_json(
                    saved, extra_headers={"ETag": capture_pair_payload_hash(saved)}
                )
                return
            if parsed.path.startswith("/api/mask-task/"):
                sample_id = unquote(parsed.path.removeprefix("/api/mask-task/"))
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 15_000_000:
                    raise ValueError("Invalid handwriting-mask request body size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict) or not isinstance(payload.get("task"), dict):
                    raise ValueError("Handwriting-mask save requires a task object")
                saved = self.store.handwriting_masks.save(
                    sample_id,
                    payload["task"],
                    payload.get("mask_png_base64"),
                    self.headers.get("If-Match"),
                )
                self._send_json(saved, extra_headers={"ETag": payload_hash(saved)})
                return
            if parsed.path.startswith("/api/output-review/"):
                parts = [unquote(part) for part in parsed.path.removeprefix("/api/output-review/").split("/")]
                if len(parts) != 2:
                    raise ValueError("Invalid output review path")
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 200_000:
                    raise ValueError("Invalid request body size")
                updated = json.loads(self.rfile.read(length).decode("utf-8"))
                saved = self.store.save_output_review(
                    parts[0], parts[1], updated, self.headers.get("If-Match")
                )
                self._send_json(saved, extra_headers={"ETag": json_hash(saved)})
                return
            if not parsed.path.startswith("/api/task/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            sample_id = unquote(parsed.path.removeprefix("/api/task/"))
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("Invalid request body size")
            updated = json.loads(self.rfile.read(length).decode("utf-8"))
            saved = self.store.save(sample_id, updated, self.headers.get("If-Match"))
            self._send_json(saved, extra_headers={"ETag": json_hash(saved)})
        except (json.JSONDecodeError, RuntimeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=409 if isinstance(exc, RuntimeError) else 400)

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/product/process":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 30_000_000:
                    raise ValueError("OCR 上传请求为空或超过 20MB 图片上限")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._send_json(
                    self.product.process_upload(payload),
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path.startswith("/api/product/feedback/"):
                sample_id = unquote(parsed.path.removeprefix("/api/product/feedback/"))
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 20_000:
                    raise ValueError("问题反馈请求为空或过大")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._send_json(
                    self.product.save_feedback(sample_id, payload),
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/collection-workspace-refresh":
                self._send_json(
                    {
                        "collection_workspace": self.store.collection_workspace_summary(
                            refresh=True
                        )
                    }
                )
                return
            if parsed.path == "/api/collection-guided-register":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 42_000_000:
                    raise ValueError("Invalid guided collection request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Guided collection upload requires a JSON object")
                record, collection_report, guided_row = register_guided_collection_image(
                    self.store.self_built_intake.dataset_root / "collection",
                    self.store.self_built_intake,
                    row_id=str(payload.get("row_id") or ""),
                    encoded_image=payload.get("image_base64"),
                    privacy_checked=payload.get("privacy_checked"),
                    contains_personal_data=payload.get("contains_personal_data"),
                    distribution_allowed=payload.get("distribution_allowed"),
                )
                self.store.handwriting_masks.sync_from_intake()
                self.store.question_boundaries.sync_from_intake()
                self.store.capture_pairs.sync_from_intake()
                self._send_json(
                    {
                        "record": record,
                        "guided_row": guided_row,
                        "items": self.store.self_built_intake.list(),
                        "summary": self.store.self_built_intake.summary(),
                        "collection_workspace": self.store._public_collection_workspace(
                            collection_report
                        ),
                    },
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path == "/api/intake-register":
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 42_000_000:
                    raise ValueError("Invalid self-built intake request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Self-built intake requires a JSON object")
                record = self.store.self_built_intake.register(
                    payload.get("metadata") or {}, payload.get("image_base64")
                )
                self.store.handwriting_masks.sync_from_intake()
                self.store.question_boundaries.sync_from_intake()
                self.store.capture_pairs.sync_from_intake()
                self._send_json(
                    {
                        "record": record,
                        "items": self.store.self_built_intake.list(),
                        "summary": self.store.self_built_intake.summary(),
                        "collection_workspace": self.store.collection_workspace_summary(),
                    },
                    status=HTTPStatus.CREATED,
                )
                return
            if parsed.path.startswith("/api/boundary-signoff/"):
                asset_id = unquote(parsed.path.removeprefix("/api/boundary-signoff/"))
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 20_000:
                    raise ValueError("Invalid boundary sign-off request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Boundary sign-off requires a JSON object")
                saved = self.store.question_boundaries.signoff(
                    asset_id,
                    str(payload.get("action") or ""),
                    payload.get("actor_id"),
                    self.headers.get("If-Match"),
                )
                self._send_json(saved, extra_headers={"ETag": boundary_payload_hash(saved)})
                return
            if parsed.path.startswith("/api/pair-signoff/"):
                capture_asset_id = unquote(parsed.path.removeprefix("/api/pair-signoff/"))
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 20_000:
                    raise ValueError("Invalid capture-pair sign-off request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Capture-pair sign-off requires a JSON object")
                saved = self.store.capture_pairs.signoff(
                    capture_asset_id,
                    str(payload.get("action") or ""),
                    payload.get("actor_id"),
                    self.headers.get("If-Match"),
                )
                self._send_json(
                    saved, extra_headers={"ETag": capture_pair_payload_hash(saved)}
                )
                return
            if parsed.path.startswith("/api/mask-signoff/"):
                sample_id = unquote(parsed.path.removeprefix("/api/mask-signoff/"))
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 20_000:
                    raise ValueError("Invalid handwriting-mask sign-off request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Handwriting-mask sign-off requires a JSON object")
                saved = self.store.handwriting_masks.signoff(
                    sample_id,
                    str(payload.get("action") or ""),
                    payload.get("actor_id"),
                    self.headers.get("If-Match"),
                )
                self._send_json(saved, extra_headers={"ETag": payload_hash(saved)})
                return
            if not parsed.path.startswith("/api/task-signoff/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            sample_id = unquote(parsed.path.removeprefix("/api/task-signoff/"))
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20_000:
                raise ValueError("Invalid request body size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Sign-off request must be a JSON object")
            saved = self.store.signoff(
                sample_id,
                str(payload.get("action") or ""),
                payload.get("actor_id"),
                self.headers.get("If-Match"),
            )
            self._send_json(saved, extra_headers={"ETag": json_hash(saved)})
        except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=409 if isinstance(exc, RuntimeError) else 400)

    def log_message(self, format: str, *args: object) -> None:
        print(f"review-ui: {format % args}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local OCR Gold review workstation")
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotations" / "gaokao_physics_ocr_gold" / "drafts",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=(PROJECT_ROOT / ".." / ".." / "datasets" / "gaokao_physics_300_v1_0_1").resolve(),
    )
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "artifacts" / "runs")
    parser.add_argument(
        "--output-review-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotations" / "output_quality_reviews",
    )
    parser.add_argument(
        "--mask-task-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotations" / "handwriting_masks" / "tasks",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotations" / "handwriting_masks" / "masks",
    )
    parser.add_argument(
        "--mask-dataset-root",
        type=Path,
        default=(PROJECT_ROOT / ".." / ".." / "datasets" / "testzip_math_physics_300_v1").resolve(),
        help="Dataset root for retained handwriting-mask candidates; independent of primary OCR Gold",
    )
    parser.add_argument(
        "--auxiliary-review-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "testzip_math_physics_ocr_gold"
            / "drafts"
        ),
        help="Read-only task metadata used to review output runs outside the primary Gold queue",
    )
    parser.add_argument(
        "--auxiliary-dataset-root",
        type=Path,
        default=(PROJECT_ROOT / ".." / ".." / "datasets" / "testzip_math_physics_300_v1").resolve(),
        help="Dataset root paired with --auxiliary-review-dir",
    )
    parser.add_argument(
        "--intake-dataset-root",
        type=Path,
        default=PROJECT_ROOT.parents[1] / "datasets" / "test7_ocr_self_built_v1",
    )
    parser.add_argument(
        "--intake-record-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotations" / "self_built_intake" / "records",
    )
    parser.add_argument(
        "--intake-summary-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "self_built_intake_summary.json",
    )
    parser.add_argument(
        "--boundary-task-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotations" / "question_boundaries" / "tasks",
    )
    parser.add_argument(
        "--boundary-manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "question_boundary_manifest.jsonl",
    )
    parser.add_argument(
        "--boundary-summary-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "question_boundary_summary.json",
    )
    parser.add_argument(
        "--capture-pair-task-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "annotations" / "capture_pairs" / "tasks",
    )
    parser.add_argument(
        "--capture-pair-manifest-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "capture_pair_manifest.jsonl",
    )
    parser.add_argument(
        "--capture-pair-summary-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "capture_pair_summary.json",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--product-input-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "product_inputs",
    )
    parser.add_argument(
        "--product-output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "product_runs",
    )
    parser.add_argument(
        "--product-feedback-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "product_feedback",
    )
    parser.add_argument(
        "--structured-service-url",
        default="http://127.0.0.1:8240",
    )
    args = parser.parse_args()
    ReviewHandler.store = ReviewStore(
        args.review_dir,
        args.dataset_root,
        args.runs_root,
        args.output_review_root,
        args.mask_task_dir,
        args.mask_dir,
        args.intake_dataset_root,
        args.intake_record_dir,
        args.intake_summary_path,
        args.boundary_task_dir,
        args.boundary_manifest_path,
        args.boundary_summary_path,
        args.capture_pair_task_dir,
        args.capture_pair_manifest_path,
        args.capture_pair_summary_path,
        args.mask_dataset_root,
        args.auxiliary_review_dir,
        args.auxiliary_dataset_root,
    )
    ReviewHandler.product = ProductOcrService(
        input_root=args.product_input_root,
        output_root=args.product_output_root,
        feedback_root=args.product_feedback_root,
        service_url=args.structured_service_url,
    )
    # Build the OCR pipeline before the HTTP port is advertised as ready.  This
    # keeps provider/readback initialization out of the first user request and
    # makes the product latency number describe actual per-image work.
    ReviewHandler.product.prepare()
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"OCR product UI: http://{args.host}:{args.port}/ocr/")
    print(f"Review workstation: http://{args.host}:{args.port}/review/")
    print(f"Gold review UI: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        ReviewHandler.product.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
