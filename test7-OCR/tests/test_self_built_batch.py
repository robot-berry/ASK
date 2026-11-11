from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from test7_ocr.capture_pair_review import CapturePairStore
from test7_ocr.handwriting_review import HandwritingMaskStore
from test7_ocr.question_boundary_review import QuestionBoundaryStore
from test7_ocr.self_built_batch import BATCH_HEADERS, SelfBuiltBatchImporter
from test7_ocr.self_built_intake import SelfBuiltIntakeStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SelfBuiltBatchImporterTests(unittest.TestCase):
    @staticmethod
    def _write_image(path: Path, color: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 480), color).save(path, format="PNG")

    @staticmethod
    def _row(row_id: str, image_path: str, role: str) -> dict[str, str]:
        return {
            "row_id": row_id,
            "image_path": image_path,
            "role": role,
            "subject": "mathematics",
            "scope": "high_school",
            "document_group": "batch-paper-001",
            "page_index": "1",
            "clean_row_id": "",
            "clean_asset_id": "",
            "device_code": "",
            "conditions": "",
            "page_layout": "not_applicable",
            "provenance": "self_created",
            "license_reference": "",
            "privacy_checked": "true",
            "contains_personal_data": "false",
            "distribution_allowed": "false",
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BATCH_HEADERS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _stores(root: Path) -> tuple[
        SelfBuiltIntakeStore,
        HandwritingMaskStore,
        QuestionBoundaryStore,
        CapturePairStore,
    ]:
        intake = SelfBuiltIntakeStore(
            root / "dataset",
            root / "records",
            root / "summary.json",
        )
        masks = HandwritingMaskStore(
            root / "mask-tasks",
            root / "masks",
            root / "primary-dataset",
            intake,
        )
        boundaries = QuestionBoundaryStore(
            root / "boundary-tasks",
            root / "boundary.jsonl",
            root / "boundary-summary.json",
            intake,
        )
        pairs = CapturePairStore(
            root / "pair-tasks",
            root / "pairs.jsonl",
            root / "pair-summary.json",
            intake,
        )
        return intake, masks, boundaries, pairs

    def _valid_batch(self, root: Path) -> Path:
        batch = root / "incoming"
        self._write_image(batch / "images" / "clean.png", "white")
        self._write_image(batch / "images" / "photo.png", "#dddddd")
        self._write_image(batch / "images" / "hand.png", "#bbbbbb")
        self._write_image(batch / "images" / "boundary.png", "#999999")

        photo = self._row("photo-001", "images/photo.png", "photo_capture")
        photo.update(
            clean_row_id="clean-001",
            device_code="device-a",
            conditions="perspective|shadow",
        )
        boundary = self._row("boundary-001", "images/boundary.png", "boundary_page")
        boundary["page_layout"] = "double_column"
        clean = self._row("clean-001", "images/clean.png", "clean_page")
        handwriting = self._row("hand-001", "images/hand.png", "handwriting_capture")
        handwriting.update(
            clean_row_id="clean-001",
            device_code="device-b",
            conditions="frontal|uneven_light",
        )
        manifest = batch / "batch.csv"
        # Captures intentionally precede their clean row to exercise two-pass dependency handling.
        self._write_csv(manifest, [photo, boundary, clean, handwriting])
        return manifest

    @staticmethod
    def _validate_report(report: dict) -> None:
        import json

        schema = json.loads(
            (PROJECT_ROOT / "schemas" / "self_built_batch_report.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self_required = set(schema["required"])
        if set(report) != self_required:
            raise AssertionError(f"report keys do not match schema: {set(report) ^ self_required}")
        if report["schema_version"] != "self_built_batch_report_v1":
            raise AssertionError("unexpected batch report schema version")
        if report["status"] not in schema["properties"]["status"]["enum"]:
            raise AssertionError("unexpected batch report status")

    def test_dry_run_is_read_only_then_apply_syncs_all_review_queues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._valid_batch(root)
            intake, masks, boundaries, pairs = self._stores(root)
            importer = SelfBuiltBatchImporter(
                intake,
                root / "reports",
                [masks.sync_from_intake, boundaries.sync_from_intake, pairs.sync_from_intake],
            )

            dry_run = importer.run(manifest, apply=False)
            self._validate_report(dry_run)
            self.assertEqual(dry_run["status"], "ready")
            self.assertEqual(dry_run["rows_in_manifest"], 4)
            self.assertEqual(intake.list(), [])
            self.assertFalse((root / "reports").exists())

            applied = importer.run(manifest, apply=True)
            self._validate_report(applied)
            self.assertEqual(applied["status"], "applied")
            self.assertEqual(len(intake.list()), 4)
            self.assertEqual(len(list((root / "mask-tasks").glob("*.json"))), 1)
            self.assertEqual(len(list((root / "boundary-tasks").glob("*.json"))), 1)
            self.assertEqual(len(list((root / "pair-tasks").glob("*.json"))), 2)
            self.assertTrue(list((root / "reports").glob("BATCH-*.json")))

            repeated = importer.run(manifest, apply=True)
            self._validate_report(repeated)
            self.assertEqual(repeated["status"], "applied")
            self.assertTrue(all(row["status"] == "already_registered" for row in repeated["rows"]))
            self.assertNotEqual(repeated["batch_id"], applied["batch_id"])
            self.assertEqual(len(intake.list()), 4)

    def test_invalid_privacy_field_rejects_whole_batch_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._valid_batch(root)
            rows: list[dict[str, str]] = []
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["privacy_checked"] = "yes"
            self._write_csv(manifest, rows)
            intake, _masks, _boundaries, _pairs = self._stores(root)

            report = SelfBuiltBatchImporter(intake, root / "reports").run(manifest, apply=True)
            self._validate_report(report)
            self.assertEqual(report["status"], "rejected_preflight")
            self.assertEqual(report["rows_in_manifest"], 4)
            self.assertEqual(intake.list(), [])

    def test_path_escape_and_duplicate_image_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "incoming"
            self._write_image(root / "outside.png", "white")
            escaped = self._row("clean-a", "../outside.png", "clean_page")
            manifest = batch / "escape.csv"
            batch.mkdir(parents=True, exist_ok=True)
            self._write_csv(manifest, [escaped])
            intake, _masks, _boundaries, _pairs = self._stores(root)
            report = SelfBuiltBatchImporter(intake, root / "reports").run(
                manifest, apply=False
            )
            self.assertEqual(report["status"], "rejected_preflight")
            self.assertIn("safe path", report["failures"][0]["error"])

            self._write_image(batch / "same.png", "#eeeeee")
            first = self._row("clean-a", "same.png", "clean_page")
            second = self._row("clean-b", "same.png", "clean_page")
            duplicate_manifest = batch / "duplicate.csv"
            self._write_csv(duplicate_manifest, [first, second])
            duplicate_report = SelfBuiltBatchImporter(intake, root / "reports").run(
                duplicate_manifest, apply=False
            )
            self.assertEqual(duplicate_report["status"], "rejected_preflight")
            self.assertIn("identical source image bytes", duplicate_report["failures"][0]["error"])
            self.assertEqual(intake.list(), [])


if __name__ == "__main__":
    unittest.main()
