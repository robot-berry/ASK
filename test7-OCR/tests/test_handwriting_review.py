from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from test7_ocr.handwriting_review import (
    HandwritingMaskStore,
    MASK_CHECK_KEYS,
    MASK_CLASS_VALUES,
    payload_hash,
)
from test7_ocr.self_built_intake import SelfBuiltIntakeStore


class HandwritingMaskStoreTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[HandwritingMaskStore, dict]:
        dataset_root = root / "dataset"
        task_dir = root / "annotations" / "tasks"
        mask_dir = root / "annotations" / "masks"
        dataset_root.mkdir()
        task_dir.mkdir(parents=True)
        source_path = dataset_root / "source.png"
        Image.new("RGB", (8, 6), "white").save(source_path)
        source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        task = {
            "schema_version": "handwriting_mask_review_v1",
            "sample_id": "sample-1",
            "partition": "frozen_test",
            "training_allowed": False,
            "source": {
                "dataset": "testzip_math_physics_300_v1",
                "image_path": "source.png",
                "image_sha256": source_sha,
                "width": 8,
                "height": 6,
                "subject": "mathematics",
                "document_group": "mathematics:Test1",
            },
            "selection_evidence": [
                {
                    "source": "output_quality_review",
                    "kind": "handwriting_contamination",
                    "note": "human-confirmed handwriting",
                }
            ],
            "annotation": {
                "status": "pending",
                "mask_path": None,
                "mask_sha256": None,
                "legend_version": "handwriting_mask_v1",
                "class_values": dict(MASK_CLASS_VALUES),
                "pixel_counts": {key: 0 for key in MASK_CLASS_VALUES},
            },
            "review": {
                "annotator_id": None,
                "reviewer_id": None,
                "checks": {key: None for key in sorted(MASK_CHECK_KEYS)},
                "notes": [],
                "signoffs": [],
                "history": [],
                "updated_at_utc": None,
            },
        }
        (task_dir / "sample-1.json").write_text(json.dumps(task), encoding="utf-8")
        return HandwritingMaskStore(task_dir, mask_dir, dataset_root), task

    @staticmethod
    def _mask_data(handwriting_value: int = 2) -> str:
        image = Image.new("L", (8, 6), 0)
        image.putpixel((3, 2), handwriting_value)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _image_data(color: str) -> str:
        image = Image.new("RGB", (640, 480), color)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _intake_metadata(role: str) -> dict:
        return {
            "role": role,
            "subject": "physics",
            "scope": "high_school",
            "document_group": "selfbuilt-physics-paper-001",
            "page_index": 2,
            "clean_asset_id": None,
            "device_code": None,
            "conditions": [],
            "page_layout": "not_applicable",
            "provenance": "self_created",
            "license_reference": None,
            "privacy_checked": True,
            "contains_personal_data": False,
            "distribution_allowed": False,
        }

    @staticmethod
    def _large_mask_data() -> str:
        image = Image.new("L", (640, 480), 0)
        image.putpixel((300, 220), MASK_CLASS_VALUES["student_handwriting"])
        output = io.BytesIO()
        image.save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    def test_save_and_two_person_signoff_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, task = self._fixture(Path(temporary))
            saved = store.save(
                "sample-1", copy.deepcopy(task), self._mask_data(), payload_hash(task)
            )
            self.assertEqual(saved["annotation"]["status"], "draft")
            self.assertEqual(saved["annotation"]["pixel_counts"]["student_handwriting"], 1)
            self.assertTrue(store.mask_path("sample-1").is_file())

            submitted = store.signoff(
                "sample-1", "submit_annotation", "annotator-a", payload_hash(saved)
            )
            self.assertEqual(submitted["annotation"]["status"], "annotated")
            with self.assertRaisesRegex(ValueError, "distinct"):
                store.signoff(
                    "sample-1", "approve_review", "annotator-a", payload_hash(submitted)
                )

            checked = copy.deepcopy(submitted)
            checked["review"]["checks"] = {key: True for key in MASK_CHECK_KEYS}
            checked = store.save("sample-1", checked, None, payload_hash(submitted))
            verified = store.signoff(
                "sample-1", "approve_review", "reviewer-b", payload_hash(checked)
            )
            self.assertEqual(verified["annotation"]["status"], "verified")
            self.assertEqual(verified["review"]["annotator_id"], "annotator-a")
            self.assertEqual(verified["review"]["reviewer_id"], "reviewer-b")
            self.assertIn("sample-1", store.progress_csv().decode("utf-8-sig"))

    def test_mask_change_invalidates_verified_signoffs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, task = self._fixture(Path(temporary))
            task = store.save("sample-1", task, self._mask_data(), payload_hash(task))
            task = store.signoff("sample-1", "submit_annotation", "annotator-a", payload_hash(task))
            updated = copy.deepcopy(task)
            updated["review"]["checks"] = {key: True for key in MASK_CHECK_KEYS}
            task = store.save("sample-1", updated, None, payload_hash(task))
            task = store.signoff("sample-1", "approve_review", "reviewer-b", payload_hash(task))

            changed = store.save("sample-1", copy.deepcopy(task), self._mask_data(3), payload_hash(task))
            self.assertEqual(changed["annotation"]["status"], "draft")
            self.assertIsNone(changed["review"]["annotator_id"])
            self.assertIsNone(changed["review"]["reviewer_id"])

    def test_frozen_source_and_class_values_cannot_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, task = self._fixture(Path(temporary))
            changed = copy.deepcopy(task)
            changed["training_allowed"] = True
            with self.assertRaisesRegex(ValueError, "immutable"):
                store.save("sample-1", changed, self._mask_data(), payload_hash(task))

    def test_annotator_cannot_submit_empty_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, task = self._fixture(Path(temporary))
            empty = Image.new("L", (8, 6), 0)
            output = io.BytesIO()
            empty.save(output, format="PNG")
            encoded = base64.b64encode(output.getvalue()).decode("ascii")
            task = store.save("sample-1", task, encoded, payload_hash(task))
            with self.assertRaisesRegex(ValueError, "at least one handwriting"):
                store.signoff("sample-1", "submit_annotation", "annotator-a", payload_hash(task))

    def test_registered_handwriting_capture_enters_quarantined_mask_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            intake = SelfBuiltIntakeStore(
                root / "self-built-dataset", root / "intake-records", root / "intake-summary.json"
            )
            clean = intake.register(self._intake_metadata("clean_page"), self._image_data("white"))
            capture_metadata = self._intake_metadata("handwriting_capture")
            capture_metadata.update(
                {
                    "clean_asset_id": clean["asset_id"],
                    "device_code": "device-a",
                    "conditions": ["perspective", "shadow"],
                }
            )
            capture = intake.register(capture_metadata, self._image_data("#eeeeee"))
            primary_root = root / "primary-dataset"
            primary_root.mkdir()
            store = HandwritingMaskStore(
                root / "mask-tasks", root / "masks", primary_root, intake
            )

            self.assertEqual(store.sync_from_intake(), 1)
            self.assertEqual(store.sync_from_intake(), 0)
            task = store.load(capture["asset_id"])
            self.assertEqual(task["schema_version"], "handwriting_mask_review_v2")
            self.assertEqual(task["partition"], "quarantine")
            self.assertFalse(task["training_allowed"])
            self.assertFalse(task["evaluation_allowed"])
            self.assertEqual(task["source"]["clean_asset_id"], clean["asset_id"])
            self.assertEqual(store.source_path(capture["asset_id"]), intake.source_path(capture["asset_id"]))
            self.assertEqual(store.list()[0]["dataset"], "test7_ocr_self_built_v1")

            task = store.save(
                capture["asset_id"], task, self._large_mask_data(), payload_hash(task)
            )
            task = store.signoff(
                capture["asset_id"], "submit_annotation", "annotator-a", payload_hash(task)
            )
            reviewed = copy.deepcopy(task)
            reviewed["review"]["checks"] = {key: True for key in MASK_CHECK_KEYS}
            task = store.save(capture["asset_id"], reviewed, None, payload_hash(task))
            task = store.signoff(
                capture["asset_id"], "approve_review", "reviewer-b", payload_hash(task)
            )
            self.assertEqual(task["annotation"]["status"], "verified")


if __name__ == "__main__":
    unittest.main()
