from __future__ import annotations

import base64
import copy
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from test7_ocr.capture_pair_review import (
    PAIR_CHECK_KEYS,
    CapturePairStore,
    capture_pair_target_errors,
    payload_hash,
)
from test7_ocr.self_built_intake import SelfBuiltIntakeStore


class CapturePairStoreTests(unittest.TestCase):
    @staticmethod
    def _image_data(color: str) -> str:
        image = Image.new("RGB", (800, 1000), color)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _base_metadata(role: str) -> dict:
        return {
            "role": role,
            "subject": "mathematics",
            "scope": "high_school",
            "document_group": "math-paper-001",
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

    def _fixture(
        self, root: Path, role: str = "photo_capture"
    ) -> tuple[CapturePairStore, str]:
        intake = SelfBuiltIntakeStore(
            root / "dataset", root / "intake-records", root / "intake-summary.json"
        )
        clean = intake.register(self._base_metadata("clean_page"), self._image_data("white"))
        metadata = self._base_metadata(role)
        metadata.update(
            {
                "clean_asset_id": clean["asset_id"],
                "device_code": "phone-a",
                "conditions": ["perspective", "shadow"],
            }
        )
        capture = intake.register(metadata, self._image_data("lightgray"))
        store = CapturePairStore(
            root / "pair-tasks",
            root / "pair-manifest.jsonl",
            root / "pair-summary.json",
            intake,
        )
        return store, capture["asset_id"]

    @staticmethod
    def _positive_assessment(role: str) -> dict:
        return {
            "same_page": True,
            "page_content_complete": True,
            "capture_usable": True,
            "conditions_confirmed": True,
            "privacy_clear": True,
            "handwriting_state": "present" if role == "handwriting_capture" else "absent",
        }

    def test_sync_save_and_two_person_signoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, capture_id = self._fixture(Path(temporary))
            self.assertEqual(store.sync_from_intake(), 1)
            task = store.load(capture_id)
            self.assertFalse(task["training_allowed"])
            updated = copy.deepcopy(task)
            updated["annotation"]["assessment"] = self._positive_assessment("photo_capture")
            task = store.save(capture_id, updated, payload_hash(task))
            self.assertEqual(task["annotation"]["status"], "draft")
            task = store.signoff(capture_id, "submit_annotation", "annotator-a", payload_hash(task))
            with self.assertRaisesRegex(ValueError, "distinct"):
                store.signoff(capture_id, "approve_review", "annotator-a", payload_hash(task))
            reviewed = copy.deepcopy(task)
            reviewed["review"]["checks"] = {key: True for key in PAIR_CHECK_KEYS}
            task = store.save(capture_id, reviewed, payload_hash(task))
            task = store.signoff(capture_id, "approve_review", "reviewer-b", payload_hash(task))
            self.assertEqual(task["annotation"]["status"], "verified")
            self.assertEqual(store.summary()["verified_pairs"], 1)

    def test_handwriting_role_requires_visible_handwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, capture_id = self._fixture(Path(temporary), "handwriting_capture")
            store.sync_from_intake()
            task = store.load(capture_id)
            updated = copy.deepcopy(task)
            updated["annotation"]["assessment"] = self._positive_assessment("photo_capture")
            task = store.save(capture_id, updated, payload_hash(task))
            with self.assertRaisesRegex(ValueError, "incomplete or inconsistent"):
                store.signoff(capture_id, "submit_annotation", "annotator-a", payload_hash(task))

    def test_assessment_change_invalidates_verified_signoffs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, capture_id = self._fixture(Path(temporary))
            store.sync_from_intake()
            task = store.load(capture_id)
            updated = copy.deepcopy(task)
            updated["annotation"]["assessment"] = self._positive_assessment("photo_capture")
            task = store.save(capture_id, updated, payload_hash(task))
            task = store.signoff(capture_id, "submit_annotation", "annotator-a", payload_hash(task))
            reviewed = copy.deepcopy(task)
            reviewed["review"]["checks"] = {key: True for key in PAIR_CHECK_KEYS}
            task = store.save(capture_id, reviewed, payload_hash(task))
            task = store.signoff(capture_id, "approve_review", "reviewer-b", payload_hash(task))
            changed = copy.deepcopy(task)
            changed["annotation"]["assessment"]["conditions_confirmed"] = False
            changed = store.save(capture_id, changed, payload_hash(task))
            self.assertEqual(changed["annotation"]["status"], "draft")
            self.assertIsNone(changed["review"]["annotator_id"])
            self.assertIsNone(changed["review"]["reviewer_id"])

    def test_strict_target_gate_reports_quantity_and_condition_gaps(self) -> None:
        errors = capture_pair_target_errors(
            {
                "targets": {
                    "clean_pages": {"current": 0, "minimum": 50},
                    "photo_pairs": {"current": 0, "minimum": 200},
                    "handwriting_pairs": {"current": 0, "minimum": 100},
                    "devices": {"current": 0, "minimum": 3},
                },
                "verified_condition_counts": {},
            }
        )
        self.assertEqual(len(errors), 8)
        self.assertTrue(any("photo_pairs=0/200" in error for error in errors))
        self.assertTrue(any("blur_or_compression" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
