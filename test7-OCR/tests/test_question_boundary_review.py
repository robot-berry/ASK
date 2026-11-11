from __future__ import annotations

import base64
import copy
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from test7_ocr.question_boundary_review import (
    BOUNDARY_CHECK_KEYS,
    QuestionBoundaryStore,
    boundary_target_errors,
    payload_hash,
)
from test7_ocr.self_built_intake import SelfBuiltIntakeStore


class QuestionBoundaryStoreTests(unittest.TestCase):
    @staticmethod
    def _image_data() -> str:
        image = Image.new("RGB", (800, 1000), "white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("ascii")

    def _fixture(self, root: Path) -> tuple[QuestionBoundaryStore, str]:
        intake = SelfBuiltIntakeStore(root / "dataset", root / "intake-records", root / "intake-summary.json")
        metadata = {
            "role": "boundary_page",
            "subject": "physics",
            "scope": "high_school",
            "document_group": "physics-paper-001",
            "page_index": 1,
            "clean_asset_id": None,
            "device_code": None,
            "conditions": [],
            "page_layout": "double_column",
            "provenance": "self_created",
            "license_reference": None,
            "privacy_checked": True,
            "contains_personal_data": False,
            "distribution_allowed": False,
        }
        record = intake.register(metadata, self._image_data())
        store = QuestionBoundaryStore(
            root / "boundary-tasks",
            root / "boundary-manifest.jsonl",
            root / "boundary-summary.json",
            intake,
        )
        return store, record["asset_id"]

    @staticmethod
    def _questions() -> list[dict]:
        return [
            {
                "question_id": "1",
                "bbox": [20, 30, 350, 400],
                "reading_order": 0,
                "parent_question_id": None,
                "column": "left",
                "continues_on_next_page": False,
                "notes": "",
            },
            {
                "question_id": "2",
                "bbox": [420, 40, 350, 420],
                "reading_order": 1,
                "parent_question_id": None,
                "column": "right",
                "continues_on_next_page": False,
                "notes": "",
            },
        ]

    def test_sync_save_and_two_person_signoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, asset_id = self._fixture(Path(temporary))
            self.assertEqual(store.sync_from_intake(), 1)
            task = store.load(asset_id)
            self.assertFalse(task["training_allowed"])
            updated = copy.deepcopy(task)
            updated["annotation"]["questions"] = self._questions()
            task = store.save(asset_id, updated, payload_hash(task))
            self.assertEqual(task["annotation"]["status"], "draft")
            task = store.signoff(asset_id, "submit_annotation", "annotator-a", payload_hash(task))
            with self.assertRaisesRegex(ValueError, "distinct"):
                store.signoff(asset_id, "approve_review", "annotator-a", payload_hash(task))
            checked = copy.deepcopy(task)
            checked["review"]["checks"] = {key: True for key in BOUNDARY_CHECK_KEYS}
            task = store.save(asset_id, checked, payload_hash(task))
            task = store.signoff(asset_id, "approve_review", "reviewer-b", payload_hash(task))
            self.assertEqual(task["annotation"]["status"], "verified")
            self.assertEqual(store.summary()["verified_questions"], 2)

    def test_boundary_change_invalidates_signoffs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, asset_id = self._fixture(Path(temporary))
            store.sync_from_intake()
            task = store.load(asset_id)
            updated = copy.deepcopy(task)
            updated["annotation"]["questions"] = self._questions()
            task = store.save(asset_id, updated, payload_hash(task))
            task = store.signoff(asset_id, "submit_annotation", "annotator-a", payload_hash(task))
            checked = copy.deepcopy(task)
            checked["review"]["checks"] = {key: True for key in BOUNDARY_CHECK_KEYS}
            task = store.save(asset_id, checked, payload_hash(task))
            task = store.signoff(asset_id, "approve_review", "reviewer-b", payload_hash(task))
            changed = copy.deepcopy(task)
            changed["annotation"]["questions"][0]["bbox"][2] += 10
            changed = store.save(asset_id, changed, payload_hash(task))
            self.assertEqual(changed["annotation"]["status"], "draft")
            self.assertIsNone(changed["review"]["annotator_id"])
            self.assertIsNone(changed["review"]["reviewer_id"])

    def test_rejects_out_of_bounds_and_noncontiguous_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store, asset_id = self._fixture(Path(temporary))
            store.sync_from_intake()
            task = store.load(asset_id)
            task["annotation"]["questions"] = self._questions()
            task["annotation"]["questions"][1]["bbox"] = [700, 900, 200, 200]
            with self.assertRaisesRegex(ValueError, "escapes"):
                store.validate(task)
            task["annotation"]["questions"] = self._questions()
            task["annotation"]["questions"][1]["reading_order"] = 3
            with self.assertRaisesRegex(ValueError, "contiguous"):
                store.validate(task)

    def test_strict_target_gate_reports_every_missing_boundary_target(self) -> None:
        missing = boundary_target_errors(
            {
                "targets": {
                    "pages": {"current": 0, "minimum": 60},
                    "questions": {"current": 0, "minimum": 300},
                    "single_column": {"current": 0, "minimum": 30},
                    "double_column": {"current": 0, "minimum": 30},
                }
            }
        )
        self.assertEqual(len(missing), 4)
        self.assertTrue(any("pages=0/60" in error for error in missing))
        self.assertTrue(any("questions=0/300" in error for error in missing))
        self.assertEqual(
            boundary_target_errors(
                {
                    "targets": {
                        "pages": {"current": 60, "minimum": 60},
                        "questions": {"current": 300, "minimum": 300},
                        "single_column": {"current": 30, "minimum": 30},
                        "double_column": {"current": 30, "minimum": 30},
                    }
                }
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
