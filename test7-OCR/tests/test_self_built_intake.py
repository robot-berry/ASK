from __future__ import annotations

import base64
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from test7_ocr.self_built_intake import SelfBuiltIntakeStore


class SelfBuiltIntakeStoreTests(unittest.TestCase):
    @staticmethod
    def _image_data(color: str = "white") -> str:
        image = Image.new("RGB", (640, 480), color)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _base_metadata(role: str) -> dict:
        return {
            "role": role,
            "subject": "mathematics",
            "scope": "high_school",
            "document_group": "selfbuilt-math-paper-001",
            "page_index": 1,
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

    def _store(self, root: Path) -> SelfBuiltIntakeStore:
        return SelfBuiltIntakeStore(
            root / "dataset",
            root / "records",
            root / "summary.json",
        )

    def test_registers_content_addressed_clean_and_paired_photo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._store(root)
            clean = store.register(self._base_metadata("clean_page"), self._image_data())
            self.assertEqual(clean["partition"], "quarantine")
            self.assertFalse(clean["training_allowed"])
            self.assertFalse(clean["evaluation_allowed"])
            self.assertTrue(store.source_path(clean["asset_id"]).is_file())

            photo_metadata = self._base_metadata("photo_capture")
            photo_metadata.update(
                {
                    "clean_asset_id": clean["asset_id"],
                    "device_code": "device-a",
                    "conditions": ["perspective", "shadow"],
                }
            )
            photo = store.register(photo_metadata, self._image_data("#eeeeee"))
            self.assertEqual(photo["metadata"]["clean_asset_id"], clean["asset_id"])
            summary, errors = store.check()
            self.assertEqual(errors, [])
            self.assertEqual(summary["counts"]["clean_page"], 1)
            self.assertEqual(summary["counts"]["photo_capture"], 1)
            self.assertEqual(summary["unique_devices"], 1)

    def test_boundary_page_requires_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store(Path(temporary))
            metadata = self._base_metadata("boundary_page")
            with self.assertRaisesRegex(ValueError, "explicit page layout"):
                store.register(metadata, self._image_data())
            metadata["page_layout"] = "double_column"
            record = store.register(metadata, self._image_data())
            self.assertEqual(record["metadata"]["page_layout"], "double_column")

    def test_capture_requires_existing_matching_clean_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store(Path(temporary))
            metadata = self._base_metadata("handwriting_capture")
            metadata.update(
                {
                    "clean_asset_id": "SBI-CLEAN-0000000000000000",
                    "device_code": "device-a",
                    "conditions": ["frontal"],
                }
            )
            with self.assertRaises(FileNotFoundError):
                store.register(metadata, self._image_data())

    def test_privacy_and_license_gates_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = self._store(Path(temporary))
            metadata = self._base_metadata("clean_page")
            metadata["privacy_checked"] = False
            with self.assertRaisesRegex(ValueError, "Privacy review"):
                store.register(metadata, self._image_data())
            metadata["privacy_checked"] = True
            metadata["provenance"] = "licensed"
            with self.assertRaisesRegex(ValueError, "license reference"):
                store.register(metadata, self._image_data())

    def test_duplicate_bytes_do_not_create_duplicate_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = self._store(root)
            metadata = self._base_metadata("clean_page")
            first = store.register(metadata, self._image_data())
            second = store.register(metadata, self._image_data())
            self.assertEqual(first["asset_id"], second["asset_id"])
            self.assertEqual(len(list((root / "dataset" / "sources").iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
