from __future__ import annotations

import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from preprocessor import Test5QuestionImageProcessor


class Test5FastPreviewTests(unittest.TestCase):
    def test_fast_preview_uses_test5_pre_ocr_stages_without_running_ocr(self) -> None:
        page = Image.new("RGB", (640, 360), (242, 240, 235))
        buffer = io.BytesIO()
        page.save(buffer, format="PNG")
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary, patch(
            "preprocessor.test5_adapter.test5.RapidOCR",
            return_value=object(),
        ), patch(
            "preprocessor.test5_adapter.test5.run_ocr",
            side_effect=AssertionError("fast preview must not run OCR"),
        ):
            processor = Test5QuestionImageProcessor(Path(temporary) / "preprocess")
            preview, audit = processor.fast_preview(buffer.getvalue())

        try:
            self.assertEqual(preview.size, (640, 360))
            self.assertEqual(
                audit["generation_method"],
                "test5-pre-ocr-geometry-light",
            )
            self.assertFalse(audit["ocr_executed"])
            self.assertGreaterEqual(audit["elapsed_ms"], 0.0)
        finally:
            preview.close()


if __name__ == "__main__":
    unittest.main()
