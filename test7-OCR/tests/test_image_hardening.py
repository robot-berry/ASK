from __future__ import annotations

import io
import unittest
from pathlib import Path
import tempfile

from PIL import Image, ImageFile

from test7_ocr.image_ops import (
    RISK_PROBE_MAX_SIDE,
    direct_crop,
    open_source_image,
    photographed_page_risk,
)
from test7_ocr.types import BBox


class ImageHardeningTests(unittest.TestCase):
    def test_empty_crop_returns_none(self) -> None:
        image = Image.new("RGB", (80, 60), "white")
        self.assertIsNone(direct_crop(image, BBox(10, 10, 0, 20)))
        self.assertIsNone(direct_crop(image, BBox(90, 10, 20, 20)))

    def test_truncated_jpeg_decodes_with_flag(self) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (160, 160), "white").save(buffer, format="JPEG", quality=95)
        data = buffer.getvalue()
        truncated = data[:-120]
        self.assertGreater(len(truncated), 200)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "truncated.jpg"
            path.write_bytes(truncated)
            previous = ImageFile.LOAD_TRUNCATED_IMAGES
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            try:
                with self.assertRaises(OSError):
                    with Image.open(path) as broken:
                        broken.load()
                image = open_source_image(path)
            finally:
                ImageFile.LOAD_TRUNCATED_IMAGES = previous
            self.assertEqual(image.mode, "RGB")
            self.assertTrue(image.info.get("truncated_decoded"))

    def test_risk_probe_downscales_huge_pages(self) -> None:
        image = Image.new("RGB", (2200, 1800), (190, 145, 105))
        risk, metrics = photographed_page_risk(image, BBox(0, 0, 2200, 1800))
        self.assertTrue(risk)
        self.assertEqual(metrics["probe_downscaled"], 1.0)
        self.assertEqual(metrics["original_width"], 2200.0)
        self.assertGreater(RISK_PROBE_MAX_SIDE, 0)
