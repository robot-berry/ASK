import unittest

from test7_ocr.warmup import (
    STRUCTURED_WARMUP_PROFILE_VERSION,
    synthetic_structured_warmup_pages,
)


class StructuredWarmupTests(unittest.TestCase):
    def test_warmup_uses_synthetic_fixed_size_pages(self) -> None:
        pages = synthetic_structured_warmup_pages()
        self.assertEqual(
            STRUCTURED_WARMUP_PROFILE_VERSION,
            "synthetic_runtime_shapes_v7_scoped_vram_preclean",
        )
        self.assertEqual(len(pages), 5)
        self.assertTrue(all(page.size == (1280, 720) for page in pages))
        self.assertTrue(all(page.mode == "RGB" for page in pages))
        self.assertTrue(all(page.getbbox() is not None for page in pages))

        # The long-row page must actually contain ink close to both horizontal
        # edges; otherwise it cannot exercise the wide recognizer shape class.
        long_row_bbox = pages[3].convert("L").point(lambda value: 255 - value).getbbox()
        self.assertIsNotNone(long_row_bbox)
        assert long_row_bbox is not None
        self.assertLess(long_row_bbox[0], 40)
        self.assertGreater(long_row_bbox[2], 1200)


if __name__ == "__main__":
    unittest.main()
