import unittest

from PIL import Image, ImageDraw

from test7_ocr.layout import detect_visual_bboxes
from test7_ocr.image_ops import photographed_page_risk
from test7_ocr.types import BBox, RecognizedBlock


class LayoutTests(unittest.TestCase):
    def test_photo_page_risk_is_conservative(self) -> None:
        clean = Image.new("RGB", (300, 180), "white")
        clean_risk, _ = photographed_page_risk(clean, BBox(0, 0, 300, 180))
        photographed = Image.new("RGB", (300, 180), (190, 145, 105))
        photo_risk, _ = photographed_page_risk(photographed, BBox(0, 0, 300, 180))
        self.assertFalse(clean_risk)
        self.assertTrue(photo_risk)

    def test_detects_large_diagram_after_text_is_removed(self) -> None:
        image = Image.new("RGB", (800, 500), "white")
        draw = ImageDraw.Draw(image)
        for y in (80, 120, 160, 200):
            draw.rectangle((40, y, 420, y + 12), fill="black")
        draw.rectangle((520, 160, 740, 390), outline="black", width=8)
        draw.line((520, 390, 740, 160), fill="black", width=8)
        blocks = [RecognizedBlock("text", BBox(35, y - 4, 395, 24), 0.99) for y in (80, 120, 160, 200)]
        boxes = detect_visual_bboxes(image, blocks, {})
        self.assertTrue(boxes)
        self.assertGreaterEqual(boxes[0].x, 480)

    def test_gray_photo_fallback_selects_only_current_numbered_figure(self) -> None:
        image = Image.new("RGB", (1280, 720), (190, 190, 190))
        draw = ImageDraw.Draw(image)
        draw.rectangle((105, 120, 615, 420), outline="black", width=7)
        draw.line((140, 380, 560, 165), fill="black", width=6)
        draw.rectangle((730, 125, 1190, 425), outline="black", width=7)
        draw.line((790, 180, 1110, 380), fill="black", width=6)
        blocks = [
            RecognizedBlock(
                "（第3题图）",
                BBox(264, 435, 134, 35),
                0.96,
                "text",
                "adjacent_question",
            ),
            RecognizedBlock(
                "（第4题图）",
                BBox(836, 438, 132, 36),
                0.95,
                "text",
                "adjacent_question",
            ),
            RecognizedBlock(
                "4. 某函数的图像如图所示",
                BBox(92, 494, 589, 35),
                0.99,
                "text",
            ),
        ]
        diagnostics: dict[str, object] = {}

        boxes = detect_visual_bboxes(
            image,
            blocks,
            {},
            question_number="4",
            diagnostics=diagnostics,
        )

        self.assertEqual(len(boxes), 1)
        self.assertGreater(boxes[0].x, 650)
        self.assertGreaterEqual(boxes[0].bottom, 474)
        self.assertEqual(diagnostics["method"], "adaptive_numbered_caption_fallback")

    def test_gray_photo_fallback_does_not_guess_without_matching_caption(self) -> None:
        image = Image.new("RGB", (800, 500), (190, 190, 190))
        draw = ImageDraw.Draw(image)
        draw.rectangle((150, 80, 650, 330), outline="black", width=7)
        blocks = [
            RecognizedBlock("（第3题图）", BBox(320, 350, 120, 30), 0.99, "text")
        ]

        boxes = detect_visual_bboxes(image, blocks, {}, question_number="4")

        self.assertEqual(boxes, [])


if __name__ == "__main__":
    unittest.main()
