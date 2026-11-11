import unittest

from formula_ocr_service import (
    layout_tokens,
    likely_handwritten_item_indexes,
    public_layout_regions,
    safe_bbox,
    structured_transcript,
)


class FormulaOcrTranscriptTests(unittest.TestCase):
    def test_spatial_merge_preserves_text_formula_and_line_order(self):
        items = [
            {"label": "ocr_text", "bbox": [10, 10, 90, 35], "score": 0.99, "text": "设函数"},
            {"label": "inline_formula", "bbox": [100, 8, 220, 38], "score": 0.95, "latex": r"f(x)=\frac{1}{x}"},
            {"label": "ocr_text", "bbox": [10, 60, 80, 85], "score": 0.99, "text": "则"},
            {"label": "inline_formula", "bbox": [90, 58, 170, 88], "score": 0.95, "latex": r"f(2)="},
        ]

        transcript, formulas = structured_transcript(items)

        self.assertEqual(transcript, "设函数 $f(x)=\\frac{1}{x}$\n则 $f(2)=$")
        self.assertEqual(formulas, [r"f(x)=\frac{1}{x}", "f(2)="])

    def test_low_confidence_and_invalid_items_are_dropped(self):
        items = [
            {"label": "ocr_text", "bbox": [0, 0, 10, 10], "score": 0.49, "text": "错字"},
            {"label": "inline_formula", "bbox": [10, 0, 30, 10], "score": 0.34, "latex": "x^2"},
            {"label": "table", "bbox": [0, 0, 20, 20], "score": 0.99, "text": "忽略"},
        ]

        self.assertEqual(layout_tokens(items), [])
        self.assertEqual(structured_transcript(items), ("", []))

    def test_bbox_validation(self):
        self.assertEqual(safe_bbox([1.2, 2.4, 10.7, 20.1]), (1, 2, 11, 20))
        self.assertIsNone(safe_bbox([0, 0, 0, 2]))
        self.assertIsNone(safe_bbox("0,0,1,1"))

    def test_visual_layout_regions_are_bounded_and_sorted(self):
        items = [
            {"label": "table", "bbox": [-5, 80, 210, 160], "score": 0.91,
             "html": "<table><tr><td>1</td></tr></table>"},
            {"label": "image", "bbox": [20, 10, 100, 70], "score": 0.88},
            {"label": "ocr_text", "bbox": [0, 0, 80, 10], "score": 0.99},
            {"label": "chart", "bbox": [1, 1, 2, 2], "score": 0.99},
            {"label": "image", "bbox": [0, 0, 50, 50], "score": 0.10},
        ]

        regions = public_layout_regions(
            items,
            image_width=200,
            image_height=180,
        )

        self.assertEqual([region["label"] for region in regions], ["image", "table"])
        self.assertEqual(regions[1]["bbox"], [0, 80, 200, 160])
        self.assertIn("<table>", regions[1]["html"])

    def test_low_confidence_trailing_display_answer_is_dropped(self):
        items = [
            {"label": "ocr_text", "bbox": [10, 10, 130, 35], "score": 0.99, "text": "17. 计算："},
            {"label": "inline_formula", "bbox": [140, 8, 500, 40], "score": 0.84, "latex": r"\sqrt{12}+|2-\sqrt3|-(\frac12)^{-1}"},
            {"label": "display_formula", "bbox": [280, 60, 380, 100], "score": 0.48, "latex": r"=\sqrt3"},
        ]

        transcript, formulas = structured_transcript(items)

        self.assertNotIn("=\\sqrt3", transcript)
        self.assertEqual(formulas, [r"\sqrt{12}+|2-\sqrt3|-(\frac12)^{-1}"])
        self.assertEqual(likely_handwritten_item_indexes(items), {2})

    def test_algebra_between_answer_cue_and_options_is_dropped(self):
        items = [
            {"label": "ocr_text", "bbox": [10, 10, 500, 40], "score": 0.99, "text": "6. 方程有两个实数根"},
            {"label": "ocr_text", "bbox": [20, 50, 120, 80], "score": 1.0, "text": "范围是"},
            {"label": "ocr_text", "bbox": [200, 45, 430, 85], "score": 0.67, "text": "4m2-4m2-4(M-2)>0"},
            {"label": "ocr_text", "bbox": [440, 50, 540, 82], "score": 0.83, "text": "x2-2=0"},
            {"label": "ocr_text", "bbox": [20, 95, 520, 125], "score": 0.99, "text": "A. m>2 B. m≥2 C. m<2 D. m≠0"},
        ]

        transcript, _formulas = structured_transcript(items)

        self.assertNotIn("4m2", transcript)
        self.assertNotIn("x2-2", transcript)
        self.assertIn("范围是", transcript)
        self.assertIn("A. m>2", transcript)


if __name__ == "__main__":
    unittest.main()
