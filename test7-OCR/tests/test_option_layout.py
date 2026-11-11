from __future__ import annotations

import unittest

from test7_ocr.pipeline import load_config
from test7_ocr.option_layout import has_standard_option_sequence, option_grid_columns
from test7_ocr.render import render_question
from test7_ocr.types import QuestionDocument

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class OptionLayoutTests(unittest.TestCase):
    def test_very_short_options_are_capped_at_two_columns_for_readback(self) -> None:
        options = [
            {"label": label, "text": text}
            for label, text in zip("ABCD", ["甲", "乙", "丙", "丁"])
        ]
        self.assertEqual(option_grid_columns(options, 704, 29), 2)

    def test_medium_formula_options_use_two_columns(self) -> None:
        options = [
            {"label": label, "text": text}
            for label, text in zip("ABCD", ["(-2,0)", "(-1,+∞)", "(-1,0)", "(-∞,-2)"])
        ]
        self.assertEqual(option_grid_columns(options, 704, 29), 2)

    def test_long_options_remain_one_per_line(self) -> None:
        options = [
            {"label": label, "text": "这是一个很长的选项，包含多个不能拥挤排列的中文条件"}
            for label in "ABCD"
        ]
        self.assertEqual(option_grid_columns(options, 704, 29), 1)

    def test_duplicate_or_out_of_order_labels_are_not_grid_candidates(self) -> None:
        options = [
            {"label": "A", "text": "甲"},
            {"label": "A", "text": "乙"},
            {"label": "D", "text": "丁"},
        ]
        self.assertFalse(has_standard_option_sequence(options))
        self.assertEqual(option_grid_columns(options, 704, 29), 1)

    def test_empty_option_is_not_a_grid_candidate(self) -> None:
        options = [
            {"label": "A", "text": ""},
            {"label": "B", "text": "被误切出来的正文"},
        ]
        self.assertFalse(has_standard_option_sequence(options))
        self.assertEqual(option_grid_columns(options, 704, 29), 1)

    def test_pillow_fallback_wraps_single_column_options(self) -> None:
        long_text = "这是一个很长的选项，包含多个不能拥挤排列的中文条件和补充说明"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="fixture",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="1. 请选择",
            display_text="1. 请选择",
            question_number="1",
            stem="1. 请选择",
            options=[{"label": label, "text": long_text} for label in "ABCD"],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        output, metadata = render_question(document, [], config)
        self.assertEqual(output.width, 800)
        self.assertGreater(metadata["line_count"], 4)


if __name__ == "__main__":
    unittest.main()
