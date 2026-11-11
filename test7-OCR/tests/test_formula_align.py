from __future__ import annotations

import unittest

from test7_ocr.external_eval import _formula_match
from test7_ocr.formula_crops import formula_mfr_candidates
from test7_ocr.text import canonicalize_formula, formulas_agree, formula_lists_agree


class FormulaAlignTests(unittest.TestCase):
    def test_whole_image_candidate_when_layout_misses_or_is_tiny(self) -> None:
        layout = [{"label": "ocr_text", "bbox": [1, 1, 10, 8], "score": 0.9}]
        candidates, source = formula_mfr_candidates(
            layout, 128, 64, whole_image_formula=True
        )
        self.assertEqual(source, "whole_image")
        self.assertEqual(candidates[0]["bbox"], [0, 0, 128, 64])

        tiny = [{"label": "inline_formula", "bbox": [2, 2, 8, 7], "score": 0.4}]
        candidates, source = formula_mfr_candidates(
            tiny, 128, 64, whole_image_formula=True
        )
        self.assertEqual(source, "whole_image")

        large = [{"label": "display_formula", "bbox": [4, 2, 120, 60], "score": 0.9}]
        candidates, source = formula_mfr_candidates(
            large, 128, 64, whole_image_formula=True
        )
        self.assertEqual(source, "layout")
        self.assertEqual(candidates[0]["bbox"], [4, 2, 120, 60])

    def test_default_recognize_formulas_keeps_layout_boxes(self) -> None:
        layout = [{"label": "inline_formula", "bbox": [2, 2, 8, 7], "score": 0.4}]
        candidates, source = formula_mfr_candidates(
            layout, 128, 64, whole_image_formula=False
        )
        self.assertEqual(source, "layout")
        self.assertEqual(len(candidates), 1)

    def test_canonicalize_hits_equivalent_latex_not_unrelated_formulas(self) -> None:
        gold = r"e ^ { i \theta } = { \frac { p - q \tau ^ { * } } { | p - q \tau ^ { * } | } }"
        pred = r"e ^ { i \theta } = \frac { p - q \tau ^ { * } } { | p - q \tau ^ { * } | } ,"
        self.assertTrue(formulas_agree(gold, pred))
        self.assertGreater(len(canonicalize_formula(gold)), 3)
        self.assertFalse(
            formulas_agree(
                r"\tilde{\alpha}=\alpha\beta^{-m}",
                r"e^{i\theta}=\frac{p-q\tau^{*}}{|p-q\tau^{*}|}",
            )
        )
        self.assertFalse(formulas_agree("a", "ab"))
        self.assertEqual(_formula_match([pred], [gold]), 1.0)
        self.assertEqual(
            _formula_match(
                [r"e^{i\theta}=\frac{p-q\tau^{*}}{|p-q\tau^{*}|}"],
                [r"\tilde{\alpha}=\alpha\beta^{-m}"],
            ),
            0.0,
        )
        self.assertTrue(formula_lists_agree([pred], [gold]))
        self.assertFalse(formula_lists_agree([pred], []))
