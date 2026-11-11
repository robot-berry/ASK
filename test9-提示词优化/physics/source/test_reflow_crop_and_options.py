from __future__ import annotations

import unittest

import numpy as np

import test5_preprocess as preprocess


class ReflowCropAndOptionTests(unittest.TestCase):
    def test_wide_question_strip_is_cropped_before_ocr(self) -> None:
        image = np.full((720, 1280, 3), 255, dtype=np.uint8)
        image[280:430, 130:1150] = 225
        image[315:325, 180:1080] = 30
        image[365:375, 180:1040] = 30

        cropped, applied, box = preprocess.crop_outer_whitespace_for_ocr(image)

        self.assertTrue(applied)
        self.assertLess(cropped.shape[0], 360)
        self.assertGreater(cropped.shape[1], 900)
        self.assertLessEqual(box[1], 280)
        self.assertGreaterEqual(box[3], 430)

    def test_full_page_is_not_aggressively_cropped(self) -> None:
        image = np.full((900, 700, 3), 235, dtype=np.uint8)
        image[80:820, 70:630] = 225

        cropped, applied, box = preprocess.crop_outer_whitespace_for_ocr(image)

        self.assertFalse(applied)
        self.assertEqual(cropped.shape, image.shape)
        self.assertEqual(box, (0, 0, 700, 900))

    def test_scientific_notation_exponents_are_restored(self) -> None:
        normalized = preprocess.normalize_reflow_text(
            [
                "A. 93.086×1010",
                "B. 93.086×10{11",
                "C. 9.3086×10¹0",
                "D. 9.3086×1011",
            ]
        )

        self.assertIn("A. 93.086×10¹⁰", normalized)
        self.assertIn("B. 93.086×10¹¹", normalized)
        self.assertIn("C. 9.3086×10¹⁰", normalized)
        self.assertIn("D. 9.3086×10¹¹", normalized)

    def test_merged_options_become_one_option_per_paragraph(self) -> None:
        options = preprocess.split_choice_options(
            "A. 93.086×10¹⁰ B. 93.086×10¹¹ C. 9.3086×10¹⁰ D. 9.3086×10¹¹"
        )

        self.assertEqual(
            options,
            [
                "A. 93.086×10¹⁰",
                "B. 93.086×10¹¹",
                "C. 9.3086×10¹⁰",
                "D. 9.3086×10¹¹",
            ],
        )

    def test_one_missing_exponent_is_recovered_from_complete_option_grid(self) -> None:
        repaired = preprocess.normalize_reflow_text(
            [
                "A. 93.086×10^",
                "B. 93.086×1011",
                "C. 9.3086×1010",
                "D. 9.3086×1011",
            ]
        )

        self.assertIn("A. 93.086×10¹⁰", repaired)

    def test_script_runs_use_plain_digits_for_compact_drawing(self) -> None:
        self.assertEqual(
            preprocess.split_script_runs("9.3×10¹¹m₂"),
            [
                ("normal", "9.3×10"),
                ("sup", "11"),
                ("normal", "m"),
                ("sub", "2"),
            ],
        )

    def test_numeric_range_uses_readable_fullwidth_wave_dash(self) -> None:
        self.assertEqual(preprocess.normalize_reflow_text(["2024年1~2月"]), "2024年1～2月")


if __name__ == "__main__":
    unittest.main()
