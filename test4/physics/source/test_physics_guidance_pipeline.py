from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

import full_benchmark as benchmark
import physics_guidance_pilot as pilot
from physics_guidance_renderer import (
    BODY_SIZE,
    CONTENT_WIDTH,
    DEFAULT_SONG,
    DEFAULT_SONG_BOLD,
    DEFAULT_TIMES,
    DEFAULT_TIMES_BOLD,
    crop_source_figure,
    crop_wide_source_figure_panel,
    detect_missing_source_figure_bbox,
    make_fonts,
    prepare_aligned_figure_group_crops,
    reflow_choice_options,
)


class ChoiceOptionLayoutTests(unittest.TestCase):
    def test_physics_choices_use_two_rows(self) -> None:
        image = Image.new("L", (800, 300), "white")
        draw = ImageDraw.Draw(image)
        fonts = make_fonts(
            BODY_SIZE,
            song_path=DEFAULT_SONG,
            song_bold_path=DEFAULT_SONG_BOLD,
            times_path=DEFAULT_TIMES,
            times_bold_path=DEFAULT_TIMES_BOLD,
        )
        result = reflow_choice_options(
            draw,
            "关于速度，下列正确的是（ ） A. 甲 B. 乙 C. 丙 D. 丁",
            fonts,
            max_width=CONTENT_WIDTH,
        )
        rows = result.splitlines()
        self.assertEqual(len(rows), 3)
        self.assertTrue(rows[1].startswith("A. 甲"))
        self.assertIn("B. 乙", rows[1])
        self.assertTrue(rows[2].startswith("C. 丙"))
        self.assertIn("D. 丁", rows[2])


class RequestImageCoordinateTests(unittest.TestCase):
    def make_source(self) -> bytes:
        image = Image.new("RGB", (2000, 1000), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((700, 300, 1299, 699), fill="black")
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def test_pilot_keeps_complete_frame_for_bbox_coordinates(self) -> None:
        source = self.make_source()
        with patch.object(Path, "read_bytes", return_value=source):
            data_url, audit = pilot.image_data_url(Path("unused.png"))

        self.assertTrue(data_url.startswith("data:image/jpeg;base64,"))
        self.assertFalse(audit["crop_applied"])
        self.assertEqual(audit["crop_box_xyxy"], [0, 0, 2000, 1000])
        self.assertEqual((audit["width"], audit["height"]), (1280, 640))
        self.assertTrue(audit["resized"])
        self.assertFalse(audit["binarized"])

    def test_normalized_bbox_still_targets_original_source(self) -> None:
        source = Image.open(io.BytesIO(self.make_source()))
        crop, audit = crop_source_figure(
            source,
            [350, 300, 650, 700],
            lock_bbox=True,
        )
        self.assertEqual(audit["model_bbox_1000"], [350.0, 300.0, 650.0, 700.0])
        self.assertEqual(audit["source_crop_size_px"], [600, 400])
        self.assertGreater(crop.width, 0)
        self.assertGreater(crop.height, 0)

    def test_nearby_figure_recovers_from_empty_model_bbox(self) -> None:
        image = Image.new("RGB", (1000, 600), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((720, 410, 770, 520), outline="black", width=4)
        crop, audit = crop_source_figure(
            image,
            [800, 650, 950, 900],
        )
        self.assertEqual(audit["mode"], "nearby_ink_recovered")
        self.assertGreater(crop.width, 0)
        self.assertGreater(crop.height, 0)

    def test_empty_bbox_and_empty_search_are_still_rejected(self) -> None:
        image = Image.new("RGB", (1000, 600), "white")
        with self.assertRaisesRegex(ValueError, "insufficient visible ink"):
            crop_source_figure(image, [800, 650, 950, 900])

    def test_aligned_multi_panel_figures_render_as_one_source_crop(self) -> None:
        image = Image.new("RGB", (1200, 700), "white")
        draw = ImageDraw.Draw(image)
        for left in (210, 510, 810):
            draw.rectangle((left, 160, left + 170, 360), outline="black", width=6)
            draw.line((left + 20, 330, left + 150, 190), fill="black", width=6)
        figures = [
            {
                "figure_id": f"fig_{index + 1}",
                "bbox_1000": bbox,
                "caption": f"第16题图{index + 1}",
            }
            for index, bbox in enumerate(
                ([0, 300, 400, 700], [400, 300, 700, 700], [700, 300, 1000, 700])
            )
        ]

        prepared, members = prepare_aligned_figure_group_crops(image, figures)

        self.assertEqual(members, {"fig_1": "fig_1", "fig_2": "fig_1", "fig_3": "fig_1"})
        crop, audit = prepared["fig_1"]
        self.assertTrue(audit["aligned_figure_group"])
        self.assertEqual(audit["merged_figure_ids"], ["fig_1", "fig_2", "fig_3"])
        self.assertGreater(crop.width, crop.height * 2)

    def test_single_wide_panel_removes_text_below_diagrams(self) -> None:
        image = Image.new("RGB", (1200, 700), "white")
        draw = ImageDraw.Draw(image)
        for left in (210, 510, 810):
            draw.rectangle((left, 160, left + 170, 350), outline="black", width=6)
            draw.line((left + 20, 320, left + 150, 190), fill="black", width=6)
        for y in (410, 445, 480, 515):
            draw.line((180, y, 1010, y), fill="black", width=3)

        crop, audit = crop_wide_source_figure_panel(image, [0, 450, 1000, 780])

        self.assertTrue(audit["wide_figure_panel_refined"])
        self.assertGreater(crop.width, crop.height * 2)
        self.assertLess(audit["refined_bbox_px"][3], 410)

    def test_missing_figure_detector_finds_three_diagram_row(self) -> None:
        image = Image.new("RGB", (1200, 700), "white")
        draw = ImageDraw.Draw(image)
        for left in (210, 510, 810):
            draw.rectangle((left, 180, left + 170, 380), outline="black", width=6)
            draw.line((left + 20, 350, left + 150, 210), fill="black", width=6)

        bbox = detect_missing_source_figure_bbox(image, "如图1、图2、图3所示")

        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertLess(bbox[0], 200)
        self.assertGreater(bbox[2], 800)


class PreprocessGuardTests(unittest.TestCase):
    def test_binarization_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "二值化"):
            benchmark.prepare_request_image(
                self.make_image(),
                "image/png",
                {"image_preprocess": {"enabled": True, "binarize": True}},
            )

    def test_model_unit_wrappers_become_textbook_units(self) -> None:
        self.assertEqual(
            pilot.normalize_formula_notation(r"R=1.0~{m}"),
            r"R=1.0\mathrm{m}",
        )
        self.assertEqual(
            pilot.normalize_formula_notation(r"g=10~{\mathrm{m\,s^{-2}}}"),
            r"g=10\mathrm{m\,s^{-2}}",
        )

    def test_infinity_subscript_is_normalized(self) -> None:
        self.assertEqual(
            pilot.normalize_formula_notation("E_∞∝x"),
            r"E_{\infty}\propto x",
        )

    def test_missing_figure_placeholder_is_inserted_in_source_order(self) -> None:
        output = {
            "question_blocks": [
                {"type": "text", "text": "题干", "figure_id": ""},
                {"type": "figure", "text": "", "figure_id": "fig_1"},
                {"type": "text", "text": "小问", "figure_id": ""},
            ],
            "source_figures": [
                {"figure_id": "fig_1", "bbox_1000": [1, 1, 2, 2]},
                {"figure_id": "fig_2", "bbox_1000": [3, 3, 4, 4]},
            ],
        }
        audit: list[dict[str, object]] = []
        repaired = pilot.align_question_figure_blocks(
            output,
            ["question_blocks与source_figures的figure_id及顺序必须完全一致"],
            audit,
        )
        observed = [
            block["figure_id"]
            for block in repaired["question_blocks"]
            if block["type"] == "figure"
        ]
        self.assertEqual(observed, ["fig_1", "fig_2"])
        self.assertEqual(audit[0]["action"], "insert_missing_figure_block")

    @staticmethod
    def make_image() -> bytes:
        output = io.BytesIO()
        Image.new("RGB", (300, 200), "white").save(output, format="PNG")
        return output.getvalue()


if __name__ == "__main__":
    unittest.main()
