from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

import full_benchmark as benchmark
import physics_guidance_pilot as pilot
from physics_guidance_renderer import crop_source_figure


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

    def test_latin_symbol_inside_text_group_is_not_a_chinese_subscript(self) -> None:
        errors: list[str] = []
        pilot.validate_formula_source(
            r"P(A)=\frac{\text{事件A的区域}}{\text{全部区域}}",
            "base_formulas[0].formula",
            errors,
        )
        self.assertFalse(any("中文说明性下标" in item for item in errors), errors)

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
