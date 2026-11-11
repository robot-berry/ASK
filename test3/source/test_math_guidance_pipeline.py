from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont, ImageOps

import full_benchmark as benchmark
import math_guidance_pilot as pilot
import rerender_test1_phy as phy_rerender
from math_guidance_renderer import crop_source_figure
from result_image_store import clean_text, formula_vertical_bounds


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

    def test_partial_table_bbox_recovers_all_rows_and_closed_border(self) -> None:
        image = Image.new("RGB", (1000, 700), "white")
        draw = ImageDraw.Draw(image)
        # Four-row table.  The model bbox ends in the first data row and is
        # therefore far too short for the ordinary 18% figure expansion.
        for y in (180, 250, 320, 390, 460):
            draw.line((180, y, 620, y), fill="black", width=3)
        for x in (180, 330, 470, 620):
            draw.line((x, 180, x, 460), fill="black", width=3)
        draw.text((205, 270), "120", fill="black")
        draw.text((490, 410), "400", fill="black")

        crop, audit = crop_source_figure(
            image,
            [165, 245, 635, 465],
        )

        self.assertEqual(audit["mode"], "closed_table_border_recovered")
        self.assertLessEqual(audit["refined_bbox_px"][1], 180)
        self.assertGreater(audit["refined_bbox_px"][3], 460)
        rule_rows = audit["closed_table_detection"]["horizontal_rule_rows_px"]
        self.assertEqual(len(rule_rows), 5)
        for observed, expected in zip(rule_rows, (180, 250, 320, 390, 460)):
            self.assertLessEqual(abs(observed - expected), 1)
        # The bottom border is still visible in the delivered resized crop.
        gray = ImageOps.grayscale(crop)
        lower_half = gray.crop((0, crop.height // 2, crop.width, crop.height))
        self.assertLess(min(lower_half.getdata()), 40)

    def test_nonoverlapping_table_does_not_capture_formula_line(self) -> None:
        image = Image.new("RGB", (1000, 700), "white")
        draw = ImageDraw.Draw(image)
        draw.line((180, 150, 600, 150), fill="black", width=3)
        # A complete table exists lower on the page, but the formula model box
        # has no vertical overlap with it and must never be expanded into it.
        for y in (330, 390, 450):
            draw.line((180, y, 520, y), fill="black", width=3)
        for x in (180, 350, 520):
            draw.line((x, 330, x, 450), fill="black", width=3)

        _, audit = crop_source_figure(image, [160, 175, 620, 260])

        self.assertNotEqual(audit["mode"], "closed_table_border_recovered")
        self.assertLess(audit["refined_bbox_px"][3], 330)

    def test_ordinary_geometry_never_expands_into_crossing_prose(self) -> None:
        image = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(image)
        # One connected geometry figure safely inside the semantic model box.
        draw.polygon(
            ((220, 300), (420, 280), (460, 520), (250, 520)),
            outline="black",
            width=4,
        )
        draw.line((190, 520, 420, 280), fill="black", width=4)
        # Adjacent question prose and working cross the top/right model edges
        # by a few pixels.  Projection refinement used to pull in the complete
        # outside components.
        draw.rectangle((205, 215, 560, 246), outline="black", width=3)
        draw.rectangle((488, 310, 560, 380), outline="black", width=4)

        _, audit = crop_source_figure(image, [180, 240, 500, 560])

        self.assertGreaterEqual(audit["refined_bbox_px"][0], 180)
        self.assertGreaterEqual(audit["refined_bbox_px"][1], 240)
        self.assertLessEqual(audit["refined_bbox_px"][2], 500)
        self.assertLessEqual(audit["refined_bbox_px"][3], 560)
        self.assertNotEqual(audit["mode"], "closed_table_border_recovered")

    def test_clipped_word_like_right_edge_fragment_is_suppressed(self) -> None:
        image = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(image)
        draw.polygon(
            ((220, 300), (420, 280), (460, 520), (250, 520)),
            outline="black",
            width=5,
        )
        draw.line((190, 520, 420, 280), fill="black", width=5)
        # A disconnected, word-sized ink component is clipped by the model's
        # right edge, mirroring a handwritten “解” beside a printed diagram.
        draw.rectangle((470, 285, 530, 355), outline="black", width=5)
        draw.line((470, 285, 505, 355), fill="black", width=5)
        draw.line((470, 355, 505, 285), fill="black", width=5)

        crop, audit = crop_source_figure(image, [180, 240, 500, 560])

        guard = audit["edge_fragment_guard"]
        self.assertTrue(guard["applied"])
        self.assertEqual(
            guard["reason"],
            "unique_clipped_word_like_component_suppressed_at_upper_right_edge",
        )
        self.assertEqual(audit["refined_bbox_px"][2], 500)
        self.assertTrue(audit["edge_fragment_suppression_applied"])
        # The rectangular crop remains full-width, so lower-right geometry is
        # not sacrificed merely because the unwanted block sits above it.
        self.assertEqual(audit["source_crop_size_px"][0], 320)
        self.assertGreater(crop.width, 0)


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

    @staticmethod
    def make_image() -> bytes:
        output = io.BytesIO()
        Image.new("RGB", (300, 200), "white").save(output, format="PNG")
        return output.getvalue()


class RenderingNormalizationTests(unittest.TestCase):
    def test_unsupported_geometry_symbols_get_semantic_display_forms(self) -> None:
        output = {
            "question_blocks": [
                {"type": "text", "text": "在 $▱ABCD$ 中", "figure_id": ""}
            ],
            "base_formulas": [
                {"formula": "△CED ∽ △FAE", "condition": "两角对应相等"}
            ],
            "guidance": {},
        }
        audit: list[dict[str, object]] = []

        normalized = pilot.normalize_output_for_render(output, audit)

        self.assertEqual(
            normalized["question_blocks"][0]["text"],
            "在 平行四边形 $ABCD$ 中",
        )
        self.assertEqual(
            normalized["base_formulas"][0]["formula"],
            r"△CED \sim △FAE",
        )
        rule_ids = {str(item.get("rule_id")) for item in audit}
        self.assertIn("display_spell_out_parallelogram_symbol", rule_ids)
        self.assertIn("display_similarity_symbol_to_latex", rule_ids)

    def test_midpoint_half_segment_denominator_repairs_with_full_evidence(self) -> None:
        output = {
            "question_blocks": [],
            "base_formulas": [
                {
                    "formula": r"DE=AE=\frac{1}{2 AD}",
                    "condition": "E为AD中点",
                }
            ],
            "guidance": {},
        }
        audit: list[dict[str, object]] = []

        normalized = pilot.normalize_output_for_render(output, audit)

        self.assertEqual(
            normalized["base_formulas"][0]["formula"],
            r"DE=AE=\frac{1}{2} AD",
        )
        self.assertTrue(any(
            item.get("rule_id")
            == "display_repair_midpoint_half_segment_denominator"
            for item in audit
        ))

    def test_midpoint_half_segment_denominator_is_not_guessed_without_evidence(self) -> None:
        repaired, change = pilot.repair_midpoint_half_segment_denominator(
            r"DE=AE=\frac{1}{2 AD}",
            "线段长度换算",
        )

        self.assertEqual(repaired, r"DE=AE=\frac{1}{2 AD}")
        self.assertIsNone(change)

    def test_segment_half_denominator_is_repaired_in_guidance_by_dimension(self) -> None:
        output = {
            "question_blocks": [],
            "base_formulas": [],
            "guidance": {
                "standard": {
                    "parts": [{
                        "part": "（1）",
                        "steps": [
                            r"利用 $E$ 为中点得 $DE=\frac{1}{2 AD}$，"
                            r"再由 $AD=BC$ 得 $DE=\frac{1}{2 BC}$"
                        ],
                        "alternative_methods": [],
                    }]
                }
            },
        }
        audit: list[dict[str, object]] = []

        normalized = pilot.normalize_output_for_render(output, audit)

        step = normalized["guidance"]["standard"]["parts"][0]["steps"][0]
        self.assertIn(r"$DE=\frac{1}{2} AD$", step)
        self.assertIn(r"$DE=\frac{1}{2} BC$", step)
        self.assertTrue(any(
            item.get("rule_id")
            == "display_repair_segment_half_denominator_by_dimension"
            for item in audit
        ))

    def test_segment_half_denominator_is_not_repaired_for_scalar_reciprocal(self) -> None:
        repaired, change = pilot.repair_dimensionally_invalid_half_segment_denominators(
            r"x=\frac{1}{2AD}"
        )

        self.assertEqual(repaired, r"x=\frac{1}{2AD}")
        self.assertIsNone(change)

    def test_parallel_slashes_become_textbook_parallel_symbol(self) -> None:
        output = {
            "question_blocks": [],
            "base_formulas": [
                {"formula": "AB // CD, AD // BC", "condition": "对边平行"}
            ],
            "guidance": {
                "advanced": {
                    "parts": [{
                        "steps": ["由 AB // CD 得内错角相等"],
                        "alternative_methods": [],
                    }]
                }
            },
        }

        normalized = pilot.normalize_output_for_render(output, [])

        self.assertEqual(
            normalized["base_formulas"][0]["formula"],
            r"AB \parallel CD, AD \parallel BC",
        )
        self.assertIn(
            "AB ∥ CD",
            normalized["guidance"]["advanced"]["parts"][0]["steps"][0],
        )

    def test_captured_parallelogram_geometry_correspondence_is_repaired(self) -> None:
        output = {
            "question_blocks": [{
                "type": "text",
                "text": (
                    "在 平行四边形 ABCD 中，F 在 BA 的延长线上，"
                    "CF 与 AD 相交于点 E；E 为 AD 的中点，BC=2CD"
                ),
                "figure_id": "",
            }],
            "base_formulas": [
                {"formula": r"△CED \sim △FAE", "condition": "两角对应相等"}
            ],
            "guidance": {
                "standard": {
                    "parts": [{
                        "part": "（1）",
                        "steps": [
                            r"判定 $△CED \sim △FAE$",
                            "利用 AD // BC 将 ∠DCE 转化为 ∠BCF",
                        ],
                        "alternative_methods": [],
                    }]
                }
            },
        }
        audit: list[dict[str, object]] = []

        normalized = pilot.normalize_output_for_render(output, audit)

        self.assertEqual(
            normalized["base_formulas"][0]["formula"],
            r"△CED \sim △FEA",
        )
        steps = normalized["guidance"]["standard"]["parts"][0]["steps"]
        self.assertIn("CD ∥ AF", steps[0])
        self.assertIn("$∠DCE$=$∠AFE$", steps[1])
        self.assertIn("$∠CED$=$∠FEA$", steps[2])
        rule_ids = {str(item.get("rule_id")) for item in audit}
        self.assertIn("geometry_repair_corresponding_triangle_order", rule_ids)
        self.assertIn("geometry_repair_captured_parallelogram_reasoning", rule_ids)

        guidance = normalized["guidance"]
        self.assertEqual(
            guidance["standard"]["parts"][0]["steps"][1],
            "由 CD ∥ AF 与 CF 为截线，得到 $∠DCE$=$∠AFE$",
        )

    def test_geometry_angle_tokens_are_atomic_for_line_wrapping(self) -> None:
        output = {
            "question_blocks": [{
                "type": "text",
                "text": "比较 ∠F 与 ∠BCF，并证明 △ABC 相似",
                "figure_id": "",
            }],
            "base_formulas": [],
            "guidance": {},
        }

        normalized = pilot.normalize_output_for_render(output, [])

        self.assertEqual(
            normalized["question_blocks"][0]["text"],
            "比较 $∠F$ 与 $∠BCF$，并证明 $△ABC$ 相似",
        )

    def test_sum_command_before_subscript_becomes_sigma(self) -> None:
        source = r"T_{n}=sum_{k=1}^{n}b_{k}"
        normalized = phy_rerender.normalize_text(source)
        rendered = clean_text(normalized)

        self.assertIn(r"\sum", normalized)
        self.assertIn("Σ", rendered)
        self.assertNotIn("sum", rendered)

    def test_vector_parentheses_and_subscript_become_arrow_notation(self) -> None:
        normalized = phy_rerender.normalize_text("vec(n_1)")
        rendered = clean_text(normalized)

        self.assertEqual(normalized, r"\vec{n}_{1}")
        self.assertIn("⃗", rendered)
        self.assertNotIn("vec", rendered)

    def test_line_plane_angle_repair_does_not_leave_visible_slash(self) -> None:
        source = r"代入公式 sinθ = |AB·n| / (|AB||n|)"
        normalized = phy_rerender.normalize_text(source)

        self.assertNotIn(r"\$", normalized)
        self.assertIn(r"$sinθ=\frac", normalized)

    def test_nested_vector_subscript_stays_outside_arrow(self) -> None:
        normalized = phy_rerender.normalize_text(r"\vec{n_{1}}")
        rendered = clean_text(normalized)

        self.assertEqual(normalized, r"\vec{n}_{1}")
        self.assertNotIn("→(", rendered)
        self.assertIn("⃗", rendered)

    def test_coordinate_dot_product_typo_is_repaired(self) -> None:
        normalized = phy_rerender.normalize_text(r"x_{1x}_{2}")

        self.assertEqual(normalized, r"x_{1}x_{2}")

    def test_primed_endpoint_vector_keeps_drawn_arrow(self) -> None:
        rendered = clean_text(r"\vec{BD'}")

        self.assertNotIn("→(", rendered)
        self.assertIn("⃗", rendered)
        self.assertTrue(rendered.endswith("'"))

    def test_vector_arrow_is_included_in_vertical_formula_bounds(self) -> None:
        body = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 24)
        script = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 16)
        vector_top, _ = formula_vertical_bounds(
            clean_text(r"\vec{n}"),
            body,
            script,
        )
        ascent, _ = body.getmetrics()

        self.assertLessEqual(vector_top, -ascent - 3)


if __name__ == "__main__":
    unittest.main()
