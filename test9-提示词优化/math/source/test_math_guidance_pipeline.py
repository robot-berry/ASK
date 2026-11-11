from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw, ImageFont, ImageOps

import full_benchmark as benchmark
import math_guidance_pilot as pilot
import rerender_test1_phy as phy_rerender
from math_guidance_renderer import (
    BODY_LINE,
    BODY_SIZE,
    CONTENT_WIDTH,
    DEFAULT_SONG,
    DEFAULT_SONG_BOLD,
    DEFAULT_TIMES,
    DEFAULT_TIMES_BOLD,
    _safe_textlength,
    base_formula_markup,
    crop_source_figure,
    make_fonts,
    markup_atoms,
    mixed_formula_atoms,
    normalize_visible_formula_spacing,
    strip_connector_punctuation,
    wrap_markup,
)
from result_image_store import (
    FRACTION_START,
    clean_text,
    draw_formula_at_baseline,
    draw_text_runs_at_baseline,
    formula_elements,
    formula_vertical_bounds,
    promote_plain_fractions,
)


class GenerationFirstRendererTests(unittest.TestCase):
    def test_multiline_text_measurement_never_reaches_pillow_textlength(self) -> None:
        image = Image.new("L", (200, 100), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(DEFAULT_SONG), 24)
        observed = _safe_textlength(draw, "第一行\n第二行", font=font)
        expected = float(draw.textlength("第一行 第二行", font=font))
        self.assertEqual(observed, expected)

    def test_mathbf_vector_glyph_uses_bold_latin_atom(self) -> None:
        image = Image.new("L", (500, 120), "white")
        draw = ImageDraw.Draw(image)
        fonts = make_fonts(
            BODY_SIZE,
            song_path=DEFAULT_SONG,
            song_bold_path=DEFAULT_SONG_BOLD,
            times_path=DEFAULT_TIMES,
            times_bold_path=DEFAULT_TIMES_BOLD,
        )

        atoms = mixed_formula_atoms(draw, r"\mathbf{a}+\mathbf{b}", fonts)

        self.assertEqual(
            [atom.value for atom in atoms if atom.kind == "plain_bold"],
            ["a", "b"],
        )

    def test_subset_symbol_uses_math_symbol_font(self) -> None:
        image = Image.new("L", (200, 100), "white")
        draw = ImageDraw.Draw(image)
        fonts = make_fonts(
            BODY_SIZE,
            song_path=DEFAULT_SONG,
            song_bold_path=DEFAULT_SONG_BOLD,
            times_path=DEFAULT_TIMES,
            times_bold_path=DEFAULT_TIMES_BOLD,
        )

        atoms = markup_atoms(draw, "a⊂α", fonts)

        self.assertTrue(any(atom.value == "⊂" for atom in atoms))

    def test_subset_latex_stays_a_symbol_in_result_image(self) -> None:
        self.assertEqual(clean_text(r"m\subset\alpha"), "m⊂α")
        self.assertEqual(clean_text(r"A\not\subseteq B"), "A⊈ B")

    def test_bare_slash_fractions_become_stacked_latex(self) -> None:
        self.assertEqual(promote_plain_fractions("表示为k/2。"), r"表示为$\frac{k}{2}$。")
        self.assertEqual(promote_plain_fractions("S=1/2×底×高"), r"S=$\frac{1}{2}$×底×高")
        self.assertEqual(promote_plain_fractions("(m+n)/(x-1)"), r"$\frac{m+n}{x-1}$")
        self.assertEqual(
            promote_plain_fractions(r"$a/2$ 与 k/3"), r"$\frac{a}{2}$ 与 $\frac{k}{3}$"
        )

    def test_bare_slash_fractions_leave_units_dates_and_paths_alone(self) -> None:
        for value in ("km/h 和 20m/s", "日期2024/5/6", "abc/def", "1/2/3"):
            self.assertEqual(promote_plain_fractions(value), value)

    def test_unary_negative_fraction_sign_stays_before_fraction(self) -> None:
        cleaned = clean_text(r"x=-\frac{8}{m}")
        elements = formula_elements(cleaned)
        self.assertEqual(
            elements,
            [("text", "x=-", None), ("fraction", "8", "m")],
        )
        self.assertEqual(
            formula_elements(clean_text(r"x=\frac{-8}{m}")),
            [("text", "x=-", None), ("fraction", "8", "m")],
        )

    def test_raw_multiplication_asterisks_become_implicit_products(self) -> None:
        normalized = pilot.normalize_formula_notation(
            r"S=\frac{1}{2}*CD*(y_A-y_B)"
        )
        self.assertEqual(
            normalized,
            r"S=\frac{1}{2}CD(y_{A}-y_{B})",
        )

    def test_implicit_product_keeps_numeric_ambiguity_explicit(self) -> None:
        self.assertEqual(pilot.normalize_formula_notation(r"2*4"), r"2\times4")
        self.assertEqual(pilot.normalize_formula_notation(r"2*m"), r"2m")
        self.assertEqual(pilot.normalize_formula_notation(r"m*2"), r"2m")
        self.assertEqual(
            pilot.normalize_formula_notation(r"\vec a\cdot\vec b"),
            r"\vec a\cdot\vec b",
        )

    def test_leading_numerator_minus_is_optically_raised(self) -> None:
        image = Image.new("L", (240, 120), "white")
        backing = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(DEFAULT_TIMES), 42)
        calls: list[tuple[str, int]] = []

        class SpyDraw:
            def text(self, xy, value, **kwargs):
                calls.append((str(value), int(xy[1])))
                return backing.text(xy, value, **kwargs)

            def textlength(self, value, **kwargs):
                return backing.textlength(value, **kwargs)

        draw_text_runs_at_baseline(
            SpyDraw(),
            20,
            90,
            "-4",
            body_font=font,
            script_font=font,
        )
        sign_baselines = [y for value, y in calls if value == "-"]
        digit_baselines = [y for value, y in calls if value == "4"]
        self.assertEqual(len(sign_baselines), 1)
        self.assertEqual(len(digit_baselines), 1)
        self.assertLess(sign_baselines[0], digit_baselines[0])

    def test_negative_fraction_sign_is_drawn_before_fraction_midline(self) -> None:
        image = Image.new("L", (300, 140), "white")
        backing = ImageDraw.Draw(image)
        font = ImageFont.truetype(str(DEFAULT_TIMES), 42)
        calls: list[tuple[str, int]] = []

        class SpyDraw:
            def text(self, xy, value, **kwargs):
                calls.append((str(value), int(xy[1])))
                return backing.text(xy, value, **kwargs)

            def textlength(self, value, **kwargs):
                return backing.textlength(value, **kwargs)

            def line(self, *args, **kwargs):
                return backing.line(*args, **kwargs)

        draw_formula_at_baseline(
            SpyDraw(),
            20,
            90,
            clean_text(r"x=-\frac{4}{m}"),
            body_font=font,
            script_font=font,
        )
        sign_baselines = [y for value, y in calls if value == "-"]
        self.assertEqual(len(sign_baselines), 1)
        self.assertLess(sign_baselines[0], 90)

    def test_bare_slash_fraction_enters_rich_formula_path(self) -> None:
        image = Image.new("L", (500, 120), "white")
        draw = ImageDraw.Draw(image)
        fonts = make_fonts(
            BODY_SIZE,
            song_path=DEFAULT_SONG,
            song_bold_path=DEFAULT_SONG_BOLD,
            times_path=DEFAULT_TIMES,
            times_bold_path=DEFAULT_TIMES_BOLD,
        )

        atoms = markup_atoms(draw, "由k/2=4可知", fonts)

        formulas = [atom.value for atom in atoms if atom.kind == "formula"]
        self.assertTrue(any(FRACTION_START in value for value in formulas), formulas)

    def test_empty_set_and_line_identifier_keep_textbook_symbols(self) -> None:
        self.assertEqual(clean_text(r"\varnothing"), "∅")
        self.assertEqual(clean_text(r"\emptyset"), "∅")
        self.assertEqual(clean_text(r"\ell_{1}"), "ℓ₁")

    def test_formula_spacing_removes_ocr_gaps_but_keeps_binary_operator_space(self) -> None:
        self.assertEqual(
            normalize_visible_formula_spacing("△ A C D"),
            "△ACD",
        )
        self.assertEqual(
            normalize_visible_formula_spacing("M (2 p, 2 p)"),
            "M(2p, 2p)",
        )
        self.assertEqual(
            normalize_visible_formula_spacing("AC=BC=1"),
            "AC = BC = 1",
        )

    def test_common_chinese_math_term_is_one_wrapping_atom(self) -> None:
        image = Image.new("L", (400, 100), "white")
        draw = ImageDraw.Draw(image)
        fonts = make_fonts(
            BODY_SIZE,
            song_path=DEFAULT_SONG,
            song_bold_path=DEFAULT_SONG_BOLD,
            times_path=DEFAULT_TIMES,
            times_bold_path=DEFAULT_TIMES_BOLD,
        )

        atoms = markup_atoms(draw, "求双曲线的标准方程", fonts)

        self.assertTrue(any(atom.value == "双曲线" for atom in atoms))
        self.assertTrue(any(atom.value == "标准方程" for atom in atoms))

    def test_wrapped_prose_never_starts_with_closing_punctuation(self) -> None:
        image = Image.new("L", (800, 300), "white")
        draw = ImageDraw.Draw(image)
        fonts = make_fonts(
            BODY_SIZE,
            song_path=DEFAULT_SONG,
            song_bold_path=DEFAULT_SONG_BOLD,
            times_path=DEFAULT_TIMES,
            times_bold_path=DEFAULT_TIMES_BOLD,
        )

        lines = wrap_markup(
            draw,
            "四枚钉子在此区域内．用一支铅笔拉紧细绳",
            fonts,
            max_width=270,
            line_height=BODY_LINE,
        )

        self.assertTrue(all(
            not line.atoms
            or (
                line.atoms[0].value.strip()
                and line.atoms[0].value not in "，。；：！？、．."
            )
            for line in lines[1:]
        ))

    def test_guidance_wrap_fills_the_first_line_instead_of_balancing_rows(self) -> None:
        image = Image.new("L", (800, 300), "white")
        draw = ImageDraw.Draw(image)
        fonts = make_fonts(
            BODY_SIZE,
            song_path=DEFAULT_SONG,
            song_bold_path=DEFAULT_SONG_BOLD,
            times_path=DEFAULT_TIMES,
            times_bold_path=DEFAULT_TIMES_BOLD,
        )
        value = "1. 要得到目标向量，需要先对已知向量的坐标做怎样的整体处理？"
        max_width = CONTENT_WIDTH - 12
        lines = wrap_markup(
            draw,
            value,
            fonts,
            max_width=max_width,
            line_height=BODY_LINE,
            left=12,
        )
        rendered = ["".join(atom.value for atom in line.atoms) for line in lines]
        first_width = sum(atom.width for atom in lines[0].atoms)

        self.assertGreaterEqual(len(lines), 2, rendered)
        self.assertGreaterEqual(first_width / max_width, 0.90, rendered)
        self.assertFalse(
            any(
                previous.endswith("坐") and following.startswith("标")
                for previous, following in zip(rendered, rendered[1:])
            ),
            rendered,
        )

    def test_medium_formula_uses_leftover_width_at_safe_operators(self) -> None:
        image = Image.new("L", (800, 300), "white")
        draw = ImageDraw.Draw(image)
        fonts = make_fonts(
            BODY_SIZE,
            song_path=DEFAULT_SONG,
            song_bold_path=DEFAULT_SONG_BOLD,
            times_path=DEFAULT_TIMES,
            times_bold_path=DEFAULT_TIMES_BOLD,
        )
        value = "由已知条件可得$a+b=200, c+d=200, a+c=270$"
        atoms = markup_atoms(draw, value, fonts)
        formula_values = [atom.value for atom in atoms if atom.kind == "formula"]
        self.assertGreaterEqual(len(formula_values), 2, [atom.value for atom in atoms])
        prefix_width = sum(
            atom.width for atom in atoms if atom.kind.startswith("plain")
        )
        first_formula_width = next(
            atom.width for atom in atoms if atom.kind == "formula"
        )
        max_width = max(200, round(prefix_width + first_formula_width + 12))
        lines = wrap_markup(
            draw,
            value,
            fonts,
            max_width=max_width,
            line_height=BODY_LINE,
        )
        rendered = ["".join(atom.value for atom in line.atoms) for line in lines]
        first_width = sum(atom.width for atom in lines[0].atoms)

        self.assertGreaterEqual(len(lines), 2, rendered)
        self.assertIn("a", rendered[0], rendered)
        self.assertGreaterEqual(first_width / max_width, 0.72, rendered)
        self.assertFalse(any(line.rstrip().endswith(("+", "=", "-", "−")) for line in rendered))

    def test_hat_is_rendered_as_math_accent_not_explanatory_prose(self) -> None:
        cleaned = clean_text(r"\hat{b}")

        self.assertEqual(cleaned, "b\u0302")
        self.assertNotIn("带帽", cleaned)

    def test_fraction_exponent_remains_one_nested_superscript(self) -> None:
        cleaned = clean_text(r"e^{\frac{2026}{2027}}")
        elements = formula_elements(cleaned)

        self.assertEqual([item[0] for item in elements], ["text", "superscript"])
        self.assertEqual(
            [item[0] for item in formula_elements(elements[1][1])],
            ["fraction"],
        )


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

    def test_sparse_crossing_prose_is_not_pulled_into_crop(self) -> None:
        # Completeness-first expansion grows the crop into contiguous figure
        # ink, but a sparse horizontal band that merely crosses the model edge
        # (the top rectangle) is below the ink-saturation bar and is never
        # pulled in.  A band that is dense and starts inside the model box
        # (the right rectangle) is now included under the completeness policy.
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
        # by a few pixels.
        draw.rectangle((205, 215, 560, 246), outline="black", width=3)
        draw.rectangle((488, 310, 560, 380), outline="black", width=4)

        _, audit = crop_source_figure(image, [180, 240, 500, 560])

        # The sparse top band stays excluded; the model top/bottom are not
        # expanded past the figure.
        self.assertGreaterEqual(audit["refined_bbox_px"][0], 180)
        self.assertLessEqual(audit["refined_bbox_px"][1], 500)
        self.assertLessEqual(audit["refined_bbox_px"][3], 560)
        self.assertNotEqual(audit["mode"], "closed_table_border_recovered")

    def test_whitespace_bar_separated_content_is_excluded(self) -> None:
        # Content separated from the figure by a real full-width whitespace
        # bar is not part of the diagram and must stay out of the crop.
        image = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(image)
        draw.polygon(
            ((220, 300), (420, 280), (460, 520), (250, 520)),
            outline="black",
            width=4,
        )
        # A separate label block well below the figure, after a clear
        # whitespace bar of several dozen pixels.
        draw.rectangle((250, 720, 420, 780), outline="black", width=3)

        _, audit = crop_source_figure(image, [180, 240, 500, 560])

        self.assertLessEqual(audit["refined_bbox_px"][3], 560)

    def test_dominant_figure_component_cut_by_model_box_expands(self) -> None:
        # One connected geometry figure whose ink extends well past the model
        # box left edge (a tight bbox).  The dominant-component recovery must
        # expand that edge so a vertex is not clipped, while the two safety
        # tests above prove disconnected prose/“解”-marks never trigger it.
        image = Image.new("RGB", (1000, 1000), "white")
        draw = ImageDraw.Draw(image)
        draw.polygon(
            ((120, 300), (420, 280), (460, 520), (250, 520)),
            outline="black",
            width=4,
        )
        draw.line((90, 520, 420, 280), fill="black", width=4)

        _, audit = crop_source_figure(image, [180, 240, 500, 560])

        self.assertIn("left", audit["dominant_component_edge_expand_px"])
        self.assertLess(audit["refined_bbox_px"][0], 180)
        self.assertGreaterEqual(audit["refined_bbox_px"][1], 240)
        self.assertLessEqual(audit["refined_bbox_px"][3], 560)

    def test_connector_punctuation_is_stripped_from_step_edges(self) -> None:
        # The provider repair model emits steps that begin with a Chinese
        # comma (``，观察图形…``), which must not render as ``1. ，观察…``.
        self.assertEqual(strip_connector_punctuation("，观察图形"), "观察图形")
        self.assertEqual(strip_connector_punctuation("观察图形，"), "观察图形")
        self.assertEqual(strip_connector_punctuation("，。；：，观察图形。；"), "观察图形")
        self.assertEqual(strip_connector_punctuation("（1）观察图形"), "（1）观察图形")
        self.assertEqual(strip_connector_punctuation("∠A=∠DCF？"), "∠A=∠DCF？")
        self.assertEqual(strip_connector_punctuation(""), "")

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
        # Completeness-first expansion grows past the model box to include the
        # fragment, and the guard then whites it out; the delivered crop is
        # therefore free of the handwritten mark.
        self.assertGreater(audit["refined_bbox_px"][2], 500)
        self.assertTrue(audit["edge_fragment_suppression_applied"])
        # The rectangular crop remains full-width, so lower-right geometry is
        # not sacrificed merely because the unwanted block sits above it.
        self.assertEqual(audit["source_crop_size_px"][0], 351)
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
    @staticmethod
    def rendering_probe() -> tuple[object, ImageDraw.ImageDraw]:
        fonts = make_fonts(
            BODY_SIZE,
            song_path=DEFAULT_SONG,
            song_bold_path=DEFAULT_SONG_BOLD,
            times_path=DEFAULT_TIMES,
            times_bold_path=DEFAULT_TIMES_BOLD,
        )
        return fonts, ImageDraw.Draw(Image.new("L", (800, 100), 255))

    @staticmethod
    def normalize_guidance_steps(*steps: str) -> tuple[list[str], list[dict[str, object]]]:
        output = {
            "question_blocks": [],
            "base_formulas": [],
            "guidance": {
                "standard": {
                    "parts": [{
                        "part": "（1）",
                        "steps": list(steps),
                        "alternative_methods": [],
                    }]
                }
            },
        }
        audit: list[dict[str, object]] = []
        normalized = pilot.normalize_output_for_render(output, audit)
        rendered_steps = normalized["guidance"]["standard"]["parts"][0]["steps"]
        return rendered_steps, audit

    def test_outer_math_delimiters_around_chinese_and_cosfrac_are_repaired(self) -> None:
        source = (
            "$对 f'(x) 使用商法则求导得 f''(x) = "
            "CoSfrac{1 - ln(1+x)}{(1+x)^{2}}$"
        )

        (step,), audit = self.normalize_guidance_steps(source)

        self.assertFalse(step.startswith("$对"), step)
        self.assertTrue(step.startswith("对 "), step)
        self.assertIn(r"\frac{1 - ln(1+x)}{(1+x)^{2}}", step)
        self.assertNotIn("CoSfrac", step)
        self.assertEqual(step.count("$") % 2, 0, step)
        self.assertGreaterEqual(step.count("$"), 2, step)
        rule_ids = {str(item.get("rule_id")) for item in audit}
        self.assertIn("display_cofrac_full_token", rule_ids)
        self.assertIn("display_unwrap_mixed_prose_math_delimiters", rule_ids)

    def test_inline_cosfrac_is_repaired_without_raw_or_unbalanced_markup(self) -> None:
        source = "代入 CoSfrac{a}{b} 得到中间量"

        (step,), _ = self.normalize_guidance_steps(source)

        self.assertIn(r"\frac{a}{b}", step)
        self.assertNotIn("CoSfrac", step)
        self.assertEqual(step.count("$") % 2, 0, step)
        self.assertGreaterEqual(step.count("$"), 2, step)

    def test_genuine_text_formula_is_not_unwrapped_as_mixed_prose(self) -> None:
        source = r"$\text{条件} x>0$"

        (step,), audit = self.normalize_guidance_steps(source)

        self.assertEqual(step, source)
        self.assertEqual(step.count("$"), 2)
        self.assertFalse(any(
            item.get("rule_id") == "display_unwrap_mixed_prose_math_delimiters"
            for item in audit
        ))

    def test_plain_algebra_in_prose_is_promoted_to_textbook_markup(self) -> None:
        steps = [
            "把x=2代入y=mx-2m+4算出点A纵坐标",
            "因为点A在y=k/x上，所以k等于横纵坐标乘积",
            "设点B坐标为(x_B, y_B)，利用它在两函数图像上列方程",
            "由BC||x轴和点C坐标为(0, y_B)",
            "求直线OA的解析式y=(y_A/2)x，令其等于y_B",
            "计算CD=x_D，高为y_A-y_B，列出S(m)",
            "需确保x_D<x_B；另外m取值要使点B存在",
            "注意m≠0，以及点A在第一象限隐含k>0",
        ]
        output = {
            "question_blocks": [],
            "base_formulas": [],
            "guidance": {
                "standard": {
                    "parts": [{
                        "part": "（1）",
                        "steps": steps,
                        "alternative_methods": [],
                    }]
                }
            },
        }
        audit: list[dict[str, object]] = []

        normalized = pilot.normalize_output_for_render(output, audit)
        rendered_steps = normalized["guidance"]["standard"]["parts"][0]["steps"]

        self.assertEqual(
            rendered_steps,
            [
                "把$x=2$代入$y=mx-2m+4$算出点A纵坐标",
                r"因为点A在$y=\frac{k}{x}$上，所以k等于横纵坐标乘积",
                r"设点B坐标为$(x_{B}, y_{B})$，利用它在两函数图像上列方程",
                r"由$BC ∥ x$轴和点C坐标为$(0, y_{B})$",
                r"求直线OA的解析式$y=(\frac{y_{A}}{2})x$，令其等于$y_{B}$",
                r"计算$CD=x_{D}$，高为$y_{A}-y_{B}$，列出$S(m)$",
                r"需确保$x_{D}<x_{B}$；另外m取值要使点B存在",
                "注意$m≠0$，以及点A在第一象限隐含$k>0$",
            ],
        )
        self.assertTrue(any(
            item.get("rule_id") == "display_promote_plain_algebra_markup"
            for item in audit
        ))

    def test_plain_algebra_promotion_is_idempotent_and_leaves_labels_alone(self) -> None:
        source = "点A到点B的距离、CD长度与m取值范围保持原意"
        self.assertEqual(pilot.promote_plain_algebra_markup(source), source)

        promoted = pilot.promote_plain_algebra_markup(
            "坐标为(x_B,y_B)，且y=k/x"
        )
        self.assertEqual(
            pilot.promote_plain_algebra_markup(promoted),
            promoted,
        )

    def test_direct_five_tier_display_guidance_gets_formula_promotion(self) -> None:
        output = {
            "question_blocks": [],
            "base_formulas": [],
            "guidance": {},
            "display_guidance": {
                "middle": {
                    "parts": [{
                        "part": "（2）",
                        "steps": [
                            "因A(2,4)为交点，x_A=2，且x_A · x_B=-8/m，得x_B=-4/m",
                        ],
                    }],
                },
            },
        }
        normalized = pilot.normalize_output_for_render(output, [])
        step = normalized["display_guidance"]["middle"]["parts"][0]["steps"][0]

        self.assertIn("$x_{A}=2$", step)
        self.assertIn(r"$x_{A}x_{B}=-\frac{8}{m}$", step)
        self.assertIn(r"$x_{B}=-\frac{4}{m}$", step)
        self.assertNotIn("x_A=2", step)
        self.assertNotIn("x_B=-4/m", step)

    def test_display_steps_drop_decorative_rules_and_defensive_closures(self) -> None:
        output = {
            "question_blocks": [],
            "base_formulas": [],
            "guidance": {},
            "display_guidance": {
                "novice": {
                    "parts": [{
                        "steps": [
                            "建立等量关系$k=2×4$。",
                            "检查计算过程，确认$m≠0$条件下A点确实存在，且$k≠0$符合题意，逻辑无误。————",
                        ],
                    }],
                },
            },
        }
        normalized = pilot.normalize_output_for_render(output, [])
        steps = normalized["display_guidance"]["novice"]["parts"][0]["steps"]
        self.assertEqual(steps, [r"建立等量关系$k=2\times4$。"])

    def test_infinity_and_scripted_names_never_remain_raw(self) -> None:
        output = {
            "question_blocks": [{
                "type": "text",
                "text": "研究区间(0, +infty)并标记x_0",
                "figure_id": "",
            }],
            "base_formulas": [{
                "formula": "若 $f'(x_0)=0$ 且左右异号，则 $x_0$ 为极值点",
                "condition": "判断驻点",
            }],
            "guidance": {},
        }

        normalized = pilot.normalize_output_for_render(output, [])
        question = normalized["question_blocks"][0]["text"]
        theorem = normalized["base_formulas"][0]["formula"]

        self.assertIn(r"$(0, +\infty)$", question)
        self.assertIn(r"$x_{0}$", question)
        self.assertEqual(
            theorem,
            r"若 f'(x_{0})=0 且左右异号，则 x_{0} 为极值点",
        )
        markup = base_formula_markup(theorem)
        self.assertEqual(markup.count("$"), 2)
        self.assertNotIn("$$", markup)

    def test_cjk_fraction_labels_use_explicit_text_groups(self) -> None:
        normalized = pilot.normalize_formula_notation(
            r"频率=\frac{频数}{样本总量}"
        )

        self.assertEqual(
            normalized,
            r"频率=\frac{\text{频数}}{\text{样本总量}}",
        )
        self.assertNotRegex(
            normalized,
            r"\frac\{[^{}]+\}\{[^{}]*[一-鿿][^{}]*\}",
        )

    def test_latin_symbol_inside_text_group_is_not_a_chinese_subscript(self) -> None:
        errors: list[str] = []
        pilot.validate_formula_source(
            r"P(A)=\frac{\text{构成事件A的区域面积}}{\text{全部可能区域的总面积}}",
            "base_formulas[0].formula",
            errors,
        )
        self.assertFalse(any("中文说明性下标" in item for item in errors), errors)

    def test_literal_left_brace_is_not_counted_as_unbalanced_grouping(self) -> None:
        errors: list[str] = []
        pilot.validate_formula_source(
            r"\left\{x-2y=1,\ \textcircled{1}\right.",
            "question_blocks[0].formula[0]",
            errors,
        )

        self.assertFalse(any("花括号不成对" in item for item in errors), errors)

        malformed_errors: list[str] = []
        pilot.validate_formula_source(
            r"\frac{1}{2",
            "question_blocks[0].formula[0]",
            malformed_errors,
        )
        self.assertTrue(
            any("花括号不成对" in item for item in malformed_errors),
            malformed_errors,
        )

    def test_simple_fraction_stops_before_top_level_products(self) -> None:
        cases = {
            r"S_{△ACD}=1/2 * |CD| * h": (
                r"S_{△ACD}=\frac{1}{2}|CD|h"
            ),
            r"S=1/2 × b × h": r"S=\frac{1}{2}bh",
            r"S=1/2 \times b \times h": r"S=\frac{1}{2}bh",
            r"a * 1/2": r"a\frac{1}{2}",
            r"a/b*c/d": r"\frac{a}{b}\frac{c}{d}",
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(
                    pilot.normalize_formula_notation(source),
                    expected,
                )

        self.assertEqual(
            pilot.normalize_formula_notation(r"S=1/(2*b)"),
            r"S=\frac{1}{(2b)}",
        )

    def test_triangle_area_slash_formula_remains_a_product_in_pipeline(self) -> None:
        output = {
            "question_blocks": [],
            "base_formulas": [{
                "formula": r"S_{△ACD} = 1/2 * |CD| * h",
                "condition": "以CD为底，h为对应的高",
            }],
            "guidance": {},
        }

        normalized = pilot.normalize_output_for_render(output, [])

        self.assertEqual(
            normalized["base_formulas"][0]["formula"],
            r"S_{△ACD} =\frac{1}{2}|CD|h",
        )
        self.assertNotIn(
            r"\frac{1}{2 * |CD| * h}",
            normalized["base_formulas"][0]["formula"],
        )

    def test_domain_condition_is_hoisted_out_of_reciprocal_denominator(self) -> None:
        source = r"y=\frac{k}{x (x>0 且 k≠0)}"
        repaired, change = pilot.repair_nested_domain_condition_denominator(source)

        self.assertEqual(repaired, r"y=\frac{k}{x} (x>0 且 k≠0)")
        self.assertEqual(
            change["rule_id"],
            "display_hoist_domain_condition_from_denominator",
        )
        unchanged, no_change = pilot.repair_nested_domain_condition_denominator(
            r"y=\frac{k}{x(x+1)}"
        )
        self.assertEqual(unchanged, r"y=\frac{k}{x(x+1)}")
        self.assertIsNone(no_change)

    def test_triangle_area_factors_leave_denominator_only_with_full_evidence(self) -> None:
        malformed = r"S =\frac{1}{2 × 底 × 高}"
        repaired, change = pilot.repair_triangle_area_product_denominator(
            malformed,
            "计算三角形ACD面积时，以CD为底，A到BC的距离为高",
        )

        self.assertEqual(
            repaired,
            r"S=\frac{1}{2}\text{底}\text{高}",
        )
        self.assertEqual(
            change["rule_id"],
            "display_repair_triangle_area_product_denominator",
        )

        unchanged, no_change = pilot.repair_triangle_area_product_denominator(
            malformed,
            "一般代数式的化简条件",
        )
        self.assertEqual(unchanged, malformed)
        self.assertIsNone(no_change)

    def test_triangle_area_denominator_repair_runs_in_base_formula_pipeline(self) -> None:
        output = {
            "question_blocks": [],
            "base_formulas": [{
                "formula": r"S =\frac{1}{2 × 底 × 高}",
                "condition": "计算△ACD面积，以CD为底，对应距离为高",
            }],
            "guidance": {},
        }
        audit: list[dict[str, object]] = []

        normalized = pilot.normalize_output_for_render(output, audit)

        self.assertEqual(
            normalized["base_formulas"][0]["formula"],
            r"S=\frac{1}{2}\text{底}\text{高}",
        )
        self.assertTrue(any(
            item.get("rule_id")
            == "display_repair_triangle_area_product_denominator"
            for item in audit
        ))

    def test_named_triangle_area_denominator_artifact_is_repaired(self) -> None:
        output = {
            "question_blocks": [],
            "base_formulas": [{
                "formula": r"S_{△ACD} =\frac{1}{2 * |CD| * h}",
                "condition": (
                    "以CD为底，h为点A到直线BC的垂直距离"
                    "（即A与C纵坐标之差的绝对值）"
                ),
            }],
            "guidance": {},
        }
        audit: list[dict[str, object]] = []

        normalized = pilot.normalize_output_for_render(output, audit)

        self.assertEqual(
            normalized["base_formulas"][0]["formula"],
            r"S_{△ACD}=\frac{1}{2}|CD|h",
        )
        self.assertTrue(any(
            item.get("rule_id")
            == "display_repair_triangle_area_product_denominator"
            for item in audit
        ))

        unchanged, no_change = pilot.repair_triangle_area_product_denominator(
            r"S_{△ACD}=\frac{1}{2 * |CD| * h}",
            "一般代数式条件",
        )
        self.assertEqual(
            unchanged,
            r"S_{△ACD}=\frac{1}{2 * |CD| * h}",
        )
        self.assertIsNone(no_change)

    def test_plain_latin_labels_and_numbers_are_atomic_at_line_edges(self) -> None:
        fonts, draw = self.rendering_probe()
        atoms = markup_atoms(
            draw,
            "平面 PDM 内总量200，临界值6.635，置信度99%",
            fonts,
        )
        values = [atom.value for atom in atoms]

        for token in ("PDM", "200", "6.635", "99%"):
            self.assertIn(token, values)
        self.assertFalse(any(atom.value in {"PD", "DM", "20", "00"} for atom in atoms))

    def test_long_formula_wraps_before_operators_not_after_them(self) -> None:
        fonts, draw = self.rendering_probe()
        value = (
            "$a+b=200, c+d=200, a+c=270, b+d=130, "
            "n=400, p+q=600, r+s=800$"
        )
        lines = wrap_markup(
            draw,
            value,
            fonts,
            max_width=360,
            line_height=BODY_LINE,
        )
        rendered = ["".join(atom.value for atom in line.atoms) for line in lines]

        self.assertGreater(len(rendered), 1)
        self.assertFalse(any(line.rstrip().endswith(("+", "=", "-", "−")) for line in rendered))
        self.assertFalse(any(line.strip() in {"+", "=", "-", "−"} for line in rendered))

    def test_fill_blank_parentheses_do_not_split_across_rows(self) -> None:
        fonts, draw = self.rendering_probe()
        value = "已知集合A与B，则A交B=（    ）"
        atoms = markup_atoms(draw, value, fonts)
        closing_index = next(
            index for index, atom in enumerate(atoms) if atom.value == "）"
        )
        width_through_opening = sum(
            atom.width for atom in atoms[:closing_index]
        )
        lines = wrap_markup(
            draw,
            value,
            fonts,
            max_width=max(80, round(width_through_opening - 2)),
            line_height=BODY_LINE,
        )
        rendered = ["".join(atom.value for atom in line.atoms) for line in lines]

        self.assertFalse(any(line.rstrip().endswith(("(", "（")) for line in rendered))
        self.assertFalse(any(line.lstrip().startswith((")", "）")) for line in rendered))

    def test_fill_blank_and_unit_are_one_wrap_atom(self) -> None:
        fonts, draw = self.rendering_probe()
        atoms = markup_atoms(draw, "宽度为____ cm，结果保留根号", fonts)

        self.assertIn("____ cm", [atom.value for atom in atoms])

        blank_index = next(
            index for index, atom in enumerate(atoms) if atom.value == "____ cm"
        )
        prefix_width = sum(atom.width for atom in atoms[:blank_index])
        lines = wrap_markup(
            draw,
            "宽度为____ cm，结果保留根号",
            fonts,
            max_width=max(80, round(prefix_width + atoms[blank_index].width / 2)),
            line_height=BODY_LINE,
        )
        rendered = ["".join(atom.value for atom in line.atoms) for line in lines]

        self.assertTrue(any("____ cm" in line for line in rendered), rendered)
        self.assertFalse(any(line.rstrip().endswith("____") for line in rendered))

    def test_prose_only_base_formula_wraps_as_prose(self) -> None:
        fonts, draw = self.rendering_probe()
        theorem = "中位线定理：三角形两边中点连线平行于第三边且等于其一半"
        markup = base_formula_markup(theorem)
        lines = wrap_markup(
            draw,
            f"3. {markup}",
            fonts,
            max_width=CONTENT_WIDTH,
            line_height=BODY_LINE,
        )

        self.assertEqual(markup, theorem)
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(sum(atom.width for atom in line.atoms) <= CONTENT_WIDTH for line in lines))

    def test_promoted_relation_merges_with_an_existing_geometry_formula(self) -> None:
        output = {
            "question_blocks": [],
            "base_formulas": [],
            "guidance": {
                "standard": {
                    "parts": [{
                        "part": "（1）",
                        "steps": ["由∠ABC=90°且AB=BC识别直角三角形"],
                        "alternative_methods": [],
                    }]
                }
            },
        }

        normalized = pilot.normalize_output_for_render(output, [])
        step = normalized["guidance"]["standard"]["parts"][0]["steps"][0]

        self.assertEqual(
            step,
            "由$∠ABC=90°$且$AB=BC$识别直角三角形",
        )
        self.assertNotIn("$$", step)

    def test_math_symbol_font_split_cannot_split_one_relation_across_lines(self) -> None:
        fonts = make_fonts(
            BODY_SIZE,
            song_path=DEFAULT_SONG,
            song_bold_path=DEFAULT_SONG_BOLD,
            times_path=DEFAULT_TIMES,
            times_bold_path=DEFAULT_TIMES_BOLD,
        )
        probe = Image.new("L", (800, 100), 255)
        draw = ImageDraw.Draw(probe)
        value = "这是一个接近行尾的条件$m≠0$"
        atoms = markup_atoms(draw, value, fonts)
        relation_start = next(
            index for index, atom in enumerate(atoms) if atom.value == "m"
        )
        relation_atoms = atoms[relation_start:relation_start + 3]
        self.assertEqual([atom.value for atom in relation_atoms], ["m", "≠", "0"])
        self.assertEqual(
            [atom.break_before for atom in relation_atoms],
            [True, False, False],
        )

        prefix_width = sum(atom.width for atom in atoms[:relation_start])
        partial_relation_width = relation_atoms[0].width + relation_atoms[1].width / 2
        lines = wrap_markup(
            draw,
            value,
            fonts,
            max_width=round(prefix_width + partial_relation_width),
            line_height=BODY_LINE,
        )
        line_values = ["".join(atom.value for atom in line.atoms) for line in lines]
        self.assertTrue(any("m≠0" in line for line in line_values), line_values)
        self.assertFalse(any(line.endswith(("m", "≠")) for line in line_values))
        self.assertFalse(any(line.startswith(("≠", "0")) for line in line_values))

        coordinate_atoms = markup_atoms(draw, "坐标$(x_B,y_B)$", fonts)
        coordinate_formula_atoms = [
            atom for atom in coordinate_atoms if atom.kind == "formula"
        ]
        self.assertEqual(len(coordinate_formula_atoms), 1)
        self.assertIn(",", coordinate_formula_atoms[0].value)

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

    def test_latex_prime_command_renders_instead_of_fallback_warning(self) -> None:
        rendered = clean_text(r"f^{\prime}(x)")

        self.assertEqual(rendered, "f′(x)")
        self.assertNotIn("公式含未支持命令", rendered)

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
