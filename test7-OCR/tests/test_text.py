import unittest

from test7_ocr.text import (
    contains_latex_source,
    has_formula_signal,
    has_visual_cue,
    latex_to_readable,
    parse_question,
    reflow_text_for_render,
    unknown_latex_commands,
)


class TextSafetyTests(unittest.TestCase):
    def test_formula_region_count_is_a_formula_signal_without_inline_operator(self) -> None:
        self.assertTrue(has_formula_signal("电荷量为100 μC", formula_count=2))
        self.assertTrue(has_formula_signal("已知 hc=1240 eV·nm", formula_count=0))
        self.assertFalse(has_formula_signal("选择正确的说法", formula_count=0))

    def test_incidental_math_variables_do_not_start_the_option_group(self) -> None:
        text = (
            "7. 动点A、B在曲线上，且OA.若条件成立，则最大值为 "
            "A. √103 B. 9 C. √73 D. 8"
        )
        number, stem, options = parse_question(text)
        self.assertEqual(number, "7")
        self.assertIn("动点A、B", stem)
        self.assertIn("OA.若条件成立", stem)
        self.assertEqual([option["label"] for option in options], list("ABCD"))
        self.assertEqual(options[-1]["text"], "8")

    def test_common_latex_is_converted_before_rendering(self) -> None:
        raw = r"1. 若 $x^{2} \geq \frac{1}{2}$，则 $x \neq 0$"
        display = latex_to_readable(raw)
        self.assertNotIn("$", display)
        self.assertNotIn(r"\frac", display)
        self.assertIn("≥", display)
        self.assertIn("1/2", display)
        self.assertIn("x²", display)
        self.assertFalse(contains_latex_source(display))

    def test_bare_ocr_exponents_are_converted_before_rendering(self) -> None:
        display = latex_to_readable("m=1.67×10^-27 kg，c=3.0×10^8 m/s")
        self.assertEqual(display, "m=1.67×10⁻²⁷ kg，c=3.0×10⁸ m/s")
        self.assertFalse(contains_latex_source(display))

    def test_parenthesized_degree_superscript_is_not_exposed(self) -> None:
        self.assertEqual(latex_to_readable("sin 53^(°)=0.8"), "sin 53°=0.8")

    def test_fraction_with_nested_radical_has_no_command_leak(self) -> None:
        display = latex_to_readable(r"$B = \frac{2\sqrt{3}}{3}$")
        self.assertEqual(display, "B = (2√3)/3")
        self.assertNotIn("frac", display)
        self.assertFalse(contains_latex_source(display))

    def test_single_prose_letter_is_not_parsed_as_an_option(self) -> None:
        number, stem, options = parse_question("14. 证明D、M、N在一条直线上，并求MN")
        self.assertEqual(number, "14")
        self.assertIn("D、M、N", stem)
        self.assertEqual(options, [])

    def test_options_are_structured(self) -> None:
        number, stem, options = parse_question("12. 选择正确答案\nA. 甲 B. 乙 C. 丙 D. 丁")
        self.assertEqual(number, "12")
        self.assertIn("选择正确答案", stem)
        self.assertEqual([item["label"] for item in options], list("ABCD"))

    def test_coherent_bare_and_comma_labels_are_recovered(self) -> None:
        warnings: list[str] = []
        number, stem, options = parse_question(
            "8. 求k的取值范围\nA [0,1] B. (0,1] C. (0,2] D [0,2]",
            recovery_warnings=warnings,
        )
        self.assertEqual(number, "8")
        self.assertEqual(stem, "8. 求k的取值范围")
        self.assertEqual([item["label"] for item in options], list("ABCD"))
        self.assertEqual(
            warnings,
            ["OPTION_SEQUENCE_RECOVERED_FROM_COHERENT_LABELS:AD"],
        )

        _number, _stem, comma_options = parse_question(
            "4. 下列说法正确的是\nA. 甲\nB，乙\nC. 丙\nD. 丁"
        )
        self.assertEqual([item["label"] for item in comma_options], list("ABCD"))

    def test_bare_math_variables_do_not_create_an_option_sequence(self) -> None:
        _number, stem, options = parse_question(
            "5. 已知P(A)与点D (1,2)，求值\nB. 乙\nD. 丁"
        )
        self.assertIn("P(A)", stem)
        self.assertEqual([item["label"] for item in options], ["B", "D"])

    def test_missing_dot_after_line_leading_question_number_is_recovered(self) -> None:
        warnings: list[str] = []
        number, stem, options = parse_question(
            "10 某工地用起重机起吊重物\nA. 甲 B. 乙 C. 丙 D. 丁",
            recovery_warnings=warnings,
        )
        self.assertEqual(number, "10")
        self.assertTrue(stem.startswith("10 某工地"))
        self.assertEqual([item["label"] for item in options], list("ABCD"))
        self.assertEqual(warnings, ["QUESTION_NUMBER_PUNCTUATION_RECOVERED:10"])

    def test_internal_roman_question_header_is_recovered_after_section_heading(self) -> None:
        number, stem, options = parse_question(
            "非选择题部分\n三、非选择题\n14-I（1）（2分）测量电阻\nA. 甲 B. 乙"
        )
        self.assertEqual(number, "14-I")
        self.assertIn("14-I", stem)
        self.assertEqual([item["label"] for item in options], ["A", "B"])

    def test_math_font_and_vector_wrappers_do_not_leak_as_words(self) -> None:
        value = latex_to_readable(r"$\{\pmb{a},\boldsymbol{b}\},\vec{F},\mathbb{R}$")
        self.assertEqual(value, "a,b,F,R")
        self.assertEqual(unknown_latex_commands(r"$\pmb{a}+\boldsymbol{b}+\vec{F}$"), [])

    def test_nested_style_wrappers_and_duplicate_commas_do_not_leak(self) -> None:
        value = latex_to_readable(
            r"$\underline{\mathbf{A}\quad \mathbf{m}/\mathbf{s}^{2}},，下一项$"
        )
        self.assertNotIn("underline", value)
        self.assertNotIn("mathbf", value)
        self.assertEqual(value, "A m/s²，下一项")

    def test_nested_radicals_and_spacing_placeholders_are_readable(self) -> None:
        value = latex_to_readable(
            r"S_n=2(\sqrt{S_{1}}+\sqrt { S _ { n } }+\cdots), "
            r"r/R=\underline{\hspace{1cm}}"
        )

        self.assertIn("√(S₁)", value)
        self.assertIn("√(Sₙ)", value)
        self.assertIn("…", value)
        self.assertIn("r/R=______", value)
        self.assertNotIn("sqrt", value)
        self.assertNotIn("hspace", value)
        self.assertNotIn("underline", value)

    def test_relation_alias_and_brace_annotation_do_not_leak(self) -> None:
        value = latex_to_readable(r"$PA\bot CM,\underbrace{10^{15}}_{}$")

        self.assertIn("PA⊥ CM", value)
        self.assertIn("10¹⁵", value)
        self.assertNotIn("underbrace", value)
        self.assertEqual(unknown_latex_commands(r"$PA\bot CM,\underbrace{10^{15}}_{}$"), [])

    def test_hypothetical_scatter_plot_does_not_require_an_image(self) -> None:
        text = "如果散点图中所有散点都落在一条直线上，则R²=1"
        self.assertFalse(has_visual_cue(text, ["如图", "图中", "下图"]))
        self.assertTrue(has_visual_cue("如图所示，小球沿斜面运动", ["如图", "图中", "下图"]))

    def test_abstract_graph_wording_does_not_require_a_source_crop(self) -> None:
        cues = ["如图", "图中", "下图", "坐标系", "图象", "图像"]
        self.assertFalse(
            has_visual_cue("若将函数f(x)的图像向右平移π/4个单位", cues)
        )
        self.assertFalse(
            has_visual_cue("在平面直角坐标系xOy中，抛物线与双曲线相交", cues)
        )
        self.assertFalse(
            has_visual_cue("函数f(x)的图象在x=1处的切线斜率为2", cues)
        )

    def test_displayed_or_selectable_graph_still_requires_a_source_crop(self) -> None:
        cues = ["如图", "图中", "下图", "坐标系", "图象", "图像"]
        self.assertTrue(has_visual_cue("某函数的图像如图所示", cues))
        self.assertTrue(has_visual_cue("下列图象中可能是f(x)图象的是", cues))
        self.assertTrue(has_visual_cue("函数f(x)的图像可能为下列选项中的", cues))

    def test_render_reflow_joins_physical_ocr_rows(self) -> None:
        value = reflow_text_for_render(
            "8. 图甲为电吹风电路图，可动触片可同时接触\n"
            "两个触点。根据以上\n"
            "信息，下列说法正确的是\n"
            "A. 选项甲\nB. 选项乙"
        )
        self.assertIn("可同时接触两个触点。根据以上信息", value)
        self.assertIn("\nA. 选项甲\nB. 选项乙", value)

    def test_render_reflow_preserves_subquestions_and_section_headers(self) -> None:
        value = reflow_text_for_render(
            "非选择题部分\n三、非选择题（共4题）\n14-I 测量电阻\n"
            "（1）完成读数\n读出电流表示数\n①写出结果\n②说明原因"
        )
        self.assertEqual(
            value.splitlines(),
            [
                "非选择题部分",
                "三、非选择题（共4题）",
                "14-I 测量电阻",
                "（1）完成读数读出电流表示数",
                "①写出结果",
                "②说明原因",
            ],
        )

    def test_render_reflow_keeps_roman_experiment_part_on_its_own_line(self) -> None:
        value = reflow_text_for_render(
            "14. 实验题（I、Ⅱ、Ⅲ共14分）\nI. （4分）在用单摆测量重力加速度的实验中\n"
            "（1）选择悬挂方式"
        )
        self.assertEqual(
            value.splitlines(),
            [
                "14. 实验题（I、Ⅱ、Ⅲ共14分）",
                "I. （4分）在用单摆测量重力加速度的实验中",
                "（1）选择悬挂方式",
            ],
        )

    def test_coherent_options_recover_comma_and_bare_math_labels(self) -> None:
        number, _stem, options = parse_question(
            r"6. 请选择 A. $m>2$ B，$m=2$ C $m<2$ D. $m\ne0$"
        )
        self.assertEqual(number, "6")
        self.assertEqual([option["label"] for option in options], list("ABCD"))
        self.assertEqual(options[1]["text"], "$m=2$")
        self.assertEqual(options[2]["text"], "$m<2$")

        _number, readable_stem, readable_options = parse_question(
            "6. 求实数范围 A. m > 2 B. m ≥ 2 C m < 2 D. m ≠ 0"
        )
        self.assertEqual(readable_stem, "6. 求实数范围")
        self.assertEqual(
            [option["label"] for option in readable_options],
            list("ABCD"),
        )
        self.assertEqual(readable_options[2]["text"], "m < 2")

    def test_reference_formulas_and_conditions_get_semantic_rows(self) -> None:
        value = reflow_text_for_render(
            "参考公式及数据：（1）x=1 (2) y=2\n"
            "条件①：a=1；条件②：b=2 条件③：c=3\n"
            "注：只选择一个条件"
        )
        self.assertEqual(
            value.splitlines(),
            [
                "参考公式及数据：",
                "（1）x=1",
                "(2) y=2",
                "条件①：a=1；",
                "条件②：b=2",
                "条件③：c=3",
                "注：只选择一个条件",
            ],
        )

    def test_reference_marker_wrapped_inside_one_formula_becomes_a_real_row(self) -> None:
        value = reflow_text_for_render(
            r"参考公式及数据：（1） $x=1\quad(2)\quad\hat{b}=2$"
        )

        self.assertEqual(
            value.splitlines(),
            ["参考公式及数据：", "（1） $x=1$", r"(2) $\hat{b}=2$"],
        )


if __name__ == "__main__":
    unittest.main()
