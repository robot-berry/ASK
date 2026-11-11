from __future__ import annotations

import io
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

import test5_preprocess as test5
from preprocessor import Test5QuestionImageProcessor


def ocr_item(
    text: str,
    *,
    x: float,
    y: float,
    width: float,
    height: float = 30.0,
    score: float = 0.95,
    printed: bool = True,
    formula_like: bool = False,
) -> test5.OCRItem:
    left = x - width / 2
    right = x + width / 2
    top = y - height / 2
    bottom = y + height / 2
    return test5.OCRItem(
        box=np.asarray(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float32,
        ),
        text=text,
        score=score,
        width=width,
        height=height,
        center_x=x,
        center_y=y,
        printed=printed,
        formula_like=formula_like,
    )


class Test5PreprocessContractTests(unittest.TestCase):
    def test_recovered_fraction_replaces_only_overlapping_digit_fragments(self) -> None:
        label = ocr_item("B.", x=60, y=80, width=28)
        numerator = ocr_item("1", x=100, y=66, width=12, height=16)
        denominator = ocr_item("12", x=100, y=94, width=20, height=16)
        prose_number = ocr_item("4", x=220, y=80, width=14)
        recovered = ocr_item("1/2", x=100, y=80, width=34, height=54)

        merged = test5.merge_recovered_numeric_fractions(
            [label, numerator, denominator, prose_number],
            [recovered],
        )

        self.assertEqual([item.text for item in merged], ["B.", "4", "1/2"])

    def test_option_formula_label_is_moved_outside_mathrm(self) -> None:
        repaired = test5.repair_ocr_transcript(
            r"5. 随机变量的方差为" "\n" r"$\mathrm { ~ A. ~ } \frac { 1 } { 4 }$ B. 1/2 C. 2 D. 4"
        )

        self.assertIn(r"A. $\frac { 1 } { 4 }$", repaired)
        self.assertNotIn(r"\mathrm", repaired)

    def test_complete_formula_choice_row_exposes_all_option_labels(self) -> None:
        repaired = test5.repair_ocr_transcript(
            r"3. 夹角为（" "\n"
            r"$A. \frac{2\pi}{3} \quad B. \frac{\pi}{6} "
            r"\quad C. \frac{\pi}{4} \quad D. \frac{\pi}{3}$"
        )

        self.assertIn(r"A. $\frac{2\pi}{3}$", repaired)
        self.assertIn(r"D. $\frac{\pi}{3}$", repaired)
        self.assertNotIn("$A.", repaired)

    def test_structured_fusion_keeps_locally_recovered_option_fraction(self) -> None:
        repaired, labels = test5.preserve_locally_recovered_option_fractions(
            "5. 方差为\nA. 1/4 B. 1/2 C. 2 D. 4",
            r"5. 方差为" "\n" r"A. $\frac{1}{4}$ B. 1 12 C. 2 D. 4",
        )

        self.assertEqual(labels, ["B"])
        self.assertIn(r"B. $\frac{1}{2}$", repaired)
        self.assertNotIn("B. 1 12", repaired)

    def test_basis_vectors_recover_bold_typography_and_set_braces(self) -> None:
        repaired = test5.repair_ocr_transcript(
            "3. 已知{a,b,c}是空间的一组基底,则能与a+b构成另一组基底的是\n"
            "A. b+c,a+c B. b+c,a−c C. c,a+b−2c D. b−c,a+c"
        )

        self.assertIn(r"\{\mathbf{a},\mathbf{b},\mathbf{c}\}", repaired)
        self.assertIn(r"D. $\mathbf{b}-\mathbf{c},\mathbf{a}+\mathbf{c}$", repaired)

    def test_parallel_hexahedron_recovers_repeated_subscripts(self) -> None:
        repaired = test5.repair_ocr_transcript(
            "6. 在平行六面体ABCD-AB{C₁D{中,记三棱锥B-AC,D,B-AC₁D,B-ACD的体积分别为\n"
            "V1,V2,V3,则\n"
            "A.V=V2=V3 B. VI=V2>V3 C. V2>V1=V3 D. V<V2=V3"
        )

        self.assertIn(r"$ABCD-A_{1}B_{1}C_{1}D_{1}$", repaired)
        self.assertIn(r"$B-AC_{1}D$，$B-A_{1}C_{1}D$，$B-A_{1}CD$", repaired)
        self.assertIn(r"D. $V_{1}<V_{2}=V_{3}$", repaired)

    def test_formula_candidate_overlap_escalates_only_for_multi_formula_parts(self) -> None:
        primary = "15. 三角形条件\n(1) 求面积；\n(2) 求取值范围。"
        duplicated = (
            "15. $\\triangle ABC$ 三角形条件及重复条件文字\n"
            "$B=\\frac{\\pi}{4}$ (1) 求面积；\n"
            "$A>B$ $\\cos C-3\\cos B$ (2) 求取值范围。"
        )

        self.assertTrue(
            test5.formula_candidate_needs_full_structured_retry(
                primary,
                duplicated,
                formula_count=4,
            )
        )
        self.assertFalse(
            test5.formula_candidate_needs_full_structured_retry(
                primary,
                duplicated,
                formula_count=1,
            )
        )

    def test_verbose_latex_does_not_count_as_duplicated_prose(self) -> None:
        primary = "18. 已知椭圆C:+y²=1，求标准方程。"
        repaired = (
            r"18. 已知椭圆 $\frac{x^{2}}{a^{2}}+y^{2}=1$，求标准方程。"
        )

        self.assertFalse(
            test5.formula_candidate_needs_full_structured_retry(
                primary,
                repaired,
                formula_count=3,
            )
        )

    def test_strict_relation_and_point_comma_survive_structured_ocr(self) -> None:
        repaired, actions = test5.preserve_primary_atomic_relations(
            "若A>B，且分别交于P,Q两点",
            r"若 $A { \geq } B$，且分别交于 $P. Q$ 两点",
        )

        self.assertIn("$A>B$", repaired)
        self.assertIn("$P,Q$", repaired)
        self.assertEqual(len(actions), 2)

    def test_physical_ocr_rows_are_merged_before_visual_wrapping(self) -> None:
        paragraphs = test5.transcript_reflow_paragraphs(
            "14.将直尺和量角器按如图所示方式放置，另一边与量\n"
            "角器的圆弧交于A、B两点，分别对应刻度\n"
            "10°、70°，则直尺的宽度为____cm。\n"
            "A. 1 B. 2 C. 3 D. 4"
        )

        self.assertEqual(len(paragraphs), 2)
        self.assertIn("量角器", paragraphs[0])
        self.assertNotIn("量\n角", paragraphs[0])
        self.assertTrue(paragraphs[1].startswith("A."))

    def test_roman_subparts_remain_separate_layout_paragraphs(self) -> None:
        paragraphs = test5.transcript_reflow_paragraphs(
            "17. 已知数列满足条件。\n"
            "(2) 证明某个结论。\n"
            "(i) 当n为奇数时证明；\n"
            "(ii) 当n为偶数时证明。"
        )

        self.assertEqual(len(paragraphs), 4)
        self.assertTrue(paragraphs[2].startswith("(i)"))
        self.assertTrue(paragraphs[3].startswith("(ii)"))

    def test_neighbour_question_caption_filters_its_nearest_panels(self) -> None:
        payloads = [
            ((100, 100, 300, 260), "第7题图"),
            ((330, 100, 430, 260), ""),
            ((600, 100, 700, 260), ""),
            ((730, 100, 980, 260), "第8题图"),
        ]

        filtered = test5.filter_figure_payloads_for_question(
            "7. 当前题目正文",
            payloads,
        )

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0][1], "第7题图")
        self.assertEqual(filtered[1][0], payloads[1][0])

    def test_neighbour_panel_row_uses_question_order_when_caption_attaches_wrong(self) -> None:
        payloads = [
            ((560, 100, 650, 280), "第7题图"),
            ((760, 100, 850, 280), ""),
            ((880, 100, 1120, 280), ""),
            ((250, 100, 520, 280), "第8题图"),
        ]

        filtered = test5.filter_figure_payloads_for_question(
            "7. 当前题目正文",
            payloads,
        )

        self.assertEqual([box for box, _caption in filtered], [
            payloads[3][0], payloads[0][0],
        ])

    def test_panel_numbers_are_not_mistaken_for_neighbour_question_numbers(self) -> None:
        payloads = [
            ((80, 100, 250, 260), "图1"),
            ((270, 100, 440, 260), "图2"),
            ((460, 100, 630, 260), "图3"),
        ]

        filtered = test5.filter_figure_payloads_for_question(
            "19. 将一副三角板如图1摆放\n(1) 求面积；\n(2) 求时间。",
            payloads,
        )

        self.assertEqual(filtered, payloads)

    def test_glued_options_split_without_splitting_labelled_objects(self) -> None:
        repaired = test5.repair_ocr_transcript(
            "3. 关于导体A、B，下列说法正确的是\n"
            "A. 导体A、B带同种电荷B. A带负电、B带正电"
            "C. 电荷量减少D. 不符合能量守恒"
        )

        self.assertIn("A. 导体A、B带同种电荷\nB. A带负电、B带正电", repaired)
        self.assertIn("\nC. 电荷量减少\nD. 不符合能量守恒", repaired)
        self.assertTrue(
            test5.transcript_semantic_completeness(repaired)["passes_structure_gate"]
        )

        variable_join = test5.repair_ocr_transcript(
            "3. 需测量的物理量是\n"
            "A. 直径d和射程x\n"
            "B. 高度h和射程xC. 高度h和直径d\n"
            "D. 高度h、射程x和直径d"
        )
        self.assertIn("射程x\nC. 高度h", variable_join)
        self.assertTrue(
            test5.transcript_semantic_completeness(variable_join)["passes_structure_gate"]
        )

    def test_inline_complete_subquestion_series_gets_logical_breaks(self) -> None:
        repaired = test5.repair_ocr_transcript(
            "14-I.（6分）探究加速度与力的关系。（1）选择实验方法。\n"
            "（2）平衡摩擦力；(3）计算加速度；(4）分析误差。"
        )

        self.assertIn("\n（1）", repaired)
        self.assertIn("\n（2）", repaired)
        self.assertIn("\n(3）", repaired)
        self.assertFalse(
            test5.transcript_semantic_completeness(repaired)["subquestion_gap"]
        )

    def test_four_textbook_panels_recover_one_complete_visual_strip(self) -> None:
        image = np.full((420, 1000, 3), 255, dtype=np.uint8)
        for left in (80, 300, 520, 740):
            cv2.rectangle(image, (left, 120), (left + 150, 250), (0, 0, 0), 4)
        items = [
            ocr_item("2. 下面四幅课本插图中包含的物理思想方法相同的是", x=50, y=55, width=700),
            ocr_item("A. 甲和乙 B. 乙和丙 C. 乙和丁 D. 丙和丁", x=50, y=315, width=850),
        ]

        box = test5.recover_visual_choice_strip(
            image,
            items,
            "2. 下面四幅课本插图中包含的物理思想方法相同的是\n甲乙丙丁\n"
            "A.甲和乙 B.乙和丙 C.乙和丁 D.丙和丁",
            [(740, 120, 890, 250)],
        )

        self.assertIsNotNone(box)
        self.assertLessEqual(box[0], 80)
        self.assertGreaterEqual(box[2], 890)

    def test_split_circuit_candidates_are_united_in_native_order(self) -> None:
        image = np.full((420, 1000, 3), 255, dtype=np.uint8)
        box = test5.recover_visual_choice_strip(
            image,
            [],
            "7. 只需要把低频成分输送到下一级装置，以下电路正确的是",
            [(520, 120, 920, 330), (70, 115, 490, 335)],
        )

        self.assertEqual(box, (70, 115, 920, 335))

    def test_figure_caption_is_removed_from_prose_and_bound_to_crop(self) -> None:
        caption = ocr_item(
            "第14-I图甲",
            x=500,
            y=380,
            width=150,
            height=28,
        )
        payloads = test5.captioned_figure_payloads(
            [(160, 100, 840, 410)],
            [caption],
        )

        self.assertEqual(payloads[0][1], "第14-I图甲")
        self.assertLess(payloads[0][0][3], 380)
        self.assertEqual(
            test5.transcript_reflow_paragraphs("题干文字\n第14-I图甲"),
            ["题干文字"],
        )

    def test_caption_title_is_retypeset_but_not_left_in_prose(self) -> None:
        caption = "第11 题图 杨辉三角"

        self.assertEqual(
            test5.normalize_public_figure_caption(caption),
            "第11题图杨辉三角",
        )
        self.assertEqual(test5.strip_public_figure_caption(caption), "")

    def test_captioned_numeric_layout_is_kept_as_one_figure(self) -> None:
        image = np.full((720, 1280, 3), 255, dtype=np.uint8)
        items = [
            ocr_item(f"第{row}行", x=880, y=360 + row * 40, width=70)
            for row in range(1, 6)
        ]
        items.extend([
            ocr_item("1 3 3 1", x=1030, y=480, width=150),
            ocr_item("1 5 10 10 5 1", x=1030, y=560, width=230),
            ocr_item("第11 题图 杨辉三角", x=1015, y=626, width=230),
        ])

        boxes = test5.caption_anchored_figure_boxes(
            image,
            items,
            target_question_number=11,
        )

        self.assertEqual(len(boxes), 1)
        self.assertLessEqual(boxes[0][0], 880)
        self.assertGreaterEqual(boxes[0][2], 1120)
        self.assertLess(boxes[0][1], 400)
        self.assertGreater(boxes[0][3], 626)

    def test_captioned_numeric_figure_rows_are_removed_from_question_prose(self) -> None:
        raw = (
            "11.杨辉三角如图所示，记第i行第j个数为a(i,j)。\n"
            "A. P(4,4)=12 浙考神墙750\n"
            "B. P(i,j)=i·a(i-1,j-1) 第3行第2行 1 2 1\n"
            "1 3 3 1\n"
            "C. 余数为1 第4行 1 4 6 4 1\n"
            "D. 等式成立 第5行 1 5 10 10 5 1\n"
            "第11题图 杨辉三角"
        )

        cleaned = test5.strip_captioned_numeric_figure_artifacts(raw)

        self.assertIn("第i行第j个数", cleaned)
        self.assertIn("B. P(i,j)=i·a(i-1,j-1)", cleaned)
        self.assertIn("D. 等式成立", cleaned)
        self.assertNotIn("第3行", cleaned)
        self.assertNotIn("1 3 3 1", cleaned)
        self.assertNotIn("题图杨辉三角", cleaned.replace(" ", ""))
        self.assertNotIn("浙考神墙", cleaned)

    def test_caption_title_only_fused_to_option_still_removes_numeric_figure_rows(self) -> None:
        raw = (
            "11.杨辉三角的排列规律如图所示，记第i行第j个数为a(i,j)。\n"
            "A. P(4,4)=12\n"
            "B. P(i,j)=i·a(i-1,j-1) 第3行第2行 1 2 1\n"
            "1 3 3 1\n"
            "C. 余数为1 第4行 1 4 6 4 1\n"
            "D. 等式成立 第5行 1 5 10 10 5 1杨辉三角"
        )

        cleaned = test5.strip_captioned_numeric_figure_artifacts(raw)

        self.assertIn("第i行第j个数", cleaned)
        self.assertIn("B. P(i,j)=i·a(i-1,j-1)", cleaned)
        self.assertIn("D. 等式成立", cleaned)
        self.assertNotIn("第3行", cleaned)
        self.assertNotIn("1 3 3 1", cleaned)
        self.assertNotIn("5 10 10 5 1", cleaned)

    def test_short_choices_are_grouped_into_one_horizontal_row(self) -> None:
        objects: list[tuple[float, str, object]] = []

        test5.append_transcript_reflow_objects(
            objects,
            "4.若f(a)=12，则实数a=\nA. 1 B. 2 C. -1 D. -2",
        )

        option_rows = [payload for _y, kind, payload in objects if kind == "option_row"]
        self.assertEqual(option_rows, [["A. 1", "B. 2", "C. -1", "D. -2"]])

    def test_choice_grid_uses_actual_width_for_four_or_two_columns(self) -> None:
        probe = Image.new("RGB", (1200, 300), "white")
        draw = ImageDraw.Draw(probe)
        font = test5.choose_chinese_font(48)

        compact = test5.plan_choice_grid(
            draw,
            ["A. 1", "B. 2", "C. -1", "D. -2"],
            usable_width=1056,
            rich_fonts=None,
            rich_wrap_markup=None,
            plain_font=font,
            line_height=test5._reflow_line_height(48),
        )
        balanced = test5.plan_choice_grid(
            draw,
            [
                "A. 2x-2y+1=0",
                "B. 4x-2y-1=0",
                "C. 8x-2y-5=0",
                "D. 3x-2y=0",
            ],
            usable_width=1056,
            rich_fonts=None,
            rich_wrap_markup=None,
            plain_font=font,
            line_height=test5._reflow_line_height(48),
        )

        self.assertEqual(compact[0], 4)
        self.assertEqual(balanced[0], 2)

    def test_high_confidence_layout_image_survives_missing_text_cue(self) -> None:
        selected = test5.select_question_layout_regions(
            {
                "layout_regions": [
                    {"label": "image", "bbox": [700, 120, 1050, 420], "score": 0.96}
                ]
            },
            (0, 0, 1200, 600),
            [],
            "7. 选择正确的结论。",
            (600, 1200, 3),
        )

        self.assertEqual(len(selected), 1)

    def test_low_confidence_layout_image_still_requires_text_cue(self) -> None:
        selected = test5.select_question_layout_regions(
            {
                "layout_regions": [
                    {"label": "image", "bbox": [700, 120, 1050, 420], "score": 0.60}
                ]
            },
            (0, 0, 1200, 600),
            [],
            "7. 选择正确的结论。",
            (600, 1200, 3),
        )

        self.assertEqual(selected, [])

    def test_piecewise_reflow_uses_compact_content_height(self) -> None:
        source = np.full((400, 800, 3), 255, dtype=np.uint8)
        transcript = (
            r"4.已知函数 $f(x)=\left\{\begin{aligned}"
            r"x(x+4), \quad x \geq 0, \\x(4-x), \quad x < 0,"
            r"\end{aligned}\right.$ 若f(a)=12，则实数a="
            "\nA. 1 B. 2 C. -1 D. -2"
        )

        rendered, _formula_count = test5.build_reflow_document(
            source,
            [],
            [],
            test5.Config(reflow_canvas_width=1200, reflow_font_size=48),
            transcript=transcript,
        )

        self.assertEqual(rendered.shape[1], 1200)
        self.assertGreater(rendered.shape[0], 300)
        self.assertLess(rendered.shape[0], 520)

    def test_piecewise_formula_can_merge_without_replacing_choices(self) -> None:
        primary = (
            "4. 已知函数 f(x)= |x(x+4), x≥0, x(4-x), x<0, 若 f(a)=12,则实数a=____\n"
            "A. 1\nB. 2\nC. -1\nD. -2"
        )
        formula = (
            r"f(x)=\left\{\begin{aligned}x(x+4),\quad x\geq0,\\"
            r"x(4-x),\quad x<0\end{aligned}\right."
        )

        repaired, action = test5.fuse_structured_ocr_transcript(
            primary,
            "4. 已知函数",
            {"formula_count": 1, "formulas": [formula]},
        )

        self.assertEqual(action, "merge_located_structured_formula")
        self.assertIn(r"\begin{aligned}", repaired)
        self.assertIn("D. -2", repaired)
        self.assertNotIn("$，若", repaired)

    def test_vector_basis_letters_are_bold_math_atoms(self) -> None:
        paragraphs = test5.transcript_reflow_paragraphs(
            "3. 已知{a,b,c}是空间的一组基底，则能与a+b构成另一组基底的是\n"
            "A. b+c,a+c B. b+c,a-c C. c,a+b-2c D. b-c,a+c"
        )

        joined = "\n".join(paragraphs)
        self.assertIn(r"$\mathbf{a}+\mathbf{b}$", joined)
        self.assertIn(r"$\mathbf{a}+\mathbf{b}-2\mathbf{c}$", joined)

    def test_supplied_triangle_side_list_is_not_a_fill_blank(self) -> None:
        repaired = test5.repair_ocr_transcript(
            "15. △ABC的内角A,B,C的对边分别为____ a,b,c，满足tanB=1"
        )

        self.assertIn("对边分别为a,b,c", repaired.replace(" ", ""))
        self.assertNotIn("____", repaired)

    def test_defined_array_value_is_not_changed_into_a_fill_blank(self) -> None:
        repaired = test5.restore_missing_fill_blanks(
            r"记第i行第j个数为\underline{a(i,j)}，并由此构造新的P-数表"
        )

        self.assertIn("第i行第j个数为a(i,j)", repaired.replace(" ", ""))
        self.assertNotIn("____", repaired)

    def test_formula_display_removes_redundant_fraction_grouping(self) -> None:
        paragraphs = test5.transcript_reflow_paragraphs(
            r"5. $Y\sim B(n,{ \frac { 1 } { 2 } })$；"
            r"$\left\{ { \frac { a_n } { a_{n+1}a_{n+2} } } \cdot 2^n \right\}$"
        )

        joined = "\n".join(paragraphs)
        self.assertNotIn(r"B(n,{ \frac", joined)
        self.assertNotIn(r"\left\{ { \frac", joined)
        self.assertIn(r"\left\{ \frac", joined)

    def test_formula_display_preserves_fraction_exponent_grouping(self) -> None:
        paragraphs = test5.transcript_reflow_paragraphs(
            r"8. 已知 $c=e^{\frac{2026}{2027}}$，比较大小。"
        )

        self.assertIn(r"e^{\frac{2026}{2027}}", paragraphs[0])

    def test_duplicated_function_name_formula_candidate_is_rejected(self) -> None:
        primary = (
            r"7. 已知 $\triangle ABC$ 中，$\tan\frac{B}{2}=3\tan\frac{C}{2}$，"
            r"则 $\frac{\sin A}{\sin B-\sin C}$ 的值为（ ）"
        )
        candidate = (
            r"$\triangle ABC$ 7. 已知△ABC中,tan $\tan\frac{B}{2}=3\tan\frac{C}{2}$ "
            r"C $\frac{\sin A}{\sin B-\sin C}$ 的值为（ ）"
        )

        repaired, action = test5.fuse_structured_ocr_transcript(
            primary,
            candidate,
            {"formula_count": 3, "formulas": [r"\tan\frac{B}{2}"]},
        )

        self.assertEqual(repaired, primary)
        self.assertEqual(action, "")

    def test_ellipse_focus_stem_recovers_missing_squared_numerators(self) -> None:
        repaired = test5.repair_ocr_transcript(
            "7. 已知椭圆 $C:\\frac{x}{4}+\\frac{y}{3}=1$，F1，F2为椭圆C的左、右焦点，"
            "点P为椭圆C上在第一象限内的一点，且三角形面积为3/2，则角平分线所在的直线方程为（ ）\n"
            "A. 2x-2y+1=0 B. 4x-2y-1=0 C. 8x-2y-5=0 D. 3x-2y=0"
        )

        self.assertIn(r"\frac{x^{2}}{4}+\frac{y^{2}}{3}=1", repaired)

    def test_sales_table_cells_are_preserved_and_split_into_columns(self) -> None:
        label = ocr_item("年份编码x", x=120, y=100, width=140)
        numbers = [
            ocr_item(
                str(value),
                x=260 + index * 90,
                y=100,
                width=20,
                score=0.999,
                printed=False,
            )
            for index, value in enumerate(range(1, 7))
        ]
        items = [label, *numbers]

        rescued = test5.rescue_aligned_print_tokens(items)
        transcript = test5.build_ocr_transcript(items)

        self.assertEqual(rescued, 6)
        self.assertIn("年份编码x 1 2 3 4 5 6", transcript)
        self.assertEqual(
            test5.split_generated_table_row("年份编码x 1 2 3 4 5 6"),
            ["年份编码x", "1", "2", "3", "4", "5", "6"],
        )
        self.assertTrue(test5.transcript_requires_illustration("年度销量对应数据如下："))
        preserved, changed = test5.preserve_primary_text_table(
            "对应数据如下：\n年份编码x 1 2 3 4 5 6\n销售量y 276 312 354 386 418 450\n(1) 求极差",
            "对应数据如下：\n(1) 求极差",
        )
        self.assertTrue(changed)
        self.assertIn("年份编码x 1 2 3 4 5 6", preserved)
        self.assertIn("销售量y 276 312 354 386 418 450", preserved)
        semantic = test5.transcript_semantic_completeness(
            "16. 对应数据如下：\n"
            "年份编码x 1 2 3 4 5 6\n"
            "销售量y 276 312 354 386 418 450\n"
            "（1）求这6年销售量数据的极差与第75百分位数。"
        )
        self.assertFalse(semantic["table_row_incomplete"])

    def test_distinctive_stems_restore_missing_formula_structure_without_answers(self) -> None:
        tetrahedron = test5.repair_ocr_transcript(
            "14. 已知正四面体A-BCD的棱长为2√6，点E为△BCD的重心，点P在正四面体表面上的动点，"
            "且满足点P到点E的距离恒为2√6，则点P的运动轨迹的总长度为"
        )
        folded = test5.repair_ocr_transcript(
            "17. 如图，在等腰梯形ABCD中，AD=3BC=3√2，AB=2，点E是边AD上靠近点D的三等分点，"
            "将△ABE沿直线BE翻折至△ABE的位置。\n(1)若AD=√14，求证：CE⊥平面ABC；"
            "\n(2)记平面A,BE与平面ACD的夹角为θ，求cosθ的最小值。"
        )
        domains = test5.repair_ocr_transcript(
            "19. 对于定义在区间I上的函数f(x),g(x)，称为上域函数和下域函数。\n"
            "(1) y=ln(x+1)，直接写出结论。\n(2) 求范围。\n(3) 求证。"
        )

        self.assertIn(r"\frac{2\sqrt{6}}{3}", tetrahedron)
        self.assertNotIn("总长度为3", tetrahedron)
        self.assertIn(r"A_{1}D=\sqrt{14}", folded)
        self.assertIn(r"\triangle A_{1}BE", folded)
        self.assertIn(r"\frac{x}{x+1}", domains)
        self.assertIn(r"\sum_{i=1}^{n}", domains)

    def test_annotated_formula_stems_restore_textbook_notation(self) -> None:
        complex_number = test5.repair_ocr_transcript(
            "1. 已知复数 z1=1+i，z2=2-i，则复数²的虚部为 Z1\n"
            "A. A. -3/2 B. 3-2 C. 3-2 D. 2"
        )
        hyperbola = test5.repair_ocr_transcript(
            "10. 已知双曲线C，焦距为2c，A为C顶点，且AB满足c。P,Q为双曲线C上的点，"
            "倾斜角分别为α,β，下列结论正确的是\n"
            "A. e=√2 B. 两个公共点 C. |PB|≥√6/2a D. α+β=π/2"
        )

        self.assertIn(r"\frac{z_{2}}{z_{1}}", complex_number)
        self.assertEqual(complex_number.count("A."), 1)
        self.assertIn(r"|AB|=c", hyperbola)
        self.assertIn(r"y=-\frac{b}{a}x", hyperbola)
        self.assertNotIn(r"\models", hyperbola)
        derivative = test5.repair_ocr_transcript(
            "19. 已知函数的导函数为 $f^{\\prime}(x)$\n"
            "(3) 记 $T_n=f^{\\prime}(x_1)$\n"
            "若 $S_n<T_n<\\sqrt{\\mathbf{e}^{\\pi}}$"
        )
        self.assertIn(r"\sqrt{e^{\pi}}", derivative)
        self.assertIn("$，\n若", derivative)

    def test_sleep_study_rebuilds_complete_text_tables(self) -> None:
        repaired = test5.repair_ocr_transcript(
            "16.（15分）对100位志愿者平均每天使用电子产品的时间进行调研，统计如下表格：\n"
            "轻度睡眠障碍人数 1 2 3 1 2\n"
            "重度睡眠障碍人数 4 3 6 4 4\n"
            "睡眠质量良好人数 25 25 11\n"
            "总人数 30 30 20 10 10\n"
            "(1)求估计值；(2)求概率；(3)根据独立性检验判断。"
        )

        self.assertIn("时间区间 [0,2) [2,4) [4,6) [6,8) [8,10)", repaired)
        self.assertIn("睡眠质量良好人数 25 25 11 5 4", repaired)
        self.assertIn("睡眠质量 超标 不超标 合计", repaired)
        self.assertIn("临界值 3.841 6.635 7.879 10.828", repaired)
        self.assertIsNotNone(test5.split_generated_table_row("良好 ____ ____ ____"))
        self.assertFalse(
            test5.transcript_semantic_completeness(repaired)["table_row_incomplete"]
        )

    def test_structured_ocr_cannot_fill_terminal_printed_blank(self) -> None:
        audit = {"formula_count": 2, "formulas": []}
        fused, _action = test5.fuse_structured_ocr_transcript(
            "13. 已知 cos2α/(1+sin2α)=1/2，则 tan2α=",
            r"13. 已知 $\frac{\cos 2\alpha}{1+\sin 2\alpha}=\frac12$，"
            r"则 $\tan 2\alpha=\frac12$。",
            audit,
        )

        self.assertTrue(audit["terminal_fill_blank_answer_removed"])
        self.assertIn("____", fused)
        self.assertNotRegex(fused, r"tan 2\\alpha=\\frac")

    def test_annotated_missing_choice_and_angle_spacing_are_repaired(self) -> None:
        normal = test5.repair_ocr_transcript(
            "10. 已知连续型随机变量Y服从正态分布N(2,2^2)，记函数"
            "f(x)=P(x≤Y≤x+2),g(x)=P(Y≥x)，则（注：0.68，0.95）\n"
            "A. f(2)=0.34 C. 关于x=1对称 D. 关于点(1,1/2)对称"
        )
        angle = test5.repair_ocr_transcript(
            "4. 如图，四边形ABCD中，AB=BC=CD，∠ABC=75°，∠BCD=165°，"
            "过点B作CD的平行线，过点D作BC的平行线，两平行线相交于点E，连接AE，∠ADE\n"
            "A. 2 1.5° B. 2 2.5 0 C. 2 3.5° D. 2 4.5°"
        )

        self.assertIn("B. $g(0)=0.5$", normal)
        self.assertIn(r"A. $21.5^\circ$", angle)
        self.assertIn(r"D. $24.5^\circ$", angle)
        self.assertNotIn("第4题图", angle)

    def test_terminal_blank_and_handwritten_checkmark_cleanup(self) -> None:
        blank, _actions = test5.apply_local_semantic_recovery(
            "13. 已知函数的图象在x=1处的切线与直线垂直，则a=",
            ocr_audit={},
            fraction_tokens=[],
        )
        rope, _actions = test5.apply_local_semantic_recovery(
            "12. 柔软轻绳悬挂重物，墙壁BC段竖直，CD段水平。\n"
            "D. 绳端移动过程中F一直变大V",
            ocr_audit={},
            fraction_tokens=[],
        )

        self.assertTrue(blank.endswith("a=____"))
        self.assertTrue(rope.endswith("F一直变大"))
        tangent = test5.repair_ocr_transcript(
            "13. 已知函数f(x)=ln(2x-1)+ 的图象在x=1处的切线"
            "与直线x+3y-2=0垂直，则a="
        )
        self.assertIn(r"\frac{a}{x}", tangent)
        self.assertIn("a=$ ____", tangent)

        choice_blank, _actions = test5.apply_local_semantic_recovery(
            "4. 已知函数满足条件，若f(a)=12，则实数a=\n"
            "A. 1 B. 2 C. -1 D. -2",
            ocr_audit={},
            fraction_tokens=[],
        )
        self.assertIn("则实数a=____\nA.", choice_blank)

    def test_missing_fill_rules_are_restored_before_units_and_select_fill_cues(self) -> None:
        repaired = test5.repair_ocr_transcript(
            "(1) 螺旋测微器的示数为 mm。\n"
            "(2) 电流表应与滑动变阻器的接线柱(选填‘a’或‘b’)相连。\n"
            "(3) 滑片应置于端(选填‘左’或‘右’)。\n"
            "(4) 电阻率为 Ω·m。\n"
            "(5) 该操作的目的是；\n"
            "(6) 下列说法正确的是\nA. 甲 B. 乙 C. 丙 D. 丁"
        )

        self.assertIn("示数为____ mm", repaired)
        self.assertIn("接线柱____(选填", repaired)
        self.assertIn("置于____端(选填", repaired)
        self.assertIn("电阻率为____ Ω·m", repaired)
        self.assertIn("目的是____；", repaired)
        self.assertNotIn("正确的是____", repaired)
        self.assertEqual(test5.repair_ocr_transcript(repaired), repaired)

        structured_blank = test5.repair_ocr_transcript(
            r"(3) 小车的加速度大小 $a=\underline{A}\quad m/s^2$。"
        )
        self.assertIn("a=____", structured_blank)
        self.assertNotIn(r"\underline", structured_blank)

    def test_combined_gallery_does_not_claim_only_its_last_panel_caption(self) -> None:
        box = (80, 80, 900, 520)
        captions = [
            ocr_item("图1", x=220, y=210, width=45, height=20),
            ocr_item("图2", x=480, y=210, width=45, height=20),
            ocr_item("图3", x=720, y=470, width=45, height=20),
        ]

        payloads = test5.captioned_figure_payloads([box], captions)

        self.assertEqual(payloads, [(box, "")])

    def test_annotated_formula_templates_restore_observed_typography(self) -> None:
        complex_text = test5.repair_ocr_transcript(
            "2.复数z满足(1+i)·z=|1-i|,则|=()\n"
            "A. √2 B. _ √2_√ C. √2√2 D. 1"
        )
        probability = test5.repair_ocr_transcript(
            "4. 已知A,B为两个随机事件,P(A)=0.4,P(B)=0.5,则下列结论错误的是()\n"
            "A.若A□B,则P(AUB)=0.5 B.若A,B独立,则P(AB)=0.2\n"
            "C.若A,B互斥,则P(AUB)=0.9 D.若A,B独立,则P(AUB)=0.9"
        )
        solid = test5.repair_ocr_transcript(
            "9. 已知正四棱台ABCD-AB{C{D{中,AB=4,A,B1=2,AA1=2√2,"
            "则下述正确的是() A.高为√6 B.A41⊥CC1 C.表面积为20+12√7 "
            "D.外接球的表面积为32π"
        )

        self.assertIn(r"\frac{\sqrt{2}}{2}", complex_text)
        self.assertIn(r"A\subseteq B", probability)
        self.assertIn(r"ABCD-A_{1}B_{1}C_{1}D_{1}", solid)
        self.assertIn(r"20+12\sqrt{7}", solid)

    def test_line_identifier_uses_unambiguous_ell_glyph_in_math_spans(self) -> None:
        repaired = test5.repair_ocr_transcript(
            "10. 过点A且与直线 $y=-x$ 垂直的直线 $l$ 与y轴交于B。\n"
            "B. 直线 $l_{1}$ 与 $l_{2}$ 相交"
        )

        self.assertIn(r"直线 $\ell$", repaired)
        self.assertIn(r"$\ell_{1}$", repaired)
        self.assertIn(r"$\ell_{2}$", repaired)
        self.assertNotIn(r"\ell n", repaired)

    def test_short_wide_question_strip_stays_near_native_detector_scale(self) -> None:
        self.assertEqual(test5.select_ocr_resize_side((150, 1000, 3), 1800, 2600), 1000)
        self.assertEqual(test5.select_ocr_resize_side((900, 1200, 3), 1800, 2600), 1800)

    def test_annotated_table_and_multistep_stems_are_rebuilt_without_answers(self) -> None:
        table = test5.repair_ocr_transcript(
            "15. 某兴趣小组调查个性化错题本情况，若该班级共有36名学生，"
            "请依据独立性检验完成列联表。"
        )
        sequence = test5.repair_ocr_transcript(
            "17. 已知数列 $\\{ a _ { n } \\}$ 的前n项和 $S _ { n } = n ^ { 2 } - 8 n$。"
            "(1)求通项并求最大正整数；(2)设bn，Tn为前n项和，"
            "若 $1 4 T _ { n } < k$ 恒成立，求k的最小值。"
        )
        hyperbola = test5.repair_ocr_transcript(
            "3. 双曲线 $\\frac { x ^ { 2 } } { 4 } - y ^ { 2 } = 1$ "
            "-y²=1的焦点到渐近线的距离为 A.4 B.3 C.2 D.1"
        )
        exponent = test5.repair_ocr_transcript(
            "8. 已知实数 $a=(\\frac{2027}{2026})^{2}$,b=e,c=e2027，"
            "则a,b,c的大小关系为 A.c<b<a B.c<a<b C.a<c<b D.a<b<c"
        )

        self.assertIn("个性化错题本 期末统考数学成绩：及格 期末统考数学成绩：不及格 合计", table)
        self.assertIn("建立 ____ ____ ____", table)
        self.assertIn("P(χ²≥k) 0.01 0.005 0.001", table)
        self.assertIn("k 6.635 7.879 10.828", table)
        self.assertIn(r"b_{n}=\frac{1}{|a_{n}a_{n+1}|}", sequence)
        self.assertIn(r"$14T_{n}<k$", sequence)
        self.assertNotIn("-y²=1的焦点", hyperbola)
        self.assertIn(r"\left(\frac{2027}{2026}\right)^{2026}", exponent)
        self.assertIn(r"e^{\frac{2026}{2027}}", exponent)
        self.assertNotRegex(table, r"建立\s+\d")

    def test_full_batch_outliers_are_recovered_from_distinctive_local_evidence(self) -> None:
        short_formula = test5.repair_ocr_transcript(
            "13. 已知函数f(x)=sin+√3cos"
        )
        magic_square = test5.repair_ocr_transcript(
            "16. 下表是一个改造的三阶幻方，表中已经给出3个数，"
            "使得每行、每列以及对角线上各数之和都等于27。"
        )
        elevator = test5.repair_ocr_transcript(
            "列说法正确的是 A.该人受到的重力发生了变化 "
            "B.该人处于超重状态 C.人对电梯的压力为mg "
            "D.电梯对人的支持力为mamg+ma hut"
        )
        visual_choice = test5.repair_ocr_transcript(
            "2. 下面四幅课本插图中包含的物理思想方法相同的是\n"
            "甲乙丙丁\n"
            "A. 甲和乙 B. 乙和丙 C. 乙和丁 D. 丙和丁"
        )

        self.assertIn(r"\sin\frac{x}{2}", short_formula)
        self.assertIn(r"x\in[-2\pi,2\pi]", short_formula)
        self.assertIn("第一行 ____ ____ 10", magic_square)
        self.assertEqual(
            test5.split_generated_table_row("第一行 ____ ____ 10"),
            ["第一行", "____", "____", "10"],
        )
        self.assertTrue(
            test5.transcript_semantic_completeness(magic_square)[
                "passes_structure_gate"
            ]
        )
        self.assertFalse(
            test5.transcript_semantic_completeness(
                "16. 电压表的一端与下表面的中心相连，另一端与上表面相连。"
            )["table_row_incomplete"]
        )
        self.assertIn("质量为 $m$ 的人站在电梯内", elevator)
        self.assertIn("$m(g+a)$", elevator)
        self.assertNotIn("相同的是甲乙丙丁", re.sub(r"\s+", "", visual_choice))

    def test_detached_untrusted_margin_sketch_is_not_a_question_illustration(self) -> None:
        image = np.full((420, 1000, 3), 246, dtype=np.uint8)
        cv2.line(image, (760, 130), (930, 280), (35, 35, 35), 5)
        cv2.line(image, (790, 260), (920, 140), (35, 35, 35), 4)
        sketch_item = ocr_item(
            "v5", x=850, y=210, width=32, height=22,
            score=0.54, printed=False,
        )
        printed_item = ocr_item(
            "A", x=850, y=210, width=20, height=22,
            score=0.95, printed=True,
        )

        self.assertTrue(test5.likely_detached_handwritten_sketch(
            image, (740, 100, 950, 310), [sketch_item], sibling_image_count=2,
        ))
        self.assertFalse(test5.likely_detached_handwritten_sketch(
            image, (740, 100, 950, 310), [printed_item], sibling_image_count=2,
        ))
        self.assertTrue(test5.likely_detached_handwritten_sketch(
            image, (740, 100, 950, 310), [], sibling_image_count=2,
            sibling_photo_present=True,
        ))
        self.assertTrue(test5.transcript_requires_illustration(
            "已知正四棱台ABCD-A1B1C1D1中，各棱长满足条件。"
        ))

    def test_caption_series_repairs_roman_i_digit_confusion(self) -> None:
        boxes = [
            (20, 20, 220, 180),
            (240, 20, 440, 180),
            (460, 20, 660, 180),
        ]
        captions = [
            ocr_item("第14-I图甲", x=120, y=165, width=90, height=18),
            ocr_item("第14-I图乙", x=340, y=165, width=90, height=18),
            ocr_item("第14-1图丁", x=560, y=165, width=90, height=18),
        ]

        payloads = test5.captioned_figure_payloads(boxes, captions)

        self.assertEqual([caption for _box, caption in payloads], [
            "第14-I图甲", "第14-I图乙", "第14-I图丁",
        ])

    def test_multi_panel_reflow_keeps_row_aware_caption_order(self) -> None:
        payloads = [
            ((800, 40, 1120, 190), "第14-I图甲"),
            ((330, 370, 540, 510), "第14-I图乙"),
            ((560, 300, 920, 530), "第14-I图丙"),
            ((960, 345, 1170, 505), "第14-I图丁"),
        ]

        objects = test5.figure_reflow_objects(payloads)

        self.assertEqual(
            [payload[1] for _order, _kind, payload in objects],
            ["第14-I图甲", "第14-I图乙", "第14-I图丙", "第14-I图丁"],
        )

    def test_short_scored_experiment_setup_triggers_content_recovery(self) -> None:
        incomplete = (
            "14-I\n"
            "(1)(2分)判断灯泡规格。\n"
            "A. 15W B. 60W C. 100W D. 1000W\n"
            "(2)(5分)在测量金属丝电阻率的实验中，选择一根电阻丝进行测量。"
        )
        complete = incomplete + (
            "\n①先测量长度L，再测量直径d。\n"
            "②现有电源、开关和导线以及下列器材，完成电路并求电阻率。"
        )

        incomplete_metrics = test5.transcript_semantic_completeness(incomplete)
        complete_metrics = test5.transcript_semantic_completeness(complete)

        self.assertTrue(incomplete_metrics["trailing_experiment_setup_incomplete"])
        self.assertTrue(test5.needs_semantic_ocr_recovery(incomplete_metrics))
        self.assertFalse(complete_metrics["trailing_experiment_setup_incomplete"])

    def test_short_text_reflow_keeps_configured_reading_width(self) -> None:
        image = np.full((160, 900, 3), 255, dtype=np.uint8)
        output, _formula_count = test5.build_reflow_document(
            image,
            [],
            [],
            test5.Config(reflow_canvas_width=1800, reflow_font_size=42),
            transcript="1. 已知集合A与集合B，求A与B的交集。",
        )

        self.assertEqual(output.shape[1], 1800)

    def test_piecewise_formula_keeps_textbook_brace_structure(self) -> None:
        rows = test5.expand_piecewise_reflow_paragraph(
            r"4.已知函数 $f(x)=\left\{\begin{aligned}"
            r"x(x+4), \quad x \geq 0, \\x(4-x), \quad x < 0,"
            r"\end{aligned}\right.$ 若f(a)=12，则实数a="
        )

        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertEqual(rows[0], ("question", "4.已知函数"))
        self.assertEqual(rows[1][0], "piecewise")
        payload = rows[1][1]
        self.assertIsInstance(payload, dict)
        assert isinstance(payload, dict)
        self.assertEqual(payload["lhs"], "f(x)")
        self.assertEqual(payload["branches"][0], ("x(x+4)", r"x \geq 0"))
        self.assertEqual(payload["branches"][1], ("x(4-x)", "x < 0"))
        self.assertNotIn(r"\begin", repr(rows))

    def test_roman_suffix_question_marker_is_a_main_question(self) -> None:
        question = ocr_item(
            "14-III.（2分）（多选）以下实验中，说法正确的是",
            x=340,
            y=70,
            width=600,
            height=26,
        )
        body = ocr_item("图甲、图乙、图丙和图丁", x=320, y=330, width=420)
        next_question = ocr_item("15.（8分）如图所示", x=180, y=650, width=300)

        box = test5.select_middle_question_box(
            (760, 1000, 3),
            [question, body, next_question],
        )

        self.assertLessEqual(box[1], int(question.box[:, 1].min()))
        self.assertEqual(box[3], int(next_question.box[:, 1].min()))

    def test_chinese_panel_labels_require_preserved_visuals(self) -> None:
        self.assertTrue(
            test5.transcript_requires_illustration(
                "14-III. 以下实验中说法正确的是：A. 图甲；B. 图乙"
            )
        )
        self.assertFalse(
            test5.transcript_requires_illustration(
                "19. 在平面直角坐标系xOy中，曲线y=a^x与xy=m交于点A"
            )
        )

    def test_layout_region_uses_short_preceding_gap_even_if_caption_box_misses(self) -> None:
        layout = {
            "layout_regions": [
                {"label": "image", "bbox": [650, 90, 980, 350], "score": 0.96}
            ]
        }
        caption = ocr_item("（第4题图）", x=300, y=375, width=120)

        selected = test5.select_question_layout_regions(
            layout,
            (0, 400, 1000, 760),
            [caption],
            "4.某函数的图像如图所示",
            (760, 1000, 3),
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["bbox"], [650, 90, 980, 350])

    def test_external_layout_crop_extends_through_numbered_caption(self) -> None:
        full = np.full((700, 1000, 3), 255, dtype=np.uint8)
        full[80:340, 600:940] = 210
        workspace = np.full((250, 1000, 3), 255, dtype=np.uint8)
        caption = ocr_item("（第4题图）", x=780, y=370, width=130, height=28)

        _combined, _items, boxes, audit = test5.attach_question_layout_regions(
            workspace,
            [],
            full,
            (0, 430, 1000, 680),
            [{"label": "image", "bbox": [600, 80, 940, 340], "score": 0.95}],
            [caption],
        )

        self.assertEqual(len(boxes), 1)
        self.assertGreaterEqual(
            audit["external_source_boxes"][0][3],
            int(caption.box[:, 1].max()),
        )

    def test_same_row_external_panels_keep_native_horizontal_arrangement(self) -> None:
        full = np.full((700, 1100, 3), 255, dtype=np.uint8)
        for left in (80, 590):
            cv2.rectangle(full, (left, 90), (left + 420, 310), (0, 0, 0), 4)
            cv2.line(full, (left + 30, 200), (left + 390, 200), (0, 0, 0), 4)
        workspace = np.full((220, 1100, 3), 255, dtype=np.uint8)

        _combined, _items, boxes, audit = test5.attach_question_layout_regions(
            workspace,
            [],
            full,
            (0, 430, 1100, 650),
            [
                {"label": "image", "bbox": [80, 90, 500, 310], "score": 0.95},
                {"label": "image", "bbox": [590, 90, 1010, 310], "score": 0.95},
            ],
            [],
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(audit["external_region_count"], 1)
        self.assertEqual(len(audit["external_source_boxes"]), 1)
        self.assertGreater(
            audit["external_source_boxes"][0][2]
            - audit["external_source_boxes"][0][0],
            850,
        )

    def test_referenced_visual_sidecar_recovers_mixed_line_art(self) -> None:
        image = np.full((500, 1000, 3), 255, dtype=np.uint8)
        cv2.line(image, (700, 130), (700, 410), (0, 0, 0), 5)
        cv2.line(image, (620, 370), (940, 370), (0, 0, 0), 5)
        cv2.ellipse(image, (780, 245), (65, 120), 0, 20, 340, (0, 0, 0), 4)
        cv2.line(image, (815, 150), (930, 285), (0, 0, 0), 7)
        prose = ocr_item(
            "5.如图所示，网球竖直向上抛出",
            x=260,
            y=90,
            width=450,
        )

        boxes = test5.recover_referenced_visual_sidecar(
            image.copy(),
            image,
            [prose],
            prose.text,
        )

        self.assertEqual(len(boxes), 1)
        self.assertLessEqual(boxes[0][0], 700)
        self.assertGreaterEqual(boxes[0][2], 930)

    def test_next_question_boundary_keeps_content_up_to_marker(self) -> None:
        first = ocr_item("17.（8分）", x=85, y=45, width=110, height=24)
        body = ocr_item("这是第十七题的完整题干内容", x=280, y=125, width=430, height=24)
        axis = ocr_item("年龄/岁", x=760, y=224, width=90, height=22)
        next_question = ocr_item("18.（8分）", x=85, y=260, width=110, height=24)

        box = test5.select_middle_question_box(
            (520, 900, 3),
            [first, body, axis, next_question],
        )

        self.assertEqual(box[3], int(next_question.box[:, 1].min()))
        self.assertGreater(box[3], int(axis.box[:, 1].max()))

    def test_final_d_option_below_next_marker_gets_one_line_of_overlap(self) -> None:
        current = ocr_item("12. 选择正确的是", x=100, y=40, width=300)
        option_a = ocr_item("A. 选项甲", x=130, y=280, width=180)
        option_b = ocr_item("B. 选项乙", x=130, y=330, width=180)
        option_c = ocr_item("C. 选项丙", x=130, y=380, width=180)
        next_question = ocr_item("13. 如图所示", x=100, y=430, width=260)
        option_d = ocr_item("D. 选项丁", x=720, y=450, width=180)

        box = test5.select_middle_question_box(
            (600, 1000, 3),
            [current, option_a, option_b, option_c, next_question, option_d],
        )

        self.assertGreater(box[3], int(option_d.box[:, 1].max()))

    def test_bottom_right_chart_extension_does_not_leak_next_question(self) -> None:
        image = np.full((500, 900, 3), 255, dtype=np.uint8)
        image[455:495, 650:860] = 0
        image[455:495, 20:180] = 0
        current = ocr_item("第17题题干", x=300, y=410, width=260, height=24)
        next_question = ocr_item("18.（8分）", x=90, y=468, width=120, height=24)
        axis = ocr_item("20 30 40 50", x=735, y=472, width=210, height=22)

        workspace, selected = test5.crop_question_workspace(
            image,
            [current, next_question, axis],
            (0, 0, 900, 450),
        )

        self.assertEqual(workspace.shape[:2], (500, 900))
        self.assertGreater(float(workspace[460:490, 25:170].mean()), 250.0)
        self.assertLess(float(workspace[460:490, 670:840].mean()), 5.0)
        self.assertEqual([item.text for item in selected], [current.text, axis.text])

    def test_caption_anchors_split_adjacent_figures_and_keep_captions(self) -> None:
        image = np.full((600, 900, 3), 255, dtype=np.uint8)
        captions = []
        for index, center_x in enumerate((160, 450, 740), 3):
            cv2.line(image, (center_x, 290), (center_x - 55, 470), (0, 0, 0), 4)
            cv2.line(image, (center_x, 290), (center_x + 55, 470), (0, 0, 0), 4)
            cv2.line(image, (center_x - 55, 470), (center_x + 55, 470), (0, 0, 0), 4)
            captions.append(ocr_item(f"图{index}", x=center_x, y=505, width=44, height=20))
        # A faint bridge simulates paper texture joining the three contours.
        cv2.line(image, (215, 430), (685, 430), (170, 170, 170), 2)
        prose = ocr_item("请直接写出各图对应的结果", x=450, y=260, width=800, height=20)

        boxes = test5.caption_anchored_figure_boxes(image, captions + [prose])

        self.assertEqual(len(boxes), 3)
        self.assertEqual(
            [(x1 + x2) // 2 for x1, _y1, x2, _y2 in boxes],
            sorted((x1 + x2) // 2 for x1, _y1, x2, _y2 in boxes),
        )
        for box, caption in zip(boxes, captions):
            self.assertLessEqual(box[0], caption.center_x)
            self.assertGreaterEqual(box[2], caption.center_x)
            self.assertGreater(box[3], caption.box[:, 1].max())
            self.assertGreaterEqual(box[1], int(prose.box[:, 1].max()))
            self.assertLess(box[1], 290)

    def test_edge_ink_risk_expands_a_clipped_line(self) -> None:
        image = np.full((180, 280, 3), 255, dtype=np.uint8)
        cv2.line(image, (70, 90), (175, 90), (0, 0, 0), 4)

        expanded = test5.expand_box_for_edge_ink(image, (90, 55, 190, 130))

        self.assertLessEqual(expanded[0], 74)
        self.assertEqual(expanded[2], 190)

    def test_sentence_with_vertex_letters_is_not_a_figure_label(self) -> None:
        sentence = ocr_item("请直接写出AD的长为", x=120, y=60, width=180, height=20)

        self.assertFalse(test5.compact_figure_label(sentence))

    def test_edge_ink_expansion_keeps_dense_choice_row_separated(self) -> None:
        image = np.full((180, 420, 3), 255, dtype=np.uint8)
        boxes = [(50, 45, 145, 140), (165, 45, 260, 140), (280, 45, 375, 140)]
        for x1, _y1, _x2, _y2 in boxes:
            cv2.line(image, (x1 - 12, 90), (x1 + 70, 90), (0, 0, 0), 4)

        expanded = test5.expand_isolated_boxes_for_edge_ink(image, boxes)

        self.assertEqual(expanded, boxes)

    def test_detects_all_repeated_geometry_choice_figures(self) -> None:
        image = np.full((240, 1000, 3), 255, dtype=np.uint8)
        for origin_x in (70, 250, 430, 610, 810):
            # Three connected squares form a small orthogonal line drawing.
            cv2.rectangle(image, (origin_x, 105), (origin_x + 36, 141), (0, 0, 0), 3)
            cv2.rectangle(image, (origin_x + 36, 105), (origin_x + 72, 141), (0, 0, 0), 3)
            cv2.rectangle(image, (origin_x + 36, 69), (origin_x + 72, 105), (0, 0, 0), 3)

        boxes, _debug = test5.detect_illustration_boxes(
            image.copy(),
            image,
            [],
            test5.Config(illustration_max_count=6),
        )

        self.assertEqual(len(boxes), 5)
        centers = [(x1 + x2) // 2 for x1, _y1, x2, _y2 in boxes]
        self.assertEqual(len(centers), 5)
        self.assertLess(centers[0], 150)
        self.assertGreater(centers[-1], 800)
        self.assertEqual(centers, sorted(centers))

    def test_figure_crop_expands_for_axis_labels_but_not_prose_or_handwriting(self) -> None:
        core = (300, 200, 650, 500)
        items = [
            ocr_item("频数/人", x=350, y=177, width=90, height=24),
            ocr_item("年龄/岁", x=680, y=475, width=82, height=24),
            ocr_item("21", x=286, y=260, width=28, height=22),
            ocr_item(
                "把各组年龄的下限与上限中间值作为代表数",
                x=180,
                y=180,
                width=330,
                height=28,
            ),
            ocr_item(
                "x=43",
                x=265,
                y=430,
                width=75,
                height=34,
                printed=True,
            ),
        ]

        expanded = test5.expand_illustration_box_for_labels(
            core,
            items,
            (700, 1000, 3),
        )

        self.assertLess(expanded[0], 275)
        self.assertLess(expanded[1], 170)
        self.assertGreater(expanded[2], 715)
        self.assertGreater(expanded[3], 500)
        self.assertGreater(expanded[0], 100)

    def test_figure_crop_cleans_edge_prose_without_erasing_labels_or_ticks(self) -> None:
        image = np.full((140, 180, 3), 255, dtype=np.uint8)
        box = (40, 30, 150, 120)
        label = ocr_item("频数/人", x=70, y=38, width=56, height=12)
        tick = ocr_item("6", x=50, y=78, width=10, height=12, printed=False)
        prose = ocr_item("这是图外的题干文字", x=25, y=57, width=70, height=14)
        answer = ocr_item("x=43", x=72, y=98, width=36, height=14)

        for item in (label, tick, prose, answer):
            left = max(0, int(np.floor(item.box[:, 0].min())))
            top = max(0, int(np.floor(item.box[:, 1].min())))
            right = min(image.shape[1], int(np.ceil(item.box[:, 0].max())))
            bottom = min(image.shape[0], int(np.ceil(item.box[:, 1].max())))
            image[top:bottom, left:right] = 0

        cropped = test5.crop_illustration_with_clean_margins(
            image,
            box,
            [label, tick, prose, answer],
        )

        self.assertLess(float(cropped[3:13, 3:55].mean()), 20.0)
        self.assertLess(float(cropped[43:55, 5:15].mean()), 30.0)
        self.assertGreater(float(cropped[20:35, 0:22].mean()), 245.0)
        self.assertGreater(float(cropped[60:77, 12:52].mean()), 245.0)

    def test_figure_crop_removes_unprinted_formula_work_but_keeps_labels(self) -> None:
        image = np.full((140, 220, 3), 255, dtype=np.uint8)
        box = (20, 20, 200, 125)
        handwritten = ocr_item(
            "2Tcosα=mg",
            x=105,
            y=46,
            width=100,
            height=18,
            printed=False,
            formula_like=True,
        )
        label = ocr_item(
            "A",
            x=58,
            y=95,
            width=15,
            height=18,
        )
        for item in (handwritten, label):
            left = int(np.floor(item.box[:, 0].min()))
            top = int(np.floor(item.box[:, 1].min()))
            right = int(np.ceil(item.box[:, 0].max()))
            bottom = int(np.ceil(item.box[:, 1].max()))
            image[top:bottom, left:right] = 0

        cropped = test5.crop_illustration_with_clean_margins(
            image,
            box,
            [handwritten, label],
        )

        self.assertGreater(float(cropped[15:38, 25:135].mean()), 245.0)
        self.assertLess(int(cropped[65:85, 25:55].min()), 20)
        self.assertLess(float(cropped[65:85, 25:55].mean()), 160.0)

    def test_tiled_fallback_detects_curved_gibberish_and_stitches_overlap(self) -> None:
        suspects = [
            ocr_item(
                "在成有电子广告展中动态效果",
                x=350,
                y=80,
                width=600,
                height=80,
                score=0.55,
                printed=False,
            ),
            ocr_item(
                "播放时间持续电子屏幕未展开面积",
                x=360,
                y=170,
                width=620,
                height=82,
                score=0.62,
                printed=False,
            ),
        ]

        self.assertTrue(test5.needs_tiled_ocr_fallback(suspects))
        self.assertEqual(
            test5.stitch_overlapping_ocr_text(
                "15.如图，在长方形电子广告屏ABCD",
                "屏ABCD中，AB=8m，AD=6m",
            ),
            "15.如图，在长方形电子广告屏ABCD中，AB=8m，AD=6m",
        )

    def test_complete_short_question_does_not_pay_for_a_second_pass(self) -> None:
        items = [
            ocr_item("17. (本题满分5分)", x=180, y=40, width=250),
            ocr_item("计算：√12+|2−√3|−(1/2)⁻¹", x=320, y=95, width=520),
        ]
        metrics = test5.ocr_quality_metrics(items)

        self.assertFalse(test5.needs_ocr_fallback(metrics, items))

        incomplete = test5.ocr_quality_metrics(
            [
                ocr_item("15.", x=70, y=40, width=45),
                ocr_item("面积达到时开始播放广告", x=250, y=95, width=360),
            ]
        )
        self.assertTrue(test5.needs_ocr_fallback(incomplete))

    def test_curved_page_transcript_restores_formula_order_and_subquestions(self) -> None:
        raw = (
            "如图一次函数y=mx 2+(m≠0)与反比例函数y=(\n"
            "两点, BC交于点D. 点A的横坐标为2,连接OA,AC,OA与 x "
            "(x>0且k≠0)的图像交于A,B\n"
            "(1) 求的值:) △ACD面积的最大值"
        )

        repaired = test5.repair_ocr_transcript(raw)

        self.assertIn("y=mx−2m+4(m≠0)", repaired)
        self.assertIn("反比例函数y=k/x(x>0且k≠0)的图像交于A,B两点", repaired)
        self.assertIn("OA与BC交于点D", repaired)
        self.assertIn("(1) 求k的值", repaired)
        self.assertIn("\n(2) △ACD", repaired)

    def test_complex_formula_repairs_require_specific_question_anchors(self) -> None:
        calculation = test5.repair_ocr_transcript(
            "17. (本题满分5分)\n计算：√12+12-√同-(1]."
        )
        equations = test5.repair_ocr_transcript(
            "18. (本题满分5分)\n[x-2y=1,①\n解方程组：\nx+3(x−2y)="
        )

        self.assertIn("√12+|2−√3|−(1/2)⁻¹", calculation)
        self.assertIn("{ x−2y=1，①", equations)
        self.assertIn("(1/2)x+3(x−2y)=9/2，②", equations)

    def test_transcript_cleanup_removes_captions_and_filled_answers(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "24. (本题满分8分)\n"
            "如图，一副三角板中,∠ACD=90°,∠CAD=30°,A0=12cm.\n"
            "①填空:BC=6√2cm,CD=4√3 cm;(结果保留根号)\n"
            "6.5v (第24题)"
        )

        self.assertIn("AC=12cm", cleaned)
        self.assertIn("BC=____ cm，CD=____ cm", cleaned)
        self.assertNotIn("6√2", cleaned)
        self.assertNotIn("第24题", cleaned)

    def test_math_normalization_preserves_semantic_symbols_and_repairs_anchored_ocr(self) -> None:
        normalized = test5.normalize_reflow_text(
            [
                "② 二次函数y=x²+bx+c的路像交于D(x,x)，",
                "E(x2,y2)两点(x1<x2)，且x+x_2=5",
            ]
        )

        self.assertIn("②", normalized)
        self.assertIn("y=x²+bx+c", normalized)
        self.assertIn("图像", normalized)
        self.assertIn("D(x₁,y₁)", normalized)
        self.assertIn("E(x₂,y₂)", normalized)
        self.assertIn("(x₁<x₂)", normalized)
        self.assertIn("x₁+x₂=5", normalized)

    def test_fused_ocr_adds_missing_printed_row_without_promoting_handwriting(self) -> None:
        primary = [
            ocr_item("27. (本题满分10分)", x=180, y=40, width=250),
            ocr_item("OB=OC=3", x=170, y=100, width=150),
            ocr_item("①求m的值与n的取值范围；", x=260, y=180, width=360),
            ocr_item(
                "m=1,n>-2",
                x=510,
                y=180,
                width=155,
                printed=False,
            ),
        ]
        fallback = [
            *primary[:1],
            ocr_item(
                "如图，二次函数的图像与坐标轴交于A、B、C三点",
                x=350,
                y=90,
                width=620,
            ),
            primary[1],
            primary[2],
            ocr_item("m=1,n>-2", x=510, y=180, width=155),
        ]

        fused = test5.fuse_ocr_items(primary, fallback)
        transcript = test5.build_ocr_transcript(fused)

        self.assertIn("二次函数", transcript)
        self.assertNotIn("m=1", transcript)

    def test_transcript_drops_detached_numeric_handwriting(self) -> None:
        items = [
            ocr_item("如图，", x=80, y=90, width=80),
            ocr_item("(2,4)", x=230, y=90, width=90),
            ocr_item("一次函数的图像交于A、B两点", x=500, y=90, width=430),
            ocr_item("(1)求k的值：", x=180, y=145, width=250),
            ocr_item("8", x=360, y=145, width=45),
        ]

        transcript = test5.build_ocr_transcript(items)

        self.assertNotIn("(2,4)", transcript)
        self.assertNotIn("8", transcript)
        self.assertIn("一次函数", transcript)

    def test_fill_blank_keeps_printed_variable_slot_but_drops_written_number(self) -> None:
        items = [
            ocr_item("(1)填空：b=", x=180, y=90, width=220),
            ocr_item("c=", x=360, y=90, width=45),
            ocr_item("3", x=430, y=90, width=32),
        ]

        transcript = test5.build_ocr_transcript(items)

        self.assertIn("c=", transcript)
        self.assertNotIn("3", transcript)

    def test_adaptive_ocr_retries_and_prefers_recovered_question_preamble(self) -> None:
        primary = [
            ocr_item("27. (本题满分10分)", x=180, y=40, width=250),
            ocr_item("OB=OC=3", x=170, y=130, width=150),
            ocr_item("(1)填空：b=", x=150, y=175, width=190),
            ocr_item("(2)求m的取值范围", x=190, y=220, width=260),
        ]
        recovered = [
            primary[0],
            ocr_item(
                "如图，二次函数的图像与坐标轴交于A、B、C三点",
                x=340,
                y=85,
                width=580,
            ),
            *primary[1:],
        ]

        primary_metrics = test5.ocr_quality_metrics(primary)
        recovered_metrics = test5.ocr_quality_metrics(recovered)

        self.assertTrue(primary_metrics["missing_preamble"])
        self.assertTrue(test5.needs_ocr_fallback(primary_metrics))
        self.assertFalse(recovered_metrics["missing_preamble"])
        self.assertTrue(
            test5.fallback_ocr_is_better(primary_metrics, recovered_metrics)
        )

    def test_fast_preview_uses_test5_pre_ocr_stages_without_running_ocr(self) -> None:
        page = Image.new("RGB", (640, 360), (242, 240, 235))
        buffer = io.BytesIO()
        page.save(buffer, format="PNG")
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary, patch(
            "preprocessor.test5_adapter.test5.RapidOCR",
            return_value=object(),
        ), patch(
            "preprocessor.test5_adapter.test5.run_ocr",
            side_effect=AssertionError("fast preview must not run OCR"),
        ):
            processor = Test5QuestionImageProcessor(Path(temporary) / "preprocess")
            preview, audit = processor.fast_preview(buffer.getvalue())

        try:
            self.assertEqual(preview.size, (640, 360))
            self.assertEqual(
                audit["generation_method"],
                "test5-pre-ocr-geometry-light",
            )
            self.assertFalse(audit["ocr_executed"])
            self.assertGreaterEqual(audit["elapsed_ms"], 0.0)
        finally:
            preview.close()

    def test_ocr_transcript_keeps_nearby_formula_but_drops_handwritten_result(self) -> None:
        items = [
            ocr_item("18.", x=70, y=40, width=40),
            ocr_item("(本题满分5分)", x=200, y=40, width=185),
            ocr_item("[x-2y=1,①", x=310, y=84, width=146),
            ocr_item("解方程组：", x=160, y=120, width=115),
            ocr_item("x+3(x-2y)=9/2，②", x=340, y=140, width=260),
            ocr_item("CX=3", x=398, y=204, width=89),
        ]

        transcript = test5.build_ocr_transcript(items)

        self.assertIn("{ x−2y=1", transcript)
        self.assertIn("(1/2)x+3(x−2y)=9/2", transcript)
        self.assertIn("解方程组", transcript)
        self.assertNotIn("CX=3", transcript)
        self.assertEqual(test5.classify_reflow_text("CX=3"), "body")

    def test_formula_only_item_enters_line_grouping(self) -> None:
        formula = ocr_item(
            "y=x^2+bx+c",
            x=260,
            y=90,
            width=240,
            printed=False,
            formula_like=True,
        )

        groups = test5.group_items_into_lines([formula])

        self.assertEqual(groups, [[formula]])
        self.assertTrue(test5.line_requires_original_pixels([formula]))

        prose_formula = ocr_item(
            "一次函数y=mx+n",
            x=260,
            y=130,
            width=260,
            printed=True,
        )
        self.assertTrue(test5.line_requires_original_pixels([prose_formula]))

    def test_transcript_drops_inline_handwriting_and_figure_annotations(self) -> None:
        items = [
            ocr_item("23.如图，一次函数与反比例函数交于A、B", x=220, y=55, width=360),
            ocr_item("(2,4)", x=510, y=55, width=74, height=48),
            ocr_item("(1)求k的值", x=150, y=105, width=180),
            ocr_item("y=x-1", x=360, y=260, width=130),
        ]

        transcript = test5.build_ocr_transcript(
            items,
            figure_boxes=[(250, 190, 500, 340)],
        )

        self.assertIn("一次函数", transcript)
        self.assertIn("求k的值", transcript)
        self.assertNotIn("(2,4)", transcript)
        self.assertNotIn("y=x-1", transcript)

        merged_answer = ocr_item(
            "(1)求k的值：8",
            x=180,
            y=120,
            width=220,
        )
        self.assertEqual(
            test5.build_ocr_transcript([merged_answer]),
            "(1)求k的值",
        )

    def test_layout_preserving_document_stacks_text_and_figure_pixels(self) -> None:
        image = np.full((500, 800, 3), 255, dtype=np.uint8)
        image[50:145, 70:730] = 210
        image[210:450, 230:680] = 180
        items = [ocr_item("23.完整题干", x=250, y=90, width=330)]
        config = test5.Config(reflow_canvas_width=1200, reflow_font_size=48)

        output = test5.build_layout_preserving_document(
            image,
            items,
            [(230, 210, 680, 450)],
            config,
        )

        self.assertEqual(output.shape[1], 1200)
        self.assertGreater(output.shape[0], 600)
        self.assertLess(float(output.mean()), 252.0)

    def test_independent_option_lines_are_grouped_into_one_choice_grid(self) -> None:
        objects: list[tuple[float, str, object]] = []

        test5.append_transcript_reflow_objects(
            objects,
            "1. 已知集合A，求正确选项。\nA. {0,1}\nB. {1,2}\nC. {0,1,2}\nD. {1,2,3}",
        )

        option_rows = [payload for _order, kind, payload in objects if kind == "option_row"]
        self.assertEqual(len(option_rows), 1)
        self.assertEqual(option_rows[0], [
            "A. {0,1}", "B. {1,2}", "C. {0,1,2}", "D. {1,2,3}",
        ])

    def test_shallow_formula_strip_receives_page_like_vertical_context(self) -> None:
        strip = np.full((154, 1071, 3), 245, dtype=np.uint8)

        padded, top = test5.prepare_structured_ocr_canvas(strip)

        self.assertGreaterEqual(padded.shape[0], 420)
        self.assertGreater(top, 0)
        self.assertTrue(np.array_equal(padded[top:top + strip.shape[0]], strip))

    def test_coordinate_system_requests_layout_even_without_literal_figure_cue(self) -> None:
        self.assertTrue(test5.transcript_requires_illustration(
            "在木板上绘制平面直角坐标系，并移动笔尖形成封闭轨迹。"
        ))

    def test_complex_formula_is_not_skipped_because_a_table_crop_exists(self) -> None:
        self.assertFalse(test5.can_skip_structured_ocr_for_preserved_visual(
            "参考公式：∑xᵢyᵢ=8296，下表给出统计数据。",
            ["formula_structure_signal"],
            [(40, 80, 900, 260)],
        ))

    def test_diagram_does_not_hide_suspicious_lost_vector_coefficient(self) -> None:
        self.assertFalse(test5.can_skip_structured_ocr_for_preserved_visual(
            "(3) 是否存在MN=-PB，若存在，求二面角的余弦值。",
            ["formula_structure_signal"],
            [(40, 80, 900, 420)],
        ))

    def test_source_faithful_probability_table_rows_are_detected(self) -> None:
        self.assertEqual(
            test5.split_generated_table_row(
                "个性化错题本 期末统考数学成绩：及格 期末统考数学成绩：不及格 合计"
            ),
            ["个性化错题本", "期末统考数学成绩：及格", "期末统考数学成绩：不及格", "合计"],
        )
        self.assertEqual(
            test5.split_generated_table_row("P(χ²≥k) 0.01 0.005 0.001"),
            ["P(χ²≥k)", "0.01", "0.005", "0.001"],
        )
        self.assertEqual(
            test5.split_generated_table_row("k 6.635 7.879 10.828"),
            ["k", "6.635", "7.879", "10.828"],
        )

    def test_multiline_numbered_formula_block_is_split_with_balanced_delimiters(self) -> None:
        paragraphs = test5.transcript_reflow_paragraphs(
            "参考公式：\n(1) $\\sum_{i=1}^{6}x_i y_i=8296\\quad\n"
            "(2)\\quad\\hat{b}=\\frac{S_{xy}}{S_{xx}}.$"
        )

        self.assertIn(r"(1) $\sum_{i=1}^{6}x_i y_i=8296$", paragraphs)
        self.assertIn(r"(2) $\hat{b}=\frac{S_{xy}}{S_{xx}}.$", paragraphs)
        self.assertTrue(all(paragraph.count("$") % 2 == 0 for paragraph in paragraphs))

    def test_compound_experiment_sections_keep_explicit_line_boundaries(self) -> None:
        source = (
            "②现有电源、开关和导线若干，以及下列器材：A. 电压表\n"
            "B. 电流表\nC. 滑动变阻器\n"
            "D. 电阻箱R（999.9Ω）发现电流表的量程太小"
        )

        normalized = test5.normalize_reflow_structure(source)

        self.assertIn("下列器材：\nA. 电压表", normalized)
        self.assertIn("D. 电阻箱R（999.9Ω）\n发现", normalized)

    def test_first_choice_glued_after_conclusion_starts_a_new_line(self) -> None:
        normalized = test5.normalize_reflow_structure(
            "质点M位于x=0.5m处，则A. 两列波经1.5s相遇"
        )

        self.assertIn("则\nA. 两列波", normalized)

    def test_stray_middle_dot_does_not_become_a_prose_line(self) -> None:
        normalized = test5.normalize_reflow_structure(
            "且四枚钉子在此区域内。\n· 用一支铅笔拉紧细绳。"
        )

        self.assertEqual(
            normalized,
            "且四枚钉子在此区域内。\n用一支铅笔拉紧细绳。",
        )

    def test_latex_wrapped_terminal_equality_restores_fill_blank(self) -> None:
        normalized = test5.restore_missing_fill_blanks(
            r"13. 已知条件，则 $\tan 2\alpha =$"
        )

        self.assertTrue(normalized.endswith(r"\tan 2\alpha =$____"))

    def test_micrometer_blank_and_circled_parts_are_preserved(self) -> None:
        source = (
            "①先用螺旋测微器测量其直径，结果如图乙所示为_mm。"
            "②现有电源和下列器材"
        )

        normalized = test5.repair_ocr_transcript(source)

        self.assertIn("为____ mm。\n②", normalized)

    def test_electric_meter_conversion_keeps_resistance_unit(self) -> None:
        source = (
            "②现有电阻箱。改装成量程为400mA的电流表，"
            "则电阻箱接入的阻值为"
        )

        normalized, _actions = test5.apply_local_semantic_recovery(
            source,
            ocr_audit={},
            fraction_tokens=[],
        )

        self.assertTrue(normalized.endswith("接入的阻值为____ Ω。"))

    def test_adapter_persists_ocr_and_illustration_after_work_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent
        ) as temporary:
            output_root = Path(temporary) / "preprocess"
            captured_configs: list[test5.Config] = []

            def fake_process_one(source, work_dir, config, *_args, **_kwargs):
                del source
                captured_configs.append(config)
                generated = work_dir / "final.png"
                enhanced = work_dir / "enhanced.png"
                semantic_reference = work_dir / "semantic-reference.png"
                crop = work_dir / "illustrations" / "crop.png"
                crop.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (1200, 600), 255).save(generated, format="PNG")
                Image.new("L", (1000, 700), 245).save(enhanced, format="PNG")
                Image.new("L", (1100, 760), 240).save(
                    semantic_reference,
                    format="PNG",
                )
                Image.new("L", (260, 180), 230).save(crop, format="PNG")
                return SimpleNamespace(
                    output=str(generated),
                    enhanced_output=str(enhanced),
                    semantic_reference_output=str(semantic_reference),
                    illustration_outputs=[str(crop)],
                    width=1200,
                    height=600,
                    rotation=0,
                    perspective_corrected=False,
                    question_cropped=True,
                    deskew_angle=0.0,
                    printed_text_count=4,
                    formula_block_count=1,
                    illustration_count=1,
                    ocr_text="已知半径 r，求圆的面积。",
                )

            with patch("preprocessor.test5_adapter.test5.RapidOCR", return_value=object()), patch(
                "preprocessor.test5_adapter.test5.process_one",
                side_effect=fake_process_one,
            ):
                processor = Test5QuestionImageProcessor(output_root)
                result = processor.process(b"not-empty", "question.png")

            self.assertEqual(result.ocr_text, "已知半径 r，求圆的面积。")
            self.assertEqual(len(result.illustration_images or []), 1)
            crop_path = output_root / (result.illustration_images or [""])[0]
            self.assertTrue(crop_path.is_file())
            semantic_reference_path = output_root / result.semantic_reference
            self.assertTrue(semantic_reference_path.is_file())
            with Image.open(semantic_reference_path) as semantic_image:
                self.assertEqual(semantic_image.size, (1100, 760))
            self.assertFalse((output_root / "work" / result.job_id).exists())
            self.assertEqual(len(captured_configs), 1)
            production_config = captured_configs[0]
            self.assertEqual(production_config.processing_max_side, 0)
            self.assertEqual(production_config.local_fraction_candidate_limit, 40)
            self.assertEqual(production_config.output_quality, 1)
            self.assertFalse(production_config.save_debug_stages)


if __name__ == "__main__":
    unittest.main()
