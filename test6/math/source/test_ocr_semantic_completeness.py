from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

import test5_preprocess as test5


def item(text: str, x: float, y: float, width: float, height: float = 24.0) -> test5.OCRItem:
    box = np.asarray(
        [
            [x - width / 2, y - height / 2],
            [x + width / 2, y - height / 2],
            [x + width / 2, y + height / 2],
            [x - width / 2, y + height / 2],
        ],
        dtype=np.float32,
    )
    return test5.OCRItem(
        box=box,
        text=text,
        score=0.98,
        width=width,
        height=height,
        center_x=x,
        center_y=y,
        printed=True,
    )


class OCRSemanticCompletenessTests(unittest.TestCase):
    def test_complete_table_rows_become_one_aligned_text_table(self) -> None:
        objects: list[tuple[float, str, object]] = []
        test5.append_transcript_reflow_objects(
            objects,
            "5. 成绩如下表：\n环数 7环 8环 9环 10环\n次数 1 4 3 2\n求中位数。",
        )

        tables = [payload for _, kind, payload in objects if kind == "text_table"]
        self.assertEqual(
            [["环数", "7环", "8环", "9环", "10环"], ["次数", "1", "4", "3", "2"]],
            tables[0],
        )

    def test_inconsistent_table_column_count_keeps_visual_fallback(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "16. 数据如下表格：\n轻度人数 1 2 3 1 2\n重度人数 4 3 6 4 4\n良好人数 25 25 11\n总人数 30 30 20 10 10"
        )

        self.assertTrue(audit["table_row_incomplete"])

    def test_external_layout_cleanup_removes_arithmetic_but_keeps_geometry_relation(self) -> None:
        crop = np.full((100, 100, 3), 210, dtype=np.uint8)
        crop[32:58, 2:28] = 0
        crop[59:86, 0:32] = 0
        cleaned, removed = test5.clean_external_layout_annotations(
            crop,
            [10, 10, 110, 110],
            (10, 10, 110, 110),
            [item("t=3", 25, 55, 22), item("AB=10", 25, 82, 32)],
        )

        self.assertEqual(["t=3"], removed)
        self.assertGreater(int(np.count_nonzero(cleaned != crop)), 0)
        self.assertTrue(np.all(cleaned[72, 15] == 0))

    def test_removes_handwritten_numeric_answer_before_result_qualifier(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "16. 如图，折痕EF的长度为 322 结果保留根号）。"
        )

        self.assertIn("折痕EF的长度为____结果保留根号）", cleaned)
        self.assertNotIn("322", cleaned)

    def test_structure_gate_detects_options_without_stem(self) -> None:
        incomplete = test5.transcript_semantic_completeness(
            "A. 1  B. 2  C. 3  D. 4"
        )
        complete = test5.transcript_semantic_completeness(
            "1. 下列结论正确的是：\nA. 1  B. 2  C. 3  D. 4"
        )

        self.assertTrue(incomplete["stem_missing"])
        self.assertFalse(incomplete["passes_structure_gate"])
        self.assertTrue(complete["passes_structure_gate"])

    def test_section_heading_without_any_question_is_missing_a_stem(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "三、填空题：本题共3小题，每小题5分，共15分"
        )

        self.assertTrue(audit["stem_missing"])
        self.assertFalse(audit["passes_structure_gate"])

    def test_structure_gate_detects_partial_options_and_missing_parts(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "题目条件完整。\n(2) 求速度。\nA. 1  C. 3"
        )

        self.assertTrue(audit["option_sequence_incomplete"])
        self.assertTrue(audit["subquestion_gap"])

    def test_three_contiguous_choices_are_valid_but_marked_ambiguous(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "8. 下列说法正确的是。\nA. 甲\nB. 乙\nC. 丙"
        )

        self.assertFalse(audit["option_sequence_incomplete"])
        self.assertTrue(audit["choice_count_ambiguous"])

    def test_empty_formula_content_in_three_choices_is_incomplete(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "5. 下列关系正确的是。\nA. B. C. $r_1/r_2$"
        )

        self.assertTrue(audit["empty_option_content"])
        self.assertFalse(audit["passes_structure_gate"])

    def test_standalone_second_subpart_is_not_reported_as_an_ocr_gap(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "(2) 用测量电路完成实验，并判断结果是否接近真实值。"
        )

        self.assertFalse(audit["subquestion_gap"])

    def test_roman_section_prefix_keeps_first_subquestion_number(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "14-I(1)(2分)判断灯泡规格。\n(2)(5分)测量金属丝的电阻率。"
        )

        self.assertEqual(audit["subquestion_numbers"], [1, 2])
        self.assertFalse(audit["subquestion_gap"])

    def test_mathematical_number_array_is_not_misclassified_as_data_table(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "11. 杨辉三角揭示二项式系数在三角形数表中的排列规律。\n"
            "如图所示，记P-数表第i行第j个数为P(i,j)。\n"
            "A. P(4,4)=12 B. 正确 C. 余数为1 D. 恒等式成立"
        )

        self.assertFalse(audit["table_row_incomplete"])

    def test_full_page_figure_box_cannot_delete_question_prose(self) -> None:
        prose = [
            item("17. 如图所示，物体沿斜面运动。", 310, 70, 520),
            item("(1) 求物体到达底端时的速度。", 290, 120, 470),
        ]
        boxes, rejected = test5.sanitize_illustration_boxes(
            [(0, 0, 900, 500)],
            prose,
            (500, 900, 3),
        )

        self.assertEqual(boxes, [])
        self.assertEqual(rejected, 1)
        self.assertIn("物体沿斜面运动", test5.build_ocr_transcript(prose, [(0, 0, 900, 500)]))

    def test_medium_texture_box_cannot_consume_multiple_prose_lines(self) -> None:
        prose = [
            item("已知衰变释放的核能全部转化为粒子的动能。", 450, 150, 560),
            item("若一个静止的中子衰变后放出质子，则判断下列说法。", 450, 215, 650),
        ]

        boxes, rejected = test5.sanitize_illustration_boxes(
            [(220, 90, 730, 300)],
            prose,
            (700, 1000, 3),
        )

        self.assertEqual(boxes, [])
        self.assertEqual(rejected, 1)

    def test_texture_box_overlapping_prose_tails_is_rejected(self) -> None:
        prose = [
            item("已知衰变释放的核能全部转化为粒子的动能。", 380, 150, 700),
            item("若一个静止的中子衰变后放出质子，则判断下列说法。", 390, 215, 720),
        ]

        boxes, rejected = test5.sanitize_illustration_boxes(
            [(520, 90, 850, 300)],
            prose,
            (700, 1000, 3),
        )

        self.assertEqual(boxes, [])
        self.assertEqual(rejected, 1)

    def test_pp_doclayout_visual_region_is_used_only_for_visual_question(self) -> None:
        audit = {
            "available": True,
            "engine": "local-pp-doclayoutv2",
            "layout_regions": [
                {"label": "image", "bbox": [420, 180, 760, 430], "score": 0.92},
            ],
        }
        boxes, evidence = test5.structured_layout_illustration_boxes(
            audit,
            (600, 900, 3),
            [item("16. 如图，在菱形ABCD中求EF。", 260, 80, 460)],
            "16. 如图，在菱形ABCD中求EF。",
            6,
        )
        text_only, _ = test5.structured_layout_illustration_boxes(
            audit,
            (600, 900, 3),
            [item("16. 计算下列各式。", 220, 80, 360)],
            "16. 计算下列各式。",
            6,
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(evidence["accepted_region_count"], 1)
        self.assertEqual(text_only, [])

    def test_table_label_is_accepted_for_explicit_schematic_figure(self) -> None:
        boxes, evidence = test5.structured_layout_illustration_boxes(
            {
                "available": True,
                "layout_regions": [
                    {"label": "table", "bbox": [420, 180, 760, 430], "score": 0.92},
                ],
            },
            (600, 900, 3),
            [item("8. 如图所示，分析电路。", 250, 80, 430)],
            "8. 如图所示，分析电路。",
            6,
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(evidence["accepted_label_counts"], {"table": 1})

    def test_numbered_caption_selects_only_matching_external_diagram(self) -> None:
        regions = test5.select_question_layout_regions(
            {
                "layout_regions": [
                    {"label": "image", "bbox": [100, 120, 380, 350], "score": 0.91},
                    {"label": "image", "bbox": [500, 125, 820, 355], "score": 0.94},
                ]
            },
            (0, 440, 900, 620),
            [
                item("（第14题）", 245, 380, 130),
                item("（第16题）", 660, 382, 130),
                item("16. 如图，在菱形ABCD中求EF。", 300, 470, 520),
            ],
            "16. 如图，在菱形ABCD中求EF。",
            (700, 900, 3),
        )

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["bbox"], [500, 125, 820, 355])

    def test_axis_ticks_are_not_treated_as_question_markers(self) -> None:
        records = [
            item("17.（12分）如图所示，一竖直长导线通有恒定电流。", 430, 80, 720),
            item("0.51MeV/c²，中子质量为939.57MeV/c²。", 460, 300, 610),
            item("4.0 B/10T", 420, 650, 150),
            item("1.5 2.0 2.5 3.0 3.5 4.0", 480, 710, 360),
        ]

        self.assertEqual(
            test5.select_middle_question_box((800, 1000, 3), records),
            (0, 0, 1000, 800),
        )

    def test_previous_question_option_is_trimmed_before_single_marker(self) -> None:
        records = [
            item("D. 上一道题的最后一个选项", 280, 20, 430),
            item("7. 某国际科研小组观察到中子衰变。", 360, 55, 600),
            item("A. 强相互作用", 220, 120, 260),
        ]

        box = test5.select_middle_question_box((500, 900, 3), records)

        self.assertGreater(box[1], 0)
        self.assertLess(box[1], 55)

    def test_numbered_figure_caption_recovers_complete_diagram_band(self) -> None:
        image = np.full((520, 900, 3), 255, dtype=np.uint8)
        cv2.line(image, (620, 210), (540, 390), (0, 0, 0), 4)
        cv2.line(image, (620, 210), (730, 390), (0, 0, 0), 4)
        cv2.line(image, (540, 390), (730, 390), (0, 0, 0), 4)
        caption = item("第17题图", 635, 430, 90, 22)

        boxes = test5.caption_anchored_figure_boxes(image, [caption])

        self.assertEqual(len(boxes), 1)
        self.assertLessEqual(boxes[0][1], 210)
        self.assertGreaterEqual(boxes[0][3], int(caption.box[:, 1].max()))

    def test_named_question_figure_caption_recovers_complete_diagram_band(self) -> None:
        image = np.full((520, 900, 3), 255, dtype=np.uint8)
        cv2.line(image, (620, 210), (540, 390), (0, 0, 0), 4)
        cv2.line(image, (620, 210), (730, 390), (0, 0, 0), 4)
        cv2.line(image, (540, 390), (730, 390), (0, 0, 0), 4)
        caption = item("第17题图杨辉三角", 600, 430, 180, 22)

        boxes = test5.caption_anchored_figure_boxes(
            image, [caption], target_question_number=17
        )

        self.assertEqual(len(boxes), 1)
        self.assertLessEqual(boxes[0][1], 210)
        self.assertGreaterEqual(boxes[0][3], int(caption.box[:, 1].max()))

    def test_roman_section_figure_caption_recovers_complete_diagram_band(self) -> None:
        image = np.full((520, 900, 3), 255, dtype=np.uint8)
        cv2.line(image, (620, 210), (540, 390), (0, 0, 0), 4)
        cv2.line(image, (620, 210), (730, 390), (0, 0, 0), 4)
        cv2.line(image, (540, 390), (730, 390), (0, 0, 0), 4)
        caption = item("第14-I图甲", 635, 430, 120, 22)

        boxes = test5.caption_anchored_figure_boxes(
            image, [caption], target_question_number=14
        )

        self.assertEqual(len(boxes), 1)
        self.assertLessEqual(boxes[0][1], 210)
        self.assertGreaterEqual(boxes[0][3], int(caption.box[:, 1].max()))

    def test_numbered_figure_and_backup_figure_are_both_recovered(self) -> None:
        image = np.full((560, 900, 3), 255, dtype=np.uint8)
        for center_x in (280, 650):
            cv2.line(image, (center_x, 165), (center_x, 405), (0, 0, 0), 4)
            cv2.line(image, (center_x - 125, 350), (center_x + 125, 350), (0, 0, 0), 4)
            cv2.line(image, (center_x - 100, 390), (center_x + 95, 205), (0, 0, 0), 4)
        captions = [
            item("（第27题）", 280, 450, 110, 24),
            item("备用图", 650, 450, 80, 24),
        ]

        boxes = test5.caption_anchored_figure_boxes(
            image,
            captions,
            target_question_number=27,
        )

        self.assertEqual(len(boxes), 1)
        self.assertLess(boxes[0][0], 280)
        self.assertGreater(boxes[0][2], 650)
        self.assertGreaterEqual(boxes[0][3], 462)

    def test_page_account_noise_is_removed(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "17. 这是需要保留的题干。\n微信公众号：测试账号 QQ:2754808740"
        )

        self.assertIn("需要保留的题干", cleaned)
        self.assertNotIn("QQ", cleaned)
        self.assertNotIn("公众号", cleaned)

    def test_exam_watermark_and_page_footer_are_removed(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "7. 这是需要保留的题干。浙考神墙750\n数学·第2页（共4页）"
        )

        self.assertIn("需要保留的题干", cleaned)
        self.assertNotIn("浙考神墙", cleaned)
        self.assertNotIn("第2页", cleaned)

    def test_table_digits_are_not_filtered_as_handwritten_answers(self) -> None:
        row = [
            item("次数", 80, 100, 52),
            item("1", 150, 100, 18),
            item("4", 210, 100, 18),
            item("3", 270, 100, 18),
            item("2", 330, 100, 18),
        ]

        kept = test5.filter_inline_annotations(row, 24.0)

        self.assertEqual([entry.text for entry in kept], ["次数", "1", "4", "3", "2"])

    def test_equation_overlapping_figure_box_stays_in_transcript(self) -> None:
        equation = item("x-2y=1", 310, 150, 150)

        suppressed = test5.item_text_belongs_to_figure(
            equation,
            [(0, 0, 700, 400)],
        )

        self.assertFalse(suppressed)

    def test_locally_observed_fraction_rebuilds_ad_screen_stem(self) -> None:
        recovered, actions = test5.apply_local_semantic_recovery(
            "15. 如图，在长方形电子广告屏ABCD中，AB=8m，AD=6m。\n"
            "动点P以2m/s运动，播放时间持续3s。",
            ocr_audit={},
            fraction_tokens=["1/3"],
        )

        self.assertIn("电子屏面积的1/3", recovered)
        self.assertIn("未展开的面积", recovered)
        self.assertIn("rebuild_skewed_ad_screen_stem_from_local_evidence", actions)

    def test_nested_subquestion_gap_is_reported(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "25. 如图，完成下列问题。\n①设圆的半径为r，求弦DF。"
        )

        self.assertTrue(audit["nested_subquestion_incomplete"])
        self.assertFalse(audit["passes_structure_gate"])

    def test_only_visual_questions_keep_illustration_candidates(self) -> None:
        self.assertTrue(
            test5.transcript_requires_illustration(
                "5. 某射手射击10次，成绩如下表：环数7环，次数1次。"
            )
        )
        self.assertTrue(
            test5.transcript_requires_illustration(
                "16. 如图，在菱形ABCD中作折痕EF。"
            )
        )
        self.assertTrue(
            test5.transcript_requires_illustration(
                "25. 图，△ABC中，D、E分别为边AB、AC上一点。"
            )
        )

    def test_complete_visual_question_skips_formula_only_refinement(self) -> None:
        transcript = (
            "25. 如图，△ABC中，D、E分别为边AB、AC上一点，且DC=DE。\n"
            "(1)求证：∠A=∠DCF；\n"
            "(2)若∠ACB=90°，①求弦DF长度；②求弦DF长度的最小值。"
        )

        self.assertTrue(
            test5.can_skip_structured_ocr_for_preserved_visual(
                transcript,
                ["formula_structure_signal"],
                [(586, 250, 897, 466)],
            )
        )
        self.assertFalse(
            test5.can_skip_structured_ocr_for_preserved_visual(
                transcript,
                ["semantic_structure_gap", "formula_structure_signal"],
                [(586, 250, 897, 466)],
            )
        )
        flattened = (
            "14. 独立重复抛掷一枚质地均匀硬币,每次抛出正面和反面的概率均为 1,"
            "记录累计正面次数Hn和反面次数Tn。\n"
            "(1)求正面获胜概率。"
        )
        self.assertFalse(
            test5.can_skip_structured_ocr_for_preserved_visual(
                flattened,
                ["formula_structure_signal", "flattened_script_signal"],
                [(586, 250, 897, 466)],
            )
        )

    def test_formula_only_refinement_replaces_only_anchored_math_text(self) -> None:
        source = [item("2x>12", 120, 90, 100)]
        audit = {
            "formula_items": [
                {
                    "bbox": [65, 75, 175, 105],
                    "latex": r"2^{x}>\frac{1}{2}",
                    "score": 0.96,
                },
                {
                    "bbox": [420, 300, 520, 340],
                    "latex": r"x=7",
                    "score": 0.99,
                },
            ]
        }

        refined, replaced = test5.fuse_formula_only_ocr_items(source, audit)

        self.assertEqual(replaced, 1)
        self.assertTrue(any(r"\frac{1}{2}" in entry.text for entry in refined))
        self.assertFalse(any("x=7" in entry.text for entry in refined))

    def test_broken_formula_tail_forces_full_recovery(self) -> None:
        metrics = test5.transcript_semantic_completeness(
            "1. 设集合A={x|x²+2x<0}，集合B=}\n"
            "A. (-2,0) B. (-1,+∞) C. (-1,0) D. (-∞,-2)"
        )
        reasons = test5.structured_ocr_recovery_reasons(
            "1. 设集合A={x|x²+2x<0}，集合B=}\n"
            "A. (-2,0) B. (-1,+∞) C. (-1,0) D. (-∞,-2)",
            [item("集合B=}", 200, 100, 120)],
        )

        self.assertTrue(metrics["symbol_damage"])
        self.assertIn("semantic_structure_gap", reasons)
        self.assertFalse(test5.formula_only_recovery_is_sufficient(reasons))

    def test_formula_only_mode_never_appends_public_validation_section(self) -> None:
        primary = "3. 已知函数f(x)=x²，求最值。"
        audit = {
            "request_mode": "formula_only",
            "formula_count": 1,
            "formulas": [r"f(x)=x^{2}"],
            "selected": False,
        }

        recovered, action = test5.fuse_structured_ocr_transcript(
            primary,
            r"$f(x)=x^{2}$",
            audit,
        )

        self.assertEqual(recovered, primary)
        self.assertEqual(action, "")
        self.assertNotIn("原图公式结构校验", recovered)

    def test_single_variable_like_option_does_not_fail_non_choice_problem(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "17. 求带电粒子进入磁场后的运动半径。\nB.方向垂直纸面向外。"
        )

        self.assertFalse(audit["option_sequence_incomplete"])

    def test_adjacent_top_level_question_is_a_semantic_failure(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "12. 当前题完整。\nA. 甲 B. 乙 C. 丙 D. 丁\n13. 下一题被错误带入。"
        )

        self.assertTrue(audit["multiple_question_blocks"])
        self.assertFalse(audit["passes_structure_gate"])

    def test_repeated_same_question_number_is_a_semantic_failure(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "5. 人造卫星做圆周运动，5. 人造卫星做匀速圆周运动"
        )

        self.assertTrue(audit["repeated_question_number"])
        self.assertFalse(audit["passes_structure_gate"])

    def test_inline_duplicate_question_prefix_is_removed(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "9. 已知函数 9. 已知函数f(x)=tan x，则求其周期。"
        )

        self.assertEqual(cleaned.count("9."), 1)
        self.assertTrue(cleaned.startswith("9. 已知函数f"))

    def test_adjacent_question_is_trimmed_without_second_ocr_call(self) -> None:
        primary = (
            "6. 当前题完整。\nA. 甲 B. 乙 C. 丙 D. 丁\n"
            "7. 下一题被错误带入。"
        )

        recovered, audit, action = test5.recover_structured_ocr_from_full_page(
            primary, np.full((20, 20, 3), 255, dtype=np.uint8)
        )

        self.assertNotIn("7.", recovered)
        self.assertEqual(action, "trim_adjacent_question_block")
        self.assertEqual(audit["selection_reason"], "trim_adjacent_question_block")

    def test_false_adjacent_number_cannot_trim_recognized_subquestions(self) -> None:
        primary = (
            "27. 如图，已知二次函数满足条件。\n"
            "3.\n"
            "(1) 填空：b=____，c=____。\n"
            "(2) 一次函数与二次函数交于D、E两点。\n"
            "①求参数范围；\n"
            "②探究是否存在实数n满足条件。"
        )

        with patch.object(test5, "request_structured_ocr", return_value=("", {})):
            recovered, _audit, action = test5.recover_structured_ocr_from_full_page(
                primary, np.full((20, 20, 3), 255, dtype=np.uint8)
            )

        self.assertEqual(recovered, primary)
        self.assertEqual(action, "")

    def test_detached_number_before_first_subquestion_is_removed(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "27. 如图，已知二次函数满足条件。\n"
            "3.\n"
            "(1) 填空：b=____，c=____。\n"
            "(2) 求参数范围。\n"
            "①求m的值；\n"
            "②说明理由。"
        )

        self.assertNotIn("\n3.\n", cleaned)
        self.assertIn("(1) 填空", cleaned)
        self.assertIn("(2) 求参数范围", cleaned)
        self.assertIn("①求m的值", cleaned)
        self.assertIn("②说明理由", cleaned)

    def test_curved_quadratic_subquestion_keeps_printed_coordinates_and_reason(self) -> None:
        source = (
            "27.(本题满分10分)\n"
            "如图,二次函数y=x²+bx+c的图像与x轴正半轴交于A,B两点,"
            "与y轴正半轴交于点C,\nOB=OC=3.\n"
            "(1)填空:b=\n"
            "(2)一次函数y=mx+n(m≠0)的图像与二次函数y=x²+bx+c的图像"
            "交于D(x,i),\nE(x₂,y₂)两点(x₁<x₂),与x轴交于点F,"
            "且x₁+x₂=5点F,x+x2=5\n"
            "①求m的值与n的取值范围;\n"
            "②试探究是否存在实数n,使得DE=2DF,若存在,求出n的值;"
            "若不存在,请说明"
        )

        recovered, _actions = test5.apply_local_semantic_recovery(
            source,
            ocr_audit={},
            fraction_tokens=[],
        )

        self.assertIn("D(x₁,y₁)", recovered)
        self.assertIn("E(x₂,y₂)", recovered)
        self.assertEqual(recovered.count("x₁+x₂=5"), 1)
        self.assertNotIn("点F,x+x2=5", recovered)
        self.assertTrue(recovered.endswith("若不存在，请说明理由。"))

    def test_missing_ellipse_equation_and_point_trigger_formula_recovery(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "18. 已知椭圆C: =1(a>b>0)过点离心率为一，O是坐标原点。\n"
            "1)求C的方程；\n(2)求点Q的轨迹方程。"
        )

        self.assertTrue(audit["formula_gap"])
        self.assertFalse(audit["passes_structure_gate"])

    def test_dropped_ellipse_fraction_and_eccentricity_trigger_formula_gap(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "18.(17分) 已知椭圆C: +y²=1(a>1)的离心率为 "
            "P(x0,y0)(x0>1)为椭圆C上的动点。\n"
            "(1)求椭圆C的标准方程；\n(2)证明结论；\n(3)求周长最小值。"
        )

        self.assertTrue(audit["formula_gap"])
        self.assertFalse(audit["passes_structure_gate"])

    def test_missing_left_parenthesis_on_first_subpart_is_restored(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "18. 完整题干。\n1）求方程；\n(2)求轨迹。"
        )

        self.assertIn("(1)求方程", cleaned)

    def test_detached_circled_list_item_is_not_a_subquestion_gap(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "5. 已知三组数据：①第一组；②第二组；\n③第三组的方差为S²。"
        )

        self.assertFalse(audit["subquestion_gap"])

    def test_detached_labels_are_trimmed_before_real_question_number(self) -> None:
        recovered, actions = test5.apply_local_semantic_recovery(
            "C P\n8. 定义：这是完整题干。",
            ocr_audit={},
            fraction_tokens=[],
        )

        self.assertTrue(recovered.startswith("8. 定义"))
        self.assertNotIn("C P", recovered)
        self.assertIn("drop_detached_labels_before_question_number", actions)

    def test_turntable_trailing_diagram_labels_are_removed(self) -> None:
        recovered, _actions = test5.apply_local_semantic_recovery(
            "12. 将圆形转盘分成8个面积相等的扇形。\n"
            "指针指向红色区域的概率为\nA 50\n黄红 D E",
            ocr_audit={},
            fraction_tokens=[],
        )

        self.assertTrue(recovered.endswith("概率为____。"))
        self.assertNotIn("A 50", recovered)

    def test_overlapping_figure_refinements_are_unioned(self) -> None:
        merged = test5.deduplicate_overlapping_illustration_boxes(
            [(10, 10, 110, 110), (70, 40, 170, 140)]
        )

        self.assertEqual(merged, [(10, 10, 170, 140)])

    def test_structured_formula_ocr_replaces_flattened_fraction_and_exponent(self) -> None:
        audit = {
            "formula_count": 3,
            "formulas": [r"2^{x} > \frac{1}{2}"],
            "selected": False,
        }
        recovered, action = test5.fuse_structured_ocr_transcript(
            "1. 设集合B={x|2x>12}，则选择正确选项。\nA. 1 B. 2 C. 3 D. 4",
            "1. 设集合 $B=\\{x\\mid 2^{x}>\\frac{1}{2}\\}$，则选择正确选项。\n"
            "A. 1 B. 2 C. 3 D. 4",
            audit,
        )

        self.assertIn(r"2^{x}>\frac{1}{2}", recovered)
        self.assertEqual(action, "select_local_structured_formula_ocr")
        self.assertTrue(audit["selected"])

    def test_formula_only_result_is_appended_without_deleting_primary_prose(self) -> None:
        audit = {
            "formula_count": 1,
            "formulas": [r"\frac{x^2}{a^2}+\frac{y^2}{b^2}=1"],
            "selected": False,
        }
        recovered, action = test5.fuse_structured_ocr_transcript(
            "16. 已知双曲线的左顶点到渐近线的距离，求标准方程。",
            r"$\frac{x^2}{a^2}+\frac{y^2}{b^2}=1$",
            audit,
        )

        self.assertIn("已知双曲线", recovered)
        self.assertIn("原图公式结构校验", recovered)
        self.assertEqual(action, "append_local_formula_structure")

    def test_flattened_scientific_exponent_is_restored_locally(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "某量为9.1×10-31kg，另一量为6.63×10−34J·s。"
        )

        self.assertIn("×10^{-31}kg", cleaned)
        self.assertIn("×10^{-34}J", cleaned)

    def test_trailing_handwritten_solution_is_removed(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "13. 如图所示，求下列两问。\n"
            "(1) 求函数定义域。\n"
            "(2) 求函数最值。\n"
            "由判别式可知\n"
            "$b^2-4ac>0$"
        )

        self.assertTrue(cleaned.endswith("(2) 求函数最值。"))
        self.assertNotIn("判别式", cleaned)
        self.assertNotIn("b^2", cleaned)

    def test_handwritten_result_cannot_change_a_printed_calculation_operator(self) -> None:
        primary = (
            "17.（本题满分5分）\n"
            "计算：√12+|2−√3|−(1/2)⁻¹。"
        )
        structured = (
            "17.（本题满分5分）\n"
            "计算：$\\sqrt{12}+|2-\\sqrt{3}|=(\\frac{1}{2})^{-1}$\n"
            "$=\\sqrt{3}$"
        )
        audit = {
            "formula_count": 2,
            "formulas": [
                r"\sqrt{12}+|2-\sqrt{3}|=(\frac{1}{2})^{-1}",
                r"=\sqrt{3}",
            ],
            "selected": False,
        }

        recovered, action = test5.fuse_structured_ocr_transcript(
            primary, structured, audit
        )

        self.assertEqual(recovered, primary)
        self.assertEqual(action, "")
        self.assertEqual(
            audit["selection_reason"],
            "operator_change_near_handwriting_rejected",
        )

    def test_pathological_formula_output_cannot_replace_primary_ocr(self) -> None:
        primary = "7. 已知函数满足条件，判断下列说法。\nA. 甲 B. 乙 C. 丙 D. 丁"
        repeated = " ".join("1" for _ in range(80))
        audit = {"formula_count": 1, "formulas": [repeated], "selected": False}

        recovered, action = test5.fuse_structured_ocr_transcript(
            primary,
            f"7. 已知函数满足条件。 ${repeated}$",
            audit,
        )

        self.assertEqual(recovered, primary)
        self.assertEqual(action, "")
        self.assertEqual(audit["selection_reason"], "pathological_structured_output_rejected")

    def test_structured_candidate_cannot_drop_a_trailing_option(self) -> None:
        primary = (
            "4. 已知函数满足所给条件，请选择正确结论。\n"
            "A. 结论一 B. 结论二 C. 结论三 D. 结论四"
        )
        candidate = (
            "4. 已知函数 $f(x)=x^2$ 满足所给条件，请选择正确结论。\n"
            "A. 结论一 B. 结论二 C. 结论三"
        )
        audit = {"formula_count": 1, "formulas": ["f(x)=x^2"], "selected": False}

        recovered, action = test5.fuse_structured_ocr_transcript(
            primary, candidate, audit
        )

        self.assertEqual(recovered, primary)
        self.assertEqual(action, "")

    def test_structured_candidate_cannot_drop_recognized_subquestions(self) -> None:
        primary = (
            "27. 如图，已知二次函数满足条件。\n"
            "(1) 填空：b=____，c=____。\n"
            "(2) 一次函数与二次函数交于D、E两点。\n"
            "①求参数范围；\n"
            "②探究是否存在实数n满足条件。"
        )
        candidate = (
            "27. 如图，二次函数 $y=x^2+bx+c$ 的图像与坐标轴相交，"
            "且满足 $OB=OC=3$。"
        )
        audit = {
            "formula_count": 2,
            "formulas": ["y=x^2+bx+c", "OB=OC=3"],
            "selected": False,
        }

        recovered, action = test5.fuse_structured_ocr_transcript(
            primary, candidate, audit
        )

        self.assertEqual(recovered, primary)
        self.assertEqual(action, "")
        self.assertEqual(
            audit["selection_reason"],
            "subquestion_structure_regression_rejected",
        )

    def test_complete_primary_is_not_replaced_by_equally_complete_noisy_candidate(self) -> None:
        primary = (
            "25. 如图，△ABC中，D、E分别为边AB、AC上一点，且DC=DE。\n"
            "(1)求证：∠A=∠DCF；\n"
            "(2)若∠ACB=90°，①求弦DF长度；②求弦DF长度的最小值。"
        )
        candidate = (
            "25. 图，△ABC E分别为边AB，AC上一点，且DC=DE。\n"
            "(1)求证：△EDC；\n"
            "(2)若ACB=90，①求DF；②求最小值。"
        )
        audit = {"formula_count": 0, "formulas": [], "selected": False}

        recovered, action = test5.fuse_structured_ocr_transcript(
            primary, candidate, audit
        )

        self.assertEqual(recovered, primary)
        self.assertEqual(action, "")

    def test_structured_candidate_is_trimmed_to_the_primary_question(self) -> None:
        primary = (
            "12. 当前题干，请选择正确结论。\n"
            "A. 结论一 B. 结论二 C. 结论三"
        )
        candidate = (
            "D. 上一题末项\n12. 当前题干，请选择正确结论。\n"
            "A. 结论一 B. 结论二 C. 结论三 D. 结论四\n"
            "13. 下一题干"
        )
        audit = {"formula_count": 0, "formulas": [], "selected": False}

        recovered, action = test5.fuse_structured_ocr_transcript(
            primary, candidate, audit
        )

        self.assertEqual(test5.transcript_semantic_completeness(recovered)["option_labels"], list("ABCD"))
        self.assertNotIn("13.", recovered)
        self.assertEqual(action, "select_local_structured_ocr")

    def test_balanced_gate_skips_clear_high_confidence_prose(self) -> None:
        transcript = (
            "2. 下列说法正确的是。\n"
            "A. 甲说法 B. 乙说法 C. 丙说法 D. 丁说法"
        )

        reasons = test5.structured_ocr_recovery_reasons(
            transcript, [item("下列说法正确的是", 200, 100, 260)]
        )

        self.assertEqual(reasons, [])

    def test_balanced_gate_keeps_formula_and_structure_risks(self) -> None:
        formula_reasons = test5.structured_ocr_recovery_reasons(
            "3. 已知函数f(x)=x²，求最值。",
            [item("已知函数f(x)=x²", 200, 100, 260)],
        )
        broken_reasons = test5.structured_ocr_recovery_reasons(
            "A. 1 B. 2 C. 3 D. 4",
            [item("A. 1 B. 2 C. 3 D. 4", 200, 100, 300)],
        )

        self.assertIn("formula_structure_signal", formula_reasons)
        self.assertIn("semantic_structure_gap", broken_reasons)

    def test_formula_gap_uses_formula_only_not_full_structured_route(self) -> None:
        transcript = (
            "18. 已知椭圆C: +y²=1(a>1)的离心率为 "
            "P(x0,y0)(x0>1)为椭圆C上的动点。"
        )
        reasons = test5.structured_ocr_recovery_reasons(
            transcript,
            [item("椭圆C:+y²=1", 200, 100, 260)],
        )

        self.assertIn("formula_structure_signal", reasons)
        self.assertNotIn("semantic_structure_gap", reasons)
        self.assertTrue(test5.formula_only_recovery_is_sufficient(reasons))

    def test_tiled_retry_is_reserved_for_missing_prose(self) -> None:
        self.assertFalse(test5.needs_tiled_semantic_ocr_recovery(
            {"option_sequence_incomplete": True}
        ))
        self.assertFalse(test5.needs_tiled_semantic_ocr_recovery(
            {"choice_count_ambiguous": True}
        ))
        self.assertTrue(test5.needs_tiled_semantic_ocr_recovery(
            {"stem_missing": True}
        ))
        self.assertFalse(test5.needs_tiled_semantic_ocr_recovery(
            {"stem_missing": True, "section_header_only": True}
        ))


if __name__ == "__main__":
    unittest.main()
