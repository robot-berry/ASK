from __future__ import annotations

import unittest

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
            "填空：该段长度为 12.5（结果保留一位小数）"
        )

        self.assertIn("该段长度为____（结果保留一位小数）", cleaned)
        self.assertNotIn("12.5", cleaned)

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
            "二、选择题：本题共5小题，每小题4分"
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

    def test_missing_left_parenthesis_on_first_subpart_is_restored(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "14. 完整题干。\n1）测量电压；\n(2)计算电阻。"
        )

        self.assertIn("(1)测量电压", cleaned)

    def test_handwritten_tail_after_friction_option_is_removed(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "B. 手捏得越松，摩擦力越小人"
        )

        self.assertEqual(cleaned, "B. 手捏得越松，摩擦力越小")

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
            [item("16. 如图，分析电路中电流变化。", 260, 80, 460)],
            "16. 如图，分析电路中电流变化。",
            6,
        )
        text_only, _ = test5.structured_layout_illustration_boxes(
            audit,
            (600, 900, 3),
            [item("16. 计算物体运动速度。", 220, 80, 360)],
            "16. 计算物体运动速度。",
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
                item("16. 如图，分析电路中电流变化。", 300, 470, 520),
            ],
            "16. 如图，分析电路中电流变化。",
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

    def test_previous_option_is_excluded_when_two_question_markers_exist(self) -> None:
        records = [
            item("D. 上一道题的最后一个选项", 280, 20, 430),
            item("12. 当前题的完整题干", 300, 60, 480),
            item("A. 甲 B. 乙 C. 丙 D. 丁", 380, 180, 620),
            item("13. 下一道题的题干", 300, 410, 480),
        ]

        box = test5.select_middle_question_box((600, 900, 3), records)

        self.assertGreater(box[1], 20)
        self.assertLess(box[1], 60)
        self.assertEqual(box[3], 398)

    def test_single_field_symbol_does_not_fail_non_choice_problem(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "17. 求粒子通过速度选择器后的速度。\nB.方向垂直纸面向外。"
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

    def test_variable_subscripts_are_not_repeated_question_markers(self) -> None:
        audit = test5.transcript_semantic_completeness(
            "13. 三种离子质量分别为m1、m2、m3，速度分别为v1、v2、v3。"
        )

        self.assertFalse(audit["repeated_question_number"])

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

    def test_page_account_noise_is_removed(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "17. 这是需要保留的题干。\n微信公众号：测试账号 QQ:2754808740"
        )

        self.assertIn("需要保留的题干", cleaned)
        self.assertNotIn("QQ", cleaned)
        self.assertNotIn("公众号", cleaned)

    def test_flattened_scientific_exponent_is_restored_locally(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "电子质量m=9.1×10-31kg，普朗克常量h=6.63×10−34J·s。"
        )

        self.assertIn("×10^{-31}kg", cleaned)
        self.assertIn("×10^{-34}J", cleaned)

    def test_trailing_handwritten_solution_is_removed(self) -> None:
        cleaned = test5.repair_ocr_transcript(
            "13. 如图所示，求下列两问。\n"
            "(1) 求金属圆筒长度。\n"
            "(2) 求电场强度大小。\n"
            "由能量守恒得\n"
            "$qU=mv^2/2$"
        )

        self.assertTrue(cleaned.endswith("(2) 求电场强度大小。"))
        self.assertNotIn("能量守恒", cleaned)
        self.assertNotIn("qU", cleaned)

    def test_structured_formula_ocr_restores_scientific_exponent(self) -> None:
        audit = {
            "formula_count": 1,
            "formulas": [r"6.63\times10^{-34}\,\mathrm{J\cdot s}"],
            "selected": False,
        }
        recovered, action = test5.fuse_structured_ocr_transcript(
            "已知普朗克常量h=6.63×10-34J·s，求物质波波长。",
            "已知普朗克常量 $h=6.63\\times10^{-34}\\,\\mathrm{J\\cdot s}$，求物质波波长。",
            audit,
        )

        self.assertIn(r"10^{-34}", recovered)
        self.assertEqual(action, "select_local_structured_formula_ocr")
        self.assertTrue(audit["selected"])

    def test_pathological_formula_output_cannot_replace_primary_ocr(self) -> None:
        primary = "7. 已知中子衰变释放能量，判断下列说法。\nA. 甲 B. 乙 C. 丙 D. 丁"
        repeated = " ".join("1" for _ in range(80))
        audit = {"formula_count": 1, "formulas": [repeated], "selected": False}

        recovered, action = test5.fuse_structured_ocr_transcript(
            primary,
            f"7. 已知中子衰变释放能量。 ${repeated}$",
            audit,
        )

        self.assertEqual(recovered, primary)
        self.assertEqual(action, "")
        self.assertEqual(audit["selection_reason"], "pathological_structured_output_rejected")

    def test_structured_candidate_cannot_drop_a_trailing_option(self) -> None:
        primary = (
            "4. 已知物体完成某运动过程，请选择正确结论。\n"
            "A. 结论一 B. 结论二 C. 结论三 D. 结论四"
        )
        candidate = (
            "4. 已知速度 $v=10m/s$，请选择正确结论。\n"
            "A. 结论一 B. 结论二 C. 结论三"
        )
        audit = {"formula_count": 1, "formulas": ["v=10m/s"], "selected": False}

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
            "3. 已知速度v=10m/s，求位移。",
            [item("已知速度v=10m/s", 200, 100, 260)],
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

    def test_damaged_figure_cue_still_preserves_visual_evidence(self) -> None:
        self.assertTrue(
            test5.transcript_requires_illustration(
                "18. 图，闭合电路中电流表与滑动变阻器连接如下。"
            )
        )

    def test_complete_visual_question_skips_formula_only_refinement(self) -> None:
        transcript = (
            "18. 如图，闭合电路中电源电压恒定，电流表示数为0.5A。\n"
            "(1)求电阻两端电压；\n(2)求电路消耗的总功率。"
        )

        self.assertTrue(
            test5.can_skip_structured_ocr_for_preserved_visual(
                transcript,
                ["formula_structure_signal"],
                [(400, 220, 820, 510)],
            )
        )
        self.assertFalse(
            test5.can_skip_structured_ocr_for_preserved_visual(
                transcript,
                ["semantic_structure_gap", "formula_structure_signal"],
                [(400, 220, 820, 510)],
            )
        )
        flattened = (
            "18. 闭合电路中电源电压恒定，电流表示数为I1=0.5A。"
            "(1)求电阻两端电压。"
        )
        self.assertFalse(
            test5.can_skip_structured_ocr_for_preserved_visual(
                flattened,
                ["formula_structure_signal", "flattened_script_signal"],
                [(400, 220, 820, 510)],
            )
        )

    def test_formula_only_refinement_rejects_unanchored_handwriting(self) -> None:
        source = [item("v=10m/s", 120, 90, 120)]
        audit = {
            "formula_items": [
                {
                    "bbox": [55, 75, 185, 105],
                    "latex": r"v=10\,\mathrm{m/s}",
                    "score": 0.96,
                },
                {
                    "bbox": [430, 300, 540, 345],
                    "latex": r"W=Fs",
                    "score": 0.99,
                },
            ]
        }

        refined, replaced = test5.fuse_formula_only_ocr_items(source, audit)

        self.assertEqual(replaced, 1)
        self.assertTrue(any(r"\mathrm{m/s}" in entry.text for entry in refined))
        self.assertFalse(any("W=Fs" in entry.text for entry in refined))

    def test_formula_only_mode_never_appends_public_validation_section(self) -> None:
        primary = "3. 已知速度v=10m/s，求位移。"
        audit = {
            "request_mode": "formula_only",
            "formula_count": 1,
            "formulas": [r"v=10\,\mathrm{m/s}"],
            "selected": False,
        }

        recovered, action = test5.fuse_structured_ocr_transcript(
            primary,
            r"$v=10\,\mathrm{m/s}$",
            audit,
        )

        self.assertEqual(recovered, primary)
        self.assertEqual(action, "")
        self.assertNotIn("原图公式结构校验", recovered)

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
