import unittest

from semantic_ocr_recovery import (
    choice_labels,
    extract_complete_choice_tail,
    extract_numbered_question,
    full_page_recovery_candidate,
    repair_missing_choice_labels,
    trim_after_complete_choice_question,
)


class SemanticOcrRecoveryTests(unittest.TestCase):
    def test_extracts_only_target_question(self):
        source = "D. 上一题末项\n12. 当前题干\nA. 一 B. 二 C. 三 D. 四\n13. 下一题干"
        self.assertEqual(
            extract_numbered_question(source, 12),
            "12. 当前题干\nA. 一 B. 二 C. 三 D. 四",
        )

    def test_extracts_roman_suffix_question_and_stops_before_next(self):
        source = (
            "14-III.（2分）（多选）以下实验中，说法正确的是\n"
            "A. 甲\nB. 乙\nC. 丙\nD. 丁\n"
            "15.（8分）下一题"
        )

        extracted = extract_numbered_question(source, 14)

        self.assertTrue(extracted.startswith("14-III."))
        self.assertNotIn("15.", extracted)

    def test_extracts_roman_suffix_followed_by_scored_subquestion(self):
        primary = (
            "非选择题部分\n14-I\n(1)(2分)灯泡实验\n"
            "A. 15W\nB. 60W\nC. 100W\nD. 1000W\n"
            "(2)(5分)测量金属丝的电阻率。"
        )
        full = (
            "14-I（1）（2分）灯泡实验\n"
            "A. 15W\nB. 60W\nC. 100W\nD. 1000W\n"
            "(2）（5分）测量金属丝的电阻率。\n"
            "①先测量长度L，再测量直径。\n②选择下列器材。"
        )

        candidate, mode = full_page_recovery_candidate(primary, full)

        self.assertEqual(mode, "target_question_block")
        self.assertIn("①先测量长度L", candidate)

    def test_duplicate_target_number_prefers_the_complete_block(self):
        source = (
            "7. 不完整首段\nA. 甲\n"
            "7. 完整题干\nA. 甲\nB. 乙\nC. 丙\nD. 丁\n"
            "8. 下一题"
        )

        extracted = extract_numbered_question(source, 7)

        self.assertIn("完整题干", extracted)
        self.assertEqual(choice_labels(extracted), ["A", "B", "C", "D"])
        self.assertNotIn("8.", extracted)

    def test_repairs_two_column_formula_labels_from_reading_order(self):
        source = "$a$ B. $b$\n$c$ D. $d$"
        repaired = repair_missing_choice_labels(source)
        self.assertIn("A. $a$ B. $b$", repaired)
        self.assertIn("C. $c$ D. $d$", repaired)

    def test_recovers_formula_only_a_before_complete_bcd_tail(self):
        source = "题干公式\n$v_0=9$\nB. 第二项\nC. 第三项\nD. 第四项\n页脚"
        self.assertEqual(
            extract_complete_choice_tail(source),
            "A. $v_0=9$\nB. 第二项\nC. 第三项\nD. 第四项",
        )

    def test_repairs_missing_punctuation_on_observable_bcd_tail(self):
        source = "$v_0=9$\nB第二项\nC. 第三项\nD. 第四项"
        self.assertEqual(
            extract_complete_choice_tail(source),
            "A. $v_0=9$\nB. 第二项\nC. 第三项\nD. 第四项",
        )

    def test_repairs_missing_punctuation_before_formula_option(self):
        source = "A. $a$ B. $b$\nC $(-\\infty,1)$ D. $(2,+\\infty)$"
        repaired = repair_missing_choice_labels(source)

        self.assertIn("C. $(-\\infty,1)$", repaired)
        self.assertEqual(
            extract_complete_choice_tail(repaired),
            "A. $a$ B. $b$\nC. $(-\\infty,1)$ D. $(2,+\\infty)$",
        )

    def test_moves_formula_wrapped_option_labels_outside_math_span(self):
        source = (
            "1. 下列说法正确的是\n"
            "$A _ {2}^{4}He$ 是一种核素\n"
            "$B _ {6}^{14}C$ 来自核外电子\n"
            "C环境会改变半衰期\nD. 距今约2100年"
        )

        repaired = repair_missing_choice_labels(source)

        self.assertEqual(choice_labels(repaired), ["A", "B", "C", "D"])
        self.assertIn("A. $_ {2}^{4}He$", repaired)
        self.assertIn("C. 环境会改变半衰期", repaired)

    def test_restores_two_unlabelled_options_between_choice_cue_and_cd(self):
        source = (
            "10. 已知函数关系如图，则下列说法正确的是\n"
            "前2小时平均速度为70千米/小时\n"
            "$0\\leq t\\leq2$时，$y=-70t+320$\n"
            "C. 共用了4小时\nD. 距离为135千米"
        )

        self.assertEqual(
            choice_labels(repair_missing_choice_labels(source)),
            ["A", "B", "C", "D"],
        )

    def test_restores_unlabelled_b_and_unpunctuated_d(self):
        source = (
            "1. 关于摩擦力的说法（ ）\nA. 第一项\n第二项\n"
            "C. 第三项\nD第四项"
        )

        repaired = repair_missing_choice_labels(source)

        self.assertEqual(choice_labels(repaired), ["A", "B", "C", "D"])
        self.assertIn("B. 第二项", repaired)
        self.assertIn("D. 第四项", repaired)

    def test_restores_wrapped_unlabelled_c_between_b_and_d(self):
        source = (
            "8. 下列说法正确的是\nA. 第一项\nB. 第二项\n"
            "若离子源发出一个粒子，将打在Q点右侧\n$2d$ 的位置\nD. 第四项"
        )

        repaired = repair_missing_choice_labels(source)

        self.assertEqual(choice_labels(repaired), ["A", "B", "C", "D"])
        self.assertIn("C. 若离子源", repaired)

    def test_restores_unlabelled_a_immediately_before_bcd_tail(self):
        source = (
            "8. 两极板距离增大，则\n电容器的电容减小\n"
            "B. 场强变小\nC. 电荷量变大\nD. 电流反向"
        )

        repaired = repair_missing_choice_labels(source)

        self.assertEqual(choice_labels(repaired), ["A", "B", "C", "D"])
        self.assertIn("A. 电容器的电容减小", repaired)

    def test_matches_target_block_when_primary_question_number_is_missing(self):
        primary = "下列说法正确的是\nB. 数据的上四分位数是8\nC. $R^2=1$\nD. 残差为3cm"
        full = (
            "9. 下列说法正确的是\nA. 独立性检验不适用\n"
            "B. 数据的上四分位数是8\nC. $R^2=1$\nD. 残差为3cm\n"
            "10. 已知三个向量，则下列说法正确的是"
        )

        candidate, mode = full_page_recovery_candidate(primary, full)

        self.assertEqual(mode, "matched_question_block")
        self.assertTrue(candidate.startswith("9."))
        self.assertNotIn("10.", candidate)

    def test_uses_single_full_question_when_primary_contains_only_choices(self):
        primary = "A. (−2,0) B. (−1,+∞) C. (−1,0) D. (−∞,−2)"
        full = (
            "1. 设集合A={x|x²+2x<0}，集合B={x|2ˣ>1/2}，则A∩B=\n"
            "A. (−2,0) B. (−1,+∞) C. (−1,0) D. (−∞,−2)"
        )

        candidate, mode = full_page_recovery_candidate(primary, full)

        self.assertEqual(mode, "single_question_block")
        self.assertTrue(candidate.startswith("1."))
        self.assertEqual(candidate.count("A."), 1)

    def test_complete_primary_choice_tail_is_not_appended_twice(self):
        primary = "A. 1 B. 2 C. 3 D. 4"

        self.assertEqual(
            full_page_recovery_candidate(primary, "A. 1\nB. 2\nC. 3\nD. 4"),
            ("", ""),
        )

    def test_trims_partial_neighbour_after_completed_option_d(self):
        source = (
            "8. 如图，选择正确说法\n"
            "A. 甲\nB. 乙\nC. 丙\nD. 丁。\n"
            "游戏中两位同学分别握住轻绳一端\n占AB终在同一水平面"
        )

        repaired = trim_after_complete_choice_question(source)

        self.assertTrue(repaired.endswith("D. 丁。"))
        self.assertNotIn("游戏中", repaired)

    def test_visible_next_question_trims_even_without_period_after_d(self):
        source = (
            "14-III.（2分）（多选）以下实验中，说法正确的是\n"
            "A. 甲\nB. 乙\nC. 丙\nD. 测量值偏小\n"
            "15.（8分）如图所示是下一题"
        )

        repaired = trim_after_complete_choice_question(source)

        self.assertTrue(repaired.endswith("D. 测量值偏小"))
        self.assertNotIn("15.", repaired)

    def test_keeps_later_subquestion_after_complete_choice_block(self):
        source = (
            "14-I（1）（2分）灯泡规格可能是\n"
            "A. 15W\nB. 60W\nC. 100W\nD. 1000W\n"
            "(2）（5分）测量金属丝的电阻率。\n"
            "①先测量长度L，再测量直径。\n②选择下列器材。"
        )

        repaired = trim_after_complete_choice_question(source)

        self.assertIn("①先测量长度L", repaired)
        self.assertIn("②选择下列器材", repaired)

    def test_primary_is_kept_when_full_page_has_no_observable_target(self):
        self.assertEqual(full_page_recovery_candidate("6. 题干", "无编号杂讯"), ("", ""))

    def test_empty_set_zero_is_repaired_only_in_set_choice_context(self):
        source = "1. 已知集合A，全集U\nA. 0 B. {-2} C. {2} D. {-2,2}"
        self.assertIn("A. ∅", repair_missing_choice_labels(source))
        self.assertEqual(repair_missing_choice_labels("A. 0 B. 1 C. 2 D. 3"), "A. 0 B. 1 C. 2 D. 3")


if __name__ == "__main__":
    unittest.main()
