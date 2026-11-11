from __future__ import annotations

import copy
import json
import unittest

from test8_ans.answer_guard import (
    derive_answer_targets,
    sanitize_display_answer,
)
from tests.test_schema import valid_answer


CHOICE_QUESTION = (
    r"设集合 $A=\{x\mid x^2+2x<0\}$，集合 $B=\{x\mid 2^x>\frac12\}$，"
    r"则 $A \bigcap B =$ A. $(-2,0)$ B. $(-1,+\infty)$ "
    r"C. $(-1,0)$ D. $(-\infty,-2)$"
)


class AnswerGuardTests(unittest.TestCase):
    def test_derives_requested_intersection_without_using_an_answer_key(self) -> None:
        targets = derive_answer_targets(CHOICE_QUESTION)
        self.assertIn("A∩B", targets.symbols)
        self.assertIn("交集", targets.prose)
        self.assertTrue(targets.choice_question)

    def test_removes_final_interval_and_option_but_keeps_intermediate_ranges(self) -> None:
        answer = valid_answer()
        answer["l4"][0]["s"] = [
            "由不等式可知集合A为(-2,0)。",
            "根据指数关系可得集合B为(-1,+∞)。",
        ]
        answer["l5"][0]["s"] = [
            "先解第一个不等式，得到集合A是(-2,0)。",
            "再解指数不等式，得到集合B是(-1,+∞)。",
            "最后求两个区间的交集，即$A\\cap B=(-1,0)$。",
            "确认答案为C选项。",
        ]
        original = copy.deepcopy(answer)

        display, report = sanitize_display_answer(
            answer, question=CHOICE_QUESTION, subject="math"
        )

        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["redaction_count"], 2)
        self.assertEqual(answer, original, "原始解析视图不得被本地脱敏改写")
        display_text = json.dumps(display, ensure_ascii=False)
        self.assertIn("集合A为(-2,0)", display_text)
        self.assertIn("集合B为(-1,+∞)", display_text)
        self.assertNotIn("A\\cap B=(-1,0)", display_text)
        self.assertNotIn("C选项", display_text)
        self.assertEqual(len(display["l5"][0]["s"]), 2)
        self.assertNotIn("最后求两个区间的交集", display_text)
        report_text = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("A\\cap B=(-1,0)", report_text)
        self.assertNotIn("C选项", report_text)

    def test_removes_requested_physics_direction(self) -> None:
        answer = valid_answer()
        answer["l5"][0]["s"][2] = "由合力关系可知，物体运动方向向右。"
        display, report = sanitize_display_answer(
            answer,
            question="物体释放后的运动方向如何？",
            subject="physics",
        )
        self.assertTrue(report["valid"])
        self.assertNotIn(
            "向右", json.dumps(display, ensure_ascii=False)
        )

    def test_plain_requested_symbol_and_proof_are_resolved(self) -> None:
        self.assertIn("F", derive_answer_targets("求拉力F的大小。").symbols)
        self.assertTrue(derive_answer_targets("证明两直线平行。").proof_question)

    def test_fails_closed_when_requested_output_cannot_be_located(self) -> None:
        _, report = sanitize_display_answer(
            valid_answer(), question="请阅读题目并分析。", subject="math"
        )
        self.assertFalse(report["valid"])
        self.assertTrue(report["errors"])


if __name__ == "__main__":
    unittest.main()
