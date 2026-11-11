from __future__ import annotations

import unittest

from test8_ans.relevance import validate_relevance
from tests.test_schema import valid_answer


class RelevanceTests(unittest.TestCase):
    def test_equation_answer_passes_local_anchor_screen(self) -> None:
        report = validate_relevance("解方程：2(x-1)=6。", valid_answer())
        self.assertTrue(report["valid"], report["errors"])

    def test_unrelated_generic_answer_fails(self) -> None:
        answer = valid_answer()
        for tier in ("l1", "l3", "l4"):
            for part in answer[tier]:
                part["s"] = ["分析题意并整理条件以后逐步求解。"] * len(part["s"])
        answer["l5"][0]["s"] = ["分析题意并整理条件以后逐步求解。"] * 4
        answer["l2"] = ["基础知识", "综合应用"]
        answer["f"] = [{"x": "a+b", "c": "一般情形"}]
        answer["pat"] = "整理条件并执行运算。"
        report = validate_relevance("一辆汽车匀速行驶，求速度。", answer)
        self.assertFalse(report["valid"])


if __name__ == "__main__":
    unittest.main()
