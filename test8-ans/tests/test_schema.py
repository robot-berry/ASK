from __future__ import annotations

import copy
import unittest

from test8_ans.schema import parse_json_content, validate_guidance


def valid_answer() -> dict:
    return {
        "l1": [{"n": "（1）", "s": ["题目中的关键等量关系藏在哪里？"]}],
        "l2": ["一元一次方程", "等式基本性质"],
        "l3": [{"n": "（1）", "s": ["括号两侧的倍数关系给出了化简入口。"]}],
        "l4": [
            {
                "n": "（1）",
                "s": [
                    "由等式两边同除相同非零数可先去掉倍数。",
                    "再根据等式性质建立未知数与常数的关系。",
                ],
            }
        ],
        "l5": [
            {
                "n": "（1）",
                "s": [
                    "先把括号整体看成一个暂时未知的量。",
                    "等式两边同时除以相同非零数仍然相等。",
                    "去掉外层倍数后再整理未知数的关系。",
                    "最后代回原等式检查左右两边是否相等。",
                ],
                "pit": "不要只除括号内的一项。",
            }
        ],
        "f": [{"x": "ax=b", "c": "a\\ne 0 时的一元一次方程"}],
        "pat": "先整体化简，再利用等式性质逐步孤立未知数。",
        "s": "c",
        "u": [],
    }


class SchemaTests(unittest.TestCase):
    def test_valid_math_answer(self) -> None:
        report = validate_guidance(valid_answer(), "math")
        self.assertTrue(report["valid"], report["errors"])

    def test_accepts_json_code_fence_for_recovery(self) -> None:
        parsed = parse_json_content(
            '```json\n{"a": 1}\n```', allow_code_fence=True
        )
        self.assertEqual(parsed, {"a": 1})

    def test_strict_parse_rejects_code_fence(self) -> None:
        with self.assertRaisesRegex(ValueError, "代码围栏"):
            parse_json_content('```json\n{"a": 1}\n```')

    def test_lossless_transport_normalization_collects_repeated_s(self) -> None:
        audit: list[dict] = []
        parsed = parse_json_content(
            '{"l4":[{"n":"1","s":"第一句分析内容。","s":"第二句分析内容。"}]}',
            normalize_sentence_transport=True,
            normalization_audit=audit,
        )
        self.assertEqual(
            parsed["l4"][0]["s"], ["第一句分析内容。", "第二句分析内容。"]
        )
        self.assertEqual(audit[0]["action"], "collect_duplicate_s_keys")

    def test_transport_normalization_rejects_other_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "无法安全恢复的重复键"):
            parse_json_content(
                '{"l4":[{"n":"1","n":"2","s":["第一句", "第二句"]}]}',
                normalize_sentence_transport=True,
            )

    def test_rejects_misaligned_question_numbers(self) -> None:
        answer = copy.deepcopy(valid_answer())
        answer["l4"][0]["n"] = "（2）"
        report = validate_guidance(answer, "math")
        self.assertFalse(report["valid"])
        self.assertTrue(any("与 l1 一致" in error for error in report["errors"]))

    def test_physics_allows_empty_formulas_but_math_does_not(self) -> None:
        answer = valid_answer()
        answer["f"] = []
        self.assertTrue(validate_guidance(answer, "physics")["valid"])
        self.assertFalse(validate_guidance(answer, "math")["valid"])


if __name__ == "__main__":
    unittest.main()
