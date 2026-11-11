from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "math" / "source" / "five_tier_guidance.py"
SPEC = importlib.util.spec_from_file_location("five_tier_guidance_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
five_tier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(five_tier)


def sample_output() -> dict:
    def part(step_count: int, marker: str, *, pitfall: str = "") -> dict:
        value = {
            "part": "(1)",
            "method_id": "M1",
            "method_name": "一元二次方程判别式法",
            "steps": [
                "先把条件整理为关于参数的方程",
                "再建立判别式与题目限制的关系",
                "按限制范围继续推导",
                "回到题问核对结论",
            ][:step_count],
            "alternative_methods": [],
        }
        if pitfall:
            value["pitfall"] = pitfall
        value["steps"][0] += marker
        return value

    return {
        "recognition_status": "clear",
        "uncertain_items": [],
        "knowledge_points": ["一元二次方程", "判别式与根的情况"],
        "base_formulas": [{"formula": "Delta=b^2-4ac", "condition": "一元二次方程"}],
        "guidance": {
            "advanced": {"title": "学霸版", "parts": [part(3, "ADV")]},
            "standard": {
                "title": "普通版",
                "parts": [part(4, "STD", pitfall="不要漏掉参数限制")],
            },
            "foundation": {
                "title": "基础版",
                "parts": [part(4, "FOUND")],
                "pattern_summary": "整理条件、建立关系、核对限制。",
            },
        },
    }


class FiveTierProjectionTests(unittest.TestCase):
    def test_math_and_physics_compact_prompts_define_all_five_boundaries(self) -> None:
        prompt_paths = (
            ROOT / "math" / "source" / "数学题解析紧凑提示词.md",
            ROOT / "physics" / "source" / "物理题解析紧凑提示词.md",
        )
        for prompt_path in prompt_paths:
            prompt = prompt_path.read_text(encoding="utf-8")
            with self.subTest(prompt=prompt_path.name):
                self.assertIn("亲自生成 L1～L5 五种程度明显不同", prompt)
                self.assertIn("不会再从一份公共骨架裁剪五层", prompt)
                self.assertIn("`l1` 学霸解析", prompt)
                self.assertIn("`l2` 中上解析", prompt)
                self.assertIn("`l3` 中等解析", prompt)
                self.assertIn("`l4` 中下解析", prompt)
                self.assertIn("`l5` 学渣解析", prompt)
                self.assertIn("本地检查后直接渲染", prompt)
                self.assertIn("渲染前删除明确终值", prompt)

    def test_contract_order_is_fixed(self) -> None:
        self.assertEqual(
            five_tier.DISPLAY_TIER_KEYS,
            ("elite", "upper", "middle", "lower", "novice"),
        )
        self.assertEqual(
            five_tier.DISPLAY_TIER_TITLES,
            {
                "elite": "学霸解析",
                "upper": "中上解析",
                "middle": "中等解析",
                "lower": "中下解析",
                "novice": "学渣解析",
            },
        )
        self.assertTrue(
            all(not title for title in five_tier.DISPLAY_TIER_SECTION_TITLES.values())
        )

    def test_each_level_obeys_its_disclosure_boundary(self) -> None:
        output = sample_output()
        pages = {
            key: five_tier.project_display_tier(key, output)
            for key in five_tier.DISPLAY_TIER_KEYS
        }

        elite = pages["elite"]
        elite_steps = elite["parts"][0]["steps"]
        self.assertFalse(elite["knowledge_points"])
        self.assertFalse(elite["base_formulas"])
        self.assertLessEqual(len(elite_steps), 3)
        self.assertTrue(all(step.endswith("？") for step in elite_steps))
        self.assertNotIn("判别式", "".join(elite_steps))
        self.assertNotIn("ADV", "".join(elite_steps))

        upper = pages["upper"]
        self.assertEqual(upper["knowledge_points"], output["knowledge_points"])
        self.assertFalse(upper["base_formulas"])
        self.assertFalse(upper["parts"])

        middle = pages["middle"]
        self.assertFalse(middle["knowledge_points"])
        self.assertEqual(len(middle["parts"][0]["steps"]), 1)
        self.assertIn("STD", middle["parts"][0]["steps"][0])
        self.assertNotIn("核对结论", middle["parts"][0]["steps"][0])

        lower = pages["lower"]
        self.assertFalse(lower["knowledge_points"])
        self.assertEqual(
            lower["parts"][0]["steps"][:-1],
            output["guidance"]["advanced"]["parts"][0]["steps"],
        )
        self.assertIn("________", lower["parts"][0]["steps"][-1])
        self.assertEqual(lower["parts"][0]["method_name"], "一元二次方程判别式法")

        novice = pages["novice"]
        self.assertEqual(novice["base_formulas"], output["base_formulas"])
        self.assertEqual(
            novice["parts"][0]["steps"][:-1],
            output["guidance"]["foundation"]["parts"][0]["steps"],
        )
        self.assertIn("________", novice["parts"][0]["steps"][-1])
        self.assertEqual(novice["parts"][0]["pitfall"], "不要漏掉参数限制")
        self.assertTrue(novice["pattern_summary"])

    def test_projection_does_not_mutate_validated_model_output(self) -> None:
        output = sample_output()
        original_first_step = output["guidance"]["standard"]["parts"][0]["steps"][0]
        page = five_tier.project_display_tier("middle", output)
        page["parts"][0]["steps"][0] = "changed"
        self.assertEqual(
            output["guidance"]["standard"]["parts"][0]["steps"][0],
            original_first_step,
        )


if __name__ == "__main__":
    unittest.main()
