import hashlib
import json
import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
import test10_prompts as library


class Test10Integration(unittest.TestCase):
    def test_grade_only_tree_is_selected(self):
        self.assertEqual(library.ROOT.resolve(), (PROJECT / "prompt_project" / "test10").resolve())
        root = library.ROOT / "提示词" / "数学"
        self.assertEqual(sorted(p.name for p in root.iterdir()), ["初中", "高中"])
        for group, grades in (("初中", ("七年级", "八年级", "九年级")), ("高中", ("高一", "高二", "高三"))):
            self.assertEqual(sorted(p.name for p in (root / group).iterdir()), sorted(grades))

    def test_unified_prompt_is_sibling_of_seal_prompt(self):
        prompts = PROJECT / "prompt_project" / "prompts"
        math_prompt = prompts / "统一数学提示词.md"
        seal_prompt = prompts / "seal_generation_v1.md"
        self.assertTrue(math_prompt.is_file())
        self.assertTrue(seal_prompt.is_file())
        manifest = json.loads(
            (PROJECT / "prompt_project" / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            manifest["math_guidance"]["template"],
            "prompts/统一数学提示词.md",
        )
        self.assertEqual(
            math_prompt.read_bytes(),
            Path(library._unified_prompt_location()[1]).read_bytes(),
        )

    def test_topic_files_have_method_cards_and_hashes(self):
        root = library.ROOT / "提示词" / "数学"
        for group in ("初中", "高中"):
            files = [p for p in (root / group).rglob("*.md") if p.parent.name in ("上学期", "下学期")]
            self.assertGreater(len(files), 0)
            for path in files:
                text = path.read_text(encoding="utf-8-sig")
                self.assertTrue("## 识别特征" in text or "## 一、识别特征" in text)
                self.assertTrue("## 方法索引" in text or "## 二、方法索引" in text)
                self.assertIn("L1", text)
                self.assertTrue(any(marker in text for marker in ("### M01", "## M01", "方法 A", "方法卡")))

    def test_junior_terms_load_only_grade_topic_sources(self):
        for grade in range(7, 10):
            for semester in (1, 2):
                term = f"g{grade}_s{semester}"
                context = library.build_context("math", {"id": term}, "解一元一次方程")
                self.assertEqual(context["status"], "loaded")
                self.assertTrue(context["sources"])
                self.assertEqual(context["prompt_file"], context["sources"][0]["path"])
                self.assertEqual(Path(context["prompt_file"]).name, "统一数学提示词.md")
                self.assertEqual(
                    [source["path"] for source in context["sources"]],
                    [context["prompt_file"]],
                )
                self.assertTrue(context["selected_topics"])
                self.assertTrue(all(item["group"] == "初中" for item in context["selected_topics"]))
                self.assertIn("严格遵守较窄范围", context["text"])
                for source in context["sources"]:
                    content = (library.ROOT / source["path"]).read_text(encoding="utf-8-sig")
                    self.assertEqual(source["sha256"], hashlib.sha256(content.encode()).hexdigest())

    def test_high_school_can_use_junior_topic_guidance(self):
        context = library.build_context("math", {"id": "g12_s2"}, "解一元一次方程")
        groups = [item["group"] for item in context["selected_topics"]]
        self.assertIn("初中", groups)
        self.assertEqual(Path(context["prompt_file"]).name, "统一数学提示词.md")

    def test_unified_prompt_embeds_all_topic_sections(self):
        library_data = library._unified_library()
        self.assertIsNotNone(library_data)
        self.assertEqual(len(library_data["records"]), 77)
        self.assertEqual(len(library_data["text"].split("TEST10_TOPIC_BEGIN")) - 1, 77)
        self.assertIn("Test9 接入约束", library_data["text"])

    def test_routing_and_non_applicable(self):
        self.assertIn("圆锥曲线", library.build_context("math", {"id": "g11_s1"}, "椭圆的离心率")["topic"])
        self.assertIn("全等三角形", library.build_context("math", {"id": "g8_s1"}, "证明两个三角形全等")["topic"])
        for subject, term in (("physics", "g12_s2"), ("math", "")):
            self.assertEqual(library.build_context(subject, {"id": term}, "导数")["text"], "")
        self.assertIsNone(library.build_context("math", {"id": "g10_s1"}, "")["topic"])

    def test_real_pipeline_prompt_expression_contains_library(self):
        source = PROJECT / "math" / "source" / "search_machine_pipeline.py"
        self.assertTrue(source.is_file())
        text = source.read_text(encoding="utf-8")
        self.assertIn("test10", text)
        context = library.build_context("math", {"id": "g10_s1"}, "函数的单调区间")
        self.assertEqual(context["status"], "loaded")
        self.assertIn("【Test10提示词工程】", context["text"])


if __name__ == "__main__":
    unittest.main()
