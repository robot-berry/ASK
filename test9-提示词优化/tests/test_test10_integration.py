import ast
import hashlib
import json
import unittest
from pathlib import Path
import test10_prompts as library

class Test10Integration(unittest.TestCase):
    def test_every_indexed_file_exists_and_all_semester_gates(self):
        topics=json.loads((library.ROOT/"提示词/索引.json").read_text(encoding="utf-8-sig"))["topics"]
        self.assertEqual(len(topics),38)
        for topic in topics:
            self.assertTrue((library.ROOT/topic["path"]).is_file())
        for term,ceiling in library.ORDER.items():
            for question in ("函数单调区间", "椭圆离心率", "导数极值", "随机变量概率", "平面向量"):
                result=library.build_context("math",{"id":term},question)
                self.assertEqual(result["status"],"loaded")
                for source in result["sources"]:
                    raw=(library.ROOT/source["path"]).read_text(encoding="utf-8-sig")
                    self.assertEqual(source["sha256"],hashlib.sha256(raw.encode()).hexdigest())
                    selected=next((t for t in topics if t["path"]==source["path"]),None)
                    if selected:
                        rank={"高一":6,"高二":8,"高三":10}[selected["grade"]]+(selected["semester"]=="下学期")
                        self.assertLessEqual(rank,ceiling)
    def test_routing_and_non_applicable(self):
        self.assertIn("圆锥曲线",library.build_context("math",{"id":"g11_s1"},"椭圆的离心率")["topic"])
        for subject,term in (("physics","g12_s2"),("math","")):
            self.assertEqual(library.build_context(subject,{"id":term},"导数")["text"],"")
        self.assertIsNone(library.build_context("math",{"id":"g10_s1"},"")["topic"])
    def test_junior_terms_load_only_junior_sources(self):
        for grade in range(7, 10):
            for semester in (1, 2):
                term = f"g{grade}_s{semester}"
                context = library.build_context("math", {"id": term}, "解一元一次方程")
                self.assertEqual(context["status"], "loaded")
                self.assertEqual([source["path"] for source in context["sources"]],
                                 [library.JUNIOR, library.JUNIOR_SYSTEM])
                self.assertIn("严格遵守较窄范围", context["text"])
                self.assertIn("候选稿", context["text"])
                for source in context["sources"]:
                    content = (library.ROOT / source["path"]).read_text(encoding="utf-8-sig")
                    self.assertEqual(source["sha256"], hashlib.sha256(content.encode()).hexdigest())

    def test_high_school_can_use_junior_general_guidance(self):
        context = library.build_context("math", {"id": "g12_s2"}, "解一元一次方程")
        self.assertIn(library.JUNIOR, [source["path"] for source in context["sources"]])
        self.assertIn(library.HIGH_ENTRY, [source["path"] for source in context["sources"]])
    def test_real_pipeline_prompt_expression_contains_library(self):
        source=Path(library.__file__).parent/"math/source/search_machine_pipeline.py"
        tree=ast.parse(source.read_text(encoding="utf-8"))
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="provider_prompt" for t in n.targets))
        context=library.build_context("math",{"id":"g10_s1"},"函数的单调区间")
        env={"prompt_snapshot":{"text":"BASE"},"curriculum_scope":{},"curriculum_scope_instruction":lambda *a,**k:"SCOPE", "local_ocr_text":"QUESTION", "state":{"curriculum_selection":{"prompt":"SEMESTER"}},"test10":context,"ocr_context":"OCR"}
        result=eval(compile(ast.Expression(assignment.value),str(source),"eval"),env)
        self.assertIn(context["text"],result)
        self.assertTrue(result.endswith("OCR"))
        self.assertIn("SEMESTER",result)
