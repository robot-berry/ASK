import unittest
from test_portal import portal,state

class PromptGuidanceDisplayTests(unittest.TestCase):
    def test_loaded_sources_exposed_without_full_prompt(self):
        payload=state()
        payload["preprocess"]={"test10_prompt":{"status":"loaded","topic":"03-圆锥曲线","term":"g11_s1","revision":"abc","text":"internal prompt", "sources":[{"path":"提示词/数学/高中/高二/上学期/03-圆锥曲线.md","sha256":"def"}]}}
        result=portal.sanitize_job("math",payload)["prompt_guidance"]
        self.assertEqual(result["topic"],"03-圆锥曲线")
        self.assertEqual(result["sources"][0]["sha256"],"def")
        self.assertNotIn("text",result)
    def test_legacy_and_non_applicable_are_distinct(self):
        payload=state()
        self.assertEqual(portal.sanitize_job("math",payload)["prompt_guidance"]["status"],"not_recorded")
        payload["preprocess"]={"test10_prompt":{"status":"not_applicable","text":"","sources":[]}}
        self.assertEqual(portal.sanitize_job("math",payload)["prompt_guidance"]["status"],"not_applicable")
