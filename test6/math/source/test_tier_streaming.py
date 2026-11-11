"""Exercise partial JSON, real tier rendering and the publication safety path."""
import copy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import search_machine_pipeline as pipeline_module
import full_benchmark as benchmark

from PIL import Image

from compact_guidance import CompactStreamCollector, clean_model_analysis_steps, expand_compact_output, local_fallback_compact
from math_guidance_renderer import tier_blocks
from search_machine_pipeline import SearchMachinePipeline
from tier_streaming import TierStreamPublisher


def sample():
    doc = local_fallback_compact(subject="mathematics", part_labels=["整题"])
    return {key: doc[key] for key in ("l1", "l2", "l3", "l4", "l5", "f", "pat", "s", "u")}


class TierStreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.state = {"job_id": "test", "queue_wait_seconds": 0.2}
        Image.new("L", (800, 160), 255).save(self.root / "01_question_knowledge.png", dpi=(225.778, 225.778))
        self.pipeline = SearchMachinePipeline.__new__(SearchMachinePipeline)

    def tearDown(self):
        self.tmp.cleanup()

    def publisher(self):
        return self.pipeline._tier_stream_publisher(
            self.root, self.state, started=time.perf_counter(),
            question_text="整理已知条件并说明所求量之间的关系。",
            curriculum_stage="junior_high", request_sha256="",
        )

    def test_real_pngs_publish_before_root_closes_without_future_tiers(self):
        publisher = self.publisher()
        doc = sample()
        prefix = "{"
        first_novice_seconds = None
        for index, (field, value) in enumerate(doc.items()):
            prefix += ("," if index else "") + json.dumps(field) + ":" + json.dumps(value, ensure_ascii=False)
            publisher(prefix, (index + 1) * 1000)
            publisher(prefix, (index + 1) * 1000)  # repeated chunk cannot republish
            if index < 4:
                self.assertEqual(len(self.state.get("images", [])), index + 2, self.state)
                self.assertFalse((self.root / "06_guidance_novice.png").exists())
            else:
                self.assertTrue((self.root / "06_guidance_novice.png").exists(), publisher.audit)
                self.assertEqual(len(self.state["images"]), 6, self.state)
                if index == 4:
                    first_novice_seconds = publisher.audit["novice"]["first_published_seconds"]
                    self.assertFalse(publisher.audit["novice"].get("metadata_refreshed"))
        self.assertEqual(len(self.state["images"]), 6, self.state)
        self.assertEqual(self.state["status"], "solving")
        self.assertTrue(all(e["published"] for e in publisher.audit.values()))
        self.assertTrue(publisher.audit["novice"].get("metadata_refreshed"))
        self.assertEqual(publisher.audit["novice"]["first_published_seconds"], first_novice_seconds)
        self.assertFalse(list(self.root.glob("*.tmp.png")))
        for metadata in self.state["images"]:
            self.assertTrue(metadata["valid"])

    def test_incomplete_array_never_publishes(self):
        publisher = self.publisher()
        raw = json.dumps({"l1": sample()["l1"]}, ensure_ascii=False)
        boundary = raw.rfind("]")
        for end in range(boundary + 1):
            publisher(raw[:end], end)
        self.assertFalse((self.root / "02_guidance_elite.png").exists())
        publisher(raw[:boundary + 1], 2000)
        self.assertTrue((self.root / "02_guidance_elite.png").exists(), publisher.audit)

    def test_missing_sentences_defer_to_final_without_publishing_placeholders(self):
        publisher = self.publisher()
        doc = sample()
        doc["l1"][0]["s"] = [doc["l1"][0]["s"][0]]
        publisher(json.dumps({"l1": doc["l1"]}, ensure_ascii=False), 1)
        self.assertFalse((self.root / "02_guidance_elite.png").exists())
        self.assertTrue(publisher.audit["elite"]["deferred_to_final"])

    def test_rejection_or_render_failure_does_not_abort_stream(self):
        audit = {}
        publisher = TierStreamPublisher(
            directory=self.root, audit=audit,
            prepare=lambda *args: (_ for _ in ()).throw(ValueError("unsafe")),
            render=lambda *args: None, inspect=lambda p: {"valid": True},
            publish=lambda *args: self.fail("unsafe tier published"),
        )
        publisher(json.dumps(sample(), ensure_ascii=False), 1000)
        self.assertEqual(len(audit), 5)
        self.assertTrue(all(e["deferred_to_final"] for e in audit.values()))

    def test_actual_answer_redaction_is_applied_before_early_render(self):
        publisher = self.publisher()
        doc = sample()
        doc["l3"][0]["s"][1] = "最终答案为42。核对题设条件的适用范围。"
        outputs = []
        real_render = publisher.render

        def record(kind, output, destination):
            outputs.append((kind, copy.deepcopy(output)))
            real_render(kind, output, destination)

        publisher.render = record
        publisher(json.dumps(doc, ensure_ascii=False), 1000)
        self.assertTrue(outputs, publisher.audit)
        for kind, output in outputs:
            visible = json.dumps(output["display_guidance"][kind], ensure_ascii=False)
            self.assertNotIn("答案为42", visible)

    def test_stream_interruption_preserves_completed_model_tiers(self):
        doc = sample()
        collector = CompactStreamCollector()
        prefix = {"l1": doc["l1"], "l2": doc["l2"]}
        collector(json.dumps(prefix, ensure_ascii=False)[:-1] + ',"l3":[{"n":"整', 900)
        partial = collector.completed_skeleton()
        self.assertEqual(partial, prefix)
        recovered = local_fallback_compact(subject="mathematics", part_labels=["整题"], partial=partial)
        self.assertEqual(recovered["l1"], doc["l1"])
        self.assertEqual(recovered["l2"], doc["l2"])

    def test_draft_cleanup_preserves_valid_facts_and_formula_punctuation(self):
        factual = "旋转保持对应边长度不变。"
        steps, drafts, duplicates = clean_model_analysis_steps([
            factual, "角度是15度？不对，应换一种方法。", factual,
            "由$A；B$之间的关系可建立方程。", "该图形不对称，需核对对应关系。",
        ])
        self.assertEqual(drafts, 1)
        self.assertEqual(duplicates, 1)
        self.assertEqual(steps, [factual, "由$A；B$之间的关系可建立方程。", "该图形不对称，需核对对应关系。"])

    def test_student_questions_are_not_removed_by_draft_cleanup(self):
        doc = sample()
        doc["l1"][0]["s"][0] = "这个推断不对时，你会先检查哪个条件？"
        recovered = local_fallback_compact(subject="mathematics", part_labels=["整题"], partial=doc)
        self.assertEqual(recovered["l1"], doc["l1"])

    def test_png_blocks_remove_duplicate_labels_and_single_part_heading(self):
        doc = sample()
        for field in ("l1", "l3", "l4", "l5"):
            doc[field][0]["n"] = "(1)"
        doc["l2"] = ["知识点：函数单调性", "考点：定义域", "范围核对"]
        doc["l5"][0]["pit"] = "易错点：不要忽略定义域"
        output, _ = expand_compact_output(doc, subject="mathematics")

        blocks = tier_blocks("novice", output)
        visible = "\n".join(block[0] for block in blocks)

        self.assertNotIn("知识点：函数单调性", visible)
        self.assertNotIn("考点：定义域", visible)
        self.assertIn("• 函数单调性", visible)
        self.assertIn("• 定义域", visible)
        self.assertNotIn("易错点：易错点：", visible)
        self.assertNotIn(("(1)", "section", 2), blocks)

    def test_status_write_recovers_from_transient_windows_reader_lock(self):
        original_replace = pipeline_module.os.replace
        attempts = []

        def sharing_violation_then_success(source, destination):
            attempts.append(1)
            if len(attempts) < 3:
                raise PermissionError("simulated Windows reader lock")
            original_replace(source, destination)

        path = self.root / "job.json"
        with patch.object(pipeline_module.os, "replace", side_effect=sharing_violation_then_success):
            pipeline_module.atomic_json(path, {"status": "solving"})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "solving"})
        self.assertEqual(len(attempts), 3)

    def test_quality_stop_only_after_all_five_tiers_have_model_prose(self):
        doc = sample()
        doc["l5"][0]["s"] += ["15度？不对，应再核对。", "30度？不对，应再核对。", "45度？不对，应再核对。"]
        raw = json.dumps(doc, ensure_ascii=False)
        cut = raw.index(', "f"')
        collector = CompactStreamCollector()
        collector(raw[:cut - 2], 1000)
        self.assertTrue(collector.should_stop_repetitive_draft())
        salvaged = collector.completed_skeleton()
        self.assertGreaterEqual(len(salvaged["l5"][0]["s"]), 5)
        self.assertEqual(salvaged["l1"], doc["l1"])

    def test_normal_long_output_is_not_stopped(self):
        doc = sample()
        doc["l5"][0]["s"] = [f"第{i}项条件用来说明不同的关系，必须核对其适用范围。" for i in range(100)]
        collector = CompactStreamCollector()
        collector(json.dumps(doc, ensure_ascii=False), 1000)
        self.assertFalse(collector.should_stop_repetitive_draft())

    def test_draft_loop_does_not_stop_before_all_parts_have_five_usable_steps(self):
        doc = sample()
        doc["l5"][0]["s"] = doc["l5"][0]["s"][:2] + ["15度？不对，应再核对。"] * 5
        raw = json.dumps(doc, ensure_ascii=False)
        collector = CompactStreamCollector()
        collector(raw[:raw.index(', "f"') - 2], 1000)
        self.assertFalse(collector.should_stop_repetitive_draft())

    def test_quality_stop_salvages_completed_l5_prefix_before_later_part(self):
        doc = sample()
        for field in ("l1", "l3", "l4", "l5"):
            first = copy.deepcopy(doc[field][0])
            first["n"] = "（1）"
            second = copy.deepcopy(first)
            second["n"] = "（2）"
            third = copy.deepcopy(first)
            third["n"] = "（3）"
            doc[field] = [first, second, third]
        doc["l5"][1]["s"] += [
            "需准确推导后再决定。",
            "是否有更小值？",
            "暂定$x_0=2$时最小。",
        ]
        doc["l5"] = doc["l5"][:2]
        raw = json.dumps(doc, ensure_ascii=False)
        collector = CompactStreamCollector()
        collector(raw[:raw.index(', "f"') - 2], 1000)

        self.assertTrue(collector.should_stop_repetitive_draft())
        salvaged = collector.completed_skeleton()
        self.assertEqual(len(salvaged["l5"]), 2)
        self.assertEqual(len(salvaged["l1"]), 3)

    def test_novice_publishes_when_l5_closes_without_metadata(self):
        publisher = self.publisher()
        doc = sample()
        prefix = {key: doc[key] for key in ("l1", "l2", "l3", "l4", "l5")}
        publisher(json.dumps(prefix, ensure_ascii=False), 1000)
        self.assertTrue((self.root / "06_guidance_novice.png").exists(), publisher.audit)
        self.assertTrue(publisher.audit["novice"]["published"])
        self.assertFalse(publisher.audit["novice"].get("metadata_refreshed"))
        first_seconds = publisher.audit["novice"]["first_published_seconds"]
        publisher(json.dumps(doc, ensure_ascii=False), 2000)
        self.assertTrue(publisher.audit["novice"].get("metadata_refreshed"))
        self.assertEqual(publisher.audit["novice"]["first_published_seconds"], first_seconds)
        publisher(json.dumps(doc, ensure_ascii=False), 3000)
        self.assertEqual(publisher.audit["novice"].get("metadata_refresh_error"), None)

    def _byte_publisher(self, **kwargs):
        audit = {}
        def render(kind, output, destination):
            destination.write_bytes(kind.encode("utf-8"))
        return TierStreamPublisher(
            directory=self.root, audit=audit,
            prepare=lambda prefix, kind: prefix,
            render=render, inspect=lambda path: {"valid": True, "sha256": path.name},
            publish=lambda *args: None, **kwargs,
        ), audit

    def test_post_lower_budget_does_not_stop_before_model_l5(self):
        publisher, audit = self._byte_publisher(post_lower_budget_seconds=0.05)
        prefix = {key: sample()[key] for key in ("l1", "l2", "l3", "l4")}
        publisher(json.dumps(prefix, ensure_ascii=False), 1000)
        self.assertTrue(audit["lower"]["published"])
        self.assertNotIn("novice", audit)
        time.sleep(0.06)
        self.assertFalse(publisher.should_stop_post_lower_budget())
        self.assertIsNone(publisher.post_lower_stop_reason)

    def test_post_lower_budget_stops_only_after_model_l5(self):
        publisher, audit = self._byte_publisher(post_lower_budget_seconds=0.05)
        prefix = {key: sample()[key] for key in ("l1", "l2", "l3", "l4", "l5")}
        publisher(json.dumps(prefix, ensure_ascii=False), 1000)
        self.assertTrue(audit["novice"]["published"])
        self.assertFalse(publisher.should_stop_post_lower_budget())
        time.sleep(0.06)
        self.assertTrue(publisher.should_stop_post_lower_budget())
        self.assertEqual(publisher.post_lower_stop_reason, "post_lower_budget_exhausted")
        self.assertTrue(publisher.should_stop_post_lower_budget())

    def test_complete_stream_within_post_lower_budget_does_not_stop(self):
        publisher, audit = self._byte_publisher(post_lower_budget_seconds=5.0)
        publisher(json.dumps(sample(), ensure_ascii=False), 1000)
        self.assertTrue(all(entry["published"] for entry in audit.values()))
        self.assertTrue(audit["novice"].get("published"))
        self.assertFalse(publisher.should_stop_post_lower_budget())
        self.assertIsNone(publisher.post_lower_stop_reason)

    def test_zero_post_lower_budget_never_stops(self):
        publisher, audit = self._byte_publisher(post_lower_budget_seconds=0)
        prefix = {key: sample()[key] for key in ("l1", "l2", "l3", "l4")}
        publisher(json.dumps(prefix, ensure_ascii=False), 1000)
        self.assertTrue(audit["lower"]["published"])
        time.sleep(0.02)
        self.assertFalse(publisher.should_stop_post_lower_budget())

    def test_sse_callback_can_end_a_confirmed_draft_loop(self):
        events = []
        for content in ("first", "never read"):
            envelope = {"choices": [{"delta": {"content": content}}]}
            events.append(("data: " + json.dumps(envelope) + "\n\n").encode())
        with benchmark.stream_content_callback(lambda *_: False):
            parsed = benchmark.parse_sse_response(io.BytesIO(b"".join(events)), benchmark.monotonic_ms(), 5)
        self.assertEqual(parsed["output_text"], "first")
        self.assertEqual(parsed["finish_reason"], "local_output_quality_stop")


if __name__ == "__main__":
    unittest.main()
