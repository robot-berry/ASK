from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test8_ans.client import Completion
from test8_ans.config import Settings
from test8_ans.models import Question
from test8_ans.prompting import read_prompt
from test8_ans.workflow import PROJECT_ROOT, GuidanceWorkflow
from tests.test_schema import valid_answer


class FakeClient:
    def __init__(self, contents: list[str], *, latency_ms: float = 0.0):
        self.contents = contents
        self.latency_ms = latency_ms
        self.calls = 0

    def build_payload(self, messages: list[dict], schema: dict) -> dict:
        return {"model": "fake-qwen-max", "messages": messages, "schema": schema}

    def complete(self, messages: list[dict], schema: dict) -> Completion:
        content = self.contents[self.calls]
        self.calls += 1
        return Completion(
            content=content,
            response={"choices": [{"message": {"content": content}}]},
            request_id=f"request-{self.calls}",
            latency_ms=self.latency_ms,
        )


class WorkflowTests(unittest.TestCase):
    def settings(self, repair_attempts: int = 1) -> Settings:
        return Settings(
            api_key="test-key",
            model="fake-qwen-max",
            repair_attempts=repair_attempts,
        )

    def item(self) -> Question:
        return Question.from_mapping(
            {
                "id": "math-001",
                "subject": "math",
                "stage": "junior",
                "question": "解方程：2(x-1)=6。",
            }
        )

    def test_only_model_section_is_sent_as_system_prompt(self) -> None:
        prompt = read_prompt(PROJECT_ROOT, "math")
        self.assertIn("苏格拉底式启发", prompt)
        self.assertNotIn("## 二、发给模型的系统提示词", prompt)
        self.assertNotIn("## 三、设备端排版约束", prompt)

    def test_transport_contract_forbids_duplicate_s_keys(self) -> None:
        workflow = GuidanceWorkflow(self.settings(), client=FakeClient([]))  # type: ignore[arg-type]
        preview = workflow.preview(self.item())
        user_message = preview["payload"]["messages"][1]["content"]
        self.assertIn("严禁重复同名键", user_message)
        self.assertIn("l5恰好4项", user_message)

    def test_successful_run_writes_replay_files(self) -> None:
        content = json.dumps(valid_answer(), ensure_ascii=False)
        fake = FakeClient([content])
        with tempfile.TemporaryDirectory() as directory:
            workflow = GuidanceWorkflow(
                self.settings(),
                project_root=PROJECT_ROOT,
                runs_root=Path(directory),
                client=fake,  # type: ignore[arg-type]
            )
            result = workflow.run(self.item())
            self.assertEqual(result.attempts, 1)
            self.assertTrue((result.run_dir / "answer.json").is_file())
            self.assertTrue((result.run_dir / "display_answer.json").is_file())
            self.assertTrue((result.run_dir / "answer_redaction.json").is_file())
            self.assertEqual(
                len(list((result.run_dir / "images_800px").glob("l*.png"))), 5
            )
            self.assertTrue(
                (
                    result.run_dir
                    / "model_answer_screenshot"
                    / "model_output.png"
                ).is_file()
            )
            self.assertTrue(result.validation["answer_guard"]["valid"])
            self.assertTrue(result.validation["render"]["valid"])
            self.assertTrue(result.validation["model_answer_screenshot"]["valid"])
            self.assertIn(
                "files_ready_with_png_ms", result.validation["latency"]
            )
            metadata = json.loads(
                (result.run_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["state"], "quality_passed")
            self.assertEqual(
                (result.run_dir / "model_output.txt").read_text(encoding="utf-8"),
                content,
            )

    def test_invalid_first_answer_is_repaired_once(self) -> None:
        content = json.dumps(valid_answer(), ensure_ascii=False)
        fake = FakeClient(["not json", content])
        with tempfile.TemporaryDirectory() as directory:
            workflow = GuidanceWorkflow(
                self.settings(repair_attempts=1),
                project_root=PROJECT_ROOT,
                runs_root=Path(directory),
                client=fake,  # type: ignore[arg-type]
            )
            result = workflow.run(self.item())
            self.assertEqual(result.attempts, 2)
            self.assertEqual(fake.calls, 2)
            self.assertTrue((result.run_dir / "attempt_01_validation.json").is_file())

    def test_final_value_is_removed_only_from_display_copy(self) -> None:
        answer = valid_answer()
        answer["l5"][0]["s"][2] = "整理关系后可得$x=4$。"
        content = json.dumps(answer, ensure_ascii=False)
        fake = FakeClient([content])
        with tempfile.TemporaryDirectory() as directory:
            workflow = GuidanceWorkflow(
                self.settings(),
                project_root=PROJECT_ROOT,
                runs_root=Path(directory),
                client=fake,  # type: ignore[arg-type]
            )
            result = workflow.run(self.item())
            raw_answer = (result.run_dir / "answer.json").read_text(encoding="utf-8")
            display = (result.run_dir / "display_answer.json").read_text(
                encoding="utf-8"
            )
            self.assertIn("x=4", raw_answer)
            self.assertNotIn("x=4", display)
            self.assertEqual(result.validation["answer_guard"]["redaction_count"], 1)
            self.assertEqual(len(result.display_answer["l5"][0]["s"]), 3)
            self.assertTrue(result.quality_gate_pass)

    def test_schema_quality_failure_still_renders_without_second_model_call(self) -> None:
        answer = valid_answer()
        answer["l5"][0]["s"].pop()
        content = json.dumps(answer, ensure_ascii=False)
        fake = FakeClient([content])
        with tempfile.TemporaryDirectory() as directory:
            workflow = GuidanceWorkflow(
                self.settings(repair_attempts=1),
                project_root=PROJECT_ROOT,
                runs_root=Path(directory),
                client=fake,  # type: ignore[arg-type]
            )
            result = workflow.run(self.item())
            self.assertEqual(fake.calls, 1)
            self.assertFalse(result.validation["schema"]["valid"])
            self.assertTrue(result.validation["display_ready"])
            self.assertTrue(result.validation["render"]["valid"])
            self.assertFalse(result.quality_gate_pass)
            self.assertEqual(
                len(list((result.run_dir / "images_800px").glob("l*.png"))), 5
            )

    def test_unresolved_answer_target_is_review_warning_not_render_blocker(self) -> None:
        content = json.dumps(valid_answer(), ensure_ascii=False)
        fake = FakeClient([content])
        item = Question.from_mapping(
            {
                "id": "math-review-warning",
                "subject": "math",
                "stage": "junior",
                "question": "请阅读题目并分析。",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            workflow = GuidanceWorkflow(
                self.settings(repair_attempts=1),
                project_root=PROJECT_ROOT,
                runs_root=Path(directory),
                client=fake,  # type: ignore[arg-type]
            )
            result = workflow.run(item)
            self.assertFalse(result.validation["answer_guard"]["valid"])
            self.assertFalse(result.validation["answer_guard"]["render_blocking"])
            self.assertTrue(result.validation["render"]["valid"])
            self.assertEqual(fake.calls, 1)

    def test_latency_over_twenty_seconds_does_not_block_png(self) -> None:
        content = json.dumps(valid_answer(), ensure_ascii=False)
        fake = FakeClient([content], latency_ms=20_001)
        with tempfile.TemporaryDirectory() as directory:
            workflow = GuidanceWorkflow(
                Settings(
                    api_key="test-key",
                    model="fake-qwen-max",
                    repair_attempts=0,
                    latency_scope="model_only",
                ),
                project_root=PROJECT_ROOT,
                runs_root=Path(directory),
                client=fake,  # type: ignore[arg-type]
            )
            result = workflow.run(self.item())
            self.assertFalse(result.validation["latency"]["valid"])
            self.assertTrue(result.validation["render"]["valid"])
            self.assertFalse(result.quality_gate_pass)


if __name__ == "__main__":
    unittest.main()
