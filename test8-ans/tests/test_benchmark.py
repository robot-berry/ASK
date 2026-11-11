from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test8_ans.benchmark import build_benchmark_summary, validate_full_promotion
from test8_ans.config import Settings
from test8_ans.dataset import EXPECTED_DATASET_ID
from test8_ans.workflow import PROJECT_ROOT


def passing_row(index: int) -> dict:
    latency = {
        "valid": True,
        "scope": "files_ready_with_png",
        "budget_ms": 20_000,
        "measured_ms": 10_000.0,
        "model_ms": 9_000.0,
        "workflow_ms": 9_100.0,
        "generation_pipeline_ms": 9_100.0,
        "files_ready_with_png_ms": 9_400.0,
        "png_render_ms": 300.0,
        "excluded_upstream_ms": 900.0,
        "transport_attempts": 1,
    }
    return {
        "id": f"q-{index}",
        "status": "completed",
        "quality_gate_pass": True,
        "schema_pass": True,
        "relevance_pass": True,
        "latency_pass": True,
        "answer_guard_pass": True,
        "display_ready": True,
        "redaction_count": 0,
        "latency": latency,
        "provider_model": "qwen-max",
        "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
        "source_quality": {"report_status": "PASS", "formula_verified": True},
        "run_dir": "unused",
        "attempts": 1,
    }


class BenchmarkTests(unittest.TestCase):
    def test_passing_smoke_can_promote_unchanged_configuration(self) -> None:
        settings = Settings(api_key="key")
        with tempfile.TemporaryDirectory() as directory:
            batch_dir = Path(directory)
            summary = build_benchmark_summary(
                batch_dir=batch_dir,
                phase="smoke",
                stage="senior",
                preparation={"dataset_id": EXPECTED_DATASET_ID},
                results=[passing_row(index) for index in range(20)],
                settings=settings,
                project_root=PROJECT_ROOT,
            )
            self.assertTrue(summary["promotion_gate_pass"])
            self.assertEqual(summary["rates"]["answer_guard_pass_rate"], 1.0)
            self.assertEqual(summary["token_usage"]["total_tokens"], 6000)
            loaded = validate_full_promotion(
                summary_path=batch_dir / "benchmark_summary.json",
                settings=settings,
                project_root=PROJECT_ROOT,
                dataset_id=EXPECTED_DATASET_ID,
                stage="senior",
            )
            self.assertEqual(loaded["phase"], "smoke")

    def test_one_of_twenty_schema_failures_still_meets_95_percent(self) -> None:
        rows = [passing_row(index) for index in range(20)]
        rows[0]["schema_pass"] = False
        rows[0]["quality_gate_pass"] = False
        settings = Settings(api_key="key")
        with tempfile.TemporaryDirectory() as directory:
            summary = build_benchmark_summary(
                batch_dir=Path(directory),
                phase="smoke",
                stage="senior",
                preparation={"dataset_id": EXPECTED_DATASET_ID},
                results=rows,
                settings=settings,
                project_root=PROJECT_ROOT,
            )
            self.assertEqual(summary["rates"]["schema_pass_rate"], 0.95)
            self.assertTrue(summary["promotion_gate_pass"])

    def test_any_answer_guard_failure_blocks_promotion(self) -> None:
        rows = [passing_row(index) for index in range(20)]
        rows[0]["answer_guard_pass"] = False
        rows[0]["display_ready"] = False
        rows[0]["quality_gate_pass"] = False
        settings = Settings(api_key="key")
        with tempfile.TemporaryDirectory() as directory:
            summary = build_benchmark_summary(
                batch_dir=Path(directory),
                phase="smoke",
                stage="senior",
                preparation={"dataset_id": EXPECTED_DATASET_ID},
                results=rows,
                settings=settings,
                project_root=PROJECT_ROOT,
            )
            self.assertEqual(summary["rates"]["answer_guard_pass_rate"], 0.95)
            self.assertFalse(summary["gate_checks"]["answer_guard_all"])
            self.assertFalse(summary["promotion_gate_pass"])


if __name__ == "__main__":
    unittest.main()
