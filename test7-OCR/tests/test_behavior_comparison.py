from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_behavior_runs import compare_runs


class BehaviorComparisonTests(unittest.TestCase):
    @staticmethod
    def _write_run(
        root: Path,
        run_name: str,
        *,
        text: str,
        status: str = "REVIEW",
        elapsed: float = 2.0,
        formula_count: int = 0,
        visual_count: int = 1,
        latency_sla_applicable: bool = True,
    ) -> Path:
        sample = root / run_name / "Q1"
        sample.mkdir(parents=True)
        document = sample / "question_document.json"
        document.write_text(
            json.dumps(
                {
                    "display_text": text,
                    "formula_count": formula_count,
                    "visual_assets": [{} for _ in range(visual_count)],
                }
            ),
            encoding="utf-8",
        )
        (sample / "report.json").write_text(
            json.dumps(
                {
                    "sample_id": "Q1",
                    "question_document": str(document),
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "measurement": {
                        "online_latency_sla_applicable": latency_sla_applicable,
                    },
                }
            ),
            encoding="utf-8",
        )
        return root / run_name

    def test_flags_text_loss_but_visual_regrouping_is_review_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._write_run(root, "baseline", text="完整题干和四个选项", visual_count=4)
            candidate = self._write_run(root, "candidate", text="题干", visual_count=1)
            result = compare_runs(baseline, candidate)
        self.assertEqual(result["summary"]["regression_count"], 1)
        sample = result["samples"][0]
        self.assertIn("TEXT_SIMILARITY_LOW", sample["regression_reasons"])
        self.assertIn("VISUAL_COUNT_CHANGED", sample["review_signals"])

    def test_equivalent_text_and_fast_latency_have_no_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._write_run(root, "baseline", text="A. 选项甲\nB. 选项乙")
            candidate = self._write_run(root, "candidate", text="A.选项甲 B.选项乙", elapsed=3.0)
            result = compare_runs(baseline, candidate)
        self.assertEqual(result["summary"]["regression_count"], 0)
        self.assertEqual(result["summary"]["latency_under_target_count"], 1)
        self.assertEqual(result["summary"]["latency_sample_count"], 1)
        self.assertFalse(result["formal_gold_scoring"])

    def test_render_only_time_is_excluded_from_online_latency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._write_run(root, "baseline", text="同一内容")
            candidate = self._write_run(
                root,
                "candidate",
                text="同一内容",
                elapsed=99.0,
                latency_sla_applicable=False,
            )
            result = compare_runs(baseline, candidate)
        sample = result["samples"][0]
        self.assertIsNone(sample["candidate_latency_seconds"])
        self.assertFalse(sample["candidate_latency_sla_applicable"])
        self.assertNotIn("LATENCY_OVER_TARGET", sample["review_signals"])
        self.assertEqual(result["summary"]["latency_sample_count"], 0)
        self.assertIsNone(result["summary"]["maximum_latency_seconds"])

    def test_relaxing_a_previous_reject_is_a_safety_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = self._write_run(root, "baseline", text="同一内容", status="REJECT")
            candidate = self._write_run(root, "candidate", text="同一内容", status="REVIEW")
            result = compare_runs(baseline, candidate)
        self.assertEqual(result["summary"]["regression_count"], 1)
        self.assertIn(
            "SAFETY_GATE_RELAXED",
            result["samples"][0]["regression_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
