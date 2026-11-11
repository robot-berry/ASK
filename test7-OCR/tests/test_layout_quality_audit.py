from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_layout_quality import audit_layout_quality


class LayoutQualityAuditTests(unittest.TestCase):
    def _report(self, root: Path, sample_id: str, render: dict[str, object]) -> None:
        directory = root / sample_id
        directory.mkdir(parents=True)
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "status": "REVIEW",
                    "output_image": str(directory / "output.png"),
                    "render": render,
                }
            ),
            encoding="utf-8",
        )

    def test_low_fill_is_review_candidate_not_a_failed_audit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._report(
                root,
                "edge-low",
                {
                    "engine": "edge-katex",
                    "stem_line_count": 3,
                    "stem_min_line_fill_ratio": 0.20,
                    "stem_min_internal_line_fill_ratio": 0.40,
                    "stem_text_wrap_applied": "balance",
                },
            )
            self._report(root, "pillow", {"engine": "pillow"})
            result = audit_layout_quality(root, internal_fill_threshold=0.50)
        self.assertTrue(result["passed"])
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["sample_id"], "edge-low")
        self.assertEqual(result["pillow_unmeasured_count"], 1)

    def test_missing_edge_metrics_and_fallback_fail_integrity_check(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._report(root, "edge-missing", {"engine": "edge-katex"})
            self._report(root, "fallback", {"engine": "pillow-safe-fallback"})
            result = audit_layout_quality(root)
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_metric_samples"], ["edge-missing"])
        self.assertEqual(result["fallback_samples"], ["fallback"])


if __name__ == "__main__":
    unittest.main()
