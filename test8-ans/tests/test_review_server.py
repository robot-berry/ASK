from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test8_ans.review_server import save_feedback


class ReviewFeedbackTests(unittest.TestCase):
    def test_feedback_is_saved_with_summary_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "comparison_manifest.jsonl").write_text(
                '\n'.join(
                    json.dumps({"id": item_id}, ensure_ascii=False)
                    for item_id in ("math-1", "physics-1")
                )
                + "\n",
                encoding="utf-8",
            )
            summary = save_feedback(
                root,
                {
                    "storage_key": "test-key",
                    "items": [
                        {
                            "id": "math-1",
                            "status": "issue",
                            "severity": "high",
                            "issues": ["源码或公式显示"],
                            "note": "分式没有正常排版",
                        }
                    ],
                },
            )
            self.assertEqual(summary["issue_count"], 1)
            self.assertEqual(summary["unreviewed_count"], 1)
            self.assertEqual(summary["issue_type_counts"]["源码或公式显示"], 1)
            self.assertTrue((root / "review_feedback.json").is_file())
            self.assertTrue((root / "review_feedback.csv").is_file())
            self.assertTrue((root / "review_feedback_summary.json").is_file())
            saved = json.loads(
                (root / "review_feedback.json").read_text(encoding="utf-8-sig")
            )
            self.assertEqual(len(saved["items"]), 2)
            self.assertEqual(saved["items"][1]["status"], "unreviewed")


if __name__ == "__main__":
    unittest.main()
