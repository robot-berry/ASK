from __future__ import annotations

import unittest

from test8_ans.dataset import balanced_sample_rows


class DatasetSampleTests(unittest.TestCase):
    def test_balanced_sample_spreads_across_subjects_and_groups(self) -> None:
        rows = []
        for subject in ("mathematics", "physics"):
            for group in range(1, 11):
                for index in range(1, 5):
                    rows.append(
                        {
                            "question_id": f"{subject}-{group}-{index}",
                            "subject": subject,
                            "source_group": f"Test{group}",
                            "source_index": index,
                            "include_in_benchmark": True,
                        }
                    )
        selected = balanced_sample_rows(rows, 50)
        self.assertEqual(len(selected), 50)
        self.assertEqual(
            sum(row["subject"] == "mathematics" for row in selected), 25
        )
        self.assertEqual(sum(row["subject"] == "physics" for row in selected), 25)
        for subject in ("mathematics", "physics"):
            groups = {
                row["source_group"] for row in selected if row["subject"] == subject
            }
            self.assertEqual(len(groups), 10)


if __name__ == "__main__":
    unittest.main()
