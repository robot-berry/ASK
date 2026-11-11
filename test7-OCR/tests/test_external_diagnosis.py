from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test7_ocr.external_diagnosis import classify_failure, diagnose_run
from test7_ocr.storage import write_json_atomic


class ExternalDiagnosisTests(unittest.TestCase):
    def test_classify_failure_kinds(self) -> None:
        self.assertEqual(classify_failure("MemoryError: Unable to allocate 142. MiB"), "huge_image_oom")
        self.assertEqual(
            classify_failure("ValueError: Empty crop after clamping: BBox(x=3024, y=628, width=0, height=294)"),
            "empty_crop_bbox",
        )
        self.assertEqual(
            classify_failure("OSError: image file is truncated (151 bytes not processed)"),
            "truncated_image",
        )

    def test_diagnose_run_counts_reject_drivers_without_enabling_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            sample = run / "ODB-1"
            sample.mkdir(parents=True)
            write_json_atomic(
                sample / "report.json",
                {
                    "sample_id": "ODB-1",
                    "status": "REJECT",
                    "issues": [
                        {"code": "FORMULA_UNVERIFIED", "severity": "review", "message": "review"},
                        {"code": "PARTIAL_OPTION_SET", "severity": "reject", "message": "reject"},
                    ],
                },
            )
            write_json_atomic(
                run / "batch_summary.json",
                {
                    "attempted": 1,
                    "completed_sample_count": 1,
                    "statuses": {"REJECT": 1},
                    "failures": [{"sample_id": "ODB-1-OOM", "error": "MemoryError: boom"}],
                },
            )
            eval_path = root / "eval.json"
            write_json_atomic(
                eval_path,
                {
                    "evaluated_count": 1,
                    "missing_count": 0,
                    "metrics": {"mean_character_error_rate": 0.5},
                    "details": [
                        {
                            "sample_id": "ODB-1",
                            "evaluated": True,
                            "character_error_rate": 0.6,
                            "token_coverage": 0.4,
                            "formula_match_rate": 1.0,
                        }
                    ],
                },
            )
            summary = diagnose_run(
                dataset_id="omnidocbench",
                label="exam_zh",
                run_directory=run,
                eval_report=eval_path,
            )
            self.assertFalse(summary["threshold_calibration_allowed"])
            self.assertFalse(summary["formal_gold_scoring"])
            self.assertEqual(summary["crash_groups"]["huge_image_oom"]["count"], 1)
            self.assertEqual(summary["primary_reject_codes"][0]["code"], "PARTIAL_OPTION_SET")
            self.assertEqual(summary["high_formula_match_high_cer"]["count"], 1)
            codes = {row["code"]: row for row in summary["issue_codes"]}
            self.assertEqual(codes["PARTIAL_OPTION_SET"]["reject_driver_samples"], 1)
            self.assertEqual(codes["FORMULA_UNVERIFIED"]["reject_driver_samples"], 0)


if __name__ == "__main__":
    unittest.main()
