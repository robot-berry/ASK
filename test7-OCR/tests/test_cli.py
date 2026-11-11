import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test7_ocr.cli import (
    _clean_source_resolver_from_config,
    _latency_guard_decision,
    _record_sample_id,
    main,
)


class CliManifestTests(unittest.TestCase):
    def test_accepts_primary_sample_id(self) -> None:
        self.assertEqual(_record_sample_id({"sample_id": "S-1"}), "S-1")

    def test_accepts_dataset_question_id(self) -> None:
        self.assertEqual(_record_sample_id({"question_id": "Q-1"}), "Q-1")

    def test_missing_identifier_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_id/question_id"):
            _record_sample_id({})

    def test_latency_guard_aborts_only_after_sustained_over_target_window(self) -> None:
        settings = {
            "enabled": True,
            "minimum_processed_samples": 6,
            "window_size": 5,
            "maximum_over_target_in_window": 3,
        }
        self.assertIsNone(
            _latency_guard_decision([3.0, 3.2, 5.2, 5.3, 5.4], 5.0, settings)
        )
        decision = _latency_guard_decision(
            [3.0, 3.2, 5.2, 5.3, 4.8, 5.4],
            5.0,
            settings,
        )
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision["reason"], "SUSTAINED_LATENCY_OVER_TARGET")
        self.assertEqual(decision["over_target_count"], 3)

    def test_latency_guard_can_be_disabled(self) -> None:
        self.assertIsNone(
            _latency_guard_decision(
                [9.0] * 20,
                5.0,
                {"enabled": False},
            )
        )

    def test_clean_source_resolution_cannot_disable_two_person_review(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot disable two-person"):
            _clean_source_resolver_from_config(
                {
                    "clean_source_resolution": {
                        "enabled": True,
                        "require_two_person_verified_pair": False,
                    }
                }
            )

    def test_batch_resource_conflict_blocks_before_model_load_or_run_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"sample_id": "sample-1", "image_path": "one.png"}) + "\n",
                encoding="utf-8",
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "storage": {"preflight_enabled": True},
                        "batch_resource_guard": {"enabled": True},
                    }
                ),
                encoding="utf-8",
            )
            output_root = root / "runs"
            ready = {"schema_version": "storage_preflight_v1", "sufficient": True}
            conflict = {
                "enabled": True,
                "status": "CONFLICT",
                "conflicts": [{"name": "other OCR"}],
                "unavailable": [],
            }
            stdout = io.StringIO()
            with (
                patch("test7_ocr.cli.preflight_batch_storage", return_value=ready),
                patch("test7_ocr.cli.probe_batch_resource_conflicts", return_value=conflict),
                patch("test7_ocr.cli._provider_from_args") as provider,
                redirect_stdout(stdout),
            ):
                exit_code = main(
                    [
                        "batch",
                        "--manifest", str(manifest),
                        "--dataset-root", str(root),
                        "--config", str(config),
                        "--output-root", str(output_root),
                        "--run-id", "blocked-run",
                    ]
                )
        self.assertEqual(exit_code, 4)
        provider.assert_not_called()
        self.assertFalse((output_root / "blocked-run").exists())
        self.assertEqual(json.loads(stdout.getvalue())["status"], "BLOCKED_RESOURCE_CONFLICT")


if __name__ == "__main__":
    unittest.main()
