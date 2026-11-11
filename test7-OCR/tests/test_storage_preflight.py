from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test7_ocr.cli import main
from test7_ocr.storage import preflight_batch_storage


class StoragePreflightTests(unittest.TestCase):
    def test_uses_recent_artifact_p95_with_safety_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "runs"
            for index, size in enumerate((100, 200, 300, 400), start=1):
                directory = output_root / "run-1" / f"sample-{index}"
                directory.mkdir(parents=True)
                (directory / "report.json").write_bytes(b"x" * size)
            with patch(
                "test7_ocr.storage.shutil.disk_usage",
                return_value=SimpleNamespace(total=50_000, used=10_000, free=40_000),
            ):
                result = preflight_batch_storage(
                    output_root,
                    10,
                    {
                        "history_sample_limit": 10,
                        "estimate_percentile": 0.95,
                        "estimate_safety_multiplier": 1.5,
                        "minimum_free_space_bytes": 1_000,
                    },
                )
        self.assertEqual(result["estimate_method"], "historical_p95")
        self.assertEqual(result["baseline_bytes_per_sample"], 400)
        self.assertEqual(result["estimated_bytes_per_sample"], 600)
        self.assertEqual(result["estimated_new_bytes"], 6_000)
        self.assertTrue(result["sufficient"])

    def test_blocks_before_batch_when_reserve_would_be_crossed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "new-runs"
            with patch(
                "test7_ocr.storage.shutil.disk_usage",
                return_value=SimpleNamespace(total=20_000, used=10_000, free=10_000),
            ):
                result = preflight_batch_storage(
                    output_root,
                    3,
                    {
                        "fallback_bytes_per_sample": 1_000,
                        "estimate_safety_multiplier": 2,
                        "minimum_free_space_bytes": 5_000,
                    },
                )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["required_free_before_start"], 11_000)
        self.assertFalse(result["sufficient"])

    def test_preflight_only_counts_only_missing_resume_reports_and_skips_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.jsonl"
            records = [
                {"sample_id": "sample-1", "image_path": "one.png"},
                {"sample_id": "sample-2", "image_path": "two.png"},
            ]
            manifest.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            config = root / "config.json"
            config.write_text(json.dumps({"storage": {"preflight_enabled": True}}), encoding="utf-8")
            output_root = root / "runs"
            completed = output_root / "run-1" / "sample-1"
            completed.mkdir(parents=True)
            (completed / "report.json").write_text("{}", encoding="utf-8")
            ready = {"schema_version": "storage_preflight_v1", "status": "READY", "sufficient": True}
            stdout = io.StringIO()
            with (
                patch("test7_ocr.cli.preflight_batch_storage", return_value=ready) as preflight,
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
                        "--run-id", "run-1",
                        "--resume",
                        "--storage-preflight-only",
                    ]
                )
        self.assertEqual(exit_code, 0)
        self.assertEqual(preflight.call_args.args[1], 1)
        provider.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["selected_sample_count"], 2)
        self.assertTrue(payload["resume_requested"])


if __name__ == "__main__":
    unittest.main()
