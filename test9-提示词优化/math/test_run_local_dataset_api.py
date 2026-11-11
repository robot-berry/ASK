#!/usr/bin/env python3
"""Offline tests for the local API dataset runner (no HTTP/provider calls)."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import run_local_dataset_api as runner


def png_bytes(width: int = 800, height: int = 120) -> bytes:
    output = BytesIO()
    Image.new("L", (width, height), 255).save(
        output,
        format="PNG",
        dpi=(800 / (9 / 2.54), 800 / (9 / 2.54)),
    )
    return output.getvalue()


class NeverNetwork:
    def request_json(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("network method must not be called")

    def download_png(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("network method must not be called")


class CompletedFakeClient:
    def __init__(self, result: bytes):
        self.result = result
        self.job_id = "offline_fake_job_1"
        self.create_calls = 0
        self.poll_calls = 0
        self.download_calls = 0

    @staticmethod
    def capabilities() -> dict:
        return {
            "api_version": "v1",
            "subjects": [{"id": "physics"}, {"id": "math"}],
            "upload": {
                "raw_endpoint_suffix": "/raw",
                "idempotency": {"header": "Idempotency-Key"},
            },
            "workflow": "asynchronous",
            "results": [
                {"sequence": sequence, "kind": kind, "filename": filename}
                for sequence, kind, filename in runner.EXPECTED_IMAGES
            ],
        }

    def envelope(self, status: str) -> dict:
        return {
            "api_version": "v1",
            "subject": "math",
            "job_id": self.job_id,
            "status": status,
            "poll_url": runner.expected_poll_path(self.job_id),
            "retry_after_seconds": 1,
            "elapsed_seconds": 12.5,
            "result_contract": {"name": "test6-six-png", "version": "1", "complete": status == "completed"},
            "images": [],
        }

    def request_json(self, path: str, **kwargs):  # type: ignore[no-untyped-def]
        if path == "/api/v1/capabilities":
            return self.capabilities(), {}
        if path == "/api/v1/math/jobs/raw":
            self.create_calls += 1
            self.last_idempotency_key = kwargs.get("idempotency_key")
            payload = self.envelope("queued")
            return payload, {
                "location": payload["poll_url"],
                "retry-after": "1",
                "idempotency-replayed": "false",
            }
        if path == runner.expected_poll_path(self.job_id):
            self.poll_calls += 1
            payload = self.envelope("completed")
            payload["images"] = [
                {
                    "sequence": sequence,
                    "kind": kind,
                    "filename": filename,
                    "media_type": "image/png",
                    "url": runner.expected_image_path(self.job_id, filename),
                    "width": 800,
                    "height": 120,
                    "mode": "L",
                    "bytes": len(self.result),
                    "sha256": runner.sha256_bytes(self.result),
                    "display_width_cm": 9.0,
                }
                for sequence, kind, filename in runner.EXPECTED_IMAGES
            ]
            return payload, {}
        raise AssertionError(f"unexpected fake request: {path}")

    def download_png(self, path: str, expected_bytes: int):
        self.download_calls += 1
        self.asserted_expected_bytes = expected_bytes
        return self.result, "image/png"


class DatasetRunnerOfflineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "question.png"
        Image.new("RGB", (64, 48), "white").save(self.source, format="PNG")
        self.manifest_path = self.root / "cases.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "name": "offline-test",
                    "cases": [
                        {
                            "id": "high_0401",
                            "image": "question.png",
                            "grade": "high",
                            "tags": ["trig"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loopback_only(self) -> None:
        self.assertEqual(
            runner.canonical_loopback_base("http://127.0.0.1:8130"),
            "http://127.0.0.1:8130/",
        )
        self.assertEqual(
            runner.canonical_loopback_base("http://[::1]:8130/"),
            "http://[::1]:8130/",
        )
        with self.assertRaises(runner.RunnerError):
            runner.canonical_loopback_base("https://example.com/")
        with self.assertRaises(runner.RunnerError):
            runner.canonical_loopback_base("http://127.0.0.1:8130/api/")

    def test_deterministic_key_and_dry_plan(self) -> None:
        manifest = runner.load_manifest(self.manifest_path)
        key1 = runner.deterministic_idempotency_key(manifest["cases"][0], 1)
        key2 = runner.deterministic_idempotency_key(manifest["cases"][0], 1)
        key3 = runner.deterministic_idempotency_key(manifest["cases"][0], 2)
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)
        self.assertRegex(key1, r"^[A-Za-z0-9][A-Za-z0-9._~-]{7,127}$")
        target = self.root / "must-not-exist"
        plan = runner.dry_run_report(manifest, "http://127.0.0.1:8130/", target)
        self.assertEqual(plan["network_requests"], 0)
        self.assertFalse(target.exists())

    def test_no_allow_keeps_case_pending_without_network(self) -> None:
        manifest = runner.load_manifest(self.manifest_path)
        run_dir = self.root / "run"
        run, progress = runner.initialize_or_load_run(
            run_dir, manifest, "http://127.0.0.1:8130/"
        )
        dataset = runner.DatasetRunner(
            run_dir,
            run,
            progress,
            NeverNetwork(),  # type: ignore[arg-type]
            allow_provider_call=False,
            retry_failures=False,
            max_new_jobs=1,
            job_timeout_seconds=1,
            poll_interval_seconds=0.01,
        )
        dataset.run_all()
        self.assertEqual(progress["cases"]["high_0401"]["phase"], "pending")
        self.assertEqual(progress["cases"]["high_0401"]["attempts"], [])
        self.assertTrue((run_dir / "run.json").is_file())
        self.assertTrue((run_dir / "progress.json").is_file())
        self.assertTrue((run_dir / "summary.json").is_file())
        self.assertTrue((run_dir / "index.html").is_file())
        self.assertTrue((run_dir / "manual_review.md").is_file())

    def test_failed_case_is_not_retried_by_default(self) -> None:
        manifest = runner.load_manifest(self.manifest_path)
        run_dir = self.root / "run"
        run, progress = runner.initialize_or_load_run(
            run_dir, manifest, "http://127.0.0.1:8130/"
        )
        state = progress["cases"]["high_0401"]
        state["current_attempt"] = 1
        state["phase"] = "failed"
        state["attempts"] = [
            {
                "number": 1,
                "phase": "failed",
                "idempotency_key": runner.deterministic_idempotency_key(
                    manifest["cases"][0], 1
                ),
                "job_id": "already_failed_job",
                "remote_status": "failed",
                "metadata": [],
                "downloads": {},
            }
        ]
        runner.persist_run_artifacts(run_dir, run, progress)
        dataset = runner.DatasetRunner(
            run_dir,
            run,
            progress,
            NeverNetwork(),  # type: ignore[arg-type]
            allow_provider_call=True,
            retry_failures=False,
            max_new_jobs=1,
            job_timeout_seconds=1,
            poll_interval_seconds=0.01,
        )
        dataset.run_all()
        self.assertEqual(state["phase"], "failed")
        self.assertEqual(len(state["attempts"]), 1)
        self.assertEqual(dataset.new_job_requests, 0)

    def test_contract_failed_case_is_not_retried_by_default(self) -> None:
        manifest = runner.load_manifest(self.manifest_path)
        run_dir = self.root / "contract-run"
        run, progress = runner.initialize_or_load_run(
            run_dir, manifest, "http://127.0.0.1:8130/"
        )
        state = progress["cases"]["high_0401"]
        state["current_attempt"] = 1
        state["phase"] = "contract_failed"
        state["attempts"] = [
            {
                "number": 1,
                "phase": "contract_failed",
                "idempotency_key": runner.deterministic_idempotency_key(
                    manifest["cases"][0], 1
                ),
                "job_id": "contract_failed_job",
                "remote_status": "completed",
                "metadata": [],
                "downloads": {},
            }
        ]
        runner.persist_run_artifacts(run_dir, run, progress)
        dataset = runner.DatasetRunner(
            run_dir,
            run,
            progress,
            NeverNetwork(),  # type: ignore[arg-type]
            allow_provider_call=True,
            retry_failures=False,
            max_new_jobs=1,
            job_timeout_seconds=1,
            poll_interval_seconds=0.01,
        )
        dataset.run_all()
        self.assertEqual(state["phase"], "contract_failed")
        self.assertEqual(len(state["attempts"]), 1)
        self.assertEqual(dataset.new_job_requests, 0)

    def test_retry_failures_creates_explicit_second_attempt(self) -> None:
        manifest = runner.load_manifest(self.manifest_path)
        run_dir = self.root / "run"
        run, progress = runner.initialize_or_load_run(
            run_dir, manifest, "http://127.0.0.1:8130/"
        )
        state = progress["cases"]["high_0401"]
        state["current_attempt"] = 1
        state["phase"] = "failed"
        state["attempts"] = [
            {
                "number": 1,
                "phase": "failed",
                "idempotency_key": runner.deterministic_idempotency_key(
                    manifest["cases"][0], 1
                ),
                "job_id": "already_failed_job",
                "remote_status": "failed",
                "metadata": [],
                "downloads": {},
            }
        ]
        runner.persist_run_artifacts(run_dir, run, progress)
        fake = CompletedFakeClient(png_bytes())
        dataset = runner.DatasetRunner(
            run_dir,
            run,
            progress,
            fake,  # type: ignore[arg-type]
            allow_provider_call=True,
            retry_failures=True,
            max_new_jobs=1,
            job_timeout_seconds=1,
            poll_interval_seconds=0.01,
        )
        dataset.run_all()
        self.assertEqual(state["phase"], "verified")
        self.assertEqual(len(state["attempts"]), 2)
        self.assertEqual(state["current_attempt"], 2)
        self.assertEqual(
            fake.last_idempotency_key,
            runner.deterministic_idempotency_key(manifest["cases"][0], 2),
        )

    def test_fake_completed_job_is_downloaded_and_offline_resumable(self) -> None:
        manifest = runner.load_manifest(self.manifest_path)
        run_dir = self.root / "run"
        run, progress = runner.initialize_or_load_run(
            run_dir, manifest, "http://127.0.0.1:8130/"
        )
        fake = CompletedFakeClient(png_bytes())
        dataset = runner.DatasetRunner(
            run_dir,
            run,
            progress,
            fake,  # type: ignore[arg-type]
            allow_provider_call=True,
            retry_failures=False,
            max_new_jobs=1,
            job_timeout_seconds=1,
            poll_interval_seconds=0.01,
        )
        dataset.run_all()
        state = progress["cases"]["high_0401"]
        self.assertEqual(state["phase"], "verified")
        self.assertEqual(fake.create_calls, 1)
        self.assertEqual(fake.poll_calls, 1)
        self.assertEqual(fake.download_calls, len(runner.EXPECTED_IMAGES))
        self.assertEqual(
            fake.last_idempotency_key,
            runner.deterministic_idempotency_key(manifest["cases"][0], 1),
        )
        self.assertTrue(runner.build_summary(run, progress)["all_verified"])

        # A second run verifies saved hashes/DPI and makes no network call.
        loaded_run, loaded_progress = runner.initialize_or_load_run(
            run_dir, manifest, "http://127.0.0.1:8130/"
        )
        resumed = runner.DatasetRunner(
            run_dir,
            loaded_run,
            loaded_progress,
            NeverNetwork(),  # type: ignore[arg-type]
            allow_provider_call=False,
            retry_failures=False,
            max_new_jobs=1,
            job_timeout_seconds=1,
            poll_interval_seconds=0.01,
        )
        resumed.run_all()
        self.assertEqual(loaded_progress["cases"]["high_0401"]["phase"], "verified")

        # Recovery metadata is untrusted: duplicates/traversal are rejected
        # before a filename can be joined to the run directory.
        resumed_state = loaded_progress["cases"]["high_0401"]
        resumed_attempt = runner.current_attempt(resumed_state)
        self.assertIsNotNone(resumed_attempt)
        with self.assertRaises(runner.RunnerError):
            runner.validate_saved_metadata(resumed_attempt["metadata"], "../bad-job")  # type: ignore[index]
        nonfinite = [dict(item) for item in resumed_attempt["metadata"]]  # type: ignore[index]
        nonfinite[0]["display_width_cm"] = float("nan")
        with self.assertRaises(runner.RunnerError):
            runner.validate_saved_metadata(nonfinite, resumed_attempt["job_id"])  # type: ignore[index]
        first = dict(resumed_attempt["metadata"][0])  # type: ignore[index]
        first["filename"] = "../escape.png"
        resumed_attempt["metadata"] = [
            dict(first) for _ in runner.EXPECTED_IMAGES
        ]  # type: ignore[index]
        resumed_state["phase"] = "verified"
        resumed.verify_saved_completed_cases()
        self.assertEqual(resumed_state["phase"], "completed")
        self.assertEqual(resumed_attempt["metadata"], [])  # type: ignore[index]
        self.assertFalse((run_dir / "cases" / "escape.png").exists())

    def test_saved_create_intent_replays_same_key(self) -> None:
        manifest = runner.load_manifest(self.manifest_path)
        run_dir = self.root / "intent-run"
        run, progress = runner.initialize_or_load_run(
            run_dir, manifest, "http://127.0.0.1:8130/"
        )
        state = progress["cases"]["high_0401"]
        preparing = runner.DatasetRunner(
            run_dir,
            run,
            progress,
            NeverNetwork(),  # type: ignore[arg-type]
            allow_provider_call=True,
            retry_failures=False,
            max_new_jobs=1,
            job_timeout_seconds=1,
            poll_interval_seconds=0.01,
        )
        intent = preparing.new_attempt(manifest["cases"][0], state)
        expected_key = intent["idempotency_key"]

        fake = CompletedFakeClient(png_bytes())
        resumed = runner.DatasetRunner(
            run_dir,
            run,
            progress,
            fake,  # type: ignore[arg-type]
            allow_provider_call=True,
            retry_failures=False,
            max_new_jobs=1,
            job_timeout_seconds=1,
            poll_interval_seconds=0.01,
        )
        resumed.run_all()
        self.assertEqual(state["phase"], "verified")
        self.assertEqual(fake.create_calls, 1)
        self.assertEqual(fake.last_idempotency_key, expected_key)

    def test_nonfinite_cli_timeouts_are_rejected(self) -> None:
        for option, value in (
            ("--job-timeout", "nan"),
            ("--request-timeout", "inf"),
            ("--poll-interval", "-inf"),
        ):
            with self.subTest(option=option, value=value):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        runner.main(
                            [
                                str(self.manifest_path),
                                "--dry-run",
                                option,
                                value,
                            ]
                        )
                self.assertEqual(caught.exception.code, 2)

    def test_result_dpi_is_part_of_contract(self) -> None:
        raw = png_bytes()
        metadata = {
            "filename": "01_question_knowledge.png",
            "sha256": runner.sha256_bytes(raw),
            "bytes": len(raw),
            "height": 120,
        }
        checked = runner.inspect_result_png(raw, metadata)
        self.assertAlmostEqual(checked["physical_width_cm"], 9.0, places=2)

        wrong = BytesIO()
        Image.new("L", (800, 120), 255).save(wrong, format="PNG", dpi=(96, 96))
        wrong_raw = wrong.getvalue()
        metadata["sha256"] = runner.sha256_bytes(wrong_raw)
        metadata["bytes"] = len(wrong_raw)
        with self.assertRaises(runner.RunnerError):
            runner.inspect_result_png(wrong_raw, metadata)


if __name__ == "__main__":
    unittest.main()
