import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from test7_ocr.deferred_verification import verify_run_artifacts, verify_sample_artifact
from test7_ocr.image_ops import sha256_file
from test7_ocr.storage import write_json_atomic
from test7_ocr.types import RecognitionResult


class FakeReadbackProvider:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def recognize(self, image_path: Path, sample_id: str | None = None) -> RecognitionResult:
        return RecognitionResult(provider="fake-readback", transcript=self.transcript)


class SequentialReadbackProvider:
    def __init__(self, transcripts: list[str]) -> None:
        self.transcripts = transcripts
        self.calls = 0

    def recognize(self, image_path: Path, sample_id: str | None = None) -> RecognitionResult:
        transcript = self.transcripts[min(self.calls, len(self.transcripts) - 1)]
        self.calls += 1
        return RecognitionResult(provider="fake-readback", transcript=transcript)


def _config() -> dict:
    return {
        "render_verification": {
            "minimum_text_coverage_ratio": 0.72,
            "deferred_option_retry_scale": 1.5,
            "deferred_option_grayscale_retry": True,
        },
        "quality": {
            "target_latency_seconds": 5.0,
            "minimum_bottom_padding_px": 10,
            "minimum_text_characters": 4,
            "latex_leakage_is_reject": True,
            "handwriting_present_is_reject": True,
            "unverified_prelabel_is_review": True,
            # This fixture intentionally uses only A/B to isolate deferred
            # render-readback behavior from the production four-option gate.
            "partial_option_set_is_reject": False,
        },
        "output": {"width_px": 800},
        "visual_assets": {"visual_cues": []},
    }


class DeferredVerificationTests(unittest.TestCase):
    def _artifact(self, root: Path, *, formula_verified: bool = True) -> Path:
        sample = root / "S1"
        sample.mkdir(parents=True)
        source = root / "source.png"
        output = sample / "output.png"
        Image.new("RGB", (800, 160), "white").save(source)
        Image.new("RGB", (800, 160), "white").save(output)
        document = {
            "schema_version": "question_document_v2",
            "sample_id": "S1",
            "source_image": str(source),
            "source_sha256": sha256_file(source),
            "provider": "fake",
            "canonical_text": "1. 已知条件，求正确答案。\nA. 甲\nB. 乙",
            "display_text": "1. 已知条件，求正确答案。\nA. 甲\nB. 乙",
            "question_number": "1",
            "stem": "已知条件，求正确答案。",
            "options": [{"label": "A", "text": "甲"}, {"label": "B", "text": "乙"}],
            "visual_assets": [],
            "handwriting_detected": False,
            "formula_verified": formula_verified,
            "formula_specialist_used": False,
            "formula_count": 0,
            "warnings": ["RENDER_READBACK_FAILED:TimeoutError: old timeout"],
            "source_blocks": [
                {
                    "block_id": "source_0001",
                    "kind": "text",
                    "text": "1. 已知条件，求正确答案。 A. 甲 B. 乙",
                    "source_bbox": {"x": 10, "y": 10, "width": 600, "height": 100},
                    "confidence": 0.99,
                    "disposition": "emitted",
                    "reason": None,
                }
            ],
        }
        document_path = sample / "question_document.json"
        write_json_atomic(document_path, document)
        report = {
            "schema_version": "process_report_v1",
            "run_id": "run",
            "sample_id": "S1",
            "status": "REVIEW",
            "source_image": str(source),
            "output_image": str(output),
            "question_document": str(document_path),
            "provider": "fake",
            "elapsed_seconds": 4.0,
            "timings_ms": {},
            "issues": [
                {"code": "RENDER_READBACK_FAILED", "severity": "review", "message": "old"}
            ],
            "source_sha256": sha256_file(source),
            "output_sha256": sha256_file(output),
            "output_width": 800,
            "output_height": 160,
            "render_readback": {"enabled": True, "passed": False},
        }
        report_path = sample / "report.json"
        write_json_atomic(report_path, report)
        return report_path

    def test_successful_deferred_readback_can_clear_online_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = self._artifact(Path(temporary))
            result = verify_sample_artifact(
                report_path,
                _config(),
                FakeReadbackProvider("1. 已知条件，求正确答案。\nA. 甲\nB. 乙"),
            )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["render_readback"]["passed"])
            self.assertTrue(result["post_verification"]["excluded_from_online_latency_sla"])
            saved_document = json.loads(
                (report_path.parent / "question_document.json").read_text(encoding="utf-8")
            )
            self.assertFalse(
                any(value.startswith("RENDER_READBACK_FAILED") for value in saved_document["warnings"])
            )

    def test_deferred_revalidation_preserves_verified_glyph_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = self._artifact(Path(temporary))
            document_path = report_path.parent / "question_document.json"
            document = json.loads(document_path.read_text(encoding="utf-8"))
            document["warnings"] = [
                "SOURCE_DUAL_OCR_MISMATCH_REVIEW",
                "SOURCE_TINY_SENTENCE_STOP_RECOVERED_REVIEW:1",
                "RENDER_READBACK_PENDING",
            ]
            write_json_atomic(document_path, document)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["issues"] = [
                {
                    "code": "RENDER_READBACK_PENDING",
                    "severity": "review",
                    "message": "pending",
                }
            ]
            report["recognition"] = {
                "diagnostics": {
                    "source_glyph_recovery_verified_categories": [
                        "SOURCE_TINY_SENTENCE_STOP_RECOVERED_REVIEW"
                    ],
                    "source_dual_ocr_mismatch_explained_by_verified_glyph_repair": True,
                }
            }
            write_json_atomic(report_path, report)

            result = verify_sample_artifact(
                report_path,
                _config(),
                FakeReadbackProvider("1. 已知条件，求正确答案。\nA. 甲\nB. 乙"),
            )

            self.assertEqual(result["status"], "PASS")
            self.assertNotIn(
                "SOURCE_GLYPH_RECOVERY_UNVERIFIED",
                [issue["code"] for issue in result["issues"]],
            )
            self.assertNotIn(
                "SOURCE_TEXT_DUAL_OCR_UNVERIFIED",
                [issue["code"] for issue in result["issues"]],
            )

    def test_run_writes_finalized_index_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            self._artifact(run_directory)
            summary = verify_run_artifacts(
                run_directory,
                _config(),
                provider=FakeReadbackProvider("1. 已知条件，求正确答案。\nA. 甲\nB. 乙"),
            )
            self.assertEqual(summary["processed"], 1)
            self.assertEqual(summary["readback_results"], {"passed": 1})
            self.assertEqual(summary["aggregate_readback_states"], {"passed": 1})
            self.assertTrue((run_directory / "index.finalized.jsonl").is_file())
            self.assertTrue((run_directory / "deferred_readback_summary.json").is_file())

    def test_hard_reject_does_not_keep_an_unprocessable_pending_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            report_path = self._artifact(run_directory)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["status"] = "REJECT"
            report["issues"] = [
                {
                    "code": "VISUAL_ASSET_MISSING",
                    "severity": "reject",
                    "message": "missing visual",
                },
                {
                    "code": "RENDER_READBACK_PENDING",
                    "severity": "review",
                    "message": "pending",
                },
            ]
            report["render_readback"] = {
                "enabled": True,
                "passed": False,
                "pending": True,
                "issues": ["RENDER_READBACK_PENDING"],
            }
            write_json_atomic(report_path, report)
            document_path = report_path.parent / "question_document.json"
            document = json.loads(document_path.read_text(encoding="utf-8"))
            document["warnings"] = ["RENDER_READBACK_PENDING"]
            write_json_atomic(document_path, document)
            provider = Mock()

            summary = verify_run_artifacts(run_directory, _config(), provider=provider)
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            saved_document = json.loads(document_path.read_text(encoding="utf-8"))

        provider.recognize.assert_not_called()
        self.assertEqual(summary["processed"], 0)
        self.assertEqual(summary["skipped"], {"hard_reject": 1})
        self.assertEqual(summary["normalized_terminal_readback_count"], 1)
        self.assertEqual(
            summary["aggregate_readback_states"],
            {"skipped_hard_reject_before_readback": 1},
        )
        self.assertEqual(
            saved_report["render_readback"]["reason"],
            "HARD_REJECT_BEFORE_READBACK",
        )
        self.assertNotIn(
            "RENDER_READBACK_PENDING",
            [issue["code"] for issue in saved_report["issues"]],
        )
        self.assertNotIn("RENDER_READBACK_PENDING", saved_document["warnings"])

    def test_owned_worker_uses_bounded_recycle_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            self._artifact(run_directory)
            config = _config()
            config["render_verification"].update(
                {
                    "deferred_readback_worker_recycle_after_requests": 3,
                    "deferred_readback_worker_recycle_after_timeout": True,
                }
            )
            worker = Mock()
            with (
                patch(
                    "test7_ocr.deferred_verification.TimedRapidOcrProvider",
                    return_value=worker,
                ) as provider_class,
                patch(
                    "test7_ocr.deferred_verification.verify_sample_artifact",
                    return_value={"status": "PASS", "render_readback": {"passed": True}},
                ),
            ):
                verify_run_artifacts(run_directory, config)

            kwargs = provider_class.call_args.kwargs
            self.assertEqual(kwargs["recycle_after_requests"], 3)
            self.assertTrue(kwargs["recycle_after_timeout"])
            worker.close.assert_called_once_with()

    def test_missing_option_label_gets_one_enlarged_deferred_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = self._artifact(Path(temporary))
            provider = SequentialReadbackProvider(
                [
                    "1. 已知条件，求正确答案。 A. 甲",
                    "1. 已知条件，求正确答案。 A. 甲 B. 乙",
                ]
            )
            result = verify_sample_artifact(report_path, _config(), provider)
        readback = result["render_readback"]
        self.assertEqual(provider.calls, 2)
        self.assertTrue(readback["passed"])
        self.assertTrue(readback["retry_used"])
        self.assertEqual([attempt["scale"] for attempt in readback["attempts"]], [1.0, 1.5])
        self.assertFalse(readback["attempts"][0]["passed"])
        self.assertTrue(readback["attempts"][1]["passed"])

    def test_low_coverage_gets_enlarged_retry_without_lowering_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = self._artifact(Path(temporary))
            provider = SequentialReadbackProvider(
                [
                    "A. 甲 B. 乙",
                    "1. 已知条件，求正确答案。 A. 甲 B. 乙",
                ]
            )
            result = verify_sample_artifact(report_path, _config(), provider)

        readback = result["render_readback"]
        self.assertEqual(provider.calls, 2)
        self.assertTrue(readback["passed"])
        self.assertEqual(readback["minimum_coverage_ratio"], 0.72)
        self.assertIn(
            "RENDER_READBACK_LOW_COVERAGE",
            readback["attempts"][0]["issues"],
        )
        self.assertTrue(readback["attempts"][1]["passed"])

    def test_grayscale_retry_can_confirm_labels_missed_in_color_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = self._artifact(Path(temporary))
            provider = SequentialReadbackProvider(
                [
                    "1. 已知条件，求正确答案。 A. 甲",
                    "1. 已知条件，求正确答案。 A. 甲",
                    "1. 已知条件，求正确答案。 A. 甲 B. 乙",
                ]
            )
            result = verify_sample_artifact(report_path, _config(), provider)
        readback = result["render_readback"]
        self.assertEqual(provider.calls, 3)
        self.assertTrue(readback["passed"])
        self.assertEqual(
            [(attempt["scale"], attempt["variant"]) for attempt in readback["attempts"]],
            [(1.0, "color"), (1.5, "color"), (1.0, "grayscale")],
        )
        self.assertTrue(readback["attempts"][2]["passed"])

    def test_run_rechecks_a_reject_caused_only_by_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            report_path = self._artifact(run_directory)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["status"] = "REJECT"
            report["issues"] = [
                {
                    "code": "RENDER_READBACK_OPTION_LOSS",
                    "severity": "reject",
                    "message": "old readback miss",
                }
            ]
            write_json_atomic(report_path, report)
            summary = verify_run_artifacts(
                run_directory,
                _config(),
                provider=FakeReadbackProvider(
                    "1. 已知条件，求正确答案。 A. 甲 B. 乙"
                ),
            )
        self.assertEqual(summary["processed"], 1)
        self.assertEqual(summary["all_statuses_after_verification"], {"PASS": 1})

    def test_hash_mismatch_is_not_silently_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report_path = self._artifact(Path(temporary))
            Image.new("RGB", (800, 160), "black").save(report_path.parent / "output.png")
            with self.assertRaisesRegex(ValueError, "Output PNG hash"):
                verify_sample_artifact(report_path, _config(), FakeReadbackProvider("anything"))


if __name__ == "__main__":
    unittest.main()
