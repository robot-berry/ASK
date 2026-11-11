from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from test7_ocr.product_inference import ProductOcrService
from test7_ocr.types import ProcessReport, ValidationIssue


def image_payload(*, image_format: str = "PNG") -> dict[str, str]:
    buffer = io.BytesIO()
    Image.new("RGB", (48, 32), "white").save(buffer, format=image_format)
    return {
        "filename": f"question.{image_format.lower()}",
        "image_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


class FakePipeline:
    def __init__(
        self,
        output_root: Path,
        status: str = "PASS",
        *,
        readback_pending: bool = False,
    ) -> None:
        self.output_root = output_root
        self.status = status
        self.readback_pending = readback_pending
        self.calls: list[tuple[Path, str]] = []
        self.closed = False

    def process(
        self,
        image_path: Path,
        sample_id: str,
        *,
        recognition_sample_id: str | None = None,
    ) -> ProcessReport:
        self.calls.append((image_path, sample_id))
        sample_root = self.output_root / "live" / sample_id
        sample_root.mkdir(parents=True)
        output_path = sample_root / "output.png"
        Image.new("RGB", (800, 120), "white").save(output_path)
        source_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
        output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        issues = (
            []
            if self.status == "PASS"
            else [ValidationIssue("FORMULA_UNVERIFIED", "review", "公式需要复核")]
        )
        report = ProcessReport(
            schema_version="process_report_v1",
            run_id="live",
            sample_id=sample_id,
            status=self.status,
            source_image=str(image_path),
            output_image=str(output_path),
            question_document=str(sample_root / "question_document.json"),
            provider="fake-structured",
            elapsed_seconds=0.25,
            timings_ms={"recognition": 100.0},
            issues=issues,
            source_sha256=source_sha256,
            output_sha256=output_sha256,
            output_width=800,
            output_height=120,
            input_source_image=str(image_path),
            input_source_sha256=source_sha256,
        )
        persisted = report.to_dict()
        persisted["render_readback"] = {
            "enabled": True,
            "passed": self.status == "PASS" and not self.readback_pending,
            "pending": self.readback_pending,
            "mode": (
                "deferred_release_gate_pending"
                if self.readback_pending
                else "online"
            ),
        }
        (sample_root / "report.json").write_text(
            json.dumps(persisted), encoding="utf-8"
        )
        return report

    def close(self) -> None:
        self.closed = True


class ProductOcrServiceTests(unittest.TestCase):
    def test_product_pipeline_enables_bounded_parallel_verification(self) -> None:
        config = {"recognition": {}, "render_verification": {}}
        fake_provider = Mock()
        fake_pipeline = Mock()
        with (
            patch(
                "test7_ocr.product_inference.load_config",
                return_value=config,
            ),
            patch(
                "test7_ocr.product_inference._provider_from_args",
                return_value=fake_provider,
            ),
            patch(
                "test7_ocr.product_inference._clean_source_resolver_from_config",
                return_value=None,
            ),
            patch(
                "test7_ocr.product_inference.QuestionPipeline",
                return_value=fake_pipeline,
            ) as pipeline_class,
        ):
            service = ProductOcrService(config_path=Path("pipeline.json"))
            self.assertIs(service._build_pipeline(), fake_pipeline)

        passed_config = pipeline_class.call_args.args[1]
        self.assertTrue(
            passed_config["recognition"][
                "defer_photo_visual_handwriting_check_when_review_locked"
            ]
        )
        verification = passed_config["render_verification"]
        self.assertTrue(verification["parallel_source_dual_ocr"])
        self.assertEqual(verification["source_dual_ocr_timeout_seconds"], 3.0)
        self.assertFalse(verification["online_render_readback_enabled"])
        self.assertEqual(verification["render_readback_max_dimension_px"], 640)
        self.assertNotIn("source_dual_ocr_max_dimension_px", verification)

    def test_real_product_upload_is_isolated_and_downloadable_only_after_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "product-runs"
            fake = FakePipeline(output_root)
            service = ProductOcrService(
                input_root=root / "product-inputs",
                output_root=output_root,
                feedback_root=root / "feedback",
                pipeline_factory=lambda: fake,
            )

            result = service.process_upload(image_payload())

            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["safe_to_download"])
            self.assertTrue(result["download_url"].endswith("?download=1"))
            self.assertEqual(result["output"]["width"], 800)
            self.assertFalse(result["storage"]["gold_or_benchmark_modified"])
            self.assertEqual(len(list((root / "product-inputs").iterdir())), 1)
            self.assertEqual(service.output_path(result["sample_id"]).name, "output.png")
            self.assertEqual(
                service.verified_download_path(result["sample_id"]).name,
                "output.png",
            )
            self.assertEqual(
                service.result_details(result["sample_id"])["output"]["width"], 800
            )
            source_token = result["source_url"].rsplit("/", 1)[1]
            self.assertTrue(service.source_path(source_token).is_file())
            service.close()
            self.assertTrue(fake.closed)

    def test_review_result_cannot_be_downloaded_as_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "product-runs"
            service = ProductOcrService(
                input_root=root / "product-inputs",
                output_root=output_root,
                feedback_root=root / "feedback",
                pipeline_factory=lambda: FakePipeline(output_root, status="REVIEW"),
            )

            result = service.process_upload(image_payload())

            self.assertFalse(result["safe_to_download"])
            self.assertIsNone(result["download_url"])
            self.assertEqual(result["issues"][0]["code"], "FORMULA_UNVERIFIED")
            with self.assertRaisesRegex(RuntimeError, "禁止下载"):
                service.verified_download_path(result["sample_id"])

    def test_pending_final_png_check_completes_in_background_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "product-runs"
            fake = FakePipeline(
                output_root,
                status="REVIEW",
                readback_pending=True,
            )
            provider = Mock()

            def verified_report(report_path, _config, received_provider):
                self.assertEqual(received_provider, provider)
                persisted = json.loads(report_path.read_text(encoding="utf-8"))
                persisted["status"] = "PASS"
                persisted["issues"] = []
                persisted["render_readback"] = {
                    "enabled": True,
                    "passed": True,
                    "mode": "deferred_release_gate",
                    "verified_at": "2026-09-11T12:00:00+00:00",
                    "verification_elapsed_seconds": 0.4,
                }
                return persisted

            with (
                patch(
                    "test7_ocr.product_inference.TimedRapidOcrProvider",
                    return_value=provider,
                ),
                patch(
                    "test7_ocr.product_inference.verify_sample_artifact",
                    side_effect=verified_report,
                ),
            ):
                service = ProductOcrService(
                    input_root=root / "product-inputs",
                    output_root=output_root,
                    feedback_root=root / "feedback",
                    pipeline_factory=lambda: fake,
                )
                result = service.process_upload(image_payload())
                self.assertEqual(result["verification"]["state"], "pending")
                self.assertFalse(result["safe_to_download"])

                service._verification_jobs[result["sample_id"]].result(timeout=5)
                completed = service.result_details(result["sample_id"])
                self.assertEqual(completed["verification"]["state"], "complete")
                self.assertEqual(completed["status"], "PASS")
                self.assertTrue(completed["safe_to_download"])
                self.assertIsNotNone(completed["download_url"])
                service.close()
                provider.close.assert_called_once_with()

    def test_source_is_content_addressed_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "product-runs"
            fake = FakePipeline(output_root)
            service = ProductOcrService(
                input_root=root / "product-inputs",
                output_root=output_root,
                feedback_root=root / "feedback",
                pipeline_factory=lambda: fake,
            )
            payload = image_payload(image_format="JPEG")

            first = service.process_upload(payload)
            second = service.process_upload(payload)

            self.assertEqual(first["input"]["sha256"], second["input"]["sha256"])
            self.assertNotEqual(first["sample_id"], second["sample_id"])
            self.assertEqual(len(list((root / "product-inputs").iterdir())), 1)

    def test_invalid_or_oversized_input_is_rejected_before_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = ProductOcrService(
                input_root=Path(temporary) / "inputs",
                output_root=Path(temporary) / "outputs",
                feedback_root=Path(temporary) / "feedback",
                pipeline_factory=lambda: self.fail("pipeline must not be constructed"),
            )
            with self.assertRaisesRegex(ValueError, "Base64"):
                service.process_upload({"image_base64": "not base64"})
            with self.assertRaisesRegex(ValueError, "20MB"):
                service.process_upload({"image_base64": "A" * 30_000_000})

    def test_path_tokens_reject_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            service = ProductOcrService(
                input_root=Path(temporary) / "inputs",
                output_root=Path(temporary) / "outputs",
                feedback_root=Path(temporary) / "feedback",
                pipeline_factory=lambda: self.fail("pipeline must not be constructed"),
            )
            with self.assertRaisesRegex(ValueError, "标识"):
                service.source_path("../secret.png")
            with self.assertRaisesRegex(ValueError, "标识"):
                service.output_path("../secret")

    def test_problem_feedback_is_hash_bound_and_enters_regression_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "product-runs"
            service = ProductOcrService(
                input_root=root / "product-inputs",
                output_root=output_root,
                feedback_root=root / "feedback",
                pipeline_factory=lambda: FakePipeline(output_root),
            )
            result = service.process_upload(image_payload())

            feedback = service.save_feedback(
                result["sample_id"],
                {
                    "issues": ["formula_error", "visual_crop_incomplete"],
                    "notes": "负号缺失，图注被裁掉",
                },
            )

            self.assertEqual(feedback["verdict"], "fail")
            self.assertTrue(feedback["regression_queue"])
            self.assertFalse(feedback["gold_modified"])
            self.assertEqual(feedback["source_sha256"], result["input"]["sha256"])
            self.assertEqual(feedback["output_sha256"], result["output"]["sha256"])
            self.assertEqual(
                service.result_details(result["sample_id"])["feedback"]["notes"],
                "负号缺失，图注被裁掉",
            )
            with self.assertRaisesRegex(RuntimeError, "禁止下载"):
                service.verified_download_path(result["sample_id"])
            self.assertEqual(service.list_feedback()[0]["sample_id"], result["sample_id"])

    def test_feedback_rejects_unknown_issue_and_changed_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "product-runs"
            service = ProductOcrService(
                input_root=root / "product-inputs",
                output_root=output_root,
                feedback_root=root / "feedback",
                pipeline_factory=lambda: FakePipeline(output_root),
            )
            result = service.process_upload(image_payload())
            with self.assertRaisesRegex(ValueError, "不支持"):
                service.save_feedback(result["sample_id"], {"issues": ["invented"]})

            Image.new("RGB", (800, 120), "black").save(
                service.output_path(result["sample_id"])
            )
            with self.assertRaisesRegex(ValueError, "成图哈希已变化"):
                service.save_feedback(result["sample_id"], {"issues": ["formula_error"]})


if __name__ == "__main__":
    unittest.main()
