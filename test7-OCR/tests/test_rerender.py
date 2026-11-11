from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image, ImageDraw

from test7_ocr.deferred_verification import verify_sample_artifact
from test7_ocr.image_ops import direct_crop, sha256_file
from test7_ocr.pipeline import load_config
from test7_ocr.rerender import rerender_run
from test7_ocr.storage import image_pixel_sha256, write_json_atomic
from test7_ocr.types import BBox, RecognitionResult


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _ReadbackProvider:
    def recognize(self, _path: Path, sample_id: str | None = None) -> RecognitionResult:
        return RecognitionResult(
            provider="fixture-readback",
            transcript="1. 请选择 A. 甲 B. 乙 C. 丙 D. 丁",
        )


class RerenderTests(unittest.TestCase):
    def _source_run(self, root: Path) -> tuple[Path, Path]:
        source_image_path = root / "source.png"
        source_image = Image.new("RGB", (400, 200), "white")
        draw = ImageDraw.Draw(source_image)
        draw.rectangle((200, 80, 300, 150), outline="black", width=4)
        source_image.save(source_image_path)
        source_sha256 = sha256_file(source_image_path)
        bbox = BBox(190, 70, 130, 100)
        crop = direct_crop(source_image, bbox)

        run = root / "source-run"
        sample = run / "sample-1"
        assets = sample / "assets"
        assets.mkdir(parents=True)
        crop.save(assets / "asset_001.png")
        old_output = Image.new("RGB", (800, 160), "white")
        old_output.save(sample / "output.png")
        document_path = sample / "question_document.json"
        report_path = sample / "report.json"
        write_json_atomic(
            document_path,
            {
                "schema_version": "question_document_v2",
                "sample_id": "sample-1",
                "source_image": str(source_image_path.resolve()),
                "source_sha256": source_sha256,
                "provider": "fixture",
                "canonical_text": "1. 请选择 A. 甲 B. 乙 C. 丙 D. 丁",
                "display_text": "1. 请选择 A. 甲 B. 乙 C. 丙 D. 丁",
                "question_number": None,
                "stem": "stale structure",
                "options": [],
                "visual_assets": [
                    {
                        "asset_id": "asset_001",
                        "source_bbox": {
                            "x": bbox.x,
                            "y": bbox.y,
                            "width": bbox.width,
                            "height": bbox.height,
                        },
                        "crop_path": "assets/asset_001.png",
                        "crop_sha256": image_pixel_sha256(crop),
                        "bbox_verified": True,
                        "edge_complete": True,
                        "extraction": "direct_source_crop",
                    }
                ],
                "handwriting_detected": False,
                "formula_verified": True,
                "formula_specialist_used": False,
                "formula_count": 0,
                "warnings": [],
                "source_blocks": [
                    {
                        "block_id": "source_0001",
                        "kind": "text",
                        "text": "1. 请选择 A. 甲 B. 乙 C. 丙 D. 丁",
                        "source_bbox": {"x": 10, "y": 10, "width": 180, "height": 30},
                        "confidence": 0.99,
                        "disposition": "emitted",
                        "reason": None,
                    }
                ],
            },
        )
        write_json_atomic(
            report_path,
            {
                "schema_version": "process_report_v1",
                "run_id": "source-run",
                "sample_id": "sample-1",
                "status": "REVIEW",
                "source_image": str(source_image_path.resolve()),
                "source_sha256": source_sha256,
                "output_image": str((sample / "output.png").resolve()),
                "output_sha256": sha256_file(sample / "output.png"),
                "question_document": str(document_path.resolve()),
                "elapsed_seconds": 1.0,
                "issues": [],
            },
        )
        return run, source_image_path

    def test_rerender_recrops_assets_and_is_resume_safe(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_run, _ = self._source_run(root)
            output_root = root / "runs"
            config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
            config["rendering"]["engine"] = "pillow"
            summary = rerender_run(source_run, output_root, "layout-run", config)
            report_path = output_root / "layout-run" / "sample-1" / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            document = json.loads(
                Path(report["question_document"]).read_text(encoding="utf-8")
            )
            with Image.open(report["output_image"]) as output:
                self.assertEqual(output.width, 800)
            self.assertEqual(report["measurement"]["mode"], "render_only")
            self.assertFalse(report["measurement"]["online_latency_sla_applicable"])
            self.assertEqual(report["rerender"]["assets_recropped_from_current_source"], 1)
            self.assertTrue(report["rerender"]["structure_reparsed_from_display_text"])
            self.assertEqual(document["question_number"], "1")
            self.assertEqual(
                [option["label"] for option in document["options"]], list("ABCD")
            )
            self.assertEqual(summary["resource_guard"]["status"], "NOT_APPLICABLE_RENDER_ONLY")
            self.assertEqual(summary["readback_reused_from_identical_output"], 0)
            self.assertFalse(
                any(
                    warning.startswith("RENDER_READBACK")
                    for warning in document["warnings"]
                )
            )

            report["render_readback"] = {
                "enabled": True,
                "mode": "deferred_release_gate",
                "passed": True,
                "issues": [],
                "verified_at": "2026-01-01T00:00:00+00:00",
                "verification_elapsed_seconds": 0.5,
            }
            report["post_verification"] = {
                "mode": "deferred_release_gate",
                "completed_at": "2026-01-01T00:00:00+00:00",
                "elapsed_seconds": 0.5,
                "excluded_from_online_latency_sla": True,
            }
            write_json_atomic(report_path, report)
            copied = rerender_run(
                output_root / "layout-run",
                output_root,
                "layout-run-identical-copy",
                config,
            )
            copied_report = json.loads(
                (
                    output_root
                    / "layout-run-identical-copy"
                    / "sample-1"
                    / "report.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(copied["readback_reused_from_identical_output"], 1)
            self.assertTrue(copied_report["render_readback"]["passed"])
            self.assertEqual(
                copied_report["render_readback"][
                    "evidence_reused_from_identical_output"
                ]["output_sha256"],
                copied_report["output_sha256"],
            )
            self.assertTrue(
                copied_report["rerender"]["readback_reused_from_identical_output"]
            )

            resumed = rerender_run(
                source_run,
                output_root,
                "layout-run",
                config,
                resume=True,
            )
            self.assertEqual(resumed["resumed_from_existing_reports"], 1)
            self.assertEqual(resumed["processed_this_invocation"], 0)

            report["elapsed_seconds"] = 10.0
            write_json_atomic(report_path, report)
            verified = verify_sample_artifact(report_path, config, _ReadbackProvider())
            self.assertNotIn("LATENCY_OVER_TARGET", [item["code"] for item in verified["issues"]])

    def test_rerender_can_select_explicit_sample_ids(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_run, _ = self._source_run(root)
            output_root = root / "runs"
            config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
            config["rendering"]["engine"] = "pillow"

            summary = rerender_run(
                source_run,
                output_root,
                "selected-run",
                config,
                sample_ids={"sample-1"},
            )
            self.assertEqual(summary["attempted"], 1)
            self.assertEqual(summary["requested_sample_ids"], ["sample-1"])

            with self.assertRaisesRegex(ValueError, "missing-sample"):
                rerender_run(
                    source_run,
                    output_root,
                    "missing-run",
                    config,
                    sample_ids={"missing-sample"},
                )


if __name__ == "__main__":
    unittest.main()
