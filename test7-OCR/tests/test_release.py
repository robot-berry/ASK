from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from test7_ocr.image_ops import direct_crop, sha256_file
from test7_ocr.output_quality import OUTPUT_ISSUE_KEYS
from test7_ocr.release import NoPublishableSamplesError, _validate_pass_sample, publish_release
from test7_ocr.storage import image_pixel_sha256
from test7_ocr.types import BBox


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class ReleaseTests(unittest.TestCase):
    def test_only_verified_pass_artifacts_are_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "r1"
            sample = run / "sample-1"
            assets = sample / "assets"
            assets.mkdir(parents=True)

            source_path = root / "source.png"
            source = Image.new("RGB", (100, 80), "white")
            ImageDraw.Draw(source).rectangle((20, 15, 60, 50), fill="black")
            source.save(source_path)
            bbox = BBox(15, 10, 55, 50)
            crop = direct_crop(source, bbox)
            crop_path = assets / "asset_001.png"
            crop.save(crop_path)
            output_path = sample / "output.png"
            Image.new("RGB", (800, 120), "white").save(output_path)

            document = {
                "sample_id": "sample-1",
                "source_image": str(source_path),
                "source_sha256": sha256_file(source_path),
                "formula_count": 0,
                "formula_verified": False,
                "handwriting_detected": False,
                "source_blocks": [
                    {
                        "block_id": "source_0001",
                        "kind": "text",
                        "text": "已知条件，求正确答案。",
                        "source_bbox": {"x": 5, "y": 5, "width": 80, "height": 20},
                        "confidence": 0.99,
                        "disposition": "emitted",
                        "reason": None,
                    },
                    {
                        "block_id": "ignored_0001",
                        "kind": "text",
                        "text": "已核验页眉",
                        "source_bbox": {"x": 5, "y": 30, "width": 60, "height": 10},
                        "confidence": 0.99,
                        "disposition": "ignored",
                        "reason": "authenticated_synthetic_source_metadata",
                    }
                ],
                "visual_assets": [
                    {
                        "asset_id": "asset_001",
                        "source_bbox": {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height},
                        "crop_path": "assets/asset_001.png",
                        "crop_sha256": image_pixel_sha256(crop),
                        "extraction": "direct_source_crop",
                        "redraw_used": False,
                        "preserve_source_pixels": True,
                        "bbox_verified": True,
                        "edge_complete": True,
                        "fallback": False,
                    }
                ],
            }
            report = {
                "run_id": "r1",
                "sample_id": "sample-1",
                "status": "PASS",
                "issues": [],
                "output_sha256": sha256_file(output_path),
                "source_reference_only": True,
                "input_copy_saved": False,
                "render_readback": {"enabled": True, "passed": True},
            }
            _write_json(sample / "question_document.json", document)
            _write_json(sample / "report.json", report)

            document["source_blocks"][1]["reason"] = None
            _write_json(sample / "question_document.json", document)
            with self.assertRaisesRegex(ValueError, "missing an explicit reason"):
                _validate_pass_sample(sample)
            document["source_blocks"][1]["reason"] = "authenticated_synthetic_source_metadata"
            _write_json(sample / "question_document.json", document)

            output_reviews = root / "output-reviews"
            with self.assertRaises(NoPublishableSamplesError):
                publish_release(
                    run,
                    root / "releases",
                    "release-before-human-review",
                    output_review_root=output_reviews,
                )
            _write_json(
                output_reviews / "r1" / "sample-1.json",
                {
                    "schema_version": "output_quality_review_v1",
                    "run_id": "r1",
                    "sample_id": "sample-1",
                    "source_sha256": document["source_sha256"],
                    "output_sha256": report["output_sha256"],
                    "verdict": "pass",
                    "reviewer_id": "human-reviewer-a",
                    "issues": {key: False for key in OUTPUT_ISSUE_KEYS},
                    "notes": [],
                    "history": [],
                    "updated_at_utc": "2026-09-05T12:00:00+00:00",
                },
            )

            summary = publish_release(
                run,
                root / "releases",
                "release-1",
                output_review_root=output_reviews,
            )
            destination = root / "releases" / "release-1"
            self.assertEqual(summary["published_count"], 1)
            self.assertTrue((destination / "sample-1" / "output.png").is_file())
            self.assertTrue((destination / "sample-1" / "assets" / "asset_001.png").is_file())
            self.assertTrue((destination / "sample-1" / "human_output_review.json").is_file())
            self.assertTrue(summary["policy"]["human_final_png_pass_required"])
            self.assertFalse((destination / "source.png").exists())

    def test_release_is_locked_when_run_has_no_pass_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "runs" / "r1" / "review-only"
            _write_json(sample / "report.json", {"status": "REVIEW"})
            with self.assertRaises(NoPublishableSamplesError):
                publish_release(root / "runs" / "r1", root / "releases")


if __name__ == "__main__":
    unittest.main()
