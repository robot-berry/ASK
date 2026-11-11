from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from test7_ocr.capture_pair_review import PAIR_CHECK_KEYS, CapturePairStore, payload_hash
from test7_ocr.clean_source import VerifiedCleanSourceResolver, validate_source_resolution
from test7_ocr.image_ops import sha256_file
from test7_ocr.pipeline import QuestionPipeline, load_config
from test7_ocr.review_server import ReviewStore
from test7_ocr.self_built_intake import SelfBuiltIntakeStore
from test7_ocr.types import BBox, RecognitionResult, RecognizedBlock


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RecordingProvider:
    name = "recording-fixture"

    def __init__(self) -> None:
        self.received_paths: list[Path] = []

    def prepare(self) -> None:
        return None

    def recognize(self, image_path: Path, sample_id: str | None = None) -> RecognitionResult:
        self.received_paths.append(Path(image_path).resolve())
        return RecognitionResult(
            provider=self.name,
            transcript="1. 求值。\nA. 1 B. 2 C. 3 D. 4",
            blocks=[
                RecognizedBlock(
                    "1. 求值。 A. 1 B. 2 C. 3 D. 4",
                    BBox(10, 10, 300, 40),
                    0.99,
                    "text",
                )
            ],
            formula_verified=True,
        )


class VerifiedCleanSourceTests(unittest.TestCase):
    @staticmethod
    def _image_data(color: str) -> str:
        image = Image.new("RGB", (640, 480), color)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _metadata(role: str) -> dict:
        return {
            "role": role,
            "subject": "mathematics",
            "scope": "high_school",
            "document_group": "verified-clean-paper-001",
            "page_index": 1,
            "clean_asset_id": None,
            "device_code": None,
            "conditions": [],
            "page_layout": "not_applicable",
            "provenance": "self_created",
            "license_reference": None,
            "privacy_checked": True,
            "contains_personal_data": False,
            "distribution_allowed": False,
        }

    def _fixture(self, root: Path, *, verify: bool) -> tuple[CapturePairStore, Path, Path, str]:
        intake = SelfBuiltIntakeStore(
            root / "dataset",
            root / "intake-records",
            root / "intake-summary.json",
        )
        clean = intake.register(self._metadata("clean_page"), self._image_data("white"))
        capture_metadata = self._metadata("handwriting_capture")
        capture_metadata.update(
            {
                "clean_asset_id": clean["asset_id"],
                "device_code": "phone-a",
                "conditions": ["frontal"],
            }
        )
        capture = intake.register(capture_metadata, self._image_data("lightgray"))
        store = CapturePairStore(
            root / "pair-tasks",
            root / "pair-manifest.jsonl",
            root / "pair-summary.json",
            intake,
        )
        store.sync_from_intake()
        if verify:
            task = store.load(capture["asset_id"])
            updated = copy.deepcopy(task)
            updated["annotation"]["assessment"] = {
                "same_page": True,
                "page_content_complete": True,
                "capture_usable": True,
                "conditions_confirmed": True,
                "privacy_clear": True,
                "handwriting_state": "present",
            }
            task = store.save(capture["asset_id"], updated, payload_hash(task))
            task = store.signoff(
                capture["asset_id"], "submit_annotation", "annotator-a", payload_hash(task)
            )
            reviewed = copy.deepcopy(task)
            reviewed["review"]["checks"] = {key: True for key in PAIR_CHECK_KEYS}
            task = store.save(capture["asset_id"], reviewed, payload_hash(task))
            store.signoff(
                capture["asset_id"], "approve_review", "reviewer-b", payload_hash(task)
            )
        return (
            store,
            intake.source_path(capture["asset_id"]),
            intake.source_path(clean["asset_id"]),
            capture["asset_id"],
        )

    def test_pipeline_uses_only_verified_clean_source_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, capture_path, clean_path, _capture_id = self._fixture(root, verify=True)
            resolver = VerifiedCleanSourceResolver(store)
            provider = RecordingProvider()
            config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
            config["render_verification"]["enabled"] = False
            pipeline = QuestionPipeline(
                provider,  # type: ignore[arg-type]
                config,
                root / "runs",
                "verified-clean-run",
                source_resolver=resolver,
            )
            with patch("test7_ocr.pipeline.photographed_page_risk", return_value=(False, {})):
                report = pipeline.process(capture_path, "verified-clean-sample")

            self.assertEqual(provider.received_paths, [clean_path.resolve()])
            self.assertEqual(report.input_source_sha256, sha256_file(capture_path))
            self.assertEqual(report.source_sha256, sha256_file(clean_path))
            self.assertTrue(report.source_resolution["substituted"])
            evidence = report.source_resolution["evidence"]
            self.assertEqual(evidence["annotator_id"], "annotator-a")
            self.assertEqual(evidence["reviewer_id"], "reviewer-b")
            self.assertNotEqual(evidence["pair_task_sha256"], "")
            forged_resolution = copy.deepcopy(report.source_resolution)
            forged_resolution["evidence"]["annotator_id"] = "forged-annotator"
            with self.assertRaisesRegex(ValueError, "matching two-person"):
                validate_source_resolution(forged_resolution)
            document = json.loads(
                Path(report.question_document).read_text(encoding="utf-8")
            )
            self.assertEqual(document["source_image"], str(clean_path.resolve()))
            self.assertEqual(document["source_resolution"], report.source_resolution)

            review_dir = root / "gold-reviews"
            review_dir.mkdir()
            task = {
                "sample_id": "verified-clean-sample",
                "source": {
                    "image_path": capture_path.relative_to(root / "dataset").as_posix(),
                    "image_sha256": sha256_file(capture_path),
                    "subject": "mathematics",
                    "include_in_benchmark": True,
                },
                "candidate_prelabels": {},
                "prior_manual_evidence": {},
                "gold_annotation": {
                    "status": "draft",
                    "question_number": "1",
                    "visual_assets": [],
                },
                "review": {
                    "status": "pending",
                    "priority": "p1",
                    "checks": {},
                    "history": [],
                },
            }
            (review_dir / "verified-clean-sample.json").write_text(
                json.dumps(task), encoding="utf-8"
            )
            review_store = ReviewStore(
                review_dir,
                root / "dataset",
                root / "runs",
                root / "output-reviews",
            )
            review_payload = review_store.load_output_review(
                "verified-clean-run", "verified-clean-sample"
            )
            self.assertTrue(review_payload["report"]["source_resolution"]["substituted"])
            self.assertTrue(
                review_store.list_output_items("verified-clean-run")[0][
                    "clean_source_substituted"
                ]
            )

    def test_unverified_pair_never_substitutes_the_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, capture_path, _clean_path, _capture_id = self._fixture(root, verify=False)
            resolver = VerifiedCleanSourceResolver(store)
            resolution = resolver.resolve(capture_path, sha256_file(capture_path))
            self.assertFalse(resolution.substituted)
            self.assertEqual(resolution.method, "direct_input")
            self.assertEqual(Path(resolution.effective_image), capture_path.resolve())
            self.assertIsNone(resolution.evidence)

    def test_pair_task_change_after_resolution_index_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store, capture_path, _clean_path, capture_id = self._fixture(root, verify=True)
            resolver = VerifiedCleanSourceResolver(store)
            task_path = store.task_dir / f"{capture_id}.json"
            task_path.write_text(task_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence changed"):
                resolver.resolve(capture_path, sha256_file(capture_path))


if __name__ == "__main__":
    unittest.main()
