from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from test7_ocr.capture_pair_review import CapturePairStore
from test7_ocr.handwriting_review import HandwritingMaskStore
from test7_ocr.question_boundary_review import QuestionBoundaryStore
from test7_ocr.self_built_freeze import DEFAULT_TARGETS, SelfBuiltFreezeBuilder
from test7_ocr.self_built_intake import SelfBuiltIntakeStore


class SelfBuiltFreezeTests(unittest.TestCase):
    def _builder(
        self, root: Path, *, targets: dict[str, int] | None = None
    ) -> SelfBuiltFreezeBuilder:
        primary = root / "primary"
        primary.mkdir()
        gold = root / "ocr_gold.jsonl"
        gold.write_text("", encoding="utf-8")
        intake = SelfBuiltIntakeStore(
            root / "self-built", root / "intake-records", root / "intake-summary.json"
        )
        masks = HandwritingMaskStore(
            root / "mask-tasks", root / "mask-files", primary, intake
        )
        boundaries = QuestionBoundaryStore(
            root / "boundary-tasks",
            root / "boundary-manifest.jsonl",
            root / "boundary-summary.json",
            intake,
        )
        pairs = CapturePairStore(
            root / "pair-tasks",
            root / "pair-manifest.jsonl",
            root / "pair-summary.json",
            intake,
        )
        return SelfBuiltFreezeBuilder(
            primary_dataset_root=primary,
            ocr_gold_path=gold,
            intake=intake,
            masks=masks,
            boundaries=boundaries,
            capture_pairs=pairs,
            freeze_root=root / "frozen",
            targets=targets,
            required_capture_conditions=set(),
        )

    def test_default_strict_audit_reports_real_target_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            builder = self._builder(Path(temporary))
            summary, errors, _payload = builder.audit(require_targets=True)
            self.assertFalse(summary["ready_to_freeze"])
            self.assertEqual(summary["counts"]["ocr_gold"], 0)
            for key, target in DEFAULT_TARGETS.items():
                if target:
                    self.assertTrue(
                        any(f"freeze target {key}: 0/{target}" in error for error in errors),
                        key,
                    )

    def test_freeze_is_immutable_and_hash_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            zero_targets = {key: 0 for key in DEFAULT_TARGETS}
            builder = self._builder(root, targets=zero_targets)
            version = "test7-self-built-v2026.09.06-01"
            manifest = builder.freeze(version)
            destination = root / "frozen" / version
            self.assertTrue((destination / "freeze_manifest.json").is_file())
            self.assertFalse(manifest["source_images_copied"])
            self.assertEqual(manifest["partition"], "frozen_test")
            self.assertEqual(manifest["files"]["ocr_gold.jsonl"]["byte_size"], 0)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                builder.freeze(version)

            (destination / "ocr_gold.jsonl").write_text("tampered", encoding="utf-8")
            _manifest, errors = builder.verify_freeze(destination)
            self.assertTrue(any("hash mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
