from __future__ import annotations

import json
from pathlib import Path
import unittest


from test7_ocr.external_datasets import local_registry_roots_under


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DatasetRoleTests(unittest.TestCase):
    def test_gaokao_physics_is_the_only_primary_ocr_gold(self) -> None:
        registry = json.loads(
            (PROJECT_ROOT / "configs" / "datasets.json").read_text(encoding="utf-8")
        )

        self.assertEqual(registry["primary_ocr_gold"], "gaokao_physics_300_v1")
        gaokao = registry["datasets"]["gaokao_physics_300_v1"]
        testzip = registry["datasets"]["testzip_math_physics_300_v1"]
        self.assertEqual(gaokao["formal_scoring_requires_verified_records"], 300)
        self.assertFalse(testzip["primary_ocr_gold"])
        self.assertEqual(
            testzip["role"],
            "historical_layout_regression_and_handwriting_candidate_source",
        )

    def test_retired_testzip_status_cannot_enable_formal_scoring(self) -> None:
        status = json.loads(
            (
                PROJECT_ROOT
                / "data"
                / "gold"
                / "testzip_math_physics_300"
                / "STATUS.json"
            ).read_text(encoding="utf-8")
        )

        self.assertFalse(status["primary_ocr_gold"])
        self.assertFalse(status["formal_scoring_enabled"])
        self.assertEqual(status["current_primary_ocr_gold"], "gaokao_physics_300_v1")
        self.assertEqual(status["required_records_for_formal_scoring"], 0)

    def test_every_local_dataset_directory_has_an_explicit_role(self) -> None:
        registry = json.loads(
            (PROJECT_ROOT / "configs" / "datasets.json").read_text(encoding="utf-8")
        )
        datasets_root = (PROJECT_ROOT / ".." / ".." / "datasets").resolve()
        registered_roots = local_registry_roots_under(datasets_root, registry)
        discovered_roots = {path.resolve() for path in datasets_root.iterdir() if path.is_dir()}
        self.assertEqual(discovered_roots, registered_roots)

    def test_external_registry_entries_cannot_become_formal_gold(self) -> None:
        registry = json.loads(
            (PROJECT_ROOT / "configs" / "datasets.json").read_text(encoding="utf-8")
        )
        external = json.loads(
            (PROJECT_ROOT / "configs" / "external_datasets.json").read_text(encoding="utf-8")
        )
        self.assertEqual(registry["primary_ocr_gold"], "gaokao_physics_300_v1")
        for dataset_id in (
            "omnidocbench",
            "unimer_test",
            "mathvision_wild",
            "warpdoc",
            "realdae",
        ):
            entry = registry["datasets"][dataset_id]
            self.assertTrue(entry["evaluation_only"])
            self.assertFalse(entry["training_allowed"])
            self.assertFalse(entry["formal_scoring_enabled"])
            self.assertFalse(entry["primary_ocr_gold"])
            self.assertFalse(entry["counts_toward_self_built_real_world_targets"])
            self.assertFalse(entry["lives_under_ask_datasets_root"])
        self.assertEqual(external["datasets"]["warpdoc"]["license_status"], "license_review_required")
        self.assertEqual(external["datasets"]["realdae"]["license_status"], "license_review_required")
        self.assertEqual(external["datasets"]["scut_ensexam"]["license_status"], "application_required")

    def test_only_quarantined_collection_counts_toward_real_world_targets(self) -> None:
        registry = json.loads(
            (PROJECT_ROOT / "configs" / "datasets.json").read_text(encoding="utf-8")
        )
        counting = {
            dataset_id
            for dataset_id, entry in registry["datasets"].items()
            if entry.get("counts_toward_self_built_real_world_targets")
        }

        self.assertEqual(counting, {"test7_ocr_self_built_v1"})
        self.assertEqual(
            registry["datasets"]["testzip_math_physics_300_v1"]["role"],
            "historical_layout_regression_and_handwriting_candidate_source",
        )
        self.assertFalse(
            registry["datasets"]["testzip_math_physics_300_v2_eink"][
                "formal_scoring_enabled"
            ]
        )

    def test_legacy_promotion_and_validation_scripts_keep_testzip_retired(self) -> None:
        promotion = (PROJECT_ROOT / "scripts" / "promote_testzip_ocr_gold.py").read_text(
            encoding="utf-8"
        )
        validation = (
            PROJECT_ROOT / "scripts" / "validate_testzip_ocr_gold.py"
        ).read_text(encoding="utf-8")

        self.assertIn('"formal_scoring_enabled": False', promotion)
        self.assertNotIn('"formal_scoring_enabled": complete', promotion)
        self.assertIn('"current_primary_ocr_gold": "gaokao_physics_300_v1"', promotion)
        self.assertIn(
            '"primary_ocr_gold: gaokao_physics_300_v1"',
            validation,
        )

    def test_self_built_guide_names_current_gold_and_keeps_testzip_retired(self) -> None:
        guide = (PROJECT_ROOT / "docs" / "SELF_BUILT_DATASET.md").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            r"D:\robot\Documents\ASK\datasets\gaokao_physics_300_v1_0_1",
            guide,
        )
        self.assertIn("`testzip_math_physics_300_v1` 已停止作为正式 OCR Gold", guide)
        self.assertIn("300 条 OCR Gold", guide)
        self.assertNotIn(
            r"D:\robot\Documents\ASK\datasets\testzip_math_physics_300_v1",
            guide,
        )
        self.assertNotIn("298 条 OCR Gold", guide)
        self.assertNotIn("必须达到 298/298", guide)


if __name__ == "__main__":
    unittest.main()
