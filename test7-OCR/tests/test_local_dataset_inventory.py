from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class LocalDatasetInventoryTests(unittest.TestCase):
    def test_inventory_keeps_non_collection_images_out_of_self_built_count(self) -> None:
        import importlib.util

        module_path = PROJECT_ROOT / "scripts" / "audit_local_dataset_inventory.py"
        spec = importlib.util.spec_from_file_location("audit_local_dataset_inventory", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gold = root / "gold"
            collection = root / "collection"
            gold.mkdir()
            collection.mkdir()
            Image.new("RGB", (16, 8), "white").save(gold / "one.png")
            Image.new("RGB", (8, 16), "black").save(collection / "two.png")
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "primary_ocr_gold": "gold",
                        "datasets": {
                            "gold": {
                                "root": str(gold),
                                "role": "primary_source_derived_ocr_gold",
                                "formal_scoring_requires_verified_records": 1,
                                "counts_toward_self_built_real_world_targets": False,
                            },
                            "collection": {
                                "root": str(collection),
                                "role": "quarantined_real_world_collection",
                                "counts_toward_self_built_real_world_targets": True,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            original_project_root = module.PROJECT_ROOT
            module.PROJECT_ROOT = root
            try:
                report = module.audit_inventory(root, registry_path)
            finally:
                module.PROJECT_ROOT = original_project_root

        self.assertEqual(report["total_image_files"], 2)
        self.assertEqual(
            report["self_built_real_world_coverage"]["eligible_dataset_ids"],
            ["collection"],
        )
        self.assertEqual(
            report["self_built_real_world_coverage"]["present_image_files"], 1
        )
        self.assertTrue(
            report["self_built_real_world_coverage"][
                "other_dataset_images_may_not_fill_collection_slots"
            ]
        )


if __name__ == "__main__":
    unittest.main()
