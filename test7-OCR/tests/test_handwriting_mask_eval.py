from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_handwriting_mask_eval.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("prepare_handwriting_mask_eval", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load handwriting mask preparation script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HandwritingMaskEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_script_module()

    def test_evidence_selected_frozen_tasks_are_valid(self) -> None:
        summary, errors = self.module.check()
        self.assertEqual(errors, [])
        self.assertEqual(summary["selected_images"], summary["task_files"])
        self.assertGreaterEqual(
            summary["selected_images"],
            len(self.module.selection_evidence(self.module.gold_tasks())),
        )

    def test_task_contract_forbids_training_and_defines_mask_classes(self) -> None:
        task_paths = sorted(self.module.TASK_DIR.glob("*.json"))
        self.assertGreaterEqual(len(task_paths), 1)
        for path in task_paths:
            task = json.loads(path.read_text(encoding="utf-8"))
            self.assertIs(task["training_allowed"], False)
            self.assertEqual(task["annotation"]["class_values"], self.module.CLASS_VALUES)
            if task["schema_version"] == "handwriting_mask_review_v1":
                self.assertEqual(task["partition"], "frozen_test")
                self.assertGreaterEqual(len(task["selection_evidence"]), 1)
            else:
                self.assertEqual(task["schema_version"], "handwriting_mask_review_v2")
                self.assertEqual(task["partition"], "quarantine")
                self.assertIs(task["evaluation_allowed"], False)

    def test_schema_matches_runtime_legend(self) -> None:
        schema = json.loads(
            (PROJECT_ROOT / "schemas" / "handwriting_mask_review.schema.json").read_text(
                encoding="utf-8"
            )
        )
        class_properties = schema["properties"]["annotation"]["properties"][
            "class_values"
        ]["properties"]
        schema_values = {key: value["const"] for key, value in class_properties.items()}
        self.assertEqual(schema_values, self.module.CLASS_VALUES)

        self_built_schema = json.loads(
            (PROJECT_ROOT / "schemas" / "handwriting_mask_review_v2.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            self_built_schema["properties"]["schema_version"]["const"],
            "handwriting_mask_review_v2",
        )
        self.assertEqual(self_built_schema["properties"]["partition"]["const"], "quarantine")
        self.assertIs(self_built_schema["properties"]["training_allowed"]["const"], False)
        self.assertIs(self_built_schema["properties"]["evaluation_allowed"]["const"], False)


if __name__ == "__main__":
    unittest.main()
