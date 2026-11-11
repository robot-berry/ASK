from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from test7_ocr.cli import main
from test7_ocr.evaluation import evaluate_run
from test7_ocr.external_datasets import (
    LicenseBlockedError,
    assert_download_allowed,
    infer_dataset_role,
    stable_unimer_sample,
)
from test7_ocr.external_eval import evaluate_external_run
from test7_ocr.external_fetch import fetch_dataset
from test7_ocr.external_manifests import (
    prepare_mathvision_wild_manifests,
    prepare_omnidocbench_manifests,
    prepare_unimer_manifests,
)
from test7_ocr.storage import write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ExternalDatasetTests(unittest.TestCase):
    def test_license_gate_blocks_warpdoc_and_realdae(self) -> None:
        with self.assertRaises(LicenseBlockedError):
            assert_download_allowed("warpdoc")
        with self.assertRaises(LicenseBlockedError):
            assert_download_allowed("realdae")
        with self.assertRaises(LicenseBlockedError):
            assert_download_allowed("scut_ensexam")
        with self.assertRaises(LicenseBlockedError):
            fetch_dataset("warpdoc")
        with self.assertRaises(LicenseBlockedError):
            fetch_dataset("realdae")

    def test_omnidoc_exam_zh_manifest_is_evaluation_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            image_path = images / "exam.png"
            Image.new("RGB", (800, 600), "white").save(image_path)
            annotation = {
                "page_info": {"page_no": 0, "image_path": "images/exam.png"},
                "page_attribute": {
                    "data_source": "exam_paper",
                    "language": "simplified_chinese",
                },
                "layout_dets": [
                    {"order": 1, "category_type": "text", "text": "已知 x=1，求 y。"},
                    {"order": 2, "category_type": "equation_isolated", "latex": "y=x^2"},
                ],
            }
            (root / "OmniDocBench.json").write_text(
                json.dumps([annotation], ensure_ascii=False),
                encoding="utf-8",
            )
            summary = prepare_omnidocbench_manifests(
                root,
                exam_manifest=root / "exam.jsonl",
                exam_gold=root / "exam_gold.jsonl",
                smoke_manifest=root / "smoke.jsonl",
                smoke_gold=root / "smoke_gold.jsonl",
            )
            self.assertEqual(summary["exam_zh"]["rows"], 1)
            row = json.loads((root / "exam.jsonl").read_text(encoding="utf-8").splitlines()[0])
            gold = json.loads((root / "exam_gold.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(row["training_allowed"])
            self.assertTrue(row["evaluation_only"])
            self.assertIn("y=x^2", gold["gold_formulas"])
            self.assertTrue(gold["ignore_answer"])

    def test_unimer_stratified_sample_is_stable(self) -> None:
        first = stable_unimer_sample([f"id-{index}" for index in range(400)], "SPE", 100)
        second = stable_unimer_sample([f"id-{index}" for index in range(400)], "SPE", 100)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for bucket in ("SPE", "CPE", "SCE", "HWE"):
                folder = root / bucket
                folder.mkdir()
                for index in range(5):
                    image = folder / f"{index}.png"
                    Image.new("RGB", (64, 64), "white").save(image)
                    image.with_suffix(".txt").write_text(f"x_{bucket}_{index}", encoding="utf-8")
            summary = prepare_unimer_manifests(
                root,
                per_bucket=3,
                manifest=root / "unimer.jsonl",
                gold=root / "unimer_gold.jsonl",
                smoke_manifest=root / "smoke.jsonl",
                smoke_gold=root / "smoke_gold.jsonl",
            )
            self.assertEqual(summary["stratified"]["rows"], 12)
            row = json.loads((root / "unimer.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["scoring_mode"], "formula_only")

    def test_unimer_list_labels_follow_filename_index_not_sort_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "spe"
            folder.mkdir()
            Image.new("RGB", (32, 32), "white").save(folder / "0000010.png")
            Image.new("RGB", (32, 32), "white").save(folder / "0000020.png")
            lines = [f"pad-{index}" for index in range(21)]
            lines[10] = r"e^{i\theta}"
            lines[20] = r"a+b"
            (root / "spe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            summary = prepare_unimer_manifests(
                root,
                per_bucket=2,
                manifest=root / "unimer.jsonl",
                gold=root / "unimer_gold.jsonl",
                smoke_manifest=root / "smoke.jsonl",
                smoke_gold=root / "smoke_gold.jsonl",
            )
            self.assertEqual(summary["stratified"]["rows"], 2)
            gold_rows = {
                json.loads(line)["sample_id"]: json.loads(line)
                for line in (root / "unimer_gold.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            self.assertEqual(gold_rows["UNIMER-SPE-0000010"]["gold_text"], r"e^{i\theta}")
            self.assertEqual(gold_rows["UNIMER-SPE-0000020"]["gold_text"], r"a+b")

    def test_mathvision_ignores_answer_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (200, 200), "white").save(images / "a.png")
            Image.new("RGB", (200, 200), "white").save(images / "b.png")
            (root / "annotations.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "mini-1",
                            "split": "testmini_photo",
                            "image": "images/a.png",
                            "question": "What is 1+1?",
                            "options": ["1", "2"],
                            "answer": "MUST-NOT-BE-USED",
                        },
                        {
                            "id": "full-9",
                            "split": "test_photo",
                            "image": "images/b.png",
                            "question": "Find the area",
                            "options": ["A", "B"],
                            "answer": "ALSO-IGNORED",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            prepare_mathvision_wild_manifests(
                root,
                frozen_manifest=root / "frozen.jsonl",
                frozen_gold=root / "frozen_gold.jsonl",
                mini_manifest=root / "mini.jsonl",
                mini_gold=root / "mini_gold.jsonl",
            )
            gold = json.loads((root / "frozen_gold.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(gold["ignore_answer"])
            self.assertNotIn("MUST-NOT-BE-USED", json.dumps(gold))
            self.assertNotIn("ALSO-IGNORED", gold["gold_text"])
            self.assertEqual(gold["gold_question"], "Find the area")

    def test_mathvision_resolves_file_name_beside_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photos = root / "test_photo"
            photos.mkdir()
            Image.new("RGB", (120, 120), "white").save(photos / "photo_1.jpg")
            (photos / "metadata.jsonl").write_text(
                json.dumps(
                    {
                        "id": "9",
                        "file_name": "photo_1.jpg",
                        "answer": "IGNORED",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            prepare_mathvision_wild_manifests(
                root,
                frozen_manifest=root / "frozen.jsonl",
                frozen_gold=root / "frozen_gold.jsonl",
                mini_manifest=root / "mini.jsonl",
                mini_gold=root / "mini_gold.jsonl",
            )
            row = json.loads((root / "frozen.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["image_path"], "test_photo/photo_1.jpg")
            self.assertTrue(row["evaluation_only"])

    def test_omnidoc_reads_page_info_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (800, 600), "white").save(images / "exam.png")
            annotation = {
                "page_info": {
                    "page_no": 0,
                    "image_path": "images/exam.png",
                    "page_attribute": {
                        "data_source": "exam_paper",
                        "language": "simplified_chinese",
                    },
                },
                "layout_dets": [{"order": 1, "category_type": "text", "text": "求值"}],
            }
            (root / "OmniDocBench.json").write_text(
                json.dumps([annotation], ensure_ascii=False),
                encoding="utf-8",
            )
            summary = prepare_omnidocbench_manifests(
                root,
                exam_manifest=root / "exam.jsonl",
                exam_gold=root / "exam_gold.jsonl",
                smoke_manifest=root / "smoke.jsonl",
                smoke_gold=root / "smoke_gold.jsonl",
            )
            self.assertEqual(summary["exam_zh"]["rows"], 1)

    def test_fetch_uses_logical_pins_not_git_revisions(self) -> None:
        from test7_ocr.external_datasets import get_external_dataset
        from test7_ocr.external_fetch import huggingface_revision_for

        omni = get_external_dataset("omnidocbench")
        unimer = get_external_dataset("unimer_test")
        mathvision = get_external_dataset("mathvision_wild")
        self.assertIsNone(huggingface_revision_for("omnidocbench", omni))
        self.assertIsNone(huggingface_revision_for("unimer_test", unimer))
        self.assertIsNone(huggingface_revision_for("mathvision_wild", mathvision))
        self.assertEqual(unimer["hf_repo"], "wanderkid/UniMER_Dataset")
        self.assertEqual(mathvision["hf_repo"], "MathLLMs/MathVision-Wild")
        self.assertNotEqual(unimer["hf_repo"], "wanderkid/UniMER-1M")
        self.assertNotEqual(mathvision["hf_repo"], "MathLLMs/MathVision")

    def test_unimer_reads_bucket_label_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "spe"
            folder.mkdir()
            for index in range(4):
                Image.new("RGB", (32, 32), "white").save(folder / f"{index}.png")
            (root / "spe.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
            summary = prepare_unimer_manifests(
                root,
                per_bucket=2,
                manifest=root / "unimer.jsonl",
                gold=root / "unimer_gold.jsonl",
                smoke_manifest=root / "smoke.jsonl",
                smoke_gold=root / "smoke_gold.jsonl",
            )
            self.assertEqual(summary["buckets"]["SPE"]["available"], 4)
            gold_text = (root / "unimer_gold.jsonl").read_text(encoding="utf-8")
            self.assertTrue(any(token in gold_text for token in ("a", "b", "c", "d")))

    def test_external_eval_never_sets_formal_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            sample = run / "ODB-EXAM-ZH-0000"
            sample.mkdir(parents=True)
            write_json_atomic(
                sample / "question_document.json",
                {
                    "display_text": "已知 x=1，求 y。 y=x^2",
                    "source_blocks": [
                        {"kind": "formula", "text": "y=x^2"},
                    ],
                },
            )
            write_json_atomic(
                sample / "report.json",
                {
                    "sample_id": "ODB-EXAM-ZH-0000",
                    "status": "REVIEW",
                    "output_width": 800,
                    "elapsed_seconds": 3.2,
                    "question_document": str(sample / "question_document.json"),
                },
            )
            (run / "index.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "ODB-EXAM-ZH-0000",
                        "question_document": str(sample / "question_document.json"),
                        "status": "REVIEW",
                        "output_width": 800,
                        "elapsed_seconds": 3.2,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            gold = root / "gold.jsonl"
            gold.write_text(
                json.dumps(
                    {
                        "sample_id": "ODB-EXAM-ZH-0000",
                        "scoring_mode": "page_text_and_formulas",
                        "gold_text": "已知 x=1，求 y。 y=x^2",
                        "gold_formulas": ["y=x^2"],
                        "ignore_answer": True,
                        "text_cer_applicable": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = evaluate_external_run(run, gold, dataset_id="omnidocbench")
            self.assertFalse(summary["formal_gold_scoring"])
            self.assertEqual(summary["evaluated_count"], 1)
            self.assertEqual(summary["metrics"]["latex_leakage_rate"], 0.0)
            output = root / "report.json"
            code = main(
                [
                    "external-eval",
                    str(run),
                    "--gold",
                    str(gold),
                    "--dataset-id",
                    "omnidocbench",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(code, 0)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(saved["formal_gold_scoring"])

    def test_formal_evaluate_run_still_locks_without_gold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(RuntimeError):
                evaluate_run(root / "missing", root / "missing.jsonl", required_gold_count=300)

    def test_infer_external_role_from_registry_root(self) -> None:
        role = infer_dataset_role(PROJECT_ROOT / "data" / "external" / "omnidocbench")
        self.assertEqual(role, "external_eval_only")
        self.assertEqual(
            infer_dataset_role(PROJECT_ROOT / "data" / "external" / "omnidocbench", explicit="ocr_external_test"),
            "ocr_external_test",
        )


if __name__ == "__main__":
    unittest.main()
