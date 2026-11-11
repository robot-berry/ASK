import tempfile
import unittest
from pathlib import Path

from test7_ocr.evaluation import (
    _gold_verification_method,
    _percentile,
    _has_answer_slot,
    _report_index_path,
    _source_provenance_pass,
    _visual_precision,
    bbox_iou,
    character_error_rate,
    normalize_text,
)


class EvaluationTests(unittest.TestCase):
    def test_source_generated_gold_is_accepted_without_fake_human_signoffs(self) -> None:
        row = {
            "handwriting": {"gold_status": "synthetic_clean_source"},
            "review": {
                "status": "source_generated_verified",
                "independent_review_count": 0,
                "required_independent_review_count": 0,
                "source_alignment": "generator_fact_source",
            },
        }
        self.assertEqual(_gold_verification_method(row), "generator_fact_source")

    def test_source_generated_gold_cannot_masquerade_as_human_review(self) -> None:
        row = {
            "handwriting": {"gold_status": "synthetic_clean_source"},
            "review": {
                "status": "source_generated_verified",
                "independent_review_count": 2,
                "required_independent_review_count": 0,
                "source_alignment": "generator_fact_source",
            },
        }
        self.assertIsNone(_gold_verification_method(row))

    def test_human_gold_still_requires_two_independent_reviews(self) -> None:
        row = {
            "review": {
                "status": "verified",
                "independent_review_count": 2,
                "required_independent_review_count": 2,
                "source_alignment": "human_verified",
            }
        }
        self.assertEqual(_gold_verification_method(row), "human_double_review")

    def test_formula_normalization(self) -> None:
        self.assertEqual(normalize_text(r"$x^{2} \geq \frac{1}{2}$"), normalize_text("x²≥1/2"))
        self.assertEqual(character_error_rate("甲乙", "甲乙"), 0.0)

    def test_bbox_iou(self) -> None:
        self.assertEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(bbox_iou([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)

    def test_visual_precision_penalizes_extra_crop(self) -> None:
        predicted = [
            {"source_bbox": [0, 0, 10, 10]},
            {"source_bbox": [50, 50, 10, 10]},
        ]
        gold = [{"source_bbox": [0, 0, 10, 10]}]
        self.assertEqual(_visual_precision(predicted, gold), 0.5)

    def test_interpolated_percentile(self) -> None:
        self.assertEqual(_percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_source_provenance_requires_emitted_valid_bboxes(self) -> None:
        valid = {
            "source_blocks": [
                {
                    "disposition": "emitted",
                    "source_bbox": {"x": 0, "y": 1, "width": 10, "height": 20},
                }
            ]
        }
        self.assertTrue(_source_provenance_pass(valid))
        invalid = {"source_blocks": [{"disposition": "review_required", "source_bbox": None}]}
        self.assertFalse(_source_provenance_pass(invalid))

    def test_source_provenance_accepts_hash_bound_ignored_blocks_with_reasons(self) -> None:
        document = {
            "source_blocks": [
                {
                    "disposition": "emitted",
                    "source_bbox": {"x": 2, "y": 3, "width": 20, "height": 10},
                },
                {
                    "disposition": "ignored",
                    "reason": "authenticated_synthetic_source_metadata",
                    "source_bbox": {"x": 0, "y": 0, "width": 20, "height": 10},
                },
            ]
        }
        self.assertTrue(_source_provenance_pass(document))
        document["source_blocks"][1]["reason"] = None
        self.assertFalse(_source_provenance_pass(document))

    def test_finalized_index_is_preferred_after_deferred_readback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "index.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertEqual(_report_index_path(root).name, "index.jsonl")
            (root / "index.finalized.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertEqual(_report_index_path(root).name, "index.finalized.jsonl")

    def test_answer_slot_is_not_erased_by_text_normalization(self) -> None:
        self.assertTrue(_has_answer_slot("总体方差为______"))
        self.assertFalse(_has_answer_slot("总体方差为"))


if __name__ == "__main__":
    unittest.main()
