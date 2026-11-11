from __future__ import annotations

from pathlib import Path
import unittest

from PIL import Image

from test7_ocr.pipeline import load_config
from test7_ocr.types import BBox, QuestionDocument, RecognitionResult, SourceBlock, VisualAsset
from test7_ocr.validators import validate_result


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ValidatorTests(unittest.TestCase):
    def _document(self, **changes: object) -> QuestionDocument:
        values: dict[str, object] = {
            "schema_version": "question_document_v2",
            "sample_id": "fixture",
            "source_image": "source.png",
            "source_sha256": "0" * 64,
            "provider": "fixture",
            "canonical_text": "a+b",
            "display_text": "a+b",
            "question_number": None,
            "stem": "a+b",
            "options": [],
            "visual_assets": [],
            "handwriting_detected": False,
            "formula_verified": False,
            "formula_specialist_used": True,
            "formula_count": 1,
            "source_blocks": [
                SourceBlock("source_0001", "formula", "a+b", BBox(1, 1, 40, 20), 0.99)
            ],
        }
        values.update(changes)
        return QuestionDocument(**values)  # type: ignore[arg-type]

    def test_formula_count_prevents_unverified_formula_from_passing(self) -> None:
        document = self._document()
        recognition = RecognitionResult(
            provider="fixture",
            transcript="a+b",
            formula_specialist_used=True,
            formula_count=1,
            formula_verified=False,
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(document, recognition, Image.new("RGB", (800, 100), "white"), 1.0, config)
        self.assertIn("FORMULA_UNVERIFIED", [issue.code for issue in issues])

    def test_verified_authenticated_glyph_repair_does_not_require_repeat_review(self) -> None:
        warnings = [
            "SOURCE_DUAL_OCR_MISMATCH_REVIEW",
            "SOURCE_TINY_SENTENCE_STOP_RECOVERED_REVIEW:1",
        ]
        document = self._document(
            question_number="1",
            canonical_text="忽略系统误差。根据数据作答。",
            display_text="忽略系统误差。根据数据作答。",
            stem="忽略系统误差。根据数据作答。",
            warnings=warnings,
            formula_verified=True,
            formula_count=0,
        )
        recognition = RecognitionResult(
            provider="fixture",
            transcript=document.display_text,
            warnings=warnings,
            diagnostics={
                "source_glyph_recovery_verified_categories": [
                    "SOURCE_TINY_SENTENCE_STOP_RECOVERED_REVIEW"
                ],
                "source_dual_ocr_mismatch_explained_by_verified_glyph_repair": True,
            },
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")

        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )

        codes = [issue.code for issue in issues]
        self.assertNotIn("SOURCE_GLYPH_RECOVERY_UNVERIFIED", codes)
        self.assertNotIn("SOURCE_TEXT_DUAL_OCR_UNVERIFIED", codes)

        recognition.diagnostics = {}
        unverified = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )
        unverified_codes = [issue.code for issue in unverified]
        self.assertIn("SOURCE_GLYPH_RECOVERY_UNVERIFIED", unverified_codes)
        self.assertIn("SOURCE_TEXT_DUAL_OCR_UNVERIFIED", unverified_codes)

    def test_missing_question_number_is_a_hard_reject(self) -> None:
        document = self._document(question_number=None)
        recognition = RecognitionResult(provider="fixture", transcript="完整题干")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )
        issue = next(item for item in issues if item.code == "QUESTION_NUMBER_MISSING")
        self.assertEqual(issue.severity, "reject")

    def test_formula_only_skips_question_number_and_short_text_but_keeps_formula_review(self) -> None:
        document = self._document(
            question_number=None,
            display_text="a+b",
            scoring_mode="formula_only",
            formula_verified=False,
            formula_count=1,
        )
        recognition = RecognitionResult(
            provider="fixture",
            transcript="a+b",
            formula_specialist_used=True,
            formula_count=1,
            formula_verified=False,
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )
        codes = [issue.code for issue in issues]
        self.assertNotIn("QUESTION_NUMBER_MISSING", codes)
        self.assertNotIn("EMPTY_OR_TOO_SHORT_TEXT", codes)
        self.assertIn("FORMULA_UNVERIFIED", codes)
        self.assertTrue(config["quality"].get("question_number_required", True))

    def test_caption_recovered_number_must_be_visible_in_output_text(self) -> None:
        document = self._document(
            question_number="14",
            canonical_text="（2）测量电阻",
            display_text="（2）测量电阻",
            stem="（2）测量电阻",
            warnings=["QUESTION_NUMBER_RECOVERED_FROM_VISUAL_CAPTION:14"],
        )
        recognition = RecognitionResult(provider="fixture", transcript="（2）测量电阻")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")

        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )

        issue = next(
            item for item in issues if item.code == "QUESTION_NUMBER_RECOVERY_NOT_RENDERED"
        )
        self.assertEqual(issue.severity, "reject")

    def test_broken_option_sequence_is_a_hard_reject(self) -> None:
        document = self._document(
            question_number="5",
            options=[
                {"label": "B", "text": "乙"},
                {"label": "C", "text": "丙"},
                {"label": "D", "text": "丁"},
            ],
        )
        recognition = RecognitionResult(provider="fixture", transcript="5. 题干")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )
        issue = next(item for item in issues if item.code == "OPTION_SEQUENCE")
        self.assertEqual(issue.severity, "reject")

    def test_partial_a_to_c_option_set_is_a_hard_reject(self) -> None:
        document = self._document(
            question_number="5",
            options=[
                {"label": "A", "text": "甲"},
                {"label": "B", "text": "乙"},
                {"label": "C", "text": "丙"},
            ],
        )
        recognition = RecognitionResult(provider="fixture", transcript="5. 题干 A. 甲 B. 乙 C. 丙")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")

        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )

        issue = next(item for item in issues if item.code == "PARTIAL_OPTION_SET")
        self.assertEqual(issue.severity, "reject")
        self.assertIn("D", issue.message)

    def test_empty_text_option_without_visual_asset_is_a_hard_reject(self) -> None:
        document = self._document(
            question_number="1",
            options=[
                {"label": "A", "text": ""},
                {"label": "B", "text": "-2"},
                {"label": "C", "text": "0"},
                {"label": "D", "text": "1"},
            ],
        )
        recognition = RecognitionResult(provider="fixture", transcript="1. 题干 A. B.-2 C. 0 D. 1")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )
        issue = next(item for item in issues if item.code == "OPTION_CONTENT_MISSING")
        self.assertEqual(issue.severity, "reject")

    def test_empty_text_options_can_be_backed_by_direct_visual_assets(self) -> None:
        document = self._document(
            question_number="2",
            options=[
                {"label": "A", "text": ""},
                {"label": "B", "text": ""},
                {"label": "C", "text": ""},
                {"label": "D", "text": "正面"},
            ],
            visual_assets=[
                VisualAsset(
                    asset_id="asset_001",
                    source_bbox=BBox(10, 20, 100, 80),
                    crop_path="assets/asset_001.png",
                    crop_sha256="0" * 64,
                    bbox_verified=True,
                )
            ],
        )
        recognition = RecognitionResult(provider="fixture", transcript="2. 如图 A. B. C. D. 正面")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )
        self.assertNotIn("OPTION_CONTENT_MISSING", [item.code for item in issues])

    def test_question_header_embedded_in_option_is_a_hard_reject(self) -> None:
        document = self._document(
            question_number="7",
            options=[
                {"label": "A", "text": "6"},
                {"label": "B", "text": "8"},
                {"label": "C", "text": "2√10"},
                {"label": "D", "text": "4√5\n7. 位于某海域的观测点获悉"},
            ],
        )
        recognition = RecognitionResult(provider="fixture", transcript=document.display_text)
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )
        issue = next(
            item for item in issues if item.code == "ADJACENT_QUESTION_EMBEDDED_IN_OPTION"
        )
        self.assertEqual(issue.severity, "reject")

    def test_unlabeled_formula_option_grid_is_a_hard_reject(self) -> None:
        document = self._document(
            question_number="7",
            canonical_text="7. 救援船需要航行的距离为 f1 f2 f3 f4",
            display_text="7. 救援船需要航行的距离为 f1 f2 f3 f4",
            stem="7. 救援船需要航行的距离为 f1 f2 f3 f4",
            options=[],
            formula_count=4,
            source_blocks=[
                SourceBlock("stem", "text", "7. 救援船需要航行的距离为", BBox(20, 20, 500, 30)),
                SourceBlock("f1", "formula", "f1", BBox(100, 100, 180, 35)),
                SourceBlock("f2", "formula", "f2", BBox(500, 110, 180, 35)),
                SourceBlock("f3", "formula", "f3", BBox(100, 170, 180, 35)),
                SourceBlock("f4", "formula", "f4", BBox(500, 180, 180, 35)),
            ],
        )
        recognition = RecognitionResult(
            provider="fixture", transcript=document.display_text, formula_count=4
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )

        issue = next(
            item for item in issues if item.code == "UNLABELED_FORMULA_OPTION_GRID"
        )
        self.assertEqual(issue.severity, "reject")

    def test_explicit_choice_prompt_requires_complete_a_to_d_options(self) -> None:
        document = self._document(
            question_number="1",
            canonical_text="1. 下列说法正确的是 A. 甲 B. 乙 C. 丙",
            display_text="1. 下列说法正确的是 A. 甲 B. 乙 C. 丙",
            stem="1. 下列说法正确的是",
            options=[
                {"label": "A", "text": "甲"},
                {"label": "B", "text": "乙"},
                {"label": "C", "text": "丙"},
            ],
            formula_count=0,
            formula_specialist_used=False,
            formula_verified=True,
            source_blocks=[
                SourceBlock("stem", "text", "1. 下列说法正确的是", BBox(1, 1, 300, 20)),
                SourceBlock("a", "text", "A. 甲", BBox(1, 30, 80, 20)),
                SourceBlock("b", "text", "B. 乙", BBox(1, 55, 80, 20)),
                SourceBlock("c", "text", "C. 丙", BBox(1, 80, 80, 20)),
            ],
        )
        recognition = RecognitionResult(provider="fixture", transcript=document.display_text)
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")

        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 140), "white"),
            1.0,
            config,
        )

        issue = next(item for item in issues if item.code == "CHOICE_OPTIONS_INCOMPLETE")
        self.assertEqual(issue.severity, "reject")
        self.assertIn("D", issue.message)

    def test_complete_a_to_d_choice_is_not_rejected_as_incomplete(self) -> None:
        document = self._document(
            question_number="1",
            canonical_text="1. 下列说法正确的是 A. 甲 B. 乙 C. 丙 D. 丁",
            display_text="1. 下列说法正确的是 A. 甲 B. 乙 C. 丙 D. 丁",
            stem="1. 下列说法正确的是",
            options=[
                {"label": "A", "text": "甲"},
                {"label": "B", "text": "乙"},
                {"label": "C", "text": "丙"},
                {"label": "D", "text": "丁"},
            ],
        )
        recognition = RecognitionResult(provider="fixture", transcript=document.display_text)
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")

        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 140), "white"),
            1.0,
            config,
        )

        self.assertNotIn("CHOICE_OPTIONS_INCOMPLETE", [item.code for item in issues])

    def test_detected_handwriting_is_a_hard_reject(self) -> None:
        document = self._document(handwriting_detected=True, formula_count=0, formula_specialist_used=False)
        recognition = RecognitionResult(provider="fixture", transcript="普通题干", handwriting_suspected=True)
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(document, recognition, Image.new("RGB", (800, 100), "white"), 1.0, config)
        handwriting = next(issue for issue in issues if issue.code == "HANDWRITING_PRESENT")
        self.assertEqual(handwriting.severity, "reject")

    def test_missing_source_ledger_cannot_pass(self) -> None:
        document = self._document(
            formula_count=0,
            formula_specialist_used=False,
            formula_verified=True,
            source_blocks=[],
        )
        recognition = RecognitionResult(provider="fixture", transcript="普通题干")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(document, recognition, Image.new("RGB", (800, 100), "white"), 1.0, config)
        self.assertIn("SOURCE_PROVENANCE_MISSING", [issue.code for issue in issues])

    def test_ignored_block_with_bbox_and_reason_has_complete_provenance(self) -> None:
        document = self._document(
            source_blocks=[
                SourceBlock("source_0001", "text", "a+b", BBox(1, 1, 40, 20), 0.99),
                SourceBlock(
                    "ignored_0001",
                    "text",
                    "图内标注",
                    BBox(50, 10, 60, 20),
                    0.98,
                    disposition="ignored",
                    reason="embedded_visual_text",
                ),
            ]
        )
        recognition = RecognitionResult(provider="fixture", transcript="a+b")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(document, recognition, Image.new("RGB", (800, 100), "white"), 1.0, config)
        self.assertNotIn("SOURCE_PROVENANCE_INCOMPLETE", [issue.code for issue in issues])

    def test_ignored_block_without_reason_has_incomplete_provenance(self) -> None:
        document = self._document(
            source_blocks=[
                SourceBlock("source_0001", "text", "a+b", BBox(1, 1, 40, 20), 0.99),
                SourceBlock(
                    "ignored_0001",
                    "text",
                    "不明内容",
                    BBox(50, 10, 60, 20),
                    0.40,
                    disposition="ignored",
                ),
            ]
        )
        recognition = RecognitionResult(provider="fixture", transcript="a+b")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(document, recognition, Image.new("RGB", (800, 100), "white"), 1.0, config)
        self.assertIn("SOURCE_PROVENANCE_INCOMPLETE", [issue.code for issue in issues])

    def test_trailing_answer_cue_without_blank_cannot_pass(self) -> None:
        document = self._document(
            canonical_text="可估计总体方差为",
            display_text="可估计总体方差为",
            stem="可估计总体方差为",
            formula_count=0,
            formula_specialist_used=False,
            formula_verified=True,
            source_blocks=[
                SourceBlock("source_0001", "text", "可估计总体方差为", BBox(1, 1, 100, 20), 0.99)
            ],
        )
        recognition = RecognitionResult(provider="fixture", transcript="可估计总体方差为")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(document, recognition, Image.new("RGB", (800, 100), "white"), 1.0, config)
        self.assertIn("ANSWER_SLOT_UNVERIFIED", [issue.code for issue in issues])

    def test_automatic_text_and_option_corrections_remain_review_gated(self) -> None:
        document = self._document(
            warnings=[
                "STACKED_TEXT_ROWS_SPLIT:1",
                "OPTION_FORMULA_SUFFIX_REMOVED_BY_SECONDARY_OCR:8:0.980",
                "STACKED_FRACTION_REFINED_FROM_SOURCE_CROP:0.720",
                "MISSING_OPTION_A_RECOVERED_BY_SECONDARY_OCR:0.980",
                "OPTION_SEQUENCE_RECOVERED_FROM_COHERENT_LABELS:BD",
                "QUESTION_NUMBER_PUNCTUATION_RECOVERED:10",
                "QUESTION_NUMBER_RECOVERED_FROM_VISUAL_CAPTION:10",
            ]
        )
        recognition = RecognitionResult(provider="fixture", transcript="D. 8")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(document, recognition, Image.new("RGB", (800, 100), "white"), 1.0, config)
        codes = [issue.code for issue in issues]
        self.assertIn("STACKED_TEXT_ROWS_SPLIT_UNVERIFIED", codes)
        self.assertIn("OPTION_FORMULA_CORRECTED_UNVERIFIED", codes)
        self.assertIn("STACKED_FRACTION_REFINED_UNVERIFIED", codes)
        self.assertIn("OPTION_LABEL_RECOVERED_UNVERIFIED", codes)
        self.assertIn("QUESTION_HEADER_CORRECTED_UNVERIFIED", codes)

    def test_photo_visual_structure_recovery_stays_hard_rejected(self) -> None:
        document = self._document(
            question_number="10",
            warnings=[
                "PHOTO_PAGE_HANDWRITING_RISK:{}",
                "OPTION_SEQUENCE_RECOVERED_FROM_COHERENT_LABELS:A",
                "QUESTION_NUMBER_PUNCTUATION_RECOVERED:10",
            ],
            visual_assets=[
                VisualAsset(
                    asset_id="asset_001",
                    source_bbox=BBox(10, 20, 100, 80),
                    crop_path="assets/asset_001.png",
                    crop_sha256="0" * 64,
                    bbox_verified=False,
                )
            ],
        )
        recognition = RecognitionResult(provider="fixture", transcript="10 题干")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )
        issue = next(
            item
            for item in issues
            if item.code == "PHOTO_VISUAL_STRUCTURE_RECOVERY_UNVERIFIED"
        )
        self.assertEqual(issue.severity, "reject")

    def test_geometry_recovered_option_labels_are_review_and_photo_visual_gated(self) -> None:
        document = self._document(
            question_number="3",
            warnings=[
                "PHOTO_PAGE_HANDWRITING_RISK:{}",
                "MISSING_TWO_COLUMN_OPTION_LABEL_RESTORED:2",
            ],
            visual_assets=[
                VisualAsset(
                    asset_id="asset_001",
                    source_bbox=BBox(10, 20, 100, 80),
                    crop_path="assets/asset_001.png",
                    crop_sha256="0" * 64,
                    bbox_verified=False,
                )
            ],
        )
        recognition = RecognitionResult(provider="fixture", transcript="3. 题干")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")

        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )

        codes = [issue.code for issue in issues]
        self.assertIn("OPTION_LABEL_RECOVERED_UNVERIFIED", codes)
        self.assertIn("PHOTO_VISUAL_STRUCTURE_RECOVERY_UNVERIFIED", codes)

    def test_long_low_confidence_text_region_is_a_hard_reject(self) -> None:
        document = self._document(
            question_number="7",
            warnings=["LOW_CONFIDENCE_LONG_TEXT_REGION:1:25"],
        )
        recognition = RecognitionResult(provider="fixture", transcript="7. 题干")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )
        issue = next(item for item in issues if item.code == "OCR_TEXT_CONFIDENCE_TOO_LOW")
        self.assertEqual(issue.severity, "reject")

    def test_failed_stacked_fraction_refinement_is_a_hard_reject(self) -> None:
        document = self._document(
            question_number="9",
            warnings=["STACKED_FRACTION_REFINEMENT_FAILED:TimeoutError"],
        )
        recognition = RecognitionResult(provider="fixture", transcript="9. 逃逸速度为 v'= 2Gm R")
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        issues = validate_result(
            document,
            recognition,
            Image.new("RGB", (800, 100), "white"),
            1.0,
            config,
        )
        issue = next(
            item for item in issues if item.code == "STACKED_FRACTION_REFINEMENT_FAILED"
        )
        self.assertEqual(issue.severity, "reject")


if __name__ == "__main__":
    unittest.main()
