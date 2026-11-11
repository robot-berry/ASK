import unittest

from test7_ocr.types import QuestionDocument, RecognitionResult
from test7_ocr.verification import verify_render_readback


class RenderReadbackTests(unittest.TestCase):
    def test_readback_accepts_line_leading_contiguous_joined_option_labels(self) -> None:
        document = self._document()
        readback = RecognitionResult(
            provider="fixture",
            transcript="question\nA. one\nB. two\nCD. three four",
        )
        result = verify_render_readback(document, readback, minimum_coverage=0.0)
        self.assertEqual(result["found_option_labels"], ["A", "B", "C", "D"])
        self.assertNotIn("RENDER_READBACK_OPTION_LOSS", result["issues"])

    def test_readback_accepts_lowercase_option_label_ocr_confusion(self) -> None:
        document = self._document()
        readback = RecognitionResult(
            provider="fixture",
            transcript="question\nA. one\nB. two\nc. three\nD. four",
        )
        result = verify_render_readback(document, readback, minimum_coverage=0.0)
        self.assertEqual(result["found_option_labels"], ["A", "B", "C", "D"])
        self.assertNotIn("RENDER_READBACK_OPTION_LOSS", result["issues"])

    def test_body_letter_pair_does_not_count_as_joined_option_labels(self) -> None:
        document = self._document()
        readback = RecognitionResult(
            provider="fixture",
            transcript="question AB. variable\nA. one\nB. two",
        )
        result = verify_render_readback(document, readback, minimum_coverage=0.0)
        self.assertEqual(result["missing_option_labels"], ["C", "D"])

    def _document(self) -> QuestionDocument:
        return QuestionDocument(
            schema_version="question_document_v1",
            sample_id="sample",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="1. 选择 A. 甲 B. 乙 C. 丙 D. 丁",
            display_text="1. 选择 A. 甲 B. 乙 C. 丙 D. 丁",
            question_number="1",
            stem="1. 选择",
            options=[{"label": label, "text": text} for label, text in zip("ABCD", "甲乙丙丁")],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
        )

    def test_readback_passes_complete_options(self) -> None:
        result = verify_render_readback(
            self._document(),
            RecognitionResult(provider="readback", transcript="1.选择 A.甲 B.乙 C.丙 D.丁"),
            0.70,
        )
        self.assertTrue(result["passed"])

    def test_readback_rejects_missing_option_and_latex_token(self) -> None:
        result = verify_render_readback(
            self._document(),
            RecognitionResult(provider="readback", transcript=r"1.选择 A.甲 B.乙 frac"),
            0.70,
        )
        self.assertIn("RENDER_READBACK_OPTION_LOSS", result["issues"])
        self.assertIn("RENDER_READBACK_LATEX_TOKEN", result["issues"])

    def test_extra_text_from_a_direct_visual_does_not_reduce_body_coverage(self) -> None:
        document = self._document()
        result = verify_render_readback(
            document,
            RecognitionResult(
                provider="readback",
                transcript=(
                    "1.选择 A.甲 B.乙 C.丙 D.丁\n"
                    "自前级输入 L C 向后级输出 电路图标签"
                ),
            ),
            0.95,
        )
        self.assertEqual(result["coverage_ratio"], 1.0)
        self.assertTrue(result["passed"])

    def test_latex_layout_commands_are_not_expected_as_visible_letters(self) -> None:
        document = self._document()
        document.display_text = (
            r"1. 已知 $\triangle ABC$，$\tan \frac{B}{2}=3\tan \frac{C}{2}$。"
            " A.1 B.2 C.3 D.4"
        )
        document.options = [
            {"label": label, "text": text}
            for label, text in zip("ABCD", "1234")
        ]
        result = verify_render_readback(
            document,
            RecognitionResult(
                provider="readback",
                transcript="1.已知△ABC，tan B/2=3 tan C/2。 A.1 B.2 C.3 D.4",
            ),
            0.95,
        )
        self.assertEqual(result["coverage_ratio"], 1.0)
        self.assertTrue(result["passed"])

    def test_readable_radicals_and_blank_slots_do_not_create_false_expected_text(self) -> None:
        document = self._document()
        document.display_text = "17. 数列 √S₁+√Sₙ+…，r/R=______"
        document.options = []

        result = verify_render_readback(
            document,
            RecognitionResult(
                provider="readback",
                transcript="17. 数列 √S1+√Sn，r/R=",
            ),
            0.72,
        )

        self.assertTrue(result["passed"], result)

    def test_missing_expected_body_still_fails_directional_coverage(self) -> None:
        result = verify_render_readback(
            self._document(),
            RecognitionResult(provider="readback", transcript="A.甲 B.乙 C.丙 D.丁"),
            0.95,
        )
        self.assertIn("RENDER_READBACK_LOW_COVERAGE", result["issues"])


if __name__ == "__main__":
    unittest.main()
