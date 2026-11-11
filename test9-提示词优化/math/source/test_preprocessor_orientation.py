#!/usr/bin/env python3
"""Regression for a photographed wide worksheet misclassified as sideways."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from preprocessor.processor import QuestionImageProcessor


class PreprocessorOrientationTests(unittest.TestCase):
    def test_dark_viewer_toolbar_detects_underexposed_screen_photo(self) -> None:
        screen = np.full((600, 900, 3), 150, dtype=np.uint8)
        screen[40:72, :] = 18
        capture_type, confidence = QuestionImageProcessor._classify_capture_type(
            screen
        )
        self.assertEqual(capture_type, "screen-photo")
        self.assertGreaterEqual(confidence, 0.75)

        paper = np.full((600, 900, 3), 230, dtype=np.uint8)
        cv2.line(paper, (100, 80), (800, 80), (20, 20, 20), 3)
        paper_type, _ = QuestionImageProcessor._classify_capture_type(paper)
        self.assertNotEqual(paper_type, "screen-photo")

    def test_many_horizontal_rows_override_wrong_sideways_document_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = QuestionImageProcessor(Path(temp_dir))
            scores = iter(((0.96, 15), (0.55, 5), (0.22, 15), (0.92, 5)))
            rows = iter(([(5, 5, 160, 12)], [], [(5, 5, 160, 12)], []))
            page = np.full((100, 200, 3), 255, dtype=np.uint8)
            with patch.object(processor, "_document_axis", return_value=(90, 0.61)), patch.object(
                processor, "_textline_upright_score", side_effect=lambda image: next(scores)
            ), patch.object(
                processor, "_detect_printed_text_lines", side_effect=lambda image: next(rows)
            ):
                angle, confidence, method = processor._auto_rotation(page)

        self.assertEqual(angle, 0)
        self.assertGreaterEqual(confidence, 0.9)
        self.assertEqual(method, "textline-axis+textline-model")

    def test_diagram_shapes_cannot_rotate_a_confident_upright_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = QuestionImageProcessor(Path(temp_dir))
            # Reproduces a landscape worksheet where house/circle geometry
            # yielded more false line crops after a 90-degree rotation.
            scores = iter(((0.995, 4), (0.970, 9), (0.309, 4), (0.897, 9)))
            rows = iter(([(5, 5, 160, 12)], [], [(5, 5, 160, 12)], []))
            page = np.full((100, 200, 3), 255, dtype=np.uint8)
            with patch.object(
                processor, "_document_axis", return_value=(0, 0.68)
            ), patch.object(
                processor, "_textline_upright_score", side_effect=lambda image: next(scores)
            ), patch.object(
                processor, "_detect_printed_text_lines", side_effect=lambda image: next(rows)
            ):
                angle, confidence, method = processor._auto_rotation(page)

        self.assertEqual(angle, 0)
        self.assertGreaterEqual(confidence, 0.9)
        self.assertEqual(method, "document+textline-model")

    def test_rotate_only_preserves_full_canvas_and_skips_page_crop(self) -> None:
        page = np.full((96, 180, 3), 236, dtype=np.uint8)
        cv2.putText(
            page, "FULL QUESTION", (3, 52), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (18, 18, 18), 1, cv2.LINE_AA,
        )
        encoded_ok, encoded = cv2.imencode(".png", page)
        self.assertTrue(encoded_ok)

        with tempfile.TemporaryDirectory() as temp_dir:
            processor = QuestionImageProcessor(
                Path(temp_dir), enable_reference_library=False
            )
            with patch.object(
                processor,
                "_document_warp",
                side_effect=AssertionError("rotate-only must not crop"),
            ), patch.object(
                processor, "_restore_from_reference",
                side_effect=AssertionError("rotate-only must not replace the page"),
            ), patch.object(
                processor, "_clean", side_effect=lambda image, mode: image.copy()
            ), patch.object(
                processor, "_write_pdf", return_value=None
            ), patch.object(
                processor, "build_analysis_reference", return_value=(None, {})
            ):
                result = processor.process(
                    encoded.tobytes(), "full-question.png", mode="color",
                    rotation="0", quality="original", extract_questions=False,
                    deskew_mode="off", geometry_mode="rotate-only",
                )
            actual = cv2.imread(str(Path(temp_dir) / result.png))

        self.assertEqual(actual.shape, page.shape)
        self.assertTrue(np.array_equal(actual, page))

    def test_large_deskew_requires_independent_frame_confirmation(self) -> None:
        page = np.full((100, 200, 3), 245, dtype=np.uint8)
        encoded_ok, encoded = cv2.imencode(".png", page)
        self.assertTrue(encoded_ok)

        with tempfile.TemporaryDirectory() as temp_dir:
            processor = QuestionImageProcessor(
                Path(temp_dir), enable_reference_library=False
            )
            with patch.object(
                processor, "_estimate_text_skew", return_value=(4.0, 0.94, "ready")
            ), patch.object(
                processor, "_estimate_frame_skew", return_value=(0.0, 0.95, "not-needed")
            ), patch.object(
                processor, "_rotate_bound",
                side_effect=AssertionError("unconfirmed large angle must not rotate"),
            ), patch.object(
                processor, "_clean", side_effect=lambda image, mode: image.copy()
            ), patch.object(
                processor, "_write_pdf", return_value=None
            ), patch.object(
                processor, "build_analysis_reference", return_value=(None, {})
            ):
                result = processor.process(
                    encoded.tobytes(), "diagram.png", mode="color", rotation="0",
                    quality="original", extract_questions=False,
                    deskew_mode="safe", geometry_mode="rotate-only",
                )

        self.assertEqual(result.deskew_status, "large-angle-unconfirmed")
        self.assertEqual(result.deskew_applied_angle, 0.0)

    def test_analysis_reference_is_deterministic_and_non_destructive(self) -> None:
        page = np.full((120, 220, 3), 247, dtype=np.uint8)
        original = page.copy()
        reading_order_lines = [
            np.full((11, 70, 3), 40, dtype=np.uint8),
            np.full((9, 60, 3), 100, dtype=np.uint8),
            np.full((13, 80, 3), 180, dtype=np.uint8),
        ]

        with patch.object(
            QuestionImageProcessor,
            "_extract_printed_lines",
            return_value=reading_order_lines,
        ):
            first, first_audit = QuestionImageProcessor.build_analysis_reference(page)
            second, second_audit = QuestionImageProcessor.build_analysis_reference(page)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertTrue(np.array_equal(page, original))
        self.assertFalse(np.shares_memory(first, page))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_audit, second_audit)
        self.assertEqual(first_audit["line_count"], 3)
        self.assertEqual(
            first_audit["generation_method"],
            "extract_printed_lines-reading-order-montage-v1",
        )
        self.assertEqual(
            first_audit["dimensions"],
            {"width": first.shape[1], "height": first.shape[0]},
        )
        self.assertFalse(first_audit["authoritative_output_modified"])
        self.assertFalse(first_audit["handwriting_erasure"])
        self.assertEqual(
            first_audit["handwriting_policy"], "selection-only-no-inpaint"
        )

        first_y = int(np.argwhere(np.all(first == 40, axis=2))[:, 0].min())
        second_y = int(np.argwhere(np.all(first == 100, axis=2))[:, 0].min())
        third_y = int(np.argwhere(np.all(first == 180, axis=2))[:, 0].min())
        self.assertLess(first_y, second_y)
        self.assertLess(second_y, third_y)

    def test_analysis_reference_returns_none_without_reliable_rows(self) -> None:
        page = np.full((80, 160, 3), 255, dtype=np.uint8)
        one_line = np.full((10, 80, 3), 80, dtype=np.uint8)

        with patch.object(
            QuestionImageProcessor,
            "_extract_printed_lines",
            return_value=[one_line],
        ):
            reference, audit = QuestionImageProcessor.build_analysis_reference(page)

        self.assertIsNone(reference)
        self.assertFalse(audit["generated"])
        self.assertEqual(audit["line_count"], 1)
        self.assertEqual(audit["dimensions"], {"width": 0, "height": 0})
        self.assertEqual(audit["reason"], "insufficient-reliable-printed-lines")
        self.assertFalse(audit["authoritative_output_modified"])
        self.assertFalse(audit["handwriting_erasure"])

    def test_analysis_reference_excludes_rows_with_large_nonprint_components(self) -> None:
        page = np.full((120, 220, 3), 255, dtype=np.uint8)

        def regular_row() -> np.ndarray:
            row = np.full((40, 180, 3), 255, dtype=np.uint8)
            for index in range(12):
                x = 5 + index * 12
                cv2.rectangle(row, (x, 14), (x + 3, 23), (20, 20, 20), -1)
            return row

        suspicious = regular_row()
        cv2.rectangle(suspicious, (120, 6), (168, 33), (20, 20, 20), -1)
        rows = [regular_row(), suspicious, regular_row()]

        with patch.object(
            QuestionImageProcessor,
            "_extract_printed_lines",
            return_value=rows,
        ):
            reference, audit = QuestionImageProcessor.build_analysis_reference(page)

        self.assertIsNotNone(reference)
        self.assertEqual(audit["raw_line_count"], 3)
        self.assertEqual(audit["line_count"], 2)
        self.assertEqual(audit["suspected_nonprint_line_count"], 1)
        self.assertEqual(audit["excluded_line_indexes"], [1])
        self.assertFalse(audit["handwriting_erasure"])

    def test_process_persists_reference_separately_from_authoritative_png(self) -> None:
        page = np.full((80, 160, 3), 255, dtype=np.uint8)
        cv2.putText(
            page,
            "PRINTED QUESTION",
            (8, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        encoded_ok, encoded = cv2.imencode(".png", page)
        self.assertTrue(encoded_ok)
        diagnostic = np.full((20, 60, 3), 231, dtype=np.uint8)
        audit = {
            "generated": True,
            "generation_method": "extract_printed_lines-reading-order-montage-v1",
            "line_count": 2,
            "dimensions": {"width": 60, "height": 20},
            "source_dimensions": {"width": 160, "height": 80},
            "authoritative_output_modified": False,
            "handwriting_erasure": False,
            "handwriting_policy": "selection-only-no-inpaint",
            "reason": "ok",
        }

        def mutate_received_copy(image: np.ndarray, **_kwargs):
            image[:] = 0
            return diagnostic.copy(), audit.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            processor = QuestionImageProcessor(
                Path(temp_dir), enable_reference_library=False
            )
            with patch.object(
                processor, "_document_warp", side_effect=lambda image: image.copy()
            ), patch.object(
                processor, "_restore_from_reference", return_value=(None, "", 0.0)
            ), patch.object(
                processor, "_reference_candidates", return_value=[]
            ), patch.object(
                processor, "_clean", side_effect=lambda image, mode: image.copy()
            ), patch.object(
                processor, "_write_pdf", return_value=None
            ), patch.object(
                processor,
                "build_analysis_reference",
                side_effect=mutate_received_copy,
            ):
                result = processor.process(
                    encoded.tobytes(),
                    "screen-photo.png",
                    mode="color",
                    rotation="0",
                    quality="original",
                    extract_questions=False,
                    deskew_mode="off",
                )

            authoritative = cv2.imread(str(Path(temp_dir) / result.png))
            persisted_reference = cv2.imread(
                str(Path(temp_dir) / result.analysis_reference)
            )

        self.assertTrue(np.array_equal(authoritative, page))
        self.assertTrue(np.array_equal(persisted_reference, diagnostic))
        self.assertNotEqual(result.analysis_reference, result.png)
        self.assertTrue(result.analysis_reference.startswith("analysis_reference/"))
        self.assertTrue(result.analysis_reference_audit["persisted"])
        self.assertFalse(
            result.analysis_reference_audit["authoritative_output_modified"]
        )
        self.assertFalse(result.analysis_reference_audit["handwriting_erasure"])


if __name__ == "__main__":
    unittest.main()
