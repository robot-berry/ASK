#!/usr/bin/env python3
"""Regression for a photographed wide worksheet misclassified as sideways."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from preprocessor.processor import QuestionImageProcessor


class PreprocessorOrientationTests(unittest.TestCase):
    def test_many_horizontal_rows_override_wrong_sideways_document_axis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = QuestionImageProcessor(Path(temp_dir))
            scores = iter(((0.96, 15), (0.55, 5), (0.22, 15), (0.92, 5)))
            page = np.full((100, 200, 3), 255, dtype=np.uint8)
            with patch.object(processor, "_document_axis", return_value=(90, 0.61)), patch.object(
                processor, "_textline_upright_score", side_effect=lambda image: next(scores)
            ):
                angle, confidence, method = processor._auto_rotation(page)

        self.assertEqual(angle, 0)
        self.assertGreaterEqual(confidence, 0.9)
        self.assertEqual(method, "textline-axis+textline-model")


if __name__ == "__main__":
    unittest.main()
