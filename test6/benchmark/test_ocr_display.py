from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from presentation_guard import (
    contains_source_markup,
    readable_ocr_text,
    source_markup_paths,
)


class OcrDisplayTests(unittest.TestCase):
    def test_converts_reported_probability_question_without_source_markup(self) -> None:
        raw = (
            r"5. 若随机变量 $X \sim N (\mu, 4),$ 随机变量Y "
            r"$\sim B\left(n,\frac{1}{2} \right),\quad "
            r"P\left(X \geq 1 \right) = \frac{1}{2} 且 "
            r"E\left(X \right) = E\left(Y \right),$ 则 $D(Y) =$"
            "\n"
            r"$\mathrm { ~ A. ~ } \frac { 1 } { 4 }$ B. 1/2 C. 2 D. 4"
        )
        visible = readable_ocr_text(raw)
        self.assertIn("X ~ N (μ, 4)", visible)
        self.assertIn("B(n,1/2)", visible)
        self.assertIn("P(X ≥ 1) = 1/2", visible)
        self.assertIn("A. 1/4", visible)
        self.assertFalse(contains_source_markup(visible))

    def test_nested_fraction_root_and_scripts_remain_semantically_readable(self) -> None:
        visible = readable_ocr_text(
            r"$y=\frac{\sqrt{5}}{2}x$，$a_{n+1}=2^{n}$"
        )
        self.assertEqual(visible, "y=√5/2x，aₙ₊₁=2ⁿ")
        self.assertFalse(contains_source_markup(visible))

    def test_unknown_command_does_not_expose_backslash(self) -> None:
        visible = readable_ocr_text(r"$A \mystery B$")
        self.assertEqual(visible, "A mystery B")
        self.assertFalse(contains_source_markup(visible))

    def test_public_conversion_is_idempotent_across_backend_and_portal(self) -> None:
        once = readable_ocr_text(r"$X \sim N(\mu,4)$")
        twice = readable_ocr_text(once)
        self.assertEqual(once, "X ~ N(μ,4)")
        self.assertEqual(twice, once)

    def test_recursive_guard_reports_only_leaking_public_fields(self) -> None:
        payload = {
            "safe": "P(A)=m/n",
            "tiers": [{"steps": [r"使用 $\frac{m}{n}$"]}],
        }
        self.assertEqual(
            source_markup_paths(payload),
            ["$.tiers[0].steps[0]"],
        )


if __name__ == "__main__":
    unittest.main()
