from __future__ import annotations

from dataclasses import dataclass
import unittest

import numpy as np

from formula_spatial_fusion import (
    fuse_formula_items,
    has_flattened_script_tokens,
    strip_flattened_neighbors_of_latex,
    suppress_overlapping_duplicate_items,
)


@dataclass
class FakeItem:
    box: np.ndarray
    text: str
    score: float = 0.98
    printed: bool = True
    formula_like: bool = False
    width: float = 0.0
    height: float = 0.0
    center_x: float = 0.0
    center_y: float = 0.0


def rect(left: float, top: float, right: float, bottom: float) -> np.ndarray:
    return np.asarray(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float32,
    )


def make_item(box, text, score, *, formula_like: bool) -> FakeItem:
    array = np.asarray(box, dtype=np.float32)
    left, top = float(array[:, 0].min()), float(array[:, 1].min())
    right, bottom = float(array[:, 0].max()), float(array[:, 1].max())
    return FakeItem(
        box=array,
        text=text,
        score=float(score),
        printed=True,
        formula_like=formula_like,
        width=max(1.0, right - left),
        height=max(1.0, bottom - top),
        center_x=(left + right) / 2,
        center_y=(top + bottom) / 2,
    )


def line_item(text: str, left: float, top: float, right: float, bottom: float) -> FakeItem:
    return make_item(rect(left, top, right, bottom), text, 0.98, formula_like=False)


class FormulaSpatialFusionTests(unittest.TestCase):
    def test_structure_tokens_do_not_hide_flattened_scripts(self) -> None:
        coin = (
            "14. 独立重复抛掷一枚质地均匀硬币,每次抛出正面和反面的概率均为 1,"
            "抛掷过程中记录累计正面次数Hn和反面次数Tn(初始H。=T=0)."
        )
        function = (
            "已知函数 f(x) =lnx+ax− ex bx "
            "三个极值点x,x2,x3(x<x2<x3). 直线1,l"
        )
        geometry = (
            "25. 如图，△ABC中，D、E分别为边AB、AC上一点，且DC=DE。"
            "(1)求证：∠A=∠DCF；(2)若∠ACB=90°，①求弦DF长度。"
        )
        self.assertTrue(has_flattened_script_tokens(coin))
        self.assertTrue(has_flattened_script_tokens(function))
        self.assertFalse(has_flattened_script_tokens(geometry))

    def test_coin_line_replaces_flattened_span_and_keeps_chinese(self) -> None:
        source = [
            line_item(
                "① 累计正面次数满足Hn≥2T+2,此时判定正面获胜;",
                50,
                245,
                900,
                290,
            )
        ]
        audit = {
            "formula_items": [
                {
                    "bbox": [287, 252, 405, 281],
                    "latex": r"H_{n} \geq 2T_{n} + 2",
                    "score": 0.95,
                }
            ]
        }

        refined, replaced = fuse_formula_items(
            source,
            audit,
            make_item=make_item,
            normalize_text=lambda tokens: " ".join(tokens),
        )
        texts = [item.text for item in refined]

        self.assertEqual(replaced, 1)
        self.assertTrue(any(r"H_{n} \geq 2T_{n} + 2" in text for text in texts))
        self.assertTrue(any("累计正面次数满足" in text for text in texts))
        self.assertTrue(any("此时判定正面获胜" in text for text in texts))
        self.assertFalse(any("Hn≥2T+2" in text for text in texts))

    def test_function_line_replaces_flattened_formula_without_duplicating(self) -> None:
        source = [
            line_item(
                "已知函数 f(x) =lnx+ax− ex bx",
                40,
                140,
                520,
                210,
            )
        ]
        audit = {
            "formula_items": [
                {
                    "bbox": [202, 154, 482, 205],
                    "latex": r"f(x)=\ln x+ax-\frac{bx}{\mathrm{e}^{x}}",
                    "score": 0.95,
                }
            ]
        }

        refined, replaced = fuse_formula_items(
            source,
            audit,
            make_item=make_item,
            normalize_text=lambda tokens: " ".join(tokens),
        )
        joined = " ".join(item.text for item in refined)

        self.assertEqual(replaced, 1)
        self.assertIn(r"\frac{bx}{\mathrm{e}^{x}}", joined)
        self.assertIn("已知函数", joined)
        self.assertNotIn("lnx+ax", joined)

    def test_unanchored_formula_is_still_rejected(self) -> None:
        source = [line_item("2x>12", 70, 78, 170, 102)]
        audit = {
            "formula_items": [
                {"bbox": [65, 75, 175, 105], "latex": r"2^{x}>\frac{1}{2}", "score": 0.96},
                {"bbox": [420, 300, 520, 340], "latex": r"x=7", "score": 0.99},
            ]
        }

        refined, replaced = fuse_formula_items(
            source,
            audit,
            make_item=make_item,
            normalize_text=lambda tokens: " ".join(tokens),
        )

        self.assertEqual(replaced, 1)
        self.assertTrue(any(r"\frac{1}{2}" in item.text for item in refined))
        self.assertFalse(any("x=7" in item.text for item in refined))

    def test_overlapping_duplicate_text_is_suppressed(self) -> None:
        long = make_item(
            rect(100, 80, 400, 110),
            "累计正面次数满足$H_{n} \\geq 2T_{n} + 2$",
            0.97,
            formula_like=True,
        )
        short = make_item(
            rect(110, 82, 280, 108),
            "累计正面次数满足",
            0.90,
            formula_like=False,
        )

        kept = suppress_overlapping_duplicate_items([long, short])
        texts = [item.text for item in kept]
        self.assertEqual(sum("累计正面次数满足" in text for text in texts), 1)

    def test_strip_flattened_neighbor_of_latex(self) -> None:
        cleaned = strip_flattened_neighbors_of_latex(
            r"每次抛出正面和反面的概率均为 $\frac{1}{2}$ 1,抛掷过程中记录累"
        )
        self.assertIn(r"$\frac{1}{2}$", cleaned)
        self.assertNotIn(r"$\frac{1}{2}$ 1", cleaned)
        self.assertIn("抛掷过程中记录累", cleaned)


if __name__ == "__main__":
    unittest.main()
