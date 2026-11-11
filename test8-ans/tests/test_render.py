from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_schema import valid_answer


class RenderTests(unittest.TestCase):
    def test_public_titles_and_all_internal_n_labels_are_hidden(self) -> None:
        from test8_ans.render import TIER_TITLES, _tier_lines

        answer = valid_answer()
        self.assertEqual(TIER_TITLES["l5"], "基础解析")
        for label in ("A", "1", "（1）", "demo-generation-001"):
            answer["l5"][0]["n"] = label
            self.assertNotIn(label, _tier_lines(answer, "l5"))

    def test_existing_pitfall_label_is_not_duplicated(self) -> None:
        from test8_ans.render import _tier_lines

        answer = valid_answer()
        answer["l5"][0]["pit"] = "易错点：不要漏掉检验。"
        lines = _tier_lines(answer, "l5")
        self.assertIn("易错点：不要漏掉检验。", lines)
        self.assertNotIn("易错点：易错点：不要漏掉检验。", lines)

    def test_loose_string_and_pitfall_array_are_renderable(self) -> None:
        from test8_ans.render import _tier_lines, assess_renderability

        answer = valid_answer()
        answer["l1"][0]["s"] = "字符串形式的引导问题？"
        answer["l5"][0]["pit"] = ["不要漏掉单位。", "注意符号方向。"]
        lines = _tier_lines(answer, "l5")
        self.assertIn("易错点：不要漏掉单位。", lines)
        self.assertIn("易错点：注意符号方向。", lines)
        self.assertTrue(assess_renderability(answer)["valid"])

    def test_public_text_does_not_leak_latex_source(self) -> None:
        from test8_ans.render import humanize_math_markup

        source = r"因此，$^{14} \\[ C$ 的含量满足 $N=N_0\\left(\\frac{1}{2}\\right)^{t/T_{1/2}}$。"
        visible = humanize_math_markup(source)
        self.assertIn("¹⁴ C", visible)
        self.assertIn("(1)/(2)", visible)
        self.assertNotIn("$", visible)
        self.assertNotIn("\\", visible)
        self.assertNotIn("^{", visible)
        self.assertNotIn("_{", visible)

    def test_five_images_are_exactly_800px_wide(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        from test8_ans.render import render_tiers

        with tempfile.TemporaryDirectory() as directory:
            paths = render_tiers(valid_answer(), Path(directory), width=800)
            self.assertEqual(len(paths), 5)
            for path in paths:
                with Image.open(path) as image:
                    self.assertEqual(image.width, 800)
                    self.assertGreater(image.height, 0)

    def test_raw_model_answer_screenshot_is_800px_wide(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed")
        from test8_ans.render import render_model_output_screenshot

        with tempfile.TemporaryDirectory() as directory:
            path = render_model_output_screenshot(
                '{\n  "l1": [{"n": "1", "s": "示例回答"}]\n}',
                Path(directory) / "model_output.png",
            )
            with Image.open(path) as image:
                self.assertEqual(image.width, 800)
                self.assertGreater(image.height, 0)

    def test_benchmark_renderer_reads_only_redacted_display_copy(self) -> None:
        from test8_ans.render import render_benchmark

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            raw = valid_answer()
            raw["l5"][0]["s"][2] = "最终可得$x=4$。"
            display = valid_answer()
            display["l5"][0]["s"].pop(2)
            (run_dir / "answer.json").write_text(
                json.dumps(raw, ensure_ascii=False), encoding="utf-8"
            )
            (run_dir / "display_answer.json").write_text(
                json.dumps(display, ensure_ascii=False), encoding="utf-8"
            )
            summary_path = root / "benchmark_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "phase": "smoke",
                        "promotion_gate_pass": True,
                        "results": [
                            {
                                "id": "math-001",
                                "run_dir": str(run_dir),
                                "quality_gate_pass": True,
                                "display_ready": True,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            captured: list[dict] = []

            def fake_render(answer: dict, output_dir: Path, *, width: int) -> list[Path]:
                captured.append(answer)
                return []

            with patch("test8_ans.render.render_tiers", side_effect=fake_render):
                report = render_benchmark(summary_path, root / "rendered")
            self.assertEqual(report["rendered_count"], 1)
            self.assertNotIn("x=4", json.dumps(captured[0], ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
