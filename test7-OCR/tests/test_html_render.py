from __future__ import annotations

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image

from test7_ocr.html_render import (
    _build_html,
    _decode_layout_markers,
    _stem_markup,
    edge_render_enabled,
    render_question_katex,
)
from test7_ocr.pipeline import load_config
from test7_ocr.types import BBox, QuestionDocument, VisualAsset


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class HtmlRenderTests(unittest.TestCase):
    def test_edge_render_policy_can_include_text_only_documents(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="text-only",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="1. 纯文字题",
            display_text="1. 纯文字题",
            question_number="1",
            stem="1. 纯文字题",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        self.assertTrue(edge_render_enabled(document, config))
        config["rendering"]["edge_for_text_only"] = False
        self.assertFalse(edge_render_enabled(document, config))
        document.formula_specialist_used = True
        self.assertTrue(edge_render_enabled(document, config))
        config["rendering"]["engine"] = "pillow"
        self.assertFalse(edge_render_enabled(document, config))

    def test_short_structured_options_render_as_two_column_grid(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="fixture",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                r"1. 请选择 $x$ A. $(-2,0)$ B. $(-1,+\infty)$ "
                r"C. $(-1,0)$ D. $(-\infty,-2)$"
            ),
            display_text="1. 请选择 x",
            question_number="1",
            stem="1. 请选择 x",
            options=[
                {"label": "A", "text": "(-2,0)"},
                {"label": "B", "text": "(-1,+∞)"},
                {"label": "C", "text": "(-1,0)"},
                {"label": "D", "text": "(-∞,-2)"},
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
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=5,
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        built = _build_html(document, PROJECT_ROOT / "vendor" / "katex", config)
        self.assertIn('id="options" class="options-2"', built)
        self.assertEqual(built.count('class="option"'), 4)
        self.assertEqual(built.count('class="option-label"'), 4)
        self.assertEqual(built.count('class="option-content"'), 4)
        self.assertNotIn("options-4", built)
        self.assertIn('src="assets/asset_001.png" alt="原题裁图 1"', built)
        self.assertIn("visibleLatexTokenCount", built)
        self.assertIn("visibleTextOutsideKatex", built)
        self.assertIn("#stem{white-space:pre-wrap;text-wrap:pretty}", built)
        self.assertIn(".option{min-width:0;white-space:pre-wrap;text-wrap:pretty}", built)
        self.assertIn("applyAdaptiveStemWrap", built)
        self.assertIn("if(!false){return 'pretty';}", built)
        self.assertIn("stem_text_wrap", str(config["typography"]))
        self.assertEqual(config["typography"]["stem_text_wrap"], "pretty")
        self.assertIn("stemLineMetrics", built)
        self.assertIn("katexHtml.children", built)
        self.assertIn("child.classList.contains('base')", built)
        self.assertIn("layout-min-fill-marker", built)
        self.assertIn("layout-internal-fill-marker", built)

    def test_geometry_proven_visual_options_keep_direct_crops_and_labels_together(self) -> None:
        options = [
            {"label": "A", "text": ""},
            {"label": "B", "text": ""},
            {"label": "C", "text": ""},
            {"label": "D", "text": "正面"},
        ]
        assets = [
            VisualAsset(
                asset_id=f"asset_{index:03d}",
                source_bbox=BBox(x, 320 if index < 5 else 280, 100, 105),
                crop_path=f"assets/asset_{index:03d}.png",
                crop_sha256="0" * 64,
                bbox_verified=False,
            )
            for index, x in enumerate((230, 390, 580, 770, 960), start=1)
        ]
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="visual-options",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="2. 如右图，主视图是 A. B. C. D. 正面",
            display_text="2. 如右图，主视图是 A. B. C. D. 正面",
            question_number="2",
            stem="2. 如右图，主视图是",
            options=options,
            visual_assets=assets,
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        built = _build_html(document, PROJECT_ROOT / "vendor" / "katex", config)
        self.assertIn('id="visual-options"', built)
        self.assertEqual(built.count('class="visual-option"'), 4)
        self.assertIn('src="assets/asset_005.png" alt="原题题图"', built)
        self.assertIn("<figcaption>正面</figcaption>", built)
        for index, label in enumerate("ABCD", start=1):
            self.assertIn(f">{label}.</span><img src=\"assets/asset_{index:03d}.png\"", built)
        self.assertNotIn('alt="原题裁图', built)

    def test_compact_multi_visuals_preserve_source_relative_layout(self) -> None:
        assets = [
            VisualAsset(
                asset_id="asset_001",
                source_bbox=BBox(100, 200, 220, 160),
                crop_path="assets/asset_001.png",
                crop_sha256="0" * 64,
            ),
            VisualAsset(
                asset_id="asset_002",
                source_bbox=BBox(360, 180, 180, 180),
                crop_path="assets/asset_002.png",
                crop_sha256="0" * 64,
            ),
        ]
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="source-layout",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="1. 如图所示",
            display_text="1. 如图所示",
            question_number="1",
            stem="1. 如图所示",
            options=[],
            visual_assets=assets,
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")

        built = _build_html(document, PROJECT_ROOT / "vendor" / "katex", config)

        self.assertIn('class="source-visual-layout"', built)
        self.assertEqual(built.count('class="source-visual-item"'), 2)
        self.assertIn('style="left:0px;top:20px;width:220px"', built)
        self.assertIn('style="left:260px;top:0px;width:180px"', built)

    def test_layout_metric_markers_decode_without_affecting_visible_content(self) -> None:
        image = Image.new("RGB", (800, 20), "white")
        image.putpixel((794, 19), (192, 53, 187))
        image.putpixel((795, 19), (4, 73, 191))
        image.putpixel((796, 19), (128, 109, 197))
        image.putpixel((797, 19), (1, 137, 211))
        metadata = _decode_layout_markers(image, 19)
        self.assertEqual(metadata["stem_line_count"], 4)
        self.assertEqual(metadata["stem_min_line_fill_ratio"], round(128 / 255, 4))
        self.assertEqual(
            metadata["stem_min_internal_line_fill_ratio"],
            round(192 / 255, 4),
        )
        self.assertEqual(metadata["stem_text_wrap_applied"], "balance")

    def test_structural_stem_rows_are_metric_segments(self) -> None:
        markup = _stem_markup("题干 $x^2$\n（1）第一问\n（2）第二问")
        self.assertEqual(markup.count('class="stem-segment"'), 3)
        self.assertEqual(markup.count("<br>"), 2)
        unsafe = _stem_markup("题干 $x^2\n+1$")
        self.assertEqual(unsafe.count('class="stem-segment"'), 1)
        self.assertIn("\n", unsafe)

    def test_prose_after_multiline_formula_starts_on_a_new_visual_row(self) -> None:
        markup = _stem_markup(
            r"已知 $f(x)=\left\{\begin{aligned}x,&x\ge0,\\-x,&x<0\end{aligned}\right.$ 若f(a)=1"
        )
        self.assertEqual(markup.count('class="stem-segment"'), 2)
        self.assertIn("</span><br><span", markup)
        self.assertIn("若f(a)=1", markup)

    def test_answer_placeholder_is_rendered_as_a_real_blank_rule(self) -> None:
        markup = _stem_markup("读数为______cm")
        self.assertIn('class="answer-slot"', markup)
        self.assertNotIn("______", markup)

    def test_nested_roman_subquestions_keep_explicit_rows(self) -> None:
        from test7_ocr.text import reflow_text_for_render

        value = reflow_text_for_render(
            "（2）求参数范围\n（i）求最大值；\n（ii）证明结论。"
        )
        self.assertEqual(
            value.splitlines(),
            ["（2）求参数范围", "（i）求最大值；", "（ii）证明结论。"],
        )

    def test_half_width_subquestion_markers_render_with_consistent_width(self) -> None:
        markup = _stem_markup("（1）第一问\n(2）第二问")
        self.assertIn("（1）第一问", markup)
        self.assertIn("（2）第二问", markup)
        self.assertNotIn("(2）", markup)

    def test_edge_subprocess_uses_configured_online_timeout(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="fixture",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=r"$x^2$",
            display_text="x²",
            question_number=None,
            stem="x²",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        expected_timeout = float(config["rendering"]["edge_timeout_seconds"])
        with TemporaryDirectory() as directory:
            with patch(
                "test7_ocr.html_render.subprocess.run",
                side_effect=subprocess.TimeoutExpired("edge", expected_timeout),
            ) as mocked:
                with self.assertRaises(subprocess.TimeoutExpired):
                    render_question_katex(document, Path(directory), config)
        self.assertEqual(mocked.call_args.kwargs["timeout"], expected_timeout)
        self.assertEqual(
            mocked.call_count,
            int(config["rendering"]["edge_timeout_retries"]) + 1,
        )
        command = mocked.call_args.args[0]
        self.assertFalse(
            any(str(argument).startswith("--virtual-time-budget=") for argument in command)
        )

    def test_edge_retries_share_the_remaining_end_to_end_budget(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="bounded-edge",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=r"$x^2$",
            display_text="x²",
            question_number=None,
            stem="x²",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        with TemporaryDirectory() as directory:
            with patch("test7_ocr.html_render.time.perf_counter", return_value=100.0):
                with patch(
                    "test7_ocr.html_render.subprocess.run",
                    side_effect=subprocess.TimeoutExpired("edge", 1.4),
                ) as mocked:
                    with self.assertRaises(subprocess.TimeoutExpired):
                        render_question_katex(
                            document,
                            Path(directory),
                            config,
                            deadline_monotonic=102.8,
                        )

        self.assertEqual(mocked.call_count, 2)
        timeouts = [call.kwargs["timeout"] for call in mocked.call_args_list]
        self.assertTrue(all(abs(value - 1.4) < 1e-9 for value in timeouts))

    def test_invalid_option_sequence_falls_back_to_unsplit_canonical_text(self) -> None:
        canonical = (
            r"5. 题干 $i_A$ A. $t_1$ 第一段 "
            r"$\left\langle \lambda_t \right\rangle_2$ D. $t_4$"
        )
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="invalid-options",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=canonical,
            display_text="5. 题干 iA A. t1 第一段 D. t4",
            question_number="5",
            stem="5. 题干",
            options=[
                {"label": "A", "text": "误切一"},
                {"label": "A", "text": "误切二"},
                {"label": "D", "text": "末项"},
            ],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=3,
        )
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        built = _build_html(document, PROJECT_ROOT / "vendor" / "katex", config)
        self.assertNotIn('id="options"', built)
        self.assertIn(r"\left\langle \lambda_t \right\rangle_2", built)


if __name__ == "__main__":
    unittest.main()
