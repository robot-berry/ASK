from __future__ import annotations

import unittest

from test7_ocr.output_normalization import normalize_document_for_output
from test7_ocr.text import latex_to_readable
from test7_ocr.types import BBox, QuestionDocument, SourceBlock, VisualAsset


def _asset(asset_id: str, bbox: BBox) -> VisualAsset:
    return VisualAsset(
        asset_id=asset_id,
        source_bbox=bbox,
        crop_path=f"assets/{asset_id}.png",
        crop_sha256="0" * 64,
        bbox_verified=False,
    )


class OutputNormalizationTests(unittest.TestCase):
    def test_missing_number_recovers_only_numeric_base_from_nearby_caption(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="caption-number-recovery",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="（2）用伏安法测量电阻。\n第14-II题图1\n第14-III题图2",
            display_text="（2）用伏安法测量电阻。\n第14-II题图1\n第14-III题图2",
            question_number=None,
            stem="（2）用伏安法测量电阻。",
            options=[],
            visual_assets=[_asset("asset_001", BBox(100, 100, 400, 200))],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("caption-1", "text", "第14-II题图1", BBox(150, 305, 130, 25), 0.94),
                SourceBlock("caption-2", "text", "第14-III题图2", BBox(310, 305, 140, 25), 0.93),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.question_number, "14")
        self.assertTrue(document.display_text.startswith("14. （2）"))
        self.assertTrue(document.canonical_text.startswith("14. （2）"))
        self.assertTrue(document.stem.startswith("14. （2）"))
        self.assertTrue(metadata["question_number_recovered_from_visual_caption"])
        self.assertIn(
            "QUESTION_NUMBER_RECOVERED_FROM_VISUAL_CAPTION:14",
            document.warnings,
        )

    def test_caption_number_is_not_recovered_from_weak_far_or_conflicting_evidence(self) -> None:
        for blocks in (
            [SourceBlock("weak", "text", "第24题", BBox(150, 305, 100, 25), 0.89)],
            [SourceBlock("far", "text", "第24题", BBox(900, 700, 100, 25), 0.99)],
            [
                SourceBlock("first", "text", "第14题图", BBox(150, 305, 100, 25), 0.99),
                SourceBlock("second", "text", "第15题图", BBox(300, 305, 100, 25), 0.99),
            ],
        ):
            with self.subTest(blocks=[block.text for block in blocks]):
                document = QuestionDocument(
                    schema_version="question_document_v2",
                    sample_id="unsafe-caption-number",
                    source_image="source.png",
                    source_sha256="0" * 64,
                    provider="fixture",
                    canonical_text="题干正文",
                    display_text="题干正文",
                    question_number=None,
                    stem="题干正文",
                    options=[],
                    visual_assets=[_asset("asset_001", BBox(100, 100, 400, 200))],
                    handwriting_detected=False,
                    formula_verified=True,
                    formula_specialist_used=False,
                    formula_count=0,
                    source_blocks=blocks,
                )

                metadata = normalize_document_for_output(document)

                self.assertIsNone(document.question_number)
                self.assertFalse(metadata["question_number_recovered_from_visual_caption"])

    def test_adjacent_captioned_question_visual_is_removed_without_redrawing(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="captioned-neighbors",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="12. 转盘概率为▲\n(第12题) (第13题)",
            display_text="12. 转盘概率为▲\n(第12题) (第13题)",
            question_number="12",
            stem="12. 转盘概率为▲\n(第12题) (第13题)",
            options=[],
            visual_assets=[
                _asset("asset_001", BBox(320, 300, 180, 190)),
                _asset("asset_002", BBox(670, 290, 280, 200)),
            ],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("text_1", "text", "(第12题)", BBox(350, 495, 120, 32)),
                SourceBlock("text_2", "text", "(第13题)", BBox(755, 488, 120, 34)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual([asset.asset_id for asset in document.visual_assets], ["asset_001"])
        self.assertEqual(metadata["adjacent_visuals_removed"], 1)
        self.assertNotIn("第12题", document.canonical_text)
        self.assertNotIn("第13题", document.display_text)
        self.assertIn("______", document.display_text)
        self.assertIn("ADJACENT_CAPTIONED_VISUAL_REMOVED:asset_002:13", document.warnings)

    def test_caption_reference_inside_prose_is_not_suppressed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="prose-reference",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="1. 如图甲所示，求速度。",
            display_text="1. 如图甲所示，求速度。",
            question_number="1",
            stem="1. 如图甲所示，求速度。",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("text_1", "text", "如图甲所示，求速度。", BBox(0, 0, 200, 30)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text, "1. 如图甲所示，求速度。")
        self.assertEqual(metadata["caption_blocks_suppressed"], 0)

    def test_extended_caption_is_kept_in_direct_crop_not_repeated_as_text(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="roman-caption",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="14-III. 请选择正确说法\n第14-III图",
            display_text="14-III. 请选择正确说法\n第14-III图",
            question_number="14-III",
            stem="14-III. 请选择正确说法\n第14-III图",
            options=[],
            visual_assets=[_asset("asset_001", BBox(20, 50, 300, 100))],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("caption", "text", "第14-III图", BBox(120, 145, 100, 25)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertNotIn("第14-III图", document.display_text)
        self.assertEqual(document.visual_assets[0].source_bbox.bottom, 176)
        self.assertEqual(metadata["caption_assets_expanded"], ["asset_001"])

    def test_standalone_caption_removal_preserves_in_prose_references(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="repeated-caption-reference",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="1. 如图1所示，沿图1翻折。\n图1",
            display_text="1. 如图1所示，沿图1翻折。\n图1",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[_asset("asset_001", BBox(20, 50, 300, 100))],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("caption", "text", "图1", BBox(120, 145, 100, 25)),
            ],
        )

        normalize_document_for_output(document)

        self.assertEqual(document.display_text, "1. 如图1所示，沿图1翻折。")

        normalize_document_for_output(document)

        self.assertEqual(document.display_text, "1. 如图1所示，沿图1翻折。")

    def test_overlapping_duplicate_long_text_is_suppressed_once(self) -> None:
        repeated = "D.图丁，利用油膜法估测油酸分子大小"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="duplicate-text",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=f"1. 请选择 {repeated} {repeated}",
            display_text=f"1. 请选择 {repeated} {repeated}",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("d1", "text", repeated, BBox(20, 100, 500, 30)),
                SourceBlock("d2", "text", repeated, BBox(22, 101, 498, 30)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text.count(repeated), 1)
        self.assertEqual(metadata["overlapping_duplicate_text_suppressed"], 1)

        normalize_document_for_output(document)

        self.assertEqual(document.display_text.count(repeated), 1)

    def test_short_and_contained_overlapping_duplicates_are_suppressed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="duplicate-fragments",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                "12. 请选择\nA. 质点在\nA. 质点在 t=3s 与与 t=7s\n"
                "B. 距离原\nB. 距离原点为1m\nC.\nC. 加速度向上"
            ),
            display_text=(
                "12. 请选择\nA. 质点在\nA. 质点在 t=3s 与与 t=7s\n"
                "B. 距离原\nB. 距离原点为1m\nC.\nC. 加速度向上"
            ),
            question_number="12",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("a1", "text", "A. 质点在", BBox(10, 50, 120, 25)),
                SourceBlock("a2", "text", "A. 质点在", BBox(11, 50, 120, 25)),
                SourceBlock("and1", "text", "与", BBox(180, 50, 20, 25)),
                SourceBlock("and2", "text", "与", BBox(181, 50, 20, 25)),
                SourceBlock("b1", "text", "B. 距离原", BBox(10, 85, 120, 25)),
                SourceBlock("b2", "text", "B. 距离原点为1m", BBox(10, 85, 220, 25)),
                SourceBlock("c1", "text", "C.", BBox(10, 120, 30, 25)),
                SourceBlock("c2", "text", "C.", BBox(10, 120, 30, 25)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text.count("A. 质点在"), 1)
        self.assertNotIn("与与", document.display_text)
        self.assertNotIn("B. 距离原\n", document.display_text)
        self.assertEqual(document.display_text.count("C."), 1)
        self.assertGreaterEqual(metadata["overlapping_duplicate_text_suppressed"], 4)

    def test_terminal_cursor_marginal_digit_and_exam_footer_are_suppressed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="terminal-noise",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="3. 请选择\nD. f(x)=ln x ST 8 浙江省2026届联考试题卷",
            display_text="3. 请选择\nD. f(x)=ln x ST 8 浙江省2026届联考试题卷",
            question_number="3",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=1,
            source_blocks=[
                SourceBlock("d", "text", "D.", BBox(600, 100, 40, 30)),
                SourceBlock("formula", "formula", "f(x)=ln x", BBox(650, 100, 145, 30)),
                SourceBlock("cursor", "text", "ST", BBox(810, 102, 28, 25)),
                SourceBlock("digit", "text", "8", BBox(1040, 104, 10, 22)),
                SourceBlock(
                    "footer", "text", "浙江省2026届联考试题卷", BBox(200, 180, 500, 20)
                ),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertNotIn("ST", document.display_text)
        self.assertNotIn(" 8 ", document.display_text)
        self.assertNotIn("试题卷", document.display_text)
        self.assertEqual(metadata["marginal_handwritten_digits_suppressed"], 1)
        self.assertEqual(metadata["terminal_cursor_tokens_suppressed"], 1)
        self.assertEqual(metadata["exam_page_footers_suppressed"], 1)

    def test_low_confidence_circled_dot_cursor_after_d_label_is_suppressed(self) -> None:
        cursor_formula = r"\stackrel { \circ } { \cdot } N \subseteq M ."
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="inline-cursor",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                r"1. 请选择\nA. $M \cup N$ B. $M \cap N$\n"
                rf"C. $M \subseteq N$ D ${cursor_formula}$"
            ),
            display_text=(
                "1. 请选择\nA. M ∪ N B. M ∩ N\n"
                "C. M ⊆ N D stackrel ° · N ⊆ M."
            ),
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=4,
            source_blocks=[
                SourceBlock("d", "text", "D", BBox(911, 524, 38, 40), confidence=1.0),
                SourceBlock(
                    "formula",
                    "formula",
                    cursor_formula,
                    BBox(953, 509, 204, 58),
                    confidence=0.5887,
                ),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn(r"D. $N \subseteq M .$", document.canonical_text)
        self.assertIn("D. N ⊆ M.", document.display_text)
        self.assertNotIn("stackrel", document.canonical_text)
        self.assertNotIn("° ·", document.display_text)
        self.assertEqual(metadata["terminal_cursor_tokens_suppressed"], 1)

    def test_missing_c_label_in_two_column_option_grid_is_restored(self) -> None:
        c_formula = r"f(x) = g(x)"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="two-column-missing-c",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                r"11. 题干 $f(x)=g(x)=x^2$ 则\n"
                r"A. $a$ B. $b$\n"
                rf"${c_formula}$ 当且仅当 $x \in I$ D. $d$"
            ),
            display_text=(
                "11. 题干 f(x)=g(x)=x² 则\n"
                "A. a B. b\n"
                "f(x) = g(x) 当且仅当 x ∈ I D. d"
            ),
            question_number="11",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=5,
            source_blocks=[
                SourceBlock("a", "text", "A.", BBox(150, 386, 26, 22)),
                SourceBlock("b", "text", "B.对任意", BBox(800, 372, 134, 31)),
                SourceBlock("c-content", "formula", c_formula, BBox(198, 458, 139, 35)),
                SourceBlock("d", "text", "D.", BBox(796, 454, 39, 31)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn(r"C. $f(x) = g(x)$", document.canonical_text)
        self.assertIn("C. f(x) = g(x)", document.display_text)
        self.assertEqual(metadata["missing_two_column_option_labels_restored"], 1)
        self.assertEqual([option["label"] for option in document.options], list("ABCD"))

    def test_missing_left_labels_in_strict_formula_grid_are_restored(self) -> None:
        formulae = [
            r"r _ { 2 } < r _ { 4 } < 0",
            r"r_{4} < r_{2} < 0",
            r"r _ { 4 } < r _ { 2 } < 0",
            r"r_{2} < r_{4} < 0",
        ]
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="two-column-missing-left-labels",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                f"3. 题干\n${formulae[0]}$ B. ${formulae[1]}$\n"
                f"${formulae[2]}$ D. ${formulae[3]}$"
            ),
            display_text=(
                f"3. 题干\n{latex_to_readable(formulae[0])} "
                f"B. {latex_to_readable(formulae[1])}\n"
                f"{latex_to_readable(formulae[2])} "
                f"D. {latex_to_readable(formulae[3])}"
            ),
            question_number="3",
            stem="题干",
            options=[
                {"label": "B", "text": latex_to_readable(formulae[1])},
                {"label": "D", "text": latex_to_readable(formulae[3])},
            ],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=4,
            source_blocks=[
                SourceBlock("a-content", "formula", formulae[0], BBox(231, 502, 202, 36), 0.9093),
                SourceBlock("b-label", "text", "B.", BBox(632, 506, 28, 26), 1.0),
                SourceBlock("b-content", "formula", formulae[1], BBox(674, 501, 208, 35), 0.9222),
                SourceBlock("c-content", "formula", formulae[2], BBox(230, 574, 203, 36), 0.9032),
                SourceBlock("d-label", "text", "D.", BBox(631, 578, 32, 28), 1.0),
                SourceBlock("d-content", "formula", formulae[3], BBox(677, 573, 206, 36), 0.9166),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(metadata["missing_two_column_option_labels_restored"], 2)
        self.assertIn("MISSING_TWO_COLUMN_OPTION_LABEL_RESTORED:2", document.warnings)
        self.assertEqual([option["label"] for option in document.options], list("ABCD"))
        self.assertIn(f"A. ${formulae[0]}$", document.canonical_text)
        self.assertIn(f"C. ${formulae[2]}$", document.canonical_text)

    def test_missing_left_labels_are_not_restored_when_any_option_content_is_absent(self) -> None:
        formulae = ["a_1", "b_1", "d_1"]
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="two-column-incomplete-content",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                f"3. 题干\n${formulae[0]}$ B. ${formulae[1]}$\nD. ${formulae[2]}$"
            ),
            display_text=(
                f"3. 题干\n{latex_to_readable(formulae[0])} "
                f"B. {latex_to_readable(formulae[1])}\n"
                f"D. {latex_to_readable(formulae[2])}"
            ),
            question_number="3",
            stem="题干",
            options=[
                {"label": "B", "text": latex_to_readable(formulae[1])},
                {"label": "D", "text": latex_to_readable(formulae[2])},
            ],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=3,
            source_blocks=[
                SourceBlock("a-content", "formula", formulae[0], BBox(231, 502, 202, 36), 0.91),
                SourceBlock("b-label", "text", "B.", BBox(632, 506, 28, 26), 1.0),
                SourceBlock("b-content", "formula", formulae[1], BBox(674, 501, 208, 35), 0.92),
                SourceBlock("d-label", "text", "D.", BBox(631, 578, 32, 28), 1.0),
                SourceBlock("d-content", "formula", formulae[2], BBox(677, 573, 206, 36), 0.92),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(metadata["missing_two_column_option_labels_restored"], 0)
        self.assertEqual([option["label"] for option in document.options], ["B", "D"])
        self.assertNotIn("A.", document.display_text)
        self.assertNotIn("C.", document.display_text)

    def test_missing_option_label_is_not_restored_without_two_column_geometry(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="single-column-no-inference",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=r"1. 题干\nA. $a$\nB. $b$\n$f(x)$\nD. $d$",
            display_text="1. 题干\nA. a\nB. b\nf(x)\nD. d",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=4,
            source_blocks=[
                SourceBlock("a", "text", "A.", BBox(100, 100, 30, 25)),
                SourceBlock("b", "text", "B.", BBox(100, 140, 30, 25)),
                SourceBlock("content", "formula", "f(x)", BBox(150, 180, 80, 30)),
                SourceBlock("d", "text", "D.", BBox(100, 220, 30, 25)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(metadata["missing_two_column_option_labels_restored"], 0)
        self.assertNotIn("C.", document.display_text)

    def test_full_width_next_question_after_complete_formula_options_is_suppressed(self) -> None:
        tail = "2. 已知空间中三个不同的平面，则下列说法正确的是"
        blocks = []
        for index, (label, x) in enumerate(zip("ABCD", (80, 350, 650, 930)), start=1):
            blocks.append(SourceBlock(f"label_{index}", "text", f"{label}.", BBox(x, 100, 40, 30)))
            blocks.append(SourceBlock(f"formula_{index}", "formula", f"f_{index}", BBox(x + 45, 98, 150, 35)))
        blocks.append(SourceBlock("next", "text", tail, BBox(30, 150, 1100, 28)))
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="post-option-next-question",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=f"1. 请选择 A. f1 B. f2 C. f3 D. f4\n{tail}",
            display_text=f"1. 请选择 A. f1 B. f2 C. f3 D. f4\n{tail}",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=True,
            formula_count=4,
            source_blocks=blocks,
        )

        metadata = normalize_document_for_output(document)

        self.assertNotIn(tail, document.display_text)
        self.assertEqual(metadata["full_width_post_option_lines_suppressed"], 1)

    def test_low_confidence_clipped_tail_below_one_row_options_is_suppressed(self) -> None:
        option_blocks = [
            SourceBlock("a", "text", "A. 调幅", BBox(61, 511, 106, 33), confidence=0.96),
            SourceBlock("b", "text", "B. 调谐", BBox(353, 511, 103, 33), confidence=0.96),
            SourceBlock("c", "text", "C. 调频", BBox(613, 510, 104, 34), confidence=0.97),
            SourceBlock("d", "text", "D. 解调", BBox(874, 511, 104, 33), confidence=0.95),
            SourceBlock("tail-1", "text", "W→ 1. ν .", BBox(62, 546, 838, 12), confidence=0.53),
            SourceBlock("tail-2", "text", "L.", BBox(923, 546, 208, 12), confidence=0.52),
        ]
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="clipped-post-option-tail",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="5. 这个过程叫作\nA. 调幅 B. 调谐 C. 调频 D. 解调\nW→ 1. ν. L.",
            display_text="5. 这个过程叫作\nA. 调幅 B. 调谐 C. 调频 D. 解调\nW→ 1. ν. L.",
            question_number="5",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=option_blocks,
        )

        metadata = normalize_document_for_output(document)

        self.assertTrue(document.display_text.endswith("D. 解调"))
        self.assertNotIn("W→", document.display_text)
        self.assertEqual(metadata["low_confidence_post_option_tails_suppressed"], 1)

    def test_text_below_right_side_visual_is_trimmed_after_complete_d_option(self) -> None:
        d_option = "D. 沿y轴正方向入射的粒子做匀速圆周运动"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="post-visual-tail",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=f"6. 请选择\n{d_option}激光h",
            display_text=f"6. 请选择\n{d_option}激光h",
            question_number="6",
            stem="",
            options=[],
            visual_assets=[_asset("asset_001", BBox(900, 100, 300, 260))],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("d", "text", d_option, BBox(80, 380, 700, 30)),
                SourceBlock("noise", "text", "激光h", BBox(950, 380, 80, 25)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertTrue(document.display_text.endswith("匀速圆周运动"))
        self.assertEqual(metadata["post_visual_tail_suppressed"], 1)

    def test_terminal_compact_handwritten_working_is_suppressed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="compact-working",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="18. 解方程组： equations ②Cx=3",
            display_text="18. 解方程组： equations ②Cx=3",
            question_number="18",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[
                SourceBlock("system", "formula", "equations", BBox(500, 100, 220, 140)),
                SourceBlock("label", "text", "②", BBox(735, 150, 25, 30)),
                SourceBlock("working", "text", "Cx=3", BBox(600, 240, 80, 14)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text, "18. 解方程组： equations")
        self.assertEqual(metadata["terminal_compact_working_suppressed"], 1)
        self.assertEqual(metadata["detached_equation_labels_suppressed"], 1)

    def test_handwritten_answer_merged_below_tall_calculation_formula_is_suppressed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="tall-calculation-answer",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                "17. （本题满分5分）\n"
                "计算： $\\sqrt{12}+|2-\\sqrt{3}|-(\\frac{1}{2})^{-1}=\\sqrt{3}$"
            ),
            display_text="17. （本题满分5分）\n计算： √12+|2-√3|-(1/2)⁻¹=√3",
            question_number="17",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[
                SourceBlock("header", "text", "17. （本题满分5分）", BBox(400, 267, 246, 38)),
                SourceBlock("calculation", "text", "计算：", BBox(446, 326, 73, 41)),
                SourceBlock(
                    "formula",
                    "formula",
                    r"\sqrt{12}+|2-\sqrt{3}|-(\frac{1}{2})^{-1}=\sqrt{3}",
                    BBox(531, 314, 226, 136),
                ),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertTrue(document.canonical_text.endswith("$"))
        self.assertNotIn("=\\sqrt{3}", document.canonical_text)
        self.assertNotIn("=√3", document.display_text)
        self.assertEqual(metadata["tall_formula_answer_tails_suppressed"], 1)

        second = normalize_document_for_output(document)
        self.assertEqual(second["tall_formula_answer_tails_suppressed"], 0)

    def test_missing_fill_line_before_unit_is_restored_from_source_gap(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="fill-line-before-unit",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="1. 则摆球的直径 cm。",
            display_text="1. 则摆球的直径 cm。",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("prompt", "text", "1. 则摆球的直径", BBox(20, 30, 300, 20), 0.99),
                SourceBlock("unit", "text", "cm。", BBox(354, 30, 50, 20), 0.99),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text, "1. 则摆球的直径 ______ cm。")
        self.assertEqual(metadata["geometric_answer_placeholders_restored"], 1)
        self.assertIn("GEOMETRIC_ANSWER_PLACEHOLDER_RESTORED:1", document.warnings)

        second = normalize_document_for_output(document)
        self.assertEqual(second["geometric_answer_placeholders_restored"], 0)
        self.assertEqual(document.display_text.count("______"), 1)

    def test_tiny_low_confidence_fill_artifact_is_replaced_by_line(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="fill-artifact",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="4. 可能的原因是 0\nA. 选项",
            display_text="4. 可能的原因是 0\nA. 选项",
            question_number="4",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("prompt", "text", "4. 可能的原因是", BBox(20, 30, 300, 20), 0.99),
                SourceBlock("artifact", "text", "0", BBox(400, 42, 10, 6), 0.60),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text, "4. 可能的原因是 ______\nA. 选项")
        self.assertEqual(metadata["geometric_answer_placeholders_restored"], 1)

    def test_narrow_v_artifact_before_acceleration_unit_is_removed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="triangle-misread-as-v",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=r"1. 加速度大小为 V $\mathrm{m/s^2}$。",
            display_text="1. 加速度大小为 V m/s²。",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[
                SourceBlock("prompt", "text", "1. 加速度大小为", BBox(20, 30, 240, 22), 1.0),
                SourceBlock("artifact", "text", "V", BBox(310, 34, 9, 18), 0.67),
                SourceBlock("unit", "formula", r"\mathrm{m/s^2}", BBox(350, 30, 70, 28), 0.60),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text, "1. 加速度大小为 ______ m/s²。")
        self.assertNotIn("______ V", document.canonical_text)
        self.assertEqual(metadata["geometric_answer_placeholders_restored"], 1)

    def test_normal_zero_and_close_unit_spacing_are_not_changed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="not-a-fill-gap",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="1. 当x为0时，长度为5cm。",
            display_text="1. 当x为0时，长度为5cm。",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("zero_prompt", "text", "1. 当x为", BBox(20, 30, 100, 20), 0.99),
                SourceBlock("zero", "text", "0", BBox(128, 30, 12, 20), 0.99),
                SourceBlock("length", "text", "时，长度为5", BBox(145, 30, 130, 20), 0.99),
                SourceBlock("unit", "text", "cm。", BBox(280, 30, 45, 20), 0.99),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text, "1. 当x为0时，长度为5cm。")
        self.assertEqual(metadata["geometric_answer_placeholders_restored"], 0)

    def test_option_vector_style_follows_symbols_established_in_stem(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="vector-style",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                r"1. 已知 $\{\pmb{a},\pmb{b},\pmb{c}\}$ 是一组基底" "\n"
                r"A. $\boldsymbol{b}+c,\boldsymbol{a}+c$ "
                r"B. $c,\boldsymbol{a}+\boldsymbol{b}-2c$"
            ),
            display_text="1. 已知 {a,b,c} 是一组基底\nA. b+c,a+c B. c,a+b-2c",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=3,
            source_blocks=[],
        )

        metadata = normalize_document_for_output(document)

        option_text = document.canonical_text.split("\nA.", 1)[1]
        self.assertNotRegex(option_text, r"(?<!boldsymbol\{)c")
        self.assertEqual(option_text.count(r"\boldsymbol{c}"), 4)
        self.assertEqual(metadata["vector_style_normalizations"], 4)

        second = normalize_document_for_output(document)
        self.assertEqual(second["vector_style_normalizations"], 0)

    def test_plain_scalar_basis_letters_without_bold_stem_are_unchanged(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="scalar-style",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="1. 基底为 $a,b,c$\nA. $a+c$ B. $b-c$",
            display_text="1. 基底为 a,b,c\nA. a+c B. b-c",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=3,
            source_blocks=[],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.canonical_text, "1. 基底为 $a,b,c$\nA. $a+c$ B. $b-c$")
        self.assertEqual(metadata["vector_style_normalizations"], 0)

    def test_low_confidence_set_complement_uses_established_universe(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="set-complement",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                r"1. 已知集合 $A=\{0,1\}$，全集 $U=\{-1,0,1\}$，"
                r"则 $C _ { V } A = 1$ ()" "\n"
                "A. 0 B. {-1} C. {1} D. {-1,1}"
            ),
            display_text=(
                "1. 已知集合 A={0,1}，全集 U={-1,0,1}，则 C _(V) A = 1 ()\n"
                "A. 0 B. {-1} C. {1} D. {-1,1}"
            ),
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=3,
            source_blocks=[
                SourceBlock("complement", "formula", "C _ { V } A = 1", BBox(300, 30, 80, 24), 0.53),
                SourceBlock("a", "text", "A. 0", BBox(20, 70, 60, 24), 0.84),
                SourceBlock("b", "text", "B. {-1}", BBox(120, 70, 80, 24), 0.90),
                SourceBlock("c", "text", "C. {1}", BBox(240, 70, 80, 24), 0.90),
                SourceBlock("d", "text", "D. {-1,1}", BBox(360, 70, 100, 24), 0.90),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn(r"C _ { U } A =", document.canonical_text)
        self.assertNotIn("= 1$ ()", document.canonical_text)
        self.assertIn(r"A. $\varnothing$", document.canonical_text)
        self.assertIn("A. ∅", document.display_text)
        self.assertEqual(metadata["set_complement_normalizations"], 3)

    def test_set_complement_is_not_changed_without_established_universe(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="set-complement-no-universe",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=r"1. 求集合 $C _ { V } A$。\nA. 0 B. {1}",
            display_text="1. 求集合 C _(V) A。\nA. 0 B. {1}",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[
                SourceBlock("complement", "formula", "C _ { V } A", BBox(100, 30, 80, 24), 0.53),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn(r"C _ { V } A", document.canonical_text)
        self.assertEqual(metadata["set_complement_normalizations"], 0)

    def test_formula_quotes_are_normalized_as_paired_chinese_quotes(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="formula-quotes",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                r'1. 则 $\text{" } a>|b| \text{" }$ 是 '
                r'$\text{" } \frac{1}{a}<\frac{1}{|b|} \text{" }$ 的'
            ),
            display_text='1. 则 " a>|b| " 是 " 1/a<1/|b| " 的',
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=2,
            source_blocks=[],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text, "1. 则 “a>|b|” 是 “1/a<1/|b|” 的")
        self.assertEqual(document.canonical_text.count(r"\text{“ }"), 2)
        self.assertEqual(document.canonical_text.count(r"\text{” }"), 2)
        self.assertEqual(metadata["formula_quote_pairs_normalized"], 2)

        second = normalize_document_for_output(document)
        self.assertEqual(second["formula_quote_pairs_normalized"], 0)

    def test_split_fraction_numerator_is_merged_when_denominator_is_empty(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="split-fraction",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=r"1. 经过 $\frac { 1 } { 4 } \frac { 3 } { } T$ 后",
            display_text="1. 经过 1/4 3/ T 后",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn(r"\frac{3}{4} T", document.canonical_text)
        self.assertEqual(document.display_text, "1. 经过 3/4 T 后")
        self.assertEqual(metadata["split_fraction_numerators_normalized"], 1)

    def test_two_valid_adjacent_fractions_are_not_merged(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="valid-fractions",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=r"1. 计算 $\frac{1}{4}+\frac{3}{5}$",
            display_text="1. 计算 1/4+3/5",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.canonical_text, r"1. 计算 $\frac{1}{4}+\frac{3}{5}$")
        self.assertEqual(metadata["split_fraction_numerators_normalized"], 0)

    def test_redundant_numeric_superscript_is_flattened(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="nested-superscript",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=r"1. 有效质量 $m=\frac{E}{c ^ { ^ 2 }}$。",
            display_text="1. 有效质量 m=E/c²。",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(
            document.canonical_text,
            r"1. 有效质量 $m=\frac{E}{c ^{2}}$。",
        )
        self.assertNotIn("^ { ^", document.canonical_text)
        self.assertEqual(metadata["redundant_numeric_superscripts_normalized"], 1)

    def test_gamma_ray_hallucinated_edge_digits_are_removed_with_line_cues(self) -> None:
        formula = r"1 \gamma _ { 1 }"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="gamma-ray-edge-digits",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=rf"11. 请选择\nA. 所采用的射线为 ${formula}$ 射线。",
            display_text="11. 请选择\nA. 所采用的射线为 1γ1 射线。",
            question_number="11",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[
                SourceBlock("left", "text", "A. 所采用的射线为", BBox(110, 446, 460, 22)),
                SourceBlock("gamma", "formula", formula, BBox(571, 448, 17, 26), confidence=0.78),
                SourceBlock("right", "text", "射线。", BBox(589, 446, 57, 22)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn(r"$\gamma$ 射线", document.canonical_text)
        self.assertIn("γ 射线", document.display_text)
        self.assertEqual(metadata["gamma_ray_symbols_normalized"], 1)

    def test_energy_level_direction_is_restored_between_adjacent_formulas(self) -> None:
        left_formula = r"n { = } 3"
        right_formula = r"n { = } 2"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="energy-transition-direction",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                rf"12. 能级跃迁时，相邻亮条纹间距最宽的是 "
                rf"${left_formula}$ ${right_formula}$ 跃迁时释放出的射线。"
            ),
            display_text=(
                "12. 能级跃迁时，相邻亮条纹间距最宽的是 "
                "n=3 n=2 跃迁时释放出的射线。"
            ),
            question_number="12",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=2,
            source_blocks=[
                SourceBlock("before", "text", "间距最宽的是", BBox(166, 669, 558, 24)),
                SourceBlock("n3", "formula", left_formula, BBox(727, 667, 51, 29), confidence=0.55),
                SourceBlock("n2", "formula", right_formula, BBox(812, 668, 51, 27), confidence=0.55),
                SourceBlock("after", "text", "跃迁时释放出的射线。", BBox(864, 669, 266, 24)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn(f"${left_formula}$ 向 ${right_formula}$", document.canonical_text)
        self.assertIn("n=3 向 n=2", document.display_text)
        self.assertEqual(metadata["energy_level_directions_restored"], 1)

    def test_terminal_heat_symbol_spurious_subscript_is_removed(self) -> None:
        formula = r"Q _ { \mathrm { ~ c ~ } }"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="terminal-heat-symbol",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=rf"15. (3) 整个过程气体吸收的热量 ${formula}$",
            display_text="15. (3) 整个过程气体吸收的热量 Q _ ~ c ~",
            question_number="15",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[
                SourceBlock("cue", "text", "(3) 整个过程气体吸收的热量", BBox(241, 437, 276, 23)),
                SourceBlock("q", "formula", formula, BBox(518, 439, 23, 24), confidence=0.75),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertTrue(document.canonical_text.endswith("$Q$"))
        self.assertTrue(document.display_text.endswith("热量 Q"))
        self.assertEqual(metadata["terminal_heat_symbols_normalized"], 1)

    def test_s_glyph_in_numeric_t_subscript_family_is_normalized_to_five(self) -> None:
        formula = r"T _ { s } = 1"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="numeric-subscript-glyph",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                r"9. 已知 $T_{n}=a_1\cdots a_n$，$T_{6}=8$\n"
                rf"A. $q=2$ B. $T_{{7}}=16$ C. ${formula}$"
            ),
            display_text=(
                "9. 已知 T_(n)=a₁…aₙ，T₆=8\n"
                "A. q=2 B. T₇=16 C. T _(s) = 1"
            ),
            question_number="9",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=4,
            source_blocks=[
                SourceBlock("ts", "formula", formula, BBox(173, 464, 76, 37), confidence=0.77)
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn(r"$T_{5} = 1$", document.canonical_text)
        self.assertEqual(metadata["numeric_subscript_glyphs_normalized"], 1)

    def test_derivative_prime_is_restored_in_parallel_definition_clause(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="derivative-prime",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                r"17. (1) 设 $f^{\prime}(x)$ 是f(x)的导函数，求 $f ( x )$ 的单调区间；"
            ),
            display_text="17. (1) 设 f^(prime)(x) 是f(x)的导函数，求 f (x) 的单调区间；",
            question_number="17",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=2,
            source_blocks=[
                SourceBlock("prime", "formula", r"f^{\prime}(x)", BBox(308, 295, 81, 40), confidence=0.69),
                SourceBlock("between", "text", "是f(x)的导函数，求", BBox(390, 291, 310, 46)),
                SourceBlock("plain", "formula", "f ( x )", BBox(701, 296, 80, 40), confidence=0.59),
                SourceBlock("after", "text", "的单调区间；", BBox(782, 291, 180, 48)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.canonical_text.count(r"f^{\prime}(x)"), 2)
        self.assertIn("求 f′(x) 的单调区间", document.display_text)
        self.assertEqual(metadata["derivative_primes_restored"], 1)

    def test_derivative_prime_restoration_accepts_normalized_formula_spacing(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="derivative-prime-normalized-spacing",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                r"17. (1) 设 $f^{\prime}(x)$ 是f(x)的导函数，求 $f (x)$ 的单调区间；"
            ),
            display_text="17. (1) 设 f^(prime)(x) 是f(x)的导函数，求 f (x) 的单调区间；",
            question_number="17",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=2,
            source_blocks=[
                SourceBlock("prime", "formula", r"f^{\prime}(x)", BBox(308, 295, 81, 40), confidence=0.69),
                SourceBlock("between", "text", "是f(x)的导函数，求", BBox(390, 291, 310, 46)),
                SourceBlock("plain", "formula", "f ( x )", BBox(701, 296, 80, 40), confidence=0.59),
                SourceBlock("after", "text", "的单调区间；", BBox(782, 291, 180, 48)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.canonical_text.count(r"f^{\prime}(x)"), 2)
        self.assertIn("求 f′(x) 的单调区间", document.display_text)
        self.assertEqual(metadata["derivative_primes_restored"], 1)

    def test_terminal_i_glyph_is_restored_to_semicolon_in_requested_quantities(self) -> None:
        first = r"F_{\mathbf{N}\mathbf{i}}"
        second = r"v _ { \mathrm { m i } }"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="terminal-semicolon-glyph",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                rf"17. (1) 管道对其压力大小 ${first}$\n"
                rf"(3) 求运动过程中的最大速度 ${second}$"
            ),
            display_text=(
                "17. (1) 管道对其压力大小 F_Ni\n"
                "(3) 求运动过程中的最大速度 v _ m i"
            ),
            question_number="17",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=2,
            source_blocks=[
                SourceBlock("cue-1", "text", "(1) 管道对其压力大小", BBox(266, 174, 465, 24)),
                SourceBlock("f", "formula", first, BBox(732, 177, 31, 22), confidence=0.64),
                SourceBlock("cue-2", "text", "求运动过程中的最大速度", BBox(310, 301, 502, 22)),
                SourceBlock("v", "formula", second, BBox(814, 305, 30, 19), confidence=0.64),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn(r"$F_{N};$", document.canonical_text)
        self.assertIn(r"$v_{m};$", document.canonical_text)
        self.assertEqual(metadata["trailing_semicolon_subscripts_normalized"], 2)

    def test_stacked_single_letter_denominator_is_merged_into_fraction(self) -> None:
        formula = r"U_{AC}=^{2E_{p}}"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="stacked-denominator",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=f"8. 请选择\nB. 电势差 ${formula}$\nq",
            display_text="8. 请选择\nB. 电势差 U_(AC)=^2E_(p)\nq",
            question_number="8",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[
                SourceBlock("formula", "formula", formula, BBox(354, 331, 107, 46), confidence=0.94),
                SourceBlock("denominator", "text", "q", BBox(426, 375, 23, 30), confidence=1.0),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn(r"$U_{AC}=\frac{2E_{p}}{q}$", document.canonical_text)
        self.assertNotRegex(document.display_text, r"(?m)^q$")
        self.assertEqual(metadata["stacked_denominator_formulas_restored"], 1)

    def test_leading_formula_and_isolated_page_fraction_are_removed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="page-noise",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="$t_3\\sim t_4$\n8. 题干正文\n2/4",
            display_text="t₃∼t₄\n8. 题干正文\n2/4",
            question_number="8",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=1,
            source_blocks=[
                SourceBlock("old", "formula", r"t_3\sim t_4", BBox(20, 5, 90, 25)),
                SourceBlock("question", "text", "8. 题干正文", BBox(20, 60, 280, 30)),
                SourceBlock("page", "text", "2/4", BBox(120, 220, 60, 25)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text, "8. 题干正文")
        self.assertEqual(metadata["leading_noise_trimmed"], 1)
        self.assertEqual(metadata["page_fraction_noise_suppressed"], 1)

    def test_vertical_fragment_merges_and_embedded_panel_row_is_suppressed(self) -> None:
        assets = [
            _asset("asset_001", BBox(0, 50, 100, 100)),
            _asset("asset_002", BBox(120, 50, 100, 100)),
            _asset("asset_003", BBox(240, 50, 100, 100)),
            _asset("asset_004", BBox(360, 80, 100, 70)),
            _asset("asset_005", BBox(360, 50, 100, 20)),
        ]
        labels = [
            SourceBlock(f"label_{index}", "text", label, BBox(x, 135, 20, 20))
            for index, (label, x) in enumerate(
                zip("甲乙丙丁", (40, 160, 280, 400)), start=1
            )
        ]
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="panel-fragment",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="2. 选择正确图片\n甲乙丙丁\nA.甲 B.乙 C.丙 D.丁",
            display_text="2. 选择正确图片\n甲乙丙丁\nA.甲 B.乙 C.丙 D.丁",
            question_number="2",
            stem="",
            options=[],
            visual_assets=assets,
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=labels,
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(len(document.visual_assets), 4)
        self.assertNotIn("\n甲乙丙丁\n", document.display_text)
        self.assertEqual(metadata["embedded_panel_labels_suppressed"], 4)
        self.assertEqual(metadata["visual_fragments_merged"], ["asset_004+asset_005"])

    def test_ignored_three_panel_row_expands_clipped_final_panel(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="three-panel-expansion",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="16. 如图甲所示，电流按乙图变化，换成图丙所示圆盘",
            display_text="16. 如图甲所示，电流按乙图变化，换成图丙所示圆盘",
            question_number="16",
            stem="",
            options=[],
            visual_assets=[_asset("asset_001", BBox(448, 344, 640, 375))],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("line", "text", "题干", BBox(104, 38, 1115, 34)),
                SourceBlock("intruding", "text", "过计算说明圆心还是外侧温度高", BBox(221, 321, 774, 30)),
                SourceBlock("a", "text", "甲", BBox(581, 640, 31, 33), disposition="ignored"),
                SourceBlock("b", "text", "乙", BBox(855, 639, 33, 32), disposition="ignored"),
                SourceBlock("c", "text", "丙", BBox(1119, 636, 32, 30), disposition="ignored"),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.visual_assets[0].source_bbox.right, 1219)
        self.assertEqual(document.visual_assets[0].source_bbox.y, 353)
        self.assertEqual(document.visual_assets[0].source_bbox.bottom, 719)
        self.assertEqual(metadata["ignored_panel_assets_expanded"], ["asset_001"])
        self.assertFalse(document.visual_assets[0].redraw_used)

    def test_flattened_contingency_tables_are_restored_as_direct_crop(self) -> None:
        tail = (
            "睡眠质量平均每天使用电子产品的时间不超标合计\n"
            "超标\n良好\n障碍（包括轻度和重度）\n合计 100\n"
            r"附： $\chi^{2}=\frac{n(ad-bc)^{2}}{(a+b)(c+d)(a+c)(b+d)}$ "
            "α 0.05 0.01 0.005 0.001\nxa 3.841 6.635 7.879 10.828"
        )
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="structured-table-region",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                "16. 如下表格：\n(3) 按所给数据，完成下面列联表，并根据独立性检验判断\n"
                + tail
            ),
            display_text=(
                "16. 如下表格：\n(3) 按所给数据，完成下面列联表，并根据独立性检验判断\n"
                + tail
            ),
            question_number="16",
            stem="",
            options=[],
            visual_assets=[_asset("asset_001", BBox(383, 88, 525, 144))],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[
                SourceBlock("part3", "text", "(3)按所给数据，完成下面列联表", BBox(347, 319, 599, 14)),
                SourceBlock("header", "text", "平均每天使用电子产品的时间", BBox(611, 521, 189, 16)),
                SourceBlock("quality", "text", "睡眠质量", BBox(481, 531, 60, 17)),
                SourceBlock("over", "text", "超标", BBox(627, 542, 33, 19)),
                SourceBlock("under", "text", "不超标", BBox(727, 542, 46, 18)),
                SourceBlock("good", "text", "良好", BBox(495, 564, 34, 21)),
                SourceBlock("issue", "text", "障碍（包括轻度和重度）", BBox(433, 588, 148, 13)),
                SourceBlock("total", "text", "合计", BBox(495, 607, 34, 21)),
                SourceBlock("appendix", "text", "附：", BBox(337, 651, 27, 23)),
                SourceBlock("chi", "formula", r"\chi^{2}=\frac{n(ad-bc)^{2}}{(a+b)(c+d)(a+c)(b+d)}", BBox(367, 641, 198, 41)),
                SourceBlock("d1", "text", "0.05", BBox(664, 653, 32, 20)),
                SourceBlock("d2", "text", "0.01", BBox(722, 652, 31, 21)),
                SourceBlock("d3", "text", "0.005", BBox(776, 653, 37, 20)),
                SourceBlock("d4", "text", "0.001", BBox(833, 653, 36, 19)),
                SourceBlock("d5", "text", "3.841", BBox(660, 684, 40, 19)),
                SourceBlock("d6", "text", "6.635", BBox(717, 684, 41, 19)),
                SourceBlock("d7", "text", "7.879", BBox(776, 684, 38, 19)),
                SourceBlock("d8", "text", "10.828", BBox(829, 684, 46, 19)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(len(document.visual_assets), 2)
        restored = document.visual_assets[1]
        self.assertEqual(restored.kind, "table_region")
        self.assertEqual(restored.extraction, "direct_source_crop")
        self.assertFalse(restored.redraw_used)
        self.assertNotIn("睡眠质量平均每天使用电子产品的时间", document.display_text)
        self.assertEqual(metadata["structured_table_region_assets_restored"], 1)

    def test_legacy_caption_reference_is_restored_and_caption_pixels_are_cropped(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="caption-before-question",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="14. 读数如所示。\n图1",
            display_text="14. 读数如所示。\n图1",
            question_number="14",
            stem="",
            options=[],
            visual_assets=[_asset("asset_001", BBox(100, 100, 240, 100))],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            warnings=["STANDALONE_VISUAL_CAPTION_SUPPRESSED:1"],
            source_blocks=[
                SourceBlock("caption", "text", "图1", BBox(190, 202, 45, 22)),
                SourceBlock("question", "text", "读数如图1所示。", BBox(20, 250, 220, 25)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn("读数如图1所示", document.display_text)
        self.assertEqual(document.display_text.count("图1"), 1)
        self.assertEqual(document.visual_assets[0].source_bbox.bottom, 230)
        self.assertEqual(metadata["inline_caption_references_restored"], 1)

        first_bbox = document.visual_assets[0].source_bbox
        second_metadata = normalize_document_for_output(document)
        self.assertEqual(document.visual_assets[0].source_bbox, first_bbox)
        self.assertEqual(second_metadata["caption_assets_expanded"], [])

    def test_panel_labels_are_kept_inside_their_direct_source_crops(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="four-panel-labels",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="2. 下面四幅插图中相同的是\n甲乙丙丁\nA.甲和乙 B.乙和丙",
            display_text="2. 下面四幅插图中相同的是\n甲乙丙丁\nA.甲和乙 B.乙和丙",
            question_number="2",
            stem="",
            options=[],
            visual_assets=[
                _asset("asset_001", BBox(20, 80, 100, 100)),
                _asset("asset_002", BBox(140, 80, 100, 100)),
                _asset("asset_003", BBox(260, 80, 100, 100)),
                _asset("asset_004", BBox(380, 80, 100, 100)),
            ],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("a", "text", "甲", BBox(55, 178, 25, 24)),
                SourceBlock("b", "text", "乙", BBox(175, 178, 25, 24)),
                SourceBlock("c", "text", "丙", BBox(295, 178, 25, 24)),
                SourceBlock("d", "text", "丁", BBox(415, 178, 25, 24)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual([asset.source_bbox.bottom for asset in document.visual_assets], [208] * 4)
        self.assertEqual(len(metadata["panel_label_assets_expanded"]), 4)
        self.assertNotIn("\n甲乙丙丁\n", document.display_text)

        first_boxes = [asset.source_bbox for asset in document.visual_assets]
        second_metadata = normalize_document_for_output(document)
        self.assertEqual(
            [asset.source_bbox for asset in document.visual_assets],
            first_boxes,
        )
        self.assertEqual(second_metadata["panel_label_assets_expanded"], [])

    def test_panel_label_at_visual_seam_follows_vertically_overlapping_asset(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="seam-panel-label",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="14-III. 电路如图甲所示。\n甲",
            display_text="14-III. 电路如图甲所示。\n甲",
            question_number="14-III",
            stem="",
            options=[],
            visual_assets=[
                _asset("micrometer", BBox(352, 391, 249, 176)),
                _asset("circuit", BBox(648, 374, 253, 219)),
            ],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[SourceBlock("label", "text", "甲", BBox(608, 567, 28, 28))],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(metadata["panel_label_assets_expanded"], ["circuit"])
        self.assertEqual(document.visual_assets[0].source_bbox, BBox(352, 391, 249, 176))
        self.assertEqual(document.visual_assets[1].source_bbox.x, 608)

    def test_close_centered_page_fraction_after_complete_options_is_removed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="close-page-number",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="8. 请选择\nA.一 B.二 C.三 D.四\n1 / 4",
            display_text="8. 请选择\nA.一 B.二 C.三 D.四\n1/4",
            question_number="8",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock(label, "text", f"{label}.选项", BBox(50 + index * 120, 100, 90, 25))
                for index, label in enumerate("ABCD")
            ] + [SourceBlock("page", "text", "1 / 4", BBox(230, 160, 50, 20))],
        )

        metadata = normalize_document_for_output(document)

        self.assertNotRegex(document.display_text, r"1\s*/\s*4")
        self.assertEqual(metadata["page_fraction_noise_suppressed"], 1)

    def test_completed_nested_question_drops_geometry_gated_restarted_part(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="restarted-next-question",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                "18. 题干\n（1）求轨迹；\n(2) 证明结论。\n"
                "(i）第一问；（ii）第二问。\n（1）提示，右两种理解"
            ),
            display_text=(
                "18. 题干\n（1）求轨迹；\n(2) 证明结论。\n"
                "(i）第一问；（ii）第二问。\n（1） 提示、右两种理解"
            ),
            question_number="18",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("part-1", "text", "（1）求轨迹；", BBox(50, 100, 200, 30)),
                SourceBlock("part-2", "text", "(2) 证明结论。", BBox(50, 160, 220, 30)),
                SourceBlock("nested", "text", "(i）第一问；（ii）第二问。", BBox(50, 220, 500, 40)),
                SourceBlock("fraction", "formula", r"\frac{3}{8}", BBox(420, 220, 40, 62)),
                SourceBlock("tail", "text", "（1）提示，右两种理解", BBox(300, 305, 300, 22)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertNotIn("提示", document.display_text)
        self.assertNotIn("提示", document.canonical_text)
        self.assertEqual(metadata["restarted_subpart_tails_suppressed"], 1)

    def test_terminal_fill_cue_before_page_number_restores_answer_line(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="terminal-answer-line",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="14. 黑球仍在甲手中的概率为\n2 / 4",
            display_text="14. 黑球仍在甲手中的概率为\n2 / 4",
            question_number="14",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("cue", "text", "黑球仍在甲手中的概率为", BBox(50, 100, 450, 28)),
                SourceBlock("page", "text", "2 / 4", BBox(220, 220, 60, 22)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertTrue(
            document.display_text.endswith("概率为 ______"),
            (repr(document.display_text), metadata),
        )
        self.assertEqual(metadata["terminal_answer_placeholders_restored"], 1)

    def test_black_triangle_misread_as_acceleration_a_is_restored_to_slot(self) -> None:
        formula = r"\underline{\mathbf{A} \quad \mathbf{m} / \mathbf{s}^{2}}"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="redacted-acceleration",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=f"14. 加速度实验\n读数为▲cm，求得小车的加速度大小为 ${formula}$。",
            display_text="14. 加速度实验\n读数为▲cm，求得小车的加速度大小为 A m/s²。",
            question_number="14",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[
                SourceBlock("line", "text", "读数为▲cm，求得小车的加速度大小为", BBox(50, 100, 500, 28)),
                SourceBlock("formula", "formula", formula, BBox(500, 100, 130, 28), confidence=0.56),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertNotIn(" A m/s²", document.display_text)
        self.assertIn("______ m/s²", document.display_text)
        self.assertEqual(metadata["redacted_acceleration_slots_restored"], 1)

    def test_low_confidence_gravity_formula_is_restored_despite_spacing_drift(self) -> None:
        formula = r"\mathcal { E } ^ { = } ."
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="gravity-answer-slot",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                r"14. 请根据图乙表示出重力加速度 $\mathcal { E } ^ { = }.$"
                "。（用a、b和π表示）"
            ),
            display_text="14. 请根据图乙表示出重力加速度 E ⁼.。（用a、b和π表示）",
            question_number="14",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[
                SourceBlock("left", "text", "请根据图乙表示出重力加速度", BBox(262, 550, 297, 22)),
                SourceBlock("formula", "formula", formula, BBox(561, 554, 29, 21), confidence=0.57),
                SourceBlock("right", "text", "。（用a、b和π表示）", BBox(712, 549, 206, 23)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn("g=______", document.display_text)
        self.assertNotIn("E ⁼", document.display_text)
        self.assertIn(r"$g=\underline{\hspace{4em}}$", document.canonical_text)
        self.assertEqual(metadata["gravity_acceleration_slots_restored"], 1)

    def test_overlapping_short_formula_tail_is_removed_and_triangle_angle_is_uppercase(self) -> None:
        angle = r"c = \frac { 2 \pi } { 3 }"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="semantic-overlap",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=f"17. 三角形的内角A,B,C，条件②：sin A-sin C=1 条件③： ${angle}$\nC. 总能量为mc² $m c$",
            display_text="17. 三角形的内角A,B,C，条件②：sin A-sin C=1 条件③： c = 2π/3\nC. 总能量为mc² m c",
            question_number="17",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=2,
            source_blocks=[
                SourceBlock("condition", "text", "条件③：", BBox(50, 100, 80, 25)),
                SourceBlock("angle", "formula", angle, BBox(135, 95, 100, 35), confidence=0.94),
                SourceBlock("energy", "text", "C. 总能量为mc²", BBox(50, 150, 249, 29)),
                SourceBlock("duplicate", "formula", "m c", BBox(255, 145, 44, 29), confidence=0.86),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn("条件③： C", document.display_text)
        self.assertNotIn("条件③： c", document.display_text)
        self.assertTrue(document.display_text.endswith("总能量为mc²"))
        self.assertEqual(metadata["triangle_angle_names_normalized"], 1)
        self.assertEqual(metadata["overlapping_formula_tails_suppressed"], 1)

    def test_numeric_subfigure_caption_is_cropped_and_not_rendered_as_text(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="numeric-subfigure-caption",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="14-I. 根据指针读数。\n第14-1图丁",
            display_text="14-I. 根据指针读数。\n第14-1图丁",
            question_number="14-I",
            stem="",
            options=[],
            visual_assets=[_asset("asset_001", BBox(500, 500, 240, 120))],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("caption", "text", "第14-1图丁", BBox(570, 625, 110, 24)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertNotIn("第14-1图丁", document.display_text)
        self.assertGreaterEqual(document.visual_assets[0].source_bbox.bottom, 655)
        self.assertEqual(metadata["caption_blocks_suppressed"], 1)
        self.assertEqual(metadata["caption_assets_expanded"], ["asset_001"])

    def test_nested_system_alignment_is_replaced_by_left_aligned_cases(self) -> None:
        formula = (
            r"\begin{aligned}&\left\{\begin{aligned}x-2y=1,\\"
            r"\frac{1}{2}x+3(x-2y)=\frac{9}{2}.\end{aligned}\right.\end{aligned}"
        )
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="system-alignment",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=f"18. 解方程组： ${formula}$",
            display_text=f"18. 解方程组： {formula}",
            question_number="18",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[SourceBlock("system", "formula", formula, BBox(100, 100, 300, 100))],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn(r"\begin{cases}", document.canonical_text)
        self.assertNotIn(r"\left\{\begin{aligned}", document.canonical_text)
        self.assertEqual(metadata["system_equation_environments_normalized"], 1)

    def test_split_terminal_toolbar_cluster_is_suppressed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="split-toolbar",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="17. 求边长。\n可口 200%",
            display_text="17. 求边长。\n可口 200%",
            question_number="17",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("stem", "text", "求边长。", BBox(50, 500, 200, 30), confidence=1.0),
                SourceBlock("icon", "text", "可口", BBox(838, 692, 62, 23), confidence=0.736),
                SourceBlock("zoom", "text", "200%", BBox(922, 695, 48, 18), confidence=1.0),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text, "17. 求边长。")
        self.assertEqual(metadata["screen_toolbar_noise_suppressed"], 2)

    def test_numbered_next_question_tail_is_suppressed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="numbered-next-question",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="18. 求点Q的坐标。\n19(1)提示。有两种理解：",
            display_text="18. 求点Q的坐标。\n19(1)提示。有两种理解：",
            question_number="18",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("answer", "text", "求点Q的坐标。", BBox(50, 608, 220, 40)),
                SourceBlock("tail", "text", "19(1)提示。有两种理解：", BBox(258, 697, 474, 22)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertEqual(document.display_text, "18. 求点Q的坐标。")
        self.assertEqual(metadata["adjacent_question_tails_suppressed"], 1)

    def test_low_confidence_margin_token_before_part_two_is_suppressed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="margin-before-part",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="14. 实验。\nSy (2)该小组采用限流电路。",
            display_text="14. 实验。\nSy (2)该小组采用限流电路。",
            question_number="14",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=True,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("noise", "text", "Sy", BBox(81, 523, 27, 27), confidence=0.5587),
                SourceBlock("part", "text", "(2)该小组采用限流电路。", BBox(256, 522, 400, 31), confidence=0.98),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertNotIn("Sy", document.display_text)
        self.assertEqual(metadata["marginal_tokens_before_subparts_suppressed"], 1)

    def test_page_section_headers_and_inline_floating_caption_are_suppressed(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="section-and-caption",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                "非选择题部分\n三、非选择题（本题共4小题，共58分）\n"
                "14-I（1）测量。\n(2)某同学选第14-I图甲\n择一根电阻丝。"
            ),
            display_text=(
                "非选择题部分\n三、非选择题（本题共4小题，共58分）\n"
                "14-I（1）测量。\n(2)某同学选第14-I图甲\n择一根电阻丝。"
            ),
            question_number="14-I",
            stem="",
            options=[],
            visual_assets=[_asset("asset_001", BBox(850, 50, 250, 130))],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("section", "text", "非选择题部分", BBox(690, 0, 160, 20)),
                SourceBlock("section2", "text", "三、非选择题（本题共4小题，共58分）", BBox(330, 20, 340, 25)),
                SourceBlock("header", "text", "14-I（1）测量。", BBox(330, 45, 500, 25)),
                SourceBlock("prose1", "text", "(2)某同学选", BBox(339, 197, 498, 27)),
                SourceBlock("caption", "text", "第14-I图甲", BBox(961, 205, 80, 24)),
                SourceBlock("prose2", "text", "择一根电阻丝。", BBox(330, 220, 437, 29)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertTrue(document.display_text.startswith("14-I"))
        self.assertNotIn("非选择题部分", document.display_text)
        self.assertNotIn("第14-I图甲", document.display_text)
        self.assertIn("某同学选\n择一根", document.display_text)
        self.assertEqual(metadata["leading_exam_section_headers_suppressed"], 2)
        self.assertEqual(metadata["inline_caption_intrusions_suppressed"], 1)

    def test_nuclear_particle_notation_uses_scripts_and_neutrino_symbol(self) -> None:
        formulas = [
            r"\frac{1}{1}H",
            r"( \frac{1}{0} n )",
            r"( \frac { 0 } { 1 } e )",
            r"v + \frac{1}{1}H \rightarrow \frac{1}{0}n + \frac{0}{1}e",
            r"\_^{0}_{-1}e+\_^{0}_{1}e\rightarrow2\gamma",
        ]
        canonical = "4.中微子（v）发生核反应，产生正电子并转变为两个光子 " + " ".join(
            f"${formula}$" for formula in formulas
        )
        display = "4.中微子（v）发生核反应，产生正电子并转变为两个光子 " + " ".join(
            __import__("test7_ocr.text", fromlist=["latex_to_readable"]).latex_to_readable(formula)
            for formula in formulas
        )
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="nuclear-notation",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=canonical,
            display_text=display,
            question_number="4",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=len(formulas),
            source_blocks=[
                SourceBlock("context", "text", "中微子发生核反应，产生正电子并转变为两个光子", BBox(10, 10, 600, 30)),
                *[
                    SourceBlock(f"formula-{index}", "formula", formula, BBox(20, 50 + index * 30, 180, 25))
                    for index, formula in enumerate(formulas)
                ],
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn("中微子（ν）", document.display_text)
        self.assertIn("¹₁H", document.display_text)
        self.assertIn("¹₀n", document.display_text)
        self.assertIn("⁰₋₁e", document.display_text)
        self.assertIn(r"\nu+{}^{1}_{1}\mathrm{H}", document.canonical_text)
        self.assertEqual(metadata["nuclear_particle_notations_normalized"], 6)

    def test_system_equation_labels_are_restored_from_second_label_geometry(self) -> None:
        formula = (
            r"\begin{aligned}&\left\{\begin{aligned}x-2y=1,\\"
            r"\frac{1}{2}x+3(x-2y)=\frac{9}{2}.\end{aligned}\right.\end{aligned}"
        )
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="system-labels",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=f"18. 解方程组： ${formula}$\n②",
            display_text=f"18. 解方程组： {formula}\n②",
            question_number="18",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=1,
            source_blocks=[
                SourceBlock("system", "formula", formula, BBox(565, 305, 212, 137)),
                SourceBlock("second-label", "text", "②", BBox(791, 361, 24, 31)),
            ],
        )

        normalize_document_for_output(document)

        self.assertIn(r"\text{①}", document.canonical_text)
        self.assertIn(r"\text{②}", document.canonical_text)
        self.assertFalse(document.display_text.rstrip().endswith("\n②"))

    def test_wrapped_unit_and_selection_suffix_restore_answer_slots(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="wrapped-answer-slots",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="14. 读数如图3所示，其示数为\nmm。滑片应置于 端。",
            display_text="14. 读数如图3所示，其示数为\nmm。滑片应置于 端。",
            question_number="14",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("wrapped-left", "text", "读数如图3所示，其示数为", BBox(302, 473, 781, 21)),
                SourceBlock("wrapped-unit", "text", "mm。", BBox(386, 504, 46, 17)),
                SourceBlock("selection-left", "text", "滑片应置于", BBox(700, 552, 294, 24)),
                SourceBlock("selection-right", "text", "端", BBox(1068, 556, 15, 12)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn("示数为 ______ mm", document.display_text)
        self.assertIn("置于 ______ 端", document.display_text)
        self.assertEqual(metadata["geometric_answer_placeholders_restored"], 2)

    def test_wrapped_option_a_is_not_treated_as_ampere_unit(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="option-a-not-unit",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text="1. 下列各数中，绝对值最大的是\nA. -3 B. -2 C. 0 D. 1",
            display_text="1. 下列各数中，绝对值最大的是\nA. -3 B. -2 C. 0 D. 1",
            question_number="1",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=0,
            source_blocks=[
                SourceBlock("stem", "text", "1. 下列各数中，绝对值最大的是", BBox(246, 331, 320, 28)),
                SourceBlock("a", "text", "A. -3", BBox(291, 362, 80, 24)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertNotIn("______", document.display_text)
        self.assertEqual(metadata["geometric_answer_placeholders_restored"], 0)

    def test_ignored_relation_line_is_restored_and_removed_from_visual_crop(self) -> None:
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="ignored-relation-line",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=(
                r"14. 加速度分别为 $a_{1} 、 a_{2}$，则" + "\n"
                r"（3）若 $a_{2}=\frac{g}{8}$，重物质量之比为"
            ),
            display_text="14. 加速度分别为 a₁、a₂，则\n（3）若 a₂=g/8，重物质量之比为",
            question_number="14",
            stem="",
            options=[],
            visual_assets=[_asset("asset_002", BBox(226, 459, 664, 153))],
            handwriting_detected=False,
            formula_verified=False,
            formula_specialist_used=True,
            formula_count=2,
            source_blocks=[
                SourceBlock("part2", "text", "加速度大小为", BBox(166, 431, 815, 23)),
                SourceBlock("relation", "text", "a1、a2满足的关系式是", BBox(211, 461, 253, 21), confidence=0.997, disposition="ignored", reason="embedded_visual_text"),
                SourceBlock("redaction", "text", "A", BBox(459, 465, 30, 17), confidence=0.606, disposition="ignored", reason="embedded_visual_text"),
                SourceBlock("part3", "text", "重物质量之比为", BBox(212, 686, 181, 23)),
            ],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn("a₁、a₂满足的关系式是 ______", document.display_text)
        self.assertTrue(document.display_text.endswith("重物质量之比为 ______"))
        self.assertEqual(document.visual_assets[0].source_bbox.y, 490)
        self.assertEqual(metadata["ignored_relation_answer_lines_restored"], 1)
        self.assertEqual(metadata["visual_text_overlap_trimmed_assets"], ["asset_002"])

    def test_short_unit_underscore_and_pendulum_length_glyph_are_normalized(self) -> None:
        formula = r"l = T^{2}"
        document = QuestionDocument(
            schema_version="question_document_v2",
            sample_id="short-slot-pendulum",
            source_image="source.png",
            source_sha256="0" * 64,
            provider="fixture",
            canonical_text=f"14-I. 用单摆测定重力加速度，结果如图乙所示为_mm。摆长/为 ${formula}$。",
            display_text="14-I. 用单摆测定重力加速度，结果如图乙所示为_mm。摆长/为 l=T²。",
            question_number="14-I",
            stem="",
            options=[],
            visual_assets=[],
            handwriting_detected=False,
            formula_verified=True,
            formula_specialist_used=False,
            formula_count=1,
            source_blocks=[SourceBlock("formula", "formula", formula, BBox(100, 100, 60, 22))],
        )

        metadata = normalize_document_for_output(document)

        self.assertIn("所示为 ______ mm", document.display_text)
        self.assertIn("摆长l为", document.display_text)
        self.assertIn("摆长$l$为", document.canonical_text)
        self.assertEqual(metadata["short_underscore_answer_slots_normalized"], 1)
        self.assertEqual(metadata["pendulum_length_glyphs_normalized"], 1)


if __name__ == "__main__":
    unittest.main()
