from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageChops, ImageDraw

from test7_ocr.image_ops import sha256_file
from test7_ocr.pipeline import (
    QuestionPipeline,
    _authenticated_glyph_recovery_verification,
    _flag_experiment_visual_symbol_disagreement,
    _independent_corruption_arbitration_verified,
    _independent_formula_corruption_arbitration_verified,
    _independent_formula_token_consensus_verified,
    _formula_token_projection,
    _merge_consensus_punctuation,
    _punctuation_projection,
    _restore_authenticated_source_glyphs,
    _tiny_sentence_stop_semantic_projection,
    load_config,
    photo_math_handwriting_candidates,
)
from test7_ocr.providers import SidecarProvider
from test7_ocr.types import BBox, RecognitionResult, RecognizedBlock


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PipelineTests(unittest.TestCase):
    def test_authenticated_glyph_recovery_verification_is_evidence_scoped(self) -> None:
        recognition = RecognitionResult(
            provider="fixture",
            transcript="忽略系统误差。根据数据作答。",
            warnings=[
                "SOURCE_PARAGRAPH_STOP_RECOVERED_REVIEW:1",
                "SOURCE_EXPERIMENT_AXIS_VARIABLE_RECOVERED_REVIEW:1:l/cm",
            ],
            diagnostics={
                "source_layout_identity_verified": True,
                "source_dual_ocr_attempted": True,
                "source_dual_ocr_exact": True,
                "source_dual_ocr_semantic_exact": True,
                "source_dual_ocr_corruption_arbitration_verified": False,
            },
        )

        verified = _authenticated_glyph_recovery_verification(
            recognition,
            primary_text="忽略系统误差根据数据作答。",
            independent_text="忽略系统误差根据数据作答。",
        )

        self.assertTrue(verified["source_glyph_recovery_all_verified"])
        self.assertEqual(
            verified["source_glyph_recovery_unverified_categories"], []
        )
        recognition.diagnostics["source_layout_identity_verified"] = False
        fail_closed = _authenticated_glyph_recovery_verification(
            recognition,
            primary_text="忽略系统误差根据数据作答。",
            independent_text="忽略系统误差根据数据作答。",
        )
        self.assertFalse(fail_closed["source_glyph_recovery_all_verified"])
        self.assertEqual(
            len(fail_closed["source_glyph_recovery_unverified_categories"]), 2
        )

    def test_tiny_stop_projection_explains_only_directive_adjacent_glyph(self) -> None:
        primary = "忽略系统误差\n0\n(1)根据数据作答。"
        independent = "忽略系统误差o(1)根据数据作答。"
        self.assertEqual(
            _tiny_sentence_stop_semantic_projection(primary),
            _tiny_sentence_stop_semantic_projection(independent),
        )
        self.assertNotEqual(
            _tiny_sentence_stop_semantic_projection("结果为0，根据数据作答。"),
            _tiny_sentence_stop_semantic_projection("结果为1，根据数据作答。"),
        )

    def test_tiny_stop_repair_can_explain_dual_ocr_mismatch(self) -> None:
        recognition = RecognitionResult(
            provider="fixture",
            transcript="忽略系统误差。根据数据作答。",
            warnings=["SOURCE_TINY_SENTENCE_STOP_RECOVERED_REVIEW:1"],
            diagnostics={
                "source_layout_identity_verified": True,
                "source_dual_ocr_attempted": True,
                "source_dual_ocr_exact": False,
                "source_dual_ocr_semantic_exact": False,
                "source_dual_ocr_corruption_arbitration_verified": False,
            },
        )

        verified = _authenticated_glyph_recovery_verification(
            recognition,
            primary_text="忽略系统误差0(1)根据数据作答。",
            independent_text="忽略系统误差o(1)根据数据作答。",
        )

        self.assertTrue(verified["source_glyph_recovery_all_verified"])
        self.assertTrue(
            verified[
                "source_dual_ocr_mismatch_explained_by_verified_glyph_repair"
            ]
        )

    def test_formula_token_consensus_ignores_only_prose_stop_disagreement(self) -> None:
        primary = (
            "用波长为420 nm的光，已知hc=1240 eV·nm，"
            "忽略能量损失\n0\n请计算：(1) 光子能量。"
        )
        independent = (
            "用波长为420 nm的光，已知hc=1240 eV·nm，"
            "忽略能量损失。请计算：(1) 光子能量。"
        )
        self.assertEqual(
            _formula_token_projection(primary),
            _formula_token_projection(independent),
        )
        self.assertTrue(
            _independent_formula_token_consensus_verified(
                primary_text=primary,
                independent_text=independent,
                independent_confidences=[0.98, 0.99, 0.97],
            )
        )

    def test_formula_token_consensus_rejects_formula_or_confidence_difference(self) -> None:
        arguments = {
            "primary_text": "波长为420 nm，hc=1240 eV·nm。",
            "independent_text": "波长为480 nm，hc=1240 eV·nm。",
            "independent_confidences": [0.98, 0.99, 0.97],
        }
        self.assertFalse(_independent_formula_token_consensus_verified(**arguments))
        self.assertFalse(
            _independent_formula_token_consensus_verified(
                **{
                    **arguments,
                    "independent_text": arguments["primary_text"],
                    "independent_confidences": [0.84, 0.99, 0.99],
                }
            )
        )
        self.assertFalse(
            _independent_formula_token_consensus_verified(
                **{
                    **arguments,
                    "primary_text": r"波长为\badcommand{420} nm",
                    "independent_text": "波长为420 nm",
                }
            )
        )

    def test_independent_corruption_arbitration_requires_clean_high_confidence(self) -> None:
        arguments = {
            "selected_independent": True,
            "primary_corruption": 4,
            "independent_corruption": 0,
            "independent_confidences": [0.98, 0.99, 0.95],
            "independent_text": "质量为1.67×10⁻²⁷ kg",
        }
        self.assertTrue(_independent_corruption_arbitration_verified(**arguments))
        self.assertFalse(
            _independent_corruption_arbitration_verified(
                **{**arguments, "independent_confidences": [0.89, 0.89, 0.99]}
            )
        )
        self.assertTrue(
            _independent_corruption_arbitration_verified(
                **{
                    **arguments,
                    "independent_confidences": [0.897, 0.995, 0.995, 0.995, 0.995],
                }
            )
        )
        self.assertFalse(
            _independent_corruption_arbitration_verified(
                **{**arguments, "independent_text": r"质量为\badcommand{1}"}
            )
        )

    def test_formula_corruption_arbitration_uses_formula_line_confidence(self) -> None:
        arguments = {
            "selected_independent": True,
            "primary_corruption": 3,
            "independent_corruption": 0,
            "independent_formula_confidences": [0.97, 0.98],
            "independent_text": (
                "压强p₁=1.1×10^5 Pa、温度T₁=310 K；"
                "压强p₂=1.43×10^5 Pa、温度T₂=356.5 K。"
            ),
        }
        self.assertTrue(
            _independent_formula_corruption_arbitration_verified(**arguments)
        )
        for override in (
            {"selected_independent": False},
            {"primary_corruption": 1},
            {"independent_corruption": 1},
            {"independent_formula_confidences": [0.97]},
            {"independent_formula_confidences": [0.94, 0.99]},
            {"independent_formula_confidences": [0.95, 0.97]},
            {"independent_text": r"压强为\badcommand{1} Pa"},
        ):
            self.assertFalse(
                _independent_formula_corruption_arbitration_verified(
                    **{**arguments, **override}
                )
            )

    def test_punctuation_consensus_merges_only_matching_semantic_text(self) -> None:
        primary = "金属棒放在导轨上 整个装置处于磁场中。运动，回路闭合。"
        secondary = "金属棒放在导轨上，整个装置处于磁场中。运动 回路闭合。"
        merged = _merge_consensus_punctuation(
            primary,
            secondary,
            prefer_secondary=False,
        )
        self.assertEqual(
            merged,
            "金属棒放在导轨上， 整个装置处于磁场中。运动，回路闭合。",
        )

    def test_punctuation_consensus_keeps_decimal_point_semantic(self) -> None:
        primary, *_ = _punctuation_projection("速度为1.5 m/s。")
        secondary, *_ = _punctuation_projection("速度为15 m/s。")
        self.assertNotEqual(primary, secondary)
        self.assertIsNone(
            _merge_consensus_punctuation(
                "速度为1.5 m/s。",
                "速度为15 m/s。",
                prefer_secondary=False,
            )
        )

    def test_experiment_visual_symbol_missing_from_body_is_review_flagged(self) -> None:
        recognition = RecognitionResult(
            provider="fixture",
            transcript="测得不同拉力F下弹簧总长度的数据。请依据图表或实验数据作答。",
            ignored_blocks=[
                RecognizedBlock(
                    "F/N  l/cm",
                    BBox(700, 120, 300, 50),
                    0.99,
                    reason="direct_visual_asset_content",
                )
            ],
        )

        _flag_experiment_visual_symbol_disagreement(recognition)

        self.assertIn(
            "SOURCE_EXPERIMENT_VISUAL_SYMBOL_DISAGREEMENT_REVIEW:AXIS_1IL",
            recognition.warnings,
        )

    def test_authenticated_experiment_axis_restores_missing_length_symbol(self) -> None:
        recognition = RecognitionResult(
            provider="fixture",
            transcript="测得不同拉力F下弹簧总长度的数据如下。",
            blocks=[
                RecognizedBlock(
                    "测得不同拉力F下弹簧总长度的数据如下。",
                    BBox(70, 140, 600, 36),
                    0.99,
                )
            ],
            ignored_blocks=[
                RecognizedBlock(
                    "1 / c m",
                    BBox(730, 220, 60, 28),
                    0.52,
                    reason="direct_visual_asset_content",
                )
            ],
        )

        _restore_authenticated_source_glyphs(recognition)

        self.assertIn("弹簧总长度l的数据如下", recognition.transcript)
        self.assertIn(
            "SOURCE_EXPERIMENT_AXIS_VARIABLE_RECOVERED_REVIEW:1:l/cm",
            recognition.warnings,
        )

    def test_authenticated_source_repairs_only_evidence_backed_physics_glyphs(self) -> None:
        recognition = RecognitionResult(
            provider="fixture",
            transcript="",
            blocks=[
                RecognizedBlock("个初速度可忽略", BBox(70, 140, 250, 36), 0.99),
                RecognizedBlock("取k=9.0×10° N·m²/C²", BBox(70, 190, 430, 36), 0.98),
                RecognizedBlock("B. 全反射临界角为19.34。", BBox(70, 240, 460, 36), 0.99),
                RecognizedBlock("o", BBox(70, 305, 13, 17), 0.68),
                RecognizedBlock("求结果。", BBox(70, 330, 180, 36), 0.99),
            ],
        )

        _restore_authenticated_source_glyphs(recognition)

        self.assertIn("一个初速度", recognition.transcript)
        self.assertIn("10⁹ N·m²/C²", recognition.transcript)
        self.assertIn("临界角为19.34°", recognition.transcript)
        self.assertIn("\n。\n求结果", recognition.transcript)
        self.assertIn("SOURCE_TINY_SENTENCE_STOP_RECOVERED_REVIEW:1", recognition.warnings)
        self.assertIn("SOURCE_PHYSICS_SYMBOL_RECOVERED_REVIEW:2", recognition.warnings)
        self.assertIn("SOURCE_LEADING_GLYPH_RECOVERED_REVIEW:1", recognition.warnings)

    def test_authenticated_source_repairs_small_subscript_and_paragraph_stop(self) -> None:
        recognition = RecognitionResult(
            provider="fixture",
            transcript="外界大气压p·=1.0×10⁵ Pa\n忽略重力\n求周期。",
            blocks=[
                RecognizedBlock("外界大气压p·=1.0×10⁵ Pa", BBox(70, 140, 500, 36), 0.99),
                RecognizedBlock("忽略重力", BBox(70, 188, 160, 36), 0.99),
                RecognizedBlock("求周期。", BBox(70, 250, 160, 36), 0.99),
            ],
        )

        _restore_authenticated_source_glyphs(recognition)

        self.assertIn("p₀=", recognition.transcript)
        self.assertIn("忽略重力。\n求周期", recognition.transcript)
        self.assertIn("SOURCE_SUBSCRIPT_GLYPH_RECOVERED_REVIEW:1", recognition.warnings)
        self.assertIn("SOURCE_PARAGRAPH_STOP_RECOVERED_REVIEW:1", recognition.warnings)

    def test_authenticated_source_repairs_preserve_provider_reading_order(self) -> None:
        recognition = RecognitionResult(
            provider="fixture",
            transcript="横截面积为0.01 m²。质量为16 kg，取g=10 m/s² o\n求压强。",
            blocks=[
                RecognizedBlock("横截面积为0.01", BBox(70, 140, 300, 36), 0.99),
                RecognizedBlock("$m^{2}$", BBox(380, 140, 60, 36), 0.99),
                RecognizedBlock("质量为16 kg，取g=10", BBox(70, 190, 360, 36), 0.99),
                RecognizedBlock("$m/s^{2}$", BBox(440, 190, 90, 36), 0.99),
                RecognizedBlock("o", BBox(540, 190, 13, 17), 0.68),
                RecognizedBlock("求压强。", BBox(70, 240, 160, 36), 0.99),
            ],
        )

        _restore_authenticated_source_glyphs(recognition)

        self.assertEqual(
            recognition.transcript,
            "横截面积为0.01 m²。质量为16 kg，取g=10 m/s² 。\n求压强。",
        )

    def test_authenticated_source_repairs_strong_boundary_after_superscript(self) -> None:
        recognition = RecognitionResult(
            provider="fixture",
            transcript="取g=10 m/s²\n根据条件求解。",
            blocks=[
                RecognizedBlock("取g=10 m/s²", BBox(70, 140, 240, 36), 0.99),
                RecognizedBlock("根据条件求解。", BBox(70, 180, 240, 36), 0.99),
            ],
        )

        _restore_authenticated_source_glyphs(recognition)

        self.assertEqual(recognition.transcript, "取g=10 m/s²。\n根据条件求解。")
        self.assertIn("SOURCE_PARAGRAPH_STOP_RECOVERED_REVIEW:1", recognition.warnings)

    def test_close_releases_readback_worker_and_provider(self) -> None:
        pipeline = QuestionPipeline.__new__(QuestionPipeline)
        readback_provider = Mock()
        source_dual_provider = Mock()
        pipeline.readback_provider = readback_provider
        pipeline.source_dual_provider = source_dual_provider
        pipeline.provider = Mock()
        executor = Mock()
        pipeline.source_dual_executor = executor

        pipeline.close()

        executor.shutdown.assert_called_once_with(
            wait=True,
            cancel_futures=True,
        )
        self.assertIsNone(pipeline.source_dual_executor)
        self.assertIsNone(pipeline.source_dual_provider)
        self.assertIsNone(pipeline.readback_provider)
        source_dual_provider.close.assert_called_once_with()
        readback_provider.close.assert_called_once_with()
        pipeline.provider.close.assert_called_once_with()

    def test_parallel_source_dual_ocr_uses_an_isolated_worker(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        config["render_verification"].update(
            {
                "enabled": True,
                "parallel_source_dual_ocr": True,
                "source_dual_ocr_timeout_seconds": 3.0,
            }
        )
        provider = Mock()
        final_readback = Mock()
        source_dual = Mock()
        with patch(
            "test7_ocr.pipeline.TimedRapidOcrProvider",
            side_effect=[final_readback, source_dual],
        ) as timed_provider:
            pipeline = QuestionPipeline(
                provider,
                config,
                PROJECT_ROOT / "artifacts" / "test-runs",
                "isolated-readback-workers",
            )
            pipeline.close()

        self.assertEqual(timed_provider.call_count, 2)
        self.assertIsNot(final_readback, source_dual)
        final_readback_kwargs = timed_provider.call_args_list[0].kwargs
        self.assertEqual(final_readback_kwargs["recycle_after_requests"], None)
        self.assertFalse(final_readback_kwargs["recycle_after_timeout"])
        final_readback.close.assert_called_once_with()
        source_dual.close.assert_called_once_with()

    def test_final_readback_worker_accepts_product_recycle_policy(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
        config["render_verification"].update(
            {
                "enabled": True,
                "render_readback_worker_recycle_after_requests": 1,
                "render_readback_worker_recycle_after_timeout": True,
            }
        )
        provider = Mock()
        final_readback = Mock()
        with patch(
            "test7_ocr.pipeline.TimedRapidOcrProvider",
            return_value=final_readback,
        ) as timed_provider:
            pipeline = QuestionPipeline(
                provider,
                config,
                PROJECT_ROOT / "artifacts" / "test-runs",
                "recycled-final-readback-worker",
            )
            pipeline.close()

        kwargs = timed_provider.call_args.kwargs
        self.assertEqual(kwargs["recycle_after_requests"], 1)
        self.assertTrue(kwargs["recycle_after_timeout"])

    def test_photo_visual_handwriting_evidence_is_review_not_confirmed_handwriting(self) -> None:
        class VisualEvidenceProvider:
            name = "fixture"

            def prepare(self) -> None:
                return None

            def recognize(self, _image_path, sample_id=None):
                return RecognitionResult(
                    provider="fixture",
                    transcript="1. 如图所示，选择正确答案。",
                    blocks=[RecognizedBlock("1. 如图所示，选择正确答案。", BBox(10, 10, 250, 30), 0.99)],
                    visual_bboxes=[BBox(100, 80, 180, 150)],
                    visual_bboxes_verified=False,
                    formula_verified=True,
                )

            def photo_visual_handwriting_evidence(self, _images):
                return {"uncertain_token_count": 3}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.png"
            Image.new("RGB", (400, 300), "white").save(image_path)
            config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
            config["render_verification"]["enabled"] = False
            pipeline = QuestionPipeline(
                VisualEvidenceProvider(),  # type: ignore[arg-type]
                config,
                root / "runs",
                "test-run",
            )
            with patch("test7_ocr.pipeline.photographed_page_risk", return_value=(True, {})):
                report = pipeline.process(image_path, "visual-handwriting")
        self.assertEqual(report.status, "REVIEW")
        issue_codes = [issue.code for issue in report.issues]
        self.assertNotIn("HANDWRITING_PRESENT", issue_codes)
        self.assertIn("PHOTO_VISUAL_HANDWRITING_RISK", issue_codes)

    def test_product_may_defer_secondary_photo_check_after_review_is_locked(self) -> None:
        class ReviewLockedProvider:
            name = "fixture"

            def __init__(self) -> None:
                self.secondary_check_called = False

            def prepare(self) -> None:
                return None

            def recognize(self, _image_path, sample_id=None):
                return RecognitionResult(
                    provider="fixture",
                    transcript="1. 如图所示，选择正确答案。",
                    blocks=[
                        RecognizedBlock(
                            "1. 如图所示，选择正确答案。",
                            BBox(10, 10, 250, 30),
                            0.99,
                        )
                    ],
                    visual_bboxes=[BBox(100, 80, 180, 150)],
                    visual_bboxes_verified=False,
                    formula_verified=True,
                )

            def photo_visual_handwriting_evidence(self, _images):
                self.secondary_check_called = True
                raise AssertionError("secondary crop OCR must be deferred")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.png"
            Image.new("RGB", (400, 300), "white").save(image_path)
            config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
            config["render_verification"]["enabled"] = False
            config.setdefault("recognition", {})[
                "defer_photo_visual_handwriting_check_when_review_locked"
            ] = True
            provider = ReviewLockedProvider()
            pipeline = QuestionPipeline(
                provider,  # type: ignore[arg-type]
                config,
                root / "runs",
                "test-run",
            )
            with patch("test7_ocr.pipeline.photographed_page_risk", return_value=(True, {})):
                report = pipeline.process(image_path, "review-locked")

            report_payload = json.loads(
                (Path(report.output_image).parent / "report.json").read_text(
                    encoding="utf-8"
                )
            )
            document = json.loads(
                (Path(report.output_image).parent / "question_document.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(report.status, "REVIEW")
        self.assertFalse(provider.secondary_check_called)
        self.assertIn(
            "PHOTO_PAGE_HANDWRITING_RISK",
            [issue.code for issue in report.issues],
        )
        self.assertIn(
            "PHOTO_VISUAL_HANDWRITING_CHECK_DEFERRED_REVIEW_LOCKED",
            document["warnings"],
        )
        self.assertEqual(
            report_payload["recognition"]["diagnostics"][
                "photo_visual_handwriting_check"
            ],
            "deferred_review_locked",
        )

    def test_photo_visual_handwritten_grid_is_automatically_rejected(self) -> None:
        class VisualEvidenceProvider:
            name = "fixture"

            def prepare(self) -> None:
                return None

            def recognize(self, _image_path, sample_id=None):
                return RecognitionResult(
                    provider="fixture",
                    transcript="1. 在表格中填入适当的数。",
                    blocks=[RecognizedBlock("1. 在表格中填入适当的数。", BBox(10, 10, 250, 30), 0.99)],
                    visual_bboxes=[BBox(100, 80, 180, 150)],
                    visual_bboxes_verified=False,
                    formula_verified=True,
                )

            def photo_visual_handwriting_evidence(self, _images):
                return {
                    "uncertain_token_count": 2,
                    "likely_handwritten_grid_asset_count": 1,
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "source.png"
            Image.new("RGB", (400, 300), "white").save(image_path)
            config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
            config["render_verification"]["enabled"] = False
            pipeline = QuestionPipeline(
                VisualEvidenceProvider(),  # type: ignore[arg-type]
                config,
                root / "runs",
                "test-run",
            )
            with patch("test7_ocr.pipeline.photographed_page_risk", return_value=(True, {})):
                report = pipeline.process(image_path, "visual-handwritten-grid")
        self.assertEqual(report.status, "REJECT")
        issue_codes = [issue.code for issue in report.issues]
        self.assertIn("HANDWRITING_PRESENT", issue_codes)

    def test_photo_math_handwriting_candidates_require_multiple_uncertain_formulas_and_visual(self) -> None:
        recognition = RecognitionResult(
            provider="fixture",
            transcript="question",
            visual_bboxes=[BBox(100, 100, 200, 120)],
            blocks=[
                RecognizedBlock("a \\rightarrow 0 F_1", BBox(20, 100, 100, 30), 0.54, "formula"),
                RecognizedBlock("BD>BQ", BBox(330, 160, 90, 30), 0.58, "formula"),
                RecognizedBlock("F_1+F_2", BBox(20, 240, 120, 30), 0.94, "formula"),
            ],
        )
        candidates = photo_math_handwriting_candidates(recognition)
        self.assertEqual([block.text for block in candidates], ["a \\rightarrow 0 F_1", "BD>BQ"])
        recognition.visual_bboxes = []
        self.assertEqual(photo_math_handwriting_candidates(recognition), [])

    def test_direct_crop_and_800px_dynamic_render(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "question.png"
            source = Image.new("RGB", (500, 320), "white")
            draw = ImageDraw.Draw(source)
            draw.rectangle((300, 100, 450, 250), outline="black", width=4)
            draw.line((300, 250, 450, 100), fill="black", width=4)
            source.save(image_path)

            sidecar_path = root / "question.ocr.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "image_sha256": sha256_file(image_path),
                        "transcript": "1. 如图，若 $x^{2} \\geq \\frac{1}{2}$，则\nA. 1 B. 2 C. 3 D. 4",
                        "blocks": [
                            {
                                "text": "1. 如图，若 x²≥1/2，则 A. 1 B. 2 C. 3 D. 4",
                                "bbox": [20, 20, 260, 60],
                                "confidence": 0.99,
                                "kind": "text",
                            }
                        ],
                        "visual_bboxes": [[295, 95, 165, 165]],
                        "visual_bboxes_verified": True,
                        "formula_verified": True,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
            config["render_verification"]["enabled"] = False
            pipeline = QuestionPipeline(
                SidecarProvider(sidecar_path),
                config,
                root / "runs",
                "test-run",
            )
            report = pipeline.process(image_path, "fixture-001")

            self.assertEqual(
                report.status,
                "PASS",
                [(issue.code, issue.message) for issue in report.issues],
            )
            self.assertEqual(report.output_width, 800)
            self.assertNotEqual(report.output_height, 720)
            with Image.open(report.output_image) as output:
                self.assertEqual(output.width, 800)

            sample_dir = Path(report.output_image).parent
            expected_crop = source.crop((295, 95, 460, 260))
            with Image.open(sample_dir / "assets" / "asset_001.png") as opened_crop:
                saved_crop = opened_crop.convert("RGB")
                self.assertIsNone(ImageChops.difference(saved_crop, expected_crop).getbbox())

            document = json.loads((sample_dir / "question_document.json").read_text(encoding="utf-8"))
            asset = document["visual_assets"][0]
            self.assertEqual(asset["extraction"], "direct_source_crop")
            self.assertFalse(asset["redraw_used"])
            self.assertEqual(document["schema_version"], "question_document_v2")
            self.assertEqual(document["source_blocks"][0]["source_bbox"]["x"], 20)
            self.assertFalse(document["source_resolution"]["substituted"])
            self.assertEqual(document["source_resolution"]["method"], "direct_input")
            self.assertEqual(
                document["source_resolution"]["input_sha256"],
                document["source_resolution"]["effective_sha256"],
            )

    def test_missing_figure_bbox_never_falls_back_to_whole_question_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "question.png"
            Image.new("RGB", (500, 320), "white").save(image_path)
            sidecar_path = root / "question.ocr.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "image_sha256": sha256_file(image_path),
                        "transcript": "1. \u5982\u56fe\u6240\u793a\uff0c\u6c42\u5c0f\u7403\u7684\u901f\u5ea6\u3002",
                        "blocks": [
                            {
                                "text": "1. \u5982\u56fe\u6240\u793a\uff0c\u6c42\u5c0f\u7403\u7684\u901f\u5ea6\u3002",
                                "bbox": [20, 20, 300, 60],
                                "confidence": 0.99,
                                "kind": "text",
                            }
                        ],
                        "visual_bboxes": [],
                        "visual_bboxes_verified": False,
                        "formula_verified": True,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
            config["render_verification"].update(
                {
                    "enabled": True,
                    "online_render_readback_enabled": False,
                    "parallel_source_dual_ocr": False,
                }
            )
            pipeline = QuestionPipeline(
                SidecarProvider(sidecar_path),
                config,
                root / "runs",
                "test-run",
            )
            report = pipeline.process(image_path, "fixture-no-figure")
            self.assertEqual(report.status, "REJECT")
            self.assertIn("VISUAL_ASSET_MISSING", [issue.code for issue in report.issues])
            sample_dir = Path(report.output_image).parent
            document = json.loads(
                (sample_dir / "question_document.json").read_text(encoding="utf-8")
            )
            self.assertEqual(document["visual_assets"], [])
            self.assertIn(
                "VISUAL_BBOX_MISSING_NO_WHOLE_QUESTION_FALLBACK",
                document["warnings"],
            )
            report_payload = json.loads(
                (sample_dir / "report.json").read_text(encoding="utf-8")
            )
            self.assertNotIn(
                "RENDER_READBACK_PENDING",
                [issue.code for issue in report.issues],
            )
            self.assertNotIn("RENDER_READBACK_PENDING", document["warnings"])
            self.assertEqual(
                report_payload["render_readback"]["reason"],
                "HARD_REJECT_BEFORE_READBACK",
            )

    def test_empty_visual_crop_is_skipped_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "question.png"
            Image.new("RGB", (120, 80), "white").save(image_path)
            sidecar_path = root / "question.ocr.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "image_sha256": sha256_file(image_path),
                        "transcript": "1. 请选择 A. 甲 B. 乙 C. 丙 D. 丁",
                        "blocks": [
                            {
                                "text": "1. 请选择 A. 甲 B. 乙 C. 丙 D. 丁",
                                "bbox": [8, 8, 90, 24],
                                "confidence": 0.99,
                                "kind": "text",
                            }
                        ],
                        "visual_bboxes": [[200, 10, 0, 40]],
                        "visual_bboxes_verified": True,
                        "formula_verified": True,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
            config["render_verification"]["enabled"] = False
            pipeline = QuestionPipeline(
                SidecarProvider(sidecar_path),
                config,
                root / "runs",
                "empty-crop-run",
            )
            report = pipeline.process(image_path, "empty-crop")
            document = json.loads(Path(report.question_document).read_text(encoding="utf-8"))
            self.assertTrue(
                any(str(item).startswith("EMPTY_CROP_SKIPPED:") for item in document["warnings"])
            )
            self.assertEqual(document["visual_assets"], [])

    def test_formula_only_does_not_require_question_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "formula.png"
            Image.new("RGB", (80, 40), "white").save(image_path)
            sidecar_path = root / "formula.ocr.json"
            sidecar_path.write_text(
                json.dumps(
                    {
                        "image_sha256": sha256_file(image_path),
                        "transcript": "a+b",
                        "blocks": [
                            {
                                "text": "a+b",
                                "bbox": [4, 4, 40, 20],
                                "confidence": 0.99,
                                "kind": "formula",
                            }
                        ],
                        "visual_bboxes": [],
                        "formula_verified": False,
                        "formula_specialist_used": True,
                        "formula_count": 1,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config = load_config(PROJECT_ROOT / "configs" / "pipeline.json")
            config["render_verification"]["enabled"] = False
            pipeline = QuestionPipeline(
                SidecarProvider(sidecar_path),
                config,
                root / "runs",
                "formula-only-run",
            )
            report = pipeline.process(image_path, "formula-crop", scoring_mode="formula_only")
            codes = [issue.code for issue in report.issues]
            self.assertNotIn("QUESTION_NUMBER_MISSING", codes)
            self.assertNotIn("EMPTY_OR_TOO_SHORT_TEXT", codes)
            self.assertIn("FORMULA_UNVERIFIED", codes)
            self.assertNotEqual(report.status, "ERROR")
            self.assertTrue(config["quality"].get("question_number_required", True))


if __name__ == "__main__":
    unittest.main()
