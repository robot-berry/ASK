from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from PIL import Image, ImageDraw

from test7_ocr.image_ops import bbox_from_polygon
from test7_ocr.providers import (
    HybridFormulaRapidProvider,
    IncompatibleStructuredServiceError,
    RapidOcrProvider,
    StructuredServiceProvider,
    TimedRapidOcrProvider,
)
from test7_ocr.types import BBox, RecognitionResult, RecognizedBlock


class StructuredProviderTests(unittest.TestCase):
    def test_timed_readback_recycles_after_limit_or_timeout(self) -> None:
        provider = TimedRapidOcrProvider.__new__(TimedRapidOcrProvider)
        provider.recycle_after_timeout = True
        provider.recycle_after_requests = 3
        provider.completed_request_count = 3
        provider._pending_timed_out_request = None
        with patch.object(provider, "_recycle") as recycle:
            provider._prepare_worker_for_request()
        recycle.assert_called_once_with()

        provider.completed_request_count = 0
        provider._pending_timed_out_request = "timed-out"
        with patch.object(provider, "_recycle") as recycle:
            provider._prepare_worker_for_request()
        recycle.assert_called_once_with()

    def test_authenticated_source_requests_deterministic_vram_cleanup(self) -> None:
        self.assertEqual(
            StructuredServiceProvider._runtime_request_headers("GKP-PHY-0001"),
            {"X-Test7-Vram-Cleanup": "mineru_default"},
        )
        self.assertEqual(
            StructuredServiceProvider._runtime_request_headers("TZIP-PHYS-T01-Q01"),
            {},
        )
        self.assertEqual(
            StructuredServiceProvider._runtime_request_headers(None),
            {},
        )

    def test_parenthesized_authenticated_type_tag_is_source_furniture(self) -> None:
        items = [
            {
                "label": "ocr_text",
                "bbox": [20, 20, 620, 52],
                "score": 0.99,
                "text": "高中物理·原创高考风格模拟测试题",
            },
            {
                "label": "ocr_text",
                "bbox": [980, 20, 1230, 52],
                "score": 0.99,
                "text": "GKP-PHY-0001",
            },
            {
                "label": "ocr_text",
                "bbox": [35, 95, 230, 135],
                "score": 0.95,
                "text": "（填空题）",
            },
            {
                "label": "ocr_text",
                "bbox": [35, 160, 700, 200],
                "score": 0.99,
                "text": "题干",
            },
        ]
        suppressed = StructuredServiceProvider._gkp_source_layout_metadata_indexes(
            items,
            image_width=1280,
            sample_id="GKP-PHY-0001",
        )
        self.assertIn(2, suppressed)
        self.assertNotIn(3, suppressed)

    def test_structured_prepare_keeps_secondary_ocr_engine_lazy(self) -> None:
        provider = StructuredServiceProvider(
            verify_roman_headers=True,
            verify_suspicious_option_formulas=True,
            recover_missing_leading_option_labels=True,
            verify_photo_visual_handwriting=True,
        )
        with patch.object(provider, "_get_header_engine") as get_engine:
            provider.prepare()
        get_engine.assert_not_called()
        self.assertIsNone(provider._header_engine)

    def test_authenticated_source_skips_redundant_postconvert_verifiers(self) -> None:
        provider = StructuredServiceProvider(
            verify_roman_headers=True,
            verify_suspicious_option_formulas=True,
        )
        response = {
            "project": "test7-OCR",
            "service_api_version": "structured_service_v2",
            "engine": "test7-local-pp-doclayoutv2-plus-pp-formulanet",
            "device": "test",
            "elapsed_ms": 10,
            "handwriting_filtered_count": 0,
            "items": [
                {
                    "label": "ocr_text",
                    "bbox": [20, 20, 620, 52],
                    "score": 0.99,
                    "text": "高中物理·原创高考风格模拟测试题",
                },
                {
                    "label": "ocr_text",
                    "bbox": [980, 20, 1230, 52],
                    "score": 0.99,
                    "text": "GKP-PHY-0001",
                },
                {
                    "label": "ocr_text",
                    "bbox": [35, 105, 700, 145],
                    "score": 0.98,
                    "text": "1. 题干",
                },
            ],
            "layout_regions": [],
        }
        provider._request_with_transient_retries = lambda *_args, **_kwargs: (response, 0)
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (1280, 720), "white").save(image_path)
            with patch.object(provider, "_verify_roman_question_header") as verify_header:
                with patch.object(
                    provider,
                    "_verify_suspicious_numeric_option_formulas",
                ) as verify_formula:
                    result = provider.recognize(image_path, sample_id="GKP-PHY-0001")
        verify_header.assert_not_called()
        verify_formula.assert_not_called()
        self.assertEqual(
            result.diagnostics[
                "structured_secondary_ocr_skipped_authenticated_source"
            ],
            ["roman_question_header", "suspicious_numeric_option_formula"],
        )

    def test_hybrid_preserves_printed_leading_affix_from_rapid_line(self) -> None:
        self.assertEqual(
            HybridFormulaRapidProvider._leading_printed_affix(
                "计算：√12+|2-√3|"
            ),
            "计算：",
        )
        self.assertEqual(
            HybridFormulaRapidProvider._trailing_printed_affix("M+m)。"),
            ")。",
        )
        self.assertEqual(
            HybridFormulaRapidProvider._trailing_printed_affix(
                "g1、g2，则下列关系正确的是（"
            ),
            "，则下列关系正确的是（",
        )

    def test_hybrid_preserves_tiny_trailing_punctuation_beside_formula(self) -> None:
        class FakeTextProvider:
            _engine = None

        provider = HybridFormulaRapidProvider(text_provider=FakeTextProvider())
        recognition = RecognitionResult(
            provider="fixture",
            transcript="M+m)。",
            blocks=[RecognizedBlock("M+m)。", BBox(100, 50, 80, 40), 0.98)],
        )
        items, warnings, diagnostics = provider._formula_aware_text_items(
            Image.new("RGB", (250, 140), "white"),
            recognition,
            [{"bbox": [90, 45, 169, 95], "latex": r"\frac{M+m}{mt_1}"}],
        )
        self.assertEqual([item["text"] for item in items], [")。"])
        self.assertEqual(diagnostics["hybrid_formula_text_lines_unresolved"], 0)
        self.assertNotIn("HYBRID_FORMULA_TEXT_SEGMENT_UNRESOLVED:1", warnings)

    def test_hybrid_filters_detached_formula_working_and_margin_note(self) -> None:
        image = Image.new("RGB", (500, 260), "white")
        draw = ImageDraw.Draw(image)
        for x in range(100, 390, 14):
            draw.rectangle((x, 90, x + 7, 120), fill="black")
        for x in range(220, 340, 16):
            draw.rectangle((x, 165, x + 8, 195), fill="black")
        rapid = RecognitionResult(
            provider="fixture",
            transcript="",
            blocks=[
                RecognizedBlock("17. （本题满分5分）", BBox(40, 40, 180, 35), 0.99),
                RecognizedBlock("G7FN", BBox(350, 40, 55, 35), 0.70),
                RecognizedBlock("计算：bad", BBox(80, 80, 330, 60), 0.60),
                RecognizedBlock("=√3", BBox(215, 155, 130, 50), 0.70),
            ],
        )
        formula_items = [
            {
                "label": "display_formula",
                "bbox": [90, 75, 410, 215],
                "score": 0.75,
                "latex": r"x+1=\sqrt{3}",
            }
        ]
        warnings, boxes, ignored, diagnostics = (
            HybridFormulaRapidProvider._filter_detached_formula_working(
                image,
                rapid,
                formula_items,
            )
        )
        self.assertEqual(warnings, ["PHOTO_DETACHED_FORMULA_WORKING_FILTERED:1"])
        self.assertEqual(diagnostics["hybrid_detached_formula_working_filtered"], 1)
        self.assertEqual(formula_items[0]["latex"], "x+1")
        self.assertEqual({block.text for block in rapid.blocks}, {"17. （本题满分5分）", "计算：bad"})
        self.assertEqual(
            {block.reason for block in ignored},
            {"detached_formula_working", "photo_margin_handwriting_annotation"},
        )
        self.assertEqual(len(boxes), 2)

    def test_hybrid_does_not_filter_a_printed_stacked_fraction_equality(self) -> None:
        image = Image.new("RGB", (500, 220), "white")
        draw = ImageDraw.Draw(image)
        for x in range(100, 390, 14):
            draw.rectangle((x, 80, x + 7, 98), fill="black")
        for x in range(100, 390, 14):
            draw.rectangle((x, 120, x + 7, 138), fill="black")
        rapid = RecognitionResult(
            provider="fixture",
            transcript="",
            blocks=[RecognizedBlock("8. 数列满足", BBox(20, 70, 420, 32), 0.99)],
        )
        formula_items = [
            {
                "label": "inline_formula",
                "bbox": [90, 70, 410, 145],
                "score": 0.96,
                "latex": r"\frac{2a_n+4}{a_{n+1}}=\frac{3a_n+2-a_{n+1}}{a_n}",
            }
        ]
        warnings, boxes, ignored, diagnostics = (
            HybridFormulaRapidProvider._filter_detached_formula_working(
                image,
                rapid,
                formula_items,
            )
        )
        self.assertEqual(warnings, [])
        self.assertEqual(boxes, [])
        self.assertEqual(ignored, [])
        self.assertEqual(diagnostics["hybrid_detached_formula_working_filtered"], 0)
        self.assertIn("=", formula_items[0]["latex"])

    def test_rapidocr_main_read_resets_persistent_engine_flags(self) -> None:
        class FakeEngine:
            def __call__(self, _image, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(boxes=None, txts=(), scores=())

        provider = object.__new__(RapidOcrProvider)
        provider._engine = FakeEngine()
        provider.use_cls = True
        provider.recognize_image(Image.new("RGB", (40, 20), "white"))
        self.assertEqual(
            provider._engine.kwargs,
            {"use_det": True, "use_cls": True, "use_rec": True},
        )

    def test_polygon_bbox_accepts_array_like_rapidocr_output(self) -> None:
        class ArrayLike:
            def __iter__(self):
                return iter(((1.2, 2.2), (10.1, 2.2), (10.1, 6.1), (1.2, 6.1)))

        self.assertEqual(bbox_from_polygon(ArrayLike()), BBox(1, 2, 10, 5))

    def test_hybrid_text_crop_restores_source_coordinates(self) -> None:
        class FakeTextProvider:
            _engine = object()

            def prepare(self):
                return None

            def recognize_image(self, image):
                self.received_size = image.size
                return RecognitionResult(
                    provider="fake-rapidocr",
                    transcript="1. 题干",
                    blocks=[
                        RecognizedBlock(
                            text="1. 题干",
                            bbox=BBox(5, 6, 20, 10),
                            confidence=0.99,
                        )
                    ],
                )

        text_provider = FakeTextProvider()
        provider = HybridFormulaRapidProvider(
            text_provider=text_provider,
            outer_white_threshold=248,
            outer_crop_padding_px=4,
        )
        response = {
            "project": "test7-OCR",
            "service_api_version": "structured_service_v2",
            "engine": "test7-local-pp-doclayoutv2-plus-pp-formulanet-formula-only",
            "device": "test",
            "formula_items": [
                {
                    "label": "inline_formula",
                    "bbox": [100, 40, 140, 70],
                    "score": 0.99,
                    "latex": "x^2",
                }
            ],
            "layout_regions": [],
            "elapsed_ms": 10,
            "layout_ms": 4,
            "formula_ms": 6,
        }
        provider._request_with_transient_retries = lambda *_args, **_kwargs: (response, 0)
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            image = Image.new("RGB", (200, 120), "white")
            image.paste("black", (40, 30, 160, 90))
            image.save(image_path)
            result = provider.recognize(image_path)

        self.assertEqual(text_provider.received_size, (128, 68))
        text_block = next(block for block in result.blocks if block.kind != "formula")
        self.assertEqual(text_block.bbox, BBox(41, 32, 20, 10))
        self.assertIn("题干", result.transcript)
        self.assertIn("x^2", result.transcript)
        self.assertEqual(result.diagnostics["hybrid_text_crop_bbox"], [36, 26, 128, 68])
        self.assertEqual(result.diagnostics["hybrid_text_block_count"], 1)

    def test_hybrid_resegments_only_text_beside_formula_boxes(self) -> None:
        class FakeEngine:
            def __init__(self):
                self.responses = iter((("题干", 0.99), ("条件", 0.98)))
                self.shapes = []

            def __call__(self, image, **kwargs):
                self.kwargs = kwargs
                self.shapes.append(image.shape)
                text, score = next(self.responses)
                return SimpleNamespace(txts=(text,), scores=(score,))

        class FakeTextProvider:
            def __init__(self):
                self._engine = FakeEngine()

        provider = HybridFormulaRapidProvider(text_provider=FakeTextProvider())
        recognition = RecognitionResult(
            provider="fake-rapidocr",
            transcript="题干 x=1 条件",
            blocks=[
                RecognizedBlock(
                    text="题干 x=1 条件",
                    bbox=BBox(10, 20, 180, 30),
                    confidence=0.99,
                )
            ],
        )
        items, warnings, diagnostics = provider._formula_aware_text_items(
            Image.new("RGB", (200, 80), "white"),
            recognition,
            [{"bbox": [80, 18, 120, 52], "latex": "x=1"}],
        )
        self.assertEqual([item["text"] for item in items], ["题干", "条件"])
        self.assertEqual([item["bbox"] for item in items], [[10, 20, 78, 50], [122, 20, 190, 50]])
        self.assertEqual(diagnostics["hybrid_formula_intersecting_text_lines"], 1)
        self.assertEqual(diagnostics["hybrid_formula_text_segments_recovered"], 2)
        self.assertEqual(diagnostics["hybrid_formula_text_lines_unresolved"], 0)
        self.assertIn("HYBRID_FORMULA_TEXT_SEGMENTS_RECOVERED:1:2", warnings)
        self.assertEqual(
            provider._header_engine.kwargs,
            {"use_det": False, "use_cls": False, "use_rec": True},
        )
        # The source segment is 68 px wide and 42 px high after vertical source
        # padding; recognizer-only mode receives an additional white safety border.
        self.assertGreater(provider._header_engine.shapes[0][1], 68)
        self.assertGreater(provider._header_engine.shapes[0][0], 42)

    def test_hybrid_rereads_low_confidence_long_printed_line(self) -> None:
        class FakeEngine:
            def __call__(self, _image, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    txts=("D. 完整恢复后的选项文字",),
                    scores=(0.97,),
                )

        class FakeTextProvider:
            def __init__(self):
                self._engine = FakeEngine()

        provider = HybridFormulaRapidProvider(text_provider=FakeTextProvider())
        recognition = RecognitionResult(
            provider="fake-rapidocr",
            transcript="乱码",
            blocks=[
                RecognizedBlock(
                    text="乱码",
                    bbox=BBox(20, 30, 360, 30),
                    confidence=0.55,
                )
            ],
        )
        warnings, diagnostics = provider._reread_low_confidence_text_lines(
            Image.new("RGB", (420, 100), "white"),
            recognition,
        )
        self.assertEqual(recognition.blocks[0].text, "D. 完整恢复后的选项文字")
        self.assertEqual(recognition.blocks[0].confidence, 0.97)
        self.assertEqual(diagnostics["hybrid_low_confidence_lines_reread"], 1)
        self.assertEqual(diagnostics["hybrid_low_confidence_lines_recovered"], 1)
        self.assertIn("HYBRID_LOW_CONFIDENCE_LINES_RECOVERED:1", warnings)
        self.assertEqual(
            provider._header_engine.kwargs,
            {"use_det": False, "use_cls": False, "use_rec": True},
        )

    def test_hybrid_recovers_text_gaps_on_formula_only_row(self) -> None:
        class FakeEngine:
            def __init__(self):
                self.responses = iter(
                    (("1. 设集合", 0.96), ("，集合", 0.99), ("，则", 0.98))
                )

            def __call__(self, _image, **kwargs):
                self.kwargs = kwargs
                text, score = next(self.responses)
                return SimpleNamespace(txts=(text,), scores=(score,))

        class FakeTextProvider:
            def __init__(self):
                self._engine = FakeEngine()

        provider = HybridFormulaRapidProvider(text_provider=FakeTextProvider())
        source = Image.new("RGB", (400, 100), "white")
        source.paste("black", (10, 35, 70, 55))
        source.paste("black", (160, 35, 210, 55))
        source.paste("black", (300, 35, 330, 55))
        items, warnings, diagnostics = provider._recover_formula_only_row_text_gaps(
            source,
            RecognitionResult(provider="fake", transcript="", blocks=[]),
            [
                {"bbox": [72, 25, 158, 65], "latex": "A"},
                {"bbox": [212, 25, 298, 65], "latex": "B"},
                {"bbox": [332, 25, 390, 65], "latex": "C"},
            ],
            BBox(0, 20, 400, 55),
        )
        self.assertEqual([item["text"] for item in items], ["1. 设集合", "，集合", "，则"])
        self.assertIn("HYBRID_FORMULA_ONLY_ROW_TEXT_RECOVERED:3", warnings)
        self.assertEqual(diagnostics["hybrid_formula_only_rows_scanned"], 1)
        self.assertEqual(diagnostics["hybrid_formula_only_gap_segments_recovered"], 3)

    def test_same_row_visuals_are_sorted_by_x_despite_y_jitter(self) -> None:
        boxes = [
            BBox(620, 100, 180, 200),
            BBox(20, 104, 180, 195),
            BBox(820, 112, 180, 188),
            BBox(220, 118, 180, 184),
        ]
        ordered = StructuredServiceProvider._visual_reading_order(boxes)
        self.assertEqual([box.x for box in ordered], [20, 220, 620, 820])

    def test_labeled_visual_strip_keeps_panels_and_labels_together(self) -> None:
        visuals = [
            BBox(100, 100, 120, 90),
            BBox(240, 96, 110, 96),
            BBox(370, 103, 100, 84),
            BBox(490, 101, 90, 86),
        ]
        items = [
            {"label": "ocr_text", "text": label, "bbox": bbox}
            for label, bbox in zip(
                ("甲", "乙", "丙", "丁"),
                (
                    [145, 194, 170, 220],
                    [280, 196, 305, 222],
                    [405, 193, 430, 219],
                    [520, 191, 545, 217],
                ),
            )
        ]
        grouped, suppressed, count = StructuredServiceProvider._group_labeled_visual_strip(
            items,
            visuals,
        )
        self.assertEqual(count, 4)
        self.assertEqual(suppressed, {0, 1, 2, 3})
        self.assertEqual(len(grouped), 1)
        self.assertLessEqual(grouped[0].x, 100)
        self.assertGreaterEqual(grouped[0].bottom, 222)

    def test_decimal_axis_labels_do_not_start_a_new_question_window(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 900, 55], "score": 1, "text": "13. 如图所示，选择正确说法"},
                {"label": "ocr_text", "bbox": [100, 180, 500, 210], "score": 1, "text": "-0.2 -0.1 0 0.1 0.2 0.3 x/cm"},
                {"label": "ocr_text", "bbox": [60, 260, 500, 290], "score": 1, "text": "A. 选项甲"},
                {"label": "ocr_text", "bbox": [60, 300, 500, 330], "score": 1, "text": "B. 选项乙"},
                {"label": "ocr_text", "bbox": [60, 340, 500, 370], "score": 1, "text": "C. 选项丙"},
                {"label": "ocr_text", "bbox": [60, 380, 500, 410], "score": 1, "text": "D. 选项丁"},
            ],
            "layout_regions": [{"bbox": [80, 120, 620, 240]}],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertIn("13. 如图所示", result.transcript)
        self.assertNotIn("0.1 0.2", result.transcript)
        self.assertEqual(result.visual_bboxes, [BBox(80, 120, 540, 120)])
        self.assertNotIn("DOMINANT_QUESTION_WINDOW_SELECTED", result.warnings)
        self.assertTrue(
            any(
                block.text.startswith("-0.2") and block.reason == "embedded_visual_text"
                for block in result.ignored_blocks
            )
        )

    def test_visual_answer_options_keep_labels_and_source_layout_in_one_crop(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "transcript": "7. question\nA B C D",
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 500, 55], "score": 1, "text": "7. question"},
                {"label": "ocr_text", "bbox": [80, 330, 105, 360], "score": 1, "text": "A"},
                {"label": "ocr_text", "bbox": [330, 332, 355, 362], "score": 1, "text": "B"},
                {"label": "ocr_text", "bbox": [580, 329, 605, 359], "score": 1, "text": "C"},
                {"label": "ocr_text", "bbox": [830, 331, 855, 361], "score": 1, "text": "D"},
            ],
            "layout_regions": [
                {"bbox": [520, 100, 680, 325]},
                {"bbox": [20, 102, 180, 325]},
                {"bbox": [770, 107, 930, 325]},
                {"bbox": [270, 110, 430, 325]},
            ],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertEqual(len(result.visual_bboxes), 1)
        self.assertLessEqual(result.visual_bboxes[0].x, 20)
        self.assertGreaterEqual(result.visual_bboxes[0].bottom, 362)
        self.assertNotIn("A B C D", result.transcript)
        self.assertIn("VISUAL_OPTION_GROUP_PRESERVED_FROM_SOURCE:4->1", result.warnings)
        self.assertEqual(
            [block.text for block in result.ignored_blocks if block.reason == "embedded_visual_text"],
            ["A", "B", "C", "D"],
        )

    def test_transient_connection_failure_is_retried_and_audited(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "project": "test7-OCR",
                        "service_api_version": "structured_service_v2",
                        "engine": "test7-local-pp-doclayoutv2-plus-pp-formulanet",
                        "transcript": "1. test",
                        "items": [
                            {
                                "label": "ocr_text",
                                "bbox": [10, 10, 100, 40],
                                "score": 0.99,
                                "text": "1. test",
                            }
                        ],
                        "layout_regions": [],
                        "handwriting_filtered_count": 0,
                    }
                ).encode("utf-8")

        provider = StructuredServiceProvider(
            transient_retry_attempts=3,
            transient_retry_base_seconds=0.01,
        )
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            with patch(
                "test7_ocr.providers.urlopen",
                side_effect=[URLError("connection refused"), FakeResponse()],
            ) as mocked_urlopen, patch("test7_ocr.providers.time.sleep") as mocked_sleep:
                result = provider.recognize(image_path)
        self.assertEqual(mocked_urlopen.call_count, 2)
        mocked_sleep.assert_called_once_with(0.01)
        self.assertIn("STRUCTURED_SERVICE_TRANSIENT_RETRY:1", result.warnings)
        self.assertEqual(result.diagnostics["structured_service_retry_count"], 1)
        self.assertEqual(
            result.diagnostics["structured_service_api_version"],
            "structured_service_v2",
        )

    def test_incompatible_structured_service_never_silently_falls_back(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "engine": "local-pp-doclayoutv2-plus-pp-formulanet",
                        "transcript": "old service",
                        "items": [],
                        "layout_regions": [],
                    }
                ).encode("utf-8")

        class ForbiddenFallback:
            def recognize(self, *_args, **_kwargs):
                raise AssertionError("incompatible service must not fall back")

        provider = StructuredServiceProvider(fallback=ForbiddenFallback())
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            with patch("test7_ocr.providers.urlopen", return_value=FakeResponse()):
                with self.assertRaises(IncompatibleStructuredServiceError):
                    provider.recognize(image_path)

    def test_non_transient_http_error_is_not_retried(self) -> None:
        provider = StructuredServiceProvider(
            transient_retry_attempts=5,
            transient_retry_base_seconds=0.0,
        )
        with self.assertRaises(HTTPError), patch(
            "test7_ocr.providers.urlopen",
            side_effect=HTTPError("http://127.0.0.1:8240/recognize", 400, "bad", {}, None),
        ) as mocked_urlopen:
            provider._request_with_transient_retries(b"not-an-image")
        mocked_urlopen.assert_called_once()

    def test_photo_visual_requires_multiple_uncertain_compact_tokens(self) -> None:
        class FakeEngine:
            def __call__(self, _image, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    boxes=None,
                    txts=("4", "9", "10", "第16题图"),
                    scores=(0.50, 0.60, 0.999, 0.999),
                )

        provider = StructuredServiceProvider(
            verify_photo_visual_handwriting=True,
            visual_low_confidence_threshold=0.72,
            visual_low_confidence_token_count=2,
        )
        engine = FakeEngine()
        provider._header_engine = engine
        evidence = provider.photo_visual_handwriting_evidence(
            [Image.new("RGB", (200, 200), "white")]
        )
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["uncertain_token_count"], 2)
        self.assertTrue(engine.kwargs["use_det"])

    def test_photo_visual_numeric_grid_is_strong_handwriting_evidence(self) -> None:
        class FakeEngine:
            def __call__(self, _image, **_kwargs):
                return SimpleNamespace(
                    boxes=None,
                    txts=("4", "9", "10"),
                    scores=(0.50, 0.60, 0.999),
                )

        grid = Image.new("RGB", (220, 220), "white")
        draw = ImageDraw.Draw(grid)
        for coordinate in (20, 80, 140, 200):
            draw.line((20, coordinate, 200, coordinate), fill="black", width=5)
            draw.line((coordinate, 20, coordinate, 200), fill="black", width=5)
        provider = StructuredServiceProvider(
            verify_photo_visual_handwriting=True,
            visual_low_confidence_threshold=0.72,
            visual_low_confidence_token_count=2,
        )
        provider._header_engine = FakeEngine()
        evidence = provider.photo_visual_handwriting_evidence([grid])
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["likely_handwritten_grid_asset_count"], 1)
        self.assertTrue(evidence["assets"][0]["likely_handwritten_grid"])

    def test_photo_visual_preserves_borderline_printed_diagram_labels(self) -> None:
        class FakeEngine:
            def __call__(self, _image, **_kwargs):
                return SimpleNamespace(
                    boxes=None,
                    txts=("D", "C", "R1", "第17题图"),
                    scores=(0.694, 0.698, 0.994, 0.999),
                )

        provider = StructuredServiceProvider(
            verify_photo_visual_handwriting=True,
            visual_low_confidence_threshold=0.65,
            visual_low_confidence_token_count=2,
        )
        provider._header_engine = FakeEngine()
        evidence = provider.photo_visual_handwriting_evidence(
            [Image.new("RGB", (200, 200), "white")]
        )
        self.assertIsNone(evidence)

    def test_photo_visual_checks_multiple_assets_in_one_collage_call(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, image, **_kwargs):
                self.calls += 1
                self.shape = image.shape
                return SimpleNamespace(
                    boxes=(
                        [[10, 10], [30, 10], [30, 30], [10, 30]],
                        [[40, 10], [60, 10], [60, 30], [40, 30]],
                        [[134, 10], [154, 10], [154, 30], [134, 30]],
                    ),
                    txts=("4", "9", "R1"),
                    scores=(0.50, 0.60, 0.99),
                )

        provider = StructuredServiceProvider(
            verify_photo_visual_handwriting=True,
            visual_low_confidence_threshold=0.65,
            visual_low_confidence_token_count=2,
        )
        engine = FakeEngine()
        provider._header_engine = engine
        evidence = provider.photo_visual_handwriting_evidence(
            [Image.new("RGB", (110, 80), "white"), Image.new("RGB", (110, 80), "white")]
        )
        self.assertEqual(engine.calls, 1)
        self.assertGreater(engine.shape[1], 220)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["assets"][0]["asset_index"], 1)
        self.assertEqual(evidence["uncertain_token_count"], 2)

    def test_photo_visual_collage_does_not_repeat_largest_cell_dimensions(self) -> None:
        class FakeEngine:
            def __call__(self, image, **_kwargs):
                self.shape = image.shape
                return SimpleNamespace(boxes=None, txts=(), scores=())

        provider = StructuredServiceProvider(verify_photo_visual_handwriting=True)
        engine = FakeEngine()
        provider._header_engine = engine
        images = [
            Image.new("RGB", (469, 214), "white"),
            Image.new("RGB", (92, 47), "white"),
            Image.new("RGB", (367, 256), "white"),
            Image.new("RGB", (94, 45), "white"),
            Image.new("RGB", (159, 41), "white"),
        ]

        self.assertIsNone(provider.photo_visual_handwriting_evidence(images))
        packed_height, packed_width = engine.shape[:2]
        self.assertLess(packed_width * packed_height, 400_000)
        self.assertGreaterEqual(packed_width, 469)

    def test_missing_leading_option_label_requires_two_source_crop_confirmations(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.responses = iter(
                    [
                        SimpleNamespace(txts=("A",), scores=(0.99,)),
                        SimpleNamespace(txts=("A.",), scores=(0.98,)),
                    ]
                )

            def __call__(self, _image, **_kwargs):
                return next(self.responses)

        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 500, 55], "score": 1, "text": "下列说法正确的是"},
                {"label": "inline_formula", "bbox": [65, 100, 110, 140], "score": 0.9, "latex": "x^2"},
                {"label": "ocr_text", "bbox": [110, 115, 500, 145], "score": 1, "text": "option one"},
                {"label": "ocr_text", "bbox": [30, 180, 500, 220], "score": 1, "text": "B. option two"},
                {"label": "ocr_text", "bbox": [30, 250, 500, 290], "score": 1, "text": "C. option three"},
                {"label": "ocr_text", "bbox": [30, 320, 500, 360], "score": 1, "text": "D. option four"},
            ],
            "layout_regions": [],
        }
        provider = StructuredServiceProvider(
            recover_missing_leading_option_labels=True,
            option_label_min_confidence=0.95,
        )
        provider._header_engine = FakeEngine()
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (800, 500), "white").save(image_path)
            provider._recover_missing_leading_option_label(image_path, payload)
        self.assertEqual(payload["option_labels_recovered_count"], 1)
        self.assertTrue(any(item.get("text") == "A." for item in payload["items"]))
        self.assertIn(
            "MISSING_OPTION_A_RECOVERED_BY_SECONDARY_OCR:0.980",
            payload["client_warnings"],
        )

    def test_horizontal_missing_a_label_requires_two_source_crop_confirmations(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.responses = iter(
                    [
                        SimpleNamespace(txts=("A",), scores=(0.99,)),
                        SimpleNamespace(txts=("A.",), scores=(0.98,)),
                    ]
                )

            def __call__(self, _image, **_kwargs):
                return next(self.responses)

        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 500, 55], "score": 1, "text": "1. question"},
                {"label": "inline_formula", "bbox": [75, 180, 120, 240], "score": 0.9, "latex": "1/2"},
                {"label": "ocr_text", "bbox": [300, 195, 335, 225], "score": 1, "text": "B."},
                {"label": "inline_formula", "bbox": [345, 180, 390, 240], "score": 0.9, "latex": "2/3"},
                {"label": "ocr_text", "bbox": [570, 194, 605, 224], "score": 1, "text": "C."},
                {"label": "inline_formula", "bbox": [615, 180, 660, 240], "score": 0.9, "latex": "3/4"},
                {"label": "ocr_text", "bbox": [840, 196, 875, 226], "score": 1, "text": "D."},
                {"label": "inline_formula", "bbox": [885, 180, 930, 240], "score": 0.9, "latex": "4/5"},
            ],
            "layout_regions": [],
        }
        provider = StructuredServiceProvider(
            recover_missing_leading_option_labels=True,
            option_label_min_confidence=0.95,
        )
        provider._header_engine = FakeEngine()
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (1000, 300), "white").save(image_path)
            provider._recover_missing_leading_option_label(image_path, payload)
        self.assertEqual(payload["option_labels_recovered_count"], 1)
        recovered = next(item for item in payload["items"] if item.get("text") == "A.")
        self.assertLess(recovered["bbox"][0], 75)

    def test_pascal_triangle_text_cluster_is_promoted_to_visual_crop(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "transcript": "11. 如图所示\nA. first 第1行 1 1\nB. second 第2行 1 2 1\nC. third\nD. fourth",
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 650, 55], "score": 1, "text": "11. 如图所示"},
                {"label": "ocr_text", "bbox": [20, 100, 250, 130], "score": 1, "text": "A. first"},
                {"label": "ocr_text", "bbox": [20, 160, 250, 190], "score": 1, "text": "B. second"},
                {"label": "ocr_text", "bbox": [20, 220, 250, 250], "score": 1, "text": "C. third"},
                {"label": "ocr_text", "bbox": [20, 280, 250, 310], "score": 1, "text": "D. fourth"},
                {"label": "ocr_text", "bbox": [720, 100, 800, 130], "score": 1, "text": "第1行"},
                {"label": "ocr_text", "bbox": [900, 100, 950, 130], "score": 1, "text": "1 1"},
                {"label": "ocr_text", "bbox": [720, 145, 800, 175], "score": 1, "text": "第2行"},
                {"label": "ocr_text", "bbox": [880, 145, 970, 175], "score": 1, "text": "1 2 1"},
                {"label": "ocr_text", "bbox": [720, 190, 800, 220], "score": 1, "text": "第3行"},
                {"label": "ocr_text", "bbox": [860, 190, 990, 220], "score": 1, "text": "1 3 3 1"},
                {"label": "ocr_text", "bbox": [760, 260, 980, 292], "score": 1, "text": "第11题图 杨辉三角"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1100)
        self.assertEqual(len(result.visual_bboxes), 1)
        self.assertEqual(result.visual_bbox_source, "structured_text_diagram_candidate")
        self.assertNotIn("第1行", result.transcript)
        self.assertIn("TEXT_DIAGRAM_PROMOTED_TO_DIRECT_CROP", result.warnings)

    def test_stacked_fraction_is_retried_on_a_tight_source_crop(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {
                    "label": "ocr_text",
                    "bbox": [20, 90, 500, 120],
                    "score": 1,
                    "text": "逃逸速度为ν'=",
                },
                {"label": "ocr_text", "bbox": [515, 80, 590, 120], "score": 1, "text": "2Gm"},
                {"label": "ocr_text", "bbox": [535, 120, 565, 150], "score": 1, "text": "R"},
            ],
            "layout_regions": [],
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "items": [
                            {
                                "label": "display_formula",
                                "bbox": [0, 0, 120, 70],
                                "score": 0.72,
                                "latex": r"v = \sqrt{\frac{2Gm}{R}}",
                            }
                        ]
                    }
                ).encode("utf-8")

        provider = StructuredServiceProvider(
            refine_stacked_fraction_formulas=True,
            refined_formula_min_confidence=0.50,
        )
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (800, 300), "white").save(image_path)
            with patch("test7_ocr.providers.urlopen", return_value=FakeResponse()):
                provider._refine_stacked_fraction_regions(image_path, payload)
        self.assertEqual(payload["formula_regions_refined_count"], 1)
        formula = next(item for item in payload["items"] if item["label"] == "inline_formula")
        self.assertEqual(formula["latex"], r"ν' = \sqrt{\frac{2Gm}{R}}")
        self.assertFalse(any(item.get("text") == "2Gm" for item in payload["items"]))
        self.assertIn(
            "STACKED_FRACTION_REFINED_FROM_SOURCE_CROP:0.720",
            payload["client_warnings"],
        )

    def test_unresolved_stacked_fraction_is_explicitly_fail_closed(self) -> None:
        payload = {
            "items": [
                {
                    "label": "ocr_text",
                    "bbox": [20, 90, 500, 120],
                    "score": 1,
                    "text": "逃逸速度为ν'=",
                },
                {"label": "ocr_text", "bbox": [515, 80, 590, 120], "score": 1, "text": "2Gm"},
                {"label": "ocr_text", "bbox": [535, 120, 565, 150], "score": 1, "text": "R"},
            ]
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return json.dumps({"items": []}).encode("utf-8")

        provider = StructuredServiceProvider(refine_stacked_fraction_formulas=True)
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (800, 300), "white").save(image_path)
            with patch("test7_ocr.providers.urlopen", return_value=FakeResponse()):
                provider._refine_stacked_fraction_regions(image_path, payload)
        self.assertIn("STACKED_FRACTION_REFINEMENT_UNRESOLVED", payload["client_warnings"])

    def test_stacked_characters_are_split_only_when_adjacent_rows_confirm_them(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "transcript": "为数 $E=mc^2$ $G=1$",
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 50, 90], "score": 1, "text": "为数"},
                {"label": "inline_formula", "bbox": [55, 20, 180, 50], "score": 0.95, "latex": "E=mc^2"},
                {"label": "inline_formula", "bbox": [55, 60, 180, 90], "score": 0.95, "latex": "G=1"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertNotIn("为数", result.transcript)
        self.assertRegex(result.transcript, r"为\s+\$E=mc\^2\$")
        self.assertRegex(result.transcript, r"数\s+\$G=1\$")
        self.assertIn("STACKED_TEXT_ROWS_SPLIT:1", result.warnings)

    def test_secondary_ocr_removes_low_confidence_numeric_option_suffix(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.responses = iter(
                    [
                        SimpleNamespace(txts=("8m",), scores=(0.75,)),
                        SimpleNamespace(txts=("8",), scores=(0.99,)),
                        SimpleNamespace(txts=("8",), scores=(0.98,)),
                    ]
                )

            def __call__(self, _image, **_kwargs):
                return next(self.responses)

        payload = {
            "handwriting_filtered_count": 0,
            "transcript": "D. $8 _ { m }$",
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 55, 55], "score": 1, "text": "D."},
                {"label": "inline_formula", "bbox": [60, 18, 100, 60], "score": 0.58, "latex": "8 _ { m }"},
            ],
            "layout_regions": [],
        }
        provider = StructuredServiceProvider(
            verify_suspicious_option_formulas=True,
            option_formula_min_confidence=0.92,
        )
        provider._header_engine = FakeEngine()
        result = provider._convert(payload, image_width=200)
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            provider._verify_suspicious_numeric_option_formulas(image_path, payload, result)
        self.assertIn("$8$", result.transcript)
        self.assertNotIn("{ m }", result.transcript)
        self.assertEqual(result.blocks[-1].text, "8")
        self.assertIn(
            "OPTION_FORMULA_SUFFIX_REMOVED_BY_SECONDARY_OCR:8:0.980",
            result.warnings,
        )

    def test_right_side_fragment_clipped_after_option_d_is_trimmed(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 800, 55], "score": 1, "text": "3. question"},
                {"label": "ocr_text", "bbox": [40, 100, 430, 130], "score": 1, "text": "A. option one"},
                {"label": "ocr_text", "bbox": [40, 150, 430, 180], "score": 1, "text": "B. option two"},
                {"label": "ocr_text", "bbox": [40, 200, 430, 230], "score": 1, "text": "C. option three"},
                {"label": "ocr_text", "bbox": [40, 250, 500, 280], "score": 1, "text": "D. option four"},
                {"label": "ocr_text", "bbox": [500, 272, 790, 302], "score": 1, "text": "\u6211\u56fd\u5c55\u793a\u4e86\u65b0\u578b\u8bbe\u5907"},
                {"label": "inline_formula", "bbox": [800, 270, 880, 302], "score": 1, "latex": "50^{\\circ}"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertNotIn("\u6211\u56fd\u5c55\u793a", result.transcript)
        self.assertNotIn("50^{\\circ}", result.transcript)
        self.assertIn("CLIPPED_POST_OPTION_FRAGMENT_TRIMMED:2", result.warnings)

    def test_new_exam_section_after_complete_options_is_trimmed(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 800, 55], "score": 1, "text": "10. question"},
                {"label": "ocr_text", "bbox": [40, 100, 430, 130], "score": 1, "text": "A. option one"},
                {"label": "ocr_text", "bbox": [40, 150, 430, 180], "score": 1, "text": "B. option two"},
                {"label": "ocr_text", "bbox": [40, 200, 430, 230], "score": 1, "text": "C. option three"},
                {"label": "ocr_text", "bbox": [40, 250, 500, 280], "score": 1, "text": "D. option four"},
                {"label": "ocr_text", "bbox": [20, 310, 900, 345], "score": 1, "text": "二、选择题Ⅱ（本题共3小题）"},
                {"label": "ocr_text", "bbox": [20, 360, 900, 395], "score": 1, "text": "每小题列出的备选项中至少有一个正确"},
            ],
            "layout_regions": [{"bbox": [600, 350, 900, 480]}],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertNotIn("选择题Ⅱ", result.transcript)
        self.assertNotIn("每小题列出", result.transcript)
        self.assertEqual(result.visual_bboxes, [])
        self.assertIn("POST_OPTION_SECTION_HEADING_TRIMMED", result.warnings)

    def test_high_confidence_secondary_ocr_corrects_roman_question_header(self) -> None:
        result = StructuredServiceProvider()._convert(
            {
                "handwriting_filtered_count": 0,
                "transcript": "14-II. question text",
                "items": [
                    {
                        "label": "ocr_text",
                        "bbox": [20, 20, 500, 55],
                        "score": 0.96,
                        "text": "14-II. question text",
                    }
                ],
                "layout_regions": [],
            },
            image_width=1000,
        )
        StructuredServiceProvider._apply_secondary_header(
            result,
            "14-II. question text",
            "14-III.",
            0.95,
            0.90,
        )
        self.assertTrue(result.transcript.startswith("14-III."))
        self.assertTrue(result.blocks[0].text.startswith("14-III."))
        self.assertIn(
            "QUESTION_HEADER_CORRECTED_BY_SECONDARY_OCR:14-II->14-III:0.950",
            result.warnings,
        )

    def test_header_verification_uses_a_tight_direct_token_crop(self) -> None:
        class FakeEngine:
            def __init__(self) -> None:
                self.kwargs = None
                self.image_shape = None

            def __call__(self, image, **kwargs):
                self.kwargs = kwargs
                self.image_shape = image.shape
                return SimpleNamespace(
                    txts=("14-III.",),
                    scores=(0.99,),
                )

        payload = {
            "handwriting_filtered_count": 0,
            "transcript": "14-II. question text",
            "items": [
                {
                    "label": "ocr_text",
                    "bbox": [20, 20, 500, 55],
                    "score": 0.96,
                    "text": "14-II. question text",
                }
            ],
            "layout_regions": [],
        }
        provider = StructuredServiceProvider(verify_roman_headers=True)
        engine = FakeEngine()
        provider._header_engine = engine
        result = provider._convert(payload, image_width=1000)
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (1000, 200), "white").save(image_path)
            provider._verify_roman_question_header(image_path, payload, result)
        self.assertFalse(engine.kwargs["use_det"])
        self.assertTrue(engine.kwargs["use_rec"])
        self.assertLess(engine.image_shape[1], 200)
        self.assertTrue(result.transcript.startswith("14-III."))

    def test_low_confidence_secondary_header_disagreement_is_not_applied(self) -> None:
        result = StructuredServiceProvider()._convert(
            {
                "handwriting_filtered_count": 0,
                "transcript": "14-II. question text",
                "items": [
                    {
                        "label": "ocr_text",
                        "bbox": [20, 20, 500, 55],
                        "score": 0.96,
                        "text": "14-II. question text",
                    }
                ],
                "layout_regions": [],
            },
            image_width=1000,
        )
        StructuredServiceProvider._apply_secondary_header(
            result,
            "14-II. question text",
            "14-III.",
            0.72,
            0.90,
        )
        self.assertTrue(result.transcript.startswith("14-II."))
        self.assertIn(
            "QUESTION_HEADER_OCR_DISAGREEMENT:14-II->14-III:0.720",
            result.warnings,
        )

    def test_separate_handwritten_solution_column_is_removed_and_flagged(self) -> None:
        payload = {
            "handwriting_filtered_count": 1,
            "items": [
                {"label": "ocr_text", "bbox": [80, 40, 700, 70], "score": 1, "text": "如图所示，求："},
                {"label": "inline_formula", "bbox": [220, 40, 290, 70], "score": 0.9, "latex": "v_B=2"},
                {"label": "ocr_text", "bbox": [80, 210, 420, 240], "score": 1, "text": "(1)求货物的加速度大小"},
                {"label": "ocr_text", "bbox": [80, 260, 430, 290], "score": 1, "text": "(2)求货物运动所用时间"},
                {"label": "display_formula", "bbox": [520, 100, 850, 145], "score": 0.6, "latex": "a=2m/s^2"},
                {"label": "display_formula", "bbox": [520, 155, 860, 200], "score": 0.6, "latex": "t=1.4s"},
                {"label": "ocr_text", "bbox": [510, 75, 600, 105], "score": 0.7, "text": "解：(1)"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertIn("v_B=2", result.transcript)
        self.assertNotIn("a=2m/s", result.transcript)
        self.assertNotIn("t=1.4", result.transcript)
        self.assertEqual(result.formula_count, 1)
        self.assertTrue(result.handwriting_suspected)
        self.assertIn("POSTFILTER_HANDWRITING_REGIONS:3", result.warnings)
        self.assertEqual(len(result.ignored_blocks), 3)
        self.assertTrue(all(block.reason == "handwritten_solution_column" for block in result.ignored_blocks))

    def test_multiple_formula_blocks_floating_right_of_options_are_flagged(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [80, 100, 400, 130], "score": 1, "text": "A. 选项甲"},
                {"label": "ocr_text", "bbox": [80, 150, 410, 180], "score": 1, "text": "B. 选项乙"},
                {"label": "ocr_text", "bbox": [80, 200, 420, 230], "score": 1, "text": "C. 选项丙"},
                {"label": "ocr_text", "bbox": [80, 250, 430, 280], "score": 1, "text": "D. 选项丁"},
                {"label": "display_formula", "bbox": [650, 135, 900, 180], "score": 0.7, "latex": "v=v_0+at"},
                {"label": "display_formula", "bbox": [700, 205, 930, 250], "score": 0.7, "latex": "v^2=2gh"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertTrue(result.handwriting_suspected)
        self.assertIn("OPTION_AREA_HANDWRITING_SUSPECTED:2", result.warnings)
        self.assertNotIn("v=v_0+at", result.transcript)
        self.assertEqual(
            {block.reason for block in result.ignored_blocks},
            {"option_area_handwriting"},
        )

    def test_low_confidence_formula_below_complete_options_is_working(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [80, 100, 400, 130], "score": 1, "text": "A. 选项甲"},
                {"label": "ocr_text", "bbox": [80, 150, 410, 180], "score": 1, "text": "B. 选项乙"},
                {"label": "ocr_text", "bbox": [80, 200, 420, 230], "score": 1, "text": "C. 选项丙"},
                {"label": "ocr_text", "bbox": [80, 250, 430, 280], "score": 1, "text": "D. 选项丁"},
                {"label": "display_formula", "bbox": [180, 310, 520, 360], "score": 0.68, "latex": "x=2"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertTrue(result.handwriting_suspected)
        self.assertIn("OPTION_AREA_HANDWRITING_SUSPECTED:1", result.warnings)
        self.assertNotIn("x=2", result.transcript)

    def test_small_printed_formula_fragments_after_options_are_not_handwriting(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [80, 100, 700, 130], "score": 1, "text": "A. 加速度大于"},
                {"label": "ocr_text", "bbox": [80, 150, 700, 180], "score": 1, "text": "B. 最终速度为"},
                {"label": "ocr_text", "bbox": [80, 200, 700, 230], "score": 1, "text": "C. 给一个初速度"},
                {"label": "ocr_text", "bbox": [80, 250, 700, 280], "score": 1, "text": "D. 继续运动"},
                {"label": "inline_formula", "bbox": [760, 102, 790, 130], "score": 0.9, "latex": "L_2"},
                {"label": "inline_formula", "bbox": [720, 152, 850, 190], "score": 0.9, "latex": "\\frac{BLCE}{m_1+m_2}"},
                {"label": "inline_formula", "bbox": [760, 202, 795, 230], "score": 0.9, "latex": "v_0"},
                {"label": "inline_formula", "bbox": [760, 252, 795, 280], "score": 0.9, "latex": "L_2"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertFalse(result.handwriting_suspected)
        self.assertFalse(
            any(warning.startswith("OPTION_AREA_HANDWRITING") for warning in result.warnings)
        )
        preserved_formulas = {block.text for block in result.blocks if block.kind == "formula"}
        self.assertIn("L_2", preserved_formulas)
        self.assertIn("v_0", preserved_formulas)

    def test_spurious_fraction_with_identical_inner_groups_is_collapsed(self) -> None:
        duplicated = (
            r"\frac { B L C E } { \frac { m _ { 1 } + m _ { 2 } } "
            r"{ m _ { 1 } + m _ { 2 } } }"
        )
        expected = r"\frac{B L C E}{m _ { 1 } + m _ { 2 }}"
        self.assertEqual(
            StructuredServiceProvider._collapse_duplicated_fraction_denominator(duplicated),
            expected,
        )
        payload = {
            "handwriting_filtered_count": 0,
            "transcript": duplicated,
            "items": [
                {
                    "label": "inline_formula",
                    "bbox": [20, 20, 300, 80],
                    "score": 0.95,
                    "latex": duplicated,
                }
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertIn(expected, result.transcript)
        self.assertIn("DUPLICATED_FRACTION_DENOMINATOR_COLLAPSED:1", result.warnings)

    def test_horizontal_formula_options_are_not_mistaken_for_handwriting(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [100, 100, 220, 135], "score": 1, "text": "A. (-2,0)"},
                {"label": "ocr_text", "bbox": [400, 100, 440, 135], "score": 1, "text": "B."},
                {"label": "inline_formula", "bbox": [450, 100, 570, 135], "score": 0.9, "latex": "(-1,+\\infty)"},
                {"label": "ocr_text", "bbox": [650, 100, 690, 135], "score": 1, "text": "C."},
                {"label": "inline_formula", "bbox": [700, 100, 780, 135], "score": 0.9, "latex": "(-1,0)"},
                {"label": "ocr_text", "bbox": [820, 100, 860, 135], "score": 1, "text": "D."},
                {"label": "inline_formula", "bbox": [870, 100, 980, 135], "score": 0.9, "latex": "(-\\infty,-2)"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertFalse(result.handwriting_suspected)
        self.assertFalse(any(warning.startswith("OPTION_AREA_HANDWRITING") for warning in result.warnings))

    def test_following_printed_question_is_trimmed_not_called_handwriting(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [30, 200, 500, 235], "score": 1, "text": "B. 选项乙"},
                {"label": "ocr_text", "bbox": [30, 270, 600, 305], "score": 1, "text": "C. 选项丙"},
                {"label": "ocr_text", "bbox": [30, 340, 700, 375], "score": 1, "text": "D. 选项丁"},
                {"label": "ocr_text", "bbox": [20, 430, 500, 465], "score": 1, "text": "10. 已知三个向量满足"},
                {"label": "inline_formula", "bbox": [540, 425, 760, 470], "score": 0.9, "latex": "a+b=c"},
                {"label": "display_formula", "bbox": [700, 500, 950, 545], "score": 0.9, "latex": "a^2+b^2=c^2"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertNotIn("10.", result.transcript)
        self.assertNotIn("a+b=c", result.transcript)
        self.assertFalse(result.handwriting_suspected)
        self.assertIn("ADJACENT_QUESTION_TRIMMED:3", result.warnings)
        self.assertEqual(len(result.ignored_blocks), 3)
        self.assertTrue(all(block.reason == "adjacent_question" for block in result.ignored_blocks))

    def test_dominant_question_window_removes_previous_and_next_fragments(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [30, 10, 500, 40], "score": 1, "text": "C. 上一题选项"},
                {"label": "ocr_text", "bbox": [30, 45, 500, 75], "score": 1, "text": "D. 上一题选项"},
                {"label": "ocr_text", "bbox": [20, 100, 900, 140], "score": 1, "text": "7. 如图所示，下列判断正确的是"},
                {"label": "ocr_text", "bbox": [40, 160, 600, 190], "score": 1, "text": "A. 选项甲"},
                {"label": "ocr_text", "bbox": [40, 200, 600, 230], "score": 1, "text": "B. 选项乙"},
                {"label": "ocr_text", "bbox": [40, 240, 600, 270], "score": 1, "text": "C. 选项丙"},
                {"label": "ocr_text", "bbox": [40, 280, 600, 310], "score": 1, "text": "D. 选项丁"},
                {"label": "ocr_text", "bbox": [20, 400, 900, 440], "score": 1, "text": "8. 下一题只有一行"},
            ],
            "layout_regions": [
                {"bbox": [500, 120, 760, 300]},
                {"bbox": [500, 430, 760, 650]},
            ],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertTrue(result.transcript.startswith("7."))
        self.assertNotIn("上一题", result.transcript)
        self.assertNotIn("8.", result.transcript)
        self.assertEqual(len(result.visual_bboxes), 1)
        self.assertIn("DOMINANT_QUESTION_WINDOW_SELECTED", result.warnings)
        self.assertEqual(len(result.ignored_blocks), 3)

    def test_target_anchor_excludes_previous_option_and_preserves_stem_formulas(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                # Its centre is 241.5: the former -8 px tolerance kept this row.
                {"label": "ocr_text", "bbox": [30, 226, 700, 257], "score": 1, "text": "D. 上一题选项"},
                {"label": "ocr_text", "bbox": [20, 249, 900, 285], "score": 1, "text": "4. 已知下列物理常量，选择正确答案"},
                {"label": "inline_formula", "bbox": [90, 290, 330, 330], "score": 0.9, "latex": "m_n=939.57\\,MeV/c^2"},
                {"label": "inline_formula", "bbox": [400, 292, 760, 332], "score": 0.9, "latex": "h=6.63\\times10^{-34}J\\cdot s"},
                {"label": "ocr_text", "bbox": [60, 380, 300, 412], "score": 1, "text": "A. 选项甲"},
                {"label": "ocr_text", "bbox": [60, 430, 300, 462], "score": 1, "text": "B. 选项乙"},
                {"label": "ocr_text", "bbox": [60, 480, 300, 512], "score": 1, "text": "C. 选项丙"},
                {"label": "ocr_text", "bbox": [60, 530, 300, 562], "score": 1, "text": "D. 选项丁"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertNotIn("上一题", result.transcript)
        self.assertIn("m_n=939.57", result.transcript)
        self.assertIn("h=6.63", result.transcript)
        self.assertFalse(result.handwriting_suspected)
        self.assertFalse(
            any(warning.startswith("OPTION_AREA_HANDWRITING") for warning in result.warnings)
        )

    def test_single_numbered_anchor_trims_packed_previous_option_row(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [40, 20, 900, 55], "score": 1, "text": "A. 1 B. 2 C. 3 D. 4"},
                {"label": "ocr_text", "bbox": [20, 100, 900, 135], "score": 1, "text": "2. 复数的虚部为"},
                {"label": "ocr_text", "bbox": [60, 180, 240, 212], "score": 1, "text": "A. 1/3"},
                {"label": "ocr_text", "bbox": [300, 180, 480, 212], "score": 1, "text": "B. -1/3"},
                {"label": "ocr_text", "bbox": [540, 180, 700, 212], "score": 1, "text": "C. 2/3"},
                {"label": "ocr_text", "bbox": [760, 180, 930, 212], "score": 1, "text": "D. -2/3"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertTrue(result.transcript.startswith("2."))
        self.assertNotIn("A. 1 B. 2", result.transcript)

    def test_single_numbered_anchor_trims_one_previous_option_row(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [40, 10, 900, 45], "score": 1, "text": "D. 上一题选项"},
                {"label": "ocr_text", "bbox": [20, 70, 900, 105], "score": 1, "text": "13.2026年，如图所示，选择正确答案"},
                {"label": "ocr_text", "bbox": [60, 140, 300, 172], "score": 1, "text": "A. 选项甲"},
                {"label": "ocr_text", "bbox": [60, 180, 300, 212], "score": 1, "text": "B. 选项乙"},
                {"label": "ocr_text", "bbox": [60, 220, 300, 252], "score": 1, "text": "C. 选项丙"},
                {"label": "ocr_text", "bbox": [60, 260, 300, 292], "score": 1, "text": "D. 选项丁"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertTrue(result.transcript.startswith("13."))
        self.assertNotIn("上一题选项", result.transcript)

    def test_caption_visual_suppresses_nearby_standalone_numeric_working(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 700, 55], "score": 1, "text": "16. 下表中填入适当的数"},
                {"label": "ocr_text", "bbox": [330, 120, 355, 150], "score": 0.99, "text": "9"},
                {"label": "ocr_text", "bbox": [300, 155, 350, 190], "score": 0.55, "text": "237"},
                {"label": "ocr_text", "bbox": [500, 310, 620, 345], "score": 1, "text": "第16题图"},
            ],
            "layout_regions": [{"bbox": [370, 90, 670, 350]}],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertNotIn("\n9", result.transcript)
        self.assertNotIn("237", result.transcript)
        self.assertEqual(len(result.visual_bboxes), 1)

    def test_short_low_confidence_option_label_artifact_is_normalized(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 900, 55], "score": 1, "text": "5. 选择正确关系"},
                {"label": "ocr_text", "bbox": [80, 100, 120, 132], "score": 0.61, "text": "AV."},
                {"label": "inline_formula", "bbox": [125, 95, 300, 140], "score": 0.95, "latex": "r_1/r_2"},
                {"label": "ocr_text", "bbox": [350, 100, 390, 132], "score": 1, "text": "B."},
                {"label": "ocr_text", "bbox": [560, 100, 600, 132], "score": 1, "text": "C."},
                {"label": "ocr_text", "bbox": [760, 100, 800, 132], "score": 1, "text": "D."},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertIn("A.", result.transcript)
        self.assertNotIn("AV.", result.transcript)
        self.assertIn("OPTION_LABEL_ARTIFACT_NORMALIZED:1", result.warnings)

    def test_vertically_stacked_options_do_not_merge_in_reverse_x_order(self) -> None:
        transcript = StructuredServiceProvider._transcript_from_items(
            [
                {"label": "ocr_text", "bbox": [71, 563, 570, 609], "text": "C. 选项丙"},
                {"label": "ocr_text", "bbox": [70, 593, 891, 657], "text": "D. 选项丁"},
            ]
        )
        self.assertEqual(transcript.splitlines(), ["C. 选项丙", "D. 选项丁"])

    def test_bare_option_labels_form_coherent_formula_option_group(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "transcript": (
                "7. 选择正确的表达式\n"
                "a^2+b^2=c^2\nA. x=a+b B x=a-b C. x=ab D x=a/b"
            ),
            "items": [
                {"label": "ocr_text", "bbox": [20, 100, 900, 135], "score": 1, "text": "7. 选择正确的表达式"},
                {"label": "display_formula", "bbox": [120, 180, 500, 220], "score": 0.9, "latex": "a^2+b^2=c^2"},
                {"label": "ocr_text", "bbox": [60, 300, 90, 332], "score": 1, "text": "A."},
                {"label": "inline_formula", "bbox": [100, 300, 340, 336], "score": 0.9, "latex": "x=a+b"},
                {"label": "ocr_text", "bbox": [520, 300, 550, 332], "score": 1, "text": "B"},
                {"label": "inline_formula", "bbox": [565, 300, 805, 336], "score": 0.9, "latex": "x=a-b"},
                {"label": "ocr_text", "bbox": [60, 370, 90, 402], "score": 1, "text": "C."},
                {"label": "inline_formula", "bbox": [100, 370, 340, 406], "score": 0.9, "latex": "x=ab"},
                {"label": "ocr_text", "bbox": [520, 370, 550, 402], "score": 1, "text": "D"},
                {"label": "inline_formula", "bbox": [565, 370, 805, 406], "score": 0.9, "latex": "x=a/b"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertIn("a^2+b^2=c^2", result.transcript)
        self.assertIn("x=a-b", result.transcript)
        self.assertIn("x=a/b", result.transcript)
        self.assertFalse(result.handwriting_suspected)

    def test_ambiguous_complete_questions_are_not_aggressively_windowed(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "transcript": "1. 下列判断正确的是\nA. 甲\nB. 乙\n2. 下列判断正确的是\nA. 丙\nB. 丁",
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 700, 55], "score": 1, "text": "1. 下列判断正确的是"},
                {"label": "ocr_text", "bbox": [40, 70, 500, 100], "score": 1, "text": "A. 甲"},
                {"label": "ocr_text", "bbox": [40, 110, 500, 140], "score": 1, "text": "B. 乙"},
                {"label": "ocr_text", "bbox": [20, 220, 700, 255], "score": 1, "text": "2. 下列判断正确的是"},
                {"label": "ocr_text", "bbox": [40, 270, 500, 300], "score": 1, "text": "A. 丙"},
                {"label": "ocr_text", "bbox": [40, 310, 500, 340], "score": 1, "text": "B. 丁"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertIn("1.", result.transcript)
        self.assertIn("2.", result.transcript)
        self.assertNotIn("DOMINANT_QUESTION_WINDOW_SELECTED", result.warnings)

    def test_caption_guidance_keeps_target_composite_and_discards_adjacent_visuals(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 900, 55], "score": 1, "text": "7. 如图所示，下列判断正确的是"},
                {"label": "ocr_text", "bbox": [60, 70, 500, 100], "score": 1, "text": "A. 选项甲"},
                {"label": "ocr_text", "bbox": [60, 110, 500, 140], "score": 1, "text": "B. 选项乙"},
                {"label": "ocr_text", "bbox": [60, 150, 500, 180], "score": 1, "text": "C. 选项丙"},
                {"label": "ocr_text", "bbox": [60, 190, 500, 220], "score": 1, "text": "D. 选项丁"},
                {"label": "ocr_text", "bbox": [120, 360, 880, 390], "score": 1, "text": "甲第7题图第8题图"},
                {"label": "ocr_text", "bbox": [150, 330, 180, 355], "score": 1, "text": "甲"},
                {"label": "ocr_text", "bbox": [670, 330, 700, 355], "score": 1, "text": "乙"},
                {"label": "ocr_text", "bbox": [20, 440, 900, 475], "score": 1, "text": "8. 下一题"},
            ],
            "layout_regions": [
                {"bbox": [80, 250, 230, 340]},
                {"bbox": [250, 240, 390, 340]},
                {"bbox": [610, 245, 750, 340]},
            ],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertEqual(len(result.visual_bboxes), 1)
        grouped = result.visual_bboxes[0]
        self.assertLessEqual(grouped.x, 80)
        self.assertGreaterEqual(grouped.right, 500)
        self.assertLess(grouped.right, 610)
        self.assertLess(grouped.bottom, 440)
        self.assertNotIn("第8题图", result.transcript)
        self.assertNotIn("甲第7题图", result.transcript)
        self.assertTrue(
            any(
                block.text == "\u4e59" and block.reason == "embedded_visual_text"
                for block in result.ignored_blocks
            )
        )
        self.assertIn("ADJACENT_VISUAL_TRIMMED:1", result.warnings)
        self.assertIn("CAPTION_GUIDED_VISUAL_GROUP:2->1", result.warnings)
        self.assertTrue(any(block.reason == "embedded_visual_text" for block in result.ignored_blocks))

    def test_only_mismatched_caption_discards_adjacent_single_visual(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 120, 900, 155], "score": 1, "text": "7. 已知条件，选择正确答案"},
                {"label": "ocr_text", "bbox": [760, 300, 900, 330], "score": 1, "text": "第6题图"},
                {"label": "ocr_text", "bbox": [60, 380, 500, 410], "score": 1, "text": "A. 选项甲"},
                {"label": "ocr_text", "bbox": [60, 420, 500, 450], "score": 1, "text": "B. 选项乙"},
                {"label": "ocr_text", "bbox": [60, 460, 500, 490], "score": 1, "text": "C. 选项丙"},
                {"label": "ocr_text", "bbox": [60, 500, 500, 530], "score": 1, "text": "D. 选项丁"},
            ],
            "layout_regions": [{"bbox": [700, 160, 930, 295]}],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertEqual(result.visual_bboxes, [])
        self.assertNotIn("第6题图", result.transcript)
        self.assertIn("ADJACENT_VISUAL_TRIMMED:1", result.warnings)

    def test_oversized_visual_is_clipped_before_following_subquestion(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 900, 55], "score": 1, "text": "14. 实验题"},
                {"label": "ocr_text", "bbox": [450, 300, 600, 330], "score": 1, "text": "第14题图"},
                {"label": "ocr_text", "bbox": [120, 342, 1050, 372], "score": 1, "text": "（1）如图是某次实验时打点计时器打出的一条纸带"},
                {"label": "ocr_text", "bbox": [120, 390, 900, 420], "score": 1, "text": "（2）求小车的加速度大小"},
            ],
            "layout_regions": [{"bbox": [100, 80, 1100, 540]}],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1200)
        self.assertEqual(len(result.visual_bboxes), 1)
        self.assertLessEqual(result.visual_bboxes[0].bottom, 344)
        self.assertIn("（1）如图是某次实验", result.transcript)
        self.assertIn("（2）求小车", result.transcript)

    def test_auxiliary_labels_cannot_chain_across_visual_rows(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 900, 55], "score": 1, "text": "14. 实验题"},
                {"label": "ocr_text", "bbox": [360, 270, 390, 300], "score": 1, "text": "甲"},
                {"label": "ocr_text", "bbox": [850, 270, 880, 300], "score": 1, "text": "乙"},
                {"label": "ocr_text", "bbox": [547, 310, 668, 337], "score": 1, "text": "第14题图1"},
                {"label": "ocr_text", "bbox": [166, 342, 1097, 368], "score": 1, "text": "（1）如图是某次实验时打点计时器打出的一条纸带"},
                {"label": "ocr_text", "bbox": [164, 429, 984, 458], "score": 1, "text": "（2）当甲乙两种方案示数相同时求加速度关系"},
                {"label": "ocr_text", "bbox": [342, 508, 368, 533], "score": 1, "text": "B"},
                {"label": "ocr_text", "bbox": [434, 507, 461, 532], "score": 1, "text": "C"},
                {"label": "ocr_text", "bbox": [488, 582, 609, 606], "score": 1, "text": "第14题图2"},
            ],
            "layout_regions": [
                {"bbox": [154, 90, 1157, 539]},
                {"bbox": [226, 495, 890, 612]},
            ],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1200)
        self.assertEqual(len(result.visual_bboxes), 2)
        self.assertLess(result.visual_bboxes[0].bottom, 380)
        self.assertIn("（1）如图是某次实验", result.transcript)
        self.assertIn("（2）当甲乙", result.transcript)

    def test_single_caption_merges_multi_part_figure_without_reflowing_internal_labels(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 900, 55], "score": 1, "text": "6. 如图所示，下列说法正确的是"},
                {"label": "ocr_text", "bbox": [400, 360, 520, 390], "score": 1, "text": "第6题图"},
                {"label": "ocr_text", "bbox": [120, 290, 250, 320], "score": 1, "text": "时间（秒）"},
            ],
            "layout_regions": [
                {"bbox": [80, 120, 250, 260]},
                {"bbox": [280, 120, 450, 260]},
                {"bbox": [80, 270, 250, 350]},
                {"bbox": [280, 270, 450, 350]},
            ],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertEqual(len(result.visual_bboxes), 1)
        self.assertNotIn("时间（秒）", result.transcript)
        self.assertNotIn("第6题图", result.transcript)
        self.assertIn("CAPTION_GUIDED_VISUAL_GROUP:4->1", result.warnings)

    def test_caption_grouping_does_not_bridge_and_suppress_question_text(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 900, 55], "score": 1, "text": "14. 如图所示，回答下列问题"},
                {"label": "ocr_text", "bbox": [80, 230, 760, 270], "score": 1, "text": "闭合开关并移动滑片，观察电流表的示数变化"},
                {"label": "ocr_text", "bbox": [420, 600, 560, 630], "score": 1, "text": "第14题图"},
                {"label": "ocr_text", "bbox": [180, 540, 210, 570], "score": 1, "text": "A"},
                {"label": "ocr_text", "bbox": [480, 540, 510, 570], "score": 1, "text": "B"},
                {"label": "ocr_text", "bbox": [780, 540, 810, 570], "score": 1, "text": "C"},
                {"label": "inline_formula", "bbox": [520, 540, 610, 570], "score": 1, "latex": "0 \\sim 3 mA"},
                {"label": "ocr_text", "bbox": [620, 540, 690, 570], "score": 1, "text": "\u91cf\u7a0b\uff1a"},
            ],
            "layout_regions": [
                {"bbox": [650, 70, 860, 210]},
                {"bbox": [100, 330, 300, 520]},
                {"bbox": [400, 330, 600, 520]},
                {"bbox": [700, 330, 900, 520]},
            ],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertEqual(len(result.visual_bboxes), 2)
        self.assertIn("闭合开关并移动滑片", result.transcript)
        self.assertNotIn("第14题图", result.transcript)
        self.assertNotIn("\nA\n", f"\n{result.transcript}\n")
        self.assertNotIn("0 \\sim 3 mA", result.transcript)
        self.assertNotIn("\u91cf\u7a0b\uff1a", result.transcript)
        self.assertIn("CAPTION_GUIDED_VISUAL_GROUP:4->2", result.warnings)

    def test_parenthesized_adjacent_figure_caption_is_not_emitted_as_question_text(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 900, 55], "score": 1, "text": "3. 如图所示，选择正确答案"},
                {"label": "ocr_text", "bbox": [60, 70, 500, 100], "score": 1, "text": "A. 甲 B. 乙 C. 丙 D. 丁"},
                {"label": "ocr_text", "bbox": [180, 330, 320, 365], "score": 1, "text": "（第3题图）"},
                {"label": "ocr_text", "bbox": [710, 330, 850, 365], "score": 1, "text": "（第4题图）"},
            ],
            "layout_regions": [
                {"bbox": [80, 140, 430, 320]},
                {"bbox": [600, 140, 920, 320]},
            ],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertEqual(len(result.visual_bboxes), 1)
        self.assertLess(result.visual_bboxes[0].right, 600)
        self.assertNotIn("第3题图", result.transcript)
        self.assertNotIn("第4题图", result.transcript)

    def test_same_row_visuals_are_split_when_union_would_duplicate_question_text(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 900, 55], "score": 1, "text": "8. 图甲和图乙给出装置参数"},
                {"label": "ocr_text", "bbox": [207, 213, 510, 249], "score": 1, "text": "信息，下列说法正确的是"},
                {"label": "ocr_text", "bbox": [620, 500, 740, 535], "score": 1, "text": "第8题图"},
                {"label": "ocr_text", "bbox": [60, 550, 500, 580], "score": 1, "text": "A. 甲 B. 乙 C. 丙 D. 丁"},
            ],
            "layout_regions": [
                {"bbox": [278, 262, 685, 464]},
                {"bbox": [755, 221, 1060, 465]},
            ],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1280)
        self.assertEqual(len(result.visual_bboxes), 3)
        self.assertIn("信息，下列说法正确的是", result.transcript)
        phrase_x, phrase_y = 358, 231
        self.assertFalse(
            any(
                box.x <= phrase_x <= box.right and box.y <= phrase_y <= box.bottom
                for box in result.visual_bboxes
            )
        )

    def test_unmatched_open_parenthesis_before_options_is_removed(self) -> None:
        cleaned = StructuredServiceProvider._strip_page_noise(
            "3. 如图所示\n（其中数值精确到0.1m）（\nA. 1.4m B. 2.8m C. 4.2m D. 5.7m"
        )
        self.assertIn("（其中数值精确到0.1m）\nA.", cleaned)
        self.assertNotIn("）（\nA.", cleaned)

    def test_numbered_panel_caption_is_not_a_text_barrier(self) -> None:
        self.assertTrue(StructuredServiceProvider._standalone_figure_caption("第14-I题图1"))
        self.assertTrue(StructuredServiceProvider._standalone_figure_caption("（第14-I题图2）"))

    def test_low_confidence_ascii_token_at_far_page_edge_is_filtered(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 900, 55], "score": 1, "text": "8. 下列说法正确的是"},
                {"label": "ocr_text", "bbox": [40, 100, 600, 135], "score": 1, "text": "B. 小风扇的内阻约为242Ω"},
                {"label": "ocr_text", "bbox": [1225, 100, 1271, 142], "score": 0.79, "text": "iuns"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1280)
        self.assertNotIn("iuns", result.transcript)
        self.assertIn("PAGE_EDGE_OCR_NOISE_FILTERED:1", result.warnings)
        self.assertTrue(
            any(
                block.text == "iuns" and block.reason == "page_edge_noise"
                for block in result.ignored_blocks
            )
        )

    def test_long_low_confidence_printed_text_is_fail_closed_signal(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "items": [
                {
                    "label": "ocr_text",
                    "bbox": [20, 20, 900, 70],
                    "score": 0.64,
                    "text": "7. 这是一整行已经无法可靠识别的印刷题干文字",
                },
                {"label": "ocr_text", "bbox": [40, 100, 600, 135], "score": 0.99, "text": "A. 选项甲"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertTrue(
            any(warning.startswith("LOW_CONFIDENCE_LONG_TEXT_REGION:1:") for warning in result.warnings)
        )

    def test_known_page_watermark_is_removed_without_dropping_the_line(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "transcript": "D. 儿子身高残差为3cm浙考神墙750",
            "items": [
                {
                    "label": "ocr_text",
                    "bbox": [20, 20, 500, 60],
                    "score": 0.99,
                    "text": "D. 儿子身高残差为3cm浙考神墙750",
                }
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertEqual(result.transcript, "D. 儿子身高残差为3cm")
        self.assertEqual(result.blocks[0].text, "D. 儿子身高残差为3cm")
        self.assertIn("PAGE_NOISE_FILTERED", result.warnings)

    def test_partial_shenqiang_watermark_fragment_is_removed(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "transcript": "([x]表示不超过实数x的神墙最大整数)",
            "items": [
                {
                    "label": "ocr_text",
                    "bbox": [20, 20, 500, 60],
                    "score": 0.99,
                    "text": "([x]表示不超过实数x的神墙最大整数)",
                }
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertEqual(result.transcript, "([x]表示不超过实数x的最大整数)")
        self.assertEqual(result.blocks[0].text, "([x]表示不超过实数x的最大整数)")
        self.assertIn("PAGE_NOISE_FILTERED", result.warnings)

    def test_corrupted_long_numeric_watermark_line_is_removed(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "transcript": "9. 选择正确答案\nA. 甲\nB. 乙\nC. 丙\nD. 丁\n纯位八人口法750002754808740",
            "items": [
                {"label": "ocr_text", "bbox": [20, 20, 500, 60], "score": 0.99, "text": "9. 选择正确答案"},
                {"label": "ocr_text", "bbox": [20, 80, 500, 120], "score": 0.99, "text": "A. 甲 B. 乙 C. 丙 D. 丁"},
                {"label": "ocr_text", "bbox": [20, 300, 500, 330], "score": 0.70, "text": "纯位八人口法750002754808740"},
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertNotIn("750002754808740", result.transcript)
        self.assertIn("PAGE_NOISE_FILTERED", result.warnings)

    def test_duplicate_question_number_punctuation_is_normalized(self) -> None:
        payload = {
            "handwriting_filtered_count": 0,
            "transcript": "5.. 赫兹通过实验观察到电火花",
            "items": [
                {
                    "label": "ocr_text",
                    "bbox": [20, 20, 500, 60],
                    "score": 0.99,
                    "text": "5.. 赫兹通过实验观察到电火花",
                }
            ],
            "layout_regions": [],
        }
        result = StructuredServiceProvider()._convert(payload, image_width=1000)
        self.assertEqual(result.transcript, "5. 赫兹通过实验观察到电火花")
        self.assertEqual(result.blocks[0].text, "5. 赫兹通过实验观察到电火花")
        self.assertIn("PAGE_NOISE_FILTERED", result.warnings)


if __name__ == "__main__":
    unittest.main()
