#!/usr/bin/env python3
"""Regression tests for the mathematics camera search-machine contract."""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from PIL import Image

from device_result_renderer import (
    DEVICE_IMAGE_KINDS,
    DEVICE_IMAGE_NAMES,
    resolve_device_fonts,
    validate_device_image_set,
)
import math_guidance_pilot as pilot
from math_guidance_renderer import (
    BODY_LINE,
    BODY_SIZE,
    BOTTOM,
    DISPLAY_WIDTH_CM,
    FORMULA_LINE,
    FORMULA_SIZE,
    MARGIN_X,
    render_four_images,
    tier_blocks,
)
import search_machine_app as app_module
from search_machine_app import public_job
from search_machine_pipeline import (
    SearchMachinePipeline,
    apply_dynamic_local_repairs,
    build_request_payload,
    infer_dynamic_requested_outputs,
    merge_guidance_only_repair,
    normalize_cross_tier_method_names,
    normalize_formula_source_figures,
    parse_model_output,
    png_data_url,
    repair_candidate_strictly_improves,
    validate_generic_answer_blind,
    validate_figure_reference_completeness,
    validate_normalized_output,
)


SOURCE_DIR = Path(__file__).resolve().parent


def sample_guidance_output() -> dict:
    """Return a strict-valid, answer-blind mathematics fixture."""

    def part(steps: list[str], *, pitfall: str | None = None) -> dict:
        value = {
            "part": "整题",
            "method_id": "M1",
            "method_name": "导数符号法",
            "steps": steps,
            "alternative_methods": [],
        }
        if pitfall is not None:
            value["pitfall"] = pitfall
        return value

    return {
        "subject": "mathematics",
        "problem_scope": "primary_to_high_school",
        "recognition_status": "clear",
        "uncertain_items": [],
        "question_blocks": [
            {
                "type": "text",
                "text": "已知函数 $f(x)$ 及其导函数 $f'(x)$，讨论函数的单调性。",
                "figure_id": "",
            }
        ],
        "source_figures": [],
        "visual_evidence": [],
        "knowledge_points": ["函数的单调性", "导数的符号"],
        "base_formulas": [
            {
                "formula": "f'(x)>0",
                "condition": "导数为正的区间上函数单调递增",
            }
        ],
        "guidance": {
            "advanced": {
                "title": "学霸版",
                "parts": [
                    part(
                        [
                            "确定定义域并求导，写出 $f'(x)$。",
                            "绘制导数符号图，并据此描述各区间的单调性。",
                        ]
                    )
                ],
            },
            "standard": {
                "title": "普通版",
                "parts": [
                    part(
                        [
                            "先写出函数的定义域。",
                            "求出 $f'(x)$，在临界点表格中标出分界位置。",
                            "逐个区间判断 $f'(x)$ 的符号，再对应写出单调性。",
                        ],
                        pitfall="定义域端点和导数无意义的点也要纳入区间边界。",
                    )
                ],
            },
            "foundation": {
                "title": "基础版",
                "parts": [
                    part(
                        [
                            "检查题目给出的函数表达式，并确定定义域。",
                            "使用导数法则求 $f'(x)$。",
                            "把临界点按从小到大的顺序放入数轴分段。",
                            "逐段判断 $f'(x)$ 的正负，并按导数判定法描述单调性。",
                        ]
                    )
                ],
                "pattern_summary": "先定定义域，再求导和分段，最后由导数符号写出单调区间。",
            },
        },
    }


class MathematicsContentContractTests(unittest.TestCase):
    def test_prompt_readme_and_page_share_the_grade_boundary(self) -> None:
        prompt = (SOURCE_DIR / "数学题解析提示词.md").read_text(encoding="utf-8")
        readme = (SOURCE_DIR.parent / "README_搜题机.md").read_text(
            encoding="utf-8"
        )
        page = (SOURCE_DIR / "search_machine_static" / "index.html").read_text(
            encoding="utf-8"
        )
        for text in (prompt, readme, page):
            self.assertIn("小学", text)
            self.assertIn("初中", text)
            self.assertIn("高中", text)
            self.assertIn("选择题", text)
            self.assertIn("填空题", text)
        self.assertIn('"problem_scope": "primary_to_high_school"', prompt)
        self.assertIn("不得用初中或高中技巧替代小学教材内的方法", prompt)

    def test_scope_contract_covers_primary_middle_and_high_school(self) -> None:
        schema = json.loads(
            (SOURCE_DIR / "math_guidance_schema.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            schema["properties"]["problem_scope"]["const"],
            "primary_to_high_school",
        )

        for question, knowledge_points, formula, condition in (
            (
                "书架原有48本书，借出19本后还剩多少本？",
                ["整数减法", "数量关系"],
                "a-b",
                "求原有量减少后的剩余量时",
            ),
            (
                "某数的2倍加3等于11，求这个数。",
                ["一元一次方程", "等式的性质"],
                "2x+3=11",
                "把文字数量关系转化为方程时",
            ),
            (
                "已知函数 $f(x)$，讨论其单调性。",
                ["函数的单调性", "导数的符号"],
                "f'(x)>0",
                "导数为正的区间上函数单调递增",
            ),
        ):
            output = sample_guidance_output()
            output["question_blocks"][0]["text"] = question
            output["knowledge_points"] = knowledge_points
            output["base_formulas"] = [{"formula": formula, "condition": condition}]
            self.assertEqual(validate_normalized_output(output), [], question)

    def test_legacy_high_school_only_scope_is_rejected(self) -> None:
        output = sample_guidance_output()
        output["problem_scope"] = "high_school_open_response"
        errors = pilot.validate_output(
            output,
            requested_outputs=None,
            device_policy=False,
        )
        self.assertTrue(any("primary_to_high_school" in item for item in errors))

    def test_fixture_is_strict_valid_mathematics_guidance(self) -> None:
        output = sample_guidance_output()
        self.assertEqual(output["subject"], "mathematics")
        self.assertEqual(
            pilot.validate_output(output, requested_outputs=None, device_policy=False),
            [],
        )

    def test_midpoint_half_segment_denominator_requires_strong_evidence(self) -> None:
        output = sample_guidance_output()
        output["base_formulas"] = [{
            "formula": r"DE=AE=\frac{1}{2 AD}",
            "condition": "线段长度换算",
        }]

        normalized = pilot.normalize_output_for_render(output)
        errors = pilot.validate_output(
            normalized,
            requested_outputs=None,
            device_policy=False,
        )

        self.assertEqual(
            normalized["base_formulas"][0]["formula"],
            r"DE=AE=\frac{1}{2 AD}",
        )
        self.assertTrue(any(
            "中点半长公式不得把整段名写进分母" in item
            for item in errors
        ))

    def test_midpoint_half_segment_denominator_repairs_only_proven_identity(self) -> None:
        output = sample_guidance_output()
        output["base_formulas"] = [{
            "formula": r"DE=AE=\frac{1}{2 AD}",
            "condition": "E为AD中点",
        }]

        normalized = pilot.normalize_output_for_render(output)
        errors = pilot.validate_output(
            normalized,
            requested_outputs=None,
            device_policy=False,
        )

        self.assertEqual(
            normalized["base_formulas"][0]["formula"],
            r"DE=AE=\frac{1}{2} AD",
        )
        self.assertFalse(any(
            "中点半长公式不得把整段名写进分母" in item
            for item in errors
        ))

    def test_schema_and_prompt_forbid_punctuation_only_steps(self) -> None:
        schema = json.loads(
            (SOURCE_DIR / "math_guidance_schema.json").read_text(encoding="utf-8")
        )
        for definition_name in (
            "advancedAlternative",
            "standardAlternative",
            "foundationAlternative",
            "advancedPart",
            "standardPart",
            "foundationPart",
        ):
            item_schema = schema["$defs"][definition_name]["properties"][
                "steps"
            ]["items"]
            self.assertGreaterEqual(item_schema["minLength"], 4)
            self.assertNotIn("pattern", item_schema)

        prompt = (SOURCE_DIR / "数学题解析提示词.md").read_text(encoding="utf-8")
        self.assertIn("不得按档位分别改写近义方法名", prompt)
        self.assertIn("单独标点/符号作为步骤占位", prompt)

    def test_repair_prompt_does_not_anchor_on_broken_guidance(self) -> None:
        broken = sample_guidance_output()
        broken["guidance"]["advanced"]["parts"][0]["steps"] = [", "]
        marker = "上一次JSON仅供定位错误，不得机械照抄：\n"

        prompt = pilot.validation_repair_prompt(
            "BASE",
            broken,
            [
                "guidance.advanced.parts[0].steps[0]必须包含文字、数字或公式内容，"
                "不能只有标点或符号"
            ],
        )

        self.assertIn("从零重建三档guidance", prompt)
        prior = json.loads(prompt.split(marker, 1)[1])
        self.assertNotIn("guidance", prior)
        self.assertEqual(prior["question_blocks"], broken["question_blocks"])

    def test_punctuation_only_steps_are_rejected_before_rendering(self) -> None:
        output = sample_guidance_output()
        for tier_key in ("advanced", "standard", "foundation"):
            steps = output["guidance"][tier_key]["parts"][0]["steps"]
            output["guidance"][tier_key]["parts"][0]["steps"] = [
                ", " for _ in steps
            ]

        errors = pilot.validate_output(
            output,
            requested_outputs=None,
            device_policy=True,
        )
        self.assertTrue(any("不能只有标点或符号" in item for item in errors), errors)
        with self.assertRaisesRegex(ValueError, "只有标点或符号"):
            tier_blocks("advanced", output)

        with tempfile.TemporaryDirectory(dir=SOURCE_DIR) as temporary:
            output_dir = Path(temporary) / "must_not_exist"
            with self.assertRaisesRegex(ValueError, "只有标点或符号"):
                render_four_images(
                    question_id="punctuation-regression",
                    model_id="mock",
                    source_image=Image.new("L", (640, 480), 255),
                    output=output,
                    output_dir=output_dir,
                )
            self.assertFalse(output_dir.exists())

    def test_near_synonym_method_names_are_safely_unified(self) -> None:
        output = sample_guidance_output()
        output["guidance"]["standard"]["parts"][0]["method_name"] = (
            "导数符号判断法"
        )
        output["guidance"]["foundation"]["parts"][0]["method_name"] = (
            "导数符号应用法"
        )
        parsed, errors = parse_model_output(json.dumps(output, ensure_ascii=False))
        self.assertIsNotNone(parsed)
        self.assertEqual(
            errors,
            ["同一小问在三个档位中的方法编号、名称和顺序必须完全一致"],
        )

        repaired, remaining, audit = apply_dynamic_local_repairs(parsed, errors)

        self.assertEqual(remaining, [])
        names = [
            repaired["guidance"][tier_key]["parts"][0]["method_name"]
            for tier_key in ("advanced", "standard", "foundation")
        ]
        self.assertEqual(names, ["导数符号法"] * 3)
        self.assertTrue(
            any(
                item.get("action") == "normalize_cross_tier_method_name"
                for item in audit
            ),
            audit,
        )

    def test_captured_parallel_line_method_labels_are_safely_unified(self) -> None:
        output = sample_guidance_output()
        output["guidance"]["advanced"]["parts"][0]["method_name"] = "平行导角法"
        output["guidance"]["standard"]["parts"][0]["method_name"] = "平行线找角法"
        output["guidance"]["foundation"]["parts"][0]["method_name"] = (
            "平行线性质应用"
        )
        parsed, errors = parse_model_output(json.dumps(output, ensure_ascii=False))
        self.assertIsNotNone(parsed)
        self.assertEqual(
            errors,
            ["同一小问在三个档位中的方法编号、名称和顺序必须完全一致"],
        )

        repaired, remaining, audit = apply_dynamic_local_repairs(parsed, errors)

        self.assertEqual(remaining, [])
        self.assertEqual(
            [
                repaired["guidance"][tier]["parts"][0]["method_name"]
                for tier in ("advanced", "standard", "foundation")
            ],
            ["平行导角法"] * 3,
        )
        self.assertTrue(audit)

    def test_captured_degenerate_shape_is_not_method_name_repaired(self) -> None:
        """Equivalent regression for the response whose 26 steps were ``, ``."""

        output = sample_guidance_output()
        output["guidance"]["standard"]["parts"][0]["method_name"] = (
            "导数符号判断法"
        )
        output["guidance"]["foundation"]["parts"][0]["method_name"] = (
            "导数符号应用法"
        )
        for tier_key in ("advanced", "standard", "foundation"):
            steps = output["guidance"][tier_key]["parts"][0]["steps"]
            output["guidance"][tier_key]["parts"][0]["steps"] = [
                ", " for _ in steps
            ]
        parsed, errors = parse_model_output(json.dumps(output, ensure_ascii=False))
        self.assertIsNotNone(parsed)
        self.assertTrue(any("不能只有标点或符号" in item for item in errors), errors)

        repaired, remaining, audit = apply_dynamic_local_repairs(parsed, errors)

        self.assertTrue(any("不能只有标点或符号" in item for item in remaining))
        self.assertEqual(
            repaired["guidance"]["standard"]["parts"][0]["method_name"],
            "导数符号判断法",
        )
        self.assertFalse(
            any(
                item.get("action") == "normalize_cross_tier_method_name"
                for item in audit
            ),
            audit,
        )
        direct_audit: list[dict] = []
        direct = normalize_cross_tier_method_names(
            parsed,
            ["同一小问在三个档位中的方法编号、名称和顺序必须完全一致"],
            direct_audit,
        )
        self.assertEqual(
            direct["guidance"]["standard"]["parts"][0]["method_name"],
            "导数符号判断法",
        )
        self.assertEqual(direct_audit, [])

    def test_method_name_normalizer_requires_identical_method_structure(self) -> None:
        output = sample_guidance_output()
        output["guidance"]["standard"]["parts"][0]["method_name"] = (
            "导数符号判断法"
        )
        output["guidance"]["foundation"]["parts"][0]["method_name"] = (
            "导数符号应用法"
        )
        output["guidance"]["foundation"]["parts"][0]["alternative_methods"] = [
            {
                "method_id": "M2",
                "method_name": "导数图像法",
                "steps": [
                    "根据题设关系画出导函数示意图。",
                    "找出导函数图像与横轴的交点。",
                    "按图像在横轴上下方判断导数符号。",
                    "将导数符号对应到函数的增减区间。",
                ],
            }
        ]
        audit: list[dict] = []

        repaired = normalize_cross_tier_method_names(
            output,
            ["同一小问在三个档位中的方法编号、名称和顺序必须完全一致"],
            audit,
        )

        self.assertEqual(
            repaired["guidance"]["standard"]["parts"][0]["method_name"],
            "导数符号判断法",
        )
        self.assertEqual(audit, [])

    def test_repair_candidate_must_strictly_improve_without_content_loss(self) -> None:
        before = sample_guidance_output()
        same_quality = copy.deepcopy(before)
        self.assertTrue(
            repair_candidate_strictly_improves(
                before,
                ["错误一", "错误二"],
                same_quality,
                ["错误一"],
            )
        )
        self.assertFalse(
            repair_candidate_strictly_improves(
                before,
                ["错误一"],
                same_quality,
                ["另一错误"],
            )
        )

        degraded = copy.deepcopy(before)
        degraded["guidance"]["advanced"]["parts"][0]["steps"][0] = ", "
        degraded["question_blocks"][0]["text"] = "讨论单调性。"
        self.assertFalse(
            repair_candidate_strictly_improves(
                before,
                ["错误一", "错误二"],
                degraded,
                [],
            )
        )

    def test_guidance_repair_preserves_validated_question_evidence(self) -> None:
        before = sample_guidance_output()
        before["guidance"]["advanced"]["parts"][0]["steps"] = [", "]
        repaired = sample_guidance_output()
        repaired["question_blocks"] = [
            {"type": "text", "text": "被修复请求缩短的题干", "figure_id": ""}
        ]
        repaired["knowledge_points"] = ["被缩减的知识点"]

        merged, errors, audit = merge_guidance_only_repair(
            before,
            [
                "guidance.advanced.parts[0].steps[0]必须包含文字、数字或公式内容，"
                "不能只有标点或符号"
            ],
            repaired,
        )

        self.assertIsNotNone(merged)
        self.assertEqual(errors, [])
        self.assertEqual(merged["question_blocks"], before["question_blocks"])
        self.assertEqual(merged["knowledge_points"], before["knowledge_points"])
        self.assertEqual(merged["guidance"], repaired["guidance"])
        self.assertEqual(audit[0]["action"], "merge_guidance_only_provider_repair")

    def test_guidance_tiers_keep_distinct_content_roles(self) -> None:
        output = sample_guidance_output()
        advanced = "\n".join(item[0] for item in tier_blocks("advanced", output))
        standard = "\n".join(item[0] for item in tier_blocks("standard", output))
        foundation = "\n".join(item[0] for item in tier_blocks("foundation", output))

        self.assertIn("导数符号图", advanced)
        self.assertIn("1. ", advanced)
        self.assertNotIn("学霸版", advanced)
        self.assertNotIn("临界点表格", advanced)
        self.assertNotIn("数轴分段", advanced)

        self.assertIn("临界点表格", standard)
        self.assertIn("易错点", standard)
        self.assertNotIn("导数符号图", standard)
        self.assertNotIn("数轴分段", standard)

        self.assertIn("数轴分段", foundation)
        self.assertIn("通用套路", foundation)
        self.assertNotIn("导数符号图", foundation)
        self.assertNotIn("临界点表格", foundation)

    def test_original_four_image_semantics_and_device_geometry(self) -> None:
        self.assertEqual(
            DEVICE_IMAGE_NAMES,
            (
                "01_question_knowledge.png",
                "02_guidance_advanced.png",
                "03_guidance_standard.png",
                "04_guidance_foundation.png",
            ),
        )
        self.assertEqual(
            DEVICE_IMAGE_KINDS,
            ("question_knowledge", "advanced", "standard", "foundation"),
        )
        self.assertEqual(DISPLAY_WIDTH_CM, 9.0)
        self.assertEqual(
            (MARGIN_X, BODY_SIZE, FORMULA_SIZE, BODY_LINE, FORMULA_LINE, BOTTOM),
            (32, 32, 34, 52, 62, 34),
        )

        with tempfile.TemporaryDirectory(dir=SOURCE_DIR) as temporary:
            output_dir = Path(temporary)
            source = Image.new("L", (1000, 600), 255)
            song, song_bold, times, times_bold = resolve_device_fonts()
            rendered = render_four_images(
                question_id="math-contract",
                model_id="mock",
                source_image=source,
                output=sample_guidance_output(),
                output_dir=output_dir,
                song_path=song,
                song_bold_path=song_bold,
                times_path=times,
                times_bold_path=times_bold,
            )
            contract = validate_device_image_set(output_dir, require_all=True)

            self.assertEqual(list(rendered["files"]), list(DEVICE_IMAGE_KINDS))
            self.assertTrue(contract["valid"], contract["errors"])
            self.assertEqual(contract["contract"]["display_width_cm"], 9.0)
            self.assertEqual(
                [item["filename"] for item in contract["files"]],
                list(DEVICE_IMAGE_NAMES),
            )
            self.assertEqual(
                [item["kind"] for item in contract["files"]],
                list(DEVICE_IMAGE_KINDS),
            )
            self.assertEqual(
                [item["sequence"] for item in contract["files"]],
                [1, 2, 3, 4],
            )
            for item in contract["files"]:
                self.assertEqual(
                    (item["width"], item["mode"], item["format"]),
                    (800, "L", "PNG"),
                )
                with Image.open(item["path"]) as image:
                    dpi = image.info["dpi"]
                width_cm = 800 / float(dpi[0]) * 2.54
                self.assertAlmostEqual(width_cm, 9.0, places=3)

            first = rendered["files"]["question_knowledge"]
            self.assertIn("source_figure_crops", first)
            self.assertNotIn("resize_only_no_crop", first)
            self.assertFalse(rendered["contract"]["visible_page_header"])
            self.assertFalse(rendered["contract"]["visible_tier_title"])
            self.assertEqual(rendered["contract"]["body_px"], 32)
            self.assertEqual(rendered["contract"]["formula_px"], 34)

    def test_figure_references_must_be_complete(self) -> None:
        output = sample_guidance_output()
        output["question_blocks"] = [
            {
                "type": "text",
                "text": "比较图1与图2中函数图像的变化趋势。",
                "figure_id": "",
            },
            {"type": "figure", "text": "", "figure_id": "fig_1"},
        ]
        output["source_figures"] = [
            {
                "figure_id": "fig_1",
                "bbox_1000": [100, 200, 450, 700],
                "caption": "图1：第一个函数图像",
            }
        ]

        errors = validate_figure_reference_completeness(output)
        self.assertEqual(len(errors), 1)
        self.assertIn("图2", errors[0])
        self.assertIn("第一张原题知识图不完整", errors[0])

        output["question_blocks"].append(
            {"type": "figure", "text": "", "figure_id": "fig_2"}
        )
        output["source_figures"].append(
            {
                "figure_id": "fig_2",
                "bbox_1000": [550, 200, 900, 700],
                "caption": "图2：第二个函数图像",
            }
        )
        self.assertEqual(validate_figure_reference_completeness(output), [])

    def test_unnumbered_figure_reference_requires_a_source_crop(self) -> None:
        output = sample_guidance_output()
        output["question_blocks"][0]["text"] = "函数图像如图所示，讨论其单调性。"
        errors = validate_figure_reference_completeness(output)
        self.assertEqual(len(errors), 1)
        self.assertIn("source_figures为空", errors[0])

        # “图像” alone describes a mathematical object rather than citing a
        # printed source figure, so it must not create a false requirement.
        output["question_blocks"][0]["text"] = "画出函数的图像。"
        self.assertEqual(validate_figure_reference_completeness(output), [])

    def test_referenced_table_requires_a_source_crop(self) -> None:
        output = sample_guidance_output()
        output["question_blocks"][0]["text"] = "产品质量统计如下表，求一级品的频率。"
        output["visual_evidence"] = ["列联表中给出两类产品的频数"]
        errors = validate_figure_reference_completeness(output)
        self.assertEqual(len(errors), 1)
        self.assertIn("表格", errors[0])
        self.assertIn("第一张原题知识图不完整", errors[0])

    def test_dynamic_target_guard_rejects_final_symbol_value(self) -> None:
        output = sample_guidance_output()
        output["question_blocks"][0]["text"] = (
            "已知函数 $f(x)$，求函数的最大值 $M$。"
        )
        targets = infer_dynamic_requested_outputs(output)
        self.assertTrue(any("M" in item["symbol_aliases"] for item in targets))
        for tier in ("advanced", "standard", "foundation"):
            output["guidance"][tier]["parts"][0]["steps"][-1] = "解得 $M=2$。"
        parsed, errors = parse_model_output(json.dumps(output, ensure_ascii=False))
        self.assertIsNotNone(parsed)
        self.assertTrue(any("单独解出了" in item for item in errors), errors)

    def test_dynamic_targets_cover_how_many_and_yes_no_questions(self) -> None:
        output = sample_guidance_output()
        output["question_blocks"] = [
            {"type": "text", "text": "（1）甲、乙一级品的频率分别是多少？", "figure_id": ""},
            {"type": "text", "text": "（2）能否有99%把握认为两者的产品质量有差异？", "figure_id": ""},
        ]
        for tier in ("advanced", "standard", "foundation"):
            original = output["guidance"][tier]["parts"][0]
            output["guidance"][tier]["parts"] = [
                {**copy.deepcopy(original), "part": "（1）"},
                {**copy.deepcopy(original), "part": "（2）"},
            ]
        targets = infer_dynamic_requested_outputs(output)
        self.assertEqual([item["part"] for item in targets], ["（1）", "（2）"])
        self.assertTrue(any("频率" in alias for alias in targets[0]["prose_aliases"]))
        self.assertTrue(any("有差异" in alias for alias in targets[1]["prose_aliases"]))

        leaked = copy.deepcopy(output)
        for tier in ("advanced", "standard", "foundation"):
            leaked["guidance"][tier]["parts"][0]["steps"][-1] = "一级品的频率分别为0.75和0.60"
            leaked["guidance"][tier]["parts"][1]["steps"][-1] = "有99%把握认为两者的产品质量有差异"
        _, errors = parse_model_output(json.dumps(leaked, ensure_ascii=False))
        self.assertTrue(any("（1）所求结论" in item for item in errors), errors)
        self.assertTrue(any("（2）的判定结论" in item for item in errors), errors)

    def test_conditional_decision_rule_is_not_mistaken_for_the_answer(self) -> None:
        output = sample_guidance_output()
        output["question_blocks"][0]["text"] = (
            "能否有99%把握认为两者的产品质量有差异？"
        )
        output["guidance"]["standard"]["parts"][0]["steps"][-1] = (
            "若计算值大于6.635，则拒绝零假设，说明有差异"
        )
        output["guidance"]["advanced"]["parts"][0]["steps"][-1] = (
            "依据比较结果判定是否有99%把握认为两者有差异"
        )
        output["guidance"]["foundation"]["parts"][0]["steps"][-1] = (
            "明确零假设H₀：两者没有差异"
        )
        _, errors = parse_model_output(json.dumps(output, ensure_ascii=False))
        self.assertFalse(any("判定结论" in item for item in errors), errors)

        for verdict in (
            "有99%把握认为两者有差异",
            "没有99%把握认为两者有差异",
            "两者有差异",
            "产品质量有差异",
            "先判断是否有差异，计算后说明两者有差异",
            "若统计量超过阈值就说明有差异；实际计算后所以两者有差异",
            "如果统计量超过阈值就说明有差异，实际计算后所以两者有差异",
        ):
            leaked = copy.deepcopy(output)
            leaked["guidance"]["advanced"]["parts"][0]["steps"][-1] = verdict
            _, leaked_errors = parse_model_output(
                json.dumps(leaked, ensure_ascii=False)
            )
            self.assertTrue(
                any("判定结论" in item for item in leaked_errors),
                (verdict, leaked_errors),
            )

    def test_general_yes_no_answers_are_rejected(self) -> None:
        cases = (
            ("命题$p$是否正确？", "正确"),
            ("方程是否有实数解？", "有实数解"),
            ("两直线是否垂直？", "两直线垂直"),
            ("两个表达式是否相等？", "两个表达式相等"),
            ("这些边长能否构成三角形？", "能构成三角形"),
            ("方程是否没有实数解？", "有实数解"),
            ("两个表达式是否不相等？", "两个表达式相等"),
        )
        for question, verdict in cases:
            output = sample_guidance_output()
            output["question_blocks"][0]["text"] = question
            output["guidance"]["advanced"]["parts"][0]["steps"][-1] = verdict
            _, errors = parse_model_output(json.dumps(output, ensure_ascii=False))
            self.assertTrue(
                any("判定结论" in item for item in errors),
                (question, verdict, errors),
            )

    def test_derived_frequency_values_are_rejected_without_banning_table_data(self) -> None:
        base = sample_guidance_output()
        base["question_blocks"][0]["text"] = "甲、乙一级品的频率分别是多少？"
        base["visual_evidence"] = [
            "甲机床一级品150件、总数200件；乙机床一级品120件、总数200件"
        ]
        for leaked_step in (
            "一级品的频率为0.75",
            "一级品的频率分别为0.75和0.60",
            "甲机床为0.75，乙机床为0.60",
        ):
            leaked = copy.deepcopy(base)
            leaked["guidance"]["advanced"]["parts"][0]["steps"][-1] = leaked_step
            _, errors = parse_model_output(json.dumps(leaked, ensure_ascii=False))
            self.assertTrue(
                any("所求结论" in item or "数值结果" in item for item in errors),
                (leaked_step, errors),
            )

        allowed = copy.deepcopy(base)
        allowed["guidance"]["advanced"]["parts"][0]["steps"][-1] = (
            "令甲机床一级品频数为150，总数为200，先列出频数与总数之比"
        )
        _, allowed_errors = parse_model_output(json.dumps(allowed, ensure_ascii=False))
        self.assertFalse(
            any("所求结论" in item or "数值结果" in item for item in allowed_errors),
            allowed_errors,
        )

    def test_terminal_how_many_target_is_not_lost_in_a_long_question(self) -> None:
        cases = (
            ("某班有40人，其中男生22人，女生人数是多少？", "女生人数为18"),
            ("抽检100件产品，其中90件合格，合格率是多少？", "合格率为0.9"),
            ("三角形两边长为3和4，第三边长是多少？", "第三边长为5"),
            ("某数加3等于8，这个数是多少？", "这个数为5"),
        )
        for question, answer in cases:
            output = sample_guidance_output()
            output["question_blocks"][0]["text"] = question
            output["guidance"]["advanced"]["parts"][0]["steps"][-1] = answer
            targets = infer_dynamic_requested_outputs(output)
            self.assertTrue(
                any(
                    alias in answer
                    for item in targets
                    for alias in item.get("prose_aliases", [])
                ),
                (question, targets),
            )
            _, errors = parse_model_output(json.dumps(output, ensure_ascii=False))
            self.assertTrue(
                any("所求结论" in item for item in errors),
                (question, answer, errors),
            )

        base = sample_guidance_output()
        base["question_blocks"][0]["text"] = "某班有40人，其中男生22人，女生人数是多少？"
        for answer in (
            "结果为18",
            "最终结果是18",
            "18",
            "约为18人",
            "因此是18",
            "$18$",
            r"$18\,\mathrm{人}$",
            "最终为$18$",
            "所以$18$",
            "女生人数$=18$",
        ):
            output = copy.deepcopy(base)
            output["guidance"]["advanced"]["parts"][0]["steps"][-1] = answer
            _, errors = parse_model_output(json.dumps(output, ensure_ascii=False))
            self.assertTrue(
                any("最终数值" in item or "所求结论" in item for item in errors),
                (answer, errors),
            )

        allowed = copy.deepcopy(base)
        allowed["question_blocks"][0]["text"] += "附临界值6.635。"
        allowed["guidance"]["advanced"]["parts"][0]["steps"][-1] = (
            "将计算结果与6.635比较"
        )
        _, allowed_errors = parse_model_output(json.dumps(allowed, ensure_ascii=False))
        self.assertFalse(any("最终数值" in item for item in allowed_errors), allowed_errors)

    def test_elementary_quantity_wording_cannot_leak_the_final_count(self) -> None:
        cases = (
            ("书架原有48本书，借出19本后还剩多少本？", "还剩29本"),
            ("红球有12个，蓝球有8个，一共有多少个球？", "一共有20个球"),
            ("24本书平均分给6人，每人分得多少本？", "每人分得4本"),
        )
        for question, leaked_step in cases:
            output = sample_guidance_output()
            output["question_blocks"][0]["text"] = question
            output["guidance"]["advanced"]["parts"][0]["steps"][-1] = leaked_step
            _, errors = parse_model_output(json.dumps(output, ensure_ascii=False))
            self.assertTrue(
                any("数量结果" in item for item in errors),
                (question, leaked_step, errors),
            )

        allowed = sample_guidance_output()
        allowed["question_blocks"][0]["text"] = (
            "书架原有48本书，借出19本后还剩多少本？"
        )
        allowed["guidance"]["advanced"]["parts"][0]["steps"][-1] = (
            "用原有本数减去借出本数，列式为 $48-19$"
        )
        _, allowed_errors = parse_model_output(
            json.dumps(allowed, ensure_ascii=False)
        )
        self.assertFalse(any("数量结果" in item for item in allowed_errors), allowed_errors)

    def test_generic_guard_rejects_proof_choice_and_fill_closure(self) -> None:
        for leaked in (
            "因此原命题成立，得证。",
            "正确选项为 B。",
            "应填 $2$。",
        ):
            output = sample_guidance_output()
            output["guidance"]["advanced"]["parts"][0]["steps"][0] = leaked
            self.assertTrue(validate_generic_answer_blind(output), leaked)

    def test_device_validation_keeps_three_tiers_structurally_consistent(self) -> None:
        output = sample_guidance_output()
        output["guidance"]["advanced"]["parts"][0]["steps"] = ["只有一步"]
        output["guidance"]["standard"]["parts"][0]["method_name"] = "另一方法"
        errors = pilot.validate_output(
            output, requested_outputs=None, device_policy=True
        )
        self.assertTrue(any("数量必须在2到4之间" in item for item in errors), errors)
        self.assertTrue(any("方法编号、名称和顺序" in item for item in errors), errors)

    def test_spurious_leading_dollar_before_chinese_prose_is_repaired(self) -> None:
        raw = sample_guidance_output()
        raw["guidance"]["advanced"]["parts"][0]["steps"][0] = (
            "$应用频率公式$f=\\frac{n_i}{N}$ 建立关系"
        )
        parsed, errors = parse_model_output(json.dumps(raw, ensure_ascii=False))
        self.assertIsNotNone(parsed)
        self.assertFalse(any("定界符$不成对" in item for item in errors), errors)
        step = parsed["guidance"]["advanced"]["parts"][0]["steps"][0]
        self.assertEqual(step, "应用频率公式$f=\\frac{n_{i}}{N}$ 建立关系")

    def test_printed_formula_is_not_misclassified_as_a_source_figure(self) -> None:
        output = sample_guidance_output()
        output["question_blocks"] = [
            {"type": "text", "text": "附：$K^{2} = $", "figure_id": ""},
            {"type": "figure", "text": "", "figure_id": "fig_formula"},
        ]
        output["source_figures"] = [
            {
                "figure_id": "fig_formula",
                "bbox_1000": [135, 565, 580, 625],
                "caption": "独立性检验统计量 $K^2$ 计算公式",
            }
        ]
        output["base_formulas"].append(
            {
                "formula": r"K^{2} = \frac{n(ad-bc)^{2}}{(a+b)(c+d)(a+c)(b+d)}",
                "condition": "2×2列联表独立性检验",
            }
        )

        repaired, audit = normalize_formula_source_figures(output)

        self.assertEqual(repaired["source_figures"], [])
        self.assertEqual(len(repaired["question_blocks"]), 1)
        self.assertIn(r"\frac{n(ad-bc)^{2}}", repaired["question_blocks"][0]["text"])
        self.assertEqual(audit[0]["figure_id"], "fig_formula")
        self.assertEqual(
            pilot.validate_output(repaired, requested_outputs=None, device_policy=True),
            [],
        )


class ModelPayloadContractTests(unittest.TestCase):
    def test_exact_preprocessed_png_is_the_only_model_image(self) -> None:
        with tempfile.TemporaryDirectory(dir=SOURCE_DIR) as temporary:
            image_path = Path(temporary) / "processed.png"
            Image.new("L", (640, 480), 245).save(image_path, format="PNG")
            data_url, audit = png_data_url(image_path)
            schema = json.loads(
                (SOURCE_DIR / "math_guidance_schema.json").read_text(encoding="utf-8")
            )
            payload = build_request_payload(
                model_id="mock",
                prompt="数学搜题机提示词",
                data_url=data_url,
                schema=schema,
                max_tokens=10,
            )

            image_items = [
                item
                for item in payload["messages"][0]["content"]
                if item.get("type") == "image_url"
            ]
            self.assertEqual(len(image_items), 1)
            transmitted = image_items[0]["image_url"]["url"]
            decoded = base64.b64decode(transmitted.split(",", 1)[1])
            self.assertEqual(decoded, image_path.read_bytes())
            self.assertTrue(audit["same_bytes_as_render_source"])
            self.assertEqual(audit["transmitted_media_type"], "image/png")

    def test_new_jobs_use_random_ids_and_reject_non_images(self) -> None:
        with tempfile.TemporaryDirectory(dir=SOURCE_DIR) as temporary:
            processor_root = Path(temporary) / "processor"

            class StubProcessor:
                output_root = processor_root

            pipeline = SearchMachinePipeline(
                runs_root=Path(temporary) / "runs",
                processor=StubProcessor(),
            )
            image_path = Path(temporary) / "camera.png"
            Image.new("RGB", (40, 30), "white").save(image_path)
            state = pipeline.new_job(image_path.read_bytes(), "camera.png")
            self.assertRegex(
                state["job_id"],
                r"^[0-9]{8}T[0-9]{6}_[0-9a-f]{32}$",
            )
            with self.assertRaisesRegex(ValueError, "图片无法识别"):
                pipeline.new_job(b"<html>bad</html>", "bad.png")

    def test_recovery_rechecks_new_answer_guards_before_rendering(self) -> None:
        with tempfile.TemporaryDirectory(dir=SOURCE_DIR) as temporary:
            root = Path(temporary)

            class StubProcessor:
                output_root = root / "processor"

            pipeline = SearchMachinePipeline(
                runs_root=root / "runs",
                processor=StubProcessor(),
            )
            job_id = "20260813T120000_" + "a" * 32
            job_dir = pipeline.runs_root / job_id
            (job_dir / "audit").mkdir(parents=True)
            Image.new("L", (320, 240), 255).save(
                job_dir / "audit" / "preprocessed_source.png"
            )
            (job_dir / "job.json").write_text(
                json.dumps({"job_id": job_id, "status": "failed"}, ensure_ascii=False),
                encoding="utf-8",
            )
            leaked = sample_guidance_output()
            leaked["question_blocks"][0]["text"] = "求函数的最大值$M$。"
            leaked["guidance"]["advanced"]["parts"][0]["steps"][-1] = (
                "解得$M=2$"
            )
            (job_dir / "audit" / "model_validation_failure.json").write_text(
                json.dumps(
                    {"model_output": leaked, "validation_errors": []},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "仍未通过数学模块强校验"):
                pipeline.recover_validation_failure(job_id)


class PublicApiAndWorkerTests(unittest.TestCase):
    def test_public_job_does_not_expose_server_paths(self) -> None:
        state = {
            "job_id": "20260813T120000_0123456789abcdef0123456789abcdef",
            "status": "completed",
            "error": {
                "type": "FileNotFoundError",
                "message": r"D:\private\secret\font.ttf 不存在",
            },
            "image_contract": {
                "valid": True,
                "expected": list(DEVICE_IMAGE_NAMES),
                "observed": list(DEVICE_IMAGE_NAMES),
                "image_count": 4,
                "files": [
                    {
                        "path": r"D:\private\01_question_knowledge.png",
                        "filename": "01_question_knowledge.png",
                    }
                ],
                "errors": [],
                "contract": {"image_count": 4},
            },
            "images": [
                {
                    "path": r"D:\private\01_question_knowledge.png",
                    "filename": "01_question_knowledge.png",
                    "kind": "question_knowledge",
                    "sequence": 1,
                }
            ],
        }
        public = public_job(state)
        serialized = json.dumps(public, ensure_ascii=False)
        self.assertNotIn(r"D:\private", serialized)
        self.assertNotIn("files", public["image_contract"])
        self.assertNotIn("path", public["images"][0])
        self.assertNotIn(r"D:\private", public["error"]["message"])
        self.assertEqual(
            public["images"][0]["url"],
            "/jobs/20260813T120000_0123456789abcdef0123456789abcdef/"
            "01_question_knowledge.png",
        )

    def test_worker_survives_one_unhandled_job_failure(self) -> None:
        calls: list[str] = []

        class FakePipeline:
            def solve(self, job_id: str) -> None:
                calls.append(job_id)
                if job_id == "bad":
                    raise RuntimeError("simulated pre-guard failure")

            def read_job(self, job_id: str) -> dict:
                return {"job_id": job_id}

            def job_path(self, job_id: str) -> Path:
                return Path(".")

            def _write_state(self, *_args) -> None:
                return None

        candidate = app_module.JobWorker(3)
        candidate.pending.put("bad")
        candidate.pending.put("good")
        candidate.pending.put(None)
        with patch.object(app_module, "pipeline", FakePipeline()):
            thread = threading.Thread(target=candidate._loop)
            thread.start()
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(calls, ["bad", "good"])
        self.assertEqual(candidate.pending.unfinished_tasks, 0)

    def test_health_reports_worker_liveness(self) -> None:
        with patch.object(app_module.worker, "is_alive", return_value=False):
            response = app_module.health()
        self.assertEqual(response.status_code, 503)
        body = json.loads(response.body)
        self.assertEqual(body["status"], "unavailable")
        self.assertFalse(body["worker_alive"])

        with patch.object(app_module.worker, "is_alive", return_value=True):
            response = app_module.health()
        self.assertEqual(response.status_code, 200)
        body = json.loads(response.body)
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["service"], "test3-math-search-machine")
        self.assertTrue(body["worker_alive"])


if __name__ == "__main__":
    unittest.main()
