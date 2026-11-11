#!/usr/bin/env python3
"""Tests for the latency-budgeted shared-skeleton provider contract."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from compact_guidance import (
    COMPACT_PREFIX_KEYS,
    CompactStreamCollector,
    adaptive_image_policy,
    classify_problem_complexity,
    curriculum_scope_instruction,
    expand_compact_output,
    infer_curriculum_stage,
    local_fallback_compact,
    parse_provider_document,
    resolve_provider_image_policy,
)
from five_tier_guidance import project_display_tier


SOURCE_DIR = Path(__file__).resolve().parent


def compact_sample() -> dict:
    return {
        "l1": [{"n": "整题", "s": [
            "哪个数在数轴上离原点最远？",
            "比较时还要避免把数本身的大小当成距离吗？",
        ]}],
        "l2": [
            "绝对值定义：把比较对象转成到原点的距离",
            "数轴几何意义：从位置读取距离的相对大小",
            "有理数比较：区分数的大小与绝对值大小",
        ],
        "l3": [{
            "n": "整题",
            "s": [
                "由绝对值的几何意义可知，应先比较各点到原点的距离。",
                "距离的比较确定后，再回到绝对值的定义判断目标量。",
            ],
        }],
        "l4": [{
            "n": "整题",
            "s": [
                "由各数在数轴上的位置读取它们到原点的距离",
                "因为绝对值表示到原点的距离，所以把距离写成绝对值",
                "比较距离前需确认数轴刻度一致，并保留最终判断步骤",
            ],
        }],
        "l5": [{
            "n": "整题",
            "s": [
                "由数轴位置可知每个数对应一个到原点的距离",
                "因为绝对值等于这个距离，所以先读取各点距离",
                "把各个距离分别写成对应的绝对值表达式",
                "比较距离大小得到仍未写出终值的大小关系",
                "最后核对没有把负数的大小与到原点距离混淆",
            ],
            "pit": "不要把数的大小与绝对值大小混淆",
        }],
        "f": [{"x": "|a|=d(a,0)", "c": "比较数到原点的距离时"}],
        "pat": "把绝对值转成到原点的距离后比较",
        "s": "c",
        "u": [],
        "q": [{"t": "t", "x": "比较各数的绝对值。", "f": ""}],
        "g": [],
        "e": [],
    }


def legacy_compact_sample() -> dict:
    sample = compact_sample()
    return {
        "p": [{
            "n": "整题",
            "m": [{
                "i": "M1",
                "n": "数轴比较法",
                "s": list(sample["l5"][0]["s"]),
                "pit": sample["l5"][0]["pit"],
            }],
        }],
        "k": list(sample["l2"]),
        "f": list(sample["f"]),
        "pat": sample["pat"],
        "s": "c",
        "u": [],
        "q": list(sample["q"]),
        "g": [],
        "e": [],
    }


class CompactExpansionTests(unittest.TestCase):
    def test_provider_json_format_repairs_preserve_model_text(self) -> None:
        raw_control = '{"l1":[{"n":"整题","s":["模型第一行\n模型第二行"]}]}'
        parsed = parse_provider_document(raw_control)

        self.assertEqual(parsed["l1"][0]["s"][0], "模型第一行\n模型第二行")
        self.assertTrue(any(
            item["action"] == "escape_raw_json_string_control"
            for item in parsed["_json_format_repairs"]
        ))

        illegal_latex_escape = r'{"l1":[{"n":"整题","s":["由$\alpha+1$建立关系"]}]}'
        parsed = parse_provider_document(illegal_latex_escape)
        self.assertEqual(parsed["l1"][0]["s"][0], r"由$\alpha+1$建立关系")
        self.assertTrue(any(
            item["action"] == "double_illegal_json_string_backslash"
            for item in parsed["_json_format_repairs"]
        ))

    def test_provider_json_restores_valid_escape_latex_ambiguity(self) -> None:
        decoded_control = r'{"l1":[{"n":"整题","s":["由$\frac{1}{2}$建立比例"]}]}'
        parsed = parse_provider_document(decoded_control)

        self.assertEqual(parsed["l1"][0]["s"][0], r"由$\frac{1}{2}$建立比例")
        self.assertTrue(any(
            item["action"] == "restore_json_decoded_latex_command"
            for item in parsed["_json_format_repairs"]
        ))

    def test_provider_json_does_not_guess_truncated_structure(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_provider_document('{"l1":[{"n":"整题"}')

    def test_curriculum_stage_inference_is_conservative_and_subject_aware(self) -> None:
        junior = infer_curriculum_stage(
            "利用频数分布直方图判断中位数并估计平均数",
            subject="mathematics",
        )
        senior_math = infer_curriculum_stage(
            "已知函数的导数，研究单调区间与极值",
            subject="mathematics",
        )
        senior_physics = infer_curriculum_stage(
            "物体做匀变速直线运动，求加速度",
            subject="physics",
        )
        unlabeled = infer_curriculum_stage(
            "根据题目中的条件求出未知量",
            subject="mathematics",
        )

        self.assertEqual(junior["stage"], "junior_high")
        self.assertTrue(junior["conservative"])
        self.assertEqual(senior_math["stage"], "senior_high")
        self.assertIn("导数", senior_math["signals"])
        self.assertEqual(senior_physics["stage"], "senior_high")
        self.assertIn("加速度", senior_physics["signals"])
        self.assertEqual(unlabeled["stage"], "senior_high")
        self.assertEqual(unlabeled["reason"], "unlabeled_senior_ceiling")

        explicit_junior = infer_curriculum_stage(
            "九年级数学：根据函数图像完成判断",
            subject="mathematics",
        )
        self.assertEqual(explicit_junior["stage"], "junior_high")
        self.assertEqual(explicit_junior["reason"], "explicit_junior_grade")

    def test_curriculum_overscope_is_reported_without_discarding_model_text(self) -> None:
        junior_sample = compact_sample()
        junior_sample["l2"][0] = "利用导数研究单调性"
        junior_output, junior_audit = expand_compact_output(
            junior_sample,
            subject="mathematics",
            curriculum_stage="junior_high",
        )
        self.assertFalse(junior_audit["curriculum_guard_passed"])
        self.assertEqual(junior_audit["curriculum_violations"], ["导数"])
        self.assertEqual(junior_audit["curriculum_validation_mode"], "report_only")
        self.assertEqual(
            junior_output["display_guidance"]["upper"]["knowledge_points"][0],
            "利用导数研究单调性",
        )

        senior_sample = compact_sample()
        senior_sample["l5"][0]["s"][0] = "使用洛必达求出该极限的中间关系"
        senior_output, senior_audit = expand_compact_output(
            senior_sample,
            subject="mathematics",
            curriculum_stage="senior_high",
        )
        self.assertFalse(senior_audit["curriculum_guard_passed"])
        self.assertEqual(senior_audit["curriculum_violations"], ["洛必达"])
        self.assertIn(
            "使用洛必达",
            senior_output["display_guidance"]["novice"]["parts"][0]["steps"][0],
        )

        allowed = compact_sample()
        allowed["l2"][0] = "导数与函数单调性"
        output, audit = expand_compact_output(
            allowed,
            subject="mathematics",
            curriculum_stage="senior_high",
        )
        self.assertTrue(audit["curriculum_guard_passed"])
        self.assertEqual(audit["curriculum_stage"], "senior_high")
        self.assertEqual(output["display_guidance"]["upper"]["knowledge_points"][0], "导数与函数单调性")

    def test_curriculum_instruction_names_the_detected_ceiling(self) -> None:
        scope = infer_curriculum_stage(
            "九年级物理：测量小灯泡的电功率",
            subject="physics",
        )
        instruction = curriculum_scope_instruction(scope)

        self.assertIn("最高学段判定为：初中", instruction)
        self.assertIn("不得", instruction)
        self.assertIn("加速度", instruction)

    def test_inverse_function_hyperbola_word_is_contextually_allowed(self) -> None:
        sample = compact_sample()
        sample["l2"][0] = "反比例函数图像是双曲线的一支"

        output, audit = expand_compact_output(
            sample,
            subject="mathematics",
            fallback_question_text="九年级数学：研究反比例函数y=k/x的图像",
            curriculum_stage="junior_high",
        )

        self.assertTrue(audit["curriculum_guard_passed"])
        self.assertIn(
            "双曲线",
            output["display_guidance"]["upper"]["knowledge_points"][0],
        )
        instruction = curriculum_scope_instruction(
            infer_curriculum_stage(
                "九年级数学：研究反比例函数y=k/x的图像",
                subject="mathematics",
            ),
            question_text="九年级数学：研究反比例函数y=k/x的图像",
        )
        self.assertIn("可将其图像形状称为“双曲线的一支/分支”", instruction)
        self.assertNotIn("禁止词示例：导数、微分、积分、极限、洛必达、泰勒、复数、平面向量、空间向量、圆锥曲线、椭圆、双曲线", instruction)

    def test_hyperbola_is_reported_without_inverse_function_context(self) -> None:
        sample = compact_sample()
        sample["l2"][0] = "利用双曲线的焦点性质"

        output, audit = expand_compact_output(
            sample,
            subject="mathematics",
            fallback_question_text="九年级数学：研究一次函数图像",
            curriculum_stage="junior_high",
        )

        self.assertFalse(audit["curriculum_guard_passed"])
        self.assertIn("双曲线", audit["curriculum_violations"])
        self.assertIn(
            "双曲线",
            output["display_guidance"]["upper"]["knowledge_points"][0],
        )

    def test_five_provider_tiers_are_rendered_without_local_projection(self) -> None:
        output, audit = expand_compact_output(
            compact_sample(), subject="mathematics"
        )

        tiers = output["display_guidance"]
        self.assertEqual(
            tuple(tiers), ("elite", "upper", "middle", "lower", "novice")
        )
        self.assertEqual(tiers["elite"]["parts"][0]["steps"], compact_sample()["l1"][0]["s"])
        self.assertEqual(tiers["upper"]["knowledge_points"], compact_sample()["l2"])
        self.assertEqual(tiers["middle"]["parts"][0]["steps"], compact_sample()["l3"][0]["s"])
        self.assertEqual(tiers["lower"]["parts"][0]["steps"], compact_sample()["l4"][0]["s"])
        self.assertEqual(tiers["novice"]["parts"][0]["steps"], compact_sample()["l5"][0]["s"])
        self.assertFalse(audit["local_tier_expansion"])
        self.assertTrue(audit["model_direct_render"])
        self.assertEqual(audit["model_tier_count"], 5)
        self.assertEqual(audit["method_count"], 1)
        self.assertEqual(audit["provider_contract"], "compact-five-natural-tiers-v4")

        for key in ("elite", "upper", "middle", "lower", "novice"):
            self.assertEqual(project_display_tier(key, output), tiers[key])

    def test_completed_leading_skeleton_survives_an_unfinished_stream(self) -> None:
        prefix = {key: compact_sample()[key] for key in COMPACT_PREFIX_KEYS}
        collector = CompactStreamCollector()
        collector(
            json.dumps(prefix, ensure_ascii=False)[:-1] + ',"s":',
            15900.0,
        )

        self.assertEqual(collector.completed_skeleton(), prefix)
        self.assertEqual(collector.provider_elapsed_ms, 15900.0)

    def test_explicit_terminal_values_are_removed_before_rendering(self) -> None:
        sample = compact_sample()
        sample["l5"][0]["s"][0] = (
            r"由样本空间可知该事件的概率为$\frac{3}{5}$。"
        )
        sample["l5"][0]["s"][1] = "若所有个体都符合条件，则概率为1。"

        output, _ = expand_compact_output(sample, subject="mathematics")
        steps = output["display_guidance"]["novice"]["parts"][0]["steps"]

        visible = "".join(steps)
        self.assertNotIn(r"\frac{3}{5}", visible)
        self.assertNotIn("概率为1", visible)
        self.assertIn("该事件的概率", steps[0])

    def test_choice_verdicts_and_spoken_ratios_are_redacted(self) -> None:
        sample = compact_sample()
        sample["l5"][0]["s"] = [
            "因此h3确实代表本次下潜的最大深度，A正确。",
            "比较可知最大加速度更大，B错误。",
            "所以C选项说加速度不变是错的。",
            "加速度向上对应斜率为正的段落，D正确。",
            "用符合条件的结果数除以总结果数，即1除以8，得到最终的概率值。；pit: 不要数错份数。",
        ]

        output, _ = expand_compact_output(sample, subject="mathematics")
        steps = output["display_guidance"]["novice"]["parts"][0]["steps"]
        visible = "".join(steps)

        self.assertNotIn("A正确", visible)
        self.assertNotIn("B错误", visible)
        self.assertNotIn("D正确", visible)
        self.assertNotIn("是错的", visible)
        self.assertNotIn("1除以8", visible)
        self.assertNotIn("pit:", visible)
        self.assertIn("h3确实代表本次下潜的最大深度", visible)
        self.assertIn("用符合条件的结果数除以总结果数", visible)

    def test_extremum_location_is_redacted_even_when_marked_provisional(self) -> None:
        sample = compact_sample()
        sample["l5"][0]["s"][4] = (
            "函数在定义域内先减后增。暂定$x_0=2$时最小。"
        )

        output, _ = expand_compact_output(sample, subject="mathematics")
        visible = "".join(
            output["display_guidance"]["novice"]["parts"][0]["steps"]
        )

        self.assertNotIn("x_0=2", visible)
        self.assertNotIn("暂定", visible)
        self.assertIn("函数在定义域内先减后增", visible)

    def test_solved_parameter_and_standard_equation_are_not_rendered(self) -> None:
        sample = compact_sample()
        sample["l4"][0]["s"] = [
            "由离心率公式建立参数关系。",
            "解这个方程可以得到$a^2=4$，所以$a=2$符合条件。",
            "将参数代回，标准方程为$\\frac{x^2}{4}+y^2=1$。",
        ]

        output, _ = expand_compact_output(sample, subject="mathematics")
        visible = "".join(
            output["display_guidance"]["lower"]["parts"][0]["steps"]
        )

        self.assertNotIn("a^2=4", visible)
        self.assertNotIn(r"\frac{x^2}{4}+y^2=1", visible)
        self.assertIn("保留最后求值", visible)

    def test_final_check_does_not_repeat_the_solved_assignment(self) -> None:
        sample = compact_sample()
        sample["l5"][0]["s"][4] = (
            "最后检查a²=4是否满足全部条件，再核对焦点位置。"
        )

        output, _ = expand_compact_output(sample, subject="mathematics")
        visible = "".join(
            output["display_guidance"]["novice"]["parts"][0]["steps"]
        )

        self.assertNotIn("a²=4", visible)
        self.assertIn("检查所得参数是否满足全部条件", visible)

    def test_fill_in_count_assembly_is_removed_from_student_pages(self) -> None:
        sample = compact_sample()
        sample["l4"][0]["s"][1] = (
            "因为红色区域只有1个扇形，所以目标事件数为1，总事件数为8。"
        )
        sample["l5"][0]["s"][3] = (
            "总共有8个扇形，所以所有可能的结果总数是8，符合条件的结果数是1。"
        )

        output, _ = expand_compact_output(sample, subject="mathematics")
        lower = "".join(output["display_guidance"]["lower"]["parts"][0]["steps"])
        novice = "".join(output["display_guidance"]["novice"]["parts"][0]["steps"])

        self.assertNotIn("目标事件数为1", lower)
        self.assertNotIn("总事件数为8", lower)
        self.assertNotIn("结果总数是8", novice)
        self.assertNotIn("符合条件的结果数是1", novice)
        self.assertIn("红色区域只有1个扇形", lower)

    def test_minor_sentence_and_label_variation_keeps_model_prose(self) -> None:
        sample = compact_sample()
        sample["l1"][0]["n"] = "问题1"
        sample["l1"][0]["s"][1] = "还有哪个条件没有在前一步使用？"
        sample["l3"][0]["n"] = "第一问"
        sample["l4"][0]["n"] = "(1)"
        sample["l4"][0]["s"][2] = "最后只需保留待求量与已知量的关系"
        sample["l5"][0]["s"].pop()

        output, audit = expand_compact_output(sample, subject="mathematics")

        self.assertTrue(audit["model_direct_render"])
        self.assertEqual(
            output["display_guidance"]["elite"]["parts"][0]["steps"],
            sample["l1"][0]["s"],
        )
        for tier in ("elite", "middle", "lower", "novice"):
            self.assertEqual(
                output["display_guidance"][tier]["parts"][0]["part"],
                "整题",
            )

    def test_short_parseable_tiers_are_locally_completed_without_replacement(self) -> None:
        sample = compact_sample()
        original_l4 = "先利用模型返回的关键关系整理这一问"
        original_l5 = [
            "先读取模型识别到的题设条件",
            "再建立模型给出的中间关系",
        ]
        sample["l4"][0]["s"] = [original_l4]
        sample["l5"][0]["s"] = list(original_l5)
        sample["l5"][0].pop("pit")

        output, audit = expand_compact_output(
            sample,
            subject="mathematics",
            curriculum_stage="senior_high",
        )

        lower_steps = output["display_guidance"]["lower"]["parts"][0]["steps"]
        novice = output["display_guidance"]["novice"]["parts"][0]
        self.assertEqual(lower_steps[0], original_l4)
        self.assertEqual(novice["steps"][:2], original_l5)
        self.assertGreaterEqual(len(lower_steps), 3)
        self.assertGreaterEqual(len(novice["steps"]), 5)
        self.assertTrue(novice["pitfall"])
        self.assertTrue(audit["model_direct_render"])
        self.assertTrue(audit["model_content_preserved"])
        self.assertTrue(audit["local_repair_applied"])
        self.assertFalse(audit.get("time_budget_fallback", False))

    def test_misaligned_tier_parts_reuse_model_prose_per_part(self) -> None:
        sample = compact_sample()
        for field in ("l1", "l3", "l4"):
            base = sample[field][0]
            sample[field] = [
                {**base, "n": "（1）"},
                {**base, "n": "（2）", "s": [f"{field}模型第二问分析"]},
            ]
        sample["l5"][0]["n"] = "（1）"

        output, audit = expand_compact_output(
            sample,
            subject="mathematics",
            curriculum_stage="senior_high",
            part_labels=["（1）", "（2）"],
        )

        novice_parts = output["display_guidance"]["novice"]["parts"]
        self.assertEqual([part["part"] for part in novice_parts], ["（1）", "（2）"])
        self.assertIn("模型第二问分析", "".join(novice_parts[1]["steps"]))
        self.assertTrue(audit["model_direct_render"])
        self.assertTrue(any(
            item["action"] == "reuse_model_prose_for_missing_tier_part"
            for item in audit["local_repair_actions"]
        ))

    def test_repeated_model_part_cycle_is_rendered_once(self) -> None:
        sample = compact_sample()
        labels = ["A", "B", "C", "D"]
        for field in ("l1", "l3", "l4", "l5"):
            base = sample[field][0]
            cycle = [
                {**base, "n": label, "s": [f"{field}模型对{label}的分析"]}
                for label in labels
            ]
            sample[field] = cycle * 3

        output, audit = expand_compact_output(
            sample,
            subject="physics",
            curriculum_stage="senior_high",
        )

        for tier in ("elite", "middle", "lower", "novice"):
            self.assertEqual(
                [part["part"] for part in output["display_guidance"][tier]["parts"]],
                labels,
            )
        self.assertTrue(any(
            item["action"] == "deduplicate_repeated_part_labels"
            for item in audit["local_repair_actions"]
        ))

    def test_local_fallback_preserves_completed_provider_prefix(self) -> None:
        prefix = {key: compact_sample()[key] for key in COMPACT_PREFIX_KEYS}
        fallback = local_fallback_compact(
            subject="mathematics", part_labels=["整题"], partial=prefix
        )

        for key in COMPACT_PREFIX_KEYS:
            self.assertEqual(fallback[key], prefix[key])
        self.assertEqual(fallback["s"], "c")
        self.assertEqual(fallback["q"], [])

    def test_local_fallback_adds_missing_detected_part(self) -> None:
        sample = compact_sample()
        for field in ("l1", "l3", "l4", "l5"):
            sample[field][0]["n"] = "（1）"
        prefix = {key: sample[key] for key in COMPACT_PREFIX_KEYS}
        fallback = local_fallback_compact(
            subject="mathematics", part_labels=["（1）", "（2）"], partial=prefix
        )

        for field in ("l1", "l3", "l4", "l5"):
            self.assertEqual(
                [part["n"] for part in fallback[field]], ["（1）", "（2）"]
            )

        empty_fallback = local_fallback_compact(
            subject="mathematics", part_labels=["整题"]
        )
        output, _ = expand_compact_output(empty_fallback, subject="mathematics")
        self.assertEqual(output["base_formulas"], [])
        self.assertFalse(output["display_guidance"]["elite"]["parts"] == [])

    def test_empty_text_block_is_removed_and_alternative_ids_are_canonical(self) -> None:
        sample = legacy_compact_sample()
        sample["q"].append({"t": "t", "x": "", "f": ""})
        primary = sample["p"][0]["m"][0]
        alternative = dict(primary)
        alternative["n"] = "定义比较法"
        sample["p"][0]["m"].append(alternative)

        output, _ = expand_compact_output(sample, subject="mathematics")

        self.assertEqual(len(output["question_blocks"]), 1)
        for tier in output["guidance"].values():
            self.assertEqual(
                tier["parts"][0]["alternative_methods"][0]["method_id"],
                "M2",
            )


class RoutingAndImagePolicyTests(unittest.TestCase):
    def test_simple_problem_uses_fast_route(self) -> None:
        result = classify_problem_complexity(
            "下列各数中绝对值最大的是哪一个", illustration_count=0
        )
        self.assertEqual(result["tier"], "simple")

    def test_proof_multi_part_and_geometry_figure_use_complex_route(self) -> None:
        self.assertEqual(
            classify_problem_complexity("证明该结论成立", illustration_count=0)["tier"],
            "complex",
        )
        self.assertEqual(
            classify_problem_complexity(
                "（1）求值（2）化简（3）说明理由", illustration_count=0
            )["tier"],
            "complex",
        )
        self.assertEqual(
            classify_problem_complexity("如图，在圆中作切线", illustration_count=1)["tier"],
            "complex",
        )
        self.assertEqual(
            classify_problem_complexity(
                "如图甲所示深度曲线，如图乙所示速度图像",
                illustration_count=2,
            )["tier"],
            "complex",
        )

    def test_pie_chart_probability_stays_on_the_fast_route(self) -> None:
        stem = (
            "将一个可以自由转动的圆形转盘分成8个面积相等的扇形"
            "(1个红色、3个黄色和4个绿色)，指针指红色区域的概率为____"
        )
        result = classify_problem_complexity(stem, illustration_count=2)
        self.assertEqual(result["tier"], "simple")
        self.assertNotIn("geometry_with_figure", result["reasons"])
        self.assertNotIn("multiple_figures", result["reasons"])

    def test_high_ocr_with_figures_sends_crops_without_full_page(self) -> None:
        base = adaptive_image_policy(
            ocr_character_count=80,
            printed_text_count=5,
            illustration_count=2,
        )
        policy = resolve_provider_image_policy(
            base,
            complexity_tier="simple",
            illustration_count=2,
        )
        self.assertFalse(policy["send_enhanced_original"])
        self.assertFalse(policy["send_semantic_reference"])
        self.assertTrue(policy["send_illustration_crops"])
        self.assertEqual(
            policy["reason"],
            "ocr_text_plus_figure_crops",
        )
        complex_policy = resolve_provider_image_policy(
            base,
            complexity_tier="complex",
            illustration_count=2,
        )
        self.assertTrue(complex_policy["send_illustration_crops"])

    def test_high_quality_text_only_request_has_no_images(self) -> None:
        base = adaptive_image_policy(
            ocr_character_count=160,
            printed_text_count=8,
            illustration_count=0,
        )
        policy = resolve_provider_image_policy(
            base,
            complexity_tier="complex",
            illustration_count=0,
        )

        self.assertFalse(policy["send_semantic_reference"])
        self.assertFalse(policy["send_illustration_crops"])
        self.assertEqual(policy["reason"], "ocr_text_only")

    def test_low_ocr_keeps_complete_page_as_safety_fallback(self) -> None:
        base = adaptive_image_policy(
            ocr_character_count=8,
            printed_text_count=1,
            illustration_count=0,
        )
        policy = resolve_provider_image_policy(
            base,
            complexity_tier="complex",
            illustration_count=0,
        )

        self.assertTrue(policy["send_semantic_reference"])
        self.assertEqual(policy["reason"], "low_ocr_plus_semantic_reference")

    def test_formula_integrity_risk_overrides_high_ocr_text_only_route(self) -> None:
        base = adaptive_image_policy(
            ocr_character_count=160,
            printed_text_count=8,
            illustration_count=0,
        )
        policy = resolve_provider_image_policy(
            base,
            complexity_tier="complex",
            illustration_count=0,
            formula_integrity_risk=True,
        )

        self.assertTrue(policy["send_semantic_reference"])
        self.assertTrue(policy["formula_integrity_risk"])
        self.assertEqual(policy["reason"], "formula_safety_fallback")

    def test_high_quality_text_sends_main_image_only(self) -> None:
        policy = adaptive_image_policy(
            ocr_character_count=100,
            printed_text_count=5,
            illustration_count=0,
        )
        self.assertFalse(policy["send_enhanced_original"])
        self.assertFalse(policy["send_illustration_crops"])

    def test_low_ocr_or_figure_enables_extra_views(self) -> None:
        low = adaptive_image_policy(
            ocr_character_count=8,
            printed_text_count=1,
            illustration_count=0,
        )
        figure = adaptive_image_policy(
            ocr_character_count=100,
            printed_text_count=5,
            illustration_count=1,
        )
        self.assertTrue(low["send_enhanced_original"])
        self.assertFalse(low["send_illustration_crops"])
        self.assertTrue(figure["send_enhanced_original"])
        self.assertTrue(figure["send_illustration_crops"])


class CompactContractBudgetTests(unittest.TestCase):
    def test_prompt_and_schema_stay_compact(self) -> None:
        prompt = (SOURCE_DIR / "数学题解析紧凑提示词.md").read_text(
            encoding="utf-8"
        )
        schema = json.loads(
            (SOURCE_DIR / "compact_guidance_schema.json").read_text(
                encoding="utf-8"
            )
        )
        schema_bytes = len(
            json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )

        self.assertLess(len(prompt), 3000)
        self.assertLess(schema_bytes, 5000)
        self.assertEqual(
            tuple(schema["required"][: len(COMPACT_PREFIX_KEYS)]),
            COMPACT_PREFIX_KEYS,
        )
        self.assertEqual(schema["$id"], "test6_compact_five_natural_tiers_math_v4")
        self.assertNotIn("method", schema["$defs"])
        self.assertEqual(schema["$defs"]["l5Part"]["required"], ["n", "s", "pit"])
        self.assertNotIn("q", schema["properties"])
        self.assertNotIn("g", schema["properties"])
        self.assertNotIn("e", schema["properties"])


if __name__ == "__main__":
    unittest.main()
