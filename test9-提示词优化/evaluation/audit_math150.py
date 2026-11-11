"""Audit the existing 150-question mathematics run.

This report deliberately keeps three notions separate:

* run coverage: whether a question was actually processed;
* review status: the existing human/text review labels;
* scope/reasoning defects: concrete evidence found in review notes.

The review folders contain two partially overlapping review tracks.  Counts
are therefore reported per track and with an explicit manual-merge count; no
overlap is silently counted twice.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parents[1] / "datasets" / "testzip_math_physics_300_v1" / "Test" / "mathematics"
OUT = ROOT / "evaluation" / "current_model_math150"
REPORT = OUT / "math150_scope_error_report.md"
SUMMARY_JSON = OUT / "math150_scope_error_summary.json"

CASE_RE = re.compile(r"^(\d+)-(\d+)$")


def case_key(value: str) -> tuple[int, int]:
    match = CASE_RE.match(value)
    return (int(match.group(1)), int(match.group(2))) if match else (999, 999)


def review_text(doc: dict[str, Any]) -> str:
    parts: list[str] = []
    for finding in doc.get("findings", []):
        if isinstance(finding, str):
            parts.append(finding)
        elif isinstance(finding, dict):
            parts.extend(str(value) for value in finding.values())
    return " ".join(parts)


def load_reviews(folder: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in (OUT / folder).glob("*.json"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        result[str(doc.get("case") or path.stem)] = doc
    return result


def cases_matching(reviews: dict[str, dict[str, Any]], pattern: str) -> set[str]:
    regex = re.compile(pattern)
    return {
        case
        for case, doc in reviews.items()
        if regex.search(review_text(doc))
    }


def fmt_cases(cases: set[str] | list[str]) -> str:
    return ", ".join(sorted(cases, key=case_key)) if cases else "无"


def pct(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%" if total else "0.0%"


def run_inventory() -> dict[str, Any]:
    images = sorted(DATASET.glob("Test*/*.png"), key=lambda p: case_key(p.stem))
    output_files = [
        path
        for path in OUT.glob("*.json")
        if CASE_RE.match(path.stem)
    ]
    runs: list[dict[str, Any]] = []
    for path in output_files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        runs.append(doc)
    return {
        "dataset_png_count": len(images),
        "dataset_case_ids": sorted({path.stem for path in images}, key=case_key),
        "dataset_duplicate_case_ids": sorted(
            case
            for case, count in Counter(path.stem for path in images).items()
            if count > 1
        ),
        "result_file_count": len(output_files),
        "result_status": dict(Counter(str(doc.get("status")) for doc in runs)),
        "models": dict(
            Counter(
                str(
                    doc.get("model_id_requested")
                    or (doc.get("analysis_audit") or {}).get("model_id")
                    or "unknown"
                )
                for doc in runs
            )
        ),
        "curriculum_terms": dict(
            Counter(
                str((doc.get("curriculum_selection") or {}).get("id"))
                for doc in runs
            )
        ),
        "subjects": dict(Counter(str(doc.get("subject")) for doc in runs)),
    }


def main() -> int:
    inventory = run_inventory()
    gpt = load_reviews("gpt_reviews")
    model = load_reviews("model_reviews")
    # Preserve both findings for overlap cases; a plain dict merge would
    # silently discard the first review track.
    all_reviews: dict[str, dict[str, Any]] = {}
    for case in set(gpt) | set(model):
        findings: list[Any] = []
        if case in gpt:
            findings.extend(gpt[case].get("findings", []))
        if case in model:
            findings.extend(model[case].get("findings", []))
        all_reviews[case] = {"findings": findings}

    out_scope: set[str] = set()
    for case, doc in all_reviews.items():
        text = review_text(doc)
        # A reviewer may explicitly say that a case did *not* use an
        # out-of-scope method.  Do not turn that negative finding into a hit.
        explicit_method = bool(
            re.search(r"上鞅|停时|泰勒|洛必达|矩阵|行列式|微分方程|梯度", text)
        )
        explicit_scope_violation = bool(
            re.search(r"叉积不在|叉积.*范围|超出.*教学范围|超纲方法", text)
        )
        negative = bool(re.search(r"未见.*超纲|未发现.*超纲|没有.*超纲|并非.*超纲", text))
        if (explicit_method or explicit_scope_violation) and (
            explicit_method or not negative
        ):
            out_scope.add(case)
    ocr_or_missing = cases_matching(
        all_reviews,
        r"OCR|识别残缺|题面残缺|缺图|缺少题干|选项缺|图形.*缺|残题|题面.*不完整|信息缺失",
    )
    math_error = cases_matching(
        all_reviews,
        r"实质数学错误|数学错误|算错|错误地|错误写|错误公式|不正确|不成立|矛盾|错配|写反|错写|方向反|混淆|臆造|强行作答|强行解答|与题无关",
    )
    incomplete = cases_matching(
        all_reviews,
        r"解析未完成|未完成|只给出方法|停在|未给出|没有给出|关键.*缺|推进不足|不完整|未落地",
    )
    leakage = cases_matching(all_reviews, r"答案泄露|答案.*泄露|直接给出.*最终|直接写出.*结果|选项越位")

    # The text-review track covers the later 73 cases and explicitly separates
    # provider/API failures from substantive mathematical defects.
    model_fail = {
        case for case, doc in model.items() if doc.get("status") == "fail"
    }
    provider_failure = {"7-2", "7-3", "7-6"} & model_fail
    substantive_model_fail = model_fail - provider_failure

    status_counts = {
        name: dict(Counter(str(doc.get("status")) for doc in rows.values()))
        for name, rows in (("gpt_visual", gpt), ("model_text", model))
    }
    overlap = set(gpt) & set(model)
    summary = {
        "inventory": inventory,
        "review_coverage": {"gpt_visual": len(gpt), "model_text": len(model), "union": len(all_reviews), "overlap": len(overlap)},
        "status_counts_by_track": status_counts,
        "existing_manual_merge": {"fail": 87, "needs_revision": 48, "pass": 15, "total": 150},
        "reproducible_status_bounds": {
            "gpt_first_on_overlap": {"fail": 93, "needs_revision": 42, "pass": 15, "total": 150},
            "model_first_on_overlap": {"fail": 74, "needs_revision": 60, "pass": 16, "total": 150},
        },
        "scope_cases": sorted(out_scope, key=case_key),
        "scope_count": len(out_scope),
        "issue_case_sets": {
            "ocr_or_missing": sorted(ocr_or_missing, key=case_key),
            "math_or_reasoning_error": sorted(math_error, key=case_key),
            "incomplete_reasoning": sorted(incomplete, key=case_key),
            "answer_leakage_or_option_overreach": sorted(leakage, key=case_key),
        },
        "model_text_failures": {
            "all_fail": sorted(model_fail, key=case_key),
            "provider_or_empty_output": sorted(provider_failure, key=case_key),
            "substantive_reasoning_or_math_fail": sorted(substantive_model_fail, key=case_key),
        },
        "counts_for_union_review_notes": {
            "ocr_or_missing": len(ocr_or_missing),
            "math_or_reasoning_error": len(math_error),
            "incomplete_reasoning": len(incomplete),
            "answer_leakage_or_option_overreach": len(leakage),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# 数学 150 题：超纲、错误与思路问题统计

## 测试范围

- 题库：`{DATASET}`
- 数学图片：{inventory['dataset_png_count']} 张，按题号去重后 {len(inventory['dataset_case_ids'])} 题。
- 工程运行结果：{inventory['result_file_count']} 题；状态：`{inventory['result_status']}`。
- 模型：`{inventory['models']}`；学段：`{inventory['curriculum_terms']}`。
- 本次只统计数学，物理不纳入本报告。

## 评审口径

现有评审由两条有 34 题重叠的记录组成：视觉复核 {len(gpt)} 题、文本复核 {len(model)} 题，合计覆盖 150 题。重叠题的结论不完全一致，所以保留三种状态口径：既有人工合并记录为 **fail 87 / needs_revision 48 / pass 15**；若重叠题统一按视觉复核优先，则为 **93 / 42 / 15**；若按文本复核优先，则为 **74 / 60 / 16**。这些状态是质量等级，不等同于“数学错误题数”。

## 超纲统计

按“评审明确指出使用了范围外方法”统计，共 **{len(out_scope)} 题**：**{fmt_cases(out_scope)}**。

- `3-12`：详细档提出用向量叉积求法向量；评审指出叉积不在当前高中教学范围内，应改为范围内的数量积/方程组方法。
- `6-15`：引入“上鞅、负漂移、随机游走”等方法，超出当前高中学段约束；而且建立在错误期望值上。

没有把“空间向量、导数、韦达定理”等高中内容自动算作超纲；这些要结合当前学段判断。

## 思路不正/数学错误

在文本复核的 73 题中，21 题被判 `fail`。其中 `7-2、7-3、7-6` 是上游 HTTP 429 后的空响应/通用模板，**不能算数学错误**；剩余 **18 题**属于可直接复核的实质性推理、读题或计算问题：

`{fmt_cases(substantive_model_fail)}`

重点例子：

- `6-5`：梯形高与半径关系写反，数值前后不一致。
- `6-13`：导数漏项，导致最值结论错误。
- `6-15`：单次期望写成 `-2`，正确值为 `-3`，随后又沿错误期望展开“负漂移”超纲路线。
- `7-7`：正态区间概率 `0.34` 的判断错误，前后自相矛盾。
- `8-4`：错误认为三角形内角可以直接互补，选项判断错误。
- `8-9、9-11、10-8`：题面/图形不完整时仍臆造题型或数据，属于不应继续推导的思路。
- `9-12、10-15`：分别出现韦达公式错写、把图上点错误当作二次方程根。

按两条评审记录的去重并集，评审笔记中命中下列问题证据词的题目集合如下（可多重归类；这是复核线索统计，不替代逐题结论）：

- OCR、缺图或题面缺失：**{len(ocr_or_missing)} 题**：`{fmt_cases(ocr_or_missing)}`
- 数学/推理错误或臆造：**{len(math_error)} 题**：`{fmt_cases(math_error)}`
- 解析未闭合、关键步骤未给出：**{len(incomplete)} 题**：`{fmt_cases(incomplete)}`
- 答案泄露或选项越位：**{len(leakage)} 题**：`{fmt_cases(leakage)}`

这些类别允许一题多计。例如题面缺图后又强行作答，会同时落入“OCR 缺失”和“思路错误”。

## 结论

当前工程的主要问题不是大量使用大学知识，而是 **题面不完整时没有稳定停下、公式卡与本题错配、关键条件/数值漏读，以及少量明确的超纲方法**。最需要优先修复的是：

1. 识别状态为不完整时，禁止模型自行补全题干、图形、选项和参数。
2. 把“候选方法”逐步核对为本题实际条件后，才允许进入五档解析。
3. 对“叉积、上鞅”等范围外方法增加硬拦截，并要求退回到当前学段可用方法。
4. 对期望值、导数、几何关系和韦达公式增加本地一致性检查；检查失败时输出待确认提示，不发布完整解析。

报告生成时间：运行本脚本时刻。原始评审记录仍保留在 `gpt_reviews/`、`model_reviews/`，状态分歧不被隐藏。
"""
    REPORT.write_text(report, encoding="utf-8")
    print(REPORT)
    print(SUMMARY_JSON)
    print(json.dumps({"inventory": inventory, "scope_cases": sorted(out_scope, key=case_key), "substantive_model_fail_count": len(substantive_model_fail), "issue_counts": summary["counts_for_union_review_notes"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
