#!/usr/bin/env python3
"""Re-render the established 20-question mathematics set without model calls.

The script intentionally consumes historical, already-recorded ``output_json``
objects.  It exercises the current Test4 normalizer and renderer only, so a
layout iteration is deterministic, fast, and has no provider cost.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import html
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(__file__).resolve().parent / "source"
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_RESULTS = WORKSPACE_ROOT / "jiexi" / "math1" / "test1" / "results.jsonl"
DEFAULT_DATASET = WORKSPACE_ROOT / "jiexi" / "math1" / "dataset"
LEGACY_PREPARE_PATH = WORKSPACE_ROOT / "jiexi" / "math1" / "rerender_test1_phy.py"
DEFAULT_OUTPUT_PARENT = PROJECT_ROOT / "deployment_acceptance"

sys.path.insert(0, str(SOURCE_ROOT))

import math_guidance_pilot as pilot  # noqa: E402
from device_result_renderer import (  # noqa: E402
    DEVICE_IMAGE_KINDS,
    DEVICE_IMAGE_NAMES,
    resolve_device_fonts,
    validate_device_image_set,
)
from math_guidance_renderer import render_four_images  # noqa: E402


RAW_MATH_PATTERNS = (
    re.compile(r"[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9]"),
    re.compile(r"[A-Za-z][A-Za-z0-9]*_\{"),
    re.compile(r"\|\|"),
    re.compile(r"(?<!\\)(?:\+|-)?(?:infty|infinity)\b", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?:\([^)]*\))?\s*[=<>≠≤≥]\s*"),
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9_]*\s*/\s*[A-Za-z0-9_(]"),
)

MALFORMED_MATH_PATTERNS = (
    re.compile(r"(?<!\\)(?:\+|-)?(?:infty|infinity)\b", re.IGNORECASE),
    re.compile(r"\\frac\{[^{}]+\}\{[^{}]*[一-鿿][^{}]*\}"),
)


def latest_success_records(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "success" and isinstance(
                record.get("output_json"), Mapping
            ):
                latest[str(record["question_id"])] = record
    return dict(sorted(latest.items()))


def load_legacy_prepare_output():
    """Load the frozen dataset compatibility shim without running its CLI."""

    spec = importlib.util.spec_from_file_location(
        "test4_math1_legacy_prepare", LEGACY_PREPARE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load compatibility shim: {LEGACY_PREPARE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prepare_output


def displayed_strings(output: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield only prose fields that the six-image renderer actually displays."""

    for index, block in enumerate(output.get("question_blocks", [])):
        if isinstance(block, Mapping) and isinstance(block.get("text"), str):
            yield f"$.question_blocks[{index}].text", block["text"]
    for index, value in enumerate(output.get("knowledge_points", [])):
        if isinstance(value, str):
            yield f"$.knowledge_points[{index}]", value
    for index, item in enumerate(output.get("base_formulas", [])):
        if isinstance(item, Mapping) and isinstance(item.get("condition"), str):
            yield f"$.base_formulas[{index}].condition", item["condition"]
    guidance = output.get("guidance", {})
    if not isinstance(guidance, Mapping):
        return
    for tier_key, tier in guidance.items():
        if not isinstance(tier, Mapping):
            continue
        for part_index, part in enumerate(tier.get("parts", [])):
            if not isinstance(part, Mapping):
                continue
            base = f"$.guidance.{tier_key}.parts[{part_index}]"
            for step_index, step in enumerate(part.get("steps", [])):
                if isinstance(step, str):
                    yield f"{base}.steps[{step_index}]", step
            for field in ("pitfall", "pattern_summary"):
                if isinstance(part.get(field), str):
                    yield f"{base}.{field}", part[field]
            for method_index, method in enumerate(part.get("alternative_methods", [])):
                if not isinstance(method, Mapping):
                    continue
                for step_index, step in enumerate(method.get("steps", [])):
                    if isinstance(step, str):
                        yield (
                            f"{base}.alternative_methods[{method_index}]"
                            f".steps[{step_index}]",
                            step,
                        )


def displayed_formula_sources(
    output: Mapping[str, Any],
) -> Iterable[tuple[str, str]]:
    for index, item in enumerate(output.get("base_formulas", [])):
        if isinstance(item, Mapping) and isinstance(item.get("formula"), str):
            yield f"$.base_formulas[{index}].formula", item["formula"]


def raw_math_issues(output: Mapping[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    for path, value in displayed_strings(output):
        pieces = value.split("$")
        for plain in pieces[::2]:
            for pattern in RAW_MATH_PATTERNS:
                match = pattern.search(plain)
                if match is None:
                    continue
                issues.append(
                    {
                        "path": path,
                        "match": match.group(0),
                        "text": value,
                    }
                )
                break
        for formula in pieces[1::2]:
            for pattern in MALFORMED_MATH_PATTERNS:
                match = pattern.search(formula)
                if match is None:
                    continue
                issues.append(
                    {
                        "path": path,
                        "match": match.group(0),
                        "text": value,
                    }
                )
                break
    for path, value in displayed_formula_sources(output):
        if "$" in value:
            issues.append({"path": path, "match": "$", "text": value})
            continue
        for pattern in MALFORMED_MATH_PATTERNS:
            match = pattern.search(value)
            if match is None:
                continue
            issues.append(
                {"path": path, "match": match.group(0), "text": value}
            )
            break
    return issues


def image_edge_ink(path: Path, *, threshold: int = 245) -> dict[str, int]:
    """Count dark pixels on the outermost row/column of a rendered page."""

    with Image.open(path) as source:
        image = source.convert("L")
        return {
            "left": sum(image.getpixel((0, y)) < threshold for y in range(image.height)),
            "right": sum(
                image.getpixel((image.width - 1, y)) < threshold
                for y in range(image.height)
            ),
            "top": sum(image.getpixel((x, 0)) < threshold for x in range(image.width)),
            "bottom": sum(
                image.getpixel((x, image.height - 1)) < threshold
                for x in range(image.width)
            ),
        }


def remove_legacy_placeholder_steps(
    output: dict[str, Any], audit: list[dict[str, Any]]
) -> None:
    """Drop punctuation-only artifacts from frozen historical fixtures.

    Production parsing already rejects these values.  The frozen 2026-era
    records predate that guard, so the offline layout lane removes only content
    that has no letters, digits or meaningful mathematical notation.  It never
    writes or invents a replacement step.
    """

    guidance = output.get("guidance", {})
    if not isinstance(guidance, Mapping):
        return
    for tier_key, tier in guidance.items():
        if not isinstance(tier, Mapping):
            continue
        for part_index, part in enumerate(tier.get("parts", [])):
            if not isinstance(part, dict):
                continue
            methods: list[tuple[str, dict[str, Any]]] = [("method[0]", part)]
            for method_index, method in enumerate(part.get("alternative_methods", [])):
                if isinstance(method, dict):
                    methods.append((f"alternative_methods[{method_index}]", method))
            for method_path, method in methods:
                before = method.get("steps", [])
                if not isinstance(before, list):
                    continue
                after = [
                    step for step in before
                    if isinstance(step, str)
                    and pilot.has_substantive_guidance_content(step)
                ]
                if after == before:
                    continue
                method["steps"] = after
                audit.append({
                    "path": (
                        f"$.guidance.{tier_key}.parts[{part_index}].{method_path}.steps"
                    ),
                    "operation": "fixture_compatibility_cleanup",
                    "rule_id": "remove_legacy_punctuation_only_steps",
                    "removed_count": len(before) - len(after),
                })


def make_contact_sheet(
    cases: list[dict[str, Any]],
    *,
    filename: str,
    destination: Path,
    columns: int = 4,
    thumb_width: int = 250,
) -> None:
    entries: list[tuple[str, Image.Image]] = []
    for case in cases:
        path = Path(case["directory"]) / filename
        with Image.open(path) as source:
            image = source.convert("L")
            target_height = max(1, round(image.height * thumb_width / image.width))
            entries.append(
                (
                    str(case["question_id"]),
                    image.resize((thumb_width, target_height), Image.Resampling.LANCZOS),
                )
            )
    if not entries:
        return
    label_height = 30
    gap = 14
    rows: list[list[tuple[str, Image.Image]]] = [
        entries[index:index + columns]
        for index in range(0, len(entries), columns)
    ]
    row_heights = [
        label_height + max(item.height for _, item in row)
        for row in rows
    ]
    width = gap + columns * (thumb_width + gap)
    height = gap + sum(item + gap for item in row_heights)
    sheet = Image.new("L", (width, height), 245)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    y = gap
    for row, row_height in zip(rows, row_heights):
        for column, (label, image) in enumerate(row):
            x = gap + column * (thumb_width + gap)
            draw.rectangle(
                (x - 1, y - 1, x + thumb_width, y + row_height - 1),
                fill=255,
                outline=190,
            )
            draw.text((x + 6, y + 7), label, font=font, fill=0)
            sheet.paste(image, (x, y + label_height))
        y += row_height + gap
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="JPEG", quality=88, optimize=True)


def write_gallery(output_dir: Path, cases: list[dict[str, Any]]) -> None:
    sections: list[str] = []
    for case in cases:
        relative = Path(case["directory"]).relative_to(output_dir).as_posix()
        images = "".join(
            f'<figure><figcaption>{html.escape(kind)}</figcaption>'
            f'<a href="{relative}/{filename}"><img loading="lazy" '
            f'src="{relative}/{filename}" alt="{html.escape(kind)}"></a></figure>'
            for kind, filename in zip(DEVICE_IMAGE_KINDS, DEVICE_IMAGE_NAMES)
        )
        sections.append(
            f'<section><h2>{html.escape(str(case["question_id"]))}</h2>'
            f'<p>contract={str(case["contract_valid"]).lower()} · '
            f'raw_math_issues={case["raw_math_issue_count"]} · '
            f'normalization_rules={case["normalization_rule_count"]}</p>'
            f'<div class="grid">{images}</div></section>'
        )
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Test4 数学本地排版数据集</title>
<style>
body{margin:0;background:#eef2f6;color:#17202a;font-family:Arial,"Microsoft YaHei",sans-serif}
header{position:sticky;top:0;z-index:2;padding:18px 24px;background:#fff;border-bottom:1px solid #ccd6df}
main{max-width:1500px;margin:auto;padding:20px}section{background:#fff;margin:0 0 24px;padding:18px;border-radius:14px}
h1,h2,p{margin:0 0 12px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}
figure{margin:0}figcaption{font-weight:700;margin-bottom:8px}img{display:block;width:100%;height:auto;border:1px solid #ccd6df;background:#fff}
@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}}@media(max-width:520px){.grid{grid-template-columns:1fr}}
</style></head><body><header><h1>Test4 数学本地排版数据集</h1>
<p>历史模型输出离线重渲染；未调用模型。</p></header><main>""" + "".join(sections) + "</main></body></html>"
    (output_dir / "index.html").write_text(document, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir = DEFAULT_OUTPUT_PARENT / f"local_math_layout_{stamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    records = latest_success_records(args.results_jsonl.resolve())
    if args.limit is not None:
        records = dict(list(records.items())[: max(0, args.limit)])
    if not records:
        raise SystemExit("No successful historical mathematics records were found.")

    song, song_bold, times, times_bold = resolve_device_fonts()
    prepare_output = load_legacy_prepare_output()
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for question_id, record in records.items():
        source_image = args.dataset_root.resolve() / record["source_image"]["relative_path"]
        case_dir = output_dir / "cases" / question_id
        case_dir.mkdir(parents=True)
        normalization_audit: list[dict[str, Any]] = []
        compatible = prepare_output(question_id, deepcopy(record["output_json"]))
        compatible["problem_scope"] = "primary_to_high_school"
        remove_legacy_placeholder_steps(compatible, normalization_audit)
        normalized = pilot.normalize_output_for_render(compatible, normalization_audit)
        try:
            render = render_four_images(
                question_id=question_id,
                model_id=str(record.get("model_id_response") or "historical-output"),
                source_image=source_image,
                output=normalized,
                output_dir=case_dir,
                song_path=song,
                song_bold_path=song_bold,
                times_path=times,
                times_bold_path=times_bold,
            )
            contract = validate_device_image_set(case_dir)
            issues = raw_math_issues(normalized)
            edge_ink = {
                filename: image_edge_ink(case_dir / filename)
                for filename in DEVICE_IMAGE_NAMES
            }
            edge_ink_total = sum(
                count
                for sides in edge_ink.values()
                for count in sides.values()
            )
            (case_dir / "normalized_output.json").write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            (case_dir / "render.json").write_text(
                json.dumps(render, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            (case_dir / "normalization_audit.json").write_text(
                json.dumps(normalization_audit, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            cases.append(
                {
                    "question_id": question_id,
                    "directory": str(case_dir),
                    "source_image": str(source_image),
                    "contract_valid": bool(contract["valid"]),
                    "contract_errors": contract["errors"],
                    "raw_math_issue_count": len(issues),
                    "raw_math_issues": issues,
                    "edge_ink_total": edge_ink_total,
                    "edge_ink": edge_ink,
                    "normalization_rule_count": len(normalization_audit),
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "question_id": question_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    for filename in DEVICE_IMAGE_NAMES:
        make_contact_sheet(
            cases,
            filename=filename,
            destination=output_dir / f"contact_{filename.replace('.png', '.jpg')}",
        )
    write_gallery(output_dir, cases)
    summary = {
        "source": "historical_output_json_offline_rerender",
        "provider_called": False,
        "results_jsonl": str(args.results_jsonl.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "output_dir": str(output_dir),
        "requested": len(records),
        "rendered": len(cases),
        "failed": len(failures),
        "all_contracts_valid": bool(cases) and all(
            case["contract_valid"] for case in cases
        ),
        "raw_math_issue_total": sum(
            int(case["raw_math_issue_count"]) for case in cases
        ),
        "edge_ink_total": sum(int(case["edge_ink_total"]) for case in cases),
        "cases": cases,
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: summary[key] for key in (
        "output_dir", "requested", "rendered", "failed",
            "all_contracts_valid", "raw_math_issue_total",
            "edge_ink_total",
        )}, ensure_ascii=False))
    return 0 if (
        not failures
        and summary["all_contracts_valid"]
        and summary["raw_math_issue_total"] == 0
        and summary["edge_ink_total"] == 0
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
