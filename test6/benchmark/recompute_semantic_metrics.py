"""Re-evaluate saved OCR transcripts with the current local semantic gates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ocr_accuracy_300 import load_test5, read_jsonl, write_outputs


ISSUE_KEYS = (
    "stem_missing",
    "option_sequence_incomplete",
    "choice_options_missing",
    "empty_option_content",
    "subquestion_gap",
    "nested_subquestion_incomplete",
    "table_row_incomplete",
    "formula_gap",
    "symbol_damage",
    "too_short",
    "multiple_question_blocks",
    "repeated_question_number",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input / "results.jsonl" if args.input.is_dir() else args.input
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    modules: dict[str, object] = {}
    rows = read_jsonl(input_path.resolve())
    for row in rows:
        if row.get("status") != "success":
            continue
        subject = str(row["subject"])
        if subject not in modules:
            modules[subject] = load_test5(subject)
        module = modules[subject]
        transcript = str(row.get("transcript") or "")
        # Saved rows may predate deterministic reading-order repairs added to
        # the production OCR path.  Reapply the same local cleanup here so a
        # metrics-only refresh evaluates exactly what the current service
        # would return (for example, an OCR line that repeats its question
        # number and opening phrase once).
        if hasattr(module, "repair_ocr_transcript"):
            transcript = module.repair_ocr_transcript(transcript)
            row["transcript"] = transcript
        metrics = module.transcript_semantic_completeness(transcript)
        issues = [key for key in ISSUE_KEYS if metrics.get(key) is True]
        table_rendered_as_text = bool(
            re.search(r"下表|如下表|表中|表格|统计表|成绩表", transcript)
            and not re.search(r"如图|图[①②③④⑤⑥1-9]", transcript)
            and metrics.get("table_row_incomplete") is not True
        )
        requires_visual = bool(
            hasattr(module, "transcript_requires_illustration")
            and module.transcript_requires_illustration(transcript)
            and not table_rendered_as_text
        ) or bool(row.get("formula_visual_evidence_required"))
        has_visual = int(row.get("illustration_count") or 0) > 0
        structure_pass = bool(metrics.get("passes_structure_gate"))
        package_pass = bool(
            (structure_pass and (not requires_visual or has_visual))
            or (set(issues) == {"table_row_incomplete"} and has_visual)
        )
        row.update(
            {
                "semantic_structure_pass": structure_pass,
                "semantic_issue_count": int(metrics.get("issue_count") or 0),
                "semantic_issue_types": issues,
                "semantic_package_pass": package_pass,
                "semantic_package_evidence": (
                    "text_plus_preserved_table_figure"
                    if set(issues) == {"table_row_incomplete"} and has_visual
                    else "text_plus_preserved_visual"
                    if requires_visual and has_visual
                    else "text_structure_gate"
                ),
                "semantic_page_noise_detected": bool(metrics.get("page_noise")),
                "semantic_option_labels": list(metrics.get("option_labels") or []),
                "semantic_choice_count_ambiguous": bool(
                    metrics.get("choice_count_ambiguous")
                ),
                "semantic_subquestion_numbers": list(
                    metrics.get("subquestion_numbers") or []
                ),
                "semantic_requires_visual": requires_visual,
            }
        )
        flags = [
            str(flag)
            for flag in (row.get("quality_flags") or [])
            if str(flag) != "semantic_structure_gate_failed"
        ]
        if not structure_pass:
            flags.append("semantic_structure_gate_failed")
        row["quality_flags"] = list(dict.fromkeys(flags))
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_outputs(args.dataset.resolve(), output, rows)
    print(f"Recomputed {len(rows)} rows: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
