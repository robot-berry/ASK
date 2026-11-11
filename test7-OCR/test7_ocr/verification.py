from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from .types import QuestionDocument, RecognitionResult


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


_VISIBLE_ASCII_MATH_FUNCTIONS = {
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "log",
    "ln",
    "exp",
    "lim",
    "max",
    "min",
}


def _normalize_expected(value: str) -> str:
    """Normalize rendered source without counting LaTeX command names as text."""
    value = unicodedata.normalize("NFKC", value or "").lower()

    def replace_command(match: re.Match[str]) -> str:
        command = match.group(1).lower()
        return command if command in _VISIBLE_ASCII_MATH_FUNCTIONS else ""

    value = re.sub(r"\\([a-zA-Z]+)", replace_command, value)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", value)


def _expected_character_coverage(expected: str, actual: str) -> float:
    """Measure expected characters recovered, without penalizing extra visual text."""
    if not expected:
        return 0.0
    matcher = SequenceMatcher(None, expected, actual, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(expected)


def verify_render_readback(
    document: QuestionDocument,
    readback: RecognitionResult,
    minimum_coverage: float,
) -> dict[str, object]:
    expected = _normalize_expected(document.display_text)
    actual = _normalize(readback.transcript)
    coverage = _expected_character_coverage(expected, actual)
    expected_labels = [str(option.get("label") or "") for option in document.options]
    found_label_set = set(
        label.upper()
        for label in re.findall(
            r"(?<![A-Za-z])([A-D])[.．、]",
            readback.transcript,
            flags=re.IGNORECASE,
        )
    )
    # Formula-heavy option rows are occasionally joined by the independent OCR
    # into a line such as ``CD.`` even though both visible labels are present.
    # Accept only a line-leading, alphabetically contiguous run; arbitrary body
    # variables such as AB or CD inside prose do not qualify.
    for match in re.finditer(
        r"(?m)^\s*([A-D]{2,4})[.．、]",
        readback.transcript,
        flags=re.IGNORECASE,
    ):
        run = match.group(1).upper()
        if all(ord(run[index]) == ord(run[0]) + index for index in range(len(run))):
            found_label_set.update(run)
    found_labels = sorted(found_label_set)
    missing_labels = [label for label in expected_labels if label not in found_labels]
    latex_tokens = sorted(
        set(
            token.lower()
            for token in re.findall(
                r"(?:\\|\b)(frac|dfrac|sqrt|begin|end|left|right|mathrm|mathbf|operatorname)\b",
                readback.transcript,
                flags=re.IGNORECASE,
            )
        )
    )
    issues: list[str] = []
    if not actual:
        issues.append("RENDER_READBACK_EMPTY")
    elif coverage < minimum_coverage:
        issues.append("RENDER_READBACK_LOW_COVERAGE")
    if missing_labels:
        issues.append("RENDER_READBACK_OPTION_LOSS")
    if latex_tokens:
        issues.append("RENDER_READBACK_LATEX_TOKEN")
    return {
        "provider": readback.provider,
        "transcript": readback.transcript,
        "coverage_ratio": round(coverage, 6),
        "minimum_coverage_ratio": minimum_coverage,
        "expected_option_labels": expected_labels,
        "found_option_labels": found_labels,
        "missing_option_labels": missing_labels,
        "latex_source_tokens": latex_tokens,
        "issues": issues,
        "passed": not issues,
    }
