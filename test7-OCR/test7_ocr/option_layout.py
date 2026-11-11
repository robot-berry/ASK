from __future__ import annotations

import re


def has_standard_option_sequence(options: list[dict[str, str]]) -> bool:
    """Return whether options are a trustworthy A-to-D prefix without gaps."""

    labels = [str(option.get("label") or "").strip().upper() for option in options]
    texts = [str(option.get("text") or "").strip() for option in options]
    return (
        2 <= len(labels) <= 4
        and labels == list("ABCD"[: len(labels)])
        and all(texts)
    )


def _estimated_em_width(value: str) -> float:
    width = 0.0
    for character in value:
        if character.isspace():
            width += 0.34
        elif "\u2e80" <= character <= "\uffff":
            width += 1.0
        elif character in ".,;:()[]{}'`|":
            width += 0.42
        elif character in "∞√∑∫∩∪≤≥≠≈±×÷":
            width += 0.85
        else:
            width += 0.62
    return width


def option_grid_columns(
    options: list[dict[str, str]],
    content_width_px: int,
    font_size_px: int,
    column_gap_px: int = 24,
) -> int:
    """Choose a conservative 1/2-column option layout.

    The estimate intentionally leaves spare width for KaTeX glyph metrics. An
    option is never split merely to force more columns.
    """

    if not has_standard_option_sequence(options) or any(
        "\n" in str(option.get("text") or "") for option in options
    ):
        return 1
    labels_and_text = [
        f"{option.get('label', '')}. {re.sub(r'\\s+', ' ', str(option.get('text') or '')).strip()}"
        for option in options
    ]
    estimated_widths = [
        _estimated_em_width(value) * font_size_px * 1.08 for value in labels_and_text
    ]
    candidates = [2]
    for columns in candidates:
        cell_width = (content_width_px - column_gap_px * (columns - 1)) / columns
        if max(estimated_widths, default=0.0) <= cell_width * 0.90:
            return columns
    return 1
