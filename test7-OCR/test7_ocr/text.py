from __future__ import annotations

import re
import unicodedata


_OPTION_RE = re.compile(r"(?<![A-Za-z])([A-D])(?:[\.．]|、(?![A-D]))\s*")
_BARE_OPTION_LINE_RE = re.compile(
    r"(?m)^[ \t]*([A-D])(?:[，,][ \t]*|(?=[\u4e00-\u9fff]|[ \t]+[\[（(]))"
)
_COMMA_OPTION_RE = re.compile(r"(?<![A-Za-z])([A-D])[，,]\s*(?=[\u4e00-\u9fff]|\$)")
_BARE_MATH_OPTION_RE = re.compile(r"(?<![A-Za-z])([A-D])(?=\s*\$)")
_BARE_READABLE_MATH_OPTION_RE = re.compile(
    r"(?<![A-Za-z])([A-D])(?=\s*[A-Za-z]\s*(?:[<>≤≥=≠∈]))"
)
_BARE_INTERVAL_OPTION_RE = re.compile(r"(?<![A-Za-z])([A-D])(?=[ \t]*[\[（(])")
_QUESTION_NUMBER_RE = re.compile(
    r"^\s*(\d{1,3}(?:\s*[-—－]\s*[A-Za-zⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)?)"
    r"\s*(?:[\.．、)]|(?=[（(]\d))\s*"
)
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
_SUP_RE = re.compile(r"\^\s*\{([^{}]+)\}")
_PAREN_DEGREE_SUP_RE = re.compile(r"\^\s*(?:\(\s*°\s*\)|°)")
_BARE_SUP_RE = re.compile(r"\^(?!\s*\{)\s*([+-]?\d+)")
_SUB_RE = re.compile(r"_\s*\{([^{}]+)\}")
_RENDER_STRUCTURAL_LINE_RE = re.compile(
    r"^(?:"
    r"[A-D][.．、]\s*|"
    r"[（(]\d{1,2}[）)]|"
    r"[（(](?:i{1,3}|iv|v|vi{0,3}|ix|x)[）)]\s*(?=[\u4e00-\u9fff])|"
    r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]|"
    r"条件[①②③④⑤⑥⑦⑧⑨⑩]\s*[:：]?|"
    r"(?:注|说明|注意)[:：]\s*|"
    r"[一二三四五六七八九十]+[、.．]\s*|"
    r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ])[.．、]\s*(?=[（(])|"
    r"\d{1,3}\s*[-—－]\s*[A-Za-zⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+"
    r")",
    re.IGNORECASE,
)
_FORMULA_SIGNAL_RE = re.compile(
    r"[=<>≤≥≠∈∉∩∪√∞⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉^]"
    r"|[A-Za-z]\s*[_()]?\s*\d"
)

_SUPERSCRIPT_TABLE = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_SUBSCRIPT_TABLE = str.maketrans("0123456789+-=()aeioxn", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₒₓₙ")


def has_formula_signal(value: str, *, formula_count: int = 0) -> bool:
    """Use model regions and readable glyphs under one consistent formula gate."""
    return int(formula_count) > 0 or bool(_FORMULA_SIGNAL_RE.search(value or ""))

_LATEX_WORDS = {
    r"\left": "",
    r"\right": "",
    r"\mid": "|",
    r"\cap": "∩",
    r"\bigcap": "∩",
    r"\cup": "∪",
    r"\bigcup": "∪",
    r"\infty": "∞",
    r"\leq": "≤",
    r"\le": "≤",
    r"\geq": "≥",
    r"\ge": "≥",
    r"\neq": "≠",
    r"\sim": "∼",
    r"\in": "∈",
    r"\notin": "∉",
    r"\subset": "⊂",
    r"\subseteq": "⊆",
    r"\supset": "⊃",
    r"\supseteq": "⊇",
    r"\approx": "≈",
    r"\equiv": "≡",
    r"\propto": "∝",
    r"\to": "→",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\leftrightarrow": "↔",
    r"\times": "×",
    r"\cdot": "·",
    r"\pm": "±",
    r"\angle": "∠",
    r"\parallel": "∥",
    r"\perp": "⊥",
    r"\bot": "⊥",
    r"\because": "∵",
    r"\therefore": "∴",
    r"\pi": "π",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\theta": "θ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\nu": "ν",
    r"\Delta": "Δ",
    r"\Omega": "Ω",
    r"\omega": "ω",
    r"\rho": "ρ",
    r"\sigma": "σ",
    r"\phi": "φ",
    r"\varphi": "φ",
    r"\eta": "η",
    r"\degree": "°",
    r"\circ": "°",
    r"\cdots": "…",
    r"\ldots": "…",
    r"\quad": " ",
    r"\qquad": "  ",
    r"\,": " ",
    r"\!": "",
}


def _compact_group(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _fraction_display(numerator: str, denominator: str) -> str:
    numerator = _compact_group(numerator)
    denominator = _compact_group(denominator)
    simple = re.compile(r"^[\wπθαβγμλρσωΔΩ.°+-]+$", re.UNICODE)
    left = numerator if simple.fullmatch(numerator) else f"({numerator})"
    right = denominator if simple.fullmatch(denominator) else f"({denominator})"
    return f"{left}/{right}"


def _braced_group(value: str, start: int) -> tuple[str, int] | None:
    """Return a balanced braced group and the first index after it."""
    if start >= len(value) or value[start] != "{":
        return None
    depth = 0
    for index in range(start, len(value)):
        if value[index] == "{" and (index == 0 or value[index - 1] != "\\"):
            depth += 1
        elif value[index] == "}" and (index == 0 or value[index - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return value[start + 1 : index], index + 1
    return None


def _replace_balanced_fractions(value: str) -> str:
    """Convert frac commands even when either argument contains nested groups."""
    pattern = re.compile(r"\\(?:d?frac)\b")
    output: list[str] = []
    cursor = 0
    while match := pattern.search(value, cursor):
        first_start = match.end()
        while first_start < len(value) and value[first_start].isspace():
            first_start += 1
        numerator_group = _braced_group(value, first_start)
        if numerator_group is None:
            output.append(value[cursor : match.end()])
            cursor = match.end()
            continue
        numerator, second_start = numerator_group
        while second_start < len(value) and value[second_start].isspace():
            second_start += 1
        denominator_group = _braced_group(value, second_start)
        if denominator_group is None:
            output.append(value[cursor : match.end()])
            cursor = match.end()
            continue
        denominator, end = denominator_group
        output.append(value[cursor : match.start()])
        output.append(
            _fraction_display(
                _replace_balanced_fractions(numerator),
                _replace_balanced_fractions(denominator),
            )
        )
        cursor = end
    output.append(value[cursor:])
    return "".join(output)


def _replace_balanced_radicals(value: str) -> str:
    """Convert radicals whose arguments contain nested script groups."""
    pattern = re.compile(r"\\sqrt\b")
    output: list[str] = []
    cursor = 0
    while match := pattern.search(value, cursor):
        group_start = match.end()
        while group_start < len(value) and value[group_start].isspace():
            group_start += 1
        group = _braced_group(value, group_start)
        if group is None:
            output.append(value[cursor : match.end()])
            cursor = match.end()
            continue
        radicand, end = group
        # Formula OCR sometimes inserts a visual gap before a script marker
        # inside a radical (for example ``S _ {n}``).  Compact only that
        # radicand-local gap; preserving spacing elsewhere is important because
        # source-token repair rules match the independently assembled display.
        radicand = re.sub(r"(?<=\w)\s+(?=[_^])", "", radicand)
        output.append(value[cursor : match.start()])
        output.append(_radical_display(_replace_balanced_radicals(radicand)))
        cursor = end
    output.append(value[cursor:])
    return "".join(output)


def _radical_display(value: str) -> str:
    value = _compact_group(value)
    if re.fullmatch(r"[\wπθαβγμλρσωΔΩ.°+-]+", value, re.UNICODE):
        return f"√{value}"
    return f"√({value})"


def _script_display(value: str, *, superscript: bool) -> str:
    compact = _compact_group(value)
    table = _SUPERSCRIPT_TABLE if superscript else _SUBSCRIPT_TABLE
    converted = compact.translate(table)
    convertible = set("0123456789+-=()n" if superscript else "0123456789+-=()aeioxn")
    if compact and all(character in convertible for character in compact):
        return converted
    return ("^(" if superscript else "_(") + compact + ")"


def latex_to_readable(text: str) -> str:
    """Convert common OCR LaTeX to readable Unicode without exposing LaTeX source."""
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\\(", "").replace("\\)", "")
    value = value.replace("\\[", "").replace("\\]", "")
    value = value.replace("$$", "").replace("$", "")

    for source, replacement in sorted(_LATEX_WORDS.items(), key=lambda item: -len(item[0])):
        if source[1:].isalpha():
            value = re.sub(re.escape(source) + r"(?![A-Za-z])", replacement, value)
        else:
            value = value.replace(source, replacement)

    value = re.sub(
        r"\\underline\s*\{\s*\\hspace\*?\s*\{[^{}]+\}\s*\}",
        "______",
        value,
    )
    value = re.sub(r"\\hspace\*?\s*\{[^{}]+\}", " ", value)
    value = _replace_balanced_fractions(value)
    value = _replace_balanced_radicals(value)
    value = _SQRT_RE.sub(lambda match: _radical_display(match.group(1)), value)
    value = _SUP_RE.sub(lambda match: _script_display(match.group(1), superscript=True), value)
    value = _PAREN_DEGREE_SUP_RE.sub("°", value)
    value = _BARE_SUP_RE.sub(
        lambda match: _script_display(match.group(1), superscript=True),
        value,
    )
    value = _SUB_RE.sub(lambda match: _script_display(match.group(1), superscript=False), value)

    value = value.replace(r"\{", "{").replace(r"\}", "}")
    value = value.replace("{,}", ",")
    wrapper_pattern = re.compile(
        r"\\(?:mathrm|mathbf|boldsymbol|pmb|mathit|mathsf|mathtt|mathbb|mathcal|text|textbf|operatorname|vec|overrightarrow|overline|underline|underbrace|overbrace|bar|hat|textcircled)\s*\{([^{}]*)\}"
    )
    for _ in range(6):
        unwrapped = wrapper_pattern.sub(r"\1", value)
        if unwrapped == value:
            break
        value = unwrapped
    value = re.sub(r"\\begin\s*\{[^{}]+\}|\\end\s*\{[^{}]+\}", " ", value)
    value = re.sub(r"\\(?:sin|cos|tan|cot|log|ln|lim|max|min)\b", lambda match: match.group(0)[1:], value)
    value = re.sub(r"\\([A-Za-z]+)", r"\1", value)
    value = value.replace(r"\\", "\n")
    value = value.replace("\\", "")
    value = re.sub(r"[{}]", "", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\(\s+", "(", value)
    value = re.sub(r"\s+\)", ")", value)
    value = re.sub(r"\s*/\s*", "/", value)
    value = re.sub(r"[,，]\s*[,，]", "，", value)
    value = re.sub(r"[,，]\s*[。.]+", "。", value)
    value = re.sub(r"[。.]+\s*[,，]", "。", value)
    value = re.sub(r" *\n *", "\n", value)
    return value.strip()


def canonicalize_formula(value: str) -> str:
    """Compact comparable form for isolated formula match and self-check."""
    compact = unicodedata.normalize("NFKC", latex_to_readable(value or ""))
    compact = compact.replace("，", ",").replace("。", ".").replace("：", ":")
    return re.sub(r"\s+", "", compact).lower()


def formulas_agree(
    left: str,
    right: str,
    *,
    min_containment_chars: int = 3,
) -> bool:
    needle = canonicalize_formula(left)
    haystack = canonicalize_formula(right)
    if not needle or not haystack:
        return False
    if needle == haystack:
        return True
    if min(len(needle), len(haystack)) < min_containment_chars:
        return False
    return needle in haystack or haystack in needle


def formula_lists_agree(predicted: list[str], independent: list[str]) -> bool:
    if not predicted or not independent:
        return False
    return all(any(formulas_agree(item, other) for other in independent) for item in predicted)


def unknown_latex_commands(text: str) -> list[str]:
    commands = set(re.findall(r"\\([A-Za-z]+)", text or ""))
    handled = {key[1:] for key in _LATEX_WORDS}
    handled.update(
        {
            "frac",
            "dfrac",
            "sqrt",
            "mathrm",
            "mathbf",
            "boldsymbol",
            "pmb",
            "mathit",
            "mathsf",
            "mathtt",
            "mathbb",
            "mathcal",
            "text",
            "textbf",
            "operatorname",
            "vec",
            "overrightarrow",
            "overline",
            "underline",
            "underbrace",
            "overbrace",
            "bar",
            "hat",
            "textcircled",
            "begin",
            "end",
            "sin",
            "cos",
            "tan",
            "cot",
            "log",
            "ln",
            "lim",
            "max",
            "min",
        }
    )
    return sorted(commands - handled)


def contains_latex_source(text: str) -> bool:
    return bool(
        re.search(
            r"\$|\\(?:frac|dfrac|sqrt|left|right|begin|end|mathrm|mathbf|text|operatorname|[A-Za-z]{2,})",
            text or "",
        )
    )


def _is_inside_inline_math(value: str, position: int) -> bool:
    delimiter_count = 0
    index = 0
    while index < position:
        if value[index] == "$" and (index == 0 or value[index - 1] != "\\"):
            delimiter_count += 1
        index += 1
    return bool(delimiter_count % 2)


def _split_reference_items(line: str) -> list[str]:
    """Split numbered reference items without leaving a newline inside LaTeX.

    Some OCR providers wrap the whole printed reference row in one pair of
    dollar delimiters, including the second item marker.  A plain newline is
    then swallowed as formula whitespace by KaTeX.  Close and reopen that
    formula only around strongly formatted ``\\quad (n) \\quad`` markers,
    then split the now-textual ordered markers into real rows.
    """

    marker_re = re.compile(r"[（(]\s*[1-9]\d?\s*[）)]")
    quad_marker_re = re.compile(
        r"\\quad\s*([（(]\s*[1-9]\d?\s*[）)])\s*\\quad"
    )

    parts: list[str] = []
    cursor = 0
    for match in quad_marker_re.finditer(line):
        if not _is_inside_inline_math(line, match.start()):
            continue
        parts.append(line[cursor : match.start()])
        parts.append("$\n" + match.group(1) + " $")
        cursor = match.end()
    parts.append(line[cursor:])
    prepared = "".join(parts)

    markers = [
        match
        for match in marker_re.finditer(prepared)
        if not _is_inside_inline_math(prepared, match.start())
    ]
    if len(markers) < 2:
        return [line]

    for match in reversed(markers):
        prefix = prepared[: match.start()]
        if prefix and not prefix.endswith("\n"):
            prepared = prefix.rstrip() + "\n" + prepared[match.start() :]
    return [part.strip() for part in prepared.splitlines() if part.strip()]


def reflow_text_for_render(text: str) -> str:
    """Remove source-photo line wrapping while preserving semantic structure.

    OCR rows describe the physical source layout, not the desired 800px
    layout.  Ordinary adjacent rows are joined and allowed to wrap at the new
    width.  Options, numbered subquestions, circled list items, and exam
    section headings remain explicit line boundaries.
    """

    source_lines = [
        line.strip()
        for line in (text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
        if line.strip()
    ]
    lines: list[str] = []
    for line in source_lines:
        # Reference formula lists and selectable condition groups have their
        # own semantic rows even when OCR merges the printed columns. Split
        # only when the cue and at least two ordered markers are both present.
        if re.search(r"参考(?:公式|数据|关系)", line):
            line = "\n".join(_split_reference_items(line))
        conditions = list(re.finditer(r"条件[①②③④⑤⑥⑦⑧⑨⑩]", line))
        if len(conditions) >= 2:
            line = re.sub(r"(?<!^)\s*(?=条件[①②③④⑤⑥⑦⑧⑨⑩])", "\n", line)
        lines.extend(part.strip() for part in line.splitlines() if part.strip())
    if not lines:
        return ""

    output: list[str] = []
    current = lines[0]
    for next_line in lines[1:]:
        hard_boundary = bool(_RENDER_STRUCTURAL_LINE_RE.match(next_line))
        if hard_boundary:
            output.append(current.rstrip())
            current = next_line
            continue

        separator = ""
        if current and next_line and current[-1].isascii() and next_line[0].isascii():
            if current[-1].isalnum() and next_line[0].isalnum():
                separator = " "
        current = current.rstrip() + separator + next_line.lstrip()
    output.append(current.rstrip())
    return "\n".join(output)


def parse_question(
    display_text: str,
    *,
    recovery_warnings: list[str] | None = None,
) -> tuple[str | None, str, list[dict[str, str]]]:
    text = display_text.strip()
    number_match = _QUESTION_NUMBER_RE.match(text)
    if number_match is None:
        number_match = re.search(
            r"(?m)^\s*(\d{1,3}(?:\s*[-—－]\s*[A-Za-zⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)?)"
            r"\s*(?:[\.．、)]|(?=[（(]\d))\s*",
            text,
        )
    if number_match is None:
        # A high-confidence text recognizer occasionally drops only the dot
        # after a short line-leading question number (``10 某工地...``).  The
        # fallback is deliberately limited to the beginning of the document,
        # one/two digit numbers, whitespace, and a Chinese stem character so a
        # year or a numerical formula cannot become a fabricated question id.
        number_match = re.match(r"^\s*(\d{1,2})[ \t]+(?=[\u4e00-\u9fff])", text)
        if number_match is not None and recovery_warnings is not None:
            recovery_warnings.append(
                f"QUESTION_NUMBER_PUNCTUATION_RECOVERED:{number_match.group(1)}"
            )
    question_number = number_match.group(1) if number_match else None

    punctuated = [
        (match.start(), match.end(), match.group(1), True)
        for match in _OPTION_RE.finditer(text)
    ]
    if len(punctuated) < 2:
        return question_number, text, []

    recovered = [
        (match.start(1), match.end(), match.group(1), False)
        for pattern in (
            _BARE_OPTION_LINE_RE,
            _COMMA_OPTION_RE,
            _BARE_MATH_OPTION_RE,
            _BARE_READABLE_MATH_OPTION_RE,
            _BARE_INTERVAL_OPTION_RE,
        )
        for match in pattern.finditer(text)
    ]
    candidates_by_position: dict[tuple[int, str], tuple[int, int, str, bool]] = {}
    for candidate in [*recovered, *punctuated]:
        key = (candidate[0], candidate[2])
        previous = candidates_by_position.get(key)
        if previous is None or candidate[3]:
            candidates_by_position[key] = candidate
    combined = sorted(candidates_by_position.values(), key=lambda value: value[0])

    def ordered_runs(
        candidates: list[tuple[int, int, str, bool]],
    ) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        labels = [candidate[2] for candidate in candidates]
        for start, label in enumerate(labels):
            if label != "A":
                continue
            length = 1
            while (
                start + length < len(labels)
                and length < 4
                and labels[start + length] == chr(ord("A") + length)
            ):
                length += 1
            if length >= 2:
                runs.append((length, start))
        return runs

    matches = punctuated
    recovered_runs = [
        (length, start)
        for length, start in ordered_runs(combined)
        if length >= 3
        and sum(
            1 for candidate in combined[start : start + length] if candidate[3]
        )
        >= 2
        and any(
            not candidate[3] for candidate in combined[start : start + length]
        )
    ]
    if recovered_runs:
        length, start = max(recovered_runs, key=lambda value: (value[0], value[1]))
        matches = combined[start : start + length]
        if recovery_warnings is not None:
            recovered_labels = "".join(
                candidate[2] for candidate in matches if not candidate[3]
            )
            recovery_warnings.append(
                "OPTION_SEQUENCE_RECOVERED_FROM_COHERENT_LABELS:"
                f"{recovered_labels}"
            )

    # Mathematical prose can contain variable names followed by punctuation,
    # for example ``A、B`` or ``OB.``.  Those tokens satisfy the lexical option
    # pattern but do not form the ordered answer group.  Prefer the longest
    # contiguous A-B-C-D run (and the later run on a tie) so the stem is not
    # split at an incidental variable occurrence.  If no ordered run exists,
    # keep every match and let the validator expose the ambiguity.
    punctuated_runs = ordered_runs(matches)
    if punctuated_runs:
        length, start = max(punctuated_runs, key=lambda value: (value[0], value[1]))
        matches = matches[start : start + length]

    stem = text[: matches[0][0]].strip()
    options: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        option_text = text[match[1] : end].strip()
        options.append({"label": match[2], "text": option_text})
    return question_number, stem, options


def has_visual_cue(text: str, cues: list[str]) -> bool:
    value = text or ""
    # These words often describe an abstract mathematical object without an
    # attached bitmap (for example, "将函数图像向右平移" or "在平面直角
    # 坐标系中").  Treat them as image requirements only when the wording
    # points to a displayed/selected picture.  Strong cues such as "如图" and
    # "图中" are still handled independently below.
    weak_visual_cues = {"图象", "图像", "坐标系"}
    weak_visual_reference = re.compile(
        r"(?:"
        r"(?:下列|以下|上图|下图|左图|右图|所给|给出)[^。；;\n]{0,12}(?:图象|图像)"
        r"|(?:图象|图像)[^。；;\n]{0,10}(?:如下|如图|所示|选项|正确的是|可能是|可能为|大致是|大致为)"
        r"|(?:坐标系)[^。；;\n]{0,12}(?:如下|如图|所示)"
        r")"
    )
    weak_reference_present = bool(weak_visual_reference.search(value))
    hypothetical_graph_spans = [
        match.span()
        for match in re.finditer(
            r"(?:如果|若)[^。；;\n]{0,24}(?:散点图|函数图|统计图)中",
            value,
        )
    ]
    for cue in cues:
        if cue in weak_visual_cues and not weak_reference_present:
            continue
        for match in re.finditer(re.escape(cue), value):
            if cue == "图中" and any(start <= match.start() < end for start, end in hypothetical_graph_spans):
                continue
            return True
    return False
