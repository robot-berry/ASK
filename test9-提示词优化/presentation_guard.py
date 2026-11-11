"""Presentation-boundary guard for OCR and model-generated math text.

Internal OCR/model data intentionally keeps formulas as LaTeX because that
preserves fractions, roots, scripts, and relation symbols for downstream
reasoning.  Every public presentation boundary must pass through this module
so implementation markers are never shown to a person.
"""

from __future__ import annotations

import re
from typing import Any


_SUPERSCRIPTS = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_SUBSCRIPTS = str.maketrans(
    "0123456789+-=()aehijklmnoprstuvx",
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ",
)
_COMMANDS = {
    "left": "",
    "right": "",
    "times": "×",
    "cdot": "·",
    "div": "÷",
    "pm": "±",
    "mp": "∓",
    "leqslant": "≤",
    "leq": "≤",
    "le": "≤",
    "geqslant": "≥",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "ne": "≠",
    "approx": "≈",
    # Microsoft YaHei used by the lightweight audit-card renderer lacks the
    # U+223C glyph on some Windows builds.  ASCII tilde is unambiguous here
    # and avoids a visible tofu square.
    "sim": "~",
    "cong": "≌",
    "equiv": "≡",
    "propto": "∝",
    "ll": "≪",
    "gg": "≫",
    "infty": "∞",
    "sum": "Σ",
    "prod": "Π",
    "int": "∫",
    "cap": "∩",
    "cup": "∪",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "subseteq": "⊆",
    "supset": "⊃",
    "supseteq": "⊇",
    "emptyset": "∅",
    "varnothing": "∅",
    "odot": "⊙",
    "circ": "°",
    "triangle": "△",
    "angle": "∠",
    "perp": "⊥",
    "parallel": "∥",
    "rightarrow": "→",
    "longrightarrow": "→",
    "leftarrow": "←",
    "longleftarrow": "←",
    "leftrightarrow": "↔",
    "to": "→",
    "Rightarrow": "⇒",
    "Leftrightarrow": "⇔",
    "uparrow": "↑",
    "downarrow": "↓",
    "mid": "|",
    "because": "∵",
    "therefore": "∴",
    "dots": "…",
    "cdots": "…",
    "ldots": "…",
    "quad": " ",
    "qquad": " ",
    "pi": "π",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ϵ",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "varphi": "ϕ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "cot": "cot",
    "log": "log",
    "ln": "ln",
    "lim": "lim",
    "max": "max",
    "min": "min",
}

_WRAPPERS = frozenset(
    {
        "text",
        "textrm",
        "mathrm",
        "mathbf",
        "mathit",
        "mathsf",
        "mathbb",
        "operatorname",
        "underline",
        "overline",
        "bar",
        "hat",
    }
)


def _braced_group(value: str, position: int) -> tuple[str, int] | None:
    while position < len(value) and value[position].isspace():
        position += 1
    if position >= len(value) or value[position] != "{":
        return None
    depth = 1
    index = position + 1
    start = index
    while index < len(value):
        if value[index] == "\\":
            index += 2
            continue
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                return value[start:index], index + 1
        index += 1
    return None


def _fraction(numerator: str, denominator: str) -> str:
    numerator = numerator.strip()
    denominator = denominator.strip()
    simple = re.compile(r"[A-Za-z0-9Α-Ωα-ωπμθφ∞√∛₀-₉⁰-⁹.+−-]+")
    left = numerator if simple.fullmatch(numerator) else f"({numerator})"
    right = denominator if simple.fullmatch(denominator) else f"({denominator})"
    return f"{left}/{right}"


def _decode_structures(value: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(value):
        if value[index] != "\\":
            output.append(value[index])
            index += 1
            continue
        command_match = re.match(r"\\([A-Za-z]+|.)", value[index:])
        if command_match is None:
            index += 1
            continue
        command = command_match.group(1)
        next_index = index + len(command_match.group(0))
        if command in {"frac", "dfrac", "tfrac"}:
            numerator = _braced_group(value, next_index)
            denominator = _braced_group(value, numerator[1]) if numerator else None
            if numerator and denominator:
                output.append(
                    _fraction(
                        _decode_structures(numerator[0]),
                        _decode_structures(denominator[0]),
                    )
                )
                index = denominator[1]
                continue
        if command == "sqrt":
            root_index = ""
            cursor = next_index
            while cursor < len(value) and value[cursor].isspace():
                cursor += 1
            if cursor < len(value) and value[cursor] == "[":
                end = value.find("]", cursor + 1)
                if end >= 0:
                    root_index = value[cursor + 1:end].strip()
                    cursor = end + 1
            radicand = _braced_group(value, cursor)
            if radicand:
                body = _decode_structures(radicand[0]).strip()
                body = body if re.fullmatch(r"[A-Za-z0-9Α-Ωα-ωπμθφ.+−-]+", body) else f"({body})"
                output.append(("∛" if root_index == "3" else f"{root_index}√" if root_index else "√") + body)
                index = radicand[1]
                continue
        if command in _WRAPPERS or command in {"vec", "overrightarrow", "boldsymbol", "bm"}:
            group = _braced_group(value, next_index)
            if group:
                body = _decode_structures(group[0]).replace("~", " ").strip()
                output.append(("→" if command in {"vec", "overrightarrow", "boldsymbol", "bm"} else "") + body)
                index = group[1]
                continue
        if command == "binom":
            upper = _braced_group(value, next_index)
            lower = _braced_group(value, upper[1]) if upper else None
            if upper and lower:
                output.append(f"C({_decode_structures(upper[0])},{_decode_structures(lower[0])})")
                index = lower[1]
                continue
        if command in {",", ";", ":", "!", " "}:
            output.append(" " if command != "!" else "")
        elif command in {"{", "}"}:
            output.append(command)
        elif command == "\\":
            output.append("\n")
        else:
            # Unknown commands remain readable without exposing a raw
            # backslash-prefixed implementation token.
            output.append(_COMMANDS.get(command, command))
        index = next_index
    return "".join(output)


def _script(match: re.Match[str], *, superscript: bool) -> str:
    value = match.group(1).strip()
    table = _SUPERSCRIPTS if superscript else _SUBSCRIPTS
    if value and all(ord(character) in table for character in value):
        return value.translate(table)
    marker = "^" if superscript else "_"
    return f"{marker}({value})"


def readable_ocr_text(value: object) -> str:
    """Return human-facing OCR text with no raw LaTeX/Markdown markers."""
    text = str(value or "").replace("\x00", "")
    text = re.sub(r"```(?:[A-Za-z0-9_-]+)?", "", text.strip())
    text = text.replace("`", "").replace("**", "")
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    text = re.sub(r"(?:\\){2,}(?=[A-Za-z])", r"\\", text)
    text = re.sub(r"\\begin\s*\{(?:cases|array|aligned|matrix)\}(?:\{[^{}]*\})?", "", text)
    text = re.sub(r"\\end\s*\{(?:cases|array|aligned|matrix)\}", "", text)
    text = re.sub(r"\\(?:begin|end)\s*\{[^{}]+\}", "\n", text)
    text = _decode_structures(text)
    text = re.sub(r"\^\{([^{}]+)\}", lambda match: _script(match, superscript=True), text)
    text = re.sub(r"_\{([^{}]+)\}", lambda match: _script(match, superscript=False), text)
    text = re.sub(r"\^([0-9n])", lambda match: _script(match, superscript=True), text)
    text = re.sub(
        r"_([0-9aehijklmnoprstuvx])",
        lambda match: _script(match, superscript=False),
        text,
    )
    text = text.replace("$", "").replace("&", "，")
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\\[A-Za-z]+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([，。；：！？、,.!?;:）)\]])", r"\1", text)
    text = re.sub(r"([（(\[])\s+", r"\1", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    return text.strip()


def public_display_text(value: object, *, limit: int | None = None) -> str:
    """Normalize one outbound text field and apply its public size limit."""
    text = readable_ocr_text(value)
    if limit is not None:
        text = text[: max(0, int(limit))]
    return text


def contains_source_markup(value: object) -> bool:
    """Detect implementation syntax that must not reach a review surface."""
    text = str(value or "")
    return bool(
        re.search(
            r"\\[A-Za-z]+|\$|```|\\begin\s*\{|\\end\s*\{|"
            r"(?m:^\s{0,3}#{1,6}\s+)",
            text,
        )
    )


def source_markup_paths(value: Any, *, path: str = "$") -> list[str]:
    """Return public payload locations that still contain source markup."""
    leaks: list[str] = []
    if isinstance(value, str):
        if contains_source_markup(value):
            leaks.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            leaks.extend(source_markup_paths(item, path=f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            leaks.extend(source_markup_paths(item, path=f"{path}[{index}]"))
    return leaks
