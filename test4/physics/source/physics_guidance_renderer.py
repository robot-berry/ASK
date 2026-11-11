#!/usr/bin/env python3
"""Render one physics problem into four 800 px document-style grayscale PNGs.

The model produces structured JSON only.  This module owns all presentation:

1. re-typeset question text + verified source figure crops + knowledge/formulae;
2. advanced guidance;
3. standard guidance;
4. foundation guidance.

Chinese glyphs use a Song-style CJK serif face and Latin/digits use a
Times-compatible serif face.  Formulae are
drawn with the existing result-image rich renderer, so fractions, radicals and
scripts are never exposed as raw LaTeX on the device.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from result_image_store import (
    clean_text,
    draw_formula_at_baseline,
    font_ascent,
    font_vertical_bounds,
    formula_atoms as rich_formula_atoms,
    formula_vertical_bounds,
    text_width as rich_text_width,
)


WIDTH = 800
DISPLAY_WIDTH_CM = 9.0
PNG_DPI = WIDTH / (DISPLAY_WIDTH_CM / 2.54)
RENDER_MODEL_FIGURE_CAPTIONS = False
SUBSTANTIVE_GUIDANCE_CONTENT = re.compile(
    r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)
# PNG stores resolution as an integer number of pixels per metre.  One unit
# therefore equals 0.0254 ppi; half a unit is the maximum correct rounding
# error when Pillow serializes the requested DPI into the pHYs chunk.
PNG_DPI_QUANTIZATION_PPI = 0.0254
PNG_DPI_QUANTIZATION_TOLERANCE = PNG_DPI_QUANTIZATION_PPI / 2 + 0.000001
PNG_PHYSICAL_WIDTH_TOLERANCE_CM = 0.001
MARGIN_X = 32
CONTENT_WIDTH = WIDTH - MARGIN_X * 2
# Audit masters are scaled from 800 px to the 640 px e-ink device width.
# Keep one source figure below 448 device pixels so it can stay intact on a
# 480 px page with 32 px of vertical breathing room.  This is a resize-only
# constraint: the reviewed source bbox is never cropped again to meet it.
DEVICE_WIDTH = 640
DEVICE_HEIGHT = 480
SOURCE_FIGURE_DEVICE_VERTICAL_WHITESPACE = 32
SOURCE_FIGURE_MAX_DEVICE_HEIGHT = (
    DEVICE_HEIGHT - SOURCE_FIGURE_DEVICE_VERTICAL_WHITESPACE
)
SOURCE_FIGURE_MAX_OUTPUT_HEIGHT = round(
    SOURCE_FIGURE_MAX_DEVICE_HEIGHT * WIDTH / DEVICE_WIDTH
)
TITLE_SIZE = 42
BODY_SIZE = 32
FORMULA_SIZE = 34
SMALL_SIZE = 27
SCRIPT_SIZE = 22
TITLE_LINE = 58
BODY_LINE = 52
FORMULA_LINE = 62
SMALL_LINE = 44
TOP = 24
BOTTOM = 34
DEFAULT_SONG = Path(r"C:\Windows\Fonts\simsun.ttc")
DEFAULT_SONG_BOLD = DEFAULT_SONG
DEFAULT_TIMES = Path(r"C:\Windows\Fonts\times.ttf")
DEFAULT_TIMES_BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")
DEFAULT_MATH_SYMBOL = Path(
    os.environ.get("TEST3_SYMBOL_FONT", r"C:\Windows\Fonts\seguisym.ttf")
)
MATH_SYMBOL_CHARACTERS = frozenset("≪∠∝⊥∥≌∽△○⊙∩∪∈≤≥≠≈")


def _safe_textlength(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
) -> float:
    """Measure model text without Pillow's multiline ``textlength`` crash."""

    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    if "\n" not in normalized:
        return float(draw.textlength(normalized, font=font))
    single_line = re.sub(r"\s*\n\s*", " ", normalized)
    return float(draw.textlength(single_line, font=font))
UNWRAPPED_GREEK_COMMANDS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "theta": "θ",
    "lambda": "λ",
    "mu": "μ",
    "rho": "ρ",
    "phi": "φ",
    "omega": "ω",
    "Delta": "Δ",
    "Phi": "Φ",
    "Omega": "Ω",
}
FORMULA_MARKER = re.compile(r"\$([^$]+)\$")
CHOICE_OPTION_MARKER = re.compile(
    r"(?<![A-Za-z0-9])([A-HＡ-Ｈ])\s*[.．、:]\s*"
)
CHOICE_OPTION_MIN_GAP = 4
PLAIN_MATH_TOKEN = re.compile(
    r"√\([^()\n]{1,80}\)|"
    r"(?<![A-Za-z0-9])(?:"
    r"\d+(?:\.\d+)?(?:[A-Za-zΑ-Ωα-ωθΘ][A-Za-zΑ-Ωα-ωθΘ0-9₀₁₂₃₄₅₆₇₈₉ₓᵧ²³]*)?"
    r"|[A-Za-zΑ-Ωα-ωθΘ][A-Za-zΑ-Ωα-ωθΘ0-9₀₁₂₃₄₅₆₇₈₉ₓᵧ²³]*"
    r")(?:/(?:\d+(?:\.\d+)?|[A-Za-zΑ-Ωα-ωθΘ][A-Za-zΑ-Ωα-ωθΘ0-9₀₁₂₃₄₅₆₇₈₉ₓᵧ²³]*))?"
    r"(?![A-Za-z0-9])"
)
SIMPLE_SLASH_FRACTION = re.compile(
    r"^(?P<numerator>(?:\d+(?:\.\d+)?[A-Za-zΑ-Ωα-ωθΘ]?|"
    r"[A-Za-zΑ-Ωα-ωθΘ][A-Za-zΑ-Ωα-ωθΘ0-9₀₁₂₃₄₅₆₇₈₉ₓᵧ²³]*))"
    r"/(?P<denominator>(?:\d+(?:\.\d+)?|"
    r"[A-Za-zΑ-Ωα-ωθΘ][A-Za-zΑ-Ωα-ωθΘ0-9₀₁₂₃₄₅₆₇₈₉ₓᵧ²³]*))$"
)
CJK_RUN = re.compile(r"([\u2e80-\u9fff\uf900-\ufaff]+)")
FORMULA_CJK_SCRIPT = re.compile(
    r"([_^])\{(?=[^{}]*[\u2e80-\u9fff\uf900-\ufaff])"
    r"([\u2e80-\u9fff\uf900-\ufaff0-9]+)\}"
)
RAW_CAPTION_FORMULA = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"[A-Za-z](?:_[A-Za-z0-9]+|\^[A-Za-z0-9]+)?"
    r"(?:\s*[-=+]\s*[A-Za-z](?:_[A-Za-z0-9]+|\^[A-Za-z0-9]+)?)+"
    r"|[A-Za-z](?:_[A-Za-z0-9]+)?\^[A-Za-z0-9]+"
    r")(?![A-Za-z0-9])"
)
FORBIDDEN_LINE_START = set("，。；：！？、）】》”’")
FORBIDDEN_LINE_END = set("∠△")
METHOD_NUMBER_NAMES = ("一", "二", "三")


@dataclass(frozen=True)
class Fonts:
    song: Any
    song_bold: Any
    song_bold_is_distinct: bool
    times: Any
    times_bold: Any
    formula: Any
    formula_song: Any
    script: Any
    script_song: Any
    math_symbol: Any


@dataclass(frozen=True)
class Atom:
    kind: str
    value: str
    width: float


@dataclass(frozen=True)
class RenderLine:
    atoms: tuple[Atom, ...]
    height: int
    left: int = 0
    font_role: str = "body"
    top_pad: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_font(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"缺少{label}字体：{resolved}")
    return resolved


def select_named_variation(
    font: ImageFont.FreeTypeFont,
    *preferred_names: str,
) -> bool:
    """Select a stable weight from a variable font when one is supplied."""

    try:
        available = font.get_variation_names()
    except (AttributeError, OSError):
        return False
    decoded = {
        (item.decode("utf-8", errors="ignore") if isinstance(item, bytes) else str(item)):
        item
        for item in available
    }
    for name in preferred_names:
        selected = decoded.get(name)
        if selected is None:
            continue
        font.set_variation_by_name(selected)
        return True
    return False


def resolve_math_symbol_path(
    math_symbol_path: Path,
    *,
    times_path: Path,
    song_path: Path,
) -> Path:
    """Resolve the actual symbol face used by both drawing and audit metadata."""

    if Path(math_symbol_path).is_file():
        return Path(math_symbol_path)
    if Path(times_path).is_file():
        return Path(times_path)
    return Path(song_path)


def create_staging_directory(output_dir: Path) -> Path:
    """Create a same-parent staging directory that inherits the parent's ACL."""
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(8):
        candidate = parent / f".{output_dir.name}.render-{uuid.uuid4().hex}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"无法为 {output_dir} 创建唯一渲染暂存目录")


def make_fonts(
    size: int,
    *,
    song_path: Path,
    song_bold_path: Path,
    times_path: Path,
    times_bold_path: Path,
    math_symbol_path: Path = DEFAULT_MATH_SYMBOL,
) -> Fonts:
    # The original test3 target is Windows. Server deployments may not have
    # Segoe UI Symbol; use an already validated face as a safe fallback.
    math_symbol_path = resolve_math_symbol_path(
        math_symbol_path,
        times_path=times_path,
        song_path=song_path,
    )
    song = ImageFont.truetype(str(song_path), size)
    select_named_variation(song, "Regular", "Normal", "Roman")
    song_bold = ImageFont.truetype(str(song_bold_path), size)
    song_bold_variation = select_named_variation(
        song_bold,
        "Bold",
        "SemiBold",
        "Medium",
    )
    times = ImageFont.truetype(str(times_path), size)
    times_bold = ImageFont.truetype(str(times_bold_path), size)
    formula_size = FORMULA_SIZE if size == BODY_SIZE else size
    return Fonts(
        song=song,
        song_bold=song_bold,
        song_bold_is_distinct=(
            Path(song_path).resolve() != Path(song_bold_path).resolve()
            or song_bold_variation
        ),
        times=times,
        times_bold=times_bold,
        formula=ImageFont.truetype(str(times_path), formula_size),
        formula_song=ImageFont.truetype(str(song_path), formula_size),
        script=ImageFont.truetype(
            str(times_path),
            max(16, round(formula_size * 0.67)),
        ),
        script_song=ImageFont.truetype(
            str(song_path),
            max(16, round(formula_size * 0.67)),
        ),
        # Times New Roman exposes a visible .notdef square for U+226A on the
        # target Windows image.  Segoe UI Symbol contains the actual textbook
        # glyph; route only the affected relation symbol through this face.
        math_symbol=ImageFont.truetype(str(math_symbol_path), formula_size),
    )


def is_song_character(character: str) -> bool:
    code = ord(character)
    return (
        code >= 0x2E80
        # Enclosed CJK/number markers such as ①～⑳ are part of printed
        # Chinese question numbering.  Times New Roman exposes only a tofu
        # fallback for these on the target Windows image, while SimSun has
        # the intended glyphs.
        or 0x2460 <= code <= 0x24FF
        or 0x2000 <= code <= 0x206F
        or character in "，。；：！？、（）【】《》“”‘’·—…"
    )


def char_font(fonts: Fonts, character: str, *, bold_latin: bool = False) -> Any:
    if character in MATH_SYMBOL_CHARACTERS:
        return fonts.math_symbol
    if is_song_character(character):
        return fonts.song_bold if bold_latin else fonts.song
    return fonts.times_bold if bold_latin else fonts.times


def atom_width(draw: ImageDraw.ImageDraw, atom: Atom) -> float:
    return atom.width


def plain_atoms(
    draw: ImageDraw.ImageDraw,
    value: str,
    fonts: Fonts,
    *,
    bold_latin: bool = False,
) -> list[Atom]:
    # Historical model output occasionally placed one Greek command in prose
    # without the required ``$...$`` boundary (for example ``\alpha 为...``).
    # Preserve the intended textbook symbol locally instead of exposing a raw
    # backslash command on the student page.  Structural formulae remain the
    # responsibility of the formula renderer and are not rewritten here.
    value = re.sub(
        r"\\([A-Za-z]+)",
        lambda match: UNWRAPPED_GREEK_COMMANDS.get(
            match.group(1),
            match.group(0),
        ),
        value,
    )
    atoms: list[Atom] = []

    def append_plain(segment: str) -> None:
        for character in segment:
            if character == "\n":
                atoms.append(Atom("break", "", 0.0))
                continue
            font = char_font(fonts, character, bold_latin=bold_latin)
            atoms.append(
                Atom(
                    "plain_bold" if bold_latin else "plain",
                    character,
                    _safe_textlength(draw, character, font=font),
                )
            )

    position = 0
    for match in PLAIN_MATH_TOKEN.finditer(value):
        append_plain(value[position : match.start()])
        atoms.extend(mixed_formula_atoms(draw, match.group(0), fonts))
        position = match.end()
    append_plain(value[position:])
    return atoms


def markup_atoms(
    draw: ImageDraw.ImageDraw,
    value: str,
    fonts: Fonts,
    *,
    bold_latin: bool = False,
) -> list[Atom]:
    atoms: list[Atom] = []
    position = 0
    for match in FORMULA_MARKER.finditer(value):
        atoms.extend(
            plain_atoms(
                draw,
                value[position:match.start()],
                fonts,
                bold_latin=bold_latin,
            )
        )
        atoms.extend(mixed_formula_atoms(draw, match.group(1).strip(), fonts))
        position = match.end()
    atoms.extend(
        plain_atoms(
            draw,
            value[position:],
            fonts,
            bold_latin=bold_latin,
        )
    )
    return atoms


def reflow_choice_options(
    draw: ImageDraw.ImageDraw,
    value: str,
    fonts: Fonts,
    *,
    max_width: int,
) -> str:
    """Lay out consecutive A/B/... choices as two columns or one per row."""

    matches = list(CHOICE_OPTION_MARKER.finditer(value))
    if len(matches) < 2:
        return value

    labels: list[str] = []
    for match in matches:
        raw = match.group(1)
        label = (
            chr(ord(raw) - ord("Ａ") + ord("A"))
            if "Ａ" <= raw <= "Ｈ"
            else raw.upper()
        )
        labels.append(label)
    expected = [chr(ord("A") + index) for index in range(len(labels))]
    if labels != expected:
        return value

    prefix = value[: matches[0].start()]
    content_end = len(value.rstrip())
    option_formula_group = (
        prefix.rstrip().endswith("$")
        and content_end > 0
        and value[:content_end].endswith("$")
    )
    if option_formula_group:
        prefix = prefix.rstrip()[:-1].rstrip()
        content_end -= 1

    options: list[str] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else content_end
        body = value[match.end() : end].strip()
        option = f"{labels[index]}. {body}".rstrip()
        if option_formula_group:
            option = f"${option}$"
        options.append(option)

    rows: list[str] = []
    for index in range(0, len(options), 2):
        pair = options[index : index + 2]
        if len(pair) == 1:
            rows.append(pair[0])
            continue
        first_width = sum(
            atom.width
            for atom in markup_atoms(draw, pair[0], fonts)
            if atom.kind != "break"
        )
        second_width = sum(
            atom.width
            for atom in markup_atoms(draw, pair[1], fonts)
            if atom.kind != "break"
        )
        space_width = max(
            1.0,
            sum(atom.width for atom in markup_atoms(draw, " ", fonts)),
        )
        second_column = max_width / 2
        gap_count = max(
            CHOICE_OPTION_MIN_GAP,
            int(math.ceil((second_column - first_width) / space_width)),
        )
        candidate = pair[0] + (" " * gap_count) + pair[1]
        rendered_width = first_width + gap_count * space_width + second_width
        if rendered_width <= max_width:
            rows.append(candidate)
        else:
            rows.extend(pair)

    option_text = "\n".join(rows)
    return f"{prefix.rstrip()}\n{option_text}" if prefix.strip() else option_text


def normalize_formula_source(value: str) -> str:
    """Remove LaTeX text wrappers before routing glyphs to fallback fonts."""
    value = re.sub(r"\\(?:text|mathrm)\{([^{}]+)\}", r"\1", value)
    return value


def normalize_caption_markup(value: str) -> str:
    """Promote simple raw graph-label notation into textbook math markup."""
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

    def normalize_plain(segment: str) -> str:
        segment = re.sub(
            r"([A-Za-z])([⁰¹²³⁴⁵⁶⁷⁸⁹]+)",
            lambda match: f"{match.group(1)}^{match.group(2).translate(superscripts)}",
            segment,
        )

        def promote(match: re.Match[str]) -> str:
            formula = match.group(0)
            formula = re.sub(
                r"\^([A-Za-z0-9]+)",
                lambda item: "^{" + (r"\circ" if item.group(1) == "circ" else item.group(1)) + "}",
                formula,
            )
            formula = re.sub(r"_([A-Za-z0-9]+)", r"_{\1}", formula)
            return f"${formula}$"

        return RAW_CAPTION_FORMULA.sub(promote, segment)

    pieces = value.split("$")
    for index in range(0, len(pieces), 2):
        pieces[index] = normalize_plain(pieces[index])
    return "$".join(pieces)


def mixed_formula_segment_atoms(
    draw: ImageDraw.ImageDraw,
    value: str,
    fonts: Fonts,
) -> list[Atom]:
    formula = clean_text(value)
    # The shared plain-text sanitizer spells ``\ll`` as Chinese so it remains
    # readable in renderers without mathematical glyph coverage.  This
    # renderer has an explicit SimSun formula fallback, so preserve the
    # conventional relation symbol used by the source question.
    if re.search(r"(?:\\)+ll(?![A-Za-z])", value):
        formula = formula.replace("远小于", "≪")
    if not formula:
        return []
    compact_formula = re.sub(r"\s+", "", formula)
    simple_fraction = SIMPLE_SLASH_FRACTION.fullmatch(compact_formula)
    if simple_fraction:
        formula = clean_text(
            rf"\frac{{{simple_fraction.group('numerator')}}}"
            rf"{{{simple_fraction.group('denominator')}}}"
        )

    # A short displayed relation is one semantic unit. Keeping it in one atom
    # moves the whole relation to the next line instead of leaving fragments
    # such as ``a=`` or ``AB=PQ=`` at the right margin.
    complete_width = rich_text_width(
        draw,
        formula,
        fonts.formula,
        fonts.script,
        fallback_body_font=fonts.formula_song,
        fallback_script_font=fonts.script_song,
    )
    if complete_width <= CONTENT_WIDTH - 12 and "≪" not in formula:
        return [Atom("formula", formula, float(complete_width))]

    # Keep rich fraction/radical tokens intact, but allow long equations to
    # wrap after safe top-level operators.  In particular, do not split on
    # Chinese text before ``clean_text`` has parsed ``\frac{中文}{中文}``.
    tokens = rich_formula_atoms(formula)
    chunks: list[str] = []
    current: list[str] = []
    for token in tokens:
        current.append(token)
        if token in {",", "，", ";", "；", "=", "+", "∝", "≤", "≥", "≈"}:
            chunks.append("".join(current))
            current = []
    if current:
        chunks.append("".join(current))
    output: list[Atom] = []
    for chunk in chunks:
        # Keep ordinary Latin/Greek formulae in Times New Roman, but split the
        # one relation glyph known to render as a .notdef square in that font.
        # Treating it as a plain atom preserves the shared line baseline while
        # ``char_font`` routes it to Segoe UI Symbol.
        for segment in re.split(r"(≪)", chunk):
            if not segment:
                continue
            if segment in MATH_SYMBOL_CHARACTERS:
                output.append(
                    Atom(
                        "plain",
                        segment,
                        _safe_textlength(draw, segment, font=fonts.math_symbol),
                    )
                )
                continue
            width = rich_text_width(
                draw,
                segment,
                fonts.formula,
                fonts.script,
                fallback_body_font=fonts.formula_song,
                fallback_script_font=fonts.script_song,
            )
            output.append(Atom("formula", segment, float(width)))
    return output


def formula_script_runs(value: str, fonts: Fonts) -> list[tuple[str, Any]]:
    """Route mixed CJK/digit scripts to fonts without splitting the script."""
    runs: list[tuple[str, Any]] = []
    for character in value:
        font = fonts.script_song if is_song_character(character) else fonts.script
        if runs and runs[-1][1] is font:
            runs[-1] = (runs[-1][0] + character, font)
        else:
            runs.append((character, font))
    return runs


def formula_script_width(
    draw: ImageDraw.ImageDraw,
    value: str,
    fonts: Fonts,
) -> float:
    return float(
        sum(
            _safe_textlength(draw, run, font=font)
            for run, font in formula_script_runs(value, fonts)
        )
    )


def draw_formula_script_at_baseline(
    draw: ImageDraw.ImageDraw,
    x: float,
    baseline: float,
    value: str,
    fonts: Fonts,
) -> float:
    """Draw all font runs of one script on exactly the same baseline."""
    cursor = x
    for run, font in formula_script_runs(value, fonts):
        draw.text(
            (cursor, baseline),
            run,
            fill=0,
            font=font,
            anchor="ls",
        )
        cursor += _safe_textlength(draw, run, font=font)
    return cursor


def mixed_formula_atoms(
    draw: ImageDraw.ImageDraw,
    value: str,
    fonts: Fonts,
) -> list[Atom]:
    """Parse the complete formula before applying Times/SimSun fallbacks."""
    return mixed_formula_segment_atoms(draw, value, fonts)


def wrap_markup(
    draw: ImageDraw.ImageDraw,
    value: str,
    fonts: Fonts,
    *,
    max_width: int,
    line_height: int,
    left: int = 0,
    bold_latin: bool = False,
    font_role: str = "body",
) -> list[RenderLine]:
    source = markup_atoms(draw, value, fonts, bold_latin=bold_latin)
    lines: list[RenderLine] = []
    current: list[Atom] = []
    width = 0.0

    def make_line(atoms: Sequence[Atom]) -> RenderLine:
        body_ascent = font_ascent(
            fonts.song,
            int(getattr(fonts.song, "size", BODY_SIZE)),
        )
        formula_bounds = [
            formula_vertical_bounds(
                atom.value,
                fonts.formula,
                fonts.script,
                fallback_body_font=fonts.formula_song,
                fallback_script_font=fonts.script_song,
            )
            for atom in atoms
            if atom.kind == "formula"
        ]
        if not formula_bounds:
            return RenderLine(tuple(atoms), line_height, left, font_role, 0)
        plain_bounds = [
            font_vertical_bounds(
                char_font(
                    fonts,
                    atom.value,
                    bold_latin=atom.kind == "plain_bold",
                ),
                atom.value,
                fallback_ascent=body_ascent,
            )
            for atom in atoms
            if atom.kind in {"plain", "plain_bold"} and atom.value.strip()
        ]
        all_bounds = formula_bounds + plain_bounds
        ink_top = min(body_ascent + top for top, _ in all_bounds)
        ink_bottom = max(body_ascent + bottom for _, bottom in all_bounds)
        safe_gap = 4
        top_pad = max(0, int(math.ceil(-ink_top)) + safe_gap)
        measured_height = top_pad + int(math.ceil(ink_bottom)) + safe_gap
        return RenderLine(
            tuple(atoms),
            max(line_height, measured_height),
            left,
            font_role,
            top_pad,
        )

    for atom in source:
        if atom.kind == "break":
            lines.append(make_line(current))
            current = []
            width = 0.0
            continue
        if current and width + atom_width(draw, atom) > max_width:
            if (
                len(current) > 1
                and current[-1].kind.startswith("plain")
                and current[-1].value in FORBIDDEN_LINE_END
            ):
                carry = current.pop()
                width -= atom_width(draw, carry)
                lines.append(make_line(current))
                current = [carry]
                width = atom_width(draw, carry)
            elif atom.kind.startswith("plain") and atom.value in FORBIDDEN_LINE_START:
                if len(current) > 1:
                    carry = current.pop()
                    width -= atom_width(draw, carry)
                    lines.append(make_line(current))
                    current = [carry]
                    width = atom_width(draw, carry)
            else:
                lines.append(make_line(current))
                current = []
                width = 0.0
        current.append(atom)
        width += atom_width(draw, atom)
    if current or not lines:
        lines.append(make_line(current))

    # Avoid a single glyph dangling on the final line of one paragraph. Move a
    # small semantic tail from the preceding line; formula atoms stay intact.
    if len(lines) >= 2:
        previous = list(lines[-2].atoms)
        final = list(lines[-1].atoms)

        def is_semantic(atom: Atom) -> bool:
            if atom.kind == "formula":
                return True
            return atom.kind.startswith("plain") and bool(
                SUBSTANTIVE_GUIDANCE_CONTENT.search(atom.value)
            )

        final_semantic = [atom for atom in final if is_semantic(atom)]
        if (
            len(final_semantic) == 1
            and final_semantic[0].kind.startswith("plain")
            and re.search(r"[A-Za-z0-9\u3400-\u9fff]", final_semantic[0].value)
        ):
            moved: list[Atom] = []
            while (
                sum(1 for atom in moved + final if is_semantic(atom)) < 3
                and sum(1 for atom in previous if is_semantic(atom)) > 3
            ):
                moved.insert(0, previous.pop())
            while previous and not previous[-1].value.strip():
                moved.insert(0, previous.pop())
            if moved:
                lines[-2] = make_line(previous)
                lines[-1] = make_line(moved + final)
    return lines


def draw_atoms(
    draw: ImageDraw.ImageDraw,
    line: RenderLine,
    *,
    x: int,
    y: int,
    fonts: Fonts,
) -> None:
    cursor = float(x + line.left)
    content_y = y + line.top_pad
    common_baseline = content_y + font_ascent(
        fonts.song,
        int(getattr(fonts.song, "size", BODY_SIZE)),
    )
    index = 0
    while index < len(line.atoms):
        atom = line.atoms[index]
        if atom.kind in {"formula_cjk_subscript", "formula_cjk_superscript"}:
            script_baseline = (
                common_baseline + 9
                if atom.kind == "formula_cjk_subscript"
                else common_baseline - 13
            )
            cursor = draw_formula_script_at_baseline(
                draw,
                cursor,
                script_baseline,
                atom.value,
                fonts,
            )
            index += 1
            continue
        if atom.kind == "formula":
            cursor = draw_formula_at_baseline(
                draw,
                cursor,
                common_baseline,
                atom.value,
                body_font=fonts.formula,
                script_font=fonts.script,
                fallback_body_font=fonts.formula_song,
                fallback_script_font=fonts.script_song,
                fill=0,
            )
            index += 1
            continue
        bold = atom.kind == "plain_bold"
        font = char_font(fonts, atom.value, bold_latin=bold)
        run = [atom.value]
        next_index = index + 1
        while next_index < len(line.atoms):
            candidate = line.atoms[next_index]
            if candidate.kind not in {"plain", "plain_bold"}:
                break
            candidate_bold = candidate.kind == "plain_bold"
            candidate_font = char_font(
                fonts,
                candidate.value,
                bold_latin=candidate_bold,
            )
            if candidate_font is not font:
                break
            run.append(candidate.value)
            next_index += 1
        run_text = "".join(run)
        chinese_bold = bold and is_song_character(run_text[0])
        draw.text(
            (cursor, common_baseline),
            run_text,
            fill=0,
            font=font,
            anchor="ls",
            stroke_width=(
                1 if chinese_bold and not fonts.song_bold_is_distinct else 0
            ),
            stroke_fill=(
                0 if chinese_bold and not fonts.song_bold_is_distinct else None
            ),
        )
        cursor += _safe_textlength(draw, run_text, font=font)
        index = next_index


def blank(height: int) -> Image.Image:
    return Image.new("L", (WIDTH, max(1, height)), 255)


def finish_png(image: Image.Image, path: Path) -> dict[str, Any]:
    document_image = ImageOps.grayscale(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    document_image.save(path, format="PNG", dpi=(PNG_DPI, PNG_DPI), optimize=True)
    with Image.open(path) as check:
        check.load()
        width, height = check.size
        mode = check.mode
        image_format = check.format
        dpi = check.info.get("dpi")
    if width != WIDTH or height <= 0 or mode != "L" or image_format != "PNG":
        raise ValueError(f"渲染图片合同不满足：{path}")
    if (
        not isinstance(dpi, tuple)
        or len(dpi) < 2
        or abs(float(dpi[0]) - PNG_DPI) > PNG_DPI_QUANTIZATION_TOLERANCE
        or abs(float(dpi[1]) - PNG_DPI) > PNG_DPI_QUANTIZATION_TOLERANCE
    ):
        raise ValueError(f"渲染图片缺少9cm展示宽度所需DPI：{path}")
    physical_width_cm = width / float(dpi[0]) * 2.54
    if (
        abs(physical_width_cm - DISPLAY_WIDTH_CM)
        > PNG_PHYSICAL_WIDTH_TOLERANCE_CM
    ):
        raise ValueError(f"渲染图片实际展示宽度不是9cm：{path}")
    return {
        "path": str(path.resolve()),
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
        "display_width_cm": DISPLAY_WIDTH_CM,
        "physical_width_cm_from_png_dpi": round(physical_width_cm, 4),
        "dpi": list(dpi) if isinstance(dpi, tuple) else None,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def crop_question(image: Image.Image) -> Image.Image:
    source = ImageOps.exif_transpose(image).convert("RGB")
    grayscale = ImageOps.grayscale(source)
    mask = grayscale.point(lambda value: 255 if value < 248 else 0)
    detected = mask.getbbox()
    if detected is not None:
        left, top, right, bottom = detected
        padding = 18
        detected = (
            max(0, left - padding),
            max(0, top - padding),
            min(source.width, right + padding),
            min(source.height, bottom + padding),
        )
        source = source.crop(detected)
    target_width = CONTENT_WIDTH
    scale = target_width / source.width
    target_height = max(1, round(source.height * scale))
    return source.resize((target_width, target_height), Image.Resampling.LANCZOS)


def projection_runs(flags: Sequence[bool], *, max_gap: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last_true: int | None = None
    for index, flag in enumerate(flags):
        if not flag:
            continue
        if start is None or last_true is None or index - last_true - 1 > max_gap:
            if start is not None and last_true is not None:
                runs.append((start, last_true + 1))
            start = index
        last_true = index
    if start is not None and last_true is not None:
        runs.append((start, last_true + 1))
    return runs


SOURCE_FIGURE_QUALITY_INK_THRESHOLD = 242
SOURCE_FIGURE_PROJECTION_INK_THRESHOLD = 210
SOURCE_FIGURE_MIN_INK_PIXELS = 24
SOURCE_FIGURE_MIN_INK_RATIO = 0.0008
SOURCE_FIGURE_MAX_SHRINK_FRACTION = 0.08
SOURCE_FIGURE_MAX_EXPAND_FRACTION = 0.18


def source_figure_ink_metrics(
    grayscale: Image.Image,
    pixel_box: tuple[int, int, int, int],
    *,
    ink_threshold: int = SOURCE_FIGURE_QUALITY_INK_THRESHOLD,
) -> dict[str, Any]:
    """Measure whether a proposed crop contains enough visible source ink."""
    left, top, right, bottom = pixel_box
    region = grayscale.crop(pixel_box)
    area = max(0, right - left) * max(0, bottom - top)
    histogram = region.histogram()
    ink_pixels = sum(histogram[:ink_threshold]) if area else 0
    ink_ratio = ink_pixels / area if area else 0.0
    minimum_ink_pixels = max(
        SOURCE_FIGURE_MIN_INK_PIXELS,
        math.ceil(area * SOURCE_FIGURE_MIN_INK_RATIO),
    )
    local_ink_bbox = region.point(
        lambda value: 255 if value < ink_threshold else 0
    ).getbbox()
    absolute_ink_bbox = None
    if local_ink_bbox is not None:
        absolute_ink_bbox = [
            left + local_ink_bbox[0],
            top + local_ink_bbox[1],
            left + local_ink_bbox[2],
            top + local_ink_bbox[3],
        ]
    return {
        "area_px": area,
        "ink_pixels": ink_pixels,
        "ink_ratio": round(ink_ratio, 8),
        "minimum_ink_pixels": minimum_ink_pixels,
        "minimum_ink_ratio": SOURCE_FIGURE_MIN_INK_RATIO,
        "ink_bbox_px": absolute_ink_bbox,
        "quality_ok": area > 0 and ink_pixels >= minimum_ink_pixels,
    }


def constrain_source_figure_refinement(
    raw_box: tuple[int, int, int, int],
    model_box: tuple[int, int, int, int],
    *,
    source_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Bound local refinement so sparse diagram endpoints remain protected."""
    left, top, right, bottom = model_box
    box_width = right - left
    box_height = bottom - top
    max_shrink_x = max(2, round(box_width * SOURCE_FIGURE_MAX_SHRINK_FRACTION))
    max_shrink_y = max(2, round(box_height * SOURCE_FIGURE_MAX_SHRINK_FRACTION))
    max_expand_x = max(4, round(box_width * SOURCE_FIGURE_MAX_EXPAND_FRACTION))
    max_expand_y = max(4, round(box_height * SOURCE_FIGURE_MAX_EXPAND_FRACTION))
    source_width, source_height = source_size
    raw_left, raw_top, raw_right, raw_bottom = raw_box

    bounded = (
        max(0, min(max(raw_left, left - max_expand_x), left + max_shrink_x)),
        max(0, min(max(raw_top, top - max_expand_y), top + max_shrink_y)),
        min(
            source_width,
            max(min(raw_right, right + max_expand_x), right - max_shrink_x),
        ),
        min(
            source_height,
            max(min(raw_bottom, bottom + max_expand_y), bottom - max_shrink_y),
        ),
    )
    if bounded[0] >= bounded[2] or bounded[1] >= bounded[3]:
        bounded = model_box

    bounded_left, bounded_top, bounded_right, bounded_bottom = bounded
    return bounded, {
        "max_shrink_px": {"x": max_shrink_x, "y": max_shrink_y},
        "max_expand_px": {"x": max_expand_x, "y": max_expand_y},
        "shrink_px": {
            "left": max(0, bounded_left - left),
            "top": max(0, bounded_top - top),
            "right": max(0, right - bounded_right),
            "bottom": max(0, bottom - bounded_bottom),
        },
        "expand_px": {
            "left": max(0, left - bounded_left),
            "top": max(0, top - bounded_top),
            "right": max(0, bounded_right - right),
            "bottom": max(0, bounded_bottom - bottom),
        },
    }


def refine_source_figure_box(
    source: Image.Image,
    pixel_box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Safely tighten a coarse model box around the dominant diagram ink."""
    left, top, right, bottom = pixel_box
    box_width = right - left
    box_height = bottom - top
    # Vision models can place a small isolated diagram just outside a coarse
    # bbox, especially near a page edge.  The search window is deliberately
    # wider than the normal refinement limit; it is used for recovery only
    # when the proposed box itself contains no usable ink.
    x_padding = max(24, round(box_width * 1.0))
    y_padding = max(20, round(box_height * 0.6))
    search_left = max(0, left - x_padding)
    search_right = min(source.width, right + x_padding)
    search_top = max(0, top - y_padding)
    search_bottom = min(source.height, bottom + y_padding)
    grayscale = ImageOps.grayscale(source)
    model_metrics = source_figure_ink_metrics(grayscale, pixel_box)
    base_audit: dict[str, Any] = {
        "ink_threshold": SOURCE_FIGURE_QUALITY_INK_THRESHOLD,
        "projection_ink_threshold": SOURCE_FIGURE_PROJECTION_INK_THRESHOLD,
        "model_bbox_px": list(pixel_box),
        "model_ink_pixels": model_metrics["ink_pixels"],
        "model_ink_ratio": model_metrics["ink_ratio"],
        "model_ink_bbox_px": model_metrics["ink_bbox_px"],
        "minimum_ink_pixels": model_metrics["minimum_ink_pixels"],
        "minimum_ink_ratio": model_metrics["minimum_ink_ratio"],
        "model_quality_ok": model_metrics["quality_ok"],
        "search_bbox_px": [search_left, search_top, search_right, search_bottom],
    }
    if not model_metrics["quality_ok"]:
        search_box = (search_left, search_top, search_right, search_bottom)
        search_metrics = source_figure_ink_metrics(grayscale, search_box)
        recovered_ink_box = search_metrics["ink_bbox_px"]
        if search_metrics["quality_ok"] and recovered_ink_box is not None:
            recovered_left, recovered_top, recovered_right, recovered_bottom = (
                int(value) for value in recovered_ink_box
            )
            recovery_padding = max(
                8,
                round(min(
                    recovered_right - recovered_left,
                    recovered_bottom - recovered_top,
                ) * 0.05),
            )
            recovered_box = (
                max(0, recovered_left - recovery_padding),
                max(0, recovered_top - recovery_padding),
                min(source.width, recovered_right + recovery_padding),
                min(source.height, recovered_bottom + recovery_padding),
            )
            recovered_metrics = source_figure_ink_metrics(
                grayscale,
                recovered_box,
            )
            if recovered_metrics["quality_ok"]:
                return recovered_box, {
                    **base_audit,
                    "mode": "nearby_ink_recovered",
                    "reason": "model_bbox_empty_but_nearby_search_has_visible_ink",
                    "quality_ok": True,
                    "raw_refined_bbox_px": list(recovered_box),
                    "refined_bbox_px": list(recovered_box),
                    "refined_ink_pixels": recovered_metrics["ink_pixels"],
                    "refined_ink_ratio": recovered_metrics["ink_ratio"],
                    "refined_ink_bbox_px": recovered_metrics["ink_bbox_px"],
                    "search_ink_pixels": search_metrics["ink_pixels"],
                    "search_ink_ratio": search_metrics["ink_ratio"],
                    "boundary_limits_px": None,
                }
        return pixel_box, {
            **base_audit,
            "mode": "model_bbox_rejected_low_ink",
            "reason": "model_bbox_has_insufficient_visible_ink",
            "quality_ok": False,
            "raw_refined_bbox_px": list(pixel_box),
            "refined_bbox_px": list(pixel_box),
            "refined_ink_pixels": model_metrics["ink_pixels"],
            "refined_ink_ratio": model_metrics["ink_ratio"],
            "boundary_limits_px": None,
        }

    def finalize(
        raw_box: tuple[int, int, int, int],
        *,
        mode: str,
        reason: str,
    ) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
        bounded_box, limits = constrain_source_figure_refinement(
            raw_box,
            pixel_box,
            source_size=source.size,
        )
        refined_metrics = source_figure_ink_metrics(grayscale, bounded_box)
        final_mode = mode
        final_reason = reason
        if not refined_metrics["quality_ok"]:
            bounded_box = pixel_box
            refined_metrics = model_metrics
            final_mode = "model_bbox_fallback"
            final_reason = "bounded_refinement_lost_too_much_visible_ink"
            _, limits = constrain_source_figure_refinement(
                pixel_box,
                pixel_box,
                source_size=source.size,
            )
        return bounded_box, {
            **base_audit,
            "mode": final_mode,
            "reason": final_reason,
            "quality_ok": True,
            "raw_refined_bbox_px": list(raw_box),
            "refined_bbox_px": list(bounded_box),
            "refined_ink_pixels": refined_metrics["ink_pixels"],
            "refined_ink_ratio": refined_metrics["ink_ratio"],
            "refined_ink_bbox_px": refined_metrics["ink_bbox_px"],
            "boundary_limits_px": limits,
        }

    pixels = grayscale.load()
    ink_threshold = SOURCE_FIGURE_PROJECTION_INK_THRESHOLD

    row_counts = [
        sum(
            1
            for x in range(search_left, search_right)
            if pixels[x, y] < ink_threshold
        )
        for y in range(search_top, search_bottom)
    ]
    row_minimum = max(4, round((search_right - search_left) * 0.012))
    row_runs = projection_runs(
        [count >= row_minimum for count in row_counts],
        max_gap=5,
    )
    row_candidates: list[tuple[int, int]] = []
    for relative_top, relative_bottom in row_runs:
        absolute_top = search_top + relative_top
        absolute_bottom = search_top + relative_bottom
        if absolute_bottom <= top or absolute_top >= bottom:
            continue
        if absolute_bottom - absolute_top >= 8:
            row_candidates.append((absolute_top, absolute_bottom))
    if not row_candidates:
        return finalize(
            pixel_box,
            mode="model_bbox_fallback",
            reason="no_dominant_row_component",
        )
    refined_top, refined_bottom = max(
        row_candidates,
        key=lambda run: (
            run[1] - run[0],
            sum(row_counts[run[0] - search_top:run[1] - search_top]),
        ),
    )

    column_counts = [
        sum(
            1
            for y in range(refined_top, refined_bottom)
            if pixels[x, y] < ink_threshold
        )
        for x in range(search_left, search_right)
    ]
    column_minimum = max(3, round((refined_bottom - refined_top) * 0.012))
    column_runs = projection_runs(
        [count >= column_minimum for count in column_counts],
        max_gap=16,
    )
    column_candidates: list[tuple[int, int]] = []
    for relative_left, relative_right in column_runs:
        absolute_left = search_left + relative_left
        absolute_right = search_left + relative_right
        if absolute_right <= left or absolute_left >= right:
            continue
        if absolute_right - absolute_left >= 4:
            column_candidates.append((absolute_left, absolute_right))
    if not column_candidates:
        return finalize(
            pixel_box,
            mode="model_bbox_fallback",
            reason="no_dominant_column_component",
        )
    refined_left, refined_right = max(
        column_candidates,
        key=lambda run: (
            sum(column_counts[run[0] - search_left:run[1] - search_left]),
            max(0, min(run[1], right) - max(run[0], left)),
            run[1] - run[0],
        ),
    )
    padding = max(8, round(min(refined_right - refined_left, refined_bottom - refined_top) * 0.05))
    raw_refined = (
        max(0, refined_left - padding),
        max(0, refined_top - padding),
        min(source.width, refined_right + padding),
        min(source.height, refined_bottom + padding),
    )
    return finalize(
        raw_refined,
        mode="model_bbox_local_ink_refined_bounded",
        reason="dominant_component_refined_with_endpoint_safety_limits",
    )


def crop_source_figure(
    image: Image.Image,
    bbox_1000: Sequence[Any],
    *,
    lock_bbox: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    if (
        len(bbox_1000) != 4
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            for value in bbox_1000
        )
    ):
        raise ValueError("题内图片bbox_1000必须含4个数")
    left, top, right, bottom = (float(value) for value in bbox_1000)
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        raise ValueError("题内图片bbox_1000越界或面积无效")
    area_ratio = (right - left) * (bottom - top) / 1_000_000
    if area_ratio < 0.001 or area_ratio > 0.55:
        raise ValueError("题内图片bbox_1000过小或疑似夹带正文/整页原图")
    source = ImageOps.exif_transpose(image).convert("RGB")
    pixel_box = (
        max(0, math.floor(left * source.width / 1000)),
        max(0, math.floor(top * source.height / 1000)),
        min(source.width, math.ceil(right * source.width / 1000)),
        min(source.height, math.ceil(bottom * source.height / 1000)),
    )
    refined_box, audit = refine_source_figure_box(source, pixel_box)
    if lock_bbox:
        automatic_refinement_audit = {
            key: audit.get(key)
            for key in (
                "raw_refined_bbox_px",
                "refined_bbox_px",
                "refined_ink_pixels",
                "refined_ink_ratio",
                "refined_ink_bbox_px",
                "boundary_limits_px",
            )
        }
        audit["auto_refined_bbox_px"] = list(refined_box)
        audit["automatic_refinement_audit"] = automatic_refinement_audit
        refined_box = pixel_box
        # All top-level ``refined_*`` fields describe the bbox that is
        # actually consumed below.  The unused automatic suggestion and its
        # metrics remain together in ``automatic_refinement_audit``.
        audit["raw_refined_bbox_px"] = list(refined_box)
        audit["refined_bbox_px"] = list(refined_box)
        audit["refined_ink_pixels"] = audit.get("model_ink_pixels", 0)
        audit["refined_ink_ratio"] = audit.get("model_ink_ratio", 0.0)
        audit["refined_ink_bbox_px"] = audit.get("model_ink_bbox_px")
        audit["boundary_limits_px"] = {
            "max_shrink_px": {"x": 0, "y": 0},
            "max_expand_px": {"x": 0, "y": 0},
            "shrink_px": {"left": 0, "top": 0, "right": 0, "bottom": 0},
            "expand_px": {"left": 0, "top": 0, "right": 0, "bottom": 0},
            "applied": False,
            "reason": "reviewed_bbox_locked",
        }
        audit["mode"] = "reviewed_bbox_locked"
        audit["reason"] = "human_reviewed_bbox_bound_to_source_sha256"
        audit["quality_ok"] = True
        audit["reviewed_bbox_locked"] = True
        audit["refinement_applied"] = False
    else:
        audit["reviewed_bbox_locked"] = False
        audit["refinement_applied"] = True
    if not audit.get("quality_ok", False):
        error = ValueError(
            "source figure crop rejected: model bbox has insufficient visible "
            f"ink ({audit.get('model_ink_pixels', 0)}/"
            f"{audit.get('minimum_ink_pixels', 0)} pixels)"
        )
        error.audit = audit  # type: ignore[attr-defined]
        raise error
    crop = source.crop(refined_box)
    source_crop_size = crop.size
    width_scale_limit = min(CONTENT_WIDTH / crop.width, 1.6)
    pre_height_limit_target = (
        max(1, round(crop.width * width_scale_limit)),
        max(1, round(crop.height * width_scale_limit)),
    )
    height_scale_limit = SOURCE_FIGURE_MAX_OUTPUT_HEIGHT / crop.height
    scale = min(width_scale_limit, height_scale_limit)
    target = (
        max(1, round(crop.width * scale)),
        max(1, round(crop.height * scale)),
    )
    if target != crop.size:
        crop = crop.resize(target, Image.Resampling.LANCZOS)
    audit["model_bbox_1000"] = [left, top, right, bottom]
    audit["refined_bbox_1000"] = [
        round(refined_box[0] * 1000 / source.width, 3),
        round(refined_box[1] * 1000 / source.height, 3),
        round(refined_box[2] * 1000 / source.width, 3),
        round(refined_box[3] * 1000 / source.height, 3),
    ]
    height_limit_applied = pre_height_limit_target[1] > SOURCE_FIGURE_MAX_OUTPUT_HEIGHT
    audit["source_crop_size_px"] = list(source_crop_size)
    audit["pre_height_limit_size_px"] = list(pre_height_limit_target)
    audit["source_figure_max_output_height_px"] = SOURCE_FIGURE_MAX_OUTPUT_HEIGHT
    audit["width_scale_limit"] = round(width_scale_limit, 9)
    audit["height_scale_limit"] = round(height_scale_limit, 9)
    audit["selected_resize_scale"] = round(scale, 9)
    audit["height_limit_applied"] = height_limit_applied
    audit["resize_only_no_pixel_crop"] = True
    audit["device_projection"] = {
        "master_width_px": WIDTH,
        "device_width_px": DEVICE_WIDTH,
        "device_page_height_px": DEVICE_HEIGHT,
        "width_scale": DEVICE_WIDTH / WIDTH,
        "projected_figure_height_px": round(crop.height * DEVICE_WIDTH / WIDTH),
        "maximum_figure_height_px": SOURCE_FIGURE_MAX_DEVICE_HEIGHT,
        "reserved_vertical_whitespace_px": SOURCE_FIGURE_DEVICE_VERTICAL_WHITESPACE,
        "fits_one_device_page": (
            round(crop.height * DEVICE_WIDTH / WIDTH)
            <= SOURCE_FIGURE_MAX_DEVICE_HEIGHT
        ),
    }
    audit["output_crop_size_px"] = [crop.width, crop.height]
    return crop, audit


def prepare_aligned_figure_group_crops(
    source: Image.Image,
    figures: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, tuple[Image.Image, dict[str, Any]]],
    dict[str, str],
]:
    """Keep a horizontal multi-panel source figure together.

    Vision models often describe a printed ``图1/图2/图3`` row as three
    figures but give every box the same broad vertical range. Rendering those
    boxes independently duplicates surrounding text and can create mostly
    blank crops. The renderer can safely identify this geometry locally: it
    unions each member's ink-refined suggestion and emits one source-pixel
    panel without asking the model again.
    """

    prepared: dict[str, tuple[Image.Image, dict[str, Any]]] = {}
    member_to_leader: dict[str, str] = {}
    available: list[tuple[Mapping[str, Any], tuple[float, float, float, float]]] = []
    for figure in figures:
        bbox = figure.get("bbox_1000")
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            continue
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in bbox):
            continue
        left, top, right, bottom = (float(value) for value in bbox)
        if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
            continue
        available.append((figure, (left, top, right, bottom)))

    consumed: set[str] = set()
    for index, (anchor, anchor_box) in enumerate(available):
        anchor_id = str(anchor.get("figure_id") or "")
        if not anchor_id or anchor_id in consumed:
            continue
        anchor_height = anchor_box[3] - anchor_box[1]
        group = [(anchor, anchor_box)]
        for candidate, candidate_box in available[index + 1 :]:
            candidate_id = str(candidate.get("figure_id") or "")
            if not candidate_id or candidate_id in consumed:
                continue
            candidate_height = candidate_box[3] - candidate_box[1]
            overlap = max(
                0.0,
                min(anchor_box[3], candidate_box[3])
                - max(anchor_box[1], candidate_box[1]),
            )
            if overlap < 0.75 * min(anchor_height, candidate_height):
                continue
            if max(anchor_height, candidate_height) > 1.5 * min(anchor_height, candidate_height):
                continue
            group.append((candidate, candidate_box))
        if len(group) < 2:
            continue
        union_left = min(item[1][0] for item in group)
        union_right = max(item[1][2] for item in group)
        widest = max(item[1][2] - item[1][0] for item in group)
        if union_right - union_left < 1.45 * widest:
            continue

        raw_boxes: list[tuple[int, int, int, int]] = []
        member_audits: list[dict[str, Any]] = []
        for figure, bbox in group:
            pixel_box = (
                max(0, math.floor(bbox[0] * source.width / 1000)),
                max(0, math.floor(bbox[1] * source.height / 1000)),
                min(source.width, math.ceil(bbox[2] * source.width / 1000)),
                min(source.height, math.ceil(bbox[3] * source.height / 1000)),
            )
            _refined, trial_audit = refine_source_figure_box(source, pixel_box)
            raw = trial_audit.get("raw_refined_bbox_px")
            if (
                not trial_audit.get("quality_ok")
                or not isinstance(raw, (list, tuple))
                or len(raw) != 4
            ):
                raw_boxes = []
                break
            raw_box = tuple(int(value) for value in raw)
            if not (raw_box[0] < raw_box[2] and raw_box[1] < raw_box[3]):
                raw_boxes = []
                break
            raw_boxes.append(raw_box)
            member_audits.append(
                {
                    "figure_id": str(figure.get("figure_id") or ""),
                    "model_bbox_1000": list(bbox),
                    "raw_refined_bbox_px": list(raw_box),
                }
            )
        if len(raw_boxes) != len(group):
            continue
        common_top = max(box[1] for box in raw_boxes)
        common_bottom = min(box[3] for box in raw_boxes)
        minimum_height = min(box[3] - box[1] for box in raw_boxes)
        if common_bottom - common_top < 0.55 * minimum_height:
            continue
        group_box = (
            min(box[0] for box in raw_boxes),
            min(box[1] for box in raw_boxes),
            max(box[2] for box in raw_boxes),
            max(box[3] for box in raw_boxes),
        )
        bbox_1000 = [
            group_box[0] * 1000 / source.width,
            group_box[1] * 1000 / source.height,
            group_box[2] * 1000 / source.width,
            group_box[3] * 1000 / source.height,
        ]
        try:
            crop, audit = crop_source_figure(source, bbox_1000, lock_bbox=True)
        except ValueError:
            continue
        group_ids = [str(item[0].get("figure_id") or "") for item in group]
        leader = group_ids[0]
        audit.update(
            {
                "aligned_figure_group": True,
                "merged_figure_ids": group_ids,
                "member_refinement_audits": member_audits,
            }
        )
        prepared[leader] = (crop, audit)
        for figure_id in group_ids:
            member_to_leader[figure_id] = leader
            consumed.add(figure_id)
    return prepared, member_to_leader


def crop_wide_source_figure_panel(
    source: Image.Image,
    bbox_1000: Sequence[Any],
) -> tuple[Image.Image, dict[str, Any]]:
    """Tighten one broad bbox that represents a horizontal figure panel."""

    if len(bbox_1000) != 4:
        raise ValueError("wide panel bbox must contain four values")
    left, top, right, bottom = (float(value) for value in bbox_1000)
    if right - left < 750 or not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        raise ValueError("not a wide source figure panel")
    pixel_box = (
        max(0, math.floor(left * source.width / 1000)),
        max(0, math.floor(top * source.height / 1000)),
        min(source.width, math.ceil(right * source.width / 1000)),
        min(source.height, math.ceil(bottom * source.height / 1000)),
    )
    _seed_box, seed_audit = refine_source_figure_box(source, pixel_box)
    raw = seed_audit.get("raw_refined_bbox_px")
    if (
        not seed_audit.get("quality_ok")
        or not isinstance(raw, (list, tuple))
        or len(raw) != 4
    ):
        raise ValueError("wide panel has no reliable vertical figure band")
    band_top = max(0, int(raw[1]))
    band_bottom = min(source.height, int(raw[3]))
    if band_bottom - band_top < 20:
        raise ValueError("wide panel vertical figure band is too small")

    grayscale = ImageOps.grayscale(source)
    pixels = grayscale.load()
    search_left = max(0, pixel_box[0] - round((pixel_box[2] - pixel_box[0]) * 0.12))
    search_right = min(
        source.width,
        pixel_box[2] + round((pixel_box[2] - pixel_box[0]) * 0.12),
    )
    minimum_column_ink = max(3, round((band_bottom - band_top) * 0.015))
    active_columns = [
        x
        for x in range(search_left, search_right)
        if sum(
            1
            for y in range(band_top, band_bottom)
            if pixels[x, y] < SOURCE_FIGURE_PROJECTION_INK_THRESHOLD
        )
        >= minimum_column_ink
    ]
    if not active_columns:
        raise ValueError("wide panel contains no stable horizontal ink")
    panel_box = (
        max(0, min(active_columns) - 8),
        max(0, band_top - 4),
        min(source.width, max(active_columns) + 9),
        min(source.height, band_bottom + 4),
    )
    panel_bbox_1000 = [
        panel_box[0] * 1000 / source.width,
        panel_box[1] * 1000 / source.height,
        panel_box[2] * 1000 / source.width,
        panel_box[3] * 1000 / source.height,
    ]
    crop, audit = crop_source_figure(source, panel_bbox_1000, lock_bbox=True)
    audit.update(
        {
            "wide_figure_panel_refined": True,
            "original_model_bbox_1000": [left, top, right, bottom],
            "panel_seed_audit": seed_audit,
        }
    )
    return crop, audit


def detect_missing_source_figure_bbox(
    source: Image.Image,
    reference_text: str = "",
) -> list[float] | None:
    """Locate a likely diagram region when the model omitted figure metadata."""

    preview = ImageOps.grayscale(source)
    preview.thumbnail((800, 800), Image.Resampling.LANCZOS)
    mask = preview.point(lambda value: 255 if value < 225 else 0)
    mask = mask.filter(ImageFilter.MaxFilter(5))
    width, height = mask.size
    pixels = mask.load()
    seen = bytearray(width * height)
    components: list[tuple[int, int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            offset = y * width + x
            if seen[offset] or pixels[x, y] == 0:
                continue
            seen[offset] = 1
            stack = [(x, y)]
            left = right = x
            top = bottom = y
            count = 0
            while stack:
                current_x, current_y = stack.pop()
                count += 1
                left = min(left, current_x)
                right = max(right, current_x)
                top = min(top, current_y)
                bottom = max(bottom, current_y)
                for next_y in range(max(0, current_y - 1), min(height, current_y + 2)):
                    for next_x in range(max(0, current_x - 1), min(width, current_x + 2)):
                        next_offset = next_y * width + next_x
                        if seen[next_offset] or pixels[next_x, next_y] == 0:
                            continue
                        seen[next_offset] = 1
                        stack.append((next_x, next_y))
            component_width = right - left + 1
            component_height = bottom - top + 1
            if (
                component_width >= max(18, round(width * 0.025))
                and component_height >= max(18, round(height * 0.045))
                and count >= 55
                and component_height <= round(height * 0.65)
            ):
                components.append((left, top, right + 1, bottom + 1, count))
    if not components:
        return None

    groups: list[list[tuple[int, int, int, int, int]]] = []
    for component in sorted(components, key=lambda item: (item[1], item[0])):
        component_center = (component[1] + component[3]) / 2
        matched: list[tuple[int, int, int, int, int]] | None = None
        for group in groups:
            group_top = min(item[1] for item in group)
            group_bottom = max(item[3] for item in group)
            group_center = (group_top + group_bottom) / 2
            overlap = max(0, min(component[3], group_bottom) - max(component[1], group_top))
            minimum_height = min(component[3] - component[1], group_bottom - group_top)
            if overlap >= 0.2 * minimum_height or abs(component_center - group_center) <= 0.45 * max(
                component[3] - component[1], group_bottom - group_top
            ):
                matched = group
                break
        if matched is None:
            groups.append([component])
        else:
            matched.append(component)

    def group_score(group: Sequence[tuple[int, int, int, int, int]]) -> float:
        group_width = max(item[2] for item in group) - min(item[0] for item in group)
        return sum(item[4] for item in group) * (1.0 + group_width / max(1, width))

    selected = (
        max(groups, key=lambda group: max(item[3] for item in group))
        if re.search(r"图\s*丁", reference_text)
        else max(groups, key=group_score)
    )
    box = (
        min(item[0] for item in selected),
        min(item[1] for item in selected),
        max(item[2] for item in selected),
        max(item[3] for item in selected),
    )
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    tallest_component = max(item[3] - item[1] for item in selected)
    if tallest_component < box_height * 0.35 and not re.search(r"图\s*丁", reference_text):
        return None
    if (
        box_width > width * 0.85
        and len(selected) == 1
        and selected[0][2] - selected[0][0] > width * 0.85
        and not re.search(r"图\s*丁", reference_text)
    ):
        return None
    if box_width < width * 0.12 or box_height < height * 0.05:
        return None
    pad_x = max(8, round(box_width * 0.06))
    pad_y = max(8, round(box_height * 0.08))
    preview_box = (
        max(0, box[0] - pad_x),
        max(0, box[1] - pad_y),
        min(width, box[2] + pad_x),
        min(height, box[3] + pad_y),
    )
    return [
        preview_box[0] * 1000 / width,
        preview_box[1] * 1000 / height,
        preview_box[2] * 1000 / width,
        preview_box[3] * 1000 / height,
    ]


def make_content_lines(
    blocks: Iterable[tuple[str, str, int]],
    *,
    fonts: Fonts,
) -> list[RenderLine]:
    probe = blank(8)
    draw = ImageDraw.Draw(probe)
    output: list[RenderLine] = []
    for value, style, gap_after in blocks:
        if style == "section":
            active = make_fonts(
                36,
                song_path=Path(fonts.song.path),
                song_bold_path=Path(fonts.song_bold.path),
                times_path=Path(fonts.times.path),
                times_bold_path=Path(fonts.times_bold.path),
            )
            line_height = 55
            left = 0
            bold = True
        elif style == "method":
            active = fonts
            line_height = BODY_LINE
            left = 12
            bold = True
        elif style == "small":
            active = make_fonts(
                SMALL_SIZE,
                song_path=Path(fonts.song.path),
                song_bold_path=Path(fonts.song_bold.path),
                times_path=Path(fonts.times.path),
                times_bold_path=Path(fonts.times_bold.path),
            )
            line_height = SMALL_LINE
            left = 12
            bold = False
        else:
            active = fonts
            line_height = FORMULA_LINE if "$" in value else BODY_LINE
            left = 12 if style == "body_indent" else 0
            bold = False
        output.extend(
            wrap_markup(
                draw,
                value,
                active,
                max_width=CONTENT_WIDTH - left,
                line_height=line_height,
                left=left,
                bold_latin=bold,
                font_role=(
                    "section"
                    if style == "section"
                    else "small"
                    if style == "small"
                    else "body"
                ),
            )
        )
        if gap_after:
            output.append(RenderLine(tuple(), gap_after, 0))
    return output


def draw_content_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[RenderLine],
    *,
    y: int,
    body_fonts: Fonts,
    song_path: Path,
    song_bold_path: Path,
    times_path: Path,
    times_bold_path: Path,
) -> int:
    section_fonts = make_fonts(
        36,
        song_path=song_path,
        song_bold_path=song_bold_path,
        times_path=times_path,
        times_bold_path=times_bold_path,
    )
    small_fonts = make_fonts(
        SMALL_SIZE,
        song_path=song_path,
        song_bold_path=song_bold_path,
        times_path=times_path,
        times_bold_path=times_bold_path,
    )
    for line in lines:
        if not line.atoms:
            y += line.height
            continue
        if line.font_role == "section":
            active = section_fonts
        elif line.font_role == "small":
            active = small_fonts
        else:
            active = body_fonts
        draw_atoms(draw, line, x=MARGIN_X, y=y, fonts=active)
        y += line.height
    return y


def render_knowledge_card(
    *,
    source_image: Path | Image.Image,
    output: Mapping[str, Any],
    destination: Path,
    song_path: Path,
    song_bold_path: Path,
    times_path: Path,
    times_bold_path: Path,
    lock_source_figure_bboxes: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    body_fonts = make_fonts(
        BODY_SIZE,
        song_path=song_path,
        song_bold_path=song_bold_path,
        times_path=times_path,
        times_bold_path=times_bold_path,
    )
    small_fonts = make_fonts(
        SMALL_SIZE,
        song_path=song_path,
        song_bold_path=song_bold_path,
        times_path=times_path,
        times_bold_path=times_bold_path,
    )
    knowledge_blocks: list[tuple[str, str, int]] = [("知识点", "section", 4)]
    for value in output["knowledge_points"]:
        knowledge_blocks.append((f"• {value}", "body", 2))
    base_formulas = output.get("base_formulas", [])
    if base_formulas:
        knowledge_blocks.append(("基础公式", "section", 4))
        for index, item in enumerate(base_formulas, 1):
            knowledge_blocks.append((f"{index}. ${item['formula']}$", "body", 0))
            knowledge_blocks.append((f"适用：{item['condition']}", "small", 8))
    knowledge_lines = make_content_lines(knowledge_blocks, fonts=body_fonts)

    source_figures = {
        str(item["figure_id"]): item
        for item in output.get("source_figures", [])
        if isinstance(item, Mapping) and item.get("figure_id")
    }
    question_items: list[tuple[str, Any, Any]] = []
    crop_audits: list[dict[str, Any]] = []
    probe = blank(16)
    probe_draw = ImageDraw.Draw(probe)
    if isinstance(source_image, Image.Image):
        # Reviewed source-pixel repairs are supplied only as an in-memory
        # image.  Copy it again so cropping/rendering cannot mutate the
        # fail-closed repaired object owned by the caller.
        source = source_image.copy()
    else:
        with Image.open(source_image) as opened:
            opened.load()
            source = opened.copy()
    question_blocks = list(output.get("question_blocks", []))
    local_figure_ids: set[str] = set()
    question_plain_text = "\n".join(
        str(block.get("text") or "")
        for block in question_blocks
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    if not source_figures and re.search(r"(?:如图|图\s*[0-9一二三四甲乙丙丁]|图示|图像)", question_plain_text):
        detected_bbox = detect_missing_source_figure_bbox(source, question_plain_text)
        if detected_bbox is not None:
            local_figure_id = "local_detected_figure"
            source_figures[local_figure_id] = {
                "figure_id": local_figure_id,
                "bbox_1000": detected_bbox,
                "caption": "本地检测的原题图形",
            }
            question_blocks.append(
                {"type": "figure", "text": "", "figure_id": local_figure_id}
            )
            local_figure_ids.add(local_figure_id)
    grouped_crops, grouped_members = prepare_aligned_figure_group_crops(
        source,
        list(source_figures.values()),
    )
    for block in question_blocks:
        if not isinstance(block, Mapping):
            raise ValueError("原题文字块必须是对象")
        if block.get("type") == "text":
            question_text = reflow_choice_options(
                probe_draw,
                str(block.get("text") or ""),
                body_fonts,
                max_width=CONTENT_WIDTH,
            )
            lines = wrap_markup(
                probe_draw,
                question_text,
                body_fonts,
                max_width=CONTENT_WIDTH,
                line_height=BODY_LINE,
            )
            for line in lines:
                question_items.append(("line", line, None))
            question_items.append(("gap", 8, None))
            continue
        if block.get("type") != "figure":
            raise ValueError("原题块type只能是text或figure")
        figure_id = str(block.get("figure_id") or "")
        figure = source_figures.get(figure_id)
        if not isinstance(figure, Mapping):
            raise ValueError(f"原题图片块缺少裁图信息：{figure_id}")
        preserve_choice_group = "选项" in str(figure.get("caption") or "")
        grouped_leader = grouped_members.get(figure_id)
        if grouped_leader and grouped_leader != figure_id:
            crop_audits.append(
                {
                    "figure_id": figure_id,
                    "rendered_via_aligned_group": grouped_leader,
                    "non_blocking": True,
                }
            )
            continue
        if grouped_leader:
            crop, crop_audit = grouped_crops[grouped_leader]
        else:
            try:
                bbox = figure.get("bbox_1000", [])
                caption = str(figure.get("caption") or "")
                use_wide_panel = (
                    isinstance(bbox, (list, tuple))
                    and len(bbox) == 4
                    and all(isinstance(value, (int, float)) for value in bbox)
                    and float(bbox[2]) - float(bbox[0]) >= 750
                    and any(marker in caption for marker in ("图", "示意", "装置", "电路", "图像"))
                    and not preserve_choice_group
                )
                if use_wide_panel:
                    crop, crop_audit = crop_wide_source_figure_panel(source, bbox)
                else:
                    crop, crop_audit = crop_source_figure(
                        source,
                        bbox,
                        lock_bbox=(
                            lock_source_figure_bboxes
                            or preserve_choice_group
                            or figure_id in local_figure_ids
                        ),
                    )
            except ValueError as exc:
                crop_audits.append(
                    {
                        "figure_id": figure_id,
                        "skipped": True,
                        "reason": str(exc),
                        "non_blocking": True,
                    }
                )
                continue
        crop_audit["choice_group_bbox_locked"] = preserve_choice_group
        crop_audit["locally_detected_missing_figure"] = figure_id in local_figure_ids
        figure_x = (WIDTH - crop.width) // 2
        crop_audit["horizontal_placement"] = {
            "canvas_width_px": WIDTH,
            "figure_width_px": crop.width,
            "left_px": figure_x,
            "right_px_exclusive": figure_x + crop.width,
            "left_margin_px": figure_x,
            "right_margin_px": WIDTH - figure_x - crop.width,
            "center_error_px": abs(
                (figure_x + crop.width / 2) - WIDTH / 2
            ),
            "centered": abs(
                (figure_x + crop.width / 2) - WIDTH / 2
            ) <= 0.5,
        }
        crop_audits.append({
            "figure_id": figure_id,
            **crop_audit,
        })
        # The caption is model-authored metadata and may duplicate or
        # misidentify text already printed inside the source crop.  Preserve
        # original caption pixels when they are part of the crop, but never
        # synthesize a second student-visible caption from model output.
        visible_caption = (
            str(figure.get("caption") or "")
            if RENDER_MODEL_FIGURE_CAPTIONS
            else ""
        )
        question_items.append(("figure", crop, visible_caption))
        question_items.append(("gap", 12, None))
    if not question_items:
        raise ValueError("原题文字块为空")

    question_height = 0
    for kind, payload, caption in question_items:
        if kind == "line":
            question_height += payload.height
        elif kind == "gap":
            question_height += int(payload)
        elif kind == "figure":
            question_height += payload.height + 2
            if caption:
                question_height += SMALL_LINE
    height = TOP + question_height + 22 + sum(
        line.height for line in knowledge_lines
    ) + BOTTOM
    image = blank(height)
    draw = ImageDraw.Draw(image)
    y = TOP
    for kind, payload, caption in question_items:
        if kind == "line":
            draw_atoms(draw, payload, x=MARGIN_X, y=y, fonts=body_fonts)
            y += payload.height
        elif kind == "gap":
            y += int(payload)
        elif kind == "figure":
            figure_x = (WIDTH - payload.width) // 2
            image.paste(ImageOps.grayscale(payload), (figure_x, y))
            draw.rectangle(
                (figure_x, y, figure_x + payload.width - 1, y + payload.height - 1),
                outline=0,
                width=1,
            )
            y += payload.height + 2
            if caption:
                caption = normalize_caption_markup(caption)
                caption_lines = wrap_markup(
                    draw,
                    caption,
                    small_fonts,
                    max_width=CONTENT_WIDTH,
                    line_height=SMALL_LINE,
                )
                for line in caption_lines:
                    visual_width = sum(atom.width for atom in line.atoms)
                    caption_x = int(
                        round((WIDTH - visual_width) / 2 - line.left)
                    )
                    draw_atoms(draw, line, x=caption_x, y=y, fonts=small_fonts)
                    y += line.height
    y += 22
    draw_content_lines(
        draw,
        knowledge_lines,
        y=y,
        body_fonts=body_fonts,
        song_path=song_path,
        song_bold_path=song_bold_path,
        times_path=times_path,
        times_bold_path=times_bold_path,
    )
    result = finish_png(image, destination)
    result["source_figure_crops"] = crop_audits
    result["render_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    return result


def part_methods(part: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one primary method followed by up to two alternatives."""
    methods = [
        {
            "method_id": str(part.get("method_id") or "M1"),
            "method_name": str(part.get("method_name") or ""),
            "steps": part.get("steps") if isinstance(part.get("steps"), list) else [],
        }
    ]
    alternatives = part.get("alternative_methods")
    if isinstance(alternatives, list):
        for alternative in alternatives:
            if not isinstance(alternative, Mapping):
                continue
            methods.append(
                {
                    "method_id": str(alternative.get("method_id") or ""),
                    "method_name": str(alternative.get("method_name") or ""),
                    "steps": (
                        alternative.get("steps")
                        if isinstance(alternative.get("steps"), list)
                        else []
                    ),
                }
            )
    return methods


def tier_part_method_counts(
    tier_key: str,
    output: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tier = output["guidance"][tier_key]
    counts: list[dict[str, Any]] = []
    for part in tier["parts"]:
        methods = part_methods(part)
        counts.append(
            {
                "part": str(part.get("part") or ""),
                "method_count": len(methods),
                "method_ids": [method["method_id"] for method in methods],
                "method_names": [method["method_name"] for method in methods],
            }
        )
    return counts


def display_part_title(tier_key: str, value: Any) -> str:
    """Map the internal unnumbered-part key to the tier analysis heading."""

    label = str(value or "").strip()
    if label != "整题":
        return label
    return {
        "advanced": "学神分析",
        "standard": "学霸分析",
        "foundation": "学渣分析",
    }[tier_key]


def tier_blocks(tier_key: str, output: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    tier = output["guidance"][tier_key]
    blocks: list[tuple[str, str, int]] = []
    if output.get("recognition_status") == "uncertain":
        items = "；".join(str(item) for item in output.get("uncertain_items", []))
        blocks.append((f"识图提醒：{items}", "small", 12))
    for part in tier["parts"]:
        blocks.append((display_part_title(tier_key, part["part"]), "section", 2))
        methods = part_methods(part)
        show_method_titles = len(methods) > 1
        for method_index, method in enumerate(methods):
            if show_method_titles:
                number_name = (
                    METHOD_NUMBER_NAMES[method_index]
                    if method_index < len(METHOD_NUMBER_NAMES)
                    else str(method_index + 1)
                )
                blocks.append(
                    (
                        f"方法{number_name}｜{method['method_name']}",
                        "method",
                        4,
                    )
                )
            steps = method["steps"]
            if not steps:
                raise ValueError(
                    f"guidance.{tier_key}.{part.get('part')}.{method['method_id']}"
                    ".steps must contain at least one substantive step"
                )
            for step_index, step in enumerate(steps, 1):
                if (
                    not isinstance(step, str)
                    or not step.strip()
                    or not SUBSTANTIVE_GUIDANCE_CONTENT.search(step)
                ):
                    raise ValueError(
                        f"guidance.{tier_key}.{part.get('part')}."
                        f"{method['method_id']}.steps[{step_index - 1}] "
                        "contains only punctuation or whitespace"
                    )
                is_last_step = step_index == len(steps)
                has_next_method = method_index + 1 < len(methods)
                gap_after = 12 if is_last_step and has_next_method else 4
                blocks.append(
                    (f"{step_index}. {step}", "body_indent", gap_after)
                )
        pitfall = part.get("pitfall")
        if pitfall:
            blocks.append((f"易错点：{pitfall}", "small", 10))
    summary = tier.get("pattern_summary")
    if summary:
        blocks.append(("通用套路", "section", 2))
        blocks.append((str(summary), "body_indent", 0))
    return blocks


def render_guidance(
    *,
    tier_key: str,
    output: Mapping[str, Any],
    destination: Path,
    song_path: Path,
    song_bold_path: Path,
    times_path: Path,
    times_bold_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    body_fonts = make_fonts(
        BODY_SIZE,
        song_path=song_path,
        song_bold_path=song_bold_path,
        times_path=times_path,
        times_bold_path=times_bold_path,
    )
    blocks = tier_blocks(tier_key, output)
    lines = make_content_lines(blocks, fonts=body_fonts)
    height = TOP + sum(line.height for line in lines) + BOTTOM
    image = blank(height)
    draw = ImageDraw.Draw(image)
    draw_content_lines(
        draw,
        lines,
        y=TOP,
        body_fonts=body_fonts,
        song_path=song_path,
        song_bold_path=song_bold_path,
        times_path=times_path,
        times_bold_path=times_bold_path,
    )
    result = finish_png(image, destination)
    result["render_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    result["part_method_counts"] = tier_part_method_counts(tier_key, output)
    result["max_method_count"] = max(
        (item["method_count"] for item in result["part_method_counts"]),
        default=0,
    )
    return result


def render_four_images(
    *,
    question_id: str,
    model_id: str,
    source_image: Path | Image.Image,
    output: Mapping[str, Any],
    output_dir: Path,
    song_path: Path = DEFAULT_SONG,
    song_bold_path: Path = DEFAULT_SONG_BOLD,
    times_path: Path = DEFAULT_TIMES,
    times_bold_path: Path = DEFAULT_TIMES_BOLD,
    lock_source_figure_bboxes: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    song_path = validate_font(song_path, "宋体")
    song_bold_path = validate_font(song_bold_path, "宋体粗体")
    times_path = validate_font(times_path, "Times New Roman")
    times_bold_path = validate_font(times_bold_path, "Times New Roman Bold")
    staging_dir = create_staging_directory(output_dir)
    files: dict[str, Any] = {}
    try:
        files["question_knowledge"] = render_knowledge_card(
            source_image=source_image,
            output=output,
            destination=staging_dir / "01_question_knowledge.png",
            song_path=song_path,
            song_bold_path=song_bold_path,
            times_path=times_path,
            times_bold_path=times_bold_path,
            lock_source_figure_bboxes=lock_source_figure_bboxes,
        )
        for index, tier_key in enumerate(("advanced", "standard", "foundation"), 2):
            files[tier_key] = render_guidance(
                tier_key=tier_key,
                output=output,
                destination=staging_dir / f"{index:02d}_guidance_{tier_key}.png",
                song_path=song_path,
                song_bold_path=song_bold_path,
                times_path=times_path,
                times_bold_path=times_bold_path,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        for item in files.values():
            staged_path = Path(str(item["path"]))
            final_path = output_dir / staged_path.name
            os.replace(staged_path, final_path)
            item["path"] = str(final_path.resolve())
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    method_counts_by_tier = {
        tier_key: files[tier_key]["part_method_counts"]
        for tier_key in ("advanced", "standard", "foundation")
    }
    max_method_count_per_part = max(
        (
            item["method_count"]
            for tier_counts in method_counts_by_tier.values()
            for item in tier_counts
        ),
        default=0,
    )
    return {
        "render_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "image_count": 4,
        "method_counts_by_tier": method_counts_by_tier,
        "max_method_count_per_part": max_method_count_per_part,
        "contract": {
            "width_px": WIDTH,
            "display_width_cm": DISPLAY_WIDTH_CM,
            "png_dpi": round(PNG_DPI, 3),
            "mode": "8-bit_grayscale_document",
            "chinese_font": str(song_path),
            "chinese_font_style": "Song-style CJK serif",
            "chinese_heading_font": str(song_bold_path),
            "chinese_heading_weight": (
                "native_bold_face"
                if song_bold_path != song_path
                else "document_synthetic_bold_1px"
            ),
            "latin_font": str(times_path),
            "math_symbol_font": str(
                resolve_math_symbol_path(
                    DEFAULT_MATH_SYMBOL,
                    times_path=times_path,
                    song_path=song_path,
                )
            ),
            "antialiasing": "native_font_grayscale",
            "binarized": False,
            "source_figures_centered": True,
            "source_figure_resize_only": True,
            "source_figure_max_master_height_px": SOURCE_FIGURE_MAX_OUTPUT_HEIGHT,
            "source_figure_max_device_height_px": SOURCE_FIGURE_MAX_DEVICE_HEIGHT,
            "source_figure_device_vertical_whitespace_px": (
                SOURCE_FIGURE_DEVICE_VERTICAL_WHITESPACE
            ),
            "source_figure_captions_centered": False,
            "model_generated_figure_captions_visible": False,
            "source_caption_pixels_preserved_with_crop": True,
            "main_nodes_bold": True,
            "single_method_title_visible": False,
            "visible_method_divider": False,
            "visible_page_header": False,
            "visible_tier_title": False,
            "visible_question_id": False,
            "visible_model_id": False,
            "visible_question_knowledge_divider": False,
            "body_px": BODY_SIZE,
            "formula_px": FORMULA_SIZE,
            "small_px": SMALL_SIZE,
        },
        "files": files,
    }


__all__ = [
    "render_four_images",
    "reflow_choice_options",
    "WIDTH",
    "DISPLAY_WIDTH_CM",
]
