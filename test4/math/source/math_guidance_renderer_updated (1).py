#!/usr/bin/env python3
"""Render one mathematics problem into four 800 px document-style grayscale PNGs.

This renderer intentionally does *not* perform photographed-document restoration
(perspective correction, deskew, illumination normalization, denoising or
sharpening).  Those operations belong in ``photo_preprocessor.py``.

The model produces structured JSON only.  This module owns all presentation:

1. re-typeset question text + verified source figure crops + knowledge/formulae;
2. advanced guidance;
3. standard guidance;
4. foundation guidance.

Chinese glyphs use SimSun and Latin/digits use Times New Roman.  Formulae are
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

from PIL import Image, ImageDraw, ImageFont, ImageOps

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
# Keep the device geometry identical to the established physics test3 output.
DISPLAY_WIDTH_CM = 9.0
PNG_DPI = WIDTH / (DISPLAY_WIDTH_CM / 2.54)
RENDER_MODEL_FIGURE_CAPTIONS = False
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
TIER_SIZE = 32
SECTION_SIZE = 36
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
    os.environ.get(
        "MATH_TEST3_SYMBOL_FONT",
        os.environ.get("TEST3_SYMBOL_FONT", r"C:\Windows\Fonts\seguisym.ttf"),
    )
)
MATH_SYMBOL_CHARACTERS = frozenset("≪∠∝⊥∥≌∽△○⊙∩∪∈≤≥≠≈≡")
MATH_SYMBOL_SPLIT = re.compile(
    "(" + "|".join(re.escape(symbol) for symbol in sorted(MATH_SYMBOL_CHARACTERS)) + ")"
)
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
PLAIN_ATOMIC_RUN = re.compile(
    r"[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*(?:%|°)?"
)
BASE_FORMULA_STRUCTURE = re.compile(
    r"\\[A-Za-z]+|[_^=<>≤≥≠≈∝∥⊥+\-−×·*/]"
)
CJK_CHARACTER = re.compile(r"[⺀-鿿豈-﫿]")
FORBIDDEN_LINE_START = set("，。；：！？、）】》”’")
FORBIDDEN_LINE_END = set("∠△")
METHOD_NUMBER_NAMES = ("一", "二", "三")
TIER_LABELS = {
    "advanced": "学霸版",
    "standard": "普通版",
    "foundation": "基础版",
}


@dataclass(frozen=True)
class Fonts:
    song: Any
    song_bold: Any
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
    # ``False`` joins this atom to the previous atom for line wrapping.  A
    # formula may need several atoms solely because relation glyphs use a
    # fallback font; that font split must never become a visible line break.
    break_before: bool = True


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
    song = ImageFont.truetype(str(song_path), size)
    song_bold = ImageFont.truetype(str(song_bold_path), size)
    times = ImageFont.truetype(str(times_path), size)
    times_bold = ImageFont.truetype(str(times_bold_path), size)
    formula_size = FORMULA_SIZE if size == BODY_SIZE else size
    return Fonts(
        song=song,
        song_bold=song_bold,
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
    # ``plain_atoms`` may supply one atomic Latin identifier/number rather
    # than a single character.  All characters in those runs use the same
    # Latin face, so route by the first code point while retaining the whole
    # run as the wrapping unit.
    probe = character[0] if character else " "
    if probe in MATH_SYMBOL_CHARACTERS:
        return fonts.math_symbol
    if is_song_character(probe):
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

    def append_character(character: str) -> None:
        if character == "\n":
            atoms.append(Atom("break", "", 0.0))
            return
        font = char_font(fonts, character, bold_latin=bold_latin)
        atoms.append(
            Atom(
                "plain_bold" if bold_latin else "plain",
                character,
                float(draw.textlength(character, font=font)),
            )
        )

    # Keep Latin identifiers and numeric literals intact.  The previous
    # character-by-character fallback could visibly split ``PDM`` into
    # ``P``/``DM`` or ``200`` into ``20``/``0`` at the right margin.
    position = 0
    for match in PLAIN_ATOMIC_RUN.finditer(value):
        for character in value[position:match.start()]:
            append_character(character)
        token = match.group(0)
        font = char_font(fonts, token[0], bold_latin=bold_latin)
        atoms.append(
            Atom(
                "plain_bold" if bold_latin else "plain",
                token,
                float(draw.textlength(token, font=font)),
            )
        )
        position = match.end()
    for character in value[position:]:
        append_character(character)
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
    # ``clean_text`` deliberately degrades ``\sim`` to ASCII ``~`` for
    # legacy renderers whose body font has no similarity glyph.  This renderer
    # has a dedicated symbol face, so keep the textbook U+223D glyph instead.
    formula_source = re.sub(r"\\sim(?![A-Za-z])", "∽", value)
    formula = clean_text(formula_source)
    # The shared plain-text sanitizer spells ``\ll`` as Chinese so it remains
    # readable in renderers without mathematical glyph coverage.  This
    # renderer has an explicit SimSun formula fallback, so preserve the
    # conventional relation symbol used by the source question.
    if re.search(r"(?:\\)+ll(?![A-Za-z])", value):
        formula = formula.replace("远小于", "≪")
    if not formula:
        return []
    # Keep rich fraction/radical tokens intact, but allow long equations to
    # wrap after safe top-level operators.  In particular, do not split on
    # Chinese text before ``clean_text`` has parsed ``\frac{中文}{中文}``.
    tokens = rich_formula_atoms(formula)
    complete_width = rich_text_width(
        draw,
        formula,
        fonts.formula,
        fonts.script,
        fallback_body_font=fonts.formula_song,
        fallback_script_font=fonts.script_song,
    )
    # A complete formula that fits on its own line is a better wrapping unit
    # than each individual operator.  It will move below a prose prefix as one
    # textbook expression instead of producing ``b+`` or ``y=`` at line end.
    if complete_width <= CONTENT_WIDTH - 12:
        chunks = [formula]
    else:
        chunks = []
        current = []
        delimiter_depth = 0
        break_before = {"=", "+", "-", "−", "∝", "≤", "≥", "≈"}
        break_after = {",", "，", ";", "；"}
        for token in tokens:
            if (
                delimiter_depth == 0
                and token in break_before
                and current
            ):
                chunks.append("".join(current))
                current = []
            current.append(token)
            if token in {"(", "[", "{"}:
                delimiter_depth += 1
            elif token in {")", "]", "}"}:
                delimiter_depth = max(0, delimiter_depth - 1)
            elif delimiter_depth == 0 and token in break_after:
                chunks.append("".join(current))
                current = []
        if current:
            chunks.append("".join(current))
    output: list[Atom] = []
    for chunk in chunks:
        chunk_atoms: list[Atom] = []
        # Keep ordinary Latin/Greek formulae in Times New Roman, but split
        # relation/geometric glyphs that need the symbol fallback font.
        # Treating each as a plain atom preserves the shared line baseline while
        # ``char_font`` routes it to Segoe UI Symbol.
        for segment in MATH_SYMBOL_SPLIT.split(chunk):
            if not segment:
                continue
            if segment in MATH_SYMBOL_CHARACTERS:
                chunk_atoms.append(
                    Atom(
                        "plain",
                        segment,
                        float(draw.textlength(segment, font=fonts.math_symbol)),
                        not chunk_atoms,
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
            chunk_atoms.append(
                Atom("formula", segment, float(width), not chunk_atoms)
            )
        output.extend(chunk_atoms)
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
            draw.textlength(run, font=font)
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
        cursor += float(draw.textlength(run, font=font))
    return cursor


def mixed_formula_atoms(
    draw: ImageDraw.ImageDraw,
    value: str,
    fonts: Fonts,
) -> list[Atom]:
    """Parse the complete formula before applying Times/SimSun fallbacks."""
    return mixed_formula_segment_atoms(draw, value, fonts)


def base_formula_markup(value: str) -> str:
    """Return safe markup for a schema ``base_formulas[].formula`` value.

    Historical successful records sometimes store a theorem sentence in this
    field, or store prose with its own ``$...$`` inline formulae.  Wrapping
    either case in one additional pair of delimiters makes the complete line an
    unbreakable formula and can also expose the inner delimiters literally.
    """

    stripped = value.strip()
    if not stripped:
        return ""
    if stripped.count("$") >= 2 and stripped.count("$") % 2 == 0:
        return stripped
    if CJK_CHARACTER.search(stripped) and not BASE_FORMULA_STRUCTURE.search(stripped):
        return stripped
    return f"${stripped}$"


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
            if not atom.break_before:
                # Move the complete current formula chunk to the next line.
                # The chunk can contain several atoms only because some
                # symbols use the fallback math font; splitting at that
                # implementation boundary produced dangling ``≠``/``∥`` or
                # detached operands on the device.
                group_start = len(current) - 1
                while group_start > 0 and not current[group_start].break_before:
                    group_start -= 1
                if group_start > 0:
                    lines.append(make_line(current[:group_start]))
                    current = current[group_start:]
                    width = sum(atom_width(draw, item) for item in current)
            elif (
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
            stroke_width=1 if chinese_bold else 0,
            stroke_fill=0 if chinese_bold else None,
        )
        cursor += float(draw.textlength(run_text, font=font))
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
# A model-provided geometry bbox is semantic evidence.  Expanding an ordinary
# diagram beyond it can pull the neighbouring question line or a student's
# working into the electronic question.  Closed printed tables have their own
# stronger border-evidence recovery below, and an actually empty model bbox
# retains the nearby-ink recovery path.  The only cross-edge expansion allowed
# is the whitespace-bounded recovery in
# :func:`whitespace_bounded_edge_offsets`.
SOURCE_FIGURE_MAX_EXPAND_FRACTION = 0.0


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


def suppress_isolated_right_edge_fragment(
    grayscale: Image.Image,
    pixel_box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Locate a clipped prose/handwriting fragment at the upper-right edge.

    This is intentionally narrower than handwriting removal.  It acts only
    when one 8-connected ink component is cut by the right border, lies in the
    outer fifth of the model box, has word-like (rather than point-label-like)
    size, and represents only a minority of the box ink.  A long axis, curve,
    table border, or other structural stroke therefore remains untouched.  A
    small source-space suppression box is returned in the audit; the semantic
    model bbox itself stays intact so a lower point label is never clipped just
    because it shares the same x-coordinate as upper handwriting.
    """
    left, top, right, bottom = pixel_box
    width = right - left
    height = bottom - top
    audit: dict[str, Any] = {
        "applied": False,
        "reason": "no_isolated_upper_right_edge_fragment",
        "original_bbox_px": list(pixel_box),
    }
    if width < 80 or height < 80:
        audit["reason"] = "model_bbox_too_small_for_edge_fragment_guard"
        return pixel_box, audit

    region = grayscale.crop(pixel_box)
    pixels = region.load()
    threshold = SOURCE_FIGURE_PROJECTION_INK_THRESHOLD
    total_ink = sum(region.histogram()[:threshold])
    if total_ink <= 0:
        audit["reason"] = "model_bbox_has_no_visible_ink"
        return pixel_box, audit

    edge_probe = max(2, round(width * 0.012))
    visited: set[tuple[int, int]] = set()
    candidates: list[dict[str, Any]] = []
    for seed_x in range(max(0, width - edge_probe), width):
        for seed_y in range(height):
            seed = (seed_x, seed_y)
            if seed in visited or pixels[seed_x, seed_y] >= threshold:
                continue
            stack = [seed]
            visited.add(seed)
            component: list[tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                component.append((x, y))
                for neighbour_y in range(max(0, y - 1), min(height, y + 2)):
                    for neighbour_x in range(max(0, x - 1), min(width, x + 2)):
                        neighbour = (neighbour_x, neighbour_y)
                        if (
                            neighbour in visited
                            or pixels[neighbour_x, neighbour_y] >= threshold
                        ):
                            continue
                        visited.add(neighbour)
                        stack.append(neighbour)
            component_left = min(point[0] for point in component)
            component_top = min(point[1] for point in component)
            component_right = max(point[0] for point in component) + 1
            component_bottom = max(point[1] for point in component) + 1
            component_width = component_right - component_left
            component_height = component_bottom - component_top
            ink_fraction = len(component) / total_ink
            evidence = {
                "bbox_local_px": [
                    component_left,
                    component_top,
                    component_right,
                    component_bottom,
                ],
                "ink_pixels": len(component),
                "ink_fraction": round(ink_fraction, 6),
            }
            if (
                component_left >= round(width * 0.78)
                and component_top <= round(height * 0.45)
                and component_width >= max(8, round(width * 0.06))
                and component_height >= max(12, round(height * 0.12))
                and component_height <= round(height * 0.38)
                and 0.05 <= ink_fraction <= 0.35
            ):
                candidates.append(evidence)

    audit["edge_probe_px"] = edge_probe
    audit["candidate_count"] = len(candidates)
    audit["candidates"] = candidates
    if len(candidates) != 1:
        if candidates:
            audit["reason"] = "edge_fragment_evidence_not_unique"
        return pixel_box, audit

    local_left, local_top, local_right, local_bottom = (
        int(value) for value in candidates[0]["bbox_local_px"]
    )
    padding = max(1, round(min(width, height) * 0.006))
    suppression_box = (
        left + max(0, local_left - padding),
        top + max(0, local_top - padding),
        left + min(width, local_right + padding),
        top + min(height, local_bottom + padding),
    )
    audit.update({
        "applied": True,
        "reason": "unique_clipped_word_like_component_suppressed_at_upper_right_edge",
        "suppression_bbox_px": list(suppression_box),
        "suppression_padding_px": padding,
    })
    return pixel_box, audit


def constrain_source_figure_refinement(
    raw_box: tuple[int, int, int, int],
    model_box: tuple[int, int, int, int],
    *,
    source_size: tuple[int, int],
    edge_expand_px: dict[str, int] | None = None,
) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
    """Bound local refinement so sparse diagram endpoints remain protected.

    ``edge_expand_px`` optionally overrides the (usually zero) ordinary
    expansion cap on specific edges, used only when the dominant figure
    component itself is cut by the model box.
    """
    left, top, right, bottom = model_box
    box_width = right - left
    box_height = bottom - top
    max_shrink_x = max(2, round(box_width * SOURCE_FIGURE_MAX_SHRINK_FRACTION))
    max_shrink_y = max(2, round(box_height * SOURCE_FIGURE_MAX_SHRINK_FRACTION))
    base_expand_x = max(0, round(box_width * SOURCE_FIGURE_MAX_EXPAND_FRACTION))
    base_expand_y = max(0, round(box_height * SOURCE_FIGURE_MAX_EXPAND_FRACTION))
    edge_expand = edge_expand_px or {}
    expand_left = int(edge_expand.get("left", base_expand_x))
    expand_right = int(edge_expand.get("right", base_expand_x))
    expand_top = int(edge_expand.get("top", base_expand_y))
    expand_bottom = int(edge_expand.get("bottom", base_expand_y))
    source_width, source_height = source_size
    raw_left, raw_top, raw_right, raw_bottom = raw_box

    bounded = (
        max(0, min(max(raw_left, left - expand_left), left + max_shrink_x)),
        max(0, min(max(raw_top, top - expand_top), top + max_shrink_y)),
        min(
            source_width,
            max(min(raw_right, right + expand_right), right - max_shrink_x),
        ),
        min(
            source_height,
            max(min(raw_bottom, bottom + expand_bottom), bottom - max_shrink_y),
        ),
    )
    if bounded[0] >= bounded[2] or bounded[1] >= bounded[3]:
        bounded = model_box

    bounded_left, bounded_top, bounded_right, bounded_bottom = bounded
    return bounded, {
        "max_shrink_px": {"x": max_shrink_x, "y": max_shrink_y},
        "max_expand_px": {
            "x": max(expand_left, expand_right),
            "y": max(expand_top, expand_bottom),
        },
        "edge_expand_px": edge_expand,
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


SOURCE_FIGURE_EXPAND_WHITESPACE_GAP = 12
SOURCE_FIGURE_EXPAND_EMPTY_TOLERANCE = 0.01
SOURCE_FIGURE_EXPAND_LABEL_MARGIN = 6


def whitespace_bounded_edge_offsets(
    grayscale: Image.Image,
    pixel_box: tuple[int, int, int, int],
    search_box: tuple[int, int, int, int],
) -> dict[str, int]:
    """Return px each model-box edge must grow to keep the whole figure.

    A vision-model bbox is frequently offset or slightly tight, clipping a
    vertex, an arc or a label.  Prioritising figure completeness, each edge is
    grown outward while the adjacent strip still contains figure ink, and
    stops at a full-width whitespace bar: a strip with at most
    ``SOURCE_FIGURE_EXPAND_EMPTY_TOLERANCE`` of the box extent is treated as
    empty, so a sparse text line or a thin crossing band does not drag question
    prose into the crop.  The tolerance is deliberately small so sparse vertex
    labels (``A``/``B``/``C`` point marks) still count as figure ink; a small
    ``SOURCE_FIGURE_EXPAND_LABEL_MARGIN`` is added to every expanded edge so a
    label sitting just outside the ink envelope is not clipped by one or two
    pixels.  ``SOURCE_FIGURE_EXPAND_WHITESPACE_GAP`` consecutive empty strips
    stop the scan, which also bridges the short gaps inside a circle/arc
    outline.  The result is capped by ``search_box``.
    """
    left, top, right, bottom = pixel_box
    s_left, s_top, s_right, s_bottom = search_box
    box_width = right - left
    box_height = bottom - top
    row_tolerance = max(1, int(SOURCE_FIGURE_EXPAND_EMPTY_TOLERANCE * box_width))
    col_tolerance = max(1, int(SOURCE_FIGURE_EXPAND_EMPTY_TOLERANCE * box_height))
    gap = max(1, SOURCE_FIGURE_EXPAND_WHITESPACE_GAP)
    label_margin = max(0, SOURCE_FIGURE_EXPAND_LABEL_MARGIN)
    region = grayscale.crop((s_left, s_top, s_right, s_bottom))
    pixels = region.load()
    threshold = SOURCE_FIGURE_PROJECTION_INK_THRESHOLD

    def row_ink(y: int) -> int:
        count = 0
        for x in range(left - s_left, right - s_left):
            if pixels[x, y - s_top] < threshold:
                count += 1
        return count

    def col_ink(x: int) -> int:
        count = 0
        for y in range(top - s_top, bottom - s_top):
            if pixels[x - s_left, y] < threshold:
                count += 1
        return count

    def scan(start: int, stop: int, step: int, ink_at, margin: int) -> int:
        offset = 0
        empty = 0
        position = start
        while (step > 0 and position < stop) or (step < 0 and position >= stop):
            if ink_at(position) > (
                row_tolerance if ink_at is row_ink else col_tolerance
            ):
                empty = 0
                offset = abs(position - start) + 1
            else:
                empty += 1
                if empty >= gap:
                    break
            position += step
        return offset + margin if offset else 0

    return {
        "left": scan(left - 1, s_left - 1, -1, col_ink, label_margin),
        "right": scan(right, s_right, 1, col_ink, 0),
        "top": scan(top - 1, s_top - 1, -1, row_ink, label_margin),
        "bottom": scan(bottom, s_bottom, 1, row_ink, label_margin),
    }


def detect_closed_table_envelope(
    grayscale: Image.Image,
    pixel_box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
    """Recover a complete axis-aligned table from a model box that cuts rows.

    Ordinary geometry refinement is not allowed to cross the semantic model
    bbox because nearby prose is easy to mistake for a diagram endpoint.  A
    printed table gives stronger evidence: at least three aligned
    horizontal rules plus continuous left and right borders.  Only that closed
    border proof is allowed to override the ordinary expansion limit.
    """
    left, top, right, bottom = pixel_box
    box_width = right - left
    box_height = bottom - top
    pixels = grayscale.load()
    ink_threshold = SOURCE_FIGURE_PROJECTION_INK_THRESHOLD
    x_padding = max(28, round(box_width * 1.0))
    y_padding = max(36, round(box_height * 2.0))
    search_left = max(0, left - x_padding)
    search_right = min(grayscale.width, right + x_padding)
    search_top = max(0, top - y_padding)
    search_bottom = min(grayscale.height, bottom + y_padding)
    search_box = (search_left, search_top, search_right, search_bottom)
    minimum_rule_width = max(36, round(box_width * 0.16))

    def longest_row_run(y: int) -> tuple[int, int] | None:
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        last_ink: int | None = None
        for x in range(search_left, search_right):
            if pixels[x, y] >= ink_threshold:
                continue
            if run_start is None or last_ink is None or x - last_ink - 1 > 2:
                if run_start is not None and last_ink is not None:
                    runs.append((run_start, last_ink + 1))
                run_start = x
            last_ink = x
        if run_start is not None and last_ink is not None:
            runs.append((run_start, last_ink + 1))
        if not runs:
            return None
        return max(runs, key=lambda run: run[1] - run[0])

    rule_rows: list[tuple[int, int, int]] = []
    for y in range(search_top, search_bottom):
        run = longest_row_run(y)
        if run is None or run[1] - run[0] < minimum_rule_width:
            continue
        rule_rows.append((y, run[0], run[1]))

    # Collapse a 2-4 pixel printed rule into one representative scanline.
    horizontal_rules: list[tuple[int, int, int]] = []
    groups: list[list[tuple[int, int, int]]] = []
    for row in rule_rows:
        if not groups or row[0] - groups[-1][-1][0] > 2:
            groups.append([row])
        else:
            groups[-1].append(row)
    for group in groups:
        horizontal_rules.append(
            max(group, key=lambda row: (row[2] - row[1], -row[0]))
        )

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for seed_y, seed_left, seed_right in horizontal_rules:
        seed_width = seed_right - seed_left
        endpoint_tolerance = max(8, round(seed_width * 0.08))
        aligned = [
            rule
            for rule in horizontal_rules
            if abs(rule[1] - seed_left) <= endpoint_tolerance
            and abs(rule[2] - seed_right) <= endpoint_tolerance
        ]
        if len(aligned) < 3:
            continue
        aligned.sort(key=lambda rule: rule[0])

        # Two separate tables can share the same width.  Split on a gap too
        # large to be one row before testing the vertical borders.
        max_row_gap = max(44, round(box_height * 1.45))
        aligned_groups: list[list[tuple[int, int, int]]] = []
        for rule in aligned:
            if not aligned_groups or rule[0] - aligned_groups[-1][-1][0] > max_row_gap:
                aligned_groups.append([rule])
            else:
                aligned_groups[-1].append(rule)
        for aligned_group in aligned_groups:
            if len(aligned_group) < 3:
                continue
            rule_top = aligned_group[0][0]
            rule_bottom = aligned_group[-1][0]
            table_left = round(
                sum(rule[1] for rule in aligned_group) / len(aligned_group)
            )
            table_right = round(
                sum(rule[2] for rule in aligned_group) / len(aligned_group)
            )
            table_width = table_right - table_left
            table_height = rule_bottom - rule_top + 1
            if table_width < 40 or table_height < 24:
                continue
            vertical_overlap = max(
                0,
                min(bottom, rule_bottom + 1) - max(top, rule_top),
            )
            if vertical_overlap < max(2, round(min(box_height, table_height) * 0.05)):
                continue
            horizontal_overlap = max(
                0,
                min(right, table_right) - max(left, table_left),
            )
            if horizontal_overlap < round(min(box_width, table_width) * 0.35):
                continue

            border_probe = max(3, round(table_width * 0.01))

            def vertical_border_coverage(x: int) -> float:
                covered = 0
                for y in range(rule_top, rule_bottom + 1):
                    if any(
                        pixels[probe_x, y] < ink_threshold
                        for probe_x in range(
                            max(search_left, x - border_probe),
                            min(search_right, x + border_probe + 1),
                        )
                    ):
                        covered += 1
                return covered / table_height

            left_coverage = vertical_border_coverage(table_left)
            # Horizontal run endpoints are exclusive, so the right border is
            # normally at ``table_right - 1``.
            right_coverage = vertical_border_coverage(table_right - 1)
            if min(left_coverage, right_coverage) < 0.72:
                continue
            key = tuple(rule[0] for rule in aligned_group)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "bbox_px": [table_left, rule_top, table_right, rule_bottom + 1],
                    "horizontal_rule_rows_px": list(key),
                    "horizontal_rule_count": len(aligned_group),
                    "left_border_coverage": round(left_coverage, 6),
                    "right_border_coverage": round(right_coverage, 6),
                    "vertical_overlap_px": vertical_overlap,
                    "horizontal_overlap_px": horizontal_overlap,
                    "score": (
                        vertical_overlap / max(1, min(box_height, table_height)),
                        horizontal_overlap / max(1, min(box_width, table_width)),
                        len(aligned_group),
                        -abs((rule_top + rule_bottom) - (top + bottom)),
                    ),
                }
            )

    audit: dict[str, Any] = {
        "search_bbox_px": list(search_box),
        "minimum_horizontal_rule_width_px": minimum_rule_width,
        "horizontal_rule_count": len(horizontal_rules),
        "closed_table_candidate_count": len(candidates),
    }
    if not candidates:
        return None, audit
    selected = max(candidates, key=lambda candidate: candidate["score"])
    table_left, table_top, table_right, table_bottom = selected["bbox_px"]
    padding = max(8, round(min(table_right - table_left, table_bottom - table_top) * 0.04))
    recovered = (
        max(0, table_left - padding),
        max(0, table_top - padding),
        min(grayscale.width, table_right + padding),
        min(grayscale.height, table_bottom + padding),
    )
    audit.update(
        {
            "selected_table_bbox_px": selected["bbox_px"],
            "recovered_bbox_px": list(recovered),
            "horizontal_rule_rows_px": selected["horizontal_rule_rows_px"],
            "left_border_coverage": selected["left_border_coverage"],
            "right_border_coverage": selected["right_border_coverage"],
            "padding_px": padding,
        }
    )
    return recovered, audit


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
    # A tight/offset model bbox clips part of the printed diagram.  Grow each
    # edge to the figure's full ink extent, stopping at a full-width whitespace
    # bar so question prose or a clipped handwriting mark is not pulled in.
    edge_expand = whitespace_bounded_edge_offsets(
        grayscale,
        pixel_box,
        (search_left, search_top, search_right, search_bottom),
    )
    base_audit["dominant_component_edge_expand_px"] = edge_expand
    table_box, table_evidence = detect_closed_table_envelope(
        grayscale,
        pixel_box,
    )
    base_audit["closed_table_detection"] = table_evidence
    if table_box is not None:
        table_metrics = source_figure_ink_metrics(grayscale, table_box)
        return table_box, {
            **base_audit,
            "mode": "closed_table_border_recovered",
            "reason": "three_or_more_aligned_rules_with_closed_vertical_borders",
            "quality_ok": table_metrics["quality_ok"],
            "raw_refined_bbox_px": list(table_box),
            "refined_bbox_px": list(table_box),
            "refined_ink_pixels": table_metrics["ink_pixels"],
            "refined_ink_ratio": table_metrics["ink_ratio"],
            "refined_ink_bbox_px": table_metrics["ink_bbox_px"],
            "boundary_limits_px": {
                "applied": False,
                "reason": "closed_table_border_evidence_override",
                "ordinary_max_expand_fraction": SOURCE_FIGURE_MAX_EXPAND_FRACTION,
            },
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
            edge_expand_px=edge_expand,
        )
        bounded_box, edge_fragment_audit = suppress_isolated_right_edge_fragment(
            grayscale,
            bounded_box,
        )
        if edge_fragment_audit.get("applied"):
            mode = "model_bbox_isolated_edge_fragment_suppressed"
            reason = str(edge_fragment_audit["reason"])
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
                edge_expand_px=edge_expand,
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
            "edge_fragment_guard": edge_fragment_audit,
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
    # Tables and multi-panel diagrams legitimately contain several horizontal
    # bands separated by whitespace.  Keep every substantial band that
    # intersects the model box; selecting only the single largest band can
    # silently remove the last table row or a second panel.
    model_intersecting_rows = [
        run for run in row_candidates
        if run[1] > top and run[0] < bottom
    ]
    if len(model_intersecting_rows) >= 2:
        refined_top = min(run[0] for run in model_intersecting_rows)
        refined_bottom = max(run[1] for run in model_intersecting_rows)
        # If the model box cuts through a table/graph band, keep the immediately
        # adjacent continuation just outside the box.  The gap is capped by the
        # original box height so nearby question prose is not pulled in.
        nearby_gap = max(8, round(box_height * 0.45))
        lower_rows = sorted(
            (run for run in row_candidates if run[0] >= refined_bottom),
            key=lambda run: run[0],
        )
        if lower_rows and lower_rows[0][0] - refined_bottom <= nearby_gap:
            refined_bottom = lower_rows[0][1]
        upper_rows = sorted(
            (run for run in row_candidates if run[1] <= refined_top),
            key=lambda run: run[1],
            reverse=True,
        )
        if upper_rows and refined_top - upper_rows[0][1] <= nearby_gap:
            refined_top = upper_rows[0][0]
    else:
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
    model_intersecting_columns = [
        run for run in column_candidates
        if run[1] > left and run[0] < right
    ]
    if len(model_intersecting_columns) >= 2:
        refined_left = min(run[0] for run in model_intersecting_columns)
        refined_right = max(run[1] for run in model_intersecting_columns)
    else:
        refined_left, refined_right = max(
            column_candidates,
            key=lambda run: (
                sum(column_counts[run[0] - search_left:run[1] - search_left]),
                max(0, min(run[1], right) - max(run[0], left)),
                run[1] - run[0],
            ),
        )
    # A strong horizontal rule touching a crop edge usually means the model
    # stopped in the middle of a table. Extend to the next/previous substantial
    # horizontal rule, but only within the local search window.
    border_rule_minimum = max(24, round(box_width * 0.28))
    table_like = len(model_intersecting_rows) >= 2
    if not table_like:
        search_width = search_right - search_left
        wide_rules = [
            count for count in row_counts
            if count >= max(40, round(search_width * 0.16))
        ]
        table_like = len(wide_rules) >= 2
    bottom_probe_start = max(search_top, bottom - 3)
    bottom_probe_end = min(search_bottom, bottom + 4)
    bottom_touches_rule = any(
        row_counts[y - search_top] >= border_rule_minimum
        for y in range(bottom_probe_start, bottom_probe_end)
    )
    # A model box can end inside the final row rather than exactly on a rule.
    # For an already identified multi-band structure, one nearby full-width
    # rule below the box is sufficient evidence that the table continues.
    if not bottom_touches_rule and table_like:
        bottom_touches_rule = any(
            row_counts[y - search_top] >= border_rule_minimum
            for y in range(bottom_probe_end, search_bottom)
        )
    if bottom_touches_rule:
        for y in range(bottom_probe_end, search_bottom):
            if row_counts[y - search_top] >= border_rule_minimum:
                refined_bottom = max(refined_bottom, y + 1)
    top_probe_start = max(search_top, top - 3)
    top_probe_end = min(search_bottom, top + 4)
    top_touches_rule = any(
        row_counts[y - search_top] >= border_rule_minimum
        for y in range(top_probe_start, top_probe_end)
    )
    if top_touches_rule:
        for y in range(search_top, top_probe_start):
            if row_counts[y - search_top] >= border_rule_minimum:
                refined_top = min(refined_top, y)
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
    edge_fragment_guard = audit.get("edge_fragment_guard")
    if (
        isinstance(edge_fragment_guard, Mapping)
        and edge_fragment_guard.get("applied")
        and isinstance(edge_fragment_guard.get("suppression_bbox_px"), list)
        and len(edge_fragment_guard["suppression_bbox_px"]) == 4
    ):
        suppress_left, suppress_top, suppress_right, suppress_bottom = (
            int(value) for value in edge_fragment_guard["suppression_bbox_px"]
        )
        local_suppression = (
            max(0, suppress_left - refined_box[0]),
            max(0, suppress_top - refined_box[1]),
            min(crop.width, suppress_right - refined_box[0]),
            min(crop.height, suppress_bottom - refined_box[1]),
        )
        if (
            local_suppression[0] < local_suppression[2]
            and local_suppression[1] < local_suppression[3]
        ):
            ImageDraw.Draw(crop).rectangle(
                local_suppression,
                fill=(255, 255, 255),
            )
            audit["edge_fragment_suppression_applied"] = True
            audit["edge_fragment_suppression_bbox_in_crop_px"] = list(
                local_suppression
            )
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
                SECTION_SIZE,
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
        SECTION_SIZE,
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


def draw_centered_title(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    y: int,
    song_path: Path,
    song_bold_path: Path,
    times_path: Path,
    times_bold_path: Path,
) -> None:
    fonts = make_fonts(
        TITLE_SIZE,
        song_path=song_path,
        song_bold_path=song_bold_path,
        times_path=times_path,
        times_bold_path=times_bold_path,
    )
    line = wrap_markup(
        draw,
        text,
        fonts,
        max_width=CONTENT_WIDTH,
        line_height=TITLE_LINE,
        bold_latin=False,
        font_role="body",
    )[0]
    visual_width = sum(atom.width for atom in line.atoms)
    draw_atoms(draw, line, x=round((WIDTH - visual_width) / 2), y=y, fonts=fonts)


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
    """Render four student-facing PNGs from one geometry-stable source image.

    ``source_image`` should be the *display* image produced by
    ``photo_preprocessor.preprocess_photo`` (or another image with identical
    geometry to the image shown to the vision model).  This is important because
    ``bbox_1000`` coordinates are only valid when model input and render input
    share the same crop/rotation/perspective transform.

    The renderer may crop source figures, resize them, convert them to grayscale
    and place them on the final page, but it deliberately does not repair photo
    lighting, perspective or sharpness.
    """
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
    knowledge_blocks.append(("基础公式", "section", 4))
    for index, item in enumerate(output["base_formulas"], 1):
        formula_markup = base_formula_markup(str(item["formula"]))
        knowledge_blocks.append((f"{index}. {formula_markup}", "body", 0))
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
    for block in output.get("question_blocks", []):
        if not isinstance(block, Mapping):
            raise ValueError("原题文字块必须是对象")
        if block.get("type") == "text":
            lines = wrap_markup(
                probe_draw,
                str(block.get("text") or ""),
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
        crop, crop_audit = crop_source_figure(
            source,
            figure.get("bbox_1000", []),
            lock_bbox=lock_source_figure_bboxes,
        )
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


def has_substantive_render_content(value: Any) -> bool:
    """Keep punctuation-only placeholders out of a published guidance PNG."""

    return isinstance(value, str) and any(character.isalnum() for character in value)


def validate_renderable_tier_steps(
    tier_key: str,
    output: Mapping[str, Any],
) -> None:
    """Fail before drawing when a tier contains an empty or symbol-only step."""

    tier = output["guidance"][tier_key]
    for part_index, part in enumerate(tier["parts"]):
        for method_index, method in enumerate(part_methods(part)):
            steps = method["steps"]
            if not steps:
                raise ValueError(
                    f"guidance.{tier_key}.parts[{part_index}].method[{method_index}]"
                    ".steps没有可渲染的数学引导"
                )
            for step_index, step in enumerate(steps):
                if not has_substantive_render_content(step):
                    raise ValueError(
                        f"guidance.{tier_key}.parts[{part_index}].method[{method_index}]"
                        f".steps[{step_index}]只有标点或符号，拒绝渲染"
                    )


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


def strip_connector_punctuation(value: str) -> str:
    """Remove leading/trailing connector punctuation from a text fragment.

    The provider repair model frequently emits guidance steps that begin with
    a Chinese comma (``，观察图形…``), which would render as ``1. ，观察图形…``.
    Connector punctuation at either end of a numbered step is presentation
    noise, not mathematical content, so it is dropped at render time.  Brackets,
    question marks, and math symbols (``？()∠△⊙①②``) are preserved.
    """
    return str(value or "").strip(" \t\r\n，,、。．；;：:·")


def tier_blocks(tier_key: str, output: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    validate_renderable_tier_steps(tier_key, output)
    tier = output["guidance"][tier_key]
    blocks: list[tuple[str, str, int]] = []
    if output.get("recognition_status") == "uncertain":
        items = "；".join(str(item) for item in output.get("uncertain_items", []))
        blocks.append((f"识图提醒：{items}", "small", 12))
    for part in tier["parts"]:
        blocks.append((str(part["part"]), "section", 2))
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
            for step_index, step in enumerate(steps, 1):
                is_last_step = step_index == len(steps)
                has_next_method = method_index + 1 < len(methods)
                gap_after = 12 if is_last_step and has_next_method else 4
                blocks.append(
                    (
                        f"{step_index}. {strip_connector_punctuation(step)}",
                        "body_indent",
                        gap_after,
                    )
                )
        pitfall = part.get("pitfall")
        if pitfall:
            blocks.append(
                (f"易错点：{strip_connector_punctuation(pitfall)}", "small", 10)
            )
    summary = tier.get("pattern_summary")
    if summary:
        blocks.append(("通用套路", "section", 2))
        blocks.append((str(strip_connector_punctuation(summary)), "body_indent", 0))
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
    # Preflight all three guidance pages before creating even the first staged
    # file, so a degenerate later tier cannot leave a partially rendered set.
    for tier_key in ("advanced", "standard", "foundation"):
        validate_renderable_tier_steps(tier_key, output)
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
            "chinese_heading_font": str(song_path),
            "chinese_heading_weight": "document_synthetic_bold_1px",
            "latin_font": str(times_path),
            "math_symbol_font": str(DEFAULT_MATH_SYMBOL),
            "antialiasing": "native_font_grayscale",
            "binarized": False,
            "source_figures_centered": True,
            "source_figure_resize_only": True,
            "photo_restoration_in_renderer": False,
            "expected_source": "geometry_aligned_preprocessed_display_image",
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


__all__ = ["render_four_images", "WIDTH", "DISPLAY_WIDTH_CM"]
