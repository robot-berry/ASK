"""Shared, deterministic cleanup for locally recognized question text.

The helpers in this module do not solve a question and do not call a model.
They only recover observable document structure (currently small OCR tables)
and remove handwriting-shaped working that appears between a printed prompt
and its printed multiple-choice options.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont


_TABLE_ROW_LABEL = re.compile(
    r"^(?:环数|次数|人数|频数|频率|组别|类别|项目|成绩|分数|数量|个数|"
    r"月份?|年份?|时间|温度|路程|速度|质量|体积|单价|总价|型号|序号)$"
)
_OPTION_START = re.compile(r"^[A-D][.．、:：]\s*")
_ANSWER_CUE = re.compile(
    r"(?:取值范围|范围|结果|答案|中位数|平均数|众数|值)(?:是|为)\s*[:：]?$"
)
@dataclass(frozen=True)
class OcrTextTable:
    """One complete rectangular table observed in consecutive OCR rows."""

    start: int
    end: int
    rows: tuple[tuple[str, ...], ...]

    @property
    def column_count(self) -> int:
        return len(self.rows[0]) if self.rows else 0


def _table_cells(line: str) -> tuple[str, ...]:
    cells = tuple(part.strip() for part in re.split(r"\s+", line.strip()) if part.strip())
    if not 3 <= len(cells) <= 10:
        return ()
    if _OPTION_START.match(cells[0]):
        return ()
    if not _TABLE_ROW_LABEL.fullmatch(cells[0]):
        return ()
    if any(len(re.sub(r"\s+", "", cell)) > 14 for cell in cells):
        return ()
    return cells


def find_ocr_text_table(lines: Sequence[str]) -> OcrTextTable | None:
    """Find a small table only when row/column structure is fully observed.

    Requiring a nearby ``表`` cue, known row labels and equal column counts
    keeps ordinary prose and A-D option rows out of this conversion.  A failed
    match deliberately returns ``None`` so callers can retain the source crop.
    """

    normalized = [str(line or "").strip() for line in lines]
    for start in range(len(normalized)):
        first = _table_cells(normalized[start])
        if not first:
            continue
        cue_window = "".join(normalized[max(0, start - 3):start])
        if "表" not in cue_window:
            continue
        rows = [first]
        end = start + 1
        while end < len(normalized):
            row = _table_cells(normalized[end])
            if not row or len(row) != len(first):
                break
            rows.append(row)
            end += 1
        if len(rows) >= 2:
            return OcrTextTable(start=start, end=end, rows=tuple(rows))
    return None


def has_non_table_figure_cue(text: str) -> bool:
    """Return whether the question explicitly refers to a non-table visual."""

    value = str(text or "").replace("如下表", "").replace("下表", "")
    return bool(
        re.search(
            r"如图|图中|示意图|图象|图像|坐标系|电路图|光路图|受力图|装置图|流程图",
            value,
        )
    )


def render_ocr_text_table(
    table: OcrTextTable,
    *,
    width: int,
    song_path: Path,
) -> Image.Image:
    """Render an OCR-recovered table as a clean, source-free text grid."""

    columns = max(1, table.column_count)
    cell_width = width / columns
    font_size = 30
    while font_size > 19:
        font = ImageFont.truetype(str(song_path), font_size)
        probe = ImageDraw.Draw(Image.new("L", (8, 8), 255))
        widest = max(
            probe.textbbox((0, 0), cell, font=font)[2]
            for row in table.rows
            for cell in row
        )
        if widest <= cell_width - 14:
            break
        font_size -= 1
    font = ImageFont.truetype(str(song_path), font_size)
    row_height = max(50, font_size + 20)
    height = row_height * len(table.rows) + 1
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    for row_index, row in enumerate(table.rows):
        top = row_index * row_height
        bottom = top + row_height
        for column_index, cell in enumerate(row):
            left = round(column_index * width / columns)
            right = round((column_index + 1) * width / columns)
            draw.rectangle((left, top, right, bottom), outline=0, width=2)
            bbox = draw.textbbox((0, 0), cell, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text(
                (
                    left + (right - left - text_width) / 2 - bbox[0],
                    top + (row_height - text_height) / 2 - bbox[1],
                ),
                cell,
                font=font,
                fill=0,
            )
    return image


def strip_likely_handwritten_work(text: str) -> tuple[str, list[str]]:
    """Remove answer working only where printed document structure proves it.

    This intentionally avoids broad language or answer-based filtering.  It
    acts on two high-confidence layouts: a trailing low-level equality after a
    direct calculation prompt, and algebra inserted after an answer cue but
    before a complete multiple-choice row.
    """

    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    actions: list[str] = []
    if not lines:
        return "", actions

    option_index = next(
        (
            index
            for index, line in enumerate(lines)
            if len(set(re.findall(r"(?:^|\s)([A-D])[.．、:：]", line))) >= 2
            or _OPTION_START.match(line)
        ),
        -1,
    )
    if option_index > 0:
        cleaned_prefix: list[str] = []
        for line in lines[:option_index]:
            match = re.match(r"^(.*?(?:取值范围|范围|结果|答案|中位数|平均数|众数|值)(?:是|为))\s+(.+)$", line)
            if match:
                suffix = match.group(2).strip()
                cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in suffix)
                operator_count = len(re.findall(r"[=<>≤≥+\-−×÷*/^√△Δ()]", suffix))
                if cjk_count <= 2 and operator_count >= 2 and len(suffix) >= 5:
                    line = match.group(1).strip()
                    actions.append("remove_working_between_prompt_and_choices")
            cleaned_prefix.append(line)
        lines = [*cleaned_prefix, *lines[option_index:]]

    if len(lines) >= 2 and re.search(r"计算|化简|求值", "".join(lines[:-1])):
        tail = lines[-1]
        unwrapped = tail.strip().strip("$").strip()
        if re.match(r"^=\s*[^=]{1,80}$", unwrapped):
            lines.pop()
            actions.append("remove_trailing_handwritten_equality")

    return "\n".join(lines).strip(), list(dict.fromkeys(actions))
