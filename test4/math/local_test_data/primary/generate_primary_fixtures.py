#!/usr/bin/env python3
"""Generate three deterministic, answer-free primary-school math fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin


ROOT = Path(__file__).resolve().parent
CANVAS = (1400, 1800)
BACKGROUND = "white"
INK = "#171717"
MUTED = "#616161"
ACCENT = "#1f5f8b"
LIGHT = "#edf5fa"
FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    if not path.is_file():
        raise FileNotFoundError(f"required Chinese font is missing: {path}")
    return ImageFont.truetype(str(path), size=size)


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    text_font: ImageFont.FreeTypeFont,
    max_width: int,
    *,
    fill: str = INK,
    line_gap: int = 22,
) -> int:
    """Draw Chinese-friendly wrapped text and return the first free y position."""
    x, y = xy
    line_height = text_font.getbbox("国Ag")[3] - text_font.getbbox("国Ag")[1]
    for paragraph in text.split("\n"):
        if not paragraph:
            y += line_height + line_gap
            continue
        current = ""
        for character in paragraph:
            candidate = current + character
            if current and draw.textlength(candidate, font=text_font) > max_width:
                if character in "，。！？；：、）》】”’":
                    draw.text((x, y), candidate, font=text_font, fill=fill)
                    y += line_height + line_gap
                    current = ""
                    continue
                draw.text((x, y), current, font=text_font, fill=fill)
                y += line_height + line_gap
                current = character
            else:
                current = candidate
        if current:
            draw.text((x, y), current, font=text_font, fill=fill)
            y += line_height + line_gap
    return y


def new_page(number: int, title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    image = Image.new("RGB", CANVAS, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((80, 70, 1320, 235), radius=30, fill=LIGHT, outline=ACCENT, width=4)
    draw.text((125, 105), f"第 {number} 题", font=font(50, bold=True), fill=ACCENT)
    draw.text((355, 105), title, font=font(50, bold=True), fill=INK)
    draw.text((100, 285), subtitle, font=font(31), fill=MUTED)
    draw.line((100, 350, 1300, 350), fill="#b7b7b7", width=3)
    return image, draw, 405


def add_footer(draw: ImageDraw.ImageDraw) -> None:
    draw.line((100, 1690, 1300, 1690), fill="#d5d5d5", width=2)
    footer = "Test4 本地合成测试样本 · 题面不含答案"
    footer_font = font(27)
    width = draw.textlength(footer, font=footer_font)
    draw.text(((CANVAS[0] - width) / 2, 1720), footer, font=footer_font, fill="#777777")


def save(image: Image.Image, filename: str, fixture_id: str) -> dict[str, object]:
    path = ROOT / filename
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("fixture_id", fixture_id)
    metadata.add_text("fixture_kind", "synthetic_pristine_primary_math")
    metadata.add_text("contains_answer", "false")
    image.save(path, format="PNG", compress_level=9, dpi=(200, 200), pnginfo=metadata)
    raw = path.read_bytes()
    return {
        "file": filename,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def application_fixture() -> dict[str, object]:
    image, draw, y = new_page(
        1,
        "多步计算与应用",
        "要求：列式计算，并用文字说明每一步在求什么。",
    )
    body = (
        "学校为运动会准备了 12 箱矿泉水，每箱 24 瓶。上午用去 85 瓶，"
        "下午又运来 4 箱同样的矿泉水。\n\n"
        "（1）现在一共有多少瓶矿泉水？请列出一个综合算式。\n\n"
        "（2）按照你的计算顺序，说明算式中每一步分别解决了什么问题。"
    )
    draw_wrapped(draw, body, (125, y), font(48), 1150, line_gap=30)
    add_footer(draw)
    return save(image, "primary_01_multistep_application.png", "primary_synth_application_001")


def fraction_measurement_fixture() -> dict[str, object]:
    image, draw, y = new_page(
        2,
        "分数与长度单位",
        "要求：写出完整思路，计算结果用最简分数表示。",
    )
    body = (
        "一条彩带长 4 米。做花环用去了全长的 3/8，包装礼盒又用去了 5/6 米。\n\n"
        "（1）这条彩带还剩多少米？请写出完整算式并计算。\n\n"
        "（2）说明题目中的“全长的 3/8”和“5/6 米”表示的量有什么不同，"
        "以及你是怎样统一它们再计算的。"
    )
    draw_wrapped(draw, body, (125, y), font(48), 1150, line_gap=30)
    add_footer(draw)
    return save(image, "primary_02_fractions_measurement.png", "primary_synth_fraction_001")


def double_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    label: str,
    *,
    vertical: bool = False,
) -> None:
    draw.line((start, end), fill=INK, width=4)
    size = 15
    if vertical:
        for x, y, direction in ((start[0], start[1], 1), (end[0], end[1], -1)):
            draw.polygon([(x, y), (x - size, y + direction * 25), (x + size, y + direction * 25)], fill=INK)
        label_font = font(34)
        draw.text((start[0] - 96, (start[1] + end[1]) / 2 - 24), label, font=label_font, fill=INK)
    else:
        for x, y, direction in ((start[0], start[1], 1), (end[0], end[1], -1)):
            draw.polygon([(x, y), (x + direction * 25, y - size), (x + direction * 25, y + size)], fill=INK)
        label_font = font(34)
        label_width = draw.textlength(label, font=label_font)
        draw.text(((start[0] + end[0] - label_width) / 2, start[1] + 12), label, font=label_font, fill=INK)


def geometry_fixture() -> dict[str, object]:
    image, draw, y = new_page(
        3,
        "组合图形",
        "下面的图形按题意画出；图中长度单位均为米。",
    )
    prompt = (
        "一个长方形花圃长 12 米、宽 8 米。园丁从它的右上角挖去一个"
        "长 5 米、宽 3 米的小长方形，剩下的部分如下图阴影所示。"
    )
    y = draw_wrapped(draw, prompt, (125, y), font(42), 1150, line_gap=23) + 20

    left, top, right, bottom = 260, y + 70, 1140, y + 600
    cut_left = right - round((right - left) * 5 / 12)
    cut_bottom = top + round((bottom - top) * 3 / 8)
    polygon = [
        (left, top),
        (cut_left, top),
        (cut_left, cut_bottom),
        (right, cut_bottom),
        (right, bottom),
        (left, bottom),
    ]
    draw.polygon(polygon, fill="#dceef8", outline=INK)
    draw.line(polygon + [polygon[0]], fill=INK, width=6, joint="curve")
    draw.rectangle((cut_left, top, right, cut_bottom), outline="#8c8c8c", width=3)
    hatch_step = 35
    offset = cut_left - cut_bottom
    for diagonal in range(offset, right - top + hatch_step, hatch_step):
        x_start = max(cut_left, diagonal + top)
        y_start = x_start - diagonal
        x_end = min(right, diagonal + cut_bottom)
        y_end = x_end - diagonal
        if top <= y_start <= cut_bottom and top <= y_end <= cut_bottom:
            draw.line((x_start, y_start, x_end, y_end), fill="#c7c7c7", width=2)
    label_font = font(29)
    label = "挖去部分\n5 米 × 3 米"
    draw.multiline_text((cut_left + 38, top + 55), label, font=label_font, fill=MUTED, spacing=12, align="center")
    draw.text((left + 80, top + 230), "剩余花圃", font=font(40, bold=True), fill=ACCENT)

    draw.line((left, bottom, left, bottom + 65), fill=INK, width=3)
    draw.line((right, bottom, right, bottom + 65), fill=INK, width=3)
    double_arrow(draw, (left, bottom + 48), (right, bottom + 48), "12 米")
    draw.line((left - 65, top, left, top), fill=INK, width=3)
    draw.line((left - 65, bottom, left, bottom), fill=INK, width=3)
    double_arrow(draw, (left - 48, top), (left - 48, bottom), "8 米", vertical=True)

    question_y = bottom + 125
    questions = (
        "（1）求剩余花圃的面积。\n"
        "（2）求剩余花圃一周的长度。\n"
        "（3）说明你怎样确定凹进去的两条边各有多长。"
    )
    draw_wrapped(draw, questions, (125, question_y), font(40), 1150, line_gap=18)
    add_footer(draw)
    return save(image, "primary_03_composite_geometry.png", "primary_synth_geometry_001")


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    generated = [
        application_fixture(),
        fraction_measurement_fixture(),
        geometry_fixture(),
    ]
    print(json.dumps({"generated": generated}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
