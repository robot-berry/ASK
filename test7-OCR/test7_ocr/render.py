from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .option_layout import has_standard_option_sequence, option_grid_columns
from .text import reflow_text_for_render
from .types import QuestionDocument


def _load_font(config: dict[str, Any]) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = int(config["typography"]["font_size_px"])
    for candidate in config["typography"].get("font_candidates", []):
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def _text_width(draw: ImageDraw.ImageDraw, value: str, font: ImageFont.ImageFont) -> int:
    if not value:
        return 0
    left, _, right, _ = draw.textbbox((0, 0), value, font=font)
    return right - left


def _wrap_line(
    draw: ImageDraw.ImageDraw,
    value: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    if not value:
        return [""]
    output: list[str] = []
    current = ""
    forbidden_line_start = set("，。；：！？、）】》〉}］〕”’％%,.;:!?)]}")
    forbidden_line_end = set("（【《〈{［〔“‘(")
    for character in value:
        candidate = current + character
        if current and _text_width(draw, candidate, font) > max_width:
            if character in forbidden_line_start and len(current) > 1:
                output.append(current[:-1].rstrip())
                current = current[-1] + character
            elif current[-1] in forbidden_line_end and len(current) > 1:
                output.append(current[:-1].rstrip())
                current = current[-1] + character
            else:
                output.append(current.rstrip())
                current = character.lstrip() if character.isspace() else character
        else:
            current = candidate
    if current or not output:
        output.append(current.rstrip())
    return output


def _layout_text(
    document: QuestionDocument,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[list[tuple[str, int]]]:
    blocks: list[list[tuple[str, int]]] = []
    structured_options = (
        document.options if has_standard_option_sequence(document.options) else []
    )
    stem_source = (
        document.stem.strip() if structured_options else document.display_text.strip()
    ) or document.display_text.strip()
    stem = reflow_text_for_render(stem_source)
    for paragraph in stem.splitlines() or [stem]:
        for line in _wrap_line(draw, paragraph, font, max_width):
            blocks.append([(line, 0)])
    columns = option_grid_columns(
        structured_options,
        max_width,
        int(getattr(font, "size", 29)),
    )
    if columns > 1:
        cell_width = max_width / columns
        for start in range(0, len(structured_options), columns):
            row: list[tuple[str, int]] = []
            for column, option in enumerate(structured_options[start : start + columns]):
                label = option.get("label", "")
                text = reflow_text_for_render(option.get("text", ""))
                row.append((f"{label}. {text}", 20 + round(column * cell_width)))
            blocks.append(row)
    else:
        for option in structured_options:
            label = option.get("label", "")
            text = option.get("text", "")
            reflowed_option = reflow_text_for_render(text)
            raw_lines = reflowed_option.splitlines() or [reflowed_option]
            for line_index, raw_line in enumerate(raw_lines):
                prefix = f"{label}. " if label and line_index == 0 else ""
                wrapped = _wrap_line(draw, prefix + raw_line, font, max_width - 20)
                blocks.extend([[(line, 20)] for line in wrapped])
    if not any(text.strip() for row in blocks for text, _ in row):
        blocks = [[("【未识别到可用文字】", 0)]]
    return blocks


def render_question(
    document: QuestionDocument,
    asset_images: list[Image.Image],
    config: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    output = config["output"]
    typography = config["typography"]
    asset_config = config["visual_assets"]
    width = int(output["width_px"])
    left = int(output["margin_left_px"])
    right = int(output["margin_right_px"])
    top = int(output["margin_top_px"])
    bottom = int(output["margin_bottom_px"])
    line_gap = int(typography["line_gap_px"])
    asset_gap = int(output["asset_gap_px"])
    text_width = width - left - right
    max_asset_width = min(int(asset_config["max_render_width_px"]), text_width)

    scratch = Image.new("RGB", (width, 100), "white")
    scratch_draw = ImageDraw.Draw(scratch)
    font = _load_font(config)
    lines = _layout_text(document, scratch_draw, font, text_width)
    font_box = scratch_draw.textbbox((0, 0), "国Ag", font=font)
    line_height = max(1, font_box[3] - font_box[1]) + line_gap

    rendered_assets: list[tuple[Image.Image, int, int]] = []
    for asset in asset_images:
        scale = min(1.0, max_asset_width / max(1, asset.width))
        render_width = max(1, round(asset.width * scale))
        render_height = max(1, round(asset.height * scale))
        rendered = asset if scale == 1.0 else asset.resize((render_width, render_height), Image.Resampling.LANCZOS)
        rendered_assets.append((rendered, render_width, render_height))

    text_height = len(lines) * line_height
    assets_height = sum(item[2] + asset_gap for item in rendered_assets)
    if rendered_assets:
        assets_height -= asset_gap
        assets_height += int(output["paragraph_gap_px"])
    height = top + text_height + assets_height + bottom
    max_height = int(output.get("max_height_px", 30000))
    if height > max_height:
        raise ValueError(f"Rendered height {height}px exceeds safety limit {max_height}px")

    canvas = Image.new("RGB", (width, max(1, height)), output.get("background", "white"))
    draw = ImageDraw.Draw(canvas)
    y = top
    for row in lines:
        for line, indent in row:
            draw.text(
                (left + indent, y),
                line,
                font=font,
                fill=output.get("foreground", "black"),
            )
        y += line_height

    if rendered_assets:
        y += int(output["paragraph_gap_px"])
    placements: list[dict[str, int]] = []
    for rendered, render_width, render_height in rendered_assets:
        x = left + max(0, (text_width - render_width) // 2)
        canvas.paste(rendered, (x, y))
        placements.append({"x": x, "y": y, "width": render_width, "height": render_height})
        y += render_height + asset_gap

    return canvas, {
        "width": canvas.width,
        "height": canvas.height,
        "line_count": len(lines),
        "asset_placements": placements,
        "bottom_content_y": y - (asset_gap if rendered_assets else 0),
    }
