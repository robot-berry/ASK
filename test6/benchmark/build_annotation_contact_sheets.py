#!/usr/bin/env python3
"""Build compact source/current contact sheets for annotation-driven QA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def fit(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return ImageOps.contain(image, size, Image.Resampling.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-page", type=int, default=5)
    args = parser.parse_args()

    annotation_payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    reviews = annotation_payload["reviews"]
    results = {
        str(row["question_id"]): row
        for row in read_jsonl(args.run / "results.jsonl")
    }
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=26)
    issues: dict[str, list[str]] = {}
    for question_id, review in reviews.items():
        for issue in str(review.get("manual_error_types") or "").split(";"):
            if issue:
                issues.setdefault(issue, []).append(question_id)

    cell_size = (880, 420)
    header_height = 42
    row_height = cell_size[1] + header_height + 18
    page_width = cell_size[0] * 2 + 48
    for issue, question_ids in sorted(issues.items()):
        for page_index in range(math.ceil(len(question_ids) / args.rows_per_page)):
            batch = question_ids[
                page_index * args.rows_per_page :
                (page_index + 1) * args.rows_per_page
            ]
            canvas = Image.new("RGB", (page_width, row_height * len(batch)), "#eef2f6")
            draw = ImageDraw.Draw(canvas)
            for row_index, question_id in enumerate(batch):
                result = results[question_id]
                y = row_index * row_height
                draw.text((16, y + 6), f"{question_id}  SOURCE", fill="#111827", font=font)
                draw.text((cell_size[0] + 32, y + 6), "CURRENT OCR", fill="#111827", font=font)
                source_path = args.dataset / str(result["image_path"])
                current_path = Path(str(result["rendered_ocr_path"]))
                for column, path in enumerate((source_path, current_path)):
                    image = fit(path, cell_size)
                    x = 16 + column * (cell_size[0] + 16)
                    image_x = x + (cell_size[0] - image.width) // 2
                    image_y = y + header_height + (cell_size[1] - image.height) // 2
                    canvas.paste(image, (image_x, image_y))
            canvas.save(args.output / f"{issue}_{page_index + 1:02d}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
