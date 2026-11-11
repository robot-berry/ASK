from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def main() -> None:
    evaluation = Path(sys.argv[1]).resolve()
    photos = Path(sys.argv[2]).resolve()
    rows = list(csv.DictReader((evaluation / "results.csv").open(encoding="utf-8-sig")))
    tiles: list[np.ndarray] = []
    for row in rows:
        source = cv2.imread(str(photos / row["name"]))
        output = cv2.imread(str(evaluation / "artifacts" / row["output_png"]))
        if source is None or output is None:
            continue
        source_tile = fit(source, 440, 300)
        output_tile = fit(output, 440, 300)
        tile = np.hstack((source_tile, output_tile))
        label = f'{row["name"]} | {row["cleanup_method"]} | rot {row["rotation"]} deskew {row["deskew_applied_angle"]}'
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 30), (255, 255, 255), -1)
        cv2.putText(tile, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (15, 15, 15), 1, cv2.LINE_AA)
        tiles.append(tile)
    columns = 2
    rows_count = (len(tiles) + columns - 1) // columns
    blank = np.full_like(tiles[0], 245)
    grid = []
    for index in range(rows_count):
        chunk = tiles[index * columns : (index + 1) * columns]
        chunk += [blank] * (columns - len(chunk))
        grid.append(np.hstack(chunk))
    sheet = np.vstack(grid)
    cv2.imwrite(str(evaluation / "contact_source_vs_final.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
