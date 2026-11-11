from pathlib import Path

from PIL import Image, ImageDraw


BASE = Path(__file__).resolve().parent
TILE_WIDTH = 380
TILE_HEIGHT = 470


sheet = Image.new("RGB", (TILE_WIDTH * 5, TILE_HEIGHT * 3), (235, 235, 235))
draw = ImageDraw.Draw(sheet)
for index in range(1, 16):
    case = f"10-{index}"
    path = BASE / "math" / case / "01_question_knowledge.png"
    with Image.open(path) as opened:
        picture = opened.convert("RGB")
        picture.thumbnail(
            (TILE_WIDTH - 24, TILE_HEIGHT - 50),
            Image.Resampling.LANCZOS,
        )
    column = (index - 1) % 5
    row = (index - 1) // 5
    x = column * TILE_WIDTH + (TILE_WIDTH - picture.width) // 2
    y = row * TILE_HEIGHT + 34 + (TILE_HEIGHT - 44 - picture.height) // 2
    sheet.paste(picture, (x, y))
    draw.text((column * TILE_WIDTH + 10, row * TILE_HEIGHT + 8), case, fill=(20, 20, 20))
    draw.rectangle(
        (
            column * TILE_WIDTH,
            row * TILE_HEIGHT,
            (column + 1) * TILE_WIDTH - 1,
            (row + 1) * TILE_HEIGHT - 1,
        ),
        outline=(150, 150, 150),
        width=1,
    )

output = BASE / "first_images_overview.jpg"
sheet.save(output, quality=90)
print(output)
