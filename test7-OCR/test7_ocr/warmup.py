from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


STRUCTURED_WARMUP_PROFILE_VERSION = "synthetic_runtime_shapes_v7_scoped_vram_preclean"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def synthetic_structured_warmup_pages() -> list[Image.Image]:
    """Create non-dataset pages that exercise common OCR runtime shape classes."""
    font = _font(28)
    small = _font(23)
    pages: list[Image.Image] = []

    dense = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(dense)
    draw.text((52, 35), "示例题：以下内容只用于模型预热，不属于测试数据。", fill="black", font=font)
    for row in range(8):
        draw.text(
            (70, 105 + row * 62),
            f"{row + 1}. 已知条件与结论，选择正确的一项并说明理由。",
            fill="black",
            font=small,
        )
    draw.text((80, 625), "A. 甲    B. 乙    C. 丙    D. 丁", fill="black", font=font)
    pages.append(dense)

    formula = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(formula)
    formula_lines = (
        "x² + 2x - 3 = 0",
        "f(x) = (x + 1) / (x - 1)",
        "a² + b² = c²",
        "F = ma,  E = mc²",
        "sin θ = 1/2,  0 ≤ θ ≤ π",
        "A ∩ B = {x | -1 < x < 2}",
    )
    draw.text((50, 32), "数学与物理公式版式预热", fill="black", font=font)
    for row, value in enumerate(formula_lines):
        draw.text((90, 115 + row * 82), value, fill="black", font=font)
    pages.append(formula)

    visual = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(visual)
    draw.text((50, 30), "图形与表格版式预热", fill="black", font=font)
    draw.rectangle((70, 120, 500, 410), outline="black", width=4)
    for x in (210, 350):
        draw.line((x, 120, x, 410), fill="black", width=3)
    for y in (215, 310):
        draw.line((70, y, 500, y), fill="black", width=3)
    draw.line((650, 450, 650, 120), fill="black", width=4)
    draw.line((650, 450, 1120, 450), fill="black", width=4)
    draw.line((680, 420, 1080, 165), fill="black", width=5)
    draw.ellipse((820, 190, 1010, 380), outline="black", width=4)
    draw.text((70, 500), "如图，读取表格并判断坐标图中的关系。", fill="black", font=font)
    pages.append(visual)

    # A 1280px camera/screenshot page can contain text crops whose normalized
    # recognizer width is much larger than the page width.  Warm long, shallow
    # rows explicitly so a real question does not pay the first-shape allocation
    # cost.  This page is fully synthetic and contains no benchmark wording.
    long_rows = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(long_rows)
    draw.text((34, 28), "长题干、长公式与连续选项的运行时形状预热", fill="black", font=font)
    rows = (
        "1. 在理想实验条件下依次记录位置、速度、时间、质量、加速度与方向，并从四个互不相同的结论中选择全部正确项。",
        "2. 已知函数关系 y=(ax²+bx+c)/(x-1)，且 F=ma、p=mv、Eₖ=mv²/2，求参数范围并判断图像变化趋势。",
        "A. 条件甲与条件乙同时成立    B. 只有条件丙成立    C. 条件甲和条件丁成立    D. 以上结论均需重新验证",
        "3. 设集合 M={x|-3<x≤5}，N={x|x²-4x-5≥0}，讨论 M∩N、M∪N 以及补集在数轴上的对应区间。",
        "4. 读取横轴、纵轴、单位、方向和关键点后完整作答，不省略题号、括号、上下标、空格或末尾的待填写横线。",
    )
    for row, value in enumerate(rows):
        draw.text((28, 105 + row * 103), value, fill="black", font=small)
    draw.line((28, 625, 1248, 625), fill=(90, 90, 90), width=2)
    pages.append(long_rows)

    # Exercise two-column detector crops, compact labels and diagram-adjacent
    # text.  It resembles a generic photographed worksheet layout but is drawn
    # from primitives, never copied from the evaluation datasets.
    mixed = Image.new("RGB", (1280, 720), (246, 244, 238))
    draw = ImageDraw.Draw(mixed)
    draw.rectangle((35, 28, 1245, 690), fill="white", outline=(150, 150, 150), width=2)
    draw.text((62, 48), "双栏图文混排预热页（程序生成）", fill="black", font=font)
    draw.line((635, 98, 635, 650), fill=(180, 180, 180), width=2)
    for row in range(5):
        draw.text(
            (62, 125 + row * 73),
            f"{row + 1}. 左栏条件、计算式与结论 x{row + 1}=v₀t+at²/2。",
            fill="black",
            font=small,
        )
    draw.text((65, 520), "A. 增大   B. 减小   C. 不变   D. 无法判断", fill="black", font=small)
    draw.line((735, 465, 735, 160), fill="black", width=4)
    draw.line((735, 465, 1165, 465), fill="black", width=4)
    draw.line((770, 420, 1090, 205), fill="black", width=5)
    draw.polygon(((870, 410), (1030, 410), (950, 270)), outline="black")
    draw.text((675, 505), "右栏示意图保留原图像素边界并验证图注。", fill="black", font=small)
    draw.text((675, 565), "F₁+F₂=ma，方向沿坐标轴正向。", fill="black", font=small)
    pages.append(mixed)
    return pages
