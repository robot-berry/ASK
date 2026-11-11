from pathlib import Path

import cv2
import numpy as np
from pypdf import PdfReader

from app.processor import QuestionImageProcessor


def test_all_three_outputs(tmp_path: Path):
    canvas = np.full((700, 1000, 3), 90, np.uint8)
    page = np.array([[150, 80], [880, 130], [820, 620], [100, 570]], np.int32)
    cv2.fillConvexPoly(canvas, page, (245, 245, 240))
    cv2.putText(canvas, "1. 2x + 3 = 9", (240, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3)
    ok, encoded = cv2.imencode(".jpg", canvas)
    assert ok
    result = QuestionImageProcessor(tmp_path).process(encoded.tobytes(), "question.jpg")
    png = tmp_path / result.png
    pdf = tmp_path / result.pdf
    md = tmp_path / result.markdown
    assert png.exists() and cv2.imread(str(png)) is not None
    assert pdf.exists() and len(PdfReader(str(pdf)).pages) == 1
    assert md.exists() and "../png/" in md.read_text(encoding="utf-8")
    assert result.width > 500 and result.height > 300
    assert result.question_count >= 1
    assert (tmp_path / result.questions_pdf).exists()
    assert len(PdfReader(str(tmp_path / result.questions_pdf)).pages) == result.question_count
    assert (tmp_path / result.questions_markdown).exists()


def test_rejects_non_image(tmp_path: Path):
    try:
        QuestionImageProcessor(tmp_path).process(b"not-image", "bad.jpg")
    except ValueError as exc:
        assert "无法读取" in str(exc)
    else:
        raise AssertionError("invalid data should fail")


def test_removes_colored_ink_but_keeps_black_print(tmp_path: Path):
    image = np.full((300, 600, 3), 255, np.uint8)
    cv2.putText(image, "2x + 3 = 9", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    cv2.putText(image, "x = 3", (260, 220), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.4, (255, 0, 0), 4)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    result = QuestionImageProcessor(tmp_path).process(
        encoded.tobytes(),
        "ink.png",
        mode="color",
        handwriting_mode="color",
        rotation="0",
    )
    cleaned = cv2.imread(str(tmp_path / result.png))
    black_region = cleaned[70:140, 30:300].mean()
    blue_region = cleaned[160:240, 250:500].mean()
    assert black_region < 245
    assert blue_region > 245
    assert result.handwriting_mode == "color"


def test_manual_rotation(tmp_path: Path):
    image = np.full((160, 320, 3), 255, np.uint8)
    cv2.putText(image, "UP", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    result = QuestionImageProcessor(tmp_path).process(
        encoded.tobytes(), "sideways.png", mode="gray", rotation="90"
    )
    assert result.width == 160
    assert result.height == 320
    assert result.rotation == 90


def test_auto_rotation_can_choose_270(tmp_path: Path, monkeypatch):
    processor = QuestionImageProcessor(tmp_path)
    monkeypatch.setattr(processor, "_document_axis", lambda image: (90, 0.20))
    scores = iter(((0.50, 2), (0.20, 8), (0.50, 2), (0.85, 8)))
    monkeypatch.setattr(processor, "_textline_upright_score", lambda image: next(scores))
    angle, confidence, method = processor._auto_rotation(np.full((100, 200, 3), 255, np.uint8))
    assert angle == 270
    assert confidence > 0.5
    assert method == "document+textline-model"


def test_auto_rotation_can_choose_180(tmp_path: Path, monkeypatch):
    processor = QuestionImageProcessor(tmp_path)
    monkeypatch.setattr(processor, "_document_axis", lambda image: (0, 0.20))
    scores = iter(((0.15, 8), (0.50, 2), (0.90, 8), (0.50, 2)))
    monkeypatch.setattr(processor, "_textline_upright_score", lambda image: next(scores))
    angle, _, _ = processor._auto_rotation(np.full((100, 200, 3), 255, np.uint8))
    assert angle == 180


def test_auto_rotation_text_rows_override_wrong_sideways_document_axis(
    tmp_path: Path, monkeypatch
):
    processor = QuestionImageProcessor(tmp_path)
    monkeypatch.setattr(processor, "_document_axis", lambda image: (90, 0.61))
    scores = iter(((0.96, 15), (0.55, 5), (0.22, 15), (0.92, 5)))
    monkeypatch.setattr(processor, "_textline_upright_score", lambda image: next(scores))
    angle, confidence, method = processor._auto_rotation(
        np.full((100, 200, 3), 255, np.uint8)
    )
    assert angle == 0
    assert confidence >= 0.9
    assert method == "textline-axis+textline-model"


def test_auto_rotation_uses_horizontal_text_rows_when_axis_model_is_wrong(
    tmp_path: Path, monkeypatch
):
    processor = QuestionImageProcessor(tmp_path)
    image = np.full((300, 900, 3), 255, np.uint8)
    monkeypatch.setattr(processor, "_document_axis", lambda source: (90, 0.05))
    monkeypatch.setattr(processor, "_textline_upright_score", lambda source: (0.5, 0))
    monkeypatch.setattr(processor, "_document_upright_score", lambda source: 0.5)
    evidence = iter(
        (
            [(20, 40, 700, 30)],
            [],
            [(180, 220, 700, 30)],
            [],
        )
    )
    monkeypatch.setattr(processor, "_detect_printed_text_lines", lambda source: next(evidence))
    angle, confidence, method = processor._auto_rotation(image)
    assert angle == 0
    assert confidence >= 0.6
    assert method == "text-row-evidence"


def test_auto_rotation_sparse_lines_uses_whole_page_vote(tmp_path: Path, monkeypatch):
    processor = QuestionImageProcessor(tmp_path)
    image = np.full((300, 900, 3), 255, np.uint8)
    monkeypatch.setattr(processor, "_document_axis", lambda source: (0, 0.2))
    line_votes = iter(((0.43, 2), (0.91, 2), (0.43, 2), (0.91, 2)))
    monkeypatch.setattr(processor, "_textline_upright_score", lambda source: next(line_votes))
    evidence = iter(
        (
            [(20, 40, 700, 30)],
            [],
            [(180, 220, 700, 30)],
            [],
        )
    )
    monkeypatch.setattr(processor, "_detect_printed_text_lines", lambda source: next(evidence))
    page_votes = iter((0.48, 0.11))
    monkeypatch.setattr(processor, "_document_upright_score", lambda source: next(page_votes))
    angle, confidence, method = processor._auto_rotation(image)
    assert angle == 0
    assert confidence >= 0.6
    assert method == "text-row-evidence"


def test_auto_rotation_sparse_180_vote_keeps_camera_orientation_when_close(
    tmp_path: Path, monkeypatch
):
    processor = QuestionImageProcessor(tmp_path)
    image = np.full((300, 900, 3), 255, np.uint8)
    monkeypatch.setattr(processor, "_document_axis", lambda source: (0, 0.2))
    line_votes = iter(((0.88, 2), (0.97, 2), (0.88, 2), (0.97, 2)))
    monkeypatch.setattr(processor, "_textline_upright_score", lambda source: next(line_votes))
    evidence = iter(
        (
            [(20, 40, 700, 30)],
            [],
            [(180, 220, 650, 30)],
            [],
        )
    )
    monkeypatch.setattr(processor, "_detect_printed_text_lines", lambda source: next(evidence))
    page_votes = iter((0.21, 0.36))
    monkeypatch.setattr(processor, "_document_upright_score", lambda source: next(page_votes))
    angle, _, method = processor._auto_rotation(image)
    assert angle == 0
    assert method == "text-row-evidence"


def test_question_card_falls_back_to_whole_page(tmp_path: Path, monkeypatch):
    processor = QuestionImageProcessor(tmp_path)
    monkeypatch.setattr(processor, "_detect_printed_text_lines", lambda image: [])
    output = np.full((240, 360), 255, np.uint8)
    cards, pdf, markdown = processor._write_question_cards(output, "fallback", False)
    assert len(cards) == 1
    assert cards[0].fallback is True
    assert cards[0].bbox == {"x": 0, "y": 0, "width": 360, "height": 240}
    assert len(PdfReader(str(tmp_path / pdf)).pages) == 1
    assert "q001.png" in (tmp_path / markdown).read_text(encoding="utf-8")


def test_groups_two_question_regions():
    image = np.full((1000, 800), 255, np.uint8)
    rows = [
        (60, 100, 650, 35),
        (80, 150, 600, 32),
        (75, 205, 500, 30),
        (55, 610, 660, 34),
        (75, 660, 590, 30),
        (80, 710, 520, 30),
    ]
    regions = QuestionImageProcessor._group_question_regions(image, rows)
    assert len(regions) == 2
    assert regions[0][1] < regions[1][1]
    assert regions[0][1] + regions[0][3] <= regions[1][1]


def test_question_group_ignores_short_rows_above_prompt():
    image = np.full((1350, 1800), 255, np.uint8)
    rows = [
        (1120, 30, 250, 18),
        (730, 70, 620, 18),
        (280, 175, 1230, 66),
        (320, 240, 1170, 64),
        (345, 310, 770, 63),
        (330, 395, 700, 43),
    ]
    regions = QuestionImageProcessor._group_question_regions(image, rows)
    assert len(regions) == 1
    assert 110 <= regions[0][1] <= 165


def test_question_crop_keeps_wide_formula_and_figure_area():
    image = np.full((1350, 1800), 255, np.uint8)
    rows = [
        (282, 176, 1227, 66),
        (321, 236, 1169, 64),
        (346, 308, 771, 63),
        (330, 396, 701, 43),
    ]
    region = QuestionImageProcessor._group_question_regions(image, rows)[0]
    x, y, width, height, _ = region
    # Conservative padding must retain a right-side diagram and content below
    # the final detected text row instead of optimizing for a tight crop.
    assert x <= 36
    assert x + width >= 1764
    assert y + height >= 900


def test_single_question_crop_keeps_safe_vertical_extent():
    image = np.full((1000, 1200), 255, np.uint8)
    rows = [
        (90, 120, 900, 38),
        (105, 172, 820, 36),
    ]
    regions = QuestionImageProcessor._group_question_regions(image, rows)
    assert len(regions) == 1
    _, _, _, crop_height, _ = regions[0]
    assert crop_height >= 420


def test_rotate_bound_preserves_colored_corners(tmp_path: Path):
    processor = QuestionImageProcessor(tmp_path)
    image = np.full((120, 220, 3), 245, np.uint8)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    points = [(0, 0), (219, 0), (0, 119), (219, 119)]
    for point, color in zip(points, colors):
        cv2.circle(image, point, 7, color, -1)
    rotated = processor._rotate_bound(image, 7.0)
    assert rotated.shape[0] > image.shape[0]
    assert rotated.shape[1] > image.shape[1]
    for color in colors:
        distance = np.abs(rotated.astype(np.int16) - np.array(color, np.int16)).sum(axis=2)
        assert int((distance < 90).sum()) > 2


def test_document_warp_rejects_small_distractor_and_preserves_page_edges(tmp_path: Path):
    processor = QuestionImageProcessor(tmp_path)
    distractor = np.full((600, 900, 3), 238, np.uint8)
    cv2.rectangle(distractor, (250, 170), (650, 430), (25, 25, 25), 8)
    rejected = processor._document_warp(distractor)
    assert rejected.shape == distractor.shape

    page = np.full((620, 900, 3), 35, np.uint8)
    polygon = np.array([[70, 55], [840, 80], [815, 570], [45, 550]], np.int32)
    cv2.fillConvexPoly(page, polygon, (250, 250, 250))
    for point in ((82, 70), (825, 94), (798, 550), (61, 533)):
        cv2.circle(page, point, 8, (0, 0, 0), -1)
    warped = processor._document_warp(page)
    assert warped.shape[0] > 460 and warped.shape[1] > 730
    dark = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) < 60
    assert int(dark.sum()) >= 120


def test_deskew_safe_mode_applies_without_clipping(tmp_path: Path, monkeypatch):
    processor = QuestionImageProcessor(tmp_path)
    image = np.full((260, 600, 3), 255, np.uint8)
    cv2.putText(image, "1. x + y = 9", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    monkeypatch.setattr(processor, "_estimate_text_skew", lambda source: (-4.0, 0.9, "ready"))
    result = processor.process(
        encoded.tobytes(), "skew.png", rotation="0", deskew_mode="safe", extract_questions=False
    )
    assert result.deskew_status == "applied"
    assert result.deskew_applied_angle == -4.0
    assert result.width > image.shape[1]
    assert result.height > image.shape[0]


def test_deskew_off_keeps_original_dimensions(tmp_path: Path):
    image = np.full((160, 320, 3), 255, np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    result = QuestionImageProcessor(tmp_path).process(
        encoded.tobytes(), "flat.png", rotation="0", deskew_mode="off", extract_questions=False
    )
    assert result.deskew_status == "off"
    assert (result.width, result.height) == (320, 160)


def test_deskew_large_angle_requires_consensus_and_preserves_canvas(tmp_path: Path):
    processor = QuestionImageProcessor(tmp_path)
    image = np.full((520, 1000, 3), 255, np.uint8)
    for line, text in enumerate(("1. x + y = 9", "2. 3a - b = 7", "A 12  B 14  C 16")):
        cv2.putText(
            image,
            text,
            (80, 140 + line * 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            (0, 0, 0),
            3,
        )
    tilted = processor._rotate_bound(image, 7.0)
    angle, confidence, status = processor._estimate_text_skew(tilted)
    assert status == "ready"
    assert 5.5 <= abs(angle) <= 8.5
    corrected = processor._rotate_bound(tilted, angle)
    assert corrected.shape[0] >= tilted.shape[0]
    assert corrected.shape[1] >= tilted.shape[1]
    assert confidence >= 0.6


def test_reference_perspective_rectification_flattens_fanned_rows(tmp_path: Path):
    processor = QuestionImageProcessor(tmp_path)
    reference = np.full((520, 1000, 3), 255, np.uint8)
    for row, text in enumerate(("10. x + y = 9", "A 12   B 14", "C 16   D 18")):
        cv2.putText(
            reference,
            text,
            (70, 130 + row * 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.35,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
    cv2.rectangle(reference, (650, 85), (900, 390), (0, 0, 0), 4)
    cv2.circle(reference, (775, 235), 70, (0, 0, 0), 4)
    source_corners = np.float32([[0, 0], [999, 0], [999, 519], [0, 519]])
    photographed_corners = np.float32([[85, 18], [930, 92], [990, 505], [12, 470]])
    inverse = cv2.getPerspectiveTransform(source_corners, photographed_corners)
    photographed = cv2.warpPerspective(
        reference,
        inverse,
        (1050, 560),
        borderValue=(235, 235, 235),
    )
    rectified, confidence, status = processor._rectify_with_reference(
        photographed, reference
    )
    assert status == "applied"
    assert confidence >= 0.75
    assert rectified is not None and rectified.shape[:2] == reference.shape[:2]
    # Registered print must substantially agree with the clean geometry.
    difference = cv2.absdiff(
        cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
    )
    assert float((difference > 45).mean()) < 0.045


def test_reference_rectification_does_not_apply_to_rotation_only(tmp_path: Path):
    processor = QuestionImageProcessor(tmp_path)
    reference = np.full((420, 850, 3), 255, np.uint8)
    for row in range(4):
        cv2.putText(
            reference,
            f"{row + 1}. ABCD 1234",
            (60, 90 + row * 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 0),
            3,
        )
    cv2.rectangle(reference, (610, 50), (790, 180), (0, 0, 0), 4)
    cv2.circle(reference, (700, 295), 65, (0, 0, 0), 4)
    rotated = QuestionImageProcessor._rotate_bound(reference, 4.0)
    rectified, confidence, status = processor._rectify_with_reference(rotated, reference)
    assert rectified is None
    assert status == "rotation-only"
    assert confidence == 1.0


def test_camera_filename_can_use_ambiguous_reference_for_geometry_only(tmp_path: Path, monkeypatch):
    processor = QuestionImageProcessor(tmp_path / "output")
    reference = np.full((260, 620, 3), 255, np.uint8)
    cv2.putText(reference, "10. PERSPECTIVE", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 0), 3)
    cv2.putText(reference, "A 12  B 14  C 16", (35, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.circle(reference, (500, 140), 60, (0, 0, 0), 4)
    reference_path = tmp_path / "geometry.png"
    assert cv2.imwrite(str(reference_path), reference)
    source_corners = np.float32([[0, 0], [619, 0], [619, 259], [0, 259]])
    camera_corners = np.float32([[60, 4], [585, 48], [620, 250], [8, 235]])
    photographed = cv2.warpPerspective(
        reference,
        cv2.getPerspectiveTransform(source_corners, camera_corners),
        (650, 280),
        borderValue=(240, 240, 240),
    )
    monkeypatch.setattr(
        processor,
        "_reference_candidates",
        lambda image, limit=10: [(30, 0.55, 50, reference_path)],
    )
    candidate = processor._reference_candidates(photographed)[0][3]
    rectified, confidence, status = processor._rectify_with_reference(
        photographed, cv2.imread(str(candidate))
    )
    assert status == "applied" and confidence >= 0.75
    assert rectified is not None and rectified.shape[:2] == reference.shape[:2]

    unrelated = np.full((280, 650, 3), 255, np.uint8)
    cv2.putText(unrelated, "UNRELATED", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    rejected, _, status = processor._rectify_with_reference(unrelated, reference)
    assert rejected is None
    assert status in {"not-enough-matches", "low-confidence", "insufficient-coverage"}


def test_binary_cleanup_suppresses_slow_stain_but_keeps_print():
    height, width = 420, 800
    yy, xx = np.mgrid[:height, :width]
    paper = 240 - 58 * np.exp(-(((xx - 580) / 185) ** 2 + ((yy - 290) / 125) ** 2))
    image = np.dstack((paper, paper, paper)).astype(np.uint8)
    cv2.putText(image, "12. x + 3 = 9", (65, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (15, 15, 15), 3)
    cleaned = QuestionImageProcessor._clean(image, "binary")
    print_region = cleaned[105:170, 55:410]
    stain_only_region = cleaned[230:370, 465:720]
    assert float((print_region == 0).mean()) > 0.025
    assert float((stain_only_region == 0).mean()) < 0.015


def test_reference_restoration_requires_clear_geometric_match(tmp_path: Path, monkeypatch):
    processor = QuestionImageProcessor(tmp_path)
    reference = np.full((300, 700, 3), 255, np.uint8)
    cv2.putText(reference, "17. 3x + 2 = 11", (35, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 3)
    cv2.rectangle(reference, (420, 170), (620, 265), (0, 0, 0), 3)
    reference_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(reference_path), reference)
    photographed = cv2.warpAffine(
        reference,
        cv2.getRotationMatrix2D((350, 150), 3.0, 1.0),
        (700, 300),
        borderValue=(225, 225, 225),
    )
    cv2.circle(photographed, (170, 230), 38, (90, 90, 90), -1)
    detector = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.02, edgeThreshold=16)
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    points, descriptors = detector.detectAndCompute(gray, None)
    monkeypatch.setattr(processor, "_load_reference_index", lambda: [(reference_path, points, descriptors)])
    restored, name, confidence = processor._restore_from_reference(photographed)
    assert restored is not None
    assert name == "reference.png"
    assert confidence >= 0.8
    assert np.array_equal(restored, reference)


def test_reference_restoration_does_not_substitute_unrelated_page(tmp_path: Path, monkeypatch):
    processor = QuestionImageProcessor(tmp_path)
    reference = np.full((260, 620, 3), 255, np.uint8)
    cv2.putText(reference, "A + B = C", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    reference_path = tmp_path / "reference.png"
    assert cv2.imwrite(str(reference_path), reference)
    detector = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.02, edgeThreshold=16)
    points, descriptors = detector.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
    monkeypatch.setattr(processor, "_load_reference_index", lambda: [(reference_path, points, descriptors)])
    unrelated = np.full((260, 620, 3), 245, np.uint8)
    cv2.circle(unrelated, (310, 130), 85, (10, 10, 10), 5)
    restored, name, confidence = processor._restore_from_reference(unrelated)
    assert restored is None
    assert name == ""
    assert confidence < 0.8


def test_dataset_filename_can_restore_exact_paired_reference(tmp_path: Path, monkeypatch):
    processor = QuestionImageProcessor(tmp_path / "output")
    reference_dir = tmp_path / "references"
    reference_dir.mkdir()
    reference = np.full((180, 420, 3), 255, np.uint8)
    cv2.putText(reference, "13. SAFE PAIR", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.rectangle(reference, (45, 125), (150, 165), (0, 0, 0), 2)
    cv2.circle(reference, (270, 145), 30, (0, 0, 0), 2)
    cv2.line(reference, (330, 118), (390, 170), (0, 0, 0), 3)
    reference_path = reference_dir / "screenshot_070_0149.png"
    assert cv2.imwrite(str(reference_path), reference)
    processor._reference_dir = reference_dir
    monkeypatch.setattr(processor, "_load_reference_index", lambda: [])
    obscured = cv2.warpAffine(
        reference,
        cv2.getRotationMatrix2D((210, 90), 3.0, 1.15),
        (600, 260),
        borderValue=(215, 205, 190),
    )
    cv2.line(obscured, (20, 220), (560, 180), (180, 80, 20), 18)
    restored, name, confidence = processor._restore_from_reference(
        obscured, "photo_070_0149.jpg"
    )
    assert restored is not None and np.array_equal(restored, reference)
    assert name == reference_path.name
    assert confidence > 0.99

    rejected, name, _ = processor._restore_from_reference(obscured, "camera.jpg")
    assert rejected is None and name == ""

    unrelated = np.full_like(obscured, 245)
    cv2.putText(unrelated, "DIFFERENT", (120, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    rejected, name, _ = processor._restore_from_reference(
        unrelated, "photo_070_0149.jpg"
    )
    assert rejected is None and name == ""


def test_matched_electronic_output_preserves_gray_diagram_fill(tmp_path: Path, monkeypatch):
    processor = QuestionImageProcessor(tmp_path)
    source = np.full((240, 480, 3), 235, np.uint8)
    reference = np.full((180, 420, 3), 255, np.uint8)
    cv2.rectangle(reference, (100, 60), (320, 145), (178, 178, 178), -1)
    cv2.line(reference, (100, 60), (320, 145), (20, 20, 20), 2)
    monkeypatch.setattr(processor, "_document_warp", lambda image: image)
    monkeypatch.setattr(
        processor,
        "_restore_from_reference",
        lambda image, filename="": (reference, "q.png", 0.99),
    )
    ok, encoded = cv2.imencode(".png", source)
    assert ok
    result = processor.process(
        encoded.tobytes(), "photo.png", mode="binary", rotation="0", deskew_mode="off"
    )
    output = cv2.imread(str(tmp_path / result.png))
    assert result.cleanup_method == "matched-electronic"
    assert 160 < float(output[120, 200].mean()) < 200
    assert result.question_count == 1
    assert result.questions[0].bbox == {
        "x": 0,
        "y": 0,
        "width": reference.shape[1],
        "height": reference.shape[0],
    }
