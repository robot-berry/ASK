from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.processor import QuestionImageProcessor


PHOTO_ROOT = Path(r"D:\robot\Documents\ASK\P_DATA\testmini_photo")
REFERENCE_ROOT = Path(r"D:\robot\Documents\ASK\P_DATA\testmini_screenshot")
OUTPUT_ROOT = PROJECT_ROOT / "output" / "_experiments" / "cleanup_hard20"

HARD20 = [
    "photo_001_0004.jpg",
    "photo_008_0016.jpg",
    "photo_011_0026.jpg",
    "photo_035_0062.jpg",
    "photo_063_0123.jpg",
    "photo_070_0149.jpg",
    "photo_091_0210.jpg",
    "photo_107_0242.jpg",
    "photo_110_0253.jpg",
    "photo_112_0255.jpg",
    "photo_120_0277.jpg",
    "photo_155_0427.jpg",
    "photo_169_0524.jpg",
    "photo_171_0526.jpg",
    "photo_201_0773.jpg",
    "photo_205_0813.jpg",
    "photo_221_1008.jpg",
    "photo_223_1039.jpg",
    "photo_256_1547.jpg",
    "photo_291_2648.jpg",
]


def _paper_luminance(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]


def _divide_background(gray: np.ndarray, sigma: float = 32.0) -> np.ndarray:
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    background = np.maximum(background, 16)
    return cv2.divide(gray, background, scale=255)


def current_adaptive(image: np.ndarray) -> np.ndarray:
    luminance = _paper_luminance(image)
    normalized = _divide_background(luminance, 25.0)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(normalized)
    return cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 12
    )


def divide_otsu(image: np.ndarray) -> np.ndarray:
    luminance = _paper_luminance(image)
    normalized = _divide_background(luminance, 34.0)
    normalized = cv2.medianBlur(normalized, 3)
    _, result = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def sauvola(image: np.ndarray, window: int = 51, k: float = 0.16) -> np.ndarray:
    luminance = _paper_luminance(image)
    normalized = _divide_background(luminance, 34.0).astype(np.float32)
    mean = cv2.boxFilter(normalized, cv2.CV_32F, (window, window), normalize=True)
    square_mean = cv2.boxFilter(
        normalized * normalized, cv2.CV_32F, (window, window), normalize=True
    )
    deviation = np.sqrt(np.maximum(0.0, square_mean - mean * mean))
    threshold = mean * (1.0 + k * (deviation / 128.0 - 1.0))
    return np.where(normalized <= threshold, 0, 255).astype(np.uint8)


def faithful_gray(image: np.ndarray) -> np.ndarray:
    """Flatten lighting and softly suppress texture without hard thresholding."""
    luminance = _paper_luminance(image)
    normalized = _divide_background(luminance, 38.0)
    normalized = cv2.fastNlMeansDenoising(normalized, None, 4, 7, 21)
    darkness = 255.0 - normalized.astype(np.float32)
    # A small soft floor removes sensor/paper grain.  Strong and thin strokes
    # remain continuous; unlike a binary threshold, gray diagram fills survive.
    darkness = np.maximum(0.0, darkness - 7.0) * 1.18
    return np.clip(255.0 - darkness, 0, 255).astype(np.uint8)


def structure_binary(image: np.ndarray) -> np.ndarray:
    """Binary scan that gates local thresholding by actual local contrast."""
    luminance = _paper_luminance(image)
    normalized = _divide_background(luminance, 38.0)
    smooth = cv2.GaussianBlur(normalized, (3, 3), 0)
    local_mean = cv2.boxFilter(smooth.astype(np.float32), cv2.CV_32F, (41, 41))
    local_darkness = local_mean - smooth.astype(np.float32)
    _, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Add only well-supported faint strokes. This avoids turning low-amplitude
    # paper texture and moire into the dense pepper noise of adaptiveThreshold.
    faint_ink = (smooth < 239) & (local_darkness > 7.0)
    core_ink = otsu == 0
    ink = core_ink | faint_ink
    result = np.where(ink, 0, 255).astype(np.uint8)
    # Remove isolated one-pixel sensor noise, but never components large enough
    # to be punctuation, subscripts, decimal points, or thin diagram segments.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - result, 8)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area <= 2 and width <= 2 and height <= 2:
            result[labels == label] = 255
    return result


def balanced_binary(image: np.ndarray) -> np.ndarray:
    """Preferred black/white mode: conservative global core + local faint ink.

    A high local-contrast gate recovers faint formulas and diagram strokes while
    preventing broad shadows, paper grain, and low-amplitude moire from becoming
    foreground.  The fallback luminance cutoff keeps real dark strokes in areas
    where background division is imperfect.
    """
    luminance = _paper_luminance(image)
    normalized = _divide_background(luminance, 38.0)
    smooth = cv2.GaussianBlur(normalized, (3, 3), 0)
    local_mean = cv2.boxFilter(smooth.astype(np.float32), cv2.CV_32F, (51, 51))
    local_darkness = local_mean - smooth.astype(np.float32)
    _, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu is excellent on clean paper but can become too permissive when a
    # large stain occupies much of the frame.  Require either high local
    # contrast or a genuinely dark source pixel for the global core.
    core = (otsu == 0) & ((local_darkness > 5.0) | (luminance < 145))
    faint = (smooth < 242) & (local_darkness > 11.0)
    ink = core | faint
    result = np.where(ink, 0, 255).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - result, 8)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area <= 2 and width <= 2 and height <= 2:
            result[labels == label] = 255
    return result


def stain_suppressed_gray(image: np.ndarray) -> np.ndarray:
    """Neutralize colored stains, then apply the conservative gray cleanup."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    chroma = np.sqrt(
        (channel_a.astype(np.float32) - 128.0) ** 2
        + (channel_b.astype(np.float32) - 128.0) ** 2
    )
    # Colored ink/stains are nuisance in grayscale electronic output.  Lift
    # their luminance proportionally, but cap the correction so black print
    # crossing a stain remains visible instead of being erased.
    lifted = luminance.astype(np.float32) + np.clip((chroma - 10.0) * 1.55, 0.0, 70.0)
    neutral = np.clip(lifted, 0, 255).astype(np.uint8)
    normalized = _divide_background(neutral, 38.0)
    normalized = cv2.fastNlMeansDenoising(normalized, None, 4, 7, 21)
    darkness = np.maximum(0.0, 255.0 - normalized.astype(np.float32) - 7.0) * 1.18
    return np.clip(255.0 - darkness, 0, 255).astype(np.uint8)


def stain_suppressed_binary(image: np.ndarray) -> np.ndarray:
    """Balanced binary scan with an early color-stain suppression stage."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    chroma = np.sqrt(
        (channel_a.astype(np.float32) - 128.0) ** 2
        + (channel_b.astype(np.float32) - 128.0) ** 2
    )
    lifted = luminance.astype(np.float32) + np.clip((chroma - 10.0) * 1.55, 0.0, 70.0)
    neutral_luminance = np.clip(lifted, 0, 255).astype(np.uint8)
    neutral = cv2.cvtColor(neutral_luminance, cv2.COLOR_GRAY2BGR)
    return balanced_binary(neutral)


METHODS = {
    "01_current_adaptive": current_adaptive,
    "02_divide_otsu": divide_otsu,
    "03_sauvola": sauvola,
    "04_faithful_gray": faithful_gray,
    "05_structure_binary": structure_binary,
    "06_balanced_binary": balanced_binary,
    "07_stain_gray": stain_suppressed_gray,
    "08_stain_binary": stain_suppressed_binary,
}


def prepare(processor: QuestionImageProcessor, image: np.ndarray) -> tuple[np.ndarray, dict]:
    image = processor._limit_long_edge(image, 1800)
    image = processor._document_warp(image)
    rotation, rotation_confidence, rotation_method = processor._auto_rotation(image)
    image = processor._rotate_clockwise(image, rotation)
    angle, skew_confidence, skew_status = processor._estimate_text_skew(image)
    applied = 0.0
    if skew_status == "ready":
        applied = angle
        image = processor._rotate_bound(image, applied)
    return image, {
        "rotation": rotation,
        "rotation_confidence": round(rotation_confidence, 4),
        "rotation_method": rotation_method,
        "skew_detected": round(angle, 4),
        "skew_applied": round(applied, 4),
        "skew_confidence": round(skew_confidence, 4),
        "skew_status": "applied" if applied else skew_status,
    }


def register_reference(source: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Project the clean paired screenshot into the rectified photo coordinates."""
    maximum = 1300.0
    source_scale = min(1.0, maximum / max(source.shape[:2]))
    reference_scale = min(1.0, maximum / max(reference.shape[:2]))
    source_small = cv2.resize(
        source, None, fx=source_scale, fy=source_scale, interpolation=cv2.INTER_AREA
    )
    reference_small = cv2.resize(
        reference, None, fx=reference_scale, fy=reference_scale, interpolation=cv2.INTER_AREA
    )
    source_gray = cv2.cvtColor(source_small, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_small, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02)
    source_keypoints, source_descriptors = detector.detectAndCompute(source_gray, None)
    reference_keypoints, reference_descriptors = detector.detectAndCompute(reference_gray, None)
    if source_descriptors is None or reference_descriptors is None:
        raise RuntimeError("not enough SIFT features")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(reference_descriptors, source_descriptors, k=2)
    good = [first for first, second in matches if first.distance < second.distance * 0.74]
    if len(good) < 8:
        raise RuntimeError(f"only {len(good)} reliable matches")
    reference_points = np.float32([reference_keypoints[item.queryIdx].pt for item in good])
    source_points = np.float32([source_keypoints[item.trainIdx].pt for item in good])
    homography_small, inlier_mask = cv2.findHomography(
        reference_points, source_points, cv2.RANSAC, 4.0
    )
    if homography_small is None or inlier_mask is None or int(inlier_mask.sum()) < 6:
        raise RuntimeError("homography rejected")
    source_matrix = np.diag([source_scale, source_scale, 1.0])
    reference_matrix = np.diag([reference_scale, reference_scale, 1.0])
    homography = np.linalg.inv(source_matrix) @ homography_small @ reference_matrix
    height, width = source.shape[:2]
    reference_height, reference_width = reference.shape[:2]
    corners = np.float32(
        [[0, 0], [reference_width - 1, 0], [reference_width - 1, reference_height - 1], [0, reference_height - 1]]
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    polygon_area = abs(float(cv2.contourArea(projected.astype(np.float32))))
    if (
        polygon_area < width * height * 0.035
        or polygon_area > width * height * 2.2
    ):
        raise RuntimeError(
            f"implausible homography area={polygon_area / (width * height):.3f}"
        )
    registered = cv2.warpPerspective(
        reference, homography, (width, height), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )
    reference_mask = np.full(reference.shape[:2], 255, dtype=np.uint8)
    overlap = cv2.warpPerspective(
        reference_mask, homography, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
    )
    return registered, overlap > 0, {
        "matches": len(good),
        "inliers": int(inlier_mask.sum()),
        "inlier_ratio": round(float(inlier_mask.mean()), 4),
        "projected_area_ratio": round(polygon_area / (width * height), 4),
    }


def score_variant(output: np.ndarray, registered: np.ndarray, overlap: np.ndarray) -> dict:
    if output.ndim == 3:
        output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    else:
        output_gray = output
    reference_gray = cv2.cvtColor(registered, cv2.COLOR_BGR2GRAY)
    reference_ink = (reference_gray < 225) & overlap
    # A common display-level threshold permits a fair comparison between the
    # binary methods and the faithful grayscale method.
    output_ink = (output_gray < 225) & overlap
    reference_near = cv2.dilate(reference_ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    output_near = cv2.dilate(output_ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    true_reference = int(reference_ink.sum())
    true_output = int(output_ink.sum())
    recall = float((reference_ink & output_near).sum()) / max(1, true_reference)
    precision = float((output_ink & reference_near).sum()) / max(1, true_output)
    f1 = 2.0 * recall * precision / max(1e-9, recall + precision)
    far_background = overlap & ~cv2.dilate(
        reference_ink.astype(np.uint8), np.ones((9, 9), np.uint8)
    ).astype(bool)
    contamination = float((output_ink & far_background).sum()) / max(1, int(far_background.sum()))
    # The pairing is especially valuable here: it lets the experiment report
    # which extra connected components are not supported by the electronic
    # original. Large unsupported components are stains/folds/borders; tiny
    # ones are paper grain or moire.
    unsupported = (output_ink & far_background).astype(np.uint8)
    component_count, _, component_stats, _ = cv2.connectedComponentsWithStats(unsupported, 8)
    large_contamination = sum(
        int(component_stats[label, cv2.CC_STAT_AREA])
        for label in range(1, component_count)
        if int(component_stats[label, cv2.CC_STAT_AREA]) >= 24
    ) / max(1, int(far_background.sum()))
    return {
        "recall": round(recall, 5),
        "precision": round(precision, 5),
        "f1": round(f1, 5),
        "background_contamination": round(contamination, 5),
        "large_contamination": round(large_contamination, 5),
    }


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def labeled_cell(image: np.ndarray, label: str, width: int, height: int) -> np.ndarray:
    cell = letterbox(image, width, height - 32)
    result = np.full((height, width, 3), 255, dtype=np.uint8)
    result[32:] = cell
    cv2.putText(result, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in METHODS:
        (OUTPUT_ROOT / name).mkdir(exist_ok=True)
    (OUTPUT_ROOT / "00_prepared").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "00_registered_reference").mkdir(exist_ok=True)

    processor = QuestionImageProcessor(OUTPUT_ROOT / "_unused_processor_output")
    rows: list[dict] = []
    contact_rows: list[np.ndarray] = []
    detailed_rows: list[np.ndarray] = []
    invalid_registrations: list[dict] = []
    detailed_names = {
        "photo_001_0004.jpg",
        "photo_063_0123.jpg",
        "photo_091_0210.jpg",
        "photo_110_0253.jpg",
        "photo_155_0427.jpg",
        "photo_221_1008.jpg",
        "photo_291_2648.jpg",
    }
    for index, filename in enumerate(HARD20, 1):
        source = cv2.imread(str(PHOTO_ROOT / filename))
        reference_name = filename.replace("photo_", "screenshot_").rsplit(".", 1)[0] + ".png"
        reference = cv2.imread(str(REFERENCE_ROOT / reference_name))
        if source is None or reference is None:
            raise FileNotFoundError(filename)
        prepared, transform = prepare(processor, source)
        cv2.imwrite(str(OUTPUT_ROOT / "00_prepared" / f"{Path(filename).stem}.jpg"), prepared)
        try:
            registered, overlap, registration = register_reference(prepared, reference)
        except RuntimeError as error:
            print(f"[{index:02d}/{len(HARD20)}] {filename}: registration skipped ({error})", flush=True)
            invalid_registrations.append({"sample": filename, "reason": str(error)})
            continue
        cv2.imwrite(
            str(OUTPUT_ROOT / "00_registered_reference" / f"{Path(filename).stem}.png"),
            registered,
        )
        row_images = [
            labeled_cell(prepared, f"{index:02d} source", 330, 210),
            labeled_cell(registered, "paired reference", 330, 210),
        ]
        detail_images = [
            labeled_cell(prepared, f"{index:02d} prepared source", 520, 350),
            labeled_cell(registered, "paired clean reference", 520, 350),
        ]
        for method_name, method in METHODS.items():
            output = method(prepared)
            output_path = OUTPUT_ROOT / method_name / f"{Path(filename).stem}.png"
            cv2.imwrite(str(output_path), output, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            score = score_variant(output, registered, overlap)
            rows.append(
                {
                    "sample": filename,
                    "reference": reference_name,
                    "method": method_name,
                    **transform,
                    **registration,
                    **score,
                }
            )
            label = (
                f"{method_name[3:]} F1 {score['f1']:.3f} "
                f"noise {score['background_contamination']:.3f}"
            )
            row_images.append(labeled_cell(output, label, 330, 210))
            detail_images.append(labeled_cell(output, label, 520, 350))
        contact_rows.append(np.hstack(row_images))
        if filename in detailed_names:
            detailed_rows.append(np.hstack(detail_images))
        print(f"[{index:02d}/{len(HARD20)}] {filename}", flush=True)

    fieldnames = list(rows[0])
    with (OUTPUT_ROOT / "scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, dict] = {}
    for method_name in METHODS:
        selected = [row for row in rows if row["method"] == method_name]
        summary[method_name] = {
            "samples": len(selected),
            "mean_recall": round(float(np.mean([row["recall"] for row in selected])), 5),
            "mean_precision": round(float(np.mean([row["precision"] for row in selected])), 5),
            "mean_f1": round(float(np.mean([row["f1"] for row in selected])), 5),
            "mean_background_contamination": round(
                float(np.mean([row["background_contamination"] for row in selected])), 5
            ),
            "mean_large_contamination": round(
                float(np.mean([row["large_contamination"] for row in selected])), 5
            ),
        }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(
            {
                "requested_samples": len(HARD20),
                "paired_registration_valid": len({row["sample"] for row in rows}),
                "invalid_registrations": invalid_registrations,
                "methods": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    cv2.imwrite(str(OUTPUT_ROOT / "contact_all20.jpg"), np.vstack(contact_rows), [cv2.IMWRITE_JPEG_QUALITY, 91])
    cv2.imwrite(
        str(OUTPUT_ROOT / "contact_difficult7.jpg"),
        np.vstack(detailed_rows),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )
    print(
        json.dumps(
            {
                "requested_samples": len(HARD20),
                "paired_registration_valid": len({row["sample"] for row in rows}),
                "methods": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
