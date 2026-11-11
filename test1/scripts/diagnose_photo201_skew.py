from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.processor import QuestionImageProcessor


PHOTO = Path(r"D:\robot\Documents\ASK\P_DATA\testmini_photo\photo_201_0773.jpg")
REFERENCE = Path(r"D:\robot\Documents\ASK\P_DATA\testmini_screenshot\screenshot_201_0773.png")
OUT = ROOT / "output" / "_experiments" / "photo201_skew_diagnosis"


def homography_metrics(source: np.ndarray, reference: np.ndarray) -> dict:
    detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.012, edgeThreshold=18)
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    scale_source = min(1.0, 1600.0 / max(source_gray.shape))
    scale_reference = min(1.0, 1600.0 / max(reference_gray.shape))
    source_small = cv2.resize(source_gray, None, fx=scale_source, fy=scale_source, interpolation=cv2.INTER_AREA)
    reference_small = cv2.resize(reference_gray, None, fx=scale_reference, fy=scale_reference, interpolation=cv2.INTER_AREA)
    source_points, source_desc = detector.detectAndCompute(source_small, None)
    reference_points, reference_desc = detector.detectAndCompute(reference_small, None)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(source_desc, reference_desc, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance]
    source_xy = np.float32([source_points[item.queryIdx].pt for item in good])
    reference_xy = np.float32([reference_points[item.trainIdx].pt for item in good])
    homography, mask = cv2.findHomography(source_xy, reference_xy, cv2.RANSAC, 6.0)
    selected = mask.ravel().astype(bool)
    inverse = np.linalg.inv(homography)

    # A horizontal vector in the clean reference mapped back into the source
    # is the true photographed baseline direction at that location.
    h, w = reference_small.shape
    samples = []
    for y_ratio in (0.25, 0.5, 0.75):
        for x_ratio in (0.25, 0.5, 0.75):
            x, y = w * x_ratio, h * y_ratio
            segment = np.float32([[[x - 20, y]], [[x + 20, y]]])
            mapped = cv2.perspectiveTransform(segment, inverse).reshape(2, 2)
            dx, dy = mapped[1] - mapped[0]
            samples.append(math.degrees(math.atan2(float(dy), float(dx))))
    return {
        "good": len(good),
        "inliers": int(selected.sum()),
        "inlier_ratio": float(selected.mean()),
        "baseline_angles_deg": [round(value, 4) for value in samples],
        "baseline_median_deg": round(float(np.median(samples)), 4),
        "baseline_range_deg": round(float(np.ptp(samples)), 4),
    }


def long_line_angles(image: np.ndarray) -> list[dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 1400.0 / max(gray.shape))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    normalized = cv2.divide(gray, np.maximum(cv2.GaussianBlur(gray, (0, 0), 35), 16), scale=255)
    edges = cv2.Canny(normalized, 45, 130)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720,
        threshold=45,
        minLineLength=int(gray.shape[1] * 0.12),
        maxLineGap=int(gray.shape[1] * 0.025),
    )
    result = []
    if lines is None:
        return result
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        angle = math.degrees(math.atan2(float(dy), float(dx)))
        angle = ((angle + 45.0) % 90.0) - 45.0
        length = math.hypot(dx, dy)
        if abs(angle) <= 20:
            result.append({"angle": round(angle, 4), "length": round(length, 2), "line": [int(x1), int(y1), int(x2), int(y2)]})
    result.sort(key=lambda item: item["length"], reverse=True)
    return result[:30]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    photo = cv2.imread(str(PHOTO))
    reference = cv2.imread(str(REFERENCE))
    processor = QuestionImageProcessor(OUT / "artifacts")
    prepared = processor._limit_long_edge(photo, 1800)
    warped = processor._document_warp(prepared)
    rotation, rotation_confidence, rotation_method = processor._auto_rotation(warped)
    upright = processor._rotate_clockwise(warped, rotation)
    detected, confidence, status = processor._estimate_text_skew(upright)
    corrected = processor._rotate_bound(upright, detected) if status == "ready" else upright.copy()
    cv2.imwrite(str(OUT / "01_upright_before_deskew.png"), upright)
    cv2.imwrite(str(OUT / "02_current_corrected.png"), corrected)
    report = {
        "rotation": rotation,
        "rotation_confidence": rotation_confidence,
        "rotation_method": rotation_method,
        "deskew_detected": detected,
        "deskew_confidence": confidence,
        "deskew_status": status,
        "before_to_reference": homography_metrics(upright, reference),
        "after_to_reference": homography_metrics(corrected, reference),
        "before_long_lines": long_line_angles(upright),
        "after_long_lines": long_line_angles(corrected),
    }
    (OUT / "diagnosis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
