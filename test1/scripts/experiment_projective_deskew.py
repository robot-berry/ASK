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
REF = Path(r"D:\robot\Documents\ASK\P_DATA\testmini_screenshot\screenshot_201_0773.png")
OUT = ROOT / "output" / "_experiments" / "projective_deskew_201"


def estimate_h(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    def shrink(gray: np.ndarray, edge: int = 1800) -> tuple[np.ndarray, float]:
        scale = min(1.0, edge / max(gray.shape))
        return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale

    src, ss = shrink(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
    dst, ds = shrink(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY))
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01, edgeThreshold=18)
    skp, sd = sift.detectAndCompute(src, None)
    dkp, dd = sift.detectAndCompute(dst, None)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(sd, dd, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    a = np.float32([skp[m.queryIdx].pt for m in good])
    b = np.float32([dkp[m.trainIdx].pt for m in good])
    h_small, mask = cv2.findHomography(a, b, cv2.RANSAC, 5.0, maxIters=10000, confidence=0.999)
    to_small_source = np.diag([ss, ss, 1.0])
    from_small_target = np.diag([1.0 / ds, 1.0 / ds, 1.0])
    h = from_small_target @ h_small @ to_small_source
    selected = mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(a[selected].reshape(-1, 1, 2), h_small).reshape(-1, 2)
    errors = np.linalg.norm(projected - b[selected], axis=1)
    return h, {
        "good": len(good),
        "inliers": int(selected.sum()),
        "ratio": float(selected.mean()),
        "error_p90": float(np.percentile(errors, 90)),
    }


def warp_bound(image: np.ndarray, h: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    source_corners = np.float32(
        [[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]]
    ).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(source_corners, h).reshape(-1, 2)
    target_h, target_w = target_shape
    all_points = np.vstack((mapped, np.float32([[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]])))
    minimum = np.floor(all_points.min(axis=0) - 8)
    maximum = np.ceil(all_points.max(axis=0) + 8)
    translation = np.array([[1.0, 0.0, -minimum[0]], [0.0, 1.0, -minimum[1]], [0.0, 0.0, 1.0]])
    composed = translation @ h
    return cv2.warpPerspective(
        image,
        composed,
        (int(maximum[0] - minimum[0] + 1), int(maximum[1] - minimum[1] + 1)),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    photo = cv2.imread(str(PHOTO))
    ref = cv2.imread(str(REF))
    processor = QuestionImageProcessor(OUT / "artifacts")
    prepared = processor._limit_long_edge(photo, 1800)
    prepared = processor._document_warp(prepared)
    rotation, _, _ = processor._auto_rotation(prepared)
    prepared = processor._rotate_clockwise(prepared, rotation)
    h, metrics = estimate_h(prepared, ref)
    corrected = warp_bound(prepared, h, ref.shape[:2])
    cleaned = processor._clean(corrected, "binary")
    cv2.imwrite(str(OUT / "projective_color.png"), corrected)
    cv2.imwrite(str(OUT / "projective_binary.png"), cleaned)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
