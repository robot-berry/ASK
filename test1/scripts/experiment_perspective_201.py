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
LATEST = ROOT / "output" / "evaluations" / "agentpicture_cleanup_release2" / "artifacts" / "png" / "20260813T023634_db514b96.png"
OUT = ROOT / "output" / "_experiments" / "perspective_201"


def fit_models(query: np.ndarray, reference: np.ndarray) -> tuple[dict, dict]:
    detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.018, edgeThreshold=16)
    q_gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    q_points, q_desc = detector.detectAndCompute(q_gray, None)
    r_points, r_desc = detector.detectAndCompute(r_gray, None)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(q_desc, r_desc, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < pair[1].distance * 0.72]
    source = np.float32([q_points[item.queryIdx].pt for item in good])
    target = np.float32([r_points[item.trainIdx].pt for item in good])
    homography, h_mask = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
    affine, a_mask = cv2.estimateAffine2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=4000, confidence=0.995
    )
    similarity, s_mask = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=4000, confidence=0.995
    )

    def residual(matrix: np.ndarray, mask: np.ndarray, perspective: bool) -> dict:
        selected = mask.ravel().astype(bool)
        points = source[selected].reshape(-1, 1, 2)
        truth = target[selected].reshape(-1, 1, 2)
        predicted = (
            cv2.perspectiveTransform(points, matrix)
            if perspective
            else cv2.transform(points, matrix)
        )
        error = np.linalg.norm(predicted - truth, axis=2).ravel()
        return {
            "inliers": int(selected.sum()),
            "ratio": float(selected.mean()),
            "median_error": float(np.median(error)),
            "p90_error": float(np.percentile(error, 90)),
            "rmse": float(np.sqrt(np.mean(error * error))),
        }

    def cross_model_residual(matrix: np.ndarray, selected: np.ndarray, perspective: bool) -> dict:
        points = source[selected].reshape(-1, 1, 2)
        truth = target[selected].reshape(-1, 1, 2)
        predicted = cv2.perspectiveTransform(points, matrix) if perspective else cv2.transform(points, matrix)
        error = np.linalg.norm(predicted - truth, axis=2).ravel()
        return {
            "median": float(np.median(error)),
            "p90": float(np.percentile(error, 90)),
            "rmse": float(np.sqrt(np.mean(error * error))),
        }

    def affine_components(matrix: np.ndarray) -> dict:
        linear = matrix[:, :2]
        u, singular, vt = np.linalg.svd(linear)
        rotation = u @ vt
        angle = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        # QR separates rotation and upper-triangular shear/scale.
        q, r = np.linalg.qr(linear)
        shear = float(r[0, 1] / max(1e-9, r[1, 1]))
        return {
            "rotation_degrees": angle,
            "singular_scales": singular.tolist(),
            "anisotropy": float(singular.max() / singular.min()),
            "shear_factor": shear,
            "shear_degrees": math.degrees(math.atan(shear)),
            "determinant": float(np.linalg.det(linear)),
        }

    h_selected = h_mask.ravel().astype(bool)
    metrics = {
        "good_matches": len(good),
        "homography": {
            **residual(homography, h_mask, True),
            "matrix": homography.tolist(),
            "projective_terms": [float(homography[2, 0]), float(homography[2, 1])],
        },
        "full_affine": {
            **residual(affine, a_mask, False),
            **affine_components(affine),
            "matrix": affine.tolist(),
            "error_on_homography_inliers": cross_model_residual(affine, h_selected, False),
        },
        "similarity": {
            **residual(similarity, s_mask, False),
            **affine_components(similarity),
            "matrix": similarity.tolist(),
            "error_on_homography_inliers": cross_model_residual(similarity, h_selected, False),
        },
    }
    payload = {
        "homography": homography,
        "affine": affine,
        "similarity": similarity,
        "source": source,
        "target": target,
        "h_mask": h_mask,
        "q_points": q_points,
        "r_points": r_points,
        "good": good,
    }
    return metrics, payload


def crop_to_matched_content(image: np.ndarray, points: np.ndarray, pad: int = 30) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x, y, width, height = cv2.boundingRect(points.astype(np.float32))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + width + pad)
    y1 = min(image.shape[0], y + height + pad)
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def letterbox(image: np.ndarray, width: int, height: int, label: str) -> np.ndarray:
    scale = min(width / image.shape[1], (height - 36) / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = 36 + (height - 36 - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    processor = QuestionImageProcessor(OUT / "_unused")
    original = cv2.imread(str(PHOTO))
    reference = cv2.imread(str(REFERENCE))
    latest = cv2.imread(str(LATEST))
    prepared = processor._limit_long_edge(original, 1800)
    prepared = processor._document_warp(prepared)
    rotation, _, _ = processor._auto_rotation(prepared)
    prepared = processor._rotate_clockwise(prepared, rotation)

    # Fit before deskew: homography itself tells us whether a single rotation
    # is sufficient, without letting a noisy deskew estimate bias the answer.
    q_scale = min(1.0, 1400.0 / max(prepared.shape[:2]))
    r_scale = min(1.0, 1400.0 / max(reference.shape[:2]))
    query = cv2.resize(prepared, None, fx=q_scale, fy=q_scale, interpolation=cv2.INTER_AREA)
    ref = cv2.resize(reference, None, fx=r_scale, fy=r_scale, interpolation=cv2.INTER_AREA)
    metrics, payload = fit_models(query, ref)

    ref_height, ref_width = ref.shape[:2]
    homography_corrected = cv2.warpPerspective(
        query, payload["homography"], (ref_width, ref_height), borderValue=(255, 255, 255)
    )
    affine_corrected = cv2.warpAffine(
        query, payload["affine"], (ref_width, ref_height), borderValue=(255, 255, 255)
    )
    similarity_corrected = cv2.warpAffine(
        query, payload["similarity"], (ref_width, ref_height), borderValue=(255, 255, 255)
    )
    cv2.imwrite(str(OUT / "01_prepared_before_deskew.jpg"), query, [cv2.IMWRITE_JPEG_QUALITY, 94])
    cv2.imwrite(str(OUT / "02_latest_binary.png"), latest)
    cv2.imwrite(str(OUT / "03_similarity_rotation_scale.png"), similarity_corrected)
    cv2.imwrite(str(OUT / "04_full_affine.png"), affine_corrected)
    cv2.imwrite(str(OUT / "05_homography.png"), homography_corrected)
    cv2.imwrite(str(OUT / "06_clean_reference.png"), ref)

    match_visual = cv2.drawMatches(
        query,
        payload["q_points"],
        ref,
        payload["r_points"],
        payload["good"],
        None,
        matchColor=(0, 175, 0),
        matchesMask=payload["h_mask"].ravel().astype(int).tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(OUT / "07_homography_inliers.jpg"), match_visual, [cv2.IMWRITE_JPEG_QUALITY, 92])
    preview = np.hstack(
        [
            letterbox(query, 500, 430, "A. prepared photo (no deskew)"),
            letterbox(latest, 500, 430, "B. latest output (rotation only)"),
            letterbox(similarity_corrected, 500, 430, "C. rotation + uniform scale"),
            letterbox(affine_corrected, 500, 430, "D. full affine correction"),
            letterbox(homography_corrected, 500, 430, "E. perspective homography"),
            letterbox(ref, 500, 430, "F. clean screenshot"),
        ]
    )
    cv2.imwrite(str(OUT / "comparison.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 94])
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
