from __future__ import annotations

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
OUT = PROJECT_ROOT / "output" / "_experiments" / "reference_guard"
FOCUS = ["photo_070_0149.jpg", "photo_201_0773.jpg"]
VALIDATION = [
    "photo_001_0004.jpg",
    "photo_008_0016.jpg",
    "photo_035_0062.jpg",
    "photo_063_0123.jpg",
    "photo_091_0210.jpg",
    "photo_110_0253.jpg",
    "photo_120_0277.jpg",
    "photo_169_0524.jpg",
    "photo_171_0526.jpg",
    "photo_205_0813.jpg",
    "photo_221_1008.jpg",
    "photo_223_1039.jpg",
    "photo_256_1547.jpg",
    "photo_291_2648.jpg",
]


def normalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    result = points.copy().astype(np.float32)
    result[:, 0] /= max(1, width)
    result[:, 1] /= max(1, height)
    return result


def hull_area_ratio(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(points.astype(np.float32))
    return abs(float(cv2.contourArea(hull))) / max(1, width * height)


def quadrant_count(points: np.ndarray, width: int, height: int) -> int:
    occupied: set[tuple[int, int]] = set()
    for x, y in points:
        occupied.add((min(1, int(x * 2 / max(1, width))), min(1, int(y * 2 / max(1, height)))))
    return len(occupied)


def grid_coverage(points: np.ndarray, width: int, height: int, grid: int = 4) -> float:
    occupied: set[tuple[int, int]] = set()
    for x, y in points:
        occupied.add(
            (
                min(grid - 1, int(x * grid / max(1, width))),
                min(grid - 1, int(y * grid / max(1, height))),
            )
        )
    return len(occupied) / float(grid * grid)


def projected_quad_metrics(homography: np.ndarray, query_width: int, query_height: int, ref_width: int, ref_height: int) -> dict:
    query_corners = np.float32(
        [[0, 0], [query_width - 1, 0], [query_width - 1, query_height - 1], [0, query_height - 1]]
    ).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(query_corners, homography).reshape(-1, 2)
    area = abs(float(cv2.contourArea(projected.astype(np.float32))))
    ref_area = max(1, ref_width * ref_height)
    inside = (
        (projected[:, 0] >= -0.15 * ref_width)
        & (projected[:, 0] <= 1.15 * ref_width)
        & (projected[:, 1] >= -0.15 * ref_height)
        & (projected[:, 1] <= 1.15 * ref_height)
    )
    edges = np.linalg.norm(np.roll(projected, -1, axis=0) - projected, axis=1)
    diagonal_1 = float(np.linalg.norm(projected[0] - projected[2]))
    diagonal_2 = float(np.linalg.norm(projected[1] - projected[3]))
    return {
        "projected_area_ratio": area / ref_area,
        "projected_corners_inside": int(inside.sum()),
        "edge_ratio": float(edges.max() / max(1e-6, edges.min())),
        "diagonal_ratio": max(diagonal_1, diagonal_2) / max(1e-6, min(diagonal_1, diagonal_2)),
        "projected_quad": projected.round(2).tolist(),
    }


def symmetric_reprojection_metrics(
    homography: np.ndarray, source: np.ndarray, target: np.ndarray, selected: np.ndarray
) -> dict:
    if not np.any(selected):
        return {
            "forward_error_median": float("inf"),
            "forward_error_p90": float("inf"),
            "backward_error_median": float("inf"),
            "backward_error_p90": float("inf"),
        }
    inlier_source = source[selected].reshape(-1, 1, 2)
    inlier_target = target[selected].reshape(-1, 1, 2)
    projected_target = cv2.perspectiveTransform(inlier_source, homography)
    forward_errors = np.linalg.norm(projected_target - inlier_target, axis=2).ravel()
    try:
        inverse = np.linalg.inv(homography)
        projected_source = cv2.perspectiveTransform(inlier_target, inverse)
        backward_errors = np.linalg.norm(projected_source - inlier_source, axis=2).ravel()
    except np.linalg.LinAlgError:
        backward_errors = np.full(len(inlier_source), np.inf, dtype=np.float32)
    return {
        "forward_error_median": float(np.median(forward_errors)),
        "forward_error_p90": float(np.percentile(forward_errors, 90)),
        "backward_error_median": float(np.median(backward_errors)),
        "backward_error_p90": float(np.percentile(backward_errors, 90)),
    }


def stage2(metrics: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    # A replacement needs support across the printed content, not a repeated
    # icon or a few letters in one local patch.  Both coordinate systems must
    # show spatial support because a degenerate homography can artificially
    # spread a compact source cluster.
    if metrics["inliers"] < 14:
        reasons.append("inliers<14")
    if metrics["inlier_ratio"] < 0.55:
        reasons.append("inlier_ratio<0.55")
    if metrics["query_hull_ratio"] < 0.012:
        reasons.append("query_hull_ratio<0.012")
    if metrics["reference_hull_ratio"] < 0.012:
        reasons.append("reference_hull_ratio<0.012")
    if metrics["query_quadrants"] < 2:
        reasons.append("query_quadrants<2")
    if metrics["reference_quadrants"] < 2:
        reasons.append("reference_quadrants<2")
    if metrics["query_grid_coverage"] < 0.125:
        reasons.append("query_grid_coverage<0.125")
    if metrics["reference_grid_coverage"] < 0.125:
        reasons.append("reference_grid_coverage<0.125")
    # The photo commonly contains a large margin around a small question, so
    # projecting all four *photo* corners outside the clean crop is normal.
    # Do not gate on whole-image projected area/corners. Symmetric error and
    # inlier support are measured on actual matched content instead.
    if metrics["forward_error_p90"] > 4.5:
        reasons.append("forward_error_p90>4.5")
    if metrics["backward_error_p90"] > 6.0:
        reasons.append("backward_error_p90>6")
    return not reasons, reasons


def draw_match(
    query: np.ndarray,
    query_points: list[cv2.KeyPoint],
    reference: np.ndarray,
    reference_points: list[cv2.KeyPoint],
    good: list[cv2.DMatch],
    mask: np.ndarray | None,
) -> np.ndarray:
    matches_mask = mask.ravel().astype(int).tolist() if mask is not None else None
    return cv2.drawMatches(
        query,
        query_points,
        reference,
        reference_points,
        good,
        None,
        matchColor=(0, 180, 0),
        singlePointColor=(0, 0, 255),
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    processor = QuestionImageProcessor(OUT / "_unused")
    index = processor._load_reference_index()
    detector = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.02, edgeThreshold=16)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    report: dict[str, dict] = {}
    for filename in FOCUS + VALIDATION:
        original = cv2.imread(str(PHOTO_ROOT / filename))
        if original is None:
            raise FileNotFoundError(filename)
        prepared = processor._limit_long_edge(original, 1800)
        prepared = processor._document_warp(prepared)
        rotation, _, _ = processor._auto_rotation(prepared)
        prepared = processor._rotate_clockwise(prepared, rotation)
        angle, _, status = processor._estimate_text_skew(prepared)
        if status == "ready":
            prepared = processor._rotate_bound(prepared, angle)
        query_gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        query_gray = processor._limit_long_edge(query_gray, 700)
        query_points, query_descriptors = detector.detectAndCompute(query_gray, None)
        candidates: list[dict] = []
        visual_payload: dict[str, tuple] = {}
        for path, reference_points, reference_descriptors in index:
            if reference_descriptors is None or query_descriptors is None:
                continue
            pairs = matcher.knnMatch(query_descriptors, reference_descriptors, k=2)
            good = [
                pair[0]
                for pair in pairs
                if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
            ]
            if len(good) < 8:
                continue
            source = np.float32([query_points[item.queryIdx].pt for item in good])
            target = np.float32([reference_points[item.trainIdx].pt for item in good])
            homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
            if homography is None or mask is None:
                continue
            selected = mask.ravel().astype(bool)
            inlier_source = source[selected]
            inlier_target = target[selected]
            reference_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            reference_gray = processor._limit_long_edge(reference_gray, 700)
            metrics = {
                "candidate": path.name,
                "good": len(good),
                "inliers": int(selected.sum()),
                "inlier_ratio": float(selected.mean()),
                "query_hull_ratio": hull_area_ratio(inlier_source, query_gray.shape[1], query_gray.shape[0]),
                "reference_hull_ratio": hull_area_ratio(
                    inlier_target, reference_gray.shape[1], reference_gray.shape[0]
                ),
                "query_quadrants": quadrant_count(inlier_source, query_gray.shape[1], query_gray.shape[0]),
                "reference_quadrants": quadrant_count(
                    inlier_target, reference_gray.shape[1], reference_gray.shape[0]
                ),
                "query_grid_coverage": grid_coverage(inlier_source, query_gray.shape[1], query_gray.shape[0]),
                "reference_grid_coverage": grid_coverage(
                    inlier_target, reference_gray.shape[1], reference_gray.shape[0]
                ),
                **projected_quad_metrics(
                    homography,
                    query_gray.shape[1],
                    query_gray.shape[0],
                    reference_gray.shape[1],
                    reference_gray.shape[0],
                ),
                **symmetric_reprojection_metrics(homography, source, target, selected),
            }
            accepted, reasons = stage2(metrics)
            metrics["stage1_pass"] = (
                metrics["inliers"] >= 14 and metrics["inlier_ratio"] >= 0.55
            )
            metrics["stage2_pass_geometry"] = accepted
            metrics["stage2_reasons"] = reasons
            candidates.append(metrics)
            visual_payload[path.name] = (
                reference_gray,
                reference_points,
                good,
                mask,
            )
        candidates.sort(key=lambda item: (item["inliers"], item["inlier_ratio"]), reverse=True)
        top = candidates[:8]
        expected = filename.replace("photo_", "screenshot_").rsplit(".", 1)[0] + ".png"
        report[filename] = {"expected": expected, "top": top}
        for rank, candidate in enumerate(top[:3] if filename in FOCUS else [], 1):
            reference_gray, reference_points, good, mask = visual_payload[candidate["candidate"]]
            visual = draw_match(
                query_gray,
                query_points,
                reference_gray,
                reference_points,
                good,
                mask,
            )
            cv2.imwrite(
                str(OUT / f"{Path(filename).stem}_rank{rank}_{Path(candidate['candidate']).stem}.jpg"),
                visual,
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )
        print(filename, expected, [(x["candidate"], x["inliers"], x["stage2_pass_geometry"]) for x in top[:5]])
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
