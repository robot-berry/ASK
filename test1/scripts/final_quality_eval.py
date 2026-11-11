from __future__ import annotations

"""Final, read-only quality gate for question-image evaluation folders.

The evaluator never calls or mutates the production processor.  It accepts the
batch ``results.csv`` written by scripts/evaluate_folder.py, the compact
``results.json`` used by spot checks, or reconstructs a photographic spot check
from its Markdown filenames.  Reference metrics are emitted only after a
geometrically reliable SIFT/RANSAC alignment; weak or known-mismatched pairs do
not contaminate aggregate acceptance numbers.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Dataset inspection proved that these nominal same-name pairs are not the same
# question (or are too weak to establish correspondence). Add entries only after
# visually verifying source and electronic reference.
DEFAULT_EXCLUDED_PAIRS = {
    "photo_107_0242.jpg",
    "photo_201_0773.jpg",
}


def _read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise RuntimeError(f"cannot read image: {path}")
    return image


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _round(value: Any, digits: int = 6) -> Any:
    return round(float(value), digits) if value is not None else None


def _output_path(evaluation: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    if (evaluation / path).exists():
        return evaluation / path
    if (evaluation / "artifacts" / path).exists():
        return evaluation / "artifacts" / path
    return evaluation / path


def _photo_from_markdown(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"photo_\d+_\d+\.(?:jpe?g|png|webp)", text, re.I)
    return match.group(0) if match else ""


def _load_rows(evaluation: Path) -> tuple[list[dict[str, Any]], str]:
    csv_path = evaluation / "results.csv"
    json_path = evaluation / "results.json"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle)), "results.csv"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("results.json must contain a list")
        rows = []
        for item in payload:
            row = dict(item)
            row.setdefault("cleanup_method", row.get("method", ""))
            row.setdefault("reference_match", row.get("match", ""))
            row.setdefault("reference_confidence", row.get("conf", ""))
            row.setdefault("rotation", row.get("rot", ""))
            row.setdefault("deskew_applied_angle", row.get("skew", ""))
            row.setdefault("deskew_status", row.get("skew_status", ""))
            row.setdefault("job_id", row.get("job", ""))
            row.setdefault("output_png", row.get("png", ""))
            rows.append(row)
        return rows, "results.json"

    # A running spot check may have artifacts before its manifest is written.
    # Reconstruct only unambiguous Markdown-to-job mappings and mark all unknown
    # processor metadata as unavailable, never inferred.
    rows = []
    for markdown in sorted((evaluation / "markdown").glob("*.md")):
        name = _photo_from_markdown(markdown)
        if not name:
            continue
        job = markdown.stem
        output = evaluation / "png" / f"{job}.png"
        if not output.exists():
            continue
        rows.append(
            {
                "name": name,
                "job_id": job,
                "output_png": f"png/{job}.png",
                "cleanup_method": "photographic",
                "reference_match": "",
                "reference_confidence": "",
                "manifest_reconstructed": True,
            }
        )
    return rows, "reconstructed-markdown"


def _question_paths(evaluation: Path, row: dict[str, Any]) -> list[Path]:
    job = str(row.get("job_id") or row.get("job") or "")
    if not job:
        return []
    roots = [evaluation / "questions" / job, evaluation / "artifacts" / "questions" / job]
    for root in roots:
        cards = sorted(root.glob("q*.png"))
        if cards:
            return cards
    return []


def _reference_path(
    row: dict[str, Any], reference_root: Path, excluded: set[str]
) -> tuple[Path | None, str]:
    name = str(row.get("name", ""))
    explicit = str(row.get("reference_match") or row.get("match") or "").strip()
    method = str(row.get("cleanup_method") or row.get("method") or "")
    if method == "matched-electronic" and explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = reference_root / path.name
        return (path, "reference_match") if path.exists() else (None, "missing-reference-match")
    if name in excluded:
        return None, "known-mismatch"
    if not name:
        return None, "missing-photo-name"
    candidate = reference_root / (Path(name).stem.replace("photo_", "screenshot_") + ".png")
    return (candidate, "same-name") if candidate.exists() else (None, "missing-same-name")


def _scaled_gray(image: np.ndarray, longest: int = 1800) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = min(1.0, longest / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, scale


def _align(reference: np.ndarray, output: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    ref_gray, ref_scale = _scaled_gray(reference)
    out_gray, out_scale = _scaled_gray(output)
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.015, edgeThreshold=12)
    ref_points, ref_desc = sift.detectAndCompute(ref_gray, None)
    out_points, out_desc = sift.detectAndCompute(out_gray, None)
    result: dict[str, Any] = {
        "alignment_reliable": False,
        "alignment_matches": 0,
        "alignment_inliers": 0,
        "alignment_inlier_ratio": 0.0,
        "alignment_median_error_px": None,
        "direction_residual_deg": None,
    }
    if ref_desc is None or out_desc is None:
        result["alignment_reason"] = "no-descriptors"
        return None, result
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(ref_desc, out_desc, k=2)
    reverse = matcher.knnMatch(out_desc, ref_desc, k=2)
    forward_good = {m.queryIdx: m.trainIdx for m, n in forward if m.distance < 0.76 * n.distance}
    reverse_good = {m.queryIdx: m.trainIdx for m, n in reverse if m.distance < 0.76 * n.distance}
    mutual = [(q, t) for q, t in forward_good.items() if reverse_good.get(t) == q]
    if len(mutual) < 8:
        mutual = list(forward_good.items())
    result["alignment_matches"] = len(mutual)
    if len(mutual) < 6:
        result["alignment_reason"] = "too-few-matches"
        return None, result
    ref_xy = np.float32([ref_points[q].pt for q, _ in mutual]) / ref_scale
    out_xy = np.float32([out_points[t].pt for _, t in mutual]) / out_scale
    homography, mask = cv2.findHomography(ref_xy, out_xy, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        result["alignment_reason"] = "homography-failed"
        return None, result
    inliers = mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    ratio = float(inliers.mean())
    result["alignment_inliers"] = inlier_count
    result["alignment_inlier_ratio"] = ratio
    projected = cv2.perspectiveTransform(ref_xy[inliers, None, :], homography)[:, 0]
    errors = np.linalg.norm(projected - out_xy[inliers], axis=1)
    median_error = float(np.median(errors)) if len(errors) else 999.0
    result["alignment_median_error_px"] = median_error

    center = np.array([reference.shape[1] / 2, reference.shape[0] / 2], np.float32)
    span = max(100.0, reference.shape[1] * 0.25)
    probe = np.array([[center, center + (span, 0)]], np.float32)
    mapped = cv2.perspectiveTransform(probe, homography)[0]
    vector = mapped[1] - mapped[0]
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    angle = ((angle + 45.0) % 90.0) - 45.0
    result["direction_residual_deg"] = float(angle)
    reliable = inlier_count >= 20 and ratio >= 0.45 and median_error <= 3.0 and abs(angle) <= 12.0
    result["alignment_reliable"] = reliable
    result["alignment_reason"] = "reliable" if reliable else "weak-geometry"
    return (homography if reliable else None), result


def _reference_metrics(
    reference: np.ndarray, output: np.ndarray, homography: np.ndarray
) -> dict[str, float]:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    out_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    height, width = out_gray.shape
    # Use relative darkness so anti-aliased light-gray diagrams remain content.
    # A fixed <180 threshold falsely labels identical matched-electronic images
    # as missing their gray fills.
    def content_mask(gray: np.ndarray) -> np.ndarray:
        paper = float(np.percentile(gray, 90))
        threshold = min(247.0, max(180.0, paper - 8.0))
        return gray.astype(np.float32) < threshold

    ref_ink = content_mask(ref_gray)
    out_ink = content_mask(out_gray)
    valid = cv2.warpPerspective(
        np.full(ref_gray.shape, 255, np.uint8), homography, (width, height), flags=cv2.INTER_NEAREST
    ) > 0
    warped_ink = cv2.warpPerspective(
        ref_ink.astype(np.uint8) * 255, homography, (width, height), flags=cv2.INTER_NEAREST
    ) > 0
    tolerance = max(2, int(round(min(height, width) * 0.004)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance + 1,) * 2)
    reference_neighborhood = cv2.dilate(warped_ink.astype(np.uint8), kernel) > 0
    background = valid & ~reference_neighborhood
    false_ink = out_ink & background
    visible_output_ink = out_ink & valid

    # Exact matched-electronic outputs can differ only in PNG decoding/layout.
    # Normalized pixel agreement is a useful sanity assertion for this path and
    # prevents a mask-definition error from failing a pixel-identical restore.
    exact_same_shape = reference.shape[:2] == output.shape[:2]
    exact_mean_abs_error = (
        float(np.mean(np.abs(reference.astype(np.float32) - output.astype(np.float32))))
        if exact_same_shape
        else None
    )

    yy, xx = np.nonzero(ref_ink)
    if len(xx) > 120_000:
        indexes = np.linspace(0, len(xx) - 1, 120_000, dtype=np.int64)
        xx, yy = xx[indexes], yy[indexes]
    points = np.column_stack((xx, yy)).astype(np.float32)[None]
    mapped = cv2.perspectiveTransform(points, homography)[0]
    inside = (
        (mapped[:, 0] >= 0)
        & (mapped[:, 0] < width)
        & (mapped[:, 1] >= 0)
        & (mapped[:, 1] < height)
    )
    # distanceTransform measures distance to zero pixels. Here content is the
    # zero class, so a reference point is retained when this distance is small.
    distance = cv2.distanceTransform((~out_ink).astype(np.uint8), cv2.DIST_L2, 3)
    retained = np.zeros(len(mapped), dtype=bool)
    inside_index = np.flatnonzero(inside)
    mx = np.clip(np.rint(mapped[inside_index, 0]).astype(int), 0, width - 1)
    my = np.clip(np.rint(mapped[inside_index, 1]).astype(int), 0, height - 1)
    retained[inside_index] = distance[my, mx] <= tolerance
    content_retention = float(retained.mean())
    content_coverage = float(inside.mean())
    background_false_ink = float(false_ink.sum() / max(int(background.sum()), 1))
    unmatched_output = float(false_ink.sum() / max(int(visible_output_ink.sum()), 1))
    if exact_mean_abs_error is not None and exact_mean_abs_error <= 0.5:
        content_retention = 1.0
        content_coverage = 1.0
        background_false_ink = 0.0
        unmatched_output = 0.0
    return {
        "content_retention": content_retention,
        "reference_content_coverage": content_coverage,
        "background_false_ink_rate": background_false_ink,
        "unmatched_output_ink_fraction": unmatched_output,
        "reference_exact_mean_abs_error": exact_mean_abs_error,
    }


def _self_ink_metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ink = gray < 180
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([], dtype=int)
    mpix = max(gray.size / 1_000_000.0, 1e-6)
    small = int(((areas >= 1) & (areas <= 24)).sum())
    large = int(areas[areas >= max(500, int(gray.size * 0.002))].sum()) if len(areas) else 0
    return {
        "self_ink_density": float(ink.mean()),
        "self_small_noise_components_per_mpix": float(small / mpix),
        "self_large_dark_region_rate": float(large / gray.size),
    }


def _self_direction(image: np.ndarray) -> dict[str, Any]:
    """Estimate residual baseline angle without a reference.

    Hough segments are filtered to long, near-horizontal strokes.  A result is
    intentionally withheld when fewer than three independent segments agree.
    """
    gray, _ = _scaled_gray(image, 1600)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(binary, 50, 150)
    min_length = max(50, int(gray.shape[1] * 0.12))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 360.0, threshold=50, minLineLength=min_length, maxLineGap=20
    )
    candidates: list[tuple[float, float]] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            angle = ((angle + 45.0) % 90.0) - 45.0
            if length >= min_length and abs(angle) <= 12.0:
                candidates.append((angle, length))
    if len(candidates) < 3:
        return {
            "self_direction_residual_deg": None,
            "self_direction_confidence": 0.0,
            "self_direction_status": "insufficient-lines",
            "self_direction_line_count": len(candidates),
        }
    ordered = sorted(candidates)
    total = sum(weight for _, weight in ordered)
    cumulative = 0.0
    median = ordered[-1][0]
    for angle, weight in ordered:
        cumulative += weight
        if cumulative >= total / 2:
            median = angle
            break
    agreeing = sum(weight for angle, weight in candidates if abs(angle - median) <= 1.25)
    confidence = agreeing / max(total, 1.0)
    return {
        "self_direction_residual_deg": float(median),
        "self_direction_confidence": float(confidence),
        "self_direction_status": "reliable" if confidence >= 0.55 else "low-consensus",
        "self_direction_line_count": len(candidates),
    }


def _card_metrics(output: np.ndarray, cards: list[np.ndarray]) -> dict[str, float | int | None]:
    if not cards:
        return {
            "question_card_count": 0,
            "question_card_area_ratio": None,
            "question_card_height_ratio": None,
            "question_card_ink_coverage": None,
        }
    output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY) if output.ndim == 3 else output
    output_threshold = min(247.0, max(180.0, float(np.percentile(output_gray, 90)) - 8.0))
    output_ink = int((output_gray < output_threshold).sum())
    card_area = sum(card.shape[0] * card.shape[1] for card in cards)
    card_ink = sum(
        int(
            (
                (cv2.cvtColor(card, cv2.COLOR_BGR2GRAY) if card.ndim == 3 else card)
                < min(
                    247.0,
                    max(
                        180.0,
                        float(
                            np.percentile(
                                cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
                                if card.ndim == 3
                                else card,
                                90,
                            )
                        )
                        - 8.0,
                    ),
                )
            ).sum()
        )
        for card in cards
    )
    largest_height = max(card.shape[0] for card in cards)
    return {
        "question_card_count": len(cards),
        "question_card_area_ratio": float(card_area / max(output_gray.size, 1)),
        "question_card_height_ratio": float(largest_height / max(output_gray.shape[0], 1)),
        # Cards can overlap in a multi-question page, so clamp the diagnostic.
        "question_card_ink_coverage": float(min(1.0, card_ink / max(output_ink, 1))),
    }


def _gate(summary: dict[str, Any]) -> dict[str, Any]:
    checks = []

    def add(name: str, observed: Any, operator: str, target: Any, passed: bool | None) -> None:
        checks.append(
            {"name": name, "observed": observed, "operator": operator, "target": target, "pass": passed}
        )

    add("success_rate", summary["success_rate"], ">=", 1.0, summary["success_rate"] >= 1.0)
    add(
        "reference_mean_abs_direction_deg",
        summary["reference_mean_abs_direction_deg"],
        "<=",
        0.8,
        None
        if summary["reference_mean_abs_direction_deg"] is None
        else summary["reference_mean_abs_direction_deg"] <= 0.8,
    )
    add(
        "reference_over_2deg_count",
        summary["reference_over_2deg_count"],
        "<=",
        1,
        None if summary["reference_pair_count"] == 0 else summary["reference_over_2deg_count"] <= 1,
    )
    add(
        "reference_mean_background_false_ink_rate",
        summary["reference_mean_background_false_ink_rate"],
        "<=",
        0.015,
        None
        if summary["reference_mean_background_false_ink_rate"] is None
        else summary["reference_mean_background_false_ink_rate"] <= 0.015,
    )
    add(
        "reference_mean_content_retention",
        summary["reference_mean_content_retention"],
        ">=",
        0.98,
        None
        if summary["reference_mean_content_retention"] is None
        else summary["reference_mean_content_retention"] >= 0.98,
    )
    add(
        "minimum_question_card_ink_coverage",
        summary["minimum_question_card_ink_coverage"],
        ">=",
        0.95,
        None
        if summary["minimum_question_card_ink_coverage"] is None
        else summary["minimum_question_card_ink_coverage"] >= 0.95,
    )
    add(
        "minimum_reference_content_coverage",
        summary["reference_min_content_coverage"],
        ">=",
        0.95,
        None
        if summary["reference_min_content_coverage"] is None
        else summary["reference_min_content_coverage"] >= 0.95,
    )
    add(
        "maximum_self_small_noise_components_per_mpix",
        summary["maximum_self_small_noise_components_per_mpix"],
        "<=",
        8000.0,
        None
        if summary["photographic_count"] == 0
        else summary["maximum_self_small_noise_components_per_mpix"] <= 8000.0,
    )
    evaluated = [item for item in checks if item["pass"] is not None]
    return {
        "status": "pass" if evaluated and all(item["pass"] for item in evaluated) else "fail",
        "coverage_complete": all(item["pass"] is not None for item in checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Final objective quality gate for an evaluation folder")
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--exclude-pair", action="append", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    evaluation = args.evaluation.resolve()
    output_dir = (args.out or (evaluation / "final_quality")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded = DEFAULT_EXCLUDED_PAIRS | {Path(name).name for name in args.exclude_pair}
    rows, manifest_source = _load_rows(evaluation)
    records: list[dict[str, Any]] = []

    for row in rows:
        name = str(row.get("name", ""))
        record: dict[str, Any] = {
            "name": name,
            "cleanup_method": str(
                row.get("cleanup_method")
                or row.get("method")
                or ("matched-electronic" if row.get("reference_match") or row.get("match") else "photographic")
            ),
            "reference_match": str(row.get("reference_match") or row.get("match") or ""),
            "reference_confidence": row.get("reference_confidence") or row.get("conf") or None,
            "ok": str(row.get("ok", "true")).lower() not in {"false", "0", "no"},
        }
        try:
            output_relative = str(row.get("output_png") or row.get("png") or "")
            output_path = _output_path(evaluation, output_relative)
            output = _read_image(output_path)
            record["output_path"] = str(output_path)
            record.update(_self_ink_metrics(output))
            record.update(_self_direction(output))
            card_paths = _question_paths(evaluation, row)
            cards = [_read_image(path) for path in card_paths]
            record.update(_card_metrics(output, cards))
            record["question_card_paths"] = [str(path) for path in card_paths]
            reference_path, reference_source = _reference_path(row, args.references, excluded)
            record["reference_source"] = reference_source
            record["reference_path"] = str(reference_path) if reference_path else ""
            if reference_path is not None:
                reference = _read_image(reference_path)
                homography, alignment = _align(reference, output)
                record.update(alignment)
                if homography is not None:
                    record.update(_reference_metrics(reference, output, homography))
            else:
                record.update(
                    alignment_reliable=False,
                    alignment_reason=reference_source,
                    direction_residual_deg=None,
                    content_retention=None,
                    reference_content_coverage=None,
                    background_false_ink_rate=None,
                    unmatched_output_ink_fraction=None,
                )
        except Exception as error:
            record["ok"] = False
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
        print(
            f"{name}: {record.get('cleanup_method')} "
            f"ref={record.get('alignment_reason', record.get('reference_source'))} "
            f"card={record.get('question_card_ink_coverage')}",
            flush=True,
        )

    # Flatten rows into a stable CSV; list fields become JSON strings.
    csv_records = []
    for record in records:
        csv_records.append(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in record.items()
            }
        )
    fields = sorted({key for record in csv_records for key in record})
    with (output_dir / "final_quality_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_records)

    reliable = [record for record in records if record.get("alignment_reliable")]
    photographic = [record for record in records if record.get("cleanup_method") == "photographic"]
    card_coverage = [
        float(record["question_card_ink_coverage"])
        for record in records
        if record.get("question_card_ink_coverage") is not None
    ]
    direction = [abs(float(record["direction_residual_deg"])) for record in reliable]
    summary: dict[str, Any] = {
        "evaluation": str(evaluation),
        "manifest_source": manifest_source,
        "sample_count": len(records),
        "success_count": sum(bool(record.get("ok")) for record in records),
        "success_rate": sum(bool(record.get("ok")) for record in records) / max(len(records), 1),
        "cleanup_methods": dict(Counter(str(record.get("cleanup_method")) for record in records)),
        "known_mismatch_exclusions": sorted(excluded),
        "reference_pair_count": len(reliable),
        "reference_unreliable_count": sum(
            bool(record.get("reference_path")) and not record.get("alignment_reliable") for record in records
        ),
        "reference_mean_abs_direction_deg": _round(_mean(direction)),
        "reference_p95_abs_direction_deg": _round(_percentile(direction, 95)),
        "reference_over_2deg_count": sum(value > 2.0 for value in direction),
        "reference_mean_content_retention": _round(
            _mean([float(record["content_retention"]) for record in reliable])
        ),
        "reference_min_content_retention": _round(
            min((float(record["content_retention"]) for record in reliable), default=None)
        ),
        "reference_mean_content_coverage": _round(
            _mean([float(record["reference_content_coverage"]) for record in reliable])
        ),
        "reference_min_content_coverage": _round(
            min((float(record["reference_content_coverage"]) for record in reliable), default=None)
        ),
        "reference_mean_background_false_ink_rate": _round(
            _mean([float(record["background_false_ink_rate"]) for record in reliable])
        ),
        "reference_max_background_false_ink_rate": _round(
            max((float(record["background_false_ink_rate"]) for record in reliable), default=None)
        ),
        "photographic_count": len(photographic),
        "photographic_self_direction_reliable_count": sum(
            record.get("self_direction_status") == "reliable" for record in photographic
        ),
        "photographic_mean_abs_self_direction_deg": _round(
            _mean(
                [
                    abs(float(record["self_direction_residual_deg"]))
                    for record in photographic
                    if record.get("self_direction_status") == "reliable"
                ]
            )
        ),
        "mean_self_small_noise_components_per_mpix": _round(
            _mean(
                [float(record["self_small_noise_components_per_mpix"]) for record in photographic]
            )
        ),
        "maximum_self_small_noise_components_per_mpix": _round(
            max(
                (
                    float(record["self_small_noise_components_per_mpix"])
                    for record in photographic
                ),
                default=0.0,
            )
        ),
        "minimum_question_card_ink_coverage": _round(min(card_coverage, default=None)),
        "mean_question_card_ink_coverage": _round(_mean(card_coverage)),
        "samples_without_cards": [
            record.get("name") for record in records if not record.get("question_card_count")
        ],
        "unreliable_references": [
            {
                "name": record.get("name"),
                "source": record.get("reference_source"),
                "reason": record.get("alignment_reason"),
                "inliers": record.get("alignment_inliers"),
                "ratio": record.get("alignment_inlier_ratio"),
            }
            for record in records
            if record.get("reference_path") and not record.get("alignment_reliable")
        ],
    }
    summary["gate"] = _gate(summary)
    (output_dir / "final_quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def pct(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.2%}"

    def num(value: Any, suffix: str = "") -> str:
        return "N/A" if value is None else f"{float(value):.2f}{suffix}"

    lines = [
        "# Final objective quality gate",
        "",
        f"- Evaluation: `{evaluation}`",
        f"- Manifest: `{manifest_source}`",
        f"- Samples: {len(records)}; successful: {summary['success_count']}",
        f"- Cleanup methods: {summary['cleanup_methods']}",
        f"- Reliable electronic-reference pairs: {len(reliable)}",
        "",
        "## Reference-backed metrics",
        "",
        f"- Mean absolute direction residual: {num(summary['reference_mean_abs_direction_deg'], ' deg')}",
        f"- P95 absolute direction residual: {num(summary['reference_p95_abs_direction_deg'], ' deg')}",
        f"- Residual >2 deg: {summary['reference_over_2deg_count']}",
        f"- Mean content retention: {pct(summary['reference_mean_content_retention'])}",
        f"- Minimum content retention: {pct(summary['reference_min_content_retention'])}",
        f"- Mean reference content inside output: {pct(summary['reference_mean_content_coverage'])}",
        f"- Minimum reference content inside output: {pct(summary['reference_min_content_coverage'])}",
        f"- Mean background false ink: {pct(summary['reference_mean_background_false_ink_rate'])}",
        f"- Maximum background false ink: {pct(summary['reference_max_background_false_ink_rate'])}",
        "",
        "## Photographic/self metrics",
        "",
        f"- Photographic samples: {len(photographic)}",
        f"- Reliable self-direction estimates: {summary['photographic_self_direction_reliable_count']}",
        f"- Mean absolute self direction: {num(summary['photographic_mean_abs_self_direction_deg'], ' deg')}",
        f"- Mean small noise components/Mpix: {num(summary['mean_self_small_noise_components_per_mpix'])}",
        f"- Maximum small noise components/Mpix: {num(summary['maximum_self_small_noise_components_per_mpix'])}",
        "",
        "## Question-card coverage",
        "",
        f"- Mean output-ink coverage: {pct(summary['mean_question_card_ink_coverage'])}",
        f"- Minimum output-ink coverage: {pct(summary['minimum_question_card_ink_coverage'])}",
        f"- Samples without cards: {summary['samples_without_cards']}",
        "",
        "## Gate",
        "",
        f"Status: **{summary['gate']['status'].upper()}**; complete evidence: {summary['gate']['coverage_complete']}",
        "",
        "| Check | Observed | Requirement | Result |",
        "|---|---:|---:|---:|",
    ]
    for check in summary["gate"]["checks"]:
        result = "N/A" if check["pass"] is None else ("PASS" if check["pass"] else "FAIL")
        lines.append(
            f"|{check['name']}|{check['observed']}|{check['operator']} {check['target']}|{result}|"
        )
    (output_dir / "final_quality_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
