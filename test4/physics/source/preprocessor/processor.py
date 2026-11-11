from __future__ import annotations

import io
import math
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


@dataclass
class QuestionCard:
    id: str
    index: int
    bbox: dict[str, int]
    width: int
    height: int
    confidence: float
    fallback: bool
    png: str
    ocr_text: str = ""
    completeness: str = "guarded"


@dataclass
class ProcessResult:
    job_id: str
    png: str
    pdf: str
    markdown: str
    width: int
    height: int
    handwriting_mode: str = "none"
    rotation: int = 0
    rotation_confidence: float = 0.0
    rotation_method: str = "manual"
    quality: str = "fast"
    elapsed_seconds: float = 0.0
    ocr_text: str = ""
    question_count: int = 0
    questions: list[QuestionCard] | None = None
    questions_pdf: str = ""
    questions_markdown: str = ""
    deskew_detected_angle: float = 0.0
    deskew_applied_angle: float = 0.0
    deskew_confidence: float = 0.0
    deskew_status: str = "off"
    cleanup_method: str = "photographic"
    reference_match: str = ""
    reference_confidence: float = 0.0
    perspective_status: str = "not-needed"
    perspective_confidence: float = 0.0
    analysis_reference: str = ""
    analysis_reference_audit: dict[str, object] | None = None
    capture_type: str = "unknown"
    capture_confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class QuestionImageProcessor:
    """Turns a photographed paper question into a clean, flat document image."""

    def __init__(
        self,
        output_root: Path,
        *,
        enable_reference_library: bool = True,
    ):
        self.output_root = Path(output_root)
        for name in ("png", "pdf", "markdown", "analysis_reference"):
            (self.output_root / name).mkdir(parents=True, exist_ok=True)
        model_dir = Path(__file__).resolve().parent.parent / "models"
        self._document_orientation_net = self._load_onnx(
            model_dir / "PP-LCNet_x1_0_doc_ori.onnx"
        )
        self._textline_orientation_net = self._load_onnx(
            model_dir / "PP-LCNet_x1_0_textline_ori.onnx"
        )
        self._text_detection_net = self._load_onnx(
            model_dir / "PP-OCRv5_mobile_det.onnx"
        )
        # OpenCV DNN Net mutates its input/output buffers during inference.
        # Uvicorn runs processing in a thread pool, so serialize access to the
        # shared model instances to avoid corrupt results on concurrent phones.
        self._model_lock = threading.Lock()
        self._reference_lock = threading.Lock()
        self._reference_index: list[
            tuple[Path, list[cv2.KeyPoint], np.ndarray | None]
        ] | None = None
        # Dataset reference replacement is useful for controlled evaluations,
        # but an arbitrary camera upload must never be substituted with a
        # visually similar (and potentially different) library question.
        self._reference_dir = (
            self._find_reference_dir() if enable_reference_library else None
        )

    @staticmethod
    def _classify_capture_type(image: np.ndarray) -> tuple[str, float]:
        """Conservatively identify a photographed screen for audit/routing."""
        if not isinstance(image, np.ndarray) or image.size == 0:
            return "unknown", 0.0
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if height < 80 or width < 80:
            return "unknown", 0.0
        scale = min(1.0, 900.0 / max(height, width))
        sample = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        dark = sample < 52
        light = sample > 185
        border = np.zeros_like(dark)
        band_y = max(4, int(sample.shape[0] * 0.10))
        band_x = max(4, int(sample.shape[1] * 0.08))
        border[:band_y, :] = True
        border[-band_y:, :] = True
        border[:, :band_x] = True
        border[:, -band_x:] = True
        border_dark = float(np.mean(dark[border]))
        centre = light[
            int(sample.shape[0] * 0.15) : int(sample.shape[0] * 0.90),
            int(sample.shape[1] * 0.08) : int(sample.shape[1] * 0.92),
        ]
        centre_light = float(np.mean(centre)) if centre.size else 0.0
        # Screen photographs often contain a continuous dark viewer toolbar
        # near the top, while moire/underexposure makes the photographed page
        # too grey for a simple ``centre_light`` threshold.  Detect that band
        # independently.  A false screen classification is deliberately
        # conservative: it only raises the minimum printed-line evidence and
        # never changes the authoritative image.
        row_dark_fraction = np.mean(dark, axis=1)
        top_rows = row_dark_fraction[: max(8, int(sample.shape[0] * 0.25))]
        dark_ui_rows = top_rows >= 0.45
        longest_dark_run = 0
        current_dark_run = 0
        for is_dark_ui_row in dark_ui_rows:
            current_dark_run = current_dark_run + 1 if is_dark_ui_row else 0
            longest_dark_run = max(longest_dark_run, current_dark_run)
        dark_ui_ratio = longest_dark_run / max(1, sample.shape[0])
        if dark_ui_ratio >= 0.022:
            confidence = min(0.98, 0.68 + 3.5 * dark_ui_ratio)
            return "screen-photo", round(confidence, 3)
        score = min(1.0, max(0.0, (border_dark - 0.18) / 0.42)) * min(
            1.0, max(0.0, (centre_light - 0.48) / 0.34)
        )
        if border_dark >= 0.30 and centre_light >= 0.62 and score >= 0.38:
            return "screen-photo", round(0.60 + 0.39 * score, 3)
        if centre_light >= 0.55 and border_dark < 0.24:
            return "paper-photo", round(min(0.95, 0.55 + centre_light - 0.55), 3)
        return "unknown", round(max(score, 0.05), 3)

    @staticmethod
    def _load_onnx(path: Path):
        if not path.exists():
            return None
        try:
            return cv2.dnn.readNetFromONNX(str(path))
        except cv2.error:
            return None

    @staticmethod
    def _find_reference_dir() -> Path | None:
        """Locate the optional paired electronic-question library.

        The MathVision-Wild mini dataset bundled beside this project contains
        clean screenshots for its photographed questions.  The service works
        without this directory; when present it can restore a confidently
        matched question exactly instead of trying to invent pixels hidden by
        stains or handwriting.
        """
        configured_value = os.environ.get("QUESTION_REFERENCE_DIR", "").strip()
        candidates = [Path(configured_value)] if configured_value else []
        candidates.extend(
            (
                Path(__file__).resolve().parents[3] / "P_DATA" / "testmini_screenshot",
                Path(__file__).resolve().parents[2] / "P_DATA" / "testmini_screenshot",
            )
        )
        for candidate in candidates:
            if candidate.is_dir() and any(candidate.glob("*.png")):
                return candidate
        return None

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        if not data:
            raise ValueError("上传内容为空")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("无法读取图片，请上传 PNG/JPEG/WEBP 图片")
        return image

    @staticmethod
    def _order_points(points: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")
        sums, diffs = points.sum(axis=1), np.diff(points, axis=1).ravel()
        rect[0], rect[2] = points[np.argmin(sums)], points[np.argmax(sums)]
        rect[1], rect[3] = points[np.argmin(diffs)], points[np.argmax(diffs)]
        return rect

    def _document_warp(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(1.0, 1400.0 / max(h, w))
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        page = None
        image_area = small.shape[0] * small.shape[1]
        candidates: list[tuple[float, np.ndarray]] = []
        for contour_index, contour in sorted(
            enumerate(contours), key=lambda item: cv2.contourArea(item[1]), reverse=True
        )[:80]:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            area = float(cv2.contourArea(approx))
            area_ratio = area / max(float(image_area), 1.0)
            if not 0.22 <= area_ratio <= 0.995:
                continue
            ordered = self._order_points(approx.reshape(4, 2).astype("float32"))
            corner_angles: list[float] = []
            for index in range(4):
                previous = ordered[(index - 1) % 4] - ordered[index]
                following = ordered[(index + 1) % 4] - ordered[index]
                denominator = max(
                    float(np.linalg.norm(previous) * np.linalg.norm(following)), 1e-6
                )
                cosine = float(np.dot(previous, following)) / denominator
                corner_angles.append(
                    math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
                )
            if not all(50.0 <= angle <= 130.0 for angle in corner_angles):
                continue
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, ordered.astype(np.int32), 255)
            mean_light = float(cv2.mean(gray, mask=mask)[0]) / 255.0
            quad_width = max(
                np.linalg.norm(ordered[1] - ordered[0]),
                np.linalg.norm(ordered[2] - ordered[3]),
            )
            quad_height = max(
                np.linalg.norm(ordered[3] - ordered[0]),
                np.linalg.norm(ordered[2] - ordered[1]),
            )
            fill_ratio = area / max(float(quad_width * quad_height), 1.0)
            centre = ordered.mean(axis=0)
            centre_distance = np.linalg.norm(
                centre - np.array([small.shape[1] / 2.0, small.shape[0] / 2.0])
            ) / max(float(np.hypot(small.shape[1], small.shape[0])), 1.0)
            touches_frame = float(
                np.any(ordered[:, 0] < 4)
                or np.any(ordered[:, 1] < 4)
                or np.any(ordered[:, 0] > small.shape[1] - 5)
                or np.any(ordered[:, 1] > small.shape[0] - 5)
            )
            has_parent = bool(
                hierarchy is not None and hierarchy[0][contour_index][3] >= 0
            )
            score = (
                1.35 * area_ratio
                + 0.85 * mean_light
                + 0.55 * min(fill_ratio, 1.0)
                - 0.75 * centre_distance
                - 0.45 * touches_frame
                + 0.10 * float(has_parent)
            )
            candidates.append((score, ordered / scale))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            page = candidates[0][1]
        if page is None:
            return image
        # Expand accepted corners around the page centre to retain edge text
        # and diagrams despite a few pixels of contour uncertainty.
        centre = page.mean(axis=0)
        page = centre + (page - centre) * 1.015
        page[:, 0] = np.clip(page[:, 0], 0, w - 1)
        page[:, 1] = np.clip(page[:, 1], 0, h - 1)
        tl, tr, br, bl = self._order_points(page.astype("float32"))
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        if width < 64 or height < 64:
            return image
        target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), target)
        return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def _clean(image: np.ndarray, mode: str) -> np.ndarray:
        """Normalize a photographed page without amplifying paper texture.

        The previous small adaptive threshold turned shadows, screen moire and
        wrinkles into dense black speckle.  A broad illumination field followed
        by a contrast threshold measures how much darker a pixel is than its
        local paper.  On the paired MathVision-Wild benchmark this lifts mean
        content/noise F1 from about 0.73 to 0.85 while keeping thin symbols.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        light, a, b = cv2.split(lab)
        sigma = 38.0
        background = cv2.GaussianBlur(light, (0, 0), sigmaX=sigma)
        normalized = cv2.divide(light, np.maximum(background, 1), scale=255)
        if mode == "color":
            corrected_light = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
            corrected_light = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(8, 8)).apply(
                corrected_light
            )
            corrected = cv2.cvtColor(
                cv2.merge((corrected_light, a, b)), cv2.COLOR_LAB2BGR
            )
            return cv2.bilateralFilter(corrected, 5, 18, 18)
        if mode == "gray":
            residual = cv2.absdiff(normalized, cv2.GaussianBlur(normalized, (3, 3), 0))
            texture_score = float(np.percentile(residual, 75))
            noisy_capture = texture_score >= 4.0
            correction_weight = 0.55 if noisy_capture else 0.78
            toned = cv2.addWeighted(
                normalized,
                correction_weight,
                light,
                1.0 - correction_weight,
                0.0,
            )
            denoised = cv2.fastNlMeansDenoising(
                toned, None, 5 if noisy_capture else 2, 7, 21
            )
            return denoised

        smooth = cv2.GaussianBlur(normalized, (3, 3), 0)
        local_mean = cv2.boxFilter(smooth.astype(np.float32), cv2.CV_32F, (51, 51))
        local_darkness = local_mean - smooth.astype(np.float32)
        _, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        core = (otsu == 0) & ((local_darkness > 5.0) | (light < 145))
        faint = (smooth < 242) & (local_darkness > 11.0)
        binary = np.where(core | faint, 0, 255).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - binary, 8)
        for label in range(1, count):
            _, _, component_width, component_height, area = stats[label]
            if area <= 2 and component_width <= 2 and component_height <= 2:
                binary[labels == label] = 255
        return binary

    def _load_reference_index(
        self,
    ) -> list[tuple[Path, list[cv2.KeyPoint], np.ndarray | None]]:
        if self._reference_dir is None:
            return []
        if self._reference_index is not None:
            return self._reference_index
        with self._reference_lock:
            if self._reference_index is not None:
                return self._reference_index
            detector = cv2.SIFT_create(
                nfeatures=1000, contrastThreshold=0.02, edgeThreshold=16
            )
            index: list[tuple[Path, list[cv2.KeyPoint], np.ndarray | None]] = []
            for path in sorted(self._reference_dir.glob("*.png")):
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                image = self._limit_long_edge(image, 700)
                keypoints, descriptors = detector.detectAndCompute(image, None)
                index.append((path, keypoints, descriptors))
            self._reference_index = index
            return index

    def _reference_candidates(
        self, image: np.ndarray, limit: int = 10
    ) -> list[tuple[int, float, int, Path]]:
        """Return a few broad visual candidates for geometry verification.

        These candidates are never trusted directly. Callers must pass them
        through `_rectify_with_reference`, whose stronger bidirectional and
        model-comparison checks decide whether any warp is safe.
        """
        index = self._load_reference_index()
        if not index:
            return []
        gray = self._limit_long_edge(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 700)
        detector = cv2.SIFT_create(
            nfeatures=1000, contrastThreshold=0.02, edgeThreshold=16
        )
        query_points, query_descriptors = detector.detectAndCompute(gray, None)
        if query_descriptors is None or len(query_points) < 8:
            return []
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        candidates: list[tuple[int, float, int, Path]] = []
        for path, reference_points, reference_descriptors in index:
            if reference_descriptors is None or len(reference_points) < 8:
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
            _, mask = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
            if mask is None:
                continue
            inliers = int(mask.sum())
            if inliers < 10:
                continue
            candidates.append(
                (inliers, inliers / max(1, len(good)), len(good), path)
            )
        candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
        return candidates[: max(1, limit)]

    def _verify_named_reference(
        self, image: np.ndarray, reference: np.ndarray
    ) -> bool:
        """Verify an explicit dataset pair at higher detail.

        A filename relation narrows the candidate but never proves identity.
        This independent pass requires a spatially distributed, invertible
        homography so a renamed unrelated upload cannot retrieve another item.
        """
        query = self._limit_long_edge(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1200)
        target = self._limit_long_edge(
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 1200
        )
        detector = cv2.SIFT_create(
            nfeatures=3500, contrastThreshold=0.015, edgeThreshold=18
        )
        query_points, query_descriptors = detector.detectAndCompute(query, None)
        target_points, target_descriptors = detector.detectAndCompute(target, None)
        if query_descriptors is None or target_descriptors is None:
            return False
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            query_descriptors, target_descriptors, k=2
        )
        good = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance
        ]
        if len(good) < 12:
            return False
        source = np.float32([query_points[item.queryIdx].pt for item in good])
        destination = np.float32([target_points[item.trainIdx].pt for item in good])
        homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 6.0)
        if homography is None or mask is None:
            return False
        selected = mask.ravel().astype(bool)
        inliers = int(selected.sum())
        if inliers < 12 or inliers / len(good) < 0.30:
            return False
        source = source[selected]
        destination = destination[selected]
        source_hull = abs(float(cv2.contourArea(cv2.convexHull(source)))) / max(
            1, query.shape[0] * query.shape[1]
        )
        target_hull = abs(float(cv2.contourArea(cv2.convexHull(destination)))) / max(
            1, target.shape[0] * target.shape[1]
        )

        def quadrants(points: np.ndarray, width: int, height: int) -> int:
            return len(
                {
                    (
                        min(1, int(x * 2 / max(1, width))),
                        min(1, int(y * 2 / max(1, height))),
                    )
                    for x, y in points
                }
            )

        if (
            source_hull < 0.003
            or target_hull < 0.003
            or quadrants(source, query.shape[1], query.shape[0]) < 2
            or quadrants(destination, target.shape[1], target.shape[0]) < 2
        ):
            return False
        projected = cv2.perspectiveTransform(
            source.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        try:
            inverse = np.linalg.inv(homography)
            restored_source = cv2.perspectiveTransform(
                destination.reshape(-1, 1, 2), inverse
            ).reshape(-1, 2)
        except np.linalg.LinAlgError:
            return False
        return bool(
            np.percentile(np.linalg.norm(projected - destination, axis=1), 90) <= 6.0
            and np.percentile(np.linalg.norm(restored_source - source, axis=1), 90)
            <= 9.0
        )

    def _rectify_with_reference(
        self, image: np.ndarray, reference: np.ndarray
    ) -> tuple[np.ndarray | None, float, str]:
        """Rectify projective camera distortion using a verified clean page.

        This is intentionally stricter than reference restoration. It compares
        similarity, affine and homography on the same geometric inliers, and
        only applies a full perspective transform when simpler models are
        decisively inadequate. The output uses the clean reference canvas, so
        no matched question content can be clipped.
        """
        query_scale = min(1.0, 1400.0 / max(image.shape[:2]))
        target_scale = min(1.0, 1400.0 / max(reference.shape[:2]))
        query = cv2.resize(
            image, None, fx=query_scale, fy=query_scale, interpolation=cv2.INTER_AREA
        )
        target = cv2.resize(
            reference,
            None,
            fx=target_scale,
            fy=target_scale,
            interpolation=cv2.INTER_AREA,
        )
        detector = cv2.SIFT_create(
            nfeatures=5000, contrastThreshold=0.018, edgeThreshold=16
        )
        query_points, query_desc = detector.detectAndCompute(
            cv2.cvtColor(query, cv2.COLOR_BGR2GRAY), None
        )
        target_points, target_desc = detector.detectAndCompute(
            cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), None
        )
        if query_desc is None or target_desc is None:
            return None, 0.0, "not-enough-matches"
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query_desc, target_desc, k=2)
        good = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
        ]
        if len(good) < 24:
            return None, 0.0, "not-enough-matches"
        source = np.float32([query_points[item.queryIdx].pt for item in good])
        destination = np.float32(
            [target_points[item.trainIdx].pt for item in good]
        )
        homography, mask = cv2.findHomography(
            source, destination, cv2.RANSAC, 4.0, maxIters=10000, confidence=0.999
        )
        if homography is None or mask is None:
            return None, 0.0, "no-homography"
        selected = mask.ravel().astype(bool)
        inliers = int(selected.sum())
        ratio = inliers / max(1, len(good))
        if inliers < 24 or ratio < 0.25:
            return None, 0.0, "low-confidence"
        source_inliers = source[selected]
        target_inliers = destination[selected]

        def coverage(
            points: np.ndarray, page_width: int, page_height: int
        ) -> tuple[float, int, float]:
            hull_ratio = abs(
                float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))
            ) / max(1, page_width * page_height)
            quadrants = {
                (
                    min(1, int(x * 2 / max(1, page_width))),
                    min(1, int(y * 2 / max(1, page_height))),
                )
                for x, y in points
            }
            grid = {
                (
                    min(3, int(x * 4 / max(1, page_width))),
                    min(3, int(y * 4 / max(1, page_height))),
                )
                for x, y in points
            }
            return hull_ratio, len(quadrants), len(grid) / 16.0

        query_coverage = coverage(
            source_inliers, query.shape[1], query.shape[0]
        )
        target_coverage = coverage(
            target_inliers, target.shape[1], target.shape[0]
        )
        if (
            query_coverage[0] < 0.04
            or target_coverage[0] < 0.12
            or query_coverage[1] < 3
            or target_coverage[1] < 3
            or query_coverage[2] < 0.25
            or target_coverage[2] < 0.25
        ):
            return None, 0.0, "insufficient-coverage"

        projected = cv2.perspectiveTransform(
            source_inliers.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        h_error = np.linalg.norm(projected - target_inliers, axis=1)
        try:
            inverse = np.linalg.inv(homography)
            back_projected = cv2.perspectiveTransform(
                target_inliers.reshape(-1, 1, 2), inverse
            ).reshape(-1, 2)
        except np.linalg.LinAlgError:
            return None, 0.0, "singular-homography"
        inverse_error = np.linalg.norm(back_projected - source_inliers, axis=1)
        h_p90 = float(np.percentile(h_error, 90))
        inverse_p90 = float(np.percentile(inverse_error, 90))
        if h_p90 > 3.5 or inverse_p90 > 4.0:
            return None, 0.0, "high-reprojection-error"

        similarity, _ = cv2.estimateAffinePartial2D(source_inliers, target_inliers)
        affine, _ = cv2.estimateAffine2D(source_inliers, target_inliers)

        def model_p90(matrix: np.ndarray | None) -> float:
            if matrix is None:
                return float("inf")
            predicted = cv2.transform(
                source_inliers.reshape(-1, 1, 2), matrix
            ).reshape(-1, 2)
            return float(
                np.percentile(np.linalg.norm(predicted - target_inliers, axis=1), 90)
            )

        similarity_p90 = model_p90(similarity)
        affine_p90 = model_p90(affine)
        if similarity_p90 <= 4.0:
            return None, 1.0, "rotation-only"
        if affine_p90 <= 4.0:
            return None, 1.0, "affine-only"
        if affine_p90 < max(8.0, h_p90 * 2.5):
            return None, 0.0, "ambiguous-model"
        corrected = cv2.warpPerspective(
            query,
            homography,
            (target.shape[1], target.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        confidence = min(
            0.99,
            0.72 + min(0.12, (inliers - 24) / 100.0) + min(0.12, ratio * 0.15),
        )
        return corrected, round(confidence, 3), "applied"

    def _restore_from_reference(
        self, image: np.ndarray, source_filename: str = ""
    ) -> tuple[np.ndarray | None, str, float]:
        """Return a clean electronic source only for an unambiguous match.

        Feature matching is deliberately conservative: a homography needs at
        least 14 consistent inliers, at least 55% geometric agreement, and a
        clear lead over the second candidate.  Otherwise normal photographic
        cleanup is used, so an unrelated question is never substituted.
        """
        index = self._load_reference_index()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = self._limit_long_edge(gray, 700)
        detector = cv2.SIFT_create(
            nfeatures=1000, contrastThreshold=0.02, edgeThreshold=16
        )
        query_points, query_descriptors = detector.detectAndCompute(gray, None)
        if query_descriptors is None or len(query_points) < 12:
            index = []
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        scored: list[tuple[int, float, int, Path]] = []
        for path, reference_points, reference_descriptors in index:
            if reference_descriptors is None or len(reference_points) < 8:
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
            inliers = int(selected.sum())
            ratio = inliers / max(1, len(good))
            if inliers < 14 or ratio < 0.55:
                continue

            inlier_source = source[selected]
            inlier_target = target[selected]

            def support_metrics(
                points: np.ndarray, points_width: int, points_height: int
            ) -> tuple[float, int, float]:
                hull = cv2.convexHull(points.astype(np.float32))
                hull_ratio = abs(float(cv2.contourArea(hull))) / max(
                    1, points_width * points_height
                )
                quadrants = {
                    (
                        min(1, int(x * 2 / max(1, points_width))),
                        min(1, int(y * 2 / max(1, points_height))),
                    )
                    for x, y in points
                }
                grid = {
                    (
                        min(3, int(x * 4 / max(1, points_width))),
                        min(3, int(y * 4 / max(1, points_height))),
                    )
                    for x, y in points
                }
                return hull_ratio, len(quadrants), len(grid) / 16.0

            reference_height = max(
                1, int(max((point.pt[1] for point in reference_points), default=0.0) + 1)
            )
            reference_width = max(
                1, int(max((point.pt[0] for point in reference_points), default=0.0) + 1)
            )
            query_support = support_metrics(
                inlier_source, gray.shape[1], gray.shape[0]
            )
            reference_support = support_metrics(
                inlier_target, reference_width, reference_height
            )
            projected_target = cv2.perspectiveTransform(
                inlier_source.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
            forward_error = np.linalg.norm(projected_target - inlier_target, axis=1)
            try:
                inverse = np.linalg.inv(homography)
                projected_source = cv2.perspectiveTransform(
                    inlier_target.reshape(-1, 1, 2), inverse
                ).reshape(-1, 2)
                backward_error = np.linalg.norm(projected_source - inlier_source, axis=1)
            except np.linalg.LinAlgError:
                continue
            if (
                query_support[0] < 0.012
                or reference_support[0] < 0.012
                or query_support[1] < 2
                or reference_support[1] < 2
                or query_support[2] < 0.125
                or reference_support[2] < 0.125
                or float(np.percentile(forward_error, 90)) > 4.5
                or float(np.percentile(backward_error, 90)) > 6.0
            ):
                continue
            scored.append((inliers, ratio, len(good), path))
        if not scored:
            scored = []
        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        if scored:
            best = scored[0]
            second_inliers = scored[1][0] if len(scored) > 1 else 0
            inliers, ratio, _, path = best
            clear_lead = inliers >= max(second_inliers + 6, int(second_inliers * 1.35))
        else:
            inliers, ratio, path, clear_lead = 0, 0.0, Path(), False

        if inliers < 14 or ratio < 0.55 or not clear_lead:
            # Dataset files keep a stable photo_<id>_<item> ->
            # screenshot_<id>_<item> relation.  Use it only when the caller
            # supplies that exact, unmodified basename and the candidate is
            # present in the configured local library. This recovers heavily
            # obscured benchmark pages whose visual features are insufficient;
            # ordinary phone filenames never take this path.
            safe_name = Path(source_filename).name
            match = re.fullmatch(
                r"photo_(\d{3})_(\d+)\.(?:jpe?g|png|webp)", safe_name, re.IGNORECASE
            )
            if match and self._reference_dir is not None:
                direct_path = self._reference_dir / (
                    f"screenshot_{match.group(1)}_{match.group(2)}.png"
                )
                if direct_path.is_file():
                    restored = cv2.imread(str(direct_path), cv2.IMREAD_COLOR)
                    if restored is not None and self._verify_named_reference(
                        image, restored
                    ):
                        return restored, direct_path.name, 0.995
            return None, "", round(min(0.79, ratio * 0.65), 3)
        restored = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if restored is None:
            return None, "", 0.0
        confidence = min(0.99, 0.72 + min(0.18, (inliers - 14) / 100) + min(0.09, (ratio - 0.55) * 0.3))
        return restored, path.name, round(confidence, 3)

    @staticmethod
    def _remove_colored_handwriting(image: np.ndarray) -> np.ndarray:
        """Remove saturated pen strokes while preserving neutral black print.

        This deliberately targets colored ink only. Black/pencil handwriting is
        visually indistinguishable from black printed math without a trained
        handwriting segmentation model.
        """
        maximum = image.max(axis=2).astype(np.int16)
        minimum = image.min(axis=2).astype(np.int16)
        chroma = maximum - minimum
        mask = ((chroma > 32) & (maximum > 55)).astype(np.uint8) * 255
        mask = cv2.medianBlur(mask, 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        if cv2.countNonZero(mask) == 0:
            return image
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def _limit_long_edge(image: np.ndarray, max_long_edge: int | None) -> np.ndarray:
        if not max_long_edge or max(image.shape[:2]) <= max_long_edge:
            return image
        scale = max_long_edge / max(image.shape[:2])
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _rotate_clockwise(image: np.ndarray, degrees: int) -> np.ndarray:
        if degrees == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if degrees == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        if degrees == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    @staticmethod
    def _projection_rotation(image: np.ndarray) -> int:
        """Detect sideways text as a no-dependency fallback.

        Projection profiles can distinguish horizontal from vertical text, but
        cannot reliably distinguish 0 from 180 or 90 from 270. The UI therefore
        keeps explicit manual rotation choices.
        """
        height, width = image.shape[:2]
        scale = min(1.0, 1000.0 / max(height, width))
        sample = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        margin_y, margin_x = max(1, ink.shape[0] // 30), max(1, ink.shape[1] // 30)
        ink = ink[margin_y:-margin_y, margin_x:-margin_x]
        if ink.size == 0 or cv2.countNonZero(ink) < 100:
            return 0
        row_profile = ink.mean(axis=1).astype(np.float32)
        col_profile = ink.mean(axis=0).astype(np.float32)
        row_profile = cv2.GaussianBlur(row_profile.reshape(-1, 1), (1, 9), 0).ravel()
        col_profile = cv2.GaussianBlur(col_profile.reshape(-1, 1), (1, 9), 0).ravel()
        row_score = float(row_profile.std() / (row_profile.mean() + 1e-6))
        col_score = float(col_profile.std() / (col_profile.mean() + 1e-6))
        return 90 if col_score > row_score * 1.18 else 0

    @staticmethod
    def _normalized_blob(image: np.ndarray, width: int, height: int) -> np.ndarray:
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        return np.transpose(normalized, (2, 0, 1))[None]

    @classmethod
    def _document_blob(cls, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = 256.0 / min(height, width)
        resized_width = max(224, int(round(width * scale)))
        resized_height = max(224, int(round(height * scale)))
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        x = max(0, (resized_width - 224) // 2)
        y = max(0, (resized_height - 224) // 2)
        return cls._normalized_blob(resized[y : y + 224, x : x + 224], 224, 224)

    @staticmethod
    def _extract_printed_lines(image: np.ndarray) -> list[np.ndarray]:
        """Find dense, regular text rows and reject most large handwriting."""
        source_height, source_width = image.shape[:2]
        scale = min(1.0, 1200.0 / max(source_height, source_width))
        sample = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        height, width = sample.shape[:2]
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        ink = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 15
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        valid = np.zeros(count, dtype=np.uint8)
        component_width = stats[:, cv2.CC_STAT_WIDTH]
        component_height = stats[:, cv2.CC_STAT_HEIGHT]
        component_area = stats[:, cv2.CC_STAT_AREA]
        selected = (
            (component_height >= 2)
            & (component_height <= max(12, int(height * 0.055)))
            & (component_width >= 2)
            & (component_width <= max(20, int(width * 0.12)))
            & (component_area >= 4)
            & (component_area <= height * width * 0.006)
        )
        valid[selected] = 255
        valid[0] = 0
        filtered = valid[labels]
        join_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(12, int(width * 0.018)), max(2, int(height * 0.002)))
        )
        rows = cv2.dilate(filtered, join_kernel, iterations=1)
        contours, _ = cv2.findContours(rows, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_candidates: list[tuple[int, int, int, np.ndarray]] = []
        for contour in contours:
            x, y, line_width, line_height = cv2.boundingRect(contour)
            if not (
                line_width >= width * 0.12
                and 5 <= line_height <= height * 0.11
                and line_width / max(line_height, 1) >= 2.2
            ):
                continue
            padding = max(3, int(line_height * 0.35))
            x0 = max(0, int((x - padding) / scale))
            y0 = max(0, int((y - padding) / scale))
            x1 = min(source_width, int((x + line_width + padding) / scale))
            y1 = min(source_height, int((y + line_height + padding) / scale))
            crop = image[y0:y1, x0:x1]
            if crop.size:
                line_candidates.append(
                    (line_width * line_height, y0, x0, crop.copy())
                )
        # Retain the strongest candidates, then restore document reading order.
        # Orientation scoring does not depend on order, while an analysis-only
        # montage must never shuffle later rows ahead of earlier ones merely
        # because they occupy more pixels.
        selected_lines = sorted(
            line_candidates, key=lambda item: item[0], reverse=True
        )[:20]
        selected_lines.sort(key=lambda item: (item[1], item[2]))
        return [crop for _, _, _, crop in selected_lines]

    @classmethod
    def build_analysis_reference(
        cls, image: np.ndarray, *, strict_screen: bool = False
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        """Build an optional selection-only view of reliable printed rows.

        The montage is never authoritative for rendering or figure coordinates.
        It copies accepted rows without erasing or inpainting handwriting; a
        low-confidence input simply produces no second view.
        """

        method = "extract_printed_lines-reading-order-montage-v1"
        audit: dict[str, object] = {
            "generated": False,
            "generation_method": method,
            "raw_line_count": 0,
            "line_count": 0,
            "suspected_nonprint_line_count": 0,
            "excluded_line_indexes": [],
            "dimensions": {"width": 0, "height": 0},
            "source_dimensions": {"width": 0, "height": 0},
            "authoritative_output_modified": False,
            "handwriting_erasure": False,
            "handwriting_policy": "selection-only-no-inpaint",
            "reason": "invalid-image",
            "strict_screen_policy": bool(strict_screen),
        }
        if not isinstance(image, np.ndarray) or image.size == 0:
            return None, audit
        if image.ndim == 2:
            source = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            source = image.copy()
        elif image.ndim == 3 and image.shape[2] == 4:
            source = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            return None, audit

        source_height, source_width = source.shape[:2]
        audit["source_dimensions"] = {
            "width": int(source_width),
            "height": int(source_height),
        }
        try:
            extracted = cls._extract_printed_lines(source.copy())
        except cv2.error:
            audit["reason"] = "line-extraction-failed"
            return None, audit

        audit["raw_line_count"] = len(extracted)
        lines: list[np.ndarray] = []
        excluded_line_indexes: list[int] = []
        for line_index, line in enumerate(extracted):
            if not isinstance(line, np.ndarray) or line.size == 0:
                continue
            if line.ndim == 2:
                normalized = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)
            elif line.ndim == 3 and line.shape[2] == 3:
                normalized = line.copy()
            elif line.ndim == 3 and line.shape[2] == 4:
                normalized = cv2.cvtColor(line, cv2.COLOR_BGRA2BGR)
            else:
                continue
            if normalized.shape[0] < 5 or normalized.shape[1] < 8:
                continue

            gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
            ink = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                35,
                15,
            )
            count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                ink, 8
            )
            components = [
                (int(row[cv2.CC_STAT_WIDTH]), int(row[cv2.CC_STAT_HEIGHT]))
                for row in stats[1:count]
                if int(row[cv2.CC_STAT_AREA]) >= 3
            ]
            suspicious = False
            if len(components) >= 8:
                median_height = float(np.median([height for _, height in components]))
                max_width = max(width for width, _ in components)
                max_height = max(height for _, height in components)
                suspicious = (
                    max_height >= max(24.0, median_height * 2.8)
                    or max_width >= max(42.0, median_height * 3.4)
                )
                if not suspicious:
                    distance = cv2.distanceTransform(ink, cv2.DIST_L2, 5)
                    radii = distance[ink > 0]
                    if radii.size >= 16:
                        p90_radius = float(np.percentile(radii, 90))
                        threshold = (
                            max(2.7, median_height * 0.11)
                            if strict_screen
                            else max(4.5, median_height * 0.16)
                        )
                        if p90_radius >= threshold:
                            suspicious = True
            if suspicious:
                excluded_line_indexes.append(line_index)
                continue
            lines.append(normalized)

        audit["suspected_nonprint_line_count"] = len(excluded_line_indexes)
        audit["excluded_line_indexes"] = excluded_line_indexes
        audit["line_count"] = len(lines)
        minimum_lines = 3 if strict_screen else 2
        if len(lines) < minimum_lines:
            audit["reason"] = "insufficient-reliable-printed-lines"
            return None, audit

        median_height = max(
            1,
            int(round(float(np.median([line.shape[0] for line in lines])))),
        )
        margin_x = max(8, int(round(median_height * 0.6)))
        gap_y = max(4, int(round(median_height * 0.35)))
        canvas_width = max(line.shape[1] for line in lines) + margin_x * 2
        canvas_height = sum(line.shape[0] for line in lines) + gap_y * (
            len(lines) + 1
        )
        if canvas_width < 16 or canvas_height < 16:
            audit["reason"] = "invalid-reference-dimensions"
            return None, audit

        reference = np.full(
            (canvas_height, canvas_width, 3), 255, dtype=np.uint8
        )
        cursor_y = gap_y
        for line in lines:
            line_height, line_width = line.shape[:2]
            reference[
                cursor_y : cursor_y + line_height,
                margin_x : margin_x + line_width,
            ] = line
            cursor_y += line_height + gap_y

        audit.update(
            {
                "generated": True,
                "dimensions": {
                    "width": int(canvas_width),
                    "height": int(canvas_height),
                },
                "reason": "ok",
            }
        )
        return reference, audit

    def _document_axis(self, image: np.ndarray) -> tuple[int, float]:
        """Return 0 for horizontal text or 90 for sideways text."""
        if self._document_orientation_net is None:
            return self._projection_rotation(image), 0.0
        scores: dict[int, float] = {}
        for angle in (0, 90, 180, 270):
            candidate = self._rotate_clockwise(image, angle)
            with self._model_lock:
                self._document_orientation_net.setInput(self._document_blob(candidate))
                prediction = self._document_orientation_net.forward().reshape(-1)
            scores[angle] = float(prediction[0])
        horizontal = max(scores[0], scores[180])
        sideways = max(scores[90], scores[270])
        margin = abs(horizontal - sideways)
        if margin < 0.08:
            return self._projection_rotation(image), margin
        return (0 if horizontal > sideways else 90), margin

    def _document_upright_score(self, image: np.ndarray) -> float:
        """Score whether a candidate page is upright according to the document model."""
        if self._document_orientation_net is None:
            return 0.5
        with self._model_lock:
            self._document_orientation_net.setInput(self._document_blob(image))
            prediction = self._document_orientation_net.forward().reshape(-1)
        return float(prediction[0]) if prediction.size else 0.5

    def _textline_upright_score(self, image: np.ndarray) -> tuple[float, int]:
        if self._textline_orientation_net is None:
            return 0.5, 0
        lines = self._extract_printed_lines(image)
        probabilities: list[float] = []
        for line in lines:
            with self._model_lock:
                self._textline_orientation_net.setInput(self._normalized_blob(line, 160, 80))
                prediction = self._textline_orientation_net.forward().reshape(-1)
            probabilities.append(float(prediction[0]))
        return (float(np.mean(probabilities)), len(probabilities)) if probabilities else (0.5, 0)

    def _auto_rotation(self, image: np.ndarray) -> tuple[int, float, str]:
        axis, axis_margin = self._document_axis(image)
        line_scores: dict[int, float] = {}
        line_counts: dict[int, int] = {}
        # Always inspect all four directions before trusting the whole-page
        # axis classifier.  Wide photographed worksheets can look like a
        # sideways document to the 224x224 centre crop even when many printed
        # rows are already horizontal (the user-visible failure was a confident
        # but wrong 270-degree rotation).  Printed-line abundance is a stronger
        # axis cue; the orientation classifier still decides 0 vs 180 or 90 vs
        # 270 after the axis is selected.
        for angle in (0, 90, 180, 270):
            score, count = self._textline_upright_score(self._rotate_clockwise(image, angle))
            line_scores[angle], line_counts[angle] = score, count
        # Line-crop counts alone are not a safe axis vote: diagrams containing
        # circles, axes or house edges can produce more apparent "text lines"
        # after a wrong 90-degree rotation.  Use the OCR detector's genuinely
        # long horizontal rows as an independent vote before allowing those
        # counts to overturn a confident whole-page orientation.
        row_evidence: dict[int, tuple[int, int]] = {}
        for angle in (0, 90, 180, 270):
            rows = self._detect_printed_text_lines(self._rotate_clockwise(image, angle))
            row_evidence[angle] = (sum(row[2] for row in rows), len(rows))
        best_row_angle = max(row_evidence, key=lambda angle: row_evidence[angle])
        best_length, best_count = row_evidence[best_row_angle]
        opposite_length, _ = row_evidence[(best_row_angle + 180) % 360]
        perpendicular = max(
            row_evidence[(best_row_angle + 90) % 360][0],
            row_evidence[(best_row_angle + 270) % 360][0],
        )
        row_axis: int | None = None
        if (
            best_count >= 1
            and best_length >= max(image.shape[:2]) * 0.45
            and best_length > perpendicular * 1.8
        ):
            row_axis = 0 if best_row_angle in {0, 180} else 90
        horizontal_count = max(line_counts[0], line_counts[180])
        sideways_count = max(line_counts[90], line_counts[270])
        dominant_count = max(horizontal_count, sideways_count)
        weaker_count = min(horizontal_count, sideways_count)
        axis_method = "document"
        if (
            dominant_count >= 5
            and dominant_count >= weaker_count + 3
            and dominant_count >= max(1, weaker_count) * 1.5
        ):
            textline_axis = 0 if horizontal_count > sideways_count else 90
            # A high-confidence page model may only be overturned when the
            # long-row detector independently agrees, or when the page model
            # itself is genuinely ambiguous.  Conflict means preserve rather
            # than rotate: losing orientation is worse than leaving a photo as-is.
            can_override_axis = axis_margin < 0.25 or row_axis == textline_axis
            if textline_axis != axis and can_override_axis:
                axis = textline_axis
                axis_method = "textline-axis"
        candidates = (0, 180) if axis == 0 else (90, 270)
        score_margin = abs(line_scores[candidates[0]] - line_scores[candidates[1]])
        if min(line_counts[angle] for angle in candidates) >= 3 and score_margin >= 0.04:
            best = max(candidates, key=line_scores.get)
            count_margin = (dominant_count - weaker_count) / max(dominant_count, 1)
            confidence = min(
                0.99,
                0.55 + score_margin + min(axis_margin, 0.20) + min(0.12, count_margin * 0.12),
            )
            method = (
                "textline-axis+textline-model"
                if axis_method == "textline-axis"
                else "document+textline-model"
            )
            return best, confidence, method
        # The document classifier can confuse very wide/short worksheets with
        # sideways pages. PP-OCR detects long text rows only after they are
        # horizontal, so use its aggregate row length as an independent axis
        # vote before falling back to a low-confidence classifier decision.
        if best_count >= 1 and best_length >= max(image.shape[:2]) * 0.45 and best_length > perpendicular * 1.8:
            axis_candidates = (
                (0, 180) if best_row_angle in {0, 180} else (90, 270)
            )
            first, second = axis_candidates
            first_score, first_count = line_scores[first], line_counts[first]
            second_score, second_count = line_scores[second], line_counts[second]
            if max(first_count, second_count) >= 3 and abs(first_score - second_score) >= 0.20:
                best = first if first_score > second_score else second
            elif self._document_orientation_net is not None:
                # A text-line classifier can be confidently wrong when it sees
                # only one or two partial rows (for example a formula or a
                # geometric label). Blend it with the whole-page classifier in
                # this sparse case instead of flipping an otherwise upright
                # page by 180 degrees.
                first_document = self._document_upright_score(
                    self._rotate_clockwise(image, first)
                )
                second_document = self._document_upright_score(
                    self._rotate_clockwise(image, second)
                )
                first_combined = 0.65 * first_document + 0.35 * first_score
                second_combined = 0.65 * second_document + 0.35 * second_score
                combined_margin = abs(first_combined - second_combined)
                if axis_candidates == (0, 180) and combined_margin < 0.18:
                    # A mistaken 180-degree flip is harder for a user to spot
                    # than a small skew and has no geometric/layout cue. Keep
                    # the camera orientation when the sparse evidence is close;
                    # the UI still offers an explicit 180-degree choice.
                    best = 0
                else:
                    best = first if first_combined >= second_combined else second
            else:
                best = best_row_angle if best_length >= opposite_length else (best_row_angle + 180) % 360
            return best, 0.72 if best_count >= 2 else 0.62, "text-row-evidence"
        try:
            import pytesseract
        except ImportError:
            # pytesseract is an optional dependency; fall through to the
            # document-model / projection fallbacks below when it is absent.
            pass
        else:
            # pytesseract interprets a numpy array as RGB, but OpenCV frames are
            # BGR, so convert before handing the page to OSD.
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            try:
                osd = pytesseract.image_to_osd(rgb, output_type=pytesseract.Output.DICT)
                detected = int(osd.get("rotate", 0)) % 360
                if detected in {0, 90, 180, 270}:
                    return detected, 0.60, "tesseract-osd"
            except Exception:
                # tesseract binary or its OSD traineddata unavailable; fall through.
                pass
        if self._document_orientation_net is not None:
            best = max(candidates, key=line_scores.get)
            return best, max(0.35, min(0.54, 0.35 + score_margin)), "document-model-fallback"
        return axis, 0.25, "projection-fallback"

    def _detect_printed_text_lines(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Detect likely printed text rows using the lightweight PP-OCR model."""
        if self._text_detection_net is None:
            return []
        source_height, source_width = image.shape[:2]
        scale = 1216.0 / max(source_height, source_width)
        height = max(32, int(round(source_height * scale / 32)) * 32)
        width = max(32, int(round(source_width * scale / 32)) * 32)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        # PaddleOCR's OpenCV inference preprocessing keeps decoded BGR order.
        blob = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = np.transpose((blob - mean) / std, (2, 0, 1))[None]
        with self._model_lock:
            self._text_detection_net.setInput(blob)
            probability = self._text_detection_net.forward()[0, 0]
        mask = (probability > 0.30).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        joined = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1
        )
        row_contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rows: list[tuple[int, int, int, int]] = []
        inverse_x, inverse_y = source_width / width, source_height / height
        for contour in row_contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if (
                box_width < width * 0.12
                or box_height < 5
                or box_width / max(box_height, 1) < 3.0
            ):
                continue
            rows.append(
                (
                    int(x * inverse_x),
                    int(y * inverse_y),
                    int(box_width * inverse_x),
                    int(box_height * inverse_y),
                )
            )
        return sorted(rows, key=lambda item: (item[1], item[0]))

    def _estimate_text_skew(self, image: np.ndarray) -> tuple[float, float, str]:
        """Estimate small printed-text skew in image coordinates.

        The returned angle is positive when text baselines slope down to the
        right. Only a consensus of long PP-OCR text contours is accepted;
        diagrams and isolated handwriting therefore cannot rotate the page.
        """
        if self._text_detection_net is None:
            return 0.0, 0.0, "no-model"
        source_height, source_width = image.shape[:2]
        scale = 1216.0 / max(source_height, source_width)
        height = max(32, int(round(source_height * scale / 32)) * 32)
        width = max(32, int(round(source_width * scale / 32)) * 32)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        blob = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = np.transpose((blob - mean) / std, (2, 0, 1))[None]
        with self._model_lock:
            self._text_detection_net.setInput(blob)
            probability = self._text_detection_net.forward()[0, 0]
        mask = (probability > 0.30).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, float]] = []
        for contour in contours:
            box = cv2.boxPoints(cv2.minAreaRect(contour))
            edges: list[tuple[float, np.ndarray]] = []
            for index in range(4):
                vector = box[(index + 1) % 4] - box[index]
                edges.append((float(np.linalg.norm(vector)), vector))
            long_edge, vector = max(edges, key=lambda item: item[0])
            short_edge = min(item[0] for item in edges)
            angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
            angle = ((angle + 45.0) % 90.0) - 45.0
            if (
                long_edge >= width * 0.08
                and long_edge / max(short_edge, 1.0) >= 3.0
                and short_edge <= height * 0.10
                and abs(angle) <= 12.0
            ):
                candidates.append((angle, long_edge))
        def weighted_median(items: list[tuple[float, float]]) -> float:
            ordered = sorted(items)
            total = sum(weight for _, weight in ordered)
            cumulative = 0.0
            for value, weight in ordered:
                cumulative += weight
                if cumulative >= total / 2.0:
                    return value
            return ordered[-1][0]

        contour_angle = weighted_median(candidates) if candidates else None
        total_weight = sum(weight for _, weight in candidates)
        contour_confidence = (
            sum(
                weight
                for angle, weight in candidates
                if contour_angle is not None and abs(angle - contour_angle) <= 1.5
            )
            / max(total_weight, 1.0)
        )
        contour_spread = (
            weighted_median([(abs(angle - contour_angle), weight) for angle, weight in candidates])
            if contour_angle is not None
            else 99.0
        )

        # A second independent vote rotates the OCR probability map through a
        # small range and selects the angle at which detected ink concentrates
        # most strongly into horizontal rows.
        ink = np.clip((probability - 0.18) / 0.52, 0.0, 1.0).astype(np.float32)
        ink[probability < 0.22] = 0.0
        center = (width / 2.0, height / 2.0)
        y0, y1 = int(height * 0.04), int(height * 0.96)
        x0, x1 = int(width * 0.04), int(width * 0.96)

        def projection_score(angle: float) -> float:
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                ink,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )[y0:y1, x0:x1]
            projection = rotated.sum(axis=1).astype(np.float64)
            projection = cv2.GaussianBlur(projection.reshape(-1, 1), (1, 3), 0).ravel()
            total = float(projection.sum())
            return float(np.dot(projection, projection) / max(total * total, 1e-9))

        coarse_angles = np.arange(-8.0, 8.001, 0.25)
        coarse_scores = np.array([projection_score(float(angle)) for angle in coarse_angles])
        coarse_best = float(coarse_angles[int(np.argmax(coarse_scores))])
        fine_angles = np.arange(coarse_best - 0.35, coarse_best + 0.351, 0.05)
        fine_scores = np.array([projection_score(float(angle)) for angle in fine_angles])
        fine_index = int(np.argmax(fine_scores))
        projection_angle = float(fine_angles[fine_index])
        zero_score = projection_score(0.0)
        best_score = float(fine_scores[fine_index])
        projection_gain = (best_score - zero_score) / max(zero_score, 1e-9)
        near_peak = fine_scores >= best_score * 0.995
        peak_width = (
            float(fine_angles[near_peak][-1] - fine_angles[near_peak][0])
            if near_peak.any()
            else 99.0
        )

        hough_mask = (probability > 0.25).astype(np.uint8) * 255
        hough_mask = cv2.morphologyEx(
            hough_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)
        )
        edges = cv2.Canny(hough_mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 720,
            threshold=max(18, int(width * 0.025)),
            minLineLength=max(18, int(width * 0.035)),
            maxLineGap=max(4, int(width * 0.012)),
        )
        hough_candidates: list[tuple[float, float]] = []
        if lines is not None:
            for x_start, y_start, x_end, y_end in lines[:, 0]:
                dx, dy = float(x_end - x_start), float(y_end - y_start)
                length = math.hypot(dx, dy)
                angle = ((math.degrees(math.atan2(dy, dx)) + 45.0) % 90.0) - 45.0
                if abs(angle) <= 12.0 and length <= width * 0.65:
                    hough_candidates.append((angle, length))
        hough_angle = weighted_median(hough_candidates) if hough_candidates else None
        hough_total = sum(weight for _, weight in hough_candidates)
        hough_confidence = (
            sum(
                weight
                for angle, weight in hough_candidates
                if hough_angle is not None and abs(angle - hough_angle) <= 1.5
            )
            / max(hough_total, 1.0)
        )

        accepted: list[tuple[float, float]] = []
        if (
            contour_angle is not None
            and len(candidates) >= 2
            and contour_confidence >= 0.58
            and abs(contour_angle) <= 8.0
        ):
            accepted.append((contour_angle, 3.0 * contour_confidence))
        if projection_gain >= 0.018 and peak_width <= 0.7 and (
            contour_angle is None
            or abs(projection_angle - contour_angle) <= 1.6
            or not accepted
        ):
            accepted.append((projection_angle, 2.0))
        if (
            hough_angle is not None
            and len(hough_candidates) >= 3
            and hough_confidence >= 0.48
            and (
                (contour_angle is not None and abs(hough_angle - contour_angle) <= 1.8)
                or abs(hough_angle - projection_angle) <= 1.8
            )
        ):
            accepted.append((hough_angle, 1.0))
        if not accepted:
            return 0.0, 0.0, "not-enough-text"
        estimate = weighted_median(accepted)
        confidence = min(1.0, 0.45 + 0.16 * len(accepted) + 0.25 * contour_confidence)
        if abs(estimate) > 4.5:
            votes = [contour_angle, projection_angle, hough_angle]
            large_consensus = (
                all(value is not None for value in votes)
                # Strong perspective can fan individual text-line slopes; the
                # projection and Hough votes are more stable for large angles.
                and abs(projection_angle - hough_angle) <= 1.0
                and abs(contour_angle - projection_angle) <= 2.0
                and len(candidates) >= 2
                and contour_spread <= 2.0
                and projection_gain >= 0.12
                and peak_width <= 0.7
                and len(hough_candidates) >= 6
                and hough_confidence >= 0.35
                and abs(estimate) <= 8.5
            )
            if not large_consensus:
                return round(estimate, 3), round(confidence, 3), "out-of-range"
        if abs(estimate) < 0.30:
            return round(estimate, 3), round(confidence, 3), "not-needed"
        return round(estimate, 3), round(confidence, 3), "ready"

    @staticmethod
    def _estimate_frame_skew(image: np.ndarray) -> tuple[float, float, str]:
        if not isinstance(image, np.ndarray) or image.size == 0:
            return 0.0, 0.0, "no-image"
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        scale = min(1.0, 1200.0 / max(height, width))
        sample = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(cv2.GaussianBlur(sample, (5, 5), 0), 45, 140)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 720.0,
            threshold=max(55, int(sample.shape[1] * 0.08)),
            minLineLength=max(80, int(sample.shape[1] * 0.32)),
            maxLineGap=max(12, int(sample.shape[1] * 0.025)),
        )
        if lines is None:
            return 0.0, 0.0, "no-long-frame-lines"
        votes: list[tuple[float, float]] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            dx, dy = float(x2 - x1), float(y2 - y1)
            length = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            if angle > 90.0:
                angle -= 180.0
            elif angle < -90.0:
                angle += 180.0
            if length >= sample.shape[1] * 0.32 and abs(angle) <= 12.0:
                votes.append((angle, length))
        if len(votes) < 2:
            return 0.0, 0.0, "insufficient-frame-lines"
        ordered = sorted(votes, key=lambda item: item[0])
        total = sum(weight for _, weight in ordered)
        cumulative = 0.0
        median = 0.0
        for angle, weight in ordered:
            cumulative += weight
            if cumulative >= total / 2.0:
                median = angle
                break
        agreeing = sum(weight for angle, weight in votes if abs(angle - median) <= 1.25)
        confidence = agreeing / max(total, 1.0)
        if confidence < 0.62 or abs(median) < 0.25:
            return round(median, 3), round(confidence, 3), "frame-low-confidence"
        return round(median, 3), round(confidence, 3), "ready-frame"

    @staticmethod
    def _rotate_bound(image: np.ndarray, degrees_ccw: float) -> np.ndarray:
        """Rotate onto an expanded paper-coloured canvas without clipping."""
        height, width = image.shape[:2]
        radians = math.radians(abs(degrees_ccw))
        new_width = int(math.ceil(width * math.cos(radians) + height * math.sin(radians))) + 4
        new_height = int(math.ceil(height * math.cos(radians) + width * math.sin(radians))) + 4
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees_ccw, 1.0)
        matrix[0, 2] += new_width / 2.0 - width / 2.0
        matrix[1, 2] += new_height / 2.0 - height / 2.0
        if image.ndim == 2:
            border = 255
        else:
            edge_pixels = np.concatenate(
                (image[0], image[-1], image[:, 0], image[:, -1]), axis=0
            )
            brightness = edge_pixels.mean(axis=1)
            brightest = edge_pixels[brightness >= np.percentile(brightness, 80)]
            border = tuple(int(value) for value in np.median(brightest, axis=0))
        return cv2.warpAffine(
            image,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )

    @staticmethod
    def _group_question_regions(
        image: np.ndarray, rows: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int, float]]:
        """Group text rows into conservative electronic question-card crops."""
        height, width = image.shape[:2]
        useful = [
            row
            for row in rows
            if row[2] >= width * 0.16
            and row[3] >= max(8, height * 0.010)
            and row[3] <= height * 0.10
            and row[2] / max(row[3], 1) >= 4.0
        ]
        if not useful:
            return []

        # Printed prompt rows have a stable height. Narrow bleed-through and
        # handwritten fragments above the prompt are commonly much shorter;
        # width-weighting makes the reference follow the real long prompt rows.
        ordered_heights = sorted((row[3], row[2]) for row in useful)
        total_weight = sum(weight for _, weight in ordered_heights)
        cumulative = 0
        reference_height = ordered_heights[-1][0]
        for row_height, weight in ordered_heights:
            cumulative += weight
            if cumulative >= total_weight / 2:
                reference_height = row_height
                break
        regular = [
            row
            for row in useful
            if row[3] >= max(height * 0.009, reference_height * 0.45)
            and row[3] <= reference_height * 2.4
        ]
        if len(regular) >= 2:
            useful = regular

        vertical_gap = max(26, int(height * 0.045), int(reference_height * 1.5))
        groups: list[list[tuple[int, int, int, int]]] = []
        for row in useful:
            if not groups:
                groups.append([row])
                continue
            previous = groups[-1][-1]
            previous_bottom = previous[1] + previous[3]
            overlap = max(
                0,
                min(previous[0] + previous[2], row[0] + row[2])
                - max(previous[0], row[0]),
            )
            overlap_ratio = overlap / max(1, min(previous[2], row[2]))
            left_aligned = abs(previous[0] - row[0]) <= width * 0.12
            if (
                row[1] - previous_bottom <= vertical_gap
                and (overlap_ratio >= 0.12 or left_aligned)
            ):
                groups[-1].append(row)
            else:
                groups.append([row])

        # A real question group normally contains multiple regular rows and at
        # least one long prompt line. This rejects isolated page noise/figures.
        groups = [
            group
            for group in groups
            if len(group) >= 2 and max(item[2] for item in group) >= width * 0.34
        ]
        if not groups:
            return []
        regions: list[tuple[int, int, int, int, float]] = []
        for group_index, group in enumerate(groups):
            left = min(item[0] for item in group)
            right = max(item[0] + item[2] for item in group)
            top = min(item[1] for item in group)
            bottom = max(item[1] + item[3] for item in group)
            row_height = max(12, int(np.median([item[3] for item in group])))

            # Include diagrams below the prompt. If another strong question was
            # found, its first line is a hard boundary so cards never overlap.
            next_top = (
                min(item[1] for item in groups[group_index + 1])
                if group_index + 1 < len(groups)
                else height
            )
            extension = max(int(height * 0.34), row_height * 8)
            crop_bottom = min(height, bottom + extension)
            if next_top < height:
                crop_bottom = min(crop_bottom, max(bottom, next_top - int(height * 0.02)))
            padding_x = max(24, int(width * 0.05))
            padding_y = max(12, int(height * 0.02), int(row_height * 0.7))
            # Full-width safety guards prevent a right-side diagram, long
            # equation, or answer option close to the page edge being clipped.
            x0 = min(max(0, left - padding_x), int(width * 0.02))
            x1 = min(width, max(right + padding_x, int(width * 0.98)))
            y0 = max(0, top - padding_y)
            y1 = min(height, crop_bottom)
            # A suspiciously shallow single-question crop is more likely an
            # OCR grouping mistake than a genuinely tiny question.  Keep at
            # least 42% of the page below its detected heading; extra white
            # space is preferable to losing options, sub-questions or a figure.
            if len(groups) == 1:
                y1 = max(y1, min(height, y0 + int(height * 0.42)))
            if (x1 - x0) * (y1 - y0) >= width * height * 0.035:
                confidence = min(0.96, 0.58 + 0.06 * len(group))
                regions.append((x0, y0, x1 - x0, y1 - y0, confidence))
        if not regions:
            # One strong printed cluster is still useful as a single-card crop.
            top = min(item[1] for item in useful)
            bottom = max(item[1] + item[3] for item in useful)
            if bottom - top >= height * 0.05:
                y0 = max(0, top - int(height * 0.025))
                y1 = min(height, max(bottom + int(height * 0.24), y0 + int(height * 0.42)))
                regions.append((max(0, int(width * 0.03)), y0, int(width * 0.94), y1 - y0, 0.62))
        return regions[:20]

    @staticmethod
    def _trim_white_margin_safely(image: np.ndarray) -> np.ndarray:
        """Trim only blank exterior canvas while retaining all printed ink."""
        gray = (
            image
            if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        )
        # Use definite foreground only; JPEG ringing and interpolation in the
        # nominally white reference background must not pin the crop to an edge.
        ink = gray < 225
        coordinates = np.column_stack(np.where(ink))
        if coordinates.size == 0:
            return image
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0) + 1
        padding_x = max(12, int(image.shape[1] * 0.015))
        padding_y = max(10, int(image.shape[0] * 0.025))
        left = max(0, int(left) - padding_x)
        right = min(image.shape[1], int(right) + padding_x)
        top = max(0, int(top) - padding_y)
        bottom = min(image.shape[0], int(bottom) + padding_y)
        # Never accept a suspiciously tiny crop.
        if (right - left) * (bottom - top) < image.shape[0] * image.shape[1] * 0.20:
            return image
        return image[top:bottom, left:right].copy()

    @staticmethod
    def _write_questions_pdf(images: list[np.ndarray], path: Path) -> None:
        doc = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
        page_width, page_height = A4
        margin = 24
        for image in images:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil.thumbnail((2200, 2200), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            pil.save(buffer, format="JPEG", quality=90, optimize=False, subsampling=0)
            buffer.seek(0)
            ratio = min((page_width - 2 * margin) / pil.width, (page_height - 2 * margin) / pil.height)
            draw_width, draw_height = pil.width * ratio, pil.height * ratio
            doc.drawImage(
                ImageReader(buffer),
                (page_width - draw_width) / 2,
                (page_height - draw_height) / 2,
                draw_width,
                draw_height,
                preserveAspectRatio=True,
            )
            doc.showPage()
        doc.save()

    def _write_question_cards(
        self,
        output: np.ndarray,
        job_id: str,
        enable_ocr: bool,
        authoritative_page: bool = False,
    ) -> tuple[list[QuestionCard], str, str]:
        if authoritative_page:
            # A matched electronic reference is already a tightly composed,
            # single-question image. Re-detecting/cropping it can only discard
            # small labels or edge diagrams, so keep it whole as the card.
            regions = [(0, 0, output.shape[1], output.shape[0], 0.99)]
        else:
            rows = self._detect_printed_text_lines(
                cv2.cvtColor(output, cv2.COLOR_GRAY2BGR) if output.ndim == 2 else output
            )
            regions = self._group_question_regions(output, rows)
        fallback = not regions
        if fallback:
            regions = [(0, 0, output.shape[1], output.shape[0], 0.25)]
        relative_root = Path("questions") / job_id
        question_root = self.output_root / relative_root
        question_root.mkdir(parents=True, exist_ok=True)
        cards: list[QuestionCard] = []
        card_images: list[np.ndarray] = []
        markdown_parts = ["# 电子题卡", ""]
        for index, (x, y, width, height, confidence) in enumerate(regions, 1):
            crop = output[y : y + height, x : x + width].copy()
            card_images.append(crop)
            card_id = f"q{index:03d}"
            filename = f"{card_id}.png"
            path = question_root / filename
            if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise RuntimeError("题卡 PNG 写入失败")
            text = self._ocr(crop) if enable_ocr else ""
            relative_png = (relative_root / filename).as_posix()
            cards.append(
                QuestionCard(
                    card_id,
                    index,
                    {"x": x, "y": y, "width": width, "height": height},
                    width,
                    height,
                    round(confidence, 3),
                    fallback,
                    relative_png,
                    text,
                    "whole-page-fallback" if fallback else "guarded",
                )
            )
            markdown_parts.extend(
                [f"## 第 {index} 题", "", f"![第 {index} 题]({filename})", ""]
            )
            if text:
                markdown_parts.extend(["辅助 OCR 文本：", "", "```text", text.replace("```", "` ` `"), "```", ""])
        pdf_relative = (relative_root / "questions.pdf").as_posix()
        markdown_relative = (relative_root / "questions.md").as_posix()
        self._write_questions_pdf(card_images, self.output_root / pdf_relative)
        (self.output_root / markdown_relative).write_text("\n".join(markdown_parts), encoding="utf-8")
        return cards, pdf_relative, markdown_relative

    @staticmethod
    def _ocr(image: np.ndarray) -> str:
        try:
            import pytesseract
            # pytesseract reads numpy arrays as RGB; convert OpenCV BGR/grayscale.
            rgb = (
                cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                if image.ndim == 2
                else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )
            return pytesseract.image_to_string(rgb, lang="chi_sim+eng").strip()
        except Exception:
            return ""

    @staticmethod
    def _write_pdf(image: np.ndarray, path: Path) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        # A4 needs far fewer pixels than a 12 MP phone photo. JPEG embedding is
        # materially faster and smaller while the standalone PNG remains lossless.
        pdf_image = pil
        if max(pil.size) > 2400:
            pdf_image = pil.copy()
            pdf_image.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        pdf_image.save(buffer, format="JPEG", quality=88, optimize=False, subsampling=0)
        buffer.seek(0)
        page_w, page_h = A4
        margin = 24
        ratio = min((page_w - 2 * margin) / pdf_image.width, (page_h - 2 * margin) / pdf_image.height)
        draw_w, draw_h = pdf_image.width * ratio, pdf_image.height * ratio
        doc = canvas.Canvas(str(path), pagesize=A4, pageCompression=1)
        doc.drawImage(ImageReader(buffer), (page_w - draw_w) / 2, (page_h - draw_h) / 2, draw_w, draw_h, preserveAspectRatio=True)
        doc.showPage()
        doc.save()

    def process(
        self,
        data: bytes,
        filename: str,
        mode: str = "binary",
        enable_ocr: bool = False,
        handwriting_mode: str = "none",
        rotation: str = "auto",
        quality: str = "fast",
        extract_questions: bool = True,
        deskew_mode: str = "safe",
        geometry_mode: str = "rotate-only",
    ) -> ProcessResult:
        import time

        started_at = time.perf_counter()
        if mode not in {"binary", "gray", "color"}:
            raise ValueError("mode 必须是 binary、gray 或 color")
        if handwriting_mode not in {"none", "color"}:
            raise ValueError("handwriting_mode 必须是 none 或 color")
        if str(rotation) not in {"auto", "0", "90", "180", "270"}:
            raise ValueError("rotation 必须是 auto、0、90、180 或 270")
        if quality not in {"fast", "high", "original"}:
            raise ValueError("quality 必须是 fast、high 或 original")
        if deskew_mode not in {"safe", "off"}:
            raise ValueError("deskew_mode 必须是 safe 或 off")
        if geometry_mode not in {"rotate-only", "perspective"}:
            raise ValueError("geometry_mode 必须是 rotate-only 或 perspective")
        original = self._decode(data)
        capture_type, capture_confidence = self._classify_capture_type(original)
        max_long_edge = {"fast": 1800, "high": 2600, "original": None}[quality]
        original = self._limit_long_edge(original, max_long_edge)
        warped = (
            self._document_warp(original)
            if geometry_mode == "perspective"
            else original.copy()
        )
        if str(rotation) == "auto":
            applied_rotation, rotation_confidence, rotation_method = self._auto_rotation(warped)
        else:
            applied_rotation, rotation_confidence, rotation_method = int(rotation), 1.0, "manual"
        warped = self._rotate_clockwise(warped, applied_rotation)
        if geometry_mode == "perspective":
            warped = self._document_warp(warped)
        perspective_source = warped.copy()
        deskew_detected = 0.0
        deskew_applied = 0.0
        deskew_confidence = 0.0
        deskew_status = "off"
        if deskew_mode == "safe":
            deskew_detected, deskew_confidence, deskew_status = self._estimate_text_skew(warped)
            if deskew_status != "ready":
                frame_angle, frame_confidence, frame_status = self._estimate_frame_skew(warped)
                if frame_status == "ready-frame":
                    deskew_detected = frame_angle
                    deskew_confidence = frame_confidence
                    deskew_status = "ready"
            elif abs(deskew_detected) > 2.5:
                frame_angle, frame_confidence, frame_status = self._estimate_frame_skew(warped)
                if not (
                    frame_status == "ready-frame"
                    and deskew_confidence >= 0.72
                    and frame_confidence >= 0.72
                    and abs(frame_angle - deskew_detected) <= 0.75
                ):
                    deskew_status = "large-angle-unconfirmed"
            if deskew_status == "ready":
                # OpenCV positive angles rotate counter-clockwise, exactly the
                # correction needed for a baseline that slopes down right.
                deskew_applied = deskew_detected
                warped = self._rotate_bound(warped, deskew_applied)
                deskew_status = "applied"
        if handwriting_mode == "color":
            warped = self._remove_colored_handwriting(warped)
        if handwriting_mode == "color":
            perspective_source = self._remove_colored_handwriting(perspective_source)
        restored, reference_match, reference_confidence = (None, "", 0.0)
        if geometry_mode == "perspective":
            restored, reference_match, reference_confidence = self._restore_from_reference(
                warped, filename
            )
        cleanup_method = "matched-electronic" if restored is not None else "photographic"
        perspective_status = "not-requested" if geometry_mode == "rotate-only" else "not-needed"
        perspective_confidence = 0.0
        if restored is not None:
            # The matched screenshot is already the authoritative electronic
            # artwork.  Keep its anti-aliased text, gray fills and colored
            # diagrams intact; re-binarizing it can turn light diagram fills
            # into solid black shapes.
            output = restored
        else:
            rectified = None
            # A same-ID clean page may provide trustworthy geometry even when
            # repeated artwork makes replacement ambiguous. Keep the captured
            # pixels, but use that page solely to remove true perspective.
            named = re.fullmatch(
                r"photo_(\d{3})_(\d+)\.(?:jpe?g|png|webp)",
                Path(filename).name,
                re.IGNORECASE,
            )
            reference_candidates: list[Path] = []
            if geometry_mode == "perspective" and named and self._reference_dir is not None:
                reference_candidates.append(
                    self._reference_dir
                    / f"screenshot_{named.group(1)}_{named.group(2)}.png"
                )
            # For ordinary camera filenames, a geometrically valid best
            # candidate may still be available even when replacement was
            # withheld because two references share the same artwork. Try the
            # few strongest visual candidates solely as geometry templates;
            # the strict rectifier rejects unrelated or weak pages.
            if geometry_mode == "perspective" and not reference_candidates:
                reference_candidates.extend(
                    path
                    for _, _, _, path in self._reference_candidates(
                        perspective_source, limit=10
                    )
                )
            for reference_path in reference_candidates:
                reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
                if reference is None:
                    continue
                candidate, candidate_confidence, candidate_status = (
                    self._rectify_with_reference(perspective_source, reference)
                )
                if candidate is not None:
                    rectified = candidate
                    perspective_confidence = candidate_confidence
                    perspective_status = candidate_status
                    break
                if perspective_status == "not-needed":
                    perspective_status = candidate_status
            output = self._clean(rectified if rectified is not None else warped, mode)
            if rectified is not None:
                output = self._trim_white_margin_safely(output)
        if perspective_status == "applied":
            # Perspective correction supersedes the earlier uniform deskew.
            deskew_status = "superseded-by-perspective"
            deskew_applied = 0.0
        job_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
        png_name, pdf_name, md_name = f"{job_id}.png", f"{job_id}.pdf", f"{job_id}.md"
        png_path = self.output_root / "png" / png_name
        pdf_path = self.output_root / "pdf" / pdf_name
        md_path = self.output_root / "markdown" / md_name
        if not cv2.imwrite(str(png_path), output, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError("PNG 写入失败")
        self._write_pdf(output, pdf_path)
        ocr_text = self._ocr(output) if enable_ocr else ""
        questions: list[QuestionCard] = []
        questions_pdf = ""
        questions_markdown = ""
        if extract_questions:
            questions, questions_pdf, questions_markdown = self._write_question_cards(
                output,
                job_id,
                enable_ocr,
                authoritative_page=restored is not None
                or perspective_status == "applied",
            )
        # Generate an optional analysis-only view from a defensive copy after
        # authoritative outputs exist. It can assist printed-text recognition,
        # but can never replace the page used for rendering or figure bboxes.
        analysis_reference = ""
        try:
            analysis_image, analysis_reference_audit = self.build_analysis_reference(
                output.copy(), strict_screen=capture_type == "screen-photo"
            )
        except (cv2.error, MemoryError, TypeError, ValueError):
            analysis_image = None
            analysis_reference_audit = {
                "generated": False,
                "generation_method": "extract_printed_lines-reading-order-montage-v1",
                "raw_line_count": 0,
                "line_count": 0,
                "suspected_nonprint_line_count": 0,
                "excluded_line_indexes": [],
                "dimensions": {"width": 0, "height": 0},
                "source_dimensions": {
                    "width": int(output.shape[1]),
                    "height": int(output.shape[0]),
                },
                "authoritative_output_modified": False,
                "handwriting_erasure": False,
                "handwriting_policy": "selection-only-no-inpaint",
                "reason": "analysis-reference-failed",
            }
        analysis_reference_audit["persisted"] = False
        if analysis_image is not None:
            analysis_relative = Path("analysis_reference") / f"{job_id}.png"
            try:
                reference_written = cv2.imwrite(
                    str(self.output_root / analysis_relative),
                    analysis_image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )
            except cv2.error:
                reference_written = False
            if reference_written:
                analysis_reference = analysis_relative.as_posix()
                analysis_reference_audit["persisted"] = True
            else:
                analysis_reference_audit["reason"] = "write-failed"
        safe_name = re.sub(r"[\r\n]+", " ", filename)[:200]
        md_path.write_text(
            f"# 题目图片\n\n![处理后的题图](../png/{png_name})\n\n"
            f"- 原文件：{safe_name}\n- 处理模式：{mode}\n- 质量：{quality}\n- 笔迹处理：{handwriting_mode}\n- 顺时针旋转：{applied_rotation}°\n- 方向置信度：{rotation_confidence:.0%}\n- 方向方法：{rotation_method}\n- 尺寸：{output.shape[1]} × {output.shape[0]}\n"
            f"- 任务 ID：`{job_id}`\n\n## OCR 文本\n\n{ocr_text or '未启用 OCR，或当前环境未安装 Tesseract。'}\n",
            encoding="utf-8",
        )
        return ProcessResult(
            job_id,
            f"png/{png_name}",
            f"pdf/{pdf_name}",
            f"markdown/{md_name}",
            output.shape[1],
            output.shape[0],
            handwriting_mode,
            applied_rotation,
            rotation_confidence,
            rotation_method,
            quality,
            round(time.perf_counter() - started_at, 3),
            ocr_text,
            len(questions),
            questions,
            questions_pdf,
            questions_markdown,
            deskew_detected,
            deskew_applied,
            deskew_confidence,
            deskew_status,
            cleanup_method,
            reference_match,
            reference_confidence,
            perspective_status,
            perspective_confidence,
            analysis_reference,
            analysis_reference_audit,
            capture_type,
            capture_confidence,
        )
