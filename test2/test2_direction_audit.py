from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent / "test1"
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_PHOTOS = Path(r"D:\robot\Documents\ASK\P_DATA\testmini_photo")
DEFAULT_REFERENCES = Path(r"D:\robot\Documents\ASK\P_DATA\testmini_screenshot")
sys.path.insert(0, str(PROJECT))

from app.processor import QuestionImageProcessor  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only direction, skew, and text-fan audit for test2 outputs."
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--photos", type=Path, default=DEFAULT_PHOTOS)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--out", type=Path, default=ROOT / "direction_audit")
    parser.add_argument("--limit", type=int, default=0, help="Audit only the first N discovered rows.")
    parser.add_argument("--only", type=str, default="", help="Comma-separated source names for focused reruns.")
    parser.add_argument("--no-reference", action="store_true", help="Skip SIFT reference checks.")
    parser.add_argument("--no-previews", action="store_true", help="Do not write risk previews/contact sheet.")
    return parser.parse_args()


def first_match(pattern: str, text: str, default: str = "") -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else default


def discover_rows(results: Path) -> tuple[list[dict], str]:
    """Prefer final results.csv; while the batch runs, reconstruct rows from Markdown."""
    csv_path = results / "results.csv"
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
        return [row for row in rows if str(row.get("ok", "")).lower() in {"true", "1", "yes"}], "results.csv"

    markdown_dir = results / "artifacts" / "markdown"
    rows: list[dict] = []
    for path in sorted(markdown_dir.glob("*.md"), key=lambda item: item.stat().st_mtime):
        text = path.read_text(encoding="utf-8-sig")
        name = first_match(r"原文件：([^\r\n]+)", text)
        png = first_match(r"\]\(\.\./png/([^\)]+)\)", text)
        if not name or not png:
            continue
        rows.append(
            {
                "name": name,
                "job_id": path.stem,
                "output_png": f"png/{png}",
                "rotation": first_match(r"顺时针旋转：(\d+)°", text, "0"),
                "rotation_confidence": str(float(first_match(r"方向置信度：(\d+)%", text, "0")) / 100.0),
                "rotation_method": first_match(r"方向方法：([^\r\n]+)", text),
                "deskew_detected_angle": "",
                "deskew_applied_angle": "",
                "deskew_status": "not-yet-recorded",
            }
        )
    return rows, "markdown-snapshot"


def read_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def probability_map(processor: QuestionImageProcessor, image: np.ndarray) -> tuple[np.ndarray, float, float]:
    source_height, source_width = image.shape[:2]
    scale = 1216.0 / max(source_height, source_width)
    height = max(32, int(round(source_height * scale / 32)) * 32)
    width = max(32, int(round(source_width * scale / 32)) * 32)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    blob = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    blob = np.transpose((blob - mean) / std, (2, 0, 1))[None]
    with processor._model_lock:
        processor._text_detection_net.setInput(blob)
        probability = processor._text_detection_net.forward()[0, 0]
    return probability, source_width / width, source_height / height


def extract_text_lines(probability: np.ndarray, inverse_x: float, inverse_y: float) -> list[dict]:
    height, width = probability.shape
    mask = (probability > 0.30).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), 1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[dict] = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        edges: list[tuple[float, np.ndarray]] = []
        for index in range(4):
            vector = box[(index + 1) % 4] - box[index]
            edges.append((float(np.linalg.norm(vector)), vector))
        long_edge, vector = max(edges, key=lambda item: item[0])
        short_edge = min(item[0] for item in edges)
        detector_angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
        # Text direction is 180-degree periodic, not 90-degree periodic.  Folding
        # by 90 would turn a vertical axis label such as "distance" into a fake
        # +3-degree horizontal row and can manufacture a perspective fan.
        detector_angle = ((detector_angle + 90.0) % 180.0) - 90.0
        if not (
            long_edge >= width * 0.075
            and long_edge / max(short_edge, 1.0) >= 3.0
            and short_edge <= height * 0.11
            and abs(detector_angle) <= 20.0
        ):
            continue
        cx, cy = rect[0]
        radians = math.radians(detector_angle)
        half = long_edge / 2.0
        dx, dy = math.cos(radians) * half, math.sin(radians) * half
        first = np.array([(cx - dx) * inverse_x, (cy - dy) * inverse_y], np.float64)
        second = np.array([(cx + dx) * inverse_x, (cy + dy) * inverse_y], np.float64)
        vector_source = second - first
        source_angle = math.degrees(math.atan2(float(vector_source[1]), float(vector_source[0])))
        lines.append(
            {
                "p1": first,
                "p2": second,
                "centre": (first + second) / 2.0,
                "angle": source_angle,
                "length": float(np.linalg.norm(vector_source)),
            }
        )
    return sorted(lines, key=lambda item: item["centre"][1])


def text_axis_evidence(probability: np.ndarray, inverse_x: float, inverse_y: float) -> dict:
    """Measure whether PP-OCR long contours are horizontal or vertical."""
    height,width=probability.shape
    mask=(probability>0.30).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT,(5,3)),1)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    horizontal=[]; vertical=[]
    for contour in contours:
        box=cv2.boxPoints(cv2.minAreaRect(contour))
        edges=[]
        for index in range(4):
            vector=box[(index+1)%4]-box[index]
            edges.append((float(np.linalg.norm(vector)),vector))
        long_edge,vector=max(edges,key=lambda item:item[0])
        short_edge=min(item[0] for item in edges)
        if long_edge<max(width,height)*0.045 or long_edge/max(short_edge,1.0)<3.0:
            continue
        angle=math.degrees(math.atan2(float(vector[1]),float(vector[0])))
        angle=((angle+90.0)%180.0)-90.0
        source_length=long_edge*math.sqrt(inverse_x*inverse_y)
        if abs(angle)<=20.0:
            horizontal.append(source_length)
        elif abs(abs(angle)-90.0)<=20.0:
            vertical.append(source_length)
    horizontal_length=float(sum(horizontal)); vertical_length=float(sum(vertical))
    vertical_dominant=bool(
        len(vertical)>=2 and vertical_length>=max(horizontal_length*1.8, max(width*inverse_x,height*inverse_y)*0.35)
    )
    return {
        "horizontal_count":len(horizontal),"vertical_count":len(vertical),
        "horizontal_length":round(horizontal_length,2),"vertical_length":round(vertical_length,2),
        "vertical_dominant":vertical_dominant,
    }


def select_coherent_text_stack(lines: list[dict], width: int) -> list[dict]:
    """Remove isolated diagram-shaped PP-OCR contours using repeated x support."""
    if len(lines) < 5:
        return lines
    intervals = [
        (min(float(item["p1"][0]), float(item["p2"][0])), max(float(item["p1"][0]), float(item["p2"][0])))
        for item in lines
    ]
    adjacency = [set() for _ in lines]
    for first in range(len(lines)):
        for second in range(first + 1, len(lines)):
            left_a, right_a = intervals[first]
            left_b, right_b = intervals[second]
            overlap = max(0.0, min(right_a, right_b) - max(left_a, left_b))
            overlap_ratio = overlap / max(min(right_a - left_a, right_b - left_b), 1.0)
            aligned = min(abs(left_a - left_b), abs(right_a - right_b)) <= width * 0.06
            if overlap_ratio >= 0.18 or aligned:
                adjacency[first].add(second)
                adjacency[second].add(first)
    remaining = set(range(len(lines)))
    components: list[list[int]] = []
    while remaining:
        seed = remaining.pop()
        component = [seed]
        frontier = [seed]
        while frontier:
            node = frontier.pop()
            for neighbour in adjacency[node]:
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.append(neighbour)
                    frontier.append(neighbour)
        components.append(component)
    components.sort(
        key=lambda indexes: (len(indexes), sum(lines[index]["length"] for index in indexes)), reverse=True
    )
    return [lines[index] for index in sorted(components[0])] if len(components[0]) >= 5 else lines


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, cumulative[-1] / 2.0)]])


def text_geometry(lines: list[dict], width: int, height: int) -> dict:
    if not lines:
        return {
            "candidate_count": 0,
            "line_count": 0,
            "median_angle": None,
            "angle_range": None,
            "y_span": 0.0,
            "correlation": None,
            "r2": None,
            "slope_deg_per_normalized_y": None,
            "median_abs_residual": None,
        }
    angles = np.array([item["angle"] for item in lines], np.float64)
    ys = np.array([item["centre"][1] / height for item in lines], np.float64)
    weights = np.sqrt(np.array([item["length"] for item in lines], np.float64))
    median_angle = weighted_median(angles, weights)
    angle_range = float(np.percentile(angles, 90) - np.percentile(angles, 10)) if len(lines) >= 2 else 0.0
    y_span = float(ys.max() - ys.min()) if len(lines) >= 2 else 0.0
    if len(lines) >= 3 and float(np.std(ys)) > 1e-6 and float(np.std(angles)) > 1e-6:
        design = np.column_stack([ys, np.ones_like(ys)])
        root_weight = np.sqrt(weights / max(weights.mean(), 1e-9))
        coefficients, *_ = np.linalg.lstsq(design * root_weight[:, None], angles * root_weight, rcond=None)
        predicted = design @ coefficients
        residuals = angles - predicted
        weighted_mean = float(np.sum(weights * angles) / np.sum(weights))
        ss_res = float(np.sum(weights * residuals**2))
        ss_total = float(np.sum(weights * (angles - weighted_mean) ** 2))
        r2 = 1.0 - ss_res / max(ss_total, 1e-9)
        correlation = float(np.corrcoef(ys, angles)[0, 1])
        slope = float(coefficients[0])
        median_residual = float(np.median(np.abs(residuals)))
    else:
        r2 = correlation = slope = median_residual = None
    return {
        "line_count": len(lines),
        "median_angle": round(median_angle, 4),
        "angle_range": round(angle_range, 4),
        "y_span": round(y_span, 4),
        "correlation": None if correlation is None else round(correlation, 4),
        "r2": None if r2 is None else round(r2, 4),
        "slope_deg_per_normalized_y": None if slope is None else round(slope, 4),
        "median_abs_residual": None if median_residual is None else round(median_residual, 4),
    }


def fan_trigger(metrics: dict) -> bool:
    return bool(
        metrics["line_count"] >= 5
        and metrics["y_span"] >= 0.15
        and metrics["angle_range"] is not None
        and metrics["angle_range"] >= 2.4
        and metrics["correlation"] is not None
        and abs(metrics["correlation"]) >= 0.72
        and metrics["r2"] is not None
        and metrics["r2"] >= 0.48
        and metrics["median_abs_residual"] is not None
        and metrics["median_abs_residual"] <= 1.35
    )


def orientation_scores(processor: QuestionImageProcessor, output: np.ndarray) -> dict:
    base_score, base_count = processor._textline_upright_score(output)
    flipped = cv2.rotate(output, cv2.ROTATE_180)
    flip_score, flip_count = processor._textline_upright_score(flipped)
    margin = base_score - flip_score
    # This is an audit flag, not an automatic mutation. Require repeated rows
    # and a material classifier margin before calling an output suspicious.
    suspicious = min(base_count, flip_count) >= 3 and margin <= -0.10
    return {
        "upright_score": round(float(base_score), 4),
        "flipped_score": round(float(flip_score), 4),
        "upright_rows": int(base_count),
        "flipped_rows": int(flip_count),
        "margin": round(float(margin), 4),
        "suspected_upside_down": bool(suspicious),
    }


def scaled_gray(image: np.ndarray, maximum: int = 1500) -> np.ndarray:
    scale = min(1.0, maximum / max(image.shape[:2]))
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)


def reference_orientation(output: np.ndarray, reference: np.ndarray) -> dict:
    """Audit-only truth: test all four output orientations against reference.

    Direct SIFT can have too few inliers when an output is wrong by 90/180.
    Rotating the output through all four candidates before matching gives an
    explicit correction vote.  The clean reference is never used by detector
    flags or image processing.
    """
    reference_gray = scaled_gray(reference)
    sift = cv2.SIFT_create(nfeatures=3500, contrastThreshold=0.018, edgeThreshold=14)
    ref_points, ref_desc = sift.detectAndCompute(reference_gray, None)
    if ref_desc is None:
        return {"reliable": False, "matches": 0, "inliers": 0}
    trials=[]
    rotations={
        0:output,
        90:cv2.rotate(output,cv2.ROTATE_90_CLOCKWISE),
        180:cv2.rotate(output,cv2.ROTATE_180),
        270:cv2.rotate(output,cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    for correction,candidate in rotations.items():
        source_gray=scaled_gray(candidate)
        source_points,source_desc=sift.detectAndCompute(source_gray,None)
        if source_desc is None:
            trials.append({"correction":correction,"matches":0,"inliers":0,"ratio":0.0,"fine_angle":None})
            continue
        pairs=matcher.knnMatch(source_desc,ref_desc,k=2)
        good=[first for first,second in pairs if first.distance<0.72*second.distance]
        if len(good)<8:
            trials.append({"correction":correction,"matches":len(good),"inliers":0,"ratio":0.0,"fine_angle":None})
            continue
        source=np.float32([source_points[item.queryIdx].pt for item in good]).reshape(-1,1,2)
        target=np.float32([ref_points[item.trainIdx].pt for item in good]).reshape(-1,1,2)
        matrix,mask=cv2.estimateAffinePartial2D(
            source,target,method=cv2.RANSAC,ransacReprojThreshold=5.0,maxIters=5000,confidence=0.995
        )
        inliers=int(mask.sum()) if mask is not None else 0
        ratio=inliers/max(len(good),1)
        fine=None
        if matrix is not None:
            fine=math.degrees(math.atan2(float(matrix[1,0]),float(matrix[0,0])))
            fine=((fine+180.0)%360.0)-180.0
        trials.append({"correction":correction,"matches":len(good),"inliers":inliers,"ratio":ratio,"fine_angle":fine})
    best=max(trials,key=lambda trial:(trial["inliers"],trial["ratio"],trial["matches"]))
    ordered=sorted(trials,key=lambda trial:(trial["inliers"],trial["ratio"],trial["matches"]),reverse=True)
    runner_up=ordered[1]
    reliable=(
        best["inliers"]>=12 and best["ratio"]>=0.18
        and (best["inliers"]>=runner_up["inliers"]+4 or best["inliers"]>=runner_up["inliers"]*1.25)
    )
    wrong_quarter_turn=bool(reliable and best["correction"]!=0)
    return {
        "reliable": bool(reliable),
        "matches": best["matches"],
        "inliers": best["inliers"],
        "inlier_ratio": round(best["ratio"], 3),
        "alignment_angle": None if best["fine_angle"] is None else round(best["fine_angle"],3),
        "required_clockwise_correction":best["correction"],
        "runner_up_inliers":runner_up["inliers"],
        "trials":[
            {"correction":trial["correction"],"matches":trial["matches"],"inliers":trial["inliers"],
             "inlier_ratio":round(trial["ratio"],3),
             "fine_angle":None if trial["fine_angle"] is None else round(trial["fine_angle"],3)}
            for trial in trials
        ],
        "wrong_quarter_turn": bool(wrong_quarter_turn),
    }


def draw_preview(image: np.ndarray, lines: list[dict], record: dict, path: Path) -> None:
    preview = image.copy()
    for item in lines:
        first = tuple(np.round(item["p1"]).astype(int))
        second = tuple(np.round(item["p2"]).astype(int))
        cv2.line(preview, first, second, (0, 0, 255), max(2, int(max(image.shape[:2]) / 900)))
        centre = tuple(np.round(item["centre"]).astype(int))
        cv2.putText(
            preview, f"{item['angle']:+.1f}", centre, cv2.FONT_HERSHEY_SIMPLEX,
            max(0.45, max(image.shape[:2]) / 2600.0), (255, 0, 0), 2, cv2.LINE_AA,
        )
    labels = ",".join(record["flags"]) or "ok"
    cv2.rectangle(preview, (0, 0), (preview.shape[1], min(74, preview.shape[0])), (255, 255, 255), -1)
    cv2.putText(
        preview, f"{record['name']} | {labels}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (0, 0, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"rot={record['rotation']} median={record['geometry']['median_angle']} fan={record['geometry']['angle_range']}",
        (12, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA,
    )
    scale = min(1.0, 1500.0 / max(preview.shape[:2]))
    if scale < 1.0:
        preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])


def contact_sheet(paths: list[Path], target: Path) -> None:
    tiles: list[np.ndarray] = []
    for path in paths[:24]:
        image = read_image(path)
        if image is None:
            continue
        scale = min(1.0, 420.0 / max(image.shape[:2]))
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        canvas = np.full((300, 440, 3), 245, np.uint8)
        y = max(0, (280 - resized.shape[0]) // 2)
        x = max(0, (440 - resized.shape[1]) // 2)
        clipped = resized[: min(resized.shape[0], 280), : min(resized.shape[1], 440)]
        canvas[y : y + clipped.shape[0], x : x + clipped.shape[1]] = clipped
        cv2.putText(canvas, path.stem[:42], (6, 294), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(canvas)
    if not tiles:
        return
    columns = 3
    rows = int(math.ceil(len(tiles) / columns))
    blank = np.full_like(tiles[0], 245)
    while len(tiles) < rows * columns:
        tiles.append(blank.copy())
    sheet = np.vstack([np.hstack(tiles[row * columns : (row + 1) * columns]) for row in range(rows)])
    cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def parse_float(row: dict, key: str) -> float | None:
    value = str(row.get(key, "")).strip()
    try:
        return float(value) if value else None
    except ValueError:
        return None


def write_csv(records: list[dict], path: Path) -> None:
    fields = [
        "name", "job_id", "rotation", "rotation_confidence", "rotation_method",
        "deskew_applied_angle", "line_count", "median_angle", "angle_range", "y_span",
        "correlation", "r2", "fan_trigger", "orientation_margin", "orientation_suspicious",
        "reference_reliable", "reference_angle", "reference_wrong_quarter_turn", "risk_score", "flags",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow(
                {
                    "name": item["name"], "job_id": item["job_id"], "rotation": item["rotation"],
                    "rotation_confidence": item["rotation_confidence"], "rotation_method": item["rotation_method"],
                    "deskew_applied_angle": item["deskew_applied_angle"],
                    "line_count": item["geometry"]["line_count"], "median_angle": item["geometry"]["median_angle"],
                    "angle_range": item["geometry"]["angle_range"], "y_span": item["geometry"]["y_span"],
                    "correlation": item["geometry"]["correlation"], "r2": item["geometry"]["r2"],
                    "fan_trigger": item["fan_trigger"], "orientation_margin": item["orientation"]["margin"],
                    "orientation_suspicious": item["orientation"]["suspected_upside_down"],
                    "reference_reliable": item["reference"].get("reliable", False),
                    "reference_angle": item["reference"].get("required_clockwise_correction", item["reference"].get("alignment_angle")),
                    "reference_wrong_quarter_turn": item["reference"].get("wrong_quarter_turn", False),
                    "risk_score": item["risk_score"], "flags": ";".join(item["flags"]),
                }
            )


def write_report(summary: dict, records: list[dict], path: Path) -> None:
    high_risk = sorted((item for item in records if item["flags"]), key=lambda item: (-item["risk_score"], item["name"]))
    lines = [
        "# test2 方向、倾斜与文字扇出审计",
        "",
        f"- 输入来源：`{summary['discovery_mode']}`",
        f"- 本次发现/审计：{summary['discovered']} / {summary['audited']} 张",
        f"- Manifest 总量：{summary['manifest_total']} 张；当前覆盖率：{summary['coverage_percent']}%",
        f"- 文字行扇出强触发：{summary['fan_trigger_count']} 张",
        f"- 输出文字中位倾角超过 1°：{summary['median_skew_over_1deg_count']} 张",
        f"- 输出文字中位倾角超过 2°：{summary['median_skew_over_2deg_count']} 张",
        f"- 无参考图方向模型疑似倒置：{summary['model_upside_down_count']} 张",
        f"- 参考图可靠匹配：{summary['reference_reliable_count']} 张；确认 90°/180° 错向：{summary['reference_wrong_quarter_turn_count']} 张",
        f"- 高风险列表：{summary['flagged_count']} 张",
        "",
        "## 判定说明",
        "",
        "文字扇出仅用处理后 PNG 的 PP-OCR 概率图：选择纵向重复、x 区间相干的文字行，要求至少 5 行、y 覆盖 >= 0.15、P10-P90 角范围 >= 2.4°、|corr(angle,y)| >= 0.72、R² >= 0.48、拟合中位绝对残差 <= 1.35°。这是强触发，宁可漏报也减少图形轮廓误报。",
        "",
        "方向分两层：第一层比较处理后图与其 180° 翻转的 PP-LCNet 文字行朝向得分；第二层仅用于审计真值，将处理后图与同名电子参考图做 SIFT+RANSAC 仿射匹配。只有第二层可靠且相对角 >=45° 才记为已确认错向。",
        "",
        "## 风险样本",
        "",
        "| 样本 | flags | 风险分 | 旋转 | 文字中位角 | 扇出范围 | 参考角 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in high_risk[:80]:
        geometry = item["geometry"]
        lines.append(
            f"| {item['name']} | {', '.join(item['flags'])} | {item['risk_score']} | {item['rotation']}° | "
            f"{geometry['median_angle']} | {geometry['angle_range']} | {item['reference'].get('required_clockwise_correction', item['reference'].get('alignment_angle'))} |"
        )
    if not high_risk:
        lines.append("| — | 当前快照未发现达到门槛的异常 | 0 | — | — | — | — |")
    lines.extend(
        [
            "",
            "## 重跑",
            "",
            "批量结束、`results.csv` 生成后，在 test2 目录运行：",
            "",
            "```powershell",
            ".\\..\\test1\\.venv\\Scripts\\python.exe .\\test2_direction_audit.py",
            "```",
            "",
            "脚本只读取原图、参考图、处理结果和正式模型；只写 `test2/direction_audit`，不修改 app 或批量输出。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    args.out.mkdir(parents=True, exist_ok=True)
    preview_dir = args.out / "previews"
    if not args.no_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)
    rows, discovery_mode = discover_rows(args.results)
    if args.only:
        selected_names={name.strip() for name in args.only.split(",") if name.strip()}
        rows=[row for row in rows if row.get("name") in selected_names]
    if args.limit:
        rows = rows[: args.limit]
    processor = QuestionImageProcessor(args.out / "unused")
    if processor._text_detection_net is None:
        processor._text_detection_net = cv2.dnn.readNetFromONNX(str(PROJECT / "models" / "PP-OCRv5_mobile_det.onnx"))
    records: list[dict] = []
    preview_paths: list[Path] = []
    for index, row in enumerate(rows, 1):
        name = row["name"]
        output_path = args.results / "artifacts" / row["output_png"]
        output = read_image(output_path)
        if output is None:
            print(f"[{index}/{len(rows)}] {name}: missing {output_path}", flush=True)
            continue
        probability, inverse_x, inverse_y = probability_map(processor, output)
        candidates = extract_text_lines(probability, inverse_x, inverse_y)
        selected = select_coherent_text_stack(candidates, output.shape[1])
        geometry = text_geometry(selected, output.shape[1], output.shape[0])
        geometry["candidate_count"] = len(candidates)
        axis_evidence=text_axis_evidence(probability,inverse_x,inverse_y)
        triggered = fan_trigger(geometry)
        orientation = orientation_scores(processor, output)
        reference_path = args.references / name.replace("photo_", "screenshot_").replace(".jpg", ".png")
        reference = read_image(reference_path) if not args.no_reference else None
        reference_result = reference_orientation(output, reference) if reference is not None else {"reliable": False}
        applied_angle = parse_float(row, "deskew_applied_angle")
        flags: list[str] = []
        risk = 0
        if triggered:
            flags.append("text-fan")
            risk += 4
        if geometry["median_angle"] is not None and abs(geometry["median_angle"]) > 2.0 and geometry["line_count"] >= 3:
            flags.append("residual-skew>2deg")
            risk += 3
        elif geometry["median_angle"] is not None and abs(geometry["median_angle"]) > 1.0 and geometry["line_count"] >= 3:
            flags.append("residual-skew>1deg")
            risk += 1
        if orientation["suspected_upside_down"]:
            flags.append("model-upside-down")
            risk += 3
        if axis_evidence["vertical_dominant"]:
            flags.append("vertical-text-axis")
            risk += 6
        if reference_result.get("wrong_quarter_turn", False):
            flags.append("reference-wrong-quarter-turn")
            risk += 8
        if applied_angle is not None and abs(applied_angle) > 4.5:
            flags.append("large-deskew-applied")
            risk += 1
        record = {
            "name": name,
            "job_id": row.get("job_id", Path(output_path).stem),
            "output_png": str(output_path),
            "rotation": int(float(row.get("rotation", 0) or 0)),
            "rotation_confidence": parse_float(row, "rotation_confidence"),
            "rotation_method": row.get("rotation_method", ""),
            "deskew_detected_angle": parse_float(row, "deskew_detected_angle"),
            "deskew_applied_angle": applied_angle,
            "deskew_status": row.get("deskew_status", ""),
            "geometry": geometry,
            "axis_evidence":axis_evidence,
            "fan_trigger": triggered,
            "orientation": orientation,
            "reference": reference_result,
            "risk_score": risk,
            "flags": flags,
        }
        records.append(record)
        if flags and not args.no_previews:
            preview_path = preview_dir / f"{Path(name).stem}_risk.jpg"
            draw_preview(output, selected, record, preview_path)
            preview_paths.append(preview_path)
        print(f"[{index}/{len(rows)}] {name}: {','.join(flags) or 'ok'}", flush=True)

    manifest = ROOT / "input_manifest.txt"
    manifest_total = len([line for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()])
    reliable_references = [item for item in records if item["reference"].get("reliable", False)]
    summary = {
        "discovery_mode": discovery_mode,
        "discovered": len(rows),
        "audited": len(records),
        "manifest_total": manifest_total,
        "coverage_percent": round(100.0 * len(records) / max(manifest_total, 1), 2),
        "rotation_counts": dict(Counter(str(item["rotation"]) for item in records)),
        "rotation_methods": dict(Counter(item["rotation_method"] for item in records)),
        "fan_trigger_count": sum(item["fan_trigger"] for item in records),
        "fan_trigger_names": [item["name"] for item in records if item["fan_trigger"]],
        "median_skew_over_1deg_count": sum(
            item["geometry"]["line_count"] >= 3
            and item["geometry"]["median_angle"] is not None
            and abs(item["geometry"]["median_angle"]) > 1.0
            for item in records
        ),
        "median_skew_over_2deg_count": sum(
            item["geometry"]["line_count"] >= 3
            and item["geometry"]["median_angle"] is not None
            and abs(item["geometry"]["median_angle"]) > 2.0
            for item in records
        ),
        "model_upside_down_count": sum(item["orientation"]["suspected_upside_down"] for item in records),
        "model_upside_down_names": [item["name"] for item in records if item["orientation"]["suspected_upside_down"]],
        "vertical_text_axis_count":sum(item["axis_evidence"]["vertical_dominant"] for item in records),
        "vertical_text_axis_names":[item["name"] for item in records if item["axis_evidence"]["vertical_dominant"]],
        "reference_reliable_count": len(reliable_references),
        "reference_wrong_quarter_turn_count": sum(
            item["reference"].get("wrong_quarter_turn", False) for item in reliable_references
        ),
        "reference_wrong_quarter_turn_names": [
            item["name"] for item in reliable_references if item["reference"].get("wrong_quarter_turn", False)
        ],
        "flagged_count": sum(bool(item["flags"]) for item in records),
        "flagged_names": [item["name"] for item in records if item["flags"]],
    }
    (args.out / "audit_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(records, args.out / "audit.csv")
    write_report(summary, records, args.out / "PARTIAL_REPORT_CN.md")
    if preview_paths and not args.no_previews:
        contact_sheet(preview_paths, args.out / "risk_contact_sheet.jpg")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
