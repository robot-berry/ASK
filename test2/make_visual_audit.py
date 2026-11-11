from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_DATASET = Path(r"D:\robot\Documents\ASK\P_DATA\testmini_photo")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a stratified source-vs-output audit sheet.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=ROOT / "input_manifest.txt")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--per-stratum", type=int, default=6)
    parser.add_argument("--max-rows", type=int, default=36)
    return parser.parse_args()


def parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def parse_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except (TypeError, ValueError):
        return default


def parse_markdown(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    png_match = re.search(r"\.\./png/([^\s)]+\.png)", text)
    source_match = re.search(r"(photo_\d+_\d+\.(?:jpg|jpeg|png))", text, re.IGNORECASE)
    if not png_match or not source_match:
        return {}
    return {
        "job_id": path.stem,
        "name": source_match.group(1),
        "output_png": f"png/{png_match.group(1)}",
        "ok": True,
        "metadata_source": "markdown-partial",
    }


def load_rows(results: Path, manifest: Path) -> tuple[list[dict], str]:
    csv_path = results / "results.csv"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
        return rows, "results.csv"

    # During an active batch results.csv is atomically written only at the end.
    # Markdown artifacts already contain the source/output association, so use
    # those for a valid progress audit and transparently mark unavailable strata.
    manifest_names = [
        line.strip().lstrip("\ufeff")
        for line in manifest.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    manifest_order = {name: index + 1 for index, name in enumerate(manifest_names)}
    rows: list[dict] = []
    markdown_dir = results / "artifacts" / "markdown"
    for path in sorted(markdown_dir.glob("*.md"), key=lambda item: item.stat().st_mtime_ns):
        row = parse_markdown(path)
        if row:
            row["dataset_index"] = manifest_order.get(row["name"], 0)
            rows.append(row)
    return rows, "markdown-progress"


def resolve_output(results: Path, row: dict) -> Path | None:
    relative = str(row.get("output_png", "")).replace("/", "\\")
    candidates = []
    if relative:
        candidates.append(results / "artifacts" / relative)
        candidates.append(results / relative)
    job_id = str(row.get("job_id", ""))
    if job_id:
        candidates.append(results / "artifacts" / "png" / f"{job_id}.png")
    for path in candidates:
        if path.is_file():
            return path
    return None


def stratify(rows: list[dict], metadata_complete: bool, per_stratum: int) -> tuple[list[dict], dict]:
    successes = [row for row in rows if str(row.get("ok", "")).lower() in {"true", "1"}]
    chosen: list[dict] = []
    selected_names: set[str] = set()
    counts: dict[str, dict] = {}

    def add(label: str, candidates: list[dict], key=None, reverse: bool = False) -> None:
        candidates = list(candidates)
        if key is not None:
            candidates.sort(key=key, reverse=reverse)
        else:
            candidates.sort(key=lambda row: parse_int(row.get("dataset_index")))
        pool_size = len(candidates)
        # Spread normal/progress sampling across the processed range instead of
        # taking only the first adjacent files.
        if pool_size > per_stratum and key is None:
            indexes = np.linspace(0, pool_size - 1, per_stratum, dtype=int)
            candidates = [candidates[index] for index in indexes]
        accepted = 0
        for row in candidates:
            name = str(row.get("name", ""))
            if not name or name in selected_names:
                continue
            tagged = dict(row)
            tagged["audit_stratum"] = label
            chosen.append(tagged)
            selected_names.add(name)
            accepted += 1
            if accepted >= per_stratum:
                break
        counts[label] = {"pool": pool_size, "selected": accepted}

    if metadata_complete:
        add(
            "fallback",
            [row for row in successes if parse_int(row.get("fallback_count")) > 0],
            key=lambda row: parse_int(row.get("fallback_count")),
            reverse=True,
        )
        # A batch can contain no accepted perspective transforms.  Rejected
        # attempts are the most useful cases to inspect in that situation.
        add(
            "perspective-review",
            [
                row
                for row in successes
                if str(row.get("perspective_status", "")) not in {"", "not-needed"}
            ],
            key=lambda row: parse_float(row.get("perspective_confidence")),
            reverse=True,
        )
        add(
            "large-deskew",
            [row for row in successes if abs(parse_float(row.get("deskew_detected_angle"))) >= 3.0],
            key=lambda row: abs(parse_float(row.get("deskew_detected_angle"))),
            reverse=True,
        )
        add(
            "slowest",
            successes,
            key=lambda row: parse_float(row.get("elapsed_seconds")),
            reverse=True,
        )
        normal = [
            row
            for row in successes
            if str(row.get("perspective_status", "")) == "not-needed"
            and abs(parse_float(row.get("deskew_detected_angle"))) < 3.0
            and parse_int(row.get("fallback_count")) == 0
        ]
        add("normal", normal)
        add("failures", [row for row in rows if str(row.get("ok", "")).lower() not in {"true", "1"}])
    else:
        add("progress-sample", successes)
        for label in ("fallback", "perspective-review", "large-deskew", "slowest", "normal", "failures"):
            counts[label] = {"pool": None, "selected": 0, "available_after_batch": True}
    return chosen, counts


def letterbox(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def make_row(source: np.ndarray, output: np.ndarray, row: dict, width: int = 700, image_height: int = 400) -> np.ndarray:
    header_height = 70
    canvas = np.full((header_height + image_height, width * 2, 3), 255, dtype=np.uint8)
    canvas[header_height:, :width] = letterbox(source, width, image_height)
    canvas[header_height:, width:] = letterbox(output, width, image_height)
    name = str(row.get("name", ""))
    stratum = str(row.get("audit_stratum", ""))
    title = f"{stratum} | {name} | source"
    detail = (
        f"output | rot={row.get('rotation', '?')} "
        f"deskew={row.get('deskew_detected_angle', '?')}({row.get('deskew_status', '?')}) "
        f"persp={row.get('perspective_status', '?')} "
        f"fallback={row.get('fallback_count', '?')} time={row.get('elapsed_seconds', '?')}s"
    )
    cv2.putText(canvas, title[:110], (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, detail[:155], (width + 10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 90, 0), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, header_height - 1), (width * 2, header_height - 1), (160, 160, 160), 1)
    cv2.line(canvas, (width, 0), (width, header_height + image_height), (160, 160, 160), 1)
    return canvas


def main() -> None:
    args = arguments()
    out = args.out or args.results / "visual_audit"
    out.mkdir(parents=True, exist_ok=True)
    rows, metadata_source = load_rows(args.results, args.manifest)
    metadata_complete = metadata_source == "results.csv"
    selected, strata = stratify(rows, metadata_complete, max(1, args.per_stratum))
    selected = selected[: max(1, args.max_rows)]
    rendered: list[np.ndarray] = []
    rendered_by_stratum: dict[str, list[np.ndarray]] = defaultdict(list)
    audit_rows: list[dict] = []
    for row in selected:
        source_path = args.dataset / str(row.get("name", ""))
        output_path = resolve_output(args.results, row)
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        output = cv2.imread(str(output_path), cv2.IMREAD_COLOR) if output_path else None
        if source is None or output is None:
            audit_rows.append({**row, "rendered": False, "source_path": str(source_path), "output_path": str(output_path or "")})
            continue
        rendered_row = make_row(source, output, row)
        rendered.append(rendered_row)
        rendered_by_stratum[str(row.get("audit_stratum", "other"))].append(rendered_row)
        audit_rows.append({**row, "rendered": True, "source_path": str(source_path), "output_path": str(output_path)})
    if rendered:
        cv2.imwrite(
            str(out / "source_vs_output.jpg"),
            np.vstack(rendered),
            [cv2.IMWRITE_JPEG_QUALITY, 91],
        )
        for label, stratum_rows in rendered_by_stratum.items():
            cv2.imwrite(
                str(out / f"source_vs_output__{label}.jpg"),
                np.vstack(stratum_rows),
                [cv2.IMWRITE_JPEG_QUALITY, 93],
            )
    fields = sorted({key for row in audit_rows for key in row})
    with (out / "audit_selection.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    manifest_count = sum(1 for line in args.manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    summary = {
        "metadata_source": metadata_source,
        "metadata_complete": metadata_complete,
        "manifest_count": manifest_count,
        "discovered_rows": len(rows),
        "rendered_rows": len(rendered),
        "strata": strata,
        "rerun_command": (
            f'& "{Path(__import__("sys").executable)}" "{Path(__file__).resolve()}" '
            f'--results "{args.results.resolve()}" --manifest "{args.manifest.resolve()}"'
        ),
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
