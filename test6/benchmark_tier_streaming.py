"""Serial live portal checks: record when each PNG first becomes downloadable.

Usage:
  --case math:path.png --case physics:path.png --output result_dir
  --manifest manifest.json --keep-going --output result_dir

Requires --confirm-live-calls; submissions invoke the configured paid provider.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import statistics
import time
import traceback
from urllib.parse import urljoin

import requests
from PIL import Image


EXPECTED_IMAGES = (
    "01_question_knowledge.png",
    "02_guidance_elite.png",
    "03_guidance_upper.png",
    "04_guidance_middle.png",
    "05_guidance_lower.png",
    "06_guidance_novice.png",
)


def percentile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(len(ordered) * proportion + 0.999) - 1))
    return ordered[index]


def load_cases(args: argparse.Namespace) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    if args.manifest:
        payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("manifest 必须包含非空 cases 数组")
        for item in raw_cases:
            subject = str(item.get("subject") or "").strip()
            if subject == "mathematics":
                subject = "math"
            image = Path(str(item.get("image") or item.get("image_path") or ""))
            case_id = str(item.get("id") or image.stem)
            if subject not in {"math", "physics"}:
                raise ValueError(f"unknown subject: {subject}")
            cases.append({"id": case_id, "subject": subject, "image": str(image)})
    for spec in args.case or []:
        subject, raw_path = spec.split(":", 1)
        if subject not in {"math", "physics"}:
            raise ValueError(subject)
        source = Path(raw_path)
        cases.append({"id": source.stem, "subject": subject, "image": str(source)})
    if not cases:
        raise ValueError("需要 --manifest 或至少一条 --case")
    return cases


def write_summary_files(output: Path, summaries: list[dict]) -> None:
    (output / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = [
        "index",
        "id",
        "subject",
        "status",
        "job_id",
        "model_id",
        "first_image_seconds",
        "elite_seconds",
        "complete_seconds",
        "image_contract_ok",
        "guidance_observed_before_complete",
        "error",
        "source",
        "directory",
    ]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


def write_report(output: Path, summaries: list[dict], started_at: str, elapsed: float) -> None:
    completed = [item for item in summaries if item.get("status") == "completed"]
    failed = [item for item in summaries if item.get("status") != "completed"]
    contract_ok = [item for item in completed if item.get("image_contract_ok") is True]
    first_images = [
        float(item["first_image_seconds"])
        for item in completed
        if isinstance(item.get("first_image_seconds"), (int, float))
    ]
    elite_times = [
        float(item["elite_seconds"])
        for item in completed
        if isinstance(item.get("elite_seconds"), (int, float))
    ]
    complete_times = [
        float(item["complete_seconds"])
        for item in completed
        if isinstance(item.get("complete_seconds"), (int, float))
    ]
    math_ok = sum(1 for item in completed if item.get("subject") == "math")
    physics_ok = sum(1 for item in completed if item.get("subject") == "physics")
    lines = [
        "# 50 题本地工程烟测",
        "",
        f"- 开始时间：{started_at}",
        f"- 结果目录：`{output}`",
        f"- 题量：{len(summaries)}（数学 {sum(1 for item in summaries if item.get('subject') == 'math')}，"
        f"物理 {sum(1 for item in summaries if item.get('subject') == 'physics')}）",
        f"- 完成：{len(completed)} / {len(summaries)}（数学 {math_ok}，物理 {physics_ok}）",
        f"- 失败：{len(failed)}",
        f"- 六图合同通过：{len(contract_ok)} / {len(completed)}",
        f"- 墙钟总耗时：{round(elapsed, 1)} s",
        "",
        "时延只统计 `status=completed` 的题目。结构门禁通过不代表公式像素正确。",
        "",
        "## 时延（秒）",
        "",
        "| 指标 | 样本 | P50 | P95 | 均值 |",
        "|---|---:|---:|---:|---:|",
    ]

    def row(label: str, values: list[float]) -> str:
        if not values:
            return f"| {label} | 0 |  |  |  |"
        return (
            f"| {label} | {len(values)} | {percentile(values, 0.50):.3f} | "
            f"{percentile(values, 0.95):.3f} | {statistics.mean(values):.3f} |"
        )

    lines.extend(
        [
            row("OCR 首图", first_images),
            row("学霸解析图", elite_times),
            row("六图完成", complete_times),
            "",
        ]
    )
    if failed:
        lines.extend(["## 失败题目", ""])
        for item in failed:
            error = str(item.get("error") or "unknown").replace("\n", " ")
            lines.append(
                f"- `{item.get('id')}` ({item.get('subject')}): {item.get('status')} — {error}"
            )
        lines.append("")
    lines.extend(
        [
            "## 逐题",
            "",
            "| # | id | 学科 | 状态 | 首图 | 学霸图 | 完成 | 合同 |",
            "|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    for item in summaries:
        lines.append(
            "| {index} | `{id}` | {subject} | {status} | {first} | {elite} | {done} | {ok} |".format(
                index=item.get("index"),
                id=item.get("id"),
                subject=item.get("subject"),
                status=item.get("status"),
                first=item.get("first_image_seconds") if item.get("first_image_seconds") is not None else "",
                elite=item.get("elite_seconds") if item.get("elite_seconds") is not None else "",
                done=item.get("complete_seconds") if item.get("complete_seconds") is not None else "",
                ok="yes" if item.get("image_contract_ok") else "no",
            )
        )
    lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_one_case(
    session: requests.Session,
    base_url: str,
    case: dict[str, str],
    directory: Path,
) -> dict:
    source = Path(case["image"])
    subject = case["subject"]
    data = source.read_bytes()
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, directory / f"source{source.suffix.lower() or '.png'}")
    started = time.perf_counter()
    response = session.post(
        urljoin(base_url, f"api/v1/{subject}/jobs"),
        files={"image": (source.name, data, "image/png")},
        timeout=30,
    )
    response.raise_for_status()
    created = response.json()
    poll_url = urljoin(base_url, created["poll_url"])
    observed: dict[str, float] = {}
    events: list[dict] = []
    state: dict = {}
    while True:
        response = session.get(poll_url, timeout=15)
        response.raise_for_status()
        state = response.json()
        for image in state.get("images", []):
            name = image["filename"]
            if name in observed:
                continue
            png = session.get(urljoin(base_url, image["url"]), timeout=15)
            png.raise_for_status()
            with Image.open(io.BytesIO(png.content)) as decoded:
                decoded.load()
                if decoded.width != 800 or decoded.mode != "L":
                    raise RuntimeError(
                        f"PNG contract mismatch: {name} {decoded.width} {decoded.mode}"
                    )
            (directory / name).write_bytes(png.content)
            observed[name] = round(time.perf_counter() - started, 3)
            event = {
                "filename": name,
                "seconds": observed[name],
                "status": state["status"],
            }
            events.append(event)
            print(
                json.dumps({"id": case["id"], **event}, ensure_ascii=False),
                flush=True,
            )
        if state.get("status") in {"completed", "failed"}:
            break
        if time.perf_counter() - started > 240:
            raise TimeoutError(f"Job still running; do not resubmit: {poll_url}")
        time.sleep(0.3)
    for image in state.get("images", []) or []:
        png = session.get(urljoin(base_url, image["url"]), timeout=15)
        png.raise_for_status()
        (directory / image["filename"]).write_bytes(png.content)
    (directory / "public_job.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (directory / "events.json").write_text(
        json.dumps(events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    local_job = (
        Path(__file__).resolve().parent
        / subject
        / "device_runs"
        / str(state.get("job_id") or "")
        / "job.json"
    )
    private = json.loads(local_job.read_text(encoding="utf-8")) if local_job.is_file() else {}
    present = [name for name in EXPECTED_IMAGES if (directory / name).is_file()]
    return {
        "id": case["id"],
        "subject": subject,
        "source": str(source),
        "job_id": state.get("job_id"),
        "status": state.get("status"),
        "observed_images": observed,
        "image_contract_ok": present == list(EXPECTED_IMAGES),
        "first_image_seconds": observed.get("01_question_knowledge.png"),
        "elite_seconds": observed.get("02_guidance_elite.png"),
        "complete_seconds": observed.get("06_guidance_novice.png")
        or round(time.perf_counter() - started, 3),
        "guidance_observed_before_complete": any(
            event["status"] == "solving" and event["filename"].startswith("02_")
            for event in events
        ),
        "timing": state.get("timing"),
        "analysis_source": (state.get("analysis_audit") or {}).get("source")
        if isinstance(state.get("analysis_audit"), dict)
        else None,
        "usage": private.get("usage"),
        "model_id": private.get("model_id_response"),
        "output_quality_stop": (private.get("provider_contract") or {}).get(
            "output_quality_stop"
        )
        if isinstance(private.get("provider_contract"), dict)
        else None,
        "local_repair_actions": (private.get("provider_contract") or {}).get(
            "local_repair_actions"
        )
        if isinstance(private.get("provider_contract"), dict)
        else None,
        "streaming": private.get("streaming"),
        "total_seconds": private.get("elapsed_seconds"),
        "error": None
        if state.get("status") == "completed"
        else str(state.get("error") or state.get("status") or "failed"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8230/")
    parser.add_argument("--confirm-live-calls", action="store_true", required=True)
    args = parser.parse_args()
    cases = load_cases(args)
    if args.output.exists():
        leftover = [path.name for path in args.output.iterdir() if path.name != "manifest.json"]
        if leftover:
            raise FileExistsError(f"输出目录已有其他文件：{args.output}")
    else:
        args.output.mkdir(parents=True)
    session = requests.Session()
    token = os.environ.get("TEST6_BENCHMARK_TOKEN")
    if token:
        session.headers["Authorization"] = "Bearer " + token
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    wall_started = time.perf_counter()
    summaries: list[dict] = []
    for index, case in enumerate(cases, 1):
        directory = args.output / f"{index:02d}_{case['subject']}_{case['id']}"
        summary = {
            "index": index,
            "id": case["id"],
            "subject": case["subject"],
            "source": case["image"],
            "directory": str(directory),
            "status": "failed",
            "error": None,
        }
        try:
            summary.update(run_one_case(session, args.base_url, case, directory))
            summary["index"] = index
            summary["directory"] = str(directory)
        except Exception as exc:
            summary["status"] = "failed"
            summary["error"] = f"{type(exc).__name__}: {exc}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "error.txt").write_text(
                summary["error"] + "\n\n" + traceback.format_exc(),
                encoding="utf-8",
            )
            if not args.keep_going:
                summaries.append(summary)
                write_summary_files(args.output, summaries)
                raise
        summaries.append(summary)
        write_summary_files(args.output, summaries)
        print(
            json.dumps(
                {
                    "completed_case": index,
                    "id": case["id"],
                    "status": summary.get("status"),
                    "early": summary.get("guidance_observed_before_complete"),
                    "error": summary.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    write_report(
        args.output,
        summaries,
        started_at,
        time.perf_counter() - wall_started,
    )
    if args.keep_going:
        return
    if not all(
        item.get("status") == "completed"
        and item.get("guidance_observed_before_complete")
        for item in summaries
    ):
        raise SystemExit(
            "Some cases did not demonstrate early tier publication; inspect summary.json"
        )


if __name__ == "__main__":
    main()
