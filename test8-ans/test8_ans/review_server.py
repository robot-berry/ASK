from __future__ import annotations

import csv
import json
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .storage import utc_now, write_json


FEEDBACK_VERSION = "test8_ans_review_feedback_v2"
VALID_STATUSES = {"unreviewed", "passed", "issue"}
VALID_SEVERITIES = {"", "low", "medium", "high"}


def _allowed_ids(artifact_root: Path) -> list[str]:
    path = artifact_root / "comparison_manifest.jsonl"
    if not path.is_file():
        raise ValueError(f"审核目录缺少 comparison_manifest.jsonl：{artifact_root}")
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            value = json.loads(line)
            ids.append(str(value["id"]))
    if not ids:
        raise ValueError("审核清单中没有题目")
    return ids


def normalize_feedback_document(
    document: Any, *, allowed_ids: list[str]
) -> dict[str, Any]:
    if not isinstance(document, dict) or not isinstance(document.get("items"), list):
        raise ValueError("反馈必须是包含 items 数组的 JSON 对象")
    allowed = set(allowed_ids)
    normalized_by_id: dict[str, dict[str, Any]] = {}
    for raw in document["items"]:
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("id") or "")
        if item_id not in allowed:
            continue
        status = str(raw.get("status") or "unreviewed")
        severity = str(raw.get("severity") or "")
        if status not in VALID_STATUSES:
            raise ValueError(f"{item_id} 的审核状态无效：{status}")
        if severity not in VALID_SEVERITIES:
            raise ValueError(f"{item_id} 的严重程度无效：{severity}")
        issues = raw.get("issues")
        if not isinstance(issues, list):
            issues = []
        normalized_by_id[item_id] = {
            "id": item_id,
            "status": status,
            "severity": severity,
            "issues": list(
                dict.fromkeys(
                    str(issue).strip() for issue in issues if str(issue).strip()
                )
            ),
            "note": str(raw.get("note") or "").strip(),
            "updated_at": raw.get("updated_at"),
        }
    items = [
        normalized_by_id.get(
            item_id,
            {
                "id": item_id,
                "status": "unreviewed",
                "severity": "",
                "issues": [],
                "note": "",
                "updated_at": None,
            },
        )
        for item_id in allowed_ids
    ]
    return {
        "version": FEEDBACK_VERSION,
        "storage_key": str(document.get("storage_key") or ""),
        "saved_at": utc_now(),
        "items": items,
    }


def summarize_feedback(document: dict[str, Any]) -> dict[str, Any]:
    items = list(document.get("items") or [])
    statuses = Counter(str(item.get("status") or "unreviewed") for item in items)
    severities = Counter(
        str(item.get("severity"))
        for item in items
        if str(item.get("severity") or "")
    )
    issues = Counter(
        str(issue)
        for item in items
        if item.get("status") == "issue"
        for issue in (item.get("issues") or [])
    )
    problem_items = [
        {
            "id": item["id"],
            "severity": item.get("severity") or "",
            "issues": item.get("issues") or [],
            "note": item.get("note") or "",
            "updated_at": item.get("updated_at"),
        }
        for item in items
        if item.get("status") == "issue"
    ]
    return {
        "generated_at": utc_now(),
        "sample_count": len(items),
        "reviewed_count": statuses["passed"] + statuses["issue"],
        "passed_count": statuses["passed"],
        "issue_count": statuses["issue"],
        "unreviewed_count": statuses["unreviewed"],
        "severity_counts": dict(severities),
        "issue_type_counts": dict(issues.most_common()),
        "problem_items": problem_items,
    }


def save_feedback(artifact_root: Path, document: Any) -> dict[str, Any]:
    allowed_ids = _allowed_ids(artifact_root)
    normalized = normalize_feedback_document(document, allowed_ids=allowed_ids)
    summary = summarize_feedback(normalized)
    write_json(artifact_root / "review_feedback.json", normalized)
    write_json(artifact_root / "review_feedback_summary.json", summary)
    with (artifact_root / "review_feedback.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["id", "status", "severity", "issues", "note", "updated_at"],
        )
        writer.writeheader()
        for item in normalized["items"]:
            writer.writerow(
                {
                    **item,
                    "issues": " | ".join(item.get("issues") or []),
                }
            )
    return summary


def _empty_document(artifact_root: Path) -> dict[str, Any]:
    return normalize_feedback_document(
        {"items": []}, allowed_ids=_allowed_ids(artifact_root)
    )


def serve_review(batch_dir: Path, *, host: str = "127.0.0.1", port: int = 0) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("审核服务只允许绑定本机回环地址")
    artifact_root = (batch_dir / "comparison_artifacts").resolve()
    overview = artifact_root / "overview.html"
    if not overview.is_file():
        raise ValueError(f"批次中没有审核页：{overview}")

    class ReviewHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, directory=str(artifact_root), **kwargs)

        def _json_response(self, status: int, value: Any) -> None:
            payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def end_headers(self) -> None:
            # 允许从旧 file:// 审核页一次性迁移浏览器内的反馈。
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")
            super().end_headers()

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/feedback":
                feedback_path = artifact_root / "review_feedback.json"
                document = (
                    json.loads(feedback_path.read_text(encoding="utf-8-sig"))
                    if feedback_path.is_file()
                    else _empty_document(artifact_root)
                )
                self._json_response(200, document)
                return
            if path == "/api/feedback-summary":
                summary_path = artifact_root / "review_feedback_summary.json"
                value = (
                    json.loads(summary_path.read_text(encoding="utf-8-sig"))
                    if summary_path.is_file()
                    else summarize_feedback(_empty_document(artifact_root))
                )
                self._json_response(200, value)
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/feedback":
                self._json_response(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                if length <= 0 or length > 2_000_000:
                    raise ValueError("反馈数据大小无效")
                document = json.loads(self.rfile.read(length).decode("utf-8"))
                summary = save_feedback(artifact_root, document)
                self._json_response(200, {"saved": True, "summary": summary})
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                self._json_response(400, {"saved": False, "error": str(exc)})

    server = ThreadingHTTPServer((host, port), ReviewHandler)
    actual_port = int(server.server_address[1])
    print(f"审核页：http://{host}:{actual_port}/overview.html", flush=True)
    print(
        f"反馈将自动保存到：{artifact_root / 'review_feedback.json'}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
