#!/usr/bin/env python3
"""Resumable image-question benchmark for Bailian's OpenAI-compatible API.

The API runner keeps credentials in memory, streams each response, validates
the four-key JSON contract locally, and writes one durable JSONL record per
sample/model pair.  Optional post-run monochrome PNG publishing uses Pillow
and never changes the durable JSONL source of truth.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
from contextlib import contextmanager
import csv
import ctypes
from ctypes import wintypes
import datetime as dt
import hashlib
import http.client
import io
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request
import uuid


LEGACY_ROOT_KEYS = (
    "classification",
    "analysis",
    "solution",
    "final_answer",
)
# Kept as the public legacy default for historical configs and tests. New
# schemas derive their ordered root fields from JSON Schema ``required``.
ROOT_KEYS = LEGACY_ROOT_KEYS
ANSWER_FIRST_ACCURACY_ROOT_KEYS = (
    "final",
    "classification",
    "analysis",
    "verification",
)
FINAL_ANSWER_COMPLETE_MILESTONE = "final_answer_complete"
MAX_REQUEST_ATTEMPTS = 5
PROVIDER_CAPACITY_PAUSE_EXIT_CODE = 75
DEFERRED_PROVIDER_CAPACITY_DISPOSITION = "deferred_provider_capacity"
VISUAL_DEPENDENCY_MARKERS = (
    "如图",
    "图中",
    "由图",
    "下图",
    "根据图",
    "结合图",
    "观察图",
    "读图",
    "示意图",
    "电路图",
    "受力图",
    "光路图",
    "统计图",
    "表中",
)
VISIBLE_INFORMATION_PREFIX = "图中可见信息："
# Bare mathematical terms such as “函数的图象/图像” do not prove that the
# photographed question depends on a drawn figure. Explicit phrases such as
# “如图”“图中”“由图”“观察图” remain visual-dependency markers.
JSON_ESCAPE_CONTROL_LATEX_COMMANDS = frozenset({
    "backslash",
    "bar",
    "because",
    "begin",
    "beta",
    "bf",
    "big",
    "bigdots",
    "binom",
    "boldsymbol",
    "bot",
    "boxed",
    "bullet",
    "flat",
    "forall",
    "frac",
    "tan",
    "tanh",
    "tau",
    "text",
    "textbf",
    "textit",
    "textnormal",
    "textstyle",
    "tfrac",
    "theta",
    "therefore",
    "thickapprox",
    "thicksim",
    "tilde",
    "times",
    "to",
    "top",
    "triangle",
    "triangledown",
    "triangleleft",
    "triangleq",
    "triangleright",
    "nabla",
    "ne",
    "neg",
    "neq",
    "not",
    "notin",
    "nu",
    "rangle",
    "rho",
    "right",
    "rightarrow",
    "rm",
    "mathrm",
})
JSON_ESCAPE_CONTROL_LATEX_ALIASES = {
    "bigdots": "cdots",
}
JSON_ESCAPE_CONTROL_PREFIXES = {
    "\b": "b",
    "\t": "t",
    "\f": "f",
    "\n": "n",
    "\r": "r",
}
JSON_CONTROL_REPAIR_AUDIT_LIMIT = 50
SOLUTION_SELF_TALK_MARKERS = (
    "我觉得",
    "答案可能是",
    "可能选",
    "可能为选项",
    "先试一下",
    "先试试",
    "再试一下",
    "等等，",
    "等等,",
    "让我想想",
    "让我重新",
    "让我再算",
    "哪里错",
    "不完全是",
    "重新分析",
    "重新检查",
    "重新看",
    "重看",
    "让我们再看",
    "让我们看一遍",
    "让我们再次确认",
    "大概答案",
    "大概选",
    "猜一个",
)
VERIFICATION_DISALLOWED_MARKERS = (
    *SOLUTION_SELF_TALK_MARKERS,
    "？",
    "?",
    "结论修正",
    "答案修正",
    "重新展开",
    "不对，",
    "不对。",
    "之前的结论",
    "刚才的结论",
)
VERIFICATION_MAX_CHARACTERS = 600
EXAM_RULE_GUESS_MARKERS = (
    "标准正确答案",
    "标准答案",
    "参考答案",
    "命题人",
    "出题人",
    "命题意图",
    "考试意图",
    "考试规则",
    "通常这种题",
    "通常这类题",
    "通常是单选",
    "通常为单选",
    "如果是多选",
    "若是多选",
    "如果为多选",
    "若为多选",
    "如果是单选",
    "若是单选",
    "如果为单选",
    "若为单选",
    "干扰项",
    "代表性",
    "教材语境",
    "旨在纠正",
    "按考试习惯",
    "按命题习惯",
    "题目来源可能",
    "题源误差",
    "题源可能有误",
    "录入错误",
    "理解偏差",
    "最高频考点",
    "考察初衷",
    "必须二选一",
    "答案权重",
)
FINAL_ANSWER_GUESS_MARKERS = (
    "答案可能是",
    "可能选",
    "可能为选项",
    "大概是",
    "大概选",
    "猜测",
    "猜一个",
    "应该是",
)
SUBJECT_ALIASES = {
    "mathematics": "mathematics",
    "math": "mathematics",
    "数学": "mathematics",
    "高中数学": "mathematics",
    "mathematics": "mathematics",
    "数学": "mathematics",
    "高中数学": "mathematics",
}
HIGH_SCHOOL_GRADES = {"高中", "高一", "高二", "高三"}
OUT_OF_SCOPE_GRADES = {
    "小学", "初中", "初一", "初二", "初三", "大学"
}
SAFE_RESPONSE_HEADERS = {
    "x-request-id",
    "request-id",
    "x-dashscope-request-id",
    "traceparent",
    "server-timing",
}
RESULT_VERSION = "2026-07-28.v2"
SUMMARY_VERSION = "2026-07-28.v3"
VALIDATION_POLICY_VERSION = "2026-07-29.usable-output.v5"
WINDOWS_SCHANNEL_SCRIPT_SHA256 = (
    "b043b1ab0b81f73bd4221f3daff16d74c77d6aff976f80e190b3539537c0c012"
)
WINDOWS_SCHANNEL_BATCH_SCRIPT_SHA256 = (
    "9f5b20cbbd77e76d1854e85adf89476606189d4d2d9524e6c79183014c20c14f"
)
WINDOWS_SCHANNEL_PERSISTENT_SCRIPT_SHA256 = (
    "ca0ddfbca94bc7e2215f870e0938a43a2623cccba5e4f6b44d03353fac8f2757"
)
WINDOWS_SCHANNEL_PERSISTENT_V2_SCRIPT_SHA256 = (
    "69972faa9e5677243a16e2601b6b72ec7888c2e1c41ff296c6061eb30586a518"
)
WINDOWS_SCHANNEL_PERSISTENT_SCRIPTS = {
    "windows_schannel_persistent_transport.ps1": {
        "sha256": WINDOWS_SCHANNEL_PERSISTENT_SCRIPT_SHA256,
        "protocol": "yuyi-persistent-iwr-v1",
        "strict_private_acl": False,
    },
    "windows_schannel_persistent_transport_v2.ps1": {
        "sha256": WINDOWS_SCHANNEL_PERSISTENT_V2_SCRIPT_SHA256,
        "protocol": "yuyi-persistent-iwr-v2",
        "strict_private_acl": True,
    },
}
PERSISTENT_IWR_PROTOCOL = "yuyi-persistent-iwr-v1"
PERSISTENT_IWR_MAX_LINE_BYTES = 65536
PERSISTENT_IWR_MAX_SINGLE_IMAGE_BYTES = 10 * 1024 * 1024
PERSISTENT_IWR_MAX_QUESTION_BYTES = 24 * 1024 * 1024
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
WRITE_LOCK = threading.Lock()
_STREAM_CONTENT_CALLBACK = threading.local()


@contextmanager
def stream_content_callback(
    callback: Callable[[str, float | None], None] | None,
) -> Iterator[None]:
    """Expose accumulated provider content to the current pipeline thread."""

    previous = getattr(_STREAM_CONTENT_CALLBACK, "callback", None)
    _STREAM_CONTENT_CALLBACK.callback = callback
    try:
        yield
    finally:
        _STREAM_CONTENT_CALLBACK.callback = previous

PHASE_DURATION_KEYS = (
    "pre_first_content_proxy",
    "classification_output",
    "analysis_output",
    "solution_output",
    "final_answer_output",
    "answer_output",
)
RECORD_TIMING_KEYS = (
    "response_headers_wait",
    "first_event",
    "first_content",
    "first_reasoning",
    "provider_total",
    "end_to_end",
    "image_load_encode",
)
RECORD_TIMING_FIELDS = {
    "response_headers_wait": "provider_response_headers_wait_ms",
    "first_event": "provider_first_event_ms",
    "first_content": "provider_first_content_ms",
    "first_reasoning": "provider_first_reasoning_ms",
    "provider_total": "provider_total_ms",
    "end_to_end": "end_to_end_ms",
    "image_load_encode": "image_load_encode_ms",
}


def schema_root_keys(schema: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return the ordered top-level contract, preserving legacy defaults."""
    if not isinstance(schema, Mapping):
        return ROOT_KEYS
    required = schema.get("required")
    if not isinstance(required, list) or not required:
        return ROOT_KEYS
    keys = tuple(
        item for item in required
        if isinstance(item, str) and item.strip()
    )
    if len(keys) != len(required) or len(set(keys)) != len(keys):
        return ROOT_KEYS
    return keys


def milestone_pattern(key: str) -> re.Pattern[str]:
    return re.compile(
        r'["\']' + re.escape(key) + r'["\']\s*:',
        re.IGNORECASE,
    )


def milestone_state(root_keys: Sequence[str]) -> dict[str, float | None]:
    """Create the stream milestone map for an ordered output contract.

    The answer-first K3 contract records both the start of ``final`` and the
    moment its first, non-empty ``answer`` field is completely decoded.
    Historical schemas retain their existing milestone keys unchanged.
    """
    state = {key: None for key in root_keys}
    if tuple(root_keys) == ANSWER_FIRST_ACCURACY_ROOT_KEYS:
        state[FINAL_ANSWER_COMPLETE_MILESTONE] = None
    return state


def completed_first_root_field(
    accumulated: str,
    expected_root_key: str,
    expected_field_key: str,
) -> Any | None:
    """Decode the first field inside the first root object from partial JSON.

    The surrounding object may still be open. Requiring both keys to be first
    prevents a nested or quoted ``answer`` fragment from being mistaken for
    the durable answer-first streaming milestone.
    """
    if not isinstance(accumulated, str) or not accumulated:
        return None
    root_start = accumulated.find("{")
    if root_start < 0:
        return None
    decoder = json.JSONDecoder()
    index = root_start + 1

    def skip_space(position: int) -> int:
        while position < len(accumulated) and accumulated[position].isspace():
            position += 1
        return position

    index = skip_space(index)
    try:
        root_key, index = decoder.raw_decode(accumulated, index)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if root_key != expected_root_key:
        return None
    index = skip_space(index)
    if index >= len(accumulated) or accumulated[index] != ":":
        return None
    index = skip_space(index + 1)
    if index >= len(accumulated) or accumulated[index] != "{":
        return None
    index = skip_space(index + 1)
    try:
        field_key, index = decoder.raw_decode(accumulated, index)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if field_key != expected_field_key:
        return None
    index = skip_space(index)
    if index >= len(accumulated) or accumulated[index] != ":":
        return None
    index = skip_space(index + 1)
    try:
        value, _ = decoder.raw_decode(accumulated, index)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0


def rounded_ms(value: float | None) -> float | None:
    return None if value is None else round(max(0.0, value), 3)


def bounded_request_attempts(request_config: Mapping[str, Any]) -> int:
    """Honor accuracy-first retry budgets while keeping a finite paid-call cap."""
    return min(
        MAX_REQUEST_ATTEMPTS,
        max(1, int(request_config.get("max_attempts", 2))),
    )


def provider_capacity_pause_policy(
    request_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the optional fail-fast provider-capacity pause contract.

    This is deliberately a batch-level pause, not an automatic model retry.
    A matched request is retained only as an operational event. Because that
    event is not written to results.jsonl, a first-pass task is naturally
    pending on the next invocation; an older durable failure still requires
    the same explicit retry mode that selected it.
    """

    raw = request_config.get("provider_capacity_pause")
    if raw is None:
        return {
            "enabled": False,
            "http_statuses": [],
            "provider_error_codes": [],
            "inter_task_delay_seconds": 0.0,
            "pause_on_first_match": True,
            "automatic_resume": False,
        }
    if not isinstance(raw, Mapping):
        raise ValueError("request.provider_capacity_pause必须是对象")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("provider_capacity_pause.enabled必须是布尔值")
    if not enabled:
        return {
            "enabled": False,
            "http_statuses": [],
            "provider_error_codes": [],
            "inter_task_delay_seconds": 0.0,
            "pause_on_first_match": True,
            "automatic_resume": False,
        }
    if raw.get("pause_on_first_match") is not True:
        raise ValueError(
            "provider_capacity_pause.pause_on_first_match必须为true"
        )
    if raw.get("automatic_resume", False) is not False:
        raise ValueError("provider_capacity_pause.automatic_resume必须为false")
    raw_statuses = raw.get("http_statuses")
    if (
        not isinstance(raw_statuses, list)
        or not raw_statuses
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            or value < 400 or value > 599
            for value in raw_statuses
        )
    ):
        raise ValueError(
            "provider_capacity_pause.http_statuses必须是非空HTTP错误码数组"
        )
    raw_codes = raw.get("provider_error_codes")
    if (
        not isinstance(raw_codes, list)
        or not raw_codes
        or any(not isinstance(value, str) or not value for value in raw_codes)
    ):
        raise ValueError(
            "provider_capacity_pause.provider_error_codes必须是非空字符串数组"
        )
    delay = raw.get("inter_task_delay_seconds", 0)
    if (
        not isinstance(delay, (int, float)) or isinstance(delay, bool)
        or not math.isfinite(float(delay)) or float(delay) < 0
        or float(delay) > 300
    ):
        raise ValueError(
            "provider_capacity_pause.inter_task_delay_seconds必须在0到300之间"
        )
    return {
        "enabled": True,
        "http_statuses": sorted(set(raw_statuses)),
        "provider_error_codes": sorted(set(raw_codes)),
        "inter_task_delay_seconds": float(delay),
        "pause_on_first_match": True,
        "automatic_resume": False,
    }


def provider_capacity_pause_match(
    record: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return a redacted pause trigger for an exact provider-capacity error."""

    if not policy.get("enabled") or record.get("status") != "failed_api":
        return None
    statuses = set(policy.get("http_statuses", []))
    codes = set(policy.get("provider_error_codes", []))
    attempts = record.get("attempts")
    if not isinstance(attempts, list):
        return None
    for attempt_index in range(len(attempts) - 1, -1, -1):
        attempt = attempts[attempt_index]
        if not isinstance(attempt, Mapping):
            continue
        status = attempt.get("http_status")
        if status not in statuses:
            continue
        code = attempt.get("provider_error_code")
        if not isinstance(code, str) or code not in codes:
            error = attempt.get("error")
            message = (
                str(error.get("message"))
                if isinstance(error, Mapping) else ""
            )
            code = next(
                (
                    candidate for candidate in codes
                    if re.search(
                        rf"(?<![A-Za-z0-9_.-]){re.escape(candidate)}"
                        rf"(?![A-Za-z0-9_.-])",
                        message,
                    )
                ),
                None,
            )
        if isinstance(code, str) and code in codes:
            return {
                "http_status": status,
                "provider_error_code": code,
                "attempt": attempt_index + 1,
            }
    return None


def mark_deferred_provider_capacity(
    record: dict[str, Any], trigger: Mapping[str, Any]
) -> None:
    record["batch_control"] = {
        "disposition": DEFERRED_PROVIDER_CAPACITY_DISPOSITION,
        "accuracy_denominator_included": False,
        "resume_requires_retry_failures": None,
        "trigger": dict(trigger),
    }


def is_deferred_provider_capacity(record: Mapping[str, Any]) -> bool:
    control = record.get("batch_control")
    return (
        isinstance(control, Mapping)
        and control.get("disposition")
        == DEFERRED_PROVIDER_CAPACITY_DISPOSITION
    )


def execute_sequential_with_provider_capacity_pause(
    task_items: Sequence[Any],
    execute_one: Any,
    record_one: Any,
    capacity_event_one: Any,
    policy: Mapping[str, Any],
    sleep_fn: Any = time.sleep,
) -> dict[str, Any]:
    """Execute one item at a time and stop before scheduling after overload."""

    completed = 0
    for index, item in enumerate(task_items):
        record = execute_one(item)
        trigger = provider_capacity_pause_match(record, policy)
        if trigger is not None:
            mark_deferred_provider_capacity(record, trigger)
            capacity_event_one(item, record)
            task_key = (
                item[2]
                if isinstance(item, (tuple, list)) and len(item) > 2
                else None
            )
            return {
                "paused": True,
                "completed_item_count": completed,
                "provider_request_event_count": completed + 1,
                "capacity_event_count": 1,
                "unstarted_item_count": len(task_items) - completed - 1,
                "trigger_task_key": task_key,
                "trigger": trigger,
            }
        record_one(item, record)
        completed += 1
        delay = float(policy.get("inter_task_delay_seconds", 0.0))
        if delay and index + 1 < len(task_items):
            sleep_fn(delay)
    return {
        "paused": False,
        "completed_item_count": completed,
        "provider_request_event_count": completed,
        "capacity_event_count": 0,
        "unstarted_item_count": 0,
        "trigger_task_key": None,
        "trigger": None,
    }


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def safe_component(value: str, label: str) -> str:
    candidate = str(value).strip()
    if not candidate or candidate in {".", ".."}:
        raise ValueError(f"{label}不能为空")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", candidate):
        raise ValueError(f"{label}只能包含字母、数字、点、下划线和连字符")
    return candidate


def sanitize_url(value: str) -> str:
    """Return a log-safe URL without userinfo, query string, or fragment."""
    parsed = urllib.parse.urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def redact_text(value: Any, secrets: Sequence[str], limit: int = 4000) -> str:
    text = str(value or "")
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    return text[:limit]


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
        delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False
    ).encode("utf-8")
    atomic_write_bytes(path, payload + b"\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    payload = json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL解析失败：{path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL记录必须是对象：{path}:{line_number}")
            records.append(value)
    return records


def load_existing_results(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load valid records and tolerate only a partial final line after a crash."""
    if not path.exists():
        return [], []
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    raw_lines = path.read_bytes().splitlines()
    for index, raw in enumerate(raw_lines, 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("record is not an object")
            records.append(value)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            if index == len(raw_lines):
                warnings.append(f"忽略疑似崩溃留下的不完整末行：{exc}")
                continue
            raise ValueError(f"历史结果损坏：{path}:{index}: {exc}") from exc
    return records, warnings


def parse_env_file(path: Path | None) -> list[str]:
    """Load KEY=VALUE pairs without returning or printing any values."""
    loaded: list[str] = []
    if path is None or not path.exists():
        return loaded
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            raise ValueError(f"env格式错误：{path}:{line_number}")
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"env变量名无效：{path}:{line_number}")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            quote = value[0]
            value = value[1:-1]
            if quote == '"':
                value = (
                    value.replace(r"\\", "\\")
                    .replace(r"\n", "\n")
                    .replace(r"\r", "\r")
                    .replace(r"\t", "\t")
                    .replace(r"\"", '"')
                )
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def resolve_existing_path(
    value: str | os.PathLike[str], bases: Sequence[Path], label: str
) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [base / raw for base in bases]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    tried = "、".join(str(item.resolve()) for item in candidates)
    raise FileNotFoundError(f"{label}不存在；已尝试：{tried}")


def normalize_models(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_models = config.get("models", [])
    output: dict[str, dict[str, Any]] = {}
    if isinstance(raw_models, Mapping):
        iterable: Iterable[tuple[str, Any]] = raw_models.items()
    elif isinstance(raw_models, list):
        iterable = (
            (str(item.get("key") or item.get("name") or ""), item)
            for item in raw_models if isinstance(item, Mapping)
        )
    else:
        raise ValueError("config.models必须是对象或对象数组")
    for key, raw in iterable:
        if not key:
            raise ValueError("每个模型必须有key")
        model = dict(raw)
        model.setdefault("key", key)
        if not model.get("model_id"):
            raise ValueError(f"模型{key}缺少model_id")
        if key in output:
            raise ValueError(f"模型key重复：{key}")
        output[key] = model
    if not output:
        raise ValueError("config.models不能为空")
    return output


def selected_model_keys(
    raw_values: Sequence[str] | None, available: Mapping[str, Any]
) -> list[str]:
    if not raw_values:
        return list(available)
    values: list[str] = []
    for raw in raw_values:
        values.extend(item.strip() for item in raw.split(",") if item.strip())
    if values == ["all"] or "*" in values:
        return list(available)
    seen: set[str] = set()
    result: list[str] = []
    model_id_to_key = {
        str(model["model_id"]): key for key, model in available.items()
    }
    for value in values:
        key = value if value in available else model_id_to_key.get(value)
        if key is None:
            raise ValueError(
                f"未知模型：{value}；可选：{', '.join(available.keys())}"
            )
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def dataset_paths(
    args: argparse.Namespace, config: Mapping[str, Any], config_dir: Path,
    repo_root: Path, workspace_root: Path
) -> tuple[Path, Path, Path]:
    dataset = config.get("dataset")
    dataset_map = dataset if isinstance(dataset, Mapping) else {}
    dataset_id = str(
        config.get("dataset_id") or dataset_map.get("dataset_id")
        or "testzip_math_mathematics_300_v1"
    )
    bases = (config_dir, workspace_root, repo_root, Path.cwd())

    root_value = (
        args.dataset_root or dataset_map.get("root")
        or dataset_map.get("dataset_root")
    )
    if root_value:
        root = resolve_existing_path(root_value, bases, "数据集目录")
    else:
        root = resolve_existing_path(
            Path("datasets") / dataset_id, bases, "数据集目录"
        )

    manifest_value = (
        args.manifest or dataset_map.get("manifest") or root / "manifest.jsonl"
    )
    manifest = resolve_existing_path(manifest_value, (root,) + bases, "manifest")
    smoke_value = (
        args.smoke_manifest or dataset_map.get("smoke")
        or dataset_map.get("smoke_manifest")
        or dataset_map.get("smoke_subset") or root / "smoke_20.jsonl"
    )
    smoke = resolve_existing_path(smoke_value, (root,) + bases, "smoke manifest")
    return root, manifest, smoke


def select_samples(
    stage: str, manifest_path: Path, smoke_path: Path, max_samples: int | None
) -> list[dict[str, Any]]:
    manifest = load_jsonl(manifest_path)
    by_id: dict[str, dict[str, Any]] = {}
    for record in manifest:
        question_id = str(record.get("question_id") or "")
        if not question_id:
            raise ValueError("manifest存在缺少question_id的记录")
        if question_id in by_id:
            raise ValueError(f"manifest question_id重复：{question_id}")
        by_id[question_id] = record

    if stage == "full":
        selected = [
            dict(record) for record in manifest
            if record.get("include_in_benchmark") is True
        ]
    else:
        selected = []
        seen: set[str] = set()
        for smoke_record in load_jsonl(smoke_path):
            question_id = str(smoke_record.get("question_id") or "")
            if question_id in seen:
                raise ValueError(f"smoke question_id重复：{question_id}")
            seen.add(question_id)
            if question_id not in by_id:
                raise ValueError(f"smoke题目不在manifest中：{question_id}")
            merged = dict(by_id[question_id])
            merged.update(smoke_record)
            if by_id[question_id].get("include_in_benchmark") is not True:
                raise ValueError(f"smoke包含已排除样本：{question_id}")
            selected.append(merged)

    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("--max-samples必须大于0")
        selected = selected[:max_samples]
    if not selected:
        raise ValueError(f"{stage}阶段没有可运行样本")
    return selected


def read_prompt(path: Path) -> str:
    if path.suffix.lower() == ".json":
        value = load_json(path)
        if isinstance(value, str):
            prompt = value
        elif isinstance(value, Mapping):
            prompt = str(
                value.get("prompt") or value.get("user_prompt")
                or value.get("system_prompt") or ""
            )
        else:
            prompt = ""
    else:
        prompt = path.read_text(encoding="utf-8-sig")
    prompt = prompt.strip()
    if not prompt:
        raise ValueError(f"提示词为空：{path}")
    return prompt


def compose_prompts(parts: Sequence[str]) -> str:
    material = [str(part).strip() for part in parts if str(part).strip()]
    if not material:
        raise ValueError("组合提示词不能为空")
    return "\n\n".join(material)


def resolve_prompt_constraint_paths(
    value: Any,
    search_roots: Sequence[Path],
    label: str,
) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, str):
        references = [value]
    elif isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray)
    ):
        references = [str(item) for item in value]
    else:
        raise ValueError(f"{label}必须是路径字符串或路径数组")
    paths: list[Path] = []
    for index, reference in enumerate(references, 1):
        if not reference.strip():
            raise ValueError(f"{label}[{index}]不能为空")
        paths.append(
            resolve_existing_path(
                reference,
                search_roots,
                f"{label}[{index}]",
            )
        )
    return paths


def effective_prompt(
    base_path: Path,
    constraint_paths: Sequence[Path],
) -> str:
    return compose_prompts(
        [read_prompt(base_path)]
        + [read_prompt(path) for path in constraint_paths]
    )


def resolve_prompt_constraints_by_subject(
    value: Any,
    search_roots: Sequence[Path],
) -> dict[str, list[Path]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("prompt_constraints_by_subject必须是对象")
    allowed = {"mathematics", "mathematics"}
    unsupported = sorted(str(key) for key in value if str(key) not in allowed)
    if unsupported:
        raise ValueError(
            "prompt_constraints_by_subject包含不支持的学科："
            + ", ".join(unsupported)
        )
    return {
        str(subject): resolve_prompt_constraint_paths(
            references,
            search_roots,
            f"prompt_constraints_by_subject.{subject}",
        )
        for subject, references in value.items()
    }


def subject_prompt_route(
    context: Mapping[str, Any], sample: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    subject = str(sample.get("subject") or "").strip().lower()
    prompts = context.get("prompts_by_subject")
    paths = context.get("prompt_constraint_paths_by_subject")
    if isinstance(prompts, Mapping) and subject in prompts:
        prompt = str(prompts[subject])
        selected_paths = paths.get(subject, []) if isinstance(paths, Mapping) else []
        return prompt, {
            "source": "manifest_subject_v1",
            "manifest_subject": subject,
            "selected_subject": subject,
            "fallback_used": False,
            "constraint_paths": [str(path) for path in selected_paths],
            "effective_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        }
    prompt = str(context["prompt"])
    return prompt, {
        "source": "default_prompt_v1",
        "manifest_subject": subject or None,
        "selected_subject": None,
        "fallback_used": True,
        "constraint_paths": [],
        "effective_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
    }


def schema_errors(value: Any, schema: Mapping[str, Any], path: str = "$") -> list[str]:
    """Validate the JSON-Schema subset used by the HSQA output schemas."""
    if "anyOf" in schema:
        branches = schema.get("anyOf")
        if not isinstance(branches, list) or not branches:
            return [f"{path}: anyOf无有效分支"]
        if any(not schema_errors(value, branch, path) for branch in branches):
            return []
        return [f"{path}: 不符合anyOf中的任何分支"]

    errors: list[str] = []
    expected_type = schema.get("type")
    type_checks = {
        "object": lambda item: isinstance(item, dict),
        "array": lambda item: isinstance(item, list),
        "string": lambda item: isinstance(item, str),
        "number": lambda item: (
            isinstance(item, (int, float)) and not isinstance(item, bool)
            and math.isfinite(float(item))
        ),
        "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "boolean": lambda item: isinstance(item, bool),
        "null": lambda item: item is None,
    }
    if expected_type in type_checks and not type_checks[expected_type](value):
        return [f"{path}: 类型应为{expected_type}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: 值不在允许范围内")
    if expected_type == "number" or expected_type == "integer":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{path}: 小于minimum")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{path}: 大于maximum")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            errors.append(f"{path}: 短于minLength")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            errors.append(f"{path}: 长于maxLength")
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}: 缺少字段{key}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: 不允许额外字段{key}")
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, Mapping):
                errors.extend(schema_errors(value[key], child_schema, f"{path}.{key}"))
    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            errors.append(f"{path}: 少于minItems")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            errors.append(f"{path}: 多于maxItems")
        if isinstance(schema.get("items"), Mapping):
            for index, item in enumerate(value):
                errors.extend(
                    schema_errors(item, schema["items"], f"{path}[{index}]")
                )
    return errors


def repair_decoded_json_latex_controls(
    value: Any,
) -> tuple[Any, dict[str, Any]]:
    r"""Restore narrowly identifiable LaTeX commands decoded as JSON controls.

    JSON accepts ``\b``, ``\t`` and ``\f`` escapes. A model that emits a
    single backslash before ``\bot``, ``\triangle`` or ``\frac`` therefore
    produces a valid JSON document whose decoded string contains a C0 control.
    Only allowlisted commands inside dollar-delimited math are restored.
    Ordinary tabs/backspaces, unknown commands, and controls outside math are
    left untouched for ``semantic_content_errors`` to reject.
    """
    repair_count = 0
    repair_items: list[dict[str, Any]] = []

    def record(
        path: str,
        control: str,
        source_command: str,
        restored_command: str,
    ) -> None:
        nonlocal repair_count
        repair_count += 1
        if len(repair_items) < JSON_CONTROL_REPAIR_AUDIT_LIMIT:
            repair_items.append({
                "path": path,
                "source_codepoint": f"U+{ord(control):04X}",
                "decoded_command": "\\" + source_command,
                "restored_command": "\\" + restored_command,
            })

    def repair_text(text: str, path: str) -> str:
        output: list[str] = []
        in_math = False
        index = 0
        while index < len(text):
            if text.startswith("$$", index):
                in_math = not in_math
                output.append("$$")
                index += 2
                continue
            char = text[index]
            if char == "$":
                slash_count = 0
                cursor = index - 1
                while cursor >= 0 and text[cursor] == "\\":
                    slash_count += 1
                    cursor -= 1
                if slash_count % 2 == 0:
                    in_math = not in_math
                output.append(char)
                index += 1
                continue
            prefix = JSON_ESCAPE_CONTROL_PREFIXES.get(char)
            if in_math and prefix is not None:
                end = index + 1
                while end < len(text) and (
                    "A" <= text[end] <= "Z"
                    or "a" <= text[end] <= "z"
                ):
                    end += 1
                command = prefix + text[index + 1:end]
                if command in JSON_ESCAPE_CONTROL_LATEX_COMMANDS:
                    restored_command = JSON_ESCAPE_CONTROL_LATEX_ALIASES.get(
                        command,
                        command,
                    )
                    trailing_slashes = 0
                    for item in reversed(output):
                        if item != "\\":
                            break
                        trailing_slashes += 1
                    if trailing_slashes % 2 == 0:
                        output.append("\\")
                    output.append(restored_command)
                    record(path, char, command, restored_command)
                    index = end
                    continue
            output.append(char)
            index += 1
        return "".join(output)

    def walk(item: Any, path: str) -> Any:
        if isinstance(item, str):
            return repair_text(item, path)
        if isinstance(item, dict):
            for key in list(item):
                item[key] = walk(item[key], f"{path}.{key}")
            return item
        if isinstance(item, Mapping):
            return {
                key: walk(child, f"{path}.{key}")
                for key, child in item.items()
            }
        if isinstance(item, list):
            for index, child in enumerate(item):
                item[index] = walk(child, f"{path}[{index}]")
            return item
        return item

    repaired = walk(value, "$")
    return repaired, {
        "count": repair_count,
        "items": repair_items,
        "truncated": repair_count > len(repair_items),
        "policy": "allowlisted_latex_commands_inside_dollar_math_only",
    }


def semantic_content_errors(
    value: Any, required_roots: Sequence[str] = ROOT_KEYS
) -> list[str]:
    """Check usable content, restoring narrow LaTeX controls first."""
    # The two-stage runner already calls this shared gate after json.loads.
    # Repairing in place keeps its frozen runner unchanged while ensuring the
    # corrected strings are also the ones returned as validated output.
    value, _ = repair_decoded_json_latex_controls(value)
    errors: list[str] = []

    def walk(item: Any, path: str) -> None:
        if isinstance(item, str):
            # Escaped CR/LF are useful for multi-line question text. Other C0
            # controls often mean a LaTeX backslash was decoded as \b, \f or \t.
            controls = sorted({
                ord(char) for char in item
                if ord(char) < 0x20 and char not in "\r\n"
            })
            if controls:
                rendered = ",".join(f"U+{code:04X}" for code in controls)
                errors.append(f"{path}: 包含控制字符{rendered}")
        elif isinstance(item, Mapping):
            for key, child in item.items():
                walk(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, "$")
    if not isinstance(value, Mapping):
        return errors

    def require_text(parent: Any, key: str, path: str) -> None:
        child = parent.get(key) if isinstance(parent, Mapping) else None
        if not isinstance(child, str) or not child.strip():
            errors.append(f"{path}: 必须是非空字符串")

    def first_marker(text: Any, markers: Sequence[str]) -> str | None:
        if not isinstance(text, str):
            return None
        return next((marker for marker in markers if marker in text), None)

    classification = value.get("classification")
    analysis = value.get("analysis")
    solution = value.get("solution")
    final_answer = value.get("final_answer")
    question = value.get("question")
    reasoning = value.get("reasoning")
    final = value.get("final")
    verification = value.get("verification")

    if "classification" in required_roots:
        for key in (
            "course_module",
            "question_type",
            "knowledge_level_1",
            "knowledge_level_2",
            "knowledge_level_3",
        ):
            require_text(classification, key, f"$.classification.{key}")
        question_type = (
            classification.get("question_type")
            if isinstance(classification, Mapping)
            else None
        )
        if (
            isinstance(question_type, str)
            and question_type.strip() in {"选择", "选择题"}
        ):
            errors.append(
                "$.classification.question_type: 不得只写“选择题”；"
                "题面未说明单选或多选时必须写"
                "“选择题（单选/多选未说明）”"
            )
    if "analysis" in required_roots:
        fast_two_output = (
            "classification" not in required_roots
            and "answer" in required_roots
        )
        if fast_two_output:
            for key in ("subject", "question_type", "explanation"):
                require_text(analysis, key, f"$.analysis.{key}")
            knowledge_points = (
                analysis.get("knowledge_points")
                if isinstance(analysis, Mapping)
                else None
            )
            if not isinstance(knowledge_points, list) or not knowledge_points:
                errors.append(
                    "$.analysis.knowledge_points: 必须是非空字符串数组"
                )
            elif any(
                not isinstance(item, str) or not item.strip()
                for item in knowledge_points
            ):
                errors.append(
                    "$.analysis.knowledge_points: 每一项必须是非空字符串"
                )
            question_type = (
                analysis.get("question_type")
                if isinstance(analysis, Mapping)
                else None
            )
            if (
                isinstance(question_type, str)
                and question_type.strip() in {"选择", "选择题"}
            ):
                errors.append(
                    "$.analysis.question_type: 不得只写“选择题”；"
                    "题面未说明单选或多选时必须写"
                    "“选择题（单选/多选未说明）”"
                )
            explanation = (
                analysis.get("explanation")
                if isinstance(analysis, Mapping)
                else None
            )
            self_talk_marker = first_marker(
                explanation,
                SOLUTION_SELF_TALK_MARKERS,
            )
            if self_talk_marker is not None:
                errors.append(
                    "$.analysis.explanation: 禁止自言自语或试探性推理标记"
                    f"“{self_talk_marker}”"
                )
            exam_marker = first_marker(
                explanation,
                EXAM_RULE_GUESS_MARKERS,
            )
            if exam_marker is not None:
                errors.append(
                    "$.analysis.explanation: 禁止根据考试规则、命题意图或"
                    f"答案习惯猜测，发现标记“{exam_marker}”"
                )
        else:
            for key in ("normalized_question", "target", "method"):
                require_text(analysis, key, f"$.analysis.{key}")
            for key in ("known_conditions", "key_relations"):
                items = (
                    analysis.get(key)
                    if isinstance(analysis, Mapping)
                    else None
                )
                if not isinstance(items, list) or not items:
                    errors.append(f"$.analysis.{key}: 必须是非空字符串数组")
                elif any(
                    not isinstance(item, str) or not item.strip()
                    for item in items
                ):
                    errors.append(
                        f"$.analysis.{key}: 每一项必须是非空字符串"
                    )
            normalized_question = (
                analysis.get("normalized_question")
                if isinstance(analysis, Mapping)
                else None
            )
            visual_marker = first_marker(
                normalized_question,
                VISUAL_DEPENDENCY_MARKERS,
            )
            if visual_marker is not None:
                known_conditions = (
                    analysis.get("known_conditions")
                    if isinstance(analysis, Mapping)
                    else None
                )
                has_visible_information = (
                    isinstance(known_conditions, list)
                    and any(
                        isinstance(item, str)
                        and item.strip().startswith(
                            VISIBLE_INFORMATION_PREFIX
                        )
                        for item in known_conditions
                    )
                )
                if not has_visible_information:
                    errors.append(
                        "$.analysis.known_conditions: normalized_question "
                        f"包含视觉依赖标记“{visual_marker}”时，至少一项必须以"
                        f"“{VISIBLE_INFORMATION_PREFIX}”开头"
                    )
            method = (
                analysis.get("method")
                if isinstance(analysis, Mapping)
                else None
            )
            exam_marker = first_marker(
                method,
                EXAM_RULE_GUESS_MARKERS,
            )
            if exam_marker is not None:
                errors.append(
                    "$.analysis.method: 禁止根据考试规则、命题意图或"
                    f"答案习惯猜测，发现标记“{exam_marker}”"
                )
    if "solution" in required_roots:
        require_text(solution, "verification", "$.solution.verification")
        steps = solution.get("steps") if isinstance(solution, Mapping) else None
        if not isinstance(steps, list) or not steps:
            errors.append("$.solution.steps: 必须是非空字符串数组")
        elif any(not isinstance(item, str) or not item.strip() for item in steps):
            errors.append("$.solution.steps: 每一项必须是非空字符串")
        if isinstance(steps, list):
            for index, step in enumerate(steps):
                marker = first_marker(step, SOLUTION_SELF_TALK_MARKERS)
                if marker is not None:
                    errors.append(
                        f"$.solution.steps[{index}]: 禁止自言自语标记"
                        f"“{marker}”"
                    )
                exam_marker = first_marker(
                    step,
                    EXAM_RULE_GUESS_MARKERS,
                )
                if exam_marker is not None:
                    errors.append(
                        f"$.solution.steps[{index}]: 禁止根据考试规则、"
                        "命题意图或干扰项设置猜答案，发现标记"
                        f"“{exam_marker}”"
                    )
        verification = (
            solution.get("verification")
            if isinstance(solution, Mapping)
            else None
        )
        verification_marker = first_marker(
            verification,
            VERIFICATION_DISALLOWED_MARKERS,
        )
        if verification_marker is not None:
            errors.append(
                "$.solution.verification: 禁止反问、自我纠错或试探性推理标记"
                f"“{verification_marker}”"
            )
        verification_exam_marker = first_marker(
            verification,
            EXAM_RULE_GUESS_MARKERS,
        )
        if verification_exam_marker is not None:
            errors.append(
                "$.solution.verification: 禁止根据考试规则、命题意图"
                "或标准答案习惯取舍选项，发现标记"
                f"“{verification_exam_marker}”"
            )
        if (
            isinstance(verification, str)
            and len(verification.strip()) > VERIFICATION_MAX_CHARACTERS
        ):
            errors.append(
                "$.solution.verification: 必须是简洁的独立复核，"
                f"不得超过{VERIFICATION_MAX_CHARACTERS}字符；"
                "不得在验证阶段重新解题或修正答案"
            )
    if "final_answer" in required_roots:
        require_text(final_answer, "answer", "$.final_answer.answer")
        answer = (
            final_answer.get("answer")
            if isinstance(final_answer, Mapping)
            else None
        )
        guess_marker = first_marker(answer, FINAL_ANSWER_GUESS_MARKERS)
        if guess_marker is not None:
            errors.append(
                "$.final_answer.answer: 禁止用猜测语言"
                f"“{guess_marker}”给出候选答案；无法可靠作答时必须转补拍或人工复核"
            )
        answer_exam_marker = first_marker(
            answer,
            EXAM_RULE_GUESS_MARKERS,
        )
        if answer_exam_marker is not None:
            errors.append(
                "$.final_answer.answer: 禁止用考试规则、命题意图或"
                f"标准答案习惯猜答案，发现标记“{answer_exam_marker}”"
            )
    if "answer" in required_roots:
        answer = value.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            errors.append("$.answer: 必须是非空字符串")
        guess_marker = first_marker(answer, FINAL_ANSWER_GUESS_MARKERS)
        if guess_marker is not None:
            errors.append(
                "$.answer: 禁止用猜测语言"
                f"“{guess_marker}”给出候选答案；无法可靠作答时必须要求补拍"
            )
        answer_exam_marker = first_marker(
            answer,
            EXAM_RULE_GUESS_MARKERS,
        )
        if answer_exam_marker is not None:
            errors.append(
                "$.answer: 禁止用考试规则、命题意图或"
                f"标准答案习惯猜答案，发现标记“{answer_exam_marker}”"
            )
    if tuple(required_roots) == ANSWER_FIRST_ACCURACY_ROOT_KEYS:
        require_text(final, "answer", "$.final.answer")
        if isinstance(final, Mapping) and next(iter(final), None) != "answer":
            errors.append("$.final: 第一字段必须是answer")
        final_value = final.get("answer") if isinstance(final, Mapping) else None
        final_guess_marker = first_marker(
            final_value,
            FINAL_ANSWER_GUESS_MARKERS,
        )
        if final_guess_marker is not None:
            errors.append(
                "$.final.answer: 禁止用猜测语言"
                f"“{final_guess_marker}”给出候选答案"
            )
        final_exam_marker = first_marker(final_value, EXAM_RULE_GUESS_MARKERS)
        if final_exam_marker is not None:
            errors.append(
                "$.final.answer: 禁止用考试规则、命题意图或"
                f"标准答案习惯猜答案，发现标记“{final_exam_marker}”"
            )

        require_text(classification, "subject", "$.classification.subject")
        steps = analysis.get("steps") if isinstance(analysis, Mapping) else None
        if not isinstance(steps, list) or not 3 <= len(steps) <= 6:
            errors.append("$.analysis.steps: 必须包含3至6个关键步骤")
        elif any(
            not isinstance(step, str) or not step.strip()
            for step in steps
        ):
            errors.append("$.analysis.steps: 每一步必须是非空字符串")
        else:
            for index, step in enumerate(steps):
                marker = first_marker(step, SOLUTION_SELF_TALK_MARKERS)
                if marker is not None:
                    errors.append(
                        f"$.analysis.steps[{index}]: 禁止自言自语标记"
                        f"“{marker}”"
                    )
                exam_marker = first_marker(step, EXAM_RULE_GUESS_MARKERS)
                if exam_marker is not None:
                    errors.append(
                        f"$.analysis.steps[{index}]: 禁止按考试规则猜答案，"
                        f"发现标记“{exam_marker}”"
                    )
        require_text(verification, "method", "$.verification.method")
        require_text(verification, "result", "$.verification.result")
        verification_result = (
            verification.get("result")
            if isinstance(verification, Mapping)
            else None
        )
        verification_marker = first_marker(
            verification_result,
            VERIFICATION_DISALLOWED_MARKERS,
        )
        if verification_marker is not None:
            errors.append(
                "$.verification.result: 禁止反问、自我纠错或"
                f"试探性推理标记“{verification_marker}”"
            )
    return errors


def apply_schema_safe_output_normalization(
    value: Any,
    schema: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Apply only lossless, auditable repairs to otherwise usable JSON.

    The raw response remains in ``attempt.output_text``. This policy may move a
    uniquely misplaced verification string into ``solution`` or split an
    overlong solution step at complete sentence boundaries. It never deletes
    formulas, derivations, or answer text merely to satisfy a schema cap.
    """
    audit: dict[str, Any] = {
        "changed": False,
        "items": [],
        "policy": "lossless_structural_normalization_v1",
    }
    if not isinstance(value, dict):
        return value, audit

    expected_roots = schema_root_keys(schema)
    if (
        expected_roots == ("analysis", "answer")
        and set(value) == {"analysis"}
        and isinstance(value.get("analysis"), dict)
        and isinstance(value["analysis"].get("answer"), str)
        and value["analysis"]["answer"].strip()
    ):
        nested_answer = value["analysis"].pop("answer")
        value["answer"] = nested_answer
        audit["changed"] = True
        audit["items"].append({
            "path": "$.analysis.answer",
            "destination": "$.answer",
            "action": "move_unique_misplaced_field",
            "content_sha256": sha256_bytes(
                nested_answer.encode("utf-8")
            ),
        })

    if (
        expected_roots == ("analysis", "answer")
        and set(value) == {"analysis", "answer"}
        and isinstance(value.get("analysis"), dict)
        and isinstance(value["analysis"].get("answer"), str)
        and isinstance(value.get("answer"), str)
    ):
        nested_answer = value["analysis"]["answer"].strip()
        root_answer = value["answer"].strip()
        redundant = (
            nested_answer == root_answer
            or (
                re.fullmatch(r"[A-D]", nested_answer, re.IGNORECASE)
                and re.match(
                    rf"^{re.escape(nested_answer)}(?:\b|[.．、:：])",
                    root_answer,
                    re.IGNORECASE,
                )
                is not None
            )
        )
        if redundant:
            value["analysis"].pop("answer")
            audit["changed"] = True
            audit["items"].append({
                "path": "$.analysis.answer",
                "destination": "$.answer",
                "action": "remove_redundant_duplicate_field",
                "content_sha256": sha256_bytes(
                    nested_answer.encode("utf-8")
                ),
            })

    solution = value.get("solution")
    expected_with_misplaced_verification = (
        set(expected_roots) | {"verification"}
    )
    if (
        set(value) == expected_with_misplaced_verification
        and isinstance(solution, dict)
        and "verification" not in solution
        and isinstance(value.get("verification"), str)
        and value["verification"].strip()
    ):
        verification = value.pop("verification")
        solution["verification"] = verification
        audit["changed"] = True
        audit["items"].append({
            "path": "$.verification",
            "destination": "$.solution.verification",
            "action": "move_unique_misplaced_field",
            "content_sha256": sha256_bytes(
                verification.encode("utf-8")
            ),
        })

    solution = value.get("solution")
    if not isinstance(solution, dict):
        return value, audit
    steps = solution.get("steps")
    if (
        not isinstance(steps, list)
        or not steps
        or not all(isinstance(item, str) and item.strip() for item in steps)
    ):
        return value, audit

    try:
        step_schema = schema["properties"]["solution"]["properties"]["steps"]
        maximum_items = int(step_schema["maxItems"])
        maximum_chars = int(step_schema["items"]["maxLength"])
    except (KeyError, TypeError, ValueError):
        return value, audit
    if maximum_items < 1 or maximum_chars < 1:
        return value, audit
    if len(steps) > maximum_items:
        return value, audit

    normalized_steps: list[str] = []
    split_items: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if len(step) <= maximum_chars:
            normalized_steps.append(step)
            continue
        sentence_parts = [
            part
            for part in re.split(r"(?<=[。！？；])", step)
            if part
        ]
        if (
            len(sentence_parts) < 2
            or any(len(part) > maximum_chars for part in sentence_parts)
        ):
            return value, audit
        groups: list[str] = []
        current = ""
        for part in sentence_parts:
            if not current or len(current) + len(part) <= maximum_chars:
                current += part
            else:
                groups.append(current)
                current = part
        if current:
            groups.append(current)
        if (
            len(groups) < 2
            or any(
                not group.strip() or len(group) > maximum_chars
                for group in groups
            )
        ):
            return value, audit
        normalized_steps.extend(groups)
        split_items.append({
            "path": f"$.solution.steps[{index}]",
            "action": "split_on_complete_sentence_boundaries",
            "original_chars": len(step),
            "output_count": len(groups),
            "content_sha256": sha256_bytes(step.encode("utf-8")),
        })

    if split_items and len(normalized_steps) <= maximum_items:
        solution["steps"] = normalized_steps
        audit["changed"] = True
        audit["items"].extend(split_items)
    return value, audit


def unwrap_single_json_code_fence(text: str) -> tuple[str, dict[str, Any]]:
    """Losslessly unwrap one whole-response JSON Markdown fence.

    Some otherwise valid model responses wrap the JSON object in exactly one
    `````json`` fence. Accepting that wrapper avoids paying for a complete
    semantic re-solve. Extra prose, multiple fences, or trailing content are
    deliberately rejected.
    """

    audit: dict[str, Any] = {
        "changed": False,
        "items": [],
        "policy": "single_whole_response_json_code_fence_v1",
    }
    match = re.fullmatch(
        r"\s*```(?:json|JSON)?\s*\r?\n?([\s\S]*?)\r?\n?```\s*",
        text,
    )
    if match is None:
        return text, audit
    inner = match.group(1).strip()
    if not inner.startswith("{") or not inner.endswith("}"):
        return text, audit
    audit["changed"] = True
    audit["items"].append({
        "path": "$",
        "action": "unwrap_single_json_code_fence",
        "content_sha256": sha256_bytes(inner.encode("utf-8")),
    })
    return inner, audit


def repair_invalid_json_string_escapes(
    text: str,
) -> tuple[str, dict[str, Any]]:
    r"""Double only backslashes that are illegal inside JSON strings.

    A single backslash before a LaTeX command such as ``\sin`` is invalid JSON
    transport syntax. Doubling only those unambiguously illegal backslashes is
    lossless with respect to the intended string and avoids a semantic re-solve.
    Text outside quoted JSON strings is never changed.
    """
    audit: dict[str, Any] = {
        "changed": False,
        "count": 0,
        "items": [],
        "policy": "illegal_json_string_backslash_escape_only_v1",
    }
    output: list[str] = []
    in_string = False
    index = 0
    valid_simple = {'"', "\\", "/", "b", "f", "n", "r", "t"}
    hexdigits = set("0123456789abcdefABCDEF")
    while index < len(text):
        character = text[index]
        if character == '"':
            in_string = not in_string
            output.append(character)
            index += 1
            continue
        if in_string and character == "\\":
            following = text[index + 1] if index + 1 < len(text) else ""
            unicode_valid = (
                following == "u"
                and index + 5 < len(text)
                and all(
                    value in hexdigits
                    for value in text[index + 2:index + 6]
                )
            )
            if following in valid_simple or unicode_valid:
                output.append(character)
                if following:
                    output.append(following)
                    index += 2
                else:
                    index += 1
                continue
            output.append("\\\\")
            audit["changed"] = True
            audit["count"] += 1
            if len(audit["items"]) < 50:
                audit["items"].append(
                    {
                        "path": "$",
                        "action": "double_illegal_json_string_backslash",
                        "offset": index,
                        "following_character": following,
                    }
                )
            index += 1
            continue
        output.append(character)
        index += 1
    return "".join(output), audit


def parse_and_validate_output(
    text: str, schema: Mapping[str, Any], expected_subject: str
) -> dict[str, Any]:
    expected_root_keys = schema_root_keys(schema)
    parsed: Any = None
    parse_error: str | None = None
    latex_control_repair: dict[str, Any] = {
        "count": 0,
        "items": [],
        "truncated": False,
        "policy": "allowlisted_latex_commands_inside_dollar_math_only",
    }
    output_normalization: dict[str, Any] = {
        "changed": False,
        "items": [],
        "policy": "lossless_structural_normalization_v1",
    }
    json_wrapper_normalization: dict[str, Any] = {
        "changed": False,
        "items": [],
        "policy": "single_whole_response_json_code_fence_v1",
    }
    json_escape_normalization: dict[str, Any] = {
        "changed": False,
        "count": 0,
        "items": [],
        "policy": "illegal_json_string_backslash_escape_only_v1",
    }
    parse_text = text.strip()
    try:
        parsed = json.loads(parse_text)
        parsed, latex_control_repair = repair_decoded_json_latex_controls(
            parsed
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        unwrapped, json_wrapper_normalization = unwrap_single_json_code_fence(
            text
        )
        repair_source = (
            unwrapped
            if json_wrapper_normalization["changed"]
            else parse_text
        )
        repaired, json_escape_normalization = (
            repair_invalid_json_string_escapes(repair_source)
        )
        if json_escape_normalization["changed"]:
            try:
                parsed = json.loads(repaired)
                parsed, latex_control_repair = repair_decoded_json_latex_controls(
                    parsed
                )
            except (json.JSONDecodeError, UnicodeDecodeError) as inner_exc:
                parse_error = f"{type(inner_exc).__name__}: {inner_exc}"
        elif json_wrapper_normalization["changed"]:
            try:
                parsed = json.loads(unwrapped)
                parsed, latex_control_repair = repair_decoded_json_latex_controls(
                    parsed
                )
            except (json.JSONDecodeError, UnicodeDecodeError) as inner_exc:
                parse_error = f"{type(inner_exc).__name__}: {inner_exc}"
        else:
            parse_error = f"{type(exc).__name__}: {exc}"

    raw_root_keys: list[str] = (
        list(parsed) if isinstance(parsed, dict) else []
    )
    raw_root_keys_exact = tuple(raw_root_keys) == expected_root_keys
    raw_errors: list[str] = []
    if parse_error:
        raw_errors.append(f"$: JSON解析失败：{parse_error}")
    elif not isinstance(parsed, dict):
        raw_errors.append("$: 根值必须是对象")
    else:
        if json_wrapper_normalization["changed"]:
            raw_errors.append("$: 原始响应包含单个JSON Markdown代码围栏")
        if json_escape_normalization["changed"]:
            raw_errors.append(
                "$: original response contained illegal JSON string escapes"
            )
        if not raw_root_keys_exact:
            raw_errors.append(
                "$: 根字段及顺序必须严格为"
                + ",".join(expected_root_keys)
            )
        raw_errors.extend(schema_errors(parsed, schema))
        parsed, output_normalization = (
            apply_schema_safe_output_normalization(parsed, schema)
        )
        transport_items: list[dict[str, Any]] = []
        if json_wrapper_normalization["changed"]:
            transport_items.extend(json_wrapper_normalization["items"])
        if json_escape_normalization["changed"]:
            transport_items.extend(json_escape_normalization["items"])
        if transport_items:
            output_normalization["changed"] = True
            output_normalization["items"] = (
                list(transport_items)
                + list(output_normalization.get("items", []))
            )
            output_normalization["policy"] = (
                "lossless_json_transport_then_structural_normalization_v1"
            )

    root_keys: list[str] = list(parsed) if isinstance(parsed, dict) else []
    root_keys_exact = tuple(root_keys) == expected_root_keys
    errors: list[str] = []
    if parse_error:
        errors.append(f"$: JSON解析失败：{parse_error}")
    elif not isinstance(parsed, dict):
        errors.append("$: 根值必须是对象")
    else:
        if not root_keys_exact:
            errors.append(
                "$: 根字段及顺序必须严格为"
                + ",".join(expected_root_keys)
            )
        errors.extend(schema_errors(parsed, schema))

    reported_subject = None
    normalized_subject = None
    subject_match = None
    reported_grade = None
    grade_scope = "unknown"
    content_errors = (
        semantic_content_errors(parsed, expected_root_keys)
        if isinstance(parsed, dict)
        else []
    )
    if isinstance(parsed, dict):
        classification = parsed.get("classification")
        if isinstance(classification, dict):
            reported_subject = classification.get("subject")
            reported_grade = classification.get("grade")
            grade_scope = classify_grade_scope(reported_grade)
            normalized_subject = normalize_subject(reported_subject)
            if expected_subject in {"mathematics", "mathematics"}:
                subject_match = normalized_subject == expected_subject
        else:
            analysis = parsed.get("analysis")
            if isinstance(analysis, dict):
                reported_subject = analysis.get("subject")
                normalized_subject = normalize_subject(reported_subject)
                if expected_subject in {"mathematics", "mathematics"}:
                    subject_match = normalized_subject == expected_subject
            else:
                question = parsed.get("question")
                if isinstance(question, dict):
                    reported_subject = question.get("subject")
                    normalized_subject = normalize_subject(reported_subject)
                    if expected_subject in {"mathematics", "mathematics"}:
                        subject_match = normalized_subject == expected_subject

    return {
        "json_parse_success": parse_error is None,
        "raw_strict_schema_valid": not raw_errors,
        "strict_schema_valid": not errors,
        "content_complete": not content_errors,
        "usable_output": not errors and not content_errors,
        "root_keys": root_keys,
        "expected_root_keys": list(expected_root_keys),
        "root_keys_exact_and_ordered": root_keys_exact,
        "errors": errors[:50],
        "raw_schema_errors": raw_errors[:50],
        "content_errors": content_errors[:50],
        "reported_subject": reported_subject,
        "normalized_reported_subject": normalized_subject,
        "expected_subject": expected_subject,
        "subject_match": subject_match,
        "reported_grade": reported_grade,
        "grade_scope": grade_scope,
        "latex_control_repair": latex_control_repair,
        "output_normalization": output_normalization,
        "json_wrapper_normalization": json_wrapper_normalization,
        "json_escape_normalization": json_escape_normalization,
        "parsed": parsed if parse_error is None else None,
    }


def normalize_subject(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in SUBJECT_ALIASES:
        return SUBJECT_ALIASES[text]
    if "数学" in text or "mathematic" in text:
        return "mathematics"
    if "数学" in text or "mathematics" in text:
        return "mathematics"
    return None


def classify_grade_scope(value: Any) -> str:
    text = str(value or "").strip()
    if text in HIGH_SCHOOL_GRADES:
        return "high_school"
    if text in OUT_OF_SCOPE_GRADES:
        return "out_of_scope"
    return "unknown"


def extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "".join(extract_text(item) for item in value)
    if isinstance(value, Mapping):
        for key in ("text", "content", "value"):
            if key in value:
                return extract_text(value[key])
    return ""


def update_milestones(
    accumulated: str, elapsed_ms: float, milestones: dict[str, float | None]
) -> None:
    for key in milestones:
        if key == FINAL_ANSWER_COMPLETE_MILESTONE:
            answer = completed_first_root_field(
                accumulated,
                "final",
                "answer",
            )
            if (
                milestones[key] is None
                and isinstance(answer, str)
                and answer.strip()
            ):
                milestones[key] = rounded_ms(elapsed_ms)
            continue
        if (
            milestones[key] is None
            and milestone_pattern(key).search(accumulated)
        ):
            milestones[key] = rounded_ms(elapsed_ms)


def normalize_usage(raw: Any) -> dict[str, Any]:
    value = raw if isinstance(raw, Mapping) else {}

    def integer(*candidates: Any) -> int | None:
        for candidate in candidates:
            if isinstance(candidate, bool):
                continue
            if isinstance(candidate, (int, float)) and candidate >= 0:
                return int(candidate)
        return None

    prompt_details = value.get("prompt_tokens_details")
    if not isinstance(prompt_details, Mapping):
        prompt_details = value.get("input_tokens_details")
    if not isinstance(prompt_details, Mapping):
        prompt_details = {}
    completion_details = value.get("completion_tokens_details")
    if not isinstance(completion_details, Mapping):
        completion_details = value.get("output_tokens_details")
    if not isinstance(completion_details, Mapping):
        completion_details = {}

    input_tokens = integer(value.get("prompt_tokens"), value.get("input_tokens"))
    output_tokens = integer(
        value.get("completion_tokens"), value.get("output_tokens")
    )
    total_tokens = integer(value.get("total_tokens"))
    total_tokens_source = "provider" if total_tokens is not None else "missing"
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
        total_tokens_source = "derived_input_plus_output"
    cached_tokens = integer(
        prompt_details.get("cached_tokens"),
        prompt_details.get("cached_input_tokens"),
        value.get("cached_tokens"),
    )
    reasoning_tokens = integer(
        completion_details.get("reasoning_tokens"),
        value.get("reasoning_tokens"),
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": cached_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "total_tokens_source": total_tokens_source,
        "reported": bool(value),
    }


def combine_usage(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    material = list(items)

    def summed(key: str) -> int | None:
        values = [item.get(key) for item in material]
        known = [value for value in values if isinstance(value, int)]
        return sum(known) if known else None

    return {
        "input_tokens": summed("input_tokens"),
        "output_tokens": summed("output_tokens"),
        "total_tokens": summed("total_tokens"),
        "cached_input_tokens": summed("cached_input_tokens"),
        "reasoning_output_tokens": summed("reasoning_output_tokens"),
        "attempts_with_usage": sum(1 for item in material if item.get("reported")),
        "attempts_total": len(material),
        "complete": bool(material) and all(item.get("reported") for item in material),
    }


def first_number(mapping: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def estimate_cost(
    usage: Mapping[str, Any], pricing_value: Any, price_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    pricing = pricing_value if isinstance(pricing_value, Mapping) else {}
    currency = str(
        pricing.get("currency") or price_snapshot.get("currency") or "CNY"
    )
    input_rate = first_number(
        pricing,
        (
            "input_cny_per_million", "input_per_million",
            "input_price_per_million", "input_price_per_1m",
        ),
    )
    output_rate = first_number(
        pricing,
        (
            "output_cny_per_million", "output_per_million",
            "output_price_per_million", "output_price_per_1m",
        ),
    )
    cached_rate = first_number(
        pricing,
        (
            "cached_input_cny_per_million", "cached_input_per_million",
            "cached_input_price_per_million",
        ),
    )
    reasoning_rate = first_number(
        pricing,
        (
            "reasoning_output_cny_per_million", "reasoning_output_per_million",
            "reasoning_output_price_per_million",
        ),
    )
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cached = usage.get("cached_input_tokens")
    reasoning = usage.get("reasoning_output_tokens")
    amount: float | None = None
    if (
        isinstance(input_tokens, int) and isinstance(output_tokens, int)
        and input_rate is not None and output_rate is not None
    ):
        cached_known = min(input_tokens, cached) if isinstance(cached, int) else 0
        reasoning_known = (
            min(output_tokens, reasoning) if isinstance(reasoning, int) else 0
        )
        noncached = input_tokens - cached_known
        nonreasoning = output_tokens - reasoning_known
        amount = (
            noncached * input_rate
            + cached_known * (cached_rate if cached_rate is not None else input_rate)
            + nonreasoning * output_rate
            + reasoning_known * (
                reasoning_rate if reasoning_rate is not None else output_rate
            )
        ) / 1_000_000.0
    return {
        "amount": None if amount is None else round(amount, 10),
        "currency": currency,
        "estimated": True,
        "input_rate_per_million": input_rate,
        "output_rate_per_million": output_rate,
        "cached_input_rate_per_million": cached_rate,
        "reasoning_output_rate_per_million": reasoning_rate,
        "price_snapshot_id": price_snapshot.get("snapshot_id"),
    }


def combine_cost(items: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    material = list(items)
    amounts = [item.get("amount") for item in material]
    known = [float(value) for value in amounts if isinstance(value, (int, float))]
    currencies = {
        str(item.get("currency")) for item in material if item.get("currency")
    }
    return {
        "amount": round(sum(known), 10) if known else None,
        "currency": next(iter(currencies)) if len(currencies) == 1 else None,
        "complete": bool(material) and len(known) == len(material),
        "estimated": True,
    }


def safe_headers(headers: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if headers is None:
        return result
    for key in SAFE_RESPONSE_HEADERS:
        value = headers.get(key)
        if value:
            result[key] = str(value)[:512]
    return result


def response_request_id(headers: Mapping[str, str]) -> str | None:
    for key in ("x-request-id", "x-dashscope-request-id", "request-id"):
        if headers.get(key):
            return headers[key]
    return None


def parse_sse_response(
    response: Any,
    network_started_ms: float,
    timeout_seconds: float,
    root_keys: Sequence[str] = ROOT_KEYS,
) -> dict[str, Any]:
    content_parts: list[str] = []
    reasoning_chars = 0
    usage_raw: Any = {}
    finish_reason = None
    response_model = None
    event_count = 0
    parse_warnings: list[str] = []
    body_bytes = 0
    first_event_ms = None
    first_content_ms = None
    first_reasoning_ms = None
    milestones = milestone_state(root_keys)
    data_lines: list[str] = []
    done = False
    deadline_ms = network_started_ms + max(0.001, timeout_seconds) * 1000

    line_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def read_stream() -> None:
        try:
            while True:
                raw_line = response.readline()
                line_queue.put(("line", raw_line))
                if not raw_line:
                    return
        except BaseException as exc:
            line_queue.put(("error", exc))

    threading.Thread(
        target=read_stream,
        name="benchmark-sse-reader",
        daemon=True,
    ).start()

    def consume_event() -> None:
        nonlocal usage_raw, finish_reason, response_model, event_count
        nonlocal first_event_ms, first_content_ms, first_reasoning_ms
        nonlocal reasoning_chars, done
        if not data_lines:
            return
        data = "\n".join(data_lines).strip()
        data_lines.clear()
        if not data:
            return
        elapsed = monotonic_ms() - network_started_ms
        if first_event_ms is None:
            first_event_ms = rounded_ms(elapsed)
        if data == "[DONE]":
            done = True
            return
        try:
            envelope = json.loads(data)
        except json.JSONDecodeError as exc:
            parse_warnings.append(f"SSE事件JSON无效：{exc.msg}")
            return
        event_count += 1
        if isinstance(envelope, Mapping):
            if envelope.get("usage"):
                usage_raw = envelope["usage"]
            if envelope.get("model"):
                response_model = envelope["model"]
            if envelope.get("error"):
                parse_warnings.append(
                    "SSE错误事件：" + extract_text(envelope.get("error"))[:500]
                )
            choices = envelope.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0] if isinstance(choices[0], Mapping) else {}
                if choice.get("finish_reason") is not None:
                    finish_reason = choice.get("finish_reason")
                message = choice.get("delta")
                if not isinstance(message, Mapping):
                    message = choice.get("message")
                if not isinstance(message, Mapping):
                    message = {}
                reasoning = extract_text(
                    message.get("reasoning_content")
                    if "reasoning_content" in message else message.get("reasoning")
                )
                if reasoning:
                    reasoning_chars += len(reasoning)
                    if first_reasoning_ms is None:
                        first_reasoning_ms = rounded_ms(elapsed)
                content = extract_text(message.get("content"))
                if content:
                    content_parts.append(content)
                    if first_content_ms is None:
                        first_content_ms = rounded_ms(elapsed)
                    accumulated = "".join(content_parts)
                    update_milestones(accumulated, elapsed, milestones)
                    callback = getattr(_STREAM_CONTENT_CALLBACK, "callback", None)
                    if callback is not None:
                        try:
                            if callback(accumulated, rounded_ms(elapsed)) is False:
                                finish_reason = "local_output_quality_stop"
                                done = True
                        except Exception as exc:
                            parse_warnings.append(
                                "流式内容回调失败：" + type(exc).__name__
                            )

    while not done:
        remaining_ms = deadline_ms - monotonic_ms()
        if remaining_ms <= 0:
            raise TimeoutError(
                f"SSE响应超过总时限{timeout_seconds:g}秒"
            )
        try:
            event_type, payload = line_queue.get(
                timeout=max(0.001, remaining_ms / 1000.0)
            )
        except queue.Empty as exc:
            raise TimeoutError(
                f"SSE响应超过总时限{timeout_seconds:g}秒"
            ) from exc
        if event_type == "error":
            raise payload
        raw = payload
        if monotonic_ms() > deadline_ms:
            raise TimeoutError(
                f"SSE响应超过总时限{timeout_seconds:g}秒"
            )
        if not raw:
            consume_event()
            break
        body_bytes += len(raw)
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            consume_event()
        elif line.startswith(":"):
            continue
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    return {
        "output_text": "".join(content_parts),
        "reasoning_chars": reasoning_chars,
        "usage_raw": usage_raw,
        "finish_reason": finish_reason,
        "response_model": response_model,
        "event_count": event_count,
        "parse_warnings": parse_warnings,
        "response_body_bytes": body_bytes,
        "first_event_ms": first_event_ms,
        "first_content_ms": first_content_ms,
        "first_reasoning_ms": first_reasoning_ms,
        "milestones_ms": milestones,
        "stream_complete": done,
    }


def parse_json_response(
    response: Any,
    network_started_ms: float,
    root_keys: Sequence[str] = ROOT_KEYS,
) -> dict[str, Any]:
    raw = response.read()
    elapsed = monotonic_ms() - network_started_ms
    envelope = json.loads(raw.decode("utf-8-sig"))
    choices = envelope.get("choices") if isinstance(envelope, Mapping) else None
    choice = choices[0] if isinstance(choices, list) and choices else {}
    if not isinstance(choice, Mapping):
        choice = {}
    message = choice.get("message")
    if not isinstance(message, Mapping):
        message = {}
    output_text = extract_text(message.get("content"))
    reasoning = extract_text(
        message.get("reasoning_content")
        if "reasoning_content" in message else message.get("reasoning")
    )
    milestones = milestone_state(root_keys)
    update_milestones(output_text, elapsed, milestones)
    return {
        "output_text": output_text,
        "reasoning_chars": len(reasoning),
        "usage_raw": envelope.get("usage", {}) if isinstance(envelope, Mapping) else {},
        "finish_reason": choice.get("finish_reason"),
        "response_model": envelope.get("model") if isinstance(envelope, Mapping) else None,
        "event_count": 1,
        "parse_warnings": [],
        "response_body_bytes": len(raw),
        "first_event_ms": rounded_ms(elapsed),
        "first_content_ms": rounded_ms(elapsed) if output_text else None,
        "first_reasoning_ms": rounded_ms(elapsed) if reasoning else None,
        "milestones_ms": milestones,
        "stream_complete": True,
    }


def endpoint_url(base_url: str) -> str:
    clean = base_url.rstrip("/")
    if clean.endswith("/chat/completions"):
        return clean
    return clean + "/chat/completions"


def is_dashscope_base_url(base_url: str) -> bool:
    """Return whether *base_url* is a DashScope API endpoint."""
    try:
        hostname = (urllib.parse.urlsplit(base_url).hostname or "").lower()
    except ValueError:
        return False
    return hostname == "dashscope.aliyuncs.com" or hostname.endswith(
        ".dashscope.aliyuncs.com"
    )


def perform_http_attempt(
    payload: Mapping[str, Any], base_url: str, api_key: str,
    workspace_id: str | None, timeout_seconds: float, secrets: Sequence[str],
    root_keys: Sequence[str] = ROOT_KEYS,
    resolve_dashscope_oss: bool = False,
) -> dict[str, Any]:
    attempt_started = monotonic_ms()
    build_started = monotonic_ms()
    body = json_bytes(payload)
    request_build_ms = monotonic_ms() - build_started
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
        "User-Agent": "yuyi-full-benchmark/1.0",
    }
    if workspace_id and is_dashscope_base_url(base_url):
        headers["X-DashScope-Workspace"] = workspace_id
    if resolve_dashscope_oss:
        headers["X-DashScope-OssResourceResolve"] = "enable"
    request = urllib.request.Request(
        endpoint_url(base_url), data=body, headers=headers, method="POST"
    )
    network_started = monotonic_ms()
    result: dict[str, Any] = {
        "request_build_ms": rounded_ms(request_build_ms),
        "request_body_bytes": len(body),
        "response_headers_wait_ms": None,
        "first_event_ms": None,
        "first_content_ms": None,
        "first_reasoning_ms": None,
        "provider_total_ms": None,
        "attempt_wall_ms": None,
        "http_status": None,
        "response_headers": {},
        "provider_request_id": None,
        "provider_error_code": None,
        "provider_error_type": None,
        "transport_success": False,
        "retryable": True,
        "error": None,
        "output_text": "",
        "output_sha256": None,
        "reasoning_chars": 0,
        "usage": normalize_usage({}),
        "finish_reason": None,
        "response_model": None,
        "event_count": 0,
        "parse_warnings": [],
        "response_body_bytes": 0,
        "milestones_ms": milestone_state(root_keys),
        "stream_complete": None,
    }
    try:
        # urllib applies this value as an idle socket timeout. Keep it below
        # the overall request deadline so a silent stream cannot block forever.
        io_timeout_seconds = min(max(0.001, timeout_seconds), 60.0)
        with urllib.request.urlopen(
            request, timeout=io_timeout_seconds
        ) as response:
            result["response_headers_wait_ms"] = rounded_ms(
                monotonic_ms() - network_started
            )
            result["http_status"] = getattr(response, "status", response.getcode())
            safe = safe_headers(response.headers)
            result["response_headers"] = safe
            result["provider_request_id"] = response_request_id(safe)
            content_type = str(response.headers.get("Content-Type", "")).lower()
            if "text/event-stream" in content_type:
                parsed = parse_sse_response(
                    response,
                    network_started,
                    timeout_seconds,
                    root_keys,
                )
            else:
                parsed = parse_json_response(
                    response,
                    network_started,
                    root_keys,
                )
            result.update(parsed)
            result["output_text"] = redact_text(
                result["output_text"], secrets, limit=2_000_000
            )
            result["output_sha256"] = sha256_bytes(
                result["output_text"].encode("utf-8")
            )
            result["usage"] = normalize_usage(parsed.get("usage_raw"))
            result.pop("usage_raw", None)
            if parsed.get("stream_complete") is False:
                result["error"] = {
                    "type": "incomplete_stream",
                    "message": (
                        "SSE response ended before the [DONE] event; "
                        "partial output is retained for audit only"
                    ),
                }
                result["retryable"] = True
            elif result.get("finish_reason") in {
                "length", "max_tokens", "max_completion_tokens"
            }:
                result["error"] = {
                    "type": "truncated_response",
                    "message": (
                        "provider stopped at the configured output-token "
                        f"limit ({result.get('finish_reason')})"
                    ),
                }
                result["retryable"] = True
            else:
                result["transport_success"] = True
                result["retryable"] = False
    except urllib.error.HTTPError as exc:
        result["response_headers_wait_ms"] = rounded_ms(
            monotonic_ms() - network_started
        )
        result["http_status"] = exc.code
        safe = safe_headers(exc.headers)
        result["response_headers"] = safe
        result["provider_request_id"] = response_request_id(safe)
        try:
            error_body = exc.read(8192).decode("utf-8", errors="replace")
        except Exception:
            error_body = ""
        result["response_body_bytes"] = len(error_body.encode("utf-8"))
        provider_error_code = None
        provider_error_type = None
        try:
            error_document = json.loads(error_body)
            error_value = (
                error_document.get("error")
                if isinstance(error_document, Mapping) else None
            )
            if isinstance(error_value, Mapping):
                if isinstance(error_value.get("code"), str):
                    provider_error_code = error_value["code"]
                if isinstance(error_value.get("type"), str):
                    provider_error_type = error_value["type"]
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            pass
        result["provider_error_code"] = provider_error_code
        result["provider_error_type"] = provider_error_type
        result["error"] = {
            "type": "http_error",
            "message": redact_text(
                f"HTTP {exc.code}: {error_body or exc.reason}", secrets
            ),
            "provider_error_code": provider_error_code,
            "provider_error_type": provider_error_type,
        }
        result["retryable"] = (
            exc.code in {408, 409, 425, 429} or 500 <= exc.code <= 599
        )
    except (
        urllib.error.URLError, TimeoutError, socket.timeout,
        ConnectionError, OSError
    ) as exc:
        result["error"] = {
            "type": "network_error",
            "message": redact_text(f"{type(exc).__name__}: {exc}", secrets),
        }
        result["retryable"] = True
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        result["error"] = {
            "type": "response_parse_error",
            "message": redact_text(f"{type(exc).__name__}: {exc}", secrets),
        }
        result["retryable"] = True
    finally:
        result["provider_total_ms"] = rounded_ms(
            monotonic_ms() - network_started
        )
        result["attempt_wall_ms"] = rounded_ms(
            monotonic_ms() - attempt_started
        )
    return result


def compact_recovery_prompt(
    recovery_kind: str,
    schema: Mapping[str, Any] | None = None,
) -> str:
    reason = (
        "上一次响应达到长度上限"
        if recovery_kind == "truncated"
        else "上一次响应未通过JSON结构或内容校验"
    )
    root_keys = schema_root_keys(schema)
    if root_keys == ANSWER_FIRST_ACCURACY_ROOT_KEYS:
        return "\n".join([
            f"【HSQA_K3_ANSWER_FIRST_RECOVERY_V1】{reason}，旧草稿不可继续使用。",
            "重新查看本次原题图与同源细节图，在内部重新完成识图、求解和一种独立验证；"
            "不得读取Gold、历史答案、文件名含义或其他模型答案。",
            "只输出一个完整JSON对象，根字段依次且仅为final、classification、"
            "analysis、verification；final必须是根对象第一项，answer必须是"
            "final第一项且先完整闭合。",
            '严格从以下骨架开始：{"final":{"answer":"无法可靠作答",'
            '"confidence":null,"needs_review":true,'
            '"review_reasons":["题面或推导未闭合"]},'
            '"classification":{"subject":"无法判断","grade":"高中",'
            '"course_module":"无法判断","question_type":"无法判断",'
            '"knowledge_level_1":"无法判断","knowledge_level_2":"无法判断",'
            '"knowledge_level_3":"无法判断"},'
            '"analysis":{"normalized_question":"无法可靠识别",'
            '"known_conditions":["关键题面不清"],"target":"无法判断",'
            '"key_relations":["无法判断"],"method":"需要复核",'
            '"steps":["确认题面","建立关系","得到结论"]},'
            '"verification":{"method":"无法验证",'
            '"result":"关键题面或推导未闭合"}}',
            "analysis.steps必须为3至6个关键步骤；key_relations最多3个；"
            "verification只写一种验证。选择题仍须在步骤中覆盖全部可见选项，"
            "final.answer必须与后续推导和验证严格一致。",
            "压缩说明文字而不是删除关键条件、公式、图中事实、步骤或验证；"
            "不得输出Markdown、额外文字、猜测答案或未闭合JSON。",
        ])
    if root_keys == ("question", "option_verdicts", "reasoning", "final"):
        return "\n".join([
            f"【HSQA_ACCURACY_RECOVERY_V1】{reason}，旧草稿不可继续使用。",
            "必须重新查看全部题图并从头求解；主提示词中的读图、逐项判断、"
            "独立验证和禁止猜测规则继续全部生效。",
            "只输出一个完整JSON对象，根字段依次且仅为question、"
            "option_verdicts、reasoning、final，不输出Markdown或额外文字。",
            "选择题必须逐个覆盖所有可见选项；final.answer必须等于所有且仅有"
            "verdict为true的选项集合。无法闭合时使用unknown并设置needs_review=true，"
            "禁止为了填满结构猜答案。",
            "不得省略题面转写、图中事实、OCR不确定项、推导和独立验证；"
            "JSON字符串中的LaTeX反斜杠必须双写。",
        ])
    if root_keys == ("review", "option_verdicts", "final"):
        return "\n".join([
            f"【HSQA_ACCURACY_JUDGE_RECOVERY_V1】{reason}，旧草稿不可继续使用。",
            "重新查看原题图并独立裁决，主提示词中的匿名候选审计、逐项证明和"
            "禁止按多数票猜测规则继续全部生效。",
            "只输出一个完整JSON对象，根字段依次且仅为review、"
            "option_verdicts、final，不输出Markdown或额外文字。",
            "final.answer必须与逐项判定一致；仍有关键不确定性时必须设置"
            "needs_additional_resolve=true，禁止强行给出高置信度答案。",
        ])
    if root_keys == ("analysis", "answer"):
        return "\n".join([
            f"【HSQA_FAST_RECOVERY_V1】{reason}，旧草稿不可继续使用。",
            "重新查看题图，只输出一个完整JSON对象，不要输出JSON之外的文字。",
            "根字段依次且仅为analysis、answer，严格使用以下骨架：",
            '{"analysis":{"subject":"无法判断",'
            '"question_type":"无法判断",'
            '"knowledge_points":["无法判断"],'
            '"explanation":"图片关键信息不清，需要补拍"},'
            '"answer":"图片关键信息不清，需要补拍"}',
            "analysis只能包含subject、question_type、knowledge_points、"
            "explanation；answer必须是非空字符串。",
            "不要新增基础公式、步骤或验证字段；不要猜答案。",
        ])
    return "\n".join([
        f"【HSQA_COMPACT_RECOVERY_V1】{reason}，该草稿不可继续使用。",
        "重新查看本次题图，从头只输出一个紧凑、完整、可解析的JSON对象；"
        "不要续写、引用旧草稿或输出JSON之外的文字。",
        "根字段必须依次且仅为classification、analysis、solution、final_answer。",
        "必须严格复制下面的JSON骨架并只替换字段值，不得改变字段、括号或数据类型：",
        '{"classification":{"subject":"无法判断","grade":"无法判断",'
        '"course_module":"无法判断","question_type":"无法判断",'
        '"difficulty":"无法判断","knowledge_level_1":"无法判断",'
        '"knowledge_level_2":"无法判断","knowledge_level_3":"无法判断",'
        '"confidence":null},"analysis":{"normalized_question":"无法判断",'
        '"known_conditions":["无法判断"],"target":"无法判断",'
        '"key_relations":["无法判断"],"method":"无法判断"},'
        '"solution":{"steps":["无法可靠作答"],"verification":"无法验证"},'
        '"final_answer":{"answer":"无法可靠作答","confidence":null}}',
        "classification必须是对象并且仅保留subject、grade、course_module、"
        "question_type、difficulty、knowledge_level_1、knowledge_level_2、"
        "knowledge_level_3、confidence；subject只能是数学/数学/无法判断，"
        "grade只能是小学/初中/初一/初二/初三/高中/高一/高二/高三/大学/无法判断，"
        "difficulty只能是基础/中档/高难/无法判断，confidence字段不可省略且"
        "只能是0至1数字或null。",
        "题面未明确单选或多选时，question_type必须写“选择题（单选/多选未说明）”。",
        "analysis必须是对象并且仅保留normalized_question、known_conditions、target、"
        "key_relations、method；known_conditions必须是1至8项非空字符串数组，"
        "key_relations必须是1至3项非空字符串数组，其他三个字段必须是非空字符串；"
        "normalized_question只写简要题意，不重抄完整题目。",
        "若normalized_question含“如图”“图中”“由图”“观察图”“表中”或其他"
        "图表依赖，known_conditions至少一项必须以“图中可见信息：”开头并写出"
        "实际读到的关系；看不清时要求补拍，禁止猜图。",
        "solution必须是对象并且仅保留steps和verification；steps必须是1至6项"
        "非空字符串数组，通常3至6项，每项不超过500字符且只写一个关键推导；"
        "超长推导必须按完整句拆成后续步骤，不能删除必要公式；verification必须是"
        "一个不超过300字符的非空字符串，只用一种可靠验证。",
        "final_answer必须是对象并且仅保留answer和confidence；answer必须是简洁的"
        "非空字符串，confidence字段不可省略且只能是0至1数字或null。",
        "禁止自言自语、反复检查、备选路线和猜测性讨论；不要新增、删除字段，"
        "不要把数组写成字符串或把字符串写成数组；不得用“标准答案”“干扰项”"
        "“命题意图”“通常单选/多选”等考试规则取舍选项。",
        "必要公式继续使用可渲染的教材式LaTeX并保留分式、根式、上下标、向量、"
        "概率条件、单位和方向；JSON原始字符串中的每个LaTeX反斜杠必须双写，"
        "例如\\\\frac、\\\\text、\\\\times、\\\\sqrt、\\\\vec，禁止产生"
        "\\b、\\f、\\t、\\n、\\r控制字符；无穷优先写Unicode字符∞，若使用"
        "LaTeX则必须写成\\\\infty，禁止写成\\\\fty。",
        "所有字段必须完整闭合；复杂题只保留必要推导。若无法在上述上限内保持"
        "完整推导，不得删关键条件、公式或强猜，必须明确要求补拍或人工复核。",
    ])


def build_payload(
    model: Mapping[str, Any], prompt: str, data_url: str | Sequence[str],
    request_config: Mapping[str, Any], schema: Mapping[str, Any],
    recovery_kind: str | None,
    provider_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    effective_prompt = prompt
    if recovery_kind in {"truncated", "format"}:
        recovery_prompt = compact_recovery_prompt(recovery_kind, schema)
        effective_prompt = (
            prompt + "\n\n" + recovery_prompt
            if request_config.get("preserve_primary_prompt_on_recovery") is True
            else recovery_prompt
        )
    image_urls = (
        [data_url]
        if isinstance(data_url, str)
        else [str(item) for item in data_url]
    )
    image_detail = request_config.get("image_detail")
    if image_detail is not None and image_detail not in {"low", "high"}:
        raise ValueError("image_detail must be 'low' or 'high'")
    payload: dict[str, Any] = {
        "model": model["model_id"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": effective_prompt},
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url,
                            **(
                                {"detail": image_detail}
                                if image_detail is not None
                                else {}
                            ),
                        },
                    }
                    for url in image_urls
                ],
            ]
        }],
        "stream": bool(request_config.get("stream", True)),
    }
    for key in ("temperature", "top_p", "max_tokens", "seed"):
        if key in request_config:
            payload[key] = request_config[key]
    if payload["stream"] and stream_options_include_usage_enabled(
        request_config, provider_config
    ):
        payload["stream_options"] = {"include_usage": True}

    response_format = request_config.get("response_format", "json_object")
    if isinstance(response_format, Mapping):
        payload["response_format"] = dict(response_format)
    elif response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    elif response_format == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": str(schema.get("$id") or "hsqa_output"),
                "strict": True,
                "schema": schema,
            },
        }

    request_overrides = model.get("request_overrides")
    if isinstance(request_overrides, Mapping):
        for key, value in request_overrides.items():
            if key not in {
                "model", "messages", "stream", "stream_options", "api_key",
                "authorization", "base_url",
            }:
                payload[key] = value
    extra_body = request_config.get("extra_body")
    if isinstance(extra_body, Mapping):
        for key, value in extra_body.items():
            if key not in {"model", "messages", "api_key", "authorization"}:
                payload[key] = value
    return payload


def configured_stream_options_include_usage(
    config: Mapping[str, Any] | None,
) -> bool | None:
    if not isinstance(config, Mapping):
        return None
    if "stream_options_include_usage" in config:
        return config.get("stream_options_include_usage") is not False
    stream_options = config.get("stream_options")
    if isinstance(stream_options, Mapping) and "include_usage" in stream_options:
        return stream_options.get("include_usage") is not False
    return None


def stream_options_include_usage_enabled(
    request_config: Mapping[str, Any],
    provider_config: Mapping[str, Any] | None = None,
) -> bool:
    """Allow either request or provider compatibility policy to disable it."""
    decisions = (
        configured_stream_options_include_usage(request_config),
        configured_stream_options_include_usage(provider_config),
    )
    return False not in decisions


def provider_settings(
    config: Mapping[str, Any], model: Mapping[str, Any]
) -> dict[str, Any]:
    root_provider = config.get("provider", "aliyun_bailian")
    provider_key = str(model.get("provider") or root_provider)
    providers = config.get("providers")
    provider = (
        providers.get(provider_key, {})
        if isinstance(providers, Mapping) else {}
    )
    if not isinstance(provider, Mapping):
        provider = {}
    api_key_env = str(provider.get("api_key_env") or "DASHSCOPE_API_KEY")
    base_url_env = str(provider.get("base_url_env") or "DASHSCOPE_BASE_URL")
    workspace_env = str(
        provider.get("workspace_id_env") or "DASHSCOPE_WORKSPACE_ID"
    )
    require_base_url_env = provider.get("require_base_url_env") is True
    default_base_template = (
        ""
        if require_base_url_env
        else str(provider.get("default_base_url_template") or "")
    )
    if require_base_url_env or "default_base_url" in provider:
        default_base = "" if require_base_url_env else str(
            provider.get("default_base_url") or ""
        )
    else:
        default_base = "" if default_base_template else DEFAULT_BASE_URL
    send_workspace_header = provider.get("send_workspace_header", True)
    return {
        "provider_key": provider_key,
        "api_key_env": api_key_env,
        "base_url_env": base_url_env,
        "workspace_id_env": workspace_env,
        "default_base_url": default_base,
        "default_base_url_template": default_base_template,
        "require_base_url_env": require_base_url_env,
        "send_workspace_header": bool(send_workspace_header),
        "stream_options_include_usage": (
            configured_stream_options_include_usage(provider)
        ),
    }


def image_media_type(sample: Mapping[str, Any], image_path: Path) -> str:
    declared = str(sample.get("media_type") or "").lower()
    if declared.startswith("image/"):
        return declared
    suffix = image_path.suffix.lower()
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "application/octet-stream")


def prepare_request_image(
    image_bytes: bytes,
    media_type: str,
    request_config: Mapping[str, Any],
) -> tuple[bytes, str, dict[str, Any]]:
    """Optionally resize/compress model input while preserving visual detail."""
    raw_policy = request_config.get("image_preprocess")
    policy = raw_policy if isinstance(raw_policy, Mapping) else {}
    enabled = bool(policy.get("enabled", False))
    if not enabled:
        return image_bytes, media_type, {
            "enabled": False,
            "bytes": len(image_bytes),
            "sha256": sha256_bytes(image_bytes),
            "media_type": media_type,
            "binarized": False,
        }
    if bool(policy.get("binarize", False)):
        raise ValueError("模型输入图禁止二值化")

    from PIL import Image, ImageOps

    max_long_edge = int(policy.get("max_long_edge", 1024))
    if max_long_edge < 256 or max_long_edge > 4096:
        raise ValueError("image_preprocess.max_long_edge必须在256到4096之间")
    output_format = str(policy.get("format") or "JPEG").upper()
    if output_format not in {"JPEG", "PNG", "WEBP"}:
        raise ValueError("image_preprocess.format只支持JPEG、PNG或WEBP")
    quality = int(policy.get("quality", 85))
    if quality < 30 or quality > 100:
        raise ValueError("image_preprocess.quality必须在30到100之间")

    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        image = ImageOps.exif_transpose(source)
        source_width, source_height = image.size
        crop_applied = False
        crop_box = (0, 0, source_width, source_height)
        if bool(policy.get("auto_crop_blank", False)):
            crop_luma_threshold = int(
                policy.get("crop_luma_threshold", 250)
            )
            if crop_luma_threshold < 200 or crop_luma_threshold > 254:
                raise ValueError(
                    "image_preprocess.crop_luma_threshold"
                    "必须在200到254之间"
                )
            crop_padding = int(policy.get("crop_padding", 16))
            if crop_padding < 0 or crop_padding > 256:
                raise ValueError(
                    "image_preprocess.crop_padding必须在0到256之间"
                )
            grayscale = ImageOps.grayscale(image)
            foreground = grayscale.point(
                [
                    255 if value <= crop_luma_threshold else 0
                    for value in range(256)
                ]
            )
            detected = foreground.getbbox()
            if detected is not None:
                left, top, right, bottom = detected
                candidate = (
                    max(0, left - crop_padding),
                    max(0, top - crop_padding),
                    min(source_width, right + crop_padding),
                    min(source_height, bottom + crop_padding),
                )
                candidate_area = (
                    (candidate[2] - candidate[0])
                    * (candidate[3] - candidate[1])
                )
                if candidate_area < source_width * source_height * 0.98:
                    crop_box = candidate
                    image = image.crop(crop_box)
                    crop_applied = True
        cropped_width, cropped_height = image.size
        scale = min(
            1.0,
            max_long_edge / max(cropped_width, cropped_height),
        )
        target_size = (
            max(1, int(round(cropped_width * scale))),
            max(1, int(round(cropped_height * scale))),
        )
        resized = target_size != image.size
        if resized:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        if output_format == "JPEG":
            if "A" in image.getbands() or "transparency" in image.info:
                foreground = image.convert("RGBA")
                background = Image.new("RGBA", foreground.size, "white")
                background.alpha_composite(foreground)
                image = background.convert("RGB")
            elif image.mode not in {"L", "RGB"}:
                image = image.convert("RGB")
        output = io.BytesIO()
        save_options: dict[str, Any] = {}
        if output_format in {"JPEG", "WEBP"}:
            save_options["quality"] = quality
        if output_format == "JPEG":
            save_options["optimize"] = True
        image.save(output, format=output_format, **save_options)
        prepared = output.getvalue()
        width, height = image.size

    requested_media_type = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }[output_format]
    return prepared, requested_media_type, {
        "enabled": True,
        "source_width": source_width,
        "source_height": source_height,
        "crop_applied": crop_applied,
        "crop_box_xyxy": list(crop_box),
        "cropped_width": cropped_width,
        "cropped_height": cropped_height,
        "width": width,
        "height": height,
        "resized": resized,
        "format": output_format,
        "quality": quality if output_format in {"JPEG", "WEBP"} else None,
        "bytes": len(prepared),
        "sha256": sha256_bytes(prepared),
        "media_type": requested_media_type,
        "binarized": False,
        "max_long_edge": max_long_edge,
    }


def prepare_request_image_variants(
    image_bytes: bytes,
    media_type: str,
    request_config: Mapping[str, Any],
) -> tuple[list[bytes], list[str], list[dict[str, Any]]]:
    """Return the complete primary image and optional lossless detail crops.

    The first item always follows ``prepare_request_image`` for backwards
    compatibility. Accuracy-first configurations disable preprocessing, so its
    bytes are exactly the source bytes. Detail crops are derived from the source
    image, encoded as PNG, and can never replace the complete original.
    """

    primary_bytes, primary_media_type, primary_audit = prepare_request_image(
        image_bytes,
        media_type,
        request_config,
    )
    payloads = [primary_bytes]
    media_types = [primary_media_type]
    audits = [{"kind": "original", **primary_audit}]
    raw_policy = request_config.get("image_detail_crops")
    policy = raw_policy if isinstance(raw_policy, Mapping) else {}
    if policy.get("enabled") is not True:
        return payloads, media_types, audits
    if bool(policy.get("binarize", False)):
        raise ValueError("模型细节裁图禁止二值化")

    from PIL import Image, ImageOps

    overlap = float(policy.get("overlap", 0.25))
    if overlap < 0.1 or overlap > 0.5:
        raise ValueError("image_detail_crops.overlap必须在0.1到0.5之间")
    minimum_axis = int(policy.get("minimum_axis", 900))
    if minimum_axis < 256 or minimum_axis > 8192:
        raise ValueError("image_detail_crops.minimum_axis必须在256到8192之间")
    maximum_count = int(policy.get("maximum_count", 2))
    if maximum_count < 0 or maximum_count > 4:
        raise ValueError("image_detail_crops.maximum_count必须在0到4之间")
    if maximum_count == 0:
        return payloads, media_types, audits

    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        audits[0].setdefault("width", width)
        audits[0].setdefault("height", height)
        if max(width, height) < minimum_axis:
            return payloads, media_types, audits
        boxes: list[tuple[int, int, int, int]] = []
        if width >= height:
            window = min(width, int(round(width * (0.5 + overlap / 2))))
            boxes.extend([(0, 0, window, height), (width - window, 0, width, height)])
        else:
            window = min(height, int(round(height * (0.5 + overlap / 2))))
            boxes.extend([(0, 0, width, window), (0, height - window, width, height)])
        if maximum_count > 2 and width >= minimum_axis and height >= minimum_axis:
            horizontal = min(width, int(round(width * (0.5 + overlap / 2))))
            vertical = min(height, int(round(height * (0.5 + overlap / 2))))
            boxes.extend([
                (0, 0, horizontal, vertical),
                (width - horizontal, height - vertical, width, height),
            ])

        seen: set[tuple[int, int, int, int]] = set()
        for box in boxes:
            if box in seen or len(payloads) - 1 >= maximum_count:
                continue
            seen.add(box)
            crop = image.crop(box)
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG", optimize=False)
            content = buffer.getvalue()
            payloads.append(content)
            media_types.append("image/png")
            audits.append({
                "kind": f"detail_{len(payloads) - 1}",
                "enabled": True,
                "bytes": len(content),
                "sha256": sha256_bytes(content),
                "media_type": "image/png",
                "width": crop.width,
                "height": crop.height,
                "box": list(box),
                "binarized": False,
                "lossless": True,
                "source": "original",
            })
    return payloads, media_types, audits


def is_transient_urllib_transport_error(exc: BaseException) -> bool:
    """Classify only connection-level failures eligible for Schannel fallback.

    HTTP responses, authentication failures, and response parsing errors are
    intentionally excluded.  ``URLError`` is unwrapped because urllib commonly
    stores the underlying TLS/socket exception in ``reason``.
    """

    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, urllib.error.URLError):
        reason = exc.reason
        return isinstance(reason, BaseException) and is_transient_urllib_transport_error(
            reason
        )
    if isinstance(exc, ssl.SSLEOFError):
        return True
    if isinstance(exc, http.client.RemoteDisconnected):
        return True
    return isinstance(
        exc,
        (
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            TimeoutError,
            socket.timeout,
        ),
    )


def _curl_config_quote(value: str) -> str:
    """Quote one curl config value without allowing config/header injection."""

    if "\x00" in value or "\r" in value or "\n" in value:
        raise ValueError("curl fallback value contains a forbidden control character")
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _schannel_fallback_enabled(policy: Mapping[str, Any]) -> bool:
    raw = policy.get("windows_schannel_curl_fallback")
    if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
        return False
    max_attempts = raw.get("max_fallback_attempts", 1)
    if max_attempts != 1:
        raise ValueError(
            "windows_schannel_curl_fallback.max_fallback_attempts must be exactly 1"
        )
    return True


def _run_windows_schannel_curl(
    config_text: str,
    *,
    timeout_seconds: float,
    operation: str,
) -> tuple[bytes, int]:
    """Run one HTTPS request with Windows curl/Schannel and stdin-only config.

    Credentials may be present in ``config_text``.  They are never placed in
    argv, environment variables, files, exceptions, or durable records.
    """

    if os.name != "nt":
        raise RuntimeError(f"{operation}: Schannel fallback is Windows-only")
    argv = ["curl.exe", "--config", "-"]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        version = subprocess.run(
            ["curl.exe", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10.0,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"{operation}: Windows curl availability check failed") from exc
    version_text = (version.stdout + version.stderr).decode("utf-8", errors="replace")
    if version.returncode != 0 or "schannel" not in version_text.lower():
        raise RuntimeError(f"{operation}: curl.exe does not report the Schannel backend")

    try:
        completed = subprocess.run(
            argv,
            input=config_text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds + 10.0,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{operation}: Schannel curl transport timeout") from exc
    except OSError as exc:
        raise RuntimeError(f"{operation}: Schannel curl launch failed") from exc

    marker = b"\n__YUYI_HTTP_STATUS__:"
    body, found, status_bytes = completed.stdout.rpartition(marker)
    if not found:
        raise RuntimeError(
            f"{operation}: Schannel curl response omitted HTTP status "
            f"(exit={completed.returncode})"
        )
    try:
        status = int(status_bytes.strip())
    except ValueError as exc:
        raise RuntimeError(f"{operation}: Schannel curl returned invalid HTTP status") from exc
    if completed.returncode != 0:
        category = "HTTP" if status >= 400 else "transport"
        raise RuntimeError(
            f"{operation}: Schannel curl {category} failure "
            f"(HTTP {status}, exit={completed.returncode})"
        )
    if status < 200 or status >= 300:
        raise RuntimeError(f"{operation}: Schannel curl HTTP failure (HTTP {status})")
    return body, status


def _curl_base_config(url: str, timeout_seconds: float) -> list[str]:
    return [
        f"url = {_curl_config_quote(url)}",
        "silent",
        "show-error",
        "no-progress-meter",
        "proto = \"=https\"",
        f"connect-timeout = {min(30.0, timeout_seconds):g}",
        f"max-time = {timeout_seconds:g}",
        'write-out = "\\n__YUYI_HTTP_STATUS__:%{http_code}"',
    ]


def _fetch_policy_with_schannel_curl(
    policy_url: str,
    api_key: str,
) -> Mapping[str, Any]:
    lines = _curl_base_config(policy_url, 60.0)
    lines.extend(
        [
            'request = "GET"',
            f"header = {_curl_config_quote('Authorization: Bearer ' + api_key)}",
            'header = "Accept: application/json"',
            'user-agent = "yuyi-full-benchmark/1.0"',
        ]
    )
    body, _ = _run_windows_schannel_curl(
        "\n".join(lines) + "\n",
        timeout_seconds=60.0,
        operation="dashscope getPolicy fallback",
    )
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("dashscope getPolicy fallback returned invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("dashscope getPolicy fallback returned a non-object JSON value")
    return value


def _upload_with_schannel_curl(
    *,
    upload_host: str,
    policy_values: Mapping[str, str],
    object_key: str,
    payload: bytes,
    media_type: str,
    filename: str,
) -> None:
    """Upload one image through curl; the payload temp file is always removed."""

    with tempfile.TemporaryDirectory(prefix="yuyi-oss-upload-") as directory:
        upload_path = Path(directory) / filename
        with upload_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        lines = _curl_base_config(upload_host, 120.0)
        lines.extend(
            [
                'request = "POST"',
                'user-agent = "yuyi-full-benchmark/1.0"',
                "form-string = "
                + _curl_config_quote(
                    "OSSAccessKeyId=" + policy_values["oss_access_key_id"]
                ),
                "form-string = "
                + _curl_config_quote("Signature=" + policy_values["signature"]),
                "form-string = "
                + _curl_config_quote("policy=" + policy_values["policy"]),
                "form-string = "
                + _curl_config_quote(
                    "x-oss-object-acl=" + policy_values["x_oss_object_acl"]
                ),
                "form-string = "
                + _curl_config_quote(
                    "x-oss-forbid-overwrite="
                    + policy_values["x_oss_forbid_overwrite"]
                ),
                "form-string = " + _curl_config_quote("key=" + object_key),
                'form-string = "success_action_status=200"',
                "form = "
                + _curl_config_quote(
                    f"file=@{upload_path};type={media_type};filename={filename}"
                ),
            ]
        )
        _run_windows_schannel_curl(
            "\n".join(lines) + "\n",
            timeout_seconds=120.0,
            operation="dashscope OSS upload fallback",
        )


def _powershell_fallback_enabled(policy: Mapping[str, Any]) -> bool:
    raw = policy.get("windows_powershell_schannel_fallback")
    if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
        return False
    if raw.get("max_fallback_attempts", 1) != 1:
        raise ValueError(
            "windows_powershell_schannel_fallback.max_fallback_attempts must be exactly 1"
        )
    return True


def _powershell_batch_fallback_enabled(policy: Mapping[str, Any]) -> bool:
    raw = policy.get("windows_powershell_batch_schannel_fallback")
    if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
        return False
    if raw.get("max_fallback_attempts", 1) != 1:
        raise ValueError(
            "windows_powershell_batch_schannel_fallback.max_fallback_attempts must be exactly 1"
        )
    return True


def _persistent_iwr_transport_enabled(policy: Mapping[str, Any]) -> bool:
    raw = policy.get("windows_powershell_persistent_iwr_transport")
    if not isinstance(raw, Mapping) or raw.get("enabled") is not True:
        return False
    if raw.get("max_network_retries_per_question") != 1:
        raise ValueError(
            "windows_powershell_persistent_iwr_transport."
            "max_network_retries_per_question must be exactly 1"
        )
    return True


def _persistent_iwr_backend_settings(
    policy: Mapping[str, Any],
) -> tuple[str, str, str, bool]:
    """Resolve one hash-bound helper and its private-ACL contract."""

    if not _persistent_iwr_transport_enabled(policy):
        raise ValueError("persistent IWR transport is not enabled")
    raw = policy.get("windows_powershell_persistent_iwr_transport")
    if not isinstance(raw, Mapping):
        raise ValueError("persistent IWR backend configuration is invalid")
    script_name = raw.get("fixed_script")
    declared_sha256 = raw.get("fixed_script_sha256")
    if (
        not isinstance(script_name, str)
        or Path(script_name).name != script_name
        or script_name not in WINDOWS_SCHANNEL_PERSISTENT_SCRIPTS
    ):
        raise ValueError("persistent IWR helper script is not trusted")
    trusted = WINDOWS_SCHANNEL_PERSISTENT_SCRIPTS[script_name]
    if declared_sha256 != trusted["sha256"]:
        raise ValueError("persistent IWR helper script hash binding failed")
    validate_acl = raw.get("validate_acl_before_file_write", False)
    if not isinstance(validate_acl, bool):
        raise ValueError("validate_acl_before_file_write must be a boolean")
    if validate_acl is not trusted["strict_private_acl"]:
        raise ValueError("persistent IWR helper ACL contract mismatch")
    return (
        script_name,
        str(trusted["sha256"]),
        str(trusted["protocol"]),
        validate_acl,
    )


def _windows_directory() -> Path:
    """Resolve the kernel-reported Windows directory without consulting PATH."""

    if os.name != "nt":
        raise RuntimeError("persistent PowerShell IWR transport is Windows-only")
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetWindowsDirectoryW(buffer, len(buffer))
    if length < 1 or length >= len(buffer):
        raise RuntimeError("trusted Windows directory is unavailable")
    root = Path(buffer.value).resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("trusted Windows directory is invalid")
    return root


def _trusted_windows_system_binary(*relative_parts: str) -> Path:
    root = _windows_directory()
    candidate = root.joinpath(*relative_parts).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("trusted Windows system binary escaped SystemRoot") from exc
    if not candidate.is_file():
        raise RuntimeError("trusted Windows system binary is unavailable")
    return candidate


def _current_windows_user_sid() -> str:
    """Return the current process token SID for an explicit temporary ACL."""

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)
    ):
        raise RuntimeError("current Windows token is unavailable")
    try:
        required = wintypes.DWORD(0)
        advapi32.GetTokenInformation(token, 1, None, 0, ctypes.byref(required))
        if required.value < ctypes.sizeof(ctypes.c_void_p):
            raise RuntimeError("current Windows token SID is unavailable")
        raw = ctypes.create_string_buffer(required.value)
        if not advapi32.GetTokenInformation(
            token, 1, raw, required, ctypes.byref(required)
        ):
            raise RuntimeError("current Windows token SID cannot be read")
        sid_pointer = ctypes.cast(raw, ctypes.POINTER(wintypes.LPVOID))[0]
        sid_text_pointer = wintypes.LPWSTR()
        if not advapi32.ConvertSidToStringSidW(
            sid_pointer, ctypes.byref(sid_text_pointer)
        ):
            raise RuntimeError("current Windows token SID cannot be encoded")
        try:
            sid = sid_text_pointer.value
        finally:
            kernel32.LocalFree(sid_text_pointer)
    finally:
        kernel32.CloseHandle(token)
    if re.fullmatch(r"S-[0-9-]+", sid or "") is None:
        raise RuntimeError("current Windows token SID is invalid")
    return sid


def _harden_windows_temp_root(
    path: Path, *, reset_explicit_acl: bool = False
) -> None:
    """Protect a fresh transport root with a current-user-and-SYSTEM ACL."""

    icacls = _trusted_windows_system_binary("System32", "icacls.exe")
    sid = _current_windows_user_sid()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    commands: list[list[str]] = []
    if reset_explicit_acl:
        # Python's Windows mode=0o700 creation may add explicit OWNER RIGHTS and
        # Administrators ACEs.  Reset first so /inheritance:r removes inherited
        # entries and the final grants form an exact two-principal DACL.
        commands.append([str(icacls), str(path), "/reset"])
    commands.append(
        [
            str(icacls),
            str(path),
            "/inheritance:r",
            "/grant:r",
            f"*{sid}:(OI)(CI)F",
            "/grant:r",
            "*S-1-5-18:(OI)(CI)F",
        ]
    )
    for command in commands:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15.0,
            creationflags=creationflags,
            env=_credential_free_child_env([]),
        )
        if completed.returncode != 0:
            raise RuntimeError("temporary image directory ACL hardening failed")


class PersistentPowerShellIwrTransport:
    """One fixed PowerShell/IWR process shared by an entire benchmark call.

    The helper is deliberately never restarted.  A timeout, protocol failure,
    or unexpected process exit poisons the session and all later questions fail
    closed.  The helper owns the sole per-question network retry budget.
    """

    def __init__(
        self,
        *,
        script_name: str = "windows_schannel_persistent_transport.ps1",
        script_sha256: str = WINDOWS_SCHANNEL_PERSISTENT_SCRIPT_SHA256,
        validate_acl_before_file_write: bool = False,
    ) -> None:
        trusted = WINDOWS_SCHANNEL_PERSISTENT_SCRIPTS.get(script_name)
        if (
            trusted is None
            or script_sha256 != trusted["sha256"]
            or validate_acl_before_file_write is not trusted["strict_private_acl"]
        ):
            raise RuntimeError("persistent IWR helper selection is not trusted")
        self._script_name = script_name
        self._script_sha256 = script_sha256
        self._protocol = str(trusted["protocol"])
        self._validate_acl_before_file_write = validate_acl_before_file_write
        self._lock = threading.Lock()
        self._closed = False
        self._poisoned = False
        self._responses: queue.Queue[tuple[str, bytes | None]] = queue.Queue()
        self._stderr_seen = False
        self._temp_root = Path(tempfile.mkdtemp(prefix="yuyi-iwr-persistent-"))
        try:
            _harden_windows_temp_root(
                self._temp_root,
                reset_explicit_acl=self._validate_acl_before_file_write,
            )
            powershell = _trusted_windows_system_binary(
                "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
            )
            script_path = Path(__file__).resolve().with_name(
                self._script_name
            )
            if not script_path.is_file():
                raise RuntimeError("fixed persistent Schannel script is missing")
            script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
            if script_sha256 != self._script_sha256:
                raise RuntimeError(
                    "fixed persistent Schannel script SHA-256 mismatch"
                )
            argv = [
                str(powershell),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ]
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
                env=_credential_free_child_env([]),
            )
            self._stdout_thread = threading.Thread(
                target=self._read_stdout,
                name="yuyi-persistent-iwr-stdout",
                daemon=True,
            )
            self._stderr_thread = threading.Thread(
                target=self._discard_stderr,
                name="yuyi-persistent-iwr-stderr",
                daemon=True,
            )
            self._stdout_thread.start()
            self._stderr_thread.start()
            ready = self._receive_json(15.0, "persistent IWR helper startup")
            if set(ready) != {"type", "protocol", "proxy", "redirects"} or ready != {
                "type": "ready",
                "protocol": self._protocol,
                "proxy": "disabled",
                "redirects": "disabled",
            }:
                raise RuntimeError("persistent IWR helper returned invalid readiness")
            if self._validate_acl_before_file_write:
                self._validate_private_acl(
                    self._temp_root,
                    operation_name="persistent IWR private root ACL validation",
                    helper_operation="validate_private_root_acl",
                )
        except Exception:
            self._poisoned = True
            self._terminate_process()
            self._close_process_streams()
            self._cleanup_root()
            raise

    @property
    def process_id(self) -> int:
        return int(self._process.pid)

    @property
    def script_sha256(self) -> str:
        return self._script_sha256

    def _read_stdout(self) -> None:
        stream = self._process.stdout
        if stream is None:
            self._responses.put(("eof", None))
            return
        while True:
            line = stream.readline(PERSISTENT_IWR_MAX_LINE_BYTES + 2)
            if not line:
                self._responses.put(("eof", None))
                return
            if len(line) > PERSISTENT_IWR_MAX_LINE_BYTES or not line.endswith(b"\n"):
                self._responses.put(("oversize", None))
                return
            self._responses.put(("line", line.rstrip(b"\r\n")))

    def _discard_stderr(self) -> None:
        stream = self._process.stderr
        if stream is None:
            return
        while True:
            chunk = stream.read(4096)
            if not chunk:
                return
            self._stderr_seen = True

    def _receive_json(self, timeout_seconds: float, operation: str) -> dict[str, Any]:
        try:
            kind, payload = self._responses.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            self._poisoned = True
            self._terminate_process()
            raise RuntimeError(f"{operation}: helper timeout; session not restarted") from exc
        if kind != "line" or payload is None:
            self._poisoned = True
            self._terminate_process()
            raise RuntimeError(
                f"{operation}: helper protocol failure; session not restarted"
            )
        try:
            result = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._poisoned = True
            self._terminate_process()
            raise RuntimeError(
                f"{operation}: helper returned invalid JSON; session not restarted"
            ) from exc
        if not isinstance(result, dict):
            self._poisoned = True
            raise RuntimeError(f"{operation}: helper response is not an object")
        return result

    def rpc(
        self,
        request_data: Mapping[str, Any],
        *,
        timeout_seconds: float,
        operation: str,
    ) -> dict[str, Any]:
        with self._lock:
            if self._closed or self._poisoned:
                raise RuntimeError(
                    f"{operation}: persistent IWR session is unavailable and is not restarted"
                )
            request_id = uuid.uuid4().hex
            payload = {"request_id": request_id, **dict(request_data)}
            encoded = json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if len(encoded) > PERSISTENT_IWR_MAX_LINE_BYTES:
                raise ValueError("persistent IWR request exceeds protocol limit")
            stdin = self._process.stdin
            if stdin is None:
                self._poisoned = True
                raise RuntimeError(f"{operation}: helper stdin is unavailable")
            try:
                stdin.write(encoded + b"\n")
                stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._poisoned = True
                self._terminate_process()
                raise RuntimeError(
                    f"{operation}: helper exited; session not restarted"
                ) from exc
            result = self._receive_json(timeout_seconds, operation)
            if result.get("request_id") != request_id:
                self._poisoned = True
                self._terminate_process()
                raise RuntimeError(f"{operation}: helper response ID mismatch")
            return result

    def _validate_private_acl(
        self,
        path: Path,
        *,
        operation_name: str,
        helper_operation: str,
    ) -> None:
        response = self.rpc(
            {"operation": helper_operation, "temp_root": str(path)},
            timeout_seconds=10.0,
            operation=operation_name,
        )
        if (
            set(response) != {"request_id", "ok", "status", "acl_valid"}
            or response.get("ok") is not True
            or response.get("status") != 200
            or response.get("acl_valid") is not True
        ):
            self._poisoned = True
            self._terminate_process()
            raise RuntimeError(f"{operation_name}: unsafe temporary ACL")

    def create_question_directory(self) -> Path:
        if self._closed or self._poisoned:
            raise RuntimeError("persistent IWR session is unavailable")
        if not self._validate_acl_before_file_write:
            # Historical v48 behavior is intentionally preserved for immutable
            # run reproduction.  Only the v49 strict profile uses the fixed path.
            directory = Path(
                tempfile.mkdtemp(prefix="question-", dir=self._temp_root)
            )
        else:
            directory = None
            for _ in range(3):
                candidate = self._temp_root / f"question-{uuid.uuid4().hex}"
                try:
                    candidate.mkdir(mode=0o777, exist_ok=False)
                except FileExistsError:
                    continue
                directory = candidate
                break
            if directory is None:
                raise RuntimeError("persistent IWR temporary directory collision")
        if directory.parent.resolve() != self._temp_root.resolve():
            raise RuntimeError("persistent IWR temporary path escaped its root")
        if directory.is_symlink():
            raise RuntimeError("persistent IWR temporary path is a reparse point")
        if self._validate_acl_before_file_write:
            try:
                self._validate_private_acl(
                    directory,
                    operation_name="persistent IWR question ACL validation",
                    helper_operation="validate_temp_acl",
                )
            except Exception:
                try:
                    self.cleanup_question_directory(directory)
                except Exception:
                    pass
                raise
        return directory

    def cleanup_question_directory(self, directory: Path) -> None:
        resolved_root = self._temp_root.resolve()
        resolved = directory.resolve()
        if resolved.parent != resolved_root or not directory.name.startswith("question-"):
            raise RuntimeError("refusing unsafe persistent IWR temporary cleanup")
        if directory.exists():
            shutil.rmtree(directory)

    def _terminate_process(self) -> None:
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _close_process_streams(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    def _cleanup_root(self) -> None:
        root = getattr(self, "_temp_root", None)
        if root is None or not root.exists():
            return
        if not root.name.startswith("yuyi-iwr-persistent-"):
            raise RuntimeError("refusing unsafe persistent IWR root cleanup")
        shutil.rmtree(root)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if not self._poisoned and self._process.poll() is None:
                try:
                    request_id = uuid.uuid4().hex
                    message = json.dumps(
                        {"request_id": request_id, "operation": "shutdown"},
                        separators=(",", ":"),
                    ).encode("ascii")
                    assert self._process.stdin is not None
                    self._process.stdin.write(message + b"\n")
                    self._process.stdin.flush()
                    response = self._receive_json(5.0, "persistent IWR shutdown")
                    if response.get("request_id") != request_id:
                        self._poisoned = True
                except Exception:
                    self._poisoned = True
            self._terminate_process()
            self._close_process_streams()
            self._closed = True
            self._cleanup_root()

    def __enter__(self) -> "PersistentPowerShellIwrTransport":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PersistentIwrTransportError(RuntimeError):
    """Sanitized per-question transport failure with count-only audit data."""

    def __init__(
        self,
        category: str,
        code: str,
        *,
        retry_used: bool,
        completed_upload_count: int,
        orphan_object_count: int,
        whole_question_transport_ms: float | None,
    ) -> None:
        super().__init__(f"persistent IWR {category} failure ({code})")
        self.category = category
        self.code = code
        self.retry_used = retry_used
        self.completed_upload_count = completed_upload_count
        self.orphan_object_count = orphan_object_count
        self.whole_question_transport_ms = whole_question_transport_ms


def _credential_free_child_env(secret_values: Iterable[str]) -> dict[str, str]:
    """Return a minimal PowerShell environment containing no supplied secret."""

    secrets_lower = [value.lower() for value in secret_values if value]
    allowed_names = ("SystemRoot", "WINDIR", "TEMP", "TMP", "COMSPEC")
    child: dict[str, str] = {}
    for name in allowed_names:
        value = os.environ.get(name)
        if not value:
            continue
        lowered = value.lower()
        if any(secret in lowered for secret in secrets_lower):
            continue
        child[name] = value
    return child


def _run_windows_powershell_schannel(
    request_data: Mapping[str, Any],
    *,
    timeout_seconds: float,
    operation: str,
    secret_values: Iterable[str],
) -> bytes:
    """Invoke the fixed .NET HttpClient script with credentials on stdin only."""

    if os.name != "nt":
        raise RuntimeError(f"{operation}: PowerShell Schannel fallback is Windows-only")
    powershell = shutil.which("powershell.exe")
    if not powershell:
        raise RuntimeError(f"{operation}: powershell.exe is unavailable")
    script_path = Path(__file__).resolve().with_name("windows_schannel_http.ps1")
    if not script_path.is_file():
        raise RuntimeError(f"{operation}: fixed Schannel script is missing")
    script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
    if script_sha256 != WINDOWS_SCHANNEL_SCRIPT_SHA256:
        raise RuntimeError(f"{operation}: fixed Schannel script SHA-256 mismatch")
    argv = [
        powershell,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
    ]
    stdin_payload = json.dumps(
        request_data,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    secret_list = [str(value) for value in secret_values if value]
    try:
        completed = subprocess.run(
            argv,
            input=stdin_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds + 15.0,
            creationflags=creationflags,
            env=_credential_free_child_env(secret_list),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{operation}: PowerShell Schannel transport timeout") from exc
    except OSError as exc:
        raise RuntimeError(f"{operation}: PowerShell Schannel launch failed") from exc
    if completed.returncode != 0:
        category = {
            20: "HTTP",
            22: "transport",
            30: "local-validation",
        }.get(completed.returncode, "process")
        raise RuntimeError(
            f"{operation}: PowerShell Schannel {category} failure "
            f"(exit={completed.returncode})"
        )
    return completed.stdout


def _run_windows_powershell_batch_schannel(
    request_data: Mapping[str, Any],
    *,
    timeout_seconds: float,
    operation: str,
    secret_values: Iterable[str],
) -> bytes:
    """Run the fixed one-process policy-plus-all-images Schannel batch script."""

    if os.name != "nt":
        raise RuntimeError(f"{operation}: PowerShell Schannel fallback is Windows-only")
    powershell = shutil.which("powershell.exe")
    if not powershell:
        raise RuntimeError(f"{operation}: powershell.exe is unavailable")
    script_path = Path(__file__).resolve().with_name(
        "windows_schannel_batch_transport.ps1"
    )
    if not script_path.is_file():
        raise RuntimeError(f"{operation}: fixed Schannel batch script is missing")
    script_sha256 = hashlib.sha256(script_path.read_bytes()).hexdigest()
    if script_sha256 != WINDOWS_SCHANNEL_BATCH_SCRIPT_SHA256:
        raise RuntimeError(f"{operation}: fixed Schannel batch script SHA-256 mismatch")
    argv = [
        powershell,
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
    ]
    stdin_payload = json.dumps(
        request_data,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    secret_list = [str(value) for value in secret_values if value]
    try:
        completed = subprocess.run(
            argv,
            input=stdin_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds + 20.0,
            creationflags=creationflags,
            env=_credential_free_child_env(secret_list),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{operation}: PowerShell Schannel batch timeout") from exc
    except OSError as exc:
        raise RuntimeError(f"{operation}: PowerShell Schannel batch launch failed") from exc
    if completed.returncode != 0:
        category = {
            20: "HTTP",
            22: "transport",
            30: "local-validation",
        }.get(completed.returncode, "process")
        raise RuntimeError(
            f"{operation}: PowerShell Schannel batch {category} failure "
            f"(exit={completed.returncode})"
        )
    return completed.stdout


def _transport_all_images_with_powershell_schannel(
    *,
    policy_url: str,
    api_key: str,
    image_payloads: Sequence[bytes],
    media_types: Sequence[str],
    filenames: Sequence[str],
) -> list[str]:
    """Fetch policy and upload every image in one fixed PowerShell process."""

    if not (
        len(image_payloads) == len(media_types) == len(filenames)
        and len(image_payloads) > 0
    ):
        raise ValueError("PowerShell Schannel batch image inputs must align")
    with tempfile.TemporaryDirectory(prefix="yuyi-oss-batch-") as directory:
        image_inputs: list[dict[str, str]] = []
        for index, (payload, media_type, filename) in enumerate(
            zip(image_payloads, media_types, filenames)
        ):
            if re.fullmatch(r"[A-Za-z0-9._-]+", filename) is None:
                raise ValueError("PowerShell Schannel batch filename is unsafe")
            image_path = Path(directory) / f"{index:02d}-{filename}"
            with image_path.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            image_inputs.append(
                {
                    "file_path": str(image_path),
                    "filename": filename,
                    "media_type": media_type,
                }
            )
        body = _run_windows_powershell_batch_schannel(
            {
                "operation": "batch_transport",
                "policy_url": policy_url,
                "api_key": api_key,
                "timeout_seconds": 120,
                "images": image_inputs,
            },
            timeout_seconds=120.0 * (len(image_inputs) + 1),
            operation="dashscope whole-question OSS fallback",
            secret_values=[api_key],
        )
        try:
            result = json.loads(body.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "dashscope whole-question OSS fallback returned invalid JSON"
            ) from exc
        objects = result.get("objects") if isinstance(result, Mapping) else None
        if (
            not isinstance(result, Mapping)
            or result.get("status") != 200
            or not isinstance(objects, list)
        ):
            raise RuntimeError(
                "dashscope whole-question OSS fallback returned invalid status"
            )
        if len(objects) != len(filenames):
            raise RuntimeError(
                "dashscope whole-question OSS fallback returned wrong object count"
            )
        urls: list[str] = []
        for value, filename in zip(objects, filenames):
            if not isinstance(value, str):
                raise RuntimeError("dashscope whole-question OSS object is invalid")
            parsed = urllib.parse.urlsplit(value)
            if (
                parsed.scheme != "oss"
                or not parsed.netloc
                or parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
                or not parsed.path.endswith("/" + filename)
            ):
                raise RuntimeError("dashscope whole-question OSS object is unsafe")
            urls.append(value)
        return urls


def _transport_all_images_with_persistent_iwr(
    session: PersistentPowerShellIwrTransport,
    *,
    policy_url: str,
    api_key: str,
    model_id: str,
    image_payloads: Sequence[bytes],
    media_types: Sequence[str],
    filenames_by_attempt: Sequence[Sequence[str]],
) -> tuple[list[str], dict[str, Any]]:
    """Transport one question through the already-running PowerShell helper."""

    count = len(image_payloads)
    if not (
        1 <= count <= 3
        and len(media_types) == count
        and len(filenames_by_attempt) == count
    ):
        raise ValueError("persistent IWR image inputs must contain one to three aligned images")
    total_bytes = sum(len(payload) for payload in image_payloads)
    if total_bytes > PERSISTENT_IWR_MAX_QUESTION_BYTES:
        raise ValueError("persistent IWR question image bytes exceed the 24 MiB limit")
    if any(
        len(payload) < 1 or len(payload) > PERSISTENT_IWR_MAX_SINGLE_IMAGE_BYTES
        for payload in image_payloads
    ):
        raise ValueError("persistent IWR image exceeds the 10 MiB per-image limit")

    directory = session.create_question_directory()
    try:
        images: list[dict[str, Any]] = []
        for index, (payload, media_type, filename_options) in enumerate(
            zip(image_payloads, media_types, filenames_by_attempt)
        ):
            options = list(filename_options)
            if len(options) != 2 or options[0] == options[1]:
                raise ValueError("persistent IWR requires two distinct attempt filenames")
            if any(re.fullmatch(r"[A-Za-z0-9._-]{1,240}", name) is None for name in options):
                raise ValueError("persistent IWR filename is unsafe")
            if media_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                raise ValueError("persistent IWR image media type is unsupported")
            image_path = directory / f"{index:02d}.payload"
            with image_path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if image_path.is_symlink() or image_path.parent.resolve() != directory.resolve():
                raise RuntimeError("persistent IWR image path failed containment validation")
            images.append(
                {
                    "file_path": str(image_path),
                    "filenames_by_attempt": options,
                    "media_type": media_type,
                    "expected_bytes": len(payload),
                    "expected_sha256": sha256_bytes(payload),
                }
            )
        result = session.rpc(
            {
                "operation": "transport_question",
                "policy_url": policy_url,
                "model_id": model_id,
                "api_key": api_key,
                "timeout_seconds": 120,
                "max_network_retries": 1,
                "temp_root": str(directory),
                "images": images,
            },
            timeout_seconds=120.0 * (count + 1) * 2 + 30.0,
            operation="dashscope persistent whole-question OSS transport",
        )
        if result.get("ok") is not True:
            allowed_failure_keys = {
                "request_id",
                "ok",
                "error_category",
                "error_code",
                "retry_used",
                "completed_upload_count",
                "orphan_object_count",
                "whole_question_transport_ms",
            }
            if not set(result).issubset(allowed_failure_keys):
                raise RuntimeError("persistent IWR failure response has unexpected fields")
            category = result.get("error_category")
            code = result.get("error_code")
            if category not in {"transport", "http", "validation", "local"}:
                category = "protocol"
            if not isinstance(code, str) or re.fullmatch(r"[a-z_]{1,64}", code) is None:
                code = "invalid_failure_code"
            completed_count = result.get("completed_upload_count", 0)
            orphan_count = result.get("orphan_object_count", 0)
            retry_used = result.get("retry_used") is True
            total_ms = result.get("whole_question_transport_ms")
            if not isinstance(completed_count, int) or completed_count < 0:
                completed_count = 0
            if not isinstance(orphan_count, int) or orphan_count < 0:
                orphan_count = 0
            if not isinstance(total_ms, (int, float)) or total_ms < 0:
                total_ms = None
            raise PersistentIwrTransportError(
                category,
                code,
                retry_used=retry_used,
                completed_upload_count=completed_count,
                orphan_object_count=orphan_count,
                whole_question_transport_ms=float(total_ms) if total_ms is not None else None,
            )
        if set(result) != {
            "request_id", "ok", "status", "objects", "verified_sha256", "metrics"
        } or result.get("status") != 200:
            raise RuntimeError("persistent IWR success response has invalid fields")
        objects = result.get("objects")
        verified = result.get("verified_sha256")
        metrics = result.get("metrics")
        if not isinstance(objects, list) or len(objects) != count:
            raise RuntimeError("persistent IWR returned the wrong object count")
        if verified != [sha256_bytes(payload) for payload in image_payloads]:
            raise RuntimeError("persistent IWR returned mismatched image digests")
        if not isinstance(metrics, dict) or set(metrics) != {
            "whole_question_transport_ms",
            "completed_upload_count",
            "orphan_object_count",
            "network_retry_used",
        }:
            raise RuntimeError("persistent IWR returned invalid transport metrics")
        if metrics.get("completed_upload_count") != count:
            raise RuntimeError("persistent IWR completed upload count is invalid")
        if not isinstance(metrics.get("orphan_object_count"), int) or metrics["orphan_object_count"] < 0:
            raise RuntimeError("persistent IWR orphan object count is invalid")
        urls: list[str] = []
        for value, filename_options in zip(objects, filenames_by_attempt):
            if not isinstance(value, str):
                raise RuntimeError("persistent IWR object reference is invalid")
            parsed = urllib.parse.urlsplit(value)
            if (
                parsed.scheme != "oss"
                or not parsed.netloc
                or parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
                or not any(parsed.path.endswith("/" + name) for name in filename_options)
            ):
                raise RuntimeError("persistent IWR object reference is unsafe")
            urls.append(value)
        return urls, dict(metrics)
    finally:
        session.cleanup_question_directory(directory)


def _temporary_oss_variant_filenames(
    image_payloads: Sequence[bytes],
    media_types: Sequence[str],
    image_audits: Sequence[Mapping[str, Any]],
    batch_nonce: str,
) -> list[str]:
    suffix_by_media_type = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    if re.fullmatch(r"[0-9a-f]{16,64}", batch_nonce) is None:
        raise ValueError("temporary OSS batch nonce is invalid")
    filenames: list[str] = []
    for index, (payload, media_type, audit) in enumerate(
        zip(image_payloads, media_types, image_audits)
    ):
        suffix = suffix_by_media_type.get(media_type)
        if suffix is None:
            raise ValueError(
                f"百炼临时图片传输不支持媒体类型{media_type}"
            )
        digest = sha256_bytes(payload)
        kind = safe_component(
            str(audit.get("kind") or f"image_{index}"), "image_kind"
        )
        filenames.append(
            f"{index:02d}-{kind}-{digest[:16]}-{batch_nonce}{suffix}"
        )
    if len(filenames) != len(image_payloads):
        raise ValueError("temporary OSS image filename inputs must align")
    return filenames


def _fetch_policy_with_powershell_schannel(
    policy_url: str,
    api_key: str,
) -> Mapping[str, Any]:
    body = _run_windows_powershell_schannel(
        {
            "operation": "get_policy",
            "url": policy_url,
            "api_key": api_key,
            "timeout_seconds": 60.0,
        },
        timeout_seconds=60.0,
        operation="dashscope getPolicy fallback",
        secret_values=[api_key],
    )
    try:
        value = json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("dashscope getPolicy fallback returned invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("dashscope getPolicy fallback returned a non-object JSON value")
    return value


def _upload_with_powershell_schannel(
    *,
    upload_host: str,
    policy_values: Mapping[str, str],
    object_key: str,
    payload: bytes,
    media_type: str,
    filename: str,
) -> None:
    """Upload through .NET HttpClient; the image temp file is always removed."""

    secret_values = list(policy_values.values())
    with tempfile.TemporaryDirectory(prefix="yuyi-oss-upload-") as directory:
        upload_path = Path(directory) / filename
        with upload_path.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        body = _run_windows_powershell_schannel(
            {
                "operation": "upload",
                "url": upload_host,
                "timeout_seconds": 120.0,
                "file_path": str(upload_path),
                "filename": filename,
                "media_type": media_type,
                "fields": {
                    "OSSAccessKeyId": policy_values["oss_access_key_id"],
                    "Signature": policy_values["signature"],
                    "policy": policy_values["policy"],
                    "x-oss-object-acl": policy_values["x_oss_object_acl"],
                    "x-oss-forbid-overwrite": policy_values[
                        "x_oss_forbid_overwrite"
                    ],
                    "key": object_key,
                    "success_action_status": "200",
                },
            },
            timeout_seconds=120.0,
            operation="dashscope OSS upload fallback",
            secret_values=secret_values,
        )
        try:
            result = json.loads(body.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("dashscope OSS upload fallback returned invalid JSON") from exc
        if not isinstance(result, Mapping) or result.get("status") != 200:
            raise RuntimeError("dashscope OSS upload fallback returned invalid status")


def request_image_urls(
    *,
    image_payloads: Sequence[bytes],
    media_types: Sequence[str],
    image_audits: list[dict[str, Any]],
    request_config: Mapping[str, Any],
    experiment_id: str,
    model_key: str,
    question_id: str,
    api_key: str | None = None,
    model_id: str | None = None,
    persistent_transport: PersistentPowerShellIwrTransport | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Encode or upload image inputs using the configured transport.

    Kimi K3 on Bailian does not accept Base64 image inputs.  ``public_url``
    stages the already-frozen request variants for a caller-managed HTTPS
    host.  ``dashscope_temp_oss`` uploads the variants to Bailian's private,
    model-bound temporary store and returns ``oss://`` references valid for
    48 hours.  Secrets and complete remote object references are never copied
    into durable result records.
    """

    if len(image_payloads) != len(media_types) or len(image_payloads) != len(image_audits):
        raise ValueError("请求图片内容、媒体类型和审计记录数量必须一致")
    transport = str(request_config.get("image_transport") or "data_url")
    if transport == "data_url":
        return (
            [
                f"data:{media_type};base64,"
                + base64.b64encode(payload).decode("ascii")
                for payload, media_type in zip(image_payloads, media_types)
            ],
            image_audits,
        )
    if transport == "dashscope_temp_oss":
        if not api_key:
            raise ValueError("dashscope_temp_oss图片传输缺少API Key")
        if not model_id:
            raise ValueError("dashscope_temp_oss图片传输缺少model_id")
        raw_policy = request_config.get("dashscope_temp_oss_transport")
        policy = raw_policy if isinstance(raw_policy, Mapping) else {}
        policy_endpoint = str(
            policy.get("policy_endpoint")
            or "https://dashscope.aliyuncs.com/api/v1/uploads"
        ).strip()
        parsed_endpoint = urllib.parse.urlsplit(policy_endpoint)
        if (
            parsed_endpoint.scheme != "https"
            or not parsed_endpoint.netloc
            or parsed_endpoint.fragment
            or parsed_endpoint.username
            or parsed_endpoint.password
        ):
            raise ValueError("dashscope_temp_oss上传凭证地址必须是无用户信息的HTTPS地址")
        persistent_iwr_enabled = _persistent_iwr_transport_enabled(policy)
        if persistent_iwr_enabled:
            if policy_endpoint != "https://dashscope.aliyuncs.com/api/v1/uploads":
                raise ValueError("persistent IWR requires the exact DashScope policy endpoint")
            competing_backends = (
                "windows_schannel_curl_fallback",
                "windows_powershell_schannel_fallback",
                "windows_powershell_batch_schannel_fallback",
            )
            if any(
                isinstance(policy.get(name), Mapping)
                and policy[name].get("enabled") is True
                for name in competing_backends
            ):
                raise ValueError("persistent IWR must be the only enabled image backend")
            if persistent_transport is None:
                raise RuntimeError("persistent IWR session was not initialized")
            query = "action=getPolicy&model=" + urllib.parse.quote(model_id, safe="")
            policy_url = policy_endpoint + "?" + query
            primary_nonce = secrets.token_hex(12)
            retry_nonce = secrets.token_hex(12)
            while retry_nonce == primary_nonce:
                retry_nonce = secrets.token_hex(12)
            primary_names = _temporary_oss_variant_filenames(
                image_payloads, media_types, image_audits, primary_nonce
            )
            retry_names = _temporary_oss_variant_filenames(
                image_payloads, media_types, image_audits, retry_nonce
            )
            urls, metrics = _transport_all_images_with_persistent_iwr(
                persistent_transport,
                policy_url=policy_url,
                api_key=api_key,
                model_id=model_id,
                image_payloads=image_payloads,
                media_types=media_types,
                filenames_by_attempt=list(zip(primary_names, retry_names)),
            )
            updated: list[dict[str, Any]] = []
            for index, (payload, audit) in enumerate(zip(image_payloads, image_audits)):
                is_first = index == 0
                updated.append(
                    {
                        **audit,
                        "transport": "dashscope_temp_oss",
                        "temporary_store": "aliyun_bailian_private_oss",
                        "temporary_url_valid_hours": 48,
                        "remote_url_persisted": False,
                        "policy_fetch_ms": None,
                        "policy_fetch_backend": (
                            "windows_powershell_persistent_iwr" if is_first else None
                        ),
                        "upload_ms": None,
                        "upload_backend": "windows_powershell_persistent_iwr",
                        "upload_content_sha256": sha256_bytes(payload),
                        "object_key_strategy": "distinct_nonce_per_internal_attempt_v1",
                        "object_key_nonce_persisted": False,
                        "whole_question_transport_ms": (
                            metrics["whole_question_transport_ms"] if is_first else None
                        ),
                        "completed_upload_count": (
                            metrics["completed_upload_count"] if is_first else None
                        ),
                        "orphan_object_count": (
                            metrics["orphan_object_count"] if is_first else None
                        ),
                        "transport_network_retry_used": (
                            metrics["network_retry_used"] if is_first else None
                        ),
                    }
                )
            return urls, updated
        curl_fallback_enabled = _schannel_fallback_enabled(policy)
        powershell_fallback_enabled = _powershell_fallback_enabled(policy)
        powershell_batch_fallback_enabled = _powershell_batch_fallback_enabled(
            policy
        )
        enabled_fallback_count = sum(
            (
                curl_fallback_enabled,
                powershell_fallback_enabled,
                powershell_batch_fallback_enabled,
            )
        )
        if enabled_fallback_count > 1:
            raise ValueError(
                "dashscope_temp_oss permits only one Windows Schannel fallback backend"
            )
        schannel_fallback_remaining = (
            1 if enabled_fallback_count else 0
        )
        fallback_backend = "batch_powershell" if powershell_batch_fallback_enabled else (
            "powershell" if powershell_fallback_enabled else "curl"
        )
        raw_fallback_config = policy.get(
            "windows_powershell_batch_schannel_fallback"
            if powershell_batch_fallback_enabled
            else (
                "windows_powershell_schannel_fallback"
                if powershell_fallback_enabled
                else "windows_schannel_curl_fallback"
            )
        )
        fallback_config = (
            raw_fallback_config if isinstance(raw_fallback_config, Mapping) else {}
        )
        object_key_strategy = str(
            fallback_config.get("object_key_strategy")
            or "legacy_content_hash_v1"
        )
        if object_key_strategy not in {
            "legacy_content_hash_v1",
            "unique_primary_and_distinct_fallback_batch_nonce_v1",
        }:
            raise ValueError(
                "unsupported dashscope_temp_oss object_key_strategy: "
                f"{object_key_strategy}"
            )
        unique_batch_keys = (
            object_key_strategy
            == "unique_primary_and_distinct_fallback_batch_nonce_v1"
        )

        query = urllib.parse.urlencode({"action": "getPolicy", "model": model_id})
        separator = "&" if parsed_endpoint.query else "?"
        policy_url = policy_endpoint + separator + query
        if fallback_backend == "batch_powershell" and not unique_batch_keys:
            raise ValueError(
                "whole-question PowerShell fallback requires distinct nonce object keys"
            )

        def run_whole_question_powershell_fallback(
            excluded_nonce: str | None = None,
        ) -> tuple[list[str], list[dict[str, Any]]]:
            fallback_nonce = secrets.token_hex(12)
            while excluded_nonce and fallback_nonce == excluded_nonce:
                fallback_nonce = secrets.token_hex(12)
            filenames = _temporary_oss_variant_filenames(
                image_payloads,
                media_types,
                image_audits,
                fallback_nonce,
            )
            fallback_started = monotonic_ms()
            urls = _transport_all_images_with_powershell_schannel(
                policy_url=policy_url,
                api_key=api_key,
                image_payloads=image_payloads,
                media_types=media_types,
                filenames=filenames,
            )
            total_ms = rounded_ms(monotonic_ms() - fallback_started)
            audits: list[dict[str, Any]] = []
            for fallback_index, (payload, audit) in enumerate(
                zip(image_payloads, image_audits)
            ):
                audits.append(
                    {
                        **audit,
                        "transport": "dashscope_temp_oss",
                        "temporary_store": "aliyun_bailian_private_oss",
                        "temporary_url_valid_hours": 48,
                        "remote_url_persisted": False,
                        "policy_fetch_ms": total_ms if fallback_index == 0 else None,
                        "policy_fetch_backend": (
                            "windows_powershell_iwr_batch_schannel_fallback"
                            if fallback_index == 0
                            else None
                        ),
                        "upload_ms": total_ms,
                        "upload_backend": (
                            "windows_powershell_iwr_batch_schannel_fallback"
                        ),
                        "upload_content_sha256": sha256_bytes(payload),
                        "object_key_strategy": object_key_strategy,
                        "object_key_nonce_persisted": False,
                        "whole_question_fallback": True,
                    }
                )
            return urls, audits

        policy_request = urllib.request.Request(
            policy_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "yuyi-full-benchmark/1.0",
            },
            method="GET",
        )
        policy_started = monotonic_ms()
        policy_fetch_backend = "python_urllib"
        try:
            with urllib.request.urlopen(policy_request, timeout=60.0) as response:
                policy_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"获取百炼临时图片上传凭证失败（HTTP {exc.code}）："
                f"{redact_text(body, [api_key], 1000)}"
            ) from exc
        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            ssl.SSLError,
            http.client.RemoteDisconnected,
            ConnectionError,
        ) as exc:
            if (
                schannel_fallback_remaining > 0
                and is_transient_urllib_transport_error(exc)
            ):
                schannel_fallback_remaining -= 1
                if fallback_backend == "batch_powershell":
                    return run_whole_question_powershell_fallback()
                if fallback_backend == "powershell":
                    policy_payload = _fetch_policy_with_powershell_schannel(
                        policy_url, api_key
                    )
                    policy_fetch_backend = (
                        "windows_powershell_dotnet_schannel_fallback"
                    )
                else:
                    policy_payload = _fetch_policy_with_schannel_curl(
                        policy_url, api_key
                    )
                    policy_fetch_backend = "windows_curl_schannel_fallback"
            else:
                raise RuntimeError(
                    f"获取百炼临时图片上传凭证失败：{type(exc).__name__}"
                ) from exc
        policy_fetch_ms = rounded_ms(monotonic_ms() - policy_started)
        policy_data = policy_payload.get("data") if isinstance(policy_payload, Mapping) else None
        if not isinstance(policy_data, Mapping):
            raise RuntimeError("百炼临时图片上传凭证响应缺少data")
        required_policy_fields = (
            "upload_dir",
            "upload_host",
            "oss_access_key_id",
            "signature",
            "policy",
            "x_oss_object_acl",
            "x_oss_forbid_overwrite",
        )
        policy_values: dict[str, str] = {}
        for field in required_policy_fields:
            value = policy_data.get(field)
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"百炼临时图片上传凭证缺少{field}")
            policy_values[field] = value
        upload_host = policy_values["upload_host"]
        parsed_host = urllib.parse.urlsplit(upload_host)
        if (
            parsed_host.scheme != "https"
            or not parsed_host.netloc
            or parsed_host.query
            or parsed_host.fragment
            or parsed_host.username
            or parsed_host.password
        ):
            raise RuntimeError("百炼临时图片上传主机不是安全的HTTPS地址")

        suffix_by_media_type = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        output_urls: list[str] = []
        updated_audits: list[dict[str, Any]] = []
        primary_upload_batch_nonce = secrets.token_hex(12) if unique_batch_keys else ""
        fallback_upload_batch_nonce = secrets.token_hex(12) if unique_batch_keys else ""
        while unique_batch_keys and fallback_upload_batch_nonce == primary_upload_batch_nonce:
            fallback_upload_batch_nonce = secrets.token_hex(12)
        for index, (payload, media_type, audit) in enumerate(
            zip(image_payloads, media_types, image_audits)
        ):
            suffix = suffix_by_media_type.get(media_type)
            if suffix is None:
                raise ValueError(
                    f"百炼临时图片传输不支持媒体类型{media_type}"
                )
            digest = sha256_bytes(payload)
            kind = safe_component(
                str(audit.get("kind") or f"image_{index}"), "image_kind"
            )
            if unique_batch_keys:
                filename = (
                    f"{index:02d}-{kind}-{digest[:16]}-"
                    f"{primary_upload_batch_nonce}{suffix}"
                )
                fallback_filename = (
                    f"{index:02d}-{kind}-{digest[:16]}-"
                    f"{fallback_upload_batch_nonce}{suffix}"
                )
            else:
                filename = f"{index:02d}-{kind}-{digest[:16]}{suffix}"
                fallback_filename = filename
            object_key = f"{policy_values['upload_dir'].rstrip('/')}/{filename}"
            fallback_object_key = (
                f"{policy_values['upload_dir'].rstrip('/')}/{fallback_filename}"
            )
            effective_object_key = object_key
            boundary = f"----yuyi{uuid.uuid4().hex}"
            chunks: list[bytes] = []

            def add_text_field(name: str, value: str) -> None:
                chunks.extend([
                    f"--{boundary}\r\n".encode("ascii"),
                    (
                        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                    ).encode("ascii"),
                    value.encode("utf-8"),
                    b"\r\n",
                ])

            add_text_field("OSSAccessKeyId", policy_values["oss_access_key_id"])
            add_text_field("Signature", policy_values["signature"])
            add_text_field("policy", policy_values["policy"])
            add_text_field("x-oss-object-acl", policy_values["x_oss_object_acl"])
            add_text_field(
                "x-oss-forbid-overwrite",
                policy_values["x_oss_forbid_overwrite"],
            )
            add_text_field("key", object_key)
            add_text_field("success_action_status", "200")
            # OSS requires the file part to be the final form field.
            chunks.extend([
                f"--{boundary}\r\n".encode("ascii"),
                (
                    f'Content-Disposition: form-data; name="file"; '
                    f'filename="{filename}"\r\n'
                ).encode("ascii"),
                f"Content-Type: {media_type}\r\n\r\n".encode("ascii"),
                payload,
                b"\r\n",
                f"--{boundary}--\r\n".encode("ascii"),
            ])
            upload_body = b"".join(chunks)
            upload_request = urllib.request.Request(
                upload_host,
                data=upload_body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Content-Length": str(len(upload_body)),
                    "User-Agent": "yuyi-full-benchmark/1.0",
                },
                method="POST",
            )
            upload_started = monotonic_ms()
            upload_backend = "python_urllib"
            try:
                with urllib.request.urlopen(upload_request, timeout=120.0) as response:
                    response.read()
                    status = getattr(response, "status", response.getcode())
                    if status != 200:
                        raise RuntimeError(f"百炼临时图片上传返回HTTP {status}")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"百炼临时图片上传失败（HTTP {exc.code}）："
                    f"{redact_text(body, [api_key, *policy_values.values()], 1000)}"
                ) from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                ssl.SSLError,
                http.client.RemoteDisconnected,
                ConnectionError,
            ) as exc:
                if (
                    schannel_fallback_remaining > 0
                    and is_transient_urllib_transport_error(exc)
                ):
                    schannel_fallback_remaining -= 1
                    if fallback_backend == "batch_powershell":
                        return run_whole_question_powershell_fallback(
                            primary_upload_batch_nonce
                        )
                    upload_kwargs = {
                        "upload_host": upload_host,
                        "policy_values": policy_values,
                        "object_key": fallback_object_key,
                        "payload": payload,
                        "media_type": media_type,
                        "filename": fallback_filename,
                    }
                    if fallback_backend == "powershell":
                        _upload_with_powershell_schannel(**upload_kwargs)
                        upload_backend = (
                            "windows_powershell_dotnet_schannel_fallback"
                        )
                    else:
                        _upload_with_schannel_curl(**upload_kwargs)
                        upload_backend = "windows_curl_schannel_fallback"
                    effective_object_key = fallback_object_key
                else:
                    raise RuntimeError(
                        f"百炼临时图片上传失败：{type(exc).__name__}"
                    ) from exc
            output_urls.append(f"oss://{effective_object_key}")
            updated_audits.append({
                **audit,
                "transport": "dashscope_temp_oss",
                "temporary_store": "aliyun_bailian_private_oss",
                "temporary_url_valid_hours": 48,
                "remote_url_persisted": False,
                "policy_fetch_ms": policy_fetch_ms if index == 0 else None,
                "policy_fetch_backend": policy_fetch_backend if index == 0 else None,
                "upload_ms": rounded_ms(monotonic_ms() - upload_started),
                "upload_backend": upload_backend,
                "upload_content_sha256": digest,
                "object_key_strategy": object_key_strategy,
                "object_key_nonce_persisted": False,
            })
        return output_urls, updated_audits

    if transport != "public_url":
        raise ValueError(
            "request.image_transport只支持data_url、public_url或dashscope_temp_oss"
        )

    raw_policy = request_config.get("public_image_transport")
    policy = raw_policy if isinstance(raw_policy, Mapping) else {}
    base_url_env = str(policy.get("base_url_env") or "KIMI_PUBLIC_IMAGE_BASE_URL")
    root_env = str(policy.get("root_env") or "KIMI_PUBLIC_IMAGE_ROOT")
    base_url = os.environ.get(base_url_env, "").strip().rstrip("/")
    root_value = os.environ.get(root_env, "").strip()
    if not base_url:
        raise ValueError(f"缺少环境变量{base_url_env}")
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"{base_url_env}必须是无查询参数和片段的HTTPS公网基础地址")
    if not root_value:
        raise ValueError(f"缺少环境变量{root_env}")
    public_root = Path(root_value).resolve()
    if public_root.parent == public_root:
        raise ValueError(f"{root_env}不能是文件系统根目录")

    components = (
        safe_component(experiment_id, "experiment_id"),
        safe_component(model_key, "model_key"),
        safe_component(question_id, "question_id"),
    )
    relative_dir = Path(*components)
    output_urls: list[str] = []
    updated_audits: list[dict[str, Any]] = []
    suffix_by_media_type = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
    }
    for index, (payload, media_type, audit) in enumerate(
        zip(image_payloads, media_types, image_audits)
    ):
        suffix = suffix_by_media_type.get(media_type)
        if suffix is None:
            raise ValueError(f"公网图片传输不支持媒体类型{media_type}")
        digest = sha256_bytes(payload)
        kind = safe_component(str(audit.get("kind") or f"image_{index}"), "image_kind")
        filename = f"{index:02d}-{kind}-{digest[:16]}{suffix}"
        relative_path = relative_dir / filename
        destination = (public_root / relative_path).resolve()
        try:
            destination.relative_to(public_root)
        except ValueError as exc:
            raise ValueError("公网图片暂存路径越界") from exc
        atomic_write_bytes(destination, payload)
        url_path = "/".join(
            urllib.parse.quote(part, safe="-._~")
            for part in relative_path.parts
        )
        output_urls.append(f"{base_url}/{url_path}")
        updated_audits.append(
            {
                **audit,
                "transport": "public_url",
                "public_url_host": parsed.netloc,
                "public_relative_path": relative_path.as_posix(),
                "public_url_persisted": False,
            }
        )
    return output_urls, updated_audits


def failed_local_result(
    context: Mapping[str, Any], sample: Mapping[str, Any],
    model: Mapping[str, Any], message: str, started_at: str,
    started_ms: float, generation: int
) -> dict[str, Any]:
    return {
        "record_version": RESULT_VERSION,
        "experiment_id": context["experiment_id"],
        "invocation_id": context["invocation_id"],
        "generation": generation,
        "task_key": f"{model['key']}::{sample['question_id']}",
        "stage": context["stage"],
        "recorded_at": utc_now(),
        "started_at": started_at,
        "question_id": sample["question_id"],
        "subject": sample.get("subject"),
        "source_group": sample.get("source_group"),
        "source_index": sample.get("source_index"),
        "image": {
            "relative_path": sample.get("image_path"),
            "bytes": None,
            "sha256": None,
            "manifest_sha256": sample.get("image_sha256"),
            "hash_match": None,
            "media_type": sample.get("media_type"),
        },
        "model": {
            "key": model["key"],
            "model_id": model["model_id"],
            "family": model.get("family"),
            "thinking_enabled": model.get("thinking_enabled"),
        },
        "status": "failed_local",
        "api_success": False,
        "validation": None,
        "timing": {
            "image_load_encode_ms": None,
            "end_to_end_ms": rounded_ms(monotonic_ms() - started_ms),
            "provider_response_headers_wait_ms": None,
            "provider_first_content_ms": None,
            "provider_total_ms": None,
            "milestones_ms": milestone_state(
                context.get("root_keys", ROOT_KEYS)
            ),
        },
        "usage": combine_usage([]),
        "estimated_cost": combine_cost([]),
        "attempts": [],
        "error": {"type": "local_error", "message": message[:4000]},
        "output_text": None,
        "output_json": None,
    }


def run_task(
    context: Mapping[str, Any], sample: Mapping[str, Any],
    model: Mapping[str, Any], generation: int
) -> dict[str, Any]:
    started_at = utc_now()
    started_ms = monotonic_ms()
    request_config = context["request_config"]
    image_path = context["dataset_root"] / str(sample["image_path"])
    provider = provider_settings(context["config"], model)
    api_key = os.environ.get(provider["api_key_env"], "")
    base_url = (
        os.environ.get(provider["base_url_env"], "").strip()
        or provider["default_base_url"]
    )
    workspace_id = os.environ.get(provider["workspace_id_env"], "").strip() or None
    if not api_key:
        return failed_local_result(
            context, sample, model,
            f"缺少环境变量{provider['api_key_env']}",
            started_at, started_ms, generation
        )
    if not base_url:
        return failed_local_result(
            context, sample, model,
            f"缺少API地址{provider['base_url_env']}",
            started_at, started_ms, generation
        )
    try:
        encode_started = monotonic_ms()
        image_bytes = image_path.read_bytes()
        actual_hash = sha256_bytes(image_bytes)
        manifest_hash = sample.get("image_sha256")
        hash_match = (
            actual_hash == manifest_hash if isinstance(manifest_hash, str) else None
        )
        if hash_match is False:
            raise ValueError(
                f"图片SHA256与冻结manifest不一致：{sample['question_id']}"
            )
        media_type = image_media_type(sample, image_path)
        (
            request_image_payloads,
            request_media_types,
            request_image_variants_metadata,
        ) = prepare_request_image_variants(
            image_bytes,
            media_type,
            request_config,
        )
        request_image_metadata = request_image_variants_metadata[0]
        image_urls, request_image_variants_metadata = request_image_urls(
            image_payloads=request_image_payloads,
            media_types=request_media_types,
            image_audits=request_image_variants_metadata,
            request_config=request_config,
            experiment_id=str(context["experiment_id"]),
            model_key=str(model["key"]),
            question_id=str(sample["question_id"]),
            api_key=api_key,
            model_id=str(model["model_id"]),
            persistent_transport=context.get("persistent_iwr_transport"),
        )
        request_image_metadata = request_image_variants_metadata[0]
        image_encode_ms = monotonic_ms() - encode_started
    except Exception as exc:
        return failed_local_result(
            context, sample, model, f"{type(exc).__name__}: {exc}",
            started_at, started_ms, generation
        )

    max_attempts = bounded_request_attempts(request_config)
    timeout_seconds = float(
        context.get("timeout_seconds")
        or request_config.get("timeout_seconds", 180)
    )
    backoff_seconds = max(
        0.0, float(request_config.get("retry_backoff_seconds", 2))
    )
    attempts: list[dict[str, Any]] = []
    recovery_kind: str | None = None
    final_validation: dict[str, Any] | None = None
    output_text: str | None = None
    output_json: Any = None
    status = "failed_api"
    terminal_error: dict[str, Any] | None = None
    request_prompt, prompt_route = subject_prompt_route(context, sample)

    for attempt_number in range(1, max_attempts + 1):
        attempt_started_at = utc_now()
        payload = build_payload(
            model, request_prompt, image_urls, request_config,
            context["schema"], recovery_kind, provider
        )
        attempt = perform_http_attempt(
            payload,
            base_url,
            api_key,
            workspace_id if provider["send_workspace_header"] else None,
            timeout_seconds,
            [api_key],
            context.get("root_keys", ROOT_KEYS),
            str(request_config.get("image_transport")) == "dashscope_temp_oss",
        )
        attempt["attempt"] = attempt_number
        attempt["started_at"] = attempt_started_at
        attempt["ended_at"] = utc_now()
        attempt["retry_backoff_ms"] = 0.0
        attempt["prompt_mode"] = (
            "compact_recovery"
            if recovery_kind in {"truncated", "format"}
            else ("primary" if attempt_number == 1 else "primary_retry")
        )
        attempt["prompt_route"] = prompt_route
        attempt["estimated_cost"] = estimate_cost(
            attempt["usage"], model.get("pricing"),
            context["price_snapshot"]
        )
        attempt["validation"] = None
        output_text = attempt.get("output_text") or None
        retryable = bool(attempt.get("retryable"))

        if attempt["transport_success"]:
            validation = parse_and_validate_output(
                attempt.get("output_text", ""), context["schema"],
                str(sample.get("subject") or "")
            )
            attempt["validation"] = {
                key: value for key, value in validation.items() if key != "parsed"
            }
            final_validation = validation
            output_json = validation.get("parsed")
            if validation["usable_output"]:
                attempt["outcome"] = "success"
                status = "success"
                terminal_error = None
                attempts.append(attempt)
                break
            attempt["outcome"] = "invalid_output"
            status = "failed_validation"
            terminal_error = {
                "type": "output_validation_error",
                "message": "；".join(
                    (validation["errors"] + validation["content_errors"])[:5]
                ),
            }
            retryable = True
            recovery_kind = "format"
        else:
            attempt["outcome"] = "api_error"
            terminal_error = attempt.get("error")
            status = "failed_api"
            error_type = (
                terminal_error.get("type")
                if isinstance(terminal_error, Mapping)
                else None
            )
            recovery_kind = (
                "truncated"
                if error_type == "truncated_response"
                else None
            )

        attempt["retryable"] = retryable
        attempts.append(attempt)
        if attempt_number >= max_attempts or not retryable:
            break
        delay = backoff_seconds * (2 ** (attempt_number - 1))
        attempts[-1]["retry_backoff_ms"] = rounded_ms(delay * 1000)
        if delay:
            time.sleep(delay)

    usage = combine_usage(attempt["usage"] for attempt in attempts)
    incurred_cost = combine_cost(
        attempt["estimated_cost"] for attempt in attempts
    )
    final_attempt = attempts[-1] if attempts else {}
    validation_for_disk = (
        {key: value for key, value in final_validation.items() if key != "parsed"}
        if final_validation else None
    )
    root_keys = tuple(context.get("root_keys", ROOT_KEYS))
    final_milestones = final_attempt.get(
        "milestones_ms",
        milestone_state(root_keys),
    )

    def positive_delta(end: Any, start: Any) -> float | None:
        if not isinstance(end, (int, float)) or not isinstance(
            start, (int, float)
        ):
            return None
        return rounded_ms(max(0.0, float(end) - float(start)))

    phase_durations = {
        # This is an observable proxy. It includes upload, network, provider
        # queueing and visual understanding, so it must not be presented as
        # pure image-model compute time.
        "pre_first_content_proxy_ms": final_attempt.get("first_content_ms"),
        "classification_output_ms": None,
        "analysis_output_ms": None,
        "solution_output_ms": None,
        "final_answer_output_ms": None,
        "answer_output_ms": None,
        "verification_output_ms": None,
    }
    if root_keys == ANSWER_FIRST_ACCURACY_ROOT_KEYS:
        answer_complete_ms = final_milestones.get(
            FINAL_ANSWER_COMPLETE_MILESTONE
        )
        phase_durations["answer_output_ms"] = positive_delta(
            answer_complete_ms,
            final_milestones.get("final"),
        )
        phase_durations["final_answer_output_ms"] = phase_durations[
            "answer_output_ms"
        ]
        phase_durations["classification_output_ms"] = positive_delta(
            final_milestones.get("analysis"),
            final_milestones.get("classification"),
        )
        phase_durations["analysis_output_ms"] = positive_delta(
            final_milestones.get("verification"),
            final_milestones.get("analysis"),
        )
        phase_durations["verification_output_ms"] = positive_delta(
            final_attempt.get("provider_total_ms"),
            final_milestones.get("verification"),
        )
    elif root_keys == ("analysis", "answer"):
        phase_durations["analysis_output_ms"] = positive_delta(
            final_milestones.get("answer"),
            final_milestones.get("analysis"),
        )
        phase_durations["answer_output_ms"] = positive_delta(
            final_attempt.get("provider_total_ms"),
            final_milestones.get("answer"),
        )
    else:
        phase_durations["classification_output_ms"] = positive_delta(
            final_milestones.get("analysis"),
            final_milestones.get("classification"),
        )
        phase_durations["analysis_output_ms"] = positive_delta(
            final_milestones.get("solution"),
            final_milestones.get("analysis"),
        )
        phase_durations["solution_output_ms"] = positive_delta(
            final_milestones.get("final_answer"),
            final_milestones.get("solution"),
        )
        phase_durations["final_answer_output_ms"] = positive_delta(
            final_attempt.get("provider_total_ms"),
            final_milestones.get("final_answer"),
        )
    return {
        "record_version": RESULT_VERSION,
        "experiment_id": context["experiment_id"],
        "invocation_id": context["invocation_id"],
        "generation": generation,
        "task_key": f"{model['key']}::{sample['question_id']}",
        "stage": context["stage"],
        "recorded_at": utc_now(),
        "started_at": started_at,
        "question_id": sample["question_id"],
        "subject": sample.get("subject"),
        "source_group": sample.get("source_group"),
        "source_index": sample.get("source_index"),
        "image": {
            "relative_path": sample.get("image_path"),
            "bytes": len(image_bytes),
            "sha256": actual_hash,
            "manifest_sha256": manifest_hash,
            "hash_match": hash_match,
            "media_type": media_type,
            "width": sample.get("width"),
            "height": sample.get("height"),
        },
        "request_image": request_image_metadata,
        "request_images": request_image_variants_metadata,
        "prompt_route": prompt_route,
        "model": {
            "key": model["key"],
            "model_id": model["model_id"],
            "family": model.get("family"),
            "thinking_enabled": model.get("thinking_enabled"),
            "provider": provider["provider_key"],
        },
        "status": status,
        "api_success": any(
            bool(attempt.get("transport_success")) for attempt in attempts
        ),
        "validation": validation_for_disk,
        "timing": {
            "image_load_encode_ms": rounded_ms(image_encode_ms),
            "end_to_end_ms": rounded_ms(monotonic_ms() - started_ms),
            "provider_response_headers_wait_ms": final_attempt.get(
                "response_headers_wait_ms"
            ),
            "provider_first_event_ms": final_attempt.get("first_event_ms"),
            "provider_first_content_ms": final_attempt.get("first_content_ms"),
            "provider_first_reasoning_ms": final_attempt.get("first_reasoning_ms"),
            "provider_total_ms": final_attempt.get("provider_total_ms"),
            "milestones_ms": final_milestones,
            "phase_durations_ms": phase_durations,
        },
        "usage": usage,
        "estimated_cost": incurred_cost,
        "attempts": attempts,
        "error": terminal_error,
        "output_text": output_text,
        "output_json": output_json,
    }


def percentile(values: Sequence[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * percentage
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)


def distribution(values: Iterable[Any]) -> dict[str, Any]:
    numbers = [
        float(value) for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    if not numbers:
        return {
            "count": 0, "min": None, "mean": None, "p50": None,
            "p90": None, "p95": None, "p99": None, "max": None,
        }
    return {
        "count": len(numbers),
        "min": round(min(numbers), 3),
        "mean": round(sum(numbers) / len(numbers), 3),
        "p50": percentile(numbers, 0.50),
        "p90": percentile(numbers, 0.90),
        "p95": percentile(numbers, 0.95),
        "p99": percentile(numbers, 0.99),
        "max": round(max(numbers), 3),
    }


def latest_by_task(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        task_key = record.get("task_key")
        if isinstance(task_key, str):
            latest[task_key] = dict(record)
    return latest


def sum_integer(records: Iterable[Mapping[str, Any]], key: str) -> int:
    total = 0
    for record in records:
        usage = record.get("usage")
        value = usage.get(key) if isinstance(usage, Mapping) else None
        if isinstance(value, int):
            total += value
    return total


def record_latency_scope(
    records: Sequence[Mapping[str, Any]], scope: str
) -> dict[str, Any]:
    """Summarize final-record timings for an explicitly named record scope."""

    def timing_values(field: str) -> list[Any]:
        return [
            record.get("timing", {}).get(field)
            for record in records
            if isinstance(record.get("timing"), Mapping)
        ]

    def phase_values(key: str) -> list[Any]:
        field = f"{key}_ms"
        return [
            record.get("timing", {}).get("phase_durations_ms", {}).get(field)
            for record in records
            if isinstance(record.get("timing"), Mapping)
            and isinstance(
                record["timing"].get("phase_durations_ms"), Mapping
            )
        ]

    milestones: dict[str, dict[str, Any]] = {}
    milestone_keys = list(ROOT_KEYS)
    for record in records:
        timing = record.get("timing")
        raw = (
            timing.get("milestones_ms")
            if isinstance(timing, Mapping)
            else None
        )
        if isinstance(raw, Mapping):
            for key in raw:
                if isinstance(key, str) and key not in milestone_keys:
                    milestone_keys.append(key)
    for root_key in milestone_keys:
        milestones[root_key] = distribution(
            record.get("timing", {}).get("milestones_ms", {}).get(root_key)
            for record in records
            if isinstance(record.get("timing"), Mapping)
            and isinstance(record["timing"].get("milestones_ms"), Mapping)
        )

    result: dict[str, Any] = {
        "scope": scope,
        "record_count": len(records),
    }
    for key in RECORD_TIMING_KEYS:
        result[key] = distribution(timing_values(RECORD_TIMING_FIELDS[key]))
    result["milestones"] = milestones
    result["phase_durations"] = {
        key: distribution(phase_values(key))
        for key in PHASE_DURATION_KEYS
    }
    return result


def attempt_and_retry_summary(
    records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build all-attempt and retry latency summaries from durable records.

    ``retry_overhead_ms`` is deliberately reconstructible: for each retried
    record it is the sum of all non-final attempt wall times plus all recorded
    retry backoffs. It excludes uninstrumented local validation/orchestration
    time and therefore does not rely on a live runner process.
    """

    attempts_by_record: list[list[Mapping[str, Any]]] = []
    for record in records:
        raw_attempts = record.get("attempts")
        attempts_by_record.append(
            [
                attempt for attempt in raw_attempts
                if isinstance(attempt, Mapping)
            ]
            if isinstance(raw_attempts, list)
            else []
        )
    all_attempts = [
        attempt for attempts in attempts_by_record for attempt in attempts
    ]
    records_with_attempts = sum(bool(attempts) for attempts in attempts_by_record)
    attempt_count = len(all_attempts)
    retry_attempt_count = sum(
        max(0, len(attempts) - 1) for attempts in attempts_by_record
    )
    retried_pairs = [
        (record, attempts)
        for record, attempts in zip(records, attempts_by_record)
        if len(attempts) > 1
    ]

    retry_backoff_values: list[float] = []
    retry_overhead_values: list[float] = []
    for _, attempts in retried_pairs:
        backoffs = [
            attempt.get("retry_backoff_ms") for attempt in attempts[:-1]
        ]
        prior_walls = [
            attempt.get("attempt_wall_ms") for attempt in attempts[:-1]
        ]
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in backoffs
        ):
            retry_backoff_values.append(sum(float(value) for value in backoffs))
        if (
            all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in prior_walls
            )
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in backoffs
            )
        ):
            retry_overhead_values.append(
                sum(float(value) for value in prior_walls)
                + sum(float(value) for value in backoffs)
            )

    retry_success_count = sum(
        1 for record, _ in retried_pairs
        if record.get("status") == "success"
    )
    all_attempt_latency = {
        "scope": "all_attempts_of_latest_completed_records",
        "completed_record_count": len(records),
        "record_count": records_with_attempts,
        "attempt_count": attempt_count,
        "response_headers_wait": distribution(
            attempt.get("response_headers_wait_ms")
            for attempt in all_attempts
        ),
        "first_event": distribution(
            attempt.get("first_event_ms") for attempt in all_attempts
        ),
        "first_content": distribution(
            attempt.get("first_content_ms") for attempt in all_attempts
        ),
        "first_reasoning": distribution(
            attempt.get("first_reasoning_ms") for attempt in all_attempts
        ),
        "provider_total": distribution(
            attempt.get("provider_total_ms") for attempt in all_attempts
        ),
        "attempt_wall": distribution(
            attempt.get("attempt_wall_ms") for attempt in all_attempts
        ),
    }
    attempts_summary = {
        "scope": "latest_completed_records",
        "completed_record_count": len(records),
        "records_with_attempts": records_with_attempts,
        "attempt_count": attempt_count,
        "retry_attempt_count": retry_attempt_count,
        "mean_per_completed_record": (
            round(attempt_count / len(records), 4) if records else None
        ),
        "count_per_record": distribution(
            len(attempts) for attempts in attempts_by_record
        ),
    }
    retry_summary = {
        "scope": "latest_completed_records_with_more_than_one_attempt",
        "retried_record_count": len(retried_pairs),
        "retry_attempt_count": retry_attempt_count,
        "retry_success_count": retry_success_count,
        "retry_backoff_ms": distribution(retry_backoff_values),
        "retry_overhead_ms": distribution(retry_overhead_values),
        "retry_overhead_definition": (
            "per retried record: sum(non-final attempt_wall_ms) + "
            "sum(retry_backoff_ms); excludes uninstrumented local "
            "validation/orchestration"
        ),
    }
    return all_attempt_latency, attempts_summary, retry_summary


def summarize_model(
    model: Mapping[str, Any], records: list[dict[str, Any]], planned: int
) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "success"]
    api_success = [record for record in records if record.get("api_success")]
    schema_valid = [
        record for record in records
        if isinstance(record.get("validation"), Mapping)
        and record["validation"].get("strict_schema_valid") is True
    ]
    raw_schema_valid = [
        record for record in records
        if isinstance(record.get("validation"), Mapping)
        and record["validation"].get("raw_strict_schema_valid") is True
    ]
    normalized_outputs = [
        record for record in records
        if isinstance(record.get("validation"), Mapping)
        and isinstance(
            record["validation"].get("output_normalization"),
            Mapping,
        )
        and record["validation"]["output_normalization"].get("changed")
        is True
    ]
    content_complete = [
        record for record in records
        if isinstance(record.get("validation"), Mapping)
        and record["validation"].get("content_complete") is True
    ]
    usable_outputs = [
        record for record in records
        if isinstance(record.get("validation"), Mapping)
        and record["validation"].get("usable_output") is True
    ]
    subject_scored = [
        record for record in records
        if isinstance(record.get("validation"), Mapping)
        and isinstance(record["validation"].get("subject_match"), bool)
    ]
    subject_matches = [
        record for record in subject_scored
        if record["validation"]["subject_match"] is True
    ]
    high_school_scope = [
        record for record in records
        if isinstance(record.get("validation"), Mapping)
        and record["validation"].get("grade_scope") == "high_school"
    ]
    out_of_scope = [
        record for record in records
        if isinstance(record.get("validation"), Mapping)
        and record["validation"].get("grade_scope") == "out_of_scope"
    ]
    unknown_scope = [
        record for record in records
        if not isinstance(record.get("validation"), Mapping)
        or record["validation"].get("grade_scope") == "unknown"
    ]
    first_attempts = [
        record["attempts"][0]
        for record in records
        if isinstance(record.get("attempts"), list) and record["attempts"]
    ]
    first_attempt_api_success = [
        attempt for attempt in first_attempts
        if attempt.get("transport_success") is True
    ]
    first_attempt_schema_valid = [
        attempt for attempt in first_attempts
        if isinstance(attempt.get("validation"), Mapping)
        and attempt["validation"].get("strict_schema_valid") is True
    ]
    first_attempt_raw_schema_valid = [
        attempt for attempt in first_attempts
        if isinstance(attempt.get("validation"), Mapping)
        and attempt["validation"].get("raw_strict_schema_valid") is True
    ]
    first_attempt_normalized_outputs = [
        attempt for attempt in first_attempts
        if isinstance(attempt.get("validation"), Mapping)
        and isinstance(
            attempt["validation"].get("output_normalization"),
            Mapping,
        )
        and attempt["validation"]["output_normalization"].get("changed")
        is True
    ]
    first_attempt_content_complete = [
        attempt for attempt in first_attempts
        if isinstance(attempt.get("validation"), Mapping)
        and attempt["validation"].get("content_complete") is True
    ]
    first_attempt_usable_outputs = [
        attempt for attempt in first_attempts
        if isinstance(attempt.get("validation"), Mapping)
        and attempt["validation"].get("usable_output") is True
    ]
    first_attempt_subject_scored = [
        attempt for attempt in first_attempts
        if isinstance(attempt.get("validation"), Mapping)
        and isinstance(attempt["validation"].get("subject_match"), bool)
    ]
    first_attempt_subject_matches = [
        attempt for attempt in first_attempt_subject_scored
        if attempt["validation"]["subject_match"] is True
    ]
    retry_success_count = sum(
        1 for record in successful
        if isinstance(record.get("attempts"), list)
        and len(record["attempts"]) > 1
    )
    failures: dict[str, int] = {}
    for record in records:
        if record.get("status") == "success":
            continue
        error = record.get("error")
        error_type = (
            str(error.get("type")) if isinstance(error, Mapping)
            else str(record.get("status"))
        )
        failures[error_type] = failures.get(error_type, 0) + 1
    total_attempts = sum(
        len(record.get("attempts", []))
        for record in records if isinstance(record.get("attempts"), list)
    )
    costs = [
        record.get("estimated_cost", {}).get("amount")
        for record in records
        if isinstance(record.get("estimated_cost"), Mapping)
    ]
    known_costs = [
        float(value) for value in costs if isinstance(value, (int, float))
    ]

    success_latency = record_latency_scope(
        successful, "status_success_latest_records"
    )
    all_completed_latency = record_latency_scope(
        records, "all_latest_completed_records"
    )
    (
        all_attempt_latency,
        attempts_summary,
        retry_summary,
    ) = attempt_and_retry_summary(records)
    total_attempts = attempts_summary["attempt_count"]
    retry_success_count = retry_summary["retry_success_count"]
    # Retain the historical success-only keys at latency_ms.* so existing
    # consumers continue to work. New consumers should select an explicit
    # scope via latency_ms.success_only or latency_ms.all_completed.
    latency_summary = {
        key: value for key, value in success_latency.items()
        if key not in {"scope", "record_count"}
    }
    latency_summary.update({
        "legacy_scope": "success_only",
        "success_only": success_latency,
        "all_completed": all_completed_latency,
        "all_attempts": all_attempt_latency,
    })
    completed = len(records)
    return {
        "model_key": model["key"],
        "model_id": model["model_id"],
        "family": model.get("family"),
        "planned_tasks": planned,
        "completed_tasks": completed,
        "success_count": len(successful),
        "failure_count": completed - len(successful),
        "api_success_count": len(api_success),
        "api_success_rate": (
            round(len(api_success) / completed, 6) if completed else None
        ),
        "first_attempt_api_success_count": len(first_attempt_api_success),
        "first_attempt_api_success_rate": (
            round(len(first_attempt_api_success) / completed, 6)
            if completed else None
        ),
        "strict_schema_count": len(schema_valid),
        "strict_schema_rate": (
            round(len(schema_valid) / completed, 6) if completed else None
        ),
        "raw_strict_schema_count": len(raw_schema_valid),
        "raw_strict_schema_rate": (
            round(len(raw_schema_valid) / completed, 6)
            if completed else None
        ),
        "normalized_output_count": len(normalized_outputs),
        "normalized_output_rate": (
            round(len(normalized_outputs) / completed, 6)
            if completed else None
        ),
        "content_complete_count": len(content_complete),
        "content_complete_rate": (
            round(len(content_complete) / completed, 6) if completed else None
        ),
        "usable_output_count": len(usable_outputs),
        "usable_output_rate": (
            round(len(usable_outputs) / completed, 6) if completed else None
        ),
        "first_attempt_strict_schema_count": len(
            first_attempt_schema_valid
        ),
        "first_attempt_strict_schema_rate": (
            round(len(first_attempt_schema_valid) / completed, 6)
            if completed else None
        ),
        "first_attempt_raw_strict_schema_count": len(
            first_attempt_raw_schema_valid
        ),
        "first_attempt_raw_strict_schema_rate": (
            round(len(first_attempt_raw_schema_valid) / completed, 6)
            if completed else None
        ),
        "first_attempt_normalized_output_count": len(
            first_attempt_normalized_outputs
        ),
        "first_attempt_normalized_output_rate": (
            round(len(first_attempt_normalized_outputs) / completed, 6)
            if completed else None
        ),
        "first_attempt_content_complete_count": len(
            first_attempt_content_complete
        ),
        "first_attempt_content_complete_rate": (
            round(len(first_attempt_content_complete) / completed, 6)
            if completed else None
        ),
        "first_attempt_usable_output_count": len(
            first_attempt_usable_outputs
        ),
        "first_attempt_usable_output_rate": (
            round(len(first_attempt_usable_outputs) / completed, 6)
            if completed else None
        ),
        "subject_scored_count": len(subject_scored),
        "subject_match_count": len(subject_matches),
        "subject_accuracy": (
            round(len(subject_matches) / len(subject_scored), 6)
            if subject_scored else None
        ),
        "grade_scope": {
            "high_school_count": len(high_school_scope),
            "out_of_scope_count": len(out_of_scope),
            "unknown_count": len(unknown_scope),
        },
        "first_attempt_subject_scored_count": len(
            first_attempt_subject_scored
        ),
        "first_attempt_subject_match_count": len(
            first_attempt_subject_matches
        ),
        "first_attempt_subject_accuracy": (
            round(
                len(first_attempt_subject_matches)
                / len(first_attempt_subject_scored),
                6,
            )
            if first_attempt_subject_scored else None
        ),
        "attempts_total": total_attempts,
        "retry_count": attempts_summary["retry_attempt_count"],
        "retry_success_count": retry_success_count,
        "average_attempts": (
            round(total_attempts / completed, 4) if completed else None
        ),
        "attempts": attempts_summary,
        "retry": retry_summary,
        "failures_by_type": failures,
        "latency_ms": latency_summary,
        "tokens": {
            "input_total": sum_integer(records, "input_tokens"),
            "output_total": sum_integer(records, "output_tokens"),
            "total": sum_integer(records, "total_tokens"),
            "cached_input_total": sum_integer(records, "cached_input_tokens"),
            "reasoning_output_total": sum_integer(
                records, "reasoning_output_tokens"
            ),
        },
        "estimated_cost": {
            "amount": round(sum(known_costs), 8) if known_costs else None,
            "currency": (
                records[0].get("estimated_cost", {}).get("currency")
                if records and isinstance(records[0].get("estimated_cost"), Mapping)
                else None
            ),
            "records_with_estimate": len(known_costs),
            "records_total": completed,
        },
    }


def build_summary(
    context: Mapping[str, Any], records: Iterable[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    task_keys = {
        f"{model['key']}::{sample['question_id']}"
        for model in models for sample in samples
    }
    latest_all = {
        key: value for key, value in latest_by_task(records).items()
        if key in task_keys
    }
    deferred_provider_capacity = {
        key: value for key, value in latest_all.items()
        if is_deferred_provider_capacity(value)
    }
    latest = {
        key: value for key, value in latest_all.items()
        if key not in deferred_provider_capacity
    }
    model_summaries = []
    for model in models:
        prefix = f"{model['key']}::"
        model_records = [
            record for key, record in latest.items() if key.startswith(prefix)
        ]
        item = summarize_model(model, model_records, len(samples))
        item["excluded_deferred_provider_capacity_record_count"] = sum(
            key.startswith(prefix) for key in deferred_provider_capacity
        )
        model_summaries.append(item)
    smoke_gates = context["config"].get("smoke_gates", {})
    if not isinstance(smoke_gates, Mapping):
        smoke_gates = {}
    for item in model_summaries:
        completed_for_model = item["completed_tasks"]
        cost_amount = item["estimated_cost"].get("amount")
        end_to_end_mean = item["latency_ms"]["all_completed"][
            "end_to_end"
        ].get("mean")
        item["full_run_projection_298"] = {
            "estimated_cost_cny": (
                round(float(cost_amount) / completed_for_model * 298, 6)
                if isinstance(cost_amount, (int, float))
                and completed_for_model > 0 else None
            ),
            "serial_wall_hours": (
                round(float(end_to_end_mean) * 298 / 3_600_000, 4)
                if isinstance(end_to_end_mean, (int, float)) else None
            ),
            "basis": (
                "linear_projection_from_all_latest_completed_records"
            ),
        }
        if context["stage"] == "smoke":
            checks = {
                "all_planned_completed": (
                    item["completed_tasks"] == item["planned_tasks"]
                ),
                "api_success_rate": (
                    isinstance(item["api_success_rate"], (int, float))
                    and item["api_success_rate"]
                    >= float(
                        smoke_gates.get("minimum_api_success_rate", 0.9)
                    )
                ),
                "strict_schema_rate": (
                    isinstance(
                        item["strict_schema_rate"],
                        (int, float),
                    )
                    and item["strict_schema_rate"]
                    >= float(
                        smoke_gates.get(
                            "minimum_strict_schema_rate", 0.8
                        )
                    )
                ),
                "usable_output_rate": (
                    isinstance(
                        item["usable_output_rate"],
                        (int, float),
                    )
                    and item["usable_output_rate"]
                    >= float(
                        smoke_gates.get(
                            "minimum_usable_output_rate", 0.8
                        )
                    )
                ),
                "subject_accuracy": (
                    isinstance(item["subject_accuracy"], (int, float))
                    and item["subject_accuracy"]
                    >= float(
                        smoke_gates.get("minimum_subject_accuracy", 0.9)
                    )
                ),
            }
            end_to_end_distribution = item["latency_ms"][
                "all_completed"
            ]["end_to_end"]
            for threshold_key, metric_name in (
                ("maximum_end_to_end_mean_ms", "mean"),
                ("maximum_end_to_end_p50_ms", "p50"),
                ("maximum_end_to_end_p95_ms", "p95"),
            ):
                if threshold_key not in smoke_gates:
                    continue
                observed = end_to_end_distribution.get(metric_name)
                checks[
                    f"end_to_end_{metric_name}_within_target"
                ] = (
                    isinstance(observed, (int, float))
                    and float(observed)
                    <= float(smoke_gates[threshold_key])
                )
            item["smoke_gate"] = {
                "pass": all(checks.values()),
                "checks": checks,
                "thresholds": dict(smoke_gates),
            }
        else:
            item["smoke_gate"] = None
    total = len(task_keys)
    completed = len(latest)
    success = sum(
        1 for record in latest.values() if record.get("status") == "success"
    )
    formal = context["config"].get("formal_scoring")
    if not isinstance(formal, Mapping):
        formal = {
            "final_answer_accuracy": False,
            "reason": "verified gold is not configured",
        }
    return {
        "summary_version": SUMMARY_VERSION,
        "source": "latest_records_from_results_jsonl",
        "generated_at": utc_now(),
        "experiment_id": context["experiment_id"],
        "stage": context["stage"],
        "dataset_id": context["dataset_id"],
        "sample_count": len(samples),
        "model_count": len(models),
        "planned_task_count": total,
        "completed_task_count": completed,
        "pending_task_count": total - completed,
        "excluded_deferred_provider_capacity_record_count": len(
            deferred_provider_capacity
        ),
        "success_count": success,
        "failure_count": completed - success,
        "success_rate": round(success / completed, 6) if completed else None,
        "formal_correctness": {
            "enabled": bool(formal.get("final_answer_accuracy")),
            "reason": formal.get("reason"),
            "accuracy": None,
        },
        "model_summaries": model_summaries,
    }


def write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    distribution_stats = ("count", "mean", "p50", "p95", "p99")

    def distribution_columns(prefix: str) -> list[str]:
        return [
            f"{prefix}_count" if stat == "count"
            else f"{prefix}_{stat}_ms"
            for stat in distribution_stats
        ]

    def add_distribution(
        row: dict[str, Any], prefix: str, value: Mapping[str, Any]
    ) -> None:
        for stat in distribution_stats:
            column = (
                f"{prefix}_count" if stat == "count"
                else f"{prefix}_{stat}_ms"
            )
            row[column] = value.get(stat)

    fieldnames = [
        "model_key", "model_id", "planned_tasks", "completed_tasks",
        "excluded_deferred_provider_capacity_record_count",
        "success_count", "failure_count", "api_success_rate",
        "first_attempt_api_success_rate", "strict_schema_rate",
        "raw_strict_schema_rate", "normalized_output_rate",
        "content_complete_rate", "usable_output_rate",
        "first_attempt_strict_schema_rate",
        "first_attempt_raw_strict_schema_rate",
        "first_attempt_normalized_output_rate",
        "first_attempt_content_complete_rate",
        "first_attempt_usable_output_rate", "subject_accuracy",
        "first_attempt_subject_accuracy", "average_attempts",
        "retry_success_count",
        "high_school_count", "out_of_scope_count", "unknown_scope_count",
        "smoke_gate_pass", "projected_full_cost_cny",
        "projected_serial_wall_hours",
        "headers_wait_mean_ms", "first_event_mean_ms",
        "first_event_p50_ms", "first_event_p95_ms", "first_event_p99_ms",
        "first_content_mean_ms",
        "provider_total_mean_ms", "provider_total_p50_ms",
        "provider_total_p95_ms", "end_to_end_mean_ms", "end_to_end_p95_ms",
        "classification_output_mean_ms", "analysis_output_mean_ms",
        "solution_output_mean_ms", "final_answer_output_mean_ms",
        "answer_output_mean_ms",
        "input_tokens", "output_tokens", "reasoning_tokens",
        "cached_input_tokens", "total_tokens", "estimated_cost",
        "currency",
    ]
    fieldnames.extend([
        "latency_success_only_scope",
        "latency_success_only_record_count",
        "latency_all_completed_scope",
        "latency_all_completed_record_count",
        "latency_all_attempts_scope",
        "latency_all_attempts_completed_record_count",
        "latency_all_attempts_record_count",
        "latency_all_attempts_attempt_count",
        "attempts_scope",
        "attempts_completed_record_count",
        "attempts_records_with_attempts",
        "attempts_attempt_count",
        "attempts_retry_attempt_count",
        "attempts_mean_per_completed_record",
        "retry_scope",
        "retry_retried_record_count",
        "retry_attempt_count",
        "retry_success_count_structured",
    ])
    for scope_name in ("success_only", "all_completed"):
        for metric in RECORD_TIMING_KEYS:
            fieldnames.extend(distribution_columns(
                f"latency_{scope_name}_{metric}"
            ))
        for phase in PHASE_DURATION_KEYS:
            fieldnames.extend(distribution_columns(
                f"latency_{scope_name}_phase_{phase}"
            ))
    for metric in ("provider_total", "attempt_wall"):
        fieldnames.extend(distribution_columns(
            f"latency_all_attempts_{metric}"
        ))
    for metric in ("backoff", "overhead"):
        fieldnames.extend(distribution_columns(f"retry_{metric}"))
    fieldnames.extend(distribution_columns("attempts_count_per_record"))

    rows: list[dict[str, Any]] = []
    for item in summary.get("model_summaries", []):
        latency = item["latency_ms"]
        tokens = item["tokens"]
        cost = item["estimated_cost"]
        row = {
            "model_key": item["model_key"],
            "model_id": item["model_id"],
            "planned_tasks": item["planned_tasks"],
            "completed_tasks": item["completed_tasks"],
            "excluded_deferred_provider_capacity_record_count": item.get(
                "excluded_deferred_provider_capacity_record_count", 0
            ),
            "success_count": item["success_count"],
            "failure_count": item["failure_count"],
            "api_success_rate": item["api_success_rate"],
            "first_attempt_api_success_rate": item[
                "first_attempt_api_success_rate"
            ],
            "strict_schema_rate": item["strict_schema_rate"],
            "raw_strict_schema_rate": item["raw_strict_schema_rate"],
            "normalized_output_rate": item["normalized_output_rate"],
            "content_complete_rate": item["content_complete_rate"],
            "usable_output_rate": item["usable_output_rate"],
            "first_attempt_strict_schema_rate": item[
                "first_attempt_strict_schema_rate"
            ],
            "first_attempt_raw_strict_schema_rate": item[
                "first_attempt_raw_strict_schema_rate"
            ],
            "first_attempt_normalized_output_rate": item[
                "first_attempt_normalized_output_rate"
            ],
            "first_attempt_content_complete_rate": item[
                "first_attempt_content_complete_rate"
            ],
            "first_attempt_usable_output_rate": item[
                "first_attempt_usable_output_rate"
            ],
            "subject_accuracy": item["subject_accuracy"],
            "first_attempt_subject_accuracy": item[
                "first_attempt_subject_accuracy"
            ],
            "average_attempts": item["average_attempts"],
            "retry_success_count": item["retry_success_count"],
            "high_school_count": item["grade_scope"]["high_school_count"],
            "out_of_scope_count": item["grade_scope"]["out_of_scope_count"],
            "unknown_scope_count": item["grade_scope"]["unknown_count"],
            "smoke_gate_pass": (
                item["smoke_gate"].get("pass")
                if isinstance(item.get("smoke_gate"), Mapping) else None
            ),
            "projected_full_cost_cny": item[
                "full_run_projection_298"
            ]["estimated_cost_cny"],
            "projected_serial_wall_hours": item[
                "full_run_projection_298"
            ]["serial_wall_hours"],
            "headers_wait_mean_ms": latency["response_headers_wait"]["mean"],
            "first_event_mean_ms": latency["first_event"]["mean"],
            "first_event_p50_ms": latency["first_event"]["p50"],
            "first_event_p95_ms": latency["first_event"]["p95"],
            "first_event_p99_ms": latency["first_event"]["p99"],
            "first_content_mean_ms": latency["first_content"]["mean"],
            "provider_total_mean_ms": latency["provider_total"]["mean"],
            "provider_total_p50_ms": latency["provider_total"]["p50"],
            "provider_total_p95_ms": latency["provider_total"]["p95"],
            "end_to_end_mean_ms": latency["end_to_end"]["mean"],
            "end_to_end_p95_ms": latency["end_to_end"]["p95"],
            "classification_output_mean_ms": latency[
                "phase_durations"
            ]["classification_output"]["mean"],
            "analysis_output_mean_ms": latency[
                "phase_durations"
            ]["analysis_output"]["mean"],
            "solution_output_mean_ms": latency[
                "phase_durations"
            ]["solution_output"]["mean"],
            "final_answer_output_mean_ms": latency[
                "phase_durations"
            ]["final_answer_output"]["mean"],
            "answer_output_mean_ms": latency[
                "phase_durations"
            ]["answer_output"]["mean"],
            "input_tokens": tokens["input_total"],
            "output_tokens": tokens["output_total"],
            "reasoning_tokens": tokens["reasoning_output_total"],
            "cached_input_tokens": tokens["cached_input_total"],
            "total_tokens": tokens["total"],
            "estimated_cost": cost["amount"],
            "currency": cost["currency"],
        }
        success_latency = latency["success_only"]
        completed_latency = latency["all_completed"]
        all_attempts = latency["all_attempts"]
        attempts = item["attempts"]
        retry = item["retry"]
        row.update({
            "latency_success_only_scope": success_latency["scope"],
            "latency_success_only_record_count": success_latency[
                "record_count"
            ],
            "latency_all_completed_scope": completed_latency["scope"],
            "latency_all_completed_record_count": completed_latency[
                "record_count"
            ],
            "latency_all_attempts_scope": all_attempts["scope"],
            "latency_all_attempts_completed_record_count": all_attempts[
                "completed_record_count"
            ],
            "latency_all_attempts_record_count": all_attempts[
                "record_count"
            ],
            "latency_all_attempts_attempt_count": all_attempts[
                "attempt_count"
            ],
            "attempts_scope": attempts["scope"],
            "attempts_completed_record_count": attempts[
                "completed_record_count"
            ],
            "attempts_records_with_attempts": attempts[
                "records_with_attempts"
            ],
            "attempts_attempt_count": attempts["attempt_count"],
            "attempts_retry_attempt_count": attempts["retry_attempt_count"],
            "attempts_mean_per_completed_record": attempts[
                "mean_per_completed_record"
            ],
            "retry_scope": retry["scope"],
            "retry_retried_record_count": retry["retried_record_count"],
            "retry_attempt_count": retry["retry_attempt_count"],
            "retry_success_count_structured": retry["retry_success_count"],
        })
        for scope_name, scope_value in (
            ("success_only", success_latency),
            ("all_completed", completed_latency),
        ):
            for metric in RECORD_TIMING_KEYS:
                add_distribution(
                    row, f"latency_{scope_name}_{metric}",
                    scope_value[metric],
                )
            for phase in PHASE_DURATION_KEYS:
                add_distribution(
                    row, f"latency_{scope_name}_phase_{phase}",
                    scope_value["phase_durations"][phase],
                )
        for metric in ("provider_total", "attempt_wall"):
            add_distribution(
                row, f"latency_all_attempts_{metric}",
                all_attempts[metric],
            )
        add_distribution(
            row, "retry_backoff", retry["retry_backoff_ms"]
        )
        add_distribution(
            row, "retry_overhead", retry["retry_overhead_ms"]
        )
        add_distribution(
            row, "attempts_count_per_record", attempts["count_per_record"]
        )
        rows.append(row)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def rebuild_summary_files(experiment_dir: Path) -> dict[str, Any]:
    """Rebuild summary.json/summary.csv from durable run artifacts only.

    This path deliberately does not load credentials, build prompts, schedule
    tasks, or call a provider. The frozen task list and selected model metadata
    come from run.json; measurements come from results.jsonl.
    """

    experiment_dir = experiment_dir.expanduser().resolve()
    run_path = experiment_dir / "run.json"
    results_path = experiment_dir / "results.jsonl"
    if not run_path.is_file():
        raise FileNotFoundError(f"缺少运行元数据：{run_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"缺少结果文件：{results_path}")
    run_doc = load_json(run_path)
    if not isinstance(run_doc, Mapping):
        raise ValueError("run.json根值必须是对象")
    raw_models = run_doc.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("run.json缺少models")
    models = [
        dict(model) for model in raw_models if isinstance(model, Mapping)
    ]
    if len(models) != len(raw_models) or any(
        not isinstance(model.get("key"), str)
        or not isinstance(model.get("model_id"), str)
        for model in models
    ):
        raise ValueError("run.json中的models格式无效")
    raw_question_ids = run_doc.get("question_ids")
    if not isinstance(raw_question_ids, list) or any(
        not isinstance(question_id, str) or not question_id
        for question_id in raw_question_ids
    ):
        raise ValueError("run.json中的question_ids格式无效")
    if len(set(raw_question_ids)) != len(raw_question_ids):
        raise ValueError("run.json中的question_ids存在重复")
    samples = [{"question_id": question_id} for question_id in raw_question_ids]

    # Smoke thresholds and formal-scoring metadata are useful when the original
    # config remains available, but latency reconstruction itself must not
    # depend on it.
    config: Mapping[str, Any] = {}
    paths = run_doc.get("paths")
    config_reference = (
        paths.get("config") if isinstance(paths, Mapping) else None
    )
    if isinstance(config_reference, str) and config_reference:
        config_path = Path(config_reference).expanduser()
        if not config_path.is_absolute():
            config_path = (experiment_dir / config_path).resolve()
        if config_path.is_file():
            loaded_config = load_json(config_path)
            if isinstance(loaded_config, Mapping):
                config = loaded_config

    records, warnings = load_existing_results(results_path)
    context = {
        "experiment_id": str(
            run_doc.get("experiment_id") or experiment_dir.name
        ),
        "stage": str(run_doc.get("stage") or "full"),
        "dataset_id": str(run_doc.get("dataset_id") or "unknown"),
        "config": config,
    }
    summary = build_summary(context, records, models, samples)
    summary["offline_rebuild"] = {
        "mode": "results_jsonl_only_no_api",
        "run_json": str(run_path),
        "results_jsonl": str(results_path),
        "loaded_record_count": len(records),
        "warnings": warnings,
    }
    atomic_write_json(experiment_dir / "summary.json", summary)
    write_summary_csv(experiment_dir / "summary.csv", summary)
    return summary


def task_signature(
    stage: str, samples: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]], hashes: Mapping[str, str]
) -> str:
    material = {
        "result_version": RESULT_VERSION,
        "validation_policy_version": VALIDATION_POLICY_VERSION,
        "stage": stage,
        "question_ids": [sample["question_id"] for sample in samples],
        "models": [
            {"key": model["key"], "model_id": model["model_id"]}
            for model in models
        ],
        "hashes": dict(hashes),
    }
    return sha256_bytes(json_bytes(material))


def progress_document(
    context: Mapping[str, Any], task_keys: set[str],
    latest: Mapping[str, Mapping[str, Any]], pending_now: set[str],
    completed_this_invocation: int, state: str
) -> dict[str, Any]:
    relevant_all = {
        key: record for key, record in latest.items() if key in task_keys
    }
    deferred = {
        key: record for key, record in relevant_all.items()
        if is_deferred_provider_capacity(record)
    }
    relevant = {
        key: record for key, record in relevant_all.items()
        if key not in deferred
    }
    successes = sum(
        1 for record in relevant.values() if record.get("status") == "success"
    )
    return {
        "experiment_id": context["experiment_id"],
        "invocation_id": context["invocation_id"],
        "updated_at": utc_now(),
        "state": state,
        "stage": context["stage"],
        "total_task_count": len(task_keys),
        "latest_result_count": len(relevant_all),
        "latest_completed_count": len(relevant),
        "latest_deferred_provider_capacity_count": len(deferred),
        "latest_success_count": successes,
        "latest_failure_count": len(relevant) - successes,
        "scheduled_this_invocation": context["scheduled_count"],
        "completed_this_invocation": completed_this_invocation,
        "pending_this_invocation": len(pending_now),
    }


def model_public_metadata(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "key": model["key"],
        "model_id": model["model_id"],
        "family": model.get("family"),
        "thinking_enabled": model.get("thinking_enabled"),
        "pricing": model.get("pricing"),
    }


def result_image_library_root(
    configured: str | None, output_root: Path
) -> Path:
    if not configured:
        return (output_root / "model_result_library").resolve()
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def publish_result_image_library(
    *,
    experiment_dir: Path,
    library_root: Path,
    model_keys: Sequence[str],
    font: str | None,
    disabled: bool,
) -> dict[str, Any]:
    if disabled:
        return {
            "status": "disabled",
            "library_root": str(library_root),
        }
    try:
        import result_image_store

        report = result_image_store.publish_experiment(
            experiment_dir=experiment_dir,
            library_root=library_root,
            selected_model_keys=model_keys,
            font=font,
            require_completed=True,
        )
        return {
            "status": "completed",
            "library_root": report["library_root"],
            "model_count": report["model_count"],
            "question_count": report["question_count"],
            "image_count": report["image_count"],
            "total_bytes": report["total_bytes"],
            "publish_ms": report["publish_ms"],
        }
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(
            f"结果图库生成失败，模型原始结果不受影响：{message}",
            file=sys.stderr,
            flush=True,
        )
        return {
            "status": "failed",
            "library_root": str(library_root),
            "error": message[:1000],
        }


def make_parser(script_dir: Path, repo_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="百炼高中图像搜题可断点续跑评测器"
    )
    parser.add_argument(
        "--config", "--config-path", dest="config",
        default=str(script_dir / "benchmark_config_v9.json"),
        help="JSON评测配置路径"
    )
    parser.add_argument(
        "--prompt", "--prompt-path", dest="prompt",
        default=None,
        help="提示词文本或JSON路径"
    )
    parser.add_argument(
        "--schema", default=None,
        help="输出JSON Schema路径；未指定时优先使用配置中的schema，否则使用v2"
    )
    parser.add_argument(
        "--env", "--env-file", dest="env_file",
        default=str(repo_root / "api_keys.env"), help="本地env路径"
    )
    parser.add_argument("--dataset-root", help="数据集根目录")
    parser.add_argument("--manifest", help="完整manifest JSONL路径")
    parser.add_argument(
        "--smoke-manifest", "--smoke", dest="smoke_manifest",
        help="冒烟子集JSONL路径"
    )
    parser.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--models", nargs="+",
        help="模型key列表，支持空格或逗号分隔；默认配置中的全部视觉模型"
    )
    parser.add_argument("--experiment-id", help="实验ID；相同ID用于断点续跑")
    parser.add_argument(
        "--output-root", default=str(script_dir / "runs"), help="输出根目录"
    )
    parser.add_argument("--dry-run", action="store_true", help="只校验和生成计划")
    parser.add_argument("--max-samples", type=int, help="仅取前N个样本")
    parser.add_argument(
        "--retry-failures", action="store_true",
        help="断点续跑时重新执行历史失败项"
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="并发请求数，默认1"
    )
    parser.add_argument("--timeout-seconds", type=float, help="覆盖单次请求超时")
    parser.add_argument(
        "--rebuild-summary-only", metavar="EXPERIMENT_DIR",
        help=(
            "只从指定实验目录的run.json与results.jsonl离线重建"
            "summary.json/summary.csv；不读取密钥、不调用API"
        ),
    )
    parser.add_argument(
        "--result-image-root",
        help=(
            "每模型最新结果图库根目录；默认"
            "<output-root>/model_result_library"
        ),
    )
    parser.add_argument(
        "--result-image-font",
        help="结果图使用的中文 TTF/TTC/OTF 字体路径",
    )
    parser.add_argument(
        "--no-result-images",
        action="store_true",
        help="不生成每模型最新黑白 PNG 结果图库",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    workspace_root = repo_root.parent.parent
    args = make_parser(script_dir, repo_root).parse_args(argv)
    if args.rebuild_summary_only:
        summary = rebuild_summary_files(Path(args.rebuild_summary_only))
        print(
            f"已离线重建汇总：{args.rebuild_summary_only}；"
            f"完成{summary['completed_task_count']}/"
            f"{summary['planned_task_count']}项"
        )
        return 0
    if args.concurrency < 1 or args.concurrency > 32:
        raise ValueError("--concurrency必须在1到32之间")
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds必须大于0")

    config_path = resolve_existing_path(
        args.config, (Path.cwd(), script_dir), "评测配置"
    )
    config = load_json(config_path)
    if not isinstance(config, Mapping):
        raise ValueError("评测配置根值必须是对象")
    prompt_reference = (
        args.prompt
        or config.get("prompt")
        or str(script_dir / "prompts" / "hsqa_image_v5.txt")
    )
    prompt_path = resolve_existing_path(
        str(prompt_reference),
        (Path.cwd(), config_path.parent, script_dir),
        "提示词",
    )
    prompt_search_roots = (Path.cwd(), config_path.parent, script_dir)
    prompt_constraint_paths = resolve_prompt_constraint_paths(
        config.get("prompt_constraints"),
        prompt_search_roots,
        "prompt_constraints",
    )
    prompt_constraint_paths_by_subject = resolve_prompt_constraints_by_subject(
        config.get("prompt_constraints_by_subject"),
        prompt_search_roots,
    )
    schema_reference = (
        args.schema
        or config.get("schema")
        or str(script_dir / "schemas" / "hsqa_output_v2.schema.json")
    )
    schema_path = resolve_existing_path(
        str(schema_reference),
        (Path.cwd(), config_path.parent, script_dir),
        "输出schema",
    )
    env_path = None
    if args.env_file:
        candidate = Path(args.env_file).expanduser()
        if candidate.is_absolute():
            env_path = candidate.resolve()
        else:
            env_path = (repo_root / candidate).resolve()
        if not env_path.exists() and not args.dry_run:
            raise FileNotFoundError(f"env文件不存在：{env_path}")
    loaded_env_names = parse_env_file(env_path)
    prompt = effective_prompt(prompt_path, prompt_constraint_paths)
    prompts_by_subject = {
        subject: effective_prompt(
            prompt_path,
            list(prompt_constraint_paths) + list(subject_paths),
        )
        for subject, subject_paths in prompt_constraint_paths_by_subject.items()
    }
    schema = load_json(schema_path)
    if not isinstance(schema, Mapping):
        raise ValueError("输出schema根值必须是对象")
    root_keys = schema_root_keys(schema)
    pipeline_config = config.get("pipeline")
    if isinstance(pipeline_config, Mapping):
        configured_order = pipeline_config.get("required_output_order")
        if (
            isinstance(configured_order, list)
            and tuple(configured_order) != root_keys
        ):
            raise ValueError(
                "pipeline.required_output_order必须与schema.required顺序一致"
            )

    dataset_root, manifest_path, smoke_path = dataset_paths(
        args, config, config_path.parent, repo_root, workspace_root
    )
    samples = select_samples(
        args.stage, manifest_path, smoke_path, args.max_samples
    )
    available_models = normalize_models(config)
    model_keys = selected_model_keys(args.models, available_models)
    models = [available_models[key] for key in model_keys]
    request_config = config.get("request")
    if not isinstance(request_config, Mapping):
        request_config = {}
    capacity_pause_policy = provider_capacity_pause_policy(request_config)
    if capacity_pause_policy["enabled"] and args.concurrency != 1:
        raise ValueError(
            "启用provider_capacity_pause时--concurrency必须为1"
        )
    if (
        capacity_pause_policy["enabled"]
        and bounded_request_attempts(request_config) != 1
    ):
        raise ValueError(
            "启用provider_capacity_pause时request.max_attempts必须为1"
        )
    price_snapshot = config.get("price_snapshot")
    if not isinstance(price_snapshot, Mapping):
        price_snapshot = {}

    dataset_id = str(
        config.get("dataset_id") or dataset_root.name
    )
    experiment_id = safe_component(
        args.experiment_id or (
            f"{dataset_id}-{args.stage}-"
            + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
        "experiment-id",
    )
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()
    result_library_root = result_image_library_root(
        args.result_image_root, output_root
    )
    experiment_dir = output_root / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    results_path = experiment_dir / "results.jsonl"
    capacity_events_path = experiment_dir / "capacity_events.jsonl"
    run_path = experiment_dir / "run.json"
    progress_path = experiment_dir / "progress.json"
    summary_path = experiment_dir / "summary.json"
    summary_csv_path = experiment_dir / "summary.csv"

    hashes = {
        "runner_script_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path),
        "prompt_sha256": sha256_file(prompt_path),
        "effective_prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "schema_sha256": sha256_file(schema_path),
        "manifest_sha256": sha256_file(manifest_path),
        "smoke_manifest_sha256": sha256_file(smoke_path),
    }
    raw_transport_policy = request_config.get("dashscope_temp_oss_transport")
    transport_policy = (
        raw_transport_policy if isinstance(raw_transport_policy, Mapping) else {}
    )
    persistent_iwr_settings: tuple[str, str, str, bool] | None = None
    if _persistent_iwr_transport_enabled(transport_policy):
        persistent_iwr_settings = _persistent_iwr_backend_settings(transport_policy)
        script_name, declared_script_sha, _, _ = persistent_iwr_settings
        persistent_script = Path(__file__).resolve().with_name(
            script_name
        )
        actual_persistent_script_sha = sha256_file(persistent_script)
        if declared_script_sha != actual_persistent_script_sha:
            raise RuntimeError("persistent IWR script hash binding failed")
        hashes["persistent_iwr_script_sha256"] = actual_persistent_script_sha
    for index, path in enumerate(prompt_constraint_paths, 1):
        hashes[f"prompt_constraint_{index}_sha256"] = sha256_file(path)
    for subject, paths in sorted(prompt_constraint_paths_by_subject.items()):
        hashes[f"effective_prompt_{subject}_sha256"] = sha256_bytes(
            prompts_by_subject[subject].encode("utf-8")
        )
        for index, path in enumerate(paths, 1):
            hashes[
                f"prompt_constraint_{subject}_{index}_sha256"
            ] = sha256_file(path)
    signature = task_signature(args.stage, samples, models, hashes)
    invocation_id = uuid.uuid4().hex
    existing_records, load_warnings = load_existing_results(results_path)
    latest = latest_by_task(existing_records)
    generation_counts: dict[str, int] = {}
    for record in existing_records:
        key = record.get("task_key")
        if isinstance(key, str):
            generation_counts[key] = generation_counts.get(key, 0) + 1

    if run_path.exists():
        run_doc = load_json(run_path)
        if run_doc.get("task_signature") != signature:
            raise RuntimeError(
                "该experiment-id已有不同的数据、模型、提示词或配置；"
                "请使用新的experiment-id"
            )
    else:
        run_doc = {
            "run_version": RESULT_VERSION,
            "experiment_id": experiment_id,
            "created_at": utc_now(),
            "task_signature": signature,
            "stage": args.stage,
            "dataset_id": dataset_id,
            "sample_count": len(samples),
            "question_ids": [sample["question_id"] for sample in samples],
            "models": [model_public_metadata(model) for model in models],
            "paths": {
                "dataset_root": str(dataset_root),
                "manifest": str(manifest_path),
                "smoke_manifest": str(smoke_path),
                "config": str(config_path),
                "prompt": str(prompt_path),
                "prompt_constraints": [
                    str(path) for path in prompt_constraint_paths
                ],
                "prompt_constraints_by_subject": {
                    subject: [str(path) for path in paths]
                    for subject, paths in sorted(
                        prompt_constraint_paths_by_subject.items()
                    )
                },
                "schema": str(schema_path),
                "result_image_library": str(result_library_root),
                "capacity_events": str(capacity_events_path),
            },
            "hashes": hashes,
            "request_policy": {
                "stream": bool(request_config.get("stream", True)),
                "max_attempts": bounded_request_attempts(request_config),
                "timeout_seconds": (
                    args.timeout_seconds
                    or request_config.get("timeout_seconds", 180)
                ),
                "concurrency_default": 1,
                "provider_capacity_pause": capacity_pause_policy,
            },
            "output_contract": {
                "schema_id": schema.get("$id"),
                "root_keys": list(root_keys),
            },
            "result_image_output": (
                dict(config.get("result_image_output"))
                if isinstance(
                    config.get("result_image_output"),
                    Mapping,
                )
                else None
            ),
            "credentials": {
                "values_persisted": False,
                "environment_names": sorted({
                    provider_settings(config, model)["api_key_env"]
                    for model in models
                }),
            },
            "invocations": [],
        }

    task_items: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for model in models:
        for sample in samples:
            key = f"{model['key']}::{sample['question_id']}"
            previous = latest.get(key)
            should_run = previous is None or (
                args.retry_failures and previous.get("status") != "success"
            )
            if should_run:
                task_items.append((sample, model, key))
    task_keys = {
        f"{model['key']}::{sample['question_id']}"
        for model in models for sample in samples
    }
    invocation = {
        "invocation_id": invocation_id,
        "started_at": utc_now(),
        "ended_at": None,
        "state": "dry_run" if args.dry_run else "running",
        "dry_run": args.dry_run,
        "retry_failures": args.retry_failures,
        "concurrency": args.concurrency,
        "loaded_env_names": loaded_env_names,
        "scheduled_task_count": len(task_items),
        "warnings": load_warnings,
    }
    run_doc.setdefault("invocations", []).append(invocation)
    run_doc.setdefault("paths", {})["result_image_library"] = str(
        result_library_root
    )
    run_doc["state"] = invocation["state"]
    atomic_write_json(run_path, run_doc)

    context: dict[str, Any] = {
        "experiment_id": experiment_id,
        "invocation_id": invocation_id,
        "stage": args.stage,
        "dataset_id": dataset_id,
        "dataset_root": dataset_root,
        "config": config,
        "request_config": request_config,
        "price_snapshot": price_snapshot,
        "prompt": prompt,
        "prompts_by_subject": prompts_by_subject,
        "prompt_constraint_paths_by_subject": (
            prompt_constraint_paths_by_subject
        ),
        "schema": schema,
        "root_keys": root_keys,
        "timeout_seconds": args.timeout_seconds,
        "scheduled_count": len(task_items),
        "provider_capacity_pause": capacity_pause_policy,
    }
    pending_now = {key for _, _, key in task_items}
    progress = progress_document(
        context, task_keys, latest, pending_now, 0,
        "dry_run" if args.dry_run else "running"
    )
    atomic_write_json(progress_path, progress)
    summary = build_summary(context, existing_records, models, samples)
    atomic_write_json(summary_path, summary)
    write_summary_csv(summary_csv_path, summary)

    print(
        f"实验 {experiment_id}: {len(samples)}题 × {len(models)}模型 = "
        f"{len(task_keys)}项；本次待执行{len(task_items)}项"
    )
    print(f"输出目录：{experiment_dir}")
    if args.dry_run:
        invocation["ended_at"] = utc_now()
        run_doc["state"] = "dry_run"
        atomic_write_json(run_path, run_doc)
        return 0
    if not task_items:
        invocation["ended_at"] = utc_now()
        invocation["state"] = "completed_noop"
        run_doc["state"] = "completed"
        atomic_write_json(run_path, run_doc)
        image_artifact = publish_result_image_library(
            experiment_dir=experiment_dir,
            library_root=result_library_root,
            model_keys=model_keys,
            font=args.result_image_font,
            disabled=args.no_result_images,
        )
        invocation["result_image_library"] = image_artifact
        run_doc.setdefault("artifacts", {})[
            "result_image_library"
        ] = image_artifact
        atomic_write_json(run_path, run_doc)
        atomic_write_json(
            progress_path,
            progress_document(context, task_keys, latest, set(), 0, "completed")
        )
        return 0

    # Refuse to begin a paid run if any selected provider lacks its API key.
    missing_envs = sorted({
        provider_settings(config, model)["api_key_env"]
        for model in models
        if not os.environ.get(provider_settings(config, model)["api_key_env"])
    })
    if missing_envs:
        raise RuntimeError("缺少API密钥环境变量：" + ", ".join(missing_envs))

    completed_this_invocation = 0
    interrupted = False
    capacity_pause: dict[str, Any] | None = None
    capacity_resume_requires_retry_failures = False
    persistent_iwr_session: PersistentPowerShellIwrTransport | None = None

    def execute_one(
        item: tuple[dict[str, Any], dict[str, Any], str]
    ) -> dict[str, Any]:
        sample, model, key = item
        try:
            return run_task(
                context, sample, model, generation_counts.get(key, 0) + 1
            )
        except Exception as exc:
            started_ms = monotonic_ms()
            return failed_local_result(
                context, sample, model,
                f"runner_internal_error: {type(exc).__name__}: {exc}",
                utc_now(), started_ms, generation_counts.get(key, 0) + 1,
            )

    def record_one(
        item: tuple[dict[str, Any], dict[str, Any], str],
        record: dict[str, Any],
    ) -> None:
        nonlocal completed_this_invocation
        sample, model, key = item
        append_jsonl(results_path, record)
        existing_records.append(record)
        latest[key] = record
        pending_now.discard(key)
        completed_this_invocation += 1
        total_ms = record.get("timing", {}).get("end_to_end_ms")
        print(
            f"[{completed_this_invocation}/{len(task_items)}] "
            f"{model['key']} {sample['question_id']} "
            f"{record['status']} {total_ms}ms",
            flush=True,
        )
        atomic_write_json(
            progress_path,
            progress_document(
                context, task_keys, latest, pending_now,
                completed_this_invocation, "running",
            ),
        )
        current_summary = build_summary(
            context, existing_records, models, samples
        )
        atomic_write_json(summary_path, current_summary)
        write_summary_csv(summary_csv_path, current_summary)

    def capacity_event_one(
        item: tuple[dict[str, Any], dict[str, Any], str],
        record: dict[str, Any],
    ) -> None:
        nonlocal capacity_resume_requires_retry_failures
        sample, model, key = item
        previous = latest.get(key)
        capacity_resume_requires_retry_failures = bool(
            isinstance(previous, Mapping)
            and previous.get("status") != "success"
        )
        control = record.get("batch_control")
        if isinstance(control, dict):
            control["resume_requires_retry_failures"] = (
                capacity_resume_requires_retry_failures
            )
        append_jsonl(capacity_events_path, record)
        print(
            f"[容量暂停] {model['key']} {sample['question_id']} "
            f"{record.get('status')}",
            flush=True,
        )

    try:
        raw_transport_policy = request_config.get("dashscope_temp_oss_transport")
        transport_policy = (
            raw_transport_policy if isinstance(raw_transport_policy, Mapping) else {}
        )
        if _persistent_iwr_transport_enabled(transport_policy):
            if persistent_iwr_settings is None:
                raise RuntimeError("persistent IWR settings were not validated")
            script_name, script_sha256, _, validate_acl = persistent_iwr_settings
            persistent_iwr_session = PersistentPowerShellIwrTransport(
                script_name=script_name,
                script_sha256=script_sha256,
                validate_acl_before_file_write=validate_acl,
            )
            context["persistent_iwr_transport"] = persistent_iwr_session
        if capacity_pause_policy["enabled"]:
            capacity_pause = execute_sequential_with_provider_capacity_pause(
                task_items,
                execute_one,
                record_one,
                capacity_event_one,
                capacity_pause_policy,
            )
            if capacity_pause["paused"]:
                capacity_pause["resume_requires_retry_failures"] = (
                    capacity_resume_requires_retry_failures
                )
                print(
                    "供应商容量过载，批次已安全暂停；未开始题目未调用，"
                    "过载题未写入模型结果，冷却后"
                    + (
                        "使用相同--retry-failures模式续跑。"
                        if capacity_resume_requires_retry_failures
                        else "可直接续跑。"
                    ),
                    flush=True,
                )
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency, thread_name_prefix="benchmark"
            ) as executor:
                futures = {
                    executor.submit(execute_one, item): item
                    for item in task_items
                }
                for future in concurrent.futures.as_completed(futures):
                    item = futures[future]
                    record_one(item, future.result())
    except KeyboardInterrupt:
        interrupted = True
        print("收到中断；已完成记录均已落盘，可用相同experiment-id续跑。")
    finally:
        if persistent_iwr_session is not None:
            persistent_iwr_session.close()
        provider_capacity_paused = bool(
            capacity_pause and capacity_pause.get("paused")
        )
        final_state = (
            "interrupted" if interrupted
            else (
                "paused_provider_capacity"
                if provider_capacity_paused else "completed"
            )
        )
        invocation["ended_at"] = utc_now()
        invocation["state"] = final_state
        invocation["completed_task_count"] = completed_this_invocation
        if provider_capacity_paused:
            invocation["provider_capacity_pause"] = capacity_pause
        run_doc["state"] = final_state
        atomic_write_json(run_path, run_doc)
        if final_state == "completed":
            image_artifact = publish_result_image_library(
                experiment_dir=experiment_dir,
                library_root=result_library_root,
                model_keys=model_keys,
                font=args.result_image_font,
                disabled=args.no_result_images,
            )
            invocation["result_image_library"] = image_artifact
            run_doc.setdefault("artifacts", {})[
                "result_image_library"
            ] = image_artifact
            atomic_write_json(run_path, run_doc)
        final_progress = progress_document(
            context, task_keys, latest, pending_now,
            completed_this_invocation, final_state,
        )
        if provider_capacity_paused:
            final_progress["provider_capacity_pause"] = capacity_pause
        atomic_write_json(progress_path, final_progress)
        summary = build_summary(context, existing_records, models, samples)
        atomic_write_json(summary_path, summary)
        write_summary_csv(summary_csv_path, summary)
    if interrupted:
        return 130
    if capacity_pause and capacity_pause.get("paused"):
        return PROVIDER_CAPACITY_PAUSE_EXIT_CODE
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
