from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def load_env_file(path: Path) -> list[str]:
    """Load simple KEY=VALUE pairs without overwriting process variables."""

    loaded: list[str] = []
    if not path.is_file():
        return loaded
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    value = default if not raw else int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return value


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    value = default if not raw else float(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = "qwen-max"
    response_format: str = "json_object"
    timeout_seconds: int = 120
    transport_retries: int = 2
    repair_attempts: int = 0
    max_tokens: int = 4096
    temperature: float = 0.0
    latency_budget_ms: int = 20_000
    latency_scope: str = "files_ready_with_png"
    min_schema_pass_rate: float = 0.95
    min_relevance_pass_rate: float = 0.95
    min_latency_pass_rate: float = 0.95
    min_combined_pass_rate: float = 0.95

    @classmethod
    def from_environment(cls, project_root: Path) -> "Settings":
        load_env_file(project_root / ".env")
        if not os.environ.get("DASHSCOPE_API_KEY", "").strip():
            workspace_key_file = project_root.parents[1] / "api_keys.env"
            load_env_file(workspace_key_file)
        settings = cls(
            api_key=os.environ.get("DASHSCOPE_API_KEY", "").strip(),
            base_url=os.environ.get("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL).strip(),
            model=os.environ.get("TEST8_MODEL", "qwen-max").strip(),
            response_format=os.environ.get(
                "TEST8_RESPONSE_FORMAT", "json_object"
            ).strip(),
            timeout_seconds=_int_env("TEST8_TIMEOUT_SECONDS", 120, 10, 600),
            transport_retries=_int_env("TEST8_TRANSPORT_RETRIES", 2, 0, 8),
            repair_attempts=_int_env("TEST8_REPAIR_ATTEMPTS", 0, 0, 3),
            max_tokens=_int_env("TEST8_MAX_TOKENS", 4096, 256, 32768),
            latency_budget_ms=_int_env(
                "TEST8_LATENCY_BUDGET_MS", 20_000, 1_000, 600_000
            ),
            latency_scope=os.environ.get(
                "TEST8_LATENCY_SCOPE", "files_ready_with_png"
            ).strip(),
            min_schema_pass_rate=_float_env(
                "TEST8_MIN_SCHEMA_PASS_RATE", 0.95, 0.0, 1.0
            ),
            min_relevance_pass_rate=_float_env(
                "TEST8_MIN_RELEVANCE_PASS_RATE", 0.95, 0.0, 1.0
            ),
            min_latency_pass_rate=_float_env(
                "TEST8_MIN_LATENCY_PASS_RATE", 0.95, 0.0, 1.0
            ),
            min_combined_pass_rate=_float_env(
                "TEST8_MIN_COMBINED_PASS_RATE", 0.95, 0.0, 1.0
            ),
        )
        settings.validate(require_api_key=False)
        return settings

    def validate(self, *, require_api_key: bool) -> None:
        if require_api_key and not self.api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY；请从 .env.example 创建 .env")
        if not self.model:
            raise ValueError("TEST8_MODEL 不能为空")
        if self.response_format not in {"json_object", "json_schema"}:
            raise ValueError(
                "TEST8_RESPONSE_FORMAT 只支持 json_object 或 json_schema"
            )
        if self.latency_scope not in {
            "model_only",
            "generation_pipeline",
            "files_ready_with_png",
        }:
            raise ValueError(
                "TEST8_LATENCY_SCOPE 只支持 model_only、generation_pipeline "
                "或 files_ready_with_png"
            )
        parsed = urlparse(self.base_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("DASHSCOPE_BASE_URL 必须是有效的 HTTPS 地址")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("DASHSCOPE_BASE_URL 不应包含凭据、查询参数或片段")
        hostname = parsed.hostname.lower()
        allowed_host = (
            hostname in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"}
            or hostname.endswith(".dashscope.aliyuncs.com")
            or hostname.endswith(".maas.aliyuncs.com")
        )
        if not allowed_host:
            raise ValueError("DASHSCOPE_BASE_URL 必须指向阿里云百炼官方域名")

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"
