from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import Settings


TRANSIENT_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class QwenAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Completion:
    content: str
    response: dict[str, Any]
    request_id: str | None
    latency_ms: float = 0.0
    transport_attempts: int = 1


class QwenClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def build_payload(
        self, messages: list[dict[str, str]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        if self.settings.response_format == "json_schema":
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "test8_five_tier_guidance",
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            response_format = {"type": "json_object"}
        return {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "response_format": response_format,
        }

    def complete(
        self, messages: list[dict[str, str]], schema: dict[str, Any]
    ) -> Completion:
        self.settings.validate(require_api_key=True)
        payload = self.build_payload(messages, schema)
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: BaseException | None = None
        started = time.perf_counter()
        for retry_index in range(self.settings.transport_retries + 1):
            request = urllib.request.Request(
                self.settings.chat_completions_url,
                data=encoded,
                headers={
                    "Authorization": f"Bearer {self.settings.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "test8-ans/0.1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.settings.timeout_seconds
                ) as response:
                    body = response.read().decode("utf-8")
                    request_id = response.headers.get("x-dashscope-request-id") or response.headers.get(
                        "x-request-id"
                    )
                document = json.loads(body)
                content = document["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise QwenAPIError("模型响应中没有可用的 choices[0].message.content")
                return Completion(
                    content=content,
                    response=document,
                    request_id=request_id,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    transport_attempts=retry_index + 1,
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = QwenAPIError(
                    f"百炼 API 返回 HTTP {exc.code}：{body[:1000]}",
                    status_code=exc.code,
                )
                if exc.code not in TRANSIENT_STATUS_CODES:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = QwenAPIError(f"连接百炼 API 失败：{exc}")
            except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
                raise QwenAPIError(f"无法解析百炼 API 响应：{exc}") from exc
            if retry_index < self.settings.transport_retries:
                time.sleep(min(2**retry_index, 4))
        assert last_error is not None
        raise last_error
