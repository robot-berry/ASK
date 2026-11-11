from __future__ import annotations

import ipaddress
import json
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _loopback_health_url(value: object) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Resource-guard health URLs must be unauthenticated loopback HTTP URLs")
    hostname = parsed.hostname.lower()
    if hostname != "localhost":
        try:
            if not ipaddress.ip_address(hostname).is_loopback:
                raise ValueError
        except ValueError as exc:
            raise ValueError("Resource-guard health URLs must use a loopback host") from exc
    if not parsed.port:
        raise ValueError("Resource-guard health URLs must specify a port")
    return url


def probe_batch_resource_conflicts(settings: dict[str, Any] | None) -> dict[str, Any]:
    settings = settings or {}
    enabled = bool(settings.get("enabled", False))
    result: dict[str, Any] = {
        "enabled": enabled,
        "status": "DISABLED" if not enabled else "CLEAR",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "conflicts": [],
        "unavailable": [],
    }
    if not enabled:
        return result

    timeout_seconds = max(0.05, float(settings.get("timeout_seconds", 0.35)))
    services = settings.get("conflicting_services") or []
    if not isinstance(services, list):
        raise ValueError("batch_resource_guard.conflicting_services must be a list")
    for position, entry in enumerate(services):
        if isinstance(entry, str):
            name = f"conflicting_service_{position + 1}"
            health_url = _loopback_health_url(entry)
        elif isinstance(entry, dict):
            name = str(entry.get("name") or f"conflicting_service_{position + 1}")
            health_url = _loopback_health_url(entry.get("health_url"))
        else:
            raise ValueError("Each conflicting service must be a URL string or an object")

        request = Request(health_url, headers={"User-Agent": "test7-ocr-resource-guard/1"})
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read(65_537)
                if len(raw) > 65_536:
                    raise ValueError("Resource-guard health response is too large")
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except (UnicodeDecodeError, json.JSONDecodeError):
                    payload = {}
                evidence = {
                    "name": name,
                    "health_url": health_url,
                    "http_status": int(getattr(response, "status", 200)),
                    "service_status": payload.get("status") if isinstance(payload, dict) else None,
                    "ready": payload.get("ready") if isinstance(payload, dict) else None,
                    "engine": payload.get("engine") if isinstance(payload, dict) else None,
                }
                result["conflicts"].append(evidence)
        except HTTPError as exc:
            result["conflicts"].append(
                {
                    "name": name,
                    "health_url": health_url,
                    "http_status": int(exc.code),
                    "service_status": None,
                    "ready": None,
                    "engine": None,
                }
            )
        except (URLError, TimeoutError, OSError) as exc:
            result["unavailable"].append(
                {
                    "name": name,
                    "health_url": health_url,
                    "reason": type(exc).__name__,
                }
            )
    if result["conflicts"]:
        result["status"] = "CONFLICT"
    return result
