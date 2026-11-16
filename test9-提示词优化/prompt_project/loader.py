"""Load and render versioned seal-generation prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = PROJECT_ROOT / "registry.json"


class PromptProjectError(RuntimeError):
    """Raised when a prompt package is missing or cannot be rendered safely."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromptProjectError(f"无法读取提示词工程文件：{path}") from exc
    if not isinstance(value, dict):
        raise PromptProjectError(f"提示词工程文件必须是对象：{path}")
    return value


def _registry() -> dict[str, Any]:
    return _read_json(REGISTRY_PATH)


def _version_entry(version: str | None) -> tuple[str, dict[str, Any]]:
    registry = _registry()
    current = registry.get("current_version")
    selected = version or current
    versions = registry.get("versions")
    if not isinstance(versions, dict) or selected not in versions:
        raise PromptProjectError(f"未登记的提示词版本：{selected}")
    entry = versions[selected]
    if not isinstance(entry, dict) or entry.get("status") != "active":
        raise PromptProjectError(f"提示词版本不可用：{selected}")
    return str(selected), entry


def render_prompt(version: str | None, variables: Mapping[str, Any]) -> str:
    selected, entry = _version_entry(version)
    required = entry.get("required_variables", [])
    missing = [name for name in required if name not in variables]
    if missing:
        raise PromptProjectError(f"提示词变量缺失（{selected}）：{', '.join(missing)}")
    template_path = PROJECT_ROOT / str(entry.get("template", ""))
    try:
        prompt = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptProjectError(f"提示词模板不存在：{template_path}") from exc
    for name, value in variables.items():
        prompt = prompt.replace("{{" + str(name) + "}}", str(value))
    unresolved = []
    for token in prompt.split("{{")[1:]:
        token = token.split("}}", 1)[0]
        if "}}" in prompt and token and "}}" in "{{" + token + "}}":
            unresolved.append(token)
    if "{{" in prompt or "}}" in prompt:
        raise PromptProjectError(f"提示词存在未解析变量（{selected}）：{', '.join(unresolved) or 'unknown'}")
    return prompt


def load_prompt_package(version: str | None, variables: Mapping[str, Any]) -> dict[str, Any]:
    selected, entry = _version_entry(version)
    prompt = render_prompt(selected, variables)
    schema_path = PROJECT_ROOT / str(entry.get("schema", ""))
    schema = _read_json(schema_path)
    return {
        "version": selected,
        "prompt": prompt,
        "schema": schema,
        "validator_contract": entry.get("validator_contract"),
    }
