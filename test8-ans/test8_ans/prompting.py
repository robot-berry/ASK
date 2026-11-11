from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import Question


PROMPT_FILES = {
    "math": "数学题解析紧凑提示词.md",
    "physics": "物理题解析紧凑提示词.md",
}
SCHEMA_FILES = {
    "math": "math.guidance.schema.json",
    "physics": "physics.guidance.schema.json",
}
STAGE_LABELS = {"primary": "小学", "junior": "初中", "senior": "高中"}
TRANSPORT_CONTRACT_VERSION = "test8_transport_v3_lossless_s_collection"


def prompt_path(project_root: Path, subject: str) -> Path:
    return project_root / "prompts" / PROMPT_FILES[subject]


def schema_path(project_root: Path, subject: str) -> Path:
    return project_root / "schemas" / SCHEMA_FILES[subject]


def read_prompt(project_root: Path, subject: str) -> str:
    document = prompt_path(project_root, subject).read_text(encoding="utf-8-sig")
    start_marker = "## 二、发给模型的系统提示词"
    end_marker = "## 三、设备端排版约束"
    start = document.find(start_marker)
    end = document.find(end_marker)
    if start < 0 or end < 0 or end <= start:
        raise ValueError("提示词缺少模型提示词边界")
    prompt = document[start + len(start_marker) : end].strip()
    if len(prompt) < 500:
        raise ValueError("提取到的模型系统提示词异常短")
    return prompt


def read_schema(project_root: Path, subject: str) -> dict[str, Any]:
    value = json.loads(
        schema_path(project_root, subject).read_text(encoding="utf-8-sig")
    )
    if not isinstance(value, dict):
        raise ValueError("Schema 根值必须是对象")
    return value


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def messages_sha256(messages: list[dict[str, str]]) -> str:
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return sha256_text(serialized)


def prompt_protocol_sha256(project_root: Path, subject: str) -> str:
    sentinel = Question(
        id="PROMPT-FINGERPRINT",
        subject=subject,
        stage="senior",
        question="PROMPT_FINGERPRINT_QUESTION",
        ocr="PROMPT_FINGERPRINT_OCR",
    )
    return messages_sha256(build_messages(project_root, sentinel))


def build_messages(project_root: Path, item: Question) -> list[dict[str, str]]:
    system_prompt = read_prompt(project_root, item.subject)
    local_boundary = (
        f"本题学段为{STAGE_LABELS[item.stage]}。"
        "所有分析、知识点与公式必须严格服从系统提示词中的对应学段边界。"
    )
    if item.question_source == "ocr_only":
        main_section = "（未提供经过人工确认的主体题面文字）"
        ocr_section = item.question
        if item.ocr and item.ocr != item.question:
            ocr_section = f"{ocr_section}\n\n补充 OCR：\n{item.ocr}"
    else:
        main_section = item.question
        ocr_section = item.ocr or "（未单独提供 OCR 辅助文字）"
    user_prompt = (
        "请针对下面这一道题生成五档解析，并且只输出 JSON。\n\n"
        f"题目编号：{item.id}\n"
        f"科目：{'数学' if item.subject == 'math' else '物理'}\n"
        f"本地课程边界：{local_boundary}\n\n"
        "【主体题面文字】\n"
        f"{main_section}\n\n"
        "【不可信 OCR 辅助文字】\n"
        f"{ocr_section}\n\n"
        "字段结构必须为："
        '{"l1":[{"n":"题号","s":["..."]}],'
        '"l2":["知识点"],'
        '"l3":[{"n":"题号","s":["..."]}],'
        '"l4":[{"n":"题号","s":["...","..."]}],'
        '"l5":[{"n":"题号","s":["...","...","...","..."],"pit":"易错点"}],'
        '"f":[{"x":"公式","c":"条件"}],'
        '"pat":"通用模式","s":"c或u","u":[]}。'
        "\n\n必须逐项遵守以下 JSON 传输规则：\n"
        "1. l1、l3、l4、l5 都是对象数组，不是字符串数组。\n"
        "2. 每个小问对象只能出现一次 n 键和一次 s 键，严禁重复同名键。\n"
        "3. 每个 s 的值必须是 JSON 字符串数组：l1/l3 恰好1项，l4恰好2项，l5恰好4项。\n"
        '4. l5 的 pit 必须与 n、s 位于同一个对象内，形式为 "pit":"易错点"。\n'
        "5. 输出前在内部核对结构，但最终只输出完整 JSON，不输出核对过程。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_repair_messages(
    base_messages: list[dict[str, str]], raw_content: str, errors: list[str]
) -> list[dict[str, str]]:
    compact_errors = "\n".join(f"- {error}" for error in errors[:20])
    return [
        *base_messages,
        {"role": "assistant", "content": raw_content},
        {
            "role": "user",
            "content": (
                "上一个输出未通过本地 JSON 校验。请保留正确的题目分析，修正以下问题，"
                "并重新输出一份完整 JSON；不要输出 Markdown 或说明。\n"
                f"{compact_errors}"
            ),
        },
    ]
