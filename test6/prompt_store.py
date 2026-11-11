"""Versioned prompt overrides shared by portal and subject workers."""
from __future__ import annotations

import hashlib
from contextlib import closing
import os
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
FILES = {"math": "数学题解析紧凑提示词.md", "physics": "物理题解析紧凑提示词.md"}
START = "## 二、发给模型的系统提示词"
END = "## 三、设备端排版约束"


def db_path():
    default = (Path("/var/lib/solver-test4/portal/prompts.sqlite3") if os.name != "nt"
               else ROOT / ".local" / "prompts.sqlite3")
    return Path(os.environ.get("TEST6_PROMPT_DB", str(default)))


def validate(text):
    if not isinstance(text, str):
        raise ValueError("提示词必须是文字")
    if START in text:
        if END not in text or text.index(END) < text.index(START):
            raise ValueError("完整提示词文件缺少正确的章节边界")
        text = text.split(START, 1)[1].split(END, 1)[0]
    text = text.strip()
    if not 500 <= len(text) <= 30000:
        raise ValueError("提示词正文须为 500～30000 个字符")
    if any(key not in text for key in ("l1", "l2", "l3", "l4", "l5")):
        raise ValueError("提示词必须保留 l1～l5 五档输出要求")
    return text


def default_prompt(subject):
    if subject not in FILES:
        raise ValueError("未知学科")
    return validate((ROOT / subject / "source" / FILES[subject]).read_text(encoding="utf-8-sig"))


def revision(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def current(subject, fallback=None):
    if subject not in FILES:
        raise ValueError("未知学科")
    text = fallback if fallback is not None else default_prompt(subject)
    result = {"subject": subject, "text": text, "revision": revision(text), "version": 0, "created_at": None}
    path = db_path()
    if path.exists():
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)) as conn:
            row = conn.execute("SELECT id,text,created_at FROM versions WHERE subject=? ORDER BY id DESC LIMIT 1", (subject,)).fetchone()
        if row:
            result.update(version=row[0], text=row[1], revision=revision(row[1]), created_at=row[2])
    return result


def history(subject):
    default_prompt(subject)
    if not db_path().exists():
        return []
    with closing(sqlite3.connect(db_path().resolve().as_uri() + "?mode=ro", uri=True, timeout=10)) as conn:
        rows = conn.execute("SELECT id,text,created_at FROM versions WHERE subject=? ORDER BY id DESC LIMIT 30", (subject,)).fetchall()
    return [{"version": r[0], "text": r[1], "created_at": r[2]} for r in rows]


class Conflict(Exception):
    pass


def save(subject, text, expected_revision):
    baseline = default_prompt(subject)
    text = validate(text)
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=10)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS versions (id INTEGER PRIMARY KEY, subject TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))")
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT text FROM versions WHERE subject=? ORDER BY id DESC LIMIT 1", (subject,)).fetchone()
        if revision(row[0] if row else baseline) != expected_revision:
            raise Conflict("已有其他人更新提示词，请重新加载后再保存")
        # Preserve the starting version as well as every subsequent edit.
        if not row:
            conn.execute("INSERT INTO versions(subject,text) VALUES (?,?)", (subject, baseline))
        conn.execute("INSERT INTO versions(subject,text) VALUES (?,?)", (subject, text))
        conn.commit()
    return current(subject)
