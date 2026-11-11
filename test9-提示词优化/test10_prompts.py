"""Read the packaged Test10 prompt library and retain source provenance.

Production releases vendor the small prompt subset needed at runtime.  The
development checkout still falls back to the sibling Test10 project so the
source library remains the single authoring location.
"""
import hashlib
import json
import os
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_PACKAGED_ROOT = _PROJECT_ROOT / "test10-提示词工程"
_SIBLING_ROOT = _PROJECT_ROOT.parent / "test10-提示词工程"


def _prompt_root() -> Path:
    """Resolve the Test10 authoring tree used by the running search service.

    ``TEST10_PROMPT_ROOT`` is intentionally first so a deployment can point at
    the maintained Test10 project directly.  The packaged subset remains the
    default for release bundles, and the sibling checkout is the development
    fallback when no packaged subset is present.
    """
    configured = os.environ.get("TEST10_PROMPT_ROOT", "").strip()
    candidates = ([Path(configured).expanduser()] if configured else [])
    candidates.extend((_PACKAGED_ROOT, _SIBLING_ROOT))
    for candidate in candidates:
        root = candidate.resolve()
        if (root / "提示词" / "索引.json").is_file():
            return root
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到 Test10 提示词库（已搜索：{searched}）")


ROOT = _prompt_root()
ORDER = {f"g{grade}_s{semester}": (grade - 7) * 2 + semester - 1
         for grade in range(7, 13) for semester in (1, 2)}
JUNIOR = "提示词/数学/初中/01_提示词/V11_数据库版本9_候选提示词.txt"
JUNIOR_SYSTEM = "提示词/数学/初中/01_提示词/请求附加_system提示词.txt"
HIGH_ENTRY = "提示词/数学/高中/00-搜题总入口.md"
HIGH_OUTPUT = "提示词/数学/高中/00-统一输出结构.md"
JUNIOR_SIGNALS = ("一元一次方程", "二元一次方程", "有理数", "整式", "勾股定理", "相似三角形", "全等三角形", "圆周角", "一次函数", "反比例函数")


def _grams(text):
    words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", str(text or "").lower())
    return {word[index:index + 2] for word in words for index in range(max(1, len(word) - 1))}


def build_context(subject, selection, question):
    term = (selection or {}).get("id", "")
    if subject != "math" or term not in ORDER:
        return {"status": "not_applicable", "text": "", "sources": []}

    grade = int(term[1:term.index("_")])
    is_junior = grade <= 9
    topic = None
    if is_junior:
        paths = [JUNIOR, JUNIOR_SYSTEM]
    else:
        topics = json.loads((ROOT / "提示词/索引.json").read_text(encoding="utf-8-sig"))["topics"]
        query = _grams(question)
        candidates = []
        for item in topics:
            rank = {"高一": 6, "高二": 8, "高三": 10}[item["grade"]] + (item["semester"] == "下学期")
            if rank > ORDER[term]:
                continue
            title = re.sub(r"^\d+-", "", item["topic"])
            terms = _grams(title + " " + " ".join(item["signals"]) + " " + " ".join(method["trigger"] for method in item["methods"]))
            score = len(query & terms) + 3 * len(query & _grams(title))
            candidates.append((score, -rank, item))
        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
        paths = [HIGH_ENTRY, HIGH_OUTPUT]
        if any(signal in str(question or "") for signal in JUNIOR_SIGNALS):
            paths.append(JUNIOR)
        if candidates and candidates[0][0] > 0:
            topic = candidates[0][2]["topic"]
            paths.append(candidates[0][2]["path"])

    chunks, sources = [], []
    for relative in paths:
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Test10 source escapes library")
        content = path.read_text(encoding="utf-8-sig")
        sources.append({"path": relative, "sha256": hashlib.sha256(content.encode()).hexdigest()})
        chunks.append(content)
    stage = {7: "初一", 8: "初二", 9: "初三", 10: "高一", 11: "高二", 12: "高三"}[grade]
    stage += "上" if term.endswith("1") else "下"
    text = "\n【Test10提示词工程】\nstage=" + stage + "; output_mode=json\n" + "\n\n".join(chunks)
    text += "\n【接入约束】只输出Test9现有l1,l2,l3,l4,l5,f,pat,s,u结构，不输出路由、方法卡或Markdown。所选学期及已学清单是知识上限，严格遵守较窄范围，不因命中提示词放宽；不输出最终答案。信息不足时说明，不编造题设或方法。\n"
    if is_junior:
        text += "初中提示词为候选稿，不能据此认定已完成教师审核或知识点路由；不得把示例条件当作本题条件。\n"
    return {"status": "loaded", "text": text, "sources": sources, "term": term,
            "topic": topic, "revision": hashlib.sha256(text.encode()).hexdigest()}
