"""Load the unified Test10 prompt library with source provenance.

The centralized Test9 prompt project is preferred. External and legacy Test10
roots remain compatible fallback sources for deployments that need an override.
"""
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_CENTRAL_ROOT = _PROJECT_ROOT / "prompt_project" / "test10"
_PACKAGED_ROOT = _PROJECT_ROOT / "test10-提示词工程"
_SIBLING_ROOT = _PROJECT_ROOT.parent / "test10-提示词工程"


def _prompt_root() -> Path:
    """Resolve the Test10 prompt tree used by the running search service.

    ``TEST10_PROMPT_ROOT`` is intentionally first so a deployment can point at
    a maintained Test10 project directly. The centralized Test9 tree is the
    default, followed by external and legacy fallback trees.
    """
    configured = os.environ.get("TEST10_PROMPT_ROOT", "").strip()
    candidates = ([Path(configured).expanduser()] if configured else [])
    candidates.extend((_CENTRAL_ROOT, _SIBLING_ROOT, _PACKAGED_ROOT))
    for candidate in candidates:
        root = candidate.resolve()
        has_index = (root / "提示词" / "索引.json").is_file()
        has_stage_entries = all(
            (root / relative).is_file()
            for relative in (
                "提示词/数学/初中/00-搜题总入口.md",
                "提示词/数学/初中/00-统一输出结构.md",
                "提示词/数学/高中/00-搜题总入口.md",
                "提示词/数学/高中/00-统一输出结构.md",
            )
        )
        has_grade_only_layout = all(
            (root / "提示词" / "数学" / subject / grade).is_dir()
            for subject, grades in (
                ("初中", ("七年级", "八年级", "九年级")),
                ("高中", ("高一", "高二", "高三")),
            )
            for grade in grades
        )
        # The maintained Test10 checkout is intentionally allowed to contain
        # only the six grade directories plus one unified prompt file.  It no
        # longer needs a public entry, composer, knowledge-card tree, or index
        # file for runtime selection.
        if has_grade_only_layout:
            return root
        if has_index and (configured or has_stage_entries or candidate == _PACKAGED_ROOT):
            return root
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到 Test10 提示词库（已搜索：{searched}）")


ROOT = _prompt_root()
ORDER = {f"g{grade}_s{semester}": (grade - 7) * 2 + semester - 1
         for grade in range(7, 13) for semester in (1, 2)}
# New Test10 layout.  The old junior files below remain available as a
# fallback for a release bundle that has not yet been repacked.
JUNIOR_ENTRY = "提示词/数学/初中/00-搜题总入口.md"
JUNIOR_OUTPUT = "提示词/数学/初中/00-统一输出结构.md"
HIGH_ENTRY = "提示词/数学/高中/00-搜题总入口.md"
HIGH_OUTPUT = "提示词/数学/高中/00-统一输出结构.md"
UNIFIED_PROMPT = "提示词/统一数学提示词.md"
CENTRAL_UNIFIED_PROMPT = "../prompts/统一数学提示词.md"
JUNIOR_INDEX = "提示词/数学/初中/索引.json"
JUNIOR = "提示词/数学/初中/01_提示词/V11_数据库版本9_候选提示词.txt"
JUNIOR_SYSTEM = "提示词/数学/初中/01_提示词/请求附加_system提示词.txt"
# The old generic paths are retained as compatibility aliases for callers
# importing them.  Runtime selection below prefers the stage-specific paths.
GENERIC_ENTRY = "提示词/数学/00-搜题总入口.md"
GENERIC_OUTPUT = "提示词/数学/00-统一输出结构.md"
JUNIOR_SIGNALS = ("一元一次方程", "二元一次方程", "有理数", "整式", "勾股定理", "相似三角形", "全等三角形", "圆周角", "一次函数", "反比例函数")
_JUNIOR_ORDER = {"七上": 0, "七下": 1, "八上": 2, "八下": 3, "九上": 4, "九下": 5, "中考": 6}


def _grams(text):
    words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", str(text or "").lower())
    return {word[index:index + 2] for word in words for index in range(max(1, len(word) - 1))}


def _exists(relative):
    return (ROOT / relative).is_file()


def _unified_prompt_location():
    """Return the unified prompt reference and its physical file path.

    The centralized Test9 package keeps this file beside the seal prompt in
    ``prompt_project/prompts``.  Standalone Test10 checkouts keep the same
    file under their own ``提示词`` directory for backwards compatibility.
    """

    central_path = (_PROJECT_ROOT / "prompt_project" / "prompts" / "统一数学提示词.md").resolve()
    internal_path = (ROOT / UNIFIED_PROMPT).resolve()
    candidates = []
    if ROOT.resolve() == _CENTRAL_ROOT.resolve():
        candidates.append((CENTRAL_UNIFIED_PROMPT, central_path))
    candidates.append((UNIFIED_PROMPT, internal_path))
    if not any(path == central_path for _, path in candidates):
        candidates.append((CENTRAL_UNIFIED_PROMPT, central_path))
    for reference, path in candidates:
        if path.is_file():
            return reference, path
    return None


def _base_paths(is_junior):
    if _grade_only_layout():
        return []
    if is_junior and _exists(JUNIOR_ENTRY) and _exists(JUNIOR_OUTPUT):
        return [JUNIOR_ENTRY, JUNIOR_OUTPUT]
    if not is_junior and _exists(HIGH_ENTRY) and _exists(HIGH_OUTPUT):
        return [HIGH_ENTRY, HIGH_OUTPUT]
    return [GENERIC_ENTRY, GENERIC_OUTPUT]


def _load_index(relative):
    if not _exists(relative):
        return []
    data = json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))
    return data if isinstance(data, list) else data.get("topics", [])


def _grade_only_layout():
    """Whether ROOT is the cleaned six-grade authoring tree."""

    return all(
        (ROOT / "提示词" / "数学" / subject / grade).is_dir()
        for subject, grades in (
            ("初中", ("七年级", "八年级", "九年级")),
            ("高中", ("高一", "高二", "高三")),
        )
        for grade in grades
    )


def _grade_only_stage(term):
    grade = int(term[1:term.index("_")])
    group = "初中" if grade <= 9 else "高中"
    label = {7: "七年级", 8: "八年级", 9: "九年级",
             10: "高一", 11: "高二", 12: "高三"}[grade]
    semester = "上学期" if term.endswith("1") else "下学期"
    return group, label, semester


def _first_heading(text):
    match = re.search(r"(?m)^#\s+(.+?)\s*$", text)
    return match.group(1).strip() if match else ""


@lru_cache(maxsize=2)
def _unified_library():
    """Load the single maintained prompt file and its embedded topic blocks.

    The full file is the canonical prompt-engineering source.  Test9 receives
    only the header plus the topic blocks selected for the current term, so a
    single source file does not turn every request into a 380k-character model
    prompt.
    """

    location = _unified_prompt_location()
    if location is None:
        return None
    reference, prompt_path = location
    text = prompt_path.read_text(encoding="utf-8-sig")
    pattern = re.compile(
        r"(?ms)^<!-- TEST10_TOPIC_BEGIN (\{.*?\}) -->\s*\n"
        r"(.*?)\n<!-- TEST10_TOPIC_END -->"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    records = []
    for match in matches:
        metadata = json.loads(match.group(1))
        content = match.group(2).strip()
        grade_number = {
            "七年级": 7,
            "八年级": 8,
            "九年级": 9,
            "高一": 10,
            "高二": 11,
            "高三": 12,
        }[metadata["grade"]]
        semester_number = 1 if metadata["semester"] == "上学期" else 2
        records.append({
            **metadata,
            "path": metadata.get("source", ""),
            "content": content,
            "routing_text": _routing_text(content),
            "rank": (grade_number - 7) * 2 + semester_number - 1,
        })
    return {
        "path": reference,
        "absolute_path": str(prompt_path),
        "text": text,
        "header": text[:matches[0].start()].strip(),
        "records": records,
    }


def _routing_text(text):
    """Keep routing signals small and deterministic.

    Full prompt bodies contain many shared words (for example “函数” and
    “方法”), which made the old approximate router select unrelated topics.
    Prefer the title, explicit signal section, keyword line, and method trigger
    rows.  The complete file is still loaded after a topic is selected.
    """

    lines = text.splitlines()
    chunks = []
    in_false_positive_section = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if "容易误投" in stripped or "误投" in stripped:
            in_false_positive_section = True
            continue
        if in_false_positive_section and stripped.startswith("##"):
            in_false_positive_section = False
        if in_false_positive_section:
            continue
        if stripped.startswith("# "):
            chunks.append(stripped)
        if "本专题关键词" in stripped or "题面出现" in stripped or "属于本主题" in stripped:
            chunks.append(stripped)
        if stripped.startswith("-") and ("→" in stripped or "→" in stripped or "出现" in stripped):
            chunks.append(stripped)
        if re.match(r"^\|.*\|", stripped) and ("M" in stripped or "触发" in stripped):
            chunks.append(stripped)
        if stripped.startswith("### M"):
            chunks.append(stripped)
    return "\n".join(chunks)


def _grade_only_topics(term):
    """Return topic records available no later than *term*."""

    group, grade_label, semester = _grade_only_stage(term)
    grade_number = int(term[1:term.index("_")])
    ceiling = ORDER[term]
    unified = _unified_library()
    if unified:
        return [
            item
            for item in unified["records"]
            if item["rank"] <= ceiling
        ]
    root = ROOT / "提示词" / "数学" / group
    records = []
    for grade_dir in root.iterdir() if root.is_dir() else ():
        if not grade_dir.is_dir():
            continue
        grade_name = grade_dir.name
        grade_number_for_dir = {
            "七年级": 7, "八年级": 8, "九年级": 9,
            "高一": 10, "高二": 11, "高三": 12,
        }.get(grade_name)
        if grade_number_for_dir is None:
            continue
        for semester_dir in grade_dir.iterdir():
            if not semester_dir.is_dir() or semester_dir.name not in ("上学期", "下学期"):
                continue
            semester_number = 1 if semester_dir.name == "上学期" else 2
            rank = (grade_number_for_dir - 7) * 2 + semester_number - 1
            if rank > ceiling:
                continue
            for path in sorted(semester_dir.glob("*.md")):
                if path.name.lower() == "readme.md" or "接入说明" in path.name:
                    continue
                try:
                    content = path.read_text(encoding="utf-8-sig")
                except OSError:
                    continue
                topic = re.sub(r"^\d+[-_]?", "", path.stem)
                heading = _first_heading(content)
                heading_parts = [part.strip() for part in re.split(r"[·|｜]", heading) if part.strip()]
                if heading_parts:
                    if any("搜题提示词" in part for part in heading_parts[1:]):
                        topic = heading_parts[0]
                    else:
                        topic = heading_parts[-1]
                records.append({
                    "topic": topic,
                    "path": path.relative_to(ROOT).as_posix(),
                    "grade": grade_name,
                    "semester": semester_dir.name,
                    "rank": rank,
                    "routing_text": _routing_text(content),
                })
    return records


def _grade_only_topic(term, question):
    query = _grams(question)
    if not query:
        return None
    question_text = str(question or "")
    # Explicit theorem/topic words must outrank a broader parent chapter.
    # For example, “证明两个三角形全等” contains the broad word “三角形”,
    # but should select a 全等三角形 prompt when one is available.
    strong_tokens = (
        "全等", "勾股", "二次根式", "平行四边形", "一次函数", "二元一次",
        "不等式", "相交线", "平面直角坐标", "反比例函数", "锐角三角函数",
        "二次函数", "一元二次", "概率", "投影与视图", "相似", "旋转",
        "圆周角", "切线", "弦", "分式", "椭圆", "离心率", "圆锥曲线",
        "双曲线", "抛物线",
    )
    candidates = []
    for item in _grade_only_topics(term):
        title_score = len(query & _grams(item["topic"]))
        signal_score = len(query & _grams(item["routing_text"]))
        direct_title = 1 if item["topic"] and item["topic"] in str(question or "") else 0
        explicit_score = sum(30 for token in strong_tokens if token in question_text and token in item["topic"])
        if "圆锥曲线" in item["topic"] and any(
            token in question_text for token in ("椭圆", "双曲线", "抛物线", "离心率")
        ):
            explicit_score += 60
        score = explicit_score + direct_title * 20 + title_score * 4 + signal_score
        if score:
            candidates.append((score, direct_title, title_score, -item["rank"], item))
    candidates.sort(key=lambda value: (value[0], value[1], value[2], value[3]), reverse=True)
    return candidates[0][4] if candidates else None


def _junior_stage(term):
    grade = int(term[1:term.index("_")])
    return {7: "七", 8: "八", 9: "九"}[grade] + ("上" if term.endswith("1") else "下")


def _junior_topic(term, question):
    stage = _junior_stage(term)
    ceiling = _JUNIOR_ORDER[stage]
    query = _grams(question)
    candidates = []
    for item in _load_index(JUNIOR_INDEX):
        item_stage = item.get("grade", "")
        if item_stage not in _JUNIOR_ORDER or _JUNIOR_ORDER[item_stage] > ceiling:
            continue
        terms = _grams(item.get("chapter", "") + " " + " ".join(item.get("keywords", [])))
        score = len(query & terms) + 3 * len(query & _grams(item.get("chapter", "")))
        if score:
            candidates.append((score, -_JUNIOR_ORDER[item_stage], item))
    candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
    if not candidates:
        return None
    return candidates[0][2]


def _knowledge_card(relative):
    """Return a matching Test10 knowledge card when the new tree provides one."""
    path = Path(relative)
    if len(path.parts) < 4:
        return None
    # Prompt path: 数学/{初中|高中}/{grade}/{semester}/topic.md
    if path.parts[2] not in ("初中", "高中"):
        return None
    card_root = ROOT / "提示词" / "数学" / path.parts[2] / "知识卡"
    stem = path.stem
    matches = list(card_root.rglob(stem + "-知识卡.md"))
    if not matches:
        return None
    return matches[0].relative_to(ROOT).as_posix()


def build_context(subject, selection, question):
    term = (selection or {}).get("id", "")
    if subject != "math" or term not in ORDER:
        return {"status": "not_applicable", "text": "", "sources": []}

    grade = int(term[1:term.index("_")])
    is_junior = grade <= 9
    topic = None
    paths = _base_paths(is_junior)
    grade_only = _grade_only_layout()
    unified = _unified_library() if grade_only else None
    selected_sections = []
    if grade_only:
        if unified:
            paths = [unified["path"]]
        selected = _grade_only_topic(term, question)
        if selected:
            topic = selected["topic"]
            if unified:
                selected_sections.append(selected)
            else:
                paths.append(selected["path"])
        if not is_junior and any(signal in str(question or "") for signal in JUNIOR_SIGNALS):
            # High-school questions may explicitly require a junior tool
            # (for example solving a one-variable linear equation).  Reuse
            # the matching grade-only junior file without restoring a public
            # entry or knowledge-card tree.
            junior_selected = _grade_only_topic("g9_s2", question)
            if junior_selected:
                if unified:
                    if junior_selected["id"] not in {item["id"] for item in selected_sections}:
                        selected_sections.append(junior_selected)
                elif junior_selected["path"] not in paths:
                    paths.append(junior_selected["path"])
    elif is_junior:
        if _exists(JUNIOR_INDEX):
            selected = _junior_topic(term, question)
            if selected:
                topic = selected.get("chapter")
                paths.append("提示词/数学/初中/" + selected["path"])
                card = _knowledge_card(paths[-1])
                if card:
                    paths.append(card)
        else:
            # Legacy packaged copy: preserve its original candidate/system
            # prompt behavior until that bundle is repacked.
            paths.extend((JUNIOR, JUNIOR_SYSTEM))
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
        if any(signal in str(question or "") for signal in JUNIOR_SIGNALS):
            if _exists(JUNIOR_ENTRY) and _exists(JUNIOR_INDEX):
                # A high-school request may deliberately rely on a junior
                # method.  Reuse the maintained junior route and knowledge
                # card, while keeping the high-school output contract active.
                paths.append(JUNIOR_ENTRY)
                selected = _junior_topic("g9_s2", question)
                if selected:
                    junior_path = "提示词/数学/初中/" + selected["path"]
                    paths.append(junior_path)
                    card = _knowledge_card(junior_path)
                    if card:
                        paths.append(card)
            else:
                paths.append(JUNIOR)
        if candidates and candidates[0][0] > 0:
            topic = candidates[0][2]["topic"]
            paths.append(candidates[0][2]["path"])
            card = _knowledge_card(paths[-1])
            if card:
                paths.append(card)

    chunks, sources = [], []
    if grade_only and unified:
        unified_path = Path(unified["absolute_path"])
        unified_text = unified_path.read_text(encoding="utf-8-sig")
        sources.append({
            "path": unified["path"],
            "sha256": hashlib.sha256(unified_text.encode()).hexdigest(),
        })
        chunks.append(unified["header"])
        chunks.append(
            "## 当前运行范围\n"
            f"term={term}; 只允许 rank≤{ORDER[term]} 的专题；"
            "未命中的内部专题不得引用。"
        )
        if selected_sections:
            for item in selected_sections:
                chunks.append(
                    "## 当前命中专题\n"
                    f"id={item['id']}; topic={item['topic']}; "
                    f"grade={item['grade']}; semester={item['semester']}\n\n"
                    + item["content"]
                )
        else:
            chunks.append("## 当前命中专题\n题面暂未匹配到可靠专题，只能依据题面已确认信息给出范围内澄清问题。")
    else:
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
    text += (
        "【最终五层覆盖规则】五层按学生当前卡点进入，不按学霸/中等/学渣身份命名："
        "L1只问=读完题想试但不知道第一步，只问一个不超过60字的元问题；"
        "L2指向=知道目标但不知道先看哪两个量或条件，写1到3条、每条不超过80字且不说关系、不写公式或步骤；"
        "L3点破=看到了量但没有连上结构关系，写1到2条、每条不超过100字且不展开计算；"
        "L4搭半桥=知道关系却卡在执行，固定3句并用（　）或————留关键空位；"
        "L5兜底详解=没有路线或两次尝试仍卡住，固定5句，最后用空位或请你完成最后一步收口。"
        "高年级可使用低年级知识，初中不得使用高中知识；所有层都服从stage、learned和教材进度。\n"
    )
    if is_junior:
        text += "初中提示词为候选稿，不能据此认定已完成教师审核或知识点路由；不得把示例条件当作本题条件。\n"
    result = {"status": "loaded", "text": text, "sources": sources, "term": term,
              "topic": topic, "revision": hashlib.sha256(text.encode()).hexdigest()}
    if grade_only:
        result["layout"] = "grade_only"
        result["prompt_file"] = unified["path"] if unified else None
        result["selected_topics"] = [
            {key: item[key] for key in ("id", "topic", "group", "grade", "semester", "path")}
            for item in selected_sections
        ]
    return result

