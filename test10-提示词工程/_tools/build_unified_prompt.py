import json
import re
from pathlib import Path

project = Path(__file__).resolve().parents[1]
prompt_root = project / '提示词'
math_root = prompt_root / '数学'
compact_path = project.parents[1] / 'picture/test9-提示词优化/math/source/数学题解析紧凑提示词.md'
out = prompt_root / '统一数学提示词.md'


def first_heading(text):
    match = re.search(r'(?m)^#\s+(.+?)\s*$', text)
    return match.group(1).strip() if match else ''


def topic_name(path, text):
    topic = re.sub(r'^\d+[-_]?', '', path.stem)
    heading = first_heading(text)
    parts = [part.strip() for part in re.split(r'[·|｜]', heading) if part.strip()]
    if parts:
        topic = parts[0] if any('搜题提示词' in part for part in parts[1:]) else parts[-1]
    return topic


def stage_info(path):
    group, grade, semester = path.relative_to(math_root).parts[:3]
    grade_number = {'七年级': 7, '八年级': 8, '九年级': 9,
                    '高一': 10, '高二': 11, '高三': 12}[grade]
    semester_number = 1 if semester == '上学期' else 2
    return group, grade, semester, grade_number, semester_number


files = [path for path in math_root.rglob('*.md')
         if path.parent.name in ('上学期', '下学期')]
files.sort(key=lambda path: (stage_info(path)[3], stage_info(path)[4],
                             path.relative_to(math_root).as_posix()))
compact = compact_path.read_text(encoding='utf-8-sig').strip()
header = '''# Test10 / Test9 统一数学提示词

## 0. 这份文件的职责

这是数学搜题的唯一统一提示词正文。它把初中七、八、九年级和高中高一、二、三的知识点、解题思路和方法卡放在同一个内部提示词库中，并保留 Test9 的五层思路引导和 JSON 输出契约。

运行时输入 `subject`、`term`、`stage`、`learned`、题目 OCR、原图/图形证据和选定学期。先按 `term` 建立知识上限，再从本文件内部专题段中选择与题面最匹配的已学专题。未被选中的专题段只作为内部资料，不能被模型引用或泄露。

专题正文是知识与方法材料；若专题正文中的旧示例、旧格式或旧层级规则与本文件的 Test9 输出契约冲突，以本文件前面的统一规则和末尾的接入约束为准。

## 1. 系统角色与输入边界

你是当前 `term` 对应学段的数学老师，同时负责给 Test9 生成可逐步揭示的解题思路。先核对题干、原图、OCR、图形、选项、小问、定义域和取值范围，再决定是否能使用某个专题。显式 `term` 优先于题面中猜出的年级；学段不明确且会影响方法选择时，说明缺失信息并暂停越界推导。

- 初中：只能使用七至九年级当前学期及之前的知识；禁止调用高中方法。
- 高中：可以调用已经学过的初中工具，但必须标明它是基础工具，并仍服从当前高中学期上限。
- 题面出现后续章节术语，不等于学生已经学过；`learned`、教材进度和当前 `term` 是硬边界。
- 只选择一条已核对的主路线；确有必要时才补充一个前置专题，不并列未经核实的多解。
- 缺少图形、符号、选项或关键条件时，在 `s/u` 中如实列出，不按题型补造条件。

## 2. 六学段路由闸门

按 `term=g7_s1…g12_s2` 将可用专题限制在当前学期及之前。路由内部记录 `topic`、`method_id`、触发词、适用条件和来源；这些内部字段不能出现在最终 Test9 JSON 中。专题命中只表示可能相关，必须再次核对题面证据、公式条件和教材范围。

## 3. Test9 的统一输出正文

'''

parts = [header, compact,
         '\n\n## 4. 统一专题方法库（内部段落）\n\n'
         '每个专题段都带有稳定的 `id`、学段、学期和原始来源。运行时只抽取当前允许且命中的段落；不要把其他学段的内容带入思路。\n']
for index, path in enumerate(files, 1):
    text = path.read_text(encoding='utf-8-sig').strip()
    group, grade, semester, grade_number, semester_number = stage_info(path)
    metadata = {
        'id': f'g{grade_number}_s{semester_number}_{index:02d}',
        'group': group,
        'grade': grade,
        'semester': semester,
        'topic': topic_name(path, text),
        'source': path.relative_to(project).as_posix(),
    }
    parts.append('\n<!-- TEST10_TOPIC_BEGIN ' +
                json.dumps(metadata, ensure_ascii=False, separators=(',', ':')) +
                ' -->\n')
    parts.append(text)
    parts.append('\n<!-- TEST10_TOPIC_END -->\n')

parts.append('''
## 5. Test9 接入约束（优先级最高）

- 最终只输出一个 JSON 对象，根字段固定为 `l1,l2,l3,l4,l5,f,pat,s,u`，不输出 Markdown、角色前言、路由字段、专题名或方法卡调试信息。
- `l1` 每个原题小问恰好一个元问题；`l2` 只指出观察对象；`l3` 点破一个结构关系；`l4` 恰好三句并给出下一步具体动作；`l5` 恰好五句并给出最后的计算、判断或检验动作。
- 五层使用同一套已核实条件和同一条主路线，只增加信息，不更换方法，不泄露最终数值、选项、区间、最值位置或完整答案等式。
- `f` 只列本题实际使用且已核对适用条件的公式、定义或等价关系；`pat` 只概括本题主路线；`s=u` 时逐项列出影响结论的缺失条件。
- 公式必须使用合法 LaTeX；不得使用大学、竞赛或当前学段未学方法；不得根据示例条件补造本题条件。
- 专题正文中的旧版空白占位、旧版多问句或直接答案表达均不覆盖本接入约束。
''')

out.write_text(''.join(parts).rstrip() + '\n', encoding='utf-8')
print(json.dumps({'output': str(out), 'sections': len(files),
                   'chars': len(out.read_text(encoding='utf-8')),
                   'bytes': out.stat().st_size}, ensure_ascii=False))
