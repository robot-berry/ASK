"""Build per-semester outlines from high school docx sources.

- Walks the archive with os.walk (glob chokes on literal [] in folder names).
- Opens long Windows paths with the \\?\ prefix.
- Keeps headings, the first paragraph after each heading, and 【点睛】 lines.

Outputs _extract/out_<semester>_<n>.txt and prints a manifest.
"""
import os
import re
import zipfile

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
QK = os.path.join(ROOT, '题库')
OUT = os.path.join(ROOT, '_extract')

HEAD = re.compile(
    r'^(技法\s*\d|题型\s*\d|专题\s*[一二三四五六七八九十\d]+|第[一二三四五六七八九十\d]+[章节]|'
    r'模块[一二三四五六七八九十\d]+|[一二三四五六七八九十]+、|\d+\.\d+(\.\d+)?\s*\D|'
    r'\d+[、．]?\s*(第)?[一二三四五六七八九十]|考点\s*\d|考向\s*\d|类型\s*[一二三四五六七八九十\d]+|'
    r'微专题|方法\s*[一二三四五六七八九十\d]+|大招\s*[一二三四五六七八九十\d]+)'
)
KEEP = re.compile(r'(点睛|关键点|解题思路|解题技巧|方法总结|通法|答题模板|规律总结)')


def long_path(p):
    p = os.path.abspath(p)
    if p.startswith('\\\\?\\'):
        return p
    if len(p) >= 240:
        if p.startswith('\\\\'):
            return '\\\\?\\UNC\\' + p[2:]
        return '\\\\?\\' + p
    return p


def docx_text(path):
    with zipfile.ZipFile(long_path(path)) as z:
        xml = z.read('word/document.xml').decode('utf-8', 'replace')
    xml = xml.replace('<w:tab/>', '\t').replace('<w:br/>', '\n')
    xml = re.sub(r'</w:p>', '\n', xml)
    text = re.sub(r'<[^>]+>', '', xml)
    text = text.replace('false', '□')
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


EXAMPLE = re.compile(
    r'^(例\s*\d|变式|【答案】|【详解】|【分析】|【解析】|【小问|【题型|'
    r'\d+\s*[．（(]|故选|答案[:：])'
)


def outline(text, intro_len=120, follow=4):
    lines = [p.strip() for p in text.split('\n') if p.strip()]
    out = []
    for i, p in enumerate(lines):
        if HEAD.match(p) and len(p) <= 120:
            out.append('\n## ' + p)
            kept = 0
            for q in lines[i + 1:i + 1 + follow * 3]:
                if HEAD.match(q) or EXAMPLE.match(q):
                    continue
                out.append(q[:intro_len])
                kept += 1
                if kept >= follow:
                    break
        elif KEEP.search(p):
            out.append('· ' + p[:260])
    # de-duplicate consecutive repeats
    dedup = []
    for line in out:
        if not dedup or line != dedup[-1]:
            dedup.append(line)
    return '\n'.join(dedup)


def find_files(subnames, name_filter=None):
    """Return docx paths whose relative path contains any of subnames."""
    found = []
    base = os.path.join(QK, '蒋老师高一数学+高二')
    for dirpath, _dirs, files in os.walk(base):
        for f in files:
            if not f.lower().endswith('.docx'):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), base)
            if any(s in rel for s in subnames) and (name_filter is None or name_filter in f):
                found.append(os.path.join(dirpath, f))
    return sorted(found)


def topic_files():
    base = os.path.join(QK, '高中数学各专题解题技巧', '高中数学各专题解题技巧')
    found = []
    for dirpath, _dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith('.docx') and '解析版' in f:
                found.append(os.path.join(dirpath, f))
    return sorted(found)


def topic_no(path):
    m = re.search(r'题型\s*0*(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else -1


SEMESTERS = {
    'gaoyishang': {
        'teacher': ['高一题型方法一本通上_docx'],
        'topics': [1, 2, 3, 4, 5, 6, 10, 11],
    },
    'gaoyixia': {
        'teacher': ['高一题型方法一本通下册_docx'],
        'topics': [12, 13, 14, 19],
    },
    'gaoershang': {
        'teacher': ['选择性必修上册_docx'],
        'topics': [20, 21, 22, 23, 24],
    },
    'gaoerxia': {
        'teacher': ['选择性必修下册_docx'],
        'topics': [7, 8, 9, 15, 16, 17, 18, 25, 26, 27],
    },
}


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    topics = {topic_no(p): p for p in topic_files()}
    for sem, cfg in SEMESTERS.items():
        files = find_files(cfg['teacher']) + [topics[n] for n in cfg['topics'] if n in topics]
        for i, src in enumerate(files):
            try:
                t = docx_text(src)
            except Exception as exc:  # noqa: BLE001
                manifest.append(f'FAIL {sem} {src} {exc}')
                continue
            o = outline(t)
            dst = os.path.join(OUT, f'out_{sem}_{i:02d}.txt')
            with open(dst, 'w', encoding='utf-8') as fh:
                fh.write(f'SOURCE: {src}\n\n' + o)
            manifest.append(f'{sem} {i:02d} full={len(t):>6} outline={len(o):>6} {os.path.basename(src)[:60]}')

    # 高三: outline only from the two big model books
    extra = [
        os.path.join(QK, '【A049】高考数学解题模型一本通', '高考数学解题模型一本通 解析册.docx'),
        os.path.join(QK, '王文勇高考数学大招集锦【26-25】', '王文勇《高考数学大招集锦》2026版（技法篇）完整版.docx'),
    ]
    for i, src in enumerate(extra):
        try:
            t = docx_text(src)
        except Exception as exc:  # noqa: BLE001
            manifest.append(f'FAIL gaosan {src} {exc}')
            continue
        o = outline(t)
        dst = os.path.join(OUT, f'out_gaosan_{i:02d}.txt')
        with open(dst, 'w', encoding='utf-8') as fh:
            fh.write(f'SOURCE: {src}\n\n' + o)
        manifest.append(f'gaosan {i:02d} full={len(t):>6} outline={len(o):>6} {os.path.basename(src)[:60]}')

    print('\n'.join(manifest))


if __name__ == '__main__':
    main()
