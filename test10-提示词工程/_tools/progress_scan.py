"""Build extraction progress manifest from filesystem state.

Scans 题库/ for docx/pdf sources, classifies each into 初中/高中 × semester,
detects whether the text layer yields usable Chinese text, and writes
数学/高中/提取进度.json + 提取进度.md + 提取进度.html.
"""
import html
import json
import os
import re
import zipfile

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
QK = os.path.join(ROOT, '题库')
HS = os.path.join(ROOT, '数学', '高中')
OUT_JSON = os.path.join(HS, '提取进度.json')
OUT_MD = os.path.join(HS, '提取进度.md')
OUT_HTML = os.path.join(HS, '提取进度.html')

SEM_ORDER = ['高一/上', '高一/下', '高二/上', '高二/下', '高三/上', '高三/下']

HIGH_KW = re.compile(r'高中|高考|新高考|选择性必修|一轮|二轮|高一|高二|高三|必刷题|大招')
MID_KW = re.compile(r'初中|中考|初一|初二|初三|七年级|八年级|九年级')
SKIP_MID = re.compile(r'^kuake-001初中数学合集$|^数学全套练习$|^i4163$')

# folder name -> semester (checked in order; first match wins)
SEM_RULES = [
    (re.compile(r'高一题型方法一本通上'), '高一/上'),
    (re.compile(r'抖音高一题型方法一本通下册'), '高一/下'),
    (re.compile(r'F0259|选择性必修上册|选必上|必修[一二]|必修一\b'), '高二/上'),
    (re.compile(r'F0260|选择性必修下册|选必下|必修二\b'), '高二/下'),
    (re.compile(r'A049|考点|一轮'), '高三/上'),
    (re.compile(r'大招集锦'), '高三/下'),
]


def long_path(p):
    if len(p) < 240:
        return p
    p = os.path.abspath(p)
    if p.startswith('\\\\?\\'):
        return p
    if p.startswith('\\\\'):
        return '\\\\?\\UNC\\' + p[2:]
    return '\\\\?\\' + p


def docx_text_chars(path):
    """Return (total Chinese chars, formula-lost proxy: count of 'false')."""
    try:
        with zipfile.ZipFile(long_path(path)) as z:
            xml = z.read('word/document.xml').decode('utf-8', 'ignore')
    except Exception:
        return -1, 0
    text = re.sub(r'<[^>]+>', '', xml)
    cn = len(re.findall(r'[\u4e00-\u9fff]', text))
    return cn, text.count('false')


def pdf_chars(path):
    """Return text-layer Chinese char count, or -1 if unreadable."""
    import subprocess
    try:
        r = subprocess.run(['pdftotext', '-l', '3', long_path(path), '-'],
                           capture_output=True, timeout=60)
    except Exception:
        return -1
    if r.returncode != 0:
        return -1
    return len(re.findall(r'[\u4e00-\u9fff]', r.stdout.decode('utf-8', 'ignore')))


def classify(rel):
    top = rel.split('/')[0]
    if SKIP_MID.search(top):
        subject = '初中'  # only under a folder that itself says 初中
    elif MID_KW.search(top):
        subject = '初中'
    elif HIGH_KW.search(top):
        subject = '高中'
    elif MID_KW.search(rel):
        subject = '初中'
    else:
        subject = '高中'
    sem = None
    for rx, s in SEM_RULES:
        if rx.search(rel):
            sem = s
            break
    if sem is None:
        sem = '高三/上'
    return subject, sem


SKIP_TOPS = {'数学全套练习', 'i4163', 'kuake-001初中数学合集'}


def scan():
    items = []
    for dirpath, dirnames, filenames in os.walk(QK):
        dirnames[:] = [d for d in dirnames if not d.startswith('_')]
        rel_dir = os.path.relpath(dirpath, QK).replace('\\', '/')
        if rel_dir == '.':
            dirnames[:] = [d for d in dirnames if d not in SKIP_TOPS]
        for fn in filenames:
            low = fn.lower()
            if not (low.endswith('.docx') or low.endswith('.pdf')):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), QK).replace('\\', '/')
            subject, sem = classify(rel)
            kind = 'docx' if low.endswith('.docx') else 'pdf'
            items.append({'rel': rel, 'subject': subject, 'sem': sem,
                          'kind': kind,
                          'size': os.path.getsize(long_path(os.path.join(dirpath, fn)))})
    return items


def status_of(item):
    if item['kind'] == 'docx':
        cn, falsec = docx_text_chars(os.path.join(QK, item['rel']))
        if cn < 0:
            return 'error', 0
        if cn < 200:
            return 'ocr', cn
        return 'done', cn
    cn = pdf_chars(os.path.join(QK, item['rel']))
    if cn < 0:
        return 'error', 0
    if cn < 50:
        return 'ocr', cn
    return 'done', cn


def main():
    import sys
    items = scan()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if limit:
        items = items[:limit]
    for it in items:
        it['status'], it['text_cn'] = status_of(it)
    data = [{'rel': i['rel'], 'subject': i['subject'], 'sem': i['sem'],
             'kind': i['kind'], 'status': i['status'], 'text_cn': i['text_cn']} for i in items]
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=0)
    print(f'{len(data)} files scanned')


if __name__ == '__main__':
    main()
