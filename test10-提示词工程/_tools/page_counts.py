"""Cache the true page count of every 高中 PDF (题库/_ocr/_pagecounts.json).

The progress dashboard needs page totals even for PDFs that have not been scanned
yet (their OCR output carries no PARTIAL marker, so the page count is unknown).
PyMuPDF reads the count in milliseconds, so this runs once and caches by path.

Usage:  python -I _tools/page_counts.py
"""
import json
import os
import subprocess
import sys

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
QK = os.path.join(ROOT, '题库')
OCR = os.path.join(QK, '_ocr')
MAN = os.path.join(ROOT, '数学', '高中', '提取进度.json')
CACHE = os.path.join(OCR, '_pagecounts.json')


def long_path(p):
    p = os.path.abspath(p)
    if len(p) < 240:
        return p
    if p.startswith('\\\\?\\'):
        return p
    if p.startswith('\\\\'):
        return '\\\\?\\UNC\\' + p[2:]
    return '\\\\?\\' + p


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    os.makedirs(OCR, exist_ok=True)
    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE, encoding='utf-8'))
    items = json.load(open(MAN, encoding='utf-8'))
    todo = [x for x in items
            if x['subject'] == '高中' and x['kind'] == 'pdf'
            and '初中' not in x['rel'] and '小学' not in x['rel']
            and x['rel'] not in cache]
    import fitz
    fail = 0
    for it in todo:
        try:
            doc = fitz.open(long_path(os.path.join(QK, it['rel'])))
            cache[it['rel']] = doc.page_count
            doc.close()
        except Exception as exc:  # noqa: BLE001
            fail += 1
            print(f'FAIL {it["rel"]} :: {exc!r}', flush=True)
    json.dump(cache, open(CACHE, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
    print(f'new {len(todo) - fail}, cached {len(cache)}, fail {fail}')


if __name__ == '__main__':
    if os.environ.get('OCR_ENV') == '1':
        main()
    else:
        env = dict(os.environ, OCR_ENV='1', PYTHONIOENCODING='utf-8')
        py = os.path.join(ROOT, '_tools', 'ocr-env', 'Scripts', 'python.exe')
        sys.exit(subprocess.call([py] + sys.argv, env=env))
