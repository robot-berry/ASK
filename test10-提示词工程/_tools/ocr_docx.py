"""OCR the embedded images of image-only docx sources (text layer < 200 chars).

These files (公式汇编、大招手卡) store every page as a picture, so 提取进度 lists
them as status 'ocr' with kind 'docx'. Output: 题库/_ocr/<rel>.txt

Usage:  ocr_docx.py [--limit N]
Resumable: existing non-empty outputs are skipped.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import zipfile

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
QK = os.path.join(ROOT, '题库')
OUT = os.path.join(QK, '_ocr')
MAN = os.path.join(ROOT, '数学', '高中', '提取进度.json')
MEDIA = re.compile(r'^word/media/[^/]+\.(png|jpe?g|bmp|gif|tiff?)$', re.I)


def long_path(p):
    p = os.path.abspath(p)
    if len(p) < 240:
        return p
    if p.startswith('\\\\?\\'):
        return p
    if p.startswith('\\\\'):
        return '\\\\?\\UNC\\' + p[2:]
    return '\\\\?\\' + p


def out_path(rel):
    return os.path.join(OUT, rel.replace('/', '__') + '.txt')


def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def ocr_docx(engine, rel):
    src = os.path.join(QK, rel)
    dst = out_path(rel)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return 'skip'
    with zipfile.ZipFile(long_path(src)) as z:
        names = sorted((n for n in z.namelist() if MEDIA.match(n)), key=natural_key)
        chunks = []
        for i, name in enumerate(names, 1):
            res, _ = engine(z.read(name))
            txt = '\n'.join(line[1] for line in (res or []))
            chunks.append(f'=== IMG{i} {name.split("/")[-1]} ===\n{txt}')
    os.makedirs(OUT, exist_ok=True)
    with open(dst, 'w', encoding='utf-8') as fh:
        fh.write(f'# 来源：题库/{rel}\n')
        fh.write(f'# OCR：RapidOCR（PP-OCRv4 模型），{len(names)} 张内嵌图片；公式识别不可靠，仅作检索用。\n\n')
        fh.write('\n'.join(chunks))
    return 'ocr'


def run(args):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    items = json.load(open(MAN, encoding='utf-8'))
    todo = [it for it in items
            if it['subject'] == '高中' and it['kind'] == 'docx' and it['status'] == 'ocr'
            and '初中' not in it['rel'] and '小学' not in it['rel']]
    if args.limit:
        todo = todo[:args.limit]
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, 'ocr_log_docx.txt'), 'a', encoding='utf-8')
    ok = fail = 0
    for n, it in enumerate(todo, 1):
        try:
            st = ocr_docx(engine, it['rel'])
            ok += 1
            line = f'{n}/{len(todo)} {st} {it["rel"]}'
        except Exception as exc:  # noqa: BLE001
            fail += 1
            line = f'{n}/{len(todo)} FAIL {it["rel"]} :: {exc!r}'
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()
    log.write(f'# done ok={ok} fail={fail}\n')
    log.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    if os.environ.get('OCR_ENV') == '1':
        run(args)
        return
    env = dict(os.environ, OCR_ENV='1', PYTHONIOENCODING='utf-8')
    py = os.path.join(ROOT, '_tools', 'ocr-env', 'Scripts', 'python.exe')
    sys.exit(subprocess.call([py] + sys.argv, env=env))


if __name__ == '__main__':
    main()
