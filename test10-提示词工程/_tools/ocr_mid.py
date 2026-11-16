"""OCR shim for 初中 sources: same engine and output format as _tools/ocr_pdf.py.

The original tool filters subject == '高中'; this one selects subject == '初中'
from the existing manifest (math/高中/提取进度.json, which already lists all 830
初中 files) and writes to the same 题库/_ocr/<rel>.txt location.

Usage:  ocr_mid.py [--limit N] [--cap SEC] [--only KW ...] [--first] [--tag NAME]
                   [--threads N] [--shard i --shards n]
Runs itself inside _tools/ocr-env. Resumable: finished files are skipped.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
QK = os.path.join(ROOT, '题库')
OUT = os.path.join(QK, '_ocr')
MAN = os.path.join(ROOT, '数学', '高中', '提取进度.json')
START = 10

HEADER = ('# 来源：题库/{rel}\n'
          '# OCR：RapidOCR（PP-OCRv4 模型），200dpi 灰度；公式与图形识别不可靠，仅作检索用。\n\n')


def long_path(p):
    p = os.path.abspath(p)
    if len(p) < 240:
        return p
    if p.startswith('\\\\?\\'):
        return p
    return '\\\\?\\' + p


def out_path(rel):
    return os.path.join(OUT, rel.replace('/', '__') + '.txt')


def read_pages(dst):
    if not os.path.exists(dst):
        return 0, 0, []
    txt = open(dst, encoding='utf-8', errors='ignore').read()
    pages, cur, buf = [], None, []
    for line in txt.splitlines():
        m = re.match(r'^=== P(\d+) ===\s*$', line)
        if m:
            if cur is not None:
                pages.append((cur, '\n'.join(buf)))
            cur, buf = int(m.group(1)), []
            continue
        if line.startswith('=== PARTIAL'):
            continue
        if line.startswith('# ') and cur is None:
            continue
        if cur is not None:
            buf.append(line)
    if cur is not None:
        pages.append((cur, '\n'.join(buf)))
    m = re.search(r'stopped at page (\d+)/(\d+)', txt)
    return (max((p for p, _ in pages), default=0), int(m.group(2)) if m else 0, pages)


def write_pages(rel, dst, pages, total):
    os.makedirs(OUT, exist_ok=True)
    have = max((p for p, _ in pages), default=0)
    with open(dst, 'w', encoding='utf-8') as fh:
        fh.write(HEADER.format(rel=rel))
        for n, body in sorted(pages):
            fh.write(f'=== P{n} ===\n{body}\n')
        if total and have < total:
            fh.write(f'=== PARTIAL: stopped at page {have}/{total} ===\n')
    return have >= total if total else True


def ocr_one(engine, rel, cap):
    import fitz
    src = os.path.join(QK, rel)
    dst = out_path(rel)
    done, total, pages = read_pages(dst)
    if done and total and done >= total:
        return 'skip'
    start = done if pages else 0
    doc = fitz.open(long_path(src))
    t0 = time.time()
    new = 0
    while start < doc.page_count:
        pix = doc[start].get_pixmap(dpi=200, colorspace=fitz.csGRAY)
        res, _ = engine(pix.tobytes('png'))
        pages.append((start + 1, '\n'.join(line[1] for line in (res or []))))
        start += 1
        new += 1
        write_pages(rel, dst, pages, doc.page_count)
        if new >= START and time.time() - t0 > cap:
            break
    total = doc.page_count
    doc.close()
    return 'done' if write_pages(rel, dst, pages, total) else 'ocr'


def partial_first(todo):
    left, fresh = collections.defaultdict(list), collections.defaultdict(list)
    for it in todo:
        done, _, _ = read_pages(out_path(it['rel']))
        (left if done else fresh)[it['rel'].split('/')[0]].append(it)
    out, queues = [], [left[k] for k in sorted(left)] + [fresh[k] for k in sorted(fresh)]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def run(args):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    todo = []
    for it in json.load(open(MAN, encoding='utf-8')):
        if it['subject'] != '初中':
            continue
        if it['kind'] == 'pdf':
            if args.only and not all(k in it['rel'] for k in args.only):
                continue
            done, total, _ = read_pages(out_path(it['rel']))
            if total and done >= total:
                continue
            todo.append(it)
    if args.limit:
        todo = todo[:args.limit]
    if args.first:
        todo = partial_first(todo)
    if args.shards > 1:
        todo = [x for i, x in enumerate(todo) if i % args.shards == args.shard]
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR(intra_op_num_threads=args.threads, inter_op_num_threads=1)
    ok = fail = 0
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, 'ocr_log_%s.txt' % (args.tag or 'mid')), 'a', encoding='utf-8')
    for n, it in enumerate(todo, 1):
        try:
            st = ocr_one(engine, it['rel'], args.cap)
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
    ap.add_argument('--cap', type=int, default=3600)
    ap.add_argument('--only', action='append', default=[])
    ap.add_argument('--first', action='store_true')
    ap.add_argument('--tag', default='')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    args = ap.parse_args()
    if os.environ.get('OCR_ENV') == '1':
        run(args)
        return
    t = str(max(args.threads, 1))
    env = dict(os.environ, OCR_ENV='1', PYTHONIOENCODING='utf-8',
               OMP_NUM_THREADS=t, MKL_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t)
    py = os.path.join(ROOT, '_tools', 'ocr-env', 'Scripts', 'python.exe')
    sys.exit(subprocess.call([py] + sys.argv, env=env))


if __name__ == '__main__':
    main()
