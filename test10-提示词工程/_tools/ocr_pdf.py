"""OCR PDFs listed in 数学/高中/提取进度.json whose status == 'ocr'.

Usage:  ocr_pdf.py [--limit N] [--sem 高三/上] [--only KW] [--first] [--jobs 1]
Output: 题库/_ocr/<rel>.txt  (one text file per PDF, UTF-8)
Resumable: existing non-empty outputs are skipped.

--only KW   keep only files whose path contains KW (repeatable); use it to finish one
            batch first (e.g. --only 解析版) instead of waiting for the whole queue.
--first     put unfinished files (PARTIAL output) ahead of new ones, round-robin across
            sources. A scan usually yields the 解析版 first pages first; finishing those
            files gives usable solutions without waiting for every source to be touched.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
import unicodedata

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
QK = os.path.join(ROOT, '题库')
OUT = os.path.join(QK, '_ocr')
MAN = os.path.join(ROOT, '数学', '高中', '提取进度.json')
START = 10  # minimum pages per file per visit


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
    safe = rel.replace('/', '__')
    return os.path.join(OUT, safe + '.txt')


def has_text_layer(path):
    r = subprocess.run(['pdftotext', '-l', '3', long_path(path), '-'],
                       capture_output=True, timeout=120)
    return r.returncode == 0 and len(r.stdout.strip()) > 40


HEADER = ('# 来源：题库/{rel}\n'
          '# OCR：RapidOCR（PP-OCRv4 模型），200dpi 灰度；公式与图形识别不可靠，仅作检索用。\n\n')


def read_pages(dst):
    """Existing page blocks in an OCR output -> (pages_done, page_total, [(page, text)]).

    pages_done is the highest page number already written; page_total is from the
    PARTIAL marker (0 when the file has none).
    """
    if not os.path.exists(dst):
        return 0, 0, []
    txt = open(dst, encoding='utf-8', errors='ignore').read()
    pages = []
    cur = None
    buf = []
    for line in txt.splitlines():
        m = re.match(r'^=== P(\d+) ===\s*$', line)
        if m:
            if cur is not None:
                pages.append((cur, '\n'.join(buf)))
            cur = int(m.group(1))
            buf = []
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
    total = int(m.group(2)) if m else 0
    return (max((p for p, _ in pages), default=0), total, pages)


def write_pages(rel, dst, pages, total_pages):
    os.makedirs(OUT, exist_ok=True)
    have = max((p for p, _ in pages), default=0)
    with open(dst, 'w', encoding='utf-8') as fh:
        fh.write(HEADER.format(rel=rel))
        for n, body in sorted(pages):
            fh.write(f'=== P{n} ===\n{body}\n')
        if total_pages and have < total_pages:
            fh.write(f'=== PARTIAL: stopped at page {have}/{total_pages} ===\n')
    return have >= total_pages if total_pages else True


def ocr_one(engine, rel, cap=60):
    """OCR a PDF until `cap` seconds have passed, but always at least START pages."""
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
        txt = '\n'.join(line[1] for line in (res or []))
        pages.append((start + 1, txt))
        start += 1
        new += 1
        write_pages(rel, dst, pages, doc.page_count)  # heartbeat: survives a kill
        if new >= START and time.time() - t0 > cap:
            break
    total_pages = doc.page_count
    doc.close()
    complete = write_pages(rel, dst, pages, total_pages)
    return 'done' if complete else 'ocr'


def partial_first(todo):
    """Unfinished files ahead of untouched ones, interleaved by source folder.

    The scanner stops after ~10 pages per file, so every 解析版 PDF holds its 知识梳理
    and the first few 题型. Finishing those files (rather than spreading thin over
    hundreds of new ones) is what turns OCR text into usable 解题思路 material.
    """
    left, fresh = collections.defaultdict(list), collections.defaultdict(list)
    for it in todo:
        done, _, _ = read_pages(out_path(it['rel']))
        bucket = left if done else fresh
        bucket[it['rel'].split('/')[0]].append(it)
    out = []
    queues = [left[k] for k in sorted(left)] + [fresh[k] for k in sorted(fresh)]
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
    items = json.load(open(MAN, encoding='utf-8'))
    todo = []
    for it in items:
        if it['subject'] != '高中':
            continue
        if '初中' in it['rel'] or '小学' in it['rel']:
            continue
        if args.sem and it['sem'] != args.sem:
            continue
        if it['status'] == 'ocr':
            todo.append(it)
        elif it['kind'] == 'pdf' and has_text_layer(os.path.join(QK, it['rel'])):
            todo.append(it)
    if args.only:
        todo = [x for x in todo if all(k in x['rel'] for k in args.only)]
    if args.limit:
        todo = todo[:args.limit]
    if args.first:
        todo = partial_first(todo)
    if args.shards > 1:
        todo = [x for i, x in enumerate(todo) if i % args.shards == args.shard]

    from rapidocr_onnxruntime import RapidOCR
    # default (-1) lets onnxruntime spawn ~24 threads per worker: with 6 workers the
    # machine oversubscribes 24 cores ~6x and every page slows to a crawl.
    engine = RapidOCR(intra_op_num_threads=args.threads,
                      inter_op_num_threads=1)
    ok = fail = 0
    os.makedirs(OUT, exist_ok=True)
    log_name = 'ocr_log.txt' if not args.tag else f'ocr_log_{args.tag}.txt'
    log = open(os.path.join(OUT, log_name), 'a', encoding='utf-8')
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
    ap.add_argument('--sem', default='')
    ap.add_argument('--cap', type=int, default=60, help='seconds per PDF')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--only', action='append', default=[],
                    help='keep only paths containing this substring (repeatable)')
    ap.add_argument('--first', action='store_true',
                    help='unfinished files before untouched ones')
    ap.add_argument('--tag', default='')
    ap.add_argument('--threads', type=int, default=2,
                    help='ONNX intra-op threads per worker (default 2; 6 workers x 2 = 12)')
    args = ap.parse_args()
    if os.environ.get('OCR_ENV') == '1':
        run(args)
        return
    # re-launch inside the OCR venv with the same arguments
    t = str(max(args.threads, 1))
    env = dict(os.environ, OCR_ENV='1', PYTHONIOENCODING='utf-8',
               OMP_NUM_THREADS=t, MKL_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t)
    py = os.path.join(ROOT, '_tools', 'ocr-env', 'Scripts', 'python.exe')
    sys.exit(subprocess.call([py] + sys.argv, env=env))


if __name__ == '__main__':
    main()
