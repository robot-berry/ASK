"""Run OCR passes until every PDF in the manifest is scanned end to end.

Each pass runs `ocr_pdf.py` in N shards; `ocr_pdf` resumes a PARTIAL file from the
page it stopped at, so repeating passes closes the remaining pages. Stops early
when no file is left unfinished (or after --rounds passes).

Usage:  ocr_rounds.py [--rounds 20] [--shards 6] [--cap 60]
"""
import argparse
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


def progress():
    """(pdf files, pages done, pages total, unfinished files) over all OCR outputs."""
    files = done = total = unfinished = 0
    for fn in os.listdir(OUT):
        if not fn.endswith('.txt') or fn.startswith('ocr_log'):
            continue
        files += 1
        txt = open(os.path.join(OUT, fn), encoding='utf-8', errors='ignore').read()
        nums = [int(m) for m in re.findall(r'^=== P(\d+) ===$', txt, re.M)]
        m = re.search(r'stopped at page (\d+)/(\d+)', txt)
        got = max(nums) if nums else 0
        tot = int(m.group(2)) if m else got
        done += got
        total += tot
        if m:
            unfinished += 1
    return files, done, total, unfinished


def todo_left():
    items = json.load(open(MAN, encoding='utf-8'))
    return [x for x in items
            if x['subject'] == '高中' and x['kind'] == 'pdf' and x['status'] == 'ocr']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=20)
    ap.add_argument('--shards', type=int, default=6)
    ap.add_argument('--cap', type=int, default=60)
    ap.add_argument('--threads', type=int, default=2,
                    help='ONNX intra-op threads per worker')
    ap.add_argument('--first', action='store_true',
                    help='forward --first: finish already-started files first')
    ap.add_argument('--only', action='append', default=[],
                    help='forward --only: restrict to paths containing this substring')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    for rnd in range(1, args.rounds + 1):
        files, done, total, unfinished = progress()
        print(f'round {rnd}: {files} files, {done}/{total} pages, {unfinished} unfinished',
              flush=True)
        if unfinished == 0 and files:
            print('all scanned', flush=True)
            return
        procs = []
        for i in range(args.shards):
            cmd = [sys.executable, '-I', os.path.join(ROOT, '_tools', 'ocr_pdf.py'),
                   '--shards', str(args.shards), '--shard', str(i),
                   '--cap', str(args.cap), '--threads', str(args.threads),
                   '--tag', f'r{rnd}s{i}']
            if args.first:
                cmd.append('--first')
            for k in args.only:
                cmd += ['--only', k]
            t = str(max(args.threads, 1))
            env = dict(os.environ, OMP_NUM_THREADS=t, MKL_NUM_THREADS=t,
                       OPENBLAS_NUM_THREADS=t)
            procs.append(subprocess.Popen(cmd, cwd=ROOT, env=env))
        for p in procs:
            p.wait()
        time.sleep(2)
    files, done, total, unfinished = progress()
    print(f'stopped after {args.rounds} rounds: {done}/{total} pages, {unfinished} unfinished',
          flush=True)


if __name__ == '__main__':
    main()
