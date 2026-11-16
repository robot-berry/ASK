"""Which PDFs were touched in the last N minutes, and how complete the 解析版 set is.

Usage:  ocr_check.py [--minutes 2]
"""
import argparse
import os
import re
import sys
import time

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
OUT = os.path.join(ROOT, '题库', '_ocr')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=2)
    args = ap.parse_args()
    now = time.time()
    recent = []
    done = partial = 0
    pd = pt = 0
    for fn in os.listdir(OUT):
        if not fn.endswith('.txt') or fn.startswith('ocr_log'):
            continue
        p = os.path.join(OUT, fn)
        if now - os.path.getmtime(p) < args.minutes * 60:
            recent.append(fn[:-4].replace('__', '/'))
        if '解析版' not in fn or '一轮复习讲义' not in fn:
            continue
        txt = open(p, encoding='utf-8', errors='ignore').read()
        nums = [int(m) for m in re.findall(r'^=== P(\d+) ===$', txt, re.M)]
        m = re.search(r'stopped at page (\d+)/(\d+)', txt)
        got = max(nums) if nums else 0
        tot = int(m.group(2)) if m else got
        pd += got
        pt += tot
        partial += 1 if m else 0
        done += 0 if m else (1 if got else 0)
    print(f'解析版: {done} complete, {partial} partial, {pd}/{pt} pages')
    print(f'written in last {args.minutes:g} min: {len(recent)} files')
    for r in sorted(recent):
        print('  ', r)


if __name__ == '__main__':
    main()
