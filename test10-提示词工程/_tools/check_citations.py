"""Verify every 依据 path exists in 题库/ (ignores trailing （...） notes)."""
import os
import re

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
HS = os.path.join(ROOT, '数学', '高中')
QK = os.path.join(ROOT, '题库')


def long_path(p):
    if len(p) < 240:
        return p
    p = os.path.abspath(p)
    if p.startswith('\\\\?\\'):
        return p
    if p.startswith('\\\\'):
        return '\\\\?\\UNC\\' + p[2:]
    return '\\\\?\\' + p


def targets():
    for dp, _dn, fns in os.walk(HS):
        for fn in fns:
            if not fn.endswith('.md'):
                continue
            path = os.path.join(dp, fn)
            for line in open(path, encoding='utf-8'):
                if not (line.startswith('- `题库/') or line.startswith('> 依据')):
                    continue
                for m in re.finditer(r'((?:题库/)?[^`；\s][^`；]*?\.docx)', line):
                    raw = m.group(1).strip()
                    raw = re.sub(r'（[^）]*）\s*$', '', raw).strip()
                    if raw.startswith('题库/'):
                        raw = raw[3:]
                    yield os.path.relpath(path, ROOT), raw


def build_basename_index():
    idx = {}
    for dp, _dn, fns in os.walk(QK):
        for fn in fns:
            if fn.endswith('.docx'):
                key = re.match(r'(题型\d+|第[一二三四五六七八九十\d]+章)', fn)
                if key:
                    idx.setdefault(key.group(1), []).append(fn)
    return idx


def main():
    seen = {}
    lines = []
    bad = 0
    idx = build_basename_index()
    for src, rel in targets():
        if rel in seen:
            continue
        p = os.path.join(QK, rel.replace('/', os.sep))
        ok = os.path.exists(long_path(p))
        seen[rel] = ok
        if not ok:
            bad += 1
            lines.append(f'MISSING\t{rel}\t<- {src}')
            key = re.match(r'(题型\d+|第[一二三四五六七八九十\d]+章)', os.path.basename(rel))
            if key:
                for h in idx.get(key.group(1), [])[:4]:
                    lines.append(f'  real\t{h}')
    lines.append(f'{len(seen)} cited paths, {bad} missing')
    open(os.path.join(ROOT, '_tools', '_cite_check.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
    print(f'{len(seen)} cited paths, {bad} missing')


if __name__ == '__main__':
    main()
