"""Tidy generated markdown: collapse >2 blank lines, strip trailing blank lines."""
import os

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程\数学\高中'

n = 0
for dp, dn, fns in os.walk(ROOT):
    for fn in fns:
        if not fn.endswith('.md'):
            continue
        p = os.path.join(dp, fn)
        txt = open(p, encoding='utf-8').read()
        lines = txt.splitlines()
        out = []
        for line in lines:
            if line.strip() == '' and out and out[-1].strip() == '':
                continue
            out.append(line.rstrip())
        new = '\n'.join(out).strip() + '\n'
        if new != txt:
            open(p, 'w', encoding='utf-8').write(new)
            n += 1
print(f'{n} files tidied')
