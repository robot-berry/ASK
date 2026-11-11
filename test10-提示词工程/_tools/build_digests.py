"""Build two compact digest types per semester from the raw outlines.

tree_<sem>.txt  : heading hierarchy only (knowledge-point inventory)
ideas_<sem>.txt : technique headings + 【点睛】/方法总结 lines (solving ideas)
"""
import os
import re
import glob

OUT = r'D:\robot\Documents\ASK\picture\test10-提示词工程\_extract'
SEMS = ['gaoyishang', 'gaoyixia', 'gaoershang', 'gaoerxia', 'gaosan']

TECH = re.compile(r'(技法|题型|专题|解题|求法|步骤|模板|方法|技巧|考点|考向|类型|模型|大招|策略)')
IDEA = re.compile(r'(点睛|点评|关键点|解题思路|解题技巧|方法总结|通法|答题模板|规律总结)')
NUMHEAD = re.compile(r'^##\s*[一二三四五六七八九十\d]+[、.．]')


def read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read().splitlines()


def main():
    for sem in SEMS:
        tree, ideas = [], []
        for path in sorted(glob.glob(os.path.join(OUT, f'out_{sem}_*.txt'))):
            lines = read(path)
            src = lines[0].replace('SOURCE: ', '').strip() if lines else ''
            short = os.path.basename(src)[:70]
            tree.append(f'\n=== {short} ===')
            ideas.append(f'\n=== {short} ===')
            for line in lines[1:]:
                s = line.strip()
                if not s:
                    continue
                if s.startswith('## '):
                    h = s[3:].strip()
                    tree.append(h)
                    if TECH.search(h) or NUMHEAD.match(s):
                        ideas.append('# ' + h)
                elif s.startswith('· '):
                    body = s[2:].strip()
                    if IDEA.search(body):
                        ideas.append('  - ' + body[:220])
        for name, rows in [('tree', tree), ('ideas', ideas)]:
            dst = os.path.join(OUT, f'{name}_{sem}.txt')
            with open(dst, 'w', encoding='utf-8') as fh:
                fh.write('\n'.join(rows))
            print(name, sem, sum(len(r) for r in rows))


if __name__ == '__main__':
    main()
