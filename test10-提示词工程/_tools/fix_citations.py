"""Rewrite 依据 blocks with exact 题库 file names (no bare '…' placeholders)."""
import os
import re

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
QK = os.path.join(ROOT, '题库')
HS = os.path.join(ROOT, '数学', '高中')

J1 = '题库/蒋老师高一数学+高二/[格式docx]高一题型方法一本通上_docx/'
J2 = '题库/蒋老师高一数学+高二/[格式docx]抖音高一题型方法一本通下册_docx/'
JB1 = '题库/蒋老师高一数学+高二/[精排精校Word版]F0259-抖音蒋数学-高二题型方法一本通选择性必修上册/[精排精校Word版]F0259-抖音蒋数学-高二题型方法一本通选择性必修上册_docx/'
JB2 = '题库/蒋老师高一数学+高二/[精排精校Word版]F0260-抖音蒋数学-高二题型方法一本通选择性必修下册/[精排精校Word版]F0260-抖音蒋数学-高二题型方法一本通选择性必修下册_docx/'
ZT = '题库/高中数学各专题解题技巧/高中数学各专题解题技巧/'
GS1 = '题库/【A049】高考数学解题模型一本通/高考数学解题模型一本通 解析册.docx'
GS2 = '题库/王文勇高考数学大招集锦【26-25】/王文勇《高考数学大招集锦》2026版（技法篇）完整版.docx'

T = {
    '01': '题型01 不等式相关解题技巧（解析版）.docx',
    '02': '题型02 函数的4大基本性质解题技巧（解析版）.docx',
    '03': '题型03 “奇函数+常函数”的最大值+最小值解题技巧（解析版）.docx',
    '04': '题型04 函数图象问题解题技巧（解析版）.docx',
    '05': '题型05 4类比较函数值大小关系解题技巧（解析版）.docx',
    '06': '题型06 5类函数选填压轴题解题技巧（解析版）.docx',
    '07': '题型07 3类导数综合问题解题技巧（解析版）.docx',
    '08': '题型08 4类函数单调性与函数极值最值（解析版）.docx',
    '09': '题型09 8类导数大题综合（解析版）.docx',
    '10': '题型10 6类三角恒等变换解题技巧（解析版）.docx',
    '11': '题型11 4类三角函数选填解题技巧（解析版）.docx',
    '12': '题型12 5类平面向量解题技巧（解析版）.docx',
    '13': '题型13 6类解三角形公式定理解题技巧（解析版）.docx',
    '14': '题型14 4类解三角形大题综合（解析版）.docx',
    '15': '题型15 等差数列、等比数列的性质及其前n项和解题技巧（解析版）.docx',
    '16': '题型16 11类数列通项公式构造解题技巧（解析版）.docx',
    '17': '题型17 5类数列求和解题技巧（解析版）.docx',
    '18': '题型18 4类数列综合（解析版）.docx',
    '19': '题型19 10类球体的外接及内切解题技巧（解析版）.docx',
    '20': '题型20 6类立体几何大题解题技巧（解析版）.docx',
    '21': '题型21 3类对称与4类切线解题技巧（解析版）.docx',
    '22': '题型22 5类圆锥曲线解题技巧（解析版）.docx',
    '23': '题型23 6类圆锥曲线离心率问题解题技巧（解析版）.docx',
    '24': '题型24 5类圆锥曲线大题综合解题技巧（解析版）.docx',
    '25': '题型25 8类排列组合与4类二项式定理解题技巧（解析版）.docx',
    '26': '题型26 5类概率统计选填解题技巧（解析版）.docx',
    '27': '题型27 5类概率统计大题综合解题技巧（解析版）.docx',
}


ZT_DIR = os.path.join(ROOT, '题库', '高中数学各专题解题技巧')


def resolve_topics():
    """Map 题型 number -> real relative path of its 解析版 docx."""
    index = {}
    for dp, _dn, fns in os.walk(ZT_DIR):
        for fn in fns:
            m = re.match(r'题型(\d\d)', fn)
            if not fn.endswith('.docx') or not m:
                continue
            rel = '题库/' + os.path.relpath(os.path.join(dp, fn), QK).replace('\\', '/')
            num = m.group(1)
            score = 2 if '解析版' in fn else 1
            if num not in index or score > index[num][0]:
                index[num] = (score, rel)
    return {k: v[1] for k, v in index.items()}


RESOLVED = resolve_topics()


def z(*nums):
    out = []
    for n in nums:
        if n not in RESOLVED:
            raise SystemExit(f'题型{n} 未找到解析版文件')
        out.append(RESOLVED[n])
    return out


CIT = {
    '高一/上学期': {
        '00-必备数值': [J1 + '06高中数学这些常见的值要记住.docx'],
        '01-集合与逻辑': [J1 + '01第一章集合与常用逻辑用语.docx'],
        '02-一元二次函数方程与不等式': [J1 + '02第二章一元二次函数、方程与不等式.docx'] + z('01'),
        '03-函数的概念与性质': [J1 + '03第三章函数的概念与性质.docx'] + z('02', '03', '04', '06'),
        '04-指数与对数函数': [J1 + '04第四章指数函数与对数函数.docx'] + z('05'),
        '05-三角函数': [J1 + '05第五章：三角函数.docx'] + z('10', '11'),
    },
    '高一/下学期': {
        '01-平面向量': [J2 + '第六章：平面向量及其应用.docx（6.1—6.3 平面向量）'] + z('12'),
        '02-解三角形': [J2 + '第六章：平面向量及其应用.docx（6.4 余弦定理、正弦定理）'] + z('13', '14'),
        '03-立体几何初步': [J2 + '第八章立体几何初步.docx'] + z('19'),
        '04-统计': [J2 + '第九章统计.docx'],
        '05-概率': [J2 + '第十章概率.docx'] + z('26'),
        '06-复数': [J2 + '第七章复数.docx'],
    },
    '高二/上学期': {
        '01-空间向量与立体几何': [JB1 + '第一章空间向量与立体几何.docx'] + z('20'),
        '02-直线与圆': [JB1 + '第二章直线与圆的方程.docx'] + z('21'),
        '03-圆锥曲线': [JB1 + '第三章圆锥曲线的方程.docx'] + z('22', '23', '24'),
    },
    '高二/下学期': {
        '01-数列': [JB2 + '第四章数列.docx'] + z('15', '16', '17', '18'),
        '02-导数及其应用': [JB2 + '第五章一元函数的导数及其应用.docx'] + z('07', '08', '09'),
        '03-计数原理': [JB2 + '第六章计数原理.docx'] + z('25'),
        '04-随机变量与分布': [JB2 + '第七章随机变量及其分布.docx'] + z('26', '27'),
        '05-统计案例': [JB2 + '第八章成对数据的统计分析.docx'] + z('27'),
    },
    '高三/上学期': {
        '01-函数与导数': [GS1 + '（模块1 函数与导数）', GS2 + '（模块1、技法篇）'] + z('02', '03', '04', '05', '06', '07', '08', '09'),
        '02-不等式': [GS1 + '（模块2 不等式）', GS2 + '（模块2）'] + z('01'),
        '03-三角函数与解三角形': [GS1 + '（模块3 三角函数与解三角形）', GS2 + '（模块3）'] + z('10', '11', '13', '14'),
        '04-平面向量与复数': [GS1 + '（模块4 平面向量与复数）', GS2 + '（模块4）'] + z('12'),
        '05-立体几何': [GS1 + '（模块5 立体几何）', GS2 + '（模块5）'] + z('19', '20'),
        '06-数列': [GS1 + '（模块6 数列）', GS2 + '（模块6）'] + z('15', '16', '17', '18'),
        '07-概率统计': [GS1 + '（模块7 概率统计）', GS2 + '（模块7）'] + z('26', '27'),
        '08-解析几何': [GS1 + '（模块8 解析几何）', GS2 + '（模块8）'] + z('21', '22', '23', '24'),
        '09-集合逻辑与计数原理': [GS1 + '（模块9 集合、逻辑、计数原理）', GS2 + '（模块9）'] + z('25'),
    },
    '高三/下学期': {
        '01-选填策略': [GS2 + '（选填大招：特殊值、函数图象、选项分析）', GS1 + '（选填策略）'],
        '02-函数与导数压轴': [GS2 + '（函数与导数大招 1—18）'] + z('07', '09'),
        '03-不等式与代数变形': [GS2 + '（不等式与代数变形）'] + z('01'),
        '04-三角与解三角形压轴': [GS2 + '（三角函数大招 2—9）'] + z('11', '13', '14'),
        '05-向量与立体几何': [GS2 + '（向量大招 1—7；外接球大招 1—10；法向量叉乘）'] + z('12', '19'),
        '06-数列压轴': [GS2 + '（数列大招 1—13）'] + z('15', '16', '17', '18'),
        '07-概率统计': [GS2 + '（概率递推大招 2—3）'] + z('26', '27'),
        '08-圆锥曲线压轴': [GS2 + '（圆锥曲线大招 1—26）'] + z('22', '23', '24'),
        '09-新题型与应试': [GS1 + '（新题型与应试策略）'],
    },
}


BONUS = {
    ('高三/下学期', '03-不等式与代数变形'): [
        '题库/【A049】高考数学解题模型一本通/高考数学解题模型一本通 解析册.docx（模块2 不等式）'],
    ('高三/下学期', '09-新题型与应试'): [
        '题库/王文勇高考数学大招集锦【26-25】/王文勇《高考数学大招集锦》2026版（技法篇）完整版.docx（选填策略）'],
}


def patch(sem_dir, folder, cites):
    for name in os.listdir(sem_dir):
        pass
    tdir = os.path.join(sem_dir, folder)
    for fn in os.listdir(tdir):
        path = os.path.join(tdir, fn)
        if not fn.endswith('.md'):
            continue
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        lines = text.splitlines()
        out = []
        if fn == '解题思路.md':
            for line in lines:
                if line.startswith('- `题库/') or line.startswith('- `高中数学各专题解题技巧/') \
                        or line.startswith('- 新高考卷题型结构') \
                        or line.startswith('- 题库中无专门讲'):
                    continue
                out.append(line)
            # replace the 依据 block
            block = ['**依据（题库出处）**', '']
            block += ['- `' + c + '`' for c in cites]
            res = []
            inserted = False
            for line in out:
                if line.strip() == '**依据（题库出处）**':
                    res.extend(block)
                    inserted = True
                else:
                    res.append(line)
        else:
            res = []
            for line in lines:
                if line.startswith('> 依据：'):
                    primary = [c for c in cites if c.endswith('.docx')][0]
                    res.append('> 依据：`' + primary + '`（其余专题文件见同目录 `解题思路.md`）')
                else:
                    res.append(line)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(res).rstrip() + '\n')


def main():
    for sem, folders in CIT.items():
        sem_dir = os.path.join(HS, *sem.split('/'))
        for folder, cites in folders.items():
            patch(sem_dir, folder, cites + BONUS.get((sem, folder), []))
        print(f'{sem}: {len(folders)} folders patched')


if __name__ == '__main__':
    main()
