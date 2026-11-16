"""Split semester 知识点清单/解题思路 into per-topic folders.

Each topic folder gets:
  - <章节名>.md   : knowledge points for that topic
  - 解题思路.md   : method blocks for that topic, with 依据 pointing into 题库

Idea blocks are assigned at ### level so mixed chapters can be split
(e.g. 统计与概率 -> 统计 / 概率).
"""
import os
import re
import shutil

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
HS = os.path.join(ROOT, '数学', '高中')
QK = '题库/'

# semester -> [(folder, knowledge heading prefix, idea rules)]
# idea rule: (## heading keyword, ### heading keyword or None for whole section)
PLAN = {
    '高一/上学期': [
        ('01-集合与逻辑', '第1章', [('集合', None)]),
        ('02-一元二次函数方程与不等式', '第2章', [('一元二次函数、方程和不等式', None)]),
        ('03-函数的概念与性质', '第3章', [('函数的概念与性质', None)]),
        ('04-指数与对数函数', '第4章', [('指数函数与对数函数', None)]),
        ('05-三角函数', '第5章', [('三角函数', None)]),
        ('00-必备数值', '本章外的必备数值', []),
    ],
    '高一/下学期': [
        ('01-平面向量', '第6章', [('平面向量', None)]),
        ('02-解三角形', '第7章 余弦', [('解三角形', None)]),
        ('03-立体几何初步', '第8章', [('立体几何', None)]),
        ('04-统计', '第9章', [('统计与概率', '抽样与统计图表')]),
        ('05-概率', '第10章', [('统计与概率', '古典概型')]),
        ('06-复数', '第7章 复数', [('复数', None)]),
    ],
    '高二/上学期': [
        ('01-空间向量与立体几何', '第1章', [('空间向量', None)]),
        ('02-直线与圆', '第2章', [('直线与圆', None)]),
        ('03-圆锥曲线', '第3章', [('圆锥曲线', None)]),
    ],
    '高二/下学期': [
        ('01-数列', '第4章', [('数列', None)]),
        ('02-导数及其应用', '第5章', [('导数', None)]),
        ('03-计数原理', '第6章', [('计数原理', None)]),
        ('04-随机变量与分布', '第7章', [('概率与统计', '条件概率'), ('概率与统计', '分布列')]),
        ('05-统计案例', '第8章', [('概率与统计', '统计案例')]),
    ],
    '高三/上学期': [
        ('01-函数与导数', '模块 1', [('函数与导数', None)]),
        ('02-不等式', '模块 2', [('不等式', None)]),
        ('03-三角函数与解三角形', '模块 3', [('三角函数', None)]),
        ('04-平面向量与复数', '模块 4', [('向量、复数', '平面向量')]),
        ('05-立体几何', '模块 5', [('向量、复数', '立体几何')]),
        ('06-数列', '模块 6', [('数列', None)]),
        ('07-概率统计', '模块 7', [('概率统计', None)]),
        ('08-解析几何', '模块 8', [('解析几何', None)]),
        ('09-集合逻辑与计数原理', '模块 9', [('集合、逻辑', None)]),
    ],
    '高三/下学期': [
        ('01-选填策略', '一、选填', [('选填策略', None)]),
        ('02-函数与导数压轴', '二、函数与导数', [('函数与导数', None)]),
        ('03-不等式与代数变形', '三、不等式', []),
        ('04-三角与解三角形压轴', '四、三角函数', [('三角函数', None)]),
        ('05-向量与立体几何', '五、向量与立体几何', [('向量与立体几何', None)]),
        ('06-数列压轴', '六、数列', [('概率与数列', '数列速算')]),
        ('07-概率统计', '七、概率统计', [('概率与数列', '概率递推')]),
        ('08-圆锥曲线压轴', '八、圆锥曲线', [('圆锥曲线', None)]),
        ('09-新题型与应试', '九、新题型', []),
    ],
}

# Primary 题库 source per topic folder (relative to 题库/).
S = {
    '01-集合与逻辑': '蒋老师高一数学+高二/[格式docx]高一题型方法一本通上_docx/01第一章集合与常用逻辑用语.docx',
    '02-一元二次函数方程与不等式': '蒋老师高一数学+高二/[格式docx]高一题型方法一本通上_docx/02第二章一元二次函数、方程与不等式.docx；高中数学各专题解题技巧 … 题型01（解析版）',
    '03-函数的概念与性质': '蒋老师高一数学+高二/[格式docx]高一题型方法一本通上_docx/03第三章函数的概念与性质.docx；高中数学各专题解题技巧 … 题型02—06（解析版）',
    '04-指数与对数函数': '蒋老师高一数学+高二/[格式docx]高一题型方法一本通上_docx/04第四章指数函数与对数函数.docx；高中数学各专题解题技巧 … 题型05（解析版）',
    '05-三角函数': '蒋老师高一数学+高二/[格式docx]高一题型方法一本通上_docx/05第五章：三角函数.docx；高中数学各专题解题技巧 … 题型10—11（解析版）',
    '00-必备数值': '蒋老师高一数学+高二/[格式docx]高一题型方法一本通上_docx/06高中数学这些常见的值要记住.docx',
    '01-平面向量': '蒋老师高一数学+高二/[格式docx]抖音高一题型方法一本通下册_docx/第六章：平面向量及其应用.docx（6.1—6.3）；高中数学各专题解题技巧 … 题型12（解析版）',
    '02-解三角形': '蒋老师高一数学+高二/[格式docx]抖音高一题型方法一本通下册_docx/第六章：平面向量及其应用.docx（6.4）；高中数学各专题解题技巧 … 题型13—14（解析版）',
    '03-立体几何初步': '蒋老师高一数学+高二/[格式docx]抖音高一题型方法一本通下册_docx/第八章立体几何初步.docx；高中数学各专题解题技巧 … 题型19（解析版）',
    '04-统计': '蒋老师高一数学+高二/[格式docx]抖音高一题型方法一本通下册_docx/第九章统计.docx',
    '05-概率': '蒋老师高一数学+高二/[格式docx]抖音高一题型方法一本通下册_docx/第十章概率.docx',
    '06-复数': '蒋老师高一数学+高二/[格式docx]抖音高一题型方法一本通下册_docx/第七章复数.docx',
    '01-空间向量与立体几何': '蒋老师高一数学+高二/[精排精校Word版]F0259-…/_docx/第一章空间向量与立体几何.docx；高中数学各专题解题技巧 … 题型20（解析版）',
    '02-直线与圆': '蒋老师高一数学+高二/[精排精校Word版]F0259-…/_docx/第二章直线与圆的方程.docx；高中数学各专题解题技巧 … 题型21（解析版）',
    '03-圆锥曲线': '蒋老师高一数学+高二/[精排精校Word版]F0259-…/_docx/第三章圆锥曲线的方程.docx；高中数学各专题解题技巧 … 题型22—24（解析版）',
    '01-数列': '蒋老师高一数学+高二/[精排精校Word版]F0260-…/_docx/第四章数列.docx；高中数学各专题解题技巧 … 题型15—18（解析版）',
    '02-导数及其应用': '蒋老师高一数学+高二/[精排精校Word版]F0260-…/_docx/第五章一元函数的导数及其应用.docx；高中数学各专题解题技巧 … 题型07—09（解析版）',
    '03-计数原理': '蒋老师高一数学+高二/[精排精校Word版]F0260-…/_docx/第六章计数原理.docx；高中数学各专题解题技巧 … 题型25（解析版）',
    '04-随机变量与分布': '蒋老师高一数学+高二/[精排精校Word版]F0260-…/_docx/第七章随机变量及其分布.docx；高中数学各专题解题技巧 … 题型26—27（解析版）',
    '05-统计案例': '蒋老师高一数学+高二/[精排精校Word版]F0260-…/_docx/第八章成对数据的统计分析.docx',
}
GS_PRIMARY = '【A049】高考数学解题模型一本通/高考数学解题模型一本通 解析册.docx'
GS_BIG = '王文勇高考数学大招集锦【26-25】/王文勇《高考数学大招集锦》2026版（技法篇）完整版.docx'


def read_lines(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read().splitlines()


def parse_blocks(lines, level2='## ', level3='### '):
    """Return list of (h2, h3 or None, body_lines)."""
    blocks = []
    cur2 = None
    cur3 = None
    body = []
    for line in lines:
        if line.startswith(level2):
            if cur3 is not None:
                blocks.append((cur2, cur3, body))
            elif cur2 is not None and body:
                blocks.append((cur2, None, body))
            cur2, cur3, body = line, None, []
        elif line.startswith(level3):
            if cur3 is not None:
                blocks.append((cur2, cur3, body))
            cur3, body = line, []
        else:
            if cur2 is not None:
                body.append(line)
    if cur3 is not None:
        blocks.append((cur2, cur3, body))
    elif cur2 is not None:
        blocks.append((cur2, None, body))
    return blocks


def pick(blocks, rules):
    if not rules:
        return []
    out = []
    for b in blocks:
        h2 = b[0] or ''
        h3 = b[1] or ''
        for kw2, kw3 in rules:
            if kw2 in h2 and (kw3 is None or kw3 in h3):
                out.append(b)
                break
    return out


def write_topic(sem_dir, folder, title, k_head, k_body, ideas, citation):
    tdir = os.path.join(sem_dir, folder)
    os.makedirs(tdir, exist_ok=True)
    kname = re.sub(r'^##\s*', '', k_head).strip()
    kpath = os.path.join(tdir, re.sub(r'[\\/:*?"<>|]', '-', kname) + '.md')
    with open(kpath, 'w', encoding='utf-8') as fh:
        fh.write(f'# {title} · 知识点\n\n')
        fh.write('> 依据：`' + QK + citation + '`\n\n')
        fh.write(f'## {kname}\n\n')
        fh.write('\n'.join(k_body).strip() + '\n')
    ipath = os.path.join(tdir, '解题思路.md')
    with open(ipath, 'w', encoding='utf-8') as fh:
        fh.write(f'# {title} · 解题思路\n\n')
        fh.write('**依据（题库出处）**\n\n')
        for part in citation.split('；'):
            fh.write(f'- `{QK}{part.strip()}`\n')
        fh.write('\n> 原始 docx 的公式为 MathType 对象，文本提取时丢失；公式按课程标准补全，'
                 '方法名称、适用条件与易错点来自原文。\n')
        if not ideas:
            fh.write('\n（该主题本轮未提取到独立思路条目，见同目录知识点与相邻章节。）\n')
        last_h2 = None
        for h2, h3, body in ideas:
            if h2 != last_h2:
                fh.write('\n' + h2 + '\n')
                last_h2 = h2
            if h3:
                fh.write('\n' + h3 + '\n')
            txt = '\n'.join(body).strip()
            if txt:
                fh.write('\n' + txt + '\n')


def main():
    for sem, topics in PLAN.items():
        sem_dir = os.path.join(HS, *sem.split('/'))
        kp = os.path.join(sem_dir, '知识点清单.md')
        ip = os.path.join(sem_dir, '解题思路.md')
        k_blocks = parse_blocks(read_lines(kp), level2='## ', level3='### ')
        i_blocks = parse_blocks(read_lines(ip), level2='## ', level3='### ')
        for folder, k_prefix, rules in topics:
            k_matches = [b for b in k_blocks if b[0] and b[0].startswith('## ' + k_prefix)]
            if not k_matches:
                k_head, k_body = '## ' + folder.split('-', 1)[1], []
            else:
                k_head = k_matches[0][0]
                k_body = []
                for _h, _h3, b in k_matches:
                    k_body.extend(b)
            ideas = pick(i_blocks, rules)
            citation = S.get(folder) or (GS_PRIMARY + '；' + GS_BIG if sem.startswith('高三') else '')
            write_topic(sem_dir, folder, folder.split('-', 1)[1], k_head, k_body, ideas, citation)
        archive = os.path.join(sem_dir, '_学期总览')
        os.makedirs(archive, exist_ok=True)
        shutil.move(kp, os.path.join(archive, '知识点清单（全）.md'))
        shutil.move(ip, os.path.join(archive, '解题思路（全）.md'))
        print(f'{sem}: {len(topics)} topics')


if __name__ == '__main__':
    main()
