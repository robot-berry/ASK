"""Render 数学/高中/提取进度.md and 提取进度.html from 提取进度.json + OCR results.

Stage model:
  1 资料盘点      — every source file inventoried           (full mark)
  2 文本可提取    — docx text layer OK / PDF text layer OK  (done)
  3 OCR 提取      — scanned PDF -> 题库/_ocr/*.txt          (ocr files)
  4 知识框架      — topic folders exist per semester        (curated markdown)
"""
import collections
import html
import json
import os
import re

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
HS = os.path.join(ROOT, '数学', '高中')
QK = os.path.join(ROOT, '题库')
OCR = os.path.join(QK, '_ocr')
OUT_MD = os.path.join(HS, '提取进度.md')
OUT_HTML = os.path.join(HS, '提取进度.html')
SEMS = ['高一/上', '高一/下', '高二/上', '高二/下', '高三/上', '高三/下']
TOPIC_FOLDERS = {
    '高一/上': 6, '高一/下': 6, '高二/上': 3, '高二/下': 5, '高三/上': 9, '高三/下': 9,
}


def load_pagecounts():
    """{rel: pages} written by _tools/page_counts.py (true totals, scanned or not)."""
    p = os.path.join(OCR, '_pagecounts.json')
    if not os.path.exists(p):
        return {}
    return json.load(open(p, encoding='utf-8'))


PAGECOUNTS = load_pagecounts()


def ocr_done(rel):
    p = os.path.join(OCR, rel.replace('/', '__') + '.txt')
    return os.path.exists(p) and os.path.getsize(p) > 0


def ocr_pages(rel):
    """(pages_processed, total_pages or 0) from an OCR output file.

    Pages are written in order, so the highest Pn is how many are on disk. The total
    comes from the PARTIAL marker, or from the _pagecounts.json cache for PDFs that
    were never scanned (so unscanned pages count as "remaining", not as nothing).
    """
    p = os.path.join(OCR, rel.replace('/', '__') + '.txt')
    if not os.path.exists(p):
        return 0, PAGECOUNTS.get(rel, 0)
    txt = open(p, encoding='utf-8', errors='ignore').read()
    nums = [int(m) for m in re.findall(r'^=== P(\d+) ===$', txt, re.M)]
    total = PAGECOUNTS.get(rel, 0)
    m = re.search(r'stopped at page (\d+)/(\d+)', txt)
    if m:
        total = max(total, int(m.group(2)))
    return (max(nums) if nums else 0), total


def ocr_complete(rel):
    done, total = ocr_pages(rel)
    return bool(done) and (total == 0 or done >= total)


CITE = re.compile(r'^- `题库/(.+?)`', re.M)
ENTRY_CITE = re.compile(r'^\*\*依据\*\*：`题库/(.+?)`', re.M)


def idea_stats():
    """Per topic folder: idea entries, how many carry a 题库 citation, and whether
    the cited sources include OCR-only (PDF) material.

    A 思考-level metric: entries written from the named 题库 files, independent of
    how far the OCR has got. The 年级/学期/主题 layout is fixed, so counting files
    on disk is enough to know how much of each stage has been written up.
    """
    st = {}
    for sem in SEMS:
        grade, term = sem.split('/')
        sem_dir = os.path.join(HS, grade, term + '学期')
        for folder in sorted(os.listdir(sem_dir)):
            if folder.startswith('_'):
                continue
            tdir = os.path.join(sem_dir, folder)
            if not os.path.isdir(tdir):
                continue
            ideas = cited = pdf_cited = knum = 0
            for fn in os.listdir(tdir):
                if not fn.endswith('.md'):
                    continue
                txt = open(os.path.join(tdir, fn), encoding='utf-8').read()
                if fn == '解题思路.md':
                    ideas = sum(1 for line in txt.splitlines() if line.startswith('### '))
                    cites = CITE.findall(txt) + ENTRY_CITE.findall(txt)
                    uniq = {c.split('（')[0].strip() for c in cites}
                    cited = len(uniq)
                    pdf_cited = sum(1 for c in uniq if c.lower().split('`')[0].endswith('.pdf'))
                else:
                    knum = sum(1 for line in txt.splitlines() if re.match(r'^- ', line))
            st[(sem, folder)] = (ideas, cited, pdf_cited, knum)
    return st


def load():
    return json.load(open(os.path.join(HS, '提取进度.json'), encoding='utf-8'))


def summarize(data):
    """Per sem counts plus OCR coverage: complete files vs pages actually OCR'd."""
    rows = []
    for sem in SEMS:
        mine = [x for x in data if x['sem'] == sem and x['subject'] == '高中']
        r = {'sem': sem, 'files': len(mine), 'docx': 0, 'pdf': 0,
             'ocr_done': 0, 'ocr_partial': 0, 'pages_done': 0, 'pages_total': 0,
             'text_ready': 0, 'topics': TOPIC_FOLDERS.get(sem, 0)}
        for x in mine:
            done, total = ocr_pages(x['rel'])
            if x['kind'] == 'docx':
                if x['status'] == 'ocr' and not done:
                    r['ocr_missing_docx'] = r.get('ocr_missing_docx', 0) + 1
                r['docx'] += 1
                r['text_ready'] += 1
                continue
            r['pdf'] += 1
            r['pages_done'] += done
            r['pages_total'] += total
            if ocr_complete(x['rel']):
                r['ocr_done'] += 1
                r['text_ready'] += 1
            elif done:
                r['ocr_partial'] += 1
        rows.append(r)
    return rows


def bar(frac, width=20):
    n = int(round(frac * width))
    return '█' * n + '░' * (width - n)


def all_citations():
    """Distinct 题库 paths cited in any 解题思路.md — header 依据 and per-讲 依据 lines."""
    cites = set()
    for sem in SEMS:
        grade, term = sem.split('/')
        sem_dir = os.path.join(HS, grade, term + '学期')
        for dirpath, _, filenames in os.walk(sem_dir):
            if '解题思路.md' not in filenames:
                continue
            txt = open(os.path.join(dirpath, '解题思路.md'), encoding='utf-8').read()
            for c in CITE.findall(txt):
                cites.add(c.split('（')[0].strip())
            for c in ENTRY_CITE.findall(txt):
                cites.add(c.split('`')[0].split('（')[0].strip())
    return cites


# 一轮复习讲义 讲 -> 高三 module folder: 37 (四心/奔驰定理) 归三角形模块，
# 35/36/38/39 归平面向量与复数，85—87 (计数原理) 归集合逻辑与计数原理。
IDEA_MODULES = [
    ('01-函数与导数', [(6, 27)]),
    ('02-不等式', [(2, 5)]),
    ('03-三角函数与解三角形', [(28, 34), (37, 37)]),
    ('04-平面向量与复数', [(35, 36), (38, 39)]),
    ('05-立体几何', [(46, 56)]),
    ('06-数列', [(40, 45)]),
    ('07-概率统计', [(83, 84), (88, 93)]),
    ('08-解析几何', [(57, 82)]),
    ('09-集合逻辑与计数原理', [(1, 1), (85, 87)]),
]

SOURCE_FAMILIES = [
    ('一轮复习讲义·解析版', lambda r: '一轮复习讲义' in r and '解析版' in r),
    ('各专题解题技巧·解析版', lambda r: '各专题解题技巧' in r and '解析版' in r),
    ('蒋老师章节 docx', lambda r: '蒋老师' in r and r.endswith('.docx')),
    ('A049 解题模型一本通', lambda r: '【A049】' in r),
    ('王文勇大招集锦 docx', lambda r: '王文勇' in r and r.endswith('.docx')),
]


def jiexi_labs(data):
    """[(讲号, rel)] for the 一轮 解析版 PDFs that carry a 讲 number."""
    out = []
    for x in data:
        rel = x['rel']
        if x['subject'] == '高中' and x['kind'] == 'pdf' \
                and '一轮复习讲义' in rel and '解析版' in rel:
            m = re.search(r'第(\d+)讲', rel)
            if m:
                out.append((int(m.group(1)), rel))
    return sorted(out)


def cited_ratio(cands, cites):
    """How many candidate source files appear in some 解题思路.md 依据 line.

    A 依据 line is the 题库 path, sometimes with a （模块N） note appended and sometimes
    cut short before the （解析版） suffix, so match on the common prefix either way.
    """
    hit = [c for c in cands if any(cite.startswith(c) or c.startswith(cite) for cite in cites)]
    return len(hit), len(cands)


def cited_pages(rel):
    """(last page cited for this 讲 anywhere, total pages now OCR'd)."""
    ocr = os.path.join(QK, '_ocr', rel.replace('/', '__') + '.txt')
    if not os.path.exists(ocr):
        return 0, 0
    txt = open(ocr, encoding='utf-8', errors='ignore').read()
    nums = [int(m) for m in re.findall(r'^=== P(\d+) ===$', txt, re.M)]
    got = max(nums) if nums else 0
    m = re.search(r'stopped at page (\d+)/(\d+)', txt)
    avail = int(m.group(2)) if m else got
    last = 0
    for sem in SEMS:
        grade, term = sem.split('/')
        p = os.path.join(HS, grade, term + '学期')
        for dirpath, _, files in os.walk(p):
            if '解题思路.md' not in files:
                continue
            md = open(os.path.join(dirpath, '解题思路.md'), encoding='utf-8').read()
            for chunk in re.split(r'^(?=### )', md, flags=re.M):
                if '`题库/' + rel + '`' not in chunk:
                    continue
                for a, b in re.findall(r'第\s*(\d+)\s*[—\-–]\s*(\d+)\s*页', chunk):
                    last = max(last, int(b))
    return last, avail or got


def addenda_points():
    """Bullet points under the OCR 增补/增补二/增补三 sections of all modules.

    The `###` count in 思路提取 counts headings, so a second and third OCR pass
    that appends bullets to an existing 讲 block leaves it unchanged; this number
    shows how much text those passes actually produced.
    """
    hs = os.path.join(HS, '高三', '上学期')
    n = 0
    for folder in os.listdir(hs):
        p = os.path.join(hs, folder, '解题思路.md')
        if not os.path.isfile(p):
            continue
        txt = open(p, encoding='utf-8').read()
        for sec in re.split(r'^## ', txt, flags=re.M)[1:]:
            if sec.startswith('增补'):
                n += sum(1 for line in sec.splitlines() if line.startswith('- '))
    return n


def coverage(data):
    """Per source family: cited / candidates, plus per-module 讲 coverage.

    For modules the third number is how many 讲 are cited through all pages OCR'd so
    far — a 讲 extracted from its first ~35 pages is complete in content but still has
    unread tail pages until the scanner gets there.
    """
    cites = all_citations()
    fams = []
    for name, pred in SOURCE_FAMILIES:
        cands = [x['rel'] for x in data
                 if x['subject'] == '高中' and pred(x['rel'])]
        fams.append((name, *cited_ratio(cands, cites)))
    labs = jiexi_labs(data)
    mods = []
    for name, ranges in IDEA_MODULES:
        cands = [rel for jn, rel in labs if any(a <= jn <= b for a, b in ranges)]
        c, t = cited_ratio(cands, cites)
        full = 0
        for rel in cands:
            if not any(cite.startswith(rel) or rel.startswith(cite) for cite in cites):
                continue
            last, avail = cited_pages(rel)
            if avail and last >= avail:
                full += 1
        mods.append((name, c, t, full))
    return fams, mods


MARK_SRC = '其他题库来源精读'


def reading_plan():
    """The 题库精读 campaign's target reading: [(rel, label, done)] in plan order.

    Only sources singled out for full-page method reading; exercise/answer/index
    volumes and PDF duplicates of already-cited docx are deliberately absent.
    `done` records that a reader covered the whole book's teaching pages (the agent's
    read range, not the citation span, since 答案/附录 pages are never cited).
    """
    return list(SRC_READ)


SRC_READ = [
    ('【A049】高考数学解题模型一本通/高考数学解题模型一本通.pdf', 'A049', True),
    ('浙大优辅-高中数学各个击破（全套23册）/高中数学各个击破', '浙大优辅', True),
    ('2027版 数学核心方法 高一 高二 高三/高三数学核心方法_351.pdf', '核心方法高三', True),
    ('2027版 数学核心方法 高一 高二 高三/高一数学核心方法_303.pdf', '核心方法高一', True),
    ('2027版 数学核心方法 高一 高二 高三/高二数学核心方法_335.pdf', '核心方法高二', True),
    ('新高考高中数学《题型总结》+高一高二培优压轴专练/新高考高中数学题型总结-必修1解析册.pdf', '申老师必修1', True),
    ('新高考高中数学《题型总结》+高一高二培优压轴专练/新高考高中数学题型总结-必修2解析册.pdf', '申老师必修2', True),
    ('新高考高中数学《题型总结》+高一高二培优压轴专练/新高考高中数学题型总结-选修1解析册.pdf', '申老师选修1', True),
    ('新高考高中数学《题型总结》+高一高二培优压轴专练/新高考高中数学题型总结-选修2解析册.pdf', '申老师选修2', True),
    ('新高考高中数学《题型总结》+高一高二培优压轴专练/新高考高中数学题型总结-选修3解析册.pdf', '申老师选修3', True),
    ('新高考高中数学《题型总结》+高一高二培优压轴专练/高一培优压轴专练-必修1解析册.pdf', '高一培优必修1', True),
    ('新高考高中数学《题型总结》+高一高二培优压轴专练/高一培优压轴专练-必修2解析册.pdf', '高一培优必修2', True),
    ('新高考高中数学《题型总结》+高一高二培优压轴专练/高二培优压轴专练-选修1解析册.pdf', '高二培优选修1', True),
    ('2026重难点培优讲义/2026重难点培优讲义教师版.pdf', '重难点培优', True),
    ('43、更高更妙的高中数学思想与方法第16版.pdf', '更高更妙', True),
    ('2026高考数学全基础题型/2026版小白书【电子版PDF】包含全内容，无水印.pdf', '小白书', True),
    ('170.高中数学万能解题模版大全解析方法共484页PDF电子版/高中数学万能解题模板.pdf', '万能模板', True),
    ('《高考数学•优质解法》（405P）.pdf', '优质解法', True),
    ('高中数学解题方法.pdf', '解题方法', True),
    ('高中数学学习思维与解题通法（第2版）去水印版.pdf', '解题通法', True),
    ('2026怎样解题-高中数学解题方法与技巧.pdf', '怎样解题', True),
    ('高中数学•解题黄金模版_2023.pdf', '黄金模板', True),
    ('高中数学方法题型一本通/高中数学方法题型一本通.pdf', '方法题型一本通', True),
    ('高中数学破题36大招(详例精编).pdf', '破题36招', True),
    ('02-【数学】解题觉醒27版/数学解题觉醒-第2册 名师大招册.pdf', '解题觉醒大招册', True),
    ('02-【数学】解题觉醒27版/数学解题觉醒-第3册 大招解析册.pdf', '解题觉醒解析册', True),
    ('高中数学解题方法与技巧_《中学生学习报》编辑部编1987.pdf', '1987解题方法', True),
    ('文档/高中数学题型方法全攻略函数篇.pdf', '函数篇', True),
    ('新高考高中数学《题型总结》+高一高二培优压轴专练/2026版申老师新高考高中数学题型总结全册word和pdf双版/2026版申老师新高考高中数学题型总结word和pdf双版/2026高三一轮复习（PDF无水印版）严禁外传电子版/2026高三第一轮复习第一册（全国通用）-279.pdf', '申老师一轮一册', True),
    ('新高考高中数学《题型总结》+高一高二培优压轴专练/2026版申老师新高考高中数学题型总结全册word和pdf双版/2026版申老师新高考高中数学题型总结word和pdf双版/2026高三一轮复习（PDF无水印版）严禁外传电子版/2026高三第一轮复习第二册（全国通用）-322.pdf', '申老师一轮二册', True),
]


def _pages_of(rel):
    p = os.path.join(OCR, rel.replace('/', '__') + '.txt')
    txt = open(p, encoding='utf-8', errors='ignore').read()
    nums = [int(m) for m in re.findall(r'^=== P(\d+) ===$', txt, re.M)]
    return max(nums) if nums else 0


def _num_pages(m):
    """（OCR 第260页）…263—264页） -> （OCR 第260—264页）"""
    nums = [int(x) for x in re.findall(r'\d+', m.group(1))]
    if not nums:
        return m.group(0)
    return f'（OCR 第{min(nums)}—{max(nums)}页'


def reading_stats():
    """Per planned source: cited page span, file pages, blocks, targets, staging state.

    Read completion = the cited span in 解题思路.md reaches the file's last page, or a
    staging file for this source exists (its blocks are merged in the same run).
    """
    spans = {}
    blocks = collections.Counter()
    for sem in SEMS:
        grade, term = sem.split('/')
        sem_dir = os.path.join(HS, grade, term + '学期')
        for dirpath, _, filenames in os.walk(sem_dir):
            if '解题思路.md' not in filenames:
                continue
            txt = open(os.path.join(dirpath, '解题思路.md'), encoding='utf-8').read()
            # Some 依据 lines carry a sloppy span such as （OCR 第235页）, （OCR 第260页）…263—264页）;
            # walk the page field and collect every number in it.
            txt = re.sub(r'（OCR 第([^）\n]*?)页', _num_pages, txt)
            cut = txt.find(MARK_SRC)
            body = txt[cut:] if cut >= 0 else ''
            blocks[(sem, os.path.basename(dirpath))] = len(re.findall(r'^### ', body, re.M))
            for rel, a, b in re.findall(
                    r'^\*\*依据\*\*：`题库/(.+?)`（OCR 第(\d+)—(\d+)页', txt, re.M):
                a, b = int(a), int(b)
                if rel in spans:
                    spans[rel] = (min(spans[rel][0], a), max(spans[rel][1], b))
                else:
                    spans[rel] = (a, b)
    return spans, blocks


def source_blocks():
    """{卷面简称: 该来源入库条目数} — blocks in the 精读 sections that cite this source,
    counted once globally so two topics citing one block don't double it."""
    seen = {}
    for sem in SEMS:
        grade, term = sem.split('/')
        sem_dir = os.path.join(HS, grade, term + '学期')
        for dirpath, _, filenames in os.walk(sem_dir):
            if '解题思路.md' not in filenames:
                continue
            txt = open(os.path.join(dirpath, '解题思路.md'), encoding='utf-8').read()
            cut = txt.find(MARK_SRC)
            if cut < 0:
                continue
            for chunk in re.split(r'^(?=### )', txt[cut:], flags=re.M)[1:]:
                hits = re.findall(r'^\*\*依据\*\*：`题库/(.+?)`', chunk, re.M)
                if not hits:
                    continue
                key = (os.path.join(sem, os.path.basename(dirpath)), chunk.split('\n')[0])
                for prefix, label, _done in SRC_READ:
                    if any(h.startswith(prefix) for h in hits):
                        seen.setdefault(key, set()).add(label)
    out = collections.Counter()
    for labels in seen.values():
        for lb in labels:
            out[lb] += 1
    return out


def read_rows(spans, data=()):
    """[(label, pages, cited_to, state, frac)] where state in 已读完 / 进行中 / 未读."""
    out = []
    for prefix, label, done in SRC_READ:
        if prefix.endswith('各个击破'):
            cands = [x for x in spans if x.startswith(prefix)]
            got = sum(_pages_of(rel) for rel in cands)
            vols = set()
            for x in data:
                if x['subject'] == '高中' and prefix in x['rel']:
                    m = re.search(r'击破(\d+)-', x['rel'])
                    if m:
                        vols.add(int(m.group(1)))
            vol = len(vols)
            if done:
                state = '已读完'
            else:
                state = '进行中' if cands else '未读'
            out.append((label, got, f'{len(cands)}/{vol} 册已读' if cands else '—',
                        state, 1.0 if done else len(cands) / max(vol, 1)))
            continue
        pages = _pages_of(prefix)
        span = spans.get(prefix)
        if span is None:
            out.append((label, pages, '—', '未读', 0.0))
            continue
        state = '已读完' if done else '进行中'
        out.append((label, pages, span[1], state, 1.0 if done else span[1] / max(pages, 1)))
        # A read span that reaches the last page is completion even if the flag lags.
        if not done and span[1] >= pages:
            out[-1] = out[-1][:3] + ('已读完', 1.0)
    return out


def md(rows, data, st):
    total = len([x for x in data if x['subject'] == '高中'])
    ready = sum(r['text_ready'] for r in rows)
    pdfs = sum(r['pdf'] for r in rows)
    ocrd = sum(r['ocr_done'] for r in rows)
    part = sum(r['ocr_partial'] for r in rows)
    pages_done = sum(r['pages_done'] for r in rows)
    pages_total = sum(r['pages_total'] for r in rows)
    cov = pages_done / max(pages_total, 1)
    ideas_total = sum(v[0] for v in st.values())
    fams, mods = coverage(data)
    fam_c, fam_t = sum(f[1] for f in fams), sum(f[2] for f in fams)
    rd = read_rows(reading_stats()[0], data)
    rd_done = sum(1 for r in rd if r[3] == '已读完')
    lines = [
        '# 高中数学 · 提取进度',
        '',
        f'> 生成时间：{__import__("datetime").datetime.now():%Y-%m-%d %H:%M}　'
        f'统计口径：`题库/` 下的高中数学资料（初中资料不计入本表）。',
        '',
        '## 总览',
        '',
        '| 阶段 | 进度 | 说明 |',
        '| --- | --- | --- |',
        f'| 1 资料盘点 | {bar(1.0)} 100% | {total} 个文件已登记（`提取进度.json`） |',
        f'| 2 文本可提取 | {bar(ready / max(total, 1))} {ready * 100 // max(total, 1)}% | 文件级：{ready}/{total}（文本层 + 已扫完 PDF） |',
        f'| 3 OCR 扫描件 | {bar(cov)} {cov * 100:.0f}% | 页级：{pages_done}/{pages_total} 页；文件级：{ocrd}/{pdfs} 扫完，{part} 份进行中 |',
        f'| 4 知识框架 | {bar(len(st) / 38)} {len(st) * 100 // 38}% | {len(st)}/38 个主题文件夹，含知识点 + 解题思路 |',
        f'| 5 思路提取 | {bar(fam_c / max(fam_t, 1))} {fam_c * 100 // max(fam_t, 1)}% | {ideas_total} 条思路条目（含一轮讲义 OCR 增补 {addenda_points()} 条要点）；依据已覆盖 {fam_c}/{fam_t} 份题库来源 |',
        f'| 6 题库精读 | {bar(rd_done / max(len(rd), 1))} {rd_done * 100 // max(len(rd), 1)}% | {rd_done}/{len(rd)} 份通读完毕，入库精读条目 {sum(source_blocks().values())} 条（各书方法并入各主题「其他题库来源精读」节） |',
        '',
        '## 分学期',
        '',
        '| 学期 | 文件 | docx | PDF | PDF 扫完 | PDF 进行中 | 已识别页 | 文本可用 | 主题文件夹 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for r in rows:
        lines.append(f'| {r["sem"]} | {r["files"]} | {r["docx"]} | {r["pdf"]} | '
                     f'{r["ocr_done"]} | {r["ocr_partial"]} | '
                     f'{r["pages_done"]}/{r["pages_total"]} | {r["text_ready"]} | {r["topics"]} |')
    lines += [
        '',
        '## 来源 · 页级 OCR',
        '',
        '| 资料 | 文件（扫完/总数） | 已识别 / 总页 | 进度 |',
        '| --- | --- | --- | --- |',
    ]
    for top, e in sorted(source_pages(data).items(),
                         key=lambda t: -(t[1]['total'] - t[1]['done'])):
        if not e['total']:
            continue
        pct = e['done'] / e['total']
        lines.append(f'| {top} | {e["complete"]}/{e["files"]} | {e["done"]}/{e["total"]} | '
                     f'{bar(pct)} {pct:.0%} |')
    lines += [
        '',
        '## 思路 · 依据覆盖',
        '',
        '| 来源 | 已用作依据 | 可作依据 | 进度 |',
        '| --- | --- | --- | --- |',
    ]
    for name, c, t in fams:
        pct = c / max(t, 1)
        lines.append(f'| {name} | {c} | {t} | {bar(pct)} {pct:.0%} |')
    spans, _ = reading_stats()
    rrows = read_rows(spans, data)
    done_n = sum(1 for r in rrows if r[3] == '已读完')
    lines += [
        '',
        '## 思路 · 题库精读（其他题库来源）',
        '',
        f'> 逐个来源通读 `题库/` 里的方法类资料，把书中方法写成条目并入对应主题的'
        f'「其他题库来源精读」节。已读完全书 {done_n}/{len(rrows)} 份。',
        '「读到页」= 该来源在 `解题思路.md` 中被引用到的最大页码；「总页」= OCR 实际页数。'
        '页段未到末尾的多是书末答案页、附录或尚未派出的页段。',
        '',
        '| 来源 | 总页 | 读到页 | 条目 | 状态 |',
        '| --- | --- | --- | --- | --- |',
    ]
    span_blocks = source_blocks()
    for label, pages, cited, state, frac in rrows:
        n = span_blocks.get(label, 0)
        lines.append(f'| {label} | {pages} | {cited} | {n} | {bar(frac)} {state} |')
    lines += [
        '',
        '## 思路 · 分模块（一轮复习讲义）',
        '',
        '> 按讲号归入高三上学期 9 个模块（讲号与模块有重叠：35/36/38/39 归平面向量，37 归三角形，85—87 归集合逻辑与计数原理）。',
        '',
        '| 模块 | 已提取讲数 | 总讲数 | 进度 | 其中已读到该讲末页 |',
        '| --- | --- | --- | --- | --- |',
    ]
    for name, c, t, full in mods:
        pct = c / max(t, 1)
        lines.append(f'| {name} | {c} | {t} | {bar(pct)} {pct:.0%} | {full} |')
    lines += [
        '',
        '## 主题明细',
        '',
        '| 学期 | 主题 | 思路条目 | 引用来源数 | 其中 PDF | 知识点条目 |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for (sem, folder), (ideas, cited, pdf_cited, knum) in st.items():
        lines.append(f'| {sem} | {folder} | {ideas} | {cited} | {pdf_cited} | {knum} |')
    lines += [
        '',
        '## 说明',
        '',
        '- 阶段 4 的「知识框架」来自 docx 精读，已落到各学期主题文件夹；PDF 的 OCR 文本主要用于补充检索与交叉核对。',
        '- 阶段 5 的「思路提取」统计已写出的 `###` 思路条目，以及「依据（题库出处）」覆盖了多少份题库来源——两者都由文件内容直接算出，不依赖 OCR 进度。',
        '- 高三上九模块 `解题思路.md` 末尾的「增补 · 一轮复习讲义（OCR）」节，来自 `2027年高考一轮复习讲义-解析版` 各讲前 ~30 页的 OCR 文本；每条以「依据」行注明讲次与所读页数。',
        '- 「增补二 · 一轮复习讲义（OCR 续页）」与「增补三 · 一轮复习讲义（OCR 尾页）」节补记该讲后续扫出的页码中新读到的题型与方法，只写前两节没有的内容；「读到末页」列说明还有多少讲的尾页待补读，扫描推进时会继续落后于已提取数。',
        '- 公式（MathType 对象 / 图片公式）在 docx 与 OCR 中均不可靠，[解题思路] 中的公式按课程标准补全并已注明。',
        '- OCR 采用 RapidOCR（PP-OCRv4），200dpi 灰度渲染；每轮每份 PDF 最多约 180 秒且至少 10 页，未扫完的末尾标注 `PARTIAL`，下一轮从断点续扫。',
    ]
    return '\n'.join(lines) + '\n'


CSS = r"""
:root{
  color-scheme:light;
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --good:#0ca30c; --warning:#fab219; --critical:#d03b3b;
}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){
  color-scheme:dark;
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70;
}
*{box-sizing:border-box}
body{margin:0;padding:32px;background:var(--plane);color:var(--ink);
  font:14px/1.55 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
.wrap{max-width:1040px;margin:0 auto}
h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--ink2);margin:0 0 24px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:20px;margin-bottom:16px}
h2{font-size:15px;margin:0 0 14px;font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(184px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--ink2);font-size:12px}
.tile .v{font-size:26px;font-weight:600;margin-top:2px}
.tile .d{color:var(--muted);font-size:12px}
.legend{display:flex;gap:18px;color:var(--ink2);font-size:12px;margin-bottom:10px;flex-wrap:wrap}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.row{display:grid;grid-template-columns:88px 1fr 132px;align-items:center;gap:12px;
  padding:5px 0}
.row .lbl{color:var(--ink2)}
.track{display:flex;height:14px;border-radius:4px;overflow:hidden;background:var(--grid);gap:2px}
.seg{height:100%}
.seg:first-child{border-radius:4px 0 0 4px}
.seg:last-child{border-radius:0 4px 4px 0}
.num{font-variant-numeric:tabular-nums;color:var(--ink2);font-size:12px;text-align:right}
.hdr{display:grid;grid-template-columns:88px 1fr 132px;gap:12px;color:var(--muted);font-size:11px;
  border-bottom:1px solid var(--grid);padding-bottom:4px;margin-bottom:4px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--grid);font-variant-numeric:tabular-nums}
th{color:var(--ink2);font-weight:600;font-size:12px}
td.src{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;word-break:break-all}
.badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;color:var(--ink)}
.dot{width:8px;height:8px;border-radius:50%}
details summary{cursor:pointer;color:var(--ink2);font-size:13px}
.note{color:var(--muted);font-size:12px}
button{font:inherit;font-size:12px;padding:4px 10px;border-radius:6px;border:1px solid var(--ring);
  background:var(--surface);color:var(--ink2);cursor:pointer}
"""


def esc(s):
    return html.escape(str(s))


def track_row(name, c, t, pct, extra=None):
    segs = ''
    if extra is not None and c:
        # 已提取且读到该讲末页的用实色，其余已提取（尾页未扫到）用浅色
        segs += f'<span class="seg" style="flex:{extra};background:var(--s1)"></span>'
        segs += f'<span class="seg" style="flex:{c - extra};background:var(--s1);opacity:.45"></span>'
    elif c:
        segs += f'<span class="seg" style="flex:{c};background:var(--s1)"></span>'
    segs += f'<span class="seg" style="flex:{max(t - c, 1)};background:var(--grid)"></span>'
    tail = '' if extra is None else f'<td>{extra}</td>'
    return (f'<tr><td class="src">{esc(name)}</td><td>{c}</td><td>{t}</td>'
            f'<td><div class="track" role="img" aria-label="{pct:.0%}" style="width:160px">'
            f'{segs}</div></td>{tail}</tr>')


def source_pages(data):
    """Per top-level source folder: pages OCR'd, true page total, file states."""
    agg = collections.defaultdict(lambda: {'done': 0, 'total': 0, 'files': 0,
                                           'complete': 0, 'partial': 0})
    for x in data:
        if x['subject'] != '高中' or x['kind'] != 'pdf':
            continue
        e = agg[x['rel'].split('/')[0]]
        done, total = ocr_pages(x['rel'])
        e['done'] += done
        e['total'] += total
        e['files'] += 1
        if ocr_complete(x['rel']):
            e['complete'] += 1
        elif done:
            e['partial'] += 1
    return agg


def segs_for(r):
    ocr_pending = r['pdf'] - r['ocr_done'] - r['ocr_partial']
    parts = []
    if r['docx']:
        parts.append(('docx 文本层', r['docx'], '--s1'))
    if r['ocr_done']:
        parts.append(('PDF 已扫完', r['ocr_done'], '--s3'))
    if r['ocr_partial']:
        parts.append(('PDF 进行中', r['ocr_partial'], '--warning'))
    if ocr_pending:
        parts.append(('PDF 未开始', ocr_pending, '--grid'))
    return parts


def build_html(rows, data, st):
    total = sum(r['files'] for r in rows)
    ready = sum(r['text_ready'] for r in rows)
    pdfs = sum(r['pdf'] for r in rows)
    ocrd = sum(r['ocr_done'] for r in rows)
    part = sum(r['ocr_partial'] for r in rows)
    pages_done = sum(r['pages_done'] for r in rows)
    pages_total = sum(r['pages_total'] for r in rows)
    cov = pages_done / max(pages_total, 1)
    ocr_pct = cov * 100
    topics = len(st)
    ideas_total = sum(v[0] for v in st.values())
    fams, mods = coverage(data)
    fam_c, fam_t = sum(f[1] for f in fams), sum(f[2] for f in fams)
    rd = read_rows(reading_stats()[0], data)
    rd_done = sum(1 for r in rd if r[3] == '已读完')
    sb = source_blocks()
    lo = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
          '<meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>高中数学 · 提取进度</title><style>' + CSS + '</style></head><body><div class="wrap">',
          '<h1>高中数学 · 提取进度</h1>',
          f'<p class="sub">题库中共 {total} 个高中文件 · 文本已可用 {ready}（{ready * 100 // max(total, 1)}%）'
          f' · PDF 页级 OCR {pages_done}/{pages_total} 页（{cov:.0%}） · 知识主题文件夹 {topics}（{ideas_total} 条思路）</p>',
          '<div class="tiles">',
          f'<div class="tile"><div class="k">1 资料盘点</div><div class="v">100%</div><div class="d">{total} 个文件已登记</div></div>',
          f'<div class="tile"><div class="k">2 文本可提取</div><div class="v">{ready * 100 // max(total, 1)}%</div>'
          f'<div class="d">{ready}/{total} 个文件</div></div>',
          f'<div class="tile"><div class="k">3 OCR 扫描件</div><div class="v">{ocr_pct:.1f}%</div>'
          f'<div class="d">{pages_done}/{pages_total} 页 · 扫完 {ocrd}/{pdfs} 份</div></div>',
          f'<div class="tile"><div class="k">4 知识框架</div><div class="v">{topics}/38</div>'
          f'<div class="d">主题文件夹 · {ideas_total} 条思路</div></div>',
          f'<div class="tile"><div class="k">5 思路提取</div><div class="v">{ideas_total}</div>'
          f'<div class="d">条思路（含 OCR 增补 {addenda_points()} 条要点）· 依据覆盖 {fam_c}/{fam_t} 份来源</div></div>',
          f'<div class="tile"><div class="k">6 题库精读</div><div class="v">{rd_done}/{len(rd)}</div>'
          f'<div class="d">份来源通读完毕 · 入库 {sum(sb.values())} 条精读条目</div></div>',
          '</div>',
          '<div class="card" style="margin-top:16px"><h2>来源 · 页级 OCR 进度</h2>',
          '<p class="note">页数以 PDF 实际页数为准（<code>_tools/page_counts.py</code> 缓存）；未开始的 PDF 也计入分母。</p>',
          '<table><thead><tr><th>资料</th><th>文件</th><th>已识别 / 总页</th>'
          '<th>进度</th></tr></thead><tbody>']
    for top, e in sorted(source_pages(data).items(),
                         key=lambda t: -(t[1]['total'] - t[1]['done'])):
        if not e['total']:
            continue
        pct = e['done'] / e['total']
        segs = ''
        if e['done']:
            segs += f'<span class="seg" style="flex:{e["done"]};background:var(--s3)"></span>'
        segs += f'<span class="seg" style="flex:{max(e["total"] - e["done"], 1)};background:var(--grid)"></span>'
        lo.append(f'<tr><td class="src">{esc(top)}</td><td>{e["complete"]}/{e["files"]} 份</td>'
                  f'<td>{e["done"]} / {e["total"]}</td>'
                  f'<td><div class="track" role="img" aria-label="{pct:.0%}" style="width:160px">'
                  f'{segs}</div></td></tr>')
    lo.append('</tbody></table></div>')
    lo.append('<div class="card"><h2>分学期提取情况</h2>')
    lo += ['<div class="legend">',
          '<span><i class="sw" style="background:var(--s1)"></i>docx 文本层（已提取）</span>',
          '<span><i class="sw" style="background:var(--s3)"></i>PDF 已扫完</span>',
          '<span><i class="sw" style="background:var(--warning)"></i>PDF 进行中（有 PARTIAL）</span>',
          '<span><i class="sw" style="background:var(--grid)"></i>PDF 未开始</span>',
          '</div>',
          '<div class="hdr"><span>学期</span><span>文件构成</span><span>文本可用 / 总数</span></div>']
    for r in rows:
        parts = segs_for(r)
        lo.append('<div class="row">')
        lo.append(f'<span class="lbl">{esc(r["sem"])}</span>')
        lo.append('<div class="track" role="img" aria-label="' +
                  esc('，'.join(f'{n} 个 {k}' for k, n, _ in parts)) + '">')
        for k, n, color in parts:
            lo.append(f'<span class="seg" style="flex:{n};background:var({color})" '
                      f'title="{esc(r["sem"])} · {esc(k)}：{n}"></span>')
        lo.append('</div>')
        lo.append(f'<span class="num">{r["text_ready"]} / {r["files"]}　·　{r["topics"]} 主题</span>')
        lo.append('</div>')
    lo.append('</div>')
    lo.append('<div class="card"><h2>解题思路 · 依据覆盖</h2>'
              '<p class="note">「已用作依据」= 该来源出现在某个 <code>解题思路.md</code> 的「依据」中'
              '（顶部出处，或增补节里逐讲的「依据」行）；「可作依据」= 题库里同类的文件数。'
              '这一步与 OCR 进度无关：思路是从 docx 与已扫出的 OCR 文本里读出来的。</p>'
              '<table><thead><tr><th>来源</th><th>已用作依据</th><th>可作依据</th>'
              '<th>进度</th></tr></thead><tbody>')
    for name, c, t in fams:
        pct = c / max(t, 1)
        lo.append(track_row(name, c, t, pct))
    lo.append('</tbody></table></div>')
    lo.append('<div class="card"><h2>解题思路 · 题库精读（其他题库来源）</h2>'
              f'<p class="note">逐个来源通读 <code>题库/</code> 里的方法类资料，把书中方法写成条目并入对应主题的'
              f'「其他题库来源精读」节。已读完全书 {rd_done}/{len(rd)} 份，入库 {sum(sb.values())} 条。'
              '「读到页」= 该来源在 <code>解题思路.md</code> 中被引用到的最大页码；进度条只反映被引用的页段覆盖，'
              '未到末尾的多是书末答案页、附录或尚未派出的页段。</p>'
              '<table><thead><tr><th>来源</th><th>总页</th><th>读到页</th><th>条目</th><th>进度</th><th>状态</th>'
              '</tr></thead><tbody>')
    for label, pages, cited, state, pct in rd:
        segs = ''
        if state == '未读' or not isinstance(cited, int):
            segs += f'<span class="seg" style="flex:1;background:var(--grid)"></span>'
        else:
            segs += f'<span class="seg" style="flex:{cited};background:var(--s1)"></span>'
            segs += f'<span class="seg" style="flex:{max(pages - cited, 1)};background:var(--grid)"></span>'
        color = 'var(--good)' if state == '已读完' else ('var(--warning)' if state == '进行中' else 'var(--muted)')
        lo.append(f'<tr><td>{esc(label)}</td><td>{pages}</td><td>{cited}</td>'
                  f'<td>{sb.get(label, 0)}</td>'
                  f'<td><div class="track" role="img" aria-label="{pct:.0%}" style="width:150px">{segs}</div></td>'
                  f'<td><span class="badge"><i class="dot" style="background:{color}"></i>{esc(state)}</span></td></tr>')
    lo.append('</tbody></table></div>')
    lo.append('<div class="card"><h2>解题思路 · 分模块（一轮复习讲义）</h2>'
              '<p class="note">按讲号归入高三上学期 9 个模块文件夹（讲号与模块有重叠：35/36/38/39 归平面向量，37 归三角形，85—87 归集合逻辑与计数原理）。'
              '「读到末页」= 该讲已 OCR 的页全部读过；其余讲只读过扫描已到达的前段（浅色段），尾页扫完后在「增补二」「增补三」节补记。</p>'
              '<table><thead><tr><th>模块</th><th>已提取讲数</th><th>总讲数</th>'
              '<th>进度</th><th>读到末页</th></tr></thead><tbody>')
    for name, c, t, full in mods:
        pct = c / max(t, 1)
        lo.append(track_row(name, c, t, pct, extra=full))
    lo.append('</tbody></table></div>')
    # OCR page progress: biggest unfinished PDFs first
    lo.append('<div class="card"><h2>OCR 页级进度（未扫完的先列）</h2>'
              '<table><thead><tr><th>学期</th><th>PDF</th><th>已识别</th>'
              '<th>总页数</th><th>剩余</th><th>状态</th></tr></thead><tbody>')
    rows_page = []
    for x in data:
        if x['subject'] != '高中' or x['kind'] != 'pdf':
            continue
        done_p, total_p = ocr_pages(x['rel'])
        if not done_p and not total_p:
            continue
        rows_page.append((x, done_p, total_p))
    rows_page.sort(key=lambda t: (0 if t[2] else 1, -(t[2] - t[1]) if t[2] else 0))
    n_show = 40
    for x, done_p, total_p in rows_page[:n_show]:
        if total_p:
            left = max(total_p - done_p, 0)
            st_txt = '进行中' if left else '已扫完'
            tot_txt = str(total_p)
        else:
            left = 0
            st_txt, tot_txt = '已扫完', '—'
        lo.append(f'<tr><td>{esc(x["sem"])}</td><td class="src">{esc(x["rel"].split("/")[-1])}</td>'
                  f'<td>{done_p}</td><td>{tot_txt}</td><td>{left or "—"}</td>'
                  f'<td>{esc(st_txt)}</td></tr>')
    lo.append('</tbody></table><p class="note">共 %d 份 PDF 已有 OCR 文本，此处按剩余页数列出前 %d 份。'
              '每轮每份最多约 180 秒（且至少 10 页），末尾标注 PARTIAL，下一轮从断点续扫。</p></div>'
              % (len(rows_page), min(n_show, len(rows_page))))
    return _html_rest(rows, data, st, lo, ideas_total, topics)


def _html_rest(rows, data, st, lo, ideas_total, topics):
    # source table grouped by top folder
    lo.append('<div class="card"><h2>来源资料清单（按顶层文件夹）</h2>')
    by_top = collections.defaultdict(lambda: {'total': 0, 'ready': 0, 'ocr': 0, 'subject': ''})
    lo.append('<table><thead><tr><th>资料</th><th>文件数</th><th>文本可用</th><th>待 OCR</th>'
              '<th>状态</th></tr></thead><tbody>')
    _placeholder = None
    for x in data:
        if x['subject'] != '高中':
            continue
        top = x['rel'].split('/')[0]
        e = by_top[top]
        e['subject'] = top
        e['total'] += 1
        if x['kind'] == 'docx' or ocr_complete(x['rel']):
            e['ready'] += 1
        if x['kind'] == 'pdf' and not ocr_complete(x['rel']):
            e['ocr'] += 1
    for top, e in sorted(by_top.items(), key=lambda t: -t[1]['total']):
        if e['ocr'] == 0:
            badge = '<span class="badge"><i class="dot" style="background:var(--good)"></i>已提取</span>'
        elif e['ready'] == 0:
            badge = '<span class="badge"><i class="dot" style="background:var(--warning)"></i>待 OCR</span>'
        else:
            badge = '<span class="badge"><i class="dot" style="background:var(--warning)"></i>部分待 OCR</span>'
        lo.append(f'<tr><td class="src">{esc(top)}</td><td>{e["total"]}</td><td>{e["ready"]}</td>'
                  f'<td>{e["ocr"]}</td><td>{badge}</td></tr>')
    lo.append('</tbody></table>')
    lo.append('</div>')
    lo.append('<div class="card"><h2>主题明细（思路条目 / 知识点条目）</h2>'
              '<table><thead><tr><th>学期</th><th>主题文件夹</th><th>思路条目</th>'
              '<th>引用来源</th><th>其中 PDF</th><th>知识点条目</th></tr></thead><tbody>')
    for (sem, folder), (ideas, cited, pdf_cited, knum) in st.items():
        lo.append(f'<tr><td>{esc(sem)}</td><td class="src">{esc(folder)}</td>'
                  f'<td>{ideas}</td><td>{cited}</td><td>{pdf_cited}</td><td>{knum}</td></tr>')
    lo.append('</tbody></table>')
    lo.append('<p class="note">OCR 结果写入 <code>题库/_ocr/</code>。重新生成：'
              '<code>python -I _tools/build_progress.py</code></p></div>')
    lo.append('</div></body></html>')
    return '\n'.join(lo)


def main():
    data = load()
    rows = summarize(data)
    st = idea_stats()
    open(OUT_MD, 'w', encoding='utf-8').write(md(rows, data, st))
    open(OUT_HTML, 'w', encoding='utf-8').write(build_html(rows, data, st))
    print('progress written')


if __name__ == '__main__':
    main()
