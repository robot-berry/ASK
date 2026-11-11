"""Dump condensed text (method-oriented lines) from selected docx sources.

Chinese paths are hard-coded because argv through the shell mangles them here.
Writes: _extract/<label>.full.txt and _extract/<label>.cond.txt
"""
import os
import re
import zipfile
import glob

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
QK = os.path.join(ROOT, '题库')
OUT = os.path.join(ROOT, '_extract')

MARKERS = re.compile(
    r'(技法\d|题型\d|考点|知识迁移|点睛|解题思路|解题技巧|方法总结|通法|模板|'
    r'^[一二三四五六七八九十]+、|^第[一二三四五六七八九十]+[章节]|^[0-9]+\.[0-9]+)'
)


def docx_text(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read('word/document.xml').decode('utf-8', 'replace')
    xml = xml.replace('<w:tab/>', '\t').replace('<w:br/>', '\n')
    xml = re.sub(r'</w:p>', '\n', xml)
    text = re.sub(r'<[^>]+>', '', xml)
    text = text.replace('false', '□')  # MathType OLE placeholder
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def condense(text):
    keep = []
    for para in text.split('\n'):
        p = para.strip()
        if not p:
            continue
        if MARKERS.search(p) or len(p) > 40:
            keep.append(p)
    return '\n'.join(keep)


def main():
    os.makedirs(OUT, exist_ok=True)
    sources = []

    # 蒋老师 high school chapter docx
    for sub in ['[格式docx]高一题型方法一本通上_docx',
                '[格式docx]抖音高一题型方法一本通下册_docx',
                '[精排精校Word版]F0259-抖音蒋数学-高二题型方法一本通选择性必修上册/[精排精校Word版]F0259-抖音蒋数学-高二题型方法一本通选择性必修上册_docx',
                '[精排精校Word版]F0260-抖音蒋数学-高二题型方法一本通选择性必修下册/[精排精校Word版]F0260-抖音蒋数学-高二题型方法一本通选择性必修下册_docx']:
        for p in glob.glob(os.path.join(QK, '蒋老师高一数学+高二', sub, '*.docx')):
            sources.append(p)

    # 专题解题技巧 解析版
    for p in glob.glob(os.path.join(QK, '高中数学各专题解题技巧', '高中数学各专题解题技巧', '*', '*', '*解析版*.docx')):
        sources.append(p)

    # 高三: 解题模型一本通 + 大招集锦技法篇
    sources.append(os.path.join(QK, '【A049】高考数学解题模型一本通', '高考数学解题模型一本通 解析册.docx'))
    sources.append(os.path.join(QK, '王文勇高考数学大招集锦【26-25】', '王文勇《高考数学大招集锦》2026版（技法篇）完整版.docx'))

    print('sources', len(sources))
    for i, src in enumerate(sources):
        if not os.path.exists(src):
            print('MISSING', src)
            continue
        label = f'{i:02d}_' + os.path.splitext(os.path.basename(src))[0][:60]
        try:
            t = docx_text(src)
        except Exception as exc:  # noqa: BLE001
            print('FAIL', label, exc)
            continue
        with open(os.path.join(OUT, label + '.full.txt'), 'w', encoding='utf-8') as fh:
            fh.write(t)
        with open(os.path.join(OUT, label + '.cond.txt'), 'w', encoding='utf-8') as fh:
            fh.write(condense(t))
        print(f'{i:02d} {len(t):>7} -> {len(condense(t)):>7}  {os.path.basename(src)[:70]}')


if __name__ == '__main__':
    main()
