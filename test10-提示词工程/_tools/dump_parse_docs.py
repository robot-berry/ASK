"""Locate and extract selected textbook (docx) sources, printing a preview.

Paths are hard-coded here because Chinese paths passed through the shell argv
get mangled by the console codepage on this machine.
"""
import os
import re
import zipfile
import glob

ROOT = r'D:\robot\Documents\ASK\picture\test10-提示词工程'
QK = os.path.join(ROOT, '题库')
OUT = os.path.join(ROOT, '_extract')


def docx_text(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read('word/document.xml').decode('utf-8', 'replace')
    xml = xml.replace('<w:tab/>', '\t').replace('<w:br/>', '\n')
    xml = re.sub(r'</w:p>', '\n', xml)
    text = re.sub(r'<[^>]+>', '', xml)
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    pattern = os.path.join(QK, '高中数学各专题解题技巧', '高中数学各专题解题技巧', '题型02*', '*', '*解析版*.docx')
    hits = glob.glob(pattern)
    print('matched', len(hits))
    for src in hits[:1]:
        t = docx_text(src)
        dst = os.path.join(OUT, 'jiexi_test.txt')
        with open(dst, 'w', encoding='utf-8') as fh:
            fh.write(t)
        print('chars', len(t), '->', dst)
        for line in t.splitlines()[:40]:
            if line.strip():
                print(line[:120])


if __name__ == '__main__':
    main()
