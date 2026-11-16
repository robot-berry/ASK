"""Extract plain text from .docx files.

Usage: python -I extract_docx.py OUTDIR FILE [FILE...]
Writes one .txt per input, named after the source file stem.
"""
import sys
import os
import re
import zipfile


def docx_text(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read('word/document.xml').decode('utf-8', 'replace')
    # Each paragraph -> newline; tabs -> \t
    xml = xml.replace('<w:tab/>', '\t')
    xml = xml.replace('<w:br/>', '\n')
    xml = re.sub(r'</w:p>', '\n', xml)
    text = re.sub(r'<[^>]+>', '', text if False else xml)
    # Collapse excessive blank lines
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def main():
    outdir, files = sys.argv[1], sys.argv[2:]
    os.makedirs(outdir, exist_ok=True)
    for src in files:
        try:
            text = docx_text(src)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print('FAIL', os.path.basename(src), exc)
            continue
        stem = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(outdir, stem + '.txt')
        with open(dst, 'w', encoding='utf-8') as fh:
            fh.write(text)
        print(f'OK {len(text):>8}  {dst}')


if __name__ == '__main__':
    main()
