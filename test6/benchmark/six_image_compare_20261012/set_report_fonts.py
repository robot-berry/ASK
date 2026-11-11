from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree

P=Path(__file__).resolve().parent
src=P/'搜题工程六图流程模型测试报告.docx'
dst=P/'搜题工程六图流程模型测试报告_宋体版.docx'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
ns={'w':W}
def tag(n):return '{'+W+'}'+n
def font(pr):
    fs=pr.find(tag('rFonts'))
    if fs is None:
        fs=etree.Element(tag('rFonts'));pr.insert(0,fs)
    for name in list(fs.attrib):
        if name.rsplit('}',1)[-1].endswith('Theme'):del fs.attrib[name]
    for n in ['ascii','hAnsi','cs']:fs.set(tag(n),'Times New Roman')
    fs.set(tag('eastAsia'),'宋体')

with ZipFile(src) as zin,ZipFile(dst,'w',ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        content=zin.read(info.filename)
        if info.filename.startswith('word/') and info.filename.endswith('.xml'):
            root=etree.fromstring(content)
            for r in root.xpath('//w:r|//w:style',namespaces=ns):
                pr=r.find(tag('rPr'))
                if pr is None:
                    pr=etree.Element(tag('rPr'));r.insert(0,pr)
                font(pr)
            for pr in root.xpath('//w:rPr',namespaces=ns):font(pr)
            content=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
        zout.writestr(info,content)
with ZipFile(src) as a,ZipFile(dst) as b:
    for n in a.namelist():
        if n.startswith('word/') and n.endswith('.xml'):
            before=etree.fromstring(a.read(n));after=etree.fromstring(b.read(n))
            assert before.xpath('//w:t/text()',namespaces=ns)==after.xpath('//w:t/text()',namespaces=ns),n
        elif n.startswith('word/media/'):
            assert a.read(n)==b.read(n),n
print(dst)
print('Verified: text and original sample images unchanged; editable text font mapping updated.')
