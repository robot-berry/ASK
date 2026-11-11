from pathlib import Path
import re
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image, ImageDraw, ImageFont
root=Path(r'D:\robot\Documents\ASK\picture\test6\txt')
work=root/'word_work'
source=(root/'专利文档_题图结构恢复与分阶段发布.txt').read_text(encoding='utf-8-sig')
out=root/'专利初稿_题图结构恢复与分阶段发布.docx'
font_path=r'C:\Windows\Fonts\simsun.ttc'
font=ImageFont.truetype(font_path,32)
small=ImageFont.truetype(font_path,26)
def flow(name, labels, notes=''):
    w=1400; h=130+len(labels)*112+(90 if notes else 0)
    im=Image.new('RGB',(w,h),'white'); d=ImageDraw.Draw(im)
    for i,label in enumerate(labels):
        y=40+i*112
        d.rectangle((180,y,1220,y+72),outline='black',width=3)
        d.text((700,y+36),label,font=font,fill='black',anchor='mm')
        if i<len(labels)-1:
            d.line((700,y+73,700,y+108),fill='black',width=3)
            d.polygon([(700,y+108),(690,y+95),(710,y+95)],fill='black')
    if notes: d.text((700,h-48),notes,font=small,fill='black',anchor='mm')
    path=work/name; im.save(path); return path
figs=[]
figs.append(flow('fig1.png',['终端上传与统一入口','任务管理及设备队列','预处理与基础文字识别','风险分流及公式空间融合','题目文字与必要来源图像','本地原题图发布     模型分档内容处理','本地分档渲染与图片规格校验','结果集合更新与终端显示'],'本地原题图独立发布 后续模型分档图片逐步加入结果集合'))
figs.append(flow('fig2.png',['S101 输入题图与预处理','S102 基础识别与结构指标','S103 按风险保留结果或选择补充识别','S104 公式空间锚定与片段融合','S105 异常输出及结构保留检查','S106 按需整页恢复并保留来源像素','S107 独立发布本地原题图','S108 请求模型分档内容','S109 解析并校验完整字段','S110 临时渲染 校验及原子发布','S111 按需刷新附加内容','S112 完整图片集合校验后标记完成']))
figs.append(flow('fig3.png',['基础文字项 中文条件  扁平公式片段  后续条件','获取文字框与公式候选框','交叠比例检查与公式候选过滤','按字符宽度权重映射候选区间','定位对应数学字符段','保留两侧文字并插入结构化公式项','抑制位置重叠的重复识别内容','本地绘制分式 根式及上下标'],'图示对应要求空间锚点的处理模式'))
figs.append(flow('fig4.png',['本地识别与重排完成  发布图片1','l1完整并通过校验  发布图片2','l2完整并通过校验  发布图片3','l3完整并通过校验  发布图片4','l4完整并通过校验  发布图片5','l5完整并通过校验  发布图片6','尾随公式或总结到达  可刷新图片6','最终内容与六图集合校验  完成或报告错误'],'未通过早期校验的分档延后至最终处理 时间间隔不按比例'))
doc=Document(); sec=doc.sections[0]
sec.page_width=Cm(21); sec.page_height=Cm(29.7)
sec.top_margin=Cm(2.3); sec.bottom_margin=Cm(2.2); sec.left_margin=Cm(2.6); sec.right_margin=Cm(2.3)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
    s=doc.styles[name]; s.font.name='Times New Roman'; s.font.color.rgb=RGBColor(0,0,0)
    s.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    s.font.size=Pt(11)
normal=doc.styles['Normal']; normal.paragraph_format.line_spacing=1.35
normal.paragraph_format.space_after=Pt(5)
for n,size in [('Title',19),('Heading 1',16),('Heading 2',13),('Heading 3',11.5)]:
    s=doc.styles[n]; s.font.size=Pt(size); s.font.bold=True
    s.paragraph_format.space_before=Pt(12); s.paragraph_format.space_after=Pt(8)
    s.paragraph_format.keep_with_next=True
p=doc.add_paragraph('一种题图结构恢复与分阶段发布方法及系统','Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p=doc.add_paragraph('发明专利初稿','Subtitle'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
for line in ['申请人：待填写    发明人：待填写','工程版本：Test6 6.3.6    Word整理日期：2026年10月14日']:
    p=doc.add_paragraph(line); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p=doc.add_paragraph('本稿涉及题图识别风险分流、公式空间融合、来源图像保留及分阶段图片发布，包含说明书摘要、权利要求书、说明书和示意附图。')
abstract=source.split('一、说明书摘要')[1].split('二、权利要求书')[0]
abstract='\n'.join(x for x in abstract.splitlines() if x.strip() and not x.startswith('==='))
doc.add_heading('说明书摘要',1)
for line in abstract.splitlines(): doc.add_paragraph(line)
claims=source.split('二、权利要求书')[1].split('三、说明书')[0]
doc.add_page_break(); doc.add_heading('权利要求书',1)
for line in claims.splitlines():
    line=line.strip()
    if not line or line.startswith('==='):continue
    p=doc.add_paragraph(line)
    if re.match(r'^\d+\.',line): p.paragraph_format.space_before=Pt(8)
    else: p.paragraph_format.first_line_indent=Cm(.74)
body=source.split('三、说明书')[1].split('四、说明书附图')[0]
doc.add_page_break(); doc.add_heading('说明书',1)
for line in body.splitlines():
    line=line.strip()
    if not line or line.startswith('==='):continue
    if re.match(r'^（[一二三四五]）',line):
        doc.add_heading(re.sub(r'^（[一二三四五]）','',line),2)
    elif re.match(r'^实施例[一二三四五六七]',line):
        doc.add_heading(line.replace('：',' '),3)
    elif re.match(r'^\d+\. ',line): doc.add_heading(re.sub(r'^\d+\. ','',line),3)
    else:
        p=doc.add_paragraph(line);p.paragraph_format.widow_control=True
for i,(path,title) in enumerate(zip(figs,['系统模块及数据流','风险分流与分阶段发布流程','公式空间锚定及文字片段替换','有序字段解析与六图分阶段发布时序'])):
    doc.add_page_break()
    if i==0:doc.add_heading('说明书附图',1)
    doc.add_heading(f'图{i+1} {title}',2)
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    im=Image.open(path); width=min(15.8,21.0*im.width/im.height)
    p.add_run().add_picture(str(path),width=Cm(width))
    if i==0:doc.add_paragraph('原题图与模型引导图片在各自可用时发布，统一经结果管理提供给终端。')
    if i==1:doc.add_paragraph('补充识别路径包括公式专用识别和完整结构化识别；未触发条件时保留基础结果。')
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
footer.add_run('第 ')
f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');footer._p.append(f)
footer.add_run(' 页')
doc.core_properties.title='一种题图结构恢复与分阶段发布方法及系统'
doc.core_properties.subject='发明专利初稿';doc.core_properties.author=''
doc.save(out)
check=Document(out)
assert len(re.findall(r'^\[\d{4}\]', '\n'.join(p.text for p in check.paragraphs),re.M))==59
assert len(check.inline_shapes)==4
print(out);print('paragraphs',len(check.paragraphs),'figures',len(check.inline_shapes))
