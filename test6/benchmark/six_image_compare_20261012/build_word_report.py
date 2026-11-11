from pathlib import Path
import json
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from PIL import Image, ImageDraw, ImageFont

P=Path(__file__).resolve().parent
data=json.loads((P/'summary.json').read_text(encoding='utf-8'))
M=data['models']; B=data['budget']
labels={'qwen_max':'Qwen3.8-Max','glm_flash':'GLM-5.3-Flash','kimi':'Kimi-K2.6','qwen_27b':'Qwen3.8-27B'}
order=list(labels)
d=Document(); sec=d.sections[0]
sec.page_width=Inches(8.5); sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.7)
sec.left_margin=sec.right_margin=Inches(.7)
sec.header_distance=sec.footer_distance=Inches(.3)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
 st=d.styles[name]; st.font.name='Microsoft YaHei'; st.font.color.rgb=RGBColor(0,0,0)
 st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'微软雅黑')
 st.font.size=Pt(11)
 st.paragraph_format.space_after=Pt(7)
 st.paragraph_format.line_spacing=Pt(16)
 st.font.italic=False
 st.font.bold=False
 st.paragraph_format.keep_with_next=False
 snap=OxmlElement('w:snapToGrid');snap.set(qn('w:val'),'0');st.element.get_or_add_pPr().append(snap)
d.styles['Title'].font.size=Pt(23);d.styles['Title'].font.bold=True
d.styles['Heading 1'].font.size=Pt(15);d.styles['Heading 1'].font.bold=True
d.styles['Heading 1'].paragraph_format.space_before=Pt(12)
d.styles['Title'].paragraph_format.line_spacing=Pt(30)
d.styles['Heading 1'].paragraph_format.line_spacing=Pt(21)
d.styles['Heading 1'].paragraph_format.keep_with_next=True
d.styles['Heading 2'].font.size=Pt(12);d.styles['Heading 2'].font.bold=True
d.styles['Caption'].font.size=Pt(9)
d.styles['Caption'].paragraph_format.line_spacing=Pt(13)
h=sec.header.paragraphs[0]; h.text='TEST6 搜题工程  模型测试报告';h.runs[0].font.size=Pt(9)
f=sec.footer.paragraphs[0]; f.alignment=WD_ALIGN_PARAGRAPH.RIGHT
f.add_run('2026年10月12日    第 ')
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');f._p.append(field)
f.add_run(' 页');
for r in f.runs:r.font.size=Pt(9)

def para(t,style=None):return d.add_paragraph(t,style)
def heading(t):d.add_heading(t,1)
def table(headers, rows, widths):
 t=d.add_table(rows=1, cols=len(headers));t.style='Table Grid';t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 for i,w in enumerate(widths):t.columns[i].width=Inches(w)
 for i,v in enumerate(headers):t.rows[0].cells[i].text=v
 for row in rows:
  cells=t.add_row().cells
  for i,v in enumerate(row):cells[i].text=str(v)
 for ri,row in enumerate(t.rows):
  trpr=row._tr.get_or_add_trPr();keep=OxmlElement('w:cantSplit');trpr.append(keep)
  if ri==0:
   repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
  for ci,c in enumerate(row.cells):
   c.width=Inches(widths[ci]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pr=c._tc.get_or_add_tcPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'DCE6F1' if ri==0 else ('F4F6F8' if ri%2==0 else 'FFFFFF'));pr.append(sh)
   b=OxmlElement('w:tcBorders')
   for edge in ['top','bottom','left','right']:
    e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
   pr.append(b);mar=OxmlElement('w:tcMar')
   for edge in ['top','bottom','left','right']:
    e=OxmlElement('w:'+edge);e.set(qn('w:w'),'90');e.set(qn('w:type'),'dxa');mar.append(e)
   pr.append(mar)
   for p in c.paragraphs:
    p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=Pt(14)
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:r.font.size=Pt(10);r.font.bold=ri==0
 para('') .paragraph_format.space_after=Pt(0)
 return t

para('搜题工程六图流程模型测试报告','Title')
para('项目 test6    测试日期 2026年10月12日    预算上限 10 元','Subtitle')
para('本报告比较当前 Qwen3.8-Max 与三个候选模型在真实六图流程中的解析呈现、逐层出图和总时延，供模型替换决策使用。测试不比对标准答案，不统计解题正确率。')
para('结论：GLM-5.3-Flash 的六图齐全速度最接近基线，但首层解析出现更晚，格式兼容性也有差距；当前证据不支持直接等效替换。建议暂时保留现有模型。')
heading('主要时延结果')
table(['模型','计入均值\n样本数','首层解析\n平均秒','六图齐全\n平均秒','任务完成\n平均秒'],[[labels[k],M[k]['direct_five_tiers'],f"{M[k]['first_guidance_mean_s']:.2f}",f"{M[k]['all_six_mean_s']:.2f}",f"{M[k]['final_mean_s']:.2f}"] for k in order],[1.7,1.1,1.4,1.45,1.45])
para('统计口径：均值仅包含六图合规、模型响应完整且未使用兜底或提前截流复用的样本。各模型有效样本不同，同题配对结果见下一页。首层解析指第 2 张图，不是本地 OCR 生成的第 1 张原题图。','Caption')
heading('测试完成情况')
para('计划为数学 10 题、物理 10 题，每题测试 4 个模型，共 80 组。实际完成 70 组正式测试，生成 420 张 PNG；另外保留 1 次基线冷启动调用记录。剩余 10 组未运行，原因是累计预算保护触发。')
para(f"上轮及本轮累计保守计费与预留为 {B['total_accounted']:.2f} 元，未超过 10 元上限。所有调用已返回，测试进程和专用 OCR 服务已关闭。本报告制作没有新增模型测试调用。")
heading('六图流程范围')
para('原题图片 → 本地 OCR 与公式识别 → 模型流式生成 → 工程校验及本地修正 → 原题图与五层解析图发布。测试直接调用后端流程，不含网页轮询、网络下载及浏览器绘制时间。')

d.add_page_break();heading('测试设置与评价口径')
para('基线使用阿里云 qwen3.8-max。硅基流动候选为 zai-org/GLM-5.3-Flash、Pro/moonshotai/Kimi-K2.6 和 Qwen/Qwen3.8-27B。本文 GLM 指 GLM-5.3-Flash，不是 GLM-5.3 旗舰版。全部使用工程提示词和 JSON Schema；串行运行，每个正式组合调用一次。')
para('统一限制输出为 4096 tokens，模型等待预算为 90 秒，学弱层后预算为 15 秒，关闭提供商修复重试。候选思考预算参数为 1024，GLM 启用思考，其余候选关闭思考。这是费用受限的测试配置，并非无限输出条件。')
para('每次重新执行 OCR。数学首题基线发生公式 OCR 冷启动超时，改用了原图输入；该次单独归档，补测后纳入正式统计，费用仍计入累计预算。其他 OCR 差异通过配对统计单列。')
heading('六图及呈现检查')
table(['模型','实测\n组数','六图文件\n合规','严格校验\n通过¹','提前截流\n复用','超时\n兜底'],[[labels[k],M[k]['tested'],M[k]['six_valid_png'],f"{M[k]['strict_valid']}/{M[k]['direct_five_tiers']}",M[k]['partial'],M[k]['fallback']] for k in order],[1.65,.85,1.15,1.25,1.2,1.0])
para('¹ 分母为完整响应样本。严格校验检查公式、字段、呈现等工程约束，不是答案正确率。六图文件合规只表示文件和图像规格通过检查，不保证原题插图、所有小问或解析内容完整。','Caption')
heading('相同输入条件下的配对时延')
table(['候选模型','可配对\n题数','同 OCR 与\n附图数量题数','这些同输入配对\n六图时延差平均秒'],[[labels[k],M[k]['matched_pairs'],M[k]['same_ocr_pairs'],f"{M[k]['same_ocr_all_six_delta_s']:+.2f}"] for k in ['glm_flash','kimi','qwen_27b']],[1.7,1.1,1.6,2.7])
para('时延差为候选减基线，负值代表候选较快。配对仅限双方均有完整响应的同题样本；“同输入”核对 OCR 文本和附图数量，并非图片字节逐一相同。GLM 的 14 个此类配对平均快 2.38 秒，但不能据此推断首层解析也更早。','Caption')

d.add_page_break();heading('主要发现与限制')
d.add_heading('逐层出图存在接入差异',2)
para('基线 18 组均出现逐层发布记录，三个候选均未出现。工程要求先依次输出 l1 至 l5；保存的候选响应未以 l1 开头。GLM 首题先返回公式字段 f，导致现有解析器等待整份输出完成。离线将同一内容重排后可以触发 L1 检测，但尚未实测适配后的实时性能。')
d.add_heading('提前截流与超时兜底需要分开判断',2)
para('基线 3 组在 L5 已发布后触发剩余时间限制，保留五层模型内容，但未完整接收尾部公式或总结。Qwen3.8-27B 的 3 组超时则使用了通用步骤兜底，不能视为该模型完成了解析。两类样本均不计入完整响应时延均值。')
d.add_heading('解析呈现尚未达到等效证据',2)
para('GLM 在 17 组完整响应中有 6 组严格校验通过，基线为 15 组中的 11 组。工程警告包括公式定界符或下标格式异常、题干引用插图但图片字段为空、小问覆盖不足；部分问题也存在于基线。因此六图出图率不能替代内容呈现质量验收。')
heading('费用与停止依据')
table(['项目','金额或状态'],[['上轮保守计费与预留',f"{B['prior_accounted']:.4f} 元"],['本轮保守计费与预留',f"{B['new_accounted']:.4f} 元"],['累计保守计费与预留',f"{B['total_accounted']:.4f} 元 / 10 元"],['两轮已报告用量的估算费用',f"{B['known_estimated']:.4f} 元"],['本轮调用数与待返回数','71 次 / 0 次']],[4.4,2.7])
para('未知用量保留调用前的费用上限预留，已知用量按估算加 10% 余量；9.77 元不是已核对的实际账单。下一次调用所需预留超过剩余额度时即停止，因此没有继续消费剩余余额。最终扣费以供应商账单为准。','Caption')
heading('使用建议')
para('保留 Qwen3.8-Max 作为当前生产模型。后续若安排独立的适配测试，优先验证 GLM 的字段顺序和公式格式修正，并重复测量首层可见时间、六图齐全时间及呈现完整性。本轮为非随机小样本，每组合仅一次，不足以证明长期稳定性或统计等效。')

d.add_page_break();heading('六图样例与核查记录')
para('下图为 GLM 数学首题的六图缩略总览，展示原题和五层解析的实际输出结构。缩略图用于版式核查，完整图片保存在对应任务目录中。')
rows=json.loads((P/'results_math.json').read_text(encoding='utf-8'))
r=rows['TZIP-MATH-T01-Q01__glm_flash'];job=Path(r['job_dir'])
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
canvas=Image.new('RGB',(1500,1580),'#eeeeee');draw=ImageDraw.Draw(canvas)
names=['原题','学霸解析','中上解析','中等解析','中下解析','学渣解析']
for i,fmeta in enumerate(r['pngs']):
 im=Image.open(job/fmeta['name']).convert('RGB');im.thumbnail((476,710))
 x=i%3*500;y=i//3*790;draw.text((x+12,y+8),f'{i+1}  {names[i]}',font=font,fill='black');canvas.paste(im,(x+12,y+55))
image=P/'word_six_sample.png';canvas.save(image)
pic=para('');pic.alignment=WD_ALIGN_PARAGRAPH.CENTER;pic.paragraph_format.line_spacing=1.0;pic.add_run().add_picture(str(image),width=Inches(5.7))
para('图 1  GLM 数学样本 TZIP-MATH-T01-Q01 六图缩略总览','Caption')
para('文件核查覆盖 420 张正式结果 PNG，均可解码且哈希与保存记录一致。人工抽查了以上六图，以及物理 Q10 四个模型的第六图；所查页面未见明显截断，Qwen27B 的通用重复步骤与兜底记录一致。抽查不等于逐页人工验收，也不涉及标准答案核对。')
para('数据来源：同目录 summary.json、results_math.json、results_physics.json、ledger.json、file_verification.json 与各任务 audit 文件。上轮费用来自 model_compare_20261011/ledger.json。','Caption')
para('报告与原始记录目录\nD:\\robot\\Documents\\ASK\\picture\\test6\\benchmark\\six_image_compare_20261012','Caption')
d.core_properties.title='搜题工程六图流程模型测试报告'
d.core_properties.subject='解析呈现与时延对比'
d.core_properties.author=''
d.core_properties.comments='依据已保存的实测结果生成，未新增模型测试调用。'
for root in [d.styles.element,d.element]:
 for element in list(root.iter(qn('w:pBdr'))):element.getparent().remove(element)
for grid in list(sec._sectPr.iter(qn('w:docGrid'))):grid.getparent().remove(grid)
out=P/'搜题工程六图流程模型测试报告.docx';d.save(out);print(out)
