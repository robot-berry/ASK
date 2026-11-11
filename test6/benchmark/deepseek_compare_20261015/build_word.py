from pathlib import Path
import json
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
P=Path(__file__).resolve().parent
S=json.loads((P/'summary.json').read_text(encoding='utf-8'));s=S['deepseek'];b=S['budget']
rows=[r for sub in ['math','physics'] for r in json.loads((P/f'results_{sub}.json').read_text(encoding='utf-8')).values()]
d=Document(P.parent/'six_image_compare_20261012/搜题工程六图流程模型测试报告_宋体版.docx')
for e in list(d.element.body):
 if e.tag!=qn('w:sectPr'):d.element.body.remove(e)
d.sections[0].footer.paragraphs[0].text='2026年10月15日    DeepSeek 补充测试'
def p(t,style=None):return d.add_paragraph(t,style)
def h(t):d.add_heading(t,1)
def fmt(x):return '—' if x is None else f'{x:.2f}'
def table(head,rs,widths):
 t=d.add_table(rows=1,cols=len(head));t.style='Table Grid';t.autofit=False
 for i,w in enumerate(widths):t.columns[i].width=Inches(w)
 for i,x in enumerate(head):t.rows[0].cells[i].text=x
 for row in rs:
  c=t.add_row().cells
  for i,x in enumerate(row):c[i].text=str(x)
 for n,row in enumerate(t.rows):
  pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
  if n==0:pr.append(OxmlElement('w:tblHeader'))
  for i,c in enumerate(row.cells):
   c.width=Inches(widths[i]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   cp=c._tc.get_or_add_tcPr();fill=OxmlElement('w:shd');fill.set(qn('w:fill'),'DCE6F1' if n==0 else ('F4F6F8' if n%2==0 else 'FFFFFF'));cp.append(fill)
   borders=OxmlElement('w:tcBorders')
   for edge in ['top','bottom','left','right']:
    x=OxmlElement('w:'+edge);x.set(qn('w:val'),'single');x.set(qn('w:sz'),'4');x.set(qn('w:color'),'D9D9D9');borders.append(x)
   cp.append(borders);mar=OxmlElement('w:tcMar')
   for edge in ['top','bottom','left','right']:
    x=OxmlElement('w:'+edge);x.set(qn('w:w'),'80');x.set(qn('w:type'),'dxa');mar.append(x)
   cp.append(mar)
   for q in c.paragraphs:
    q.paragraph_format.space_after=Pt(0);q.paragraph_format.line_spacing=Pt(14)
    q.alignment=WD_ALIGN_PARAGRAPH.LEFT if i==0 else WD_ALIGN_PARAGRAPH.CENTER
    for r in q.runs:r.font.size=Pt(10);r.font.bold=n==0
 p('').paragraph_format.space_after=Pt(0)

p('DeepSeek 六图流程补充测试报告','Title')
p('test6 搜题工程    2026年10月15日    本轮新增预算 7 元','Subtitle')
p('本轮加入硅基流动 deepseek-ai/DeepSeek-V4-Pro，沿用原有六图流程，比较解析呈现、首层出图和完整出图时延。对照来自 10 月 12 日已保存的四模型结果，没有重新付费运行基线，也没有做标准答案比对。')
p((P/'conclusion.txt').read_text(encoding='utf-8'))
h('测试覆盖与费用')
p(f"共评估 {s['cases']} 道固定题，实际模型请求 {b['calls']} 次，完整响应 {s['complete_responses']} 题，非图片原因的工程兜底 {s['fallback_excluding_incompatible']} 题，图片输入不兼容 {s['image_incompatible']} 题。完整响应中严格呈现校验通过 {s['strict_valid']} 题。")
p(f"本轮保守计费与预留 {b['accounted_cny']:.4f} 元，低于 7 元上限；已报告用量的估算费用 {b['known_estimated_cny']:.4f} 元。该新增预算独立于上轮预算，不把预留金额写成实际账单扣费。")
h('完整响应时延')
names={'qwen_max':'Qwen3.8-Max','glm_flash':'GLM-5.3-Flash','kimi':'Kimi-K2.6','qwen_27b':'Qwen3.8-27B'}
ss=[('DeepSeek-V4-Pro',s)]+[(names[k],v) for k,v in S['historical'].items()]
table(['模型','完整响应\n样本数','首层解析\n平均秒','六图齐全\n平均秒','最终完成\n平均秒'],[[n,v['complete_responses'],*[fmt(x) for x in v['mean_seconds'][1:]]] for n,v in ss],[1.85,1.05,1.4,1.4,1.4])
p('以上均值排除兜底、提前截流复用和图片不兼容请求，分母与测试日期不同。它们是描述性统计，不能直接证明模型等效或稳定性能排序。首层解析是第 2 张图；第 1 张图由本地 OCR 生成。','Caption')
h('相同输入条件下的同题配对')
v=S['same_input_paired']
p(f"共有 {v['n']} 题同时满足双方响应完整、OCR 文本一致及附图数量一致。相较历史 Qwen3.8-Max，DeepSeek 首层解析平均时延差 {fmt(v['delta_seconds'][0])} 秒，六图齐全差 {fmt(v['delta_seconds'][1])} 秒，最终完成差 {fmt(v['delta_seconds'][2])} 秒。正值表示更慢。附图数量一致不保证图像字节完全一致。")

d.add_page_break();h('测试条件与兼容性')
p('测试按原题图片、本地 OCR 与公式识别、真实模型调用、工程校验及修正、六图发布依次执行。没有用缓存 OCR 替代预处理，没有删除附图来伪造同条件成功。OCR 已提前预热，预热不产生云端模型费用。')
p('DeepSeek 采用 Non-think 设置，enable_thinking=false，最大输出 4096 tokens，模型等待预算 90 秒，学弱层后的预算 15 秒，关闭提供商修复重试。保留工程提示词和 JSON Schema。此结论仅适用于本次设置。')
p('六图时延取后端图片首次写入公开任务状态的时间，最终完成还包括后处理或重绘；不包含网页轮询、图片下载和浏览器绘制。网络、服务负载与测试日期差异可能影响时延。')
h('呈现检查与异常样本')
p((P/'quality_notes.txt').read_text(encoding='utf-8'))
p(f"本轮完整响应中的逐层发布表现以任务审计为准；全部评估样本里有 {s['progressive']} 题出现逐层发布记录。六张图存在并不等于模型完成解析，兜底页面和无效内容被分开记录。工程严格校验通过也不等于所有公式均排版正确或所有题目信息均完整。")
h('预算控制与数据来源')
p('硅基流动价格页在测试当天显示输入 12 元、输出 24 元／百万 tokens。估算不计缓存折扣；已知用量增加 10% 余量，未知用量保留调用前上限预留。每次调用前检查本轮累计预留，超过 7 元即停止。实际费用以供应商账单为准。')
p('价格来源：https://www.siliconflow.cn/pricing','Caption')
p('数据文件：summary.json、results_math.json、results_physics.json、ledger.json、manifest.json；每题 jobs 目录保存实际 PNG 与模型响应审计。历史对照读取 six_image_compare_20261012，未修改原始测试结果。','Caption')
p(f"PNG 文件核查：{S['file_audit']['checked_pngs']} 张通过解码与哈希检查，发现 {len(S['file_audit']['errors'])} 个文件问题。仅对代表性页面做人工呈现抽查，不宣称全部页面经过人工内容验收。")
p('本轮结论不涉及解题正确率，也不构成长期稳定性或统计等效证明。')

d.add_page_break();h('逐题结果明细')
p('数学和物理沿用上轮各 10 道固定样本。M 表示数学，P 表示物理，编号对应样本中的原题编号。')
tr=[]
for r in rows:
 c=r.get('provider_contract') or {};ts=r['first_public_seconds'];alltime=max(ts.values()) if len(ts)==6 else None
 kind='图片不兼容' if r.get('image_incompatible') else '兜底' if c.get('time_budget_fallback') else '提前截流' if c.get('partial_skeleton_reused') else '完整响应' if r['status']=='completed' else r['status']
 tr.append([r['case'].replace('TZIP-MATH-T01-Q','M').replace('TZIP-PHYS-T01-Q','P'),r.get('request_image_count'),kind,len(r['pngs']),fmt(alltime)])
table(['样本','附图数','结果类别','图数','六图齐全秒'],tr,[1.0,.9,2.5,.8,1.9])
p('兜底样本的六图时间仅表示工程输出了页面，不计作模型完成有效解析。未发起模型调用的图片不兼容样本不计费；接口拒绝情况及后续跳过策略保留在调用审计中。','Caption')
for root in [d.styles.element,d.element]:
 for border in list(root.iter(qn('w:pBdr'))):border.getparent().remove(border)
 for pr in root.iter(qn('w:rPr')):
  fs=pr.find(qn('w:rFonts'))
  if fs is None:fs=OxmlElement('w:rFonts');pr.insert(0,fs)
  for key in list(fs.attrib):
   if key.endswith('Theme'):del fs.attrib[key]
  for key in ['ascii','hAnsi','cs']:fs.set(qn('w:'+key),'Times New Roman')
  fs.set(qn('w:eastAsia'),'宋体')
d.core_properties.title='DeepSeek 六图流程补充测试报告'
out=P/'DeepSeek六图流程补充测试报告_宋体版.docx';d.save(out);print(out)
