from pathlib import Path
import ast,json
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
P=Path(__file__).resolve().parent; B=P.parent
def load(folder):return json.loads((B/folder/'summary.json').read_text(encoding='utf-8'))
a=load('model_compare_20261011'); b=load('six_image_compare_20261012'); c=load('deepseek_compare_20261015')
d=Document(B/'six_image_compare_20261012/搜题工程六图流程模型测试报告_宋体版.docx')
for e in list(d.element.body):
 if e.tag!=qn('w:sectPr'):d.element.body.remove(e)
d.sections[0].header.paragraphs[0].text='TEST6 搜题工程  三轮模型测试汇总'
d.sections[0].footer.paragraphs[0].text='2026年10月16日    模型测试汇总'
tree=ast.parse((B/'deepseek_compare_20261015/build_word.py').read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['p','h','fmt','table']],type_ignores=[]),'<report-format>','exec'))
names={'qwen_max':'Qwen3.8-Max','glm_flash':'GLM-5.3-Flash','kimi':'Kimi-K2.6','qwen_27b':'Qwen3.8-27B'}
keys=list(names)
p('搜题工程模型测试汇总报告','Title')
p('汇总日期 2026年10月16日    范围 三轮测试与五个模型','Subtitle')
p('本报告汇总模型解析层测试、完整六图流程测试和 DeepSeek 补充测试，用于判断候选模型能否达到当前工程的解析呈现与时延要求。所有结论均来自已保存结果；本次汇总没有新增模型调用，也没有比对标准答案或统计解题正确率。')
p('总体结论：保留 Qwen3.8-Max 作为现有基线。GLM-5.3-Flash 最值得继续适配验证，但尚无直接等效替换的证据。Kimi-K2.6 和 Qwen3.8-27B 在本次接入设置下更慢；DeepSeek-V4-Pro 不接受图片输入，不能直接覆盖当前完整流程。')
h('三轮测试范围')
table(['轮次','测试内容','覆盖与调用'],[['第一轮','模型解析层','20题 × 4模型，80组合；85次请求含失败和重试'],['第二轮','完整六图流程','正式完成70/80组合；71次请求含1次冷启动'],['第三轮','DeepSeek补充','评估20题；11次请求，另9题调用前跳过']],[.85,1.5,4.75])
p('第一轮结果目录日期为 10 月 11 日；第二轮为 10 月 12 日；DeepSeek 实测为 10 月 15 日，10 月 16 日完成复核。各轮重复使用同一组数学10题、物理10题，不应将累计组合数理解为独立题目数量。','Caption')
h('完整六图的核心时延')
stats=[(names[k],c['historical'][k]) for k in keys]+[('DeepSeek-V4-Pro',c['deepseek'])]
table(['模型','完整响应\n样本数','首层解析\n平均秒','六图齐全\n平均秒','最终完成\n平均秒'],[[name,v['complete_responses'],*[fmt(x) for x in v['mean_seconds'][1:]]] for name,v in stats],[1.85,1.05,1.4,1.4,1.4])
p('时延均值排除兜底、提前截流复用和图片不兼容样本。各模型样本和测试日期不同，横向均值仅作描述；同题配对结果见下一页。首层解析指第2张图，不是本地OCR生成的第1张图。','Caption')

d.add_page_break();h('第一轮模型解析层结果')
table(['模型','完整返回','五层齐全','Schema通过','完整返回\n平均秒'],[[names[k],f"{a['models'][k]['completed']}/20",f"{a['models'][k]['five_tiers']}/20",f"{a['models'][k]['schema_pass']}/20",fmt(a['models'][k]['full_seconds']['mean'])] for k in keys],[1.85,1.2,1.2,1.3,1.55])
p('第一轮主要比较模型输出结构和响应时间，未计入重新OCR、工程逐层发布与最终六图渲染。其Schema检查与后续工程严格校验不是同一指标，不能混合计算通过率或平均时延。GLM有1例未完成，Qwen27B有2例传输超时；仅完整返回样本计入该表时延均值。','Caption')
h('完整流程的质量与兼容性')
table(['模型','实际请求\n正式题数','完整响应','严格校验\n通过¹','主要异常'],[[names[k],c['historical'][k]['provider_calls'],c['historical'][k]['complete_responses'],f"{c['historical'][k]['strict_valid']}/{c['historical'][k]['complete_responses']}",{'qwen_max':'3例提前截流复用','glm_flash':'无逐层发布记录','kimi':'无逐层发布记录','qwen_27b':'3例超时兜底'}[k]] for k in keys]+[['DeepSeek-V4-Pro',11,9,'8/9','1例截断；10题图片不兼容²']],[1.7,1.0,1.0,1.2,2.2])
p('¹ 严格校验限完整响应，检查工程格式及呈现约束，不代表答案正确率。² DeepSeek图片不兼容包括1次服务端拒绝和9次调用前跳过；首次拒绝还触发本地兜底，不能重复计为额外失败。','Caption')
p('基线3例截流发生在第五层已发布后，保留五层模型内容，但尾部公式或总结未完整接收。GLM、Kimi、Qwen27B以及DeepSeek在本次完整流程中均无逐层发布记录；已保存响应的字段顺序与工程从l1开始的解析要求不一致。离线重排可触发检测，但适配后的实时性能尚未实测。')
h('与基线相同输入条件的配对')
table(['候选模型','配对题数','六图齐全平均时延差'],[[names[k],b['models'][k]['same_ocr_pairs'],f"{b['models'][k]['same_ocr_all_six_delta_s']:+.2f} 秒"] for k in ['glm_flash','kimi','qwen_27b']]+[['DeepSeek-V4-Pro',c['same_input_paired']['n'],f"{c['same_input_paired']['delta_seconds'][1]:+.2f} 秒"]],[2.5,1.25,3.35])
p('仅比较双方响应完整、OCR文本和附图数量相同的同题样本。差值为候选减基线，负值更快。附图数量相同不保证图片字节相同，DeepSeek还存在跨日期比较限制。','Caption')

d.add_page_break();h('费用汇总与预算边界')
costs=[]
for folder,label in [('model_compare_20261011','第一轮 模型层'),('six_image_compare_20261012','第二轮 六图'),('deepseek_compare_20261015','第三轮 DeepSeek')]:
 ledger=json.loads((B/folder/'ledger.json').read_text(encoding='utf-8'))
 costs.append({'round':label,'calls':len(ledger),'estimated':sum(x.get('estimated_cny') or 0 for x in ledger),'accounted':sum(x['accounted_cny'] for x in ledger),'pending':sum(x.get('status')=='pending' for x in ledger)})
total=sum(x['accounted'] for x in costs);known=sum(x['estimated'] for x in costs)
table(['阶段','请求次数','已知用量估算元','保守计费及预留元'],[[x['round'],x['calls'],f"{x['estimated']:.4f}",f"{x['accounted']:.4f}"] for x in costs]+[['合计',sum(x['calls'] for x in costs),f'{known:.4f}',f'{total:.4f}']],[2.0,1.0,1.95,2.15])
p('第一轮与第二轮共用最初的10元预算，合计保守计费与预留9.7656元；第三轮使用新增7元预算，计0.6574元。两次授权上限合计17元，不能把第二轮累计金额再与第一轮重复相加。所有请求均已结束，没有待返回调用。')
p('已知用量估算不包含未知用量请求的实际账单；未知用量保留调用前上限预留，已知用量增加10%余量。累计保守计费与预留不是实际扣费，最终以供应商账单为准。本次整理不产生新的模型调用费用。','Caption')
h('选型建议')
p('当前继续使用 Qwen3.8-Max。GLM-5.3-Flash 六图齐全速度最接近基线，但首层呈现晚、公式格式兼容仍需改进；若安排后续工作，优先验证字段顺序和公式排版适配，再重复检查首层与六图时延。这里的 GLM 指 Flash 版本，并非 GLM-5.3 旗舰版。')
p('Kimi-K2.6和Qwen3.8-27B暂不作为直接替换方案。DeepSeek-V4-Pro虽能处理部分OCR文本题，但服务端实测拒绝图片输入；更改为纯文本专用路线属于新的集成方案，不能用本轮结果宣称完整六图流程等效。')
h('适用范围与限制')
p('样本为非随机的高中数学和物理题，单组合主要测试一次。完整流程采用4096-token输出上限、90秒模型预算、学弱层后15秒预算，并关闭提供商修复重试。结果只适用于该配置，不代表各模型全部能力或长期稳定性。后端发布计时不含网页轮询、下载及浏览器绘制。')
p('六图文件合规不等于题目信息完整。抽查和工程警告发现原始公式文本、下标或定界符异常、引用插图未带入、部分小问覆盖不足等现象；基线也有部分问题。DeepSeek的完整文本响应中同样有题目提示无法获取完整示意图。报告未做标准答案比对。')
h('可追溯资料')
p('资料均位于 test6/benchmark：model_compare_20261011、six_image_compare_20261012、deepseek_compare_20261015。各目录内的summary.json、ledger.json、REPORT.md及逐题响应是本汇总依据；第二、三轮还保存实际PNG和任务审计。','Caption')
for root in [d.styles.element,d.element]:
 for pr in root.iter(qn('w:rPr')):
  fs=pr.find(qn('w:rFonts'))
  if fs is None:fs=OxmlElement('w:rFonts');pr.insert(0,fs)
  for key in list(fs.attrib):
   if key.endswith('Theme'):del fs.attrib[key]
  for key in ['ascii','hAnsi','cs']:fs.set(qn('w:'+key),'Times New Roman')
  fs.set(qn('w:eastAsia'),'宋体')
d.core_properties.title='搜题工程模型测试汇总报告'
out=P/'搜题工程三轮模型测试汇总报告_宋体版.docx';d.save(out)
(P/'cost_summary.json').write_text(json.dumps({'rounds':costs,'total_accounted':total,'total_known_estimated':known,'authorized_budget':17},ensure_ascii=False,indent=2),encoding='utf-8')
print(out);print('costs',total,known)
