from pathlib import Path
import json,statistics,hashlib
from PIL import Image
P=Path(__file__).resolve().parent
OLD=P.parent/'six_image_compare_20261012'
def read(p,default):return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
def rows(p):return [r for s in ['math','physics'] for r in read(p/f'results_{s}.json',{}).values()]
def good(r):
 c=r.get('provider_contract') or {}
 return bool(r['provider_called'] and r['status']=='completed' and len(r['pngs'])==6 and all(f.get('metadata',{}).get('valid') for f in r['pngs']) and c.get('model_tier_count')==5 and not c.get('time_budget_fallback') and not c.get('partial_skeleton_reused') and not r.get('image_incompatible'))
def mean(xs):return round(statistics.mean(xs),3) if xs else None
def times(r):
 t=r['first_public_seconds'];return [t.get('01_question_knowledge.png'),t.get('02_guidance_elite.png'),max(t.values()) if len(t)==6 else None,r['elapsed_seconds']]
def stats(rs):
 gs=[r for r in rs if good(r)]
 return {'cases':len(rs),'provider_calls':sum(r['provider_called'] for r in rs),'six_png':sum(len(r['pngs'])==6 for r in rs),'complete_responses':len(gs),'strict_valid':sum(bool((r.get('validation') or {}).get('strict_valid')) for r in gs),'fallback':sum(bool((r.get('provider_contract') or {}).get('time_budget_fallback')) for r in rs),'partial':sum(bool((r.get('provider_contract') or {}).get('partial_skeleton_reused')) for r in rs),'image_incompatible':sum(bool(r.get('image_incompatible')) for r in rs),'progressive':sum(any(v.get('published') for v in (r.get('streaming') or {}).get('guidance_tiers',{}).values()) for r in rs),'mean_seconds':[mean([times(r)[i] for r in gs if times(r)[i] is not None]) for i in range(4)]}
rs=rows(P);old=rows(OLD);ledger=read(P/'ledger.json',[])
summary={'model':'deepseek-ai/DeepSeek-V4-Pro','test_date':'2026-10-15','budget':{'limit_cny':7,'accounted_cny':sum(e['accounted_cny'] for e in ledger),'known_estimated_cny':sum(e.get('estimated_cny') or 0 for e in ledger),'calls':len(ledger),'pending':sum(e['status']=='pending' for e in ledger)},'deepseek':stats(rs),'historical':{k:stats([r for r in old if r['label']==k and r['provider_called']]) for k in ['qwen_max','glm_flash','kimi','qwen_27b']}}
base={r['case']:r for r in old if r['label']=='qwen_max' and good(r)}
pairs=[(r,base[r['case']]) for r in rs if good(r) and r['case'] in base]
same=[(a,b) for a,b in pairs if (a.get('preprocess') or {}).get('ocr_text')==(b.get('preprocess') or {}).get('ocr_text') and (a.get('provider_contract') or {}).get('request_image_count')==(b.get('provider_contract') or {}).get('request_image_count')]
def paired(ps):return {'n':len(ps),'deepseek_mean_seconds':[mean([times(a)[i] for a,b in ps]) for i in [1,2,3]],'baseline_mean_seconds':[mean([times(b)[i] for a,b in ps]) for i in [1,2,3]],'delta_seconds':[mean([times(a)[i]-times(b)[i] for a,b in ps]) for i in [1,2,3]]}
summary['paired']=paired(pairs);summary['same_input_paired']=paired(same)
summary['deepseek']['fallback_excluding_incompatible']=sum(bool((r.get('provider_contract') or {}).get('time_budget_fallback')) and not r.get('image_incompatible') for r in rs)
summary['deepseek']['image_incompatible_called']=sum(bool(r.get('image_incompatible')) and r['provider_called'] for r in rs)
summary['deepseek']['image_incompatible_skipped']=sum(bool(r.get('image_incompatible')) and not r['provider_called'] for r in rs)
checked=0;errors=[]
for r in rs:
 for f in r['pngs']:
  p=Path(r['job_dir'])/f['name']
  try:
   with Image.open(p) as im:im.verify()
   assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
   checked+=1
  except Exception as exc:errors.append({'file':str(p),'error':str(exc)})
summary['file_audit']={'checked_pngs':checked,'errors':errors}
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# DeepSeek 六图流程补充测试报告','','测试日期：2026年10月15日。新增预算上限 7 元，独立于上轮 10 元预算。仅比较解析呈现与时延，不做标准答案对比。','','## 型号与范围','','新增模型：`deepseek-ai/DeepSeek-V4-Pro`，通过硅基流动 API 调用。账号模型列表已确认该 ID 可用。采用 Non-think 配置，enable_thinking=false，max_tokens=4096；保留原工程提示词、JSON Schema、90 秒预算、学弱层后 15 秒预算及原图输入策略。未启用提供商修复重试。','','模型价格按输入 12 元、输出 24 元／百万 tokens 计算，未计缓存折扣，已知用量增加 10% 余量，未知用量保留请求前上限预留。价格核实来源：[硅基流动价格页](https://www.siliconflow.cn/pricing)。','','测试使用上轮固定的数学 10 题、物理 10 题。每次从原题执行真实 OCR、模型调用、校验、修正与六图发布；不使用缓存 OCR 替代真实预处理。预热 OCR 的本地调用不计入模型时延。与 10 月 12 日保存的基线比较，未新增基线付费调用，日期及网络负载差异可能影响时延。','','## 覆盖与时延','','| 模型 | 评估题数 | 实际调用 | 六图文件 | 完整响应 | 严格校验通过 | 首层解析均秒 | 六图齐全均秒 | 最终完成均秒 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,s in [('DeepSeek-V4-Pro',summary['deepseek'])]+list(summary['historical'].items()):
 lines.append(f"| {name} | {s['cases']} | {s['provider_calls']} | {s['six_png']} | {s['complete_responses']} | {s['strict_valid']}/{s['complete_responses']} | {s['mean_seconds'][1]} | {s['mean_seconds'][2]} | {s['mean_seconds'][3]} |")
lines += ['', '均值限完整响应，不含通用兜底、提前截流复用和不兼容调用。样本集不同，不应直接将非配对均值之差解释为模型性能差距。六图文件存在及尺寸合规不等于解析正确或信息完整。首层解析为第 2 张图，非本地 OCR 的第 1 张图。后端发布计时不含浏览器显示。','','## 同题配对','','| 条件 | 题数 | 首层时延差均秒 | 六图齐全时延差均秒 | 最终完成时延差均秒 |','|---|---:|---:|---:|---:|']
for name,key in [('双方完整响应','paired'),('另要求 OCR 文本及附图数量相同','same_input_paired')]:
 s=summary[key];lines.append(f"| {name} | {s['n']} | {s['delta_seconds'][0]} | {s['delta_seconds'][1]} | {s['delta_seconds'][2]} |")
lines += ['', '差值为 DeepSeek 减历史基线。附图数量相同不代表图像字节完全相同，未据此宣称统计等效。','','## 费用','','本轮保守计费与预留：%.6f / 7 元；已报告用量估算：%.6f 元；待返回调用：%d 次。最终实际扣费以供应商账单为准。'%(summary['budget']['accounted_cny'],summary['budget']['known_estimated_cny'],summary['budget']['pending']),'','## 逐题记录','','| 题目 | 状态 | 是否调用 | 附图数 | 图数 | 完整响应 | 图片不兼容 | 首层解析秒 | 六图齐全秒 |','|---|---|---|---:|---:|---|---|---:|---:|']
for r in rs:lines.append(f"| {r['case']} | {r['status']} | {r['provider_called']} | {r.get('request_image_count')} | {len(r['pngs'])} | {good(r)} | {r.get('image_incompatible')} | {times(r)[1]} | {times(r)[2]} |")
lines += ['', '若图片请求被服务端明确拒绝，后续相同能力需求的请求在发送前跳过，仍保留 OCR 和不兼容状态记录。没有删除图像以伪造同条件测试成功。','','原始资料：results_math.json、results_physics.json、ledger.json、manifest.json；每题 jobs 目录内保留六图和 provider 审计；图片文件核查结果见 summary.json。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for note in ['conclusion.txt','quality_notes.txt']:
 if (P/note).exists():
  with (P/'REPORT.md').open('a',encoding='utf-8') as f:f.write('\n'+(P/note).read_text(encoding='utf-8')+'\n')
print(json.dumps(summary,ensure_ascii=False))
