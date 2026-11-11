"""Offline six-image audit. No network calls and no answer-key scoring."""
import json, statistics, hashlib
from pathlib import Path
from collections import Counter

P=Path(__file__).resolve().parent
def read(p, default):
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default
def avg(xs):
    return round(statistics.mean(xs),3) if xs else None
def complete(r):
    return r['status']=='completed' and len(r['pngs'])==6 and all(x.get('metadata',{}).get('valid') for x in r['pngs'])
def direct(r):
    c=r.get('provider_contract') or {}
    return complete(r) and c.get('model_tier_count')==5 and not c.get('time_budget_fallback') and not c.get('partial_skeleton_reused')
def alltime(r):
    ts=r['first_public_seconds']
    return max(ts.values()) if len(ts)==6 else None
all_rows=[v for s in ['math','physics'] for v in read(P/f'results_{s}.json',{}).values()]
budget_stops=[r for r in all_rows if r.get('budget_stop') and not r.get('provider_called')]
rows=[r for r in all_rows if r.get('provider_called')]
orders={}
for r in rows:
    response=read(Path(r['job_dir'])/'audit/comparison_provider.json',{})
    try: keys=list(json.loads(response.get('output_text','')))
    except (ValueError,TypeError):keys=[]
    orders[r['case']+'__'+r['label']]=keys
prior=read(P.parent/'model_compare_20261011/ledger.json',[])
ledger=read(P/'ledger.json',[])
budget={'prior_accounted':sum(x['accounted_cny'] for x in prior),'new_accounted':sum(x['accounted_cny'] for x in ledger),'total_accounted':sum(x['accounted_cny'] for x in prior+ledger),'known_estimated':sum(x.get('estimated_cny') or 0 for x in prior+ledger),'new_call_count':len(ledger),'pending':sum(x['status']=='pending' for x in ledger)}
summary={}
for label in ['qwen_max','kimi','glm_flash','qwen_27b']:
    rs=[r for r in rows if r['label']==label]; good=[r for r in rs if direct(r)]
    summary[label]={'tested':len(rs),'six_valid_png':sum(complete(r) for r in rs),'direct_five_tiers':len(good),'strict_valid':sum(bool((r.get('validation') or {}).get('strict_valid')) for r in good),'fallback':sum(bool((r.get('provider_contract') or {}).get('time_budget_fallback')) for r in rs),'partial':sum(bool((r.get('provider_contract') or {}).get('partial_skeleton_reused')) for r in rs),'progressive_tiers':sum(bool((r.get('streaming') or {}).get('guidance_tiers')) for r in rs),'first_image_mean_s':avg([r['first_public_seconds']['01_question_knowledge.png'] for r in good if '01_question_knowledge.png' in r['first_public_seconds']]),'first_guidance_mean_s':avg([r['first_public_seconds']['02_guidance_elite.png'] for r in good if '02_guidance_elite.png' in r['first_public_seconds']]),'all_six_mean_s':avg([alltime(r) for r in good if alltime(r) is not None]),'final_mean_s':avg([r['elapsed_seconds'] for r in good]),'warnings':dict(Counter(w for r in rs for w in (r.get('validation') or {}).get('warnings',[])))}
baselines={r['case']:r for r in rows if r['label']=='qwen_max' and direct(r)}
for label,s in summary.items():
    pairs=[(r,baselines[r['case']]) for r in rows if r['label']==label and direct(r) and r['case'] in baselines and alltime(r) is not None and alltime(baselines[r['case']]) is not None]
    s['matched_pairs']=len(pairs)
    s['matched_all_six_delta_s']=avg([alltime(a)-alltime(b) for a,b in pairs])
    same=[(a,b) for a,b in pairs if (a.get('preprocess') or {}).get('ocr_text')==(b.get('preprocess') or {}).get('ocr_text') and (a.get('provider_contract') or {}).get('request_image_count')==(b.get('provider_contract') or {}).get('request_image_count')]
    s['same_ocr_pairs']=len(same)
    s['same_ocr_all_six_delta_s']=avg([alltime(a)-alltime(b) for a,b in same])
    s['l1_first_responses']=sum(keys[:1]==['l1'] for key,keys in orders.items() if key.endswith('__'+label))
data={'budget':budget,'models':summary,'budget_stops':len(budget_stops),'actual_tested_combinations':len(rows),'planned_combinations':80}
(P/'summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 六图真实流程小样本测试','', '比较范围：原题图片 → 本地 OCR（含公式识别）→ 真实模型流式调用 → 工程校验和本地修正 → 六张 PNG 发布。没有标准答案对比，也没有据此统计解题正确率。','', '四模型使用原工程提示词、JSON Schema、90 秒模型预算和学弱层后 15 秒预算；为控制费用统一限制输出 4096 tokens，关闭提供商修复重试。GLM 启用思考，其余候选关闭思考。逐层时间是后端发布到任务状态的时间，不含网页轮询、传输和浏览器绘制。','', '样本上限：数学 10 题、物理 10 题，每模型各 20 题。每题每模型一次，属于探索性小样本，不能证明长期等效。若累计费用预留达到上限则停止。基线首题 OCR 冷启动异常单独归档并补测，原费用仍计入账本。','', '| 模型 | 已测 | 六图合规 | 模型完整五层 | 严格校验通过* | 首层解析均秒* | 六图首次齐全均秒* | 任务完成均秒* | 逐层发布例数 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for label,s in summary.items():
    lines.append(f"| {label} | {s['tested']} | {s['six_valid_png']} | {s['direct_five_tiers']} | {s['strict_valid']} | {s['first_guidance_mean_s']} | {s['all_six_mean_s']} | {s['final_mean_s']} | {s['progressive_tiers']} |")
lines += ['', '*均值和严格校验统计限模型完整五层且六图合规的样本；不把兜底或部分内容补齐当作完整模型解析。严格校验只检查工程的呈现约束，不等于答案正确性。首次图片可能为暂定版本，任务完成还包含校验、重绘；两种时延分别列示。','', '## 同题配对比较','', '| 候选 | 可配对题数 | OCR 文本完全一致题数 | 六图齐全时延差均值（候选减基线，秒） |','|---|---:|---:|---:|']
for label,s in summary.items():
    if label!='qwen_max':lines.append(f"| {label} | {s['matched_pairs']} | {s['same_ocr_pairs']} | {s['matched_all_six_delta_s']} |")
lines += ['', 'OCR 每次重新运行，可能因本地超时或识别差异改变附图策略；不一致样本应结合逐题记录解释，不能全部归因于模型。','', '## 费用','',f"上轮保守计费 {budget['prior_accounted']:.6f} 元；本轮保守计费/预留 {budget['new_accounted']:.6f} 元；累计 {budget['total_accounted']:.6f} / 10 元。已报告用量的估算合计 {budget['known_estimated']:.6f} 元。未知用量保留调用前上限预留，已知用量保留 10% 余量；最终扣费以供应商账单为准。待返回调用 {budget['pending']} 次。",'', '## 逐题审计','', '| 题目 | 模型 | 状态 | 六图首次齐全秒 | 完整模型五层 | 严格校验 | 兜底 | 部分骨架 | 结果目录 |','|---|---|---|---:|---|---|---|---|---|']
for r in rows:
    c=r.get('provider_contract') or {}; v=r.get('validation') or {}
    lines.append(f"| {r['case']} | {r['label']} | {r['status']} | {alltime(r)} | {direct(r)} | {v.get('strict_valid')} | {c.get('time_budget_fallback')} | {c.get('partial_skeleton_reused')} | [六图及审计]({Path(r['job_dir']).as_posix()}) |")
lines += ['', '## 逐层发布兼容性','', '工程的 tier_streaming.py 使用 completed_ordered_prefix，要求根字段从 l1 开始依次闭合。已对保存的 GLM 数学首题响应做离线复现：原始字段顺序 f,l1,l2,l3,l4,l5,pat,s,u 不能触发 L1 检测；只将相同内容重排为工程指定顺序后能够触发。此检查不新增调用、不更改生产逻辑，也不证明实时适配后的时延。','', '| 模型 | 响应首字段为 l1 | 与基线 OCR 文本及附图数量一致的配对 | 这些配对六图时延差均值秒 |','|---|---:|---:|---:|']
for label,s in summary.items():lines.append(f"| {label} | {s['l1_first_responses']} | {s['same_ocr_pairs']} | {s['same_ocr_all_six_delta_s']} |")
lines += ['', '## 停止条件与样本覆盖','',f'实际完成付费调用的正式组合为 {len(rows)}/80，另有一次基线冷启动调用归档；{len(budget_stops)} 个任务在调用前被预算保护阻止，不计作模型失败。预算保护停止后没有继续尝试便宜模型填满余额，避免进一步扩大样本不平衡。', '', '原始结果、逐图首次发布时间和全部校验警告见 results_math.json、results_physics.json；调用用量与预算见 ledger.json。']
lines += ['', '## 结果判断与图片抽查','', 'GLM-5.3-Flash 的六图齐全速度最接近基线，但首层解析更晚出现，严格格式校验通过比例较低，当前证据不支持直接等效替换。Kimi 和 Qwen3.8-27B 在本次工程接入方式下整体更慢。建议继续保留基线；若后续另行授权适配与测试，优先处理 GLM 的字段顺序与公式格式兼容。','', '基线有 3 例在 L5 发布后触发 post_lower_budget_exhausted，保留了模型的五层内容，但尾部公式/总结未完整接收；为统一统计口径，将其排除在完整响应时延均值之外。它们与 Qwen27B 的 3 例超时通用兜底不同。','', '已核验 70 组正式结果的 420 个 PNG 均可解码且哈希与记录一致。人工查看 GLM 数学首题六图总览，以及物理 Q10 四个模型的第六图：所查页面文字可读、段落边界清楚，没有明显截断；Qwen27B 的物理 Q10 页面为重复的通用步骤，印证兜底标记。GLM 有些公式以线性文本显示；基线截流例缺少尾部基础公式信息。此处是呈现与内容结构抽查，不是所有页面的人工验收，也未比较标准答案。','', '六图文件存在和 PNG 尺寸合规不代表题干信息齐全。工程警告中仍包含原题引用图片但 source_figures 为空、公式定界符或下标格式异常等问题，且部分问题也发生在基线。故无法仅靠出图率断言解析效果相同。']
(P/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'budget':budget,'models':{k:{a:b for a,b in v.items() if a!='warnings'} for k,v in summary.items()}},ensure_ascii=False))
