"""Read-only analysis of saved provider runs. Does not modify the budget ledger."""
from pathlib import Path
import csv, json, math, statistics

ROOT=Path(__file__).resolve().parent
LABELS=['qwen_max','kimi','glm_flash','qwen_27b']

def metric(values):
    if not values:return {'n':0}
    s=sorted(values)
    return {'n':len(s),'mean':round(statistics.mean(s),3),'median':round(statistics.median(s),3),
            'p90':round(s[max(0,math.ceil(.9*len(s))-1)],3),'min':min(s),'max':max(s)}

def sentences(p,tier):
    value=p.get(tier,[])
    if not isinstance(value,list):return []
    if tier=='l2':return [s for s in value if isinstance(s,str)]
    return [s for part in value if isinstance(part,dict) for s in part.get('s',[]) if isinstance(s,str)]

def main():
    rows=[json.loads(f.read_text(encoding='utf-8')) for f in sorted(ROOT.glob('TZIP*__*.json'))]
    ledger=json.loads((ROOT/'ledger.json').read_text(encoding='utf-8'))
    cases=json.loads((ROOT/'cases.json').read_text(encoding='utf-8'))
    expected={c['id'] for c in cases}
    audits=[]
    for r in rows:
        p=r.get('parsed') or {}
        tiers={k:sentences(p,k) for k in ['l1','l2','l3','l4','l5']}
        labels=[[part.get('n') for part in p.get(k,[]) if isinstance(part,dict)] for k in ['l1','l3','l4','l5']]
        repeated=sorted({s for i,k in enumerate(tiers) for j,k2 in enumerate(tiers) if i<j for s in set(tiers[k])&set(tiers[k2])})
        audit={k:r.get(k) for k in ['case','label','status','http_status','error_code','error_message','elapsed_seconds','first_content_seconds','tier_seconds','usage','estimated_cny','schema_errors','five_tiers_present','thinking_budget','enable_thinking','image_count']}
        audit.update({'matching_part_labels':bool(labels[0]) and all(x==labels[0] for x in labels),
                      'tier_char_counts':{k:sum(map(len,v)) for k,v in tiers.items()},
                      'l1_formula_markup':[s for s in tiers['l1'] if '$' in s],
                      'l2_formula_markup':[s for s in tiers['l2'] if '$' in s],
                      'formula_field_dollar_markup':[x.get('x','') for x in p.get('f',[]) if isinstance(x,dict) and '$' in x.get('x','')],
                      'verbatim_cross_tier_duplicates':repeated})
        audits.append(audit)
    by_key={(r['case'],r['label']):r for r in rows}
    summaries={}
    for label in LABELS:
        rr=[r for r in rows if r['label']==label]; aa=[a for a in audits if a['label']==label]
        ok=[r for r in rr if r['status']=='ok']; pairs=[(by_key[(r['case'],'qwen_max')],r) for r in ok if (r['case'],'qwen_max') in by_key and by_key[(r['case'],'qwen_max')]['status']=='ok']
        summaries[label]={'attempted_cases':len(rr),'completed':len(ok),'missing_cases':sorted(expected-{r['case'] for r in rr}),
                          'schema_pass':sum(not r.get('schema_errors') for r in ok),
                          'five_tiers':sum(bool(r.get('five_tiers_present')) for r in ok),
                          'matching_parts':sum(a['matching_part_labels'] for a in aa if a['status']=='ok'),
                          'l1_formula_flag_count':sum(bool(a['l1_formula_markup']) for a in aa if a['status']=='ok'),
                          'l2_formula_flag_count':sum(bool(a['l2_formula_markup']) for a in aa if a['status']=='ok'),
                          'full_seconds':metric([r['elapsed_seconds'] for r in ok]),
                          'all_terminal_seconds':metric([r['elapsed_seconds'] for r in rr]),
                          'first_content_seconds':metric([r['first_content_seconds'] for r in ok if r.get('first_content_seconds') is not None]),
                          'l1_seconds':metric([r['tier_seconds']['l1'] for r in ok if 'l1' in r.get('tier_seconds',{})]),
                          'l5_seconds':metric([r['tier_seconds']['l5'] for r in ok if 'l5' in r.get('tier_seconds',{})]),
                          'paired_full_ratio_vs_max':metric([b['elapsed_seconds']/a['elapsed_seconds'] for a,b in pairs]),
                          'matched_max_full_seconds':metric([a['elapsed_seconds'] for a,b in pairs]),
                          'paired_cases':len(pairs), 'errors':[{'case':r['case'],'status':r['status'],'http':r.get('http_status'),'code':r.get('error_code')} for r in rr if r['status']!='ok']}
    analysis={'status':'collection_finished_with_failures' if all(s['attempted_cases']==20 for s in summaries.values()) and not any(e['status']=='pending' for e in ledger) else 'incomplete',
              'calls_including_retries':len(ledger),'known_usage_estimate_cny':sum(e.get('estimated_cny') or 0 for e in ledger),
              'accounted_with_reserves_cny':sum(e['accounted_cny'] for e in ledger),'budget_cny':10,
              'pending_calls':sum(e['status']=='pending' for e in ledger),'models':summaries}
    (ROOT/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'quality_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 逐题模型解析审阅', '', '仅比较分层解析，不做标准答案评分；保留模型原文。']
    for c in cases:
        lines+=['',f'## {c["id"]}', '', c['text'], '']
        for label in LABELS:
            r=by_key.get((c['id'],label))
            lines += [f'### {label}', '']
            if not r:
                lines += ['尚无响应',''];continue
            lines += [f'状态：{r["status"]}；耗时：{r["elapsed_seconds"]}s；结构问题：{r.get("schema_errors")}', '']
            if 'parsed' not in r: continue
            for tier in ['l1','l2','l3','l4','l5']:
                lines += [f'**{tier.upper()}**','']+[f'- {s}' for s in sentences(r['parsed'],tier)]+['']
    (ROOT/'ALL_RESPONSES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(analysis,ensure_ascii=False))

if __name__=='__main__': main()
