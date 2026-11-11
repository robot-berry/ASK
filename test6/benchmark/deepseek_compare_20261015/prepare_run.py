from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
old=P.parent/'six_image_compare_20261012/run_six.py'
t=old.read_text(encoding='utf-8')
def change(a,b):
 global t
 assert a in t,a[:80]
 t=t.replace(a,b)
change('Every provider call reserves budget against BOTH comparison rounds before POST.','Every provider call reserves budget against the new authorized CNY 7 round before POST.')
change('from run_compare import MODELS, credentials, save',"from run_compare import credentials, save\nMODELS=[{'label':'deepseek_pro','id':'deepseek-ai/DeepSeek-V4-Pro','provider':'siliconflow','input':12.0,'output':24.0}]\nLIMIT=7.0")
change("    prior=json.loads((PRIOR/'ledger.json').read_text(encoding='utf-8'))\n    return sum(e['accounted_cny'] for e in prior+entries)","    return sum(e['accounted_cny'] for e in entries)")
change("        if total_reserved(entries)+reserve>10:","        if total_reserved(entries)+reserve>LIMIT:")
change('Cumulative 10 CNY budget would be exceeded','New round 7 CNY budget would be exceeded')
change("        entries=ledger()\n", "        context['request_image_count']=images\n        if images and (OUT/'image_incompatible.json').exists():\n            context['image_incompatible']=True\n            raise RuntimeError('Previously verified image input incompatibility; paid call skipped')\n        entries=ledger()\n")
change("        result=original_attempt(request,base,key,workspace,timeout,secrets,*args)","        try:\n            result=original_attempt(request,base,key,workspace,timeout,secrets,*args)\n        except Exception as exc:\n            entry.update({'status':'exception_reserved','error_type':type(exc).__name__})\n            save(OUT/'ledger.json',entries)\n            raise")
change("        if result.get('http_status') in (400,401,402,403,404,422):context['provider_blocked']=True", "        if result.get('http_status') in (400,422) and images:\n            error_text=json.dumps(result,ensure_ascii=False).lower()\n            if any(x in error_text for x in ['image','vision','multimodal']):\n                context['image_incompatible']=True\n                save(OUT/'image_incompatible.json',{'job_id':context['job_id'],'case':context['case']['id'],'http_status':result.get('http_status'),'audit_path':str(context['job_dir']/'audit'/'comparison_provider.json')})\n            else:context['provider_blocked']=True\n        elif result.get('http_status') in (400,401,402,403,404,422):context['provider_blocked']=True")
change("'model_response':final.get('model_id_response'),'budget_stop':context.get('budget_stop',False)","'model_response':final.get('model_id_response'),'budget_stop':context.get('budget_stop',False),\n                 'image_incompatible':context.get('image_incompatible',False),'request_image_count':context.get('request_image_count')")
change("        for model in MODELS[i%4:]+MODELS[:i%4]:","        for model in MODELS:")
change("            if context.get('provider_blocked'):disabled.add(model['label'])","            if context.get('provider_blocked'):return 3")
(P/'run_six.py').write_text(t,encoding='utf-8')
root=P.parents[1]
manifest={'budget_cny':7,'model':'deepseek-ai/DeepSeek-V4-Pro','input_cny_per_million':12,'output_cny_per_million':24,'price_source':'https://www.siliconflow.cn/pricing','price_checked':'2026-10-15','new_budget_separate_from_prior':True,'baseline':'six_image_compare_20261012 historical results, not simultaneous','planned_cases':20,'source_sha256':{}}
for sub in ['math','physics']:
 for f in (root/sub/'source').glob('*.py'):
  manifest['source_sha256'][str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
(P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared independent 7 CNY runner; no paid calls yet')
