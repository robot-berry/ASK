"""Run the real Test6 OCR -> live streaming -> validation -> six PNG pipeline.

Isolated worker per subject avoids module-name collisions. No production edits.
Every provider call reserves budget against the new authorized CNY 7 round before POST.
"""
from pathlib import Path
import argparse, copy, hashlib, json, os, subprocess, sys, time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PRIOR=OUT.parent/'model_compare_20261011'
sys.path.insert(0,str(PRIOR))
from run_compare import credentials, save
MODELS=[{'label':'deepseek_pro','id':'deepseek-ai/DeepSeek-V4-Pro','provider':'siliconflow','input':12.0,'output':24.0}]
LIMIT=7.0
NAMES=['01_question_knowledge.png','02_guidance_elite.png','03_guidance_upper.png',
       '04_guidance_middle.png','05_guidance_lower.png','06_guidance_novice.png']

def ledger():
    return json.loads((OUT/'ledger.json').read_text(encoding='utf-8')) if (OUT/'ledger.json').exists() else []

def total_reserved(entries):
    return sum(e['accounted_cny'] for e in entries)

class BudgetStop(Exception): pass

def worker(subject,limit,dry):
    os.environ['TEST6_FORMULA_OCR_URL']='http://127.0.0.1:18240'
    sys.path.insert(0,str(ROOT/subject/'source'))
    import search_machine_pipeline as pl
    from PIL import Image
    import requests
    health=requests.get(os.environ['TEST6_FORMULA_OCR_URL']+'/health',timeout=5).json()
    if not health.get('ready'): raise RuntimeError('Test OCR service not ready')
    creds=credentials()
    cases=[c for c in json.loads((PRIOR/'cases.json').read_text(encoding='utf-8')) if c['subject']==subject][:limit]
    state_path=OUT/f'results_{subject}.json'
    results=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    context={}
    original_attempt=pl.benchmark.perform_http_attempt

    def attempt(payload,base,key,workspace,timeout,secrets,*args):
        model=context['model']; request=copy.deepcopy(payload)
        token_limit=4096; thinking=1024 if model['provider']=='siliconflow' else 0
        request.pop('max_completion_tokens',None)
        request['max_tokens']=token_limit
        if model['provider']=='siliconflow':
            request['thinking_budget']=thinking
            request['enable_thinking']=model['label']=='glm_flash'
        # Exact production prompt + JSON Schema retained, only vendor parameter names adapted.
        text_bytes=0; images=0
        for message in request['messages']:
            content=message.get('content','')
            if isinstance(content,str): text_bytes+=len(content.encode('utf-8'))
            else:
                for part in content:
                    if part.get('type')=='text':text_bytes+=len(part['text'].encode('utf-8'))
                    elif part.get('type')=='image_url':images+=1
        text_bytes+=len(json.dumps(request.get('response_format',{}),ensure_ascii=False).encode('utf-8'))
        bound=text_bytes+2048+images*32768
        reserve=(bound*model['input']+(token_limit+thinking)*model['output'])/1e6*1.1
        context['request_image_count']=images
        if images and (OUT/'image_incompatible.json').exists():
            context['image_incompatible']=True
            raise RuntimeError('Previously verified image input incompatibility; paid call skipped')
        entries=ledger()
        if total_reserved(entries)+reserve>LIMIT:
            context['budget_stop']=True
            raise BudgetStop('New round 7 CNY budget would be exceeded')
        entry={'case':context['case']['id'],'model':model['id'],'label':model['label'],'job_id':context['job_id'],
               'status':'pending','reserved_cny':reserve,'accounted_cny':reserve,'input_bound':bound,
               'image_count':images,'max_tokens':token_limit,'thinking_budget':thinking,
               'payload_sha256':hashlib.sha256(json.dumps(request,ensure_ascii=False).encode('utf-8')).hexdigest()}
        entries.append(entry);save(OUT/'ledger.json',entries)
        context['provider_called']=True
        try:
            result=original_attempt(request,base,key,workspace,timeout,secrets,*args)
        except Exception as exc:
            entry.update({'status':'exception_reserved','error_type':type(exc).__name__})
            save(OUT/'ledger.json',entries)
            raise
        usage=result.get('usage') or {}
        inp,out=usage.get('input_tokens'),usage.get('output_tokens')
        entry.update({'status':'returned','http_status':result.get('http_status'),'usage':usage,
                      'finish_reason':result.get('finish_reason'),'transport_success':result.get('transport_success'),
                      'error_code':result.get('provider_error_code'),'elapsed_ms':result.get('attempt_wall_ms')})
        if isinstance(inp,int) and isinstance(out,int):
            cost=(inp*model['input']+out*model['output'])/1e6
            entry['estimated_cny']=cost;entry['accounted_cny']=cost*1.1
            if inp>bound or out>token_limit+thinking:context['budget_anomaly']=True
        elif result.get('http_status') in (400,401,402,403,404,422,429):entry['accounted_cny']=0
        save(OUT/'ledger.json',entries)
        save(context['job_dir']/'audit'/'comparison_provider.json',result)
        if result.get('http_status') in (400,422) and images:
            error_text=json.dumps(result,ensure_ascii=False).lower()
            if any(x in error_text for x in ['image','vision','multimodal']):
                context['image_incompatible']=True
                save(OUT/'image_incompatible.json',{'job_id':context['job_id'],'case':context['case']['id'],'http_status':result.get('http_status'),'audit_path':str(context['job_dir']/'audit'/'comparison_provider.json')})
            else:context['provider_blocked']=True
        elif result.get('http_status') in (400,401,402,403,404,422):context['provider_blocked']=True
        return result

    class TimedPipeline(pl.SearchMachinePipeline):
        def _write_state(self,job_dir,state,status):
            super()._write_state(job_dir,state,status)
            if not context.get('start'):return
            now=round(time.perf_counter()-context['start'],4)
            for item in state.get('images',[]):
                if not isinstance(item,dict):continue
                name=Path(str(item.get('path') or item.get('filename') or item.get('name') or '')).name
                if name in NAMES and (job_dir/name).exists():context['first_public'].setdefault(name,now)
            context['last_status']=status

    pipeline=TimedPipeline(OUT/'jobs'/subject,timeout_seconds=90,max_tokens=4096,
                           max_provider_repair_rounds=0,post_lower_budget_seconds=15,api_attempt=attempt)
    if dry:
        print(json.dumps({'subject':subject,'pipeline_ready':True,'ocr_ready':True,'cases':len(cases),'prior_and_new_budget_reserved':total_reserved(ledger())}),flush=True);return 0
    disabled=set()
    for i,case in enumerate(cases):
        for model in MODELS:
            identifier=case['id']+'__'+model['label']
            if identifier in results or model['label'] in disabled:continue
            context.clear();context.update({'case':case,'model':model,'first_public':{},'provider_called':False})
            selected_creds=creds[model['provider']]
            pl.load_credentials=lambda: (selected_creds[1],selected_creds[0],selected_creds[2])
            pipeline.model_id=pipeline.complex_model_id=model['id']
            image=Path(case['original'])
            start=time.perf_counter();context['start']=start
            state=pipeline.new_job(image.read_bytes(),image.name)
            job_id=state['job_id']; job_dir=pipeline.job_path(job_id)
            context.update({'job_id':job_id,'job_dir':job_dir})
            print(json.dumps({'event':'start','case':case['id'],'model':model['label'],'job_id':job_id}),flush=True)
            try:
                final=pipeline.solve(job_id,use_model=True)
            except Exception as exc:
                final={'status':'runner_error','error':{'type':type(exc).__name__,'message':str(exc)[:250]}}
            elapsed=round(time.perf_counter()-start,4)
            pngs=[]
            for name in NAMES:
                file=job_dir/name
                if file.exists():
                    try:
                        meta=pl.inspect_device_png(file)
                        pngs.append({'name':name,'metadata':meta,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
                    except Exception as exc:pngs.append({'name':name,'error':type(exc).__name__})
            contract=final.get('provider_contract') or {}
            row={'case':case['id'],'subject':subject,'model':model['id'],'label':model['label'],
                 'job_id':job_id,'job_dir':str(job_dir),'status':final.get('status'),'elapsed_seconds':elapsed,
                 'first_public_seconds':context['first_public'],'pngs':pngs,'provider_called':context['provider_called'],
                 'image_contract':final.get('image_contract'),'validation':final.get('validation'),
                 'provider_contract':contract,'preprocess':final.get('preprocess'),'streaming':final.get('streaming'),
                 'usage':final.get('usage'),'error':final.get('error'),'http_status':final.get('http_status'),
                 'model_response':final.get('model_id_response'),'budget_stop':context.get('budget_stop',False),
                 'image_incompatible':context.get('image_incompatible',False),'request_image_count':context.get('request_image_count')}
            results[identifier]=row;save(state_path,results)
            print(json.dumps({'event':'end','case':case['id'],'model':model['label'],'status':row['status'],
                              'png_count':len(pngs),'public_count':len(context['first_public']),
                              'elapsed_seconds':elapsed,'fallback':contract.get('time_budget_fallback'),
                              'partial':contract.get('partial_skeleton_reused'),'budget_total':round(total_reserved(ledger()),4)},ensure_ascii=False),flush=True)
            if context.get('budget_stop') or context.get('budget_anomaly'):return 2
            if context.get('provider_blocked'):return 3
    return 0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--subject',choices=['math','physics']);ap.add_argument('--limit',type=int,default=10);ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.subject:return worker(args.subject,args.limit,args.dry_run)
    for subject in ['math','physics']:
        cmd=[sys.executable,str(Path(__file__).resolve()),'--subject',subject,'--limit',str(args.limit)]
        if args.dry_run:cmd.append('--dry-run')
        rc=subprocess.run(cmd).returncode
        if rc:return rc
    return 0

if __name__=='__main__':raise SystemExit(main())
