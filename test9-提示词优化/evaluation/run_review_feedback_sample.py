import argparse,json,time,urllib.request,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parents[1]/"datasets/testzip_math_physics_300_v1/Test/mathematics"
OUT=ROOT/"evaluation/review_feedback_20261105/retest"
OUT.mkdir(parents=True,exist_ok=True)
STATE=OUT/"progress.json"
def save(d):
    temp=STATE.with_suffix(".tmp");temp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8");temp.replace(STATE)
def request(url,data=None,headers=None):
    req=urllib.request.Request("http://127.0.0.1:9130"+url,data=data,headers=headers or {})
    with urllib.request.urlopen(req,timeout=45) as r:return json.load(r)
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--limit",type=int,default=150);args=parser.parse_args()
    progress=json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"dataset":str(DATA),"curriculum_term":"g12_s2","model_policy":"use_current_project_model","cases":{}}
    files=sorted(DATA.glob("Test*/*.png"),key=lambda p:tuple(map(int,p.stem.split("-"))))
    assert len(files)==150
    selected = {"6-5", "6-8", "6-15", "7-2", "7-4", "7-15", "8-4", "8-6"}
    for image in [p for p in files if p.stem in selected][:args.limit]:
        key=image.stem;entry=progress["cases"].get(key,{})
        if entry.get("status") in ("completed","failed"):continue
        if not entry.get("job_id"):
            boundary="----"+uuid.uuid4().hex
            body=(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{image.name}"\r\nContent-Type: image/png\r\n\r\n').encode()+image.read_bytes()+f'\r\n--{boundary}--\r\n'.encode()
            result=request("/api/v1/math/jobs",body,{"Content-Type":"multipart/form-data; boundary="+boundary,"X-Curriculum-Term":"g12_s2","Idempotency-Key":"review-feedback-20261105-v1-"+key})
            entry={"job_id":result["job_id"],"source":str(image),"status":"queued"};progress["cases"][key]=entry;save(progress)
        while True:
            result=request("/api/v1/math/jobs/"+entry["job_id"])
            entry["status"]=result["status"]
            job=ROOT/"math/device_runs"/entry["job_id"]/"job.json"
            if job.exists():
                audit=json.loads(job.read_text(encoding="utf-8"))
                entry["actual_model"]=audit.get("model_id_requested")
            save(progress)
            if result["status"] in ("completed","failed"):
                (OUT/(key+".json")).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
                print(key,result["status"],flush=True);break
            time.sleep(3)
if __name__=="__main__":main()
