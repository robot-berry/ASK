from pathlib import Path
import json
from pypdf import PdfReader
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / 'datasets/testzip_math_physics_300_v1'
ROWS = ROOT / 'benchmark/ocr_annotation_driven_full300_round16_final_20260907/results.jsonl'
SELECTION = {'MATH': [1,2,3,5,7,9,10,12,13,15], 'PHYS': [1,3,5,7,8,9,10,11,14,15]}
cases = []
for line in ROWS.read_text(encoding='utf-8').splitlines():
    r = json.loads(line)
    code = 'MATH' if r['subject'] == 'mathematics' else 'PHYS'
    if r['source_group'] != 'Test1' or r['source_index'] not in SELECTION[code]:
        continue
    subject = 'math' if code == 'MATH' else 'physics'
    full = bool(r.get('illustration_paths') or r.get('full_question_visual_fallback'))
    evidence = Path(r['selected_source_path'])
    if not evidence.exists():
        raise FileNotFoundError(evidence)
    cases.append({'id':r['question_id'], 'subject':subject, 'index':r['source_index'],
                  'text':r['transcript'], 'original':str(DATA/r['image_path']),
                  'evidence':str(evidence), 'images':[str(evidence)] if full else [],
                  'illustration_count':len(r.get('illustration_paths',[])),
                  'input_policy':'cached OCR + full selected question evidence when illustrated; no answer PDF sent'})
(OUT/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
for subject in ('math','physics'):
    items = [c for c in cases if c['subject']==subject]
    canvas = Image.new('RGB',(1400,1800),'#eeeeee')
    draw=ImageDraw.Draw(canvas)
    for i,c in enumerate(items):
        pic=Image.open(c['evidence']).convert('RGB')
        pic.thumbnail((680,320))
        x=(i%2)*700; y=(i//2)*360
        draw.text((x+10,y+5),c['id'],fill='black')
        canvas.paste(pic,(x+10,y+30))
    canvas.save(OUT/f'{subject}_contact.png')
for folder in ('mathematics','physics'):
    pdf=next((DATA/'Test'/folder).glob('*-1.pdf'))
    reader=PdfReader(pdf)
    for n,page in enumerate(reader.pages,1):
        for j,img in enumerate(page.images):
            (OUT/f'gold_{folder}_{n}_{j}.{img.name.rsplit(".",1)[-1]}').write_bytes(img.data)
print(json.dumps({'case_count':len(cases),'images':sum(bool(c['images']) for c in cases),'gold_pages_extracted':True}))
