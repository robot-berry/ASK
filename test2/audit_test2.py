from __future__ import annotations

"""Incremental, read-only audit for the running test2 batch.

The runner writes its manifest only at the end. This script can audit while the
batch is live by recovering job/photo names from Markdown artifacts. Cached
per-job JSON records make subsequent invocations process only new or changed
outputs. A same-name screenshot is never trusted solely by filename: its
geometry must pass SIFT/RANSAC and, for matched-electronic outputs, its pixels
must agree with the processor-reported reference.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_REFERENCES = Path(r"D:\robot\Documents\ASK\P_DATA\testmini_screenshot")
DEFAULT_SOURCES = Path(r"D:\robot\Documents\ASK\P_DATA\testmini_photo")
AUDIT_VERSION = 3
KNOWN_WRONG_SAME_NAME = {
    "photo_107_0242.jpg",
    "photo_117_0270.jpg",
    "photo_201_0773.jpg",
}

# Final visual adjudication.  These labels are deliberately separate from the
# objective first pass so an unreliable homography is never silently promoted
# to either a pass or a failure.
MANUAL_FINAL_REVIEW = {
    "photo_080_0176.jpg": {
        "same_question": True,
        "question_complete": True,
        "upright": True,
        "issues": ["blue-stain-binarized-as-black-interference"],
        "severity": "major-cleanup-defect",
    },
    "photo_141_0354.jpg": {
        "same_question": True,
        "question_complete": True,
        "upright": True,
        "issues": ["dense-background-speckles", "page-edge-residue"],
        "severity": "major-cleanup-defect",
    },
    "photo_191_0706.jpg": {
        "same_question": True,
        "question_complete": True,
        "upright": False,
        "issues": ["output-rotated-sideways-90deg", "bottom-frame-residue"],
        "severity": "critical-direction-failure",
    },
    "photo_196_0761.jpg": {
        "same_question": True,
        "question_complete": True,
        "upright": True,
        "issues": [],
        "severity": "pass",
    },
    "photo_206_0819.jpg": {
        "same_question": True,
        "question_complete": True,
        "upright": True,
        "issues": [],
        "severity": "pass",
    },
    "photo_241_1284.jpg": {
        "same_question": True,
        "question_complete": True,
        "upright": False,
        "issues": ["approximately-8deg-residual-skew", "top-left-frame-residue"],
        "severity": "critical-direction-failure",
    },
}


def read(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    return image


def photo_from_markdown(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"photo_\d+_\d+\.(?:jpe?g|png|webp)", text, re.I)
    return match.group(0) if match else ""


def current_jobs(results: Path) -> tuple[list[dict[str, Any]], str]:
    manifest = results / "results.csv"
    if manifest.exists():
        rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
        return rows, "results.csv"
    rows=[]
    for markdown in sorted((results / "artifacts/markdown").glob("*.md")):
        name=photo_from_markdown(markdown)
        job=markdown.stem
        output=results / f"artifacts/png/{job}.png"
        if name and output.exists():
            rows.append({"name":name,"job_id":job,"output_png":f"png/{job}.png","ok":"True"})
    return rows,"live-markdown"


def merge_manifest_metadata(rows: list[dict[str, Any]], results: Path) -> list[dict[str, Any]]:
    """Overlay final runner metadata without invalidating pixel-metric cache."""
    manifest = results / "results.csv"
    if not manifest.exists():
        return rows
    final_rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
    by_job = {str(row.get("job_id", "")): row for row in final_rows}
    merged=[]
    for record in rows:
        row=by_job.get(str(record.get("job_id", "")), {})
        record=dict(record)
        for key in (
            "cleanup_method", "reference_match", "reference_confidence", "rotation",
            "rotation_confidence", "rotation_method", "deskew_detected_angle",
            "deskew_applied_angle", "deskew_status", "perspective_status",
            "perspective_confidence", "fallback_count", "question_count", "ok", "error",
        ):
            if key in row:
                record[f"reported_{key}"]=row[key]
        merged.append(record)
    return merged


def artifact_path(results: Path, relative: str) -> Path:
    path=Path(relative)
    if path.is_absolute(): return path
    if (results/path).exists(): return results/path
    return results/"artifacts"/path


def card_paths(results:Path,job:str)->list[Path]:
    return sorted((results/f"artifacts/questions/{job}").glob("q*.png"))


def fingerprint(paths:list[Path])->str:
    h=hashlib.sha1()
    for path in paths:
        stat=path.stat(); h.update(str(path).encode()); h.update(str(stat.st_size).encode()); h.update(str(stat.st_mtime_ns).encode())
    return h.hexdigest()


def scale_gray(image:np.ndarray,longest:int=1800)->tuple[np.ndarray,float]:
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    scale=min(1.0,longest/max(gray.shape))
    if scale<1: gray=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    return gray,scale


def align(
    reference: np.ndarray,
    output: np.ndarray,
    *,
    min_inliers: int = 20,
    min_ratio: float = .45,
    max_error_px: float = 3.0,
    max_direction_deg: float = 12.0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    ref,rs=scale_gray(reference); out,os=scale_gray(output)
    sift=cv2.SIFT_create(nfeatures=6000,contrastThreshold=.015,edgeThreshold=12)
    kr,dr=sift.detectAndCompute(ref,None); ko,do=sift.detectAndCompute(out,None)
    result={"alignment_reliable":False,"alignment_matches":0,"alignment_inliers":0,"alignment_ratio":0.0,"direction_residual_deg":None}
    if dr is None or do is None:
        result["alignment_reason"]="no-descriptors"; return None,result
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    forward=matcher.knnMatch(dr,do,k=2); reverse=matcher.knnMatch(do,dr,k=2)
    fg={m.queryIdx:m.trainIdx for m,n in forward if m.distance<.76*n.distance}
    rg={m.queryIdx:m.trainIdx for m,n in reverse if m.distance<.76*n.distance}
    pairs=[(q,t) for q,t in fg.items() if rg.get(t)==q]
    if len(pairs)<8: pairs=list(fg.items())
    result["alignment_matches"]=len(pairs)
    if len(pairs)<6:
        result["alignment_reason"]="too-few-matches"; return None,result
    src=np.float32([kr[q].pt for q,_ in pairs])/rs; dst=np.float32([ko[t].pt for _,t in pairs])/os
    H,mask=cv2.findHomography(src,dst,cv2.RANSAC,5)
    if H is None or mask is None:
        result["alignment_reason"]="homography-failed"; return None,result
    inside=mask.ravel().astype(bool); count=int(inside.sum()); ratio=float(inside.mean())
    projected=cv2.perspectiveTransform(src[inside,None,:],H)[:,0]
    error=float(np.median(np.linalg.norm(projected-dst[inside],axis=1))) if count else 999
    center=np.array([reference.shape[1]/2,reference.shape[0]/2],np.float32); span=max(100,reference.shape[1]*.25)
    probe=np.array([[center,center+(span,0)]],np.float32); mapped=cv2.perspectiveTransform(probe,H)[0]; v=mapped[1]-mapped[0]
    angle=((math.degrees(math.atan2(float(v[1]),float(v[0])))+45)%90)-45
    reliable=(
        count >= min_inliers
        and ratio >= min_ratio
        and error <= max_error_px
        and abs(angle) <= max_direction_deg
    )
    result.update(alignment_inliers=count,alignment_ratio=ratio,alignment_error_px=error,direction_residual_deg=angle,alignment_reliable=reliable,alignment_reason="reliable" if reliable else "weak-geometry")
    return (H if reliable else None),result


def content_mask(gray:np.ndarray)->np.ndarray:
    paper=float(np.percentile(gray,90)); threshold=min(247,max(180,paper-8))
    return gray.astype(np.float32)<threshold


def reference_metrics(reference:np.ndarray,output:np.ndarray,H:np.ndarray)->dict[str,float]:
    rg=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY); og=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY); h,w=og.shape
    ri=content_mask(rg); oi=content_mask(og)
    exact=float(np.mean(np.abs(reference.astype(np.float32)-output.astype(np.float32)))) if reference.shape==output.shape else None
    if exact is not None and exact<=.5:
        return {"content_retention":1.0,"reference_content_coverage":1.0,"background_false_ink_rate":0.0,"reference_exact_mae":exact}
    valid=cv2.warpPerspective(np.full(rg.shape,255,np.uint8),H,(w,h),flags=cv2.INTER_NEAREST)>0
    warped=cv2.warpPerspective(ri.astype(np.uint8)*255,H,(w,h),flags=cv2.INTER_NEAREST)>0
    tolerance=max(2,int(round(min(h,w)*.004))); kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tolerance+1,)*2)
    neighborhood=cv2.dilate(warped.astype(np.uint8),kernel)>0; background=valid&~neighborhood; false=oi&background
    yy,xx=np.nonzero(ri)
    if len(xx)>120000:
        idx=np.linspace(0,len(xx)-1,120000,dtype=np.int64); xx,yy=xx[idx],yy[idx]
    mapped=cv2.perspectiveTransform(np.column_stack((xx,yy)).astype(np.float32)[None],H)[0]
    inside=(mapped[:,0]>=0)&(mapped[:,0]<w)&(mapped[:,1]>=0)&(mapped[:,1]<h)
    distance=cv2.distanceTransform((~oi).astype(np.uint8),cv2.DIST_L2,3); retained=np.zeros(len(mapped),bool); idx=np.flatnonzero(inside)
    mx=np.clip(np.rint(mapped[idx,0]).astype(int),0,w-1); my=np.clip(np.rint(mapped[idx,1]).astype(int),0,h-1); retained[idx]=distance[my,mx]<=tolerance
    return {"content_retention":float(retained.mean()),"reference_content_coverage":float(inside.mean()),"background_false_ink_rate":float(false.sum()/max(int(background.sum()),1)),"reference_exact_mae":exact}


def self_direction(image:np.ndarray)->dict[str,Any]:
    gray,_=scale_gray(image,1600); ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]; edges=cv2.Canny(ink,50,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/360,50,minLineLength=max(50,int(gray.shape[1]*.12)),maxLineGap=20)
    candidates=[]
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            angle=((math.degrees(math.atan2(y2-y1,x2-x1))+45)%90)-45; length=math.hypot(x2-x1,y2-y1)
            # Exclude extreme border bands and extraordinarily long page edges.
            cy=(y1+y2)/2; cx=(x1+x2)/2
            if abs(angle)<=12 and length<gray.shape[1]*.65 and gray.shape[0]*.06<cy<gray.shape[0]*.94 and gray.shape[1]*.04<cx<gray.shape[1]*.96:
                candidates.append((angle,length))
    if len(candidates)<3:return {"self_direction_deg":None,"self_direction_status":"insufficient","self_direction_confidence":0.0}
    ordered=sorted(candidates); total=sum(w for _,w in ordered); cum=0; median=ordered[-1][0]
    for angle,weight in ordered:
        cum+=weight
        if cum>=total/2:median=angle;break
    confidence=sum(w for a,w in candidates if abs(a-median)<=1.25)/max(total,1)
    return {"self_direction_deg":median,"self_direction_status":"reliable" if confidence>=.55 else "low-consensus","self_direction_confidence":confidence}


def card_metrics(output:np.ndarray,cards:list[np.ndarray])->dict[str,Any]:
    if not cards:return {"card_count":0,"card_ink_coverage":None,"card_pixel_identical":False}
    og=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY); oi=content_mask(og); card_ink=0
    for card in cards: card_ink+=int(content_mask(cv2.cvtColor(card,cv2.COLOR_BGR2GRAY)).sum())
    identical=len(cards)==1 and cards[0].shape==output.shape and int(np.max(np.abs(cards[0].astype(np.int16)-output.astype(np.int16))))==0
    return {"card_count":len(cards),"card_ink_coverage":float(min(1,card_ink/max(int(oi.sum()),1))),"card_pixel_identical":identical}


def prefixed(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def card_reference_metrics(
    reference: np.ndarray,
    output: np.ndarray,
    cards: list[np.ndarray],
    output_record: dict[str, Any],
) -> dict[str, Any]:
    """Measure how much reference question content survives in the best card.

    Ink(card)/ink(output) is useful for spotting a crop, but it cannot tell a
    harmless margin crop from a lost equation.  This second metric maps the
    clean reference directly into each extracted card.
    """
    if not cards:
        return {
            "card_reference_alignment_reliable": False,
            "card_reference_content_coverage": None,
            "card_content_retention": None,
        }
    if (
        len(cards) == 1
        and output_record.get("card_pixel_identical")
        and output_record.get("alignment_reliable")
    ):
        return {
            "card_reference_alignment_reliable": True,
            "card_reference_content_coverage": output_record.get("reference_content_coverage"),
            "card_content_retention": output_record.get("content_retention"),
            "card_reference_alignment_inliers": output_record.get("alignment_inliers"),
        }
    candidates: list[dict[str, Any]] = []
    for index, card in enumerate(cards, start=1):
        H, geometry = align(reference, card)
        candidate: dict[str, Any] = {"card_index": index, **geometry}
        if H is not None:
            candidate.update(reference_metrics(reference, card, H))
        candidates.append(candidate)
    reliable = [candidate for candidate in candidates if candidate.get("alignment_reliable")]
    if not reliable:
        best = max(candidates, key=lambda item: int(item.get("alignment_inliers", 0)))
        return {
            "card_reference_alignment_reliable": False,
            "card_reference_content_coverage": None,
            "card_content_retention": None,
            "card_reference_alignment_inliers": best.get("alignment_inliers", 0),
        }
    best = max(
        reliable,
        key=lambda item: (
            float(item.get("reference_content_coverage", 0.0)),
            float(item.get("content_retention", 0.0)),
            int(item.get("alignment_inliers", 0)),
        ),
    )
    return {
        "card_reference_alignment_reliable": True,
        "card_reference_content_coverage": best.get("reference_content_coverage"),
        "card_content_retention": best.get("content_retention"),
        "card_reference_alignment_inliers": best.get("alignment_inliers"),
        "card_reference_best_index": best.get("card_index"),
    }


def audit(row:dict[str,Any],results:Path,references:Path,sources:Path)->dict[str,Any]:
    name=str(row.get("name","")); job=str(row.get("job_id","")); outpath=artifact_path(results,str(row.get("output_png") or f"png/{job}.png")); paths=[outpath]+card_paths(results,job)
    record={"audit_version":AUDIT_VERSION,"name":name,"job_id":job,"fingerprint":fingerprint(paths),"output_path":str(outpath),"cleanup_method":row.get("cleanup_method") or "unknown","reported_reference":row.get("reference_match") or ""}
    output=read(outpath); cards=[read(p) for p in paths[1:]]; record.update(self_direction(output)); record.update(card_metrics(output,cards))
    reference_name=str(row.get("reference_match") or "").strip()
    if reference_name:
        refpath=references/Path(reference_name).name; source_kind="reported-reference"
    elif name not in KNOWN_WRONG_SAME_NAME:
        refpath=references/(Path(name).stem.replace("photo_","screenshot_")+".png"); source_kind="same-name"
    else: refpath=None; source_kind="known-wrong-same-name"
    record["reference_source"]=source_kind; record["reference_path"]=str(refpath) if refpath else ""
    if refpath and refpath.exists():
        reference=read(refpath); H,metrics=align(reference,output); record.update(metrics)
        if H is not None:record.update(reference_metrics(reference,output,H))
        record.update(card_reference_metrics(reference, output, cards, record))
        # This only confirms that the processor emitted the reference it
        # reported; it does not yet prove that reference is the same question.
        exact=record.get("reference_exact_mae")
        record["output_matches_reported_reference"]=bool(
            (exact is not None and exact <= .5)
            or (record.get("alignment_inliers", 0) >= 35 and record.get("alignment_ratio", 0) >= .60)
        )

        # Independently compare the raw source photo with the chosen reference.
        # Perspective/rotation are permitted here; coherent local geometry is
        # the evidence that no unrelated clean question was substituted.
        source_path = sources / name
        record["source_path"] = str(source_path)
        if source_path.exists():
            source = read(source_path)
            _, source_geometry = align(
                reference,
                source,
                min_inliers=14,
                min_ratio=.25,
                max_error_px=5.0,
                max_direction_deg=45.0,
            )
            record.update(prefixed(source_geometry, "source_reference_"))
            if source_geometry.get("alignment_reliable"):
                record["reference_identity_status"] = "confirmed-source-geometry"
                record["reference_identity_safe"] = True
            else:
                record["reference_identity_status"] = "unverified-source-geometry"
                record["reference_identity_safe"] = None
        else:
            record["reference_identity_status"] = "source-missing"
            record["reference_identity_safe"] = None
    else:
        record.update(alignment_reliable=False,alignment_reason=source_kind,direction_residual_deg=None,content_retention=None,reference_content_coverage=None,background_false_ink_rate=None,reference_identity_safe=None,reference_identity_status=source_kind,output_matches_reported_reference=None,card_reference_alignment_reliable=False,card_reference_content_coverage=None,card_content_retention=None)
    return record


def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--results",type=Path,default=DEFAULT_RESULTS); parser.add_argument("--references",type=Path,default=DEFAULT_REFERENCES); parser.add_argument("--sources",type=Path,default=DEFAULT_SOURCES); parser.add_argument("--out",type=Path); parser.add_argument("--refresh",action="store_true")
    args=parser.parse_args(); out=args.out or args.results/"partial_audit"; cache=out/"cache"; cache.mkdir(parents=True,exist_ok=True)
    rows,manifest_source=current_jobs(args.results); records=[]; new=0; reused=0
    for row in rows:
        job=str(row.get("job_id","")); output=artifact_path(args.results,str(row.get("output_png") or f"png/{job}.png")); paths=[output]+card_paths(args.results,job)
        if not output.exists():continue
        fp=fingerprint(paths); cachepath=cache/f"{job}.json"
        if cachepath.exists() and not args.refresh:
            saved=json.loads(cachepath.read_text(encoding="utf-8"))
            if saved.get("fingerprint")==fp and saved.get("audit_version")==AUDIT_VERSION:record=saved; reused+=1
            else:record=audit(row,args.results,args.references,args.sources);cachepath.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8");new+=1
        else:record=audit(row,args.results,args.references,args.sources);cachepath.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8");new+=1
        records.append(record); print(f"{record['name']}: {record.get('alignment_reason')} identity={record.get('reference_identity_safe')} card={record.get('card_ink_coverage')}")
    records=merge_manifest_metadata(records,args.results)
    fields=sorted({k for r in records for k in r});
    with (out/"partial_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    reliable=[r for r in records if r.get("alignment_reliable")]; identity_confirmed=[r for r in records if r.get("reference_identity_safe") is True]; identity_unverified=[r for r in records if r.get("reference_identity_status")=="unverified-source-geometry"]; cards=[float(r["card_ink_coverage"]) for r in records if r.get("card_ink_coverage") is not None]; card_ref=[float(r["card_reference_content_coverage"]) for r in records if r.get("card_reference_content_coverage") is not None]; directions=[abs(float(r["direction_residual_deg"])) for r in reliable]
    # A weak match is reported as unverified, never automatically as a wrong
    # replacement.  Confirmed wrong substitutions require visual review (or a
    # strong competing-reference match) and are therefore intentionally empty
    # in this objective first pass.
    confirmed_wrong=[r["name"] for r in records if r.get("reference_identity_status")=="confirmed-wrong-reference"]
    summary={"manifest_source":manifest_source,"observed_jobs":len(records),"newly_audited":new,"cache_reused":reused,"cleanup_methods":dict(Counter(str(r.get("cleanup_method")) for r in records)),"reliable_reference_pairs":len(reliable),"weak_or_excluded_pairs":len(records)-len(reliable),"confirmed_wrong_reference_replacements":confirmed_wrong,"unverified_reference_identities":[r["name"] for r in identity_unverified],"mean_abs_direction_deg":float(np.mean(directions)) if directions else None,"p95_abs_direction_deg":float(np.percentile(directions,95)) if directions else None,"over_2deg_count":sum(v>2 for v in directions),"mean_content_retention":float(np.mean([r["content_retention"] for r in reliable])) if reliable else None,"minimum_content_retention":min((r["content_retention"] for r in reliable),default=None),"mean_reference_content_coverage":float(np.mean([r["reference_content_coverage"] for r in reliable])) if reliable else None,"minimum_reference_content_coverage":min((r["reference_content_coverage"] for r in reliable),default=None),"mean_background_false_ink_rate":float(np.mean([r["background_false_ink_rate"] for r in reliable])) if reliable else None,"minimum_card_ink_coverage":min(cards,default=None),"cards_below_95pct":[r["name"] for r in records if r.get("card_ink_coverage") is not None and float(r["card_ink_coverage"])<.95],"mean_card_reference_content_coverage":float(np.mean(card_ref)) if card_ref else None,"minimum_card_reference_content_coverage":min(card_ref,default=None),"cards_below_95pct_reference_coverage":[r["name"] for r in records if r.get("card_reference_content_coverage") is not None and float(r["card_reference_content_coverage"])<.95],"reference_identity_confirmed":len(identity_confirmed),"reference_identity_unverified_count":len(identity_unverified)}

    if manifest_source == "results.csv":
        by_name = {r["name"]: r for r in records}
        manual = {
            name: review
            for name, review in MANUAL_FINAL_REVIEW.items()
            if name in by_name
        }
        summary["manual_final_review"] = manual
        summary["manual_direction_failures"] = [
            name for name, review in manual.items() if review.get("upright") is False
        ]
        summary["manual_incomplete_questions"] = [
            name for name, review in manual.items() if review.get("question_complete") is False
        ]
        summary["manual_wrong_reference_replacements"] = [
            name for name, review in manual.items() if review.get("same_question") is False
        ]
        summary["final_unverified_reference_identities"] = [
            name
            for name in summary["unverified_reference_identities"]
            if not manual.get(name, {}).get("same_question")
        ]
        summary["final_gate"] = "fail" if (
            summary["manual_direction_failures"]
            or summary["manual_incomplete_questions"]
            or summary["manual_wrong_reference_replacements"]
        ) else "pass"
    (out/"partial_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# test2 incremental partial audit","",f"Observed jobs: {len(records)}; newly audited: {new}; cache reused: {reused}.","",f"- Reliable reference pairs: {len(reliable)}",f"- Confirmed wrong reference replacements: {summary['confirmed_wrong_reference_replacements']}",f"- Unverified source/reference identities: {summary['unverified_reference_identities']}",f"- Mean/P95 absolute direction: {summary['mean_abs_direction_deg']} / {summary['p95_abs_direction_deg']} deg",f"- Mean/min content retention: {summary['mean_content_retention']} / {summary['minimum_content_retention']}",f"- Mean/min reference coverage: {summary['mean_reference_content_coverage']} / {summary['minimum_reference_content_coverage']}",f"- Mean false background ink: {summary['mean_background_false_ink_rate']}",f"- Minimum card ink coverage: {summary['minimum_card_ink_coverage']}",f"- Cards below 95% ink ratio: {summary['cards_below_95pct']}",f"- Mean/min card reference coverage: {summary['mean_card_reference_content_coverage']} / {summary['minimum_card_reference_content_coverage']}",f"- Cards below 95% reference coverage: {summary['cards_below_95pct_reference_coverage']}"]
    if "final_gate" in summary:
        lines.extend(["", "## Final visual adjudication", "", f"- Gate: **{summary['final_gate'].upper()}**", f"- Direction failures: {summary['manual_direction_failures']}", f"- Incomplete questions: {summary['manual_incomplete_questions']}", f"- Wrong reference replacements: {summary['manual_wrong_reference_replacements']}", f"- Remaining unverified identities: {summary['final_unverified_reference_identities']}"])
    (out/"PARTIAL_AUDIT.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
