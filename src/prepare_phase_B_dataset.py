#!/usr/bin/env python3
"""Build the validated single-upload Phase B package; never fabricate predictions."""
from __future__ import annotations
import argparse,csv,json,math,shutil,sys,zipfile
from pathlib import Path

LABELS={"left":0,"forward":1,"right":2}
# Also exclude target-validity/label-rule proxies: using them would reconstruct the rule.
LEAK={"normalized_path_offset","normalized_path_offset_missing","path_center_x","path_center_x_missing","absolute_path_deviation","absolute_path_deviation_missing","path_left_boundary","path_right_boundary","path_left_boundary_missing","path_right_boundary_missing","path_width_lower","path_width_lower_missing","path_area_ratio","path_area_ratio_missing","path_continuity","path_continuity_missing","missing_path","missing_path_missing"}
def read(p):
 with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def write(p,rows,fields):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open("w",encoding="utf-8",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def num(v):
 try:return math.isfinite(float(v))
 except (ValueError,TypeError):return False
def cfg(p):
 return {a.strip():b.strip() for x in p.read_text().splitlines() if ":" in x and not x.lstrip().startswith("#") for a,b in [x.split(":",1)]}
def rebuild_targets(features,c):
 """Committed Phase-A geometry rules, dependency-free; target provenance is retained."""
 c=cfg(c);L=float(c["left_threshold"]);R=float(c["right_threshold"]);A=float(c["minimum_path_area_ratio"]);W=float(c["minimum_valid_path_width"]);C=float(c["minimum_path_continuity"]);S=float(c["stop_uncertain_threshold"]);out=[]
 for r in features:
  a=float(r["path_area_ratio"]) if num(r.get("path_area_ratio")) else float("nan");w=float(r["path_width_lower"]) if num(r.get("path_width_lower")) else float("nan");q=float(r["path_continuity"]) if num(r.get("path_continuity")) else float("nan");o=float(r["normalized_path_offset"]) if num(r.get("normalized_path_offset")) else float("nan")
  flags=[]
  if not num(a) or a<A:flags.append("low_area")
  if not num(w) or w<W:flags.append("narrow_path")
  if not num(q) or q<C:flags.append("discontinuous_path")
  if not num(o):flags.append("missing_path_center")
  quality=min(max(0,min(1,a/A)) if num(a) else 0,max(0,min(1,w/W)) if num(w) else 0,q if num(q) else 0)
  if quality<S:flags.append("low_quality")
  label="stop_or_uncertain" if flags else ("left" if o<L else "right" if o>R else "forward")
  out.append({"image_id":r["image_id"],"split":r["split"],"continuous_target":r.get("normalized_path_offset",""),"normalized_offset":r.get("normalized_path_offset",""),"discrete_target":label,"target_confidence":quality,"warning_flags":";".join(flags),"target_status":"provisional_geometry_derived_rebuilt_from_committed_features"})
 return out
def main():
 a=argparse.ArgumentParser();a.add_argument("--project-root",type=Path,default=Path.cwd());a.add_argument("--images-root",type=Path,required=True);a.add_argument("--output-dir",type=Path,required=True);x=a.parse_args();root=x.project_root.resolve();src=x.images_root.resolve();out=x.output_dir.resolve()
 fp=root/"data/processed/features_ground_truth.csv";mp=root/"data/processed/image_manifest.csv";cp=root/"configs/navigation_targets.yaml";sp=root/"configs/phaseB_feature_selection.yaml"
 for p in [fp,mp,cp,sp,root/"configs/classes.yaml"]:
  if not p.is_file():raise FileNotFoundError(f"Required input missing: {p}")
 if not (src/"images").is_dir():raise FileNotFoundError(f"--images-root must contain images/: {src}")
 f,m=read(fp),read(mp);tp=root/"data/processed/navigation_targets.csv";t=read(tp) if tp.exists() else rebuild_targets(f,cp);origin="executed_phase_a" if tp.exists() else "rebuilt_from_committed_ground_truth_features"
 ids=[r["image_id"] for r in m]
 if len(ids)!=len(set(ids)):raise ValueError("Duplicate image_id in manifest")
 by={r["image_id"]:r for r in t}
 if len(by)!=len(t) or set(by)-set(ids):raise ValueError("Target IDs are not unique/aligned")
 ds=out/"phase_B_training_dataset"
 if ds.exists():shutil.rmtree(ds)
 for d in ["images/train","images/valid","images/test","tabular","configs","metadata"]:(ds/d).mkdir(parents=True)
 shutil.copy2(fp,ds/"tabular/features_ground_truth.csv")
 pred=next(root.rglob("features_predicted_masks.csv"),None)
 if pred:shutil.copy2(pred,ds/"tabular/features_predicted_masks.csv")
 for p in [root/"configs/classes.yaml",cp,sp]:shutil.copy2(p,ds/"configs"/p.name)
 manifest=[];reg=[];cl=[];missing=[]
 for r in m:
  split="valid" if r["split"]=="validation" else r["split"];p=src/"images"/split/Path(r["image_path"]).name
  if not p.is_file():missing.append(str(p));continue
  dest=ds/"images"/split/p.name;shutil.copy2(p,dest);q=by.get(r["image_id"],{});rv=q.get("continuous_target",q.get("normalized_offset",""));lab=q.get("discrete_target","").lower().strip();row={"image_id":r["image_id"],"relative_image_path":dest.relative_to(ds).as_posix(),"split":split,"image_width":r.get("image_width",f[0].get("image_width","")),"image_height":r.get("image_height",f[0].get("image_height","")),"has_valid_regression_target":num(rv),"has_valid_classification_target":lab in LABELS};manifest.append(row)
  if num(rv):reg.append({**{k:row[k] for k in ["image_id","relative_image_path","split"]},"normalized_offset":rv})
  if lab in LABELS:cl.append({**{k:row[k] for k in ["image_id","relative_image_path","split"]},"direction_label":lab,"direction_id":LABELS[lab]})
 if missing:shutil.rmtree(ds);raise FileNotFoundError(f"{len(missing)} images absent; first: {missing[0]}")
 fields=list(f[0]);approved=[z for z in fields if z not in LEAK and z not in {"image_id","split","image_width","image_height"} and not z.endswith("_missing")]
 write(ds/"metadata/image_manifest.csv",manifest,list(manifest[0]));write(ds/"tabular/regression_targets.csv",reg,["image_id","relative_image_path","split","normalized_offset"]);write(ds/"tabular/classification_targets.csv",cl,["image_id","relative_image_path","split","direction_label","direction_id"]);write(ds/"tabular/navigation_targets_original.csv",t,list(t[0]));write(ds/"tabular/feature_dictionary.csv",[{"feature":z,"source":"ground_truth_masks","eligible_for_regression":str(z in approved).lower(),"notes":"Excluded direct target or label-rule proxy" if z in LEAK else "Header-derived metadata"} for z in fields],["feature","source","eligible_for_regression","notes"]);write(ds/"tabular/feature_selection.csv",[{"feature":z,"status":"approved" if z in approved else "excluded","reason":"Non-leaking candidate" if z in approved else "Target/validity-rule leakage or metadata"} for z in fields],["feature","status","reason"])
 summary=[{"split":s,"images":sum(z["split"]==s for z in manifest),"valid_regression":sum(z["split"]==s for z in reg),"valid_classification":sum(z["split"]==s for z in cl)} for s in ["train","valid","test"]];write(ds/"metadata/split_summary.csv",summary,list(summary[0]));write(ds/"metadata/target_distribution.csv",[{"split":s,"direction_label":l,"count":sum(z["split"]==s and z["direction_label"]==l for z in cl)} for s in ["train","valid","test"] for l in LABELS],["split","direction_label","count"])
 validation={"status":"passed","total_images":len(manifest),"split_summary":summary,"approved_feature_count":len(approved),"target_origin":origin,"predicted_mask_features_included":bool(pred),"predicted_mask_note":"Absent: no executed Phase A predicted-mask table was available; no substitute was created." if not pred else "Exploratory if train rows are in-sample.","leakage_exclusions":sorted(LEAK),"checks":{"unique_ids":True,"images_exist":True,"splits_preserved":True}}
 (ds/"metadata/data_validation.json").write_text(json.dumps(validation,indent=2));(ds/"dataset_manifest.json").write_text(json.dumps({"format_version":1,**validation},indent=2));(ds/"README.md").write_text("# Phase B dataset\n\nTargets are provisional image-geometry research labels, not robot commands. Ground-truth-mask features are oracle-only. Predicted-mask features are omitted when no executed source exists.\n");(ds/"data_quality_report.md").write_text("# Data quality report\n\nAll manifest images were copied with their original split. Direct target and target-validity-rule features are excluded; missing values are retained for train-only imputation.\n")
 zp=out/"cassava_navigation_phase_B_dataset.zip"
 with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
  for p in ds.rglob("*"):
   if p.is_file():z.write(p,p.relative_to(out))
 with zipfile.ZipFile(zp) as z:
  if "phase_B_training_dataset/metadata/data_validation.json" not in z.namelist():raise RuntimeError("ZIP verification failed")
 print(json.dumps({"zip":str(zp),"size_bytes":zp.stat().st_size,**validation},indent=2))
if __name__=="__main__":
 try:main()
 except Exception as e:print(f"ERROR: {e}",file=sys.stderr);raise SystemExit(2)
