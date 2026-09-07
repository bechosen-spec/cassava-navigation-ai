"""GPU baseline with explicit unavailable status and post-training exports.
Run from project root: python src/segmentation/baseline.py [--train].
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib-cache'))
import numpy as np
import pandas as pd
import yaml
from features.feature_extraction import manifest,polygons,masks_from_polygons,run as features
OUT=ROOT/'results/segmentation_baseline'
FIG=ROOT/'figures/segmentation_baseline'
MODEL=ROOT/'models/segmentation_baseline'

def environment():
    try:version=importlib.metadata.version('ultralytics')
    except importlib.metadata.PackageNotFoundError:version=None
    device=None
    try:
        import torch
        if torch.cuda.is_available():device='0'
        elif torch.backends.mps.is_available():device='mps'
    except ImportError:pass
    return dict(python=platform.python_version(),machine=platform.machine(),ultralytics=version,device=device)

def prepare():
    for d in [OUT,FIG,MODEL]:d.mkdir(parents=True,exist_ok=True)
    mf=manifest(); stage=ROOT/'data/processed/segmentation_dataset'
    # Copy into a derived training view: Ultralytics label caches cannot touch raw data.
    for row in mf.to_dict('records'):
        for kind,key in [('images','image_path'),('labels','label_path')]:
            src=ROOT/row[key];dest=stage/kind/row['split']/src.name;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(src,dest)
    for split in ['train','valid','test']:
        (stage/f'{split}.txt').write_text('\n'.join('./images/'+split+'/'+Path(p).name for p in mf.loc[mf.split==split,'image_path'])+'\n')
    data=dict(path=str(stage),train='train.txt',val='valid.txt',test='test.txt',names={0:'path',1:'cassava_leaves',2:'ridge'})
    # Committed YAML is portable; runtime YAML resolves root for this machine.
    (ROOT/'configs/cassava_segmentation.yaml').write_text(yaml.safe_dump({**data,'path':'data/processed/segmentation_dataset'},sort_keys=False))
    (OUT/'runtime_dataset.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    env=environment();(OUT/'environment.json').write_text(json.dumps(env,indent=2))
    return env,mf

def unavailable(env):
    if (MODEL/'best.pt').exists():raise RuntimeError('Existing trained artifacts found: do not overwrite with unavailable status')
    pd.DataFrame([dict(status='not_trained',model='yolo11n-seg.pt',split=s,mask_precision=np.nan,mask_recall=np.nan,mask_map50=np.nan,mask_map50_95=np.nan,box_precision=np.nan,box_recall=np.nan,box_map50=np.nan,box_map50_95=np.nan,best_epoch=np.nan,inference_ms=np.nan,fps=np.nan,model_size_mb=np.nan,parameter_count=np.nan) for s in ['valid','test']]).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame([dict(status='not_trained',split=s,class_id=c,class_name=n,mask_precision=np.nan,mask_recall=np.nan,mask_map50=np.nan,mask_map50_95=np.nan) for s in ['valid','test'] for c,n in enumerate(['path','cassava_leaves','ridge'])]).to_csv(OUT/'per_class_metrics.csv',index=False)
    pd.DataFrame(columns=['image_id','split','inference_ms','wall_ms','mean_union_iou']).to_csv(OUT/'inference_results.csv',index=False)
    pd.DataFrame(columns=['image_id','split','mean_union_iou','reason']).to_csv(OUT/'failure_cases.csv',index=False)
    note='''# Segmentation baseline — training pending

Model actually trained: **No**. Planned architecture: YOLO11n-seg (nano); fallback YOLOv8n-seg for older installed Ultralytics. Local Intel x86_64 Python 3.13 has no compatible PyTorch wheel (pip dry-run: no matching distribution); no usable CUDA/MPS runtime was available. Hardware graphics alone does not establish supported training acceleration. Ultralytics is not installed locally. Colab notebook 03 installs a pinned compatible stack and re-detects version/device before training.

Best epoch, validation/test mask and box metrics, inference speed, FPS, size and parameter count are unavailable. metrics.csv and per_class_metrics.csv explicitly say not_trained; empty inference/failure CSVs are schemas, not results. No weights, predicted-mask features, loss/PR/F1 curves, performance charts, predictions or failure images are represented as generated. These are produced by baseline.py --train after a successful GPU run. The notebook is structurally and syntactically verified locally; remote GPU execution remains pending.

Protocol: fixed seed 42, 640px, 80 maximum epochs, AdamW lr=0.001, batch 8, patience 15. Validation-only early stopping/checkpoint choice; test evaluated after best checkpoint is frozen. Moderate HSV brightness/saturation, rotation +/-5 degrees, scale +/-15%, translation 5%. Vertical and horizontal flips disabled (no calibrated geometry justification), mosaic/mixup disabled. Optional blur/noise/contrast augmentation is not added; version-dependent hidden Albumentations is explicitly disabled. Deterministic settings improve reproducibility but do not promise identical results across hardware.

Training uses a copied derived dataset, preserving original files and split membership; 602 train/120 valid/80 test after checkpoint exclusion. All 216 suspicious polygons (198 small, 18 large) retained. Inference exports report batch-one latency including preprocessing/inference/postprocessing, separate wall time, and hardware context. Semantic union IoU is only a diagnostic, not instance mask AP. Good/failure rankings use test IoU descriptively, never to tune. Train predicted-mask features are in-sample and must not be treated as out-of-fold navigation inputs.

References: [Ultralytics segmentation](https://docs.ultralytics.com/tasks/segment/), [YOLO11](https://docs.ultralytics.com/models/yolo11/), [training](https://docs.ultralytics.com/modes/train/).
'''
    (ROOT/'reports/segmentation_baseline_report.md').write_text(note)
    (MODEL/'README.md').write_text('No trained weights yet. Run notebooks/03_segmentation_baseline_colab.ipynb on a GPU. Successful runs save best.pt here.\n')
    (FIG/'README.md').write_text('Training-dependent figures are pending a real GPU run. No placeholder curves or predictions are fabricated.\n')

def train(env,mf):
    if env['device'] is None: raise RuntimeError('No usable GPU. Use the Colab notebook.')
    from ultralytics import YOLO
    from packaging.version import Version
    import ultralytics.data.augment as aug
    # Ensure reproducibility independent of whether Colab happens to have albumentations.
    def no_albumentations(self,p=1.0,transforms=None):self.p=p;self.transform=None;self.contains_spatial=False
    aug.Albumentations.__init__=no_albumentations
    cfg=yaml.safe_load((ROOT/'configs/yolo_baseline.yaml').read_text())
    cfg['model']='yolo11n-seg.pt' if Version(env['ultralytics'])>=Version('8.3.0') else 'yolov8n-seg.pt'
    cfg['device']=env['device'];(OUT/'resolved_training_config.yaml').write_text(yaml.safe_dump(cfg))
    model=YOLO(cfg.pop('model'))
    best_epochs=[]
    def record_best(trainer):
        if trainer.fitness == trainer.best_fitness:
            best_epochs.append(trainer.epoch+1)
    model.add_callback('on_model_save',record_best)
    model.train(data=str(OUT/'runtime_dataset.yaml'),project=str(OUT),name='train',exist_ok=False,plots=True,**cfg)
    best=Path(model.trainer.best);shutil.copy2(best,MODEL/'best.pt');net=YOLO(MODEL/'best.pt')
    history=pd.read_csv(Path(model.trainer.save_dir)/'results.csv');history.columns=history.columns.str.strip()
    # Read the epoch captured when Ultralytics actually saved the best checkpoint.
    best_epoch=best_epochs[-1] if best_epochs else np.nan
    metrics=[];per=[]
    for split in ['val','test']:
        m=net.val(data=str(OUT/'runtime_dataset.yaml'),split=split,device=env['device'],imgsz=cfg['imgsz'],plots=True,project=str(OUT),name=split)
        row=dict(status='trained',model=net.ckpt_path,split='valid' if split=='val' else split,best_epoch=best_epoch,parameter_count=sum(p.numel() for p in net.model.parameters()),model_size_mb=(MODEL/'best.pt').stat().st_size/1e6)
        for name,obj in [('mask',m.seg),('box',m.box)]:
            row.update({f'{name}_{k}':float(v) for k,v in zip(['precision','recall','map50','map50_95'],obj.mean_results())})
        metrics.append(row)
        for j,c in enumerate(m.seg.ap_class_index):
            per.append(dict(split=row['split'],class_id=int(c),class_name=net.names[int(c)],**{f'mask_{k}':float(v) for k,v in zip(['precision','recall','map50','map50_95'],m.seg.class_result(j))}))
        for p in Path(m.save_dir).glob('*.png'):shutil.copy2(p,FIG/f'{split}_{p.name}')
    import cv2
    predicted={};inf=[]
    # Warm up separately; do not include first-run initialization in measured samples.
    net.predict(str(ROOT/mf.iloc[0].image_path),device=env['device'],verbose=False)
    for row in mf.to_dict('records'):
        begin=time.perf_counter();r=net.predict(str(ROOT/row['image_path']),device=env['device'],verbose=False,retina_masks=True,conf=.25)[0];wall=(time.perf_counter()-begin)*1000
        ps={c:[] for c in range(3)}
        if r.masks is not None:
            for c,p in zip(r.boxes.cls.cpu().numpy().astype(int),r.masks.xyn):ps[c].append(p)
        predicted[row['image_id']]=ps;h,w=r.orig_shape
        a=masks_from_polygons(polygons(ROOT/row['label_path']),h,w);b=masks_from_polygons(ps,h,w)
        ious=[(x&y).sum()/(x|y).sum() for x,y in zip(a,b) if (x|y).any()]
        inf.append(dict(image_id=row['image_id'],split=row['split'],inference_ms=sum(r.speed.values()),wall_ms=wall,mean_union_iou=np.mean(ious) if ious else np.nan))
        if row['split']=='test':cv2.imwrite(str(FIG/(Path(row['image_id']).name+'.jpg')),r.plot())
    table=pd.DataFrame(inf);table.to_csv(OUT/'inference_results.csv',index=False)
    failures=table[table.split=='test'].nsmallest(10,'mean_union_iou').copy();failures['reason']='lowest test semantic-union IoU (relative ranking; not a control failure)';failures.to_csv(OUT/'failure_cases.csv',index=False)
    for rank,part in [('good',table[table.split=='test'].nlargest(5,'mean_union_iou')),('failure',failures.head(5))]:
        for i,row in enumerate(part.itertuples()):shutil.copy2(FIG/(Path(row.image_id).name+'.jpg'),FIG/f'{rank}_{i+1}.jpg')
    for row in metrics:
        samples=table[table.split==row['split']];row['inference_ms']=samples.inference_ms.mean();row['fps']=1000/row['inference_ms'];row['wall_ms']=samples.wall_ms.mean()
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False);pd.DataFrame(per).to_csv(OUT/'per_class_metrics.csv',index=False)
    import matplotlib.pyplot as plt
    for prefix in ['train','val']:
        fig,ax=plt.subplots(figsize=(8,4))
        for c in history:
            if c.startswith(prefix+'/') and 'loss' in c:ax.plot(history.epoch,history[c],label=c)
        ax.set(xlabel='Epoch',ylabel='Loss');ax.legend();fig.tight_layout();fig.savefig(FIG/f'{prefix}_loss.png',dpi=240);plt.close(fig)
    pc=pd.DataFrame(per);fig,ax=plt.subplots(figsize=(8,4));pc[pc.split=='valid'].plot.bar(x='class_name',y=['mask_map50','mask_map50_95'],ax=ax);ax.set_ylim(0,1);fig.tight_layout();fig.savefig(FIG/'per_class_performance.png',dpi=240);plt.close(fig)
    features(predicted)
    (ROOT/'reports/segmentation_baseline_report.md').write_text('# Trained segmentation baseline\n\n'+pd.DataFrame(metrics).to_markdown(index=False)+'\n\nEnvironment: '+json.dumps(env)+'\n\nSeed 42. Validation-only selection. Test evaluated after freezing best checkpoint. All test predictions saved. Semantic union IoU ranks diagnostics, not instance AP. Latency includes preprocessing, inference and postprocessing; wall latency also reported. Predicted train features are in-sample, not out-of-fold.\n')
    (MODEL/'README.md').write_text('Trained best.pt saved by baseline.py. See results/segmentation_baseline for configuration, environment and metrics.\n')
    (FIG/'README.md').write_text('Actual training curves, validation PR/F1 curves, per-class metrics, all test predictions and ranked good/failure examples.\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--train',action='store_true');args=parser.parse_args();env,mf=prepare()
    if args.train:train(env,mf)
    else:unavailable(env)
    print(env)
if __name__=='__main__':main()
