#!/usr/bin/env python3
"""Validate Phase B exports and create only evidence-backed research figures."""
from pathlib import Path
import ast, shutil, zipfile, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mean_absolute_error(a,p): return float(np.mean(np.abs(np.asarray(a)-np.asarray(p))))
def mean_squared_error(a,p): return float(np.mean((np.asarray(a)-np.asarray(p))**2))
def median_absolute_error(a,p): return float(np.median(np.abs(np.asarray(a)-np.asarray(p))))
def r2_score(a,p):
    a=np.asarray(a); return float(1-np.sum((a-np.asarray(p))**2)/np.sum((a-a.mean())**2))
def confusion_matrix(a,p,labels): return np.array([[sum((np.asarray(a)==x)&(np.asarray(p)==y)) for y in labels] for x in labels])
def precision_recall_fscore_support(a,p,labels=None,average=None,zero_division=0):
    labs=list(labels or sorted(set(a)|set(p))); cm=confusion_matrix(a,p,labs); tp=np.diag(cm); ps=cm.sum(0); rs=cm.sum(1)
    pr=np.divide(tp,ps,out=np.zeros_like(tp,dtype=float),where=ps!=0); rc=np.divide(tp,rs,out=np.zeros_like(tp,dtype=float),where=rs!=0); f=np.divide(2*pr*rc,pr+rc,out=np.zeros_like(pr),where=(pr+rc)!=0); sup=rs
    if average=='macro': return pr.mean(),rc.mean(),f.mean(),None
    if average=='weighted': return np.average(pr,weights=sup),np.average(rc,weights=sup),np.average(f,weights=sup),None
    return pr,rc,f,sup
def accuracy_score(a,p): return float(np.mean(np.asarray(a)==np.asarray(p)))
def balanced_accuracy_score(a,p): return float(precision_recall_fscore_support(a,p,labels=sorted(set(a)|set(p)))[1].mean())
def cohen_kappa_score(a,p,labels):
    cm=confusion_matrix(a,p,labels); n=cm.sum(); po=np.trace(cm)/n; pe=(cm.sum(0)*cm.sum(1)).sum()/n**2; return float((po-pe)/(1-pe)) if pe!=1 else 1.
def matthews_corrcoef(a,p):
    labs=sorted(set(a)|set(p)); cm=confusion_matrix(a,p,labs); t=cm.sum(); c=np.trace(cm); s=cm.sum(1); q=cm.sum(0); den=np.sqrt((t*t-(q*q).sum())*(t*t-(s*s).sum())); return float((c*t-(s*q).sum())/den) if den else 0.

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'input_results/phase_B_results_original'
FIG=ROOT/'research_figures'; VAL=ROOT/'validated_results'; REP=ROOT/'reports'
for d in [FIG,VAL/ 'regression', VAL/'classification',REP]: d.mkdir(parents=True,exist_ok=True)
for d in ['01_dataset','02_segmentation_navigation','03_classical_ml','04_anfis','05_deep_learning','06_model_comparison']:(FIG/d).mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'legend.fontsize':8})
rows=[]
def out(n, section, title, source, samples, finding, limitation, draw):
    folder={'Dataset and annotation analysis':'01_dataset','Segmentation and navigation targets':'02_segmentation_navigation','Classical ML':'03_classical_ml','ANFIS':'04_anfis','CNN and transfer learning':'05_deep_learning','Overall model comparison':'06_model_comparison'}[section]
    stem=f'figure_{n:02d}_'+''.join(c if c.isalnum() else '_' for c in title.lower()).strip('_')
    p=FIG/folder/(stem+'.png')
    fig=draw(); fig.suptitle(f'Figure {n}. {title}',y=.995,fontsize=13,fontweight='bold'); fig.tight_layout()
    fig.savefig(p,dpi=300,bbox_inches='tight'); fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight'); plt.close(fig)
    rows.append(dict(figure_number=n,figure_title=title,research_section=section,source_data=source,sample_count=samples,figure_filename=str(p.relative_to(FIG)),generation_status='generated',main_finding=finding,important_limitation=limitation))
def unavailable(n, section, title, why):
    rows.append(dict(figure_number=n,figure_title=title,research_section=section,source_data='Not available in supplied executed outputs',sample_count='',figure_filename='',generation_status='requires additional evaluation',main_finding='',important_limitation=why))
def bar(names,vals,ylabel,colors=None,rot=0):
 def f():
  fig,ax=plt.subplots(figsize=(7,4)); ax.bar(names,vals,color=colors); ax.set_ylabel(ylabel); ax.tick_params(axis='x',rotation=rot); ax.grid(axis='y',alpha=.25); return fig
 return f

# Dataset / annotation figures
split=pd.read_csv(ROOT/'phase_B_training_dataset/metadata/split_summary.csv')
out(1,'Dataset and annotation analysis','Dataset split distribution','metadata/split_summary.csv',int(split.images.sum()),'The prepared dataset contains 602 train, 120 validation, and 80 test images.','Counts are image-level; target eligibility is lower.',bar(split['split'],split['images'],'Images',['#457b9d','#e9c46a','#e76f51']))
classes=pd.read_csv(REP/'class_distribution.csv')
out(2,'Dataset and annotation analysis','Semantic class distribution','reports/class_distribution.csv',int(classes.annotation_count.sum()),'Ridge annotations are the most numerous class.','Annotation instances are not independent image samples.',bar(classes.class_name,classes.annotation_count,'Annotation instances',['#457b9d','#2a9d8f','#e9c46a']))
prop=pd.read_csv(REP/'image_properties.csv');
out(3,'Dataset and annotation analysis','Image resolution distribution','reports/image_properties.csv',len(prop),'All audited images have the recorded image dimensions.','The audit includes an extra checkpoint image noted in data-quality outputs.',lambda: (lambda fig,ax:(ax.hist(prop.width,bins=20,color='#457b9d'),ax.set(xlabel='Image width (pixels)',ylabel='Images'),ax.grid(alpha=.25),fig)[-1])(*plt.subplots(figsize=(7,4))))
poly=pd.read_csv(REP/'polygon_quality_checks.csv'); names={0:'path',1:'cassava_leaves',2:'ridge'}
out(4,'Dataset and annotation analysis','Polygon area distribution by class','reports/polygon_quality_checks.csv',len(poly),'Polygon areas vary substantially across semantic classes.','Areas are normalized image fractions and may overlap spatially.',lambda: (lambda fig,ax:( [ax.hist(poly.loc[poly.class_id==k,'area_normalized'],bins=40,alpha=.55,label=v) for k,v in names.items()],ax.set(xlabel='Normalized polygon area',ylabel='Polygons'),ax.legend(),ax.grid(alpha=.25),fig)[-1])(*plt.subplots(figsize=(7,4))))
ann=pd.read_csv(REP/'annotation_statistics.csv')
out(5,'Dataset and annotation analysis','Annotation density','reports/annotation_statistics.csv',len(ann),'Annotation density is heterogeneous across images.','One audit-only checkpoint image may be represented.',lambda: (lambda fig,ax:(ax.hist(ann.annotation_count,bins=30,color='#2a9d8f'),ax.set(xlabel='Annotation instances per image',ylabel='Images'),ax.grid(alpha=.25),fig)[-1])(*plt.subplots(figsize=(7,4))))
valid=poly.is_valid_row & poly.is_valid_area; suspicious=poly.is_suspicious
out(6,'Dataset and annotation analysis','Annotation quality analysis','reports/polygon_quality_checks.csv',len(poly),f'{valid.sum()} polygon rows satisfy row and area checks; {suspicious.sum()} are flagged suspicious.','Flagged/suspicious can overlap valid rows.',bar(['Valid rows and areas','Suspicious flags','Invalid row or area'],[int(valid.sum()),int(suspicious.sum()),int((~valid).sum())],'Polygon rows',['#2a9d8f','#e9c46a','#e76f51']))
existing=ROOT/'figures/annotated_samples.png'
if existing.exists():
 def old():
  im=plt.imread(existing); fig,ax=plt.subplots(figsize=(10,6)); ax.imshow(im); ax.axis('off'); return fig
 out(7,'Dataset and annotation analysis','Representative annotated dataset samples','figures/annotated_samples.png', '', 'Panel is a previously generated dataset-audit visualization.','Recovered image; its original selection protocol is documented only in Phase A.',old)
else: unavailable(7,'Dataset and annotation analysis','Representative annotated dataset samples','No annotation-overlay source was supplied in the Phase B package.')

# Navigation
targets=pd.read_csv(ROOT/'phase_B_training_dataset/tabular/classification_targets.csv'); offs=pd.read_csv(ROOT/'phase_B_training_dataset/tabular/regression_targets.csv')
order=['left','forward','right','stop_or_uncertain']; cnt=targets.direction_label.value_counts().reindex(order,fill_value=0)
out(13,'Segmentation and navigation targets','Directional navigation-target distribution','tabular/classification_targets.csv',len(targets),'Only left, forward, and right occur among eligible classification targets; labels are derived navigation targets.','No physical STOP command is established; stop_or_uncertain has no eligible records.',bar(cnt.index,cnt.values,'Eligible samples',['#e76f51','#2a9d8f','#457b9d','#999999']))
out(14,'Segmentation and navigation targets','Continuous navigation-target distribution','tabular/regression_targets.csv',len(offs),'Offsets cover the normalized path-centre range among 395 eligible samples.','These are geometry-derived labels, not measured steering commands.',lambda: (lambda fig,ax:(ax.hist(offs.normalized_offset.dropna(),bins=30,color='#457b9d'),ax.axvline(0,color='black',lw=1),ax.set(xlabel='Normalized path-centre offset',ylabel='Eligible samples'),ax.grid(alpha=.25),fig)[-1])(*plt.subplots(figsize=(7,4))))
for n,t,w in [(8,'Segmentation training curves','Epoch-level segmentation training logs are absent; baseline metrics say not_trained.'),(9,'Segmentation precision and recall','Segmentation baseline is explicitly not_trained.'),(10,'Segmentation mAP comparison','Segmentation baseline is explicitly not_trained.'),(11,'Ground truth versus predicted masks','No segmentation predictions or masks were supplied.'),(12,'Segmentation failure cases','No segmentation inference/failure records were supplied.')]: unavailable(n,'Segmentation and navigation targets',t,w)

# Validate regression predictions
reg=pd.read_csv(SRC/'classical_ml/predictions_regression.csv'); regtest=reg[reg.split.eq('test')].copy()
def reg_metrics(df,model,source='oracle'):
 a=df.actual.astype(float); p=df.prediction.astype(float); return dict(model=model,feature_source=source,n=len(df),MAE=mean_absolute_error(a,p),RMSE=mean_squared_error(a,p)**.5,R2=r2_score(a,p),median_absolute_error=median_absolute_error(a,p))
rmet=[]
for (source, feature_set, model), x in regtest.groupby(['source','feature_set','model']):
    item=reg_metrics(x,model,source); item['feature_set']=feature_set; rmet.append(item)
anf=pd.read_csv(SRC/'anfis/predictions.csv'); rmet.append(reg_metrics(anf,'anfis','oracle'))
deep=[]
for fn in ['custom_cnn','mobilenetv3','resnet18']:
 d=pd.read_csv(SRC/f'deep_learning/predictions_{fn}.csv'); d['prediction']=d.prediction.map(lambda x: float(ast.literal_eval(x)[0]) if str(x).startswith('[') else float(x)); deep.append(d); rmet.append(reg_metrics(d,fn,'images'))
rmet=pd.DataFrame(rmet); rmet.to_csv(VAL/'regression/recomputed_test_metrics.csv',index=False)
# Saved classical test predictions contain one chosen model per feature set. The
# only defensible multi-model comparison is validation-only selection output.
classic_valid=pd.read_csv(SRC/'classical_ml/regression_metrics.csv')
classic_valid=classic_valid[(classic_valid.source.eq('oracle'))&(classic_valid.feature_set.eq('all_non_leaking'))&(classic_valid.split.eq('valid'))].sort_values('MAE')
for n,col,title in [(15,'MAE','Classical regression validation MAE comparison'),(16,'RMSE','Classical regression validation RMSE comparison'),(17,'R2','Classical regression validation R² comparison')]: out(n,'Classical ML',title,'classical_ml/regression_metrics.csv',56,'All models are compared on the validation-only model-selection output.','A multi-model held-out test comparison is unavailable; all models use oracle ground-truth-mask features.',bar(classic_valid.model,classic_valid[col],col,['#457b9d']*len(classic_valid)))
sel=regtest[(regtest.model.eq('rf'))&(regtest.feature_set.eq('all_non_leaking'))].copy(); lo=min(sel.actual.min(),sel.prediction.min()); hi=max(sel.actual.max(),sel.prediction.max())
out(18,'Classical ML','Classical RF actual versus predicted offset','classical_ml/predictions_regression.csv',len(sel),'The selected random forest is evaluated on saved oracle-feature test predictions.','Predictions are derived targets and model uses oracle features.',lambda: (lambda fig,ax:(ax.scatter(sel.actual,sel.prediction,color='#457b9d'),ax.plot([lo,hi],[lo,hi],'k--'),ax.set(xlabel='Actual normalized offset',ylabel='Predicted normalized offset'),ax.grid(alpha=.25),fig)[-1])(*plt.subplots(figsize=(6,5))))
out(19,'Classical ML','Classical RF residual distribution','classical_ml/predictions_regression.csv',len(sel),'Residual spread reveals errors relative to geometry-derived offset targets.','Oracle feature source prevents deployment interpretation.',lambda: (lambda fig,ax:(ax.hist(sel.prediction-sel.actual,bins=15,color='#457b9d'),ax.axvline(0,color='black'),ax.set(xlabel='Residual (predicted − actual offset)',ylabel='Test samples'),ax.grid(alpha=.25),fig)[-1])(*plt.subplots(figsize=(7,4))))
# classification: select logistic, fixed class order
clf=pd.read_csv(SRC/'classical_ml/predictions_classification.csv'); cp=clf[(clf.model.eq('logistic'))&(clf.feature_set.eq('all_non_leaking'))]; labs=['left','forward','right']; y=cp.actual; p=cp.prediction
pr,rc,f1,_=precision_recall_fscore_support(y,p,labels=labs,zero_division=0); cm=confusion_matrix(y,p,labels=labs)
summary=pd.DataFrame([dict(model='logistic',n=len(cp),accuracy=accuracy_score(y,p),balanced_accuracy=balanced_accuracy_score(y,p),macro_precision=precision_recall_fscore_support(y,p,average='macro',zero_division=0)[0],macro_recall=precision_recall_fscore_support(y,p,average='macro',zero_division=0)[1],macro_f1=precision_recall_fscore_support(y,p,average='macro',zero_division=0)[2],weighted_f1=precision_recall_fscore_support(y,p,average='weighted',zero_division=0)[2],kappa=cohen_kappa_score(y,p,labels=labs),MCC=matthews_corrcoef(y,p))]); summary.to_csv(VAL/'classification/recomputed_classification_metrics.csv',index=False); pd.DataFrame({'class':labs,'precision':pr,'recall':rc,'f1':f1}).to_csv(VAL/'classification/per_class_metrics.csv',index=False); pd.DataFrame(cm,index=labs,columns=labs).to_csv(VAL/'classification/confusion_matrix.csv')
out(20,'Classical ML','Directional classification confusion matrix','validated_results/classification/confusion_matrix.csv',len(cp),'Metrics are recomputed with class order left, forward, right.','Only 16 test samples; class estimates are unstable.',lambda: (lambda fig,ax:(im:=ax.imshow(cm,cmap='Blues'),ax.set(xticks=range(3),yticks=range(3),xticklabels=labs,yticklabels=labs,xlabel='Predicted derived direction',ylabel='Actual derived direction'),[ax.text(j,i,str(cm[i,j]),ha='center',va='center') for i in range(3) for j in range(3)],fig.colorbar(im,ax=ax,label='Samples'),fig)[-1])(*plt.subplots(figsize=(6,5))))
imp=pd.read_csv(SRC/'classical_ml/feature_importance.csv').sort_values('permutation_importance',ascending=True).tail(15)
out(21,'Classical ML','RF permutation feature importance','classical_ml/feature_importance.csv',len(sel),'Permutation importance is from the selected oracle random forest.','Importance is model- and subset-specific; negative values are retained.',lambda: (lambda fig,ax:(ax.barh(imp.feature,imp.permutation_importance,color='#2a9d8f'),ax.set(xlabel='Permutation importance',ylabel='Feature'),ax.grid(axis='x',alpha=.25),fig)[-1])(*plt.subplots(figsize=(8,6))))

# ANFIS
vloss=SRC/'figures/anfis/validation_loss.png'
if vloss.exists():
 def anfplot():
  fig,ax=plt.subplots(figsize=(7,4)); ax.imshow(plt.imread(vloss)); ax.axis('off'); return fig
 out(22,'ANFIS','ANFIS recorded validation-loss plot','figures/anfis/validation_loss.png',len(anf),'Recovered original ANFIS visualization.','Underlying epoch values are not exported, so the figure is reproduced as an image.',anfplot)
for n,title,kind in [(24,'ANFIS actual versus predicted offset','scatter'),(25,'ANFIS residual distribution','hist'),(26,'ANFIS absolute error by sample','line')]:
 def make(kind=kind):
  fig,ax=plt.subplots(figsize=(7,4)); e=anf.prediction-anf.actual
  if kind=='scatter': ax.scatter(anf.actual,anf.prediction,color='#e76f51'); ax.plot([-1,1],[-1,1],'k--'); ax.set(xlabel='Actual normalized offset',ylabel='Predicted normalized offset')
  elif kind=='hist': ax.hist(e,bins=15,color='#e76f51'); ax.axvline(0,color='black'); ax.set(xlabel='Residual (predicted − actual offset)',ylabel='Test samples')
  else: ax.plot(range(1,len(anf)+1),abs(e),marker='o',ms=3,color='#e76f51'); ax.set(xlabel='Stable saved-prediction row order',ylabel='Absolute error (normalized offset)')
  ax.grid(alpha=.25); return fig
 out(n,'ANFIS',title,'anfis/predictions.csv',len(anf),'Plot uses saved ANFIS test predictions.','The ANFIS uses oracle features and only a 42-sample test set.',make)
unavailable(23,'ANFIS','ANFIS membership functions','Checkpoint serialization lacks independently documented membership-function extraction metadata.')
unavailable(27,'ANFIS','ANFIS response surface','Requires checkpoint recovery evaluation with documented feature scaling and reference values.')

# Deep learning
for n,m in [(28,'custom_cnn'),(29,'mobilenetv3'),(30,'resnet18')]:
 h=pd.read_csv(SRC/f'deep_learning/training_history_{m}.csv')
 out(n,'CNN and transfer learning',f'{m} validation MAE across epochs',f'deep_learning/training_history_{m}.csv',len(h),f'The saved history records validation MAE across {len(h)} epochs.','Training/validation loss was not exported; this is not a loss curve.',lambda h=h,m=m:(lambda fig,ax:(ax.plot(h.epoch,h.validation_MAE,marker='o',label=m),ax.set(xlabel='Epoch',ylabel='Validation MAE (normalized offset)'),ax.legend(),ax.grid(alpha=.25),fig)[-1])(*plt.subplots(figsize=(7,4))))
dm=rmet[rmet.feature_source.eq('images')].sort_values('MAE')
for n,c,t in [(31,'MAE','Deep-learning MAE comparison'),(32,'RMSE','Deep-learning RMSE comparison')]: out(n,'CNN and transfer learning',t,'validated_results/regression/recomputed_test_metrics.csv',42,'All three image models use predictions for the same 42 saved test IDs.','Scores concern provisional geometry-derived targets.',bar(dm.model,dm[c],c,['#e76f51','#e9c46a','#457b9d']))
def dscatter():
 fig,ax=plt.subplots(figsize=(7,5));
 for d in deep: ax.scatter(d.actual,d.prediction,label=d.model.iloc[0],alpha=.75)
 ax.plot([-1,1],[-1,1],'k--'); ax.set(xlabel='Actual normalized offset',ylabel='Predicted normalized offset'); ax.legend(); ax.grid(alpha=.25); return fig
out(33,'CNN and transfer learning','Deep-learning actual versus predicted offsets','deep_learning/predictions_*.csv',42,'Model predictions are compared on common saved test image IDs.','No confidence intervals or repeated runs are available.',dscatter)
# real failure images largest ResNet abs errors
fail=deep[-1].assign(err=lambda x:abs(x.prediction-x.actual)).nlargest(6,'err')
def failpanel():
 fig,axs=plt.subplots(2,3,figsize=(12,7));
 for ax,(_,r) in zip(axs.flat,fail.iterrows()):
  fn=r.image_id+'.jpg'; p=ROOT/'phase_B_training_dataset'/('images/'+r.image_id.split('/')[0]+'/'+Path(fn).name)
  if p.exists(): ax.imshow(plt.imread(p))
  ax.set_title(f"actual {r.actual:.3f}; pred {r.prediction:.3f}",fontsize=9); ax.axis('off')
 return fig
out(34,'CNN and transfer learning','ResNet18 largest-error real test images','deep_learning/predictions_resnet18.csv plus phase_B_training_dataset/images',len(fail),'Panel shows the six highest absolute-error saved ResNet18 test predictions.','Errors are against derived offsets; it is not a segmentation-error analysis.',failpanel)

# Overall
overall=rmet.sort_values('MAE')
for n,c,t in [(35,'MAE','Overall test MAE comparison'),(36,'RMSE','Overall test RMSE comparison'),(37,'R2','Overall test R² comparison')]: out(n,'Overall model comparison',t,'validated_results/regression/recomputed_test_metrics.csv',42,'Metrics are recomputed from available saved test predictions.','Classical RF and ANFIS are oracle-feature models; deep models use images.',bar(overall.model,overall[c],c,['#457b9d' if s=='oracle' else '#e76f51' for s in overall.feature_source],45))
mm=pd.read_csv(SRC/'deep_learning/model_metrics.csv');
out(38,'Overall model comparison','Deep-model size versus prediction error','deep_learning/model_metrics.csv',42,'The deep checkpoints show a size/error trade-off.','Only deep model sizes are consistently stored; serialization details matter.',lambda: (lambda fig,ax:(ax.scatter(mm.parameter_count/1e6,mm.MAE,s=80,color='#e76f51'),[ax.annotate(x,(a/1e6,b),xytext=(4,4),textcoords='offset points') for x,a,b in zip(mm.model,mm.parameter_count,mm.MAE)],ax.set(xlabel='Parameters (millions)',ylabel='Test MAE'),ax.grid(alpha=.25),fig)[-1])(*plt.subplots(figsize=(7,4))))
out(39,'Overall model comparison','Deep-model latency versus prediction error','deep_learning/model_metrics.csv',42,'Saved deep-model timing values can be inspected alongside MAE.','Timing hardware/batch semantics require the notebook environment; no cross-family claim is made.',lambda: (lambda fig,ax:(ax.scatter(mm.inference_time,mm.MAE,s=80,color='#e9c46a'),[ax.annotate(x,(a,b),xytext=(4,4),textcoords='offset points') for x,a,b in zip(mm.model,mm.inference_time,mm.MAE)],ax.set(xlabel='Recorded inference time',ylabel='Test MAE'),ax.grid(alpha=.25),fig)[-1])(*plt.subplots(figsize=(7,4))))
common=set.intersection(*[set(d.image_id) for d in [sel[['image_id']],anf[['image_id']]]+deep]); err=[]
for name,d in [('rf',sel),('anfis',anf)]+[(d.model.iloc[0],d) for d in deep]: err.append(pd.DataFrame({'model':name,'absolute_error':abs(d.set_index('image_id').loc[sorted(common)].prediction-d.set_index('image_id').loc[sorted(common)].actual)}))
errs=pd.concat(err)
out(40,'Overall model comparison','Absolute error distribution across models','saved regression prediction CSVs',len(common),'All available regression models are aligned to common saved test image IDs.','RF/ANFIS use oracle features whereas deep models use images.',lambda: (lambda fig,ax:(ax.boxplot([errs[errs.model.eq(m)].absolute_error for m in errs.model.unique()],tick_labels=errs.model.unique()),ax.set(xlabel='Model',ylabel='Absolute error (normalized offset)'),ax.grid(axis='y',alpha=.25),fig)[-1])(*plt.subplots(figsize=(8,4))))

# Inventory, reports and package
pd.DataFrame(rows).sort_values('figure_number').to_csv(FIG/'figure_index.csv',index=False)
inv=[]
for p in sorted(SRC.rglob('*')):
 if p.is_file(): inv.append({'path':str(p.relative_to(SRC)),'bytes':p.stat().st_size,'usable_for_figures':p.suffix.lower() in {'.csv','.png'},'notes':'Original archive retained unchanged'})
pd.DataFrame(inv).to_csv(REP/'phase_B_results_inventory.csv',index=False)
generated=pd.DataFrame(rows).query("generation_status == 'generated'")
explain=['# Research figures: explanations','',f'Generated evidence-backed figures: **{len(generated)}** of 40 requested. Entries marked `requires additional evaluation` are deliberately not fabricated.','']
for r in pd.DataFrame(rows).sort_values('figure_number').itertuples(): explain += [f'## Figure {r.figure_number} — {r.figure_title}','',f'**Status:** {r.generation_status}. **Source:** {r.source_data}. **Sample count:** {r.sample_count or "not available"}.', '',f'**What it shows and interpretation:** {r.main_finding or "No claim is made because the required stored evidence is unavailable."}', '',f'**Limitation:** {r.important_limitation}', '', '**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.', '']
(REP/'RESEARCH_FIGURES_EXPLANATION.md').write_text('\n'.join(explain))
(REP/'phase_B_experiment_audit.md').write_text('# Phase B experiment audit\n\nThe executed notebook completed classical oracle-feature regression/classification, ANFIS oracle-feature regression, and three image-regression models. It did **not** train the segmentation baseline: its stored metrics are `not_trained`. Prediction files support 42 regression test rows; directional classification has 16 test rows (left 4, forward 4, right 8). The notebook reports 297/56/42 eligible regression rows and 116/22/16 classification rows for train/validation/test. Oracle features were the only available source; predicted masks were unavailable.\n')
(REP/'phase_B_results_validation.md').write_text('# Phase B results validation\n\nRegression metrics were independently recomputed from every saved test-prediction file and saved under `validated_results/regression/`. Classification metrics and the confusion matrix were recomputed from saved logistic predictions using fixed class order left, forward, right, under `validated_results/classification/`. The original exports remain under `input_results/phase_B_results_original/`.\n\nImportant comparability: the saved 42 regression image IDs overlap across all available regression prediction files. Classical RF and ANFIS use oracle ground-truth-mask features, while the CNNs use images; their scores therefore do not establish equivalent deployability. Directional classification has only 16 test samples, so per-class estimates are highly uncertain. Segmentation was not trained and has no evaluation outputs.\n')
alloc=['# Figure work allocation','', '| Contributor | Area | Figures |', '|---|---|---|', '| 1 | Dataset and annotation analysis | 1–7 |','| 2 | Segmentation and navigation targets | 8–14 |','| 3 | Classical ML | 15–21 |','| 4 | ANFIS | 22–27 |','| 5 | CNN and transfer learning | 28–34 |','| 6 | Overall comparison and synthesis | 35–40 |','', 'All contributors should state that labels are provisional geometry-derived navigation targets. Contributors 2 and 4 should explicitly report unavailable figures as evidence gaps, not negative findings. Oracle-feature work (classical ML/ANFIS) must not be described as predicted-mask deployment performance. Each contributor should write the assigned results/discussion section, cite source files in the figure index, discuss limitations, and avoid claiming independent data collection.']
(REP/'FIGURE_WORK_ALLOCATION.md').write_text('\n'.join(alloc))
(FIG/'README.md').write_text('# Research figures\n\nThis directory contains only evidence-backed figures generated from supplied Phase A/Phase B files. See `figure_index.csv` and `../reports/RESEARCH_FIGURES_EXPLANATION.md`; unavailable requested figures are indexed explicitly.\n')
# Recovery notebook intentionally concise: no execution / no training
nb={'cells':[{'cell_type':'markdown','metadata':{},'source':['# Phase B figure recovery\n','Load the existing Phase B results archive and existing checkpoints only. Do not retrain Phase B models. Recover segmentation predictions, ANFIS membership functions/response surface, or documented timing only when their original checkpoint configuration and feature scaling are available.']},{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':['# Upload the original dataset/results archives, then implement only the missing recovery target.\n'] }],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
import json
(ROOT/'notebooks/05_phase_B_figures_recovery_colab.ipynb').write_text(json.dumps(nb,indent=2))
zip_path=ROOT/'cassava_navigation_research_figures.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as z:
 for p in list(FIG.rglob('*'))+list(VAL.rglob('*'))+[REP/'RESEARCH_FIGURES_EXPLANATION.md',REP/'FIGURE_WORK_ALLOCATION.md',REP/'phase_B_results_validation.md',REP/'phase_B_experiment_audit.md',REP/'phase_B_results_inventory.csv',ROOT/'scripts/generate_phase_b_research_package.py',ROOT/'notebooks/05_phase_B_figures_recovery_colab.ipynb']:
  if p.is_file(): z.write(p,p.relative_to(ROOT))
print(f'Generated {len(generated)} figures; package: {zip_path}')
