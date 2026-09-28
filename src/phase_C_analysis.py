"""Phase C: validate frozen Phase B evidence and create comparative outputs.

Run with the repository virtual environment: ``./.venv/bin/python src/phase_C_analysis.py``.
The script only reads `input_results/phase_B_results_original` and writes new Phase C artifacts.
"""
from __future__ import annotations

import ast, csv, json, shutil, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat as nbf

from phase_C_statistics import bootstrap_ci, paired_tests, regression_metrics
from phase_C_visualization import actual_predicted, error_box, metric_bars, residuals

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / 'input_results/phase_B_results_original'
OUT = ROOT / 'phase_C_results'
TABLES, STATS, FIGS = OUT/'tables', OUT/'statistics', OUT/'figures'

def parse_num(value):
    if isinstance(value, str) and value.startswith('['): return float(ast.literal_eval(value)[0])
    return float(value)

def read_predictions():
    sources = {
        'Random Forest': IN/'classical_ml/predictions_regression.csv',
        'ANFIS': IN/'anfis/predictions.csv',
        'Custom CNN': IN/'deep_learning/predictions_custom_cnn.csv',
        'MobileNetV3': IN/'deep_learning/predictions_mobilenetv3.csv',
        'ResNet18': IN/'deep_learning/predictions_resnet18.csv',
    }
    output = {}
    for name, path in sources.items():
        data = pd.read_csv(path)
        if name == 'Random Forest': data = data[(data.model == 'rf') & (data.split == 'test')].copy()
        data = data[['image_id','actual','prediction']].copy()
        data.prediction = data.prediction.map(parse_num)
        data.actual = data.actual.astype(float)
        data = data.drop_duplicates('image_id').sort_values('image_id').reset_index(drop=True)
        output[name] = data
    return output

def classification_metrics(actual, prediction, labels=('left','forward','right')):
    cm = np.array([[np.sum((actual == a) & (prediction == b)) for b in labels] for a in labels])
    n = cm.sum(); accuracy = np.trace(cm)/n
    precision = np.divide(np.diag(cm), cm.sum(axis=0), out=np.zeros(3), where=cm.sum(axis=0)!=0)
    recall = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(3), where=cm.sum(axis=1)!=0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(3), where=(precision+recall)!=0)
    expected = (cm.sum(1)*cm.sum(0)).sum()/n**2
    kappa = (accuracy-expected)/(1-expected) if expected != 1 else 0
    # multiclass MCC formula
    c, s = np.trace(cm), n; pk, tk = cm.sum(0), cm.sum(1)
    mcc = (c*s - np.dot(pk,tk))/np.sqrt((s*s-np.dot(pk,pk))*(s*s-np.dot(tk,tk)))
    overall = {'accuracy':accuracy, 'balanced_accuracy':float(recall.mean()), 'macro_precision':float(precision.mean()),
               'macro_recall':float(recall.mean()), 'macro_f1':float(f1.mean()),
               'weighted_precision':float(np.average(precision,weights=cm.sum(1))), 'weighted_recall':accuracy,
               'weighted_f1':float(np.average(f1,weights=cm.sum(1))), 'kappa':kappa, 'MCC':mcc}
    per = [{'class':label,'precision':precision[i],'recall':recall[i],'f1':f1[i],'support':int(cm.sum(1)[i])} for i,label in enumerate(labels)]
    return overall, per, cm

def write_notebook():
    cells = [nbf.v4.new_markdown_cell('# Phase C Comparative Analysis\nThis notebook orchestrates validation and reporting from frozen Phase B exports. It does not train models.'),
             nbf.v4.new_markdown_cell('## Workflow\n1. Input audit\n2. Prediction validation\n3. Common test alignment\n4. Statistics and bootstrap intervals\n5. Error and computational analysis\n6. Tables, figures and report export'),
             nbf.v4.new_code_cell("from pathlib import Path\nimport subprocess, sys\nroot = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd()\nsubprocess.run([sys.executable, str(root/'src'/'phase_C_analysis.py')], check=True)"),
             nbf.v4.new_markdown_cell('## Interpretation guardrails\nClassical ML and ANFIS use oracle ground-truth-mask features and must not be treated as deployment-equivalent to image-only networks. Paired tests use only shared image IDs; all intervals use an image-level bootstrap with a fixed seed.')]
    nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}})
    nbf.write(nb, ROOT/'notebooks/06_phase_C_comparative_analysis.ipynb')

def main():
    for folder in (TABLES, STATS, FIGS): folder.mkdir(parents=True, exist_ok=True)
    models = read_predictions()
    verified=[]; bootstrap=[]
    for name, frame in models.items():
        metric=regression_metrics(frame.actual.to_numpy(),frame.prediction.to_numpy())
        verified.append({'model':name,'feature_source':'oracle ground-truth masks' if name in ('Random Forest','ANFIS') else 'images','test_n':len(frame), **metric,
                         'actual_min':frame.actual.min(),'actual_max':frame.actual.max(),'prediction_min':frame.prediction.min(),'prediction_max':frame.prediction.max()})
        ci=bootstrap_ci(frame.actual.to_numpy(),frame.prediction.to_numpy())
        for metric_name, bounds in ci.items(): bootstrap.append({'model':name,'metric':metric_name,'estimate':metric[metric_name],'ci95_low':bounds[0],'ci95_high':bounds[1],'repetitions':5000,'seed':20260928})
    verified_df=pd.DataFrame(verified); verified_df.to_csv(TABLES/'regression_metrics_verified.csv',index=False)
    boot=pd.DataFrame(bootstrap); boot.to_csv(STATS/'bootstrap_confidence_intervals.csv',index=False)
    common_ids=sorted(set.intersection(*(set(f.image_id) for f in models.values())))
    manifest=[]; common={}
    for name,frame in models.items():
        missing=sorted(set(frame.image_id)-set(common_ids)); common[name]=frame.set_index('image_id').loc[common_ids].reset_index()
        manifest.append({'model':name,'original_test_n':len(frame),'common_test_n':len(common_ids),'excluded_n':len(missing),'excluded_ids':';'.join(missing),'reason':'not present in every selected Phase B prediction export' if missing else 'none'})
    pd.DataFrame(manifest).to_csv(TABLES/'common_test_manifest.csv',index=False)
    errors={name:np.abs(f.prediction.to_numpy()-f.actual.to_numpy()) for name,f in common.items()}
    stats=pd.DataFrame(paired_tests(errors)); stats.to_csv(STATS/'statistical_comparisons.csv',index=False)
    common_metrics=pd.DataFrame([{'model':n, **regression_metrics(f.actual.to_numpy(),f.prediction.to_numpy())} for n,f in common.items()]).set_index('model')
    common_metrics.to_csv(TABLES/'common_test_regression_metrics.csv')
    # classification correction for all test-exported classifiers
    cls=pd.read_csv(IN/'classical_ml/predictions_classification.csv'); all_overall=[]; all_per=[]
    for model,part in cls.groupby('model'):
        overall, per, cm=classification_metrics(part.actual.to_numpy(),part.prediction.to_numpy())
        all_overall.append({'model':model,'test_n':len(part),**overall})
        all_per += [{'model':model,**row} for row in per]
        if model == 'logistic':
            pd.DataFrame(cm,index=['left','forward','right'],columns=['left','forward','right']).to_csv(TABLES/'classification_confusion_matrix_logistic.csv')
            pd.DataFrame(cm/cm.sum(axis=1,keepdims=True),index=['left','forward','right'],columns=['left','forward','right']).to_csv(TABLES/'classification_confusion_matrix_normalized_logistic.csv')
    pd.DataFrame(all_overall).to_csv(TABLES/'classification_metrics_corrected.csv',index=False)
    pd.DataFrame(all_per).to_csv(TABLES/'classification_per_class_corrected.csv',index=False)
    # figures
    indexed=verified_df.set_index('model')
    metric_bars(indexed,'MAE',FIGS/'phase_c_01_verified_mae','Verified test MAE by model')
    metric_bars(indexed,'RMSE',FIGS/'phase_c_02_verified_rmse','Verified test RMSE by model')
    metric_bars(indexed,'R2',FIGS/'phase_c_03_verified_r2','Verified test R2 by model')
    ci_wide=indexed.copy()
    for metric in ('MAE','RMSE'):
        section=boot[boot.metric==metric].set_index('model'); ci_wide[f'{metric}_ci_low']=section.ci95_low; ci_wide[f'{metric}_ci_high']=section.ci95_high
        metric_bars(ci_wide,metric,FIGS/f'phase_c_0{4 if metric=="MAE" else 5}_{metric.lower()}_bootstrap_ci',f'{metric} with 95% bootstrap confidence intervals',True)
    metric_bars(common_metrics,'MAE',FIGS/'phase_c_06_common_test_mae','Common-test MAE comparison')
    metric_bars(common_metrics,'RMSE',FIGS/'phase_c_07_common_test_rmse','Common-test RMSE comparison')
    error_box(errors,FIGS/'phase_c_08_common_test_absolute_error_boxplots')
    actual_predicted(common,FIGS/'phase_c_09_actual_vs_predicted')
    residuals(common,FIGS/'phase_c_10_residual_comparison')
    # figure index
    figure_rows=[]
    for i,png in enumerate(sorted(FIGS.glob('*.png')),1): figure_rows.append({'figure_number':f'C{i:02d}','title':png.stem.replace('_',' '),'phase':'Phase C','source_data':'saved Phase B prediction exports','model':'multiple','sample_count':len(common_ids),'path':str(png.relative_to(ROOT)),'main_report':True,'limitation':'Oracle and image-only model results are not deployment-equivalent.'})
    pd.DataFrame(figure_rows).to_csv(OUT/'figure_index.csv',index=False)
    write_notebook()
    print(json.dumps({'models': list(models), 'common_n':len(common_ids), 'out':str(OUT)},indent=2))

if __name__ == '__main__': main()
