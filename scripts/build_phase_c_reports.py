"""Create Phase C narrative reports, a shareable DOCX/PDF, and ZIP packages."""
from __future__ import annotations
import csv, shutil, zipfile
from datetime import date
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'; C=ROOT/'phase_C_results'; T=C/'tables'; S=C/'statistics'; F=C/'figures'
R.mkdir(exist_ok=True)

def rows(path):
    with open(path, newline='') as f: return list(csv.DictReader(f))
def md_table(items, columns):
    header='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'
    return header + ''.join('| '+' | '.join(str(item.get(c,''))[:180] for c in columns)+' |\n' for item in items)
def write(name, text): (R/name).write_text(text.strip()+'\n', encoding='utf8')

metrics=rows(T/'regression_metrics_verified.csv'); common=rows(T/'common_test_regression_metrics.csv'); stats=rows(S/'statistical_comparisons.csv'); boot=rows(S/'bootstrap_confidence_intervals.csv'); cls=rows(T/'classification_metrics_corrected.csv'); percls=rows(T/'classification_per_class_corrected.csv')
by={r['model']:r for r in metrics}
rf,anfis,mob=by['Random Forest'],by['ANFIS'],by['MobileNetV3']

def build_markdown_reports():
    audit=[
      {'Artifact':'Dataset audit / annotation reports','Phase':'1-2','Path':'reports/dataset_audit_report.md; reports/annotation_quality_report.md','Purpose':'Integrity, polygons, classes and split evidence','Generated':'Yes','Used in Phase C':'Context and limitations','Limitation':'No proof of temporal/site independence'},
      {'Artifact':'Ground-truth feature table','Phase':'A','Path':'phase_B_training_dataset/tabular/features_ground_truth.csv','Purpose':'Oracle segmentation features','Generated':'Yes','Used in Phase C':'Interpretation of tabular models','Limitation':'Not deployable mask source'},
      {'Artifact':'Phase B prediction exports','Phase':'B','Path':'input_results/phase_B_results_original','Purpose':'Per-image frozen model evidence','Generated':'Yes','Used in Phase C':'Metric validation and paired statistics','Limitation':'42 common test images'},
      {'Artifact':'Segmentation baseline metrics','Phase':'A','Path':'results/segmentation_baseline/metrics.csv','Purpose':'Schema/status record','Generated':'No trained result','Used in Phase C':'Limitation','Limitation':'status=not_trained'},
    ]
    write('phase_C_input_audit.md', '# Phase C Input Audit\n\n'+md_table(audit,list(audit[0]))+'\nPhase C reads preserved Phase B exports and writes only to `phase_C_results/`. The local segmentation baseline is explicitly not trained; no segmentation metric is invented.\n')
    write('EXPERIMENTAL_WORKFLOW_TRACE.md', '''# Experimental Workflow Trace

Dataset archive -> dataset audit -> annotation and semantic-class validation -> polygon/ground-truth feature extraction -> geometry-derived continuous and directional research targets -> leakage screening -> classical ML and ANFIS oracle-feature experiments -> image-only Custom CNN, MobileNetV3 and ResNet18 regression -> frozen Phase B exports -> Phase C independent validation, common-test alignment, bootstrap intervals, paired tests and reporting.

The source tree confirms 3 semantic classes (`path`, `cassava_leaves`, `ridge`). Navigation offset is derived from path geometry, not measured steering or motor control. The supplied local segmentation baseline remains `not_trained`; Phase C therefore does not claim trained segmentation performance or a predicted-mask downstream evaluation.''')
    figures=[]; seen=set(); n=0
    for png in sorted((ROOT/'research_figures').rglob('*.png'))+sorted(F.glob('*.png')):
        stem=png.stem
        if stem in seen: continue
        seen.add(stem); n+=1
        phase='Phase C' if png.is_relative_to(F) else ('Phase B' if 'research_figures' in str(png) else 'Phase 1/2')
        figures.append({'Figure':n,'Title':stem.replace('_',' '),'Phase':phase,'Path':str(png.relative_to(ROOT)),'Use':'Main report or appendix','Limitation':'Interpret only with its source metric/prediction export'})
    write('PROJECT_FIGURE_INVENTORY.md','# Project Figure Inventory\n\nScientific figures are counted once even when PNG and PDF versions coexist. Total unique figures indexed: **%d**.\n\n%s' %(n,md_table(figures,list(figures[0]))))
    # copy comprehensive index to requested result location
    with open(C/'figure_index.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(figures[0])); w.writeheader(); w.writerows(figures)
    write('RESEARCH_OBJECTIVE_TRACEABILITY.md', '''# Research Objective Traceability

| Objective | Evidence and method | Finding | Limitation | Status |
| --- | --- | --- | --- | --- |
| Validate cassava-field dataset | Audit, polygon validation, split checks | 802 usable labelled samples; 3 confirmed semantic classes | Adjacent-frame/site correlation not ruled out | Fully addressed |
| Model ANFIS navigation predictor | Five selected oracle features; 3 membership functions; 243 rules | Test MAE %.3f on 42 exported images | Oracle masks and derived target | Partially addressed |
| Model image navigation predictors | Custom CNN, MobileNetV3, ResNet18 per-image exports | MobileNetV3 had lowest image-model MAE (%.3f) | 42-image test; no robot validation | Partially addressed |
| Compare model families fairly | Common IDs, bootstrap and Holm-adjusted paired tests | RF has lower paired AE than the image models in this export | Oracle and image-only settings differ | Partially addressed |
| Produce reproducible results | Phase C script/notebook, CSVs and figures | Frozen-evidence analysis is rerunnable locally | Training artefacts remain incomplete | Fully addressed |'''%(float(anfis['MAE']),float(mob['MAE'])))
    write('PROJECT_EXPLAINED_SIMPLY.md', '''# Project Explained Simply

## What problem were we solving?
We investigated whether a camera view of a cassava field can provide a useful direction-related signal for future robot navigation research. This project does not control a robot or validate hardware safety.

## What did the pipeline do?
Images and YOLO polygon labels were audited, checked for class/annotation issues, and turned into scene descriptions. The path centre relative to the image centre became a normalised research offset: negative is image-left and positive is image-right. Direction labels were derived from that geometry, rather than measured steering commands.

## What models were compared?
Random Forest and ANFIS used engineered features generated from ground-truth masks, so they are oracle upper-bound experiments. Custom CNN, MobileNetV3 and ResNet18 use images directly. MobileNetV3 is a compact transfer-learning model; ResNet18 uses residual/skip connections; ANFIS combines fuzzy membership rules with learned numeric consequents.

## What did we discover?
On the saved 42-image common test set, Random Forest had the lowest MAE (%.3f) and RMSE (%.3f), but this cannot be called a deployment win because it used oracle mask features. Among image-only models, MobileNetV3 had the lowest MAE (%.3f). Phase C found adjusted paired evidence that RF absolute errors differed from each image model, while most comparisons among ANFIS/CNN/transfer models were inconclusive after Holm correction.

## What are the limitations?
The target is derived image geometry, not a steering label; the test set is small; oracle features are not available to a real robot; no predicted-mask downstream experiment, segmentation run, probability scores, physical robot trial or edge-device trial is verified.

## How to explain this project in 2 minutes
We used annotated cassava-field images to locate the visible path and turn its image position into a provisional left/forward/right navigation signal. We compared interpretable mask-derived features, ANFIS and image networks. The best numerical model used ground-truth masks, so it is an upper bound rather than a direct deployment comparison. The best saved image-only MAE belonged to MobileNetV3. Phase C independently recomputed all saved metrics and tested models on exactly the same images.

## How to explain this project in 5 minutes
Start with the dataset audit and annotation checks, then explain segmentation-derived scene features and the path-centre formula. Explain that classical ML/ANFIS use oracle masks whereas CNNs see only images. State that testing uses saved predictions, with bootstrap intervals and paired Wilcoxon tests. Finish with the need for real steering labels, predicted-mask evaluation and robot trials.

## How to explain this project in 10 minutes
Use the full technical report: walk through data integrity, labels, feature/leakage controls, every model family, the 42-image common-test set, uncertainty/statistics, failure analysis and limitations.

## Questions and answers

**Why YOLO segmentation?** It can represent the path, leaves and ridges as pixels/polygons rather than only boxes, which helps geometry reasoning.

**Why segmentation rather than detection?** Navigation depends on path boundaries and centre, which are spatial shapes.

**What is mAP?** It is a detection/segmentation ranking metric. No trained segmentation mAP is claimed locally because the stored baseline says not trained.

**What is path-centre offset?** The horizontal path-centre displacement divided by half image width.

**What is target leakage?** Giving a model the path centre while asking it to predict a formula based on that centre would reveal the answer.

**Why Random Forest and ANFIS?** They provide interpretable, nonlinear feature-based baselines; ANFIS additionally represents fuzzy rules.

**Why CNN, MobileNetV3 and ResNet18?** They test direct image-to-offset learning, including compact and residual transfer-learning approaches.

**What are MAE, RMSE and R2?** MAE is average absolute error; RMSE penalises larger errors more; R2 compares predictions with predicting the mean. Negative R2 means the mean baseline did better on that sample.

**Why balanced accuracy and F1?** They avoid accuracy alone hiding uneven direction-class performance.

**Why no ROC/AUC?** Saved exports contain hard labels, not probability scores.

**Why are oracle features limited?** Ground-truth masks will not be available during deployment.

**Why no physical robot claim?** Camera/controller calibration, safety and field trials are outside the provided evidence.

**What should improve next?** Collect recorded steering labels across farms and conditions; export predicted-mask/out-of-fold features; test on embedded hardware and a safety-controlled robot.'''%(float(rf['MAE']),float(rf['RMSE']),float(mob['MAE'])))
    write('COMPLETE_FIGURE_EXPLANATIONS.md','# Complete Figure Explanations\n\n'+''.join('## Figure %s - %s\n**Phase/source:** %s; `%s`. **Purpose:** document an evidence-backed analysis step. **Axes/sample:** see labels; Phase C common-test plots use 42 paired images. **Observation and interpretation:** values show the recorded comparison, not a universal deployment claim. **Limitation:** oracle-feature and image-only settings differ; absent segmentation outputs cannot be inferred. **Placement:** %s.\n\n'%(x['Figure'],x['Title'],x['Phase'],x['Path'],x['Use']) for x in figures))
    tables=[('Regression metrics verified',T/'regression_metrics_verified.csv','Independent recomputation from per-image predictions.'),('Corrected classification metrics',T/'classification_metrics_corrected.csv','Hard-label overall metrics; no score-based curves.'),('Common test manifest',T/'common_test_manifest.csv','Confirms paired 42-image comparisons.'),('Statistical comparisons',S/'statistical_comparisons.csv','Wilcoxon tests with Holm correction and bootstrap CIs.')]
    write('COMPLETE_TABLE_EXPLANATIONS.md','# Complete Table Explanations\n\n'+''.join('## %s\n**Source:** `%s`. **Purpose:** %s **Interpretation:** values trace to frozen Phase B exports and Phase C recomputation. **Limitation:** numerical precision does not remove the experimental limitations documented in the report.\n\n'%(a,str(b.relative_to(ROOT)),c) for a,b,c in tables))
    write('REPRODUCIBILITY_GUIDE.md','''# Reproducibility Guide

1. Create the project Python environment from `requirements.txt` (Phase C uses pandas, NumPy, SciPy, Matplotlib and nbformat).
2. Keep `input_results/phase_B_results_original/` unchanged; it is the frozen Phase B evidence.
3. Run `MPLCONFIGDIR=/private/tmp/cassava-mpl ./.venv/bin/python src/phase_C_analysis.py`.
4. Run `scripts/build_phase_c_reports.py` to assemble reports and packages.
5. Open `notebooks/06_phase_C_comparative_analysis.ipynb` for the orchestration route.

Dataset audit, annotation validation, Phase A/feature preparation and Phase C run locally. Segmentation and model training may require GPU according to their notebooks, but Phase C itself uses saved predictions and is CPU-sufficient. Results are in `phase_C_results/`; reports are in `reports/`.''')
    report='''# Vision Based Navigation and Control for Autonomous Cassava Farm Robots
## Complete Research Technical Report
**Version:** Phase C evidence-based comparative analysis  \n**Date:** %s

## Executive Summary
This research examined how annotated cassava-farm images may support vision-based navigation research. The workflow audited a 3-class YOLO polygon dataset, checked annotations, derived provisional path-centre navigation targets, constructed engineered features, and evaluated classical ML, ANFIS and image-only deep regression models. The supplied evidence contains 42 aligned saved test predictions for Random Forest, ANFIS, Custom CNN, MobileNetV3 and ResNet18. Phase C recomputed each metric from these predictions, constructed a common test set, and used image-level bootstrap confidence intervals and paired Wilcoxon tests with Holm correction.

Random Forest produced the lowest saved-test MAE (%.3f) and RMSE (%.3f); however, its features come from ground-truth segmentation masks, so it is an oracle upper-bound result, not a direct deployment-equivalent result. Among image-only models, MobileNetV3 had the lowest MAE (%.3f). No physical robot, measured steering command, trained local segmentation result, predicted-mask downstream pipeline or edge deployment is verified by this repository.

## 1. Introduction and research aim
Cassava rows contain variable leaves, ridges and visible paths. A future autonomous robot needs perception that can describe where the path lies. This work evaluates an AI/ML research pipeline, not a validated motor controller. The stated objectives were to validate the data, investigate ANFIS and deep models, compare model families fairly and produce reproducible statistical results.

## 2. Dataset, audit and annotation validation
The usable project data contain 802 labelled samples: 602 train, 120 validation and 80 test. The authoritative classes are `path`, `cassava_leaves` and `ridge`. Audit reports record 28,655 structurally valid polygons, 216 size warnings and 278 manual-review candidates. Exact-hash duplicate checks did not identify cross-split duplicates, but cannot establish independence of nearby video frames or field sites. These are quality checks, not a claim of broad field generalisation.

## 3. Navigation targets and feature engineering
The continuous target is the normalised horizontal displacement between path centre and image centre. Directional labels are left, forward and right (with a broader stop/uncertain concept in configuration); they are derived labels rather than motor commands. Feature engineering includes geometry, colour, texture, orientation and obstruction cues. Leakage prevention excludes path-centre equivalents from target prediction. Classical ML and ANFIS use ground-truth-mask features: this is an oracle condition and is explicitly separated from image-only networks.

## 4. Models and Phase B evidence
Random Forest was the completed classical regression reference. ANFIS used five selected features and 3 membership functions per input, yielding 243 recorded rules. The image networks are a custom CNN, transfer-learned MobileNetV3 and transfer-learned ResNet18. MobileNetV3 is relevant to future constrained devices because it is designed for efficiency; no actual device deployment is claimed. ResNet18 uses residual connections that help optimisation of deeper representations. Training histories and configurations are discussed only where stored; Phase C does not reconstruct missing training evidence.

## 5. Phase C validation and comparison
Phase C parsed saved actual/predicted values and independently recomputed MAE, RMSE, R2, median absolute error, signed bias, residual standard deviation and maximum absolute error. All five eligible models have exactly 42 identical image IDs, so the comparison is paired. The corrected classification table recomputes accuracy, balanced accuracy, macro/weighted metrics, kappa, MCC and class metrics from saved hard labels. ROC/AUC, PR and calibration curves are not reported because scores/probabilities were not supplied.

### Verified regression results

%s

![Verified test MAE](../phase_C_results/figures/phase_c_01_verified_mae.png)

**Figure - verified test MAE.** The bars show MAE in normalised offset units calculated from each saved prediction file. Lower is better. Random Forest is lowest, but its oracle mask source means the bar should be treated as an upper-bound tabular experiment rather than a deployment ranking.

![Common test error distributions](../phase_C_results/figures/phase_c_08_common_test_absolute_error_boxplots.png)

**Figure - paired absolute errors.** Each distribution contains the same 42 images, allowing the spread and large errors to be inspected alongside mean metrics. The figure shows residual variability rather than a causal explanation for individual failures.

## 6. Statistical and uncertainty analysis
Bootstrap intervals resample image-level prediction/target pairs 5,000 times with seed 20260928 and use percentile 95%% intervals. Pairwise Wilcoxon tests operate on image-level absolute-error differences; Holm correction controls the family-wise error rate across pairwise comparisons. The saved comparison table records effect sizes as rank-biserial summaries. Lower aggregate RMSE alone is not labelled statistically better.

%s

## 7. Error and computational interpretation
Residual figures plot prediction minus actual offset against actual offset. They reveal where an output under- or over-estimates the geometry-derived target but do not prove visual causes. The deepest evidence supports only the stored timing values, parameter counts and model sizes; timings from different environments must not be compared as a hardware benchmark. Any future robot must be tested with camera calibration, latency and safety constraints.

![Actual versus predicted offsets](../phase_C_results/figures/phase_c_09_actual_vs_predicted.png)

**Figure - actual versus predicted common-test offsets.** The dashed line is perfect prediction. Scatter around it demonstrates the difficulty of the 42-image test set and makes the compression of several models toward zero visible. It does not demonstrate real steering accuracy.

## 8. Directional classification
The corrected saved-label results are shown below. Balanced accuracy and macro F1 are included because ordinary accuracy alone can conceal uneven class performance. This dataset export has no probability scores, so confidence calibration and ROC/AUC are intentionally omitted.

%s

## 9. Limitations, practical implications and future work
The main limitations are small test evidence, geometry-derived targets, oracle-mask features, absent predicted-mask downstream features, missing trained segmentation outputs locally, hard-label-only classification exports, incomplete histories, environmental diversity uncertainty and no physical robot validation. The results can inform a future perception/control study but cannot be connected directly to motors. Next work should collect measured steering/control data across farms and lighting, generate out-of-fold predicted-mask features, validate segmentation, test temporal/sensor-fusion methods, profile on a target device and conduct safety-controlled robot trials.

## 10. Conclusion
Phase C turns the supplied frozen Phase B outputs into independently validated metrics, paired comparison tables and reproducible uncertainty analysis. It supports a careful conclusion: the oracle Random Forest was numerically strongest on these exports, while MobileNetV3 was the strongest saved image-only MAE result. The research is technically documented for manuscript preparation, subject to preserving its limitations and not overclaiming deployment validity.

## Appendix - Figures and reproducibility
The complete project figure inventory, figure explanations, table explanations and rerun instructions are in the companion reports. Phase C figures are high-resolution PNG/PDF pairs and are indexed in `phase_C_results/figure_index.csv`.
'''%(date.today().isoformat(),float(rf['MAE']),float(rf['RMSE']),float(mob['MAE']),md_table(metrics,['model','feature_source','test_n','MAE','RMSE','R2','bias']),md_table(stats,['model_a','model_b','common_n','difference_a_minus_b','ci95_low','ci95_high','holm_adjusted_p_value','interpretation']),md_table(cls,['model','test_n','accuracy','balanced_accuracy','macro_f1','kappa','MCC']))
    write('COMPLETE_RESEARCH_TECHNICAL_REPORT.md',report)

def make_docx():
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.section import WD_SECTION
    doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.8); sec.bottom_margin=Inches(.8)
    normal=doc.styles['Normal']; normal.font.name='Aptos'; normal.font.size=Pt(10)
    title=doc.add_paragraph(style='Title'); title.alignment=WD_ALIGN_PARAGRAPH.CENTER; title.add_run('Vision Based Navigation and Control for Autonomous Cassava Farm Robots')
    sub=doc.add_paragraph(); sub.alignment=WD_ALIGN_PARAGRAPH.CENTER; sub.add_run('Complete Research Technical Report\nPhase C Comparative Analysis\n'+date.today().isoformat())
    doc.add_page_break()
    # Add report headings/body and figures based on concise sections
    sections=[
      ('Executive Summary','This report documents the Phase C validation of frozen Phase B evidence. Random Forest has the lowest saved-test error but uses oracle ground-truth-mask features. MobileNetV3 is the lowest-MAE image-only export. The common paired test set contains 42 images.'),
      ('Dataset and workflow','The usable data contain 802 samples, 3 confirmed semantic classes and geometry-derived navigation targets. Classical ML and ANFIS use oracle feature inputs; direct CNN/transfer models use images. No physical robot claim is made.'),
      ('Verified comparative results','Metrics were recomputed from saved actual/predicted pairs. Bootstrap confidence intervals use 5,000 image-level resamples; paired Wilcoxon tests use the 42 shared image IDs and Holm correction.'),
      ('Interpretation and limitations','The oracle/image-only distinction prevents a direct deployment ranking. Targets are not measured steering commands, and segmentation execution, predicted-mask features, edge testing and robot validation remain unavailable.')]
    for heading, body in sections:
        doc.add_heading(heading,level=1); doc.add_paragraph(body)
        if heading=='Verified comparative results':
            for img,caption in [('phase_c_01_verified_mae.png','Verified test MAE by model'),('phase_c_08_common_test_absolute_error_boxplots.png','Paired absolute-error distribution'),('phase_c_09_actual_vs_predicted.png','Actual versus predicted common-test offsets'),('phase_c_10_residual_comparison.png','Residual comparison')]:
                p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(F/img),width=Inches(6.2)); c=doc.add_paragraph('Figure - '+caption); c.alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading('Verified regression table',level=1)
    table=doc.add_table(rows=1,cols=5); table.style='Table Grid'
    for cell,value in zip(table.rows[0].cells,['Model','Feature source','N','MAE','RMSE']): cell.text=value
    for row in metrics:
        cells=table.add_row().cells
        for cell,key in zip(cells,['model','feature_source','test_n','MAE','RMSE']): cell.text=str(row[key])[:30]
    doc.add_heading('Appendix and reproducibility',level=1); doc.add_paragraph('See the accompanying Markdown report, figure catalogue, table catalogue, objective traceability and reproducibility guide for complete traceability.')
    footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER; footer.text='Cassava Navigation AI - Phase C'
    doc.save(R/'COMPLETE_RESEARCH_TECHNICAL_REPORT.docx')

def package():
    files=['notebooks/06_phase_C_comparative_analysis.ipynb','src/phase_C_analysis.py','src/phase_C_statistics.py','src/phase_C_visualization.py']
    for dest, subset in [(ROOT/'cassava_navigation_phase_C_results.zip', ['phase_C_results','reports']), (ROOT/'cassava_navigation_complete_research_report_package.zip',['reports','phase_C_results/figures','phase_C_results/tables'])]:
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
            for rel in files:
                p=ROOT/rel
                if p.exists(): z.write(p,rel)
            for prefix in subset:
                p=ROOT/prefix
                for f in p.rglob('*'):
                    if f.is_file() and not f.name.endswith('.pdf') or (f.is_file() and f.name=='COMPLETE_RESEARCH_TECHNICAL_REPORT.pdf'):
                        z.write(f,f.relative_to(ROOT))

if __name__=='__main__':
    build_markdown_reports(); make_docx(); package()
