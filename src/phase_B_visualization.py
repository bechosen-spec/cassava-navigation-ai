"""Recovery-first Phase B figure generation.

This module intentionally never trains a model.  It consumes the exported Phase B
CSV files, recomputes display metrics from saved predictions, and records which
plots cannot be recovered when histories/checkpoints lack the required metadata.
It is imported by the Phase B Colab notebook in ``RECOVER_RESULTS`` mode.
"""
from __future__ import annotations

import ast
import csv
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score,
                             median_absolute_error)

CLASS_ORDER = ["left", "forward", "right"]
ROOT = Path(__file__).resolve().parents[1]


def _prediction_number(value):
    """Read scalar strings and historic '[scalar]' CSV values safely."""
    if isinstance(value, str) and value.strip().startswith("["):
        value = ast.literal_eval(value)[0]
    return float(value)


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path.with_suffix(".png")


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.25)


def _scatter(ax, frame, title):
    actual, pred = frame.actual.to_numpy(), frame.prediction.to_numpy()
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    ax.scatter(actual, pred, s=28, alpha=.8, color="#1769aa")
    ax.plot([lo, hi], [lo, hi], "--", color="#c62828", label="Ideal prediction")
    _style(ax, title, "Actual normalized path-centre offset", "Predicted normalized path-centre offset")
    ax.legend(fontsize=8)


def _append_index(index_path, entries):
    existing = pd.read_csv(index_path) if index_path.exists() else pd.DataFrame()
    fields = ["figure_number", "figure_title", "research_section", "source_data", "sample_count", "figure_filename", "generation_status", "main_finding", "important_limitation"]
    old = set(existing.figure_filename.astype(str)) if len(existing) else set()
    add = pd.DataFrame([e for e in entries if e["figure_filename"] not in old])
    pd.concat([existing, add], ignore_index=True)[fields].to_csv(index_path, index=False)


def _entry(number, title, group, source, n, path, finding, limitation):
    return dict(figure_number=number, figure_title=title, research_section=group,
                source_data=source, sample_count=n, figure_filename=str(path),
                generation_status="generated", main_finding=finding,
                important_limitation=limitation)


def generate_recovery_figures(results_root, output_root=None):
    """Generate supplementary evidence-backed Phase B figures and return entries.

    ``results_root`` is the extracted original results directory.  The method
    raises rather than falling back to training if any required prediction table
    is absent.
    """
    results_root, output_root = Path(results_root), Path(output_root or ROOT / "research_figures")
    croot, droot, aroot = results_root / "classical_ml", results_root / "deep_learning", results_root / "anfis"
    reg = pd.read_csv(croot / "predictions_regression.csv")
    reg.prediction = reg.prediction.map(_prediction_number)
    rf = reg[(reg.model == "rf") & (reg.split == "test")].copy()
    if rf.empty:
        raise FileNotFoundError("Saved RF test predictions are required; recovery will not retrain it.")
    entries, n = [], len(rf)
    # 41--43: selected RF diagnostic views, all saved test predictions.
    residual = rf.prediction - rf.actual
    fig, ax = plt.subplots(figsize=(6.5, 4.5)); ax.scatter(rf.prediction, residual, color="#1769aa", s=28); ax.axhline(0, color="#c62828", ls="--")
    _style(ax, f"Classical RF residuals versus prediction (n={n}; oracle features)", "Predicted normalized offset", "Residual (prediction − actual)")
    p = _save(fig, output_root / "phase_B/classical_ml_regression/figure_41_rf_residuals_vs_prediction")
    entries.append(_entry(41, "Classical RF residuals versus predicted offset", "Classical ML", "classical_ml/predictions_regression.csv", n, p.relative_to(output_root), "Residuals identify offset regions with systematic error.", "RF uses oracle ground-truth-mask features and derived targets."))
    fig, ax = plt.subplots(figsize=(6.5, 4.5)); ax.scatter(abs(rf.actual), abs(residual), color="#1769aa", s=28)
    _style(ax, f"Classical RF absolute error by actual offset magnitude (n={n})", "|Actual normalized offset|", "Absolute prediction error")
    p = _save(fig, output_root / "phase_B/classical_ml_regression/figure_42_rf_absolute_error_by_offset_magnitude")
    entries.append(_entry(42, "Classical RF absolute error by actual offset magnitude", "Classical ML", "classical_ml/predictions_regression.csv", n, p.relative_to(output_root), "Error can be assessed at central and extreme derived offsets.", "RF uses oracle features; bins are not a physical-control validation."))
    direction = pd.cut(rf.actual, [-np.inf, -.2, .2, np.inf], labels=CLASS_ORDER)
    grouped = pd.DataFrame({"direction": direction, "absolute_error": abs(residual)}).groupby("direction", observed=False).absolute_error
    fig, ax = plt.subplots(figsize=(6.5, 4.5)); ax.bar(grouped.mean().index.astype(str), grouped.mean(), yerr=grouped.std().fillna(0), capsize=4, color=["#4c78a8", "#59a14f", "#e15759"])
    _style(ax, f"Classical RF error by derived navigation direction (n={n})", "Derived direction (threshold ±0.2)", "Mean absolute error")
    p = _save(fig, output_root / "phase_B/classical_ml_regression/figure_43_rf_error_by_derived_direction")
    entries.append(_entry(43, "Classical RF error by derived navigation direction", "Classical ML", "classical_ml/predictions_regression.csv", n, p.relative_to(output_root), "Mean RF error differs across derived target ranges.", "Directions are derived from continuous geometry targets; they are not recorded commands."))
    # Classification validation tables are comparisons; direct test prediction is separately recomputed.
    cm = pd.read_csv(croot / "classification_metrics.csv")
    selected = cm[cm.source.eq("oracle") & cm.feature_set.eq("all_non_leaking")]
    for number, column, label in [(44, "accuracy", "accuracy"), (45, "balanced_accuracy", "balanced accuracy")]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5)); ax.bar(selected.model, selected[column], color="#4c78a8"); ax.set_ylim(0, 1)
        _style(ax, f"Classical directional-classification validation {label}", "Model", label.title())
        p = _save(fig, output_root / f"phase_B/classical_ml_classification/figure_{number}_classification_{column}")
        entries.append(_entry(number, f"Classification validation {label} comparison", "Classical ML", "classical_ml/classification_metrics.csv", len(selected), p.relative_to(output_root), "Saved validation metrics compare directly trained classifiers.", "Validation metrics are not a multi-model held-out test comparison."))
    fig, ax = plt.subplots(figsize=(7, 4.8)); x=np.arange(len(selected)); w=.24
    for i, col in enumerate(["macro_precision", "macro_recall", "macro_f1"]): ax.bar(x+(i-1)*w, selected[col], width=w, label=col.replace("macro_", ""))
    ax.set_xticks(x, selected.model); ax.set_ylim(0,1); ax.legend(); _style(ax, "Classical directional-classification macro metrics", "Model", "Validation score")
    p = _save(fig, output_root / "phase_B/classical_ml_classification/figure_46_classification_macro_metrics")
    entries.append(_entry(46, "Classification macro precision, recall and F1", "Classical ML", "classical_ml/classification_metrics.csv", len(selected), p.relative_to(output_root), "Macro metrics show class-balanced validation behaviour.", "Only saved validation summary is available for multi-model comparison."))
    clf = pd.read_csv(croot / "predictions_classification.csv"); logistic = clf[clf.model.eq("logistic")]
    matrix = confusion_matrix(logistic.actual, logistic.prediction, labels=CLASS_ORDER, normalize="true")
    fig, ax = plt.subplots(figsize=(5.4, 4.7)); im=ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues"); plt.colorbar(im, ax=ax, label="Row-normalized proportion")
    ax.set_xticks(range(3), CLASS_ORDER); ax.set_yticks(range(3), CLASS_ORDER); ax.set_xlabel("Predicted direction"); ax.set_ylabel("Actual direction"); ax.set_title(f"Logistic directional classifier: normalized confusion matrix (n={len(logistic)})", weight="bold")
    for i in range(3):
        for j in range(3): ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center")
    p = _save(fig, output_root / "phase_B/classical_ml_classification/figure_47_logistic_normalized_confusion_matrix")
    entries.append(_entry(47, "Logistic classifier normalized confusion matrix", "Classical ML", "classical_ml/predictions_classification.csv", len(logistic), p.relative_to(output_root), "Rows show class-normalized direct-classifier predictions.", "Test/validation split is inferred only from saved prediction export; sample size is limited."))
    pr, re, f1, _ = precision_recall_fscore_support(logistic.actual, logistic.prediction, labels=CLASS_ORDER, zero_division=0)
    fig, ax=plt.subplots(figsize=(7,4.8)); x=np.arange(3); w=.24
    for i,(name,v) in enumerate(zip(["Precision","Recall","F1"],[pr,re,f1])): ax.bar(x+(i-1)*w,v,w,label=name)
    ax.set_xticks(x,CLASS_ORDER); ax.set_ylim(0,1); ax.legend(); _style(ax,"Logistic classifier per-class metrics from saved predictions","Direction class","Score")
    p=_save(fig, output_root / "phase_B/classical_ml_classification/figure_48_logistic_per_class_metrics")
    entries.append(_entry(48, "Logistic classifier per-class precision, recall and F1", "Classical ML", "classical_ml/predictions_classification.csv", len(logistic), p.relative_to(output_root), "Per-class scores are recomputed from saved direct classifier labels.", "Small class counts make estimates unstable."))
    # Deep prediction diagnostics: one actual/predicted and one residual distribution per image model.
    deep_frames={}
    for model, number in [("custom_cnn",49),("mobilenetv3",51),("resnet18",53)]:
        frame=pd.read_csv(droot/f"predictions_{model}.csv"); frame.prediction=frame.prediction.map(_prediction_number); deep_frames[model]=frame
        fig,ax=plt.subplots(figsize=(6.5,4.5)); _scatter(ax,frame,f"{model.replace('_',' ').title()} actual versus predicted offsets (n={len(frame)})")
        p=_save(fig,output_root/f"phase_B/{model}/figure_{number}_{model}_actual_vs_predicted")
        entries.append(_entry(number, f"{model} actual versus predicted offsets", "CNN and transfer learning", f"deep_learning/predictions_{model}.csv", len(frame), p.relative_to(output_root), "Saved image-model predictions are compared with their derived targets.", "Image model evaluates geometry-derived labels, not robot controls."))
        fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.hist(frame.prediction-frame.actual,bins=min(12,len(frame)),color="#59a14f",edgecolor="white"); ax.axvline(0,color="#c62828",ls="--")
        _style(ax,f"{model.replace('_',' ').title()} residual distribution (n={len(frame)})","Residual (prediction − actual)","Test samples")
        p=_save(fig,output_root/f"phase_B/{model}/figure_{number+1}_{model}_residual_distribution")
        entries.append(_entry(number+1, f"{model} residual distribution", "CNN and transfer learning", f"deep_learning/predictions_{model}.csv", len(frame), p.relative_to(output_root), "The distribution shows signed error around the derived target.", "No repeated runs or uncertainty estimates are available."))
    metrics=[]
    for model, frame in deep_frames.items(): metrics.append((model, r2_score(frame.actual,frame.prediction)))
    fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.bar([x[0] for x in metrics],[x[1] for x in metrics],color="#4c78a8"); _style(ax,"Deep-learning test R² comparison from saved predictions","Image regression model","Test R²")
    p=_save(fig,output_root/"phase_B/overall_model_comparison/figure_55_deep_learning_r2_comparison")
    entries.append(_entry(55,"Deep-learning test R² comparison","Overall model comparison","deep_learning/predictions_*.csv",len(deep_frames["custom_cnn"]),p.relative_to(output_root),"R² values are recomputed on the shared saved test IDs.","These models share image inputs, but results are from one run."))
    paired=rf.merge(deep_frames["mobilenetv3"][["image_id","actual","prediction"]],on="image_id",suffixes=("_rf","_mobile")); diff=abs(paired.prediction_rf-paired.actual_rf)-abs(paired.prediction_mobile-paired.actual_mobile)
    fig,ax=plt.subplots(figsize=(6.5,4.5)); ax.hist(diff,bins=min(12,len(diff)),color="#f28e2b",edgecolor="white"); ax.axvline(0,color="black",ls="--")
    _style(ax,"Paired absolute-error difference: RF minus MobileNetV3","|RF error| − |MobileNetV3 error|","Common test images")
    p=_save(fig,output_root/"phase_B/overall_model_comparison/figure_56_paired_rf_mobilenet_error_difference")
    entries.append(_entry(56,"Paired RF versus MobileNetV3 absolute-error difference","Overall model comparison","classical_ml and deep_learning saved predictions",len(paired),p.relative_to(output_root),"Each bar compares errors on a common saved image ID.","RF uses oracle features while MobileNetV3 uses images; this is diagnostic, not a deployability ranking."))
    _append_index(output_root / "figure_index.csv", entries)
    return entries


def write_recovery_reports(entries, output_root=None):
    output_root=Path(output_root or ROOT / "research_figures")
    report=ROOT / "reports"
    report.mkdir(exist_ok=True)
    lines=["# Phase B supplementary figure explanations", "", "These figures are generated only from saved Phase B exports. Classical RF and ANFIS use oracle ground-truth-mask features; image models use original images. All targets are geometry-derived navigation targets, not recorded steering commands.", ""]
    for e in entries:
        lines += [f"## Figure {e['figure_number']} — {e['figure_title']}", "", f"**What it shows.** {e['main_finding']} The axes and sample count are labelled in the figure. Source: `{e['source_data']}`.", "", f"**Interpretation and limitation.** {e['important_limitation']}", ""]
    (report / "PHASE_B_SUPPLEMENTARY_FIGURES_EXPLANATION.md").write_text("\n".join(lines))
    inventory=["# Project figure inventory", "", "This inventory is a recovery audit. PNG is the canonical counted format; paired PDFs are export representations, not separate figures.", "", "| Canonical figure set | Phase | Scientific status | Duplicate rule |", "|---|---|---|---|", "| `research_figures/01_dataset` | Dataset audit | useful | paired PDF excluded |", "| `research_figures/02_segmentation_navigation` | Phase A targets | useful where generated | unavailable segmentation slots excluded |", "| `research_figures/03_classical_ml`, `04_anfis`, `05_deep_learning`, `06_model_comparison` | Phase B original package | useful | paired PDF excluded |", "| `research_figures/phase_B` | Phase B supplementary recovery | useful | paired PDF excluded |", "", "No existing valid figure was deleted. Segmentation figures cannot be recovered because stored segmentation status is `not_trained` and no predictions/logs are present."]
    (report / "PROJECT_FIGURE_INVENTORY.md").write_text("\n".join(inventory)+"\n")
    count=len(list(output_root.rglob("*.png")))
    plan=f"# Project-wide figure plan\n\nCanonical PNG figures currently indexed/generated: **{count}**. The research target is 100, leaving a documented shortfall of **{max(0,100-count)}**; it must be filled only by additional supported analyses or new experiments, not format duplicates. Phase B supplementary recovery contributes {len(entries)} new figures.\n"
    (report / "PROJECT_WIDE_FIGURE_PLAN.md").write_text(plan)
    return count


def export_archives(results_root, output_root=None):
    """Create portable archives, excluding raw datasets; preserve original exports."""
    output_root=Path(output_root or ROOT / "research_figures"); results_root=Path(results_root)
    targets=[(ROOT/"cassava_navigation_research_figures.zip", [output_root, ROOT/"reports/RESEARCH_FIGURES_EXPLANATION.md", ROOT/"reports/PHASE_B_SUPPLEMENTARY_FIGURES_EXPLANATION.md", ROOT/"reports/PROJECT_FIGURE_INVENTORY.md", ROOT/"reports/PROJECT_WIDE_FIGURE_PLAN.md"]), (ROOT/"cassava_navigation_phase_B_results_complete.zip", [results_root, output_root/"phase_B", ROOT/"reports/PROJECT_FIGURE_INVENTORY.md", ROOT/"reports/PROJECT_WIDE_FIGURE_PLAN.md"])]
    for archive, sources in targets:
        with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
            for source in sources:
                if source.exists():
                    for p in ([source] if source.is_file() else source.rglob("*")):
                        if p.is_file(): z.write(p,p.relative_to(ROOT))
