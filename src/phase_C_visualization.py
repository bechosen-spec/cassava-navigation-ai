"""Publication-oriented Phase C plotting helpers."""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def metric_bars(metrics, metric, out, title, intervals=None):
    labels = list(metrics.index); vals = metrics[metric].to_numpy()
    fig, ax = plt.subplots(figsize=(8.3, 4.7))
    err = None
    if intervals:
        err = np.array([[metrics.loc[m, f'{metric}_ci_low'] for m in labels],
                        [metrics.loc[m, f'{metric}_ci_high'] for m in labels]])
        err = np.vstack((vals - err[0], err[1] - vals))
    ax.bar(labels, vals, yerr=err, color='#2b6cb0', capsize=4)
    ax.set_title(title); ax.set_ylabel(metric); ax.tick_params(axis='x', rotation=25)
    save(fig, out)


def error_box(errors, out):
    fig, ax = plt.subplots(figsize=(8.3, 4.7))
    ax.boxplot([errors[k] for k in errors], tick_labels=list(errors), showmeans=True)
    ax.set_ylabel('Absolute error (normalised offset)'); ax.set_title('Common-test absolute-error distributions')
    ax.tick_params(axis='x', rotation=25); save(fig, out)


def actual_predicted(models, out):
    names = list(models)
    fig, axes = plt.subplots(1, len(names), figsize=(4.3*len(names), 3.8), sharex=True, sharey=True)
    if len(names) == 1: axes = [axes]
    for ax, name in zip(axes, names):
        frame = models[name]
        ax.scatter(frame.actual, frame.prediction, s=24, alpha=.75, color='#2b6cb0')
        ax.plot([-1, 1], [-1, 1], '--', color='black', linewidth=1)
        ax.set_title(name); ax.set_xlabel('Actual offset')
    axes[0].set_ylabel('Predicted offset')
    save(fig, out)


def residuals(models, out):
    fig, ax = plt.subplots(figsize=(8.3, 4.7))
    for name, frame in models.items():
        ax.scatter(frame.actual, frame.prediction-frame.actual, s=18, alpha=.6, label=name)
    ax.axhline(0, color='black', linewidth=1); ax.set_xlabel('Actual offset'); ax.set_ylabel('Residual (prediction - actual)')
    ax.legend(ncol=2); ax.set_title('Residuals on the aligned common test set'); save(fig, out)
