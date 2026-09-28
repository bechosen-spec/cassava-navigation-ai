"""Reproducible, dependency-light statistics for Phase C."""
from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np
from scipy.stats import wilcoxon


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - actual
    absolute = np.abs(residual)
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "MAE": float(np.mean(absolute)),
        "RMSE": float(math.sqrt(np.mean(residual ** 2))),
        "R2": float(1 - ss_res / ss_tot) if ss_tot else float("nan"),
        "median_absolute_error": float(np.median(absolute)),
        "bias": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=1)),
        "max_absolute_error": float(np.max(absolute)),
    }


def bootstrap_ci(actual: np.ndarray, prediction: np.ndarray, repetitions: int = 5000,
                 seed: int = 20260928) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap intervals from resampled image-level pairs."""
    rng = np.random.default_rng(seed)
    n = len(actual)
    values = {"MAE": [], "RMSE": [], "R2": []}
    for _ in range(repetitions):
        take = rng.integers(0, n, n)
        metrics = regression_metrics(actual[take], prediction[take])
        for name in values:
            values[name].append(metrics[name])
    return {name: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for name, v in values.items()}


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = list(p_values)
    order = sorted(range(len(p)), key=lambda i: p[i])
    adjusted = [0.0] * len(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def paired_tests(errors: dict[str, np.ndarray], repetitions: int = 10000,
                 seed: int = 20260928) -> list[dict[str, object]]:
    """Wilcoxon tests and percentile CIs for pairwise absolute-error differences."""
    rng = np.random.default_rng(seed)
    rows = []
    for left, right in itertools.combinations(sorted(errors), 2):
        delta = errors[left] - errors[right]  # positive means right has lower MAE
        try:
            statistic, p_value = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            statistic, p_value = float("nan"), 1.0
        means = np.empty(repetitions)
        n = len(delta)
        for j in range(repetitions):
            means[j] = np.mean(delta[rng.integers(0, n, n)])
        nonzero = delta[delta != 0]
        rank_biserial = (np.sum(nonzero > 0) - np.sum(nonzero < 0)) / len(nonzero) if len(nonzero) else 0.0
        rows.append({
            "model_a": left, "model_b": right, "common_n": n,
            "metric": "absolute_error", "difference_a_minus_b": float(np.mean(delta)),
            "ci95_low": float(np.percentile(means, 2.5)), "ci95_high": float(np.percentile(means, 97.5)),
            "wilcoxon_statistic": float(statistic), "raw_p_value": float(p_value),
            "effect_size_rank_biserial": float(rank_biserial),
        })
    adjusted = holm_adjust([row["raw_p_value"] for row in rows])
    for row, p in zip(rows, adjusted):
        row["holm_adjusted_p_value"] = p
        row["interpretation"] = "evidence of different paired absolute errors" if p < 0.05 else "no adjusted evidence of a difference"
    return rows
