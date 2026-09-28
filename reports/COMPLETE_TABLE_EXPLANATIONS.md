# Complete Table Explanations

## Regression metrics verified
**Source:** `phase_C_results/tables/regression_metrics_verified.csv`. **Purpose:** Independent recomputation from per-image predictions. **Interpretation:** values trace to frozen Phase B exports and Phase C recomputation. **Limitation:** numerical precision does not remove the experimental limitations documented in the report.

## Corrected classification metrics
**Source:** `phase_C_results/tables/classification_metrics_corrected.csv`. **Purpose:** Hard-label overall metrics; no score-based curves. **Interpretation:** values trace to frozen Phase B exports and Phase C recomputation. **Limitation:** numerical precision does not remove the experimental limitations documented in the report.

## Common test manifest
**Source:** `phase_C_results/tables/common_test_manifest.csv`. **Purpose:** Confirms paired 42-image comparisons. **Interpretation:** values trace to frozen Phase B exports and Phase C recomputation. **Limitation:** numerical precision does not remove the experimental limitations documented in the report.

## Statistical comparisons
**Source:** `phase_C_results/statistics/statistical_comparisons.csv`. **Purpose:** Wilcoxon tests with Holm correction and bootstrap CIs. **Interpretation:** values trace to frozen Phase B exports and Phase C recomputation. **Limitation:** numerical precision does not remove the experimental limitations documented in the report.
