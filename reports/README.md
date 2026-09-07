# Reports and Results Index

This index identifies source-of-truth outputs and distinguishes completed audit artifacts from pending executed Phase A artifacts. See the root `README.md` for methodology and `PROJECT_STATUS.md` for milestones.

## Dataset Audit

- `dataset_audit_report.md` — Phase 1 structure, format, split, image, annotation, quality, and suitability findings.
- `dataset_summary.csv` — compact counts and detected format.
- `class_distribution.csv` — polygon/image counts with confirmed class names.
- `image_properties.csv` — dimensions, format, channels, sizes, and hashes.
- `annotation_statistics.csv` — annotation-level geometry/statistics.
- `data_quality_issues.csv` — original-tree missing-label and duplicate-artifact findings.

Related `figures/` outputs include distributions, imbalance, object counts, polygon areas, and annotated samples.

## Annotation and Class Validation

- `annotation_quality_report.md` — Phase 2 class status, validity, warnings, leakage check, and readiness.
- `class_semantics_review.csv` — confirmed mapping and human-confirmation status.
- `polygon_quality_checks.csv` — one row per parsed polygon with validity/warning flags.
- `manual_review_queue.csv` — 278 images selected for review.
- `split_consistency.csv` — split and class consistency summaries.

Class/multiclass/edge-case contact sheets are under `figures/`. The 198 small and 18 large polygons remain warnings.

## Phase A Notebook Validation

- `colab_notebook_validation.md` — schema, compilation, security, root-discovery, and synthetic helper checks.
- `environment_local_freeze.txt` — local helper-preparation dependencies, not the GPU environment.

## Segmentation Results

- `segmentation_baseline_report.md` — currently documents local training unavailability and the GPU protocol.
- `results/segmentation_baseline/metrics.csv` — `status=not_trained`; unavailable fields are blank.
- `per_class_metrics.csv` — expected split/class schema without trained values.
- `inference_results.csv` and `failure_cases.csv` — empty schemas.
- `environment.json` — local detection only, not executed Colab evidence.
- segmentation figure/model READMEs explain why plots/checkpoints are absent.

Expected after import: actual validation/test metrics, raw metrics, histories/logs, available PR/F1/confusion figures, test predictions, representative/failure panels, `best.pt`, and `last.pt`.

## Feature Results

- `data/processed/features_ground_truth.csv` — 802 oracle-mask rows.
- `data/processed/image_manifest.csv` — canonical usable IDs, paths, splits, and hashes.
- `data/processed/parsed_annotations.csv` — reusable parsed polygon table.

Awaiting executed-output import: `features_predicted_masks.csv`, `feature_dictionary.csv`, feature summaries/missingness/correlations/distributions/leakage checks, and `feature_quality_report.md`.

## Navigation-Target Results

No executed target CSV is present. `configs/navigation_targets.yaml` holds provisional local defaults; the executed workflow should record training-derived thresholds.

Awaiting import: `navigation_targets.csv`, counts/distributions, 30+ diagnostic overlays, `navigation_target_examples.png`, and `navigation_target_report.md`.

These targets are geometry-derived research labels, not recorded motor commands.

## Model Checkpoints

`models/segmentation_baseline/` contains only a status README. The external `best.pt` and `last.pt` have not been imported. If each verified file is reasonably small and within GitHub's limit, `.gitignore` permits those two canonical checkpoints. Otherwise, document a persistent storage URL and SHA-256 checksum.

## Source-of-Truth Rule

Do not replace missing fields with approximate reported results. A Phase A result becomes repository-verified only after reconciling an imported executed CSV/report with the checkpoint, environment, configuration, IDs, and splits.
