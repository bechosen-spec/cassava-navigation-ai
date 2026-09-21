# Research figures: explanations

Generated evidence-backed figures: **33** of 40 requested. Entries marked `requires additional evaluation` are deliberately not fabricated.

## Figure 1 — Dataset split distribution

**Status:** generated. **Source:** metadata/split_summary.csv. **Sample count:** 802.

**What it shows and interpretation:** The prepared dataset contains 602 train, 120 validation, and 80 test images.

**Limitation:** Counts are image-level; target eligibility is lower.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 2 — Semantic class distribution

**Status:** generated. **Source:** reports/class_distribution.csv. **Sample count:** 28655.

**What it shows and interpretation:** Ridge annotations are the most numerous class.

**Limitation:** Annotation instances are not independent image samples.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 3 — Image resolution distribution

**Status:** generated. **Source:** reports/image_properties.csv. **Sample count:** 803.

**What it shows and interpretation:** All audited images have the recorded image dimensions.

**Limitation:** The audit includes an extra checkpoint image noted in data-quality outputs.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 4 — Polygon area distribution by class

**Status:** generated. **Source:** reports/polygon_quality_checks.csv. **Sample count:** 28655.

**What it shows and interpretation:** Polygon areas vary substantially across semantic classes.

**Limitation:** Areas are normalized image fractions and may overlap spatially.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 5 — Annotation density

**Status:** generated. **Source:** reports/annotation_statistics.csv. **Sample count:** 803.

**What it shows and interpretation:** Annotation density is heterogeneous across images.

**Limitation:** One audit-only checkpoint image may be represented.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 6 — Annotation quality analysis

**Status:** generated. **Source:** reports/polygon_quality_checks.csv. **Sample count:** 28655.

**What it shows and interpretation:** 28655 polygon rows satisfy row and area checks; 216 are flagged suspicious.

**Limitation:** Flagged/suspicious can overlap valid rows.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 7 — Representative annotated dataset samples

**Status:** generated. **Source:** figures/annotated_samples.png. **Sample count:** not available.

**What it shows and interpretation:** Panel is a previously generated dataset-audit visualization.

**Limitation:** Recovered image; its original selection protocol is documented only in Phase A.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 8 — Segmentation training curves

**Status:** requires additional evaluation. **Source:** Not available in supplied executed outputs. **Sample count:** not available.

**What it shows and interpretation:** No claim is made because the required stored evidence is unavailable.

**Limitation:** Epoch-level segmentation training logs are absent; baseline metrics say not_trained.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 9 — Segmentation precision and recall

**Status:** requires additional evaluation. **Source:** Not available in supplied executed outputs. **Sample count:** not available.

**What it shows and interpretation:** No claim is made because the required stored evidence is unavailable.

**Limitation:** Segmentation baseline is explicitly not_trained.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 10 — Segmentation mAP comparison

**Status:** requires additional evaluation. **Source:** Not available in supplied executed outputs. **Sample count:** not available.

**What it shows and interpretation:** No claim is made because the required stored evidence is unavailable.

**Limitation:** Segmentation baseline is explicitly not_trained.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 11 — Ground truth versus predicted masks

**Status:** requires additional evaluation. **Source:** Not available in supplied executed outputs. **Sample count:** not available.

**What it shows and interpretation:** No claim is made because the required stored evidence is unavailable.

**Limitation:** No segmentation predictions or masks were supplied.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 12 — Segmentation failure cases

**Status:** requires additional evaluation. **Source:** Not available in supplied executed outputs. **Sample count:** not available.

**What it shows and interpretation:** No claim is made because the required stored evidence is unavailable.

**Limitation:** No segmentation inference/failure records were supplied.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 13 — Directional navigation-target distribution

**Status:** generated. **Source:** tabular/classification_targets.csv. **Sample count:** 154.

**What it shows and interpretation:** Only left, forward, and right occur among eligible classification targets; labels are derived navigation targets.

**Limitation:** No physical STOP command is established; stop_or_uncertain has no eligible records.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 14 — Continuous navigation-target distribution

**Status:** generated. **Source:** tabular/regression_targets.csv. **Sample count:** 395.

**What it shows and interpretation:** Offsets cover the normalized path-centre range among 395 eligible samples.

**Limitation:** These are geometry-derived labels, not measured steering commands.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 15 — Classical regression validation MAE comparison

**Status:** generated. **Source:** classical_ml/regression_metrics.csv. **Sample count:** 56.

**What it shows and interpretation:** All models are compared on the validation-only model-selection output.

**Limitation:** A multi-model held-out test comparison is unavailable; all models use oracle ground-truth-mask features.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 16 — Classical regression validation RMSE comparison

**Status:** generated. **Source:** classical_ml/regression_metrics.csv. **Sample count:** 56.

**What it shows and interpretation:** All models are compared on the validation-only model-selection output.

**Limitation:** A multi-model held-out test comparison is unavailable; all models use oracle ground-truth-mask features.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 17 — Classical regression validation R² comparison

**Status:** generated. **Source:** classical_ml/regression_metrics.csv. **Sample count:** 56.

**What it shows and interpretation:** All models are compared on the validation-only model-selection output.

**Limitation:** A multi-model held-out test comparison is unavailable; all models use oracle ground-truth-mask features.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 18 — Classical RF actual versus predicted offset

**Status:** generated. **Source:** classical_ml/predictions_regression.csv. **Sample count:** 42.

**What it shows and interpretation:** The selected random forest is evaluated on saved oracle-feature test predictions.

**Limitation:** Predictions are derived targets and model uses oracle features.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 19 — Classical RF residual distribution

**Status:** generated. **Source:** classical_ml/predictions_regression.csv. **Sample count:** 42.

**What it shows and interpretation:** Residual spread reveals errors relative to geometry-derived offset targets.

**Limitation:** Oracle feature source prevents deployment interpretation.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 20 — Directional classification confusion matrix

**Status:** generated. **Source:** validated_results/classification/confusion_matrix.csv. **Sample count:** 16.

**What it shows and interpretation:** Metrics are recomputed with class order left, forward, right.

**Limitation:** Only 16 test samples; class estimates are unstable.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 21 — RF permutation feature importance

**Status:** generated. **Source:** classical_ml/feature_importance.csv. **Sample count:** 42.

**What it shows and interpretation:** Permutation importance is from the selected oracle random forest.

**Limitation:** Importance is model- and subset-specific; negative values are retained.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 22 — ANFIS recorded validation-loss plot

**Status:** generated. **Source:** figures/anfis/validation_loss.png. **Sample count:** 42.

**What it shows and interpretation:** Recovered original ANFIS visualization.

**Limitation:** Underlying epoch values are not exported, so the figure is reproduced as an image.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 23 — ANFIS membership functions

**Status:** requires additional evaluation. **Source:** Not available in supplied executed outputs. **Sample count:** not available.

**What it shows and interpretation:** No claim is made because the required stored evidence is unavailable.

**Limitation:** Checkpoint serialization lacks independently documented membership-function extraction metadata.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 24 — ANFIS actual versus predicted offset

**Status:** generated. **Source:** anfis/predictions.csv. **Sample count:** 42.

**What it shows and interpretation:** Plot uses saved ANFIS test predictions.

**Limitation:** The ANFIS uses oracle features and only a 42-sample test set.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 25 — ANFIS residual distribution

**Status:** generated. **Source:** anfis/predictions.csv. **Sample count:** 42.

**What it shows and interpretation:** Plot uses saved ANFIS test predictions.

**Limitation:** The ANFIS uses oracle features and only a 42-sample test set.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 26 — ANFIS absolute error by sample

**Status:** generated. **Source:** anfis/predictions.csv. **Sample count:** 42.

**What it shows and interpretation:** Plot uses saved ANFIS test predictions.

**Limitation:** The ANFIS uses oracle features and only a 42-sample test set.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 27 — ANFIS response surface

**Status:** requires additional evaluation. **Source:** Not available in supplied executed outputs. **Sample count:** not available.

**What it shows and interpretation:** No claim is made because the required stored evidence is unavailable.

**Limitation:** Requires checkpoint recovery evaluation with documented feature scaling and reference values.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 28 — custom_cnn validation MAE across epochs

**Status:** generated. **Source:** deep_learning/training_history_custom_cnn.csv. **Sample count:** 20.

**What it shows and interpretation:** The saved history records validation MAE across 20 epochs.

**Limitation:** Training/validation loss was not exported; this is not a loss curve.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 29 — mobilenetv3 validation MAE across epochs

**Status:** generated. **Source:** deep_learning/training_history_mobilenetv3.csv. **Sample count:** 20.

**What it shows and interpretation:** The saved history records validation MAE across 20 epochs.

**Limitation:** Training/validation loss was not exported; this is not a loss curve.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 30 — resnet18 validation MAE across epochs

**Status:** generated. **Source:** deep_learning/training_history_resnet18.csv. **Sample count:** 20.

**What it shows and interpretation:** The saved history records validation MAE across 20 epochs.

**Limitation:** Training/validation loss was not exported; this is not a loss curve.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 31 — Deep-learning MAE comparison

**Status:** generated. **Source:** validated_results/regression/recomputed_test_metrics.csv. **Sample count:** 42.

**What it shows and interpretation:** All three image models use predictions for the same 42 saved test IDs.

**Limitation:** Scores concern provisional geometry-derived targets.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 32 — Deep-learning RMSE comparison

**Status:** generated. **Source:** validated_results/regression/recomputed_test_metrics.csv. **Sample count:** 42.

**What it shows and interpretation:** All three image models use predictions for the same 42 saved test IDs.

**Limitation:** Scores concern provisional geometry-derived targets.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 33 — Deep-learning actual versus predicted offsets

**Status:** generated. **Source:** deep_learning/predictions_*.csv. **Sample count:** 42.

**What it shows and interpretation:** Model predictions are compared on common saved test image IDs.

**Limitation:** No confidence intervals or repeated runs are available.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 34 — ResNet18 largest-error real test images

**Status:** generated. **Source:** deep_learning/predictions_resnet18.csv plus phase_B_training_dataset/images. **Sample count:** 6.

**What it shows and interpretation:** Panel shows the six highest absolute-error saved ResNet18 test predictions.

**Limitation:** Errors are against derived offsets; it is not a segmentation-error analysis.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 35 — Overall test MAE comparison

**Status:** generated. **Source:** validated_results/regression/recomputed_test_metrics.csv. **Sample count:** 42.

**What it shows and interpretation:** Metrics are recomputed from available saved test predictions.

**Limitation:** Classical RF and ANFIS are oracle-feature models; deep models use images.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 36 — Overall test RMSE comparison

**Status:** generated. **Source:** validated_results/regression/recomputed_test_metrics.csv. **Sample count:** 42.

**What it shows and interpretation:** Metrics are recomputed from available saved test predictions.

**Limitation:** Classical RF and ANFIS are oracle-feature models; deep models use images.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 37 — Overall test R² comparison

**Status:** generated. **Source:** validated_results/regression/recomputed_test_metrics.csv. **Sample count:** 42.

**What it shows and interpretation:** Metrics are recomputed from available saved test predictions.

**Limitation:** Classical RF and ANFIS are oracle-feature models; deep models use images.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 38 — Deep-model size versus prediction error

**Status:** generated. **Source:** deep_learning/model_metrics.csv. **Sample count:** 42.

**What it shows and interpretation:** The deep checkpoints show a size/error trade-off.

**Limitation:** Only deep model sizes are consistently stored; serialization details matter.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 39 — Deep-model latency versus prediction error

**Status:** generated. **Source:** deep_learning/model_metrics.csv. **Sample count:** 42.

**What it shows and interpretation:** Saved deep-model timing values can be inspected alongside MAE.

**Limitation:** Timing hardware/batch semantics require the notebook environment; no cross-family claim is made.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.

## Figure 40 — Absolute error distribution across models

**Status:** generated. **Source:** saved regression prediction CSVs. **Sample count:** 42.

**What it shows and interpretation:** All available regression models are aligned to common saved test image IDs.

**Limitation:** RF/ANFIS use oracle features whereas deep models use images.

**Research objective:** documents dataset quality, derived navigation targets, or Phase B model behavior without treating derived targets as physical robot-control measurements.
