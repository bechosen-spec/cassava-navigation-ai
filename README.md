# Vision-Based Navigation and Control for Autonomous Cassava Farm Robots

## Project Overview

This repository contains the AI and machine-learning research workflow for **Vision-Based Navigation and Control Techniques for Autonomous Cassava Farm Robots**. It studies how cassava-field images can support perception and navigation prediction through segmentation, engineered features, ANFIS, classical machine learning, custom CNNs, and transfer learning.

The robotics/control team remains responsible for physical control laws, motors, hardware, safety validation, and controller integration. Nothing currently in this repository is a validated motor controller or should be connected directly to a robot.

## Research Aim

The research compares interpretable feature-based navigation prediction, ANFIS, and end-to-end image models using a custom cassava-farm dataset. The eventual comparison will consider predictive performance, computational cost, latency, model size, interpretability, and possible future embedded deployment.

## Research Objectives

1. Model an ANFIS controller from interpretable image features for navigation research.
2. Model deep-learning networks for vision-based navigation prediction.
3. Develop and validate a custom cassava-farm perception dataset.
4. Compare classical ML, ANFIS, custom CNN, and transfer-learning approaches under one protocol.
5. Produce reproducible statistical analyses and paper-ready results.

## AI/ML Scope

The AI/ML scope includes dataset and annotation validation, polygon segmentation, scene understanding, feature extraction, provisional target derivation, classical regression/classification, ANFIS, image-based deep learning, and comparative statistics. Physical actuation and final control integration are outside the present scope.

## Dataset

### Dataset Structure

The source archive is a Roboflow-style YOLO polygon dataset:

```text
data/
├── images/{train,valid,test}/
└── labels/{train,valid,test}/
```

Analyses exclude `.ipynb_checkpoints`, `.DS_Store`, `__MACOSX`, and other hidden/system artifacts without modifying the source data.

### Dataset Statistics

| Item | Verified value |
|---|---:|
| Images in original tree | 803 |
| Label files | 802 |
| Original train split | 603 images / 602 labels |
| Usable train samples after artifact filtering | 602 |
| Validation split | 120 images / 120 labels |
| Test split | 80 images / 80 labels |
| Usable Phase A preparation set | 802 images |
| Semantic classes | 3 |

The extra training image is a duplicate under `.ipynb_checkpoints` without a corresponding label. It is excluded as a system artifact. The 602/120/80 usable split membership is preserved; samples are not reassigned.

### Semantic Classes

The project owner human-confirmed this authoritative mapping:

| ID | Meaning |
|---:|---|
| 0 | `path` |
| 1 | `cassava_leaves` |
| 2 | `ridge` |

The canonical source is [`configs/classes.yaml`](configs/classes.yaml). Do not change the mapping without explicit domain-owner instruction.

### Annotation Format

Each label row is `class_id x1 y1 ... xn yn`, with image-normalized polygon coordinates. The audit found 28,655 structurally valid polygon rows.

### Annotation Quality

| Finding | Count |
|---|---:|
| Valid polygons | 28,655 |
| Suspicious polygon warnings | 216 |
| Extremely small polygons | 198 |
| Extremely large polygons | 18 |
| Images in manual-review queue | 278 |
| Exact cross-split duplicate hashes | 0 |

The 216 size findings are review warnings and are retained unless technically invalid. Exact-hash checks do not prove independence of adjacent video frames, field sessions, plants, or locations.

## Research Workflow

```text
Phase 1: Dataset audit
    ↓
Phase 2: Class confirmation and annotation-quality validation
    ↓
Phase A: Segmentation → mask predictions → features → provisional targets
    ↓
Phase B: Classical ML + ANFIS + custom CNN/transfer learning
    ↓
Phase C: Comparative statistics + paper-ready results
```

Training data fits models and preprocessing. Validation data supports selection and early stopping. Test data is used once for final evaluation of frozen choices; it must never select features, thresholds, hyperparameters, or checkpoints.

## Current Project Status

### Completed — Phase 1: Dataset Audit

Phase 1 inspected the archive/folder structure, identified YOLO polygons, counted and matched image/label files, measured image and polygon properties, checked missing/corrupt files and duplicate hashes, and created reports and figures.

Principal artifacts:

- [`src/dataset_audit.py`](src/dataset_audit.py)
- [`notebooks/01_dataset_audit.ipynb`](notebooks/01_dataset_audit.ipynb)
- [`reports/dataset_audit_report.md`](reports/dataset_audit_report.md)
- dataset/class/image/annotation/quality CSVs and audit figures.

### Completed — Phase 2: Annotation Validation

Phase 2 added polygon structural checks, a manual-review queue, split consistency, class contact sheets, and artifact filtering. Commit `6e0e661` records the milestone. Later confirmation commits replace the historical lack of metadata with the human-confirmed class names.

Principal artifacts:

- [`src/annotation_validation.py`](src/annotation_validation.py)
- [`notebooks/02_annotation_validation.ipynb`](notebooks/02_annotation_validation.ipynb)
- [`configs/classes.yaml`](configs/classes.yaml)
- [`reports/annotation_quality_report.md`](reports/annotation_quality_report.md)
- polygon checks, review queue, split summary, and contact sheets.

### Phase A: Segmentation, Features, and Navigation Targets

The complete self-contained GPU workflow is implemented in [`notebooks/03_phase_A_colab.ipynb`](notebooks/03_phase_A_colab.ipynb). It validates an uploaded ZIP, stages a clean training view, selects one compatible nano YOLO segmentation checkpoint, trains/evaluates it, generates predictions, extracts features, derives training-calibrated provisional targets, creates diagnostics/reports, and packages the results.

This checkout contains the validated 32-cell Colab workflow, reusable Phase A modules/configurations, an 802-row ground-truth-mask feature table, the image manifest, and parsed annotations. Its local segmentation CSVs explicitly say `status=not_trained`.

An external Colab run has been reported as completed, but its results package has **not been imported into this checkout**. There is no `best.pt`, `last.pt`, `features_predicted_masks.csv`, `navigation_targets.csv`, executed Phase A notebook, or trained YOLO26 metrics CSV here. Consequently, approximate reported GPU metrics and target counts are not promoted to verified repository results. Importing and validating those artifacts is the immediate prerequisite to Phase B.

## Phase A Results

### Segmentation Model

Repository-verified trained architecture: **N/A — executed artifacts are absent**. The committed notebook records its actual selected checkpoint at runtime. The local status CSV's planned `yolo11n-seg.pt` is not evidence of a trained model.

### Validation Results

Mask precision, recall, mAP@0.5, and mAP@0.5:0.95 are **N/A in this checkout**. [`results/segmentation_baseline/metrics.csv`](results/segmentation_baseline/metrics.csv) has `status=not_trained` and deliberately blank metric fields.

### Test Results

Test mask metrics are **N/A in this checkout**. No approximate number is treated as source-of-truth without the executed CSV.

### Per-Class Results

Per-class results for `path`, `cassava_leaves`, and `ridge` are **N/A in this checkout**. Only the empty result schema exists locally.

### Runtime and Model Size

Best epoch, T4 latency/FPS, parameters, and checkpoint size are **N/A in this checkout**. They must be reconciled from the imported metrics, environment report, and checkpoint.

## Feature Extraction

[`src/features/feature_extraction.py`](src/features/feature_extraction.py) implements:

- polygon counts, union-mask area/ratio, polygon areas, centroid, bounds, aspect ratio, perimeter, and extrema per class;
- path centre/offset, widths, boundaries, continuity, and missing-path indicators;
- cassava coverage, left/right imbalance, obstruction, and density;
- ridge coverage/distribution, orientation, path proximity, overlap, and geometric relationship;
- RGB/HSV summaries and histograms, Excess Green/Red, normalized green-red difference;
- GLCM, entropy, LBP, edge, contour, and line-orientation descriptors;
- explicit missing flags rather than silent zero imputation.

Current artifacts:

- `features_ground_truth.csv`: present, 802 rows and 412 total columns; oracle-mask scenario;
- `image_manifest.csv`: present, 802 usable image IDs/splits/hashes;
- `features_predicted_masks.csv`: awaiting executed-output import;
- `feature_dictionary.csv` and feature-quality reports: awaiting executed-output import.

## Navigation-Target Generation

The source dataset has no recorded steering angle, motor command, or operator navigation label. The primary provisional target is:

```text
normalized_offset = (path_center_x - image_center_x) / (image_width / 2)
```

Negative means the path centre is image-left, zero is image-centred, and positive is image-right. This is image geometry, not a calibrated controller convention.

Secondary provisional classes are `left`, `forward`, `right`, and `stop_or_uncertain`. Thresholds and quality rules belong in `configs/navigation_targets.yaml`; an executed workflow fits them from training distributions. The committed file contains initial provisional defaults. Target counts remain unverified locally because `navigation_targets.csv` is absent.

## Important Methodological Notes

### Derived Targets

Navigation labels are geometry-derived research labels, not manually recorded commands or controller ground truth. The control team must validate their direction, camera geometry, uncertainty rules, and operational meaning.

### Target Leakage

A feature that defines a target cannot be used to predict that same target. Supplying path centre or an algebraic equivalent to predict `normalized_offset` creates identity leakage. Path area, width, and continuity also help define `stop_or_uncertain`; using them to reproduce that rule must be labeled as rule reconstruction, not learned-navigation performance. Phase B feature selection must use training data only and document excluded proxies.

### Oracle Features

Ground-truth-mask features represent perfect/oracle segmentation. They can support an upper-bound study but are unavailable to a deployed robot.

### Deployment Features

YOLO-predicted-mask features are closer to deployment conditions and must be reported separately. Direct image models are another deployable path because they do not consume annotation masks.

### Training-Set Prediction Leakage

Predicted training masks from the same segmentation model fitted on those images are in-sample and may inflate downstream results. Phase B should use out-of-fold segmentation predictions, held-out feature generation, or another leakage-resistant method where feasible.

## Work Remaining

### Phase B — Classical ML, ANFIS, and Deep Learning

Classical regression should compare Random Forest, XGBoost, SVR, and KNN with a simple baseline. The primary endpoint is valid continuous-offset prediction; report MAE, RMSE, and R².

Secondary classification may compare Random Forest, XGBoost, SVM, KNN, and Logistic Regression/Decision Tree. Use valid `left`/`forward`/`right` rows. Keep `stop_or_uncertain` separate unless its meaning and imbalance treatment are defensible. Report accuracy, balanced accuracy, macro precision/recall/F1, Cohen's kappa, MCC, and confusion matrices.

ANFIS should use approximately 3–6 interpretable, nonredundant, non-leaking inputs. Potential variables include cassava imbalance, ridge imbalance/orientation, obstruction, and carefully justified path-quality measures. Path width/area/continuity can only be used when they do not encode the chosen target rule. Apply the same split protocol and comparable metrics.

Deep-learning navigation experiments should compare a custom CNN, MobileNetV3, and ResNet using original images and common target definitions. Report histories, final metrics, inference speed, model size, plots, and failures.

### Phase C — Comparative and Statistical Analysis

Phase C will unify classical ML, ANFIS, and deep-learning results; perform significance tests and bootstrap confidence intervals; compare performance against latency/model size; assess interpretability; conduct ablations; recommend a final model; and generate research-paper tables, figures, and narrative.

## Exact Next Step

The next execution stage is **Phase B — Classical ML + ANFIS + CNN/Transfer Learning**.

Before running it, import the executed `03_phase_A_colab.ipynb` and `cassava_navigation_phase_A_results.zip`; verify checksums, checkpoints, metrics, figures, feature/target rows, IDs, and split alignment; then replace the local `not_trained` artifacts only with verified outputs.

The next practical deliverable should be:

```text
notebooks/04_phase_B_colab.ipynb
```

It is not created in this task. Its protocol must use Phase A datasets, respect target leakage, use the valid continuous subset for primary regression, use valid left/forward/right rows for secondary classification, keep stop/uncertain separate unless justified, separate oracle and predicted-mask experiments, preserve splits, and never tune on test data.

## Project Directory Structure

```text
cassava-navigation-ai/
├── configs/                     # Confirmed classes and experiment settings
├── data/{raw,extracted,processed}/
├── figures/                     # Audit/review and eventual model figures
├── models/segmentation_baseline/
├── notebooks/                   # Phase 1, Phase 2, and canonical Phase A notebooks
├── reports/                     # Narrative reports and machine-readable tables
├── results/segmentation_baseline/
├── scripts/                     # Notebook validation
├── src/{segmentation,features,navigation}/
├── tests/
├── PROJECT_STATUS.md
├── requirements.txt
└── requirements-segmentation.txt
```

## Reproduction Instructions

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/dataset_audit.py --zip-path /path/to/dataset.zip
python src/annotation_validation.py
python scripts/validate_phase_a_notebook.py  # validation only; no training
```

The confirmed class YAML is authoritative and is preserved by the Phase 2 script. Full Phase A GPU execution belongs in the canonical Colab notebook.

## Colab Workflow

1. Upload `notebooks/03_phase_A_colab.ipynb` to Colab.
2. Select **Runtime > Change runtime type > GPU**.
3. Choose manual ZIP upload or edit the Google Drive ZIP path.
4. Run all cells in order and review pretraining validation.
5. Download `/content/cassava_navigation_phase_A_results.zip`.
6. Download the executed notebook through **File > Download > Download .ipynb**.
7. Import both artifacts and verify them before Phase B.

The notebook supports `valid`/`val`, ignores generated dataset roots, and can reuse one uploaded ZIP in the same runtime.

## Git/GitHub Workflow

The canonical remote is `origin`. Source ZIPs, extracted/staged images, virtual environments, caches, credentials, and temporary Ultralytics outputs are ignored. Compact CSVs, reports, configs, notebooks, and figures are versioned. Verified `best.pt`/`last.pt` files may be committed under `models/segmentation_baseline/` when they remain below GitHub's file limit.

Use ordinary commits and non-force pushes:

```bash
git status
git add <reviewed files>
git commit -m "Describe the completed milestone"
git push
```

## Research Limitations

- Executed Phase A results and checkpoints have not been imported.
- Navigation labels are provisional and unvalidated by the control team.
- Direct path-centre equivalents leak the continuous target formula.
- Ground-truth-mask features are oracle features.
- Same-model training predictions are in-sample unless out-of-fold masks are generated.
- Exact hashes cannot exclude temporal/site correlation.
- Camera calibration, robot pose, depth, actuation, and safety evidence are unavailable.
- Polygon-size warnings remain for manual review.
- No Phase B model or comparative statistical conclusion exists.

## Expected Final Deliverables

Final deliverables include validated dataset documentation, segmentation checkpoints/metrics, oracle and deployment feature tables, provisional target documentation, classical ML/ANFIS/deep-learning experiments, leakage audits, statistical comparisons, ablations, computational/failure analysis, and paper-ready tables, figures, and narrative.
