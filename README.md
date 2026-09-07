> Phase A update: classes confirmed (0=path, 1=cassava_leaves, 2=ridge). Historical nominal counts are 603/120/80; artifact filtering gives 602/120/80 usable images. The extra training image is a checkpoint duplicate, not an additional sample.

# Vision-Based Navigation and Control Techniques for Autonomous Cassava Farm Robots

This repository contains reproducible dataset-audit and annotation-validation workflows for the `cassava-navigation-ai` research project. The dataset supports later work on segmentation, machine learning, ANFIS, deep learning, and robot navigation research.

No machine learning or deep learning model has been trained yet.

## Research Objectives

- Inspect and document the cassava farm image dataset structure.
- Detect the annotation format automatically.
- Validate image-label matching and identify data quality issues.
- Quantify image properties, annotation properties, class distribution, and split balance.
- Generate publication-quality audit figures and annotated image samples.
- Determine whether explicit robot navigation targets are present.
- Provide recommendations before later segmentation or navigation-control phases.
- Confirm whether class semantics are available from reliable metadata.
- Validate polygon-level annotation quality and prepare manual review contact sheets.

## Dataset Overview

The audited dataset is a Roboflow-style image dataset with:

- `803` readable images
- `802` YOLO label files
- `3` detected class IDs
- YOLO polygon segmentation annotations
- Splits: `603` train images, `120` validation images, and `80` test images
- No explicit navigation-command targets such as `move forward`, `turn left`, `turn right`, `stop`, steering angle, path-centre offset, or traversable-area labels

Classes are now human-confirmed: **0 = path, 1 = cassava_leaves, 2 = ridge**. The historical audit initially lacked metadata; this mapping supersedes that uncertainty.

Phase 2 excludes hidden/system artifacts such as `.ipynb_checkpoints` from every analysis. After filtering those artifacts, the analyzed dataset contains `802` images with labels and `28,655` structurally valid polygon rows.

## Project Structure

```text
cassava-navigation-ai/
├── data/
│   ├── raw/
│   ├── extracted/
│   └── processed/
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   └── 02_annotation_validation.ipynb
├── src/
│   ├── dataset_audit.py
│   └── annotation_validation.py
├── configs/
│   └── classes.yaml
├── reports/
├── figures/
├── requirements.txt
├── README.md
└── .gitignore
```

Large raw datasets, extracted dataset files, model weights, virtual environments, caches, and temporary files are intentionally ignored by Git.

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce The Dataset Audit

Place the dataset ZIP in the project root or pass its path explicitly. The original ZIP file is not committed to this repository.

```bash
python src/dataset_audit.py --zip-path data-20260712T141137Z-2-001.zip
```

The script defaults to `/mnt/data/data-20260610T182545Z-3-001.zip` when that file exists. Otherwise it uses the first ZIP in the project root.

The audit extracts data into `data/extracted/`, writes derived tables to `data/processed/`, and regenerates the report and figures.

## Reproduce The Annotation Validation

After Phase 1 has extracted the dataset, run:

```bash
python src/annotation_validation.py
```

This Phase 2 command:

- filters `.ipynb_checkpoints` and hidden/system artifacts from analysis
- searches metadata, notebooks, previous reports, and annotation export files for class names
- writes `configs/classes.yaml`
- validates every YOLO polygon row
- creates class-specific and edge-case contact sheets for manual review
- does not modify the extracted dataset
- does not train a model

## Generated Reports

The audit writes:

- `reports/dataset_summary.csv`
- `reports/class_distribution.csv`
- `reports/image_properties.csv`
- `reports/annotation_statistics.csv`
- `reports/data_quality_issues.csv`
- `reports/dataset_audit_report.md`
- `reports/class_semantics_review.csv`
- `reports/polygon_quality_checks.csv`
- `reports/manual_review_queue.csv`
- `reports/split_consistency.csv`
- `reports/annotation_quality_report.md`

The main narrative report is:

```bash
reports/dataset_audit_report.md
```

## Figures

Generated figures are stored in `figures/`:

- `annotated_samples.png`
- `dataset_split_distribution.png`
- `annotation_count_by_class.png`
- `images_containing_each_class.png`
- `objects_per_image.png`
- `image_width_distribution.png`
- `image_height_distribution.png`
- `image_aspect_ratio_distribution.png`
- `annotation_area_distribution.png`
- `class_imbalance.png`
- `class_0_review.png`
- `class_1_review.png`
- `class_2_review.png`
- `multiclass_review.png`
- `annotation_edge_cases.png`

The original ZIP is not modified. Extracted raw dataset contents under `data/extracted/` are left unchanged after extraction.

## Notebook

Start Jupyter and open the audit notebook:

```bash
jupyter notebook notebooks/01_dataset_audit.ipynb
```

Run all cells. The notebook imports and executes the same reusable Python functions as the command-line script, so notebook and script outputs remain consistent.

For Phase 2:

```bash
jupyter notebook notebooks/02_annotation_validation.ipynb
```

Run all cells to regenerate the class-semantics review tables, polygon quality checks, manual review queue, split consistency summary, and contact sheets.

## Current Status

Phase 1 is complete:

- Dataset extraction and folder-structure documentation completed.
- YOLO polygon segmentation format detected and validated.
- Image and annotation statistics generated.
- Data quality issues documented.
- Annotated samples and graphs generated.
- Navigation-target availability assessed.

Phase 2 is complete:

- Hidden/system artifacts are excluded from analyses without modifying the original dataset.
- No reliable class-name metadata was found.
- `configs/classes.yaml` locks the human-confirmed mapping.
- `28,655` polygon rows are structurally valid.
- `216` polygons are flagged for manual review due to extremely small or extremely large normalized area.
- No exact duplicate image hashes were found across train, validation, and test splits.

## Phase A — self-contained GPU Colab notebook

Open **[notebooks/03_phase_A_colab.ipynb](notebooks/03_phase_A_colab.ipynb)** in Google Colab. This is the complete notebook for the compressed Phase A workflow; it requires only your dataset ZIP, not the project repository. It supersedes the earlier project-dependent Colab scaffold.

1. Go to [Google Colab](https://colab.research.google.com/), choose **File > Upload notebook**, and select `03_phase_A_colab.ipynb`.
2. Choose **Runtime > Change runtime type > GPU**.
3. Edit the settings at the top if needed. Defaults are seed 42, image size 640, 60 epochs, batch 8, patience 12 and two workers.
4. Choose `INPUT_METHOD = 'upload'` and upload one dataset ZIP when prompted, or choose `'drive'` and set `ZIP_PATH` to your ZIP in Google Drive. Any ZIP filename is supported; the dataset root and `valid`/`val` split are discovered automatically. If a full project ZIP also contains `data/processed/segmentation_dataset`, that generated copy is ignored in favor of the single original source root.
5. Choose **Runtime > Run all**. Dependency installation, validation, one segmentation training run, evaluation, predictions, feature extraction, provisional targets, reports and packaging run in order.
6. Download `cassava_navigation_phase_A_results.zip` from the final cell. Separately choose **File > Download > Download .ipynb** to save the executed notebook with outputs. Return **both files** for review.

The notebook extracts into `/content/cassava-navigation-ai/data/extracted`, retains the confirmed mapping (0 path, 1 cassava_leaves, 2 ridge), and keeps suspicious polygon sizes as review warnings. Technical errors are documented and excluded only from a derived training copy. It stops on duplicate IDs or exact cross-split duplicates; it never silently reassigns splits.

The results ZIP includes best/last checkpoints, metrics, per-class results, timing, test predictions, diagnostic figures, feature and target CSVs, feature dictionary, configurations, reusable Python helpers and reports. It excludes the original dataset ZIP and dataset images/labels. Save/download outputs before the Colab runtime disconnects. A completed training cell can reuse matching checkpoints; changed data/configuration or an incomplete training run requires deliberate recovery in a fresh runtime.

No model was trained locally. The notebook has been validated for JSON/schema, code compilation and targeted helper behavior; full GPU execution and scientific results remain pending your Colab run. See [the local validation report](reports/colab_notebook_validation.md).

Continuous image-space path-centre offset is the primary **provisional research target**: negative means image-left and positive means image-right. It is not a recorded steering command. Thresholds are fitted from training distributions only and saved explicitly. Direct path-derived inputs can leak the target formula, so the primary feature recommendations exclude them. Ground-truth masks are oracle inputs; predicted training masks are in-sample, not out-of-fold. Future end-to-end modeling requires out-of-fold training predictions, temporal/site leakage review and control-team validation.

Stop after Phase A. The notebook does not train classical ML, ANFIS, CNN navigation or transfer-learning navigation models, and it does not control a robot.
