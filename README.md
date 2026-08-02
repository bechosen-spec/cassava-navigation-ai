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

Class names were not available in metadata, so the audit reports them conservatively as `Class 0`, `Class 1`, and `Class 2`.

Phase 2 excludes hidden/system artifacts such as `.ipynb_checkpoints` from every analysis. After filtering those artifacts, the validation set contains `802` images with labels and `28,655` structurally valid polygon rows.

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
- `configs/classes.yaml` marks all classes as unconfirmed.
- `28,655` polygon rows are structurally valid.
- `216` polygons are flagged for manual review due to extremely small or extremely large normalized area.
- No exact duplicate image hashes were found across train, validation, and test splits.

Recommended next step: manually confirm the semantic meaning of Class 0, Class 1, and Class 2 using the generated contact sheets before any segmentation training.
