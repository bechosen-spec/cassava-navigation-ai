# Vision-Based Navigation and Control Techniques for Autonomous Cassava Farm Robots

This repository contains Phase 1 of the `cassava-navigation-ai` research project: a reproducible dataset audit for cassava farm imagery intended to support later work on segmentation, machine learning, ANFIS, deep learning, and robot navigation research.

No machine learning or deep learning model is trained in this phase.

## Research Objectives

- Inspect and document the cassava farm image dataset structure.
- Detect the annotation format automatically.
- Validate image-label matching and identify data quality issues.
- Quantify image properties, annotation properties, class distribution, and split balance.
- Generate publication-quality audit figures and annotated image samples.
- Determine whether explicit robot navigation targets are present.
- Provide recommendations before later segmentation or navigation-control phases.

## Dataset Overview

The audited dataset is a Roboflow-style image dataset with:

- `803` readable images
- `802` YOLO label files
- `3` detected class IDs
- YOLO polygon segmentation annotations
- Splits: `603` train images, `120` validation images, and `80` test images
- No explicit navigation-command targets such as `move forward`, `turn left`, `turn right`, `stop`, steering angle, path-centre offset, or traversable-area labels

Class names were not available in metadata, so the audit reports them conservatively as `Class 0`, `Class 1`, and `Class 2`.

## Project Structure

```text
cassava-navigation-ai/
├── data/
│   ├── raw/
│   ├── extracted/
│   └── processed/
├── notebooks/
│   └── 01_dataset_audit.ipynb
├── src/
│   └── dataset_audit.py
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

## Generated Reports

The audit writes:

- `reports/dataset_summary.csv`
- `reports/class_distribution.csv`
- `reports/image_properties.csv`
- `reports/annotation_statistics.csv`
- `reports/data_quality_issues.csv`
- `reports/dataset_audit_report.md`

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

The original ZIP is not modified. Extracted raw dataset contents under `data/extracted/` are left unchanged after extraction.

## Notebook

Start Jupyter and open the audit notebook:

```bash
jupyter notebook notebooks/01_dataset_audit.ipynb
```

Run all cells. The notebook imports and executes the same reusable Python functions as the command-line script, so notebook and script outputs remain consistent.

## Current Status

Phase 1 is complete:

- Dataset extraction and folder-structure documentation completed.
- YOLO polygon segmentation format detected and validated.
- Image and annotation statistics generated.
- Data quality issues documented.
- Annotated samples and graphs generated.
- Navigation-target availability assessed.

Recommended next step: confirm the semantic meaning of each class ID, then decide whether Phase 2 targets segmentation only or requires additional navigation-control labels.
