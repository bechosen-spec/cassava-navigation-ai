"""Dataset audit utilities for cassava farm robot vision research.

This module extracts an image dataset, detects common annotation formats,
validates image/label consistency, writes CSV summaries, and creates figures
without training any model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = {".png", ".bmp", ".tif", ".tiff"}
SPLIT_ALIASES = {"train": "train", "training": "train", "valid": "valid", "val": "valid", "validation": "valid", "test": "test"}
NAVIGATION_TERMS = {
    "move forward",
    "forward",
    "turn left",
    "left",
    "turn right",
    "right",
    "stop",
    "steering angle",
    "steering",
    "path-centre offset",
    "path-center offset",
    "offset",
    "traversable-area",
    "traversable area",
}


@dataclass(frozen=True)
class Paths:
    """Project paths used by the audit."""

    project_root: Path
    zip_path: Path
    extracted_dir: Path
    processed_dir: Path
    reports_dir: Path
    figures_dir: Path


def normalize_path(path: Path) -> Path:
    """Return an expanded absolute path."""

    return path.expanduser().resolve()


def safe_extract_zip(zip_path: Path, destination: Path) -> list[Path]:
    """Extract a ZIP archive while preventing path traversal.

    Existing extracted files are overwritten by the ZIP extractor, but the
    original archive is never modified.
    """

    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    dest_root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(dest_root)):
                raise ValueError(f"Unsafe archive member path: {member.filename}")
            archive.extract(member, destination)
            extracted.append(target)
    return extracted


def find_files(root: Path, extensions: set[str]) -> list[Path]:
    """Find files under root with extensions in a case-insensitive set."""

    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def detect_split(path: Path) -> str:
    """Infer dataset split from path components."""

    for part in path.parts:
        split = SPLIT_ALIASES.get(part.lower())
        if split:
            return split
    return "unsplit"


def folder_tree(root: Path, max_depth: int = 5) -> str:
    """Return a compact text tree for a folder."""

    lines = [f"{root.name}/"]
    root_depth = len(root.parts)
    for path in sorted(root.rglob("*")):
        depth = len(path.parts) - root_depth
        if depth > max_depth:
            continue
        indent = "  " * depth
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{indent}{path.name}{suffix}")
    return "\n".join(lines)


def load_class_names(root: Path, annotations: list[dict[str, Any]]) -> tuple[dict[int, str], list[str]]:
    """Load class names from known metadata files, or fall back to generic IDs."""

    sources: list[str] = []
    names: dict[int, str] = {}

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if lowered == "classes.txt":
            for idx, line in enumerate(line.strip() for line in text.splitlines() if line.strip()):
                names[idx] = line
            sources.append(str(path.relative_to(root)))
        elif lowered in {"data.yaml", "data.yml"}:
            parsed = parse_names_from_yaml_like(text)
            if parsed:
                names.update(parsed)
                sources.append(str(path.relative_to(root)))

    ids = sorted({int(row["class_id"]) for row in annotations if row.get("class_id") is not None})
    for class_id in ids:
        names.setdefault(class_id, f"Class {class_id}")
    return names, sources


def parse_names_from_yaml_like(text: str) -> dict[int, str]:
    """Parse class names from simple YOLO data.yaml content without requiring PyYAML."""

    names: dict[int, str] = {}
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        value = stripped.split(":", 1)[1].strip()
        if value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            return {i: item for i, item in enumerate(items)}
        for child in lines[idx + 1 :]:
            if child and not child.startswith((" ", "\t", "-")):
                break
            child_stripped = child.strip()
            if not child_stripped:
                continue
            if child_stripped.startswith("-"):
                names[len(names)] = child_stripped[1:].strip().strip("'\"")
            elif ":" in child_stripped:
                key, raw = child_stripped.split(":", 1)
                if key.strip().isdigit():
                    names[int(key.strip())] = raw.strip().strip("'\"")
        break
    return names


def parse_yolo_label(label_path: Path, image_path: Path | None, issue_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse YOLO bbox or polygon labels and collect validation issues."""

    rows: list[dict[str, Any]] = []
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        issue_rows.append(issue(label_path, "unreadable_label", str(exc)))
        return rows
    if not lines:
        issue_rows.append(issue(label_path, "empty_label_file", "Label file contains no rows"))
        return rows

    split = detect_split(label_path)
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 5:
            issue_rows.append(issue(label_path, "invalid_label_row", f"Line {line_number}: expected at least 5 columns"))
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            issue_rows.append(issue(label_path, "invalid_label_row", f"Line {line_number}: non-numeric value"))
            continue

        if len(values) == 4:
            fmt = "YOLO bounding-box"
            x_center, y_center, width, height = values
            area = width * height
            point_count = 0
            if width <= 0 or height <= 0:
                issue_rows.append(issue(label_path, "invalid_coordinate_values", f"Line {line_number}: bbox width/height must be positive"))
        elif len(values) >= 6 and len(values) % 2 == 0:
            fmt = "YOLO polygon segmentation"
            xs = values[0::2]
            ys = values[1::2]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            x_center = min(xs) + width / 2
            y_center = min(ys) + height / 2
            area = polygon_area(xs, ys)
            point_count = len(xs)
            if point_count < 3:
                issue_rows.append(issue(label_path, "invalid_coordinate_values", f"Line {line_number}: polygon has fewer than 3 points"))
        else:
            issue_rows.append(issue(label_path, "invalid_label_row", f"Line {line_number}: unsupported YOLO row length {len(parts)}"))
            continue

        if any(value < 0 or value > 1 for value in values):
            issue_rows.append(issue(label_path, "coordinates_outside_yolo_range", f"Line {line_number}: coordinates are outside [0, 1]"))

        rows.append(
            {
                "label_path": str(label_path),
                "image_path": str(image_path) if image_path else "",
                "split": split,
                "class_id": class_id,
                "format": fmt,
                "x_center": x_center,
                "y_center": y_center,
                "bbox_width": width,
                "bbox_height": height,
                "area_normalized": area,
                "polygon_point_count": point_count,
                "raw_values": values,
            }
        )
    return rows


def polygon_area(xs: list[float], ys: list[float]) -> float:
    """Return normalized polygon area using the shoelace formula."""

    if len(xs) < 3:
        return 0.0
    return abs(sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i] for i in range(len(xs)))) / 2


def issue(path: Path, issue_type: str, detail: str) -> dict[str, str]:
    """Create a data quality issue row."""

    return {"path": str(path), "issue_type": issue_type, "detail": detail}


def image_key(path: Path) -> str:
    """Return a matching key for an image or label path."""

    return f"{detect_split(path)}::{path.stem}"


def file_hash(path: Path, block_size: int = 65536) -> str:
    """Return an MD5 hash for duplicate detection."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze_images(image_paths: list[Path], issue_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Read image metadata and report corrupt or unsupported files."""

    rows: list[dict[str, Any]] = []
    filename_counts = Counter(path.name for path in image_paths)
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    for path in image_paths:
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            issue_rows.append(issue(path, "unsupported_image_format", f"Extension {path.suffix} is not in {sorted(IMAGE_EXTENSIONS)}"))
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
                fmt = image.format or path.suffix.lower().lstrip(".").upper()
        except Exception as exc:  # noqa: BLE001 - record all image decoder failures.
            issue_rows.append(issue(path, "corrupted_or_unreadable_image", str(exc)))
            continue
        try:
            digest = file_hash(path)
            hash_to_paths[digest].append(str(path))
        except OSError as exc:
            issue_rows.append(issue(path, "unreadable_image", str(exc)))
            digest = ""
        rows.append(
            {
                "image_path": str(path),
                "filename": path.name,
                "split": detect_split(path),
                "width": width,
                "height": height,
                "aspect_ratio": width / height if height else math.nan,
                "format": fmt,
                "file_size_bytes": path.stat().st_size,
                "colour_channels": channels_from_mode(mode),
                "mode": mode,
                "md5": digest,
                "duplicate_filename_count": filename_counts[path.name],
            }
        )

    for filename, count in filename_counts.items():
        if count > 1:
            matching = [str(path) for path in image_paths if path.name == filename]
            issue_rows.append({"path": filename, "issue_type": "duplicate_filename", "detail": "; ".join(matching)})
    for digest, paths in hash_to_paths.items():
        if digest and len(paths) > 1:
            issue_rows.append({"path": digest, "issue_type": "duplicate_image", "detail": "; ".join(paths)})
    return pd.DataFrame(rows)


def channels_from_mode(mode: str) -> int:
    """Map PIL mode to channel count."""

    return {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}.get(mode, len(mode))


def detect_annotation_sources(root: Path) -> dict[str, list[str]]:
    """Detect likely annotation files/folders by format."""

    return {
        "yolo_txt": [str(path) for path in sorted(root.rglob("*.txt")) if "label" in str(path).lower()],
        "coco_json": [str(path) for path in sorted(root.rglob("*.json"))],
        "pascal_voc_xml": [str(path) for path in sorted(root.rglob("*.xml"))],
        "mask_images": [str(path) for path in sorted(root.rglob("*")) if path.is_file() and "mask" in str(path).lower() and path.suffix.lower() in MASK_EXTENSIONS],
    }


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, output: Path, xlabel: str = "") -> None:
    """Write a high-resolution bar chart."""

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df[x].astype(str), df[y], color="#3f7f5f")
    ax.set_title(title)
    ax.set_xlabel(xlabel or x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_hist(values: Iterable[float], title: str, xlabel: str, output: Path, bins: int = 30) -> None:
    """Write a high-resolution histogram."""

    clean = [value for value in values if pd.notna(value)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(clean, bins=bins, color="#4f6f95", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def draw_annotated_samples(
    image_df: pd.DataFrame,
    annotations: list[dict[str, Any]],
    class_names: dict[int, str],
    output: Path,
    sample_count: int,
    seed: int,
) -> None:
    """Draw random images with YOLO annotations overlaid."""

    if image_df.empty:
        return
    rng = np.random.default_rng(seed)
    selected_paths: list[str] = []
    for split in ["train", "valid", "test"]:
        split_paths = image_df.loc[image_df["split"] == split, "image_path"].tolist()
        if split_paths:
            selected_paths.append(str(rng.choice(split_paths)))
    remaining = [path for path in image_df["image_path"].tolist() if path not in selected_paths]
    if remaining:
        needed = max(0, sample_count - len(selected_paths))
        selected_paths.extend(rng.choice(remaining, size=min(needed, len(remaining)), replace=False).tolist())

    annotation_map: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        annotation_map[row["image_path"]].append(row)

    cols = 4
    rows = math.ceil(len(selected_paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes_array = np.atleast_1d(axes).ravel()
    for ax, image_path in zip(axes_array, selected_paths):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            draw = ImageDraw.Draw(image, "RGBA")
            width, height = image.size
            for ann in annotation_map.get(image_path, []):
                values = ann["raw_values"]
                class_id = int(ann["class_id"])
                label = class_names.get(class_id, f"Class {class_id}")
                color = (255, 75, 75, 180)
                if ann["format"] == "YOLO polygon segmentation":
                    points = [(values[i] * width, values[i + 1] * height) for i in range(0, len(values), 2)]
                    draw.polygon(points, outline=color, fill=(255, 75, 75, 45))
                    x_text, y_text = points[0]
                else:
                    x_center, y_center, box_w, box_h = values
                    x0 = (x_center - box_w / 2) * width
                    y0 = (y_center - box_h / 2) * height
                    x1 = (x_center + box_w / 2) * width
                    y1 = (y_center + box_h / 2) * height
                    draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                    x_text, y_text = x0, y0
                draw.rectangle([x_text, y_text, x_text + 7 * len(label) + 8, y_text + 16], fill=(0, 0, 0, 160))
                draw.text((x_text + 4, y_text + 2), label, fill=(255, 255, 255, 255))
            ax.imshow(image)
        ax.set_title(f"{Path(image_path).name}\n{detect_split(Path(image_path))}", fontsize=8)
        ax.axis("off")
    for ax in axes_array[len(selected_paths) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    """Write rows or a DataFrame to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_numeric(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Create summary statistics for numeric DataFrame columns."""

    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in df:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append(
            {
                "property": column,
                "count": int(values.count()),
                "min": values.min(),
                "max": values.max(),
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(ddof=0),
            }
        )
    return rows


def assess_navigation_targets(class_names: dict[int, str], root: Path) -> tuple[bool, list[str]]:
    """Detect explicit navigation target labels or metadata."""

    hits: list[str] = []
    for name in class_names.values():
        lowered = name.lower()
        if any(term == lowered or term in lowered for term in NAVIGATION_TERMS):
            hits.append(f"class:{name}")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".yaml", ".yml", ".json", ".csv"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if any(term in text for term in NAVIGATION_TERMS):
            hits.append(str(path.relative_to(root)))
    return bool(hits), sorted(set(hits))


def run_audit(paths: Paths, sample_count: int = 12, seed: int = 42) -> dict[str, Any]:
    """Run the complete reproducible dataset audit."""

    for directory in [paths.extracted_dir, paths.processed_dir, paths.reports_dir, paths.figures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    safe_extract_zip(paths.zip_path, paths.extracted_dir)
    image_paths = find_files(paths.extracted_dir, IMAGE_EXTENSIONS)
    all_txt_paths = sorted(paths.extracted_dir.rglob("*.txt"))
    label_paths = [path for path in all_txt_paths if "label" in str(path).lower()]
    if not label_paths:
        label_paths = all_txt_paths
    annotation_sources = detect_annotation_sources(paths.extracted_dir)

    issues: list[dict[str, Any]] = []
    image_df = analyze_images(image_paths, issues)
    image_by_key = {image_key(Path(row.image_path)): Path(row.image_path) for row in image_df.itertuples(index=False)}
    label_by_key = {image_key(path): path for path in label_paths}

    for key, image_path in image_by_key.items():
        if key not in label_by_key:
            issues.append(issue(image_path, "image_without_label_file", "No matching label file with same split and stem"))
    for key, label_path in label_by_key.items():
        if key not in image_by_key:
            issues.append(issue(label_path, "label_file_without_image", "No matching image file with same split and stem"))

    annotations: list[dict[str, Any]] = []
    for key, label_path in label_by_key.items():
        annotations.extend(parse_yolo_label(label_path, image_by_key.get(key), issues))

    class_names, class_sources = load_class_names(paths.extracted_dir, annotations)
    navigation_present, navigation_hits = assess_navigation_targets(class_names, paths.extracted_dir)
    annotation_format = detect_annotation_format(annotation_sources, annotations)

    annotation_df = pd.DataFrame(annotations)
    if not annotation_df.empty:
        annotation_df["class_name"] = annotation_df["class_id"].map(class_names)

    class_distribution = build_class_distribution(annotation_df, class_names)
    per_image_stats = build_per_image_annotation_stats(image_df, annotation_df)
    split_counts = build_split_counts(image_df, label_paths)
    issue_df = pd.DataFrame(issues, columns=["path", "issue_type", "detail"])

    write_outputs(
        paths,
        image_df,
        annotation_df,
        class_distribution,
        per_image_stats,
        split_counts,
        issue_df,
        annotations,
        class_names,
        annotation_format,
        annotation_sources,
        class_sources,
        navigation_present,
        navigation_hits,
        sample_count,
        seed,
    )

    return {
        "total_images": int(len(image_df)),
        "total_label_files": int(len(label_paths)),
        "num_classes": int(len(class_names)),
        "annotation_format": annotation_format,
        "split_counts": split_counts.to_dict("records"),
        "data_quality_problems": int(len(issue_df)),
        "navigation_targets_present": navigation_present,
    }


def detect_annotation_format(annotation_sources: dict[str, list[str]], annotations: list[dict[str, Any]]) -> str:
    """Determine the annotation format from detected files and parsed rows."""

    parsed_formats = Counter(row["format"] for row in annotations)
    if parsed_formats:
        if len(parsed_formats) == 1:
            return next(iter(parsed_formats))
        return "Mixed YOLO labels: " + ", ".join(f"{name} ({count})" for name, count in parsed_formats.items())
    if annotation_sources["coco_json"]:
        return "COCO JSON"
    if annotation_sources["pascal_voc_xml"]:
        return "Pascal VOC XML"
    if annotation_sources["mask_images"]:
        return "Mask images"
    return "Unknown or unsupported"


def build_class_distribution(annotation_df: pd.DataFrame, class_names: dict[int, str]) -> pd.DataFrame:
    """Build class-level annotation and image counts."""

    rows: list[dict[str, Any]] = []
    for class_id, class_name in sorted(class_names.items()):
        subset = annotation_df[annotation_df["class_id"] == class_id] if not annotation_df.empty else pd.DataFrame()
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "annotation_count": int(len(subset)),
                "image_count": int(subset["image_path"].nunique()) if not subset.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def build_per_image_annotation_stats(image_df: pd.DataFrame, annotation_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-image object/polygon statistics."""

    counts = annotation_df.groupby("image_path").size().to_dict() if not annotation_df.empty else {}
    rows: list[dict[str, Any]] = []
    for row in image_df.itertuples(index=False):
        image_annotations = annotation_df[annotation_df["image_path"] == row.image_path] if not annotation_df.empty else pd.DataFrame()
        rows.append(
            {
                "image_path": row.image_path,
                "split": row.split,
                "annotation_count": int(counts.get(row.image_path, 0)),
                "mean_annotation_area": image_annotations["area_normalized"].mean() if not image_annotations.empty else 0,
                "max_annotation_area": image_annotations["area_normalized"].max() if not image_annotations.empty else 0,
                "mean_polygon_point_count": image_annotations["polygon_point_count"].mean() if not image_annotations.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def build_split_counts(image_df: pd.DataFrame, label_paths: list[Path]) -> pd.DataFrame:
    """Build train/valid/test image and label counts."""

    image_counts = image_df["split"].value_counts().to_dict() if not image_df.empty else {}
    label_counts = Counter(detect_split(path) for path in label_paths)
    splits = sorted(set(image_counts) | set(label_counts), key=lambda value: ["train", "valid", "test", "unsplit"].index(value) if value in ["train", "valid", "test", "unsplit"] else 99)
    return pd.DataFrame(
        [{"split": split, "image_count": int(image_counts.get(split, 0)), "label_file_count": int(label_counts.get(split, 0))} for split in splits]
    )


def write_outputs(
    paths: Paths,
    image_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    class_distribution: pd.DataFrame,
    per_image_stats: pd.DataFrame,
    split_counts: pd.DataFrame,
    issue_df: pd.DataFrame,
    annotations: list[dict[str, Any]],
    class_names: dict[int, str],
    annotation_format: str,
    annotation_sources: dict[str, list[str]],
    class_sources: list[str],
    navigation_present: bool,
    navigation_hits: list[str],
    sample_count: int,
    seed: int,
) -> None:
    """Write all requested CSV, figure, and Markdown report outputs."""

    summary_rows = [
        {"metric": "total_images", "value": len(image_df)},
        {"metric": "total_label_files", "value": int(split_counts["label_file_count"].sum()) if "label_file_count" in split_counts else 0},
        {"metric": "num_classes", "value": len(class_names)},
        {"metric": "annotation_format", "value": annotation_format},
        {"metric": "data_quality_problems", "value": len(issue_df)},
        {"metric": "navigation_targets_present", "value": navigation_present},
    ]
    write_csv(paths.reports_dir / "dataset_summary.csv", summary_rows)
    write_csv(paths.reports_dir / "class_distribution.csv", class_distribution)
    write_csv(paths.reports_dir / "image_properties.csv", image_df)
    write_csv(paths.reports_dir / "annotation_statistics.csv", per_image_stats)
    write_csv(paths.reports_dir / "data_quality_issues.csv", issue_df)
    if not annotation_df.empty:
        write_csv(paths.processed_dir / "parsed_annotations.csv", annotation_df.drop(columns=["raw_values"]))

    plot_bar(split_counts, "split", "image_count", "Dataset Split Distribution", "Images", paths.figures_dir / "dataset_split_distribution.png")
    if not class_distribution.empty:
        plot_bar(class_distribution, "class_name", "annotation_count", "Annotation Count by Class", "Annotations", paths.figures_dir / "annotation_count_by_class.png")
        plot_bar(class_distribution, "class_name", "image_count", "Images Containing Each Class", "Images", paths.figures_dir / "images_containing_each_class.png")
        imbalance = class_distribution.copy()
        max_count = max(imbalance["annotation_count"].max(), 1)
        imbalance["imbalance_ratio_vs_max"] = imbalance["annotation_count"] / max_count
        plot_bar(imbalance, "class_name", "imbalance_ratio_vs_max", "Class Imbalance Relative to Majority Class", "Ratio", paths.figures_dir / "class_imbalance.png")
    plot_hist(per_image_stats["annotation_count"], "Objects or Polygons per Image", "Count", paths.figures_dir / "objects_per_image.png", bins=30)
    plot_hist(image_df["width"], "Image Width Distribution", "Width (px)", paths.figures_dir / "image_width_distribution.png")
    plot_hist(image_df["height"], "Image Height Distribution", "Height (px)", paths.figures_dir / "image_height_distribution.png")
    plot_hist(image_df["aspect_ratio"], "Image Aspect Ratio Distribution", "Width / Height", paths.figures_dir / "image_aspect_ratio_distribution.png")
    if not annotation_df.empty:
        plot_hist(annotation_df["area_normalized"], "Annotation Area Distribution", "Normalized area", paths.figures_dir / "annotation_area_distribution.png")
    draw_annotated_samples(image_df, annotations, class_names, paths.figures_dir / "annotated_samples.png", sample_count, seed)

    report = build_markdown_report(
        paths,
        image_df,
        annotation_df,
        class_distribution,
        per_image_stats,
        split_counts,
        issue_df,
        annotation_format,
        annotation_sources,
        class_sources,
        navigation_present,
        navigation_hits,
    )
    (paths.reports_dir / "dataset_audit_report.md").write_text(report, encoding="utf-8")


def build_markdown_report(
    paths: Paths,
    image_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    class_distribution: pd.DataFrame,
    per_image_stats: pd.DataFrame,
    split_counts: pd.DataFrame,
    issue_df: pd.DataFrame,
    annotation_format: str,
    annotation_sources: dict[str, list[str]],
    class_sources: list[str],
    navigation_present: bool,
    navigation_hits: list[str],
) -> str:
    """Build the Markdown dataset audit report."""

    image_summary = pd.DataFrame(summarize_numeric(image_df, ["width", "height", "aspect_ratio", "file_size_bytes", "colour_channels"]))
    annotation_summary = pd.DataFrame(summarize_numeric(annotation_df, ["bbox_width", "bbox_height", "area_normalized", "polygon_point_count"])) if not annotation_df.empty else pd.DataFrame()
    majority = int(class_distribution["annotation_count"].max()) if not class_distribution.empty else 0
    minority = int(class_distribution["annotation_count"].min()) if not class_distribution.empty else 0
    imbalance_ratio = (majority / minority) if minority else math.inf
    nav_text = (
        f"Navigation targets appear to be present: {', '.join(navigation_hits)}."
        if navigation_present
        else "No explicit navigation targets were found. Segmentation models can still be trained, but navigation classification or steering prediction will require additional labels or a scientifically justified target-generation method."
    )
    return "\n".join(
        [
            "# Cassava Farm Image Dataset Audit",
            "",
            "## Dataset overview",
            f"- ZIP archive: `{paths.zip_path}`",
            f"- Extracted directory: `{paths.extracted_dir}`",
            f"- Total readable images: {len(image_df)}",
            f"- Total parsed annotations: {len(annotation_df)}",
            f"- Total data quality issues: {len(issue_df)}",
            "",
            "## Folder structure",
            "```text",
            folder_tree(paths.extracted_dir),
            "```",
            "",
            "## Discovered folders and metadata",
            f"- Image folders: {', '.join(sorted({str(Path(p).parent.relative_to(paths.extracted_dir)) for p in image_df['image_path']})) if not image_df.empty else 'None'}",
            f"- Label folders: {', '.join(sorted({str(Path(p).parent.relative_to(paths.extracted_dir)) for p in annotation_df['label_path']})) if not annotation_df.empty else 'None'}",
            f"- Metadata/class files: {', '.join(class_sources) if class_sources else 'None found'}",
            "",
            "## Annotation format",
            f"- Detected format: {annotation_format}",
            "- Identification method: the audit searched for COCO JSON, Pascal VOC XML, mask images, and YOLO `.txt` labels. It parsed label rows as `class_id` followed by either four normalized values for bounding boxes or an even-length coordinate sequence for polygons.",
            f"- Annotation source counts: `{json.dumps({key: len(value) for key, value in annotation_sources.items()})}`",
            "",
            "## Dataset split statistics",
            split_counts.to_markdown(index=False),
            "",
            "## Class distribution",
            class_distribution.to_markdown(index=False) if not class_distribution.empty else "No classes found.",
            "",
            "## Image properties",
            image_summary.to_markdown(index=False) if not image_summary.empty else "No readable images found.",
            "",
            "## Annotation properties",
            annotation_summary.to_markdown(index=False) if not annotation_summary.empty else "No annotations found.",
            "",
            "## Data quality problems",
            issue_df["issue_type"].value_counts().rename_axis("issue_type").reset_index(name="count").to_markdown(index=False) if not issue_df.empty else "No data quality problems were detected by this audit.",
            "",
            "## Class imbalance assessment",
            f"The majority/minority annotation ratio is {'infinite because one class has zero annotations' if math.isinf(imbalance_ratio) else f'{imbalance_ratio:.2f}'}. A high ratio may require stratified sampling, class-aware augmentation, loss weighting, or targeted data collection.",
            "",
            "## Suitability assessment",
            f"- Segmentation training: {'Suitable for segmentation experiments if the labels are accepted as polygons.' if 'polygon' in annotation_format.lower() else 'Potentially suitable for detection; segmentation requires polygon/mask labels.'}",
            "- Traditional machine learning: Suitable after feature engineering from images or annotations, but raw pixels alone are high-dimensional and require careful validation.",
            "- ANFIS: Possible only after deriving compact numeric features, such as plant/weed area ratios, row geometry, texture descriptors, or object counts.",
            f"- Navigation-command prediction: {nav_text}",
            "",
            "## Missing information before later phases",
            "- Explicit robot action labels or continuous control targets, if navigation prediction is required.",
            "- Documentation of collection conditions, camera calibration, robot pose, and field geometry.",
            "- A scientifically justified method for turning segmentation outputs into navigation labels, if manual navigation labels are unavailable.",
            "",
            "## Recommendations for next step",
            "1. Review the quality issue CSV and inspect any severe image/label mismatches.",
            "2. Confirm the semantic meaning of each class ID before model training.",
            "3. Decide whether Phase 2 targets segmentation only or navigation control; collect navigation labels if the latter is required.",
            "4. Create a small frozen validation protocol before any training begins.",
            "",
        ]
    )


def choose_default_zip(project_root: Path) -> Path:
    """Choose the requested ZIP path if present, otherwise the first local ZIP."""

    requested = Path("/mnt/data/data-20260610T182545Z-3-001.zip")
    if requested.exists():
        return requested
    zips = sorted(project_root.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("No ZIP file found. Pass --zip-path explicitly.")
    return zips[0]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Audit cassava farm image dataset without training a model.")
    parser.add_argument("--zip-path", type=Path, default=None, help="Dataset ZIP path.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root.")
    parser.add_argument("--extracted-dir", type=Path, default=None, help="Directory for extracted data.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample figures.")
    parser.add_argument("--sample-count", type=int, default=12, help="Number of annotated samples to draw.")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    project_root = normalize_path(args.project_root)
    zip_path = normalize_path(args.zip_path) if args.zip_path else choose_default_zip(project_root)
    paths = Paths(
        project_root=project_root,
        zip_path=zip_path,
        extracted_dir=normalize_path(args.extracted_dir) if args.extracted_dir else project_root / "data" / "extracted",
        processed_dir=project_root / "data" / "processed",
        reports_dir=project_root / "reports",
        figures_dir=project_root / "figures",
    )
    summary = run_audit(paths, sample_count=args.sample_count, seed=args.seed)
    print("Dataset audit complete")
    print(f"Images: {summary['total_images']}")
    print(f"Label files: {summary['total_label_files']}")
    print(f"Classes: {summary['num_classes']}")
    print(f"Annotation format: {summary['annotation_format']}")
    print(f"Split counts: {summary['split_counts']}")
    print(f"Data quality problems: {summary['data_quality_problems']}")
    print(f"Navigation targets present: {summary['navigation_targets_present']}")
    print("Recommended next task: review reports/data_quality_issues.csv and confirm class semantics before Phase 2.")


if __name__ == "__main__":
    main()
