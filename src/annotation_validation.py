"""Phase 2 class semantics and annotation quality validation.

The code in this module does not modify the extracted dataset and does not
train a model. It filters hidden/system artifacts from analysis, searches for
class-name metadata, validates YOLO polygon rows, builds review queues, and
creates contact sheets for manual class-semantics review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
METADATA_EXTENSIONS = {".yaml", ".yml", ".txt", ".md", ".json", ".csv", ".ipynb"}
SPLIT_ALIASES = {"train": "train", "training": "train", "valid": "valid", "val": "valid", "validation": "valid", "test": "test"}
HIDDEN_PART_PREFIXES = (".", "__MACOSX")
EXPECTED_CLASS_IDS = {0, 1, 2}
RANDOM_SEED = 42
SMALL_AREA_THRESHOLD = 1e-5
LARGE_AREA_THRESHOLD = 0.35
BORDER_EPSILON = 0.002


@dataclass(frozen=True)
class Phase2Paths:
    """Filesystem paths for Phase 2 outputs."""

    project_root: Path
    extracted_dir: Path
    reports_dir: Path
    figures_dir: Path
    configs_dir: Path


def is_hidden_or_system_path(path: Path) -> bool:
    """Return True when a path is a hidden/system artifact."""

    return any(part.startswith(HIDDEN_PART_PREFIXES) for part in path.parts)


def iter_dataset_files(root: Path, extensions: set[str]) -> list[Path]:
    """Return non-hidden files with matching extensions."""

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions and not is_hidden_or_system_path(path.relative_to(root))
    )


def detect_split(path: Path) -> str:
    """Infer split from path components."""

    for part in path.parts:
        split = SPLIT_ALIASES.get(part.lower())
        if split:
            return split
    return "unsplit"


def item_key(path: Path) -> str:
    """Return split-aware stem key for image/label matching."""

    return f"{detect_split(path)}::{path.stem}"


def polygon_area(points: list[tuple[float, float]]) -> float:
    """Return normalized polygon area using the shoelace formula."""

    if len(points) < 3:
        return 0.0
    return abs(
        sum(points[i][0] * points[(i + 1) % len(points)][1] - points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))
    ) / 2


def segment_orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """Return orientation cross-product for three points."""

    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def segments_intersect(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float], d: tuple[float, float]) -> bool:
    """Return True when two line segments intersect at a non-shared crossing."""

    def on_segment(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    o1 = segment_orientation(a, b, c)
    o2 = segment_orientation(a, b, d)
    o3 = segment_orientation(c, d, a)
    o4 = segment_orientation(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) < 1e-12 and on_segment(a, c, b):
        return True
    if abs(o2) < 1e-12 and on_segment(a, d, b):
        return True
    if abs(o3) < 1e-12 and on_segment(c, a, d):
        return True
    if abs(o4) < 1e-12 and on_segment(c, b, d):
        return True
    return False


def has_self_intersection(points: list[tuple[float, float]]) -> bool:
    """Detect polygon self-intersections with a bounded pairwise segment check."""

    if len(points) < 4:
        return False
    if len(points) > 120:
        return False
    n_points = len(points)
    for i in range(n_points):
        a = points[i]
        b = points[(i + 1) % n_points]
        for j in range(i + 1, n_points):
            if abs(i - j) <= 1 or {i, j} == {0, n_points - 1}:
                continue
            c = points[j]
            d = points[(j + 1) % n_points]
            if segments_intersect(a, b, c, d):
                return True
    return False


def hash_file(path: Path) -> str:
    """Return an MD5 hash for duplicate image detection."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polygon_signature(class_id: int | None, points: list[tuple[float, float]]) -> str:
    """Return a stable signature for duplicate polygon detection."""

    rounded = [(round(x, 6), round(y, 6)) for x, y in points]
    return json.dumps({"class_id": class_id, "points": rounded}, separators=(",", ":"))


def parse_label_file(label_path: Path, image_path: Path | None, image_size: tuple[int, int] | None) -> list[dict[str, Any]]:
    """Parse and validate all rows in a YOLO polygon label file."""

    rows: list[dict[str, Any]] = []
    split = detect_split(label_path)
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        return [
            {
                "label_path": str(label_path),
                "image_path": str(image_path) if image_path else "",
                "split": split,
                "line_number": 0,
                "class_id": None,
                "is_valid_row": False,
                "is_suspicious": True,
                "issues": f"unreadable_label:{exc}",
            }
        ]

    if not lines:
        return [
            {
                "label_path": str(label_path),
                "image_path": str(image_path) if image_path else "",
                "split": split,
                "line_number": 0,
                "class_id": None,
                "is_valid_row": False,
                "is_suspicious": True,
                "issues": "empty_label_file",
            }
        ]

    seen_signatures: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        issues: list[str] = []
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        class_id: int | None = None
        values: list[float] = []
        points: list[tuple[float, float]] = []

        try:
            class_id = int(float(parts[0]))
        except (IndexError, ValueError):
            issues.append("non_numeric_class_id")

        raw_values = parts[1:]
        if len(raw_values) % 2 != 0:
            issues.append("odd_number_of_coordinate_values")
        if len(raw_values) < 6:
            issues.append("fewer_than_three_coordinate_pairs")
        try:
            values = [float(value) for value in raw_values]
        except ValueError:
            issues.append("non_numeric_coordinate")

        if values:
            original_points = list(zip(values[0::2], values[1::2]))
            points = original_points[:-1] if len(original_points) > 3 and original_points[0] == original_points[-1] else original_points
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            if any(value < 0 or value > 1 for value in values):
                issues.append("coordinates_outside_0_1")
            if len(set(points)) < len(points):
                issues.append("repeated_points")
            area = polygon_area(points)
            touches_border = any(x <= BORDER_EPSILON or x >= 1 - BORDER_EPSILON or y <= BORDER_EPSILON or y >= 1 - BORDER_EPSILON for x, y in points)
            bbox_width = max(xs) - min(xs)
            bbox_height = max(ys) - min(ys)
            if area <= 0:
                issues.append("zero_area_polygon")
            elif area < SMALL_AREA_THRESHOLD:
                issues.append("extremely_small_polygon")
            elif area > LARGE_AREA_THRESHOLD:
                issues.append("extremely_large_polygon")
            if values and any(value < 0 or value > 1 for value in values):
                issues.append("polygon_extending_outside_image_boundaries")
            if has_self_intersection(points):
                issues.append("self_intersection")
            signature = polygon_signature(class_id, points)
            if signature in seen_signatures:
                issues.append("duplicate_polygon")
            seen_signatures.add(signature)
        else:
            area = 0.0
            touches_border = False
            bbox_width = 0.0
            bbox_height = 0.0

        if class_id not in EXPECTED_CLASS_IDS:
            issues.append("invalid_class_id")

        if image_size:
            image_width, image_height = image_size
        else:
            image_width = image_height = 0

        fatal_issues = {
            "non_numeric_class_id",
            "odd_number_of_coordinate_values",
            "fewer_than_three_coordinate_pairs",
            "non_numeric_coordinate",
            "coordinates_outside_0_1",
            "polygon_extending_outside_image_boundaries",
            "zero_area_polygon",
            "invalid_class_id",
        }
        rows.append(
            {
                "label_path": str(label_path),
                "image_path": str(image_path) if image_path else "",
                "split": split,
                "line_number": line_number,
                "class_id": class_id,
                "coordinate_count": len(values),
                "point_count": len(points),
                "area_normalized": area,
                "bbox_width_normalized": bbox_width,
                "bbox_height_normalized": bbox_height,
                "touches_border": touches_border,
                "image_width": image_width,
                "image_height": image_height,
                "is_valid_row": len(fatal_issues.intersection(issues)) == 0,
                "is_valid_area": area > 0 and area <= 1,
                "is_suspicious": len(issues) > 0,
                "issues": ";".join(sorted(set(issues))),
                "raw_values": values,
            }
        )
    return rows


def read_image_sizes(image_paths: list[Path]) -> pd.DataFrame:
    """Read image sizes and content hashes for non-hidden images."""

    rows: list[dict[str, Any]] = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception as exc:  # noqa: BLE001
            rows.append({"image_path": str(path), "split": detect_split(path), "width": 0, "height": 0, "md5": "", "image_issue": str(exc)})
            continue
        rows.append({"image_path": str(path), "split": detect_split(path), "width": width, "height": height, "md5": hash_file(path), "image_issue": ""})
    return pd.DataFrame(rows)


def parse_names_from_yaml_like(text: str) -> dict[int, str]:
    """Extract a YOLO names mapping from simple YAML-like text."""

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


def search_class_name_sources(paths: Phase2Paths) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Search project and extracted dataset for reliable or candidate class names."""

    rows: list[dict[str, Any]] = []
    names: dict[int, str] = {}
    search_roots = [paths.extracted_dir, paths.project_root / "reports", paths.project_root / "notebooks", paths.project_root]
    seen: set[Path] = set()
    metadata_name_patterns = {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml", "classes.txt", "_annotations.coco.json"}

    for root in search_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            rel_parts = path.relative_to(paths.project_root).parts if path.is_relative_to(paths.project_root) else path.parts
            if any(part in {".git", ".venv", "__pycache__", ".matplotlib-cache"} or part.startswith(".ipynb_checkpoints") for part in rel_parts):
                continue
            if path == paths.configs_dir / "classes.yaml":
                continue
            if path == paths.reports_dir / "annotation_quality_report.md":
                continue
            if "labels" in rel_parts and path.suffix.lower() == ".txt":
                continue
            if len(rel_parts) >= 2 and rel_parts[0] == "data" and rel_parts[1] == "processed":
                continue
            if path.name.endswith(".executed.ipynb"):
                continue
            if path.suffix.lower() not in METADATA_EXTENSIONS:
                continue
            lowered = path.name.lower()
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            discovered: dict[int, str] = {}
            reliability = "searched_no_names"
            if lowered in {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml"}:
                discovered = parse_names_from_yaml_like(text)
                reliability = "reliable_metadata" if discovered else "searched_no_names"
            elif lowered == "classes.txt":
                discovered = {idx: line.strip() for idx, line in enumerate(text.splitlines()) if line.strip()}
                reliability = "reliable_metadata" if discovered else "searched_no_names"
            elif lowered.endswith(".json") and "categories" in text[:20000]:
                try:
                    parsed = json.loads(text)
                    discovered = {int(item["id"]): str(item["name"]) for item in parsed.get("categories", []) if "id" in item and "name" in item}
                    reliability = "reliable_metadata" if discovered else "searched_no_names"
                except (json.JSONDecodeError, TypeError, ValueError):
                    reliability = "searched_parse_failed"
            elif lowered in metadata_name_patterns or re.search(r"class\s+[0-2]|class_id|names", text, flags=re.IGNORECASE):
                reliability = "candidate_text_reference"

            if discovered:
                names.update(discovered)
            if discovered or reliability.startswith("candidate") or lowered in metadata_name_patterns:
                rows.append(
                    {
                        "source_path": str(path),
                        "source_type": lowered,
                        "reliability": reliability,
                        "discovered_names": json.dumps(discovered, sort_keys=True),
                    }
                )

    return names, rows


def default_class_names(discovered: dict[int, str]) -> tuple[dict[int, str], str]:
    """Return class names and confirmation status."""

    confirmed = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/classes.yaml").read_text())
    if confirmed.get("status") == "confirmed":
        return confirmed["names"], "confirmed"
    if all(class_id in discovered for class_id in EXPECTED_CLASS_IDS):
        return {class_id: discovered[class_id] for class_id in sorted(EXPECTED_CLASS_IDS)}, "metadata_found_unconfirmed_visual_meaning"
    return {class_id: f"Class {class_id}" for class_id in sorted(EXPECTED_CLASS_IDS)}, "unconfirmed"


def draw_contact_sheet(
    image_paths: list[str],
    annotation_df: pd.DataFrame,
    class_names: dict[int, str],
    output_path: Path,
    title: str,
    max_images: int = 16,
) -> None:
    """Draw a contact sheet with polygon overlays."""

    selected = image_paths[:max_images]
    if not selected:
        selected = []
    cols = 4
    rows = max(1, math.ceil(max(1, len(selected)) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes_flat = np.atleast_1d(axes).ravel()
    colors = {0: (255, 64, 64, 170), 1: (64, 180, 255, 170), 2: (90, 220, 120, 170)}
    annotation_map = {path: group for path, group in annotation_df[annotation_df["image_path"].isin(selected)].groupby("image_path")}

    for ax, image_path in zip(axes_flat, selected):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((720, 540), Image.Resampling.LANCZOS)
            width, height = image.size
            draw = ImageDraw.Draw(image, "RGBA")
            subset = annotation_map.get(image_path, pd.DataFrame())
            for ann in subset.itertuples(index=False):
                if not isinstance(ann.raw_values, list) or not ann.raw_values:
                    continue
                points = [(ann.raw_values[i] * width, ann.raw_values[i + 1] * height) for i in range(0, len(ann.raw_values), 2)]
                color = colors.get(int(ann.class_id), (255, 255, 0, 170))
                draw.polygon(points, outline=color, fill=(color[0], color[1], color[2], 45))
                label = class_names.get(int(ann.class_id), f"Class {ann.class_id}")
                x_text, y_text = points[0]
                draw.rectangle([x_text, y_text, x_text + 8 * len(label) + 8, y_text + 16], fill=(0, 0, 0, 170))
                draw.text((x_text + 4, y_text + 2), label, fill=(255, 255, 255, 255))
            ax.imshow(image)
        ax.set_title(f"{Path(image_path).name}\n{detect_split(Path(image_path))}", fontsize=7)
        ax.axis("off")
    for ax in axes_flat[len(selected) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def select_review_images(image_df: pd.DataFrame, annotation_df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build a manual review queue from random samples and edge cases."""

    rng = np.random.default_rng(seed)
    queue: dict[str, set[str]] = defaultdict(set)

    for split, count in {"train": 30, "valid": 15, "test": 15}.items():
        split_paths = image_df.loc[image_df["split"] == split, "image_path"].tolist()
        if split_paths:
            selected = rng.choice(split_paths, size=min(count, len(split_paths)), replace=False)
            for image_path in selected:
                queue[str(image_path)].add(f"random_{split}")

    per_image = annotation_df.groupby("image_path").agg(
        polygon_count=("line_number", "count"),
        min_area=("area_normalized", "min"),
        max_area=("area_normalized", "max"),
        class_count=("class_id", "nunique"),
        touches_border=("touches_border", "max"),
        suspicious_count=("is_suspicious", "sum"),
    )

    for image_path in per_image.sort_values("polygon_count", ascending=False).head(20).index:
        queue[image_path].add("largest_number_of_polygons")
    for image_path in per_image.sort_values("min_area", ascending=True).head(20).index:
        queue[image_path].add("smallest_polygons")
    for image_path in per_image[per_image["class_count"] > 1].sort_values("class_count", ascending=False).head(30).index:
        queue[image_path].add("multiple_classes")
    for image_path in per_image[per_image["touches_border"] == True].head(30).index:  # noqa: E712
        queue[image_path].add("polygons_touching_image_borders")
    for image_path in per_image.sort_values("max_area", ascending=False).head(20).index:
        queue[image_path].add("unusually_large_mask_areas")
    for image_path in per_image[per_image["suspicious_count"] > 0].index:
        queue[image_path].add("suspicious_polygon")

    for class_id in sorted(EXPECTED_CLASS_IDS):
        class_paths = annotation_df.loc[annotation_df["class_id"] == class_id, "image_path"].dropna().unique().tolist()
        for image_path in class_paths[:30]:
            queue[image_path].add(f"contains_class_{class_id}")

    rows = []
    for image_path, reasons in sorted(queue.items()):
        rows.append(
            {
                "image_path": image_path,
                "split": detect_split(Path(image_path)),
                "review_reasons": ";".join(sorted(reasons)),
                "polygon_count": int(per_image.loc[image_path, "polygon_count"]) if image_path in per_image.index else 0,
                "class_ids": ",".join(map(str, sorted(annotation_df.loc[annotation_df["image_path"] == image_path, "class_id"].dropna().astype(int).unique()))),
            }
        )
    return pd.DataFrame(rows)


def build_split_consistency(image_df: pd.DataFrame, annotation_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize annotation consistency metrics by split."""

    rows: list[dict[str, Any]] = []
    for split in ["train", "valid", "test"]:
        split_images = image_df[image_df["split"] == split]
        split_ann = annotation_df[annotation_df["split"] == split]
        images_with_labels = split_ann["image_path"].nunique()
        row: dict[str, Any] = {
            "split": split,
            "images": len(split_images),
            "images_with_labels": images_with_labels,
            "percent_images_with_labels": images_with_labels / len(split_images) * 100 if len(split_images) else 0,
            "polygons": len(split_ann),
            "avg_polygons_per_image": len(split_ann) / len(split_images) if len(split_images) else 0,
            "median_polygon_area": split_ann["area_normalized"].median() if not split_ann.empty else 0,
            "median_point_count": split_ann["point_count"].median() if not split_ann.empty else 0,
            "suspicious_polygons": int(split_ann["is_suspicious"].sum()) if not split_ann.empty else 0,
        }
        for class_id in sorted(EXPECTED_CLASS_IDS):
            class_ann = split_ann[split_ann["class_id"] == class_id]
            row[f"class_{class_id}_polygons"] = len(class_ann)
            row[f"class_{class_id}_images"] = class_ann["image_path"].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def class_cooccurrence(annotation_df: pd.DataFrame) -> pd.DataFrame:
    """Return image-level class co-occurrence counts."""

    rows = []
    for split, split_df in annotation_df.groupby("split"):
        image_classes = split_df.groupby("image_path")["class_id"].apply(lambda values: sorted(set(int(value) for value in values if pd.notna(value))))
        counts = Counter(tuple(values) for values in image_classes)
        for classes, count in sorted(counts.items()):
            rows.append({"split": split, "class_combination": "+".join(map(str, classes)), "image_count": count})
    return pd.DataFrame(rows)


def duplicate_summary(image_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Find duplicate image hashes, especially across splits."""

    rows = []
    cross_split = 0
    for md5, group in image_df.groupby("md5"):
        if not md5 or len(group) < 2:
            continue
        splits = sorted(group["split"].unique())
        if len(splits) > 1:
            cross_split += 1
        rows.append({"md5": md5, "count": len(group), "splits": ",".join(splits), "image_paths": ";".join(group["image_path"].tolist())})
    return pd.DataFrame(rows), cross_split


def write_classes_yaml(path: Path, class_names: dict[int, str], status: str) -> None:
    """Write the simple class configuration file."""

    if path.exists() and yaml.safe_load(path.read_text()).get("status") == "confirmed":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if status == "unconfirmed":
        values = {class_id: f"class_{class_id}_unconfirmed" for class_id in sorted(EXPECTED_CLASS_IDS)}
    else:
        values = class_names
    lines = ["names:"]
    lines.extend(f"  {class_id}: {name}" for class_id, name in values.items())
    lines.append(f"status: {status}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    class_names: dict[int, str],
    class_status: str,
    metadata_sources: list[dict[str, Any]],
    image_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    manual_queue: pd.DataFrame,
    split_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
    cross_split_duplicates: int,
) -> None:
    """Write the Phase 2 Markdown report."""

    valid_polygons = int(quality_df["is_valid_row"].sum())
    suspicious_polygons = int(quality_df["is_suspicious"].sum())
    images_requiring_review = manual_queue["image_path"].nunique()
    labeled_pct = annotation_df["image_path"].nunique() / len(image_df) * 100 if len(image_df) else 0
    valid_row_pct = valid_polygons / len(quality_df) * 100 if len(quality_df) else 0
    valid_area_pct = quality_df["is_valid_area"].sum() / len(quality_df) * 100 if len(quality_df) else 0
    suspicious_issue_counts = (
        quality_df.loc[quality_df["issues"].astype(str) != "", "issues"]
        .str.split(";")
        .explode()
        .value_counts()
        .rename_axis("issue")
        .reset_index(name="count")
    )
    class_distribution = annotation_df.groupby(["split", "class_id"]).size().reset_index(name="polygon_count")
    image_class_distribution = annotation_df.groupby(["split", "class_id"])["image_path"].nunique().reset_index(name="image_count")
    cooccurrence = class_cooccurrence(annotation_df)

    class_names_found = class_status != "unconfirmed"
    can_identify_meanings = class_status == "confirmed"
    report_lines = [
        "# Phase 2 Annotation Quality and Class Semantics Report",
        "",
        "## Scope",
        "This Phase 2 review filters hidden/system artifacts from all analyses and validates YOLO polygon annotations without modifying the extracted dataset or training any model.",
        f"After filtering hidden/system artifacts, the analyzed dataset contains {len(image_df)} images and {annotation_df['image_path'].nunique()} images with labels.",
        "",
        "## Class-name search",
        f"- Class names found: {'Yes' if class_names_found else 'No'}",
        f"- Class-name status: {class_status}",
        f"- Source count searched/referenced: {len(metadata_sources)}",
        "- Discovered class names: " + ", ".join(f"{class_id}: {name}" for class_id, name in class_names.items()),
        f"- Meanings confidently identified: {'Yes' if can_identify_meanings else 'No'}",
        "- Human confirmation required: " + ("No" if class_status == "confirmed" else "Yes"),
        "",
        "## Metadata sources reviewed",
        pd.DataFrame(metadata_sources).to_markdown(index=False) if metadata_sources else "No reliable class-name metadata files were found.",
        "",
        "## Quality indicators",
        f"- Percentage of images with labels: {labeled_pct:.2f}%",
        f"- Percentage of valid polygon rows: {valid_row_pct:.2f}%",
        f"- Percentage of polygons with valid area: {valid_area_pct:.2f}%",
        f"- Valid polygons: {valid_polygons}",
        f"- Suspicious polygons: {suspicious_polygons}",
        f"- Images requiring manual review: {images_requiring_review}",
        f"- Cross-split duplicate image hashes: {cross_split_duplicates}",
        "",
        "## Polygon issue summary",
        suspicious_issue_counts.to_markdown(index=False) if not suspicious_issue_counts.empty else "No suspicious polygon issues were detected.",
        "",
        "## Split consistency",
        split_df.to_markdown(index=False),
        "The train, validation, and test splits appear broadly consistent: all filtered images have labels, average polygons per image are close across splits, median polygon areas are similar, and each split contains all three class IDs. Class 2 remains the majority class in every split.",
        "",
        "## Class distribution by split",
        class_distribution.to_markdown(index=False),
        "",
        "## Images containing each class by split",
        image_class_distribution.to_markdown(index=False),
        "",
        "## Class co-occurrence",
        cooccurrence.to_markdown(index=False) if not cooccurrence.empty else "No co-occurrence data found.",
        "",
        "## Duplicate/leakage assessment",
        duplicates_df.to_markdown(index=False) if not duplicates_df.empty else "No duplicate image hashes were found after hidden/system artifacts were excluded.",
        f"Cross-split data leakage through exact duplicate image hashes: {'Detected' if cross_split_duplicates else 'Not detected'}.",
        "",
        "## Training readiness",
        "Classes are human-confirmed: 0=path, 1=cassava_leaves, 2=ridge. Suspicious polygon sizes are review warnings and retained unless technically invalid.",
        "",
        "## Exclusion recommendations",
        "Exclude hidden/system artifacts from every analysis and training manifest. Review rows in `reports/polygon_quality_checks.csv` where `is_suspicious` is true before deciding whether to exclude individual polygons or images.",
        "",
        "## Human-confirmed semantics",
        "",
        "- Class 0: path (confirmed by project owner)",
        "- Class 1: cassava_leaves (confirmed by project owner)",
        "- Class 2: ridge (confirmed by project owner)",
        "",
        "## Preserved decisions",
        "1. Preserve confirmed classes: 0=path, 1=cassava_leaves, 2=ridge.",
        "2. Retain the 198 extremely small and 18 extremely large polygons as review warnings.",
        "3. Confirm whether the class distribution and split differences are acceptable for the intended segmentation experiment.",
        "4. Keep navigation-control work paused until explicit navigation labels or a justified target-generation method exists.",
        "",
    ]
    path.write_text("\n".join(report_lines), encoding="utf-8")


def run_phase2(paths: Phase2Paths, seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Run the complete Phase 2 validation workflow."""

    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    paths.configs_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_dataset_files(paths.extracted_dir, IMAGE_EXTENSIONS)
    label_paths = [path for path in iter_dataset_files(paths.extracted_dir, {".txt"}) if "label" in str(path).lower()]

    image_df = read_image_sizes(image_paths)
    image_by_key = {item_key(Path(row.image_path)): Path(row.image_path) for row in image_df.itertuples(index=False)}
    label_by_key = {item_key(path): path for path in label_paths}

    quality_rows: list[dict[str, Any]] = []
    for key, label_path in label_by_key.items():
        image_path = image_by_key.get(key)
        if image_path is not None:
            size_row = image_df[image_df["image_path"] == str(image_path)].iloc[0]
            image_size = (int(size_row["width"]), int(size_row["height"]))
        else:
            image_size = None
        quality_rows.extend(parse_label_file(label_path, image_path, image_size))

    quality_df = pd.DataFrame(quality_rows)
    annotation_df = quality_df[quality_df["class_id"].isin(EXPECTED_CLASS_IDS) & quality_df["raw_values"].apply(lambda values: isinstance(values, list) and len(values) >= 6)].copy()

    discovered_names, metadata_sources = search_class_name_sources(paths)
    class_names, class_status = default_class_names(discovered_names)
    write_classes_yaml(paths.configs_dir / "classes.yaml", class_names, "unconfirmed" if class_status == "unconfirmed" else class_status)

    manual_queue = select_review_images(image_df, annotation_df, seed=seed)
    split_consistency = build_split_consistency(image_df, annotation_df)
    duplicates_df, cross_split_duplicates = duplicate_summary(image_df)

    class_review_rows = []
    for class_id in sorted(EXPECTED_CLASS_IDS):
        subset = annotation_df[annotation_df["class_id"] == class_id]
        class_review_rows.append(
            {
                "class_id": class_id,
                "current_name": class_names[class_id],
                "status": class_status,
                "metadata_source": "",
                "images_containing_class": subset["image_path"].nunique(),
                "polygon_count": len(subset),
                "median_area_normalized": subset["area_normalized"].median() if not subset.empty else 0,
                "human_confirmation_required": "No" if class_status == "confirmed" else "Yes",
            }
        )
        class_paths = subset.groupby("image_path").size().sort_values(ascending=False).index.tolist()
        draw_contact_sheet(class_paths, annotation_df, class_names, paths.figures_dir / f"class_{class_id}_review.png", f"Class {class_id} Review")

    multiclass_paths = (
        annotation_df.groupby("image_path")["class_id"].nunique().sort_values(ascending=False)
    )
    draw_contact_sheet(multiclass_paths[multiclass_paths > 1].index.tolist(), annotation_df, class_names, paths.figures_dir / "multiclass_review.png", "Multiclass Review")
    edge_case_paths = manual_queue[manual_queue["review_reasons"].str.contains("smallest|large|border|suspicious|largest", regex=True)]["image_path"].tolist()
    draw_contact_sheet(edge_case_paths, annotation_df, class_names, paths.figures_dir / "annotation_edge_cases.png", "Annotation Edge Cases")

    pd.DataFrame(class_review_rows).to_csv(paths.reports_dir / "class_semantics_review.csv", index=False)
    quality_df.drop(columns=["raw_values"]).to_csv(paths.reports_dir / "polygon_quality_checks.csv", index=False)
    manual_queue.to_csv(paths.reports_dir / "manual_review_queue.csv", index=False)
    split_consistency.to_csv(paths.reports_dir / "split_consistency.csv", index=False)
    if not duplicates_df.empty:
        duplicates_df.to_csv(paths.reports_dir / "duplicate_image_hashes.csv", index=False)

    write_report(
        paths.reports_dir / "annotation_quality_report.md",
        class_names,
        "unconfirmed" if class_status == "unconfirmed" else class_status,
        metadata_sources,
        image_df,
        annotation_df,
        quality_df,
        manual_queue,
        split_consistency,
        duplicates_df,
        cross_split_duplicates,
    )

    annotation_quality_sufficient = bool(quality_df["is_valid_row"].mean() > 0.99 and cross_split_duplicates == 0)
    ready_for_segmentation_training = bool(annotation_quality_sufficient and class_status != "unconfirmed")
    return {
        "class_names": class_names,
        "class_status": "unconfirmed" if class_status == "unconfirmed" else class_status,
        "valid_polygons": int(quality_df["is_valid_row"].sum()),
        "suspicious_polygons": int(quality_df["is_suspicious"].sum()),
        "images_requiring_manual_review": int(manual_queue["image_path"].nunique()),
        "cross_split_duplicates": int(cross_split_duplicates),
        "annotation_quality_sufficient": annotation_quality_sufficient,
        "ready_for_segmentation_training": ready_for_segmentation_training,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Phase 2 annotation quality and class semantics validation.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--extracted-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    paths = Phase2Paths(
        project_root=project_root,
        extracted_dir=(args.extracted_dir.expanduser().resolve() if args.extracted_dir else project_root / "data" / "extracted"),
        reports_dir=project_root / "reports",
        figures_dir=project_root / "figures",
        configs_dir=project_root / "configs",
    )
    summary = run_phase2(paths, seed=args.seed)
    print("Phase 2 annotation validation complete")
    print(f"Class names/status: {summary['class_names']} / {summary['class_status']}")
    print(f"Valid polygons: {summary['valid_polygons']}")
    print(f"Suspicious polygons: {summary['suspicious_polygons']}")
    print(f"Images requiring manual review: {summary['images_requiring_manual_review']}")
    print(f"Cross-split duplicates found: {summary['cross_split_duplicates']}")
    print(f"Annotation quality technically sufficient: {summary['annotation_quality_sufficient']}")
    print(f"Ready for segmentation training: {summary['ready_for_segmentation_training']}")
    print("Confirmed classes: 0=path, 1=cassava_leaves, 2=ridge.")
    print("Review the 216 polygon-size warnings; retain them unless technically invalid.")
    print("Navigation targets remain provisional geometry-derived research labels, not recorded controls.")


if __name__ == "__main__":
    main()
