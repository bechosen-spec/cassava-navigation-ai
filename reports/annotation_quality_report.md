> Phase A update: classes confirmed (0=path, 1=cassava_leaves, 2=ridge). Historical nominal counts are 603/120/80; artifact filtering gives 602/120/80 usable images. The extra training image is a checkpoint duplicate, not an additional sample.

# Phase 2 Annotation Quality and Class Semantics Report

## Scope
This Phase 2 review filters hidden/system artifacts from all analyses and validates YOLO polygon annotations without modifying the extracted dataset or training any model.
After filtering hidden/system artifacts, the analyzed dataset contains 802 images and 802 images with labels.

## Class-name search
- Class names confirmed by human: Yes
- Class-name status: confirmed
- Source count searched/referenced: 6
- Confirmed class names: 0: path, 1: cassava_leaves, 2: ridge
- Meanings human-confirmed: Yes
- No further class confirmation required; preserve this mapping.

## Metadata sources reviewed
| source_path                                                                       | source_type                | reliability              | discovered_names   |
|:----------------------------------------------------------------------------------|:---------------------------|:-------------------------|:-------------------|
| /Users/macbook/Documents/cassava-navigation-ai/reports/class_distribution.csv     | class_distribution.csv     | candidate_text_reference | {}                 |
| /Users/macbook/Documents/cassava-navigation-ai/reports/class_semantics_review.csv | class_semantics_review.csv | candidate_text_reference | {}                 |
| /Users/macbook/Documents/cassava-navigation-ai/reports/dataset_audit_report.md    | dataset_audit_report.md    | candidate_text_reference | {}                 |
| /Users/macbook/Documents/cassava-navigation-ai/reports/manual_review_queue.csv    | manual_review_queue.csv    | candidate_text_reference | {}                 |
| /Users/macbook/Documents/cassava-navigation-ai/reports/polygon_quality_checks.csv | polygon_quality_checks.csv | candidate_text_reference | {}                 |
| /Users/macbook/Documents/cassava-navigation-ai/README.md                          | readme.md                  | candidate_text_reference | {}                 |

## Quality indicators
- Percentage of images with labels: 100.00%
- Percentage of valid polygon rows: 100.00%
- Percentage of polygons with valid area: 100.00%
- Valid polygons: 28655
- Suspicious polygons: 216
- Images requiring manual review: 278
- Cross-split duplicate image hashes: 0

## Polygon issue summary
| issue                   |   count |
|:------------------------|--------:|
| extremely_small_polygon |     198 |
| extremely_large_polygon |      18 |

## Split consistency
| split   |   images |   images_with_labels |   percent_images_with_labels |   polygons |   avg_polygons_per_image |   median_polygon_area |   median_point_count |   suspicious_polygons |   class_0_polygons |   class_0_images |   class_1_polygons |   class_1_images |   class_2_polygons |   class_2_images |
|:--------|---------:|---------------------:|-----------------------------:|-----------:|-------------------------:|----------------------:|---------------------:|----------------------:|-------------------:|-----------------:|-------------------:|-----------------:|-------------------:|-----------------:|
| train   |      602 |                  602 |                          100 |      21460 |                  35.6478 |            0.00167211 |                   48 |                   143 |               3988 |              591 |               3340 |              594 |              14132 |              601 |
| valid   |      120 |                  120 |                          100 |       4408 |                  36.7333 |            0.00155958 |                   46 |                    44 |                752 |              118 |                707 |              120 |               2949 |              120 |
| test    |       80 |                   80 |                          100 |       2787 |                  34.8375 |            0.00158819 |                   46 |                    29 |                561 |               78 |                447 |               78 |               1779 |               80 |
The train, validation, and test splits appear broadly consistent: all filtered images have labels, average polygons per image are close across splits, median polygon areas are similar, and each split contains all three class IDs. Class 2 remains the majority class in every split.

## Class distribution by split
| split   |   class_id |   polygon_count |
|:--------|-----------:|----------------:|
| test    |          0 |             561 |
| test    |          1 |             447 |
| test    |          2 |            1779 |
| train   |          0 |            3988 |
| train   |          1 |            3340 |
| train   |          2 |           14132 |
| valid   |          0 |             752 |
| valid   |          1 |             707 |
| valid   |          2 |            2949 |

## Images containing each class by split
| split   |   class_id |   image_count |
|:--------|-----------:|--------------:|
| test    |          0 |            78 |
| test    |          1 |            78 |
| test    |          2 |            80 |
| train   |          0 |           591 |
| train   |          1 |           594 |
| train   |          2 |           601 |
| valid   |          0 |           118 |
| valid   |          1 |           120 |
| valid   |          2 |           120 |

## Class co-occurrence
| split   | class_combination   |   image_count |
|:--------|:--------------------|--------------:|
| test    | 0+1+2               |            76 |
| test    | 0+2                 |             2 |
| test    | 1+2                 |             2 |
| train   | 0+1                 |             1 |
| train   | 0+1+2               |           582 |
| train   | 0+2                 |             8 |
| train   | 1+2                 |            11 |
| valid   | 0+1+2               |           118 |
| valid   | 1+2                 |             2 |

## Duplicate/leakage assessment
No duplicate image hashes were found after hidden/system artifacts were excluded.
Cross-split data leakage through exact duplicate image hashes: Not detected.

## Training readiness
The classes are human-confirmed and the dataset is technically usable. Retain 198 extremely small and 18 extremely large polygons as warnings, not automatic errors.

## Exclusion recommendations
Exclude hidden/system artifacts from every analysis and training manifest. Review rows in `reports/polygon_quality_checks.csv` where `is_suspicious` is true before deciding whether to exclude individual polygons or images.

## Manual confirmation section

Class 0:
Observed visual content:
Proposed meaning:
Confidence:
Human confirmation required: No (confirmed by project owner)

Class 1:
Observed visual content:
Proposed meaning:
Confidence:
Human confirmation required: No (confirmed by project owner)

Class 2:
Observed visual content:
Proposed meaning:
Confidence:
Human confirmation required: No (confirmed by project owner)

## Decisions required before training
1. Preserve the human-confirmed class mapping.
2. Retain suspicious polygons unless technically invalid.
3. Confirm whether the class distribution and split differences are acceptable for the intended segmentation experiment.
4. Keep navigation-control work paused until explicit navigation labels or a justified target-generation method exists.
