# Phase C Input Audit

| Artifact | Phase | Path | Purpose | Generated | Used in Phase C | Limitation |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset audit / annotation reports | 1-2 | reports/dataset_audit_report.md; reports/annotation_quality_report.md | Integrity, polygons, classes and split evidence | Yes | Context and limitations | No proof of temporal/site independence |
| Ground-truth feature table | A | phase_B_training_dataset/tabular/features_ground_truth.csv | Oracle segmentation features | Yes | Interpretation of tabular models | Not deployable mask source |
| Phase B prediction exports | B | input_results/phase_B_results_original | Per-image frozen model evidence | Yes | Metric validation and paired statistics | 42 common test images |
| Segmentation baseline metrics | A | results/segmentation_baseline/metrics.csv | Schema/status record | No trained result | Limitation | status=not_trained |

Phase C reads preserved Phase B exports and writes only to `phase_C_results/`. The local segmentation baseline is explicitly not trained; no segmentation metric is invented.
