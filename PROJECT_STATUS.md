# Project Status

Last reviewed: 2026-09-22

## Milestones

- [x] Phase 1 — Dataset audit and validation
- [x] Phase 2 — Class and annotation-quality validation
- [x] Phase A — Segmentation, features and navigation-target workflow prepared
- [x] Phase B — Initial model training/export and independent results validation
- [x] Phase C — Comparative/statistical analysis and comprehensive technical documentation

## Current Phase

Phase C is complete for the supplied frozen evidence. It independently validates saved regression predictions, corrects classification metrics from hard labels, aligns all five regression families on a 42-image common test set, computes 5,000-repeat bootstrap intervals, performs paired Wilcoxon tests with Holm correction, creates high-resolution figures/tables, and produces the complete technical-report package. The segmentation baseline remains not trained.

## Verified Dataset and Outputs

- Original tree: approximately 803 images and 802 label files.
- Original splits: train 603 images / 602 labels; validation 120/120; test 80/80.
- Usable splits: train 602; validation 120; test 80.
- Format: YOLO polygon segmentation.
- Classes: `0=path`, `1=cassava_leaves`, `2=ridge`.
- Valid polygons: 28,655.
- Warnings: 216 polygons (198 extremely small; 18 extremely large).
- Manual review: 278 images.
- Exact cross-split duplicate hashes: 0.
- Ground-truth feature data: 802 rows and 412 total columns.
- Segmentation metrics/checkpoints: baseline records `not_trained`; no predictions or epoch logs were supplied.
- Predicted-mask features: absent; the Phase B package does not fabricate a substitute.
- Navigation targets: rebuilt reproducibly from the committed ground-truth feature table and committed geometry rules for dataset preparation; they remain provisional geometry-derived labels.

Approximate externally reported Phase A values are not promoted to verified metrics until their executed CSVs and checkpoints are imported.

## Known Limitations

- Provisional targets are geometry-derived, not recorded steering commands.
- Control-team validation is outstanding.
- Direct path-centre equivalents leak the continuous target formula.
- Ground-truth features are an oracle scenario.
- Same-model training predictions are in-sample without out-of-fold generation.
- Exact hashes do not rule out temporal/site correlation.
- Camera/controller calibration and physical safety validation are unavailable.

## Phase C Deliverables

- `notebooks/06_phase_C_comparative_analysis.ipynb`
- `src/phase_C_analysis.py`, `src/phase_C_statistics.py`, `src/phase_C_visualization.py`
- `phase_C_results/` validated metrics, corrected classification metrics, manifests, statistics, figures and indexes
- `reports/COMPLETE_RESEARCH_TECHNICAL_REPORT.md` plus DOCX/PDF
- `reports/PROJECT_EXPLAINED_SIMPLY.md`, objective traceability, figure/table catalogues and reproducibility guide
- `reports/CASSAVA_NAVIGATION_COMPLETE_TECHNICAL_REPORT.md` plus its complete DOCX/PDF rebuild, evidence manifest, indexes and completeness audit

## Next Action

Prepare a manuscript using the Phase C report, retaining all stated evidence boundaries. Import a genuine trained-segmentation/predicted-mask package only if a deployment-oriented downstream comparison is required.

Phase B must preserve splits, make all choices on train/validation only, separate oracle from deployable features, and audit target leakage.

## Git Status

- Repository: `bechosen-spec/cassava-navigation-ai`
- Branch: `main`
- Remote: `origin`
- Last completed milestone: Phase A Colab workflow preparation and dataset-root fixes.
- The documentation milestone is recorded by the Git commit that adds this file.
