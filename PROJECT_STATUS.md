# Project Status

Last reviewed: 2026-09-22

## Milestones

- [x] Phase 1 — Dataset audit and validation
- [x] Phase 2 — Class and annotation-quality validation
- [x] Phase A — Segmentation, features and navigation-target workflow prepared
- [x] Phase B — Initial model training/export and independent results validation
- [ ] Phase C — Comparative/statistical analysis and paper-ready results

## Current Phase

Phase B results reviewed from the supplied executed notebook and results ZIP. Classical oracle-feature models, ANFIS oracle-feature regression, and three image-regression models exported predictions; 33 evidence-backed research figures were generated. The segmentation baseline remains not trained.

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

## Next Action

1. Use `notebooks/05_phase_B_figures_recovery_colab.ipynb` only to recover missing segmentation predictions/logs or ANFIS visualization metadata from original checkpoints.
2. Run Phase C comparative/statistical analysis only after resolving comparability gaps and documenting timing conditions.
3. Preserve the oracle-versus-image feature distinction in all research writing.

Phase B must preserve splits, make all choices on train/validation only, separate oracle from deployable features, and audit target leakage.

## Git Status

- Repository: `bechosen-spec/cassava-navigation-ai`
- Branch: `main`
- Remote: `origin`
- Last completed milestone: Phase A Colab workflow preparation and dataset-root fixes.
- The documentation milestone is recorded by the Git commit that adds this file.
