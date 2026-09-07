# Project Status

Last reviewed: 2026-09-07

## Milestones

- [x] Phase 1 — Dataset audit and validation
- [x] Phase 2 — Class and annotation-quality validation
- [~] Phase A — Workflow and ground-truth features prepared; executed GPU artifacts awaiting import and verification
- [ ] Phase B — Classical ML, ANFIS, and deep-learning navigation prediction
- [ ] Phase C — Comparative/statistical analysis and paper-ready results

## Current Phase

Phase A artifact reconciliation. The self-contained GPU notebook exists and an external run has been reported, but this checkout contains only ground-truth features and explicit `not_trained` segmentation status files. It has no trained checkpoints, predicted-mask features, navigation targets, or executed metric outputs.

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
- Trained segmentation metrics/checkpoints: N/A in this checkout.
- Predicted-mask features and navigation targets: absent.

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

1. Import the executed `03_phase_A_colab.ipynb` and `cassava_navigation_phase_A_results.zip`.
2. Verify their inventories, checksums, checkpoints, metrics, figures, IDs, rows, and splits.
3. Replace local `not_trained` status artifacts only with verified executed outputs.
4. Start **Phase B — Classical ML + ANFIS + CNN/Transfer Learning** in a future `notebooks/04_phase_B_colab.ipynb`.

Phase B must preserve splits, make all choices on train/validation only, separate oracle from deployable features, and audit target leakage.

## Git Status

- Repository: `bechosen-spec/cassava-navigation-ai`
- Branch: `main`
- Remote: `origin`
- Last completed milestone: Phase A Colab workflow preparation and dataset-root fixes.
- The documentation milestone is recorded by the Git commit that adds this file.
