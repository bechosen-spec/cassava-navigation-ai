# Colab Phase A notebook validation

Notebook: `notebooks/03_phase_A_colab.ipynb`.

Status: **local validation passed; Colab GPU execution pending**. No segmentation training or navigation-model training was performed locally. No executed notebook, checkpoints or evaluation results are claimed by these checks.

The notebook contains 32 cells (17 code cells) and five embedded reusable Python modules. All required workflow sections are present: settings/GPU detection, upload/Drive input, safe extraction/root discovery, pretraining validation, confirmed classes, one compatible segmentation baseline, validation/test metrics, predictions, GT/predicted features, training-derived provisional thresholds, target/feature analysis, reports, archive verification, download and structured actual-results summary.

Passed checks:

- Notebook JSON and nbformat schema validation.
- Python compilation of every code cell and all five embedded modules.
- No local Mac paths in Colab code.
- Temporary synthetic-fixture checks for ZIP traversal rejection, arbitrary ZIP filenames, `val` discovery, generated-root exclusion, and ignored checkpoint artifacts.
- Small/large polygons retained; invalid class rows excluded only from derived copies; original labels unchanged.
- Feature metadata retained; overlapping polygons union correctly; absent path centres remain NaN with missing flags and yield `stop_or_uncertain`.
- Full-project ZIPs containing both original data and `data/processed/segmentation_dataset` select the original source root and report the ignored generated root.
- Metric extraction accepts raw API dictionaries and preserves unavailable fields as NaN.

Reproduce these checks (no model training):

```bash
.venv/bin/python scripts/validate_phase_a_notebook.py
```

Limitations: these are schema, syntax and targeted helper checks, not full Colab execution. Dependency installation, GPU memory availability, checkpoint downloads, training, validation API outputs and the complete scientific-result archive must be verified by running all notebook cells in Colab. Training failures remain explicit. Unavailable metrics/figures are reported rather than fabricated. Return the executed `.ipynb` plus `cassava_navigation_phase_A_results.zip` for review.
