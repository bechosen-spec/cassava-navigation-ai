# Phase B Colab guide

1. In Google Colab, upload and open `notebooks/04_phase_B_colab.ipynb`.
2. Select **Runtime → Change runtime type → GPU**.
3. Run the setup cell.
4. Upload `cassava_navigation_phase_B_dataset.zip` when prompted. The notebook extracts it, finds `phase_B_training_dataset`, and loads `tabular/regression_targets.csv` and `tabular/classification_targets.csv` automatically; do not edit file paths or merge CSVs.
5. Run the remaining cells in order. Validation data selects configurations and early stopping; do not use the test output to tune models.
6. At completion, download `/content/cassava_navigation_phase_B_results.zip` and the executed notebook from Colab's File menu.

If direct upload is unreliable, put the ZIP in Google Drive, mount Drive in the optional input cell, and set the one Drive ZIP variable. If the session disconnects, reconnect, run setup again, and reuse the already extracted package or upload the same ZIP again. Results are only valid when the notebook has completed its data-quality checks.

Notes: targets are provisional image-geometry research labels, not physical steering commands. The package contains oracle ground-truth-mask features; no executed predicted-mask feature table was available, so that exploratory experiment is skipped rather than substituted.

The first Phase B attempt failed before training because an older notebook searched only for the absent Phase A filename `navigation_targets.csv`. This corrected notebook reads the target tables actually packaged for Phase B and raises an error instead of downloading an empty results ZIP if required inputs or model metrics are unavailable.
