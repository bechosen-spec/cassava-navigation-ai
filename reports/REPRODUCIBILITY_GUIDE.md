# Reproducibility Guide

1. Create the project Python environment from `requirements.txt` (Phase C uses pandas, NumPy, SciPy, Matplotlib and nbformat).
2. Keep `input_results/phase_B_results_original/` unchanged; it is the frozen Phase B evidence.
3. Run `MPLCONFIGDIR=/private/tmp/cassava-mpl ./.venv/bin/python src/phase_C_analysis.py`.
4. Run `scripts/build_phase_c_reports.py` to assemble reports and packages.
5. Open `notebooks/06_phase_C_comparative_analysis.ipynb` for the orchestration route.

Dataset audit, annotation validation, Phase A/feature preparation and Phase C run locally. Segmentation and model training may require GPU according to their notebooks, but Phase C itself uses saved predictions and is CPU-sufficient. Results are in `phase_C_results/`; reports are in `reports/`.
