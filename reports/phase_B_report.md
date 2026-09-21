# Phase B report — validated exported results

## Completed evidence

The executed Phase B notebook completed classical regression/classification using oracle ground-truth-mask features, ANFIS regression using oracle features, and image-based regression using a Custom CNN, MobileNetV3, and ResNet18. The notebook reports eligible regression counts of 297/56/42 and directional-classification counts of 116/22/16 across train/validation/test. The primary directional class order is left, forward, right; `stop_or_uncertain` is not a verified physical STOP command.

## Independent validation

Saved test predictions were recomputed in `validated_results/regression/recomputed_test_metrics.csv`. The selected all-non-leaking oracle random forest has MAE 0.4506, RMSE 0.5421, and R² 0.3024 on 42 saved test rows. Classification metrics were recomputed from the 16 saved all-non-leaking logistic test predictions in the fixed left/forward/right order; accuracy is 0.7500 and macro F1 is 0.7044. Original exports are preserved under `input_results/phase_B_results_original/`.

## Scope and limitations

Classical and ANFIS outputs are oracle-feature experiments, not predicted-mask deployment results. The three image models share 42 saved test IDs with the selected RF and ANFIS predictions, but the differing feature sources prevent a simple deployability comparison. Segmentation is explicitly `not_trained` in stored metrics, so segmentation curves, mAP, predicted-mask panels, and failure cases are unavailable. Classification has only 16 test samples, making per-class estimates uncertain. Derived navigation targets are not recorded steering commands or validated robot controls.

## Figures and Phase C

`research_figures/` contains 33 high-resolution, evidence-backed PNG/PDF figures and a 40-slot index. Seven requested figures are marked as requiring additional evaluation rather than reconstructed from final metrics. The complete portable package is `cassava_navigation_research_figures.zip`. Phase C is not complete: it requires any needed recovery execution, pre-specified comparative statistics, and clear oracle-versus-deployable reporting.
