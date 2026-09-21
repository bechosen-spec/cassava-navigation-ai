# Phase B results validation

Regression metrics were independently recomputed from every saved test-prediction file and saved under `validated_results/regression/`. Classification metrics and the confusion matrix were recomputed from saved logistic predictions using fixed class order left, forward, right, under `validated_results/classification/`. The original exports remain under `input_results/phase_B_results_original/`.

Important comparability: the saved 42 regression image IDs overlap across all available regression prediction files. Classical RF and ANFIS use oracle ground-truth-mask features, while the CNNs use images; their scores therefore do not establish equivalent deployability. Directional classification has only 16 test samples, so per-class estimates are highly uncertain. Segmentation was not trained and has no evaluation outputs.
