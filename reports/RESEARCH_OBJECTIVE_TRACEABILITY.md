# Research Objective Traceability

| Objective | Evidence and method | Finding | Limitation | Status |
| --- | --- | --- | --- | --- |
| Validate cassava-field dataset | Audit, polygon validation, split checks | 802 usable labelled samples; 3 confirmed semantic classes | Adjacent-frame/site correlation not ruled out | Fully addressed |
| Model ANFIS navigation predictor | Five selected oracle features; 3 membership functions; 243 rules | Test MAE 0.589 on 42 exported images | Oracle masks and derived target | Partially addressed |
| Model image navigation predictors | Custom CNN, MobileNetV3, ResNet18 per-image exports | MobileNetV3 had lowest image-model MAE (0.592) | 42-image test; no robot validation | Partially addressed |
| Compare model families fairly | Common IDs, bootstrap and Holm-adjusted paired tests | RF has lower paired AE than the image models in this export | Oracle and image-only settings differ | Partially addressed |
| Produce reproducible results | Phase C script/notebook, CSVs and figures | Frozen-evidence analysis is rerunnable locally | Training artefacts remain incomplete | Fully addressed |
