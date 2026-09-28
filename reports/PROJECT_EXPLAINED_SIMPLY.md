# Project Explained Simply

## What problem were we solving?
We investigated whether a camera view of a cassava field can provide a useful direction-related signal for future robot navigation research. This project does not control a robot or validate hardware safety.

## What did the pipeline do?
Images and YOLO polygon labels were audited, checked for class/annotation issues, and turned into scene descriptions. The path centre relative to the image centre became a normalised research offset: negative is image-left and positive is image-right. Direction labels were derived from that geometry, rather than measured steering commands.

## What models were compared?
Random Forest and ANFIS used engineered features generated from ground-truth masks, so they are oracle upper-bound experiments. Custom CNN, MobileNetV3 and ResNet18 use images directly. MobileNetV3 is a compact transfer-learning model; ResNet18 uses residual/skip connections; ANFIS combines fuzzy membership rules with learned numeric consequents.

## What did we discover?
On the saved 42-image common test set, Random Forest had the lowest MAE (0.451) and RMSE (0.542), but this cannot be called a deployment win because it used oracle mask features. Among image-only models, MobileNetV3 had the lowest MAE (0.592). Phase C found adjusted paired evidence that RF absolute errors differed from each image model, while most comparisons among ANFIS/CNN/transfer models were inconclusive after Holm correction.

## What are the limitations?
The target is derived image geometry, not a steering label; the test set is small; oracle features are not available to a real robot; no predicted-mask downstream experiment, segmentation run, probability scores, physical robot trial or edge-device trial is verified.

## How to explain this project in 2 minutes
We used annotated cassava-field images to locate the visible path and turn its image position into a provisional left/forward/right navigation signal. We compared interpretable mask-derived features, ANFIS and image networks. The best numerical model used ground-truth masks, so it is an upper bound rather than a direct deployment comparison. The best saved image-only MAE belonged to MobileNetV3. Phase C independently recomputed all saved metrics and tested models on exactly the same images.

## How to explain this project in 5 minutes
Start with the dataset audit and annotation checks, then explain segmentation-derived scene features and the path-centre formula. Explain that classical ML/ANFIS use oracle masks whereas CNNs see only images. State that testing uses saved predictions, with bootstrap intervals and paired Wilcoxon tests. Finish with the need for real steering labels, predicted-mask evaluation and robot trials.

## How to explain this project in 10 minutes
Use the full technical report: walk through data integrity, labels, feature/leakage controls, every model family, the 42-image common-test set, uncertainty/statistics, failure analysis and limitations.

## Questions and answers

**Why YOLO segmentation?** It can represent the path, leaves and ridges as pixels/polygons rather than only boxes, which helps geometry reasoning.

**Why segmentation rather than detection?** Navigation depends on path boundaries and centre, which are spatial shapes.

**What is mAP?** It is a detection/segmentation ranking metric. No trained segmentation mAP is claimed locally because the stored baseline says not trained.

**What is path-centre offset?** The horizontal path-centre displacement divided by half image width.

**What is target leakage?** Giving a model the path centre while asking it to predict a formula based on that centre would reveal the answer.

**Why Random Forest and ANFIS?** They provide interpretable, nonlinear feature-based baselines; ANFIS additionally represents fuzzy rules.

**Why CNN, MobileNetV3 and ResNet18?** They test direct image-to-offset learning, including compact and residual transfer-learning approaches.

**What are MAE, RMSE and R2?** MAE is average absolute error; RMSE penalises larger errors more; R2 compares predictions with predicting the mean. Negative R2 means the mean baseline did better on that sample.

**Why balanced accuracy and F1?** They avoid accuracy alone hiding uneven direction-class performance.

**Why no ROC/AUC?** Saved exports contain hard labels, not probability scores.

**Why are oracle features limited?** Ground-truth masks will not be available during deployment.

**Why no physical robot claim?** Camera/controller calibration, safety and field trials are outside the provided evidence.

**What should improve next?** Collect recorded steering labels across farms and conditions; export predicted-mask/out-of-fold features; test on embedded hardware and a safety-controlled robot.
