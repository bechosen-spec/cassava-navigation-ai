# Experimental Workflow Trace

Dataset archive -> dataset audit -> annotation and semantic-class validation -> polygon/ground-truth feature extraction -> geometry-derived continuous and directional research targets -> leakage screening -> classical ML and ANFIS oracle-feature experiments -> image-only Custom CNN, MobileNetV3 and ResNet18 regression -> frozen Phase B exports -> Phase C independent validation, common-test alignment, bootstrap intervals, paired tests and reporting.

The source tree confirms 3 semantic classes (`path`, `cassava_leaves`, `ridge`). Navigation offset is derived from path geometry, not measured steering or motor control. The supplied local segmentation baseline remains `not_trained`; Phase C therefore does not claim trained segmentation performance or a predicted-mask downstream evaluation.
