# Phase B experiment audit

The executed notebook completed classical oracle-feature regression/classification, ANFIS oracle-feature regression, and three image-regression models. It did **not** train the segmentation baseline: its stored metrics are `not_trained`. Prediction files support 42 regression test rows; directional classification has 16 test rows (left 4, forward 4, right 8). The notebook reports 297/56/42 eligible regression rows and 116/22/16 classification rows for train/validation/test. Oracle features were the only available source; predicted masks were unavailable.
