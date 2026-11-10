# Classification metrics: recovery-success predictor (TMB-P, pi=0.5, q=0.5)

**Unit:** decision point (after a failed tool call). **Positive class (Y=1):** continuing would succeed. **Test set:** all 2,160 pool episodes, never used for fitting or for choosing a classification threshold. Refit models reproduce the saved E1 AUROC/Brier exactly (asserted).

**Support (test decision points):** n=8660, Y=1: 1167, Y=0: 7493 (from 1800 episodes with ≥1 failure; 360 episodes never fail and are never classified). Train: n=8575, Y=1: 1127, Y=0: 7448.

95% CIs: cluster bootstrap over episodes (1,000 resamples).


## Threshold 0.5 (pre-specified) — all test decision points

| model | thr | Accuracy | Bal. acc. | Precision | Recall | Specificity | F1 (Y=1) | Macro F1 | Weighted F1 | AUROC | AUPRC | Brier | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gradient_boosting | 0.500 | 0.865 [0.853, 0.876] | 0.554 [0.545, 0.564] | 0.495 [0.435, 0.551] | 0.129 [0.111, 0.148] | 0.980 [0.977, 0.983] | 0.204 [0.177, 0.232] | 0.565 [0.551, 0.580] | 0.829 | 0.870 [0.858, 0.881] | 0.431 [0.397, 0.462] | 0.089 [0.083, 0.096] | 7340 | 153 | 1017 | 150 |
| logistic_regression | 0.500 | 0.869 [0.856, 0.880] | 0.549 [0.541, 0.558] | 0.567 [0.504, 0.630] | 0.112 [0.096, 0.130] | 0.987 [0.984, 0.989] | 0.187 [0.162, 0.212] | 0.558 [0.544, 0.572] | 0.829 | 0.871 [0.859, 0.882] | 0.448 [0.415, 0.480] | 0.089 [0.082, 0.096] | 7393 | 100 | 1036 | 131 |

**Sensitivity analysis (one prediction per episode — first decision point only; n=1800, Y=1: 630):**

| model | Accuracy | Bal. acc. | Precision | Recall | Specificity | Macro F1 | AUROC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gradient_boosting | 0.648 | 0.554 | 0.495 | 0.238 | 0.869 | 0.542 | 0.693 | 1017 | 153 | 480 | 150 |
| logistic_regression | 0.667 | 0.561 | 0.567 | 0.208 | 0.915 | 0.543 | 0.703 | 1070 | 100 | 499 | 131 |

## Threshold selected on training data (Youden J, out-of-fold) — all test decision points

| model | thr | Accuracy | Bal. acc. | Precision | Recall | Specificity | F1 (Y=1) | Macro F1 | Weighted F1 | AUROC | AUPRC | Brier | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gradient_boosting | 0.159 | 0.769 [0.759, 0.780] | 0.799 [0.785, 0.813] | 0.351 [0.323, 0.378] | 0.840 [0.814, 0.865] | 0.758 [0.747, 0.770] | 0.495 [0.467, 0.523] | 0.673 [0.658, 0.687] | 0.803 | 0.870 [0.858, 0.881] | 0.431 [0.397, 0.462] | 0.089 [0.083, 0.096] | 5683 | 1810 | 187 | 980 |
| logistic_regression | 0.115 | 0.750 [0.738, 0.760] | 0.807 [0.794, 0.819] | 0.337 [0.311, 0.363] | 0.884 [0.862, 0.907] | 0.729 [0.716, 0.740] | 0.488 [0.459, 0.514] | 0.661 [0.646, 0.676] | 0.788 | 0.871 [0.859, 0.882] | 0.448 [0.415, 0.480] | 0.089 [0.082, 0.096] | 5461 | 2032 | 135 | 1032 |

**Sensitivity analysis (one prediction per episode — first decision point only; n=1800, Y=1: 630):**

| model | Accuracy | Bal. acc. | Precision | Recall | Specificity | Macro F1 | AUROC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gradient_boosting | 0.566 | 0.660 | 0.445 | 0.975 | 0.345 | 0.560 | 0.693 | 404 | 766 | 16 | 614 |
| logistic_regression | 0.524 | 0.631 | 0.423 | 0.987 | 0.275 | 0.511 | 0.703 | 322 | 848 | 8 | 622 |

## Operating point of the stopping policy (CRC lambda, alpha=0.1; NOT a classification threshold)

lambda chosen on each E1 calibration half, metrics on the matching test half, mean over 50 splits.

| model | mean lambda | Accuracy | Precision (Y=1) | Recall (Y=1) | Specificity | NPV (precision of 'stop') | Mean TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|
| gradient_boosting | 0.095 | 0.708 | 0.306 | 0.919 | 0.675 | 0.982 | 2523.9 | 1216.4 | 47.3 | 534.0 |
| logistic_regression | 0.093 | 0.706 | 0.305 | 0.918 | 0.674 | 0.981 | 2519.9 | 1220.3 | 48.3 | 533.0 |

## Paired comparison, gradient boosting − logistic regression (same test points; episode cluster bootstrap)

| metric | mean difference | 95% CI |
|---|---|---|
| auroc | -0.0012 | [-0.0063, +0.0043] |
| auprc | -0.0169 | [-0.0392, +0.0058] |
| brier | +0.0005 | [-0.0009, +0.0018] |
| balanced_accuracy@T_train | -0.0074 | [-0.0163, +0.0013] |
| first_dp_auroc | -0.0093 | [-0.0284, +0.0097] |

First-decision-point AUROC 95% CI: gradient_boosting 0.670–0.716; logistic_regression 0.678–0.727

Note: the CRC-row confusion counts treat every decision point as an independent classification. The stopping policy acts only at the FIRST decision point below lambda, so its episode-level FAR_cond (README, E1) - not this table - is the policy's error rate.
