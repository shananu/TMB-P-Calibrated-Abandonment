# E8 q-shift (train+calibrate q=0.5, alpha=0.1, 50 splits, seed 20261109)
mean frozen lambda: learned 0.0971, table 0.0891

## target q=0.2  (test episodes 1080; counterfactually successful Y=1: 489.6; Y=1 with >=1 failure: 327.4)
| policy | FAR_cond mean [2.5%, 97.5%] | splits FAR_cond>alpha | success | mean calls | waste reduction |
|---|---|---|---|---|---|
| frozen_learned | 0.028 [0.013, 0.044] | 0.00 | 0.445 | 2.45 | 0.580 |
| frozen_table | 0.022 [0.010, 0.034] | 0.00 | 0.447 | 2.67 | 0.525 |
| retry_3 | 0.008 [0.003, 0.013] | 0.00 | 0.451 | 2.88 | 0.472 |
| never_stop | 0.000 [0.000, 0.000] | 0.00 | 0.453 | 4.71 | 0.000 |
| recalibrated_learned | 0.099 [0.060, 0.144] | 0.50 | 0.423 | 1.98 | 0.696 |
| retrained_recalibrated_learned | 0.098 [0.063, 0.136] | 0.42 | 0.424 | 1.96 | 0.701 |

## target q=0.8  (test episodes 1080; counterfactually successful Y=1: 405.1; Y=1 with >=1 failure: 242.9)
| policy | FAR_cond mean [2.5%, 97.5%] | splits FAR_cond>alpha | success | mean calls | waste reduction |
|---|---|---|---|---|---|
| frozen_learned | 0.324 [0.260, 0.408] | 1.00 | 0.302 | 2.84 | 0.580 |
| frozen_table | 0.295 [0.243, 0.384] | 1.00 | 0.309 | 3.09 | 0.525 |
| retry_3 | 0.321 [0.286, 0.363] | 1.00 | 0.303 | 3.31 | 0.474 |
| never_stop | 0.000 [0.000, 0.000] | 0.00 | 0.375 | 5.72 | 0.000 |
| recalibrated_learned | 0.096 [0.053, 0.161] | 0.38 | 0.354 | 4.01 | 0.360 |
| retrained_recalibrated_learned | 0.097 [0.053, 0.163] | 0.44 | 0.353 | 3.97 | 0.371 |
