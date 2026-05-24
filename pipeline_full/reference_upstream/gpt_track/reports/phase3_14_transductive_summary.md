# Phase 3.14 — controlled transductive / semi-supervised summary

Anchor: `phase3_10_horizon_blend_v2` (LB **0.91093**, weighted OOF 0.8811, hep 0.8500, dea 0.9537).

Anchor fold: hep std/min=0.1084/0.6000, dea std/min=0.0123/0.9269.

## 1. Methods tried

| track | description | active? |
|---|---|---|
| A1 | train/test distribution audit | report only |
| A2 | combined train+test preprocessing variants | no |
| B | PCA-10 + KMeans-8 cluster features (current_state_v2) | yes |
| C | extreme consensus pseudo-labelling (hep h3 LGBM) | no |
| D | conservative kNN smoothing (current_state_v2) | yes |

## 2. Track A1 — distribution audit summary

| feature_set       |   n_features |   median_miss_diff |   p95_miss_diff |   median_mean_shift_z_abs |   p95_mean_shift_z_abs |   median_ks |   p95_ks |   n_features_ks_gt_0_2 |
|:------------------|-------------:|-------------------:|----------------:|--------------------------:|-----------------------:|------------:|---------:|-----------------------:|
| current_state_v2  |          222 |            -0.1056 |          0.1432 |                    0.0633 |                 0.1989 |      0.0509 |   0.1263 |                      0 |
| NIT_plus_scores   |           55 |            -0.0809 |          0.1876 |                    0.0486 |                 0.1005 |      0.0516 |   0.1282 |                      0 |
| biomarker_only    |          127 |            -0.1066 |          0.1432 |                    0.0445 |                 0.1097 |      0.0502 |   0.0765 |                      0 |
| v3_hepatic_schema |          215 |            -0.1056 |          0.1432 |                    0.0612 |                 0.1968 |      0.0509 |   0.1271 |                      0 |

## 3. Track A2 — preprocessing variants per anchor component

| variant                | label                                            | endpoint   |   horizon | model           |     ci |   fold_std |   fold_min |   rho_with_anchor_component_oof |   wall_seconds |
|:-----------------------|:-------------------------------------------------|:-----------|----------:|:----------------|-------:|-----------:|-----------:|--------------------------------:|---------------:|
| default                | NIT_plus_scores__hepatic__h1__lgbm_binary        | hepatic    |    1.0000 | lgbm_binary     | 0.7187 |     0.1388 |     0.4372 |                          1.0000 |         1.1596 |
| default                | v3_hepatic_schema__hepatic__h3__lgbm_binary__s4  | hepatic    |    3.0000 | lgbm_binary     | 0.7741 |     0.1260 |     0.5397 |                          1.0000 |         1.8005 |
| default                | current_state_v2__death__h5__catboost_binary__s3 | death      |    5.0000 | catboost_binary | 0.9095 |     0.0484 |     0.7831 |                          1.0000 |         7.9352 |
| default                | NIT_plus_scores__death__h4__catboost_binary      | death      |    4.0000 | catboost_binary | 0.8351 |     0.0595 |     0.7107 |                          1.0000 |         3.4957 |
| combined_median_impute | NIT_plus_scores__hepatic__h1__lgbm_binary        | hepatic    |    1.0000 | lgbm_binary     | 0.6822 |     0.1254 |     0.4651 |                          0.6510 |         0.9355 |
| combined_median_impute | v3_hepatic_schema__hepatic__h3__lgbm_binary__s4  | hepatic    |    3.0000 | lgbm_binary     | 0.7195 |     0.1159 |     0.5092 |                          0.8538 |         1.4801 |
| combined_median_impute | current_state_v2__death__h5__catboost_binary__s3 | death      |    5.0000 | catboost_binary | 0.8875 |     0.0468 |     0.8000 |                          0.9022 |         7.8384 |
| combined_median_impute | NIT_plus_scores__death__h4__catboost_binary      | death      |    4.0000 | catboost_binary | 0.7990 |     0.0629 |     0.6232 |                          0.7848 |         3.5614 |
| combined_clip_1_99     | NIT_plus_scores__hepatic__h1__lgbm_binary        | hepatic    |    1.0000 | lgbm_binary     | 0.6921 |     0.1363 |     0.4378 |                          0.9536 |         1.1108 |
| combined_clip_1_99     | v3_hepatic_schema__hepatic__h3__lgbm_binary__s4  | hepatic    |    3.0000 | lgbm_binary     | 0.7647 |     0.1271 |     0.5206 |                          0.9544 |         1.9102 |
| combined_clip_1_99     | current_state_v2__death__h5__catboost_binary__s3 | death      |    5.0000 | catboost_binary | 0.9046 |     0.0477 |     0.8011 |                          0.9575 |         7.7308 |
| combined_clip_1_99     | NIT_plus_scores__death__h4__catboost_binary      | death      |    4.0000 | catboost_binary | 0.8314 |     0.0576 |     0.6661 |                          0.9250 |         3.6287 |

## 4. Track B — cluster-feature augmented anchor components

| label                                            | endpoint   |   horizon |     ci |   fold_std |   fold_min |   rho_with_anchor_component_oof |
|:-------------------------------------------------|:-----------|----------:|-------:|-----------:|-----------:|--------------------------------:|
| NIT_plus_scores__hepatic__h1__lgbm_binary        | hepatic    |    1.0000 | 0.6564 |     0.1136 |     0.4295 |                          0.7223 |
| v3_hepatic_schema__hepatic__h3__lgbm_binary__s4  | hepatic    |    3.0000 | 0.7314 |     0.1410 |     0.4474 |                          0.9157 |
| current_state_v2__death__h5__catboost_binary__s3 | death      |    5.0000 | 0.9171 |     0.0335 |     0.8542 |                          0.9477 |
| NIT_plus_scores__death__h4__catboost_binary      | death      |    4.0000 | 0.8250 |     0.0586 |     0.6868 |                          0.8039 |

## 5. Track C — pseudo-label sweep (hep h3 LGBM, simulated transductive 5x3 CV)

Reference (no pseudo-label) C-index: 0.7741

|   top_q |   bottom_q |   pseudo_weight |     ci |   ci_delta_vs_ref |   fold_std |   fold_min |   rho_test_anchor_component |   n_pseudo_pos_total |   n_pseudo_neg_total |
|--------:|-----------:|----------------:|-------:|------------------:|-----------:|-----------:|----------------------------:|---------------------:|---------------------:|
|  0.0500 |     0.2000 |          0.0500 | 0.7517 |           -0.0224 |     0.1176 |     0.5504 |                      0.9672 |             189.0000 |             750.0000 |
|  0.0500 |     0.2000 |          0.1000 | 0.7435 |           -0.0306 |     0.1153 |     0.5614 |                      0.9648 |             189.0000 |             750.0000 |
|  0.0500 |     0.2000 |          0.2000 | 0.7494 |           -0.0247 |     0.1152 |     0.5628 |                      0.9589 |             189.0000 |             750.0000 |
|  0.0750 |     0.2250 |          0.0500 | 0.7564 |           -0.0177 |     0.1220 |     0.5261 |                      0.9582 |             285.0000 |             840.0000 |
|  0.0750 |     0.2250 |          0.1000 | 0.7505 |           -0.0236 |     0.1204 |     0.5364 |                      0.9513 |             285.0000 |             840.0000 |
|  0.0750 |     0.2250 |          0.2000 | 0.7532 |           -0.0209 |     0.1230 |     0.5342 |                      0.9455 |             285.0000 |             840.0000 |
|  0.1000 |     0.2500 |          0.0500 | 0.7567 |           -0.0174 |     0.1099 |     0.5690 |                      0.9459 |             375.0000 |             939.0000 |
|  0.1000 |     0.2500 |          0.1000 | 0.7547 |           -0.0194 |     0.1163 |     0.5659 |                      0.9390 |             375.0000 |             939.0000 |
|  0.1000 |     0.2500 |          0.2000 | 0.7517 |           -0.0224 |     0.1215 |     0.5438 |                      0.9328 |             375.0000 |             939.0000 |

## 6. Track D — kNN smoothing (current_state_v2, k=10)

|       k |    lam |   hep_oof |   death_oof |   weighted_oof |   hep_fold_std |   hep_fold_min |   dea_fold_std |   dea_fold_min |   delta_w_vs_anchor |   delta_h_vs_anchor |   delta_d_vs_anchor |   rho_h_test_anchor |   rho_d_test_anchor |
|--------:|-------:|----------:|------------:|---------------:|---------------:|---------------:|---------------:|---------------:|--------------------:|--------------------:|--------------------:|--------------------:|--------------------:|
| 10.0000 | 0.0200 |    0.8502 |      0.9538 |         0.8813 |         0.1095 |         0.5969 |         0.0122 |         0.9258 |              0.0002 |              0.0002 |              0.0001 |              0.9999 |              1.0000 |
| 10.0000 | 0.0500 |    0.8507 |      0.9544 |         0.8818 |         0.1098 |         0.5969 |         0.0121 |         0.9269 |              0.0007 |              0.0007 |              0.0006 |              0.9997 |              0.9998 |
| 10.0000 | 0.1000 |    0.8516 |      0.9545 |         0.8825 |         0.1097 |         0.6000 |         0.0119 |         0.9258 |              0.0014 |              0.0016 |              0.0008 |              0.9991 |              0.9994 |

## 7. Candidates emitted

| candidate                                 |   weighted_oof |   delta_weighted |   hep_oof |   delta_hep |   death_oof |   delta_dea |   hep_fold_std |   dea_fold_std |   rho_h_test_anchor |   rho_d_test_anchor |   h_test_max_shift |   d_test_max_shift | recommended   |
|:------------------------------------------|---------------:|-----------------:|----------:|------------:|------------:|------------:|---------------:|---------------:|--------------------:|--------------------:|-------------------:|-------------------:|:--------------|
| phase3_14_transductive_preprocess_cluster |         0.8811 |          -0.0001 |    0.8500 |      0.0000 |      0.9535 |     -0.0002 |         0.1084 |         0.0127 |              1.0000 |              0.9996 |             0.0000 |             0.0355 | False         |
| phase3_14_knn_smooth                      |         0.8825 |           0.0014 |    0.8516 |      0.0016 |      0.9545 |      0.0008 |         0.1097 |         0.0119 |              0.9991 |              0.9994 |             0.0378 |             0.0449 | False         |
| phase3_14_transductive_combo              |         0.8819 |           0.0007 |    0.8510 |      0.0009 |      0.9540 |      0.0003 |         0.1096 |         0.0118 |              0.9998 |              0.9997 |             0.0201 |             0.0272 | False         |

## 8. Recommendation

**No submission recommended.** None of the four transductive tracks delivered an OOF improvement of ≥ +0.002 weighted or ≥ +0.003 hepatic over `phase3_10_horizon_blend_v2`, and none had a stability rationale strong enough to override that. Hold the anchor and explore non-transductive directions next.


## 9. Notes

- All transductive methods use unlabeled test features only — no event/
  censoring-age columns are touched.
- Track C uses a single-source consensus rule for simplicity (the model trained on the training fold acts as the consensus); we keep top/bottom quantiles tight to reduce reliance on this proxy.
- Track D smooths in the `current_state_v2` feature space; train-train neighbours are used for the OOF analog, test-test for the test predictions.
- All evaluations are OOF only; we never tuned weights, thresholds, or selection on the public LB.
