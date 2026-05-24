# Phase 3.11 — controlled residual / negative-weight ensemble

Reference candidate: `phase3_10_horizon_blend_v2` (public LB **0.91093**).

Reference OOF: hep=0.8500 / death=0.9537 / weighted=0.8811.

## Pool

### Hepatic pool
| name                           | description                                                              | endpoint   |
|:-------------------------------|:-------------------------------------------------------------------------|:-----------|
| phase3_10_horizon_blend_v2     | current public best                                                      | hepatic    |
| phase3_9_horizon_blend         | previous best                                                            | hepatic    |
| phase3_5_v3_hepatic_focused    | phase3_5 hepatic-focused                                                 | hepatic    |
| hep_h1_lgbm_binary             | best hepatic h1 horizon: NIT_plus_scores__hepatic__h1__lgbm_binary       | hepatic    |
| hep_h2_catboost_binary         | best hepatic h2 horizon: v3_hepatic_schema__hepatic__h2__catboost_binary | hepatic    |
| hep_h3_s4                      | best hepatic h3 horizon: v3_hepatic_schema__hepatic__h3__lgbm_binary__s4 | hepatic    |
| hep_h4_catboost_binary         | best hepatic h4 horizon: v3_hepatic_schema__hepatic__h4__catboost_binary | hepatic    |
| hep_h5_lgbm_binary             | best hepatic h5 horizon: NIT_plus_scores__hepatic__h5__lgbm_binary       | hepatic    |
| hep_h6_catboost_binary         | best hepatic h6 horizon: v3_hepatic_schema__hepatic__h6__catboost_binary | hepatic    |
| current_state_NIT_plus_scores  | survival pool: phase2_NIT_plus_scores_longitudinal                       | hepatic    |
| phase2_aggressive_longitudinal | survival pool: phase2_aggressive_longitudinal                            | hepatic    |
| robust_longitudinal            | survival pool: phase2_longitudinal_no_followup_proxies                   | hepatic    |
| current_state_biomarker_only   | survival pool: phase2_labs_longitudinal_only                             | hepatic    |

### Death pool
| name                           | description                                                             | endpoint   |
|:-------------------------------|:------------------------------------------------------------------------|:-----------|
| phase3_10_horizon_blend_v2     | current public best                                                     | death      |
| phase3_9_horizon_blend         | previous best                                                           | death      |
| phase3_5_v3_hepatic_focused    | phase3_5 hepatic-focused                                                | death      |
| dea_h5_s3                      | best death h5 horizon: current_state_v2__death__h5__catboost_binary__s3 | death      |
| dea_h6_catboost_binary         | best death h6 horizon: current_state_v2__death__h6__catboost_binary     | death      |
| current_state_NIT_plus_scores  | survival pool: phase2_NIT_plus_scores_longitudinal                      | death      |
| phase2_aggressive_longitudinal | survival pool: phase2_aggressive_longitudinal                           | death      |
| robust_longitudinal            | survival pool: phase2_longitudinal_no_followup_proxies                  | death      |
| current_state_biomarker_only   | survival pool: phase2_labs_longitudinal_only                            | death      |

## Component baselines (OOF survival C-index per pool member)

| endpoint   | component                      |   oof_cindex |   fold_mean |   fold_std |   fold_min |   rho_with_anchor_oof |
|:-----------|:-------------------------------|-------------:|------------:|-----------:|-----------:|----------------------:|
| death      | phase3_10_horizon_blend_v2     |       0.9537 |      0.9527 |     0.0123 |     0.9269 |                1.0000 |
| death      | phase3_9_horizon_blend         |       0.9507 |      0.9515 |     0.0125 |     0.9269 |                0.9960 |
| death      | phase3_5_v3_hepatic_focused    |       0.9496 |      0.9502 |     0.0115 |     0.9269 |                0.9939 |
| death      | dea_h6_catboost_binary         |       0.9179 |      0.9195 |     0.0304 |     0.8565 |                0.8410 |
| death      | dea_h5_s3                      |       0.9095 |      0.9123 |     0.0484 |     0.7831 |                0.8250 |
| death      | phase2_aggressive_longitudinal |       0.8640 |      0.8572 |     0.0444 |     0.7779 |                0.4688 |
| death      | robust_longitudinal            |       0.8495 |      0.8440 |     0.0454 |     0.7867 |                0.3906 |
| death      | current_state_NIT_plus_scores  |       0.8320 |      0.8203 |     0.0485 |     0.7559 |                0.4018 |
| death      | current_state_biomarker_only   |       0.8114 |      0.8079 |     0.0514 |     0.7206 |                0.3174 |
| hepatic    | phase3_10_horizon_blend_v2     |       0.8500 |      0.8485 |     0.1084 |     0.6000 |                1.0000 |
| hepatic    | phase3_9_horizon_blend         |       0.8451 |      0.8432 |     0.1061 |     0.6016 |                0.9932 |
| hepatic    | phase3_5_v3_hepatic_focused    |       0.8415 |      0.8370 |     0.0991 |     0.6202 |                0.9808 |
| hepatic    | phase2_aggressive_longitudinal |       0.7820 |      0.7894 |     0.1090 |     0.5952 |                0.7812 |
| hepatic    | hep_h3_s4                      |       0.7741 |      0.7841 |     0.1260 |     0.5397 |                0.5664 |
| hepatic    | current_state_NIT_plus_scores  |       0.7713 |      0.7753 |     0.1227 |     0.5558 |                0.6344 |
| hepatic    | robust_longitudinal            |       0.7649 |      0.7764 |     0.1154 |     0.5785 |                0.7276 |
| hepatic    | hep_h4_catboost_binary         |       0.7571 |      0.7692 |     0.1243 |     0.5636 |                0.6216 |
| hepatic    | current_state_biomarker_only   |       0.7554 |      0.7645 |     0.0930 |     0.5944 |                0.5503 |
| hepatic    | hep_h6_catboost_binary         |       0.7460 |      0.7557 |     0.1141 |     0.5227 |                0.5270 |
| hepatic    | hep_h5_lgbm_binary             |       0.7455 |      0.7561 |     0.1226 |     0.5171 |                0.5176 |
| hepatic    | hep_h2_catboost_binary         |       0.7248 |      0.7255 |     0.1093 |     0.5139 |                0.4516 |
| hepatic    | hep_h1_lgbm_binary             |       0.7187 |      0.7217 |     0.1388 |     0.4372 |                0.4018 |

## Per-endpoint scheme results

### hepatic

| scheme       |   ci_inner_oof |   ci_fullfit_train |   fold_mean |   fold_std |   fold_min |   n_neg_components |   neg_total_magnitude |   max_abs_weight |
|:-------------|---------------:|-------------------:|------------:|-----------:|-----------:|-------------------:|----------------------:|-----------------:|
| A_nonneg     |         0.8487 |             0.8487 |      0.8464 |     0.1086 |     0.5953 |                  0 |               -0.0000 |           0.5000 |
| B_neg_005    |         0.8355 |             0.8355 |      0.8288 |     0.0862 |     0.6326 |                 10 |                0.5000 |           0.5000 |
| C_neg_010    |         0.8190 |             0.8370 |      0.8154 |     0.0951 |     0.6310 |                  9 |                0.8760 |           0.5000 |
| D_neg_020    |         0.7884 |             0.8317 |      0.7879 |     0.1025 |     0.5835 |                  7 |                1.3423 |           0.5000 |
| E_ridge      |         0.8173 |             0.8391 |      0.8205 |     0.1122 |     0.5912 |                  7 |                1.2326 |           1.2013 |
| F_greedy_sub |         0.8500 |             0.8500 |      0.8485 |     0.1084 |     0.6000 |                  0 |               -0.0000 |           1.0000 |

#### Final weights (full-data fit) — hepatic

| component                      |   A_nonneg |   B_neg_005 |   C_neg_010 |   D_neg_020 |   E_ridge |   F_greedy_sub |
|:-------------------------------|-----------:|------------:|------------:|------------:|----------:|---------------:|
| phase3_10_horizon_blend_v2     |     0.5000 |      0.5000 |      0.5000 |      0.5000 |   -0.5979 |         1.0000 |
| phase3_9_horizon_blend         |     0.5000 |      0.5000 |      0.5000 |      0.5000 |   -0.1589 |         0.0000 |
| phase3_5_v3_hepatic_focused    |     0.0000 |      0.5000 |      0.5000 |      0.5000 |    1.2013 |         0.0000 |
| hep_h1_lgbm_binary             |     0.0000 |     -0.0500 |     -0.1000 |     -0.2000 |    0.1005 |         0.0000 |
| hep_h2_catboost_binary         |     0.0000 |     -0.0500 |     -0.1000 |     -0.2000 |    0.0673 |         0.0000 |
| hep_h3_s4                      |     0.0000 |     -0.0500 |      0.3760 |      0.5000 |    0.2979 |         0.0000 |
| hep_h4_catboost_binary         |     0.0000 |     -0.0500 |     -0.1000 |     -0.1423 |   -0.0964 |         0.0000 |
| hep_h5_lgbm_binary             |     0.0000 |     -0.0500 |     -0.0760 |      0.1931 |   -0.0124 |         0.0000 |
| hep_h6_catboost_binary         |     0.0000 |     -0.0500 |     -0.1000 |     -0.2000 |   -0.1032 |         0.0000 |
| current_state_NIT_plus_scores  |     0.0000 |     -0.0500 |     -0.1000 |     -0.2000 |    0.2231 |         0.0000 |
| phase2_aggressive_longitudinal |     0.0000 |     -0.0500 |     -0.1000 |      0.1492 |   -0.2349 |         0.0000 |
| robust_longitudinal            |     0.0000 |     -0.0500 |     -0.1000 |     -0.2000 |   -0.0289 |         0.0000 |
| current_state_biomarker_only   |     0.0000 |     -0.0500 |     -0.1000 |     -0.2000 |    0.3425 |         0.0000 |

#### Per-fold weight stability (std across 5 inner folds) — hepatic

| component                      |   A_nonneg |   B_neg_005 |   C_neg_010 |   D_neg_020 |   E_ridge |   F_greedy_sub |
|:-------------------------------|-----------:|------------:|------------:|------------:|----------:|---------------:|
| phase3_10_horizon_blend_v2     |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.0454 |         0.0000 |
| phase3_9_horizon_blend         |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.0647 |         0.0000 |
| phase3_5_v3_hepatic_focused    |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.0654 |         0.0000 |
| hep_h1_lgbm_binary             |     0.0000 |      0.0000 |      0.0000 |      0.0550 |    0.0175 |         0.0000 |
| hep_h2_catboost_binary         |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.0361 |         0.0000 |
| hep_h3_s4                      |     0.0000 |      0.0000 |      0.1970 |      0.1656 |    0.0578 |         0.0000 |
| hep_h4_catboost_binary         |     0.0000 |      0.0000 |      0.0000 |      0.1869 |    0.0962 |         0.0000 |
| hep_h5_lgbm_binary             |     0.0000 |      0.0000 |      0.1974 |      0.2793 |    0.0581 |         0.0000 |
| hep_h6_catboost_binary         |     0.0000 |      0.0000 |      0.0000 |      0.0793 |    0.0996 |         0.0000 |
| current_state_NIT_plus_scores  |     0.0000 |      0.0000 |      0.0000 |      0.0400 |    0.0317 |         0.0000 |
| phase2_aggressive_longitudinal |     0.0000 |      0.0000 |      0.1298 |      0.2978 |    0.0723 |         0.0000 |
| robust_longitudinal            |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.0555 |         0.0000 |
| current_state_biomarker_only   |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.0751 |         0.0000 |

### death

| scheme       |   ci_inner_oof |   ci_fullfit_train |   fold_mean |   fold_std |   fold_min |   n_neg_components |   neg_total_magnitude |   max_abs_weight |
|:-------------|---------------:|-------------------:|------------:|-----------:|-----------:|-------------------:|----------------------:|-----------------:|
| A_nonneg     |         0.9516 |             0.9526 |      0.9515 |     0.0128 |     0.9269 |                  0 |               -0.0000 |           0.5000 |
| B_neg_005    |         0.9520 |             0.9522 |      0.9514 |     0.0123 |     0.9302 |                  6 |                0.3000 |           0.5000 |
| C_neg_010    |         0.9507 |             0.9507 |      0.9506 |     0.0121 |     0.9302 |                  5 |                0.5000 |           0.5000 |
| D_neg_020    |         0.9323 |             0.9323 |      0.9293 |     0.0301 |     0.8716 |                  5 |                1.0000 |           0.5000 |
| E_ridge      |         0.8063 |             0.8206 |      0.7989 |     0.0671 |     0.6508 |                  4 |                1.3046 |           0.8803 |
| F_greedy_sub |         0.9537 |             0.9537 |      0.9527 |     0.0123 |     0.9269 |                  0 |               -0.0000 |           1.0000 |

#### Final weights (full-data fit) — death

| component                      |   A_nonneg |   B_neg_005 |   C_neg_010 |   D_neg_020 |   E_ridge |   F_greedy_sub |
|:-------------------------------|-----------:|------------:|------------:|------------:|----------:|---------------:|
| phase3_10_horizon_blend_v2     |     0.5000 |      0.5000 |      0.5000 |      0.5000 |    0.5567 |         1.0000 |
| phase3_9_horizon_blend         |     0.2477 |      0.3950 |      0.5000 |      0.5000 |   -0.4315 |         0.0000 |
| phase3_5_v3_hepatic_focused    |     0.2523 |      0.4050 |      0.5000 |      0.5000 |   -0.5307 |         0.0000 |
| dea_h5_s3                      |     0.0000 |     -0.0500 |     -0.1000 |     -0.2000 |    0.6356 |         0.0000 |
| dea_h6_catboost_binary         |     0.0000 |     -0.0500 |     -0.0000 |      0.5000 |   -0.2460 |         0.0000 |
| current_state_NIT_plus_scores  |     0.0000 |     -0.0500 |     -0.1000 |     -0.2000 |    0.2079 |         0.0000 |
| phase2_aggressive_longitudinal |     0.0000 |     -0.0500 |     -0.1000 |     -0.2000 |   -0.0964 |         0.0000 |
| robust_longitudinal            |     0.0000 |     -0.0500 |     -0.1000 |     -0.2000 |    0.8803 |         0.0000 |
| current_state_biomarker_only   |     0.0000 |     -0.0500 |     -0.1000 |     -0.2000 |    0.0240 |         0.0000 |

#### Per-fold weight stability (std across 5 inner folds) — death

| component                      |   A_nonneg |   B_neg_005 |   C_neg_010 |   D_neg_020 |   E_ridge |   F_greedy_sub |
|:-------------------------------|-----------:|------------:|------------:|------------:|----------:|---------------:|
| phase3_10_horizon_blend_v2     |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.1592 |         0.0000 |
| phase3_9_horizon_blend         |     0.1997 |      0.0741 |      0.0000 |      0.0000 |    0.0896 |         0.0000 |
| phase3_5_v3_hepatic_focused    |     0.1997 |      0.0741 |      0.0000 |      0.0000 |    0.0814 |         0.0000 |
| dea_h5_s3                      |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.0623 |         0.0000 |
| dea_h6_catboost_binary         |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.0463 |         0.0000 |
| current_state_NIT_plus_scores  |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.1036 |         0.0000 |
| phase2_aggressive_longitudinal |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.1297 |         0.0000 |
| robust_longitudinal            |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.1511 |         0.0000 |
| current_state_biomarker_only   |     0.0000 |      0.0000 |      0.0000 |      0.0000 |    0.0743 |         0.0000 |

## (hep × death) scheme-pair leaderboard (top 15 by weighted OOF)

| hep_scheme   | dea_scheme   |   hep_oof |   death_oof |   weighted_oof |   hep_fold_std |   hep_fold_min |   dea_fold_std |   dea_fold_min |   delta_weighted_vs_p10 |   delta_hep_vs_p10 |   delta_dea_vs_p10 |   hep_neg_total |   dea_neg_total |   hep_neg_components |   dea_neg_components |   hep_max_abs_weight |   dea_max_abs_weight |
|:-------------|:-------------|----------:|------------:|---------------:|---------------:|---------------:|---------------:|---------------:|------------------------:|-------------------:|-------------------:|----------------:|----------------:|---------------------:|---------------------:|---------------------:|---------------------:|
| F_greedy_sub | F_greedy_sub |    0.8500 |      0.9537 |         0.8811 |         0.1084 |         0.6000 |         0.0123 |         0.9269 |                  0.0000 |             0.0000 |             0.0000 |         -0.0000 |         -0.0000 |                    0 |                    0 |               1.0000 |               1.0000 |
| F_greedy_sub | B_neg_005    |    0.8500 |      0.9520 |         0.8806 |         0.1084 |         0.6000 |         0.0123 |         0.9302 |                 -0.0005 |             0.0000 |            -0.0017 |         -0.0000 |          0.3000 |                    0 |                    6 |               1.0000 |               0.5000 |
| F_greedy_sub | A_nonneg     |    0.8500 |      0.9516 |         0.8805 |         0.1084 |         0.6000 |         0.0128 |         0.9269 |                 -0.0006 |             0.0000 |            -0.0022 |         -0.0000 |         -0.0000 |                    0 |                    0 |               1.0000 |               0.5000 |
| F_greedy_sub | C_neg_010    |    0.8500 |      0.9507 |         0.8802 |         0.1084 |         0.6000 |         0.0121 |         0.9302 |                 -0.0009 |             0.0000 |            -0.0030 |         -0.0000 |          0.5000 |                    0 |                    5 |               1.0000 |               0.5000 |
| A_nonneg     | F_greedy_sub |    0.8487 |      0.9537 |         0.8802 |         0.1086 |         0.5953 |         0.0123 |         0.9269 |                 -0.0009 |            -0.0014 |             0.0000 |         -0.0000 |         -0.0000 |                    0 |                    0 |               0.5000 |               1.0000 |
| A_nonneg     | B_neg_005    |    0.8487 |      0.9520 |         0.8797 |         0.1086 |         0.5953 |         0.0123 |         0.9302 |                 -0.0015 |            -0.0014 |            -0.0017 |         -0.0000 |          0.3000 |                    0 |                    6 |               0.5000 |               0.5000 |
| A_nonneg     | A_nonneg     |    0.8487 |      0.9516 |         0.8795 |         0.1086 |         0.5953 |         0.0128 |         0.9269 |                 -0.0016 |            -0.0014 |            -0.0022 |         -0.0000 |         -0.0000 |                    0 |                    0 |               0.5000 |               0.5000 |
| A_nonneg     | C_neg_010    |    0.8487 |      0.9507 |         0.8793 |         0.1086 |         0.5953 |         0.0121 |         0.9302 |                 -0.0018 |            -0.0014 |            -0.0030 |         -0.0000 |          0.5000 |                    0 |                    5 |               0.5000 |               0.5000 |
| F_greedy_sub | D_neg_020    |    0.8500 |      0.9323 |         0.8747 |         0.1084 |         0.6000 |         0.0301 |         0.8716 |                 -0.0064 |             0.0000 |            -0.0214 |         -0.0000 |          1.0000 |                    0 |                    5 |               1.0000 |               0.5000 |
| A_nonneg     | D_neg_020    |    0.8487 |      0.9323 |         0.8737 |         0.1086 |         0.5953 |         0.0301 |         0.8716 |                 -0.0074 |            -0.0014 |            -0.0214 |         -0.0000 |          1.0000 |                    0 |                    5 |               0.5000 |               0.5000 |
| B_neg_005    | F_greedy_sub |    0.8355 |      0.9537 |         0.8709 |         0.0862 |         0.6326 |         0.0123 |         0.9269 |                 -0.0102 |            -0.0145 |             0.0000 |          0.5000 |         -0.0000 |                   10 |                    0 |               0.5000 |               1.0000 |
| B_neg_005    | B_neg_005    |    0.8355 |      0.9520 |         0.8704 |         0.0862 |         0.6326 |         0.0123 |         0.9302 |                 -0.0107 |            -0.0145 |            -0.0017 |          0.5000 |          0.3000 |                   10 |                    6 |               0.5000 |               0.5000 |
| B_neg_005    | A_nonneg     |    0.8355 |      0.9516 |         0.8703 |         0.0862 |         0.6326 |         0.0128 |         0.9269 |                 -0.0108 |            -0.0145 |            -0.0022 |          0.5000 |         -0.0000 |                   10 |                    0 |               0.5000 |               0.5000 |
| B_neg_005    | C_neg_010    |    0.8355 |      0.9507 |         0.8700 |         0.0862 |         0.6326 |         0.0121 |         0.9302 |                 -0.0111 |            -0.0145 |            -0.0030 |          0.5000 |          0.5000 |                   10 |                    5 |               0.5000 |               0.5000 |
| B_neg_005    | D_neg_020    |    0.8355 |      0.9323 |         0.8645 |         0.0862 |         0.6326 |         0.0301 |         0.8716 |                 -0.0166 |            -0.0145 |            -0.0214 |          0.5000 |          1.0000 |                   10 |                    5 |               0.5000 |               0.5000 |

## Promotion decision

**No submission recommended.** No (hep_scheme, dea_scheme) combination meets the criteria (Δweighted ≥ +0.002 or Δhep ≥ +0.003 over phase3_10_horizon_blend_v2, with stable weights and no single-component negative dominance). The top weighted-OOF candidate was (F_greedy_sub, F_greedy_sub) with Δweighted=+0.0000, Δhep=+0.0000.


## Did negative weights help?

Hepatic-only ablation (death held to scheme A):
| hep_scheme   |   hep_oof |   delta_vs_A |   neg_total |
|:-------------|----------:|-------------:|------------:|
| A_nonneg     |    0.8487 |       0.0000 |     -0.0000 |
| B_neg_005    |    0.8355 |      -0.0132 |      0.5000 |
| C_neg_010    |    0.8190 |      -0.0296 |      0.8760 |
| D_neg_020    |    0.7884 |      -0.0602 |      1.3423 |

## Notes

- All weights are constrained to |w| ≤ 0.50 with sum = 1, applied to centred percentile ranks (rank − 0.5).
- Inner CV: 5-fold stratified by event, separate per endpoint.
- The objective for schemes A–D is the smooth concordance loss on comparable pairs; for E, ridge regression on the centred event indicator with α=1.0; for F, discrete greedy ±0.05 steps subject to the same |w| ≤ 0.50 and anchor ≥ −0.20 floor.
- We never include strict_time_aligned components or any feature derived from event/censoring ages.
