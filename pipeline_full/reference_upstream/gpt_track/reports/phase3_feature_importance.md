# Phase 2 feature importance

Top features per (feature_set, endpoint, model). `suspicious=True` flags features that match known follow-up / missingness / cutoff patterns (n_visits, followup, follow_up, last_observed_age, age_last, time_since_baseline, miss_total, miss_visit_, _count, gap_max, gap_min, gap_mean). Treat any model whose top-15 contains such features with caution.

## NIT_plus_scores_longitudinal__hepatic__catboost_binary  (top-15, 0 suspicious)

| feature                       |   importance | suspicious   |
|:------------------------------|-------------:|:-------------|
| fibrotest_BM_2_last           |       3.6731 | False        |
| fibrotest_BM_2_min            |       3.5256 | False        |
| ast_alt_ratio_latest          |       3.4122 | False        |
| fibrotest_BM_2_std            |       3.4097 | False        |
| fibrotest_BM_2_delta          |       3.3680 | False        |
| fibrotest_BM_2_max            |       3.0394 | False        |
| Age_v1                        |       2.9748 | False        |
| fibs_stiffness_med_BM_1_max   |       2.8951 | False        |
| fib4_latest                   |       2.8146 | False        |
| aixp_aix_result_BM_3_last     |       2.7433 | False        |
| fibs_stiffness_med_BM_1_slope |       2.7234 | False        |
| fibs_stiffness_med_BM_1_min   |       2.5799 | False        |
| ast_alt_ratio_v2              |       2.5674 | False        |
| fibs_stiffness_med_BM_1_mean  |       2.3457 | False        |
| apri_v2                       |       2.3362 | False        |

### SHAP top-15 (0 suspicious)

| feature                        |   shap_mean_abs | suspicious   |
|:-------------------------------|----------------:|:-------------|
| ast_alt_ratio_latest           |          0.3581 | False        |
| fibrotest_BM_2_std             |          0.3479 | False        |
| fibrotest_BM_2_max             |          0.3393 | False        |
| aixp_aix_result_BM_3_last      |          0.3330 | False        |
| fibrotest_BM_2_last            |          0.3318 | False        |
| fib4_latest                    |          0.2737 | False        |
| fibrotest_BM_2_delta           |          0.2451 | False        |
| fibs_stiffness_med_BM_1_median |          0.2304 | False        |
| fibs_stiffness_med_BM_1_min    |          0.2293 | False        |
| fibrotest_BM_2_min             |          0.2179 | False        |
| ast_alt_ratio_v2               |          0.2155 | False        |
| apri_v2                        |          0.2096 | False        |
| Hypertension                   |          0.1928 | False        |
| fibs_stiffness_med_BM_1_max    |          0.1796 | False        |
| Age_v1                         |          0.1733 | False        |

## NIT_plus_scores_longitudinal__hepatic__lgbm_binary  (top-15, 0 suspicious)

| feature                       |   importance | suspicious   |
|:------------------------------|-------------:|:-------------|
| fibrotest_BM_2_std            |     316.0000 | False        |
| ast_alt_ratio_latest          |     296.0000 | False        |
| Age_v1                        |     281.0000 | False        |
| ast_alt_ratio_v2              |     273.0000 | False        |
| ast_alt_ratio_v3              |     245.0000 | False        |
| fibrotest_BM_2_slope          |     235.0000 | False        |
| ast_alt_ratio_slope           |     235.0000 | False        |
| fibs_stiffness_med_BM_1_std   |     232.0000 | False        |
| apri_latest                   |     216.0000 | False        |
| ast_alt_ratio_delta           |     215.0000 | False        |
| fibrotest_BM_2_last           |     211.0000 | False        |
| fibs_stiffness_med_BM_1_slope |     210.0000 | False        |
| fibrotest_BM_2_delta          |     204.0000 | False        |
| fibs_stiffness_med_BM_1_min   |     196.0000 | False        |
| fib4_latest                   |     192.0000 | False        |

### SHAP top-15 (0 suspicious)

| feature                       |   shap_mean_abs | suspicious   |
|:------------------------------|----------------:|:-------------|
| Dyslipidaemia                 |          0.3961 | False        |
| fib4_latest                   |          0.3941 | False        |
| ast_alt_ratio_v2              |          0.3841 | False        |
| ast_alt_ratio_v3              |          0.2760 | False        |
| fibs_stiffness_med_BM_1_first |          0.2724 | False        |
| fibrotest_BM_2_last           |          0.2641 | False        |
| Hypertension                  |          0.2370 | False        |
| fibrotest_BM_2_std            |          0.2352 | False        |
| ast_alt_ratio_latest          |          0.2303 | False        |
| fibrotest_BM_2_slope          |          0.2291 | False        |
| Age_v1                        |          0.1954 | False        |
| fibs_stiffness_med_BM_1_max   |          0.1855 | False        |
| ast_alt_ratio_v1              |          0.1741 | False        |
| aixp_aix_result_BM_3_slope    |          0.1733 | False        |
| apri_v2                       |          0.1687 | False        |

## aggressive_longitudinal__hepatic__lgbm_binary  (top-15, 0 suspicious)

| feature                       |   importance | suspicious   |
|:------------------------------|-------------:|:-------------|
| Age_v1                        |     178.0000 | False        |
| fibs_stiffness_med_BM_1_std   |     154.0000 | False        |
| BMI_slope                     |     127.0000 | False        |
| fibs_stiffness_med_BM_1_last  |     124.0000 | False        |
| triglyc_slope                 |     121.0000 | False        |
| ast_alt_ratio_latest          |     113.0000 | False        |
| ast_alt_ratio_v1              |     111.0000 | False        |
| fibs_stiffness_med_BM_1_min   |     107.0000 | False        |
| Dyslipidaemia                 |     105.0000 | False        |
| aixp_aix_result_BM_3_first    |      98.0000 | False        |
| fibrotest_BM_2_std            |      94.0000 | False        |
| fibs_stiffness_med_BM_1_first |      94.0000 | False        |
| fibrotest_BM_2_delta          |      91.0000 | False        |
| triglyc_delta                 |      89.0000 | False        |
| ggt_median                    |      89.0000 | False        |

### SHAP top-15 (0 suspicious)

| feature                       |   shap_mean_abs | suspicious   |
|:------------------------------|----------------:|:-------------|
| Dyslipidaemia                 |          0.3951 | False        |
| BMI_slope                     |          0.2860 | False        |
| ast_alt_ratio_v1              |          0.2461 | False        |
| triglyc_std                   |          0.2455 | False        |
| fib4_latest                   |          0.2453 | False        |
| ggt_median                    |          0.2106 | False        |
| BMI_last                      |          0.2085 | False        |
| fibs_stiffness_med_BM_1_first |          0.1838 | False        |
| alt_median                    |          0.1795 | False        |
| fibs_stiffness_med_BM_1_max   |          0.1663 | False        |
| BMI_delta                     |          0.1616 | False        |
| ast_alt_ratio_latest          |          0.1568 | False        |
| fibs_stiffness_med_BM_1_std   |          0.1566 | False        |
| chol_median                   |          0.1546 | False        |
| bilirubin_slope               |          0.1545 | False        |

## current_state_v2__death__xgb_cox  (top-15, 2 suspicious)

| feature                      |   importance | suspicious   |
|:-----------------------------|-------------:|:-------------|
| ggt_min                      |       0.0277 | False        |
| miss_fibs_stiffness_med_BM_1 |       0.0250 | False        |
| ggt_latest                   |       0.0234 | False        |
| ast_alt_ratio_v3             |       0.0226 | False        |
| BMI_max                      |       0.0221 | False        |
| fibrotest_BM_2_std           |       0.0217 | False        |
| chol_age_latest              |       0.0200 | False        |
| followup_span                |       0.0192 | True         |
| plt_latest                   |       0.0173 | False        |
| fib4_v1                      |       0.0151 | False        |
| x_stiffness_astalt_latest    |       0.0148 | False        |
| miss_visit_3                 |       0.0133 | True         |
| ggt_last                     |       0.0129 | False        |
| triglyc_latest               |       0.0129 | False        |
| miss_Age                     |       0.0127 | False        |

## current_state_v2__hepatic__catboost_binary  (top-15, 1 suspicious)

| feature                        |   importance | suspicious   |
|:-------------------------------|-------------:|:-------------|
| fibs_stiffness_med_BM_1_first  |       2.3970 | False        |
| BMI_slope                      |       1.6648 | False        |
| fibs_stiffness_med_BM_1_mean   |       1.6128 | False        |
| gluc_fast_rel_delta            |       1.5753 | False        |
| x_bmi_fib4_latest              |       1.4679 | False        |
| fibs_stiffness_med_BM_1_median |       1.4181 | False        |
| aixp_aix_result_BM_3_latest    |       1.3543 | False        |
| Hypertension                   |       1.3109 | False        |
| x_fibrotest_astalt_latest      |       1.2906 | False        |
| ast_alt_ratio_latest           |       1.2676 | False        |
| followup_span                  |       1.2673 | True         |
| gluc_fast_slope                |       1.2527 | False        |
| x_fibrotest_stiffness_latest   |       1.2521 | False        |
| fibrotest_BM_2_max             |       1.2441 | False        |
| fibrotest_BM_2_median          |       1.2411 | False        |

## current_state_v2__hepatic__lgbm_binary  (top-15, 2 suspicious)

| feature                           |   importance | suspicious   |
|:----------------------------------|-------------:|:-------------|
| fibs_stiffness_med_BM_1_std       |     146.0000 | False        |
| BMI_slope                         |     124.0000 | False        |
| fibrotest_BM_2_std                |     123.0000 | False        |
| followup_span                     |     116.0000 | True         |
| ast_alt_ratio_v3                  |     101.0000 | False        |
| bilirubin_min                     |      95.0000 | False        |
| fibs_stiffness_med_BM_1_min       |      95.0000 | False        |
| ast_alt_ratio_v1                  |      90.0000 | False        |
| Dyslipidaemia                     |      88.0000 | False        |
| age_last_visit                    |      88.0000 | True         |
| BMI_last                          |      86.0000 | False        |
| ast_alt_ratio_latest              |      85.0000 | False        |
| triglyc_std                       |      85.0000 | False        |
| alt_rel_delta                     |      82.0000 | False        |
| fibs_stiffness_med_BM_1_rel_delta |      82.0000 | False        |

## longitudinal_no_followup_proxies__death__xgb_cox  (top-15, 0 suspicious)

| feature                     |   importance | suspicious   |
|:----------------------------|-------------:|:-------------|
| ast_delta                   |       0.1381 | False        |
| chol_rel_delta              |       0.0677 | False        |
| Age_v1                      |       0.0253 | False        |
| aixp_aix_result_BM_3_slope  |       0.0200 | False        |
| BMI_slope                   |       0.0193 | False        |
| plt_median                  |       0.0174 | False        |
| ggt_slope                   |       0.0168 | False        |
| fibs_stiffness_med_BM_1_min |       0.0149 | False        |
| ggt_max                     |       0.0139 | False        |
| aixp_aix_result_BM_3_median |       0.0138 | False        |
| fibrotest_BM_2_slope        |       0.0137 | False        |
| BMI_max                     |       0.0126 | False        |
| ast_slope                   |       0.0124 | False        |
| fibrotest_BM_2_first        |       0.0119 | False        |
| plt_slope                   |       0.0115 | False        |

### SHAP top-15 (0 suspicious)

| feature                     |   shap_mean_abs | suspicious   |
|:----------------------------|----------------:|:-------------|
| Age_v1                      |          0.9330 | False        |
| ast_delta                   |          0.5001 | False        |
| BMI_slope                   |          0.4782 | False        |
| bilirubin_slope             |          0.4779 | False        |
| chol_rel_delta              |          0.4603 | False        |
| plt_slope                   |          0.4222 | False        |
| aixp_aix_result_BM_3_slope  |          0.3720 | False        |
| fibrotest_BM_2_std          |          0.3652 | False        |
| fibrotest_BM_2_min          |          0.3142 | False        |
| alt_slope                   |          0.2229 | False        |
| triglyc_first               |          0.2041 | False        |
| ggt_slope                   |          0.1847 | False        |
| bilirubin_std               |          0.1763 | False        |
| fibs_stiffness_med_BM_1_min |          0.1384 | False        |
| alt_std                     |          0.1322 | False        |

## longitudinal_no_followup_proxies__hepatic__xgb_cox  (top-15, 0 suspicious)

| feature                        |   importance | suspicious   |
|:-------------------------------|-------------:|:-------------|
| fibs_stiffness_med_BM_1_median |       0.0706 | False        |
| bilirubin_rel_delta            |       0.0233 | False        |
| fibs_stiffness_med_BM_1_min    |       0.0211 | False        |
| fibs_stiffness_med_BM_1_mean   |       0.0199 | False        |
| fibrotest_BM_2_max             |       0.0190 | False        |
| fibs_stiffness_med_BM_1_first  |       0.0190 | False        |
| aixp_aix_result_BM_3_min       |       0.0173 | False        |
| T2DM                           |       0.0171 | False        |
| aixp_aix_result_BM_3_last      |       0.0167 | False        |
| ast_min                        |       0.0167 | False        |
| BMI_rel_delta                  |       0.0154 | False        |
| aixp_aix_result_BM_3_median    |       0.0153 | False        |
| triglyc_std                    |       0.0137 | False        |
| ast_mean                       |       0.0128 | False        |
| BMI_first                      |       0.0122 | False        |

### SHAP top-15 (0 suspicious)

| feature                       |   shap_mean_abs | suspicious   |
|:------------------------------|----------------:|:-------------|
| BMI_slope                     |          0.4717 | False        |
| fibs_stiffness_med_BM_1_first |          0.4126 | False        |
| fibs_stiffness_med_BM_1_last  |          0.3505 | False        |
| chol_first                    |          0.2952 | False        |
| alt_first                     |          0.2835 | False        |
| gluc_fast_slope               |          0.2816 | False        |
| triglyc_std                   |          0.2765 | False        |
| plt_last                      |          0.2637 | False        |
| bilirubin_rel_delta           |          0.2372 | False        |
| Dyslipidaemia                 |          0.2296 | False        |
| Age_v1                        |          0.2153 | False        |
| alt_mean                      |          0.2101 | False        |
| ggt_min                       |          0.1774 | False        |
| ggt_first                     |          0.1719 | False        |
| gluc_fast_max                 |          0.1703 | False        |
