# Phase 3.5 — `phase3_current_state_v2` decomposition

Public LB: **0.88521** (current best). Local OOF: hepatic 0.8378, death 0.9495, weighted 0.8713.

Method per endpoint:

- hepatic: `greedy`
- death:   `seed_bagged`

## Feature group counts in `current_state_v2`

| group                       |   n_features |
|:----------------------------|-------------:|
| labs_biomarker              |          108 |
| missingness_pattern         |           36 |
| nit_liver_stiffness         |           36 |
| derived_fibrosis_score      |           18 |
| interaction                 |           10 |
| demographics_comorbidities  |            7 |
| visit_history_current_state |            7 |

No target-derived columns (event_age, death_age) are included in any group.

## Hepatic components

| component                                     |   weight | model        | feature_set      |   oof_cindex |   fold_mean |   fold_std |   fold_min |   fold_max |   rank_corr_with_final_oof |   rank_corr_with_final_test |
|:----------------------------------------------|---------:|:-------------|:-----------------|-------------:|------------:|-----------:|-----------:|-----------:|---------------------------:|----------------------------:|
| phase3_current_state_v2::hepatic__rsf__s0     |   1.0000 | rsf (s0)     | current_state_v2 |       0.8357 |      0.8305 |     0.1031 |     0.6056 |     0.9622 |                     0.9523 |                      0.9735 |
| phase3_current_state_v2::hepatic__rsf__s1     |   1.0000 | rsf (s1)     | current_state_v2 |       0.8333 |      0.8340 |     0.1066 |     0.6140 |     0.9553 |                     0.9469 |                      0.9690 |
| phase3_current_state_v2::hepatic__xgb_aft__s0 |   1.0000 | xgb_aft (s0) | current_state_v2 |       0.7999 |      0.8025 |     0.1047 |     0.5953 |     0.9411 |                     0.8381 |                      0.8835 |

### Pairwise rank correlation (hepatic)

|                                               |   phase3_current_state_v2::hepatic__rsf__s0 |   phase3_current_state_v2::hepatic__xgb_aft__s0 |   phase3_current_state_v2::hepatic__rsf__s1 |
|:----------------------------------------------|--------------------------------------------:|------------------------------------------------:|--------------------------------------------:|
| phase3_current_state_v2::hepatic__rsf__s0     |                                       1     |                                           0.658 |                                       0.952 |
| phase3_current_state_v2::hepatic__xgb_aft__s0 |                                       0.658 |                                           1     |                                       0.644 |
| phase3_current_state_v2::hepatic__rsf__s1     |                                       0.952 |                                           0.644 |                                       1     |

## Death components

| component                                       |   weight | model            | feature_set      |   oof_cindex |   fold_mean |   fold_std |   fold_min |   fold_max |   rank_corr_with_final_oof |   rank_corr_with_final_test |
|:------------------------------------------------|---------:|:-----------------|:-----------------|-------------:|------------:|-----------:|-----------:|-----------:|---------------------------:|----------------------------:|
| phase3_current_state_v2::death__rsf__s0         |   0.2727 | rsf (s0)         | current_state_v2 |       0.9268 |      0.9266 |     0.0161 |     0.9009 |     0.9612 |                     0.8789 |                      0.9345 |
| phase3_current_state_v2::death__rsf__s1         |   0.1364 | rsf (s1)         | current_state_v2 |       0.9262 |      0.9242 |     0.0161 |     0.8936 |     0.9647 |                     0.8793 |                      0.9368 |
| phase3_current_state_v2::death__xgb_cox__s0     |   0.2121 | xgb_cox (s0)     | current_state_v2 |       0.9261 |      0.9291 |     0.0294 |     0.8621 |     0.9716 |                     0.9083 |                      0.9607 |
| phase3_current_state_v2::death__xgb_cox__s1     |   0.1970 | xgb_cox (s1)     | current_state_v2 |       0.9242 |      0.9263 |     0.0345 |     0.8404 |     0.9735 |                     0.8603 |                      0.9588 |
| phase3_current_state_v2::death__xgb_aft__s0     |   0.1061 | xgb_aft (s0)     | current_state_v2 |       0.9237 |      0.9223 |     0.0217 |     0.8757 |     0.9579 |                     0.8323 |                      0.8694 |
| phase3_current_state_v2::death__xgb_aft__s1     |   0.0303 | xgb_aft (s1)     | current_state_v2 |       0.9237 |      0.9223 |     0.0217 |     0.8757 |     0.9579 |                     0.8323 |                      0.8691 |
| phase3_current_state_v2::death__lgbm_binary__s0 |   0.0152 | lgbm_binary (s0) | current_state_v2 |       0.7782 |      0.7724 |     0.0546 |     0.6948 |     0.8912 |                     0.1017 |                      0.1780 |
| phase3_current_state_v2::death__lgbm_binary__s1 |   0.0152 | lgbm_binary (s1) | current_state_v2 |       0.7775 |      0.7705 |     0.0582 |     0.6819 |     0.8927 |                     0.0991 |                      0.1758 |
| phase3_current_state_v2::death__xgb_binary__s0  |   0.0152 | xgb_binary (s0)  | current_state_v2 |       0.7739 |      0.7651 |     0.0541 |     0.6734 |     0.8695 |                     0.1020 |                      0.1706 |

### Pairwise rank correlation (death)

|                                                 |   phase3_current_state_v2::death__rsf__s0 |   phase3_current_state_v2::death__xgb_cox__s0 |   phase3_current_state_v2::death__xgb_aft__s0 |   phase3_current_state_v2::death__lgbm_binary__s0 |   phase3_current_state_v2::death__xgb_binary__s0 |   phase3_current_state_v2::death__rsf__s1 |   phase3_current_state_v2::death__xgb_cox__s1 |   phase3_current_state_v2::death__xgb_aft__s1 |   phase3_current_state_v2::death__lgbm_binary__s1 |
|:------------------------------------------------|------------------------------------------:|----------------------------------------------:|----------------------------------------------:|--------------------------------------------------:|-------------------------------------------------:|------------------------------------------:|----------------------------------------------:|----------------------------------------------:|--------------------------------------------------:|
| phase3_current_state_v2::death__rsf__s0         |                                     1     |                                         0.68  |                                         0.602 |                                            -0.176 |                                           -0.19  |                                     0.991 |                                         0.615 |                                         0.602 |                                            -0.179 |
| phase3_current_state_v2::death__xgb_cox__s0     |                                     0.68  |                                         1     |                                         0.754 |                                             0.129 |                                            0.134 |                                     0.692 |                                         0.853 |                                         0.754 |                                             0.128 |
| phase3_current_state_v2::death__xgb_aft__s0     |                                     0.602 |                                         0.754 |                                         1     |                                             0.497 |                                            0.514 |                                     0.596 |                                         0.713 |                                         1     |                                             0.495 |
| phase3_current_state_v2::death__lgbm_binary__s0 |                                    -0.176 |                                         0.129 |                                         0.497 |                                             1     |                                            0.953 |                                    -0.192 |                                         0.115 |                                         0.497 |                                             0.987 |
| phase3_current_state_v2::death__xgb_binary__s0  |                                    -0.19  |                                         0.134 |                                         0.514 |                                             0.953 |                                            1     |                                    -0.206 |                                         0.126 |                                         0.514 |                                             0.953 |
| phase3_current_state_v2::death__rsf__s1         |                                     0.991 |                                         0.692 |                                         0.596 |                                            -0.192 |                                           -0.206 |                                     1     |                                         0.616 |                                         0.596 |                                            -0.196 |
| phase3_current_state_v2::death__xgb_cox__s1     |                                     0.615 |                                         0.853 |                                         0.713 |                                             0.115 |                                            0.126 |                                     0.616 |                                         1     |                                         0.713 |                                             0.113 |
| phase3_current_state_v2::death__xgb_aft__s1     |                                     0.602 |                                         0.754 |                                         1     |                                             0.497 |                                            0.514 |                                     0.596 |                                         0.713 |                                         1     |                                             0.495 |
| phase3_current_state_v2::death__lgbm_binary__s1 |                                    -0.179 |                                         0.128 |                                         0.495 |                                             0.987 |                                            0.953 |                                    -0.196 |                                         0.113 |                                         0.495 |                                             1     |

## Top-12 native importances per family

### xgb_cox__death

| feature                      |   importance | group                       | suspicious   |
|:-----------------------------|-------------:|:----------------------------|:-------------|
| ggt_min                      |    0.0277087 | labs_biomarker              | False        |
| miss_fibs_stiffness_med_BM_1 |    0.0249582 | missingness_pattern         | False        |
| ggt_latest                   |    0.0234367 | labs_biomarker              | False        |
| ast_alt_ratio_v3             |    0.0226492 | derived_fibrosis_score      | False        |
| BMI_max                      |    0.0220971 | labs_biomarker              | False        |
| fibrotest_BM_2_std           |    0.0217251 | nit_liver_stiffness         | False        |
| chol_age_latest              |    0.019974  | labs_biomarker              | False        |
| followup_span                |    0.0192046 | visit_history_current_state | True         |
| plt_latest                   |    0.0172993 | labs_biomarker              | False        |
| fib4_v1                      |    0.015059  | derived_fibrosis_score      | False        |
| x_stiffness_astalt_latest    |    0.0148399 | interaction                 | False        |
| miss_visit_3                 |    0.0133141 | missingness_pattern         | True         |

## Notes

- All components share the same `current_state_v2` feature set; diversity comes from model family and seed, not from feature space.
- Hepatic ensemble is dominated by RSF (two seeds) plus xgb_aft; binary classifiers were not selected by greedy.
- Death ensemble is broader (9 components) — seed_bagged picked every model except coxnet/coxph (not in the Phase 3 config) and catboost_binary (less stable on the death endpoint here).
- A handful of suspicious features (`followup_span`, `age_last_visit`, missingness columns) appear in the top-12 importances. These are permitted by the Phase 3 spec, but the simplified candidate (Phase 3.5 ablation) shows what happens if we remove them.
