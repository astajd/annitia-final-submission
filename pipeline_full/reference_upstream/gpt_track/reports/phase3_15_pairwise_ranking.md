# Phase 3.15 — pairwise / learning-to-rank for hepatic

Anchor: `phase3_10_horizon_blend_v2` (LB **0.91093**, weighted OOF 0.8811, hep 0.8500, dea 0.9537).

Anchor fold: hep std/min=0.1084/0.6000, dea std/min=0.0123/0.9269.

## A. Pair construction

Risk-set rule: for each event patient *i* in the training fold, comparable controls are training-fold patients *j* with `time_j > time_i` (regardless of whether *j* eventually has the event). All comparisons use only `(event, time)` from the survival endpoint — no event/censoring-age columns are used as features.

- LightGBM/XGBoost: query group per event = (event patient, label=1) + sampled controls (label=0).
- Pairwise logistic: difference vectors x_i − x_j with label 1 plus the symmetric x_j − x_i with label 0.
- Cap at K controls per event (K ∈ {20, 50}) so a few events with many comparable controls don't dominate.


## B. Ranker sweep results

| label                                            | endpoint   | model_kind      | feature_set       |   n_controls |   seed |   ci_oof |   fold_std |   fold_min |   rho_oof_anchor |   rho_test_anchor |   median_events_per_fold |   median_pairs_per_fold |   wall_seconds |
|:-------------------------------------------------|:-----------|:----------------|:------------------|-------------:|-------:|---------:|-----------:|-----------:|-----------------:|------------------:|-------------------------:|------------------------:|---------------:|
| dea__xgb_ranker__current_state_v2__K50__s0       | death      | xgb_ranker      | current_state_v2  |           50 |      0 |   0.9341 |     0.0236 |     0.8754 |           0.8931 |            0.9540 |                       61 |                    2706 |       118.0606 |
| dea__lgbm_ranker__current_state_v2__K50__s0      | death      | lgbm_ranker     | current_state_v2  |           50 |      0 |   0.8903 |     0.0418 |     0.8084 |           0.8480 |            0.9486 |                       61 |                    2706 |         3.5172 |
| hep__lgbm_ranker__v3_hepatic_schema__K50__s0     | hepatic    | lgbm_ranker     | v3_hepatic_schema |           50 |      0 |   0.7831 |     0.1260 |     0.5347 |           0.6584 |            0.7722 |                       38 |                    1771 |         3.2457 |
| hep__lgbm_ranker__v3_hepatic_schema__K50__s2     | hepatic    | lgbm_ranker     | v3_hepatic_schema |           50 |      2 |   0.7830 |     0.1268 |     0.5259 |           0.6696 |            0.7770 |                       38 |                    1771 |         3.1715 |
| hep__lgbm_ranker__NIT_plus_scores__K50__s0       | hepatic    | lgbm_ranker     | NIT_plus_scores   |           50 |      0 |   0.7786 |     0.1204 |     0.5494 |           0.6424 |            0.7108 |                       38 |                    1771 |         1.8988 |
| hep__lgbm_ranker__v3_hepatic_schema__K50__bag    | hepatic    | lgbm_ranker     | v3_hepatic_schema |           50 |     -1 |   0.7785 |     0.1306 |     0.5155 |           0.6823 |            0.7728 |                       -1 |                      -1 |         0.0000 |
| hep__xgb_ranker__NIT_plus_scores__K50__s0        | hepatic    | xgb_ranker      | NIT_plus_scores   |           50 |      0 |   0.7759 |     0.1216 |     0.5259 |           0.6455 |            0.7020 |                       38 |                    1771 |        81.7580 |
| hep__lgbm_ranker__cross_fs_ensemble              | hepatic    | lgbm_ranker     | ensemble          |           -1 |     -1 |   0.7740 |     0.1193 |     0.5865 |           0.7312 |            0.8077 |                       -1 |                      -1 |         0.0000 |
| hep__lgbm_ranker__biomarker_only__K20__s0        | hepatic    | lgbm_ranker     | biomarker_only    |           20 |      0 |   0.7716 |     0.1213 |     0.5769 |           0.7010 |            0.7832 |                       38 |                     736 |         2.1319 |
| hep__lgbm_ranker__biomarker_only__K50__s0        | hepatic    | lgbm_ranker     | biomarker_only    |           50 |      0 |   0.7688 |     0.1283 |     0.4932 |           0.6345 |            0.7518 |                       38 |                    1771 |         2.5730 |
| hep__lgbm_ranker__v3_hepatic_schema__K20__s0     | hepatic    | lgbm_ranker     | v3_hepatic_schema |           20 |      0 |   0.7652 |     0.1158 |     0.5873 |           0.6973 |            0.8068 |                       38 |                     736 |         2.8298 |
| hep__xgb_ranker__biomarker_only__K50__s0         | hepatic    | xgb_ranker      | biomarker_only    |           50 |      0 |   0.7623 |     0.1262 |     0.4932 |           0.6449 |            0.7326 |                       38 |                    1771 |        77.4408 |
| hep__lgbm_ranker__v3_hepatic_schema__K50__s1     | hepatic    | lgbm_ranker     | v3_hepatic_schema |           50 |      1 |   0.7614 |     0.1350 |     0.5070 |           0.6372 |            0.7628 |                       38 |                    1771 |         3.2173 |
| hep__lgbm_ranker__v3_hepatic_schema__K50__s3     | hepatic    | lgbm_ranker     | v3_hepatic_schema |           50 |      3 |   0.7614 |     0.1268 |     0.5012 |           0.6599 |            0.7670 |                       38 |                    1771 |         3.2685 |
| hep__xgb_ranker__biomarker_only__K20__s0         | hepatic    | xgb_ranker      | biomarker_only    |           20 |      0 |   0.7570 |     0.1273 |     0.5203 |           0.6915 |            0.7648 |                       38 |                     736 |        42.7512 |
| hep__xgb_ranker__v3_hepatic_schema__K50__s0      | hepatic    | xgb_ranker      | v3_hepatic_schema |           50 |      0 |   0.7536 |     0.1311 |     0.4892 |           0.6605 |            0.7614 |                       38 |                    1771 |        53.1611 |
| hep__lgbm_ranker__NIT_plus_scores__K20__s0       | hepatic    | lgbm_ranker     | NIT_plus_scores   |           20 |      0 |   0.7509 |     0.1338 |     0.5625 |           0.6667 |            0.7410 |                       38 |                     736 |         1.5475 |
| hep__xgb_ranker__NIT_plus_scores__K20__s0        | hepatic    | xgb_ranker      | NIT_plus_scores   |           20 |      0 |   0.7481 |     0.1222 |     0.5636 |           0.6669 |            0.7091 |                       38 |                     736 |        17.1163 |
| hep__xgb_ranker__v3_hepatic_schema__K20__s0      | hepatic    | xgb_ranker      | v3_hepatic_schema |           20 |      0 |   0.7480 |     0.1346 |     0.5116 |           0.7056 |            0.7969 |                       38 |                     736 |       108.6088 |
| hep__pairwise_logreg__NIT_plus_scores__K20__s0   | hepatic    | pairwise_logreg | NIT_plus_scores   |           20 |      0 |   0.6396 |     0.1201 |     0.5129 |           0.1080 |            0.3471 |                       38 |                     736 |         0.2035 |
| hep__pairwise_logreg__v3_hepatic_schema__K20__s0 | hepatic    | pairwise_logreg | v3_hepatic_schema |           20 |      0 |   0.6364 |     0.0961 |     0.5474 |           0.0061 |            0.5600 |                       38 |                     736 |         0.4675 |
| hep__pairwise_logreg__biomarker_only__K20__s0    | hepatic    | pairwise_logreg | biomarker_only    |           20 |      0 |   0.6235 |     0.1139 |     0.3937 |           0.1344 |            0.3775 |                       38 |                     736 |         0.3749 |
| hep__pairwise_logreg__biomarker_only__K50__s0    | hepatic    | pairwise_logreg | biomarker_only    |           50 |      0 |   0.6089 |     0.0732 |     0.4710 |           0.0629 |            0.3350 |                       38 |                    1771 |         1.0948 |
| hep__pairwise_logreg__NIT_plus_scores__K50__s0   | hepatic    | pairwise_logreg | NIT_plus_scores   |           50 |      0 |   0.6078 |     0.1141 |     0.4574 |           0.0238 |            0.3290 |                       38 |                    1771 |         0.4944 |
| hep__pairwise_logreg__v3_hepatic_schema__K50__s0 | hepatic    | pairwise_logreg | v3_hepatic_schema |           50 |      0 |   0.6007 |     0.0886 |     0.5008 |          -0.0094 |            0.4819 |                       38 |                    1771 |         1.2491 |

## C. Pair counts per fold (typical)

| endpoint   |   median_events_per_fold |   median_pairs_per_fold |
|:-----------|-------------------------:|------------------------:|
| hepatic    |                       38 |                     736 |
| death      |                       61 |                    2706 |

## D. Blend grid vs anchor

| blend                |   alpha | side     | source     |   hep_oof |   death_oof |   weighted_oof |   hep_fold_std |   hep_fold_min |   dea_fold_std |   dea_fold_min |   rho_h_test_anchor |   rho_d_test_anchor |   delta_w_vs_anchor |   delta_h_vs_anchor |   delta_d_vs_anchor |
|:---------------------|--------:|:---------|:-----------|----------:|------------:|---------------:|---------------:|---------------:|---------------:|---------------:|--------------------:|--------------------:|--------------------:|--------------------:|--------------------:|
| alpha=0.95_hep_only  |  0.9500 | hep_only | hep_ranker |    0.8502 |      0.9537 |         0.8813 |         0.1104 |         0.5953 |         0.0123 |         0.9269 |              0.9995 |              1.0000 |              0.0002 |              0.0002 |              0.0000 |
| capped=0.05_hep_only |  0.9500 | hep_only | capped     |    0.8502 |      0.9537 |         0.8813 |         0.1104 |         0.5953 |         0.0123 |         0.9269 |              0.9995 |              1.0000 |              0.0002 |              0.0002 |              0.0000 |
| alpha=0.9_hep_only   |  0.9000 | hep_only | hep_ranker |    0.8498 |      0.9537 |         0.8810 |         0.1119 |         0.5969 |         0.0123 |         0.9269 |              0.9980 |              1.0000 |             -0.0002 |             -0.0002 |              0.0000 |
| capped=0.10_hep_only |  0.9000 | hep_only | capped     |    0.8498 |      0.9537 |         0.8810 |         0.1119 |         0.5969 |         0.0123 |         0.9269 |              0.9980 |              1.0000 |             -0.0002 |             -0.0002 |              0.0000 |
| capped=0.15_hep_only |  0.8500 | hep_only | capped     |    0.8494 |      0.9537 |         0.8807 |         0.1134 |         0.5938 |         0.0123 |         0.9269 |              0.9952 |              1.0000 |             -0.0005 |             -0.0006 |              0.0000 |
| alpha=0.85_hep_only  |  0.8500 | hep_only | hep_ranker |    0.8494 |      0.9537 |         0.8807 |         0.1134 |         0.5938 |         0.0123 |         0.9269 |              0.9952 |              1.0000 |             -0.0005 |             -0.0006 |              0.0000 |
| capped=0.20_hep_only |  0.8000 | hep_only | capped     |    0.8486 |      0.9537 |         0.8801 |         0.1146 |         0.5922 |         0.0123 |         0.9269 |              0.9913 |              1.0000 |             -0.0010 |             -0.0014 |              0.0000 |
| alpha=0.8_hep_only   |  0.8000 | hep_only | hep_ranker |    0.8486 |      0.9537 |         0.8801 |         0.1146 |         0.5922 |         0.0123 |         0.9269 |              0.9913 |              1.0000 |             -0.0010 |             -0.0014 |              0.0000 |
| alpha=0.75_hep_only  |  0.7500 | hep_only | hep_ranker |    0.8472 |      0.9537 |         0.8791 |         0.1156 |         0.5907 |         0.0123 |         0.9269 |              0.9860 |              1.0000 |             -0.0020 |             -0.0028 |              0.0000 |
| capped=0.25_hep_only |  0.7500 | hep_only | capped     |    0.8472 |      0.9537 |         0.8791 |         0.1156 |         0.5907 |         0.0123 |         0.9269 |              0.9860 |              1.0000 |             -0.0020 |             -0.0028 |              0.0000 |

## E. Candidates emitted

**No candidates emitted.** No (ranker, blend) combination produced an OOF improvement of ≥ +0.002 weighted or ≥ +0.003 hepatic over `phase3_10_horizon_blend_v2`.

## F. Recommendation

**Do not submit.** Hold the anchor `phase3_10_horizon_blend_v2` (LB 0.91093). The pairwise ranking objective did not move OOF enough to justify a public-LB slot.


## Notes

- All ranker training is fold-internal: groups/pairs are built only from training-fold rows; validation folds receive *one* prediction per patient and are scored with the standard survival C-index.
- LightGBM/XGBoost rankers use the actual ranker objectives (`lambdarank`, `rank:pairwise`); the pairwise logistic baseline uses a linear score `w · x` derived from logistic regression on difference vectors.
- No event/censoring-age features. The horizon-classifier components of the anchor are unchanged in any blended candidate.
