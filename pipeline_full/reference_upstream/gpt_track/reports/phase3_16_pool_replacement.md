# Phase 3.16 — pool-replacement test (LambdaRanker into phase3_10 pool)

Anchor: `phase3_10_horizon_blend_v2` (LB **0.91093**, weighted OOF 0.8811, hep 0.8500, dea 0.9537).

Anchor fold: hep std/min=0.1084/0.6000, dea std/min=0.0123/0.9269.

## A. LambdaRanker components added to the hepatic pool

| component                                          |   ci_oof |   rho_h_oof_anchor |   rho_h_test_anchor |
|:---------------------------------------------------|---------:|-------------------:|--------------------:|
| hep__lambdaranker__v3_hepatic_schema__K50__s0      |   0.7831 |             0.6584 |              0.7722 |
| hep__lambdaranker__v3_hepatic_schema__K50__s2      |   0.7830 |             0.6696 |              0.7770 |
| hep__lambdaranker__v3_hepatic_schema__K50__bag_s02 |   0.7875 |             0.6809 |              0.7759 |

## B. Greedy capped at 25% horizon contribution (re-run with extended pool)

- hepatic pool size = 142, death pool size = 77
- hepatic chosen weights: {'_v3': 0.7905138339920948, 'NIT_plus_scores__hepatic__h1__lgbm_binary': 0.11857707509881421, 'NIT_plus_scores__hepatic__h3p5__lgbm_binary__soft_weight': 0.09090909090909091}
- hepatic chosen survival C-index: 0.8513
- death chosen weights: {'_v3': 0.7905138339920948, 'current_state_v2__death__h5__catboost_binary__s3': 0.11857707509881421, 'NIT_plus_scores__death__h4__catboost_binary': 0.09090909090909091}
- death chosen survival C-index: 0.9537

## C. Endpoint-specific greedy ensembles (alpha-grid components)

- hepatic ensemble: ['hep__lambdaranker__v3_hepatic_schema__K50__bag_s02', 'NIT_plus_scores__hepatic__h3p5__lgbm_binary__soft_weight', 'v3_hepatic_schema__hepatic__h4__catboost_binary__downweight', 'hep__lambdaranker__v3_hepatic_schema__K50__s0', 'NIT_plus_scores__hepatic__h1__lgbm_binary__soft_weight'] → 0.8192
- death ensemble: ['current_state_v2__death__h5__catboost_binary__downweight__s3', 'current_state_v2__death__h6__catboost_binary', 'current_state_v2__death__h4__xgb_binary', 'current_state_v2__death__h6p5__catboost_binary__s3', 'NIT_plus_scores__death__h6__xgb_binary', 'current_state_v2__death__h5__catboost_binary__s3', 'NIT_plus_scores__death__h4__catboost_binary', 'current_state_v2__death__h6p5__catboost_binary__s1'] → 0.9452

## D. Blend grid vs anchor

| blend                        |    alpha | side     | source        |   hep_oof |   death_oof |   weighted_oof |   hep_fold_std |   hep_fold_min |   dea_fold_std |   dea_fold_min |   rho_h_test_anchor |   rho_d_test_anchor |   rho_h_oof_anchor |   rho_d_oof_anchor |
|:-----------------------------|---------:|:---------|:--------------|----------:|------------:|---------------:|---------------:|---------------:|---------------:|---------------:|--------------------:|--------------------:|-------------------:|-------------------:|
| ensemble_alpha=0.8_both      |   0.8000 | both     | ensemble      |    0.8519 |      0.9544 |         0.8827 |         0.1142 |         0.5899 |         0.0142 |         0.9237 |              0.9904 |              0.9975 |             0.9883 |             0.9960 |
| ensemble_alpha=0.75_both     |   0.7500 | both     | ensemble      |    0.8519 |      0.9542 |         0.8826 |         0.1149 |         0.5907 |         0.0152 |         0.9215 |              0.9846 |              0.9961 |             0.9810 |             0.9936 |
| ensemble_alpha=0.9_both      |   0.9000 | both     | ensemble      |    0.8518 |      0.9543 |         0.8825 |         0.1123 |         0.5907 |         0.0134 |         0.9248 |              0.9977 |              0.9994 |             0.9973 |             0.9990 |
| ensemble_alpha=0.75_hep_only |   0.7500 | hep_only | ensemble      |    0.8519 |      0.9537 |         0.8825 |         0.1149 |         0.5907 |         0.0123 |         0.9269 |              0.9846 |              1.0000 |             0.9810 |             1.0000 |
| ensemble_alpha=0.8_hep_only  |   0.8000 | hep_only | ensemble      |    0.8519 |      0.9537 |         0.8825 |         0.1142 |         0.5899 |         0.0123 |         0.9269 |              0.9904 |              1.0000 |             0.9883 |             1.0000 |
| ensemble_alpha=0.85_both     |   0.8500 | both     | ensemble      |    0.8517 |      0.9543 |         0.8825 |         0.1131 |         0.5907 |         0.0138 |         0.9237 |              0.9946 |              0.9986 |             0.9936 |             0.9978 |
| ensemble_alpha=0.9_hep_only  |   0.9000 | hep_only | ensemble      |    0.8518 |      0.9537 |         0.8824 |         0.1123 |         0.5907 |         0.0123 |         0.9269 |              0.9977 |              1.0000 |             0.9973 |             1.0000 |
| ensemble_alpha=0.85_hep_only |   0.8500 | hep_only | ensemble      |    0.8517 |      0.9537 |         0.8823 |         0.1131 |         0.5907 |         0.0123 |         0.9269 |              0.9946 |              1.0000 |             0.9936 |             1.0000 |
| greedy_cap25_both            | nan      | both     | greedy_capped |    0.8513 |      0.9537 |         0.8821 |         0.1094 |         0.5984 |         0.0123 |         0.9269 |              0.9971 |              1.0000 |             0.9959 |             1.0000 |
| greedy_cap25_hep_only        | nan      | hep_only | greedy_capped |    0.8513 |      0.9537 |         0.8821 |         0.1094 |         0.5984 |         0.0123 |         0.9269 |              0.9971 |              1.0000 |             0.9959 |             1.0000 |
| ensemble_alpha=0.95_both     |   0.9500 | both     | ensemble      |    0.8510 |      0.9541 |         0.8819 |         0.1114 |         0.5922 |         0.0127 |         0.9269 |              0.9994 |              0.9998 |             0.9993 |             0.9998 |
| ensemble_alpha=0.95_hep_only |   0.9500 | hep_only | ensemble      |    0.8510 |      0.9537 |         0.8818 |         0.1114 |         0.5922 |         0.0123 |         0.9269 |              0.9994 |              1.0000 |             0.9993 |             1.0000 |
| ensemble_alpha=0.8_dea_only  |   0.8000 | dea_only | ensemble      |    0.8500 |      0.9544 |         0.8813 |         0.1084 |         0.6000 |         0.0142 |         0.9237 |              1.0000 |              0.9975 |             1.0000 |             0.9960 |
| ensemble_alpha=0.9_dea_only  |   0.9000 | dea_only | ensemble      |    0.8500 |      0.9543 |         0.8813 |         0.1084 |         0.6000 |         0.0134 |         0.9248 |              1.0000 |              0.9994 |             1.0000 |             0.9990 |
| ensemble_alpha=0.85_dea_only |   0.8500 | dea_only | ensemble      |    0.8500 |      0.9543 |         0.8813 |         0.1084 |         0.6000 |         0.0138 |         0.9237 |              1.0000 |              0.9986 |             1.0000 |             0.9978 |
| ensemble_alpha=0.75_dea_only |   0.7500 | dea_only | ensemble      |    0.8500 |      0.9542 |         0.8813 |         0.1084 |         0.6000 |         0.0152 |         0.9215 |              1.0000 |              0.9961 |             1.0000 |             0.9936 |
| ensemble_alpha=0.95_dea_only |   0.9500 | dea_only | ensemble      |    0.8500 |      0.9541 |         0.8813 |         0.1084 |         0.6000 |         0.0127 |         0.9269 |              1.0000 |              0.9998 |             1.0000 |             0.9998 |
| single_alpha=0.95_hep_only   |   0.9500 | hep_only | best_single   |    0.8502 |      0.9537 |         0.8812 |         0.1100 |         0.5969 |         0.0123 |         0.9269 |              0.9995 |              1.0000 |             0.9993 |             1.0000 |
| single_alpha=0.95_both       |   0.9500 | both     | best_single   |    0.8502 |      0.9537 |         0.8812 |         0.1100 |         0.5969 |         0.0141 |         0.9226 |              0.9995 |              0.9997 |             0.9993 |             0.9995 |
| anchor_phase3_10_v2          |   1.0000 | none     | anchor        |    0.8500 |      0.9537 |         0.8811 |         0.1084 |         0.6000 |         0.0123 |         0.9269 |              1.0000 |              1.0000 |             1.0000 |             1.0000 |
| single_alpha=0.95_dea_only   |   0.9500 | dea_only | best_single   |    0.8500 |      0.9537 |         0.8811 |         0.1084 |         0.6000 |         0.0141 |         0.9226 |              1.0000 |              0.9997 |             1.0000 |             0.9995 |
| greedy_cap25_dea_only        | nan      | dea_only | greedy_capped |    0.8500 |      0.9537 |         0.8811 |         0.1084 |         0.6000 |         0.0123 |         0.9269 |              1.0000 |              1.0000 |             1.0000 |             1.0000 |
| single_alpha=0.9_hep_only    |   0.9000 | hep_only | best_single   |    0.8499 |      0.9537 |         0.8811 |         0.1120 |         0.5969 |         0.0123 |         0.9269 |              0.9980 |              1.0000 |             0.9973 |             1.0000 |
| single_alpha=0.9_dea_only    |   0.9000 | dea_only | best_single   |    0.8500 |      0.9535 |         0.8810 |         0.1084 |         0.6000 |         0.0151 |         0.9193 |              1.0000 |              0.9987 |             1.0000 |             0.9980 |
| single_alpha=0.9_both        |   0.9000 | both     | best_single   |    0.8499 |      0.9535 |         0.8810 |         0.1120 |         0.5969 |         0.0151 |         0.9193 |              0.9980 |              0.9987 |             0.9973 |             0.9980 |
| single_alpha=0.85_dea_only   |   0.8500 | dea_only | best_single   |    0.8500 |      0.9531 |         0.8809 |         0.1084 |         0.6000 |         0.0166 |         0.9171 |              1.0000 |              0.9971 |             1.0000 |             0.9953 |
| single_alpha=0.8_dea_only    |   0.8000 | dea_only | best_single   |    0.8500 |      0.9526 |         0.8808 |         0.1084 |         0.6000 |         0.0187 |         0.9138 |              1.0000 |              0.9949 |             1.0000 |             0.9915 |
| single_alpha=0.85_hep_only   |   0.8500 | hep_only | best_single   |    0.8494 |      0.9537 |         0.8807 |         0.1129 |         0.5953 |         0.0123 |         0.9269 |              0.9952 |              1.0000 |             0.9936 |             1.0000 |
| single_alpha=0.75_dea_only   |   0.7500 | dea_only | best_single   |    0.8500 |      0.9521 |         0.8806 |         0.1084 |         0.6000 |         0.0197 |         0.9106 |              1.0000 |              0.9920 |             1.0000 |             0.9864 |
| single_alpha=0.85_both       |   0.8500 | both     | best_single   |    0.8494 |      0.9531 |         0.8805 |         0.1129 |         0.5953 |         0.0166 |         0.9171 |              0.9952 |              0.9971 |             0.9936 |             0.9953 |
| single_alpha=0.8_hep_only    |   0.8000 | hep_only | best_single   |    0.8479 |      0.9537 |         0.8796 |         0.1146 |         0.5922 |         0.0123 |         0.9269 |              0.9914 |              1.0000 |             0.9881 |             1.0000 |
| single_alpha=0.8_both        |   0.8000 | both     | best_single   |    0.8479 |      0.9526 |         0.8793 |         0.1146 |         0.5922 |         0.0187 |         0.9138 |              0.9914 |              0.9949 |             0.9881 |             0.9915 |
| single_alpha=0.75_hep_only   |   0.7500 | hep_only | best_single   |    0.8470 |      0.9537 |         0.8790 |         0.1153 |         0.5922 |         0.0123 |         0.9269 |              0.9863 |              1.0000 |             0.9808 |             1.0000 |
| single_alpha=0.75_both       |   0.7500 | both     | best_single   |    0.8470 |      0.9521 |         0.8785 |         0.1153 |         0.5922 |         0.0197 |         0.9106 |              0.9863 |              0.9920 |             0.9808 |             0.9864 |

## E. Candidate decision

**No submission recommended.** Even with the LambdaRankers in the candidate pool, no blend variant clears the +0.002 weighted or +0.003 hepatic threshold over `phase3_10_horizon_blend_v2`.


## Notes

- LambdaRanker training reuses the Phase 3.15 recipe: LightGBM with `objective='lambdarank'`, K=50 controls per event, fold-internal training only.
- The candidate pool is the union of phase3_9 + phase3_10 + phase3_12 horizon classifiers plus the LambdaRanker(s); the same greedy capped logic that built phase3_10_horizon_blend_v2 is applied.
- Death components are unchanged from the anchor's pool unless a death-side improvement clearly emerges from the greedy logic.
- No event/censoring-age features used for any model.
