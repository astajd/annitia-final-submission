# Phase 3.9 — horizon-specific risk models

Reference: `phase3_5_current_state_v3_hepatic_focused` LB **0.89147**, OOF hepatic=0.8415 / death=0.9496 / weighted=0.8739.

## Horizon label counts

| endpoint   |   horizon_years |   n_total |   n_usable |   n_positive |   n_negative |   n_censored_before_h |   positive_rate |
|:-----------|----------------:|----------:|-----------:|-------------:|-------------:|----------------------:|----------------:|
| hepatic    |          1.0000 |      1253 |       1114 |           12 |         1102 |                   139 |          0.0108 |
| hepatic    |          3.0000 |      1253 |        851 |           21 |          830 |                   402 |          0.0247 |
| hepatic    |          5.0000 |      1253 |        573 |           25 |          548 |                   680 |          0.0436 |
| death      |          1.0000 |      1253 |       1114 |            0 |         1114 |                   139 |          0.0000 |
| death      |          3.0000 |      1253 |        860 |            4 |          856 |                   393 |          0.0047 |
| death      |          5.0000 |      1253 |        595 |           16 |          579 |                   658 |          0.0269 |

## Per-run results (sorted by survival C-index per endpoint)

| label                                           | feature_set       | endpoint   |   horizon_years | model           |   n_features |   n_usable_train |   n_positive_train |   horizon_auc |   surv_cindex_mean |   surv_cindex_std |   surv_cindex_min |   wall_seconds |   rho_oof_v3 |   rho_test_v3 |
|:------------------------------------------------|:------------------|:-----------|----------------:|:----------------|-------------:|-----------------:|-------------------:|--------------:|-------------------:|------------------:|------------------:|---------------:|-------------:|--------------:|
| current_state_v2__death__h5__catboost_binary    | current_state_v2  | death      |          5.0000 | catboost_binary |          222 |              595 |                 16 |        0.9805 |             0.9032 |            0.0485 |            0.7808 |         7.8510 |       0.7876 |        0.8968 |
| biomarker_only__death__h5__xgb_binary           | biomarker_only    | death      |          5.0000 | xgb_binary      |          127 |              595 |                 16 |        0.9399 |             0.8864 |            0.0322 |            0.8108 |        27.9137 |       0.6413 |        0.6776 |
| v3_hepatic_schema__death__h5__xgb_binary        | v3_hepatic_schema | death      |          5.0000 | xgb_binary      |          215 |              595 |                 16 |        0.9284 |             0.8833 |            0.0273 |            0.8219 |        48.9336 |       0.6844 |        0.7168 |
| v3_hepatic_schema__death__h5__catboost_binary   | v3_hepatic_schema | death      |          5.0000 | catboost_binary |          215 |              595 |                 16 |        0.9207 |             0.8776 |            0.0356 |            0.7898 |         7.7480 |       0.6880 |        0.7443 |
| current_state_v2__death__h5__xgb_binary         | current_state_v2  | death      |          5.0000 | xgb_binary      |          222 |              595 |                 16 |        0.9780 |             0.8750 |            0.0380 |            0.8136 |        20.7460 |       0.7181 |        0.7704 |
| biomarker_only__death__h5__lgbm_binary          | biomarker_only    | death      |          5.0000 | lgbm_binary     |          127 |              595 |                 16 |        0.9058 |             0.8696 |            0.0314 |            0.7884 |         1.1230 |       0.6324 |        0.7022 |
| biomarker_only__death__h5__catboost_binary      | biomarker_only    | death      |          5.0000 | catboost_binary |          127 |              595 |                 16 |        0.9076 |             0.8637 |            0.0614 |            0.7459 |         5.6687 |       0.5737 |        0.6560 |
| v3_hepatic_schema__death__h5__lgbm_binary       | v3_hepatic_schema | death      |          5.0000 | lgbm_binary     |          215 |              595 |                 16 |        0.9018 |             0.8630 |            0.0263 |            0.8248 |         1.2550 |       0.6658 |        0.7344 |
| current_state_v2__death__h5__lgbm_binary        | current_state_v2  | death      |          5.0000 | lgbm_binary     |          222 |              595 |                 16 |        0.9737 |             0.8586 |            0.0440 |            0.7619 |         1.0220 |       0.7467 |        0.8046 |
| NIT_plus_scores__death__h5__catboost_binary     | NIT_plus_scores   | death      |          5.0000 | catboost_binary |           55 |              595 |                 16 |        0.9121 |             0.8563 |            0.0603 |            0.6852 |         3.5600 |       0.6016 |        0.5988 |
| NIT_plus_scores__death__h5__xgb_binary          | NIT_plus_scores   | death      |          5.0000 | xgb_binary      |           55 |              595 |                 16 |        0.8977 |             0.8551 |            0.0543 |            0.7297 |        10.6024 |       0.6336 |        0.6507 |
| NIT_plus_scores__death__h5__lgbm_binary         | NIT_plus_scores   | death      |          5.0000 | lgbm_binary     |           55 |              595 |                 16 |        0.8875 |             0.8464 |            0.0555 |            0.7190 |         0.9439 |       0.6515 |        0.6632 |
| v3_hepatic_schema__hepatic__h3__lgbm_binary     | v3_hepatic_schema | hepatic    |          3.0000 | lgbm_binary     |          215 |              851 |                 21 |        0.7956 |             0.7672 |            0.1256 |            0.5460 |         1.8352 |       0.4221 |        0.5247 |
| NIT_plus_scores__hepatic__h3__lgbm_binary       | NIT_plus_scores   | hepatic    |          3.0000 | lgbm_binary     |           55 |              851 |                 21 |        0.8079 |             0.7618 |            0.1023 |            0.6209 |         1.2400 |       0.3907 |        0.3964 |
| NIT_plus_scores__hepatic__h5__lgbm_binary       | NIT_plus_scores   | hepatic    |          5.0000 | lgbm_binary     |           55 |              573 |                 25 |        0.7972 |             0.7455 |            0.1226 |            0.5171 |         1.0542 |       0.4131 |        0.5478 |
| v3_hepatic_schema__hepatic__h3__catboost_binary | v3_hepatic_schema | hepatic    |          3.0000 | catboost_binary |          215 |              851 |                 21 |        0.7663 |             0.7448 |            0.1343 |            0.5124 |         7.7949 |       0.4827 |        0.6214 |
| current_state_v2__hepatic__h3__catboost_binary  | current_state_v2  | hepatic    |          3.0000 | catboost_binary |          222 |              851 |                 21 |        0.7730 |             0.7399 |            0.1332 |            0.4908 |         8.0411 |       0.4846 |        0.6180 |
| NIT_plus_scores__hepatic__h3__xgb_binary        | NIT_plus_scores   | hepatic    |          3.0000 | xgb_binary      |           55 |              851 |                 21 |        0.7834 |             0.7383 |            0.1122 |            0.5612 |        13.6124 |       0.3981 |        0.4139 |
| v3_hepatic_schema__hepatic__h3__xgb_binary      | v3_hepatic_schema | hepatic    |          3.0000 | xgb_binary      |          215 |              851 |                 21 |        0.7788 |             0.7369 |            0.1100 |            0.5401 |        77.5589 |       0.4702 |        0.5643 |
| v3_hepatic_schema__hepatic__h5__lgbm_binary     | v3_hepatic_schema | hepatic    |          5.0000 | lgbm_binary     |          215 |              573 |                 25 |        0.7862 |             0.7318 |            0.1338 |            0.4498 |         1.4275 |       0.4698 |        0.6125 |
| v3_hepatic_schema__hepatic__h5__catboost_binary | v3_hepatic_schema | hepatic    |          5.0000 | catboost_binary |          215 |              573 |                 25 |        0.7947 |             0.7312 |            0.1178 |            0.5402 |         7.7905 |       0.4886 |        0.6575 |
| current_state_v2__hepatic__h3__lgbm_binary      | current_state_v2  | hepatic    |          3.0000 | lgbm_binary     |          222 |              851 |                 21 |        0.7744 |             0.7310 |            0.1487 |            0.4643 |         1.7710 |       0.4305 |        0.5753 |
| NIT_plus_scores__hepatic__h5__xgb_binary        | NIT_plus_scores   | hepatic    |          5.0000 | xgb_binary      |           55 |              573 |                 25 |        0.7862 |             0.7295 |            0.1294 |            0.5147 |         5.1979 |       0.4006 |        0.5213 |
| NIT_plus_scores__hepatic__h1__lgbm_binary       | NIT_plus_scores   | hepatic    |          1.0000 | lgbm_binary     |           55 |             1114 |                 12 |        0.7948 |             0.7187 |            0.1388 |            0.4372 |         1.0030 |       0.2363 |        0.2499 |
| current_state_v2__hepatic__h3__xgb_binary       | current_state_v2  | hepatic    |          3.0000 | xgb_binary      |          222 |              851 |                 21 |        0.7768 |             0.7174 |            0.1167 |            0.5158 |        56.1935 |       0.4758 |        0.5987 |
| biomarker_only__hepatic__h3__lgbm_binary        | biomarker_only    | hepatic    |          3.0000 | lgbm_binary     |          127 |              851 |                 21 |        0.7373 |             0.7169 |            0.1195 |            0.4279 |         1.6014 |       0.4151 |        0.5541 |
| current_state_v2__hepatic__h5__catboost_binary  | current_state_v2  | hepatic    |          5.0000 | catboost_binary |          222 |              573 |                 25 |        0.8736 |             0.7142 |            0.1126 |            0.5269 |         7.8718 |       0.4247 |        0.5918 |
| v3_hepatic_schema__hepatic__h5__xgb_binary      | v3_hepatic_schema | hepatic    |          5.0000 | xgb_binary      |          215 |              573 |                 25 |        0.7619 |             0.7131 |            0.1421 |            0.4502 |        43.0647 |       0.4706 |        0.6146 |
| biomarker_only__hepatic__h5__lgbm_binary        | biomarker_only    | hepatic    |          5.0000 | lgbm_binary     |          127 |              573 |                 25 |        0.7602 |             0.7128 |            0.1112 |            0.4916 |         1.3569 |       0.4373 |        0.5853 |
| NIT_plus_scores__hepatic__h3__catboost_binary   | NIT_plus_scores   | hepatic    |          3.0000 | catboost_binary |           55 |              851 |                 21 |        0.7173 |             0.7118 |            0.1219 |            0.5535 |         3.6549 |       0.4235 |        0.4660 |
| biomarker_only__hepatic__h5__catboost_binary    | biomarker_only    | hepatic    |          5.0000 | catboost_binary |          127 |              573 |                 25 |        0.7579 |             0.7109 |            0.1219 |            0.4797 |         5.7496 |       0.4687 |        0.6164 |
| biomarker_only__hepatic__h5__xgb_binary         | biomarker_only    | hepatic    |          5.0000 | xgb_binary      |          127 |              573 |                 25 |        0.7545 |             0.7083 |            0.1162 |            0.4789 |        25.6001 |       0.4331 |        0.5961 |
| biomarker_only__hepatic__h3__catboost_binary    | biomarker_only    | hepatic    |          3.0000 | catboost_binary |          127 |              851 |                 21 |        0.7102 |             0.7081 |            0.1237 |            0.4757 |         5.7207 |       0.4607 |        0.5831 |
| current_state_v2__hepatic__h1__catboost_binary  | current_state_v2  | hepatic    |          1.0000 | catboost_binary |          222 |             1114 |                 12 |        0.7760 |             0.7069 |            0.1008 |            0.4861 |         8.1471 |       0.2607 |        0.3097 |
| v3_hepatic_schema__hepatic__h1__catboost_binary | v3_hepatic_schema | hepatic    |          1.0000 | catboost_binary |          215 |             1114 |                 12 |        0.7501 |             0.7030 |            0.1158 |            0.4637 |         7.9393 |       0.2381 |        0.3023 |
| biomarker_only__hepatic__h3__xgb_binary         | biomarker_only    | hepatic    |          3.0000 | xgb_binary      |          127 |              851 |                 21 |        0.7281 |             0.7027 |            0.1015 |            0.4972 |        23.5743 |       0.4433 |        0.5723 |
| current_state_v2__hepatic__h5__lgbm_binary      | current_state_v2  | hepatic    |          5.0000 | lgbm_binary     |          222 |              573 |                 25 |        0.8228 |             0.6922 |            0.1286 |            0.4702 |         1.5156 |       0.4095 |        0.5728 |
| NIT_plus_scores__hepatic__h1__xgb_binary        | NIT_plus_scores   | hepatic    |          1.0000 | xgb_binary      |           55 |             1114 |                 12 |        0.7717 |             0.6912 |            0.1109 |            0.4932 |        52.9294 |       0.2150 |        0.1732 |
| current_state_v2__hepatic__h5__xgb_binary       | current_state_v2  | hepatic    |          5.0000 | xgb_binary      |          222 |              573 |                 25 |        0.8097 |             0.6837 |            0.1367 |            0.4297 |        36.6137 |       0.4299 |        0.5721 |
| NIT_plus_scores__hepatic__h1__catboost_binary   | NIT_plus_scores   | hepatic    |          1.0000 | catboost_binary |           55 |             1114 |                 12 |        0.6776 |             0.6761 |            0.1128 |            0.4781 |         3.6456 |       0.2821 |        0.3635 |
| NIT_plus_scores__hepatic__h5__catboost_binary   | NIT_plus_scores   | hepatic    |          5.0000 | catboost_binary |           55 |              573 |                 25 |        0.7131 |             0.6749 |            0.1278 |            0.5195 |         3.4692 |       0.4301 |        0.5080 |
| current_state_v2__hepatic__h1__xgb_binary       | current_state_v2  | hepatic    |          1.0000 | xgb_binary      |          222 |             1114 |                 12 |        0.6975 |             0.6691 |            0.1197 |            0.4151 |        48.7036 |       0.2429 |        0.2377 |
| biomarker_only__hepatic__h1__catboost_binary    | biomarker_only    | hepatic    |          1.0000 | catboost_binary |          127 |             1114 |                 12 |        0.6887 |             0.6595 |            0.1265 |            0.3586 |         5.9585 |       0.2832 |        0.3859 |
| v3_hepatic_schema__hepatic__h1__xgb_binary      | v3_hepatic_schema | hepatic    |          1.0000 | xgb_binary      |          215 |             1114 |                 12 |        0.6417 |             0.6362 |            0.1249 |            0.3785 |        42.8621 |       0.2443 |        0.2521 |
| biomarker_only__hepatic__h1__xgb_binary         | biomarker_only    | hepatic    |          1.0000 | xgb_binary      |          127 |             1114 |                 12 |        0.7092 |             0.6228 |            0.1096 |            0.3785 |        23.6688 |       0.2251 |        0.2558 |
| biomarker_only__hepatic__h1__lgbm_binary        | biomarker_only    | hepatic    |          1.0000 | lgbm_binary     |          127 |             1114 |                 12 |        0.6646 |             0.6080 |            0.1222 |            0.3731 |         1.5329 |       0.2105 |        0.2215 |
| current_state_v2__hepatic__h1__lgbm_binary      | current_state_v2  | hepatic    |          1.0000 | lgbm_binary     |          222 |             1114 |                 12 |        0.6003 |             0.6038 |            0.1360 |            0.4295 |         1.8386 |       0.2152 |        0.2534 |
| v3_hepatic_schema__hepatic__h1__lgbm_binary     | v3_hepatic_schema | hepatic    |          1.0000 | lgbm_binary     |          215 |             1114 |                 12 |        0.6020 |             0.5953 |            0.1408 |            0.4047 |         1.7790 |       0.1928 |        0.2388 |

## Best horizon model per endpoint

- **hepatic**: `v3_hepatic_schema__hepatic__h3__lgbm_binary` survC=0.7672 std=0.1256 AUC@h=0.7956 (rank-corr w/ v3 OOF 0.422, test 0.525)
- **death**: `current_state_v2__death__h5__catboost_binary` survC=0.9032 std=0.0485 AUC@h=0.9805 (rank-corr w/ v3 OOF 0.788, test 0.897)

## Blends with v3 (rank-space)

| blend               |   alpha | side     |   hep_oof |   death_oof |   weighted_oof |   hep_fold_std |   hep_fold_min |   dea_fold_std |   dea_fold_min |
|:--------------------|--------:|:---------|----------:|------------:|---------------:|---------------:|---------------:|---------------:|---------------:|
| alpha=0.95_hep_only |  0.9500 | hep_only |    0.8434 |      0.9496 |         0.8753 |         0.1018 |         0.6109 |         0.0115 |         0.9269 |
| alpha=0.95_dea_only |  0.9500 | dea_only |    0.8415 |      0.9505 |         0.8742 |         0.0991 |         0.6202 |         0.0126 |         0.9269 |
| alpha=0.95_both     |  0.9500 | both     |    0.8434 |      0.9505 |         0.8755 |         0.1018 |         0.6109 |         0.0126 |         0.9269 |
| alpha=0.9_hep_only  |  0.9000 | hep_only |    0.8442 |      0.9496 |         0.8758 |         0.1044 |         0.6047 |         0.0115 |         0.9269 |
| alpha=0.9_dea_only  |  0.9000 | dea_only |    0.8415 |      0.9509 |         0.8743 |         0.0991 |         0.6202 |         0.0126 |         0.9280 |
| alpha=0.9_both      |  0.9000 | both     |    0.8442 |      0.9509 |         0.8762 |         0.1044 |         0.6047 |         0.0126 |         0.9280 |
| alpha=0.85_hep_only |  0.8500 | hep_only |    0.8451 |      0.9496 |         0.8765 |         0.1061 |         0.6016 |         0.0115 |         0.9269 |
| alpha=0.85_dea_only |  0.8500 | dea_only |    0.8415 |      0.9507 |         0.8742 |         0.0991 |         0.6202 |         0.0125 |         0.9269 |
| alpha=0.85_both     |  0.8500 | both     |    0.8451 |      0.9507 |         0.8768 |         0.1061 |         0.6016 |         0.0125 |         0.9269 |
| alpha=0.8_hep_only  |  0.8000 | hep_only |    0.8445 |      0.9496 |         0.8760 |         0.1075 |         0.5984 |         0.0115 |         0.9269 |
| alpha=0.8_dea_only  |  0.8000 | dea_only |    0.8415 |      0.9512 |         0.8744 |         0.0991 |         0.6202 |         0.0136 |         0.9248 |
| alpha=0.8_both      |  0.8000 | both     |    0.8445 |      0.9512 |         0.8765 |         0.1075 |         0.5984 |         0.0136 |         0.9248 |
| alpha=0.7_hep_only  |  0.7000 | hep_only |    0.8439 |      0.9496 |         0.8756 |         0.1114 |         0.5953 |         0.0115 |         0.9269 |
| alpha=0.7_dea_only  |  0.7000 | dea_only |    0.8415 |      0.9500 |         0.8740 |         0.0991 |         0.6202 |         0.0166 |         0.9193 |
| alpha=0.7_both      |  0.7000 | both     |    0.8439 |      0.9500 |         0.8757 |         0.1114 |         0.5953 |         0.0166 |         0.9193 |

## Promotion decision

**Promoted**: `alpha=0.85_both` (weighted Δ=+0.0029, hepatic Δ=+0.0037, contributing rho=0.422, hep_fold_std Δ=+nan). Submission: `gpt/submissions/20260428_0027_phase3_9_horizon_blend.csv`.

## Notes

- Horizon labels exclude patients censored before the horizon (default mode). We did not need the down-weight variant because the exclusion still leaves most patients usable at H = 1y / 3y / 5y for both endpoints.
- Risk score = horizon classifier's predicted positive-class probability; we evaluate it against the survival C-index of the original endpoint, which is what the contest scores.
- All training is fold-internal (preprocessing fit only on the training fold); no event/censoring-age columns are used to build features, only to define labels.
