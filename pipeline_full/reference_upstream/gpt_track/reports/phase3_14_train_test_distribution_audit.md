# Phase 3.14 — Track A1: train vs test distribution audit

Per feature set, per-feature drift metrics (KS statistic, mean shift, missingness diff). Scoped to feature sets used by `phase3_10_horizon_blend_v2`:

- `current_state_v2` (anchor's biggest hepatic+death contributor)
- `v3_hepatic_schema` (hep h3 LGBM seed=4 schema)
- `NIT_plus_scores` (hep h1 LGBM and dea h4 CatBoost)
- `biomarker_only` (longitudinal_no_followup_proxies, used in death horizon ensembles)

## Per feature set summary

| feature_set       |   n_features |   median_miss_diff |   p95_miss_diff |   median_mean_shift_z_abs |   p95_mean_shift_z_abs |   median_ks |   p95_ks |   n_features_ks_gt_0_2 |
|:------------------|-------------:|-------------------:|----------------:|--------------------------:|-----------------------:|------------:|---------:|-----------------------:|
| current_state_v2  |          222 |            -0.1056 |          0.1432 |                    0.0633 |                 0.1989 |      0.0509 |   0.1263 |                      0 |
| NIT_plus_scores   |           55 |            -0.0809 |          0.1876 |                    0.0486 |                 0.1005 |      0.0516 |   0.1282 |                      0 |
| biomarker_only    |          127 |            -0.1066 |          0.1432 |                    0.0445 |                 0.1097 |      0.0502 |   0.0765 |                      0 |
| v3_hepatic_schema |          215 |            -0.1056 |          0.1432 |                    0.0612 |                 0.1968 |      0.0509 |   0.1271 |                      0 |

## Top 25 features by KS across all sets

| feature_set       | feature               |   miss_train |   miss_test |   miss_diff |   mean_train |   mean_test |   mean_shift_z |     ks |
|:------------------|:----------------------|-------------:|------------:|------------:|-------------:|------------:|---------------:|-------:|
| current_state_v2  | bariatric_surgery_age |       0.9425 |      0.9267 |     -0.0158 |      49.5417 |     50.5806 |         0.0837 | 0.1635 |
| biomarker_only    | bariatric_surgery_age |       0.9425 |      0.9267 |     -0.0158 |      49.5417 |     50.5806 |         0.0837 | 0.1635 |
| NIT_plus_scores   | bariatric_surgery_age |       0.9425 |      0.9267 |     -0.0158 |      49.5417 |     50.5806 |         0.0837 | 0.1635 |
| v3_hepatic_schema | bariatric_surgery_age |       0.9425 |      0.9267 |     -0.0158 |      49.5417 |     50.5806 |         0.0837 | 0.1635 |
| v3_hepatic_schema | miss_chol             |       0.0000 |      0.0000 |      0.0000 |       0.8475 |      0.8188 |        -0.1939 | 0.1432 |
| current_state_v2  | miss_chol             |       0.0000 |      0.0000 |      0.0000 |       0.8475 |      0.8188 |        -0.1939 | 0.1432 |
| current_state_v2  | miss_total            |       0.0000 |      0.0000 |      0.0000 |       0.8298 |      0.8028 |        -0.2062 | 0.1424 |
| v3_hepatic_schema | miss_total            |       0.0000 |      0.0000 |      0.0000 |       0.8298 |      0.8028 |        -0.2062 | 0.1424 |
| v3_hepatic_schema | miss_bilirubin        |       0.0000 |      0.0000 |      0.0000 |       0.8148 |      0.7800 |        -0.1928 | 0.1424 |
| v3_hepatic_schema | miss_fibrotest_BM_2   |       0.0000 |      0.0000 |      0.0000 |       0.8155 |      0.7815 |        -0.1885 | 0.1424 |
| current_state_v2  | miss_fibrotest_BM_2   |       0.0000 |      0.0000 |      0.0000 |       0.8155 |      0.7815 |        -0.1885 | 0.1424 |
| current_state_v2  | miss_bilirubin        |       0.0000 |      0.0000 |      0.0000 |       0.8148 |      0.7800 |        -0.1928 | 0.1424 |
| current_state_v2  | miss_visit_3          |       0.0000 |      0.0000 |      0.0000 |       0.4521 |      0.3455 |        -0.2905 | 0.1420 |
| v3_hepatic_schema | miss_visit_3          |       0.0000 |      0.0000 |      0.0000 |       0.4521 |      0.3455 |        -0.2905 | 0.1420 |
| NIT_plus_scores   | fib4_v1               |       0.7366 |      0.6879 |     -0.0487 |       1.3536 |      1.4321 |         0.1000 | 0.1409 |
| current_state_v2  | fib4_v1               |       0.7366 |      0.6879 |     -0.0487 |       1.3536 |      1.4321 |         0.1000 | 0.1409 |
| v3_hepatic_schema | fib4_v1               |       0.7366 |      0.6879 |     -0.0487 |       1.3536 |      1.4321 |         0.1000 | 0.1409 |
| NIT_plus_scores   | fib4_v2               |       0.6760 |      0.6548 |     -0.0211 |       1.4379 |      1.3792 |        -0.0679 | 0.1322 |
| v3_hepatic_schema | fib4_v2               |       0.6760 |      0.6548 |     -0.0211 |       1.4379 |      1.3792 |        -0.0679 | 0.1322 |
| current_state_v2  | fib4_v2               |       0.6760 |      0.6548 |     -0.0211 |       1.4379 |      1.3792 |        -0.0679 | 0.1322 |
| current_state_v2  | miss_ggt              |       0.0000 |      0.0000 |      0.0000 |       0.7951 |      0.7625 |        -0.1956 | 0.1321 |
| v3_hepatic_schema | miss_ggt              |       0.0000 |      0.0000 |      0.0000 |       0.7951 |      0.7625 |        -0.1956 | 0.1321 |
| v3_hepatic_schema | miss_alt              |       0.0000 |      0.0000 |      0.0000 |       0.7920 |      0.7588 |        -0.1989 | 0.1288 |
| current_state_v2  | miss_alt              |       0.0000 |      0.0000 |      0.0000 |       0.7920 |      0.7588 |        -0.1989 | 0.1288 |
| current_state_v2  | miss_visit_1          |       0.0000 |      0.0000 |      0.0000 |       0.3773 |      0.3001 |        -0.2757 | 0.1287 |

## Top 25 features by missingness diff (test - train)

| feature_set       | feature                 |   miss_train |   miss_test |   miss_diff |     ks |
|:------------------|:------------------------|-------------:|------------:|------------:|-------:|
| v3_hepatic_schema | Hypertension            |       0.2139 |      0.0236 |     -0.1902 | 0.0158 |
| biomarker_only    | Hypertension            |       0.2139 |      0.0236 |     -0.1902 | 0.0158 |
| NIT_plus_scores   | Hypertension            |       0.2139 |      0.0236 |     -0.1902 | 0.0158 |
| current_state_v2  | Hypertension            |       0.2139 |      0.0236 |     -0.1902 | 0.0158 |
| biomarker_only    | Dyslipidaemia           |       0.2131 |      0.0236 |     -0.1894 | 0.0282 |
| v3_hepatic_schema | Dyslipidaemia           |       0.2131 |      0.0236 |     -0.1894 | 0.0282 |
| NIT_plus_scores   | Dyslipidaemia           |       0.2131 |      0.0236 |     -0.1894 | 0.0282 |
| current_state_v2  | Dyslipidaemia           |       0.2131 |      0.0236 |     -0.1894 | 0.0282 |
| v3_hepatic_schema | T2DM                    |       0.2123 |      0.0236 |     -0.1886 | 0.0377 |
| current_state_v2  | T2DM                    |       0.2123 |      0.0236 |     -0.1886 | 0.0377 |
| NIT_plus_scores   | T2DM                    |       0.2123 |      0.0236 |     -0.1886 | 0.0377 |
| biomarker_only    | T2DM                    |       0.2123 |      0.0236 |     -0.1886 | 0.0377 |
| NIT_plus_scores   | bariatric_surgery       |       0.2203 |      0.0331 |     -0.1872 | 0.0089 |
| biomarker_only    | bariatric_surgery       |       0.2203 |      0.0331 |     -0.1872 | 0.0089 |
| current_state_v2  | bariatric_surgery       |       0.2203 |      0.0331 |     -0.1872 | 0.0089 |
| v3_hepatic_schema | bariatric_surgery       |       0.2203 |      0.0331 |     -0.1872 | 0.0089 |
| v3_hepatic_schema | x_fibrotest_fib4_latest |       0.5108 |      0.3617 |     -0.1491 | 0.0584 |
| current_state_v2  | x_fibrotest_fib4_latest |       0.5108 |      0.3617 |     -0.1491 | 0.0584 |
| current_state_v2  | chol_max                |       0.2945 |      0.1513 |     -0.1432 | 0.0522 |
| current_state_v2  | chol_delta              |       0.2945 |      0.1513 |     -0.1432 | 0.0602 |
| current_state_v2  | chol_last               |       0.2945 |      0.1513 |     -0.1432 | 0.0455 |
| current_state_v2  | chol_mean               |       0.2945 |      0.1513 |     -0.1432 | 0.0546 |
| current_state_v2  | chol_median             |       0.2945 |      0.1513 |     -0.1432 | 0.0447 |
| current_state_v2  | chol_std                |       0.2945 |      0.1513 |     -0.1432 | 0.0781 |
| current_state_v2  | chol_min                |       0.2945 |      0.1513 |     -0.1432 | 0.0766 |
