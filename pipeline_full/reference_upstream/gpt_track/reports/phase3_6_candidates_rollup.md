# Phase 3.6 candidate roll-up

Reference: phase3_5_current_state_v3_hepatic_focused OOF hep=0.8415 / death=0.9496 / weighted=0.8739 | LB=**0.89147**.

## Candidates

| label                                           |   hep_oof |   hep_fold_std |   hep_fold_min |   death_oof |   death_fold_std |   death_fold_min |   weighted |   rho_oof_hep_vs_v3 |   rho_test_hep_vs_v3 |   rho_test_dea_vs_v3 |   n_components_hep |   n_components_dea | submission_csv                                                                                                   |
|:------------------------------------------------|----------:|---------------:|---------------:|------------:|-----------------:|-----------------:|-----------:|--------------------:|---------------------:|---------------------:|-------------------:|-------------------:|:-----------------------------------------------------------------------------------------------------------------|
| phase3_6_v4_hepatic_focused_bagged              |    0.8416 |         0.1063 |         0.6031 |      0.9496 |           0.0115 |           0.9269 |     0.8740 |              1.0000 |               0.9899 |               1.0000 |                  8 |                  4 | gpt/submissions/20260427_1431_phase3_6_v4_hepatic_focused_bagged.csv              |
| phase3_6_v4_hepatic_focused_biomarker_augmented |    0.8402 |         0.1019 |         0.6247 |      0.9496 |           0.0115 |           0.9269 |     0.8730 |              0.9760 |               0.9973 |               1.0000 |                  1 |                  4 | gpt/submissions/20260427_1431_phase3_6_v4_hepatic_focused_biomarker_augmented.csv |

## v3 hepatic pool diagnostics (LOO + forced methods)

| variant                      | method_picked    |   n |    oof |
|:-----------------------------|:-----------------|----:|-------:|
| bagged_hepatic_pool_baseline | seed_bagged      |   8 | 0.8416 |
| loo_hepatic__rsf__s2         | seed_bagged      |   8 | 0.8416 |
| loo_hepatic__rsf__s3         | seed_bagged      |   8 | 0.8416 |
| loo_hepatic__rsf__s3         | seed_bagged      |   8 | 0.8416 |
| seed_bagged_only             | seed_bagged_only |   8 | 0.8416 |
| loo_hepatic__rsf__s1         | seed_bagged      |   7 | 0.8415 |
| loo_hepatic__rsf__s2         | seed_bagged      |   8 | 0.8414 |
| loo_hepatic__rsf__s1         | seed_bagged      |   8 | 0.8414 |
| loo_hepatic__rsf__s4         | seed_bagged      |   7 | 0.8409 |
| loo_hepatic__rsf__s0         | seed_bagged      |   7 | 0.8409 |
| loo_hepatic__rsf__s0         | seed_bagged      |   8 | 0.8407 |
| loo_hepatic__xgb_aft__s0     | greedy           |   1 | 0.8402 |
| greedy_only                  | greedy_only      |   1 | 0.8402 |
| loo_hepatic__rsf__s4         | seed_bagged      |   8 | 0.8401 |
| equal_only                   | equal_only       |  11 | 0.8357 |