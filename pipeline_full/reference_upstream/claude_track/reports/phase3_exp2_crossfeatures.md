# Phase 3 Experiment 2 — OOF cross-feature stacking

Date: 2026-04-28. CV: 5-fold × 10-repeat stratified on the endpoint's event indicator (50 folds, base_seed=42).

Hepatic cohort: n=1253 (events 47), `drop_missing_death`. Death cohort: n=1253 (events 76), `censor_missing_death_at_last`. Filter A in hep cohort: n=1238.

## Leakage check

5 random hepatic rows + 5 random death rows; for each, lists the (repeat, fold) where the row was in val. Each row appears in val exactly N_REPEATS=10 times. Train-isolation verified per (rep, fold).

**Verdict:** PASS.

Sampled rows:
| endpoint | patient_id | row | n_val | first 3 (rep,fold) | train_isolation |
|---|---|---|---|---|---|
| hepatic | JPX16F1KYNTL | 796 | 10 | [(0, 0), (1, 3), (2, 1)] | OK |
| hepatic | 4NTHZM32HZS3 | 639 | 10 | [(0, 0), (1, 0), (2, 3)] | OK |
| hepatic | W4C6LBPJ8T22 | 337 | 10 | [(0, 4), (1, 2), (2, 2)] | OK |
| hepatic | X8L008UHBXLB | 385 | 10 | [(0, 2), (1, 4), (2, 4)] | OK |
| hepatic | 7SU62J8LBDSG | 1062 | 10 | [(0, 3), (1, 3), (2, 0)] | OK |
| death | DKN2K7IT7XXD | 1015 | 10 | [(0, 0), (1, 4), (2, 2)] | OK |
| death | RG29D2FF5L6B | 630 | 10 | [(0, 3), (1, 1), (2, 0)] | OK |
| death | 3HZCD3GEQ10F | 811 | 10 | [(0, 4), (1, 1), (2, 3)] | OK |
| death | 826BHYZ1OV5V | 1141 | 10 | [(0, 3), (1, 2), (2, 0)] | OK |
| death | HLI5L692TULG | 760 | 10 | [(0, 0), (1, 2), (2, 3)] | OK |

## Round 1 — CV bake-off (base vs aug)

| model | endpoint | base m-s | aug m-s | Δ |
|---|---|---|---|---|
| landmark_3y_RSF (filter A) | hepatic | 0.6712 | 0.6806 | **+0.0094** |
| XGB-Cox/longitudinal_summary | hepatic | 0.6949 | 0.6976 | **+0.0027** |
| XGB-Cox/longitudinal_summary | death | 0.9313 | 0.9304 | **-0.0009** |

Best hepatic Δ m-s = **+0.0094** (threshold for action: ≥ 0.005).

OOF rank-corr (base vs aug): landmark = 0.617, xgb-hep = 0.993.

## Round 2 — using round-1 augmented OOFs as cross-features

| model | round-1 aug m-s | round-2 m-s | Δ vs round-1 |
|---|---|---|---|
| landmark | 0.6806 | 0.6781 | -0.0025 |
| xgb-hep | 0.6976 | 0.6955 | -0.0021 |
| death | 0.9304 | 0.9278 | -0.0025 |

## Submission

Built `submissions/phase3_blend_with_crossfeatures.csv`. 30/70 blend with the augmented hepatic component(s) plugged in. Test-time cross-features come from full-train models (no train-OOF leakage on test).