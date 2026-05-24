# Phase 2 Experiment 2 — Optuna tuning + verification

Date: 2026-04-27. Wall-clock: 9.5 min.

Selection metric (pre-registered): mean − 1×std hepatic C-index.

Inner CV (Optuna): 5-fold × 5-repeat (25 folds). Final CV: 5-fold × 10-repeat (50 folds, base_seed=42).

Bootstrap: 1000 resamples on per-fold C-indices, 95% percentile CI.


## Best params (Optuna 50–100 trials, TPE)

| id | model | feature_set | trials | best mean | best std | m−s |
|---|---|---|---|---|---|---|
| `rsf_baseline_v1` | rsf | baseline_v1 | 50 | 0.8104 | 0.0747 | 0.7357 |
| `xgb_cox_baseline_v1` | xgb_cox | baseline_v1 | 50 | 0.7834 | 0.0887 | 0.6947 |
| `rsf_nit_only_baseline_only` | rsf | nit_only_baseline_only | 50 | 0.7850 | 0.0740 | 0.7110 |
| `rsf_early_v1_v3` | rsf | early_v1_v3 | 50 | 0.7744 | 0.0695 | 0.7049 |
| `catboost_bin_early_v1_v3` | catboost_bin | early_v1_v3 | 50 | 0.7474 | 0.0742 | 0.6732 |
| `coxnet_baseline_v1` | coxnet | baseline_v1 | 50 | 0.7256 | 0.1097 | 0.6159 |

### Best param values

- **`rsf_baseline_v1`**: n_estimators=279, min_samples_leaf=12, min_samples_split=71, max_features=log2
- **`rsf_nit_only_baseline_only`**: n_estimators=439, min_samples_leaf=23, min_samples_split=85, max_features=log2
- **`rsf_early_v1_v3`**: n_estimators=283, min_samples_leaf=19, min_samples_split=14, max_features=log2
- **`xgb_cox_baseline_v1`**: n_estimators=320, learning_rate=0.011627388738686578, max_depth=8, min_child_weight=1.8809463655465164, subsample=0.7457174290570576, colsample_bytree=0.6941935551421055, reg_lambda=0.3040863647001995
- **`catboost_bin_early_v1_v3`**: iterations=549, learning_rate=0.02702898783568698, depth=8, l2_leaf_reg=2.634770870600858, horizon=5
- **`coxnet_baseline_v1`**: l1_ratio=0.8530090194461845, alpha_min_ratio=0.0317701661709683

## Final 5×10 CV (after tuning) — appended to phase2_cv_results.csv

Compared with the un-tuned Phase 2 grid number, where it exists.

| id | tuned mean | tuned std | tuned m−s | untuned mean | untuned std | Δmean | Δm−s |
|---|---|---|---|---|---|---|---|
| `rsf_baseline_v1` | 0.8057 | 0.0861 | 0.7196 | 0.8011 | 0.0868 | +0.0046 | +0.0053 |
| `xgb_cox_baseline_v1` | 0.7907 | 0.0839 | 0.7068 | 0.7517 | 0.0949 | +0.0390 | +0.0500 |
| `rsf_nit_only_baseline_only` | 0.7752 | 0.0810 | 0.6942 | 0.7719 | 0.0841 | +0.0033 | +0.0064 |
| `rsf_early_v1_v3` | 0.7718 | 0.0931 | 0.6787 | 0.7595 | 0.0980 | +0.0124 | +0.0173 |
| `catboost_bin_early_v1_v3` | 0.7498 | 0.0846 | 0.6652 | 0.7283 | 0.0881 | +0.0216 | +0.0251 |
| `coxnet_baseline_v1` | 0.7269 | 0.1013 | 0.6256 | 0.7029 | 0.1038 | +0.0241 | +0.0265 |

## Bootstrap CIs (1000 resamples on the 50 per-fold C-indices)

| id | mean | std | 95% CI | width |
|---|---|---|---|---|
| `rsf_baseline_v1` | 0.8057 | 0.0861 | [0.7827, 0.8291] | 0.0463 |
| `xgb_cox_baseline_v1` | 0.7907 | 0.0839 | [0.7679, 0.8146] | 0.0467 |
| `rsf_nit_only_baseline_only` | 0.7752 | 0.0810 | [0.7523, 0.7978] | 0.0455 |
| `rsf_early_v1_v3` | 0.7718 | 0.0931 | [0.7464, 0.7970] | 0.0506 |
| `catboost_bin_early_v1_v3` | 0.7498 | 0.0846 | [0.7264, 0.7726] | 0.0461 |
| `coxnet_baseline_v1` | 0.7269 | 0.1013 | [0.7023, 0.7538] | 0.0515 |

File: `reports/phase2_bootstrap_cis.csv`.


## Multi-seed ensembling (top 2 RSFs, seeds 42–51)

Per fold: 10 RSF fits (one per seed), risks rank-averaged. Compared to single-seed (42) on the same fold split.

| id | single mean | single std | multi mean | multi std | Δmean | Δstd |
|---|---|---|---|---|---|---|
| `rsf_baseline_v1` | 0.8057 | 0.0861 | 0.8075 | 0.0855 | +0.0018 | -0.0007 |
| `rsf_nit_only_baseline_only` | 0.7752 | 0.0810 | 0.7725 | 0.0809 | -0.0027 | -0.0000 |

File: `reports/phase2_multiseed.csv`.


## Recommendation for ensemble selection

Top 3 by tuned mean (from bootstrap table):
- **rsf_baseline_v1** — mean 0.8057, 95% CI [0.7827, 0.8291]
- **xgb_cox_baseline_v1** — mean 0.7907, 95% CI [0.7679, 0.8146]
- **rsf_nit_only_baseline_only** — mean 0.7752, 95% CI [0.7523, 0.7978]

For a rank-averaged hepatic ensemble, prefer the model classes that differ most in tuned configuration, ideally one tree-based (RSF or XGB-Cox) and one linear (Coxnet) over different feature scopes (`baseline_v1` vs `nit_only_baseline_only`). Multi-seed lifts (above) indicate whether RSF benefits from seed averaging — if Δmean > 0 and Δstd < 0 for top-RSF, include the multi-seed RSF in the ensemble.


## Stop point

Experiment 2 complete. No tuned model exceeded the 0.85 suspicious threshold during tuning. No new submissions generated. Awaiting instruction on Experiment 3+ or submission generation.
