# Experiment: 006_strict_time_aligned

- feature_set: `strict_time_aligned` (leakage risk: low/moderate)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set         | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |
|:-----------|:----------------|:--------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|
| hepatic    | rsf             | strict_time_aligned | low/moderate   |                  15 |        0.9541 |       0.0256 |       0.8937 |
| hepatic    | xgb_cox         | strict_time_aligned | low/moderate   |                  15 |        0.9743 |       0.0129 |       0.9481 |
| hepatic    | lgbm_binary     | strict_time_aligned | low/moderate   |                  15 |        0.8973 |       0.0575 |       0.7769 |
| hepatic    | catboost_binary | strict_time_aligned | low/moderate   |                  15 |        0.8915 |       0.0619 |       0.7769 |
| death      | rsf             | strict_time_aligned | low/moderate   |                  15 |        0.9518 |       0.0132 |       0.9335 |
| death      | xgb_cox         | strict_time_aligned | low/moderate   |                  15 |        0.9253 |       0.0353 |       0.8481 |
| death      | lgbm_binary     | strict_time_aligned | low/moderate   |                  15 |        0.7594 |       0.0616 |       0.6160 |
| death      | catboost_binary | strict_time_aligned | low/moderate   |                  15 |        0.7564 |       0.0629 |       0.6249 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.9563319542284482,
  "cindex_hepatic_std": 0.017179982774557543,
  "cindex_hepatic_min": 0.9285714285714286,
  "cindex_death_mean": 0.8994109165588043,
  "cindex_death_std": 0.03292759738095945,
  "cindex_death_min": 0.8409961685823755,
  "score_mean": 0.9392556429275553,
  "score_std": 0.014014159814733413,
  "score_min": 0.9217041170260896,
  "n_folds": 15
}
```
