# Experiment: phase2_NIT_plus_scores_longitudinal

- feature_set: `NIT_plus_scores_longitudinal` (leakage risk: moderate)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set                  | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:-----------------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | NIT_plus_scores_longitudinal | moderate       |                  15 |        0.8030 |       0.1192 |       0.6024 |       0.9651 |                     0.7434 |
| hepatic    | xgb_cox         | NIT_plus_scores_longitudinal | moderate       |                  15 |        0.7711 |       0.1312 |       0.5171 |       0.9563 |                     0.7055 |
| hepatic    | lgbm_binary     | NIT_plus_scores_longitudinal | moderate       |                  15 |        0.7590 |       0.1205 |       0.5829 |       0.9491 |                     0.6987 |
| hepatic    | catboost_binary | NIT_plus_scores_longitudinal | moderate       |                  15 |        0.6954 |       0.1379 |       0.5090 |       0.9622 |                     0.6264 |
| hepatic    | xgb_binary      | NIT_plus_scores_longitudinal | moderate       |                  15 |        0.7499 |       0.1159 |       0.5674 |       0.9197 |                     0.6919 |
| death      | rsf             | NIT_plus_scores_longitudinal | moderate       |                  15 |        0.8552 |       0.0339 |       0.8029 |       0.9235 |                     0.8383 |
| death      | xgb_cox         | NIT_plus_scores_longitudinal | moderate       |                  15 |        0.8703 |       0.0363 |       0.8212 |       0.9418 |                     0.8522 |
| death      | lgbm_binary     | NIT_plus_scores_longitudinal | moderate       |                  15 |        0.7512 |       0.0624 |       0.6554 |       0.8849 |                     0.7200 |
| death      | catboost_binary | NIT_plus_scores_longitudinal | moderate       |                  15 |        0.7034 |       0.0699 |       0.6122 |       0.8477 |                     0.6684 |
| death      | xgb_binary      | NIT_plus_scores_longitudinal | moderate       |                  15 |        0.7201 |       0.0675 |       0.6117 |       0.8624 |                     0.6864 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7753014443821581,
  "cindex_hepatic_std": 0.12696582273543772,
  "cindex_hepatic_min": 0.5557768924302788,
  "cindex_hepatic_max": 0.965843023255814,
  "cindex_hepatic_mean_minus_halfsd": 0.7118185330144392,
  "cindex_death_mean": 0.8202764004027696,
  "cindex_death_std": 0.05023678681991148,
  "cindex_death_min": 0.7559322033898305,
  "cindex_death_max": 0.918979744936234,
  "cindex_death_mean_minus_halfsd": 0.7951580069928139,
  "score_mean": 0.7887939311883413,
  "score_std": 0.08900689927719287,
  "score_min": 0.6198948885309823,
  "score_max": 0.9415475820174555,
  "score_mean_minus_halfsd": 0.7442904815497449,
  "n_folds": 15
}
```
