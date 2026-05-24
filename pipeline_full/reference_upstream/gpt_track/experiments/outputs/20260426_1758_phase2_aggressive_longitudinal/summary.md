# Experiment: phase2_aggressive_longitudinal

- feature_set: `aggressive_longitudinal` (leakage risk: moderate-high)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set             | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:------------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | aggressive_longitudinal | moderate-high  |                  15 |        0.8045 |       0.1042 |       0.5984 |       0.9564 |                     0.7524 |
| hepatic    | xgb_cox         | aggressive_longitudinal | moderate-high  |                  15 |        0.7596 |       0.1377 |       0.5020 |       0.9248 |                     0.6908 |
| hepatic    | lgbm_binary     | aggressive_longitudinal | moderate-high  |                  15 |        0.7606 |       0.1043 |       0.6071 |       0.9499 |                     0.7084 |
| hepatic    | catboost_binary | aggressive_longitudinal | moderate-high  |                  15 |        0.7411 |       0.1097 |       0.5710 |       0.8953 |                     0.6862 |
| hepatic    | xgb_binary      | aggressive_longitudinal | moderate-high  |                  15 |        0.7626 |       0.1071 |       0.5767 |       0.9462 |                     0.7090 |
| death      | rsf             | aggressive_longitudinal | moderate-high  |                  15 |        0.8848 |       0.0332 |       0.8362 |       0.9378 |                     0.8682 |
| death      | xgb_cox         | aggressive_longitudinal | moderate-high  |                  15 |        0.8793 |       0.0327 |       0.8223 |       0.9237 |                     0.8629 |
| death      | lgbm_binary     | aggressive_longitudinal | moderate-high  |                  15 |        0.7737 |       0.0561 |       0.6648 |       0.8747 |                     0.7457 |
| death      | catboost_binary | aggressive_longitudinal | moderate-high  |                  15 |        0.7461 |       0.0642 |       0.6103 |       0.8597 |                     0.7140 |
| death      | xgb_binary      | aggressive_longitudinal | moderate-high  |                  15 |        0.7539 |       0.0688 |       0.6277 |       0.8897 |                     0.7195 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7894271418884291,
  "cindex_hepatic_std": 0.11277758712971468,
  "cindex_hepatic_min": 0.595219123505976,
  "cindex_hepatic_max": 0.9498546511627907,
  "cindex_hepatic_mean_minus_halfsd": 0.7330383483235717,
  "cindex_death_mean": 0.8571782312476083,
  "cindex_death_std": 0.04596522304496421,
  "cindex_death_min": 0.7779369627507163,
  "cindex_death_max": 0.9259259259259259,
  "cindex_death_mean_minus_halfsd": 0.8341956197251261,
  "score_mean": 0.8097524686961827,
  "score_std": 0.07905635099156043,
  "score_min": 0.6650576417733322,
  "score_max": 0.9380268847699729,
  "score_mean_minus_halfsd": 0.7702242932004024,
  "n_folds": 15
}
```
