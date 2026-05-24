# Experiment: 004_all_visits_longitudinal

- feature_set: `all_visits_longitudinal` (leakage risk: high)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set             | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |
|:-----------|:----------------|:------------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|
| hepatic    | rsf             | all_visits_longitudinal | high           |                  15 |        0.8041 |       0.1128 |       0.5743 |
| hepatic    | xgb_cox         | all_visits_longitudinal | high           |                  15 |        0.7462 |       0.1433 |       0.4422 |
| hepatic    | lgbm_binary     | all_visits_longitudinal | high           |                  15 |        0.7192 |       0.1112 |       0.5238 |
| hepatic    | catboost_binary | all_visits_longitudinal | high           |                  15 |        0.7227 |       0.1053 |       0.5664 |
| death      | rsf             | all_visits_longitudinal | high           |                  15 |        0.9494 |       0.0127 |       0.9260 |
| death      | xgb_cox         | all_visits_longitudinal | high           |                  15 |        0.9285 |       0.0365 |       0.8424 |
| death      | lgbm_binary     | all_visits_longitudinal | high           |                  15 |        0.7871 |       0.0524 |       0.7120 |
| death      | catboost_binary | all_visits_longitudinal | high           |                  15 |        0.7554 |       0.0661 |       0.6203 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7859479559645383,
  "cindex_hepatic_std": 0.1267157480109165,
  "cindex_hepatic_min": 0.5739514348785872,
  "cindex_death_mean": 0.9146521227529102,
  "cindex_death_std": 0.029907096458909942,
  "cindex_death_min": 0.8595988538681948,
  "score_mean": 0.8245592060010497,
  "score_std": 0.08973449712618847,
  "score_min": 0.6840997012525247,
  "n_folds": 15
}
```
