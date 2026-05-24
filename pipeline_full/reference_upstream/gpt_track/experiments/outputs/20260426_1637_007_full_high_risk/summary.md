# Experiment: 007_full_high_risk

- feature_set: `full_high_risk` (leakage risk: high)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set    | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |
|:-----------|:----------------|:---------------|:---------------|--------------------:|--------------:|-------------:|-------------:|
| hepatic    | rsf             | full_high_risk | high           |                  15 |        0.8220 |       0.1073 |       0.6181 |
| hepatic    | xgb_cox         | full_high_risk | high           |                  15 |        0.7545 |       0.1506 |       0.4547 |
| hepatic    | lgbm_binary     | full_high_risk | high           |                  15 |        0.7528 |       0.1082 |       0.5649 |
| hepatic    | catboost_binary | full_high_risk | high           |                  15 |        0.7313 |       0.1210 |       0.5475 |
| death      | rsf             | full_high_risk | high           |                  15 |        0.9522 |       0.0135 |       0.9335 |
| death      | xgb_cox         | full_high_risk | high           |                  15 |        0.9354 |       0.0364 |       0.8410 |
| death      | lgbm_binary     | full_high_risk | high           |                  15 |        0.7661 |       0.0639 |       0.6418 |
| death      | catboost_binary | full_high_risk | high           |                  15 |        0.7576 |       0.0588 |       0.6783 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.8099529250930414,
  "cindex_hepatic_std": 0.1228551204893478,
  "cindex_hepatic_min": 0.5848605577689243,
  "cindex_death_mean": 0.9104542344160328,
  "cindex_death_std": 0.03829344149056801,
  "cindex_death_min": 0.8402225755166932,
  "score_mean": 0.8401033178899389,
  "score_std": 0.08548077175760486,
  "score_min": 0.6881257946935662,
  "n_folds": 15
}
```
