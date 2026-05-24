# Experiment: 002_baseline_v1_censor_missing_death

- feature_set: `baseline_v1` (leakage risk: low)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 5 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set   | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |
|:-----------|:----------------|:--------------|:---------------|--------------------:|--------------:|-------------:|-------------:|
| hepatic    | coxnet          | baseline_v1   | low            |                  25 |        0.7439 |       0.1029 |       0.4916 |
| hepatic    | rsf             | baseline_v1   | low            |                  25 |        0.7904 |       0.1071 |       0.5085 |
| hepatic    | xgb_cox         | baseline_v1   | low            |                  25 |        0.7374 |       0.1046 |       0.4948 |
| hepatic    | lgbm_binary     | baseline_v1   | low            |                  25 |        0.7265 |       0.0887 |       0.4961 |
| hepatic    | catboost_binary | baseline_v1   | low            |                  25 |        0.6898 |       0.0929 |       0.4233 |
| death      | coxnet          | baseline_v1   | low            |                  25 |        0.7912 |       0.0564 |       0.6880 |
| death      | rsf             | baseline_v1   | low            |                  25 |        0.8070 |       0.0578 |       0.6952 |
| death      | xgb_cox         | baseline_v1   | low            |                  25 |        0.7665 |       0.0533 |       0.6812 |
| death      | lgbm_binary     | baseline_v1   | low            |                  25 |        0.6840 |       0.0859 |       0.5225 |
| death      | catboost_binary | baseline_v1   | low            |                  25 |        0.6416 |       0.0987 |       0.4260 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7771513318677299,
  "cindex_hepatic_std": 0.09828040707966126,
  "cindex_hepatic_min": 0.4821705426356589,
  "cindex_death_mean": 0.7694590164672427,
  "cindex_death_std": 0.06825711593688351,
  "cindex_death_min": 0.6790780141843972,
  "score_mean": 0.7748436372475838,
  "score_std": 0.07104756372200095,
  "score_min": 0.5609320782576597,
  "n_folds": 25
}
```
