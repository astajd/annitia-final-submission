# Experiment: 001_baseline_v1_drop_missing_death

- feature_set: `baseline_v1` (leakage risk: low)
- death_target_mode: `drop_missing_death`
- folds: 5 x 5 repeats
- hepatic events: 47 / 1253
- death events:   76 / 984

## Per-model CV summary

| endpoint   | model           | feature_set   | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |
|:-----------|:----------------|:--------------|:---------------|--------------------:|--------------:|-------------:|-------------:|
| hepatic    | coxnet          | baseline_v1   | low            |                  25 |        0.7439 |       0.1029 |       0.4916 |
| hepatic    | rsf             | baseline_v1   | low            |                  25 |        0.7904 |       0.1071 |       0.5085 |
| hepatic    | xgb_cox         | baseline_v1   | low            |                  25 |        0.7374 |       0.1046 |       0.4948 |
| hepatic    | lgbm_binary     | baseline_v1   | low            |                  25 |        0.7265 |       0.0887 |       0.4961 |
| hepatic    | catboost_binary | baseline_v1   | low            |                  25 |        0.6898 |       0.0929 |       0.4233 |
| death      | coxnet          | baseline_v1   | low            |                  25 |        0.7896 |       0.0569 |       0.6872 |
| death      | rsf             | baseline_v1   | low            |                  25 |        0.8077 |       0.0604 |       0.6938 |
| death      | xgb_cox         | baseline_v1   | low            |                  25 |        0.7621 |       0.0553 |       0.6423 |
| death      | lgbm_binary     | baseline_v1   | low            |                  25 |        0.7022 |       0.0828 |       0.5208 |
| death      | catboost_binary | baseline_v1   | low            |                  25 |        0.6549 |       0.1011 |       0.4434 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7771513318677299,
  "cindex_hepatic_std": 0.09828040707966126,
  "cindex_hepatic_min": 0.4821705426356589,
  "cindex_death_mean": 0.7727044981754538,
  "cindex_death_std": 0.06435782833526754,
  "cindex_death_min": 0.6772046589018302,
  "score_mean": 0.775817281760047,
  "score_std": 0.07074742868549917,
  "score_min": 0.5645035068290882,
  "n_folds": 25
}
```
