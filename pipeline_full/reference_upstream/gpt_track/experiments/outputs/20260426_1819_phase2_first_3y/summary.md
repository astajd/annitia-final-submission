# Experiment: phase2_first_3y

- feature_set: `first_3y` (leakage risk: low)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set   | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:--------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | first_3y      | low            |                  15 |        0.7462 |       0.1107 |       0.5163 |       0.9346 |                     0.6909 |
| hepatic    | xgb_cox         | first_3y      | low            |                  15 |        0.6583 |       0.1181 |       0.4163 |       0.8028 |                     0.5992 |
| hepatic    | lgbm_binary     | first_3y      | low            |                  15 |        0.7101 |       0.1081 |       0.4760 |       0.9266 |                     0.6560 |
| hepatic    | catboost_binary | first_3y      | low            |                  15 |        0.7040 |       0.1247 |       0.4434 |       0.8779 |                     0.6417 |
| hepatic    | xgb_binary      | first_3y      | low            |                  15 |        0.7109 |       0.1174 |       0.4326 |       0.9237 |                     0.6522 |
| death      | rsf             | first_3y      | low            |                  15 |        0.8221 |       0.0380 |       0.7695 |       0.8950 |                     0.8031 |
| death      | xgb_cox         | first_3y      | low            |                  15 |        0.8078 |       0.0442 |       0.7280 |       0.8855 |                     0.7857 |
| death      | lgbm_binary     | first_3y      | low            |                  15 |        0.7208 |       0.0876 |       0.6003 |       0.9079 |                     0.6770 |
| death      | catboost_binary | first_3y      | low            |                  15 |        0.7159 |       0.1051 |       0.4392 |       0.8505 |                     0.6633 |
| death      | xgb_binary      | first_3y      | low            |                  15 |        0.7225 |       0.0880 |       0.5661 |       0.8867 |                     0.6785 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7476840314970771,
  "cindex_hepatic_std": 0.1046997665866591,
  "cindex_hepatic_min": 0.5519379844961241,
  "cindex_hepatic_max": 0.9578488372093024,
  "cindex_hepatic_mean_minus_halfsd": 0.6953341482037475,
  "cindex_death_mean": 0.8024101902168077,
  "cindex_death_std": 0.06107334747777713,
  "cindex_death_min": 0.7096045197740113,
  "cindex_death_max": 0.9032258064516129,
  "cindex_death_mean_minus_halfsd": 0.7718735164779191,
  "score_mean": 0.7641018791129962,
  "score_std": 0.07558923182045722,
  "score_min": 0.6081819859726836,
  "score_max": 0.8884592098938884,
  "score_mean_minus_halfsd": 0.7263072632027676,
  "n_folds": 15
}
```
