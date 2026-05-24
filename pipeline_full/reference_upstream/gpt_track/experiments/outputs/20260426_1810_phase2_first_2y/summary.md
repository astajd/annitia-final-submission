# Experiment: phase2_first_2y

- feature_set: `first_2y` (leakage risk: low)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set   | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:--------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | first_2y      | low            |                  15 |        0.7364 |       0.1062 |       0.5147 |       0.9186 |                     0.6832 |
| hepatic    | xgb_cox         | first_2y      | low            |                  15 |        0.6622 |       0.1300 |       0.4233 |       0.8089 |                     0.5972 |
| hepatic    | lgbm_binary     | first_2y      | low            |                  15 |        0.6867 |       0.0920 |       0.5287 |       0.8561 |                     0.6407 |
| hepatic    | catboost_binary | first_2y      | low            |                  15 |        0.6738 |       0.1260 |       0.4031 |       0.8576 |                     0.6108 |
| hepatic    | xgb_binary      | first_2y      | low            |                  15 |        0.6561 |       0.1380 |       0.3634 |       0.8525 |                     0.5871 |
| death      | rsf             | first_2y      | low            |                  15 |        0.8306 |       0.0470 |       0.7394 |       0.9040 |                     0.8071 |
| death      | xgb_cox         | first_2y      | low            |                  15 |        0.7914 |       0.0491 |       0.7034 |       0.8811 |                     0.7669 |
| death      | lgbm_binary     | first_2y      | low            |                  15 |        0.7016 |       0.0501 |       0.6137 |       0.8157 |                     0.6766 |
| death      | catboost_binary | first_2y      | low            |                  15 |        0.7060 |       0.0713 |       0.5648 |       0.8110 |                     0.6704 |
| death      | xgb_binary      | first_2y      | low            |                  15 |        0.7120 |       0.0618 |       0.5940 |       0.8384 |                     0.6811 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7286883668013332,
  "cindex_hepatic_std": 0.11663413912183843,
  "cindex_hepatic_min": 0.5069767441860465,
  "cindex_hepatic_max": 0.9476744186046512,
  "cindex_hepatic_mean_minus_halfsd": 0.670371297240414,
  "cindex_death_mean": 0.7915148251798635,
  "cindex_death_std": 0.04840617301982022,
  "cindex_death_min": 0.7175572519083969,
  "cindex_death_max": 0.8660915228807202,
  "cindex_death_mean_minus_halfsd": 0.7673117386699534,
  "score_mean": 0.7475363043148924,
  "score_std": 0.08120607199908127,
  "score_min": 0.5982508264813088,
  "score_max": 0.884198802085259,
  "score_mean_minus_halfsd": 0.7069332683153517,
  "n_folds": 15
}
```
