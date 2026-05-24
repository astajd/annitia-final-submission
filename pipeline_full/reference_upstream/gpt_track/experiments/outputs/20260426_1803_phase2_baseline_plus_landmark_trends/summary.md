# Experiment: phase2_baseline_plus_landmark_trends

- feature_set: `baseline_plus_landmark_trends` (leakage risk: low)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set                   | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:------------------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | baseline_plus_landmark_trends | low            |                  15 |        0.7360 |       0.1294 |       0.4682 |       0.9181 |                     0.6713 |
| hepatic    | xgb_cox         | baseline_plus_landmark_trends | low            |                  15 |        0.6240 |       0.1114 |       0.3809 |       0.7735 |                     0.5683 |
| hepatic    | lgbm_binary     | baseline_plus_landmark_trends | low            |                  15 |        0.6698 |       0.1122 |       0.4590 |       0.8329 |                     0.6137 |
| hepatic    | catboost_binary | baseline_plus_landmark_trends | low            |                  15 |        0.6316 |       0.1039 |       0.4062 |       0.7762 |                     0.5797 |
| hepatic    | xgb_binary      | baseline_plus_landmark_trends | low            |                  15 |        0.6609 |       0.0970 |       0.4598 |       0.8241 |                     0.6124 |
| death      | rsf             | baseline_plus_landmark_trends | low            |                  15 |        0.8136 |       0.0486 |       0.7280 |       0.8882 |                     0.7893 |
| death      | xgb_cox         | baseline_plus_landmark_trends | low            |                  15 |        0.7815 |       0.0571 |       0.6450 |       0.8830 |                     0.7530 |
| death      | lgbm_binary     | baseline_plus_landmark_trends | low            |                  15 |        0.6671 |       0.0852 |       0.5316 |       0.8369 |                     0.6245 |
| death      | catboost_binary | baseline_plus_landmark_trends | low            |                  15 |        0.6187 |       0.0729 |       0.5289 |       0.7689 |                     0.5823 |
| death      | xgb_binary      | baseline_plus_landmark_trends | low            |                  15 |        0.6414 |       0.0715 |       0.5418 |       0.8051 |                     0.6057 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7080590869632829,
  "cindex_hepatic_std": 0.1350566590259483,
  "cindex_hepatic_min": 0.4589641434262948,
  "cindex_hepatic_max": 0.9222383720930233,
  "cindex_hepatic_mean_minus_halfsd": 0.6405307574503087,
  "cindex_death_mean": 0.7660035196764862,
  "cindex_death_std": 0.06283685373573486,
  "cindex_death_min": 0.6597774244833068,
  "cindex_death_max": 0.8687171792948237,
  "cindex_death_mean_minus_halfsd": 0.7345850928086187,
  "score_mean": 0.7254424167772439,
  "score_std": 0.0889887906770536,
  "score_min": 0.5510621344409595,
  "score_max": 0.8460826334632439,
  "score_mean_minus_halfsd": 0.680948021438717,
  "n_folds": 15
}
```
