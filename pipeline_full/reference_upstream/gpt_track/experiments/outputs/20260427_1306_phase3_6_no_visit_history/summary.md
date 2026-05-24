# Experiment: phase3_6_no_visit_history

- feature_set: `current_state_v2_no_visit_history` (leakage risk: moderate-high)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model   | feature_set                       | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:--------|:----------------------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf     | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8191 |       0.1095 |       0.5953 |       0.9522 |                     0.7643 |
| hepatic    | xgb_aft | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.7845 |       0.1086 |       0.5850 |       0.9238 |                     0.7302 |
| hepatic    | rsf     | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8254 |       0.1097 |       0.6062 |       0.9543 |                     0.7706 |
| hepatic    | xgb_aft | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.7845 |       0.1086 |       0.5850 |       0.9238 |                     0.7302 |
| hepatic    | rsf     | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8198 |       0.1142 |       0.5904 |       0.9533 |                     0.7627 |
| hepatic    | xgb_aft | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.7845 |       0.1086 |       0.5850 |       0.9238 |                     0.7302 |
| hepatic    | rsf     | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8221 |       0.1127 |       0.6016 |       0.9522 |                     0.7658 |
| hepatic    | xgb_aft | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.7845 |       0.1086 |       0.5850 |       0.9238 |                     0.7302 |
| hepatic    | rsf     | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8205 |       0.1121 |       0.5969 |       0.9573 |                     0.7645 |
| hepatic    | xgb_aft | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.7845 |       0.1086 |       0.5850 |       0.9238 |                     0.7302 |
| death      | rsf     | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8841 |       0.0303 |       0.8282 |       0.9400 |                     0.8690 |
| death      | xgb_aft | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8717 |       0.0345 |       0.8124 |       0.9299 |                     0.8544 |
| death      | rsf     | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8811 |       0.0352 |       0.8156 |       0.9520 |                     0.8635 |
| death      | xgb_aft | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8717 |       0.0345 |       0.8124 |       0.9299 |                     0.8544 |
| death      | rsf     | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8810 |       0.0314 |       0.8335 |       0.9433 |                     0.8653 |
| death      | xgb_aft | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8717 |       0.0345 |       0.8124 |       0.9299 |                     0.8544 |
| death      | rsf     | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8798 |       0.0318 |       0.8230 |       0.9466 |                     0.8639 |
| death      | xgb_aft | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8717 |       0.0345 |       0.8124 |       0.9299 |                     0.8544 |
| death      | rsf     | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8807 |       0.0325 |       0.8261 |       0.9433 |                     0.8645 |
| death      | xgb_aft | current_state_v2_no_visit_history | moderate-high  |                  15 |        0.8722 |       0.0336 |       0.8203 |       0.9299 |                     0.8554 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.8260179132135336,
  "cindex_hepatic_std": 0.11109332475350235,
  "cindex_hepatic_min": 0.6015503875968993,
  "cindex_hepatic_max": 0.9651162790697675,
  "cindex_hepatic_mean_minus_halfsd": 0.7704712508367824,
  "cindex_death_mean": 0.8996498663929887,
  "cindex_death_std": 0.021014952999597408,
  "cindex_death_min": 0.8648648648648649,
  "cindex_death_max": 0.9422028353326063,
  "cindex_death_mean_minus_halfsd": 0.88914238989319,
  "score_mean": 0.8481074991673699,
  "score_std": 0.07539136778178405,
  "score_min": 0.695688445921004,
  "score_max": 0.9412500991101433,
  "score_mean_minus_halfsd": 0.8104118152764779,
  "n_folds": 15
}
```
