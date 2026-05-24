# Experiment: phase3_current_state_v2

- feature_set: `current_state_v2` (leakage risk: moderate-high)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set      | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:-----------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | current_state_v2 | moderate-high  |                  15 |        0.8185 |       0.1135 |       0.5969 |       0.9553 |                     0.7618 |
| hepatic    | xgb_cox         | current_state_v2 | moderate-high  |                  15 |        0.7580 |       0.1430 |       0.4702 |       0.9411 |                     0.6865 |
| hepatic    | xgb_aft         | current_state_v2 | moderate-high  |                  15 |        0.7825 |       0.1151 |       0.5578 |       0.9268 |                     0.7250 |
| hepatic    | lgbm_binary     | current_state_v2 | moderate-high  |                  15 |        0.7482 |       0.1082 |       0.5721 |       0.9281 |                     0.6941 |
| hepatic    | catboost_binary | current_state_v2 | moderate-high  |                  15 |        0.7409 |       0.1139 |       0.5809 |       0.9186 |                     0.6839 |
| hepatic    | xgb_binary      | current_state_v2 | moderate-high  |                  15 |        0.7581 |       0.1071 |       0.5783 |       0.9288 |                     0.7046 |
| hepatic    | rsf             | current_state_v2 | moderate-high  |                  15 |        0.8249 |       0.1172 |       0.5960 |       0.9553 |                     0.7663 |
| hepatic    | xgb_cox         | current_state_v2 | moderate-high  |                  15 |        0.7537 |       0.1461 |       0.4669 |       0.9502 |                     0.6807 |
| hepatic    | xgb_aft         | current_state_v2 | moderate-high  |                  15 |        0.7825 |       0.1151 |       0.5578 |       0.9268 |                     0.7250 |
| hepatic    | lgbm_binary     | current_state_v2 | moderate-high  |                  15 |        0.7545 |       0.1056 |       0.5578 |       0.9273 |                     0.7016 |
| hepatic    | catboost_binary | current_state_v2 | moderate-high  |                  15 |        0.7348 |       0.1136 |       0.5562 |       0.9150 |                     0.6780 |
| hepatic    | xgb_binary      | current_state_v2 | moderate-high  |                  15 |        0.7604 |       0.1113 |       0.5705 |       0.9375 |                     0.7047 |
| death      | rsf             | current_state_v2 | moderate-high  |                  15 |        0.9302 |       0.0195 |       0.8946 |       0.9723 |                     0.9204 |
| death      | xgb_cox         | current_state_v2 | moderate-high  |                  15 |        0.9358 |       0.0250 |       0.8725 |       0.9684 |                     0.9233 |
| death      | xgb_aft         | current_state_v2 | moderate-high  |                  15 |        0.9192 |       0.0270 |       0.8582 |       0.9603 |                     0.9057 |
| death      | lgbm_binary     | current_state_v2 | moderate-high  |                  15 |        0.7715 |       0.0555 |       0.6948 |       0.8912 |                     0.7437 |
| death      | catboost_binary | current_state_v2 | moderate-high  |                  15 |        0.7524 |       0.0488 |       0.6705 |       0.8530 |                     0.7281 |
| death      | xgb_binary      | current_state_v2 | moderate-high  |                  15 |        0.7713 |       0.0513 |       0.6734 |       0.8695 |                     0.7456 |
| death      | rsf             | current_state_v2 | moderate-high  |                  15 |        0.9318 |       0.0215 |       0.8851 |       0.9749 |                     0.9210 |
| death      | xgb_cox         | current_state_v2 | moderate-high  |                  15 |        0.9399 |       0.0241 |       0.8797 |       0.9771 |                     0.9278 |
| death      | xgb_aft         | current_state_v2 | moderate-high  |                  15 |        0.9188 |       0.0268 |       0.8582 |       0.9603 |                     0.9054 |
| death      | lgbm_binary     | current_state_v2 | moderate-high  |                  15 |        0.7717 |       0.0569 |       0.6819 |       0.8927 |                     0.7432 |
| death      | catboost_binary | current_state_v2 | moderate-high  |                  15 |        0.7511 |       0.0592 |       0.6418 |       0.8545 |                     0.7216 |
| death      | xgb_binary      | current_state_v2 | moderate-high  |                  15 |        0.7623 |       0.0553 |       0.6418 |       0.8755 |                     0.7346 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.8046074824859357,
  "cindex_hepatic_std": 0.12378207019825901,
  "cindex_hepatic_min": 0.5848605577689243,
  "cindex_hepatic_max": 0.9634146341463414,
  "cindex_hepatic_mean_minus_halfsd": 0.7427164473868062,
  "cindex_death_mean": 0.903255673356344,
  "cindex_death_std": 0.034075903924082294,
  "cindex_death_min": 0.8391494002181025,
  "cindex_death_max": 0.9534883720930233,
  "cindex_death_mean_minus_halfsd": 0.8862177213943029,
  "score_mean": 0.8342019397470583,
  "score_std": 0.0867601141578824,
  "score_min": 0.6785513266084597,
  "score_max": 0.9604367555303459,
  "score_mean_minus_halfsd": 0.790821882668117,
  "n_folds": 15
}
```
