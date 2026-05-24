# Experiment: phase3_6_csv2_extra_seeds

- feature_set: `current_state_v2` (leakage risk: moderate-high)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model   | feature_set      | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:--------|:-----------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf     | current_state_v2 | moderate-high  |                  15 |        0.8195 |       0.1067 |       0.5969 |       0.9553 |                     0.7662 |
| hepatic    | xgb_aft | current_state_v2 | moderate-high  |                  15 |        0.7825 |       0.1151 |       0.5578 |       0.9268 |                     0.7250 |
| hepatic    | xgb_cox | current_state_v2 | moderate-high  |                  15 |        0.7527 |       0.1432 |       0.4797 |       0.9360 |                     0.6811 |
| hepatic    | rsf     | current_state_v2 | moderate-high  |                  15 |        0.8195 |       0.1112 |       0.5938 |       0.9583 |                     0.7639 |
| hepatic    | xgb_aft | current_state_v2 | moderate-high  |                  15 |        0.7825 |       0.1151 |       0.5578 |       0.9268 |                     0.7250 |
| hepatic    | xgb_cox | current_state_v2 | moderate-high  |                  15 |        0.7504 |       0.1403 |       0.4805 |       0.9441 |                     0.6803 |
| hepatic    | rsf     | current_state_v2 | moderate-high  |                  15 |        0.8202 |       0.1136 |       0.5984 |       0.9482 |                     0.7634 |
| hepatic    | xgb_aft | current_state_v2 | moderate-high  |                  15 |        0.7825 |       0.1151 |       0.5578 |       0.9268 |                     0.7250 |
| hepatic    | xgb_cox | current_state_v2 | moderate-high  |                  15 |        0.7570 |       0.1479 |       0.4665 |       0.9431 |                     0.6831 |
| death      | rsf     | current_state_v2 | moderate-high  |                  15 |        0.9336 |       0.0216 |       0.8904 |       0.9738 |                     0.9228 |
| death      | xgb_aft | current_state_v2 | moderate-high  |                  15 |        0.9192 |       0.0270 |       0.8582 |       0.9603 |                     0.9057 |
| death      | xgb_cox | current_state_v2 | moderate-high  |                  15 |        0.9408 |       0.0261 |       0.8682 |       0.9782 |                     0.9278 |
| death      | rsf     | current_state_v2 | moderate-high  |                  15 |        0.9309 |       0.0207 |       0.8872 |       0.9709 |                     0.9205 |
| death      | xgb_aft | current_state_v2 | moderate-high  |                  15 |        0.9192 |       0.0270 |       0.8582 |       0.9603 |                     0.9057 |
| death      | xgb_cox | current_state_v2 | moderate-high  |                  15 |        0.9365 |       0.0276 |       0.8668 |       0.9760 |                     0.9227 |
| death      | rsf     | current_state_v2 | moderate-high  |                  15 |        0.9320 |       0.0241 |       0.8767 |       0.9727 |                     0.9200 |
| death      | xgb_aft | current_state_v2 | moderate-high  |                  15 |        0.9192 |       0.0270 |       0.8582 |       0.9603 |                     0.9057 |
| death      | xgb_cox | current_state_v2 | moderate-high  |                  15 |        0.9383 |       0.0270 |       0.8582 |       0.9727 |                     0.9248 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.8176329643228717,
  "cindex_hepatic_std": 0.11895255945791067,
  "cindex_hepatic_min": 0.5953488372093023,
  "cindex_hepatic_max": 0.9654471544715447,
  "cindex_hepatic_mean_minus_halfsd": 0.7581566845939164,
  "cindex_death_mean": 0.9491111882978227,
  "cindex_death_std": 0.01300333867741696,
  "cindex_death_min": 0.9236641221374046,
  "cindex_death_max": 0.9672846237731734,
  "cindex_death_mean_minus_halfsd": 0.9426095189591143,
  "score_mean": 0.8570764315153568,
  "score_std": 0.08181587635479927,
  "score_min": 0.7028552971576227,
  "score_max": 0.9632098573423844,
  "score_mean_minus_halfsd": 0.8161684933379572,
  "n_folds": 15
}
```
