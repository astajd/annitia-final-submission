# Experiment: phase3_6_hepatic_aug

- feature_set: `current_state_v2_hepatic_aug` (leakage risk: moderate-high)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model   | feature_set                  | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:--------|:-----------------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf     | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.8129 |       0.1178 |       0.5920 |       0.9522 |                     0.7540 |
| hepatic    | xgb_aft | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.7816 |       0.1129 |       0.5416 |       0.9248 |                     0.7251 |
| hepatic    | xgb_cox | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.7601 |       0.1430 |       0.4901 |       0.9482 |                     0.6886 |
| hepatic    | rsf     | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.8149 |       0.1106 |       0.6000 |       0.9512 |                     0.7596 |
| hepatic    | xgb_aft | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.7816 |       0.1129 |       0.5416 |       0.9248 |                     0.7251 |
| hepatic    | xgb_cox | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.7594 |       0.1428 |       0.4741 |       0.9482 |                     0.6880 |
| hepatic    | rsf     | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.8155 |       0.1147 |       0.6024 |       0.9553 |                     0.7581 |
| hepatic    | xgb_aft | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.7816 |       0.1129 |       0.5416 |       0.9248 |                     0.7251 |
| hepatic    | xgb_cox | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.7595 |       0.1448 |       0.4805 |       0.9543 |                     0.6871 |
| death      | rsf     | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.9294 |       0.0231 |       0.8841 |       0.9723 |                     0.9179 |
| death      | xgb_aft | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.9198 |       0.0295 |       0.8506 |       0.9603 |                     0.9050 |
| death      | xgb_cox | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.9379 |       0.0282 |       0.8625 |       0.9836 |                     0.9238 |
| death      | rsf     | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.9265 |       0.0233 |       0.8767 |       0.9723 |                     0.9148 |
| death      | xgb_aft | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.9198 |       0.0295 |       0.8506 |       0.9603 |                     0.9050 |
| death      | xgb_cox | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.9417 |       0.0222 |       0.8883 |       0.9782 |                     0.9306 |
| death      | rsf     | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.9297 |       0.0237 |       0.8830 |       0.9702 |                     0.9179 |
| death      | xgb_aft | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.9198 |       0.0295 |       0.8506 |       0.9603 |                     0.9050 |
| death      | xgb_cox | current_state_v2_hepatic_aug | moderate-high  |                  15 |        0.9379 |       0.0264 |       0.8825 |       0.9760 |                     0.9247 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.8165398831378174,
  "cindex_hepatic_std": 0.11815735041593035,
  "cindex_hepatic_min": 0.595219123505976,
  "cindex_hepatic_max": 0.9654471544715447,
  "cindex_hepatic_mean_minus_halfsd": 0.7574612079298523,
  "cindex_death_mean": 0.9466011748878549,
  "cindex_death_std": 0.011673302142242768,
  "cindex_death_min": 0.9247546346782988,
  "cindex_death_max": 0.9661941112322792,
  "cindex_death_mean_minus_halfsd": 0.9407645238167335,
  "score_mean": 0.8555582706628285,
  "score_std": 0.08179995271154386,
  "score_min": 0.7006959396456726,
  "score_max": 0.9636599698705164,
  "score_mean_minus_halfsd": 0.8146582943070566,
  "n_folds": 15
}
```
