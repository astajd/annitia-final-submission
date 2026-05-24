# Experiment: phase2_first_3_visits

- feature_set: `first_3_visits` (leakage risk: low)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set    | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:---------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | first_3_visits | low            |                  15 |        0.7584 |       0.1332 |       0.5085 |       0.9448 |                     0.6918 |
| hepatic    | xgb_cox         | first_3_visits | low            |                  15 |        0.7291 |       0.1203 |       0.4930 |       0.8791 |                     0.6689 |
| hepatic    | lgbm_binary     | first_3_visits | low            |                  15 |        0.7347 |       0.1002 |       0.5178 |       0.9055 |                     0.6846 |
| hepatic    | catboost_binary | first_3_visits | low            |                  15 |        0.7364 |       0.1089 |       0.4853 |       0.9116 |                     0.6819 |
| hepatic    | xgb_binary      | first_3_visits | low            |                  15 |        0.7322 |       0.1082 |       0.4713 |       0.9035 |                     0.6781 |
| death      | rsf             | first_3_visits | low            |                  15 |        0.7999 |       0.0633 |       0.7039 |       0.8876 |                     0.7683 |
| death      | xgb_cox         | first_3_visits | low            |                  15 |        0.7742 |       0.0580 |       0.6463 |       0.8414 |                     0.7452 |
| death      | lgbm_binary     | first_3_visits | low            |                  15 |        0.6706 |       0.0687 |       0.5561 |       0.7840 |                     0.6363 |
| death      | catboost_binary | first_3_visits | low            |                  15 |        0.6717 |       0.0672 |       0.5603 |       0.7854 |                     0.6381 |
| death      | xgb_binary      | first_3_visits | low            |                  15 |        0.6781 |       0.0751 |       0.5166 |       0.8066 |                     0.6406 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7649188484533614,
  "cindex_hepatic_std": 0.10178189452946844,
  "cindex_hepatic_min": 0.6103585657370518,
  "cindex_hepatic_max": 0.9585755813953488,
  "cindex_hepatic_mean_minus_halfsd": 0.7140279011886272,
  "cindex_death_mean": 0.7579747666820212,
  "cindex_death_std": 0.06575424025907307,
  "cindex_death_min": 0.6359300476947536,
  "cindex_death_max": 0.8425925925925926,
  "cindex_death_mean_minus_halfsd": 0.7250976465524847,
  "score_mean": 0.7628356239219592,
  "score_std": 0.06935101698135841,
  "score_min": 0.6315063151648724,
  "score_max": 0.9032751328075921,
  "score_mean_minus_halfsd": 0.72816011543128,
  "n_folds": 15
}
```
