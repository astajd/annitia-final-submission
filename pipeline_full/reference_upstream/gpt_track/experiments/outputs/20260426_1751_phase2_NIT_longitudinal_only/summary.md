# Experiment: phase2_NIT_longitudinal_only

- feature_set: `NIT_longitudinal_only` (leakage risk: moderate)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set           | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:----------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | NIT_longitudinal_only | moderate       |                  15 |        0.7874 |       0.1244 |       0.5829 |       0.9557 |                     0.7252 |
| hepatic    | xgb_cox         | NIT_longitudinal_only | moderate       |                  15 |        0.7350 |       0.1343 |       0.4773 |       0.9512 |                     0.6678 |
| hepatic    | lgbm_binary     | NIT_longitudinal_only | moderate       |                  15 |        0.7376 |       0.1145 |       0.5785 |       0.9455 |                     0.6804 |
| hepatic    | catboost_binary | NIT_longitudinal_only | moderate       |                  15 |        0.7008 |       0.1292 |       0.5578 |       0.9491 |                     0.6362 |
| hepatic    | xgb_binary      | NIT_longitudinal_only | moderate       |                  15 |        0.7199 |       0.1390 |       0.5335 |       0.9634 |                     0.6504 |
| death      | rsf             | NIT_longitudinal_only | moderate       |                  15 |        0.8508 |       0.0420 |       0.7713 |       0.9130 |                     0.8298 |
| death      | xgb_cox         | NIT_longitudinal_only | moderate       |                  15 |        0.8646 |       0.0462 |       0.7616 |       0.9231 |                     0.8416 |
| death      | lgbm_binary     | NIT_longitudinal_only | moderate       |                  15 |        0.7242 |       0.0741 |       0.6003 |       0.8651 |                     0.6872 |
| death      | catboost_binary | NIT_longitudinal_only | moderate       |                  15 |        0.6847 |       0.0987 |       0.5158 |       0.8373 |                     0.6353 |
| death      | xgb_binary      | NIT_longitudinal_only | moderate       |                  15 |        0.6942 |       0.0847 |       0.5244 |       0.8519 |                     0.6519 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7502350323801773,
  "cindex_hepatic_std": 0.12950752477657834,
  "cindex_hepatic_min": 0.5362549800796813,
  "cindex_hepatic_max": 0.970203488372093,
  "cindex_hepatic_mean_minus_halfsd": 0.6854812699918882,
  "cindex_death_mean": 0.7996665912565064,
  "cindex_death_std": 0.05422935180458374,
  "cindex_death_min": 0.7039258451472192,
  "cindex_death_max": 0.9032258064516129,
  "cindex_death_mean_minus_halfsd": 0.7725519153542145,
  "score_mean": 0.7650645000430759,
  "score_std": 0.09343287512327159,
  "score_min": 0.6104848690345003,
  "score_max": 0.9439352216102805,
  "score_mean_minus_halfsd": 0.7183480624814401,
  "n_folds": 15
}
```
