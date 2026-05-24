# Experiment: phase2_labs_longitudinal_only

- feature_set: `labs_longitudinal_only` (leakage risk: moderate)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set            | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:-----------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | labs_longitudinal_only | moderate       |                  15 |        0.6861 |       0.1070 |       0.5232 |       0.8750 |                     0.6326 |
| hepatic    | xgb_cox         | labs_longitudinal_only | moderate       |                  15 |        0.7376 |       0.1264 |       0.4916 |       0.9268 |                     0.6744 |
| hepatic    | lgbm_binary     | labs_longitudinal_only | moderate       |                  15 |        0.7156 |       0.0940 |       0.5411 |       0.9142 |                     0.6686 |
| hepatic    | catboost_binary | labs_longitudinal_only | moderate       |                  15 |        0.7229 |       0.0768 |       0.5380 |       0.8273 |                     0.6846 |
| hepatic    | xgb_binary      | labs_longitudinal_only | moderate       |                  15 |        0.7332 |       0.0843 |       0.4899 |       0.8670 |                     0.6911 |
| death      | rsf             | labs_longitudinal_only | moderate       |                  15 |        0.8575 |       0.0400 |       0.7961 |       0.9233 |                     0.8375 |
| death      | xgb_cox         | labs_longitudinal_only | moderate       |                  15 |        0.8479 |       0.0439 |       0.7736 |       0.9219 |                     0.8259 |
| death      | lgbm_binary     | labs_longitudinal_only | moderate       |                  15 |        0.7152 |       0.0735 |       0.5917 |       0.8912 |                     0.6784 |
| death      | catboost_binary | labs_longitudinal_only | moderate       |                  15 |        0.6978 |       0.0734 |       0.6103 |       0.9063 |                     0.6611 |
| death      | xgb_binary      | labs_longitudinal_only | moderate       |                  15 |        0.6987 |       0.0809 |       0.5645 |       0.8822 |                     0.6582 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.764499820256378,
  "cindex_hepatic_std": 0.09624388903356389,
  "cindex_hepatic_min": 0.5944223107569722,
  "cindex_hepatic_max": 0.8982558139534884,
  "cindex_hepatic_mean_minus_halfsd": 0.716377875739596,
  "cindex_death_mean": 0.8079379661933535,
  "cindex_death_std": 0.053251326241888566,
  "cindex_death_min": 0.7206303724928367,
  "cindex_death_max": 0.8912386706948641,
  "cindex_death_mean_minus_halfsd": 0.7813123030724092,
  "score_mean": 0.7775312640374706,
  "score_std": 0.07089540563550906,
  "score_min": 0.6480105111469018,
  "score_max": 0.8694858165760952,
  "score_mean_minus_halfsd": 0.7420835612197161,
  "n_folds": 15
}
```
