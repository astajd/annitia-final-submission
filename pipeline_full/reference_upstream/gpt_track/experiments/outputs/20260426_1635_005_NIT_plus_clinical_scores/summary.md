# Experiment: 005_NIT_plus_clinical_scores

- feature_set: `NIT_plus_clinical_scores` (leakage risk: low/moderate)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set              | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |
|:-----------|:----------------|:-------------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|
| hepatic    | coxnet          | NIT_plus_clinical_scores | low/moderate   |                  15 |        0.7028 |       0.1199 |       0.4584 |
| hepatic    | rsf             | NIT_plus_clinical_scores | low/moderate   |                  15 |        0.8192 |       0.1135 |       0.5891 |
| hepatic    | xgb_cox         | NIT_plus_clinical_scores | low/moderate   |                  15 |        0.7715 |       0.1451 |       0.4680 |
| hepatic    | lgbm_binary     | NIT_plus_clinical_scores | low/moderate   |                  15 |        0.7582 |       0.1168 |       0.5827 |
| hepatic    | catboost_binary | NIT_plus_clinical_scores | low/moderate   |                  15 |        0.7132 |       0.1271 |       0.5416 |
| death      | coxnet          | NIT_plus_clinical_scores | low/moderate   |                  15 |        0.9032 |       0.0402 |       0.7871 |
| death      | rsf             | NIT_plus_clinical_scores | low/moderate   |                  15 |        0.9282 |       0.0177 |       0.9003 |
| death      | xgb_cox         | NIT_plus_clinical_scores | low/moderate   |                  15 |        0.9471 |       0.0201 |       0.9040 |
| death      | lgbm_binary     | NIT_plus_clinical_scores | low/moderate   |                  15 |        0.7574 |       0.0604 |       0.6590 |
| death      | catboost_binary | NIT_plus_clinical_scores | low/moderate   |                  15 |        0.7427 |       0.0704 |       0.6052 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.788887020305271,
  "cindex_hepatic_std": 0.12852247337234096,
  "cindex_hepatic_min": 0.5808764940239044,
  "cindex_death_mean": 0.9165195275517874,
  "cindex_death_std": 0.026632928664699147,
  "cindex_death_min": 0.8702290076335878,
  "score_mean": 0.8271767724792258,
  "score_std": 0.08741206034799194,
  "score_min": 0.6821454607103501,
  "n_folds": 15
}
```
