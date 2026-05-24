# Experiment: phase2_first_1y

- feature_set: `first_1y` (leakage risk: low)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set   | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:--------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | first_1y      | low            |                  15 |        0.7552 |       0.1243 |       0.4860 |       0.9339 |                     0.6930 |
| hepatic    | xgb_cox         | first_1y      | low            |                  15 |        0.6806 |       0.1302 |       0.3821 |       0.8968 |                     0.6155 |
| hepatic    | lgbm_binary     | first_1y      | low            |                  15 |        0.7106 |       0.1012 |       0.5194 |       0.8968 |                     0.6600 |
| hepatic    | catboost_binary | first_1y      | low            |                  15 |        0.7003 |       0.1240 |       0.4899 |       0.9150 |                     0.6383 |
| hepatic    | xgb_binary      | first_1y      | low            |                  15 |        0.6961 |       0.1035 |       0.4589 |       0.8801 |                     0.6444 |
| death      | rsf             | first_1y      | low            |                  15 |        0.8094 |       0.0649 |       0.6628 |       0.9137 |                     0.7769 |
| death      | xgb_cox         | first_1y      | low            |                  15 |        0.7511 |       0.0673 |       0.6322 |       0.8768 |                     0.7175 |
| death      | lgbm_binary     | first_1y      | low            |                  15 |        0.6648 |       0.0805 |       0.4861 |       0.7952 |                     0.6245 |
| death      | catboost_binary | first_1y      | low            |                  15 |        0.6493 |       0.0816 |       0.5437 |       0.7952 |                     0.6085 |
| death      | xgb_binary      | first_1y      | low            |                  15 |        0.6550 |       0.0900 |       0.5031 |       0.8021 |                     0.6100 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7510072321405338,
  "cindex_hepatic_std": 0.11520606391049916,
  "cindex_hepatic_min": 0.5457364341085271,
  "cindex_hepatic_max": 0.9709302325581395,
  "cindex_hepatic_mean_minus_halfsd": 0.6934042001852841,
  "cindex_death_mean": 0.7694572028149435,
  "cindex_death_std": 0.06927056201117229,
  "cindex_death_min": 0.6066411238825032,
  "cindex_death_max": 0.8767720828789531,
  "cindex_death_mean_minus_halfsd": 0.7348219218093573,
  "score_mean": 0.7565422233428566,
  "score_std": 0.0817604560732218,
  "score_min": 0.6177297895902547,
  "score_max": 0.9047703996746403,
  "score_mean_minus_halfsd": 0.7156619953062457,
  "n_folds": 15
}
```
