# Experiment: phase2_longitudinal_no_followup_proxies

- feature_set: `longitudinal_no_followup_proxies` (leakage risk: moderate)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set                      | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:---------------------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | longitudinal_no_followup_proxies | moderate       |                  15 |        0.7933 |       0.1181 |       0.5879 |       0.9583 |                     0.7343 |
| hepatic    | xgb_cox         | longitudinal_no_followup_proxies | moderate       |                  15 |        0.7542 |       0.1221 |       0.4876 |       0.9126 |                     0.6932 |
| hepatic    | lgbm_binary     | longitudinal_no_followup_proxies | moderate       |                  15 |        0.7542 |       0.1098 |       0.5876 |       0.9400 |                     0.6993 |
| hepatic    | catboost_binary | longitudinal_no_followup_proxies | moderate       |                  15 |        0.7456 |       0.1052 |       0.5872 |       0.9070 |                     0.6931 |
| hepatic    | xgb_binary      | longitudinal_no_followup_proxies | moderate       |                  15 |        0.7498 |       0.1132 |       0.5721 |       0.9238 |                     0.6932 |
| death      | rsf             | longitudinal_no_followup_proxies | moderate       |                  15 |        0.8707 |       0.0347 |       0.8124 |       0.9205 |                     0.8533 |
| death      | xgb_cox         | longitudinal_no_followup_proxies | moderate       |                  15 |        0.8665 |       0.0318 |       0.8240 |       0.9418 |                     0.8505 |
| death      | lgbm_binary     | longitudinal_no_followup_proxies | moderate       |                  15 |        0.7699 |       0.0608 |       0.6418 |       0.8943 |                     0.7395 |
| death      | catboost_binary | longitudinal_no_followup_proxies | moderate       |                  15 |        0.7251 |       0.0696 |       0.6032 |       0.8731 |                     0.6903 |
| death      | xgb_binary      | longitudinal_no_followup_proxies | moderate       |                  15 |        0.7581 |       0.0646 |       0.6649 |       0.8867 |                     0.7258 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7764491908078284,
  "cindex_hepatic_std": 0.11945299676131842,
  "cindex_hepatic_min": 0.5784860557768924,
  "cindex_hepatic_max": 0.9496951219512195,
  "cindex_hepatic_mean_minus_halfsd": 0.7167226924271691,
  "cindex_death_mean": 0.8439626852028508,
  "cindex_death_std": 0.047019618261246385,
  "cindex_death_min": 0.7867177522349936,
  "cindex_death_max": 0.9129782445611403,
  "cindex_death_mean_minus_halfsd": 0.8204528760722276,
  "score_mean": 0.7967032391263351,
  "score_std": 0.08552103818939835,
  "score_min": 0.6432381113842502,
  "score_max": 0.9386800587341957,
  "score_mean_minus_halfsd": 0.753942720031636,
  "n_folds": 15
}
```
