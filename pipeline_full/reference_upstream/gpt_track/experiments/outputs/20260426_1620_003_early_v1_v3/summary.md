# Experiment: 003_early_v1_v3

- feature_set: `early_v1_v3` (leakage risk: low/moderate)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 5 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set   | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |
|:-----------|:----------------|:--------------|:---------------|--------------------:|--------------:|-------------:|-------------:|
| hepatic    | coxnet          | early_v1_v3   | low/moderate   |                   0 |      nan      |     nan      |     nan      |
| hepatic    | rsf             | early_v1_v3   | low/moderate   |                   0 |      nan      |     nan      |     nan      |
| hepatic    | xgb_cox         | early_v1_v3   | low/moderate   |                  25 |        0.7371 |       0.1135 |       0.5052 |
| hepatic    | lgbm_binary     | early_v1_v3   | low/moderate   |                  25 |        0.7635 |       0.0937 |       0.5386 |
| hepatic    | catboost_binary | early_v1_v3   | low/moderate   |                  25 |        0.7601 |       0.1224 |       0.4574 |
| death      | coxnet          | early_v1_v3   | low/moderate   |                   0 |      nan      |     nan      |     nan      |
| death      | rsf             | early_v1_v3   | low/moderate   |                   0 |      nan      |     nan      |     nan      |
| death      | xgb_cox         | early_v1_v3   | low/moderate   |                  25 |        0.7735 |       0.0596 |       0.6472 |
| death      | lgbm_binary     | early_v1_v3   | low/moderate   |                  25 |        0.7317 |       0.0849 |       0.5724 |
| death      | catboost_binary | early_v1_v3   | low/moderate   |                  25 |        0.6961 |       0.0915 |       0.5147 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.7901580637150378,
  "cindex_hepatic_std": 0.10259369479123708,
  "cindex_hepatic_min": 0.4821705426356589,
  "cindex_death_mean": 0.7654218128345094,
  "cindex_death_std": 0.07178757332286018,
  "cindex_death_min": 0.6480865224625624,
  "score_mean": 0.7827371884508793,
  "score_std": 0.07124795479866443,
  "score_min": 0.5811701734957548,
  "n_folds": 25
}
```
