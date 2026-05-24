# Experiment: phase2_clinical_scores_dynamic

- feature_set: `clinical_scores_dynamic` (leakage risk: low)
- death_target_mode: `censor_missing_death_at_last_visit`
- folds: 5 x 3 repeats
- hepatic events: 47 / 1253
- death events:   76 / 1253

## Per-model CV summary

| endpoint   | model           | feature_set             | leakage_risk   |   n_folds_evaluated |   cindex_mean |   cindex_std |   cindex_min |   cindex_max |   cindex_mean_minus_halfsd |
|:-----------|:----------------|:------------------------|:---------------|--------------------:|--------------:|-------------:|-------------:|-------------:|---------------------------:|
| hepatic    | rsf             | clinical_scores_dynamic | low            |                  15 |        0.6345 |       0.0890 |       0.4853 |       0.8032 |                     0.5900 |
| hepatic    | xgb_cox         | clinical_scores_dynamic | low            |                  15 |        0.6669 |       0.1129 |       0.3721 |       0.8102 |                     0.6105 |
| hepatic    | lgbm_binary     | clinical_scores_dynamic | low            |                  15 |        0.6359 |       0.1020 |       0.4088 |       0.7809 |                     0.5849 |
| hepatic    | catboost_binary | clinical_scores_dynamic | low            |                  15 |        0.5843 |       0.1129 |       0.4223 |       0.7358 |                     0.5278 |
| hepatic    | xgb_binary      | clinical_scores_dynamic | low            |                  15 |        0.6407 |       0.1063 |       0.4031 |       0.7707 |                     0.5876 |
| death      | rsf             | clinical_scores_dynamic | low            |                  15 |        0.7932 |       0.0641 |       0.6701 |       0.9040 |                     0.7612 |
| death      | xgb_cox         | clinical_scores_dynamic | low            |                  15 |        0.8229 |       0.0725 |       0.6862 |       0.9365 |                     0.7867 |
| death      | lgbm_binary     | clinical_scores_dynamic | low            |                  15 |        0.6191 |       0.0845 |       0.4770 |       0.7719 |                     0.5768 |
| death      | catboost_binary | clinical_scores_dynamic | low            |                  15 |        0.6480 |       0.0701 |       0.5409 |       0.7706 |                     0.6129 |
| death      | xgb_binary      | clinical_scores_dynamic | low            |                  15 |        0.6090 |       0.0823 |       0.4557 |       0.7376 |                     0.5679 |

## Ensemble CV summary

```json
{
  "cindex_hepatic_mean": 0.6401218471660658,
  "cindex_hepatic_std": 0.09853802700490229,
  "cindex_hepatic_min": 0.45219123505976094,
  "cindex_hepatic_max": 0.7501930501930502,
  "cindex_hepatic_mean_minus_halfsd": 0.5908528336636146,
  "cindex_death_mean": 0.741811075973348,
  "cindex_death_std": 0.08403420838201121,
  "cindex_death_min": 0.6125886524822695,
  "cindex_death_max": 0.8724100327153762,
  "cindex_death_mean_minus_halfsd": 0.6997939717823425,
  "score_mean": 0.6706286158082506,
  "score_std": 0.07303516195995037,
  "score_min": 0.5003104602865135,
  "score_max": 0.7468656453296645,
  "score_mean_minus_halfsd": 0.6341110348282755,
  "n_folds": 15
}
```
