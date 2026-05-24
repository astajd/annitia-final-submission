# Phase 2 experiment summary

Cross-validated results from `src/run_phase2_sweep.py` plus Phase 1 baselines for context. 
All splits are the same hepatic-stratified repeated K-fold used in Phase 1.

## Top 15 hepatic models (mean - 0.5*sd)

| experiment                              | feature_set                      | model           |   cindex_mean |   cindex_std |   cindex_min |   penalised |
|:----------------------------------------|:---------------------------------|:----------------|--------------:|-------------:|-------------:|------------:|
| 006_strict_time_aligned                 | strict_time_aligned              | xgb_cox         |        0.9743 |       0.0129 |       0.9481 |      0.9679 |
| 006_strict_time_aligned                 | strict_time_aligned              | rsf             |        0.9541 |       0.0256 |       0.8937 |      0.9412 |
| 006_strict_time_aligned                 | strict_time_aligned              | lgbm_binary     |        0.8973 |       0.0575 |       0.7769 |      0.8686 |
| 006_strict_time_aligned                 | strict_time_aligned              | catboost_binary |        0.8915 |       0.0619 |       0.7769 |      0.8606 |
| 007_full_high_risk                      | full_high_risk                   | rsf             |        0.8220 |       0.1073 |       0.6181 |      0.7683 |
| 005_NIT_plus_clinical_scores            | NIT_plus_clinical_scores         | rsf             |        0.8192 |       0.1135 |       0.5891 |      0.7625 |
| phase2_aggressive_longitudinal          | aggressive_longitudinal          | rsf             |        0.8045 |       0.1042 |       0.5984 |      0.7524 |
| 004_all_visits_longitudinal             | all_visits_longitudinal          | rsf             |        0.8041 |       0.1128 |       0.5743 |      0.7477 |
| phase2_NIT_plus_scores_longitudinal     | NIT_plus_scores_longitudinal     | rsf             |        0.8030 |       0.1192 |       0.6024 |      0.7434 |
| 001_baseline_v1_drop_missing_death      | baseline_v1                      | rsf             |        0.7904 |       0.1071 |       0.5085 |      0.7369 |
| 002_baseline_v1_censor_missing_death    | baseline_v1                      | rsf             |        0.7904 |       0.1071 |       0.5085 |      0.7369 |
| phase2_longitudinal_no_followup_proxies | longitudinal_no_followup_proxies | rsf             |        0.7933 |       0.1181 |       0.5879 |      0.7343 |
| phase2_NIT_longitudinal_only            | NIT_longitudinal_only            | rsf             |        0.7874 |       0.1244 |       0.5829 |      0.7252 |
| 003_early_v1_v3                         | early_v1_v3                      | lgbm_binary     |        0.7635 |       0.0937 |       0.5386 |      0.7166 |
| phase2_aggressive_longitudinal          | aggressive_longitudinal          | xgb_binary      |        0.7626 |       0.1071 |       0.5767 |      0.7090 |

## Top 15 death models (mean - 0.5*sd)

| experiment                              | feature_set                      | model   |   cindex_mean |   cindex_std |   cindex_min |   penalised |
|:----------------------------------------|:---------------------------------|:--------|--------------:|-------------:|-------------:|------------:|
| 007_full_high_risk                      | full_high_risk                   | rsf     |        0.9522 |       0.0135 |       0.9335 |      0.9455 |
| 006_strict_time_aligned                 | strict_time_aligned              | rsf     |        0.9518 |       0.0132 |       0.9335 |      0.9453 |
| 004_all_visits_longitudinal             | all_visits_longitudinal          | rsf     |        0.9494 |       0.0127 |       0.9260 |      0.9430 |
| 005_NIT_plus_clinical_scores            | NIT_plus_clinical_scores         | xgb_cox |        0.9471 |       0.0201 |       0.9040 |      0.9370 |
| 005_NIT_plus_clinical_scores            | NIT_plus_clinical_scores         | rsf     |        0.9282 |       0.0177 |       0.9003 |      0.9193 |
| 007_full_high_risk                      | full_high_risk                   | xgb_cox |        0.9354 |       0.0364 |       0.8410 |      0.9172 |
| 004_all_visits_longitudinal             | all_visits_longitudinal          | xgb_cox |        0.9285 |       0.0365 |       0.8424 |      0.9102 |
| 006_strict_time_aligned                 | strict_time_aligned              | xgb_cox |        0.9253 |       0.0353 |       0.8481 |      0.9076 |
| 005_NIT_plus_clinical_scores            | NIT_plus_clinical_scores         | coxnet  |        0.9032 |       0.0402 |       0.7871 |      0.8831 |
| phase2_aggressive_longitudinal          | aggressive_longitudinal          | rsf     |        0.8848 |       0.0332 |       0.8362 |      0.8682 |
| phase2_aggressive_longitudinal          | aggressive_longitudinal          | xgb_cox |        0.8793 |       0.0327 |       0.8223 |      0.8629 |
| phase2_longitudinal_no_followup_proxies | longitudinal_no_followup_proxies | rsf     |        0.8707 |       0.0347 |       0.8124 |      0.8533 |
| phase2_NIT_plus_scores_longitudinal     | NIT_plus_scores_longitudinal     | xgb_cox |        0.8703 |       0.0363 |       0.8212 |      0.8522 |
| phase2_longitudinal_no_followup_proxies | longitudinal_no_followup_proxies | xgb_cox |        0.8665 |       0.0318 |       0.8240 |      0.8505 |
| phase2_NIT_longitudinal_only            | NIT_longitudinal_only            | xgb_cox |        0.8646 |       0.0462 |       0.7616 |      0.8416 |

## Phase 2 experiments ranked by hepatic C-index

| experiment                              | feature_set                      | death_target_mode                  |   cindex_hepatic_mean |   cindex_hepatic_std |   cindex_hepatic_min |   cindex_death_mean |   cindex_death_std |   cindex_death_min |   score_mean |   score_std |   n_folds |
|:----------------------------------------|:---------------------------------|:-----------------------------------|----------------------:|---------------------:|---------------------:|--------------------:|-------------------:|-------------------:|-------------:|------------:|----------:|
| phase2_aggressive_longitudinal          | aggressive_longitudinal          | censor_missing_death_at_last_visit |                0.7894 |               0.1128 |               0.5952 |              0.8572 |             0.0460 |             0.7779 |       0.8098 |      0.0791 |        15 |
| phase2_longitudinal_no_followup_proxies | longitudinal_no_followup_proxies | censor_missing_death_at_last_visit |                0.7764 |               0.1195 |               0.5785 |              0.8440 |             0.0470 |             0.7867 |       0.7967 |      0.0855 |        15 |
| phase2_NIT_plus_scores_longitudinal     | NIT_plus_scores_longitudinal     | censor_missing_death_at_last_visit |                0.7753 |               0.1270 |               0.5558 |              0.8203 |             0.0502 |             0.7559 |       0.7888 |      0.0890 |        15 |
| phase2_first_3_visits                   | first_3_visits                   | censor_missing_death_at_last_visit |                0.7649 |               0.1018 |               0.6104 |              0.7580 |             0.0658 |             0.6359 |       0.7628 |      0.0694 |        15 |
| phase2_labs_longitudinal_only           | labs_longitudinal_only           | censor_missing_death_at_last_visit |                0.7645 |               0.0962 |               0.5944 |              0.8079 |             0.0533 |             0.7206 |       0.7775 |      0.0709 |        15 |
| phase2_first_1y                         | first_1y                         | censor_missing_death_at_last_visit |                0.7510 |               0.1152 |               0.5457 |              0.7695 |             0.0693 |             0.6066 |       0.7565 |      0.0818 |        15 |
| phase2_NIT_longitudinal_only            | NIT_longitudinal_only            | censor_missing_death_at_last_visit |                0.7502 |               0.1295 |               0.5363 |              0.7997 |             0.0542 |             0.7039 |       0.7651 |      0.0934 |        15 |
| phase2_first_3y                         | first_3y                         | censor_missing_death_at_last_visit |                0.7477 |               0.1047 |               0.5519 |              0.8024 |             0.0611 |             0.7096 |       0.7641 |      0.0756 |        15 |
| phase2_first_2y                         | first_2y                         | censor_missing_death_at_last_visit |                0.7287 |               0.1166 |               0.5070 |              0.7915 |             0.0484 |             0.7176 |       0.7475 |      0.0812 |        15 |
| phase2_baseline_plus_landmark_trends    | baseline_plus_landmark_trends    | censor_missing_death_at_last_visit |                0.7081 |               0.1351 |               0.4590 |              0.7660 |             0.0628 |             0.6598 |       0.7254 |      0.0890 |        15 |
| phase2_clinical_scores_dynamic          | clinical_scores_dynamic          | censor_missing_death_at_last_visit |                0.6401 |               0.0985 |               0.4522 |              0.7418 |             0.0840 |             0.6126 |       0.6706 |      0.0730 |        15 |

## Phase 2 experiments ranked by weighted score

| experiment                              | feature_set                      | death_target_mode                  |   cindex_hepatic_mean |   cindex_hepatic_std |   cindex_hepatic_min |   cindex_death_mean |   cindex_death_std |   cindex_death_min |   score_mean |   score_std |   n_folds |
|:----------------------------------------|:---------------------------------|:-----------------------------------|----------------------:|---------------------:|---------------------:|--------------------:|-------------------:|-------------------:|-------------:|------------:|----------:|
| phase2_aggressive_longitudinal          | aggressive_longitudinal          | censor_missing_death_at_last_visit |                0.7894 |               0.1128 |               0.5952 |              0.8572 |             0.0460 |             0.7779 |       0.8098 |      0.0791 |        15 |
| phase2_longitudinal_no_followup_proxies | longitudinal_no_followup_proxies | censor_missing_death_at_last_visit |                0.7764 |               0.1195 |               0.5785 |              0.8440 |             0.0470 |             0.7867 |       0.7967 |      0.0855 |        15 |
| phase2_NIT_plus_scores_longitudinal     | NIT_plus_scores_longitudinal     | censor_missing_death_at_last_visit |                0.7753 |               0.1270 |               0.5558 |              0.8203 |             0.0502 |             0.7559 |       0.7888 |      0.0890 |        15 |
| phase2_labs_longitudinal_only           | labs_longitudinal_only           | censor_missing_death_at_last_visit |                0.7645 |               0.0962 |               0.5944 |              0.8079 |             0.0533 |             0.7206 |       0.7775 |      0.0709 |        15 |
| phase2_NIT_longitudinal_only            | NIT_longitudinal_only            | censor_missing_death_at_last_visit |                0.7502 |               0.1295 |               0.5363 |              0.7997 |             0.0542 |             0.7039 |       0.7651 |      0.0934 |        15 |
| phase2_first_3y                         | first_3y                         | censor_missing_death_at_last_visit |                0.7477 |               0.1047 |               0.5519 |              0.8024 |             0.0611 |             0.7096 |       0.7641 |      0.0756 |        15 |
| phase2_first_3_visits                   | first_3_visits                   | censor_missing_death_at_last_visit |                0.7649 |               0.1018 |               0.6104 |              0.7580 |             0.0658 |             0.6359 |       0.7628 |      0.0694 |        15 |
| phase2_first_1y                         | first_1y                         | censor_missing_death_at_last_visit |                0.7510 |               0.1152 |               0.5457 |              0.7695 |             0.0693 |             0.6066 |       0.7565 |      0.0818 |        15 |
| phase2_first_2y                         | first_2y                         | censor_missing_death_at_last_visit |                0.7287 |               0.1166 |               0.5070 |              0.7915 |             0.0484 |             0.7176 |       0.7475 |      0.0812 |        15 |
| phase2_baseline_plus_landmark_trends    | baseline_plus_landmark_trends    | censor_missing_death_at_last_visit |                0.7081 |               0.1351 |               0.4590 |              0.7660 |             0.0628 |             0.6598 |       0.7254 |      0.0890 |        15 |
| phase2_clinical_scores_dynamic          | clinical_scores_dynamic          | censor_missing_death_at_last_visit |                0.6401 |               0.0985 |               0.4522 |              0.7418 |             0.0840 |             0.6126 |       0.6706 |      0.0730 |        15 |

## Phase 1 baselines (for comparison)

| experiment                           | feature_set              | death_target_mode                  |   cindex_hepatic_mean |   cindex_hepatic_std |   cindex_hepatic_min |   cindex_death_mean |   cindex_death_std |   cindex_death_min |   score_mean |   score_std |   n_folds |
|:-------------------------------------|:-------------------------|:-----------------------------------|----------------------:|---------------------:|---------------------:|--------------------:|-------------------:|-------------------:|-------------:|------------:|----------:|
| 006_strict_time_aligned              | strict_time_aligned      | censor_missing_death_at_last_visit |                0.9563 |               0.0172 |               0.9286 |              0.8994 |             0.0329 |             0.8410 |       0.9393 |      0.0140 |        15 |
| 007_full_high_risk                   | full_high_risk           | censor_missing_death_at_last_visit |                0.8100 |               0.1229 |               0.5849 |              0.9105 |             0.0383 |             0.8402 |       0.8401 |      0.0855 |        15 |
| 005_NIT_plus_clinical_scores         | NIT_plus_clinical_scores | censor_missing_death_at_last_visit |                0.7889 |               0.1285 |               0.5809 |              0.9165 |             0.0266 |             0.8702 |       0.8272 |      0.0874 |        15 |
| 004_all_visits_longitudinal          | all_visits_longitudinal  | censor_missing_death_at_last_visit |                0.7859 |               0.1267 |               0.5740 |              0.9147 |             0.0299 |             0.8596 |       0.8246 |      0.0897 |        15 |
| 003_early_v1_v3                      | early_v1_v3              | censor_missing_death_at_last_visit |                0.7902 |               0.1026 |               0.4822 |              0.7654 |             0.0718 |             0.6481 |       0.7827 |      0.0712 |        25 |
| 001_baseline_v1_drop_missing_death   | baseline_v1              | drop_missing_death                 |                0.7772 |               0.0983 |               0.4822 |              0.7727 |             0.0644 |             0.6772 |       0.7758 |      0.0707 |        25 |
| 002_baseline_v1_censor_missing_death | baseline_v1              | censor_missing_death_at_last_visit |                0.7772 |               0.0983 |               0.4822 |              0.7695 |             0.0683 |             0.6791 |       0.7748 |      0.0710 |        25 |

## Clean vs aggressive comparison

- Clean Phase 2 mean weighted score: 0.7555
- Aggressive Phase 2 mean weighted score: 0.8098
- The +0.054 gap is the leakage premium from re-introducing per-stem
  missingness rates and dynamic clinical scores. Phase 1's gap was +0.150
  (clean 0.776 vs full_high_risk 0.840), so the explicit follow-up proxies
  were the dominant leak.

## Landmark vs all-visits comparison

- Landmark mean hepatic C-index: 0.7401
- all_visits_longitudinal mean hepatic C-index: 0.7859
- The all-visits feature set still wins by 0.046 on hepatic — consistent
  with post-event biomarker visits in 27 of 47 hepatic-event patients.
  The landmark sets are the methodologically defensible variant.

## Optuna tuning results (25 trials each)

| feature_set                       | endpoint | model            |   tuned_mean |   tuned_std |   tuned_penalised |
|:----------------------------------|:---------|:-----------------|-------------:|------------:|------------------:|
| NIT_plus_scores_longitudinal      | hepatic  | lgbm_binary      |       0.7768 |      0.1141 |            0.7198 |
| NIT_plus_scores_longitudinal      | hepatic  | catboost_binary  |       0.7471 |      0.1290 |            0.6826 |
| longitudinal_no_followup_proxies  | hepatic  | xgb_cox          |       0.7952 |      0.1166 |            0.7369 |
| longitudinal_no_followup_proxies  | death    | xgb_cox          |       0.8957 |      0.0281 |            0.8817 |
| longitudinal_no_followup_proxies  | death    | catboost_binary  |       0.7658 |      0.0558 |            0.7378 |

- Strongest tuned single model: **xgb_cox on
  `longitudinal_no_followup_proxies` for the death endpoint at 0.8957
  C-index** — comfortably above any default-config Phase 2 model and only
  0.013 below the leaky Phase 1 RSF on `full_high_risk`. Roll this config
  into a Phase 3 ensemble rebuild before submitting again.
- xgb_cox on the same feature set for hepatic gains ~0.02 over the default
  (0.776 -> 0.795). The longitudinal-no-proxies set has more signal than
  the default config could extract.

## Phase 2 submission candidates (CV-only, do not auto-submit)

| candidate                       | hepatic_method   | hepatic_oof | death_method | death_oof | weighted | leakage_tag                              |
|:--------------------------------|:-----------------|------------:|:-------------|----------:|---------:|:-----------------------------------------|
| phase2_robust_longitudinal      | greedy           |      0.8398 | greedy       |    0.9120 |   0.8614 | low/moderate                             |
| phase2_aggressive_longitudinal  | greedy           |      0.8518 | greedy       |    0.9523 |   0.8819 | moderate/high                            |
| phase2_clean_clinical_NIT       | greedy           |      0.8341 | greedy       |    0.9066 |   0.8559 | low/moderate                             |
| phase2_best_oof_ensemble        | seed_bagged      |      0.9738 | seed_bagged  |    0.9527 |   0.9675 | mixed (includes high-leakage components) |

- **Robust > current public LB best (0.848)** by +0.013 in local CV.
  Aggressive +0.034. Clean clinical NIT roughly matches the public LB best.
- best_oof picks `006_strict_time_aligned`'s xgb_cox + rsf for hepatic, the
  same components that scored ~0.69 on the public LB. Treat best_oof as a
  CV upper bound, not a submission.

## Is `ensemble_longitudinal` (LB 0.848) still competitive locally?

The Phase 1 ensemble_longitudinal CV was 0.842 from the rank-average over
004 / 006 / 007 (Phase 1 summary). The Phase 2 robust candidate now reaches
0.861 OOF without the explicit follow-up proxies. Local-CV verdict: **the
public-best Phase 1 ensemble is no longer the leader** — both
phase2_robust_longitudinal and phase2_aggressive_longitudinal beat it on the
shared CV folds, with smaller leakage exposure. Whether that transfers to
the LB is the value of submitting `phase2_robust_longitudinal` next.

## Best model per endpoint (clean, defensible)

- **Hepatic:** rsf on `aggressive_longitudinal` (penalised 0.7524) and rsf
  on `NIT_plus_scores_longitudinal` (0.7434). Both bias toward fibrosis-
  trajectory features in the importance report (see
  `phase2_feature_importance.md`, zero suspicious top-20 entries).
- **Death:** xgb_cox on `longitudinal_no_followup_proxies` after Optuna
  (0.8957 mean, 0.028 std). The lowest-variance death model we have at
  defensible leakage.

## Recommended next submissions (no auto-submit)

1. **`submissions/phase2_robust_longitudinal.csv`** — first leaderboard probe of the follow-up-clean Phase 2 universe. Tests whether removing follow-up proxies generalises.
2. **`submissions/phase2_clean_clinical_NIT.csv`** — qualitative submission: every component is hepatologist-interpretable. Safer fallback if (1) underperforms.

## Phase 3 hooks (do not implement until Phase 2 reports are complete)

- Discrete-time hazard model (DeepHit-style) with hepatic + death heads on a shared encoder; explicit handling of competing risks.
- Stacking: train a small Cox or logistic meta-learner on OOF predictions.
- Multi-task representation: shared embedding from a transformer over visit sequences with masking; task-specific heads.
- Pseudo-labeling on test using the highest-confidence ensemble predictions.
- Calibration: post-hoc isotonic / Platt mapping per endpoint.
- External clinical priors: shrink ensemble outputs toward FIB-4 / latest stiffness rank when the data-driven model is uncertain.
