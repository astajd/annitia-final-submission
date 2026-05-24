# Phase 1 experiment summary

Cross-validation results from `python -m src.run_phase1`. Competition score is
`0.7 * C-index(hepatic) + 0.3 * C-index(death)`. All experiments share the same
hepatic-event-stratified repeated K-fold splits, so deltas across rows are not
contaminated by fold randomness.

## Per-experiment CV results

| experiment                           | feature_set              | death_target_mode                  |   score_mean |   score_std |   cindex_hepatic_mean |   cindex_death_mean |   n_folds | leakage_risk    |
|:-------------------------------------|:-------------------------|:-----------------------------------|-------------:|------------:|----------------------:|--------------------:|----------:|:----------------|
| 001_baseline_v1_drop_missing_death   | baseline_v1              | drop_missing_death                 |       0.7758 |      0.0707 |                0.7772 |              0.7727 |        25 | low             |
| 002_baseline_v1_censor_missing_death | baseline_v1              | censor_missing_death_at_last_visit |       0.7748 |      0.0710 |                0.7772 |              0.7695 |        25 | low             |
| 003_early_v1_v3                      | early_v1_v3              | censor_missing_death_at_last_visit |       0.7827 |      0.0712 |                0.7902 |              0.7654 |        25 | low/moderate    |
| 004_all_visits_longitudinal          | all_visits_longitudinal  | censor_missing_death_at_last_visit |       0.8246 |      0.0897 |                0.7859 |              0.9147 |        15 | high            |
| 005_NIT_plus_clinical_scores         | NIT_plus_clinical_scores | censor_missing_death_at_last_visit |       0.8272 |      0.0874 |                0.7889 |              0.9165 |        15 | low/moderate*   |
| 006_strict_time_aligned              | strict_time_aligned      | censor_missing_death_at_last_visit |       0.9393 |      0.0140 |                0.9563 |              0.8994 |        15 | low/moderate**  |
| 007_full_high_risk                   | full_high_risk           | censor_missing_death_at_last_visit |       0.8401 |      0.0855 |                0.8100 |              0.9105 |        15 | high            |

\* the NIT/clinical bundle re-uses `all_visits_longitudinal` for the NIT stems, so it inherits its leakage.

\** strict masking removes biomarker values past the event, but the *count*, *age_first*, *age_last*, and *time_since_baseline* features still encode the cutoff age and act as leakage proxies. Treat the headline 0.94 with skepticism. See "Caveats" below.

## Best per metric

- best weighted score (raw CV): **006_strict_time_aligned** (0.9393)
  - inflated by cutoff-encoding leakage; not the pick for the qualitative submission.
- best hepatic C-index (raw CV): **006_strict_time_aligned** (0.9563)
- best death C-index (raw CV): **005_NIT_plus_clinical_scores** (0.9165)
- best **clean** weighted score: **003_early_v1_v3** (0.7827)
  - this is the strongest score that does not depend on follow-up/cutoff leakage.

## What the leakage audit told us

`reports/leakage_audit.md` quantifies how much pure follow-up bookkeeping
already separates events from non-events:

- 27 of 47 hepatic-event patients have visits *after* the recorded event age.
  This is why `all_visits_longitudinal` aggregates over post-event measurements
  and why 004 / 005 / 007 are flagged high-leakage.
- Single-feature C-index of `-followup_years` against the death endpoint: **0.97**.
  This is essentially the death-time variable masquerading as a feature, because
  in train `followup_years = death_age - Age_v1` for died patients and
  `last_age - Age_v1` for everyone else. Test does not have this property.
- `Age_v1` alone gives 0.79 on death — biology, not leakage.

Together: any model that consumes the longitudinal *cadence* of train ends up
ranking patients by their cutoff time. CV is rosy, hidden test is not.

## Death NaN handling: 001 vs 002

001 (`drop_missing_death`) and 002 (`censor_missing_death_at_last_visit`) are
within 0.001 score of each other. The 269 patients with missing death status
look very similar to censored ones (median 2 visits, 1-year follow-up, near-zero
hepatic-event rate), so treating them as censored at the last visit costs almost
nothing while keeping every training row — that is the recommended default.
The drop variant edges 002 by 0.0010 on the death endpoint here, so we will
keep both modes in Phase 2 hyper-tuning.

## Cross-experiment ensembles

| ensemble                | components                          | submission                                                  |
|:------------------------|:------------------------------------|:------------------------------------------------------------|
| ensemble_clean          | 001, 002, 003, 005, 006             | `submissions/20260426_1640_ensemble_clean.csv`              |
| ensemble_longitudinal   | 004, 006, 007                        | `submissions/20260426_1640_ensemble_longitudinal.csv`       |
| ensemble_hybrid         | all of the above                    | `submissions/20260426_1640_ensemble_hybrid.csv`             |

`ensemble_clean` is the recommended **qualitative** submission: stable across
folds, defensible methodology, no obviously leaked features. `ensemble_hybrid`
is the **leaderboard** probe — if it scores meaningfully higher than
`ensemble_clean` on the public board, the gap is the leakage premium and it
will likely shrink on the private board.

## Caveats / known issues

- **Cutoff-encoding leakage in strict_time_aligned (006).** Even after masking
  post-event biomarker values, the longitudinal summary keeps `count`,
  `age_first`, `age_last`, and `time_since_baseline`. For event patients these
  collapse to the event time; for censored patients they equal the censoring
  time. The model trivially learns "small time_since_baseline -> high risk".
  Phase 2 should re-run 006 with a "feature-clean" variant that drops these
  columns and uses time-fixed snapshots (e.g. v1 + 12-month + 24-month).
- **CoxNet/RSF on early_v1_v3 (003).** Both estimators failed every fold
  (likely `inf` slopes when ages tied between v1 and v3). Phase 2 should clip
  slopes and rerun. The remaining models in 003 still produced a usable score.
- **Per-fold variance is large for hepatic** (std ~0.10 in the clean
  experiments). 47 events split across 5 folds is ~9 per fold, so worst-fold
  C-index can dip below 0.5. Increase `n_repeats` to 10 in Phase 2 to tighten
  the mean estimate.

## Recommendations

### First leaderboard submission

Submit `submissions/20260426_1640_ensemble_clean.csv` (rank-average of 001, 002,
003, 005, 006). It is the strongest CV score whose generalization story we
trust. Also submit `ensemble_hybrid` as a sanity probe of the leakage premium.

### Phase 2 (overnight)

1. **Feature-clean strict alignment.** Re-derive the strict_time_aligned
   feature set without the cutoff-revealing fields (`*_count`, `*_age_first`,
   `*_age_last`, `*_time_since_baseline`). Compare CV vs leaderboard delta
   against the leaky variant.
2. **More repeats.** 5 folds x 10 repeats for the clean experiments to tighten
   the mean and stabilize the worst-fold metric used to make decisions.
3. **Bayesian tuning** of CoxNet (l1_ratio, alphas), RSF (max_depth,
   min_samples_leaf), XGB-Cox (max_depth, lr, n_estimators), per endpoint.
4. **CV-weighted ensembling.** Replace the equal-weight rank average with
   weights chosen to maximize OOF score per endpoint (constrained convex
   combination on the simplex).
5. **Multi-task / shared-trunk hazard models.** A discrete-time hazard
   model (DeepHit / DeepSurv head per endpoint, shared encoder) can borrow
   strength across the rare hepatic events. Hooks already exist in
   `src/models/__init__.py`; add a `multitask_hazard.py` wrapper.
6. **Stacking.** Use OOF predictions as features for a small Cox meta-learner.
7. **Death NaN sensitivity.** Re-run 001-vs-002 with bigger repeats to confirm
   the effect size is below CV noise.
