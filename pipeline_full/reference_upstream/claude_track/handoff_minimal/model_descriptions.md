# Model descriptions — top 3 submissions above 0.88 LB

## 1. phase2_blend_2way_optimal (LB **0.8965**) — champion

Two-model rank blend:
- **landmark_3y_RSF**: RandomSurvivalForest tuned via Optuna, trained on `landmark_3y` (Tier 2) — 49 features = `baseline_v1` static + LOCF values at the 3-year landmark + slope of each variable from baseline to landmark. Trained on the at-risk-at-3y cohort (filter A inclusion).
- **permissive_ensemble_avg**: rank-mean of three Tier-4 models that use the full per-visit trajectory regardless of timing relative to the event. Members:
  1. `rsf_longitudinal_summary` (RSF on `_last`/`_max`/`_count`/`_std`/`_slope` aggregates)
  2. `xgb_cox_longitudinal_plus_meta` (XGBoost survival:cox + visit metadata)
  3. `xgb_cox_longitudinal_summary` (XGBoost survival:cox on aggregates)

Blend weight `0.30·rank(landmark) + 0.70·rank(permissive_avg)` was selected by OOF stacking
(`phase2_stack_2way.py`) on the 5×10 fold structure: scanning `w` ∈ {0.0, 0.05, …, 1.0}
and picking the maximum mean−std hepatic C-index. The 0.30 minimum was robust to a
LOFO check (drop one fold-repeat at a time; argmax stayed in {0.30, 0.35, 0.40}).

Death predictor: XGB-Cox on `longitudinal_summary` (`n_estimators=300, learning_rate=0.05`).
Identical death predictor across all Phase 2 submissions, so cross-submission deltas are
hepatic-driven.

Why it works: the two hepatic components have OOF Spearman 0.478, comfortably above the
0.5-or-better diversity threshold we found yields LB lift. The permissive component
contributes most of the predictive signal (70% weight); the landmark component acts as
a regularizer pulling toward the at-risk-at-3y subpopulation rather than as a primary
predictor.

## 2. phase2_blend_2way_strongpermissive (LB **0.8867**) — CV-LB inversion #1

Same recipe as the champion, but the 3-member permissive ensemble was replaced with the
single best member (`rsf_longitudinal_summary`, OOF mean−std 0.7356 vs the ensemble's
0.7197). OOF re-stacking gave `w = 0.23·landmark + 0.77·rsf_longitudinal_summary`.
Despite +0.014 CV gain, LB dropped 0.010. Conclusion: the 3-XGB redundancy in the
permissive ensemble was acting as variance reduction on the public LB, not signal
dilution as the CV had implied. This was the first of three consecutive CV→LB
inversions that established a noise-floor lower bound on what CV improvement is required
to expect LB lift.

## 3. phase2_blend_landmark_permissive (LB **0.880**) — equal-weight predecessor

Same components as the champion, but with naive 50/50 weights. The OOF re-weighting from
50/50 → 30/70 amplified to +0.016 LB. This established that, in our regime, even small
CV gains on the optimal-weights search can transfer to non-trivial LB gains as long as
the underlying ensemble structure (3 permissive XGB-RSF members + 1 landmark RSF) is
preserved.
