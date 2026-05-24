# Phase 3 Experiment 1 — Negative result note for the methodology writeup

Date: 2026-04-28.

## TL;DR

Trajectory-shape features were tested as a candidate diversity-add to the
Phase 2 best blend. **They did not qualify.** The best shape-using model
(RSF on `baseline_plus_shape`, 100 features) had OOF Spearman 0.754 with the
existing `permissive_ensemble_avg` and 0.693 with `phase2_blend_2way_optimal`
(the LB 0.8965 anchor) — both above the pre-registered 0.6 diversity bar.
Stand-alone `trajectory_shape` underperformed (m-s 0.566-0.582 vs
`baseline_v1` 0.7143), and `baseline_plus_shape` (0.7066) was below
`baseline_v1` alone — the shape features actively hurt the baseline.

## Methodology-writeup language

Trajectory-shape features (per-variable monotonicity, smoothness, stability
period, recent-vs-baseline ratio, volatility, trend acceleration, presence)
were tested for diversity addition to the production ensemble. They were
found redundant with existing `longitudinal_summary` aggregations
(`_last`/`_max`/`_count`/`_std`/`_slope`): the two share OOF Spearman 0.754,
and combining shape features with `baseline_v1` did not improve over
`baseline_v1` alone. This demonstrates that the existing Phase 2 feature
engineering already extracts the geometric signal in NIT/lab trajectories;
explicit shape descriptors do not contribute additional information once
the standard summary statistics are present. The 0.85 single-feature
C-index leakage threshold was cleared with margin (top shape feature
0.630), so the negative result is not a leakage artifact.

## Files
- `reports/phase3_exp1_trajectory_shape.md` — full audit, bake-off, correlation matrix
- `reports/phase3_exp1_trajectory_shape.json` — machine-readable diagnostic
- `experiments/phase3_exp1_trajectory_shape.py` — script
- `src/features.py` — `SHAPE_VARS`, `_trajectory_shape_features`,
  `trajectory_shape` and `baseline_plus_shape` feature sets (kept for
  reproducibility; both tagged Tier 4 in `FEATURE_SET_RISK`).
