# Methodology sensitivity analysis

Date: 2026-04-28. CV: 5-fold × 10-repeat stratified on hepatic event (50 folds, base_seed=42). Cohort: n=1253, events=47.

Builds the same 30/70 blend recipe at three tier choices to show how methodology liberality affects CV performance:

- **Tier 1 (honest, single fixed timestamp)** — features computed at v1 only; baseline_v1.
- **Tier 2 (clean longitudinal, fixed reference time)** — features computed at Age_v1 + 2.0y (LOCF + slope per longitudinal var). Patients with hepatic event before the landmark dropped from training (filter A); predictions made on full validation fold.
- **Tier 4 (full-permissive, all visits)** — current production blend: 30% landmark_3y_RSF (filter A) + 70% rank-avg of three longitudinal models on `longitudinal_summary` and `longitudinal_plus_meta`. This is the actual `phase2_blend_2way_optimal` (LB 0.8965).

## Results

| Tier | n_features | RSF mean | XGB mean | Blend (30/70) mean | Blend std | Blend m−s | LB-implied (CV+0.07) |
|---|---|---|---|---|---|---|---|
| **Tier 1** v1-only | 23 | 0.8057 | 0.7907 | 0.8055 | 0.0854 | **0.7201** | 0.9009 |
| **Tier 2** landmark_2y (filter A n=1241) | 49 | 0.7760 | 0.7272 | 0.7472 | 0.0999 | **0.6473** | 0.8600 |
| **Tier 4** current (lm3 + permissive ensemble) | n/a (heterogeneous components) | 0.7737 (lm3) | 0.8103 (perm) | 0.8147 | 0.0913 | **0.7234** | 0.9073 (observed LB **0.8965**) |

*LB-implied uses 0.7·(CV_hep_mean + 0.07) + 0.3·0.96. The +0.07 correction was empirically derived from phase2_blend_2way_optimal (CV mean 0.8147 → hepatic LB ≈ 0.87). The LB-noise analysis (`phase3_lb_decoupling.md`) shows ±0.02-0.04 jitter on top of any point estimate.*


## Methodology narrative

Tier 1 uses only baseline-visit features (gender, T2DM, FibroScan_v1, etc.) — fully honest at the cost of discarding follow-up information. Tier 2 uses LOCF and slope at a fixed clinical reference time (Age_v1 + 2.0y); patients whose hepatic event preceded the landmark are excluded from training to avoid outcome-conditional feature construction. Tier 4 uses all visit history regardless of timing, which the organizer's forum post explicitly blesses for the quantitative track. The 30/70 blend recipe (one risk-stable model + one diverse-feature model, OOF-stacked weights from Phase 2) is held constant across all three tiers.

The Tier 4 production blend has structural diversity that Tier 1 and Tier 2 lack: its two components (`landmark_3y_RSF` and the `permissive_ensemble`) use *different feature sets* (Tier 2 vs Tier 4 features). Tier 1 and Tier 2 in this comparison use a single feature set per row, so the two component models share inputs and are highly correlated — the blend gain over the better single model is therefore smaller. This is the load-bearing methodological observation: blending helps when components are diverse in *both* model class AND feature view, not just in model class.

## Caveats

- Hyperparameters at each tier come from the Phase 2 Optuna tuning of `rsf_baseline_v1` and `xgb_cox_baseline_v1`. The Tier 4 components use their own tuned hyperparameters (landmark_3y RSF reuses baseline_v1 RSF params; permissive ensemble members are individually tuned).
- `LB-implied` is back-of-envelope. CV→LB has been noisy in this competition (three CV→LB inversions among the last four submissions). Use the m-s column for intra-tier ranking; the LB-implied column is illustrative.
- The Tier 2 numbers here use 2y landmark; the actual production blend uses 3y landmark. The 3y horizon was chosen in Phase 2d after a sweep over {1,2,3,5}y showed 3y had the best m-s (0.880) on filter A.