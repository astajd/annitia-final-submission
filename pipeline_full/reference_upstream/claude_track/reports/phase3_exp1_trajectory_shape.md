# Phase 3 Experiment 1 — Trajectory-shape features

Date: 2026-04-28. CV: 5-fold × 10-repeat stratified on hepatic event (50 folds, base_seed=42). Cohort: n=1253, events=47.

Feature set: per-variable shape summaries over **11 longitudinal vars** (`BMI, alt, ast, ggt, plt, gluc_fast, triglyc, chol, fibs_stiffness_med_BM_1, fibrotest_BM_2, aixp_aix_result_BM_3`). 7 features per variable: `present, monotonicity, smoothness, stability, recent_ratio, volatility, acceleration` → **77** trajectory-shape columns. Tier 4 (uses all visits; designed to be robust to post-event ambiguity).

## 1. Single-feature C-index audit

Audited 77 shape features. Best-sign C-index per feature (positive vs negative orientation, max).

**Leakage threshold:** C > 0.85. **Flagged:** 0.

Top 15 by best-sign C:
| feature | c_pos | c_neg | c_best | best_sign | leak_flag |
|---|---|---|---|---|---|
| `fibs_stiffness_med_BM_1_shape_present` | 0.370 | 0.630 | **0.630** | - |  |
| `triglyc_shape_present` | 0.394 | 0.606 | **0.606** | - |  |
| `gluc_fast_shape_present` | 0.395 | 0.605 | **0.605** | - |  |
| `plt_shape_present` | 0.395 | 0.605 | **0.605** | - |  |
| `BMI_shape_recent_ratio` | 0.399 | 0.601 | **0.601** | - |  |
| `ggt_shape_present` | 0.401 | 0.599 | **0.599** | - |  |
| `ast_shape_volatility` | 0.592 | 0.408 | **0.592** | + |  |
| `ast_shape_present` | 0.409 | 0.591 | **0.591** | - |  |
| `chol_shape_volatility` | 0.591 | 0.409 | **0.591** | + |  |
| `chol_shape_present` | 0.415 | 0.585 | **0.585** | - |  |
| `alt_shape_monotonicity` | 0.582 | 0.418 | **0.582** | + |  |
| `alt_shape_present` | 0.420 | 0.580 | **0.580** | - |  |
| `BMI_shape_present` | 0.421 | 0.579 | **0.579** | - |  |
| `fibrotest_BM_2_shape_present` | 0.431 | 0.569 | **0.569** | - |  |
| `alt_shape_acceleration` | 0.435 | 0.565 | **0.565** | - |  |

## 2. CV bake-off (5×10, hepatic m-s)

| model | feature_set | n_feat | mean | std | m-s | elapsed |
|---|---|---|---|---|---|---|
| rsf | `baseline_plus_shape` | 100 | 0.7826 | 0.0759 | **0.7066** | 24s |
| xgb_cox | `baseline_plus_shape` | 100 | 0.7618 | 0.0701 | **0.6917** | 38s |
| lgbm_bin | `baseline_plus_shape` | 100 | 0.7164 | 0.0915 | **0.6250** | 30s |
| xgb_cox | `trajectory_shape` | 84 | 0.6774 | 0.0949 | **0.5824** | 28s |
| lgbm_bin | `trajectory_shape` | 84 | 0.6705 | 0.1045 | **0.5659** | 28s |
| rsf | `trajectory_shape` | 84 | 0.6712 | 0.1057 | **0.5655** | 26s |

Reference (from `phase2_cv_results.csv`):

| feature_set | model | mean | std | m-s |
|---|---|---|---|---|
| `longitudinal_summary` | rsf | 0.8127 | 0.0960 | 0.7167 |
| `baseline_v1` | rsf | 0.8011 | 0.0868 | 0.7143 |
| `baseline_v1` | xgb_cox | 0.7517 | 0.0949 | 0.6568 |
| `longitudinal_summary` | xgb_cox | 0.7589 | 0.1043 | 0.6546 |

## 3. OOF rank-correlation vs Phase 2 components

Best new model: **rsf × `baseline_plus_shape`** (m-s 0.7066). OOF stitched by rank-averaging across the 50 folds and Spearman vs the cached Phase 2 OOF.

| comparison | Spearman ρ |
|---|---|
| vs landmark 3y RSF | **0.208** |
| vs permissive ensemble avg | **0.754** |
| vs blend 2way optimal | **0.693** |

**Decision rule:** corr < 0.6 vs all three → qualifies as ensemble member.

→ **DOES NOT QUALIFY** as a new ensemble member.
  (corr below threshold only vs: ['vs_landmark_3y_RSF'])

## Verdict / next steps

- Trajectory-shape's best model does NOT clear the 0.6 corr bar vs all three Phase 2 components. Adding it to the ensemble would be redundant with what we already have. Stops here per the pre-registered rule.

*No submissions built. Ends here per spec.*
