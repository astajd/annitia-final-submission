# Phase 2d — Landmark analysis (Tier 2 features)

Date: 2026-04-27. Wall-clock: 5.7 min.

Features at each landmark = `baseline_v1` + LOCF + slope-since-v1 for each longitudinal variable. Tier 2 (clean longitudinal): cutoff is fixed by the calendar, not by the outcome. Training cohort drops patients whose hepatic event preceded the landmark (not-at-risk).

Anchors (Experiment 2 tuned, baseline_v1 only): RSF m−s = **0.7196**, XGB-Cox m−s = **0.7068**.

CV: 5-fold × 10-repeat stratified on hepatic event. Hyperparameters fixed at Experiment 2's tuned baseline_v1 winners (not re-tuned here).


## Results

| Landmark | Model | n / events | n_features | mean | std | m−s | Δm−s vs baseline_v1 |
|---|---|---|---|---|---|---|---|
| 1.0y | rsf | 1245 / 39 | 49 | 0.8437 | 0.0947 | 0.7490 | +0.0294 |
| 1.0y | xgb_cox | 1245 / 39 | 49 | 0.7976 | 0.1016 | 0.6960 | -0.0108 |
| 2.0y | rsf | 1241 / 35 | 49 | 0.8269 | 0.1159 | 0.7109 | -0.0087 |
| 2.0y | xgb_cox | 1241 / 35 | 49 | 0.7678 | 0.1143 | 0.6536 | -0.0532 |
| 3.0y | rsf | 1238 / 32 | 49 | 0.8799 | 0.0648 | 0.8152 | +0.0956 |
| 3.0y | xgb_cox | 1238 / 32 | 49 | 0.8297 | 0.0805 | 0.7492 | +0.0424 |
| 5.0y | rsf | 1229 / 23 | 49 | 0.8149 | 0.1118 | 0.7031 | -0.0165 |
| 5.0y | xgb_cox | 1229 / 23 | 49 | 0.8024 | 0.1311 | 0.6713 | -0.0355 |

**Best landmark/model:** `landmark_3y` × rsf at m−s 0.8152 (mean 0.8799, std 0.0648).

→ Landmark beats baseline_v1 anchor by +0.0956 m−s. Real Tier-2 longitudinal signal exists at this horizon.

## Files

- Rows appended to `reports/phase2_cv_results.csv` (feature_set=`landmark_<Y>y`, model=`{rsf,xgb_cox}_landmark`).
- Builder: `src/features.py::build_landmark_features` + `at_risk_at_landmark`.