# OOF rank-correlation matrix — Phase 2 candidates

Date: 2026-04-27. CV: 5-fold × 10-repeat (50 folds), OOF averaged across repeats. Correlation computed on filter-A intersection (n=1238, events=32).

Threshold: rank-corr < 0.6 with BOTH `landmark_3y_RSF` AND `permissive_ensemble_avg` → diverse third blend candidate.


## 8×8 Spearman correlation matrix

| | `landmark_3y_RSF` | `permissive_ensemble_avg` | `tuned_rsf_baseline_v1` | `tuned_xgb_cox_baseline_v1` | `rsf_nit_only_baseline_only` | `catboost_bin_early_v1_v3` | `rsf_longitudinal_summary` | `xgb_cox_longitudinal_plus_meta` |
|---|---|---|---|---|---|---|---|---|
| `landmark_3y_RSF` | 1.000 | 0.345 | 0.725 | 0.492 | 0.541 | 0.117 | 0.443 | 0.277 |
| `permissive_ensemble_avg` | 0.345 | 1.000 | 0.429 | 0.433 | 0.382 | 0.651 | 0.908 | 0.975 |
| `tuned_rsf_baseline_v1` | 0.725 | 0.429 | 1.000 | 0.734 | 0.664 | 0.294 | 0.448 | 0.392 |
| `tuned_xgb_cox_baseline_v1` | 0.492 | 0.433 | 0.734 | 1.000 | 0.593 | 0.349 | 0.360 | 0.429 |
| `rsf_nit_only_baseline_only` | 0.541 | 0.382 | 0.664 | 0.593 | 1.000 | 0.298 | 0.325 | 0.377 |
| `catboost_bin_early_v1_v3` | 0.117 | 0.651 | 0.294 | 0.349 | 0.298 | 1.000 | 0.557 | 0.656 |
| `rsf_longitudinal_summary` | 0.443 | 0.908 | 0.448 | 0.360 | 0.325 | 0.557 | 1.000 | 0.815 |
| `xgb_cox_longitudinal_plus_meta` | 0.277 | 0.975 | 0.392 | 0.429 | 0.377 | 0.656 | 0.815 | 1.000 |

## Diversity vs both anchors

| candidate | corr vs `landmark_3y_RSF` | corr vs `permissive_ensemble_avg` | diverse? |
|---|---|---|---|
| `tuned_rsf_baseline_v1` | 0.725 | 0.429 | no |
| `tuned_xgb_cox_baseline_v1` | 0.492 | 0.433 | **yes** |
| `rsf_nit_only_baseline_only` | 0.541 | 0.382 | **yes** |
| `catboost_bin_early_v1_v3` | 0.117 | 0.651 | no |
| `rsf_longitudinal_summary` | 0.443 | 0.908 | no |
| `xgb_cox_longitudinal_plus_meta` | 0.277 | 0.975 | no |

## Result

**2 candidate(s) clear the threshold.** Top pick by lowest max-corr: **`tuned_xgb_cox_baseline_v1`** (landmark 0.492, permissive 0.433).
- Also: `rsf_nit_only_baseline_only` (landmark 0.541, permissive 0.382)