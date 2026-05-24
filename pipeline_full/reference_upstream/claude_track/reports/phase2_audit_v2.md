# Phase 2 audit v2 — Tasks 1+2+3 

Date: 2026-04-28. Triggered by phase2_blend_2way_optimal LB 0.8965 (+0.016 over 50/50). All numbers below: 5×10 stratified CV (50 folds, base_seed=42), full hepatic-valid cohort (n=1253, events=47).

## Task 1 — finer-grain weight curve

User-asked weights (CV m-s on hepatic):
| w (landmark) | w (permissive) | mean | std | m-s |
|---|---|---|---|---|
| 0.10 | 0.90 | 0.8126 | 0.0904 | 0.7222 |
| 0.20 | 0.80 | 0.8132 | 0.0908 | 0.7225 |
| 0.40 | 0.60 | 0.8140 | 0.0938 | 0.7202 |

**Fine-grid optimum (0.01 step):** w* = 0.29, m-s = 0.7235 (mean 0.8147 ± 0.0912).

**Vs the 0.30 anchor that scored LB 0.8965:** |Δw| = 0.01, Δm-s = +0.0011.

→ BUILD _v2 (triggers: |Δw|≥0.05 or Δm-s≥0.0005).

## Task 2 — permissive component audit

**Per-member 5×10 hepatic m-s:**
| model | mean | std | m-s |
|---|---|---|---|
| `rsf_longitudinal_summary` | 0.8277 | 0.0921 | 0.7356 |
| `xgb_cox_longitudinal_plus_meta` | 0.7881 | 0.0959 | 0.6922 |
| `xgb_cox_longitudinal_summary` | 0.7881 | 0.0931 | 0.6949 |
| `permissive_ensemble_avg` | 0.8103 | 0.0906 | 0.7197 |
| `landmark_3y_RSF` | 0.7737 | 0.1025 | 0.6712 |

**Pairwise OOF rank-corr (filter-A intersection, n=1238):**

| | `landmark_3y_RSF` | `rsf_longitudinal_summary` | `xgb_cox_longitudinal_plus_meta` | `xgb_cox_longitudinal_summary` | `permissive_ensemble_avg` |
|---|---|---|---|---|---|
| `landmark_3y_RSF` | 1.000 | 0.448 | 0.275 | 0.242 | 0.344 |
| `rsf_longitudinal_summary` | 0.448 | 1.000 | 0.815 | 0.767 | 0.908 |
| `xgb_cox_longitudinal_plus_meta` | 0.275 | 0.815 | 1.000 | 0.970 | 0.975 |
| `xgb_cox_longitudinal_summary` | 0.242 | 0.767 | 0.970 | 1.000 | 0.956 |
| `permissive_ensemble_avg` | 0.344 | 0.908 | 0.975 | 0.956 | 1.000 |

**Test rank-corr vs landmark_3y_RSF (n=423):**
- `rsf_longitudinal_summary`: ρ = 0.548
- `xgb_cox_longitudinal_plus_meta`: ρ = 0.411
- `xgb_cox_longitudinal_summary`: ρ = 0.399
- `permissive_ensemble_avg`: ρ = 0.478

**Strongest individual permissive member**: `rsf_longitudinal_summary` (m-s 0.7356, test corr vs landmark 0.548).

→ SKIP _strongpermissive (triggers: strongest member's test corr vs landmark < 0.5).

## Task 3 — submissions built

- `phase2_blend_2way_optimal_v2.csv` — w_landmark=0.29, m-s=0.7235.
- _strongpermissive: not built (no single member with test corr vs landmark below 0.5).