# Component test predictions — convention notes

Files in this folder, all keyed by `trustii_id`:

| file | column | scale |
|---|---|---|
| `test_landmark_3y_rsf.csv` | `hepatic_risk` | RSF cumulative-hazard sum (raw) |
| `test_rsf_baseline_v1_tuned.csv` | `hepatic_risk` | RSF cumulative-hazard sum (raw) |
| `test_rsf_longitudinal_summary.csv` | `hepatic_risk` | RSF cumulative-hazard sum (raw) |
| `test_permissive_ensemble_avg.csv` | `hepatic_risk` | **mean of rankdata across 3 permissive members** |

Training recipe (matches the production submission):

- `test_landmark_3y_rsf.csv` — RSF tuned on `landmark_3y` features, trained on
  the at-risk-at-3y cohort (filter B, n=1238, events=32). This matches the
  landmark component inside `phase2_blend_2way_optimal.csv`.
- `test_rsf_baseline_v1_tuned.csv` — RSF tuned on `baseline_v1` (Tier 1),
  full train (n=1253, events=47). Clean Tier-1 anchor; not in any submission
  by itself.
- `test_rsf_longitudinal_summary.csv` — RSF tuned on `longitudinal_summary`
  (Tier 4), full train. Strongest individual permissive member.
- `test_permissive_ensemble_avg.csv` — rank-mean across 3 members:
  `rsf_longitudinal_summary` + `xgb_cox_longitudinal_plus_meta` +
  `xgb_cox_longitudinal_summary`. All three trained on full train. This
  is exactly the permissive arm inside `phase2_blend_2way_optimal.csv`.

Hyperparameters live in `configs/optuna_*.json` in the source repo; they
are copied verbatim into this script (`build_merge_handoff.py`, top of file).

Death-side test predictions are not included separately because every Phase 2
submission shares the same death predictor (XGB-Cox / `longitudinal_summary`,
`n_estimators=300, learning_rate=0.05`); see `ensemble_details.md` for why.
