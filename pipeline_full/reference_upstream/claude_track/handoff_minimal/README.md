# ANNITA handoff_minimal — Anthropic-track team

Best public LB: **0.8965** (`phase2_blend_2way_optimal.csv`, 2026-04-28).

## Top 3 submissions above 0.88 LB

1. **phase2_blend_2way_optimal (LB 0.8965)** — OOF-stacked 30/70 rank blend of
   `landmark_3y_RSF` (Tier 2) and `permissive_ensemble_avg` (Tier 4, 3 members).
2. **phase2_blend_2way_strongpermissive (LB 0.8867)** — same skeleton, single permissive
   member (`rsf_longitudinal_summary`); CV-LB inversion #1.
3. **phase2_blend_landmark_permissive (LB 0.880)** — 50/50 equal-weight predecessor of
   the champion.

See `model_descriptions.md` and `ensemble_details.md` for the full story.

## OOF predictions

Yes, included for the top 4 hepatic / death components, in `optional_oof_predictions/`:

- `oof_landmark_3y_rsf.csv` — landmark_3y RSF, mean-of-ranks across 10 repeats
- `oof_permissive_ensemble_avg.csv` — Tier-4 permissive ensemble (3 members)
- `oof_blend_2way_optimal.csv` — the 30/70 champion blend
- `oof_death_xgb_longitudinal_summary.csv` — death predictor (XGB-Cox / longitudinal_summary)

All four use the same fold structure: 5-fold × 10-repeat StratifiedKFold on the event
indicator, `random_state = 42 + repeat`, `shuffle=True`. Each patient appears in exactly
one validation fold per repeat. Risks reported are mean ranks across repeats (so they
are comparable across files via `scipy.stats.spearmanr` or rank-blending).

## Caveats — please read before blending against your models

- **Submission count.** Ten submissions are logged here, not the seven mentioned in the
  request. We listed all of them in `submission_log.csv` so the comparison is complete.
- **CV-LB decoupling.** We observed three consecutive CV→LB inversions in late Phase 2
  / early Phase 3 (`strongpermissive`, `phase2_blend_3way`, `phase3_blend_with_crossfeatures`).
  The empirical LB noise floor at ρ_hep=1 is ~0.028, and the analytical SE_combined on a
  ~25–50 event public-LB cohort is 0.034–0.048. Treat any single LB delta below 0.03
  as noise. Plan blends accordingly.
- **Tier-5 leakage.** `phase2_strict_leaky_probe.csv` (LB 0.692) is included as
  documentation of an audit-only Tier-5 feature set (`strict_time_aligned`). It uses
  the per-row event/censoring time to define a per-row trajectory window, which leaks
  outcome through `Age_delta` (single-feature C ≈ 0.958). **Do not use for shipping.**
- **Death predictor is shared.** All Phase 2 submissions share the same death predictor
  (XGB-Cox on `longitudinal_summary`, `n_estimators=300, learning_rate=0.05`). Cross-
  submission LB deltas are entirely hepatic. If your team's death-side model is stronger,
  rank-blending with our hepatic side and your death side is a clean experiment.

## Files

- `submission_log.csv` — 10 submissions with feature sets, model families, ensemble method, LB
- `metrics_summary.csv` — OOF metrics per major standalone model + per submission
- `model_descriptions.md` — half-page each for the top 3 submissions above 0.88 LB
- `ensemble_details.md` — components, weights, OOF stacking procedure, blending convention
- `best_submissions/` — 5 representative submission CSVs
- `optional_oof_predictions/` — 4 OOF prediction CSVs (see above)
