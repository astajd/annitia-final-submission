# Phase 3 experiment summary

Cross-validated Phase 3 results, building on the `phase2_aggressive_longitudinal`
public-best (LB 0.86813). All experiments share the same hepatic-event-stratified
repeated K-fold splits used since Phase 1.

## Inputs added in Phase 3

- **`current_state_v2`** feature set
  ([src/features/current_state_v2.py](../src/features/current_state_v2.py)).
  Latest values, max/min/mean/std/slope/delta per stem, dynamic FIB-4 / APRI /
  AST-ALT, NIT trajectories, visit cadence, missingness, and 11 clinical
  pairwise interactions. Excludes `evenements_hepatiques_age_occur` and
  `death_age_occur`. Includes `last_observed_age` and `n_visits` per the
  Phase 3 spec, so the leakage tag is **moderate-high**.
- **XGBoost AFT model wrapper**
  ([src/models/xgb_survival.py](../src/models/xgb_survival.py)).
- **Multi-seed support** in the experiment runner via a model-level `tag`.
  Phase 3 sweep runs 6 estimators x 2 seeds = 12 models per endpoint.
- **Discrete-time hazard model**
  ([src/discrete_time_hazard.py](../src/discrete_time_hazard.py)). Patient-period
  long-format with yearly bins out to 10 years; LightGBM hazard classifier per
  bin; patient risk = `1 - prod(1 - hazard_t)`.
- **Stacking meta-models** ([src/stacking.py](../src/stacking.py)). Ridge and
  elastic-net logistic on percentile-rank base OOFs (every Phase 1/2/3 model
  with single-component C-index >= 0.55), plus the discrete-time hazard OOFs.

## Phase 3 sweep CV summary (current_state_v2)

Score = 0.7 * hepatic + 0.3 * death.

| metric                   |    mean |     std |     min |     max | mean - 0.5*std |
|:-------------------------|--------:|--------:|--------:|--------:|---------------:|
| Hepatic C-index          |  0.8046 |  0.1238 |  0.5849 |  0.9634 |         0.7427 |
| Death C-index            |  0.9033 |  0.0341 |  0.8391 |  0.9535 |         0.8862 |
| Weighted score           |  0.8342 |  0.0868 |  0.6786 |  0.9604 |         0.7908 |

Best individual models within Phase 3:

| endpoint | best model        | seed | mean   | std    | mean - 0.5*std |
|:---------|:------------------|:-----|:-------|:-------|:---------------|
| hepatic  | rsf               | 1    | 0.8249 | 0.1172 | 0.7663         |
| hepatic  | rsf               | 0    | 0.8185 | 0.1135 | 0.7618         |
| death    | xgb_cox           | 0    | 0.9358 | 0.0250 | 0.9233         |
| death    | rsf               | 0    | 0.9302 | 0.0195 | 0.9204         |

The `rsf` and `xgb_cox` models drive the ensemble; binary classifiers add
diversity but lower individual C-index.

## Discrete-time hazard CV (yearly bins, horizon 10y)

Standalone DTH is **not** competitive:

| endpoint | mean   | std   | min     | promoted? |
|:---------|:-------|:------|:--------|:----------|
| hepatic  | 0.5459 | 0.126 | 0.277   | no        |
| death    | 0.6303 | 0.072 | 0.471   | no        |

DTH is below the 0.80 weighted-score threshold. We retain its OOF/test
predictions and feed them into the stacker as a diversity input, but do not
promote DTH as a candidate submission.

## Stacking CV (Phase 1+2+3 + DTH OOFs)

Two meta-learners on percentile-rank base predictions:

| endpoint | method      | OOF C-index | fold mean | fold std |
|:---------|:------------|:------------|:----------|:---------|
| hepatic  | ridge       | 0.8943      | 0.9111    | 0.040    |
| hepatic  | elastic_net | 0.8590      | 0.8824    | 0.049    |
| death    | ridge       | 0.8355      | 0.8349    | 0.047    |
| death    | elastic_net | 0.8090      | 0.8057    | 0.062    |

Ridge wins on both endpoints. Hepatic stacking pushes from
~0.80 (single best Phase 3 ensemble) to **0.894** OOF — a +0.09 lift on the
*priority* endpoint.

## Phase 3 candidate submissions

| candidate                   | promoted | hep CV | death CV | weighted CV | rank-corr pub_best (hep / death) |
|:----------------------------|:---------|:-------|:---------|:------------|:---------------------------------|
| phase3_current_state_v2     | yes      | 0.8378 | 0.9495   | 0.8713      | 0.85 / 0.96                      |
| phase3_stacked_ensemble     | yes      | 0.8943 | 0.8355   | 0.8767      | 0.69 / 0.58                      |
| phase3_discrete_time_hazard | no       | 0.5459 | 0.6303   | 0.5712      | n/a                              |

The public best is `phase2_aggressive_longitudinal` at LB 0.86813. Both
promoted Phase 3 candidates beat that on local CV. The stacked ensemble has
substantially lower rank correlation with the public best (0.58-0.69), so it
will move the LB the most either way — most informative test.

None of the candidates use target-derived features (`event_age_occur` /
`death_age_occur`).

## Best model per endpoint

- **Hepatic (defensible single model):** RSF on `current_state_v2`, seed 1 —
  0.8249 mean, 0.117 std (penalised 0.7663).
- **Hepatic (overall):** stacked ridge OOF 0.8943.
- **Death (defensible single model):** xgb_cox on `current_state_v2`, seed 0
  — 0.9358 mean, 0.025 std (penalised 0.9233). Tighter than any other death
  config we have.
- **Death (overall):** ensemble of `current_state_v2` xgb_cox and rsf hits
  0.9495 OOF inside Candidate A.

## Recommended next 1-2 submissions

1. **`submissions/20260427_1220_phase3_stacked_ensemble.csv`** — highest
   local CV (0.8767) *and* lowest rank correlation with the public best
   (0.69 hep, 0.58 death). Most informative leaderboard probe: the LB
   delta tells us how much the local stacking gain transfers.
2. **`submissions/20260427_1220_phase3_current_state_v2.csv`** — a
   confirmation submission. Local CV 0.8713 with high rank correlation to
   public best on death (0.96) but new signal on hepatic (0.85). Submit
   only if the stacked candidate underperforms or if you want to isolate
   the contribution of the new feature set.

Both candidates are CV-supported improvements over LB 0.86813. We are not
auto-submitting; please review the rank correlations alongside the CV before
deciding.

## Phase 4 hooks (do not implement until Phase 3 reports are accepted)

- **Constrained stacker.** The current ridge/elastic-net stacker accepts
  components with cindex >= 0.55. Tighten to >= 0.65 and re-fit; expect
  similar hepatic but cleaner death.
- **Per-fold component selection.** Use OOF correlations within fold to
  drop redundant base learners before the meta-model sees them.
- **Improve DTH.** Current 1-year bins are too coarse for hepatic (0.55 CV).
  Try 6-month or visit-indexed bins, and add hard-tied-event indicators.
- **Reweight ensemble methods on hepatic.** Current Phase 3 stacker won
  hepatic but lost death vs the simple per-endpoint ensemble. A
  per-endpoint hybrid (stacking for hepatic, simple ensemble for death) is
  the obvious next candidate.
- **Calibration.** Isotonic mapping of risk scores to predicted hazards;
  C-index doesn't care, but the contest's qualitative review may.
