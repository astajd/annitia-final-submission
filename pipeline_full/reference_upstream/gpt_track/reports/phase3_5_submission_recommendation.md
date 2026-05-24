# Phase 3.5 — submission recommendation

Public LB so far:
- `phase3_current_state_v2` = **0.88521** (current best, +0.014 from OOF 0.8713)
- `phase3_stacked_ensemble` = 0.78676 (overfit OOF 0.8767, transferred -0.090 — see [phase3_5_stack_failure_analysis.md](phase3_5_stack_failure_analysis.md))
- `phase2_aggressive_longitudinal` = 0.86813 (transferred from OOF 0.8819, -0.014)
- `phase2_robust_longitudinal` (untested on LB)

Phase 3.5 is a **refinement of `current_state_v2` only**. No stacking, no
strict_time_aligned, no target-derived features, no public-LB tuning.

## Candidate roll-up

| label                                     | leakage_tag    |   hep_oof |   death_oof |   weighted |   rho_oof_hep_vs_v2 |   rho_oof_dea_vs_v2 |   rho_test_hep_vs_v2 |   rho_test_dea_vs_v2 |
|:------------------------------------------|:---------------|----------:|------------:|-----------:|--------------------:|--------------------:|---------------------:|---------------------:|
| phase3_5_current_state_v3_simplified      | moderate-high  |    0.8378 |      0.9496 |     0.8713 |              1.0000 |              0.9989 |               1.0000 |               0.9992 |
| phase3_5_current_state_v3_bagged          | moderate-high  |    0.8378 |      0.9501 |     0.8715 |              0.9975 |              0.9990 |               0.9982 |               0.9990 |
| phase3_5_current_state_v3_hepatic_focused | moderate-high  |    0.8415 |      0.9496 |   **0.8739** |              0.9187 |              0.9989 |               0.9647 |               0.9992 |
| phase3_5_current_state_v3_private_robust  | moderate-high  |    0.8355 |      0.9405 |     0.8670 |              0.9906 |              0.9871 |               0.9953 |               0.9902 |

Reference: `phase3_current_state_v2` OOF hep=0.8378 / death=0.9495 / weighted=0.8713 | LB=0.88521.

`phase3_5_current_state_v3_simplified` and `..._bagged` are essentially the
**same prediction** as `phase3_current_state_v2` (test rank-corr ≥ 0.998).
Submitting them would burn an LB attempt for ~zero new information.

## Recommended submissions (at most two)

### 1. High-upside: `submissions/20260427_1255_phase3_5_current_state_v3_hepatic_focused.csv`

- **OOF hepatic C-index**: 0.8415 (+0.0037 over v2)
- **OOF death C-index**: 0.9496 (~unchanged)
- **OOF weighted score**: 0.8739 (+0.0026 over v2)
- **Rank correlation with v2 (test)**: hepatic 0.965, death 0.999
- **Components**:
  - hepatic: greedy ensemble of v2 RSF (s0/s1) + xgb_aft + a freshly-trained
    RSF on `current_state_v2` *without* visit-history features
    (`n_visits`, `age_first/last_visit`, `followup_span`, `gap_*`,
    `period_*`). The no-visit-history retrain is the source of the new
    signal — ablation E showed it standalone reaches 0.8415 hepatic OOF
    while keeping its rank correlation with the original v2 hepatic at 0.92.
  - death: seed-bagged ensemble on the *simplified* death pool (binary
    classifiers dropped). Identical OOF to v2 death (0.9496) with tighter
    fold variance (std 0.011 vs 0.012, fold_min 0.929 vs 0.926).
- **Why submit**: this is the only Phase 3.5 candidate that materially moves
  the prediction (rank-corr 0.96 hep) while improving local OOF on the
  priority endpoint without any cost on death. The hepatic improvement
  comes from genuinely different evidence (biomarker trajectories instead
  of visit cadence), so the LB delta tells us whether the +0.014 transfer
  premium of v2 was driven by visit cadence or by biomarker structure.
- **Expected interpretation**:
  - If LB > 0.88521: the no-visit-history hepatic RSF transfers, biomarker
    trajectories are the primary signal source. We'd then iterate by
    growing the no-visit-history component family (xgb_cox, xgb_aft on the
    trimmed feature set).
  - If LB ≈ 0.88521 (within ±0.005): hepatic gain is local-only. Stay on
    v2 and look elsewhere.
  - If LB < 0.88521: v2's hepatic gain depends on visit-history features
    after all. Drop the hepatic-focused direction and lean private_robust.

### 2. Private-robust: `submissions/20260427_1255_phase3_5_current_state_v3_private_robust.csv`

- **OOF hepatic C-index**: 0.8355 (-0.0023 vs v2)
- **OOF death C-index**: 0.9405 (-0.0090 vs v2)
- **OOF weighted score**: 0.8670 (-0.0043 vs v2)
- **Rank correlation with v2 (test)**: hepatic 0.995, death 0.990
- **Components**: rank-space blend `0.7 * v2_per_endpoint_ensemble + 0.3 *
  phase2_aggressive_longitudinal_per_endpoint_ensemble`. The 70/30 split
  was chosen as a round prior, not tuned to OOF or LB.
  - v2 component: `current_state_v2` greedy hepatic + seed_bagged death.
  - phase2 component: greedy on the previously-LB-validated
    `phase2_aggressive_longitudinal` pool (LB 0.86813).
- **Why submit**: hedges against `phase3_current_state_v2` being unusually
  lucky on public test. The two halves of the blend transferred to LB
  0.885 and LB 0.868 respectively, so a 70/30 blend should land near
  0.88 on a public test that respects the OOF ranking and noticeably
  closer to 0.87 if v2 was overfitting public test. Useful primarily as
  a **private-LB insurance policy** rather than a public-LB push.
- **Expected interpretation**:
  - If LB ≈ 0.88: we are roughly on the OOF prior; v2's public LB was not
    inflated, hepatic_focused remains the high-upside direction.
  - If LB ≈ 0.87 or below: v2 was getting public-test luck; the private
    LB will likely be closer to phase2_aggressive's true rank. Submit
    private_robust as the final submission instead of v2.
  - If LB > 0.88: surprising — a hedged blend rarely beats its anchor on
    public LB. Treat as noise and ignore.

## Submission order

1. Submit **hepatic_focused** first. The information it produces (whether
   the no-visit-history hepatic improvement is real on the LB) is what
   shapes the rest of Phase 3.5 / Phase 4.
2. Hold **private_robust** unless step 1 returns an LB ≤ 0.880, at which
   point use it as a safer final submission.

## What we deliberately did NOT do

- No new stacking experiments, no new meta-models. The
  [stack failure analysis](phase3_5_stack_failure_analysis.md) makes it
  clear stacking with our current base pool overfits.
- No `strict_time_aligned` re-introduction; LB 0.69 confirmed it does
  not transfer.
- No event-age / death-age features anywhere. Verified by the
  [decomposition report](phase3_5_current_state_v2_decomposition.md):
  every top-feature group is biology / care-cadence / missingness, not
  outcome timing.
- No tuning of the 70/30 private-robust blend ratio against the public LB.
  We picked it once, before submitting.

## Phase 4 hooks (do not implement until Phase 3.5 LB results are in)

- If hepatic_focused improves: build `current_state_v3_no_cadence` as a
  full feature set (drop visit-history group entirely) and run the same
  multi-seed model sweep on it.
- If hepatic_focused stalls: explore stronger NIT-only and biomarker-
  trajectory feature sets with the same model multi-seed treatment.
- Either way, revisit the discrete-time hazard with sub-yearly bins; the
  current 1-year bin missed too many hepatic events and produced an
  unhelpful 0.55 OOF.
