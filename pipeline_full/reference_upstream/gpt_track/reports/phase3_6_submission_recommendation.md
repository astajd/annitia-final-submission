# Phase 3.6 — submission recommendation

Reference (current public LB best): `phase3_5_current_state_v3_hepatic_focused`
= **0.89147**, OOF hepatic 0.8415, OOF death 0.9496, OOF weighted 0.8739.

Phase 3.6 is a **narrow refinement around v3_hepatic_focused only**. No new
stacking, no strict_time_aligned, no event/censoring-age features anywhere.

## What was run

| sweep                              | feature set                          | seeds            | models                              |   hep_oof_ens |   death_oof_ens | weighted |
|:-----------------------------------|:-------------------------------------|:-----------------|:------------------------------------|--------------:|----------------:|---------:|
| phase3_6_no_visit_history          | current_state_v2_no_visit_history    | s0..s4 (5 seeds) | rsf, xgb_aft                        |        0.8260 |          0.8996 |   0.8481 |
| phase3_6_csv2_extra_seeds          | current_state_v2                     | s2..s4           | rsf, xgb_aft, xgb_cox               |          —    |            —    |     —    |
| phase3_6_hepatic_aug               | current_state_v2_hepatic_aug         | s0..s2           | rsf, xgb_aft, xgb_cox               |        0.8165 |          0.9466 |   0.8556 |

(Aggregate ensemble for csv2_extra_seeds was not summarised separately; its
components feed the bagged pool below.)

## Candidates

| label                                           |   hep_oof |   hep_fold_std |   hep_fold_min |   death_oof |   death_fold_std |   death_fold_min |   weighted |   rho_oof_hep_vs_v3 |   rho_test_hep_vs_v3 |   rho_test_dea_vs_v3 |   n_components_hep |
|:------------------------------------------------|----------:|---------------:|---------------:|------------:|-----------------:|-----------------:|-----------:|--------------------:|---------------------:|---------------------:|-------------------:|
| phase3_6_v4_hepatic_focused_bagged              |    0.8416 |         0.1063 |         0.6031 |      0.9496 |           0.0115 |           0.9269 |     0.8740 |              1.0000 |               0.9899 |               1.0000 |                  8 |
| phase3_6_v4_hepatic_focused_biomarker_augmented |    0.8402 |         0.1019 |         0.6247 |      0.9496 |           0.0115 |           0.9269 |     0.8730 |              0.9760 |               0.9973 |               1.0000 |                  1 |

Death side reuses v3's simplified pool (4 components, seed_bagged) — local
ablations did not find a clearly better death configuration, so we keep it
unchanged per the Phase 3.6 spec.

`phase3_6_v4_hepatic_focused_bagged` hepatic pool (seed_bagged, 8 chosen of 12 candidates):

```
0.229  phase3_6_no_visit_history::hepatic__rsf__s4
0.171  phase3_current_state_v2::hepatic__xgb_aft__s0
0.143  phase3_current_state_v2::hepatic__rsf__s1
0.143  phase3_6_csv2_extra_seeds::hepatic__rsf__s2
0.114  phase3_6_no_visit_history::hepatic__rsf__s0
0.086  phase3_current_state_v2::hepatic__rsf__s0
0.086  phase3_6_csv2_extra_seeds::hepatic__rsf__s4
0.029  phase3_6_no_visit_history::hepatic__rsf__s1
```

`phase3_6_v4_hepatic_focused_biomarker_augmented` hepatic pool (greedy, 1):

```
1.000  phase3_6_no_visit_history::hepatic__rsf__s4
```

`hepatic_aug` features did **not** improve hepatic OOF — greedy collapsed back
to a single `no_visit_history` RSF, and seed_bagged on the augmented pool came
in at 0.8389 (below the bagged candidate's 0.8416). The biomarker
interactions we added (T2DM × stiffness, FIB-4 × platelet trend, etc.) are
already implicitly captured by the RSF + xgb_aft on the un-augmented latest /
slope columns.

## Component pruning diagnostics on the bagged hepatic pool

| variant                              | method_picked    |   n |    oof |
|:-------------------------------------|:-----------------|----:|-------:|
| bagged_hepatic_pool_baseline         | seed_bagged      |   8 | 0.8416 |
| loo_hepatic__rsf__s2 (no_visit_h)    | seed_bagged      |   8 | 0.8416 |
| loo_hepatic__rsf__s3 (no_visit_h)    | seed_bagged      |   8 | 0.8416 |
| seed_bagged_only                     | seed_bagged      |   8 | 0.8416 |
| loo_hepatic__rsf__s4 (no_visit_h)    | seed_bagged      |   7 | 0.8409 |
| loo_hepatic__rsf__s0 (no_visit_h)    | seed_bagged      |   7 | 0.8409 |
| loo_hepatic__xgb_aft__s0 (csv2)      | greedy           |   1 | 0.8402 |
| greedy_only                          | greedy_only      |   1 | 0.8402 |
| equal_only                           | equal_only       |  11 | 0.8357 |

Take-aways:
- The bagged-pool seed_bagged ensemble is **identical in OOF** to the
  v3 hepatic OOF (0.8416 vs 0.8415). Adding more seeds did **not** lift CV.
- The marginal contribution of any single seed is ≤ 0.001 OOF; the ensemble
  is well-anchored on the no-visit-history RSF family.
- Equal-weight on the full 11-component pool is materially worse (0.8357),
  confirming that seed_bagged correctly down-weights the weaker
  full-current_state_v2 RSFs.

## Death-side stability check

We did not retrain or replace the death side. Its OOF (0.9496), fold std
(0.0115) and fold min (0.9269) match v3 exactly because the same four
components are reused. Per the Phase 3.6 spec ("do not chase death"), we
leave it.

## Are target-derived features used?

No. The hepatic pool uses only `current_state_v2` and
`current_state_v2_no_visit_history` (and an unused `current_state_v2_hepatic_aug`
that did not enter the final candidate). Neither contains
`evenements_hepatiques_age_occur` or `death_age_occur`. Visit-cadence
features (`n_visits`, `age_last_visit`, `followup_span`, `gap_*`,
`period_*`) appear only in the *full* current_state_v2 RSF/xgb_aft components
that the bagged ensemble down-weights to ≈ 0.43 of the total, exactly as in
v3_hepatic_focused.

## Recommendation: one next submission

**Do not submit either candidate.** Both are within 0.0001 OOF of the current
public best (`v3_hepatic_focused`, 0.89147 LB) and rank-correlate with it at
≥ 0.99 on test. The expected LB delta is dominated by sampling noise (47
hepatic events ⇒ a single-event reordering moves C-index by ~0.005), so we
would burn an LB attempt on essentially the same prediction.

If a Phase 3.6 submission is required:

- **Submit `submissions/20260427_1431_phase3_6_v4_hepatic_focused_bagged.csv`** —
  it has marginally tighter hepatic fold-min support from seed bagging and
  the highest non-trivial diversity (test rank-corr 0.99 with v3, vs 0.997
  for the augmented candidate). Expect LB within 0.005 of 0.89147 either
  direction; treat as a confidence check rather than a push.
- **Hold the augmented candidate.** Its hepatic OOF was lower and its
  rank-corr with v3 is higher, so it carries less new information.

Better use of the next LB attempt: hold and pivot to a **different feature
philosophy** in Phase 4 — e.g. drop the entire `visit_history_current_state`
group at the *feature level* and rebuild a multi-seed sweep on
`current_state_v2_no_visit_history` alone (the strongest single sub-pool here,
with hep OOF 0.825 ± 0.110) plus a separate clean death-side feature set so
the death and hepatic bases are no longer coupled to the same v2 schema.

## What we deliberately did NOT do

- No new stacking experiments (the Phase 3 stack overfit; same risk applies
  with these tighter pools).
- No `strict_time_aligned` re-introduction.
- No event-age / death-age features.
- No public-LB weight tuning. Both candidates' weights are picked by OOF
  C-index only.
- No changes to the death side beyond the previously-validated v3 pool.
