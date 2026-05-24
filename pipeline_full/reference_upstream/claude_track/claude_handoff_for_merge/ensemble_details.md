# Ensemble details

## Rank vs raw blending

All Phase 2 ensembles blend **ranks**, not raw risks. C-index depends only on the
relative ordering of risks, so rank-blending across heterogeneous risk-magnitude
distributions (RSF risks ∈ [0, 1] vs XGB-Cox log-hazards ∈ ℝ) is the principled
aggregation. Concretely:

```python
from scipy.stats import rankdata
risk_blend = w * rankdata(risk_landmark) + (1 - w) * rankdata(risk_permissive_avg)
```

The same convention is used inside the permissive ensemble (`mean(rankdata(member_i))`
across 3 members) and across CV repeats (`mean(rankdata(per_repeat_risk_i))` across
10 repeats; see OOF stitching in `experiments/build_handoff_package.py`).

## Endpoint-specific blending

Hepatic and death are blended **separately**. Submission columns are
`risk_hepatic_event` and `risk_death`; the LB scorer applies `0.7·C_hepatic + 0.3·C_death`.
We never mix endpoints inside an ensemble. The death predictor is identical across all
Phase 2 submissions (XGB-Cox on `longitudinal_summary`, `n_estimators=300, learning_rate=0.05`),
so all Phase 2 LB deltas are attributable to hepatic.

We did not OOF-tune the death side — Phase 1 had already shown the death endpoint
saturates near 0.95 (driven by `followup_yrs` alone hitting C=0.969) and 95% of remaining
gain is on hepatic.

## OOF stacking vs heuristic

The 30/70 weight in the champion is **OOF-stacked**, not picked by hand. Procedure
(`experiments/phase2_stack_2way.py`):

1. Run 5×10 stratified-by-event CV. For each fold, predict on the validation set with
   each of the 4 sub-models (1 landmark + 3 permissive members).
2. Stitch into per-repeat OOF arrays (each patient appears in exactly one validation
   fold per repeat → 1253 OOF risks per repeat per sub-model).
3. Form `permissive_avg = mean(rank-of-3-members)` per repeat.
4. Sweep `w ∈ {0.0, 0.05, …, 1.0}`. For each `w`, compute hepatic C-index per repeat
   on `w·rank(landmark) + (1-w)·rank(permissive_avg)`. Pick `w` maximizing
   mean−std across the 10 repeats.
5. Apply that `w` to test-set predictions: train each sub-model on full train, score test,
   rank, blend with the same `w`.

Optimum `w_landmark = 0.30`. LOFO sensitivity (drop one of the 50 fold-repeats at a time
and rerun the sweep) kept the argmax in `{0.30, 0.35, 0.40}` — the minimum is stable.
The CV mean−std improvement from 50/50 → 30/70 was +0.0076 hepatic C-index; this
amplified ~2.1× on LB (+0.016).

The 3-way variant (`phase2_blend_3way.csv`, LB 0.829) added a tuned XGB-Cox/baseline_v1
as a third stacker input. OOF training-time pairwise correlations 0.49 / 0.43 looked
diversifying, but LB regressed 0.051. The third's "diversity" was OOD-flavoured rather
than additive — adding more dimensions to the stacker is not a free win.

## Component lineage at a glance

```
landmark_3y_RSF ──┐
                  ├── 0.30/0.70 OOF-stacked ──→ phase2_blend_2way_optimal (LB 0.8965)
permissive_avg ───┘                                                  ▲
   │                                                                  │
   ├── rsf_longitudinal_summary (Tier 4)                              │
   ├── xgb_cox_longitudinal_plus_meta (Tier 4)                        │
   └── xgb_cox_longitudinal_summary (Tier 4)                          │
                                                                      │
death side (constant across all Phase 2 submissions): ────────────────┘
   xgb_cox / longitudinal_summary, n_est=300, lr=0.05
```
