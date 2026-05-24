# Phase 3.10 — submission recommendation

Reference: `phase3_9_horizon_blend` (public LB 0.90419, weighted OOF 0.8768, hepatic 0.8451, death 0.9507).

## Recommendation: submit ONE candidate

**File**: `gpt/submissions/20260428_0059_phase3_10_horizon_blend_v2.csv`

**Blend**: `greedy_cap25_both` (alpha=None, side=both)

- weighted OOF: 0.8811 (Δ vs Phase 3.9 = +0.0043)
- hepatic OOF:  0.8500 (Δ vs Phase 3.9 = +0.0049)
- death OOF:    0.9537 (Δ vs Phase 3.9 = +0.0031)
- hep fold std/min: 0.1084/0.6000
- dea fold std/min: 0.0123/0.9269
- rank-corr w/ phase3_9_horizon_blend test: hep=0.994, dea=0.997, weighted=0.995

### How it differs from phase3_9_horizon_blend

- hepatic components: ['NIT_plus_scores__hepatic__h1__lgbm_binary', 'v3_hepatic_schema__hepatic__h3__lgbm_binary__s4']
- death components:   ['current_state_v2__death__h5__catboost_binary__s3', 'NIT_plus_scores__death__h4__catboost_binary']
- weights: hep={'_v3': 0.7905138339920948, 'NIT_plus_scores__hepatic__h1__lgbm_binary': 0.11857707509881421, 'v3_hepatic_schema__hepatic__h3__lgbm_binary__s4': 0.09090909090909091}, dea={'_v3': 0.7905138339920948, 'current_state_v2__death__h5__catboost_binary__s3': 0.11857707509881421, 'NIT_plus_scores__death__h4__catboost_binary': 0.09090909090909091}

### Public-LB outcome interpretation

- If LB > 0.90419 by ≥ +0.001 → horizon-ensemble extension transferred; Phase 3.11 should iterate on the ensemble selection.
- If LB ≈ 0.90419 (within ±0.001) → the OOF gain didn't transfer; private split has different horizon mass; treat horizons as exhausted.
- If LB < 0.90419 by ≥ -0.002 → likely a private-vs-public horizon mismatch; fall back to Phase 3.9 candidate and explore non-horizon directions.

### Why we did not promote the robust variant

The OOF-best variant already had competitive fold stability; the second candidate provides a smaller-OOF fallback (saved at `gpt/submissions/20260428_0059_phase3_10_horizon_private_robust.csv`) but should not be submitted unless Phase 3.11 reveals a private-vs-public mismatch.
