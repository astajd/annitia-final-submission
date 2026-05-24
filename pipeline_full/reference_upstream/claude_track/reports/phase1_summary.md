# ANNITIA Phase 1 — Leakage Audit & CV Summary

Date: 2026-04-26
Status: Phase 1 complete (initial bake-off + audit). Phase 2 to run on silentbase.

## Headline findings

1. **Death C-index is mostly free.** `followup_yrs` alone (sign-flipped) gives C-index ≈ **0.969**. Proxy-only Coxnet gives **0.946 ± 0.025**. Differentiation between death models will happen in the third decimal — death is not where the competition is decided.

2. **Hepatic is the real game.** `Age_v1` alone gives ~0.63. Random Survival Forest on `baseline_v1` (clean, low-leakage features) gives **0.797 ± 0.078**. There is real, learnable signal — and that's the main lever for the leaderboard.

3. **`strict_time_aligned` is itself leaky** (newly discovered during the audit). Trajectory features computed on the per-row "before-event window" encode the event time through `Age_delta`, `_count`, etc. Coxnet on strict gives 0.938, but a single feature `Age_delta` alone gives **0.958** — definitionally leakage. **Discard the strict approach as currently implemented.** Time-aligned features need to be at a *fixed* clinical reference time, not a per-row cutoff defined by the outcome.

4. **NaN-death cohort doesn't materially change death modeling.** `drop_missing_death` and `censor_missing_death_at_last` give within-noise results (0.946 vs 0.943 for proxy-only, 0.950 vs 0.952 for longitudinal+xgb). Use the censor mode as default — uses more data, statistically more correct.

5. **Hepatic event data has structural quirks.** 27/47 hepatic-event patients have visits AFTER the recorded event (avg 3.3 post-event visits). The dataset includes follow-up after diagnosis. **Permissive feature aggregation can use these visits**, but they leak the outcome. The cleanest hepatic features are `baseline_v1` and `nit_only`.

## Key numbers (all CV: 5-fold × 3 repeats, stratified by event)

### Hepatic (n=1253, events=47)

| Feature set | Model | C-index | Risk |
|---|---|---|---|
| baseline_v1 | RSF | **0.797 ± 0.078** | low |
| nit_only | XGB-Cox | 0.789 ± 0.117 | moderate |
| longitudinal_plus_meta | XGB-Cox | 0.752 ± 0.094 | high (count proxies) |
| longitudinal_summary | XGB-Cox | 0.749 ± 0.094 | moderate |
| early_v1_v3 | XGB-Cox | 0.744 ± 0.093 | low-moderate |
| baseline_v1 | XGB-Cox | 0.728 ± 0.113 | low |
| baseline_v1 | LGBM h=5 | 0.677 ± 0.100 | low |
| **strict_time_aligned** | Coxnet | **0.938 ± 0.054** | LEAKY (see audit) |
| followup_proxy_only | Coxnet | 0.573 ± 0.132 | reference |

Headline: cleanest honest model is **RSF on baseline_v1** at 0.797.

### Death (n=984 drop, 1253 censored, events=76)

| Feature set | Model | mode | C-index |
|---|---|---|---|
| longitudinal_summary | XGB-Cox | censor | 0.952 ± 0.014 |
| longitudinal_summary | XGB-Cox | drop | 0.950 ± 0.019 |
| followup_proxy_only | Coxnet | drop | 0.946 ± 0.025 |
| longitudinal_summary | Coxnet | drop | 0.930 ± 0.031 |
| baseline_v1 | XGB-Cox | censor | 0.785 ± 0.097 |
| baseline_v1 | Coxnet | drop | 0.752 ± 0.108 |

Headline: **0.95 is the practical ceiling** for death; longitudinal features get there.

### Combined weighted score (0.7·hep + 0.3·death)

Best honest configuration so far:
- Hepatic: RSF on baseline_v1 → 0.797
- Death: XGB-Cox on longitudinal_summary, censor mode → 0.952
- **Weighted: 0.7×0.797 + 0.3×0.952 = 0.844**

If the strict trick generalizes to test (it almost certainly doesn't, but worth noting):
- Hepatic: 0.938, Death: 0.952 → 0.943

## Leakage audit details

### Audit method

For each candidate feature/feature-set, compute its single-feature C-index (trying both signs) and compare to honest baselines. Anything where a single feature gives high C-index without clinical justification is suspect.

### Death-side leakage (definitional, not exploitable)

| Feature | C-index alone (best sign) |
|---|---|
| `followup_yrs` | 0.969 |
| `max_age` | 0.949 |
| `n_visits` | 0.846 |
| `Age_v1` | 0.624 |

This is **not** "leakage" in the cheat sense — for a censored survival problem, the censoring time literally encodes "still alive at this time", so longer follow-up = lower hazard. Every model will pick this up. **Practical implication:** all death models converge to ~0.95 quickly. Don't over-engineer.

### Hepatic-side proxy strength

| Feature | C-index alone |
|---|---|
| `Age_v1` | 0.631 |
| `n_visits` | 0.572 (sign-flipped) |
| `followup_yrs` | 0.560 (sign-flipped) |
| `max_age` | 0.580 |

These are weak — hepatic does not have free runs from metadata. The 0.797 RSF baseline reflects real learnable biology.

### Strict time-alignment leakage (NEWLY DISCOVERED)

In `strict_time_aligned`, features are computed using only visits with `age ≤ Age_v1 + time`. For event patients, `time = age_at_event - Age_v1` (small). For censored, `time = max_age - Age_v1` (large, all visits).

This means strict-aligned features encode "is this an event patient" through:

| Strict feature | C-index alone |
|---|---|
| `Age_delta` (max − min age in window) | **0.958** |
| `Age_rel_delta` | 0.946 |
| `Age_std` | 0.938 |
| `fibs_stiffness_count` | 0.841 |
| `Age_count` | 0.836 |
| `ast_count` | 0.819 |

**This is hard leakage.** Even with `_count` columns dropped, the remaining strict features still give 0.926 — because `_min`/`_max`/`_last`/`_mean` of NIT values within a smaller window encode "fewer visits" implicitly. **Do not use strict-aligned features as currently constructed.**

The fix for honest time-alignment is **fixed reference time**: build features at a single clinically-meaningful timestamp (e.g., 2 years after Age_v1 with LOCF) for every patient, regardless of when the event happens. This is a Phase 2 task.

### Post-event visits in train data

| | n | % |
|---|---|---|
| Total event patients (hepatic) | 47 | 100% |
| Have ≥1 post-event visit | 27 | 57% |
| Avg post-event visits per affected patient | 3.3 | — |

When using `longitudinal_summary`, last-visit features for these patients are *post-event* values (e.g., post-decompensation labs). This biases trajectory features. The fact that `longitudinal_summary` doesn't beat `baseline_v1` (0.749 vs 0.728 XGB-Cox; 0.681 vs 0.797 RSF) suggests this leakage is mostly *unhelpful* noise — but it inflates training C-index estimates.

## What this means for strategy

1. **The hepatic ceiling honest models can hit:** somewhere in 0.78–0.85 with good feature engineering. RSF on baseline already at 0.80.

2. **Permissive longitudinal features are not actually winning** — they're not significantly better than `baseline_v1` on hepatic. The leakage they contain may also hurt generalization to test (if test was constructed with cleaner cutoffs).

3. **NIT-only model is competitive AND defensible** (0.789 XGB-Cox, n=38 features). For the qualitative deliverable, this is a strong candidate as the "clinical backbone" model.

4. **Optimize hepatic, accept death.** Death is essentially a free 0.95. Spend 95% of Phase-2 effort on hepatic.

5. **Don't trust the leaderboard early.** With 47 hepatic events and probably ~70 in test public+private split, single-LB-fold C-index has SD ≈ 0.05. CV-based selection > LB-based selection.

## Phase 1 deliverables produced

- `data/raw/{train,test,dictionary,hello_world_submission}.csv`
- `src/{config,data,features,cv,models}.py` — modular pipeline
- `experiments/01_leakage_audit.py` — produces `reports/leakage_audit.json`
- `experiments/run_one.py` — single-config CV
- `experiments/phase1_incremental.py` — full grid with resume
- `experiments/03_make_submission.py` — rank-averaged ensemble submission
- `reports/phase1_cv_results.csv` — all CV results
- `reports/leakage_audit.json` — single-feature audit
- `submissions/phase1_ensemble.csv` — first valid submission
- `submissions/phase1_ensemble.json` — ensemble metadata

## Open questions for the team

1. **Are the strict trajectory features really 100% leaky in the test set?** The test set has a similar visit-cutoff structure. If the organizer constructed test with the same window-end policy, then `Age_delta` etc. would correlate with the test outcome too — but only because the data generator produced the artifact. Worth submitting one strict-features submission to test this. Marked HIGH RISK.

2. **Was the synthetic data generated per-patient or via a population model?** If population, our leakage findings transfer. If per-patient with hidden labels, post-event visits may contain real biological signal we should use.

3. **What is the scoring time horizon?** The competition description doesn't specify a fixed horizon. C-index handles all horizons jointly. But knowing whether they emphasize 5y or 10y predictions would influence horizon choice for binary classifiers.
