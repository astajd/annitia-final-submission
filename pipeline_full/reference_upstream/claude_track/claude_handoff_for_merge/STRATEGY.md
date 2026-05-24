# Anthropic-track strategy summary

## Top 3 things that worked
1. Cross-method blending (landmark + permissive, ρ 0.478): +0.069 LB jump 0.811 → 0.880
2. OOF-stacked weight optimization (50/50 → 30/70): +0.016 LB jump 0.880 → 0.8965
3. Tier-2 fixed-reference-time landmark methodology (3y RSF): clean, defensible single-model anchor

## Top 3 things that didn't work
1. Within-method ensembling: multi-landmark gave LB 0.81 (tied single 3y), internal corrs 0.81-0.88 too high
2. Trajectory-shape features: redundant with longitudinal_summary's _last/_std/_slope (corr 0.754 with permissive)
3. OOF cross-feature stacking (death pred → hepatic feature): +0.009 CV but -0.024 LB; CV-LB decoupling

## Methodology directions NOT tried (why)
- Discrete-time hazard reformulation: estimated low marginal value over RSF/XGB-Cox partial likelihood
- Pseudo-labeling: deferred after CV-LB decoupling — pseudo-labels assume calibrated test predictions
- Deep multi-task survival: no clear case for it given dataset size (1253 rows, 47 hepatic events)
- Visit-level pre/post-event classifier: too ambitious, replaced with simpler sensitivity analysis

## Key insights about the data
1. **LB noise floor empirically ±0.028** (analytical SE ~0.034-0.048). Three CV→LB inversions consistent with sampling noise, not systematic. See reports/phase3_lb_decoupling.md.
2. **Honest captures 99% of permissive on hepatic.** Tier 1 v1-only blend within 0.009 m-s of Tier 4 production. See reports/sensitivity_analysis.md.
3. **Death is saturated at ~0.95** from follow-up information. 95% of hepatic optimization effort is correct allocation.
4. **Per-row outcome-dependent feature cutoffs leak.** Strict-time-aligned probe scored LB 0.692 vs CV 0.94 — confirmed train-test asymmetry. Tier 5 is OUT.

## Forum intel used
- Organizer (April 14, 6:20): longitudinal features valid, post-event observations are present, smart handling rewarded in subjective scoring
- Organizer (April 14, 9:42): test set has post-event-equivalent observations, methodology choice considered in qualitative review

## Final ensemble structure
phase2_blend_2way_optimal.csv:
- Hepatic: 30% rank(landmark_3y_RSF) + 70% rank(permissive_ensemble_avg)
- Death: XGB-Cox on longitudinal_summary, censor_missing_death_at_last
- Weights from OOF stacking (5×10 fold structure, random_state=42+repeat)
- Rank-blending (not raw scores)
- Endpoint-independent: separate models per endpoint, no cross-endpoint blending in submission

## Honest assessment of LB position
Our progression: 0.799 → 0.811 → 0.880 → 0.8965. Mostly preplanned, audited gains. Single-shot 0.8965 is one draw from a true distribution likely centered around 0.87-0.92. The 0.054 gap to GPT track's 0.911 may be real signal, may be sampling — to be revealed by ρ analysis.
