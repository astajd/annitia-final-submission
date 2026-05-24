# Phase 2 Experiment 1 — Reproduction + Expanded Audit

Date: 2026-04-26
Status: Experiment 1 complete. Experiment 2 (Optuna tuning) not started.

CV protocol for Phase 2: 5-fold × 10-repeat stratified on event indicator (50 folds). Selection metric: `mean − 1×std`.

Phase 2 grid wall-clock: 1467s (24.4 min) on the local machine. 105 cells (7 feature sets × 5 models × 3 endpoint-modes), all completed successfully.

## (a) Reproduction check vs Phase 1

Inner-joined Phase 1 (`reports/phase1_cv_results.csv`, 32 cells at 3 repeats) and Phase 2 (`reports/phase2_cv_results.csv`, 105 cells at 10 repeats) on `(endpoint, death_mode, feature_set, model)`.

| Statistic | Value |
|---|---|
| Matched cells | 32 |
| max \|Δmean\| | 0.0295 |
| mean \|Δmean\| | 0.0079 |
| Cells with \|Δmean\| > 0.005 | 15 / 32 |
| Cells with \|Δmean\| > 0.02 | 4 / 32 |

15 of 32 cells trip the 0.005 flag, but the directional pattern is consistent (12 of 15 are positive — Phase 2 mean > Phase 1 mean) and all sit comfortably inside the Phase 2 per-cell std. CV seeds 42–44 are shared between the runs by construction (`base_seed + r`), so the drift is the seven *additional* repeats moving the mean estimate. Nothing here looks like a code-path or feature-build regression.

**Flagged rows (|Δmean| > 0.005), descending:**

| endpoint | mode | feature_set | model | P1 mean | P2 mean | Δ | P2 std |
|---|---|---|---|---|---|---|---|
| hepatic | n/a | baseline_v1 | catboost_h5 | 0.635 | 0.664 | +0.0295 | 0.093 |
| hepatic | n/a | baseline_v1 | xgb_cox | 0.728 | 0.752 | +0.0239 | 0.095 |
| hepatic | n/a | longitudinal_summary | lgbm_h5 | 0.707 | 0.686 | −0.0215 | 0.105 |
| hepatic | n/a | baseline_v1 | coxnet | 0.682 | 0.703 | +0.0209 | 0.104 |
| death | drop_missing_death | baseline_v1 | coxnet | 0.752 | 0.770 | +0.0176 | 0.073 |
| hepatic | n/a | longitudinal_plus_meta | coxnet | 0.680 | 0.692 | +0.0122 | 0.117 |
| hepatic | n/a | longitudinal_summary | coxnet | 0.681 | 0.693 | +0.0119 | 0.116 |
| death | censor_missing_death_at_last | baseline_v1 | xgb_cox | 0.785 | 0.774 | −0.0110 | 0.086 |
| hepatic | n/a | longitudinal_summary | xgb_cox | 0.749 | 0.759 | +0.0101 | 0.104 |
| hepatic | n/a | longitudinal_plus_meta | xgb_cox | 0.752 | 0.761 | +0.0098 | 0.101 |
| death | drop_missing_death | longitudinal_summary | coxnet | 0.930 | 0.939 | +0.0093 | 0.025 |
| death | censor_missing_death_at_last | baseline_v1 | coxnet | 0.737 | 0.746 | +0.0093 | 0.109 |
| death | drop_missing_death | baseline_v1 | xgb_cox | 0.767 | 0.775 | +0.0083 | 0.071 |
| hepatic | n/a | early_v1_v3 | catboost_h5 | 0.722 | 0.728 | +0.0063 | 0.088 |
| hepatic | n/a | nit_only | xgb_cox | 0.789 | 0.783 | −0.0061 | 0.098 |

**Spot check — Phase 1 headline (RSF on baseline_v1, hepatic):** P1 0.7975±0.0780 → P2 0.8011±0.0868. Δ = +0.0036, *not* flagged. Headline reproduces.

**Take-away:** the 3-repeat Phase 1 numbers are usable as direction-of-travel signals but should not be quoted to three decimals. Use the 10-repeat Phase 2 numbers from here on.

### Notable cells visible only after Phase 2 (i.e. configurations missing from Phase 1's 32 successful cells)

These are not drift, but new measurements worth flagging.

| endpoint | feature_set | model | mean | std | mean − std |
|---|---|---|---|---|---|
| hepatic | nit_only | rsf | **0.822** | 0.081 | **0.741** |
| death (censor) | longitudinal_plus_meta | rsf | 0.953 | 0.019 | 0.935 |
| death (drop) | longitudinal_plus_meta | rsf | 0.957 | 0.017 | 0.940 |
| death (censor) | nit_only | rsf | 0.862 | 0.037 | 0.825 |

**RSF on `nit_only` is now the best honest hepatic model** (0.822 ± 0.081, m−s = 0.741), edging RSF on `baseline_v1` (0.801 ± 0.087, m−s = 0.714). Phase 1's recorded best honest hepatic was RSF on baseline_v1; the NIT-only RSF cell wasn't in Phase 1's CSV. This needs to feed Experiment 2 candidate-list selection.

## (b) Expanded single-feature audit

Seven new features added to `experiments/01_leakage_audit.py` and persisted to `reports/leakage_audit.json`. Each is a patient-level scalar derived purely from raw `<var>_v<i>` columns.

**Definitions** (all Tier-3-or-worse a priori — measurement cadence and missingness can correlate with both monitoring intensity and follow-up length):

| Feature | Definition |
|---|---|
| `missingness_count` | NaN cells across all longitudinal vars × all visits |
| `missingness_count_NIT` | NaN cells across NIT vars only |
| `time_since_last_NIT` | `max(Age_v<i> over visits with any NIT) − Age_v1`; NaN if no NITs |
| `n_NIT_measurements` | # visits where ≥1 NIT non-NaN |
| `n_visits_FibroScan` | # visits where `fibs_stiffness_med_BM_1_v*` non-NaN |
| `n_visits_FibroTest` | # visits where `fibrotest_BM_2_v*` non-NaN |
| `n_visits_Aixplorer` | # visits where `aixp_aix_result_BM_3_v*` non-NaN |

**Single-feature C-index, best sign per cell, against each endpoint:**

| Feature | Hepatic (n=1253, ev=47) | Death drop (n=984, ev=76) | Death censor (n=1253, ev=76) |
|---|---|---|---|
| `missingness_count` | 0.570 (pos) | 0.774 (pos) | 0.764 (pos) |
| `missingness_count_NIT` | 0.551 (pos) | 0.807 (pos) | 0.794 (pos) |
| `time_since_last_NIT` | 0.579 (neg) | **0.968 (neg)** | **0.969 (neg)** |
| `n_NIT_measurements` | 0.573 (neg) | 0.809 (neg) | 0.803 (neg) |
| `n_visits_FibroScan` | **0.633 (neg)** | 0.808 (neg) | 0.809 (neg) |
| `n_visits_FibroTest` | 0.529 (neg) | 0.751 (neg) | 0.737 (neg) |
| `n_visits_Aixplorer` | 0.527 (pos) | 0.683 (neg) | 0.676 (neg) |

For reference (from the existing audit, unchanged): `followup_yrs` death = 0.969, `n_visits` death = 0.846, `Age_v1` hepatic = 0.631.

## (c) Tier interpretation

Rubric (calibrated against Phase 1 baselines `followup_yrs` hep=0.560/death=0.969, `n_visits` hep=0.572/death=0.846):

- **Tier 3 (cadence, acceptable):** hepatic C ∈ [0.55, 0.70] AND death C < 0.90. Carries clinically interpretable monitoring-intensity information without doubling as a follow-up-length proxy.
- **Tier 4 (high risk):** hepatic C > 0.70, OR death C ∈ [0.90, 0.95], OR feature definition can include post-event visits.
- **Leakage proxy:** death C > 0.95 — the feature is essentially `followup_yrs` rebadged.

| Feature | Hep | Death (max) | Tier | Notes |
|---|---|---|---|---|
| `missingness_count` | 0.570 | 0.774 | **Tier 3** | Hepatic mild, death below the 0.90 threshold. Cadence-style. |
| `missingness_count_NIT` | 0.551 | 0.807 | **Tier 3** | Same shape. NIT cells leave less missingness when monitored, both for sicker patients and for longer-followed ones. |
| `time_since_last_NIT` | 0.579 | **0.968** | **Leakage proxy** | Death C-index is identical to `followup_yrs`. Structurally Tier-5-flavoured anyway: for event patients the last NIT visit can be post-event, so this also encodes outcome timing. **Exclude from feature sets.** |
| `n_NIT_measurements` | 0.573 | 0.809 | **Tier 3** | Cadence. |
| `n_visits_FibroScan` | **0.633** | 0.808 | **Tier 3 (boundary)** | Strongest hepatic single-feature signal among the new candidates. Below the 0.70 hep threshold but worth a sensitivity comparison in Experiment 4 (with vs without). |
| `n_visits_FibroTest` | 0.529 | 0.751 | **Tier 3** | Cadence; weaker because FibroTest is less commonly measured. |
| `n_visits_Aixplorer` | 0.527 | 0.683 | **Tier 3** | Sparse measurements drive the lower death C; mostly cadence. |

**Implications for Phase 2 work:**

1. `time_since_last_NIT` is out — the death C-index alone disqualifies it, and the structural argument (post-event NIT visits in the train set) is independent confirmation.
2. The remaining six can populate the `missingness_and_visit_cadence` feature set (Experiment 3e in `SILENTBASE_HANDOFF.md`) without immediate concern. They are Tier 3.
3. `n_visits_FibroScan` is the strongest hepatic-discriminator and sits at the Tier 3/4 boundary. Plan an ablation in Experiment 4: `baseline_v1 + cadence_minus_FibroScan` vs `baseline_v1 + cadence_with_FibroScan`. If the FibroScan-included variant wins by more than the per-cell std, that's where the cadence signal is concentrated and we should treat it as a high-risk feature even though the rubric labels it Tier 3.
4. Death modeling does not benefit from any cadence feature beyond what `longitudinal_summary` already extracts (death is already at 0.95). Cadence work should be evaluated only on hepatic.

## Files produced this experiment

- `reports/leakage_audit.json` — replaced; now contains 12 single-feature entries per endpoint loop (5 original + 7 new) under `single_features`. `strict_features_top10` and `post_event_visits` blocks unchanged.
- `reports/phase2_cv_results.csv` — new; 105 rows, schema-identical to `phase1_cv_results.csv`.
- `reports/phase1_cv_results.csv` — byte-unchanged (md5 88d55d3dfc00fb92b8be070cf4afec64).

Code changes:
- `src/config.py` — `CV_N_REPEATS = 10`.
- `experiments/01_leakage_audit.py` — added `compute_audit_extra_features` + `audit_features` helpers.
- `experiments/phase1_incremental.py` — argparse `--output PATH` and `--repeats INT`. Default behaviour preserved.

## Experiment 1.5: nit_only honest verification

Phase 2 Experiment 1 surfaced RSF on `nit_only` at 0.822 ± 0.081 (m−s = 0.741) as the best honest hepatic candidate. `nit_only` mixes NIT *values* at v1 with full trajectory features (`_last`, `_max`, `_count`, slope, etc.), so some of the lift could come from post-event NIT measurements (27 of 47 hepatic-event patients have ≥1 post-event visit, per the existing audit).

Comparator added: **`nit_only_baseline_only`** — Tier 1 clean baseline, no trajectory features. Contents: `fibs_stiffness_med_BM_1_v1`, `fibrotest_BM_2_v1`, `aixp_aix_result_BM_3_v1`, `Age_v1`, three presence flags (`had_FibroScan_at_v1`, `had_FibroTest_at_v1`, `had_Aixplorer_at_v1`), and the `STATIC_FEATURES` block. 14 features total. Tagged `low` in `FEATURE_SET_RISK`.

CV: 5-fold × 10-repeat stratified on hepatic event, `drop_missing_death`. Appended to `reports/phase2_cv_results.csv`.

| Model | `nit_only` (38 feat) | `nit_only_baseline_only` (14 feat) | Δ (lift from trajectory) |
|---|---|---|---|
| coxnet | 0.738 ± 0.119 | 0.697 ± 0.096 | +0.041 |
| rsf | **0.822 ± 0.081** | **0.772 ± 0.084** | +0.051 |
| xgb_cox | 0.783 ± 0.098 | 0.714 ± 0.091 | +0.070 |

**Best honest-honest hepatic with this feature set:** RSF on `nit_only_baseline_only` at **0.772 ± 0.084** (m−s = 0.688).

Trajectory features add +0.04 to +0.07 across all three models. Tree models capture more of it than linear (RSF +0.051, XGB +0.070, Coxnet +0.041) — consistent with `_count` and `_last` carrying nonlinear information about monitoring intensity and post-event values that trees can split on.

**Read against the pre-registered decision rule:** RSF baseline-only at 0.772 sits in the 0.74–0.78 band → partial leakage in the 0.822 figure. The honest core of NIT-v1 signal is real and meaningfully above `Age_v1` alone (0.631), but ~0.05 of the original lift comes from trajectory features and likely includes post-event NIT measurements. Anchor Experiment 2's Optuna budget on `baseline_v1`; carry `nit_only` and `nit_only_baseline_only` as secondary candidates with 0.822 treated as a ceiling, not a floor.

## Stop point

Experiment 2 (Optuna tuning) not started, per instruction.
