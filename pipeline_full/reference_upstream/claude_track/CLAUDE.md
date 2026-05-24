# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ANNITIA — Trustii.io / IHU ICAN data challenge. Predict major hepatic events and all-cause death in MASLD patients from longitudinal NIT data. Scored by `0.7 × C_hepatic + 0.3 × C_death` (concordance index). Phase 1 is complete; Phase 2 work is specified in `SILENTBASE_HANDOFF.md`.

Read `reports/phase1_summary.md` and `SILENTBASE_HANDOFF.md` before making non-trivial changes — they contain the leakage audit findings and pre-registered protocol decisions that constrain what is acceptable.

## Setup & commands

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pipeline (run in this order to reproduce Phase 1, ~30–60 min on a laptop):

```bash
python3 experiments/01_leakage_audit.py        # writes reports/leakage_audit.json
python3 experiments/phase1_incremental.py      # writes reports/phase1_cv_results.csv (resumable)
python3 experiments/03_make_submission.py      # writes submissions/phase1_ensemble.csv
```

Single-config CV (useful for debugging one cell of the bake-off grid):

```bash
python3 experiments/run_one.py <hepatic|death> <death_mode|n/a> <feature_set> <model>
# e.g. python3 experiments/run_one.py hepatic n/a baseline_v1 rsf
```

`phase1_incremental.py` checks `reports/phase1_cv_results.csv` and skips already-completed (endpoint, death_mode, feature_set, model) rows on restart — delete the CSV (or specific rows) to force re-run.

## Architecture

Single-Python-package pipeline rooted at `src/`. The five modules form a tight chain that experiments compose; do not bypass them.

- **`src/config.py`** — single source of truth for paths, column groups (`STATIC_FEATURES`, `LONGITUDINAL_VARS`, `NIT_VARS`), `MAX_VISITS=22`, and the pre-registered CV protocol (`CV_N_SPLITS=5`, `CV_N_REPEATS`, `CV_RANDOM_STATE=42`). Hepatic/death weights live here too.
- **`src/data.py`** — raw CSV loading and survival target construction. `build_targets(df, death_mode)` returns `(hepatic_t, death_t)` DataFrames each with `<endpoint>_event`, `<endpoint>_time`, `<endpoint>_valid` columns. Two death modes exist: `drop_missing_death` (excludes NaN-death rows) and `censor_missing_death_at_last` (censors them at last visit, preferred default — uses all 1253 patients).
- **`src/features.py`** — `build_features(df, feature_set, *, event=None, time=None)` is the only feature-builder entry point. Stateless and deterministic from raw columns, so train/test feature builds can be done independently and column-aligned downstream. Each feature set has a leakage-risk label in `FEATURE_SET_RISK`.
- **`src/cv.py`** — repeated stratified K-fold by event indicator. `evaluate_cv(fit_predict_fn, X, event, time, ...)` is the universal CV harness; it expects models conforming to the `(X_tr, e_tr, t_tr, X_va) -> risk_va` contract.
- **`src/models.py`** — model factories that all return a closure with the same `(X_tr, e_tr, t_tr, X_va) -> risk_va` signature. Includes `coxnet`, `rsf`, `xgb_cox`, `lgbm_bin`, `catboost_bin`, `logreg_bin`. Binary classifiers convert survival to `event within horizon` and drop censored-before-horizon rows.

The contract that ties it together: every feature set is built from raw columns alone (with two exceptions noted below), every model is a `fit_predict` closure, and CV iterates folds × repeats over a stratified split on the event indicator. New features and new models slot in without touching the harness.

## Critical constraints (do not violate)

1. **`strict_time_aligned` is leaky and must not be used for shipped models.** Trajectory features computed over a per-row outcome-defined cutoff window encode the event time through `Age_delta`, `_count`, `_std`, etc. Confirmed in `reports/leakage_audit.json`: `Age_delta` alone gives C ≈ 0.958. The whole *category* of "summarize visits up to this row's event/censoring time" is Tier-5 leakage. The honest replacement is **fixed reference time** (same calendar cutoff for all patients, e.g. `Age_v1 + 2y`), described in `SILENTBASE_HANDOFF.md` Experiment 5.

2. **`strict_time_aligned` is also the only feature set that takes `event`/`time` arguments to `build_features`.** This is by design — it should never be built on test data (no event/time exists), and only the leakage-audit experiment may build it on train data. Any new feature set whose definition depends on the row's outcome is forbidden.

3. **Pre-registered CV protocol is 5-fold × N-repeats stratified on the event indicator.** Phase 1 used 3 repeats (47 hepatic events, std ~0.08); Phase 2 uses 10 repeats. Don't change `CV_N_SPLITS` to 10 — that gives only 4-5 events per validation fold, which is too noisy. Stratify by event, not by anything else.

4. **Selection metric is `mean_cindex - 1×std_cindex`.** Don't optimize raw mean. Stable models > lucky models.

5. **Death endpoint saturates around 0.95** (driven by `followup_yrs` alone hitting 0.969). 95% of Phase-2 effort goes into hepatic, which has a real ceiling around 0.80 with current features. Don't over-engineer death.

6. **Submission strategy is rank-averaging across an ensemble**, not blending raw risks. C-index depends only on relative ordering, so averaging ranks is the principled aggregation.

## Leakage tier framework

Tag every new feature set with one of these tiers (see `SILENTBASE_HANDOFF.md` for full table):

- Tier 1 (clean): single fixed timestamp, no outcome dependence — e.g. `baseline_v1`
- Tier 2 (clean-longitudinal): up to a *fixed* clinical reference time — e.g. LOCF at `Age_v1 + 2y`
- Tier 3 (cadence/missingness): pattern-of-measurement only, no values — high risk but auditable
- Tier 4 (full-permissive): all visits regardless of timing — acceptable but flagged HIGH RISK
- Tier 5 (outcome-dependent): uses event/censoring time of the same row — **forbidden**

Add to `FEATURE_SET_RISK` in `src/features.py` when introducing new sets.

## Data layout

`data/raw/` holds `train.csv`, `test.csv`, `dictionary.csv`, `hello_world_submission.csv`. CSVs are gitignored. ID columns differ between sets: train uses `patient_id_anon`; test uses `trustii_id` + `patient_id_anon`. The submission schema is `trustii_id, risk_hepatic_event, risk_death`.

Visit columns follow the `<var>_v<i>` pattern with `i ∈ 1..22`. `add_visit_metadata` derives `min_age`, `max_age`, `n_visits`, `followup_yrs` from `Age_v*` columns.

## Output artifact convention

When adding new experiments, follow the `experiments/<name>/` layout from `SILENTBASE_HANDOFF.md`: `config.json`, `cv_results.csv`, `summary.json`, `feature_importance.csv`, `notes.md`, optional `submission.csv`. Use timestamped names (`exp_20260427_optuna_xgb_baseline/`).
