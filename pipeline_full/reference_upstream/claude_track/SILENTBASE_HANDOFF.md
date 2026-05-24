# Silentbase / Claude Code Handoff — ANNITIA Phase 2

## Context

ANNITIA is a Trustii.io / IHU ICAN data challenge: predict major hepatic events
and all-cause death in MASLD patients, scored by:

```
score = 0.7 × C_hepatic + 0.3 × C_death
```

Phase 1 (this repo) established a working pipeline, leakage audit, and first
submission. Read `reports/phase1_summary.md` first.

Headline numbers from Phase 1 (5-fold × 3-repeat stratified CV):

| Endpoint | Best honest model | C-index |
|---|---|---|
| Hepatic | RSF on baseline_v1 | 0.797 ± 0.078 |
| Hepatic | XGB-Cox on nit_only | 0.789 ± 0.117 |
| Death | XGB-Cox on longitudinal_summary | 0.952 ± 0.014 |

**Key finding:** death is essentially capped at ~0.95 by follow-up-time information; hepatic at ~0.80 with current features. **Phase 2 effort should be 95% hepatic.**

**Critical:** Phase 1 audit found that the `strict_time_aligned` feature set
itself leaks (trajectory features encode event time via `Age_delta`, `_count`).
**Do not use it as a feature set, and do not propose variants of "longitudinal
features computed only from pre-event visits per row" — that whole *category*
is leaky** because the cutoff is defined by the outcome. The fix is *fixed*
reference time (Experiment 5 below), not per-row alignment. This is non-obvious
and easy to re-derive incorrectly, so: any feature whose computation involves
the event/censoring time of the same row is OUT.

## Mission

Build the strongest honest model for the hepatic endpoint, with rigorous
local validation. Treat the leaderboard as a sanity check, not the optimization
target.

The 30% qualitative score makes a clean, interpretable, well-documented
submission worth real points. Code quality matters.

## Setup

```bash
cd ~/annitia
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # if missing, see Phase 1 src/__init__.py imports
```

Required packages: `pandas numpy scikit-learn scikit-survival xgboost lightgbm
catboost optuna shap matplotlib seaborn`.

For Phase 3+ (deep): `torch pycox`.

Run with the RTX 4090. Tree models are CPU-bound; deep models (Phase 3) use GPU.

## Project structure (already in place)

```
annitia/
├── data/raw/                     # train.csv, test.csv, dictionary.csv
├── src/
│   ├── config.py                 # paths, column groups, CV protocol
│   ├── data.py                   # load + target builders + visit metadata
│   ├── features.py               # feature-set builders
│   ├── cv.py                     # repeated stratified K-fold + C-index
│   └── models.py                 # uniform fit_predict wrappers
├── experiments/
│   ├── 01_leakage_audit.py       # single-feature C-index audit (RUN FIRST)
│   ├── run_one.py                # CV one (endpoint, fs, model)
│   ├── phase1_incremental.py     # full grid with resume
│   └── 03_make_submission.py     # rank-averaged ensemble submission
├── reports/                      # summaries, audit
├── submissions/                  # CSV + JSON metadata
└── configs/                      # YAML configs (see Phase 2 below)
```

## Phase 2 experiment queue (overnight on silentbase)

Run in this order. Each writes its own output and is independent of later steps.

### Experiment 1: Reproduce Phase 1 with full CV protocol + expanded audit

**Why:** Phase 1 used 3 repeats for time. With 47 hepatic events, std ~0.08
across folds — we need 10 repeats for stable rankings.

**Protocol decision (pre-registered):** 5-fold × 10 repeats stratified on
hepatic event = 50 folds. Reason: 10-fold gives only 4-5 events per validation
fold which is too noisy at 47 events; 5-fold gives 9-10 per fold (more stable).

```bash
# Edit src/config.py: set CV_N_REPEATS = 10
# Edit experiments/phase1_incremental.py: set PHASE1_REPEATS = 10
python3 experiments/phase1_incremental.py
```

Compare to Phase 1 numbers. If a model swings >0.03 in mean, it was unstable
and should be deprioritized.

**Also expand the single-feature audit** in `01_leakage_audit.py` to include:
- `missingness_count` per patient (total NaN cells across all visit columns)
- `missingness_count_NIT` (NaN count restricted to NIT columns)
- `time_since_last_NIT` (Age_v1 to last visit with any NIT measurement)
- `n_NIT_measurements` (visits with at least one NIT)
- For each, report C-index alone, both signs.

These are clinically meaningful (sicker = monitored more) but also potential
leakage proxies. The audit tells us which.

### Experiment 2: Hyperparameter optimization for top hepatic models

Tools: Optuna, 50 trials per model. Use the 10×5 CV harness from `cv.py`.

Top candidates from Phase 1:
- RSF on baseline_v1 (best honest)
- XGB-Cox on nit_only
- XGB-Cox on baseline_v1
- LGBM binary h=5 on baseline_v1

Tune (RSF): `n_estimators ∈ [200, 800]`, `min_samples_leaf ∈ [5, 50]`,
`min_samples_split ∈ [10, 100]`, `max_features ∈ {sqrt, log2, 0.3, 0.5}`.

Tune (XGB-Cox): `n_estimators ∈ [200, 800]`, `learning_rate ∈ [0.01, 0.1]`,
`max_depth ∈ [3, 8]`, `min_child_weight ∈ [1, 20]`, `subsample ∈ [0.6, 1.0]`,
`colsample_bytree ∈ [0.5, 1.0]`, `reg_lambda ∈ [0.1, 10.0]`.

Tune (LGBM bin): same as XGB plus `num_leaves`, `horizon ∈ {3, 5, 7, 10}`.

**Selection rule (pre-registered):** rank by `mean_cindex - 1×std_cindex`. We
want stable models, not lucky ones.

Save: `experiments/phase2_optuna/<model>_<fs>.json` with best params + CV history.

### Experiment 3: Feature engineering expansion

Add to `src/features.py`:

a. **`baseline_plus_clinical`** — baseline_v1 + extended clinical scores:
   - FIB-4, APRI, AST/ALT (already added at v1)
   - **Hepamet score** (age, sex, T2DM, AST, ALT, BMI, PLT) — known MASLD-fibrosis score
   - **NAFLD fibrosis score (NFS)** — needs albumin (we don't have it) or substitute
   - **AST × age / PLT** (modified)
   - PLT/(BMI + age) and similar interaction terms

b. **`fixed_reference_time`** — for each patient, take their last visit at or
   before age `Age_v1 + 2.0` (two years after baseline). This is honest because
   the cutoff is fixed by the calendar, not by the outcome. Compute trajectory
   features over `[Age_v1, Age_v1 + 2]`.

c. **`baseline_with_missingness`** — add `_was_missing` indicator columns for
   every baseline_v1 feature. Missingness pattern itself can be informative.

d. **`nit_combined`** — synthesize NITs:
   - max of (FibroScan_v1, FibroTest_v1×100, Aixplorer_v1) standardized
   - presence pattern (which NITs were measured at v1)
   - count of NITs at v1

e. **`missingness_and_visit_cadence`** — features derived purely from the
   *pattern* of measurement, not the values:
   - total NaN count across all longitudinal columns
   - NaN count restricted to NIT columns
   - n_visits with any NIT measurement
   - n_visits with FibroScan / FibroTest / Aixplorer specifically
   - time-since-last-NIT (Age_v1 reference)
   - inter-visit gaps (mean, max, std of consecutive Age_v* differences)
   - "visits per follow-up year" rate
   - flag: had_any_NIT_at_baseline

   This is a leakage-tier-3 set (high risk: sicker → monitored more, but also
   correlated with follow-up length). Include in the bake-off explicitly so
   we can quantify how much of the score is value-signal vs cadence-signal.
   If `baseline_v1 + missingness_and_visit_cadence` ≈ `baseline_v1` alone,
   cadence isn't really helping. If it adds 0.03+, we know what we're paying
   for.

Re-run the bake-off with these new feature sets. Particularly, `fixed_reference_time`
is the *honest* longitudinal feature set.

### Experiment 4: Honest-vs-permissive head-to-head

For the 4 top model configurations, build:
- `fixed_reference_time` (Experiment 3b — honest, fixed cutoff)
- `longitudinal_summary` (permissive — all visits, including post-event)
- `baseline_v1` (clean reference)
- `missingness_and_visit_cadence` only (Tier 3 audit)
- `baseline_v1 + missingness_and_visit_cadence` (combined)

Compare CV C-index. Specifically interested in:
1. Does `fixed_reference_time` beat `baseline_v1`? If yes → real longitudinal
   signal exists. If no (within 1 SE) → first-visit features are essentially
   sufficient, simplify.
2. How much of `longitudinal_summary`'s score is cadence vs values? Compare
   to `missingness_and_visit_cadence` alone and `baseline_v1+cadence`.

**Note: `strict_time_aligned` is NOT in this comparison.** It's leaky (proven
in Phase 1) and only kept in the codebase for the audit-probe submission
(Experiment 7).

### Experiment 5: Honest fixed-reference-time features

Generalize: build feature sets for several reference horizons:
- `Age_v1` (= baseline_v1)
- `Age_v1 + 1y`
- `Age_v1 + 2y`
- `Age_v1 + 3y`
- `Age_v1 + 5y`

For each, the features are: LOCF values + slope-since-baseline + count-of-visits-in-window.

For event-patient training rows, the row is excluded if their event happened
before the reference time (otherwise we leak the outcome).

This is the cleanest extension of the longitudinal idea. **Important deliverable.**

### Experiment 6: Survival ensembles

Once the top 4-6 models are tuned (Exp 2), combine via:

a. **Rank-average ensemble** — simple, robust, current default. No fitting,
   no overfitting risk. The Phase 1 submission uses this.

b. **Regularized stacking with linear meta-learner** — risk_final = β₀ +
   Σ βᵢ × rank(model_i), where β are tuned on out-of-fold predictions. Use
   isotonic/Cox regression for the meta layer. **Heavy regularization required**
   (L2 with α tuned on inner CV) because with 50 folds and 5-6 base models,
   fitting a 6-d weight vector unregularized produces unstable weights on 47
   events. Compare β stability across CV repeats — if any βᵢ flips sign across
   repeats, drop the regularization further or just use rank-average.

c. **Bayesian model averaging** by CV C-index — weight ∝ exp(γ × CV_score).
   γ is a hyperparameter; γ→∞ recovers winner-take-all, γ→0 recovers uniform.

d. **OOF cross-feature stacking (cross-endpoint)** — cheap version of
   multi-task learning. Procedure:
   1. Fit hepatic models with 5-fold OOF, save OOF risk predictions
      `pred_hep_oof` (one per train row).
   2. Fit death models with 5-fold OOF, save `pred_death_oof`.
   3. Add `pred_hep_oof` as a feature for the death endpoint, refit death
      models. The death model can now learn that hepatic-risky patients
      die sooner.
   4. Add `pred_death_oof` as a feature for the hepatic endpoint, refit
      hepatic models.
   5. Optionally iterate (1-2 passes only — diminishing returns, increasing
      overfit risk).

   Test-time inference: predict hepatic and death risks independently first,
   then re-predict each with the *test-set* cross-endpoint risk as a feature.
   Be careful not to use train-set OOF predictions of the *test* rows here —
   for test rows, you must use the model fit on full train (no OOF).

   **Why this matters:** the two endpoints are biologically correlated (death
   often follows hepatic decompensation). Multi-task learning would capture
   this; cross-feature stacking captures most of it without adding the
   degrees of freedom of a deep multi-task model. Worth ~0.005-0.015 lift
   if the correlation is strong; could also hurt if base models are already
   capturing the shared signal. Test it, don't assume.

Compare a/b/c/d. Stacking usually adds 0.005–0.015 over rank-average if base
models are diverse; cross-feature stacking can add similar lift on top of
that, *or* nothing if endpoints decouple in this dataset.

### Experiment 7: Generate multiple submissions

For each major decision (death mode, ensemble strategy, feature set), produce
a labeled submission file:

```
submissions/
├── phase2_honest_baseline.csv         # baseline_v1 ensemble
├── phase2_honest_extended.csv         # baseline + fixed-reference + cadence
├── phase2_permissive.csv              # longitudinal_summary ensemble (high-risk)
├── phase2_strict_audit_probe.csv      # strict_time_aligned (ONE-SHOT AUDIT ONLY)
├── phase2_nit_only.csv                # NIT-only clinical model
└── phase2_best_cv.csv                 # whichever has best CV mean − std
```

**Submission strategy:**
1. Submit `phase2_best_cv.csv` first. This is our actual ship candidate.
2. **One-shot audit:** submit `phase2_strict_audit_probe.csv` once to learn
   whether the test set has the same per-row outcome-dependent visit-cutoff
   structure as train. If LB ≈ CV (~0.94 hepatic), the test was generated
   the same way — useful diagnostic but the model still isn't shippable for
   qualitative review (visit-count features have zero clinical justification).
   If LB << CV, we've confirmed strict-leaky doesn't generalize.
3. Submit `phase2_honest_extended.csv` if it beats best_cv on a second probe.
4. The strict-audit-probe never goes in the rotation as a finalist — 30% of
   the score is qualitative review and a 0.94 hepatic from `Age_delta` will
   be transparently bad to the IHU ICAN reviewers (they wrote the data).

## Phase 3 (after Phase 2 finishes)

Only proceed once Phase 2 has selected a stable hepatic model.

### Joint multi-task survival

Shared-representation neural network with two heads (hepatic, death). Use pycox
or build with PyTorch directly:

- Input: standardized features (use baseline_v1 + fixed-reference)
- Shared trunk: 2-layer MLP, 64-128 hidden units, dropout 0.3
- Hepatic head: DeepSurv-style Cox loss
- Death head: DeepSurv-style Cox loss
- Total loss: 0.7×L_hep + 0.3×L_death

Compare to independent XGB-Cox baselines. Multi-task often helps when one
endpoint is data-starved (hepatic, 47 events).

### Discrete-time hazard reformulation

Reshape to long format:
- For each (patient, year-after-baseline), create one row
- Label: did event happen in that year given still-at-risk at start?
- Features: LOCF up to year start
- Train as binary classifier (LGBM, NN)

This converts 47 hepatic events into ~5000 (patient, year) rows. Often wins
small-event-count survival problems.

### Semi-supervised pseudo-labeling

After Phase 2 best model is locked:
- Predict test set risks
- Take top-K most confident test patients (e.g., top 20 highest-risk and
  bottom 20 lowest-risk)
- Pseudo-label them as event/no-event for hepatic
- Retrain with extended training set
- Compare CV (should not degrade — if it does, abandon)

### External data (LATE-stage)

If we hit a ceiling and want to push further:
- NHANES, UKBB MASLD subsets — but harmonization is hard
- Pretrain a representation on external longitudinal MASLD data, fine-tune

Probably not worth it unless we're top-3 on CV and want a tiebreaker.

## Pre-registered selection criteria

To avoid LB chasing:

1. **Primary metric:** CV `mean_C_hepatic - 1×std_C_hepatic`
2. **Tiebreaker:** weighted score `0.7×mean_hep + 0.3×mean_death`
3. **Sanity check:** LB score within 1 SD of CV mean

If LB diverges from CV by >2 SD, **trust CV**. The most likely cause is LB-PB split noise (47/(public+private) ≈ small).

## Reporting / qualitative deliverable

Write `reports/methodology.md` covering:

1. Data audit findings (death follow-up encoding, post-event visits, strict leakage)
2. CV protocol (pre-registered, repeated stratified)
3. Feature engineering rationale (why baseline_v1 / nit_only beats permissive)
4. Model choice (RSF stability + XGB-Cox + binary classifiers, no deep learning rationale)
5. Ensemble strategy (rank average, why)
6. Limitations and what we'd do with more time
7. SHAP / feature importance for the top model — clinical plausibility check

This is what the qualitative judges read. Make it clean and short — 3-5 pages.

## Output artifact specification

Every experiment must save:

```
experiments/<experiment_name>/
├── config.json              # all hyperparams, feature set, model, mode
├── cv_results.csv           # per-fold per-repeat metrics
├── summary.json             # mean/std/median/min/max
├── feature_importance.csv   # if available
├── notes.md                 # what was tried, what worked, gotchas
└── submission.csv           # if applicable
```

Use timestamps in experiment names: `exp_20260427_optuna_xgb_baseline/`.

## Configuration files

Create YAML configs in `configs/` for reusable experiment specs:

```yaml
# configs/exp_xgb_baseline.yaml
endpoint: hepatic
death_mode: censor_missing_death_at_last
feature_set: baseline_v1
model:
  type: xgb_cox
  params:
    n_estimators: 500
    learning_rate: 0.03
    max_depth: 4
cv:
  n_splits: 5
  n_repeats: 10
  random_state: 42
```

Build `experiments/run_config.py` that reads YAML and dispatches. Lets us
queue many experiments cleanly.

## Time budget guidance

Phase 2 should fit in ~4-8 hours of compute on the 4090 box.

Approximate per-experiment cost (Phase 1 timings ×10/3 for repeats):
- Reproduce with 10 repeats: ~10 min total
- Optuna (50 trials × 4 models × 50s/trial): ~3 hours
- New feature sets (5 sets × 4 models × 60s): ~20 min
- Strict-vs-permissive heads-to-heads: ~30 min
- Ensemble experiments: ~30 min
- Submissions + reports: ~30 min

**Total: ~5 hours.** Leave headroom for surprises.

## Anti-patterns to avoid

- ❌ Using `strict_time_aligned` features (proven leaky)
- ❌ Building NEW features that summarize "visits before this row's event" — same leakage class as strict_time_aligned, just renamed
- ❌ Optimizing for high mean C-index without checking std
- ❌ Submitting many LB probes before exhausting CV variants (we have a daily limit)
- ❌ Adding 100+ features without checking that they help CV
- ❌ Discarding the 269 NaN-death cohort for hepatic (no reason to)
- ❌ Building separate test-side feature pipelines (use the same `build_features` API)
- ❌ "Looks good on this fold" — every claim needs ≥30 fold-evaluations behind it
- ❌ Polishing the model before fixing the audit (we already wasted 0.938 on a leaky number)
- ❌ Fitting unregularized stacking weights with only 47 events

## Leakage tier framework (formal)

Every feature must be tagged with one of:

| Tier | Definition | Examples |
|---|---|---|
| 1: clean | computable from a single fixed timestamp, no outcome dependence | Age_v1, BMI_v1, FibroScan_v1, FIB-4 at v1 |
| 2: clean-longitudinal | computable up to a *fixed* clinical reference time (same for all patients) | LOCF at Age_v1+2y, slope over [v1, v1+2y] |
| 3: cadence/missingness | uses pattern of measurement, not outcome timing | n_visits_in_first_2y, NIT_count_at_baseline |
| 4: full-permissive | uses all visits regardless of timing | last-available-value, all-visit slope |
| 5: outcome-dependent | uses event/censoring time of the same row | strict_time_aligned, "visits before event" |

**Tier 5 is OUT.** Tier 4 is acceptable but tagged HIGH RISK because it can
include post-event visits. Tiers 1-3 are the bulk of honest modeling.

The `feature_set_risk_label()` helper in `src/features.py` reports the tier
for each feature set. New feature sets must be added to `FEATURE_SET_RISK`
with their tier classification.

## When to stop / declare a "ship candidate"

A configuration is ship-ready when:
1. CV (10×5 = 50 folds) is computed and saved
2. mean − std exceeds the current best by ≥0.01
3. SHAP/feature importance is clinically plausible
4. Honest feature provenance (no per-row outcome-dependent cutoffs)
5. LB result, if obtained, within 0.05 of CV mean

Aim for 2-3 ship candidates with different feature philosophies (e.g., one
NIT-only, one full-clinical, one ensemble) so the final submission can be a
defensible mixture.

## Final submission day checklist

- [ ] All experiments documented in `reports/methodology.md`
- [ ] SHAP explanations for the top model
- [ ] CV table with confidence intervals (bootstrap CI on per-fold C-index)
- [ ] Submission file validated against `hello_world_submission.csv` schema
- [ ] Backup submission generated (in case primary fails sanity)
- [ ] Code reproducibility check — fresh clone, run scripts, get same numbers
- [ ] Submitted notebook is clean, narrated, and runs end-to-end
- [ ] Anonymous credit to A. Štajduhar (KB Dubrava / Med Zagreb) once allowed
