# RETRAINING.md — from-raw reproduction of slot1

> Scope: technical description of the from-raw retraining command (Path A),
> verified in this submitted repository. The fast cached-output verification
> path (Path B: `build_slot1_only.py` + validators) is unchanged and remains
> the quick reviewer check.

## Command

```bash
bash pipeline_full/runnable/retrain_all_from_raw.sh
```

This single command retrains **every** slot1 component from the raw challenge
data and runs the deterministic final assembly, then validates the result
against `frozen/slot1_prediction.csv`.

## Raw data

Required at `data/raw/` (organizer-provided; included in the private review repo
with permission):

```
data/raw/train.csv
data/raw/test.csv
data/raw/dictionary.csv
data/raw/hello_world_submission.csv
```

The orchestrator stages these into per-track working copies (the GPT track uses
the original `DB-…csv` filenames internally; the staging copy handles the
renaming — content is identical).

## Tested environment

- Python **3.11.4**
- Pinned packages: `pipeline_full/runnable/requirements_retrain.txt`
  (numpy 2.4.4, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.6.1,
  scikit-survival 0.27.0, xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10,
  PyYAML 6.0.3).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r pipeline_full/runnable/requirements_retrain.txt
```

Byte-identical reproduction requires this pinned environment; materially
different library versions may produce close but not byte-identical outputs.

## What it runs (14 stages, all must exit 0)

- **Track B (GPT), 10 steps:** `phase3_current_state_v2`,
  `phase2_NIT_plus_scores_longitudinal`, `phase3_6_no_visit_history`,
  `phase3_6_csv2_extra_seeds`, `phase3_6_hepatic_aug`, **`build_phase3_submissions`**
  (writes the `phase3_current_state_v2.json` sidecar that `build_phase3_5_candidates`
  requires), `build_phase3_5_candidates`, `build_phase3_6_candidates`,
  `run_phase3_9_horizon`, `run_phase3_10_horizon` → Track B hepatic/death anchor.
- **Track A (Claude), 1 step:** `phase2_stack_2way.py` → 2-way OOF-stacked anchor.
- **Death, 1 step:** `task3_tree_survival.py` → CWGBSA + GBSA survival models.
- **Final assembly:** the LS=12 gate and the 50/50 merge are derived from the
  regenerated anchors using the verified upstream functions, then
  `build_slot1_only.py` is run **verbatim** over the regenerated components
  (LS=12 gate + q95 disagreement override + 0.85·CWGBSA / 0.15·GBSA death blend).
- **Validation:** compares the assembled file to `frozen/slot1_prediction.csv`.

The chain consumes only freshly-generated artifacts and raw data — it does not
read `cached_intermediates/` or any old OOF/test/submission CSV as a model input.

## Expected runtime

About **41–45 minutes** on the tested 32-core host with the thread caps applied.

### Thread caps are intentional

The orchestrator exports, before any training:

```
PYTHONHASHSEED=0
LOKY_MAX_CPU_COUNT=8
OMP_NUM_THREADS=2
MKL_NUM_THREADS=2
OPENBLAS_NUM_THREADS=2
NUMEXPR_NUM_THREADS=2
```

These are **required**, not cosmetic. Without them, scikit-survival's
`RandomSurvivalForest(n_jobs=-1)` over-subscribes threads (~168 on this host) and
thrashes — a single experiment failed to finish a single model in 3.5 h. With the
caps it completes in minutes. The caps also help determinism.

## Output

```
pipeline_full/runnable/retrain_outputs/final_retrain_prediction.csv
```

plus per-stage logs in `pipeline_full/runnable/retrain_outputs/logs/` and a
`comparison_to_frozen.txt`.

## Result (verified in the submitted repo, tested environment)

`final_retrain_prediction.csv` is **byte-for-byte identical** to
`frozen/slot1_prediction.csv`, verified by running the command in this
submitted repository (runtime ≈ 41 min):

- whole-file **MD5 `fb04658b99f89ec822e9c604d537dcae`** (both files)
- both endpoints rank-identical **and** float-exact (max abs diff = 0.0,
  Spearman ρ = 1.0), 423 rows, schema `trustii_id, risk_hepatic_event, risk_death`.

## Generated directories (not committed)

`pipeline_full/runnable/retrain_work/` (per-track working copies + staged inputs)
and `pipeline_full/runnable/retrain_outputs/` (final file, logs, reports) are
**generated** each run and are git-ignored. The orchestrator removes and rebuilds
`retrain_work/` on every invocation.

## Relationship to the fast verification path

The fast cached-output verification path is **still available and unchanged**
for a quick check that does not require retraining (using `requirements.txt`):

```bash
bash frozen/verify.sh
cd pipeline_full/runnable
python build_slot1_only.py
python validate_slot1_only.py
```

This document covers the heavier from-raw path.
