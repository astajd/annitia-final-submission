# Reproducibility

The byte-frozen submission file `frozen/slot1_prediction.csv` (sha256
`5b0f1043f0b27addab5cc8dde33e2774371e9370c231633462f2f29337e57618`, md5
`fb04658b99f89ec822e9c604d537dcae`) is the authoritative artifact. This repo
provides **two verification paths**, both of which have been run in the real
submitted repository.

## Path A — full raw-data retraining (verified)

```bash
bash pipeline_full/runnable/retrain_all_from_raw.sh
```

This single command retrains **every** slot1 component from the raw challenge
data in `data/raw/` (Track A / Track B hepatic anchors, the LS=12 gate, the
50/50 merge, and the CWGBSA / GBSA death models) and runs the deterministic
final assembly (`build_slot1_only.py` logic) over the freshly regenerated
components. It does not read `cached_intermediates/` as a model input.

**Verified result in the real submitted repo:** the generated
`pipeline_full/runnable/retrain_outputs/final_retrain_prediction.csv` is
**byte-for-byte identical** to `frozen/slot1_prediction.csv` —

- whole-file **md5 `fb04658b99f89ec822e9c604d537dcae`** (both files),
- both endpoints rank-identical **and** float-exact (Spearman ρ = 1.0,
  max abs diff = 0.0), 423 rows.

Runtime was approximately **41 minutes** on the tested host; budget about
**41–45 minutes**. Full stage-by-stage detail is in `docs/RETRAINING.md`.

**Environment.** Byte-exact retraining is verified under the pinned/tested
environment in `pipeline_full/runnable/requirements_retrain.txt` (Python
3.11.4; numpy 2.4.4, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.6.1,
scikit-survival 0.27.0, xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10,
PyYAML 6.0.3) with the orchestrator's thread caps applied. The upstream
survival models are stochastic and library-version sensitive, so byte-exact
reproduction is claimed **under this pinned environment**; reruns under
materially different library versions may differ.

## Path B — fast cached-output verification

A quicker deterministic check from saved model-level outputs, requiring no
retraining and no raw data:

1. **`pipeline_full/runnable/build_slot1_only.py`** — reads only the cached
   prediction CSVs under `pipeline_full/runnable/cached_intermediates/`,
   applies the slot1 recipe (LS=12 gate **read from cache**, q95 disagreement
   override → 50/50 merge fallback, `0.85·rank(CWGBSA) + 0.15·rank(GBSA)` for
   death), and emits `generated_slot1_prediction.csv`. Verify with
   `validate_slot1_only.py`. **No raw data, no `ANNITIA_DATA_ROOT` required.**

2. **`pipeline_full/runnable/build_final_3.py`** — the historical
   multi-candidate producer that emitted three `finalprobe_*` candidates
   during the sprint. Preserved for traceability. Slot1 is `finalprobe_3`.
   This script calls `load_labels()` unconditionally and therefore requires
   raw `train.csv`/`test.csv`; the slot1 column values it writes are
   mathematically determined by cached intermediates only. Use
   `build_slot1_only.py` for the clean reviewer check.

```bash
# 1. byte-frozen file
bash frozen/verify.sh
# → slot1_prediction.csv: OK

# 2. set up runnable env
cd pipeline_full/runnable
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. regenerate slot1 from cached intermediates (NO raw data needed)
python build_slot1_only.py
python validate_slot1_only.py
# → SUCCESS — generated_slot1_prediction.csv is RANK-IDENTICAL to frozen/slot1_prediction.csv
```

## Path B does not read raw data

**Path B does not read raw data regardless of whether `data/raw/` is
present.** `build_slot1_only.py` consumes only saved model-level outputs;
its result (rank-identity for both `risk_hepatic_event` and `risk_death`,
with float-exact equality reported as informational) is identical whether
`data/raw/` exists, `ANNITIA_DATA_ROOT` is unset, or `ANNITIA_DATA_ROOT`
points at an empty directory.

The script's IO is restricted to six CSVs under `cached_intermediates/`:

- `gpt_track_handoff/best_submissions/20260428_0059_phase3_10_horizon_blend_v2.csv`
- `claude_track_handoff/best_submissions/phase2_blend_2way_optimal.csv`
- `merge_sprint/submissions/merge_A_best_50_50_both.csv`
- `model_zoo_sprint/predictions/test__gate_LSthr12__hep.csv`
- `model_zoo_sprint/predictions/test__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv`
- `model_zoo_sprint/predictions/test__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv`

No `os.environ` / `os.getenv` calls, no `load_raw()`, no `load_labels()`,
no liver-stiffness recomputation. The LS=12 gate is read from the cached
prediction CSV (`test__gate_LSthr12__hep.csv`); recomputing it would
reintroduce a raw-data dependency and is explicitly avoided.
