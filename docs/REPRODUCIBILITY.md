# Reproducibility

## What this repo reproduces

The byte-frozen submission file `frozen/slot1_prediction.csv` (sha256
`5b0f1043f0b27addab5cc8dde33e2774371e9370c231633462f2f29337e57618`) is the
authoritative artifact. This repo provides two scripts that demonstrate
the deterministic recipe that produces it from cached model-level
prediction outputs:

1. **`pipeline_full/runnable/build_slot1_only.py`** — the reviewer-facing
   **authoritative verification script**. Reads only the cached prediction
   CSVs under `pipeline_full/runnable/cached_intermediates/`, applies the
   slot1 recipe (LS=12 gate **read from cache**, q95 disagreement override
   → 50/50 merge fallback, `0.85·rank(CWGBSA) + 0.15·rank(GBSA)` for
   death), and emits `generated_slot1_prediction.csv`. Verify with
   `validate_slot1_only.py`. **No raw data required, no
   `ANNITIA_DATA_ROOT` required.**

2. **`pipeline_full/runnable/build_final_3.py`** — the historical
   multi-candidate producer that emitted three finalprobe_* candidates
   during the sprint. Preserved for traceability. Slot1 is `finalprobe_3`.
   This script calls `load_labels()` unconditionally and therefore
   requires raw `train.csv`/`test.csv`; even though the slot1 column
   values it writes are mathematically determined by cached
   intermediates only, the script will not run to completion without
   raw data. Use `build_slot1_only.py` for clean reviewer verification.

## What this repo does NOT claim

**Exact raw-to-submission retraining is NOT claimed.**

- The GPT hepatic anchor (`20260428_0059_phase3_10_horizon_blend_v2.csv`)
  is a stochastic LGBM/CatBoost/XGB greedy rank-blend; library-version
  sensitive.
- The Claude hepatic anchor (`phase2_blend_2way_optimal.csv`) is a
  stochastic RSF + xgb_cox blend whose weight 0.30 is OOF-selected (not
  LB-tuned); library-version sensitive.
- The death components (CWGBSA / GBSA from `scikit-survival`) are
  seed- and hyperparameter-pinned but library-version sensitive.
- The slot among three `finalprobe_*` candidates was selected by
  public-LB candidate pruning on a now-closed leaderboard.

Raw challenge data are needed only for:
- inspecting / rerunning upstream reference code under
  `pipeline_full/reference_upstream/`, or
- running the historical multi-candidate `build_final_3.py`.

The authoritative slot1 verification (`build_slot1_only.py`) needs none
of that — only the files already in
`pipeline_full/runnable/cached_intermediates/`.

## Authoritative reviewer command sequence

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

## Verified no-raw run

The no-raw-data claim has been demonstrated:

- With `ANNITIA_DATA_ROOT` unset AND `data/raw/` absent: `build_slot1_only.py`
  runs to completion; `validate_slot1_only.py` reports rank-identity for
  both `risk_hepatic_event` and `risk_death`; float-exact equality also
  holds (informational).
- With `ANNITIA_DATA_ROOT` pointed at an empty directory and `data/raw/`
  absent: same result.

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
