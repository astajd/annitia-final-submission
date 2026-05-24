# pipeline_full/runnable

Deterministic regeneration of the submitted slot1 file from **cached
prediction-level intermediates**. Verifies rank-identity to
`../../frozen/slot1_prediction.csv`.

This is the "verified path." Raw-to-submission reproducibility is NOT
claimed — see `../../PROVENANCE_FINDINGS.md`, `../../audit/`, and
`../reference_upstream/` for the upstream tracks and why bit/rank-identical
raw reproduction is not credible.

## What runs

- `build_final_3.py` — the immediate producer of the slot1 file. Reads
  cached prediction CSVs from `cached_intermediates/`, applies the
  deterministic final blend (LS=12 gate, q95 disagreement override,
  0.85/0.15 death blend), and writes the regenerated CSV to
  `_outputs/submissions/finalprobe_3_lsthr12_disagq95_cw85gbsa15.csv`.

- `validate_reproduction.py` — checks the regenerated file against
  `../../frozen/slot1_prediction.csv`: schema, 423 rows, no NaNs,
  `trustii_id` alignment, and **rank-identity** to the frozen file.
  Float-exact equality is reported as informational only — rank-identity
  is the pass criterion.

- `upstream_scripts/` — the producers of the cached intermediates:
  `task3_tree_survival.py` (CWGBSA / GBSA dea), `task5_gated_ensemble.py`
  (LS=12 gate), `build_submissions.py` (zoo submission helpers),
  `build_sprint.py` (50/50 merge). Included for traceability; they are
  **not re-executed** by the smoke test. Re-running them requires
  raw challenge data (see `../../data/README.md`) and is subject to
  library-version sensitivity (sksurv in particular).

## How to verify (reviewer command sequence)

```bash
cd pipeline_full/runnable
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place raw data per ../../data/README.md (only needed for label/LS derivation).
export ANNITIA_DATA_ROOT="/abs/path/to/data/raw"   # OR drop train.csv/test.csv into ../../data/raw/

python build_final_3.py
python validate_reproduction.py
```

Expected validate output ends with:

```
SUCCESS — regenerated slot1 is RANK-IDENTICAL to frozen/slot1_prediction.csv
```

## Layout

```
runnable/
├── build_final_3.py            # entry point (deterministic blend over cached preds)
├── validate_reproduction.py    # schema / row count / NaN / id / rank-identity check
├── requirements.txt
├── README.md
├── lib/
│   ├── zoo_utils.py            # path-patched copy of the zoo utilities
│   └── claude_src/             # minimal copy of claude/src (cv, data, features, models, config)
├── upstream_scripts/           # cached-intermediate producers (reference; not re-executed)
│   ├── task3_tree_survival.py
│   ├── task5_gated_ensemble.py
│   ├── build_submissions.py
│   └── build_sprint.py
├── configs/
│   ├── finalprobe_3_metadata.json   # exact slot1 metadata (sprint-original)
│   └── official_portfolio.yaml      # 5-slot portfolio definition (for cross-reference)
└── cached_intermediates/       # MODEL-OUTPUT CSVs (NOT raw data) — see README in this dir
    ├── gpt_track_handoff/
    ├── claude_track_handoff/
    ├── merge_sprint/
    └── model_zoo_sprint/
```

## What is and is not reproduced here

| stage | from cached intermediates | from raw |
|---|---|---|
| build_final_3.py final blend | YES, RANK-IDENTICAL (verified by smoke test) | (cached intermediates required) |
| LS=12 gate emission | YES, deterministic from anchor CSVs + raw LS | inherits anchor break |
| 50/50 merge | YES, deterministic from anchor CSVs | inherits anchor break |
| CWGBSA / GBSA dea | (cached files used directly) | runnable, library-version sensitive |
| GPT hep anchor | (cached file used directly) | producer present; stochastic ensemble, not bit/rank-identical |
| Claude hep anchor | (cached file used directly) | producer present; weight OOF-selected; stochastic ensemble |
| Slot1 selection over alternatives | (encoded by `finalprobe_3` being the only submitted file) | manual / closed-LB pruning, not codified |
