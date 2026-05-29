# pipeline_full/runnable

Slot1 regeneration and verification for the ANNITIA submission. This
directory provides **two verification paths**, matching the root `README.md`.

## Path A — full raw-data retraining

```bash
bash retrain_all_from_raw.sh
```

Retrains **every** slot1 component from the raw challenge data in
`../../data/raw/` and regenerates the submitted prediction file
**byte-for-byte** in the tested environment. The generated
`retrain_outputs/final_retrain_prediction.csv` has whole-file MD5
`fb04658b99f89ec822e9c604d537dcae` — identical to
`../../frozen/slot1_prediction.csv` (both endpoints rank-identical and
float-exact under the pinned environment in `requirements_retrain.txt`).
See `../../docs/RETRAINING.md`.

## Path B — fast cached-output verification

```bash
# from repo root: bash frozen/verify.sh
cd pipeline_full/runnable
python build_slot1_only.py
python validate_slot1_only.py
```

A faster deterministic check that regenerates slot1 from saved
model-level outputs in `cached_intermediates/` (no retraining).
`build_slot1_only.py` writes `generated_slot1_prediction.csv`, and
`validate_slot1_only.py` confirms it is **rank-identical** to
`../../frozen/slot1_prediction.csv` for both endpoints, ending with:

```
SUCCESS — generated_slot1_prediction.csv is RANK-IDENTICAL to frozen/slot1_prediction.csv
```

Set up the fast-path environment first with
`pip install -r requirements.txt`. Path B does **not** retrain and does
**not** require raw data.

## Other scripts (historical / provenance)

- `build_final_3.py` / `validate_reproduction.py` — the original
  cached-intermediate producer and validator from the assembly sprint.
  Retained for **provenance** to document how the slot1 blend was first
  assembled; the reviewer entry points are the Path A and Path B
  commands above, not these.

- `upstream_scripts/` — the producers of the cached intermediates:
  `task3_tree_survival.py` (CWGBSA / GBSA death models),
  `task5_gated_ensemble.py` (LS=12 gate), `build_submissions.py` (zoo
  submission helpers), `build_sprint.py` (50/50 merge). Included for
  traceability and exercised by Path A's from-raw retraining; Path B
  does not re-execute them. See `../../docs/PROVENANCE.md`.

## Layout

```
runnable/
├── retrain_all_from_raw.sh     # Path A: full from-raw retraining orchestrator
├── retrain_lib/                # Path A: from-raw retraining stages
├── build_slot1_only.py         # Path B: deterministic slot1 from cached intermediates
├── validate_slot1_only.py      # Path B: rank-identity check vs frozen
├── build_final_3.py            # historical/provenance: original cached-pred blend
├── validate_reproduction.py    # historical/provenance: rank-identity check
├── requirements.txt            # Path B (fast) environment
├── requirements_retrain.txt    # Path A (from-raw, pinned) environment
├── README.md
├── lib/
│   ├── zoo_utils.py            # path-patched copy of the zoo utilities
│   └── claude_src/             # minimal copy of claude/src (cv, data, features, models, config)
├── upstream_scripts/           # cached-intermediate producers (used by Path A; Path B does not re-execute)
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

## What each path reproduces

| stage | Path B (from cached intermediates) | Path A (from raw, pinned env) |
|---|---|---|
| final slot1 blend | YES, RANK-IDENTICAL to frozen | YES, BYTE-IDENTICAL to frozen (MD5 fb04658b99f89ec822e9c604d537dcae) |
| LS=12 gate emission | deterministic from anchor CSVs + raw LS | retrained from raw |
| 50/50 merge | deterministic from anchor CSVs | retrained from raw |
| CWGBSA / GBSA death | cached files used directly | retrained from raw under pinned env |
| GPT hep anchor | cached file used directly | retrained from raw under pinned env |
| Claude hep anchor | cached file used directly | retrained from raw under pinned env |
| Slot1 selection over alternatives | encoded by `finalprobe_3` being the only submitted file | encoded; alternatives are not re-derived |

Byte-exactness of Path A is verified under the pinned environment in
`requirements_retrain.txt`; reruns under materially different library
versions may differ. See `../../docs/REPRODUCIBILITY.md`.
