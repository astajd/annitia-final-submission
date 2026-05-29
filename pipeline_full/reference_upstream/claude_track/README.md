# Reference upstream — Claude track

This is a reference-only copy of the Claude track source tree, included so
reviewers can trace how the **Claude hepatic anchor** was produced.

The anchor itself — the cached test-side CSV consumed by `build_final_3.py` —
is in:

```
../../runnable/cached_intermediates/claude_track_handoff/best_submissions/phase2_blend_2way_optimal.csv
```

The track's original top-level README is preserved as
`ORIGINAL_CLAUDE_README.md`; this file is the reference-upstream wrapper.

Raw challenge data (`data/raw/`) has been excluded; the `data/` directory
is intentionally empty. See `../../../data/README.md` for placement
instructions.

## Cached artifact this track relates to

- **`runnable/cached_intermediates/claude_track_handoff/best_submissions/phase2_blend_2way_optimal.csv`**
  — the Claude hepatic anchor (TEST predictions), produced by
  `experiments/phase2_stack_2way.py`.
- Plus the OOF reconstructions in
  `runnable/cached_intermediates/claude_track_handoff/optional_oof_predictions/`.

## Step 1 classification

**Exact producer present; weight OOF-selected.**

- **Producer script:** `experiments/phase2_stack_2way.py`. The matching
  JSON sidecar `submissions/phase2_blend_2way_optimal.json` pins:
  - `"weight_landmark": 0.3`, `"weight_permissive": 0.7`
  - Hepatic components: `landmark_3y_RSF` (tuned RSF on `baseline_v1`,
    trained on `train ∩ filter-A`) + `permissive_ensemble_avg`
    (rank-mean of `rsf_longitudinal_summary`,
    `xgb_cox_longitudinal_plus_meta`, `xgb_cox_longitudinal_summary`).
  - Death: `xgb_cox` on `longitudinal_summary` (143 cols),
    `n_estimators=300`, `learning_rate=0.05`,
    `death_mode="censor_missing_death_at_last"`.

- **Weight selection — OOF, not LB.** The script's pre-registered
  decision rule (`phase2_stack_2way.py` L22-29) is:
  > if `w*` ∈ [0.4, 0.6] → stay with 50/50.
  > if `w*` outside that AND m-s improves ≥0.005 → build phase2_blend_2way_optimal.csv.

  where `w*` is argmax of `mean − std` on the OOF weight grid
  (5-fold × 10-repeat stratified CV, 50 folds, `base_seed=42`,
  weight grid 0.00–1.00 step 0.05). Honest-LOFO median pick 0.35
  corroborates the choice. **Not LB-tuned.**

- **The "LB 0.8965" reference.** This appears only in *downstream*
  sidecars (`experiments/phase2_audit_v2.py` L399 / L460 / L474, and
  `experiments/phase2_build_strongpermissive.py` L159), as a record
  of the OOF-selected weight's observed public-LB outcome and as a
  trigger for follow-up audits — NOT as a tuning input. See
  `../../../docs/PROVENANCE.md`.

## Seeds / env pinning

- Per-fold seeds pinned: `base_seed=42`, repeats 0–9.
- Hyperparameters pinned via `configs/optuna_*.json` (referenced by
  `phase2_stack_2way.py`).
- **Library versions are NOT bit-pinned at this level.** The track's
  `requirements.txt` lists open ranges. RSF / xgb_cox training is
  stochastic and version-sensitive — bit/rank-identical
  raw-to-anchor reproduction is therefore not guaranteed even with
  pinned hyperparameters.

## Reproducibility claim

**Raw-to-anchor rank-identical reproduction is NOT claimed from this
directory.** The cached anchor CSV
(`runnable/cached_intermediates/claude_track_handoff/best_submissions/phase2_blend_2way_optimal.csv`)
is the authoritative artifact for the Claude side; the final blend in
`runnable/build_final_3.py` consumes that CSV directly. See
`../../../docs/PROVENANCE.md` and `../../../docs/REPRODUCIBILITY.md`.

## What is in this tree

- `src/` — Claude pipeline library (`cv`, `data`, `features`,
  `models`, `config`).
- `experiments/` — Phase-1 and Phase-2 experiments, including the
  anchor producer (`phase2_stack_2way.py`).
- `submissions/` — Claude track candidate CSVs and JSON sidecars,
  including `phase2_blend_2way_optimal.{csv,json}`.
- `configs/` — Optuna hyperparameter configs referenced by experiment
  scripts.
- `reports/` — internal track reports / phase summaries / OOF caches.
- `claude_handoff_for_merge/`, `handoff_minimal/` — packaged handoff
  bundles that were delivered into the merged sprint.
