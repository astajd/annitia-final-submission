# Reference upstream — GPT track

This is a reference-only copy of the GPT track source tree, included so
reviewers can trace how the **GPT hepatic anchor** was produced.

The anchor itself — the cached test-side CSV consumed by `build_final_3.py` —
is in:

```
../../runnable/cached_intermediates/gpt_track_handoff/best_submissions/20260428_0059_phase3_10_horizon_blend_v2.csv
```

Raw challenge data (`data/raw/`) has been excluded; the `data/` directory
is intentionally empty. See `../../../data/README.md` for placement
instructions.

## Cached artifact this track relates to

- **`runnable/cached_intermediates/gpt_track_handoff/best_submissions/20260428_0059_phase3_10_horizon_blend_v2.csv`**
  — the GPT hepatic anchor (TEST predictions).
- Plus the OOF component CSVs under
  `runnable/cached_intermediates/gpt_track_handoff/oof_predictions/`
  (used by `lib/zoo_utils.load_oof_baselines` for an OOF proxy
  reconstruction — see below).

## Step 1 classification

**Exact producer script present + frozen intermediates; raw-to-anchor
stochastic.**

- **Producer script:** `src/run_phase3_10_horizon.py`. The matching
  JSON sidecar
  `submissions/20260428_0059_phase3_10_horizon_blend_v2.json` pins:
  - `"blend_id": "greedy_cap25_both"` — greedy rank-blend with a 25% cap
  - `"components_hepatic"`:
    `NIT_plus_scores__hepatic__h1__lgbm_binary`,
    `v3_hepatic_schema__hepatic__h3__lgbm_binary__s4`
  - `"weights_hepatic"`:
    `_v3` → 0.7905138339920948,
    `NIT_plus_scores__hepatic__h1__lgbm_binary` → 0.11857707509881421,
    `v3_hepatic_schema__hepatic__h3__lgbm_binary__s4` → 0.09090909090909091
  - `"uses_target_derived_features": false`
  - Identical recipe on the death side (different components).

- **Intermediate horizon-model OOF/test CSVs** are persisted under
  `experiments/outputs/phase3_10_horizon/<run>/{oof,test}.csv` (77
  run directories). Given those frozen intermediates, the greedy
  blend in the producer script is deterministic.

- **Underlying training is stochastic.** The horizon classifiers are
  LGBM / CatBoost / XGB binary models (seed 0 for the main runs; seeds
  1–4 for the multi-seed targets). Re-running from raw is library-version
  sensitive and not guaranteed bit/rank-identical.

- **Independent flag — the merged-tree "reconstructed" proxy.** The
  merged-pipeline file `model_zoo_sprint/scripts/zoo_utils.py` (also
  shipped as `runnable/lib/zoo_utils.py`) uses HARD-CODED PROXY weights
  `0.7905 / 0.1186 / 0.0909` in `load_oof_baselines` to reconstruct the
  GPT anchor on the OOF side from underlying horizon-component OOF
  CSVs. Those proxy weights are an OOF-only reconstruction internal
  to the merged tree, NOT the producer of the test-side anchor CSV.
  See `../../../PROVENANCE_FINDINGS.md` section (a).

## Seeds / env pinning

- Per-component seeds pinned in `run_phase3_10_horizon.py` (`seed=0`
  for main runs; `seeds_extra = [1,2,3,4]` for the multi-seed targets
  on the best hep and death families).
- Greedy blend weights are emergent from the OOF-driven greedy search;
  they are pinned in the JSON sidecar (`weights_hepatic`,
  `weights_death`) and the frozen intermediates make the greedy
  selection deterministic.
- **Library versions are NOT bit-pinned at this level.**
  `experiments/configs/` carries per-experiment YAML configs but no
  global env lock. LGBM / CatBoost / XGB training is stochastic — bit/
  rank-identical raw-to-anchor reproduction is therefore not guaranteed.

## Reproducibility claim

**Raw-to-anchor rank-identical reproduction is NOT claimed from this
directory.** The cached anchor CSV
(`runnable/cached_intermediates/gpt_track_handoff/best_submissions/20260428_0059_phase3_10_horizon_blend_v2.csv`)
is the authoritative artifact for the GPT side; the final blend in
`runnable/build_final_3.py` consumes that CSV directly. See
`../../../audit/TRACE_REPORT.md` and `../../../PROVENANCE_FINDINGS.md`.

## What is in this tree

- `src/` — GPT pipeline library and phase-specific run scripts
  (including `run_phase3_10_horizon.py`, the anchor producer).
- `experiments/configs/` — per-experiment YAML configs.
- `experiments/outputs/` — frozen per-run OOF/test CSVs from each
  horizon classifier (input to the greedy blend).
- `submissions/` — GPT track candidate CSVs and JSON sidecars,
  including
  `20260428_0059_phase3_10_horizon_blend_v2.{csv,json}` (the anchor).
- `notebooks/`, `reports/` — internal analyses, sensitivity probes,
  and phase summaries.
