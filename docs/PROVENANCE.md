# Provenance

## Submitted file

`frozen/slot1_prediction.csv` — sha256
`5b0f1043f0b27addab5cc8dde33e2774371e9370c231633462f2f29337e57618`,
423 rows, columns `trustii_id, risk_hepatic_event, risk_death`. Both
risk columns are emitted as pure ranks (`scipy.stats.rankdata`).

## Two reproduction scripts in this repo

1. **`pipeline_full/runnable/build_slot1_only.py`** — the reviewer-facing
   **authoritative verification script**. Slot1-only. Reads only cached
   prediction CSVs under `pipeline_full/runnable/cached_intermediates/`.
   No raw data, no `ANNITIA_DATA_ROOT`. The LS=12 gate is **read** from
   the cached gate prediction CSV; it is NOT recomputed from any
   liver-stiffness column.

2. **`pipeline_full/runnable/build_final_3.py`** — the historical
   multi-candidate producer that the sprint used to emit three
   `finalprobe_*` candidates. Slot1 is `finalprobe_3`. This script calls
   `load_labels()` unconditionally and so requires raw `train.csv` /
   `test.csv` to run end-to-end (even though the slot1 column values
   are mathematically determined by cached intermediates only). It is
   preserved here for traceability; the authoritative reproduction
   path is `build_slot1_only.py`.

## Slot1 recipe (deterministic, hardcoded)

Hepatic endpoint:
1. Base gate: read `test__gate_LSthr12__hep.csv` (the LS=12 gate
   prediction, produced upstream by `task5_gated_ensemble.py`; rule:
   `if LS_last >= 12 kPa: rank(Claude anchor) else rank(GPT anchor)`).
2. Disagreement override (`q = 0.95`): compute
   `d = |rank(GPT) - rank(Claude)| / N`,
   `τ = quantile(d, 0.95)`; where `d > τ`, override with
   `rank(merge_50_50_hep)`; else keep `rank(base_gate)`.
3. Final hepatic column = `rankdata(<above>)`.

Death endpoint:
1. `dea = 0.85 · rank(CWGBSA test) + 0.15 · rank(GBSA test)`.
2. Final death column = `rankdata(dea)`.

Output ordering: by `trustii_id` ascending (from the GPT anchor CSV).

## What is NOT claimed

- **Exact raw-to-submission reproduction is NOT claimed.** The hepatic
  anchors and the death components are stochastic and library-version
  sensitive; the upstream training environment is not bit-pinned.
- **The slot1-vs-alternative selection was not codified.** `finalprobe_3`
  was chosen over OOF-superior `finalv2_LS14` via public-LB candidate
  pruning on a now-closed leaderboard.

## Provenance of each cached input (consumed by `build_slot1_only.py`)

| input file | producer (reference only) | producer status |
|---|---|---|
| `gpt_track_handoff/best_submissions/20260428_0059_phase3_10_horizon_blend_v2.csv` | `gpt/src/run_phase3_10_horizon.py` (`greedy_cap25_both`, weights pinned in JSON sidecar) | stochastic ensemble training; library-version sensitive |
| `claude_track_handoff/best_submissions/phase2_blend_2way_optimal.csv` | `claude/experiments/phase2_stack_2way.py` — a 0.30/0.70 blend of a landmark RSF component and a permissive component (rank-mean of RSF and XGB-Cox models); weight OOF-selected by argmax mean−std on a 5×10 CV grid, honest-LOFO median 0.35 | stochastic ensemble training; OOF-selected weight (NOT LB-tuned) |
| `merge_sprint/submissions/merge_A_best_50_50_both.csv` | `merge_sprint/scripts/build_sprint.py` (0.5·rank(GPT) + 0.5·rank(Claude)) | deterministic given the two anchors |
| `model_zoo_sprint/predictions/test__gate_LSthr12__hep.csv` | `model_zoo_sprint/scripts/task5_gated_ensemble.py` (`conditional_rule_blend(var="ls", thr=12.0)`) | deterministic given the anchors + raw LS; reads cached anchors |
| `model_zoo_sprint/predictions/test__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv` | `model_zoo_sprint/scripts/task3_tree_survival.py` (sksurv `ComponentwiseGradientBoostingSurvivalAnalysis`, `n_estimators=300, lr=0.05, subsample=0.8`, per-repeat `random_state=42+r`, 5×10 stratified CV mean-of-ranks) | seed-pinned; sksurv version sensitive |
| `model_zoo_sprint/predictions/test__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv` | `model_zoo_sprint/scripts/task3_tree_survival.py` (sksurv GBSA, `n_estimators=200, lr=0.05, max_depth=3, subsample=0.8, max_features="sqrt"`, per-repeat `random_state=42+r`) | seed-pinned; sksurv version sensitive |

The upstream producing code is shipped read-only under
`pipeline_full/reference_upstream/`. Raw data are required only if a
reviewer wants to run those upstream scripts or `build_final_3.py`; the
slot1 verification path (`build_slot1_only.py`) does not need them.
