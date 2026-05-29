# Provenance

## Submitted file

`frozen/slot1_prediction.csv` — sha256
`5b0f1043f0b27addab5cc8dde33e2774371e9370c231633462f2f29337e57618`,
423 rows, columns `trustii_id, risk_hepatic_event, risk_death`. Both
risk columns are emitted as pure ranks (`scipy.stats.rankdata`).

## Raw data

The official challenge data are included under `data/raw/`
(`train.csv`, `test.csv`, `dictionary.csv`, `hello_world_submission.csv`),
following organizer clarification. They are the source from which every
slot1 component is regenerated in Path A.

## Regeneration from raw (Path A)

`pipeline_full/runnable/retrain_all_from_raw.sh` regenerates every slot1
component from `data/raw/` and then runs the deterministic final assembly:

- **Track B (GPT) anchor** is regenerated from raw through the full phase3
  chain, including `build_phase3_submissions` (which writes the
  `phase3_current_state_v2.json` sidecar that `build_phase3_5_candidates`
  requires), `build_phase3_5_candidates`, `build_phase3_6_candidates`,
  `run_phase3_9_horizon`, and `run_phase3_10_horizon`.
- **Track A (Claude) anchor** is regenerated from raw via
  `phase2_stack_2way.py` (2-way OOF-stacked anchor).
- **Death CWGBSA / GBSA** survival models are regenerated from raw via
  `task3_tree_survival.py`.
- **The LS=12 gate and the 50/50 merge** are regenerated from the
  regenerated Track A / Track B anchors and the raw liver-stiffness column,
  using the verified upstream functions.
- **Final assembly** uses the `build_slot1_only.py` logic (LS=12 gate +
  q95 disagreement override → 50/50 merge fallback +
  `0.85·rank(CWGBSA) + 0.15·rank(GBSA)` death blend) over the freshly
  regenerated components — it does not read `cached_intermediates/`.

The final generated file
(`pipeline_full/runnable/retrain_outputs/final_retrain_prediction.csv`) is
**md5-identical to `frozen/slot1_prediction.csv`**
(`fb04658b99f89ec822e9c604d537dcae`), both endpoints rank-identical and
float-exact, in the tested environment. See `docs/RETRAINING.md`.

## Two verification scripts in this repo (Path B)

1. **`pipeline_full/runnable/build_slot1_only.py`** — the fast cached-output
   verification script. Slot1-only. Reads only cached prediction CSVs under
   `pipeline_full/runnable/cached_intermediates/`. No raw data, no
   `ANNITIA_DATA_ROOT`. The LS=12 gate is **read** from the cached gate
   prediction CSV; it is NOT recomputed from any liver-stiffness column.
   This is also the final-assembly logic reused by Path A.

2. **`pipeline_full/runnable/build_final_3.py`** — the historical
   multi-candidate producer that the sprint used to emit three
   `finalprobe_*` candidates. Slot1 is `finalprobe_3`. This script calls
   `load_labels()` unconditionally and so requires raw `train.csv` /
   `test.csv` to run end-to-end (even though the slot1 column values
   are mathematically determined by cached intermediates only). It is
   preserved here for traceability.

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

## Scope and caveats

- **Path A reproduces the submitted file byte-for-byte under the pinned
  environment.** `retrain_all_from_raw.sh` regenerates every component from
  raw and produces a file md5-identical to `frozen/slot1_prediction.csv`
  (`fb04658b99f89ec822e9c604d537dcae`) in the tested environment
  (`requirements_retrain.txt`). The hepatic anchors and death components are
  stochastic and library-version sensitive, so this byte-exactness holds
  under that pinned environment rather than across arbitrary versions.
- **The slot1-vs-alternative selection was public-leaderboard-informed.**
  `finalprobe_3` was chosen over OOF-superior `finalv2_LS14` via public-LB
  candidate pruning on a now-closed leaderboard. This caveat stands
  independently of the byte-for-byte reproduction above.

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
`pipeline_full/reference_upstream/`. Raw data (`data/raw/`) are required for
Path A (full from-raw retraining), for running those upstream scripts, and
for `build_final_3.py`; the fast cached-output verification
(`build_slot1_only.py`, Path B) does not need them.
