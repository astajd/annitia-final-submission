# Cached intermediates

These are saved model-level prediction outputs used in the submitted
ensemble. They are not raw challenge data and are not used to claim
raw-to-submission reproducibility. The upstream code that produced them is
included under `reference_upstream/` where available; exact regeneration
of these intermediates from raw data is not claimed.

The submitted slot1 file (see `../../../frozen/slot1_prediction.csv`) is
regenerated **rank-identically** from the files in this directory by
`../build_final_3.py`. See `../../../audit/` for the full provenance trace
and `../../../PROVENANCE_FINDINGS.md` for the four anchor/gate/death
findings that classify each upstream.

## Layout

```
cached_intermediates/
├── gpt_track_handoff/
│   ├── best_submissions/
│   │   └── 20260428_0059_phase3_10_horizon_blend_v2.csv     ← GPT hep anchor (TEST predictions)
│   └── oof_predictions/
│       ├── survival_models/
│       │   ├── 20260427_1306_phase3_6_no_visit_history__oof.csv
│       │   └── 20260427_1154_phase3_current_state_v2__oof.csv
│       └── horizon_components/
│           ├── NIT_plus_scores__hepatic__h1__lgbm_binary__oof.csv
│           ├── v3_hepatic_schema__hepatic__h3__lgbm_binary__s4__oof.csv
│           ├── current_state_v2__death__h5__catboost_binary__s3__oof.csv
│           └── NIT_plus_scores__death__h4__catboost_binary__oof.csv
├── claude_track_handoff/
│   ├── best_submissions/
│   │   └── phase2_blend_2way_optimal.csv                    ← Claude hep anchor (TEST predictions)
│   └── optional_oof_predictions/
│       ├── oof_blend_2way_optimal.csv
│       └── oof_death_xgb_longitudinal_summary.csv
├── merge_sprint/
│   └── submissions/
│       └── merge_A_best_50_50_both.csv                      ← 50/50 rank-blend of the two anchors
└── model_zoo_sprint/
    └── predictions/
        ├── oof__gate_LSthr12__hep.csv                       ← LS=12 gate OOF predictions
        ├── test__gate_LSthr12__hep.csv                      ← LS=12 gate TEST predictions
        ├── oof__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv
        ├── test__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv
        ├── oof__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv
        └── test__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv
```

## Provenance notes (per file class)

- **GPT hep anchor** (`gpt_track_handoff/best_submissions/...horizon_blend_v2.csv`):
  produced by `gpt/src/run_phase3_10_horizon.py`. Greedy rank-blend over
  horizon-model OOF/test components; weights pinned in the JSON sidecar
  (`_v3`: 0.7905…, `NIT_plus_scores__h1`: 0.1186, `v3_hepatic_schema__h3__s4`: 0.0909).
  Underlying LGBM/CatBoost/XGB component training is stochastic and library-version
  sensitive — rerunning from raw is not guaranteed bit/rank-identical. See
  PROVENANCE_FINDINGS.md section (a).

- **Claude hep anchor** (`claude_track_handoff/best_submissions/phase2_blend_2way_optimal.csv`):
  produced by `claude/experiments/phase2_stack_2way.py`. Weights
  `0.30 * rank(landmark_3y_RSF) + 0.70 * rank(permissive_ensemble_avg)`.
  Weight selected by argmax of OOF mean-minus-std on a 0.05 grid (5×10
  stratified CV), corroborated by honest-LOFO median 0.35. NOT LB-tuned.
  See PROVENANCE_FINDINGS.md section (b).

- **GPT OOF reconstructions** (`gpt_track_handoff/oof_predictions/...`):
  per-component OOF predictions used by `lib/zoo_utils.load_oof_baselines`
  to reconstruct an OOF version of the GPT anchor. The reconstruction
  uses ROUNDED proxy weights (0.7905/0.1186/0.0909) and is a proxy for
  the GPT anchor OOF, NOT the exact GPT anchor (which is the test-side
  CSV in `best_submissions/`). Used only for OOF-side gate computations
  inside the merged pipeline. See PROVENANCE_FINDINGS.md section (a).

- **Claude OOF predictions** (`claude_track_handoff/optional_oof_predictions/...`):
  per-source OOF predictions used by `lib/zoo_utils.load_oof_baselines` for
  the Claude side of the same merge.

- **`merge_sprint/submissions/merge_A_best_50_50_both.csv`**: produced by
  `upstream_scripts/build_sprint.py` —
  `0.5 * rank(GPT anchor) + 0.5 * rank(Claude anchor)` over both endpoints.

- **`model_zoo_sprint/predictions/{oof,test}__gate_LSthr12__hep.csv`**:
  produced by `upstream_scripts/task5_gated_ensemble.py`. Rule:
  `if LS_last >= 12 kPa: rank(Claude anchor) else rank(GPT anchor)`. Patients
  with no LS measurement fall through the inequality as False → GPT side.

- **`model_zoo_sprint/predictions/{oof,test}__survtree__dea__longitudinal_summary__{cwgbs_300_lr05,gbsa_200_lr05_d3}.csv`**:
  produced by `upstream_scripts/task3_tree_survival.py`. sksurv CWGBSA / GBSA
  on the Claude `longitudinal_summary` 143-column feature set, 5×10 stratified
  CV, per-repeat `random_state = 42 + repeat`, mean-of-ranks aggregation.
  Hyperparameters: CWGBSA (loss=coxph, n_estimators=300, lr=0.05,
  subsample=0.8, pre-scaled with StandardScaler); GBSA (loss=coxph,
  n_estimators=200, lr=0.05, max_depth=3, subsample=0.8, max_features=sqrt).
  Re-running from raw is subject to scikit-survival / scikit-learn
  version sensitivity — bit/rank-identical reproduction not claimed.
