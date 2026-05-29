# ANNITIA Data Challenge — slot1 submission

This is the **ANNITIA slot1 submission repository** for the ANNITIA /
Trustii.io / IHU ICAN data challenge: survival prediction in patients with
metabolic dysfunction-associated steatotic liver disease (MASLD) for two
endpoints, a hepatic-event endpoint and an all-cause-death endpoint.
Submissions are scored by a weighted concordance index
(0.7 × C_hepatic + 0.3 × C_death), which depends only on the ordering of
predicted risks; the submitted file emits rank-valued risks accordingly.

**Submitted file: `frozen/slot1_prediction.csv`** — 423 rows; columns
`trustii_id`, `risk_hepatic_event`, `risk_death`. SHA-256
`5b0f1043f0b27addab5cc8dde33e2774371e9370c231633462f2f29337e57618`
(recorded in `frozen/SHA256SUMS`); MD5 `fb04658b99f89ec822e9c604d537dcae`.

## Two verification paths

Both paths have been run in this submitted repository.

### Path A — full raw-data retraining

```bash
bash pipeline_full/runnable/retrain_all_from_raw.sh
```

Retrains **every** slot1 component from the raw challenge data in `data/raw/`
and regenerates the submitted prediction file **byte-for-byte** in the tested
environment. The generated
`pipeline_full/runnable/retrain_outputs/final_retrain_prediction.csv` has
whole-file MD5 `fb04658b99f89ec822e9c604d537dcae` — identical to
`frozen/slot1_prediction.csv` (both endpoints rank-identical and float-exact,
Spearman ρ = 1.0, max abs diff = 0.0). Expected runtime is about **41–45
minutes** on the tested host. Use the pinned environment in
`pipeline_full/runnable/requirements_retrain.txt`. See `docs/RETRAINING.md`.

### Path B — fast cached-output verification

```bash
bash frozen/verify.sh
cd pipeline_full/runnable
python build_slot1_only.py
python validate_slot1_only.py
```

A faster deterministic check that regenerates slot1 from saved model-level
outputs (no retraining). `validate_slot1_only.py` confirms the regenerated
`generated_slot1_prediction.csv` is rank-identical to
`frozen/slot1_prediction.csv` for both endpoints, ending with:

```
SUCCESS — generated_slot1_prediction.csv is RANK-IDENTICAL to frozen/slot1_prediction.csv
```

Set up the environment first with
`pip install -r pipeline_full/runnable/requirements.txt`.

## Reproducibility status

- **Full raw-data retraining is implemented and verified in this repo.**
  Path A retrains every slot1 component from raw and produces a file
  byte-identical to `frozen/slot1_prediction.csv`
  (MD5 `fb04658b99f89ec822e9c604d537dcae`) in the tested environment.
- **Byte-exactness is environment-pinned.** It is verified under the
  pinned/tested environment in `requirements_retrain.txt`; reruns under
  materially different library versions may differ.
- **Raw challenge data are included** in `data/raw/` following organizer
  clarification (see `data/README.md`).
- **Path B is the fast deterministic check** from saved model-level outputs;
  it needs only the files under
  `pipeline_full/runnable/cached_intermediates/`.

Full details: `docs/RETRAINING.md`, `docs/REPRODUCIBILITY.md`, and
`docs/PROVENANCE.md`.

## Repository structure

| path | contents |
|---|---|
| `frozen/` | byte-frozen submission file, SHA256SUMS, `verify.sh` |
| `pipeline_full/runnable/` | from-raw retraining orchestrator (`retrain_all_from_raw.sh`, `retrain_lib/`), deterministic slot1 verification script, cached intermediates, lib, configs, requirements |
| `pipeline_full/reference_upstream/` | upstream Track A (Claude) and Track B (GPT) source code, read-only for reference |
| `docs/` | retraining, methodology, reproducibility, provenance, leakage audit, limitations, explainability |
| `reports/explainability/` | per-component attribution tables, structural notes, references, inventory, readiness report, CWGBSA death-model audit |
| `data/raw/` | official challenge data (included in this private review repo per organizer clarification) |

## Method summary

**Hepatic endpoint.** Two independently developed track anchors:

- Track A (Claude): a survival-model ensemble combining a random survival
  forest and gradient-boosted Cox models (XGBoost survival objective).
- Track B (GPT): a rank-blend of gradient-boosted survival models across
  multiple prediction horizons.

Patients are routed by their most recent liver-stiffness measurement using
a 12 kPa gate (LS ≥ 12 kPa → Track A; LS < 12 kPa or missing → Track B).
A disagreement override then identifies the top 5% of patients (q95 of the
normalized rank disagreement between the two anchors) and replaces the
gated value with a 50/50 rank-average merge. The final hepatic column is
the rank transform of the result. The 12 kPa gate is a clinically
interpretable MASLD / non-invasive-test high-risk threshold and is **not**
the Baveno VII rule-in threshold for clinically significant portal
hypertension, which uses a higher liver-stiffness value.

**Death endpoint.** A fixed-weight rank blend of two gradient-boosted
survival models on a 143-feature longitudinal-summary feature set:

```
risk_death = rankdata( 0.85 · rank(CWGBSA) + 0.15 · rank(GBSA) )
```

CWGBSA is `sksurv.ensemble.ComponentwiseGradientBoostingSurvivalAnalysis`;
GBSA is `sksurv.ensemble.GradientBoostingSurvivalAnalysis`.

**Outputs.** Both endpoint risks are emitted as pure ranks
(`scipy.stats.rankdata`), consistent with the rank-based C-index objective.

## Documentation

- [`docs/RETRAINING.md`](docs/RETRAINING.md) — Path A: the full from-raw
  retraining command, stages, environment, and byte-exact result.
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — modeling approach and
  selection decisions for each parameter.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — what this repo
  reproduces (both verification paths) and the environment caveats.
- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — submitted file, raw data,
  the from-raw regeneration chain, and the deterministic slot1 recipe.
- [`docs/LEAKAGE_AUDIT.md`](docs/LEAKAGE_AUDIT.md) — exactly how each
  selection decision was made (cross-validation, clinical reasoning, or
  public leaderboard).
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — reproducibility, selection,
  data, and explainability limitations.
- [`docs/EXPLAINABILITY.md`](docs/EXPLAINABILITY.md) — post-hoc interpretation
  of the submitted ensemble, with citation-matched clinical wording.

## Data

The official ANNITIA / Trustii.io / IHU ICAN challenge data are **included**
in this private review repository under `data/raw/`, following organizer
clarification. They are required for Path A (full from-raw retraining) and
for the upstream reference code under `pipeline_full/reference_upstream/`.
Path B (`bash frozen/verify.sh` followed by `build_slot1_only.py` /
`validate_slot1_only.py`) does **not** require raw data.

`data/README.md` documents the filenames, row counts, schema, and loader
resolution order. If this repository is later made public, redistribution of
the dataset should be reconfirmed with the data provider.

## Sharing

This repository is intended to be shared **privately** with reviewers on
request — not made public by default.
