# ANNITIA Data Challenge — slot1 submission

This repository accompanies the slot1 submission to the ANNITIA / Trustii.io /
IHU ICAN data challenge: survival prediction in patients with metabolic
dysfunction-associated steatotic liver disease (MASLD) for two endpoints,
a hepatic-event endpoint and an all-cause-death endpoint. Submissions are
scored by a weighted concordance index (0.7 × C_hepatic + 0.3 × C_death),
which depends only on the ordering of predicted risks; the submitted file
emits rank-valued risks accordingly.

## Submitted file

- `frozen/slot1_prediction.csv` — 423 rows; columns `trustii_id`,
  `risk_hepatic_event`, `risk_death`.
- SHA-256: `5b0f1043f0b27addab5cc8dde33e2774371e9370c231633462f2f29337e57618`
  (recorded in `frozen/SHA256SUMS`).
- Verify with `bash frozen/verify.sh`.

## Quick verification

```bash
# 1. Verify the byte-frozen submission file.
bash frozen/verify.sh

# 2. Set up the verification environment.
cd pipeline_full/runnable
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Regenerate slot1 from cached intermediates (no raw data required).
unset ANNITIA_DATA_ROOT
python build_slot1_only.py
python validate_slot1_only.py
```

### Expected result

`validate_slot1_only.py` confirms that the regenerated
`generated_slot1_prediction.csv` is rank-identical to
`frozen/slot1_prediction.csv` for both `risk_hepatic_event` and
`risk_death`. The final line is:

```
SUCCESS — generated_slot1_prediction.csv is RANK-IDENTICAL to frozen/slot1_prediction.csv
```

Float-exact equality may additionally be reported as informational, but
the headline reproducibility claim is rank-identity, because the challenge
metric is rank-based.

## Reproducibility claim

- The submitted prediction file is reproduced **rank-identically** from
  saved model-level prediction outputs via a deterministic slot1
  verification script (`pipeline_full/runnable/build_slot1_only.py`).
- **Exact raw-data-to-submission retraining is not claimed.** The upstream
  hepatic anchors and death components are stochastic and library-version
  sensitive; reruns from raw data are not guaranteed to reproduce the
  cached outputs rank-identically.
- The upstream source code under `pipeline_full/reference_upstream/` is
  included for provenance and methodological review, not as a bit-exact
  retraining path for the submitted file.
- **Raw challenge data are not required** for authoritative slot1
  verification. The verification script reads only the cached intermediates
  under `pipeline_full/runnable/cached_intermediates/`.
- **Raw challenge data are not redistributed** in this repository. See
  `data/README.md` for hashes and placement instructions if a reviewer
  wishes to inspect or rerun the reference upstream code.

Full reproducibility details: `docs/REPRODUCIBILITY.md` and
`docs/PROVENANCE.md`.

## Repository structure

| path | contents |
|---|---|
| `frozen/` | byte-frozen submission file, SHA256SUMS, `verify.sh` |
| `pipeline_full/runnable/` | deterministic slot1 verification script, cached intermediates, lib, configs, requirements |
| `pipeline_full/reference_upstream/` | upstream Track A (Claude) and Track B (GPT) source code, read-only for reference |
| `docs/` | methodology, reproducibility, provenance, leakage audit, limitations, explainability |
| `reports/explainability/` | per-component attribution tables, structural notes, references, inventory, readiness report, CWGBSA death-model audit |
| `data/` | raw-data placement instructions (no data shipped) |

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

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — modeling approach and
  selection decisions for each parameter.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — what this repo
  reproduces, the authoritative verification path, and what is not claimed.
- [`docs/PROVENANCE.md`](docs/PROVENANCE.md) — submitted file, cached input
  provenance, and the deterministic slot1 recipe.
- [`docs/LEAKAGE_AUDIT.md`](docs/LEAKAGE_AUDIT.md) — exactly how each
  selection decision was made (cross-validation, clinical reasoning, or
  public leaderboard).
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — reproducibility, selection,
  data, and explainability limitations.
- [`docs/EXPLAINABILITY.md`](docs/EXPLAINABILITY.md) — post-hoc interpretation
  of the submitted ensemble, with citation-matched clinical wording.

## Data

The ANNITIA / Trustii.io / IHU ICAN dataset is licensed by the data
provider and **is not redistributed** in this repository. The authoritative
slot1 verification path (`bash frozen/verify.sh` followed by
`build_slot1_only.py` / `validate_slot1_only.py`) does **not** require raw
data.

Raw files are needed only for optional inspection or rerunning of the
upstream reference code under `pipeline_full/reference_upstream/`. In that
case, `data/README.md` documents the expected filenames, row counts, schema,
and placement directory.

## Sharing

This repository is intended to be shared **privately** with reviewers on
request — not made public by default.
