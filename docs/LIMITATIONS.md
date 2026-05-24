# Limitations

This document states the limitations of the submitted solution and its reproducibility, plainly.

## Reproducibility

- **Exact raw-to-submission retraining is not claimed.** The submitted file is reproduced rank-identically (and float-exactly) from saved model-level prediction outputs via a deterministic recipe, using `build_slot1_only.py`, which requires no raw data. It is not reproduced by retraining all models from the raw challenge data in a single command. See `REPRODUCIBILITY.md`.
- **The authoritative verification path needs only cached intermediates.** `build_slot1_only.py` isolates the submitted slot1 branch and runs with no raw data and no `ANNITIA_DATA_ROOT`. The historical multi-candidate producer `build_final_3.py` additionally loads raw data and generates non-submitted alternatives; it is retained for provenance, not as the reviewer entry point.
- **Upstream models are stochastic.** The track anchors (random survival forest and gradient-boosted Cox components for Track A; gradient-boosted survival models for Track B) and the death models are sensitive to library versions and training nondeterminism. Even with seeds pinned, reruns from raw data are not guaranteed to reproduce the cached outputs rank-identically.
- **The public leaderboard is closed.** A regenerated alternative file cannot be scored against the leaderboard to confirm equivalence, which is why the submitted file is treated as the frozen authoritative artifact.

## Selection and modeling

- **The final candidate selection was public-leaderboard-informed.** Among several candidates, the submitted one was chosen with reference to the public leaderboard rather than purely by out-of-fold evaluation; an OOF-superior alternative was not submitted because it scored lower on the public leaderboard. The parameter choices within the recipe (q95 override, death blend, anchor weights) were cross-validation-selected; the liver-stiffness gate threshold was a clinical choice. See `LEAKAGE_AUDIT.md`.
- **The liver-stiffness gate trades a small OOF margin for clinical defensibility.** OOF evaluation marginally favored a higher threshold than the 12 kPa MASLD/NIT high-risk value that was used. (12 kPa is a clinically interpretable advanced-fibrosis threshold, not the Baveno VII CSPH rule-in value.)

## Data

- **The dataset is synthetic.** Feature-effect patterns reflect the data-generating process, not validated clinical associations. Clinical insights derived from the model (see `EXPLAINABILITY.md`) are framed accordingly and should not be read as causal or clinically established.
- **Raw challenge data are not redistributed.** Reviewers wishing to inspect or rerun the reference-upstream code or the historical multi-candidate script must place the official files locally per `data/README.md`.

## Explainability scope

- **Feature attributions are post-hoc interpretations of the submitted models, not part of the modeling process.** Any attribution analysis in `EXPLAINABILITY.md` was produced to interpret the submitted models after the fact; it did not inform model construction or selection.
- **Per-component attributions do not directly explain the gated ensemble output.** The final hepatic rank is a gated selection between two anchors with a disagreement override, which is not an additive combination. Attributions computed on individual anchors or death models explain those components, not the non-additive ensemble output. This is discussed in `EXPLAINABILITY.md`.

## Future work

Interpretability and clinical validation are natural next steps that were out of scope for the challenge submission. Faithful attribution for the final hepatic prediction would require methods that account for the gating and override logic rather than per-model attribution, and any clinical claims would require validation on real rather than synthetic data.
