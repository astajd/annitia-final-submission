# Limitations

This document states the limitations of the submitted solution and its reproducibility, plainly.

## Reproducibility

- **Byte-exact reproduction is environment-pinned.** Full from-raw retraining (`bash pipeline_full/runnable/retrain_all_from_raw.sh`) regenerates the submitted file byte-for-byte (md5 `fb04658b99f89ec822e9c604d537dcae`) in the tested environment. Because the upstream survival models are stochastic and library-version sensitive, byte-exactness is verified under the pinned environment in `requirements_retrain.txt` (Python 3.11.4 and the listed package versions) with the orchestrator's thread caps applied; reruns under materially different library versions may differ. See `REPRODUCIBILITY.md` and `RETRAINING.md`.
- **Two verification paths exist.** Path A retrains everything from raw (~41–45 min); Path B (`build_slot1_only.py`) is a faster deterministic check from saved model-level outputs and needs no raw data or `ANNITIA_DATA_ROOT`. The historical multi-candidate producer `build_final_3.py` additionally loads raw data and generates non-submitted alternatives; it is retained for provenance, not as a reviewer entry point.
- **The public leaderboard is closed.** A regenerated alternative file cannot be scored against the leaderboard to confirm equivalence, which is why the submitted file is treated as the frozen authoritative artifact.

## Selection and modeling

- **The final candidate selection was public-leaderboard-informed.** Among several candidates, the submitted one was chosen with reference to the public leaderboard rather than purely by out-of-fold evaluation; an OOF-superior alternative was not submitted because it scored lower on the public leaderboard. The parameter choices within the recipe (q95 override, death blend, anchor weights) were cross-validation-selected; the liver-stiffness gate threshold was a clinical choice. See `LEAKAGE_AUDIT.md`.
- **The liver-stiffness gate trades a small OOF margin for clinical defensibility.** OOF evaluation marginally favored a higher threshold than the 12 kPa MASLD/NIT high-risk value that was used. (12 kPa is a clinically interpretable advanced-fibrosis threshold, not the Baveno VII CSPH rule-in value.)

## Data

- **The dataset is synthetic.** Feature-effect patterns reflect the data-generating process, not validated clinical associations. Clinical insights derived from the model (see `EXPLAINABILITY.md`) are framed accordingly and should not be read as causal or clinically established.
- **The challenge data are included for review under license.** The official files are present in `data/raw/` in this private review repository following organizer clarification (see `data/README.md`). If the repository is later made public, redistribution of the dataset should be reconfirmed with the data provider.

## Explainability scope

- **Feature attributions are post-hoc interpretations of the submitted models, not part of the modeling process.** Any attribution analysis in `EXPLAINABILITY.md` was produced to interpret the submitted models after the fact; it did not inform model construction or selection.
- **Per-component attributions do not directly explain the gated ensemble output.** The final hepatic rank is a gated selection between two anchors with a disagreement override, which is not an additive combination. Attributions computed on individual anchors or death models explain those components, not the non-additive ensemble output. This is discussed in `EXPLAINABILITY.md`.

## Future work

Interpretability and clinical validation are natural next steps that were out of scope for the challenge submission. Faithful attribution for the final hepatic prediction would require methods that account for the gating and override logic rather than per-model attribution, and any clinical claims would require validation on real rather than synthetic data.
