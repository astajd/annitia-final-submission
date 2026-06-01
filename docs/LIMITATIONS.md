# Limitations

This document states the limitations of the submitted solution and its reproducibility, plainly.

## Reproducibility

- **Byte-exact reproduction is environment-pinned.** Full from-raw retraining (`bash pipeline_full/runnable/retrain_all_from_raw.sh`) regenerates the submitted file byte-for-byte (md5 `fb04658b99f89ec822e9c604d537dcae`) in the tested environment. Because the upstream survival models are stochastic and library-version sensitive, byte-exactness is verified under the pinned environment in `requirements_retrain.txt` (Python 3.11.4 and the listed package versions) with the orchestrator's thread caps applied; reruns under materially different library versions may differ. See `REPRODUCIBILITY.md` and `RETRAINING.md`.
- **Two verification paths exist.** Path A retrains everything from raw (~41–45 min); Path B (`build_slot1_only.py`) is a faster deterministic check from saved model-level outputs and needs no raw data or `ANNITIA_DATA_ROOT`. The historical multi-candidate producer `build_final_3.py` additionally loads raw data and generates non-submitted alternatives; it is retained for provenance, not as a reviewer entry point.
- **The public leaderboard is closed.** A regenerated alternative file cannot be scored against the leaderboard to confirm equivalence, which is why the submitted file is treated as the frozen authoritative artifact.

## Selection and modeling

- **Candidate selection involved two separable decisions.** The liver-stiffness gate threshold (12 kPa) was a clinical choice: out-of-fold evaluation marginally favored an 18 kPa gate (the OOF-best alternative `finalprobe_1_lsthr18_disagq95_cw85gbsa15`, which differs from the submitted `finalprobe_3_lsthr12_disagq95_cw85gbsa15` only in that threshold) by about +0.0012 weighted-C; 12 kPa was retained for clinical defensibility, and the 18 kPa variant was OOF-confirmed only and never scored on the public leaderboard. Separately, the public leaderboard informed the primary submission among the candidates sharing the 12 kPa gate, which differ in the death blend and the disagreement override (not the gate threshold). The q95 override, the death blend, and the anchor weights were cross-validation-selected. See `LEAKAGE_AUDIT.md`.
- **The 12 kPa gate is a clinically grounded MASLD/NIT high-risk or advanced-fibrosis threshold.** It is used as a clinically interpretable liver-stiffness value, not the Baveno VII CSPH rule-in value (which is higher).
- **The death endpoint's strong discrimination rests on a follow-up-window signal.** The dominant death component (CWGBSA, weight 0.85) selects age-trajectory / follow-up-window features (`Age_slope`, `Age_rel_delta`) rather than biomarker physiology; an age-features-only model matches the full model's out-of-fold C-index (0.9704 vs 0.9703) and a single follow-up-duration feature reaches 0.969 (`reports/explainability/audit_CWGBSA_death_model.md`). On this synthetic cohort that signal is real and transfers to the test split, but the endpoint's discrimination may not transfer to clinical data with a different censoring / follow-up structure. See `EXPLAINABILITY.md` and `LEAKAGE_AUDIT.md`.

## Data

- **The dataset is synthetic.** Feature-effect patterns reflect the data-generating process, not validated clinical associations. Clinical insights derived from the model (see `EXPLAINABILITY.md`) are framed accordingly and should not be read as causal or clinically established.
- **The challenge data are included following organizer clarification.** The official files are present in `data/raw/` in this repository following organizer clarification (see `data/README.md`). Any further redistribution beyond this repository should be reconfirmed with the data provider.

## Explainability scope

- **Feature attributions are post-hoc interpretations of the submitted models, not part of the modeling process.** Any attribution analysis in `EXPLAINABILITY.md` was produced to interpret the submitted models after the fact; it did not inform model construction or selection.
- **Per-component attributions do not directly explain the gated ensemble output.** The final hepatic rank is a gated selection between two anchors with a disagreement override, which is not an additive combination. Attributions computed on individual anchors or death models explain those components, not the non-additive ensemble output. This is discussed in `EXPLAINABILITY.md`.

## Future work

Interpretability and clinical validation are natural next steps that were out of scope for the challenge submission. Faithful attribution for the final hepatic prediction would require methods that account for the gating and override logic rather than per-model attribution, and any clinical claims would require validation on real rather than synthetic data.
