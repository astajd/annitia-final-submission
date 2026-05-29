# Leakage Audit

This document records the steps taken to avoid information leakage, and states precisely how each selection decision was made (cross-validation, clinical reasoning, or public leaderboard). The aim is to be exact about provenance rather than to present a uniformly clean narrative.

## How each decision was made

The distinction that matters for a fair evaluation is whether a choice was driven by out-of-fold (OOF) cross-validation, by clinical reasoning, or by the public leaderboard.

- **q95 disagreement-override threshold: cross-validation.** Selected by OOF grid search over candidate quantiles (0.75, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99). Not leaderboard-tuned.
- **0.85/0.15 death-model blend: cross-validation.** Selected by OOF grid search over candidate ratios. Not leaderboard-tuned.
- **Track A (Claude) anchor blend weight (0.30/0.70): cross-validation.** Selected as the argmax of a mean-minus-standard-deviation criterion over a 0.05-spaced weight grid, under 5x10 stratified CV, with a pre-registered parity-band and minimum-improvement rule; an independent leave-one-feature-out check corroborated the selected weight. Public leaderboard values appear in the producing script (`phase2_stack_2way.py`) only as post-selection reference metadata, written after the weight was fixed; they do not enter the grid, the objective, the selection logic, or the prediction functions. The anchor's prediction-generation code was separately verified to contain no leaderboard input.
- **Track B (GPT) anchor blend weights: cross-validation.** The anchor is a rank-blend with weights pinned in a metadata sidecar, selected from per-component OOF evaluation.
- **Liver-stiffness gate threshold (12 kPa): clinical reasoning.** A clinically motivated choice aligned with a clinically interpretable MASLD/NIT high-risk or advanced-fibrosis liver-stiffness threshold (not the Baveno VII CSPH rule-in threshold, which uses a higher value). OOF evaluation marginally favored a higher threshold; 12 kPa was retained for clinical defensibility, with a small OOF cost accepted knowingly. This is the one final-recipe parameter not chosen by OOF optimization.
- **Final candidate (slot) selection: public leaderboard.** Among several generated candidates, the submitted one was chosen with reference to the public leaderboard; an OOF-superior alternative was not submitted because it scored lower on the public leaderboard. This is the leaderboard-informed step and is disclosed here, in `PROVENANCE.md`, and in `LIMITATIONS.md`.

Stating this split openly is the point: most parameters were OOF-selected, the gate was a deliberate clinical choice, and the final candidate selection used the public leaderboard.

## Feature and target leakage controls

- **No use of private or hidden labels.** No private leaderboard labels or held-out outcome information were used at any point.
- **No manual patient-level intervention.** No predictions were hand-edited, reordered, or patched against any reference file. The submitted file is the deterministic output of the assembly recipe.
- **Audited and excluded leakage-prone feature constructions.** During development, time-aligned and censoring-aware feature variants were tested. Variants that encoded target timing or post-outcome information produced implausibly strong internal validation but did not generalize, and were excluded from the final selected solution. They are not part of the submitted recipe.
- **Outputs are model predictions, not data.** The cached intermediates consumed by the final assembly are saved model-level prediction outputs, not raw challenge data, and are labeled as such (see `pipeline_full/runnable/cached_intermediates/`).

## Note on the synthetic dataset

The challenge dataset is synthetic. Any feature-effect patterns observed (see `EXPLAINABILITY.md`) reflect the structure of the data-generating process rather than validated clinical associations, and should not be interpreted as causal or clinically established effects.

## Note on the closed leaderboard

Because the public leaderboard is now closed, no regenerated alternative file can be scored to confirm equivalence to the submitted file. This is why the frozen submitted file (`frozen/slot1_prediction.csv`) is treated as the authoritative artifact and the reproduction claim is made relative to it. The repository provides two verification paths against that artifact: full from-raw retraining via `retrain_all_from_raw.sh` (byte-for-byte identical under the pinned environment) and fast cached-output verification via `build_slot1_only.py` (rank-identical from saved model-level outputs, without raw data). See `REPRODUCIBILITY.md` and `RETRAINING.md`.
