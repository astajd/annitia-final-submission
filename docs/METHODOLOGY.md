# Methodology

## Objective

The challenge requires risk predictions for two endpoints in patients with metabolic dysfunction-associated steatotic liver disease (MASLD): a hepatic-event endpoint and an all-cause-death endpoint. Submissions are scored by a weighted concordance index (C-index) combining the two endpoints, with the hepatic endpoint weighted more heavily (0.7 hepatic, 0.3 death). Because the metric depends only on the ordering of predicted risks, the final outputs are expressed as ranks rather than calibrated probabilities or survival times.

## Overview

The final solution is an ensemble that models the two endpoints separately and combines outputs from two independently developed modeling tracks, referred to here as Track A (Claude track) and Track B (GPT track). The hepatic endpoint uses a clinically motivated gating strategy between the two track anchors, with a disagreement-based fallback. The death endpoint uses a fixed-weight blend of two gradient-boosted survival models. All final combination parameters are fixed, and the final assembly step is fully deterministic.

The two-track design was deliberate: each track developed its own preprocessing, feature engineering, cross-validation protocol, and survival models independently. Combining them at the end exploits the fact that independently constructed models tend to make partly uncorrelated errors, so a principled combination is more robust than either track alone.

## How selection decisions were made

Component-level model and blend-parameter selection was driven by out-of-fold (OOF) cross-validation. Two choices were exceptions, disclosed explicitly: the 12 kPa liver-stiffness gate threshold was a clinical choice (out-of-fold evaluation marginally favored a higher threshold), and the final candidate among several was selected with reference to the public leaderboard. The precise basis for each decision is tabulated in `LEAKAGE_AUDIT.md`.

The Track A models used 5x10 repeated stratified cross-validation (5 folds, 10 repeats), stratified by event indicator, with a base random seed of 42 and per-repeat seed offsets, aggregating across folds by mean-of-ranks. Repeated CV was chosen to reduce the variance of fold estimates, which matters when selecting between candidates whose true performance differs by small margins.

## Hepatic endpoint

The hepatic risk is constructed from two track-level anchor predictions, a liver-stiffness gate, and a disagreement override.

### Model families

The Track A (Claude) hepatic anchor is a survival-model ensemble combining a random survival forest and gradient-boosted Cox models (XGBoost survival objective); exact component lineage is given in `PROVENANCE.md`. The Track B (GPT) hepatic anchor is a rank-blend of gradient-boosted survival models. Both are genuine survival-model ensembles, not heuristic combinations.

### Liver-stiffness gate

Patients are routed by their most recent liver-stiffness measurement (LS) using a threshold of 12 kPa. Patients with LS at or above 12 kPa receive their hepatic risk from the Track A anchor; patients below 12 kPa, or with no available LS measurement, receive it from the Track B anchor.

The 12 kPa threshold is a clinically motivated and interpretable liver-stiffness boundary, used here as a MASLD/NIT high-risk or advanced-fibrosis threshold. It should not be interpreted as the Baveno VII rule-in threshold for clinically significant portal hypertension, which uses a higher liver-stiffness value. Out-of-fold evaluation marginally favored a higher threshold; 12 kPa was retained for clinical defensibility and for its stability across cross-validation and transfer checks. A small amount of measured OOF performance was deliberately traded for a clinically grounded boundary. The clinical rationale and references are in `EXPLAINABILITY.md`.

### Disagreement override

After gating, a disagreement override is applied. For each patient, the rank disagreement between the two track anchors is computed as the normalized absolute rank difference. Patients whose disagreement falls in the top 5% (the q95 quantile of the disagreement distribution) have their gated hepatic risk replaced by a 50/50 rank-average merge of the two anchors. The rationale: when two independently developed models disagree most strongly about a patient, committing to either single track is riskier than falling back to their consensus. The q95 threshold was selected by out-of-fold grid search over candidate quantiles (0.75, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99), not by leaderboard comparison.

## Death endpoint

The death risk is a fixed-weight blend of two gradient-boosted survival models, both trained on a longitudinal-summary feature set (143 features summarizing each patient's longitudinal record):

- A componentwise gradient-boosting survival model (CWGBSA): scikit-survival `ComponentwiseGradientBoostingSurvivalAnalysis`, Cox loss, 300 estimators, learning rate 0.05, subsample 0.8, features standardized.
- A gradient-boosting survival model (GBSA): scikit-survival `GradientBoostingSurvivalAnalysis`, Cox loss, 200 estimators, learning rate 0.05, maximum depth 3, subsample 0.8, max features sqrt.

Both used `random_state = 42 + repeat` under the 5x10 stratified-by-event CV protocol with mean-of-ranks aggregation. The final death risk is `0.85 * rank(CWGBSA) + 0.15 * rank(GBSA)`. The 0.85/0.15 weighting was selected by out-of-fold grid search over candidate ratios, not by leaderboard comparison; the componentwise model dominated in OOF evaluation, reflected in its larger weight.

## Final output and assembly

Both endpoint risks are converted to ranks for submission, consistent with the rank-based C-index objective. The submitted file contains `trustii_id`, `risk_hepatic_event`, and `risk_death`. The deterministic assembly of the submitted file from the saved model-level outputs is reproducible without raw data via `build_slot1_only.py`; see `REPRODUCIBILITY.md`.

## Candidate selection

Several final candidates were generated. The submitted candidate was selected with reference to the public leaderboard: an alternative configuration that scored marginally higher on out-of-fold evaluation was not chosen because it scored lower on the public leaderboard. This leaderboard-informed step is distinct from the parameter choices above (q95 override, death blend, anchor blend weights), which were cross-validation-selected. It is disclosed in `LEAKAGE_AUDIT.md` and `LIMITATIONS.md`.
