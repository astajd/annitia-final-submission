# Explainability

This document interprets the submitted slot1 solution after the fact. It did not influence model construction, model selection, or the submitted predictions; it is provided so a reviewer can understand why the ensemble produces the rankings it does, and where interpretation is and is not faithful. The submitted file `frozen/slot1_prediction.csv` is unchanged.

Two scope points are foregrounded throughout. The challenge dataset is synthetic, so any feature-effect pattern reflects the data-generating process rather than validated clinical biology, and no causal claim is made. And the final hepatic prediction is structurally interpretable but not additive, so per-component feature attributions explain the upstream anchors, not the gated ensemble output.

## Structural interpretability of the ensemble

The clearest explanation of the submitted solution is structural rather than feature-based, because the final recipe is a small, fully specified set of deterministic operations on the component model outputs.

For the hepatic endpoint, each patient is routed by their most recent liver-stiffness measurement: at or above 12 kPa the prediction comes from the Track A (Claude) anchor, below 12 kPa or with no measurement it comes from the Track B (GPT) anchor. A disagreement override then acts on the patients where the two anchors most strongly disagree: for the top 5% by normalized rank disagreement, the gated value is replaced by a 50/50 rank-average of the two anchors. The intuition is that when two independently developed models disagree most, a consensus is safer than committing to either. The final hepatic column is the rank transform of the result.

For the death endpoint, the prediction is a fixed rank blend, `rankdata(0.85 * rank(CWGBSA) + 0.15 * rank(GBSA))`, dominated by the componentwise model.

This structure is the most honest "explanation" of the submission: a reviewer can read the exact decision path for any patient (which anchor, whether overridden, how the death rank is composed) from `METHODOLOGY.md` and `PROVENANCE.md`, without relying on attribution heuristics.

## Hepatic endpoint

The hepatic risk is a gated selection between two independently developed survival-model ensembles. Track A is a 0.30/0.70 blend of a landmark random survival forest and a permissive rank-mean of random survival forest and XGBoost-Cox models. Track B is a greedy rank-blend across gradient-boosted models (LightGBM, CatBoost, XGBoost) trained at multiple prediction horizons.

Faithful per-feature attribution is limited on this endpoint, and this is stated rather than worked around. For Track A, no per-feature attribution is available for the submitted blend, because its fitted constituents were not preserved; the Track A contribution is therefore described structurally and by its model families rather than by a feature ranking. For Track B, the available attribution evidence comes from phase-2 proxy component models on the same feature families, not from the submitted phase-3 horizon-blend anchor whose fitted constituents were likewise not preserved. The Track B proxy aggregate is dominated by non-invasive liver-related and trajectory features (BMI slope, FIB-4, AST/ALT ratio, liver-stiffness summaries, FibroTest summaries), which is consistent with the modeling intent of using non-invasive-test features for hepatic risk, but it is included as approximate component context only.

Crucially, even where component attributions exist, they do not explain the final hepatic output. The gated ensemble is a discrete switch followed by a top-5% override to a consensus merge; standard additive attribution methods have no faithful decomposition under this structure. The per-component evidence should be read as describing each anchor in isolation, not as a global explanation of the gated rank.

## Death endpoint

The death endpoint is where post-hoc attribution is most feasible, and it produced the most important finding in this analysis, which is reported transparently.

The two components were interpreted by their model-native importances from a single seed-42 reference fit on the full training labels, using the same hyperparameters and 143-feature longitudinal-summary set as the cached predictions. No retraining of the full 5x10 cross-validation ensemble was performed, and the cached predictions are unchanged. The method is model-native because it is faithful and cheap for these model classes: scikit-survival does not expose a native survival-aware SHAP explainer, and forcing one would add engineering without improving fidelity over the sparse coefficients (CWGBSA) and impurity importances (GBSA) the models already provide.

The componentwise model (CWGBSA, weight 0.85) is intrinsically sparse: only two features receive nonzero coefficients, `Age_slope` (coefficient magnitude 0.53) and `Age_rel_delta` (0.38). Both are age-trajectory and follow-up-window features rather than disease-physiology markers. `Age_slope` effectively encodes whether a patient has two or more visits, and `Age_rel_delta` encodes the observed visit-window length. This must be read as model behavior on a synthetic survival dataset, not as a clinical mortality mechanism: on this cohort, the death endpoint is largely predictable from follow-up-duration-correlated features rather than from biomarker physiology. An existing internal audit (`reports/explainability/audit_CWGBSA_death_model.md`) corroborates this, showing that an age-features-only model matches the full model's out-of-fold C-index (0.9704 versus 0.9703) and that a single follow-up-duration feature reaches 0.969. This is disclosed openly; its leakage-risk implications are discussed in `LEAKAGE_AUDIT.md`, and the practical consequence (the death endpoint's strong discrimination would not transfer to a cohort with different follow-up structure) is recorded in `LIMITATIONS.md`.

The gradient-boosting model (GBSA, weight 0.15) is denser. Its top impurity importances are still dominated by age-trajectory summaries (`Age_delta`, `Age_std`, `Age_rel_delta`, `Age_count`, and related), with biomarker summaries such as FibroTest and liver-stiffness measures appearing lower (for example `fibrotest_BM_2_last` at rank 4). The same follow-up-duration caveat applies, with somewhat more biomarker contribution than CWGBSA. Impurity importances are biased toward high-cardinality continuous features and should be read as a relative ranking, not as effect sizes.

Because the final death rank is a linear combination of ranks, the CWGBSA structure propagates with weight 0.85 to the submitted column, and the GBSA component adds a small amount of biomarker-summary information on top of the dominant follow-up-duration signal.

## Clinical rationale for the 12 kPa gate

The 12 kPa gate threshold is a clinically interpretable choice, and its support is matched precisely to the literature. It is used here as a MASLD/non-invasive-test high-risk or advanced-fibrosis liver-stiffness threshold. Under the AASLD 2023 NAFLD practice guidance (Rinella et al., Hepatology 2023), vibration-controlled transient elastography liver stiffness below 8 kPa rules out advanced fibrosis, 8 to 12 kPa is indeterminate, and above 12 kPa is associated with a high likelihood of advanced fibrosis; the 12 kPa gate aligns with the upper boundary of that three-tier scheme.

This is explicitly not the Baveno VII threshold for clinically significant portal hypertension. Baveno VII (de Franchis et al., Journal of Hepatology 2022) uses a "rule of five" (10, 15, 20, 25 kPa) for risk stratification, with 25 kPa or above as the rule-in for clinically significant portal hypertension and 15 kPa or below (with platelets at or above 150) as rule-out. Baveno VII is cited here only for those values, not as support for the 12 kPa gate. The contemporaneous EASL-EASD-EASO 2024 MASLD guidelines place 12 kPa within the broader range used for MASLD non-invasive-test risk stratification. Full citation-to-claim mapping is in `reports/explainability/REFERENCES.md`.

Out-of-fold evaluation marginally favored a higher threshold; 12 kPa was retained for clinical defensibility, accepting a small out-of-fold cost knowingly, as recorded in `LEAKAGE_AUDIT.md` and `LIMITATIONS.md`.

## Limitations of this analysis

The dataset is synthetic, so feature-effect patterns reflect the data-generating process and no causal or clinically established claim is supported. This analysis is post-hoc and did not inform model construction, selection, or the submitted predictions. Component attributions explain individual models, not the non-additive gated hepatic ensemble, for which no faithful global attribution is provided. The longitudinal-summary feature set is highly correlated, so impurity importances and proxy SHAP rank aggregates are subject to known collinearity biases. And the dominant death component's discrimination rests on follow-up-duration-correlated features specific to this cohort, which would not transfer to data with a different follow-up structure.

## Source material

Per-component tables, the structural notes, the artifact inventory, the claim-matched references, and the readiness report are under `reports/explainability/`. The CWGBSA death-model audit is at `reports/explainability/audit_CWGBSA_death_model.md`.
