---
title: "ANNITIA Data Challenge: Solution Report"
author: "Andrija Štajduhar"
date: "1 June 2026"
subject: "Survival risk prediction in MASLD: hepatic-event and all-cause-death endpoints"
---

*Repository: github.com/astajd/annitia-final-submission*

## 1. Executive summary

This report describes the final submission to the ANNITIA / Trustii.io /
IHU ICAN data challenge. The task is to rank patients with MASLD by risk for
two endpoints: major hepatic events and all-cause death. Submissions are
scored by a weighted concordance index (C-index), $0.7 \cdot C_{\mathrm{hepatic}} + 0.3 \cdot C_{\mathrm{death}}$, 
which depends only on the ordering of the
predicted risks. Because the metric is rank-based, the submitted predictions
are emitted as pure ranks rather than calibrated probabilities or survival
times.

The submitted solution is a deterministic ensemble that models the two
endpoints separately and combines two independently developed modeling tracks.
The hepatic endpoint uses a clinically motivated liver-stiffness gate between
the two track anchors, with a disagreement-based fallback. The death endpoint
uses a fixed-weight rank blend of two gradient-boosted survival models. Every
combination parameter is fixed and the final assembly is fully deterministic.

The submitted file (referred to in the repository as `slot1_prediction.csv`)
has 423 rows: one identifier column and two rank-valued risk columns. The
repository provides two verification paths against this frozen artifact: a full
from-raw retraining of every component, which reproduces the file byte-for-byte
under a pinned environment, and a fast cached-output check that regenerates the
submitted prediction from saved model-level outputs without raw data. Both have
been run in the submitted repository.

This report presents only documented results and verification facts. It makes
no claim about ranking, standing, or any final official score.

## 2. Challenge and evaluation

The challenge provides a longitudinal MASLD cohort with repeated clinical and
non-invasive-test (NIT) measurements per patient, together with two
time-to-event outcomes. The objective is to produce, for each test patient, a
risk score for hepatic events and a risk score for death such that higher
scores correspond to earlier/with-event patients under the C-index.

**Why rank-valued outputs.** The challenge evaluation uses the
scikit-survival `concordance_index_censored` implementation. This metric depends
on the ordering of predicted risks rather than their absolute calibration. The
submitted pipeline therefore optimizes and emits rank-aggregated quantities
throughout: blends are computed on `scipy.stats.rankdata` values and the final
columns are themselves ranks. Calibration is neither required nor attempted for
this metric.

**Data.** The challenge dataset is synthetic. It contains static features and
longitudinal `<variable>_v<i>` columns across repeated visits, plus the two
outcomes in the training file. The test file has the same feature schema and no
outcome columns. Because the data are synthetic, any feature-effect pattern
observed during interpretation reflects the data-generating process rather than
validated clinical biology; no causal claim is made anywhere in this work. The
official files are present under `data/raw/` in this repository
following organizer clarification (see `data/README.md`).

**Submitted artifact.** `frozen/slot1_prediction.csv` has 423 rows and the
required schema `trustii_id, risk_hepatic_event, risk_death`, with both risk
columns expressed as ranks. Its SHA-256 and MD5 checksums are recorded in
`frozen/SHA256SUMS`; the MD5 is also reproduced in the verification table in
Section 5.

A note on terminology: the repository refers to the two modeling tracks as
**Track A** and **Track B**. These are internal labels for two independently
developed modeling pipelines; the labels carry no meaning beyond that
separation and are used consistently in the code and documentation.

## 3. Methodology

### 3.1 Two-track design

The solution combines two modeling tracks that were developed independently,
each with its own preprocessing, feature engineering, cross-validation
protocol, and survival models. Combining them at the end exploits the fact that
independently constructed models tend to make partly uncorrelated errors, so a
principled combination is more stable than either track alone. The two tracks
supply *anchor* predictions for the hepatic endpoint; the death endpoint is
built from a separate pair of survival models.

### 3.2 Track anchors (hepatic)

- **Track A anchor.** A 0.30/0.70 blend of a landmark random survival forest
  component and a permissive rank-mean component that itself combines additional
  random survival forest and XGBoost-Cox models. The 0.30/0.70 blend weight was
  selected by cross-validation (argmax of a mean-minus-standard-deviation
  criterion on a 5×10 CV grid), not tuned on the leaderboard; the exact
  component lineage is in `docs/PROVENANCE.md`.
- **Track B anchor.** A rank-blend of gradient-boosted survival models
  (LightGBM, CatBoost, XGBoost) trained across multiple prediction horizons,
  with blend weights pinned in a metadata sidecar and selected from
  per-component out-of-fold (OOF) evaluation.

Both anchors are genuine survival-model ensembles rather than heuristic
combinations.

### 3.3 Hepatic endpoint recipe

The hepatic risk is assembled from the two anchors in three deterministic
steps:

1. **Liver-stiffness gate (12 kPa).** Each patient is routed by their most
   recent liver-stiffness (LS) measurement. Patients with LS ≥ 12 kPa take their
   hepatic risk from the Track A anchor; patients with LS < 12 kPa, or with no
   available LS measurement, take it from the Track B anchor.
2. **Disagreement override (q95).** For each patient the normalized absolute
   rank difference between the two anchors is computed,
   `d = |rank(Track B) − rank(Track A)| / N`. For the top 5% of patients by this
   disagreement (the q95 quantile), the gated value is replaced by a 50/50
   rank-average merge of the two anchors. The rationale is that where two
   independently developed models disagree most strongly, a consensus is safer
   than committing to either single track.
3. **Rank transform.** The result is converted to ranks for the final
   hepatic-event risk column.

The 12 kPa value is a clinically motivated, interpretable liver-stiffness
boundary used here as a MASLD/NIT high-risk or advanced-fibrosis threshold. It
is **not** the Baveno VII rule-in threshold for clinically significant portal
hypertension, which uses a higher liver-stiffness value (see Section 6 and
`reports/explainability/REFERENCES.md`). The clinical-versus-statistical basis
for choosing this threshold is detailed in Section 4.

### 3.4 Death endpoint recipe

The death risk is a fixed-weight rank blend of two gradient-boosted survival
models, both trained on a 143-feature longitudinal-summary feature set:

- **CWGBSA**: scikit-survival `ComponentwiseGradientBoostingSurvivalAnalysis`,
  Cox loss, 300 estimators, learning rate 0.05, subsample 0.8, standardized
  features.
- **GBSA**: scikit-survival `GradientBoostingSurvivalAnalysis`, Cox loss, 200
  estimators, learning rate 0.05, maximum depth 3, subsample 0.8,
  `max_features="sqrt"`.

Both were trained with `random_state = 42 + repeat` under a 5×10
stratified-by-event cross-validation protocol with mean-of-ranks aggregation.
The final death risk is

$$
\mathrm{risk}_{\mathrm{death}} =
\operatorname{rankdata}\left(
0.85 \cdot \operatorname{rank}(\mathrm{CWGBSA})
+
0.15 \cdot \operatorname{rank}(\mathrm{GBSA})
\right).
$$

The 0.85/0.15 weighting was selected by OOF grid search over candidate ratios,
not by leaderboard comparison; the componentwise model dominated in OOF
evaluation, which is reflected in its larger weight.

### 3.5 Final prediction architecture

The figure below summarizes the two endpoint pipelines as they run at execution
time. Every step shown is a fixed, deterministic operation; the selection
history behind these choices is discussed separately in Section 4.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.92\linewidth]{docs/figures/figure1.png}
\caption{Final prediction architecture. The hepatic endpoint uses a liver-stiffness gate followed by a q95 disagreement override; the death endpoint uses a fixed 0.85/0.15 rank blend.}
\end{figure}

### 3.6 Final assembly

Both endpoint risks are converted to ranks for submission, consistent with the
rank-based objective. The output is ordered by `trustii_id` ascending. The
final assembly logic (`build_slot1_only.py`) is the same code path used both for
the fast verification (Path B) and, over freshly regenerated components, for the
full from-raw retraining (Path A).

## 4. Selection discipline and leakage controls

A central aim of this submission is to be exact about *how* each decision was
made (by cross-validation, by clinical reasoning, or by reference to the public
leaderboard) rather than to present a uniformly clean narrative. The following
table summarizes the final-recipe decisions; the prose below it covers the one
decision that requires care.

| Decision | Selection method | Notes |
|---|---|---|
| q95 disagreement-override threshold | Cross-validation (OOF) | Grid search over quantiles {0.75, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99}; not leaderboard-tuned. |
| 0.85/0.15 death-model blend | Cross-validation (OOF) | Grid search over ratios; componentwise model dominated in OOF. |
| Track A anchor blend weight (0.30/0.70) | Cross-validation (OOF) | Argmax of mean−std on a 5×10 CV grid; leaderboard values appear only as post-selection reference metadata, not as inputs. |
| Track B anchor blend weights | Cross-validation (OOF) | Per-component OOF evaluation; weights pinned in a metadata sidecar. |
| Liver-stiffness gate threshold (12 kPa) | Clinical reasoning | The one final-recipe parameter not chosen by OOF optimization; see below. |
| Primary submission among LS=12 candidates | Public leaderboard | Choice among candidates that differ in death blend / override, not the gate threshold; see below. |

### 4.1 The gate threshold and the primary-candidate choice are two separable decisions

These two decisions are easy to conflate, so they are stated separately and
precisely.

**The 12 kPa gate threshold was a clinical choice.** Out-of-fold evaluation
marginally favored an 18 kPa gate, by about +0.0012 weighted-C. The OOF-best
candidate, referred to in the repository as `finalprobe_1`, is identical to the
submitted candidate, `finalprobe_3`, in every component except the gate
threshold. Both use the q95 disagreement override and the same 0.85/0.15 death
blend; the only difference is the gate threshold, 18 kPa instead of 12 kPa. The
12 kPa threshold was retained for clinical defensibility, knowingly accepting
that small out-of-fold cost. The 18 kPa variant was OOF-confirmed only and was
never scored on the public leaderboard.

**The primary submission was selected within the LS=12 family using the public
leaderboard.** Among the submitted candidates (all sharing the 12 kPa gate and
differing only in the death blend and the disagreement override, not in the gate
threshold), the public leaderboard informed which configuration was submitted as
primary. The submitted candidate `finalprobe_3` had a documented public-LB score
of 0.92198 on the now-closed board. This leaderboard step did not concern the
gate threshold.

In short: most parameters were OOF-selected, the gate threshold was a deliberate
clinical choice (over an OOF-best 18 kPa gate that was never leaderboard-scored),
and the public leaderboard informed only the primary-submission choice within the
LS=12 candidate family. The 0.92198 value is reported here strictly as that
candidate's public-leaderboard score in the selection/provenance context; it is
not a final official score, rank, or standing.

### 4.2 Feature and target leakage controls

- **No private or hidden labels.** No private leaderboard labels or held-out
  outcome information were used at any point.
- **No manual patient-level intervention.** No predictions were hand-edited,
  reordered, or patched against any reference file. The submitted file is the
  deterministic output of the assembly recipe.
- **Leakage-prone feature constructions were audited and excluded.** During
  development, time-aligned and censoring-aware feature variants were tested.
  Variants that encoded target timing or post-outcome information produced
  implausibly strong internal validation but did not generalize, and were
  excluded from the final selected solution.
- **Cached intermediates are predictions, not data.** The cached files consumed
  by the fast verification path are saved model-level prediction outputs, not raw
  challenge data, and are labeled as such.

One model-behavior finding on the death endpoint warrants disclosure as a
generalization caveat rather than as concealed leakage; it is described in
Section 6 and revisited in Section 7. Full detail is in `docs/LEAKAGE_AUDIT.md`.

## 5. Reproducibility

The frozen submission file is the authoritative artifact, and the repository
provides two independent ways to verify it. Both have been run in the submitted
repository.

### 5.1 Path A: full raw-data retraining

```bash
bash pipeline_full/runnable/retrain_all_from_raw.sh
```

This single command retrains **every** model component from the raw challenge
data in `data/raw/` (the Track A and Track B hepatic anchors, the 12 kPa gate,
the 50/50 merge, and the CWGBSA / GBSA death models), and then runs the
deterministic final assembly over the freshly regenerated components. It does
not read the cached intermediates as model inputs.

In the submitted repository, the full retraining output is **byte-for-byte
identical** to the frozen submitted file. The verification result is summarized
below.

| Check              | Result                                            |
| ------------------ | ------------------------------------------------- |
| Submitted file     | `frozen/slot1_prediction.csv`                     |
| Rows               | 423                                               |
| MD5 match          | `fb04658b99f89ec822e9c604d537dcae`                |
| Endpoint agreement | rank-identical and float-exact for both endpoints |
| Runtime            | approximately 41 minutes on the tested host       |

**Byte-exactness is environment-pinned.** It is verified under the pinned
environment recorded in the retraining requirements file (Python 3.11.4; numpy
2.4.4, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.6.1, scikit-survival 0.27.0,
xgboost 3.2.0, lightgbm 4.6.0, catboost 1.2.10, PyYAML 6.0.3) with the
orchestrator's thread caps applied. The upstream survival models are stochastic
and library-version sensitive, so byte-exact reproduction is claimed under this
pinned environment; reruns under materially different library versions may
produce close but not byte-identical outputs. The thread caps are functional,
not cosmetic: without them the survival forests over-subscribe threads and
thrash. Stage-by-stage detail is in the retraining documentation
(`docs/RETRAINING.md`).

### 5.2 Path B: fast cached-output verification

Path B is the fast cached-output verification. It runs `frozen/verify.sh`,
rebuilds the submitted prediction with `build_slot1_only.py`, and checks it with
`validate_slot1_only.py`. It does not retrain models and does not read raw data.
The rebuild applies the full assembly recipe (the 12 kPa gate read from cache, the
q95 disagreement override with 50/50 fallback, and the 0.85/0.15 death blend) to
the saved model-level outputs, and the verifier confirms that the regenerated
prediction is rank-identical to the frozen submitted file. This path consumes
only the cached prediction outputs and is independent of whether `data/raw/` is
present.

### 5.3 Relationship between the two paths and the closed leaderboard

Path A demonstrates that the submitted file can be regenerated from raw data;
Path B is a fast integrity check on the deterministic assembly recipe. Because
the public leaderboard is now closed, a regenerated alternative cannot be
re-scored on it, which is precisely why the frozen submitted file is treated as
the authoritative artifact and all reproduction claims are stated relative to
it.

## 6. Explainability

### 6.1 Structural interpretation is the most faithful explanation

The clearest explanation of the submission is structural, because the final
recipe is a small, fully specified set of deterministic operations on component
model outputs. For any patient a reviewer can read the exact decision path: which
anchor supplied the hepatic value (via the 12 kPa gate), whether that value was
overridden (top-5% disagreement → 50/50 merge), and how the death rank was
composed (0.85 CWGBSA + 0.15 GBSA). This structural account does not rely on
attribution heuristics and is given in full in Sections 3 and 4.

A consequence of this structure is that per-component feature attributions do
**not** explain the final hepatic output. The gated hepatic ensemble is a
discrete switch followed by a top-5% override to a consensus merge, not an
additive combination, so standard additive attribution methods have no faithful
decomposition of it. Attributions computed on the individual anchors describe
those anchors in isolation, not the gated rank. This is stated rather than worked
around.

### 6.2 The 12 kPa gate: clinical grounding and the Baveno disclaimer

The 12 kPa gate is a clinically interpretable threshold, with its wording matched
to the literature. It is used here as a MASLD/NIT high-risk or advanced-fibrosis
liver-stiffness threshold: under the AASLD 2023 NAFLD practice guidance,
vibration-controlled transient elastography liver stiffness below 8 kPa rules out
advanced fibrosis, 8–12 kPa is indeterminate, and above 12 kPa is associated with
a high likelihood of advanced fibrosis. The 12 kPa gate aligns with the upper
boundary of that three-tier scheme.

This is explicitly **not** the Baveno VII threshold for clinically significant
portal hypertension. Baveno VII uses a "rule of five" (10, 15, 20, 25 kPa), with
≥25 kPa as the rule-in for clinically significant portal hypertension; it is
cited only for those values, not as support for the 12 kPa gate. The
citation-to-claim mapping is in `reports/explainability/REFERENCES.md`. As noted
in Section 4, out-of-fold evaluation marginally favored a higher threshold and
12 kPa was retained for clinical defensibility.

### 6.3 The death endpoint: a follow-up-window signal, disclosed

The death endpoint is where post-hoc attribution is most feasible, and it
produced the most important finding in this analysis. The two death components
were interpreted by their model-native importances (sparse coefficients for
CWGBSA, impurity importances for GBSA) from a single seed-42 reference fit;
no retraining of the cross-validation ensemble was done and the cached
predictions are unchanged.

The componentwise model (CWGBSA, weight 0.85) is intrinsically sparse: only two
features receive nonzero coefficients, `Age_slope` (coefficient magnitude 0.53)
and `Age_rel_delta` (0.38). Both are age-trajectory and follow-up-window features
rather than disease-physiology markers: `Age_slope` effectively encodes whether
a patient has two or more visits, and `Age_rel_delta` encodes the observed
visit-window length. This must be read as model behavior on a synthetic survival
dataset, not as a clinical mortality mechanism: on this cohort, the death endpoint
is largely predictable from follow-up-duration-correlated features rather than
from biomarker physiology.

An internal audit (`reports/explainability/audit_CWGBSA_death_model.md`)
corroborates this: an age-features-only model matches the full model's
out-of-fold C-index (**0.9704 versus 0.9703**), and a single follow-up-duration
feature reaches **0.969**. This is not concealed target leakage in the sense of
directly using event or censoring columns as features; it is a synthetic-cohort
follow-up-window signal that is informative here and that transfers to the test
split. Its practical consequence, that this discrimination may not transfer to a
cohort with a different follow-up structure, is recorded as a limitation in
Section 7.

The gradient-boosting model (GBSA, weight 0.15) is denser. Its top impurity
importances are still dominated by age-trajectory summaries, with biomarker
summaries such as FibroTest and liver-stiffness measures appearing lower. The
same follow-up-window caveat applies, with somewhat more biomarker contribution
than CWGBSA. Impurity importances are biased toward high-cardinality continuous
features and should be read as a relative ranking, not as effect sizes.

All explainability material is post-hoc: it interprets the submitted models after
the fact and did not influence model construction, model selection, or the
submitted predictions. The supporting tables, scripts, and references are under
`reports/explainability/`.

## 7. Limitations

- **Synthetic data; no causal clinical claims.** The dataset is synthetic, so
  feature-effect patterns reflect the data-generating process rather than
  validated clinical associations. None of the interpretation in Section 6 should
  be read as causal or clinically established.
- **Byte-exact reproduction is environment-pinned.** Path A reproduces the
  submitted file byte-for-byte under the pinned environment in
  `requirements_retrain.txt` with the thread caps applied; the survival models
  are stochastic and library-version sensitive, so reruns under materially
  different versions may differ.
- **Primary-slot selection was public-leaderboard-informed (within the LS=12
  family).** The submitted candidate was chosen with reference to the public
  leaderboard among candidates that share the 12 kPa gate and differ only in the
  death blend and the disagreement override. The within-recipe parameters (q95
  override, death blend, anchor weights) were cross-validation-selected, and the
  gate threshold was a clinical choice over an OOF-best 18 kPa variant that was
  never leaderboard-scored.
- **Architecture-level choices were not exhaustively ablated.** The final hepatic
  recipe combines a liver-stiffness gate, a disagreement override, and rank
  merging. Individual parameters were documented and selected by cross-validation,
  clinical reasoning, or public-leaderboard comparison as described above, but the
  broader architecture was not presented as the result of an exhaustive search
  over all possible ensemble structures.
- **The death endpoint's discrimination rests on a follow-up-window signal.** As
  shown in Section 6, the dominant death component is driven by follow-up-window
  / age-trajectory features (`Age_slope`, `Age_rel_delta`); an age-features-only
  model matches the full model's OOF C-index (0.9704 vs 0.9703) and a single
  follow-up-duration feature reaches 0.969. On this synthetic cohort the signal
  is real and transfers to the test split, but the endpoint's strong
  discrimination may not transfer to clinical data with a different censoring /
  follow-up structure.
- **The closed leaderboard.** No regenerated alternative can be re-scored on the
  now-closed public leaderboard, which is why the frozen file is the authoritative
  artifact.
- **Revalidation before any clinical use.** Faithful attribution for the gated
  hepatic prediction would require methods that account for the gating and
  override logic rather than per-model attribution, and any clinical claim would
  require validation on real rather than synthetic data. Both are out of scope for
  this challenge submission.

## 8. Conclusion

The final submission is a deterministic two-track ensemble: a clinically gated,
disagreement-aware combination of two independently developed survival-model
anchors for the hepatic endpoint, and a fixed-weight rank blend of two
gradient-boosted survival models for the death endpoint. Selection discipline is
documented decision by decision: most parameters cross-validation-selected, the
12 kPa gate a deliberate clinical choice, and the public leaderboard used only to
pick the primary submission within the LS=12 candidate family. The submitted file
is verifiable two ways: a full from-raw retraining that reproduces it
byte-for-byte under a pinned environment, and a fast cached-output check that
confirms the deterministic recipe without raw data. The principal interpretive
finding, that the death endpoint's discrimination rests on a synthetic-cohort
follow-up-window signal, is disclosed openly, along with its generalization
limit. The intent throughout is an honest, reproducible, and transparently
documented submission.

Full methodological, provenance, reproducibility, leakage-audit, explainability,
and limitation details are provided in the repository documentation under
`docs/`, with citation-to-claim mapping under
`reports/explainability/REFERENCES.md`.
