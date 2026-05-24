# Audit — CWGBSA death model

**Model:** sksurv `ComponentwiseGradientBoostingSurvivalAnalysis`,
loss=coxph, n_estimators=300, lr=0.05, subsample=0.8, random_state=42.
Trained on Claude `longitudinal_summary` features (143 cols),
StandardScaler-normalised. Saved OOF dea C-index 0.9703.

## 1. Per-fold dea C-index (deterministic fold reconstruction)

Reconstructing the saved OOF onto Claude 5×10 fold structure
(StratifiedKFold on hep-event, seed=42+repeat). **Per-fold C-index
computed on validation indices using the aggregated OOF (mean of
ranks across repeats).**

| metric | value |
|---|---:|
| mean | 0.9718 |
| std | 0.0155 |
| min (worst fold) | 0.9247 (repeat=2, fold=3) |
| max (best fold) | 0.9964 |
| 25th / 50th / 75th pct | 0.9620 / 0.9734 / 0.9844 |

| repeat | mean dea C across 5 folds |
|---:|---:|
| 0 | 0.9740 |
| 1 | 0.9716 |
| 2 | 0.9694 |
| 3 | 0.9709 |
| 4 | 0.9705 |
| 5 | 0.9721 |
| 6 | 0.9715 |
| 7 | 0.9714 |
| 8 | 0.9751 |
| 9 | 0.9715 |

(Per-repeat means cluster tightly between 0.969 and 0.975.)

CSV: `logs/audit_cwgbsa_perfold.csv`.

## 2. Held-out 80/20 sanity check (5 seeds)

Stratified split on death event, train on 80 %, evaluate on 20 %, repeat
across seeds 42–46.

| seed | n_va | dea events | held-out C |
|---:|---:|---:|---:|
| 42 | 251 | 15 | 0.9849 |
| 43 | 251 | 15 | 0.9627 |
| 44 | 251 | 15 | 0.9757 |
| 45 | 251 | 15 | 0.9855 |
| 46 | 251 | 15 | 0.9564 |

Mean **0.9730**, std 0.0131, min 0.9564, max 0.9855.

Consistent with the CV mean of 0.9718. CSV: `logs/audit_cwgbsa_holdout.csv`.

## 3. Feature leakage audit — clean on label fields

Suspicious term scan over all 143 column names:
`event`, `death`, `occur`, `censor`, `label`, `time_to`, `outcome`,
`endpoint`, `evenement`, `majeur`, `hepatique`.

**Result: 0 columns flagged.** No event/death/occur/censor/label/
time-to-event columns reach the model.

Feature breakdown:

| group | count | examples |
|---|---:|---|
| Age-derived | 10 | Age_first, Age_last, Age_mean, Age_min, Age_max, Age_std, Age_delta, Age_rel_delta, Age_slope, Age_count |
| Static | 7 | gender, T2DM, Hypertension, Dyslipidaemia, bariatric_surgery, bariatric_surgery_age, bariatric_surgery_done |
| Visit-cadence (`*_count`) | 13 | Age_count, BMI_count, alt_count, ast_count, plt_count, ggt_count, … |
| Biomarker summaries | 114 | BMI_first, BMI_last, BMI_max, BMI_slope, alt_first, alt_slope, … |

## 4. Selected components — CRITICAL FINDING

CWGBSA on full train selects **only 2 features with nonzero
coefficients**:

| coefficient | feature |
|---:|---|
| **−0.5339** | **`Age_slope`** |
| **−0.3839** | **`Age_rel_delta`** |

(All 141 other features have coefficient = 0.)

CSV: `logs/audit_cwgbsa_coefs.csv`.

### What these features actually encode

| feature | computation | what it measures |
|---|---|---|
| `Age_slope` | `polyfit(Age_v*, Age_v*, 1)[0]` — degenerate, always 1.0 with ≥2 visits, NaN→0.0 with <2 visits | **single-visit indicator**: 1 if patient has ≥2 Age_v* observations, 0 otherwise |
| `Age_rel_delta` | `(Age_last - Age_first) / Age_first` | **relative observed follow-up duration** |

These are both **follow-up duration / attrition proxies**, not disease
physiology features.

## 5. Ablation — does the gain come from the Age trajectory?

Re-trained CWGBSA under Claude 5×10 CV with the same hyperparameters,
varying the feature set:

| feature set | n_features | dea OOF | fold std |
|---|---:|---:|---:|
| **full** (`longitudinal_summary`) | 143 | **0.9703** | 0.012 |
| **only Age_*** (10 features) | 10 | **0.9704** | 0.012 |
| no Age_* | 133 | 0.8402 | 0.047 |
| no Age_* and no `*_count` | 121 | 0.7433 | 0.074 |

**The full 143-feature model achieves an identical OOF C-index to a
10-feature Age-only model.** Removing the Age trajectory features
collapses the model to OOF 0.84 (a ~0.13 drop), and removing Age +
visit-cadence collapses it further to 0.74. CSV: `logs/audit_cwgbsa_ablation.csv`.

## 6. Univariate diagnostic — Age_* features alone

C-index of each Age feature, used as a univariate dea risk score
(both directions tested):

| feature | C-index (positive direction) | C-index (negative direction) |
|---|---:|---:|
| `Age_first` (= Age_v1) | 0.7953 | 0.2047 |
| `Age_last` (= max observed visit age) | 0.6421 | 0.3579 |
| `Age_mean` | 0.7357 | 0.2643 |
| `Age_std` | 0.0388 | **0.9612** |
| `Age_delta` (= Age_last − Age_first) | 0.0310 | **0.9690** |
| `Age_rel_delta` | 0.0331 | **0.9669** |
| `Age_slope` (≥2-visit indicator) | 0.4819 | 0.5181 |
| `Age_count` (n visits) | 0.2202 | **0.7798** |

`Age_delta` ALONE (univariate) achieves dea C-index 0.969 — i.e.,
**98 % of the CWGBSA gain comes from the relationship between observed
follow-up duration and death status in the training data**.

## 7. Mechanism — why this works on train, why it transfers on LB,
and why it is methodologically fragile

### Why it works on train

- Censored patients use `time = max(Age_v*) − Age_v1 = Age_delta`
  as the survival time.
- Death-event patients use `time = death_age_occur − Age_v1`.
  Their `Age_delta` is the time from first visit to last *observed*
  visit; this is typically shorter for patients who died earlier in
  the observation window.
- Concordance is dominated by event-vs-censored pairs. The C-index
  rewards "high predicted risk for the patient who actually had the
  event sooner". A predictor of `−Age_delta` therefore ranks early
  events above late censorings, which gives high C-index without
  encoding any disease state.

This is **not strict label leakage** (no event/death/occur/censoring
columns enter the feature). It is **survival-time encoding via the
follow-up duration**, which is mathematically present in
`time = Age_last − Age_v1` for censored patients.

### Why it transferred on LB

LB +0.7% (0.91426 → 0.92135 on the headline blend) confirms the
follow-up structure on the test cohort mirrors train:
- Test patients are drawn from the same registry.
- Their observed `Age_v*` series has the same attrition pattern.
- Patients with short observed follow-up are more likely to have
  experienced (un-observed) events → ranking them as high-risk is
  correct *for this competition*.

### Why it is methodologically fragile

The model is **not learning disease physiology**. On a cohort with a
different follow-up structure (e.g., a fixed 5-year scheduled study,
or a different registry's attrition pattern), the Age_delta predictor
would lose all signal and the model would degrade dramatically.

This is a **competition-specific artefact**, not a transferable
clinical biomarker.

## 8. Methodology write-up implications

- **Do** describe the dea component as "ComponentwiseGBSA on Claude
  longitudinal_summary features".
- **Do** disclose that the model selects predominantly age-trajectory
  / follow-up-cadence features, and that the OOF gain is dominated by
  the relationship between observed follow-up duration and death
  status in this cohort.
- **Do not** claim the dea component encodes a novel disease
  physiology signal — the audit shows it does not.
- **Do** note that this is the same family of follow-up-cadence
  features that are present in the GPT track's `current_state_v2`
  and Claude track's `longitudinal_summary` (the dominant feature
  sets used by both anchors); the CWGBSA result is exposing this
  pre-existing signal rather than discovering a new one.

## 9. Reproducibility checklist

| item | status |
|---|---|
| Hyperparameters (lr=0.05, n=300, subs=0.8, seed=42) | documented |
| Feature builder (`longitudinal_summary`) | from `claude.src.features.build_features` |
| StandardScaler fit on full train | yes |
| Per-repeat seed = base+r | yes |
| OOF saved as mean-of-ranks across 10 repeats | yes |
| Per-fold C-index reconstruction | yes (`audit_cwgbsa_perfold.csv`) |
| Holdout 80/20 sanity, 5 seeds | yes (`audit_cwgbsa_holdout.csv`) |
| Coefficient export | yes (`audit_cwgbsa_coefs.csv`) |
| Feature ablation | yes (`audit_cwgbsa_ablation.csv`) |

## 10. Files

> Note: the file paths below refer to the original development workspace. The full development tree is not included in this review repository. This audit is included here as provenance for the death-model explainability finding; the submitted prediction file, deterministic slot1 verification script, and cached intermediates needed for verification are included in the review repository.

- `model_zoo_sprint/scripts/audit_CWGBSA.py`
- `model_zoo_sprint/scripts/audit_CWGBSA_ablation.py`
- `model_zoo_sprint/logs/audit_cwgbsa_perfold.csv`
- `model_zoo_sprint/logs/audit_cwgbsa_holdout.csv`
- `model_zoo_sprint/logs/audit_cwgbsa_coefs.csv`
- `model_zoo_sprint/logs/audit_cwgbsa_ablation.csv`
- This report.
