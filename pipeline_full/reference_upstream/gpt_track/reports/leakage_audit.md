# Leakage audit

Diagnostics that quantify how much of the survival signal is coming from follow-up bookkeeping (visit count, follow-up duration, missingness) rather than disease biology. Run from `python -m src.leakage_audit`.

## Visits per patient

- train: {'n': 1253, 'mean': 4.760574620909816, 'median': 4.0, 'min': 1.0, 'max': 19.0}
- test:  {'n': 423, 'mean': 5.349881796690307, 'median': 4.0, 'min': 1.0, 'max': 22.0}

## Last observed age (years)

- train: {'n': 1253, 'mean': 59.719074221867515, 'median': 62.0, 'min': 19.0, 'max': 89.0}
- test:  {'n': 423, 'mean': 61.2387706855792, 'median': 63.0, 'min': 24.0, 'max': 88.0}

## Follow-up years (last_age - Age_v1)

- train: {'n': 1253, 'mean': 5.293695131683958, 'median': 4.0, 'min': 0.0, 'max': 21.0}
- test:  {'n': 423, 'mean': 6.439716312056738, 'median': 5.0, 'min': 0.0, 'max': 21.0}

## Visits *after* the recorded event age

- hepatic event patients with at least one post-event visit: 27 of 47
- death patients with at least one post-death visit: 0 of 76

  These rows are why the **strict_time_aligned** feature set masks post-event visits during training.

## Missing-death subgroup profile

- n: 269
- median_visits: 2.0
- median_followup_years: 1.0
- hepatic_event_rate: 0.0037174721189591076

## Single-feature C-index benchmarks

| feature            |   cindex_hepatic |   cindex_death |
|:-------------------|-----------------:|---------------:|
| Age_v1             |           0.6311 |         0.7953 |
| n_visits           |           0.4314 |         0.2202 |
| last_observed_age  |           0.6055 |         0.6421 |
| followup_years     |           0.4204 |         0.0310 |
| total_missingness  |           0.5701 |         0.7642 |
| missingness_v1_v3  |           0.5854 |         0.4732 |
| neg_followup_years |           0.5796 |         0.9690 |
| neg_n_visits       |           0.5686 |         0.7798 |

## Interpretation

- A feature like `Age_v1` reaching ~0.65 on the death endpoint is expected (older patients die sooner) and is *not* leakage — it is biology and is fair game for the model.

- However, `n_visits` and `followup_years` carry the survivor's footprint: someone who is still alive accumulates more visits. A C-index well above 0.5 for `neg_followup_years` (i.e. *fewer* follow-up years -> higher risk) is a signature of follow-up leakage. Track this number across the contest; in the synthetic Trustii data follow-up is partly aligned with outcomes by construction.

- Total missingness behaves similarly to follow-up. The high-leakage feature sets (`all_visits_longitudinal`, `full_high_risk`, `missingness_and_visit_cadence`) inherit this.

- Patients with missing `death` status look very similar to censored patients (median follow-up close to the censored cohort, low hepatic event rate). Treating them as censored at the last visit is the default; a sensitivity comparison is run by experiments 002/003.
