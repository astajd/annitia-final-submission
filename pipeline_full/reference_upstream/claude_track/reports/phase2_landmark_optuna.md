# Task 2 — Optuna tune of landmark-3y RSF (Filter B)

Date: 2026-04-28. Cohort: Filter B (n=831, events=27). Inner CV: 5-fold × 5-repeat. Final eval: 5-fold × 10-repeat.

## Best params (30 trials)

```json
{
  "n_estimators": 352,
  "min_samples_leaf": 10,
  "min_samples_split": 67,
  "max_features": "log2"
}
```

- Inner CV m-s = 0.8166 (mean 0.8807 ± 0.0640).
- Final 5×10 m-s = 0.8130 (mean 0.8824 ± 0.0694).
- Filter-B baseline m-s = 0.8120.
- **Δ m-s vs baseline = +0.0010** (threshold for action: ≥0.01).
- Decision: **STAY** with current params; no submission built.