# Phase 2d audit — landmark-3y verification

Date: 2026-04-27. Headline 3y RSF result: mean=0.8798, std=0.0650, **m−s=0.8148** (filter A, 5×10 CV with tuned baseline_v1 RSF params).

Tuned RSF params used: {"n_estimators": 279, "min_samples_leaf": 12, "min_samples_split": 71, "max_features": "log2"}


## Audit #2 — full landmark sweep, two filters

Filter A: drop event-before-landmark (the filter used in phase2d_landmark.py).

Filter B: A ∩ {patient has visits reaching ≥ landmark age}. Filter B is the population for which the LOCF-at-landmark feature is *actually* computed from data observed at landmark — for filter-A-only rows that lack data through landmark, LOCF is just their last observed value (effectively early-truncated).

| Landmark | filter A: n / events | filter B: n / events | A but no data |
|---|---|---|---|
| 1.0y | 1245 / 39 | 1106 / 39 | 139 |
| 2.0y | 1241 / 35 | 977 / 32 | 264 |
| 3.0y | 1238 / 32 | 831 / 27 | 407 |
| 5.0y | 1229 / 23 | 546 / 20 | 683 |


## Audit #1 — 3y RSF deep-dive


### [1] Headline mean / std
- Filter A (n=1238, events=32): **mean=0.8798, std=0.0650, m−s=0.8148**
- Filter B (n=831, events=27): mean=0.8823, std=0.0703, m−s=0.8120

User's question: *m−s 0.815 implies mean ≈ 0.86–0.87*. Confirmed: mean is **0.8798** at filter A.


### [2] Sample size with stricter filter
- Filter A (event-free at 3y): n=1238, events=32.
- Filter B (event-free AT 3y AND has visit through 3y): n=831, events=27.
- 407 of the filter-A patients have no observed visit reaching the 3y landmark — for them, `*_locf_at_landmark` is identical to baseline LOCF.

→ **Filter B has only 27 events.** Per the pre-registered rule (n<30 events = unstable CV), the 3y landmark result is suspect.

### [3] Per-fold C-index distribution (filter A, 50 folds)
```
min    = 0.5933
p25    = 0.8541
median = 0.8901
p75    = 0.9194
max    = 0.9710
spread = 0.3777
folds ≥0.90: 21/50    folds <0.80: 5/50
```

### [4] Test-set predictability at landmark cutoffs
How many test patients have a visit reaching the landmark age? (For those that don't, the LOCF features are based on data before the landmark — the model was trained on a mix of these regimes.)
| Landmark | test rows reaching landmark | % |
|---|---|---|
| 1y | 414 | 97.9% |
| 2y | 385 | 91.0% |
| 3y | 333 | 78.7% |
| 5y | 223 | 52.7% |

### [5] Single-feature C-index audit on 3y landmark features
Top-15 of 48 features by best-sign single-feature C on the filter-A cohort. Threshold for concern: > 0.85.

| feature | best C | sign |
|---|---|---|
| `fibs_stiffness_med_BM_1_locf_at_landmark` | 0.822 • | pos |
| `fibrotest_BM_2_locf_at_landmark` | 0.780 | pos |
| `ggt_v1` | 0.778 | pos |
| `ggt_locf_at_landmark` | 0.771 | pos |
| `ast_locf_at_landmark` | 0.735 | pos |
| `fibs_stiffness_med_BM_1_v1` | 0.725 | pos |
| `fibrotest_BM_2_v1` | 0.708 | pos |
| `ast_v1` | 0.703 | pos |
| `aixp_aix_result_BM_3_locf_at_landmark` | 0.669 | pos |
| `Age_v1` | 0.655 | pos |
| `Age_locf_at_landmark` | 0.648 | pos |
| `ast_alt_v1` | 0.641 | pos |
| `ast_slope_v1_to_landmark` | 0.635 | neg |
| `alt_locf_at_landmark` | 0.633 | pos |
| `bilirubin_slope_v1_to_landmark` | 0.606 | pos |

Totals: features with C>0.85: **0**; with C>0.80: 1.

→ No single feature crosses 0.85; the 0.86 mean comes from feature *combinations*, not a hidden proxy.