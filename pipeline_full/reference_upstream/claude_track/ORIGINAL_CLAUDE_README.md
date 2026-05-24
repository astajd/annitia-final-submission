# ANNITIA — Phase 1

Trustii / IHU ICAN data challenge: predict hepatic events and all-cause death
in MASLD patients from longitudinal NIT data.

Scoring: `0.7 × C_hepatic + 0.3 × C_death` (concordance index).

## Phase 1 status: DONE

What's here:
- Modular pipeline (data → features → CV → models → submission)
- Leakage audit (revealed `strict_time_aligned` is leaky — DO NOT USE)
- 32-row CV bake-off across 7 feature sets × 5 models × 2 endpoints × 2 death modes
- First valid submission via rank-averaged ensemble
- Phase 2 handoff document for silentbase

Read in this order:
1. `reports/phase1_summary.md` — findings and CV table
2. `reports/leakage_audit.json` — single-feature audit numbers
3. `reports/phase1_cv_results.csv` — full CV results
4. `SILENTBASE_HANDOFF.md` — what to run next, on the 4090

## Quick reproduce

```bash
# Re-run the full Phase 1 grid (about 30-60 min on a laptop)
python3 experiments/01_leakage_audit.py
python3 experiments/phase1_incremental.py
python3 experiments/03_make_submission.py
```

Submission file written to `submissions/phase1_ensemble.csv`.

## Headline numbers (5×3 stratified CV)

| Endpoint | Best honest model | C-index |
|---|---|---|
| Hepatic | RSF on baseline_v1 | **0.797 ± 0.078** |
| Hepatic | XGB-Cox on nit_only | 0.789 ± 0.117 |
| Death | XGB-Cox on longitudinal_summary | **0.952 ± 0.014** |
| Combined (weighted) | — | **0.844** |

## Key insights

1. **Death is mostly free.** `followup_yrs` alone gives C ≈ 0.97. Don't over-optimize.
2. **Hepatic is the game.** Honest ceiling around 0.80 with current features; real signal is there.
3. **Strict time-aligned features leak through `Age_delta`/`_count`.** Use fixed-reference-time alternatives in Phase 2.
4. **NaN-death cohort doesn't hurt death modeling either way.** Use `censor_missing_death_at_last`.
5. **70/30 weighting understates hepatic dominance.** Death saturates at 0.95 quickly; differentiation happens on hepatic.

## Files

```
.
├── README.md                          # this file
├── SILENTBASE_HANDOFF.md              # Phase 2 spec
├── data/raw/                          # train, test, dictionary, hello-world sub
├── src/                               # pipeline modules
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── cv.py
│   ├── models.py
│   └── __init__.py
├── experiments/
│   ├── 01_leakage_audit.py
│   ├── run_one.py
│   ├── phase1_incremental.py
│   └── 03_make_submission.py
├── reports/
│   ├── phase1_summary.md
│   ├── leakage_audit.json
│   └── phase1_cv_results.csv
└── submissions/
    ├── phase1_ensemble.csv            # first submission
    └── phase1_ensemble.json           # ensemble metadata
```
