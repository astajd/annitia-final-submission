# Phase 2d audit — chat summary

(Verbatim copy of the chat message after the audit run, saved here because the
terminal cropped it. Full machine-generated report is at
`reports/phase2d_audit.md`. Audit script: `experiments/phase2d_audit.py`.)

---

Audits complete. Full report at `reports/phase2d_audit.md`. Headlines:

## Audit #2 — landmark sweep with two filters

Filter A = "event-free at landmark" (the filter the original 2d CV used).
Filter B = A ∩ "patient has visits actually reaching landmark age" (the cohort for whom LOCF-at-landmark is computed from data observed *at* landmark).

| Landmark | Filter A: n / events | Filter B: n / events | Filter A but no data through landmark |
|---|---|---|---|
| 1y | 1245 / 39 | 1106 / 39 | 139 |
| 2y | 1241 / 35 | 977 / 32 | 264 |
| **3y** | **1238 / 32** | **831 / 27** | **407** |
| 5y | 1229 / 23 | 546 / 20 | 683 |

407 of the filter-A patients at 3y have no observed visit reaching the 3y landmark — for them, LOCF-at-landmark is just their last observed value, which may be at 1.5y or earlier. The model was trained on a mix of these regimes.

## Audit #1 — 3y RSF deep-dive

**[1] Mean / std re-confirmed (5×10 CV, tuned RSF baseline_v1 hyperparams):**
- Filter A (n=1238, 32 events): **mean 0.8798 ± std 0.0650, m−s 0.8148** — your guess was right.
- Filter B (n=831, **27 events**): mean 0.8823 ± std 0.0703, m−s 0.8120.

**[2] Sample size:** Filter B at 3y has **27 events — below your pre-registered 30-event threshold** for "unstable CV". Filter A has 32 events, which barely clears it. But the two filters give virtually identical numbers (Δm−s = −0.003), which is reassuring on its own.

**[3] Per-fold C-index distribution, filter A, 50 folds:**
```
min    = 0.5933
p25    = 0.8541
median = 0.8901
p75    = 0.9194
max    = 0.9710
spread = 0.3777
folds ≥ 0.90: 21/50    folds < 0.80: 5/50
```
**This is not a single lucky fold.** 21 of 50 folds clear 0.90; the median is 0.89. But the spread is genuinely wide — 5 folds drop below 0.80, with a worst-case 0.59 (near random). With 32 events / 5 splits = ~6.4 events per validation fold, that volatility is structural, not pathological.

**[4] Test-set predictability:**

| Landmark | test rows reaching landmark | % |
|---|---|---|
| 1y | 414 | 97.9% |
| 2y | 385 | 91.0% |
| **3y** | **333** | **78.7%** |
| 5y | 223 | 52.7% |

**~21% of test patients (90 rows) don't have 3y of follow-up.** They'll get predictions, but their landmark features are based on whatever they had before they were last seen — out-of-distribution relative to the 67% of train patients with full 3y data.

**[5] Single-feature C-index audit on 49 landmark features:**

| feature | best C | sign |
|---|---|---|
| `fibs_stiffness_med_BM_1_locf_at_landmark` | **0.822** | pos |
| `fibrotest_BM_2_locf_at_landmark` | 0.780 | pos |
| `ggt_v1` | 0.778 | pos |
| `ggt_locf_at_landmark` | 0.771 | pos |
| `ast_locf_at_landmark` | 0.735 | pos |
| `fibs_stiffness_med_BM_1_v1` | 0.725 | pos |
| `fibrotest_BM_2_v1` | 0.708 | pos |
| `ast_v1` | 0.703 | pos |

**Zero features cross 0.85. Exactly one crosses 0.80** (FibroScan LOCF at landmark, which is exactly the most clinically informative MASLD-fibrosis marker). The 0.88 mean comes from feature *combinations*, not a hidden proxy.

## Reading

The 3y landmark RSF result is a real signal, not a leakage artifact:
- No hidden single-feature leak (max 0.822, clinically expected)
- Stable across filter A vs filter B (Δm−s = 0.003)
- Median per-fold C 0.89 with 21/50 folds ≥ 0.90 — broad-based, not one lucky fold
- Top contributor (FibroScan LOCF) is clinically the right answer for hepatic-fibrosis risk

But three caveats keep it from being an unambiguous winner:
- **27 events at filter B falls below your pre-registered 30-event suspicious-result threshold.** The number is borderline.
- **Per-fold spread is huge (0.59–0.97).** LB realization is one draw from this distribution; could easily land 0.78 or 0.92. CV mean is informative; CV mean - 1·std (0.815) probably overstates expected LB.
- **21% of test rows are partially out-of-distribution** at 3y. Generalization to those rows is genuinely uncertain.

Stopped per instruction. Awaiting your call on whether to ship it, ship something else, or further audit (e.g., 1y landmark — 97.9% test predictability and only 8 events lost — has different tradeoffs and might be the safer landmark choice).
