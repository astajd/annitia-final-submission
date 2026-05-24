"""Build handoff_minimal/ package for parallel team comparison.

Spec (verbatim from user):
- handoff_minimal/
  - submission_log.csv
  - metrics_summary.csv
  - model_descriptions.md
  - ensemble_details.md
  - best_submissions/  (5 CSVs)
  - optional_oof_predictions/  (4 CSVs)
  - README.md
- Zip as handoff_minimal.zip; must be < 5 MB.

Hepatic OOFs are stitched from reports/phase2_oof_perfold_cache.pkl (no retrain).
Death OOF is regenerated once with XGB-Cox / longitudinal_summary
(n_estimators=300, learning_rate=0.05).
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import StratifiedKFold

ROOT = Path(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

from src.data import load_raw, build_targets  # noqa: E402
from src.features import build_features  # noqa: E402
from src.models import make_xgb_cox  # noqa: E402

OUT = ROOT / "handoff_minimal"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"
CACHE = REPORTS / "phase2_oof_perfold_cache.pkl"

CV_N_SPLITS = 5
CV_N_REPEATS = 10
BASE_SEED = 42


def fresh_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)


def cindex(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    return float(
        concordance_index_censored(event.astype(bool), time.astype(float), risk.astype(float))[0]
    )


def stitch_oof_hepatic(cache: list[dict], n_rows: int) -> dict[str, np.ndarray]:
    """For each repeat, every patient appears in exactly one validation fold,
    so risk per repeat is well-defined; we average ranks across repeats."""
    by_repeat: dict[int, list[dict]] = {}
    for rec in cache:
        by_repeat.setdefault(rec["repeat"], []).append(rec)

    member_keys = [
        "risk_lm",
        "risk_rsf_longitudinal_summary",
        "risk_xgb_cox_longitudinal_plus_meta",
        "risk_xgb_cox_longitudinal_summary",
    ]
    repeat_ranks = {k: [] for k in member_keys}
    for r, recs in sorted(by_repeat.items()):
        per_repeat = {k: np.full(n_rows, np.nan) for k in member_keys}
        for rec in recs:
            idx = np.asarray(rec["val_idx"], dtype=int)
            for k in member_keys:
                per_repeat[k][idx] = np.asarray(rec[k], dtype=float)
        for k in member_keys:
            assert not np.isnan(per_repeat[k]).any(), f"repeat {r} missing rows for {k}"
            repeat_ranks[k].append(rankdata(per_repeat[k]))

    # average ranks across repeats; this is the OOF risk we report
    out: dict[str, np.ndarray] = {}
    for k in member_keys:
        out[k] = np.mean(np.stack(repeat_ranks[k]), axis=0)

    # permissive_ensemble_avg = mean of per-repeat ranks of 3 permissive members
    perm_keys = [
        "risk_rsf_longitudinal_summary",
        "risk_xgb_cox_longitudinal_plus_meta",
        "risk_xgb_cox_longitudinal_summary",
    ]
    perm_per_repeat = []
    for r in sorted(by_repeat.keys()):
        per = {k: np.full(n_rows, np.nan) for k in perm_keys}
        for rec in by_repeat[r]:
            idx = np.asarray(rec["val_idx"], dtype=int)
            for k in perm_keys:
                per[k][idx] = np.asarray(rec[k], dtype=float)
        # rank within repeat first, then mean across members
        avg = np.mean(np.stack([rankdata(per[k]) for k in perm_keys]), axis=0)
        perm_per_repeat.append(avg)
    perm_avg = np.mean(np.stack(perm_per_repeat), axis=0)
    out["permissive_ensemble_avg"] = perm_avg

    # blend_2way_optimal = 0.30·rank(landmark) + 0.70·rank(perm_avg) per repeat
    blend_per_repeat = []
    for ri, r in enumerate(sorted(by_repeat.keys())):
        lm = repeat_ranks["risk_lm"][ri]
        pm = perm_per_repeat[ri]
        blend_per_repeat.append(0.30 * rankdata(lm) + 0.70 * rankdata(pm))
    out["blend_2way_optimal"] = np.mean(np.stack(blend_per_repeat), axis=0)

    return out


def regenerate_death_oof(train: pd.DataFrame, dea_t: pd.DataFrame) -> np.ndarray:
    """XGB-Cox / longitudinal_summary, 5×10 repeated stratified, base_seed=42.
    Returns mean rank across repeats over the full cohort (death cohort = full 1253)."""
    X = build_features(train, "longitudinal_summary").reset_index(drop=True)
    e = dea_t["death_event"].to_numpy().astype(int)
    t = dea_t["death_time"].to_numpy().astype(float)
    n = len(train)
    repeat_ranks = []
    for rep in range(CV_N_REPEATS):
        seed = BASE_SEED + rep
        skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=seed)
        risk_full = np.full(n, np.nan)
        for tr, va in skf.split(np.zeros(n), e):
            fit_predict = make_xgb_cox(n_estimators=300, learning_rate=0.05)
            risk_va = fit_predict(X.iloc[tr], e[tr], t[tr], X.iloc[va])
            risk_full[va] = risk_va
        assert not np.isnan(risk_full).any()
        repeat_ranks.append(rankdata(risk_full))
    return np.mean(np.stack(repeat_ranks), axis=0)


def write_oof_csvs(train: pd.DataFrame, hep_oof: dict, death_oof: np.ndarray) -> dict[str, dict]:
    out_dir = OUT / "optional_oof_predictions"
    out_dir.mkdir(exist_ok=True)
    pid = train["patient_id_anon"].astype(str).to_numpy()

    files = {
        "oof_landmark_3y_rsf.csv": ("hepatic_oof_risk", hep_oof["risk_lm"]),
        "oof_permissive_ensemble_avg.csv": ("hepatic_oof_risk", hep_oof["permissive_ensemble_avg"]),
        "oof_blend_2way_optimal.csv": ("hepatic_oof_risk", hep_oof["blend_2way_optimal"]),
        "oof_death_xgb_longitudinal_summary.csv": ("death_oof_risk", death_oof),
    }
    summary: dict[str, dict] = {}
    for fname, (col, arr) in files.items():
        df = pd.DataFrame({"patient_id_anon": pid, col: arr})
        df.to_csv(out_dir / fname, index=False)
        summary[fname] = {"rows": len(df), "col": col, "min": float(arr.min()), "max": float(arr.max())}
    return summary


# Submissions → metadata for the log/metrics tables
SUBMISSION_SPEC = [
    {
        "filename": "phase1_ensemble.csv",
        "date": "2026-04-26",
        "lb": 0.798559,
        "short": "Phase 1 baseline rank-averaged ensemble",
        "feature_sets": "baseline_v1, early_v1_v3, longitudinal_summary, longitudinal_plus_meta, nit_only_baseline_only",
        "models": "RSF, XGB-Cox, CoxNet, LightGBM, CatBoost, LogReg",
        "ensemble": "rank-average across 5×3=15 (feature_set, model) cells per endpoint",
        "notes": "Phase 1 grid; CV ~0.844. Death rank-avg likely miscalibrated relative to small public LB cohort.",
    },
    {
        "filename": "phase2_strict_leaky_probe.csv",
        "date": "2026-04-26",
        "lb": 0.692,
        "short": "Tier-5 outcome-dependent leakage probe (audit only)",
        "feature_sets": "strict_time_aligned (Tier 5)",
        "models": "RSF",
        "ensemble": "single model",
        "notes": "Audit probe — confirms strict_time_aligned does NOT transfer (CV ~0.94, LB 0.692). Tier-5 forbidden.",
    },
    {
        "filename": "phase2_honest_ensemble.csv",
        "date": "2026-04-27",
        "lb": 0.7509,
        "short": "Tier-1-only conservative blend",
        "feature_sets": "baseline_v1, nit_only_baseline_only (Tier 1)",
        "models": "RSF + XGB-Cox",
        "ensemble": "rank-average",
        "notes": "Tier-1 purity left ~0.05 LB hep on the table; trajectory features we audited away are organizer-blessed signal.",
    },
    {
        "filename": "phase2_landmark_3y.csv",
        "date": "2026-04-27",
        "lb": 0.8109,
        "short": "Single tuned RSF on landmark_3y (Tier 2)",
        "feature_sets": "landmark_3y (49 features = baseline_v1 + LOCF + slope-since-baseline at 3y horizon)",
        "models": "RSF (Optuna-tuned)",
        "ensemble": "single model on filter-B (at-risk-at-3y) cohort",
        "notes": "First clean Tier-2 submission. Hep CV 0.880 ± 0.065; established the landmark recipe.",
    },
    {
        "filename": "phase2_blend_landmark_permissive.csv",
        "date": "2026-04-27",
        "lb": 0.88033,
        "short": "50/50 rank blend of landmark_3y_RSF + permissive ensemble",
        "feature_sets": "landmark_3y (Tier 2) + longitudinal_summary, longitudinal_plus_meta (Tier 4)",
        "models": "RSF + (RSF + XGB-Cox + XGB-Cox)",
        "ensemble": "0.5·rank(landmark) + 0.5·rank(permissive_avg)",
        "notes": "Cross-method diversity (landmark × permissive, OOF Spearman 0.478) yielded +0.069 LB over phase2_landmark_3y.",
    },
    {
        "filename": "phase2_landmark_multi.csv",
        "date": "2026-04-27",
        "lb": 0.81,
        "short": "Multi-horizon landmark blend (1y, 2y, 3y)",
        "feature_sets": "landmark_1y, landmark_2y, landmark_3y (Tier 2)",
        "models": "RSF (Optuna-tuned, per horizon)",
        "ensemble": "rank-average across 3 horizons",
        "notes": "Within 0.001 of single-3y (LB 0.811). Within-method ensembling exhausted — diversity must come from method class.",
    },
    {
        "filename": "phase2_blend_3way.csv",
        "date": "2026-04-27",
        "lb": 0.829,
        "short": "3-way blend: landmark + permissive + tuned XGB-Cox baseline",
        "feature_sets": "landmark_3y + permissive_avg + baseline_v1",
        "models": "RSF + permissive ensemble + XGB-Cox",
        "ensemble": "1/3 · rank each",
        "notes": "Down 0.051 from 2-way. The honest XGB-Cox/baseline_v1 third pulled distribution toward Tier-1 which under-rewards on LB.",
    },
    {
        "filename": "phase2_blend_2way_optimal.csv",
        "date": "2026-04-28",
        "lb": 0.8965,
        "short": "OOF-stacked weighted blend (champion)",
        "feature_sets": "landmark_3y (Tier 2) + longitudinal_summary, longitudinal_plus_meta (Tier 4)",
        "models": "RSF + (RSF + XGB-Cox + XGB-Cox)",
        "ensemble": "0.30·rank(landmark) + 0.70·rank(permissive_avg); weight from 5×10 OOF stacking",
        "notes": "+0.016 LB over 50/50. Permissive component carries 70% — it is the engine of the blend, not landmark.",
    },
    {
        "filename": "phase2_blend_2way_strongpermissive.csv",
        "date": "2026-04-28",
        "lb": 0.8867,
        "short": "Replaces 3-member permissive with single rsf_longitudinal_summary; OOF re-stacked w=0.23/0.77",
        "feature_sets": "landmark_3y + longitudinal_summary",
        "models": "RSF + RSF",
        "ensemble": "0.23·rank(landmark) + 0.77·rank(rsf_longitudinal_summary)",
        "notes": "CV-LB inversion #1: +0.014 CV, −0.010 LB. Removing the 3-XGB redundancy hurt LB despite CV gain.",
    },
    {
        "filename": "phase3_blend_with_crossfeatures.csv",
        "date": "2026-04-28",
        "lb": 0.872,
        "short": "Cross-endpoint OOF stack: augments landmark RSF with oof_death_risk feature",
        "feature_sets": "landmark_3y + oof_death_risk (cross-endpoint)",
        "models": "RSF (augmented) + permissive ensemble",
        "ensemble": "30/70 rank blend with augmented landmark replacing original",
        "notes": "CV-LB inversion #2: +0.0094 CV on landmark, −0.025 LB overall. Cross-endpoint stack didn't transfer.",
    },
]

BEST_TO_COPY = [
    "phase2_blend_2way_optimal.csv",
    "phase2_blend_landmark_permissive.csv",
    "phase2_landmark_3y.csv",
    "phase2_strict_leaky_probe.csv",
    "phase1_ensemble.csv",
]


def build_submission_log() -> pd.DataFrame:
    rows = []
    for s in SUBMISSION_SPEC:
        rows.append({
            "filename": s["filename"],
            "public_lb_score": s["lb"],
            "short_description": s["short"],
            "feature_sets_used": s["feature_sets"],
            "model_families": s["models"],
            "ensemble_method": s["ensemble"],
            "notes": s["notes"],
        })
    return pd.DataFrame(rows)


def build_metrics_summary(hep_oof: dict, death_oof: np.ndarray, hep_t: pd.DataFrame, dea_t: pd.DataFrame) -> pd.DataFrame:
    e_h = hep_t["hepatic_event"].to_numpy().astype(int)
    t_h = hep_t["hepatic_time"].to_numpy().astype(float)
    e_d = dea_t["death_event"].to_numpy().astype(int)
    t_d = dea_t["death_time"].to_numpy().astype(float)

    hep_lm = cindex(e_h, t_h, hep_oof["risk_lm"])
    hep_perm = cindex(e_h, t_h, hep_oof["permissive_ensemble_avg"])
    hep_blend = cindex(e_h, t_h, hep_oof["blend_2way_optimal"])
    dea_xgb = cindex(e_d, t_d, death_oof)

    rows = []

    # Standalone models (OOF C-index from cache / regen)
    rows.append({
        "model_or_submission_name": "landmark_3y_RSF (standalone)",
        "public_lb_score": "",
        "oof_hepatic_cindex": round(hep_lm, 4),
        "oof_death_cindex": "",
        "oof_weighted_score": "",
        "cv_method": "5×10 stratified hepatic",
        "notes": "Tier 2 RSF on landmark_3y (filter A, full cohort). Component of phase2_blend_2way_optimal.",
    })
    rows.append({
        "model_or_submission_name": "permissive_ensemble_avg (standalone)",
        "public_lb_score": "",
        "oof_hepatic_cindex": round(hep_perm, 4),
        "oof_death_cindex": "",
        "oof_weighted_score": "",
        "cv_method": "5×10 stratified hepatic",
        "notes": "Tier 4 mean-of-ranks: rsf_longitudinal_summary + xgb_cox_longitudinal_plus_meta + xgb_cox_longitudinal_summary.",
    })
    rows.append({
        "model_or_submission_name": "xgb_cox_death_longitudinal_summary (standalone)",
        "public_lb_score": "",
        "oof_hepatic_cindex": "",
        "oof_death_cindex": round(dea_xgb, 4),
        "oof_weighted_score": "",
        "cv_method": "5×10 stratified death",
        "notes": "Death predictor used in all Phase 2 submissions (n_estimators=300, learning_rate=0.05).",
    })

    # Submissions with derivable OOF
    rows.append({
        "model_or_submission_name": "phase2_blend_2way_optimal",
        "public_lb_score": 0.8965,
        "oof_hepatic_cindex": round(hep_blend, 4),
        "oof_death_cindex": round(dea_xgb, 4),
        "oof_weighted_score": round(0.7 * hep_blend + 0.3 * dea_xgb, 4),
        "cv_method": "5×10 stratified hepatic",
        "notes": "Champion. 30/70 OOF-stacked rank blend.",
    })

    # Other submissions: LB only (no full OOF reconstruction)
    for s in SUBMISSION_SPEC:
        name = s["filename"].replace(".csv", "")
        if name == "phase2_blend_2way_optimal":
            continue
        rows.append({
            "model_or_submission_name": name,
            "public_lb_score": s["lb"],
            "oof_hepatic_cindex": "",
            "oof_death_cindex": "",
            "oof_weighted_score": "",
            "cv_method": "5×10 stratified hepatic",
            "notes": "OOF not reconstructed for this submission.",
        })

    return pd.DataFrame(rows)


MODEL_DESCRIPTIONS_MD = """# Model descriptions — top 3 submissions above 0.88 LB

## 1. phase2_blend_2way_optimal (LB **0.8965**) — champion

Two-model rank blend:
- **landmark_3y_RSF**: RandomSurvivalForest tuned via Optuna, trained on `landmark_3y` (Tier 2) — 49 features = `baseline_v1` static + LOCF values at the 3-year landmark + slope of each variable from baseline to landmark. Trained on the at-risk-at-3y cohort (filter A inclusion).
- **permissive_ensemble_avg**: rank-mean of three Tier-4 models that use the full per-visit trajectory regardless of timing relative to the event. Members:
  1. `rsf_longitudinal_summary` (RSF on `_last`/`_max`/`_count`/`_std`/`_slope` aggregates)
  2. `xgb_cox_longitudinal_plus_meta` (XGBoost survival:cox + visit metadata)
  3. `xgb_cox_longitudinal_summary` (XGBoost survival:cox on aggregates)

Blend weight `0.30·rank(landmark) + 0.70·rank(permissive_avg)` was selected by OOF stacking
(`phase2_stack_2way.py`) on the 5×10 fold structure: scanning `w` ∈ {0.0, 0.05, …, 1.0}
and picking the maximum mean−std hepatic C-index. The 0.30 minimum was robust to a
LOFO check (drop one fold-repeat at a time; argmax stayed in {0.30, 0.35, 0.40}).

Death predictor: XGB-Cox on `longitudinal_summary` (`n_estimators=300, learning_rate=0.05`).
Identical death predictor across all Phase 2 submissions, so cross-submission deltas are
hepatic-driven.

Why it works: the two hepatic components have OOF Spearman 0.478, comfortably above the
0.5-or-better diversity threshold we found yields LB lift. The permissive component
contributes most of the predictive signal (70% weight); the landmark component acts as
a regularizer pulling toward the at-risk-at-3y subpopulation rather than as a primary
predictor.

## 2. phase2_blend_2way_strongpermissive (LB **0.8867**) — CV-LB inversion #1

Same recipe as the champion, but the 3-member permissive ensemble was replaced with the
single best member (`rsf_longitudinal_summary`, OOF mean−std 0.7356 vs the ensemble's
0.7197). OOF re-stacking gave `w = 0.23·landmark + 0.77·rsf_longitudinal_summary`.
Despite +0.014 CV gain, LB dropped 0.010. Conclusion: the 3-XGB redundancy in the
permissive ensemble was acting as variance reduction on the public LB, not signal
dilution as the CV had implied. This was the first of three consecutive CV→LB
inversions that established a noise-floor lower bound on what CV improvement is required
to expect LB lift.

## 3. phase2_blend_landmark_permissive (LB **0.880**) — equal-weight predecessor

Same components as the champion, but with naive 50/50 weights. The OOF re-weighting from
50/50 → 30/70 amplified to +0.016 LB. This established that, in our regime, even small
CV gains on the optimal-weights search can transfer to non-trivial LB gains as long as
the underlying ensemble structure (3 permissive XGB-RSF members + 1 landmark RSF) is
preserved.
"""

ENSEMBLE_DETAILS_MD = """# Ensemble details

## Rank vs raw blending

All Phase 2 ensembles blend **ranks**, not raw risks. C-index depends only on the
relative ordering of risks, so rank-blending across heterogeneous risk-magnitude
distributions (RSF risks ∈ [0, 1] vs XGB-Cox log-hazards ∈ ℝ) is the principled
aggregation. Concretely:

```python
from scipy.stats import rankdata
risk_blend = w * rankdata(risk_landmark) + (1 - w) * rankdata(risk_permissive_avg)
```

The same convention is used inside the permissive ensemble (`mean(rankdata(member_i))`
across 3 members) and across CV repeats (`mean(rankdata(per_repeat_risk_i))` across
10 repeats; see OOF stitching in `experiments/build_handoff_package.py`).

## Endpoint-specific blending

Hepatic and death are blended **separately**. Submission columns are
`risk_hepatic_event` and `risk_death`; the LB scorer applies `0.7·C_hepatic + 0.3·C_death`.
We never mix endpoints inside an ensemble. The death predictor is identical across all
Phase 2 submissions (XGB-Cox on `longitudinal_summary`, `n_estimators=300, learning_rate=0.05`),
so all Phase 2 LB deltas are attributable to hepatic.

We did not OOF-tune the death side — Phase 1 had already shown the death endpoint
saturates near 0.95 (driven by `followup_yrs` alone hitting C=0.969) and 95% of remaining
gain is on hepatic.

## OOF stacking vs heuristic

The 30/70 weight in the champion is **OOF-stacked**, not picked by hand. Procedure
(`experiments/phase2_stack_2way.py`):

1. Run 5×10 stratified-by-event CV. For each fold, predict on the validation set with
   each of the 4 sub-models (1 landmark + 3 permissive members).
2. Stitch into per-repeat OOF arrays (each patient appears in exactly one validation
   fold per repeat → 1253 OOF risks per repeat per sub-model).
3. Form `permissive_avg = mean(rank-of-3-members)` per repeat.
4. Sweep `w ∈ {0.0, 0.05, …, 1.0}`. For each `w`, compute hepatic C-index per repeat
   on `w·rank(landmark) + (1-w)·rank(permissive_avg)`. Pick `w` maximizing
   mean−std across the 10 repeats.
5. Apply that `w` to test-set predictions: train each sub-model on full train, score test,
   rank, blend with the same `w`.

Optimum `w_landmark = 0.30`. LOFO sensitivity (drop one of the 50 fold-repeats at a time
and rerun the sweep) kept the argmax in `{0.30, 0.35, 0.40}` — the minimum is stable.
The CV mean−std improvement from 50/50 → 30/70 was +0.0076 hepatic C-index; this
amplified ~2.1× on LB (+0.016).

The 3-way variant (`phase2_blend_3way.csv`, LB 0.829) added a tuned XGB-Cox/baseline_v1
as a third stacker input. OOF training-time pairwise correlations 0.49 / 0.43 looked
diversifying, but LB regressed 0.051. The third's "diversity" was OOD-flavoured rather
than additive — adding more dimensions to the stacker is not a free win.

## Component lineage at a glance

```
landmark_3y_RSF ──┐
                  ├── 0.30/0.70 OOF-stacked ──→ phase2_blend_2way_optimal (LB 0.8965)
permissive_avg ───┘                                                  ▲
   │                                                                  │
   ├── rsf_longitudinal_summary (Tier 4)                              │
   ├── xgb_cox_longitudinal_plus_meta (Tier 4)                        │
   └── xgb_cox_longitudinal_summary (Tier 4)                          │
                                                                      │
death side (constant across all Phase 2 submissions): ────────────────┘
   xgb_cox / longitudinal_summary, n_est=300, lr=0.05
```
"""

README_MD = """# ANNITA handoff_minimal — Anthropic-track team

Best public LB: **0.8965** (`phase2_blend_2way_optimal.csv`, 2026-04-28).

## Top 3 submissions above 0.88 LB

1. **phase2_blend_2way_optimal (LB 0.8965)** — OOF-stacked 30/70 rank blend of
   `landmark_3y_RSF` (Tier 2) and `permissive_ensemble_avg` (Tier 4, 3 members).
2. **phase2_blend_2way_strongpermissive (LB 0.8867)** — same skeleton, single permissive
   member (`rsf_longitudinal_summary`); CV-LB inversion #1.
3. **phase2_blend_landmark_permissive (LB 0.880)** — 50/50 equal-weight predecessor of
   the champion.

See `model_descriptions.md` and `ensemble_details.md` for the full story.

## OOF predictions

Yes, included for the top 4 hepatic / death components, in `optional_oof_predictions/`:

- `oof_landmark_3y_rsf.csv` — landmark_3y RSF, mean-of-ranks across 10 repeats
- `oof_permissive_ensemble_avg.csv` — Tier-4 permissive ensemble (3 members)
- `oof_blend_2way_optimal.csv` — the 30/70 champion blend
- `oof_death_xgb_longitudinal_summary.csv` — death predictor (XGB-Cox / longitudinal_summary)

All four use the same fold structure: 5-fold × 10-repeat StratifiedKFold on the event
indicator, `random_state = 42 + repeat`, `shuffle=True`. Each patient appears in exactly
one validation fold per repeat. Risks reported are mean ranks across repeats (so they
are comparable across files via `scipy.stats.spearmanr` or rank-blending).

## Caveats — please read before blending against your models

- **Submission count.** Ten submissions are logged here, not the seven mentioned in the
  request. We listed all of them in `submission_log.csv` so the comparison is complete.
- **CV-LB decoupling.** We observed three consecutive CV→LB inversions in late Phase 2
  / early Phase 3 (`strongpermissive`, `phase2_blend_3way`, `phase3_blend_with_crossfeatures`).
  The empirical LB noise floor at ρ_hep=1 is ~0.028, and the analytical SE_combined on a
  ~25–50 event public-LB cohort is 0.034–0.048. Treat any single LB delta below 0.03
  as noise. Plan blends accordingly.
- **Tier-5 leakage.** `phase2_strict_leaky_probe.csv` (LB 0.692) is included as
  documentation of an audit-only Tier-5 feature set (`strict_time_aligned`). It uses
  the per-row event/censoring time to define a per-row trajectory window, which leaks
  outcome through `Age_delta` (single-feature C ≈ 0.958). **Do not use for shipping.**
- **Death predictor is shared.** All Phase 2 submissions share the same death predictor
  (XGB-Cox on `longitudinal_summary`, `n_estimators=300, learning_rate=0.05`). Cross-
  submission LB deltas are entirely hepatic. If your team's death-side model is stronger,
  rank-blending with our hepatic side and your death side is a clean experiment.

## Files

- `submission_log.csv` — 10 submissions with feature sets, model families, ensemble method, LB
- `metrics_summary.csv` — OOF metrics per major standalone model + per submission
- `model_descriptions.md` — half-page each for the top 3 submissions above 0.88 LB
- `ensemble_details.md` — components, weights, OOF stacking procedure, blending convention
- `best_submissions/` — 5 representative submission CSVs
- `optional_oof_predictions/` — 4 OOF prediction CSVs (see above)
"""


def main() -> None:
    fresh_dir(OUT)
    (OUT / "best_submissions").mkdir()

    # Load
    train, _ = load_raw()
    hep_t, dea_t = build_targets(train, death_mode="censor_missing_death_at_last")
    n_rows = len(train)

    # Stitch hepatic OOFs from cache
    print("Loading OOF cache…")
    with open(CACHE, "rb") as f:
        cache = pickle.load(f)
    hep_oof = stitch_oof_hepatic(cache, n_rows=n_rows)
    print(f"  stitched {len(hep_oof)} hepatic OOF series ({n_rows} rows each)")

    # Regenerate death OOF
    print("Regenerating death OOF (XGB-Cox / longitudinal_summary, 5×10)…")
    death_oof = regenerate_death_oof(train, dea_t)
    print(f"  done; {len(death_oof)} rows")

    # Sanity-check OOF C-indices
    e_h, t_h = hep_t["hepatic_event"].to_numpy(), hep_t["hepatic_time"].to_numpy()
    e_d, t_d = dea_t["death_event"].to_numpy(), dea_t["death_time"].to_numpy()
    print(f"  C(hep, landmark)         = {cindex(e_h, t_h, hep_oof['risk_lm']):.4f}")
    print(f"  C(hep, perm_avg)         = {cindex(e_h, t_h, hep_oof['permissive_ensemble_avg']):.4f}")
    print(f"  C(hep, blend 30/70)      = {cindex(e_h, t_h, hep_oof['blend_2way_optimal']):.4f}")
    print(f"  C(death, xgb_long_sum)   = {cindex(e_d, t_d, death_oof):.4f}")

    # Write OOF CSVs
    write_oof_csvs(train, hep_oof, death_oof)

    # Submission log + metrics summary
    sub_log = build_submission_log()
    sub_log.to_csv(OUT / "submission_log.csv", index=False)
    metrics = build_metrics_summary(hep_oof, death_oof, hep_t, dea_t)
    metrics.to_csv(OUT / "metrics_summary.csv", index=False)

    # Markdown docs
    (OUT / "model_descriptions.md").write_text(MODEL_DESCRIPTIONS_MD)
    (OUT / "ensemble_details.md").write_text(ENSEMBLE_DETAILS_MD)
    (OUT / "README.md").write_text(README_MD)

    # Copy 5 best submissions
    for fname in BEST_TO_COPY:
        src = SUBMISSIONS / fname
        dst = OUT / "best_submissions" / fname
        shutil.copyfile(src, dst)

    # Zip & verify size
    zip_path = ROOT / "handoff_minimal.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in OUT.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT.parent))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\nZip: {zip_path}  ({size_mb:.2f} MB)")
    if size_mb >= 5.0:
        print("  WARNING: zip ≥ 5 MB — please prune.")
    else:
        print("  OK (< 5 MB).")

    # Manifest
    print("\nFiles:")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
