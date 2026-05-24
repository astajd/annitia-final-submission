"""Build claude_handoff_for_merge/ — enriched handoff package for the merge analyst session.

Adds to handoff_minimal/:
- STRATEGY.md (verbatim from user spec)
- reports/ (leakage_audit.json, sensitivity_analysis.md, phase3_lb_decoupling.md, forum_intel.md)
- component_test_predictions/ (4 test-set CSVs)
- README.md merge-context section
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

from src.data import load_raw, build_targets  # noqa: E402
from src.features import build_features, build_landmark_features, at_risk_at_landmark  # noqa: E402
from src.models import make_rsf, make_xgb_cox  # noqa: E402

OUT = ROOT / "claude_handoff_for_merge"
HANDOFF_MIN = ROOT / "handoff_minimal"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"

# Tuned hyperparameters from configs/optuna_*.json
RSF_BASELINE_V1 = dict(n_estimators=279, min_samples_leaf=12, min_samples_split=71, max_features="log2")
RSF_LANDMARK_3Y = dict(n_estimators=352, min_samples_leaf=10, min_samples_split=67, max_features="log2")
RSF_LONG_SUM = dict(n_estimators=679, min_samples_leaf=5, min_samples_split=76, max_features="sqrt")
XGB_COX_LONG_SUM = dict(
    n_estimators=444, learning_rate=0.013890990747050674, max_depth=5,
    min_child_weight=1.0288407478484667, subsample=0.9361937553113701,
    colsample_bytree=0.675639310110157, reg_lambda=6.478763832062464,
)
XGB_COX_LONG_PLUS_META = dict(
    n_estimators=270, learning_rate=0.010003759579011472, max_depth=7,
    min_child_weight=2.2844473344460754, subsample=0.9569061485784357,
    colsample_bytree=0.9970399773242598, reg_lambda=3.1655144979108063,
)


def fresh_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)


def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reindex both to the union of columns (test gets NaN for train-only cols, vice versa).
    Both feature builders are stateless on raw columns, so column lists should already match —
    this is just a defensive guard."""
    cols = list(X_train.columns)
    X_test_aligned = X_test.reindex(columns=cols)
    return X_train, X_test_aligned


def predict_landmark_3y_rsf(train: pd.DataFrame, test: pd.DataFrame, hep_t: pd.DataFrame) -> np.ndarray:
    """Match champion recipe: filter B (at-risk-at-3y) cohort, n=1238."""
    e = hep_t["hepatic_event"].to_numpy().astype(int)
    t = hep_t["hepatic_time"].to_numpy().astype(float)
    mask = at_risk_at_landmark(e, t, 3.0)
    X_tr_full = build_landmark_features(train, 3.0)
    X_tr = X_tr_full.loc[mask].reset_index(drop=True)
    e_tr, t_tr = e[mask], t[mask]
    X_te = build_landmark_features(test, 3.0)
    X_tr, X_te = align_columns(X_tr, X_te)
    fit_predict = make_rsf(**RSF_LANDMARK_3Y)
    return fit_predict(X_tr, e_tr, t_tr, X_te)


def predict_rsf_longitudinal_summary(train: pd.DataFrame, test: pd.DataFrame, hep_t: pd.DataFrame) -> np.ndarray:
    e = hep_t["hepatic_event"].to_numpy().astype(int)
    t = hep_t["hepatic_time"].to_numpy().astype(float)
    X_tr = build_features(train, "longitudinal_summary")
    X_te = build_features(test, "longitudinal_summary")
    X_tr, X_te = align_columns(X_tr, X_te)
    fit_predict = make_rsf(**RSF_LONG_SUM)
    return fit_predict(X_tr, e, t, X_te)


def predict_rsf_baseline_v1_tuned(train: pd.DataFrame, test: pd.DataFrame, hep_t: pd.DataFrame) -> np.ndarray:
    e = hep_t["hepatic_event"].to_numpy().astype(int)
    t = hep_t["hepatic_time"].to_numpy().astype(float)
    X_tr = build_features(train, "baseline_v1")
    X_te = build_features(test, "baseline_v1")
    X_tr, X_te = align_columns(X_tr, X_te)
    fit_predict = make_rsf(**RSF_BASELINE_V1)
    return fit_predict(X_tr, e, t, X_te)


def predict_xgb_cox_long_sum(train: pd.DataFrame, test: pd.DataFrame, hep_t: pd.DataFrame) -> np.ndarray:
    e = hep_t["hepatic_event"].to_numpy().astype(int)
    t = hep_t["hepatic_time"].to_numpy().astype(float)
    X_tr = build_features(train, "longitudinal_summary")
    X_te = build_features(test, "longitudinal_summary")
    X_tr, X_te = align_columns(X_tr, X_te)
    fit_predict = make_xgb_cox(**XGB_COX_LONG_SUM)
    return fit_predict(X_tr, e, t, X_te)


def predict_xgb_cox_long_plus_meta(train: pd.DataFrame, test: pd.DataFrame, hep_t: pd.DataFrame) -> np.ndarray:
    e = hep_t["hepatic_event"].to_numpy().astype(int)
    t = hep_t["hepatic_time"].to_numpy().astype(float)
    X_tr = build_features(train, "longitudinal_plus_meta")
    X_te = build_features(test, "longitudinal_plus_meta")
    X_tr, X_te = align_columns(X_tr, X_te)
    fit_predict = make_xgb_cox(**XGB_COX_LONG_PLUS_META)
    return fit_predict(X_tr, e, t, X_te)


# ──────────────────────────────────────────────────────────────────────────────
# Document content
# ──────────────────────────────────────────────────────────────────────────────

STRATEGY_MD = """# Anthropic-track strategy summary

## Top 3 things that worked
1. Cross-method blending (landmark + permissive, ρ 0.478): +0.069 LB jump 0.811 → 0.880
2. OOF-stacked weight optimization (50/50 → 30/70): +0.016 LB jump 0.880 → 0.8965
3. Tier-2 fixed-reference-time landmark methodology (3y RSF): clean, defensible single-model anchor

## Top 3 things that didn't work
1. Within-method ensembling: multi-landmark gave LB 0.81 (tied single 3y), internal corrs 0.81-0.88 too high
2. Trajectory-shape features: redundant with longitudinal_summary's _last/_std/_slope (corr 0.754 with permissive)
3. OOF cross-feature stacking (death pred → hepatic feature): +0.009 CV but -0.024 LB; CV-LB decoupling

## Methodology directions NOT tried (why)
- Discrete-time hazard reformulation: estimated low marginal value over RSF/XGB-Cox partial likelihood
- Pseudo-labeling: deferred after CV-LB decoupling — pseudo-labels assume calibrated test predictions
- Deep multi-task survival: no clear case for it given dataset size (1253 rows, 47 hepatic events)
- Visit-level pre/post-event classifier: too ambitious, replaced with simpler sensitivity analysis

## Key insights about the data
1. **LB noise floor empirically ±0.028** (analytical SE ~0.034-0.048). Three CV→LB inversions consistent with sampling noise, not systematic. See reports/phase3_lb_decoupling.md.
2. **Honest captures 99% of permissive on hepatic.** Tier 1 v1-only blend within 0.009 m-s of Tier 4 production. See reports/sensitivity_analysis.md.
3. **Death is saturated at ~0.95** from follow-up information. 95% of hepatic optimization effort is correct allocation.
4. **Per-row outcome-dependent feature cutoffs leak.** Strict-time-aligned probe scored LB 0.692 vs CV 0.94 — confirmed train-test asymmetry. Tier 5 is OUT.

## Forum intel used
- Organizer (April 14, 6:20): longitudinal features valid, post-event observations are present, smart handling rewarded in subjective scoring
- Organizer (April 14, 9:42): test set has post-event-equivalent observations, methodology choice considered in qualitative review

## Final ensemble structure
phase2_blend_2way_optimal.csv:
- Hepatic: 30% rank(landmark_3y_RSF) + 70% rank(permissive_ensemble_avg)
- Death: XGB-Cox on longitudinal_summary, censor_missing_death_at_last
- Weights from OOF stacking (5×10 fold structure, random_state=42+repeat)
- Rank-blending (not raw scores)
- Endpoint-independent: separate models per endpoint, no cross-endpoint blending in submission

## Honest assessment of LB position
Our progression: 0.799 → 0.811 → 0.880 → 0.8965. Mostly preplanned, audited gains. Single-shot 0.8965 is one draw from a true distribution likely centered around 0.87-0.92. The 0.054 gap to GPT track's 0.911 may be real signal, may be sampling — to be revealed by ρ analysis.
"""

README_MERGE_APPENDIX = """

## Merge-analyst-specific notes

- **Path expectation:** the merge-analyst session is expected to operate in
  `annita/merged/`, with both track packages (this one and the GPT track's)
  unzipped side-by-side as `claude_handoff_for_merge/` and the GPT-track
  equivalent. STRATEGY.md and reports/ assume that layout.

- **Component-level test predictions** in `component_test_predictions/` are
  provided so you can do **cross-track component correlation**, not just
  submission-level. The four files cover the two hepatic backbones
  (`landmark_3y_rsf`, `permissive_ensemble_avg`) plus their strongest
  individual permissive member (`rsf_longitudinal_summary`) and a clean
  Tier-1 anchor (`rsf_baseline_v1_tuned`). Use these when building
  cross-track Spearman matrices.

- **CV protocol** (applies to all OOF predictions in `optional_oof_predictions/`):
  5-fold × 10-repeat StratifiedKFold on the hepatic event indicator, with
  `random_state = 42 + repeat`, `shuffle=True`. Each patient appears in
  exactly one validation fold per repeat.

- **Risk-score conventions — read this before blending so you don't double-rank:**
  - **OOF predictions** (`optional_oof_predictions/`): each value is the
    **mean of per-repeat ranks** across 10 repeats. Already rank-transformed.
  - **Test predictions** (`component_test_predictions/`): **raw model scores**
    (RSF cumulative-hazard sums for RSF, partial-hazard log-risks for XGB-Cox).
    The single exception is `test_permissive_ensemble_avg.csv`, which is the
    mean of `rankdata(member_test_score)` across the 3 permissive members —
    necessarily rank-transformed because it is itself an aggregation. The
    column header in that one file is `hepatic_risk` like the others; the
    rank convention is documented in
    `component_test_predictions/README_components.md`.
  - When blending across tracks, apply `scipy.stats.rankdata(...)` once per
    track per file before forming the convex combination. Don't rank-of-rank.
"""

COMPONENTS_README = """# Component test predictions — convention notes

Files in this folder, all keyed by `trustii_id`:

| file | column | scale |
|---|---|---|
| `test_landmark_3y_rsf.csv` | `hepatic_risk` | RSF cumulative-hazard sum (raw) |
| `test_rsf_baseline_v1_tuned.csv` | `hepatic_risk` | RSF cumulative-hazard sum (raw) |
| `test_rsf_longitudinal_summary.csv` | `hepatic_risk` | RSF cumulative-hazard sum (raw) |
| `test_permissive_ensemble_avg.csv` | `hepatic_risk` | **mean of rankdata across 3 permissive members** |

Training recipe (matches the production submission):

- `test_landmark_3y_rsf.csv` — RSF tuned on `landmark_3y` features, trained on
  the at-risk-at-3y cohort (filter B, n=1238, events=32). This matches the
  landmark component inside `phase2_blend_2way_optimal.csv`.
- `test_rsf_baseline_v1_tuned.csv` — RSF tuned on `baseline_v1` (Tier 1),
  full train (n=1253, events=47). Clean Tier-1 anchor; not in any submission
  by itself.
- `test_rsf_longitudinal_summary.csv` — RSF tuned on `longitudinal_summary`
  (Tier 4), full train. Strongest individual permissive member.
- `test_permissive_ensemble_avg.csv` — rank-mean across 3 members:
  `rsf_longitudinal_summary` + `xgb_cox_longitudinal_plus_meta` +
  `xgb_cox_longitudinal_summary`. All three trained on full train. This
  is exactly the permissive arm inside `phase2_blend_2way_optimal.csv`.

Hyperparameters live in `configs/optuna_*.json` in the source repo; they
are copied verbatim into this script (`build_merge_handoff.py`, top of file).

Death-side test predictions are not included separately because every Phase 2
submission shares the same death predictor (XGB-Cox / `longitudinal_summary`,
`n_estimators=300, learning_rate=0.05`); see `ensemble_details.md` for why.
"""


def main() -> None:
    fresh_dir(OUT)
    (OUT / "best_submissions").mkdir()
    (OUT / "optional_oof_predictions").mkdir()
    (OUT / "component_test_predictions").mkdir()
    (OUT / "reports").mkdir()

    # Verify handoff_minimal exists
    if not HANDOFF_MIN.exists():
        raise FileNotFoundError(f"{HANDOFF_MIN} missing — run build_handoff_package.py first.")

    # Copy the bulk of handoff_minimal
    for fname in ["submission_log.csv", "metrics_summary.csv", "model_descriptions.md", "ensemble_details.md"]:
        shutil.copyfile(HANDOFF_MIN / fname, OUT / fname)
    for f in (HANDOFF_MIN / "best_submissions").iterdir():
        shutil.copyfile(f, OUT / "best_submissions" / f.name)
    for f in (HANDOFF_MIN / "optional_oof_predictions").iterdir():
        shutil.copyfile(f, OUT / "optional_oof_predictions" / f.name)

    # README.md = base + merge appendix
    base_readme = (HANDOFF_MIN / "README.md").read_text()
    (OUT / "README.md").write_text(base_readme + README_MERGE_APPENDIX)

    # STRATEGY.md
    (OUT / "STRATEGY.md").write_text(STRATEGY_MD)

    # reports/
    shutil.copyfile(REPORTS / "leakage_audit.json", OUT / "reports" / "leakage_audit.json")
    shutil.copyfile(REPORTS / "sensitivity_analysis.md", OUT / "reports" / "sensitivity_analysis.md")
    shutil.copyfile(REPORTS / "phase3_lb_decoupling.md", OUT / "reports" / "phase3_lb_decoupling.md")
    shutil.copyfile(REPORTS / "forum_intel.md", OUT / "reports" / "forum_intel.md")

    # Component test predictions
    print("Building component test predictions…")
    train, test = load_raw()
    hep_t, _ = build_targets(train, death_mode="censor_missing_death_at_last")
    trustii_id = test["trustii_id"].astype(str).to_numpy()

    print("  [1/4] landmark_3y_rsf (filter B)…")
    risk_lm = predict_landmark_3y_rsf(train, test, hep_t)
    print("  [2/4] rsf_baseline_v1_tuned (full train)…")
    risk_b1 = predict_rsf_baseline_v1_tuned(train, test, hep_t)
    print("  [3/4] rsf_longitudinal_summary (full train)…")
    risk_long = predict_rsf_longitudinal_summary(train, test, hep_t)
    print("  [4/4] permissive members for ensemble (full train)…")
    risk_xgb_lp = predict_xgb_cox_long_plus_meta(train, test, hep_t)
    risk_xgb_ls = predict_xgb_cox_long_sum(train, test, hep_t)

    # Write the four CSVs
    cdir = OUT / "component_test_predictions"
    pd.DataFrame({"trustii_id": trustii_id, "hepatic_risk": risk_lm}).to_csv(
        cdir / "test_landmark_3y_rsf.csv", index=False
    )
    pd.DataFrame({"trustii_id": trustii_id, "hepatic_risk": risk_b1}).to_csv(
        cdir / "test_rsf_baseline_v1_tuned.csv", index=False
    )
    pd.DataFrame({"trustii_id": trustii_id, "hepatic_risk": risk_long}).to_csv(
        cdir / "test_rsf_longitudinal_summary.csv", index=False
    )
    permissive_avg = (
        rankdata(risk_long) + rankdata(risk_xgb_lp) + rankdata(risk_xgb_ls)
    ) / 3.0
    pd.DataFrame({"trustii_id": trustii_id, "hepatic_risk": permissive_avg}).to_csv(
        cdir / "test_permissive_ensemble_avg.csv", index=False
    )
    (cdir / "README_components.md").write_text(COMPONENTS_README)

    # Sanity-check: cross-component Spearman on test
    rng = np.corrcoef(rankdata(risk_lm), rankdata(permissive_avg))[0, 1]
    rng_b1 = np.corrcoef(rankdata(risk_lm), rankdata(risk_b1))[0, 1]
    rng_long = np.corrcoef(rankdata(risk_long), rankdata(permissive_avg))[0, 1]
    print(f"  test Spearman: landmark vs permissive_avg = {rng:.3f}")
    print(f"  test Spearman: landmark vs baseline_v1    = {rng_b1:.3f}")
    print(f"  test Spearman: long_sum vs permissive_avg = {rng_long:.3f}")

    # Zip
    zip_path = ROOT / "handoff_for_merge.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in OUT.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(OUT.parent))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\nZip: {zip_path}  ({size_mb:.2f} MB)")
    if size_mb >= 5.0:
        print("  WARNING: ≥ 5 MB — please prune.")
    else:
        print("  OK (< 5 MB).")

    print("\nFiles:")
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT)}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
