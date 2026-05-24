"""Phase 2d — landmark analysis (Tier 2: clean longitudinal, fixed reference time).

For each landmark in {1, 2, 3, 5} years from Age_v1:
  1. Filter training rows: drop those whose hepatic event happened before the
     landmark (not at risk at landmark).
  2. Build features = baseline_v1 + (LOCF, slope) at landmark per longitudinal
     var.
  3. Run RSF and XGB-Cox at 5×10 CV.

Compare m−s to baseline_v1 anchor (RSF tuned 0.720 from Experiment 2).

Append rows to phase2_cv_results.csv with model = "{rsf,xgb_cox}_landmark"
and feature_set = "landmark_{Y}y". Save reports/phase2d_landmark_summary.md.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import REPORTS
from src.cv import evaluate_cv, summarize
from src.data import load_raw, build_targets
from src.features import build_landmark_features, at_risk_at_landmark
from src.models import make_rsf, make_xgb_cox

PHASE2_CV_CSV = REPORTS / "phase2_cv_results.csv"
SUMMARY_MD = REPORTS / "phase2d_landmark_summary.md"

LANDMARKS = [1.0, 2.0, 3.0, 5.0]
FINAL_SPLITS = 5
FINAL_REPEATS = 10

MODELS = [
    ("rsf",     lambda: make_rsf(n_estimators=300, min_samples_leaf=12,
                                 min_samples_split=71, max_features="log2")),
    ("xgb_cox", lambda: make_xgb_cox(n_estimators=320, learning_rate=0.012,
                                     max_depth=8, min_child_weight=1.88,
                                     subsample=0.75, colsample_bytree=0.69,
                                     reg_lambda=0.30)),
]
# Models use Optuna best_params from Experiment 2 (the baseline_v1 winners) —
# re-using those settings for fair Tier-2 comparison.


def append_cv_row(cv_res: pd.DataFrame, fs_name: str, model_name: str,
                  n_features: int, n_rows: int, n_events: int, elapsed: float):
    s = summarize(cv_res)
    row = {
        "endpoint": "hepatic", "death_mode": "n/a",
        "feature_set": fs_name,
        "leakage_risk": "low (Tier 2 fixed reference)",
        "model": model_name,
        "n_rows": n_rows, "n_events": n_events, "n_features": n_features,
        **s, "elapsed_s": round(elapsed, 1),
    }
    df_row = pd.DataFrame([row])
    if PHASE2_CV_CSV.exists():
        df_row.to_csv(PHASE2_CV_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(PHASE2_CV_CSV, mode="w", header=True, index=False)
    return s


def main():
    overall_start = time.time()
    train, _ = load_raw()
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e_full = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t_full = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df_full = train.loc[valid].reset_index(drop=True)
    print(f"Train (pre-landmark): n={len(e_full)}, events={int(e_full.sum())}",
          flush=True)

    rows = []
    for landmark in LANDMARKS:
        keep = at_risk_at_landmark(e_full, t_full, landmark)
        df_lm = df_full.loc[keep].reset_index(drop=True)
        e = e_full[keep]
        t = t_full[keep]
        n_rows, n_events = int(len(e)), int(e.sum())
        n_dropped = int((~keep).sum())
        print(f"\n--- Landmark {landmark}y --- "
              f"kept n={n_rows}, events={n_events}  "
              f"(dropped {n_dropped} pre-landmark events)", flush=True)

        X = build_landmark_features(df_lm, landmark_time=landmark)
        print(f"  features: {X.shape[1]}", flush=True)
        fs_name = f"landmark_{int(landmark*10)/10}y".replace(".0y", "y")

        for model_name, factory in MODELS:
            t0 = time.time()
            try:
                cv_res = evaluate_cv(factory(), X, e, t,
                                     n_splits=FINAL_SPLITS,
                                     n_repeats=FINAL_REPEATS)
                s = append_cv_row(cv_res, fs_name, f"{model_name}_landmark",
                                  X.shape[1], n_rows, n_events,
                                  time.time() - t0)
                ms = s["mean"] - s["std"]
                print(f"  [{model_name}]  mean={s['mean']:.4f} "
                      f"std={s['std']:.4f}  m−s={ms:.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
                rows.append({
                    "landmark_yrs": landmark,
                    "feature_set": fs_name,
                    "model": model_name,
                    "n_rows": n_rows, "n_events": n_events,
                    "n_features": X.shape[1],
                    "n_dropped_pre_landmark": n_dropped,
                    **s,
                })
            except Exception as ex:
                print(f"  [{model_name}] FAIL: {ex!r}", flush=True)

    # Summary
    df = pd.DataFrame(rows)
    if not df.empty:
        df["mean_minus_std"] = df["mean"] - df["std"]
        df_sorted = df.sort_values("mean_minus_std", ascending=False)

    BASELINE_V1_RSF_MS = 0.7196  # from Exp 2 final 5×10
    BASELINE_V1_XGB_MS = 0.7068

    lines = []
    add = lines.append
    add("# Phase 2d — Landmark analysis (Tier 2 features)\n")
    add(f"Date: 2026-04-27. Wall-clock: {(time.time()-overall_start)/60:.1f} min.\n")
    add("Features at each landmark = `baseline_v1` + LOCF + slope-since-v1 "
        "for each longitudinal variable. Tier 2 (clean longitudinal): cutoff "
        "is fixed by the calendar, not by the outcome. Training cohort drops "
        "patients whose hepatic event preceded the landmark (not-at-risk).\n")
    add(f"Anchors (Experiment 2 tuned, baseline_v1 only): "
        f"RSF m−s = **{BASELINE_V1_RSF_MS:.4f}**, XGB-Cox m−s = "
        f"**{BASELINE_V1_XGB_MS:.4f}**.\n")
    add("CV: 5-fold × 10-repeat stratified on hepatic event. Hyperparameters "
        "fixed at Experiment 2's tuned baseline_v1 winners (not re-tuned here).\n")

    add("\n## Results\n")
    add("| Landmark | Model | n / events | n_features | mean | std | m−s | Δm−s vs baseline_v1 |")
    add("|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        anchor = BASELINE_V1_RSF_MS if r["model"] == "rsf" else BASELINE_V1_XGB_MS
        delta = r["mean_minus_std"] - anchor
        add(f"| {r['landmark_yrs']:.1f}y | {r['model']} | "
            f"{r['n_rows']} / {r['n_events']} | {r['n_features']} | "
            f"{r['mean']:.4f} | {r['std']:.4f} | "
            f"{r['mean_minus_std']:.4f} | {delta:+.4f} |")

    if not df.empty:
        best = df_sorted.iloc[0]
        add(f"\n**Best landmark/model:** `{best['feature_set']}` × "
            f"{best['model']} at m−s {best['mean_minus_std']:.4f} "
            f"(mean {best['mean']:.4f}, std {best['std']:.4f}).")
        anchor = (BASELINE_V1_RSF_MS if best["model"] == "rsf"
                  else BASELINE_V1_XGB_MS)
        if best["mean_minus_std"] > anchor:
            add(f"\n→ Landmark beats baseline_v1 anchor by "
                f"{best['mean_minus_std']-anchor:+.4f} m−s. Real Tier-2 "
                f"longitudinal signal exists at this horizon.")
        else:
            add(f"\n→ Landmark does NOT beat baseline_v1 anchor "
                f"({best['mean_minus_std']-anchor:+.4f} m−s). First-visit "
                f"features are essentially sufficient at the honest tier.")

    add("\n## Files\n")
    add("- Rows appended to `reports/phase2_cv_results.csv` "
        "(feature_set=`landmark_<Y>y`, model=`{rsf,xgb_cox}_landmark`).")
    add("- Builder: `src/features.py::build_landmark_features` + "
        "`at_risk_at_landmark`.")

    SUMMARY_MD.write_text("\n".join(lines))
    print(f"\nWrote {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
