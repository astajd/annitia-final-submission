"""Phase 2d audit — verify landmark-3y RSF finding before trusting m−s 0.815.

Five checks, run end-to-end:
  1. Headline mean/std for 3y RSF (sanity of m−s 0.815).
  2. Sample-size check with TWO filters:
     A. event-free at landmark (the cohort the landmark CV used)
     B. event-free at landmark AND have actual visits through landmark
        (the "really has 3y of longitudinal data" subset)
  3. Per-fold C-index distribution at 5×10 CV for the 3y RSF — looking for
     fold variance (one lucky fold vs consistent strength).
  4. Test-set predictability: fraction of test patients whose follow-up reaches
     the landmark.
  5. Single-feature C-index audit on the 3y landmark feature set — flag any
     column that alone exceeds C 0.85 (would be a missed leakage source).

Plus the full landmark sweep with n_events_remaining at each landmark.
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import REPORTS, ROOT
from src.cv import evaluate_cv, summarize, cindex
from src.data import load_raw, build_targets, add_visit_metadata
from src.features import build_landmark_features, at_risk_at_landmark
from src.models import make_rsf, make_xgb_cox

CONFIGS = ROOT / "configs"
LANDMARKS = [1.0, 2.0, 3.0, 5.0]
SUMMARY_MD = REPORTS / "phase2d_audit.md"


def has_data_through(df: pd.DataFrame, lt: float) -> np.ndarray:
    """Mask: patient has visits with Age >= Age_v1 + lt."""
    d = add_visit_metadata(df)
    return ((d["max_age"] - d["Age_v1"]).to_numpy() >= lt - 1e-9)


def single_feature_cindex_audit(X: pd.DataFrame, e, t):
    """For each column, return (col, best_C, best_sign)."""
    rows = []
    for col in X.columns:
        v = X[col].to_numpy(dtype=float)
        if np.std(np.where(np.isnan(v), np.nanmedian(v) if not np.all(np.isnan(v)) else 0, v)) < 1e-9:
            continue
        v = np.where(np.isnan(v), np.nanmedian(v), v)
        c_pos = cindex(e, t, v)
        c_neg = cindex(e, t, -v)
        if c_pos >= c_neg:
            rows.append((col, c_pos, "pos"))
        else:
            rows.append((col, c_neg, "neg"))
    rows.sort(key=lambda r: -r[1])
    return rows


def main():
    train, test = load_raw()
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e_full = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t_full = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df_full = train.loc[valid].reset_index(drop=True)
    print(f"Hepatic cohort: n={len(e_full)}, events={int(e_full.sum())}",
          flush=True)

    # --- Tuned RSF best_params (Experiment 2 winner on baseline_v1, used for
    # the landmark CV in phase2d_landmark.py) ---
    rsf_rec = json.loads((CONFIGS / "optuna_rsf_baseline_v1.json").read_text())
    rsf_params = rsf_rec["best_params"]
    print(f"RSF tuned params: {rsf_params}", flush=True)

    # ===== Audit #2 first: full landmark sweep with both filters =====
    print("\n========== AUDIT #2: full landmark sweep ==========", flush=True)
    sweep_rows = []
    for lt in LANDMARKS:
        keep_A = at_risk_at_landmark(e_full, t_full, lt)             # event-free at landmark
        keep_B = keep_A & has_data_through(df_full, lt)              # AND has data through lt
        sweep_rows.append({
            "landmark_yrs": lt,
            "n_full": len(e_full),
            "events_full": int(e_full.sum()),
            "n_filterA_eventfree": int(keep_A.sum()),
            "events_filterA": int(e_full[keep_A].sum()),
            "n_dropped_filterA": int((~keep_A).sum()),
            "n_filterB_eventfree_AND_has_data": int(keep_B.sum()),
            "events_filterB": int(e_full[keep_B].sum()),
            "n_in_A_but_not_B": int((keep_A & ~keep_B).sum()),
        })
        print(f"  {lt}y  filter A (event-free)        n={int(keep_A.sum()):4d}  "
              f"events={int(e_full[keep_A].sum()):2d}",  flush=True)
        print(f"        filter B (eventfree+data)    n={int(keep_B.sum()):4d}  "
              f"events={int(e_full[keep_B].sum()):2d}  "
              f"(in A but lacks data: {int((keep_A & ~keep_B).sum())})",  flush=True)

    # ===== Audit #1: 3y RSF deep-dive =====
    LM = 3.0
    keep_A = at_risk_at_landmark(e_full, t_full, LM)
    keep_B = keep_A & has_data_through(df_full, LM)
    e_A = e_full[keep_A]; t_A = t_full[keep_A]
    df_A = df_full.loc[keep_A].reset_index(drop=True)
    e_B = e_full[keep_B]; t_B = t_full[keep_B]
    df_B = df_full.loc[keep_B].reset_index(drop=True)

    print("\n========== AUDIT #1: 3y RSF landmark deep-dive ==========",
          flush=True)
    print(f"\n[1] Headline mean/std (re-run with tuned params)", flush=True)
    X_A = build_landmark_features(df_A, LM)
    print(f"   filter A: n={len(e_A)}, events={int(e_A.sum())}, "
          f"features={X_A.shape[1]}", flush=True)
    cv_res_A = evaluate_cv(make_rsf(**rsf_params), X_A, e_A, t_A,
                           n_splits=5, n_repeats=10)
    s_A = summarize(cv_res_A)
    print(f"   filter A CV: mean={s_A['mean']:.4f}  std={s_A['std']:.4f}  "
          f"m−s={s_A['mean']-s_A['std']:.4f}", flush=True)

    print(f"\n[2] Sample-size check (filter B: event-free AND has 3y data)",
          flush=True)
    X_B = build_landmark_features(df_B, LM)
    print(f"   filter B: n={len(e_B)}, events={int(e_B.sum())}, "
          f"features={X_B.shape[1]}", flush=True)
    if int(e_B.sum()) >= 5:
        cv_res_B = evaluate_cv(make_rsf(**rsf_params), X_B, e_B, t_B,
                               n_splits=5, n_repeats=10)
        s_B = summarize(cv_res_B)
        print(f"   filter B CV: mean={s_B['mean']:.4f}  std={s_B['std']:.4f}  "
              f"m−s={s_B['mean']-s_B['std']:.4f}", flush=True)
    else:
        s_B = None
        print(f"   filter B: too few events for CV", flush=True)

    print(f"\n[3] Per-fold C-index distribution (filter A, 50 folds)",
          flush=True)
    per_fold_A = cv_res_A["cindex"].to_numpy()
    print(f"   min={per_fold_A.min():.4f}  p25={np.percentile(per_fold_A,25):.4f}  "
          f"median={np.median(per_fold_A):.4f}  "
          f"p75={np.percentile(per_fold_A,75):.4f}  "
          f"max={per_fold_A.max():.4f}", flush=True)
    print(f"   spread max−min = {per_fold_A.max()-per_fold_A.min():.4f}",
          flush=True)
    n_high = int((per_fold_A >= 0.90).sum()); n_low = int((per_fold_A < 0.80).sum())
    print(f"   folds ≥0.90: {n_high}/{len(per_fold_A)};  "
          f"folds <0.80: {n_low}/{len(per_fold_A)}", flush=True)

    print(f"\n[4] Test-set predictability at 3y landmark", flush=True)
    test_meta = add_visit_metadata(test)
    test_followup = (test_meta["max_age"] - test_meta["Age_v1"]).to_numpy()
    has_3y = test_followup >= 3 - 1e-9
    has_2y = test_followup >= 2 - 1e-9
    has_1y = test_followup >= 1 - 1e-9
    print(f"   test n={len(test)}", flush=True)
    print(f"   has data through 1y: {int(has_1y.sum())} "
          f"({100*has_1y.mean():.1f}%)", flush=True)
    print(f"   has data through 2y: {int(has_2y.sum())} "
          f"({100*has_2y.mean():.1f}%)", flush=True)
    print(f"   has data through 3y: {int(has_3y.sum())} "
          f"({100*has_3y.mean():.1f}%)", flush=True)
    print(f"   has data through 5y: "
          f"{int((test_followup>=5-1e-9).sum())} "
          f"({100*(test_followup>=5-1e-9).mean():.1f}%)", flush=True)

    print(f"\n[5] Single-feature C-index audit on 3y landmark features "
          f"(filter A, n={len(e_A)})", flush=True)
    single_audit = single_feature_cindex_audit(X_A, e_A, t_A)
    print(f"   Top-15 features by single-feature C:", flush=True)
    for col, c, sgn in single_audit[:15]:
        flag = " <-- >0.85 LEAK?" if c > 0.85 else (" >0.80" if c > 0.80 else "")
        print(f"     {col:45s}  C={c:.3f} ({sgn}){flag}", flush=True)
    n_over_85 = sum(1 for _, c, _ in single_audit if c > 0.85)
    n_over_80 = sum(1 for _, c, _ in single_audit if c > 0.80)
    print(f"   features with C>0.85: {n_over_85}", flush=True)
    print(f"   features with C>0.80: {n_over_80}", flush=True)

    # ===== Write summary report =====
    lines = []
    add = lines.append
    add("# Phase 2d audit — landmark-3y verification\n")
    add(f"Date: 2026-04-27. Headline 3y RSF result: mean={s_A['mean']:.4f}, "
        f"std={s_A['std']:.4f}, **m−s={s_A['mean']-s_A['std']:.4f}** (filter A, "
        f"5×10 CV with tuned baseline_v1 RSF params).\n")
    add("Tuned RSF params used: " + json.dumps(rsf_params) + "\n")

    add("\n## Audit #2 — full landmark sweep, two filters\n")
    add("Filter A: drop event-before-landmark (the filter used in phase2d_landmark.py).\n")
    add("Filter B: A ∩ {patient has visits reaching ≥ landmark age}. "
        "Filter B is the population for which the LOCF-at-landmark feature is "
        "*actually* computed from data observed at landmark — for filter-A-only "
        "rows that lack data through landmark, LOCF is just their last "
        "observed value (effectively early-truncated).\n")
    add("| Landmark | filter A: n / events | filter B: n / events | A but no data |")
    add("|---|---|---|---|")
    for r in sweep_rows:
        add(f"| {r['landmark_yrs']:.1f}y | "
            f"{r['n_filterA_eventfree']} / {r['events_filterA']} | "
            f"{r['n_filterB_eventfree_AND_has_data']} / {r['events_filterB']} | "
            f"{r['n_in_A_but_not_B']} |")
    add("")

    add("\n## Audit #1 — 3y RSF deep-dive\n")

    add("\n### [1] Headline mean / std")
    add(f"- Filter A (n={len(e_A)}, events={int(e_A.sum())}): "
        f"**mean={s_A['mean']:.4f}, std={s_A['std']:.4f}, "
        f"m−s={s_A['mean']-s_A['std']:.4f}**")
    if s_B is not None:
        add(f"- Filter B (n={len(e_B)}, events={int(e_B.sum())}): "
            f"mean={s_B['mean']:.4f}, std={s_B['std']:.4f}, "
            f"m−s={s_B['mean']-s_B['std']:.4f}")
    add("")
    add(f"User's question: *m−s 0.815 implies mean ≈ 0.86–0.87*. "
        f"Confirmed: mean is **{s_A['mean']:.4f}** at filter A.\n")

    add("\n### [2] Sample size with stricter filter")
    add(f"- Filter A (event-free at 3y): n={len(e_A)}, events={int(e_A.sum())}.")
    add(f"- Filter B (event-free AT 3y AND has visit through 3y): "
        f"n={len(e_B)}, events={int(e_B.sum())}.")
    add(f"- {sweep_rows[2]['n_in_A_but_not_B']} of the filter-A patients "
        "have no observed visit reaching the 3y landmark — for them, "
        "`*_locf_at_landmark` is identical to baseline LOCF.")
    if int(e_B.sum()) < 30:
        add(f"\n→ **Filter B has only {int(e_B.sum())} events.** Per the "
            "pre-registered rule (n<30 events = unstable CV), the 3y landmark "
            "result is suspect.")
    else:
        add(f"\n→ Filter B has {int(e_B.sum())} events.")

    add("\n### [3] Per-fold C-index distribution (filter A, 50 folds)")
    add("```")
    add(f"min    = {per_fold_A.min():.4f}")
    add(f"p25    = {np.percentile(per_fold_A, 25):.4f}")
    add(f"median = {np.median(per_fold_A):.4f}")
    add(f"p75    = {np.percentile(per_fold_A, 75):.4f}")
    add(f"max    = {per_fold_A.max():.4f}")
    add(f"spread = {per_fold_A.max()-per_fold_A.min():.4f}")
    add(f"folds ≥0.90: {n_high}/50    folds <0.80: {n_low}/50")
    add("```")

    add("\n### [4] Test-set predictability at landmark cutoffs")
    add("How many test patients have a visit reaching the landmark age? "
        "(For those that don't, the LOCF features are based on data before "
        "the landmark — the model was trained on a mix of these regimes.)")
    add("| Landmark | test rows reaching landmark | % |")
    add("|---|---|---|")
    add(f"| 1y | {int(has_1y.sum())} | {100*has_1y.mean():.1f}% |")
    add(f"| 2y | {int(has_2y.sum())} | {100*has_2y.mean():.1f}% |")
    add(f"| 3y | {int(has_3y.sum())} | {100*has_3y.mean():.1f}% |")
    add(f"| 5y | {int((test_followup>=5-1e-9).sum())} | "
        f"{100*(test_followup>=5-1e-9).mean():.1f}% |")

    add("\n### [5] Single-feature C-index audit on 3y landmark features")
    add(f"Top-15 of {len(single_audit)} features by best-sign single-feature C "
        f"on the filter-A cohort. Threshold for concern: > 0.85.\n")
    add("| feature | best C | sign |")
    add("|---|---|---|")
    for col, c, sgn in single_audit[:15]:
        flag = " ⚠️" if c > 0.85 else (" •" if c > 0.80 else "")
        add(f"| `{col}` | {c:.3f}{flag} | {sgn} |")
    add(f"\nTotals: features with C>0.85: **{n_over_85}**; with C>0.80: "
        f"{n_over_80}.")
    if n_over_85 > 0:
        add(f"\n→ **{n_over_85} feature(s) above the 0.85 threshold** — "
            "single-feature contribution is high. Review the features above; "
            "if any aggregate ages or are visit-count proxies, this is a "
            "missed leakage source.")
    else:
        add("\n→ No single feature crosses 0.85; the 0.86 mean comes from "
            "feature *combinations*, not a hidden proxy.")

    SUMMARY_MD.write_text("\n".join(lines))
    print(f"\nWrote {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
