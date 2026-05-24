"""Phase 3 Experiment 1 — trajectory-shape features.

Single experiment, four parts (no submissions built):
  1. Single-feature C-index audit on the trajectory_shape columns. Flag any
     single feature with C > 0.85 (would indicate hidden outcome leakage).
  2. CV bake-off: 5×10 CV hepatic m-s for RSF, XGB-Cox, LGBM-binary on
       (trajectory_shape, baseline_plus_shape).
     Tuned Tier-4 hyperparameters reused from Phase 2c (the
     longitudinal_summary / longitudinal_plus_meta winners).
     Reference: pull baseline_v1 + longitudinal_summary RSF/XGB numbers
     from reports/phase2_cv_results.csv.
  3. OOF rank-correlation analysis: for the best (model × feature_set),
     stitch repeat-averaged OOF, then compute Spearman vs cached OOF for
     landmark_3y_RSF, permissive_ensemble_avg, and blend_2way_optimal
     (0.30·rank(landmark) + 0.70·rank(permissive), per-fold then stitched).
     Decision: corr < 0.6 with ALL THREE → qualifies as a new ensemble member.
  4. Report → reports/phase3_exp1_trajectory_shape.{json,md}.

Stops after the report. Does not build any submission.
"""
from __future__ import annotations
import sys, json, time, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.config import REPORTS, ROOT
from src.cv import repeated_stratified_folds, cindex
from src.data import load_raw, build_targets
from src.features import build_features, SHAPE_VARS
from src.models import make_rsf, make_xgb_cox, make_lgbm_binary

CONFIGS = ROOT / "configs"
N_SPLITS, N_REPEATS, BASE_SEED = 5, 10, 42
LEAKAGE_FLAG_C = 0.85
QUALIFY_CORR = 0.6
BLEND_W_LANDMARK = 0.30  # current best Phase 2 blend weight
PERMISSIVE_MEMBERS = [
    "rsf_longitudinal_summary",
    "xgb_cox_longitudinal_plus_meta",
    "xgb_cox_longitudinal_summary",
]

OUT_JSON = REPORTS / "phase3_exp1_trajectory_shape.json"
OUT_MD = REPORTS / "phase3_exp1_trajectory_shape.md"
CACHE_PKL = REPORTS / "phase2_oof_perfold_cache.pkl"
PHASE2_CV_CSV = REPORTS / "phase2_cv_results.csv"


def load_optuna_params(cid: str) -> dict:
    return json.loads((CONFIGS / f"optuna_{cid}.json").read_text())["best_params"]


def main():
    t_overall = time.time()

    train, _ = load_raw()
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)
    n = len(e)
    print(f"Cohort: n={n}, events={int(e.sum())}", flush=True)

    # ---------- Build feature sets once ----------
    print("\n--- Building feature sets ---", flush=True)
    t0 = time.time()
    X_shape = build_features(df, "trajectory_shape")
    print(f"  trajectory_shape:  {X_shape.shape}  ({time.time()-t0:.1f}s)",
          flush=True)
    t0 = time.time()
    X_combined = build_features(df, "baseline_plus_shape")
    print(f"  baseline_plus_shape: {X_combined.shape}  ({time.time()-t0:.1f}s)",
          flush=True)

    shape_cols = [c for c in X_shape.columns if "_shape_" in c]
    print(f"  shape feature columns: {len(shape_cols)} "
          f"(expected {7 * len(SHAPE_VARS)} = 7 features × {len(SHAPE_VARS)} vars)",
          flush=True)

    # ---------- 1) Single-feature audit ----------
    print("\n--- Single-feature C-index audit (hepatic) ---", flush=True)
    audit_rows = []
    for col in shape_cols:
        vals = X_shape[col].to_numpy(dtype=float)
        if np.all(np.isnan(vals)):
            continue
        med = np.nanmedian(vals)
        v = np.where(np.isnan(vals), med, vals)
        c_pos = cindex(e, t, v)
        c_neg = cindex(e, t, -v)
        c_best = max(c_pos, c_neg)
        sign = "+" if c_pos >= c_neg else "-"
        audit_rows.append({
            "feature": col,
            "c_pos": float(c_pos), "c_neg": float(c_neg),
            "c_best": float(c_best), "best_sign": sign,
            "leak_flag": bool(c_best > LEAKAGE_FLAG_C),
        })
    audit_df = pd.DataFrame(audit_rows).sort_values(
        "c_best", ascending=False).reset_index(drop=True)
    flagged = audit_df[audit_df["leak_flag"]]
    print(f"  audited {len(audit_df)} shape features", flush=True)
    print(f"  top 10 by best-sign C:", flush=True)
    print(audit_df.head(10).to_string(index=False,
          float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"\n  flagged (C > {LEAKAGE_FLAG_C}): {len(flagged)}", flush=True)
    if len(flagged):
        print(flagged.to_string(index=False,
              float_format=lambda x: f"{x:.3f}"), flush=True)

    # ---------- 2) CV bake-off ----------
    print("\n--- CV bake-off (5×10 CV) ---", flush=True)
    rsf_params = load_optuna_params("rsf_longitudinal_summary")
    xgb_params = load_optuna_params("xgb_cox_longitudinal_summary")
    lgbm_params = load_optuna_params("lgbm_bin_longitudinal_plus_meta")
    print(f"  rsf_params: {rsf_params}", flush=True)
    print(f"  xgb_params: {xgb_params}", flush=True)
    print(f"  lgbm_params: {lgbm_params}", flush=True)

    def make_factories():
        return [
            ("rsf",      make_rsf(**rsf_params)),
            ("xgb_cox",  make_xgb_cox(**xgb_params)),
            ("lgbm_bin", make_lgbm_binary(**{
                **lgbm_params,
                "horizon": float(lgbm_params.get("horizon", 5))})),
        ]

    feature_sets = [
        ("trajectory_shape",   X_shape),
        ("baseline_plus_shape", X_combined),
    ]

    bakeoff_rows = []
    perfold = {}    # (model_name, fs_name) -> list of {repeat, fold, val_idx, risk}

    for fs_name, X in feature_sets:
        for m_name, factory in make_factories():
            t_cv = time.time()
            cis = []
            preds = []
            for r, f, tr_idx, va_idx in repeated_stratified_folds(
                    e, N_SPLITS, N_REPEATS, BASE_SEED):
                risk = factory(X.iloc[tr_idx], e[tr_idx], t[tr_idx],
                               X.iloc[va_idx])
                ci = cindex(e[va_idx], t[va_idx], risk)
                cis.append(ci)
                preds.append({"repeat": r, "fold": f,
                              "val_idx": np.asarray(va_idx),
                              "risk": np.asarray(risk)})
            cis = np.asarray(cis)
            mean = float(cis.mean())
            std = float(cis.std(ddof=1))
            ms = mean - std
            elapsed = time.time() - t_cv
            bakeoff_rows.append({
                "model": m_name, "feature_set": fs_name,
                "n_features": X.shape[1],
                "mean": mean, "std": std, "mean_minus_std": ms,
                "elapsed_s": round(elapsed, 1),
            })
            perfold[(m_name, fs_name)] = preds
            print(f"  {m_name:9s} × {fs_name:21s}  "
                  f"mean={mean:.4f} std={std:.4f} m-s={ms:.4f} "
                  f"({elapsed:.0f}s)", flush=True)

    bakeoff_df = pd.DataFrame(bakeoff_rows).sort_values(
        "mean_minus_std", ascending=False).reset_index(drop=True)
    print("\n  bake-off, sorted by m-s:", flush=True)
    print(bakeoff_df.to_string(index=False,
          float_format=lambda x: f"{x:.4f}"), flush=True)

    # Reference baseline_v1 + longitudinal_summary numbers from phase2_cv_results.csv
    print("\n  Reference numbers from phase2_cv_results.csv:", flush=True)
    reference_table = []
    if PHASE2_CV_CSV.exists():
        ph2 = pd.read_csv(PHASE2_CV_CSV)
        ref_filter = (
            (ph2["endpoint"] == "hepatic")
            & (ph2["death_mode"].fillna("n/a") == "n/a")
            & (ph2["feature_set"].isin(["baseline_v1", "longitudinal_summary"]))
            & (ph2["model"].isin(["rsf", "xgb_cox", "lgbm_bin"]))
        )
        ref_df = ph2.loc[ref_filter, ["feature_set", "model",
                                      "mean", "std"]].copy()
        ref_df["mean_minus_std"] = ref_df["mean"] - ref_df["std"]
        ref_df = ref_df.sort_values(
            "mean_minus_std", ascending=False).reset_index(drop=True)
        print(ref_df.to_string(index=False,
              float_format=lambda x: f"{x:.4f}"), flush=True)
        reference_table = ref_df.to_dict(orient="records")
    else:
        print(f"  (csv not found: {PHASE2_CV_CSV})", flush=True)

    # ---------- 3) Correlation analysis ----------
    print("\n--- OOF rank-correlation analysis ---", flush=True)
    best = bakeoff_df.iloc[0]
    best_key = (best["model"], best["feature_set"])
    print(f"  Best new model: {best_key[0]} × {best_key[1]} "
          f"(m-s={best['mean_minus_std']:.4f})", flush=True)

    # Stitch new model's OOF (rank-stitch since fold-level distributions vary)
    def stitch_rank_avg(records, key="risk"):
        sumv = np.zeros(n)
        cnt = np.zeros(n, dtype=int)
        for rec in records:
            va = rec["val_idx"]
            r = rec[key] if isinstance(key, str) else key(rec)
            sumv[va] += rankdata(r)
            cnt[va] += 1
        out = np.full(n, np.nan)
        out[cnt > 0] = sumv[cnt > 0] / cnt[cnt > 0]
        return out

    oof_new = stitch_rank_avg(perfold[best_key])

    # Load cached Phase 2 OOF
    if not CACHE_PKL.exists():
        raise SystemExit(f"Missing cached OOF: {CACHE_PKL}. "
                         "Re-run phase2_audit_v2.py to regenerate.")
    with open(CACHE_PKL, "rb") as fh:
        cache = pickle.load(fh)
    print(f"  loaded {len(cache)} fold records from {CACHE_PKL}", flush=True)

    def perm_avg_per_fold(rec):
        return np.mean([rankdata(rec[f"risk_{cid}"])
                        for cid in PERMISSIVE_MEMBERS], axis=0)

    def blend_per_fold(rec):
        return (BLEND_W_LANDMARK * rankdata(rec["risk_lm"])
                + (1 - BLEND_W_LANDMARK) * perm_avg_per_fold(rec))

    oof_lm   = stitch_rank_avg(cache, key="risk_lm")
    oof_perm = stitch_rank_avg(cache, key=perm_avg_per_fold)
    oof_blend = stitch_rank_avg(cache, key=blend_per_fold)

    def safe_spearman(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 100:
            return float("nan")
        rho, _ = spearmanr(a[mask], b[mask])
        return float(rho)

    corrs = {
        "vs_landmark_3y_RSF":         safe_spearman(oof_new, oof_lm),
        "vs_permissive_ensemble_avg": safe_spearman(oof_new, oof_perm),
        "vs_blend_2way_optimal":      safe_spearman(oof_new, oof_blend),
    }
    print(f"\n  OOF Spearman vs Phase 2 components (full hepatic-valid cohort):",
          flush=True)
    for k, v in corrs.items():
        print(f"    {k:35s}  ρ = {v:.3f}", flush=True)

    qualifies_all = all(c < QUALIFY_CORR for c in corrs.values())
    qualifies_any_below = [k for k, v in corrs.items() if v < QUALIFY_CORR]
    print(f"\n  Qualifies as ensemble member (corr < {QUALIFY_CORR} vs ALL three)? "
          f"{'YES' if qualifies_all else 'NO'}", flush=True)
    if not qualifies_all:
        print(f"    below threshold vs: {qualifies_any_below or 'none'}",
              flush=True)

    # ---------- 4) Report ----------
    diag = {
        "experiment": "Phase 3 Exp 1 — trajectory-shape features",
        "cv_protocol": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS,
                        "base_seed": BASE_SEED},
        "single_feature_audit": {
            "n_audited": int(len(audit_df)),
            "leakage_flag_threshold": LEAKAGE_FLAG_C,
            "n_flagged": int(len(flagged)),
            "flagged_features": flagged.to_dict(orient="records"),
            "top10": audit_df.head(10).to_dict(orient="records"),
        },
        "bakeoff": bakeoff_df.to_dict(orient="records"),
        "reference_phase2_cv": reference_table,
        "best_new": {
            "model": best["model"], "feature_set": best["feature_set"],
            "mean": float(best["mean"]), "std": float(best["std"]),
            "mean_minus_std": float(best["mean_minus_std"]),
            "n_features": int(best["n_features"]),
        },
        "oof_correlation_vs_phase2": corrs,
        "decision_rule": {
            "qualify_corr": QUALIFY_CORR,
            "qualifies_as_ensemble_member": qualifies_all,
            "below_threshold_vs": qualifies_any_below,
        },
    }
    OUT_JSON.write_text(json.dumps(diag, indent=2, default=str))

    # Markdown
    lines = [
        "# Phase 3 Experiment 1 — Trajectory-shape features\n",
        f"Date: 2026-04-28. CV: 5-fold × 10-repeat stratified on hepatic event "
        f"(50 folds, base_seed={BASE_SEED}). Cohort: n={n}, events={int(e.sum())}.\n",
        f"Feature set: per-variable shape summaries over **{len(SHAPE_VARS)} "
        f"longitudinal vars** "
        f"(`{', '.join(SHAPE_VARS)}`). 7 features per variable: "
        f"`present, monotonicity, smoothness, stability, recent_ratio, "
        f"volatility, acceleration` → **{7 * len(SHAPE_VARS)}** trajectory-"
        f"shape columns. Tier 4 (uses all visits; designed to be robust to "
        f"post-event ambiguity).\n",
        "## 1. Single-feature C-index audit\n",
        f"Audited {len(audit_df)} shape features. Best-sign C-index per feature "
        f"(positive vs negative orientation, max).",
        f"\n**Leakage threshold:** C > {LEAKAGE_FLAG_C}. "
        f"**Flagged:** {len(flagged)}.\n",
        "Top 15 by best-sign C:",
        "| feature | c_pos | c_neg | c_best | best_sign | leak_flag |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in audit_df.head(15).iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['c_pos']:.3f} | {r['c_neg']:.3f} | "
            f"**{r['c_best']:.3f}** | {r['best_sign']} | "
            f"{'⚠️' if r['leak_flag'] else ''} |")

    lines.append("\n## 2. CV bake-off (5×10, hepatic m-s)\n")
    lines.append("| model | feature_set | n_feat | mean | std | m-s | elapsed |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in bakeoff_df.iterrows():
        lines.append(
            f"| {r['model']} | `{r['feature_set']}` | {int(r['n_features'])} | "
            f"{r['mean']:.4f} | {r['std']:.4f} | **{r['mean_minus_std']:.4f}** "
            f"| {r['elapsed_s']:.0f}s |")

    if reference_table:
        lines.append("\nReference (from `phase2_cv_results.csv`):\n")
        lines.append("| feature_set | model | mean | std | m-s |")
        lines.append("|---|---|---|---|---|")
        for r in reference_table:
            lines.append(
                f"| `{r['feature_set']}` | {r['model']} | "
                f"{r['mean']:.4f} | {r['std']:.4f} | "
                f"{r['mean_minus_std']:.4f} |")

    lines.append("\n## 3. OOF rank-correlation vs Phase 2 components\n")
    lines.append(f"Best new model: **{best_key[0]} × `{best_key[1]}`** "
                 f"(m-s {best['mean_minus_std']:.4f}). OOF stitched by "
                 f"rank-averaging across the 50 folds and Spearman vs the "
                 f"cached Phase 2 OOF.")
    lines.append("\n| comparison | Spearman ρ |")
    lines.append("|---|---|")
    for k, v in corrs.items():
        lines.append(f"| {k.replace('_', ' ')} | **{v:.3f}** |")

    lines.append(f"\n**Decision rule:** corr < {QUALIFY_CORR} vs all three → "
                 f"qualifies as ensemble member.")
    lines.append(f"\n→ {'**QUALIFIES**' if qualifies_all else '**DOES NOT QUALIFY**'} "
                 f"as a new ensemble member.")
    if not qualifies_all:
        lines.append(f"  (corr below threshold only vs: "
                     f"{qualifies_any_below or '*none*'})")

    lines.append("\n## Verdict / next steps\n")
    if qualifies_all:
        lines.append("- Trajectory-shape passes the single-feature leakage "
                     "audit and clears the 0.6 OOF rank-correlation bar "
                     "vs landmark, permissive_ensemble, and the 30/70 blend. "
                     "Eligible to add to the Phase 2 ensemble.")
    else:
        lines.append("- Trajectory-shape's best model does NOT clear the "
                     "0.6 corr bar vs all three Phase 2 components. Adding "
                     "it to the ensemble would be redundant with what we "
                     "already have. Stops here per the pre-registered rule.")
    if len(flagged) > 0:
        lines.append(f"- ⚠️ {len(flagged)} shape feature(s) above the "
                     f"{LEAKAGE_FLAG_C} single-feature C threshold — "
                     "investigate before any downstream use.")

    lines.append("\n*No submissions built. Ends here per spec.*\n")
    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}", flush=True)
    print(f"\nTotal elapsed: {(time.time()-t_overall)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
