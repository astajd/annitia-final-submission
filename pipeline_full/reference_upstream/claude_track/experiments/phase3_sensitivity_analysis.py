"""Phase 3 Priority 1 — methodology sensitivity analysis.

Deliverable for the qualitative writeup, not a new submission attempt.

Builds the 30/70 blend's components on three feature regimes and reports
hepatic 5×10 CV m-s for each:
  - Tier 1 (honest):    30% RSF/baseline_v1 + 70% XGB-Cox/baseline_v1
  - Tier 2 (landmark):  30% RSF/landmark_2y + 70% XGB-Cox/landmark_2y
  - Tier 4 (current):   30% RSF/landmark_3y (filter A) + 70% permissive
                        rank-avg(RSF/long_summary, XGB/long_plus_meta,
                                 XGB/long_summary)

The user-documented +0.07 correction (CV-mean → LB) is applied as an
explicit, transparent BACK-OF-ENVELOPE estimate, with the explicit caveat
that the correction was derived from a single submission and the LB-noise
analysis (phase3_lb_decoupling) shows ±0.02-0.04 jitter on top of it.

Output: reports/sensitivity_analysis.md (writeup-ready table) + .json.
"""
from __future__ import annotations
import sys, json, time, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import REPORTS, ROOT
from src.cv import repeated_stratified_folds, cindex
from src.data import load_raw, build_targets
from src.features import (build_features, build_landmark_features,
                          at_risk_at_landmark)
from src.models import make_rsf, make_xgb_cox

CONFIGS = ROOT / "configs"
N_SPLITS, N_REPEATS, BASE_SEED = 5, 10, 42
LANDMARK_TIER_2 = 2.0
LANDMARK_TIER_4 = 3.0
PERMISSIVE_MEMBERS = ["rsf_longitudinal_summary",
                      "xgb_cox_longitudinal_plus_meta",
                      "xgb_cox_longitudinal_summary"]
BLEND_W_LANDMARK = 0.30
LB_CORRECTION = 0.07     # observed CV_mean_hep → hep_LB on _optimal
DEATH_LB_ASSUMED = 0.96  # at saturation
W_HEP, W_DEATH = 0.7, 0.3

OUT_JSON = REPORTS / "sensitivity_analysis.json"
OUT_MD = REPORTS / "sensitivity_analysis.md"
HEP_CACHE_PKL = REPORTS / "phase2_oof_perfold_cache.pkl"


def load_optuna_params(cid: str) -> dict:
    return json.loads((CONFIGS / f"optuna_{cid}.json").read_text())["best_params"]


def perfold_oof(factory_fn, X, e, t, *, subfilter=None):
    """5×10 OOF: returns list of {repeat,fold,val_idx,risk}."""
    out = []
    for r, f, tr_idx, va_idx in repeated_stratified_folds(
            e, N_SPLITS, N_REPEATS, BASE_SEED):
        if subfilter is not None:
            tr = tr_idx[subfilter[tr_idx]]
        else:
            tr = tr_idx
        risk = factory_fn()(X.iloc[tr], e[tr], t[tr], X.iloc[va_idx])
        out.append({"repeat": r, "fold": f,
                    "val_idx": np.asarray(va_idx),
                    "risk": np.asarray(risk)})
    return out


def cv_blend_metrics(perfold_a, perfold_b, e, t, w_a):
    """Per-fold C-index of the blend w_a·rank(a) + (1-w_a)·rank(b)."""
    cis = []
    for ra, rb in zip(perfold_a, perfold_b):
        assert ra["repeat"] == rb["repeat"] and ra["fold"] == rb["fold"]
        va = ra["val_idx"]
        blend = w_a * rankdata(ra["risk"]) + (1 - w_a) * rankdata(rb["risk"])
        cis.append(cindex(e[va], t[va], blend))
    cis = np.asarray(cis)
    return {"mean": float(cis.mean()),
            "std": float(cis.std(ddof=1)),
            "mean_minus_std": float(cis.mean() - cis.std(ddof=1))}


def cv_single(perfold, e, t):
    cis = []
    for rec in perfold:
        va = rec["val_idx"]
        cis.append(cindex(e[va], t[va], rec["risk"]))
    cis = np.asarray(cis)
    return {"mean": float(cis.mean()),
            "std": float(cis.std(ddof=1)),
            "mean_minus_std": float(cis.mean() - cis.std(ddof=1))}


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

    rsf_baseline = load_optuna_params("rsf_baseline_v1")
    xgb_baseline = load_optuna_params("xgb_cox_baseline_v1")
    print(f"  rsf_baseline_v1 params: {rsf_baseline}", flush=True)
    print(f"  xgb_cox_baseline_v1 params: {xgb_baseline}", flush=True)

    results = {}

    # -----------------------------------------------------------------
    # Tier 1 — baseline_v1 (RSF + XGB)
    # -----------------------------------------------------------------
    print("\n=== Tier 1: baseline_v1 (RSF + XGB-Cox) ===", flush=True)
    X_v1 = build_features(df, "baseline_v1")
    print(f"  X_v1 shape: {X_v1.shape}", flush=True)

    t0 = time.time()
    rsf_v1_pf = perfold_oof(lambda: make_rsf(**rsf_baseline), X_v1, e, t)
    rsf_v1_m = cv_single(rsf_v1_pf, e, t)
    print(f"  RSF/v1: mean={rsf_v1_m['mean']:.4f} std={rsf_v1_m['std']:.4f} "
          f"m-s={rsf_v1_m['mean_minus_std']:.4f} ({time.time()-t0:.0f}s)",
          flush=True)
    t0 = time.time()
    xgb_v1_pf = perfold_oof(lambda: make_xgb_cox(**xgb_baseline), X_v1, e, t)
    xgb_v1_m = cv_single(xgb_v1_pf, e, t)
    print(f"  XGB/v1: mean={xgb_v1_m['mean']:.4f} std={xgb_v1_m['std']:.4f} "
          f"m-s={xgb_v1_m['mean_minus_std']:.4f} ({time.time()-t0:.0f}s)",
          flush=True)
    blend_v1 = cv_blend_metrics(rsf_v1_pf, xgb_v1_pf, e, t, BLEND_W_LANDMARK)
    print(f"  blend (30·RSF + 70·XGB on v1): mean={blend_v1['mean']:.4f} "
          f"std={blend_v1['std']:.4f} m-s={blend_v1['mean_minus_std']:.4f}",
          flush=True)
    results["tier_1_baseline_v1"] = {
        "rsf": rsf_v1_m, "xgb": xgb_v1_m, "blend_30_70": blend_v1,
        "n_features": int(X_v1.shape[1]),
        "components_share_features": True,
    }

    # -----------------------------------------------------------------
    # Tier 2 — landmark_2y (RSF + XGB)
    # -----------------------------------------------------------------
    print("\n=== Tier 2: landmark_2y (RSF + XGB-Cox) ===", flush=True)
    keep_A_2y = at_risk_at_landmark(e, t, LANDMARK_TIER_2)
    X_lm2 = build_landmark_features(df, LANDMARK_TIER_2)
    print(f"  X_lm2 shape: {X_lm2.shape}; filter A (event-free at "
          f"{LANDMARK_TIER_2}y): n={int(keep_A_2y.sum())}", flush=True)

    t0 = time.time()
    rsf_lm2_pf = perfold_oof(lambda: make_rsf(**rsf_baseline), X_lm2, e, t,
                             subfilter=keep_A_2y)
    rsf_lm2_m = cv_single(rsf_lm2_pf, e, t)
    print(f"  RSF/lm2 (filter A train): mean={rsf_lm2_m['mean']:.4f} "
          f"std={rsf_lm2_m['std']:.4f} m-s={rsf_lm2_m['mean_minus_std']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    xgb_lm2_pf = perfold_oof(lambda: make_xgb_cox(**xgb_baseline), X_lm2, e, t,
                             subfilter=keep_A_2y)
    xgb_lm2_m = cv_single(xgb_lm2_pf, e, t)
    print(f"  XGB/lm2 (filter A train): mean={xgb_lm2_m['mean']:.4f} "
          f"std={xgb_lm2_m['std']:.4f} m-s={xgb_lm2_m['mean_minus_std']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    blend_lm2 = cv_blend_metrics(rsf_lm2_pf, xgb_lm2_pf, e, t, BLEND_W_LANDMARK)
    print(f"  blend (30·RSF + 70·XGB on lm2): mean={blend_lm2['mean']:.4f} "
          f"std={blend_lm2['std']:.4f} m-s={blend_lm2['mean_minus_std']:.4f}",
          flush=True)
    results["tier_2_landmark_2y"] = {
        "rsf": rsf_lm2_m, "xgb": xgb_lm2_m, "blend_30_70": blend_lm2,
        "n_features": int(X_lm2.shape[1]),
        "filter_A_n": int(keep_A_2y.sum()),
        "components_share_features": True,
    }

    # -----------------------------------------------------------------
    # Tier 4 — current production blend (cached)
    # -----------------------------------------------------------------
    print("\n=== Tier 4: current production blend (cached) ===", flush=True)
    if not HEP_CACHE_PKL.exists():
        raise SystemExit(f"Missing cache: {HEP_CACHE_PKL}")
    with open(HEP_CACHE_PKL, "rb") as fh:
        cache = pickle.load(fh)
    print(f"  loaded {len(cache)} per-fold records", flush=True)

    # Per-fold blend for Tier 4 = 30% lm + 70% perm_avg
    cis_blend = []
    cis_lm = []
    cis_perm = []
    for rec in cache:
        va = rec["val_idx"]
        perm_avg = np.mean([rankdata(rec[f"risk_{cid}"])
                            for cid in PERMISSIVE_MEMBERS], axis=0)
        blend = (BLEND_W_LANDMARK * rankdata(rec["risk_lm"])
                 + (1 - BLEND_W_LANDMARK) * perm_avg)
        cis_blend.append(cindex(e[va], t[va], blend))
        cis_lm.append(cindex(e[va], t[va], rec["risk_lm"]))
        cis_perm.append(cindex(e[va], t[va], perm_avg))
    cis_blend = np.asarray(cis_blend)
    cis_lm = np.asarray(cis_lm)
    cis_perm = np.asarray(cis_perm)
    blend_t4 = {"mean": float(cis_blend.mean()),
                "std": float(cis_blend.std(ddof=1)),
                "mean_minus_std": float(cis_blend.mean() - cis_blend.std(ddof=1))}
    lm_t4 = {"mean": float(cis_lm.mean()),
             "std": float(cis_lm.std(ddof=1)),
             "mean_minus_std": float(cis_lm.mean() - cis_lm.std(ddof=1))}
    perm_t4 = {"mean": float(cis_perm.mean()),
               "std": float(cis_perm.std(ddof=1)),
               "mean_minus_std": float(cis_perm.mean() - cis_perm.std(ddof=1))}
    print(f"  landmark_3y_RSF (filter A): mean={lm_t4['mean']:.4f} "
          f"std={lm_t4['std']:.4f} m-s={lm_t4['mean_minus_std']:.4f}",
          flush=True)
    print(f"  permissive_ensemble: mean={perm_t4['mean']:.4f} "
          f"std={perm_t4['std']:.4f} m-s={perm_t4['mean_minus_std']:.4f}",
          flush=True)
    print(f"  blend (30·lm3 + 70·perm): mean={blend_t4['mean']:.4f} "
          f"std={blend_t4['std']:.4f} m-s={blend_t4['mean_minus_std']:.4f} "
          f"  ← actual phase2_blend_2way_optimal (LB 0.8965)", flush=True)
    results["tier_4_current"] = {
        "lm3": lm_t4, "perm_ensemble": perm_t4, "blend_30_70": blend_t4,
        "components_share_features": False,
        "observed_LB": 0.8965,
    }

    # -----------------------------------------------------------------
    # Compute LB-implied estimates (back-of-envelope)
    # -----------------------------------------------------------------
    def lb_implied(hep_mean):
        """LB ≈ 0.7 × (hep_CV_mean + 0.07) + 0.3 × death_LB_assumed."""
        return W_HEP * (hep_mean + LB_CORRECTION) + W_DEATH * DEATH_LB_ASSUMED

    rows = []
    for label, r in results.items():
        b = r["blend_30_70"]
        rows.append({
            "tier": label,
            "blend_mean": b["mean"], "blend_std": b["std"],
            "blend_mean_minus_std": b["mean_minus_std"],
            "lb_implied": lb_implied(b["mean"]),
        })
    summary_df = pd.DataFrame(rows)

    print("\n=== Sensitivity table ===", flush=True)
    print(summary_df.to_string(index=False,
          float_format=lambda x: f"{x:.4f}"), flush=True)

    # -----------------------------------------------------------------
    # Persist
    # -----------------------------------------------------------------
    diag = {
        "experiment": "Phase 3 Priority 1 — methodology sensitivity",
        "cv_protocol": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS,
                        "base_seed": BASE_SEED},
        "blend_weights": {"landmark_or_RSF": BLEND_W_LANDMARK,
                          "permissive_or_XGB": 1 - BLEND_W_LANDMARK},
        "lb_correction": LB_CORRECTION,
        "death_lb_assumed": DEATH_LB_ASSUMED,
        "results": results,
        "summary_table": summary_df.to_dict(orient="records"),
        "caveat": ("LB-implied uses CV_mean + 0.07 and death_LB_assumed "
                   "= 0.96; the +0.07 correction was empirically derived "
                   "from a single submission (phase2_blend_2way_optimal); "
                   "the LB-noise analysis (phase3_lb_decoupling) shows "
                   "±0.02-0.04 jitter on top of any point estimate."),
    }
    OUT_JSON.write_text(json.dumps(diag, indent=2, default=str))

    # ---- markdown ----
    lines = [
        "# Methodology sensitivity analysis\n",
        "Date: 2026-04-28. CV: 5-fold × 10-repeat stratified on hepatic event "
        f"(50 folds, base_seed={BASE_SEED}). Cohort: n={n}, events="
        f"{int(e.sum())}.\n",
        "Builds the same 30/70 blend recipe at three tier choices to show "
        "how methodology liberality affects CV performance:\n",
        "- **Tier 1 (honest, single fixed timestamp)** — features computed at "
        "v1 only; baseline_v1.",
        "- **Tier 2 (clean longitudinal, fixed reference time)** — features "
        f"computed at Age_v1 + {LANDMARK_TIER_2}y (LOCF + slope per longitudinal "
        "var). Patients with hepatic event before the landmark dropped from "
        "training (filter A); predictions made on full validation fold.",
        "- **Tier 4 (full-permissive, all visits)** — current production blend: "
        f"30% landmark_3y_RSF (filter A) + 70% rank-avg of three longitudinal "
        f"models on `longitudinal_summary` and `longitudinal_plus_meta`. This "
        f"is the actual `phase2_blend_2way_optimal` (LB 0.8965).\n",
        "## Results\n",
        "| Tier | n_features | RSF mean | XGB mean | Blend (30/70) mean | "
        "Blend std | Blend m−s | LB-implied (CV+0.07) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    # Tier 1 row
    r = results["tier_1_baseline_v1"]
    lines.append(
        f"| **Tier 1** v1-only | {r['n_features']} | "
        f"{r['rsf']['mean']:.4f} | {r['xgb']['mean']:.4f} | "
        f"{r['blend_30_70']['mean']:.4f} | "
        f"{r['blend_30_70']['std']:.4f} | "
        f"**{r['blend_30_70']['mean_minus_std']:.4f}** | "
        f"{lb_implied(r['blend_30_70']['mean']):.4f} |")
    r = results["tier_2_landmark_2y"]
    lines.append(
        f"| **Tier 2** landmark_2y (filter A n={r['filter_A_n']}) | "
        f"{r['n_features']} | "
        f"{r['rsf']['mean']:.4f} | {r['xgb']['mean']:.4f} | "
        f"{r['blend_30_70']['mean']:.4f} | "
        f"{r['blend_30_70']['std']:.4f} | "
        f"**{r['blend_30_70']['mean_minus_std']:.4f}** | "
        f"{lb_implied(r['blend_30_70']['mean']):.4f} |")
    r = results["tier_4_current"]
    lines.append(
        f"| **Tier 4** current (lm3 + permissive ensemble) | n/a "
        f"(heterogeneous components) | "
        f"{r['lm3']['mean']:.4f} (lm3) | {r['perm_ensemble']['mean']:.4f} (perm) | "
        f"{r['blend_30_70']['mean']:.4f} | "
        f"{r['blend_30_70']['std']:.4f} | "
        f"**{r['blend_30_70']['mean_minus_std']:.4f}** | "
        f"{lb_implied(r['blend_30_70']['mean']):.4f} (observed LB **0.8965**) |")

    lines.append("\n*LB-implied uses 0.7·(CV_hep_mean + 0.07) + 0.3·0.96. "
                 "The +0.07 correction was empirically derived from "
                 "phase2_blend_2way_optimal (CV mean 0.8147 → hepatic LB "
                 "≈ 0.87). The LB-noise analysis (`phase3_lb_decoupling.md`) "
                 "shows ±0.02-0.04 jitter on top of any point estimate.*\n")

    lines.append("\n## Methodology narrative\n")
    lines.append(
        "Tier 1 uses only baseline-visit features (gender, T2DM, FibroScan_v1, "
        "etc.) — fully honest at the cost of discarding follow-up information. "
        "Tier 2 uses LOCF and slope at a fixed clinical reference time "
        f"(Age_v1 + {LANDMARK_TIER_2}y); patients whose hepatic event preceded "
        "the landmark are excluded from training to avoid outcome-conditional "
        "feature construction. Tier 4 uses all visit history regardless of "
        "timing, which the organizer's forum post explicitly blesses for "
        "the quantitative track. The 30/70 blend recipe (one risk-stable "
        "model + one diverse-feature model, OOF-stacked weights from Phase 2) "
        "is held constant across all three tiers.")
    lines.append(
        "\nThe Tier 4 production blend has structural diversity that Tier 1 "
        "and Tier 2 lack: its two components (`landmark_3y_RSF` and the "
        "`permissive_ensemble`) use *different feature sets* (Tier 2 vs "
        "Tier 4 features). Tier 1 and Tier 2 in this comparison use a single "
        "feature set per row, so the two component models share inputs and "
        "are highly correlated — the blend gain over the better single model "
        "is therefore smaller. This is the load-bearing methodological "
        "observation: blending helps when components are diverse in *both* "
        "model class AND feature view, not just in model class.")

    lines.append("\n## Caveats\n")
    lines.append("- Hyperparameters at each tier come from the Phase 2 Optuna "
                 "tuning of `rsf_baseline_v1` and `xgb_cox_baseline_v1`. "
                 "The Tier 4 components use their own tuned hyperparameters "
                 "(landmark_3y RSF reuses baseline_v1 RSF params; permissive "
                 "ensemble members are individually tuned).")
    lines.append("- `LB-implied` is back-of-envelope. CV→LB has been "
                 "noisy in this competition (three CV→LB inversions among "
                 "the last four submissions). Use the m-s column for "
                 "intra-tier ranking; the LB-implied column is illustrative.")
    lines.append("- The Tier 2 numbers here use 2y landmark; the actual "
                 "production blend uses 3y landmark. The 3y horizon was "
                 "chosen in Phase 2d after a sweep over {1,2,3,5}y showed "
                 "3y had the best m-s (0.880) on filter A.")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}", flush=True)
    print(f"\nTotal elapsed: {(time.time()-t_overall)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
