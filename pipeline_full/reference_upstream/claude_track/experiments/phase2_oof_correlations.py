"""OOF rank-correlation matrix across 8 hepatic candidates.

For each candidate, generate 5-fold × 10-repeat OOF predictions on the training
data, average across repeats, then compute pairwise Spearman correlations on
the filter-A intersection (1238 rows = patients event-free at 3y, the cohort
landmark_3y_RSF was trained on).

For permissive_ensemble_avg: rank-average within each CV fold across the three
member models (matches test-time predict_ensemble logic), then OOF-stitch.

Looking for: any model with rank-corr < 0.6 with BOTH:
  - landmark_3y_RSF
  - permissive_ensemble_avg
That's the candidate for a third diverse blend member.

If no such model exists, diversity is exhausted within Phase 2 candidates.

Outputs:
  reports/phase2_oof_correlations.csv  (8x8 Spearman matrix)
  reports/phase2_oof_correlations.md   (interpretation, candidate id if any)
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.config import REPORTS, ROOT
from src.cv import repeated_stratified_folds
from src.data import load_raw, build_targets
from src.features import build_features, build_landmark_features, at_risk_at_landmark
from src.models import (make_rsf, make_xgb_cox, make_coxnet,
                        make_catboost_binary, make_lgbm_binary)

CONFIGS = ROOT / "configs"
N_SPLITS = 5
N_REPEATS = 10
BASE_SEED = 42
LANDMARK = 3.0
LOW_CORR_THRESHOLD = 0.6

OUT_CSV = REPORTS / "phase2_oof_correlations.csv"
OUT_MD = REPORTS / "phase2_oof_correlations.md"


# Permissive ensemble members (match phase2_blend_landmark_permissive.py)
PERMISSIVE_MEMBERS = [
    "rsf_longitudinal_summary",
    "xgb_cox_longitudinal_plus_meta",
    "xgb_cox_longitudinal_summary",
]


def make_factory_from_id(cid: str):
    rec = json.loads((CONFIGS / f"optuna_{cid}.json").read_text())
    p = dict(rec["best_params"])
    m = rec["model"]
    if m == "rsf":          return make_rsf(**p), rec["feature_set"]
    if m == "xgb_cox":      return make_xgb_cox(**p), rec["feature_set"]
    if m == "coxnet":       return make_coxnet(**p), rec["feature_set"]
    if m == "lgbm_bin":
        h = p.pop("horizon"); return make_lgbm_binary(horizon=float(h), **p), rec["feature_set"]
    if m == "catboost_bin":
        h = p.pop("horizon"); return make_catboost_binary(horizon=float(h), **p), rec["feature_set"]
    raise ValueError(m)


def oof_single_model(factory, X, e, t, mask=None):
    """OOF predictions averaged over repeats. mask: optional bool array
    selecting cohort rows (cohort indexes are local to factory)."""
    oof_sum = np.zeros(len(e))
    oof_count = np.zeros(len(e), dtype=int)
    for r, f, tr_idx, va_idx in repeated_stratified_folds(
            e, N_SPLITS, N_REPEATS, BASE_SEED):
        risk = factory(X.iloc[tr_idx], e[tr_idx], t[tr_idx], X.iloc[va_idx])
        oof_sum[va_idx] += np.asarray(risk)
        oof_count[va_idx] += 1
    out = np.full(len(e), np.nan)
    nz = oof_count > 0
    out[nz] = oof_sum[nz] / oof_count[nz]
    return out


def oof_permissive_ensemble(member_ids, df, e, t):
    """OOF predictions for the rank-averaged permissive ensemble."""
    members = []
    for mid in member_ids:
        factory, fs = make_factory_from_id(mid)
        members.append((mid, factory, build_features(df, fs)))

    oof_sum = np.zeros(len(e))
    oof_count = np.zeros(len(e), dtype=int)
    for r, f, tr_idx, va_idx in repeated_stratified_folds(
            e, N_SPLITS, N_REPEATS, BASE_SEED):
        rank_sum = np.zeros(len(va_idx))
        for mid, factory, X in members:
            risk = factory(X.iloc[tr_idx], e[tr_idx], t[tr_idx], X.iloc[va_idx])
            rank_sum += rankdata(risk)
        avg_rank = rank_sum / len(members)
        oof_sum[va_idx] += avg_rank
        oof_count[va_idx] += 1
    out = np.full(len(e), np.nan)
    nz = oof_count > 0
    out[nz] = oof_sum[nz] / oof_count[nz]
    return out


def main():
    t_overall = time.time()
    train, _ = load_raw()

    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e_full = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t_full = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df_full = train.loc[valid].reset_index(drop=True)
    n_full = len(e_full)
    print(f"Full cohort: n={n_full}, events={int(e_full.sum())}", flush=True)

    keep_A = at_risk_at_landmark(e_full, t_full, LANDMARK)
    filter_A_idx = np.where(keep_A)[0]
    print(f"Filter-A intersection (event-free at {LANDMARK}y): "
          f"n={int(keep_A.sum())} (correlation will use these rows)",
          flush=True)

    oof = {}

    # ---- Single models on full cohort (1253 rows) ----
    print("\n--- OOF on full cohort (1253 rows) ---", flush=True)
    full_models = [
        ("tuned_rsf_baseline_v1",        "rsf_baseline_v1"),
        ("tuned_xgb_cox_baseline_v1",    "xgb_cox_baseline_v1"),
        ("rsf_nit_only_baseline_only",   "rsf_nit_only_baseline_only"),
        ("catboost_bin_early_v1_v3",     "catboost_bin_early_v1_v3"),
        ("rsf_longitudinal_summary",     "rsf_longitudinal_summary"),
        ("xgb_cox_longitudinal_plus_meta","xgb_cox_longitudinal_plus_meta"),
    ]
    for label, cid in full_models:
        factory, fs = make_factory_from_id(cid)
        X = build_features(df_full, fs)
        t0 = time.time()
        oof[label] = oof_single_model(factory, X, e_full, t_full)
        print(f"  {label:32s} done ({time.time()-t0:.0f}s, "
              f"{X.shape[1]} features)", flush=True)

    # ---- Permissive ensemble (rank-avg of 3 members) on full cohort ----
    print("\n--- OOF for permissive_ensemble_avg ---", flush=True)
    t0 = time.time()
    oof["permissive_ensemble_avg"] = oof_permissive_ensemble(
        PERMISSIVE_MEMBERS, df_full, e_full, t_full)
    print(f"  permissive_ensemble_avg          done "
          f"({time.time()-t0:.0f}s, members={PERMISSIVE_MEMBERS})", flush=True)

    # ---- Landmark 3y RSF on filter-A subcohort ----
    print(f"\n--- OOF for landmark_{int(LANDMARK)}y_RSF (filter-A subcohort) ---",
          flush=True)
    df_lm = df_full.loc[keep_A].reset_index(drop=True)
    e_lm, t_lm = e_full[keep_A], t_full[keep_A]
    rsf_params = json.loads(
        (CONFIGS / "optuna_rsf_baseline_v1.json").read_text())["best_params"]
    X_lm = build_landmark_features(df_lm, LANDMARK)
    factory_lm = make_rsf(**rsf_params)
    t0 = time.time()
    oof_lm_local = oof_single_model(factory_lm, X_lm, e_lm, t_lm)
    print(f"  landmark_{int(LANDMARK)}y_RSF                done "
          f"({time.time()-t0:.0f}s, n={len(e_lm)}, events={int(e_lm.sum())})",
          flush=True)
    # Lift to full-cohort indexing with NaN outside filter A
    oof_lm_full = np.full(n_full, np.nan)
    oof_lm_full[filter_A_idx] = oof_lm_local
    oof[f"landmark_{int(LANDMARK)}y_RSF"] = oof_lm_full

    # ---- Build correlation matrix on filter-A intersection ----
    print(f"\n--- Spearman correlation matrix (filter-A intersection, "
          f"n={int(keep_A.sum())}) ---", flush=True)
    cand_order = [
        f"landmark_{int(LANDMARK)}y_RSF",
        "permissive_ensemble_avg",
        "tuned_rsf_baseline_v1",
        "tuned_xgb_cox_baseline_v1",
        "rsf_nit_only_baseline_only",
        "catboost_bin_early_v1_v3",
        "rsf_longitudinal_summary",
        "xgb_cox_longitudinal_plus_meta",
    ]

    M = pd.DataFrame(index=cand_order, columns=cand_order, dtype=float)
    for a in cand_order:
        va = oof[a][filter_A_idx]
        for b in cand_order:
            vb = oof[b][filter_A_idx]
            mask = ~(np.isnan(va) | np.isnan(vb))
            if mask.sum() < 50:
                M.loc[a, b] = np.nan
            else:
                rho, _ = spearmanr(va[mask], vb[mask])
                M.loc[a, b] = rho

    M.to_csv(OUT_CSV)
    print(f"\nWrote {OUT_CSV}", flush=True)

    # Pretty-print
    print("\n" + "=" * 95, flush=True)
    print(f"{'':32s} " + " ".join(f"{c[:10]:>10s}" for c in cand_order),
          flush=True)
    for a in cand_order:
        row = " ".join(f"{M.loc[a, b]:10.3f}" for b in cand_order)
        print(f"{a[:32]:32s} {row}", flush=True)
    print("=" * 95, flush=True)

    # ---- Identify low-corr candidate vs BOTH anchors ----
    anchor_lm = f"landmark_{int(LANDMARK)}y_RSF"
    anchor_perm = "permissive_ensemble_avg"
    print(f"\nLooking for any candidate with rank-corr < {LOW_CORR_THRESHOLD} "
          f"vs BOTH {anchor_lm} AND {anchor_perm}:", flush=True)
    candidates_to_consider = [c for c in cand_order
                              if c not in (anchor_lm, anchor_perm)]
    diverse_picks = []
    for c in candidates_to_consider:
        rho_lm = M.loc[c, anchor_lm]
        rho_perm = M.loc[c, anchor_perm]
        flag = "✓ DIVERSE" if (rho_lm < LOW_CORR_THRESHOLD
                              and rho_perm < LOW_CORR_THRESHOLD) else ""
        print(f"  {c:35s}  vs landmark={rho_lm:.3f}  "
              f"vs permissive={rho_perm:.3f}  {flag}", flush=True)
        if rho_lm < LOW_CORR_THRESHOLD and rho_perm < LOW_CORR_THRESHOLD:
            diverse_picks.append({
                "id": c, "corr_vs_landmark": float(rho_lm),
                "corr_vs_permissive": float(rho_perm),
                "max_corr": float(max(rho_lm, rho_perm)),
            })

    # ---- Write markdown summary ----
    lines = ["# OOF rank-correlation matrix — Phase 2 candidates\n"]
    lines.append(f"Date: 2026-04-27. CV: 5-fold × 10-repeat (50 folds), "
                 f"OOF averaged across repeats. Correlation computed on "
                 f"filter-A intersection (n={int(keep_A.sum())}, "
                 f"events={int(e_full[keep_A].sum())}).\n")
    lines.append(f"Threshold: rank-corr < {LOW_CORR_THRESHOLD} with BOTH "
                 f"`{anchor_lm}` AND `{anchor_perm}` → diverse third "
                 "blend candidate.\n")
    lines.append("\n## 8×8 Spearman correlation matrix\n")
    lines.append("| | " + " | ".join(f"`{c}`" for c in cand_order) + " |")
    lines.append("|" + "|".join(["---"] * (len(cand_order) + 1)) + "|")
    for a in cand_order:
        row = "| `" + a + "` | " + " | ".join(
            f"{M.loc[a, b]:.3f}" if not pd.isna(M.loc[a, b]) else "n/a"
            for b in cand_order) + " |"
        lines.append(row)

    lines.append(f"\n## Diversity vs both anchors\n")
    lines.append(f"| candidate | corr vs `{anchor_lm}` | corr vs `{anchor_perm}` | diverse? |")
    lines.append("|---|---|---|---|")
    for c in candidates_to_consider:
        rho_lm = M.loc[c, anchor_lm]; rho_perm = M.loc[c, anchor_perm]
        diverse = (rho_lm < LOW_CORR_THRESHOLD
                   and rho_perm < LOW_CORR_THRESHOLD)
        lines.append(f"| `{c}` | {rho_lm:.3f} | {rho_perm:.3f} | "
                     f"{'**yes**' if diverse else 'no'} |")

    lines.append(f"\n## Result\n")
    if diverse_picks:
        diverse_picks.sort(key=lambda r: r["max_corr"])
        best = diverse_picks[0]
        lines.append(f"**{len(diverse_picks)} candidate(s) clear the threshold.** "
                     f"Top pick by lowest max-corr: **`{best['id']}`** "
                     f"(landmark {best['corr_vs_landmark']:.3f}, "
                     f"permissive {best['corr_vs_permissive']:.3f}).")
        for p in diverse_picks[1:]:
            lines.append(f"- Also: `{p['id']}` "
                         f"(landmark {p['corr_vs_landmark']:.3f}, "
                         f"permissive {p['corr_vs_permissive']:.3f})")
    else:
        lines.append("**No candidate clears the threshold.** Diversity is "
                     "exhausted within Phase 2 candidates. Per the user's "
                     "decision rule, the next ensemble lift would have to "
                     "come from Phase 3 ideas: cross-feature stacking, "
                     "discrete-time hazard reformulation, trajectory-shape "
                     "features, or external data.")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_MD}", flush=True)
    print(f"\nTotal: {(time.time()-t_overall)/60:.1f} min", flush=True)

    # Persist for the optional 3-way blend script
    pick_path = REPORTS / "phase2_oof_diverse_pick.json"
    if diverse_picks:
        pick_path.write_text(json.dumps(diverse_picks[0], indent=2))
        print(f"Wrote {pick_path}", flush=True)
    else:
        if pick_path.exists():
            pick_path.unlink()


if __name__ == "__main__":
    main()
