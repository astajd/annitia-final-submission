"""Task 1 — OOF-stacked blend weights for the 2-way (landmark + permissive) blend.

Generates per-fold OOF predictions (5-fold × 10-repeat, identical fold IDs to
the rest of Phase 2) for:
  - landmark_3y_RSF: trained on (train_fold ∩ filter-A), predicts on full val fold
                     (matches the test-time protocol: RSF trained on filter-A,
                     predicts on all test rows including non-filter-A ones).
  - permissive_ensemble_avg: rank-avg of 3 longitudinal members on full cohort.

Two evaluations:

(1) Naive grid-search: for each weight w in {0.00, 0.05, ..., 1.00},
    compute per-fold C-index of `w*rank(lm) + (1-w)*rank(perm)` on the val
    fold's hepatic event/time. Aggregate over 50 folds → mean, std, m-s.
    Pick w* = argmax m-s.

(2) Honest leave-one-fold-out (LOFO): for each fold, find best w on the
    pooled OOF predictions of the OTHER 49 folds (concatenated, scored by
    their hepatic events/times), then evaluate that fold's blend C-index
    under the per-fold-selected w. Aggregate.

Decision rule (pre-registered):
  - If w* ∈ [0.4, 0.6]                       → stay with 50/50.
  - If w* outside that AND m-s improves ≥0.005 → build phase2_blend_2way_optimal.csv.

Outputs:
  reports/phase2_stack_2way.json  — full diagnostic
  reports/phase2_stack_2way.md    — short summary
  submissions/phase2_blend_2way_optimal.{csv,json}  (only if criteria met)
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sksurv.metrics import concordance_index_censored

from src.config import REPORTS, SUBMISSIONS, ROOT
from src.cv import repeated_stratified_folds, cindex
from src.data import load_raw, build_targets
from src.features import (build_features, build_landmark_features,
                          at_risk_at_landmark)
from src.models import (make_rsf, make_xgb_cox, make_coxnet,
                        make_lgbm_binary, make_catboost_binary)

CONFIGS = ROOT / "configs"
N_SPLITS = 5
N_REPEATS = 10
BASE_SEED = 42
LANDMARK = 3.0

PERMISSIVE_MEMBERS = [
    "rsf_longitudinal_summary",
    "xgb_cox_longitudinal_plus_meta",
    "xgb_cox_longitudinal_summary",
]

OUT_JSON = REPORTS / "phase2_stack_2way.json"
OUT_MD = REPORTS / "phase2_stack_2way.md"
OPT_CSV = SUBMISSIONS / "phase2_blend_2way_optimal.csv"
OPT_META = SUBMISSIONS / "phase2_blend_2way_optimal.json"

WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)
PARITY_BAND = (0.4, 0.6)
IMPROVEMENT_THRESHOLD = 0.005


def make_factory_from_id(cid: str):
    rec = json.loads((CONFIGS / f"optuna_{cid}.json").read_text())
    p = dict(rec["best_params"])
    m = rec["model"]
    if m == "rsf":          return make_rsf(**p), rec
    if m == "xgb_cox":      return make_xgb_cox(**p), rec
    if m == "coxnet":       return make_coxnet(**p), rec
    if m == "lgbm_bin":
        h = p.pop("horizon"); return make_lgbm_binary(horizon=float(h), **p), rec
    if m == "catboost_bin":
        h = p.pop("horizon"); return make_catboost_binary(horizon=float(h), **p), rec
    raise ValueError(m)


def hepatic_targets(train):
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)
    return df, e, t


def generate_per_fold_oof(df_full, e_full, t_full, keep_A):
    """Returns lists of per-fold dicts:
       fold_records[k] = {repeat, fold, val_idx, risk_lm, risk_perm}
    """
    rsf_params = json.loads(
        (CONFIGS / "optuna_rsf_baseline_v1.json").read_text())["best_params"]

    # Pre-build features once (stateless, row-wise) — feature builds are the
    # dominant cost when called per-fold; built once outside the loop here.
    print("  pre-building landmark features (full cohort)...", flush=True)
    t_pre = time.time()
    X_lm_full = build_landmark_features(df_full, LANDMARK)
    print(f"    {X_lm_full.shape[1]} landmark features ({time.time()-t_pre:.1f}s)",
          flush=True)

    print("  pre-building permissive-member features...", flush=True)
    perm_factories = []
    for cid in PERMISSIVE_MEMBERS:
        t_pre = time.time()
        factory, rec = make_factory_from_id(cid)
        X = build_features(df_full, rec["feature_set"])
        perm_factories.append((cid, factory, X))
        print(f"    {cid:35s} fs={rec['feature_set']:30s} "
              f"{X.shape[1]} feat ({time.time()-t_pre:.1f}s)", flush=True)

    fold_records = []
    fold_idx = 0
    n_total = N_SPLITS * N_REPEATS
    for r, f, tr_idx, va_idx in repeated_stratified_folds(
            e_full, N_SPLITS, N_REPEATS, BASE_SEED):
        t0 = time.time()

        # Landmark: train on filter-A subset of train fold, predict on full val
        tr_keep_A = tr_idx[keep_A[tr_idx]]
        e_tr_lm = e_full[tr_keep_A]
        t_tr_lm = t_full[tr_keep_A]
        risk_lm = make_rsf(**rsf_params)(
            X_lm_full.iloc[tr_keep_A], e_tr_lm, t_tr_lm,
            X_lm_full.iloc[va_idx])

        # Permissive ensemble: 3 members rank-avg, full cohort
        rank_sum = np.zeros(len(va_idx))
        for cid, factory, X in perm_factories:
            risk = factory(X.iloc[tr_idx], e_full[tr_idx], t_full[tr_idx],
                           X.iloc[va_idx])
            rank_sum += rankdata(risk)
        risk_perm = rank_sum / len(perm_factories)

        fold_records.append({
            "repeat": r, "fold": f, "val_idx": np.asarray(va_idx),
            "risk_lm": np.asarray(risk_lm),
            "risk_perm": np.asarray(risk_perm),
        })

        fold_idx += 1
        elapsed = time.time() - t0
        print(f"  fold {fold_idx:2d}/{n_total}  rep={r} fold={f}  "
              f"n_va={len(va_idx)}  e_va={int(e_full[va_idx].sum())}  "
              f"({elapsed:.1f}s)", flush=True)

    return fold_records


def fold_blend_cindex(rec, w, e_full, t_full):
    va = rec["val_idx"]
    e_va, t_va = e_full[va], t_full[va]
    blend = w * rankdata(rec["risk_lm"]) + (1 - w) * rankdata(rec["risk_perm"])
    return cindex(e_va, t_va, blend)


def per_fold_cindex_curve(fold_records, weights, e_full, t_full):
    """Returns DataFrame with one row per (repeat, fold, weight)."""
    rows = []
    for rec in fold_records:
        for w in weights:
            ci = fold_blend_cindex(rec, w, e_full, t_full)
            rows.append({"repeat": rec["repeat"], "fold": rec["fold"],
                         "weight": float(w), "cindex": ci})
    return pd.DataFrame(rows)


def aggregate_curve(curve_df):
    g = curve_df.groupby("weight")["cindex"]
    out = pd.DataFrame({
        "weight": list(g.groups.keys()),
        "mean": [g.get_group(w).mean() for w in g.groups],
        "std":  [g.get_group(w).std(ddof=1) for w in g.groups],
        "n":    [len(g.get_group(w)) for w in g.groups],
    }).sort_values("weight").reset_index(drop=True)
    out["mean_minus_std"] = out["mean"] - out["std"]
    return out


def lofo_honest(fold_records, weights, e_full, t_full):
    """Honest CV: for each fold, pick best w on pooled OTHER folds, score this fold.

    The "score" on the pooled other folds: mean of per-fold C-indices across
    those 49 folds at weight w. (Pooling raw predictions across folds is wrong
    because they live on disjoint patient sets; per-fold C is the consistent
    aggregator.)
    """
    n = len(fold_records)
    rows = []
    for held_out in range(n):
        # For each candidate weight, compute mean C-index over the 49 other folds
        means = []
        for w in weights:
            cis = [fold_blend_cindex(fold_records[k], w, e_full, t_full)
                   for k in range(n) if k != held_out]
            means.append(np.mean(cis))
        means = np.asarray(means)
        w_pick = float(weights[int(np.argmax(means))])
        c_held = fold_blend_cindex(fold_records[held_out], w_pick,
                                   e_full, t_full)
        rows.append({
            "repeat": fold_records[held_out]["repeat"],
            "fold": fold_records[held_out]["fold"],
            "w_pick": w_pick,
            "cindex": c_held,
        })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    train, test = load_raw()
    df_full, e_full, t_full = hepatic_targets(train)
    keep_A = at_risk_at_landmark(e_full, t_full, LANDMARK)
    n_full = len(e_full)
    print(f"Full hepatic-valid cohort: n={n_full}, events={int(e_full.sum())}",
          flush=True)
    print(f"Filter A (event-free at {LANDMARK}y): "
          f"n={int(keep_A.sum())}, events={int(e_full[keep_A].sum())}",
          flush=True)

    print(f"\n--- Generating per-fold OOF predictions "
          f"({N_SPLITS}×{N_REPEATS}={N_SPLITS*N_REPEATS} folds) ---", flush=True)
    fold_records = generate_per_fold_oof(df_full, e_full, t_full, keep_A)
    print(f"OOF gen elapsed: {(time.time()-t0)/60:.1f} min", flush=True)

    # Per-fold C-index per weight
    curve_df = per_fold_cindex_curve(fold_records, WEIGHT_GRID, e_full, t_full)
    agg = aggregate_curve(curve_df)
    print("\n--- C-index vs weight (w on landmark, 1-w on permissive) ---",
          flush=True)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
          flush=True)

    # Naive optimum
    i_opt = int(agg["mean_minus_std"].idxmax())
    w_star = float(agg.loc[i_opt, "weight"])
    ms_star = float(agg.loc[i_opt, "mean_minus_std"])
    mean_star = float(agg.loc[i_opt, "mean"])
    std_star = float(agg.loc[i_opt, "std"])

    # 50/50 baseline reference
    i_50 = int(np.argmin(np.abs(agg["weight"].values - 0.5)))
    ms_50 = float(agg.loc[i_50, "mean_minus_std"])
    mean_50 = float(agg.loc[i_50, "mean"])
    std_50 = float(agg.loc[i_50, "std"])

    print(f"\nNaive optimum: w*={w_star:.2f}  m-s={ms_star:.4f} "
          f"(mean={mean_star:.4f}, std={std_star:.4f})", flush=True)
    print(f"50/50 default: w =0.50  m-s={ms_50:.4f} "
          f"(mean={mean_50:.4f}, std={std_50:.4f})", flush=True)
    print(f"Δ m-s = {ms_star - ms_50:+.4f}  "
          f"(threshold for action: ≥{IMPROVEMENT_THRESHOLD})", flush=True)

    # Honest LOFO
    print("\n--- Honest LOFO weight selection ---", flush=True)
    lofo_df = lofo_honest(fold_records, WEIGHT_GRID, e_full, t_full)
    lofo_mean = float(lofo_df["cindex"].mean())
    lofo_std = float(lofo_df["cindex"].std(ddof=1))
    lofo_ms = lofo_mean - lofo_std
    pick_counts = lofo_df["w_pick"].value_counts().sort_index().to_dict()
    print(f"LOFO honest: m-s={lofo_ms:.4f} "
          f"(mean={lofo_mean:.4f}, std={lofo_std:.4f})", flush=True)
    print(f"LOFO weight picks (count): {pick_counts}", flush=True)
    median_pick = float(lofo_df["w_pick"].median())
    print(f"LOFO median pick w = {median_pick:.2f}", flush=True)

    # Decision
    in_parity_band = PARITY_BAND[0] <= w_star <= PARITY_BAND[1]
    improves_enough = (ms_star - ms_50) >= IMPROVEMENT_THRESHOLD
    will_build = (not in_parity_band) and improves_enough
    decision_text = (
        f"w*={w_star:.2f} {'IN' if in_parity_band else 'OUT OF'} parity band "
        f"[{PARITY_BAND[0]:.1f}, {PARITY_BAND[1]:.1f}]; "
        f"Δm-s = {ms_star - ms_50:+.4f} "
        f"({'≥' if improves_enough else '<'}{IMPROVEMENT_THRESHOLD}). "
        f"→ {'BUILD phase2_blend_2way_optimal.csv' if will_build else 'STAY 50/50, no submission built'}."
    )
    print(f"\n{decision_text}", flush=True)

    # Persist diagnostics
    diag = {
        "task": "Task 1 — OOF stacking weights for 2-way blend",
        "cv": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS,
               "base_seed": BASE_SEED, "landmark_yrs": LANDMARK},
        "weight_grid": list(map(float, WEIGHT_GRID)),
        "weight_curve": agg.to_dict(orient="records"),
        "naive_optimum": {
            "w_star": w_star, "mean": mean_star, "std": std_star,
            "mean_minus_std": ms_star,
        },
        "default_50_50": {
            "w": 0.5, "mean": mean_50, "std": std_50, "mean_minus_std": ms_50,
        },
        "delta_ms_vs_50_50": ms_star - ms_50,
        "honest_lofo": {
            "mean": lofo_mean, "std": lofo_std, "mean_minus_std": lofo_ms,
            "median_pick": median_pick,
            "pick_counts": {f"{k:.2f}": int(v) for k, v in pick_counts.items()},
        },
        "decision_rule": {
            "parity_band": list(PARITY_BAND),
            "improvement_threshold": IMPROVEMENT_THRESHOLD,
            "w_in_parity_band": in_parity_band,
            "improves_enough": improves_enough,
            "build_optimal_submission": will_build,
            "summary": decision_text,
        },
    }
    OUT_JSON.write_text(json.dumps(diag, indent=2, default=str))
    print(f"Wrote {OUT_JSON}", flush=True)

    # Markdown summary
    lines = [
        "# Task 1 — OOF-stacked blend weights for 2-way blend\n",
        f"Date: 2026-04-28. CV: 5-fold × 10-repeat stratified on hepatic event "
        f"(50 folds, base_seed={BASE_SEED}). "
        f"Landmark RSF: tuned baseline_v1 hyperparameters, trained on "
        f"(train ∩ filter-A) and predicting full val. "
        f"Permissive ensemble: rank-avg of {len(PERMISSIVE_MEMBERS)} members on full cohort.\n",
        "## Weight curve\n",
        "| w (landmark) | mean C | std C | m−s | n_folds |",
        "|---|---|---|---|---|",
    ]
    for _, row in agg.iterrows():
        lines.append(
            f"| {row['weight']:.2f} | {row['mean']:.4f} | {row['std']:.4f} | "
            f"{row['mean_minus_std']:.4f} | {int(row['n'])} |")
    lines.append("\n## Headline\n")
    lines.append(f"- **Naive w\\*** = {w_star:.2f}, m−s = {ms_star:.4f} "
                 f"(mean {mean_star:.4f} ± {std_star:.4f}).")
    lines.append(f"- **50/50 default**: m−s = {ms_50:.4f} "
                 f"(mean {mean_50:.4f} ± {std_50:.4f}).")
    lines.append(f"- **Δ m−s vs 50/50** = {ms_star - ms_50:+.4f}.")
    lines.append(f"- **Honest LOFO** m−s = {lofo_ms:.4f} "
                 f"(mean {lofo_mean:.4f} ± {lofo_std:.4f}); "
                 f"median per-fold pick w = {median_pick:.2f}.")
    lines.append(f"- LOFO weight pick distribution: "
                 f"{', '.join(f'{k:.2f}→{v}' for k, v in sorted(pick_counts.items()))}.")
    lines.append(f"\n## Decision\n\n{decision_text}\n")
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}", flush=True)

    # Optionally build submission
    if will_build:
        print("\n--- Building optimal-weighted submission ---", flush=True)
        # Reuse blend builder
        from experiments.phase2_q1_q2_investigate import (  # type: ignore
            predict_landmark_3y, predict_permissive_ensemble, predict_death,
        )
        risk_lm, lm_info = predict_landmark_3y(df_full, e_full, t_full, test)
        risk_perm, perm_info = predict_permissive_ensemble(df_full, e_full, t_full, test)
        blend = (w_star * rankdata(risk_lm)
                 + (1 - w_star) * rankdata(risk_perm))
        risk_death, d_info = predict_death(train, test)
        sub = pd.DataFrame({
            "trustii_id": test["trustii_id"].values,
            "risk_hepatic_event": blend,
            "risk_death": risk_death,
        })
        sub.to_csv(OPT_CSV, index=False)
        meta = {
            "submission_kind": "2-way blend with OOF-stacked weights",
            "weight_landmark": w_star,
            "weight_permissive": 1 - w_star,
            "cv_diagnostic": diag["naive_optimum"],
            "lofo_diagnostic": diag["honest_lofo"],
            "hepatic": {
                "ensemble_strategy":
                    f"{w_star:.2f}*rank(landmark) + {1-w_star:.2f}*rank(permissive)",
                "components": {"a_landmark_3y": lm_info,
                               "b_permissive": perm_info},
            },
            "death": d_info,
            "reference_LBs": {
                "phase2_blend_landmark_permissive_50_50": 0.88033,
                "phase2_landmark_3y": 0.8109,
            },
        }
        OPT_META.write_text(json.dumps(meta, indent=2, default=str))
        print(f"Wrote {OPT_CSV}\nWrote {OPT_META}", flush=True)
    else:
        print("\nNo submission built (decision rule did not trigger).",
              flush=True)

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
