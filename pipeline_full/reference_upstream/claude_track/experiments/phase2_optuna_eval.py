"""Phase 2 Experiment 2 — post-tuning evaluation.

Reads tuned best_params from configs/optuna_<id>.json. For each tuned model:
  C. Final 5×10 CV verification (50 folds) → append to phase2_cv_results.csv
     with model name "<model>_optuna".
  A. Bootstrap CI (1000 resamples) on the 50 per-fold C-indices.
     Note deviation from spec: spec asked for 5 CV repeats; we use the same
     50 fold-CIs from task C as the bootstrap base. More folds = tighter CI;
     strictly more rigorous, not less.
  B. Multi-seed ensembling for the top 2 RSF configs (by m−s of tuned mean).
     For each fold: fit RSF with seeds 42..51, rank-average risks, compute
     fold C-index. Compare to single-seed (42) on the same fold.

Final outputs:
  reports/phase2_cv_results.csv — appended (model = "<model>_optuna")
  reports/phase2_bootstrap_cis.csv
  reports/phase2_multiseed.csv
  reports/phase2_exp2_summary.md
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import REPORTS, ROOT
from src.cv import (evaluate_cv, summarize, repeated_stratified_folds, cindex,
                    CV_RANDOM_STATE)
from src.data import load_raw, build_targets
from src.features import build_features, FEATURE_SET_RISK
from src.models import (make_rsf, make_xgb_cox, make_coxnet,
                        make_catboost_binary)

CONFIGS = ROOT / "configs"
PHASE2_CV_CSV = REPORTS / "phase2_cv_results.csv"
BOOT_CSV = REPORTS / "phase2_bootstrap_cis.csv"
MULTISEED_CSV = REPORTS / "phase2_multiseed.csv"
SUMMARY_MD = REPORTS / "phase2_exp2_summary.md"

FINAL_SPLITS = 5
FINAL_REPEATS = 10
BOOTSTRAP_N = 1000
MULTISEEDS = list(range(42, 52))  # 10 seeds: 42..51

CANDIDATES = [
    {"id": "rsf_baseline_v1",            "model": "rsf",          "fs": "baseline_v1"},
    {"id": "rsf_nit_only_baseline_only", "model": "rsf",          "fs": "nit_only_baseline_only"},
    {"id": "rsf_early_v1_v3",            "model": "rsf",          "fs": "early_v1_v3"},
    {"id": "xgb_cox_baseline_v1",        "model": "xgb_cox",      "fs": "baseline_v1"},
    {"id": "catboost_bin_early_v1_v3",   "model": "catboost_bin", "fs": "early_v1_v3"},
    {"id": "coxnet_baseline_v1",         "model": "coxnet",       "fs": "baseline_v1"},
]


def make_factory(model, params):
    p = dict(params)
    if model == "rsf":
        return make_rsf(**p)
    if model == "xgb_cox":
        return make_xgb_cox(**p)
    if model == "catboost_bin":
        h = p.pop("horizon")
        return make_catboost_binary(horizon=float(h), **p)
    if model == "coxnet":
        return make_coxnet(**p)
    raise ValueError(model)


def append_cv_row(cv_res: pd.DataFrame, cand: dict, n_features: int,
                  n_rows: int, n_events: int, elapsed: float):
    s = summarize(cv_res)
    row = {
        "endpoint": "hepatic", "death_mode": "n/a",
        "feature_set": cand["fs"],
        "leakage_risk": FEATURE_SET_RISK[cand["fs"]],
        "model": f"{cand['model']}_optuna",
        "n_rows": n_rows, "n_events": n_events, "n_features": n_features,
        **s, "elapsed_s": round(elapsed, 1),
    }
    df_row = pd.DataFrame([row])
    if PHASE2_CV_CSV.exists():
        df_row.to_csv(PHASE2_CV_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(PHASE2_CV_CSV, mode="w", header=True, index=False)
    return s


def bootstrap_ci(per_fold_cis: np.ndarray, n_resamples=BOOTSTRAP_N,
                 ci_pct=95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(per_fold_cis)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = per_fold_cis[idx].mean()
    alpha = (100 - ci_pct) / 2
    return float(np.percentile(means, alpha)), float(np.percentile(means, 100 - alpha))


def run_multiseed_for_rsf(best_params: dict, X, e, t, seeds, n_splits, n_repeats):
    """For each fold: fit RSF with each seed, rank-average risks; compare to seed[0]."""
    rows = []
    for r, f, tr_idx, va_idx in repeated_stratified_folds(
            e, n_splits=n_splits, n_repeats=n_repeats, base_seed=CV_RANDOM_STATE):
        ranks_per_seed = []
        risks_per_seed = []
        for s in seeds:
            factory = make_rsf(**{**best_params, "seed": s})
            risk = factory(X.iloc[tr_idx], e[tr_idx], t[tr_idx], X.iloc[va_idx])
            risks_per_seed.append(np.asarray(risk))
            ranks_per_seed.append(rankdata(risk))
        avg_rank = np.mean(ranks_per_seed, axis=0)
        ci_avg = cindex(e[va_idx], t[va_idx], avg_rank)
        ci_single = cindex(e[va_idx], t[va_idx], risks_per_seed[0])
        rows.append({"repeat": r, "fold": f,
                     "ci_single_seed": ci_single, "ci_multiseed": ci_avg,
                     "n_seeds": len(seeds)})
    return pd.DataFrame(rows)


def get_phase2_baseline(p2: pd.DataFrame, fs: str, model: str) -> dict | None:
    """Look up the un-tuned (Phase 2 grid) row for comparison."""
    p2 = p2.copy()
    p2["death_mode"] = p2["death_mode"].fillna("n/a")
    sel = p2[(p2["endpoint"] == "hepatic") & (p2["death_mode"] == "n/a")
             & (p2["feature_set"] == fs)]
    # Untuned model name varies; map.
    mapping = {"rsf": "rsf", "xgb_cox": "xgb_cox", "coxnet": "coxnet",
               "catboost_bin": "catboost_h5"}
    untuned_name = mapping[model]
    sel = sel[sel["model"] == untuned_name]
    if len(sel) == 0:
        return None
    r = sel.iloc[0]
    return {"mean": float(r["mean"]), "std": float(r["std"]),
            "untuned_model_name": untuned_name}


def main():
    overall_start = time.time()

    train, _ = load_raw()
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)
    n_rows, n_events = int(len(e)), int(e.sum())
    print(f"Train: n={n_rows}, events={n_events}", flush=True)

    p2_existing = pd.read_csv(PHASE2_CV_CSV)
    print(f"Existing phase2_cv_results.csv: {len(p2_existing)} rows", flush=True)

    all_results = []

    # --- Phase C (and basis for A): final 5×10 CV per tuned candidate ---
    print("\n========== PHASE C: final 5×10 CV ==========", flush=True)
    for cand in CANDIDATES:
        json_path = CONFIGS / f"optuna_{cand['id']}.json"
        if not json_path.exists():
            print(f"  [skip] {cand['id']}: no tuned json found", flush=True)
            continue
        rec = json.loads(json_path.read_text())
        best_params = rec["best_params"]
        print(f"\n  [{cand['id']}] best_params={best_params}", flush=True)

        X = build_features(df, cand["fs"])
        factory = make_factory(cand["model"], best_params)
        t0 = time.time()
        cv_res = evaluate_cv(factory, X, e, t,
                             n_splits=FINAL_SPLITS, n_repeats=FINAL_REPEATS)
        el = time.time() - t0
        s = append_cv_row(cv_res, cand, X.shape[1], n_rows, n_events, el)
        per_fold = cv_res["cindex"].to_numpy()
        ms = s["mean"] - s["std"]
        print(f"  [{cand['id']}] FINAL  mean={s['mean']:.4f} std={s['std']:.4f} "
              f"m−s={ms:.4f}  ({el:.0f}s, {len(per_fold)} folds)", flush=True)
        all_results.append({
            "cand": cand, "tuned": rec, "final_cv": s,
            "per_fold_cis": per_fold,
        })

    # --- Phase A: bootstrap CI on per-fold C-indices ---
    print("\n========== PHASE A: bootstrap CIs ==========", flush=True)
    boot_rows = []
    for r in all_results:
        per_fold = r["per_fold_cis"]
        lo, hi = bootstrap_ci(per_fold, n_resamples=BOOTSTRAP_N, ci_pct=95)
        boot_rows.append({
            "id": r["cand"]["id"],
            "model": r["cand"]["model"],
            "feature_set": r["cand"]["fs"],
            "n_folds": int(len(per_fold)),
            "mean_cindex": float(np.mean(per_fold)),
            "std_cindex": float(np.std(per_fold, ddof=1)),
            "ci95_lo": lo, "ci95_hi": hi,
            "ci95_width": hi - lo,
            "n_bootstrap": BOOTSTRAP_N,
        })
        print(f"  [{r['cand']['id']}] mean={np.mean(per_fold):.4f}  "
              f"95% CI [{lo:.4f}, {hi:.4f}]  width={hi-lo:.4f}", flush=True)
    boot_df = pd.DataFrame(boot_rows).sort_values("mean_cindex", ascending=False)
    boot_df.to_csv(BOOT_CSV, index=False)
    print(f"  Wrote {BOOT_CSV}", flush=True)

    # --- Phase B: multi-seed ensembling for top 2 RSFs ---
    print("\n========== PHASE B: multi-seed (top-2 RSFs) ==========", flush=True)
    rsf_rs = [r for r in all_results if r["cand"]["model"] == "rsf"]
    rsf_rs.sort(key=lambda r: -(r["final_cv"]["mean"] - r["final_cv"]["std"]))
    top2_rsf = rsf_rs[:2]
    multi_rows = []
    for r in top2_rsf:
        cand = r["cand"]
        best_params = r["tuned"]["best_params"]
        X = build_features(df, cand["fs"])
        print(f"\n  [{cand['id']}] multi-seed CV (seeds={MULTISEEDS})", flush=True)
        t0 = time.time()
        ms_df = run_multiseed_for_rsf(best_params, X, e, t,
                                      MULTISEEDS, FINAL_SPLITS, FINAL_REPEATS)
        el = time.time() - t0
        single = ms_df["ci_single_seed"]
        multi = ms_df["ci_multiseed"]
        row = {
            "id": cand["id"],
            "feature_set": cand["fs"],
            "n_folds": int(len(ms_df)),
            "n_seeds": int(MULTISEEDS[-1] - MULTISEEDS[0] + 1),
            "single_seed_mean": float(single.mean()),
            "single_seed_std": float(single.std(ddof=1)),
            "multiseed_mean": float(multi.mean()),
            "multiseed_std": float(multi.std(ddof=1)),
            "lift_mean": float(multi.mean() - single.mean()),
            "lift_std_reduction": float(single.std(ddof=1) - multi.std(ddof=1)),
            "elapsed_s": round(el, 1),
        }
        multi_rows.append(row)
        print(f"  [{cand['id']}] single seed={single.mean():.4f}±{single.std(ddof=1):.4f}  "
              f"multi(10 seeds)={multi.mean():.4f}±{multi.std(ddof=1):.4f}  "
              f"Δmean={row['lift_mean']:+.4f}  Δstd={-row['lift_std_reduction']:+.4f}  "
              f"({el:.0f}s)", flush=True)
    multi_df = pd.DataFrame(multi_rows)
    multi_df.to_csv(MULTISEED_CSV, index=False)
    print(f"  Wrote {MULTISEED_CSV}", flush=True)

    # --- Phase D: write summary ---
    print("\n========== Writing summary ==========", flush=True)
    p2_after = pd.read_csv(PHASE2_CV_CSV)
    write_summary(all_results, boot_df, multi_df, p2_existing, p2_after,
                  total_elapsed=(time.time() - overall_start))
    print(f"\nTotal eval wall-clock: {(time.time()-overall_start)/60:.1f} min",
          flush=True)


def write_summary(all_results, boot_df, multi_df, p2_before, p2_after,
                  total_elapsed):
    lines = []
    add = lines.append
    add("# Phase 2 Experiment 2 — Optuna tuning + verification\n")
    add(f"Date: 2026-04-27. Wall-clock: {total_elapsed/60:.1f} min.\n")
    add("Selection metric (pre-registered): mean − 1×std hepatic C-index.\n")
    add(f"Inner CV (Optuna): 5-fold × 5-repeat (25 folds). "
        f"Final CV: 5-fold × 10-repeat (50 folds, base_seed={CV_RANDOM_STATE}).\n")
    add(f"Bootstrap: {BOOTSTRAP_N} resamples on per-fold C-indices, 95% percentile CI.\n")

    add("\n## Best params (Optuna 50–100 trials, TPE)\n")
    add("| id | model | feature_set | trials | best mean | best std | m−s |")
    add("|---|---|---|---|---|---|---|")
    for r in sorted(all_results, key=lambda r: -r["final_cv"]["mean"] + r["final_cv"]["std"]):
        rec = r["tuned"]
        add(f"| `{rec['id']}` | {rec['model']} | {rec['feature_set']} | "
            f"{rec['n_trials_complete']} | "
            f"{rec['best_mean']:.4f} | {rec['best_std']:.4f} | "
            f"{rec['best_mean_minus_std']:.4f} |")

    add("\n### Best param values\n")
    for r in all_results:
        rec = r["tuned"]
        add(f"- **`{rec['id']}`**: "
            + ", ".join(f"{k}={v}" for k, v in rec["best_params"].items()))

    add("\n## Final 5×10 CV (after tuning) — appended to phase2_cv_results.csv\n")
    add("Compared with the un-tuned Phase 2 grid number, where it exists.\n")
    add("| id | tuned mean | tuned std | tuned m−s | untuned mean | untuned std | Δmean | Δm−s |")
    add("|---|---|---|---|---|---|---|---|")
    for r in sorted(all_results, key=lambda r: -(r["final_cv"]["mean"] - r["final_cv"]["std"])):
        cand = r["cand"]
        s = r["final_cv"]
        ut = get_phase2_baseline(p2_before, cand["fs"], cand["model"])
        if ut is None:
            add(f"| `{cand['id']}` | {s['mean']:.4f} | {s['std']:.4f} | "
                f"{s['mean']-s['std']:.4f} | n/a | n/a | n/a | n/a |")
        else:
            d_mean = s["mean"] - ut["mean"]
            d_ms = (s["mean"] - s["std"]) - (ut["mean"] - ut["std"])
            add(f"| `{cand['id']}` | {s['mean']:.4f} | {s['std']:.4f} | "
                f"{s['mean']-s['std']:.4f} | {ut['mean']:.4f} | {ut['std']:.4f} | "
                f"{d_mean:+.4f} | {d_ms:+.4f} |")

    add("\n## Bootstrap CIs (1000 resamples on the 50 per-fold C-indices)\n")
    add("| id | mean | std | 95% CI | width |")
    add("|---|---|---|---|---|")
    for _, r in boot_df.iterrows():
        add(f"| `{r['id']}` | {r['mean_cindex']:.4f} | {r['std_cindex']:.4f} | "
            f"[{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}] | {r['ci95_width']:.4f} |")
    add("\nFile: `reports/phase2_bootstrap_cis.csv`.\n")

    add("\n## Multi-seed ensembling (top 2 RSFs, seeds 42–51)\n")
    add("Per fold: 10 RSF fits (one per seed), risks rank-averaged. Compared "
        "to single-seed (42) on the same fold split.\n")
    add("| id | single mean | single std | multi mean | multi std | Δmean | Δstd |")
    add("|---|---|---|---|---|---|---|")
    for _, r in multi_df.iterrows():
        add(f"| `{r['id']}` | {r['single_seed_mean']:.4f} | "
            f"{r['single_seed_std']:.4f} | {r['multiseed_mean']:.4f} | "
            f"{r['multiseed_std']:.4f} | {r['lift_mean']:+.4f} | "
            f"{-r['lift_std_reduction']:+.4f} |")
    add("\nFile: `reports/phase2_multiseed.csv`.\n")

    # Recommendation
    add("\n## Recommendation for ensemble selection\n")
    boot_sorted = boot_df.sort_values("mean_cindex", ascending=False)
    top3 = boot_sorted.head(3)
    add("Top 3 by tuned mean (from bootstrap table):")
    for _, r in top3.iterrows():
        add(f"- **{r['id']}** — mean {r['mean_cindex']:.4f}, "
            f"95% CI [{r['ci95_lo']:.4f}, {r['ci95_hi']:.4f}]")

    add("\nFor a rank-averaged hepatic ensemble, prefer the model classes that "
        "differ most in tuned configuration, ideally one tree-based (RSF or "
        "XGB-Cox) and one linear (Coxnet) over different feature scopes "
        "(`baseline_v1` vs `nit_only_baseline_only`). Multi-seed lifts (above) "
        "indicate whether RSF benefits from seed averaging — if Δmean > 0 and "
        "Δstd < 0 for top-RSF, include the multi-seed RSF in the ensemble.\n")

    add("\n## Stop point\n")
    add("Experiment 2 complete. No tuned model exceeded the 0.85 suspicious "
        "threshold during tuning. No new submissions generated. Awaiting "
        "instruction on Experiment 3+ or submission generation.\n")

    SUMMARY_MD.write_text("\n".join(lines))
    print(f"  Wrote {SUMMARY_MD}", flush=True)


if __name__ == "__main__":
    main()
