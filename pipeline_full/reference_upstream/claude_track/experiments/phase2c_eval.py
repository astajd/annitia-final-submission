"""Phase 2c — final 5×10 CV verification + bootstrap CIs for permissive track."""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import REPORTS, ROOT
from src.cv import evaluate_cv, summarize
from src.data import load_raw, build_targets
from src.features import build_features, FEATURE_SET_RISK
from src.models import make_rsf, make_xgb_cox, make_lgbm_binary

CONFIGS = ROOT / "configs"
PHASE2_CV_CSV = REPORTS / "phase2_cv_results.csv"
BOOT_CSV = REPORTS / "phase2_bootstrap_cis.csv"

FINAL_SPLITS = 5
FINAL_REPEATS = 10
BOOTSTRAP_N = 1000

CANDIDATES = [
    {"id": "rsf_longitudinal_summary",        "model": "rsf",      "fs": "longitudinal_summary"},
    {"id": "xgb_cox_longitudinal_summary",    "model": "xgb_cox",  "fs": "longitudinal_summary"},
    {"id": "xgb_cox_longitudinal_plus_meta",  "model": "xgb_cox",  "fs": "longitudinal_plus_meta"},
    {"id": "lgbm_bin_longitudinal_plus_meta", "model": "lgbm_bin", "fs": "longitudinal_plus_meta"},
]


def make_factory(model, params):
    p = dict(params)
    if model == "rsf":      return make_rsf(**p)
    if model == "xgb_cox":  return make_xgb_cox(**p)
    if model == "lgbm_bin":
        h = p.pop("horizon")
        return make_lgbm_binary(horizon=float(h), **p)
    raise ValueError(model)


def append_cv_row(cv_res: pd.DataFrame, cand: dict, n_features, n_rows,
                  n_events, elapsed):
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


def bootstrap_ci(per_fold_cis, n_resamples=BOOTSTRAP_N, ci_pct=95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(per_fold_cis)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = per_fold_cis[idx].mean()
    alpha = (100 - ci_pct) / 2
    return float(np.percentile(means, alpha)), float(np.percentile(means, 100 - alpha))


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

    new_boot_rows = []
    for cand in CANDIDATES:
        json_path = CONFIGS / f"optuna_{cand['id']}.json"
        if not json_path.exists():
            print(f"  [skip] {cand['id']}: no tuned json", flush=True)
            continue
        rec = json.loads(json_path.read_text())
        bp = rec["best_params"]
        print(f"\n  [{cand['id']}] best_params={bp}", flush=True)

        X = build_features(df, cand["fs"])
        factory = make_factory(cand["model"], bp)
        t0 = time.time()
        cv_res = evaluate_cv(factory, X, e, t,
                             n_splits=FINAL_SPLITS, n_repeats=FINAL_REPEATS)
        el = time.time() - t0
        s = append_cv_row(cv_res, cand, X.shape[1], n_rows, n_events, el)
        per_fold = cv_res["cindex"].to_numpy()
        ms = s["mean"] - s["std"]
        print(f"  [{cand['id']}] FINAL  mean={s['mean']:.4f} std={s['std']:.4f} "
              f"m−s={ms:.4f}  ({el:.0f}s)", flush=True)

        lo, hi = bootstrap_ci(per_fold)
        new_boot_rows.append({
            "id": cand["id"], "model": cand["model"],
            "feature_set": cand["fs"], "n_folds": int(len(per_fold)),
            "mean_cindex": float(np.mean(per_fold)),
            "std_cindex": float(np.std(per_fold, ddof=1)),
            "ci95_lo": lo, "ci95_hi": hi, "ci95_width": hi - lo,
            "n_bootstrap": BOOTSTRAP_N,
        })
        print(f"  [{cand['id']}] 95% CI [{lo:.4f}, {hi:.4f}]", flush=True)

    # Append to existing bootstrap CSV
    if new_boot_rows:
        new_df = pd.DataFrame(new_boot_rows)
        if BOOT_CSV.exists():
            existing = pd.read_csv(BOOT_CSV)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(BOOT_CSV, index=False)
        print(f"\nAppended {len(new_boot_rows)} rows to {BOOT_CSV}", flush=True)

    print(f"Phase 2c eval wall-clock: {(time.time()-overall_start)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
