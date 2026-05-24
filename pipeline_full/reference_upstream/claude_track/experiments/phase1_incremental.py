"""Phase 1 with incremental saving — robust to timeouts.

Reads the existing CSV at start, skips already-completed (endpoint, mode, fs, model)
rows, appends new ones.
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import REPORTS, SUBMISSIONS
from src.data import load_raw, build_targets
from src.cv import evaluate_cv, summarize
from src.features import build_features, FEATURE_SET_RISK
from src.models import (make_coxnet, make_rsf, make_xgb_cox,
                        make_lgbm_binary, make_catboost_binary)

warnings.filterwarnings("ignore")

PHASE1_REPEATS = 3   # default; override with --repeats
PHASE1_SPLITS = 5

FEATURE_SETS = [
    "baseline_v1",
    "early_v1_v3",
    "longitudinal_summary",
    "longitudinal_plus_meta",
    "nit_only",
    "strict_time_aligned",
    "followup_proxy_only",
]

MODELS = {
    "coxnet":      lambda: make_coxnet(l1_ratio=0.5),
    "rsf":         lambda: make_rsf(n_estimators=150),
    "xgb_cox":     lambda: make_xgb_cox(n_estimators=200, learning_rate=0.05),
    "lgbm_h5":     lambda: make_lgbm_binary(horizon=5.0, n_estimators=200),
    "catboost_h5": lambda: make_catboost_binary(horizon=5.0, iterations=200),
}

DEATH_MODES = ["drop_missing_death", "censor_missing_death_at_last"]
DEFAULT_RESULTS_CSV = REPORTS / "phase1_cv_results.csv"


def already_done(df_done, endpoint, dm, fs, mn):
    if df_done is None or len(df_done) == 0:
        return False
    sel = ((df_done["endpoint"] == endpoint) &
           (df_done["death_mode"].fillna("n/a") == (dm or "n/a")) &
           (df_done["feature_set"] == fs) &
           (df_done["model"] == mn))
    return sel.any()


def run_one(endpoint, fs_name, model_factory, df_full, target_df,
            n_splits=PHASE1_SPLITS, n_repeats=PHASE1_REPEATS):
    valid = target_df[f"{endpoint}_valid"].to_numpy()
    e = target_df.loc[valid, f"{endpoint}_event"].to_numpy().astype(bool)
    t = target_df.loc[valid, f"{endpoint}_time"].to_numpy().astype(float)
    df = df_full.loc[valid].reset_index(drop=True)
    if fs_name == "strict_time_aligned":
        X = build_features(df, fs_name, event=e, time=t)
    else:
        X = build_features(df, fs_name)
    cv_res = evaluate_cv(model_factory(), X, e, t,
                        n_splits=n_splits, n_repeats=n_repeats)
    return summarize(cv_res), X.shape[1], int(e.sum()), len(e)


def append_row(row, results_csv: Path):
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if results_csv.exists():
        df_row.to_csv(results_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(results_csv, mode="w", header=True, index=False)


def load_done(results_csv: Path):
    if results_csv.exists():
        return pd.read_csv(results_csv)
    return None


def main(only_endpoint=None, results_csv: Path = DEFAULT_RESULTS_CSV,
         n_repeats: int = PHASE1_REPEATS, n_splits: int = PHASE1_SPLITS):
    results_csv = Path(results_csv)
    train, _ = load_raw()
    df_done = load_done(results_csv)
    print(f"Output: {results_csv}")
    print(f"CV: {n_splits}-fold × {n_repeats}-repeat = {n_splits * n_repeats} folds")
    print(f"Already done: {0 if df_done is None else len(df_done)} rows")
    t_start = time.time()

    # Hepatic
    if only_endpoint in (None, "hepatic"):
        hep_t, _ = build_targets(train, "drop_missing_death")
        for fs in FEATURE_SETS:
            for mn, mf in MODELS.items():
                if already_done(df_done, "hepatic", "n/a", fs, mn):
                    continue
                tag = f"[H] {fs:24s} {mn:12s}"
                print(tag, end="  ", flush=True)
                t0 = time.time()
                try:
                    s, nfeat, nev, nrow = run_one("hepatic", fs, mf, train, hep_t,
                                                  n_splits=n_splits, n_repeats=n_repeats)
                    el = time.time() - t0
                    print(f"C={s['mean']:.3f}±{s['std']:.3f} nfeat={nfeat}  ({el:.0f}s)")
                    append_row({"endpoint": "hepatic", "death_mode": "n/a",
                                "feature_set": fs,
                                "leakage_risk": FEATURE_SET_RISK[fs],
                                "model": mn, "n_rows": nrow, "n_events": nev,
                                "n_features": nfeat, **s,
                                "elapsed_s": round(el, 1)}, results_csv)
                except Exception as ex:
                    print(f"FAIL: {ex!r}")
                    append_row({"endpoint": "hepatic", "death_mode": "n/a",
                                "feature_set": fs, "model": mn, "error": str(ex)},
                               results_csv)

    # Death
    if only_endpoint in (None, "death"):
        for dm in DEATH_MODES:
            _, dt = build_targets(train, dm)
            for fs in FEATURE_SETS:
                for mn, mf in MODELS.items():
                    if already_done(df_done, "death", dm, fs, mn):
                        continue
                    tag = f"[D mode={dm[:5]}] {fs:24s} {mn:12s}"
                    print(tag, end="  ", flush=True)
                    t0 = time.time()
                    try:
                        s, nfeat, nev, nrow = run_one("death", fs, mf, train, dt,
                                                      n_splits=n_splits, n_repeats=n_repeats)
                        el = time.time() - t0
                        print(f"C={s['mean']:.3f}±{s['std']:.3f} nfeat={nfeat}  ({el:.0f}s)")
                        append_row({"endpoint": "death", "death_mode": dm,
                                    "feature_set": fs,
                                    "leakage_risk": FEATURE_SET_RISK[fs],
                                    "model": mn, "n_rows": nrow, "n_events": nev,
                                    "n_features": nfeat, **s,
                                    "elapsed_s": round(el, 1)}, results_csv)
                    except Exception as ex:
                        print(f"FAIL: {ex!r}")
                        append_row({"endpoint": "death", "death_mode": dm,
                                    "feature_set": fs, "model": mn, "error": str(ex)},
                                   results_csv)

    print(f"\nTotal elapsed: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run Phase 1 bake-off grid (resumable).")
    p.add_argument("endpoint", nargs="?", default=None,
                   choices=["hepatic", "death"],
                   help="optional positional: limit to one endpoint")
    p.add_argument("--output", type=Path, default=DEFAULT_RESULTS_CSV,
                   help=f"output CSV path (default: {DEFAULT_RESULTS_CSV})")
    p.add_argument("--repeats", type=int, default=PHASE1_REPEATS,
                   help=f"CV repeats (default: {PHASE1_REPEATS})")
    args = p.parse_args()
    main(only_endpoint=args.endpoint, results_csv=args.output, n_repeats=args.repeats)
