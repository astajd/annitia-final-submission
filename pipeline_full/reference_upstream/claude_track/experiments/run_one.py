"""Run a single (endpoint, death_mode, feature_set, model) configuration.

Usage: python3 run_one.py <endpoint> <death_mode_or_n/a> <feature_set> <model_name>
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd
from src.config import REPORTS
from src.data import load_raw, build_targets
from src.cv import evaluate_cv, summarize
from src.features import build_features, FEATURE_SET_RISK
from src.models import (make_coxnet, make_rsf, make_xgb_cox,
                        make_lgbm_binary, make_catboost_binary)

MODELS = {
    "coxnet":      lambda: make_coxnet(l1_ratio=0.5),
    "rsf":         lambda: make_rsf(n_estimators=150),
    "xgb_cox":     lambda: make_xgb_cox(n_estimators=200, learning_rate=0.05),
    "lgbm_h5":     lambda: make_lgbm_binary(horizon=5.0, n_estimators=200),
    "catboost_h5": lambda: make_catboost_binary(horizon=5.0, iterations=200),
}

RESULTS_CSV = REPORTS / "phase1_cv_results.csv"


def main(endpoint, death_mode, fs, model_name):
    train, _ = load_raw()
    if endpoint == "hepatic":
        target_df, _ = build_targets(train, "drop_missing_death")
    else:
        _, target_df = build_targets(train, death_mode)

    valid = target_df[f"{endpoint}_valid"].to_numpy()
    e = target_df.loc[valid, f"{endpoint}_event"].to_numpy().astype(bool)
    t = target_df.loc[valid, f"{endpoint}_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)

    if fs == "strict_time_aligned":
        X = build_features(df, fs, event=e, time=t)
    else:
        X = build_features(df, fs)

    factory = MODELS[model_name]
    t0 = time.time()
    cv_res = evaluate_cv(factory(), X, e, t, n_splits=5, n_repeats=3)
    s = summarize(cv_res)
    elapsed = time.time() - t0

    row = {"endpoint": endpoint, "death_mode": death_mode if endpoint == "death" else "n/a",
           "feature_set": fs, "leakage_risk": FEATURE_SET_RISK[fs],
           "model": model_name, "n_rows": len(e), "n_events": int(e.sum()),
           "n_features": X.shape[1], **s, "elapsed_s": round(elapsed, 1)}

    REPORTS.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        df_row.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(RESULTS_CSV, mode="w", header=True, index=False)

    print(f"[{endpoint}/{death_mode}] {fs}/{model_name}: "
          f"C={s['mean']:.3f}±{s['std']:.3f} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
