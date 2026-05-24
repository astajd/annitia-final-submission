"""Generate Phase-1 submission via rank-averaged ensemble.

Strategy:
- Pick a small ensemble per endpoint, hand-curated from CV results.
- Fit each model on full train, predict test.
- Rank-transform each model's risk scores (per endpoint), average ranks.
- Output trustii_id, risk_hepatic_event, risk_death.

Rank-averaging is the right ensembling strategy for C-index because only
relative ordering matters.
"""
from __future__ import annotations
import sys, time, warnings, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import REPORTS, SUBMISSIONS, DATA_RAW
from src.data import load_raw, build_targets
from src.features import build_features
from src.models import (make_coxnet, make_rsf, make_xgb_cox,
                        make_lgbm_binary, make_catboost_binary)


def fit_predict_full(model_factory, X_train, e_train, t_train, X_test):
    """Fit on full train, predict test."""
    return model_factory()(X_train, e_train, t_train, X_test)


# ---------------------------------------------------------------------------
# Ensemble specification
# ---------------------------------------------------------------------------
# Choices below are HONEST defaults: we avoid `longitudinal_summary` and 
# `strict_time_aligned` because the audit revealed leakage in trajectory features.
# We rely on `baseline_v1` and `nit_only` (clean) plus `early_v1_v3` (mild risk).

HEPATIC_ENSEMBLE = [
    ("baseline_v1", make_rsf, {"n_estimators": 300}),
    ("baseline_v1", make_xgb_cox, {"n_estimators": 300, "learning_rate": 0.05}),
    ("nit_only", make_xgb_cox, {"n_estimators": 300, "learning_rate": 0.05}),
    ("early_v1_v3", make_xgb_cox, {"n_estimators": 300, "learning_rate": 0.05}),
]

DEATH_ENSEMBLE = [
    ("baseline_v1", make_xgb_cox, {"n_estimators": 300, "learning_rate": 0.05}),
    ("baseline_v1", make_coxnet, {}),
    ("longitudinal_summary", make_xgb_cox, {"n_estimators": 300, "learning_rate": 0.05}),
]

DEATH_MODE = "censor_missing_death_at_last"  # uses all 1253 patients


def build_features_for_test(df_test_with_train_ids, fs):
    """Test-side feature builder — never uses event/time."""
    return build_features(df_test_with_train_ids, fs)


def predict_endpoint(train_df, test_df, target_df, endpoint, ensemble):
    """Returns per-test-row averaged rank score."""
    valid = target_df[f"{endpoint}_valid"].to_numpy()
    e = target_df.loc[valid, f"{endpoint}_event"].to_numpy().astype(bool)
    t = target_df.loc[valid, f"{endpoint}_time"].to_numpy().astype(float)
    train_df_v = train_df.loc[valid].reset_index(drop=True)

    test_n = len(test_df)
    rank_sum = np.zeros(test_n)
    n_models = len(ensemble)

    for fs, factory, kwargs in ensemble:
        # Build features on combined frame so columns align
        # (features are deterministic from raw cols, so build separately is fine
        # when no fitting transformer is used — our build_features is stateless)
        X_tr = build_features(train_df_v, fs)
        X_te = build_features_for_test(test_df, fs)
        # Align columns: take intersection, fill missing with NaN
        common = [c for c in X_tr.columns if c in X_te.columns]
        X_tr = X_tr[common]
        X_te = X_te[common]

        model = factory(**kwargs)
        risk = model(X_tr, e, t, X_te)
        # Rank-transform (higher rank = higher risk)
        ranks = rankdata(risk)
        rank_sum += ranks
        print(f"  [{endpoint}] {fs:24s} {factory.__name__:24s}  "
              f"risk range [{risk.min():.3f}, {risk.max():.3f}]")

    return rank_sum / n_models


def main():
    t0 = time.time()
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    train, test = load_raw()
    print(f"Train: {train.shape}  Test: {test.shape}")

    # Hepatic
    print("\n--- HEPATIC ENSEMBLE ---")
    hep_t, _ = build_targets(train, "drop_missing_death")
    risk_hep = predict_endpoint(train, test, hep_t, "hepatic", HEPATIC_ENSEMBLE)

    # Death
    print("\n--- DEATH ENSEMBLE ---")
    _, death_t = build_targets(train, DEATH_MODE)
    risk_death = predict_endpoint(train, test, death_t, "death", DEATH_ENSEMBLE)

    # Build submission
    sub = pd.DataFrame({
        "trustii_id": test["trustii_id"].values,
        "risk_hepatic_event": risk_hep,
        "risk_death": risk_death,
    })
    out_path = SUBMISSIONS / "phase1_ensemble.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(sub)} rows, {time.time()-t0:.0f}s)")
    print(sub.head())

    # Save metadata
    meta = {
        "hepatic_ensemble": [{"fs": fs, "model": f.__name__, "kwargs": kw}
                             for fs, f, kw in HEPATIC_ENSEMBLE],
        "death_ensemble":   [{"fs": fs, "model": f.__name__, "kwargs": kw}
                             for fs, f, kw in DEATH_ENSEMBLE],
        "death_mode": DEATH_MODE,
        "ranks_averaged": True,
    }
    (SUBMISSIONS / "phase1_ensemble.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote metadata.")


if __name__ == "__main__":
    main()
