"""Phase 2e — generate two ship-candidate hepatic ensembles (no submission).

Honest ensemble (qualitative-track-friendly, Tier 1):
  - tuned RSF baseline_v1
  - tuned XGB-Cox baseline_v1
  - tuned RSF nit_only_baseline_only

Permissive ensemble (quantitative track, Tier 4 — organizer-blessed):
  - top 3 from Phase 2c by m−s

Death endpoint for both: untuned XGB-Cox on longitudinal_summary,
censor_missing_death_at_last (Phase 2 grid had this at 0.948±0.018, saturated).

Hepatic ensemble strategy: rank-average risks across base models per endpoint.

Outputs:
  submissions/phase2_honest_ensemble.{csv,json}
  submissions/phase2_permissive_ensemble.{csv,json}
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import REPORTS, ROOT, SUBMISSIONS, HEPATIC_WEIGHT, DEATH_WEIGHT
from src.cv import evaluate_cv, summarize, repeated_stratified_folds, cindex
from src.data import load_raw, build_targets
from src.features import build_features
from src.models import make_rsf, make_xgb_cox, make_lgbm_binary, make_coxnet, make_catboost_binary

CONFIGS = ROOT / "configs"

HONEST_MEMBERS = ["rsf_baseline_v1", "xgb_cox_baseline_v1",
                  "rsf_nit_only_baseline_only"]


def make_factory(model, params):
    p = dict(params)
    if model == "rsf":      return make_rsf(**p)
    if model == "xgb_cox":  return make_xgb_cox(**p)
    if model == "coxnet":   return make_coxnet(**p)
    if model == "lgbm_bin":
        h = p.pop("horizon")
        return make_lgbm_binary(horizon=float(h), **p)
    if model == "catboost_bin":
        h = p.pop("horizon")
        return make_catboost_binary(horizon=float(h), **p)
    raise ValueError(model)


def load_member(member_id):
    rec = json.loads((CONFIGS / f"optuna_{member_id}.json").read_text())
    return {
        "id": member_id,
        "model": rec["model"],
        "feature_set": rec["feature_set"],
        "params": rec["best_params"],
    }


def cv_ensemble_score(members, df_train, e, t, n_splits=5, n_repeats=10):
    """OOF-fold rank-average ensemble C-index across folds."""
    Xs = {m["id"]: build_features(df_train, m["feature_set"]) for m in members}
    fold_cis = []
    for r, f, tr_idx, va_idx in repeated_stratified_folds(
            e, n_splits=n_splits, n_repeats=n_repeats):
        rank_sum = np.zeros(len(va_idx))
        for m in members:
            X = Xs[m["id"]]
            risk = make_factory(m["model"], m["params"])(
                X.iloc[tr_idx], e[tr_idx], t[tr_idx], X.iloc[va_idx])
            rank_sum += rankdata(risk)
        avg_rank = rank_sum / len(members)
        fold_cis.append(cindex(e[va_idx], t[va_idx], avg_rank))
    fold_cis = np.array(fold_cis)
    return {
        "mean": float(fold_cis.mean()),
        "std": float(fold_cis.std(ddof=1)),
        "n_folds": int(len(fold_cis)),
    }


def predict_ensemble(members, df_train, e, t, df_test):
    """Fit each member on full train, predict test, rank-average."""
    n_test = len(df_test)
    rank_sum = np.zeros(n_test)
    member_ranges = []
    for m in members:
        X_tr = build_features(df_train, m["feature_set"])
        X_te = build_features(df_test, m["feature_set"])
        common = [c for c in X_tr.columns if c in X_te.columns]
        X_tr, X_te = X_tr[common], X_te[common]
        risk = make_factory(m["model"], m["params"])(X_tr, e, t, X_te)
        member_ranges.append({
            "id": m["id"], "n_features": len(common),
            "risk_min": float(np.min(risk)), "risk_max": float(np.max(risk)),
        })
        rank_sum += rankdata(risk)
    return rank_sum / len(members), member_ranges


def predict_death(df_train, e_d, t_d, df_test):
    """Death model: untuned XGB-Cox on longitudinal_summary, censor mode."""
    fs = "longitudinal_summary"
    X_tr = build_features(df_train, fs)
    X_te = build_features(df_test, fs)
    common = [c for c in X_tr.columns if c in X_te.columns]
    X_tr, X_te = X_tr[common], X_te[common]
    factory = make_xgb_cox(n_estimators=300, learning_rate=0.05)
    risk = factory(X_tr, e_d, t_d, X_te)
    return risk, len(common)


def write_ensemble(members, kind, label, df_train_h, e_h, t_h,
                   df_train_d, e_d, t_d, test, hep_score):
    sub_csv = SUBMISSIONS / f"phase2_{kind}_ensemble.csv"
    sub_json = SUBMISSIONS / f"phase2_{kind}_ensemble.json"

    print(f"\n--- {kind.upper()} ENSEMBLE ---", flush=True)
    print(f"  CV (5×10): mean={hep_score['mean']:.4f} std={hep_score['std']:.4f} "
          f"m−s={hep_score['mean']-hep_score['std']:.4f}", flush=True)

    # Hepatic: fit on full train, predict test
    risk_hep, hep_member_info = predict_ensemble(
        members, df_train_h, e_h, t_h, test)

    # Death
    risk_death, n_death_features = predict_death(df_train_d, e_d, t_d, test)
    DEATH_MEAN = 0.948  # from Phase 2 grid (longitudinal_summary, xgb_cox, censor)
    DEATH_STD = 0.018

    sub = pd.DataFrame({
        "trustii_id": test["trustii_id"].values,
        "risk_hepatic_event": risk_hep,
        "risk_death": risk_death,
    })
    sub.to_csv(sub_csv, index=False)

    weighted = HEPATIC_WEIGHT * hep_score["mean"] + DEATH_WEIGHT * DEATH_MEAN
    weighted_ms = (HEPATIC_WEIGHT * (hep_score["mean"] - hep_score["std"])
                   + DEATH_WEIGHT * (DEATH_MEAN - DEATH_STD))

    meta = {
        "submission_kind": label,
        "hepatic": {
            "ensemble_strategy": "rank-average across members",
            "members": [{"id": m["id"], "model": m["model"],
                         "feature_set": m["feature_set"],
                         "params": m["params"],
                         **info}
                        for m, info in zip(members, hep_member_info)],
            "cv_protocol": "5-fold × 10-repeat stratified, oof rank-average per fold",
            "cv_mean": hep_score["mean"],
            "cv_std": hep_score["std"],
            "cv_mean_minus_std": hep_score["mean"] - hep_score["std"],
            "n_folds": hep_score["n_folds"],
        },
        "death": {
            "feature_set": "longitudinal_summary",
            "model": "xgb_cox",
            "params": {"n_estimators": 300, "learning_rate": 0.05},
            "death_mode": "censor_missing_death_at_last",
            "n_features": n_death_features,
            "cv_train_side_cindex": "0.948 ± 0.018 (Phase 2 grid)",
        },
        "weighted_score": {
            "formula": "0.7 * hep_mean + 0.3 * death_mean",
            "implied_cv_mean": weighted,
            "implied_cv_mean_minus_std": weighted_ms,
        },
    }
    sub_json.write_text(json.dumps(meta, indent=2, default=str))
    print(f"  Wrote {sub_csv} ({len(sub)} rows)", flush=True)
    print(f"  Implied weighted CV: {weighted:.4f} "
          f"(m−s {weighted_ms:.4f})", flush=True)
    return meta


def main():
    overall_start = time.time()
    train, test = load_raw()
    print(f"Train {train.shape}  Test {test.shape}", flush=True)

    # Hepatic targets / cohort
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e_h = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t_h = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df_train_h = train.loc[valid].reset_index(drop=True)
    print(f"Hepatic cohort: n={len(e_h)}, events={int(e_h.sum())}", flush=True)

    # Death targets / cohort
    _, d_t = build_targets(train, "censor_missing_death_at_last")
    valid_d = d_t["death_valid"].to_numpy()
    e_d = d_t.loc[valid_d, "death_event"].to_numpy().astype(bool)
    t_d = d_t.loc[valid_d, "death_time"].to_numpy().astype(float)
    df_train_d = train.loc[valid_d].reset_index(drop=True)
    print(f"Death cohort: n={len(e_d)}, events={int(e_d.sum())}", flush=True)

    # ---- HONEST ensemble ----
    honest_members = [load_member(mid) for mid in HONEST_MEMBERS]
    print(f"\nHonest ensemble members:")
    for m in honest_members:
        print(f"  - {m['id']} ({m['model']} on {m['feature_set']})", flush=True)
    honest_score = cv_ensemble_score(honest_members, df_train_h, e_h, t_h)
    write_ensemble(honest_members, "honest",
                   "honest ensemble — Tier 1+2 only (qualitative-track-safe)",
                   df_train_h, e_h, t_h, df_train_d, e_d, t_d, test,
                   honest_score)

    # ---- PERMISSIVE ensemble (top 3 from Phase 2c by m−s) ----
    p2c_records = []
    for cid in ["rsf_longitudinal_summary",
                "xgb_cox_longitudinal_summary",
                "xgb_cox_longitudinal_plus_meta",
                "lgbm_bin_longitudinal_plus_meta"]:
        path = CONFIGS / f"optuna_{cid}.json"
        if path.exists():
            r = json.loads(path.read_text())
            p2c_records.append(r)
    p2c_records.sort(key=lambda r: -r["best_mean_minus_std"])
    top3_p2c = p2c_records[:3]
    permissive_members = [
        {"id": r["id"], "model": r["model"],
         "feature_set": r["feature_set"], "params": r["best_params"]}
        for r in top3_p2c
    ]
    print(f"\nPermissive ensemble members (top 3 from Phase 2c by m−s):")
    for r in top3_p2c:
        print(f"  - {r['id']}  m−s={r['best_mean_minus_std']:.4f}", flush=True)
    permissive_score = cv_ensemble_score(permissive_members, df_train_h, e_h, t_h)
    write_ensemble(permissive_members, "permissive",
                   "permissive ensemble — Tier 4 (organizer-blessed quantitative track)",
                   df_train_h, e_h, t_h, df_train_d, e_d, t_d, test,
                   permissive_score)

    print(f"\nPhase 2e wall-clock: {(time.time()-overall_start)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
