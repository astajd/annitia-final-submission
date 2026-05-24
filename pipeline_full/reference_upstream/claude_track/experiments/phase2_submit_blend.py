"""Build submissions/phase2_blend_landmark_permissive.{csv,json}.

Hepatic = 50/50 rank-average of:
  (a) 3y RSF landmark single-model test predictions (Tier 2)
  (b) Phase 2c permissive ensemble hepatic test predictions (Tier 4):
      tuned RSF/longitudinal_summary + tuned XGB-Cox/longitudinal_plus_meta
      + tuned XGB-Cox/longitudinal_summary, rank-averaged within (b).

Death = same as previous submissions (XGB-Cox/longitudinal_summary/censor).

Hypothesis: landmark methodology (Tier 2) and permissive features (Tier 4)
carry partially independent signal. If they do, the blend's LB should beat
the better of the two single submissions. If they don't, the blend just
averages two correlated predictors and lands between them.
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import SUBMISSIONS, ROOT, HEPATIC_WEIGHT, DEATH_WEIGHT
from src.data import load_raw, build_targets, add_visit_metadata
from src.features import build_features, build_landmark_features, at_risk_at_landmark
from src.models import make_rsf, make_xgb_cox

CONFIGS = ROOT / "configs"
LANDMARK = 3.0
OUT_CSV = SUBMISSIONS / "phase2_blend_landmark_permissive.csv"
OUT_JSON = SUBMISSIONS / "phase2_blend_landmark_permissive.json"


def make_factory(model, params):
    p = dict(params)
    if model == "rsf":     return make_rsf(**p)
    if model == "xgb_cox": return make_xgb_cox(**p)
    raise ValueError(model)


def predict_landmark_3y(train, test):
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)
    keep = at_risk_at_landmark(e, t, LANDMARK)
    df_lm = df.loc[keep].reset_index(drop=True)
    e_lm, t_lm = e[keep], t[keep]

    rsf_params = json.loads(
        (CONFIGS / "optuna_rsf_baseline_v1.json").read_text())["best_params"]
    X_tr = build_landmark_features(df_lm, LANDMARK)
    X_te = build_landmark_features(test, LANDMARK)
    common = [c for c in X_tr.columns if c in X_te.columns]
    risk = make_rsf(**rsf_params)(X_tr[common], e_lm, t_lm, X_te[common])
    return risk, {
        "feature_set": f"landmark_{int(LANDMARK)}y",
        "model": "rsf",
        "params": rsf_params,
        "n_train": int(len(e_lm)),
        "n_train_events": int(e_lm.sum()),
        "n_features": len(common),
        "cv_mean": 0.8798, "cv_std": 0.0650, "cv_mean_minus_std": 0.8148,
    }


def predict_permissive_ensemble(train, test):
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)

    members = []
    for cid in ["rsf_longitudinal_summary",
                "xgb_cox_longitudinal_plus_meta",
                "xgb_cox_longitudinal_summary"]:
        rec = json.loads((CONFIGS / f"optuna_{cid}.json").read_text())
        members.append({"id": cid, "model": rec["model"],
                        "feature_set": rec["feature_set"],
                        "params": rec["best_params"]})

    rank_sum = np.zeros(len(test))
    for m in members:
        X_tr = build_features(df, m["feature_set"])
        X_te = build_features(test, m["feature_set"])
        common = [c for c in X_tr.columns if c in X_te.columns]
        risk = make_factory(m["model"], m["params"])(
            X_tr[common], e, t, X_te[common])
        rank_sum += rankdata(risk)
    avg_rank = rank_sum / len(members)
    return avg_rank, {
        "members": [{"id": m["id"], "model": m["model"],
                     "feature_set": m["feature_set"]} for m in members],
        "n_train": int(len(e)),
        "n_train_events": int(e.sum()),
        "cv_mean": 0.8103, "cv_std": 0.0906, "cv_mean_minus_std": 0.7197,
    }


def predict_death(train, test):
    _, d_t = build_targets(train, "censor_missing_death_at_last")
    valid = d_t["death_valid"].to_numpy()
    e = d_t.loc[valid, "death_event"].to_numpy().astype(bool)
    t = d_t.loc[valid, "death_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)
    fs = "longitudinal_summary"
    X_tr = build_features(df, fs)
    X_te = build_features(test, fs)
    common = [c for c in X_tr.columns if c in X_te.columns]
    risk = make_xgb_cox(n_estimators=300, learning_rate=0.05)(
        X_tr[common], e, t, X_te[common])
    return risk, {
        "feature_set": fs, "model": "xgb_cox",
        "params": {"n_estimators": 300, "learning_rate": 0.05},
        "death_mode": "censor_missing_death_at_last",
        "n_features": len(common),
        "cv_train_side_cindex": "0.948 ± 0.018",
    }


def main():
    t0 = time.time()
    train, test = load_raw()
    print(f"Train {train.shape}  Test {test.shape}", flush=True)

    print("\n--- Hepatic component (a): 3y RSF landmark ---", flush=True)
    risk_lm, lm_info = predict_landmark_3y(train, test)
    print(f"  range [{risk_lm.min():.3f}, {risk_lm.max():.3f}]", flush=True)

    print("\n--- Hepatic component (b): permissive ensemble ---", flush=True)
    risk_perm, perm_info = predict_permissive_ensemble(train, test)
    print(f"  rank-avg range [{risk_perm.min():.1f}, {risk_perm.max():.1f}]",
          flush=True)

    print("\n--- Blend: 50/50 rank-average ---", flush=True)
    blend = 0.5 * rankdata(risk_lm) + 0.5 * rankdata(risk_perm)
    rank_corr = np.corrcoef(rankdata(risk_lm), rankdata(risk_perm))[0, 1]
    print(f"  Spearman corr between (a) and (b): {rank_corr:.4f}",
          flush=True)

    print("\n--- Death ---", flush=True)
    risk_death, d_info = predict_death(train, test)

    sub = pd.DataFrame({
        "trustii_id": test["trustii_id"].values,
        "risk_hepatic_event": blend,
        "risk_death": risk_death,
    })
    sub.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(sub)} rows)", flush=True)

    # Predicted LB band: landmark alone got 0.811 LB, permissive predicted 0.79-0.83.
    # If signals are partially independent (rank corr < 0.95), blend should
    # be in [0.81, 0.84]. If highly correlated (>0.95), blend ≈ avg ≈ 0.80.
    if rank_corr > 0.95:
        predicted = "0.79–0.82 (highly correlated components, expect ~average)"
    elif rank_corr > 0.85:
        predicted = "0.81–0.84 (partial complementarity)"
    else:
        predicted = "0.82–0.85 (strong complementarity expected)"

    weighted_landmark = (HEPATIC_WEIGHT * lm_info["cv_mean"]
                         + DEATH_WEIGHT * 0.948)
    weighted_perm = (HEPATIC_WEIGHT * perm_info["cv_mean"]
                     + DEATH_WEIGHT * 0.948)

    meta = {
        "submission_kind": "blend candidate — Tier-2 + Tier-4",
        "hypothesis": ("Landmark (Tier 2 fixed-reference-time) and permissive "
                       "(Tier 4 organizer-blessed full-trajectory) features "
                       "carry partially complementary signal. Blend should "
                       "beat single-component LBs if rank-correlation is "
                       "below ~0.95."),
        "hepatic": {
            "ensemble_strategy": "50/50 rank-average across (a) and (b)",
            "spearman_rank_correlation_a_vs_b": float(rank_corr),
            "component_a_landmark_3y": lm_info,
            "component_b_permissive": perm_info,
        },
        "death": d_info,
        "reference_LBs": {
            "phase2_landmark_3y_alone": 0.8109,
            "phase2_permissive_ensemble_alone": "pending",
            "phase1_ensemble": 0.7986,
        },
        "implied_cv_components": {
            "landmark_alone_weighted": weighted_landmark,
            "permissive_alone_weighted": weighted_perm,
        },
        "predicted_LB_band": predicted,
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Predicted LB: {predicted}", flush=True)
    print(f"Elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
