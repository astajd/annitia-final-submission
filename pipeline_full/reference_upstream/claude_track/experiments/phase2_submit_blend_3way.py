"""Build submissions/phase2_blend_3way.{csv,json}.

Hepatic = 1/3 rank-average of:
  (a) 3y RSF landmark single-model (Tier 2)
  (b) Phase 2c permissive ensemble (Tier 4: rank-avg of 3 longitudinal models)
  (c) tuned XGB-Cox on baseline_v1 (Tier 1) — selected by OOF correlation
      audit (rank-corr 0.492 vs landmark, 0.433 vs permissive, both < 0.6)

Death = same XGB-Cox/longitudinal_summary/censor as previous submissions.

Hypothesis: a third member that's < 0.6 rank-correlated with both anchors on
training OOF will add information that 2-way blend can't capture. If LB beats
0.880, cross-method diversity has another tier of headroom; if LB ≈ 0.880,
the 3rd model's signal is largely redundant with the 2-way blend on test.
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import REPORTS, SUBMISSIONS, ROOT, HEPATIC_WEIGHT, DEATH_WEIGHT
from src.data import load_raw, build_targets
from src.features import (build_features, build_landmark_features,
                          at_risk_at_landmark)
from src.models import (make_rsf, make_xgb_cox, make_coxnet,
                        make_lgbm_binary, make_catboost_binary)

CONFIGS = ROOT / "configs"
PICK_JSON = REPORTS / "phase2_oof_diverse_pick.json"
OUT_CSV = SUBMISSIONS / "phase2_blend_3way.csv"
OUT_JSON = SUBMISSIONS / "phase2_blend_3way.json"
LANDMARK = 3.0


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


def predict_landmark_3y(df_train, e_train, t_train, test):
    keep = at_risk_at_landmark(e_train, t_train, LANDMARK)
    df_lm = df_train.loc[keep].reset_index(drop=True)
    e_lm, t_lm = e_train[keep], t_train[keep]
    rsf_params = json.loads(
        (CONFIGS / "optuna_rsf_baseline_v1.json").read_text())["best_params"]
    X_tr = build_landmark_features(df_lm, LANDMARK)
    X_te = build_landmark_features(test, LANDMARK)
    common = [c for c in X_tr.columns if c in X_te.columns]
    risk = make_rsf(**rsf_params)(X_tr[common], e_lm, t_lm, X_te[common])
    return risk, {
        "feature_set": f"landmark_{int(LANDMARK)}y",
        "model": "rsf", "params": rsf_params,
        "n_train": int(len(e_lm)), "n_train_events": int(e_lm.sum()),
        "n_features": len(common),
    }


def predict_permissive_ensemble(df_train, e, t, test):
    member_ids = ["rsf_longitudinal_summary",
                  "xgb_cox_longitudinal_plus_meta",
                  "xgb_cox_longitudinal_summary"]
    rank_sum = np.zeros(len(test))
    member_info = []
    for cid in member_ids:
        factory, rec = make_factory_from_id(cid)
        X_tr = build_features(df_train, rec["feature_set"])
        X_te = build_features(test, rec["feature_set"])
        common = [c for c in X_tr.columns if c in X_te.columns]
        risk = factory(X_tr[common], e, t, X_te[common])
        rank_sum += rankdata(risk)
        member_info.append({"id": cid, "model": rec["model"],
                            "feature_set": rec["feature_set"]})
    return rank_sum / len(member_ids), {
        "members": member_info,
        "n_train": int(len(e)), "n_train_events": int(e.sum()),
    }


def predict_single(third_id, df_train, e, t, test):
    factory, rec = make_factory_from_id(third_id)
    X_tr = build_features(df_train, rec["feature_set"])
    X_te = build_features(test, rec["feature_set"])
    common = [c for c in X_tr.columns if c in X_te.columns]
    risk = factory(X_tr[common], e, t, X_te[common])
    return risk, {
        "id": third_id,
        "feature_set": rec["feature_set"],
        "model": rec["model"],
        "params": rec["best_params"],
        "n_train": int(len(e)), "n_train_events": int(e.sum()),
        "n_features": len(common),
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
    if not PICK_JSON.exists():
        print(f"No diverse pick in {PICK_JSON} — Task 1 found no candidate "
              f"with rank-corr < 0.6 vs both anchors. Aborting blend_3way.",
              flush=True)
        return
    pick = json.loads(PICK_JSON.read_text())
    label = pick["id"]
    # OOF report uses report-friendly label (e.g. "tuned_xgb_cox_baseline_v1");
    # config files are stored without the "tuned_" prefix.
    third_id = label[len("tuned_"):] if label.startswith("tuned_") else label
    print(f"Diverse third member: {label} (config id: {third_id}) "
          f"(corr vs landmark = {pick['corr_vs_landmark']:.3f}, "
          f"vs permissive = {pick['corr_vs_permissive']:.3f})", flush=True)

    train, test = load_raw()
    print(f"Train {train.shape}  Test {test.shape}", flush=True)

    df_train, e, t = hepatic_targets(train)

    print("\n--- (a) 3y RSF landmark ---", flush=True)
    risk_lm, lm_info = predict_landmark_3y(df_train, e, t, test)
    print(f"  range [{risk_lm.min():.3f}, {risk_lm.max():.3f}]", flush=True)

    print("\n--- (b) permissive ensemble (rank-avg of 3) ---", flush=True)
    risk_perm, perm_info = predict_permissive_ensemble(df_train, e, t, test)
    print(f"  rank-avg range [{risk_perm.min():.1f}, {risk_perm.max():.1f}]",
          flush=True)

    print(f"\n--- (c) third member: {third_id} ---", flush=True)
    risk_third, third_info = predict_single(third_id, df_train, e, t, test)
    print(f"  range [{risk_third.min():.3f}, {risk_third.max():.3f}]",
          flush=True)

    # 1/3 rank-average
    blend = (rankdata(risk_lm)
             + rankdata(risk_perm)
             + rankdata(risk_third)) / 3.0

    # Test-set pairwise correlations
    pair_corrs = {
        "landmark_vs_permissive":
            float(np.corrcoef(rankdata(risk_lm), rankdata(risk_perm))[0, 1]),
        "landmark_vs_third":
            float(np.corrcoef(rankdata(risk_lm), rankdata(risk_third))[0, 1]),
        "permissive_vs_third":
            float(np.corrcoef(rankdata(risk_perm), rankdata(risk_third))[0, 1]),
    }
    print(f"\n  test pairwise rank-corrs: {pair_corrs}", flush=True)

    print("\n--- death ---", flush=True)
    risk_death, d_info = predict_death(train, test)

    sub = pd.DataFrame({
        "trustii_id": test["trustii_id"].values,
        "risk_hepatic_event": blend,
        "risk_death": risk_death,
    })
    sub.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(sub)} rows)", flush=True)

    max_pair_corr = max(pair_corrs.values())
    if max_pair_corr > 0.85:
        predicted = (f"0.86–0.89 (one pair corr {max_pair_corr:.3f} — "
                     "redundancy may cap gain)")
    elif max_pair_corr > 0.65:
        predicted = (f"0.87–0.90 (max pair corr {max_pair_corr:.3f} — "
                     "modest gain over 2-way blend)")
    else:
        predicted = (f"0.88–0.91 (max pair corr {max_pair_corr:.3f} — "
                     "strong 3-way diversity)")

    meta = {
        "submission_kind": "3-way blend (Tier-2 + Tier-4 + diverse Tier-1)",
        "hypothesis": ("Adding a Tier-1 model with OOF rank-corr < 0.6 vs both "
                       "the 2-way blend's anchors should add information that "
                       "the 2-way blend cannot capture. Pre-registered: if LB "
                       "beats 0.88033, diversity has another tier; if LB "
                       "≈ 0.88, the 3rd model is redundant on test."),
        "hepatic": {
            "ensemble_strategy": "1/3 rank-average across (a), (b), (c)",
            "components": {
                "a_landmark_3y": lm_info,
                "b_permissive_ensemble": perm_info,
                "c_diverse_third": {
                    **third_info,
                    "oof_corr_vs_landmark": pick["corr_vs_landmark"],
                    "oof_corr_vs_permissive": pick["corr_vs_permissive"],
                    "selection_rule": ("OOF rank-corr < 0.6 vs BOTH "
                                       "landmark_3y AND permissive_ensemble"),
                },
            },
            "test_pairwise_rank_correlations": pair_corrs,
        },
        "death": d_info,
        "reference_LBs": {
            "phase2_blend_landmark_permissive": 0.88033,
            "phase2_landmark_3y": 0.8109,
            "phase2_landmark_multi": 0.81,
            "phase1_ensemble": 0.7986,
            "phase2_honest_ensemble": 0.7509,
        },
        "predicted_LB_band": predicted,
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"\nPredicted LB: {predicted}", flush=True)
    print(f"Elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
