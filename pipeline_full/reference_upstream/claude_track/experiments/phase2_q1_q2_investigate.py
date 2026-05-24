"""Diagnose why phase2_blend_3way underperformed (LB 0.829 vs 2-way 0.880).

Q1 (weight vs component, in-memory only):
  Build a 50/30/20 weighted version of the same three components used in
  phase2_blend_3way:
      0.5 * rank(landmark) + 0.3 * rank(permissive) + 0.2 * rank(xgb_baseline)
  and compute its Spearman rank-correlation against the already-shipped 2-way
  blend (0.5*rank(landmark) + 0.5*rank(permissive)).
    - corr > 0.95  → the 50/30/20 ordering is essentially the 2-way; failure was
                     equal-weighting the third member, not the third member itself.
    - 0.85-0.92    → re-weighting could meaningfully recover, worth submitting.
    - < 0.85       → the third member's signal really is OOD on test;
                     re-weighting won't save it.
  No file is saved for Q1.

Q2 (different third member):
  Build phase2_blend_3way_catboost.{csv,json} as the 1/3 rank-average of
  (landmark_3y_RSF, permissive_ensemble, tuned_catboost_bin_early_v1_v3).
  Catboost was the only candidate in the 8-model OOF audit with rank-corr
  *below* 0.2 vs landmark (0.117); it just-failed the < 0.6 vs both anchors
  bar (0.651 vs permissive). It is the most-different-flavoured single model
  we have access to, which is the natural alternative to xgb_cox_baseline_v1.
  Pre-registered: if LB > 0.880, the 3-way recipe is right and xgb_cox_v1 was
  the wrong third member; if LB ≈ 0.83, single Tier-1 thirds don't help on
  test regardless of who they are.
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.config import REPORTS, SUBMISSIONS, ROOT
from src.data import load_raw, build_targets
from src.features import (build_features, build_landmark_features,
                          at_risk_at_landmark)
from src.models import (make_rsf, make_xgb_cox, make_coxnet,
                        make_lgbm_binary, make_catboost_binary)

CONFIGS = ROOT / "configs"
OUT_CSV = SUBMISSIONS / "phase2_blend_3way_catboost.csv"
OUT_JSON = SUBMISSIONS / "phase2_blend_3way_catboost.json"
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
    }


def main():
    t0 = time.time()
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

    print("\n--- (c-xgb) tuned_xgb_cox_baseline_v1 (Q1 third member) ---",
          flush=True)
    risk_xgb, xgb_info = predict_single("xgb_cox_baseline_v1",
                                        df_train, e, t, test)
    print(f"  range [{risk_xgb.min():.3f}, {risk_xgb.max():.3f}]", flush=True)

    print("\n--- (c-cat) tuned_catboost_bin_early_v1_v3 (Q2 third member) ---",
          flush=True)
    risk_cat, cat_info = predict_single("catboost_bin_early_v1_v3",
                                        df_train, e, t, test)
    print(f"  range [{risk_cat.min():.3f}, {risk_cat.max():.3f}]", flush=True)

    # --- Q1 ---
    r_lm, r_perm, r_xgb = rankdata(risk_lm), rankdata(risk_perm), rankdata(risk_xgb)
    blend_2way = 0.5 * r_lm + 0.5 * r_perm
    blend_503020 = 0.5 * r_lm + 0.3 * r_perm + 0.2 * r_xgb
    blend_3way_equal = (r_lm + r_perm + r_xgb) / 3.0  # for reference

    rho_503020_vs_2way = float(spearmanr(blend_503020, blend_2way).statistic)
    rho_equal_vs_2way = float(spearmanr(blend_3way_equal, blend_2way).statistic)
    rho_503020_vs_equal = float(spearmanr(blend_503020, blend_3way_equal).statistic)

    print("\n=== Q1: weight-vs-component diagnostic ===", flush=True)
    print(f"  Spearman( 50/30/20 blend , 2-way blend ) = "
          f"{rho_503020_vs_2way:.4f}", flush=True)
    print(f"  Spearman( equal 1/3 blend , 2-way blend ) = "
          f"{rho_equal_vs_2way:.4f}  (ref: equal blend got LB 0.829)",
          flush=True)
    print(f"  Spearman( 50/30/20 blend , equal 1/3 blend ) = "
          f"{rho_503020_vs_equal:.4f}", flush=True)

    if rho_503020_vs_2way > 0.95:
        q1_verdict = (
            f"corr={rho_503020_vs_2way:.3f} > 0.95: the 50/30/20 ordering is "
            "essentially the 2-way ordering. xgb_cox_v1 was *down-weighted* "
            "enough that the blend reverts to 2-way — re-weighting alone would "
            "give us back ~0.880 LB but no upside. Failure mode at equal "
            "weight was over-weighting the third member, not the component."
        )
    elif rho_503020_vs_2way > 0.85:
        q1_verdict = (
            f"corr={rho_503020_vs_2way:.3f} in (0.85, 0.95]: 50/30/20 keeps "
            "most of the 2-way ordering but with a clear xgb-flavoured shift. "
            "Re-weighting could meaningfully recover; worth a submission slot."
        )
    else:
        q1_verdict = (
            f"corr={rho_503020_vs_2way:.3f} ≤ 0.85: even down-weighted, the "
            "xgb component shifts the blend significantly. The third member's "
            "signal really is OOD on test; re-weighting won't save it."
        )
    print(f"\n  Q1 verdict: {q1_verdict}", flush=True)

    # --- Q2: 1/3 blend with catboost as third ---
    r_cat = rankdata(risk_cat)
    blend_cat = (r_lm + r_perm + r_cat) / 3.0

    pair_corrs_cat = {
        "landmark_vs_permissive":
            float(np.corrcoef(r_lm, r_perm)[0, 1]),
        "landmark_vs_catboost":
            float(np.corrcoef(r_lm, r_cat)[0, 1]),
        "permissive_vs_catboost":
            float(np.corrcoef(r_perm, r_cat)[0, 1]),
    }
    print("\n=== Q2: 3-way blend with catboost as third ===", flush=True)
    print(f"  test pairwise rank-corrs: {pair_corrs_cat}", flush=True)

    # Spearman of new blend vs (2-way, equal-xgb 3-way) for context
    rho_cat_vs_2way = float(spearmanr(blend_cat, blend_2way).statistic)
    rho_cat_vs_equal_xgb = float(spearmanr(blend_cat, blend_3way_equal).statistic)
    print(f"  Spearman( catboost 3-way , 2-way blend )      = "
          f"{rho_cat_vs_2way:.4f}", flush=True)
    print(f"  Spearman( catboost 3-way , xgb 3-way (LB.829)) = "
          f"{rho_cat_vs_equal_xgb:.4f}", flush=True)

    print("\n--- death ---", flush=True)
    risk_death, d_info = predict_death(train, test)

    sub = pd.DataFrame({
        "trustii_id": test["trustii_id"].values,
        "risk_hepatic_event": blend_cat,
        "risk_death": risk_death,
    })
    sub.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(sub)} rows)", flush=True)

    meta = {
        "submission_kind": (
            "3-way blend (Tier-2 + Tier-4 + Tier-1 catboost) — Q2 alt third"
        ),
        "hypothesis": (
            "If equal-weight 3-way fails because xgb_cox_baseline_v1's signal "
            "is OOD-flavoured on test (LB 0.829), swap it for the most-"
            "different single Tier-1 model we have: catboost_bin_early_v1_v3 "
            "(OOF rank-corr 0.117 vs landmark, the lowest of any candidate). "
            "Pre-registered: LB > 0.880 → recipe is right and xgb_v1 was the "
            "wrong third; LB ≈ 0.83 → single Tier-1 thirds don't help on "
            "test regardless of which one."
        ),
        "hepatic": {
            "ensemble_strategy":
                "1/3 rank-average across (a) landmark, (b) permissive, (c) catboost",
            "components": {
                "a_landmark_3y": lm_info,
                "b_permissive_ensemble": perm_info,
                "c_third_catboost": {
                    **cat_info,
                    "oof_corr_vs_landmark": 0.117,
                    "oof_corr_vs_permissive": 0.651,
                    "selection_rationale": (
                        "Lowest OOF rank-corr vs landmark of any model in the "
                        "8-model audit (0.117); just-failed the <0.6 vs both "
                        "anchors bar (0.651 vs permissive) but is the most-"
                        "different-flavoured single model available."
                    ),
                },
            },
            "test_pairwise_rank_correlations": pair_corrs_cat,
            "spearman_vs_other_blends": {
                "vs_2way_LB_0.880": rho_cat_vs_2way,
                "vs_xgb_3way_LB_0.829": rho_cat_vs_equal_xgb,
            },
        },
        "death": d_info,
        "Q1_diagnostic": {
            "spearman_503020_vs_2way": rho_503020_vs_2way,
            "spearman_equal_vs_2way": rho_equal_vs_2way,
            "spearman_503020_vs_equal": rho_503020_vs_equal,
            "verdict": q1_verdict,
        },
        "reference_LBs": {
            "phase2_blend_landmark_permissive": 0.88033,
            "phase2_blend_3way_xgb_v1_equal": 0.829,
            "phase2_landmark_3y": 0.8109,
        },
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"\nElapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
