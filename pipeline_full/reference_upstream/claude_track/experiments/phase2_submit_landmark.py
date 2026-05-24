"""Generate submissions/phase2_landmark_3y.{csv,json}.

Hepatic: 3y landmark features (Tier 2, fixed reference time) + tuned RSF
(reuses Optuna-best baseline_v1 params). Train cohort filtered to event-free
at landmark; test cohort is unfiltered (we predict everyone).

Death: XGB-Cox on longitudinal_summary, censor_missing_death_at_last
(Phase 2 grid had this saturated at 0.948 ± 0.018).

Single model per endpoint — no within-endpoint ensembling, to keep the
LB result interpretable as a clean signal on the landmark-3y hypothesis.
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import SUBMISSIONS, ROOT, HEPATIC_WEIGHT, DEATH_WEIGHT
from src.data import load_raw, build_targets, add_visit_metadata
from src.features import build_landmark_features, at_risk_at_landmark, build_features
from src.models import make_rsf, make_xgb_cox

CONFIGS = ROOT / "configs"
LANDMARK = 3.0


def main():
    t0 = time.time()
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    train, test = load_raw()
    print(f"Train {train.shape}  Test {test.shape}", flush=True)

    # ===== Hepatic =====
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e_full = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t_full = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df_full = train.loc[valid].reset_index(drop=True)

    keep = at_risk_at_landmark(e_full, t_full, LANDMARK)
    df_train_h = df_full.loc[keep].reset_index(drop=True)
    e_h = e_full[keep]
    t_h = t_full[keep]
    print(f"Hepatic landmark cohort (filter A, event-free at {LANDMARK}y): "
          f"n={len(e_h)}, events={int(e_h.sum())} "
          f"(dropped {int((~keep).sum())} pre-landmark events)", flush=True)

    rsf_rec = json.loads((CONFIGS / "optuna_rsf_baseline_v1.json").read_text())
    rsf_params = rsf_rec["best_params"]
    print(f"RSF tuned params: {rsf_params}", flush=True)

    X_tr_h = build_landmark_features(df_train_h, LANDMARK)
    X_te_h = build_landmark_features(test, LANDMARK)
    common_h = [c for c in X_tr_h.columns if c in X_te_h.columns]
    X_tr_h, X_te_h = X_tr_h[common_h], X_te_h[common_h]
    print(f"Hep features: {len(common_h)}", flush=True)

    # Test predictability check
    test_meta = add_visit_metadata(test)
    test_followup = (test_meta["max_age"] - test_meta["Age_v1"]).to_numpy()
    n_test_with_3y = int((test_followup >= LANDMARK - 1e-9).sum())

    factory_h = make_rsf(**rsf_params)
    risk_hep = factory_h(X_tr_h, e_h, t_h, X_te_h)
    print(f"Hep risk range: [{risk_hep.min():.3f}, {risk_hep.max():.3f}]",
          flush=True)

    # ===== Death =====
    _, d_t = build_targets(train, "censor_missing_death_at_last")
    valid_d = d_t["death_valid"].to_numpy()
    e_d = d_t.loc[valid_d, "death_event"].to_numpy().astype(bool)
    t_d = d_t.loc[valid_d, "death_time"].to_numpy().astype(float)
    df_train_d = train.loc[valid_d].reset_index(drop=True)
    print(f"Death cohort: n={len(e_d)}, events={int(e_d.sum())}", flush=True)

    X_tr_d = build_features(df_train_d, "longitudinal_summary")
    X_te_d = build_features(test, "longitudinal_summary")
    common_d = [c for c in X_tr_d.columns if c in X_te_d.columns]
    X_tr_d, X_te_d = X_tr_d[common_d], X_te_d[common_d]
    factory_d = make_xgb_cox(n_estimators=300, learning_rate=0.05)
    risk_death = factory_d(X_tr_d, e_d, t_d, X_te_d)
    print(f"Death risk range: [{risk_death.min():.3f}, {risk_death.max():.3f}]",
          flush=True)

    # ===== Submission =====
    sub = pd.DataFrame({
        "trustii_id": test["trustii_id"].values,
        "risk_hepatic_event": risk_hep,
        "risk_death": risk_death,
    })
    out_csv = SUBMISSIONS / "phase2_landmark_3y.csv"
    sub.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(sub)} rows)", flush=True)

    # CV numbers (from phase2d_audit, filter A, 5×10 CV with tuned RSF)
    HEP_CV_MEAN = 0.8798
    HEP_CV_STD = 0.0650
    HEP_CV_MS = 0.8148
    DEATH_CV_MEAN = 0.948
    DEATH_CV_STD = 0.018

    weighted_mean = HEPATIC_WEIGHT * HEP_CV_MEAN + DEATH_WEIGHT * DEATH_CV_MEAN
    weighted_ms = (HEPATIC_WEIGHT * HEP_CV_MS
                   + DEATH_WEIGHT * (DEATH_CV_MEAN - DEATH_CV_STD))

    meta = {
        "submission_kind": "ship candidate — Tier-2 landmark, audited",
        "hepatic": {
            "feature_set": f"landmark_{int(LANDMARK)}y",
            "feature_builder": "src/features.py::build_landmark_features",
            "tier": "2 (clean longitudinal, fixed reference time)",
            "model": "rsf",
            "params": rsf_params,
            "params_source": "Optuna best for rsf_baseline_v1 (Experiment 2)",
            "train_cohort": {
                "filter": "at_risk_at_landmark = drop event-before-landmark",
                "n_train": int(len(e_h)),
                "n_train_events": int(e_h.sum()),
                "n_dropped_pre_landmark": int((~keep).sum()),
            },
            "n_features": len(common_h),
            "cv_mean": HEP_CV_MEAN,
            "cv_std": HEP_CV_STD,
            "cv_mean_minus_std": HEP_CV_MS,
            "cv_protocol": "5-fold × 10-repeat stratified, filter A cohort",
        },
        "death": {
            "feature_set": "longitudinal_summary",
            "model": "xgb_cox",
            "params": {"n_estimators": 300, "learning_rate": 0.05},
            "death_mode": "censor_missing_death_at_last",
            "n_features": len(common_d),
            "cv_train_side_cindex": f"{DEATH_CV_MEAN} ± {DEATH_CV_STD} (Phase 2 grid)",
        },
        "ensemble": "single model per endpoint (no within-endpoint blending)",
        "weighted_score": {
            "formula": "0.7 * hep_mean + 0.3 * death_mean",
            "implied_cv_mean": weighted_mean,
            "implied_cv_mean_minus_std": weighted_ms,
            "expected_LB_user_estimate": "0.83 ± 0.05 (per phase2d_audit conclusion)",
        },
        "ood_risk": {
            "fraction_test_without_landmark_data": 1 - n_test_with_3y / len(test),
            "n_test_without_landmark_data": int(len(test) - n_test_with_3y),
            "note": (
                f"{len(test) - n_test_with_3y} of {len(test)} test patients "
                f"({100*(1 - n_test_with_3y/len(test)):.1f}%) have no observed "
                f"visit reaching {LANDMARK}y after Age_v1. For these rows the "
                f"LOCF-at-landmark feature is their last-observed value, which "
                f"may be at <{LANDMARK}y. The model was trained on a mixture "
                f"of these regimes (per Phase 2d audit), so predictions are "
                f"defined for all test rows but the OOD subset is the main "
                f"source of CV-vs-LB gap risk."
            ),
        },
        "audit_reference": "reports/phase2d_audit.md",
    }
    out_json = SUBMISSIONS / "phase2_landmark_3y.json"
    out_json.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote {out_json}", flush=True)
    print(f"\nImplied CV weighted: {weighted_mean:.4f} (m−s {weighted_ms:.4f})  "
          f"  Elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
