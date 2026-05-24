"""Build submissions/phase2_landmark_multi.{csv,json}.

Steps:
  1. Run 4y filter-A CV (RSF, tuned params) — fills missing row in
     phase2_cv_results.csv to complete the {1,2,3,4,5}y landmark sweep.
  2. Run filter-B CV at {1,2,3}y — these are the cohorts the multi-landmark
     submission's component models are fit on, so the metadata reports
     honest CV numbers.
  3. Build hepatic ensemble: rank-average of RSF predictions at landmarks
     {1,2,3}y, each model fit on its own filter-B cohort (train), predicted
     on the full test set using LOCF features at that landmark.
  4. Death = same XGB-Cox/longitudinal_summary/censor.

Hypothesis: multi-landmark rank-averaging reduces variance from the
landmark-3y single-fold instability (per-fold spread 0.59–0.97 in the audit)
without sacrificing mean. Predicted LB: 0.81–0.84 if landmarks carry
partially independent signal; ~0.81 if highly correlated.
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
from src.cv import evaluate_cv, summarize
from src.data import load_raw, build_targets, add_visit_metadata
from src.features import build_features, build_landmark_features, at_risk_at_landmark, FEATURE_SET_RISK
from src.models import make_rsf, make_xgb_cox

CONFIGS = ROOT / "configs"
PHASE2_CV_CSV = REPORTS / "phase2_cv_results.csv"
OUT_CSV = SUBMISSIONS / "phase2_landmark_multi.csv"
OUT_JSON = SUBMISSIONS / "phase2_landmark_multi.json"

ENSEMBLE_LANDMARKS = [1.0, 2.0, 3.0]
SWEEP_FILL = [4.0]  # 1y, 2y, 3y, 5y already in CSV; 4y is missing


def filter_B_mask(df_subset: pd.DataFrame, landmark: float) -> np.ndarray:
    """Within the subset (already filter-A), keep only patients with
    visits actually reaching landmark age."""
    d = add_visit_metadata(df_subset)
    return ((d["max_age"] - d["Age_v1"]).to_numpy() >= landmark - 1e-9)


def append_phase2_cv_row(s, fs_name, model_name, n_features, n_rows,
                         n_events, elapsed):
    row = {
        "endpoint": "hepatic", "death_mode": "n/a",
        "feature_set": fs_name,
        "leakage_risk": "low (Tier 2 fixed reference)",
        "model": model_name,
        "n_rows": n_rows, "n_events": n_events, "n_features": n_features,
        **s, "elapsed_s": round(elapsed, 1),
    }
    df_row = pd.DataFrame([row])
    if PHASE2_CV_CSV.exists():
        df_row.to_csv(PHASE2_CV_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(PHASE2_CV_CSV, mode="w", header=True, index=False)


def main():
    overall_start = time.time()
    train, test = load_raw()
    print(f"Train {train.shape}  Test {test.shape}", flush=True)

    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e_full = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t_full = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df_full = train.loc[valid].reset_index(drop=True)

    rsf_params = json.loads(
        (CONFIGS / "optuna_rsf_baseline_v1.json").read_text())["best_params"]
    print(f"RSF tuned params: {rsf_params}", flush=True)

    # ===== 1. 4y filter-A CV (fill missing row) =====
    print("\n========== 1. 4y filter-A CV (fill grid) ==========", flush=True)
    for lt in SWEEP_FILL:
        keep = at_risk_at_landmark(e_full, t_full, lt)
        df_lm = df_full.loc[keep].reset_index(drop=True)
        e, t = e_full[keep], t_full[keep]
        X = build_landmark_features(df_lm, lt)
        t0 = time.time()
        cv_res = evaluate_cv(make_rsf(**rsf_params), X, e, t,
                             n_splits=5, n_repeats=10)
        s = summarize(cv_res)
        el = time.time() - t0
        fs_name = f"landmark_{int(lt)}y"
        append_phase2_cv_row(s, fs_name, "rsf_landmark", X.shape[1],
                             int(len(e)), int(e.sum()), el)
        print(f"  {lt}y filter A: n={len(e)}, events={int(e.sum())}, "
              f"mean={s['mean']:.4f} std={s['std']:.4f} "
              f"m−s={s['mean']-s['std']:.4f} ({el:.0f}s)", flush=True)

    # ===== 2. Filter-B CV at ensemble landmarks =====
    print("\n========== 2. Filter-B CV at ensemble landmarks ==========",
          flush=True)
    filter_b_results = {}
    for lt in ENSEMBLE_LANDMARKS:
        keep_A = at_risk_at_landmark(e_full, t_full, lt)
        df_A = df_full.loc[keep_A].reset_index(drop=True)
        e_A, t_A = e_full[keep_A], t_full[keep_A]
        keep_B_local = filter_B_mask(df_A, lt)
        df_B = df_A.loc[keep_B_local].reset_index(drop=True)
        e_B, t_B = e_A[keep_B_local], t_A[keep_B_local]

        X_B = build_landmark_features(df_B, lt)
        t0 = time.time()
        cv_res = evaluate_cv(make_rsf(**rsf_params), X_B, e_B, t_B,
                             n_splits=5, n_repeats=10)
        s = summarize(cv_res)
        el = time.time() - t0
        filter_b_results[lt] = {
            "n_train": int(len(e_B)), "n_train_events": int(e_B.sum()),
            "cv_mean": s["mean"], "cv_std": s["std"],
            "cv_mean_minus_std": s["mean"] - s["std"],
            "n_features": int(X_B.shape[1]),
        }
        print(f"  {lt}y filter B: n={len(e_B)}, events={int(e_B.sum())}, "
              f"mean={s['mean']:.4f} std={s['std']:.4f} "
              f"m−s={s['mean']-s['std']:.4f} ({el:.0f}s)", flush=True)

    # ===== 3. Build ensemble: fit each landmark on filter-B cohort, predict test =====
    print("\n========== 3. Multi-landmark test prediction ==========",
          flush=True)
    test_meta = add_visit_metadata(test)
    test_followup = (test_meta["max_age"] - test_meta["Age_v1"]).to_numpy()

    rank_sum = np.zeros(len(test))
    member_test_predictability = []
    risk_per_landmark = {}
    for lt in ENSEMBLE_LANDMARKS:
        keep_A = at_risk_at_landmark(e_full, t_full, lt)
        df_A = df_full.loc[keep_A].reset_index(drop=True)
        e_A, t_A = e_full[keep_A], t_full[keep_A]
        keep_B_local = filter_B_mask(df_A, lt)
        df_B = df_A.loc[keep_B_local].reset_index(drop=True)
        e_B, t_B = e_A[keep_B_local], t_A[keep_B_local]

        X_tr = build_landmark_features(df_B, lt)
        X_te = build_landmark_features(test, lt)
        common = [c for c in X_tr.columns if c in X_te.columns]
        risk = make_rsf(**rsf_params)(X_tr[common], e_B, t_B, X_te[common])
        risk_per_landmark[lt] = risk
        rank_sum += rankdata(risk)
        n_pred = int((test_followup >= lt - 1e-9).sum())
        print(f"  {lt}y: trained on n={len(e_B)} events={int(e_B.sum())}, "
              f"test rows reaching landmark: {n_pred}/{len(test)} "
              f"({100*n_pred/len(test):.1f}%)", flush=True)
        member_test_predictability.append({
            "landmark_yrs": lt,
            "test_rows_reaching_landmark": n_pred,
            "test_rows_OOD_below_landmark": int(len(test) - n_pred),
        })
    avg_rank_hep = rank_sum / len(ENSEMBLE_LANDMARKS)

    # Pairwise rank-correlations between landmark predictions
    pair_corrs = {}
    for i, l1 in enumerate(ENSEMBLE_LANDMARKS):
        for l2 in ENSEMBLE_LANDMARKS[i+1:]:
            c = float(np.corrcoef(rankdata(risk_per_landmark[l1]),
                                  rankdata(risk_per_landmark[l2]))[0, 1])
            pair_corrs[f"{l1}y_vs_{l2}y"] = c
    print(f"\n  Pairwise Spearman correlations: {pair_corrs}", flush=True)

    # ===== 4. Death =====
    print("\n========== 4. Death ==========", flush=True)
    _, d_t = build_targets(train, "censor_missing_death_at_last")
    valid_d = d_t["death_valid"].to_numpy()
    e_d = d_t.loc[valid_d, "death_event"].to_numpy().astype(bool)
    t_d = d_t.loc[valid_d, "death_time"].to_numpy().astype(float)
    df_train_d = train.loc[valid_d].reset_index(drop=True)
    fs = "longitudinal_summary"
    X_tr_d = build_features(df_train_d, fs)
    X_te_d = build_features(test, fs)
    common_d = [c for c in X_tr_d.columns if c in X_te_d.columns]
    risk_death = make_xgb_cox(n_estimators=300, learning_rate=0.05)(
        X_tr_d[common_d], e_d, t_d, X_te_d[common_d])
    print(f"  death range [{risk_death.min():.3f}, {risk_death.max():.3f}]",
          flush=True)

    sub = pd.DataFrame({
        "trustii_id": test["trustii_id"].values,
        "risk_hepatic_event": avg_rank_hep,
        "risk_death": risk_death,
    })
    sub.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(sub)} rows)", flush=True)

    # Pick a predicted LB band based on max pair correlation
    max_corr = max(pair_corrs.values())
    if max_corr > 0.95:
        predicted = (f"0.80–0.83 (max landmark-pair corr={max_corr:.3f}, "
                     "high redundancy — expect ≈ landmark_3y alone)")
    elif max_corr > 0.85:
        predicted = (f"0.81–0.84 (max corr={max_corr:.3f}, partial "
                     "complementarity — modest gain over 3y alone)")
    else:
        predicted = (f"0.82–0.85 (max corr={max_corr:.3f}, strong "
                     "complementarity — meaningful variance reduction)")

    meta = {
        "submission_kind": "multi-landmark candidate (Tier 2)",
        "hypothesis": ("Rank-averaging RSF predictions across landmarks "
                       f"{ENSEMBLE_LANDMARKS} reduces the per-fold "
                       "instability seen at single-landmark 3y "
                       "(audit: spread 0.59–0.97 across 50 folds). "
                       "Variance reduction expected if the three landmark "
                       "predictions are < ~0.95 rank-correlated."),
        "hepatic": {
            "ensemble_strategy": "rank-average across landmarks",
            "landmarks": ENSEMBLE_LANDMARKS,
            "filter": "B (event-free at landmark AND has data through landmark)",
            "rsf_params": rsf_params,
            "rsf_params_source": "Optuna best for rsf_baseline_v1 (Experiment 2)",
            "filter_B_cv_per_landmark": {
                str(lt): filter_b_results[lt] for lt in ENSEMBLE_LANDMARKS
            },
            "test_predictability_per_landmark": member_test_predictability,
            "pairwise_test_rank_correlations": pair_corrs,
        },
        "death": {
            "feature_set": fs, "model": "xgb_cox",
            "params": {"n_estimators": 300, "learning_rate": 0.05},
            "death_mode": "censor_missing_death_at_last",
            "n_features": len(common_d),
            "cv_train_side_cindex": "0.948 ± 0.018 (Phase 2 grid)",
        },
        "reference_LBs": {
            "phase2_landmark_3y_alone": 0.8109,
            "phase1_ensemble": 0.7986,
            "phase2_honest_ensemble": 0.7509,
        },
        "predicted_LB_band": predicted,
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Predicted LB: {predicted}", flush=True)
    print(f"\nTotal elapsed: {(time.time()-overall_start)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
