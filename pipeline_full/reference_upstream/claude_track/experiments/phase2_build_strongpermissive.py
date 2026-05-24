"""Build phase2_blend_2way_strongpermissive.csv under a relaxed criterion.

The Task 2 audit found rsf_longitudinal_summary alone has 5×10 m-s 0.7356,
beating the 3-member permissive ensemble's 0.7197 by +0.016 — the same
magnitude as the re-weighting LB win (+0.016). The strict decision rule
(test rank-corr vs landmark < 0.5) did NOT fire (rsf_longitudinal_summary
has 0.548, just above), but the m-s gap is too large to ignore.

Builds the submission with the strongest individual permissive member
(rsf_longitudinal_summary) replacing the 3-member rank-avg. Re-stacks the
blend weight against landmark via OOF grid search on the cached per-fold
OOF predictions.
"""
from __future__ import annotations
import sys, json, time, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import REPORTS, SUBMISSIONS, ROOT
from src.cv import cindex
from src.data import load_raw, build_targets
from src.features import (build_features, build_landmark_features,
                          at_risk_at_landmark)
from src.models import make_rsf, make_xgb_cox

CONFIGS = ROOT / "configs"
CACHE_PKL = REPORTS / "phase2_oof_perfold_cache.pkl"
LANDMARK = 3.0
MEMBER_ID = "rsf_longitudinal_summary"
WEIGHT_FINE = np.round(np.arange(0.0, 1.0001, 0.01), 4)
SP_CSV = SUBMISSIONS / "phase2_blend_2way_strongpermissive.csv"
SP_META = SUBMISSIONS / "phase2_blend_2way_strongpermissive.json"


def main():
    t0 = time.time()
    train, test = load_raw()
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e_full = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t_full = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df_full = train.loc[valid].reset_index(drop=True)
    keep_A = at_risk_at_landmark(e_full, t_full, LANDMARK)

    # Load cached per-fold OOF
    with open(CACHE_PKL, "rb") as fh:
        fold_records = pickle.load(fh)
    print(f"Loaded {len(fold_records)} fold records from {CACHE_PKL}",
          flush=True)

    # Re-stack weight: landmark vs rsf_longitudinal_summary
    member_key = f"risk_{MEMBER_ID}"

    def fold_cindex_at_w(w):
        cis = []
        for rec in fold_records:
            va = rec["val_idx"]
            blend = (w * rankdata(rec["risk_lm"])
                     + (1 - w) * rankdata(rec[member_key]))
            cis.append(cindex(e_full[va], t_full[va], blend))
        return np.asarray(cis)

    rows = []
    for w in WEIGHT_FINE:
        cis = fold_cindex_at_w(w)
        rows.append({"weight": float(w), "mean": float(cis.mean()),
                     "std": float(cis.std(ddof=1)),
                     "mean_minus_std": float(cis.mean() - cis.std(ddof=1))})
    curve = pd.DataFrame(rows)
    i_opt = int(curve["mean_minus_std"].idxmax())
    w_sp = float(curve.loc[i_opt, "weight"])
    ms_sp = float(curve.loc[i_opt, "mean_minus_std"])
    mean_sp = float(curve.loc[i_opt, "mean"])
    std_sp = float(curve.loc[i_opt, "std"])

    print(f"\n--- weight curve (landmark vs {MEMBER_ID}) ---", flush=True)
    print("Top 11 weights by m-s:", flush=True)
    print(curve.nlargest(11, "mean_minus_std").to_string(
        index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print(f"\nChosen: w_landmark={w_sp:.2f}, m-s={ms_sp:.4f} "
          f"(mean={mean_sp:.4f}, std={std_sp:.4f})", flush=True)

    # ---- Train on full data, predict test ----
    print("\n--- training on full data, predicting test ---", flush=True)
    rsf_params = json.loads(
        (CONFIGS / "optuna_rsf_baseline_v1.json").read_text())["best_params"]

    # Landmark
    df_lm = df_full.loc[keep_A].reset_index(drop=True)
    X_lm_tr = build_landmark_features(df_lm, LANDMARK)
    X_lm_te = build_landmark_features(test, LANDMARK)
    common_lm = [c for c in X_lm_tr.columns if c in X_lm_te.columns]
    risk_lm_test = make_rsf(**rsf_params)(
        X_lm_tr[common_lm], e_full[keep_A], t_full[keep_A], X_lm_te[common_lm])
    print(f"  landmark: range [{risk_lm_test.min():.3f}, "
          f"{risk_lm_test.max():.3f}]", flush=True)

    # rsf_longitudinal_summary
    rec = json.loads((CONFIGS / f"optuna_{MEMBER_ID}.json").read_text())
    member_params = rec["best_params"]
    X_tr = build_features(df_full, rec["feature_set"])
    X_te = build_features(test, rec["feature_set"])
    common_m = [c for c in X_tr.columns if c in X_te.columns]
    risk_member_test = make_rsf(**member_params)(
        X_tr[common_m], e_full, t_full, X_te[common_m])
    print(f"  {MEMBER_ID}: range [{risk_member_test.min():.3f}, "
          f"{risk_member_test.max():.3f}]", flush=True)

    blend = (w_sp * rankdata(risk_lm_test)
             + (1 - w_sp) * rankdata(risk_member_test))

    # Death (same as previous submissions)
    _, d_t = build_targets(train, "censor_missing_death_at_last")
    valid_d = d_t["death_valid"].to_numpy()
    e_d = d_t.loc[valid_d, "death_event"].to_numpy().astype(bool)
    t_d = d_t.loc[valid_d, "death_time"].to_numpy().astype(float)
    df_d = train.loc[valid_d].reset_index(drop=True)
    fs_d = "longitudinal_summary"
    X_d_tr = build_features(df_d, fs_d)
    X_d_te = build_features(test, fs_d)
    common_d = [c for c in X_d_tr.columns if c in X_d_te.columns]
    risk_death = make_xgb_cox(n_estimators=300, learning_rate=0.05)(
        X_d_tr[common_d], e_d, t_d, X_d_te[common_d])

    sub = pd.DataFrame({
        "trustii_id": test["trustii_id"].values,
        "risk_hepatic_event": blend,
        "risk_death": risk_death,
    })
    sub.to_csv(SP_CSV, index=False)

    meta = {
        "submission_kind":
            "2-way blend, strong-permissive replacement (single member, no rank-avg)",
        "rationale": (
            "Task 2 audit: rsf_longitudinal_summary alone has 5×10 m-s "
            "0.7356, beating the 3-member permissive_ensemble_avg's 0.7197 "
            "by +0.016 — same magnitude as the re-weighting LB win. The two "
            "XGB-cox members are nearly identical (OOF rank-corr 0.97) and "
            "individually weaker (~0.69 m-s); the 3-way average dilutes the "
            "RSF's signal. Strict criterion (test corr vs landmark < 0.5) "
            "did NOT fire (rsf_longitudinal_summary has 0.548, just above "
            "the bar) but the m-s gap is too large to ignore — building "
            "this as a relaxed-criterion candidate."
        ),
        "permissive_member_id": MEMBER_ID,
        "permissive_member_params": member_params,
        "permissive_member_5x10_ms": 0.7356,
        "permissive_ensemble_avg_5x10_ms": 0.7197,
        "permissive_member_test_corr_vs_landmark": 0.548,
        "weight_landmark": w_sp, "weight_member": float(1 - w_sp),
        "blend_5x10_cv": {"mean": mean_sp, "std": std_sp,
                          "mean_minus_std": ms_sp},
        "vs_phase2_blend_2way_optimal_LB_0.8965": {
            "permissive_replaced_with": MEMBER_ID,
            "weight_change": f"0.30/0.70 → {w_sp:.2f}/{1-w_sp:.2f}",
        },
        "death": {"feature_set": fs_d, "model": "xgb_cox",
                  "params": {"n_estimators": 300, "learning_rate": 0.05}},
    }
    SP_META.write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nWrote {SP_CSV}\nWrote {SP_META}", flush=True)
    print(f"Elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
