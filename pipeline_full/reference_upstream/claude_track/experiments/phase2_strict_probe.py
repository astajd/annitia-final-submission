"""One-shot leaderboard probe: strict_time_aligned (LEAKY) for hepatic.

This is an AUDIT PROBE, not a ship candidate. Phase 1 proved strict_time_aligned
features are leaky: trajectory features computed over the per-row "before-event"
window encode the event time through Age_delta, _count, etc.

The probe asks: did the test set get generated with the same per-row truncation
artifact? Concretely, the train/test asymmetry is:

  Train cutoff: age_v1 + time, where time = (event_age - age_v1) for events,
                (max_age - age_v1) for censored. Event patients get short windows.
  Test cutoff:  age_v1 + (max_age - age_v1) = max_age. All visits kept, since
                we have no outcome labels.

If the test set was constructed by truncating event-patient visits the same way
train was, then test event patients also have small max_age. Coxnet trained on
"small Age_delta → high risk" then transfers, and LB ≈ CV (~0.94 hepatic). If
test is fully observed (no truncation), Age_delta is large for everyone and the
trick collapses (LB ~0.5-0.6).

Death endpoint uses the standard XGB-Cox + longitudinal_summary +
censor_missing_death_at_last to keep this submission focused on the hepatic
question.

No within-endpoint ensembling.
"""
from __future__ import annotations
import sys, time, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import SUBMISSIONS
from src.data import load_raw, build_targets, add_visit_metadata
from src.features import build_features
from src.models import make_coxnet, make_xgb_cox

OUT_CSV = SUBMISSIONS / "phase2_strict_leaky_probe.csv"
OUT_JSON = SUBMISSIONS / "phase2_strict_leaky_probe.json"


def quantiles(s: pd.Series) -> dict:
    s = s.dropna()
    return {
        "n_nonnull": int(len(s)),
        "median": float(s.median()) if len(s) else float("nan"),
        "p5": float(s.quantile(0.05)) if len(s) else float("nan"),
        "p95": float(s.quantile(0.95)) if len(s) else float("nan"),
        "min": float(s.min()) if len(s) else float("nan"),
        "max": float(s.max()) if len(s) else float("nan"),
    }


def main():
    t0 = time.time()
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    train, test = load_raw()
    print(f"Train: {train.shape}  Test: {test.shape}")

    # ---- HEPATIC: strict_time_aligned + Coxnet (THE LEAKY PROBE) ----
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e_train = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t_train = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    train_v = train.loc[valid].reset_index(drop=True)

    # Train: per-row outcome-defined cutoff (THE LEAKAGE)
    X_tr_hep = build_features(train_v, "strict_time_aligned",
                              event=e_train, time=t_train)

    # Test: synthetic time = max_age - Age_v1, so cutoff_age = max_age (all visits).
    test_meta = add_visit_metadata(test)
    synthetic_time_test = np.maximum(
        (test_meta["max_age"] - test_meta["Age_v1"]).to_numpy(dtype=float),
        0.001,
    )
    synthetic_event_test = np.zeros(len(test_meta), dtype=bool)
    X_te_hep = build_features(test_meta, "strict_time_aligned",
                              event=synthetic_event_test,
                              time=synthetic_time_test)

    common_hep = [c for c in X_tr_hep.columns if c in X_te_hep.columns]
    X_tr_hep = X_tr_hep[common_hep]
    X_te_hep = X_te_hep[common_hep]
    print(f"\nHepatic strict features: {len(common_hep)}  "
          f"(train n={int(valid.sum())}, events={int(e_train.sum())})")

    coxnet_factory = make_coxnet(l1_ratio=0.5)
    risk_hep = coxnet_factory(X_tr_hep, e_train, t_train, X_te_hep)
    print(f"Hepatic risk range: [{risk_hep.min():.3f}, {risk_hep.max():.3f}]")

    # ---- DEATH: longitudinal_summary + XGB-Cox + censor mode ----
    _, death_t = build_targets(train, "censor_missing_death_at_last")
    valid_d = death_t["death_valid"].to_numpy()
    e_d = death_t.loc[valid_d, "death_event"].to_numpy().astype(bool)
    t_d = death_t.loc[valid_d, "death_time"].to_numpy().astype(float)
    train_d = train.loc[valid_d].reset_index(drop=True)

    X_tr_death = build_features(train_d, "longitudinal_summary")
    X_te_death = build_features(test, "longitudinal_summary")
    common_d = [c for c in X_tr_death.columns if c in X_te_death.columns]
    X_tr_death = X_tr_death[common_d]
    X_te_death = X_te_death[common_d]
    print(f"\nDeath longitudinal_summary features: {len(common_d)}  "
          f"(train n={int(valid_d.sum())}, events={int(e_d.sum())})")

    xgb_factory = make_xgb_cox(n_estimators=300, learning_rate=0.05)
    risk_death = xgb_factory(X_tr_death, e_d, t_d, X_te_death)
    print(f"Death risk range: [{risk_death.min():.3f}, {risk_death.max():.3f}]")

    # ---- Distribution check: train (split by event) vs test ----
    print("\n=== TRAIN vs TEST: Age_delta / _count distributions ===")
    cmp_cols = ["Age_delta", "Age_count", "Age_rel_delta", "Age_std", "Age_last"]
    cmp_cols += [c for c in X_tr_hep.columns
                 if c.endswith("_count") and c != "Age_count"]
    dist = {}
    for col in cmp_cols:
        if col not in X_tr_hep.columns:
            continue
        tr_e = X_tr_hep.loc[e_train, col]
        tr_c = X_tr_hep.loc[~e_train, col]
        te = X_te_hep[col]
        dist[col] = {
            "train_event": quantiles(tr_e),
            "train_censored": quantiles(tr_c),
            "test": quantiles(te),
        }
        print(f"  {col:35s}  "
              f"tr_event med={dist[col]['train_event']['median']:.2f}  "
              f"tr_cens med={dist[col]['train_censored']['median']:.2f}  "
              f"test med={dist[col]['test']['median']:.2f}")

    # ---- Submission file ----
    sub = pd.DataFrame({
        "trustii_id": test["trustii_id"].values,
        "risk_hepatic_event": risk_hep,
        "risk_death": risk_death,
    })
    sub.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(sub)} rows)")

    # ---- Metadata ----
    meta = {
        "submission_kind": "AUDIT PROBE — NOT A SHIP CANDIDATE",
        "purpose": (
            "Test whether the test set has the same per-row outcome-dependent "
            "visit-cutoff structure as train. Phase 1 audit proved "
            "strict_time_aligned is leaky on train (Age_delta alone gives "
            "C ≈ 0.958 hepatic). If LB ≈ CV (~0.94), the leaderboard top "
            "score is exploitable through this trick; if LB collapses, the "
            "asymmetry doesn't transfer and the leader has a different trick."
        ),
        "hepatic_model": {
            "feature_set": "strict_time_aligned (LEAKY by Phase 1 audit)",
            "model": "coxnet",
            "params": {"l1_ratio": 0.5, "alpha_min_ratio": 0.01},
            "n_features": len(common_hep),
            "n_train_rows": int(valid.sum()),
            "n_train_events": int(e_train.sum()),
            "train_cutoff_definition": (
                "cutoff_age = age_v1 + time, where "
                "time = (event_age - age_v1) for events, "
                "(max_age - age_v1) for censored"
            ),
            "test_cutoff_definition": (
                "cutoff_age = age_v1 + (max_age - age_v1) = max_age "
                "(equivalent to using all available visits, since no "
                "outcome labels are visible at test time)"
            ),
            "asymmetry_note": (
                "Train events get short windows; train censored and all test "
                "rows get full windows. This asymmetry IS the leakage being "
                "probed."
            ),
            "cv_train_side_cindex": (
                "0.938 ± 0.052 (5-fold × 10-repeat, drop_missing_death) — "
                "see reports/phase2_cv_results.csv. Note: XGB-Cox on the same "
                "feature set gives 0.980 ± 0.014; Coxnet selected here for "
                "monotone-linear interpretability of the leakage signal."
            ),
        },
        "death_model": {
            "feature_set": "longitudinal_summary",
            "model": "xgb_cox",
            "params": {"n_estimators": 300, "learning_rate": 0.05},
            "death_mode": "censor_missing_death_at_last",
            "n_features": len(common_d),
            "n_train_rows": int(valid_d.sum()),
            "n_train_events": int(e_d.sum()),
            "cv_train_side_cindex": (
                "0.948 ± 0.018 (5-fold × 10-repeat) — "
                "see reports/phase2_cv_results.csv"
            ),
        },
        "ensemble": "single model per endpoint (no rank-averaging)",
        "interpretation_of_LB_result": {
            "hep_LB_~0.94": (
                "test was generated with the same train-side truncation; "
                "leader is exploiting this. Does not change our ship strategy "
                "(we do not ship leaky models), but answers the question."
            ),
            "hep_LB_~0.5_to_0.6": (
                "asymmetry doesn't transfer; leader has a different trick. "
                "Investigate longitudinal_plus_meta or fixed_reference_time."
            ),
            "hep_LB_~0.7_to_0.85": (
                "partial transfer; test set has some outcome-dependent "
                "truncation but not full. Ambiguous but suggests leader's "
                "trick is more elaborate than naive strict-features."
            ),
        },
        "feature_distribution_comparison": dist,
    }
    OUT_JSON.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_JSON}  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
