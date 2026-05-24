"""Phase 3 Experiment 2 — OOF cross-feature stacking.

Adds the OOF predicted risk for one endpoint as a feature for the other
endpoint's model. Cheap multi-task substitute that exploits the biological
correlation between hepatic decompensation and death.

Pipeline (one round, optional iteration):
  1. Generate baseline OOF predictions:
       oof_hep   = blend_2way_optimal OOF (30·rank(landmark)+70·rank(perm))
                   stitched from the cached per-fold landmark / permissive
                   predictions in reports/phase2_oof_perfold_cache.pkl.
       oof_death = XGB-Cox/longitudinal_summary/censor_missing_death_at_last
                   over 5×10 CV stratified on death event (death cohort,
                   n=1253).

  2. Leakage check: print 5 random patients with their (rep, fold) val
     assignments for both endpoints — confirms each OOF prediction comes
     from a model trained on a partition that excluded that row.

  3. Round-1 CV bake-off (5×10, identical fold protocol to baseline):
       a. landmark_3y_RSF on filter-A train, full val
            base   = baseline_v1 + landmark_locf/slope features
            aug    = base + oof_death column
       b. XGB-Cox on longitudinal_summary, full hep cohort
            base   = longitudinal_summary
            aug    = base + oof_death column
       c. XGB-Cox on longitudinal_summary, death cohort
            base   = longitudinal_summary
            aug    = base + oof_hep_blend column

     Compare m-s base vs aug for each.

  4. Round-2 (optional, one iteration only): if hepatic m-s improves by
     ≥ 0.005 in round 1, regenerate cross-features from the round-1
     augmented OOFs and re-run CV. Compare round 2 to round 1.

  5. Submission: if best round shows hepatic m-s improvement ≥ 0.005,
     build phase3_blend_with_crossfeatures.csv with the augmented hepatic
     component plugged into the 30/70 blend. Test-time cross-features
     come from FULL-train (no OOF) models, never from train OOF.

Outputs:
  reports/phase3_exp2_crossfeatures.{json,md}
  submissions/phase3_blend_with_crossfeatures.{csv,json}  (conditional)
"""
from __future__ import annotations
import sys, json, time, pickle, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from src.config import REPORTS, SUBMISSIONS, ROOT
from src.cv import repeated_stratified_folds, cindex
from src.data import load_raw, build_targets
from src.features import (build_features, build_landmark_features,
                          at_risk_at_landmark)
from src.models import make_rsf, make_xgb_cox

CONFIGS = ROOT / "configs"
N_SPLITS, N_REPEATS, BASE_SEED = 5, 10, 42
LANDMARK = 3.0
PERMISSIVE_MEMBERS = [
    "rsf_longitudinal_summary",
    "xgb_cox_longitudinal_plus_meta",
    "xgb_cox_longitudinal_summary",
]
BLEND_W_LANDMARK = 0.30
DEATH_FS = "longitudinal_summary"
DEATH_PARAMS = {"n_estimators": 300, "learning_rate": 0.05}
IMPROVEMENT_THRESHOLD = 0.005

OUT_JSON = REPORTS / "phase3_exp2_crossfeatures.json"
OUT_MD = REPORTS / "phase3_exp2_crossfeatures.md"
SUB_CSV = SUBMISSIONS / "phase3_blend_with_crossfeatures.csv"
SUB_META = SUBMISSIONS / "phase3_blend_with_crossfeatures.json"
HEP_CACHE_PKL = REPORTS / "phase2_oof_perfold_cache.pkl"


def load_optuna_params(cid):
    return json.loads((CONFIGS / f"optuna_{cid}.json").read_text())["best_params"]


def hepatic_targets_with_idx(train):
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    idx = np.where(valid)[0]
    e = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df = train.iloc[idx].reset_index(drop=True)
    return df, e, t, idx


def death_targets_with_idx(train):
    _, d_t = build_targets(train, "censor_missing_death_at_last")
    valid = d_t["death_valid"].to_numpy()
    idx = np.where(valid)[0]
    e = d_t.loc[valid, "death_event"].to_numpy().astype(bool)
    t = d_t.loc[valid, "death_time"].to_numpy().astype(float)
    df = train.iloc[idx].reset_index(drop=True)
    return df, e, t, idx


def stitch_perfold(perfold, n, key="risk", reduce="rank_avg"):
    """Stitch per-fold predictions to per-row OOF.
       reduce='rank_avg': rank-stitched mean (preserves rank invariance across folds)
       reduce='mean': raw average (use only when fold predictions are on consistent scale).
    """
    sumv = np.zeros(n)
    cnt = np.zeros(n, dtype=int)
    for rec in perfold:
        va = rec["val_idx"]
        r = rec[key] if isinstance(key, str) else key(rec)
        if reduce == "rank_avg":
            r = rankdata(r)
        sumv[va] += r
        cnt[va] += 1
    out = np.full(n, np.nan)
    out[cnt > 0] = sumv[cnt > 0] / cnt[cnt > 0]
    return out


def perm_avg_in_fold(rec):
    """Per-fold permissive ensemble = rank-avg of 3 members."""
    return np.mean([rankdata(rec[f"risk_{cid}"])
                    for cid in PERMISSIVE_MEMBERS], axis=0)


def blend_in_fold(rec):
    """Per-fold blend_2way_optimal = 0.3·rank(lm) + 0.7·rank(perm_avg)."""
    return (BLEND_W_LANDMARK * rankdata(rec["risk_lm"])
            + (1 - BLEND_W_LANDMARK) * perm_avg_in_fold(rec))


def cv_aug(factory_fn, X_df, e, t, *, extra=None, subfilter=None):
    """5×10 CV with optional extra column and optional train-side subfilter
       (e.g. filter-A for landmark). Returns metrics + per-fold predictions.
       factory_fn() must return a fresh fit_predict closure each call (so
       per-fold model state is not shared)."""
    X = X_df.copy()
    if extra is not None:
        X["__cross_feat"] = extra
    cis = []
    perfold = []
    for r, f, tr_idx, va_idx in repeated_stratified_folds(
            e, N_SPLITS, N_REPEATS, BASE_SEED):
        if subfilter is not None:
            tr_use = tr_idx[subfilter[tr_idx]]
        else:
            tr_use = tr_idx
        risk = factory_fn()(
            X.iloc[tr_use], e[tr_use], t[tr_use], X.iloc[va_idx])
        ci = cindex(e[va_idx], t[va_idx], risk)
        cis.append(ci)
        perfold.append({"repeat": r, "fold": f,
                        "val_idx": np.asarray(va_idx),
                        "risk": np.asarray(risk)})
    cis = np.asarray(cis)
    return {
        "mean": float(cis.mean()), "std": float(cis.std(ddof=1)),
        "mean_minus_std": float(cis.mean() - cis.std(ddof=1)),
        "perfold": perfold,
    }


def fold_assignments_for_endpoint(e, n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                  seed=BASE_SEED):
    """For each row, list (repeat, fold) where it was in val, plus
       confirms (rep, fold) train_idx never contains that row."""
    assigns = {i: [] for i in range(len(e))}
    for r, f, tr_idx, va_idx in repeated_stratified_folds(
            e, n_splits, n_repeats, seed):
        for v in va_idx:
            assigns[v].append((r, f))
    return assigns


def main():
    t_overall = time.time()
    train, test = load_raw()
    print(f"train {train.shape}, test {test.shape}", flush=True)

    df_h, e_h, t_h, hep_idx = hepatic_targets_with_idx(train)
    df_d, e_d, t_d, death_idx = death_targets_with_idx(train)
    n_h, n_d = len(e_h), len(e_d)
    print(f"hepatic cohort: n={n_h} events={int(e_h.sum())} "
          f"(drop_missing_death)", flush=True)
    print(f"death   cohort: n={n_d} events={int(e_d.sum())} "
          f"(censor_missing_death_at_last)", flush=True)
    keep_A = at_risk_at_landmark(e_h, t_h, LANDMARK)
    print(f"filter A (event-free at {LANDMARK}y in hep cohort): "
          f"n={int(keep_A.sum())}", flush=True)

    # ----------------------------------------------------------------------
    # 1) Generate baseline OOF predictions
    # ----------------------------------------------------------------------
    print("\n=== STEP 1: baseline OOF predictions ===", flush=True)

    # Hepatic blend OOF — use cached per-fold predictions, build blend per
    # fold, then stitch (averaged across the 10 repeats).
    if not HEP_CACHE_PKL.exists():
        raise SystemExit(f"Missing {HEP_CACHE_PKL}; run phase2_audit_v2.py first.")
    with open(HEP_CACHE_PKL, "rb") as fh:
        hep_cache = pickle.load(fh)
    print(f"  loaded {len(hep_cache)} hepatic per-fold records "
          f"from {HEP_CACHE_PKL.name}", flush=True)

    oof_hep_blend = stitch_perfold(hep_cache, n_h, key=blend_in_fold,
                                   reduce="mean")
    n_oof_hep = int(np.sum(~np.isnan(oof_hep_blend)))
    print(f"  oof_hep_blend stitched: {n_oof_hep}/{n_h} rows have OOF",
          flush=True)

    # Death OOF — fresh CV
    print("\n  generating death OOF (XGB-Cox / longitudinal_summary, "
          f"5×{N_REPEATS}, censor mode) ...", flush=True)
    X_d = build_features(df_d, DEATH_FS)
    print(f"  death feature matrix: {X_d.shape}", flush=True)
    death_factory = lambda: make_xgb_cox(**DEATH_PARAMS)
    t0 = time.time()
    death_base = cv_aug(death_factory, X_d, e_d, t_d)
    print(f"  death base CV: mean={death_base['mean']:.4f} "
          f"std={death_base['std']:.4f} m-s={death_base['mean_minus_std']:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    oof_death = stitch_perfold(death_base["perfold"], n_d, key="risk",
                               reduce="mean")

    # Cross-feature alignment between cohorts
    death_pos_for_hep = hep_idx
    oof_death_for_hep = oof_death[death_pos_for_hep]

    hep_to_pos = {orig: i for i, orig in enumerate(hep_idx)}
    oof_hep_for_death = np.full(n_d, np.nan)
    for i, orig in enumerate(death_idx):
        if orig in hep_to_pos:
            oof_hep_for_death[i] = oof_hep_blend[hep_to_pos[orig]]
    n_hep_for_death = int(np.sum(~np.isnan(oof_hep_for_death)))
    print(f"  oof_hep_for_death: {n_hep_for_death}/{n_d} rows have OOF "
          f"(NaN where death-cohort patient is not in hep cohort)",
          flush=True)

    # ----------------------------------------------------------------------
    # 2) Leakage check — 5 random patients × fold assignments
    # ----------------------------------------------------------------------
    print("\n=== STEP 2: leakage check ===", flush=True)
    rng = np.random.default_rng(0)
    h_assigns = fold_assignments_for_endpoint(e_h)
    d_assigns = fold_assignments_for_endpoint(e_d)

    sample_h = rng.choice(n_h, 5, replace=False)
    sample_d = rng.choice(n_d, 5, replace=False)

    print(f"\n  Hepatic CV (n={n_h}, stratified on hepatic event):",
          flush=True)
    leak_check_rows = []
    for ix in sample_h:
        pid = str(train.iloc[hep_idx[ix]]["patient_id_anon"])
        folds = h_assigns[ix]
        n_in_val = len(folds)
        # Confirm: for each (r,f) where this row was in val, the corresponding
        # train fold did NOT include this row. (Trivially true for k-fold,
        # but verify explicitly.)
        ok = True
        for r, f, tr_idx, va_idx in repeated_stratified_folds(
                e_h, N_SPLITS, N_REPEATS, BASE_SEED):
            if (r, f) in folds:
                if ix in tr_idx:
                    ok = False
        print(f"    patient_id={pid:>7} hep-row={ix:>4}  "
              f"in-val for {n_in_val:>2} (rep,fold) pairs: "
              f"{folds[:3]}... train-isolation={'OK' if ok else 'FAIL'}",
              flush=True)
        leak_check_rows.append({
            "endpoint": "hepatic", "patient_id": pid,
            "row_in_cohort": int(ix),
            "n_val_appearances": n_in_val,
            "first_three_val_folds": folds[:3],
            "train_isolation": ok,
        })

    print(f"\n  Death CV (n={n_d}, stratified on death event):", flush=True)
    for ix in sample_d:
        pid = str(train.iloc[death_idx[ix]]["patient_id_anon"])
        folds = d_assigns[ix]
        n_in_val = len(folds)
        ok = True
        for r, f, tr_idx, va_idx in repeated_stratified_folds(
                e_d, N_SPLITS, N_REPEATS, BASE_SEED):
            if (r, f) in folds:
                if ix in tr_idx:
                    ok = False
        print(f"    patient_id={pid:>7} death-row={ix:>4}  "
              f"in-val for {n_in_val:>2} (rep,fold) pairs: "
              f"{folds[:3]}... train-isolation={'OK' if ok else 'FAIL'}",
              flush=True)
        leak_check_rows.append({
            "endpoint": "death", "patient_id": pid,
            "row_in_cohort": int(ix),
            "n_val_appearances": n_in_val,
            "first_three_val_folds": folds[:3],
            "train_isolation": ok,
        })

    leak_pass = all(r["train_isolation"] and r["n_val_appearances"] == N_REPEATS
                    for r in leak_check_rows)
    leak_msg = ("PASS — every sampled row appears in val exactly "
                "N_REPEATS=10 times and never in its own train fold"
                if leak_pass else "FAIL")
    print(f"\n  Leakage check: {leak_msg}", flush=True)

    # ----------------------------------------------------------------------
    # 3) Round-1 CV bake-off
    # ----------------------------------------------------------------------
    print("\n=== STEP 3: round 1 CV bake-off ===", flush=True)

    rsf_lm_params = load_optuna_params("rsf_baseline_v1")
    xgb_lng_params = load_optuna_params("xgb_cox_longitudinal_summary")
    print(f"  landmark RSF params: {rsf_lm_params}", flush=True)
    print(f"  XGB-Cox/longitudinal_summary params: {xgb_lng_params}",
          flush=True)

    # Build features once
    X_lm_h = build_landmark_features(df_h, LANDMARK)
    X_lng_h = build_features(df_h, DEATH_FS)  # longitudinal_summary on hep cohort

    print("\n  ---- (a) landmark_3y_RSF on filter-A ----", flush=True)
    lm_factory = lambda: make_rsf(**rsf_lm_params)
    t0 = time.time()
    lm_base = cv_aug(lm_factory, X_lm_h, e_h, t_h, subfilter=keep_A)
    print(f"    base  : m-s={lm_base['mean_minus_std']:.4f} "
          f"(mean={lm_base['mean']:.4f}, std={lm_base['std']:.4f}) "
          f"({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    lm_aug = cv_aug(lm_factory, X_lm_h, e_h, t_h,
                    extra=oof_death_for_hep, subfilter=keep_A)
    print(f"    aug   : m-s={lm_aug['mean_minus_std']:.4f} "
          f"(mean={lm_aug['mean']:.4f}, std={lm_aug['std']:.4f}) "
          f"({time.time()-t0:.0f}s)", flush=True)
    lm_delta = lm_aug["mean_minus_std"] - lm_base["mean_minus_std"]
    print(f"    Δ m-s = {lm_delta:+.4f}", flush=True)

    print("\n  ---- (b) XGB-Cox / longitudinal_summary on hep cohort ----",
          flush=True)
    xgb_h_factory = lambda: make_xgb_cox(**xgb_lng_params)
    t0 = time.time()
    xgb_h_base = cv_aug(xgb_h_factory, X_lng_h, e_h, t_h)
    print(f"    base  : m-s={xgb_h_base['mean_minus_std']:.4f} "
          f"(mean={xgb_h_base['mean']:.4f}, std={xgb_h_base['std']:.4f}) "
          f"({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    xgb_h_aug = cv_aug(xgb_h_factory, X_lng_h, e_h, t_h,
                       extra=oof_death_for_hep)
    print(f"    aug   : m-s={xgb_h_aug['mean_minus_std']:.4f} "
          f"(mean={xgb_h_aug['mean']:.4f}, std={xgb_h_aug['std']:.4f}) "
          f"({time.time()-t0:.0f}s)", flush=True)
    xgb_h_delta = xgb_h_aug["mean_minus_std"] - xgb_h_base["mean_minus_std"]
    print(f"    Δ m-s = {xgb_h_delta:+.4f}", flush=True)

    print("\n  ---- (c) XGB-Cox / longitudinal_summary on death cohort ----",
          flush=True)
    death_aug = cv_aug(death_factory, X_d, e_d, t_d, extra=oof_hep_for_death)
    print(f"    base  : m-s={death_base['mean_minus_std']:.4f} "
          f"(from step 1)", flush=True)
    print(f"    aug   : m-s={death_aug['mean_minus_std']:.4f} "
          f"(mean={death_aug['mean']:.4f}, std={death_aug['std']:.4f})",
          flush=True)
    death_delta = death_aug["mean_minus_std"] - death_base["mean_minus_std"]
    print(f"    Δ m-s = {death_delta:+.4f}", flush=True)

    # Best hepatic improvement
    best_hep_delta = max(lm_delta, xgb_h_delta)
    print(f"\n  Best hepatic Δ m-s = {best_hep_delta:+.4f} "
          f"(threshold for action: ≥ {IMPROVEMENT_THRESHOLD})", flush=True)

    # OOF rank correlation between original and augmented hepatic predictions
    oof_lm_base   = stitch_perfold(lm_base["perfold"],   n_h)
    oof_lm_aug    = stitch_perfold(lm_aug["perfold"],    n_h)
    oof_xgb_base  = stitch_perfold(xgb_h_base["perfold"], n_h)
    oof_xgb_aug   = stitch_perfold(xgb_h_aug["perfold"],  n_h)
    def safe_spearman(a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 100: return float("nan")
        return float(spearmanr(a[mask], b[mask]).statistic)
    rho_lm = safe_spearman(oof_lm_base, oof_lm_aug)
    rho_xgb = safe_spearman(oof_xgb_base, oof_xgb_aug)
    print(f"  OOF rank-corr (base vs aug):  landmark {rho_lm:.3f}  "
          f"xgb-hep {rho_xgb:.3f}", flush=True)

    # ----------------------------------------------------------------------
    # 4) Round 2 (conditional)
    # ----------------------------------------------------------------------
    round2 = {"ran": False}
    if best_hep_delta >= IMPROVEMENT_THRESHOLD:
        print(f"\n=== STEP 4: round 2 (round 1 helped) ===", flush=True)
        # Build round-1 augmented OOFs as the new cross-features
        oof_hep_aug = stitch_perfold(lm_aug["perfold"], n_h)
        oof_death_aug = stitch_perfold(death_aug["perfold"], n_d)

        oof_death_for_hep_2 = oof_death_aug[hep_idx]
        oof_hep_for_death_2 = np.full(n_d, np.nan)
        for i, orig in enumerate(death_idx):
            if orig in hep_to_pos:
                oof_hep_for_death_2[i] = oof_hep_aug[hep_to_pos[orig]]

        print("\n  ---- (a) landmark with round-2 oof_death ----", flush=True)
        lm_aug2 = cv_aug(lm_factory, X_lm_h, e_h, t_h,
                         extra=oof_death_for_hep_2, subfilter=keep_A)
        print(f"    m-s={lm_aug2['mean_minus_std']:.4f}  "
              f"Δ vs round-1 aug = "
              f"{lm_aug2['mean_minus_std']-lm_aug['mean_minus_std']:+.4f}",
              flush=True)

        print("  ---- (b) xgb-hep with round-2 oof_death ----", flush=True)
        xgb_h_aug2 = cv_aug(xgb_h_factory, X_lng_h, e_h, t_h,
                            extra=oof_death_for_hep_2)
        print(f"    m-s={xgb_h_aug2['mean_minus_std']:.4f}  "
              f"Δ vs round-1 aug = "
              f"{xgb_h_aug2['mean_minus_std']-xgb_h_aug['mean_minus_std']:+.4f}",
              flush=True)

        print("  ---- (c) death with round-2 oof_hep ----", flush=True)
        death_aug2 = cv_aug(death_factory, X_d, e_d, t_d,
                            extra=oof_hep_for_death_2)
        print(f"    m-s={death_aug2['mean_minus_std']:.4f}  "
              f"Δ vs round-1 aug = "
              f"{death_aug2['mean_minus_std']-death_aug['mean_minus_std']:+.4f}",
              flush=True)

        round2 = {"ran": True,
                  "lm_aug2_ms": lm_aug2["mean_minus_std"],
                  "xgb_h_aug2_ms": xgb_h_aug2["mean_minus_std"],
                  "death_aug2_ms": death_aug2["mean_minus_std"],
                  "lm_delta_vs_round1": lm_aug2["mean_minus_std"]-lm_aug["mean_minus_std"],
                  "xgb_h_delta_vs_round1": xgb_h_aug2["mean_minus_std"]-xgb_h_aug["mean_minus_std"],
                  "death_delta_vs_round1": death_aug2["mean_minus_std"]-death_aug["mean_minus_std"]}
    else:
        print("\n=== STEP 4: skipped (round 1 below threshold) ===", flush=True)

    # ----------------------------------------------------------------------
    # 5) Submission build (conditional)
    # ----------------------------------------------------------------------
    will_build = best_hep_delta >= IMPROVEMENT_THRESHOLD
    print(f"\n=== STEP 5: submission build? {will_build} ===", flush=True)

    sub_meta = None
    if will_build:
        # Decide which augmented model(s) to ship.
        # If landmark improved the most, augment landmark in the blend.
        # If xgb_h improved more, augment that single permissive member
        # (and re-rank-avg with the other 2 unchanged).
        augment_landmark = lm_delta >= IMPROVEMENT_THRESHOLD
        augment_xgb_lng_perm = xgb_h_delta >= IMPROVEMENT_THRESHOLD
        print(f"  augment landmark: {augment_landmark}  "
              f"augment xgb-cox/long_summary in permissive: "
              f"{augment_xgb_lng_perm}", flush=True)

        # Test-time cross features: train models on FULL data, predict test
        print("\n  --- test-time cross-features (full-train models) ---",
              flush=True)
        X_d_te = build_features(test, DEATH_FS)
        common_d = [c for c in X_d.columns if c in X_d_te.columns]
        test_oof_death = make_xgb_cox(**DEATH_PARAMS)(
            X_d[common_d], e_d, t_d, X_d_te[common_d])
        print(f"    test_oof_death: range [{test_oof_death.min():.3f}, "
              f"{test_oof_death.max():.3f}]", flush=True)

        # Hepatic submission components
        # 1) Landmark (augmented if helped, base otherwise)
        keep_A_full = keep_A
        df_h_lmtr = df_h.loc[keep_A_full].reset_index(drop=True)
        e_h_lmtr = e_h[keep_A_full]
        t_h_lmtr = t_h[keep_A_full]
        X_lm_tr = X_lm_h.loc[keep_A_full].reset_index(drop=True)
        X_lm_te = build_landmark_features(test, LANDMARK)
        common_lm = [c for c in X_lm_tr.columns if c in X_lm_te.columns]
        X_lm_tr = X_lm_tr[common_lm].copy()
        X_lm_te = X_lm_te[common_lm].copy()
        if augment_landmark:
            X_lm_tr["__cross_feat"] = oof_death_for_hep[keep_A_full]
            X_lm_te["__cross_feat"] = test_oof_death
        risk_lm_test = make_rsf(**rsf_lm_params)(
            X_lm_tr, e_h_lmtr, t_h_lmtr, X_lm_te)
        print(f"    landmark test pred: range [{risk_lm_test.min():.3f}, "
              f"{risk_lm_test.max():.3f}]", flush=True)

        # 2) Permissive ensemble — 3 members
        rank_sum = np.zeros(len(test))
        perm_info = []
        for cid in PERMISSIVE_MEMBERS:
            rec = json.loads((CONFIGS / f"optuna_{cid}.json").read_text())
            params = rec["best_params"]
            X_tr_m = build_features(df_h, rec["feature_set"])
            X_te_m = build_features(test, rec["feature_set"])
            common = [c for c in X_tr_m.columns if c in X_te_m.columns]
            X_tr_m = X_tr_m[common].copy()
            X_te_m = X_te_m[common].copy()
            # Augment the xgb_cox_longitudinal_summary member (if helped)
            is_target = (cid == "xgb_cox_longitudinal_summary"
                         and augment_xgb_lng_perm)
            if is_target:
                X_tr_m["__cross_feat"] = oof_death_for_hep
                X_te_m["__cross_feat"] = test_oof_death
            if rec["model"] == "rsf":
                fac = make_rsf(**params)
            elif rec["model"] == "xgb_cox":
                fac = make_xgb_cox(**params)
            else:
                raise ValueError(rec["model"])
            risk_m = fac(X_tr_m, e_h, t_h, X_te_m)
            rank_sum += rankdata(risk_m)
            perm_info.append({"id": cid, "augmented": bool(is_target),
                              "model": rec["model"],
                              "feature_set": rec["feature_set"]})
        risk_perm_test = rank_sum / len(PERMISSIVE_MEMBERS)

        # Blend
        blend = (BLEND_W_LANDMARK * rankdata(risk_lm_test)
                 + (1 - BLEND_W_LANDMARK) * risk_perm_test)

        # Death (use the death model for the submission; with cross-feature
        # if the death-side aug helped)
        augment_death = death_delta >= IMPROVEMENT_THRESHOLD
        # Hepatic test predictions are needed as cross-feature for death.
        # We'll use the BLEND test prediction as the hepatic risk for death.
        # NOTE: this is fit-on-full-train, so no train-OOF leakage on test.
        if augment_death:
            test_oof_hep = blend  # blend trained on full hep cohort
            # However, we must train the death model with TRAIN OOF hep
            # (oof_hep_blend) and predict test with FULL-train hep blend
            # ranks. We DO have oof_hep_blend for train but it's only on
            # the hep cohort. For the 269 NaN-death-only rows, oof_hep is
            # NaN — XGB handles NaN.
            X_d_train_aug = X_d[common_d].copy()
            X_d_train_aug["__cross_feat"] = oof_hep_for_death
            X_d_test_aug = X_d_te[common_d].copy()
            X_d_test_aug["__cross_feat"] = rankdata(test_oof_hep)
            risk_death = make_xgb_cox(**DEATH_PARAMS)(
                X_d_train_aug, e_d, t_d, X_d_test_aug)
        else:
            risk_death = make_xgb_cox(**DEATH_PARAMS)(
                X_d[common_d], e_d, t_d, X_d_te[common_d])

        sub = pd.DataFrame({
            "trustii_id": test["trustii_id"].values,
            "risk_hepatic_event": blend,
            "risk_death": risk_death,
        })
        sub.to_csv(SUB_CSV, index=False)
        sub_meta = {
            "submission_kind": "phase3 cross-feature stacked blend",
            "augmented_components": {
                "landmark_3y_RSF": augment_landmark,
                "xgb_cox_longitudinal_summary_in_permissive": augment_xgb_lng_perm,
                "death_model": augment_death,
            },
            "blend_weights": {"landmark": BLEND_W_LANDMARK,
                              "permissive": 1 - BLEND_W_LANDMARK},
            "permissive_members": perm_info,
            "cv_diagnostic": {
                "lm_base_ms": lm_base["mean_minus_std"],
                "lm_aug_ms": lm_aug["mean_minus_std"],
                "lm_delta": lm_delta,
                "xgb_h_base_ms": xgb_h_base["mean_minus_std"],
                "xgb_h_aug_ms": xgb_h_aug["mean_minus_std"],
                "xgb_h_delta": xgb_h_delta,
                "death_base_ms": death_base["mean_minus_std"],
                "death_aug_ms": death_aug["mean_minus_std"],
                "death_delta": death_delta,
            },
            "improvement_threshold": IMPROVEMENT_THRESHOLD,
            "vs_phase2_blend_2way_optimal_LB_0.8965": (
                "augmented version of the same components; "
                "compare against 0.8965 anchor"),
        }
        SUB_META.write_text(json.dumps(sub_meta, indent=2, default=str))
        print(f"\n  Wrote {SUB_CSV}\n  Wrote {SUB_META}", flush=True)
    else:
        print("  (skipped — best hepatic Δ m-s below threshold)", flush=True)

    # ----------------------------------------------------------------------
    # 6) Final report
    # ----------------------------------------------------------------------
    diag = {
        "experiment": "Phase 3 Exp 2 — OOF cross-feature stacking",
        "cv_protocol": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS,
                        "base_seed": BASE_SEED},
        "cohorts": {"hep_n": n_h, "hep_events": int(e_h.sum()),
                    "death_n": n_d, "death_events": int(e_d.sum()),
                    "filter_A_n": int(keep_A.sum())},
        "leakage_check": {"sampled_rows": leak_check_rows,
                          "verdict": "PASS" if leak_pass else "FAIL"},
        "round_1": {
            "landmark":   {"base": lm_base["mean_minus_std"],
                           "aug":  lm_aug["mean_minus_std"],
                           "delta": lm_delta,
                           "rank_corr_base_aug": rho_lm},
            "xgb_h":      {"base": xgb_h_base["mean_minus_std"],
                           "aug":  xgb_h_aug["mean_minus_std"],
                           "delta": xgb_h_delta,
                           "rank_corr_base_aug": rho_xgb},
            "death":      {"base": death_base["mean_minus_std"],
                           "aug":  death_aug["mean_minus_std"],
                           "delta": death_delta},
            "best_hepatic_delta": best_hep_delta,
            "improvement_threshold": IMPROVEMENT_THRESHOLD,
        },
        "round_2": round2,
        "submission_built": will_build,
        "submission_meta": sub_meta,
    }
    OUT_JSON.write_text(json.dumps(diag, indent=2, default=str))

    # Markdown
    lines = [
        "# Phase 3 Experiment 2 — OOF cross-feature stacking\n",
        f"Date: 2026-04-28. CV: 5-fold × 10-repeat stratified on the "
        f"endpoint's event indicator (50 folds, base_seed={BASE_SEED}).\n",
        f"Hepatic cohort: n={n_h} (events {int(e_h.sum())}), "
        f"`drop_missing_death`. Death cohort: n={n_d} (events "
        f"{int(e_d.sum())}), `censor_missing_death_at_last`. "
        f"Filter A in hep cohort: n={int(keep_A.sum())}.\n",
        "## Leakage check\n",
        f"5 random hepatic rows + 5 random death rows; for each, lists the "
        f"(repeat, fold) where the row was in val. Each row appears in val "
        f"exactly N_REPEATS=10 times. Train-isolation verified per "
        f"(rep, fold).\n",
        f"**Verdict:** {'PASS' if leak_pass else 'FAIL'}.\n",
        "Sampled rows:",
        "| endpoint | patient_id | row | n_val | first 3 (rep,fold) | train_isolation |",
        "|---|---|---|---|---|---|",
    ]
    for r in leak_check_rows:
        lines.append(
            f"| {r['endpoint']} | {r['patient_id']} | {r['row_in_cohort']} | "
            f"{r['n_val_appearances']} | "
            f"{r['first_three_val_folds']} | "
            f"{'OK' if r['train_isolation'] else 'FAIL'} |")

    lines.append("\n## Round 1 — CV bake-off (base vs aug)\n")
    lines.append("| model | endpoint | base m-s | aug m-s | Δ |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| landmark_3y_RSF (filter A) | hepatic | "
                 f"{lm_base['mean_minus_std']:.4f} | "
                 f"{lm_aug['mean_minus_std']:.4f} | "
                 f"**{lm_delta:+.4f}** |")
    lines.append(f"| XGB-Cox/longitudinal_summary | hepatic | "
                 f"{xgb_h_base['mean_minus_std']:.4f} | "
                 f"{xgb_h_aug['mean_minus_std']:.4f} | "
                 f"**{xgb_h_delta:+.4f}** |")
    lines.append(f"| XGB-Cox/longitudinal_summary | death | "
                 f"{death_base['mean_minus_std']:.4f} | "
                 f"{death_aug['mean_minus_std']:.4f} | "
                 f"**{death_delta:+.4f}** |")
    lines.append(f"\nBest hepatic Δ m-s = **{best_hep_delta:+.4f}** "
                 f"(threshold for action: ≥ {IMPROVEMENT_THRESHOLD}).")
    lines.append(f"\nOOF rank-corr (base vs aug): "
                 f"landmark = {rho_lm:.3f}, xgb-hep = {rho_xgb:.3f}.")

    if round2["ran"]:
        lines.append("\n## Round 2 — using round-1 augmented OOFs as cross-features\n")
        lines.append("| model | round-1 aug m-s | round-2 m-s | Δ vs round-1 |")
        lines.append("|---|---|---|---|")
        lines.append(f"| landmark | {lm_aug['mean_minus_std']:.4f} | "
                     f"{round2['lm_aug2_ms']:.4f} | "
                     f"{round2['lm_delta_vs_round1']:+.4f} |")
        lines.append(f"| xgb-hep | {xgb_h_aug['mean_minus_std']:.4f} | "
                     f"{round2['xgb_h_aug2_ms']:.4f} | "
                     f"{round2['xgb_h_delta_vs_round1']:+.4f} |")
        lines.append(f"| death | {death_aug['mean_minus_std']:.4f} | "
                     f"{round2['death_aug2_ms']:.4f} | "
                     f"{round2['death_delta_vs_round1']:+.4f} |")
    else:
        lines.append("\n## Round 2 — skipped\n")
        lines.append("Round-1 hepatic Δ m-s below threshold; one-iteration cap respected.")

    lines.append("\n## Submission\n")
    if will_build:
        lines.append(f"Built `submissions/phase3_blend_with_crossfeatures.csv`. "
                     "30/70 blend with the augmented hepatic component(s) "
                     "plugged in. Test-time cross-features come from "
                     "full-train models (no train-OOF leakage on test).")
    else:
        lines.append("Not built. Best hepatic Δ m-s did not clear the "
                     f"{IMPROVEMENT_THRESHOLD} bar.")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}", flush=True)
    print(f"\nTotal elapsed: {(time.time()-t_overall)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
