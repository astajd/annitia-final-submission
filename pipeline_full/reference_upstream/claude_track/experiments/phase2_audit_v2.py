"""Phase 2 audit v2 — Tasks 1+2+3 in one script.

Triggered by phase2_blend_2way_optimal LB 0.8965 (vs 50/50 0.88033, Δ +0.016).
The +0.0076 CV improvement amplified ~2.1× on LB; permissive component is
under-valued. Path forward: strengthen permissive, not chase finer weights.

Task 1 — Probe weight curve at finer granularity.
  Generate per-fold OOF for landmark + 3-member permissive ensemble, then
  evaluate m-s at 0.01-step weight grid. Decision (Task 3): if a finer-grid
  optimum w_new differs meaningfully from 0.30 (|w_new − 0.30| ≥ 0.05 OR
  m-s improves by ≥0.0005) → build phase2_blend_2way_optimal_v2.csv.

Task 2 — Strengthen the permissive component.
  Same OOF run also generates per-member predictions. Compute:
    a) Per-member hepatic 5×10 m-s.
    b) Pairwise OOF Spearman rank-corr (members + landmark, on filter-A).
    c) Test-set Spearman rank-corr of each member vs landmark_3y.
  Decision (Task 3): if one permissive member is the STRONGEST individual
  (highest m-s among members) AND its test rank-corr vs landmark < 0.5,
  replace the 3-member ensemble with that single model and re-stack the
  blend weight against landmark via OOF grid search; build
  phase2_blend_2way_strongpermissive.csv with (w_new, single_member).

Outputs:
  reports/phase2_audit_v2.{json,md}
  submissions/phase2_blend_2way_optimal_v2.{csv,json}      (conditional)
  submissions/phase2_blend_2way_strongpermissive.{csv,json}(conditional)
"""
from __future__ import annotations
import sys, json, time, warnings, pickle
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
WEIGHT_FINE = np.round(np.arange(0.0, 1.0001, 0.01), 4)
CURRENT_OPT_W = 0.30
CURRENT_OPT_MS = 0.7224

# Task 3 thresholds
V2_W_DELTA = 0.05      # |w_new - 0.30| must exceed this OR
V2_MS_DELTA = 0.0005   # m-s must improve by at least this
STRONG_CORR_THRESHOLD = 0.5  # member's test corr vs landmark must be below

OUT_JSON = REPORTS / "phase2_audit_v2.json"
OUT_MD = REPORTS / "phase2_audit_v2.md"
V2_CSV = SUBMISSIONS / "phase2_blend_2way_optimal_v2.csv"
V2_META = SUBMISSIONS / "phase2_blend_2way_optimal_v2.json"
SP_CSV = SUBMISSIONS / "phase2_blend_2way_strongpermissive.csv"
SP_META = SUBMISSIONS / "phase2_blend_2way_strongpermissive.json"


def make_factory_from_id(cid):
    rec = json.loads((CONFIGS / f"optuna_{cid}.json").read_text())
    p = dict(rec["best_params"])
    if rec["model"] == "rsf":     return make_rsf(**p), rec
    if rec["model"] == "xgb_cox": return make_xgb_cox(**p), rec
    raise ValueError(rec["model"])


def hepatic_targets(train):
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)
    return df, e, t


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
    return risk, {"feature_set": fs, "model": "xgb_cox",
                  "params": {"n_estimators": 300, "learning_rate": 0.05},
                  "death_mode": "censor_missing_death_at_last"}


def main():
    t_overall = time.time()
    train, test = load_raw()
    df_full, e_full, t_full = hepatic_targets(train)
    keep_A = at_risk_at_landmark(e_full, t_full, LANDMARK)
    n_full = len(e_full)
    print(f"Full hepatic-valid cohort: n={n_full}, events={int(e_full.sum())}",
          flush=True)
    print(f"Filter A: n={int(keep_A.sum())}, events={int(e_full[keep_A].sum())}",
          flush=True)

    # ---------- Pre-build features once ----------
    print("\n--- pre-building features ---", flush=True)
    rsf_params = json.loads(
        (CONFIGS / "optuna_rsf_baseline_v1.json").read_text())["best_params"]
    t0 = time.time()
    X_lm = build_landmark_features(df_full, LANDMARK)
    print(f"  landmark (full): {X_lm.shape[1]} cols ({time.time()-t0:.1f}s)",
          flush=True)
    perm_factories = []
    for cid in PERMISSIVE_MEMBERS:
        t0 = time.time()
        factory, rec = make_factory_from_id(cid)
        X = build_features(df_full, rec["feature_set"])
        perm_factories.append((cid, factory, X))
        print(f"  {cid:35s} fs={rec['feature_set']:30s} "
              f"{X.shape[1]} cols ({time.time()-t0:.1f}s)", flush=True)

    # ---------- Per-fold OOF ----------
    print(f"\n--- per-fold OOF ({N_SPLITS}×{N_REPEATS}={N_SPLITS*N_REPEATS} folds) ---",
          flush=True)
    fold_records = []
    for r, f, tr_idx, va_idx in repeated_stratified_folds(
            e_full, N_SPLITS, N_REPEATS, BASE_SEED):
        t0 = time.time()
        # Landmark: train filter-A, predict full val
        tr_keep_A = tr_idx[keep_A[tr_idx]]
        risk_lm = make_rsf(**rsf_params)(
            X_lm.iloc[tr_keep_A], e_full[tr_keep_A], t_full[tr_keep_A],
            X_lm.iloc[va_idx])
        # Permissive members
        member_risks = {}
        for cid, factory, X in perm_factories:
            r_m = factory(X.iloc[tr_idx], e_full[tr_idx], t_full[tr_idx],
                          X.iloc[va_idx])
            member_risks[cid] = np.asarray(r_m)
        rec = {"repeat": r, "fold": f, "val_idx": np.asarray(va_idx),
               "risk_lm": np.asarray(risk_lm)}
        for cid in PERMISSIVE_MEMBERS:
            rec[f"risk_{cid}"] = member_risks[cid]
        fold_records.append(rec)
        print(f"  fold {len(fold_records):2d}/{N_SPLITS*N_REPEATS} "
              f"rep={r} fold={f} ({time.time()-t0:.1f}s)", flush=True)

    print(f"OOF gen: {(time.time()-t_overall)/60:.1f} min", flush=True)

    # Cache fold_records
    with open(REPORTS / "phase2_oof_perfold_cache.pkl", "wb") as fh:
        pickle.dump(fold_records, fh)

    # ---------- Helper: per-fold C-index for a given risk function ----------
    def fold_cindex_curve(risk_fn):
        """risk_fn(rec) -> array of risk values for that fold's val_idx.
        Returns list of 50 C-index values."""
        out = []
        for rec in fold_records:
            va = rec["val_idx"]
            out.append(cindex(e_full[va], t_full[va], risk_fn(rec)))
        return np.asarray(out)

    def ms(arr):
        return float(arr.mean()), float(arr.std(ddof=1)), \
               float(arr.mean() - arr.std(ddof=1))

    # ---------- TASK 1: Fine-grained weight curve ----------
    print("\n=== TASK 1: fine-grained weight curve ===", flush=True)
    perm_avg_risk = lambda rec: np.mean(
        [rankdata(rec[f"risk_{cid}"]) for cid in PERMISSIVE_MEMBERS], axis=0)

    weight_curve = []
    for w in WEIGHT_FINE:
        cis = fold_cindex_curve(
            lambda rec, w=w: w * rankdata(rec["risk_lm"])
                              + (1 - w) * perm_avg_risk(rec))
        m, s, mms = ms(cis)
        weight_curve.append({"weight": float(w), "mean": m, "std": s,
                             "mean_minus_std": mms})

    wc_df = pd.DataFrame(weight_curve)
    i_opt = int(wc_df["mean_minus_std"].idxmax())
    w_new = float(wc_df.loc[i_opt, "weight"])
    ms_new = float(wc_df.loc[i_opt, "mean_minus_std"])
    mean_new = float(wc_df.loc[i_opt, "mean"])
    std_new = float(wc_df.loc[i_opt, "std"])

    # User-asked specific weights
    user_weights = [(0.10, 0.90), (0.20, 0.80), (0.40, 0.60)]
    user_table = []
    for w_lm, w_pm in user_weights:
        i = int(np.argmin(np.abs(wc_df["weight"].values - w_lm)))
        row = wc_df.iloc[i]
        user_table.append({
            "w_landmark": w_lm, "w_permissive": w_pm,
            "mean": float(row["mean"]), "std": float(row["std"]),
            "mean_minus_std": float(row["mean_minus_std"]),
        })

    print("\nUser-asked weights:")
    for r in user_table:
        print(f"  w_landmark={r['w_landmark']:.2f}, w_perm={r['w_permissive']:.2f}: "
              f"mean={r['mean']:.4f}, std={r['std']:.4f}, "
              f"m-s={r['mean_minus_std']:.4f}", flush=True)
    print(f"\nFine optimum (0.01 grid): w*={w_new:.2f}, m-s={ms_new:.4f} "
          f"(mean={mean_new:.4f}, std={std_new:.4f})", flush=True)
    print(f"Previous optimum (0.05 grid): w*=0.30, m-s={CURRENT_OPT_MS:.4f}",
          flush=True)
    print(f"Δ vs 0.30: {ms_new - CURRENT_OPT_MS:+.4f}, |Δw|={abs(w_new-0.30):.2f}",
          flush=True)

    # Top neighborhood for context
    print("\nTop 11 weights by m-s:", flush=True)
    print(wc_df.nlargest(11, "mean_minus_std").to_string(
        index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    # ---------- TASK 2: per-member audit ----------
    print("\n=== TASK 2: per-member audit ===", flush=True)

    # 2a. Per-member 5×10 m-s
    per_member_ms = {}
    for cid in PERMISSIVE_MEMBERS:
        cis = fold_cindex_curve(lambda rec, cid=cid: rec[f"risk_{cid}"])
        m, s, mms = ms(cis)
        per_member_ms[cid] = {"mean": m, "std": s, "mean_minus_std": mms}
    cis_perm_avg = fold_cindex_curve(perm_avg_risk)
    m, s, mms = ms(cis_perm_avg)
    per_member_ms["permissive_ensemble_avg"] = {
        "mean": m, "std": s, "mean_minus_std": mms}
    cis_lm = fold_cindex_curve(lambda rec: rec["risk_lm"])
    m, s, mms = ms(cis_lm)
    per_member_ms["landmark_3y_RSF"] = {
        "mean": m, "std": s, "mean_minus_std": mms}

    print("\nPer-member 5×10 hepatic m-s (full cohort, 50 folds):", flush=True)
    for k, v in per_member_ms.items():
        print(f"  {k:42s} mean={v['mean']:.4f}  std={v['std']:.4f}  "
              f"m-s={v['mean_minus_std']:.4f}", flush=True)

    # 2b. Pairwise OOF Spearman rank-corr (filter-A intersection)
    # Stitch repeat-averaged OOF for each model.
    def stitch_oof(key):
        sumv = np.zeros(n_full)
        cnt = np.zeros(n_full, dtype=int)
        for rec in fold_records:
            va = rec["val_idx"]
            sumv[va] += rec[key]
            cnt[va] += 1
        out = np.full(n_full, np.nan)
        nz = cnt > 0
        out[nz] = sumv[nz] / cnt[nz]
        return out

    oof = {"landmark_3y_RSF": stitch_oof("risk_lm")}
    for cid in PERMISSIVE_MEMBERS:
        oof[cid] = stitch_oof(f"risk_{cid}")
    # Permissive average (per-fold rank-avg, then stitched)
    sumv = np.zeros(n_full)
    cnt = np.zeros(n_full, dtype=int)
    for rec in fold_records:
        va = rec["val_idx"]
        sumv[va] += perm_avg_risk(rec)
        cnt[va] += 1
    out = np.full(n_full, np.nan)
    out[cnt > 0] = sumv[cnt > 0] / cnt[cnt > 0]
    oof["permissive_ensemble_avg"] = out

    keys = list(oof.keys())
    filter_A_idx = np.where(keep_A)[0]
    corr_oof = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for a in keys:
        for b in keys:
            va = oof[a][filter_A_idx]
            vb = oof[b][filter_A_idx]
            mask = ~(np.isnan(va) | np.isnan(vb))
            if mask.sum() < 50:
                corr_oof.loc[a, b] = np.nan
            else:
                rho, _ = spearmanr(va[mask], vb[mask])
                corr_oof.loc[a, b] = rho
    print("\nPairwise OOF rank-corr (filter-A intersection, "
          f"n={int(keep_A.sum())}):", flush=True)
    print(corr_oof.to_string(float_format=lambda x: f"{x:.3f}"), flush=True)

    # 2c. Test rank-corr vs landmark
    print("\n--- training each model on full data, predicting test ---",
          flush=True)
    df_lm_full = df_full.loc[keep_A].reset_index(drop=True)
    X_lm_tr = build_landmark_features(df_lm_full, LANDMARK)
    X_lm_te = build_landmark_features(test, LANDMARK)
    common = [c for c in X_lm_tr.columns if c in X_lm_te.columns]
    test_lm = make_rsf(**rsf_params)(
        X_lm_tr[common], e_full[keep_A], t_full[keep_A], X_lm_te[common])
    test_preds = {"landmark_3y_RSF": test_lm}

    for cid, factory, X in perm_factories:
        rec = json.loads((CONFIGS / f"optuna_{cid}.json").read_text())
        X_te = build_features(test, rec["feature_set"])
        common = [c for c in X.columns if c in X_te.columns]
        test_preds[cid] = factory(X[common], e_full, t_full, X_te[common])

    # Permissive ensemble average on test
    test_preds["permissive_ensemble_avg"] = np.mean(
        [rankdata(test_preds[cid]) for cid in PERMISSIVE_MEMBERS], axis=0)

    print("\nTest rank-corr vs landmark_3y_RSF (n={}):".format(len(test)),
          flush=True)
    test_corr_vs_lm = {}
    r_lm = rankdata(test_preds["landmark_3y_RSF"])
    for k, v in test_preds.items():
        if k == "landmark_3y_RSF": continue
        rho, _ = spearmanr(r_lm, rankdata(v))
        test_corr_vs_lm[k] = float(rho)
        print(f"  {k:42s}  ρ = {rho:.3f}", flush=True)

    # ---------- TASK 3: decisions ----------
    print("\n=== TASK 3: decisions ===", flush=True)

    # Decision A: build _v2?
    build_v2 = (abs(w_new - CURRENT_OPT_W) >= V2_W_DELTA) or \
               ((ms_new - CURRENT_OPT_MS) >= V2_MS_DELTA)
    print(f"\n_v2 build? w_new={w_new:.2f} (Δ {abs(w_new-CURRENT_OPT_W):.2f} "
          f"vs 0.30); m-s Δ {ms_new - CURRENT_OPT_MS:+.4f}.  "
          f"Trigger: |Δw|≥{V2_W_DELTA} or Δm-s≥{V2_MS_DELTA}.  "
          f"→ {'BUILD' if build_v2 else 'SKIP'}", flush=True)

    # Decision B: strongpermissive replacement?
    member_ms_only = {cid: per_member_ms[cid]["mean_minus_std"]
                      for cid in PERMISSIVE_MEMBERS}
    strongest_member = max(member_ms_only, key=member_ms_only.get)
    strongest_member_ms = member_ms_only[strongest_member]
    strongest_corr_lm = test_corr_vs_lm[strongest_member]
    print(f"\nStrongest individual permissive member: {strongest_member}",
          flush=True)
    print(f"  m-s = {strongest_member_ms:.4f}", flush=True)
    print(f"  test rank-corr vs landmark = {strongest_corr_lm:.3f}", flush=True)
    print(f"  permissive_ensemble_avg m-s = "
          f"{per_member_ms['permissive_ensemble_avg']['mean_minus_std']:.4f}",
          flush=True)
    build_sp = strongest_corr_lm < STRONG_CORR_THRESHOLD
    print(f"  → strongest member corr<{STRONG_CORR_THRESHOLD}? "
          f"{'YES — BUILD strongpermissive' if build_sp else 'NO — SKIP'}",
          flush=True)

    # ---------- Optional builds ----------
    diag = {
        "task1_weight_curve": weight_curve,
        "task1_user_weights": user_table,
        "task1_fine_optimum": {
            "w": w_new, "mean": mean_new, "std": std_new,
            "mean_minus_std": ms_new,
            "delta_vs_0p30_w": abs(w_new - CURRENT_OPT_W),
            "delta_vs_0p30_ms": ms_new - CURRENT_OPT_MS,
        },
        "task1_decision": {"build_v2": build_v2,
                           "thresholds": {"w_delta": V2_W_DELTA,
                                          "ms_delta": V2_MS_DELTA}},
        "task2_per_member_5x10_ms": per_member_ms,
        "task2_oof_corr_filterA":
            {a: {b: float(corr_oof.loc[a, b]) for b in keys} for a in keys},
        "task2_test_rank_corr_vs_landmark": test_corr_vs_lm,
        "task2_strongest_member": {
            "id": strongest_member,
            "mean_minus_std": strongest_member_ms,
            "test_corr_vs_landmark": strongest_corr_lm,
            "perm_avg_ms": per_member_ms["permissive_ensemble_avg"]["mean_minus_std"],
            "build_strongpermissive": build_sp,
            "threshold_corr_vs_landmark": STRONG_CORR_THRESHOLD,
        },
    }

    # Build _v2 if triggered
    if build_v2:
        print(f"\n--- building _v2 with w={w_new:.2f} ---", flush=True)
        blend = (w_new * rankdata(test_preds["landmark_3y_RSF"])
                 + (1 - w_new)
                 * rankdata(test_preds["permissive_ensemble_avg"]))
        risk_death, d_info = predict_death(train, test)
        sub = pd.DataFrame({"trustii_id": test["trustii_id"].values,
                            "risk_hepatic_event": blend,
                            "risk_death": risk_death})
        sub.to_csv(V2_CSV, index=False)
        meta = {
            "submission_kind": "2-way blend, finer-grid optimal weight",
            "weight_landmark": w_new, "weight_permissive": float(1 - w_new),
            "cv_diagnostic": diag["task1_fine_optimum"],
            "vs_phase2_blend_2way_optimal_LB_0.8965": {
                "weight_change": w_new - CURRENT_OPT_W,
                "ms_change": ms_new - CURRENT_OPT_MS,
            },
            "death": d_info,
        }
        V2_META.write_text(json.dumps(meta, indent=2, default=str))
        print(f"Wrote {V2_CSV}\nWrote {V2_META}", flush=True)

    # Build _strongpermissive if triggered
    if build_sp:
        print(f"\n--- building _strongpermissive with member="
              f"{strongest_member} ---", flush=True)
        # Re-stack: find best weight w in 0.01 grid for landmark vs single member
        sp_curve = []
        member_key = f"risk_{strongest_member}"
        for w in WEIGHT_FINE:
            cis = fold_cindex_curve(
                lambda rec, w=w: w * rankdata(rec["risk_lm"])
                                  + (1 - w) * rankdata(rec[member_key]))
            m, s, mms = ms(cis)
            sp_curve.append({"weight": float(w), "mean": m, "std": s,
                             "mean_minus_std": mms})
        sp_df = pd.DataFrame(sp_curve)
        i_sp = int(sp_df["mean_minus_std"].idxmax())
        w_sp = float(sp_df.loc[i_sp, "weight"])
        ms_sp = float(sp_df.loc[i_sp, "mean_minus_std"])
        mean_sp = float(sp_df.loc[i_sp, "mean"])
        std_sp = float(sp_df.loc[i_sp, "std"])
        print(f"  re-stacked: w_landmark={w_sp:.2f}  "
              f"m-s={ms_sp:.4f} (mean={mean_sp:.4f}, std={std_sp:.4f})",
              flush=True)
        diag["task3_strongpermissive_weight_curve"] = sp_curve
        diag["task3_strongpermissive_chosen"] = {
            "member": strongest_member, "w_landmark": w_sp,
            "mean": mean_sp, "std": std_sp, "mean_minus_std": ms_sp,
        }

        blend = (w_sp * rankdata(test_preds["landmark_3y_RSF"])
                 + (1 - w_sp) * rankdata(test_preds[strongest_member]))
        risk_death, d_info = predict_death(train, test)
        sub = pd.DataFrame({"trustii_id": test["trustii_id"].values,
                            "risk_hepatic_event": blend,
                            "risk_death": risk_death})
        sub.to_csv(SP_CSV, index=False)
        meta = {
            "submission_kind":
                "2-way blend with single permissive member (no avg)",
            "rationale": (f"{strongest_member} is the strongest individual "
                          f"permissive member by 5×10 m-s "
                          f"({strongest_member_ms:.4f}) AND has test "
                          f"rank-corr vs landmark "
                          f"({strongest_corr_lm:.3f}) below the "
                          f"{STRONG_CORR_THRESHOLD} diversity bar. Replaces "
                          "the 3-member rank-avg with this single model and "
                          "re-stacks the blend weight via OOF grid search."),
            "weight_landmark": w_sp, "weight_member": float(1 - w_sp),
            "permissive_member_id": strongest_member,
            "cv_diagnostic": {
                "mean": mean_sp, "std": std_sp, "mean_minus_std": ms_sp,
            },
            "vs_phase2_blend_2way_optimal_LB_0.8965": {
                "permissive_replaced_with": strongest_member,
                "weight": w_sp,
            },
            "death": d_info,
        }
        SP_META.write_text(json.dumps(meta, indent=2, default=str))
        print(f"Wrote {SP_CSV}\nWrote {SP_META}", flush=True)

    OUT_JSON.write_text(json.dumps(diag, indent=2, default=str))

    # ---------- markdown summary ----------
    lines = [
        "# Phase 2 audit v2 — Tasks 1+2+3 \n",
        f"Date: 2026-04-28. Triggered by phase2_blend_2way_optimal LB 0.8965 "
        f"(+0.016 over 50/50). All numbers below: 5×{N_REPEATS} stratified CV "
        f"(50 folds, base_seed={BASE_SEED}), full hepatic-valid cohort "
        f"(n={n_full}, events={int(e_full.sum())}).\n",
        "## Task 1 — finer-grain weight curve\n",
        "User-asked weights (CV m-s on hepatic):",
        "| w (landmark) | w (permissive) | mean | std | m-s |",
        "|---|---|---|---|---|",
    ]
    for r in user_table:
        lines.append(
            f"| {r['w_landmark']:.2f} | {r['w_permissive']:.2f} | "
            f"{r['mean']:.4f} | {r['std']:.4f} | "
            f"{r['mean_minus_std']:.4f} |")
    lines.append(f"\n**Fine-grid optimum (0.01 step):** w* = {w_new:.2f}, "
                 f"m-s = {ms_new:.4f} (mean {mean_new:.4f} ± {std_new:.4f}).")
    lines.append(f"\n**Vs the 0.30 anchor that scored LB 0.8965:** "
                 f"|Δw| = {abs(w_new - CURRENT_OPT_W):.2f}, "
                 f"Δm-s = {ms_new - CURRENT_OPT_MS:+.4f}.")
    lines.append(f"\n→ {'BUILD' if build_v2 else 'SKIP'} _v2 "
                 f"(triggers: |Δw|≥{V2_W_DELTA} or Δm-s≥{V2_MS_DELTA}).")

    lines.append("\n## Task 2 — permissive component audit\n")
    lines.append("**Per-member 5×10 hepatic m-s:**")
    lines.append("| model | mean | std | m-s |")
    lines.append("|---|---|---|---|")
    for k, v in per_member_ms.items():
        lines.append(f"| `{k}` | {v['mean']:.4f} | {v['std']:.4f} | "
                     f"{v['mean_minus_std']:.4f} |")

    lines.append("\n**Pairwise OOF rank-corr (filter-A intersection, "
                 f"n={int(keep_A.sum())}):**\n")
    lines.append("| | " + " | ".join(f"`{k}`" for k in keys) + " |")
    lines.append("|" + "|".join(["---"] * (len(keys) + 1)) + "|")
    for a in keys:
        row = "| `" + a + "` | " + " | ".join(
            f"{corr_oof.loc[a, b]:.3f}" if not pd.isna(corr_oof.loc[a, b])
            else "n/a" for b in keys) + " |"
        lines.append(row)

    lines.append("\n**Test rank-corr vs landmark_3y_RSF (n={}):**".format(len(test)))
    for k, v in test_corr_vs_lm.items():
        lines.append(f"- `{k}`: ρ = {v:.3f}")

    lines.append(f"\n**Strongest individual permissive member**: "
                 f"`{strongest_member}` "
                 f"(m-s {strongest_member_ms:.4f}, test corr vs landmark "
                 f"{strongest_corr_lm:.3f}).")
    lines.append(f"\n→ {'BUILD' if build_sp else 'SKIP'} _strongpermissive "
                 f"(triggers: strongest member's test corr vs landmark "
                 f"< {STRONG_CORR_THRESHOLD}).")

    lines.append("\n## Task 3 — submissions built\n")
    if build_v2:
        lines.append(f"- `phase2_blend_2way_optimal_v2.csv` — "
                     f"w_landmark={w_new:.2f}, m-s={ms_new:.4f}.")
    else:
        lines.append("- _v2: not built (curve flat near 0.30; no meaningful "
                     "improvement at finer granularity).")
    if build_sp:
        diag_sp = diag["task3_strongpermissive_chosen"]
        lines.append(f"- `phase2_blend_2way_strongpermissive.csv` — "
                     f"member=`{strongest_member}`, "
                     f"w_landmark={diag_sp['w_landmark']:.2f}, "
                     f"m-s={diag_sp['mean_minus_std']:.4f}.")
    else:
        lines.append("- _strongpermissive: not built (no single member with "
                     f"test corr vs landmark below {STRONG_CORR_THRESHOLD}).")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_MD}", flush=True)
    print(f"\nTotal elapsed: {(time.time()-t_overall)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
