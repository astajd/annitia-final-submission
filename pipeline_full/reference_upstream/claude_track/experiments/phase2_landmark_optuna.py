"""Task 2 — Optuna tune RSF specifically on the 3y landmark feature set (Filter B).

Current 3y landmark RSF reuses the rsf_baseline_v1 Optuna best params (23-feature
tuning). The landmark feature set is structurally different (~49 features incl
LOCF + slopes). Re-tune RSF with the same search space as Experiment 2 (Filter
B cohort: event-free at landmark AND has data through landmark, n=831, 27 events).

Inner CV: 5-fold × 5-repeat (matches Experiment 2 Optuna inner CV).
Search space: identical to Experiment 2 RSF (n_estimators 200-800, leaf 5-50,
split 10-100, max_features ['sqrt','log2',0.3,0.5]).
Selection metric: mean − 1·std of C-index. 30 trials, TPE.

Final eval at 5×10 CV on Filter B with the tuned params; compare to current
filter-B baseline (m−s 0.812 — see SILENTBASE/audit notes).

If tuned m-s beats current by ≥0.01: integrate into 2-way blend with same
permissive ensemble + 50/50 weighting; save phase2_blend_2way_landmarktuned.csv.

Outputs:
  configs/optuna_rsf_landmark_3y.{db,json}
  reports/phase2_landmark_optuna.{json,md}
  submissions/phase2_blend_2way_landmarktuned.{csv,json}  (only if criteria met)
"""
from __future__ import annotations
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from scipy.stats import rankdata

from src.config import REPORTS, SUBMISSIONS, ROOT
from src.cv import evaluate_cv, summarize
from src.data import load_raw, build_targets, add_visit_metadata
from src.features import (build_features, build_landmark_features,
                          at_risk_at_landmark)
from src.models import make_rsf

CONFIGS = ROOT / "configs"
LANDMARK = 3.0
N_TRIALS = 30
INNER_SPLITS = 5
INNER_REPEATS = 5
FINAL_SPLITS = 5
FINAL_REPEATS = 10
SEED = 42
HEARTBEAT = 5
ID = "rsf_landmark_3y"

# Filter-B baseline (tuned baseline_v1 RSF on filter-B at 5×10 CV); see
# phase2d_audit.py output. We keep the same anchor for the Δ comparison.
FILTER_B_BASELINE_MS = 0.812  # 0.882 - 0.070, from prior audit

OUT_JSON = REPORTS / "phase2_landmark_optuna.json"
OUT_MD = REPORTS / "phase2_landmark_optuna.md"
OPT_DB = CONFIGS / f"optuna_{ID}.db"
OPT_JSON = CONFIGS / f"optuna_{ID}.json"
LT_CSV = SUBMISSIONS / "phase2_blend_2way_landmarktuned.csv"
LT_META = SUBMISSIONS / "phase2_blend_2way_landmarktuned.json"
IMPROVEMENT_THRESHOLD = 0.01

PERMISSIVE_MEMBERS = [
    "rsf_longitudinal_summary",
    "xgb_cox_longitudinal_plus_meta",
    "xgb_cox_longitudinal_summary",
]

optuna.logging.set_verbosity(optuna.logging.WARNING)


def has_data_through(df: pd.DataFrame, lt: float) -> np.ndarray:
    d = add_visit_metadata(df)
    return ((d["max_age"] - d["Age_v1"]).to_numpy() >= lt - 1e-9)


def suggest_rsf(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 10, 100),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.3, 0.5]),
    }


def objective(trial, X, e, t):
    params = suggest_rsf(trial)
    factory = make_rsf(**params)
    cv_res = evaluate_cv(factory, X, e, t,
                         n_splits=INNER_SPLITS, n_repeats=INNER_REPEATS)
    s = summarize(cv_res)
    trial.set_user_attr("mean", s["mean"])
    trial.set_user_attr("std", s["std"])
    return s["mean"] - s["std"]


def main():
    t_start = time.time()

    train, test = load_raw()
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e_full = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t_full = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df_full = train.loc[valid].reset_index(drop=True)

    keep_A = at_risk_at_landmark(e_full, t_full, LANDMARK)
    keep_B = keep_A & has_data_through(df_full, LANDMARK)

    df_B = df_full.loc[keep_B].reset_index(drop=True)
    e_B = e_full[keep_B]
    t_B = t_full[keep_B]
    print(f"Filter B (event-free at {LANDMARK}y AND has data through "
          f"{LANDMARK}y): n={len(e_B)}, events={int(e_B.sum())}", flush=True)

    X_B = build_landmark_features(df_B, LANDMARK)
    print(f"Landmark features: {X_B.shape[1]}", flush=True)

    # ---- Optuna study ----
    storage = f"sqlite:///{OPT_DB}"
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        study_name=ID, storage=storage, load_if_exists=True,
    )
    n_done = sum(1 for tr in study.trials
                 if tr.state == optuna.trial.TrialState.COMPLETE)
    print(f"\n=== Optuna study {ID} === existing complete trials: {n_done}",
          flush=True)

    last_hb = [n_done]
    def cb(study_, trial_):
        n = sum(1 for tr in study_.trials
                if tr.state == optuna.trial.TrialState.COMPLETE)
        if n - last_hb[0] >= HEARTBEAT:
            last_hb[0] = n
            try:
                bm = study_.best_trial.user_attrs.get("mean", float("nan"))
                bs = study_.best_trial.user_attrs.get("std", float("nan"))
                bv = study_.best_value
                print(f"  trial {n}: best m-s={bv:.4f} "
                      f"(mean={bm:.4f} std={bs:.4f})", flush=True)
            except (ValueError, RuntimeError):
                pass

    obj = lambda tr: objective(tr, X_B, e_B, t_B)
    if n_done < N_TRIALS:
        study.optimize(obj, n_trials=N_TRIALS - n_done, callbacks=[cb])
    n_complete = sum(1 for tr in study.trials
                     if tr.state == optuna.trial.TrialState.COMPLETE)

    best = study.best_trial
    rec = {
        "id": ID, "model": "rsf",
        "feature_set": f"landmark_{int(LANDMARK)}y",
        "cohort": f"filter B (n={len(e_B)}, events={int(e_B.sum())})",
        "sampler": "TPE",
        "inner_cv": f"{INNER_SPLITS}-fold × {INNER_REPEATS}-repeat",
        "selection_metric": "mean − 1×std of C-index",
        "n_trials_complete": n_complete,
        "best_trial_number": best.number,
        "best_params": {k: (v if not isinstance(v, np.generic) else v.item())
                        for k, v in best.params.items()},
        "best_mean": float(best.user_attrs.get("mean", float("nan"))),
        "best_std": float(best.user_attrs.get("std", float("nan"))),
        "best_mean_minus_std": float(study.best_value),
    }
    OPT_JSON.write_text(json.dumps(rec, indent=2, default=str))
    print(f"\nBest params: {rec['best_params']}", flush=True)
    print(f"Inner-CV best: mean={rec['best_mean']:.4f} "
          f"std={rec['best_std']:.4f} m-s={rec['best_mean_minus_std']:.4f}",
          flush=True)
    print(f"Tuning elapsed: {(time.time()-t_start)/60:.1f} min", flush=True)

    # ---- Final eval at 5×10 CV ----
    print("\n--- Final 5×10 CV with tuned params on filter B ---", flush=True)
    factory_final = make_rsf(**rec["best_params"])
    final_cv = evaluate_cv(factory_final, X_B, e_B, t_B,
                           n_splits=FINAL_SPLITS, n_repeats=FINAL_REPEATS)
    s_final = summarize(final_cv)
    final_ms = s_final["mean"] - s_final["std"]
    print(f"Final 5×10 m-s = {final_ms:.4f} "
          f"(mean={s_final['mean']:.4f}, std={s_final['std']:.4f})", flush=True)

    delta = final_ms - FILTER_B_BASELINE_MS
    print(f"Δ m-s vs filter-B baseline ({FILTER_B_BASELINE_MS:.4f}) "
          f"= {delta:+.4f}  (threshold for action: ≥{IMPROVEMENT_THRESHOLD})",
          flush=True)

    will_build = delta >= IMPROVEMENT_THRESHOLD

    diag = {
        "task": "Task 2 — Optuna tune of landmark-3y RSF on Filter B",
        "filter_B_cohort": {"n": int(len(e_B)), "events": int(e_B.sum())},
        "n_trials_complete": n_complete,
        "best_inner_cv_5x5": {
            "mean": rec["best_mean"], "std": rec["best_std"],
            "mean_minus_std": rec["best_mean_minus_std"],
            "best_params": rec["best_params"],
        },
        "final_5x10_cv": {
            "mean": s_final["mean"], "std": s_final["std"],
            "mean_minus_std": final_ms,
        },
        "filter_B_baseline_ms": FILTER_B_BASELINE_MS,
        "delta_ms_vs_baseline": delta,
        "improvement_threshold": IMPROVEMENT_THRESHOLD,
        "build_landmarktuned_submission": will_build,
    }
    OUT_JSON.write_text(json.dumps(diag, indent=2, default=str))
    print(f"Wrote {OUT_JSON}", flush=True)

    # Markdown
    lines = [
        f"# Task 2 — Optuna tune of landmark-3y RSF (Filter B)\n",
        f"Date: 2026-04-28. Cohort: Filter B (n={len(e_B)}, events={int(e_B.sum())}). "
        f"Inner CV: 5-fold × 5-repeat. Final eval: 5-fold × 10-repeat.\n",
        f"## Best params ({n_complete} trials)\n",
        "```json", json.dumps(rec["best_params"], indent=2), "```\n",
        f"- Inner CV m-s = {rec['best_mean_minus_std']:.4f} "
        f"(mean {rec['best_mean']:.4f} ± {rec['best_std']:.4f}).",
        f"- Final 5×10 m-s = {final_ms:.4f} "
        f"(mean {s_final['mean']:.4f} ± {s_final['std']:.4f}).",
        f"- Filter-B baseline m-s = {FILTER_B_BASELINE_MS:.4f}.",
        f"- **Δ m-s vs baseline = {delta:+.4f}** "
        f"(threshold for action: ≥{IMPROVEMENT_THRESHOLD}).",
        f"- Decision: "
        f"{'**BUILD** phase2_blend_2way_landmarktuned.csv' if will_build else '**STAY** with current params; no submission built'}.",
    ]
    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}", flush=True)

    # ---- Conditionally build submission ----
    if not will_build:
        print(f"\nNo submission built. Total elapsed: "
              f"{(time.time()-t_start)/60:.1f} min", flush=True)
        return

    print("\n--- Building phase2_blend_2way_landmarktuned.csv ---", flush=True)
    # Landmark on filter-A train (test prediction protocol), with tuned params
    keep_A_full = at_risk_at_landmark(e_full, t_full, LANDMARK)
    df_lm_train = df_full.loc[keep_A_full].reset_index(drop=True)
    e_lm_train = e_full[keep_A_full]
    t_lm_train = t_full[keep_A_full]
    X_lm_tr = build_landmark_features(df_lm_train, LANDMARK)
    X_lm_te = build_landmark_features(test, LANDMARK)
    common_lm = [c for c in X_lm_tr.columns if c in X_lm_te.columns]
    risk_lm = make_rsf(**rec["best_params"])(
        X_lm_tr[common_lm], e_lm_train, t_lm_train, X_lm_te[common_lm])

    # Permissive ensemble (full cohort)
    rank_sum = np.zeros(len(test))
    perm_members_info = []
    for cid in PERMISSIVE_MEMBERS:
        sub_rec = json.loads((CONFIGS / f"optuna_{cid}.json").read_text())
        p = dict(sub_rec["best_params"])
        if sub_rec["model"] == "rsf":
            from src.models import make_rsf as mk
            factory = mk(**p)
        elif sub_rec["model"] == "xgb_cox":
            from src.models import make_xgb_cox as mk
            factory = mk(**p)
        else:
            raise ValueError(sub_rec["model"])
        X_tr_m = build_features(df_full, sub_rec["feature_set"])
        X_te_m = build_features(test, sub_rec["feature_set"])
        common = [c for c in X_tr_m.columns if c in X_te_m.columns]
        risk_m = factory(X_tr_m[common], e_full, t_full, X_te_m[common])
        rank_sum += rankdata(risk_m)
        perm_members_info.append({"id": cid, "model": sub_rec["model"],
                                  "feature_set": sub_rec["feature_set"]})
    risk_perm = rank_sum / len(PERMISSIVE_MEMBERS)

    # 50/50 blend (per task spec — only the landmark component is tuned, blend stays default)
    blend = 0.5 * rankdata(risk_lm) + 0.5 * rankdata(risk_perm)

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
    from src.models import make_xgb_cox
    risk_death = make_xgb_cox(n_estimators=300, learning_rate=0.05)(
        X_d_tr[common_d], e_d, t_d, X_d_te[common_d])

    sub = pd.DataFrame({
        "trustii_id": test["trustii_id"].values,
        "risk_hepatic_event": blend,
        "risk_death": risk_death,
    })
    sub.to_csv(LT_CSV, index=False)
    meta = {
        "submission_kind": "2-way blend with landmark-tuned RSF",
        "weight_landmark": 0.5, "weight_permissive": 0.5,
        "hepatic": {
            "ensemble_strategy": "0.50*rank(landmark) + 0.50*rank(permissive)",
            "components": {
                "a_landmark_3y_tuned": {
                    "feature_set": f"landmark_{int(LANDMARK)}y",
                    "model": "rsf",
                    "params": rec["best_params"],
                    "tuning_cohort_filter_B_5x5_ms": rec["best_mean_minus_std"],
                    "final_5x10_filter_B_ms": final_ms,
                    "n_train_filterA_for_test_pred": int(len(e_lm_train)),
                    "n_train_events": int(e_lm_train.sum()),
                    "n_features": len(common_lm),
                },
                "b_permissive": {"members": perm_members_info},
            },
        },
        "death": {"feature_set": fs_d, "model": "xgb_cox",
                  "params": {"n_estimators": 300, "learning_rate": 0.05}},
        "reference_LBs": {
            "phase2_blend_landmark_permissive_50_50": 0.88033,
        },
    }
    LT_META.write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote {LT_CSV}\nWrote {LT_META}", flush=True)
    print(f"Total elapsed: {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
