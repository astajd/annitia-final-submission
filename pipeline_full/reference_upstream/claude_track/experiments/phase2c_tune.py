"""Phase 2c — Optuna tuning for the permissive (longitudinal) track.

Four candidates, 30 trials each, TPE sampler, resumable SQLite studies.
Same selection metric (mean − 1×std), same inner CV protocol (5×5 = 25 folds).
Stop condition: mean > 0.92 → halt, possible new leakage source.

Re-uses suggest_params / make_factory from phase2_optuna_tune.py.
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

from src.config import ROOT
from src.cv import evaluate_cv, summarize
from src.data import load_raw, build_targets
from src.features import build_features
from src.models import (make_rsf, make_xgb_cox, make_lgbm_binary)

CONFIGS = ROOT / "configs"
CONFIGS.mkdir(parents=True, exist_ok=True)

OPTUNA_SPLITS = 5
OPTUNA_TUNING_REPEATS = 5
N_TRIALS = 30
HEARTBEAT = 10
SUSPICIOUS_THRESHOLD = 0.92  # higher than Exp 2 because permissive features
                              # legitimately reach 0.78–0.85; flag only on >0.92

CANDIDATES = [
    {"id": "rsf_longitudinal_summary",        "model": "rsf",      "fs": "longitudinal_summary"},
    {"id": "xgb_cox_longitudinal_summary",    "model": "xgb_cox",  "fs": "longitudinal_summary"},
    {"id": "xgb_cox_longitudinal_plus_meta",  "model": "xgb_cox",  "fs": "longitudinal_plus_meta"},
    {"id": "lgbm_bin_longitudinal_plus_meta", "model": "lgbm_bin", "fs": "longitudinal_plus_meta"},
]

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial, model):
    if model == "rsf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 100),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5]),
        }
    if model == "xgb_cox":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
    if model == "lgbm_bin":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10.0, log=True),
            "horizon": trial.suggest_categorical("horizon", [3, 5, 7, 10]),
        }
    raise ValueError(model)


def make_factory(model, params):
    p = dict(params)
    if model == "rsf":
        return make_rsf(**p)
    if model == "xgb_cox":
        return make_xgb_cox(**p)
    if model == "lgbm_bin":
        h = p.pop("horizon")
        return make_lgbm_binary(horizon=float(h), **p)
    raise ValueError(model)


def objective(trial, model, X, e, t):
    params = suggest_params(trial, model)
    factory = make_factory(model, params)
    cv_res = evaluate_cv(factory, X, e, t,
                         n_splits=OPTUNA_SPLITS,
                         n_repeats=OPTUNA_TUNING_REPEATS)
    s = summarize(cv_res)
    trial.set_user_attr("mean", s["mean"])
    trial.set_user_attr("std", s["std"])
    return s["mean"] - s["std"]


def tune_one(cand, X, e, t):
    cid = cand["id"]
    db_path = CONFIGS / f"optuna_{cid}.db"
    storage = f"sqlite:///{db_path}"
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=cid,
        storage=storage,
        load_if_exists=True,
    )
    n_done = sum(1 for tr in study.trials
                 if tr.state == optuna.trial.TrialState.COMPLETE)
    print(f"\n=== {cid} === (existing: {n_done})", flush=True)

    last_hb = [n_done]

    def cb(study_, trial_):
        n = sum(1 for tr in study_.trials
                if tr.state == optuna.trial.TrialState.COMPLETE)
        if n - last_hb[0] >= HEARTBEAT:
            last_hb[0] = n
            try:
                bm = study_.best_trial.user_attrs.get("mean", float("nan"))
                bs = study_.best_trial.user_attrs.get("std", float("nan"))
                print(f"  [{cid}] trial {n}: best m−s={study_.best_value:.4f} "
                      f"(mean={bm:.4f} std={bs:.4f})", flush=True)
            except (ValueError, RuntimeError):
                pass

    obj = lambda tr: objective(tr, cand["model"], X, e, t)
    if n_done < N_TRIALS:
        study.optimize(obj, n_trials=N_TRIALS - n_done, callbacks=[cb])
    return study


def write_candidate_json(cand, study, elapsed):
    best = study.best_trial
    rec = {
        "id": cand["id"],
        "model": cand["model"],
        "feature_set": cand["fs"],
        "sampler": "TPE",
        "inner_cv": f"{OPTUNA_SPLITS}-fold × {OPTUNA_TUNING_REPEATS}-repeat",
        "selection_metric": "mean − 1×std of C-index",
        "n_trials_complete": sum(1 for tr in study.trials
                                 if tr.state == optuna.trial.TrialState.COMPLETE),
        "n_trials_failed": sum(1 for tr in study.trials
                               if tr.state == optuna.trial.TrialState.FAIL),
        "best_trial_number": best.number,
        "best_params": {k: (v if not isinstance(v, np.generic) else v.item())
                        for k, v in best.params.items()},
        "best_mean": float(best.user_attrs.get("mean", float("nan"))),
        "best_std": float(best.user_attrs.get("std", float("nan"))),
        "best_mean_minus_std": float(study.best_value),
        "tuning_seconds": round(elapsed, 1),
    }
    out = CONFIGS / f"optuna_{cand['id']}.json"
    out.write_text(json.dumps(rec, indent=2, default=str))
    return rec


def main():
    overall_start = time.time()
    train, _ = load_raw()
    hep_t, _ = build_targets(train, "drop_missing_death")
    valid = hep_t["hepatic_valid"].to_numpy()
    e = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)
    print(f"Train: n={len(e)}, events={int(e.sum())}", flush=True)

    summaries = []
    for cand in CANDIDATES:
        X = build_features(df, cand["fs"])
        print(f"\n[{cand['id']}] feature_set={cand['fs']} ({X.shape[1]} features)",
              flush=True)
        t0 = time.time()
        study = tune_one(cand, X, e, t)
        el = time.time() - t0

        rec = write_candidate_json(cand, study, el)
        summaries.append(rec)
        print(f"  [{cand['id']}] done: mean={rec['best_mean']:.4f} "
              f"std={rec['best_std']:.4f} m−s={rec['best_mean_minus_std']:.4f} "
              f"({rec['n_trials_complete']} trials, {el:.0f}s)", flush=True)

        if rec["best_mean"] > SUSPICIOUS_THRESHOLD:
            print(f"\n*** SUSPICIOUS *** {cand['id']} mean={rec['best_mean']:.4f} > {SUSPICIOUS_THRESHOLD}",
                  flush=True)
            (CONFIGS / "TUNING_2C_HALTED.txt").write_text(
                f"Halted on {cand['id']} mean={rec['best_mean']:.4f}\n"
            )
            return summaries

    print("\n=== PHASE 2C TUNING SUMMARY ===", flush=True)
    for r in sorted(summaries, key=lambda r: -r["best_mean_minus_std"]):
        print(f"  {r['id']:35s} mean={r['best_mean']:.4f}±{r['best_std']:.4f} "
              f"m−s={r['best_mean_minus_std']:.4f} "
              f"({r['n_trials_complete']} trials)", flush=True)
    print(f"Wall-clock: {(time.time()-overall_start)/60:.1f} min", flush=True)
    return summaries


if __name__ == "__main__":
    main()
