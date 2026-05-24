"""Phase 2 Experiment 2 — Optuna hyperparameter tuning for hepatic candidates.

Six candidates, TPE sampler, per-model resumable SQLite studies.

Selection metric (pre-registered): mean − 1×std of hepatic C-index across
5-fold × 5-repeat stratified CV. Stable models > lucky models.

Trial budget: 50 base. If best at trial 40 has improved by >0.005 over best
at trial 30, extend to 100 for that model.

Stop condition: if any tuned model's best mean exceeds 0.85, halt and report.
This is suspicious given the honest ceiling (~0.80 baseline_v1 RSF) and likely
indicates leakage; warrants investigation before proceeding.

Outputs:
  configs/optuna_<id>.db    — SQLite study (resumable)
  configs/optuna_<id>.json  — best params + summary
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
from src.models import (make_rsf, make_xgb_cox, make_coxnet,
                        make_catboost_binary)

CONFIGS = ROOT / "configs"
CONFIGS.mkdir(parents=True, exist_ok=True)

OPTUNA_SPLITS = 5
OPTUNA_TUNING_REPEATS = 5
N_TRIALS_BASE = 50
N_TRIALS_MAX = 100
HEARTBEAT = 10
SUSPICIOUS_THRESHOLD = 0.85

CANDIDATES = [
    {"id": "rsf_baseline_v1",            "model": "rsf",          "fs": "baseline_v1"},
    {"id": "rsf_nit_only_baseline_only", "model": "rsf",          "fs": "nit_only_baseline_only"},
    {"id": "rsf_early_v1_v3",            "model": "rsf",          "fs": "early_v1_v3"},
    {"id": "xgb_cox_baseline_v1",        "model": "xgb_cox",      "fs": "baseline_v1"},
    {"id": "catboost_bin_early_v1_v3",   "model": "catboost_bin", "fs": "early_v1_v3"},
    {"id": "coxnet_baseline_v1",         "model": "coxnet",       "fs": "baseline_v1"},
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
    if model == "catboost_bin":
        return {
            "iterations": trial.suggest_int("iterations", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "horizon": trial.suggest_categorical("horizon", [3, 5, 7, 10]),
        }
    if model == "coxnet":
        return {
            "l1_ratio": trial.suggest_float("l1_ratio", 0.1, 0.9),
            "alpha_min_ratio": trial.suggest_float(
                "alpha_min_ratio", 0.001, 0.1, log=True),
        }
    raise ValueError(model)


def make_factory(model, params):
    p = dict(params)
    if model == "rsf":
        return make_rsf(**p)
    if model == "xgb_cox":
        return make_xgb_cox(**p)
    if model == "catboost_bin":
        h = p.pop("horizon")
        return make_catboost_binary(horizon=float(h), **p)
    if model == "coxnet":
        return make_coxnet(**p)
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


def best_value_at(study, k):
    """Best objective value among the first k completed trials."""
    vals = [tr.value for tr in study.trials[:k]
            if tr.state == optuna.trial.TrialState.COMPLETE and tr.value is not None]
    return max(vals) if vals else None


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
    print(f"\n=== {cid} === (existing complete trials: {n_done})", flush=True)

    last_heartbeat = [n_done]

    def cb(study_, trial_):
        n = sum(1 for tr in study_.trials
                if tr.state == optuna.trial.TrialState.COMPLETE)
        if n - last_heartbeat[0] >= HEARTBEAT:
            last_heartbeat[0] = n
            try:
                bm = study_.best_trial.user_attrs.get("mean", float("nan"))
                bs = study_.best_trial.user_attrs.get("std", float("nan"))
                bv = study_.best_value
                print(f"  [{cid}] trial {n}: best m−s={bv:.4f} "
                      f"(mean={bm:.4f} std={bs:.4f})", flush=True)
            except (ValueError, RuntimeError):
                pass

    obj = lambda tr: objective(tr, cand["model"], X, e, t)

    # Run to N_TRIALS_BASE
    target = N_TRIALS_BASE
    if n_done < target:
        study.optimize(obj, n_trials=target - n_done, callbacks=[cb])

    # Extension check
    n_complete = sum(1 for tr in study.trials
                     if tr.state == optuna.trial.TrialState.COMPLETE)
    if n_complete >= 40:
        b30 = best_value_at(study, 30)
        b40 = best_value_at(study, 40)
        if b30 is not None and b40 is not None:
            delta = b40 - b30
            if delta > 0.005 and n_complete < N_TRIALS_MAX:
                print(f"  [{cid}] still improving at 40 trials "
                      f"(Δ={delta:+.4f} vs 30), extending to {N_TRIALS_MAX}",
                      flush=True)
                study.optimize(obj, n_trials=N_TRIALS_MAX - n_complete,
                               callbacks=[cb])
            else:
                print(f"  [{cid}] plateau at 40 trials "
                      f"(Δ={delta:+.4f} vs 30); stopping at {n_complete}",
                      flush=True)

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


def main(time_budget_s: float = 4 * 3600):
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
        if time.time() - overall_start > time_budget_s:
            print(f"\n*** TIME BUDGET EXCEEDED ({(time.time()-overall_start)/60:.0f} min) — STOP ***",
                  flush=True)
            return summaries

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
            print("HALTING tuning. Run leakage check before proceeding.",
                  flush=True)
            (CONFIGS / "TUNING_HALTED.txt").write_text(
                f"Halted on {cand['id']} mean={rec['best_mean']:.4f}\n"
            )
            return summaries

    print("\n=== TUNING SUMMARY ===", flush=True)
    summaries_sorted = sorted(summaries, key=lambda r: -r["best_mean_minus_std"])
    for r in summaries_sorted:
        print(f"  {r['id']:35s} mean={r['best_mean']:.4f}±{r['best_std']:.4f} "
              f"m−s={r['best_mean_minus_std']:.4f} "
              f"({r['n_trials_complete']} trials)", flush=True)
    print(f"Total tuning wall-clock: {(time.time()-overall_start)/60:.1f} min",
          flush=True)
    return summaries


if __name__ == "__main__":
    main()
