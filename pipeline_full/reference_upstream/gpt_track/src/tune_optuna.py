"""Hyperparameter tuning with Optuna.

We tune one model at a time on a single (feature_set, endpoint) pair, scoring
each trial by the Phase 2 penalised mean (mean - 0.5 * std) of the per-fold
C-index. The penalty discourages high-variance configs that look great on a
lucky fold but fail elsewhere.

Search budget defaults to 50 trials; bump with ``--n-trials``. Studies are
persisted to ``experiments/outputs/phase2_optuna/`` so we can resume.

Usage::

    python -m src.tune_optuna --model lgbm_binary --feature-set NIT_plus_scores_longitudinal --endpoint hepatic --n-trials 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .features import build_feature_set
from .metrics import cindex
from .models import build_model
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger, set_seed, timestamp, write_json
from .validation import build_folds

_LOG = get_logger(__name__)


def _suggest(trial, model: str) -> dict:
    if model == "lgbm_binary":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", -1, 8),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        )
    if model == "catboost_binary":
        return dict(
            iterations=trial.suggest_int("iterations", 300, 1200, step=100),
            depth=trial.suggest_int("depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 10.0, log=True),
        )
    if model == "xgb_binary":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
        )
    if model == "xgb_cox":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 10.0, log=True),
        )
    if model == "rsf":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 600, step=100),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            min_samples_split=trial.suggest_int("min_samples_split", 4, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 15),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        )
    if model == "coxnet":
        return dict(
            l1_ratio=trial.suggest_float("l1_ratio", 0.05, 1.0),
            alpha_min_ratio=trial.suggest_float("alpha_min_ratio", 1e-3, 0.5, log=True),
            n_alphas=trial.suggest_int("n_alphas", 30, 150),
        )
    raise KeyError(f"no search space defined for model {model}")


def _evaluate(model_name: str, params: dict, fs, endpoint, splits) -> tuple[float, float]:
    """Mean and std of fold-wise C-index for the given config."""
    Xtr = fs.X_train
    n = len(Xtr)
    mask = np.asarray(endpoint.mask, dtype=bool)
    fold_scores: list[float] = []
    for s in splits:
        tr_idx = np.array([i for i in s.train_idx if mask[i]], dtype=int)
        va_idx = np.array(s.valid_idx, dtype=int)
        if len(tr_idx) < 5 or endpoint.event[tr_idx].sum() == 0:
            continue
        fold_mask = pd.Series(False, index=Xtr.index)
        fold_mask.iloc[tr_idx] = True
        try:
            m = build_model(model_name, params)
            m.fit(Xtr, endpoint, mask=fold_mask)
            preds = m.predict_risk(Xtr.iloc[va_idx])
            sc = cindex(endpoint.event[va_idx], endpoint.time[va_idx], preds).cindex
            if np.isfinite(sc):
                fold_scores.append(sc)
        except Exception as e:  # noqa: BLE001
            _LOG.warning("trial fold failed: %s", e)
    if not fold_scores:
        return float("nan"), float("nan")
    return float(np.mean(fold_scores)), float(np.std(fold_scores))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--feature-set", required=True)
    p.add_argument("--endpoint", choices=["hepatic", "death"], required=True)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    args = p.parse_args()

    set_seed(args.seed)

    import optuna

    out_root = cfg.EXPERIMENT_OUTPUTS / "phase2_optuna"
    out_root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_root}/optuna.db"
    study_name = f"{args.feature_set}__{args.endpoint}__{args.model}"

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(
        ds.train_df,
        hepatic_event=hep.event.astype(int),
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )
    fs = build_feature_set(args.feature_set, ds)
    endpoint = hep if args.endpoint == "hepatic" else death

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest(trial, args.model)
        mean, std = _evaluate(args.model, params, fs, endpoint, splits)
        if not np.isfinite(mean):
            return float("-inf")
        trial.set_user_attr("cindex_mean", mean)
        trial.set_user_attr("cindex_std", std)
        return mean - 0.5 * std

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best = {
        "study_name": study_name,
        "model": args.model,
        "feature_set": args.feature_set,
        "endpoint": args.endpoint,
        "n_trials": len(study.trials),
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "best_user_attrs": dict(study.best_trial.user_attrs),
    }
    out_dir = out_root / study_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"best_{timestamp()}.json", best)
    _LOG.info("study %s best=%.4f params=%s", study_name, best["best_value"], best["best_params"])


if __name__ == "__main__":
    main()
