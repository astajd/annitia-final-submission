"""Single-experiment runner.

Reads a YAML config, builds a feature set, fits one or more models for each
endpoint under repeated stratified CV, optionally rank-averages the models,
and writes:

    experiments/outputs/<TIMESTAMP>_<NAME>/
        config.json
        cv_metrics.csv          one row per (repeat, fold)
        cv_summary.json         mean/std/min summary
        oof_predictions.csv     out-of-fold predictions per row
        test_predictions.csv    averaged test predictions
        feature_importance.csv  if available
        summary.md              human-readable report

It also writes a submission CSV in ``submissions/``.

Usage::

    python -m src.run_experiment --config experiments/configs/001_baseline_v1.yaml
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import config as cfg
from .data_loading import last_observed_age, load_dataset
from .features import build_feature_set
from .metrics import cindex, weighted_score
from .models import build_model
from .models.ensemble import rank_average
from .submission import make_submission
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import StopWatch, get_logger, set_seed, timestamp, write_json
from .validation import build_folds, fold_event_counts, save_folds, summarize_cv

_LOG = get_logger(__name__)


def _hepatic_cutoff(train_df, age_cols) -> pd.Series:
    """Per-row cutoff age for the strict_time_aligned feature set on the hepatic endpoint."""
    hep_age = train_df[cfg.HEPATIC_EVENT_AGE_COL]
    last_age = last_observed_age(train_df, age_cols)
    return hep_age.where(train_df[cfg.HEPATIC_EVENT_COL] == 1, last_age)


def _death_cutoff(train_df, age_cols) -> pd.Series:
    death_age = train_df[cfg.DEATH_EVENT_AGE_COL]
    last_age = last_observed_age(train_df, age_cols)
    return death_age.where(train_df[cfg.DEATH_EVENT_COL] == 1, last_age)


def _build_feature_set_for(endpoint_name: str, fs_name: str, ds, opts: dict | None):
    """For strict_time_aligned, the cutoff depends on the endpoint."""
    opts = dict(opts or {})
    if fs_name == "strict_time_aligned":
        if endpoint_name == "hepatic":
            opts["cutoff_age_train"] = _hepatic_cutoff(ds.train_df, ds.age_visit_cols)
        elif endpoint_name == "death":
            opts["cutoff_age_train"] = _death_cutoff(ds.train_df, ds.age_visit_cols)
    return build_feature_set(fs_name, ds, opts)


def _train_predict_one_model(
    model_name: str,
    model_params: dict,
    fs,
    endpoint,
    splits,
) -> tuple[np.ndarray, list[float], np.ndarray]:
    """Run repeated CV for one (model, endpoint) pair.

    Returns (oof_predictions, per_fold_cindex, full_train_test_predictions).
    """
    Xtr = fs.X_train
    Xte = fs.X_test
    n = len(Xtr)
    oof = np.full(n, np.nan)
    fold_scores: list[float] = []
    test_preds = np.zeros(len(Xte))
    n_test_contributions = 0

    mask = np.asarray(endpoint.mask, dtype=bool)

    for s in splits:
        tr_idx = np.array([i for i in s.train_idx if mask[i]], dtype=int)
        va_idx = np.array(s.valid_idx, dtype=int)
        if len(tr_idx) < 5 or endpoint.event[tr_idx].sum() == 0:
            continue
        # Build a full-length boolean Series so the model wrapper can index endpoint.event/time.
        fold_mask = pd.Series(False, index=Xtr.index)
        fold_mask.iloc[tr_idx] = True
        try:
            model = build_model(model_name, model_params)
            model.fit(Xtr, endpoint, mask=fold_mask)
            preds = model.predict_risk(Xtr.iloc[va_idx])
            oof[va_idx] = preds
            va_event = endpoint.event[va_idx]
            va_time = endpoint.time[va_idx]
            score = cindex(va_event, va_time, preds).cindex
            fold_scores.append(score)
            test_preds += model.predict_risk(Xte)
            n_test_contributions += 1
        except Exception as e:  # noqa: BLE001
            _LOG.warning("fold (rep=%d, fold=%d) %s failed: %s",
                         s.repeat, s.fold, model_name, e)

    if n_test_contributions:
        test_preds /= n_test_contributions
    return oof, fold_scores, test_preds


def run_config(cfg_path: Path) -> Path:
    cfg.ensure_dirs()
    cfg_path = Path(cfg_path)
    with cfg_path.open() as f:
        c = yaml.safe_load(f)

    set_seed(c.get("seed", cfg.RANDOM_SEED))

    name = c["name"]
    out_dir = cfg.EXPERIMENT_OUTPUTS / f"{timestamp()}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", c)
    _LOG.info("=== experiment %s -> %s ===", name, out_dir)

    ds = load_dataset()

    # Endpoints.
    death_mode = c.get("death_target_mode", "censor_missing_death_at_last_visit")
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols, mode=death_mode)

    # Folds (built on the same hepatic stratifier across all experiments).
    n_splits = int(c.get("n_splits", cfg.N_SPLITS))
    n_repeats = int(c.get("n_repeats", cfg.N_REPEATS))
    splits = build_folds(
        ds.train_df,
        hepatic_event=hep.event.astype(int),
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=c.get("seed", cfg.RANDOM_SEED),
    )
    save_folds(splits, out_dir / "folds.csv")
    fold_event_counts(splits, hep.event.astype(int)).to_csv(out_dir / "fold_event_counts_hepatic.csv", index=False)
    fold_event_counts(splits, death.event.astype(int)).to_csv(out_dir / "fold_event_counts_death.csv", index=False)

    fs_name = c["feature_set"]
    models_cfg: list[dict] = c["models"]

    # Build per-endpoint feature sets (strict_time_aligned is the only one that varies).
    fs_hep = _build_feature_set_for("hepatic", fs_name, ds, c.get("feature_opts"))
    fs_death = _build_feature_set_for("death", fs_name, ds, c.get("feature_opts"))

    per_model_oof: dict[str, np.ndarray] = {}
    per_model_test: dict[str, np.ndarray] = {}
    per_endpoint_metrics: list[dict] = []
    per_model_summary: list[dict] = []

    for endpoint_name, endpoint, fs in [
        ("hepatic", hep, fs_hep),
        ("death", death, fs_death),
    ]:
        for m in models_cfg:
            mname = m["name"]
            mparams = m.get("params", {}) or {}
            tag = m.get("tag")
            label = f"{endpoint_name}__{mname}" if tag is None else f"{endpoint_name}__{mname}__{tag}"
            with StopWatch(f"fit {label}"):
                oof, fold_scores, test_preds = _train_predict_one_model(
                    mname, mparams, fs, endpoint, splits
                )
            per_model_oof[label] = oof
            per_model_test[label] = test_preds

            valid_scores = [s for s in fold_scores if np.isfinite(s)]
            mean = float(np.mean(valid_scores)) if valid_scores else float("nan")
            std = float(np.std(valid_scores)) if valid_scores else float("nan")
            mn = float(np.min(valid_scores)) if valid_scores else float("nan")
            mx = float(np.max(valid_scores)) if valid_scores else float("nan")
            # Penalised score we use for hyperparameter selection (Phase 2):
            # mean - 0.5 * std, so we stop trusting models that vary wildly.
            penalised = mean - 0.5 * std if valid_scores else float("nan")
            per_model_summary.append(
                {
                    "endpoint": endpoint_name,
                    "model": mname,
                    "feature_set": fs.name,
                    "leakage_risk": fs.leakage_risk,
                    "n_folds_evaluated": len(valid_scores),
                    "cindex_mean": mean,
                    "cindex_std": std,
                    "cindex_min": mn,
                    "cindex_max": mx,
                    "cindex_mean_minus_halfsd": penalised,
                }
            )

    summary_df = pd.DataFrame(per_model_summary)
    summary_df.to_csv(out_dir / "per_model_summary.csv", index=False)

    # Build the rank-average ensemble per endpoint, dropping models that failed
    # every fold (their OOF is all-NaN and would poison the rank average).
    ensemble_oof = {}
    ensemble_test = {}
    fold_metric_rows: list[dict] = []
    for endpoint_name, endpoint in [("hepatic", hep), ("death", death)]:
        keys = [
            k for k in per_model_oof
            if k.startswith(f"{endpoint_name}__")
            and np.isfinite(per_model_oof[k]).any()
        ]
        if not keys:
            _LOG.warning("no usable models for endpoint %s; skipping ensemble", endpoint_name)
            continue
        oof_dict = {k: per_model_oof[k] for k in keys}
        test_dict = {k: per_model_test[k] for k in keys}
        ensemble_oof[endpoint_name] = rank_average(oof_dict)
        ensemble_test[endpoint_name] = rank_average(test_dict)

        # Per-fold ensemble score; tolerate folds where OOF is partially NaN.
        for s in splits:
            va = s.valid_idx
            preds = ensemble_oof[endpoint_name][va]
            finite = np.isfinite(preds)
            if finite.sum() == 0:
                continue
            ev = endpoint.event[va][finite]
            t = endpoint.time[va][finite]
            sc = cindex(ev, t, preds[finite]).cindex
            fold_metric_rows.append(
                {
                    "repeat": s.repeat,
                    "fold": s.fold,
                    f"cindex_{endpoint_name}": sc,
                }
            )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    if not fold_metrics.empty:
        fold_metrics = fold_metrics.groupby(["repeat", "fold"]).first().reset_index()
        fold_metrics["score"] = (
            cfg.WEIGHT_HEPATIC * fold_metrics.get("cindex_hepatic", np.nan)
            + cfg.WEIGHT_DEATH * fold_metrics.get("cindex_death", np.nan)
        )
    fold_metrics.to_csv(out_dir / "cv_metrics.csv", index=False)
    write_json(out_dir / "cv_summary.json", summarize_cv(fold_metrics))

    oof_df = pd.DataFrame(per_model_oof)
    oof_df[cfg.PATIENT_ID_COL] = ds.train_df[cfg.PATIENT_ID_COL].values
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)

    test_df_out = pd.DataFrame(per_model_test)
    test_df_out[cfg.TRUSTII_ID_COL] = ds.test_df[cfg.TRUSTII_ID_COL].values
    if "hepatic" in ensemble_test:
        test_df_out["ensemble_hepatic"] = ensemble_test["hepatic"]
    if "death" in ensemble_test:
        test_df_out["ensemble_death"] = ensemble_test["death"]
    test_df_out.to_csv(out_dir / "test_predictions.csv", index=False)

    # Submission from the rank-average ensemble.
    if "hepatic" in ensemble_test and "death" in ensemble_test:
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=ensemble_test["hepatic"],
            risk_death=ensemble_test["death"],
            sample_submission=ds.sample_submission,
            model_name=name,
        )
        _LOG.info("submission written to %s", sub_path)

    # Human-readable summary.
    summary_md = out_dir / "summary.md"
    with summary_md.open("w") as f:
        f.write(f"# Experiment: {name}\n\n")
        f.write(f"- feature_set: `{fs_name}` (leakage risk: {fs_hep.leakage_risk})\n")
        f.write(f"- death_target_mode: `{death_mode}`\n")
        f.write(f"- folds: {n_splits} x {n_repeats} repeats\n")
        f.write(f"- hepatic events: {int(hep.event.sum())} / {len(hep.event)}\n")
        f.write(f"- death events:   {int(death.event.sum())} / {death.mask.sum()}\n\n")
        f.write("## Per-model CV summary\n\n")
        f.write(summary_df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")
        f.write("## Ensemble CV summary\n\n")
        if not fold_metrics.empty:
            f.write(f"```json\n{json.dumps(summarize_cv(fold_metrics), indent=2)}\n```\n")
        else:
            f.write("(no folds evaluated successfully)\n")

    _LOG.info("done %s", out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_config(args.config)


if __name__ == "__main__":
    main()
