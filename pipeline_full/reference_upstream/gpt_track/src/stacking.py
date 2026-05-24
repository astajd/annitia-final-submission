"""Stacking / meta-learning over OOF predictions.

We assemble every base-model OOF prediction across phases (and the
discrete-time hazard OOFs if available) and train one of two meta-learners
per endpoint:

- ``ridge``: ``sklearn.linear_model.RidgeClassifier`` on percentile-rank inputs.
- ``elastic_net``: ``sklearn.linear_model.LogisticRegression`` with elastic-net
  regularization (saga solver) on the same rank-transformed features.

The meta-learner is trained on the *same* repeated stratified folds we have
used since Phase 1, so its OOF is honest. Test predictions average the
per-fold meta-model outputs (no refit on full train, mirroring the way we
generated the base OOFs).

The stacking module is not an ensemble selector — it is orthogonal: we feed
its OOFs back into the ensemble search alongside other components.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .endpoint_ensemble import collect_predictions
from .metrics import cindex
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger
from .validation import build_folds

_LOG = get_logger(__name__)


@dataclass
class StackResult:
    endpoint: str
    method: str
    oof_cindex: float
    fold_scores: list[float]
    oof: np.ndarray
    test: np.ndarray
    components: list[str]


def _to_rank_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].rank(method="average", pct=True, na_option="keep").to_numpy()


def _label_for(event: np.ndarray) -> np.ndarray:
    return event.astype(int)


def _fit_meta(method: str, X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression, RidgeClassifier

    if method == "ridge":
        m = RidgeClassifier(alpha=2.0, random_state=0)
        m.fit(X, y)
        return m
    if method == "elastic_net":
        m = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            C=1.0, max_iter=2000, random_state=0,
        )
        m.fit(X, y)
        return m
    raise KeyError(method)


def _predict_score(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def stack_endpoint(
    method: str,
    endpoint_name: str,
    oof_df: pd.DataFrame,
    test_df: pd.DataFrame,
    component_cols: list[str],
    splits,
    train_index_pos: np.ndarray,
    test_index_pos: np.ndarray,
    n_train: int,
    n_test: int,
    event: np.ndarray,
    time: np.ndarray,
) -> StackResult:
    R_train = np.full((n_train, len(component_cols)), np.nan)
    raw_train = oof_df[component_cols].rank(method="average", pct=True, na_option="keep").to_numpy()
    R_train[train_index_pos] = raw_train

    R_test = np.full((n_test, len(component_cols)), np.nan)
    raw_test = test_df[component_cols].rank(method="average", pct=True, na_option="keep").to_numpy()
    R_test[test_index_pos] = raw_test
    R_test = np.nan_to_num(R_test, nan=0.5)

    oof = np.full(n_train, np.nan)
    test_sum = np.zeros(n_test)
    test_n = 0
    fold_scores: list[float] = []

    for s in splits:
        tr = s.train_idx
        va = s.valid_idx
        # Replace NaN in fold-train with 0.5 (neutral rank).
        Xt = np.nan_to_num(R_train[tr], nan=0.5)
        Xv = np.nan_to_num(R_train[va], nan=0.5)
        yt = _label_for(event[tr])
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            m = _fit_meta(method, Xt, yt)
            preds_v = _predict_score(m, Xv)
            preds_t = _predict_score(m, R_test)
        except Exception as e:  # noqa: BLE001
            _LOG.warning("stack %s fold (rep=%d, fold=%d) failed: %s", method, s.repeat, s.fold, e)
            continue
        oof[va] = preds_v
        sc = cindex(event[va], time[va], preds_v).cindex
        if np.isfinite(sc):
            fold_scores.append(sc)
        test_sum += preds_t
        test_n += 1

    test = test_sum / test_n if test_n else np.full(n_test, np.nan)
    cv_score = cindex(event[~np.isnan(oof)], time[~np.isnan(oof)], oof[~np.isnan(oof)]).cindex
    return StackResult(endpoint_name, method, cv_score, fold_scores, oof, test, component_cols)


def main() -> None:
    out_root = cfg.EXPERIMENT_OUTPUTS / "phase3_stacking"
    out_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)

    exp_dirs = [d for d in cfg.EXPERIMENT_OUTPUTS.iterdir()
                if d.is_dir() and (d / "oof_predictions.csv").exists()]
    oof_df, test_df, meta = collect_predictions(exp_dirs)

    # Optionally append discrete-time hazard OOFs (rename test column to match the
    # train OOF column name so the stacker treats them as the same feature).
    dth_root = cfg.EXPERIMENT_OUTPUTS / "phase3_discrete_time_hazard"
    if dth_root.exists():
        for ep in ("hepatic", "death"):
            of = dth_root / f"oof_dth_{ep}.csv"
            tf = dth_root / f"test_dth_{ep}.csv"
            if of.exists():
                oof_df = oof_df.merge(pd.read_csv(of), on=cfg.PATIENT_ID_COL, how="left")
            if tf.exists():
                tdf = pd.read_csv(tf)
                tdf = tdf.rename(columns={f"test_dth_{ep}": f"oof_dth_{ep}"})
                test_df = test_df.merge(tdf, on=cfg.TRUSTII_ID_COL, how="left")

    pid_col = cfg.PATIENT_ID_COL
    tid_col = cfg.TRUSTII_ID_COL
    train_pid_to_row = {pid: i for i, pid in enumerate(ds.train_df[pid_col].values)}
    pos = oof_df[pid_col].map(train_pid_to_row).to_numpy()
    keep = ~pd.isna(pos)
    pos = pos[keep].astype(int)
    oof_df = oof_df.loc[keep].reset_index(drop=True)

    test_idx_by_id = {tid: i for i, tid in enumerate(ds.test_df[tid_col].values)}
    tpos = test_df[tid_col].map(test_idx_by_id).to_numpy()
    tkeep = ~pd.isna(tpos)
    tpos = tpos[tkeep].astype(int)
    test_df = test_df.loc[tkeep].reset_index(drop=True)

    n_train = len(ds.train_df)
    n_test = len(ds.test_df)

    summary = {}
    for endpoint_name, ep in [("hepatic", hep), ("death", death)]:
        component_cols = [
            c for c in oof_df.columns
            if c != pid_col and (
                f"::{endpoint_name}__" in c or c == f"oof_dth_{endpoint_name}"
            ) and "::ensemble_" not in c
        ]
        # Keep components with reasonable individual C-index (>= 0.55) to
        # avoid noise dominating the meta-model.
        good_cols = []
        for c in component_cols:
            v = np.full(n_train, np.nan)
            v[pos] = oof_df[c].to_numpy()
            finite = np.isfinite(v)
            if finite.sum() == 0 or ep.event[finite].sum() == 0:
                continue
            ci = cindex(ep.event[finite], ep.time[finite], v[finite]).cindex
            if np.isfinite(ci) and ci >= 0.55:
                good_cols.append(c)
        _LOG.info("stacking %s: %d components retained (>= 0.55)", endpoint_name, len(good_cols))

        results = {}
        for method in ("ridge", "elastic_net"):
            r = stack_endpoint(
                method, endpoint_name,
                oof_df, test_df, good_cols, splits,
                pos, tpos, n_train, n_test,
                ep.event, ep.time,
            )
            _LOG.info("stack %s/%s OOF C-index = %.4f", endpoint_name, method, r.oof_cindex)
            results[method] = r

        # Persist OOFs/tests for the better method.
        best_method = max(results, key=lambda m: results[m].oof_cindex if np.isfinite(results[m].oof_cindex) else -1)
        best = results[best_method]
        pd.DataFrame({
            pid_col: ds.train_df[pid_col].values,
            f"oof_stack_{endpoint_name}": best.oof,
        }).to_csv(out_root / f"oof_stack_{endpoint_name}.csv", index=False)
        pd.DataFrame({
            tid_col: ds.test_df[tid_col].values,
            f"test_stack_{endpoint_name}": best.test,
        }).to_csv(out_root / f"test_stack_{endpoint_name}.csv", index=False)
        summary[endpoint_name] = {
            "best_method": best_method,
            "components": good_cols,
            "n_components": len(good_cols),
            "results": {
                m: {
                    "oof_cindex": float(r.oof_cindex) if np.isfinite(r.oof_cindex) else None,
                    "fold_mean": float(np.mean(r.fold_scores)) if r.fold_scores else None,
                    "fold_std": float(np.std(r.fold_scores)) if r.fold_scores else None,
                }
                for m, r in results.items()
            },
        }

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    _LOG.info("stacking summary written to %s", out_root / "summary.json")


if __name__ == "__main__":
    main()
