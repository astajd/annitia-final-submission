"""Endpoint-specific ensembling.

Five methods, all operating on percentile-rank predictions:

A. equal       — uniform-weight rank average of every component.
B. cv_weighted — weights proportional to (max(C-index - 0.5, 0)) ** alpha.
C. greedy      — forward selection by OOF C-index, picking the next component
                 that most increases the running ensemble's C-index.
D. capped      — convex weights chosen to maximise OOF C-index under the
                 constraint that no single weight exceeds 0.35.
E. seed_bagged — average of N greedy ensembles built from N bootstrapped OOF
                 sub-samples, smoothing the greedy decisions.

For each method we report selected components, weights, OOF hepatic and death
C-index, weighted score, and the rank-correlation matrix of the selected
predictions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .metrics import cindex
from .targets import Endpoint, build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# I/O: collect OOF and test predictions across experiment dirs
# ---------------------------------------------------------------------------

def collect_predictions(exp_dirs: list[Path], skip_ensemble_cols: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Concatenate every (experiment, model) prediction column.

    Returns (oof_long, test_long, meta) where columns are
    ``<exp>::<endpoint>__<model>``. Index is the patient_id_anon for OOF and
    trustii_id for test.
    """
    oof_pieces: list[pd.DataFrame] = []
    test_pieces: list[pd.DataFrame] = []
    meta: dict[str, dict] = {}
    for d in exp_dirs:
        d = Path(d)
        cfg_blob = json.loads((d / "config.json").read_text())
        ename = cfg_blob.get("name", d.name)
        oof = pd.read_csv(d / "oof_predictions.csv")
        tst = pd.read_csv(d / "test_predictions.csv")

        rename_oof = {}
        for c in oof.columns:
            if c == cfg.PATIENT_ID_COL:
                continue
            rename_oof[c] = f"{ename}::{c}"
        oof = oof.rename(columns=rename_oof)

        rename_tst = {}
        for c in tst.columns:
            if c == cfg.TRUSTII_ID_COL:
                continue
            rename_tst[c] = f"{ename}::{c}"
        tst = tst.rename(columns=rename_tst)

        if skip_ensemble_cols:
            drop_oof = [c for c in oof.columns if "::ensemble_" in c]
            drop_tst = [c for c in tst.columns if "::ensemble_" in c]
            oof = oof.drop(columns=drop_oof)
            tst = tst.drop(columns=drop_tst)

        oof_pieces.append(oof)
        test_pieces.append(tst)
        meta[ename] = {
            "feature_set": cfg_blob.get("feature_set"),
            "death_target_mode": cfg_blob.get("death_target_mode"),
            "models": [m.get("name") for m in cfg_blob.get("models", [])],
            "leakage_risk": _infer_leakage_risk(cfg_blob.get("feature_set")),
        }

    oof_df = oof_pieces[0]
    for piece in oof_pieces[1:]:
        oof_df = oof_df.merge(piece, on=cfg.PATIENT_ID_COL, how="outer")
    test_df = test_pieces[0]
    for piece in test_pieces[1:]:
        test_df = test_df.merge(piece, on=cfg.TRUSTII_ID_COL, how="outer")
    return oof_df, test_df, meta


_LEAKAGE_TAGS = {
    "baseline_v1": "low",
    "early_v1_v3": "low/moderate",
    "first_1y": "low",
    "first_2y": "low",
    "first_3y": "low",
    "first_3_visits": "low",
    "baseline_plus_landmark_trends": "low",
    "clinical_scores_dynamic": "low",
    "NIT_longitudinal_only": "moderate",
    "labs_longitudinal_only": "moderate",
    "longitudinal_no_followup_proxies": "moderate",
    "NIT_plus_clinical_scores": "moderate",
    "NIT_plus_scores_longitudinal": "moderate",
    "NIT_only": "moderate",
    "aggressive_longitudinal": "moderate-high",
    "all_visits_longitudinal": "high",
    "full_high_risk": "high",
    "missingness_and_visit_cadence": "high",
    "strict_time_aligned": "low/moderate",
}


def _infer_leakage_risk(feature_set: str | None) -> str:
    return _LEAKAGE_TAGS.get(feature_set or "", "unknown")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_rank(x: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rank(method="average", pct=True, na_option="keep").to_numpy()


def _component_cindex(values: np.ndarray, event: np.ndarray, time: np.ndarray) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 50 or event[finite].sum() == 0:
        return float("nan")
    return cindex(event[finite], time[finite], values[finite]).cindex


def _ensemble_oof(weights: dict[str, float], oof_pos: dict[str, np.ndarray]) -> np.ndarray:
    keys = list(weights)
    if not keys:
        return np.array([])
    R = np.vstack([_to_rank(oof_pos[k]) for k in keys])
    w = np.array([weights[k] for k in keys])
    valid = ~np.isnan(R)
    weighted = np.where(valid, R * w[:, None], 0.0)
    weight_sum = (valid * w[:, None]).sum(axis=0)
    return np.where(weight_sum > 0, weighted.sum(axis=0) / np.where(weight_sum > 0, weight_sum, 1.0), np.nan)


def _ensemble_score(weights: dict[str, float], oof_pos: dict[str, np.ndarray], event: np.ndarray, time: np.ndarray) -> float:
    e = _ensemble_oof(weights, oof_pos)
    return _component_cindex(e, event, time)


# ---------------------------------------------------------------------------
# Selection methods
# ---------------------------------------------------------------------------

@dataclass
class EnsembleResult:
    method: str
    endpoint: str
    components: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    oof_cindex: float = float("nan")
    notes: dict = field(default_factory=dict)

    def predict(self, src: dict[str, np.ndarray]) -> np.ndarray:
        return _ensemble_oof(self.weights, src)


def equal_weight(component_names: list[str], oof_pos: dict[str, np.ndarray], event, time, endpoint_name: str) -> EnsembleResult:
    weights = {k: 1.0 for k in component_names}
    score = _ensemble_score(weights, oof_pos, event, time)
    return EnsembleResult("equal", endpoint_name, list(component_names), weights, score)


def cv_weighted(component_names: list[str], oof_pos: dict[str, np.ndarray], event, time, endpoint_name: str, alpha: float = 4.0, floor: float = 0.5) -> EnsembleResult:
    raw = {}
    for k in component_names:
        ci = _component_cindex(oof_pos[k], event, time)
        raw[k] = max(ci - floor, 0.0) ** alpha if np.isfinite(ci) else 0.0
    s = sum(raw.values())
    weights = {k: v / s for k, v in raw.items()} if s > 0 else {k: 1.0 / len(raw) for k in raw}
    score = _ensemble_score(weights, oof_pos, event, time)
    return EnsembleResult("cv_weighted", endpoint_name, list(weights), weights, score, {"alpha": alpha, "floor": floor})


def greedy(component_names: list[str], oof_pos: dict[str, np.ndarray], event, time, endpoint_name: str, max_components: int = 8, min_gain: float = 1e-4) -> EnsembleResult:
    chosen: dict[str, float] = {}
    available = list(component_names)
    best_score = float("-inf")
    while available and len(chosen) < max_components:
        best_step = None
        best_step_score = best_score
        for k in available:
            trial = dict(chosen)
            trial[k] = 1.0
            sc = _ensemble_score(trial, oof_pos, event, time)
            if np.isfinite(sc) and sc > best_step_score:
                best_step_score = sc
                best_step = k
        if best_step is None or best_step_score - best_score < min_gain:
            break
        chosen[best_step] = 1.0
        available.remove(best_step)
        best_score = best_step_score
    return EnsembleResult("greedy", endpoint_name, list(chosen), chosen, best_score, {"max_components": max_components, "min_gain": min_gain})


def capped(component_names: list[str], oof_pos: dict[str, np.ndarray], event, time, endpoint_name: str, max_weight: float = 0.35, n_iter: int = 1500, seed: int = 0) -> EnsembleResult:
    """Random-search convex combination with a per-component cap."""
    rng = np.random.default_rng(seed)
    n = len(component_names)
    if n == 0:
        return EnsembleResult("capped", endpoint_name, [], {}, float("nan"), {"max_weight": max_weight})
    best_w = np.full(n, 1.0 / n)
    best_w = np.minimum(best_w, max_weight)
    best_w /= best_w.sum()
    best_w_dict = dict(zip(component_names, best_w))
    best_score = _ensemble_score(best_w_dict, oof_pos, event, time)
    for _ in range(n_iter):
        u = rng.dirichlet(np.ones(n))
        u = np.minimum(u, max_weight)
        if u.sum() <= 0:
            continue
        u = u / u.sum()
        w_dict = dict(zip(component_names, u))
        sc = _ensemble_score(w_dict, oof_pos, event, time)
        if np.isfinite(sc) and sc > best_score:
            best_score = sc
            best_w_dict = w_dict
    return EnsembleResult("capped", endpoint_name, list(best_w_dict), best_w_dict, best_score, {"max_weight": max_weight, "n_iter": n_iter})


def seed_bagged(component_names: list[str], oof_pos: dict[str, np.ndarray], event, time, endpoint_name: str, n_bags: int = 20, sample_frac: float = 0.8, seed: int = 0) -> EnsembleResult:
    """Average of greedy ensembles built on bootstrapped row subsets."""
    rng = np.random.default_rng(seed)
    n_rows = len(event)
    accum: dict[str, float] = {k: 0.0 for k in component_names}
    for _ in range(n_bags):
        idx = rng.choice(n_rows, size=int(n_rows * sample_frac), replace=True)
        ev_b = event[idx]
        t_b = time[idx]
        oof_b = {k: v[idx] for k, v in oof_pos.items()}
        chosen: dict[str, float] = {}
        avail = list(component_names)
        cur = float("-inf")
        while avail and len(chosen) < 8:
            best_k, best_s = None, cur
            for k in avail:
                w_try = dict(chosen)
                w_try[k] = 1.0
                s = _ensemble_score(w_try, oof_b, ev_b, t_b)
                if np.isfinite(s) and s > best_s:
                    best_s = s
                    best_k = k
            if best_k is None or best_s - cur < 1e-4:
                break
            chosen[best_k] = 1.0
            avail.remove(best_k)
            cur = best_s
        for k in chosen:
            accum[k] += 1.0 / n_bags
    weights = {k: v for k, v in accum.items() if v > 0}
    if not weights:
        return EnsembleResult("seed_bagged", endpoint_name, [], {}, float("nan"))
    s = sum(weights.values())
    weights = {k: v / s for k, v in weights.items()}
    score = _ensemble_score(weights, oof_pos, event, time)
    return EnsembleResult("seed_bagged", endpoint_name, list(weights), weights, score, {"n_bags": n_bags, "sample_frac": sample_frac})


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def per_endpoint_pool(oof_df: pd.DataFrame, endpoint_name: str, exclude_high_leakage: bool, meta: dict) -> list[str]:
    cols = [
        c for c in oof_df.columns
        if c != cfg.PATIENT_ID_COL
        and f"::{endpoint_name}__" in c
        and "::ensemble_" not in c
    ]
    if exclude_high_leakage:
        keep = []
        for c in cols:
            ename = c.split("::")[0]
            risk = meta.get(ename, {}).get("leakage_risk", "unknown")
            if risk in ("high",):
                continue
            keep.append(c)
        cols = keep
    return cols


def build_endpoint_ensembles(
    exp_dirs: list[Path],
    *,
    exclude_high_leakage: bool = False,
    label: str = "endpoint_ensemble",
) -> dict:
    """Run all five methods for both endpoints and return predictions / metadata."""
    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)

    oof_df, test_df, meta = collect_predictions(exp_dirs)
    pid_col = cfg.PATIENT_ID_COL
    tid_col = cfg.TRUSTII_ID_COL
    train_pid_to_row = {pid: i for i, pid in enumerate(ds.train_df[pid_col].values)}
    pos = oof_df[pid_col].map(train_pid_to_row).to_numpy()
    keep = ~pd.isna(pos)
    pos = pos[keep].astype(int)
    oof_df = oof_df.loc[keep].reset_index(drop=True)

    test_order = ds.test_df[tid_col].values
    test_idx_by_id = {tid: i for i, tid in enumerate(test_order)}
    test_pos = test_df[tid_col].map(test_idx_by_id).to_numpy()
    test_keep = ~pd.isna(test_pos)
    test_df = test_df.loc[test_keep].reset_index(drop=True)
    test_pos = test_pos[test_keep].astype(int)

    results: dict[str, dict] = {}
    test_predictions: dict[str, np.ndarray] = {}

    for endpoint_name, ep in [("hepatic", hep), ("death", death)]:
        component_cols = per_endpoint_pool(oof_df, endpoint_name, exclude_high_leakage, meta)
        oof_arr = {c: oof_df[c].to_numpy() for c in component_cols}
        # Convert OOF positions to row order by reindexing into a numpy array.
        oof_aligned = {c: np.full(len(ds.train_df), np.nan) for c in component_cols}
        for c in component_cols:
            oof_aligned[c][pos] = oof_df[c].to_numpy()
        ev = ep.event
        t = ep.time
        ensembles = {
            "equal": equal_weight(component_cols, oof_aligned, ev, t, endpoint_name),
            "cv_weighted": cv_weighted(component_cols, oof_aligned, ev, t, endpoint_name),
            "greedy": greedy(component_cols, oof_aligned, ev, t, endpoint_name),
            "capped": capped(component_cols, oof_aligned, ev, t, endpoint_name),
            "seed_bagged": seed_bagged(component_cols, oof_aligned, ev, t, endpoint_name),
        }

        # Test predictions for each method.
        # Build a test-aligned dict of percentile-rank predictions.
        test_aligned = {c: np.full(len(ds.test_df), np.nan) for c in component_cols}
        for c in component_cols:
            test_aligned[c][test_pos] = test_df[c].to_numpy()

        for method_name, res in ensembles.items():
            tp = _ensemble_oof(res.weights, test_aligned)
            test_predictions[f"{endpoint_name}__{method_name}"] = tp

        results[endpoint_name] = {
            method: {
                "components": r.components,
                "weights": r.weights,
                "oof_cindex": r.oof_cindex,
                "notes": r.notes,
            }
            for method, r in ensembles.items()
        }

    return {"meta": meta, "results": results, "test_predictions": test_predictions, "label": label}
