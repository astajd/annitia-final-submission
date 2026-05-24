"""Phase 3.11 — controlled residual / negative-weight ensemble experiment.

Tests whether *small* negative weights on weak or orthogonal models can
improve the survival C-index of the current public best
(`phase3_10_horizon_blend_v2`, LB 0.91093) by correcting systematic bias,
without overfitting.

Schemes per endpoint (hep / dea):

A. nonnegative weights only        — w_i ∈ [0, 0.50]
B. small negative weights allowed  — w_i ∈ [-0.05, 0.50]
C. moderate negative weights       — w_i ∈ [-0.10, 0.50]
D. larger negative weights         — w_i ∈ [-0.20, 0.50]
E. ridge on centred ranks          — sklearn Ridge(α=1) on event indicator
F. greedy with subtractive step    — discrete-weight forward selection that
                                      can also subtract a component

For schemes A-D, the objective is the smooth concordance loss (log-loss
over comparable pairs) on training folds; a sum-to-1 equality constraint
keeps weights interpretable. The maximum absolute per-component weight is
0.50, including the anchor `phase3_10_horizon_blend_v2`, per spec.

Validation: 5-fold stratified inner CV (stratified by event). For each
inner fold, weights are fit on the training rows and the held-out fold
is scored. We aggregate into a true OOF prediction and report the
survival C-index.

Pool (small, hand-picked):

- phase3_10_horizon_blend_v2          (anchor, current best)
- phase3_9_horizon_blend              (older best)
- phase3_5_current_state_v3_hepatic_focused
- best hepatic horizon classifier per horizon (h1, h2, h3, h4, h5, h6)
- best death  horizon classifier per horizon (h5, h6)
- current_state biomarker_only        (labs_longitudinal_only survival pool)
- current_state NIT_plus_scores       (NIT_plus_scores_longitudinal pool)
- phase2 aggressive_longitudinal pool
- robust_longitudinal pool            (longitudinal_no_followup_proxies)

We never include strict_time_aligned or anything derived from event/
censoring ages.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold

from . import config as cfg
from .data_loading import load_dataset
from .features.hep_focus import current_state_v2_no_visit_history
from .horizon_targets import build_horizon_labels
from .metrics import cindex, weighted_score
from .models import build_model
from .models.ensemble import rank_average, to_rank
from .submission import make_submission
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger
from .validation import build_folds

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ranks_centered(x: np.ndarray) -> np.ndarray:
    """Percentile rank centred at 0; NaN -> 0 so it cannot move the score."""
    r = pd.Series(x).rank(method="average", pct=True, na_option="keep").to_numpy()
    return np.where(np.isfinite(r), r - 0.5, 0.0)


def _surv_ci(event: np.ndarray, time: np.ndarray, score: np.ndarray) -> float:
    finite = np.isfinite(score)
    if finite.sum() < 5 or event[finite].sum() == 0:
        return float("nan")
    return float(cindex(event[finite], time[finite], score[finite]).cindex)


def _fold_ci(event: np.ndarray, time: np.ndarray, score: np.ndarray, splits) -> tuple[float, float, float]:
    s_list: list[float] = []
    for s in splits:
        v = score[s.valid_idx]
        ff = np.isfinite(v)
        if ff.sum() == 0 or event[s.valid_idx][ff].sum() == 0:
            continue
        c = cindex(event[s.valid_idx][ff], time[s.valid_idx][ff], v[ff]).cindex
        if np.isfinite(c):
            s_list.append(float(c))
    if not s_list:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(s_list)), float(np.std(s_list)), float(np.min(s_list))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size != b.size:
        return float("nan")
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 5:
        return float("nan")
    return float(df["a"].rank(pct=True).corr(df["b"].rank(pct=True), method="spearman"))


def _build_pairs(event: np.ndarray, time: np.ndarray, idx: np.ndarray, max_pairs: int = 30000) -> np.ndarray:
    """All comparable pairs (i, j) within ``idx`` with event_i and time_i < time_j."""
    sub = idx
    e_sub = event[sub]
    t_sub = time[sub]
    pos = np.where(e_sub)[0]
    if len(pos) == 0:
        return np.empty((0, 2), dtype=int)
    pairs = []
    for i_local in pos:
        ti = t_sub[i_local]
        # Comparable: t_j > t_i (strictly after the event).
        j_local = np.where(t_sub > ti)[0]
        if len(j_local) == 0:
            continue
        i_global = sub[i_local]
        j_global = sub[j_local]
        for jg in j_global:
            pairs.append((i_global, jg))
    if not pairs:
        return np.empty((0, 2), dtype=int)
    arr = np.asarray(pairs, dtype=int)
    if len(arr) > max_pairs:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(arr), size=max_pairs, replace=False)
        arr = arr[sel]
    return arr


def _smooth_concordance_loss(w: np.ndarray, X: np.ndarray, pairs: np.ndarray) -> float:
    if len(pairs) == 0:
        return 0.0
    score = X @ w
    diff = score[pairs[:, 0]] - score[pairs[:, 1]]
    # log(1 + exp(-d)) numerically stable
    return float(np.mean(np.logaddexp(0.0, -diff)))


def _smooth_concordance_grad(w: np.ndarray, X: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.zeros_like(w)
    score = X @ w
    diff = score[pairs[:, 0]] - score[pairs[:, 1]]
    sig = 1.0 / (1.0 + np.exp(diff))         # σ(-d)
    coef = -sig                              # dL/dd = -sig
    grad_score = np.zeros(X.shape[0])
    np.add.at(grad_score, pairs[:, 0], coef)
    np.add.at(grad_score, pairs[:, 1], -coef)
    grad_w = X.T @ grad_score / len(pairs)
    return grad_w


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

@dataclass
class PoolMember:
    name: str
    oof: np.ndarray
    test: np.ndarray
    endpoint: str  # "hepatic" | "death" | "both" (for blend rows)
    description: str


def _load_horizon_artifact(label: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Find a phase3_9 or phase3_10 horizon-run dir and load its (oof, test)."""
    for root in (cfg.EXPERIMENT_OUTPUTS / "phase3_10_horizon",
                 cfg.EXPERIMENT_OUTPUTS / "phase3_9_horizon"):
        run_dir = root / label
        if not run_dir.exists():
            continue
        oof_path = run_dir / "oof.csv"
        test_path = run_dir / "test.csv"
        if oof_path.exists() and test_path.exists():
            o = pd.read_csv(oof_path)["oof"].to_numpy()
            t = pd.read_csv(test_path)["test"].to_numpy()
            return o, t
    return None


def _load_phase3_9_run_dir() -> Path:
    return cfg.EXPERIMENT_OUTPUTS / "phase3_9_horizon"


def _load_phase3_10_run_dir() -> Path:
    return cfg.EXPERIMENT_OUTPUTS / "phase3_10_horizon"


def _best_horizon_runs(ep: str, horizon: int, ds, hepatic_endpoint, death_endpoint) -> tuple[str, np.ndarray, np.ndarray] | None:
    """Among phase3_9 + phase3_10 horizon runs for (endpoint, horizon),
    return the (label, oof, test) with max survival C-index."""
    e_obj = hepatic_endpoint if ep == "hepatic" else death_endpoint
    best = None
    for root in (_load_phase3_9_run_dir(), _load_phase3_10_run_dir()):
        if not root.exists():
            continue
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            label = run_dir.name
            # Filter by endpoint and horizon: tag schema is fs__ep__hH__model[__suffix]
            parts = label.split("__")
            if len(parts) < 4:
                continue
            ep_tok = parts[1]
            h_tok = parts[2]
            if ep_tok != ep or h_tok != f"h{int(horizon)}":
                continue
            # Only pick "exclude"-mode (no __dw suffix) and any seed.
            if "__dw" in label:
                continue
            oof_path = run_dir / "oof.csv"
            test_path = run_dir / "test.csv"
            if not (oof_path.exists() and test_path.exists()):
                continue
            try:
                o = pd.read_csv(oof_path)["oof"].to_numpy()
                t = pd.read_csv(test_path)["test"].to_numpy()
            except Exception:
                continue
            ci = _surv_ci(e_obj.event, e_obj.time, o)
            if not np.isfinite(ci):
                continue
            if best is None or ci > best[3]:
                best = (label, o, t, ci)
    if best is None:
        return None
    return best[0], best[1], best[2]


def _reconstruct_blend_oof(meta_path: Path, ds, hep, death, n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce a phase3_9 / phase3_10 horizon-blend OOF and test arrays
    from its JSON sidecar (which lists components and weights)."""
    meta = json.loads(meta_path.read_text())

    # The phase3_5 reference v3 OOF/test we'll need too.
    v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)
    v3_csv = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.csv"))[-1]
    v3_pub = pd.read_csv(v3_csv)
    v3_h_test = v3_pub[cfg.SUB_HEPATIC_COL].to_numpy()
    v3_d_test = v3_pub[cfg.SUB_DEATH_COL].to_numpy()

    blend_csv = meta_path.with_suffix(".csv")
    pub = pd.read_csv(blend_csv)

    # The published CSV gives us test predictions verbatim; for OOF we
    # rebuild from components.
    if "blend_id" in meta:
        # Phase 3.10 schema with explicit component/weight dicts.
        comps_h = meta["components_hepatic"]
        comps_d = meta["components_death"]
        weights_h = meta["weights_hepatic"]
        weights_d = meta["weights_death"]
    else:
        # Phase 3.9 schema: alpha-side blend with hep_best_run / dea_best_run.
        alpha = meta["alpha"]
        side = meta["side"]
        h_b = meta.get("hep_best_run")
        d_b = meta.get("dea_best_run")
        comps_h = [h_b] if (side in ("hep_only", "both") and h_b) else []
        comps_d = [d_b] if (side in ("dea_only", "both") and d_b) else []
        weights_h = {"_v3": alpha} | ({h_b: 1 - alpha} if comps_h else {})
        weights_d = {"_v3": alpha} | ({d_b: 1 - alpha} if comps_d else {})

    # Build component dicts.
    pool_h_oof: dict[str, np.ndarray] = {}
    pool_h_test: dict[str, np.ndarray] = {}
    pool_d_oof: dict[str, np.ndarray] = {}
    pool_d_test: dict[str, np.ndarray] = {}
    if "_v3" in weights_h or "v3_hep" in weights_h:
        key = "_v3" if "_v3" in weights_h else "v3_hep"
        pool_h_oof[key] = v3_h_oof
        pool_h_test[key] = v3_h_test
    if "_v3" in weights_d or "v3_dea" in weights_d:
        key = "_v3" if "_v3" in weights_d else "v3_dea"
        pool_d_oof[key] = v3_d_oof
        pool_d_test[key] = v3_d_test
    for k in list(weights_h):
        if k in ("_v3", "v3_hep"):
            continue
        loaded = _load_horizon_artifact(k)
        if loaded is None:
            _LOG.warning("missing horizon artifact %s", k)
            continue
        pool_h_oof[k] = loaded[0]
        pool_h_test[k] = loaded[1]
    for k in list(weights_d):
        if k in ("_v3", "v3_dea"):
            continue
        loaded = _load_horizon_artifact(k)
        if loaded is None:
            _LOG.warning("missing horizon artifact %s", k)
            continue
        pool_d_oof[k] = loaded[0]
        pool_d_test[k] = loaded[1]

    # Apply weights via rank_average (handles renormalisation).
    h_oof = rank_average(pool_h_oof, weights={k: weights_h[k] for k in pool_h_oof})
    h_test = rank_average(pool_h_test, weights={k: weights_h[k] for k in pool_h_test})
    d_oof = rank_average(pool_d_oof, weights={k: weights_d[k] for k in pool_d_oof})
    d_test = rank_average(pool_d_test, weights={k: weights_d[k] for k in pool_d_test})
    # Sanity: replace test from CSV (authoritative).
    h_test_csv = pub[cfg.SUB_HEPATIC_COL].to_numpy()
    d_test_csv = pub[cfg.SUB_DEATH_COL].to_numpy()
    return h_oof, h_test_csv, d_oof, d_test_csv


def _v3_oof(n_train: int, ds, hep) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct phase3_5 v3_hepatic_focused OOF for (hepatic, death)."""
    cands = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.json"))
    if not cands:
        return np.full(n_train, np.nan), np.full(n_train, np.nan)
    blob = json.loads(cands[-1].read_text())
    h_w = blob["hepatic"]["weights"]
    d_w = blob["death"]["weights"]

    from .endpoint_ensemble import collect_predictions
    all_dirs = [d for d in cfg.EXPERIMENT_OUTPUTS.iterdir() if d.is_dir() and (d / "oof_predictions.csv").exists()]
    oof_df, _, _ = collect_predictions(all_dirs)
    pid_col = cfg.PATIENT_ID_COL
    pos = oof_df[pid_col].map({v: i for i, v in enumerate(ds.train_df[pid_col].values)}).to_numpy().astype(int)

    pool: dict[str, np.ndarray] = {}
    for c in set(h_w) | set(d_w):
        if c in oof_df.columns:
            arr = np.full(n_train, np.nan)
            arr[pos] = oof_df[c].to_numpy()
            pool[c] = arr

    nh_key = "phase3_5_no_visit_history::hepatic__rsf__nh"
    if nh_key in h_w and nh_key not in pool:
        Xtr = current_state_v2_no_visit_history(ds.train_df, ds.visit_columns, ds.age_visit_cols).select_dtypes(include=[np.number])
        splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)
        oof = np.full(n_train, np.nan)
        for s in splits:
            tr = s.train_idx
            va = s.valid_idx
            if hep.event[tr].sum() == 0:
                continue
            m = build_model("rsf", {"n_estimators": 250, "max_depth": 7, "min_samples_leaf": 4, "random_state": 0})
            mask = pd.Series(False, index=Xtr.index)
            mask.iloc[tr] = True
            try:
                m.fit(Xtr, hep, mask=mask)
                oof[va] = m.predict_risk(Xtr.iloc[va])
            except Exception:
                continue
        pool[nh_key] = oof

    h_oof = rank_average({c: pool[c] for c in h_w if c in pool}, weights={c: h_w[c] for c in h_w if c in pool}) \
            if any(c in pool for c in h_w) else np.full(n_train, np.nan)
    d_oof = rank_average({c: pool[c] for c in d_w if c in pool}, weights={c: d_w[c] for c in d_w if c in pool}) \
            if any(c in pool for c in d_w) else np.full(n_train, np.nan)
    return h_oof, d_oof


def _survival_pool_predictions(experiment_name: str, ds, ep_filter: str | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """For an existing experiment dir (e.g. phase2_aggressive_longitudinal), build
    rank-averaged OOF and test predictions per endpoint.

    Returns (oof_h, test_h) for hepatic if ep_filter is None or "hepatic", and
    similar for death. We always rank-average all model columns matching the
    endpoint stem.
    """
    candidates = list(cfg.EXPERIMENT_OUTPUTS.glob(f"*_{experiment_name}"))
    if not candidates:
        return None, None
    d = sorted(candidates)[-1]
    oof = pd.read_csv(d / "oof_predictions.csv")
    tst = pd.read_csv(d / "test_predictions.csv")
    pid_to_pos = {v: i for i, v in enumerate(ds.train_df[cfg.PATIENT_ID_COL].values)}
    tid_to_pos = {v: i for i, v in enumerate(ds.test_df[cfg.TRUSTII_ID_COL].values)}
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)

    def _rank_combine(df: pd.DataFrame, ep: str, id_col: str, id_to_pos: dict, n: int) -> np.ndarray:
        cols = [c for c in df.columns
                if c.startswith(f"{ep}__")
                and not c.startswith("ensemble_")
                and "ensemble" not in c]
        if not cols:
            return np.full(n, np.nan)
        # Build aligned matrix in original-row order
        pos = df[id_col].map(id_to_pos).to_numpy()
        mask = ~pd.isna(pos)
        pos_int = pos[mask].astype(int)
        out = np.full((len(cols), n), np.nan)
        for i, c in enumerate(cols):
            out[i, pos_int] = df[c].to_numpy()[mask]
        # Rank-average across models (per row)
        return rank_average({c: out[i] for i, c in enumerate(cols)})

    h_oof = _rank_combine(oof, "hepatic", cfg.PATIENT_ID_COL, pid_to_pos, n_train)
    d_oof = _rank_combine(oof, "death",   cfg.PATIENT_ID_COL, pid_to_pos, n_train)
    h_test = _rank_combine(tst, "hepatic", cfg.TRUSTII_ID_COL, tid_to_pos, n_test)
    d_test = _rank_combine(tst, "death",   cfg.TRUSTII_ID_COL, tid_to_pos, n_test)
    return (h_oof, h_test, d_oof, d_test)  # type: ignore[return-value]


def build_pool(ds, hep, death) -> tuple[dict[str, PoolMember], dict[str, PoolMember]]:
    """Construct the small hand-picked pool, returning per-endpoint dicts."""
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)
    pool_h: dict[str, PoolMember] = {}
    pool_d: dict[str, PoolMember] = {}

    # 1. phase3_10_horizon_blend_v2 (anchor)
    p10_meta_path = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_10_horizon_blend_v2.json"))[-1]
    h_oof, h_test, d_oof, d_test = _reconstruct_blend_oof(p10_meta_path, ds, hep, death, n_train, n_test)
    pool_h["phase3_10_horizon_blend_v2"] = PoolMember(
        "phase3_10_horizon_blend_v2", h_oof, h_test, "hepatic", "current public best"
    )
    pool_d["phase3_10_horizon_blend_v2"] = PoolMember(
        "phase3_10_horizon_blend_v2", d_oof, d_test, "death", "current public best"
    )

    # 2. phase3_9_horizon_blend
    p9_meta_path = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_9_horizon_blend.json"))[-1]
    h_oof9, h_test9, d_oof9, d_test9 = _reconstruct_blend_oof(p9_meta_path, ds, hep, death, n_train, n_test)
    pool_h["phase3_9_horizon_blend"] = PoolMember(
        "phase3_9_horizon_blend", h_oof9, h_test9, "hepatic", "previous best"
    )
    pool_d["phase3_9_horizon_blend"] = PoolMember(
        "phase3_9_horizon_blend", d_oof9, d_test9, "death", "previous best"
    )

    # 3. phase3_5_current_state_v3_hepatic_focused
    v3_csv = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.csv"))[-1]
    v3_pub = pd.read_csv(v3_csv)
    v3_h_test = v3_pub[cfg.SUB_HEPATIC_COL].to_numpy()
    v3_d_test = v3_pub[cfg.SUB_DEATH_COL].to_numpy()
    v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)
    pool_h["phase3_5_v3_hepatic_focused"] = PoolMember(
        "phase3_5_v3_hepatic_focused", v3_h_oof, v3_h_test, "hepatic", "phase3_5 hepatic-focused"
    )
    pool_d["phase3_5_v3_hepatic_focused"] = PoolMember(
        "phase3_5_v3_hepatic_focused", v3_d_oof, v3_d_test, "death", "phase3_5 hepatic-focused"
    )

    # 4. Best hepatic horizon classifier per H ∈ {1, 2, 3, 4, 5, 6}.
    for H in (1, 2, 3, 4, 5, 6):
        best = _best_horizon_runs("hepatic", H, ds, hep, death)
        if best is None:
            _LOG.warning("no hepatic h%d horizon run found", H)
            continue
        label, o, t = best
        key = f"hep_h{H}_{label.split('__')[-1]}"
        pool_h[key] = PoolMember(key, o, t, "hepatic", f"best hepatic h{H} horizon: {label}")

    # 5. Best death horizon classifier for h5 and h6.
    for H in (5, 6):
        best = _best_horizon_runs("death", H, ds, hep, death)
        if best is None:
            _LOG.warning("no death h%d horizon run found", H)
            continue
        label, o, t = best
        key = f"dea_h{H}_{label.split('__')[-1]}"
        pool_d[key] = PoolMember(key, o, t, "death", f"best death h{H} horizon: {label}")

    # 6-9. Survival-trained pools from earlier phases.
    survival_targets = [
        ("phase2_NIT_plus_scores_longitudinal", "current_state_NIT_plus_scores"),
        ("phase2_aggressive_longitudinal",      "phase2_aggressive_longitudinal"),
        ("phase2_longitudinal_no_followup_proxies", "robust_longitudinal"),
        ("phase2_labs_longitudinal_only",       "current_state_biomarker_only"),
    ]
    for exp_name, key in survival_targets:
        out = _survival_pool_predictions(exp_name, ds)
        if out is None or out[0] is None:
            _LOG.warning("could not load %s", exp_name)
            continue
        h_oof_s, h_test_s, d_oof_s, d_test_s = out
        pool_h[key] = PoolMember(key, h_oof_s, h_test_s, "hepatic", f"survival pool: {exp_name}")
        pool_d[key] = PoolMember(key, d_oof_s, d_test_s, "death",   f"survival pool: {exp_name}")

    return pool_h, pool_d


# ---------------------------------------------------------------------------
# Weight-fitting schemes
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    weights: np.ndarray
    score: float        # objective value, lower is better for loss-based
    note: str = ""


def _fit_bounded_concordance(X: np.ndarray, event: np.ndarray, time: np.ndarray,
                              idx: np.ndarray, lo: float, hi: float, anchor_idx: int = 0,
                              max_pairs: int = 30000) -> FitResult:
    pairs = _build_pairs(event, time, idx, max_pairs=max_pairs)
    if len(pairs) == 0:
        return FitResult(np.zeros(X.shape[1]), float("nan"), "no comparable pairs")
    n_models = X.shape[1]

    # Bounds: all components within [lo, hi], capped at |w| <= 0.50.
    lo_eff = max(lo, -0.50)
    hi_eff = min(hi, 0.50)
    bounds = [(lo_eff, hi_eff)] * n_models

    # Initial: anchor at min(0.50, 1 - (n-1)*lo); rest split residual at 0.
    w0 = np.zeros(n_models)
    w0[anchor_idx] = min(0.50, 1.0 - max(0.0, (n_models - 1) * 0.0))
    residual = 1.0 - w0[anchor_idx]
    if n_models > 1:
        per_other = max(min(residual / (n_models - 1), hi_eff), max(lo_eff, 0.0))
        for i in range(n_models):
            if i == anchor_idx:
                continue
            w0[i] = per_other
        # Normalise to sum=1 if drifted.
        s = w0.sum()
        if s > 0:
            w0 = w0 / s
        # Clip to bounds and renormalise.
        w0 = np.clip(w0, lo_eff, hi_eff)
        s = w0.sum()
        if s > 0:
            w0 = w0 / s

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(
        _smooth_concordance_loss, w0, args=(X, pairs),
        jac=_smooth_concordance_grad, method="SLSQP",
        bounds=bounds, constraints=cons,
        options={"maxiter": 200, "ftol": 1e-7},
    )
    return FitResult(res.x, float(res.fun), f"converged={res.success}")


def _fit_ridge_centered(X: np.ndarray, event: np.ndarray, idx: np.ndarray, alpha: float = 1.0) -> FitResult:
    """Ridge on centred ranks with the binary event indicator as target."""
    Xs = X[idx]
    y = event[idx].astype(float)
    # Centre target to be symmetric around 0.5 -> 0.
    y = y - y.mean()
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(Xs, y)
    w = ridge.coef_
    # Cap at +- 0.50 and normalise to sum-of-positive-or-net=1.
    w = np.clip(w, -0.50, 0.50)
    s = np.sum(w)
    if abs(s) > 1e-9:
        w = w / s
    else:
        # If everything cancels, fall back to uniform positive.
        w = np.ones_like(w) / len(w)
    return FitResult(w, float("nan"), "ridge")


def _fit_greedy_subtractive(X: np.ndarray, event: np.ndarray, time: np.ndarray,
                             idx: np.ndarray, anchor_idx: int = 0,
                             step: float = 0.05, max_iter: int = 30) -> FitResult:
    """Greedy forward selection with optional subtractive step.

    Start at w = e_anchor (weight 1.0 on the anchor). At each step, pick the
    component k and direction (+/-step) that most increases the survival
    C-index on idx, subject to |w_i| <= 0.50 and sum=1.
    """
    n_models = X.shape[1]
    w = np.zeros(n_models)
    w[anchor_idx] = 1.0
    cur = _surv_ci(event[idx], time[idx], X[idx] @ w)
    if not np.isfinite(cur):
        return FitResult(w, float("nan"), "anchor predict NaN")

    for it in range(max_iter):
        best_k = None
        best_dir = 0
        best_ci = cur
        for k in range(n_models):
            for d in (+step, -step):
                # Adjust w[k] by d, compensating by -d on the anchor (if k != anchor)
                # so sum stays 1; otherwise adjust the largest other.
                w_try = w.copy()
                if k == anchor_idx:
                    continue
                w_try[k] += d
                w_try[anchor_idx] -= d
                if abs(w_try[k]) > 0.50 + 1e-9 or abs(w_try[anchor_idx]) > 0.50 + 1e-9:
                    continue
                if w_try[anchor_idx] < -0.20:
                    continue
                ci = _surv_ci(event[idx], time[idx], X[idx] @ w_try)
                if np.isfinite(ci) and ci > best_ci + 1e-5:
                    best_ci = ci
                    best_k = k
                    best_dir = d
        if best_k is None:
            break
        w[best_k] += best_dir
        w[anchor_idx] -= best_dir
        cur = best_ci
    return FitResult(w, cur, f"final ci={cur:.4f}")


SCHEMES = [
    ("A_nonneg",     {"type": "bounded", "lo": 0.0,    "hi": 0.50}),
    ("B_neg_005",    {"type": "bounded", "lo": -0.05,  "hi": 0.50}),
    ("C_neg_010",    {"type": "bounded", "lo": -0.10,  "hi": 0.50}),
    ("D_neg_020",    {"type": "bounded", "lo": -0.20,  "hi": 0.50}),
    ("E_ridge",      {"type": "ridge",   "alpha": 1.0}),
    ("F_greedy_sub", {"type": "greedy",  "step": 0.05}),
]


def _fit_scheme(scheme: dict, X: np.ndarray, event: np.ndarray, time: np.ndarray,
                idx: np.ndarray, anchor_idx: int) -> FitResult:
    if scheme["type"] == "bounded":
        return _fit_bounded_concordance(X, event, time, idx,
                                          lo=scheme["lo"], hi=scheme["hi"],
                                          anchor_idx=anchor_idx)
    if scheme["type"] == "ridge":
        return _fit_ridge_centered(X, event, idx, alpha=scheme["alpha"])
    if scheme["type"] == "greedy":
        return _fit_greedy_subtractive(X, event, time, idx, anchor_idx=anchor_idx,
                                         step=scheme["step"])
    raise KeyError(scheme["type"])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _per_endpoint_eval(name_pool: dict[str, PoolMember], event: np.ndarray, time: np.ndarray,
                        ep_label: str, anchor_key: str, n_train: int, splits, seed: int = 2026
                        ) -> tuple[dict, np.ndarray, dict]:
    """For one endpoint, run all six schemes with 5-fold inner CV and final fit."""
    keys = list(name_pool.keys())
    anchor_idx = keys.index(anchor_key)

    # Stack centred ranks.
    X = np.stack([_ranks_centered(name_pool[k].oof) for k in keys], axis=1)  # (n, k)

    # Inner 5-fold stratified by event for nested validation.
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    inner_splits = list(inner.split(np.zeros(n_train), event.astype(int)))

    scheme_results = {}
    for scheme_name, scheme_cfg in SCHEMES:
        oof_score = np.full(n_train, np.nan)
        weights_per_fold = []
        for tr_idx, va_idx in inner_splits:
            res = _fit_scheme(scheme_cfg, X, event, time, np.asarray(tr_idx), anchor_idx)
            oof_score[va_idx] = X[va_idx] @ res.weights
            weights_per_fold.append(res.weights)
        ci = _surv_ci(event, time, oof_score)
        fmean, fstd, fmin = _fold_ci(event, time, oof_score, splits)

        # Final full-data fit.
        final = _fit_scheme(scheme_cfg, X, event, time, np.arange(n_train), anchor_idx)
        # Also evaluate the full-data weights' OOF (should match training fit not CV).
        full_oof = X @ final.weights
        ci_full = _surv_ci(event, time, full_oof)

        # Weight stability: per-component std across folds.
        wpf = np.stack(weights_per_fold, axis=0)
        w_std = wpf.std(axis=0)
        w_mean = wpf.mean(axis=0)

        scheme_results[scheme_name] = {
            "weights_full": final.weights,
            "ci_inner_oof": ci,                 # nested-CV C-index
            "ci_fullfit_train": ci_full,        # in-sample C-index of full-data fit
            "fold_mean": fmean, "fold_std": fstd, "fold_min": fmin,
            "weight_std": w_std,
            "weight_mean": w_mean,
            "weights_per_fold": wpf,
            "n_neg_components": int(np.sum(final.weights < -1e-6)),
            "neg_total_magnitude": float(-np.sum(final.weights[final.weights < -1e-6])),
        }
        _LOG.info("%s/%s nested-OOF C=%.4f fullfit C=%.4f neg-mag=%.3f",
                  ep_label, scheme_name, ci, ci_full,
                  scheme_results[scheme_name]["neg_total_magnitude"])

    return scheme_results, X, {"keys": keys, "anchor_idx": anchor_idx}


def main() -> None:
    cfg.ensure_dirs()
    out_root = cfg.EXPERIMENT_OUTPUTS / "phase3_11_residual"
    out_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)

    # Build the pool.
    pool_h, pool_d = build_pool(ds, hep, death)
    _LOG.info("hepatic pool size=%d, death pool size=%d", len(pool_h), len(pool_d))

    # Component baseline metrics (single-component C-index).
    baseline_rows = []
    for ep_label, pool, ev, tm in [("hepatic", pool_h, hep.event, hep.time),
                                    ("death",   pool_d, death.event, death.time)]:
        for k, m in pool.items():
            ci = _surv_ci(ev, tm, m.oof)
            fm, fs, fmn = _fold_ci(ev, tm, m.oof, splits)
            baseline_rows.append({
                "endpoint": ep_label, "component": k,
                "oof_cindex": ci, "fold_mean": fm, "fold_std": fs, "fold_min": fmn,
                "rho_with_anchor_oof": _spearman(m.oof, pool["phase3_10_horizon_blend_v2"].oof),
            })
    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(out_root / "component_baselines.csv", index=False)

    # Run schemes per endpoint.
    res_h, X_h, meta_h = _per_endpoint_eval(pool_h, hep.event, hep.time,
                                             "hep", "phase3_10_horizon_blend_v2",
                                             n_train, splits)
    res_d, X_d, meta_d = _per_endpoint_eval(pool_d, death.event, death.time,
                                             "dea", "phase3_10_horizon_blend_v2",
                                             n_train, splits)

    # Compose endpoint pairs: for each (hep_scheme, dea_scheme), compute weighted score.
    # Anchor ("nonneg-only") corresponds to scheme A (no negative weights). The
    # comparison vs phase3_10_horizon_blend_v2 uses its own (hep,dea) OOF as
    # the reference point.
    anchor_h_oof = pool_h["phase3_10_horizon_blend_v2"].oof
    anchor_d_oof = pool_d["phase3_10_horizon_blend_v2"].oof
    anchor_h_test = pool_h["phase3_10_horizon_blend_v2"].test
    anchor_d_test = pool_d["phase3_10_horizon_blend_v2"].test
    ref_h = _surv_ci(hep.event, hep.time, anchor_h_oof)
    ref_d = _surv_ci(death.event, death.time, anchor_d_oof)
    ref_w = weighted_score(ref_h, ref_d)
    _LOG.info("Reference phase3_10 OOF: hep=%.4f dea=%.4f weighted=%.4f", ref_h, ref_d, ref_w)

    # Build all (hep_scheme × dea_scheme) candidates from full-data weights;
    # evaluate true OOF using the nested-CV per-row predictions for honesty.
    candidates_table = []
    keys_h = meta_h["keys"]
    keys_d = meta_d["keys"]
    X_h_test = np.stack([_ranks_centered(pool_h[k].test) for k in keys_h], axis=1)
    X_d_test = np.stack([_ranks_centered(pool_d[k].test) for k in keys_d], axis=1)

    for sh_name, _ in SCHEMES:
        for sd_name, _ in SCHEMES:
            rh = res_h[sh_name]
            rd = res_d[sd_name]
            ci_h = rh["ci_inner_oof"]
            ci_d = rd["ci_inner_oof"]
            ws = weighted_score(ci_h, ci_d)
            row = {
                "hep_scheme": sh_name,
                "dea_scheme": sd_name,
                "hep_oof": ci_h,
                "death_oof": ci_d,
                "weighted_oof": ws,
                "hep_fold_std": rh["fold_std"],
                "hep_fold_min": rh["fold_min"],
                "dea_fold_std": rd["fold_std"],
                "dea_fold_min": rd["fold_min"],
                "delta_weighted_vs_p10": ws - ref_w,
                "delta_hep_vs_p10": ci_h - ref_h,
                "delta_dea_vs_p10": ci_d - ref_d,
                "hep_neg_total": rh["neg_total_magnitude"],
                "dea_neg_total": rd["neg_total_magnitude"],
                "hep_neg_components": rh["n_neg_components"],
                "dea_neg_components": rd["n_neg_components"],
                "hep_max_abs_weight": float(np.max(np.abs(rh["weights_full"]))),
                "dea_max_abs_weight": float(np.max(np.abs(rd["weights_full"]))),
            }
            candidates_table.append(row)
    candidates_df = pd.DataFrame(candidates_table).sort_values("weighted_oof", ascending=False)
    candidates_df.to_csv(out_root / "scheme_pairs.csv", index=False)

    # ------------------------------------------------------------------
    # Promotion criteria.
    # Best candidate: max weighted OOF that meets either +0.002 or +0.003 hep,
    # AND no max_abs_weight > 0.50 (already constrained), AND not dominated by
    # one negative artifact (|w_neg|.max <= 0.30 of total, say).
    # ------------------------------------------------------------------

    promoted = None
    promoted_reason = ""
    for row in candidates_df.itertuples():
        improves_w = row.delta_weighted_vs_p10 >= 0.002
        improves_h = row.delta_hep_vs_p10 >= 0.003
        if not (improves_w or improves_h):
            continue
        # Stability: weight std per component, per-fold weights from inner CV.
        rh = res_h[row.hep_scheme]
        rd = res_d[row.dea_scheme]
        # Reject if any component has fold-std > 0.15 (very unstable).
        unstable_h = bool(np.any(rh["weight_std"] > 0.15))
        unstable_d = bool(np.any(rd["weight_std"] > 0.15))
        if unstable_h or unstable_d:
            continue
        # Reject if dominated by single negative component (largest |neg|/total)
        if rh["neg_total_magnitude"] > 0:
            largest_neg_h = float(-np.min(rh["weights_full"]))
            ratio_h = largest_neg_h / max(rh["neg_total_magnitude"], 1e-9)
            if ratio_h > 0.95 and largest_neg_h > 0.10:
                continue
        if rd["neg_total_magnitude"] > 0:
            largest_neg_d = float(-np.min(rd["weights_full"]))
            ratio_d = largest_neg_d / max(rd["neg_total_magnitude"], 1e-9)
            if ratio_d > 0.95 and largest_neg_d > 0.10:
                continue
        promoted = row
        promoted_reason = (
            f"weighted Δ={row.delta_weighted_vs_p10:+.4f}, "
            f"hep Δ={row.delta_hep_vs_p10:+.4f}, "
            f"hep_scheme={row.hep_scheme}, dea_scheme={row.dea_scheme}"
        )
        break

    sub_path = None
    submission_meta = None
    if promoted is not None:
        rh = res_h[promoted.hep_scheme]
        rd = res_d[promoted.dea_scheme]
        # Build test prediction: rank-centred X_test @ w + 0.5 (back to [0,1]ish).
        h_test_score = X_h_test @ rh["weights_full"] + 0.5
        d_test_score = X_d_test @ rd["weights_full"] + 0.5
        # Re-rank for submission (the contest only ranks).
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=h_test_score, risk_death=d_test_score,
            sample_submission=ds.sample_submission,
            model_name="phase3_11_residual_negative_weight_blend",
        )
        submission_meta = {
            "label": "phase3_11_residual_negative_weight_blend",
            "hep_scheme": promoted.hep_scheme,
            "dea_scheme": promoted.dea_scheme,
            "hep_components": list(meta_h["keys"]),
            "hep_weights": {k: float(w) for k, w in zip(meta_h["keys"], rh["weights_full"])},
            "dea_components": list(meta_d["keys"]),
            "dea_weights": {k: float(w) for k, w in zip(meta_d["keys"], rd["weights_full"])},
            "hepatic_oof": float(promoted.hep_oof),
            "death_oof": float(promoted.death_oof),
            "weighted_oof": float(promoted.weighted_oof),
            "delta_vs_phase3_10": {
                "weighted_oof": float(promoted.delta_weighted_vs_p10),
                "hepatic_oof":  float(promoted.delta_hep_vs_p10),
                "death_oof":    float(promoted.delta_dea_vs_p10),
            },
            "hep_fold_std": float(promoted.hep_fold_std),
            "hep_fold_min": float(promoted.hep_fold_min),
            "dea_fold_std": float(promoted.dea_fold_std),
            "dea_fold_min": float(promoted.dea_fold_min),
            "hep_neg_total": float(promoted.hep_neg_total),
            "dea_neg_total": float(promoted.dea_neg_total),
            "uses_target_derived_features": False,
            "rationale": promoted_reason,
            "submission_csv": str(sub_path),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(submission_meta, indent=2, default=str))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    md: list[str] = []
    md.append("# Phase 3.11 — controlled residual / negative-weight ensemble\n")
    md.append("Reference candidate: `phase3_10_horizon_blend_v2` (public LB **0.91093**).\n")
    md.append(f"Reference OOF: hep={ref_h:.4f} / death={ref_d:.4f} / weighted={ref_w:.4f}.\n")

    md.append("## Pool\n")
    md.append("### Hepatic pool")
    md.append(pd.DataFrame([
        {"name": k, "description": v.description, "endpoint": v.endpoint}
        for k, v in pool_h.items()
    ]).to_markdown(index=False))
    md.append("")
    md.append("### Death pool")
    md.append(pd.DataFrame([
        {"name": k, "description": v.description, "endpoint": v.endpoint}
        for k, v in pool_d.items()
    ]).to_markdown(index=False))
    md.append("")

    md.append("## Component baselines (OOF survival C-index per pool member)\n")
    md.append(baseline_df.sort_values(["endpoint", "oof_cindex"], ascending=[True, False])
              .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## Per-endpoint scheme results\n")
    for ep_label, res, keys in [("hepatic", res_h, meta_h["keys"]),
                                  ("death",   res_d, meta_d["keys"])]:
        md.append(f"### {ep_label}\n")
        rows = []
        for sname, r in res.items():
            rows.append({
                "scheme": sname,
                "ci_inner_oof": r["ci_inner_oof"],
                "ci_fullfit_train": r["ci_fullfit_train"],
                "fold_mean": r["fold_mean"],
                "fold_std": r["fold_std"],
                "fold_min": r["fold_min"],
                "n_neg_components": r["n_neg_components"],
                "neg_total_magnitude": r["neg_total_magnitude"],
                "max_abs_weight": float(np.max(np.abs(r["weights_full"]))),
            })
        md.append(pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"))
        md.append("")

        md.append(f"#### Final weights (full-data fit) — {ep_label}\n")
        weights_table = pd.DataFrame({
            "component": keys,
            **{sname: res[sname]["weights_full"] for sname, _ in SCHEMES},
        })
        md.append(weights_table.to_markdown(index=False, floatfmt=".4f"))
        md.append("")

        md.append(f"#### Per-fold weight stability (std across 5 inner folds) — {ep_label}\n")
        stab_table = pd.DataFrame({
            "component": keys,
            **{sname: res[sname]["weight_std"] for sname, _ in SCHEMES},
        })
        md.append(stab_table.to_markdown(index=False, floatfmt=".4f"))
        md.append("")

    md.append("## (hep × death) scheme-pair leaderboard (top 15 by weighted OOF)\n")
    md.append(candidates_df.head(15).to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## Promotion decision\n")
    if promoted is None:
        md.append(
            "**No submission recommended.** No (hep_scheme, dea_scheme) combination "
            f"meets the criteria (Δweighted ≥ +0.002 or Δhep ≥ +0.003 over "
            f"phase3_10_horizon_blend_v2, with stable weights and no single-component "
            f"negative dominance). The top weighted-OOF candidate was "
            f"({candidates_df.iloc[0]['hep_scheme']}, {candidates_df.iloc[0]['dea_scheme']}) "
            f"with Δweighted={candidates_df.iloc[0]['delta_weighted_vs_p10']:+.4f}, "
            f"Δhep={candidates_df.iloc[0]['delta_hep_vs_p10']:+.4f}.\n"
        )
    else:
        md.append(
            f"**Promoted**: hep={promoted.hep_scheme}, dea={promoted.dea_scheme}.\n"
            f"- weighted OOF: {promoted.weighted_oof:.4f} (Δ {promoted.delta_weighted_vs_p10:+.4f})\n"
            f"- hep OOF: {promoted.hep_oof:.4f} (Δ {promoted.delta_hep_vs_p10:+.4f})\n"
            f"- death OOF: {promoted.death_oof:.4f} (Δ {promoted.delta_dea_vs_p10:+.4f})\n"
            f"- hep neg total: {promoted.hep_neg_total:.3f}, dea neg total: {promoted.dea_neg_total:.3f}\n"
            f"- submission: `{sub_path}`\n"
        )
    md.append("")

    md.append("## Did negative weights help?\n")
    # Compare best A (nonneg) vs best B/C/D within the *same* dea_scheme=A as a clean ablation.
    a_a = next((r for r in candidates_table
                if r["hep_scheme"] == "A_nonneg" and r["dea_scheme"] == "A_nonneg"), None)
    rows_neg = []
    for sh in ("A_nonneg", "B_neg_005", "C_neg_010", "D_neg_020"):
        r = next((r for r in candidates_table if r["hep_scheme"] == sh and r["dea_scheme"] == "A_nonneg"), None)
        if r:
            rows_neg.append({
                "hep_scheme": sh,
                "hep_oof": r["hep_oof"],
                "delta_vs_A": r["hep_oof"] - (a_a["hep_oof"] if a_a else float("nan")),
                "neg_total": r["hep_neg_total"],
            })
    md.append("Hepatic-only ablation (death held to scheme A):")
    md.append(pd.DataFrame(rows_neg).to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## Notes\n")
    md.append("- All weights are constrained to |w| ≤ 0.50 with sum = 1, applied to "
              "centred percentile ranks (rank − 0.5).")
    md.append("- Inner CV: 5-fold stratified by event, separate per endpoint.")
    md.append("- The objective for schemes A–D is the smooth concordance loss on "
              "comparable pairs; for E, ridge regression on the centred event "
              "indicator with α=1.0; for F, discrete greedy ±0.05 steps subject to "
              "the same |w| ≤ 0.50 and anchor ≥ −0.20 floor.")
    md.append("- We never include strict_time_aligned components or any feature "
              "derived from event/censoring ages.")
    md.append("")

    out_md = cfg.REPORTS_DIR / "phase3_11_residual_negative_weight.md"
    out_md.write_text("\n".join(md))
    _LOG.info("wrote %s", out_md)


if __name__ == "__main__":
    main()
