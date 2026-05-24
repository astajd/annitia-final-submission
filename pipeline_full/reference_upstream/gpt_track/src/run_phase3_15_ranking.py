"""Phase 3.15 — direct pairwise / learning-to-rank for hepatic survival.

Anchor: ``phase3_10_horizon_blend_v2`` (LB **0.91093**).

Hepatic is the bottleneck (anchor OOF 0.8500). The horizon-binary
classifiers approximate ranking with thresholded labels; here we try
formulations whose objective is closer to the C-index itself.

Three ranker formulations:

1. **LightGBM LGBMRanker (LambdaRank)** — for each event patient ``i``,
   build a query group containing ``i`` (label 1) plus K controls from the
   risk set (label 0). Risk-set comparability rule: ``time_j > time_i``.

2. **XGBoost XGBRanker (rank:pairwise / rank:ndcg)** — same query groups.

3. **Pairwise logistic regression on difference vectors** — for each
   comparable pair ``(i, j)`` (event ``i`` at ``t_i``, ``t_j > t_i``)
   form ``x_i - x_j`` with label 1 and the symmetric ``x_j - x_i`` with
   label 0. Linear scoring at inference: ``risk = w · x``.

Compact strong feature sets only:
- ``v3_hepatic_schema`` (currently used by anchor's hep h3 LGBM)
- ``NIT_plus_scores`` (anchor's hep h1 LGBM and dea h4 catboost)
- ``biomarker_only`` (longitudinal_no_followup_proxies)

Death rankers are run on ``current_state_v2`` only as a quick
sanity-check; death is already at 0.9537 and is not the bottleneck.

We never use event/censoring-age features. The pair construction uses
the survival ``(event, time)`` pair only — same information the C-index
metric itself relies on.
"""
from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config as cfg
from .data_loading import load_dataset
from .features import build_feature_set
from .features.refined_longitudinal import (
    LAB_STEMS,
    NIT_STEMS,
    longitudinal_no_followup_proxies,
)
from .features.hep_focus import current_state_v2_no_visit_history
from .metrics import cindex, weighted_score
from .models import build_model
from .models._preprocess import fill_for_tree
from .models.ensemble import rank_average, to_rank
from .submission import make_submission
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger
from .validation import build_folds

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True, na_option="keep").to_numpy()


def _blend_two(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    ra, rb = _rank(a), _rank(b)
    valid_a, valid_b = np.isfinite(ra), np.isfinite(rb)
    return np.where(valid_a & valid_b, alpha * ra + (1 - alpha) * rb,
                    np.where(valid_a, ra, np.where(valid_b, rb, np.nan)))


def _surv_ci(event: np.ndarray, time: np.ndarray, score: np.ndarray) -> float:
    finite = np.isfinite(score)
    if finite.sum() < 5 or event[finite].sum() == 0:
        return float("nan")
    return float(cindex(event[finite], time[finite], score[finite]).cindex)


def _fold_ci(event, time, score, splits) -> tuple[float, float, float]:
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


def assemble_feature_sets(ds) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}
    fs1 = build_feature_set("current_state_v2", ds)
    out["current_state_v2"] = {"train": fs1.X_train, "test": fs1.X_test}
    fs2 = build_feature_set("NIT_plus_scores_longitudinal", ds)
    out["NIT_plus_scores"] = {"train": fs2.X_train, "test": fs2.X_test}
    Xtr = longitudinal_no_followup_proxies(ds.train_df, ds.visit_columns, ds.age_visit_cols,
                                           keep_stems=LAB_STEMS + NIT_STEMS)
    Xte = longitudinal_no_followup_proxies(ds.test_df, ds.visit_columns, ds.age_visit_cols,
                                           keep_stems=LAB_STEMS + NIT_STEMS)
    out["biomarker_only"] = {"train": Xtr, "test": Xte}
    fs4 = build_feature_set("current_state_v2_no_visit_history", ds)
    out["v3_hepatic_schema"] = {"train": fs4.X_train, "test": fs4.X_test}
    for name, blob in out.items():
        for k in ("train", "test"):
            blob[k] = blob[k].select_dtypes(include=[np.number])
    return out


# ---------------------------------------------------------------------------
# Pair / group construction
# ---------------------------------------------------------------------------

@dataclass
class GroupSpec:
    """Per-fold ranking groups (one query per event patient)."""
    rows_X: np.ndarray        # (sum_group_sizes, n_features)
    rows_y: np.ndarray        # (sum_group_sizes,) labels (1 for the event)
    group_sizes: np.ndarray   # (n_events,) sizes per query
    n_event_patients: int
    n_pairs: int


def _build_groups(X_arr: np.ndarray, event: np.ndarray, time: np.ndarray,
                   train_idx: np.ndarray, *, n_controls: int = 50,
                   seed: int = 0) -> GroupSpec:
    """For each event in ``train_idx``, sample up to ``n_controls`` comparable
    controls and form a single query group."""
    rng = np.random.default_rng(seed)
    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []
    group_sizes: list[int] = []
    n_pairs = 0
    n_event_patients = 0
    train_set = set(train_idx.tolist())
    for i in train_idx:
        if not event[i]:
            continue
        ti = time[i]
        # comparable controls within training fold: time_j > t_i
        controls = [j for j in train_idx if time[j] > ti and j != i]
        if len(controls) < 2:
            continue
        if len(controls) > n_controls:
            controls = rng.choice(controls, size=n_controls, replace=False).tolist()
        rows_X.append(X_arr[i])
        rows_y.append(1)
        for c in controls:
            rows_X.append(X_arr[c])
            rows_y.append(0)
        group_sizes.append(len(controls) + 1)
        n_pairs += len(controls)
        n_event_patients += 1
    if not group_sizes:
        return GroupSpec(np.empty((0, X_arr.shape[1])), np.empty(0, dtype=int),
                           np.empty(0, dtype=int), 0, 0)
    return GroupSpec(np.array(rows_X), np.array(rows_y, dtype=int),
                       np.array(group_sizes, dtype=int),
                       n_event_patients, n_pairs)


def _build_difference_pairs(X_arr: np.ndarray, event: np.ndarray, time: np.ndarray,
                              train_idx: np.ndarray, *, n_controls: int = 50,
                              seed: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
    """For each event ``i`` in ``train_idx``, sample up to ``n_controls`` controls
    and form (x_i - x_j, y=1) and (x_j - x_i, y=0) symmetric pairs."""
    rng = np.random.default_rng(seed)
    diffs: list[np.ndarray] = []
    labels: list[int] = []
    n_event_patients = 0
    for i in train_idx:
        if not event[i]:
            continue
        ti = time[i]
        controls = [j for j in train_idx if time[j] > ti and j != i]
        if len(controls) < 2:
            continue
        if len(controls) > n_controls:
            controls = rng.choice(controls, size=n_controls, replace=False).tolist()
        n_event_patients += 1
        xi = X_arr[i]
        for j in controls:
            xj = X_arr[j]
            diffs.append(xi - xj)
            labels.append(1)
            diffs.append(xj - xi)
            labels.append(0)
    if not diffs:
        return np.empty((0, X_arr.shape[1])), np.empty(0, dtype=int), 0
    return np.asarray(diffs, dtype=np.float32), np.asarray(labels, dtype=int), n_event_patients


# ---------------------------------------------------------------------------
# Rankers
# ---------------------------------------------------------------------------

def _train_lgbm_ranker(group: GroupSpec, *, seed: int = 0):
    import lightgbm as lgb
    if group.rows_X.shape[0] == 0:
        return None
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=5, reg_lambda=1.0,
        random_state=seed, verbose=-1,
    )
    ranker.fit(group.rows_X, group.rows_y, group=group.group_sizes)
    return ranker


def _train_xgb_ranker(group: GroupSpec, *, seed: int = 0):
    import xgboost as xgb
    if group.rows_X.shape[0] == 0:
        return None
    ranker = xgb.XGBRanker(
        objective="rank:pairwise",
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        verbosity=0, tree_method="hist", random_state=seed,
    )
    ranker.fit(group.rows_X, group.rows_y, group=group.group_sizes)
    return ranker


def _train_pairwise_logreg(diff_X: np.ndarray, diff_y: np.ndarray, *, C: float = 1.0,
                             seed: int = 0):
    if diff_X.shape[0] == 0:
        return None
    clf = LogisticRegression(
        penalty="l2", C=C, max_iter=2000, fit_intercept=False,
        solver="liblinear", random_state=seed,
    )
    clf.fit(diff_X, diff_y)
    return clf.coef_[0]  # (n_features,)


# ---------------------------------------------------------------------------
# OOF/test prediction over the 5x3 folds
# ---------------------------------------------------------------------------

def _ranker_predict(ranker_obj, X_arr: np.ndarray) -> np.ndarray:
    return ranker_obj.predict(X_arr)


def _logreg_predict(coef: np.ndarray, X_arr: np.ndarray) -> np.ndarray:
    return X_arr @ coef


def _run_ranker_cv(model_kind: str, fs_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                    event: np.ndarray, time: np.ndarray, splits, *,
                    n_controls: int = 50, seed: int = 0
                    ) -> tuple[np.ndarray, np.ndarray, dict]:
    Xtr_p = fill_for_tree(X_train).astype(np.float32, errors="ignore")
    Xte_p = fill_for_tree(X_test).reindex(columns=Xtr_p.columns, fill_value=0).astype(np.float32, errors="ignore")
    Xtr_arr = Xtr_p.to_numpy()
    Xte_arr = Xte_p.to_numpy()
    n_train = Xtr_arr.shape[0]
    n_test = Xte_arr.shape[0]

    oof = np.full(n_train, np.nan)
    test_sum = np.zeros(n_test)
    test_n = 0
    fold_pair_counts: list[int] = []
    fold_event_counts: list[int] = []

    for s in splits:
        tr_idx = np.asarray(s.train_idx, dtype=int)
        va_idx = np.asarray(s.valid_idx, dtype=int)
        if event[tr_idx].sum() < 3:
            continue
        if model_kind in ("lgbm_ranker", "xgb_ranker"):
            grp = _build_groups(Xtr_arr, event, time, tr_idx,
                                  n_controls=n_controls, seed=seed)
            fold_pair_counts.append(int(grp.n_pairs))
            fold_event_counts.append(int(grp.n_event_patients))
            if grp.n_event_patients < 3:
                continue
            try:
                if model_kind == "lgbm_ranker":
                    ranker = _train_lgbm_ranker(grp, seed=seed)
                else:
                    ranker = _train_xgb_ranker(grp, seed=seed)
            except Exception as e:  # noqa: BLE001
                _LOG.warning("ranker fit failed: %s", e)
                continue
            oof[va_idx] = _ranker_predict(ranker, Xtr_arr[va_idx])
            test_sum += _ranker_predict(ranker, Xte_arr)
            test_n += 1
        elif model_kind == "pairwise_logreg":
            # Median-impute NaN per fold (logreg can't handle NaN).
            tr_means = np.nanmedian(Xtr_arr[tr_idx], axis=0)
            tr_means = np.where(np.isfinite(tr_means), tr_means, 0.0)
            Xtr_imp = np.where(np.isfinite(Xtr_arr), Xtr_arr, tr_means[None, :])
            Xte_imp = np.where(np.isfinite(Xte_arr), Xte_arr, tr_means[None, :])
            # Standardize so the L2 penalty is meaningful.
            std = np.where(np.std(Xtr_imp[tr_idx], axis=0) > 0, np.std(Xtr_imp[tr_idx], axis=0), 1.0)
            Xtr_imp = Xtr_imp / std
            Xte_imp = Xte_imp / std
            diff_X, diff_y, n_ep = _build_difference_pairs(
                Xtr_imp, event, time, tr_idx, n_controls=n_controls, seed=seed)
            fold_pair_counts.append(int(diff_X.shape[0] // 2))
            fold_event_counts.append(int(n_ep))
            if n_ep < 3:
                continue
            coef = _train_pairwise_logreg(diff_X, diff_y, seed=seed)
            if coef is None:
                continue
            oof[va_idx] = _logreg_predict(coef, Xtr_imp[va_idx])
            test_sum += _logreg_predict(coef, Xte_imp)
            test_n += 1
        else:
            raise KeyError(model_kind)

    test = test_sum / test_n if test_n else np.full(n_test, np.nan)
    info = {
        "fold_pair_counts": fold_pair_counts,
        "fold_event_counts": fold_event_counts,
        "median_pairs_per_fold": int(np.median(fold_pair_counts)) if fold_pair_counts else 0,
        "median_events_per_fold": int(np.median(fold_event_counts)) if fold_event_counts else 0,
    }
    return oof, test, info


# ---------------------------------------------------------------------------
# Anchor reconstruction
# ---------------------------------------------------------------------------

def _v3_oof(n_train: int, ds, hep) -> tuple[np.ndarray, np.ndarray]:
    cands = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.json"))
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
    h_oof = rank_average({c: pool[c] for c in h_w if c in pool}, weights={c: h_w[c] for c in h_w if c in pool})
    d_oof = rank_average({c: pool[c] for c in d_w if c in pool}, weights={c: d_w[c] for c in d_w if c in pool})
    return h_oof, d_oof


def _load_horizon_artifact(label: str) -> tuple[np.ndarray, np.ndarray] | None:
    for root in (cfg.EXPERIMENT_OUTPUTS / "phase3_12_horizon",
                 cfg.EXPERIMENT_OUTPUTS / "phase3_10_horizon",
                 cfg.EXPERIMENT_OUTPUTS / "phase3_9_horizon"):
        run_dir = root / label
        if not run_dir.exists():
            continue
        oof_p = run_dir / "oof.csv"
        test_p = run_dir / "test.csv"
        if oof_p.exists() and test_p.exists():
            o = pd.read_csv(oof_p)["oof"].to_numpy()
            t = pd.read_csv(test_p)["test"].to_numpy()
            return o, t
    return None


def _load_anchor(ds, hep, death, n_train, n_test):
    p10_meta_path = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_10_horizon_blend_v2.json"))[-1]
    meta = json.loads(p10_meta_path.read_text())
    blend_csv = p10_meta_path.with_suffix(".csv")
    pub = pd.read_csv(blend_csv)
    h_test = pub[cfg.SUB_HEPATIC_COL].to_numpy()
    d_test = pub[cfg.SUB_DEATH_COL].to_numpy()
    v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)
    weights_h = meta["weights_hepatic"]
    weights_d = meta["weights_death"]
    pool_h = {"_v3": v3_h_oof}
    pool_d = {"_v3": v3_d_oof}
    for k in weights_h:
        if k == "_v3":
            continue
        out = _load_horizon_artifact(k)
        if out is not None:
            pool_h[k] = out[0]
    for k in weights_d:
        if k == "_v3":
            continue
        out = _load_horizon_artifact(k)
        if out is not None:
            pool_d[k] = out[0]
    h_oof = rank_average(pool_h, weights={k: weights_h[k] for k in pool_h})
    d_oof = rank_average(pool_d, weights={k: weights_d[k] for k in pool_d})
    return h_oof, h_test, d_oof, d_test, meta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    cfg.ensure_dirs()
    out_root = cfg.EXPERIMENT_OUTPUTS / "phase3_15_ranking"
    out_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)
    feature_sets = assemble_feature_sets(ds)

    a_h_oof, a_h_test, a_d_oof, a_d_test, anchor_meta = _load_anchor(ds, hep, death, n_train, n_test)
    a_h_ci = _surv_ci(hep.event, hep.time, a_h_oof)
    a_d_ci = _surv_ci(death.event, death.time, a_d_oof)
    a_w = weighted_score(a_h_ci, a_d_ci)
    a_h_fold = _fold_ci(hep.event, hep.time, a_h_oof, splits)
    a_d_fold = _fold_ci(death.event, death.time, a_d_oof, splits)
    _LOG.info("Anchor OOF: hep=%.4f dea=%.4f weighted=%.4f", a_h_ci, a_d_ci, a_w)

    # ------------------------------------------------------------------
    # Hepatic ranker grid: 3 ranker types × 3 feature sets × n_controls ∈ {20, 50}
    # ------------------------------------------------------------------

    runs: list[dict] = []
    artifacts: dict[str, dict[str, np.ndarray]] = {}

    rankers = ["lgbm_ranker", "xgb_ranker", "pairwise_logreg"]
    feature_set_names = ["v3_hepatic_schema", "NIT_plus_scores", "biomarker_only"]
    n_controls_grid = [20, 50]

    _LOG.info("Hepatic ranker sweep")
    for fs_name in feature_set_names:
        blob = feature_sets[fs_name]
        for kind in rankers:
            for K in n_controls_grid:
                label = f"hep__{kind}__{fs_name}__K{K}__s0"
                t0 = _time.time()
                try:
                    oof, test, info = _run_ranker_cv(
                        kind, fs_name, blob["train"], blob["test"],
                        hep.event, hep.time, splits,
                        n_controls=K, seed=0,
                    )
                except Exception as e:  # noqa: BLE001
                    _LOG.error("%s failed: %s", label, e)
                    continue
                dt = _time.time() - t0
                ci = _surv_ci(hep.event, hep.time, oof)
                m, sd, mn = _fold_ci(hep.event, hep.time, oof, splits)
                rho_oof_anchor = _spearman(oof, a_h_oof)
                rho_test_anchor = _spearman(test, a_h_test)
                runs.append({
                    "label": label, "endpoint": "hepatic", "model_kind": kind,
                    "feature_set": fs_name, "n_controls": K, "seed": 0,
                    "ci_oof": ci, "fold_std": sd, "fold_min": mn,
                    "rho_oof_anchor": rho_oof_anchor, "rho_test_anchor": rho_test_anchor,
                    "median_pairs_per_fold": info["median_pairs_per_fold"],
                    "median_events_per_fold": info["median_events_per_fold"],
                    "wall_seconds": dt,
                })
                artifacts[label] = {"oof": oof, "test": test}
                _LOG.info("%s ci=%.4f rho=%.3f pairs=%d (%.1fs)", label, ci,
                          rho_oof_anchor, info["median_pairs_per_fold"], dt)

    # Multi-seed for the top hepatic ranker config (best ci_oof).
    hep_runs = [r for r in runs if r["endpoint"] == "hepatic" and np.isfinite(r["ci_oof"])]
    if hep_runs:
        best_hep = max(hep_runs, key=lambda r: r["ci_oof"])
        _LOG.info("Best single-seed hep ranker: %s ci=%.4f", best_hep["label"], best_hep["ci_oof"])
        for seed in (1, 2, 3):
            label = f"hep__{best_hep['model_kind']}__{best_hep['feature_set']}__K{best_hep['n_controls']}__s{seed}"
            t0 = _time.time()
            try:
                oof, test, info = _run_ranker_cv(
                    best_hep["model_kind"], best_hep["feature_set"],
                    feature_sets[best_hep["feature_set"]]["train"],
                    feature_sets[best_hep["feature_set"]]["test"],
                    hep.event, hep.time, splits,
                    n_controls=best_hep["n_controls"], seed=seed,
                )
            except Exception as e:  # noqa: BLE001
                _LOG.error("%s failed: %s", label, e)
                continue
            dt = _time.time() - t0
            ci = _surv_ci(hep.event, hep.time, oof)
            m, sd, mn = _fold_ci(hep.event, hep.time, oof, splits)
            rho_oof_anchor = _spearman(oof, a_h_oof)
            rho_test_anchor = _spearman(test, a_h_test)
            runs.append({
                "label": label, "endpoint": "hepatic", "model_kind": best_hep["model_kind"],
                "feature_set": best_hep["feature_set"], "n_controls": best_hep["n_controls"], "seed": seed,
                "ci_oof": ci, "fold_std": sd, "fold_min": mn,
                "rho_oof_anchor": rho_oof_anchor, "rho_test_anchor": rho_test_anchor,
                "median_pairs_per_fold": info["median_pairs_per_fold"],
                "median_events_per_fold": info["median_events_per_fold"],
                "wall_seconds": dt,
            })
            artifacts[label] = {"oof": oof, "test": test}
            _LOG.info("seed-bag %s ci=%.4f", label, ci)

        # Build a hepatic ranker ensemble by rank-averaging the multi-seed runs of the best config.
        bag_keys = [k for k in artifacts.keys()
                     if k.startswith(f"hep__{best_hep['model_kind']}__{best_hep['feature_set']}__K{best_hep['n_controls']}")]
        if len(bag_keys) >= 2:
            bag_oof = rank_average({k: artifacts[k]["oof"] for k in bag_keys})
            bag_test = rank_average({k: artifacts[k]["test"] for k in bag_keys})
            ci = _surv_ci(hep.event, hep.time, bag_oof)
            m, sd, mn = _fold_ci(hep.event, hep.time, bag_oof, splits)
            rho_oof_anchor = _spearman(bag_oof, a_h_oof)
            rho_test_anchor = _spearman(bag_test, a_h_test)
            label = f"hep__{best_hep['model_kind']}__{best_hep['feature_set']}__K{best_hep['n_controls']}__bag"
            runs.append({
                "label": label, "endpoint": "hepatic", "model_kind": best_hep["model_kind"],
                "feature_set": best_hep["feature_set"], "n_controls": best_hep["n_controls"], "seed": -1,
                "ci_oof": ci, "fold_std": sd, "fold_min": mn,
                "rho_oof_anchor": rho_oof_anchor, "rho_test_anchor": rho_test_anchor,
                "median_pairs_per_fold": -1,
                "median_events_per_fold": -1,
                "wall_seconds": 0.0,
            })
            artifacts[label] = {"oof": bag_oof, "test": bag_test}
            _LOG.info("seed bag (%d seeds) %s ci=%.4f", len(bag_keys), label, ci)
    else:
        best_hep = None

    # Cross-feature-set hepatic ensemble: rank-average the best lgbm_ranker
    # output per feature set as a diversity probe.
    fs_best_keys = []
    for fs_name in feature_set_names:
        cand = [r for r in runs if r["endpoint"] == "hepatic" and r["feature_set"] == fs_name
                and r["model_kind"] == "lgbm_ranker" and r["seed"] == 0
                and np.isfinite(r["ci_oof"])]
        if cand:
            fs_best_keys.append(cand[0]["label"])
    if len(fs_best_keys) >= 2:
        fs_h_oof = rank_average({k: artifacts[k]["oof"] for k in fs_best_keys})
        fs_h_test = rank_average({k: artifacts[k]["test"] for k in fs_best_keys})
        ci = _surv_ci(hep.event, hep.time, fs_h_oof)
        m, sd, mn = _fold_ci(hep.event, hep.time, fs_h_oof, splits)
        runs.append({
            "label": "hep__lgbm_ranker__cross_fs_ensemble", "endpoint": "hepatic",
            "model_kind": "lgbm_ranker", "feature_set": "ensemble", "n_controls": -1, "seed": -1,
            "ci_oof": ci, "fold_std": sd, "fold_min": mn,
            "rho_oof_anchor": _spearman(fs_h_oof, a_h_oof),
            "rho_test_anchor": _spearman(fs_h_test, a_h_test),
            "median_pairs_per_fold": -1, "median_events_per_fold": -1, "wall_seconds": 0.0,
        })
        artifacts["hep__lgbm_ranker__cross_fs_ensemble"] = {"oof": fs_h_oof, "test": fs_h_test}
        _LOG.info("cross-fs hep ensemble ci=%.4f", ci)

    # ------------------------------------------------------------------
    # Optional death ranker — quick sweep with lgbm_ranker on current_state_v2.
    # ------------------------------------------------------------------

    _LOG.info("Death ranker (optional sanity check)")
    blob = feature_sets["current_state_v2"]
    for kind in ("lgbm_ranker", "xgb_ranker"):
        for K in (50,):
            label = f"dea__{kind}__current_state_v2__K{K}__s0"
            t0 = _time.time()
            try:
                oof, test, info = _run_ranker_cv(
                    kind, "current_state_v2", blob["train"], blob["test"],
                    death.event, death.time, splits,
                    n_controls=K, seed=0,
                )
            except Exception as e:  # noqa: BLE001
                _LOG.error("%s failed: %s", label, e)
                continue
            dt = _time.time() - t0
            ci = _surv_ci(death.event, death.time, oof)
            m, sd, mn = _fold_ci(death.event, death.time, oof, splits)
            rho_oof_anchor = _spearman(oof, a_d_oof)
            rho_test_anchor = _spearman(test, a_d_test)
            runs.append({
                "label": label, "endpoint": "death", "model_kind": kind,
                "feature_set": "current_state_v2", "n_controls": K, "seed": 0,
                "ci_oof": ci, "fold_std": sd, "fold_min": mn,
                "rho_oof_anchor": rho_oof_anchor, "rho_test_anchor": rho_test_anchor,
                "median_pairs_per_fold": info["median_pairs_per_fold"],
                "median_events_per_fold": info["median_events_per_fold"],
                "wall_seconds": dt,
            })
            artifacts[label] = {"oof": oof, "test": test}
            _LOG.info("%s ci=%.4f rho=%.3f (%.1fs)", label, ci, rho_oof_anchor, dt)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(out_root / "all_runs.csv", index=False)

    # ------------------------------------------------------------------
    # Blend grid: anchor + best hepatic ranker (and best death ranker if any)
    # ------------------------------------------------------------------

    # Best hepatic ranker by OOF C-index (consider both single-seed and bag).
    hep_runs2 = [r for r in runs if r["endpoint"] == "hepatic" and np.isfinite(r["ci_oof"])]
    best_hep_run = max(hep_runs2, key=lambda r: r["ci_oof"]) if hep_runs2 else None
    dea_runs = [r for r in runs if r["endpoint"] == "death" and np.isfinite(r["ci_oof"])]
    best_dea_run = max(dea_runs, key=lambda r: r["ci_oof"]) if dea_runs else None

    blend_rows: list[dict] = []
    if best_hep_run is not None:
        h_oof_r = artifacts[best_hep_run["label"]]["oof"]
        h_test_r = artifacts[best_hep_run["label"]]["test"]
        for alpha in (0.95, 0.90, 0.85, 0.80, 0.75):
            h_oof_blend = _blend_two(a_h_oof, h_oof_r, alpha)
            h_test_blend = _blend_two(a_h_test, h_test_r, alpha)
            ci_h = _surv_ci(hep.event, hep.time, h_oof_blend)
            mh, sh, mnh = _fold_ci(hep.event, hep.time, h_oof_blend, splits)
            rho_h_test = _spearman(h_test_blend, a_h_test)
            blend_rows.append({
                "blend": f"alpha={alpha}_hep_only",
                "alpha": alpha, "side": "hep_only", "source": "hep_ranker",
                "hep_oof": ci_h,
                "death_oof": a_d_ci,
                "weighted_oof": weighted_score(ci_h, a_d_ci),
                "hep_fold_std": sh, "hep_fold_min": mnh,
                "dea_fold_std": a_d_fold[1], "dea_fold_min": a_d_fold[2],
                "rho_h_test_anchor": rho_h_test,
                "rho_d_test_anchor": 1.0,
                "delta_w_vs_anchor": weighted_score(ci_h, a_d_ci) - a_w,
                "delta_h_vs_anchor": ci_h - a_h_ci,
                "delta_d_vs_anchor": 0.0,
                "h_oof": h_oof_blend, "d_oof": a_d_oof,
                "h_test": h_test_blend, "d_test": a_d_test,
                "components_h": [best_hep_run["label"]],
                "components_d": [],
                "weights_h": {"anchor": alpha, best_hep_run["label"]: 1 - alpha},
                "weights_d": {"anchor": 1.0},
            })

    # Greedy capped at 25% — only horizon ranker (single best) since pool is small.
    if best_hep_run is not None:
        # Try alpha grid starting from anchor and adding ranker up to 25% weight.
        for total_ranker in (0.05, 0.10, 0.15, 0.20, 0.25):
            alpha = 1.0 - total_ranker
            h_oof_blend = _blend_two(a_h_oof, artifacts[best_hep_run["label"]]["oof"], alpha)
            h_test_blend = _blend_two(a_h_test, artifacts[best_hep_run["label"]]["test"], alpha)
            ci_h = _surv_ci(hep.event, hep.time, h_oof_blend)
            blend_rows.append({
                "blend": f"capped={total_ranker:.2f}_hep_only",
                "alpha": alpha, "side": "hep_only", "source": "capped",
                "hep_oof": ci_h,
                "death_oof": a_d_ci,
                "weighted_oof": weighted_score(ci_h, a_d_ci),
                "hep_fold_std": _fold_ci(hep.event, hep.time, h_oof_blend, splits)[1],
                "hep_fold_min": _fold_ci(hep.event, hep.time, h_oof_blend, splits)[2],
                "dea_fold_std": a_d_fold[1], "dea_fold_min": a_d_fold[2],
                "rho_h_test_anchor": _spearman(h_test_blend, a_h_test),
                "rho_d_test_anchor": 1.0,
                "delta_w_vs_anchor": weighted_score(ci_h, a_d_ci) - a_w,
                "delta_h_vs_anchor": ci_h - a_h_ci,
                "delta_d_vs_anchor": 0.0,
                "h_oof": h_oof_blend, "d_oof": a_d_oof,
                "h_test": h_test_blend, "d_test": a_d_test,
                "components_h": [best_hep_run["label"]],
                "components_d": [],
                "weights_h": {"anchor": alpha, best_hep_run["label"]: 1 - alpha},
                "weights_d": {"anchor": 1.0},
            })

    # Both-endpoint blend if best death ranker improves death OOF.
    if best_dea_run is not None and best_dea_run["ci_oof"] > a_d_ci - 0.001:
        for alpha in (0.90, 0.85, 0.80):
            h_oof_blend = _blend_two(a_h_oof, artifacts[best_hep_run["label"]]["oof"], alpha)
            h_test_blend = _blend_two(a_h_test, artifacts[best_hep_run["label"]]["test"], alpha)
            d_oof_blend = _blend_two(a_d_oof, artifacts[best_dea_run["label"]]["oof"], alpha)
            d_test_blend = _blend_two(a_d_test, artifacts[best_dea_run["label"]]["test"], alpha)
            ci_h = _surv_ci(hep.event, hep.time, h_oof_blend)
            ci_d = _surv_ci(death.event, death.time, d_oof_blend)
            blend_rows.append({
                "blend": f"alpha={alpha}_both",
                "alpha": alpha, "side": "both", "source": "ranker_both",
                "hep_oof": ci_h, "death_oof": ci_d,
                "weighted_oof": weighted_score(ci_h, ci_d),
                "hep_fold_std": _fold_ci(hep.event, hep.time, h_oof_blend, splits)[1],
                "hep_fold_min": _fold_ci(hep.event, hep.time, h_oof_blend, splits)[2],
                "dea_fold_std": _fold_ci(death.event, death.time, d_oof_blend, splits)[1],
                "dea_fold_min": _fold_ci(death.event, death.time, d_oof_blend, splits)[2],
                "rho_h_test_anchor": _spearman(h_test_blend, a_h_test),
                "rho_d_test_anchor": _spearman(d_test_blend, a_d_test),
                "delta_w_vs_anchor": weighted_score(ci_h, ci_d) - a_w,
                "delta_h_vs_anchor": ci_h - a_h_ci,
                "delta_d_vs_anchor": ci_d - a_d_ci,
                "h_oof": h_oof_blend, "d_oof": d_oof_blend,
                "h_test": h_test_blend, "d_test": d_test_blend,
                "components_h": [best_hep_run["label"]],
                "components_d": [best_dea_run["label"]],
                "weights_h": {"anchor": alpha, best_hep_run["label"]: 1 - alpha},
                "weights_d": {"anchor": alpha, best_dea_run["label"]: 1 - alpha},
            })

    blend_df = pd.DataFrame([{k: v for k, v in r.items()
                                if k not in ("h_oof", "d_oof", "h_test", "d_test",
                                              "components_h", "components_d",
                                              "weights_h", "weights_d")}
                              for r in blend_rows])
    blend_df.to_csv(out_root / "blends.csv", index=False)

    # ------------------------------------------------------------------
    # Promotion criteria
    # ------------------------------------------------------------------

    candidate_hep_only = None
    candidate_full = None
    for row in sorted(blend_rows, key=lambda r: -r["weighted_oof"]):
        if row["delta_w_vs_anchor"] >= 0.002 or row["delta_h_vs_anchor"] >= 0.003:
            if row["side"] == "hep_only" and candidate_hep_only is None:
                candidate_hep_only = row
            elif row["side"] == "both" and candidate_full is None:
                candidate_full = row
        if candidate_hep_only is not None and candidate_full is not None:
            break

    candidate_paths = {}
    if candidate_hep_only is not None:
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=candidate_hep_only["h_test"],
            risk_death=candidate_hep_only["d_test"],
            sample_submission=ds.sample_submission,
            model_name="phase3_15_pairwise_hepatic_blend",
        )
        meta = {
            "label": "phase3_15_pairwise_hepatic_blend",
            "model_type": "pairwise_ranker_blend_hep_only",
            "feature_set_hep": best_hep_run["feature_set"] if best_hep_run else None,
            "ranker_kind": best_hep_run["model_kind"] if best_hep_run else None,
            "n_controls_per_event": best_hep_run["n_controls"] if best_hep_run else None,
            "blend_alpha": candidate_hep_only["alpha"],
            "components_hepatic": candidate_hep_only["components_h"],
            "weights_hepatic": candidate_hep_only["weights_h"],
            "components_death":  candidate_hep_only["components_d"],
            "weights_death":     candidate_hep_only["weights_d"],
            "hepatic_oof": candidate_hep_only["hep_oof"],
            "death_oof":   candidate_hep_only["death_oof"],
            "weighted_oof": candidate_hep_only["weighted_oof"],
            "delta_vs_anchor": {
                "weighted": candidate_hep_only["delta_w_vs_anchor"],
                "hepatic":  candidate_hep_only["delta_h_vs_anchor"],
                "death":    candidate_hep_only["delta_d_vs_anchor"],
            },
            "rho_test_anchor": {
                "hepatic": candidate_hep_only["rho_h_test_anchor"],
                "death":   candidate_hep_only["rho_d_test_anchor"],
            },
            "uses_target_derived_features": False,
            "recommended": True,
            "submission_csv": str(sub_path),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        candidate_paths["phase3_15_pairwise_hepatic_blend"] = sub_path

    if candidate_full is not None:
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=candidate_full["h_test"],
            risk_death=candidate_full["d_test"],
            sample_submission=ds.sample_submission,
            model_name="phase3_15_pairwise_full_blend",
        )
        meta = {
            "label": "phase3_15_pairwise_full_blend",
            "model_type": "pairwise_ranker_blend_both",
            "ranker_kind_hep": best_hep_run["model_kind"] if best_hep_run else None,
            "ranker_kind_dea": best_dea_run["model_kind"] if best_dea_run else None,
            "blend_alpha": candidate_full["alpha"],
            "components_hepatic": candidate_full["components_h"],
            "weights_hepatic": candidate_full["weights_h"],
            "components_death":  candidate_full["components_d"],
            "weights_death":     candidate_full["weights_d"],
            "hepatic_oof": candidate_full["hep_oof"],
            "death_oof":   candidate_full["death_oof"],
            "weighted_oof": candidate_full["weighted_oof"],
            "delta_vs_anchor": {
                "weighted": candidate_full["delta_w_vs_anchor"],
                "hepatic":  candidate_full["delta_h_vs_anchor"],
                "death":    candidate_full["delta_d_vs_anchor"],
            },
            "rho_test_anchor": {
                "hepatic": candidate_full["rho_h_test_anchor"],
                "death":   candidate_full["rho_d_test_anchor"],
            },
            "uses_target_derived_features": False,
            "recommended": False,  # secondary
            "submission_csv": str(sub_path),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        candidate_paths["phase3_15_pairwise_full_blend"] = sub_path

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    md: list[str] = []
    md.append("# Phase 3.15 — pairwise / learning-to-rank for hepatic\n")
    md.append("Anchor: `phase3_10_horizon_blend_v2` (LB **0.91093**, weighted OOF "
              f"{a_w:.4f}, hep {a_h_ci:.4f}, dea {a_d_ci:.4f}).\n")
    md.append(f"Anchor fold: hep std/min={a_h_fold[1]:.4f}/{a_h_fold[2]:.4f}, "
              f"dea std/min={a_d_fold[1]:.4f}/{a_d_fold[2]:.4f}.\n")

    md.append("## A. Pair construction\n")
    md.append(
        "Risk-set rule: for each event patient *i* in the training fold, "
        "comparable controls are training-fold patients *j* with `time_j > time_i` "
        "(regardless of whether *j* eventually has the event). All comparisons "
        "use only `(event, time)` from the survival endpoint — no event/"
        "censoring-age columns are used as features.\n"
    )
    md.append("- LightGBM/XGBoost: query group per event = (event patient, "
              "label=1) + sampled controls (label=0).\n"
              "- Pairwise logistic: difference vectors x_i − x_j with "
              "label 1 plus the symmetric x_j − x_i with label 0.\n"
              "- Cap at K controls per event (K ∈ {20, 50}) so a few events "
              "with many comparable controls don't dominate.\n")
    md.append("")

    md.append("## B. Ranker sweep results\n")
    if not runs_df.empty:
        cols = ["label", "endpoint", "model_kind", "feature_set", "n_controls", "seed",
                 "ci_oof", "fold_std", "fold_min",
                 "rho_oof_anchor", "rho_test_anchor",
                 "median_events_per_fold", "median_pairs_per_fold", "wall_seconds"]
        md.append(runs_df[cols].sort_values(["endpoint", "ci_oof"], ascending=[True, False])
                  .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## C. Pair counts per fold (typical)\n")
    if not runs_df.empty and "median_events_per_fold" in runs_df.columns:
        ev_med = runs_df.loc[runs_df["seed"] == 0, ["endpoint", "median_events_per_fold", "median_pairs_per_fold"]].drop_duplicates(subset=["endpoint"]).reset_index(drop=True)
        md.append(ev_med.to_markdown(index=False))
    md.append("")

    md.append("## D. Blend grid vs anchor\n")
    if not blend_df.empty:
        md.append(blend_df.sort_values("weighted_oof", ascending=False)
                  .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## E. Candidates emitted\n")
    if not candidate_paths:
        md.append(
            "**No candidates emitted.** No (ranker, blend) combination produced "
            "an OOF improvement of ≥ +0.002 weighted or ≥ +0.003 hepatic over "
            "`phase3_10_horizon_blend_v2`.\n"
        )
    else:
        for name, path in candidate_paths.items():
            md.append(f"- `{name}` → `{path}`")
        md.append("")

    md.append("## F. Recommendation\n")
    if candidate_hep_only is not None:
        md.append(
            f"**Submit one**: `{candidate_paths['phase3_15_pairwise_hepatic_blend']}`\n"
            f"- blend `alpha={candidate_hep_only['alpha']}` (anchor + hep-only ranker)\n"
            f"- weighted OOF Δ vs anchor: {candidate_hep_only['delta_w_vs_anchor']:+.4f}\n"
            f"- hepatic OOF Δ vs anchor: {candidate_hep_only['delta_h_vs_anchor']:+.4f}\n"
            f"- rho with anchor on test: hep={candidate_hep_only['rho_h_test_anchor']:.3f}\n"
        )
        if candidate_full is not None:
            md.append(
                f"\nA secondary candidate `phase3_15_pairwise_full_blend` is also "
                f"emitted (death ranker also helps), saved at "
                f"`{candidate_paths['phase3_15_pairwise_full_blend']}`. Consider "
                f"only as a follow-up after the hep-only candidate has been probed.\n"
            )
    else:
        md.append(
            "**Do not submit.** Hold the anchor `phase3_10_horizon_blend_v2` "
            "(LB 0.91093). The pairwise ranking objective did not move OOF "
            "enough to justify a public-LB slot.\n"
        )

    md.append("\n## Notes\n")
    md.append("- All ranker training is fold-internal: groups/pairs are built only "
               "from training-fold rows; validation folds receive *one* prediction "
               "per patient and are scored with the standard survival C-index.")
    md.append("- LightGBM/XGBoost rankers use the actual ranker objectives "
               "(`lambdarank`, `rank:pairwise`); the pairwise logistic baseline "
               "uses a linear score `w · x` derived from logistic regression on "
               "difference vectors.")
    md.append("- No event/censoring-age features. The horizon-classifier components "
               "of the anchor are unchanged in any blended candidate.")
    md.append("")

    out_md = cfg.REPORTS_DIR / "phase3_15_pairwise_ranking.md"
    out_md.write_text("\n".join(md))
    _LOG.info("wrote %s", out_md)


if __name__ == "__main__":
    main()
