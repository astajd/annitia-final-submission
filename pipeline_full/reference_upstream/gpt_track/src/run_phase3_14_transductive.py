"""Phase 3.14 — controlled transductive / semi-supervised experiments.

Anchor: ``phase3_10_horizon_blend_v2`` (LB **0.91093**, weighted OOF 0.8811,
hep 0.8500, dea 0.9537).

Four tracks, run in order. Each track only contributes a candidate if it
clears its promotion bar; otherwise we record the result and move on:

- **Track A** — Train+test distribution audit and preprocessing variants
  (combined-train+test median imputation, percentile clipping). Refit the
  anchor's four horizon components per variant; check OOF deltas + rank-corr
  with anchor.

- **Track B** — Unsupervised cluster/embedding features (PCA-10 components,
  KMeans-8 cluster IDs + distances) fit on combined train+test of the
  ``current_state_v2`` feature space. Refit the anchor's horizon components
  with the augmented feature matrix.

- **Track C** — Extreme consensus pseudo-labelling for hepatic h3. Use the
  test predictions of phase3_5 v3 hepatic, phase3_9, phase3_10, plus the
  best hep h3 / h3.5 / h2.5 / h4 horizon classifiers, to build consensus
  pseudo-positive (top 7.5%) and pseudo-negative (bottom 22.5%) sets.
  Run a *simulated transductive 5-fold CV* (treat the validation fold as
  pseudo-test in each split). Promote only if simulated OOF improves and
  shifts are concentrated on the labelled extremes.

- **Track D** — Conservative kNN risk smoothing on the anchor's test
  predictions in the ``current_state_v2`` feature space (k=10, λ ∈
  {0.02, 0.05, 0.10}). OOF analog: smooth OOF using train-only neighbours
  in the same feature space.

Then assemble at most three submission candidates:

- ``phase3_14_transductive_preprocess_cluster.csv`` (Tracks A and/or B)
- ``phase3_14_extreme_pseudolabel_horizon.csv`` (Track C)
- ``phase3_14_knn_smooth.csv`` (Track D)

A combo file is emitted only if two independent transductive methods both
look useful and their combined rank shift is not excessive.
"""
from __future__ import annotations

import json
import time as _time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .data_loading import load_dataset
from .features import build_feature_set
from .features.refined_longitudinal import (
    LAB_STEMS,
    NIT_STEMS,
    longitudinal_no_followup_proxies,
)
from .features.hep_focus import current_state_v2_no_visit_history
from .horizon_targets import build_horizon_labels
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
# Shared helpers
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
    pool_h_oof = {"_v3": v3_h_oof}
    pool_d_oof = {"_v3": v3_d_oof}
    for k in weights_h:
        if k == "_v3":
            continue
        out = _load_horizon_artifact(k)
        if out is not None:
            pool_h_oof[k] = out[0]
    for k in weights_d:
        if k == "_v3":
            continue
        out = _load_horizon_artifact(k)
        if out is not None:
            pool_d_oof[k] = out[0]
    h_oof = rank_average(pool_h_oof, weights={k: weights_h[k] for k in pool_h_oof})
    d_oof = rank_average(pool_d_oof, weights={k: weights_d[k] for k in pool_d_oof})
    return h_oof, h_test, d_oof, d_test, meta


# ---------------------------------------------------------------------------
# Horizon retrain helper (used by Tracks A and B)
# ---------------------------------------------------------------------------

def _train_one_horizon_with_X(
    model_name: str,
    Xtr_arr: np.ndarray,
    Xte_arr: np.ndarray,
    horizon_label: np.ndarray,
    horizon_mask: np.ndarray,
    splits,
    *,
    n_train: int,
    n_test: int,
    seed: int = 0,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a horizon binary model with explicit feature arrays."""
    oof = np.full(n_train, np.nan)
    test_sum = np.zeros(n_test)
    test_n = 0

    for s in splits:
        tr_idx = np.array([i for i in s.train_idx if horizon_mask[i]], dtype=int)
        va_idx = np.array(s.valid_idx, dtype=int)
        if len(tr_idx) < 30 or horizon_label[tr_idx].sum() < 3:
            continue
        ytr = horizon_label[tr_idx].astype(int)
        wtr = sample_weight[tr_idx] if sample_weight is not None else None
        n_pos = max(int(ytr.sum()), 1)
        n_neg = max(int((1 - ytr).sum()), 1)
        spw = n_neg / n_pos
        if model_name == "lgbm_binary":
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, scale_pos_weight=spw, random_state=seed, verbose=-1,
            )
        elif model_name == "catboost_binary":
            from catboost import CatBoostClassifier
            clf = CatBoostClassifier(
                iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
                class_weights=[1.0, n_neg / n_pos], random_seed=seed,
                verbose=0, allow_writing_files=False,
            )
        else:
            raise KeyError(model_name)
        try:
            if wtr is not None:
                clf.fit(Xtr_arr[tr_idx], ytr, sample_weight=wtr)
            else:
                clf.fit(Xtr_arr[tr_idx], ytr)
            oof[va_idx] = clf.predict_proba(Xtr_arr[va_idx])[:, 1]
            test_sum += clf.predict_proba(Xte_arr)[:, 1]
            test_n += 1
        except Exception as e:  # noqa: BLE001
            _LOG.warning("fold %s seed=%d failed: %s", model_name, seed, e)
    test = test_sum / test_n if test_n else np.full(n_test, np.nan)
    return oof, test


# ---------------------------------------------------------------------------
# Track A1: distribution audit
# ---------------------------------------------------------------------------

def _distribution_audit(feature_sets: dict, ds) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute train vs test summary statistics per feature set.

    Returns (per_feature_df, per_set_summary_df).
    """
    rows: list[dict] = []
    for fs_name, blob in feature_sets.items():
        Xtr = blob["train"].select_dtypes(include=[np.number])
        Xte = blob["test"].select_dtypes(include=[np.number])
        common = [c for c in Xtr.columns if c in Xte.columns]
        for c in common:
            tr = Xtr[c].to_numpy()
            te = Xte[c].to_numpy()
            tr_finite = tr[np.isfinite(tr)]
            te_finite = te[np.isfinite(te)]
            miss_tr = float(np.mean(~np.isfinite(tr)))
            miss_te = float(np.mean(~np.isfinite(te)))
            mean_tr = float(np.nanmean(tr)) if tr_finite.size else float("nan")
            mean_te = float(np.nanmean(te)) if te_finite.size else float("nan")
            std_tr = float(np.nanstd(tr)) if tr_finite.size else float("nan")
            ks = float("nan")
            if tr_finite.size > 5 and te_finite.size > 5:
                try:
                    ks = float(ks_2samp(tr_finite, te_finite).statistic)
                except Exception:
                    pass
            mean_shift_z = (mean_te - mean_tr) / std_tr if std_tr and std_tr > 0 else float("nan")
            rows.append({
                "feature_set": fs_name, "feature": c,
                "miss_train": miss_tr, "miss_test": miss_te,
                "miss_diff": miss_te - miss_tr,
                "mean_train": mean_tr, "mean_test": mean_te,
                "mean_shift_z": mean_shift_z,
                "ks": ks,
            })
    feat_df = pd.DataFrame(rows)

    summary_rows: list[dict] = []
    for fs_name in feat_df["feature_set"].unique():
        sub = feat_df[feat_df["feature_set"] == fs_name]
        summary_rows.append({
            "feature_set": fs_name,
            "n_features": len(sub),
            "median_miss_diff": float(sub["miss_diff"].median()),
            "p95_miss_diff": float(sub["miss_diff"].abs().quantile(0.95)),
            "median_mean_shift_z_abs": float(sub["mean_shift_z"].abs().median()),
            "p95_mean_shift_z_abs": float(sub["mean_shift_z"].abs().quantile(0.95)),
            "median_ks": float(sub["ks"].median()),
            "p95_ks": float(sub["ks"].quantile(0.95)),
            "n_features_ks_gt_0_2": int((sub["ks"] > 0.2).sum()),
        })
    summary_df = pd.DataFrame(summary_rows)
    return feat_df, summary_df


# ---------------------------------------------------------------------------
# Track A2: preprocessing variants
# ---------------------------------------------------------------------------

@dataclass
class HorizonComponentSpec:
    label: str
    fs_name: str
    endpoint: str
    horizon: float
    model: str
    seed: int


def _anchor_components() -> list[HorizonComponentSpec]:
    return [
        HorizonComponentSpec("NIT_plus_scores__hepatic__h1__lgbm_binary",
                              "NIT_plus_scores", "hepatic", 1.0, "lgbm_binary", 0),
        HorizonComponentSpec("v3_hepatic_schema__hepatic__h3__lgbm_binary__s4",
                              "v3_hepatic_schema", "hepatic", 3.0, "lgbm_binary", 4),
        HorizonComponentSpec("current_state_v2__death__h5__catboost_binary__s3",
                              "current_state_v2", "death", 5.0, "catboost_binary", 3),
        HorizonComponentSpec("NIT_plus_scores__death__h4__catboost_binary",
                              "NIT_plus_scores", "death", 4.0, "catboost_binary", 0),
    ]


def _preprocess_combined_median(Xtr: pd.DataFrame, Xte: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    Xtr_p = fill_for_tree(Xtr).astype(np.float32, errors="ignore")
    Xte_p = fill_for_tree(Xte).reindex(columns=Xtr_p.columns, fill_value=0).astype(np.float32, errors="ignore")
    combined = pd.concat([Xtr_p, Xte_p], axis=0, ignore_index=True)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(combined.to_numpy())
    Xtr_arr = imputer.transform(Xtr_p.to_numpy())
    Xte_arr = imputer.transform(Xte_p.to_numpy())
    return Xtr_arr, Xte_arr


def _preprocess_combined_clipping(Xtr: pd.DataFrame, Xte: pd.DataFrame,
                                    lo: float = 1.0, hi: float = 99.0
                                    ) -> tuple[np.ndarray, np.ndarray]:
    Xtr_p = fill_for_tree(Xtr).astype(np.float32, errors="ignore")
    Xte_p = fill_for_tree(Xte).reindex(columns=Xtr_p.columns, fill_value=0).astype(np.float32, errors="ignore")
    combined = pd.concat([Xtr_p, Xte_p], axis=0, ignore_index=True).to_numpy()
    lo_arr = np.nanpercentile(combined, lo, axis=0)
    hi_arr = np.nanpercentile(combined, hi, axis=0)
    Xtr_arr = np.clip(Xtr_p.to_numpy(), lo_arr, hi_arr)
    Xte_arr = np.clip(Xte_p.to_numpy(), lo_arr, hi_arr)
    return Xtr_arr, Xte_arr


def _preprocess_default(Xtr: pd.DataFrame, Xte: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    Xtr_p = fill_for_tree(Xtr).astype(np.float32, errors="ignore")
    Xte_p = fill_for_tree(Xte).reindex(columns=Xtr_p.columns, fill_value=0).astype(np.float32, errors="ignore")
    return Xtr_p.to_numpy(), Xte_p.to_numpy()


# ---------------------------------------------------------------------------
# Track B: PCA + KMeans on combined train+test
# ---------------------------------------------------------------------------

def _build_unsupervised_features(Xtr: pd.DataFrame, Xte: pd.DataFrame,
                                   n_pca: int = 10, n_clusters: int = 8,
                                   seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xtr_p = fill_for_tree(Xtr).astype(np.float32, errors="ignore")
    Xte_p = fill_for_tree(Xte).reindex(columns=Xtr_p.columns, fill_value=0).astype(np.float32, errors="ignore")
    combined = pd.concat([Xtr_p, Xte_p], axis=0, ignore_index=True)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(imputer.fit_transform(combined.to_numpy()))
    pca = PCA(n_components=n_pca, random_state=seed)
    P = pca.fit_transform(Z)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(Z)
    centers = km.cluster_centers_
    dists = np.linalg.norm(Z[:, None, :] - centers[None, :, :], axis=2)  # (n, k)

    # Add columns
    pca_cols = [f"pca_{i}" for i in range(n_pca)]
    cluster_cols = [f"cluster_dist_{i}" for i in range(n_clusters)]
    new_df = pd.DataFrame(np.concatenate([P, dists], axis=1),
                           columns=pca_cols + cluster_cols)
    new_df["cluster_id"] = labels
    n_train = len(Xtr_p)
    new_train = new_df.iloc[:n_train].reset_index(drop=True)
    new_test = new_df.iloc[n_train:].reset_index(drop=True)
    return new_train, new_test


# ---------------------------------------------------------------------------
# Track C: extreme consensus pseudo-labelling with simulated transductive CV
# ---------------------------------------------------------------------------

def _build_consensus_extremes(test_pred_pool: dict[str, np.ndarray],
                                top_q: float = 0.075, bottom_q: float = 0.225,
                                min_models_in_extreme: int = 3
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (pseudo_pos_mask, pseudo_neg_mask) over test rows using the
    intersection-of-tops rule across the prediction pool."""
    n = len(next(iter(test_pred_pool.values())))
    high_top = np.zeros(n)
    low_bottom = np.zeros(n)
    for k, v in test_pred_pool.items():
        r = pd.Series(v).rank(pct=True).to_numpy()
        high_top += (r >= 1 - top_q).astype(int)
        low_bottom += (r <= bottom_q).astype(int)
    pos = high_top >= min_models_in_extreme
    neg = low_bottom >= min_models_in_extreme
    # Don't allow the same row to be both.
    overlap = pos & neg
    pos = pos & ~overlap
    neg = neg & ~overlap
    return pos, neg


def _simulated_transductive_cv(spec: HorizonComponentSpec, ds, hep, death, splits,
                                 X_train: pd.DataFrame, X_test: pd.DataFrame,
                                 horizon_label: np.ndarray, horizon_mask: np.ndarray,
                                 *, pseudo_top_q: float = 0.075,
                                 pseudo_bottom_q: float = 0.225,
                                 pseudo_weight: float = 0.10) -> tuple[np.ndarray, np.ndarray, dict]:
    """Treat each validation fold as 'test'. Generate consensus pseudo-labels
    on validation rows from training-fold predictions, then refit including
    those pseudo-labelled validation rows at low weight; predict validation.
    Returns (oof, test, info).
    """
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)
    Xtr_p = fill_for_tree(X_train).astype(np.float32, errors="ignore")
    Xte_p = fill_for_tree(X_test).reindex(columns=Xtr_p.columns, fill_value=0).astype(np.float32, errors="ignore")
    Xtr_arr = Xtr_p.to_numpy()
    Xte_arr = Xte_p.to_numpy()

    oof = np.full(n_train, np.nan)
    test_sum = np.zeros(n_test)
    test_n = 0
    n_pseudo_pos = 0
    n_pseudo_neg = 0

    for s in splits:
        tr_idx = np.array([i for i in s.train_idx if horizon_mask[i]], dtype=int)
        va_idx = np.array(s.valid_idx, dtype=int)
        if len(tr_idx) < 30 or horizon_label[tr_idx].sum() < 3:
            continue
        ytr = horizon_label[tr_idx].astype(int)
        n_pos = max(int(ytr.sum()), 1)
        n_neg = max(int((1 - ytr).sum()), 1)
        spw = n_neg / n_pos

        # Step 1: train initial classifier on training fold only.
        if spec.model == "lgbm_binary":
            import lightgbm as lgb
            clf0 = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, scale_pos_weight=spw, random_state=spec.seed, verbose=-1,
            )
        elif spec.model == "catboost_binary":
            from catboost import CatBoostClassifier
            clf0 = CatBoostClassifier(
                iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
                class_weights=[1.0, n_neg / n_pos], random_seed=spec.seed,
                verbose=0, allow_writing_files=False,
            )
        else:
            raise KeyError(spec.model)
        clf0.fit(Xtr_arr[tr_idx], ytr)
        proba_va = clf0.predict_proba(Xtr_arr[va_idx])[:, 1]

        # Step 2: select consensus extremes within the validation fold using a
        # *single* predictor (the one we just trained). Two-source consensus
        # would require additional fold-internal models; we keep the single-
        # model rule for simplicity but require very tight quantiles.
        n_va = len(va_idx)
        if n_va < 30:
            continue
        order = np.argsort(proba_va)
        n_top = max(int(round(pseudo_top_q * n_va)), 1)
        n_bottom = max(int(round(pseudo_bottom_q * n_va)), 1)
        pseudo_pos_local = order[-n_top:]
        pseudo_neg_local = order[:n_bottom]
        n_pseudo_pos += len(pseudo_pos_local)
        n_pseudo_neg += len(pseudo_neg_local)

        # Step 3: build augmented training set and refit.
        aug_idx = np.concatenate([tr_idx, va_idx[pseudo_pos_local], va_idx[pseudo_neg_local]])
        y_aug = np.concatenate([ytr,
                                 np.ones(len(pseudo_pos_local), dtype=int),
                                 np.zeros(len(pseudo_neg_local), dtype=int)])
        w_aug = np.concatenate([np.ones(len(tr_idx)),
                                 np.full(len(pseudo_pos_local), pseudo_weight),
                                 np.full(len(pseudo_neg_local), pseudo_weight)])
        if spec.model == "lgbm_binary":
            import lightgbm as lgb
            clf1 = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, scale_pos_weight=spw, random_state=spec.seed, verbose=-1,
            )
            clf1.fit(Xtr_arr[aug_idx], y_aug, sample_weight=w_aug)
        else:  # catboost
            from catboost import CatBoostClassifier
            clf1 = CatBoostClassifier(
                iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
                class_weights=[1.0, n_neg / n_pos], random_seed=spec.seed,
                verbose=0, allow_writing_files=False,
            )
            clf1.fit(Xtr_arr[aug_idx], y_aug, sample_weight=w_aug)

        oof[va_idx] = clf1.predict_proba(Xtr_arr[va_idx])[:, 1]
        test_sum += clf1.predict_proba(Xte_arr)[:, 1]
        test_n += 1

    test = test_sum / test_n if test_n else np.full(n_test, np.nan)
    info = {"n_pseudo_pos_total": int(n_pseudo_pos), "n_pseudo_neg_total": int(n_pseudo_neg)}
    return oof, test, info


# ---------------------------------------------------------------------------
# Track D: kNN risk smoothing
# ---------------------------------------------------------------------------

def _knn_smooth(scores: np.ndarray, neighbors: np.ndarray, lam: float) -> np.ndarray:
    """Smooth ``scores`` by averaging with the mean of each row's neighbours.

    ``neighbors`` is shape (n_target, k), entries are indices into ``scores``.
    """
    valid_neigh = neighbors >= 0
    nbr_score = np.where(valid_neigh, scores[neighbors], np.nan)
    nbr_mean = np.nanmean(nbr_score, axis=1)
    return (1 - lam) * scores + lam * nbr_mean


def _build_test_test_neighbors(Xtr: pd.DataFrame, Xte: pd.DataFrame,
                                k: int = 10) -> np.ndarray:
    """Return (n_test, k) array of test-row neighbour indices in the test set."""
    Xtr_p = fill_for_tree(Xtr).astype(np.float32, errors="ignore")
    Xte_p = fill_for_tree(Xte).reindex(columns=Xtr_p.columns, fill_value=0).astype(np.float32, errors="ignore")
    combined = pd.concat([Xtr_p, Xte_p], axis=0, ignore_index=True)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(imputer.fit_transform(combined.to_numpy()))
    n_train = len(Xtr_p)
    Z_test = Z[n_train:]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Z_test)
    _, idx = nn.kneighbors(Z_test)
    # Drop self (first neighbour).
    return idx[:, 1:]


def _build_train_train_neighbors(Xtr: pd.DataFrame, Xte: pd.DataFrame,
                                   k: int = 10) -> np.ndarray:
    """Return (n_train, k) array of train-row neighbour indices in the train set,
    excluding self. Combined train+test scaling but neighbour pool is train-only."""
    Xtr_p = fill_for_tree(Xtr).astype(np.float32, errors="ignore")
    Xte_p = fill_for_tree(Xte).reindex(columns=Xtr_p.columns, fill_value=0).astype(np.float32, errors="ignore")
    combined = pd.concat([Xtr_p, Xte_p], axis=0, ignore_index=True)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(imputer.fit_transform(combined.to_numpy()))
    n_train = len(Xtr_p)
    Z_train = Z[:n_train]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Z_train)
    _, idx = nn.kneighbors(Z_train)
    return idx[:, 1:]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    cfg.ensure_dirs()
    out_root = cfg.EXPERIMENT_OUTPUTS / "phase3_14_transductive"
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
    weights_h = anchor_meta["weights_hepatic"]
    weights_d = anchor_meta["weights_death"]
    _LOG.info("Anchor: hep=%.4f dea=%.4f weighted=%.4f", a_h_ci, a_d_ci, a_w)

    # ------------------------------------------------------------------
    # Track A1: Distribution audit
    # ------------------------------------------------------------------

    _LOG.info("Track A1: distribution audit")
    feat_df, summary_df = _distribution_audit(feature_sets, ds)
    feat_df.to_csv(out_root / "track_a1_feature_audit.csv", index=False)
    summary_df.to_csv(out_root / "track_a1_summary.csv", index=False)

    # Save audit report.
    audit_md: list[str] = []
    audit_md.append("# Phase 3.14 — Track A1: train vs test distribution audit\n")
    audit_md.append("Per feature set, per-feature drift metrics (KS statistic, mean shift, "
                     "missingness diff). Scoped to feature sets used by `phase3_10_horizon_blend_v2`:\n")
    audit_md.append("- `current_state_v2` (anchor's biggest hepatic+death contributor)")
    audit_md.append("- `v3_hepatic_schema` (hep h3 LGBM seed=4 schema)")
    audit_md.append("- `NIT_plus_scores` (hep h1 LGBM and dea h4 CatBoost)")
    audit_md.append("- `biomarker_only` (longitudinal_no_followup_proxies, used in death horizon ensembles)")
    audit_md.append("")
    audit_md.append("## Per feature set summary\n")
    audit_md.append(summary_df.to_markdown(index=False, floatfmt=".4f"))
    audit_md.append("")
    audit_md.append("## Top 25 features by KS across all sets\n")
    audit_md.append(feat_df.sort_values("ks", ascending=False).head(25)
                     .to_markdown(index=False, floatfmt=".4f"))
    audit_md.append("")
    audit_md.append("## Top 25 features by missingness diff (test - train)\n")
    audit_md.append(feat_df.assign(abs_miss_diff=feat_df["miss_diff"].abs())
                     .sort_values("abs_miss_diff", ascending=False).head(25)
                     [["feature_set","feature","miss_train","miss_test","miss_diff","ks"]]
                     .to_markdown(index=False, floatfmt=".4f"))
    audit_md.append("")
    (cfg.REPORTS_DIR / "phase3_14_train_test_distribution_audit.md").write_text("\n".join(audit_md))
    _LOG.info("wrote phase3_14_train_test_distribution_audit.md")

    # ------------------------------------------------------------------
    # Track A2: preprocessing variants on anchor's components
    # ------------------------------------------------------------------

    _LOG.info("Track A2: preprocessing variants")
    a2_results: list[dict] = []
    pp_variants = [
        ("default", _preprocess_default),
        ("combined_median_impute", _preprocess_combined_median),
        ("combined_clip_1_99", lambda Xtr, Xte: _preprocess_combined_clipping(Xtr, Xte, 1.0, 99.0)),
    ]

    component_predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}  # variant -> label -> {oof, test}
    for variant_name, pp_fn in pp_variants:
        component_predictions[variant_name] = {}
        for spec in _anchor_components():
            ep = hep if spec.endpoint == "hepatic" else death
            lab = build_horizon_labels(ep, spec.horizon, censored_mode="exclude")
            blob = feature_sets[spec.fs_name]
            t0 = _time.time()
            try:
                Xtr_arr, Xte_arr = pp_fn(blob["train"], blob["test"])
                oof, test = _train_one_horizon_with_X(
                    spec.model, Xtr_arr, Xte_arr,
                    lab.label, lab.mask, splits,
                    n_train=n_train, n_test=n_test, seed=spec.seed,
                )
            except Exception as e:  # noqa: BLE001
                _LOG.error("A2/%s/%s failed: %s", variant_name, spec.label, e)
                continue
            dt = _time.time() - t0
            ci = _surv_ci(ep.event, ep.time, oof)
            mean_, std_, min_ = _fold_ci(ep.event, ep.time, oof, splits)
            anchor_artifact = _load_horizon_artifact(spec.label)
            anchor_oof = anchor_artifact[0] if anchor_artifact else None
            rho_oof = _spearman(oof, anchor_oof) if anchor_oof is not None else float("nan")
            a2_results.append({
                "variant": variant_name, "label": spec.label,
                "endpoint": spec.endpoint, "horizon": spec.horizon, "model": spec.model,
                "ci": ci, "fold_std": std_, "fold_min": min_,
                "rho_with_anchor_component_oof": rho_oof,
                "wall_seconds": dt,
            })
            component_predictions[variant_name][spec.label] = {"oof": oof, "test": test}
            _LOG.info("A2 variant=%s component=%s ci=%.4f rho=%.3f (%.1fs)",
                      variant_name, spec.label, ci, rho_oof, dt)
    a2_df = pd.DataFrame(a2_results)
    a2_df.to_csv(out_root / "track_a2_results.csv", index=False)

    # Build a "Track A2 candidate" by replacing the anchor's horizon components
    # with the best-performing variant per component.
    track_a_oof_h = a_h_oof
    track_a_oof_d = a_d_oof
    track_a_test_h = a_h_test
    track_a_test_d = a_d_test
    track_a_active = False
    if not a2_df.empty:
        # For each component, pick the variant with best ci (only if it
        # improves the default).
        best_variants: dict[str, str] = {}
        for spec in _anchor_components():
            sub = a2_df[a2_df["label"] == spec.label]
            if sub.empty:
                continue
            default_ci = sub.loc[sub["variant"] == "default", "ci"].squeeze() \
                if (sub["variant"] == "default").any() else float("nan")
            sub_better = sub[sub["ci"] > (default_ci if np.isfinite(default_ci) else -np.inf) + 0.001]
            if sub_better.empty:
                continue
            best_row = sub_better.loc[sub_better["ci"].idxmax()]
            best_variants[spec.label] = str(best_row["variant"])
        if best_variants:
            track_a_active = True
            # Rebuild hep / dea OOF + test by mixing in the new component
            # predictions (using anchor's weights).
            new_h_pool = {"_v3": _v3_oof(n_train, ds, hep)[0]}
            v3_csv = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.csv"))[-1]
            v3_pub = pd.read_csv(v3_csv)
            v3_h_test = v3_pub[cfg.SUB_HEPATIC_COL].to_numpy()
            v3_d_test = v3_pub[cfg.SUB_DEATH_COL].to_numpy()
            v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)
            new_h_oof_pool = {"_v3": v3_h_oof}
            new_h_test_pool = {"_v3": v3_h_test}
            new_d_oof_pool = {"_v3": v3_d_oof}
            new_d_test_pool = {"_v3": v3_d_test}
            for spec in _anchor_components():
                if spec.label in best_variants:
                    src = component_predictions[best_variants[spec.label]][spec.label]
                else:
                    src = component_predictions["default"].get(spec.label) \
                          or {"oof": _load_horizon_artifact(spec.label)[0],
                              "test": _load_horizon_artifact(spec.label)[1]}
                if spec.endpoint == "hepatic":
                    new_h_oof_pool[spec.label] = src["oof"]
                    new_h_test_pool[spec.label] = src["test"]
                else:
                    new_d_oof_pool[spec.label] = src["oof"]
                    new_d_test_pool[spec.label] = src["test"]
            track_a_oof_h = rank_average(new_h_oof_pool, weights={k: weights_h[k] for k in new_h_oof_pool})
            track_a_test_h = rank_average(new_h_test_pool, weights={k: weights_h[k] for k in new_h_test_pool})
            track_a_oof_d = rank_average(new_d_oof_pool, weights={k: weights_d[k] for k in new_d_oof_pool})
            track_a_test_d = rank_average(new_d_test_pool, weights={k: weights_d[k] for k in new_d_test_pool})

    track_a_h_ci = _surv_ci(hep.event, hep.time, track_a_oof_h)
    track_a_d_ci = _surv_ci(death.event, death.time, track_a_oof_d)
    track_a_w = weighted_score(track_a_h_ci, track_a_d_ci)
    _LOG.info("Track A composite OOF: hep=%.4f dea=%.4f weighted=%.4f (active=%s)",
              track_a_h_ci, track_a_d_ci, track_a_w, track_a_active)

    # ------------------------------------------------------------------
    # Track B: PCA + KMeans on combined train+test of current_state_v2
    # ------------------------------------------------------------------

    _LOG.info("Track B: PCA + KMeans cluster features")
    b_results: list[dict] = []
    cs_blob = feature_sets["current_state_v2"]
    pca_train, pca_test = _build_unsupervised_features(cs_blob["train"], cs_blob["test"],
                                                          n_pca=10, n_clusters=8, seed=0)
    component_predictions_b: dict[str, dict[str, np.ndarray]] = {}

    for spec in _anchor_components():
        ep = hep if spec.endpoint == "hepatic" else death
        lab = build_horizon_labels(ep, spec.horizon, censored_mode="exclude")
        # Augment the spec's native feature set with PCA + cluster features.
        blob = feature_sets[spec.fs_name]
        Xtr_aug = pd.concat([blob["train"].reset_index(drop=True), pca_train], axis=1)
        Xte_aug = pd.concat([blob["test"].reset_index(drop=True), pca_test], axis=1)
        Xtr_arr, Xte_arr = _preprocess_default(Xtr_aug, Xte_aug)
        try:
            oof, test = _train_one_horizon_with_X(
                spec.model, Xtr_arr, Xte_arr,
                lab.label, lab.mask, splits,
                n_train=n_train, n_test=n_test, seed=spec.seed,
            )
        except Exception as e:  # noqa: BLE001
            _LOG.error("B/%s failed: %s", spec.label, e)
            continue
        ci = _surv_ci(ep.event, ep.time, oof)
        mean_, std_, min_ = _fold_ci(ep.event, ep.time, oof, splits)
        anchor_artifact = _load_horizon_artifact(spec.label)
        anchor_oof = anchor_artifact[0] if anchor_artifact else None
        rho_oof = _spearman(oof, anchor_oof) if anchor_oof is not None else float("nan")
        b_results.append({
            "label": spec.label, "endpoint": spec.endpoint, "horizon": spec.horizon,
            "ci": ci, "fold_std": std_, "fold_min": min_,
            "rho_with_anchor_component_oof": rho_oof,
        })
        component_predictions_b[spec.label] = {"oof": oof, "test": test}
        _LOG.info("B %s ci=%.4f rho=%.3f", spec.label, ci, rho_oof)
    b_df = pd.DataFrame(b_results)
    b_df.to_csv(out_root / "track_b_results.csv", index=False)

    # Track B candidate composite (replace anchor components only when augmented improves).
    track_b_active = False
    track_b_oof_h = a_h_oof
    track_b_oof_d = a_d_oof
    track_b_test_h = a_h_test
    track_b_test_d = a_d_test
    if not b_df.empty:
        v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)
        v3_csv = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.csv"))[-1]
        v3_pub = pd.read_csv(v3_csv)
        v3_h_test = v3_pub[cfg.SUB_HEPATIC_COL].to_numpy()
        v3_d_test = v3_pub[cfg.SUB_DEATH_COL].to_numpy()
        new_h_oof_pool = {"_v3": v3_h_oof}
        new_h_test_pool = {"_v3": v3_h_test}
        new_d_oof_pool = {"_v3": v3_d_oof}
        new_d_test_pool = {"_v3": v3_d_test}
        any_improved = False
        for spec in _anchor_components():
            anchor_artifact = _load_horizon_artifact(spec.label)
            anchor_ci = _surv_ci(hep.event if spec.endpoint == "hepatic" else death.event,
                                  hep.time  if spec.endpoint == "hepatic" else death.time,
                                  anchor_artifact[0]) if anchor_artifact else float("-inf")
            if spec.label in component_predictions_b:
                src = component_predictions_b[spec.label]
                new_ci = _surv_ci(hep.event if spec.endpoint == "hepatic" else death.event,
                                    hep.time  if spec.endpoint == "hepatic" else death.time,
                                    src["oof"])
                if np.isfinite(new_ci) and new_ci > anchor_ci + 0.001:
                    any_improved = True
                    if spec.endpoint == "hepatic":
                        new_h_oof_pool[spec.label] = src["oof"]
                        new_h_test_pool[spec.label] = src["test"]
                    else:
                        new_d_oof_pool[spec.label] = src["oof"]
                        new_d_test_pool[spec.label] = src["test"]
                    continue
            # Fall back to anchor predictions.
            if anchor_artifact is not None:
                if spec.endpoint == "hepatic":
                    new_h_oof_pool[spec.label] = anchor_artifact[0]
                    new_h_test_pool[spec.label] = anchor_artifact[1]
                else:
                    new_d_oof_pool[spec.label] = anchor_artifact[0]
                    new_d_test_pool[spec.label] = anchor_artifact[1]
        if any_improved:
            track_b_active = True
            track_b_oof_h = rank_average(new_h_oof_pool, weights={k: weights_h[k] for k in new_h_oof_pool})
            track_b_test_h = rank_average(new_h_test_pool, weights={k: weights_h[k] for k in new_h_test_pool})
            track_b_oof_d = rank_average(new_d_oof_pool, weights={k: weights_d[k] for k in new_d_oof_pool})
            track_b_test_d = rank_average(new_d_test_pool, weights={k: weights_d[k] for k in new_d_test_pool})

    track_b_h_ci = _surv_ci(hep.event, hep.time, track_b_oof_h)
    track_b_d_ci = _surv_ci(death.event, death.time, track_b_oof_d)
    track_b_w = weighted_score(track_b_h_ci, track_b_d_ci)
    _LOG.info("Track B composite OOF: hep=%.4f dea=%.4f weighted=%.4f (active=%s)",
              track_b_h_ci, track_b_d_ci, track_b_w, track_b_active)

    # ------------------------------------------------------------------
    # Track C: extreme consensus pseudo-label for the hepatic h3 component
    # ------------------------------------------------------------------

    _LOG.info("Track C: extreme consensus pseudo-labelling (hep h3 LGBM)")
    c_results: list[dict] = []
    pseudo_thresholds = [(0.05, 0.20), (0.075, 0.225), (0.10, 0.25)]
    pseudo_weights = [0.05, 0.10, 0.20]
    spec_c = HorizonComponentSpec("v3_hepatic_schema__hepatic__h3__lgbm_binary__s4",
                                    "v3_hepatic_schema", "hepatic", 3.0, "lgbm_binary", 4)
    blob_c = feature_sets[spec_c.fs_name]
    lab_c = build_horizon_labels(hep, spec_c.horizon, censored_mode="exclude")

    track_c_best_oof = None
    track_c_best_test = None
    track_c_best_meta = {}

    # Reference: standard OOF with no pseudo-labels.
    ref_oof, ref_test = _train_one_horizon_with_X(
        spec_c.model, *_preprocess_default(blob_c["train"], blob_c["test"]),
        lab_c.label, lab_c.mask, splits,
        n_train=n_train, n_test=n_test, seed=spec_c.seed,
    )
    ref_ci = _surv_ci(hep.event, hep.time, ref_oof)
    _LOG.info("Track C reference (no pseudo): ci=%.4f", ref_ci)

    for top_q, bottom_q in pseudo_thresholds:
        for pl_w in pseudo_weights:
            try:
                oof_c, test_c, info = _simulated_transductive_cv(
                    spec_c, ds, hep, death, splits,
                    blob_c["train"], blob_c["test"], lab_c.label, lab_c.mask,
                    pseudo_top_q=top_q, pseudo_bottom_q=bottom_q,
                    pseudo_weight=pl_w,
                )
            except Exception as e:  # noqa: BLE001
                _LOG.error("C top=%.3f bot=%.3f w=%.2f failed: %s", top_q, bottom_q, pl_w, e)
                continue
            ci = _surv_ci(hep.event, hep.time, oof_c)
            mean_, std_, min_ = _fold_ci(hep.event, hep.time, oof_c, splits)
            rho_test = _spearman(test_c, _load_horizon_artifact(spec_c.label)[1]
                                  if _load_horizon_artifact(spec_c.label) else None)
            c_results.append({
                "top_q": top_q, "bottom_q": bottom_q, "pseudo_weight": pl_w,
                "ci": ci, "ci_delta_vs_ref": ci - ref_ci,
                "fold_std": std_, "fold_min": min_,
                "rho_test_anchor_component": rho_test,
                "n_pseudo_pos_total": info["n_pseudo_pos_total"],
                "n_pseudo_neg_total": info["n_pseudo_neg_total"],
            })
            _LOG.info("C top=%.3f bot=%.3f w=%.2f ci=%.4f Δ=%.4f", top_q, bottom_q, pl_w, ci, ci - ref_ci)
            if track_c_best_meta.get("ci", -np.inf) < ci:
                track_c_best_meta = {"top_q": top_q, "bottom_q": bottom_q,
                                      "pseudo_weight": pl_w, "ci": ci,
                                      "ci_delta_vs_ref": ci - ref_ci, "fold_std": std_, "fold_min": min_}
                track_c_best_oof = oof_c
                track_c_best_test = test_c
    c_df = pd.DataFrame(c_results)
    c_df.to_csv(out_root / "track_c_results.csv", index=False)

    track_c_active = False
    track_c_oof_h = a_h_oof
    track_c_test_h = a_h_test
    track_c_oof_d = a_d_oof
    track_c_test_d = a_d_test
    if track_c_best_meta and track_c_best_meta.get("ci_delta_vs_ref", -np.inf) > 0.001:
        # Replace the hep h3 component in the anchor's recipe with the pseudo-labelled version.
        v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)
        v3_csv = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.csv"))[-1]
        v3_pub = pd.read_csv(v3_csv)
        v3_h_test = v3_pub[cfg.SUB_HEPATIC_COL].to_numpy()
        v3_d_test = v3_pub[cfg.SUB_DEATH_COL].to_numpy()
        new_h_oof_pool = {"_v3": v3_h_oof}
        new_h_test_pool = {"_v3": v3_h_test}
        for spec in _anchor_components():
            if spec.endpoint != "hepatic":
                continue
            if spec.label == spec_c.label:
                new_h_oof_pool[spec.label] = track_c_best_oof
                new_h_test_pool[spec.label] = track_c_best_test
            else:
                art = _load_horizon_artifact(spec.label)
                if art:
                    new_h_oof_pool[spec.label] = art[0]
                    new_h_test_pool[spec.label] = art[1]
        track_c_oof_h = rank_average(new_h_oof_pool, weights={k: weights_h[k] for k in new_h_oof_pool})
        track_c_test_h = rank_average(new_h_test_pool, weights={k: weights_h[k] for k in new_h_test_pool})
        # Death side unchanged.
        track_c_active = True

    track_c_h_ci = _surv_ci(hep.event, hep.time, track_c_oof_h)
    track_c_d_ci = _surv_ci(death.event, death.time, track_c_oof_d)
    track_c_w = weighted_score(track_c_h_ci, track_c_d_ci)
    _LOG.info("Track C composite OOF: hep=%.4f dea=%.4f weighted=%.4f (active=%s)",
              track_c_h_ci, track_c_d_ci, track_c_w, track_c_active)

    # ------------------------------------------------------------------
    # Track D: kNN smoothing
    # ------------------------------------------------------------------

    _LOG.info("Track D: kNN smoothing on current_state_v2 feature space")
    d_results: list[dict] = []
    cs_blob = feature_sets["current_state_v2"]
    test_test_neigh = _build_test_test_neighbors(cs_blob["train"], cs_blob["test"], k=10)
    train_train_neigh = _build_train_train_neighbors(cs_blob["train"], cs_blob["test"], k=10)

    track_d_best_meta = {}
    track_d_best_oof_h = None
    track_d_best_test_h = None
    track_d_best_oof_d = None
    track_d_best_test_d = None
    for k in (10,):
        for lam in (0.02, 0.05, 0.10):
            # OOF: smooth with train-train neighbours.
            sm_h_oof = _knn_smooth(a_h_oof, train_train_neigh, lam)
            sm_d_oof = _knn_smooth(a_d_oof, train_train_neigh, lam)
            ci_h = _surv_ci(hep.event, hep.time, sm_h_oof)
            ci_d = _surv_ci(death.event, death.time, sm_d_oof)
            ws = weighted_score(ci_h, ci_d)
            mean_h, std_h, min_h = _fold_ci(hep.event, hep.time, sm_h_oof, splits)
            mean_d, std_d, min_d = _fold_ci(death.event, death.time, sm_d_oof, splits)
            # Test: smooth with test-test neighbours.
            sm_h_test = _knn_smooth(a_h_test, test_test_neigh, lam)
            sm_d_test = _knn_smooth(a_d_test, test_test_neigh, lam)
            rho_h = _spearman(sm_h_test, a_h_test)
            rho_d = _spearman(sm_d_test, a_d_test)
            d_results.append({
                "k": k, "lam": lam,
                "hep_oof": ci_h, "death_oof": ci_d, "weighted_oof": ws,
                "hep_fold_std": std_h, "hep_fold_min": min_h,
                "dea_fold_std": std_d, "dea_fold_min": min_d,
                "delta_w_vs_anchor": ws - a_w,
                "delta_h_vs_anchor": ci_h - a_h_ci,
                "delta_d_vs_anchor": ci_d - a_d_ci,
                "rho_h_test_anchor": rho_h,
                "rho_d_test_anchor": rho_d,
            })
            _LOG.info("D k=%d lam=%.2f w=%.4f Δ=%+.4f hep=%.4f dea=%.4f", k, lam, ws, ws - a_w, ci_h, ci_d)
            if (track_d_best_meta.get("weighted_oof", -np.inf) < ws):
                track_d_best_meta = {
                    "k": k, "lam": lam,
                    "hep_oof": ci_h, "death_oof": ci_d, "weighted_oof": ws,
                    "delta_w_vs_anchor": ws - a_w,
                    "delta_h_vs_anchor": ci_h - a_h_ci,
                    "hep_fold_std": std_h, "hep_fold_min": min_h,
                    "dea_fold_std": std_d, "dea_fold_min": min_d,
                    "rho_h_test_anchor": rho_h, "rho_d_test_anchor": rho_d,
                }
                track_d_best_oof_h = sm_h_oof
                track_d_best_test_h = sm_h_test
                track_d_best_oof_d = sm_d_oof
                track_d_best_test_d = sm_d_test
    d_df = pd.DataFrame(d_results)
    d_df.to_csv(out_root / "track_d_results.csv", index=False)

    track_d_active = False
    track_d_oof_h = a_h_oof
    track_d_test_h = a_h_test
    track_d_oof_d = a_d_oof
    track_d_test_d = a_d_test
    if track_d_best_meta:
        if track_d_best_meta["weighted_oof"] > a_w + 0.0005:
            track_d_active = True
            track_d_oof_h = track_d_best_oof_h
            track_d_test_h = track_d_best_test_h
            track_d_oof_d = track_d_best_oof_d
            track_d_test_d = track_d_best_test_d

    # ------------------------------------------------------------------
    # Track E: candidate emission
    # ------------------------------------------------------------------

    candidates: dict[str, dict] = {}

    def _candidate_meta(label, source_track, h_oof, d_oof, h_test, d_test, extra=None):
        ci_h = _surv_ci(hep.event, hep.time, h_oof)
        ci_d = _surv_ci(death.event, death.time, d_oof)
        ws = weighted_score(ci_h, ci_d)
        h_mean, h_std, h_min = _fold_ci(hep.event, hep.time, h_oof, splits)
        d_mean, d_std, d_min = _fold_ci(death.event, death.time, d_oof, splits)
        rho_h_test = _spearman(h_test, a_h_test)
        rho_d_test = _spearman(d_test, a_d_test)
        rho_h_oof = _spearman(h_oof, a_h_oof)
        rho_d_oof = _spearman(d_oof, a_d_oof)
        h_rank = pd.Series(h_test).rank(pct=True).to_numpy()
        d_rank = pd.Series(d_test).rank(pct=True).to_numpy()
        a_h_rank = pd.Series(a_h_test).rank(pct=True).to_numpy()
        a_d_rank = pd.Series(a_d_test).rank(pct=True).to_numpy()
        h_shift = h_rank - a_h_rank
        d_shift = d_rank - a_d_rank
        return {
            "label": label, "source_track": source_track,
            "hepatic_oof": ci_h, "death_oof": ci_d, "weighted_oof": ws,
            "delta_weighted": ws - a_w, "delta_hep": ci_h - a_h_ci, "delta_dea": ci_d - a_d_ci,
            "hep_fold_std": h_std, "hep_fold_min": h_min,
            "dea_fold_std": d_std, "dea_fold_min": d_min,
            "rho_h_test_anchor": rho_h_test, "rho_d_test_anchor": rho_d_test,
            "rho_h_oof_anchor": rho_h_oof, "rho_d_oof_anchor": rho_d_oof,
            "h_test_rank_shift_max_abs": float(np.max(np.abs(h_shift))),
            "d_test_rank_shift_max_abs": float(np.max(np.abs(d_shift))),
            "h_test_rank_shift_mean_abs": float(np.mean(np.abs(h_shift))),
            "d_test_rank_shift_mean_abs": float(np.mean(np.abs(d_shift))),
            "uses_target_derived_features": False,
            "uses_test_features": True,
            "h_test": h_test, "d_test": d_test,
            "h_oof": h_oof, "d_oof": d_oof,
            **(extra or {}),
        }

    # Candidate 1 (Track A and/or B): preprocess + cluster
    pc_h_oof = track_b_oof_h if track_b_active else track_a_oof_h
    pc_h_test = track_b_test_h if track_b_active else track_a_test_h
    pc_d_oof = track_b_oof_d if track_b_active else track_a_oof_d
    pc_d_test = track_b_test_d if track_b_active else track_a_test_d
    if track_a_active or track_b_active:
        candidates["phase3_14_transductive_preprocess_cluster"] = _candidate_meta(
            "phase3_14_transductive_preprocess_cluster",
            "track_A_or_B", pc_h_oof, pc_d_oof, pc_h_test, pc_d_test,
            extra={"track_a_active": track_a_active, "track_b_active": track_b_active,
                    "uses_pseudo_labels": False},
        )

    # Candidate 2 (Track C): extreme pseudo-label
    if track_c_active:
        candidates["phase3_14_extreme_pseudolabel_horizon"] = _candidate_meta(
            "phase3_14_extreme_pseudolabel_horizon",
            "track_C", track_c_oof_h, track_c_oof_d, track_c_test_h, track_c_test_d,
            extra={"pseudo_label_meta": track_c_best_meta,
                    "uses_pseudo_labels": True, "track_c_active": True},
        )

    # Candidate 3 (Track D): kNN smooth
    if track_d_active:
        candidates["phase3_14_knn_smooth"] = _candidate_meta(
            "phase3_14_knn_smooth",
            "track_D", track_d_oof_h, track_d_oof_d, track_d_test_h, track_d_test_d,
            extra={"knn_meta": track_d_best_meta, "uses_pseudo_labels": False, "track_d_active": True},
        )

    # Candidate 4 (combo): only if two tracks both look useful and combined shifts modest.
    active_tracks = sum([track_a_active or track_b_active, track_c_active, track_d_active])
    if active_tracks >= 2:
        # Average ranks of the active candidates.
        active_h = []
        active_d = []
        active_t_h = []
        active_t_d = []
        for cn in ("phase3_14_transductive_preprocess_cluster",
                   "phase3_14_extreme_pseudolabel_horizon",
                   "phase3_14_knn_smooth"):
            if cn in candidates:
                active_h.append(candidates[cn]["h_oof"])
                active_d.append(candidates[cn]["d_oof"])
                active_t_h.append(candidates[cn]["h_test"])
                active_t_d.append(candidates[cn]["d_test"])
        combo_h_oof = rank_average({f"k{i}": a for i, a in enumerate(active_h)})
        combo_d_oof = rank_average({f"k{i}": a for i, a in enumerate(active_d)})
        combo_h_test = rank_average({f"k{i}": a for i, a in enumerate(active_t_h)})
        combo_d_test = rank_average({f"k{i}": a for i, a in enumerate(active_t_d)})
        # Shift sanity check.
        h_rank_combo = pd.Series(combo_h_test).rank(pct=True).to_numpy()
        a_h_rank = pd.Series(a_h_test).rank(pct=True).to_numpy()
        max_h_shift = float(np.max(np.abs(h_rank_combo - a_h_rank)))
        if max_h_shift < 0.20:
            candidates["phase3_14_transductive_combo"] = _candidate_meta(
                "phase3_14_transductive_combo",
                "combo", combo_h_oof, combo_d_oof, combo_h_test, combo_d_test,
                extra={"composing_tracks": [t for t in ("A_or_B", "C", "D")
                                              if (t == "A_or_B" and (track_a_active or track_b_active))
                                              or (t == "C" and track_c_active)
                                              or (t == "D" and track_d_active)],
                        "uses_pseudo_labels": track_c_active},
            )

    # Apply promotion criteria and decide submission recommendation.
    promotion_log = {}
    for name, meta in candidates.items():
        d_w = meta["delta_weighted"]
        d_h = meta["delta_hep"]
        max_shift = max(meta["h_test_rank_shift_max_abs"], meta["d_test_rank_shift_max_abs"])
        promotes = (d_w >= 0.002 or d_h >= 0.003)
        very_stable = max_shift < 0.10
        promotion_log[name] = {
            "promotes_by_oof": bool(promotes),
            "very_stable": bool(very_stable),
            "delta_weighted": d_w, "delta_hep": d_h,
            "max_shift_test": max_shift,
        }
    promoted_name = None
    if candidates:
        # Pick the candidate with the largest weighted OOF delta among those
        # that meet the promotion bar; break ties by hepatic delta then
        # smaller max-shift.
        valid = [(name, meta) for name, meta in candidates.items()
                  if meta["delta_weighted"] >= 0.002 or meta["delta_hep"] >= 0.003]
        if valid:
            valid.sort(key=lambda nm: (-nm[1]["delta_weighted"], -nm[1]["delta_hep"],
                                         nm[1]["h_test_rank_shift_max_abs"]))
            promoted_name = valid[0][0]
        else:
            # No OOF improvement; consider stability rationale.
            stable = [(name, meta) for name, meta in candidates.items()
                       if meta["delta_weighted"] >= -0.0005
                       and (meta["hep_fold_std"] < a_h_fold[1] - 0.005
                            or meta["dea_fold_std"] < a_d_fold[1] - 0.0010)]
            if stable:
                stable.sort(key=lambda nm: (nm[1]["hep_fold_std"], nm[1]["dea_fold_std"]))
                promoted_name = stable[0][0]

    # Emit candidate CSVs and JSON sidecars.
    for name, meta in candidates.items():
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=meta["h_test"], risk_death=meta["d_test"],
            sample_submission=ds.sample_submission, model_name=name,
        )
        meta_to_persist = {k: v for k, v in meta.items() if k not in ("h_test", "d_test", "h_oof", "d_oof")}
        meta_to_persist["submission_csv"] = str(sub_path)
        meta_to_persist["promotion"] = promotion_log.get(name, {})
        meta_to_persist["recommended"] = (name == promoted_name)
        sub_path.with_suffix(".json").write_text(json.dumps(meta_to_persist, indent=2, default=str))

    # ------------------------------------------------------------------
    # Final summary report
    # ------------------------------------------------------------------

    md: list[str] = []
    md.append("# Phase 3.14 — controlled transductive / semi-supervised summary\n")
    md.append("Anchor: `phase3_10_horizon_blend_v2` (LB **0.91093**, weighted OOF "
              f"{a_w:.4f}, hep {a_h_ci:.4f}, dea {a_d_ci:.4f}).\n")
    md.append(f"Anchor fold: hep std/min={a_h_fold[1]:.4f}/{a_h_fold[2]:.4f}, "
              f"dea std/min={a_d_fold[1]:.4f}/{a_d_fold[2]:.4f}.\n")

    md.append("## 1. Methods tried\n")
    md.append("| track | description | active? |")
    md.append("|---|---|---|")
    md.append(f"| A1 | train/test distribution audit | report only |")
    md.append(f"| A2 | combined train+test preprocessing variants | {'yes' if track_a_active else 'no'} |")
    md.append(f"| B | PCA-10 + KMeans-8 cluster features (current_state_v2) | {'yes' if track_b_active else 'no'} |")
    md.append(f"| C | extreme consensus pseudo-labelling (hep h3 LGBM) | {'yes' if track_c_active else 'no'} |")
    md.append(f"| D | conservative kNN smoothing (current_state_v2) | {'yes' if track_d_active else 'no'} |")
    md.append("")

    md.append("## 2. Track A1 — distribution audit summary\n")
    md.append(summary_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## 3. Track A2 — preprocessing variants per anchor component\n")
    if not a2_df.empty:
        md.append(a2_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## 4. Track B — cluster-feature augmented anchor components\n")
    if not b_df.empty:
        md.append(b_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## 5. Track C — pseudo-label sweep (hep h3 LGBM, simulated transductive 5x3 CV)\n")
    md.append(f"Reference (no pseudo-label) C-index: {ref_ci:.4f}\n")
    if not c_df.empty:
        md.append(c_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## 6. Track D — kNN smoothing (current_state_v2, k=10)\n")
    if not d_df.empty:
        md.append(d_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## 7. Candidates emitted\n")
    if not candidates:
        md.append("**No candidates emitted.** No track produced an OOF improvement that "
                   "met the +0.002 weighted or +0.003 hepatic threshold, or a stability "
                   "rationale strong enough to override.\n")
    else:
        rows_cand = []
        for name, meta in candidates.items():
            rows_cand.append({
                "candidate": name,
                "weighted_oof": meta["weighted_oof"],
                "delta_weighted": meta["delta_weighted"],
                "hep_oof": meta["hepatic_oof"],
                "delta_hep": meta["delta_hep"],
                "death_oof": meta["death_oof"],
                "delta_dea": meta["delta_dea"],
                "hep_fold_std": meta["hep_fold_std"],
                "dea_fold_std": meta["dea_fold_std"],
                "rho_h_test_anchor": meta["rho_h_test_anchor"],
                "rho_d_test_anchor": meta["rho_d_test_anchor"],
                "h_test_max_shift": meta["h_test_rank_shift_max_abs"],
                "d_test_max_shift": meta["d_test_rank_shift_max_abs"],
                "recommended": (name == promoted_name),
            })
        md.append(pd.DataFrame(rows_cand).to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## 8. Recommendation\n")
    if promoted_name is None:
        md.append(
            "**No submission recommended.** None of the four transductive tracks "
            "delivered an OOF improvement of ≥ +0.002 weighted or ≥ +0.003 "
            "hepatic over `phase3_10_horizon_blend_v2`, and none had a stability "
            "rationale strong enough to override that. Hold the anchor and "
            "explore non-transductive directions next.\n"
        )
    else:
        meta = candidates[promoted_name]
        md.append(
            f"**Submit one**: `submissions/.../{promoted_name}.csv`\n"
            f"- weighted OOF Δ vs anchor: {meta['delta_weighted']:+.4f}\n"
            f"- hepatic OOF Δ vs anchor: {meta['delta_hep']:+.4f}\n"
            f"- death OOF Δ vs anchor: {meta['delta_dea']:+.4f}\n"
            f"- rank-corr w/ anchor (test): hep={meta['rho_h_test_anchor']:.3f}, "
            f"dea={meta['rho_d_test_anchor']:.3f}\n"
            f"- max test rank shift: hep={meta['h_test_rank_shift_max_abs']:.3f}, "
            f"dea={meta['d_test_rank_shift_max_abs']:.3f}\n"
        )

    md.append("\n## 9. Notes\n")
    md.append("- All transductive methods use unlabeled test features only — no event/")
    md.append("  censoring-age columns are touched.")
    md.append("- Track C uses a single-source consensus rule for simplicity (the model "
               "trained on the training fold acts as the consensus); we keep top/bottom "
               "quantiles tight to reduce reliance on this proxy.")
    md.append("- Track D smooths in the `current_state_v2` feature space; train-train "
               "neighbours are used for the OOF analog, test-test for the test "
               "predictions.")
    md.append("- All evaluations are OOF only; we never tuned weights, thresholds, or "
               "selection on the public LB.")
    md.append("")

    out_md = cfg.REPORTS_DIR / "phase3_14_transductive_summary.md"
    out_md.write_text("\n".join(md))
    _LOG.info("wrote %s", out_md)


if __name__ == "__main__":
    main()
