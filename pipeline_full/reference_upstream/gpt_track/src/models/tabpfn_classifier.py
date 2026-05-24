"""TabPFN binary-classifier wrapper.

Used as a ranking component (we evaluate it with the survival C-index against
the original endpoint, not log-loss). Binary label = "event observed in the
training window". Risk score = predicted positive-class probability.

Per fold the wrapper:
1. Optional fold-internal feature screen (top-k by ANOVA-F).
2. Median imputation; optional rank-quantile transform.
3. Fit TabPFN classifier on the fold's training rows.
4. Return calibrated proba on the validation rows + average over folds for test.

TabPFN 2.x is run in CPU mode here; ``ignore_pretraining_limits`` is enabled so
we can use up to ~200 features (the official limit is 100).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils import get_logger

_LOG = get_logger(__name__)


@dataclass
class TabPFNRunResult:
    feature_set_name: str
    endpoint: str
    preprocess: str
    n_features: int
    fold_scores: list[float]
    oof: np.ndarray
    test: np.ndarray
    fit_seconds_total: float


class _RankQuantileTransform:
    """Per-feature rank-to-uniform transform fit on training rows only."""

    def fit(self, X: np.ndarray) -> "_RankQuantileTransform":
        # Sort each column; keep sorted values for inverse-CDF lookup.
        self._sorted = np.sort(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.empty_like(X, dtype=np.float32)
        for j in range(X.shape[1]):
            ref = self._sorted[:, j]
            # searchsorted gives [0, n] → divide by n to get [0,1]
            ranks = np.searchsorted(ref, X[:, j], side="right")
            out[:, j] = (ranks / max(len(ref), 1)).astype(np.float32)
        return out


def _impute_median_columns(Xtr: np.ndarray, Xother: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Median-impute numeric matrix; replace constant-after-impute columns with 0."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xtr_i = np.where(np.isnan(Xtr), med, Xtr)
    others_i = []
    for X in Xother:
        Xi = np.where(np.isnan(X), med, X)
        others_i.append(Xi)
    return Xtr_i.astype(np.float32), [x.astype(np.float32) for x in others_i], med


def _topk_by_anova(Xtr: np.ndarray, ytr: np.ndarray, k: int) -> np.ndarray:
    """Return indices of top-k columns by ANOVA-F statistic (fold-internal)."""
    from sklearn.feature_selection import SelectKBest, f_classif

    if k >= Xtr.shape[1]:
        return np.arange(Xtr.shape[1])
    sel = SelectKBest(f_classif, k=k).fit(Xtr, ytr)
    return np.where(sel.get_support())[0]


def run_tabpfn_cv(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    event: np.ndarray,
    time: np.ndarray,
    splits,
    feature_set_name: str,
    endpoint_name: str,
    preprocess: str = "raw",       # "raw" or "rank_quantile"
    select_topk: int | None = None,
    n_estimators: int = 4,
    device: str = "cpu",
    n_jobs: int = -1,
    log_every: int = 1,
) -> TabPFNRunResult:
    """Run TabPFN under repeated stratified CV; return OOF + averaged test."""
    import time as _time

    from tabpfn import TabPFNClassifier
    from .._cindex_imports import cindex  # local import to avoid circular at module import time

    n_train = len(X_train)
    n_test = len(X_test)
    Xtr_full = X_train.to_numpy(dtype=np.float32, copy=False)
    Xte_full = X_test.to_numpy(dtype=np.float32, copy=False)

    oof = np.full(n_train, np.nan)
    test_sum = np.zeros(n_test)
    test_n = 0
    fold_scores: list[float] = []
    t_total = 0.0

    for i, s in enumerate(splits):
        tr_idx = np.asarray(s.train_idx)
        va_idx = np.asarray(s.valid_idx)
        ytr = event[tr_idx].astype(int)
        if ytr.sum() < 3:
            _LOG.info("skip fold %d/%d: too few positives", i, len(splits))
            continue

        Xtr = Xtr_full[tr_idx]
        Xva = Xtr_full[va_idx]
        Xte = Xte_full

        Xtr_i, [Xva_i, Xte_i], _ = _impute_median_columns(Xtr, [Xva, Xte])

        if preprocess == "rank_quantile":
            tr = _RankQuantileTransform().fit(Xtr_i)
            Xtr_i = tr.transform(Xtr_i)
            Xva_i = tr.transform(Xva_i)
            Xte_i = tr.transform(Xte_i)

        if select_topk is not None:
            keep = _topk_by_anova(Xtr_i, ytr, select_topk)
            Xtr_i = Xtr_i[:, keep]
            Xva_i = Xva_i[:, keep]
            Xte_i = Xte_i[:, keep]

        t0 = _time.time()
        try:
            clf = TabPFNClassifier(
                device=device, n_estimators=n_estimators,
                ignore_pretraining_limits=True, random_state=0, n_jobs=n_jobs,
            )
            clf.fit(Xtr_i, ytr)
            va_proba = clf.predict_proba(Xva_i)[:, 1]
            te_proba = clf.predict_proba(Xte_i)[:, 1]
        except Exception as e:  # noqa: BLE001
            _LOG.warning("fold %d failed: %s", i, e)
            continue
        dt = _time.time() - t0
        t_total += dt

        oof[va_idx] = va_proba
        test_sum += te_proba
        test_n += 1
        sc = cindex(event[va_idx], time[va_idx], va_proba).cindex
        if np.isfinite(sc):
            fold_scores.append(sc)
        if i % log_every == 0:
            _LOG.info("fold %d/%d (%s/%s/%s feats=%d): %.2fs C=%.4f",
                      i, len(splits), feature_set_name, endpoint_name, preprocess,
                      Xtr_i.shape[1], dt, sc)

    test = test_sum / test_n if test_n else np.full(n_test, np.nan)
    return TabPFNRunResult(
        feature_set_name=feature_set_name,
        endpoint=endpoint_name,
        preprocess=preprocess,
        n_features=Xtr_i.shape[1] if test_n else int(Xtr_full.shape[1]),
        fold_scores=fold_scores,
        oof=oof,
        test=test,
        fit_seconds_total=float(t_total),
    )
