"""Cox-family survival models (CoxNet + CoxPH) via scikit-survival.

Both wrappers share the same preprocessing (median-impute + standardize) and
emit a higher-is-riskier prediction so the C-index needs no sign flip.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..targets import Endpoint, to_sksurv
from ..utils import get_logger
from ._preprocess import fit_preprocessor

_LOG = get_logger(__name__)


class _BaseCox:
    name = "cox_base"

    def __init__(self, **params):
        self.params = params
        self._pre = None
        self._model = None

    def fit(self, X: pd.DataFrame, endpoint: Endpoint, mask: np.ndarray | None = None) -> "_BaseCox":
        if mask is not None:
            X = X.loc[mask]
            event = endpoint.event[mask.values if hasattr(mask, "values") else mask]
            time = endpoint.time[mask.values if hasattr(mask, "values") else mask]
        else:
            event, time = endpoint.event, endpoint.time
        self._pre = fit_preprocessor(X, scale_numeric=True)
        Xn = self._pre.transform(X)
        y = to_sksurv(np.asarray(event), np.asarray(time)) if False else _y(event, time)
        self._fit_inner(Xn, y)
        return self

    def _fit_inner(self, Xn: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        Xn = self._pre.transform(X)
        # sksurv predict() returns the log partial hazard; higher = higher risk.
        return self._model.predict(Xn)


def _y(event, time) -> np.ndarray:
    arr = np.empty(len(event), dtype=[("event", "?"), ("time", "<f8")])
    arr["event"] = np.asarray(event).astype(bool)
    arr["time"] = np.asarray(time).astype(float)
    return arr


class CoxNetWrapper(_BaseCox):
    name = "coxnet"

    def __init__(
        self,
        l1_ratio: float = 0.9,
        alpha_min_ratio: float | str = 0.01,
        n_alphas: int = 100,
        max_iter: int = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.l1_ratio = l1_ratio
        self.alpha_min_ratio = alpha_min_ratio
        self.n_alphas = n_alphas
        self.max_iter = max_iter

    def _fit_inner(self, Xn: np.ndarray, y: np.ndarray) -> None:
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        self._model = CoxnetSurvivalAnalysis(
            l1_ratio=self.l1_ratio,
            alpha_min_ratio=self.alpha_min_ratio,
            n_alphas=self.n_alphas,
            max_iter=self.max_iter,
        )
        try:
            self._model.fit(Xn, y)
        except Exception as e:
            _LOG.warning("CoxNet fit failed (%s); falling back to CoxPH", e)
            from sksurv.linear_model import CoxPHSurvivalAnalysis

            self._model = CoxPHSurvivalAnalysis(alpha=1.0)
            self._model.fit(Xn, y)


class CoxPHWrapper(_BaseCox):
    name = "coxph"

    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def _fit_inner(self, Xn: np.ndarray, y: np.ndarray) -> None:
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        self._model = CoxPHSurvivalAnalysis(alpha=self.alpha)
        self._model.fit(Xn, y)
