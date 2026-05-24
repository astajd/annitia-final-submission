"""Random Survival Forest wrapper (scikit-survival)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..targets import Endpoint
from ._preprocess import fit_preprocessor


def _y(event, time) -> np.ndarray:
    arr = np.empty(len(event), dtype=[("event", "?"), ("time", "<f8")])
    arr["event"] = np.asarray(event).astype(bool)
    arr["time"] = np.asarray(time).astype(float)
    return arr


class RSFWrapper:
    name = "rsf"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 6,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str | float = "sqrt",
        random_state: int = 0,
        n_jobs: int = -1,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._pre = None
        self._model = None

    def fit(self, X: pd.DataFrame, endpoint: Endpoint, mask=None) -> "RSFWrapper":
        if mask is not None:
            X = X.loc[mask]
            event = endpoint.event[mask.values if hasattr(mask, "values") else mask]
            time = endpoint.time[mask.values if hasattr(mask, "values") else mask]
        else:
            event, time = endpoint.event, endpoint.time
        self._pre = fit_preprocessor(X, scale_numeric=False)
        Xn = self._pre.transform(X)
        from sksurv.ensemble import RandomSurvivalForest

        self._model = RandomSurvivalForest(**self.params)
        self._model.fit(Xn, _y(event, time))
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        Xn = self._pre.transform(X)
        return self._model.predict(Xn)
