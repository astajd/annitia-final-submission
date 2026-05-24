"""Binary-classifier wrappers used as event-occurrence ranking models.

The C-index only evaluates ranking, so a probability of "ever experiences the
event" is a perfectly reasonable risk score. We handle class imbalance with
``scale_pos_weight`` / class weights but keep the parameters conservative to
avoid overfitting on the rare positives (47 hepatic events, ~76 deaths).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..targets import Endpoint
from ._preprocess import fill_for_tree


class _BaseBinary:
    name = "binary_base"

    def __init__(self, **params):
        self.params = params
        self._model = None
        self._cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, endpoint: Endpoint, mask=None) -> "_BaseBinary":
        if mask is not None:
            X = X.loc[mask]
            event = endpoint.event[mask.values if hasattr(mask, "values") else mask]
        else:
            event = endpoint.event
        Xn = fill_for_tree(X)
        self._cols = list(Xn.columns)
        y = np.asarray(event).astype(int)
        self._fit_inner(Xn.values, y)
        return self

    def _fit_inner(self, Xn: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        Xn = fill_for_tree(X).reindex(columns=self._cols, fill_value=0)
        proba = self._model.predict_proba(Xn.values)[:, 1]
        return proba


class LightGBMBinaryWrapper(_BaseBinary):
    name = "lgbm_binary"

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        random_state: int = 0,
    ):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
        )

    def _fit_inner(self, Xn: np.ndarray, y: np.ndarray) -> None:
        import lightgbm as lgb

        n_pos = max(int(y.sum()), 1)
        n_neg = max(int((1 - y).sum()), 1)
        spw = n_neg / n_pos
        self._model = lgb.LGBMClassifier(
            **self.params,
            class_weight=None,
            scale_pos_weight=spw,
            verbose=-1,
        )
        self._model.fit(Xn, y)


class XGBoostBinaryWrapper(_BaseBinary):
    name = "xgb_binary"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        random_state: int = 0,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=random_state,
        )

    def _fit_inner(self, Xn: np.ndarray, y: np.ndarray) -> None:
        import xgboost as xgb

        n_pos = max(int(y.sum()), 1)
        n_neg = max(int((1 - y).sum()), 1)
        self._model = xgb.XGBClassifier(
            **self.params,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=n_neg / n_pos,
            tree_method="hist",
            verbosity=0,
        )
        self._model.fit(Xn, y)


class CatBoostBinaryWrapper(_BaseBinary):
    name = "catboost_binary"

    def __init__(
        self,
        iterations: int = 500,
        depth: int = 5,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        random_state: int = 0,
    ):
        super().__init__(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            random_state=random_state,
        )

    def _fit_inner(self, Xn: np.ndarray, y: np.ndarray) -> None:
        from catboost import CatBoostClassifier

        n_pos = max(int(y.sum()), 1)
        n_neg = max(int((1 - y).sum()), 1)
        self._model = CatBoostClassifier(
            **self.params,
            verbose=0,
            allow_writing_files=False,
            class_weights=[1.0, n_neg / n_pos],
        )
        self._model.fit(Xn, y)
