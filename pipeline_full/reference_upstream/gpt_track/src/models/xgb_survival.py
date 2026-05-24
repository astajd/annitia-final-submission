"""XGBoost Cox-survival wrapper.

Uses ``objective='survival:cox'`` with the XGBoost convention that survival
times are passed as a signed label: positive for events, negative for censored
observations. The raw margin is the predicted log hazard, so higher = higher
risk (no sign flip needed for the C-index).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..targets import Endpoint
from ._preprocess import fill_for_tree


class XGBoostCoxWrapper:
    name = "xgb_cox"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        min_child_weight: float = 1.0,
        random_state: int = 0,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=random_state,
            objective="survival:cox",
            tree_method="hist",
            verbosity=0,
        )
        self._model = None
        self._cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, endpoint: Endpoint, mask=None) -> "XGBoostCoxWrapper":
        if mask is not None:
            X = X.loc[mask]
            event = endpoint.event[mask.values if hasattr(mask, "values") else mask]
            time = endpoint.time[mask.values if hasattr(mask, "values") else mask]
        else:
            event, time = endpoint.event, endpoint.time

        Xn = fill_for_tree(X)
        self._cols = list(Xn.columns)

        # XGBoost Cox label: positive for event, negative for censored (absolute value = time).
        label = np.where(event, np.asarray(time), -np.asarray(time))

        import xgboost as xgb

        self._model = xgb.XGBRegressor(**self.params)
        self._model.fit(Xn.values, label)
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        Xn = fill_for_tree(X).reindex(columns=self._cols, fill_value=0)
        return self._model.predict(Xn.values)


class XGBoostAFTWrapper:
    """XGBoost survival:aft (Accelerated Failure Time).

    AFT predicts log(time-to-event); short predicted time means high risk, so
    we return the negative prediction as the risk score.
    """

    name = "xgb_aft"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        aft_loss_distribution: str = "normal",
        aft_loss_distribution_scale: float = 1.0,
        random_state: int = 0,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            objective="survival:aft",
            eval_metric="aft-nloglik",
            aft_loss_distribution=aft_loss_distribution,
            aft_loss_distribution_scale=aft_loss_distribution_scale,
            tree_method="hist",
            verbosity=0,
        )
        self._booster = None
        self._cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, endpoint, mask=None) -> "XGBoostAFTWrapper":
        if mask is not None:
            X = X.loc[mask]
            event = endpoint.event[mask.values if hasattr(mask, "values") else mask]
            time = endpoint.time[mask.values if hasattr(mask, "values") else mask]
        else:
            event, time = endpoint.event, endpoint.time

        Xn = fill_for_tree(X)
        self._cols = list(Xn.columns)

        # AFT requires interval labels: lower = upper = time for events,
        # lower = time, upper = +inf for censored.
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=bool)
        lower = time.copy()
        upper = np.where(event, time, np.inf)

        import xgboost as xgb

        dtrain = xgb.DMatrix(Xn.values)
        dtrain.set_float_info("label_lower_bound", lower)
        dtrain.set_float_info("label_upper_bound", upper)

        params = {
            "objective": self.params["objective"],
            "eval_metric": self.params["eval_metric"],
            "aft_loss_distribution": self.params["aft_loss_distribution"],
            "aft_loss_distribution_scale": self.params["aft_loss_distribution_scale"],
            "max_depth": self.params["max_depth"],
            "eta": self.params["learning_rate"],
            "subsample": self.params["subsample"],
            "colsample_bytree": self.params["colsample_bytree"],
            "lambda": self.params["reg_lambda"],
            "tree_method": self.params["tree_method"],
            "verbosity": self.params["verbosity"],
        }
        self._booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.params["n_estimators"],
        )
        return self

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        Xn = fill_for_tree(X).reindex(columns=self._cols, fill_value=0)
        d = xgb.DMatrix(Xn.values)
        # AFT returns predicted time-to-event; invert sign so that higher = riskier.
        return -np.asarray(self._booster.predict(d), dtype=float)
