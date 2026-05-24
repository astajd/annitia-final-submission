"""Model registry.

Each model exposes ``fit(X, endpoint)`` and ``predict_risk(X) -> np.ndarray``.
Higher prediction values must mean higher risk so that the C-index can be
applied without sign flipping.
"""
from __future__ import annotations

from typing import Callable

from .cox import CoxNetWrapper, CoxPHWrapper
from .gbm_binary import CatBoostBinaryWrapper, LightGBMBinaryWrapper, XGBoostBinaryWrapper
from .rsf import RSFWrapper
from .xgb_survival import XGBoostAFTWrapper, XGBoostCoxWrapper

ModelBuilder = Callable[[dict], object]

_REGISTRY: dict[str, ModelBuilder] = {
    "coxnet": lambda p: CoxNetWrapper(**(p or {})),
    "coxph": lambda p: CoxPHWrapper(**(p or {})),
    "rsf": lambda p: RSFWrapper(**(p or {})),
    "xgb_cox": lambda p: XGBoostCoxWrapper(**(p or {})),
    "xgb_aft": lambda p: XGBoostAFTWrapper(**(p or {})),
    "lgbm_binary": lambda p: LightGBMBinaryWrapper(**(p or {})),
    "xgb_binary": lambda p: XGBoostBinaryWrapper(**(p or {})),
    "catboost_binary": lambda p: CatBoostBinaryWrapper(**(p or {})),
}


def build_model(name: str, params: dict | None = None):
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name](params or {})


def available_models() -> list[str]:
    return sorted(_REGISTRY)
