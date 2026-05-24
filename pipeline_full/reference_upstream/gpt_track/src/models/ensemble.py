"""Rank-average ensembling helpers.

The C-index is purely rank-based, so we standardize each model's predictions
to percentile ranks before averaging. This makes outputs from CoxNet
(log-hazard, unbounded) and binary classifiers (probabilities) comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_rank(x: np.ndarray) -> np.ndarray:
    """Return percentile ranks in [0, 1]; ties get average ranks."""
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rank(method="average", pct=True, na_option="keep").to_numpy()


def rank_average(predictions: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
    """Average percentile ranks across models, optionally with weights.

    Position-wise NaN handling: where a model's prediction is NaN, that model
    is omitted from the average for that position; the remaining weights are
    re-normalized. If every model is NaN at a position, the result is NaN.
    """
    if not predictions:
        raise ValueError("predictions is empty")
    keys = list(predictions)
    if weights is None:
        w = np.ones(len(keys))
    else:
        w = np.array([float(weights.get(k, 1.0)) for k in keys])

    ranks = np.vstack([to_rank(predictions[k]) for k in keys])  # (n_models, n_obs)
    valid = ~np.isnan(ranks)
    weighted = np.where(valid, ranks * w[:, None], 0.0)
    weight_sum = (valid * w[:, None]).sum(axis=0)
    out = np.where(weight_sum > 0, weighted.sum(axis=0) / np.where(weight_sum > 0, weight_sum, 1.0), np.nan)
    return out
