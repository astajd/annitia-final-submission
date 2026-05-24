"""Concordance index utilities used everywhere in the pipeline.

Higher predicted risk should correspond to shorter survival, so the C-index is
applied to the raw predicted risk score (not its negative). Matches scikit-
survival's ``concordance_index_censored`` convention.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class CIndexResult:
    cindex: float
    n_pairs: int
    n_concordant: int
    n_tied_risk: int
    n_tied_time: int

    def as_dict(self) -> dict:
        return {
            "cindex": float(self.cindex),
            "n_pairs": int(self.n_pairs),
            "n_concordant": int(self.n_concordant),
            "n_tied_risk": int(self.n_tied_risk),
            "n_tied_time": int(self.n_tied_time),
        }


def cindex(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> CIndexResult:
    """Concordance index where higher ``risk`` means shorter survival.

    Falls back gracefully when sksurv is unavailable (returns NaN with no pairs)
    and when the labels contain no events (NaN).
    """
    event = np.asarray(event).astype(bool)
    time = np.asarray(time).astype(float)
    risk = np.asarray(risk).astype(float)

    if event.sum() == 0:
        return CIndexResult(np.nan, 0, 0, 0, 0)

    try:
        from sksurv.metrics import concordance_index_censored
    except Exception:
        return CIndexResult(np.nan, 0, 0, 0, 0)

    c, n_concordant, n_discordant, n_tied_risk, n_tied_time = concordance_index_censored(
        event, time, risk
    )
    n_pairs = n_concordant + n_discordant + n_tied_risk
    return CIndexResult(float(c), n_pairs, n_concordant, n_tied_risk, n_tied_time)


def weighted_score(c_hepatic: float, c_death: float) -> float:
    """Competition score: 0.7*hepatic + 0.3*death."""
    return config.WEIGHT_HEPATIC * c_hepatic + config.WEIGHT_DEATH * c_death
