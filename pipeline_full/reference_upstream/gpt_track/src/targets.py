"""Construct survival endpoints from the raw train CSV.

Produces sklearn-survival-compatible structured arrays (``event``, ``time``)
for each endpoint, with explicit handling of:
- patients without an event (administrative censoring at last observed age)
- patients with missing ``death`` status (two configurable modes)
- non-positive survival times (logged + epsilon-clamped, never silently dropped)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .data_loading import last_observed_age
from .utils import get_logger

_LOG = get_logger(__name__)

DeathMode = str  # "drop_missing_death" | "censor_missing_death_at_last_visit"


@dataclass
class Endpoint:
    """Survival endpoint for a single risk.

    ``event`` is a boolean array (True iff the event was observed).
    ``time`` is a float array of years from baseline.
    ``mask`` is a boolean array on the original DataFrame index marking which
    rows are usable for fitting (False = dropped, e.g. unknown death status
    in ``drop_missing_death`` mode).
    """

    name: str
    event: np.ndarray
    time: np.ndarray
    mask: np.ndarray
    notes: dict


def _sksurv_y(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Pack event/time into a structured array for scikit-survival."""
    y = np.empty(
        len(event),
        dtype=[("event", "?"), ("time", "<f8")],
    )
    y["event"] = event.astype(bool)
    y["time"] = time.astype(float)
    return y


def to_sksurv(endpoint: Endpoint) -> np.ndarray:
    """Public helper to turn an :class:`Endpoint` into the sksurv y-array."""
    return _sksurv_y(endpoint.event, endpoint.time)


def _clamp_times(time: np.ndarray, label: str) -> tuple[np.ndarray, dict]:
    """Replace non-positive times with TIME_EPSILON, return diagnostics."""
    bad = ~(time > 0)
    if bad.any():
        _LOG.warning("%s: %d rows have non-positive survival time, clamping to %g",
                     label, int(bad.sum()), config.TIME_EPSILON)
    out = np.where(bad, config.TIME_EPSILON, time)
    notes = {
        "non_positive_times": int(bad.sum()),
        "clamped_to": config.TIME_EPSILON,
    }
    return out, notes


def build_hepatic_endpoint(
    train_df: pd.DataFrame, age_cols: list[str]
) -> Endpoint:
    """Hepatic-event endpoint.

    For event patients we use ``evenements_hepatiques_age_occur - Age_v1``.
    For censored patients we use ``last_observed_age - Age_v1``.
    """
    age_v1 = train_df["Age_v1"].astype(float).to_numpy()
    last_age = last_observed_age(train_df, age_cols).astype(float).to_numpy()
    raw_event = train_df[config.HEPATIC_EVENT_COL]
    event = raw_event.fillna(0).astype(int).to_numpy().astype(bool)

    event_age = train_df[config.HEPATIC_EVENT_AGE_COL].astype(float).to_numpy()
    time = np.where(event, event_age - age_v1, last_age - age_v1)

    # Hepatic outcome is well-defined for everyone (treat NaN as no-event).
    n_event = int(event.sum())
    mask = np.isfinite(time)
    n_dropped = int((~mask).sum())
    if n_dropped:
        _LOG.warning("hepatic: dropping %d rows with non-finite time", n_dropped)

    time = np.where(mask, time, config.TIME_EPSILON)
    time, time_notes = _clamp_times(time, "hepatic")

    notes = {
        "n_total": int(len(event)),
        "n_event": n_event,
        "n_dropped_nonfinite": n_dropped,
        **time_notes,
    }
    _LOG.info("hepatic endpoint: %s", notes)
    return Endpoint("hepatic", event, time, mask, notes)


def build_death_endpoint(
    train_df: pd.DataFrame,
    age_cols: list[str],
    mode: DeathMode = "censor_missing_death_at_last_visit",
) -> Endpoint:
    """Death endpoint with two NaN-handling modes.

    ``drop_missing_death``           -> mask=False on rows where ``death`` is NaN.
    ``censor_missing_death_at_last_visit`` -> treat NaN as censored at last visit.
    """
    if mode not in {"drop_missing_death", "censor_missing_death_at_last_visit"}:
        raise ValueError(f"unknown death_target_mode={mode}")

    age_v1 = train_df["Age_v1"].astype(float).to_numpy()
    last_age = last_observed_age(train_df, age_cols).astype(float).to_numpy()
    raw_death = train_df[config.DEATH_EVENT_COL]
    raw_death_age = train_df[config.DEATH_EVENT_AGE_COL].astype(float).to_numpy()

    is_missing = raw_death.isna().to_numpy()
    event = raw_death.fillna(0).astype(int).to_numpy().astype(bool)

    time = np.where(event, raw_death_age - age_v1, last_age - age_v1)

    if mode == "drop_missing_death":
        mask = ~is_missing & np.isfinite(time)
        n_dropped = int((~mask).sum())
        _LOG.info("death/drop_missing_death: dropping %d rows", n_dropped)
    else:
        # Treat NaN as not-dead, censored at last_observed_age - Age_v1.
        event = np.where(is_missing, False, event)
        mask = np.isfinite(time)
        n_dropped = int((~mask).sum())
        _LOG.info("death/censor_missing_death_at_last_visit: %d NaN deaths censored at last visit",
                  int(is_missing.sum()))

    time = np.where(np.isfinite(time), time, config.TIME_EPSILON)
    time, time_notes = _clamp_times(time, f"death[{mode}]")

    notes = {
        "mode": mode,
        "n_total": int(len(event)),
        "n_event": int(event.sum()),
        "n_missing_death": int(is_missing.sum()),
        "n_dropped_nonfinite": n_dropped,
        **time_notes,
    }
    _LOG.info("death endpoint: %s", notes)
    return Endpoint("death", event, time, mask, notes)


def followup_years(train_df: pd.DataFrame, age_cols: list[str]) -> pd.Series:
    """Years from baseline visit to last observed visit (administrative)."""
    return last_observed_age(train_df, age_cols) - train_df["Age_v1"]
