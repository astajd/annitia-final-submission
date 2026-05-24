"""Horizon-specific binary labels for hepatic and death endpoints.

For each (endpoint, horizon ``H``) we produce:

- ``label = 1`` if the event was observed within ``H`` years from baseline
- ``label = 0`` if the patient was observed event-free *beyond* H years
  (i.e. their administrative follow-up >= H or they had the event after H)
- ``mask = False`` for patients censored before H — usable only with
  inverse-censoring weighting, which we do not implement; the default mode
  excludes them so the binary classifier sees a clean label.

Per spec, we also expose a ``downweight`` mode that keeps censored-before-H
patients with a small sample weight; both are returned and the CV driver
picks "exclude" by default.

We never use the true event/censoring age beyond what the survival target
(``Endpoint.event``, ``Endpoint.time``) already exposed in Phase 1; horizon
construction is done from those two arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .targets import Endpoint


@dataclass
class HorizonLabels:
    endpoint_name: str
    horizon_years: float
    label: np.ndarray            # 0/1 per row, 0 where mask is False
    mask: np.ndarray             # bool, True = usable for classification
    weight: np.ndarray           # positive weights for the downweight mode
    n_total: int
    n_usable: int
    n_positive: int
    notes: dict


def build_horizon_labels(
    endpoint: Endpoint,
    horizon_years: float,
    *,
    censored_mode: str = "exclude",   # "exclude" | "downweight" | "soft_weight"
    censored_weight: float = 0.25,
    soft_weight_floor: float = 0.1,
    soft_weight_ceil: float = 1.0,
) -> HorizonLabels:
    """Construct binary labels for ``endpoint`` at horizon ``H`` years.

    The implementation follows the survival-data convention encoded in
    :class:`Endpoint`:

    - ``event[i]==True`` means the event was observed at ``time[i]`` years
      after baseline.
    - ``event[i]==False`` means the patient was *administratively censored*
      at ``time[i]`` years after baseline (no event observed yet).

    For horizon H:

    - ``label = 1`` iff ``event and time <= H``  (positive at horizon)
    - ``label = 0`` iff ``time > H``              (still event-free at H,
        regardless of whether the patient eventually had the event)
    - patient with ``not event and time < H``  -> censored before horizon;
      handled per ``censored_mode``:
        * ``exclude``     -> mask=False (default Phase 3.9 behaviour).
        * ``downweight``  -> mask=True, label=0, sample_weight=censored_weight.
        * ``soft_weight`` -> mask=True, label=0,
                              sample_weight = clip(time / H, floor, ceil).
    """
    if censored_mode not in {"exclude", "downweight", "soft_weight"}:
        raise ValueError(censored_mode)

    event = np.asarray(endpoint.event, dtype=bool)
    time = np.asarray(endpoint.time, dtype=float)
    base_mask = np.asarray(endpoint.mask, dtype=bool)

    label = np.zeros_like(event, dtype=int)
    label[event & (time <= horizon_years)] = 1
    label[(~event) & (time >= horizon_years)] = 0
    label[event & (time > horizon_years)] = 0  # event after horizon: still event-free at H

    censored_before = (~event) & (time < horizon_years)

    weight = np.ones_like(event, dtype=float)
    if censored_mode == "exclude":
        keep = base_mask & ~censored_before
    elif censored_mode == "downweight":
        keep = base_mask
        weight = np.where(censored_before, censored_weight, 1.0)
    else:  # soft_weight
        keep = base_mask
        soft_w = np.clip(time / max(horizon_years, 1e-9), soft_weight_floor, soft_weight_ceil)
        weight = np.where(censored_before, soft_w, 1.0)

    n_total = int(len(event))
    n_usable = int(keep.sum())
    n_positive = int(((label == 1) & keep).sum())

    notes = {
        "horizon_years": float(horizon_years),
        "censored_mode": censored_mode,
        "censored_weight": float(censored_weight),
        "soft_weight_floor": float(soft_weight_floor),
        "soft_weight_ceil": float(soft_weight_ceil),
        "n_censored_before_horizon": int(censored_before.sum()),
        "n_event_before_horizon": int(((label == 1) & keep).sum()),
        "n_eventfree_at_horizon": int(((label == 0) & keep).sum()),
        "endpoint_name": endpoint.name,
    }

    return HorizonLabels(
        endpoint_name=endpoint.name,
        horizon_years=float(horizon_years),
        label=label,
        mask=keep,
        weight=weight,
        n_total=n_total,
        n_usable=n_usable,
        n_positive=n_positive,
        notes=notes,
    )
