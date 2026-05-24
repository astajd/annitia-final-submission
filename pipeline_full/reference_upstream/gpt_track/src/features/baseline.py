"""Baseline (visit-1 only) feature set.

The simplest, lowest-leakage view of each patient: static demographics plus
every biomarker measured at the first visit. Models trained on this should
generalize well because they cannot "peek" at follow-up.
"""
from __future__ import annotations

import pandas as pd

from .. import config
from ..utils import get_logger

_LOG = get_logger(__name__)

# Static (non-visit-suffixed) columns we want to keep as features.
_STATIC_KEEP = (
    "gender",
    "T2DM",
    "Hypertension",
    "Dyslipidaemia",
    "bariatric_surgery",
    "bariatric_surgery_age",
)


def baseline_v1_features(
    df: pd.DataFrame, visit_columns: dict[str, list[str]]
) -> pd.DataFrame:
    """Return one row per patient with v1-only features.

    Drops the patient ID and any endpoint columns. Adds ``Age_v1`` and the
    first-visit measurement of every biomarker.
    """
    cols: list[str] = []
    for c in _STATIC_KEEP:
        if c in df.columns:
            cols.append(c)

    for stem, vcols in visit_columns.items():
        if not vcols:
            continue
        # vcols is sorted by visit index; first entry is _v1 (or the lowest).
        cols.append(vcols[0])

    out = df[cols].copy()
    _LOG.info("baseline_v1_features: %d rows x %d cols", len(out), out.shape[1])
    return out


def early_v1_v3_features(
    df: pd.DataFrame, visit_columns: dict[str, list[str]]
) -> pd.DataFrame:
    """v1+v2+v3 measurements with simple deltas and slopes.

    For each biomarker stem we keep raw v1/v2/v3 plus:
    - delta_v3_v1 = v3 - v1
    - mean_v1_v3
    - slope_v1_v3 = (v3 - v1) / (Age_v3 - Age_v1) when ages are available
    """
    static = [c for c in _STATIC_KEEP if c in df.columns]
    pieces: list[pd.DataFrame] = [df[static].copy()] if static else []

    age_v1 = df.get("Age_v1")
    age_v3 = df.get("Age_v3")

    for stem, vcols in visit_columns.items():
        v1 = vcols[0] if len(vcols) >= 1 else None
        v2 = vcols[1] if len(vcols) >= 2 else None
        v3 = vcols[2] if len(vcols) >= 3 else None

        block: dict[str, pd.Series] = {}
        if v1 is not None:
            block[v1] = df[v1]
        if v2 is not None:
            block[v2] = df[v2]
        if v3 is not None:
            block[v3] = df[v3]

        if v1 is not None and v3 is not None and stem != "Age":
            block[f"{stem}_delta_v1_v3"] = df[v3] - df[v1]
            block[f"{stem}_mean_v1_v3"] = df[[v1, v2, v3]].mean(axis=1) if v2 is not None else df[[v1, v3]].mean(axis=1)
            if age_v1 is not None and age_v3 is not None:
                dt = (age_v3 - age_v1).replace(0, pd.NA)
                block[f"{stem}_slope_v1_v3"] = (df[v3] - df[v1]) / dt

        if block:
            pieces.append(pd.DataFrame(block, index=df.index))

    out = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)
    _LOG.info("early_v1_v3_features: %d rows x %d cols", len(out), out.shape[1])
    return out
