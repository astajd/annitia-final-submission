"""Visit-cadence and missingness features.

Covers the H "missingness_and_visit_cadence" feature set: number of visits,
average / max gap between visits, recency, missingness rate per visit-group.
These are heavily correlated with administrative follow-up time and are
flagged as high-leakage when used alone or alongside endpoint targets that
were derived from the same follow-up window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger, visit_index

_LOG = get_logger(__name__)


def visit_cadence(df: pd.DataFrame, age_visit_cols: list[str]) -> pd.DataFrame:
    """Per-patient cadence features: visit count, gaps, recency."""
    a = df[age_visit_cols].to_numpy(dtype=float)
    n_visits = (~np.isnan(a)).sum(axis=1)
    age_first = np.nanmin(a, axis=1) if a.size else np.full(len(df), np.nan)
    age_last = np.nanmax(a, axis=1) if a.size else np.full(len(df), np.nan)
    span = age_last - age_first

    gaps_mean = np.full(len(df), np.nan)
    gaps_max = np.full(len(df), np.nan)
    gaps_min = np.full(len(df), np.nan)
    for i in range(a.shape[0]):
        row = a[i][~np.isnan(a[i])]
        row = np.sort(row)
        if len(row) >= 2:
            diffs = np.diff(row)
            gaps_mean[i] = float(diffs.mean())
            gaps_max[i] = float(diffs.max())
            gaps_min[i] = float(diffs.min())

    out = pd.DataFrame(
        {
            "n_visits": n_visits.astype(int),
            "age_first_visit": age_first,
            "age_last_visit": age_last,
            "followup_span": span,
            "gap_mean": gaps_mean,
            "gap_max": gaps_max,
            "gap_min": gaps_min,
        },
        index=df.index,
    )
    return out


def missingness_by_group(
    df: pd.DataFrame, visit_columns: dict[str, list[str]]
) -> pd.DataFrame:
    """Fraction of missing values per biomarker stem and per visit number."""
    out = pd.DataFrame(index=df.index)
    out["miss_total"] = df[[c for cols in visit_columns.values() for c in cols]].isna().mean(axis=1)
    for stem, cols in visit_columns.items():
        out[f"miss_{stem}"] = df[cols].isna().mean(axis=1)

    # Missingness collapsed by visit index across all biomarkers.
    by_visit: dict[int, list[str]] = {}
    for stem, cols in visit_columns.items():
        for c in cols:
            k = visit_index(c)
            if k is None:
                continue
            by_visit.setdefault(k, []).append(c)
    for k in sorted(by_visit):
        out[f"miss_visit_{k}"] = df[by_visit[k]].isna().mean(axis=1)
    return out
