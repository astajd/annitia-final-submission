"""Longitudinal summaries across all visits per biomarker.

This module supports two modes:
- "all_visits": use every available visit (high leakage risk on train because
  for hepatic-event patients the visit set may extend past the event).
- "strict_time_aligned": for training, mask out visits whose Age exceeds the
  event time (or censoring time). For test data we have no event time, so we
  use all visits.

The strict mode requires per-patient cutoff ages, which the caller supplies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger

_LOG = get_logger(__name__)


def _summarize_one_stem(
    values: pd.DataFrame, ages: pd.DataFrame, age_v1: pd.Series, stem: str
) -> pd.DataFrame:
    """Compute longitudinal summary stats for a single biomarker.

    ``values`` and ``ages`` are aligned (rows: patients, cols: visits).
    Cells in ``values`` may be NaN (visit didn't measure this biomarker)
    and cells in ``ages`` may be NaN (visit didn't happen at all).
    """
    v = values.to_numpy(dtype=float)
    a = ages.to_numpy(dtype=float)

    valid = ~np.isnan(v)
    n = valid.sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(v, axis=1)
        std = np.nanstd(v, axis=1, ddof=0)
        vmin = np.nanmin(v, axis=1)
        vmax = np.nanmax(v, axis=1)
        median = np.nanmedian(v, axis=1)

    # First / last (in visit order, ignoring NaN cells).
    first = np.full(v.shape[0], np.nan)
    last = np.full(v.shape[0], np.nan)
    age_first = np.full(v.shape[0], np.nan)
    age_last = np.full(v.shape[0], np.nan)
    for i in range(v.shape[0]):
        idx = np.where(valid[i])[0]
        if len(idx):
            first[i] = v[i, idx[0]]
            last[i] = v[i, idx[-1]]
            age_first[i] = a[i, idx[0]]
            age_last[i] = a[i, idx[-1]]

    delta = last - first
    rel_delta = np.where(np.abs(first) > 1e-9, delta / first, np.nan)

    span = age_last - age_first
    slope = np.where(span > 1e-6, delta / span, np.nan)

    age_v1_arr = age_v1.to_numpy(dtype=float)
    time_since_last = age_last - age_v1_arr  # years from baseline to last measurement

    out = pd.DataFrame(
        {
            f"{stem}_first": first,
            f"{stem}_last": last,
            f"{stem}_mean": mean,
            f"{stem}_median": median,
            f"{stem}_std": std,
            f"{stem}_min": vmin,
            f"{stem}_max": vmax,
            f"{stem}_delta": delta,
            f"{stem}_rel_delta": rel_delta,
            f"{stem}_slope": slope,
            f"{stem}_count": n,
            f"{stem}_age_first": age_first,
            f"{stem}_age_last": age_last,
            f"{stem}_time_since_baseline": time_since_last,
        },
        index=values.index,
    )
    return out


def all_visits_longitudinal(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
    skip_stems: tuple[str, ...] = ("Age",),
) -> pd.DataFrame:
    """Per-biomarker longitudinal summary using *every* available visit.

    This is the high-leakage feature set: hepatic-event patients may have
    visits *after* the event, so the summary leaks the event onto the patient
    even at training time.
    """
    age_v1 = df["Age_v1"]
    ages = df[age_visit_cols]
    pieces: list[pd.DataFrame] = []
    for stem, cols in visit_columns.items():
        if stem in skip_stems:
            continue
        # Align ages with this stem's visits by visit index.
        same_visits_age = _ages_for_visits(df, cols, age_visit_cols)
        block = _summarize_one_stem(df[cols], same_visits_age, age_v1, stem)
        pieces.append(block)
    out = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)
    _LOG.info("all_visits_longitudinal: %d rows x %d cols", len(out), out.shape[1])
    return out


def _ages_for_visits(
    df: pd.DataFrame, biomarker_cols: list[str], age_visit_cols: list[str]
) -> pd.DataFrame:
    """Return a DataFrame of Age_vK aligned column-by-column with ``biomarker_cols``.

    Uses the visit suffix to look up the matching Age column. If the matching
    Age column doesn't exist (rare), uses NaN.
    """
    from ..utils import visit_index

    age_lookup = {visit_index(c): c for c in age_visit_cols if visit_index(c) is not None}
    aligned = pd.DataFrame(index=df.index)
    for c in biomarker_cols:
        idx = visit_index(c)
        age_col = age_lookup.get(idx)
        aligned[c] = df[age_col] if age_col in df.columns else np.nan
    return aligned


def strict_time_aligned_longitudinal(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
    cutoff_age: pd.Series | np.ndarray | None,
    skip_stems: tuple[str, ...] = ("Age",),
) -> pd.DataFrame:
    """Like :func:`all_visits_longitudinal` but masks out post-cutoff visits.

    For training data, ``cutoff_age`` is the event time for event patients
    (hepatic_age_occur or death_age_occur) and the last observed age for
    censored patients. Any biomarker measurement at an Age > cutoff is set to
    NaN before summarization.

    For test data, pass ``cutoff_age=None`` to get the same behavior as
    :func:`all_visits_longitudinal`. The asymmetry between train and test is
    documented in the leakage report.
    """
    age_v1 = df["Age_v1"]
    pieces: list[pd.DataFrame] = []
    for stem, cols in visit_columns.items():
        if stem in skip_stems:
            continue
        ages = _ages_for_visits(df, cols, age_visit_cols)
        if cutoff_age is not None:
            cutoff = pd.Series(cutoff_age, index=df.index).astype(float)
            mask = ages.le(cutoff, axis=0)
            values = df[cols].where(mask)
        else:
            values = df[cols]
        block = _summarize_one_stem(values, ages, age_v1, stem)
        pieces.append(block)
    out = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=df.index)
    _LOG.info("strict_time_aligned_longitudinal: %d rows x %d cols", len(out), out.shape[1])
    return out
