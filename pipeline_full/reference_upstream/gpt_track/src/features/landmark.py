"""Landmark (calendar-window) feature sets.

Each set freezes a fixed time window after Age_v1 and only uses measurements
inside that window. Crucially, the cutoff is **target-independent** — it does
not depend on whether/when the patient experienced an event. That eliminates
the Phase 1 strict_time_aligned leakage where the cutoff age itself encoded
the event time.

For each biomarker we compute first/last/mean/min/max/std/delta/slope inside
the window. We also keep `Age_v1` and the static demographics as baseline
context, but explicitly drop visit counts and last-observed-age inside-window
since those would correlate with the patient's future visit cadence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger, visit_index

_LOG = get_logger(__name__)

_STATIC_KEEP = (
    "gender",
    "T2DM",
    "Hypertension",
    "Dyslipidaemia",
    "bariatric_surgery",
    "bariatric_surgery_age",
)


def _ages_for_visits(
    df: pd.DataFrame, biomarker_cols: list[str], age_visit_cols: list[str]
) -> pd.DataFrame:
    age_lookup = {visit_index(c): c for c in age_visit_cols if visit_index(c) is not None}
    aligned = pd.DataFrame(index=df.index)
    for c in biomarker_cols:
        idx = visit_index(c)
        age_col = age_lookup.get(idx)
        aligned[c] = df[age_col] if age_col in df.columns else np.nan
    return aligned


def _summarize_window(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
    *,
    window_years: float | None = None,
    max_visit_index: int | None = None,
    skip_stems: tuple[str, ...] = ("Age",),
) -> pd.DataFrame:
    """Summarize biomarker trajectories inside a fixed window.

    Either ``window_years`` (calendar window from Age_v1) or
    ``max_visit_index`` (cap at visit number K) must be given. The cutoff is
    independent of the survival target.

    Returned columns: per stem first/last/mean/min/max/std/delta/slope. We
    deliberately omit ``count``, ``age_first``, ``age_last``, and any visit-
    cadence summary inside the window, since those leak total follow-up.
    """
    if (window_years is None) == (max_visit_index is None):
        raise ValueError("provide exactly one of window_years or max_visit_index")

    age_v1 = df["Age_v1"].astype(float)
    pieces: list[pd.DataFrame] = []

    static = [c for c in _STATIC_KEEP if c in df.columns]
    if static:
        pieces.append(df[static].copy())
    pieces.append(df[["Age_v1"]].copy())

    for stem, cols in visit_columns.items():
        if stem in skip_stems or not cols:
            continue
        ages = _ages_for_visits(df, cols, age_visit_cols)
        values = df[cols]

        if window_years is not None:
            cutoff = age_v1 + float(window_years)
            mask = ages.le(cutoff, axis=0)
        else:
            mask = pd.DataFrame(False, index=df.index, columns=cols)
            for c in cols:
                k = visit_index(c)
                if k is not None and k <= max_visit_index:
                    mask[c] = True

        v = values.where(mask).to_numpy(dtype=float)
        a = ages.where(mask).to_numpy(dtype=float)

        valid = ~np.isnan(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(v, axis=1)
            std = np.nanstd(v, axis=1, ddof=0)
            vmin = np.nanmin(v, axis=1)
            vmax = np.nanmax(v, axis=1)
            median = np.nanmedian(v, axis=1)

        first = np.full(v.shape[0], np.nan)
        last = np.full(v.shape[0], np.nan)
        a_first = np.full(v.shape[0], np.nan)
        a_last = np.full(v.shape[0], np.nan)
        for i in range(v.shape[0]):
            idx = np.where(valid[i])[0]
            if len(idx):
                first[i] = v[i, idx[0]]
                last[i] = v[i, idx[-1]]
                a_first[i] = a[i, idx[0]]
                a_last[i] = a[i, idx[-1]]
        delta = last - first
        rel_delta = np.where(np.abs(first) > 1e-9, delta / first, np.nan)
        span = a_last - a_first
        slope = np.where(span > 1e-6, delta / span, np.nan)

        block = pd.DataFrame(
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
            },
            index=df.index,
        )
        pieces.append(block)

    out = pd.concat(pieces, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    _LOG.info(
        "landmark window_years=%s max_visit_index=%s -> %s",
        window_years, max_visit_index, out.shape,
    )
    return out


def first_window(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
    years: float,
) -> pd.DataFrame:
    return _summarize_window(df, visit_columns, age_visit_cols, window_years=years)


def first_n_visits(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
    n_visits: int,
) -> pd.DataFrame:
    return _summarize_window(df, visit_columns, age_visit_cols, max_visit_index=n_visits)


def baseline_plus_landmark_trends(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
    years_list: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """Baseline (v1) values augmented with deltas/slopes from each landmark window.

    For each biomarker stem we keep the v1 value plus, for each window length,
    `_delta_<W>y` and `_slope_<W>y` summarising the trend during that window.
    """
    pieces: list[pd.DataFrame] = []
    static = [c for c in _STATIC_KEEP if c in df.columns]
    if static:
        pieces.append(df[static].copy())
    pieces.append(df[["Age_v1"]].copy())

    age_v1 = df["Age_v1"].astype(float)

    for stem, cols in visit_columns.items():
        if stem == "Age" or not cols:
            continue
        v1_col = cols[0]
        block = pd.DataFrame({f"{stem}_v1": df[v1_col]}, index=df.index)
        ages = _ages_for_visits(df, cols, age_visit_cols)

        for w in years_list:
            cutoff = age_v1 + float(w)
            mask = ages.le(cutoff, axis=0)
            v = df[cols].where(mask).to_numpy(dtype=float)
            a = ages.where(mask).to_numpy(dtype=float)
            valid = ~np.isnan(v)
            first = np.full(v.shape[0], np.nan)
            last = np.full(v.shape[0], np.nan)
            a_first = np.full(v.shape[0], np.nan)
            a_last = np.full(v.shape[0], np.nan)
            for i in range(v.shape[0]):
                idx = np.where(valid[i])[0]
                if len(idx):
                    first[i] = v[i, idx[0]]
                    last[i] = v[i, idx[-1]]
                    a_first[i] = a[i, idx[0]]
                    a_last[i] = a[i, idx[-1]]
            delta = last - first
            span = a_last - a_first
            slope = np.where(span > 1e-6, delta / span, np.nan)
            block[f"{stem}_delta_{int(w)}y"] = delta
            block[f"{stem}_slope_{int(w)}y"] = slope

        pieces.append(block)

    out = pd.concat(pieces, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    _LOG.info("baseline_plus_landmark_trends -> %s", out.shape)
    return out
