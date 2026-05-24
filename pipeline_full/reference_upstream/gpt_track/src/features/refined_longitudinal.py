"""Refined longitudinal feature sets — explicitly drop follow-up proxies.

These sets are designed to retain biomarker trajectory information while
shedding the obvious leakage observed in Phase 1: visit count, last observed
age, follow-up span, and time-since-baseline. Per-stem we keep the trajectory
shape (first/last/mean/min/max/std/delta/slope) but not the bookkeeping that
encodes how long the patient was followed.

We do *not* mask post-event visits in train. The methodological argument is:
- For the qualitative submission we should also produce a separately-tagged
  feature set that masks them at training time, but masking introduces a
  train/test asymmetry (covered by Phase 1's strict_time_aligned).
- These "no-followup-proxies" sets keep train and test symmetric.
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

NIT_STEMS = (
    "fibs_stiffness_med_BM_1",
    "fibrotest_BM_2",
    "aixp_aix_result_BM_3",
)

LAB_STEMS = (
    "alt",
    "ast",
    "ggt",
    "bilirubin",
    "plt",
    "gluc_fast",
    "triglyc",
    "chol",
    "BMI",
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


def _summarize_stem_no_proxies(
    values: pd.DataFrame, ages: pd.DataFrame, age_v1: pd.Series, stem: str
) -> pd.DataFrame:
    """Trajectory summary excluding count / age_first / age_last / time_since_baseline."""
    v = values.to_numpy(dtype=float)
    a = ages.to_numpy(dtype=float)

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
        },
        index=values.index,
    )
    return out


def longitudinal_no_followup_proxies(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
    keep_stems: tuple[str, ...] | None = None,
    skip_stems: tuple[str, ...] = ("Age",),
    add_missingness: bool = False,
) -> pd.DataFrame:
    """Per-stem trajectory summary minus follow-up bookkeeping.

    ``keep_stems``: if given, only include these biomarker stems.
    ``add_missingness``: if True, also include a per-stem fraction-missing column.
        Documented as moderate leakage in the report.
    """
    age_v1 = df["Age_v1"]
    pieces: list[pd.DataFrame] = []

    static = [c for c in _STATIC_KEEP if c in df.columns]
    if static:
        pieces.append(df[static].copy())
    pieces.append(df[["Age_v1"]].copy())

    for stem, cols in visit_columns.items():
        if stem in skip_stems or not cols:
            continue
        if keep_stems is not None and stem not in keep_stems:
            continue
        ages = _ages_for_visits(df, cols, age_visit_cols)
        block = _summarize_stem_no_proxies(df[cols], ages, age_v1, stem)
        if add_missingness:
            block[f"{stem}_miss_rate"] = df[cols].isna().mean(axis=1).values
        pieces.append(block)

    out = pd.concat(pieces, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    _LOG.info(
        "longitudinal_no_followup_proxies (keep=%s, miss=%s) -> %s",
        keep_stems, add_missingness, out.shape,
    )
    return out


def clinical_scores_dynamic(
    df: pd.DataFrame, visit_columns: dict[str, list[str]]
) -> pd.DataFrame:
    """FIB-4 / APRI / AST-ALT at v1, v2, v3, latest, plus deltas and slopes.

    Latest is the last visit where all four inputs are available, which keeps
    the latest trajectory snapshot stable across patients without depending on
    visit count itself.
    """
    age_cols = visit_columns.get("Age", [])
    ast_cols = visit_columns.get("ast", [])
    alt_cols = visit_columns.get("alt", [])
    plt_cols = visit_columns.get("plt", [])
    if not (age_cols and ast_cols and alt_cols and plt_cols):
        _LOG.warning("clinical_scores_dynamic: missing inputs, returning empty frame")
        return pd.DataFrame(index=df.index)

    age_lookup = {visit_index(c): c for c in age_cols}
    ast_lookup = {visit_index(c): c for c in ast_cols}
    alt_lookup = {visit_index(c): c for c in alt_cols}
    plt_lookup = {visit_index(c): c for c in plt_cols}
    indices = sorted(
        set(age_lookup) & set(ast_lookup) & set(alt_lookup) & set(plt_lookup)
    )

    fib4 = pd.DataFrame(index=df.index)
    apri = pd.DataFrame(index=df.index)
    ratio = pd.DataFrame(index=df.index)
    age_mat = pd.DataFrame(index=df.index)

    for k in indices:
        age = df[age_lookup[k]].astype(float)
        ast = df[ast_lookup[k]].astype(float)
        alt = df[alt_lookup[k]].astype(float)
        plt_ = df[plt_lookup[k]].astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            fib4[k] = age * ast / (plt_ * np.sqrt(alt.clip(lower=1e-3))).replace(0, np.nan)
            apri[k] = (ast / 40.0) / plt_.replace(0, np.nan) * 100.0
            ratio[k] = ast / alt.replace(0, np.nan)
        age_mat[k] = age

    out = pd.DataFrame(index=df.index)
    for name, mat in [("fib4", fib4), ("apri", apri), ("ast_alt_ratio", ratio)]:
        for k in (1, 2, 3):
            if k in mat.columns:
                out[f"{name}_v{k}"] = mat[k]

        # latest at the largest visit index where the value is present per row
        v = mat.to_numpy(dtype=float)
        a = age_mat.to_numpy(dtype=float)
        valid = ~np.isnan(v)
        latest = np.full(v.shape[0], np.nan)
        a_latest = np.full(v.shape[0], np.nan)
        first = np.full(v.shape[0], np.nan)
        a_first = np.full(v.shape[0], np.nan)
        for i in range(v.shape[0]):
            idx = np.where(valid[i])[0]
            if len(idx):
                first[i] = v[i, idx[0]]
                latest[i] = v[i, idx[-1]]
                a_first[i] = a[i, idx[0]]
                a_latest[i] = a[i, idx[-1]]
        out[f"{name}_latest"] = latest
        out[f"{name}_delta"] = latest - first
        span = a_latest - a_first
        out[f"{name}_slope"] = np.where(span > 1e-6, (latest - first) / span, np.nan)

    return out


def labs_longitudinal_only(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
) -> pd.DataFrame:
    return longitudinal_no_followup_proxies(
        df, visit_columns, age_visit_cols, keep_stems=LAB_STEMS,
    )


def nit_longitudinal_only(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
) -> pd.DataFrame:
    return longitudinal_no_followup_proxies(
        df, visit_columns, age_visit_cols, keep_stems=NIT_STEMS,
    )


def nit_plus_scores_longitudinal(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
) -> pd.DataFrame:
    nit = nit_longitudinal_only(df, visit_columns, age_visit_cols)
    scores = clinical_scores_dynamic(df, visit_columns)
    out = pd.concat([nit, scores], axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def aggressive_longitudinal(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
) -> pd.DataFrame:
    """All trajectory summaries + per-stem missingness rates.

    Keeps `*_miss_rate` (moderate leakage) but excludes the most explicit
    follow-up proxies (n_visits, followup_span, last age).
    """
    base = longitudinal_no_followup_proxies(
        df, visit_columns, age_visit_cols, add_missingness=True
    )
    scores = clinical_scores_dynamic(df, visit_columns)
    out = pd.concat([base, scores], axis=1)
    return out.loc[:, ~out.columns.duplicated()]
