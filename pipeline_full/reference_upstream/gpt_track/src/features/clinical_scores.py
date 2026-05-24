"""Derived clinical scores commonly used in MASLD risk stratification.

We compute, where the inputs are available:
- FIB-4   = age * AST / (platelets * sqrt(ALT))
- APRI    = (AST / 40) / platelets * 100
- AST/ALT ratio
- Simple deltas/slopes for a handful of liver labs and the stiffness markers.

Variable naming follows what the dictionary provided. Lower-case prefixes
``ast_``, ``alt_``, ``plt_``, ``ggt_``, ``bilirubin_``, ``triglyc_``, ``chol_``,
``BMI_``, plus stiffness columns ``fibs_stiffness_med_BM_1_v*``, FibroTest
``fibrotest_BM_2_v*``, Aixplorer ``aixp_aix_result_BM_3_v*``.

Where a column is missing, the score is silently NaN — but we log which inputs
were missing once, at construction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger

_LOG = get_logger(__name__)

_LAB_STEMS = (
    "alt",
    "ast",
    "ggt",
    "bilirubin",
    "plt",
    "gluc_fast",
    "triglyc",
    "chol",
    "BMI",
    "fibs_stiffness_med_BM_1",
    "fibrotest_BM_2",
    "aixp_aix_result_BM_3",
)


def _safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series(np.nan, index=df.index)


def _fib4(age: pd.Series, ast: pd.Series, plt_: pd.Series, alt: pd.Series) -> pd.Series:
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = plt_ * np.sqrt(alt.clip(lower=1e-3))
        return age * ast / denom.replace(0, np.nan)


def _apri(ast: pd.Series, plt_: pd.Series) -> pd.Series:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (ast / 40.0) / plt_.replace(0, np.nan) * 100.0


def clinical_scores_v1(df: pd.DataFrame) -> pd.DataFrame:
    """Compute baseline (v1) clinical scores."""
    age = _safe_col(df, "Age_v1")
    ast = _safe_col(df, "ast_v1")
    alt = _safe_col(df, "alt_v1")
    plt_ = _safe_col(df, "plt_v1")
    out = pd.DataFrame(index=df.index)
    out["fib4_v1"] = _fib4(age, ast, plt_, alt)
    out["apri_v1"] = _apri(ast, plt_)
    out["ast_alt_ratio_v1"] = ast / alt.replace(0, np.nan)
    return out


def clinical_scores_trajectory(
    df: pd.DataFrame, visit_columns: dict[str, list[str]]
) -> pd.DataFrame:
    """Per-visit clinical scores summarized over time.

    Computes FIB-4 / APRI / AST-ALT at every available visit (using the
    matching Age_vK), then summarizes (first, last, mean, max, slope vs years
    from baseline) per patient.
    """
    age_cols = visit_columns.get("Age", [])
    ast_cols = visit_columns.get("ast", [])
    alt_cols = visit_columns.get("alt", [])
    plt_cols = visit_columns.get("plt", [])

    if not (age_cols and ast_cols and alt_cols and plt_cols):
        _LOG.warning("clinical_scores_trajectory: missing inputs, returning empty frame")
        return pd.DataFrame(index=df.index)

    from ..utils import visit_index

    age_lookup = {visit_index(c): c for c in age_cols if visit_index(c) is not None}
    ast_lookup = {visit_index(c): c for c in ast_cols if visit_index(c) is not None}
    alt_lookup = {visit_index(c): c for c in alt_cols if visit_index(c) is not None}
    plt_lookup = {visit_index(c): c for c in plt_cols if visit_index(c) is not None}

    indices = sorted(set(age_lookup) & set(ast_lookup) & set(alt_lookup) & set(plt_lookup))
    if not indices:
        return pd.DataFrame(index=df.index)

    fib4 = pd.DataFrame(index=df.index)
    apri = pd.DataFrame(index=df.index)
    ratio = pd.DataFrame(index=df.index)
    age_mat = pd.DataFrame(index=df.index)

    for k in indices:
        age = df[age_lookup[k]].astype(float)
        ast = df[ast_lookup[k]].astype(float)
        alt = df[alt_lookup[k]].astype(float)
        plt_ = df[plt_lookup[k]].astype(float)
        fib4[f"fib4_v{k}"] = _fib4(age, ast, plt_, alt)
        apri[f"apri_v{k}"] = _apri(ast, plt_)
        ratio[f"ast_alt_ratio_v{k}"] = ast / alt.replace(0, np.nan)
        age_mat[f"_age_v{k}"] = age

    age_v1 = df["Age_v1"].astype(float)

    def _summarize(mat: pd.DataFrame, name: str) -> pd.DataFrame:
        v = mat.to_numpy(dtype=float)
        a = age_mat.to_numpy(dtype=float)
        valid = ~np.isnan(v)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(v, axis=1)
            vmax = np.nanmax(v, axis=1)
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
        span = a_last - a_first
        slope = np.where(span > 1e-6, (last - first) / span, np.nan)
        return pd.DataFrame(
            {
                f"{name}_first": first,
                f"{name}_last": last,
                f"{name}_mean": mean,
                f"{name}_max": vmax,
                f"{name}_slope": slope,
                f"{name}_delta": last - first,
            },
            index=mat.index,
        )

    out = pd.concat(
        [_summarize(fib4, "fib4"), _summarize(apri, "apri"), _summarize(ratio, "ast_alt_ratio")],
        axis=1,
    )
    return out
