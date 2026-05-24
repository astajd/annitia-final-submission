"""Phase 3.6 hepatic-focused feature variants.

Two builders:

- ``current_state_v2_no_visit_history`` — `current_state_v2` minus the
  `visit_history_current_state` group (n_visits, age_first/last_visit,
  followup_span, gap_*). This is the feature view that drove the +0.0037
  hepatic OOF gain in Phase 3.5 ablation E.
- ``current_state_v2_hepatic_aug`` — `current_state_v2` plus a small set of
  clinically motivated hepatic interactions: T2DM x stiffness, FIB-4 x
  platelet trend, AST/ALT trend interactions. No target-derived columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger
from .current_state_v2 import current_state_v2 as _csv2_full

_LOG = get_logger(__name__)


_VISIT_HISTORY_COLS = (
    "n_visits",
    "age_first_visit",
    "age_last_visit",
    "followup_span",
    "gap_mean",
    "gap_max",
    "gap_min",
)


def current_state_v2_no_visit_history(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
) -> pd.DataFrame:
    full = _csv2_full(df, visit_columns, age_visit_cols)
    drop = [c for c in _VISIT_HISTORY_COLS if c in full.columns]
    out = full.drop(columns=drop)
    _LOG.info("current_state_v2_no_visit_history -> %s (dropped %d cols)", out.shape, len(drop))
    return out


def _safe(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].astype(float) if col in df.columns else pd.Series(np.nan, index=df.index)


def _hepatic_aug_extras(out: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add hepatic-focused interactions on top of an existing feature frame."""
    extras = pd.DataFrame(index=df.index)

    # Pull the latest/slope columns we need from the existing frame so the
    # interactions are consistent with the rest of current_state_v2.
    stiff_latest = _safe(out, "fibs_stiffness_med_BM_1_latest")
    fibrotest_latest = _safe(out, "fibrotest_BM_2_latest")
    fibrotest_slope = _safe(out, "fibrotest_BM_2_slope")
    fibrotest_max = _safe(out, "fibrotest_BM_2_max")
    fib4_latest = _safe(out, "fib4_latest")
    fib4_max = _safe(out, "fib4_max" if "fib4_max" in out.columns else "fib4_latest")
    fib4_slope = _safe(out, "fib4_slope")
    apri_latest = _safe(out, "apri_latest")
    apri_max = _safe(out, "apri_max" if "apri_max" in out.columns else "apri_latest")
    apri_slope = _safe(out, "apri_slope")
    plt_slope = _safe(out, "plt_slope")
    ast_alt_slope = _safe(out, "ast_alt_ratio_slope")

    t2dm = df["T2DM"].astype(float) if "T2DM" in df.columns else pd.Series(0, index=df.index, dtype=float)
    htn = df["Hypertension"].astype(float) if "Hypertension" in df.columns else pd.Series(0, index=df.index, dtype=float)
    dyslip = df["Dyslipidaemia"].astype(float) if "Dyslipidaemia" in df.columns else pd.Series(0, index=df.index, dtype=float)

    extras["aug_t2dm_x_stiffness_latest"] = t2dm * stiff_latest
    extras["aug_htn_x_stiffness_latest"] = htn * stiff_latest
    extras["aug_dyslip_x_stiffness_latest"] = dyslip * stiff_latest
    extras["aug_fib4_x_plt_slope"] = fib4_latest * plt_slope
    extras["aug_fib4_x_astalt_slope"] = fib4_latest * ast_alt_slope
    extras["aug_fibrotest_x_plt_slope"] = fibrotest_latest * plt_slope
    extras["aug_stiffness_x_plt_slope"] = stiff_latest * plt_slope
    extras["aug_apri_max_x_stiffness_latest"] = apri_max * stiff_latest
    extras["aug_fib4_max_x_stiffness_latest"] = fib4_max * stiff_latest
    extras["aug_fibrotest_max_x_age_v1"] = fibrotest_max * df["Age_v1"].astype(float) if "Age_v1" in df.columns else fibrotest_max
    extras["aug_fib4_slope_x_apri_slope"] = fib4_slope * apri_slope
    extras["aug_fibrotest_slope_x_age_v1"] = fibrotest_slope * df["Age_v1"].astype(float) if "Age_v1" in df.columns else fibrotest_slope

    return extras


def current_state_v2_hepatic_aug(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
) -> pd.DataFrame:
    full = _csv2_full(df, visit_columns, age_visit_cols)
    extras = _hepatic_aug_extras(full, df)
    out = pd.concat([full, extras], axis=1)
    _LOG.info("current_state_v2_hepatic_aug -> %s (+%d hep extras)", out.shape, extras.shape[1])
    return out
