"""Current-state v2: rich snapshot using all available row information.

Per the Phase 3 spec we include:
- latest, max, min, mean, slope, delta per biomarker stem (no target use)
- visit cadence: n_visits, gap stats, follow-up span, current/last observed age
- missingness rates per stem and per visit
- NIT trajectories
- fibrosis-score trajectories (FIB-4 / APRI / AST-ALT) at v1, v2, v3, latest
- clinically meaningful interactions (FIB-4_latest x stiffness_latest, etc.)

This *does* lean on follow-up cadence — the user's Phase 3 ask explicitly
permits ``current age / last observed age``. Strict-target-cutoff fields
(event_age, censoring_age) are still excluded; the leakage tag is therefore
"moderate-high" because cadence/last_age behave like follow-up proxies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger, visit_index
from .missingness import missingness_by_group, visit_cadence
from .refined_longitudinal import (
    LAB_STEMS,
    NIT_STEMS,
    clinical_scores_dynamic,
    longitudinal_no_followup_proxies,
)

_LOG = get_logger(__name__)

_STATIC = ("gender", "T2DM", "Hypertension", "Dyslipidaemia", "bariatric_surgery", "bariatric_surgery_age")


def _ages_for_visits(
    df: pd.DataFrame, biomarker_cols: list[str], age_visit_cols: list[str]
) -> pd.DataFrame:
    age_lookup = {visit_index(c): c for c in age_visit_cols if visit_index(c) is not None}
    aligned = pd.DataFrame(index=df.index)
    for c in biomarker_cols:
        aligned[c] = df[age_lookup[visit_index(c)]] if visit_index(c) in age_lookup else np.nan
    return aligned


def _latest_per_stem(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
    skip_stems: tuple[str, ...] = ("Age",),
) -> pd.DataFrame:
    """Latest non-null value per biomarker stem, plus the age at which it was taken."""
    out = pd.DataFrame(index=df.index)
    for stem, cols in visit_columns.items():
        if stem in skip_stems or not cols:
            continue
        v = df[cols].to_numpy(dtype=float)
        a = _ages_for_visits(df, cols, age_visit_cols).to_numpy(dtype=float)
        latest = np.full(v.shape[0], np.nan)
        a_latest = np.full(v.shape[0], np.nan)
        for i in range(v.shape[0]):
            idx = np.where(~np.isnan(v[i]))[0]
            if len(idx):
                latest[i] = v[i, idx[-1]]
                a_latest[i] = a[i, idx[-1]]
        out[f"{stem}_latest"] = latest
        out[f"{stem}_age_latest"] = a_latest
    return out


def _interactions(latest: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Clinically meaningful pairwise interactions on the latest snapshots."""
    cand = {}
    fibrotest = latest.get("fibrotest_BM_2_latest")
    stiffness = latest.get("fibs_stiffness_med_BM_1_latest")
    aixp = latest.get("aixp_aix_result_BM_3_latest")
    fib4 = scores.get("fib4_latest")
    apri = scores.get("apri_latest")
    ast_alt = scores.get("ast_alt_ratio_latest")
    plt_ = latest.get("plt_latest")
    bmi = latest.get("BMI_latest")
    age_latest = latest.get("fibs_stiffness_med_BM_1_age_latest")

    def _mul(a, b, name):
        if a is not None and b is not None:
            cand[name] = a * b

    _mul(fibrotest, stiffness, "x_fibrotest_stiffness_latest")
    _mul(fibrotest, fib4, "x_fibrotest_fib4_latest")
    _mul(stiffness, fib4, "x_stiffness_fib4_latest")
    _mul(stiffness, ast_alt, "x_stiffness_astalt_latest")
    _mul(aixp, fib4, "x_aixp_fib4_latest")
    _mul(stiffness, age_latest, "x_stiffness_age_latest")
    _mul(fib4, apri, "x_fib4_apri_latest")
    _mul(bmi, fib4, "x_bmi_fib4_latest")
    if plt_ is not None and stiffness is not None:
        cand["x_stiffness_over_plt_latest"] = stiffness / plt_.replace(0, np.nan)
    if fibrotest is not None and ast_alt is not None:
        cand["x_fibrotest_astalt_latest"] = fibrotest * ast_alt
    return pd.DataFrame(cand, index=latest.index) if cand else pd.DataFrame(index=latest.index)


def current_state_v2(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
) -> pd.DataFrame:
    """Build the current-state v2 feature matrix."""
    pieces: list[pd.DataFrame] = []

    static = [c for c in _STATIC if c in df.columns]
    if static:
        pieces.append(df[static].copy())
    pieces.append(df[["Age_v1"]].copy())

    # Trajectories without explicit follow-up proxies (first/last/mean/std/...).
    pieces.append(longitudinal_no_followup_proxies(df, visit_columns, age_visit_cols))

    # Latest snapshots + age-at-latest per stem.
    latest = _latest_per_stem(df, visit_columns, age_visit_cols)
    pieces.append(latest)

    # Dynamic clinical scores at v1/v2/v3/latest plus deltas/slopes.
    scores = clinical_scores_dynamic(df, visit_columns)
    pieces.append(scores)

    # Visit cadence + missingness (the user explicitly allows current age /
    # last observed age, so these stay in).
    pieces.append(visit_cadence(df, age_visit_cols))
    pieces.append(missingness_by_group(df, visit_columns))

    # Interactions on the latest snapshot.
    inter = _interactions(latest, scores)
    if not inter.empty:
        pieces.append(inter)

    out = pd.concat(pieces, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    _LOG.info("current_state_v2 -> %s", out.shape)
    return out
