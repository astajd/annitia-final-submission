"""Feature-set orchestration.

Each named feature set is a function that takes the loaded :class:`Dataset`
plus optional per-row cutoff ages (for strict alignment) and returns aligned
``X_train`` / ``X_test`` DataFrames.

Every feature set carries a ``leakage_risk`` tag so downstream code can flag
high-risk experiments in the report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..data_loading import Dataset
from ..utils import get_logger
from .baseline import baseline_v1_features, early_v1_v3_features
from .clinical_scores import clinical_scores_trajectory, clinical_scores_v1
from .current_state_v2 import current_state_v2
from .hep_focus import current_state_v2_hepatic_aug, current_state_v2_no_visit_history
from .landmark import baseline_plus_landmark_trends, first_n_visits, first_window
from .longitudinal import all_visits_longitudinal, strict_time_aligned_longitudinal
from .missingness import missingness_by_group, visit_cadence
from .refined_longitudinal import (
    aggressive_longitudinal,
    clinical_scores_dynamic,
    labs_longitudinal_only,
    longitudinal_no_followup_proxies,
    nit_longitudinal_only,
    nit_plus_scores_longitudinal,
)

_LOG = get_logger(__name__)

_NIT_STEMS = (
    "fibs_stiffness_med_BM_1",
    "fibrotest_BM_2",
    "aixp_aix_result_BM_3",
)


@dataclass
class FeatureSet:
    name: str
    leakage_risk: str  # "low" | "low/moderate" | "moderate" | "high"
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    description: str = ""


FeatureBuilder = Callable[[object, dict], "FeatureSet"]


def _filter_nit(visit_columns: dict[str, list[str]]) -> dict[str, list[str]]:
    return {s: cols for s, cols in visit_columns.items() if s in _NIT_STEMS}


def _align(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure both frames share identical columns in the same order."""
    cols = list(X_train.columns)
    extra = [c for c in X_test.columns if c not in cols]
    if extra:
        X_test = X_test.drop(columns=extra)
    missing = [c for c in cols if c not in X_test.columns]
    for m in missing:
        X_test[m] = np.nan
    X_test = X_test[cols]
    return X_train, X_test


def build_baseline_v1(ds, opts=None) -> FeatureSet:
    Xtr = baseline_v1_features(ds.train_df, ds.visit_columns)
    Xte = baseline_v1_features(ds.test_df, ds.visit_columns)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("baseline_v1", "low", Xtr, Xte,
                      "Static demographics + every biomarker measured at v1.")


def build_early_v1_v3(ds, opts=None) -> FeatureSet:
    Xtr = early_v1_v3_features(ds.train_df, ds.visit_columns)
    Xte = early_v1_v3_features(ds.test_df, ds.visit_columns)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("early_v1_v3", "low/moderate", Xtr, Xte,
                      "v1+v2+v3 raw + simple deltas/slopes.")


def build_strict_time_aligned(ds, opts) -> FeatureSet:
    cutoff = opts.get("cutoff_age_train")
    if cutoff is None:
        raise ValueError("strict_time_aligned requires opts['cutoff_age_train']")
    Xtr = strict_time_aligned_longitudinal(
        ds.train_df, ds.visit_columns, ds.age_visit_cols, cutoff
    )
    Xte = strict_time_aligned_longitudinal(
        ds.test_df, ds.visit_columns, ds.age_visit_cols, cutoff_age=None
    )
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet(
        "strict_time_aligned",
        "low/moderate",
        Xtr,
        Xte,
        "Longitudinal summaries; on train, post-event/censoring visits masked. "
        "On test, all visits used (event time unknown). Asymmetry is documented.",
    )


def build_all_visits_longitudinal(ds, opts=None) -> FeatureSet:
    Xtr = all_visits_longitudinal(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte = all_visits_longitudinal(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet(
        "all_visits_longitudinal",
        "high",
        Xtr,
        Xte,
        "Per-stem longitudinal summaries over every visit. High leakage on train.",
    )


def build_nit_only(ds, opts=None) -> FeatureSet:
    nit_visits = _filter_nit(ds.visit_columns)
    Xtr = all_visits_longitudinal(ds.train_df, nit_visits, ds.age_visit_cols)
    Xte = all_visits_longitudinal(ds.test_df, nit_visits, ds.age_visit_cols)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("NIT_only", "low/moderate", Xtr, Xte,
                      "Longitudinal summaries on stiffness / FibroTest / Aixplorer only.")


def build_nit_plus_clinical_scores(ds, opts=None) -> FeatureSet:
    nit_visits = _filter_nit(ds.visit_columns)
    nit_tr = all_visits_longitudinal(ds.train_df, nit_visits, ds.age_visit_cols)
    nit_te = all_visits_longitudinal(ds.test_df, nit_visits, ds.age_visit_cols)

    base_tr = baseline_v1_features(ds.train_df, ds.visit_columns)
    base_te = baseline_v1_features(ds.test_df, ds.visit_columns)
    cs1_tr = clinical_scores_v1(ds.train_df)
    cs1_te = clinical_scores_v1(ds.test_df)
    cst_tr = clinical_scores_trajectory(ds.train_df, ds.visit_columns)
    cst_te = clinical_scores_trajectory(ds.test_df, ds.visit_columns)

    Xtr = pd.concat([base_tr, cs1_tr, cst_tr, nit_tr], axis=1)
    Xte = pd.concat([base_te, cs1_te, cst_te, nit_te], axis=1)
    Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]
    Xte = Xte.loc[:, ~Xte.columns.duplicated()]
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet(
        "NIT_plus_clinical_scores",
        "low/moderate",
        Xtr,
        Xte,
        "Baseline + FIB-4/APRI/AST-ALT (v1 and trajectory) + NIT longitudinal.",
    )


def build_missingness_and_visit_cadence(ds, opts=None) -> FeatureSet:
    cad_tr = visit_cadence(ds.train_df, ds.age_visit_cols)
    cad_te = visit_cadence(ds.test_df, ds.age_visit_cols)
    miss_tr = missingness_by_group(ds.train_df, ds.visit_columns)
    miss_te = missingness_by_group(ds.test_df, ds.visit_columns)
    Xtr = pd.concat([cad_tr, miss_tr], axis=1)
    Xte = pd.concat([cad_te, miss_te], axis=1)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet(
        "missingness_and_visit_cadence",
        "high",
        Xtr,
        Xte,
        "Visit count, gap stats, missingness rates per stem and per visit.",
    )


def build_full_high_risk(ds, opts=None) -> FeatureSet:
    base = baseline_v1_features(ds.train_df, ds.visit_columns)
    base_te = baseline_v1_features(ds.test_df, ds.visit_columns)
    long_tr = all_visits_longitudinal(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    long_te = all_visits_longitudinal(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    cad_tr = visit_cadence(ds.train_df, ds.age_visit_cols)
    cad_te = visit_cadence(ds.test_df, ds.age_visit_cols)
    miss_tr = missingness_by_group(ds.train_df, ds.visit_columns)
    miss_te = missingness_by_group(ds.test_df, ds.visit_columns)
    cs1_tr = clinical_scores_v1(ds.train_df)
    cs1_te = clinical_scores_v1(ds.test_df)
    cst_tr = clinical_scores_trajectory(ds.train_df, ds.visit_columns)
    cst_te = clinical_scores_trajectory(ds.test_df, ds.visit_columns)
    Xtr = pd.concat([base, long_tr, cad_tr, miss_tr, cs1_tr, cst_tr], axis=1)
    Xte = pd.concat([base_te, long_te, cad_te, miss_te, cs1_te, cst_te], axis=1)
    Xtr = Xtr.loc[:, ~Xtr.columns.duplicated()]
    Xte = Xte.loc[:, ~Xte.columns.duplicated()]
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("full_high_risk", "high", Xtr, Xte,
                      "Everything: baseline + longitudinal + cadence + missingness + scores.")


def build_first_1y(ds, opts=None) -> FeatureSet:
    Xtr = first_window(ds.train_df, ds.visit_columns, ds.age_visit_cols, years=1.0)
    Xte = first_window(ds.test_df, ds.visit_columns, ds.age_visit_cols, years=1.0)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("first_1y", "low", Xtr, Xte,
                      "Measurements within 1 year after Age_v1; target-independent cutoff.")


def build_first_2y(ds, opts=None) -> FeatureSet:
    Xtr = first_window(ds.train_df, ds.visit_columns, ds.age_visit_cols, years=2.0)
    Xte = first_window(ds.test_df, ds.visit_columns, ds.age_visit_cols, years=2.0)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("first_2y", "low", Xtr, Xte,
                      "Measurements within 2 years after Age_v1; target-independent cutoff.")


def build_first_3y(ds, opts=None) -> FeatureSet:
    Xtr = first_window(ds.train_df, ds.visit_columns, ds.age_visit_cols, years=3.0)
    Xte = first_window(ds.test_df, ds.visit_columns, ds.age_visit_cols, years=3.0)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("first_3y", "low", Xtr, Xte,
                      "Measurements within 3 years after Age_v1; target-independent cutoff.")


def build_first_3_visits(ds, opts=None) -> FeatureSet:
    Xtr = first_n_visits(ds.train_df, ds.visit_columns, ds.age_visit_cols, n_visits=3)
    Xte = first_n_visits(ds.test_df, ds.visit_columns, ds.age_visit_cols, n_visits=3)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("first_3_visits", "low", Xtr, Xte,
                      "Visits 1-3 only; target-independent cutoff.")


def build_baseline_plus_landmark_trends(ds, opts=None) -> FeatureSet:
    Xtr = baseline_plus_landmark_trends(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte = baseline_plus_landmark_trends(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("baseline_plus_landmark_trends", "low", Xtr, Xte,
                      "Baseline + 1y/2y/3y deltas and slopes per stem.")


def build_longitudinal_no_followup_proxies(ds, opts=None) -> FeatureSet:
    Xtr = longitudinal_no_followup_proxies(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte = longitudinal_no_followup_proxies(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("longitudinal_no_followup_proxies", "moderate", Xtr, Xte,
                      "All-visit trajectory summaries minus follow-up bookkeeping (no count/age_last/etc).")


def build_NIT_longitudinal_only(ds, opts=None) -> FeatureSet:
    Xtr = nit_longitudinal_only(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte = nit_longitudinal_only(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("NIT_longitudinal_only", "moderate", Xtr, Xte,
                      "FibroScan + FibroTest + Aixplorer trajectories, no follow-up proxies.")


def build_labs_longitudinal_only(ds, opts=None) -> FeatureSet:
    Xtr = labs_longitudinal_only(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte = labs_longitudinal_only(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("labs_longitudinal_only", "moderate", Xtr, Xte,
                      "Lab and BMI trajectories only; no follow-up proxies.")


def build_clinical_scores_dynamic(ds, opts=None) -> FeatureSet:
    Xtr = clinical_scores_dynamic(ds.train_df, ds.visit_columns)
    Xte = clinical_scores_dynamic(ds.test_df, ds.visit_columns)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("clinical_scores_dynamic", "low", Xtr, Xte,
                      "FIB-4 / APRI / AST-ALT at v1, v2, v3, latest plus deltas.")


def build_NIT_plus_scores_longitudinal(ds, opts=None) -> FeatureSet:
    Xtr = nit_plus_scores_longitudinal(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte = nit_plus_scores_longitudinal(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("NIT_plus_scores_longitudinal", "moderate", Xtr, Xte,
                      "NIT trajectories + dynamic clinical scores; no follow-up proxies.")


def _build_current_state_v2_no_visit_history(ds, opts=None) -> FeatureSet:
    Xtr = current_state_v2_no_visit_history(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte = current_state_v2_no_visit_history(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet(
        "current_state_v2_no_visit_history", "moderate-high", Xtr, Xte,
        "current_state_v2 minus visit-history group; basis of Phase 3.5 hepatic gain.",
    )


def _build_current_state_v2_hepatic_aug(ds, opts=None) -> FeatureSet:
    Xtr = current_state_v2_hepatic_aug(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte = current_state_v2_hepatic_aug(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet(
        "current_state_v2_hepatic_aug", "moderate-high", Xtr, Xte,
        "current_state_v2 plus clinically motivated hepatic interactions.",
    )


def build_current_state_v2(ds, opts=None) -> FeatureSet:
    Xtr = current_state_v2(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte = current_state_v2(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet(
        "current_state_v2",
        "moderate-high",
        Xtr,
        Xte,
        "Latest values + trajectories + dynamic scores + cadence + missingness "
        "+ clinical interactions. No event/censoring age.",
    )


def build_aggressive_longitudinal(ds, opts=None) -> FeatureSet:
    Xtr = aggressive_longitudinal(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte = aggressive_longitudinal(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    Xtr, Xte = _align(Xtr, Xte)
    return FeatureSet("aggressive_longitudinal", "moderate-high", Xtr, Xte,
                      "All trajectories + per-stem missingness; excludes explicit follow-up proxies.")


_BUILDERS: dict[str, FeatureBuilder] = {
    "baseline_v1": build_baseline_v1,
    "early_v1_v3": build_early_v1_v3,
    "strict_time_aligned": build_strict_time_aligned,
    "all_visits_longitudinal": build_all_visits_longitudinal,
    "NIT_only": build_nit_only,
    "NIT_plus_clinical_scores": build_nit_plus_clinical_scores,
    "missingness_and_visit_cadence": build_missingness_and_visit_cadence,
    "full_high_risk": build_full_high_risk,
    # Phase 2 landmark features (target-independent).
    "first_1y": build_first_1y,
    "first_2y": build_first_2y,
    "first_3y": build_first_3y,
    "first_3_visits": build_first_3_visits,
    "baseline_plus_landmark_trends": build_baseline_plus_landmark_trends,
    # Phase 2 refined longitudinal (no follow-up proxies).
    "longitudinal_no_followup_proxies": build_longitudinal_no_followup_proxies,
    "NIT_longitudinal_only": build_NIT_longitudinal_only,
    "labs_longitudinal_only": build_labs_longitudinal_only,
    "clinical_scores_dynamic": build_clinical_scores_dynamic,
    "NIT_plus_scores_longitudinal": build_NIT_plus_scores_longitudinal,
    "aggressive_longitudinal": build_aggressive_longitudinal,
    # Phase 3 current-state.
    "current_state_v2": build_current_state_v2,
    "current_state_v2_no_visit_history": _build_current_state_v2_no_visit_history,
    "current_state_v2_hepatic_aug": _build_current_state_v2_hepatic_aug,
}


def build_feature_set(name: str, ds, opts: dict | None = None) -> FeatureSet:
    if name not in _BUILDERS:
        raise KeyError(f"unknown feature set {name}; have {sorted(_BUILDERS)}")
    fs = _BUILDERS[name](ds, opts or {})
    _LOG.info(
        "built feature set '%s' [risk=%s] -> X_train %s, X_test %s",
        fs.name, fs.leakage_risk, fs.X_train.shape, fs.X_test.shape,
    )
    return fs


def available_feature_sets() -> list[str]:
    return sorted(_BUILDERS)
