"""Data loading + survival target construction.

Death-target modes:
  'drop_missing_death'           : exclude NaN-death rows
  'censor_missing_death_at_last' : treat them as right-censored at last visit
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal
from .config import TRAIN_CSV, TEST_CSV, MAX_VISITS

DeathMode = Literal["drop_missing_death", "censor_missing_death_at_last"]


def load_raw():
    return pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)


def visit_columns(var: str, max_visits: int = MAX_VISITS) -> list[str]:
    return [f"{var}_v{i}" for i in range(1, max_visits + 1)]


def add_visit_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    age_cols = [c for c in visit_columns("Age") if c in df.columns]
    df["min_age"] = df[age_cols].min(axis=1)
    df["max_age"] = df[age_cols].max(axis=1)
    df["n_visits"] = df[age_cols].notna().sum(axis=1)
    df["followup_yrs"] = df["max_age"] - df["min_age"]
    return df


def build_hepatic_target(df: pd.DataFrame) -> pd.DataFrame:
    df = add_visit_metadata(df)
    out = pd.DataFrame({"patient_id_anon": df["patient_id_anon"].values})
    is_event = (df["evenements_hepatiques_majeurs"] == 1).values
    bad = is_event & df["evenements_hepatiques_age_occur"].isna().values
    if bad.any():
        print(f"[hepatic] WARNING: {bad.sum()} rows event=1 but age_occur NaN")
    event_age = df["evenements_hepatiques_age_occur"].values
    age_v1 = df["Age_v1"].values
    max_age = df["max_age"].values
    time = np.where(is_event, event_age - age_v1, max_age - age_v1)
    out["hepatic_event"] = is_event
    out["hepatic_time"] = np.maximum(time.astype(float), 0.001)
    out["hepatic_valid"] = ~bad
    return out


def build_death_target(df: pd.DataFrame, mode: DeathMode) -> pd.DataFrame:
    df = add_visit_metadata(df)
    out = pd.DataFrame({"patient_id_anon": df["patient_id_anon"].values})
    death = df["death"].values
    death_age = df["death_age_occur"].values
    age_v1 = df["Age_v1"].values
    max_age = df["max_age"].values
    is_event_obs = np.asarray(death == 1)
    is_alive = np.asarray(death == 0)
    is_unknown = np.asarray(pd.isna(death))
    bad_event = is_event_obs & pd.isna(death_age)
    if bad_event.any():
        print(f"[death] WARNING: {int(bad_event.sum())} rows death=1 but age NaN")
    if mode == "drop_missing_death":
        valid = (is_alive | is_event_obs) & ~bad_event
    elif mode == "censor_missing_death_at_last":
        valid = (is_alive | is_event_obs | is_unknown) & ~bad_event
    else:
        raise ValueError(f"Unknown death mode: {mode}")
    time = np.where(
        is_event_obs,
        np.where(np.isnan(death_age), 0.001, death_age - age_v1),
        max_age - age_v1,
    )
    out["death_event"] = is_event_obs.astype(bool)
    out["death_time"] = np.maximum(time.astype(float), 0.001)
    out["death_valid"] = valid
    return out


def build_targets(df: pd.DataFrame, death_mode: DeathMode):
    return build_hepatic_target(df), build_death_target(df, death_mode)


def survival_y(events: np.ndarray, times: np.ndarray):
    from sksurv.util import Surv
    return Surv.from_arrays(event=events.astype(bool), time=times.astype(float))
