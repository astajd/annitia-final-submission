"""Feature engineering — strict, permissive, and clinical features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Literal
from .config import LONGITUDINAL_VARS, NIT_VARS, STATIC_FEATURES, MAX_VISITS

FeatureSet = Literal[
    "baseline_v1", "early_v1_v3", "strict_time_aligned",
    "longitudinal_summary", "longitudinal_plus_meta",
    "followup_proxy_only", "nit_only", "nit_only_baseline_only",
    "trajectory_shape", "baseline_plus_shape",
]

# Variables to compute trajectory-shape features over. Excludes Age (monotonic
# by construction) and bilirubin (per Phase 3 Experiment 1 spec — explicit
# user-provided list).
SHAPE_VARS = [
    "BMI", "alt", "ast", "ggt", "plt", "gluc_fast", "triglyc", "chol",
    "fibs_stiffness_med_BM_1", "fibrotest_BM_2", "aixp_aix_result_BM_3",
]


def _visit_cols(df: pd.DataFrame, var: str) -> list[str]:
    return [c for c in (f"{var}_v{i}" for i in range(1, MAX_VISITS + 1)) if c in df.columns]


def _summarize_trajectory(values: np.ndarray, ages: np.ndarray) -> dict:
    valid = ~(np.isnan(values) | np.isnan(ages))
    n = int(valid.sum())
    if n == 0:
        return dict(first=np.nan, last=np.nan, mean=np.nan, min=np.nan,
                    max=np.nan, std=np.nan, delta=np.nan, rel_delta=np.nan,
                    slope=np.nan, count=0)
    v, a = values[valid], ages[valid]
    first, last = v[0], v[-1]
    delta = last - first
    rel_delta = (delta / first) if first not in (0,) and not np.isnan(first) else np.nan
    if n >= 2 and a.max() > a.min():
        slope = float(np.polyfit(a, v, 1)[0])
    else:
        slope = np.nan
    return dict(
        first=first, last=last, mean=float(np.mean(v)),
        min=float(np.min(v)), max=float(np.max(v)),
        std=float(np.std(v, ddof=0)) if n >= 2 else 0.0,
        delta=float(delta),
        rel_delta=float(rel_delta) if not np.isnan(rel_delta) else np.nan,
        slope=slope, count=n,
    )


def _trajectory_features(df: pd.DataFrame, var: str) -> pd.DataFrame:
    val_cols = _visit_cols(df, var)
    age_cols = _visit_cols(df, "Age")[: len(val_cols)]
    vals = df[val_cols].to_numpy(dtype=float)
    ages = df[age_cols].to_numpy(dtype=float)
    rows = [_summarize_trajectory(vals[i], ages[i]) for i in range(len(df))]
    out = pd.DataFrame(rows)
    out.columns = [f"{var}_{k}" for k in out.columns]
    return out


def _strict_trajectory_features(df, var, event, time, age_v1) -> pd.DataFrame:
    val_cols = _visit_cols(df, var)
    age_cols = _visit_cols(df, "Age")[: len(val_cols)]
    cutoff_age = age_v1 + time
    vals = df[val_cols].to_numpy(dtype=float)
    ages = df[age_cols].to_numpy(dtype=float)
    rows = []
    for i in range(len(df)):
        keep = ages[i] <= cutoff_age[i] + 1e-9
        v_i = np.where(keep, vals[i], np.nan)
        a_i = np.where(keep, ages[i], np.nan)
        rows.append(_summarize_trajectory(v_i, a_i))
    out = pd.DataFrame(rows)
    out.columns = [f"{var}_{k}" for k in out.columns]
    return out


def _add_clinical_scores(df: pd.DataFrame, suffix: str = "_v1") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    cols = [f"Age{suffix}", f"ast{suffix}", f"alt{suffix}", f"plt{suffix}"]
    if all(c in df.columns for c in cols):
        age, ast, alt, plt = (df[c] for c in cols)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"fib4{suffix}"] = age * ast / (plt * np.sqrt(np.maximum(alt, 1e-9)))
            out[f"apri{suffix}"] = (ast / 40.0) / plt * 100.0
            out[f"ast_alt{suffix}"] = ast / np.maximum(alt, 1e-9)
    return out


def _trajectory_shape_features(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Tier-4 trajectory-shape features for a single variable.

    Per patient (one row of df), summarizes trajectory *shape* without
    computing the actual visit values directly. Designed to be informative
    about disease course while being relatively robust to whether visits
    happen pre- or post-event:
      - monotonicity: Spearman corr between (age, value), in [-1, +1]
      - smoothness:   1 - sum(|diff|) / range; 0 if monotonic, < 0 if zigzag
      - stability:    longest run of consecutive visits within ±10% of the
                      patient's overall median value for this variable
      - recent_ratio: median(last 2 visits) / v1
      - volatility:   std / |mean| (coefficient of variation)
      - acceleration: slope(second half visits) − slope(first half visits)
      - present:      1 if patient has ≥3 measurements of this variable

    Shape features are NaN if the patient has < 3 valid measurements; the
    `present` flag remains a non-NaN integer 0/1.
    """
    from scipy.stats import spearmanr

    val_cols = _visit_cols(df, var)
    n = len(df)
    out_cols = {
        f"{var}_shape_present":      np.zeros(n, dtype=int),
        f"{var}_shape_monotonicity": np.full(n, np.nan),
        f"{var}_shape_smoothness":   np.full(n, np.nan),
        f"{var}_shape_stability":    np.full(n, np.nan),
        f"{var}_shape_recent_ratio": np.full(n, np.nan),
        f"{var}_shape_volatility":   np.full(n, np.nan),
        f"{var}_shape_acceleration": np.full(n, np.nan),
    }
    if not val_cols:
        return pd.DataFrame(out_cols, index=df.index)

    age_cols = _visit_cols(df, "Age")[: len(val_cols)]
    vals = df[val_cols].to_numpy(dtype=float)
    ages = df[age_cols].to_numpy(dtype=float)

    for i in range(n):
        valid = ~(np.isnan(vals[i]) | np.isnan(ages[i]))
        n_val = int(valid.sum())
        if n_val < 3:
            continue
        out_cols[f"{var}_shape_present"][i] = 1
        v = vals[i, valid]
        a = ages[i, valid]
        order = np.argsort(a)
        v = v[order]
        a = a[order]

        # 1. Monotonicity (Spearman age vs value)
        rho, _ = spearmanr(a, v)
        if not np.isnan(rho):
            out_cols[f"{var}_shape_monotonicity"][i] = float(rho)

        # 2. Smoothness: 1 - sum|diff| / range
        diffs = np.diff(v)
        sum_abs = float(np.sum(np.abs(diffs)))
        rng = float(v.max() - v.min())
        if rng > 1e-12:
            out_cols[f"{var}_shape_smoothness"][i] = 1.0 - sum_abs / rng

        # 3. Stability: longest run within ±10% of overall median
        med = float(np.median(v))
        tol = 0.1 * abs(med) if abs(med) > 1e-12 else 1e-12
        stable = np.abs(v - med) <= tol
        max_run = 0
        cur = 0
        for s in stable:
            if s:
                cur += 1
                if cur > max_run:
                    max_run = cur
            else:
                cur = 0
        out_cols[f"{var}_shape_stability"][i] = float(max_run)

        # 4. Recent-vs-baseline ratio (median of last 2 / v1)
        v1 = v[0]
        if abs(v1) > 1e-12:
            recent_med = float(np.median(v[-2:]))
            out_cols[f"{var}_shape_recent_ratio"][i] = recent_med / float(v1)

        # 5. Volatility (std / |mean|)
        m = float(np.mean(v))
        if abs(m) > 1e-12:
            out_cols[f"{var}_shape_volatility"][i] = (
                float(np.std(v, ddof=0)) / abs(m))

        # 6. Trend acceleration (slope of 2nd half − slope of 1st half).
        #    Need ≥4 visits and non-degenerate ages on both halves.
        if n_val >= 4:
            mid = n_val // 2
            a1, v1_arr = a[:mid], v[:mid]
            a2, v2_arr = a[mid:], v[mid:]
            if a1.max() > a1.min() and a2.max() > a2.min():
                s1 = float(np.polyfit(a1, v1_arr, 1)[0])
                s2 = float(np.polyfit(a2, v2_arr, 1)[0])
                out_cols[f"{var}_shape_acceleration"][i] = s2 - s1

    return pd.DataFrame(out_cols, index=df.index)


def _last_available_visit_value(df: pd.DataFrame, var: str) -> pd.Series:
    cols = _visit_cols(df, var)
    if not cols:
        return pd.Series(np.nan, index=df.index, name=f"{var}_last_avail")
    arr = df[cols].to_numpy(dtype=float)
    last = np.full(len(arr), np.nan)
    for i in range(len(arr)):
        nz = np.where(~np.isnan(arr[i]))[0]
        if len(nz):
            last[i] = arr[i, nz[-1]]
    return pd.Series(last, index=df.index, name=f"{var}_last_avail")


def build_features(df: pd.DataFrame, feature_set: FeatureSet, *,
                   event=None, time=None) -> pd.DataFrame:
    df = df.copy()
    if "min_age" not in df.columns:
        from .data import add_visit_metadata
        df = add_visit_metadata(df)

    static = df[[c for c in STATIC_FEATURES if c in df.columns]].copy()
    if "bariatric_surgery_age" in static.columns:
        static["bariatric_surgery_done"] = static["bariatric_surgery_age"].notna().astype(int)
    pieces = [static]

    if feature_set == "baseline_v1":
        cols = [f"{v}_v1" for v in LONGITUDINAL_VARS if f"{v}_v1" in df.columns]
        pieces.append(df[cols].copy())
        pieces.append(_add_clinical_scores(df, "_v1"))

    elif feature_set == "early_v1_v3":
        for vi in (1, 2, 3):
            cols = [f"{v}_v{vi}" for v in LONGITUDINAL_VARS if f"{v}_v{vi}" in df.columns]
            pieces.append(df[cols].copy())
            pieces.append(_add_clinical_scores(df, f"_v{vi}"))

    elif feature_set == "longitudinal_summary":
        for v in LONGITUDINAL_VARS:
            pieces.append(_trajectory_features(df, v))
        pieces.append(_add_clinical_scores(df, "_v1"))
        for var in ["alt", "ast", "plt"]:
            pieces.append(_last_available_visit_value(df, var).to_frame())

    elif feature_set == "longitudinal_plus_meta":
        for v in LONGITUDINAL_VARS:
            pieces.append(_trajectory_features(df, v))
        pieces.append(_add_clinical_scores(df, "_v1"))
        pieces.append(df[["min_age", "max_age", "n_visits", "followup_yrs"]].copy())

    elif feature_set == "followup_proxy_only":
        pieces.append(df[["min_age", "max_age", "n_visits", "followup_yrs"]].copy())

    elif feature_set == "nit_only":
        for v in NIT_VARS:
            pieces.append(_trajectory_features(df, v))
        pieces.append(df[["Age_v1"]].copy())

    elif feature_set == "nit_only_baseline_only":
        nit_v1_cols = [f"{v}_v1" for v in NIT_VARS if f"{v}_v1" in df.columns]
        pieces.append(df[nit_v1_cols].copy())
        pieces.append(df[["Age_v1"]].copy())
        flag_name_map = {
            "fibs_stiffness_med_BM_1": "had_FibroScan_at_v1",
            "fibrotest_BM_2":          "had_FibroTest_at_v1",
            "aixp_aix_result_BM_3":    "had_Aixplorer_at_v1",
        }
        flags = pd.DataFrame(index=df.index)
        for var, fname in flag_name_map.items():
            c = f"{var}_v1"
            flags[fname] = df[c].notna().astype(int) if c in df.columns else 0
        pieces.append(flags)

    elif feature_set == "trajectory_shape":
        for v in SHAPE_VARS:
            pieces.append(_trajectory_shape_features(df, v))

    elif feature_set == "baseline_plus_shape":
        cols = [f"{v}_v1" for v in LONGITUDINAL_VARS if f"{v}_v1" in df.columns]
        pieces.append(df[cols].copy())
        pieces.append(_add_clinical_scores(df, "_v1"))
        for v in SHAPE_VARS:
            pieces.append(_trajectory_shape_features(df, v))

    elif feature_set == "strict_time_aligned":
        if event is None or time is None:
            raise ValueError("strict_time_aligned requires event/time")
        age_v1 = df["Age_v1"].to_numpy(dtype=float)
        for v in LONGITUDINAL_VARS:
            pieces.append(_strict_trajectory_features(df, v, event, time, age_v1))
        pieces.append(_add_clinical_scores(df, "_v1"))

    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    X = pd.concat(pieces, axis=1)
    X = X.select_dtypes(include=[np.number]).copy()
    return X


def build_landmark_features(df: pd.DataFrame, landmark_time: float) -> pd.DataFrame:
    """Tier-2 landmark feature builder (clean longitudinal, fixed reference time).

    For each patient, summarizes their visit history *up to a fixed clinical
    cutoff* `Age_v1 + landmark_time`. This is honest by construction: the
    cutoff is defined by the calendar, not by the outcome.

    Features = baseline_v1 + (LOCF at landmark, slope from v1 to landmark) per
    longitudinal var.

    Caller responsibility (training only): drop rows where the event happened
    before `landmark_time`, since those patients are not at-risk at the
    landmark — use `at_risk_at_landmark(event, time, landmark_time)`.
    """
    df = df.copy()
    if "min_age" not in df.columns:
        from .data import add_visit_metadata
        df = add_visit_metadata(df)

    pieces = [build_features(df, "baseline_v1")]
    age_v1 = df["Age_v1"].to_numpy(dtype=float)
    landmark_age = age_v1 + landmark_time

    age_cols_all = _visit_cols(df, "Age")
    ages_all = df[age_cols_all].to_numpy(dtype=float) if age_cols_all else None

    for v in LONGITUDINAL_VARS:
        val_cols = _visit_cols(df, v)
        if not val_cols:
            continue
        age_cols = _visit_cols(df, "Age")[: len(val_cols)]
        vals = df[val_cols].to_numpy(dtype=float)
        ages = df[age_cols].to_numpy(dtype=float)

        n = len(df)
        locf = np.full(n, np.nan)
        slope = np.full(n, np.nan)
        for i in range(n):
            mask = ((ages[i] <= landmark_age[i] + 1e-9)
                    & ~np.isnan(vals[i]) & ~np.isnan(ages[i]))
            if mask.sum() == 0:
                continue
            v_i = vals[i, mask]
            a_i = ages[i, mask]
            locf[i] = float(v_i[-1])
            if mask.sum() >= 2 and a_i.max() > a_i.min():
                slope[i] = float(np.polyfit(a_i, v_i, 1)[0])

        pieces.append(pd.DataFrame({
            f"{v}_locf_at_landmark": locf,
            f"{v}_slope_v1_to_landmark": slope,
        }, index=df.index))

    X = pd.concat(pieces, axis=1)
    X = X.select_dtypes(include=[np.number]).copy()
    return X


def at_risk_at_landmark(event, time, landmark_time: float) -> np.ndarray:
    """Standard landmark filter (per Phase 2d spec): drop only patients whose
    event happened before the landmark. Censored-before-landmark are kept.
    """
    e = np.asarray(event, dtype=bool)
    t = np.asarray(time, dtype=float)
    return ~(e & (t < landmark_time))


FEATURE_SET_RISK = {
    "baseline_v1":             "low",
    "early_v1_v3":             "low-moderate",
    "strict_time_aligned":     "defensible",
    "nit_only":                "moderate",
    "nit_only_baseline_only":  "low",
    "longitudinal_summary":    "moderate",
    "longitudinal_plus_meta":  "high (visit-count/followup proxies)",
    "followup_proxy_only":     "very high (audit only)",
    "trajectory_shape":        "Tier 4 (uses all visits; shape features "
                               "designed to be robust to post-event ambiguity)",
    "baseline_plus_shape":     "Tier 4 (baseline_v1 + trajectory_shape)",
}


def feature_set_risk_label(name: FeatureSet) -> str:
    return FEATURE_SET_RISK[name]
