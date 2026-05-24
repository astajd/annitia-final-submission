"""Discrete-time hazard model in patient-period (long) format.

For each patient we expand the survival data into one row per yearly time bin
between baseline and the patient's last observation (or event). The label in
each row is 1 iff the event happened *during* that bin, else 0. Features in
each row are the patient's covariates as known at the *start* of that bin —
that is, biomarkers from visits whose Age <= bin_start_age.

A LightGBM binary classifier trained on this long-format dataset learns the
hazard rate per bin. Patient-level risk is the cumulative incidence
``1 - prod(1 - hazard_t)`` over all bins, which is monotone in the hazard
sum, so the C-index applies.

This is independent of the wide-format Phase 1/2 models and provides a
genuine source of diversity for stacking. The user explicitly tagged it as
experimental; we evaluate it under the same repeated-stratified CV and only
promote to a candidate submission if it is locally competitive.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .metrics import cindex
from .models._preprocess import fill_for_tree
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger, visit_index
from .validation import build_folds

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Long-format builder
# ---------------------------------------------------------------------------

def _last_known_value_at_age(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
    cutoff_age: pd.Series,
    skip_stems: tuple[str, ...] = ("Age",),
) -> pd.DataFrame:
    """Per-row latest non-null biomarker value with visit Age <= cutoff."""
    out = pd.DataFrame(index=df.index)
    age_lookup = {visit_index(c): c for c in age_visit_cols if visit_index(c) is not None}
    cutoff = cutoff_age.to_numpy(dtype=float)
    for stem, cols in visit_columns.items():
        if stem in skip_stems or not cols:
            continue
        v = df[cols].to_numpy(dtype=float)
        a = np.zeros_like(v)
        for j, c in enumerate(cols):
            ac = age_lookup.get(visit_index(c))
            a[:, j] = df[ac].to_numpy(dtype=float) if ac else np.nan
        # mask post-cutoff visits
        m = ~(a > cutoff[:, None]) & ~np.isnan(v)
        latest = np.full(v.shape[0], np.nan)
        for i in range(v.shape[0]):
            idx = np.where(m[i])[0]
            if len(idx):
                latest[i] = v[i, idx[-1]]
        out[f"{stem}_latest@cutoff"] = latest
    return out


def build_long_dataset(
    df: pd.DataFrame,
    visit_columns: dict[str, list[str]],
    age_visit_cols: list[str],
    *,
    event_indicator: np.ndarray,
    survival_time: np.ndarray,
    bin_width_years: float = 1.0,
    max_horizon_years: float = 10.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Expand a wide-format cohort into patient-period long format.

    Returns:
        long_X (DataFrame): one row per (patient, period) with the patient's
            features at the start of that period plus a ``period`` column
            (mid-bin years from baseline).
        labels (np.ndarray): 0/1 per long-format row.
        patient_idx (np.ndarray): which patient each long row belongs to
            (positional index into ``df``).
    """
    n = len(df)
    age_v1 = df["Age_v1"].to_numpy(dtype=float)
    n_bins = int(np.ceil(max_horizon_years / bin_width_years))

    static_cols = ["gender", "T2DM", "Hypertension", "Dyslipidaemia",
                   "bariatric_surgery", "bariatric_surgery_age", "Age_v1"]
    static_cols = [c for c in static_cols if c in df.columns]
    static_block = df[static_cols].copy()

    rows_X: list[pd.DataFrame] = []
    rows_label: list[np.ndarray] = []
    rows_pid: list[np.ndarray] = []
    rows_period: list[np.ndarray] = []

    for k in range(n_bins):
        bin_start = k * bin_width_years
        bin_end = (k + 1) * bin_width_years

        # Each patient is in this bin iff they were still observed at bin_start.
        active = survival_time > bin_start
        if active.sum() == 0:
            continue

        # Bin label: event happened in (bin_start, bin_end] AND event_indicator true.
        in_bin = (survival_time > bin_start) & (survival_time <= bin_end) & event_indicator
        # Censored in this bin: survival_time in (bin_start, bin_end] and event False.
        censored_in_bin = (survival_time > bin_start) & (survival_time <= bin_end) & ~event_indicator

        # Active = those who reach bin_start. Patients censored *during* the bin
        # contribute a row labelled 0 (they did not have the event by end of bin
        # they actually observed); we keep only patients alive at bin_start.
        idx_pos = np.where(active)[0]

        cutoff_age = pd.Series(age_v1[idx_pos] + bin_start, index=df.index[idx_pos])
        latest = _last_known_value_at_age(
            df.iloc[idx_pos], visit_columns, age_visit_cols, cutoff_age,
        )
        period_age = age_v1[idx_pos] + bin_start  # patient's age at bin start
        block = static_block.iloc[idx_pos].copy()
        block["period_start_years"] = bin_start
        block["period_age"] = period_age
        block = pd.concat([block.reset_index(drop=True), latest.reset_index(drop=True)], axis=1)

        labels = np.zeros(len(idx_pos), dtype=int)
        labels[in_bin[idx_pos]] = 1

        # Patients censored mid-bin still contribute their bin row with label 0
        # (they survived past bin_start without an event before censoring).
        rows_X.append(block)
        rows_label.append(labels)
        rows_pid.append(idx_pos)
        rows_period.append(np.full(len(idx_pos), bin_start))

    long_X = pd.concat(rows_X, axis=0, ignore_index=True)
    labels = np.concatenate(rows_label)
    patient_idx = np.concatenate(rows_pid)
    return long_X, labels, patient_idx


# ---------------------------------------------------------------------------
# Patient-level risk from a fitted hazard model
# ---------------------------------------------------------------------------

def patient_cumulative_incidence(
    proba_per_row: np.ndarray, patient_idx: np.ndarray, n_patients: int
) -> np.ndarray:
    """``1 - prod(1 - p_t)`` per patient across their bins.

    Patients with no rows return 0.5 (no information).
    """
    out = np.full(n_patients, np.nan)
    surv = np.ones(n_patients)
    has = np.zeros(n_patients, dtype=bool)
    for p, pid in zip(proba_per_row, patient_idx):
        surv[pid] *= (1.0 - float(p))
        has[pid] = True
    cum = 1.0 - surv
    out[has] = cum[has]
    out[~has] = 0.5
    return out


# ---------------------------------------------------------------------------
# CV evaluation
# ---------------------------------------------------------------------------

@dataclass
class DTHResult:
    endpoint: str
    cv_oof: np.ndarray
    cv_test: np.ndarray
    fold_cindex: list[float]


def run_cv(endpoint_name: str, *, bin_width_years: float = 1.0, max_horizon_years: float = 10.0,
           lgbm_params: dict | None = None) -> DTHResult:
    """Run repeated stratified CV for the discrete-time hazard model."""
    import lightgbm as lgb

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    ep = hep if endpoint_name == "hepatic" else death

    # Folds shared with all other Phase work.
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)

    n = len(ds.train_df)
    oof = np.full(n, np.nan)
    test_pred_sum = np.zeros(len(ds.test_df))
    test_pred_n = 0
    fold_scores: list[float] = []

    train_long_X, train_long_y, train_long_pid = build_long_dataset(
        ds.train_df, ds.visit_columns, ds.age_visit_cols,
        event_indicator=ep.event.astype(bool),
        survival_time=ep.time.astype(float),
        bin_width_years=bin_width_years,
        max_horizon_years=max_horizon_years,
    )

    test_long_X, _, test_long_pid = build_long_dataset(
        ds.test_df, ds.visit_columns, ds.age_visit_cols,
        # Test doesn't have labels; we fake survival_time = inf so every patient
        # gets a row in every bin.
        event_indicator=np.zeros(len(ds.test_df), dtype=bool),
        survival_time=np.full(len(ds.test_df), np.inf),
        bin_width_years=bin_width_years,
        max_horizon_years=max_horizon_years,
    )

    train_long_X_pre = fill_for_tree(train_long_X)
    cols = list(train_long_X_pre.columns)
    test_long_X_pre = fill_for_tree(test_long_X).reindex(columns=cols, fill_value=0)

    lgbm_params = {
        "n_estimators": 600,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 30,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": 0,
        "verbose": -1,
        **(lgbm_params or {}),
    }

    for s in splits:
        tr_pids = set(s.train_idx.tolist())
        va_pids = set(s.valid_idx.tolist())
        tr_mask_long = np.isin(train_long_pid, list(tr_pids))
        va_mask_long = np.isin(train_long_pid, list(va_pids))
        if tr_mask_long.sum() < 100 or train_long_y[tr_mask_long].sum() == 0:
            continue
        n_pos = int(train_long_y[tr_mask_long].sum())
        n_neg = int((1 - train_long_y[tr_mask_long]).sum())
        spw = n_neg / max(n_pos, 1)

        clf = lgb.LGBMClassifier(**lgbm_params, scale_pos_weight=spw)
        clf.fit(train_long_X_pre.values[tr_mask_long], train_long_y[tr_mask_long])

        va_proba = clf.predict_proba(train_long_X_pre.values[va_mask_long])[:, 1]
        va_risk = patient_cumulative_incidence(
            va_proba, train_long_pid[va_mask_long], n_patients=n,
        )
        # Only score patients that are in this fold.
        va_idx = np.array(sorted(va_pids), dtype=int)
        risk_va = va_risk[va_idx]
        finite = np.isfinite(risk_va)
        if finite.sum() == 0 or ep.event[va_idx][finite].sum() == 0:
            continue
        sc = cindex(ep.event[va_idx][finite], ep.time[va_idx][finite], risk_va[finite]).cindex
        fold_scores.append(sc)
        oof[va_idx[finite]] = risk_va[finite]

        # Test predictions for this fold.
        te_proba = clf.predict_proba(test_long_X_pre.values)[:, 1]
        te_risk = patient_cumulative_incidence(te_proba, test_long_pid, n_patients=len(ds.test_df))
        test_pred_sum += te_risk
        test_pred_n += 1

    test_pred = test_pred_sum / test_pred_n if test_pred_n else np.full(len(ds.test_df), np.nan)
    return DTHResult(endpoint_name, oof, test_pred, fold_scores)


def main() -> None:
    out_root = cfg.EXPERIMENT_OUTPUTS / "phase3_discrete_time_hazard"
    out_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    pid_col = cfg.PATIENT_ID_COL
    tid_col = cfg.TRUSTII_ID_COL

    results = {}
    for ep in ("hepatic", "death"):
        _LOG.info("DTH CV for endpoint=%s", ep)
        r = run_cv(ep)
        cm = float(np.mean(r.fold_cindex)) if r.fold_cindex else float("nan")
        cs = float(np.std(r.fold_cindex)) if r.fold_cindex else float("nan")
        cmin = float(np.min(r.fold_cindex)) if r.fold_cindex else float("nan")
        _LOG.info("DTH %s: mean=%.4f std=%.4f min=%.4f", ep, cm, cs, cmin)
        results[ep] = {"mean": cm, "std": cs, "min": cmin, "fold_scores": r.fold_cindex}
        # Persist oof/test as one-column frames for downstream stacking.
        pd.DataFrame({pid_col: ds.train_df[pid_col].values, f"oof_dth_{ep}": r.cv_oof}) \
            .to_csv(out_root / f"oof_dth_{ep}.csv", index=False)
        pd.DataFrame({tid_col: ds.test_df[tid_col].values, f"test_dth_{ep}": r.cv_test}) \
            .to_csv(out_root / f"test_dth_{ep}.csv", index=False)

    import json as _json
    (out_root / "summary.json").write_text(_json.dumps(results, indent=2))
    _LOG.info("DTH summary written to %s", out_root / "summary.json")


if __name__ == "__main__":
    main()
