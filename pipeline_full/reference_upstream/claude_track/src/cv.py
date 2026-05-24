"""Cross-validation: repeated stratified K-fold by event indicator."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored
from .config import CV_N_SPLITS, CV_N_REPEATS, CV_RANDOM_STATE


def cindex(event, time, risk) -> float:
    e = np.asarray(event, dtype=bool)
    t = np.asarray(time, dtype=float)
    r = np.asarray(risk, dtype=float)
    if np.isnan(r).any():
        r = np.where(np.isnan(r), np.nanmedian(r), r)
    return float(concordance_index_censored(e, t, r)[0])


def repeated_stratified_folds(event, n_splits=CV_N_SPLITS,
                              n_repeats=CV_N_REPEATS, base_seed=CV_RANDOM_STATE):
    e = np.asarray(event).astype(int)
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=base_seed + r)
        for f, (tr, va) in enumerate(skf.split(np.zeros(len(e)), e)):
            yield r, f, tr, va


def evaluate_cv(fit_predict_fn, X, event, time, *,
                n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS,
                base_seed=CV_RANDOM_STATE, verbose=False) -> pd.DataFrame:
    rows = []
    for r, f, tr, va in repeated_stratified_folds(event, n_splits, n_repeats, base_seed):
        risk_va = fit_predict_fn(X.iloc[tr], event[tr], time[tr], X.iloc[va])
        ci = cindex(event[va], time[va], risk_va)
        rows.append({"repeat": r, "fold": f,
                    "n_train_events": int(event[tr].sum()),
                    "n_valid_events": int(event[va].sum()),
                    "cindex": ci})
        if verbose:
            print(f"  rep={r} fold={f}  n_va_events={int(event[va].sum())}  C={ci:.4f}")
    return pd.DataFrame(rows)


def summarize(cv_results: pd.DataFrame) -> dict:
    ci = cv_results["cindex"].to_numpy()
    return dict(
        mean=float(np.mean(ci)),
        std=float(np.std(ci, ddof=1)) if len(ci) > 1 else 0.0,
        median=float(np.median(ci)),
        min=float(np.min(ci)),
        max=float(np.max(ci)),
        n_folds=int(len(ci)),
    )


def weighted_score(c_hep, c_death, w_hep=0.7, w_death=0.3) -> float:
    return w_hep * c_hep + w_death * c_death
