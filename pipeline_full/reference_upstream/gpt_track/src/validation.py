"""Repeated stratified patient-level cross-validation.

Folds are stratified primarily by hepatic event (the rare endpoint) to keep
event counts comparable across folds. Fold assignments are deterministic given
``RANDOM_SEED`` and are cached on disk so every experiment evaluates on the
same splits.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from . import config
from .utils import get_logger, write_json

_LOG = get_logger(__name__)


@dataclass
class FoldSplit:
    repeat: int
    fold: int
    train_idx: np.ndarray  # positional indices into the train DataFrame
    valid_idx: np.ndarray


def build_folds(
    train_df: pd.DataFrame,
    hepatic_event: np.ndarray,
    n_splits: int = config.N_SPLITS,
    n_repeats: int = config.N_REPEATS,
    seed: int = config.RANDOM_SEED,
) -> list[FoldSplit]:
    """Generate repeated stratified K-fold splits.

    Stratifies on the (rare) hepatic event so each fold has at least a few
    positives. ``hepatic_event`` is a 0/1 vector aligned with ``train_df``.
    """
    if len(hepatic_event) != len(train_df):
        raise ValueError("hepatic_event length must match train_df length")
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    splits: list[FoldSplit] = []
    for i, (tr, va) in enumerate(rskf.split(np.zeros(len(train_df)), hepatic_event)):
        rep = i // n_splits
        fold = i % n_splits
        splits.append(FoldSplit(rep, fold, tr.astype(int), va.astype(int)))
    _LOG.info("built %d folds (%d repeats x %d splits)",
              len(splits), n_repeats, n_splits)
    return splits


def save_folds(splits: list[FoldSplit], path: Path) -> None:
    """Persist fold assignments as a long-format CSV (one row per (repeat,fold,row))."""
    rows = []
    for s in splits:
        for idx in s.valid_idx:
            rows.append({"repeat": s.repeat, "fold": s.fold, "row_idx": int(idx)})
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    _LOG.info("saved fold assignments to %s", path)


def load_folds(path: Path, n_splits: int = config.N_SPLITS) -> list[FoldSplit]:
    """Load fold assignments saved by :func:`save_folds`."""
    df = pd.read_csv(path)
    out: list[FoldSplit] = []
    all_rows = sorted(df["row_idx"].unique())
    for (rep, fold), grp in df.groupby(["repeat", "fold"], sort=True):
        valid = np.array(sorted(grp["row_idx"].tolist()), dtype=int)
        train = np.array([i for i in all_rows if i not in set(valid.tolist())], dtype=int)
        out.append(FoldSplit(int(rep), int(fold), train, valid))
    return out


def fold_event_counts(splits: list[FoldSplit], event: np.ndarray) -> pd.DataFrame:
    """Return a DataFrame with per-fold event counts for sanity checks."""
    rows = []
    for s in splits:
        rows.append(
            {
                "repeat": s.repeat,
                "fold": s.fold,
                "n_train": int(len(s.train_idx)),
                "n_valid": int(len(s.valid_idx)),
                "events_train": int(event[s.train_idx].sum()),
                "events_valid": int(event[s.valid_idx].sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_cv(metrics: pd.DataFrame) -> dict:
    """Summarize a per-fold metrics DataFrame.

    Expected columns: ``cindex_hepatic``, ``cindex_death``, ``score``.
    Reports mean / std / min / max / penalised (mean - 0.5 * std).
    """
    out: dict[str, float | int] = {}
    for col in ("cindex_hepatic", "cindex_death", "score"):
        if col not in metrics.columns:
            out[f"{col}_mean"] = float("nan")
            out[f"{col}_std"] = float("nan")
            out[f"{col}_min"] = float("nan")
            out[f"{col}_max"] = float("nan")
            out[f"{col}_mean_minus_halfsd"] = float("nan")
            continue
        s = metrics[col].dropna()
        out[f"{col}_mean"] = float(s.mean()) if len(s) else float("nan")
        out[f"{col}_std"] = float(s.std()) if len(s) else float("nan")
        out[f"{col}_min"] = float(s.min()) if len(s) else float("nan")
        out[f"{col}_max"] = float(s.max()) if len(s) else float("nan")
        out[f"{col}_mean_minus_halfsd"] = (
            float(s.mean() - 0.5 * s.std()) if len(s) else float("nan")
        )
    out["n_folds"] = int(len(metrics))
    return out


def write_cv_summary_json(metrics: pd.DataFrame, path: Path) -> None:
    """Persist :func:`summarize_cv` output as JSON."""
    write_json(path, summarize_cv(metrics))
