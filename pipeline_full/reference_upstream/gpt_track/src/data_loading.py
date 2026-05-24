"""Load the Trustii MASLD competition CSVs and align train/test schemas.

The raw data has:
- 22 visit slots per biomarker (suffix ``_v1`` .. ``_v22``).
- Train carries 4 endpoint columns (``death``, ``death_age_occur``,
  ``evenements_hepatiques_majeurs``, ``evenements_hepatiques_age_occur``)
  that are absent from test.
- Test carries an extra ``trustii_id`` column used for the submission.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config
from .utils import base_name, get_logger, visit_index

_LOG = get_logger(__name__)

_ENDPOINT_COLS = {
    config.HEPATIC_EVENT_COL,
    config.HEPATIC_EVENT_AGE_COL,
    config.DEATH_EVENT_COL,
    config.DEATH_EVENT_AGE_COL,
}


@dataclass
class Dataset:
    """Aligned train/test bundle.

    ``train_df`` contains feature columns plus the 4 endpoint columns.
    ``test_df`` contains the same feature columns plus ``trustii_id``.
    Both share the same feature column order (``feature_cols``) so downstream
    code never has to wonder which columns to use.
    """

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    dictionary: pd.DataFrame
    sample_submission: pd.DataFrame
    feature_cols: list[str] = field(default_factory=list)
    visit_columns: dict[str, list[str]] = field(default_factory=dict)
    age_visit_cols: list[str] = field(default_factory=list)
    static_cols: list[str] = field(default_factory=list)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        _LOG.warning("UTF-8 decode failed on %s; retrying with latin-1", path.name)
        return pd.read_csv(path, encoding="latin-1")


def discover_visit_columns(columns: list[str]) -> dict[str, list[str]]:
    """Group columns by biomarker stem.

    Returns a mapping ``stem -> [col_v1, col_v2, ...]`` sorted by visit index.
    Columns without a ``_v<int>`` suffix are skipped (they are static).
    """
    grouped: dict[str, list[tuple[int, str]]] = {}
    for c in columns:
        stem = base_name(c)
        idx = visit_index(c)
        if stem is None or idx is None:
            continue
        grouped.setdefault(stem, []).append((idx, c))
    return {stem: [c for _, c in sorted(pairs)] for stem, pairs in grouped.items()}


def load_dataset() -> Dataset:
    """Load raw CSVs, align schemas, and return a :class:`Dataset`.

    Logs every column that appears in only one of train/test, so we never
    silently drop data.
    """
    train_df = _read_csv(config.TRAIN_CSV)
    test_df = _read_csv(config.TEST_CSV)
    dictionary = _read_csv(config.DICT_CSV)
    sample_sub = _read_csv(config.SAMPLE_SUBMISSION_CSV)

    train_only = sorted(set(train_df.columns) - set(test_df.columns))
    test_only = sorted(set(test_df.columns) - set(train_df.columns))
    _LOG.info("train-only columns: %s", train_only)
    _LOG.info("test-only columns:  %s", test_only)

    expected_train_only = _ENDPOINT_COLS
    expected_test_only = {config.TRUSTII_ID_COL}
    unexpected = (set(train_only) - expected_train_only) | (
        set(test_only) - expected_test_only
    )
    if unexpected:
        _LOG.warning("unexpected schema mismatch (will be kept): %s", unexpected)

    # Feature columns = all train columns minus endpoints. Test must contain them all.
    feature_cols = [c for c in train_df.columns if c not in _ENDPOINT_COLS]
    missing_in_test = [c for c in feature_cols if c not in test_df.columns]
    if missing_in_test:
        raise RuntimeError(
            f"test missing {len(missing_in_test)} feature cols: {missing_in_test[:10]}"
        )

    visit_columns = discover_visit_columns(feature_cols)
    age_visit_cols = visit_columns.get("Age", [])
    if not age_visit_cols:
        raise RuntimeError("no Age_v* columns discovered; data layout changed?")

    visit_col_set = {c for cols in visit_columns.values() for c in cols}
    static_cols = [
        c for c in feature_cols
        if c not in visit_col_set and c != config.PATIENT_ID_COL
    ]

    n_visit_stems = len(visit_columns)
    _LOG.info(
        "loaded train=%s test=%s | %d visit stems | %d static cols",
        train_df.shape, test_df.shape, n_visit_stems, len(static_cols),
    )

    return Dataset(
        train_df=train_df,
        test_df=test_df,
        dictionary=dictionary,
        sample_submission=sample_sub,
        feature_cols=feature_cols,
        visit_columns=visit_columns,
        age_visit_cols=age_visit_cols,
        static_cols=static_cols,
    )


def last_observed_age(df: pd.DataFrame, age_cols: list[str]) -> pd.Series:
    """Return the maximum non-missing age across visit columns, per row."""
    return df[age_cols].max(axis=1, skipna=True)


def n_visits_observed(df: pd.DataFrame, age_cols: list[str]) -> pd.Series:
    """Count visits with a non-null Age (proxy for visit count)."""
    return df[age_cols].notna().sum(axis=1).astype(int)
