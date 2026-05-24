"""Submission file generator.

Reads the sample submission to determine the required column order
(``trustii_id``, ``risk_hepatic_event``, ``risk_death``) and writes a CSV in
``submissions/<timestamp>_<modelname>.csv``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger, timestamp

_LOG = get_logger(__name__)


def make_submission(
    test_df: pd.DataFrame,
    risk_hepatic: np.ndarray,
    risk_death: np.ndarray,
    sample_submission: pd.DataFrame,
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    """Write a submission CSV. Returns the path on disk.

    Asserts ordering matches the sample submission's ``trustii_id`` order.
    """
    if config.TRUSTII_ID_COL not in test_df.columns:
        raise RuntimeError(f"test_df missing {config.TRUSTII_ID_COL}")
    if len(risk_hepatic) != len(test_df) or len(risk_death) != len(test_df):
        raise RuntimeError("risk vector length must match test_df")

    sub = pd.DataFrame(
        {
            config.TRUSTII_ID_COL: test_df[config.TRUSTII_ID_COL].values,
            config.SUB_HEPATIC_COL: risk_hepatic,
            config.SUB_DEATH_COL: risk_death,
        }
    )
    # Preserve sample order to be safe.
    sub = sub.set_index(config.TRUSTII_ID_COL).loc[sample_submission[config.TRUSTII_ID_COL].values].reset_index()

    out_dir = out_dir or config.SUBMISSIONS_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{timestamp()}_{model_name}.csv"
    sub.to_csv(path, index=False)
    _LOG.info("wrote submission %s", path)
    return path
