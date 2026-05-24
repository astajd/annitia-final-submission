"""Promote a previously run experiment's test predictions to a submission CSV.

Usage::

    python -m src.make_submission --experiment experiments/outputs/<id>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .submission import make_submission
from .utils import get_logger

_LOG = get_logger(__name__)


def from_experiment(exp_dir: Path) -> Path:
    exp_dir = Path(exp_dir)
    pred_path = exp_dir / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    preds = pd.read_csv(pred_path)
    if "ensemble_hepatic" not in preds or "ensemble_death" not in preds:
        raise RuntimeError("expected ensemble_hepatic / ensemble_death columns")

    ds = load_dataset()
    ordered = preds.set_index(cfg.TRUSTII_ID_COL).loc[ds.test_df[cfg.TRUSTII_ID_COL].values]
    out = make_submission(
        ds.test_df,
        risk_hepatic=ordered["ensemble_hepatic"].to_numpy(),
        risk_death=ordered["ensemble_death"].to_numpy(),
        sample_submission=ds.sample_submission,
        model_name=exp_dir.name,
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, type=Path)
    args = p.parse_args()
    out = from_experiment(args.experiment)
    print(out)


if __name__ == "__main__":
    main()
