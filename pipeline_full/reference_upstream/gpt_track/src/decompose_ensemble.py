"""Decompose an existing ensemble into its component models.

Reads OOF/test predictions from one or more experiment output directories,
computes pairwise rank correlations, and writes a markdown report so we can
see which components add diversity vs which are redundant.

Usage::

    python -m src.decompose_ensemble \
        --label ensemble_longitudinal \
        --experiments 20260426_1632_004_all_visits_longitudinal \
                      20260426_1636_006_strict_time_aligned \
                      20260426_1640_007_full_high_risk \
        --out reports/phase2_ensemble_longitudinal_decomposition.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .metrics import cindex
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger

_LOG = get_logger(__name__)


def _load_oof(exp_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Concatenate model-level OOF and test predictions across experiments.

    Returns (oof_df, test_df, meta) where columns are tagged ``<exp>::<model>``.
    """
    oof_pieces, test_pieces = [], []
    meta = {}
    for d in exp_dirs:
        d = Path(d)
        cfg_blob = json.loads((d / "config.json").read_text())
        ename = cfg_blob.get("name", d.name)
        oof = pd.read_csv(d / "oof_predictions.csv")
        tst = pd.read_csv(d / "test_predictions.csv")
        rename = {}
        for c in oof.columns:
            if c == cfg.PATIENT_ID_COL or c == cfg.TRUSTII_ID_COL:
                continue
            rename[c] = f"{ename}::{c}"
        oof = oof.rename(columns=rename)
        tst = tst.rename(columns={c: rename[c] for c in tst.columns if c in rename})
        oof_pieces.append(oof)
        test_pieces.append(tst)
        meta[ename] = {
            "feature_set": cfg_blob.get("feature_set"),
            "death_target_mode": cfg_blob.get("death_target_mode"),
            "models": [m.get("name") for m in cfg_blob.get("models", [])],
        }

    pid_col = cfg.PATIENT_ID_COL
    tid_col = cfg.TRUSTII_ID_COL
    oof_df = oof_pieces[0]
    for p in oof_pieces[1:]:
        oof_df = oof_df.merge(p, on=pid_col, how="outer")
    test_df = test_pieces[0]
    for p in test_pieces[1:]:
        test_df = test_df.merge(p, on=tid_col, how="outer")
    return oof_df, test_df, meta


def _per_model_cindex(oof_df: pd.DataFrame, ds, hep, death) -> pd.DataFrame:
    """Compute hepatic and death OOF C-index per (experiment::model) column."""
    rows = []
    pid_col = cfg.PATIENT_ID_COL
    train_pid_to_row = {pid: i for i, pid in enumerate(ds.train_df[pid_col].values)}
    pos = oof_df[pid_col].map(train_pid_to_row).to_numpy()
    keep = ~pd.isna(pos)
    pos = pos[keep].astype(int)
    oof_df = oof_df.loc[keep].reset_index(drop=True)

    for c in oof_df.columns:
        if c == pid_col:
            continue
        if "::ensemble_" in c:
            continue
        endpoint_tag = "hepatic" if "hepatic" in c.lower() else "death" if "death" in c.lower() else "unknown"
        col_vals = oof_df[c].to_numpy()
        if endpoint_tag == "hepatic":
            ev, t = hep.event[pos], hep.time[pos]
        elif endpoint_tag == "death":
            ev, t = death.event[pos], death.time[pos]
        else:
            continue
        finite = np.isfinite(col_vals)
        if finite.sum() < 50 or ev[finite].sum() == 0:
            rows.append({"component": c, "endpoint": endpoint_tag, "cindex": float("nan")})
            continue
        ci = cindex(ev[finite], t[finite], col_vals[finite]).cindex
        rows.append({"component": c, "endpoint": endpoint_tag, "cindex": float(ci)})
    return pd.DataFrame(rows)


def _correlation_matrix(oof_df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    cols = [c for c in oof_df.columns if endpoint in c.lower() and "ensemble_" not in c]
    if not cols:
        return pd.DataFrame()
    sub = oof_df[cols].rank(method="average", pct=True)
    return sub.corr(method="spearman")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    exp_dirs = []
    for e in args.experiments:
        candidate = cfg.EXPERIMENT_OUTPUTS / e
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        exp_dirs.append(candidate)

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)

    oof_df, test_df, meta = _load_oof(exp_dirs)
    per_model = _per_model_cindex(oof_df, ds, hep, death)

    corr_hep = _correlation_matrix(oof_df, "hepatic")
    corr_death = _correlation_matrix(oof_df, "death")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Decomposition of `{args.label}`\n")
    lines.append("## Component experiments\n")
    for ename, m in meta.items():
        lines.append(f"- **{ename}** | feature_set=`{m['feature_set']}` | death_mode=`{m['death_target_mode']}` | models=`{m['models']}`")
    lines.append("\n## Per-component OOF C-index\n")
    lines.append(per_model.sort_values(["endpoint", "cindex"], ascending=[True, False]).to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## Hepatic OOF rank correlations (Spearman)\n")
    lines.append(corr_hep.round(3).to_markdown() if not corr_hep.empty else "(no hepatic OOF columns)")
    lines.append("\n## Death OOF rank correlations (Spearman)\n")
    lines.append(corr_death.round(3).to_markdown() if not corr_death.empty else "(no death OOF columns)")

    lines.append("\n## Diversity / redundancy notes\n")
    lines.append(
        "- A component pair with rank correlation > 0.95 is essentially "
        "redundant — drop the one with the lower individual C-index.\n"
        "- A component with strong individual C-index (>= ensemble mean) but "
        "low pairwise correlation (< 0.85) with the rest is the highest-value "
        "addition.\n"
        "- A component whose C-index is below 0.55 should be reviewed "
        "separately even if it correlates lowly — it might be adding noise."
    )
    out.write_text("\n".join(lines))
    _LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
