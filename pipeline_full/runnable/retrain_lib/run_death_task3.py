"""Run the death tree-survival models (CWGBSA / GBSA / EST) from RAW data.

This is a thin RUNNER around the vendored, unchanged
`upstream_scripts/task3_tree_survival.py`. It does not reimplement any model
arithmetic. It only redirects I/O so that:

  - model OUTPUTS are written to a fresh ``retrain_work`` predictions dir
    (NEVER into ``cached_intermediates/``), and
  - ``zoo_utils.load_anchors`` (used by task3 only for diagnostic rank-corr
    columns, never as a model input) returns the REGENERATED Track A/B anchors
    instead of cached anchor CSVs.

Raw labels/features come from ``ANNITIA_DATA_ROOT`` (set by the orchestrator)
via the unchanged ``claude_src.data.load_raw`` -> ``zoo_utils.load_labels``.

Usage:
  python run_death_task3.py --merged-root <dir> --gpt-anchor <csv> --cl-anchor <csv>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-root", required=True, type=Path,
                    help="fresh retrain_work/merged dir (predictions written here)")
    ap.add_argument("--gpt-anchor", required=True, type=Path)
    ap.add_argument("--cl-anchor", required=True, type=Path)
    ap.add_argument("--runnable", required=True, type=Path,
                    help="pipeline_full/runnable (for lib/ and upstream_scripts/)")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(args.runnable / "upstream_scripts"))
    sys.path.insert(0, str(args.runnable / "lib"))

    RM = args.merged_root.resolve()
    (RM / "model_zoo_sprint" / "predictions").mkdir(parents=True, exist_ok=True)
    (RM / "model_zoo_sprint" / "logs").mkdir(parents=True, exist_ok=True)

    # --- redirect zoo_utils I/O to the fresh retrain dir (no cached writes) ---
    import zoo_utils
    zoo_utils.ROOT = RM
    zoo_utils.ZOO = RM / "model_zoo_sprint"
    zoo_utils.PRED_DIR = RM / "model_zoo_sprint" / "predictions"

    # --- regenerated anchors for diagnostics only (NOT model inputs) ---
    gpt = pd.read_csv(args.gpt_anchor).sort_values("trustii_id").reset_index(drop=True)
    cl = pd.read_csv(args.cl_anchor).sort_values("trustii_id").reset_index(drop=True)
    if not (gpt["trustii_id"].to_numpy() == cl["trustii_id"].to_numpy()).all():
        raise SystemExit("FATAL: regenerated gpt/cl anchor trustii_id mismatch")
    merge = pd.DataFrame({
        "trustii_id": gpt["trustii_id"].to_numpy(),
        "risk_hepatic_event": 0.5 * rankdata(gpt["risk_hepatic_event"]) + 0.5 * rankdata(cl["risk_hepatic_event"]),
        "risk_death": 0.5 * rankdata(gpt["risk_death"]) + 0.5 * rankdata(cl["risk_death"]),
    })

    def _load_anchors_regen():
        return {"gpt_anchor_csv": gpt, "cl_anchor_csv": cl, "merge5050_csv": merge}

    zoo_utils.load_anchors = _load_anchors_regen

    import task3_tree_survival as t3
    # task3 bound these names at import; repoint to patched versions / fresh dirs
    t3.load_anchors = _load_anchors_regen
    t3.ROOT = RM
    t3.LOGS = RM / "model_zoo_sprint" / "logs"

    print(f"[run_death_task3] PRED_DIR = {zoo_utils.PRED_DIR}", flush=True)
    t3.main()

    # Fail loudly if the two slot1-relevant death files were not produced fresh.
    need = [
        "test__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv",
        "test__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv",
    ]
    for fn in need:
        p = zoo_utils.PRED_DIR / fn
        if not p.exists():
            raise SystemExit(f"FATAL: expected freshly-generated death file missing: {p}")
    print("[run_death_task3] OK: slot1-relevant CWGBSA/GBSA death files present.", flush=True)


if __name__ == "__main__":
    main()
