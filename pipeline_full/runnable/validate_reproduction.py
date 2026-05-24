"""Validate that build_final_3.py reproduces the frozen slot1 file rank-identically.

Pass criteria:
  - Schema matches: columns == ['trustii_id', 'risk_hepatic_event', 'risk_death']
  - Row count == 423
  - No NaNs in any column
  - trustii_id sets and order alignment match between regenerated and frozen
  - For each risk column, rankdata(regenerated) == rankdata(frozen) elementwise
    after aligning by trustii_id (this is the canonical rank-identity check;
    the C-index depends only on ranks).

Float-exact equality is reported as informational only — rank-identity is the
pass criterion (see PROVENANCE_FINDINGS.md and audit/ for why).

Run after `python build_final_3.py`. Expects:
  _outputs/submissions/finalprobe_3_lsthr12_disagq95_cw85gbsa15.csv
  ../../frozen/slot1_prediction.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REGEN = HERE / "_outputs" / "submissions" / "finalprobe_3_lsthr12_disagq95_cw85gbsa15.csv"
FROZEN = HERE.parents[1] / "frozen" / "slot1_prediction.csv"

EXPECTED_COLS = ["trustii_id", "risk_hepatic_event", "risk_death"]
EXPECTED_ROWS = 423


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def info(msg: str) -> None:
    print(f"INFO: {msg}")


def main() -> None:
    if not REGEN.exists():
        fail(f"regenerated file not found: {REGEN}\nRun `python build_final_3.py` first.")
    if not FROZEN.exists():
        fail(f"frozen file not found: {FROZEN}")

    regen = pd.read_csv(REGEN)
    frozen = pd.read_csv(FROZEN)

    if list(regen.columns) != EXPECTED_COLS:
        fail(f"regenerated columns mismatch: {list(regen.columns)} vs {EXPECTED_COLS}")
    if list(frozen.columns) != EXPECTED_COLS:
        fail(f"frozen columns mismatch: {list(frozen.columns)} vs {EXPECTED_COLS}")
    ok(f"schema columns: {EXPECTED_COLS}")

    if len(regen) != EXPECTED_ROWS:
        fail(f"regenerated row count: {len(regen)} != {EXPECTED_ROWS}")
    if len(frozen) != EXPECTED_ROWS:
        fail(f"frozen row count: {len(frozen)} != {EXPECTED_ROWS}")
    ok(f"row count: {EXPECTED_ROWS}")

    if regen.isna().any().any():
        fail(f"NaNs found in regenerated: {regen.isna().sum().to_dict()}")
    if frozen.isna().any().any():
        fail(f"NaNs found in frozen: {frozen.isna().sum().to_dict()}")
    ok("no NaNs in either file")

    if set(regen["trustii_id"]) != set(frozen["trustii_id"]):
        fail("trustii_id sets differ between regenerated and frozen")
    ok("trustii_id sets match")

    regen_sorted = regen.sort_values("trustii_id").reset_index(drop=True)
    frozen_sorted = frozen.sort_values("trustii_id").reset_index(drop=True)
    if not (regen_sorted["trustii_id"].to_numpy() == frozen_sorted["trustii_id"].to_numpy()).all():
        fail("trustii_id alignment failed after sort")
    ok("trustii_id alignment confirmed")

    rank_pass = True
    for col in ("risk_hepatic_event", "risk_death"):
        rr = rankdata(regen_sorted[col].to_numpy())
        rf = rankdata(frozen_sorted[col].to_numpy())
        if not np.array_equal(rr, rf):
            n_diff = int((rr != rf).sum())
            fail(f"RANK MISMATCH in {col}: {n_diff} positions differ")
            rank_pass = False
        else:
            ok(f"rank-identity confirmed for {col}")

    for col in ("risk_hepatic_event", "risk_death"):
        rv = regen_sorted[col].to_numpy()
        fv = frozen_sorted[col].to_numpy()
        if np.array_equal(rv, fv):
            info(f"{col}: float-exact equality holds")
        else:
            diff = np.abs(rv - fv)
            info(f"{col}: float values differ (max |Δ|={diff.max():.6g}, mean |Δ|={diff.mean():.6g}); rank-identical per check above")

    if rank_pass:
        print("\nSUCCESS — regenerated slot1 is RANK-IDENTICAL to frozen/slot1_prediction.csv")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
