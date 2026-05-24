"""Reproduce ONLY the submitted slot1 file (finalprobe_3_lsthr12_disagq95_cw85gbsa15)
from cached prediction-level intermediates.

This is the reviewer-facing AUTHORITATIVE verification script. Unlike the
historical multi-candidate producer `build_final_3.py`, this script:

  - does NOT load raw challenge data (no load_labels(), no load_raw())
  - does NOT read ANNITIA_DATA_ROOT
  - does NOT compute the LS=12 gate from any liver-stiffness column
    (the gate output is READ from the cached prediction CSV
    `cached_intermediates/model_zoo_sprint/predictions/test__gate_LSthr12__hep.csv`)
  - does NOT compute finalprobe_1 / finalprobe_2 (LS=18 candidates)
  - does NOT retrain any model
  - applies exactly the same q95 disagreement-override and 0.85/0.15 death-blend
    formulae as the slot1 branch of `build_final_3.py`

It writes a single file: `generated_slot1_prediction.csv` (alongside this
script). Verify rank-identity to `../../frozen/slot1_prediction.csv` with
`validate_slot1_only.py`.

The submitted file is authoritative; this script is a verification of the
deterministic recipe. If the regenerated file is NOT rank-identical to the
frozen file, STOP and report the discrepancy — do NOT modify the script,
the cached intermediates, or the output to force a match.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
CACHED = HERE / "cached_intermediates"
ZOO_PRED = CACHED / "model_zoo_sprint" / "predictions"
OUT_CSV = HERE / "generated_slot1_prediction.csv"

GPT_ANCHOR_CSV = CACHED / "gpt_track_handoff/best_submissions/20260428_0059_phase3_10_horizon_blend_v2.csv"
CL_ANCHOR_CSV = CACHED / "claude_track_handoff/best_submissions/phase2_blend_2way_optimal.csv"
MERGE5050_CSV = CACHED / "merge_sprint/submissions/merge_A_best_50_50_both.csv"
GATE_LS12_TEST_CSV = ZOO_PRED / "test__gate_LSthr12__hep.csv"
CWGBS_TEST_CSV = ZOO_PRED / "test__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv"
GBSA_TEST_CSV = ZOO_PRED / "test__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv"


def _load_test(path: Path, col: str, tid: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path)
    return df.set_index("trustii_id").reindex(tid)[col].to_numpy()


def disag_override(base_h, gpt_h, cl_h, m50_h, q):
    """Where |rank(GPT) - rank(Cl)| > q-quantile, replace with rank(m50); else rank(base_h)."""
    g_r = rankdata(gpt_h); c_r = rankdata(cl_h)
    d = np.abs(g_r - c_r) / max(g_r.size, 1)
    tau = float(np.quantile(d, q))
    return np.where(d > tau, rankdata(m50_h), rankdata(base_h)), tau, int((d > tau).sum())


def main():
    gpt_anchor = pd.read_csv(GPT_ANCHOR_CSV).sort_values("trustii_id").reset_index(drop=True)
    cl_anchor = pd.read_csv(CL_ANCHOR_CSV).sort_values("trustii_id").reset_index(drop=True)
    merge_5050 = pd.read_csv(MERGE5050_CSV).sort_values("trustii_id").reset_index(drop=True)

    tid = gpt_anchor["trustii_id"].to_numpy()
    if not (cl_anchor["trustii_id"].to_numpy() == tid).all():
        raise SystemExit("trustii_id mismatch: claude anchor vs gpt anchor")
    if not (merge_5050["trustii_id"].to_numpy() == tid).all():
        raise SystemExit("trustii_id mismatch: merge_50_50 vs gpt anchor")

    gpt_hep_te = gpt_anchor["risk_hepatic_event"].to_numpy()
    cl_hep_te = cl_anchor["risk_hepatic_event"].to_numpy()
    m50_hep_te = merge_5050["risk_hepatic_event"].to_numpy()

    g12_hep_te = _load_test(GATE_LS12_TEST_CSV, "hepatic_risk", tid)
    cwgbs_te = _load_test(CWGBS_TEST_CSV, "death_risk", tid)
    gbsa_te = _load_test(GBSA_TEST_CSV, "death_risk", tid)

    for name, arr in [("g12_hep_te", g12_hep_te), ("cwgbs_te", cwgbs_te), ("gbsa_te", gbsa_te)]:
        if np.isnan(arr).any():
            raise SystemExit(f"unexpected NaN in cached test array: {name}")

    p3_h_te, tau_te_p3, n_over_te_p3 = disag_override(
        g12_hep_te, gpt_hep_te, cl_hep_te, m50_hep_te, 0.95)

    dea_blend_te = 0.85 * rankdata(cwgbs_te) + 0.15 * rankdata(gbsa_te)

    out = pd.DataFrame({
        "trustii_id": tid,
        "risk_hepatic_event": rankdata(p3_h_te),
        "risk_death": rankdata(dea_blend_te),
    })

    if len(out) != 423:
        raise SystemExit(f"unexpected row count: {len(out)} (expected 423)")
    if out.isna().any().any():
        raise SystemExit("unexpected NaN in generated output")

    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    print(f"  rows={len(out)}  cols={list(out.columns)}")
    print(f"  disag q95 tau (test side) = {tau_te_p3:.6f}")
    print(f"  disag q95 n_overridden    = {n_over_te_p3}")


if __name__ == "__main__":
    main()
