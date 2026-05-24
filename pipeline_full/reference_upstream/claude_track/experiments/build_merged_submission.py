"""Build a 50/50 rank-averaged merged submission across the two tracks.

Inputs:
  - Anthropic-track champion:  submissions/phase2_blend_2way_optimal.csv (LB 0.8965)
  - GPT-track champion:        ../gpt/submissions/20260428_0059_phase3_10_horizon_blend_v2.csv (LB 0.91093)

Output:
  - submissions/merged_50_50_claude_gpt.csv  (trustii_id, risk_hepatic_event, risk_death)

The two CSVs use different risk-score scales (RSF cumulative-hazard vs scaled
log-hazard). C-index depends only on ordering, so we rank-transform each per
endpoint and then average.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(str(Path(__file__).resolve().parents[1]))
CLAUDE_CSV = ROOT / "submissions" / "phase2_blend_2way_optimal.csv"
GPT_CSV = Path(str(Path(__file__).resolve().parents[2] / "gpt_track/submissions/20260428_0059_phase3_10_horizon_blend_v2.csv")
OUT_CSV = ROOT / "submissions" / "merged_50_50_claude_gpt.csv"


def main() -> None:
    a = pd.read_csv(CLAUDE_CSV).sort_values("trustii_id").reset_index(drop=True)
    b = pd.read_csv(GPT_CSV).sort_values("trustii_id").reset_index(drop=True)
    assert (a["trustii_id"].to_numpy() == b["trustii_id"].to_numpy()).all(), "trustii_id mismatch"

    rho_h = spearmanr(a["risk_hepatic_event"], b["risk_hepatic_event"]).statistic
    rho_d = spearmanr(a["risk_death"], b["risk_death"]).statistic
    print(f"Spearman across tracks: hepatic = {rho_h:.4f}, death = {rho_d:.4f}")

    # Rank-average per endpoint
    hep = (rankdata(a["risk_hepatic_event"]) + rankdata(b["risk_hepatic_event"])) / 2.0
    dea = (rankdata(a["risk_death"]) + rankdata(b["risk_death"])) / 2.0

    out = pd.DataFrame({
        "trustii_id": a["trustii_id"],
        "risk_hepatic_event": hep,
        "risk_death": dea,
    })
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} ({len(out)} rows)")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
