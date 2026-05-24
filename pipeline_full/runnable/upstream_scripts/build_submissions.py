"""Build candidate submission CSVs by combining new components with anchors.

Each submission has:
  trustii_id, risk_hepatic_event, risk_death

Rank-transform once per source. No public-LB-tuned weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
RUNNABLE = HERE.parent
sys.path.insert(0, str(RUNNABLE / "lib"))
from zoo_utils import load_anchors  # noqa: E402

ROOT = RUNNABLE / "cached_intermediates"
PRED = ROOT / "model_zoo_sprint" / "predictions"
SUB = RUNNABLE / "_outputs" / "submissions"
SUB.mkdir(parents=True, exist_ok=True)


def load_test_pred(fname, col):
    return pd.read_csv(PRED / fname).sort_values("trustii_id")[col].to_numpy()


def main():
    A = load_anchors()
    tid = A["gpt_anchor_csv"]["trustii_id"].to_numpy()
    gpt_hep = A["gpt_anchor_csv"]["risk_hepatic_event"].to_numpy()
    gpt_dea = A["gpt_anchor_csv"]["risk_death"].to_numpy()
    cl_hep  = A["cl_anchor_csv"]["risk_hepatic_event"].to_numpy()
    cl_dea  = A["cl_anchor_csv"]["risk_death"].to_numpy()
    m50_hep = A["merge5050_csv"]["risk_hepatic_event"].to_numpy()
    m50_dea = A["merge5050_csv"]["risk_death"].to_numpy()

    # 1) gate_LSthr12 hep + GPT-Claude 50/50 death
    g12_hep = load_test_pred("test__gate_LSthr12__hep.csv", "hepatic_risk")
    sub1 = pd.DataFrame({
        "trustii_id": tid,
        "risk_hepatic_event": rankdata(g12_hep),
        "risk_death": m50_dea,
    })
    sub1.to_csv(SUB / "zoo_gateLSthr12_hep_with_m50_dea.csv", index=False)

    # 2) (1) but with GPT-only death (cleaner provenance)
    sub2 = pd.DataFrame({
        "trustii_id": tid,
        "risk_hepatic_event": rankdata(g12_hep),
        "risk_death": rankdata(gpt_dea),
    })
    sub2.to_csv(SUB / "zoo_gateLSthr12_hep_with_gpt_dea.csv", index=False)

    # 3) Hep blend: 0.85 * merge_50_50 hep + 0.15 * threshold__hep_h3_lgbm (most-diverse)
    th_lgbm = load_test_pred("test__threshold__hep_h3_lgbm.csv", "hepatic_risk")
    blend_hep = 0.85 * rankdata(m50_hep) + 0.15 * rankdata(th_lgbm)
    sub3 = pd.DataFrame({
        "trustii_id": tid,
        "risk_hepatic_event": rankdata(blend_hep),
        "risk_death": m50_dea,
    })
    sub3.to_csv(SUB / "zoo_m50_hep_plus15_threshold_lgbm.csv", index=False)

    # 4) Hep blend: 0.85 * merge_50_50 + 0.15 * threshold__hep_h3_catboost (different family)
    th_cb = load_test_pred("test__threshold__hep_h3_catboost.csv", "hepatic_risk")
    blend_hep2 = 0.85 * rankdata(m50_hep) + 0.15 * rankdata(th_cb)
    sub4 = pd.DataFrame({
        "trustii_id": tid,
        "risk_hepatic_event": rankdata(blend_hep2),
        "risk_death": m50_dea,
    })
    sub4.to_csv(SUB / "zoo_m50_hep_plus15_threshold_catboost.csv", index=False)

    # Compute correlations vs PUBLIC_BEST
    rows = []
    for f in [
        "zoo_gateLSthr12_hep_with_m50_dea.csv",
        "zoo_gateLSthr12_hep_with_gpt_dea.csv",
        "zoo_m50_hep_plus15_threshold_lgbm.csv",
        "zoo_m50_hep_plus15_threshold_catboost.csv",
    ]:
        s = pd.read_csv(SUB / f).sort_values("trustii_id")
        rh = float(spearmanr(s["risk_hepatic_event"], m50_hep).statistic)
        rd = float(spearmanr(s["risk_death"], m50_dea).statistic)
        rows.append(dict(submission=f, rho_hep_vs_m50=rh, rho_dea_vs_m50=rd))
        print(f"  {f}: rho_hep={rh:.3f}, rho_dea={rd:.3f}")
    pd.DataFrame(rows).to_csv(SUB / "submission_correlations.csv", index=False)
    print(f"\nSubmissions saved to {SUB}")


if __name__ == "__main__":
    main()
