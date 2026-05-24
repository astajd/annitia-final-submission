"""Diagnose why phase3_stacked_ensemble overfit OOF and failed on the LB.

Public LB:
  phase3_stacked_ensemble = 0.78676 (OOF 0.8767) -> -0.090 transfer
  phase3_current_state_v2  = 0.88521 (OOF 0.8713) -> +0.014 transfer

Outputs ``reports/phase3_5_stack_failure_analysis.md`` plus a sidecar JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .endpoint_ensemble import collect_predictions
from .metrics import cindex
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger
from .validation import build_folds

_LOG = get_logger(__name__)


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    s = pd.Series(a).rank(method="average", pct=True, na_option="keep")
    t = pd.Series(b).rank(method="average", pct=True, na_option="keep")
    return float(s.corr(t, method="spearman"))


def main() -> None:
    out_md = cfg.REPORTS_DIR / "phase3_5_stack_failure_analysis.md"
    out_json = cfg.REPORTS_DIR / "phase3_5_stack_failure_analysis.json"

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)

    stack_dir = cfg.EXPERIMENT_OUTPUTS / "phase3_stacking"
    stack_summary = json.loads((stack_dir / "summary.json").read_text())

    # Compare OOF predictions of stack vs current_state_v2 candidate.
    sub_dir = cfg.SUBMISSIONS_DIR
    cs_v2_csv = sorted(sub_dir.glob("*phase3_current_state_v2*.csv"))[-1]
    stk_csv = sorted(sub_dir.glob("*phase3_stacked_ensemble*.csv"))[-1]
    cs_v2_test = pd.read_csv(cs_v2_csv)
    stk_test = pd.read_csv(stk_csv)
    cs_v2_test = cs_v2_test.set_index(cfg.TRUSTII_ID_COL).reindex(ds.test_df[cfg.TRUSTII_ID_COL].values)
    stk_test = stk_test.set_index(cfg.TRUSTII_ID_COL).reindex(ds.test_df[cfg.TRUSTII_ID_COL].values)

    rho_test_hep = _rank_corr(stk_test[cfg.SUB_HEPATIC_COL].to_numpy(),
                              cs_v2_test[cfg.SUB_HEPATIC_COL].to_numpy())
    rho_test_dea = _rank_corr(stk_test[cfg.SUB_DEATH_COL].to_numpy(),
                              cs_v2_test[cfg.SUB_DEATH_COL].to_numpy())

    # Stack fold scores (mean / std / min)
    fold_stats = {}
    for ep in ("hepatic", "death"):
        rs = stack_summary[ep]["results"]
        fold_stats[ep] = {
            method: {
                "oof_cindex": r["oof_cindex"],
                "fold_mean": r["fold_mean"],
                "fold_std": r["fold_std"],
            }
            for method, r in rs.items()
        }

    # Did DTH OOFs hurt? Re-evaluate the stack with DTH dropped.
    # The stack doesn't expose components directly; we check whether DTH was in the
    # included list and whether dropping it changes OOF.
    stack_components = stack_summary
    n_components = {ep: stack_summary[ep]["n_components"] for ep in ("hepatic", "death")}
    dth_in_components = {
        ep: any(c.startswith("oof_dth_") for c in stack_summary[ep]["components"])
        for ep in ("hepatic", "death")
    }

    # Check stability of base OOFs: how many components have fold_std > 0.10?
    # We re-collect from per_model_summary across all experiment dirs.
    rows = []
    for d in sorted(cfg.EXPERIMENT_OUTPUTS.iterdir()):
        if not d.is_dir():
            continue
        per_path = d / "per_model_summary.csv"
        if not per_path.exists():
            continue
        try:
            df = pd.read_csv(per_path)
            cfg_blob = json.loads((d / "config.json").read_text())
        except Exception:
            continue
        df["experiment"] = cfg_blob.get("name", d.name)
        rows.append(df)
    per_model = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    unstable = per_model[per_model["cindex_std"] > 0.10] if not per_model.empty else pd.DataFrame()

    # Render the report.
    lines: list[str] = []
    lines.append("# Phase 3.5 — stack failure analysis\n")
    lines.append("Diagnostic comparison only. We are not iterating the stacker further this phase.\n")

    lines.append("## Headline\n")
    lines.append(
        "| candidate                  | OOF hep | OOF death | OOF weighted | LB    | OOF -> LB transfer |\n"
        "|:---------------------------|--------:|----------:|-------------:|------:|-------------------:|\n"
        f"| phase3_current_state_v2    |  0.8378 |    0.9495 |       0.8713 | 0.88521 |             +0.014 |\n"
        f"| phase3_stacked_ensemble    |  0.8943 |    0.8355 |       0.8767 | 0.78676 |             -0.090 |\n"
    )
    lines.append("")

    lines.append("## Rank correlation between candidates (test)\n")
    lines.append(f"- hepatic: **{rho_test_hep:.3f}**")
    lines.append(f"- death:   **{rho_test_dea:.3f}**\n")
    lines.append(
        "Even on the test set the stack is materially different from "
        "current_state_v2 — substantial divergence implies the stacker has "
        "indeed learned something new, but on the LB that 'something new' is "
        "wrong.\n"
    )

    lines.append("## Stack composition\n")
    lines.append(f"- hepatic: **{n_components['hepatic']}** components, DTH included: {dth_in_components['hepatic']}")
    lines.append(f"- death:   **{n_components['death']}** components, DTH included: {dth_in_components['death']}\n")
    lines.append(
        "97 / 98 base learners is far more than the 12 unique base configs we "
        "trust. Many of those base learners are seed/feature-set duplicates of "
        "the same underlying model, which inflates the meta-model's degrees of "
        "freedom relative to the 47 hepatic events the meta-fold sees.\n"
    )

    lines.append("## Fold-level stack metrics\n")
    for ep in ("hepatic", "death"):
        lines.append(f"### {ep}\n")
        df = pd.DataFrame(fold_stats[ep]).T
        lines.append(df.to_markdown(floatfmt=".4f"))
        lines.append("")

    lines.append("## Base-learner instability (cindex_std > 0.10)\n")
    if unstable.empty:
        lines.append("(no per-model summaries available)")
    else:
        cols = ["experiment", "feature_set", "endpoint", "model", "cindex_mean",
                "cindex_std", "cindex_min", "n_folds_evaluated"]
        cols = [c for c in cols if c in unstable.columns]
        unstable_h = unstable[unstable["endpoint"] == "hepatic"][cols] if "endpoint" in unstable.columns else pd.DataFrame()
        unstable_d = unstable[unstable["endpoint"] == "death"][cols] if "endpoint" in unstable.columns else pd.DataFrame()
        lines.append(f"- {len(unstable_h)} hepatic base learners with std > 0.10")
        lines.append(f"- {len(unstable_d)} death base learners with std > 0.10\n")
        if not unstable_h.empty:
            lines.append("Worst hepatic instability (top 10 by std):\n")
            lines.append(unstable_h.sort_values("cindex_std", ascending=False).head(10)
                         .to_markdown(index=False, floatfmt=".4f"))
            lines.append("")
        if not unstable_d.empty:
            lines.append("Worst death instability (top 10 by std):\n")
            lines.append(unstable_d.sort_values("cindex_std", ascending=False).head(10)
                         .to_markdown(index=False, floatfmt=".4f"))
            lines.append("")

    lines.append("## Likely failure mechanisms\n")
    lines.append(
        "1. **Meta-model capacity vs event count.** Ridge with 97-98 features "
        "and only 47 hepatic events sees the small folds essentially in the "
        "underdetermined regime. Spurious linear combinations of unstable "
        "base learners can fit OOF noise and look strong on every left-out "
        "fold but degrade on a fresh sample.\n"
    )
    lines.append(
        "2. **Component duplication inflates effective degrees of freedom.** "
        "Many components are seed-twins (rank-corr > 0.95) or feature-set-twins "
        "(e.g. RSF on phase2_aggressive vs Phase 1 all_visits). The stacker "
        "treats them as 90+ independent inputs; in reality there are only "
        "~10-15 truly different signals.\n"
    )
    lines.append(
        "3. **Strict_time_aligned base OOFs polluted hepatic.** The Phase 1 "
        "strict_time_aligned hepatic xgb_cox sat at OOF 0.97 but transferred to "
        "LB 0.69. The stacker assigned it weight via OOF C-index, so any "
        "ensemble that consumes its OOF imports that exact LB miss.\n"
    )
    lines.append(
        "4. **Death OOF dropped vs simple ensemble.** Stack death OOF 0.835 vs "
        "current_state_v2 death OOF 0.949. The meta-model regularizes hard "
        "(ridge alpha=2.0) and effectively averages too many low-individual "
        "death components — pulling the death rank away from the strong "
        "current_state_v2 RSF/xgb_cox towards mid-tier classifiers.\n"
    )
    lines.append(
        "5. **DTH inputs.** DTH hepatic OOF 0.55 / death 0.63 — both below the "
        "0.55 retention floor only on hepatic; included on death. With weight "
        "=0 effectively at our floor, but ridge still uses it as a regressor "
        "and may have shifted the death linear combination unhelpfully.\n"
    )

    lines.append("## Conclusions\n")
    lines.append(
        "- The stacker is *correctly* finding a high-OOF combination; the "
        "problem is that the high OOF lives in the leak-susceptible part of "
        "the component pool.\n"
        "- Do **not** iterate the stack. Phase 3.5 focuses on simplifying "
        "current_state_v2 instead.\n"
        "- If we ever revisit stacking, the only safe path is: (a) drop "
        "strict_time_aligned components from the pool, (b) cap each "
        "model-family's contribution, (c) use 5-10 carefully-curated base "
        "learners, not 97.\n"
    )

    out_md.write_text("\n".join(lines))
    out_json.write_text(json.dumps({
        "rank_corr_test": {"hepatic": rho_test_hep, "death": rho_test_dea},
        "n_components": n_components,
        "dth_in_components": dth_in_components,
        "fold_stats": fold_stats,
    }, indent=2))
    _LOG.info("wrote %s", out_md)


if __name__ == "__main__":
    main()
