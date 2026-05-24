"""Leakage audit: produce ``reports/leakage_audit.md``.

Quantifies follow-up-related signal that could leak event status onto the
training features. Single-feature C-index baselines tell us what an absurdly
simple model can already extract — if any of these scores is unexpectedly high,
real models built on top of those features are at risk of memorising the
follow-up window rather than the disease biology.

Run as ``python -m src.leakage_audit``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data_loading import load_dataset, last_observed_age, n_visits_observed
from .metrics import cindex
from .targets import build_death_endpoint, build_hepatic_endpoint, followup_years
from .utils import get_logger

_LOG = get_logger(__name__)


def _post_event_visits(df: pd.DataFrame, age_cols: list[str], event_age: pd.Series) -> pd.Series:
    """Per-row count of visits with Age strictly greater than event age."""
    a = df[age_cols].to_numpy(dtype=float)
    cutoff = event_age.to_numpy(dtype=float)[:, None]
    with np.errstate(invalid="ignore"):
        return pd.Series(np.nansum(a > cutoff, axis=1).astype(int), index=df.index)


def _summary_stats(s: pd.Series) -> dict:
    s = s.dropna()
    return {
        "n": int(len(s)),
        "mean": float(s.mean()) if len(s) else float("nan"),
        "median": float(s.median()) if len(s) else float("nan"),
        "min": float(s.min()) if len(s) else float("nan"),
        "max": float(s.max()) if len(s) else float("nan"),
    }


def run_audit(out_path: Path | None = None) -> Path:
    config.ensure_dirs()
    out_path = out_path or (config.REPORTS_DIR / "leakage_audit.md")

    ds = load_dataset()
    tr = ds.train_df
    te = ds.test_df
    age_cols = ds.age_visit_cols

    n_v_tr = n_visits_observed(tr, age_cols)
    n_v_te = n_visits_observed(te, age_cols)

    last_age_tr = last_observed_age(tr, age_cols)
    last_age_te = last_observed_age(te, age_cols)

    fu_tr = followup_years(tr, age_cols)
    fu_te = last_age_te - te["Age_v1"]

    hep = build_hepatic_endpoint(tr, age_cols)
    death_keep = build_death_endpoint(tr, age_cols, mode="censor_missing_death_at_last_visit")

    hep_event_age = tr[config.HEPATIC_EVENT_AGE_COL]
    death_event_age = tr[config.DEATH_EVENT_AGE_COL]

    post_hep = _post_event_visits(tr, age_cols, hep_event_age).where(tr[config.HEPATIC_EVENT_COL] == 1)
    post_death = _post_event_visits(tr, age_cols, death_event_age).where(tr[config.DEATH_EVENT_COL] == 1)

    death_missing_mask = tr[config.DEATH_EVENT_COL].isna()
    miss_profile = {
        "n": int(death_missing_mask.sum()),
        "median_visits": float(n_v_tr[death_missing_mask].median()),
        "median_followup_years": float(fu_tr[death_missing_mask].median()),
        "hepatic_event_rate": float(tr.loc[death_missing_mask, config.HEPATIC_EVENT_COL].mean()),
    }

    # Single-feature C-index benchmarks. We test both endpoints so we can see
    # which target is leaked by which feature.
    age_v1 = tr["Age_v1"].to_numpy()
    n_visits_arr = n_v_tr.to_numpy()
    last_age_arr = last_age_tr.to_numpy()
    fu_arr = fu_tr.to_numpy()
    miss_total = tr[[c for cols in ds.visit_columns.values() for c in cols]].isna().mean(axis=1).to_numpy()
    miss_by_visit = (
        tr[[c for c in tr.columns if c.endswith(("_v1", "_v2", "_v3"))]].isna().mean(axis=1).to_numpy()
    )

    benchmarks: list[dict] = []
    for fname, x in [
        ("Age_v1", age_v1),
        ("n_visits", n_visits_arr),
        ("last_observed_age", last_age_arr),
        ("followup_years", fu_arr),
        ("total_missingness", miss_total),
        ("missingness_v1_v3", miss_by_visit),
        ("neg_followup_years", -fu_arr),
        ("neg_n_visits", -n_visits_arr),
    ]:
        ch = cindex(hep.event, hep.time, x).cindex
        cd = cindex(death_keep.event, death_keep.time, x).cindex
        benchmarks.append({"feature": fname, "cindex_hepatic": ch, "cindex_death": cd})
    bench_df = pd.DataFrame(benchmarks)

    # Render the markdown report.
    lines: list[str] = []
    lines.append("# Leakage audit\n")
    lines.append(
        "Diagnostics that quantify how much of the survival signal is coming "
        "from follow-up bookkeeping (visit count, follow-up duration, missingness) "
        "rather than disease biology. Run from `python -m src.leakage_audit`.\n"
    )

    lines.append("## Visits per patient\n")
    lines.append(f"- train: {_summary_stats(n_v_tr)}")
    lines.append(f"- test:  {_summary_stats(n_v_te)}\n")

    lines.append("## Last observed age (years)\n")
    lines.append(f"- train: {_summary_stats(last_age_tr)}")
    lines.append(f"- test:  {_summary_stats(last_age_te)}\n")

    lines.append("## Follow-up years (last_age - Age_v1)\n")
    lines.append(f"- train: {_summary_stats(fu_tr)}")
    lines.append(f"- test:  {_summary_stats(fu_te)}\n")

    lines.append("## Visits *after* the recorded event age\n")
    lines.append(
        f"- hepatic event patients with at least one post-event visit: "
        f"{int((post_hep > 0).sum())} of {int(tr[config.HEPATIC_EVENT_COL].sum())}"
    )
    lines.append(
        f"- death patients with at least one post-death visit: "
        f"{int((post_death > 0).sum())} of {int(tr[config.DEATH_EVENT_COL].fillna(0).sum())}\n"
    )
    lines.append(
        "  These rows are why the **strict_time_aligned** feature set masks "
        "post-event visits during training.\n"
    )

    lines.append("## Missing-death subgroup profile\n")
    for k, v in miss_profile.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Single-feature C-index benchmarks\n")
    lines.append(bench_df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## Interpretation\n")
    lines.append(
        "- A feature like `Age_v1` reaching ~0.65 on the death endpoint is "
        "expected (older patients die sooner) and is *not* leakage — it is "
        "biology and is fair game for the model.\n"
    )
    lines.append(
        "- However, `n_visits` and `followup_years` carry the survivor's "
        "footprint: someone who is still alive accumulates more visits. A "
        "C-index well above 0.5 for `neg_followup_years` (i.e. *fewer* "
        "follow-up years -> higher risk) is a signature of follow-up leakage. "
        "Track this number across the contest; in the synthetic Trustii data "
        "follow-up is partly aligned with outcomes by construction.\n"
    )
    lines.append(
        "- Total missingness behaves similarly to follow-up. The high-leakage "
        "feature sets (`all_visits_longitudinal`, `full_high_risk`, "
        "`missingness_and_visit_cadence`) inherit this.\n"
    )
    lines.append(
        "- Patients with missing `death` status look very similar to censored "
        "patients (median follow-up close to the censored cohort, low hepatic "
        "event rate). Treating them as censored at the last visit is the "
        "default; a sensitivity comparison is run by experiments 002/003.\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    _LOG.info("wrote leakage audit to %s", out_path)
    return out_path


if __name__ == "__main__":
    run_audit()
