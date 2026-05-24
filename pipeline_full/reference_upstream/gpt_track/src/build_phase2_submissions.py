"""Build the four Phase 2 candidate submissions and associated metadata.

For each candidate we:
1. Pick a pool of experiment output directories (defines the leakage profile).
2. Build an endpoint-specific ensemble for hepatic and death using all five
   selection methods (equal / cv_weighted / greedy / capped / seed_bagged).
3. Take the *best OOF* per-endpoint method for the candidate's hepatic and
   death predictions (separately — hepatic and death need not share a method).
4. Write `submissions/phase2_<label>.csv` and a sidecar
   `submissions/phase2_<label>.json` with full metadata (components, weights,
   OOF C-indices, weighted score, leakage tag, interpretation).

Candidates:

- `phase2_robust_longitudinal` — refined longitudinal sets without obvious
  follow-up proxies (Phase 2 sets), excluding the high-leakage Phase 1
  experiments.
- `phase2_aggressive_longitudinal` — adds aggressive_longitudinal /
  all_visits_longitudinal / NIT_plus_clinical_scores (moderate-to-high but
  excludes strict_time_aligned which the leaderboard already disproved).
- `phase2_clean_clinical_NIT` — only NIT, clinical scores, baseline, and
  landmark feature sets. Strongest defensibility.
- `phase2_best_oof_ensemble` — picks the best ensemble across *all* available
  experiments, regardless of leakage tag, as a CV-only upper bound. The
  metadata explicitly flags it as exploratory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .endpoint_ensemble import build_endpoint_ensembles
from .metrics import weighted_score
from .submission import make_submission
from .utils import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pool composition
# ---------------------------------------------------------------------------

def _phase1_dirs() -> dict[str, Path]:
    """Map Phase 1 experiment names to their (most recent) output dirs."""
    out: dict[str, Path] = {}
    for d in sorted(cfg.EXPERIMENT_OUTPUTS.iterdir()):
        if d.is_dir() and not d.name.startswith("phase2"):
            try:
                ename = json.loads((d / "config.json").read_text()).get("name")
            except Exception:
                continue
            if ename:
                out[ename] = d
    return out


def _phase2_dirs() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in sorted(cfg.EXPERIMENT_OUTPUTS.iterdir()):
        if not d.is_dir():
            continue
        try:
            ename = json.loads((d / "config.json").read_text()).get("name")
        except Exception:
            continue
        if ename and ename.startswith("phase2_"):
            out[ename] = d
    return out


_ROBUST_NAMES = {
    "phase2_first_1y",
    "phase2_first_2y",
    "phase2_first_3y",
    "phase2_first_3_visits",
    "phase2_baseline_plus_landmark_trends",
    "phase2_clinical_scores_dynamic",
    "phase2_NIT_longitudinal_only",
    "phase2_labs_longitudinal_only",
    "phase2_NIT_plus_scores_longitudinal",
    "phase2_longitudinal_no_followup_proxies",
    "001_baseline_v1_drop_missing_death",
    "002_baseline_v1_censor_missing_death",
    "003_early_v1_v3",
}

_AGGRESSIVE_NAMES = _ROBUST_NAMES | {
    "phase2_aggressive_longitudinal",
    "004_all_visits_longitudinal",
    "005_NIT_plus_clinical_scores",
    "007_full_high_risk",
}

_CLEAN_CLINICAL_NIT_NAMES = {
    "phase2_first_1y",
    "phase2_first_2y",
    "phase2_first_3y",
    "phase2_first_3_visits",
    "phase2_baseline_plus_landmark_trends",
    "phase2_clinical_scores_dynamic",
    "phase2_NIT_longitudinal_only",
    "phase2_NIT_plus_scores_longitudinal",
    "001_baseline_v1_drop_missing_death",
    "002_baseline_v1_censor_missing_death",
    "003_early_v1_v3",
}


def _resolve(names: set[str]) -> list[Path]:
    p1 = _phase1_dirs()
    p2 = _phase2_dirs()
    all_dirs = {**p1, **p2}
    return [all_dirs[n] for n in sorted(names) if n in all_dirs]


# ---------------------------------------------------------------------------
# Candidate builder
# ---------------------------------------------------------------------------

def _pick_best_per_endpoint(results: dict, test_predictions: dict) -> dict:
    """For each endpoint, pick the method with the highest OOF C-index."""
    chosen = {}
    for endpoint_name, methods in results.items():
        best_method = max(methods, key=lambda m: methods[m]["oof_cindex"] if np.isfinite(methods[m]["oof_cindex"]) else -1)
        chosen[endpoint_name] = {
            "method": best_method,
            **methods[best_method],
            "test": test_predictions[f"{endpoint_name}__{best_method}"],
        }
    return chosen


def build_candidate(label: str, names: set[str], description: str, leakage_tag: str) -> dict:
    dirs = _resolve(names)
    if not dirs:
        _LOG.warning("candidate %s: no resolved dirs from names=%s", label, names)
        return {}

    bundle = build_endpoint_ensembles(dirs, label=label)
    chosen = _pick_best_per_endpoint(bundle["results"], bundle["test_predictions"])

    ds = load_dataset()
    sub_path = make_submission(
        ds.test_df,
        risk_hepatic=chosen["hepatic"]["test"],
        risk_death=chosen["death"]["test"],
        sample_submission=ds.sample_submission,
        model_name=label,
    )

    cv_score = weighted_score(chosen["hepatic"]["oof_cindex"], chosen["death"]["oof_cindex"])
    metadata = {
        "label": label,
        "description": description,
        "leakage_tag": leakage_tag,
        "submission_csv": str(sub_path),
        "feature_sets": sorted({bundle["meta"][n]["feature_set"] for n in bundle["meta"]}),
        "experiments": sorted(bundle["meta"]),
        "hepatic": {
            "method": chosen["hepatic"]["method"],
            "components": chosen["hepatic"]["components"],
            "weights": chosen["hepatic"]["weights"],
            "oof_cindex": chosen["hepatic"]["oof_cindex"],
        },
        "death": {
            "method": chosen["death"]["method"],
            "components": chosen["death"]["components"],
            "weights": chosen["death"]["weights"],
            "oof_cindex": chosen["death"]["oof_cindex"],
        },
        "weighted_score_cv": cv_score,
        "all_methods_oof": {
            ep: {m: r["oof_cindex"] for m, r in bundle["results"][ep].items()}
            for ep in bundle["results"]
        },
    }
    meta_path = sub_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    _LOG.info(
        "%s -> hepatic=%.4f (%s) | death=%.4f (%s) | weighted=%.4f",
        label,
        chosen["hepatic"]["oof_cindex"],
        chosen["hepatic"]["method"],
        chosen["death"]["oof_cindex"],
        chosen["death"]["method"],
        cv_score,
    )
    return metadata


def main() -> None:
    cands = {
        "phase2_robust_longitudinal": (
            _ROBUST_NAMES,
            "Refined longitudinal + landmark + early Phase 1 baselines, no follow-up proxies.",
            "low/moderate",
        ),
        "phase2_aggressive_longitudinal": (
            _AGGRESSIVE_NAMES,
            "Robust pool plus aggressive_longitudinal and Phase 1 high-leakage models (excluding strict_time_aligned).",
            "moderate/high",
        ),
        "phase2_clean_clinical_NIT": (
            _CLEAN_CLINICAL_NIT_NAMES,
            "NIT + clinical scores + baselines + landmark windows. Strongest qualitative defensibility.",
            "low/moderate",
        ),
    }
    summaries = []
    for label, (names, desc, tag) in cands.items():
        m = build_candidate(label, names, desc, tag)
        if m:
            summaries.append(m)

    # phase2_best_oof_ensemble: union of every available experiment.
    all_names = set(_phase1_dirs()) | set(_phase2_dirs())
    best = build_candidate(
        "phase2_best_oof_ensemble",
        all_names,
        "Best OOF ensemble across all available experiments. Exploratory CV upper bound — not the qualitative pick.",
        "mixed (includes high-leakage components)",
    )
    if best:
        summaries.append(best)

    # Roll-up CSV.
    rollup = pd.DataFrame(
        [
            {
                "label": s["label"],
                "leakage_tag": s["leakage_tag"],
                "hepatic_method": s["hepatic"]["method"],
                "hepatic_oof": s["hepatic"]["oof_cindex"],
                "death_method": s["death"]["method"],
                "death_oof": s["death"]["oof_cindex"],
                "weighted_score_cv": s["weighted_score_cv"],
                "submission": s["submission_csv"],
            }
            for s in summaries
        ]
    )
    out_path = cfg.REPORTS_DIR / "phase2_submission_candidates.md"
    lines = [
        "# Phase 2 submission candidates\n",
        "All four candidates produced by `python -m src.build_phase2_submissions`. ",
        "OOF figures are *cross-validated* on the patient-level repeated stratified split shared across the contest. ",
        "These are local upper-bounds; the leaderboard delta indicates the leakage premium.\n",
        rollup.to_markdown(index=False, floatfmt=".4f"),
        "\n## Notes\n",
        "- `phase2_robust_longitudinal` is the **first leaderboard probe** of the Phase 2 cycle. "
        "It avoids the explicit follow-up proxies that we now know inflate CV.\n",
        "- `phase2_clean_clinical_NIT` is the **qualitative submission**: every component is "
        "interpretable to a hepatologist (NIT trajectories, FIB-4/APRI, demographics).\n",
        "- `phase2_aggressive_longitudinal` is the **leakage probe** — submit only after the robust "
        "candidate so we can compare LB deltas.\n",
        "- `phase2_best_oof_ensemble` is exploratory: it is the highest local CV but mixes leaky "
        "Phase 1 models. Do not submit unless investigating the local-vs-LB gap further.\n",
    ]
    out_path.write_text("\n".join(lines))
    _LOG.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
