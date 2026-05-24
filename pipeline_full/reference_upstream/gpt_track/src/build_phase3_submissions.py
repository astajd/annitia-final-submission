"""Build the three Phase 3 candidate submissions.

A. ``phase3_current_state_v2`` — endpoint-specific ensemble using only the
   current_state_v2 sweep components. Tests whether the rich current-state
   feature set alone can beat the public LB.
B. ``phase3_stacked_ensemble`` — stacked OOF meta-model (ridge/elastic-net)
   trained on every available base OOF including the discrete-time hazard.
C. ``phase3_discrete_time_hazard`` — promoted only if the DTH OOF C-index is
   competitive (>= 0.80 weighted score by itself).

Each candidate gets a sidecar JSON with: feature_set / experiments,
component list, weights, OOF C-indices per endpoint, weighted score, leakage
tag, rank correlation against the current public-best
(`phase2_aggressive_longitudinal`), and a short interpretation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .endpoint_ensemble import (
    build_endpoint_ensembles,
    collect_predictions,
    cv_weighted,
    equal_weight,
    greedy,
    seed_bagged,
)
from .metrics import cindex, weighted_score
from .submission import make_submission
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger

_LOG = get_logger(__name__)


_CURRENT_STATE_NAME = "phase3_current_state_v2"


def _phase3_dth_dir() -> Path:
    return cfg.EXPERIMENT_OUTPUTS / "phase3_discrete_time_hazard"


def _phase3_stack_dir() -> Path:
    return cfg.EXPERIMENT_OUTPUTS / "phase3_stacking"


def _experiment_dir_by_name(name: str) -> Path | None:
    for d in sorted(cfg.EXPERIMENT_OUTPUTS.iterdir()):
        if not d.is_dir():
            continue
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        try:
            blob = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if blob.get("name") == name:
            return d
    return None


def _public_best_test_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Return (hep, death) test predictions of the public-best ensemble.

    We rebuild it from `phase2_aggressive_longitudinal`'s metadata so the rank
    correlation reflects what is actually on the LB.
    """
    submissions_dir = cfg.SUBMISSIONS_DIR
    candidates = sorted(submissions_dir.glob("*phase2_aggressive_longitudinal*.csv"))
    if candidates:
        latest = candidates[-1]
        df = pd.read_csv(latest)
        return df[cfg.SUB_HEPATIC_COL].to_numpy(), df[cfg.SUB_DEATH_COL].to_numpy()
    return np.array([]), np.array([])


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return float("nan")
    s = pd.Series(a).rank(method="average")
    t = pd.Series(b).rank(method="average")
    return float(s.corr(t, method="spearman"))


# ---------------------------------------------------------------------------
# Candidate A: current_state_v2-only
# ---------------------------------------------------------------------------

def candidate_current_state_v2() -> dict:
    d = _experiment_dir_by_name(_CURRENT_STATE_NAME)
    if d is None:
        _LOG.warning("phase3_current_state_v2 directory not found")
        return {}
    bundle = build_endpoint_ensembles([d], label="phase3_current_state_v2")
    chosen = {ep: max(meth, key=lambda m: meth[m]["oof_cindex"]
                      if np.isfinite(meth[m]["oof_cindex"]) else -1)
              for ep, meth in bundle["results"].items()}
    ds = load_dataset()
    test_h = bundle["test_predictions"][f"hepatic__{chosen['hepatic']}"]
    test_d = bundle["test_predictions"][f"death__{chosen['death']}"]
    sub_path = make_submission(
        ds.test_df, risk_hepatic=test_h, risk_death=test_d,
        sample_submission=ds.sample_submission, model_name="phase3_current_state_v2",
    )
    pub_h, pub_d = _public_best_test_predictions()
    cv = weighted_score(
        bundle["results"]["hepatic"][chosen["hepatic"]]["oof_cindex"],
        bundle["results"]["death"][chosen["death"]]["oof_cindex"],
    )
    metadata = {
        "label": "phase3_current_state_v2",
        "description": "Endpoint-specific ensemble built only from the current_state_v2 sweep. Multi-seed runs over CatBoost / LightGBM / XGB-Cox / XGB-binary / XGB-AFT / RSF.",
        "leakage_tag": "moderate-high (uses current age / last observed age and missingness, by spec)",
        "submission_csv": str(sub_path),
        "uses_target_derived_features": False,
        "feature_sets": ["current_state_v2"],
        "experiments": [d.name],
        "hepatic": {
            "method": chosen["hepatic"],
            "components": bundle["results"]["hepatic"][chosen["hepatic"]]["components"],
            "weights": bundle["results"]["hepatic"][chosen["hepatic"]]["weights"],
            "oof_cindex": bundle["results"]["hepatic"][chosen["hepatic"]]["oof_cindex"],
        },
        "death": {
            "method": chosen["death"],
            "components": bundle["results"]["death"][chosen["death"]]["components"],
            "weights": bundle["results"]["death"][chosen["death"]]["weights"],
            "oof_cindex": bundle["results"]["death"][chosen["death"]]["oof_cindex"],
        },
        "weighted_score_cv": cv,
        "all_methods_oof": {
            ep: {m: r["oof_cindex"] for m, r in bundle["results"][ep].items()}
            for ep in bundle["results"]
        },
        "rank_corr_with_public_best": {
            "hepatic": _rank_corr(test_h, pub_h),
            "death": _rank_corr(test_d, pub_d),
        },
    }
    sub_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))
    _LOG.info("A: phase3_current_state_v2 weighted=%.4f hep=%.4f death=%.4f", cv,
              metadata["hepatic"]["oof_cindex"], metadata["death"]["oof_cindex"])
    return metadata


# ---------------------------------------------------------------------------
# Candidate B: stacked ensemble (uses src.stacking outputs)
# ---------------------------------------------------------------------------

def candidate_stacked_ensemble() -> dict:
    stack_dir = _phase3_stack_dir()
    if not stack_dir.exists():
        _LOG.warning("phase3_stacking dir not found; run src.stacking first")
        return {}
    summary = json.loads((stack_dir / "summary.json").read_text())
    ds = load_dataset()
    # Test predictions written by src.stacking.
    test_hep = pd.read_csv(stack_dir / "test_stack_hepatic.csv")
    test_dea = pd.read_csv(stack_dir / "test_stack_death.csv")
    # Align to test order.
    test_hep = test_hep.set_index(cfg.TRUSTII_ID_COL).reindex(ds.test_df[cfg.TRUSTII_ID_COL].values)
    test_dea = test_dea.set_index(cfg.TRUSTII_ID_COL).reindex(ds.test_df[cfg.TRUSTII_ID_COL].values)
    risk_h = test_hep["test_stack_hepatic"].fillna(0.5).to_numpy()
    risk_d = test_dea["test_stack_death"].fillna(0.5).to_numpy()
    sub_path = make_submission(
        ds.test_df, risk_hepatic=risk_h, risk_death=risk_d,
        sample_submission=ds.sample_submission, model_name="phase3_stacked_ensemble",
    )
    pub_h, pub_d = _public_best_test_predictions()
    hep_oof = summary["hepatic"]["results"][summary["hepatic"]["best_method"]]["oof_cindex"] or float("nan")
    dea_oof = summary["death"]["results"][summary["death"]["best_method"]]["oof_cindex"] or float("nan")
    cv = weighted_score(hep_oof, dea_oof)
    metadata = {
        "label": "phase3_stacked_ensemble",
        "description": "Linear meta-learner on percentile-rank base OOFs (Phase 1+2+3 + DTH if available).",
        "leakage_tag": "mixed — inherits from base components; meta-model is rank-only.",
        "submission_csv": str(sub_path),
        "uses_target_derived_features": False,
        "stacking_summary": summary,
        "hepatic_oof_cindex": hep_oof,
        "death_oof_cindex": dea_oof,
        "weighted_score_cv": cv,
        "rank_corr_with_public_best": {
            "hepatic": _rank_corr(risk_h, pub_h),
            "death": _rank_corr(risk_d, pub_d),
        },
    }
    sub_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))
    _LOG.info("B: phase3_stacked_ensemble weighted=%.4f hep=%.4f death=%.4f", cv, hep_oof, dea_oof)
    return metadata


# ---------------------------------------------------------------------------
# Candidate C: discrete-time hazard alone (only if competitive)
# ---------------------------------------------------------------------------

def candidate_discrete_time_hazard(threshold: float = 0.80) -> dict:
    dth_dir = _phase3_dth_dir()
    if not dth_dir.exists():
        return {}
    summary = json.loads((dth_dir / "summary.json").read_text())
    hep_mean = summary.get("hepatic", {}).get("mean")
    dea_mean = summary.get("death", {}).get("mean")
    if hep_mean is None or dea_mean is None:
        return {}
    cv = weighted_score(hep_mean, dea_mean)
    if cv < threshold:
        _LOG.info("C: DTH weighted=%.4f below %.2f, not promoting", cv, threshold)
        return {"label": "phase3_discrete_time_hazard", "promoted": False, "weighted_score_cv": cv,
                "hepatic_oof_cindex": hep_mean, "death_oof_cindex": dea_mean}

    ds = load_dataset()
    test_hep = pd.read_csv(dth_dir / "test_dth_hepatic.csv")
    test_dea = pd.read_csv(dth_dir / "test_dth_death.csv")
    test_hep = test_hep.set_index(cfg.TRUSTII_ID_COL).reindex(ds.test_df[cfg.TRUSTII_ID_COL].values)
    test_dea = test_dea.set_index(cfg.TRUSTII_ID_COL).reindex(ds.test_df[cfg.TRUSTII_ID_COL].values)
    risk_h = test_hep["test_dth_hepatic"].fillna(0.5).to_numpy()
    risk_d = test_dea["test_dth_death"].fillna(0.5).to_numpy()
    sub_path = make_submission(
        ds.test_df, risk_hepatic=risk_h, risk_death=risk_d,
        sample_submission=ds.sample_submission, model_name="phase3_discrete_time_hazard",
    )
    pub_h, pub_d = _public_best_test_predictions()
    metadata = {
        "label": "phase3_discrete_time_hazard",
        "description": "Patient-period long-format LightGBM hazard model. Patient risk = 1 - prod(1 - hazard_t).",
        "leakage_tag": "low/moderate (target enters only via the binned event indicator within the patient's observation window).",
        "submission_csv": str(sub_path),
        "uses_target_derived_features": False,
        "promoted": True,
        "hepatic_oof_cindex": hep_mean,
        "death_oof_cindex": dea_mean,
        "weighted_score_cv": cv,
        "rank_corr_with_public_best": {
            "hepatic": _rank_corr(risk_h, pub_h),
            "death": _rank_corr(risk_d, pub_d),
        },
        "summary": summary,
    }
    sub_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))
    _LOG.info("C: phase3_discrete_time_hazard weighted=%.4f hep=%.4f death=%.4f", cv, hep_mean, dea_mean)
    return metadata


def main() -> None:
    A = candidate_current_state_v2()
    B = candidate_stacked_ensemble()
    C = candidate_discrete_time_hazard()

    rollup_rows = []
    for cand in (A, B, C):
        if not cand:
            continue
        rollup_rows.append({
            "label": cand.get("label"),
            "promoted": cand.get("promoted", True),
            "leakage_tag": cand.get("leakage_tag"),
            "hep_cindex": cand.get("hepatic", {}).get("oof_cindex") or cand.get("hepatic_oof_cindex"),
            "death_cindex": cand.get("death", {}).get("oof_cindex") or cand.get("death_oof_cindex"),
            "weighted_score_cv": cand.get("weighted_score_cv"),
            "rank_corr_pub_hep": cand.get("rank_corr_with_public_best", {}).get("hepatic"),
            "rank_corr_pub_death": cand.get("rank_corr_with_public_best", {}).get("death"),
            "submission_csv": cand.get("submission_csv"),
        })
    df = pd.DataFrame(rollup_rows)
    out = cfg.REPORTS_DIR / "phase3_submission_candidates.md"
    if df.empty:
        out.write_text("# Phase 3 submission candidates\n\n(no candidates produced)\n")
    else:
        lines = [
            "# Phase 3 submission candidates\n",
            "All Phase 3 candidates produced by `python -m src.build_phase3_submissions`. ",
            "OOF figures use the same hepatic-stratified repeated K-fold as Phase 1/2. ",
            "Rank correlations are versus the current public best (`phase2_aggressive_longitudinal`, LB 0.86813).\n",
            df.to_markdown(index=False, floatfmt=".4f"),
            "\n## Notes\n",
            "- Submit the candidate with the best **defensible** OOF score whose rank correlation against the public best is below ~0.95 (otherwise it provides no new information on the LB).\n",
            "- The stacked candidate is the most informative diversity probe; the current-state-v2 candidate is the cleanest single-feature-set probe.\n",
            "- DTH is only promoted to a submission if its weighted CV >= 0.80; below that we keep its OOFs in the stacker but do not submit alone.\n",
        ]
        out.write_text("\n".join(lines))
    _LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
