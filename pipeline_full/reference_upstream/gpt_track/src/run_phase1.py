"""Run the full Phase 1 pipeline end-to-end.

Steps:
1. Build the leakage audit report.
2. Run experiments 001..007.
3. Generate two cross-experiment ensemble submissions:
   - ensemble_clean       = mean rank of 001/002/003/005/006 ensembles
   - ensemble_longitudinal = mean rank of 004/006/007 ensembles
   - ensemble_hybrid       = mean rank of all of the above

4. Write reports/experiment_summary.md.

Usage::

    python -m src.run_phase1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .leakage_audit import run_audit
from .models.ensemble import rank_average
from .run_experiment import run_config
from .submission import make_submission
from .utils import get_logger, timestamp

_LOG = get_logger(__name__)


_PHASE1_CONFIGS = [
    "001_baseline_v1_drop_missing_death.yaml",
    "002_baseline_v1_censor_missing_death.yaml",
    "003_early_v1_v3.yaml",
    "004_all_visits_longitudinal.yaml",
    "005_NIT_plus_clinical_scores.yaml",
    "006_strict_time_aligned.yaml",
    "007_full_high_risk.yaml",
]


def _read_summary(out_dir: Path) -> dict:
    cv_path = out_dir / "cv_summary.json"
    cfg_path = out_dir / "config.json"
    pred_path = out_dir / "test_predictions.csv"
    return {
        "out_dir": out_dir,
        "config": json.loads(cfg_path.read_text()) if cfg_path.exists() else {},
        "cv_summary": json.loads(cv_path.read_text()) if cv_path.exists() else {},
        "test_predictions": pd.read_csv(pred_path) if pred_path.exists() else None,
    }


def main() -> None:
    cfg.ensure_dirs()

    _LOG.info("[1/3] running leakage audit")
    run_audit()

    results: list[dict] = []
    _LOG.info("[2/3] running experiments")
    for fname in _PHASE1_CONFIGS:
        path = cfg.EXPERIMENT_CONFIGS / fname
        if not path.exists():
            _LOG.warning("config %s not found, skipping", path)
            continue
        out_dir = run_config(path)
        results.append(_read_summary(out_dir))

    _LOG.info("[3/3] building cross-experiment ensembles")

    ds = load_dataset()
    n_test = len(ds.test_df)

    def _gather(filter_fn) -> tuple[dict, dict]:
        hep_preds = {}
        death_preds = {}
        for r in results:
            cfg_blob = r["config"]
            preds = r["test_predictions"]
            if preds is None or not filter_fn(cfg_blob):
                continue
            label = cfg_blob.get("name", str(r["out_dir"].name))
            if "ensemble_hepatic" in preds:
                hep_preds[label] = preds["ensemble_hepatic"].to_numpy()
            if "ensemble_death" in preds:
                death_preds[label] = preds["ensemble_death"].to_numpy()
        return hep_preds, death_preds

    clean_names = {
        "001_baseline_v1_drop_missing_death",
        "002_baseline_v1_censor_missing_death",
        "003_early_v1_v3",
        "005_NIT_plus_clinical_scores",
        "006_strict_time_aligned",
    }
    long_names = {
        "004_all_visits_longitudinal",
        "006_strict_time_aligned",
        "007_full_high_risk",
    }

    submissions_written: list[Path] = []

    for label, predicate in [
        ("ensemble_clean", lambda c: c.get("name") in clean_names),
        ("ensemble_longitudinal", lambda c: c.get("name") in long_names),
        ("ensemble_hybrid", lambda c: True),
    ]:
        hep_preds, death_preds = _gather(predicate)
        if not hep_preds or not death_preds:
            _LOG.warning("%s: no predictions to ensemble", label)
            continue
        hep = rank_average(hep_preds)
        dth = rank_average(death_preds)
        path = make_submission(
            ds.test_df,
            risk_hepatic=hep,
            risk_death=dth,
            sample_submission=ds.sample_submission,
            model_name=label,
        )
        submissions_written.append(path)

    # Build the cross-experiment summary.
    summary_path = cfg.REPORTS_DIR / "experiment_summary.md"
    rows = []
    for r in results:
        c = r["config"]
        s = r["cv_summary"]
        rows.append(
            {
                "experiment": c.get("name"),
                "feature_set": c.get("feature_set"),
                "death_target_mode": c.get("death_target_mode"),
                "score_mean": s.get("score_mean"),
                "score_std": s.get("score_std"),
                "score_min": s.get("score_min"),
                "cindex_hepatic_mean": s.get("cindex_hepatic_mean"),
                "cindex_death_mean": s.get("cindex_death_mean"),
                "n_folds": s.get("n_folds"),
            }
        )
    df = pd.DataFrame(rows)

    with summary_path.open("w") as f:
        f.write("# Experiment summary (Phase 1)\n\n")
        f.write(
            "Cross-validation results from `python -m src.run_phase1`. "
            "Score = 0.7 * C-index hepatic + 0.3 * C-index death. "
            "All experiments share the same hepatic-stratified repeated K-fold splits.\n\n"
        )
        f.write("## Per-experiment CV summary\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n\n")

        if not df.empty and df["score_mean"].notna().any():
            best_score = df.loc[df["score_mean"].idxmax()]
            best_hep = df.loc[df["cindex_hepatic_mean"].idxmax()]
            best_death = df.loc[df["cindex_death_mean"].idxmax()]
            f.write("## Best per metric\n\n")
            f.write(f"- best weighted score: **{best_score['experiment']}** (mean={best_score['score_mean']:.4f})\n")
            f.write(f"- best hepatic C-index: **{best_hep['experiment']}** (mean={best_hep['cindex_hepatic_mean']:.4f})\n")
            f.write(f"- best death C-index:   **{best_death['experiment']}** (mean={best_death['cindex_death_mean']:.4f})\n\n")
        f.write("## Cross-experiment submissions\n\n")
        for p in submissions_written:
            f.write(f"- `{p}`\n")
        f.write("\n## Recommendations\n\n")
        f.write(
            "- For the *first* leaderboard submission, prefer the cleanest model "
            "with comparable CV: typically `ensemble_clean` (no all-visits-leakage). "
            "Use `ensemble_longitudinal` to probe the leakage ceiling without making "
            "it the qualitative submission.\n"
        )
        f.write(
            "- Compare 001 vs 002 to decide death-NaN handling. The censor mode "
            "keeps every patient and usually wins on the death endpoint.\n"
        )
        f.write(
            "- Phase 2 candidates (overnight): bigger repeats (e.g. 5x10), Bayesian "
            "tuning of model hyperparameters, multi-task discrete-time hazards, "
            "shared-trunk neural nets, and stacking the model-level ranks with a "
            "Cox meta-learner.\n"
        )

    _LOG.info("phase 1 done: %s", summary_path)


if __name__ == "__main__":
    main()
