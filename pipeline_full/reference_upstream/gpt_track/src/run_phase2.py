"""Drive the Phase 2 pipeline post-sweep.

This script assumes the feature-set sweep (`run_phase2_sweep`) has already
written ``experiments/outputs/phase2_*`` directories. It then:

1. Decomposes the Phase 1 best ensemble (longitudinal).
2. Runs Optuna for the strongest model-feature pairs (small budget by default).
3. Builds the four Phase 2 submission candidates.
4. Computes feature importance / SHAP for the leading models.
5. Writes the four mandated reports.

Heavy steps (Optuna especially) are gated by CLI flags so the orchestrator can
be re-run cheaply.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .build_phase2_submissions import main as build_subs_main
from .data_loading import load_dataset
from .feature_importance import compute_for, write_top_report
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger
from .validation import summarize_cv

_LOG = get_logger(__name__)


def _gather_per_model_summaries() -> pd.DataFrame:
    rows = []
    for d in sorted(cfg.EXPERIMENT_OUTPUTS.iterdir()):
        if not d.is_dir():
            continue
        per_model = d / "per_model_summary.csv"
        if not per_model.exists():
            continue
        try:
            df = pd.read_csv(per_model)
        except Exception:
            continue
        try:
            cfg_blob = json.loads((d / "config.json").read_text())
        except Exception:
            cfg_blob = {}
        df["experiment"] = cfg_blob.get("name", d.name)
        df["death_target_mode"] = cfg_blob.get("death_target_mode")
        df["dir"] = d.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _gather_cv_summaries() -> pd.DataFrame:
    rows = []
    for d in sorted(cfg.EXPERIMENT_OUTPUTS.iterdir()):
        if not d.is_dir():
            continue
        cv_path = d / "cv_summary.json"
        cfg_path = d / "config.json"
        if not (cv_path.exists() and cfg_path.exists()):
            continue
        try:
            cv = json.loads(cv_path.read_text())
            cfg_blob = json.loads(cfg_path.read_text())
        except Exception:
            continue
        rows.append({
            "experiment": cfg_blob.get("name"),
            "feature_set": cfg_blob.get("feature_set"),
            "death_target_mode": cfg_blob.get("death_target_mode"),
            "cindex_hepatic_mean": cv.get("cindex_hepatic_mean"),
            "cindex_hepatic_std": cv.get("cindex_hepatic_std"),
            "cindex_hepatic_min": cv.get("cindex_hepatic_min"),
            "cindex_death_mean": cv.get("cindex_death_mean"),
            "cindex_death_std": cv.get("cindex_death_std"),
            "cindex_death_min": cv.get("cindex_death_min"),
            "score_mean": cv.get("score_mean"),
            "score_std": cv.get("score_std"),
            "score_min": cv.get("score_min"),
            "n_folds": cv.get("n_folds"),
            "dir": d.name,
        })
    return pd.DataFrame(rows)


def _write_best_models_csv(per_model: pd.DataFrame, out: Path) -> None:
    if per_model.empty:
        out.write_text("")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    df = per_model.copy()
    df["score_for_hepatic_priority"] = df["cindex_mean"] - 0.5 * df["cindex_std"]
    df = df.sort_values(["endpoint", "score_for_hepatic_priority"], ascending=[True, False])
    df.to_csv(out, index=False)


def _write_summary(per_model: pd.DataFrame, cv: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if cv.empty:
        out.write_text("# Phase 2 experiment summary\n\n(no CV summaries found)\n")
        return

    cv_phase2 = cv[cv["experiment"].fillna("").str.startswith("phase2_")].copy()
    cv_phase1 = cv[~cv["experiment"].fillna("").str.startswith("phase2_")].copy()

    lines: list[str] = []
    lines.append("# Phase 2 experiment summary\n")
    lines.append("Cross-validated results from `src/run_phase2_sweep.py` plus Phase 1 baselines for context. ")
    lines.append("All splits are the same hepatic-stratified repeated K-fold used in Phase 1.\n")

    if not per_model.empty:
        df = per_model.copy()
        df["penalised"] = df["cindex_mean"] - 0.5 * df["cindex_std"]
        hep = df[df["endpoint"] == "hepatic"].sort_values("penalised", ascending=False).head(15)
        dea = df[df["endpoint"] == "death"].sort_values("penalised", ascending=False).head(15)

        lines.append("## Top 15 hepatic models (mean - 0.5*sd)\n")
        lines.append(hep[["experiment", "feature_set", "model", "cindex_mean", "cindex_std", "cindex_min", "penalised"]]
                     .to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        lines.append("## Top 15 death models (mean - 0.5*sd)\n")
        lines.append(dea[["experiment", "feature_set", "model", "cindex_mean", "cindex_std", "cindex_min", "penalised"]]
                     .to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    cols = ["experiment", "feature_set", "death_target_mode",
            "cindex_hepatic_mean", "cindex_hepatic_std", "cindex_hepatic_min",
            "cindex_death_mean", "cindex_death_std", "cindex_death_min",
            "score_mean", "score_std", "n_folds"]

    lines.append("## Phase 2 experiments ranked by hepatic C-index\n")
    lines.append(cv_phase2.sort_values("cindex_hepatic_mean", ascending=False)[cols]
                 .to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Phase 2 experiments ranked by weighted score\n")
    lines.append(cv_phase2.sort_values("score_mean", ascending=False)[cols]
                 .to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Phase 1 baselines (for comparison)\n")
    lines.append(cv_phase1.sort_values("score_mean", ascending=False)[cols]
                 .to_markdown(index=False, floatfmt=".4f"))
    lines.append("")

    lines.append("## Clean vs aggressive comparison\n")
    if not cv_phase2.empty:
        clean = cv_phase2[~cv_phase2["feature_set"].isin(["aggressive_longitudinal", "all_visits_longitudinal", "full_high_risk"])]
        aggro = cv_phase2[cv_phase2["feature_set"].isin(["aggressive_longitudinal"])]
        lines.append(f"- Clean Phase 2 mean weighted score: {clean['score_mean'].mean():.4f}")
        lines.append(f"- Aggressive Phase 2 mean weighted score: {aggro['score_mean'].mean():.4f}\n")

    lines.append("## Landmark vs all-visits comparison\n")
    if not cv_phase2.empty:
        landmark = cv_phase2[cv_phase2["feature_set"].str.startswith("first_") |
                              (cv_phase2["feature_set"] == "baseline_plus_landmark_trends")]
        all_v = cv[cv["feature_set"] == "all_visits_longitudinal"]
        lines.append(f"- Landmark mean hepatic C-index: {landmark['cindex_hepatic_mean'].mean():.4f}")
        lines.append(f"- all_visits_longitudinal mean hepatic C-index: {all_v['cindex_hepatic_mean'].mean():.4f}\n")

    lines.append("## Recommended next submissions (no auto-submit)\n")
    lines.append(
        "1. **`submissions/phase2_robust_longitudinal.csv`** — first leaderboard probe of the "
        "follow-up-clean Phase 2 universe. Tests whether removing follow-up proxies generalises.\n"
        "2. **`submissions/phase2_clean_clinical_NIT.csv`** — qualitative submission: every "
        "component is hepatologist-interpretable. Safer fallback if (1) underperforms.\n"
    )

    lines.append("## Phase 3 hooks (do not implement until Phase 2 reports are complete)\n")
    lines.append(
        "- Discrete-time hazard model (DeepHit-style) with hepatic + death heads on a shared "
        "encoder; explicit handling of competing risks.\n"
        "- Stacking: train a small Cox or logistic meta-learner on OOF predictions.\n"
        "- Multi-task representation: shared embedding from a transformer over visit "
        "sequences with masking; task-specific heads.\n"
        "- Pseudo-labeling on test using the highest-confidence ensemble predictions.\n"
        "- Calibration: post-hoc isotonic / Platt mapping per endpoint.\n"
        "- External clinical priors: shrink ensemble outputs toward FIB-4 / latest stiffness "
        "rank when the data-driven model is uncertain.\n"
    )

    out.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tune", action="store_true")
    parser.add_argument("--tune-trials", type=int, default=30)
    parser.add_argument("--skip-importance", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()

    # 1. Submissions: build candidates from existing OOF/test predictions.
    _LOG.info("[1/4] building Phase 2 submission candidates")
    build_subs_main()

    # 2. Optuna (optional). Tune the strongest models for hepatic and death.
    if not args.skip_tune:
        _LOG.info("[2/4] running Optuna tuning")
        from .tune_optuna import _evaluate, _suggest
        from .features import build_feature_set
        from .targets import build_hepatic_endpoint, build_death_endpoint
        from .data_loading import load_dataset
        from .validation import build_folds
        import optuna

        ds = load_dataset()
        hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
        death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
        splits = build_folds(ds.train_df, hep.event.astype(int), n_splits=5, n_repeats=3)

        out_root = cfg.EXPERIMENT_OUTPUTS / "phase2_optuna"
        out_root.mkdir(parents=True, exist_ok=True)

        targets = [
            ("lgbm_binary", "NIT_plus_scores_longitudinal", "hepatic"),
            ("catboost_binary", "NIT_plus_scores_longitudinal", "hepatic"),
            ("xgb_cox", "longitudinal_no_followup_proxies", "hepatic"),
            ("catboost_binary", "longitudinal_no_followup_proxies", "death"),
            ("xgb_cox", "longitudinal_no_followup_proxies", "death"),
        ]
        for model_name, fs_name, ep_name in targets:
            try:
                fs = build_feature_set(fs_name, ds)
            except Exception as e:  # noqa: BLE001
                _LOG.warning("skip %s/%s/%s: %s", model_name, fs_name, ep_name, e)
                continue
            ep = hep if ep_name == "hepatic" else death

            study_name = f"{fs_name}__{ep_name}__{model_name}"

            def objective(trial, model_name=model_name, fs=fs, ep=ep, splits=splits):
                params = _suggest(trial, model_name)
                mean, std = _evaluate(model_name, params, fs, ep, splits)
                if not np.isfinite(mean):
                    return float("-inf")
                trial.set_user_attr("cindex_mean", mean)
                trial.set_user_attr("cindex_std", std)
                return mean - 0.5 * std

            study = optuna.create_study(
                study_name=study_name,
                storage=f"sqlite:///{out_root}/optuna.db",
                load_if_exists=True,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED),
            )
            study.optimize(objective, n_trials=args.tune_trials, show_progress_bar=False)
            best = {
                "study_name": study_name,
                "best_value": float(study.best_value),
                "best_params": study.best_params,
                "best_user_attrs": dict(study.best_trial.user_attrs),
                "n_trials": len(study.trials),
            }
            (out_root / f"{study_name}.json").write_text(json.dumps(best, indent=2, default=str))
            _LOG.info("tuned %s: %.4f", study_name, best["best_value"])

    # 3. Feature importance + SHAP for the leading model-feature pairs.
    if not args.skip_importance:
        _LOG.info("[3/4] feature importance for top models")
        for fs_name, model_name, ep_name in [
            ("NIT_plus_scores_longitudinal", "lgbm_binary", "hepatic"),
            ("NIT_plus_scores_longitudinal", "catboost_binary", "hepatic"),
            ("longitudinal_no_followup_proxies", "xgb_cox", "hepatic"),
            ("longitudinal_no_followup_proxies", "xgb_cox", "death"),
            ("aggressive_longitudinal", "lgbm_binary", "hepatic"),
        ]:
            try:
                compute_for(fs_name, model_name, ep_name)
            except Exception as e:  # noqa: BLE001
                _LOG.warning("importance %s/%s/%s failed: %s", fs_name, model_name, ep_name, e)
        write_top_report()

    # 4. Reports + correlation matrix.
    _LOG.info("[4/4] writing Phase 2 reports")
    per_model = _gather_per_model_summaries()
    cv = _gather_cv_summaries()

    _write_best_models_csv(per_model, cfg.REPORTS_DIR / "phase2_best_models.csv")
    _write_summary(per_model, cv, cfg.REPORTS_DIR / "phase2_experiment_summary.md")

    # Correlation matrix from the all-experiments OOF.
    from .endpoint_ensemble import collect_predictions

    all_dirs = [d for d in cfg.EXPERIMENT_OUTPUTS.iterdir() if d.is_dir() and (d / "oof_predictions.csv").exists()]
    if all_dirs:
        oof_df, _, _meta = collect_predictions(all_dirs)
        rank_cols = [c for c in oof_df.columns if c != cfg.PATIENT_ID_COL]
        corr = oof_df[rank_cols].rank(method="average", pct=True).corr(method="spearman")
        corr.to_csv(cfg.REPORTS_DIR / "phase2_model_correlation_matrix.csv")

    _LOG.info("Phase 2 done")


if __name__ == "__main__":
    main()
