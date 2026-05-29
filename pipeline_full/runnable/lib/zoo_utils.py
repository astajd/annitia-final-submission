"""Shared utilities for the model-zoo sprint (adapted for assembled repo).

The only behavioural change vs the original
`merged/model_zoo_sprint/scripts/zoo_utils.py` is path resolution: this copy
reads cached intermediates from `pipeline_full/runnable/cached_intermediates/`
rather than from absolute paths under `/home/.../annita/merged/`. The
`load_oof_baselines` function still uses the rounded proxy weights
(0.7905/0.1186/0.0909) — see docs/PROVENANCE.md. Those proxy
weights produce GPT-anchor *OOF reconstructions* only; the test-side anchor
is the frozen CSV in `cached_intermediates/gpt_track_handoff/best_submissions/`.

Convention enforced (from cross_method_schema_notes.md):
- patient_id_anon for OOF; trustii_id for test
- column names: hepatic_oof_risk / death_oof_risk for OOF;
  hepatic_risk / death_risk for test components.
- All blends rank-transform once.
- Higher score = higher risk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from claude_src.cv import cindex, repeated_stratified_folds  # noqa: F401
from claude_src.data import load_raw, build_targets, add_visit_metadata  # noqa: F401

# This file lives at:
#   <repo_root>/pipeline_full/runnable/lib/zoo_utils.py
# CACHED is <repo_root>/pipeline_full/runnable/cached_intermediates/
HERE = Path(__file__).resolve()
RUNNABLE = HERE.parents[1]
CACHED = RUNNABLE / "cached_intermediates"
PRED_DIR = CACHED / "model_zoo_sprint" / "predictions"


def load_labels():
    """Returns dict with pid, tid, hep_event/time, dea_event/time arrays.
    Requires raw challenge data — see data/README.md for placement."""
    train, test = load_raw()
    train = add_visit_metadata(train)
    test = add_visit_metadata(test)
    hep_t, dea_t = build_targets(train, death_mode="censor_missing_death_at_last")
    return dict(
        train=train, test=test,
        pid=train["patient_id_anon"].to_numpy(),
        tid=test["trustii_id"].to_numpy(),
        hep_event=hep_t["hepatic_event"].to_numpy().astype(bool),
        hep_time=hep_t["hepatic_time"].to_numpy().astype(float),
        dea_event=dea_t["death_event"].to_numpy().astype(bool),
        dea_time=dea_t["death_time"].to_numpy().astype(float),
    )


def load_anchors():
    """Returns dict of test-space arrays + OOF-space arrays for the 3 anchors."""
    out = {}
    out["gpt_anchor_csv"] = pd.read_csv(
        CACHED / "gpt_track_handoff/best_submissions/20260428_0059_phase3_10_horizon_blend_v2.csv"
    ).sort_values("trustii_id").reset_index(drop=True)
    out["cl_anchor_csv"] = pd.read_csv(
        CACHED / "claude_track_handoff/best_submissions/phase2_blend_2way_optimal.csv"
    ).sort_values("trustii_id").reset_index(drop=True)
    out["merge5050_csv"] = pd.read_csv(
        CACHED / "merge_sprint/submissions/merge_A_best_50_50_both.csv"
    ).sort_values("trustii_id").reset_index(drop=True)
    return out


def load_oof_baselines(pid):
    """Reconstructed OOFs we use repeatedly. Keyed by patient_id_anon order=pid.

    NOTE: the GPT-anchor OOF reconstruction here uses the rounded proxy
    weights 0.7905/0.1186/0.0909, NOT the actual blend weights from the GPT
    JSON sidecar. See docs/PROVENANCE.md. The test-side
    anchor is the frozen CSV (load_anchors), not derived here.
    """
    base = CACHED / "gpt_track_handoff/oof_predictions"
    surv_h = pd.read_csv(base / "survival_models/20260427_1306_phase3_6_no_visit_history__oof.csv").set_index("patient_id_anon").reindex(pid)
    nh = np.mean([rankdata(surv_h[c].to_numpy()) for c in surv_h.columns if c.startswith("hepatic__rsf__s")], axis=0)
    h1 = pd.read_csv(base / "horizon_components/NIT_plus_scores__hepatic__h1__lgbm_binary__oof.csv").set_index("patient_id_anon").reindex(pid)["oof"].to_numpy()
    h3s4 = pd.read_csv(base / "horizon_components/v3_hepatic_schema__hepatic__h3__lgbm_binary__s4__oof.csv").set_index("patient_id_anon").reindex(pid)["oof"].to_numpy()
    gpt_hep_oof = 0.7905 * nh + 0.1186 * rankdata(h1) + 0.0909 * rankdata(h3s4)

    surv_d = pd.read_csv(base / "survival_models/20260427_1154_phase3_current_state_v2__oof.csv").set_index("patient_id_anon").reindex(pid)
    v3_w = {"death__rsf__s0": 0.3654, "death__xgb_cox__s0": 0.2692,
            "death__xgb_aft__s0": 0.1731, "death__xgb_cox__s1": 0.1923}
    v3 = sum(w * rankdata(surv_d[k].to_numpy()) for k, w in v3_w.items())
    v3 = rankdata(v3)
    h5 = pd.read_csv(base / "horizon_components/current_state_v2__death__h5__catboost_binary__s3__oof.csv").set_index("patient_id_anon").reindex(pid)["oof"].to_numpy()
    h4 = pd.read_csv(base / "horizon_components/NIT_plus_scores__death__h4__catboost_binary__oof.csv").set_index("patient_id_anon").reindex(pid)["oof"].to_numpy()
    gpt_dea_oof = 0.7905 * v3 + 0.1186 * rankdata(h5) + 0.0909 * rankdata(h4)

    cl_hep_oof = pd.read_csv(CACHED / "claude_track_handoff/optional_oof_predictions/oof_blend_2way_optimal.csv").set_index("patient_id_anon").reindex(pid)["hepatic_oof_risk"].to_numpy()
    cl_dea_oof = pd.read_csv(CACHED / "claude_track_handoff/optional_oof_predictions/oof_death_xgb_longitudinal_summary.csv").set_index("patient_id_anon").reindex(pid)["death_oof_risk"].to_numpy()

    merge_5050_hep = 0.5 * rankdata(gpt_hep_oof) + 0.5 * rankdata(cl_hep_oof)
    merge_5050_dea = 0.5 * rankdata(gpt_dea_oof) + 0.5 * rankdata(cl_dea_oof)

    return dict(
        gpt_hep=gpt_hep_oof, gpt_dea=gpt_dea_oof,
        cl_hep=cl_hep_oof, cl_dea=cl_dea_oof,
        merge_hep=merge_5050_hep, merge_dea=merge_5050_dea,
    )


def cv_oof_test(fit_predict_fn, X_tr, X_te, event, time,
                n_splits=5, n_repeats=10, base_seed=42):
    """Run Claude 5x10 CV, return (oof_rank_mean, test_rank_mean, fold_cis)."""
    n_tr = len(X_tr); n_te = len(X_te)
    per_repeat_oof = np.full((n_repeats, n_tr), np.nan)
    per_repeat_test = np.full((n_repeats, n_te), np.nan)
    fold_cis = []
    for r, f, tr_idx, va_idx in repeated_stratified_folds(event, n_splits, n_repeats, base_seed):
        pred_va = fit_predict_fn(X_tr.iloc[tr_idx], event[tr_idx], time[tr_idx], X_tr.iloc[va_idx])
        per_repeat_oof[r, va_idx] = pred_va
        ci = cindex(event[va_idx], time[va_idx], pred_va)
        fold_cis.append(ci)
    for r in range(n_repeats):
        per_repeat_test[r] = fit_predict_fn(X_tr, event, time, X_te)
    oof = np.zeros(n_tr); te = np.zeros(n_te)
    for r in range(n_repeats):
        oof += rankdata(per_repeat_oof[r])
        te += rankdata(per_repeat_test[r])
    return oof / n_repeats, te / n_repeats, np.array(fold_cis)


def fold_seed_cv_oof_test(fit_predict_fn_factory, X_tr, X_te, event, time,
                          n_splits=5, n_repeats=10, base_seed=42):
    """Like cv_oof_test but factory takes a seed argument so per-repeat training
    differs by seed (deterministic reproducibility)."""
    n_tr = len(X_tr); n_te = len(X_te)
    per_repeat_oof = np.full((n_repeats, n_tr), np.nan)
    per_repeat_test = np.full((n_repeats, n_te), np.nan)
    fold_cis = []
    for r, f, tr_idx, va_idx in repeated_stratified_folds(event, n_splits, n_repeats, base_seed):
        fp = fit_predict_fn_factory(seed=base_seed + r)
        pred_va = fp(X_tr.iloc[tr_idx], event[tr_idx], time[tr_idx], X_tr.iloc[va_idx])
        per_repeat_oof[r, va_idx] = pred_va
        ci = cindex(event[va_idx], time[va_idx], pred_va)
        fold_cis.append(ci)
    for r in range(n_repeats):
        fp = fit_predict_fn_factory(seed=base_seed + r)
        per_repeat_test[r] = fp(X_tr, event, time, X_te)
    oof = np.zeros(n_tr); te = np.zeros(n_te)
    for r in range(n_repeats):
        oof += rankdata(per_repeat_oof[r])
        te += rankdata(per_repeat_test[r])
    return oof / n_repeats, te / n_repeats, np.array(fold_cis)


def save_pred(name, pid, oof_arr, tid, test_arr, kind="hep"):
    """kind ∈ {hep, dea}"""
    oof_col = f"{'hepatic' if kind == 'hep' else 'death'}_oof_risk"
    test_col = f"{'hepatic' if kind == 'hep' else 'death'}_risk"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id_anon": pid, oof_col: oof_arr}).to_csv(PRED_DIR / f"oof__{name}.csv", index=False)
    pd.DataFrame({"trustii_id": tid, test_col: test_arr}).to_csv(PRED_DIR / f"test__{name}.csv", index=False)


def report_metrics(name, oof_arr, event, time, test_arr, anchors, baseline_label="merge_50_50"):
    """Return a metrics dict for the candidate."""
    c = cindex(event, time, oof_arr)
    rho_g = float(spearmanr(test_arr, anchors["gpt_anchor_csv"]["risk_hepatic_event"]).statistic)
    return c
