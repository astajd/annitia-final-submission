"""Phase 3.5 ablations on `phase3_current_state_v2`.

We re-use the existing OOF / test predictions from the
``phase3_current_state_v2`` experiment dir (no re-training required) and
compose endpoint-specific ensembles under each ablation. Where ablations
require training a new model — for example "remove visit-history features
from the input" — we retrain that single configuration on the same folds.

Outputs:
  reports/phase3_5_ablations.md
  reports/phase3_5_ablations.csv
  experiments/outputs/phase3_5_ablations/<label>/oof.csv  (when retrained)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .endpoint_ensemble import (
    capped,
    collect_predictions,
    cv_weighted,
    equal_weight,
    greedy,
    seed_bagged,
)
from .features import build_feature_set
from .features.current_state_v2 import current_state_v2 as build_current_state_v2_df
from .metrics import cindex, weighted_score
from .models import build_model
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger
from .validation import build_folds

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Component pool builder (reads phase3 OOF/test)
# ---------------------------------------------------------------------------

@dataclass
class Pool:
    train_idx: np.ndarray
    test_idx: np.ndarray
    oof: dict[str, np.ndarray] = field(default_factory=dict)
    test: dict[str, np.ndarray] = field(default_factory=dict)
    n_train: int = 0
    n_test: int = 0
    splits: list = field(default_factory=list)


def _aligned(col_name: str, df: pd.DataFrame, positions: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    if col_name in df.columns:
        out[positions] = df[col_name].to_numpy()
    return out


def load_phase3_pool() -> tuple[Pool, dict[str, list[str]]]:
    """Load the per-(model, endpoint) OOF/test prediction columns."""
    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)

    phase3_dirs = []
    for d in cfg.EXPERIMENT_OUTPUTS.iterdir():
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        try:
            blob = json.loads((d / "config.json").read_text())
        except Exception:
            continue
        if blob.get("name") == "phase3_current_state_v2":
            phase3_dirs.append(d)
    if not phase3_dirs:
        raise RuntimeError("phase3_current_state_v2 experiment dir not found")

    oof_df, test_df, _ = collect_predictions(phase3_dirs)

    pid_col = cfg.PATIENT_ID_COL
    tid_col = cfg.TRUSTII_ID_COL
    train_pid_to_row = {pid: i for i, pid in enumerate(ds.train_df[pid_col].values)}
    pos = oof_df[pid_col].map(train_pid_to_row).to_numpy().astype(int)
    test_idx_by_id = {tid: i for i, tid in enumerate(ds.test_df[tid_col].values)}
    tpos = test_df[tid_col].map(test_idx_by_id).to_numpy().astype(int)

    n_train = len(ds.train_df)
    n_test = len(ds.test_df)
    pool = Pool(train_idx=pos, test_idx=tpos, n_train=n_train, n_test=n_test, splits=splits)

    cols_by_endpoint: dict[str, list[str]] = {"hepatic": [], "death": []}
    for c in oof_df.columns:
        if c == pid_col:
            continue
        if "::ensemble_" in c:
            continue
        if "::hepatic__" in c:
            cols_by_endpoint["hepatic"].append(c)
        elif "::death__" in c:
            cols_by_endpoint["death"].append(c)

    for c in cols_by_endpoint["hepatic"] + cols_by_endpoint["death"]:
        pool.oof[c] = _aligned(c, oof_df, pos, n_train)
        pool.test[c] = _aligned(c, test_df, tpos, n_test)

    return pool, cols_by_endpoint


# ---------------------------------------------------------------------------
# Method runner
# ---------------------------------------------------------------------------

def best_ensemble(component_names: list[str], pool: Pool, event: np.ndarray, time: np.ndarray, endpoint: str) -> dict:
    """Try equal/greedy/cv_weighted/seed_bagged and return the best by OOF C-index."""
    if not component_names:
        return {"method": None, "components": [], "weights": {}, "oof_cindex": float("nan"),
                "test": np.full(pool.n_test, np.nan)}
    methods = {
        "equal": equal_weight(component_names, pool.oof, event, time, endpoint),
        "greedy": greedy(component_names, pool.oof, event, time, endpoint),
        "cv_weighted": cv_weighted(component_names, pool.oof, event, time, endpoint),
        "seed_bagged": seed_bagged(component_names, pool.oof, event, time, endpoint),
    }
    best_method = max(methods, key=lambda m: methods[m].oof_cindex if np.isfinite(methods[m].oof_cindex) else -1)
    res = methods[best_method]
    test = res.predict(pool.test)
    return {
        "method": best_method,
        "components": list(res.weights),
        "weights": res.weights,
        "oof_cindex": float(res.oof_cindex),
        "test": test,
        "all_methods_oof": {m: float(r.oof_cindex) if np.isfinite(r.oof_cindex) else None for m, r in methods.items()},
    }


def per_fold_cindex(oof: np.ndarray, splits, event: np.ndarray, time: np.ndarray) -> tuple[float, float]:
    out: list[float] = []
    for s in splits:
        v = oof[s.valid_idx]
        finite = np.isfinite(v)
        if finite.sum() == 0 or event[s.valid_idx][finite].sum() == 0:
            continue
        ci = cindex(event[s.valid_idx][finite], time[s.valid_idx][finite], v[finite]).cindex
        if np.isfinite(ci):
            out.append(ci)
    if not out:
        return float("nan"), float("nan")
    return float(np.std(out)), float(np.min(out))


# ---------------------------------------------------------------------------
# Ablation specifications
# ---------------------------------------------------------------------------

_HEPATIC_BASELINE = {
    "phase3_current_state_v2::hepatic__rsf__s0",
    "phase3_current_state_v2::hepatic__rsf__s1",
    "phase3_current_state_v2::hepatic__xgb_aft__s0",
}
_DEATH_BASELINE = {
    "phase3_current_state_v2::death__rsf__s0",
    "phase3_current_state_v2::death__rsf__s1",
    "phase3_current_state_v2::death__xgb_cox__s0",
    "phase3_current_state_v2::death__xgb_cox__s1",
    "phase3_current_state_v2::death__xgb_aft__s0",
    "phase3_current_state_v2::death__xgb_aft__s1",
    "phase3_current_state_v2::death__lgbm_binary__s0",
    "phase3_current_state_v2::death__lgbm_binary__s1",
    "phase3_current_state_v2::death__xgb_binary__s0",
}


def _filter_models(comps: set[str], remove: tuple[str, ...]) -> list[str]:
    out = []
    for c in comps:
        m = c.split("::", 1)[1].split("__")[1]
        if m in remove:
            continue
        out.append(c)
    return out


def _drop_correlated(comps: list[str], pool: Pool, threshold: float = 0.98) -> list[str]:
    """Drop the *lower-individual-C-index* of any pair with rank-corr above threshold."""
    if len(comps) <= 1:
        return list(comps)
    R = pd.DataFrame({c: pd.Series(pool.oof[c]).rank(pct=True, na_option="keep") for c in comps})
    corr = R.corr(method="spearman").to_numpy()
    keep = list(comps)
    drop: set[str] = set()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if corr[i, j] > threshold and comps[i] not in drop and comps[j] not in drop:
                drop.add(comps[j])  # later index dropped
    return [c for c in keep if c not in drop]


def _retrain_without_groups(
    drop_groups: tuple[str, ...],
    endpoint_name: str,
    label: str,
    pool: Pool,
) -> dict:
    """Retrain RSF on current_state_v2 with selected feature groups removed.

    We use a single best-individual model (RSF) for both endpoints to keep the
    cost manageable. The label of the resulting OOF/test column matches the
    ablation label so it slots into the report.
    """
    from .features.current_state_v2 import current_state_v2 as build_csv2
    from .data_loading import load_dataset
    from .features import _align as align_pair  # type: ignore[attr-defined]
    from .features import FeatureSet
    from .targets import build_death_endpoint, build_hepatic_endpoint
    from .features.current_state_v2 import (  # noqa: F401  (uses helper module)
        _interactions,
        _latest_per_stem,
    )
    from .features.refined_longitudinal import (  # noqa: F401
        clinical_scores_dynamic,
        longitudinal_no_followup_proxies,
    )
    from .decompose_current_state_v2 import classify_feature

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    ep = hep if endpoint_name == "hepatic" else death

    Xtr_full = build_csv2(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte_full = build_csv2(ds.test_df, ds.visit_columns, ds.age_visit_cols)

    keep_cols = [c for c in Xtr_full.columns if classify_feature(c) not in drop_groups]
    Xtr = Xtr_full[keep_cols]
    Xte = Xte_full.reindex(columns=keep_cols)

    oof = np.full(pool.n_train, np.nan)
    test_sum = np.zeros(pool.n_test)
    test_n = 0
    for s in pool.splits:
        tr_idx = s.train_idx
        va_idx = s.valid_idx
        if ep.event[tr_idx].sum() == 0:
            continue
        m = build_model("rsf", {"n_estimators": 250, "max_depth": 7, "min_samples_leaf": 4, "random_state": 0})
        fold_mask = pd.Series(False, index=Xtr.index)
        fold_mask.iloc[tr_idx] = True
        try:
            m.fit(Xtr, ep, mask=fold_mask)
            oof[va_idx] = m.predict_risk(Xtr.iloc[va_idx])
            test_sum += m.predict_risk(Xte)
            test_n += 1
        except Exception as e:  # noqa: BLE001
            _LOG.warning("ablation %s fold (%d,%d) failed: %s", label, s.repeat, s.fold, e)
    test = test_sum / test_n if test_n else np.full(pool.n_test, np.nan)
    finite = np.isfinite(oof)
    if finite.sum() == 0 or ep.event[finite].sum() == 0:
        return {"method": "rsf_drop", "components": [label], "weights": {label: 1.0},
                "oof_cindex": float("nan"), "test": test}
    ci = cindex(ep.event[finite], ep.time[finite], oof[finite]).cindex
    pool.oof[label] = oof
    pool.test[label] = test
    return {"method": "rsf_drop", "components": [label], "weights": {label: 1.0},
            "oof_cindex": float(ci), "test": test}


# ---------------------------------------------------------------------------
# Top-level ablation driver
# ---------------------------------------------------------------------------

def run_all_ablations() -> pd.DataFrame:
    pool, _ = load_phase3_pool()
    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)

    rows: list[dict] = []

    def _record(label: str, hep_res: dict, dea_res: dict, note: str):
        h_oof = hep_res["oof_cindex"]
        d_oof = dea_res["oof_cindex"]
        h_oof_arr = pd.Series(np.full(pool.n_train, np.nan))
        # Build the OOF arrays from weights
        from .models.ensemble import rank_average
        h_arr = rank_average({c: pool.oof[c] for c in hep_res["weights"]}, weights=hep_res["weights"]) if hep_res["weights"] else np.full(pool.n_train, np.nan)
        d_arr = rank_average({c: pool.oof[c] for c in dea_res["weights"]}, weights=dea_res["weights"]) if dea_res["weights"] else np.full(pool.n_train, np.nan)
        h_std, h_min = per_fold_cindex(h_arr, pool.splits, hep.event, hep.time)
        d_std, d_min = per_fold_cindex(d_arr, pool.splits, death.event, death.time)
        # Rank correlation with current_state_v2 final OOF (rebuild).
        meta_path = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_current_state_v2*.json"))[-1]
        meta = json.loads(meta_path.read_text())
        full_h = rank_average({c: pool.oof[c] for c in meta["hepatic"]["weights"]}, weights=meta["hepatic"]["weights"])
        full_d = rank_average({c: pool.oof[c] for c in meta["death"]["weights"]}, weights=meta["death"]["weights"])

        def _rho(a, b):
            return float(pd.Series(a).rank(pct=True).corr(pd.Series(b).rank(pct=True), method="spearman"))

        rho_h = _rho(h_arr, full_h)
        rho_d = _rho(d_arr, full_d)
        rows.append({
            "label": label,
            "note": note,
            "method_hepatic": hep_res["method"],
            "method_death": dea_res["method"],
            "n_components_hep": len(hep_res["components"]),
            "n_components_dea": len(dea_res["components"]),
            "oof_hepatic": h_oof,
            "oof_death": d_oof,
            "oof_weighted": weighted_score(h_oof, d_oof) if (np.isfinite(h_oof) and np.isfinite(d_oof)) else float("nan"),
            "fold_std_hep": h_std,
            "fold_min_hep": h_min,
            "fold_std_dea": d_std,
            "fold_min_dea": d_min,
            "rank_corr_with_current_state_v2_hep": rho_h,
            "rank_corr_with_current_state_v2_dea": rho_d,
        })

    # Recover full pools per endpoint.
    hep_pool = list(_HEPATIC_BASELINE)
    dea_pool = list(_DEATH_BASELINE)

    # Baseline (the candidate's actual ensemble).
    h0 = best_ensemble(hep_pool, pool, hep.event, hep.time, "hepatic")
    d0 = best_ensemble(dea_pool, pool, death.event, death.time, "death")
    _record("baseline_current_state_v2", h0, d0, "Existing best per endpoint method.")

    # A. leave-one-component-out (loop)
    for c in hep_pool:
        sub = [x for x in hep_pool if x != c]
        h = best_ensemble(sub, pool, hep.event, hep.time, "hepatic")
        _record(f"A_loo_hep_{c.split('__')[1]}_{c.split('__')[2] if c.count('__')>=2 else ''}",
                h, d0, f"Hepatic leave-out: {c}")
    for c in dea_pool:
        sub = [x for x in dea_pool if x != c]
        d = best_ensemble(sub, pool, death.event, death.time, "death")
        _record(f"A_loo_dea_{c.split('__')[1]}_{c.split('__')[2] if c.count('__')>=2 else ''}",
                h0, d, f"Death leave-out: {c}")

    # B. remove RSF
    h = best_ensemble(_filter_models(set(hep_pool), ("rsf",)), pool, hep.event, hep.time, "hepatic")
    d = best_ensemble(_filter_models(set(dea_pool), ("rsf",)), pool, death.event, death.time, "death")
    _record("B_remove_rsf", h, d, "Both endpoints, RSF dropped.")

    # C. remove XGB Cox
    h = best_ensemble(_filter_models(set(hep_pool), ("xgb_cox",)), pool, hep.event, hep.time, "hepatic")
    d = best_ensemble(_filter_models(set(dea_pool), ("xgb_cox",)), pool, death.event, death.time, "death")
    _record("C_remove_xgb_cox", h, d, "Both endpoints, xgb_cox dropped.")

    # D. remove LightGBM/CatBoost binary
    h = best_ensemble(_filter_models(set(hep_pool), ("lgbm_binary", "catboost_binary", "xgb_binary")), pool, hep.event, hep.time, "hepatic")
    d = best_ensemble(_filter_models(set(dea_pool), ("lgbm_binary", "catboost_binary", "xgb_binary")), pool, death.event, death.time, "death")
    _record("D_remove_binary_classifiers", h, d, "Drop lgbm/cat/xgb binary classifiers.")

    # E. remove visit-history/care-stage features (retrain RSF)
    h = _retrain_without_groups(("visit_history_current_state",), "hepatic", "E_no_visit_history_hep", pool)
    d = _retrain_without_groups(("visit_history_current_state",), "death", "E_no_visit_history_dea", pool)
    _record("E_no_visit_history", h, d, "Drop n_visits/age_last/etc; retrain RSF only.")

    # F. remove missingness features (retrain RSF)
    h = _retrain_without_groups(("missingness_pattern",), "hepatic", "F_no_missingness_hep", pool)
    d = _retrain_without_groups(("missingness_pattern",), "death", "F_no_missingness_dea", pool)
    _record("F_no_missingness", h, d, "Drop miss_* columns; retrain RSF only.")

    # G. remove biomarker trajectory features = drop labs_biomarker AND nit_liver_stiffness trajectory cols
    # We approximate by dropping {labs_biomarker, nit_liver_stiffness} groups.
    h = _retrain_without_groups(("labs_biomarker", "nit_liver_stiffness"), "hepatic", "G_no_biomarker_traj_hep", pool)
    d = _retrain_without_groups(("labs_biomarker", "nit_liver_stiffness"), "death", "G_no_biomarker_traj_dea", pool)
    _record("G_no_biomarker_trajectories", h, d, "Drop labs+NIT trajectory features; demographics/scores/missing/cadence remain.")

    # H. remove death components with high weight but high SD (we have a few weak ones)
    weak_death = {"phase3_current_state_v2::death__lgbm_binary__s0",
                  "phase3_current_state_v2::death__lgbm_binary__s1",
                  "phase3_current_state_v2::death__xgb_binary__s0"}
    d = best_ensemble([c for c in dea_pool if c not in weak_death], pool, death.event, death.time, "death")
    _record("H_remove_unstable_death", h0, d, "Drop binary classifiers from death pool.")

    # I. remove highly-correlated > 0.98
    hep_dec = _drop_correlated(hep_pool, pool, threshold=0.98)
    dea_dec = _drop_correlated(dea_pool, pool, threshold=0.98)
    h = best_ensemble(hep_dec, pool, hep.event, hep.time, "hepatic")
    d = best_ensemble(dea_dec, pool, death.event, death.time, "death")
    _record("I_drop_corr_gt_0_98", h, d,
            f"Drop one of any rank-corr>0.98 pair. hep_kept={len(hep_dec)} dea_kept={len(dea_dec)}.")

    # J. equal/greedy/cv_weighted comparison on full pool (per endpoint)
    for method_name, method_fn in [
        ("J_equal", equal_weight),
        ("J_greedy", greedy),
        ("J_cv_weighted", cv_weighted),
    ]:
        h_res = method_fn(hep_pool, pool.oof, hep.event, hep.time, "hepatic")
        d_res = method_fn(dea_pool, pool.oof, death.event, death.time, "death")
        h = {"method": method_name, "components": list(h_res.weights), "weights": h_res.weights,
             "oof_cindex": float(h_res.oof_cindex), "test": h_res.predict(pool.test)}
        d = {"method": method_name, "components": list(d_res.weights), "weights": d_res.weights,
             "oof_cindex": float(d_res.oof_cindex), "test": d_res.predict(pool.test)}
        _record(method_name, h, d, "Force a single ensemble method on the full pool.")

    df = pd.DataFrame(rows)
    df = df.sort_values("oof_weighted", ascending=False)
    return df


def main() -> None:
    df = run_all_ablations()
    out_csv = cfg.REPORTS_DIR / "phase3_5_ablations.csv"
    out_md = cfg.REPORTS_DIR / "phase3_5_ablations.md"
    df.to_csv(out_csv, index=False)
    lines = [
        "# Phase 3.5 — `current_state_v2` ablations\n",
        "All ablations operate on the same hepatic-stratified repeated 5x3 CV splits "
        "and use the existing OOF/test predictions when possible. Group-removal "
        "ablations (E, F, G) retrain a single RSF on `current_state_v2` minus the "
        "indicated feature groups.\n",
        df.to_markdown(index=False, floatfmt=".4f"),
        "\n## Reading guide\n",
        "- `oof_weighted` is the score we ultimately care about; `fold_std`/`fold_min` "
        "tell you whether the gain is fragile.\n",
        "- `rank_corr_with_current_state_v2_*` indicates how much new information the "
        "variant carries vs the current public best (lower = more diversity, but "
        "also more risk of drifting away).\n",
    ]
    out_md.write_text("\n".join(lines))
    _LOG.info("wrote %s and %s", out_md, out_csv)


if __name__ == "__main__":
    main()
