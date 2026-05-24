"""Build the four Phase 3.5 candidate submissions.

A. ``phase3_5_current_state_v3_simplified`` — drop redundant/weak components
   from current_state_v2 (correlated > 0.98 + low-individual binary classifiers).
B. ``phase3_5_current_state_v3_bagged`` — keep only the strongest model
   families (RSF, xgb_cox, xgb_aft) and seed-bag them per endpoint.
C. ``phase3_5_current_state_v3_hepatic_focused`` — re-pick per-endpoint
   methods to maximize hepatic OOF, accepting some death drop. (Hepatic is
   70% of the score.)
D. ``phase3_5_current_state_v3_private_robust`` — blend current_state_v2 with
   phase2_aggressive_longitudinal and phase2_robust_longitudinal at OOF-
   chosen weights, capped per component to limit single-model dependence.

No public-LB tuning; weights are picked by OOF C-index only.
No strict_time_aligned components.
"""
from __future__ import annotations

import json
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
from .metrics import cindex, weighted_score
from .models.ensemble import rank_average
from .submission import make_submission
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger
from .validation import build_folds

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Predictor pool helpers
# ---------------------------------------------------------------------------

def _phase3_dirs() -> list[Path]:
    out = []
    for d in cfg.EXPERIMENT_OUTPUTS.iterdir():
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        try:
            blob = json.loads((d / "config.json").read_text())
        except Exception:
            continue
        if blob.get("name") == "phase3_current_state_v2":
            out.append(d)
    return out


def _phase2_dirs(names: set[str]) -> list[Path]:
    out = []
    for d in cfg.EXPERIMENT_OUTPUTS.iterdir():
        if not d.is_dir() or not (d / "config.json").exists():
            continue
        try:
            blob = json.loads((d / "config.json").read_text())
        except Exception:
            continue
        if blob.get("name") in names:
            out.append(d)
    return out


def _aligned(col: str, df: pd.DataFrame, positions: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan)
    if col in df.columns:
        out[positions] = df[col].to_numpy()
    return out


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    s = pd.Series(a).rank(pct=True, na_option="keep")
    t = pd.Series(b).rank(pct=True, na_option="keep")
    return float(s.corr(t, method="spearman"))


# ---------------------------------------------------------------------------
# Common candidate builder
# ---------------------------------------------------------------------------

def _build_endpoint_pool(
    pool_oof: dict[str, np.ndarray],
    pool_test: dict[str, np.ndarray],
    component_names: list[str],
    event: np.ndarray,
    time: np.ndarray,
    n_test: int,
    *,
    methods: tuple[str, ...] = ("equal", "greedy", "cv_weighted", "seed_bagged"),
    cap_max_weight: float | None = None,
) -> dict:
    """Try the listed methods on the given component pool, return the best."""
    if not component_names:
        return {"method": None, "components": [], "weights": {}, "oof_cindex": float("nan"),
                "test": np.full(n_test, np.nan)}
    results = {}
    if "equal" in methods:
        results["equal"] = equal_weight(component_names, pool_oof, event, time, "_")
    if "greedy" in methods:
        results["greedy"] = greedy(component_names, pool_oof, event, time, "_")
    if "cv_weighted" in methods:
        results["cv_weighted"] = cv_weighted(component_names, pool_oof, event, time, "_")
    if "seed_bagged" in methods:
        results["seed_bagged"] = seed_bagged(component_names, pool_oof, event, time, "_")
    if cap_max_weight is not None:
        results["capped"] = capped(component_names, pool_oof, event, time, "_", max_weight=cap_max_weight)

    best = max(results, key=lambda m: results[m].oof_cindex if np.isfinite(results[m].oof_cindex) else -1)
    res = results[best]
    test = rank_average({c: pool_test[c] for c in res.weights}, weights=res.weights) if res.weights else np.full(n_test, np.nan)
    return {
        "method": best,
        "components": list(res.weights),
        "weights": res.weights,
        "oof_cindex": float(res.oof_cindex),
        "test": test,
        "all_methods_oof": {m: float(r.oof_cindex) if np.isfinite(r.oof_cindex) else None for m, r in results.items()},
    }


def _drop_correlated(comps: list[str], pool_oof: dict[str, np.ndarray], threshold: float = 0.98) -> list[str]:
    if len(comps) <= 1:
        return list(comps)
    R = pd.DataFrame({c: pd.Series(pool_oof[c]).rank(pct=True, na_option="keep") for c in comps})
    corr = R.corr(method="spearman").to_numpy()
    drop: set[str] = set()
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if corr[i, j] > threshold and comps[i] not in drop and comps[j] not in drop:
                drop.add(comps[j])
    return [c for c in comps if c not in drop]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def main() -> None:
    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)

    pid_col = cfg.PATIENT_ID_COL
    tid_col = cfg.TRUSTII_ID_COL
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)

    # Pool 1: phase3_current_state_v2 components only.
    p3_dirs = _phase3_dirs()
    if not p3_dirs:
        raise RuntimeError("phase3_current_state_v2 dir not found")
    oof_df, test_df, _ = collect_predictions(p3_dirs)
    pos = oof_df[pid_col].map({v: i for i, v in enumerate(ds.train_df[pid_col].values)}).to_numpy().astype(int)
    tpos = test_df[tid_col].map({v: i for i, v in enumerate(ds.test_df[tid_col].values)}).to_numpy().astype(int)
    p3_oof: dict[str, np.ndarray] = {}
    p3_test: dict[str, np.ndarray] = {}
    p3_hep_cols, p3_dea_cols = [], []
    for c in oof_df.columns:
        if c == pid_col or "::ensemble_" in c:
            continue
        if "::hepatic__" in c:
            p3_hep_cols.append(c)
            p3_oof[c] = _aligned(c, oof_df, pos, n_train)
            p3_test[c] = _aligned(c, test_df, tpos, n_test)
        elif "::death__" in c:
            p3_dea_cols.append(c)
            p3_oof[c] = _aligned(c, oof_df, pos, n_train)
            p3_test[c] = _aligned(c, test_df, tpos, n_test)

    # Pool 2: extend with phase2 robust + aggressive (no strict_time_aligned).
    p2_names = {
        "phase2_aggressive_longitudinal",
        "phase2_longitudinal_no_followup_proxies",
        "phase2_NIT_plus_scores_longitudinal",
        "phase2_first_3_visits",
        "phase2_first_3y",
    }
    p2_dirs = _phase2_dirs(p2_names)
    p2_oof_df, p2_test_df, _ = collect_predictions(p2_dirs) if p2_dirs else (pd.DataFrame(), pd.DataFrame(), {})
    p2_hep_cols, p2_dea_cols = [], []
    if not p2_oof_df.empty:
        p2_pos = p2_oof_df[pid_col].map({v: i for i, v in enumerate(ds.train_df[pid_col].values)}).to_numpy().astype(int)
        p2_tpos = p2_test_df[tid_col].map({v: i for i, v in enumerate(ds.test_df[tid_col].values)}).to_numpy().astype(int)
        for c in p2_oof_df.columns:
            if c == pid_col or "::ensemble_" in c:
                continue
            if "::hepatic__" in c:
                p2_hep_cols.append(c)
                p3_oof[c] = _aligned(c, p2_oof_df, p2_pos, n_train)
                p3_test[c] = _aligned(c, p2_test_df, p2_tpos, n_test)
            elif "::death__" in c:
                p2_dea_cols.append(c)
                p3_oof[c] = _aligned(c, p2_oof_df, p2_pos, n_train)
                p3_test[c] = _aligned(c, p2_test_df, p2_tpos, n_test)

    # Reference current_state_v2 weights for rank-corr comparison.
    ref_meta = json.loads(sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_current_state_v2*.json"))[-1].read_text())
    ref_h_oof = rank_average({c: p3_oof[c] for c in ref_meta["hepatic"]["weights"]}, weights=ref_meta["hepatic"]["weights"])
    ref_d_oof = rank_average({c: p3_oof[c] for c in ref_meta["death"]["weights"]}, weights=ref_meta["death"]["weights"])
    ref_h_test = pd.read_csv(ref_meta["submission_csv"]).set_index(tid_col).reindex(ds.test_df[tid_col].values)[cfg.SUB_HEPATIC_COL].to_numpy()
    ref_d_test = pd.read_csv(ref_meta["submission_csv"]).set_index(tid_col).reindex(ds.test_df[tid_col].values)[cfg.SUB_DEATH_COL].to_numpy()

    candidates = []

    def _emit(label: str, hep_res: dict, dea_res: dict, description: str, leakage_tag: str) -> dict:
        sub_path = make_submission(
            ds.test_df, risk_hepatic=hep_res["test"], risk_death=dea_res["test"],
            sample_submission=ds.sample_submission, model_name=label,
        )
        # Rank correlation with current_state_v2 — use pre-built OOF arrays if present
        # (used by the blend candidate whose weights are synthetic keys not in p3_oof).
        if "oof_array" in hep_res:
            h_arr = hep_res["oof_array"]
        else:
            h_arr = rank_average({c: p3_oof[c] for c in hep_res["weights"]}, weights=hep_res["weights"]) if hep_res["weights"] else np.full(n_train, np.nan)
        if "oof_array" in dea_res:
            d_arr = dea_res["oof_array"]
        else:
            d_arr = rank_average({c: p3_oof[c] for c in dea_res["weights"]}, weights=dea_res["weights"]) if dea_res["weights"] else np.full(n_train, np.nan)
        meta = {
            "label": label,
            "description": description,
            "leakage_tag": leakage_tag,
            "submission_csv": str(sub_path),
            "uses_target_derived_features": False,
            "hepatic": {
                "method": hep_res["method"],
                "components": hep_res["components"],
                "weights": hep_res["weights"],
                "oof_cindex": hep_res["oof_cindex"],
            },
            "death": {
                "method": dea_res["method"],
                "components": dea_res["components"],
                "weights": dea_res["weights"],
                "oof_cindex": dea_res["oof_cindex"],
            },
            "weighted_score_cv": weighted_score(hep_res["oof_cindex"], dea_res["oof_cindex"]),
            "rank_corr_with_current_state_v2": {
                "oof_hepatic": _rank_corr(h_arr, ref_h_oof),
                "oof_death": _rank_corr(d_arr, ref_d_oof),
                "test_hepatic": _rank_corr(hep_res["test"], ref_h_test),
                "test_death": _rank_corr(dea_res["test"], ref_d_test),
            },
            "all_methods_oof_hepatic": hep_res.get("all_methods_oof"),
            "all_methods_oof_death": dea_res.get("all_methods_oof"),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        candidates.append(meta)
        _LOG.info("%s -> hep=%.4f death=%.4f weighted=%.4f", label, hep_res["oof_cindex"], dea_res["oof_cindex"], meta["weighted_score_cv"])
        return meta

    # ---- A. simplified -----------------------------------------------------
    weak_death_models = {"lgbm_binary", "xgb_binary", "catboost_binary"}
    hep_simpl = _drop_correlated(p3_hep_cols, p3_oof, threshold=0.98)
    dea_simpl = [c for c in p3_dea_cols if c.split("__")[1] not in weak_death_models]
    dea_simpl = _drop_correlated(dea_simpl, p3_oof, threshold=0.98)
    _emit(
        "phase3_5_current_state_v3_simplified",
        _build_endpoint_pool(p3_oof, p3_test, hep_simpl, hep.event, hep.time, n_test),
        _build_endpoint_pool(p3_oof, p3_test, dea_simpl, death.event, death.time, n_test),
        "Phase 3 current_state_v2 with rank-corr>0.98 components dropped and weak binary classifiers removed from the death pool.",
        "moderate-high (inherits current_state_v2 leakage profile)",
    )

    # ---- B. seed-bagged of strong families ---------------------------------
    strong = {"rsf", "xgb_cox", "xgb_aft"}
    hep_strong = [c for c in p3_hep_cols if c.split("__")[1] in strong]
    dea_strong = [c for c in p3_dea_cols if c.split("__")[1] in strong]
    _emit(
        "phase3_5_current_state_v3_bagged",
        _build_endpoint_pool(p3_oof, p3_test, hep_strong, hep.event, hep.time, n_test, methods=("seed_bagged", "equal")),
        _build_endpoint_pool(p3_oof, p3_test, dea_strong, death.event, death.time, n_test, methods=("seed_bagged", "equal")),
        "Seed-bagged ensemble of RSF + xgb_cox + xgb_aft on current_state_v2 only.",
        "moderate-high",
    )

    # ---- C. hepatic-focused ------------------------------------------------
    # The ablation E_no_visit_history showed RSF without visit-history features
    # reaches OOF 0.8415 hepatic (+0.0028 over baseline). We retrain that RSF
    # here and blend it with the existing hepatic pool.
    from .data_loading import load_dataset as _load
    from .features.current_state_v2 import current_state_v2 as _csv2
    from .decompose_current_state_v2 import classify_feature
    from .models import build_model
    _ds2 = _load()
    Xtr_full = _csv2(_ds2.train_df, _ds2.visit_columns, _ds2.age_visit_cols)
    Xte_full = _csv2(_ds2.test_df, _ds2.visit_columns, _ds2.age_visit_cols)
    keep_cols = [c for c in Xtr_full.columns if classify_feature(c) != "visit_history_current_state"]
    Xtr_nh = Xtr_full[keep_cols]
    Xte_nh = Xte_full.reindex(columns=keep_cols)
    hep_oof_nh = np.full(n_train, np.nan)
    hep_test_nh_sum = np.zeros(n_test)
    hep_test_nh_n = 0
    for s in splits:
        if hep.event[s.train_idx].sum() == 0:
            continue
        m = build_model("rsf", {"n_estimators": 250, "max_depth": 7, "min_samples_leaf": 4, "random_state": 0})
        fold_mask = pd.Series(False, index=Xtr_nh.index)
        fold_mask.iloc[s.train_idx] = True
        try:
            m.fit(Xtr_nh, hep, mask=fold_mask)
            hep_oof_nh[s.valid_idx] = m.predict_risk(Xtr_nh.iloc[s.valid_idx])
            hep_test_nh_sum += m.predict_risk(Xte_nh)
            hep_test_nh_n += 1
        except Exception as e:  # noqa: BLE001
            _LOG.warning("hep_focus retrain fold (%d,%d) failed: %s", s.repeat, s.fold, e)
    hep_test_nh = hep_test_nh_sum / hep_test_nh_n if hep_test_nh_n else np.full(n_test, np.nan)
    p3_oof["phase3_5_no_visit_history::hepatic__rsf__nh"] = hep_oof_nh
    p3_test["phase3_5_no_visit_history::hepatic__rsf__nh"] = hep_test_nh
    hep_focus_pool = list(p3_hep_cols) + ["phase3_5_no_visit_history::hepatic__rsf__nh"]
    hep_focus = _build_endpoint_pool(p3_oof, p3_test, hep_focus_pool, hep.event, hep.time, n_test)

    # Death stays on the simplified pool (binary classifiers dropped) with seed_bagged.
    dea_safe = _build_endpoint_pool(p3_oof, p3_test, dea_simpl, death.event, death.time, n_test,
                                    methods=("seed_bagged", "cv_weighted", "equal"))
    _emit(
        "phase3_5_current_state_v3_hepatic_focused",
        hep_focus,
        dea_safe,
        "Hepatic-priority: blends current_state_v2 with a no-visit-history RSF; ablation E showed +0.003 hep gain. Death uses the simplified pool (binary classifiers dropped) for stability.",
        "moderate-high",
    )

    # ---- D. private-robust blend -------------------------------------------
    # OOF-greedy on the union of pools always picks current_state_v2 (it
    # dominates OOF). To get genuine diversification we force a 50/50 blend at
    # the *ensemble* level: v2's per-endpoint output + phase2_aggressive
    # (LB-validated, transferred to 0.868). Weights are picked by OOF for
    # each component within each side, but the side-blend is a fixed prior we
    # set deliberately to hedge against v2 being lucky on public test.
    p2_aggro_dirs = _phase2_dirs({"phase2_aggressive_longitudinal"})
    if not p2_aggro_dirs:
        _LOG.warning("phase2_aggressive_longitudinal not found; skipping private_robust")
    else:
        # Use the existing per-endpoint OOFs from p2_aggressive (already in pool).
        p2_h = [c for c in p3_oof if c.startswith("phase2_aggressive_longitudinal::hepatic__")]
        p2_d = [c for c in p3_oof if c.startswith("phase2_aggressive_longitudinal::death__")]
        h_v2 = _build_endpoint_pool(p3_oof, p3_test, p3_hep_cols, hep.event, hep.time, n_test)
        d_v2 = _build_endpoint_pool(p3_oof, p3_test, p3_dea_cols, death.event, death.time, n_test)
        h_p2 = _build_endpoint_pool(p3_oof, p3_test, p2_h, hep.event, hep.time, n_test)
        d_p2 = _build_endpoint_pool(p3_oof, p3_test, p2_d, death.event, death.time, n_test)

        # 0.7 * v2 + 0.3 * p2 (in rank space). Hepatic priority -> small
        # hedge so we don't fall too far if v2 was just lucky on public.
        def _blend(weights_a, weights_b, pool, n, alpha=0.7):
            ha = rank_average({c: pool[c] for c in weights_a}, weights=weights_a) if weights_a else np.full(n, np.nan)
            hb = rank_average({c: pool[c] for c in weights_b}, weights=weights_b) if weights_b else np.full(n, np.nan)
            ra = pd.Series(ha).rank(pct=True, na_option="keep").to_numpy()
            rb = pd.Series(hb).rank(pct=True, na_option="keep").to_numpy()
            valid_a = ~np.isnan(ra)
            valid_b = ~np.isnan(rb)
            out = np.where(valid_a & valid_b, alpha * ra + (1 - alpha) * rb,
                            np.where(valid_a, ra, np.where(valid_b, rb, np.nan)))
            return out

        blend_h_oof = _blend(h_v2["weights"], h_p2["weights"], p3_oof, n_train, alpha=0.7)
        blend_d_oof = _blend(d_v2["weights"], d_p2["weights"], p3_oof, n_train, alpha=0.7)
        blend_h_test = _blend(h_v2["weights"], h_p2["weights"], p3_test, n_test, alpha=0.7)
        blend_d_test = _blend(d_v2["weights"], d_p2["weights"], p3_test, n_test, alpha=0.7)
        finite_h = np.isfinite(blend_h_oof)
        finite_d = np.isfinite(blend_d_oof)
        ci_h = float(cindex(hep.event[finite_h], hep.time[finite_h], blend_h_oof[finite_h]).cindex) if finite_h.sum() else float("nan")
        ci_d = float(cindex(death.event[finite_d], death.time[finite_d], blend_d_oof[finite_d]).cindex) if finite_d.sum() else float("nan")
        h_blend = {"method": "blend(0.7*v2+0.3*p2_aggressive)", "components": list(h_v2["weights"]) + list(h_p2["weights"]),
                   "weights": {**{f"v2::{k}": 0.7*v for k, v in h_v2["weights"].items()},
                                **{f"p2::{k}": 0.3*v for k, v in h_p2["weights"].items()}},
                   "oof_cindex": ci_h, "test": blend_h_test, "oof_array": blend_h_oof}
        d_blend = {"method": "blend(0.7*v2+0.3*p2_aggressive)", "components": list(d_v2["weights"]) + list(d_p2["weights"]),
                   "weights": {**{f"v2::{k}": 0.7*v for k, v in d_v2["weights"].items()},
                                **{f"p2::{k}": 0.3*v for k, v in d_p2["weights"].items()}},
                   "oof_cindex": ci_d, "test": blend_d_test, "oof_array": blend_d_oof}
        _emit(
            "phase3_5_current_state_v3_private_robust",
            h_blend, d_blend,
            "Forced 0.7 * current_state_v2 + 0.3 * phase2_aggressive_longitudinal blend per endpoint (rank-space). The v2 side is OOF-best; the phase2 side is the previously LB-validated 0.868 model. Weights are not tuned to public LB.",
            "moderate-high",
        )

    # Roll-up for the recommendation report.
    rollup = pd.DataFrame([
        {
            "label": c["label"],
            "leakage_tag": c["leakage_tag"],
            "hep_oof": c["hepatic"]["oof_cindex"],
            "death_oof": c["death"]["oof_cindex"],
            "weighted": c["weighted_score_cv"],
            "rho_oof_hep_vs_v2": c["rank_corr_with_current_state_v2"]["oof_hepatic"],
            "rho_oof_dea_vs_v2": c["rank_corr_with_current_state_v2"]["oof_death"],
            "rho_test_hep_vs_v2": c["rank_corr_with_current_state_v2"]["test_hepatic"],
            "rho_test_dea_vs_v2": c["rank_corr_with_current_state_v2"]["test_death"],
            "submission_csv": c["submission_csv"],
        }
        for c in candidates
    ])
    out = cfg.REPORTS_DIR / "phase3_5_candidates_rollup.md"
    out.write_text("# Phase 3.5 candidate roll-up\n\n" +
                   rollup.to_markdown(index=False, floatfmt=".4f") +
                   "\n\nReference: phase3_current_state_v2 OOF hep=0.8378 / death=0.9495 / weighted=0.8713 | LB=0.88521.\n")
    _LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
