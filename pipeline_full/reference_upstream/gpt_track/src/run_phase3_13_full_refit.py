"""Phase 3.13 — full-train refit experiment.

Question: does retraining the components of ``phase3_10_horizon_blend_v2`` on
the full training set (no CV folds) improve test ranking, or is the existing
CV-fold ensemble (each base model trained 15 times on different folds and
averaged) more robust?

Anchor: ``phase3_10_horizon_blend_v2`` (LB **0.91093**).

The anchor's structure (from its JSON sidecar) is:

- hepatic = 0.79 · v3_hepatic_focused (RSF on ``current_state_v2_no_visit_history``)
            + 0.119 · ``NIT_plus_scores__hepatic__h1__lgbm_binary``
            + 0.091 · ``v3_hepatic_schema__hepatic__h3__lgbm_binary__s4``
- death   = 0.79 · v3_hepatic_focused (4-model seed-bag on ``current_state_v2``)
            + 0.119 · ``current_state_v2__death__h5__catboost_binary__s3``
            + 0.091 · ``NIT_plus_scores__death__h4__catboost_binary``

We retrain *each* base model on the full training data (no folds) using the
same feature set and hyperparameters, then rank-blend the retrained
components with the anchor's weights to get a "full_retrain_only" prediction.
We then evaluate four candidates:

A. ``full_retrain_only``                = full retrain test predictions only
B. ``cv90_full10`` = 0.90 anchor + 0.10 full retrain (rank-space)
C. ``cv75_full25`` = 0.75 anchor + 0.25 full retrain
D. ``cv50_full50`` = 0.50 anchor + 0.50 full retrain

Diagnostics: rank correlation per endpoint, prediction-shift distribution,
top-shift patients. Full-retrain has no honest OOF, so we never invent one;
we report agreement with the anchor and let the rank-correlation drive the
recommendation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .features import build_feature_set
from .features.refined_longitudinal import (
    LAB_STEMS,
    NIT_STEMS,
    longitudinal_no_followup_proxies,
)
from .features.hep_focus import current_state_v2_no_visit_history
from .horizon_targets import build_horizon_labels
from .models import build_model
from .models._preprocess import fill_for_tree
from .models.ensemble import rank_average, to_rank
from .submission import make_submission
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True, na_option="keep").to_numpy()


def _blend_two(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    ra, rb = _rank(a), _rank(b)
    valid_a, valid_b = np.isfinite(ra), np.isfinite(rb)
    return np.where(valid_a & valid_b, alpha * ra + (1 - alpha) * rb,
                    np.where(valid_a, ra, np.where(valid_b, rb, np.nan)))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size != b.size:
        return float("nan")
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 5:
        return float("nan")
    return float(df["a"].rank(pct=True).corr(df["b"].rank(pct=True), method="spearman"))


# ---------------------------------------------------------------------------
# Feature sets (the four also used in Phase 3.10 / 3.12)
# ---------------------------------------------------------------------------

def assemble_feature_sets(ds) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}

    fs1 = build_feature_set("current_state_v2", ds)
    out["current_state_v2"] = {"train": fs1.X_train, "test": fs1.X_test}

    fs2 = build_feature_set("NIT_plus_scores_longitudinal", ds)
    out["NIT_plus_scores"] = {"train": fs2.X_train, "test": fs2.X_test}

    Xtr = longitudinal_no_followup_proxies(ds.train_df, ds.visit_columns, ds.age_visit_cols,
                                           keep_stems=LAB_STEMS + NIT_STEMS)
    Xte = longitudinal_no_followup_proxies(ds.test_df, ds.visit_columns, ds.age_visit_cols,
                                           keep_stems=LAB_STEMS + NIT_STEMS)
    out["biomarker_only"] = {"train": Xtr, "test": Xte}

    fs4 = build_feature_set("current_state_v2_no_visit_history", ds)
    out["v3_hepatic_schema"] = {"train": fs4.X_train, "test": fs4.X_test}

    for name, blob in out.items():
        for k in ("train", "test"):
            blob[k] = blob[k].select_dtypes(include=[np.number])
    return out


# ---------------------------------------------------------------------------
# Full-train refit primitives
# ---------------------------------------------------------------------------

def _train_horizon_full(model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame,
                        label: np.ndarray, mask: np.ndarray, weight: np.ndarray,
                        *, seed: int = 0, use_sample_weight: bool = False) -> np.ndarray:
    Xtr_pre = fill_for_tree(X_train).astype(np.float32, errors="ignore")
    Xte_pre = fill_for_tree(X_test).reindex(columns=Xtr_pre.columns, fill_value=0).astype(np.float32, errors="ignore")
    Xtr_arr = Xtr_pre.to_numpy()
    Xte_arr = Xte_pre.to_numpy()

    keep_idx = np.where(mask)[0]
    ytr = label[keep_idx].astype(int)
    wtr = weight[keep_idx] if use_sample_weight else None
    n_pos = max(int(ytr.sum()), 1)
    n_neg = max(int((1 - ytr).sum()), 1)
    spw = n_neg / n_pos

    if model_name == "lgbm_binary":
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, scale_pos_weight=spw, random_state=seed, verbose=-1,
        )
    elif model_name == "xgb_binary":
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            min_child_weight=1.0, scale_pos_weight=spw,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", verbosity=0, random_state=seed,
        )
    elif model_name == "catboost_binary":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(
            iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
            class_weights=[1.0, n_neg / n_pos], random_seed=seed,
            verbose=0, allow_writing_files=False,
        )
    else:
        raise KeyError(model_name)

    if use_sample_weight and wtr is not None:
        clf.fit(Xtr_arr[keep_idx], ytr, sample_weight=wtr)
    else:
        clf.fit(Xtr_arr[keep_idx], ytr)
    return clf.predict_proba(Xte_arr)[:, 1]


def _train_survival_full(model_name: str, params: dict, X_train: pd.DataFrame,
                         X_test: pd.DataFrame, endpoint) -> np.ndarray:
    m = build_model(model_name, params)
    full_mask = pd.Series(True, index=X_train.index)
    m.fit(X_train, endpoint, mask=full_mask)
    return m.predict_risk(X_test)


# ---------------------------------------------------------------------------
# Anchor reconstruction (for OOF and test alignment)
# ---------------------------------------------------------------------------

def _load_horizon_artifact(label: str) -> tuple[np.ndarray, np.ndarray] | None:
    for root in (cfg.EXPERIMENT_OUTPUTS / "phase3_12_horizon",
                 cfg.EXPERIMENT_OUTPUTS / "phase3_10_horizon",
                 cfg.EXPERIMENT_OUTPUTS / "phase3_9_horizon"):
        run_dir = root / label
        if not run_dir.exists():
            continue
        oof_p = run_dir / "oof.csv"
        test_p = run_dir / "test.csv"
        if oof_p.exists() and test_p.exists():
            o = pd.read_csv(oof_p)["oof"].to_numpy()
            t = pd.read_csv(test_p)["test"].to_numpy()
            return o, t
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    cfg.ensure_dirs()
    out_root = cfg.EXPERIMENT_OUTPUTS / "phase3_13_full_refit"
    out_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)

    feature_sets = assemble_feature_sets(ds)

    # 1. Anchor
    anchor_meta_path = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_10_horizon_blend_v2.json"))[-1]
    anchor_csv = anchor_meta_path.with_suffix(".csv")
    anchor_meta = json.loads(anchor_meta_path.read_text())
    anchor_pub = pd.read_csv(anchor_csv)
    a_h_test = anchor_pub[cfg.SUB_HEPATIC_COL].to_numpy()
    a_d_test = anchor_pub[cfg.SUB_DEATH_COL].to_numpy()
    weights_h = anchor_meta["weights_hepatic"]
    weights_d = anchor_meta["weights_death"]
    _LOG.info("Anchor: %s", anchor_meta_path.name)
    _LOG.info("Hep weights: %s", weights_h)
    _LOG.info("Dea weights: %s", weights_d)

    # 2. Retrain v3 hepatic component on FULL train.
    _LOG.info("Retraining v3 hepatic (RSF on current_state_v2_no_visit_history) on full train.")
    Xtr_v3hep = current_state_v2_no_visit_history(ds.train_df, ds.visit_columns, ds.age_visit_cols).select_dtypes(include=[np.number])
    Xte_v3hep = current_state_v2_no_visit_history(ds.test_df, ds.visit_columns, ds.age_visit_cols).select_dtypes(include=[np.number])
    # Align test columns to train columns.
    Xte_v3hep = Xte_v3hep.reindex(columns=Xtr_v3hep.columns)
    full_v3_hep_test = _train_survival_full(
        "rsf",
        {"n_estimators": 250, "max_depth": 7, "min_samples_leaf": 4, "random_state": 0},
        Xtr_v3hep, Xte_v3hep, hep,
    )

    # 3. Retrain v3 death components on FULL train (4 models, seed-bagged blend).
    _LOG.info("Retraining v3 death (4-model seed bag on current_state_v2) on full train.")
    Xtr_v3dea = feature_sets["current_state_v2"]["train"]
    Xte_v3dea = feature_sets["current_state_v2"]["test"]
    Xte_v3dea = Xte_v3dea.reindex(columns=Xtr_v3dea.columns)

    # phase3_5 v3 death seed-bag references the phase3_current_state_v2 run
    # whose hyperparameters are recorded in its config.json.
    v3_dea_components = {
        "phase3_current_state_v2::death__rsf__s0":
            ("rsf",     {"n_estimators": 300, "max_depth": 7, "min_samples_leaf": 4, "random_state": 0}),
        "phase3_current_state_v2::death__xgb_cox__s0":
            ("xgb_cox", {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "random_state": 0}),
        "phase3_current_state_v2::death__xgb_aft__s0":
            ("xgb_aft", {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "random_state": 0}),
        "phase3_current_state_v2::death__xgb_cox__s1":
            ("xgb_cox", {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05, "random_state": 1}),
    }
    v3_dea_inner_weights = {
        "phase3_current_state_v2::death__rsf__s0":     0.3653846153846154,
        "phase3_current_state_v2::death__xgb_cox__s0": 0.2692307692307692,
        "phase3_current_state_v2::death__xgb_aft__s0": 0.17307692307692302,
        "phase3_current_state_v2::death__xgb_cox__s1": 0.19230769230769224,
    }
    v3_dea_full_preds: dict[str, np.ndarray] = {}
    for cname, (mname, params) in v3_dea_components.items():
        try:
            pred = _train_survival_full(mname, params, Xtr_v3dea, Xte_v3dea, death)
            v3_dea_full_preds[cname] = pred
            _LOG.info("  retrained %s -> n_test=%d", cname, len(pred))
        except Exception as e:  # noqa: BLE001
            _LOG.error("  failed to retrain %s: %s", cname, e)

    full_v3_dea_test = rank_average(v3_dea_full_preds, weights=v3_dea_inner_weights)

    # 4. Retrain horizon components on FULL train.
    _LOG.info("Retraining hepatic horizon components on full train.")

    def _retrain_horizon(label: str, ep_name: str, fs_name: str, model_name: str,
                          horizon: float, seed: int) -> np.ndarray:
        ep = hep if ep_name == "hepatic" else death
        lab = build_horizon_labels(ep, horizon, censored_mode="exclude")
        blob = feature_sets[fs_name]
        return _train_horizon_full(model_name, blob["train"], blob["test"],
                                     lab.label, lab.mask, lab.weight, seed=seed,
                                     use_sample_weight=False)

    # Hepatic horizon components.
    hep_horizon_specs = {
        "NIT_plus_scores__hepatic__h1__lgbm_binary":
            ("hepatic", "NIT_plus_scores",   "lgbm_binary", 1.0, 0),
        "v3_hepatic_schema__hepatic__h3__lgbm_binary__s4":
            ("hepatic", "v3_hepatic_schema", "lgbm_binary", 3.0, 4),
    }
    full_hep_horizon_test: dict[str, np.ndarray] = {}
    for label, (ep_n, fs_n, mn, H, seed) in hep_horizon_specs.items():
        if label not in weights_h:
            continue
        full_hep_horizon_test[label] = _retrain_horizon(label, ep_n, fs_n, mn, H, seed)
        _LOG.info("  retrained %s", label)

    _LOG.info("Retraining death horizon components on full train.")
    dea_horizon_specs = {
        "current_state_v2__death__h5__catboost_binary__s3":
            ("death", "current_state_v2", "catboost_binary", 5.0, 3),
        "NIT_plus_scores__death__h4__catboost_binary":
            ("death", "NIT_plus_scores",  "catboost_binary", 4.0, 0),
    }
    full_dea_horizon_test: dict[str, np.ndarray] = {}
    for label, (ep_n, fs_n, mn, H, seed) in dea_horizon_specs.items():
        if label not in weights_d:
            continue
        full_dea_horizon_test[label] = _retrain_horizon(label, ep_n, fs_n, mn, H, seed)
        _LOG.info("  retrained %s", label)

    # 5. Persist per-component test predictions for reuse.
    pid = ds.test_df[cfg.TRUSTII_ID_COL].values
    pd.DataFrame({cfg.TRUSTII_ID_COL: pid, "test": full_v3_hep_test}).to_csv(out_root / "full_v3_hep.csv", index=False)
    pd.DataFrame({cfg.TRUSTII_ID_COL: pid, "test": full_v3_dea_test}).to_csv(out_root / "full_v3_dea.csv", index=False)
    for k, v in full_hep_horizon_test.items():
        pd.DataFrame({cfg.TRUSTII_ID_COL: pid, "test": v}).to_csv(out_root / f"full_{k}.csv", index=False)
    for k, v in full_dea_horizon_test.items():
        pd.DataFrame({cfg.TRUSTII_ID_COL: pid, "test": v}).to_csv(out_root / f"full_{k}.csv", index=False)

    # 6. Combine retrained components with the anchor's weights.
    full_h_pool: dict[str, np.ndarray] = {"_v3": full_v3_hep_test}
    full_h_pool.update(full_hep_horizon_test)
    full_d_pool: dict[str, np.ndarray] = {"_v3": full_v3_dea_test}
    full_d_pool.update(full_dea_horizon_test)
    full_h_test = rank_average(full_h_pool, weights={k: weights_h[k] for k in full_h_pool})
    full_d_test = rank_average(full_d_pool, weights={k: weights_d[k] for k in full_d_pool})

    # 7. Build candidates A-D.
    candidates = {
        "phase3_13_full_retrain_only": (full_h_test,             full_d_test),
        "phase3_13_cv90_full10":       (_blend_two(a_h_test, full_h_test, 0.90),
                                         _blend_two(a_d_test, full_d_test, 0.90)),
        "phase3_13_cv75_full25":       (_blend_two(a_h_test, full_h_test, 0.75),
                                         _blend_two(a_d_test, full_d_test, 0.75)),
        "phase3_13_cv50_full50":       (_blend_two(a_h_test, full_h_test, 0.50),
                                         _blend_two(a_d_test, full_d_test, 0.50)),
    }

    # 8. Diagnostics.
    diagnostics: list[dict] = []
    pat_id = ds.test_df[cfg.TRUSTII_ID_COL].values
    a_h_rank = pd.Series(a_h_test).rank(pct=True).to_numpy()
    a_d_rank = pd.Series(a_d_test).rank(pct=True).to_numpy()

    top_shifts: dict[str, dict] = {}
    for name, (h_pred, d_pred) in candidates.items():
        rho_h = _spearman(h_pred, a_h_test)
        rho_d = _spearman(d_pred, a_d_test)
        rho_w = cfg.WEIGHT_HEPATIC * rho_h + cfg.WEIGHT_DEATH * rho_d

        h_rank = pd.Series(h_pred).rank(pct=True).to_numpy()
        d_rank = pd.Series(d_pred).rank(pct=True).to_numpy()
        h_shift = h_rank - a_h_rank
        d_shift = d_rank - a_d_rank

        diagnostics.append({
            "candidate": name,
            "rho_hepatic": rho_h,
            "rho_death": rho_d,
            "rho_weighted": rho_w,
            "h_shift_mean_abs": float(np.mean(np.abs(h_shift))),
            "h_shift_std": float(np.std(h_shift)),
            "h_shift_max_abs": float(np.max(np.abs(h_shift))),
            "h_shift_p95": float(np.percentile(np.abs(h_shift), 95)),
            "d_shift_mean_abs": float(np.mean(np.abs(d_shift))),
            "d_shift_std": float(np.std(d_shift)),
            "d_shift_max_abs": float(np.max(np.abs(d_shift))),
            "d_shift_p95": float(np.percentile(np.abs(d_shift), 95)),
        })

        # Top-20 patients with largest hepatic / death rank shifts.
        h_top = np.argsort(-np.abs(h_shift))[:20]
        d_top = np.argsort(-np.abs(d_shift))[:20]
        top_shifts[name] = {
            "hepatic_top20": pd.DataFrame({
                cfg.TRUSTII_ID_COL: pat_id[h_top],
                "anchor_rank":   a_h_rank[h_top],
                "candidate_rank": h_rank[h_top],
                "shift": h_shift[h_top],
            }).to_dict(orient="records"),
            "death_top20": pd.DataFrame({
                cfg.TRUSTII_ID_COL: pat_id[d_top],
                "anchor_rank":   a_d_rank[d_top],
                "candidate_rank": d_rank[d_top],
                "shift": d_shift[d_top],
            }).to_dict(orient="records"),
        }

    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(out_root / "diagnostics.csv", index=False)

    # 9. Emit candidate CSVs and JSON sidecars.
    candidate_meta = {}
    for name, (h_pred, d_pred) in candidates.items():
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=h_pred, risk_death=d_pred,
            sample_submission=ds.sample_submission, model_name=name,
        )
        diag_row = next(r for r in diagnostics if r["candidate"] == name)
        meta = {
            "label": name,
            "anchor": "phase3_10_horizon_blend_v2",
            "anchor_test_csv": str(anchor_csv),
            "blend_recipe": ("0.90 anchor + 0.10 full_retrain" if name.endswith("cv90_full10")
                              else "0.75 anchor + 0.25 full_retrain" if name.endswith("cv75_full25")
                              else "0.50 anchor + 0.50 full_retrain" if name.endswith("cv50_full50")
                              else "100% full retrain (CV ensemble replaced)"),
            "components_hepatic": list(full_h_pool.keys()),
            "components_death":   list(full_d_pool.keys()),
            "weights_hepatic": weights_h,
            "weights_death":   weights_d,
            "rho_hepatic_test_anchor": diag_row["rho_hepatic"],
            "rho_death_test_anchor":   diag_row["rho_death"],
            "rho_weighted_test_anchor": diag_row["rho_weighted"],
            "h_rank_shift_mean_abs": diag_row["h_shift_mean_abs"],
            "h_rank_shift_p95":      diag_row["h_shift_p95"],
            "h_rank_shift_max_abs":  diag_row["h_shift_max_abs"],
            "d_rank_shift_mean_abs": diag_row["d_shift_mean_abs"],
            "d_rank_shift_p95":      diag_row["d_shift_p95"],
            "d_rank_shift_max_abs":  diag_row["d_shift_max_abs"],
            "uses_target_derived_features": False,
            "submission_csv": str(sub_path),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        candidate_meta[name] = meta
        _LOG.info("emitted %s -> %s", name, sub_path)

    # 10. Recommendation.
    # Heuristic: prefer cv90_full10 unless full_retrain_only is very close in
    # rank to the anchor (rho_weighted ≥ 0.985) and shifts look modest
    # (mean_abs ≤ 0.03 hep & dea). Otherwise cv75_full25 if cv90_full10 looks
    # too cautious. Never cv50 first — too aggressive a swing for a single
    # public-LB probe.
    rec_name = None
    rec_reason = ""
    full_only = next(r for r in diagnostics if r["candidate"] == "phase3_13_full_retrain_only")
    cv90 = next(r for r in diagnostics if r["candidate"] == "phase3_13_cv90_full10")
    cv75 = next(r for r in diagnostics if r["candidate"] == "phase3_13_cv75_full25")
    if (full_only["rho_weighted"] >= 0.985
        and full_only["h_shift_mean_abs"] <= 0.03
        and full_only["d_shift_mean_abs"] <= 0.03):
        rec_name = "phase3_13_full_retrain_only"
        rec_reason = (
            "full_retrain_only is essentially identical in test ranking to the "
            "anchor (weighted rank-corr ≥ 0.985 and mean |Δrank| ≤ 0.03 on both "
            "endpoints). Submitting it is a clean test of the CV-vs-full-train "
            "question with minimal LB downside."
        )
    elif cv90["rho_weighted"] >= 0.99:
        rec_name = "phase3_13_cv90_full10"
        rec_reason = (
            "cv90_full10 keeps 90% anchor weight and adds a small full-retrain "
            "tilt. This is the most conservative way to probe whether full-train "
            "shifts the LB; the rank-correlation with the anchor is ≥ 0.99."
        )
    else:
        rec_name = "phase3_13_cv75_full25"
        rec_reason = (
            "cv75_full25 tilts more toward the full-retrain prediction while "
            "still anchoring at 75% on the LB-validated CV ensemble. Use this "
            "if the cv90_full10 blend is too close to the anchor to detect a "
            "real signal."
        )

    # 11. Report.
    md: list[str] = []
    md.append("# Phase 3.13 — full-train refit experiment\n")
    md.append("Anchor: `phase3_10_horizon_blend_v2` (LB **0.91093**).\n")

    md.append("## A. Components retrained on the full training set\n")
    md.append("### Hepatic")
    md.append(pd.DataFrame([
        {"component": k, "anchor_weight": weights_h[k]}
        for k in full_h_pool
    ]).to_markdown(index=False, floatfmt=".4f"))
    md.append("")
    md.append("### Death")
    md.append(pd.DataFrame([
        {"component": k, "anchor_weight": weights_d[k]}
        for k in full_d_pool
    ]).to_markdown(index=False, floatfmt=".4f"))
    md.append("")
    md.append("v3 inner death seed-bag (refit, retains v3's internal weights):")
    md.append(pd.DataFrame([
        {"sub_component": k, "v3_inner_weight": w}
        for k, w in v3_dea_inner_weights.items()
    ]).to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## B. Diagnostics — agreement with the anchor (test predictions)\n")
    md.append(diag_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    for name in candidates:
        md.append(f"### Top-20 hepatic rank shifts — {name}\n")
        md.append(pd.DataFrame(top_shifts[name]["hepatic_top20"])
                  .to_markdown(index=False, floatfmt=".4f"))
        md.append("")
        md.append(f"### Top-20 death rank shifts — {name}\n")
        md.append(pd.DataFrame(top_shifts[name]["death_top20"])
                  .to_markdown(index=False, floatfmt=".4f"))
        md.append("")

    md.append("## C. Stability assessment\n")
    md.append(
        f"- `full_retrain_only` rho with anchor: hepatic={full_only['rho_hepatic']:.4f}, "
        f"death={full_only['rho_death']:.4f}, weighted={full_only['rho_weighted']:.4f}.\n"
        f"- mean |Δrank|: hepatic={full_only['h_shift_mean_abs']:.4f}, "
        f"death={full_only['d_shift_mean_abs']:.4f}.\n"
        f"- max |Δrank|: hepatic={full_only['h_shift_max_abs']:.4f}, "
        f"death={full_only['d_shift_max_abs']:.4f}.\n"
    )
    if full_only["h_shift_max_abs"] > 0.30 or full_only["d_shift_max_abs"] > 0.30:
        md.append(
            "**Caution**: max rank shift exceeds 0.30 on at least one endpoint. "
            "The full-retrain model likely has a few patients whose ranking is "
            "very different from the anchor — could indicate instability or "
            "overfitting on full data without held-out averaging.\n"
        )
    md.append("")

    md.append("## D. Recommendation\n")
    md.append(f"**Submit one**: `{rec_name}.csv` (saved as `{candidate_meta[rec_name]['submission_csv']}`).\n")
    md.append(f"**Reason**: {rec_reason}\n")
    md.append("")
    md.append("### Public-LB outcome interpretation\n")
    md.append(
        "- LB > 0.91093 by ≥ +0.001 → full-train refit improves test ranking; "
        "iterate further on the same recipe (consider higher full weight).\n"
        "- LB ≈ 0.91093 (within ±0.001) → CV ensemble and full retrain are "
        "interchangeable on this dataset; prefer the CV ensemble for stability.\n"
        "- LB < 0.91093 by ≥ −0.002 → CV averaging is genuinely helping; the "
        "anchor's bagging across folds is not redundant. Revert to anchor.\n"
    )

    md.append("## Notes\n")
    md.append(
        "- Same feature sets as Phase 3.10 / 3.12. Same hyperparameters. Only the "
        "training scheme changes: each component is now trained on the full "
        "1253-row training set with no fold splits.\n"
        "- Test predictions are produced by a single fit per component instead of "
        "the 15 fold-fits that the anchor averages.\n"
        "- We do not invent an OOF estimate for the full-retrain model — there is "
        "no honest way to compute one. Recommendation rests on test-prediction "
        "rank-correlation with the anchor.\n"
    )

    out_md = cfg.REPORTS_DIR / "phase3_13_full_refit.md"
    out_md.write_text("\n".join(md))
    _LOG.info("wrote %s", out_md)


if __name__ == "__main__":
    main()
