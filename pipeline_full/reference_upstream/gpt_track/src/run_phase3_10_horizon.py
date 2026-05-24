"""Phase 3.10 — refinement of the horizon-blend family.

Builds on Phase 3.9 by:

1. Decomposing the current public best ``phase3_9_horizon_blend`` into its
   v3 and horizon components, with rank-correlations and per-fold stability.
2. Expanding the horizon grid: hepatic 2y/3y/4y/5y/6y and death 4y/5y/6y
   (h<5 dropped for death because positives are too few). Default mode is
   "exclude patients censored before H"; we also run a small downweight
   check for the strongest single-model run per horizon as a sanity probe.
3. Running multi-seed (5 extra seeds) for the best hepatic h3 LGBM family
   and the best death h5 CatBoost family.
4. Building endpoint-specific horizon ensembles via greedy rank selection.
5. Running a blend grid against ``phase3_5_current_state_v3_hepatic_focused``
   (the v3 base of Phase 3.9): alphas {0.95, 0.90, 0.85, 0.80, 0.75},
   sides {hep_only, dea_only, both}, plus a greedy rank blend capped at
   25% horizon contribution.
6. Comparing every blend against ``phase3_9_horizon_blend`` (the current
   public best).

Output:
- reports/phase3_10_horizon_blend_decomposition.md
- reports/phase3_10_submission_recommendation.md
- up to two candidate submissions (phase3_10_horizon_blend_v2 and/or
  phase3_10_horizon_private_robust) with metadata sidecars.

No target-derived feature construction — horizons enter only as binary
labels and usability masks. We never use event/censoring ages as features.
"""
from __future__ import annotations

import json
import time
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
from .metrics import cindex, weighted_score
from .models import build_model
from .models.ensemble import rank_average
from .submission import make_submission
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger, timestamp
from .validation import build_folds

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature sets (same four as Phase 3.9 to keep comparability)
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
# v3 OOF + test (matches Phase 3.9 implementation exactly)
# ---------------------------------------------------------------------------

def _v3_oof(n_train: int, ds, hep) -> tuple[np.ndarray, np.ndarray]:
    cands = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.json"))
    if not cands:
        return np.full(n_train, np.nan), np.full(n_train, np.nan)
    blob = json.loads(cands[-1].read_text())
    h_w = blob["hepatic"]["weights"]
    d_w = blob["death"]["weights"]

    from .endpoint_ensemble import collect_predictions
    all_dirs = [d for d in cfg.EXPERIMENT_OUTPUTS.iterdir() if d.is_dir() and (d / "oof_predictions.csv").exists()]
    oof_df, _, _ = collect_predictions(all_dirs)
    pid_col = cfg.PATIENT_ID_COL
    pos = oof_df[pid_col].map({v: i for i, v in enumerate(ds.train_df[pid_col].values)}).to_numpy().astype(int)

    pool: dict[str, np.ndarray] = {}
    for c in set(h_w) | set(d_w):
        if c in oof_df.columns:
            arr = np.full(n_train, np.nan)
            arr[pos] = oof_df[c].to_numpy()
            pool[c] = arr

    nh_key = "phase3_5_no_visit_history::hepatic__rsf__nh"
    if nh_key in h_w and nh_key not in pool:
        Xtr = current_state_v2_no_visit_history(ds.train_df, ds.visit_columns, ds.age_visit_cols).select_dtypes(include=[np.number])
        splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)
        oof = np.full(n_train, np.nan)
        for s in splits:
            tr = s.train_idx
            va = s.valid_idx
            if hep.event[tr].sum() == 0:
                continue
            m = build_model("rsf", {"n_estimators": 250, "max_depth": 7, "min_samples_leaf": 4, "random_state": 0})
            mask = pd.Series(False, index=Xtr.index)
            mask.iloc[tr] = True
            try:
                m.fit(Xtr, hep, mask=mask)
                oof[va] = m.predict_risk(Xtr.iloc[va])
            except Exception:
                continue
        pool[nh_key] = oof

    h_oof = rank_average({c: pool[c] for c in h_w if c in pool}, weights={c: h_w[c] for c in h_w if c in pool}) \
            if any(c in pool for c in h_w) else np.full(n_train, np.nan)
    d_oof = rank_average({c: pool[c] for c in d_w if c in pool}, weights={c: d_w[c] for c in d_w if c in pool}) \
            if any(c in pool for c in d_w) else np.full(n_train, np.nan)
    return h_oof, d_oof


# ---------------------------------------------------------------------------
# CV runner (one (model, feature_set, endpoint, horizon, seed) at a time)
# ---------------------------------------------------------------------------

def _train_one_horizon(
    model_name: str,
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    horizon_label: np.ndarray,
    horizon_mask: np.ndarray,
    horizon_weight: np.ndarray,
    splits,
    *,
    n_train: int,
    n_test: int,
    seed: int = 0,
    use_sample_weight: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (oof, test) for one horizon classifier."""
    from .models._preprocess import fill_for_tree

    Xtr_pre = fill_for_tree(Xtr).astype(np.float32, errors="ignore")
    Xte_pre = fill_for_tree(Xte).reindex(columns=Xtr_pre.columns, fill_value=0).astype(np.float32, errors="ignore")
    Xtr_arr = Xtr_pre.to_numpy()
    Xte_arr = Xte_pre.to_numpy()

    oof = np.full(n_train, np.nan)
    test_sum = np.zeros(n_test)
    test_n = 0

    for s in splits:
        tr_idx = np.array([i for i in s.train_idx if horizon_mask[i]], dtype=int)
        va_idx = np.array(s.valid_idx, dtype=int)
        if len(tr_idx) < 30 or horizon_label[tr_idx].sum() < 3:
            continue
        ytr = horizon_label[tr_idx].astype(int)
        wtr = horizon_weight[tr_idx] if use_sample_weight else None
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

        try:
            if use_sample_weight and wtr is not None:
                clf.fit(Xtr_arr[tr_idx], ytr, sample_weight=wtr)
            else:
                clf.fit(Xtr_arr[tr_idx], ytr)
            oof[va_idx] = clf.predict_proba(Xtr_arr[va_idx])[:, 1]
            test_sum += clf.predict_proba(Xte_arr)[:, 1]
            test_n += 1
        except Exception as e:  # noqa: BLE001
            _LOG.warning("fold %s/%s/seed=%d failed: %s", model_name, str(seed), seed, e)

    test = test_sum / test_n if test_n else np.full(n_test, np.nan)
    return oof, test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    finite = np.isfinite(score)
    if finite.sum() == 0:
        return float("nan")
    yt = y_true[finite]
    sc = score[finite]
    if len(np.unique(yt)) < 2:
        return float("nan")
    return float(roc_auc_score(yt, sc))


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True, na_option="keep").to_numpy()


def _blend_two(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    ra, rb = _rank(a), _rank(b)
    valid_a, valid_b = np.isfinite(ra), np.isfinite(rb)
    return np.where(valid_a & valid_b, alpha * ra + (1 - alpha) * rb,
                    np.where(valid_a, ra, np.where(valid_b, rb, np.nan)))


def _surv_ci(event: np.ndarray, time: np.ndarray, score: np.ndarray) -> float:
    finite = np.isfinite(score)
    if finite.sum() < 5 or event[finite].sum() == 0:
        return float("nan")
    return float(cindex(event[finite], time[finite], score[finite]).cindex)


def _fold_ci(event, time, score, splits) -> tuple[float, float, float]:
    """Return (mean, std, min) per-fold survival C-index."""
    s_list: list[float] = []
    for s in splits:
        v = score[s.valid_idx]
        ff = np.isfinite(v)
        if ff.sum() == 0 or event[s.valid_idx][ff].sum() == 0:
            continue
        c = cindex(event[s.valid_idx][ff], time[s.valid_idx][ff], v[ff]).cindex
        if np.isfinite(c):
            s_list.append(float(c))
    if not s_list:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(s_list)), float(np.std(s_list)), float(np.min(s_list))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size != b.size:
        return float("nan")
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < 5:
        return float("nan")
    return float(s["a"].rank(pct=True).corr(s["b"].rank(pct=True), method="spearman"))


# ---------------------------------------------------------------------------
# Phase 3.9 OOF/test loader (reuse what's already on disk)
# ---------------------------------------------------------------------------

def _load_phase3_9_artifacts(out_root: Path, ds) -> dict[str, dict[str, np.ndarray]]:
    """Load all pre-computed Phase 3.9 OOF/test arrays keyed by run label."""
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    pid_to_pos = {v: i for i, v in enumerate(ds.train_df[cfg.PATIENT_ID_COL].values)}
    tid_to_pos = {v: i for i, v in enumerate(ds.test_df[cfg.TRUSTII_ID_COL].values)}
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)
    if not out_root.exists():
        return artifacts
    for run_dir in sorted(out_root.iterdir()):
        oof_path = run_dir / "oof.csv"
        test_path = run_dir / "test.csv"
        if not (oof_path.exists() and test_path.exists()):
            continue
        try:
            oof_df = pd.read_csv(oof_path)
            test_df = pd.read_csv(test_path)
            oof = np.full(n_train, np.nan)
            test = np.full(n_test, np.nan)
            pos = oof_df[cfg.PATIENT_ID_COL].map(pid_to_pos).to_numpy()
            mask = ~pd.isna(pos)
            oof[pos[mask].astype(int)] = oof_df["oof"].to_numpy()[mask]
            tpos = test_df[cfg.TRUSTII_ID_COL].map(tid_to_pos).to_numpy()
            tmask = ~pd.isna(tpos)
            test[tpos[tmask].astype(int)] = test_df["test"].to_numpy()[tmask]
            artifacts[run_dir.name] = {"oof": oof, "test": test}
        except Exception as e:  # noqa: BLE001
            _LOG.warning("could not load %s: %s", run_dir, e)
    return artifacts


# ---------------------------------------------------------------------------
# Greedy rank-ensemble of horizon models
# ---------------------------------------------------------------------------

def _greedy_horizon_ensemble(
    candidates: dict[str, np.ndarray],
    event: np.ndarray,
    time: np.ndarray,
    *,
    min_individual_ci: float = 0.55,
    max_size: int = 6,
) -> tuple[list[str], dict[str, float], float]:
    """Greedy forward selection in rank-space, optimising survival C-index."""
    pool = {}
    for k, v in candidates.items():
        ci = _surv_ci(event, time, v)
        if np.isfinite(ci) and ci >= min_individual_ci:
            pool[k] = v
    if not pool:
        return [], {}, float("nan")

    selected: list[str] = []
    selected_arrs: list[np.ndarray] = []
    best_ci = -np.inf
    while len(selected) < min(max_size, len(pool)):
        best_k = None
        best_new_ci = best_ci
        best_new_arr = None
        for k, v in pool.items():
            if k in selected:
                continue
            arrs = selected_arrs + [v]
            blended = rank_average({f"k{i}": a for i, a in enumerate(arrs)})
            ci = _surv_ci(event, time, blended)
            if ci > best_new_ci + 1e-5:
                best_new_ci = ci
                best_k = k
                best_new_arr = blended
        if best_k is None:
            break
        selected.append(best_k)
        selected_arrs.append(pool[best_k])
        best_ci = best_new_ci
    weights = {k: 1.0 / len(selected) for k in selected} if selected else {}
    return selected, weights, best_ci


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    cfg.ensure_dirs()
    p9_root = cfg.EXPERIMENT_OUTPUTS / "phase3_9_horizon"
    p10_root = cfg.EXPERIMENT_OUTPUTS / "phase3_10_horizon"
    p10_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)

    feature_sets = assemble_feature_sets(ds)

    # ------------------------------------------------------------------
    # 0. Reference v3 + Phase 3.9 horizon-blend predictions
    # ------------------------------------------------------------------

    v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)
    v3_csv = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.csv"))[-1]
    v3_pub = pd.read_csv(v3_csv)
    v3_h_test = v3_pub[cfg.SUB_HEPATIC_COL].to_numpy()
    v3_d_test = v3_pub[cfg.SUB_DEATH_COL].to_numpy()
    v3_meta = json.loads(v3_csv.with_suffix(".json").read_text())
    v3_h_oof_ref = v3_meta["hepatic"]["oof_cindex"]
    v3_d_oof_ref = v3_meta["death"]["oof_cindex"]
    v3_w_ref = v3_meta["weighted_score_cv"]

    v3_h_fold = _fold_ci(hep.event, hep.time, v3_h_oof, splits)
    v3_d_fold = _fold_ci(death.event, death.time, v3_d_oof, splits)

    # Phase 3.9 horizon-blend test predictions and re-derived OOF blend.
    p9_csv = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_9_horizon_blend.csv"))[-1]
    p9_pub = pd.read_csv(p9_csv)
    p9_h_test = p9_pub[cfg.SUB_HEPATIC_COL].to_numpy()
    p9_d_test = p9_pub[cfg.SUB_DEATH_COL].to_numpy()
    p9_meta = json.loads(p9_csv.with_suffix(".json").read_text())
    _LOG.info("Phase 3.9 blend: alpha=%s side=%s hep_best=%s dea_best=%s",
              p9_meta["alpha"], p9_meta["side"], p9_meta["hep_best_run"], p9_meta["dea_best_run"])

    # ------------------------------------------------------------------
    # 1. Load Phase 3.9 horizon OOFs (reuse on disk)
    # ------------------------------------------------------------------

    p9_artifacts = _load_phase3_9_artifacts(p9_root, ds)
    _LOG.info("loaded %d Phase 3.9 horizon artifacts", len(p9_artifacts))

    p9_hep_best_label = p9_meta["hep_best_run"]
    p9_dea_best_label = p9_meta["dea_best_run"]
    p9_hep_best_oof = p9_artifacts[p9_hep_best_label]["oof"]
    p9_dea_best_oof = p9_artifacts[p9_dea_best_label]["oof"]
    p9_hep_best_test = p9_artifacts[p9_hep_best_label]["test"]
    p9_dea_best_test = p9_artifacts[p9_dea_best_label]["test"]

    # The Phase 3.9 blend itself, reproduced in OOF space (rank blend).
    p9_blend_h_oof = _blend_two(v3_h_oof, p9_hep_best_oof, p9_meta["alpha"])
    p9_blend_d_oof = _blend_two(v3_d_oof, p9_dea_best_oof, p9_meta["alpha"])

    # ------------------------------------------------------------------
    # 2. Run new horizons (h2, h4, h6 hep; h4, h6 death) + multi-seed for the
    #    best families. Reuse Phase 3.9 results for h1/h3/h5 hep and h5 death.
    # ------------------------------------------------------------------

    runs: list[dict] = []
    new_artifacts: dict[str, dict[str, np.ndarray]] = {}

    horizons_hep = [2.0, 3.0, 4.0, 5.0, 6.0]
    horizons_dea = [4.0, 5.0, 6.0]
    models = ["lgbm_binary", "catboost_binary", "xgb_binary"]

    label_blobs: dict[tuple[str, float], object] = {}
    horizon_summary: list[dict] = []
    for ep_name, ep in [("hepatic", hep), ("death", death)]:
        hzs = horizons_hep if ep_name == "hepatic" else horizons_dea
        for H in hzs:
            for mode in ("exclude", "downweight"):
                lab = build_horizon_labels(ep, H, censored_mode=mode)
                label_blobs[(ep_name, H, mode)] = lab
            lab_e = label_blobs[(ep_name, H, "exclude")]
            horizon_summary.append({
                "endpoint": ep_name,
                "horizon_years": H,
                "n_total": lab_e.n_total,
                "n_usable_exclude": lab_e.n_usable,
                "n_positive_exclude": lab_e.n_positive,
                "n_censored_before_h": lab_e.notes["n_censored_before_horizon"],
                "positive_rate": lab_e.n_positive / max(lab_e.n_usable, 1),
            })

    def _process_run(label: str, oof: np.ndarray, test: np.ndarray,
                     fs_name: str, ep_name: str, H: float, model_name: str,
                     n_features: int, lab, mode: str = "exclude",
                     seed: int = 0, dt: float = 0.0):
        ep = hep if ep_name == "hepatic" else death
        ci_surv = _surv_ci(ep.event, ep.time, oof)
        mean_, std_, min_ = _fold_ci(ep.event, ep.time, oof, splits)
        use = lab.mask & np.isfinite(oof)
        auc = _safe_auc(lab.label[use], oof[use]) if use.sum() else float("nan")
        rho_v3_oof = _spearman(oof, v3_h_oof if ep_name == "hepatic" else v3_d_oof)
        rho_v3_test = _spearman(test, v3_h_test if ep_name == "hepatic" else v3_d_test)
        runs.append({
            "label": label,
            "feature_set": fs_name,
            "endpoint": ep_name,
            "horizon_years": H,
            "model": model_name,
            "mode": mode,
            "seed": seed,
            "n_features": n_features,
            "n_usable_train": int(lab.n_usable),
            "n_positive_train": int(lab.n_positive),
            "horizon_auc": auc,
            "surv_cindex_mean": ci_surv,
            "surv_cindex_std": std_,
            "surv_cindex_min": min_,
            "wall_seconds": dt,
            "rho_oof_v3": rho_v3_oof,
            "rho_test_v3": rho_v3_test,
        })
        new_artifacts[label] = {"oof": oof, "test": test}

    # Re-import Phase 3.9 results into the runs table.
    for ep_name, ep in [("hepatic", hep), ("death", death)]:
        hzs_p9 = [1.0, 3.0, 5.0] if ep_name == "hepatic" else [5.0]
        for H in hzs_p9:
            lab = label_blobs.get((ep_name, H, "exclude"))
            if lab is None:
                lab = build_horizon_labels(ep, H, censored_mode="exclude")
                label_blobs[(ep_name, H, "exclude")] = lab
                horizon_summary.append({
                    "endpoint": ep_name, "horizon_years": H,
                    "n_total": lab.n_total, "n_usable_exclude": lab.n_usable,
                    "n_positive_exclude": lab.n_positive,
                    "n_censored_before_h": lab.notes["n_censored_before_horizon"],
                    "positive_rate": lab.n_positive / max(lab.n_usable, 1),
                })
            for fs_name, blob in feature_sets.items():
                for model_name in models:
                    label = f"{fs_name}__{ep_name}__h{int(H)}__{model_name}"
                    if label not in p9_artifacts:
                        continue
                    a = p9_artifacts[label]
                    _process_run(label, a["oof"], a["test"], fs_name, ep_name,
                                 H, model_name, blob["train"].shape[1], lab,
                                 mode="exclude", seed=0, dt=0.0)

    _LOG.info("imported %d Phase 3.9 runs into tables", len(runs))

    # New horizons (h2, h4, h6 hep; h4, h6 death) — exclude mode.
    new_horizons_hep = [2.0, 4.0, 6.0]
    new_horizons_dea = [4.0, 6.0]
    for ep_name, ep, hzs in [("hepatic", hep, new_horizons_hep), ("death", death, new_horizons_dea)]:
        for H in hzs:
            lab = label_blobs[(ep_name, H, "exclude")]
            if lab.n_positive < 5:
                _LOG.warning("skip %s h%.0f: only %d positives", ep_name, H, lab.n_positive)
                continue
            for fs_name, blob in feature_sets.items():
                for model_name in models:
                    label = f"{fs_name}__{ep_name}__h{int(H)}__{model_name}"
                    t0 = time.time()
                    try:
                        oof, test = _train_one_horizon(
                            model_name, blob["train"], blob["test"],
                            lab.label, lab.mask, lab.weight, splits,
                            n_train=n_train, n_test=n_test, seed=0,
                        )
                    except Exception as e:  # noqa: BLE001
                        _LOG.error("%s failed: %s", label, e)
                        continue
                    dt = time.time() - t0
                    # Persist OOF/test (so future phases can reuse).
                    run_dir = p10_root / label
                    run_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({cfg.PATIENT_ID_COL: ds.train_df[cfg.PATIENT_ID_COL].values, "oof": oof}).to_csv(run_dir / "oof.csv", index=False)
                    pd.DataFrame({cfg.TRUSTII_ID_COL: ds.test_df[cfg.TRUSTII_ID_COL].values, "test": test}).to_csv(run_dir / "test.csv", index=False)
                    _process_run(label, oof, test, fs_name, ep_name, H, model_name,
                                 blob["train"].shape[1], lab, mode="exclude", seed=0, dt=dt)
                    _LOG.info("%s survC=%.4f AUC=%.4f (%.1fs)", label,
                              runs[-1]["surv_cindex_mean"], runs[-1]["horizon_auc"], dt)

    # Multi-seed for best Phase 3.9 hep family (LGBM, v3_hepatic_schema, h3) and
    # best death family (CatBoost, current_state_v2, h5). Use seeds 1..4.
    seeds_extra = [1, 2, 3, 4]
    seed_targets = [
        ("v3_hepatic_schema", "hepatic", 3.0, "lgbm_binary"),
        ("current_state_v2", "death", 5.0, "catboost_binary"),
    ]
    for fs_name, ep_name, H, model_name in seed_targets:
        ep = hep if ep_name == "hepatic" else death
        lab = label_blobs.get((ep_name, H, "exclude"))
        if lab is None:
            lab = build_horizon_labels(ep, H, censored_mode="exclude")
            label_blobs[(ep_name, H, "exclude")] = lab
        blob = feature_sets[fs_name]
        for seed in seeds_extra:
            label = f"{fs_name}__{ep_name}__h{int(H)}__{model_name}__s{seed}"
            t0 = time.time()
            try:
                oof, test = _train_one_horizon(
                    model_name, blob["train"], blob["test"],
                    lab.label, lab.mask, lab.weight, splits,
                    n_train=n_train, n_test=n_test, seed=seed,
                )
            except Exception as e:  # noqa: BLE001
                _LOG.error("%s failed: %s", label, e)
                continue
            dt = time.time() - t0
            run_dir = p10_root / label
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({cfg.PATIENT_ID_COL: ds.train_df[cfg.PATIENT_ID_COL].values, "oof": oof}).to_csv(run_dir / "oof.csv", index=False)
            pd.DataFrame({cfg.TRUSTII_ID_COL: ds.test_df[cfg.TRUSTII_ID_COL].values, "test": test}).to_csv(run_dir / "test.csv", index=False)
            _process_run(label, oof, test, fs_name, ep_name, H, model_name,
                         blob["train"].shape[1], lab, mode="exclude", seed=seed, dt=dt)
            _LOG.info("seed-bag %s survC=%.4f (%.1fs)", label, runs[-1]["surv_cindex_mean"], dt)

    # Optional: downweight-mode probe for the strongest model per horizon. Limit to
    # the v3_hepatic_schema/LGBM family for hep and current_state_v2/CatBoost for
    # death, across all horizons we're evaluating. Cheap signal.
    downweight_probes = [
        ("v3_hepatic_schema", "hepatic", "lgbm_binary"),
        ("current_state_v2",  "death",   "catboost_binary"),
    ]
    for fs_name, ep_name, model_name in downweight_probes:
        ep = hep if ep_name == "hepatic" else death
        hzs = horizons_hep if ep_name == "hepatic" else horizons_dea
        for H in hzs:
            lab = label_blobs.get((ep_name, H, "downweight"))
            if lab is None:
                lab = build_horizon_labels(ep, H, censored_mode="downweight")
                label_blobs[(ep_name, H, "downweight")] = lab
            if lab.n_positive < 5:
                continue
            blob = feature_sets[fs_name]
            label = f"{fs_name}__{ep_name}__h{int(H)}__{model_name}__dw"
            t0 = time.time()
            try:
                oof, test = _train_one_horizon(
                    model_name, blob["train"], blob["test"],
                    lab.label, lab.mask, lab.weight, splits,
                    n_train=n_train, n_test=n_test, seed=0, use_sample_weight=True,
                )
            except Exception as e:  # noqa: BLE001
                _LOG.error("%s failed: %s", label, e)
                continue
            dt = time.time() - t0
            run_dir = p10_root / label
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({cfg.PATIENT_ID_COL: ds.train_df[cfg.PATIENT_ID_COL].values, "oof": oof}).to_csv(run_dir / "oof.csv", index=False)
            pd.DataFrame({cfg.TRUSTII_ID_COL: ds.test_df[cfg.TRUSTII_ID_COL].values, "test": test}).to_csv(run_dir / "test.csv", index=False)
            _process_run(label, oof, test, fs_name, ep_name, H, model_name,
                         blob["train"].shape[1], lab, mode="downweight", seed=0, dt=dt)
            _LOG.info("dw %s survC=%.4f (%.1fs)", label, runs[-1]["surv_cindex_mean"], dt)

    # Combine all artifacts (Phase 3.9 + Phase 3.10) for ensembling.
    all_artifacts: dict[str, dict[str, np.ndarray]] = {}
    all_artifacts.update(p9_artifacts)
    all_artifacts.update(new_artifacts)

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(p10_root / "all_runs.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Endpoint-specific horizon ensembles via greedy rank selection
    # ------------------------------------------------------------------

    hep_pool = {r["label"]: all_artifacts[r["label"]]["oof"]
                for r in runs if r["endpoint"] == "hepatic" and r["label"] in all_artifacts}
    dea_pool = {r["label"]: all_artifacts[r["label"]]["oof"]
                for r in runs if r["endpoint"] == "death" and r["label"] in all_artifacts}

    hep_sel, hep_w, hep_ens_ci = _greedy_horizon_ensemble(hep_pool, hep.event, hep.time)
    dea_sel, dea_w, dea_ens_ci = _greedy_horizon_ensemble(dea_pool, death.event, death.time)
    _LOG.info("hepatic horizon ensemble: %s -> survC=%.4f", hep_sel, hep_ens_ci)
    _LOG.info("death horizon ensemble: %s -> survC=%.4f", dea_sel, dea_ens_ci)

    # OOF and test predictions for the ensembles.
    hep_ens_oof = rank_average({k: all_artifacts[k]["oof"] for k in hep_sel}, weights=hep_w) \
                  if hep_sel else np.full(n_train, np.nan)
    hep_ens_test = rank_average({k: all_artifacts[k]["test"] for k in hep_sel}, weights=hep_w) \
                   if hep_sel else np.full(n_test, np.nan)
    dea_ens_oof = rank_average({k: all_artifacts[k]["oof"] for k in dea_sel}, weights=dea_w) \
                  if dea_sel else np.full(n_train, np.nan)
    dea_ens_test = rank_average({k: all_artifacts[k]["test"] for k in dea_sel}, weights=dea_w) \
                   if dea_sel else np.full(n_test, np.nan)

    # Best single horizon model per endpoint (matches Phase 3.9 selection rule).
    hep_runs = [r for r in runs if r["endpoint"] == "hepatic" and np.isfinite(r["surv_cindex_mean"])]
    dea_runs = [r for r in runs if r["endpoint"] == "death" and np.isfinite(r["surv_cindex_mean"])]
    hep_best = max(hep_runs, key=lambda r: r["surv_cindex_mean"]) if hep_runs else None
    dea_best = max(dea_runs, key=lambda r: r["surv_cindex_mean"]) if dea_runs else None

    hep_best_oof = all_artifacts[hep_best["label"]]["oof"] if hep_best else None
    hep_best_test = all_artifacts[hep_best["label"]]["test"] if hep_best else None
    dea_best_oof = all_artifacts[dea_best["label"]]["oof"] if dea_best else None
    dea_best_test = all_artifacts[dea_best["label"]]["test"] if dea_best else None

    # ------------------------------------------------------------------
    # 4. Blend grid against v3 (the "currentBest" of Phase 3.9). For each
    #    candidate variant we compute OOF metrics, fold stability, and rank-
    #    correlation with phase3_9_horizon_blend.
    # ------------------------------------------------------------------

    blend_rows: list[dict] = []

    def _eval_blend(name: str, h_oof: np.ndarray, d_oof: np.ndarray,
                    h_test: np.ndarray, d_test: np.ndarray,
                    components_h: list[str], components_d: list[str],
                    weight_h: dict, weight_d: dict, alpha: float | None,
                    side: str, source: str):
        ci_h = _surv_ci(hep.event, hep.time, h_oof)
        ci_d = _surv_ci(death.event, death.time, d_oof)
        h_mean, h_std, h_min = _fold_ci(hep.event, hep.time, h_oof, splits)
        d_mean, d_std, d_min = _fold_ci(death.event, death.time, d_oof, splits)
        rho_h_p9 = _spearman(h_test, p9_h_test)
        rho_d_p9 = _spearman(d_test, p9_d_test)
        rho_h_p9_oof = _spearman(h_oof, p9_blend_h_oof)
        rho_d_p9_oof = _spearman(d_oof, p9_blend_d_oof)
        blend_rows.append({
            "blend": name,
            "alpha": alpha,
            "side": side,
            "source": source,
            "hep_oof": ci_h,
            "death_oof": ci_d,
            "weighted_oof": weighted_score(ci_h, ci_d),
            "hep_fold_std": h_std,
            "hep_fold_min": h_min,
            "dea_fold_std": d_std,
            "dea_fold_min": d_min,
            "rho_test_p9": (cfg.WEIGHT_HEPATIC * rho_h_p9 + cfg.WEIGHT_DEATH * rho_d_p9)
                            if np.isfinite(rho_h_p9) and np.isfinite(rho_d_p9) else float("nan"),
            "rho_h_test_p9": rho_h_p9,
            "rho_d_test_p9": rho_d_p9,
            "rho_h_oof_p9": rho_h_p9_oof,
            "rho_d_oof_p9": rho_d_p9_oof,
            "components_h": components_h,
            "components_d": components_d,
            "weights_h": weight_h,
            "weights_d": weight_d,
            "h_test": h_test,
            "d_test": d_test,
            "h_oof": h_oof,
            "d_oof": d_oof,
        })

    # 4a. Best-single horizon blends (mirrors Phase 3.9 grid).
    if hep_best is not None and dea_best is not None:
        for alpha in (0.95, 0.90, 0.85, 0.80, 0.75):
            for side in ("hep_only", "dea_only", "both"):
                if side in ("hep_only", "both"):
                    h_oof = _blend_two(v3_h_oof, hep_best_oof, alpha)
                    h_test = _blend_two(v3_h_test, hep_best_test, alpha)
                    comp_h = [hep_best["label"]]
                    w_h = {"v3_hep": alpha, hep_best["label"]: 1 - alpha}
                else:
                    h_oof = v3_h_oof.copy()
                    h_test = v3_h_test.copy()
                    comp_h = []
                    w_h = {"v3_hep": 1.0}
                if side in ("dea_only", "both"):
                    d_oof = _blend_two(v3_d_oof, dea_best_oof, alpha)
                    d_test = _blend_two(v3_d_test, dea_best_test, alpha)
                    comp_d = [dea_best["label"]]
                    w_d = {"v3_dea": alpha, dea_best["label"]: 1 - alpha}
                else:
                    d_oof = v3_d_oof.copy()
                    d_test = v3_d_test.copy()
                    comp_d = []
                    w_d = {"v3_dea": 1.0}
                _eval_blend(f"single_alpha={alpha}_{side}", h_oof, d_oof, h_test, d_test,
                            comp_h, comp_d, w_h, w_d, alpha, side, "best_single")

    # 4b. Endpoint-ensemble blends.
    if hep_sel and dea_sel:
        for alpha in (0.95, 0.90, 0.85, 0.80, 0.75):
            for side in ("hep_only", "dea_only", "both"):
                if side in ("hep_only", "both"):
                    h_oof = _blend_two(v3_h_oof, hep_ens_oof, alpha)
                    h_test = _blend_two(v3_h_test, hep_ens_test, alpha)
                    comp_h = list(hep_sel)
                    w_h = {"v3_hep": alpha, **{k: (1 - alpha) * w for k, w in hep_w.items()}}
                else:
                    h_oof = v3_h_oof.copy()
                    h_test = v3_h_test.copy()
                    comp_h = []
                    w_h = {"v3_hep": 1.0}
                if side in ("dea_only", "both"):
                    d_oof = _blend_two(v3_d_oof, dea_ens_oof, alpha)
                    d_test = _blend_two(v3_d_test, dea_ens_test, alpha)
                    comp_d = list(dea_sel)
                    w_d = {"v3_dea": alpha, **{k: (1 - alpha) * w for k, w in dea_w.items()}}
                else:
                    d_oof = v3_d_oof.copy()
                    d_test = v3_d_test.copy()
                    comp_d = []
                    w_d = {"v3_dea": 1.0}
                _eval_blend(f"ensemble_alpha={alpha}_{side}", h_oof, d_oof, h_test, d_test,
                            comp_h, comp_d, w_h, w_d, alpha, side, "ensemble")

    # 4c. Greedy rank blend with max 25% horizon contribution. Build as
    # weighted rank average of [v3, top horizon contributors]; add components
    # one by one so long as the *total horizon weight* stays <= 0.25.
    def _greedy_capped(v3_oof: np.ndarray, v3_test: np.ndarray,
                       pool: dict[str, dict[str, np.ndarray]],
                       event: np.ndarray, time: np.ndarray,
                       cap: float = 0.25) -> tuple[dict, dict, np.ndarray, np.ndarray]:
        chosen_w: dict[str, float] = {"_v3": 1.0}
        cur_oof = v3_oof.copy()
        cur_test = v3_test.copy()
        cur_ci = _surv_ci(event, time, cur_oof)
        improved = True
        while improved:
            improved = False
            best_k = None
            best_dw = None
            best_ci = cur_ci
            best_oof = None
            best_test = None
            for k, ar in pool.items():
                if k in chosen_w:
                    continue
                # Try adding this component with several incremental weights.
                horizon_w_now = sum(w for kk, w in chosen_w.items() if kk != "_v3")
                room = cap - horizon_w_now
                if room <= 0.001:
                    continue
                for dw in (0.05, 0.10, 0.15):
                    if dw > room + 1e-9:
                        continue
                    new_w = dict(chosen_w)
                    new_w[k] = dw
                    # renormalise so v3 + sum(others) = 1
                    s = sum(new_w.values())
                    new_w = {kk: vv / s for kk, vv in new_w.items()}
                    # Build rank-blend.
                    ranks_oof = {"_v3": _rank(v3_oof)}
                    ranks_test = {"_v3": _rank(v3_test)}
                    for kk in new_w:
                        if kk == "_v3":
                            continue
                        ranks_oof[kk] = _rank(pool[kk]["oof"])
                        ranks_test[kk] = _rank(pool[kk]["test"])
                    blended_oof = np.zeros_like(v3_oof)
                    blended_test = np.zeros_like(v3_test)
                    weight_total_oof = np.zeros_like(v3_oof)
                    weight_total_test = np.zeros_like(v3_test)
                    for kk, w in new_w.items():
                        ro = ranks_oof[kk]
                        rt = ranks_test[kk]
                        valid_o = np.isfinite(ro)
                        valid_t = np.isfinite(rt)
                        blended_oof = np.where(valid_o, blended_oof + w * np.where(valid_o, ro, 0), blended_oof)
                        blended_test = np.where(valid_t, blended_test + w * np.where(valid_t, rt, 0), blended_test)
                        weight_total_oof = weight_total_oof + w * valid_o.astype(float)
                        weight_total_test = weight_total_test + w * valid_t.astype(float)
                    blended_oof = np.where(weight_total_oof > 0, blended_oof / np.where(weight_total_oof > 0, weight_total_oof, 1), np.nan)
                    blended_test = np.where(weight_total_test > 0, blended_test / np.where(weight_total_test > 0, weight_total_test, 1), np.nan)
                    ci = _surv_ci(event, time, blended_oof)
                    if ci > best_ci + 1e-5:
                        best_ci = ci
                        best_k = k
                        best_dw = dw
                        best_oof = blended_oof
                        best_test = blended_test
            if best_k is not None:
                chosen_w[best_k] = best_dw
                s = sum(chosen_w.values())
                chosen_w = {kk: vv / s for kk, vv in chosen_w.items()}
                cur_ci = best_ci
                cur_oof = best_oof
                cur_test = best_test
                improved = True
        return chosen_w, {"ci": cur_ci}, cur_oof, cur_test

    hep_pool_for_greedy = {k: {"oof": all_artifacts[k]["oof"], "test": all_artifacts[k]["test"]}
                            for k in hep_pool
                            if _surv_ci(hep.event, hep.time, all_artifacts[k]["oof"]) >= 0.55}
    dea_pool_for_greedy = {k: {"oof": all_artifacts[k]["oof"], "test": all_artifacts[k]["test"]}
                            for k in dea_pool
                            if _surv_ci(death.event, death.time, all_artifacts[k]["oof"]) >= 0.55}

    g_h_w, _g_h_meta, g_h_oof, g_h_test = _greedy_capped(v3_h_oof, v3_h_test, hep_pool_for_greedy,
                                                         hep.event, hep.time, cap=0.25)
    g_d_w, _g_d_meta, g_d_oof, g_d_test = _greedy_capped(v3_d_oof, v3_d_test, dea_pool_for_greedy,
                                                         death.event, death.time, cap=0.25)
    comp_h = [k for k in g_h_w if k != "_v3"]
    comp_d = [k for k in g_d_w if k != "_v3"]
    _eval_blend("greedy_cap25_both", g_h_oof, g_d_oof, g_h_test, g_d_test,
                comp_h, comp_d, g_h_w, g_d_w, alpha=None, side="both", source="greedy_capped")

    # 4d. The Phase 3.9 blend itself, recomputed on the same metric grid for
    # apples-to-apples comparison.
    p9_blend_h_test = _blend_two(v3_h_test, p9_hep_best_test, p9_meta["alpha"])
    p9_blend_d_test = _blend_two(v3_d_test, p9_dea_best_test, p9_meta["alpha"])
    _eval_blend("phase3_9_horizon_blend", p9_blend_h_oof, p9_blend_d_oof,
                p9_blend_h_test, p9_blend_d_test,
                [p9_hep_best_label], [p9_dea_best_label],
                {"v3_hep": p9_meta["alpha"], p9_hep_best_label: 1 - p9_meta["alpha"]},
                {"v3_dea": p9_meta["alpha"], p9_dea_best_label: 1 - p9_meta["alpha"]},
                p9_meta["alpha"], p9_meta["side"], "phase3_9")

    # v3 alone.
    _eval_blend("v3_alone", v3_h_oof, v3_d_oof, v3_h_test, v3_d_test,
                [], [], {"v3_hep": 1.0}, {"v3_dea": 1.0}, alpha=1.0, side="none", source="v3")

    # ------------------------------------------------------------------
    # 5. Pick candidates
    # ------------------------------------------------------------------

    p9_row = next(r for r in blend_rows if r["blend"] == "phase3_9_horizon_blend")
    p9_w = p9_row["weighted_oof"]
    p9_h = p9_row["hep_oof"]
    p9_d = p9_row["death_oof"]
    p9_h_std = p9_row["hep_fold_std"]
    p9_h_min = p9_row["hep_fold_min"]

    # Best OOF improvement: maximise weighted_oof - p9_w; require >= +0.0010.
    # Among those, prefer hepatic gain over death-only gains.
    candidates_for_v2: list[dict] = []
    for row in blend_rows:
        if row["blend"] in ("phase3_9_horizon_blend", "v3_alone"):
            continue
        delta_w = row["weighted_oof"] - p9_w
        delta_h = row["hep_oof"] - p9_h
        if delta_w >= 0.0010 or delta_h >= 0.0015:
            candidates_for_v2.append(row)

    candidate_v2 = None
    if candidates_for_v2:
        candidate_v2 = max(candidates_for_v2, key=lambda r: (r["weighted_oof"], r["hep_oof"]))

    # Private-robust: same OOF or slightly worse, but lower hep_fold_std.
    candidates_robust: list[dict] = []
    for row in blend_rows:
        if row["blend"] in ("phase3_9_horizon_blend", "v3_alone"):
            continue
        # Allow up to 0.0010 lower weighted OOF if std is meaningfully smaller.
        std_better = (np.isfinite(row["hep_fold_std"]) and np.isfinite(p9_h_std)
                      and row["hep_fold_std"] < p9_h_std - 0.005)
        min_better = (np.isfinite(row["hep_fold_min"]) and np.isfinite(p9_h_min)
                      and row["hep_fold_min"] > p9_h_min + 0.005)
        weighted_acceptable = row["weighted_oof"] >= p9_w - 0.0010
        if weighted_acceptable and (std_better or min_better):
            candidates_robust.append(row)

    candidate_robust = None
    if candidates_robust:
        # Prefer lowest fold std, then highest hep_fold_min, then highest weighted.
        candidate_robust = sorted(
            candidates_robust,
            key=lambda r: (-r["hep_fold_min"], r["hep_fold_std"], -r["weighted_oof"]),
        )[0]

    # If the private-robust candidate is identical to candidate_v2, drop it.
    if candidate_v2 is not None and candidate_robust is not None and \
       candidate_robust["blend"] == candidate_v2["blend"]:
        candidate_robust = None

    # ------------------------------------------------------------------
    # 6. Emit candidate CSVs and JSON sidecars
    # ------------------------------------------------------------------

    def _emit(row: dict, label: str, rationale: str) -> tuple[Path, dict]:
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=row["h_test"], risk_death=row["d_test"],
            sample_submission=ds.sample_submission, model_name=label,
        )
        meta = {
            "label": label,
            "blend_id": row["blend"],
            "alpha": row["alpha"],
            "side": row["side"],
            "components_hepatic": row["components_h"],
            "components_death": row["components_d"],
            "weights_hepatic": row["weights_h"],
            "weights_death": row["weights_d"],
            "hepatic_oof": row["hep_oof"],
            "death_oof": row["death_oof"],
            "weighted_oof": row["weighted_oof"],
            "hepatic_fold_std": row["hep_fold_std"],
            "hepatic_fold_min": row["hep_fold_min"],
            "death_fold_std": row["dea_fold_std"],
            "death_fold_min": row["dea_fold_min"],
            "rho_test_with_phase3_9_horizon_blend": {
                "hepatic": row["rho_h_test_p9"],
                "death":   row["rho_d_test_p9"],
                "weighted": row["rho_test_p9"],
            },
            "rho_oof_with_phase3_9_horizon_blend": {
                "hepatic": row["rho_h_oof_p9"],
                "death":   row["rho_d_oof_p9"],
            },
            "delta_vs_phase3_9": {
                "weighted_oof": row["weighted_oof"] - p9_w,
                "hepatic_oof":  row["hep_oof"] - p9_h,
                "death_oof":    row["death_oof"] - p9_d,
            },
            "uses_target_derived_features": False,
            "rationale": rationale,
            "submission_csv": str(sub_path),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        _LOG.info("candidate %s -> %s", label, sub_path)
        return sub_path, meta

    candidate_v2_meta = None
    candidate_robust_meta = None
    candidate_v2_path = None
    candidate_robust_path = None
    if candidate_v2 is not None:
        candidate_v2_path, candidate_v2_meta = _emit(
            candidate_v2,
            "phase3_10_horizon_blend_v2",
            "Best OOF improvement over phase3_9_horizon_blend, restricted to "
            "rank-space horizon blends with no target-derived features.",
        )
    if candidate_robust is not None:
        candidate_robust_path, candidate_robust_meta = _emit(
            candidate_robust,
            "phase3_10_horizon_private_robust",
            "Conservative variant: trades a small slice of OOF for better "
            "fold stability (lower hepatic fold std and/or higher fold min).",
        )

    # ------------------------------------------------------------------
    # 7. Decomposition report
    # ------------------------------------------------------------------

    md: list[str] = []
    md.append("# Phase 3.10 — horizon-blend decomposition\n")
    md.append("Reference candidate: `phase3_9_horizon_blend` (public LB **0.90419**).\n")
    md.append("## A. Phase 3.9 components and metrics\n")
    md.append("| field | value |")
    md.append("|---|---|")
    md.append(f"| blend ID | `{p9_meta['blend']}` |")
    md.append(f"| alpha | {p9_meta['alpha']} |")
    md.append(f"| side  | {p9_meta['side']} |")
    md.append(f"| hepatic horizon component | `{p9_hep_best_label}` |")
    md.append(f"| death horizon component   | `{p9_dea_best_label}` |")
    md.append(f"| OOF hepatic C-index       | {p9_h:.4f} |")
    md.append(f"| OOF death   C-index       | {p9_d:.4f} |")
    md.append(f"| weighted OOF              | {p9_w:.4f} |")
    md.append(f"| hep fold std / min        | {p9_h_std:.4f} / {p9_h_min:.4f} |")
    md.append(f"| dea fold std / min        | {p9_row['dea_fold_std']:.4f} / {p9_row['dea_fold_min']:.4f} |")
    md.append("")

    md.append("### Components vs reference\n")
    rows_dec = []
    for label, oof, test, ep in [
        ("v3_hepatic_focused (hepatic side)", v3_h_oof, v3_h_test, "hepatic"),
        (f"horizon hep ({p9_hep_best_label})", p9_hep_best_oof, p9_hep_best_test, "hepatic"),
        ("v3_hepatic_focused (death side)", v3_d_oof, v3_d_test, "death"),
        (f"horizon dea ({p9_dea_best_label})", p9_dea_best_oof, p9_dea_best_test, "death"),
        ("phase3_9_horizon_blend hep", p9_blend_h_oof, p9_blend_h_test, "hepatic"),
        ("phase3_9_horizon_blend dea", p9_blend_d_oof, p9_blend_d_test, "death"),
    ]:
        ev = hep.event if ep == "hepatic" else death.event
        tm = hep.time  if ep == "hepatic" else death.time
        ci = _surv_ci(ev, tm, oof)
        m, sd, mn = _fold_ci(ev, tm, oof, splits)
        rows_dec.append({
            "component": label, "endpoint": ep,
            "oof_cindex": ci, "fold_mean": m, "fold_std": sd, "fold_min": mn,
            "rho_oof_v3": _spearman(oof, v3_h_oof if ep == "hepatic" else v3_d_oof),
            "rho_oof_p9_blend": _spearman(oof, p9_blend_h_oof if ep == "hepatic" else p9_blend_d_oof),
            "rho_test_v3": _spearman(test, v3_h_test if ep == "hepatic" else v3_d_test),
            "rho_test_p9_blend": _spearman(test, p9_h_test if ep == "hepatic" else p9_d_test),
        })
    md.append(pd.DataFrame(rows_dec).to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## B. Horizon expansion\n")
    md.append("### Horizon label counts\n")
    md.append(pd.DataFrame(horizon_summary).sort_values(["endpoint", "horizon_years"])
              .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("### Per-run results (sorted by survival C-index per endpoint)\n")
    if not runs_df.empty:
        md.append(runs_df.sort_values(["endpoint", "surv_cindex_mean"], ascending=[True, False])
                  .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## C. Endpoint-specific horizon ensembles\n")
    md.append(f"- **hepatic ensemble** (greedy, equal-weight): "
              f"{hep_sel} -> survival C-index {hep_ens_ci:.4f}")
    md.append(f"- **death ensemble** (greedy, equal-weight): "
              f"{dea_sel} -> survival C-index {dea_ens_ci:.4f}")
    md.append("")

    md.append("## D. Blend grid vs phase3_9_horizon_blend\n")
    blend_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("h_test", "d_test", "h_oof", "d_oof",
                                                    "components_h", "components_d",
                                                    "weights_h", "weights_d")}
        for r in blend_rows
    ])
    md.append(blend_df.sort_values("weighted_oof", ascending=False)
              .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## E. Top horizon-model feature importances\n")
    md.append("Top 15 LightGBM features for the best hepatic h3 model "
              "(`v3_hepatic_schema__hepatic__h3__lgbm_binary`, seed 0):\n")
    try:
        from .models._preprocess import fill_for_tree
        import lightgbm as lgb
        fs_blob = feature_sets["v3_hepatic_schema"]
        Xtr_pre = fill_for_tree(fs_blob["train"]).astype(np.float32, errors="ignore")
        lab3 = label_blobs[("hepatic", 3.0, "exclude")]
        m = lab3.mask
        ytr = lab3.label[m].astype(int)
        n_pos = max(int(ytr.sum()), 1)
        n_neg = max(int((1 - ytr).sum()), 1)
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, scale_pos_weight=n_neg / n_pos, random_state=0, verbose=-1,
        )
        clf.fit(Xtr_pre.iloc[m].to_numpy(), ytr)
        importances = pd.Series(clf.feature_importances_, index=Xtr_pre.columns)
        top = importances.sort_values(ascending=False).head(15)
        md.append(top.to_frame("gain").to_markdown(floatfmt=".0f"))
        md.append("")
    except Exception as e:  # noqa: BLE001
        md.append(f"_failed: {e}_\n")

    out_dec = cfg.REPORTS_DIR / "phase3_10_horizon_blend_decomposition.md"
    out_dec.write_text("\n".join(md))
    _LOG.info("wrote %s", out_dec)

    # ------------------------------------------------------------------
    # 8. Recommendation report
    # ------------------------------------------------------------------

    rec: list[str] = []
    rec.append("# Phase 3.10 — submission recommendation\n")
    rec.append("Reference: `phase3_9_horizon_blend` (public LB 0.90419, weighted OOF "
               f"{p9_w:.4f}, hepatic {p9_h:.4f}, death {p9_d:.4f}).\n")

    if candidate_v2 is None and candidate_robust is None:
        rec.append("## Recommendation: do not submit\n")
        rec.append("No horizon variant beat `phase3_9_horizon_blend` on OOF or fold "
                   "stability by a margin large enough to be meaningful. Hold the "
                   "current public-LB best and continue investigating.\n")
    else:
        primary = candidate_v2 if candidate_v2 is not None else candidate_robust
        primary_path = candidate_v2_path if candidate_v2 is not None else candidate_robust_path
        rec.append("## Recommendation: submit ONE candidate\n")
        rec.append(f"**File**: `{primary_path}`\n")
        rec.append(f"**Blend**: `{primary['blend']}` (alpha={primary['alpha']}, side={primary['side']})\n")
        rec.append(f"- weighted OOF: {primary['weighted_oof']:.4f} "
                   f"(Δ vs Phase 3.9 = {primary['weighted_oof']-p9_w:+.4f})")
        rec.append(f"- hepatic OOF:  {primary['hep_oof']:.4f} "
                   f"(Δ vs Phase 3.9 = {primary['hep_oof']-p9_h:+.4f})")
        rec.append(f"- death OOF:    {primary['death_oof']:.4f} "
                   f"(Δ vs Phase 3.9 = {primary['death_oof']-p9_d:+.4f})")
        rec.append(f"- hep fold std/min: {primary['hep_fold_std']:.4f}/{primary['hep_fold_min']:.4f}")
        rec.append(f"- dea fold std/min: {primary['dea_fold_std']:.4f}/{primary['dea_fold_min']:.4f}")
        rec.append(f"- rank-corr w/ phase3_9_horizon_blend test: hep={primary['rho_h_test_p9']:.3f}, "
                   f"dea={primary['rho_d_test_p9']:.3f}, weighted={primary['rho_test_p9']:.3f}")
        rec.append("")
        rec.append("### How it differs from phase3_9_horizon_blend\n")
        rec.append(f"- hepatic components: {primary['components_h']}")
        rec.append(f"- death components:   {primary['components_d']}")
        rec.append(f"- weights: hep={primary['weights_h']}, dea={primary['weights_d']}")
        rec.append("")
        rec.append("### Public-LB outcome interpretation\n")
        rec.append(
            "- If LB > 0.90419 by ≥ +0.001 → horizon-ensemble extension transferred; "
            "Phase 3.11 should iterate on the ensemble selection.\n"
            "- If LB ≈ 0.90419 (within ±0.001) → the OOF gain didn't transfer; private "
            "split has different horizon mass; treat horizons as exhausted.\n"
            "- If LB < 0.90419 by ≥ -0.002 → likely a private-vs-public horizon mismatch; "
            "fall back to Phase 3.9 candidate and explore non-horizon directions.\n"
        )
        if candidate_robust is not None and candidate_v2 is not None:
            rec.append("### Why we did not promote the robust variant\n")
            rec.append("The OOF-best variant already had competitive fold stability; the "
                       "second candidate provides a smaller-OOF fallback (saved at "
                       f"`{candidate_robust_path}`) but should not be submitted unless "
                       "Phase 3.11 reveals a private-vs-public mismatch.\n")
    out_rec = cfg.REPORTS_DIR / "phase3_10_submission_recommendation.md"
    out_rec.write_text("\n".join(rec))
    _LOG.info("wrote %s", out_rec)


if __name__ == "__main__":
    main()
