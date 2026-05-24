"""Phase 3.12 — horizon-label refinement around the public best.

Anchor: ``phase3_10_horizon_blend_v2`` (LB **0.91093**).

What we add over Phase 3.9 / 3.10:

1. Fractional hepatic horizons (2.5, 3.5, 4.5) and fractional death horizons
   (5.5, 6.5). Existing integer-year horizons are reused on disk.
2. Censoring-policy comparison: ``exclude`` (default), ``soft_weight`` (per-row
   weight = clip(time/H, 0.1, 1.0)), and ``downweight`` (fixed 0.25).
3. Models restricted to LightGBM-binary for hepatic horizons and
   CatBoost-binary for death horizons; XGBoost-binary added selectively for
   death where it was competitive in earlier phases (current_state_v2,
   biomarker_only, v3_hepatic_schema).
4. Multi-seed (4 extra seeds) for any new horizon configuration that
   matches or beats the existing best single-horizon model on survival
   C-index.
5. Endpoint-specific horizon ensembles via greedy rank selection.
6. Blend grid against the anchor (alphas {0.90, 0.85, 0.80, 0.75},
   sides {hep_only, dea_only, both}, plus a greedy capped-at-25%-horizon
   blend).

Output:
- reports/phase3_12_horizon_refinement.md
- experiments/outputs/phase3_12_horizon/<run_label>/(oof.csv,test.csv)
- up to two candidates:
    * submissions/<ts>_phase3_12_horizon_refined_v3.csv
    * submissions/<ts>_phase3_12_horizon_hepatic_only.csv

No target/event/censoring-age-derived feature construction; the horizon
enters only through the binary label and the per-row weight.
"""
from __future__ import annotations

import json
import time as _time
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
from .utils import get_logger
from .validation import build_folds

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature sets (same four as Phase 3.9 / 3.10)
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
# Anchor (phase3_10_horizon_blend_v2) reconstruction
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


def _load_anchor(ds, hep, death, n_train: int, n_test: int):
    """Reconstruct phase3_10_horizon_blend_v2 OOF (hep, dea) and read test CSV."""
    p10_meta_path = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_10_horizon_blend_v2.json"))[-1]
    meta = json.loads(p10_meta_path.read_text())
    blend_csv = p10_meta_path.with_suffix(".csv")
    pub = pd.read_csv(blend_csv)
    h_test = pub[cfg.SUB_HEPATIC_COL].to_numpy()
    d_test = pub[cfg.SUB_DEATH_COL].to_numpy()

    v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)
    weights_h = meta["weights_hepatic"]
    weights_d = meta["weights_death"]
    pool_h_oof: dict[str, np.ndarray] = {"_v3": v3_h_oof}
    pool_d_oof: dict[str, np.ndarray] = {"_v3": v3_d_oof}
    for k in weights_h:
        if k == "_v3":
            continue
        loaded = _load_horizon_artifact(k)
        if loaded is None:
            _LOG.warning("missing horizon artifact %s", k)
            continue
        pool_h_oof[k] = loaded[0]
    for k in weights_d:
        if k == "_v3":
            continue
        loaded = _load_horizon_artifact(k)
        if loaded is None:
            _LOG.warning("missing horizon artifact %s", k)
            continue
        pool_d_oof[k] = loaded[0]
    h_oof = rank_average(pool_h_oof, weights={k: weights_h[k] for k in pool_h_oof})
    d_oof = rank_average(pool_d_oof, weights={k: weights_d[k] for k in pool_d_oof})
    return h_oof, h_test, d_oof, d_test, meta


# ---------------------------------------------------------------------------
# CV runner
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
            _LOG.warning("fold %s seed=%d failed: %s", model_name, seed, e)

    test = test_sum / test_n if test_n else np.full(n_test, np.nan)
    return oof, test


# ---------------------------------------------------------------------------
# Helpers (same as Phase 3.10)
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
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 5:
        return float("nan")
    return float(df["a"].rank(pct=True).corr(df["b"].rank(pct=True), method="spearman"))


def _greedy_horizon_ensemble(
    candidates: dict[str, np.ndarray],
    event: np.ndarray,
    time: np.ndarray,
    *,
    min_individual_ci: float = 0.55,
    max_size: int = 8,
) -> tuple[list[str], dict[str, float], float]:
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
        for k, v in pool.items():
            if k in selected:
                continue
            arrs = selected_arrs + [v]
            blended = rank_average({f"k{i}": a for i, a in enumerate(arrs)})
            ci = _surv_ci(event, time, blended)
            if ci > best_new_ci + 1e-5:
                best_new_ci = ci
                best_k = k
        if best_k is None:
            break
        selected.append(best_k)
        selected_arrs.append(pool[best_k])
        best_ci = best_new_ci
    weights = {k: 1.0 / len(selected) for k in selected} if selected else {}
    return selected, weights, best_ci


def _greedy_capped(v3_oof, v3_test, pool, event, time, cap=0.25):
    """Greedy rank-blend with v3 anchor and total horizon contribution ≤ ``cap``."""
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
            horizon_w_now = sum(w for kk, w in chosen_w.items() if kk != "_v3")
            room = cap - horizon_w_now
            if room <= 0.001:
                continue
            for dw in (0.05, 0.10, 0.15):
                if dw > room + 1e-9:
                    continue
                new_w = dict(chosen_w)
                new_w[k] = dw
                s = sum(new_w.values())
                new_w = {kk: vv / s for kk, vv in new_w.items()}
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
    return chosen_w, cur_oof, cur_test, cur_ci


# ---------------------------------------------------------------------------
# Loader for prior-phase horizon artifacts
# ---------------------------------------------------------------------------

def _load_phase_artifacts(out_root: Path, ds, n_train: int, n_test: int) -> dict[str, dict[str, np.ndarray]]:
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    pid_to_pos = {v: i for i, v in enumerate(ds.train_df[cfg.PATIENT_ID_COL].values)}
    tid_to_pos = {v: i for i, v in enumerate(ds.test_df[cfg.TRUSTII_ID_COL].values)}
    if not out_root.exists():
        return artifacts
    for run_dir in sorted(out_root.iterdir()):
        if not run_dir.is_dir():
            continue
        oof_p = run_dir / "oof.csv"
        test_p = run_dir / "test.csv"
        if not (oof_p.exists() and test_p.exists()):
            continue
        try:
            o_df = pd.read_csv(oof_p)
            t_df = pd.read_csv(test_p)
            oof = np.full(n_train, np.nan)
            test = np.full(n_test, np.nan)
            pos = o_df[cfg.PATIENT_ID_COL].map(pid_to_pos).to_numpy()
            mask = ~pd.isna(pos)
            oof[pos[mask].astype(int)] = o_df["oof"].to_numpy()[mask]
            tpos = t_df[cfg.TRUSTII_ID_COL].map(tid_to_pos).to_numpy()
            tmask = ~pd.isna(tpos)
            test[tpos[tmask].astype(int)] = t_df["test"].to_numpy()[tmask]
            artifacts[run_dir.name] = {"oof": oof, "test": test}
        except Exception as e:  # noqa: BLE001
            _LOG.warning("could not load %s: %s", run_dir, e)
    return artifacts


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    cfg.ensure_dirs()
    p9_root = cfg.EXPERIMENT_OUTPUTS / "phase3_9_horizon"
    p10_root = cfg.EXPERIMENT_OUTPUTS / "phase3_10_horizon"
    p12_root = cfg.EXPERIMENT_OUTPUTS / "phase3_12_horizon"
    p12_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)

    feature_sets = assemble_feature_sets(ds)

    # Reference: phase3_10_horizon_blend_v2 (anchor).
    a_h_oof, a_h_test, a_d_oof, a_d_test, anchor_meta = _load_anchor(ds, hep, death, n_train, n_test)
    a_h_ci = _surv_ci(hep.event, hep.time, a_h_oof)
    a_d_ci = _surv_ci(death.event, death.time, a_d_oof)
    a_w = weighted_score(a_h_ci, a_d_ci)
    a_h_fold = _fold_ci(hep.event, hep.time, a_h_oof, splits)
    a_d_fold = _fold_ci(death.event, death.time, a_d_oof, splits)
    _LOG.info("Anchor phase3_10_v2: hep=%.4f dea=%.4f weighted=%.4f", a_h_ci, a_d_ci, a_w)

    # ------------------------------------------------------------------
    # New horizons + policies. Reuse all phase3_9 / phase3_10 OOFs.
    # ------------------------------------------------------------------

    runs: list[dict] = []
    new_artifacts: dict[str, dict[str, np.ndarray]] = {}

    p9_artifacts = _load_phase_artifacts(p9_root, ds, n_train, n_test)
    p10_artifacts = _load_phase_artifacts(p10_root, ds, n_train, n_test)
    prior_artifacts = {**p9_artifacts, **p10_artifacts}
    _LOG.info("loaded %d prior horizon artifacts", len(prior_artifacts))

    horizons_hep_new = [2.5, 3.5, 4.5]
    horizons_dea_new = [5.5, 6.5]
    horizons_hep_all = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
    horizons_dea_all = [4.0, 5.0, 5.5, 6.0, 6.5]

    # Pre-compute horizon labels for each (endpoint, horizon, mode).
    label_blobs: dict[tuple[str, float, str], object] = {}
    horizon_summary: list[dict] = []
    for ep_name, ep in [("hepatic", hep), ("death", death)]:
        hzs = horizons_hep_all if ep_name == "hepatic" else horizons_dea_all
        for H in hzs:
            for mode in ("exclude", "soft_weight", "downweight"):
                lab = build_horizon_labels(ep, H, censored_mode=mode)
                label_blobs[(ep_name, H, mode)] = lab
            le = label_blobs[(ep_name, H, "exclude")]
            horizon_summary.append({
                "endpoint": ep_name,
                "horizon_years": H,
                "n_total": le.n_total,
                "n_usable_exclude": le.n_usable,
                "n_positive_exclude": le.n_positive,
                "n_censored_before_h": le.notes["n_censored_before_horizon"],
                "positive_rate": le.n_positive / max(le.n_usable, 1),
            })

    # Step B: new fractional hepatic horizons (LightGBM only) + new fractional
    # death horizons (CatBoost primary, plus XGBoost on top fs).
    hep_models = ["lgbm_binary"]
    dea_models = ["catboost_binary", "xgb_binary"]

    def _process_run(label: str, oof, test, fs_name, ep_name, H, model_name,
                     n_features, lab, mode, seed=0, dt=0.0):
        ep = hep if ep_name == "hepatic" else death
        ci_surv = _surv_ci(ep.event, ep.time, oof)
        mean_, std_, min_ = _fold_ci(ep.event, ep.time, oof, splits)
        use = lab.mask & np.isfinite(oof)
        auc = _safe_auc(lab.label[use], oof[use]) if use.sum() else float("nan")
        rho_anchor_oof = _spearman(oof, a_h_oof if ep_name == "hepatic" else a_d_oof)
        rho_anchor_test = _spearman(test, a_h_test if ep_name == "hepatic" else a_d_test)
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
            "rho_oof_anchor": rho_anchor_oof,
            "rho_test_anchor": rho_anchor_test,
        })
        new_artifacts[label] = {"oof": oof, "test": test}

    def _persist(label, oof, test):
        run_dir = p12_root / label
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({cfg.PATIENT_ID_COL: ds.train_df[cfg.PATIENT_ID_COL].values, "oof": oof}).to_csv(run_dir / "oof.csv", index=False)
        pd.DataFrame({cfg.TRUSTII_ID_COL: ds.test_df[cfg.TRUSTII_ID_COL].values, "test": test}).to_csv(run_dir / "test.csv", index=False)

    def _label(fs_name, ep_name, H, model_name, mode, seed):
        h_str = f"h{H:g}".replace(".", "p")
        suffix = ""
        if mode != "exclude":
            suffix += f"__{mode}"
        if seed != 0:
            suffix += f"__s{seed}"
        return f"{fs_name}__{ep_name}__{h_str}__{model_name}{suffix}"

    # Step B.1: new hepatic fractional horizons (LGBM only, exclude mode), 4 fs
    _LOG.info("Step B.1: new fractional hepatic horizons (exclude mode, LGBM)")
    for H in horizons_hep_new:
        lab = label_blobs[("hepatic", H, "exclude")]
        if lab.n_positive < 5:
            _LOG.warning("skip hep H=%.1f: only %d positives", H, lab.n_positive)
            continue
        for fs_name, blob in feature_sets.items():
            for model_name in hep_models:
                label = _label(fs_name, "hepatic", H, model_name, "exclude", 0)
                t0 = _time.time()
                try:
                    oof, test = _train_one_horizon(
                        model_name, blob["train"], blob["test"],
                        lab.label, lab.mask, lab.weight, splits,
                        n_train=n_train, n_test=n_test, seed=0,
                    )
                except Exception as e:  # noqa: BLE001
                    _LOG.error("%s failed: %s", label, e)
                    continue
                dt = _time.time() - t0
                _persist(label, oof, test)
                _process_run(label, oof, test, fs_name, "hepatic", H, model_name,
                             blob["train"].shape[1], lab, "exclude", 0, dt)
                _LOG.info("%s survC=%.4f AUC=%.4f (%.1fs)", label,
                          runs[-1]["surv_cindex_mean"], runs[-1]["horizon_auc"], dt)

    # Step B.2: new death fractional horizons (CatBoost + XGB on current_state_v2 / v3_hepatic_schema)
    _LOG.info("Step B.2: new fractional death horizons (exclude mode, CatBoost+XGB on top fs)")
    dea_xgb_fs = {"current_state_v2", "v3_hepatic_schema"}
    for H in horizons_dea_new:
        lab = label_blobs[("death", H, "exclude")]
        if lab.n_positive < 5:
            _LOG.warning("skip dea H=%.1f: only %d positives", H, lab.n_positive)
            continue
        for fs_name, blob in feature_sets.items():
            for model_name in dea_models:
                if model_name == "xgb_binary" and fs_name not in dea_xgb_fs:
                    continue
                label = _label(fs_name, "death", H, model_name, "exclude", 0)
                t0 = _time.time()
                try:
                    oof, test = _train_one_horizon(
                        model_name, blob["train"], blob["test"],
                        lab.label, lab.mask, lab.weight, splits,
                        n_train=n_train, n_test=n_test, seed=0,
                    )
                except Exception as e:  # noqa: BLE001
                    _LOG.error("%s failed: %s", label, e)
                    continue
                dt = _time.time() - t0
                _persist(label, oof, test)
                _process_run(label, oof, test, fs_name, "death", H, model_name,
                             blob["train"].shape[1], lab, "exclude", 0, dt)
                _LOG.info("%s survC=%.4f AUC=%.4f (%.1fs)", label,
                          runs[-1]["surv_cindex_mean"], runs[-1]["horizon_auc"], dt)

    # Step C: censoring policy comparison. For each (endpoint, horizon) we pick
    # the best fs+model from the exclude mode and run it under soft_weight and
    # downweight policies.
    _LOG.info("Step C: censoring policy comparison on best fs+model per horizon")
    runs_df_so_far = pd.DataFrame(runs) if runs else pd.DataFrame(columns=["endpoint", "horizon_years"])

    def _all_horizon_runs_for(ep_name: str, H: float, mode_filter: str = "exclude"):
        rows = [r for r in runs if r["endpoint"] == ep_name and abs(r["horizon_years"] - H) < 1e-6 and r["mode"] == mode_filter]
        # Add prior phase artifacts (always exclude mode and seed 0/non-multi)
        for k, v in prior_artifacts.items():
            parts = k.split("__")
            if len(parts) < 4:
                continue
            ep_tok = parts[1]
            h_tok = parts[2]
            try:
                h_val = float(h_tok.replace("h", "").replace("p", "."))
            except ValueError:
                continue
            if "__dw" in k or "__soft_weight" in k:
                continue
            if ep_tok != ep_name or abs(h_val - H) > 1e-6:
                continue
            ep = hep if ep_name == "hepatic" else death
            ci = _surv_ci(ep.event, ep.time, v["oof"])
            if not np.isfinite(ci):
                continue
            rows.append({
                "label": k, "feature_set": parts[0], "model": parts[3].split("__")[0] if "__" in parts[3] else parts[3],
                "endpoint": ep_tok, "horizon_years": h_val, "mode": "exclude",
                "surv_cindex_mean": ci, "_oof_external": v["oof"], "_test_external": v["test"],
            })
        return rows

    for ep_name, ep, hzs, models_used in [
        ("hepatic", hep, horizons_hep_all, hep_models),
        ("death",   death, horizons_dea_all, dea_models),
    ]:
        for H in hzs:
            cands = _all_horizon_runs_for(ep_name, H, "exclude")
            cands = [c for c in cands if np.isfinite(c["surv_cindex_mean"])]
            if not cands:
                continue
            best = max(cands, key=lambda r: r["surv_cindex_mean"])
            best_fs = best["feature_set"]
            # Resolve model from label parts: e.g. "lgbm_binary" or "catboost_binary"
            best_label_parts = best["label"].split("__")
            if len(best_label_parts) >= 4:
                # e.g. fs__ep__h3__lgbm_binary[__suffix]
                # model is the 4th piece up to next __ that isn't a suffix
                base_model = best_label_parts[3]
            else:
                base_model = best.get("model", "lgbm_binary")
            if best_fs not in feature_sets:
                continue
            fs_blob = feature_sets[best_fs]
            for new_mode in ("soft_weight", "downweight"):
                lab = label_blobs[(ep_name, H, new_mode)]
                if lab.n_positive < 5:
                    continue
                label = _label(best_fs, ep_name, H, base_model, new_mode, 0)
                if label in {r["label"] for r in runs}:
                    continue
                t0 = _time.time()
                try:
                    oof, test = _train_one_horizon(
                        base_model, fs_blob["train"], fs_blob["test"],
                        lab.label, lab.mask, lab.weight, splits,
                        n_train=n_train, n_test=n_test, seed=0, use_sample_weight=True,
                    )
                except Exception as e:  # noqa: BLE001
                    _LOG.error("%s failed: %s", label, e)
                    continue
                dt = _time.time() - t0
                _persist(label, oof, test)
                _process_run(label, oof, test, best_fs, ep_name, H, base_model,
                             fs_blob["train"].shape[1], lab, new_mode, 0, dt)
                _LOG.info("policy %s survC=%.4f (anchor=%s)", label,
                          runs[-1]["surv_cindex_mean"], best["label"])

    # Step D: multi-seed for any new horizon configuration that beats the
    # exclude-mode Phase 3.9/3.10 best for the same horizon.
    _LOG.info("Step D: multi-seed for promising configurations")
    seeds_extra = [1, 2, 3, 4]
    multi_seed_targets = []
    for ep_name, ep, hzs in [("hepatic", hep, horizons_hep_all), ("death", death, horizons_dea_all)]:
        for H in hzs:
            new_runs_here = [r for r in runs if r["endpoint"] == ep_name and abs(r["horizon_years"] - H) < 1e-6 and r["seed"] == 0]
            if not new_runs_here:
                continue
            best_run = max(new_runs_here, key=lambda r: r["surv_cindex_mean"])
            # Compare to the prior-phase exclude best for the same horizon.
            prior_for_H = []
            for k, v in prior_artifacts.items():
                parts = k.split("__")
                if len(parts) < 4 or "__dw" in k or "__soft_weight" in k:
                    continue
                if parts[1] != ep_name:
                    continue
                try:
                    h_val = float(parts[2].replace("h", "").replace("p", "."))
                except ValueError:
                    continue
                if abs(h_val - H) > 1e-6:
                    continue
                ci = _surv_ci(ep.event, ep.time, v["oof"])
                if np.isfinite(ci):
                    prior_for_H.append({"label": k, "ci": ci})
            prior_best_ci = max((p["ci"] for p in prior_for_H), default=float("-inf"))
            if best_run["surv_cindex_mean"] >= prior_best_ci - 0.005:  # match or beat
                multi_seed_targets.append((best_run["feature_set"], ep_name, H,
                                            best_run["model"], best_run["mode"]))

    for fs_name, ep_name, H, model_name, mode in multi_seed_targets:
        ep = hep if ep_name == "hepatic" else death
        lab = label_blobs[(ep_name, H, mode)]
        blob = feature_sets[fs_name]
        for seed in seeds_extra:
            label = _label(fs_name, ep_name, H, model_name, mode, seed)
            if label in {r["label"] for r in runs}:
                continue
            t0 = _time.time()
            try:
                oof, test = _train_one_horizon(
                    model_name, blob["train"], blob["test"],
                    lab.label, lab.mask, lab.weight, splits,
                    n_train=n_train, n_test=n_test, seed=seed,
                    use_sample_weight=(mode != "exclude"),
                )
            except Exception as e:  # noqa: BLE001
                _LOG.error("%s failed: %s", label, e)
                continue
            dt = _time.time() - t0
            _persist(label, oof, test)
            _process_run(label, oof, test, fs_name, ep_name, H, model_name,
                         blob["train"].shape[1], lab, mode, seed, dt)
            _LOG.info("seed-bag %s survC=%.4f (%.1fs)", label, runs[-1]["surv_cindex_mean"], dt)

    runs_df = pd.DataFrame(runs)
    if not runs_df.empty:
        runs_df.to_csv(p12_root / "all_runs.csv", index=False)

    # ------------------------------------------------------------------
    # All artifacts: prior + new
    # ------------------------------------------------------------------

    all_artifacts: dict[str, dict[str, np.ndarray]] = {}
    all_artifacts.update(prior_artifacts)
    all_artifacts.update(new_artifacts)

    # ------------------------------------------------------------------
    # Step E: endpoint-specific horizon ensembles (greedy rank)
    # ------------------------------------------------------------------

    def _pool_for(ep_name: str) -> dict[str, np.ndarray]:
        ep = hep if ep_name == "hepatic" else death
        out = {}
        for k, v in all_artifacts.items():
            parts = k.split("__")
            if len(parts) < 4 or parts[1] != ep_name:
                continue
            ci = _surv_ci(ep.event, ep.time, v["oof"])
            if not np.isfinite(ci) or ci < 0.55:
                continue
            out[k] = v["oof"]
        return out

    hep_pool = _pool_for("hepatic")
    dea_pool = _pool_for("death")
    hep_sel, hep_w, hep_ens_ci = _greedy_horizon_ensemble(hep_pool, hep.event, hep.time)
    dea_sel, dea_w, dea_ens_ci = _greedy_horizon_ensemble(dea_pool, death.event, death.time)
    _LOG.info("hep ensemble: %s -> %.4f", hep_sel, hep_ens_ci)
    _LOG.info("dea ensemble: %s -> %.4f", dea_sel, dea_ens_ci)

    hep_ens_oof = rank_average({k: all_artifacts[k]["oof"] for k in hep_sel}, weights=hep_w) if hep_sel else np.full(n_train, np.nan)
    hep_ens_test = rank_average({k: all_artifacts[k]["test"] for k in hep_sel}, weights=hep_w) if hep_sel else np.full(n_test, np.nan)
    dea_ens_oof = rank_average({k: all_artifacts[k]["oof"] for k in dea_sel}, weights=dea_w) if dea_sel else np.full(n_train, np.nan)
    dea_ens_test = rank_average({k: all_artifacts[k]["test"] for k in dea_sel}, weights=dea_w) if dea_sel else np.full(n_test, np.nan)

    # Best single horizon per endpoint (sanity).
    hep_runs = [r for r in runs if r["endpoint"] == "hepatic" and np.isfinite(r["surv_cindex_mean"])]
    dea_runs = [r for r in runs if r["endpoint"] == "death" and np.isfinite(r["surv_cindex_mean"])]
    hep_best = max(hep_runs, key=lambda r: r["surv_cindex_mean"]) if hep_runs else None
    dea_best = max(dea_runs, key=lambda r: r["surv_cindex_mean"]) if dea_runs else None

    # ------------------------------------------------------------------
    # Step E.2: blends with anchor (phase3_10_v2)
    # ------------------------------------------------------------------

    blend_rows: list[dict] = []

    def _eval_blend(name, h_oof, d_oof, h_test, d_test, comp_h, comp_d, w_h, w_d, alpha, side, source):
        ci_h = _surv_ci(hep.event, hep.time, h_oof)
        ci_d = _surv_ci(death.event, death.time, d_oof)
        h_mean, h_std, h_min = _fold_ci(hep.event, hep.time, h_oof, splits)
        d_mean, d_std, d_min = _fold_ci(death.event, death.time, d_oof, splits)
        rho_h_a_test = _spearman(h_test, a_h_test)
        rho_d_a_test = _spearman(d_test, a_d_test)
        rho_h_a_oof = _spearman(h_oof, a_h_oof)
        rho_d_a_oof = _spearman(d_oof, a_d_oof)
        blend_rows.append({
            "blend": name, "alpha": alpha, "side": side, "source": source,
            "hep_oof": ci_h, "death_oof": ci_d, "weighted_oof": weighted_score(ci_h, ci_d),
            "hep_fold_std": h_std, "hep_fold_min": h_min,
            "dea_fold_std": d_std, "dea_fold_min": d_min,
            "rho_h_test_anchor": rho_h_a_test, "rho_d_test_anchor": rho_d_a_test,
            "rho_h_oof_anchor": rho_h_a_oof,   "rho_d_oof_anchor": rho_d_a_oof,
            "components_h": comp_h, "components_d": comp_d,
            "weights_h": w_h, "weights_d": w_d,
            "h_test": h_test, "d_test": d_test,
            "h_oof": h_oof, "d_oof": d_oof,
        })

    # Anchor itself for reference
    _eval_blend("anchor_phase3_10_v2", a_h_oof, a_d_oof, a_h_test, a_d_test,
                [], [], {"anchor": 1.0}, {"anchor": 1.0}, alpha=1.0, side="none", source="anchor")

    if hep_best is not None and dea_best is not None and hep_sel and dea_sel:
        hep_best_oof = all_artifacts[hep_best["label"]]["oof"] if hep_best["label"] in all_artifacts else None
        hep_best_test = all_artifacts[hep_best["label"]]["test"] if hep_best["label"] in all_artifacts else None
        dea_best_oof = all_artifacts[dea_best["label"]]["oof"] if dea_best["label"] in all_artifacts else None
        dea_best_test = all_artifacts[dea_best["label"]]["test"] if dea_best["label"] in all_artifacts else None
        for alpha in (0.90, 0.85, 0.80, 0.75):
            for side in ("hep_only", "dea_only", "both"):
                # ensemble blend
                if side in ("hep_only", "both"):
                    h_oof_e = _blend_two(a_h_oof, hep_ens_oof, alpha)
                    h_test_e = _blend_two(a_h_test, hep_ens_test, alpha)
                    comp_h_e = list(hep_sel)
                    w_h_e = {"anchor_h": alpha, **{k: (1 - alpha) * w for k, w in hep_w.items()}}
                else:
                    h_oof_e = a_h_oof.copy()
                    h_test_e = a_h_test.copy()
                    comp_h_e = []
                    w_h_e = {"anchor_h": 1.0}
                if side in ("dea_only", "both"):
                    d_oof_e = _blend_two(a_d_oof, dea_ens_oof, alpha)
                    d_test_e = _blend_two(a_d_test, dea_ens_test, alpha)
                    comp_d_e = list(dea_sel)
                    w_d_e = {"anchor_d": alpha, **{k: (1 - alpha) * w for k, w in dea_w.items()}}
                else:
                    d_oof_e = a_d_oof.copy()
                    d_test_e = a_d_test.copy()
                    comp_d_e = []
                    w_d_e = {"anchor_d": 1.0}
                _eval_blend(f"ensemble_alpha={alpha}_{side}", h_oof_e, d_oof_e,
                            h_test_e, d_test_e, comp_h_e, comp_d_e, w_h_e, w_d_e,
                            alpha, side, "ensemble")
                # single-best blend
                if hep_best_oof is not None and dea_best_oof is not None:
                    if side in ("hep_only", "both"):
                        h_oof_s = _blend_two(a_h_oof, hep_best_oof, alpha)
                        h_test_s = _blend_two(a_h_test, hep_best_test, alpha)
                        comp_h_s = [hep_best["label"]]
                        w_h_s = {"anchor_h": alpha, hep_best["label"]: 1 - alpha}
                    else:
                        h_oof_s = a_h_oof.copy(); h_test_s = a_h_test.copy()
                        comp_h_s = []; w_h_s = {"anchor_h": 1.0}
                    if side in ("dea_only", "both"):
                        d_oof_s = _blend_two(a_d_oof, dea_best_oof, alpha)
                        d_test_s = _blend_two(a_d_test, dea_best_test, alpha)
                        comp_d_s = [dea_best["label"]]
                        w_d_s = {"anchor_d": alpha, dea_best["label"]: 1 - alpha}
                    else:
                        d_oof_s = a_d_oof.copy(); d_test_s = a_d_test.copy()
                        comp_d_s = []; w_d_s = {"anchor_d": 1.0}
                    _eval_blend(f"single_alpha={alpha}_{side}", h_oof_s, d_oof_s,
                                h_test_s, d_test_s, comp_h_s, comp_d_s, w_h_s, w_d_s,
                                alpha, side, "best_single")

        # Greedy capped at 25% horizon contribution, layered on top of the anchor.
        hep_pool_for_greedy = {k: {"oof": all_artifacts[k]["oof"], "test": all_artifacts[k]["test"]}
                                for k in hep_pool}
        dea_pool_for_greedy = {k: {"oof": all_artifacts[k]["oof"], "test": all_artifacts[k]["test"]}
                                for k in dea_pool}
        g_h_w, g_h_oof, g_h_test, g_h_ci = _greedy_capped(
            a_h_oof, a_h_test, hep_pool_for_greedy, hep.event, hep.time, cap=0.25)
        g_d_w, g_d_oof, g_d_test, g_d_ci = _greedy_capped(
            a_d_oof, a_d_test, dea_pool_for_greedy, death.event, death.time, cap=0.25)
        comp_h_g = [k for k in g_h_w if k != "_v3"]
        comp_d_g = [k for k in g_d_w if k != "_v3"]
        _eval_blend("greedy_cap25_both", g_h_oof, g_d_oof, g_h_test, g_d_test,
                    comp_h_g, comp_d_g, g_h_w, g_d_w, alpha=None, side="both", source="greedy_capped")
        # hepatic-only greedy
        _eval_blend("greedy_cap25_hep_only", g_h_oof, a_d_oof, g_h_test, a_d_test,
                    comp_h_g, [], g_h_w, {"anchor_d": 1.0}, alpha=None, side="hep_only", source="greedy_capped")
        # death-only greedy
        _eval_blend("greedy_cap25_dea_only", a_h_oof, g_d_oof, a_h_test, g_d_test,
                    [], comp_d_g, {"anchor_h": 1.0}, g_d_w, alpha=None, side="dea_only", source="greedy_capped")

    # ------------------------------------------------------------------
    # Step F: candidate selection
    # ------------------------------------------------------------------

    # Candidate 1: best-OOF blend that meets the +0.002 weighted or +0.003 hep threshold.
    candidate_v3 = None
    for row in sorted(blend_rows, key=lambda r: -r["weighted_oof"]):
        if row["blend"] == "anchor_phase3_10_v2":
            continue
        delta_w = row["weighted_oof"] - a_w
        delta_h = row["hep_oof"] - a_h_ci
        if delta_w >= 0.002 or delta_h >= 0.003:
            candidate_v3 = row
            break

    # Candidate 2: hepatic-only refinement that meaningfully improves hep without changing death.
    candidate_hep_only = None
    for row in sorted(blend_rows, key=lambda r: -r["hep_oof"]):
        if row["blend"] == "anchor_phase3_10_v2":
            continue
        if row["side"] != "hep_only":
            continue
        delta_h = row["hep_oof"] - a_h_ci
        # death must be unchanged (within 0.0005)
        delta_d = row["death_oof"] - a_d_ci
        if delta_h >= 0.003 and abs(delta_d) <= 0.0005:
            candidate_hep_only = row
            break

    # Avoid duplicate emission.
    if candidate_v3 is not None and candidate_hep_only is not None and \
       candidate_v3["blend"] == candidate_hep_only["blend"]:
        candidate_hep_only = None

    def _emit(row, label, rationale):
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=row["h_test"], risk_death=row["d_test"],
            sample_submission=ds.sample_submission, model_name=label,
        )
        meta = {
            "label": label, "blend_id": row["blend"],
            "alpha": row["alpha"], "side": row["side"],
            "components_hepatic": row["components_h"],
            "components_death":   row["components_d"],
            "weights_hepatic":    row["weights_h"],
            "weights_death":      row["weights_d"],
            "hepatic_oof": row["hep_oof"], "death_oof": row["death_oof"],
            "weighted_oof": row["weighted_oof"],
            "hepatic_fold_std": row["hep_fold_std"], "hepatic_fold_min": row["hep_fold_min"],
            "death_fold_std":   row["dea_fold_std"], "death_fold_min":   row["dea_fold_min"],
            "rho_test_anchor":  {"hepatic": row["rho_h_test_anchor"],
                                  "death":   row["rho_d_test_anchor"]},
            "rho_oof_anchor":   {"hepatic": row["rho_h_oof_anchor"],
                                  "death":   row["rho_d_oof_anchor"]},
            "delta_vs_anchor": {
                "weighted_oof": row["weighted_oof"] - a_w,
                "hepatic_oof":  row["hep_oof"] - a_h_ci,
                "death_oof":    row["death_oof"] - a_d_ci,
            },
            "uses_target_derived_features": False,
            "rationale": rationale,
            "submission_csv": str(sub_path),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        return sub_path, meta

    candidate_v3_path = candidate_v3_meta = None
    candidate_hep_only_path = candidate_hep_only_meta = None
    if candidate_v3 is not None:
        candidate_v3_path, candidate_v3_meta = _emit(
            candidate_v3, "phase3_12_horizon_refined_v3",
            "Refined horizon ensemble + anchor blend; meets +0.002 weighted or +0.003 hep threshold.",
        )
    if candidate_hep_only is not None:
        candidate_hep_only_path, candidate_hep_only_meta = _emit(
            candidate_hep_only, "phase3_12_horizon_hepatic_only",
            "Hepatic-only refinement, death side identical to anchor.",
        )

    # ------------------------------------------------------------------
    # Step G: report
    # ------------------------------------------------------------------

    md: list[str] = []
    md.append("# Phase 3.12 — horizon-label refinement\n")
    md.append("Anchor: `phase3_10_horizon_blend_v2` (public LB **0.91093**).\n")
    md.append(f"Anchor OOF: hepatic={a_h_ci:.4f} / death={a_d_ci:.4f} / weighted={a_w:.4f}.\n")
    md.append(f"Anchor fold: hep std/min={a_h_fold[1]:.4f}/{a_h_fold[2]:.4f}, "
              f"dea std/min={a_d_fold[1]:.4f}/{a_d_fold[2]:.4f}.\n")

    md.append("## A. Horizon label counts\n")
    md.append(pd.DataFrame(horizon_summary)
              .sort_values(["endpoint", "horizon_years"])
              .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## B. New horizon results (sorted by survival C-index per endpoint)\n")
    if not runs_df.empty:
        md.append(runs_df.sort_values(["endpoint", "surv_cindex_mean"], ascending=[True, False])
                  .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## C. Censoring policy comparison (best fs+model per horizon)\n")
    if not runs_df.empty:
        # Per (endpoint, horizon) compare modes.
        rows_pol = []
        for (ep_name, H), grp in runs_df.groupby(["endpoint", "horizon_years"]):
            modes_present = sorted(grp["mode"].unique())
            for mode in modes_present:
                sub = grp[grp["mode"] == mode]
                best = sub.loc[sub["surv_cindex_mean"].idxmax()] if len(sub) else None
                if best is None:
                    continue
                rows_pol.append({
                    "endpoint": ep_name, "horizon_years": H, "mode": mode,
                    "label": best["label"], "surv_cindex_mean": best["surv_cindex_mean"],
                    "surv_cindex_std": best["surv_cindex_std"],
                    "surv_cindex_min": best["surv_cindex_min"],
                    "horizon_auc": best["horizon_auc"],
                    "rho_anchor_oof": best["rho_oof_anchor"],
                })
        md.append(pd.DataFrame(rows_pol)
                  .sort_values(["endpoint", "horizon_years", "mode"])
                  .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## D. Endpoint-specific horizon ensembles\n")
    md.append(f"- **hepatic ensemble** (greedy, equal-weight, |pool|={len(hep_pool)}): "
              f"{hep_sel} -> survival C-index {hep_ens_ci:.4f}")
    md.append(f"- **death ensemble** (greedy, equal-weight, |pool|={len(dea_pool)}): "
              f"{dea_sel} -> survival C-index {dea_ens_ci:.4f}")
    md.append("")

    md.append("## E. Blend grid vs anchor\n")
    blend_table = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("h_test", "d_test", "h_oof", "d_oof",
                                                    "components_h", "components_d",
                                                    "weights_h", "weights_d")}
        for r in blend_rows
    ])
    md.append(blend_table.sort_values("weighted_oof", ascending=False)
              .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## F. Candidate decisions\n")
    if candidate_v3 is None and candidate_hep_only is None:
        md.append(
            "**No submission recommended.** No blend variant beat the anchor by "
            "≥ +0.002 weighted or ≥ +0.003 hepatic OOF. Continue holding "
            "`phase3_10_horizon_blend_v2` as the public best.\n"
        )
    else:
        if candidate_v3 is not None:
            md.append(f"**Candidate v3**: `{candidate_v3_path}`")
            md.append(f"- blend: `{candidate_v3['blend']}` (alpha={candidate_v3['alpha']}, side={candidate_v3['side']})")
            md.append(f"- weighted OOF: {candidate_v3['weighted_oof']:.4f} (Δ {candidate_v3['weighted_oof'] - a_w:+.4f})")
            md.append(f"- hep OOF: {candidate_v3['hep_oof']:.4f} (Δ {candidate_v3['hep_oof'] - a_h_ci:+.4f})")
            md.append(f"- death OOF: {candidate_v3['death_oof']:.4f} (Δ {candidate_v3['death_oof'] - a_d_ci:+.4f})")
            md.append(f"- hep fold std/min: {candidate_v3['hep_fold_std']:.4f}/{candidate_v3['hep_fold_min']:.4f}")
            md.append(f"- dea fold std/min: {candidate_v3['dea_fold_std']:.4f}/{candidate_v3['dea_fold_min']:.4f}")
            md.append(f"- rank-corr w/ anchor (test): hep={candidate_v3['rho_h_test_anchor']:.3f}, dea={candidate_v3['rho_d_test_anchor']:.3f}")
            md.append("")
        if candidate_hep_only is not None:
            md.append(f"**Candidate hepatic-only**: `{candidate_hep_only_path}`")
            md.append(f"- blend: `{candidate_hep_only['blend']}` (alpha={candidate_hep_only['alpha']}, side={candidate_hep_only['side']})")
            md.append(f"- weighted OOF: {candidate_hep_only['weighted_oof']:.4f} (Δ {candidate_hep_only['weighted_oof'] - a_w:+.4f})")
            md.append(f"- hep OOF: {candidate_hep_only['hep_oof']:.4f} (Δ {candidate_hep_only['hep_oof'] - a_h_ci:+.4f})")
            md.append(f"- death OOF: {candidate_hep_only['death_oof']:.4f} (Δ {candidate_hep_only['death_oof'] - a_d_ci:+.4f})")
            md.append("")

    md.append("## G. Recommendation\n")
    if candidate_v3 is None and candidate_hep_only is None:
        md.append(
            "**Do not submit.** The fractional-horizon expansion and censoring-"
            "policy alternatives did not produce a blend that meaningfully beats "
            "`phase3_10_horizon_blend_v2`. Hold the public best and continue "
            "investigating other directions.\n"
        )
    else:
        primary = candidate_v3 if candidate_v3 is not None else candidate_hep_only
        primary_path = candidate_v3_path if candidate_v3 is not None else candidate_hep_only_path
        md.append(f"**Submit one**: `{primary_path}`")
        md.append(f"- blend `{primary['blend']}`")
        md.append(f"- weighted OOF Δ vs anchor: {primary['weighted_oof'] - a_w:+.4f}")
        md.append(f"- hepatic OOF Δ vs anchor: {primary['hep_oof'] - a_h_ci:+.4f}")
        md.append("")
        md.append("### Public-LB outcome interpretation")
        md.append(
            "- LB > 0.91093 by ≥ +0.001 → fractional horizons / refined censoring "
            "policy transferred; iterate further on the same recipe.\n"
            "- LB ≈ 0.91093 (within ±0.001) → the OOF gain didn't transfer; the "
            "private split has different horizon mass; treat horizons as "
            "exhausted and explore other directions.\n"
            "- LB < 0.91093 by ≥ -0.002 → likely overfit on OOF; revert to anchor.\n"
        )

    md.append("## Notes\n")
    md.append("- Horizon labels are constructed from `Endpoint.event` and "
              "`Endpoint.time` only; we never use event/censoring-age columns "
              "as features.")
    md.append("- Censoring-policy modes: `exclude` (drop censored-before-H), "
              "`soft_weight` (label=0, sample_weight = clip(time/H, 0.1, 1.0)), "
              "`downweight` (label=0, sample_weight = 0.25).")
    md.append("- All training is fold-internal; test predictions are averaged "
              "across folds before being persisted as `test.csv`.\n")

    out_md = cfg.REPORTS_DIR / "phase3_12_horizon_refinement.md"
    out_md.write_text("\n".join(md))
    _LOG.info("wrote %s", out_md)


if __name__ == "__main__":
    main()
