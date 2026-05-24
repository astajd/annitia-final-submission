"""Phase 3.9 — horizon-specific risk models.

For each (endpoint, horizon, feature_set, model) we train a binary
classifier on patients usable at that horizon, take the predicted positive-
class probability as a risk score, and evaluate against:

- AUC at the horizon (sanity check, classification metric)
- the *survival* C-index of the original endpoint (competition metric)

We then blend the strongest horizon model per endpoint with the public best
``phase3_5_current_state_v3_hepatic_focused`` (LB 0.89147) at multiple fixed
alphas and apply the Phase 3.9 promotion criteria. One candidate
``submissions/<ts>_phase3_9_horizon_blend.csv`` is emitted only if criteria
are met.

No target-derived feature construction; the horizon enters only through the
binary label and the patient-usability mask.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
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
# Feature sets
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
        _LOG.info("feature %s: train=%s test=%s",
                  name, blob["train"].shape, blob["test"].shape)
    return out


# ---------------------------------------------------------------------------
# Reference (v3) OOF + test
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
# CV runner
# ---------------------------------------------------------------------------

def _train_one_horizon(
    model_name: str,
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    horizon_label: np.ndarray,
    horizon_mask: np.ndarray,
    splits,
    *,
    n_train: int,
    n_test: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (oof, test) for one (model, feature_set, endpoint, horizon)."""
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
        va_idx = np.array(s.valid_idx, dtype=int)  # validate over all rows; mask is for train only
        if len(tr_idx) < 30 or horizon_label[tr_idx].sum() < 3:
            continue
        ytr = horizon_label[tr_idx].astype(int)
        n_pos = max(int(ytr.sum()), 1)
        n_neg = max(int((1 - ytr).sum()), 1)
        spw = n_neg / n_pos

        if model_name == "lgbm_binary":
            import lightgbm as lgb
            clf = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                reg_lambda=1.0, scale_pos_weight=spw, random_state=0, verbose=-1,
            )
        elif model_name == "xgb_binary":
            import xgboost as xgb
            clf = xgb.XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                min_child_weight=1.0, scale_pos_weight=spw,
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", verbosity=0, random_state=0,
            )
        elif model_name == "catboost_binary":
            from catboost import CatBoostClassifier
            clf = CatBoostClassifier(
                iterations=500, depth=5, learning_rate=0.05, l2_leaf_reg=3.0,
                class_weights=[1.0, n_neg / n_pos], random_seed=0,
                verbose=0, allow_writing_files=False,
            )
        elif model_name == "elastic_logistic":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            scaler.fit(Xtr_arr[tr_idx])
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", l1_ratio=0.5,
                C=1.0, max_iter=2000, random_state=0, class_weight="balanced",
            )
            try:
                clf.fit(scaler.transform(Xtr_arr[tr_idx]), ytr)
                va_proba = clf.predict_proba(scaler.transform(Xtr_arr[va_idx]))[:, 1]
                te_proba = clf.predict_proba(scaler.transform(Xte_arr))[:, 1]
                oof[va_idx] = va_proba
                test_sum += te_proba
                test_n += 1
                continue
            except Exception:
                continue
        else:
            raise KeyError(model_name)

        try:
            clf.fit(Xtr_arr[tr_idx], ytr)
            oof[va_idx] = clf.predict_proba(Xtr_arr[va_idx])[:, 1]
            test_sum += clf.predict_proba(Xte_arr)[:, 1]
            test_n += 1
        except Exception as e:  # noqa: BLE001
            _LOG.warning("fold (rep=%d, fold=%d) %s failed: %s", s.repeat, s.fold, model_name, e)

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


def _blend(a, b, alpha):
    ra, rb = _rank(a), _rank(b)
    valid_a, valid_b = np.isfinite(ra), np.isfinite(rb)
    return np.where(valid_a & valid_b, alpha * ra + (1 - alpha) * rb,
                    np.where(valid_a, ra, np.where(valid_b, rb, np.nan)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    cfg.ensure_dirs()
    out_root = cfg.EXPERIMENT_OUTPUTS / "phase3_9_horizon"
    out_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)

    feature_sets = assemble_feature_sets(ds)
    horizons = [1.0, 3.0, 5.0]
    endpoints = {"hepatic": hep, "death": death}
    models = ["lgbm_binary", "xgb_binary", "catboost_binary"]

    runs: list[dict] = []
    horizon_summary: list[dict] = []

    # Pre-compute horizon labels once per (endpoint, horizon)
    label_blobs = {}
    for ep_name, ep in endpoints.items():
        for H in horizons:
            lab = build_horizon_labels(ep, H)
            label_blobs[(ep_name, H)] = lab
            horizon_summary.append({
                "endpoint": ep_name,
                "horizon_years": H,
                "n_total": lab.n_total,
                "n_usable": lab.n_usable,
                "n_positive": lab.n_positive,
                "n_negative": lab.n_usable - lab.n_positive,
                "n_censored_before_h": lab.notes["n_censored_before_horizon"],
                "positive_rate": lab.n_positive / max(lab.n_usable, 1),
            })
            _LOG.info("horizon labels %s/%.0fy: usable=%d pos=%d neg=%d (excluded %d censored<H)",
                      ep_name, H, lab.n_usable, lab.n_positive, lab.n_usable - lab.n_positive,
                      lab.notes["n_censored_before_horizon"])

    # Main sweep.
    for fs_name, blob in feature_sets.items():
        for ep_name, ep in endpoints.items():
            for H in horizons:
                lab = label_blobs[(ep_name, H)]
                if lab.n_positive < 5:
                    _LOG.warning("skip %s/%s/%.0fy: only %d positives", fs_name, ep_name, H, lab.n_positive)
                    continue
                for model_name in models:
                    label = f"{fs_name}__{ep_name}__h{int(H)}__{model_name}"
                    t0 = time.time()
                    try:
                        oof, test = _train_one_horizon(
                            model_name, blob["train"], blob["test"],
                            lab.label, lab.mask, splits,
                            n_train=n_train, n_test=n_test,
                        )
                    except Exception as e:  # noqa: BLE001
                        _LOG.error("%s failed: %s", label, e)
                        continue
                    dt = time.time() - t0

                    # Survival C-index (competition metric).
                    finite = np.isfinite(oof)
                    if finite.sum() < 50 or ep.event[finite].sum() == 0:
                        ci_surv = float("nan")
                    else:
                        ci_surv = float(cindex(ep.event[finite], ep.time[finite], oof[finite]).cindex)

                    # Per-fold survival C-index for stability.
                    fold_scores: list[float] = []
                    for s in splits:
                        v = oof[s.valid_idx]
                        ff = np.isfinite(v)
                        if ff.sum() == 0 or ep.event[s.valid_idx][ff].sum() == 0:
                            continue
                        sc = cindex(ep.event[s.valid_idx][ff], ep.time[s.valid_idx][ff], v[ff]).cindex
                        if np.isfinite(sc):
                            fold_scores.append(float(sc))
                    surv_std = float(np.std(fold_scores)) if fold_scores else float("nan")
                    surv_min = float(np.min(fold_scores)) if fold_scores else float("nan")

                    # Horizon AUC restricted to usable patients.
                    use = lab.mask & np.isfinite(oof)
                    auc = _safe_auc(lab.label[use], oof[use])

                    # Persist OOF/test.
                    run_dir = out_root / label
                    run_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({cfg.PATIENT_ID_COL: ds.train_df[cfg.PATIENT_ID_COL].values, "oof": oof}).to_csv(run_dir / "oof.csv", index=False)
                    pd.DataFrame({cfg.TRUSTII_ID_COL: ds.test_df[cfg.TRUSTII_ID_COL].values, "test": test}).to_csv(run_dir / "test.csv", index=False)

                    runs.append({
                        "label": label,
                        "feature_set": fs_name,
                        "endpoint": ep_name,
                        "horizon_years": H,
                        "model": model_name,
                        "n_features": blob["train"].shape[1],
                        "n_usable_train": int(lab.n_usable),
                        "n_positive_train": int(lab.n_positive),
                        "horizon_auc": auc,
                        "surv_cindex_mean": ci_surv,
                        "surv_cindex_std": surv_std,
                        "surv_cindex_min": surv_min,
                        "wall_seconds": dt,
                        "_oof": oof,
                        "_test": test,
                    })
                    _LOG.info("%s AUC=%.4f survC=%.4f std=%.4f (%.1fs)",
                              label, auc, ci_surv, surv_std, dt)

    # Reference v3 OOF and test predictions for blending and rank-corr.
    v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)
    cands = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.csv"))
    pub = pd.read_csv(cands[-1]) if cands else None
    pub_h_test = pub[cfg.SUB_HEPATIC_COL].to_numpy() if pub is not None else np.array([])
    pub_d_test = pub[cfg.SUB_DEATH_COL].to_numpy() if pub is not None else np.array([])

    runs_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in runs])

    # Add rank correlations vs v3 to runs_df
    for r in runs:
        ep = r["endpoint"]
        v3_oof = v3_h_oof if ep == "hepatic" else v3_d_oof
        v3_test = pub_h_test if ep == "hepatic" else pub_d_test
        r["rho_oof_v3"] = float(pd.Series(r["_oof"]).rank(pct=True).corr(pd.Series(v3_oof).rank(pct=True), method="spearman")) if v3_oof.size else float("nan")
        r["rho_test_v3"] = float(pd.Series(r["_test"]).rank(pct=True).corr(pd.Series(v3_test).rank(pct=True), method="spearman")) if v3_test.size else float("nan")

    runs_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in runs])

    # Best horizon model per endpoint by survival C-index
    hep_runs = [r for r in runs if r["endpoint"] == "hepatic" and np.isfinite(r["surv_cindex_mean"])]
    dea_runs = [r for r in runs if r["endpoint"] == "death" and np.isfinite(r["surv_cindex_mean"])]
    hep_best = max(hep_runs, key=lambda r: r["surv_cindex_mean"]) if hep_runs else None
    dea_best = max(dea_runs, key=lambda r: r["surv_cindex_mean"]) if dea_runs else None

    # Blends with v3 — try alphas {0.95, 0.9, 0.8, 0.7} for hep_only / dea_only / both
    blend_rows = []
    if hep_best and dea_best and v3_h_oof.size and v3_d_oof.size:
        for alpha in (0.95, 0.9, 0.85, 0.8, 0.7):
            for tag, h_other, d_other in [
                ("hep_only", hep_best["_oof"], None),
                ("dea_only", None, dea_best["_oof"]),
                ("both",     hep_best["_oof"], dea_best["_oof"]),
            ]:
                h_blend_oof = _blend(v3_h_oof, h_other, alpha) if h_other is not None else v3_h_oof
                d_blend_oof = _blend(v3_d_oof, d_other, alpha) if d_other is not None else v3_d_oof
                h_blend_test = _blend(pub_h_test, hep_best["_test"], alpha) if h_other is not None else pub_h_test
                d_blend_test = _blend(pub_d_test, dea_best["_test"], alpha) if d_other is not None else pub_d_test
                fh, fd = np.isfinite(h_blend_oof), np.isfinite(d_blend_oof)
                ci_h = float(cindex(hep.event[fh], hep.time[fh], h_blend_oof[fh]).cindex) if fh.sum() else float("nan")
                ci_d = float(cindex(death.event[fd], death.time[fd], d_blend_oof[fd]).cindex) if fd.sum() else float("nan")
                # fold stability of the blend
                fold_h = []
                fold_d = []
                for s in splits:
                    v = h_blend_oof[s.valid_idx]
                    ff = np.isfinite(v)
                    if ff.sum() and hep.event[s.valid_idx][ff].sum():
                        fold_h.append(cindex(hep.event[s.valid_idx][ff], hep.time[s.valid_idx][ff], v[ff]).cindex)
                    v = d_blend_oof[s.valid_idx]
                    ff = np.isfinite(v)
                    if ff.sum() and death.event[s.valid_idx][ff].sum():
                        fold_d.append(cindex(death.event[s.valid_idx][ff], death.time[s.valid_idx][ff], v[ff]).cindex)
                blend_rows.append({
                    "blend": f"alpha={alpha}_{tag}",
                    "alpha": alpha,
                    "side": tag,
                    "hep_oof": ci_h,
                    "death_oof": ci_d,
                    "weighted_oof": weighted_score(ci_h, ci_d),
                    "hep_fold_std": float(np.std(fold_h)) if fold_h else float("nan"),
                    "hep_fold_min": float(np.min(fold_h)) if fold_h else float("nan"),
                    "dea_fold_std": float(np.std(fold_d)) if fold_d else float("nan"),
                    "dea_fold_min": float(np.min(fold_d)) if fold_d else float("nan"),
                    "h_test": h_blend_test,
                    "d_test": d_blend_test,
                })

    # v3 reference numbers
    v3_meta = json.loads(cands[-1].with_suffix(".json").read_text()) if cands and cands[-1].with_suffix(".json").exists() else {}
    v3_h = v3_meta.get("hepatic", {}).get("oof_cindex", float("nan"))
    v3_d = v3_meta.get("death", {}).get("oof_cindex", float("nan"))
    v3_w = v3_meta.get("weighted_score_cv", float("nan"))
    v3_h_fold_std = v3_meta.get("hepatic", {}).get("fold_std", float("nan"))
    v3_d_fold_std = v3_meta.get("death", {}).get("fold_std", float("nan"))

    # Promotion check
    promoted = None
    reason = ""
    for row in blend_rows:
        if row["alpha"] == 1.0:
            continue
        improves_w = np.isfinite(row["weighted_oof"]) and (row["weighted_oof"] - v3_w >= 0.002)
        improves_h = np.isfinite(row["hep_oof"]) and (row["hep_oof"] - v3_h >= 0.003)
        # Diversity criterion: rank-corr of best contributing horizon vs v3 below 0.95
        # AND blend improves hep or weighted by any positive amount.
        contributing_oof = hep_best["_oof"] if row["side"] in ("hep_only", "both") else dea_best["_oof"]
        contributing_endpoint = "hepatic" if row["side"] in ("hep_only", "both") else "death"
        v3_oof_for_corr = v3_h_oof if contributing_endpoint == "hepatic" else v3_d_oof
        rho = float(pd.Series(contributing_oof).rank(pct=True).corr(pd.Series(v3_oof_for_corr).rank(pct=True), method="spearman"))
        improves_stability = (row["hep_oof"] - v3_h > 0) and (row["hep_fold_std"] < v3_h_fold_std - 0.005) if np.isfinite(v3_h_fold_std) else False
        diverse_and_helps = (rho < 0.95) and ((row["weighted_oof"] - v3_w > 0.001) or improves_stability)

        if improves_w or improves_h or diverse_and_helps:
            if promoted is None or row["weighted_oof"] > promoted["weighted_oof"]:
                promoted = row
                reason = (
                    f"weighted Δ={row['weighted_oof']-v3_w:+.4f}, "
                    f"hepatic Δ={row['hep_oof']-v3_h:+.4f}, "
                    f"contributing rho={rho:.3f}, "
                    f"hep_fold_std Δ={row['hep_fold_std']-v3_h_fold_std:+.4f}"
                )

    sub_path = None
    if promoted is not None:
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=promoted["h_test"], risk_death=promoted["d_test"],
            sample_submission=ds.sample_submission, model_name="phase3_9_horizon_blend",
        )
        meta = {
            "label": "phase3_9_horizon_blend",
            "blend": promoted["blend"],
            "alpha": promoted["alpha"],
            "side": promoted["side"],
            "hep_best_run": hep_best["label"] if hep_best else None,
            "dea_best_run": dea_best["label"] if dea_best else None,
            "hepatic_oof": promoted["hep_oof"],
            "death_oof": promoted["death_oof"],
            "weighted_oof": promoted["weighted_oof"],
            "uses_target_derived_features": False,
            "recommended": True,
            "promotion_reason": reason,
            "submission_csv": str(sub_path),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        _LOG.info("PROMOTED %s -> %s", promoted["blend"], sub_path)

    # Report
    md: list[str] = []
    md.append("# Phase 3.9 — horizon-specific risk models\n")
    md.append(
        f"Reference: `phase3_5_current_state_v3_hepatic_focused` LB **0.89147**, "
        f"OOF hepatic={v3_h:.4f} / death={v3_d:.4f} / weighted={v3_w:.4f}.\n"
    )
    md.append("## Horizon label counts\n")
    md.append(pd.DataFrame(horizon_summary).to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## Per-run results (sorted by survival C-index per endpoint)\n")
    if not runs_df.empty:
        md.append(runs_df.sort_values(["endpoint", "surv_cindex_mean"], ascending=[True, False])
                  .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## Best horizon model per endpoint\n")
    if hep_best:
        md.append(f"- **hepatic**: `{hep_best['label']}` survC={hep_best['surv_cindex_mean']:.4f} "
                  f"std={hep_best['surv_cindex_std']:.4f} AUC@h={hep_best['horizon_auc']:.4f} "
                  f"(rank-corr w/ v3 OOF {hep_best['rho_oof_v3']:.3f}, test {hep_best['rho_test_v3']:.3f})")
    if dea_best:
        md.append(f"- **death**: `{dea_best['label']}` survC={dea_best['surv_cindex_mean']:.4f} "
                  f"std={dea_best['surv_cindex_std']:.4f} AUC@h={dea_best['horizon_auc']:.4f} "
                  f"(rank-corr w/ v3 OOF {dea_best['rho_oof_v3']:.3f}, test {dea_best['rho_test_v3']:.3f})")
    md.append("")

    md.append("## Blends with v3 (rank-space)\n")
    if blend_rows:
        b_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("h_test", "d_test")} for r in blend_rows])
        md.append(b_df.to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## Promotion decision\n")
    if promoted:
        md.append(f"**Promoted**: `{promoted['blend']}` ({reason}). Submission: `{sub_path}`.")
    else:
        md.append(
            "**Not promoted.** No horizon blend met the criteria:\n"
            "- weighted OOF improves over v3 by ≥ 0.002\n"
            "- hepatic OOF improves over v3 by ≥ 0.003\n"
            "- contributing horizon model has rank-corr < 0.95 with v3 *and* improves the OOF blend or fold stability\n"
        )
    md.append("")

    md.append("## Notes\n")
    md.append(
        "- Horizon labels exclude patients censored before the horizon (default mode). "
        "We did not need the down-weight variant because the exclusion still leaves "
        "most patients usable at H = 1y / 3y / 5y for both endpoints.\n"
        "- Risk score = horizon classifier's predicted positive-class probability; we "
        "evaluate it against the survival C-index of the original endpoint, which is "
        "what the contest scores.\n"
        "- All training is fold-internal (preprocessing fit only on the training fold); "
        "no event/censoring-age columns are used to build features, only to define labels.\n"
    )

    out_md = cfg.REPORTS_DIR / "phase3_9_horizon_models.md"
    out_md.write_text("\n".join(md))
    _LOG.info("wrote %s", out_md)


if __name__ == "__main__":
    main()
