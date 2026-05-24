"""Phase 3.16 — final component-pool replacement test.

Anchor: ``phase3_10_horizon_blend_v2`` (LB **0.91093**).

Phase 3.15 found that LightGBM LambdaRanker on `v3_hepatic_schema` with K=50
controls reaches OOF survival C-index 0.7831 — about +0.009 better than the
comparable h3 LGBM binary horizon classifier. A simple post-hoc blend with
the anchor did not transfer that gain. This phase asks the harder question:
does the LambdaRanker help when *inserted into the same component pool*
that built `phase3_10_horizon_blend_v2`, and selected by the same greedy
capped-at-25% logic?

Plan:

1. Retrain the two best Phase 3.15 hepatic LambdaRankers (seeds 0, 2) to
   regenerate (OOF, test) arrays — the Phase 3.15 driver did not persist
   per-run artifacts.
2. Load the full phase3_9 + phase3_10 horizon-classifier pool.
3. Add the LambdaRanker outputs to the hepatic pool only.
4. Re-run the same greedy_capped(0.25) + α blend grid as Phase 3.10.
5. Apply Phase 3.10's promotion criteria (Δweighted ≥ +0.002 or Δhep ≥ +0.003).
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
from .metrics import cindex, weighted_score
from .models import build_model
from .models._preprocess import fill_for_tree
from .models.ensemble import rank_average, to_rank
from .submission import make_submission
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger
from .validation import build_folds

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers (consistent with Phase 3.10/3.12)
# ---------------------------------------------------------------------------

def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True, na_option="keep").to_numpy()


def _blend_two(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    ra, rb = _rank(a), _rank(b)
    valid_a, valid_b = np.isfinite(ra), np.isfinite(rb)
    return np.where(valid_a & valid_b, alpha * ra + (1 - alpha) * rb,
                    np.where(valid_a, ra, np.where(valid_b, rb, np.nan)))


def _surv_ci(event, time, score) -> float:
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


def _spearman(a, b) -> float:
    if a is None or b is None or a.size != b.size:
        return float("nan")
    df = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(df) < 5:
        return float("nan")
    return float(df["a"].rank(pct=True).corr(df["b"].rank(pct=True), method="spearman"))


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
# Anchor reconstruction (matches Phase 3.10/3.12 behaviour)
# ---------------------------------------------------------------------------

def _v3_oof(n_train: int, ds, hep) -> tuple[np.ndarray, np.ndarray]:
    cands = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.json"))
    blob = json.loads(cands[-1].read_text())
    h_w = blob["hepatic"]["weights"]
    d_w = blob["death"]["weights"]
    from .endpoint_ensemble import collect_predictions
    all_dirs = [d for d in cfg.EXPERIMENT_OUTPUTS.iterdir()
                if d.is_dir() and (d / "oof_predictions.csv").exists()]
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
    h_oof = rank_average({c: pool[c] for c in h_w if c in pool}, weights={c: h_w[c] for c in h_w if c in pool})
    d_oof = rank_average({c: pool[c] for c in d_w if c in pool}, weights={c: d_w[c] for c in d_w if c in pool})
    return h_oof, d_oof


def _load_horizon_artifact(label: str) -> tuple[np.ndarray, np.ndarray] | None:
    for root in (cfg.EXPERIMENT_OUTPUTS / "phase3_12_horizon",
                 cfg.EXPERIMENT_OUTPUTS / "phase3_10_horizon",
                 cfg.EXPERIMENT_OUTPUTS / "phase3_9_horizon",
                 cfg.EXPERIMENT_OUTPUTS / "phase3_16_pool"):
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


def _load_anchor(ds, hep, death, n_train, n_test):
    p10_meta_path = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_10_horizon_blend_v2.json"))[-1]
    meta = json.loads(p10_meta_path.read_text())
    blend_csv = p10_meta_path.with_suffix(".csv")
    pub = pd.read_csv(blend_csv)
    h_test = pub[cfg.SUB_HEPATIC_COL].to_numpy()
    d_test = pub[cfg.SUB_DEATH_COL].to_numpy()
    v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)
    weights_h = meta["weights_hepatic"]
    weights_d = meta["weights_death"]
    pool_h = {"_v3": v3_h_oof}
    pool_d = {"_v3": v3_d_oof}
    for k in weights_h:
        if k == "_v3":
            continue
        out = _load_horizon_artifact(k)
        if out is not None:
            pool_h[k] = out[0]
    for k in weights_d:
        if k == "_v3":
            continue
        out = _load_horizon_artifact(k)
        if out is not None:
            pool_d[k] = out[0]
    h_oof = rank_average(pool_h, weights={k: weights_h[k] for k in pool_h})
    d_oof = rank_average(pool_d, weights={k: weights_d[k] for k in pool_d})
    return h_oof, h_test, d_oof, d_test, meta


# ---------------------------------------------------------------------------
# LambdaRanker training (same recipe as Phase 3.15)
# ---------------------------------------------------------------------------

def _build_groups(X_arr, event, time, train_idx, *, n_controls=50, seed=0):
    rng = np.random.default_rng(seed)
    rows_X: list[np.ndarray] = []
    rows_y: list[int] = []
    group_sizes: list[int] = []
    for i in train_idx:
        if not event[i]:
            continue
        ti = time[i]
        controls = [j for j in train_idx if time[j] > ti and j != i]
        if len(controls) < 2:
            continue
        if len(controls) > n_controls:
            controls = rng.choice(controls, size=n_controls, replace=False).tolist()
        rows_X.append(X_arr[i])
        rows_y.append(1)
        for c in controls:
            rows_X.append(X_arr[c])
            rows_y.append(0)
        group_sizes.append(len(controls) + 1)
    if not group_sizes:
        return None
    return (np.asarray(rows_X), np.asarray(rows_y, dtype=int),
            np.asarray(group_sizes, dtype=int))


def _train_lgbm_ranker_cv(fs_blob: dict, event, time, splits, *,
                            n_controls: int = 50, seed: int = 0,
                            n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    import lightgbm as lgb

    Xtr_p = fill_for_tree(fs_blob["train"]).astype(np.float32, errors="ignore")
    Xte_p = fill_for_tree(fs_blob["test"]).reindex(columns=Xtr_p.columns, fill_value=0).astype(np.float32, errors="ignore")
    Xtr_arr = Xtr_p.to_numpy()
    Xte_arr = Xte_p.to_numpy()

    oof = np.full(n_train, np.nan)
    test_sum = np.zeros(n_test)
    test_n = 0
    for s in splits:
        tr_idx = np.asarray(s.train_idx, dtype=int)
        va_idx = np.asarray(s.valid_idx, dtype=int)
        grp = _build_groups(Xtr_arr, event, time, tr_idx, n_controls=n_controls, seed=seed)
        if grp is None or grp[0].shape[0] == 0:
            continue
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, reg_lambda=1.0,
            random_state=seed, verbose=-1,
        )
        try:
            ranker.fit(grp[0], grp[1], group=grp[2])
        except Exception as e:  # noqa: BLE001
            _LOG.warning("ranker fit failed seed=%d: %s", seed, e)
            continue
        oof[va_idx] = ranker.predict(Xtr_arr[va_idx])
        test_sum += ranker.predict(Xte_arr)
        test_n += 1
    test = test_sum / test_n if test_n else np.full(n_test, np.nan)
    return oof, test


# ---------------------------------------------------------------------------
# Greedy capped + ensemble (matches Phase 3.10/3.12)
# ---------------------------------------------------------------------------

def _greedy_horizon_ensemble(candidates, event, time, *, min_individual_ci=0.55, max_size=8):
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
        for k in pool:
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
    p16_root = cfg.EXPERIMENT_OUTPUTS / "phase3_16_pool"
    p16_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)
    feature_sets = assemble_feature_sets(ds)

    # Anchor.
    a_h_oof, a_h_test, a_d_oof, a_d_test, anchor_meta = _load_anchor(ds, hep, death, n_train, n_test)
    a_h_ci = _surv_ci(hep.event, hep.time, a_h_oof)
    a_d_ci = _surv_ci(death.event, death.time, a_d_oof)
    a_w = weighted_score(a_h_ci, a_d_ci)
    a_h_fold = _fold_ci(hep.event, hep.time, a_h_oof, splits)
    a_d_fold = _fold_ci(death.event, death.time, a_d_oof, splits)
    _LOG.info("Anchor: hep=%.4f dea=%.4f weighted=%.4f", a_h_ci, a_d_ci, a_w)

    # 1. Reconstruct phase3_10 component pool — load all phase3_9, phase3_10,
    #    phase3_12 horizon outputs.
    p9_artifacts = _load_phase_artifacts(p9_root, ds, n_train, n_test)
    p10_artifacts = _load_phase_artifacts(p10_root, ds, n_train, n_test)
    p12_artifacts = _load_phase_artifacts(p12_root, ds, n_train, n_test)
    all_artifacts: dict[str, dict[str, np.ndarray]] = {}
    all_artifacts.update(p9_artifacts)
    all_artifacts.update(p10_artifacts)
    all_artifacts.update(p12_artifacts)
    _LOG.info("Loaded %d horizon artifacts (p9=%d, p10=%d, p12=%d)",
              len(all_artifacts), len(p9_artifacts), len(p10_artifacts), len(p12_artifacts))

    # 2. Train LambdaRanker(s) and persist OOF/test.
    ranker_specs = [
        ("hep__lambdaranker__v3_hepatic_schema__K50__s0", "v3_hepatic_schema", 50, 0),
        ("hep__lambdaranker__v3_hepatic_schema__K50__s2", "v3_hepatic_schema", 50, 2),
    ]
    ranker_artifacts: dict[str, dict[str, np.ndarray]] = {}
    for label, fs_name, K, seed in ranker_specs:
        # Cache: only retrain if not on disk.
        run_dir = p16_root / label
        if (run_dir / "oof.csv").exists() and (run_dir / "test.csv").exists():
            ranker_artifacts[label] = {
                "oof": pd.read_csv(run_dir / "oof.csv")["oof"].to_numpy(),
                "test": pd.read_csv(run_dir / "test.csv")["test"].to_numpy(),
            }
            _LOG.info("loaded cached %s", label)
            continue
        t0 = _time.time()
        oof, test = _train_lgbm_ranker_cv(
            feature_sets[fs_name], hep.event, hep.time, splits,
            n_controls=K, seed=seed, n_train=n_train, n_test=n_test,
        )
        dt = _time.time() - t0
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({cfg.PATIENT_ID_COL: ds.train_df[cfg.PATIENT_ID_COL].values, "oof": oof}).to_csv(run_dir / "oof.csv", index=False)
        pd.DataFrame({cfg.TRUSTII_ID_COL: ds.test_df[cfg.TRUSTII_ID_COL].values, "test": test}).to_csv(run_dir / "test.csv", index=False)
        ranker_artifacts[label] = {"oof": oof, "test": test}
        ci = _surv_ci(hep.event, hep.time, oof)
        rho_anchor = _spearman(oof, a_h_oof)
        _LOG.info("retrained %s ci=%.4f rho_anchor=%.3f (%.1fs)", label, ci, rho_anchor, dt)

    # Verify the seed=0 ranker matches Phase 3.15's reported 0.7831 ± noise.
    s0_ci = _surv_ci(hep.event, hep.time, ranker_artifacts["hep__lambdaranker__v3_hepatic_schema__K50__s0"]["oof"])
    s2_ci = _surv_ci(hep.event, hep.time, ranker_artifacts["hep__lambdaranker__v3_hepatic_schema__K50__s2"]["oof"])
    rho_s0_s2 = _spearman(
        ranker_artifacts["hep__lambdaranker__v3_hepatic_schema__K50__s0"]["oof"],
        ranker_artifacts["hep__lambdaranker__v3_hepatic_schema__K50__s2"]["oof"],
    )
    _LOG.info("LambdaRanker seed0 ci=%.4f, seed2 ci=%.4f, rho(s0,s2)=%.3f", s0_ci, s2_ci, rho_s0_s2)

    # 3. Add LambdaRankers to the candidate pool.
    all_artifacts.update(ranker_artifacts)
    # We also rank-average the two seeds and add the bag as a third candidate.
    ranker_bag_oof = rank_average({k: ranker_artifacts[k]["oof"] for k in ranker_artifacts})
    ranker_bag_test = rank_average({k: ranker_artifacts[k]["test"] for k in ranker_artifacts})
    bag_label = "hep__lambdaranker__v3_hepatic_schema__K50__bag_s02"
    all_artifacts[bag_label] = {"oof": ranker_bag_oof, "test": ranker_bag_test}
    bag_ci = _surv_ci(hep.event, hep.time, ranker_bag_oof)
    rho_bag_anchor = _spearman(ranker_bag_oof, a_h_oof)
    _LOG.info("LambdaRanker bag ci=%.4f rho_anchor=%.3f", bag_ci, rho_bag_anchor)

    # 4. Re-run greedy_capped(0.25) and the same alpha grid as Phase 3.10/3.12.
    def _pool_for(ep_name: str) -> dict[str, np.ndarray]:
        ep = hep if ep_name == "hepatic" else death
        out = {}
        for k, v in all_artifacts.items():
            parts = k.split("__")
            if len(parts) < 4:
                continue
            ep_tok = parts[1] if not parts[0].startswith("hep") and not parts[0].startswith("dea") else None
            # Schema has two flavours: phase3_10/12 uses "fs__ep__hH__model" while
            # the LambdaRanker uses "hep__lambdaranker__fs__K..__sN".
            if parts[0] in ("hep", "dea"):
                ep_tok = "hepatic" if parts[0] == "hep" else "death"
            else:
                ep_tok = parts[1]
            if ep_tok != ep_name:
                continue
            ci = _surv_ci(ep.event, ep.time, v["oof"])
            if not np.isfinite(ci) or ci < 0.55:
                continue
            out[k] = v["oof"]
        return out

    hep_pool = _pool_for("hepatic")
    dea_pool = _pool_for("death")
    _LOG.info("Hepatic pool size after adding LambdaRankers: %d", len(hep_pool))
    _LOG.info("Death pool size: %d", len(dea_pool))

    # Endpoint-specific greedy ensembles (for the alpha-grid path).
    hep_sel, hep_w, hep_ens_ci = _greedy_horizon_ensemble(hep_pool, hep.event, hep.time)
    dea_sel, dea_w, dea_ens_ci = _greedy_horizon_ensemble(dea_pool, death.event, death.time)
    _LOG.info("hep ensemble: %s -> %.4f", hep_sel, hep_ens_ci)
    _LOG.info("dea ensemble: %s -> %.4f", dea_sel, dea_ens_ci)
    hep_ens_oof = rank_average({k: all_artifacts[k]["oof"] for k in hep_sel}, weights=hep_w) if hep_sel else np.full(n_train, np.nan)
    hep_ens_test = rank_average({k: all_artifacts[k]["test"] for k in hep_sel}, weights=hep_w) if hep_sel else np.full(n_test, np.nan)
    dea_ens_oof = rank_average({k: all_artifacts[k]["oof"] for k in dea_sel}, weights=dea_w) if dea_sel else np.full(n_train, np.nan)
    dea_ens_test = rank_average({k: all_artifacts[k]["test"] for k in dea_sel}, weights=dea_w) if dea_sel else np.full(n_test, np.nan)

    # Best single horizon per endpoint.
    hep_pool_with_test = {k: {"oof": all_artifacts[k]["oof"], "test": all_artifacts[k]["test"]}
                            for k in hep_pool}
    dea_pool_with_test = {k: {"oof": all_artifacts[k]["oof"], "test": all_artifacts[k]["test"]}
                            for k in dea_pool}
    hep_best_label = max(hep_pool, key=lambda k: _surv_ci(hep.event, hep.time, hep_pool[k]))
    dea_best_label = max(dea_pool, key=lambda k: _surv_ci(death.event, death.time, dea_pool[k]))
    _LOG.info("hep best single: %s -> %.4f", hep_best_label,
              _surv_ci(hep.event, hep.time, hep_pool[hep_best_label]))
    _LOG.info("dea best single: %s -> %.4f", dea_best_label,
              _surv_ci(death.event, death.time, dea_pool[dea_best_label]))
    hep_best_oof = all_artifacts[hep_best_label]["oof"]
    hep_best_test = all_artifacts[hep_best_label]["test"]
    dea_best_oof = all_artifacts[dea_best_label]["oof"]
    dea_best_test = all_artifacts[dea_best_label]["test"]

    # v3 OOF/test for greedy_capped seeding.
    v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)
    v3_csv = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.csv"))[-1]
    v3_pub = pd.read_csv(v3_csv)
    v3_h_test = v3_pub[cfg.SUB_HEPATIC_COL].to_numpy()
    v3_d_test = v3_pub[cfg.SUB_DEATH_COL].to_numpy()

    blend_rows: list[dict] = []

    def _eval_blend(name, h_oof, d_oof, h_test, d_test, comp_h, comp_d, w_h, w_d, alpha, side, source):
        ci_h = _surv_ci(hep.event, hep.time, h_oof)
        ci_d = _surv_ci(death.event, death.time, d_oof)
        h_mean, h_std, h_min = _fold_ci(hep.event, hep.time, h_oof, splits)
        d_mean, d_std, d_min = _fold_ci(death.event, death.time, d_oof, splits)
        rho_h_test = _spearman(h_test, a_h_test)
        rho_d_test = _spearman(d_test, a_d_test)
        rho_h_oof = _spearman(h_oof, a_h_oof)
        rho_d_oof = _spearman(d_oof, a_d_oof)
        blend_rows.append({
            "blend": name, "alpha": alpha, "side": side, "source": source,
            "hep_oof": ci_h, "death_oof": ci_d,
            "weighted_oof": weighted_score(ci_h, ci_d),
            "hep_fold_std": h_std, "hep_fold_min": h_min,
            "dea_fold_std": d_std, "dea_fold_min": d_min,
            "rho_h_test_anchor": rho_h_test, "rho_d_test_anchor": rho_d_test,
            "rho_h_oof_anchor": rho_h_oof, "rho_d_oof_anchor": rho_d_oof,
            "components_h": comp_h, "components_d": comp_d,
            "weights_h": w_h, "weights_d": w_d,
            "h_test": h_test, "d_test": d_test,
            "h_oof": h_oof, "d_oof": d_oof,
        })

    # Anchor reference.
    _eval_blend("anchor_phase3_10_v2", a_h_oof, a_d_oof, a_h_test, a_d_test,
                [], [], {"anchor": 1.0}, {"anchor": 1.0}, alpha=1.0, side="none", source="anchor")

    # Alpha-grid (single best + ensemble) blends.
    for alpha in (0.95, 0.90, 0.85, 0.80, 0.75):
        for side in ("hep_only", "dea_only", "both"):
            # Single-best
            if side in ("hep_only", "both"):
                h_oof_s = _blend_two(a_h_oof, hep_best_oof, alpha)
                h_test_s = _blend_two(a_h_test, hep_best_test, alpha)
                comp_h_s = [hep_best_label]
                w_h_s = {"anchor": alpha, hep_best_label: 1 - alpha}
            else:
                h_oof_s = a_h_oof.copy(); h_test_s = a_h_test.copy()
                comp_h_s = []; w_h_s = {"anchor": 1.0}
            if side in ("dea_only", "both"):
                d_oof_s = _blend_two(a_d_oof, dea_best_oof, alpha)
                d_test_s = _blend_two(a_d_test, dea_best_test, alpha)
                comp_d_s = [dea_best_label]
                w_d_s = {"anchor": alpha, dea_best_label: 1 - alpha}
            else:
                d_oof_s = a_d_oof.copy(); d_test_s = a_d_test.copy()
                comp_d_s = []; w_d_s = {"anchor": 1.0}
            _eval_blend(f"single_alpha={alpha}_{side}", h_oof_s, d_oof_s,
                        h_test_s, d_test_s, comp_h_s, comp_d_s, w_h_s, w_d_s,
                        alpha, side, "best_single")
            # Ensemble
            if hep_sel and dea_sel:
                if side in ("hep_only", "both"):
                    h_oof_e = _blend_two(a_h_oof, hep_ens_oof, alpha)
                    h_test_e = _blend_two(a_h_test, hep_ens_test, alpha)
                    comp_h_e = list(hep_sel)
                    w_h_e = {"anchor_h": alpha, **{k: (1 - alpha) * w for k, w in hep_w.items()}}
                else:
                    h_oof_e = a_h_oof.copy(); h_test_e = a_h_test.copy()
                    comp_h_e = []; w_h_e = {"anchor_h": 1.0}
                if side in ("dea_only", "both"):
                    d_oof_e = _blend_two(a_d_oof, dea_ens_oof, alpha)
                    d_test_e = _blend_two(a_d_test, dea_ens_test, alpha)
                    comp_d_e = list(dea_sel)
                    w_d_e = {"anchor_d": alpha, **{k: (1 - alpha) * w for k, w in dea_w.items()}}
                else:
                    d_oof_e = a_d_oof.copy(); d_test_e = a_d_test.copy()
                    comp_d_e = []; w_d_e = {"anchor_d": 1.0}
                _eval_blend(f"ensemble_alpha={alpha}_{side}", h_oof_e, d_oof_e,
                            h_test_e, d_test_e, comp_h_e, comp_d_e, w_h_e, w_d_e,
                            alpha, side, "ensemble")

    # Greedy capped at 25%, seeded by v3 — both endpoints, plus hep-only / dea-only flavours.
    g_h_w, g_h_oof, g_h_test, g_h_ci = _greedy_capped(
        v3_h_oof, v3_h_test, hep_pool_with_test, hep.event, hep.time, cap=0.25)
    g_d_w, g_d_oof, g_d_test, g_d_ci = _greedy_capped(
        v3_d_oof, v3_d_test, dea_pool_with_test, death.event, death.time, cap=0.25)
    comp_h_g = [k for k in g_h_w if k != "_v3"]
    comp_d_g = [k for k in g_d_w if k != "_v3"]
    _eval_blend("greedy_cap25_both", g_h_oof, g_d_oof, g_h_test, g_d_test,
                comp_h_g, comp_d_g, g_h_w, g_d_w, alpha=None, side="both", source="greedy_capped")
    _eval_blend("greedy_cap25_hep_only", g_h_oof, a_d_oof, g_h_test, a_d_test,
                comp_h_g, [], g_h_w, {"anchor_d": 1.0}, alpha=None, side="hep_only", source="greedy_capped")
    _eval_blend("greedy_cap25_dea_only", a_h_oof, g_d_oof, a_h_test, g_d_test,
                [], comp_d_g, {"anchor_h": 1.0}, g_d_w, alpha=None, side="dea_only", source="greedy_capped")

    blend_df = pd.DataFrame([{k: v for k, v in r.items()
                                if k not in ("h_oof", "d_oof", "h_test", "d_test",
                                              "components_h", "components_d",
                                              "weights_h", "weights_d")}
                              for r in blend_rows])
    blend_df.to_csv(p16_root / "blends.csv", index=False)

    # 5. Promotion criteria.
    candidate = None
    for row in sorted(blend_rows, key=lambda r: -r["weighted_oof"]):
        if row["blend"] == "anchor_phase3_10_v2":
            continue
        d_w = row["weighted_oof"] - a_w
        d_h = row["hep_oof"] - a_h_ci
        if d_w >= 0.002 or d_h >= 0.003:
            candidate = row
            break

    sub_path = None
    cand_meta = None
    if candidate is not None:
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=candidate["h_test"], risk_death=candidate["d_test"],
            sample_submission=ds.sample_submission,
            model_name="phase3_16_pool_replaced",
        )
        cand_meta = {
            "label": "phase3_16_pool_replaced",
            "blend_id": candidate["blend"],
            "alpha": candidate["alpha"], "side": candidate["side"],
            "components_hepatic": candidate["components_h"],
            "components_death":   candidate["components_d"],
            "weights_hepatic": candidate["weights_h"],
            "weights_death":   candidate["weights_d"],
            "hepatic_oof": candidate["hep_oof"],
            "death_oof":   candidate["death_oof"],
            "weighted_oof": candidate["weighted_oof"],
            "delta_vs_anchor": {
                "weighted_oof": candidate["weighted_oof"] - a_w,
                "hepatic_oof":  candidate["hep_oof"] - a_h_ci,
                "death_oof":    candidate["death_oof"] - a_d_ci,
            },
            "hep_fold_std": candidate["hep_fold_std"], "hep_fold_min": candidate["hep_fold_min"],
            "dea_fold_std": candidate["dea_fold_std"], "dea_fold_min": candidate["dea_fold_min"],
            "rho_test_anchor": {"hepatic": candidate["rho_h_test_anchor"],
                                  "death":   candidate["rho_d_test_anchor"]},
            "rho_oof_anchor":  {"hepatic": candidate["rho_h_oof_anchor"],
                                  "death":   candidate["rho_d_oof_anchor"]},
            "uses_target_derived_features": False,
            "rationale": "Pool-replacement: LambdaRanker(s) added to phase3_10's hepatic component pool, "
                         "selected by greedy_capped/ensemble logic.",
            "submission_csv": str(sub_path),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(cand_meta, indent=2, default=str))

    # 6. Report.
    md: list[str] = []
    md.append("# Phase 3.16 — pool-replacement test (LambdaRanker into phase3_10 pool)\n")
    md.append("Anchor: `phase3_10_horizon_blend_v2` (LB **0.91093**, weighted OOF "
              f"{a_w:.4f}, hep {a_h_ci:.4f}, dea {a_d_ci:.4f}).\n")
    md.append(f"Anchor fold: hep std/min={a_h_fold[1]:.4f}/{a_h_fold[2]:.4f}, "
              f"dea std/min={a_d_fold[1]:.4f}/{a_d_fold[2]:.4f}.\n")

    md.append("## A. LambdaRanker components added to the hepatic pool\n")
    rows_lr = []
    for label in ("hep__lambdaranker__v3_hepatic_schema__K50__s0",
                   "hep__lambdaranker__v3_hepatic_schema__K50__s2",
                   bag_label):
        if label not in all_artifacts:
            continue
        ci = _surv_ci(hep.event, hep.time, all_artifacts[label]["oof"])
        rho_h_oof = _spearman(all_artifacts[label]["oof"], a_h_oof)
        rho_h_test = _spearman(all_artifacts[label]["test"], a_h_test)
        rows_lr.append({
            "component": label, "ci_oof": ci,
            "rho_h_oof_anchor": rho_h_oof, "rho_h_test_anchor": rho_h_test,
        })
    md.append(pd.DataFrame(rows_lr).to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## B. Greedy capped at 25% horizon contribution (re-run with extended pool)\n")
    md.append(f"- hepatic pool size = {len(hep_pool)}, death pool size = {len(dea_pool)}")
    md.append(f"- hepatic chosen weights: {g_h_w}")
    md.append(f"- hepatic chosen survival C-index: {g_h_ci:.4f}")
    md.append(f"- death chosen weights: {g_d_w}")
    md.append(f"- death chosen survival C-index: {g_d_ci:.4f}")
    md.append("")

    md.append("## C. Endpoint-specific greedy ensembles (alpha-grid components)\n")
    md.append(f"- hepatic ensemble: {hep_sel} → {hep_ens_ci:.4f}")
    md.append(f"- death ensemble: {dea_sel} → {dea_ens_ci:.4f}")
    md.append("")

    md.append("## D. Blend grid vs anchor\n")
    md.append(blend_df.sort_values("weighted_oof", ascending=False)
              .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## E. Candidate decision\n")
    if candidate is None:
        md.append(
            "**No submission recommended.** Even with the LambdaRankers in the "
            "candidate pool, no blend variant clears the +0.002 weighted or "
            "+0.003 hepatic threshold over `phase3_10_horizon_blend_v2`.\n"
        )
    else:
        md.append(
            f"**Submit one**: `{sub_path}`\n"
            f"- blend `{candidate['blend']}` (source={candidate['source']}, alpha={candidate['alpha']}, side={candidate['side']})\n"
            f"- weighted OOF: {candidate['weighted_oof']:.4f} (Δ {candidate['weighted_oof'] - a_w:+.4f})\n"
            f"- hep OOF: {candidate['hep_oof']:.4f} (Δ {candidate['hep_oof'] - a_h_ci:+.4f})\n"
            f"- death OOF: {candidate['death_oof']:.4f} (Δ {candidate['death_oof'] - a_d_ci:+.4f})\n"
            f"- hep fold std/min: {candidate['hep_fold_std']:.4f}/{candidate['hep_fold_min']:.4f}\n"
            f"- dea fold std/min: {candidate['dea_fold_std']:.4f}/{candidate['dea_fold_min']:.4f}\n"
            f"- rho with anchor (test): hep={candidate['rho_h_test_anchor']:.3f}, dea={candidate['rho_d_test_anchor']:.3f}\n"
            f"- hepatic components: {candidate['components_h']}\n"
            f"- death components: {candidate['components_d']}\n"
        )
        md.append("")
        md.append("### Public-LB outcome interpretation\n")
        md.append(
            "- LB > 0.91093 by ≥ +0.001 → LambdaRanker as a pool component "
            "transferred; iterate further on ranker-based components.\n"
            "- LB ≈ 0.91093 (within ±0.001) → ranker overlap with existing pool "
            "is too high to move LB; treat ranker route as exhausted.\n"
            "- LB < 0.91093 by ≥ -0.002 → revert to anchor.\n"
        )

    md.append("\n## Notes\n")
    md.append("- LambdaRanker training reuses the Phase 3.15 recipe: LightGBM with "
              "`objective='lambdarank'`, K=50 controls per event, fold-internal "
              "training only.")
    md.append("- The candidate pool is the union of phase3_9 + phase3_10 + phase3_12 "
              "horizon classifiers plus the LambdaRanker(s); the same greedy capped "
              "logic that built phase3_10_horizon_blend_v2 is applied.")
    md.append("- Death components are unchanged from the anchor's pool unless a "
              "death-side improvement clearly emerges from the greedy logic.")
    md.append("- No event/censoring-age features used for any model.\n")

    out_md = cfg.REPORTS_DIR / "phase3_16_pool_replacement.md"
    out_md.write_text("\n".join(md))
    _LOG.info("wrote %s", out_md)


if __name__ == "__main__":
    main()
