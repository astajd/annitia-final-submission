"""Phase 3.8 — controlled TabPFN experiment.

Runs TabPFN as a binary ranking classifier on a small set of curated feature
sets, evaluates against the survival C-index of the original endpoints, and
blends with `phase3_5_current_state_v3_hepatic_focused` (current LB best
0.89147) at multiple fixed weights. Promotion criteria are applied; if none
is met we do **not** emit a candidate submission.

Outputs:
    reports/phase3_8_tabpfn_experiment.md
    experiments/outputs/phase3_8_tabpfn/<run>/oof.csv + test.csv
    submissions/<ts>_phase3_8_tabpfn_blend.csv (only if promoted)
    submissions/<ts>_phase3_8_tabpfn_blend.json

No public-LB tuning. No target-derived features (the only target enters via
binary event indicator on the training fold).
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
    nit_plus_scores_longitudinal,
)
from .features.current_state_v2 import current_state_v2
from .metrics import cindex, weighted_score
from .models.ensemble import rank_average
from .models.tabpfn_classifier import run_tabpfn_cv
from .submission import make_submission
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger, timestamp
from .validation import build_folds

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature-set assemblers
# ---------------------------------------------------------------------------

def _drop_visit_history(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in (
        "n_visits", "age_first_visit", "age_last_visit",
        "followup_span", "gap_mean", "gap_max", "gap_min",
    ) if c in df.columns]
    return df.drop(columns=drop) if drop else df


def assemble_feature_sets(ds) -> dict[str, dict[str, pd.DataFrame]]:
    """Return ``name -> {'train': df, 'test': df}``.

    Compact / curated sets only — TabPFN is at its best on small feature
    matrices.
    """
    out: dict[str, dict[str, pd.DataFrame]] = {}

    # A: NIT + clinical scores (already implemented).
    fs = build_feature_set("NIT_plus_scores_longitudinal", ds)
    out["A_NIT_plus_scores"] = {"train": fs.X_train, "test": fs.X_test}

    # B: biomarker-only — labs + NITs trajectories (refined_longitudinal),
    # exclude pure visit-history columns.
    from .features.refined_longitudinal import longitudinal_no_followup_proxies
    Xtr = longitudinal_no_followup_proxies(
        ds.train_df, ds.visit_columns, ds.age_visit_cols,
        keep_stems=LAB_STEMS + NIT_STEMS,
    )
    Xte = longitudinal_no_followup_proxies(
        ds.test_df, ds.visit_columns, ds.age_visit_cols,
        keep_stems=LAB_STEMS + NIT_STEMS,
    )
    out["B_biomarker_only"] = {"train": Xtr, "test": Xte}

    # C: hepatic NIT+scores current state — trim NIT_plus_scores to the
    # hepatic-relevant stems and add latest snapshots.
    Xtr_c = nit_plus_scores_longitudinal(ds.train_df, ds.visit_columns, ds.age_visit_cols)
    Xte_c = nit_plus_scores_longitudinal(ds.test_df, ds.visit_columns, ds.age_visit_cols)
    out["C_hepatic_NIT_scores"] = {"train": Xtr_c, "test": Xte_c}

    # E: current_state_v2_no_visit_history (already in registry)
    fs2 = build_feature_set("current_state_v2_no_visit_history", ds)
    out["E_csv2_no_visit_history"] = {"train": fs2.X_train, "test": fs2.X_test}

    # Top-k variants (D): top-50/100/200 by univariate score will be applied
    # *inside* the TabPFN runner with select_topk; we just supply the full
    # current_state_v2 frame as the source.
    fs3 = build_feature_set("current_state_v2", ds)
    out["D_csv2_full"] = {"train": fs3.X_train, "test": fs3.X_test}

    # All frames must be numeric for TabPFN. Cast / drop non-numeric.
    for name, blob in out.items():
        for k in ("train", "test"):
            df = blob[k]
            num = df.select_dtypes(include=[np.number])
            blob[k] = num
        _LOG.info("feature set %s: train=%s test=%s", name, blob["train"].shape, blob["test"].shape)
    return out


# ---------------------------------------------------------------------------
# Reference + blending
# ---------------------------------------------------------------------------

def _public_best_test() -> tuple[np.ndarray, np.ndarray]:
    cands = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.csv"))
    if not cands:
        return np.array([]), np.array([])
    df = pd.read_csv(cands[-1])
    return df[cfg.SUB_HEPATIC_COL].to_numpy(), df[cfg.SUB_DEATH_COL].to_numpy()


def _v3_oof_predictions(n_train: int) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct v3_hepatic_focused OOF from the candidate JSON's weights.

    v3 used a synthetic in-memory key for the no-visit-history retrain that
    isn't in the on-disk OOF pool. To produce honest blend OOF we recompute
    that piece here.
    """
    from .endpoint_ensemble import collect_predictions

    cands = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.json"))
    if not cands:
        return np.full(n_train, np.nan), np.full(n_train, np.nan)
    blob = json.loads(cands[-1].read_text())
    h_w = blob["hepatic"]["weights"]
    d_w = blob["death"]["weights"]

    # Pull every available phase3 / phase3_5 / phase3_6 OOF (we only need the
    # specific component keys).
    all_dirs = [d for d in cfg.EXPERIMENT_OUTPUTS.iterdir()
                if d.is_dir() and (d / "oof_predictions.csv").exists()]
    oof_df, _, _ = collect_predictions(all_dirs)
    pid_col = cfg.PATIENT_ID_COL
    pos = oof_df[pid_col].map({v: i for i, v in enumerate(load_dataset().train_df[pid_col].values)}).to_numpy().astype(int)

    pool: dict[str, np.ndarray] = {}
    for c in set(h_w) | set(d_w):
        if c in oof_df.columns:
            arr = np.full(n_train, np.nan)
            arr[pos] = oof_df[c].to_numpy()
            pool[c] = arr

    # If the no-visit-history retrain key is missing, materialize it once now.
    nh_key = "phase3_5_no_visit_history::hepatic__rsf__nh"
    if nh_key in h_w and nh_key not in pool:
        ds = load_dataset()
        from .features.hep_focus import current_state_v2_no_visit_history
        from .models import build_model
        Xtr = current_state_v2_no_visit_history(ds.train_df, ds.visit_columns, ds.age_visit_cols)
        Xte = current_state_v2_no_visit_history(ds.test_df, ds.visit_columns, ds.age_visit_cols)
        Xtr = Xtr.select_dtypes(include=[np.number])
        Xte = Xte.reindex(columns=Xtr.columns)
        hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
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
            except Exception as e:  # noqa: BLE001
                _LOG.warning("nh retrain fold (%d,%d) failed: %s", s.repeat, s.fold, e)
        pool[nh_key] = oof

    h_oof = rank_average({c: pool[c] for c in h_w if c in pool}, weights={c: h_w[c] for c in h_w if c in pool}) \
            if any(c in pool for c in h_w) else np.full(n_train, np.nan)
    d_oof = rank_average({c: pool[c] for c in d_w if c in pool}, weights={c: d_w[c] for c in d_w if c in pool}) \
            if any(c in pool for c in d_w) else np.full(n_train, np.nan)
    return h_oof, d_oof


def _rank(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x).rank(method="average", pct=True, na_option="keep")
    return s.to_numpy()


def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """alpha * rank(a) + (1-alpha) * rank(b)."""
    ra = _rank(a)
    rb = _rank(b)
    valid = np.isfinite(ra) & np.isfinite(rb)
    out = np.where(valid, alpha * ra + (1 - alpha) * rb,
                   np.where(np.isfinite(ra), ra, rb))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    cfg.ensure_dirs()
    out_root = cfg.EXPERIMENT_OUTPUTS / "phase3_8_tabpfn"
    out_root.mkdir(parents=True, exist_ok=True)

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)

    feature_sets = assemble_feature_sets(ds)
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)

    # Reference v3 predictions (OOF + test) for blending and rank-corr.
    pub_h_test, pub_d_test = _public_best_test()
    v3_h_oof, v3_d_oof = _v3_oof_predictions(n_train)

    runs: list[dict] = []

    # Run table:
    #   feature_set, endpoint, preprocess, select_topk
    plans = [
        ("A_NIT_plus_scores", "raw", None),
        ("B_biomarker_only", "raw", None),
        ("C_hepatic_NIT_scores", "raw", None),
        ("E_csv2_no_visit_history", "raw", None),
        # Top-k variants on the full csv2.
        ("D_csv2_full", "raw", 50),
        ("D_csv2_full", "raw", 100),
        # Rank-quantile pre-processing on the strongest set.
        ("E_csv2_no_visit_history", "rank_quantile", None),
    ]

    for fs_name, prep, topk in plans:
        Xtr = feature_sets[fs_name]["train"]
        Xte = feature_sets[fs_name]["test"]
        for endpoint_name, ep in [("hepatic", hep), ("death", death)]:
            label = f"{fs_name}__{endpoint_name}__{prep}__top{topk}"
            t0 = time.time()
            try:
                res = run_tabpfn_cv(
                    Xtr, Xte,
                    event=ep.event, time=ep.time,
                    splits=splits,
                    feature_set_name=fs_name,
                    endpoint_name=endpoint_name,
                    preprocess=prep,
                    select_topk=topk,
                )
            except Exception as e:  # noqa: BLE001
                _LOG.error("%s failed: %s", label, e)
                continue
            dt = time.time() - t0
            mean = float(np.mean(res.fold_scores)) if res.fold_scores else float("nan")
            std = float(np.std(res.fold_scores)) if res.fold_scores else float("nan")
            mn = float(np.min(res.fold_scores)) if res.fold_scores else float("nan")
            _LOG.info("%s mean=%.4f std=%.4f min=%.4f total=%.1fs", label, mean, std, mn, dt)

            # Save OOF/test for downstream blending.
            run_dir = out_root / label
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                cfg.PATIENT_ID_COL: ds.train_df[cfg.PATIENT_ID_COL].values,
                "oof": res.oof,
            }).to_csv(run_dir / "oof.csv", index=False)
            pd.DataFrame({
                cfg.TRUSTII_ID_COL: ds.test_df[cfg.TRUSTII_ID_COL].values,
                "test": res.test,
            }).to_csv(run_dir / "test.csv", index=False)

            # rank correlation with v3
            if endpoint_name == "hepatic":
                v3_oof = v3_h_oof
                v3_test = pub_h_test
            else:
                v3_oof = v3_d_oof
                v3_test = pub_d_test
            rho_oof = float(pd.Series(res.oof).rank(pct=True).corr(pd.Series(v3_oof).rank(pct=True), method="spearman")) \
                if v3_oof.size else float("nan")
            rho_test = float(pd.Series(res.test).rank(pct=True).corr(pd.Series(v3_test).rank(pct=True), method="spearman")) \
                if v3_test.size else float("nan")

            runs.append({
                "label": label,
                "feature_set": fs_name,
                "endpoint": endpoint_name,
                "preprocess": prep,
                "select_topk": topk,
                "n_features": int(res.n_features),
                "cindex_mean": mean,
                "cindex_std": std,
                "cindex_min": mn,
                "rho_oof_with_v3": rho_oof,
                "rho_test_with_v3": rho_test,
                "fit_seconds_total": float(res.fit_seconds_total),
                "wall_seconds": float(dt),
                "_oof": res.oof,
                "_test": res.test,
            })

    # ---- Blends with v3 ---------------------------------------------------
    blend_rows: list[dict] = []
    if pub_h_test.size and pub_d_test.size:
        # Pick the best TabPFN run per endpoint by OOF cindex.
        hep_runs = [r for r in runs if r["endpoint"] == "hepatic" and np.isfinite(r["cindex_mean"])]
        dea_runs = [r for r in runs if r["endpoint"] == "death" and np.isfinite(r["cindex_mean"])]
        if hep_runs and dea_runs:
            hep_best = max(hep_runs, key=lambda r: r["cindex_mean"])
            dea_best = max(dea_runs, key=lambda r: r["cindex_mean"])
            for alpha in (1.0, 0.9, 0.8, 0.7, 0.5):
                # alpha * v3 + (1-alpha) * tabpfn
                # Hepatic-only blend
                h_blend_oof = _blend(v3_h_oof, hep_best["_oof"], alpha)
                d_oof = v3_d_oof
                h_test_blend = _blend(pub_h_test, hep_best["_test"], alpha)
                d_test = pub_d_test
                finite_h = np.isfinite(h_blend_oof) & np.isfinite(hep.event.astype(float))
                ci_h = float(cindex(hep.event[finite_h], hep.time[finite_h], h_blend_oof[finite_h]).cindex) \
                       if finite_h.sum() else float("nan")
                ci_d = float(cindex(death.event[np.isfinite(d_oof)], death.time[np.isfinite(d_oof)],
                                    d_oof[np.isfinite(d_oof)]).cindex) if np.isfinite(d_oof).any() else float("nan")
                blend_rows.append({
                    "blend": f"alpha={alpha}_hep_only",
                    "alpha": alpha,
                    "hep_oof": ci_h,
                    "death_oof": ci_d,
                    "weighted_oof": weighted_score(ci_h, ci_d),
                    "h_blend_test": h_test_blend,
                    "d_blend_test": d_test,
                })
                # Death-only blend
                h_oof = v3_h_oof
                d_blend_oof = _blend(v3_d_oof, dea_best["_oof"], alpha)
                h_test = pub_h_test
                d_test_blend = _blend(pub_d_test, dea_best["_test"], alpha)
                finite_d = np.isfinite(d_blend_oof)
                ci_h2 = float(cindex(hep.event[np.isfinite(h_oof)], hep.time[np.isfinite(h_oof)],
                                     h_oof[np.isfinite(h_oof)]).cindex) if np.isfinite(h_oof).any() else float("nan")
                ci_d2 = float(cindex(death.event[finite_d], death.time[finite_d], d_blend_oof[finite_d]).cindex) \
                        if finite_d.sum() else float("nan")
                blend_rows.append({
                    "blend": f"alpha={alpha}_dea_only",
                    "alpha": alpha,
                    "hep_oof": ci_h2,
                    "death_oof": ci_d2,
                    "weighted_oof": weighted_score(ci_h2, ci_d2),
                    "h_blend_test": h_test,
                    "d_blend_test": d_test_blend,
                })
                # Both endpoints
                h_blend_oof = _blend(v3_h_oof, hep_best["_oof"], alpha)
                d_blend_oof = _blend(v3_d_oof, dea_best["_oof"], alpha)
                h_test_blend = _blend(pub_h_test, hep_best["_test"], alpha)
                d_test_blend = _blend(pub_d_test, dea_best["_test"], alpha)
                finite_h = np.isfinite(h_blend_oof)
                finite_d = np.isfinite(d_blend_oof)
                ci_h3 = float(cindex(hep.event[finite_h], hep.time[finite_h], h_blend_oof[finite_h]).cindex) if finite_h.sum() else float("nan")
                ci_d3 = float(cindex(death.event[finite_d], death.time[finite_d], d_blend_oof[finite_d]).cindex) if finite_d.sum() else float("nan")
                blend_rows.append({
                    "blend": f"alpha={alpha}_both",
                    "alpha": alpha,
                    "hep_oof": ci_h3,
                    "death_oof": ci_d3,
                    "weighted_oof": weighted_score(ci_h3, ci_d3),
                    "h_blend_test": h_test_blend,
                    "d_blend_test": d_test_blend,
                })

    # ---- Promotion criteria -----------------------------------------------
    # v3 reference numbers from the candidate metadata.
    v3_meta = {}
    cands = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.json"))
    if cands:
        v3_meta = json.loads(cands[-1].read_text())
    v3_h = v3_meta.get("hepatic", {}).get("oof_cindex", float("nan"))
    v3_d = v3_meta.get("death", {}).get("oof_cindex", float("nan"))
    v3_w = v3_meta.get("weighted_score_cv", float("nan"))

    promoted_blend = None
    promotion_reason = ""
    for row in blend_rows:
        if row["alpha"] == 1.0:
            continue  # alpha=1 is just v3 itself
        improves_weighted = (np.isfinite(row["weighted_oof"]) and np.isfinite(v3_w)
                             and row["weighted_oof"] - v3_w >= 0.002)
        improves_hepatic = (np.isfinite(row["hep_oof"]) and np.isfinite(v3_h)
                            and row["hep_oof"] - v3_h >= 0.003)
        if improves_weighted or improves_hepatic:
            if promoted_blend is None or row["weighted_oof"] > promoted_blend["weighted_oof"]:
                promoted_blend = row
                promotion_reason = (
                    f"+weighted={row['weighted_oof'] - v3_w:.4f}, "
                    f"+hepatic={row['hep_oof'] - v3_h:.4f}"
                )

    # ---- Emit submission if promoted -------------------------------------
    if promoted_blend is not None:
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=promoted_blend["h_blend_test"],
            risk_death=promoted_blend["d_blend_test"],
            sample_submission=ds.sample_submission,
            model_name="phase3_8_tabpfn_blend",
        )
        meta = {
            "label": "phase3_8_tabpfn_blend",
            "feature_set": "see runs (hepatic + death TabPFN best)",
            "target_setup": "binary event indicator on training fold; risk = TabPFN positive-class probability",
            "preprocess": "median imputation; raw or rank_quantile (per run)",
            "blend": promoted_blend["blend"],
            "alpha": promoted_blend["alpha"],
            "hepatic_oof": promoted_blend["hep_oof"],
            "death_oof": promoted_blend["death_oof"],
            "weighted_oof": promoted_blend["weighted_oof"],
            "rank_corr_with_v3": "see report",
            "uses_target_derived_features": False,
            "recommended": True,
            "promotion_reason": promotion_reason,
            "submission_csv": str(sub_path),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        _LOG.info("PROMOTED: %s -> %s", promoted_blend["blend"], sub_path)
    else:
        _LOG.info("No TabPFN blend met promotion criteria.")

    # ---- Report ------------------------------------------------------------
    md_lines: list[str] = []
    md_lines.append("# Phase 3.8 — TabPFN experiment\n")
    md_lines.append(
        f"TabPFN version 2.2.1, device=cpu (NVIDIA driver too old for the cu130 wheel; "
        f"v2.2.1 is the last release that does not require a license token from priorlabs.ai). "
        f"Reference: `phase3_5_current_state_v3_hepatic_focused` LB **0.89147**, "
        f"OOF hep={v3_h:.4f} / death={v3_d:.4f} / weighted={v3_w:.4f}.\n"
    )

    runs_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in runs
    ])
    md_lines.append("## TabPFN per-run CV\n")
    if not runs_df.empty:
        md_lines.append(runs_df.sort_values(["endpoint", "cindex_mean"], ascending=[True, False])
                        .to_markdown(index=False, floatfmt=".4f"))
    else:
        md_lines.append("(no runs)")
    md_lines.append("")

    md_lines.append("## Blends with v3\n")
    if blend_rows:
        b_df = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("h_blend_test") and not k.startswith("d_blend_test")}
            for r in blend_rows
        ])
        md_lines.append(b_df.to_markdown(index=False, floatfmt=".4f"))
    else:
        md_lines.append("(no blends)")
    md_lines.append("")

    md_lines.append("## Promotion decision\n")
    if promoted_blend:
        md_lines.append(f"**Promoted**: `{promoted_blend['blend']}` ({promotion_reason}).")
    else:
        md_lines.append(
            "Not promoted. None of the TabPFN-only or blended candidates improved "
            "weighted OOF by 0.002+ or hepatic OOF by 0.003+ over v3."
        )
    md_lines.append("")

    md_lines.append("## Notes\n")
    md_lines.append(
        "- TabPFN v7.x (current) requires a one-time license token from priorlabs.ai; "
        "v2.2.1 is the last open-weights release and is what we use here.\n"
        "- All runs use median imputation; the rank_quantile preprocess is tested only on "
        "the strongest base set to keep wall time reasonable.\n"
        "- Top-k feature selection is fold-internal (ANOVA-F on training rows) so the "
        "selection is honest under repeated stratified CV.\n"
        "- Risk score = TabPFN's predicted positive-class probability; evaluated against "
        "the *survival* C-index of the original endpoint.\n"
    )

    out_md = cfg.REPORTS_DIR / "phase3_8_tabpfn_experiment.md"
    out_md.write_text("\n".join(md_lines))
    _LOG.info("wrote %s", out_md)


if __name__ == "__main__":
    main()
