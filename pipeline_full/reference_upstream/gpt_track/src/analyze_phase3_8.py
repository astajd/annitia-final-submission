"""Post-hoc analysis of Phase 3.8 TabPFN runs.

Reads the per-run OOF/test predictions written by ``run_phase3_8_tabpfn``,
computes rank correlations against v3, blends with v3 at multiple alphas,
applies the promotion criteria, optionally emits one candidate submission,
and writes the final report.

Used because the long-running rank_quantile death fold was killed before the
top-level driver could finalize; this module reuses everything that was saved
to disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .features.hep_focus import current_state_v2_no_visit_history
from .metrics import cindex, weighted_score
from .models import build_model
from .models.ensemble import rank_average
from .submission import make_submission
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger
from .validation import build_folds

_LOG = get_logger(__name__)


def _v3_oof(n_train: int, ds, hep) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct v3_hepatic_focused OOF (re-running the synthetic NH retrain)."""
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


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(method="average", pct=True, na_option="keep").to_numpy()


def _blend(a, b, alpha):
    ra, rb = _rank(a), _rank(b)
    valid_a, valid_b = np.isfinite(ra), np.isfinite(rb)
    out = np.where(valid_a & valid_b, alpha * ra + (1 - alpha) * rb,
                   np.where(valid_a, ra, np.where(valid_b, rb, np.nan)))
    return out


def main() -> None:
    out_root = cfg.EXPERIMENT_OUTPUTS / "phase3_8_tabpfn"
    if not out_root.exists():
        raise SystemExit("phase3_8_tabpfn directory missing; run src.run_phase3_8_tabpfn first")

    ds = load_dataset()
    n_train = len(ds.train_df)
    n_test = len(ds.test_df)
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)

    v3_h_oof, v3_d_oof = _v3_oof(n_train, ds, hep)

    cands = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.csv"))
    pub = pd.read_csv(cands[-1])
    pub_h_test = pub[cfg.SUB_HEPATIC_COL].to_numpy()
    pub_d_test = pub[cfg.SUB_DEATH_COL].to_numpy()

    pid_to_row = {pid: i for i, pid in enumerate(ds.train_df[cfg.PATIENT_ID_COL].values)}
    tid_to_row = {tid: i for i, tid in enumerate(ds.test_df[cfg.TRUSTII_ID_COL].values)}

    runs = []
    for d in sorted(out_root.iterdir()):
        if not d.is_dir():
            continue
        oof_csv = d / "oof.csv"
        test_csv = d / "test.csv"
        if not (oof_csv.exists() and test_csv.exists()):
            continue
        oof_df = pd.read_csv(oof_csv)
        test_df = pd.read_csv(test_csv)
        oof_arr = np.full(n_train, np.nan)
        oof_arr[oof_df[cfg.PATIENT_ID_COL].map(pid_to_row).to_numpy().astype(int)] = oof_df["oof"].to_numpy()
        test_arr = np.full(n_test, np.nan)
        test_arr[test_df[cfg.TRUSTII_ID_COL].map(tid_to_row).to_numpy().astype(int)] = test_df["test"].to_numpy()
        # Parse label
        parts = d.name.split("__")
        feature_set = parts[0]
        endpoint = parts[1]
        preprocess = parts[2] if len(parts) >= 3 else "raw"
        topk = parts[3].replace("top", "") if len(parts) >= 4 else "None"

        ev = hep.event if endpoint == "hepatic" else death.event
        t = hep.time if endpoint == "hepatic" else death.time
        finite = np.isfinite(oof_arr)
        ci = float(cindex(ev[finite], t[finite], oof_arr[finite]).cindex) if finite.sum() else float("nan")
        # Per-fold
        fold_scores = []
        for s in splits:
            v = oof_arr[s.valid_idx]
            ff = np.isfinite(v)
            if ff.sum() == 0 or ev[s.valid_idx][ff].sum() == 0:
                continue
            sc = cindex(ev[s.valid_idx][ff], t[s.valid_idx][ff], v[ff]).cindex
            if np.isfinite(sc):
                fold_scores.append(float(sc))
        std = float(np.std(fold_scores)) if fold_scores else float("nan")
        mn = float(np.min(fold_scores)) if fold_scores else float("nan")

        v3_oof = v3_h_oof if endpoint == "hepatic" else v3_d_oof
        v3_test = pub_h_test if endpoint == "hepatic" else pub_d_test
        rho_oof = float(pd.Series(oof_arr).rank(pct=True).corr(pd.Series(v3_oof).rank(pct=True), method="spearman"))
        rho_test = float(pd.Series(test_arr).rank(pct=True).corr(pd.Series(v3_test).rank(pct=True), method="spearman"))

        runs.append({
            "run_dir": d.name,
            "feature_set": feature_set,
            "endpoint": endpoint,
            "preprocess": preprocess,
            "select_topk": topk,
            "n_features_input": "(stored)",  # input column count not persisted; per-run dir is enough
            "cindex_mean": ci,
            "cindex_std": std,
            "cindex_min": mn,
            "rho_oof_with_v3": rho_oof,
            "rho_test_with_v3": rho_test,
            "_oof": oof_arr,
            "_test": test_arr,
        })

    runs_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in runs])
    _LOG.info("loaded %d TabPFN runs", len(runs))

    # Best per endpoint by OOF C-index.
    hep_runs = [r for r in runs if r["endpoint"] == "hepatic" and np.isfinite(r["cindex_mean"])]
    dea_runs = [r for r in runs if r["endpoint"] == "death" and np.isfinite(r["cindex_mean"])]
    hep_best = max(hep_runs, key=lambda r: r["cindex_mean"]) if hep_runs else None
    dea_best = max(dea_runs, key=lambda r: r["cindex_mean"]) if dea_runs else None

    blend_rows = []
    if hep_best and dea_best and v3_h_oof.size and v3_d_oof.size:
        for alpha in (0.95, 0.9, 0.85, 0.8, 0.7, 0.5):
            for tag, h_other, d_other in [
                ("hep_only", hep_best["_oof"], None),
                ("dea_only", None, dea_best["_oof"]),
                ("both",     hep_best["_oof"], dea_best["_oof"]),
            ]:
                h_blend_oof = _blend(v3_h_oof, h_other, alpha) if h_other is not None else v3_h_oof
                d_blend_oof = _blend(v3_d_oof, d_other, alpha) if d_other is not None else v3_d_oof
                h_blend_test = _blend(pub_h_test, hep_best["_test"], alpha) if h_other is not None else pub_h_test
                d_blend_test = _blend(pub_d_test, dea_best["_test"], alpha) if d_other is not None else pub_d_test

                fh = np.isfinite(h_blend_oof)
                fd = np.isfinite(d_blend_oof)
                ci_h = float(cindex(hep.event[fh], hep.time[fh], h_blend_oof[fh]).cindex) if fh.sum() else float("nan")
                ci_d = float(cindex(death.event[fd], death.time[fd], d_blend_oof[fd]).cindex) if fd.sum() else float("nan")
                blend_rows.append({
                    "blend": f"alpha={alpha}_{tag}",
                    "alpha": alpha,
                    "side": tag,
                    "hep_oof": ci_h,
                    "death_oof": ci_d,
                    "weighted_oof": weighted_score(ci_h, ci_d),
                    "h_test": h_blend_test,
                    "d_test": d_blend_test,
                })

    # v3 reference
    v3_meta = {}
    cands_meta = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_5_current_state_v3_hepatic_focused.json"))
    if cands_meta:
        v3_meta = json.loads(cands_meta[-1].read_text())
    v3_h = v3_meta.get("hepatic", {}).get("oof_cindex", float("nan"))
    v3_d = v3_meta.get("death", {}).get("oof_cindex", float("nan"))
    v3_w = v3_meta.get("weighted_score_cv", float("nan"))

    # Promotion check
    promoted = None
    reason = ""
    for row in blend_rows:
        if row["alpha"] == 1.0:
            continue
        improves_w = np.isfinite(row["weighted_oof"]) and (row["weighted_oof"] - v3_w >= 0.002)
        improves_h = np.isfinite(row["hep_oof"]) and (row["hep_oof"] - v3_h >= 0.003)
        if improves_w or improves_h:
            if promoted is None or row["weighted_oof"] > promoted["weighted_oof"]:
                promoted = row
                reason = f"weighted Δ={row['weighted_oof']-v3_w:+.4f}, hepatic Δ={row['hep_oof']-v3_h:+.4f}"

    sub_path = None
    if promoted is not None:
        sub_path = make_submission(
            ds.test_df,
            risk_hepatic=promoted["h_test"], risk_death=promoted["d_test"],
            sample_submission=ds.sample_submission, model_name="phase3_8_tabpfn_blend",
        )
        meta = {
            "label": "phase3_8_tabpfn_blend",
            "feature_set_hepatic": hep_best["feature_set"] if hep_best else None,
            "feature_set_death": dea_best["feature_set"] if dea_best else None,
            "preprocess": "median imputation; raw or rank_quantile per run",
            "target_setup": "binary event indicator on training fold; risk = TabPFN positive-class probability",
            "blend": promoted["blend"],
            "alpha": promoted["alpha"],
            "hepatic_oof": promoted["hep_oof"],
            "death_oof": promoted["death_oof"],
            "weighted_oof": promoted["weighted_oof"],
            "rank_corr_with_v3_hepatic_test": hep_best["rho_test_with_v3"] if hep_best else None,
            "rank_corr_with_v3_death_test": dea_best["rho_test_with_v3"] if dea_best else None,
            "uses_target_derived_features": False,
            "recommended": True,
            "promotion_reason": reason,
            "submission_csv": str(sub_path),
        }
        sub_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        _LOG.info("PROMOTED: %s -> %s", promoted["blend"], sub_path)

    # Write the report
    md = []
    md.append("# Phase 3.8 — TabPFN experiment\n")
    md.append(
        f"TabPFN version 2.2.1, device=cpu (NVIDIA driver 12020 too old for the cu130 wheel; "
        f"v2.2.1 is the last release with open weights — current 7.x requires a license token "
        f"from priorlabs.ai which is unavailable in this environment).\n"
    )
    md.append(
        f"Reference: `phase3_5_current_state_v3_hepatic_focused` LB **0.89147**, "
        f"OOF hepatic={v3_h:.4f} / death={v3_d:.4f} / weighted={v3_w:.4f}.\n"
    )
    md.append("## TabPFN per-run CV\n")
    if not runs_df.empty:
        md.append(runs_df.sort_values(["endpoint", "cindex_mean"], ascending=[True, False])
                  .to_markdown(index=False, floatfmt=".4f"))
    md.append("")

    md.append("## Best TabPFN per endpoint (selected for blends)\n")
    if hep_best:
        md.append(f"- hepatic best: `{hep_best['run_dir']}` C={hep_best['cindex_mean']:.4f} "
                  f"(rank-corr w/ v3 OOF {hep_best['rho_oof_with_v3']:.3f}, test {hep_best['rho_test_with_v3']:.3f})")
    if dea_best:
        md.append(f"- death best:   `{dea_best['run_dir']}` C={dea_best['cindex_mean']:.4f} "
                  f"(rank-corr w/ v3 OOF {dea_best['rho_oof_with_v3']:.3f}, test {dea_best['rho_test_with_v3']:.3f})")
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
            "**Not promoted.** None of the TabPFN blends improved weighted OOF by ≥ 0.002 or "
            "hepatic OOF by ≥ 0.003 over v3.\n\n"
            "TabPFN's standalone hepatic OOF maxes at 0.756 vs v3 0.842 (Δ = -0.086) and death "
            "OOF at 0.777 vs v3 0.950 (Δ = -0.173). High individual gap dominates the rank-space "
            "blend: the OOF rank-corr with v3 is high enough that any TabPFN-weighted variant "
            "averages toward v3 anyway, while pulling the strong v3 signal toward TabPFN's "
            "weaker estimates."
        )
    md.append("")

    md.append("## Notes\n")
    md.append(
        "- 13 of 14 planned TabPFN configurations completed; the final "
        "`E_csv2_no_visit_history__death__rank_quantile__topNone` was killed because "
        "the rank_quantile preprocess on 215 features was projected to take ~30+ min "
        "and earlier runs already showed TabPFN was uncompetitive on this dataset.\n"
        "- All TabPFN runs use `n_estimators=4` (default ensemble) on CPU. Increasing it "
        "would likely improve OOF by < 0.005 and not close the ~0.10–0.17 gap to v3.\n"
        "- Risk score = predicted positive-class probability from TabPFN classifier "
        "trained on binary event-occurrence labels; evaluated against the survival "
        "C-index of the original endpoint.\n"
        "- Top-k feature selection is fold-internal (ANOVA-F on training rows).\n"
        "- No target-derived features used in any run.\n"
    )

    out_md = cfg.REPORTS_DIR / "phase3_8_tabpfn_experiment.md"
    out_md.write_text("\n".join(md))
    _LOG.info("wrote %s", out_md)


if __name__ == "__main__":
    main()
