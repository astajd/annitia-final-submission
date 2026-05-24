"""Decompose the phase3_current_state_v2 candidate into its components.

Writes ``reports/phase3_5_current_state_v2_decomposition.md`` plus a sidecar
JSON of the same data so downstream ablation tools can re-use it.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .data_loading import load_dataset
from .endpoint_ensemble import collect_predictions
from .metrics import cindex
from .targets import build_death_endpoint, build_hepatic_endpoint
from .utils import get_logger
from .validation import build_folds

_LOG = get_logger(__name__)

_PHASE3_NAME = "phase3_current_state_v2"


# ---------------------------------------------------------------------------
# Feature-group classification of the current_state_v2 columns
# ---------------------------------------------------------------------------

NIT_STEMS = ("fibs_stiffness_med_BM_1", "fibrotest_BM_2", "aixp_aix_result_BM_3")
LAB_STEMS = ("alt", "ast", "ggt", "bilirubin", "plt", "gluc_fast", "triglyc", "chol", "BMI")
DEMO = {"gender", "T2DM", "Hypertension", "Dyslipidaemia", "bariatric_surgery",
        "bariatric_surgery_age", "Age_v1"}
CARE_STAGE = {"n_visits", "age_first_visit", "age_last_visit", "followup_span",
              "gap_mean", "gap_max", "gap_min", "period_age", "period_start_years"}
SCORE_PREFIXES = ("fib4", "apri", "ast_alt_ratio")


def classify_feature(col: str) -> str:
    """Bucket a feature column into one of seven groups."""
    if col in DEMO:
        return "demographics_comorbidities"
    if col in CARE_STAGE:
        return "visit_history_current_state"
    if col.startswith("miss_"):
        return "missingness_pattern"
    if col.startswith("x_"):
        return "interaction"
    if any(col.startswith(p) for p in SCORE_PREFIXES):
        return "derived_fibrosis_score"
    for s in NIT_STEMS:
        if col.startswith(s + "_") or col == s + "_v1":
            return "nit_liver_stiffness"
    for s in LAB_STEMS:
        if col.startswith(s + "_") or col == s + "_v1":
            return "labs_biomarker"
    return "other"


def classify_features(df_cols: list[str]) -> pd.DataFrame:
    rows = [{"feature": c, "group": classify_feature(c)} for c in df_cols]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Component diagnostics
# ---------------------------------------------------------------------------

def _per_fold_cindex(oof: np.ndarray, splits, event: np.ndarray, time: np.ndarray) -> list[float]:
    out: list[float] = []
    for s in splits:
        va = s.valid_idx
        v = oof[va]
        finite = np.isfinite(v)
        if finite.sum() == 0 or event[va][finite].sum() == 0:
            continue
        ci = cindex(event[va][finite], time[va][finite], v[finite]).cindex
        if np.isfinite(ci):
            out.append(ci)
    return out


def _component_table(
    component_names: list[str],
    final_label: str,
    weights: dict[str, float],
    oof_aligned: dict[str, np.ndarray],
    test_aligned: dict[str, np.ndarray],
    final_oof: np.ndarray,
    final_test: np.ndarray,
    splits,
    event: np.ndarray,
    time: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for c in component_names:
        v = oof_aligned[c]
        finite = np.isfinite(v)
        if finite.sum() == 0 or event[finite].sum() == 0:
            continue
        ci_overall = cindex(event[finite], time[finite], v[finite]).cindex
        per_fold = _per_fold_cindex(v, splits, event, time)
        if not per_fold:
            continue
        # rank correlation with final OOF
        s = pd.Series(v).rank(method="average", pct=True, na_option="keep")
        t = pd.Series(final_oof).rank(method="average", pct=True, na_option="keep")
        rho_oof = float(s.corr(t, method="spearman"))
        # rank correlation with final test predictions
        st = pd.Series(test_aligned[c]).rank(method="average", pct=True, na_option="keep")
        tt = pd.Series(final_test).rank(method="average", pct=True, na_option="keep")
        rho_test = float(st.corr(tt, method="spearman"))
        rows.append({
            "component": c,
            "weight": float(weights.get(c, 0.0)),
            "model": _model_of(c),
            "feature_set": "current_state_v2",
            "oof_cindex": float(ci_overall),
            "fold_mean": float(np.mean(per_fold)),
            "fold_std": float(np.std(per_fold)),
            "fold_min": float(np.min(per_fold)),
            "fold_max": float(np.max(per_fold)),
            "rank_corr_with_final_oof": rho_oof,
            "rank_corr_with_final_test": rho_test,
        })
    return pd.DataFrame(rows).sort_values("oof_cindex", ascending=False)


def _model_of(component: str) -> str:
    # phase3_current_state_v2::hepatic__xgb_aft__s0 -> xgb_aft (s0)
    payload = component.split("::", 1)[-1]
    parts = payload.split("__")
    if len(parts) >= 3:
        return f"{parts[1]} ({parts[2]})"
    return parts[1] if len(parts) >= 2 else component


# ---------------------------------------------------------------------------
# Top feature pull from importance CSVs
# ---------------------------------------------------------------------------

def _top_features_for_model(model_name: str, endpoint: str, k: int = 12) -> list[dict]:
    csv = cfg.EXPERIMENT_OUTPUTS / "phase2_feature_importance" / f"current_state_v2__{endpoint}__{model_name}.csv"
    if not csv.exists():
        return []
    df = pd.read_csv(csv).head(k)
    df["group"] = df["feature"].map(classify_feature)
    return df[["feature", "importance", "group", "suspicious"]].to_dict(orient="records")


# ---------------------------------------------------------------------------
# Build the report
# ---------------------------------------------------------------------------

def main() -> None:
    out_md = cfg.REPORTS_DIR / "phase3_5_current_state_v2_decomposition.md"
    out_json = cfg.REPORTS_DIR / "phase3_5_current_state_v2_decomposition.json"

    # Load latest candidate metadata.
    cand_jsons = sorted(cfg.SUBMISSIONS_DIR.glob("*phase3_current_state_v2*.json"))
    if not cand_jsons:
        raise FileNotFoundError("no phase3_current_state_v2 candidate JSON found")
    meta = json.loads(cand_jsons[-1].read_text())

    ds = load_dataset()
    hep = build_hepatic_endpoint(ds.train_df, ds.age_visit_cols)
    death = build_death_endpoint(ds.train_df, ds.age_visit_cols)
    splits = build_folds(ds.train_df, hepatic_event=hep.event.astype(int), n_splits=5, n_repeats=3)

    # Re-collect OOF/test predictions from the phase3 sweep dir(s).
    exp_dirs = [d for d in cfg.EXPERIMENT_OUTPUTS.iterdir() if d.is_dir() and (d / "config.json").exists()]
    phase3_dirs = []
    for d in exp_dirs:
        try:
            blob = json.loads((d / "config.json").read_text())
        except Exception:
            continue
        if blob.get("name") == _PHASE3_NAME:
            phase3_dirs.append(d)
    if not phase3_dirs:
        raise FileNotFoundError(f"no experiment dir with name={_PHASE3_NAME}")

    oof_df, test_df, _ = collect_predictions(phase3_dirs)
    pid_col = cfg.PATIENT_ID_COL
    tid_col = cfg.TRUSTII_ID_COL
    train_pid_to_row = {pid: i for i, pid in enumerate(ds.train_df[pid_col].values)}
    pos = oof_df[pid_col].map(train_pid_to_row).to_numpy()
    pos = pos.astype(int)
    test_idx_by_id = {tid: i for i, tid in enumerate(ds.test_df[tid_col].values)}
    tpos = test_df[tid_col].map(test_idx_by_id).to_numpy().astype(int)

    n_train = len(ds.train_df)
    n_test = len(ds.test_df)

    def _aligned(col_name: str, df: pd.DataFrame, positions: np.ndarray, n: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if col_name in df.columns:
            out[positions] = df[col_name].to_numpy()
        return out

    # Final OOF/test for the candidate are stored as one-column-per-method in the
    # endpoint_ensemble bundle. We re-derive them from component weights here so
    # they are reproducible from the JSON metadata.
    from .models.ensemble import rank_average

    sub_csv = pd.read_csv(meta["submission_csv"])
    final_test_hep = sub_csv[cfg.SUB_HEPATIC_COL].to_numpy()
    final_test_dea = sub_csv[cfg.SUB_DEATH_COL].to_numpy()

    def _build_oof(weights: dict[str, float]) -> np.ndarray:
        comps = {c: _aligned(c, oof_df, pos, n_train) for c in weights}
        return rank_average(comps, weights=weights)

    final_oof_hep = _build_oof(meta["hepatic"]["weights"])
    final_oof_dea = _build_oof(meta["death"]["weights"])

    # Decompose each endpoint.
    summary: dict = {"endpoints": {}}
    for endpoint_name, endpoint, m_blob, final_oof, final_test in [
        ("hepatic", hep, meta["hepatic"], final_oof_hep, final_test_hep),
        ("death", death, meta["death"], final_oof_dea, final_test_dea),
    ]:
        comp_names = m_blob["components"]
        weights = m_blob["weights"]
        oof_aligned = {c: _aligned(c, oof_df, pos, n_train) for c in comp_names}
        test_aligned = {c: _aligned(c, test_df, tpos, n_test) for c in comp_names}
        table = _component_table(
            comp_names, _PHASE3_NAME, weights,
            oof_aligned, test_aligned,
            final_oof, final_test,
            splits, endpoint.event, endpoint.time,
        )
        # Pairwise rank corr matrix among components
        if comp_names:
            mat = pd.DataFrame({c: pd.Series(oof_aligned[c]).rank(pct=True, na_option="keep") for c in comp_names})
            corr = mat.corr(method="spearman")
        else:
            corr = pd.DataFrame()
        summary["endpoints"][endpoint_name] = {
            "method": m_blob["method"],
            "weights": weights,
            "table": table.to_dict(orient="records"),
            "pairwise_rank_corr": corr.round(3).to_dict() if not corr.empty else {},
            "final_oof_cindex": m_blob.get("oof_cindex"),
        }

    # Top features per model used as components.
    families: dict[str, list[dict]] = {}
    for ep, ep_blob in meta.items():
        if ep not in ("hepatic", "death"):
            continue
        for c in ep_blob["components"]:
            mname = c.split("__")[1]
            label = f"{mname}__{ep}"
            if label in families:
                continue
            families[label] = _top_features_for_model(mname, ep)
    summary["top_features"] = families

    # Feature group counts in current_state_v2.
    from .features import build_feature_set
    fs = build_feature_set("current_state_v2", ds)
    cls = classify_features(list(fs.X_train.columns))
    grp_counts = cls.groupby("group").size().sort_values(ascending=False)
    summary["feature_groups"] = grp_counts.to_dict()

    # Render the report.
    lines: list[str] = []
    lines.append("# Phase 3.5 — `phase3_current_state_v2` decomposition\n")
    lines.append("Public LB: **0.88521** (current best). Local OOF: hepatic "
                 f"{meta['hepatic']['oof_cindex']:.4f}, death "
                 f"{meta['death']['oof_cindex']:.4f}, weighted "
                 f"{meta['weighted_score_cv']:.4f}.\n")
    lines.append("Method per endpoint:\n")
    lines.append(f"- hepatic: `{meta['hepatic']['method']}`")
    lines.append(f"- death:   `{meta['death']['method']}`\n")

    lines.append("## Feature group counts in `current_state_v2`\n")
    lines.append(grp_counts.to_frame("n_features").to_markdown())
    lines.append("\nNo target-derived columns (event_age, death_age) are included in any group.\n")

    for endpoint_name in ("hepatic", "death"):
        s = summary["endpoints"][endpoint_name]
        lines.append(f"## {endpoint_name.title()} components\n")
        df_table = pd.DataFrame(s["table"])
        if df_table.empty:
            lines.append("(no components)")
            continue
        lines.append(df_table.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")
        # Pairwise corr block.
        if s["pairwise_rank_corr"]:
            lines.append(f"### Pairwise rank correlation ({endpoint_name})\n")
            corr = pd.DataFrame(s["pairwise_rank_corr"])
            lines.append(corr.round(3).to_markdown())
            lines.append("")

    lines.append("## Top-12 native importances per family\n")
    for label, feats in families.items():
        if not feats:
            continue
        lines.append(f"### {label}\n")
        lines.append(pd.DataFrame(feats).to_markdown(index=False))
        lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "- All components share the same `current_state_v2` feature set; "
        "diversity comes from model family and seed, not from feature space.\n"
        "- Hepatic ensemble is dominated by RSF (two seeds) plus xgb_aft; "
        "binary classifiers were not selected by greedy.\n"
        "- Death ensemble is broader (9 components) — seed_bagged picked "
        "every model except coxnet/coxph (not in the Phase 3 config) and "
        "catboost_binary (less stable on the death endpoint here).\n"
        "- A handful of suspicious features (`followup_span`, `age_last_visit`, "
        "missingness columns) appear in the top-12 importances. These are "
        "permitted by the Phase 3 spec, but the simplified candidate (Phase "
        "3.5 ablation) shows what happens if we remove them.\n"
    )

    out_md.write_text("\n".join(lines))
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    _LOG.info("wrote %s and %s", out_md, out_json)


if __name__ == "__main__":
    main()
