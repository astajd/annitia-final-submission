"""Collect existing-artifact feature importance for hepatic Track A / Track B
proxy models. No retraining. No new fits. Reads only previously written
files under reports/ and pipeline_full/reference_upstream/.

CRITICAL CAVEAT
---------------
The submitted Track A and Track B hepatic anchors are stochastic
ensembles (Track A = 0.30*RSF_landmark + 0.70*permissive[RSF + xgb_cox];
Track B = phase3 horizon-binary rank-blend over LGBM / CatBoost / XGB
across multiple horizons and seeds). The fitted ensemble objects were
not preserved, so per-component importances for the exact submitted
anchors cannot be reproduced.

The tables produced here are importances for PHASE-2 PROXY models
trained during track development on the same feature families. They
are useful for indicative feature ranking but should NOT be read as
attributions for the submitted anchors. Each row carries the proxy-model
label explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TABLES = REPO / "reports" / "explainability" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

GPT_SHAP_DIR = (
    REPO
    / "pipeline_full"
    / "reference_upstream"
    / "gpt_track"
    / "experiments"
    / "outputs"
    / "phase2_feature_importance"
)


def make_row(rank, feature, value, method, component, source, caveat):
    return dict(
        rank=rank,
        feature_name=feature,
        importance_value=value,
        direction="",
        method=method,
        endpoint="hepatic_event",
        component=component,
        source_artifact_or_script=source,
        caveat=caveat,
    )


def aggregate_track_b_hepatic_proxies():
    """Track B (GPT) hepatic anchor — proxy importance.

    We aggregate SHAP mean-abs across the GPT phase-2 hepatic LGBM /
    CatBoost / XGB-Cox binary models that share the longitudinal feature
    family used by the submitted phase-3 horizon blend. This is a proxy:
    the submitted anchor (`20260428_0059_phase3_10_horizon_blend_v2.csv`)
    is a different (greedy rank-blend across horizons) configuration
    whose fitted constituents were not preserved.
    """
    files = [
        ("NIT_plus_scores_longitudinal__hepatic__catboost_binary__shap.csv",
         "phase2 NIT_plus_scores_longitudinal x hepatic x catboost_binary (SHAP, proxy)"),
        ("NIT_plus_scores_longitudinal__hepatic__lgbm_binary__shap.csv",
         "phase2 NIT_plus_scores_longitudinal x hepatic x lgbm_binary (SHAP, proxy)"),
        ("aggressive_longitudinal__hepatic__lgbm_binary__shap.csv",
         "phase2 aggressive_longitudinal x hepatic x lgbm_binary (SHAP, proxy)"),
        ("longitudinal_no_followup_proxies__hepatic__xgb_cox__shap.csv",
         "phase2 longitudinal_no_followup_proxies x hepatic x xgb_cox (SHAP, proxy)"),
    ]
    accum = {}
    sources = {}
    for fname, label in files:
        p = GPT_SHAP_DIR / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        col = "shap_mean_abs" if "shap_mean_abs" in df.columns else df.columns[1]
        # Mean-rank-normalised so models with different scales aggregate fairly.
        ranks = df.set_index("feature")[col].rank(ascending=False, method="min")
        for feat, r in ranks.items():
            accum.setdefault(feat, []).append(float(r))
            sources.setdefault(feat, set()).add(label)
    rows_raw = []
    for feat, ranks in accum.items():
        # Rank-aggregation: lower mean rank = more important
        rows_raw.append((feat, sum(ranks) / len(ranks), len(ranks), sources[feat]))
    rows_raw.sort(key=lambda r: r[1])
    out = []
    for i, (feat, mean_rank, n_models, src) in enumerate(rows_raw[:30], start=1):
        out.append(
            make_row(
                rank=i,
                feature=feat,
                value=round(mean_rank, 3),
                method="aggregated_mean_rank_across_phase2_proxy_SHAP_tables",
                component="Track B (GPT) hepatic anchor — phase-2 proxy aggregate",
                source="pipeline_full/reference_upstream/gpt_track/experiments/outputs/phase2_feature_importance/*.csv",
                caveat=(
                    f"Aggregated mean rank across {n_models} GPT phase-2 hepatic proxy "
                    "SHAP tables (LGBM / CatBoost / XGB-Cox binary). These are proxy "
                    "models, NOT the submitted phase-3 horizon-blend anchor "
                    "(20260428_0059_phase3_10_horizon_blend_v2.csv) whose fitted "
                    "constituents were not preserved. Treat as indicative feature ranking only."
                ),
            )
        )
    return out


def collect_track_a_pointer_table():
    """Track A (Claude) hepatic anchor — no direct importance.

    The fitted RSF and xgb_cox constituents of phase2_blend_2way_optimal
    were not preserved, and no SHAP / permutation artifact exists for
    the blend itself. We emit pointer rows indicating where structural
    development evidence lives, rather than fabricating attributions.
    """
    rows = [
        dict(
            rank=1,
            feature_name="(not available)",
            importance_value="",
            direction="",
            method="not_available",
            endpoint="hepatic_event",
            component="Track A (Claude) hepatic anchor: 0.30*RSF_landmark + 0.70*permissive_blend",
            source_artifact_or_script="pipeline_full/reference_upstream/claude_track/reports/phase2_stack_2way.md (blend selection); phase3_exp1_trajectory_shape.md (trajectory-shape feature audit)",
            caveat=(
                "Fitted RSF and xgb_cox blend members were not preserved. "
                "Computing model-native importance would require a full retrain "
                "matching the 5x10 stratified CV; the brief disallows retraining "
                "for non-lightweight reproductions. Structural development "
                "evidence is documented in the linked Claude-track reports; "
                "the submitted anchor's feature emphasis is therefore not "
                "represented by any per-feature table in this package."
            ),
        )
    ]
    return rows


def main():
    track_b = aggregate_track_b_hepatic_proxies()
    track_a = collect_track_a_pointer_table()
    pd.DataFrame(track_b).to_csv(TABLES / "hepatic_track_b_top_features.csv", index=False)
    pd.DataFrame(track_a).to_csv(TABLES / "hepatic_track_a_top_features.csv", index=False)
    print(f"Wrote {TABLES / 'hepatic_track_b_top_features.csv'} ({len(track_b)} rows)")
    print(f"Wrote {TABLES / 'hepatic_track_a_top_features.csv'} ({len(track_a)} rows)")


if __name__ == "__main__":
    main()
