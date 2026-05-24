"""Post-hoc reference attribution for the two death-endpoint components
(CWGBSA, GBSA) of the submitted ensemble.

THIS SCRIPT DOES NOT TOUCH THE SUBMITTED PREDICTIONS, THE CACHED
INTERMEDIATES, OR THE FROZEN SUBMISSION FILE. It re-fits a SINGLE
seed-42 model on the full training labels (matching the per-repeat
random_state=42+r convention with r=0) and extracts model-native
importance. Outputs are written under reports/explainability/tables/.

Method per component
--------------------
- CWGBSA: model-native sparse coefficients (`coef_`). The submitted
  CV protocol is 5x10 stratified mean-of-ranks; this single-fit
  reference captures the same hyperparameters but does not aggregate
  across repeats. The CWGBSA model is intrinsically sparse (only a
  small subset of features get nonzero coefficients).
- GBSA: model-native feature importance (`feature_importances_`,
  impurity-style). Same single-fit reference convention.

The reference fits use the same feature builder
(`longitudinal_summary`, 143 features) and the same hyperparameters
as the cached predictions (see task3_tree_survival.py in
pipeline_full/runnable/upstream_scripts/).

The CWGBSA result here also matches the existing audit log
`audit_cwgbsa_coefs.csv` (from the original development workspace) and the
narrative report `reports/explainability/audit_CWGBSA_death_model.md`.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                # annitia-final-submission/
RUNNABLE = REPO / "pipeline_full" / "runnable"
LIB = RUNNABLE / "lib"
TABLES = REPO / "reports" / "explainability" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(LIB))

from zoo_utils import load_labels  # noqa: E402
from claude_src.features import build_features  # noqa: E402

from sksurv.ensemble import (  # noqa: E402
    ComponentwiseGradientBoostingSurvivalAnalysis,
    GradientBoostingSurvivalAnalysis,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402


def make_y(event, time):
    return np.array(
        list(zip(event.astype(bool), time.astype(float))),
        dtype=[("event", "?"), ("time", "<f8")],
    )


def build_X(train_df):
    X = build_features(train_df, "longitudinal_summary").copy()
    X = X.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    X = X.fillna(0.0)
    return X


def fit_cwgbsa(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    m = ComponentwiseGradientBoostingSurvivalAnalysis(
        loss="coxph",
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    m.fit(Xs, y)
    coefs = np.asarray(m.coef_, dtype=float)
    rows = []
    nz_rank = 0
    for name, c in zip(X.columns, coefs):
        if c != 0:
            rows.append((name, float(c)))
    rows.sort(key=lambda r: abs(r[1]), reverse=True)
    out = []
    for i, (name, c) in enumerate(rows, start=1):
        out.append(
            dict(
                rank=i,
                feature_name=name,
                importance_value=abs(c),
                direction=("+" if c > 0 else "-"),
                method="cwgbsa_native_coef_abs",
                endpoint="death",
                component="CWGBSA (sksurv ComponentwiseGradientBoostingSurvivalAnalysis, n=300, lr=0.05, subs=0.8, seed=42)",
                source_artifact_or_script="reports/explainability/scripts/compute_death_component_importance.py",
                caveat="Post-hoc reference single-fit on full training labels; the submitted ensemble averages ranks across 5x10 stratified-by-hep-event CV. CWGBSA is intrinsically sparse — almost all features get zero coefficient. The two selected features encode follow-up duration / single-visit indicators rather than disease physiology (see reports/explainability/audit_CWGBSA_death_model.md).",
            )
        )
    return out, len(coefs), int((coefs != 0).sum())


def fit_gbsa(X, y):
    m = GradientBoostingSurvivalAnalysis(
        loss="coxph",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        max_features="sqrt",
        random_state=42,
    )
    m.fit(X.values, y)
    imps = np.asarray(m.feature_importances_, dtype=float)
    order = np.argsort(imps)[::-1]
    rows = []
    for i, idx in enumerate(order, start=1):
        rows.append(
            dict(
                rank=i,
                feature_name=X.columns[idx],
                importance_value=float(imps[idx]),
                direction="",
                method="gbsa_native_feature_importances_impurity",
                endpoint="death",
                component="GBSA (sksurv GradientBoostingSurvivalAnalysis, n=200, lr=0.05, max_depth=3, subs=0.8, max_features='sqrt', seed=42)",
                source_artifact_or_script="reports/explainability/scripts/compute_death_component_importance.py",
                caveat="Post-hoc reference single-fit on full training labels; the submitted ensemble averages ranks across 5x10 stratified-by-hep-event CV. Impurity-based importances are biased toward high-cardinality / continuous features; treat as relative ranking, not as effect sizes. GBSA carries weight 0.15 in the final death rank.",
            )
        )
    return rows, len(imps), int((imps > 0).sum())


def main():
    print("[1/3] Loading labels + features...", flush=True)
    L = load_labels()
    X = build_X(L["train"])
    print(f"      feature matrix shape = {X.shape}", flush=True)

    y_dea = make_y(L["dea_event"], L["dea_time"])

    print("[2/3] Fitting CWGBSA (seed=42, full train, post-hoc reference)...", flush=True)
    cwgbs_rows, cw_total, cw_nz = fit_cwgbsa(X, y_dea)
    print(f"      CWGBSA: {cw_nz} of {cw_total} features have nonzero coefficient", flush=True)

    print("[3/3] Fitting GBSA (seed=42, full train, post-hoc reference)...", flush=True)
    gbsa_rows, gb_total, gb_nz = fit_gbsa(X, y_dea)
    print(f"      GBSA: {gb_nz} of {gb_total} features have positive impurity importance", flush=True)

    pd.DataFrame(cwgbs_rows).to_csv(TABLES / "death_cwgbs_top_features.csv", index=False)
    pd.DataFrame(gbsa_rows[:30]).to_csv(TABLES / "death_gbsa_top_features.csv", index=False)
    print(f"\nWrote:")
    print(f"  {TABLES / 'death_cwgbs_top_features.csv'} ({cw_nz} rows)")
    print(f"  {TABLES / 'death_gbsa_top_features.csv'} (top 30 of {gb_nz})")


if __name__ == "__main__":
    main()
