"""Task 3 — Tree survival diversity.

Train sksurv models not yet used by either track:
  - ExtraSurvivalTrees (EST)
  - GradientBoostingSurvivalAnalysis (GBSA, coxph)
  - ComponentwiseGradientBoostingSurvivalAnalysis (CWGBS) — sparse, additive
on a stable feature set (current_state_v2, no_visit_history, baseline_v1).

OOF + test under Claude 5x10 CV.  Hep + dea endpoints.
Saves per-model OOF/test, summary CSV, and a markdown report.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore")

# Path resolution for the assembled repo (see lib/zoo_utils.py).
HERE = Path(__file__).resolve().parent
RUNNABLE = HERE.parent
sys.path.insert(0, str(RUNNABLE / "lib"))

from zoo_utils import load_labels, load_anchors, fold_seed_cv_oof_test, save_pred  # noqa: E402
from claude_src.cv import cindex  # noqa: E402

ROOT = RUNNABLE / "cached_intermediates"
LOGS = RUNNABLE / "_outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def get_features(name, train, test):
    """Return X_train (1253, p), X_test (423, p) using Claude builder."""
    from claude_src.features import build_features
    Xtr = build_features(train, name).copy()
    Xte = build_features(test, name).copy()
    Xtr = Xtr.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    Xte = Xte.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    common = [c for c in Xtr.columns if c in Xte.columns]
    Xtr = Xtr[common].fillna(0.0)
    Xte = Xte[common].fillna(0.0)
    return Xtr, Xte


def make_y(event, time):
    return np.array(list(zip(event.astype(bool), time.astype(float))),
                    dtype=[("event", "?"), ("time", "<f8")])


def fit_predict_est_factory(seed, n_estimators=400, min_samples_leaf=15, max_features="sqrt"):
    from sksurv.ensemble import ExtraSurvivalTrees

    def fp(X_tr, e_tr, t_tr, X_va):
        y = make_y(e_tr, t_tr)
        m = ExtraSurvivalTrees(n_estimators=n_estimators,
                               min_samples_leaf=min_samples_leaf,
                               min_samples_split=2 * min_samples_leaf,
                               max_features=max_features,
                               n_jobs=-1, random_state=seed)
        m.fit(X_tr.values, y)
        return m.predict(X_va.values)

    return fp


def fit_predict_gbsa_factory(seed, n_estimators=200, learning_rate=0.05,
                              max_depth=3, subsample=0.8, max_features="sqrt"):
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis

    def fp(X_tr, e_tr, t_tr, X_va):
        y = make_y(e_tr, t_tr)
        m = GradientBoostingSurvivalAnalysis(loss="coxph", n_estimators=n_estimators,
                                              learning_rate=learning_rate,
                                              max_depth=max_depth, subsample=subsample,
                                              max_features=max_features,
                                              random_state=seed)
        m.fit(X_tr.values, y)
        return m.predict(X_va.values)

    return fp


def fit_predict_cwgbs_factory(seed, n_estimators=300, learning_rate=0.05, subsample=0.8):
    from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
    from sklearn.preprocessing import StandardScaler

    def fp(X_tr, e_tr, t_tr, X_va):
        y = make_y(e_tr, t_tr)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(X_tr.values)
        Xva_s = scaler.transform(X_va.values)
        m = ComponentwiseGradientBoostingSurvivalAnalysis(loss="coxph",
                                                          n_estimators=n_estimators,
                                                          learning_rate=learning_rate,
                                                          subsample=subsample,
                                                          random_state=seed)
        m.fit(Xtr_s, y)
        return m.predict(Xva_s)

    return fp


def run_endpoint(L, A, kind, feature_name, factories, rows):
    Xtr, Xte = get_features(feature_name, L["train"], L["test"])
    print(f"  feature `{feature_name}` shape={Xtr.shape}", flush=True)
    if kind == "hep":
        ev, tm = L["hep_event"], L["hep_time"]
        anchor_col = "risk_hepatic_event"
    else:
        ev, tm = L["dea_event"], L["dea_time"]
        anchor_col = "risk_death"
    from zoo_utils import PRED_DIR
    for tag, factory_fn in factories:
        full_name = f"survtree__{kind}__{feature_name}__{tag}"
        # Skip if both files already exist
        if (PRED_DIR / f"oof__{full_name}.csv").exists() and (PRED_DIR / f"test__{full_name}.csv").exists():
            print(f"    {tag}: SKIP (predictions exist)", flush=True)
            continue
        try:
            oof, te, fcis = fold_seed_cv_oof_test(factory_fn, Xtr, Xte, ev, tm)
            c = cindex(ev, tm, oof)
            rho_m50 = float(spearmanr(te, A["merge5050_csv"][anchor_col]).statistic)
            rho_g   = float(spearmanr(te, A["gpt_anchor_csv"][anchor_col]).statistic)
            rho_c   = float(spearmanr(te, A["cl_anchor_csv"][anchor_col]).statistic)
            save_pred(full_name, L["pid"], oof, L["tid"], te, kind=kind)
            row = dict(model=full_name, kind=kind, feature=feature_name, variant=tag,
                       oof_c=float(c), fold_mean=float(fcis.mean()),
                       fold_std=float(fcis.std(ddof=1)), worst_fold=float(fcis.min()),
                       rho_m50=rho_m50, rho_gpt=rho_g, rho_claude=rho_c,
                       n_features=Xtr.shape[1])
            rows.append(row)
            # Append-write summary so we don't lose progress
            pd.DataFrame([row]).to_csv(LOGS / "task3_tree_survival_summary.csv",
                                        mode="a", header=not (LOGS / "task3_tree_survival_summary.csv").exists(),
                                        index=False)
            print(f"    {tag}: OOF {kind}={c:.4f}  fold_std={fcis.std(ddof=1):.3f}  rho_m50={rho_m50:.3f}", flush=True)
        except Exception as e:
            print(f"    {tag}: FAILED ({type(e).__name__}: {e})", flush=True)
            rows.append(dict(model=full_name, kind=kind, feature=feature_name, variant=tag,
                             oof_c=float("nan"), error=str(e)))


def main():
    L = load_labels()
    A = load_anchors()
    rows = []

    # Hepatic — use baseline_v1 (lean) and longitudinal_summary (richer Claude set)
    for fname in ["baseline_v1", "longitudinal_summary"]:
        factories = [
            ("est_400", lambda seed: fit_predict_est_factory(seed, n_estimators=400, min_samples_leaf=15)),
            ("gbsa_200_lr05_d3", lambda seed: fit_predict_gbsa_factory(seed, n_estimators=200, learning_rate=0.05, max_depth=3)),
            ("cwgbs_300_lr05", lambda seed: fit_predict_cwgbs_factory(seed, n_estimators=300, learning_rate=0.05)),
        ]
        run_endpoint(L, A, "hep", fname, factories, rows)

    # Death — use longitudinal_summary (richer Claude) + baseline
    for fname in ["longitudinal_summary", "baseline_v1"]:
        factories = [
            ("est_400", lambda seed: fit_predict_est_factory(seed, n_estimators=400, min_samples_leaf=15)),
            ("gbsa_200_lr05_d3", lambda seed: fit_predict_gbsa_factory(seed, n_estimators=200, learning_rate=0.05, max_depth=3)),
            ("cwgbs_300_lr05", lambda seed: fit_predict_cwgbs_factory(seed, n_estimators=300, learning_rate=0.05)),
        ]
        run_endpoint(L, A, "dea", fname, factories, rows)

    # Rebuild summary by reading every survtree OOF/test on disk
    from zoo_utils import PRED_DIR
    final_rows = []
    for fname in sorted((PRED_DIR).glob("oof__survtree__*.csv")):
        name = fname.name.replace("oof__", "").replace(".csv", "")
        parts = name.split("__")
        if len(parts) < 4:
            continue
        kind = parts[1]; feat = parts[2]; tag = "__".join(parts[3:])
        oof_col = "hepatic_oof_risk" if kind == "hep" else "death_oof_risk"
        te_col  = "hepatic_risk" if kind == "hep" else "death_risk"
        anchor_col = "risk_hepatic_event" if kind == "hep" else "risk_death"
        ev = L["hep_event"] if kind == "hep" else L["dea_event"]
        tm = L["hep_time"] if kind == "hep" else L["dea_time"]
        oof = pd.read_csv(fname).set_index("patient_id_anon").reindex(L["pid"])[oof_col].to_numpy()
        te = pd.read_csv(PRED_DIR / fname.name.replace("oof__", "test__")).sort_values("trustii_id")[te_col].to_numpy()
        c = cindex(ev, tm, oof)
        rho_m50 = float(spearmanr(te, A["merge5050_csv"][anchor_col]).statistic)
        rho_g   = float(spearmanr(te, A["gpt_anchor_csv"][anchor_col]).statistic)
        rho_c   = float(spearmanr(te, A["cl_anchor_csv"][anchor_col]).statistic)
        final_rows.append(dict(model=name, kind=kind, feature=feat, variant=tag,
                                oof_c=float(c), rho_m50=rho_m50, rho_gpt=rho_g, rho_claude=rho_c))
    pd.DataFrame(final_rows).to_csv(LOGS / "task3_tree_survival_summary.csv", index=False)
    print("\nDone — summary at logs/task3_tree_survival_summary.csv", flush=True)


if __name__ == "__main__":
    main()
