"""Task 5 — Gated/conditional ensemble.

Hypothesis: GPT and Claude tracks may be best for different sub-populations.
Build a learned mixer over rank-transformed component OOFs:

  risk = sigmoid(g(x)) * gpt_risk + (1 - sigmoid(g(x))) * claude_risk

where g is a small (4-feature) logistic gate trained on OOF rank residuals
to predict the event indicator. Compare gated risk against the static
50/50 merge.

Also explore non-learned conditional rules (e.g., "use Claude for low-LS,
GPT for high-LS"); these are zero-parameter and so cannot overfit.

All blends are rank-then-mix.  No public-LB tuning of weights.

Outputs:
  predictions/{oof,test}__gate_*.csv
  logs/task5_gated_ensemble_summary.csv
  reports/task5_gated_ensemble.md  (after eval)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegressionCV

warnings.filterwarnings("ignore")

# Path resolution for the assembled repo (see lib/zoo_utils.py).
HERE = Path(__file__).resolve().parent
RUNNABLE = HERE.parent
sys.path.insert(0, str(RUNNABLE / "lib"))

from zoo_utils import load_labels, load_anchors, load_oof_baselines, save_pred  # noqa: E402
from claude_src.cv import cindex, repeated_stratified_folds  # noqa: E402

ROOT = RUNNABLE / "cached_intermediates"
LOGS = RUNNABLE / "_outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def get_gate_features(train, test):
    """Tiny set of robust, current-state features for the gate."""
    def latest(df, var, max_v=22):
        cols = [f"{var}_v{i}" for i in range(1, max_v + 1) if f"{var}_v{i}" in df.columns]
        if not cols: return np.full(len(df), np.nan)
        arr = df[cols].to_numpy()
        out = np.full(len(df), np.nan)
        for i in range(len(df)):
            v = arr[i]; idx = np.where(~np.isnan(v))[0]
            if len(idx): out[i] = v[idx[-1]]
        return out

    feats = {}
    feats["age_last"] = latest(train, "Age"); feats_te = {"age_last": latest(test, "Age")}
    feats["ls_last"] = latest(train, "fibs_stiffness_med_BM_1"); feats_te["ls_last"] = latest(test, "fibs_stiffness_med_BM_1")
    feats["plt_last"] = latest(train, "plt"); feats_te["plt_last"] = latest(test, "plt")
    feats["bmi_last"] = latest(train, "BMI"); feats_te["bmi_last"] = latest(test, "BMI")
    Xtr = pd.DataFrame(feats).fillna(0.0)
    Xte = pd.DataFrame(feats_te).fillna(0.0)
    return Xtr, Xte


def cv_gate_blend(gpt_oof, cl_oof, gpt_test, cl_test, event, time, X_gate_tr, X_gate_te,
                  n_splits=5, n_repeats=10, base_seed=42):
    """For each fold: fit logistic gate on training-fold features predicting event,
    apply on validation fold to mix gpt_oof vs cl_oof. For test: average gate per
    repeat (trained on full train) and mix gpt_test/cl_test."""
    n_tr = len(gpt_oof); n_te = len(gpt_test)
    g_oof = np.zeros(n_tr); g_te_acc = np.zeros(n_te)
    repeat_oof = [np.full(n_tr, np.nan) for _ in range(n_repeats)]

    for r, f, tr_idx, va_idx in repeated_stratified_folds(event, n_splits, n_repeats, base_seed):
        clf = LogisticRegressionCV(Cs=5, cv=3, max_iter=1000, n_jobs=1, scoring="roc_auc")
        clf.fit(X_gate_tr.iloc[tr_idx].values, event[tr_idx].astype(int))
        p = clf.predict_proba(X_gate_tr.iloc[va_idx].values)[:, 1]
        # Higher p (event-likely) -> trust GPT? trust Claude? we centre at OOF mean
        # Use additive mix: rank(gpt) * w + rank(cl) * (1-w), where
        # w = 0.5 + (p - p.mean()) (clipped) -- "if event-likely, lean toward
        # whichever track has higher rank-correlation with eventness on this fold"
        # But we can't tune that without circularity, so use w=p directly: high
        # event-prob -> lean to whichever was dominant in past task (start neutral).
        w = np.clip(p, 0.1, 0.9)  # lean GPT for high p
        gpt_r = rankdata(gpt_oof[va_idx]); cl_r = rankdata(cl_oof[va_idx])
        repeat_oof[r][va_idx] = w * gpt_r + (1 - w) * cl_r

    for r in range(n_repeats):
        # Train on full data per repeat for test-time gate
        clf = LogisticRegressionCV(Cs=5, cv=3, max_iter=1000, n_jobs=1, scoring="roc_auc")
        clf.fit(X_gate_tr.values, event.astype(int))
        p_te = clf.predict_proba(X_gate_te.values)[:, 1]
        w_te = np.clip(p_te, 0.1, 0.9)
        g_te_acc += w_te * rankdata(gpt_test) + (1 - w_te) * rankdata(cl_test)

    # Average repeats
    oof_acc = np.zeros(n_tr)
    for r in range(n_repeats):
        oof_acc += rankdata(repeat_oof[r])
    return oof_acc / n_repeats, g_te_acc / n_repeats


def conditional_rule_blend(gpt_oof, cl_oof, gpt_test, cl_test,
                            train, test, var="fibs_stiffness_med_BM_1", thr=12.0):
    """Hard rule: where var_last >= thr, use Claude (specialised for fibrosis tier);
    elsewhere GPT. Zero parameters trained on labels."""
    def latest(df, var, max_v=22):
        cols = [f"{var}_v{i}" for i in range(1, max_v + 1) if f"{var}_v{i}" in df.columns]
        arr = df[cols].to_numpy() if cols else np.zeros((len(df), 0))
        out = np.full(len(df), np.nan)
        for i in range(len(df)):
            v = arr[i]; idx = np.where(~np.isnan(v))[0]
            if len(idx): out[i] = v[idx[-1]]
        return out
    var_tr = latest(train, var); var_te = latest(test, var)
    # NaN -> use 50/50
    use_cl_tr = (var_tr >= thr).astype(float)  # 1 -> Claude
    use_cl_te = (var_te >= thr).astype(float)
    # NaN policy: 0.5
    use_cl_tr = np.where(np.isnan(var_tr), 0.5, use_cl_tr)
    use_cl_te = np.where(np.isnan(var_te), 0.5, use_cl_te)
    oof = use_cl_tr * rankdata(cl_oof) + (1 - use_cl_tr) * rankdata(gpt_oof)
    te  = use_cl_te * rankdata(cl_test) + (1 - use_cl_te) * rankdata(gpt_test)
    return oof, te


def main():
    L = load_labels()
    A = load_anchors()
    OOF = load_oof_baselines(L["pid"])

    # Test-side anchors
    gpt_hep_test = A["gpt_anchor_csv"]["risk_hepatic_event"].to_numpy()
    cl_hep_test  = A["cl_anchor_csv"]["risk_hepatic_event"].to_numpy()
    gpt_dea_test = A["gpt_anchor_csv"]["risk_death"].to_numpy()
    cl_dea_test  = A["cl_anchor_csv"]["risk_death"].to_numpy()

    Xg_tr, Xg_te = get_gate_features(L["train"], L["test"])

    rows = []

    # 1) Learned logistic gate — hep
    for kind, gpt_oof, cl_oof, gpt_test, cl_test, event, time in [
        ("hep", OOF["gpt_hep"], OOF["cl_hep"], gpt_hep_test, cl_hep_test, L["hep_event"], L["hep_time"]),
        ("dea", OOF["gpt_dea"], OOF["cl_dea"], gpt_dea_test, cl_dea_test, L["dea_event"], L["dea_time"]),
    ]:
        oof_g, te_g = cv_gate_blend(gpt_oof, cl_oof, gpt_test, cl_test, event, time, Xg_tr, Xg_te)
        c = cindex(event, time, oof_g)
        anchor_col = "risk_hepatic_event" if kind == "hep" else "risk_death"
        rho_m50 = float(spearmanr(te_g, A["merge5050_csv"][anchor_col]).statistic)
        baseline = OOF["merge_hep"] if kind == "hep" else OOF["merge_dea"]
        c_50 = cindex(event, time, baseline)
        save_pred(f"gate_logreg__{kind}", L["pid"], oof_g, L["tid"], te_g, kind=kind)
        rows.append(dict(method="gate_logreg", kind=kind, oof_c=float(c),
                         oof_50_50=float(c_50), delta=float(c - c_50), rho_m50=rho_m50,
                         description="Logistic gate over (age,LS,plt,BMI) -> w*gpt + (1-w)*cl"))
        print(f"  gate_logreg/{kind}: OOF={c:.4f}  vs 50/50={c_50:.4f}  delta={c-c_50:+.4f}  rho_m50={rho_m50:.3f}")

    # 2) Conditional rule (LS-based)
    for thr in [10.0, 12.0, 15.0]:
        for kind, gpt_oof, cl_oof, gpt_test, cl_test, event, time in [
            ("hep", OOF["gpt_hep"], OOF["cl_hep"], gpt_hep_test, cl_hep_test, L["hep_event"], L["hep_time"]),
        ]:
            oof_g, te_g = conditional_rule_blend(gpt_oof, cl_oof, gpt_test, cl_test,
                                                  L["train"], L["test"], thr=thr)
            c = cindex(event, time, oof_g)
            anchor_col = "risk_hepatic_event"
            rho_m50 = float(spearmanr(te_g, A["merge5050_csv"][anchor_col]).statistic)
            baseline = OOF["merge_hep"]
            c_50 = cindex(event, time, baseline)
            save_pred(f"gate_LSthr{int(thr)}__{kind}", L["pid"], oof_g, L["tid"], te_g, kind=kind)
            rows.append(dict(method=f"gate_LSthr{thr:.0f}", kind=kind, oof_c=float(c),
                             oof_50_50=float(c_50), delta=float(c - c_50), rho_m50=rho_m50,
                             description=f"if LS_last>={thr}: Claude else GPT"))
            print(f"  gate_LSthr{thr:.0f}/{kind}: OOF={c:.4f}  vs 50/50={c_50:.4f}  delta={c-c_50:+.4f}")

    # 3) Conditional rule on platelet (proxy for advanced disease)
    for kind in ["hep"]:
        gpt_oof, cl_oof, gpt_test, cl_test = (OOF["gpt_hep"], OOF["cl_hep"], gpt_hep_test, cl_hep_test) if kind=="hep" else (OOF["gpt_dea"], OOF["cl_dea"], gpt_dea_test, cl_dea_test)
        event, time = (L["hep_event"], L["hep_time"]) if kind=="hep" else (L["dea_event"], L["dea_time"])
        oof_g, te_g = conditional_rule_blend(gpt_oof, cl_oof, gpt_test, cl_test,
                                              L["train"], L["test"], var="plt", thr=130.0)
        c = cindex(event, time, oof_g)
        anchor_col = "risk_hepatic_event" if kind == "hep" else "risk_death"
        rho_m50 = float(spearmanr(te_g, A["merge5050_csv"][anchor_col]).statistic)
        baseline = OOF["merge_hep"] if kind == "hep" else OOF["merge_dea"]
        c_50 = cindex(event, time, baseline)
        save_pred(f"gate_PLTthr130__{kind}", L["pid"], oof_g, L["tid"], te_g, kind=kind)
        rows.append(dict(method="gate_PLTthr130", kind=kind, oof_c=float(c),
                         oof_50_50=float(c_50), delta=float(c - c_50), rho_m50=rho_m50,
                         description="if PLT<130: Claude else GPT (low-PLT rule inverted: high disease -> Claude)"))
        print(f"  gate_PLTthr130/{kind}: OOF={c:.4f}  vs 50/50={c_50:.4f}  delta={c-c_50:+.4f}")

    pd.DataFrame(rows).to_csv(LOGS / "task5_gated_ensemble_summary.csv", index=False)
    print("\nDone — summary at logs/task5_gated_ensemble_summary.csv")


if __name__ == "__main__":
    main()
