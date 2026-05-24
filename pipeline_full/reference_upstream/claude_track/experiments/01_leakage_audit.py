"""Leakage audit: does any single proxy feature alone give a high C-index?"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.config import REPORTS, LONGITUDINAL_VARS, NIT_VARS, MAX_VISITS
from src.data import load_raw, build_targets, add_visit_metadata
from src.cv import cindex, evaluate_cv, summarize
from src.features import build_features
from src.models import make_coxnet


# Existing meta proxies (Phase 1) + new missingness/cadence candidates (Phase 2)
META_FEATURES = ["min_age", "max_age", "n_visits", "followup_yrs", "Age_v1"]
EXTRA_FEATURES = [
    "missingness_count",
    "missingness_count_NIT",
    "time_since_last_NIT",
    "n_NIT_measurements",
    "n_visits_FibroScan",
    "n_visits_FibroTest",
    "n_visits_Aixplorer",
]
NIT_NAME_MAP = {
    "fibs_stiffness_med_BM_1": "n_visits_FibroScan",
    "fibrotest_BM_2": "n_visits_FibroTest",
    "aixp_aix_result_BM_3": "n_visits_Aixplorer",
}


def _existing_visit_cols(df: pd.DataFrame, var: str) -> list[str]:
    return [c for c in (f"{var}_v{i}" for i in range(1, MAX_VISITS + 1)) if c in df.columns]


def compute_audit_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Patient-level missingness / measurement-cadence features for the audit.

    All derived from raw `<var>_v<i>` columns. Tier-3-or-worse candidates: they
    encode follow-up length and clinical-monitoring intensity, both potentially
    correlated with outcome timing.
    """
    out = pd.DataFrame(index=df.index)

    long_cols = [c for v in LONGITUDINAL_VARS for c in _existing_visit_cols(df, v)]
    out["missingness_count"] = df[long_cols].isna().sum(axis=1).astype(float)

    nit_cols = [c for v in NIT_VARS for c in _existing_visit_cols(df, v)]
    out["missingness_count_NIT"] = df[nit_cols].isna().sum(axis=1).astype(float)

    # Per-visit NIT presence (any of the 3 NITs non-NaN at that visit)
    n_max = MAX_VISITS
    nit_present = np.zeros((len(df), n_max), dtype=bool)
    for v in NIT_VARS:
        for c in _existing_visit_cols(df, v):
            i = int(c.split("_v")[-1]) - 1
            nit_present[:, i] |= df[c].notna().to_numpy()
    out["n_NIT_measurements"] = nit_present.sum(axis=1).astype(float)

    # time_since_last_NIT = max(Age over visits with any NIT) − Age_v1
    age_arr = np.full((len(df), n_max), np.nan)
    for c in _existing_visit_cols(df, "Age"):
        i = int(c.split("_v")[-1]) - 1
        age_arr[:, i] = df[c].to_numpy(dtype=float)
    nit_age = np.where(nit_present, age_arr, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        all_nan = np.all(np.isnan(nit_age), axis=1)
        last_nit_age = np.where(all_nan, np.nan, np.nanmax(nit_age, axis=1))
    out["time_since_last_NIT"] = last_nit_age - df["Age_v1"].to_numpy(dtype=float)

    for v, name in NIT_NAME_MAP.items():
        cs = _existing_visit_cols(df, v)
        out[name] = (df[cs].notna().sum(axis=1).astype(float)
                     if cs else pd.Series(0.0, index=df.index))

    return out


def single_feature_cindex(arr, feature, event, time):
    vals = arr[feature].fillna(arr[feature].median()).to_numpy()
    return {
        "positive_sign_cindex": cindex(event, time, vals),
        "negative_sign_cindex": cindex(event, time, -vals),
    }


def audit_features(df_meta: pd.DataFrame, e: np.ndarray, t: np.ndarray,
                   feature_names: list[str], label: str) -> dict:
    """Run single-feature C-index audit for all `feature_names` against (e, t)."""
    out = {}
    for f in feature_names:
        if f not in df_meta.columns:
            print(f"  {f:24s}: MISSING — skipped")
            continue
        r = single_feature_cindex(df_meta, f, e, t)
        out[f] = r
        print(f"  {f:24s}: pos={r['positive_sign_cindex']:.3f}  "
              f"neg={r['negative_sign_cindex']:.3f}")
    return out


def _prepare_audit_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Combine visit metadata with the new missingness/cadence features."""
    df_meta = add_visit_metadata(df)
    extras = compute_audit_extra_features(df)
    extras.index = df_meta.index
    return pd.concat([df_meta, extras], axis=1)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    train, _ = load_raw()

    hep_t, _ = build_targets(train, "drop_missing_death")
    _, death_drop = build_targets(train, "drop_missing_death")
    _, death_cens = build_targets(train, "censor_missing_death_at_last")

    audit = {}
    all_features = META_FEATURES + EXTRA_FEATURES

    # Hepatic
    print("=== HEPATIC ===")
    valid = hep_t["hepatic_valid"].to_numpy()
    e = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)
    df_aud = _prepare_audit_frame(df)

    single = audit_features(df_aud, e, t, all_features, "hepatic")

    # Proxy-only CV
    X_proxy = build_features(df, "followup_proxy_only")
    res = evaluate_cv(make_coxnet(), X_proxy, e, t, n_splits=5, n_repeats=3)
    s = summarize(res)
    print(f"  proxy-only Coxnet: {s['mean']:.3f} ± {s['std']:.3f}")

    # Post-event visits
    he = df[e]
    age_cols = [c for c in he.columns if c.startswith("Age_v")]
    event_age = he["evenements_hepatiques_age_occur"].to_numpy()
    visit_ages = he[age_cols].to_numpy(dtype=float)
    n_post = (visit_ages > event_age[:, None]).sum(axis=1)

    audit["hepatic"] = {
        "n_rows": int(len(df)), "n_events": int(e.sum()),
        "single_features": single,
        "followup_proxy_only_cv": s,
        "post_event_visits": {
            "n_event_patients": int(len(he)),
            "n_with_post_event_visits": int((n_post > 0).sum()),
            "fraction_with_post_event": float((n_post > 0).mean()),
            "avg_post_event_visits": float(n_post[n_post > 0].mean()) if (n_post > 0).any() else 0.0,
        },
    }

    # Death
    print("\n=== DEATH ===")
    out_death = {}
    for mode_name, dt in [("drop_missing_death", death_drop),
                          ("censor_missing_death_at_last", death_cens)]:
        valid = dt["death_valid"].to_numpy()
        e = dt.loc[valid, "death_event"].to_numpy().astype(bool)
        t = dt.loc[valid, "death_time"].to_numpy().astype(float)
        df_d = train.loc[valid].reset_index(drop=True)
        df_aud = _prepare_audit_frame(df_d)
        print(f"\nmode={mode_name}  n={len(df_d)}  events={int(e.sum())}")
        single_d = audit_features(df_aud, e, t, all_features, f"death/{mode_name}")
        X_proxy_d = build_features(df_d, "followup_proxy_only")
        res = evaluate_cv(make_coxnet(), X_proxy_d, e, t, n_splits=5, n_repeats=3)
        sd = summarize(res)
        print(f"  proxy-only Coxnet: {sd['mean']:.3f} ± {sd['std']:.3f}")
        out_death[mode_name] = {
            "n_rows": int(len(df_d)), "n_events": int(e.sum()),
            "single_features": single_d, "followup_proxy_only_cv": sd,
        }
    audit["death"] = out_death

    # Strict-feature-set audit (NEWLY DISCOVERED LEAKAGE)
    print("\n=== STRICT FEATURE LEAKAGE ===")
    valid = hep_t["hepatic_valid"].to_numpy()
    e = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)
    Xs = build_features(df, "strict_time_aligned", event=e, time=t)
    sig = []
    for col in Xs.columns:
        v = Xs[col].fillna(Xs[col].median()).to_numpy()
        if np.std(v) < 1e-9: continue
        c = cindex(e, t, v); cn = cindex(e, t, -v)
        sig.append((col, max(c, cn)))
    sig.sort(key=lambda x: -x[1])
    print("Top-10 single-feature C-index in strict_time_aligned:")
    top10 = []
    for col, ci in sig[:10]:
        print(f"  {col:50s}: {ci:.3f}")
        top10.append({"feature": col, "best_cindex": ci})
    audit["strict_features_top10"] = top10

    out = REPORTS / "leakage_audit.json"
    out.write_text(json.dumps(audit, indent=2))
    print(f"\nSaved {out}")
    return audit


if __name__ == "__main__":
    main()
