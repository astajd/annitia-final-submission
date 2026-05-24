"""Experiment 1.5: verify nit_only honesty by comparing with nit_only_baseline_only.

Phase 2 Experiment 1 surfaced RSF on `nit_only` at 0.822 ± 0.081 hepatic. The
question is whether that signal is honest (NIT *values* at v1) or pumped by
post-event NIT measurements that the trajectory features include. The clean
comparator is `nit_only_baseline_only`: NIT v1 values + Age_v1 + presence
flags, no trajectory features.

Runs 3 hepatic models (coxnet, rsf, xgb_cox) at 5×10 CV. Appends rows to
`reports/phase2_cv_results.csv` with the same schema as the rest of the grid.
Idempotent: skips cells already in the CSV.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd

from src.config import REPORTS
from src.cv import evaluate_cv, summarize
from src.data import load_raw, build_targets
from src.features import build_features, FEATURE_SET_RISK
from src.models import make_coxnet, make_rsf, make_xgb_cox

OUT_CSV = REPORTS / "phase2_cv_results.csv"
N_SPLITS = 5
N_REPEATS = 10
FS = "nit_only_baseline_only"

MODELS = {
    "coxnet":  lambda: make_coxnet(l1_ratio=0.5),
    "rsf":     lambda: make_rsf(n_estimators=150),
    "xgb_cox": lambda: make_xgb_cox(n_estimators=200, learning_rate=0.05),
}


def already_done(df_done, endpoint, dm, fs, mn):
    if df_done is None or df_done.empty:
        return False
    sel = ((df_done["endpoint"] == endpoint) &
           (df_done["death_mode"].fillna("n/a") == (dm or "n/a")) &
           (df_done["feature_set"] == fs) &
           (df_done["model"] == mn))
    return sel.any()


def append_row(row, csv: Path):
    csv.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if csv.exists():
        df_row.to_csv(csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv, mode="w", header=True, index=False)


def main():
    train, _ = load_raw()
    hep_t, _ = build_targets(train, "drop_missing_death")

    valid = hep_t["hepatic_valid"].to_numpy()
    e = hep_t.loc[valid, "hepatic_event"].to_numpy().astype(bool)
    t = hep_t.loc[valid, "hepatic_time"].to_numpy().astype(float)
    df = train.loc[valid].reset_index(drop=True)

    X = build_features(df, FS)
    print(f"{FS}: {X.shape[1]} features ({list(X.columns)})")
    print(f"n_rows={len(e)}  n_events={int(e.sum())}\n")

    df_done = pd.read_csv(OUT_CSV) if OUT_CSV.exists() else None

    for mn, mf in MODELS.items():
        if already_done(df_done, "hepatic", "n/a", FS, mn):
            print(f"[skip] {FS}/{mn} already in CSV")
            continue
        t0 = time.time()
        try:
            cv_res = evaluate_cv(mf(), X, e, t,
                                 n_splits=N_SPLITS, n_repeats=N_REPEATS)
            s = summarize(cv_res)
            el = time.time() - t0
            print(f"[H] {FS:30s} {mn:12s}  "
                  f"C={s['mean']:.3f}±{s['std']:.3f} m−s={s['mean']-s['std']:.3f}"
                  f"  ({el:.0f}s)")
            append_row({
                "endpoint": "hepatic", "death_mode": "n/a",
                "feature_set": FS,
                "leakage_risk": FEATURE_SET_RISK[FS],
                "model": mn, "n_rows": len(e), "n_events": int(e.sum()),
                "n_features": X.shape[1], **s,
                "elapsed_s": round(el, 1),
            }, OUT_CSV)
        except Exception as ex:
            print(f"[H] {FS}/{mn} FAIL: {ex!r}")


if __name__ == "__main__":
    main()
