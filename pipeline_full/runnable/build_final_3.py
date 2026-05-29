"""Build the final 3 probe submissions, metadata, and per-probe diagnostics.

Adapted from the original `merged/final_probe_sprint/scripts/build_final_3.py`.
The only behavioural change vs the original is path resolution: this copy
imports zoo utilities from `lib/`, reads cached intermediates from
`cached_intermediates/`, and writes outputs to `_outputs/`. The hyperparameters,
gate direction, disagreement override, and dea blend weights are byte-identical
to the original.

This script is the IMMEDIATE PRODUCER of the submitted slot1 file from
cached model-level prediction CSVs. Raw challenge data are required only for
label and LS-variable derivation via `lib/zoo_utils.load_labels()`.

Final 3 probes (slot1 is finalprobe_3):
  1. finalprobe_1_lsthr18_disagq95_cw85_gbsa15  (MAX OOF, Δw +0.0035)
  2. finalprobe_2_lsthr18_cw85_gbsa15            (no disag, Δw +0.0017)
  3. finalprobe_3_lsthr12_disagq95_cw85_gbsa15   (LS=12 SLOT1, Δw +0.0023)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
RUNNABLE = HERE
LIB = RUNNABLE / "lib"
CACHED = RUNNABLE / "cached_intermediates"
ZOO_PRED = CACHED / "model_zoo_sprint" / "predictions"
OUT_DIR = RUNNABLE / "_outputs"
SUB = OUT_DIR / "submissions"
META = OUT_DIR / "metadata"
LOGS = OUT_DIR / "logs"
for d in (SUB, META, LOGS):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(LIB))

from zoo_utils import load_labels, load_anchors, load_oof_baselines  # noqa: E402
from claude_src.cv import cindex  # noqa: E402


def latest(df, var, max_v=22):
    cols = [f"{var}_v{i}" for i in range(1, max_v + 1) if f"{var}_v{i}" in df.columns]
    arr = df[cols].to_numpy() if cols else np.zeros((len(df), 0))
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        v = arr[i]
        idx = np.where(~np.isnan(v))[0]
        if len(idx):
            out[i] = v[idx[-1]]
    return out


def _load_oof(name, col, pid):
    return pd.read_csv(ZOO_PRED / name).set_index("patient_id_anon").reindex(pid)[col].to_numpy()


def _load_test(name, col, tid):
    return pd.read_csv(ZOO_PRED / name).set_index("trustii_id").reindex(tid)[col].to_numpy()


def w_oof(c_h, c_d):
    return 0.7 * c_h + 0.3 * c_d


def gate_lsthr(thr, ls_arr, gpt_h, cl_h):
    use_cl = (ls_arr >= thr).astype(float)
    return use_cl * rankdata(cl_h) + (1 - use_cl) * rankdata(gpt_h)


def disag_override(base_h, gpt_h, cl_h, m50_h, q):
    """Where |rank(GPT) - rank(Cl)| > q-quantile, replace with rank(m50); else rank(base_h)."""
    g_r = rankdata(gpt_h); c_r = rankdata(cl_h)
    d = np.abs(g_r - c_r) / max(g_r.size, 1)
    tau = float(np.quantile(d, q))
    return np.where(d > tau, rankdata(m50_h), rankdata(base_h)), tau, int((d > tau).sum())


def main():
    L = load_labels()
    A = load_anchors()
    OOF = load_oof_baselines(L["pid"])
    pid = L["pid"]; tid = A["gpt_anchor_csv"]["trustii_id"].to_numpy()

    gpt_hep_te = A["gpt_anchor_csv"]["risk_hepatic_event"].to_numpy()
    cl_hep_te = A["cl_anchor_csv"]["risk_hepatic_event"].to_numpy()
    m50_hep_te = A["merge5050_csv"]["risk_hepatic_event"].to_numpy()
    m50_dea_te = A["merge5050_csv"]["risk_death"].to_numpy()

    cwgbs_oof = _load_oof("oof__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv",
                          "death_oof_risk", pid)
    cwgbs_te = _load_test("test__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv",
                          "death_risk", tid)
    g12_hep_oof = _load_oof("oof__gate_LSthr12__hep.csv", "hepatic_oof_risk", pid)
    g12_hep_te = _load_test("test__gate_LSthr12__hep.csv", "hepatic_risk", tid)
    gbsa_oof = _load_oof("oof__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv",
                         "death_oof_risk", pid)
    gbsa_te = _load_test("test__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv",
                         "death_risk", tid)

    ls_tr = latest(L["train"], "fibs_stiffness_med_BM_1")
    ls_te = latest(L["test"], "fibs_stiffness_med_BM_1")

    cur_w = w_oof(cindex(L["hep_event"], L["hep_time"], g12_hep_oof),
                  cindex(L["dea_event"], L["dea_time"], cwgbs_oof))

    dea_blend_oof = 0.85 * rankdata(cwgbs_oof) + 0.15 * rankdata(gbsa_oof)
    dea_blend_te = 0.85 * rankdata(cwgbs_te) + 0.15 * rankdata(gbsa_te)

    candidates = []

    h18_oof = gate_lsthr(18, ls_tr, OOF["gpt_hep"], OOF["cl_hep"])
    h18_te = gate_lsthr(18, ls_te, gpt_hep_te, cl_hep_te)
    p1_h_oof, tau_oof_p1, n_over_oof_p1 = disag_override(
        h18_oof, OOF["gpt_hep"], OOF["cl_hep"], OOF["merge_hep"], 0.95)
    p1_h_te, tau_te_p1, n_over_te_p1 = disag_override(
        h18_te, gpt_hep_te, cl_hep_te, m50_hep_te, 0.95)
    candidates.append(dict(
        name="finalprobe_1_lsthr18_disagq95_cw85gbsa15",
        hep_oof=p1_h_oof, hep_te=p1_h_te,
        dea_oof=dea_blend_oof, dea_te=dea_blend_te,
        meta=dict(
            hep_strategy="LS gate at LS_last>=18 kPa (Claude/GPT) + top-5%-disagreement override → merge_50_50 hep",
            dea_strategy="0.85 * rank(CWGBSA dea) + 0.15 * rank(GBSA dea), both on Claude longitudinal_summary features",
            ls_threshold_kPa=18,
            disag_quantile=0.95,
            disag_n_overridden_train=n_over_oof_p1,
            disag_n_overridden_test=n_over_te_p1,
            dea_blend_weight_cwgbs=0.85,
            dea_blend_weight_gbsa=0.15,
            priority=1,
            qualitative_risk="moderate",
            rationale=("LS-threshold + disagreement override + dea-blend, all OOF-grid selected. "
                       "Δw_oof +0.0035 vs current best."),
        ),
    ))

    p2_h_oof = h18_oof.copy()
    p2_h_te = h18_te.copy()
    candidates.append(dict(
        name="finalprobe_2_lsthr18_cw85gbsa15",
        hep_oof=p2_h_oof, hep_te=p2_h_te,
        dea_oof=dea_blend_oof, dea_te=dea_blend_te,
        meta=dict(
            hep_strategy="LS gate at LS_last>=18 kPa (Claude/GPT). No disagreement override.",
            dea_strategy="0.85 * rank(CWGBSA dea) + 0.15 * rank(GBSA dea)",
            ls_threshold_kPa=18,
            disag_quantile=None,
            dea_blend_weight_cwgbs=0.85,
            dea_blend_weight_gbsa=0.15,
            priority=2,
            qualitative_risk="low–moderate",
            rationale=("Isolates LS-threshold + dea-blend without disagreement override. Δw_oof +0.0017."),
        ),
    ))

    p3_h_oof, tau_oof_p3, n_over_oof_p3 = disag_override(
        g12_hep_oof, OOF["gpt_hep"], OOF["cl_hep"], OOF["merge_hep"], 0.95)
    p3_h_te, tau_te_p3, n_over_te_p3 = disag_override(
        g12_hep_te, gpt_hep_te, cl_hep_te, m50_hep_te, 0.95)
    candidates.append(dict(
        name="finalprobe_3_lsthr12_disagq95_cw85gbsa15",
        hep_oof=p3_h_oof, hep_te=p3_h_te,
        dea_oof=dea_blend_oof, dea_te=dea_blend_te,
        meta=dict(
            hep_strategy="LS gate at LS_last>=12 kPa (Baveno VII rule-in for cACLD; Claude/GPT) + top-5%-disagreement override → merge_50_50 hep",
            dea_strategy="0.85 * rank(CWGBSA dea) + 0.15 * rank(GBSA dea)",
            ls_threshold_kPa=12,
            disag_quantile=0.95,
            disag_n_overridden_train=n_over_oof_p3,
            disag_n_overridden_test=n_over_te_p3,
            dea_blend_weight_cwgbs=0.85,
            dea_blend_weight_gbsa=0.15,
            priority=3,
            qualitative_risk="low",
            rationale=("Preserves Baveno VII 12 kPa rule-in cutoff for the LS gate, with disagreement-override and dea-blend gains. "
                       "Δw_oof +0.0023. SLOT1 SUBMITTED."),
        ),
    ))

    rows = []
    for c in candidates:
        c_h = cindex(L["hep_event"], L["hep_time"], c["hep_oof"])
        c_d = cindex(L["dea_event"], L["dea_time"], c["dea_oof"])
        wt = w_oof(c_h, c_d)

        rh_cur = float(spearmanr(c["hep_te"], g12_hep_te).statistic)
        rd_cur = float(spearmanr(c["dea_te"], cwgbs_te).statistic)
        rh_m50 = float(spearmanr(c["hep_te"], m50_hep_te).statistic)
        rd_m50 = float(spearmanr(c["dea_te"], m50_dea_te).statistic)

        n_te = len(tid)
        cur_h_rank = rankdata(g12_hep_te)
        cur_d_rank = rankdata(cwgbs_te)
        new_h_rank = rankdata(c["hep_te"])
        new_d_rank = rankdata(c["dea_te"])
        h_shift = np.abs(new_h_rank - cur_h_rank) / n_te
        d_shift = np.abs(new_d_rank - cur_d_rank) / n_te

        sub_path = SUB / f"{c['name']}.csv"
        pd.DataFrame({
            "trustii_id": tid,
            "risk_hepatic_event": rankdata(c["hep_te"]),
            "risk_death": rankdata(c["dea_te"]),
        }).to_csv(sub_path, index=False)

        meta = c["meta"].copy()
        meta.update(dict(
            candidate_name=c["name"],
            submission_csv=str(sub_path.relative_to(RUNNABLE)),
            source_components=dict(
                hep_anchor_gpt="cached_intermediates/gpt_track_handoff/best_submissions/20260428_0059_phase3_10_horizon_blend_v2.csv",
                hep_anchor_claude="cached_intermediates/claude_track_handoff/best_submissions/phase2_blend_2way_optimal.csv",
                hep_merge50_50="cached_intermediates/merge_sprint/submissions/merge_A_best_50_50_both.csv",
                dea_cwgbs_oof="cached_intermediates/model_zoo_sprint/predictions/oof__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv",
                dea_cwgbs_test="cached_intermediates/model_zoo_sprint/predictions/test__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv",
                dea_gbsa_oof="cached_intermediates/model_zoo_sprint/predictions/oof__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv",
                dea_gbsa_test="cached_intermediates/model_zoo_sprint/predictions/test__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv",
                ls_variable="fibs_stiffness_med_BM_1_v* (last non-NaN)",
            ),
            model_families=["CWGBSA (sksurv) for dea", "GBSA (sksurv) for dea", "rank-mix gates for hep"],
            feature_sets=["Claude longitudinal_summary (143 cols, dea models)", "GPT and Claude anchor test predictions (hep)"],
            hyperparameters=dict(
                cwgbs=dict(loss="coxph", n_estimators=300, learning_rate=0.05, subsample=0.8),
                gbsa=dict(loss="coxph", n_estimators=200, learning_rate=0.05, max_depth=3,
                          subsample=0.8, max_features="sqrt"),
            ),
            seeds=dict(
                cwgbs="42 + repeat (10 repeats, mean-of-ranks)",
                gbsa="42 + repeat (10 repeats, mean-of-ranks)",
            ),
            oof_hepatic_C=float(c_h),
            oof_death_C=float(c_d),
            weighted_oof_C=float(wt),
            delta_w_vs_current_best=float(wt - cur_w),
            rho_hep_vs_current_best=rh_cur,
            rho_death_vs_current_best=rd_cur,
            rho_hep_vs_merge_50_50=rh_m50,
            rho_death_vs_merge_50_50=rd_m50,
            mean_rank_shift_hep_vs_current_best=float(h_shift.mean()),
            max_rank_shift_hep_vs_current_best=float(h_shift.max()),
            mean_rank_shift_dea_vs_current_best=float(d_shift.mean()),
            max_rank_shift_dea_vs_current_best=float(d_shift.max()),
            target_derived_features_used=False,
            strict_time_aligned_used=False,
            public_LB_tuned_weights=False,
            reproducibility_status=("Regenerated rank-identically from cached intermediates by this script. "
                                    "Full from-raw retraining is provided by retrain_all_from_raw.sh under the pinned environment. "
                                    "See docs/PROVENANCE.md."),
        ))
        meta_path = META / f"{c['name']}.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        rows.append(dict(
            name=c["name"],
            hep_oof=c_h, dea_oof=c_d, weighted_oof=wt,
            delta_w=wt - cur_w,
            rho_hep_cur=rh_cur, rho_dea_cur=rd_cur,
            rho_hep_m50=rh_m50, rho_dea_m50=rd_m50,
            mean_h_shift=float(h_shift.mean()), max_h_shift=float(h_shift.max()),
            mean_d_shift=float(d_shift.mean()), max_d_shift=float(d_shift.max()),
            priority=c["meta"]["priority"],
        ))
        print(f"Built {c['name']}")
        print(f"  hep_oof={c_h:.4f}  dea_oof={c_d:.4f}  w={wt:.4f}  Δw={wt-cur_w:+.4f}")
        print(f"  ρ_hep_cur={rh_cur:.3f}  ρ_dea_cur={rd_cur:.3f}  ρ_hep_m50={rh_m50:.3f}  ρ_dea_m50={rd_m50:.3f}")
        print(f"  CSV: {sub_path}")
        print(f"  Meta: {meta_path}")
        print()

    pd.DataFrame(rows).to_csv(LOGS / "final_probe_metrics.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
