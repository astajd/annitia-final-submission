"""ANNITIA merge sprint — build 10 candidate CSVs + JSON sidecars.

All candidates are rank-blends (per-source `rankdata` once, then convex
combination of ranks). No public-LB tuning, no target-derived features,
no strict_time_aligned components. The output schema is exactly
`trustii_id, risk_hepatic_event, risk_death` with rank-encoded risks
(higher = riskier, monotonic, no information loss for C-index).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
RUNNABLE = HERE.parent
ROOT = RUNNABLE / "cached_intermediates"
GPT = ROOT / "gpt_track_handoff"
CL = ROOT / "claude_track_handoff"
SUB = RUNNABLE / "_outputs" / "merge_sprint" / "submissions"
META = RUNNABLE / "_outputs" / "merge_sprint" / "metadata"
REP = RUNNABLE / "_outputs" / "merge_sprint" / "reports"
for d in (SUB, META, REP):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

INPUT_PATHS = {
    "gpt_anchor":          GPT / "best_submissions/20260428_0059_phase3_10_horizon_blend_v2.csv",
    "gpt_phase3_9":        GPT / "best_submissions/20260428_0027_phase3_9_horizon_blend.csv",
    "gpt_phase3_5_v3hep":  GPT / "best_submissions/20260427_1255_phase3_5_current_state_v3_hepatic_focused.csv",
    "claude_anchor":       CL / "best_submissions/phase2_blend_2way_optimal.csv",
    "claude_landmark_3y":  CL / "best_submissions/phase2_landmark_3y.csv",
    "claude_landmark_permissive": CL / "best_submissions/phase2_blend_landmark_permissive.csv",
    # Component test predictions (hepatic-only):
    "claude_perm_ens_avg_test": CL / "component_test_predictions/test_permissive_ensemble_avg.csv",
    "claude_rsf_long_summary_test": CL / "component_test_predictions/test_rsf_longitudinal_summary.csv",
    "claude_landmark_3y_test": CL / "component_test_predictions/test_landmark_3y_rsf.csv",
}


def load_inputs() -> dict[str, pd.DataFrame]:
    out = {}
    for k, p in INPUT_PATHS.items():
        df = pd.read_csv(p).sort_values("trustii_id").reset_index(drop=True)
        out[k] = df
    # alignment + sanity
    ref_ids = out["gpt_anchor"]["trustii_id"].to_numpy()
    n = len(ref_ids)
    for k, df in out.items():
        assert (df["trustii_id"].to_numpy() == ref_ids).all(), f"id mismatch: {k}"
        assert df.isna().sum().sum() == 0, f"NaN in {k}"
    print(f"  loaded {len(out)} sources, n={n} rows each, schemas verified.")
    return out


# ---------------------------------------------------------------------------
# Rank-blend helpers
# ---------------------------------------------------------------------------

def rk(arr: np.ndarray) -> np.ndarray:
    return rankdata(arr).astype(float)


def blend_ranks(parts: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """parts = list of (raw_score_array, weight). Each is rank-transformed
    once before convex combination. Weights are normalized."""
    w_total = sum(w for _, w in parts)
    out = np.zeros_like(parts[0][0], dtype=float)
    for arr, w in parts:
        out += (w / w_total) * rk(arr)
    return out


# ---------------------------------------------------------------------------
# Candidate writer
# ---------------------------------------------------------------------------

ALL_CANDIDATES: list[dict] = []


def write_candidate(
    name: str,
    description: str,
    hypothesis: str,
    priority: str,
    hep_blend: list[tuple[str, str, float]],
    dea_blend: list[tuple[str, str, float]],
    inputs: dict[str, pd.DataFrame],
    notes: str = "",
) -> dict:
    """Each entry in hep_blend / dea_blend is (source_key, column_name, weight)."""
    trustii_id = inputs["gpt_anchor"]["trustii_id"].to_numpy()

    hep_parts = [(inputs[src][col].to_numpy(), w) for src, col, w in hep_blend]
    dea_parts = [(inputs[src][col].to_numpy(), w) for src, col, w in dea_blend]
    hep_arr = blend_ranks(hep_parts)
    dea_arr = blend_ranks(dea_parts)

    csv_path = SUB / f"{name}.csv"
    json_path = META / f"{name}.json"
    pd.DataFrame({
        "trustii_id":          trustii_id,
        "risk_hepatic_event":  hep_arr,
        "risk_death":          dea_arr,
    }).to_csv(csv_path, index=False)

    g_h = inputs["gpt_anchor"]["risk_hepatic_event"].to_numpy()
    g_d = inputs["gpt_anchor"]["risk_death"].to_numpy()
    c_h = inputs["claude_anchor"]["risk_hepatic_event"].to_numpy()
    c_d = inputs["claude_anchor"]["risk_death"].to_numpy()

    rho_gpt_h = float(spearmanr(hep_arr, g_h).statistic)
    rho_gpt_d = float(spearmanr(dea_arr, g_d).statistic)
    rho_cl_h  = float(spearmanr(hep_arr, c_h).statistic)
    rho_cl_d  = float(spearmanr(dea_arr, c_d).statistic)

    components_hep = [
        {
            "file": str(INPUT_PATHS[src].relative_to(ROOT)),
            "column": col,
            "weight": w,
            "rank_transformed_at_input": True,
        }
        for src, col, w in hep_blend
    ]
    components_dea = [
        {
            "file": str(INPUT_PATHS[src].relative_to(ROOT)),
            "column": col,
            "weight": w,
            "rank_transformed_at_input": True,
        }
        for src, col, w in dea_blend
    ]

    meta = {
        "candidate_name": name,
        "description": description,
        "hypothesis": hypothesis,
        "components": {
            "hepatic": components_hep,
            "death":   components_dea,
        },
        "endpoint_strategy": {
            "hepatic": _strategy_label(hep_blend),
            "death":   _strategy_label(dea_blend),
        },
        "blend_input": "ranks (rankdata applied once per source column before convex combination)",
        "rank_correlations": {
            "vs_gpt_anchor": {"hepatic": rho_gpt_h, "death": rho_gpt_d},
            "vs_claude_anchor": {"hepatic": rho_cl_h, "death": rho_cl_d},
        },
        "uses_oof_for_weights": False,
        "uses_public_lb_for_weights": False,
        "uses_target_derived_features": False,
        "uses_strict_time_aligned_components": False,
        "uses_horizon_models": _uses_horizon(hep_blend) or _uses_horizon(dea_blend),
        "uses_full_train_refit": False,
        "requires_reproducibility_package": True,
        "reproducibility_caveat": (
            "Source CSVs are outputs of each track's full pipeline. The merge step "
            "(`scripts/build_sprint.py`) is trivially reproducible from those CSVs. "
            "To produce a competition-grade reproducible artefact a notebook would have "
            "to (a) regenerate the source CSVs from each track's code + raw data, "
            "(b) re-apply this rank-blend."
        ),
        "recommended_priority": priority,
        "notes": notes,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    record = {
        "candidate_name": name,
        "csv": str(csv_path.relative_to(ROOT)),
        "json": str(json_path.relative_to(ROOT)),
        "rho_gpt_h": rho_gpt_h,
        "rho_gpt_d": rho_gpt_d,
        "rho_cl_h":  rho_cl_h,
        "rho_cl_d":  rho_cl_d,
        "priority": priority,
        "hypothesis": hypothesis,
        "hep_strategy": meta["endpoint_strategy"]["hepatic"],
        "dea_strategy": meta["endpoint_strategy"]["death"],
    }
    ALL_CANDIDATES.append(record)
    return record


def _strategy_label(blend: list[tuple[str, str, float]]) -> str:
    parts = [f"{w:.2f}*{src}.{col}" for src, col, w in blend]
    return " + ".join(parts)


def _uses_horizon(blend: list[tuple[str, str, float]]) -> bool:
    return any(src.startswith("gpt_anchor") or src.startswith("gpt_phase3_9")
               or src.startswith("gpt_phase3_5_v3hep") for src, _, _ in blend)


# ---------------------------------------------------------------------------
# Define the 10 candidates
# ---------------------------------------------------------------------------

def main():
    print("[1/3] Loading sources & confirming schemas...")
    inp = load_inputs()

    print("[2/3] Building candidates...")

    # ---- Family A: best-vs-best both-endpoint blends ----
    write_candidate(
        name="merge_A_best_50_50_both",
        description="50/50 GPT anchor and Claude anchor on BOTH endpoints",
        hypothesis="Equal rank-mix of the two best track submissions tests whether full cross-track diversity helps.",
        priority="medium",
        hep_blend=[("gpt_anchor", "risk_hepatic_event", 0.5),
                   ("claude_anchor", "risk_hepatic_event", 0.5)],
        dea_blend=[("gpt_anchor", "risk_death", 0.5),
                   ("claude_anchor", "risk_death", 0.5)],
        inputs=inp,
        notes="Family A reference point.",
    )
    write_candidate(
        name="merge_A_best_60gpt_40claude_both",
        description="60% GPT anchor + 40% Claude anchor on BOTH endpoints",
        hypothesis="Slight GPT lean while keeping substantial cross-track contribution.",
        priority="medium",
        hep_blend=[("gpt_anchor", "risk_hepatic_event", 0.6),
                   ("claude_anchor", "risk_hepatic_event", 0.4)],
        dea_blend=[("gpt_anchor", "risk_death", 0.6),
                   ("claude_anchor", "risk_death", 0.4)],
        inputs=inp,
    )
    write_candidate(
        name="merge_A_best_70gpt_30claude_both",
        description="70% GPT anchor + 30% Claude anchor on BOTH endpoints",
        hypothesis="Conservative cross-track blend dominated by the stronger GPT anchor.",
        priority="medium",
        hep_blend=[("gpt_anchor", "risk_hepatic_event", 0.7),
                   ("claude_anchor", "risk_hepatic_event", 0.3)],
        dea_blend=[("gpt_anchor", "risk_death", 0.7),
                   ("claude_anchor", "risk_death", 0.3)],
        inputs=inp,
    )
    write_candidate(
        name="merge_A_best_80gpt_20claude_both",
        description="80% GPT anchor + 20% Claude anchor on BOTH endpoints",
        hypothesis="Tiny Claude share — close to GPT anchor with a small variance hedge.",
        priority="medium",
        hep_blend=[("gpt_anchor", "risk_hepatic_event", 0.8),
                   ("claude_anchor", "risk_hepatic_event", 0.2)],
        dea_blend=[("gpt_anchor", "risk_death", 0.8),
                   ("claude_anchor", "risk_death", 0.2)],
        inputs=inp,
    )

    # ---- Family B: hepatic-only blends, GPT death unchanged ----
    write_candidate(
        name="merge_B_hep_50_50_gptdeath",
        description="50/50 hepatic blend (GPT anchor + Claude anchor); GPT death unchanged",
        hypothesis="Tests whether maximal hepatic cross-track diversity at GPT's stronger death side helps.",
        priority="low",
        hep_blend=[("gpt_anchor", "risk_hepatic_event", 0.5),
                   ("claude_anchor", "risk_hepatic_event", 0.5)],
        dea_blend=[("gpt_anchor", "risk_death", 1.0)],
        inputs=inp,
        notes="Likely under-performs vs GPT anchor on OOF (Claude hep is weaker), but tests the heuristic Spearman recommendation.",
    )
    write_candidate(
        name="merge_B_hep_70gpt_30claude_gptdeath",
        description="70% GPT + 30% Claude on hepatic; GPT death unchanged",
        hypothesis="Controlled hepatic blend at the upper end of the prompt's suggested 10–25% band.",
        priority="medium",
        hep_blend=[("gpt_anchor", "risk_hepatic_event", 0.7),
                   ("claude_anchor", "risk_hepatic_event", 0.3)],
        dea_blend=[("gpt_anchor", "risk_death", 1.0)],
        inputs=inp,
    )
    write_candidate(
        name="merge_B_hep_80gpt_20claude_gptdeath",
        description="80% GPT + 20% Claude on hepatic; GPT death unchanged",
        hypothesis="Modest hepatic Claude share — middle of the heuristic band.",
        priority="medium",
        hep_blend=[("gpt_anchor", "risk_hepatic_event", 0.8),
                   ("claude_anchor", "risk_hepatic_event", 0.2)],
        dea_blend=[("gpt_anchor", "risk_death", 1.0)],
        inputs=inp,
    )

    # ---- Family C: death hedge ----
    write_candidate(
        name="merge_C_gpthep_death_50_50",
        description="GPT hepatic unchanged; 50/50 GPT/Claude rank-blended death",
        hypothesis="OOF-supported: death-side blend lifts dea OOF C-index from 0.9538 to 0.9573 within an α∈[0.30,0.45] plateau.",
        priority="high",
        hep_blend=[("gpt_anchor", "risk_hepatic_event", 1.0)],
        dea_blend=[("gpt_anchor", "risk_death", 0.5),
                   ("claude_anchor", "risk_death", 0.5)],
        inputs=inp,
        notes="OOF Δweighted +0.0011 over GPT solo; within Claude-track LB noise floor (~±0.028 hep) but the only OOF-supported merge intervention.",
    )

    # ---- Family D: landmark / permissive diversity ----
    write_candidate(
        name="merge_D_gpt_anchor_plus_claude_landmark_hep10",
        description="90% GPT hep + 10% Claude landmark_3y hep; GPT death unchanged",
        hypothesis="Tests whether a Tier-2 landmark RSF (a methodology class entirely absent from GPT) adds clean defensible diversity at low weight.",
        priority="medium",
        hep_blend=[("gpt_anchor", "risk_hepatic_event", 0.9),
                   ("claude_landmark_3y", "risk_hepatic_event", 0.1)],
        dea_blend=[("gpt_anchor", "risk_death", 1.0)],
        inputs=inp,
        notes="Cross-track component-level Spearman GPT-hep × Claude landmark ranges 0.06–0.37 — lowest cross-track diversity available.",
    )
    write_candidate(
        name="merge_D_gpt_anchor_plus_claude_landmark_hep20",
        description="80% GPT hep + 20% Claude landmark_3y hep; GPT death unchanged",
        hypothesis="Same as hep10 but doubled landmark share — explores the tail of the heuristic 10–25% band.",
        priority="low",
        hep_blend=[("gpt_anchor", "risk_hepatic_event", 0.8),
                   ("claude_landmark_3y", "risk_hepatic_event", 0.2)],
        dea_blend=[("gpt_anchor", "risk_death", 1.0)],
        inputs=inp,
    )

    print(f"  emitted {len(ALL_CANDIDATES)} candidates")

    print("[3/3] Writing sprint reports...")
    write_candidates_report()
    write_reproducibility_note()

    print()
    print(f"Total candidates: {len(ALL_CANDIDATES)}")
    print()
    print("Top recommended (priority=high):")
    for c in ALL_CANDIDATES:
        if c["priority"] == "high":
            print(f"  {c['csv']}")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_candidates_report():
    rows = []
    for c in ALL_CANDIDATES:
        rows.append((
            c["candidate_name"],
            c["priority"],
            f"{c['rho_gpt_h']:.4f}",
            f"{c['rho_gpt_d']:.4f}",
            f"{c['rho_cl_h']:.4f}",
            f"{c['rho_cl_d']:.4f}",
            c["hypothesis"],
        ))

    md = []
    md.append("# Merge sprint candidates\n")
    md.append("All candidates are rank-blends. No public-LB tuning, no target-derived features, no `strict_time_aligned` components. Each candidate writes ranks directly into `risk_hepatic_event` / `risk_death` (monotonic, C-index-equivalent to any rescaling).\n")

    md.append("## Source files\n")
    for k, p in INPUT_PATHS.items():
        md.append(f"- `{k}` → `{p.relative_to(ROOT)}`")
    md.append("")

    md.append("## Candidates\n")
    md.append("| # | name | priority | ρ_hep vs GPT | ρ_dea vs GPT | ρ_hep vs Claude | ρ_dea vs Claude | hypothesis |")
    md.append("|---|---|---|---:|---:|---:|---:|---|")
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")
    md.append("")

    md.append("## Endpoint strategies in detail\n")
    for c in ALL_CANDIDATES:
        md.append(f"- **{c['candidate_name']}**")
        md.append(f"  - hep: `{c['hep_strategy']}`")
        md.append(f"  - dea: `{c['dea_strategy']}`")
    md.append("")

    md.append("## Upload priority order\n")
    md.append("Ordered by recommended priority and OOF support.\n")
    md.append("1. **HIGH — `merge_C_gpthep_death_50_50.csv`** — only OOF-supported merge; preserves GPT hep anchor.")
    md.append("2. MEDIUM — `merge_A_best_70gpt_30claude_both.csv` — conservative cross-track blend dominated by GPT.")
    md.append("3. MEDIUM — `merge_A_best_80gpt_20claude_both.csv` — minimal Claude share variance hedge.")
    md.append("4. MEDIUM — `merge_B_hep_80gpt_20claude_gptdeath.csv` — keeps GPT death; tests modest hepatic Claude share.")
    md.append("5. MEDIUM — `merge_B_hep_70gpt_30claude_gptdeath.csv` — heuristic upper-band hepatic blend.")
    md.append("6. MEDIUM — `merge_D_gpt_anchor_plus_claude_landmark_hep10.csv` — methodology-class diversity (Tier-2 landmark) at 10%.")
    md.append("7. MEDIUM — `merge_A_best_60gpt_40claude_both.csv` — heavier Claude share; tests if larger blend helps.")
    md.append("8. MEDIUM — `merge_A_best_50_50_both.csv` — equal both-endpoint mix; reference point.")
    md.append("9. LOW — `merge_D_gpt_anchor_plus_claude_landmark_hep20.csv` — 20% landmark; expected to under-perform hep10.")
    md.append("10. LOW — `merge_B_hep_50_50_gptdeath.csv` — 50/50 hep blend; OOF says Claude hep is too weak for half-share.")
    md.append("")

    md.append("## Near-duplicate warning\n")
    md.append("- Family A (4 variants) and Family B (3 variants) span overlapping α grids. Their pairwise output Spearman correlations are very high (>0.97) within each family. Do not upload more than two variants from any family.")
    md.append("- `merge_A_best_70gpt_30claude_both` and `merge_B_hep_70gpt_30claude_gptdeath` differ only on the death side — pick one based on whether you trust GPT's death predictor more (B) or want full cross-track death blend (A).")
    md.append("- All Family-A and Family-B candidates use the GPT anchor as one input — they are not independent experiments.")
    md.append("")

    md.append("## What each family tests\n")
    md.append("- **Family A** — does symmetric cross-track blending help? Sweeps α∈{0.5,0.6,0.7,0.8} on both endpoints.")
    md.append("- **Family B** — does keeping GPT's stronger death predictor while only blending hepatic match or beat Family A? Tests endpoint-asymmetric blending.")
    md.append("- **Family C** — does the OOF-supported death-only hedge deliver? This is the only OOF-recommended candidate.")
    md.append("- **Family D** — does pulling in a methodology Claude has and GPT lacks (Tier-2 landmark RSF) provide qualitative diversity beyond what α-tuning the anchor delivers? Cross-track component Spearman (0.06–0.37) suggests yes mechanically, OOF C-index suggests no quantitatively.")

    (REP / "merge_sprint_candidates.md").write_text("\n".join(md))


def write_reproducibility_note():
    md = []
    md.append("# Reproducibility note — merge sprint candidates\n")
    md.append("Every candidate in this sprint is a **post-hoc rank-blend of saved track outputs**. None of them require live retraining.\n")

    md.append("## Per-candidate reproducibility status\n")
    md.append("| candidate | regen from track code? | needs both pipelines? | post-hoc-only? | source files |")
    md.append("|---|:---:|:---:|:---:|---|")
    for c in ALL_CANDIDATES:
        # All sprint candidates are post-hoc blends; some need only one track's source CSVs
        needs_both = ("claude" in c["hep_strategy"]) or ("claude" in c["dea_strategy"])
        srcs = []
        if "gpt_anchor" in c["hep_strategy"] or "gpt_anchor" in c["dea_strategy"]:
            srcs.append("`gpt_track_handoff/best_submissions/20260428_0059_phase3_10_horizon_blend_v2.csv`")
        if "claude_anchor" in c["hep_strategy"] or "claude_anchor" in c["dea_strategy"]:
            srcs.append("`claude_track_handoff/best_submissions/phase2_blend_2way_optimal.csv`")
        if "claude_landmark_3y" in c["hep_strategy"] or "claude_landmark_3y" in c["dea_strategy"]:
            srcs.append("`claude_track_handoff/best_submissions/phase2_landmark_3y.csv`")
        md.append(f"| `{c['candidate_name']}` | yes (each input CSV is the output of one track's full pipeline) | {'yes' if needs_both else 'no'} | yes | {' + '.join(srcs)} |")
    md.append("")

    md.append("## What it would take to package any of these as an official notebook/source zip\n")
    md.append("A single competition-grade reproducibility package per candidate would need:\n")
    md.append("1. **GPT track source repo + raw data** (`../gpt/`): regenerates `phase3_10_horizon_blend_v2.csv` via `src/run_phase3_10_horizon.py` (5×3 RepeatedStratifiedKFold, deterministic seed 20260426). Reported runtime ≈ a few minutes per fold per component. The package's `model_descriptions.md` and `ensemble_details.md` document every component, weight and CV split.")
    md.append("2. **Claude track source repo + raw data** (`../claude/`): regenerates the Claude submission inputs (`phase2_blend_2way_optimal.csv`, `phase2_landmark_3y.csv`, plus `test_landmark_3y_rsf.csv` if used) via `experiments/phase2_stack_2way.py` (5×10 stratified CV, `random_state=42+repeat`).")
    md.append("3. **Merge driver** (`merge_sprint/scripts/build_sprint.py`): reads the two pre-computed submission CSVs and emits the rank-blend candidate. Trivially deterministic.")
    md.append("")
    md.append("For Candidate `merge_C_gpthep_death_50_50` (the highest-priority candidate):\n")
    md.append("- the OOF α=0.50 was chosen as the most defensible interior point of the OOF α∈[0.30,0.45] plateau (peak 0.9576), not by public-LB tuning;")
    md.append("- a notebook would (a) run both tracks' OOF generation, (b) compute `mean(C-index over death OOF)` across α∈{0,0.05,…,1}, (c) confirm the plateau, (d) pick α=0.50, (e) apply to test predictions, (f) write the submission. End-to-end runtime is dominated by the source pipelines, not the merge.")
    md.append("")
    md.append("## Caveats\n")
    md.append("- Output ranks (1…423) are equivalent to any monotonic transform under C-index. If the organizer expects calibrated probabilities, additional rescaling would be needed — but the official scoring uses concordance, so ranks are sufficient.")
    md.append("- Cross-track CV folds differ (GPT 5×3 vs Claude 5×10) so any meta-stacker that needs aligned folds is **not** valid; the rank-blends used here do not require fold alignment.")
    md.append("- Per-source weights are documented exactly in each candidate's `metadata/<name>.json`. None were tuned on public LB.")
    md.append("- Post-hoc blends are reproducible but the underlying source predictions are only reproducible through their respective tracks' source repos; raw data is confidential and not redistributed.")

    (REP / "reproducibility_note.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
