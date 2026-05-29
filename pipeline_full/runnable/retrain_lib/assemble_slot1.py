"""Assemble slot1 from REGENERATED components, using the authoritative
`build_slot1_only.py` UNCHANGED.

Strategy (no final-arithmetic reimplementation):
  1. Derive the two deterministic intermediate components from the regenerated
     track anchors, reusing the VERIFIED upstream functions:
       - merge_A_best_50_50_both  ->  build_sprint.blend_ranks  (50/50 both endpoints)
       - test__gate_LSthr12__hep  ->  task5_gated_ensemble.conditional_rule_blend
         (zero-parameter LS>=12 rule; test side only; raw LS from data root)
  2. Stage all SIX regenerated inputs into <work>/regenerated_intermediates/
     in exactly the relative layout build_slot1_only.py expects.
  3. Copy build_slot1_only.py VERBATIM into <work>/ and create a symlink
     `cached_intermediates -> regenerated_intermediates` so the unchanged script
     reads the regenerated files (its `CACHED = HERE/"cached_intermediates"`).
  4. Run it as a subprocess; copy its `generated_slot1_prediction.csv` to --out.

Nothing here recomputes the q95 disagreement-override or the 0.85/0.15 death
blend — that arithmetic lives only in build_slot1_only.py and is executed verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _die(msg: str) -> None:
    raise SystemExit(f"FATAL [assemble_slot1]: {msg}")


def _require(p: Path, what: str) -> Path:
    if not p.exists():
        _die(f"expected freshly-generated artifact missing ({what}): {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpt-anchor", required=True, type=Path)   # regenerated Track B
    ap.add_argument("--cl-anchor", required=True, type=Path)    # regenerated Track A
    ap.add_argument("--cwgbs", required=True, type=Path)        # regenerated death CWGBSA test
    ap.add_argument("--gbsa", required=True, type=Path)         # regenerated death GBSA test
    ap.add_argument("--raw-root", required=True, type=Path)     # data/raw (train.csv,test.csv)
    ap.add_argument("--runnable", required=True, type=Path)     # pipeline_full/runnable
    ap.add_argument("--work", required=True, type=Path)         # retrain_work/slot1_assembly
    ap.add_argument("--out", required=True, type=Path)          # final_retrain_prediction.csv
    args = ap.parse_args()

    sys.path.insert(0, str(args.runnable / "upstream_scripts"))
    sys.path.insert(0, str(args.runnable / "lib"))
    from build_sprint import blend_ranks                       # verified 50/50 rank-mix
    from task5_gated_ensemble import conditional_rule_blend    # verified LS gate

    for p, w in [(args.gpt_anchor, "gpt anchor"), (args.cl_anchor, "claude anchor"),
                 (args.cwgbs, "death CWGBSA"), (args.gbsa, "death GBSA")]:
        _require(p, w)
    _require(args.raw_root / "train.csv", "raw train.csv")
    _require(args.raw_root / "test.csv", "raw test.csv")

    gpt = pd.read_csv(args.gpt_anchor).sort_values("trustii_id").reset_index(drop=True)
    cl = pd.read_csv(args.cl_anchor).sort_values("trustii_id").reset_index(drop=True)
    tid = gpt["trustii_id"].to_numpy()
    if not (cl["trustii_id"].to_numpy() == tid).all():
        _die("regenerated gpt/cl anchor trustii_id mismatch")
    gpt_h = gpt["risk_hepatic_event"].to_numpy()
    cl_h = cl["risk_hepatic_event"].to_numpy()

    train = pd.read_csv(args.raw_root / "train.csv")
    test = pd.read_csv(args.raw_root / "test.csv")

    # ---- (1a) merge_A_best_50_50_both via verified blend_ranks ----
    merge = pd.DataFrame({
        "trustii_id": tid,
        "risk_hepatic_event": blend_ranks([(gpt_h, 0.5), (cl_h, 0.5)]),
        "risk_death": blend_ranks([(gpt["risk_death"].to_numpy(), 0.5),
                                   (cl["risk_death"].to_numpy(), 0.5)]),
    })

    # ---- (1b) test__gate_LSthr12__hep via verified conditional_rule_blend ----
    # OOF args are unused for the test-side return; pass dummies of train length.
    dummy = np.zeros(len(train))
    _, gate_te_in_testorder = conditional_rule_blend(
        dummy, dummy, gpt_h, cl_h, train, test,
        var="fibs_stiffness_med_BM_1", thr=12.0)
    # conditional_rule_blend follows raw `test` row order; realign to anchor tid order.
    test_tid = test["trustii_id"].to_numpy()
    order = pd.Series(np.arange(len(test_tid)), index=test_tid).reindex(tid)
    if order.isna().any():
        _die("gate: raw test trustii_id does not cover anchor trustii_id")
    gate_h = gate_te_in_testorder[order.to_numpy().astype(int)]
    gate_df = pd.DataFrame({"trustii_id": tid, "hepatic_risk": gate_h})

    # ---- (2) stage SIX regenerated inputs in build_slot1_only's layout ----
    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work)
    regen = work / "regenerated_intermediates"
    (regen / "gpt_track_handoff" / "best_submissions").mkdir(parents=True, exist_ok=True)
    (regen / "claude_track_handoff" / "best_submissions").mkdir(parents=True, exist_ok=True)
    (regen / "merge_sprint" / "submissions").mkdir(parents=True, exist_ok=True)
    (regen / "model_zoo_sprint" / "predictions").mkdir(parents=True, exist_ok=True)

    shutil.copyfile(args.gpt_anchor, regen / "gpt_track_handoff" / "best_submissions" / "20260428_0059_phase3_10_horizon_blend_v2.csv")
    shutil.copyfile(args.cl_anchor, regen / "claude_track_handoff" / "best_submissions" / "phase2_blend_2way_optimal.csv")
    merge.to_csv(regen / "merge_sprint" / "submissions" / "merge_A_best_50_50_both.csv", index=False)
    gate_df.to_csv(regen / "model_zoo_sprint" / "predictions" / "test__gate_LSthr12__hep.csv", index=False)
    shutil.copyfile(args.cwgbs, regen / "model_zoo_sprint" / "predictions" / "test__survtree__dea__longitudinal_summary__cwgbs_300_lr05.csv")
    shutil.copyfile(args.gbsa, regen / "model_zoo_sprint" / "predictions" / "test__survtree__dea__longitudinal_summary__gbsa_200_lr05_d3.csv")

    # ---- (3) copy build_slot1_only.py VERBATIM + symlink expected input dir ----
    src_script = _require(args.runnable / "build_slot1_only.py", "build_slot1_only.py")
    staged_script = work / "build_slot1_only.py"
    shutil.copyfile(src_script, staged_script)              # byte-for-byte, logic unchanged
    # Byte-identity guard: the assembly script must be the authoritative one, untouched.
    if hashlib.md5(src_script.read_bytes()).hexdigest() != hashlib.md5(staged_script.read_bytes()).hexdigest():
        _die("staged build_slot1_only.py is not byte-identical to the authoritative copy")
    link = work / "cached_intermediates"
    os.symlink(regen.name, link)   # cached_intermediates -> regenerated_intermediates

    # ---- (4) run the unchanged assembler ----
    print("[assemble_slot1] running build_slot1_only.py (verbatim) over regenerated inputs", flush=True)
    r = subprocess.run([sys.executable, str(staged_script)], cwd=str(work),
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        _die(f"build_slot1_only.py exited {r.returncode}")

    gen = _require(work / "generated_slot1_prediction.csv", "assembled slot1 output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gen, args.out)
    print(f"[assemble_slot1] wrote final assembled prediction -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
