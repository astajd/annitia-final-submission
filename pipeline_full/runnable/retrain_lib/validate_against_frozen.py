"""Compare the retrain-assembled slot1 prediction against frozen/slot1_prediction.csv.

Reports (per endpoint): schema, row count, trustii_id alignment, Spearman rho,
rank identity, max abs diff, float equality; plus md5 of both files.

Exit code 0 only if BOTH endpoints are rank-identical (the authoritative pass
criterion). Float-exact equality is reported and, when true, noted as GOLD.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--frozen", required=True, type=Path)
    args = ap.parse_args()

    if not args.pred.exists():
        print(f"FAIL: generated prediction missing: {args.pred}")
        return 2
    if not args.frozen.exists():
        print(f"FAIL: frozen reference missing: {args.frozen}")
        return 2

    new = pd.read_csv(args.pred)
    fr = pd.read_csv(args.frozen)

    print("=" * 72)
    print("RETRAIN slot1  vs  frozen/slot1_prediction.csv")
    print("=" * 72)
    print(f"generated : {args.pred}")
    print(f"frozen    : {args.frozen}")
    print(f"md5 generated : {md5(args.pred)}")
    print(f"md5 frozen    : {md5(args.frozen)}")
    print(f"file md5 equal : {md5(args.pred) == md5(args.frozen)}")
    print(f"schema new : {list(new.columns)}  rows={len(new)}")
    print(f"schema ref : {list(fr.columns)}  rows={len(fr)}")

    cols = ["trustii_id", "risk_hepatic_event", "risk_death"]
    if list(new.columns) != cols or list(fr.columns) != cols:
        print(f"FAIL: schema mismatch (expected {cols})")
        return 2
    if len(new) != 423 or len(fr) != 423:
        print(f"FAIL: row count not 423 (new={len(new)}, frozen={len(fr)})")
        return 2

    n = new.sort_values("trustii_id").reset_index(drop=True)
    o = fr.sort_values("trustii_id").reset_index(drop=True)
    id_ok = set(n.trustii_id) == set(o.trustii_id) and bool((n.trustii_id.values == o.trustii_id.values).all())
    print(f"trustii_id alignment (set & order): {id_ok}")
    if not id_ok:
        print("FAIL: trustii_id alignment")
        return 2

    all_rank_ident = True
    all_float_exact = True
    for c in ["risk_hepatic_event", "risk_death"]:
        a = n[c].to_numpy(); b = o[c].to_numpy()
        rho = spearmanr(a, b).statistic
        ri = bool((pd.Series(a).rank().values == pd.Series(b).rank().values).all())
        ex = bool(np.array_equal(a, b))
        md = float(np.max(np.abs(a - b)))
        all_rank_ident &= ri
        all_float_exact &= ex
        print(f"\n[{c}]")
        print(f"  Spearman rho   : {rho:.10f}")
        print(f"  rank identity  : {ri}")
        print(f"  float equality : {ex}")
        print(f"  max abs diff   : {md:.6e}")

    print("\n" + "-" * 72)
    if all_rank_ident and all_float_exact:
        print("SUCCESS — regenerated slot1 is FLOAT-EXACT (GOLD) to frozen/slot1_prediction.csv")
        return 0
    if all_rank_ident:
        print("SUCCESS — regenerated slot1 is RANK-IDENTICAL to frozen/slot1_prediction.csv")
        print("(float differences present but rank order identical — passes C-index criterion)")
        return 0
    print("FAIL — regenerated slot1 is NOT rank-identical to frozen/slot1_prediction.csv")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
