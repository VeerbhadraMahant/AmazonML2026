"""Fine-grained tau sweep (overall + per-country best) on a val_scored_*.parquet.

Usage: python evaluate_fine.py ctx
"""
import sys

import numpy as np
import polars as pl

import config
import decode
from evaluate import build_val_ground_truth


def main():
    suffix = "_" + sys.argv[1] if len(sys.argv) > 1 else ""
    scored = pl.read_parquet(config.WORK_DIR / f"val_scored{suffix}.parquet").select(
        "source1_entity_id", "entity_id", "pred")
    gt, val_ids = build_val_ground_truth()
    best = (None, -1, None)
    for tau in np.round(np.arange(0.70, 0.991, 0.01), 2):
        assigned = decode.exclusive_assign(scored, tau)
        m = decode.f_beta_macro(decode.to_result_frame(val_ids, assigned), gt)
        pc = {r["country"]: round(r["macro_f"], 4) for r in m["per_country"]}
        print(f"tau={tau:.2f}  macro_F0.5={m['macro_f']:.4f}  {pc}", flush=True)
        if m["macro_f"] > best[1]:
            best = (tau, m["macro_f"], pc)
    print(f"\nBEST tau={best[0]:.2f} macro_F0.5={best[1]:.4f} {best[2]}")


if __name__ == "__main__":
    main()
