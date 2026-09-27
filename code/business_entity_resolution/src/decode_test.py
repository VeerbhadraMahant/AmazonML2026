"""Decode an already-scored test parquet (source1_entity_id, entity_id, pred)
into a matching_results.tsv with the exclusive-assignment decoder.

Usage: python decode_test.py ctx 0.91 <out_path>
"""
import sys
import time

import polars as pl

import config
import decode
import io_utils

if __name__ == "__main__":
    t0 = time.time()
    suffix, tau, out_path = "_" + sys.argv[1], float(sys.argv[2]), sys.argv[3]
    s1 = pl.read_parquet(config.WORK_DIR / "norm_test_s1.parquet", columns=["entity_id"])
    scored = pl.read_parquet(config.WORK_DIR / f"test_scored{suffix}.parquet")
    assigned = decode.exclusive_assign(scored, tau)
    out = decode.to_result_frame(s1["entity_id"].to_list(), assigned)
    io_utils.write_tsv(out, out_path)
    n_empty = (out["matched_entity_ids"] == "").sum()
    print(f"wrote {out_path}: {out.height} S1 rows, {assigned.height} matched pairs, "
          f"singleton rate {n_empty/out.height:.4f}, tau={tau} ({round(time.time()-t0,1)}s)")
