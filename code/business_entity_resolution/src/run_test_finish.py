"""Final step for the test split: score the featurized candidates with the
trained matcher, decode into exclusive per-S1 assignments, and write both
submission files into output/.

Run this AFTER run_retrieve.py --split test and run_features.py --split test
have both completed (each is its own process, so memory from one stage is
fully released before the next starts -- see run_test_pipeline.py's
docstring/history for why that matters here).

Usage: python run_test_finish.py
"""
import shutil
import time

import polars as pl

import config
import decode
import io_utils
import score_matcher

import sys

TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 0.91  # from the validation F0.5 sweep


def main():
    t0 = time.time()
    s1 = pl.read_parquet(config.WORK_DIR / "norm_test_s1.parquet")

    suffix = "_" + sys.argv[1] if len(sys.argv) > 1 else ""
    scored_path = config.WORK_DIR / f"test_scored{suffix}.parquet"
    scored = score_matcher.score(
        config.WORK_DIR / f"pairs_features_test{suffix}.parquet",
        config.WORK_DIR / f"lgbm_matcher{suffix}.txt",
        scored_path,
    )
    print(f"[test] scored {scored.height} pairs ({round(time.time()-t0,1)}s)")

    assigned = decode.exclusive_assign(scored, TAU)
    match_out = decode.to_result_frame(s1["entity_id"].to_list(), assigned)
    io_utils.write_tsv(match_out, config.OUTPUT_DIR / "matching_results.tsv")
    print(f"[test] wrote {config.OUTPUT_DIR / 'matching_results.tsv'} "
          f"({match_out.height} S1 rows, {assigned.height} matched pairs, tau={TAU})")

    cand_src = config.WORK_DIR / "candidate_pairs_test.tsv"
    cand_dst = config.OUTPUT_DIR / "candidate_pairs.tsv"
    shutil.copyfile(cand_src, cand_dst)
    print(f"[test] copied {cand_src} -> {cand_dst}")

    print(f"[test] total {round(time.time()-t0,1)}s")


if __name__ == "__main__":
    main()
