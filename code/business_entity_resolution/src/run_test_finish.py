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

TAU = 0.91  # picked from the flat peak (0.90-0.92) of the validation F0.5 sweep


def main():
    t0 = time.time()
    s1 = pl.read_parquet(config.WORK_DIR / "norm_test_s1.parquet")

    scored_path = config.WORK_DIR / "test_scored.parquet"
    scored = score_matcher.score(
        config.WORK_DIR / "pairs_features_test.parquet",
        config.WORK_DIR / "lgbm_matcher.txt",
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
