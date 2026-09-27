"""Score the test candidate features with a trained matcher (no decoding), so
the decoding threshold can be picked afterwards without re-scoring.

Usage: python score_test.py ctx   -> work/test_scored_ctx.parquet
"""
import sys
import time

import config
import score_matcher

if __name__ == "__main__":
    t0 = time.time()
    suffix = "_" + sys.argv[1] if len(sys.argv) > 1 else ""
    out = score_matcher.score(
        config.WORK_DIR / f"pairs_features_test{suffix}.parquet",
        config.WORK_DIR / f"lgbm_matcher{suffix}.txt",
        config.WORK_DIR / f"test_scored{suffix}.parquet",
    )
    print(f"[test] scored {out.height} pairs ({round(time.time()-t0,1)}s)")
