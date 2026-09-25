"""Driver for Stage 2 feature computation. Must run as a script (not from
stdin) -- Windows' multiprocessing 'spawn' start method needs a real,
re-importable __main__ module.

Usage: python run_features.py --split train
       python run_features.py --split test
"""
import argparse
import time

import config
import features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--batch-rows", type=int, default=2_000_000)
    ap.add_argument("--n-workers", type=int, default=None)
    args = ap.parse_args()

    prefix = "train" if args.split == "train" else "test"
    candidates_path = config.WORK_DIR / f"candidates_{prefix}.parquet"
    s1_path = config.WORK_DIR / f"norm_{prefix}_s1.parquet"
    other_paths = [config.WORK_DIR / f"norm_{prefix}_s2.parquet",
                   config.WORK_DIR / f"norm_{prefix}_s3.parquet"]
    augmented_path = config.WORK_DIR / f"candidates_augmented_{prefix}.parquet"
    out_path = config.WORK_DIR / f"pairs_features_{prefix}.parquet"
    gt_path = config.TRAIN_DIR / "train_ground_truth.tsv" if args.split == "train" else None

    t0 = time.time()
    features.augment_candidates(candidates_path, s1_path, augmented_path, gt_path=gt_path)
    print(f"[{args.split}] augment_candidates done ({round(time.time()-t0,1)}s)")

    t1 = time.time()
    n = features.compute_features_streaming(augmented_path, s1_path, other_paths, out_path,
                                             batch_rows=args.batch_rows, n_workers=args.n_workers)
    print(f"[{args.split}] compute_features_streaming: {n} rows ({round(time.time()-t1,1)}s), "
          f"wrote {out_path}, total {round(time.time()-t0,1)}s")


if __name__ == "__main__":
    main()
