"""Driver for Stage 0 + Stage 1: normalize (if not cached), embed (if not
cached), generate candidates, and write candidate_pairs.tsv for a split.

Usage: python run_retrieve.py --split train
       python run_retrieve.py --split test
"""
import argparse
import time

import polars as pl

import config
import embed
import io_utils
import normalize
import retrieve


def load_split(split: str):
    src_dir = config.TRAIN_DIR if split == "train" else config.TEST_DIR
    prefix = "train" if split == "train" else "test"
    sources = {}
    for i in (1, 2, 3):
        path = src_dir / f"{prefix}_source{i}.tsv"
        cache = config.WORK_DIR / f"norm_{prefix}_s{i}.parquet"
        df = normalize.normalize_and_cache(io_utils.read_source(path), cache)
        sources[i] = df
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--k", type=int, default=config.TOP_K_CANDIDATES)
    args = ap.parse_args()

    t0 = time.time()
    sources = load_split(args.split)
    s1, s2, s3 = sources[1], sources[2], sources[3]
    print(f"[{args.split}] normalized: s1={s1.height} s2={s2.height} s3={s3.height}"
          f" ({round(time.time()-t0,1)}s)")

    prefix = "train" if args.split == "train" else "test"
    t1 = time.time()
    s1e = embed.embed_source_and_cache(s1, "passage", config.WORK_DIR / f"emb_{prefix}_s1.npy")
    s2e = embed.embed_source_and_cache(s2, "query", config.WORK_DIR / f"emb_{prefix}_s2.npy")
    s3e = embed.embed_source_and_cache(s3, "query", config.WORK_DIR / f"emb_{prefix}_s3.npy")
    print(f"[{args.split}] embedded ({round(time.time()-t1,1)}s)")

    t2 = time.time()
    candidates_path = config.WORK_DIR / f"candidates_{prefix}.parquet"
    if candidates_path.exists():
        candidates = pl.read_parquet(candidates_path)
        print(f"[{args.split}] candidates: loaded cached {candidates.height} pairs")
    else:
        parts_dir = config.WORK_DIR / f"candidate_parts_{prefix}"
        candidates = retrieve.build_candidates(s1, s1e, [(s2, s2e), (s3, s3e)], parts_dir, k=args.k)
        candidates.write_parquet(candidates_path)
    print(f"[{args.split}] candidates: {candidates.height} pairs "
          f"({round(time.time()-t2,1)}s)")

    out = retrieve.to_candidate_pairs_tsv(s1["entity_id"].to_list(), candidates)
    out_path = config.WORK_DIR / f"candidate_pairs_{prefix}.tsv"
    io_utils.write_tsv(out, out_path)
    print(f"[{args.split}] wrote {out_path} ({out.height} S1 rows), "
          f"total wall time {round(time.time()-t0,1)}s")


if __name__ == "__main__":
    main()
