"""Rebuild train_sample.parquet with hard-negative mining.

The original split_and_sample.py keeps a uniform random 15% of negatives.
That throws away most of the near-miss negatives (same retrieval top-ranks,
same core name / number) that the exclusive-argmax decoder is most sensitive
to -- those are exactly the pairs that can outscore a weak true positive for
the same S2/3 record. This rebuilds the training sample keeping:
  - all positives
  - all "hard" negatives: cand_rank <= 3 (near the top of the embedding
    retrieval) OR cand_rank is null (lexical-only candidate: exact core-name
    or exact first-token+number match that the embedding retriever didn't
    even put in its own top-10 -- i.e. a genuine generic-name/number
    collision, the textbook confusor)
  - a subsample of the remaining "easy" negatives, to keep total volume
    similar to before

val_full.parquet (the un-subsampled validation set used for decode-time
F0.5) is untouched -- only the training sample changes.
"""
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import config
from split_and_sample import make_split_ids


def build(features_path, s1_path, train_out, easy_neg_keep_frac=0.06,
          batch_rows=3_000_000, seed=config.RANDOM_SEED):
    train_ids, val_ids = make_split_ids(s1_path)
    rng = np.random.default_rng(seed)

    pf = pq.ParquetFile(str(features_path))
    writer = None
    n_pos = n_hard = n_easy = 0
    for batch in pf.iter_batches(batch_size=batch_rows):
        df = pl.from_arrow(pa.Table.from_batches([batch]))
        df = df.filter(~df["source1_entity_id"].is_in(val_ids))
        if df.height == 0:
            continue

        pos = df.filter(pl.col("label") == 1)
        neg = df.filter(pl.col("label") == 0)
        hard = neg.filter(pl.col("cand_rank").is_null() | (pl.col("cand_rank") <= 3))
        easy = neg.filter(pl.col("cand_rank").is_not_null() & (pl.col("cand_rank") > 3))
        if easy.height > 0 and easy_neg_keep_frac < 1.0:
            keep_mask = rng.random(easy.height) < easy_neg_keep_frac
            easy = easy.filter(pl.Series(keep_mask))

        out = pl.concat([pos, hard, easy])
        n_pos += pos.height
        n_hard += hard.height
        n_easy += easy.height
        if out.height > 0:
            tbl = out.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(str(train_out), tbl.schema)
            writer.write_table(tbl)
    if writer is not None:
        writer.close()
    return n_pos, n_hard, n_easy


if __name__ == "__main__":
    n_pos, n_hard, n_easy = build(
        config.WORK_DIR / "pairs_features_train.parquet",
        config.WORK_DIR / "norm_train_s1.parquet",
        config.WORK_DIR / "train_sample_hardneg.parquet",
    )
    total = n_pos + n_hard + n_easy
    print(f"positives={n_pos} hard_neg={n_hard} easy_neg_kept={n_easy} total={total}")
