"""Split the featurized candidate pairs into a training sample (positives +
a subsampled negative pool, to keep LightGBM training fast) and a full,
un-subsampled validation set (needed for realistic decode-time F0.5 -- the
decoder must see every candidate for a held-out S1 entity, not a sample).

Split is by source1_entity_id (not by pair) so no S1 entity's candidates
leak across train/val.
"""
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import config


def make_split_ids(s1_path, val_fraction=config.VAL_FRACTION, seed=config.RANDOM_SEED):
    ids = pl.read_parquet(s1_path)["entity_id"].to_list()
    rng = np.random.default_rng(seed)
    ids = np.array(ids)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_fraction)
    return set(ids[n_val:].tolist()), set(ids[:n_val].tolist())  # train_ids, val_ids


def split_and_sample(features_path, s1_path, train_out, val_out,
                      neg_keep_frac=0.15, batch_rows=3_000_000, seed=config.RANDOM_SEED):
    train_ids, val_ids = make_split_ids(s1_path)
    rng = np.random.default_rng(seed)

    pf = pq.ParquetFile(str(features_path))
    train_writer = None
    val_writer = None
    n_train, n_val = 0, 0
    for batch in pf.iter_batches(batch_size=batch_rows):
        df = pl.from_arrow(pa.Table.from_batches([batch]))
        is_val = df["source1_entity_id"].is_in(val_ids)
        val_part = df.filter(is_val)
        train_part = df.filter(~is_val)

        pos = train_part.filter(pl.col("label") == 1)
        neg = train_part.filter(pl.col("label") == 0)
        if neg.height > 0 and neg_keep_frac < 1.0:
            keep_mask = rng.random(neg.height) < neg_keep_frac
            neg = neg.filter(pl.Series(keep_mask))
        train_sample = pl.concat([pos, neg])

        if train_sample.height > 0:
            tbl = train_sample.to_arrow()
            if train_writer is None:
                train_writer = pq.ParquetWriter(str(train_out), tbl.schema)
            train_writer.write_table(tbl)
            n_train += train_sample.height
        if val_part.height > 0:
            tbl = val_part.to_arrow()
            if val_writer is None:
                val_writer = pq.ParquetWriter(str(val_out), tbl.schema)
            val_writer.write_table(tbl)
            n_val += val_part.height

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()
    return n_train, n_val, len(train_ids), len(val_ids)


if __name__ == "__main__":
    n_train, n_val, n_train_ids, n_val_ids = split_and_sample(
        config.WORK_DIR / "pairs_features_train.parquet",
        config.WORK_DIR / "norm_train_s1.parquet",
        config.WORK_DIR / "train_sample.parquet",
        config.WORK_DIR / "val_full.parquet",
    )
    print(f"train_sample: {n_train} rows from {n_train_ids} S1 ids")
    print(f"val_full: {n_val} rows from {n_val_ids} S1 ids")
