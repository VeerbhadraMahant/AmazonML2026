"""2-fold out-of-fold stage-1 predictions over ALL train candidate pairs.

Stage-2 (stack_stage2.py) builds competition features from stage-1 preds over
every S1 competing for the same S2/S3 record. On test every S1 is present, so
the stage-2 training data must also have preds for every train S1 -- and
those preds must be out-of-sample, or the stacker learns from overconfident
in-sample scores. Folds are by source1_entity_id. Test preds are the average
of the two fold models, so train (OOF) and test preds come from models of the
same strength.

Usage: python oof_stage1.py
"""
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import config
from train_matcher import CONTEXT_FEATURE_COLS, FEATURE_COLS

FEATS = FEATURE_COLS + CONTEXT_FEATURE_COLS
NEG_KEEP = 0.15
N_ROUNDS = 339  # best_iteration of the full ctx model
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5, verbose=-1,
              seed=config.RANDOM_SEED, num_threads=20)


def fold_expr():
    return (pl.col("source1_entity_id").hash(seed=0) % 2).cast(pl.Int8).alias("fold")


def build_samples(path):
    rng = np.random.default_rng(config.RANDOM_SEED)
    parts = {0: [], 1: []}
    for batch in pq.ParquetFile(str(path)).iter_batches(
            batch_size=5_000_000, columns=["source1_entity_id", "label", *FEATS]):
        df = pl.from_arrow(pa.Table.from_batches([batch])).with_columns(fold_expr())
        keep = (df["label"] == 1).to_numpy() | (rng.random(df.height) < NEG_KEEP)
        df = df.filter(pl.Series(keep))
        for k in (0, 1):
            d = df.filter(pl.col("fold") == k)
            parts[k].append((d.select(FEATS).to_numpy().astype(np.float32),
                             d["label"].to_numpy().astype(np.float32)))
    return {k: (np.vstack([p[0] for p in v]), np.concatenate([p[1] for p in v]))
            for k, v in parts.items()}


def score_file(path, models, out_path, with_label):
    cols = ["source1_entity_id", "entity_id", *FEATS] + (["label"] if with_label else [])
    writer = None
    for batch in pq.ParquetFile(str(path)).iter_batches(batch_size=5_000_000, columns=cols):
        df = pl.from_arrow(pa.Table.from_batches([batch]))
        X = df.select(FEATS).to_numpy().astype(np.float32)
        if with_label:  # OOF: row in fold k is scored by the model trained on fold 1-k
            df = df.with_columns(fold_expr())
            fold = df["fold"].to_numpy()
            pred = np.where(fold == 0, models[1].predict(X), models[0].predict(X))
            keep = ["source1_entity_id", "entity_id", "label", "fold"]
        else:
            pred = 0.5 * (models[0].predict(X) + models[1].predict(X))
            keep = ["source1_entity_id", "entity_id"]
        tbl = df.select(keep).with_columns(pl.Series("pred", pred.astype(np.float32))).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), tbl.schema)
        writer.write_table(tbl)
        del df, X, pred
    writer.close()


if __name__ == "__main__":
    t0 = time.time()
    W = config.WORK_DIR
    samples = build_samples(W / "pairs_features_train_ctx.parquet")
    print(f"samples built: fold0={len(samples[0][1])} fold1={len(samples[1][1])} "
          f"({round(time.time()-t0,1)}s)", flush=True)
    models = {}
    for k in (0, 1):
        X, y = samples[k]
        models[k] = lgb.train(PARAMS, lgb.Dataset(X, label=y, feature_name=FEATS), N_ROUNDS)
        models[k].save_model(str(W / f"lgbm_oof_fold{k}.txt"))
        print(f"trained fold{k} ({round(time.time()-t0,1)}s)", flush=True)
    del samples
    score_file(W / "pairs_features_train_ctx.parquet", models, W / "oof_train_scored.parquet", True)
    print(f"wrote oof_train_scored.parquet ({round(time.time()-t0,1)}s)", flush=True)
    score_file(W / "pairs_features_test_ctx.parquet", models, W / "test_scored_oof.parquet", False)
    print(f"wrote test_scored_oof.parquet ({round(time.time()-t0,1)}s)", flush=True)
