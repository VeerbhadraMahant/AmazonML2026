"""Val entity windows only contain val-S1 rows (~1.7 of ~12 candidates per
record), unlike test where every S1 competitor is present. To make stage-2
entity-side features honest, score (stage-1) ALL train pairs whose entity_id
appears in val, and save the narrow (hs, he, pred, is_val) frame.

Output: work/stage2_val_ctx_narrow.parquet
"""
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import config
from stack_stage2 import BASE_COLS, hash_keys

t0 = time.time()
val = pl.read_parquet(config.WORK_DIR / "val_scored_ctx.parquet", columns=["source1_entity_id", "entity_id"])
val_s1 = val["source1_entity_id"].unique()
val_ent = val["entity_id"].unique()
del val
model = lgb.Booster(model_file=str(config.WORK_DIR / "lgbm_matcher_ctx.txt"))
pf = pq.ParquetFile(str(config.WORK_DIR / "pairs_features_train_ctx.parquet"))
parts = []
seen = 0
for batch in pf.iter_batches(batch_size=4_000_000, columns=["source1_entity_id", "entity_id", *BASE_COLS]):
    df = pl.from_arrow(pa.Table.from_batches([batch]))
    seen += df.height
    df = df.filter(pl.col("entity_id").is_in(val_ent))
    if df.height:
        pred = model.predict(df.select(BASE_COLS).to_numpy().astype(np.float32),
                             num_iteration=model.best_iteration)
        parts.append(hash_keys(df.select("source1_entity_id", "entity_id")).select(
            "hs", "he", pl.col("source1_entity_id").is_in(val_s1).alias("is_val"),
            pl.Series("pred", pred.astype(np.float32))))
    print(f"seen {seen}, kept {sum(p.height for p in parts)} ({time.time()-t0:.0f}s)", flush=True)
out = pl.concat(parts)
out.write_parquet(config.WORK_DIR / "stage2_val_ctx_narrow.parquet")
print(f"done {out.height} rows, is_val={out['is_val'].sum()} ({time.time()-t0:.0f}s)")
