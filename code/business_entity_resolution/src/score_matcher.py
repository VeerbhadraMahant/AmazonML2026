"""Apply the trained LightGBM matcher to a featurized (unlabeled) candidate
set -- used for the test split, which has no ground truth.

Predicts in parquet batches: the full test feature matrix (120M rows x 25
features) would be ~24GB as float64, more than this machine's RAM."""
import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from train_matcher import CONTEXT_FEATURE_COLS, FEATURE_COLS


def score(features_path, model_path, out_path, batch_rows=5_000_000) -> pl.DataFrame:
    model = lgb.Booster(model_file=str(model_path))
    pf = pq.ParquetFile(str(features_path))
    cols_present = pf.schema_arrow.names
    feature_cols = FEATURE_COLS + [c for c in CONTEXT_FEATURE_COLS if c in cols_present]
    parts = []
    for batch in pf.iter_batches(batch_size=batch_rows,
                                 columns=["source1_entity_id", "entity_id", *feature_cols]):
        df = pl.from_arrow(pa.Table.from_batches([batch]))
        X = df.select(feature_cols).to_numpy().astype(np.float32)
        pred = model.predict(X, num_iteration=model.best_iteration)
        parts.append(df.select("source1_entity_id", "entity_id").with_columns(
            pl.Series("pred", pred.astype(np.float32))))
        del df, X, pred
    out = pl.concat(parts)
    out.write_parquet(out_path)
    return out
