"""Apply the trained LightGBM matcher to a featurized (unlabeled) candidate
set -- used for the test split, which has no ground truth."""
import lightgbm as lgb
import polars as pl

from train_matcher import FEATURE_COLS


def score(features_path, model_path, out_path) -> pl.DataFrame:
    model = lgb.Booster(model_file=str(model_path))
    df = pl.read_parquet(features_path)
    X = df.select(FEATURE_COLS).to_numpy()
    pred = model.predict(X, num_iteration=model.best_iteration)
    out = df.select("source1_entity_id", "entity_id").with_columns(pl.Series("pred", pred))
    out.write_parquet(out_path)
    return out
