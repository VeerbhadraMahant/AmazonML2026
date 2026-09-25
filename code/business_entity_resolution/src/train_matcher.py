"""Train the LightGBM pairwise matcher (MIT-licensed gradient-boosted trees,
not a neural net -- trivially inside the challenge's 8B-parameter cap) on the
featurized candidate pairs, and evaluate on the held-out validation split.

Country is deliberately NOT a feature (see features.py docstring / plan):
France never appears in training, so a country feature would make the model
unreliable exactly where we can't check it.
"""
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

import config

FEATURE_COLS = [
    "sim", "cand_rank", "cand_margin", "s1_name_core_freq",
    "name_ratio", "name_core_ratio", "name_token_set_ratio",
    "addr_ratio", "addr_token_set_ratio",
    "name_first_token_eq", "number_exact", "number_prefix", "number_jaccard",
    "name_len_diff", "addr_len_diff", "either_non_ascii",
]


def load_xy(path):
    df = pl.read_parquet(path)
    X = df.select(FEATURE_COLS).to_numpy()
    y = df["label"].to_numpy()
    return df, X, y


def train(train_path, val_path, model_out):
    t0 = time.time()
    train_df, X_train, y_train = load_xy(train_path)
    val_df, X_val, y_val = load_xy(val_path)
    print(f"train: {X_train.shape}, positives={y_train.sum()} ({y_train.mean():.4f}); "
          f"val: {X_val.shape}, positives={y_val.sum()} ({y_val.mean():.4f})")

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_set)

    params = dict(
        objective="binary",
        metric=["auc", "average_precision"],
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        seed=config.RANDOM_SEED,
    )
    model = lgb.train(
        params, train_set, num_boost_round=500,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    print(f"trained in {round(time.time()-t0,1)}s, best_iteration={model.best_iteration}")

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, val_pred)
    ap = average_precision_score(y_val, val_pred)
    print(f"val AUC={auc:.4f} AP={ap:.4f}")

    importance = dict(zip(FEATURE_COLS, model.feature_importance(importance_type="gain").tolist()))
    print("feature importance (gain):", json.dumps(
        dict(sorted(importance.items(), key=lambda kv: -kv[1])), indent=2))

    model.save_model(str(model_out))

    val_df = val_df.with_columns(pl.Series("pred", val_pred.astype(np.float32)))
    val_scored_path = config.WORK_DIR / "val_scored.parquet"
    val_df.write_parquet(val_scored_path)
    print(f"wrote {model_out} and {val_scored_path}")
    return model


if __name__ == "__main__":
    train(
        config.WORK_DIR / "train_sample.parquet",
        config.WORK_DIR / "val_full.parquet",
        config.WORK_DIR / "lgbm_matcher.txt",
    )
