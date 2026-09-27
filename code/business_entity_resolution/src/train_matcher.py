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

# Context/competition features added by add_context_features.py -- signal for
# "does this S2/S3 record also look like a good match for other S1s" and
# "how much better is this S1's best candidate than its runner-up", missing
# from the base FEATURE_COLS above (cand_rank/cand_margin there are computed
# only within each S1's own candidate list).
CONTEXT_FEATURE_COLS = [
    "ent_n_cands", "ent_sim_rank", "ent_sim_margin", "ent_sim_gap2",
    "ent_lex_rank", "ent_lex_margin",
    "s1_n_cands", "s1_lex_rank", "s1_lex_margin",
]


def load_xy(path, feature_cols):
    df = pl.read_parquet(path)
    X = df.select(feature_cols).to_numpy()
    y = df["label"].to_numpy()
    return df, X, y


def train(train_path, val_path, model_out, val_scored_out=None):
    t0 = time.time()
    cols_present = pl.read_parquet(train_path, n_rows=1).columns
    feature_cols = FEATURE_COLS + [c for c in CONTEXT_FEATURE_COLS if c in cols_present]
    print(f"using {len(feature_cols)} features: {feature_cols}")

    train_df, X_train, y_train = load_xy(train_path, feature_cols)
    val_df, X_val, y_val = load_xy(val_path, feature_cols)
    print(f"train: {X_train.shape}, positives={y_train.sum()} ({y_train.mean():.4f}); "
          f"val: {X_val.shape}, positives={y_val.sum()} ({y_val.mean():.4f})")

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_set)

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

    importance = dict(zip(feature_cols, model.feature_importance(importance_type="gain").tolist()))
    print("feature importance (gain):", json.dumps(
        dict(sorted(importance.items(), key=lambda kv: -kv[1])), indent=2))

    model.save_model(str(model_out))

    val_df = val_df.with_columns(pl.Series("pred", val_pred.astype(np.float32)))
    val_scored_path = val_scored_out or (config.WORK_DIR / "val_scored.parquet")
    val_df.write_parquet(val_scored_path)
    print(f"wrote {model_out} and {val_scored_path}")
    return model


if __name__ == "__main__":
    import sys
    suffix = "_" + sys.argv[1] if len(sys.argv) > 1 else ""
    train_path = config.WORK_DIR / f"train_sample{suffix}.parquet"
    val_path = config.WORK_DIR / f"val_full{suffix}.parquet"
    model_out = config.WORK_DIR / f"lgbm_matcher{suffix}.txt"
    val_scored_out = config.WORK_DIR / f"val_scored{suffix}.parquet"
    train(train_path, val_path, model_out, val_scored_out)
