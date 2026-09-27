"""Stage-2 stacker + decoders on top of the stage-1 LightGBM matcher preds.

Stage-2 features are aggregates of the stage-1 `pred` over each S2/S3 record
(entity_id) and over each S1 (source1_entity_id), joined back onto the rows.
Everything is computed from a narrow (hashed-key, pred) frame via group_by +
join, so it scales to the 120M-row test set with modest memory.

Usage:
  python stack_stage2.py cv                 # 2-fold CV on val, decoder sweep
  python stack_stage2.py train              # train on ALL val -> work/lgbm_stage2.txt + stage2_config.json
  python stack_stage2.py apply_test OUT.tsv # test: stage-2 preds + chosen decoder -> results TSV
  python stack_stage2.py apply_val  OUT.tsv # same pipeline on val-shaped files (smoke test)
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import config
import decode
import io_utils
from train_matcher import CONTEXT_FEATURE_COLS, FEATURE_COLS

BASE_COLS = FEATURE_COLS + CONTEXT_FEATURE_COLS
ENT_COLS = ["ent_p_max", "ent_p_second", "ent_p_gap2", "ent_p_n05", "ent_p_sum",
            "p_is_ent_argmax", "p_minus_ent_max", "p_minus_ent_other"]
S1_COLS = ["s1_p_max", "s1_p_second", "s1_p_sum", "s1_p_n05", "s1_p_n09",
           "s1_p_nargmax05", "p_minus_s1_max", "p_minus_s1_other"]
SRC_COLS = ["is_s3", "s1_src_max", "p_minus_s1_src_max", "s1_src_n05", "s1_osrc_n05", "s1_osrc_max"]
STAGE2_COLS = ["pred"] + BASE_COLS + ENT_COLS + S1_COLS + SRC_COLS

MODEL_PATH = config.WORK_DIR / "lgbm_stage2.txt"
CFG_PATH = config.WORK_DIR / "stage2_config.json"


def model_cfg_paths(tag=""):
    if not tag:
        return MODEL_PATH, CFG_PATH
    if tag == "ens":
        return config.WORK_DIR / "xgb_stage2.json", config.WORK_DIR / "stage2_ens_config.json"
    if tag == "ens3":
        return config.WORK_DIR / "xgb_stage2_c.json", config.WORK_DIR / "stage2_ens3_config.json"
    if tag == "ens2":
        return config.WORK_DIR / "xgb_stage2_b.json", config.WORK_DIR / "stage2_ens2_config.json"
    return config.WORK_DIR / f"lgbm_stage2_{tag}.txt", config.WORK_DIR / f"stage2_{tag}_config.json"
B2 = 0.25


# ---------------------------------------------------------------- features
def hash_keys(df: pl.DataFrame | pl.LazyFrame):
    return df.with_columns(pl.col("source1_entity_id").hash(seed=1).alias("hs"),
                           pl.col("entity_id").hash(seed=2).alias("he"),
                           pl.col("entity_id").str.starts_with("S3").cast(pl.Int8).alias("is_s3"))


def build_src_aggs(narrow: pl.DataFrame) -> pl.DataFrame:
    """narrow: hs, is_s3, pred -> per-S1 max / n>0.5 separately for S2 and S3 candidates."""
    p = pl.col("pred")
    return narrow.group_by("hs").agg(
        p.filter(pl.col("is_s3") == 0).max().fill_null(0.0).alias("_max_s2"),
        p.filter(pl.col("is_s3") == 1).max().fill_null(0.0).alias("_max_s3"),
        ((p > 0.5) & (pl.col("is_s3") == 0)).sum().cast(pl.Float32).alias("_n05_s2"),
        ((p > 0.5) & (pl.col("is_s3") == 1)).sum().cast(pl.Float32).alias("_n05_s3"))


def add_src_feats(df: pl.DataFrame, src: pl.DataFrame) -> pl.DataFrame:
    df = df.join(src, on="hs", how="left")
    s3 = pl.col("is_s3") == 1
    df = df.with_columns(
        pl.when(s3).then(pl.col("_max_s3")).otherwise(pl.col("_max_s2")).alias("s1_src_max"),
        pl.when(s3).then(pl.col("_n05_s3")).otherwise(pl.col("_n05_s2")).alias("s1_src_n05"),
        pl.when(s3).then(pl.col("_n05_s2")).otherwise(pl.col("_n05_s3")).alias("s1_osrc_n05"),
        pl.when(s3).then(pl.col("_max_s2")).otherwise(pl.col("_max_s3")).alias("s1_osrc_max"),
    ).with_columns((pl.col("pred") - pl.col("s1_src_max")).alias("p_minus_s1_src_max"))
    return df.drop("_max_s2", "_max_s3", "_n05_s2", "_n05_s3")


def _second(key, pcol, maxcol):
    """2nd-largest pred per key (== max if max is tied, 0 if single row)."""
    return (pl.when(pl.col("_n_at_max") > 1).then(pl.col(maxcol))
            .otherwise(pl.col("_second_raw").fill_null(0.0)))


def build_aggs(narrow: pl.DataFrame, pcol: str = "pred"):
    """narrow: hs, he, <pcol>. Returns (ent_agg keyed by he, s1_agg keyed by hs)."""
    p = pl.col(pcol)
    ent = narrow.group_by("he").agg(p.max().alias("ent_p_max"), p.sum().alias("ent_p_sum"),
                                    (p > 0.5).sum().cast(pl.UInt16).alias("ent_p_n05"))
    tmp = narrow.select("hs", "he", pcol).join(ent.select("he", "ent_p_max"), on="he", how="left")
    ent2 = tmp.group_by("he").agg(
        (p == pl.col("ent_p_max")).sum().alias("_n_at_max"),
        p.filter(p < pl.col("ent_p_max")).max().alias("_second_raw"))
    ent = ent.join(ent2, on="he", how="left").with_columns(
        _second("he", pcol, "ent_p_max").cast(pl.Float32).alias("ent_p_second")
    ).with_columns((pl.col("ent_p_max") - pl.col("ent_p_second")).alias("ent_p_gap2")
                   ).drop("_n_at_max", "_second_raw")
    del ent2
    tmp = tmp.with_columns(((p >= pl.col("ent_p_max")) & (p > 0.5)).alias("_am05"))
    s1 = tmp.group_by("hs").agg(p.max().alias("s1_p_max"), p.sum().alias("s1_p_sum"),
                                (p > 0.5).sum().cast(pl.UInt16).alias("s1_p_n05"),
                                (p > 0.9).sum().cast(pl.UInt16).alias("s1_p_n09"),
                                pl.col("_am05").sum().cast(pl.UInt16).alias("s1_p_nargmax05"))
    del tmp
    tmp = narrow.select("hs", pcol).join(s1.select("hs", "s1_p_max"), on="hs", how="left")
    s12 = tmp.group_by("hs").agg(
        (p == pl.col("s1_p_max")).sum().alias("_n_at_max"),
        p.filter(p < pl.col("s1_p_max")).max().alias("_second_raw"))
    del tmp
    s1 = s1.join(s12, on="hs", how="left").with_columns(
        _second("hs", pcol, "s1_p_max").cast(pl.Float32).alias("s1_p_second")
    ).drop("_n_at_max", "_second_raw")
    return ent, s1


def add_row_feats(df: pl.DataFrame, ent: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """df has hs, he, pred (+ base features). Joins aggs, adds row-level diffs."""
    df = df.join(ent, on="he", how="left").join(s1, on="hs", how="left")
    p = pl.col("pred")
    return df.with_columns(
        (p >= pl.col("ent_p_max")).cast(pl.Float32).alias("p_is_ent_argmax"),
        (p - pl.col("ent_p_max")).alias("p_minus_ent_max"),
        (p - pl.when(p >= pl.col("ent_p_max")).then(pl.col("ent_p_second"))
         .otherwise(pl.col("ent_p_max"))).alias("p_minus_ent_other"),
        (p - pl.col("s1_p_max")).alias("p_minus_s1_max"),
        (p - pl.when(p >= pl.col("s1_p_max")).then(pl.col("s1_p_second"))
         .otherwise(pl.col("s1_p_max"))).alias("p_minus_s1_other"),
    )


def build_stage2_features(df_with_pred: pl.DataFrame) -> pl.DataFrame:
    """df_with_pred: source1_entity_id, entity_id, pred, base features.
    Returns df + hs/he + stage-2 feature columns (same row order)."""
    df = hash_keys(df_with_pred)
    ent, s1 = build_aggs(df.select("hs", "he", "pred"))
    src = build_src_aggs(df.select("hs", "is_s3", "pred"))
    df = df.with_row_index("_ri")
    out = add_src_feats(add_row_feats(df, ent, s1), src).sort("_ri").drop("_ri")
    return out


# ---------------------------------------------------------------- scoring
def fast_macro_f(assigned: pl.DataFrame, gt_pairs: pl.DataFrame, gt_s1: pl.DataFrame) -> dict:
    """assigned: source1_entity_id, entity_id. gt_pairs: exploded gold pairs.
    gt_s1: source1_entity_id, country, n_gold (all val S1s)."""
    npred = assigned.group_by("source1_entity_id").agg(pl.len().alias("n_pred"))
    tp = assigned.join(gt_pairs, on=["source1_entity_id", "entity_id"], how="inner") \
        .group_by("source1_entity_id").agg(pl.len().alias("tp"))
    j = gt_s1.join(npred, on="source1_entity_id", how="left").join(tp, on="source1_entity_id", how="left") \
        .with_columns(pl.col("n_pred").fill_null(0), pl.col("tp").fill_null(0))
    j = j.with_columns(
        pl.when((pl.col("n_gold") == 0) & (pl.col("n_pred") == 0)).then(1.0)
        .otherwise((1 + B2) * pl.col("tp") / (B2 * pl.col("n_gold") + pl.col("n_pred")).clip(lower_bound=1e-9))
        .alias("f"))
    res = {"all": round(j["f"].mean(), 5)}
    for r in j.group_by("country").agg(pl.col("f").mean()).sort("country").iter_rows():
        res[r[0]] = round(r[1], 5)
    return res


def load_gt():
    from evaluate import build_val_ground_truth
    gt, val_ids = build_val_ground_truth()
    gt = gt.filter(pl.col("source1_entity_id").is_in(val_ids))
    pairs = gt.select("source1_entity_id", pl.col("matched_entity_ids").str.split(",").alias("entity_id")) \
        .explode("entity_id").filter(pl.col("entity_id").is_not_null() & (pl.col("entity_id") != ""))
    n = pairs.group_by("source1_entity_id").agg(pl.len().alias("n_gold"))
    gt_s1 = gt.select("source1_entity_id", "country").join(n, on="source1_entity_id", how="left") \
        .with_columns(pl.col("n_gold").fill_null(0))
    return pairs, gt_s1, gt, val_ids


# ---------------------------------------------------------------- decoders
def argmax_rows(df: pl.DataFrame, pcol: str) -> pl.DataFrame:
    """One row per entity: its argmax S1 (ties broken arbitrarily but deterministically)."""
    return df.sort([pcol, "source1_entity_id"], descending=[True, False]).unique(
        subset=["entity_id"], keep="first")


def decode_rows(am: pl.DataFrame, pcol: str, cfg: dict) -> pl.DataFrame:
    """am: entity-argmax rows (source1_entity_id, entity_id, pcol).
    cfg: {"decoder": "tau"|"tau_single"|"expf", "tau":..., "tau_s":..., "floor":...}.
    Returns kept rows (source1_entity_id, entity_id, pcol)."""
    d = cfg["decoder"]
    p = pl.col(pcol)
    r = cfg.get("prior_r", 1.0)
    if r != 1.0:  # prior shift for a distractor-heavier population: p' = r p / (r p + 1 - p)
        am = am.with_columns((r * p / (r * p + 1 - p)).alias(pcol))
    if d == "tau":
        return am.filter(p >= cfg["tau"])
    if d == "tau_single":
        kept = am.filter(p >= cfg["tau"])
        return kept.filter(p.max().over("source1_entity_id") >= cfg["tau_s"])
    if d == "expf":
        c = am.filter(p >= cfg.get("floor", 0.02)).sort([ "source1_entity_id", pcol],
                                                        descending=[False, True])
        c = c.with_columns(
            p.cum_sum().over("source1_entity_id").alias("_cs"),
            pl.int_range(1, pl.len() + 1).over("source1_entity_id").alias("_k"),
            (p.sum().over("source1_entity_id") * cfg.get("etrue_scale", 1.0)).alias("_et"),
            (1 - p).log().sum().over("source1_entity_id").exp().alias("_ef0"),
        ).with_columns(((1 + B2) * pl.col("_cs") / (B2 * pl.col("_et") + pl.col("_k"))).alias("_ef"))
        best = c.group_by("source1_entity_id").agg(
            pl.col("_ef").max().alias("_bef"),
            pl.col("_k").get(pl.col("_ef").arg_max()).alias("_bk"),
            pl.col("_ef0").first().alias("_ef0"))
        best = best.filter(pl.col("_bef") * cfg.get("gain", 1.0) > pl.col("_ef0"))
        return c.join(best.select("source1_entity_id", "_bk"), on="source1_entity_id", how="inner") \
            .filter(pl.col("_k") <= pl.col("_bk")).select(am.columns)
    raise ValueError(d)


def decoder_grid():
    g = [{"decoder": "tau", "tau": t} for t in np.round(np.arange(0.60, 0.99, 0.02), 2)]
    for t in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]:
        for ts in [0.85, 0.9, 0.93, 0.95, 0.97]:
            if ts > t:
                g.append({"decoder": "tau_single", "tau": t, "tau_s": ts})
    for fl in [0.05, 0.2, 0.4]:
        for sc in [1.0]:
            for gain in [0.6, 0.8, 1.0, 1.25, 1.6]:
                g.append({"decoder": "expf", "floor": fl, "etrue_scale": sc, "gain": gain})
    return g


def sweep(df, pcol, pairs, gt_s1, label):
    src = df.filter(pl.col("full_argmax")) if "full_argmax" in df.columns else df
    am = argmax_rows(src.select("source1_entity_id", "entity_id", pcol), pcol)
    rows = []
    for cfg in decoder_grid():
        r = fast_macro_f(decode_rows(am, pcol, cfg), pairs, gt_s1)
        rows.append((cfg, r))
    best = {}
    for cfg, r in rows:
        d = cfg["decoder"]
        if d not in best or r["all"] > best[d][1]["all"]:
            best[d] = (cfg, r)
    for d, (cfg, r) in best.items():
        print(f"  [{label}] best {d}: {r}  cfg={cfg}", flush=True)
    return rows, best


def sweep_prior(df, pcol, pairs, gt_s1, label):
    src = df.filter(pl.col("full_argmax")) if "full_argmax" in df.columns else df
    am = argmax_rows(src.select("source1_entity_id", "entity_id", pcol), pcol)
    grid = [{"decoder": "tau", "tau": float(t)} for t in np.round(np.arange(0.40, 0.99, 0.02), 2)]
    for r in [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2]:
        for fl in [0.2, 0.4]:
            for g in [0.7, 0.8, 1.0]:
                grid.append({"decoder": "expf", "floor": fl, "gain": g, "etrue_scale": 1.0, "prior_r": r})
    rows = [(c, fast_macro_f(decode_rows(am, pcol, c), pairs, gt_s1)) for c in grid]
    best = {}
    for c, r in rows:
        key = c["decoder"] + (f"_r{c['prior_r']}" if "prior_r" in c else "")
        if key not in best or r["all"] > best[key][1]["all"]:
            best[key] = (c, r)
    for k, (c, r) in best.items():
        print(f"  [{label}] best {k}: {r} cfg={c}", flush=True)
    return rows, best


# ---------------------------------------------------------------- CV / train
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, verbose=-1,
              seed=config.RANDOM_SEED, num_threads=10)


CTX_NARROW = config.WORK_DIR / "stage2_val_ctx_narrow.parquet"


def load_val_feats():
    """Val rows + stage-2 features. If the full-context narrow frame exists
    (stage2_val_context.py: stage-1 preds for ALL train pairs of val entities),
    entity aggregates see every S1 competitor as on test; `full_argmax` flags
    rows that are the stage-1 argmax among ALL candidates of their entity."""
    df = pl.read_parquet(config.WORK_DIR / "val_scored_ctx.parquet",
                         columns=["source1_entity_id", "entity_id", "label", "pred"] + BASE_COLS)
    if not CTX_NARROW.exists():
        print("NOTE: no full-context narrow frame; entity windows are val-only (optimistic)")
        return build_stage2_features(df).with_columns(pl.lit(True).alias("full_argmax"))
    narrow = pl.read_parquet(CTX_NARROW)
    ent, s1 = build_aggs(narrow.select("hs", "he", "pred"))
    other_max = narrow.filter(~pl.col("is_val")).group_by("he").agg(pl.col("pred").max().alias("_omax"))
    del narrow
    df = hash_keys(df).with_row_index("_ri")
    src = build_src_aggs(df.select("hs", "is_s3", "pred"))  # val S1 windows are complete
    df = add_src_feats(add_row_feats(df, ent, s1), src).join(other_max, on="he", how="left").sort("_ri").drop("_ri")
    return df.with_columns((pl.col("pred") >= pl.col("_omax").fill_null(-1.0)).alias("full_argmax")).drop("_omax")


def cv(feature_set="full", rounds=400):
    t0 = time.time()
    pairs, gt_s1, _, _ = load_gt()
    df = load_val_feats()
    print(f"features built {df.shape} ({time.time()-t0:.0f}s)", flush=True)
    cols = STAGE2_COLS if feature_set == "full" else [c for c in STAGE2_COLS if c not in ENT_COLS]
    fold = (pl.col("hs") % 2).cast(pl.Int8)
    df = df.with_columns(fold.alias("fold"))
    print("baseline (stage-1 pred):", flush=True)
    sweep(df, "pred", pairs, gt_s1, "stage1")
    p2 = np.zeros(df.height, dtype=np.float32)
    fold_np = df["fold"].to_numpy()
    X = df.select(cols).to_numpy().astype(np.float32)
    y = df["label"].to_numpy()
    iters = []
    for k in (0, 1):
        tr, te = fold_np != k, fold_np == k
        dtr = lgb.Dataset(X[tr], label=y[tr], feature_name=cols, free_raw_data=True)
        dte = lgb.Dataset(X[te], label=y[te], reference=dtr)
        m = lgb.train({**PARAMS, "metric": "binary_logloss"}, dtr, num_boost_round=rounds,
                      valid_sets=[dte], callbacks=[lgb.early_stopping(30, verbose=False)])
        iters.append(m.best_iteration)
        p2[te] = m.predict(X[te], num_iteration=m.best_iteration)
        print(f"fold {k}: best_iter={m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
        if k == 0:
            imp = dict(zip(cols, m.feature_importance("gain")))
            print("top gain:", sorted(imp.items(), key=lambda kv: -kv[1])[:15], flush=True)
    del X
    df = df.with_columns(pl.Series("p2", p2))
    print(f"stage2 [{feature_set}]:", flush=True)
    rows, best = sweep(df, "p2", pairs, gt_s1, f"stage2-{feature_set}")
    df.select("source1_entity_id", "entity_id", "label", "pred", "p2", "fold", "full_argmax").write_parquet(
        config.WORK_DIR / f"stage2_cv_{feature_set}{'_ctx' if CTX_NARROW.exists() else ''}.parquet")
    print(f"done ({time.time()-t0:.0f}s) iters={iters}")
    return best, iters


def train_all(feature_set, rounds, decoder_cfg):
    df = load_val_feats()
    cols = STAGE2_COLS if feature_set == "full" else [c for c in STAGE2_COLS if c not in ENT_COLS]
    X = df.select(cols).to_numpy().astype(np.float32)
    m = lgb.train({**PARAMS, "num_threads": 10}, lgb.Dataset(X, label=df["label"].to_numpy(),
                                                             feature_name=cols), num_boost_round=rounds)
    m.save_model(str(MODEL_PATH))
    cfg = {"use_stage2": True, "feature_cols": cols, "decoder": decoder_cfg, "rounds": rounds}
    CFG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {MODEL_PATH} and {CFG_PATH}")


# ---------------------------------------------------------------- OOF-train path
def train_oof(oof_path, feats_path, train_frac=0.12, rounds=300, save=True, tag=""):
    """Stage-2 on out-of-fold stage-1 preds for ALL train pairs (windows see every
    S1 competitor, as on test). Trains on a random `train_frac` of non-val train
    S1s, evaluates decoders on the held-out val S1s, saves model + config."""
    t0 = time.time()
    pairs, gt_s1, _, val_ids = load_gt()
    narrow = hash_keys(pl.scan_parquet(oof_path).select("source1_entity_id", "entity_id", "pred", "label"))         .select("hs", "he", "is_s3", "pred", "label").collect()
    print(f"oof narrow {narrow.shape} ({time.time()-t0:.0f}s)", flush=True)
    ent, s1 = build_aggs(narrow.select("hs", "he", "pred"))
    src = build_src_aggs(narrow.select("hs", "is_s3", "pred"))
    val_h = pl.DataFrame({"source1_entity_id": val_ids}).select(pl.col("source1_entity_id").hash(seed=1).alias("hs"))
    all_h = narrow.select("hs").unique()
    rng_mask = (all_h["hs"] % 1000) < int(train_frac * 1000)
    tr_h = all_h.filter(rng_mask).join(val_h, on="hs", how="anti")
    sel = pl.concat([tr_h.with_columns(pl.lit(0, pl.Int8).alias("is_eval")),
                     val_h.with_columns(pl.lit(1, pl.Int8).alias("is_eval"))])
    narrow = narrow.join(sel, on="hs", how="inner")
    print(f"selected rows {narrow.height} (train S1 {tr_h.height}, val S1 {val_h.height}) "
          f"({time.time()-t0:.0f}s)", flush=True)
    pf = pq.ParquetFile(str(feats_path))
    parts = []
    for batch in pf.iter_batches(batch_size=4_000_000, columns=["source1_entity_id", "entity_id", *BASE_COLS]):
        fb = hash_keys(pl.from_arrow(pa.Table.from_batches([batch])))
        fb = fb.join(narrow.select("hs", "he", "pred", "label", "is_eval"), on=["hs", "he"], how="inner")
        if fb.height:
            fb = add_src_feats(add_row_feats(fb.drop("is_s3").with_columns(
                pl.col("entity_id").str.starts_with("S3").cast(pl.Int8).alias("is_s3")), ent, s1), src)
            keep = ["source1_entity_id", "entity_id", "label", "is_eval"] + STAGE2_COLS
            parts.append(fb.select(keep).with_columns(pl.col(pl.Float64).cast(pl.Float32)))
    del narrow, ent, s1, src
    df = pl.concat(parts)
    del parts
    print(f"features joined {df.shape} ({time.time()-t0:.0f}s)", flush=True)
    df = df.with_columns((pl.col("p_is_ent_argmax") > 0).alias("full_argmax"))
    tr = df.filter(pl.col("is_eval") == 0)
    ev = df.filter(pl.col("is_eval") == 1)
    del df
    print("baseline (oof stage-1 pred, full-set argmax) on val S1s:", flush=True)
    sweep(ev, "pred", pairs, gt_s1, "stage1-oof")
    params = {**PARAMS, "metric": "binary_logloss"}
    if tag:  # big run: early-stop on a held-out slice of TRAIN S1s (not val)
        params.update(min_data_in_leaf=500, num_threads=18)
        es_mask = ((tr["source1_entity_id"].hash(seed=7) % 20) == 0).to_numpy()
        ytr = tr["label"].to_numpy()
        Xtr = tr.select(STAGE2_COLS).to_numpy().astype(np.float32)
        del tr
        dtr = lgb.Dataset(Xtr[~es_mask], label=ytr[~es_mask], feature_name=STAGE2_COLS)
        dho = lgb.Dataset(Xtr[es_mask], label=ytr[es_mask], reference=dtr)
        dtr.construct(); dho.construct()
        del Xtr
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dho],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(25)])
        print(f"best_iter={m.best_iteration}", flush=True)
    else:
        dtr = lgb.Dataset(tr.select(STAGE2_COLS).to_numpy().astype(np.float32), label=tr["label"].to_numpy(),
                          feature_name=STAGE2_COLS)
        del tr
        m = lgb.train(params, dtr, num_boost_round=rounds)
    del dtr
    Xev = ev.select(STAGE2_COLS).to_numpy().astype(np.float32)
    ev = ev.with_columns(pl.Series("p2", m.predict(Xev, num_iteration=m.best_iteration or None).astype(np.float32)))
    print(f"trained ({time.time()-t0:.0f}s)", flush=True)
    rows, best = sweep(ev, "p2", pairs, gt_s1, "stage2-oof")
    ev.select("source1_entity_id", "entity_id", "label", "pred", "p2", "full_argmax").write_parquet(
        config.WORK_DIR / "stage2_oof_eval.parquet")
    if save:
        mp, cp = model_cfg_paths(tag)
        m.save_model(str(mp), num_iteration=m.best_iteration or None)
        top = max(rows, key=lambda r: r[1]["all"])
        cfg = {"use_stage2": True, "feature_cols": STAGE2_COLS,
               "decoder": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in top[0].items()},
               "cv": top[1], "trained_on": str(oof_path)}
        cp.write_text(json.dumps(cfg, indent=2))
        print(f"saved {mp}, {cp}: {cfg['decoder']} {top[1]}", flush=True)
    return best


def train_big(oof_path, feats_path, train_frac=0.25, rounds=800, tag="big", drop_frac=0.0, eval_models=(),
              learner="lgb"):
    """Memory-lean big stage-2: train rows kept only as float32 numpy per batch
    (no id strings), early stopping on a 5% held-out slice of TRAIN S1s,
    decoders evaluated on the val S1s. Saves lgbm_stage2_<tag>.txt + stage2_<tag>_config.json."""
    t0 = time.time()
    pairs, gt_s1, _, val_ids = load_gt()
    lf = hash_keys(pl.scan_parquet(oof_path).select("source1_entity_id", "entity_id", "pred", "label"))
    if drop_frac > 0:  # test-like: drop a random share of NON-val S1s -> their records become distractors
        lf = lf.filter(~((pl.col("source1_entity_id").hash(seed=7) % 1000 < int(drop_frac * 1000))
                         & ~pl.col("source1_entity_id").is_in(pl.Series(val_ids))))
    narrow = lf.select("hs", "he", "is_s3", "pred", "label").collect()
    print(f"narrow rows {narrow.height} (drop_frac={drop_frac})", flush=True)
    ent, s1 = build_aggs(narrow.select("hs", "he", "pred"))
    src = build_src_aggs(narrow.select("hs", "is_s3", "pred"))
    val_h = pl.DataFrame({"source1_entity_id": val_ids}).select(pl.col("source1_entity_id").hash(seed=1).alias("hs"))
    all_h = narrow.select("hs").unique()
    import os
    off = int(os.environ.get("XGB_OFFSET", "0"))  # different train-S1 sample per model
    tr_h = all_h.filter(((all_h["hs"] % 1000 + off) % 1000) < int(train_frac * 1000)).join(val_h, on="hs", how="anti")
    # role: 0 train, 1 val-eval, 2 early-stop holdout (train S1s)
    tr_h = tr_h.with_columns(pl.when((pl.col("hs") // 1000) % 20 == 0).then(2).otherwise(0).cast(pl.Int8).alias("role"))
    sel = pl.concat([tr_h, val_h.with_columns(pl.lit(1, pl.Int8).alias("role"))])
    narrow = narrow.join(sel, on="hs", how="inner").select("hs", "he", "pred", "label", "role")
    print(f"selected rows {narrow.height} train S1 {tr_h.height} ({time.time()-t0:.0f}s)", flush=True)
    del all_h, tr_h, sel
    pf = pq.ParquetFile(str(feats_path))
    cnt = dict(narrow.group_by("role").len().iter_rows())
    F = len(STAGE2_COLS)
    X = {r: np.empty((cnt.get(r, 0), F), dtype=np.float32) for r in (0, 2)}
    Y = {r: np.empty(cnt.get(r, 0), dtype=np.int8) for r in (0, 2)}
    pos = {0: 0, 2: 0}; ev_parts = []
    for batch in pf.iter_batches(batch_size=4_000_000, columns=["source1_entity_id", "entity_id", *BASE_COLS]):
        fb = hash_keys(pl.from_arrow(pa.Table.from_batches([batch])))
        fb = fb.join(narrow, on=["hs", "he"], how="inner")
        if not fb.height:
            continue
        fb = add_src_feats(add_row_feats(fb, ent, s1), src)
        for r in (0, 2):
            part = fb.filter(pl.col("role") == r)
            k = part.height
            X[r][pos[r]:pos[r] + k] = part.select(STAGE2_COLS).to_numpy().astype(np.float32)
            Y[r][pos[r]:pos[r] + k] = part["label"].to_numpy()
            pos[r] += k
        ev_parts.append(fb.filter(pl.col("role") == 1).select(
            ["source1_entity_id", "entity_id", "label"] + STAGE2_COLS).with_columns(pl.col(pl.Float64).cast(pl.Float32)))
        del fb
    assert pos[0] == len(Y[0]) and pos[2] == len(Y[2]), (pos, cnt)
    del narrow, ent, s1, src
    ev = pl.concat(ev_parts); del ev_parts
    Xtr, ytr, Xho, yho = X[0], Y[0], X[2], Y[2]
    del X, Y
    print(f"train {Xtr.shape} holdout {Xho.shape} eval {ev.height} ({time.time()-t0:.0f}s)", flush=True)
    ev = ev.with_columns((pl.col("p_is_ent_argmax") > 0).alias("full_argmax"))
    Xev = ev.select(STAGE2_COLS).to_numpy().astype(np.float32)
    if learner == "xgb":
        return _train_xgb_and_ensemble(Xtr, ytr, Xho, yho, Xev, ev, pairs, gt_s1, rounds, t0)
    print("stage-1 pred on simulated eval:", flush=True)
    sweep_prior(ev, "pred", pairs, gt_s1, "stage1")
    for mname in eval_models:
        em = lgb.Booster(model_file=str(config.WORK_DIR / mname))
        ev = ev.with_columns(pl.Series("p_old", em.predict(Xev).astype(np.float32)))
        print(f"existing model {mname} on simulated eval:", flush=True)
        sweep_prior(ev, "p_old", pairs, gt_s1, mname)
    if rounds == 0:
        return
    params = {**PARAMS, "metric": "binary_logloss", "min_data_in_leaf": 500, "num_threads": 18}
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=STAGE2_COLS)
    dho = lgb.Dataset(Xho, label=yho, reference=dtr)
    dtr.construct(); dho.construct()
    del Xtr, Xho
    m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dho],
                  callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)])
    print(f"best_iter={m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    del dtr, dho
    ev = ev.with_columns(pl.Series("p2", m.predict(Xev, num_iteration=m.best_iteration).astype(np.float32)))
    rows, best = sweep_prior(ev, "p2", pairs, gt_s1, f"stage2-{tag}")
    ev.select("source1_entity_id", "entity_id", "label", "pred", "p2", "full_argmax").write_parquet(
        config.WORK_DIR / f"stage2_{tag}_eval.parquet")
    mp, cp = model_cfg_paths(tag)
    m.save_model(str(mp), num_iteration=m.best_iteration)
    top = max(rows, key=lambda r: r[1]["all"])
    cfg = {"use_stage2": True, "feature_cols": STAGE2_COLS,
           "decoder": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in top[0].items()},
           "cv": top[1], "trained_on": str(oof_path), "best_iter": m.best_iteration}
    cp.write_text(json.dumps(cfg, indent=2))
    print(f"saved {mp}, {cp}: {cfg['decoder']} {top[1]} ({time.time()-t0:.0f}s)", flush=True)


XGB_PATH = config.WORK_DIR / "xgb_stage2.json"
ENS_LGB = ["lgbm_stage2_final.txt", "lgbm_stage2_big.txt"]


def small_sweep(ev, pcol, pairs, gt_s1, label):
    src = ev.filter(pl.col("full_argmax"))
    am = argmax_rows(src.select("source1_entity_id", "entity_id", pcol), pcol)
    grid = [{"decoder": "tau", "tau": t} for t in (0.6, 0.64, 0.68, 0.72)]
    grid += [{"decoder": "expf", "floor": f, "gain": g, "etrue_scale": 1.0, "prior_r": 1.0}
             for f in (0.2, 0.4) for g in (0.7, 0.8, 0.9)]
    rows = [(c, fast_macro_f(decode_rows(am, pcol, c), pairs, gt_s1)) for c in grid]
    top = max(rows, key=lambda r: r[1]["all"])
    print(f"  [{label}] best {top[1]} cfg={top[0]}", flush=True)
    return top


def _train_xgb_and_ensemble(Xtr, ytr, Xho, yho, Xev, ev, pairs, gt_s1, rounds, t0):
    import os
    import xgboost as xgb
    mb = 512 if (os.environ.get("XGB2") == "1" or os.environ.get("XGB3") == "1") else 256
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=mb,
                              feature_names=STAGE2_COLS)
    dho = xgb.QuantileDMatrix(Xho, label=yho, ref=dtr, feature_names=STAGE2_COLS,
                              max_bin=mb)
    del Xtr, Xho
    import os
    v3 = os.environ.get("XGB3") == "1"
    v2 = os.environ.get("XGB2") == "1" or v3
    params = dict(objective="binary:logistic", eval_metric="logloss", device="cuda", tree_method="hist",
                  max_depth=10 if v2 else 9, eta=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=5, max_bin=512 if v2 else 256,
                  seed=11 if v3 else (7 if v2 else config.RANDOM_SEED))
    xgb_path = config.WORK_DIR / ("xgb_stage2_c.json" if v3 else "xgb_stage2_b.json") if v2 else XGB_PATH
    ens_cfg_path = config.WORK_DIR / ("stage2_ens3_config.json" if v3 else
                                      "stage2_ens2_config.json" if v2 else "stage2_ens_config.json")
    cbs = []
    stop_at = os.environ.get("XGB_STOP_AT")  # "HH:MM" local wall-clock budget
    if stop_at:
        import datetime as _dt

        class _WallStop(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                return _dt.datetime.now().strftime("%H:%M") >= stop_at
        cbs.append(_WallStop())
    cont = os.environ.get("XGB_CONT")  # "b" or "c": continue a saved booster
    if cont:
        base = xgb.Booster(); base.load_model(str(config.WORK_DIR / f"xgb_stage2_{cont}.json"))
        bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dho, "ho")],
                        early_stopping_rounds=50, verbose_eval=100, callbacks=cbs, xgb_model=base)
        print(f"cont best_iter={bst.best_iteration} total={bst.num_boosted_rounds()} ({time.time()-t0:.0f}s)", flush=True)
        del dtr, dho
        bst.save_model(str(config.WORK_DIR / f"xgb_stage2_{cont}2.json"))
        bst.set_param({"device": "cuda"})
        np.save(config.WORK_DIR / f"ev_pred_{cont}2.npy",
                bst.inplace_predict(Xev, iteration_range=(0, bst.best_iteration + 1)).astype(np.float32))
        ev.select("source1_entity_id", "entity_id", "full_argmax").write_parquet(config.WORK_DIR / "ev_ids.parquet")
        print(f"cont saved ({time.time()-t0:.0f}s)", flush=True)
        return
    bst = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dho, "ho")],
                    early_stopping_rounds=50, verbose_eval=100, callbacks=cbs)
    print(f"xgb best_iter={bst.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    del dtr, dho
    bst.save_model(str(xgb_path))
    bst.set_param({"device": "cuda"})
    preds = {"xgb": bst.inplace_predict(Xev, iteration_range=(0, bst.best_iteration + 1)).astype(np.float32)}
    if v2:
        b1 = xgb.Booster(); b1.load_model(str(XGB_PATH)); b1.set_param({"device": "cuda"})
        preds["xgb1"] = b1.inplace_predict(Xev).astype(np.float32)
    if v3:
        b2 = xgb.Booster(); b2.load_model(str(config.WORK_DIR / "xgb_stage2_b.json")); b2.set_param({"device": "cuda"})
        preds["xgb2"] = b2.inplace_predict(Xev).astype(np.float32)
    for mn in ENS_LGB:
        preds[mn] = lgb.Booster(model_file=str(config.WORK_DIR / mn)).predict(Xev).astype(np.float32)
    print(f"eval preds done ({time.time()-t0:.0f}s)", flush=True)
    del Xev
    combos = {"xgb": {"xgb": 1}, "final": {ENS_LGB[0]: 1},
              "final+xgb": {ENS_LGB[0]: 1, "xgb": 1},
              "final+big+xgb": {ENS_LGB[0]: 1, ENS_LGB[1]: 1, "xgb": 1},
              "final+2xgb": {ENS_LGB[0]: 1, "xgb": 2}, "2final+xgb": {ENS_LGB[0]: 2, "xgb": 1}}
    if v2:
        combos = {"xgb2": {"xgb": 1}, "xgb1+xgb2": {"xgb1": 1, "xgb": 1},
                  "xgb1+xgb2+final": {"xgb1": 1, "xgb": 1, ENS_LGB[0]: 0.5}}
    if v3:
        combos = {"xgb3": {"xgb": 1}, "xgb1+xgb2+xgb3": {"xgb1": 1, "xgb2": 1, "xgb": 1},
                  "xgb2+xgb3": {"xgb2": 1, "xgb": 1}, "xgb1+xgb3": {"xgb1": 1, "xgb": 1},
                  "xgb1+xgb2": {"xgb1": 1, "xgb2": 1}}
    results = {}
    for name, w in combos.items():
        pe = sum(preds[k] * v for k, v in w.items()) / sum(w.values())
        ev = ev.with_columns(pl.Series("pe", pe.astype(np.float32)))
        results[name] = (w, small_sweep(ev, "pe", pairs, gt_s1, name))
    name, (w, (dec, score)) = max(results.items(), key=lambda kv: kv[1][1][1]["all"])
    cfg = {"use_stage2": True, "feature_cols": STAGE2_COLS, "decoder": dec, "cv": score,
           "ensemble": {"lgb": {k: v for k, v in w.items() if k not in ("xgb", "xgb1", "xgb2")},
                        "xgb": {xgb_path.name: w.get("xgb", 0), XGB_PATH.name: w.get("xgb1", 0),
                                "xgb_stage2_b.json": w.get("xgb2", 0)} if v3 else
                        {xgb_path.name: w.get("xgb", 0), XGB_PATH.name: w.get("xgb1", 0)}
                        if v2 else {XGB_PATH.name: w.get("xgb", 0)}}, "combo": name,
           "all_results": {k: v[1][1] for k, v in results.items()}}
    ens_cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"ENS best={name} {score} dec={dec} ({time.time()-t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- final decode / apply
def decode_final(df_with_stage2_pred: pl.DataFrame, cfg: dict | None = None,
                 pcol: str = "p2") -> pl.DataFrame:
    """Rows: source1_entity_id, entity_id, <pcol>. Returns kept assignments."""
    cfg = cfg or json.loads(CFG_PATH.read_text())
    am = argmax_rows(df_with_stage2_pred.select("source1_entity_id", "entity_id", pcol), pcol)
    return decode_rows(am, pcol, cfg["decoder"])


def apply(scored_path, feats_path, s1_ids, out_path, batch_rows=4_000_000, tag=""):
    """Memory-lean: hashed keys + preds (~2.5GB for 120M rows), features streamed in batches."""
    t0 = time.time()
    model_path, cfg_path = model_cfg_paths(tag)
    cfg = json.loads(cfg_path.read_text())
    use2 = cfg.get("use_stage2", True)
    print(f"model={model_path.name} cfg={cfg_path.name} decoder={cfg['decoder']}", flush=True)
    narrow = hash_keys(pl.scan_parquet(scored_path).select("source1_entity_id", "entity_id", "pred")) \
        .select("hs", "he", "is_s3", "pred").collect()
    n = narrow.height
    print(f"loaded {n} scored rows ({time.time()-t0:.0f}s)", flush=True)
    if use2:
        cols = cfg["feature_cols"]
        if "ensemble" in cfg:
            ens = [(lgb.Booster(model_file=str(config.WORK_DIR / k)), w, "lgb") for k, w in cfg["ensemble"]["lgb"].items() if w]
            import xgboost as xgb
            for k, w in cfg["ensemble"]["xgb"].items():
                if w:
                    b = xgb.Booster(); b.load_model(str(config.WORK_DIR / k)); b.set_param({"device": "cuda"})
                    ens.append((b, w, "xgb"))
            wsum = sum(w for _, w, _ in ens)

            def predict(X):
                return sum(w * (m.predict(X) if kind == "lgb" else m.inplace_predict(X)) for m, w, kind in ens) / wsum
        else:
            model = lgb.Booster(model_file=str(model_path))

            def predict(X):
                return model.predict(X)
        ent, s1 = build_aggs(narrow.select("hs", "he", "pred"))
        src = build_src_aggs(narrow.select("hs", "is_s3", "pred"))
        print(f"aggs built ent={ent.height} s1={s1.height} ({time.time()-t0:.0f}s)", flush=True)
        base_needed = [c for c in cols if c in BASE_COLS]
        pf = pq.ParquetFile(str(feats_path))
        assert pf.metadata.num_rows == n, "row count mismatch scored vs features"
        p2 = np.empty(n, dtype=np.float32)
        off = 0
        for batch in pf.iter_batches(batch_size=batch_rows,
                                     columns=["source1_entity_id", "entity_id", *base_needed]):
            fb = hash_keys(pl.from_arrow(pa.Table.from_batches([batch]))).drop(
                "source1_entity_id", "entity_id")
            m = fb.height
            sl = narrow.slice(off, m)
            if not (sl["hs"].equals(fb["hs"]) and sl["he"].equals(fb["he"])):
                # score_matcher preserves feature-file row order, so this should not
                # happen; fall back to a key join for this batch if it does.
                print("  WARNING: row order differs in this batch; joining on keys", flush=True)
                fb = sl.select("hs", "he").join(fb, on=["hs", "he"], how="left", maintain_order="left")
                assert fb.height == m and fb[base_needed[0]].null_count() == 0, "key join failed"
            fb = fb.with_columns(sl["pred"])
            fb = add_src_feats(add_row_feats(fb, ent, s1), src)
            p2[off:off + m] = predict(fb.select(cols).to_numpy().astype(np.float32))
            off += m
            del fb, sl
            print(f"  stage2 predicted {off}/{n} ({time.time()-t0:.0f}s)", flush=True)
        del ent, s1, src
        narrow = narrow.with_columns(pl.Series("p2", p2))
        del p2
    else:
        narrow = narrow.with_columns(pl.col("pred").alias("p2"))
    # decode on hashed keys with row index, then map kept rows back to string ids
    narrow = narrow.with_row_index("_ri").rename({"hs": "source1_entity_id", "he": "entity_id"})
    am = narrow.select("_ri", "source1_entity_id", "entity_id", "p2").sort(
        ["p2", "source1_entity_id"], descending=[True, False]).unique(subset=["entity_id"], keep="first")
    del narrow
    kept_idx = decode_rows(am, "p2", cfg["decoder"])["_ri"].sort()
    del am
    ids = pl.scan_parquet(scored_path).select("source1_entity_id", "entity_id").with_row_index("_ri") \
        .join(pl.LazyFrame({"_ri": kept_idx}), on="_ri", how="semi").collect()
    out = decode.to_result_frame(s1_ids, ids.select("source1_entity_id", "entity_id"))
    io_utils.write_tsv(out, out_path)
    n_empty = (out["matched_entity_ids"] == "").sum()
    print(f"wrote {out_path}: {out.height} S1 rows, {ids.height} pairs, singleton rate "
          f"{n_empty/out.height:.4f}, cfg={cfg['decoder']} stage2={use2} ({time.time()-t0:.0f}s)")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "cv":
        cv(sys.argv[2] if len(sys.argv) > 2 else "full")
    elif cmd == "train":
        train_all(sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4]))
    elif cmd == "train_oof":
        train_oof(config.WORK_DIR / "oof_train_scored.parquet", config.WORK_DIR / "pairs_features_train_ctx.parquet",
                  train_frac=float(sys.argv[2]) if len(sys.argv) > 2 else 0.12,
                  rounds=int(sys.argv[3]) if len(sys.argv) > 3 else 300,
                  tag=sys.argv[4] if len(sys.argv) > 4 else "")
    elif cmd == "train_big":
        train_big(config.WORK_DIR / "oof_train_scored.parquet", config.WORK_DIR / "pairs_features_train_ctx.parquet",
                  train_frac=float(sys.argv[2]) if len(sys.argv) > 2 else 0.25)
    elif cmd == "train_drop":
        # python stack_stage2.py train_drop DROP_FRAC ROUNDS(0=eval only) [TAG]
        train_big(config.WORK_DIR / "oof_train_scored.parquet", config.WORK_DIR / "pairs_features_train_ctx.parquet",
                  train_frac=0.3, drop_frac=float(sys.argv[2]), rounds=int(sys.argv[3]),
                  tag=sys.argv[4] if len(sys.argv) > 4 else "drop",
                  eval_models=("lgbm_stage2_big.txt",))
    elif cmd == "train_xgb":
        # python stack_stage2.py train_xgb [TRAIN_FRAC] [ROUNDS]
        train_big(config.WORK_DIR / "oof_train_scored.parquet", config.WORK_DIR / "pairs_features_train_ctx.parquet",
                  train_frac=float(sys.argv[2]) if len(sys.argv) > 2 else 0.35, drop_frac=0.21,
                  rounds=int(sys.argv[3]) if len(sys.argv) > 3 else 2000, tag="xgb", learner="xgb")
    elif cmd == "apply_test":
        # python stack_stage2.py apply_test OUT.tsv [scored parquet in work/ (default test_scored_ctx.parquet)] [tag e.g. big]
        scored = sys.argv[3] if len(sys.argv) > 3 else "test_scored_ctx.parquet"
        tag = sys.argv[4] if len(sys.argv) > 4 else ""
        s1_ids = pl.read_parquet(config.WORK_DIR / "norm_test_s1.parquet", columns=["entity_id"])["entity_id"].to_list()
        apply(config.WORK_DIR / scored, config.WORK_DIR / "pairs_features_test_ctx.parquet",
              s1_ids, sys.argv[2], tag=tag)
    elif cmd == "apply_val":
        _, _, _, val_ids = load_gt()
        apply(sys.argv[3], sys.argv[4], val_ids, sys.argv[2])
