"""Add entity-side (S2/S3 record) and S1-side context/competition features
that are missing from features.py: cand_rank/cand_margin there are computed
only *within* each S1's candidate list, so the model has no signal for "does
this S2/S3 record also look like a good match for several other S1s" or
"how much better is this S1's best candidate than its runner-up". Diagnostics
showed an oracle classifier over the existing candidates already reaches
macro F0.5 ~0.99 -- the gap to the shipped 0.936 is a classifier/precision
problem, not a blocking-recall problem, so this adds discriminating features
instead of re-running retrieval.

Reads the small, already-computed pairs_features_*.parquet (id + score cols
only), adds window features over both `entity_id` and `source1_entity_id`,
and writes a new parquet with the extra columns appended (same row order).
"""
import time
from pathlib import Path

import polars as pl

import config

def add_context_features(features_path: Path, out_path: Path) -> int:
    t0 = time.time()
    df = pl.read_parquet(features_path)  # keep ALL original columns/features

    df = df.with_columns(
        (pl.col("name_core_ratio") + pl.col("addr_token_set_ratio")).alias("_lex")
    )

    # Entity-side (S2/S3 record) competition features: for each S2/S3 record,
    # how does this candidate S1 compare to the record's other candidate S1s?
    df = df.with_columns(
        pl.len().over("entity_id").alias("ent_n_cands"),
        pl.col("sim").rank(method="ordinal", descending=True).over("entity_id").alias("ent_sim_rank"),
        (pl.col("sim") - pl.col("sim").max().over("entity_id")).alias("ent_sim_margin"),
        pl.col("_lex").rank(method="ordinal", descending=True).over("entity_id").alias("ent_lex_rank"),
        (pl.col("_lex") - pl.col("_lex").max().over("entity_id")).alias("ent_lex_margin"),
    )
    # gap to 2nd best sim per entity (0 if only 1 candidate)
    top2 = (
        df.select("entity_id", "sim")
        .sort("sim", descending=True)
        .group_by("entity_id", maintain_order=True)
        .agg(pl.col("sim").head(2).alias("_top2"))
        .with_columns(
            pl.when(pl.col("_top2").list.len() >= 2)
            .then(pl.col("_top2").list.get(0) - pl.col("_top2").list.get(1))
            .otherwise(0.0)
            .alias("ent_sim_gap2")
        )
        .select("entity_id", "ent_sim_gap2")
    )
    df = df.join(top2, on="entity_id", how="left")
    del top2

    # S1-side lexical competition (cand_rank/cand_margin already cover sim;
    # add the same for the combined lexical score)
    df = df.with_columns(
        pl.len().over("source1_entity_id").alias("s1_n_cands"),
        pl.col("_lex").rank(method="ordinal", descending=True).over("source1_entity_id").alias("s1_lex_rank"),
        (pl.col("_lex") - pl.col("_lex").max().over("source1_entity_id")).alias("s1_lex_margin"),
    )

    df = df.drop("_lex")
    df.write_parquet(out_path)
    n = df.height
    print(f"add_context_features: {features_path.name} -> {out_path.name} "
          f"{n} rows ({round(time.time()-t0,1)}s)")
    return n


if __name__ == "__main__":
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    add_context_features(
        config.WORK_DIR / f"pairs_features_{split}.parquet",
        config.WORK_DIR / f"pairs_features_{split}_ctx.parquet",
    )
