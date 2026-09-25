"""Stage 2: pairwise features for the candidate set.

At ~120M candidate pairs, a single lazy "join everything + window functions +
sink_parquet" plan turned out to still blow through available RAM (polars'
streaming engine doesn't keep every operator bounded, and a Rust-level OOM
kills the process outright, unrecoverable in Python). This runs in two
deliberately small-memory passes instead:

1. `augment_candidates()` works on the *id-only* candidate table (just two id
   strings + a float + a source tag per row, ~5GB at 120M rows) -- computes
   per-S1 rank/margin and joins the ground-truth label, entirely in memory,
   then writes it back out. No wide text columns touch this pass.
2. `compute_features_streaming()` re-reads that table in row-group batches
   (e.g. 2M rows), and for each batch does a small hash join against the S1
   and S2+S3 text tables (each held once, fully in RAM -- ~200MB and ~1GB
   respectively, not per batch) to fetch just that batch's text, computes the
   rapidfuzz features (parallelized across CPU cores), and streams the result
   to the final parquet. The only large-ish object alive at any time is one
   batch, not the full 120M-row join.
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz

import config

TEXT_COLS = ["name_full", "name_core", "address_norm", "addr_numbers", "name_first_token",
             "name_was_non_ascii"]


def _s1_name_core_freq(s1: pl.DataFrame) -> pl.DataFrame:
    """How many times this S1 name_core recurs in S1 -- a "generic name,
    trust the address more" signal (e.g. "Primary Care Group" recurs 253x)."""
    return (
        s1.group_by("name_core").len().rename({"len": "s1_name_core_freq"})
        .join(s1.select("entity_id", "name_core"), on="name_core")
        .select(pl.col("entity_id").alias("source1_entity_id"), "s1_name_core_freq")
    )


def augment_candidates(candidates_path: Path, s1_path: Path, out_path: Path,
                        gt_path: Path | None = None) -> int:
    """Id-only pass: adds cand_rank, cand_margin, s1_name_core_freq, and
    (if gt_path given) label. No text columns involved."""
    cands = pl.read_parquet(candidates_path)
    cands = cands.with_columns(
        pl.col("sim").rank(method="ordinal", descending=True)
        .over("source1_entity_id").alias("cand_rank"),
        (pl.col("sim") - pl.col("sim").max().over("source1_entity_id")).alias("cand_margin"),
    )

    s1 = pl.read_parquet(s1_path).select("entity_id", "name_core")
    freq = _s1_name_core_freq(s1)
    cands = cands.join(freq, on="source1_entity_id", how="left").with_columns(
        pl.col("s1_name_core_freq").fill_null(1)
    )
    del s1, freq

    if gt_path is not None:
        gt = pl.read_csv(gt_path, separator="\t", quote_char=None, infer_schema=False)
        gt_pairs = (
            gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .rename({"matched_entity_ids": "entity_id"})
            .with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
        )
        cands = cands.join(gt_pairs, on=["source1_entity_id", "entity_id"], how="left")
        cands = cands.with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
        del gt, gt_pairs

    cands.write_parquet(out_path)
    n = cands.height
    del cands
    return n


def _feature_batch(args) -> dict:
    (name_1, name_2, core_1, core_2, addr_1, addr_2, nums_1, nums_2,
     first_1, first_2, na_1, na_2) = args
    n = len(name_1)
    out = {k: np.empty(n, dtype=np.float32) for k in
           ["name_ratio", "name_core_ratio", "name_token_set_ratio", "addr_ratio",
            "addr_token_set_ratio", "name_first_token_eq", "number_exact",
            "number_prefix", "number_jaccard", "name_len_diff", "addr_len_diff",
            "either_non_ascii"]}
    for i in range(n):
        a, b = name_1[i] or "", name_2[i] or ""
        ca, cb = core_1[i] or "", core_2[i] or ""
        da, db = addr_1[i] or "", addr_2[i] or ""
        out["name_ratio"][i] = fuzz.ratio(a, b) / 100.0
        out["name_core_ratio"][i] = fuzz.ratio(ca, cb) / 100.0
        out["name_token_set_ratio"][i] = fuzz.token_set_ratio(a, b) / 100.0
        out["addr_ratio"][i] = fuzz.ratio(da, db) / 100.0
        out["addr_token_set_ratio"][i] = fuzz.token_set_ratio(da, db) / 100.0
        out["name_first_token_eq"][i] = float(first_1[i] == first_2[i] and first_1[i] != "")
        out["name_len_diff"][i] = abs(len(a) - len(b))
        out["addr_len_diff"][i] = abs(len(da) - len(db))
        out["either_non_ascii"][i] = float(bool(na_1[i]) or bool(na_2[i]))

        s1n, s2n = set(nums_1[i] or []), set(nums_2[i] or [])
        if s1n or s2n:
            out["number_jaccard"][i] = len(s1n & s2n) / len(s1n | s2n) if (s1n | s2n) else 0.0
        else:
            out["number_jaccard"][i] = 0.0
        n1_first = next(iter(nums_1[i]), None) if nums_1[i] else None
        n2_first = next(iter(nums_2[i]), None) if nums_2[i] else None
        if n1_first and n2_first:
            out["number_exact"][i] = float(n1_first == n2_first)
            out["number_prefix"][i] = float(n1_first.startswith(n2_first) or n2_first.startswith(n1_first))
        else:
            out["number_exact"][i] = 0.0
            out["number_prefix"][i] = 0.0
    return out


def _chunk(lst, n_chunks):
    size = max(1, len(lst) // n_chunks + 1)
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def compute_features_streaming(augmented_path: Path, s1_path: Path, other_paths: list[Path],
                                out_path: Path, batch_rows: int = 2_000_000,
                                n_workers: int | None = None) -> int:
    n_workers = n_workers or max(1, mp.cpu_count() - 2)

    s1 = pl.read_parquet(s1_path).select("entity_id", *TEXT_COLS)
    s1 = s1.rename({c: f"{c}_1" for c in ["entity_id", *TEXT_COLS]})
    other = pl.concat([pl.read_parquet(p).select("entity_id", *TEXT_COLS) for p in other_paths])
    other = other.rename({c: f"{c}_2" for c in ["entity_id", *TEXT_COLS]})

    pf = pq.ParquetFile(str(augmented_path))
    writer = None
    total = 0
    with mp.Pool(n_workers) as pool:
        for batch in pf.iter_batches(batch_size=batch_rows):
            df = pl.from_arrow(pa.Table.from_batches([batch]))
            n = df.height
            if n == 0:
                continue
            df = df.join(s1, left_on="source1_entity_id", right_on="entity_id_1", how="left")
            df = df.join(other, left_on="entity_id", right_on="entity_id_2", how="left")

            name_1 = df["name_full_1"].to_list(); name_2 = df["name_full_2"].to_list()
            core_1 = df["name_core_1"].to_list(); core_2 = df["name_core_2"].to_list()
            addr_1 = df["address_norm_1"].to_list(); addr_2 = df["address_norm_2"].to_list()
            nums_1 = df["addr_numbers_1"].to_list(); nums_2 = df["addr_numbers_2"].to_list()
            first_1 = df["name_first_token_1"].to_list(); first_2 = df["name_first_token_2"].to_list()
            na_1 = df["name_was_non_ascii_1"].to_list(); na_2 = df["name_was_non_ascii_2"].to_list()

            idxs = list(range(n))
            chunks = _chunk(idxs, n_workers)
            args = [(
                [name_1[i] for i in c], [name_2[i] for i in c],
                [core_1[i] for i in c], [core_2[i] for i in c],
                [addr_1[i] for i in c], [addr_2[i] for i in c],
                [nums_1[i] for i in c], [nums_2[i] for i in c],
                [first_1[i] for i in c], [first_2[i] for i in c],
                [na_1[i] for i in c], [na_2[i] for i in c],
            ) for c in chunks]
            results = pool.map(_feature_batch, args)
            merged = {k: np.concatenate([r[k] for r in results]) for k in results[0]}

            keep_cols = [c for c in ["source1_entity_id", "entity_id", "source", "sim",
                                      "cand_rank", "cand_margin", "s1_name_core_freq", "label"]
                         if c in df.columns]
            out_df = df.select(keep_cols).with_columns(
                [pl.Series(k, v) for k, v in merged.items()]
            )
            out_tbl = out_df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), out_tbl.schema)
            writer.write_table(out_tbl)
            total += n
            del df, name_1, name_2, core_1, core_2, addr_1, addr_2, nums_1, nums_2
            del first_1, first_2, na_1, na_2, results, merged, out_df, out_tbl
    if writer is not None:
        writer.close()
    return total
