"""Merge the top-50 expansion pairs (retrieve_k50.py) into the old feature set.

Produces <ER_WORK_DIR>/pairs_features_<split>_ctx.parquet with exactly the
columns/dtypes/order of work/pairs_features_<split>_ctx.parquet, over the
UNION of old candidate pairs + candidates_new_<split>.parquet:

1. new pairs only: lean augment (s1_name_core_freq, label for train) +
   features.compute_features_streaming -> pairs_features_new_<split>.parquet
2. cand_rank / cand_margin recomputed over the union (same exprs as
   features.augment_candidates; old rows come first in their file order, so
   ordinal tie-breaks among old rows are unchanged)
3. the 9 context features of add_context_features.py recomputed over the union

Memory: window aggregates run on a narrow frame (u64-hashed ids, sim with
nulls, _lex, is_new) ~25 B/row; derived columns are written to
memory-mapped Arrow IPC temp files, then the base features are streamed in
batches (old file then new file, same order) and hstacked with the matching
slice of the derived columns.

Must run as a script (Windows multiprocessing spawn).
Usage (from src/):  ER_WORK_DIR=.../work_v2 python merge_v2.py --split train
"""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import config
import features
import io_utils

ID1, ID2 = "source1_entity_id", "entity_id"
CTX_COLS = [  # order and dtypes as written by add_context_features.py
    ("ent_n_cands", pa.uint32()), ("ent_sim_rank", pa.uint32()), ("ent_sim_margin", pa.float32()),
    ("ent_lex_rank", pa.uint32()), ("ent_lex_margin", pa.float32()), ("ent_sim_gap2", pa.float32()),
    ("s1_n_cands", pa.uint32()), ("s1_lex_rank", pa.uint32()), ("s1_lex_margin", pa.float32()),
]
S1_DERIVED = ["cand_rank", "cand_margin", "s1_n_cands", "s1_lex_rank", "s1_lex_margin"]
ENT_DERIVED = ["ent_n_cands", "ent_sim_rank", "ent_sim_margin", "ent_lex_rank", "ent_lex_margin",
               "ent_sim_gap2"]


def log(msg, t0):
    print(f"[{round(time.time() - t0, 1):>7}s] {msg}", flush=True)


def _n_rows(path: Path | None) -> int:
    if path is None or not Path(path).exists():
        return 0
    return pq.ParquetFile(str(path)).metadata.num_rows


# ---------------------------------------------------------------- step 1 ----
def lean_augment(new_cands: Path, s1_path: Path, out_path: Path, gt_path: Path | None) -> None:
    """augment_candidates minus the in-memory window pass (cand_rank/margin
    are recomputed over the union later) -- a streaming join with two small
    tables, so it never materializes the 50-100M string rows."""
    s1 = pl.read_parquet(s1_path, columns=["entity_id", "name_core"])
    freq = features._s1_name_core_freq(s1)
    del s1
    lf = (pl.scan_parquet(new_cands).select(ID1, ID2, "source", "sim")
          .join(freq.lazy(), on=ID1, how="left")
          .with_columns(pl.col("s1_name_core_freq").fill_null(1)))
    if gt_path is not None:
        gt = pl.read_csv(gt_path, separator="\t", quote_char=None, infer_schema=False)
        gt_pairs = (
            gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
            .rename({"matched_entity_ids": ID2})
            .with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
            .unique([ID1, ID2])
        )
        del gt
        lf = (lf.join(gt_pairs.lazy(), on=[ID1, ID2], how="left")
              .with_columns(pl.col("label").fill_null(0).cast(pl.Int8)))
    lf.sink_parquet(out_path, engine="streaming")


# ---------------------------------------------------------------- step 2/3 --
def build_narrow(paths: list[Path], n_total: int, batch_rows: int) -> pl.DataFrame:
    """One row per union pair, in (old file, new file) order."""
    s1h = np.empty(n_total, np.uint64)
    enth = np.empty(n_total, np.uint64)
    sim = np.empty(n_total, np.float32)
    valid = np.empty(n_total, np.bool_)
    lex = np.empty(n_total, np.float32)
    off = 0
    cols = [ID1, ID2, "sim", "name_core_ratio", "addr_token_set_ratio"]
    for p in paths:
        for b in pq.ParquetFile(str(p)).iter_batches(batch_size=batch_rows, columns=cols):
            d = pl.from_arrow(b).select(
                pl.col(ID1).hash().alias("a"), pl.col(ID2).hash().alias("b"),
                pl.col("sim").cast(pl.Float32).fill_null(0.0).alias("s"),
                pl.col("sim").is_not_null().alias("v"),
                (pl.col("name_core_ratio") + pl.col("addr_token_set_ratio")).cast(pl.Float32).alias("l"),
            )
            n = d.height
            s1h[off:off + n] = d["a"].to_numpy()
            enth[off:off + n] = d["b"].to_numpy()
            sim[off:off + n] = d["s"].to_numpy()
            valid[off:off + n] = d["v"].to_numpy()
            lex[off:off + n] = d["l"].to_numpy()
            off += n
            del d, b
    assert off == n_total, (off, n_total)
    bitmap = pa.py_buffer(np.packbits(valid, bitorder="little"))
    sim_arr = pa.Array.from_buffers(pa.float32(), n_total, [bitmap, pa.py_buffer(sim)],
                                    null_count=int(n_total - valid.sum()))
    del valid
    df = pl.DataFrame([pl.Series("s1h", s1h), pl.Series("enth", enth),
                       pl.Series("sim", pl.from_arrow(sim_arr)), pl.Series("_lex", lex)])
    del s1h, enth, sim, lex, sim_arr, bitmap
    gc.collect()
    return df


def _is_new_series(n: int, n_old: int) -> pl.Series:
    m = np.zeros(n, np.bool_)
    m[n_old:] = True
    return pl.Series("_is_new", m)


def derive_s1(df: pl.DataFrame, n_old: int) -> pl.DataFrame:
    is_new = _is_new_series(df.height, n_old)
    out = df.with_columns(is_new).select(
        pl.col("sim").rank(method="ordinal", descending=True).over("s1h").alias("cand_rank"),
        (pl.col("sim") - pl.col("sim").max().over("s1h")).alias("cand_margin"),
        pl.len().over("s1h").alias("s1_n_cands"),
        pl.col("_lex").rank(method="ordinal", descending=True).over("s1h").alias("s1_lex_rank"),
        (pl.col("_lex") - pl.col("_lex").max().over("s1h")).alias("s1_lex_margin"),
        pl.col("_is_new").any().over("s1h").alias("_s1_has_new"),
    )
    return out


def derive_ent(df: pl.DataFrame, n_old: int) -> pl.DataFrame:
    is_new = _is_new_series(df.height, n_old)
    g = "enth"
    d = df.with_columns(is_new, pl.col("sim").rank(method="ordinal", descending=True).over(g)
                        .alias("ent_sim_rank"))
    # ent_sim_gap2 (add_context_features: sort sim desc -- nulls FIRST -- take
    # head(2) per entity; len>=2 -> top0-top1 else 0.0). So: n<2 -> 0.0; any
    # null sim in the entity -> null; else max - value at ordinal rank 2.
    second = pl.when(pl.col("ent_sim_rank") == 2).then(pl.col("sim")).max().over(g)
    out = d.select(
        pl.len().over(g).alias("ent_n_cands"),
        "ent_sim_rank",
        (pl.col("sim") - pl.col("sim").max().over(g)).alias("ent_sim_margin"),
        pl.col("_lex").rank(method="ordinal", descending=True).over(g).alias("ent_lex_rank"),
        (pl.col("_lex") - pl.col("_lex").max().over(g)).alias("ent_lex_margin"),
        pl.when(pl.len().over(g) < 2).then(pl.lit(0.0, dtype=pl.Float32))
        .when(pl.col("sim").null_count().over(g) > 0).then(pl.lit(None, dtype=pl.Float32))
        .otherwise(pl.col("sim").max().over(g) - second).cast(pl.Float32).alias("ent_sim_gap2"),
        pl.col("_is_new").any().over(g).alias("_ent_has_new"),
    )
    return out


def open_ipc(path: Path) -> pa.Table:
    return pa.ipc.open_file(pa.memory_map(str(path), "r")).read_all()


def assemble(paths: list[Path], n_old: int, A: pa.Table, B: pa.Table, out_path: Path,
             target: pa.Schema, batch_rows: int, t0: float) -> int:
    base_cols = [f.name for f in target if f.name not in {"cand_rank", "cand_margin", *dict(CTX_COLS)}]
    writer = pq.ParquetWriter(str(out_path), target, compression="zstd")
    off = 0
    chk = {"rows_s1_no_new": 0, "rank_mismatch": 0, "margin_mismatch": 0}
    for fi, p in enumerate(paths):
        read_cols = base_cols + (["cand_rank", "cand_margin"] if fi == 0 else [])
        for b in pq.ParquetFile(str(p)).iter_batches(batch_size=batch_rows, columns=read_cols):
            n = b.num_rows
            base = pl.from_arrow(b)
            a = pl.from_arrow(A.slice(off, n))
            e = pl.from_arrow(B.slice(off, n))
            if fi == 0:  # old rows whose S1 got no new pair must keep their old rank/margin
                m = ~a["_s1_has_new"]
                chk["rows_s1_no_new"] += int(m.sum())
                chk["rank_mismatch"] += int((~base["cand_rank"].eq_missing(a["cand_rank"]) & m).sum())
                chk["margin_mismatch"] += int((~base["cand_margin"].eq_missing(a["cand_margin"]) & m).sum())
                base = base.drop("cand_rank", "cand_margin")
            df = pl.concat([base, a.drop("_s1_has_new"), e.drop("_ent_has_new")], how="horizontal")
            tbl = df.select(target.names).to_arrow().cast(target)
            writer.write_table(tbl)
            off += n
            del base, a, e, df, tbl, b
        log(f"assembled {p.name}: cumulative {off} rows", t0)
    writer.close()
    assert off == A.num_rows == B.num_rows, (off, A.num_rows, B.num_rows)
    print(f"check (old rows, S1 without new pairs): {chk}", flush=True)
    return off


def target_schema(old_pf: Path) -> pa.Schema:
    s = pq.read_schema(str(old_pf))
    s = pa.schema([pa.field(f.name, f.type) for f in s])  # drop metadata
    return pa.schema(list(s) + [pa.field(n, t) for n, t in CTX_COLS])


# ---------------------------------------------------------------- step 4 ----
def write_candidate_tsv(s1_path: Path, old_tsv: Path, new_cands: Path | None, out_path: Path):
    """Old grouped candidate_pairs + new pairs appended per S1 (one row per S1
    id from norm_*_s1, including empty) -- same shape as
    retrieve.to_candidate_pairs_tsv."""
    ids = pl.read_parquet(s1_path, columns=["entity_id"]).rename({"entity_id": ID1})
    old = io_utils.read_tsv(old_tsv).rename({"candidate_entity_ids": "_old"})
    out = ids.join(old, on=ID1, how="left", maintain_order="left")
    del old
    if new_cands is not None:
        new = (pl.scan_parquet(new_cands).select(ID1, ID2).group_by(ID1)
               .agg(pl.col(ID2).str.join(",").alias("_new")).collect(engine="streaming"))
        out = out.join(new, on=ID1, how="left", maintain_order="left")
        del new
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("_new"))
    out = out.select(
        ID1,
        pl.concat_str([pl.col("_old").replace("", None), pl.col("_new")], separator=",",
                      ignore_nulls=True).fill_null("").alias("candidate_entity_ids"),
    )
    io_utils.write_tsv(out, out_path)
    return out.height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--old-dir", type=Path, default=config.ROOT / "work")
    ap.add_argument("--norm-dir", type=Path, default=config.WORK_DIR)
    ap.add_argument("--batch-rows", type=int, default=1_000_000)
    ap.add_argument("--feat-batch-rows", type=int, default=2_000_000)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--redo-new-features", action="store_true")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    sp, W, OLD, NORM = args.split, config.WORK_DIR, args.old_dir, args.norm_dir
    assert W.resolve() != OLD.resolve(), "ER_WORK_DIR must not be the old work dir"
    old_pf = OLD / f"pairs_features_{sp}.parquet"
    new_cands = W / f"candidates_new_{sp}.parquet"
    new_aug = W / f"candidates_new_augmented_{sp}.parquet"
    new_pf = W / f"pairs_features_new_{sp}.parquet"
    out_path = W / f"pairs_features_{sp}_ctx.parquet"
    s1_path = NORM / f"norm_{sp}_s1.parquet"
    other_paths = [NORM / f"norm_{sp}_s2.parquet", NORM / f"norm_{sp}_s3.parquet"]
    gt_path = config.TRAIN_DIR / "train_ground_truth.tsv" if sp == "train" else None
    tmp_a, tmp_b = W / f"_merge_tmp_s1_{sp}.arrow", W / f"_merge_tmp_ent_{sp}.arrow"
    print(f"[{sp}] WORK_DIR={W} OLD={OLD}", flush=True)

    # 1. features for the new pairs
    n_new_cands = _n_rows(new_cands)
    if n_new_cands == 0:
        log("no new candidate pairs -- union == old", t0)
        new_pf = None
    elif new_pf.exists() and not args.redo_new_features:
        log(f"reusing {new_pf.name} ({_n_rows(new_pf)} rows)", t0)
    else:
        lean_augment(new_cands, s1_path, new_aug, gt_path)
        log(f"lean augment: {_n_rows(new_aug)} rows -> {new_aug.name}", t0)
        n = features.compute_features_streaming(new_aug, s1_path, other_paths, new_pf,
                                                batch_rows=args.feat_batch_rows,
                                                n_workers=args.n_workers)
        log(f"compute_features_streaming: {n} rows -> {new_pf.name}", t0)
        new_aug.unlink()
    n_old, n_new = _n_rows(old_pf), _n_rows(new_pf)
    if new_pf is not None:
        assert n_new == n_new_cands, (n_new, n_new_cands)
    paths = [old_pf] + ([new_pf] if new_pf is not None else [])
    N = n_old + n_new
    log(f"union rows: old {n_old} + new {n_new} = {N}", t0)

    # 2/3. window features on the narrow frame
    df = build_narrow(paths, N, max(args.batch_rows, 2_000_000))
    log(f"narrow frame built ({df.estimated_size() / 2**30:.2f} GiB)", t0)
    a = derive_s1(df, n_old)
    a.write_ipc(tmp_a, compression="uncompressed")
    s1_new = int(a["_s1_has_new"].sum())
    del a
    gc.collect()
    log(f"S1-side window features -> {tmp_a.name} (rows in S1 groups touched by new: {s1_new})", t0)
    b = derive_ent(df, n_old)
    b.write_ipc(tmp_b, compression="uncompressed")
    ent_new = int(b["_ent_has_new"].sum())
    del b, df
    gc.collect()
    log(f"entity-side window features -> {tmp_b.name} (rows in entity groups touched by new: {ent_new})", t0)

    # stream-assemble the output
    A, B = open_ipc(tmp_a), open_ipc(tmp_b)
    n_out = assemble(paths, n_old, A, B, out_path, target_schema(old_pf), args.batch_rows, t0)
    del A, B
    gc.collect()
    if not args.keep_tmp:
        for f in (tmp_a, tmp_b):
            try:
                f.unlink()
            except OSError as ex:  # memory map may linger on Windows
                print(f"could not delete {f}: {ex}")
    log(f"wrote {out_path} ({n_out} rows = {n_old} old + {n_new} new)", t0)

    # 4. candidate_pairs tsv (test only)
    if sp == "test":
        tsv = W / f"candidate_pairs_{sp}.tsv"
        h = write_candidate_tsv(s1_path, OLD / f"candidate_pairs_{sp}.tsv",
                                new_cands if n_new_cands else None, tsv)
        log(f"wrote {tsv} ({h} S1 rows)", t0)
    log("done", t0)


if __name__ == "__main__":
    main()
