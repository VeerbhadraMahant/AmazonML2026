"""Stage 1: candidate generation / blocking.

Combines a GPU embedding top-K search (catches fuzzy/transliterated/reordered
matches) with two cheap exact-key lexical joins (a safety net for obvious
matches the embedding model might rank outside top-K), always restricted to
matching country -- 100% of true pairs agree on country in the training data,
and this is also what makes France (test-only) handled for free.

Runs country-by-country and streams each part straight to disk (instead of
holding tens of millions of candidate rows plus multi-GB embedding arrays all
in RAM at once) -- at ~24GB system RAM this matters once source2/3 embeddings
alone are ~4-8GB each.
"""
import gc
from pathlib import Path

import numpy as np
import polars as pl
import torch

import config


def topk_gpu(query_emb: np.ndarray, passage_emb: np.ndarray, k: int, max_sim_elems: int = 3 * 10**8):
    """Return (idx, score) each shaped (n_queries, k): for each query row, the
    top-k passage row indices and their cosine similarities (embeddings are
    pre-normalized, so dot product == cosine).

    The (chunk, n_passages) similarity matrix is the real memory hog -- with
    ~1.3M passages (US), a naive 8192-row query chunk would need a ~21GB
    matrix. `chunk` is picked so chunk * n_passages stays under
    `max_sim_elems` (default budget ~600MB at fp16), regardless of how big
    either side is."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k = min(k, passage_emb.shape[0])
    p = torch.from_numpy(np.ascontiguousarray(passage_emb)).to(device)  # (P, D) fp16
    n = query_emb.shape[0]
    chunk = max(64, min(8192, max_sim_elems // max(1, passage_emb.shape[0])))
    idx_out = np.empty((n, k), dtype=np.int32)
    score_out = np.empty((n, k), dtype=np.float16)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        q = torch.from_numpy(np.ascontiguousarray(query_emb[start:end])).to(device)  # (c, D)
        sims = q @ p.T  # (c, P)
        vals, inds = torch.topk(sims, k=k, dim=1)
        idx_out[start:end] = inds.cpu().numpy().astype(np.int32)
        score_out[start:end] = vals.cpu().numpy().astype(np.float16)
        del q, sims, vals, inds
    del p
    if device == "cuda":
        torch.cuda.empty_cache()
    return idx_out, score_out


def _flatten_list_cols(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    exprs = [pl.col(c).list.first().fill_null("").alias(c)
             for c in cols if df.schema[c] == pl.List(pl.Utf8)]
    return df.with_columns(exprs) if exprs else df


def embedding_candidates_to_parts(s1: pl.DataFrame, s1_emb: np.ndarray, other: pl.DataFrame,
                                   other_emb: np.ndarray, k: int, out_dir: Path, tag: str) -> None:
    """other is a normalized S2 or S3 dataframe (must have a 'source' column).
    Writes one parquet per country to out_dir, freeing memory between
    countries instead of accumulating everything before returning."""
    s1_country = s1["country"].to_numpy()
    other_country = other["country"].to_numpy()
    s1_id_all = s1["entity_id"].to_numpy()
    o_id_all = other["entity_id"].to_numpy()
    o_source_all = other["source"].to_numpy()

    for country in sorted(set(s1_country.tolist())):
        s1_mask = s1_country == country
        o_mask = other_country == country
        n_s1, n_o = int(s1_mask.sum()), int(o_mask.sum())
        if n_s1 == 0 or n_o == 0:
            continue
        s1_ids_local = s1_id_all[s1_mask]
        o_ids_local = o_id_all[o_mask]
        o_source_local = o_source_all[o_mask]
        idx, score = topk_gpu(np.asarray(other_emb[o_mask]), np.asarray(s1_emb[s1_mask]), k=k)
        kk = idx.shape[1]
        s1_matched = s1_ids_local[idx.reshape(-1)]
        part = pl.DataFrame({
            "source1_entity_id": s1_matched,
            "entity_id": np.repeat(o_ids_local, kk),
            "source": np.repeat(o_source_local, kk),
            "sim": score.reshape(-1).astype(np.float32),
        })
        safe_country = "".join(c if c.isalnum() else "_" for c in country)
        part.write_parquet(out_dir / f"emb_{tag}_{safe_country}.parquet")
        del idx, score, s1_matched, part, s1_ids_local, o_ids_local, o_source_local
        gc.collect()


def lexical_candidates(s1: pl.DataFrame, other: pl.DataFrame, key_cols: list[str],
                        max_group_size: int = 40) -> pl.DataFrame:
    """Exact-key join on the given normalized columns (e.g. ['country',
    'name_core']), skipping keys so common they'd blow up the candidate set --
    those generic-name cases are left to the embedding + number features to
    disambiguate instead."""
    s1_small = _flatten_list_cols(s1.select(["entity_id", *key_cols]), key_cols)
    s1_small = s1_small.rename({"entity_id": "source1_entity_id"})
    other_small = _flatten_list_cols(other.select(["entity_id", "source", *key_cols]), key_cols)
    for c in key_cols:
        if c != "country":
            s1_small = s1_small.filter(pl.col(c).str.len_chars() > 0)
            other_small = other_small.filter(pl.col(c).str.len_chars() > 0)
    s1_grp_size = s1_small.group_by(key_cols).len()
    ok_keys = s1_grp_size.filter(pl.col("len") <= max_group_size).drop("len")
    s1_small = s1_small.join(ok_keys, on=key_cols, how="inner")
    pairs = s1_small.join(other_small, on=key_cols, how="inner")
    return pairs.select("source1_entity_id", "entity_id", "source").with_columns(
        pl.lit(None, dtype=pl.Float32).alias("sim"),
    )


def build_candidates(s1: pl.DataFrame, s1_emb: np.ndarray,
                      others: list[tuple[pl.DataFrame, np.ndarray]],
                      parts_dir: Path, k: int = config.TOP_K_CANDIDATES) -> pl.DataFrame:
    """others: list of (normalized_df, embeddings) for S2 and S3. Streams
    per-country embedding parts to `parts_dir`, then does one streaming
    group-by dedupe pass over everything (embedding parts + lexical parts)."""
    parts_dir = Path(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    for f in parts_dir.glob("*.parquet"):
        f.unlink()

    for i, (other, other_emb) in enumerate(others):
        tag = f"o{i}"
        embedding_candidates_to_parts(s1, s1_emb, other, other_emb, k, parts_dir, tag)
        lex1 = lexical_candidates(s1, other, ["country", "name_core"])
        lex1.write_parquet(parts_dir / f"lex1_{tag}.parquet")
        del lex1
        lex2 = lexical_candidates(s1, other, ["country", "name_first_token", "addr_numbers"])
        lex2.write_parquet(parts_dir / f"lex2_{tag}.parquet")
        del lex2
        gc.collect()

    lazy = pl.scan_parquet(str(parts_dir / "*.parquet"))
    deduped = (
        lazy.group_by(["source1_entity_id", "entity_id"])
        .agg(pl.col("sim").max().alias("sim"), pl.col("source").first().alias("source"))
        .collect(engine="streaming")
    )
    return deduped


def to_candidate_pairs_tsv(all_s1_ids: list[str], candidates: pl.DataFrame) -> pl.DataFrame:
    """Group into the submission shape: one row per S1 id (even if empty)."""
    grouped = (
        candidates.group_by("source1_entity_id")
        .agg(pl.col("entity_id").alias("ids"))
        .with_columns(pl.col("ids").list.join(",").alias("candidate_entity_ids"))
        .select("source1_entity_id", "candidate_entity_ids")
    )
    base = pl.DataFrame({"source1_entity_id": all_s1_ids})
    out = base.join(grouped, on="source1_entity_id", how="left").with_columns(
        pl.col("candidate_entity_ids").fill_null("")
    )
    return out
