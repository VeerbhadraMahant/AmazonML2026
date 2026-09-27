"""Top-50 candidate expansion (recall rebuild).

The original blocking keeps each S2/S3 record's embedding top-10 S1 neighbours
(+ two lexical joins). On validation, 2.65% of true pairs never become
candidates, and 34% of those sit at embedding rank 11-50. Keeping all of
ranks 11-50 would 5x the candidate set, so a rank 11-50 neighbour is only
added when it also shares the name's first token or any address number with
the record (keeps 86% of the recoverable misses, 24% of the extra pairs).

Writes <ER_WORK_DIR>/candidates_new_<split>.parquet: only pairs NOT already in
work/candidates_<split>.parquet, with columns source1_entity_id, entity_id,
source, sim.

Usage: ER_WORK_DIR=.../work_v2 python retrieve_k50.py --split train
"""
import argparse
import time
import zlib

import numpy as np
import polars as pl
import torch

import config

K_LO, K_HI = 10, 50
OLD_WORK = config.ROOT / "work"


def _codes(values: list, table: dict) -> np.ndarray:
    return np.array([table.setdefault(v or "", len(table)) if v else -1 for v in values], dtype=np.int64)


def _num_codes(lists, width=3) -> np.ndarray:
    """First `width` address numbers as int codes, -1 padded (-2 on the S1 side
    so padding never matches padding)."""
    out = np.full((len(lists), width), -1, dtype=np.int64)
    for i, lst in enumerate(lists):
        if lst:
            for j, v in enumerate(lst[:width]):
                out[i, j] = zlib.crc32(v.encode())
    return out


def run(split: str):
    t0 = time.time()
    W = config.WORK_DIR
    cols = ["entity_id", "country", "name_first_token", "addr_numbers"]
    s1 = pl.read_parquet(W / f"norm_{split}_s1.parquet", columns=cols)
    e1 = np.load(W / f"emb_{split}_s1.npy", mmap_mode="r")
    tok_table: dict = {}
    s1_tok = _codes(s1["name_first_token"].to_list(), tok_table)
    s1_num = _num_codes(s1["addr_numbers"].to_list())
    s1_num[s1_num == -1] = -2
    s1_country = s1["country"].to_numpy()
    s1_ids = s1["entity_id"].to_numpy()
    parts = []
    for k in (2, 3):
        o = pl.read_parquet(W / f"norm_{split}_s{k}.parquet", columns=cols)
        ek = np.load(W / f"emb_{split}_s{k}.npy", mmap_mode="r")
        o_tok = _codes(o["name_first_token"].to_list(), tok_table)
        o_num = _num_codes(o["addr_numbers"].to_list())
        o_country = o["country"].to_numpy()
        o_ids = o["entity_id"].to_numpy()
        for c in sorted(set(s1_country.tolist())):
            m1 = np.flatnonzero(s1_country == c)
            mo = np.flatnonzero(o_country == c)
            if len(m1) == 0 or len(mo) == 0:
                continue
            P = torch.from_numpy(np.ascontiguousarray(e1[m1])).cuda().half()
            chunk = max(64, min(4096, 3 * 10**8 // len(m1)))
            kk = min(K_HI, len(m1))
            for a in range(0, len(mo), chunk):
                q_idx = mo[a:a + chunk]
                Q = torch.from_numpy(np.ascontiguousarray(ek[q_idx])).cuda().half()
                v, ix = torch.topk(Q @ P.T, kk, dim=1)
                v = v[:, K_LO:].float().cpu().numpy()
                ix = ix[:, K_LO:].cpu().numpy()
                if ix.size == 0:
                    continue
                s1_rows = m1[ix]                          # (c, 40) global S1 row ids
                q_rows = np.repeat(q_idx[:, None], ix.shape[1], axis=1)
                tok_ok = (s1_tok[s1_rows] == o_tok[q_rows]) & (o_tok[q_rows] >= 0)
                num_ok = np.zeros_like(tok_ok)
                for i in range(3):
                    for j in range(3):
                        num_ok |= o_num[q_rows, i] == s1_num[s1_rows, j]
                keep = tok_ok | num_ok
                parts.append(pl.DataFrame({
                    "source1_entity_id": s1_ids[s1_rows[keep]],
                    "entity_id": o_ids[q_rows[keep]],
                    "source": np.full(int(keep.sum()), f"S{k}"),
                    "sim": v[keep].astype(np.float32),
                }))
                del Q, v, ix
            del P
            torch.cuda.empty_cache()
            print(f"[{split}] s{k} {c}: done ({round(time.time()-t0)}s)", flush=True)
    new = pl.concat(parts)
    old = pl.scan_parquet(OLD_WORK / f"candidates_{split}.parquet").select("source1_entity_id", "entity_id").collect()
    new = new.join(old, on=["source1_entity_id", "entity_id"], how="anti").unique(["source1_entity_id", "entity_id"])
    new.write_parquet(W / f"candidates_new_{split}.parquet")
    print(f"[{split}] new candidate pairs: {new.height} (old {old.height}) ({round(time.time()-t0)}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    run(ap.parse_args().split)
