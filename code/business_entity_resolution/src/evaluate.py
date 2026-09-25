"""Sweeps the decoding threshold tau on the held-out validation split and
reports the actual competition metric (macro F0.5 per S1 entity), not just
the pairwise AUC/AP train_matcher.py prints. This is what threshold we ship.
"""
import numpy as np
import polars as pl

import config
import decode
from split_and_sample import make_split_ids


def build_val_ground_truth() -> pl.DataFrame:
    _, val_ids = make_split_ids(config.WORK_DIR / "norm_train_s1.parquet")
    gt = pl.read_csv(config.TRAIN_DIR / "train_ground_truth.tsv", separator="\t",
                      quote_char=None, infer_schema=False)
    gt = gt.filter(pl.col("source1_entity_id").is_in(val_ids)).with_columns(
        pl.col("matched_entity_ids").fill_null("")
    )
    s1 = pl.read_parquet(config.WORK_DIR / "norm_train_s1.parquet").select(
        pl.col("entity_id").alias("source1_entity_id"), "country"
    )
    gt = gt.join(s1, on="source1_entity_id", how="left")
    return gt, sorted(val_ids)


def sweep(scored_path, taus=None):
    taus = taus if taus is not None else np.round(np.arange(0.30, 0.96, 0.05), 2)
    scored = pl.read_parquet(scored_path).select("source1_entity_id", "entity_id", "pred")
    gt, val_ids = build_val_ground_truth()

    results = []
    for tau in taus:
        assigned = decode.exclusive_assign(scored, tau)
        pred_frame = decode.to_result_frame(val_ids, assigned)
        metrics = decode.f_beta_macro(pred_frame, gt)
        results.append((tau, metrics["macro_f"], metrics.get("per_country")))
        print(f"tau={tau:.2f}  macro_F0.5={metrics['macro_f']:.4f}"
              f"  n_matched_s1={assigned['source1_entity_id'].n_unique()}")

    best = max(results, key=lambda r: r[1])
    print(f"\nBEST tau={best[0]:.2f}  macro_F0.5={best[1]:.4f}")
    if best[2]:
        for row in best[2]:
            print("  ", row)
    return best


if __name__ == "__main__":
    sweep(config.WORK_DIR / "val_scored.parquet")
