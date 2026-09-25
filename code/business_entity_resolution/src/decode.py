"""Stage 3: decoding. Turns per-pair match probabilities into the final
submission shape.

Each S2/S3 record belongs to at most one S1 entity in the ground truth (we
verified this: zero records matched >1 S1 in training), so instead of
thresholding every pair independently we assign each S2/S3 record to its
single highest-probability S1 match, and only keep that assignment if
confidence clears a threshold tau -- tuned to maximize macro F0.5, which
weights precision 2x over recall, so tau is deliberately biased toward
"say nothing" over a shaky guess.
"""
import polars as pl


def exclusive_assign(scored: pl.DataFrame, tau: float) -> pl.DataFrame:
    """scored: source1_entity_id, entity_id, pred (+ other cols ok).
    Returns one row per (entity_id that cleared tau and won its own
    argmax over source1 candidates): source1_entity_id, entity_id, pred."""
    best_per_entity = (
        scored.sort("pred", descending=True)
        .unique(subset=["entity_id"], keep="first")
    )
    return best_per_entity.filter(pl.col("pred") >= tau).select(
        "source1_entity_id", "entity_id", "pred"
    )


def to_result_frame(all_s1_ids: list[str], assigned: pl.DataFrame, id_col: str = "matched_entity_ids") -> pl.DataFrame:
    grouped = (
        assigned.group_by("source1_entity_id")
        .agg(pl.col("entity_id").alias("ids"))
        .with_columns(pl.col("ids").list.join(",").alias(id_col))
        .select("source1_entity_id", id_col)
    )
    base = pl.DataFrame({"source1_entity_id": all_s1_ids})
    out = base.join(grouped, on="source1_entity_id", how="left").with_columns(
        pl.col(id_col).fill_null("")
    )
    return out


def f_beta_macro(pred_frame: pl.DataFrame, gt_frame: pl.DataFrame, beta: float = 0.5) -> dict:
    """pred_frame, gt_frame: source1_entity_id, matched_entity_ids (comma-joined
    string, '' for none). Returns overall macro F_beta plus per-country if
    'country' is present in gt_frame."""
    b2 = beta * beta
    joined = gt_frame.join(pred_frame, on="source1_entity_id", how="left", suffix="_pred")
    joined = joined.with_columns(pl.col("matched_entity_ids_pred").fill_null(""))

    def _score(row):
        gold = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()
        pred = set(row["matched_entity_ids_pred"].split(",")) if row["matched_entity_ids_pred"] else set()
        if not gold and not pred:
            return 1.0
        if not pred:
            return 0.0
        tp = len(gold & pred)
        precision = tp / len(pred)
        recall = tp / len(gold) if gold else 0.0
        if precision == 0 and recall == 0:
            return 0.0
        return (1 + b2) * precision * recall / (b2 * precision + recall) if (b2 * precision + recall) > 0 else 0.0

    scores = [_score(r) for r in joined.iter_rows(named=True)]
    joined = joined.with_columns(pl.Series("f_score", scores))
    result = {"macro_f": sum(scores) / len(scores), "n": len(scores)}
    if "country" in joined.columns:
        per_country = joined.group_by("country").agg(pl.col("f_score").mean().alias("macro_f"), pl.len())
        result["per_country"] = per_country.to_dicts()
    return result
