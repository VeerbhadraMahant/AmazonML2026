"""TSV I/O helpers. All challenge files are tab-separated; commas appear inside
addresses and ID lists, so a naive comma-split or default read_csv is wrong."""
import polars as pl


def read_tsv(path) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def write_tsv(df: pl.DataFrame, path) -> None:
    df.write_csv(path, separator="\t", quote_style="never")


def read_source(path) -> pl.DataFrame:
    """entity_id, business_name, business_address, country -- with a source tag
    derived from the entity_id prefix (no separate source column in the data)."""
    df = read_tsv(path)
    return df.with_columns(pl.col("entity_id").str.slice(0, 2).alias("source"))


def read_ground_truth(path) -> pl.DataFrame:
    return read_tsv(path)
