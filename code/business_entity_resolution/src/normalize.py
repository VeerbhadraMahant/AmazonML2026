"""Normalization for business_name / business_address.

Design notes (from EDA on train_source1/2/3 + train_ground_truth):
- S2/S3 sometimes carry non-Latin scripts (Devanagari, Gujarati, Kannada) for the
  same business that S1 always spells in Latin -> transliterate to ASCII.
- Noise seen: leading/embedded `***`, `--`, `[..]`, `#`, homoglyph digits inside
  words (m0tors, lbbie), Sri/Shri honorifics added, legal-suffix drift
  (Corp/Corporation, Pvt/Private, Ltd/Limited, and FR SAS/SARL/EURL/SCI/SNC),
  `&` vs `and` vs French `et`.
- Address noise: Rd/Road, St/Street, Ave/Avenue, Blvd/Boulevard, Dr/Drive, and the
  French R./Av./Bd./Ch./All. abbreviations; truncated house numbers (`501` vs
  `5014`); `##` prefixes on house numbers; component reordering; missing PIN/state.
This module only applies fixed, in-code rules -- no external data or lookups.
"""
import re
import polars as pl
from anyascii import anyascii

NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")

# Legal-form / honorific tokens stripped to build the "core" name used for
# strict blocking keys and for a suffix-invariant similarity feature.
LEGAL_FORM_TOKENS = [
    # US / generic
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "llp",
    "ltd", "limited", "lp", "plc",
    # India
    "pvt", "private", "opc",
    # France
    "sas", "sasu", "sarl", "eurl", "sci", "snc", "sa", "ei",
]
HONORIFIC_TOKENS = ["sri", "shri", "smt", "mr", "mrs", "ms", "dr", "m", "mme", "mlle"]
_LEGAL_RE = re.compile(r"\b(" + "|".join(LEGAL_FORM_TOKENS) + r")\b")
_HONORIFIC_RE = re.compile(r"\b(" + "|".join(HONORIFIC_TOKENS) + r")\b")

# Address token abbreviation expansion (English + French). Applied as whole-word
# replacements after lowercasing/punctuation stripping.
ADDR_ABBREV = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "pkwy": "parkway", "ct": "court", "pl": "place",
    "sq": "square", "ter": "terrace", "apt": "apartment", "ste": "suite",
    "bldg": "building", "no": "number", "hno": "house number",
    "r": "rue", "ch": "chemin", "all": "allee", "imp": "impasse",
}
_ADDR_ABBREV_RE = re.compile(r"\b(" + "|".join(ADDR_ABBREV.keys()) + r")\b")


def _transliterate_col(s: pl.Series) -> pl.Series:
    """Transliterate only the rows that actually contain non-ASCII characters;
    the vast majority are already ASCII, so skip the (relatively slow) Python
    call for them."""
    out = s.to_list()
    for i, v in enumerate(out):
        if v is not None and NON_ASCII_RE.search(v):
            out[i] = anyascii(v)
    return pl.Series(s.name, out, dtype=pl.Utf8)


def _addr_abbrev_expand(text: str) -> str:
    return _ADDR_ABBREV_RE.sub(lambda m: ADDR_ABBREV[m.group(1)], text)


def _basic_clean_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"[^a-z0-9 ]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def normalize_source(df: pl.DataFrame) -> pl.DataFrame:
    """Add normalized columns to a source dataframe (entity_id, business_name,
    business_address, country, source)."""
    df = df.with_columns(
        _transliterate_col(df["business_name"]).alias("_name_ascii"),
        _transliterate_col(df["business_address"].fill_null("")).alias("_addr_ascii"),
    )

    df = df.with_columns(
        _basic_clean_expr("_name_ascii").alias("name_full"),
    ).drop("_name_ascii")

    # name_core: strip honorific + legal-form tokens, collapse spaces
    name_core = (
        df["name_full"]
        .map_elements(
            lambda t: re.sub(r"\s+", " ", _LEGAL_RE.sub(" ", _HONORIFIC_RE.sub(" ", t))).strip(),
            return_dtype=pl.Utf8,
        )
    )
    df = df.with_columns(name_core.alias("name_core"))

    df = df.with_columns(_basic_clean_expr("_addr_ascii").alias("_addr_clean")).drop("_addr_ascii")
    addr_expanded = df["_addr_clean"].map_elements(_addr_abbrev_expand, return_dtype=pl.Utf8)
    df = df.with_columns(addr_expanded.alias("address_norm")).drop("_addr_clean")

    df = df.with_columns(
        pl.col("address_norm")
        .str.extract_all(r"\d+")
        .alias("addr_numbers"),
        pl.col("name_full").str.split(" ").list.first().alias("name_first_token"),
        pl.col("name_core").str.len_chars().eq(0).alias("name_core_empty"),
        pl.col("business_address").is_null().alias("addr_is_null"),
        (pl.col("business_name").str.contains(NON_ASCII_RE.pattern)).alias("name_was_non_ascii"),
    )
    return df


def normalize_and_cache(df: pl.DataFrame, cache_path) -> pl.DataFrame:
    from pathlib import Path
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pl.read_parquet(cache_path)
    out = normalize_source(df)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(cache_path)
    return out
