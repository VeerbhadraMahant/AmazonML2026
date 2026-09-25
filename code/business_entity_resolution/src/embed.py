"""GPU sentence embeddings for blocking (Stage 1).

Uses intfloat/multilingual-e5-small (MIT license, 118M params) -- comfortably
within the challenge's 8B-parameter cap. E5 models expect an
"query: "/"passage: " instruction prefix: Source 1 is the fixed reference
index we search into (passage), Source 2/3 records are the queries.
"""
import numpy as np
import polars as pl
import torch
from sentence_transformers import SentenceTransformer

import config

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(config.EMBED_MODEL_NAME, device=device)
        _model.max_seq_length = config.MAX_SEQ_LEN
        if device == "cuda":
            _model.half()
    return _model


def build_text(df: pl.DataFrame) -> list[str]:
    return (df["name_full"] + " , " + df["address_norm"]).to_list()


def embed_texts(texts: list[str], role: str, batch_size: int = 2048) -> np.ndarray:
    """role is 'query' (S2/S3) or 'passage' (S1), per E5 convention."""
    assert role in ("query", "passage")
    model = get_model()
    prefixed = [f"{role}: {t}" for t in texts]
    with torch.no_grad():
        emb = model.encode(
            prefixed,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    return emb.astype(np.float16)


def embed_source_and_cache(df: pl.DataFrame, role: str, cache_path) -> np.ndarray:
    from pathlib import Path
    cache_path = Path(cache_path)
    if cache_path.exists():
        # mmap: boolean-masking a country subset only realizes the touched
        # pages in RAM instead of pinning the whole multi-GB array resident.
        return np.load(cache_path, mmap_mode="r")
    texts = build_text(df)
    emb = embed_texts(texts, role=role)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, emb)
    return emb
