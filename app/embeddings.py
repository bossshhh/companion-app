"""
Embedding provider abstraction.

`fake_embedding()` is a deterministic, dependency-free stand-in used for
local development and tests. `real_embed_text()` uses the same
all-MiniLM-L6-v2 sentence-transformers model Mahir's persona pipeline
uses, so both pipelines produce vectors in the same embedding space.
"""

from __future__ import annotations

import hashlib

EMBEDDING_DIM = 64


def fake_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic pseudo-embedding derived from a hash of the text. NOT
    semantically meaningful - used only to keep tests fast and offline."""
    if not text:
        raise ValueError("text must not be empty")

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    i = 0
    while len(values) < dim:
        byte = digest[i % len(digest)]
        mixed = (byte + i * 31) % 256
        values.append((mixed / 255.0) * 2.0 - 1.0)
        i += 1
    return values


def embed_text(text: str) -> list[float]:
    """Fast, dependency-free embedding used as the DEFAULT everywhere in
    this codebase (tests, demo.py, eval_conversations.py). Production code
    should pass real_embed_text as embed_fn to build_graph() instead."""
    return fake_embedding(text)


# Matches Mahir's persona pipeline's embedding model - same model matters
# here, not just "any real model": two different embedding models produce
# vectors in different, non-comparable spaces.
SENTENCE_TRANSFORMER_MODEL_NAME = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Real embedding model wrapper. Lazily imports sentence_transformers
    inside __init__, not at module level, so importing this file never
    requires the package or model download to be present."""

    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("text must not be empty")
        vector = self._model.encode(text, convert_to_numpy=True)
        return vector.tolist()


_real_embedder: SentenceTransformerEmbedder | None = None


def real_embed_text(text: str) -> list[float]:
    """Real embedding entry point - pass as embed_fn to build_graph() in
    production. Lazily constructs and caches a single embedder instance."""
    global _real_embedder
    if _real_embedder is None:
        _real_embedder = SentenceTransformerEmbedder()
    return _real_embedder.embed(text)
