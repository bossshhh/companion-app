"""
Embedding provider abstraction.

`fake_embedding()` is a deterministic, dependency-free stand-in used for
local development and tests so the retrieval-scoring logic can be built and
verified without waiting on the real embedding pipeline (whichever model
the vector-storage teammate picks - likely a sentence-transformers model or
an API-based embedder).

Swap `embed_text()` for a real call once that pipeline exists. The rest of
this codebase only depends on "text in, list[float] out, fixed dimension" -
nothing else needs to change.
"""

from __future__ import annotations

import hashlib

EMBEDDING_DIM = 64


def fake_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """
    Deterministic pseudo-embedding derived from a hash of the text.

    NOT semantically meaningful across arbitrary text - it will not cluster
    synonyms the way a real embedding model does. It IS deterministic and
    stable (same text -> same vector), which is exactly what's needed to
    unit-test the scoring math (relevance, recency, importance composition)
    without a real model in the loop. Replace with a real embedder before
    the retrieval quality actually matters for the demo.
    """
    if not text:
        raise ValueError("text must not be empty")

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand the 32-byte digest into `dim` floats in [-1, 1] deterministically.
    values: list[float] = []
    i = 0
    while len(values) < dim:
        byte = digest[i % len(digest)]
        # mix in the index so repeated cycles through the digest don't repeat values
        mixed = (byte + i * 31) % 256
        values.append((mixed / 255.0) * 2.0 - 1.0)
        i += 1
    return values


def embed_text(text: str) -> list[float]:
    """
    Public entry point used by the rest of the app. Currently delegates to
    the fake embedding. Replace this function's body with a real embedding
    API/model call when the teammate's pipeline is ready - callers don't
    need to change.
    """
    return fake_embedding(text)
