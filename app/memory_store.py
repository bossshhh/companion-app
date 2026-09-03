"""
Memory store abstraction + weighted retrieval scoring.

This module defines an ABSTRACT interface (`MemoryStore`) and a concrete
`InMemoryMockStore` implementation. The mock exists so the orchestration
graph never blocks on the teammate building the real vector database:
swap `InMemoryMockStore` for a `PineconeStore` / `ChromaStore` / whatever
he builds, and nothing else in this codebase changes, because both
implementations satisfy the same interface.

Composite retrieval score:

    score = gamma * relevance + beta * importance + alpha * recency

  - relevance: cosine similarity between query embedding and memory embedding, in [0, 1]  (cosine sim is [-1,1], we clip to [0,1])
  - importance: the LLM's normalized 1-10 -> 0-1 importance rating, stored on the memory
  - recency: exponential decay based on hours since last access, in (0, 1]

Default weights are alpha=0.5, beta=1.5, gamma=1.0 - importance weighted
ABOVE recency, not equal to it. Equal weighting (all 1.0) was the initial
assumption but breaks the actual design goal: since recency and importance
both live in [0, 1], equal weights mean recency alone can swing the score
by up to 1.0, which is >= the largest possible importance gap (also 1.0).
A one-hour-old trivial memory would then always be able to outscore a
three-week-old critical one, regardless of how important the old one is -
the opposite of "importance should be able to fight recency and win."
Weighting importance higher is what actually makes that claim true. See
tests/test_memory_store.py::test_high_importance_beats_recency, which
caught this with equal weights before this fix.

Safety-flagged memories bypass this score entirely: they are always
returned, always ranked first, regardless of relevance/importance/recency.
The reasoning: a matter-of-fact statement like "I felt dizzy this morning"
carries low emotional charge and would under-rank on pure importance
scoring, but it is exactly the kind of thing an elderly-care companion
must never fail to surface.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import numpy as np

from app.models import MemoryRecord, ScoredMemory


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors, clipped to [0, 1]."""
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0
    sim = float(np.dot(va, vb) / denom)
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def recency_score(last_accessed_at: datetime, now: datetime, decay_lambda: float = 0.01) -> float:
    """
    Exponential decay: score = e^(-lambda * hours_elapsed).

    decay_lambda=0.01 means a memory retains ~55% of its recency score after
    24 hours, ~30% after 5 days. This is intentionally gentle so that a high
    importance score can still "win" against a moderately old memory -
    importance should be able to fight recency, not be erased by it.
    """
    hours_elapsed = max(0.0, (now - last_accessed_at).total_seconds() / 3600.0)
    return math.exp(-decay_lambda * hours_elapsed)


class MemoryStore(ABC):
    """Abstract interface every memory backend must satisfy."""

    @abstractmethod
    def add(self, record: MemoryRecord) -> None: ...

    @abstractmethod
    def all_records(self) -> list[MemoryRecord]: ...

    @abstractmethod
    def touch(self, record_id: str, when: datetime) -> None:
        """Update last_accessed_at / access_count after a memory is retrieved."""
        ...

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        alpha: float = 0.5,
        beta: float = 1.5,
        gamma: float = 1.0,
        now: datetime | None = None,
    ) -> list[ScoredMemory]:
        """
        Score and rank all stored memories against a query.

        Safety-flagged memories are always included and always ranked first,
        ahead of top_k scored results - they don't count against the budget
        derived from scoring, they're additive.
        """
        now = now or datetime.now(timezone.utc)
        records = self.all_records()

        safety_hits: list[ScoredMemory] = []
        scored: list[ScoredMemory] = []

        for rec in records:
            rel = cosine_similarity(query_embedding, rec.embedding)
            rec_score = recency_score(rec.last_accessed_at, now)
            imp = rec.importance
            composite = gamma * rel + beta * imp + alpha * rec_score

            sm = ScoredMemory(
                memory=rec,
                relevance_score=rel,
                recency_score=rec_score,
                importance_score=imp,
                composite_score=composite,
                bypassed_by_safety=rec.safety_flag,
            )

            if rec.safety_flag:
                safety_hits.append(sm)
            else:
                scored.append(sm)

        scored.sort(key=lambda s: s.composite_score, reverse=True)
        top_scored = scored[:top_k]

        for sm in safety_hits + top_scored:
            self.touch(sm.memory.id, now)

        # Safety hits are prepended unconditionally - they are not subject to top_k.
        return safety_hits + top_scored


class InMemoryMockStore(MemoryStore):
    """
    In-process, non-persistent store. Use this for development and for the
    entire graph build while the vector-storage teammate's real backend is
    not ready. Zero external dependencies, zero network calls.
    """

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def add(self, record: MemoryRecord) -> None:
        self._records[record.id] = record

    def all_records(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def touch(self, record_id: str, when: datetime) -> None:
        rec = self._records.get(record_id)
        if rec is not None:
            rec.last_accessed_at = when
            rec.access_count += 1
