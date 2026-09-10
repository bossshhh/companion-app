"""
Memory store abstraction + weighted retrieval scoring.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import numpy as np

from app.models import MemoryRecord, ScoredMemory


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0
    sim = float(np.dot(va, vb) / denom)
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def recency_score(last_accessed_at: datetime, now: datetime, decay_lambda: float = 0.01) -> float:
    hours_elapsed = max(0.0, (now - last_accessed_at).total_seconds() / 3600.0)
    return math.exp(-decay_lambda * hours_elapsed)


class MemoryStore(ABC):
    """Abstract interface every memory backend must satisfy."""

    @abstractmethod
    def add(self, record: MemoryRecord) -> None: ...

    @abstractmethod
    def all_records(self) -> list[MemoryRecord]: ...

    @abstractmethod
    def touch(self, record_id: str, when: datetime) -> None: ...

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        max_safety_bypass: int = 3,
        alpha: float = 0.5,
        beta: float = 1.5,
        gamma: float = 1.0,
        now: datetime | None = None,
    ) -> list[ScoredMemory]:
        """
        Safety-flagged memories bypass the composite score and are always
        ranked first - but ONLY up to max_safety_bypass of them, chosen by
        composite score among safety-flagged memories themselves. Without
        this cap, every safety-flagged memory ever stored gets injected
        into every prompt regardless of relevance - fine at demo scale but
        breaks down at real scale (e.g. 25 safety memories flooding every
        single turn's context).
        """
        now = now or datetime.now(timezone.utc)
        records = self.all_records()

        safety_candidates: list[ScoredMemory] = []
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
                safety_candidates.append(sm)
            else:
                scored.append(sm)

        safety_candidates.sort(key=lambda s: s.composite_score, reverse=True)
        safety_hits = safety_candidates[:max_safety_bypass]

        scored.sort(key=lambda s: s.composite_score, reverse=True)
        top_scored = scored[:top_k]

        for sm in safety_hits + top_scored:
            self.touch(sm.memory.id, now)

        return safety_hits + top_scored


class InMemoryMockStore(MemoryStore):
    """In-process, non-persistent store. Zero external dependencies."""

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
