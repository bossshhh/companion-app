"""
ChromaDB-backed MemoryStore.

Implements the same MemoryStore interface as InMemoryMockStore (see
app/memory_store.py): add, all_records, touch. The weighted retrieval
scoring (relevance + importance + recency, safety bypass) is defined ONCE
on the abstract MemoryStore base class and works unchanged here - this
class only handles persistence.

SCALING NOTE: retrieve() (inherited) calls all_records() and scores every
record in Python on every query. Fine for demo/sprint scale (dozens to low
thousands of memories), NOT how you'd use ChromaDB at real scale - Chroma's
value is efficient ANN search via collection.query(), which this class
does not use for the relevance component. If memory volume grows past what
brute-force scoring can handle fast enough for a live voice interface, the
fix is to have retrieve() call collection.query() for an initial top-N
candidate set, then apply importance/recency/safety weighting only to that
smaller set.

Embeddings are stored/queried as pre-computed vectors (via
app/embeddings.py), not generated internally by ChromaDB - keeps the
embedding model swappable and avoids Chroma's default embedding function
trying to download a model on first use.
"""

from __future__ import annotations

from datetime import datetime, timezone

import chromadb

from app.memory_store import MemoryStore
from app.models import MemoryCategory, MemoryRecord

COLLECTION_NAME = "companion_memories"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _record_to_metadata(record: MemoryRecord) -> dict:
    return {
        "importance": record.importance,
        "category": record.category.value,
        "safety_flag": record.safety_flag,
        "created_at": _iso(record.created_at),
        "last_accessed_at": _iso(record.last_accessed_at),
        "access_count": record.access_count,
    }


def _metadata_to_record(record_id: str, text: str, embedding: list[float], metadata: dict) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        text=text,
        embedding=embedding,
        importance=metadata["importance"],
        category=MemoryCategory(metadata["category"]),
        safety_flag=metadata["safety_flag"],
        created_at=_parse_iso(metadata["created_at"]),
        last_accessed_at=_parse_iso(metadata["last_accessed_at"]),
        access_count=metadata["access_count"],
    )


class ChromaMemoryStore(MemoryStore):
    """Persistent, disk-backed memory store using ChromaDB."""

    def __init__(self, persist_path: str = "./chroma_data", collection_name: str = COLLECTION_NAME) -> None:
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )

    def add(self, record: MemoryRecord) -> None:
        self._collection.add(
            ids=[record.id],
            embeddings=[record.embedding],
            documents=[record.text],
            metadatas=[_record_to_metadata(record)],
        )

    def all_records(self) -> list[MemoryRecord]:
        result = self._collection.get(include=["embeddings", "documents", "metadatas"])
        records = []
        for record_id, text, embedding, metadata in zip(
            result["ids"], result["documents"], result["embeddings"], result["metadatas"]
        ):
            records.append(_metadata_to_record(record_id, text, list(embedding), metadata))
        return records

    def touch(self, record_id: str, when: datetime) -> None:
        existing = self._collection.get(ids=[record_id], include=["metadatas"])
        if not existing["ids"]:
            return
        metadata = existing["metadatas"][0]
        metadata["last_accessed_at"] = _iso(when)
        metadata["access_count"] = metadata["access_count"] + 1
        self._collection.update(ids=[record_id], metadatas=[metadata])

    def count(self) -> int:
        """Not part of the MemoryStore interface - handy for sanity checks."""
        return self._collection.count()
