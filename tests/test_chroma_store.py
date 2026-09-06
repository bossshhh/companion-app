"""
Tests for ChromaMemoryStore. Mirrors test_memory_store.py - same interface,
same weighted-retrieval guarantees, different backend. Uses pytest's
tmp_path fixture for an isolated, disposable ChromaDB directory per test.
"""

from datetime import datetime, timedelta, timezone

from app.chroma_store import ChromaMemoryStore
from app.embeddings import embed_text
from app.models import MemoryCategory, MemoryRecord


def make_record(text: str, importance: float, hours_ago: float, safety_flag: bool = False) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        text=text,
        embedding=embed_text(text),
        importance=importance,
        category=MemoryCategory.OTHER,
        safety_flag=safety_flag,
        created_at=now - timedelta(hours=hours_ago),
        last_accessed_at=now - timedelta(hours=hours_ago),
    )


def test_add_and_all_records_roundtrip(tmp_path):
    store = ChromaMemoryStore(persist_path=str(tmp_path))
    rec = make_record("Enjoys gardening in the afternoon", importance=0.4, hours_ago=2)
    store.add(rec)

    records = store.all_records()
    assert len(records) == 1
    fetched = records[0]
    assert fetched.id == rec.id
    assert fetched.text == rec.text
    assert fetched.importance == rec.importance
    assert fetched.category == rec.category
    assert fetched.safety_flag == rec.safety_flag
    assert fetched.access_count == 0


def test_count_matches_number_added(tmp_path):
    store = ChromaMemoryStore(persist_path=str(tmp_path))
    assert store.count() == 0
    store.add(make_record("first memory", importance=0.3, hours_ago=1))
    store.add(make_record("second memory", importance=0.5, hours_ago=1))
    assert store.count() == 2


def test_touch_updates_access_metadata_and_persists(tmp_path):
    store = ChromaMemoryStore(persist_path=str(tmp_path))
    rec = make_record("test memory", importance=0.5, hours_ago=10)
    store.add(rec)

    now = datetime.now(timezone.utc)
    store.touch(rec.id, now)

    fetched = store.all_records()[0]
    assert fetched.access_count == 1
    assert abs((fetched.last_accessed_at - now).total_seconds()) < 2


def test_touch_on_nonexistent_id_does_not_raise(tmp_path):
    store = ChromaMemoryStore(persist_path=str(tmp_path))
    store.touch("does-not-exist", datetime.now(timezone.utc))


def test_high_importance_beats_recency_via_inherited_retrieve(tmp_path):
    store = ChromaMemoryStore(persist_path=str(tmp_path))

    important_old = make_record(
        "I miss my husband so much, especially in the evenings", importance=0.9, hours_ago=24 * 21
    )
    trivial_recent = make_record("I like tea", importance=0.2, hours_ago=1)

    store.add(important_old)
    store.add(trivial_recent)

    query = embed_text("How are you feeling today?")
    results = store.retrieve(query, top_k=5)

    scores_by_id = {r.memory.id: r.composite_score for r in results}
    assert scores_by_id[important_old.id] > scores_by_id[trivial_recent.id]


def test_safety_flagged_memory_always_returned_and_ranked_first(tmp_path):
    store = ChromaMemoryStore(persist_path=str(tmp_path))

    safety_mem = make_record(
        "I felt a bit dizzy when I stood up this morning", importance=0.3, hours_ago=48, safety_flag=True
    )
    high_importance_but_not_safety = make_record("I miss my husband terribly", importance=0.95, hours_ago=1)

    store.add(high_importance_but_not_safety)
    store.add(safety_mem)

    query = embed_text("Tell me how you're doing")
    results = store.retrieve(query, top_k=1)

    ids = [r.memory.id for r in results]
    assert safety_mem.id in ids
    assert ids[0] == safety_mem.id
    assert high_importance_but_not_safety.id in ids


def test_persists_across_store_instances(tmp_path):
    path = str(tmp_path)
    store1 = ChromaMemoryStore(persist_path=path)
    store1.add(make_record("persisted memory", importance=0.6, hours_ago=5))

    store2 = ChromaMemoryStore(persist_path=path)
    records = store2.all_records()
    assert len(records) == 1
    assert records[0].text == "persisted memory"
