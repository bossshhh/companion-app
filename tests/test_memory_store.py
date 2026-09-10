from datetime import datetime, timedelta, timezone

from app.embeddings import embed_text
from app.memory_store import InMemoryMockStore, cosine_similarity, recency_score
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


def test_cosine_similarity_identical_vectors_is_one():
    v = embed_text("hello world")
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_dimension_mismatch_raises():
    try:
        cosine_similarity([1.0, 2.0], [1.0])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_recency_score_decays_over_time():
    now = datetime.now(timezone.utc)
    fresh = recency_score(now, now)
    old = recency_score(now - timedelta(hours=100), now)
    assert fresh > old
    assert 0.0 < old <= 1.0
    assert fresh == 1.0


def test_high_importance_beats_recency():
    """
    The core design claim from the spec: an emotionally important memory
    from weeks ago should be able to outrank a trivial memory from
    yesterday. This test locks that behavior in so nobody accidentally
    breaks it while tuning weights later in the sprint.
    """
    store = InMemoryMockStore()

    important_old = make_record("I miss my husband so much, especially in the evenings", importance=0.9, hours_ago=24 * 21)
    trivial_recent = make_record("I like tea", importance=0.2, hours_ago=1)

    store.add(important_old)
    store.add(trivial_recent)

    query = embed_text("How are you feeling today?")
    results = store.retrieve(query, top_k=5)

    scores_by_id = {r.memory.id: r.composite_score for r in results}
    assert scores_by_id[important_old.id] > scores_by_id[trivial_recent.id]


def test_safety_flagged_memory_always_returned_and_ranked_first():
    """
    A calmly-stated safety concern must never be buried by pure
    importance/recency scoring. It bypasses the score entirely.
    """
    store = InMemoryMockStore()

    # Low emotional charge, stated flatly - would score poorly on pure
    # importance heuristics, but is exactly what must never be missed.
    safety_mem = make_record("I felt a bit dizzy when I stood up this morning", importance=0.3, hours_ago=48, safety_flag=True)
    high_importance_but_not_safety = make_record(
        "I miss my husband terribly", importance=0.95, hours_ago=1
    )

    store.add(high_importance_but_not_safety)
    store.add(safety_mem)

    query = embed_text("Tell me how you're doing")
    results = store.retrieve(query, top_k=1)  # top_k=1 to prove safety is ADDITIVE, not competing for the slot

    ids = [r.memory.id for r in results]
    assert safety_mem.id in ids, "safety-flagged memory must always be present regardless of top_k"
    assert ids[0] == safety_mem.id, "safety-flagged memories must be ranked first"
    # top_scored result should still be included on top of the safety hit
    assert high_importance_but_not_safety.id in ids


def test_touch_updates_access_metadata():
    store = InMemoryMockStore()
    rec = make_record("test memory", importance=0.5, hours_ago=10)
    store.add(rec)

    original_access_count = rec.access_count
    query = embed_text("anything")
    store.retrieve(query, top_k=5)

    stored = store.all_records()[0]
    assert stored.access_count == original_access_count + 1


def test_safety_bypass_is_capped_at_scale():
    """
    Locks in the fix for a real scaling problem: with many safety-flagged
    memories stored, the safety bypass must NOT return all of them
    unconditionally - only the top max_safety_bypass by composite score.
    """
    store = InMemoryMockStore()

    for i in range(10):
        store.add(make_record(f"Safety concern number {i}", importance=0.5, hours_ago=i, safety_flag=True))

    query = embed_text("How is your day going?")
    results = store.retrieve(query, top_k=5, max_safety_bypass=3)

    safety_results = [r for r in results if r.bypassed_by_safety]
    assert len(safety_results) == 3, "safety bypass must be capped, not unbounded"
