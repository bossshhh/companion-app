"""
Graph-level tests using a stubbed StructuredReplyGenerator - no Anthropic
API key or network call required. These prove the LangGraph wiring
(conditional routing, state threading, memory write-back) is correct
independent of actual LLM output quality, which you verify separately
by hand during prompt iteration (Day 4 of the plan).
"""

import uuid

from app.graph import build_graph
from app.llm_node import LLMResult
from app.memory_store import InMemoryMockStore
from app.models import MemoryCategory, StructuredReply


class StubGenerator:
    """Drop-in replacement for StructuredReplyGenerator that returns a
    pre-programmed response instead of calling the real API."""

    def __init__(self, response: StructuredReply, succeeded: bool = True):
        self._response = response
        self._succeeded = succeeded
        self.calls: list[tuple[str, str]] = []

    def generate(self, user_input: str, memory_context: str) -> LLMResult:
        self.calls.append((user_input, memory_context))
        return LLMResult(structured=self._response, succeeded=self._succeeded, used_fallback=not self._succeeded)


def run_turn(generator: StubGenerator, user_input: str, store=None):
    store = store or InMemoryMockStore()
    app = build_graph(store, generator)
    user_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": user_id}}
    # Partial dict, not a full GraphState - see note in app/main.py on why
    # a full instance would clobber checkpointed state with field defaults.
    update = {"user_id": user_id, "user_input": user_input}
    result = app.invoke(update, config=config)
    return result, store


def test_routine_turn_skips_safety_followup_and_stores_memory():
    response = StructuredReply(
        reply_text="That sounds lovely!",
        memory_worthy=True,
        memory_summary="Enjoys gardening in the afternoon",
        importance=4,
        category=MemoryCategory.HABIT,
        safety_flag=False,
    )
    generator = StubGenerator(response)
    result, store = run_turn(generator, "I spent the afternoon in my garden")

    assert result["final_reply_text"] == "That sounds lovely!"
    assert result["safety_triggered"] is False
    assert len(store.all_records()) == 1
    assert store.all_records()[0].text == "Enjoys gardening in the afternoon"


def test_safety_flagged_turn_routes_through_safety_followup():
    response = StructuredReply(
        reply_text="I'm sorry to hear that - please sit down and rest for a moment.",
        memory_worthy=True,
        memory_summary="Felt dizzy after standing up",
        importance=9,
        category=MemoryCategory.SAFETY,
        safety_flag=True,
    )
    generator = StubGenerator(response)
    result, store = run_turn(generator, "I felt really dizzy when I stood up")

    assert result["safety_triggered"] is True
    stored = store.all_records()
    assert len(stored) == 1
    assert stored[0].safety_flag is True
    assert stored[0].importance == 0.9  # normalized from 9/10


def test_non_memory_worthy_turn_does_not_store():
    response = StructuredReply(
        reply_text="It sure is a nice day!",
        memory_worthy=False,
        memory_summary=None,
        importance=2,
        category=MemoryCategory.SMALL_TALK,
        safety_flag=False,
    )
    generator = StubGenerator(response)
    result, store = run_turn(generator, "Nice weather today")

    assert result["final_reply_text"] == "It sure is a nice day!"
    assert len(store.all_records()) == 0


def test_failed_llm_call_skips_memory_write_but_still_replies():
    """
    Simulates the retry-then-degrade fallback path: succeeded=False means
    the structured call failed and we fell back to plain text. The graph
    must still produce a reply and must NOT store an unreliable memory.
    """
    fallback_response = StructuredReply(
        reply_text="I'm here with you - could you say that again for me?",
        memory_worthy=False,
        memory_summary=None,
        importance=5,
        category=MemoryCategory.OTHER,
        safety_flag=False,
    )
    generator = StubGenerator(fallback_response, succeeded=False)
    result, store = run_turn(generator, "some garbled input")

    assert result["final_reply_text"]
    assert result["llm_call_failed"] is True
    assert len(store.all_records()) == 0


def test_conversation_history_accumulates_across_turns():
    response = StructuredReply(
        reply_text="Got it!",
        memory_worthy=False,
        memory_summary=None,
        importance=1,
        category=MemoryCategory.SMALL_TALK,
        safety_flag=False,
    )
    generator = StubGenerator(response)
    store = InMemoryMockStore()
    app = build_graph(store, generator)
    user_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": user_id}}

    result1 = app.invoke({"user_id": user_id, "user_input": "Hello there"}, config=config)
    assert len(result1["conversation_history"]) == 2

    result2 = app.invoke({"user_id": user_id, "user_input": "How are you?"}, config=config)
    # checkpointer carries prior history forward within the same thread_id
    assert len(result2["conversation_history"]) == 4


def test_safety_flag_forces_safety_category_even_if_llm_disagrees():
    """
    Locks in the fix from the eval run: a real LLM call returned
    safety_flag=True with category='habit' for "forgot my morning pills" -
    the LLM treats these as independent fields, but a safety-flagged record
    must always be filed under category=safety regardless of what category
    the LLM assigned, since downstream alerting/dashboards will filter by
    category expecting that invariant to hold.
    """
    response = StructuredReply(
        reply_text="Please take them now if you can, and check with your doctor if unsure.",
        memory_worthy=True,
        memory_summary="Forgot to take morning pills today",
        importance=9,
        category=MemoryCategory.HABIT,  # LLM's (inconsistent) choice
        safety_flag=True,
    )
    generator = StubGenerator(response)
    result, store = run_turn(generator, "I think I forgot to take my morning pills today")

    assert result["safety_triggered"] is True
    stored = store.all_records()
    assert len(stored) == 1
    assert stored[0].category == MemoryCategory.SAFETY, "safety_flag=True must force category=safety"


def test_custom_embed_fn_is_actually_used():
    """
    Proves build_graph()'s embed_fn injection works end-to-end.
    """
    calls: list[str] = []

    def tracking_embed_fn(text: str) -> list[float]:
        calls.append(text)
        return [1.0, 0.0, 0.0]

    response = StructuredReply(
        reply_text="Noted.",
        memory_worthy=True,
        memory_summary="Enjoys painting on weekends",
        importance=4,
        category=MemoryCategory.HABIT,
        safety_flag=False,
    )
    generator = StubGenerator(response)
    store = InMemoryMockStore()
    app = build_graph(store, generator, embed_fn=tracking_embed_fn)
    user_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": user_id}}

    app.invoke({"user_id": user_id, "user_input": "I paint on weekends"}, config=config)

    assert "I paint on weekends" in calls
    assert "Enjoys painting on weekends" in calls
    stored = store.all_records()
    assert len(stored) == 1
    assert stored[0].embedding == [1.0, 0.0, 0.0]
