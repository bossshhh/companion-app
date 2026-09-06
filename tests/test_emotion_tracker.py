"""
Tests for the Emotion Tracker node, using a stub EmotionClassifier so no
real model download (torch/transformers weights) is needed to verify the
graph wiring is correct.
"""

import uuid

from app.emotion_tracker import EmotionClassifier, EmotionResult
from app.graph import build_graph
from app.llm_node import LLMResult
from app.memory_store import InMemoryMockStore
from app.models import MemoryCategory, StructuredReply


class StubGenerator:
    def __init__(self, response: StructuredReply):
        self._response = response

    def generate(self, user_input: str, memory_context: str) -> LLMResult:
        return LLMResult(structured=self._response, succeeded=True, used_fallback=False)


class StubEmotionClassifier(EmotionClassifier):
    """Returns a fixed emotion result, standing in for the real HF model."""

    def __init__(self, label: str, score: float):
        self._label = label
        self._score = score
        self.calls: list[str] = []

    def classify(self, text: str) -> EmotionResult:
        self.calls.append(text)
        return EmotionResult(label=self._label, score=self._score)


def _routine_response() -> StructuredReply:
    return StructuredReply(
        reply_text="I hear you.",
        memory_worthy=False,
        memory_summary=None,
        importance=3,
        category=MemoryCategory.OTHER,
        safety_flag=False,
    )


def test_emotion_classifier_none_skips_node_entirely():
    """
    When no classifier is passed, the detect_emotion node is never added to
    the graph, so it never runs and never writes to state. LangGraph's
    invoke() result only includes keys that some node actually wrote during
    that run - it does NOT backfill the Pydantic schema's declared defaults
    for untouched fields. So the correct expectation here is that the keys
    are simply ABSENT from the result dict, not present with value None.
    """
    store = InMemoryMockStore()
    generator = StubGenerator(_routine_response())
    app = build_graph(store, generator, emotion_classifier=None)
    user_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": user_id}}

    result = app.invoke({"user_id": user_id, "user_input": "hello"}, config=config)

    assert result.get("detected_emotion") is None
    assert result.get("detected_emotion_score") is None


def test_emotion_classifier_populates_state():
    store = InMemoryMockStore()
    generator = StubGenerator(_routine_response())
    classifier = StubEmotionClassifier(label="sadness", score=0.87)
    app = build_graph(store, generator, emotion_classifier=classifier)
    user_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": user_id}}

    result = app.invoke({"user_id": user_id, "user_input": "I miss my husband"}, config=config)

    assert result["detected_emotion"] == "sadness"
    assert result["detected_emotion_score"] == 0.87
    assert classifier.calls == ["I miss my husband"]


def test_emotion_classifier_runs_before_reply_generation():
    store = InMemoryMockStore()
    generator = StubGenerator(_routine_response())
    classifier = StubEmotionClassifier(label="joy", score=0.95)
    app = build_graph(store, generator, emotion_classifier=classifier)
    user_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": user_id}}

    result = app.invoke({"user_id": user_id, "user_input": "what a wonderful day"}, config=config)

    assert result["detected_emotion"] == "joy"
    assert result["final_reply_text"] == "I hear you."
