"""Tests for the FastAPI /chat endpoint, using stub components."""

from fastapi.testclient import TestClient

from app.llm_node import LLMResult
from app.memory_store import InMemoryMockStore
from app.models import MemoryCategory, StructuredReply
from app.server import create_app


class StubGenerator:
    def __init__(self, response: StructuredReply, succeeded: bool = True):
        self._response = response
        self._succeeded = succeeded

    def generate(self, user_input: str, memory_context: str) -> LLMResult:
        return LLMResult(structured=self._response, succeeded=self._succeeded, used_fallback=not self._succeeded)


def _routine_response() -> StructuredReply:
    return StructuredReply(
        reply_text="That sounds lovely!",
        memory_worthy=False,
        memory_summary=None,
        importance=2,
        category=MemoryCategory.SMALL_TALK,
        safety_flag=False,
    )


def test_health_endpoint():
    app = create_app(InMemoryMockStore(), StubGenerator(_routine_response()))
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_expected_shape():
    app = create_app(InMemoryMockStore(), StubGenerator(_routine_response()))
    client = TestClient(app)
    response = client.post("/chat", json={"user_id": "test-user-1", "message": "hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["reply_text"] == "That sounds lovely!"
    assert body["safety_triggered"] is False


def test_chat_reflects_safety_flag():
    response_obj = StructuredReply(
        reply_text="Please sit down and rest for a moment.",
        memory_worthy=True,
        memory_summary="Felt dizzy",
        importance=9,
        category=MemoryCategory.SAFETY,
        safety_flag=True,
    )
    app = create_app(InMemoryMockStore(), StubGenerator(response_obj))
    client = TestClient(app)
    response = client.post("/chat", json={"user_id": "test-user-2", "message": "I felt dizzy"})
    assert response.status_code == 200
    assert response.json()["safety_triggered"] is True


def test_chat_missing_message_returns_422():
    app = create_app(InMemoryMockStore(), StubGenerator(_routine_response()))
    client = TestClient(app)
    response = client.post("/chat", json={"user_id": "test-user-3"})
    assert response.status_code == 422


def test_chat_empty_message_returns_422():
    app = create_app(InMemoryMockStore(), StubGenerator(_routine_response()))
    client = TestClient(app)
    response = client.post("/chat", json={"user_id": "test-user-4", "message": ""})
    assert response.status_code == 422


def test_conversation_persists_across_requests_for_same_user():
    app = create_app(InMemoryMockStore(), StubGenerator(_routine_response()))
    client = TestClient(app)
    r1 = client.post("/chat", json={"user_id": "persistent-user", "message": "hello"})
    r2 = client.post("/chat", json={"user_id": "persistent-user", "message": "how are you"})
    assert r1.status_code == 200
    assert r2.status_code == 200
