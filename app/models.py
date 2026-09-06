"""
Data models for the companion app.

These schemas are the contract between:
  - the LangGraph orchestration layer (this codebase)
  - the vector storage / data pipeline teammate (memory persistence)
  - the voice synthesis teammate (ElevenLabs input)
  - the mobile UI teammate (what the app displays)

Keep this file as the single source of truth for the memory record shape.
If the vector-storage teammate needs a different field name, change it here
and everything downstream stays consistent.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryCategory(str, Enum):
    """Coarse category tag, used for routing and analytics, not for scoring."""

    HABIT = "habit"
    EMOTIONAL = "emotional"
    SAFETY = "safety"
    SMALL_TALK = "small_talk"
    OTHER = "other"


class MemoryRecord(BaseModel):
    """
    A single stored memory (one turn or one extracted fact from a turn).

    `importance` is on a 0.0-1.0 scale (normalized from the LLM's 1-10 rating)
    so it composes cleanly with the other 0.0-1.0 components of the retrieval
    score. `safety_flag` is a hard override: safety-flagged memories bypass
    the weighted score entirely at retrieval time and are always surfaced.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., min_length=1, description="The memory content, in natural language.")
    embedding: list[float] = Field(
        ..., description="Vector embedding of `text`. Populated by the vector-store teammate's pipeline."
    )
    importance: float = Field(..., ge=0.0, le=1.0, description="Normalized importance, 0=trivial, 1=critical.")
    category: MemoryCategory = MemoryCategory.OTHER
    safety_flag: bool = Field(False, description="True if this memory indicates a health/safety concern.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0

    @field_validator("embedding")
    @classmethod
    def embedding_non_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("embedding must not be empty")
        return v


class ScoredMemory(BaseModel):
    """A MemoryRecord plus its computed retrieval score, returned by the retriever."""

    memory: MemoryRecord
    relevance_score: float
    recency_score: float
    importance_score: float
    composite_score: float
    bypassed_by_safety: bool = False


class StructuredReply(BaseModel):
    """
    The exact shape we force the LLM to return via `instructor`, in a single
    call, so we get the conversational reply and the memory-scoring metadata
    for free (no second LLM call to "rate importance" separately).
    """

    reply_text: str = Field(..., min_length=1, description="What the assistant says back to the user.")
    memory_worthy: bool = Field(
        ..., description="Whether this turn contains anything worth storing as a long-term memory."
    )
    memory_summary: Optional[str] = Field(
        None, description="If memory_worthy, a short first-person summary of the fact/event to store."
    )
    importance: int = Field(5, ge=1, le=10, description="1=trivial, 10=critical. Only meaningful if memory_worthy.")
    category: MemoryCategory = MemoryCategory.OTHER
    safety_flag: bool = Field(
        False, description="True if the user's message indicates a health/safety concern (fall, pain, confusion, etc.)."
    )

    @field_validator("memory_summary")
    @classmethod
    def summary_required_if_worthy(cls, v: Optional[str], info) -> Optional[str]:
        # Soft validation only - instructor will retry on hard failures, but we
        # don't want to hard-fail a valid reply just because the summary was
        # left blank. We backfill it downstream instead of raising here.
        return v


class ConversationTurn(BaseModel):
    """One turn in the conversation, as stored in the graph's message history."""

    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GraphState(BaseModel):
    """
    The state object threaded through every LangGraph node.

    LangGraph nodes each receive this, mutate a subset of it, and return the
    subset that changed. Keeping it as one Pydantic model (rather than a bare
    TypedDict) gives us validation for free at every node boundary.
    """

    user_id: str
    user_input: str = ""
    conversation_history: list[ConversationTurn] = Field(default_factory=list)

    retrieved_memories: list[ScoredMemory] = Field(default_factory=list)

    structured_reply: Optional[StructuredReply] = None
    final_reply_text: str = ""

    # emotion tracker output - independent of the LLM's own StructuredReply
    # category/importance judgment, so the two can be compared/cross-checked
    detected_emotion: Optional[str] = None
    detected_emotion_score: Optional[float] = None

    # routing / control flags set by nodes, read by conditional edges
    safety_triggered: bool = False
    llm_call_failed: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)
