"""
LangGraph orchestration for the companion app.

Topology:

    retrieve_memories -> [detect_emotion] -> generate_reply --(conditional)--> safety_followup -> store_memory -> END
                                                              \\--(conditional)--> store_memory -> END
"""

from __future__ import annotations

import logging
from typing import Callable

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.embeddings import embed_text
from app.emotion_tracker import EmotionClassifier
from app.llm_node import StructuredReplyGenerator
from app.memory_store import MemoryStore
from app.models import ConversationTurn, GraphState, MemoryCategory, MemoryRecord

logger = logging.getLogger("companion_app.graph")


def _format_memory_context(state: GraphState) -> str:
    if not state.retrieved_memories:
        return ""
    lines = []
    for sm in state.retrieved_memories:
        tag = " [SAFETY]" if sm.bypassed_by_safety else ""
        lines.append(f"- {sm.memory.text}{tag}")
    return "\n".join(lines)


def build_graph(
    store: MemoryStore,
    generator: StructuredReplyGenerator,
    emotion_classifier: EmotionClassifier | None = None,
    embed_fn: Callable[[str], list[float]] = embed_text,
):
    """
    embed_fn defaults to the fast fake embedding (app.embeddings.embed_text)
    so every existing caller/test keeps working unchanged - pass
    app.embeddings.real_embed_text here in production.
    """

    def retrieve_memories_node(state: GraphState) -> dict:
        query_embedding = embed_fn(state.user_input)
        scored = store.retrieve(query_embedding, top_k=5)
        return {"retrieved_memories": scored}

    def detect_emotion_node(state: GraphState) -> dict:
        result = emotion_classifier.classify(state.user_input)
        return {"detected_emotion": result.label, "detected_emotion_score": result.score}

    def generate_reply_node(state: GraphState) -> dict:
        memory_context = _format_memory_context(state)
        result = generator.generate(state.user_input, memory_context)
        structured = result.structured

        return {
            "structured_reply": structured,
            "final_reply_text": structured.reply_text,
            "safety_triggered": structured.safety_flag,
            "llm_call_failed": not result.succeeded,
            "conversation_history": state.conversation_history
            + [
                ConversationTurn(role="user", content=state.user_input),
                ConversationTurn(role="assistant", content=structured.reply_text),
            ],
        }

    def safety_followup_node(state: GraphState) -> dict:
        logger.warning(
            "SAFETY FLAG triggered for user_id=%s | input=%r | reply=%r",
            state.user_id,
            state.user_input,
            state.final_reply_text,
        )
        return {}

    def store_memory_node(state: GraphState) -> dict:
        structured = state.structured_reply
        if structured is None or not structured.memory_worthy or state.llm_call_failed:
            return {}

        summary_text = structured.memory_summary or state.user_input
        category = MemoryCategory.SAFETY if structured.safety_flag else MemoryCategory(structured.category)
        record = MemoryRecord(
            text=summary_text,
            embedding=embed_fn(summary_text),
            importance=structured.importance / 10.0,
            category=category,
            safety_flag=structured.safety_flag,
        )
        store.add(record)
        return {}

    def route_after_reply(state: GraphState) -> str:
        return "safety_followup" if state.safety_triggered else "store_memory"

    graph = StateGraph(GraphState)
    graph.add_node("retrieve_memories", retrieve_memories_node)
    graph.add_node("generate_reply", generate_reply_node)
    graph.add_node("safety_followup", safety_followup_node)
    graph.add_node("store_memory", store_memory_node)

    graph.set_entry_point("retrieve_memories")
    if emotion_classifier is not None:
        graph.add_node("detect_emotion", detect_emotion_node)
        graph.add_edge("retrieve_memories", "detect_emotion")
        graph.add_edge("detect_emotion", "generate_reply")
    else:
        graph.add_edge("retrieve_memories", "generate_reply")
    graph.add_conditional_edges(
        "generate_reply",
        route_after_reply,
        {"safety_followup": "safety_followup", "store_memory": "store_memory"},
    )
    graph.add_edge("safety_followup", "store_memory")
    graph.add_edge("store_memory", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
