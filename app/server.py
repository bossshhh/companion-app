"""
FastAPI backend wrapping the companion app's LangGraph pipeline.

Run locally:
    uvicorn app.server:app --reload --host 0.0.0.0 --port 8000

Then:
    curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"user_id": "demo-user", "message": "hello"}'
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.chroma_store import ChromaMemoryStore
from app.embeddings import real_embed_text
from app.emotion_tracker import EmotionClassifier, HuggingFaceEmotionClassifier
from app.graph import build_graph
from app.llm_node import StructuredReplyGenerator
from app.memory_store import MemoryStore
from app.voice_synthesis import ElevenLabsSynthesizer, VoiceSynthesisError, VoiceSynthesizer

logger = logging.getLogger("companion_app.server")


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Stable identifier for this user's conversation thread.")
    message: str = Field(..., min_length=1, description="What the user said, as plain text.")


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize into speech - typically /chat's reply_text.")


class ChatResponse(BaseModel):
    reply_text: str = Field(..., description="The assistant's reply - send this to voice synthesis as-is.")
    safety_triggered: bool = Field(..., description="True if this turn was flagged as a safety concern.")
    detected_emotion: str | None = Field(None, description="Emotion Tracker output, if enabled - otherwise null.")
    detected_emotion_score: float | None = Field(None, description="Confidence score for detected_emotion.")
    llm_call_failed: bool = Field(
        ..., description="True if the LLM call degraded to fallback this turn (reply still valid, just simpler)."
    )


def create_app(
    store: MemoryStore,
    generator: StructuredReplyGenerator,
    emotion_classifier: EmotionClassifier | None = None,
    embed_fn=None,
    voice_synthesizer: VoiceSynthesizer | None = None,
) -> FastAPI:
    graph = build_graph(
        store,
        generator,
        emotion_classifier=emotion_classifier,
        **({"embed_fn": embed_fn} if embed_fn is not None else {}),
    )

    fastapi_app = FastAPI(title="Companion App API", version="0.1.0")

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @fastapi_app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        try:
            update = {"user_id": request.user_id, "user_input": request.message}
            config = {"configurable": {"thread_id": request.user_id}}
            result = graph.invoke(update, config=config)
        except Exception as exc:
            logger.exception("Unhandled error processing /chat for user_id=%s", request.user_id)
            raise HTTPException(status_code=500, detail="Something went wrong processing that message.") from exc

        return ChatResponse(
            reply_text=result["final_reply_text"],
            safety_triggered=result["safety_triggered"],
            detected_emotion=result.get("detected_emotion"),
            detected_emotion_score=result.get("detected_emotion_score"),
            llm_call_failed=result["llm_call_failed"],
        )

    @fastapi_app.post("/speak")
    def speak(request: SpeakRequest) -> Response:
        if voice_synthesizer is None:
            raise HTTPException(
                status_code=503,
                detail="Voice synthesis is not configured on this server (no ELEVENLABS_API_KEY set).",
            )
        try:
            audio_bytes = voice_synthesizer.synthesize(request.text)
        except VoiceSynthesisError as exc:
            logger.exception("Voice synthesis failed")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return Response(content=audio_bytes, media_type="audio/mpeg")

    return fastapi_app


def _build_real_app() -> FastAPI:
    store = ChromaMemoryStore()
    generator = StructuredReplyGenerator()

    emotion_classifier: EmotionClassifier | None = None
    if os.environ.get("ENABLE_EMOTION_TRACKER", "").lower() == "true":
        logger.info("Loading Emotion Tracker model (ENABLE_EMOTION_TRACKER=true)...")
        emotion_classifier = HuggingFaceEmotionClassifier()

    voice_synthesizer: VoiceSynthesizer | None = None
    if os.environ.get("ELEVENLABS_API_KEY"):
        logger.info("ELEVENLABS_API_KEY set - enabling /speak endpoint")
        voice_synthesizer = ElevenLabsSynthesizer()

    return create_app(
        store,
        generator,
        emotion_classifier=emotion_classifier,
        embed_fn=real_embed_text,
        voice_synthesizer=voice_synthesizer,
    )


app: FastAPI
try:
    app = _build_real_app()
except Exception as exc:
    logger.error("Could not build the real app at import time: %s", exc)

    app = FastAPI(title="Companion App API (misconfigured)")

    @app.get("/health")
    def health_misconfigured() -> dict:
        return {"status": "misconfigured", "error": str(exc)}
