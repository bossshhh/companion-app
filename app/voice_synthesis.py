"""
Voice synthesis module - converts the companion's text replies into speech
via ElevenLabs.

Built against the raw REST API (not the elevenlabs Python SDK) on purpose:
SDKs for fast-moving products like this one version-drift and occasionally
break (we already hit this with Anthropic's SDK and instructor's exception
paths earlier) - a documented REST contract is more stable to build
against for a 9-day sprint.

API reference: POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import requests

DEFAULT_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel - default pre-made voice
DEFAULT_MODEL_ID = "eleven_flash_v2_5"  # low-latency model, right for live conversation
API_BASE_URL = "https://api.elevenlabs.io/v1"


class VoiceSynthesisError(Exception):
    """Raised when speech synthesis fails."""


class VoiceSynthesizer(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Returns raw audio bytes (MP3) for the given text."""
        ...


class ElevenLabsSynthesizer(VoiceSynthesizer):
    def __init__(
        self,
        api_key: str | None = None,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
    ) -> None:
        api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ELEVENLABS_API_KEY not set. Export it or pass api_key= "
                "explicitly before constructing ElevenLabsSynthesizer."
            )
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = model_id

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            raise ValueError("text must not be empty")

        url = f"{API_BASE_URL}/text-to-speech/{self._voice_id}"
        headers = {"xi-api-key": self._api_key, "Content-Type": "application/json"}
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise VoiceSynthesisError(f"ElevenLabs synthesis failed: {exc}") from exc

        return response.content
