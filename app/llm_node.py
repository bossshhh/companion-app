"""
The structured-reply LLM node.

Design decisions (from the spec worked out with the team):

1. ONE LLM call per turn, not two. The call returns both the conversational
   reply AND the memory-scoring metadata (memory_worthy, importance,
   category, safety_flag) via `instructor`-enforced structured output
   against `StructuredReply`. This avoids a second "rate this memory's
   importance" call, which would double cost and latency on every turn -
   unacceptable for a live voice interface.

2. Safety detection is TWO-LAYERED, not LLM-only:
   - A cheap keyword/regex pass runs BEFORE the LLM call. If it matches,
     `safety_flag` is forced True regardless of what the LLM returns. This
     is the non-negotiable floor: it cannot fail silently because the LLM
     had an off turn.
   - The LLM's own `safety_flag` judgment (from context, not just keywords)
     is OR'd in on top of the keyword layer, so context-only safety signals
     ("I couldn't remember how to get home today") are still caught even
     without a matching keyword.

3. Failure handling is retry-then-degrade, never crash-to-user:
   - `instructor` validates the LLM's JSON against `StructuredReply` and
     retries automatically (see `max_retries` below) with an error message
     appended so the model can self-correct.
   - If it still fails after retries, we fall back to a plain (unstructured)
     completion call for just the reply text, skip storing a memory for
     that turn, and log the failure. The user never sees an error.

4. PROVIDER-AGNOSTIC by design: `provider="anthropic"` or `provider="groq"`.
   This exists because Anthropic API access requires paid credit, while
   Groq has a genuine no-credit-card free tier - useful for development/
   testing during the sprint before billing is sorted, or as a fallback if
   Anthropic credit runs out mid-demo. Swapping providers is a constructor
   argument or an env var, not a code change - see `LLM_PROVIDER` below.
   Note: prompt behavior (especially safety_flag reliability) has only been
   manually spot-checked on Groq's model, not rigorously validated the way
   Anthropic was during Day 1-3 build - treat Groq as good enough for
   plumbing/integration testing, and re-verify carefully before relying on
   it for anything closer to the actual demo.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

import anthropic
import instructor
from pydantic import ValidationError

try:
    from instructor.core.exceptions import InstructorRetryException
except ImportError:
    try:
        from instructor.v2.core.errors import InstructorRetryException
    except ImportError:
        from instructor.exceptions import InstructorRetryException

try:
    import groq
except ImportError:
    groq = None

from app.models import StructuredReply

logger = logging.getLogger("companion_app.llm_node")

Provider = Literal["anthropic", "groq"]

DEFAULT_MODELS: dict[Provider, str] = {
    "anthropic": "claude-sonnet-4-6",
    "groq": "openai/gpt-oss-120b",
}
API_KEY_ENV_VARS: dict[Provider, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
}
MAX_STRUCTURED_RETRIES = 2

_EXPECTED_API_ERRORS: tuple[type[Exception], ...] = (
    ValidationError,
    anthropic.APIError,
    InstructorRetryException,
) + ((groq.APIError,) if groq is not None else ())

SAFETY_KEYWORDS = [
    r"\bfell\b", r"\bfall\b", r"\bfalling\b", r"\bfainted?\b",
    r"\bdizzy\b", r"\bdizziness\b",
    r"\bpain\b", r"\bhurts?\b", r"\baching?\b",
    r"\bchest\b", r"\bbreath(e|ing)?\b", r"\bcan'?t breathe\b",
    r"\bconfused\b", r"\bforgot where\b", r"\blost\b",
    r"\bmedication\b", r"\bpills?\b", r"\bmissed my dose\b",
    r"\bbleeding\b", r"\bnumb\b", r"\bemergency\b",
    r"\bcan'?t (get up|move|stand)\b",
]
_SAFETY_PATTERN = re.compile("|".join(SAFETY_KEYWORDS), re.IGNORECASE)


def keyword_safety_check(text: str) -> bool:
    return bool(_SAFETY_PATTERN.search(text))


SYSTEM_PROMPT = """\
You are a warm, patient conversational companion for an elderly user. \
Your job is twofold every turn:

1. Reply naturally and warmly to what the user just said. Keep replies \
short (1-3 sentences) and easy to follow when spoken aloud - avoid complex \
sentence structure, avoid lists, avoid jargon. Use the retrieved memories \
below (if any) to make the conversation feel continuous and personal, but \
never recite them mechanically - weave them in the way a person who \
actually remembers would.

2. Assess the turn for memory-worthiness and safety, and report it in the \
structured fields:
   - memory_worthy: true if this turn contains a fact, event, feeling, or \
     habit worth remembering for future conversations. Small talk with no \
     lasting content ("nice weather today") is usually NOT memory-worthy.
   - memory_summary: if memory_worthy, a short first-person summary of \
     what to remember (e.g. "Misses her late husband, especially in the \
     evenings" or "Takes blood pressure medication at 8am").
   - importance: 1-10. Anchor your scale like this: casual preferences \
     ("likes tea") are 2-3. Habits and routines are 4-5. Emotionally \
     significant statements (grief, loneliness, strong affection) are \
     7-8. Anything touching physical safety, health changes, or confusion \
     is 9-10 REGARDLESS of how calmly it was stated - a matter-of-fact \
     "I felt dizzy this morning" is a 9, not a 3, even though it carries \
     little emotional charge.
   - category: pick the best fit - habit, emotional, safety, small_talk, \
     or other.
   - safety_flag: true if the user's message indicates ANY physical or \
     cognitive safety concern - falls, pain, dizziness, breathing \
     trouble, confusion, medication issues, getting lost, or similar. \
     When in doubt, set this true; false positives are far cheaper than \
     false negatives here.

Retrieved memories for context (may be empty):
{memory_context}
"""


@dataclass
class LLMResult:
    structured: StructuredReply | None
    succeeded: bool
    used_fallback: bool


class StructuredReplyGenerator:
    def __init__(
        self,
        provider: Provider | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        provider = provider or os.environ.get("LLM_PROVIDER", "anthropic")
        if provider not in ("anthropic", "groq"):
            raise ValueError(f"Unknown provider {provider!r}. Must be 'anthropic' or 'groq'.")

        env_var = API_KEY_ENV_VARS[provider]
        api_key = api_key or os.environ.get(env_var)
        if not api_key:
            raise RuntimeError(
                f"{env_var} not set for provider={provider!r}. Export it or pass "
                f"api_key= explicitly before constructing StructuredReplyGenerator."
            )

        self._provider = provider
        self._model = model or DEFAULT_MODELS[provider]

        if provider == "anthropic":
            self._raw_client = anthropic.Anthropic(api_key=api_key)
            self._client = instructor.from_anthropic(self._raw_client)
        else:
            if groq is None:
                raise RuntimeError(
                    "provider='groq' requires the `groq` package. Install it with "
                    "`pip install groq` (already in requirements.txt)."
                )
            self._raw_client = groq.Groq(api_key=api_key)
            self._client = instructor.from_groq(self._raw_client)

        logger.info("StructuredReplyGenerator using provider=%s model=%s", self._provider, self._model)

    def generate(self, user_input: str, memory_context: str) -> LLMResult:
        keyword_flag = keyword_safety_check(user_input)

        try:
            structured: StructuredReply = self._client.chat.completions.create(
                model=self._model,
                max_tokens=1024,
                max_retries=MAX_STRUCTURED_RETRIES,
                response_model=StructuredReply,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(memory_context=memory_context or "(none)")},
                    {"role": "user", "content": user_input},
                ],
            )
            if keyword_flag and not structured.safety_flag:
                structured.safety_flag = True
            if structured.memory_worthy and not structured.memory_summary:
                structured.memory_summary = user_input[:200]

            return LLMResult(structured=structured, succeeded=True, used_fallback=False)

        except _EXPECTED_API_ERRORS as exc:
            logger.warning("Structured reply generation failed after retries: %s", exc)
            return self._fallback(user_input, keyword_flag)
        except Exception as exc:
            logger.error("Unexpected error in structured reply generation: %s", exc)
            return self._fallback(user_input, keyword_flag)

    def _fallback(self, user_input: str, keyword_flag: bool) -> LLMResult:
        fallback_system = (
            "You are a warm, patient companion for an elderly user. "
            "Reply naturally in 1-3 short, simple spoken-friendly sentences."
        )
        try:
            if self._provider == "anthropic":
                resp = self._raw_client.messages.create(
                    model=self._model,
                    max_tokens=300,
                    system=fallback_system,
                    messages=[{"role": "user", "content": user_input}],
                )
                reply_text = "".join(block.text for block in resp.content if hasattr(block, "text")).strip()
            else:
                resp = self._raw_client.chat.completions.create(
                    model=self._model,
                    max_tokens=300,
                    messages=[
                        {"role": "system", "content": fallback_system},
                        {"role": "user", "content": user_input},
                    ],
                )
                reply_text = (resp.choices[0].message.content or "").strip()

            if not reply_text:
                reply_text = "I'm here with you - could you say that again for me?"
        except Exception as exc:
            logger.error("Fallback plain-text call also failed: %s", exc)
            reply_text = "I'm here with you - could you say that again for me?"

        structured = StructuredReply(
            reply_text=reply_text,
            memory_worthy=False,
            memory_summary=None,
            importance=5,
            category="other",
            safety_flag=keyword_flag,
        )
        return LLMResult(structured=structured, succeeded=False, used_fallback=True)
