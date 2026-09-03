# Companion App — Orchestration Layer

LangGraph orchestration + prompt engineering for the elderly-companion conversational app.
This is Bosh's part of the 9-day build: prompt engineering + LLM/LangGraph orchestration.

## Architecture

```
retrieve_memories -> generate_reply --(safety?)--> safety_followup -> store_memory -> END
                                     \\--(routine)-----------------> store_memory -> END
```

- **retrieve_memories**: embeds the user's message, pulls the top-k weighted-relevant
  memories from the store (safety-flagged memories are always included, on top of top-k).
- **generate_reply**: ONE LLM call (via `instructor`) returns the reply text AND memory
  scoring metadata (importance, category, safety_flag, memory-worthiness) in the same
  structured response. No second "rate this memory" call.
- **safety_followup**: hook point for a real caregiver-alert integration later; currently logs.
- **store_memory**: writes a new memory record if the turn was memory-worthy and the LLM
  call succeeded (failed/fallback turns are never stored, since their importance score
  isn't trustworthy).

State persists across turns via a LangGraph checkpointer keyed on `thread_id` = `user_id`.

## Contracts for the team

### Vector storage teammate
Implement the `MemoryStore` abstract class in `app/memory_store.py` (methods: `add`,
`all_records`, `touch`). `InMemoryMockStore` is the reference implementation and is what
this codebase runs against until a real backend exists — swap it in `build_graph(store, ...)`
and nothing else changes. The record shape it must persist is `MemoryRecord` in
`app/models.py`. Also implement `embed_text()` in `app/embeddings.py` for real (currently a
deterministic hash-based stub for testing).

### Voice synthesis teammate (ElevenLabs)
Input contract: `result["final_reply_text"]` from a graph invocation — plain text, already
short (1-3 sentences) and written to be spoken aloud per the system prompt in `app/llm_node.py`.

### Mobile UI teammate
Call `build_graph(store, generator).invoke({"user_id": ..., "user_input": ...}, config={"configurable": {"thread_id": user_id}})`
from whatever backend endpoint wraps this (FastAPI recommended). Read `result["final_reply_text"]`
for display/TTS. **Important:** always invoke with a partial dict (`{"user_id":..., "user_input":...}`),
never a full `GraphState(...)` instance — see the comment in `app/main.py` for why.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run tests (no API key required — LLM calls are stubbed)

```bash
python -m pytest tests/ -v
```

## Manual smoke test (requires ANTHROPIC_API_KEY)

```bash
python -m app.main
```

## Known open items / next steps for the sprint

- `app/embeddings.py` uses a deterministic fake embedding for dev/testing. Replace
  `embed_text()` with the real embedding call once the vector-storage teammate's pipeline
  is ready — nothing else in this codebase needs to change.
- Composite score weights (`alpha`, `beta`, `gamma` in `MemoryStore.retrieve`) are a starting
  point (importance weighted above recency — see the comment above `retrieve()` in
  `memory_store.py` for why equal weighting doesn't actually work). Re-tune against real
  handwritten conversation transcripts on Day 4, don't assume these are final.
  test_high_importance_beats_recency` locks in the qualitative behavior; if you change the
  weights, keep that test passing or consciously decide to change the claim it encodes.
- The habit database mentioned in the spec isn't wired up yet — `MemoryCategory.HABIT` exists
  as a tag but there's no separate habit-specific read/write path yet. Decide whether habits
  are just a memory category or need their own store/table.
- `safety_followup_node` currently only logs. If "notify a caregiver" is in scope, that's
  where it plugs in.
