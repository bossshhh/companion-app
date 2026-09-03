"""
Manual CLI for exercising the full graph with a real LLM call.

Requires ANTHROPIC_API_KEY to be set in the environment. Run:

    python -m app.main

Type messages; Ctrl+C to quit. This is for manual smoke-testing during the
sprint, not the mobile app's actual entry point - the mobile UI teammate
will call `build_graph(...).invoke(...)` from wherever the app's backend
lives (a FastAPI endpoint, most likely).
"""

from __future__ import annotations

import logging
import sys
import uuid

from app.graph import build_graph
from app.llm_node import StructuredReplyGenerator
from app.memory_store import InMemoryMockStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    store = InMemoryMockStore()
    generator = StructuredReplyGenerator()
    app = build_graph(store, generator)

    user_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": user_id}}

    print("Companion app CLI. Ctrl+C to quit.\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            # IMPORTANT: invoke with a partial dict, not a full GraphState
            # instance. GraphState's unset fields (e.g. conversation_history)
            # default to [] - passing the whole model would overwrite the
            # checkpointed history with that default on every turn, silently
            # wiping conversation memory. A partial dict only updates the
            # keys it names; the checkpointer fills in everything else from
            # the saved state for this thread_id.
            update = {"user_id": user_id, "user_input": user_input}
            result = app.invoke(update, config=config)
            print(f"Assistant: {result['final_reply_text']}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
