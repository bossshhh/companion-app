"""
Seeds ChromaMemoryStore with generic sample data (seed_data/generic_memories.json).
GENERIC data only - no specific named person, no simulated relative.

Usage:
    python seed_generic_memories.py                # real embeddings (slower, first run downloads model)
    python seed_generic_memories.py --fake-embed    # fast fake embeddings, for pipeline testing only
    python seed_generic_memories.py --reset         # wipe existing data first
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from app.chroma_store import ChromaMemoryStore
from app.embeddings import embed_text, real_embed_text
from app.models import MemoryCategory, MemoryRecord

SEED_FILE = Path(__file__).parent / "seed_data" / "generic_memories.json"
DEFAULT_PERSIST_PATH = "./chroma_data"


def load_seed_entries() -> list[dict]:
    with open(SEED_FILE) as f:
        return json.load(f)


def seed(persist_path: str = DEFAULT_PERSIST_PATH, use_fake_embeddings: bool = False, reset: bool = False) -> None:
    if reset:
        path = Path(persist_path)
        if path.exists():
            print(f"Removing existing data at {persist_path}...")
            shutil.rmtree(path)

    entries = load_seed_entries()
    embed_fn = embed_text if use_fake_embeddings else real_embed_text

    store = ChromaMemoryStore(persist_path=persist_path)
    before_count = store.count()

    print(f"Seeding {len(entries)} generic memories into {persist_path} "
          f"({'fake' if use_fake_embeddings else 'real sentence-transformers'} embeddings)...")

    for i, entry in enumerate(entries, start=1):
        record = MemoryRecord(
            text=entry["text"],
            embedding=embed_fn(entry["text"]),
            importance=entry["importance"],
            category=MemoryCategory(entry["category"]),
            safety_flag=entry["safety_flag"],
        )
        store.add(record)
        if i % 20 == 0 or i == len(entries):
            print(f"  {i}/{len(entries)} seeded...")

    after_count = store.count()
    print(f"\nDone. Store had {before_count} records, now has {after_count} (+{after_count - before_count}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persist-path", default=DEFAULT_PERSIST_PATH)
    parser.add_argument("--fake-embed", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    seed(persist_path=args.persist_path, use_fake_embeddings=args.fake_embed, reset=args.reset)


if __name__ == "__main__":
    main()
