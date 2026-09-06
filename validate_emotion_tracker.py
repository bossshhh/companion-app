"""
Validates the Emotion Tracker (app/emotion_tracker.py) against GoEmotions -
the dataset your proposal (Section 3.4) actually names for this purpose.

This does NOT validate against the elder-care Kaggle dataset's own
emotion_label column - see the note in emotion_tracker.py for why that
would be circular.

Usage:
    pip install -r requirements-dev.txt
    python validate_emotion_tracker.py --n 500

Requires internet access to huggingface.co to download GoEmotions on
first run (cached afterward).
"""

from __future__ import annotations

import argparse
from collections import Counter

from app.emotion_tracker import HuggingFaceEmotionClassifier

GOEMOTIONS_TO_EKMAN: dict[str, str | None] = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": None,
    "caring": None,
    "confusion": None,
    "curiosity": None,
    "desire": None,
    "disappointment": "sadness",
    "disapproval": "disgust",
    "disgust": "disgust",
    "embarrassment": "fear",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": None,
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}


def load_goemotions_sample(n: int) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    label_names = ds.features["labels"].feature.names

    pairs: list[tuple[str, str]] = []
    for row in ds:
        if len(row["labels"]) != 1:
            continue
        raw_label = label_names[row["labels"][0]]
        collapsed = GOEMOTIONS_TO_EKMAN.get(raw_label)
        if collapsed is None:
            continue
        pairs.append((row["text"], collapsed))
        if len(pairs) >= n:
            break
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading up to {args.n} single-label GoEmotions test examples...")
    pairs = load_goemotions_sample(args.n)
    print(f"Loaded {len(pairs)} examples after filtering.\n")

    print("Loading emotion classifier (downloads model weights on first run)...")
    classifier = HuggingFaceEmotionClassifier(model_name="j-hartmann/emotion-english-distilroberta-base")

    correct = 0
    confusion: Counter = Counter()
    per_class_total: Counter = Counter()
    per_class_correct: Counter = Counter()

    for text, true_label in pairs:
        result = classifier.classify(text)
        predicted = result.label
        per_class_total[true_label] += 1
        confusion[(true_label, predicted)] += 1
        if predicted == true_label:
            correct += 1
            per_class_correct[true_label] += 1

    accuracy = correct / len(pairs) if pairs else 0.0

    print("=" * 60)
    print(f"Overall accuracy: {accuracy:.3f}  ({correct}/{len(pairs)})")
    print("=" * 60)
    print("\nPer-class accuracy:")
    for label in sorted(per_class_total):
        total = per_class_total[label]
        acc = per_class_correct[label] / total if total else 0.0
        print(f"  {label:10s}  {acc:.3f}  ({per_class_correct[label]}/{total})")

    print("\nConfusion matrix (true -> predicted : count), errors only:")
    for (true_label, predicted), count in sorted(confusion.items(), key=lambda kv: -kv[1]):
        if true_label != predicted:
            print(f"  {true_label:10s} -> {predicted:10s} : {count}")

    print(
        "\nNote: measures general accuracy on GoEmotions (Reddit comments) - "
        "does not confirm performance on elderly speech patterns or "
        "grief-specific language."
    )


if __name__ == "__main__":
    main()
