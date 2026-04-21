"""
Runs sentiment analysis on Steam review texts using
distilbert-base-uncased-finetuned-sst-2-english and writes the results
to output/reviews_with_sentiment.csv.

Label mapping:
  POSITIVE  → confidence >= threshold
  NEGATIVE  → confidence >= threshold
  NEUTRAL   → model confidence < threshold  (low-signal / very short reviews)

Reviews longer than the model's 512-token limit are truncated to the first
512 tokens by the tokenizer (truncation=True), preserving as much signal
as possible without splitting into multiple chunks.
"""

import csv
import logging
import os
import time
from pathlib import Path

# Anaconda bundles its own OpenMP runtime; PyTorch ships another copy.
# This env var prevents a fatal crash on Windows when both are loaded.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from transformers import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_CSV  = Path("output/reviews.csv")
OUTPUT_CSV = Path("output/reviews_with_sentiment.csv")

MODEL_NAME          = "distilbert-base-uncased-finetuned-sst-2-english"
BATCH_SIZE          = 64    # increase if you have a GPU with more VRAM
NEUTRAL_THRESHOLD   = 0.65  # confidence below this → NEUTRAL
EMPTY_LABEL         = "NEUTRAL"
EMPTY_CONFIDENCE    = 0.0
LOG_EVERY           = 250   # log progress every N reviews

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label(raw_label: str, confidence: float) -> str:
    if confidence < NEUTRAL_THRESHOLD:
        return "NEUTRAL"
    return raw_label.upper()


def _load_pipeline() -> pipeline:
    device = 0 if torch.cuda.is_available() else -1
    device_name = f"GPU (cuda:{device})" if device >= 0 else "CPU"
    log.info("Loading model '%s' on %s", MODEL_NAME, device_name)
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        device=device,
        truncation=True,
        max_length=512,
    )


def _run_batch(nlp, texts: list[str]) -> list[tuple[str, float]]:
    """Returns list of (label, confidence) for each text in the batch."""
    results = nlp(texts, batch_size=BATCH_SIZE, truncation=True)
    return [(_label(r["label"], r["score"]), round(r["score"], 4)) for r in results]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_CSV}. Run fetch_reviews.py first.")

    with INPUT_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    log.info("Loaded %d reviews from %s", len(rows), INPUT_CSV)

    nlp = _load_pipeline()

    output_fields = list(rows[0].keys()) + ["sentiment_label", "sentiment_confidence"]

    texts       = [r["review_text"] for r in rows]
    sentiments  = [""] * len(rows)
    confidences = [0.0] * len(rows)

    t0      = time.time()
    total   = len(texts)
    batches = range(0, total, BATCH_SIZE)

    for batch_start in batches:
        batch_texts = texts[batch_start : batch_start + BATCH_SIZE]

        # Separate empty/whitespace-only reviews — model errors on blank input
        indices        = list(range(batch_start, batch_start + len(batch_texts)))
        non_empty_idx  = [i for i, t in zip(indices, batch_texts) if t.strip()]
        non_empty_text = [texts[i] for i in non_empty_idx]

        if non_empty_text:
            results = _run_batch(nlp, non_empty_text)
            for i, (label, conf) in zip(non_empty_idx, results):
                sentiments[i]  = label
                confidences[i] = conf

        # empty reviews stay at defaults: NEUTRAL / 0.0
        for i in indices:
            if not texts[i].strip():
                sentiments[i]  = EMPTY_LABEL
                confidences[i] = EMPTY_CONFIDENCE

        done = min(batch_start + BATCH_SIZE, total)
        if done % LOG_EVERY < BATCH_SIZE or done == total:
            elapsed = time.time() - t0
            rate    = done / elapsed if elapsed else 0
            log.info("Progress: %d / %d  (%.1f reviews/s)", done, total, rate)

    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row, label, conf in zip(rows, sentiments, confidences):
            writer.writerow({**row, "sentiment_label": label, "sentiment_confidence": conf})

    elapsed = time.time() - t0
    pos  = sentiments.count("POSITIVE")
    neg  = sentiments.count("NEGATIVE")
    neu  = sentiments.count("NEUTRAL")
    log.info("Done in %.1fs — POSITIVE: %d  NEGATIVE: %d  NEUTRAL: %d", elapsed, pos, neg, neu)
    log.info("Output saved → %s", OUTPUT_CSV)


if __name__ == "__main__":
    main()
